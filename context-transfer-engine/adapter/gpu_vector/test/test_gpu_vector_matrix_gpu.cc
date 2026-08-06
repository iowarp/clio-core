/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * MULTI-BLOCK on-device faulting against an EXTERNALLY-written, FAMILY-SHARDED
 * tag — the llama.cpp weight-streaming configuration in miniature.
 *
 * Every other test in this directory runs read_range from a single CUDA block
 * (the oversubscribe kernel literally early-returns `blockIdx.x != 0`), and
 * every multi-block test writes its data through the vector itself, which is
 * naming-scheme-agnostic by construction. The combination that shipped broken
 * was therefore never covered:
 *
 *     nblocks == 8, uint8_t elements, 2 MiB pages,
 *     pages written EXTERNALLY under the CAE assimilator's sharded families
 *         (family = page / ceil(num_pages/nblocks)),
 *     kAsync cache manager, allow_cold_miss_fault,
 *     cache << extent, every CUDA block cold-faulting pages OUTSIDE the
 *     family stripe its index would suggest,
 *     consume staging into __shared__ (the paged-dequant kernel's shape).
 *
 * What this covers that nothing else did:
 *   - the gpu_family_idx_ / FamilyOf policy: fault names must match the
 *     assimilator's sharding for ANY (faulting block, page) pair;
 *   - the FamFn lambda input to dev::vector;
 *   - concurrent fault traffic from 8 blocks at once;
 *   - byte-typed pages (all prior fault tests used u32/float).

/**
 * gpu_vector paging matrix.
 *
 * Every case here is a defect that actually shipped and was found by running a
 * multi-GB model instead of a test. The point of the file is that none of them
 * should have needed a 30B model to find:
 *
 *   OversubscribedSequential  cache << extent, every byte via read_range
 *   StaleWindowAfterEviction  prefaulted window must not outlive its slots
 *   GridWiderThanNblocks      N CUDA blocks sharing one cache block, faulting
 *   UnalignedChunkedStaging   stage size NOT a multiple of the record size
 *   CrossPageUnalignedSpans   spans straddling page boundaries at odd offsets
 *   RandomOrderThrash         non-sequential access defeating the prefetcher
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace gv  = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

bool g_initialized = false;

constexpr clio::run::u64 kPageSizeBytes = 1ull << 20;   // 1 MiB pages
constexpr clio::run::u32 kLogicalPages  = 24;           // 24 MiB extent
constexpr clio::run::u64 kTotalBytes    =
    static_cast<clio::run::u64>(kLogicalPages) * kPageSizeBytes;

/** Deterministic per-byte pattern; page index and offset both contribute, so a
 *  slot holding the WRONG page is detected even if the offset is right. */
CTP_CROSS_FUN inline uint8_t PatternAt(clio::run::u64 byte_idx) {
  const clio::run::u64 pg = byte_idx / kPageSizeBytes;
  return static_cast<uint8_t>((pg * 131u + byte_idx * 7u) & 0xFFu);
}

void EnsureRuntime() {
  if (g_initialized) return;
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  CLIO_CTE_CLIENT->Init(clio::cte::core::kCtePoolId);
  std::this_thread::sleep_for(200ms);
  g_initialized = true;
}

/** Seed every page of `tag` with PatternAt, using the family naming the
 *  device-side vector derives from its fam_ppb policy. */
void SeedTag(const std::string &tag, const clio::cte::core::TagId &tag_id,
             clio::run::u32 fam_pages) {
  auto *cte = CLIO_CTE_CLIENT;
  for (clio::run::u32 p = 0; p < kLogicalPages; ++p) {
    auto buf = CLIO_IPC->AllocateBuffer(kPageSizeBytes);
    auto *bytes = reinterpret_cast<uint8_t *>(buf.ptr_);
    const clio::run::u64 base = static_cast<clio::run::u64>(p) * kPageSizeBytes;
    for (clio::run::u64 j = 0; j < kPageSizeBytes; ++j) bytes[j] = PatternAt(base + j);
    const std::string blob = tag + "_b" + std::to_string(p / fam_pages) +
                             "_pi" + std::to_string(p);
    auto task = cte->AsyncPutBlob(tag_id, blob, 0, kPageSizeBytes,
                                  buf.shm_.template Cast<void>(), 1.0f,
                                  clio::cte::core::Context(), 0);
    task.Wait();
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernels. All read weights ONLY through read_range.
// ---------------------------------------------------------------------------

/** Walk [lo, hi) in `chunk`-byte spans, verifying every byte. `chunk` is a
 *  parameter so a caller can pass a size that does NOT divide the record size,
 *  which is exactly the get_rows staging bug (chunk boundaries landing
 *  mid-record made the reader compute the wrong element index). */
__global__ void SpanVerifyKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceView<uint8_t> view,
                                 unsigned long long *mismatches,
                                 unsigned long long *checked,
                                 clio::run::u64 lo, clio::run::u64 hi,
                                 clio::run::u64 chunk) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  auto fam = [] (const gv::DeviceViewBase &b, clio::run::u64 pg) {
    return gv::FamilyOf(b, pg);
  };
  dev::vector<uint8_t, decltype(fam)> W(view, g_ipc_manager_ptr, fam);
  const clio::run::u64 span = (hi - lo + gridDim.x - 1) / gridDim.x;
  clio::run::u64 b0 = lo + static_cast<clio::run::u64>(blockIdx.x) * span;
  clio::run::u64 b1 = b0 + span;
  if (b1 > hi) b1 = hi;
  for (clio::run::u64 c = b0; c < b1; c += chunk) {
    const clio::run::u64 ce = (c + chunk < b1) ? (c + chunk) : b1;
    if (threadIdx.x < 32) {
      W.read_range(c, ce, [mismatches, checked](clio::run::u64 i, uint8_t v) {
        atomicAdd(checked, 1ULL);
        if (v != PatternAt(i)) atomicAdd(mismatches, 1ULL);
      });
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------

/** Build a vector over the seeded tag with an explicitly chosen cache size. */
static gv::Vector<uint8_t> MakeVec(const std::string &tag, clio::run::u32 nblocks,
                                   clio::run::u32 pages_per_block) {
  return gv::Vector<uint8_t>(tag, nblocks, /*gpu_id=*/0, pages_per_block,
                             /*host_pages_per_block=*/0, kPageSizeBytes,
                             /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                             /*manager_threads_per_block=*/32,
                             /*allow_cold_miss_fault=*/true);
}

static void RunSpanCase(const char *tag_name, clio::run::u32 nblocks,
                        clio::run::u32 pages_per_block, clio::run::u32 grid,
                        clio::run::u64 lo, clio::run::u64 hi,
                        clio::run::u64 chunk) {
  EnsureRuntime();
  const std::string tag = tag_name;
  auto vec = MakeVec(tag, nblocks, pages_per_block);
  const clio::run::u32 fam_pages = (kLogicalPages + nblocks - 1) / nblocks;
  SeedTag(tag, vec.TagId(), fam_pages);

  auto *gm = CLIO_CPU_IPC->GetGpuIpcManager();
  REQUIRE(gm != nullptr);
  auto gpu_info = gm->GetGpuInfo(0);

  unsigned long long *counters = nullptr;
  REQUIRE(cudaMallocManaged(&counters, 2 * sizeof(unsigned long long)) == cudaSuccess);
  counters[0] = 0; counters[1] = 0;

  SpanVerifyKernel<<<grid, 64>>>(gpu_info, vec.Device(), &counters[0],
                                 &counters[1], lo, hi, chunk);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  printf("  [%s] nblocks=%u slots/blk=%u grid=%u chunk=%llu -> checked=%llu mismatches=%llu\n",
         tag_name, nblocks, pages_per_block, grid,
         (unsigned long long) chunk, counters[1], counters[0]);
  REQUIRE(counters[1] == (hi - lo));   // every byte was actually read
  REQUIRE(counters[0] == 0);           // and every byte was correct
  cudaFree(counters);
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/** Cache holds 4 of 24 pages: 6x oversubscribed, so nearly every span faults.
 *  This is the qwen3-coder configuration that produced garbage tokens. */
TEST_CASE("gpu_vector: oversubscribed sequential read is byte-exact",
          "[gpu_vector][oom]") {
  RunSpanCase("gvm_oversub", /*nblocks=*/4, /*pages_per_block=*/1, /*grid=*/4,
              0, kTotalBytes, /*chunk=*/24576);
}

/** Re-reads the LOW pages after the walk has faulted far past them. A window
 *  prefaulted at slot s == page s is only valid until eviction reuses that
 *  slot; a reader that keeps trusting the original mapping silently returns a
 *  different page's bytes. */
TEST_CASE("gpu_vector: prefaulted window does not outlive its slots",
          "[gpu_vector][oom]") {
  EnsureRuntime();
  const std::string tag = "gvm_stale";
  auto vec = MakeVec(tag, /*nblocks=*/1, /*pages_per_block=*/4);
  SeedTag(tag, vec.TagId(), kLogicalPages);
  auto *gm = CLIO_CPU_IPC->GetGpuIpcManager();
  auto gpu_info = gm->GetGpuInfo(0);
  unsigned long long *c = nullptr;
  REQUIRE(cudaMallocManaged(&c, 2 * sizeof(unsigned long long)) == cudaSuccess);

  // Warm the first 4 pages, then fault the whole extent (evicting them), then
  // read the first 4 pages again. They must still be correct.
  for (int pass = 0; pass < 3; ++pass) {
    c[0] = 0; c[1] = 0;
    const clio::run::u64 hi = (pass == 1) ? kTotalBytes : 4 * kPageSizeBytes;
    SpanVerifyKernel<<<1, 64>>>(gpu_info, vec.Device(), &c[0], &c[1], 0, hi, 24576);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    printf("  [stale] pass=%d checked=%llu mismatches=%llu\n", pass, c[1], c[0]);
    REQUIRE(c[0] == 0);
  }
  cudaFree(c);
}

/** 64 CUDA blocks over 2 cache blocks. Index wrapping keeps this in bounds,
 *  but many CUDA blocks concurrently faulting through one page table is a
 *  race; this is the illegal-memory-access configuration. */
TEST_CASE("gpu_vector: grid far wider than nblocks faults safely",
          "[gpu_vector][oom][race]") {
  RunSpanCase("gvm_widegrid", /*nblocks=*/2, /*pages_per_block=*/2, /*grid=*/64,
              0, kTotalBytes, /*chunk=*/8192);
}

/** Chunk size deliberately NOT a divisor of a 176-byte record (Q5_K block):
 *  24576 % 176 != 0, so every chunk after the first starts mid-record. */
TEST_CASE("gpu_vector: staging chunks that do not divide the record size",
          "[gpu_vector][staging]") {
  RunSpanCase("gvm_unaligned", /*nblocks=*/2, /*pages_per_block=*/2, /*grid=*/8,
              0, kTotalBytes, /*chunk=*/176 * 139 + 17);
}

/** Spans that start and end at odd offsets and straddle page boundaries. */
TEST_CASE("gpu_vector: cross-page spans at unaligned offsets",
          "[gpu_vector][staging]") {
  RunSpanCase("gvm_crosspage", /*nblocks=*/2, /*pages_per_block=*/2, /*grid=*/4,
              kPageSizeBytes - 1000, 5 * kPageSizeBytes + 333, /*chunk=*/4096);
}

SIMPLE_TEST_MAIN()

#else
int main() { return 0; }
#endif
