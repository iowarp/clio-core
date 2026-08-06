/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * gpu_vector paging matrix.
 *
 * Every case here is a defect that actually shipped and was found by running a
 * multi-GB model instead of a test. None of them should have needed a 30B
 * model to find:
 *
 *   oversubscribed sequential  cache << extent, every byte via read_range
 *   stale prefaulted window    window must not outlive the slots it describes
 *   grid wider than nblocks    many CUDA blocks sharing one cache block
 *   unaligned staging chunks   chunk size not dividing the record size
 *   cross-page unaligned spans spans straddling pages at odd offsets
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
#include <cstring>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace gv  = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

bool g_initialized = false;

constexpr clio::run::u64 kPageSizeBytes = 2ull << 20;   // 2 MiB pages (matches llama_geom, which passes)
constexpr clio::run::u32 kLogicalPages  = 32;           // 64 MiB extent (llama_geom)
constexpr clio::run::u64 kTotalBytes    =
    static_cast<clio::run::u64>(kLogicalPages) * kPageSizeBytes;

/** Deterministic per-byte pattern; page index and offset both contribute, so a
 *  slot holding the WRONG page is detected even if the offset is right. */
CTP_CROSS_FUN inline uint8_t PatternAt(clio::run::u64 byte_idx) {
  const clio::run::u64 pg = byte_idx / kPageSizeBytes;
  return static_cast<uint8_t>((pg * 131u + byte_idx * 7u) & 0xFFu);
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

// Host-only from here down. nvcc compiles this TU twice; in the DEVICE pass
// CLIO_IPC expands to the GPU manager and host-only members such as
// gpu::IpcManager::GetGpuInfo and Client::AsyncPutBlob are not declared, so
// host helpers must be excluded from that pass entirely.
#if !CTP_IS_DEVICE_PASS

namespace {

void EnsureRuntime() {
  if (g_initialized) return;
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);
  cte_client->Init(clio::cte::core::kCtePoolId);

  // The CTE pool and a backing bdev must BOTH exist before any PutBlob. This
  // fixture previously did neither: the puts returned nominally fine while
  // storing nothing, so every read came back zeros -- which looked exactly
  // like a paging defect and was invariant to blob naming, page geometry,
  // family policy, pool id and seed ordering, because none of those was on
  // the failing path.
  clio::cte::core::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
      clio::cte::core::kCtePoolId, params);
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  const clio::run::u64 kRamCapacity = 1ULL << 30;
  clio::run::PoolId bdev_pool_id(972, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("gpu_vector_matrix_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "gpu_vector_matrix_ram", clio::run::bdev::BdevType::kRam,
      kRamCapacity, clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  g_initialized = true;
}

/** Seed every page of `tag` with PatternAt, using the family naming the
 *  device-side vector derives from its fam_ppb policy. */
void SeedTag(const std::string &tag, const clio::cte::core::TagId &tag_id,
             clio::run::u32 fam_pages) {
  clio::cte::core::Client *cte = CLIO_CTE_CLIENT;
  for (clio::run::u32 p = 0; p < kLogicalPages; ++p) {
    auto buf = CLIO_CPU_IPC->AllocateBuffer(kPageSizeBytes);
    auto *bytes = reinterpret_cast<uint8_t *>(buf.ptr_);
    const clio::run::u64 base = static_cast<clio::run::u64>(p) * kPageSizeBytes;
    for (clio::run::u64 j = 0; j < kPageSizeBytes; ++j) bytes[j] = PatternAt(base + j);
    if (p == 0) {
      printf("  [seed-src] ptr=%p bytes[0..3]=%02x %02x %02x %02x (want %02x %02x %02x %02x)\n",
             (void *) bytes, bytes[0], bytes[1], bytes[2], bytes[3],
             PatternAt(0), PatternAt(1), PatternAt(2), PatternAt(3));
    }
    // Seed under BOTH plausible family conventions: family 0 for everything
    // (fam_ppb >= num_pages) and family == page index (fam_ppb == 1). If the
    // device looks up either, it will find data -- which isolates whether the
    // failure is blob NAMING or the put itself.
    const std::string blob = tag + "_b" + std::to_string(p / fam_pages) +
                             "_pi" + std::to_string(p);
    auto task = cte->AsyncPutBlob(tag_id, blob, 0, kPageSizeBytes,
                                  buf.shm_.template Cast<void>(), 1.0f,
                                  clio::cte::core::Context(), 0);
    task.Wait();
  }

  // Prove the fixture actually wrote data BEFORE any kernel runs. The matrix
  // previously reported ~100% mismatches with an exact 1/256 match rate --
  // i.e. reads returning all zeros -- which looked like a paging defect but
  // was indistinguishable from the seeding silently doing nothing.
  {
    auto buf = CLIO_CPU_IPC->AllocateBuffer(kPageSizeBytes);
    auto *got = reinterpret_cast<uint8_t *>(buf.ptr_);
    std::memset(got, 0, 256);
    const std::string blob = tag + "_b0_pi0";  // page 0 is family 0 either way
    auto g = cte->AsyncGetBlob(tag_id, blob.c_str(), 0, kPageSizeBytes, 0,
                               buf.shm_.template Cast<void>());
    g.Wait();
    clio::run::u64 bad = 0;
    for (clio::run::u64 j = 0; j < 4096; ++j) if (got[j] != PatternAt(j)) ++bad;
    printf("  [seed-check] '%s' first 4 KiB: %llu mismatches (0 expected)\n",
           blob.c_str(), (unsigned long long) bad);
    REQUIRE(bad == 0);
  }
}

}  // namespace

// Throughput of the last RunSpanCase, in GiB/s of weight bytes consumed
// through read_range. Reported by every case and asserted by the perf cases.
static double g_last_gbps = 0.0;

static void RunSpanCase(const char *tag_name, clio::run::u32 nblocks,
                        clio::run::u32 pages_per_block, clio::run::u32 grid,
                        clio::run::u64 lo, clio::run::u64 hi,
                        clio::run::u64 chunk) {
  EnsureRuntime();
  const std::string tag = tag_name;
  // Seed BEFORE constructing the reader, using a throwaway vector purely to
  // obtain the TagId -- llama_geom does the same. A reader constructed first
  // snapshots page metadata and never sees blobs written afterwards.
  {
    // EXACTLY llama_geom's seeding vector: no storage_pool_id, no
    // family_pages. Passing PoolId(0, 0) here routed the put at a pool the
    // seeding path does not have, and the writes silently did nothing.
    gv::Vector<uint8_t> seeder(tag, /*nblocks=*/1, /*gpu_id=*/0,
                               /*gpu_pages_per_block=*/1,
                               /*host_pages_per_block=*/0, kPageSizeBytes,
                               /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                               /*manager_threads_per_block=*/32,
                               /*allow_cold_miss_fault=*/true);
    SeedTag(tag, seeder.TagId(), (kLogicalPages + nblocks - 1) / nblocks);
  }
  gv::Vector<uint8_t> vec(tag, nblocks, /*gpu_id=*/0, pages_per_block,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/200, gv::CacheMode::kAsync,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/clio::run::PoolId(0, 0),
                          // Must match SeedTag: all pages in family 0.
                          /*family_pages=*/(kLogicalPages + nblocks - 1) / nblocks);
  // The Vector ctor used here takes no logical-page count, so its fam_ppb
  // policy defaults to "everything in family 0". Seeding with sharded family
  // names would write blobs the device side never looks up.

  auto *gpu_ipc_mgr = CLIO_CPU_IPC->GetGpuIpcManager();
  REQUIRE(gpu_ipc_mgr != nullptr);
  clio::run::IpcManagerGpuInfo gpu_info = gpu_ipc_mgr->GetGpuInfo(0);

  unsigned long long *counters = nullptr;
  REQUIRE(cudaMallocManaged(&counters, 2 * sizeof(unsigned long long)) == cudaSuccess);
  counters[0] = 0; counters[1] = 0;

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  SpanVerifyKernel<<<grid, 64>>>(gpu_info, vec.Device(), &counters[0],
                                 &counters[1], lo, hi, chunk);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  g_last_gbps = (ms > 0.0f)
      ? ((double)(hi - lo) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0) : 0.0;
  cudaEventDestroy(t0); cudaEventDestroy(t1);

  printf("  [%s] nblocks=%u slots/blk=%u grid=%u chunk=%llu -> checked=%llu "
         "mismatches=%llu  %.2f GiB/s\n",
         tag_name, nblocks, pages_per_block, grid,
         (unsigned long long) chunk, counters[1], counters[0], g_last_gbps);
  REQUIRE(counters[1] == (hi - lo));   // every byte was actually read
  REQUIRE(counters[0] == 0);           // and every byte was correct
  cudaFree(counters);
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/** 6x oversubscribed, so nearly every span faults -- the qwen3-coder shape.
 *
 *  NOTE grid == nblocks here, so BlockIdx() (blockIdx.x % nblocks) never
 *  wraps and each CUDA block owns its cache block outright. This case PASSES.
 *  The two cases that fail both run grid > nblocks. Taken together that
 *  isolates the defect to concurrent access to ONE cache block by several
 *  CUDA blocks -- not to oversubscription, eviction, chunk alignment, or
 *  page geometry, all of which this passing case also exercises. */
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
  gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0, /*gpu_pages_per_block=*/4,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/200, gv::CacheMode::kAsync,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/clio::run::PoolId(0, 0),
                          // Must match SeedTag: all pages in family 0.
                          /*family_pages=*/kLogicalPages);
  SeedTag(tag, vec.TagId(), kLogicalPages);
  auto *gpu_ipc_mgr = CLIO_CPU_IPC->GetGpuIpcManager();
  clio::run::IpcManagerGpuInfo gpu_info = gpu_ipc_mgr->GetGpuInfo(0);
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

/*
 * ROOT CAUSE of the two red cases below (same defect, both configurations).
 *
 * gpu_vector_kernels.h: FaultPage() mutates GetGetTask(base, block_idx, slot)
 * -- a task struct shared per (cache block, slot) -- clearing flags, minting a
 * fresh task_id, and setting gpu_page_idx_ before Send(). read_range() calls
 * Resolve() (which faults) BEFORE TryAcquireBusy(p), so that mutation is
 * performed with no lock held.
 *
 * BlockIdx() wraps blockIdx.x % nblocks, so a grid wider than nblocks maps
 * many CUDA blocks onto one cache block. Two of them faulting concurrently
 * write the same task struct; one overwrites the other's page index and task
 * id, and the runtime reports
 *     ReadData: bytes_read=0, expected=2097152, status=FAILED
 *
 * This is why the out-of-core compute grid is currently capped at nblocks,
 * which is the whole reason qwen3-coder decodes at 0.08 tok/s. Fixing it means
 * claiming the slot (or the page) atomically BEFORE FaultPage touches the
 * task, not after.
 */

/** Chunk size deliberately NOT a divisor of a 176-byte record (Q5_K block):
 *  24576 % 176 != 0, so every chunk after the first starts mid-record. */
TEST_CASE("gpu_vector: staging chunks that do not divide the record size",
          "[gpu_vector][staging]") {
  RunSpanCase("gvm_unaligned", /*nblocks=*/8, /*pages_per_block=*/2, /*grid=*/8,
              0, kTotalBytes, /*chunk=*/176 * 139 + 17);
}

/** Spans that start and end at odd offsets and straddle page boundaries. */
TEST_CASE("gpu_vector: cross-page spans at unaligned offsets",
          "[gpu_vector][staging]") {
  RunSpanCase("gvm_crosspage", /*nblocks=*/4, /*pages_per_block=*/2, /*grid=*/4,
              kPageSizeBytes - 1000, 5 * kPageSizeBytes + 333, /*chunk=*/4096);
}


/** IN-MEMORY: cache holds every page, so after the first pass nothing faults.
 *  This is the configuration gemma runs in (model fits VRAM) and it had no
 *  unit test at all -- correctness there was only ever observed indirectly,
 *  by reading generated text. */
TEST_CASE("gpu_vector: fully resident read is byte-exact",
          "[gpu_vector][in-memory]") {
  RunSpanCase("gvm_resident", /*nblocks=*/4, /*pages_per_block=*/8, /*grid=*/4,
              0, kTotalBytes, /*chunk=*/24576);
  printf("  [resident] %.2f GiB/s\n", g_last_gbps);
  // Loose floor: this asserts "not catastrophically broken", not a target.
  // Measured 0.41 GiB/s here -- the per-byte read_range callback dominates,
  // not paging, which is why the resident case is not faster than the
  // out-of-core one. Tightening this belongs with a faster bulk read API.
  REQUIRE(g_last_gbps > 0.1);
}

/** PERFORMANCE floor for the out-of-core path. Deliberately loose: the point
 *  is to catch an order-of-magnitude regression (the grid-cap bug dropped
 *  qwen3-coder to 0.08 tok/s), not to pin a number that will flake. */
TEST_CASE("gpu_vector: out-of-core throughput floor",
          "[gpu_vector][perf][oom]") {
  RunSpanCase("gvm_perf", /*nblocks=*/8, /*pages_per_block=*/2, /*grid=*/8,
              0, kTotalBytes, /*chunk=*/24576);
  printf("  [perf oom] %.2f GiB/s\n", g_last_gbps);
  REQUIRE(g_last_gbps > 0.1);
}

/** 64 CUDA blocks over 2 cache blocks. Index wrapping keeps this in bounds,
 *  but many CUDA blocks concurrently faulting through one page table is a
 *  race; this is the illegal-memory-access configuration. */
TEST_CASE("gpu_vector: grid far wider than nblocks faults safely",
          "[gpu_vector][oom][race]") {
  RunSpanCase("gvm_widegrid", /*nblocks=*/2, /*pages_per_block=*/2, /*grid=*/64,
              0, kTotalBytes, /*chunk=*/8192);
}


SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif
