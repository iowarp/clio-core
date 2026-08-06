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

// llama geometry, scaled down: 2 MiB byte pages, 8 blocks.
constexpr clio::run::u64 kPageSizeBytes = 2ull << 20;
constexpr clio::run::u32 kLogicalPages  = 32;   // 64 MiB extent
constexpr clio::run::u32 kNblocks       = 8;
constexpr clio::run::u32 kCachePerBlock = 4;    // cold + eviction pressure
// CAE sharding: family = page / ceil(num_pages / nblocks).
constexpr clio::run::u64 kFamPages      = (kLogicalPages + kNblocks - 1) / kNblocks;
// Same staging size as the llama paged-dequant kernel.
constexpr clio::run::u32 kStageBytes    = 12288;

/** Deterministic per-byte pattern: page and offset both influence it. */
CTP_CROSS_FUN inline uint8_t PatternAt(clio::run::u64 byte_idx) {
  const clio::run::u64 pg = byte_idx / kPageSizeBytes;
  return static_cast<uint8_t>((pg * 131u + byte_idx * 7u) & 0xFFu);
}

}  // namespace

/**
 * The llama paged-GEMV shape: each CUDA block walks a byte stripe in
 * kStageBytes chunks; read_range streams the bytes into __shared__ staging;
 * the warp then verifies the staged bytes. Stripes are REVERSED across blocks
 * (block b reads stripe kNblocks-1-b) so every block reads pages whose family
 * is NOT the block's own index — the exact cross-family faulting the family
 * policy exists for.
 */
__global__ void LlamaGeomReadKernel(clio::run::IpcManagerGpuInfo info,
                                    gv::DeviceView<uint8_t> view,
                                    unsigned long long *mismatches,
                                    unsigned long long *bytes_checked,
                                    clio::run::u64 total_bytes) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  // The page->family mapping handed to the device-side vector as a lambda —
  // here the view's fam_ppb policy (== the CAE assimilator's sharding).
  auto fam = [] (const gv::DeviceViewBase &b, clio::run::u64 pg) {
    return gv::FamilyOf(b, pg);
  };
  dev::vector<uint8_t, decltype(fam)> v(view, g_ipc_manager_ptr, fam);
  const clio::run::u32 lane = threadIdx.x & 31;

  __shared__ uint8_t stage_s[kStageBytes];
  uint8_t *const stage = stage_s;

  const clio::run::u64 stripe_bytes = total_bytes / gridDim.x;
  const clio::run::u32 stripe       = gridDim.x - 1 - blockIdx.x;  // reversed
  const clio::run::u64 lo0          = static_cast<clio::run::u64>(stripe) * stripe_bytes;
  const clio::run::u64 hi0          = lo0 + stripe_bytes;

  unsigned long long local_bad = 0, local_n = 0;
  for (clio::run::u64 lo = lo0; lo < hi0; lo += kStageBytes) {
    clio::run::u64 hi = lo + kStageBytes;
    if (hi > hi0) hi = hi0;
    v.read_range(lo, hi, [stage, lo] (clio::run::u64 i, uint8_t val) {
      stage[i - lo] = val;
    });
    __syncwarp();
    for (clio::run::u64 i = lo + lane; i < hi; i += 32) {
      ++local_n;
      if (stage[i - lo] != PatternAt(i)) ++local_bad;
    }
    __syncwarp();
  }
  atomicAdd(mismatches, local_bad);
  atomicAdd(bytes_checked, local_n);
  (void)g_ipc_manager;
}

/** Every block reads the SAME [base, base+span) — llama's small-tensor GEMV
 *  shape: concurrent same-page faults into 8 private caches, unaligned. */
__global__ void SamePageReadKernel(clio::run::IpcManagerGpuInfo info,
                                   gv::DeviceView<uint8_t> view,
                                   unsigned long long *mismatches,
                                   unsigned long long *bytes_checked,
                                   clio::run::u64 base, clio::run::u64 span) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  auto fam = [] (const gv::DeviceViewBase &b, clio::run::u64 pg) {
    return gv::FamilyOf(b, pg);
  };
  dev::vector<uint8_t, decltype(fam)> v(view, g_ipc_manager_ptr, fam);
  const clio::run::u32 lane = threadIdx.x & 31;

  __shared__ uint8_t stage_s[kStageBytes];
  uint8_t *const stage = stage_s;

  unsigned long long local_bad = 0, local_n = 0;
  for (clio::run::u64 lo = base; lo < base + span; lo += kStageBytes) {
    clio::run::u64 hi = lo + kStageBytes;
    if (hi > base + span) hi = base + span;
    v.read_range(lo, hi, [stage, lo] (clio::run::u64 i, uint8_t val) {
      stage[i - lo] = val;
    });
    __syncwarp();
    for (clio::run::u64 i = lo + lane; i < hi; i += 32) {
      ++local_n;
      if (stage[i - lo] != PatternAt(i)) ++local_bad;
    }
    __syncwarp();
  }
  atomicAdd(mismatches, local_bad);
  atomicAdd(bytes_checked, local_n);
  (void)g_ipc_manager;
}

/** The llama perf-kernel structure: 256 threads, warp 0 drives read_range
 *  into __shared__ staging, all 8 warps consume. Reproduces (or clears) the
 *  hang seen only at blockDim>32 in the llama process. */
__global__ void Warp0Read256Kernel(clio::run::IpcManagerGpuInfo info,
                                   gv::DeviceView<uint8_t> view,
                                   unsigned long long *mismatches,
                                   unsigned long long *bytes_checked,
                                   clio::run::u64 total_bytes) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  auto fam = [] (const gv::DeviceViewBase &b, clio::run::u64 pg) {
    return gv::FamilyOf(b, pg);
  };
  dev::vector<uint8_t, decltype(fam)> v(view, g_ipc_manager_ptr, fam);
  const clio::run::u32 warp = threadIdx.x >> 5;

  __shared__ uint8_t stage_s[kStageBytes];
  uint8_t *const stage = stage_s;

  const clio::run::u64 stripe_bytes = total_bytes / gridDim.x;
  const clio::run::u32 stripe       = gridDim.x - 1 - blockIdx.x;
  const clio::run::u64 lo0          = static_cast<clio::run::u64>(stripe) * stripe_bytes;
  const clio::run::u64 hi0          = lo0 + stripe_bytes;

  unsigned long long local_bad = 0, local_n = 0;
  for (clio::run::u64 lo = lo0; lo < hi0; lo += kStageBytes) {
    clio::run::u64 hi = lo + kStageBytes;
    if (hi > hi0) hi = hi0;
    if (warp == 0) {
      v.read_range(lo, hi, [stage, lo] (clio::run::u64 i, uint8_t val) {
        stage[i - lo] = val;
      });
    }
    __syncthreads();
    for (clio::run::u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      ++local_n;
      if (stage[i - lo] != PatternAt(i)) ++local_bad;
    }
    __syncthreads();
  }
  atomicAdd(mismatches, local_bad);
  atomicAdd(bytes_checked, local_n);
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

namespace {

void EnsureInit() {
  if (g_initialized) return;
  std::fprintf(stderr, "[INIT] Starting Clio server (llama-geom test)\n");
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);
  cte_client->Init(clio::cte::core::kCtePoolId);
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
      clio::run::PoolQuery::Dynamic(), std::string("gpu_vector_llamageom_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "gpu_vector_llamageom_ram", clio::run::bdev::BdevType::kRam,
      kRamCapacity, clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  g_initialized = true;
}

/**
 * Seed the tag EXACTLY the way the CAE ModelWeightsAssimilator does: external
 * PutBlob of "<tag>_b<page/kFamPages>_pi<page>" — sharded families, written by
 * a component that knows nothing about the reader's cache layout.
 */
void PopulateCaeSharded(const std::string &tag) {
  gv::Vector<uint8_t> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                          /*gpu_pages_per_block=*/1,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true);
  auto tag_id = vec.TagId();
  auto *cte = CLIO_CTE_CLIENT;
  for (clio::run::u32 p = 0; p < kLogicalPages; ++p) {
    auto buf = CLIO_IPC->AllocateBuffer(kPageSizeBytes);
    auto *bytes = reinterpret_cast<uint8_t *>(buf.ptr_);
    const clio::run::u64 base = static_cast<clio::run::u64>(p) * kPageSizeBytes;
    for (clio::run::u64 j = 0; j < kPageSizeBytes; ++j) {
      bytes[j] = PatternAt(base + j);
    }
    std::string blob_name = tag + "_b" + std::to_string(p / kFamPages) +
                            "_pi" + std::to_string(p);
    auto task = cte->AsyncPutBlob(tag_id, blob_name, 0, kPageSizeBytes,
                                  buf.shm_.template Cast<void>(), 1.0f,
                                  clio::cte::core::Context(), 0);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
  }
  std::fprintf(stderr, "[PUTBLOB] wrote %u CAE-sharded pages to '%s' "
                       "(fam_ppb=%llu)\n",
               kLogicalPages, tag.c_str(),
               static_cast<unsigned long long>(kFamPages));
}

void RunAndVerify(const std::string &tag) {
  auto *gpu_ipc_mgr = CLIO_CPU_IPC->GetGpuIpcManager();
  REQUIRE(gpu_ipc_mgr != nullptr);
  clio::run::IpcManagerGpuInfo gpu_info = gpu_ipc_mgr->GetGpuInfo(0);

  // FRESH multi-block vector, cold cache, CAE family policy — llama's config.
  gv::Vector<uint8_t> vec(tag, kNblocks, /*gpu_id=*/0, kCachePerBlock,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/200, gv::CacheMode::kAsync,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/clio::run::PoolId(0, 0),
                          /*family_pages=*/kFamPages);

  auto *mismatches =
      ctp::GpuApi::MallocHost<unsigned long long>(2 * sizeof(unsigned long long));
  REQUIRE(mismatches != nullptr);
  mismatches[0] = 0;
  mismatches[1] = 0;

  const clio::run::u64 total =
      static_cast<clio::run::u64>(kLogicalPages) * kPageSizeBytes;
  std::fprintf(stderr,
               "[READ] %u blocks x %u cached pages vs %u logical pages "
               "(%.1fx per-block oversubscription), reversed stripes\n",
               kNblocks, kCachePerBlock, kLogicalPages,
               static_cast<double>(kLogicalPages) / kCachePerBlock);
  LlamaGeomReadKernel<<<kNblocks, 32>>>(gpu_info, vec.Device(), &mismatches[0],
                                        &mismatches[1], total);
  cudaError_t err = cudaDeviceSynchronize();
  std::fprintf(stderr, "[READ] kernel -> %s | checked=%llu mismatches=%llu\n",
               cudaGetErrorString(err), mismatches[1], mismatches[0]);
  REQUIRE(err == cudaSuccess);
  REQUIRE(mismatches[1] == total);  // every byte covered (32 lanes x stripes)
  REQUIRE(mismatches[0] == 0);      // and correct
}

void RunSamePageStress(const std::string &tag) {
  auto *gpu_ipc_mgr = CLIO_CPU_IPC->GetGpuIpcManager();
  REQUIRE(gpu_ipc_mgr != nullptr);
  clio::run::IpcManagerGpuInfo gpu_info = gpu_ipc_mgr->GetGpuInfo(0);

  gv::Vector<uint8_t> vec(tag, kNblocks, /*gpu_id=*/0, kCachePerBlock,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/200, gv::CacheMode::kAsync,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/clio::run::PoolId(0, 0),
                          /*family_pages=*/kFamPages);

  auto *mismatches =
      ctp::GpuApi::MallocHost<unsigned long long>(2 * sizeof(unsigned long long));
  REQUIRE(mismatches != nullptr);

  // inp_gate-like span: ~1.5 MiB starting at an odd sub-page offset inside
  // page 17 (family 4) — every block reads the SAME overlapping span.
  const clio::run::u64 base = 17ull * kPageSizeBytes + 887296;  // unaligned
  const clio::run::u64 span = 1536ull * 256ull * 4ull;          // 1.5 MiB
  for (int launch = 0; launch < 6; ++launch) {                  // B=6 shape
    mismatches[0] = 0;
    mismatches[1] = 0;
    SamePageReadKernel<<<kNblocks, 32>>>(gpu_info, vec.Device(),
                                         &mismatches[0], &mismatches[1],
                                         base, span);
    cudaError_t err = cudaDeviceSynchronize();
    std::fprintf(stderr,
                 "[SAMEPAGE] launch %d -> %s | checked=%llu mismatches=%llu\n",
                 launch, cudaGetErrorString(err), mismatches[1], mismatches[0]);
    REQUIRE(err == cudaSuccess);
    REQUIRE(mismatches[1] ==
            static_cast<unsigned long long>(span) * kNblocks);
    REQUIRE(mismatches[0] == 0);
  }
}

}  // namespace

TEST_CASE("gpu_vector: multi-block cold faults on a CAE-sharded tag",
          "[gpu_vector][cte][fault][family][llama-geom]") {
  EnsureInit();
  const std::string tag = "gpu_vector_llamageom";
  PopulateCaeSharded(tag);
  RunAndVerify(tag);
}

TEST_CASE("gpu_vector: warp0-driven read_range at blockDim=256",
          "[gpu_vector][cte][fault][family][llama-geom]") {
  EnsureInit();
  const std::string tag = "gpu_vector_llamageom3";
  PopulateCaeSharded(tag);

  auto *gpu_ipc_mgr = CLIO_CPU_IPC->GetGpuIpcManager();
  REQUIRE(gpu_ipc_mgr != nullptr);
  clio::run::IpcManagerGpuInfo gpu_info = gpu_ipc_mgr->GetGpuInfo(0);
  gv::Vector<uint8_t> vec(tag, kNblocks, /*gpu_id=*/0, kCachePerBlock,
                          /*host_pages_per_block=*/0, kPageSizeBytes,
                          /*cache_period_us=*/200, gv::CacheMode::kAsync,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/clio::run::PoolId(0, 0),
                          /*family_pages=*/kFamPages);
  auto *mm = ctp::GpuApi::MallocHost<unsigned long long>(2 * sizeof(unsigned long long));
  REQUIRE(mm != nullptr);
  mm[0] = 0; mm[1] = 0;
  const clio::run::u64 total =
      static_cast<clio::run::u64>(kLogicalPages) * kPageSizeBytes;
  Warp0Read256Kernel<<<kNblocks, 256>>>(gpu_info, vec.Device(), &mm[0], &mm[1], total);
  cudaError_t err = cudaDeviceSynchronize();
  std::fprintf(stderr, "[W0R256] kernel -> %s | checked=%llu mismatches=%llu\n",
               cudaGetErrorString(err), mm[1], mm[0]);
  REQUIRE(err == cudaSuccess);
  REQUIRE(mm[1] == total);
  REQUIRE(mm[0] == 0);
}

TEST_CASE("gpu_vector: all blocks fault the SAME page, unaligned, repeated",
          "[gpu_vector][cte][fault][family][llama-geom]") {
  // The llama first-GEMV shape that the stripe case does not cover:
  // blk.0.inp_gate.weight is smaller than one page, so every CUDA block
  // faults the SAME global page concurrently (8 private cache copies), the
  // tensor base is NOT page-aligned, and the batch loop launches the kernel
  // 6x back-to-back with no intervening sync.
  EnsureInit();
  const std::string tag = "gpu_vector_llamageom2";
  PopulateCaeSharded(tag);
  RunSamePageStress(tag);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
