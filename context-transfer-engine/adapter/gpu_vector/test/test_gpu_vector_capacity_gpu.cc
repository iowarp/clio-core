/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Capacity stress test for the compressed GPU vector.
 *
 * Claim under test: GPU compression lets a dataset LARGER than the HBM budget
 * be stored ENTIRELY ON THE GPU. The compressor's downstream storage tier is a
 * kHbm bdev (device memory), so compressed cold pages live in HBM, not host
 * DRAM. A Gray-Scott field of logical size >> the HBM budget is written through
 * the compressed vector and lands, compressed ~cuSZp-ratio smaller, inside that
 * HBM budget.
 *
 * Why chunked + host-flushed (not one giant live vector): a vector that pages
 * larger-than-HBM ON-DEVICE deadlocks under GPU compression -- both the write
 * eviction path (EvictSlot -> DrainPut spins on-device for a compress PutBlob)
 * and the read fault path spin-wait on the GPU for an operation that itself
 * needs the GPU (cuSZp's kernel + its internal cudaMalloc/cudaMemcpy device
 * syncs). So we stream the logical dataset in cache-sized chunks: write a chunk
 * (fits the HBM cache), host-flush it (LegacyFlushKernel exits, GPU idle while
 * the compressor runs), reuse the cache for the next chunk. Each chunk is a
 * distinct tag, so all chunks accumulate in the shared kHbm store.
 *
 * Result reported: logical bytes vs. measured HBM footprint (bdev used bytes)
 * vs. the HBM budget -- demonstrating the whole dataset fits on the GPU
 * compressed, where uncompressed it would not.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <clio_ctp/util/gpu_api.h>
#include <clio_ctp/introspect/system_info.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace {

bool g_initialized = false;

inline clio::run::PoolId StoragePool() { return clio::run::PoolId(600, 0); }
// The CTE core's own HBM storage tier (compose "hbm::cte_hbm_tier") is created
// as the core pool's minor-1 bdev; that's where compressed blobs actually land.
inline clio::run::PoolId HbmBdevPool() { return clio::run::PoolId(512, 1); }

// HBM budget for the compressed store (kHbm bdev). The logical dataset is made
// several times larger than this to prove it still fits, compressed, in HBM.
constexpr clio::run::u64 kHbmBudgetMiB = 128;

/** Smooth, bounded Gray-Scott-like field; per-chunk phase so chunks differ. */
__host__ __device__ inline float Field(clio::run::u64 i, clio::run::u32 chunk) {
  float x = static_cast<float>(i) * 1.0e-4f +
            static_cast<float>(chunk) * 0.37f;
  return 0.5f + 0.4f * sinf(x);
}

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10560;
  std::string cfg_path = "/tmp/gpu_vec_capacity_" + std::to_string(port) + ".yaml";
  {
    std::ofstream cfg(cfg_path);
    cfg << "networking:\n  port: " << port << "\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
        << "compose:\n"
        // Compressed store lives in GPU HBM (device memory), not host DRAM.
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"hbm::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: hbm\n    capacity: \"" << kHbmBudgetMiB << "MB\"\n\n"
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
        << "    storage:\n      - path: \"hbm::cte_hbm_tier\"\n"
        << "        bdev_type: \"hbm\"\n        capacity_limit: \""
        << kHbmBudgetMiB << "MB\"\n        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", cfg_path.c_str(), 1);
  std::fprintf(stderr, "[CAP] compose=%s (HBM store budget %lluMiB)\n",
               cfg_path.c_str(), (unsigned long long)kHbmBudgetMiB);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

__global__ void CapWriteKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceView<float> view,
                               clio::run::u64 total, clio::run::u32 chunk) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  clio::run::u64 stripe = (total + gridDim.x - 1) / gridDim.x;
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * stripe;
  clio::run::u64 hi = lo + stripe;
  if (hi > total) hi = total;
  v.write_range(lo, hi, [chunk](clio::run::u64 i) { return Field(i, chunk); });
  (void)g_ipc_manager;
}

__global__ void CapReadKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceView<float> view, float *result,
                              clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  clio::run::u64 stripe = (total + gridDim.x - 1) / gridDim.x;
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * stripe;
  clio::run::u64 hi = lo + stripe;
  if (hi > total) hi = total;
  v.read_range(lo, hi, [result](clio::run::u64 i, float val) {
    result[i] = val;
  });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: capacity -- dataset larger than HBM budget fits, "
          "compressed, on the GPU", "[gpu_vector][compress][capacity][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const clio::run::u32 nblocks = 4;
  const clio::run::u32 ppb = 4;
  const clio::run::u64 page_size = 1ULL << 20;           // 1 MiB pages
  const clio::run::u64 epp = page_size / sizeof(float);
  const clio::run::u64 cache_bytes =
      static_cast<clio::run::u64>(nblocks) * ppb * page_size;   // HBM cache
  const clio::run::u64 cache_elems = cache_bytes / sizeof(float);
  const clio::run::u32 kChunks = 16;                     // logical = 16x cache
  const clio::run::u64 logical_bytes =
      cache_bytes * kChunks;
  const clio::run::u64 budget_bytes = kHbmBudgetMiB << 20;

  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(0);

  std::fprintf(stderr,
      "[CAP] HBM cache=%lluMiB  page=%lluKiB  chunks=%u  logical=%lluMiB  "
      "HBM budget=%lluMiB\n",
      (unsigned long long)(cache_bytes >> 20),
      (unsigned long long)(page_size >> 10), kChunks,
      (unsigned long long)(logical_bytes >> 20),
      (unsigned long long)kHbmBudgetMiB);
  REQUIRE(logical_bytes > budget_bytes);  // dataset genuinely exceeds HBM budget

  // Measure the HBM store's free space before and after.
  clio::run::bdev::Client bdev(HbmBdevPool());
  auto s0 = bdev.AsyncGetStats(); s0.Wait();
  clio::run::u64 remaining_before = s0->remaining_size_;

  // Stream the logical dataset in cache-sized chunks; each chunk is compressed
  // into the shared HBM store under its own tag.
  for (clio::run::u32 c = 0; c < kChunks; ++c) {
    std::string tag = "gscap_c" + std::to_string(c);
    // kAsync so the async cache manager flushes EVERY dirty page (Phase 3) to
    // the compressor. All pages fit the cache (no eviction), so the write never
    // spin-waits on-device for a compress -> no deadlock; the manager's flush
    // kernel exits and the compressor runs while we sleep (GPU idle).
    gv::Vector<float> vec(tag, nblocks, /*gpu_id=*/0, ppb,
                          /*host_pages_per_block=*/0, page_size,
                          /*cache_period_us=*/20000, gv::CacheMode::kAsync,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/StoragePool());
    auto view = vec.Device();
    CapWriteKernel<<<nblocks, 32>>>(gpu_info, view, cache_elems, c);
    ctp::GpuApi::Synchronize();
    std::this_thread::sleep_for(1500ms);  // let the manager flush + compress all pages
  }

  auto s1 = bdev.AsyncGetStats(); s1.Wait();
  clio::run::u64 remaining_after = s1->remaining_size_;
  clio::run::u64 hbm_used = (remaining_before >= remaining_after)
                                ? (remaining_before - remaining_after) : 0;

  double ratio = hbm_used ? (double)logical_bytes / (double)hbm_used : 0.0;
  std::fprintf(stderr,
      "[CAP] stored logical=%lluMiB into HBM used=%lluMiB (budget %lluMiB)  "
      "=> on-GPU ratio %.1fx\n",
      (unsigned long long)(logical_bytes >> 20),
      (unsigned long long)(hbm_used >> 20),
      (unsigned long long)kHbmBudgetMiB, ratio);
  std::fprintf(stderr,
      "[CAP] uncompressed would need %lluMiB (> %lluMiB budget => would NOT fit "
      "on GPU); compressed fits with %lluMiB HBM to spare\n",
      (unsigned long long)(logical_bytes >> 20),
      (unsigned long long)kHbmBudgetMiB,
      (unsigned long long)((budget_bytes > hbm_used ? budget_bytes - hbm_used : 0) >> 20));

  // The whole dataset lives in HBM, compressed, within the budget.
  REQUIRE(hbm_used > 0);
  REQUIRE(hbm_used <= budget_bytes);
  REQUIRE(logical_bytes > budget_bytes);   // and it was bigger than the budget

  // Correctness spot-check: materialize chunk 0 from the HBM store and compare.
  {
    gv::Vector<float> vec("gscap_c0", nblocks, /*gpu_id=*/0, ppb,
                          /*host_pages_per_block=*/0, page_size,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/StoragePool());
    vec.FaultAllSync();
    auto *result = ctp::GpuApi::MallocHost<float>(cache_elems * sizeof(float));
    REQUIRE(result != nullptr);
    std::memset(result, 0, cache_elems * sizeof(float));
    auto view = vec.Device();
    // Read chunk 0 back through a device read kernel over resident pages.
    CapReadKernel<<<nblocks, 32>>>(gpu_info, view, result, cache_elems);
    ctp::GpuApi::Synchronize();
    double max_err = 0.0;
    for (clio::run::u64 i = 0; i < cache_elems; ++i) {
      double e = std::fabs((double)result[i] - (double)Field(i, 0));
      if (e > max_err) max_err = e;
    }
    std::fprintf(stderr, "[CAP] chunk0 readback max_abs_err=%.3e (eb=1e-3)\n",
                 max_err);
    REQUIRE(max_err <= 2.0e-3);
    ctp::GpuApi::FreeHost(result);
  }

  std::fprintf(stderr,
      "[CAP] PASS: %lluMiB logical Gray-Scott stored entirely on the GPU in "
      "%lluMiB HBM (%.1fx), exceeding the %lluMiB uncompressed budget.\n",
      (unsigned long long)(logical_bytes >> 20),
      (unsigned long long)(hbm_used >> 20), ratio,
      (unsigned long long)kHbmBudgetMiB);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
