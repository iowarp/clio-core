/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Tier-SPILL stress test for the compressed GPU vector.
 *
 * The capacity test proves a dataset larger than the HBM budget fits ENTIRELY in
 * HBM once compressed (compressed footprint < HBM). This test drives the case
 * one step further: the *compressed* footprint itself EXCEEDS the HBM tier, so
 * the tiering must engage -- HBM fills first, the overflow spills to a slower
 * DRAM tier -- and reads must still round-trip pages from BOTH tiers.
 *
 * Compose: the compressor's store (pool 512) has TWO tiers, and the max_bw DPE
 * fills the higher-score (HBM) tier until full, then the DRAM tier:
 *   tier 0  HBM  (kHbm, small, score 1.0)  -> bdev pool 512.1
 *   tier 1  DRAM (kRam, large, score 0.5)  -> bdev pool 513.1
 * (Storage device i on node n gets bdev pool PoolId(512+i, 1+n) --
 * core_runtime.cc RegisterTarget.)
 *
 * The logical dataset is sized so that even compressed it is several times the
 * HBM tier, guaranteeing a spill. Streamed in cache-sized chunks (like the
 * capacity test) so the on-device write path never deadlocks.
 *
 * Asserts: (a) HBM tier holds compressed data and is ~full (<= its cap);
 * (b) the DRAM tier holds compressed data too (spill actually happened);
 * (c) a chunk written early (HBM) and a chunk written after HBM filled (DRAM)
 * both decompress correctly -- i.e. the fault path works across the tier
 * boundary, including from the spilled DRAM tier.
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
// Two compressed-store tiers (see the compose): device 0 -> 512.1 (HBM),
// device 1 -> 513.1 (DRAM overflow).
inline clio::run::PoolId HbmTierPool() { return clio::run::PoolId(512, 1); }
inline clio::run::PoolId DramTierPool() { return clio::run::PoolId(513, 1); }

// Small HBM store so the compressed footprint clearly overruns it and spills.
constexpr clio::run::u64 kHbmTierMiB = 24;
constexpr clio::run::u64 kDramTierMiB = 4096;

/** Smooth, bounded Gray-Scott-like field; per-chunk phase so chunks differ. */
__host__ __device__ inline float Field(clio::run::u64 i, clio::run::u32 chunk) {
  float x = static_cast<float>(i) * 1.0e-4f + static_cast<float>(chunk) * 0.37f;
  return 0.5f + 0.4f * sinf(x);
}

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10580;
  std::string cfg_path = "/tmp/gpu_vec_spill_" + std::to_string(port) + ".yaml";
  {
    std::ofstream cfg(cfg_path);
    cfg << "networking:\n  port: " << port << "\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"hbm::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: hbm\n    capacity: \"64MB\"\n\n"
        << "  - mod_name: clio_cte_compressor\n"
        << "    pool_name: cte_compressor\n    pool_query: local\n"
        << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
        << "    storage:\n"
        // Tier 0: HBM (device memory), high score -> filled first.
        << "      - path: \"hbm::cte_hbm_tier\"\n"
        << "        bdev_type: \"hbm\"\n        capacity_limit: \""
        << kHbmTierMiB << "MB\"\n        score: 1.0\n"
        // Tier 1: DRAM (host memory), lower score -> receives the overflow.
        << "      - path: \"ram::cte_dram_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \""
        << kDramTierMiB << "MB\"\n        score: 0.5\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg_path.c_str(), 1);
  std::fprintf(stderr, "[SPILL] compose=%s (HBM tier %lluMiB, DRAM tier %lluMiB)\n",
               cfg_path.c_str(), (unsigned long long)kHbmTierMiB,
               (unsigned long long)kDramTierMiB);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

__global__ void SpillWriteKernel(clio::run::IpcManagerGpuInfo info,
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

__global__ void SpillReadKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceView<float> view, float *result,
                                clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  clio::run::u64 stripe = (total + gridDim.x - 1) / gridDim.x;
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * stripe;
  clio::run::u64 hi = lo + stripe;
  if (hi > total) hi = total;
  v.read_range(lo, hi, [result](clio::run::u64 i, float val) { result[i] = val; });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: tier spill -- compressed footprint exceeds HBM, overflow "
          "spills to DRAM, reads round-trip from both tiers",
          "[gpu_vector][compress][spill][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const clio::run::u32 nblocks = 4;
  const clio::run::u32 ppb = 4;
  const clio::run::u64 page_size = 1ULL << 20;            // 1 MiB pages
  const clio::run::u64 epp = page_size / sizeof(float);
  const clio::run::u64 cache_bytes =
      static_cast<clio::run::u64>(nblocks) * ppb * page_size;    // HBM cache
  const clio::run::u64 cache_elems = cache_bytes / sizeof(float);
  const clio::run::u32 kChunks = 64;                     // logical = 64x cache
  const clio::run::u64 logical_bytes = cache_bytes * kChunks;    // 1 GiB
  const clio::run::u64 hbm_tier_bytes = kHbmTierMiB << 20;

  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(0);

  std::fprintf(stderr,
      "[SPILL] HBM cache=%lluMiB  page=%lluKiB  chunks=%u  logical=%lluMiB  "
      "HBM tier=%lluMiB  DRAM tier=%lluMiB\n",
      (unsigned long long)(cache_bytes >> 20),
      (unsigned long long)(page_size >> 10), kChunks,
      (unsigned long long)(logical_bytes >> 20),
      (unsigned long long)kHbmTierMiB, (unsigned long long)kDramTierMiB);

  clio::run::bdev::Client hbm(HbmTierPool());
  clio::run::bdev::Client dram(DramTierPool());
  auto h0 = hbm.AsyncGetStats(); h0.Wait();
  auto d0 = dram.AsyncGetStats(); d0.Wait();
  clio::run::u64 hbm_rem0 = h0->remaining_size_, dram_rem0 = d0->remaining_size_;

  // Stream the logical dataset in cache-sized chunks; each chunk compresses into
  // the shared store. The DPE fills the HBM tier first, then spills to DRAM.
  for (clio::run::u32 c = 0; c < kChunks; ++c) {
    std::string tag = "gsspill_c" + std::to_string(c);
    gv::Vector<float> vec(tag, nblocks, /*gpu_id=*/0, ppb,
                          /*host_pages_per_block=*/0, page_size,
                          /*cache_period_us=*/20000, gv::CacheMode::kAsync,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/StoragePool());
    auto view = vec.Device();
    SpillWriteKernel<<<nblocks, 32>>>(gpu_info, view, cache_elems, c);
    ctp::GpuApi::Synchronize();
    std::this_thread::sleep_for(800ms);  // let the manager flush + compress
  }

  auto h1 = hbm.AsyncGetStats(); h1.Wait();
  auto d1 = dram.AsyncGetStats(); d1.Wait();
  clio::run::u64 hbm_used =
      (hbm_rem0 >= h1->remaining_size_) ? hbm_rem0 - h1->remaining_size_ : 0;
  clio::run::u64 dram_used =
      (dram_rem0 >= d1->remaining_size_) ? dram_rem0 - d1->remaining_size_ : 0;
  clio::run::u64 total_stored = hbm_used + dram_used;

  std::fprintf(stderr,
      "[SPILL] compressed footprint: HBM tier=%lluMiB (cap %lluMiB) + "
      "DRAM tier=%lluMiB = %lluMiB total (logical %lluMiB, ratio %.1fx)\n",
      (unsigned long long)(hbm_used >> 20), (unsigned long long)kHbmTierMiB,
      (unsigned long long)(dram_used >> 20),
      (unsigned long long)(total_stored >> 20),
      (unsigned long long)(logical_bytes >> 20),
      total_stored ? (double)logical_bytes / (double)total_stored : 0.0);

  // (a) HBM tier holds compressed data and never exceeds its cap.
  REQUIRE(hbm_used > 0);
  REQUIRE(hbm_used <= hbm_tier_bytes);
  // (b) The compressed footprint overran HBM, so the overflow spilled to DRAM.
  REQUIRE(dram_used > 0);
  REQUIRE(total_stored > hbm_tier_bytes);  // compressed data exceeds HBM tier
  std::fprintf(stderr,
      "[SPILL] SPILL CONFIRMED: compressed %lluMiB > %lluMiB HBM tier; "
      "%lluMiB overflow lives in the DRAM tier.\n",
      (unsigned long long)(total_stored >> 20), (unsigned long long)kHbmTierMiB,
      (unsigned long long)(dram_used >> 20));

  // (c) Reads round-trip from BOTH tiers: chunk 0 (written first -> HBM) and the
  // last chunk (written after HBM filled -> DRAM).
  auto check_chunk = [&](clio::run::u32 c) {
    gv::Vector<float> vec("gsspill_c" + std::to_string(c), nblocks, /*gpu_id=*/0,
                          ppb, /*host_pages_per_block=*/0, page_size,
                          /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                          /*manager_threads_per_block=*/32,
                          /*allow_cold_miss_fault=*/true,
                          /*storage_pool_id=*/StoragePool());
    vec.FaultAllSync();
    auto *result = ctp::GpuApi::MallocHost<float>(cache_elems * sizeof(float));
    REQUIRE(result != nullptr);
    std::memset(result, 0, cache_elems * sizeof(float));
    auto view = vec.Device();
    SpillReadKernel<<<nblocks, 32>>>(gpu_info, view, result, cache_elems);
    ctp::GpuApi::Synchronize();
    double max_err = 0.0;
    for (clio::run::u64 i = 0; i < cache_elems; ++i)
      max_err = std::max(max_err,
                         std::fabs((double)result[i] - (double)Field(i, c)));
    std::fprintf(stderr, "[SPILL] chunk %u readback max_abs_err=%.3e (eb=1e-3)\n",
                 c, max_err);
    REQUIRE(max_err <= 2.0e-3);
    ctp::GpuApi::FreeHost(result);
  };
  check_chunk(0);              // early -> HBM tier
  check_chunk(kChunks - 1);    // late  -> spilled DRAM tier

  std::fprintf(stderr,
      "[SPILL] PASS: %lluMiB logical -> %lluMiB compressed split across HBM "
      "(%lluMiB) + DRAM (%lluMiB); reads round-trip from both tiers.\n",
      (unsigned long long)(logical_bytes >> 20),
      (unsigned long long)(total_stored >> 20),
      (unsigned long long)(hbm_used >> 20),
      (unsigned long long)(dram_used >> 20));
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
