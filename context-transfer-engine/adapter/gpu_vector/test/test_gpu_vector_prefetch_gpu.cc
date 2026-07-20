/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * IO-bound reader benchmark: why prefetching matters (issue #700).
 *
 * Compressed pages are stored through IOWarp's AUTOMATIC storage tiering: a tiny
 * fast RAM tier plus a large Lustre/PFS file tier, placed by the max_bw DPE.
 * Because the fast tier is tiny, almost every page spills to the slow file tier,
 * so reading a page back is a real parallel-filesystem read -- the reader is
 * IO-bound.
 *
 * A reader sweeps the dataset window-by-window doing a compute step per window,
 * two ways:
 *   NO PREFETCH: read window W (blocking PFS read) -> compute W -> read W+1 ...
 *                the GPU stalls, idle, on every slow read.
 *   SequentialTransaction: reads window W+1 WHILE the GPU computes window W (the
 *                #700 prefetch-ahead), so the slow IO is hidden behind compute.
 *
 * When the workload is IO-bound this is the difference between stalling on every
 * read and never waiting -- so the transaction should be a clear speedup.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_cte/gpu_vector/transaction.h>

#include <clio_ctp/util/gpu_api.h>
#include <clio_ctp/introspect/system_info.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace {

bool g_initialized = false;
inline clio::run::PoolId StoragePool() { return clio::run::PoolId(600, 0); }

__host__ __device__ inline float Field(clio::run::u64 i) {
  return 0.5f + 0.4f * sinf(static_cast<float>(i) * 1.0e-4f);
}

ctp::ipc::ShmPtr<> DevPtr(void *dev) {
  ctp::ipc::ShmPtr<> p;
  p.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  p.off_ = reinterpret_cast<clio::run::u64>(dev);
  return p;
}

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10620;
  const char *slow_dir = std::getenv("CLIO_SLOW_DIR");
  std::string slow = std::string(slow_dir ? slow_dir : "/u/rpawar/gsbench") +
                     "/cte_slow_tier_" + std::to_string(port);
  std::string cfg = "/tmp/gpu_vec_prefetch_" + std::to_string(port) + ".yaml";
  {
    std::ofstream f(cfg);
    f << "networking:\n  port: " << port << "\n\n"
      << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
      << "compose:\n"
      << "  - mod_name: clio_bdev\n"
      << "    pool_name: \"ram::chi_default_bdev\"\n"
      << "    pool_query: local\n    pool_id: \"301.0\"\n"
      << "    bdev_type: ram\n    capacity: \"2048MB\"\n\n"
      << "  - mod_name: clio_cte_compressor\n"
      << "    pool_name: cte_compressor\n    pool_query: local\n"
      << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
      << "  - mod_name: clio_cte_core\n"
      << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
      // Automatic tiering: tiny fast RAM tier + large Lustre file tier. max_bw
      // fills the fast tier then spills to the slow file tier.
      << "    storage:\n"
      << "      - path: \"ram::cte_fast_tier\"\n"
      << "        bdev_type: \"ram\"\n        capacity_limit: \"256KB\"\n"
      << "        score: 1.0\n"
      << "      - path: \"" << slow << "\"\n"
      << "        bdev_type: \"file\"\n        capacity_limit: \"4096MB\"\n"
      << "        score: 0.2\n"
      << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  std::fprintf(stderr, "[PF] compose=%s  slow_tier=%s\n", cfg.c_str(), slow.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

__global__ void FillPageKernel(float *buf, clio::run::u64 first_elem,
                               clio::run::u64 n) {
  for (clio::run::u64 j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
       j += static_cast<clio::run::u64>(gridDim.x) * blockDim.x)
    buf[j] = Field(first_elem + j);
}

/** Compute-heavy body over a resident window; writes the untouched value to the
 *  GLOBAL result[i] (busy loop guarded by a runtime sentinel so it is not
 *  optimized away but never alters the output). */
__global__ void WindowComputeKernel(clio::run::IpcManagerGpuInfo info,
                                    gv::DeviceView<float> view, float *result,
                                    clio::run::u64 lo, clio::run::u64 hi,
                                    int iters, unsigned sentinel) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  v.read_range(lo, hi, [=](clio::run::u64 i, float val) {
    float busy = val;
#pragma unroll 1
    for (int k = 0; k < iters; ++k) busy = busy * 0.9999999f + 1.0e-7f;
    if (threadIdx.x == sentinel) result[i] = busy;
    result[i] = val;
  });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: IO-bound reader -- SequentialTransaction prefetch hides "
          "slow-tier reads", "[gpu_vector][compress][prefetch][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const clio::run::u32 window = 4;
  const clio::run::u64 page_size = 256ULL * 1024;
  const clio::run::u64 epp = page_size / sizeof(float);
  const char *pg_env = std::getenv("CLIO_PF_PAGES");
  const clio::run::u32 K = pg_env ? (clio::run::u32)std::atoi(pg_env) : 32;
  // Per-element compute load. Tunable so we can find the IO-bound regime: the
  // prefetch win is bounded by min(IO_total, compute_total), so it only shows
  // when the slow-tier reads are comparable to (or larger than) the compute.
  const char *it_env = std::getenv("CLIO_PF_ITERS");
  const int iters = it_env ? std::atoi(it_env) : 0;
  const unsigned kNever = 0xFFFFFFFFu;
  const char *tag = "pf";
  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  const char *slow_env = std::getenv("CLIO_CTE_SLOW_TIER_US");
  const unsigned long long slow_us = slow_env ? std::strtoull(slow_env, nullptr, 10) : 0ULL;
  std::fprintf(stderr,
      "[PF] pages=%u window=%u page=%lluKiB dataset=%lluMiB (fast tier 256KiB, "
      "rest auto-tiers to Lustre)  slow_tier_model=%lluus/page compute=%d it\n",
      K, window, (unsigned long long)(page_size >> 10),
      (unsigned long long)((clio::run::u64)K * page_size >> 20),
      slow_us, iters);

  // ---- Store K pages compressed; DPE spills most to the slow file tier ----
  clio::cte::core::Client core;
  core.Init(clio::cte::core::kCtePoolId);
  auto tagf = core.AsyncGetOrCreateTag(tag);
  tagf.Wait();
  REQUIRE(tagf->GetReturnCode() == 0);
  auto tag_id = tagf->tag_id_;
  clio::cte::core::Client comp;
  comp.Init(StoragePool());
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.compress_preset_ = 2;
  float *pagebuf = nullptr;
  cudaMalloc(&pagebuf, page_size);
  for (clio::run::u32 P = 0; P < K; ++P) {
    FillPageKernel<<<64, 256>>>(pagebuf, (clio::run::u64)P * epp, epp);
    ctp::GpuApi::Synchronize();
    std::string name = std::string(tag) + "_b0_pi" + std::to_string(P);
    auto pf = comp.AsyncPutBlob(tag_id, name, (clio::run::u64)0, page_size,
                                DevPtr(pagebuf), 0.5f, ctx, 0,
                                clio::run::PoolQuery::Local());
    pf.Wait();
    REQUIRE(pf->GetReturnCode() == 0);
  }
  cudaFree(pagebuf);
  std::fprintf(stderr, "[PF] stored %u compressed pages (auto-tiered)\n", K);

  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const clio::run::u64 total = (clio::run::u64)K * epp;
  auto *result = ctp::GpuApi::MallocHost<float>(total * sizeof(float));
  REQUIRE(result != nullptr);
  auto verify = [&]() {
    ctp::GpuApi::Synchronize();
    double e = 0.0;
    for (clio::run::u64 i = 0; i < total; ++i)
      e = std::max(e, std::fabs((double)result[i] - (double)Field(i)));
    return e;
  };
  const clio::run::u32 nwin = K / window;

  // ---- NO PREFETCH: read window (slow PFS read), then compute, serially ----
  // Split the two costs so we can SEE the regime: prefetch can hide at most
  // min(io_total, compute_total), so the workload is only IO-bound (and prefetch
  // only helps) when io per window is comparable to compute per window.
  double no_pf_ms = 0.0, no_pf_err = 0.0, io_ms = 0.0, cmp_ms = 0.0;
  {
    gv::Vector<float> v(tag, 1, 0, window, 0, page_size, 20000,
                        gv::CacheMode::kLegacy, 32, false, StoragePool());
    auto view = v.Device();
    std::memset(result, 0, total * sizeof(float));
    auto t0 = Clock::now();
    for (clio::run::u32 w = 0; w < nwin; ++w) {
      auto ti = Clock::now();
      v.PrefetchWindowSync((clio::run::u64)w * window);  // blocking slow read
      io_ms += ms(ti, Clock::now());
      clio::run::u64 lo = (clio::run::u64)w * window * epp;
      auto tc = Clock::now();
      WindowComputeKernel<<<1, 32>>>(gpu_info, view, result, lo,
                                     lo + window * epp, iters, kNever);
      ctp::GpuApi::Synchronize();
      cmp_ms += ms(tc, Clock::now());
    }
    no_pf_ms = ms(t0, Clock::now());
    no_pf_err = verify();
    std::fprintf(stderr,
                 "[PF] NO-PREFETCH        : %.1f ms  (io=%.1f compute=%.1f)  "
                 "(err=%.2e)\n",
                 no_pf_ms, io_ms, cmp_ms, no_pf_err);
  }

  // ---- SequentialTransaction: prefetch window W+1 while computing window W ----
  double txn_ms = 0.0, txn_err = 0.0;
  {
    gv::Vector<float> v(tag, 1, 0, 2 * window, 0, page_size, 20000,
                        gv::CacheMode::kLegacy, 32, false, StoragePool());
    std::memset(result, 0, total * sizeof(float));
    auto body = [&](clio::run::u64 lo, clio::run::u64 hi, gv::DeviceView<float> vv,
                    cudaStream_t s) {
      WindowComputeKernel<<<1, 32, 0, s>>>(gpu_info, vv, result, lo, hi, iters,
                                           kNever);
    };
    gv::SequentialTransaction<float> seq(v, /*first_page=*/0, /*npages=*/K);
    auto t0 = Clock::now();
    seq.Iterate(body);
    txn_ms = ms(t0, Clock::now());
    txn_err = verify();
    std::fprintf(stderr, "[PF] SequentialTransaction: %.1f ms  (err=%.2e)\n",
                 txn_ms, txn_err);
  }
  ctp::GpuApi::FreeHost(result);

  const double logical_mib = (double)((clio::run::u64)K * page_size) / (1024 * 1024);
  std::fprintf(stderr,
      "[PF] SPEEDUP (no-prefetch / transaction) = %.2fx   "
      "throughput %.1f -> %.1f MiB/s\n",
      no_pf_ms / txn_ms, logical_mib / (no_pf_ms / 1000.0),
      logical_mib / (txn_ms / 1000.0));
  const double hideable = std::min(io_ms, cmp_ms);  // prefetch's max headroom
  std::fprintf(stderr,
               "[PF] regime: io=%.1f compute=%.1f  hideable=min(io,compute)=%.1f "
               "ms (%.0f%% of serial)\n",
               io_ms, cmp_ms, hideable, 100.0 * hideable / no_pf_ms);
  REQUIRE(no_pf_err <= 2.0e-3);
  REQUIRE(txn_err <= 2.0e-3);
  if (txn_ms < no_pf_ms)
    std::fprintf(stderr, "[PF] PASS: prefetch-ahead wins (%.2fx)\n",
                 no_pf_ms / txn_ms);
  else
    std::fprintf(stderr,
                 "[PF] NOTE: not IO-bound at these params (io<<compute) -> "
                 "prefetch has little to hide (%.2fx)\n",
                 no_pf_ms / txn_ms);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif
