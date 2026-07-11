/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * SequentialTransaction (windowed prefetch) for the compressed GPU vector.
 *
 * The #700 direction: instead of faulting pages ON-DEVICE (which deadlocks under
 * GPU compression -- a spin-waiting fault kernel cannot co-run with the cuSZp
 * decompress kernel), the host PREFETCHES a window of pages ahead of use while
 * the GPU is idle, then the device kernel reads that resident window. A dataset
 * far larger than the HBM cache is swept window-by-window:
 *
 *     for each window W of the logical range:
 *        vec.PrefetchWindowSync(first_page(W));   // host decompress -> HBM, GPU idle
 *        read_kernel over W;                       // pages resident, no fault
 *        advance
 *
 * Here: K pages are stored compressed; the vector's HBM cache holds only
 * `window` pages (window << K), and we sweep all K correctly. No on-device
 * fault, no deadlock, and the resident footprint is one window regardless of
 * dataset size.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
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

/** Smooth, bounded field over the GLOBAL element index. */
__host__ __device__ inline float Field(clio::run::u64 i) {
  return 0.5f + 0.4f * sinf(static_cast<float>(i) * 1.0e-4f);
}

/** Zero-copy ShmPtr wrapping a raw device address. */
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
  int port = port_env ? std::atoi(port_env) : 10600;
  std::string cfg = "/tmp/gpu_vec_txn_" + std::to_string(port) + ".yaml";
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
      << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
      << "        bdev_type: \"ram\"\n        capacity_limit: \"2048MB\"\n"
      << "        score: 1.0\n"
      << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  std::fprintf(stderr, "[TXN] compose=%s\n", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

/** Fill a page buffer with Field() over global elements [page*epp, +epp). */
__global__ void FillPageKernel(float *buf, clio::run::u64 first_elem,
                               clio::run::u64 n) {
  for (clio::run::u64 j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
       j += static_cast<clio::run::u64>(gridDim.x) * blockDim.x) {
    buf[j] = Field(first_elem + j);
  }
}

/** Read a resident window [lo, hi) into result[i - lo]. */
__global__ void WindowReadKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceView<float> view, float *result,
                                 clio::run::u64 lo, clio::run::u64 hi) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  v.read_range(lo, hi, [result, lo](clio::run::u64 i, float val) {
    result[i - lo] = val;
  });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: SequentialTransaction windowed prefetch over a dataset "
          "larger than the HBM cache", "[gpu_vector][compress][txn][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const clio::run::u32 window = 4;                      // resident pages
  const clio::run::u64 page_size = 256ULL * 1024;       // 256 KiB (cuSZp-correct)
  const clio::run::u64 epp = page_size / sizeof(float);
  const clio::run::u32 K = 32;                          // total pages (8x window)
  const char *tag = "txn";
  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(0);

  std::fprintf(stderr,
      "[TXN] pages=%u window=%u page=%lluKiB  dataset=%lluMiB  resident=%lluMiB "
      "(%.0fx window)\n",
      K, window, (unsigned long long)(page_size >> 10),
      (unsigned long long)((clio::run::u64)K * page_size >> 20),
      (unsigned long long)((clio::run::u64)window * page_size >> 20),
      (double)K / window);

  // ---- Store K pages compressed under global names "txn_b0_pi<P>" ----
  // Direct compressor PutBlobs (host-issued, GPU idle) -- no vector eviction.
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
  std::fprintf(stderr, "[TXN] stored %u compressed pages\n", K);

  // ---- Sweep-read all K pages via a windowed transaction ----
  gv::Vector<float> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                        /*gpu_pages_per_block=*/window,
                        /*host_pages_per_block=*/0, page_size,
                        /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                        /*manager_threads_per_block=*/32,
                        /*allow_cold_miss_fault=*/false,
                        /*storage_pool_id=*/StoragePool());
  auto view = vec.Device();

  auto *result = ctp::GpuApi::MallocHost<float>(window * epp * sizeof(float));
  REQUIRE(result != nullptr);

  double max_abs_err = 0.0;
  clio::run::u64 checked = 0, first_bad = (clio::run::u64)K * epp;
  const clio::run::u32 nwin = K / window;
  for (clio::run::u32 w = 0; w < nwin; ++w) {
    clio::run::u64 first_page = (clio::run::u64)w * window;
    vec.PrefetchWindowSync(first_page);  // host decompress window into HBM, GPU idle
    clio::run::u64 lo = first_page * epp;
    clio::run::u64 hi = lo + (clio::run::u64)window * epp;
    std::memset(result, 0, window * epp * sizeof(float));
    // nblocks==1: read_range uses blockIdx.x as the block index, so launch a
    // single block (one warp) that reads the whole window range.
    WindowReadKernel<<<1, 32>>>(gpu_info, view, result, lo, hi);
    ctp::GpuApi::Synchronize();
    for (clio::run::u64 j = 0; j < (clio::run::u64)window * epp; ++j) {
      clio::run::u64 gi = lo + j;
      double e = std::fabs((double)result[j] - (double)Field(gi));
      if (e > max_abs_err) max_abs_err = e;
      if (e > 2.0e-3 && first_bad == (clio::run::u64)K * epp) first_bad = gi;
      ++checked;
    }
  }
  ctp::GpuApi::FreeHost(result);

  std::fprintf(stderr,
      "[TXN] swept %u windows, read %llu elems  max_abs_err=%.3e (eb=1e-3)\n",
      nwin, (unsigned long long)checked, max_abs_err);
  if (first_bad != (clio::run::u64)K * epp) {
    std::fprintf(stderr, "[TXN] first out-of-bound at %llu: exp %.6f\n",
                 (unsigned long long)first_bad, Field(first_bad));
  }
  REQUIRE(checked == (clio::run::u64)K * epp);
  REQUIRE(max_abs_err <= 2.0e-3);
  std::fprintf(stderr,
      "[TXN] PASS: %lluMiB dataset swept correctly with only %lluMiB resident "
      "(windowed prefetch, no on-device fault).\n",
      (unsigned long long)((clio::run::u64)K * page_size >> 20),
      (unsigned long long)((clio::run::u64)window * page_size >> 20));
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
