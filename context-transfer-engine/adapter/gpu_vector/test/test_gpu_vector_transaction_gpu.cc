/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * SequentialTransaction / PseudoRandomTransaction correctness (issue #700).
 *
 * The Transaction API sweeps a dataset larger than the HBM cache one window at a
 * time, PREFETCHING the next window (into the other of two buffers) while the
 * body reads the current one -- host-orchestrated, so no on-device fault. This
 * test checks correctness of both access patterns; the IO-bound speedup that the
 * prefetch-ahead buys is measured in test_gpu_vector_prefetch.
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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

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
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  std::fprintf(stderr, "[TXN] compose=%s\n", cfg.c_str());
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

/** Read a resident window [lo, hi) into a GLOBAL result buffer (result[i]). */
__global__ void WindowReadKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceView<float> view, float *result,
                                 clio::run::u64 lo, clio::run::u64 hi) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  v.read_range(lo, hi, [result](clio::run::u64 i, float val) { result[i] = val; });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: Sequential + PseudoRandom transactions round-trip a "
          "dataset larger than the HBM cache", "[gpu_vector][compress][txn]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const clio::run::u32 window = 4;             // pages per window
  const clio::run::u64 page_size = 256ULL * 1024;
  const clio::run::u64 epp = page_size / sizeof(float);
  const clio::run::u32 K = 32;                 // total pages
  const char *tag = "txn";
  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  std::fprintf(stderr,
      "[TXN] pages=%u window=%u page=%lluKiB  dataset=%lluMiB  resident=%lluMiB\n",
      K, window, (unsigned long long)(page_size >> 10),
      (unsigned long long)((clio::run::u64)K * page_size >> 20),
      (unsigned long long)((clio::run::u64)2 * window * page_size >> 20));

  // ---- Store K pages compressed under global names "txn_b0_pi<P>" ----
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

  // Reader vector: 2*window slots (double-buffered for prefetch-ahead).
  gv::Vector<float> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                        /*gpu_pages_per_block=*/2 * window,
                        /*host_pages_per_block=*/0, page_size,
                        /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                        /*manager_threads_per_block=*/32,
                        /*allow_cold_miss_fault=*/false,
                        /*storage_pool_id=*/StoragePool());

  const clio::run::u64 total = (clio::run::u64)K * epp;
  auto *result = ctp::GpuApi::MallocHost<float>(total * sizeof(float));
  REQUIRE(result != nullptr);

  // Body: read the current window into the global result buffer (async on the
  // transaction's stream; the transaction overlaps the next window's prefetch).
  auto body = [&](clio::run::u64 lo, clio::run::u64 hi, gv::DeviceView<float> v,
                  cudaStream_t s) {
    WindowReadKernel<<<1, 32, 0, s>>>(gpu_info, v, result, lo, hi);
  };
  auto check = [&](const char *what) {
    ctp::GpuApi::Synchronize();
    double max_err = 0.0;
    for (clio::run::u64 i = 0; i < total; ++i)
      max_err = std::max(max_err, std::fabs((double)result[i] - (double)Field(i)));
    std::fprintf(stderr, "[TXN] %s: read %llu elems  max_abs_err=%.3e (eb=1e-3)\n",
                 what, (unsigned long long)total, max_err);
    REQUIRE(max_err <= 2.0e-3);
  };

  // 1) SequentialTransaction (ascending sweep, prefetch-ahead).
  std::memset(result, 0, total * sizeof(float));
  gv::SequentialTransaction<float> seq(vec, /*first_page=*/0, /*npages=*/K);
  seq.Iterate(body);
  check("Sequential");

  // 2) PseudoRandomTransaction (same windows, shuffled order).
  std::vector<clio::run::u64> starts;
  for (clio::run::u32 w = 0; w < K / window; ++w)
    starts.push_back((clio::run::u64)w * window);
  std::vector<clio::run::u64> shuffled;
  for (size_t i = 0; i < starts.size(); i += 2) shuffled.push_back(starts[i]);
  for (size_t i = 1; i < starts.size(); i += 2) shuffled.push_back(starts[i]);
  std::memset(result, 0, total * sizeof(float));
  gv::PseudoRandomTransaction<float> rnd(vec, shuffled);
  rnd.Iterate(body);
  check("PseudoRandom");

  ctp::GpuApi::FreeHost(result);
  std::fprintf(stderr,
      "[TXN] PASS: %lluMiB dataset swept via Sequential + PseudoRandom "
      "transactions with only %lluMiB resident (prefetch-ahead, no on-device "
      "fault).\n",
      (unsigned long long)((clio::run::u64)K * page_size >> 20),
      (unsigned long long)((clio::run::u64)2 * window * page_size >> 20));
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
