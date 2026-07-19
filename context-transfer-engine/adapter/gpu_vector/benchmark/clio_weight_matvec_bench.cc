/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Weight-application transaction demo / benchmark.
 *
 * Exercises `gpu_vector_matvec.h` — the CTE-vector analogue of llama.cpp's
 * `mul_mat_vec` (ggml/src/ggml-cuda/mmvq.cu). A row-major weight matrix
 * W[n_rows, n_cols] is stored in a Vector<float> (paged over HBM -> pinned
 * DRAM -> CTE blob storage), then y = W * x is computed by streaming each
 * weight row through `dev::vector::read_range`. The sequential read_range
 * transaction self-describes its access pattern via rescore lookahead, so the
 * async cache manager prefetches upcoming weight pages ahead of the dot
 * product that consumes them.
 *
 * Row/block mapping (see gpu_vector_matvec.h): n_rows == blocks * rows_per_block
 * and block b owns the contiguous element stripe
 *     [ b*rows_per_block*n_cols , (b+1)*rows_per_block*n_cols ).
 *
 * Verification: W[i] = (i % 7), x[c] = 1  =>  y[row] = sum_c ((row*n_cols+c)%7).
 *
 *   --blocks N              Cache blocks == parallel row stripes (default 8)
 *   --rows-per-block R      Output rows per block (default 16)
 *   --n-cols K              Reduction dim / weights per row (default 1024)
 *   --page-size BYTES       Page size (default 65536)
 *   --pages-per-block P     HBM cache slots per block (default 4)
 *   --host-pages-per-block P  DRAM cache slots per block (default 0 = legacy)
 *   --cache-period-us N     Manager tick (default 0)
 *   --bdev-capacity-mib N   kRam bdev capacity (default = 4x weights, min 64)
 *   --gpu-id N              GPU index (default 0)
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_info.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_cte/gpu_vector/gpu_vector_matvec.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

namespace {

struct Opts {
  clio::run::u32 nblocks = 8;
  clio::run::u32 rows_per_block = 16;
  clio::run::u64 n_cols = 1024;
  clio::run::u64 page_size = 1ULL << 16;  // 64 KiB
  clio::run::u32 pages_per_block = 4;
  clio::run::u32 host_pages_per_block = 0;
  clio::run::u32 cache_period_us = 0;
  clio::run::u64 bdev_capacity_mib = 0;
  clio::run::u32 gpu_id = 0;
};

#if !CTP_IS_DEVICE_PASS
void PrintUsage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s [--blocks N] [--rows-per-block R] [--n-cols K]\n"
               "          [--page-size BYTES] [--pages-per-block P]\n"
               "          [--host-pages-per-block P] [--cache-period-us N]\n"
               "          [--bdev-capacity-mib N] [--gpu-id N]\n",
               prog);
}

bool ParseOpts(int argc, char *argv[], Opts &o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") { PrintUsage(argv[0]); std::exit(0); }
    else if (a == "--blocks") o.nblocks = std::atoi(next("--blocks"));
    else if (a == "--rows-per-block")
      o.rows_per_block = std::atoi(next("--rows-per-block"));
    else if (a == "--n-cols")
      o.n_cols = std::strtoull(next("--n-cols"), nullptr, 10);
    else if (a == "--page-size")
      o.page_size = std::strtoull(next("--page-size"), nullptr, 10);
    else if (a == "--pages-per-block")
      o.pages_per_block = std::atoi(next("--pages-per-block"));
    else if (a == "--host-pages-per-block")
      o.host_pages_per_block = std::atoi(next("--host-pages-per-block"));
    else if (a == "--cache-period-us")
      o.cache_period_us = std::atoi(next("--cache-period-us"));
    else if (a == "--bdev-capacity-mib")
      o.bdev_capacity_mib = std::strtoull(next("--bdev-capacity-mib"), nullptr, 10);
    else if (a == "--gpu-id") o.gpu_id = std::atoi(next("--gpu-id"));
    else {
      std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}
#endif  // !CTP_IS_DEVICE_PASS

}  // namespace

/** Populate the weight matrix in the Vector: W[i] = i % 7. Uses write_range
 *  so the per-page dirty-range bookkeeping happens once per page and the
 *  inner loop is a warp-cooperative stride-1 store — the same streaming
 *  transaction as the read side, just producing instead of consuming. */
__global__ void WriteWeightsKernel(clio::run::IpcManagerGpuInfo info,
                                    gv::DeviceView<float> view,
                                    clio::run::u64 elems_per_block) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * elems_per_block;
  clio::run::u64 hi = lo + elems_per_block;
  v.write_range(lo, hi, [] (clio::run::u64 i) {
    return static_cast<float>(i % 7);
  });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

namespace {

void EnsureInit(clio::run::u64 bdev_capacity_bytes) {
  std::fprintf(stderr, "[INIT] Starting Clio server\n");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    std::fprintf(stderr, "[INIT] CLIO_INIT failed\n");
    std::exit(2);
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[INIT] CLIO_CTE_CLIENT_INIT failed\n");
    std::exit(2);
  }
  auto *cte_client = CLIO_CTE_CLIENT;
  cte_client->Init(clio::cte::core::kCtePoolId);
  clio::cte::core::CreateParams params;
  auto create_task = cte_client->AsyncCreate(
      clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
      clio::cte::core::kCtePoolId, params);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] CTE pool create failed rc=%u\n",
                 create_task->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);

  clio::run::PoolId bdev_pool_id(951, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("weight_matvec_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, bdev_capacity_bytes);
  bdev_create.Wait();
  if (bdev_create->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] bdev create failed rc=%u\n",
                 bdev_create->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "weight_matvec_ram", clio::run::bdev::BdevType::kRam,
      bdev_capacity_bytes, clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  if (reg_task->GetReturnCode() != 0) {
    std::fprintf(stderr, "[INIT] RegisterTarget failed rc=%u\n",
                 reg_task->GetReturnCode());
    std::exit(2);
  }
  std::this_thread::sleep_for(50ms);
}

}  // namespace

int main(int argc, char *argv[]) {
  Opts o;
  if (!ParseOpts(argc, argv, o)) return 2;
  if (o.nblocks == 0 || o.rows_per_block == 0 || o.n_cols == 0 ||
      o.page_size % sizeof(float) != 0) {
    std::fprintf(stderr, "invalid geometry (page_size must be a multiple of 4)\n");
    return 2;
  }

  const clio::run::u64 n_rows =
      static_cast<clio::run::u64>(o.nblocks) * o.rows_per_block;
  const clio::run::u64 elems_per_block =
      static_cast<clio::run::u64>(o.rows_per_block) * o.n_cols;
  const clio::run::u64 total_elems = n_rows * o.n_cols;
  const clio::run::u64 total_bytes = total_elems * sizeof(float);

  clio::run::u64 bdev_capacity_bytes =
      o.bdev_capacity_mib > 0 ? o.bdev_capacity_mib * (1ULL << 20)
                              : std::max<clio::run::u64>(64ULL << 20, total_bytes * 4);

  std::fprintf(stderr,
               "[MATVEC] blocks=%u rows/block=%u n_cols=%llu -> n_rows=%llu\n"
               "[MATVEC] weights=%llu B page=%llu B pages/block=%u host_pages=%u\n",
               o.nblocks, o.rows_per_block, (unsigned long long)o.n_cols,
               (unsigned long long)n_rows, (unsigned long long)total_bytes,
               (unsigned long long)o.page_size, o.pages_per_block,
               o.host_pages_per_block);

  EnsureInit(bdev_capacity_bytes);
  auto *ipc = CLIO_CPU_IPC;
  clio::run::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->GetGpuInfo(o.gpu_id);

  // Async-only reads require prefetch hints in flight; enabling cold-miss
  // fault makes the read path self-sufficient regardless of tier config so
  // the demo is correct whether or not the weights fit in cache.
  gv::Vector<float> W("weight_matvec", o.nblocks, o.gpu_id, o.pages_per_block,
                      o.host_pages_per_block, o.page_size, o.cache_period_us,
                      o.host_pages_per_block > 0 ? gv::CacheMode::kAsync
                                                 : gv::CacheMode::kLegacy,
                      /*manager_threads_per_block=*/32,
                      /*allow_cold_miss_fault=*/true);
  auto view = W.Device();

  cudaStream_t stream = nullptr;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // 1. Fill the weight matrix and flush it down to storage.
  WriteWeightsKernel<<<o.nblocks, 32, 0, stream>>>(gpu_info, view,
                                                   elems_per_block);
  cudaStreamSynchronize(stream);
  W.FlushAllSync();

  // 2. Activation x[c] = 1 (device-resident), output y (device).
  std::vector<float> hx(o.n_cols, 1.0f);
  float *dx = nullptr;
  float *dy = nullptr;
  cudaMalloc(&dx, o.n_cols * sizeof(float));
  cudaMalloc(&dy, n_rows * sizeof(float));
  cudaMemcpy(dx, hx.data(), o.n_cols * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dy, 0, n_rows * sizeof(float));

  // 3. Run the weight-application transaction: y = W * x.
  auto t0 = std::chrono::steady_clock::now();
  gv::LaunchWeightMatVec(W, o.nblocks, o.gpu_id, dx, dy, o.n_cols,
                         o.rows_per_block, stream);
  cudaStreamSynchronize(stream);
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();

  // 4. Verify against the closed form y[row] = sum_c ((row*n_cols + c) % 7).
  std::vector<float> hy(n_rows, 0.0f);
  cudaMemcpy(hy.data(), dy, n_rows * sizeof(float), cudaMemcpyDeviceToHost);

  clio::run::u64 mismatches = 0;
  for (clio::run::u64 row = 0; row < n_rows; ++row) {
    double expect = 0.0;
    for (clio::run::u64 c = 0; c < o.n_cols; ++c) {
      expect += static_cast<double>((row * o.n_cols + c) % 7);
    }
    if (static_cast<double>(hy[row]) != expect) {
      if (mismatches < 8) {
        std::fprintf(stderr, "[MATVEC] mismatch row %llu: got %.1f want %.1f\n",
                     (unsigned long long)row, hy[row], expect);
      }
      ++mismatches;
    }
  }

  std::fprintf(stderr, "[MATVEC] matvec kernel: %lld us over %llu rows\n",
               (long long)us, (unsigned long long)n_rows);
  if (mismatches == 0) {
    std::fprintf(stderr, "[MATVEC] PASS: all %llu rows correct\n",
                 (unsigned long long)n_rows);
  } else {
    std::fprintf(stderr, "[MATVEC] FAIL: %llu / %llu rows wrong\n",
                 (unsigned long long)mismatches, (unsigned long long)n_rows);
  }

  cudaFree(dx);
  cudaFree(dy);
  cudaStreamDestroy(stream);
  return mismatches == 0 ? 0 : 1;
}

#endif  // !CTP_IS_DEVICE_PASS

#else  // no CUDA/ROCm

int main() { return 0; }

#endif
