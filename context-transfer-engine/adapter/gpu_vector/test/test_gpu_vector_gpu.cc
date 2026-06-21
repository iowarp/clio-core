/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GPU vector unit test (CUDA / ROCm).
 *
 * 1. Bring up the CLIO Runtime server + CTE core pool.
 * 2. Create a clio::cte::gpu_vector::Vector<uint32_t> with 4 blocks,
 *    4 pages per block, 4 KiB pages.
 * 3. Launch a write kernel that does v[i] = i*2 over a striped pattern
 *    that crosses page boundaries.
 * 4. FlushAllSync().
 * 5. Launch a read kernel that copies v[i] into a result array, then
 *    verify on the host.
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
#include <thread>

using namespace std::chrono_literals;
using TaskT = clio::run::admin::CreateTask;

namespace {

bool g_initialized = false;

/** Bring up CLIO Runtime + CTE core pool exactly once. Body gated to host
 *  pass: the device pass parses the function but never runs it, and
 *  cte_client->AsyncCreate is CTP_IS_HOST-only. */
void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  std::fprintf(stderr, "[INIT] Starting Clio server (gpu_vector test)\n");
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

  // Register a kRam bdev target so PutBlob/GetBlob have somewhere to
  // land. Required for eviction tests since dirty pages are flushed
  // through the bdev runtime.
  const clio::run::u64 kRamCapacity = 4ULL << 30;  // 4 GiB
  clio::run::PoolId bdev_pool_id(950, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), std::string("gpu_vector_ram"),
      bdev_pool_id, clio::run::bdev::BdevType::kRam, kRamCapacity);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);
  auto reg_task = cte_client->AsyncRegisterTarget(
      "gpu_vector_ram", clio::run::bdev::BdevType::kRam, kRamCapacity,
      clio::run::PoolQuery::Local(), bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);
  std::this_thread::sleep_for(50ms);

  g_initialized = true;
#endif
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

/** Write v[i] = i*2 for the first total elements. */
__global__ void GpuVectorWriteKernel(clio::run::IpcManagerGpuInfo info,
                                      gv::DeviceView<clio::run::u32> view,
                                      clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<clio::run::u32> v(view, g_ipc_manager_ptr);
  clio::run::u64 stripe = (total + gridDim.x - 1) / gridDim.x;
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * stripe;
  clio::run::u64 hi = lo + stripe;
  if (hi > total) hi = total;
  v.write_range(lo, hi, [] (clio::run::u64 i) {
    return static_cast<clio::run::u32>(i * 2u);
  });
  (void)g_ipc_manager;
}

__global__ void GpuVectorReadKernel(clio::run::IpcManagerGpuInfo info,
                                     gv::DeviceView<clio::run::u32> view,
                                     clio::run::u32 *result, clio::run::u64 total) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<clio::run::u32> v(view, g_ipc_manager_ptr);
  clio::run::u64 stripe = (total + gridDim.x - 1) / gridDim.x;
  clio::run::u64 lo = static_cast<clio::run::u64>(blockIdx.x) * stripe;
  clio::run::u64 hi = lo + stripe;
  if (hi > total) hi = total;
  v.read_range(lo, hi, [result] (clio::run::u64 i, clio::run::u32 val) {
    result[i] = val;
  });
  (void)g_ipc_manager;
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: write then read round-trip",
          "[gpu_vector][cte][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  const clio::run::u32 nblocks = 4;
  const clio::run::u32 pages_per_block = 4;
  const clio::run::u64 page_size_bytes = 4096;

  // Legacy mode (host_pages_per_block=0); enable cold-miss fault since
  // the test kernel doesn't push lookahead hints ahead of access.
  gv::Vector<clio::run::u32> vec("gpu_vector_smoke", nblocks, /*gpu_id=*/0,
                            pages_per_block,
                            /*host_pages_per_block=*/0,
                            page_size_bytes,
                            /*cache_period_us=*/20000,
                            gv::CacheMode::kLegacy,
                            /*manager_threads_per_block=*/32,
                            /*allow_cold_miss_fault=*/true);

  clio::run::u64 elements_per_page = page_size_bytes / sizeof(clio::run::u32);
  clio::run::u64 total = static_cast<clio::run::u64>(nblocks) * pages_per_block *
                    elements_per_page;
  std::fprintf(stderr, "[GPUVEC] total=%llu elements\n",
               (unsigned long long)total);

  auto view = vec.Device();
  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  // ctp::GpuApi::MallocHost takes BYTES (not elements).
  auto *result = ctp::GpuApi::MallocHost<clio::run::u32>(
      total * sizeof(clio::run::u32));
  REQUIRE(result != nullptr);
  std::memset(result, 0, total * sizeof(clio::run::u32));

  // 1. Write v[i] = i*2 across all blocks.
  GpuVectorWriteKernel<<<nblocks, 32>>>(gpu_info, view, total);
  ctp::GpuApi::Synchronize();

  // 2. Drain in-flight puts before tearing down the cache.
  vec.FlushAllSync();

  // 3. Read back. Reads will fault any pages that the cache evicted.
  GpuVectorReadKernel<<<nblocks, 32>>>(gpu_info, view, result, total);
  ctp::GpuApi::Synchronize();

  // 4. Verify.
  for (clio::run::u64 i = 0; i < total; ++i) {
    if (result[i] != static_cast<clio::run::u32>(i * 2u)) {
      std::fprintf(stderr,
                   "[GPUVEC] mismatch at %llu: got %u expected %u\n",
                   (unsigned long long)i, result[i],
                   static_cast<clio::run::u32>(i * 2u));
      REQUIRE(result[i] == static_cast<clio::run::u32>(i * 2u));
    }
  }
  std::fprintf(stderr, "[GPUVEC] OK %llu / %llu\n",
               (unsigned long long)total, (unsigned long long)total);

  ctp::GpuApi::FreeHost(result);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else

int main() { return 0; }

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
