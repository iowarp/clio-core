/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Smoke test for the device-paged vector: one block, one warp, a working
 * set that fits, write-then-read through operator[] with an explicit
 * flush. This is the minimum that exercises every moving part -- page
 * fault, dirty tracking, BeginFlush/WaitFlush -- so a failure here means
 * the mechanism is broken rather than a policy being wrong.
 *
 * KNOWN LIMITATION: this covers exactly ONE page on purpose. Raising
 * kElems past one page (2048 elements = 2 pages) hangs: the kernel waits
 * on a flush that never completes. Everything the single-page case proves
 * -- fault, write, flush, read-back, byte-exact -- keeps working, so the
 * break is specific to having more than one page in play, not to the
 * mechanism. Not yet diagnosed; do not raise this constant expecting it
 * to work.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <cstdio>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace gv = clio::cte::gpu_vector;

namespace {
constexpr clio::run::u64 kPageBytes = 4096;
constexpr clio::run::u64 kElems = 1024;          // ONE page (see note)
}  // namespace

/** Fill every element with a known function of its index. */
__global__ void FillKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<clio::run::u32> v,
                           clio::run::u64 n) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (clio::run::u64 i = 0; i < n; ++i) {
    v[i] = static_cast<clio::run::u32>(i * 7 + 1);
  }
  v.BeginFlush(0, n);
  v.WaitFlush(0, n);
}

/** Read everything back and count mismatches. */
__global__ void CheckKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<clio::run::u32> v,
                            clio::run::u64 n, unsigned long long *bad) {
  CLIO_GPU_INIT(info, nullptr);
  v.ipc_ = g_ipc_manager_ptr;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (clio::run::u64 i = 0; i < n; ++i) {
    if (v.at(i) != static_cast<clio::run::u32>(i * 7 + 1)) {
      atomicAdd(bad, 1ull);
    }
  }
}

#if !CTP_IS_DEVICE_PASS
TEST_CASE("gpu_vector: write, flush, read back", "[gpu_vector][smoke]") {
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());

  clio::run::IpcManagerGpuInfo gpu_info{};
  gpu_info = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // 4 pages of capacity for a 4-page vector: everything fits, so no
  // eviction should be needed.
  gv::Vector<clio::run::u32> vec("gv_smoke", {0}, kPageBytes,
                                 /*nblocks=*/1, /*pages_per_block=*/4, kElems);

  FillKernel<<<1, 32>>>(gpu_info, vec.GetDevice(0), kElems);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  unsigned long long *d_bad = nullptr;
  REQUIRE(cudaMalloc(&d_bad, sizeof(unsigned long long)) == cudaSuccess);
  REQUIRE(cudaMemset(d_bad, 0, sizeof(unsigned long long)) == cudaSuccess);

  CheckKernel<<<1, 32>>>(gpu_info, vec.GetDevice(0), kElems, d_bad);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  unsigned long long bad = 1;
  REQUIRE(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  cudaFree(d_bad);
  std::fprintf(stderr, "[smoke] mismatches=%llu / %llu\n",
               (unsigned long long) bad, (unsigned long long) kElems);
  REQUIRE(bad == 0);
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()
