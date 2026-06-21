/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GPU producer-only stress test (CUDA / ROCm).
 *
 * Submits kNumTasks GpuSubmitTasks from the GPU concurrently — one per
 * grid block, with thread 0 of each block doing the Send + Wait. The
 * full pipeline runs:
 *
 *   per-block thread 0: CLIO_IPC->Send(task_fp[blockIdx.x])
 *                       -> IpcGpu2Cpu::ClientSend pushes onto the
 *                          per-device gpu2cpu_queue
 *                       -> CPU GPU worker pops, dispatches MOD_NAME::Runtime::GpuSubmit
 *                       -> RuntimeSend signals FUTURE_COMPLETE
 *                       -> kernel block returns from future.Wait()
 *
 * Suffix `_gpu.cc` matches the `*_gpu.cc` glob in
 * context-runtime/src/CMakeLists.txt and is compiled by NVCC (when
 * CLIO_CORE_ENABLE_CUDA=ON) or HIPCC (when CLIO_CORE_ENABLE_ROCM=ON).
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"
#include "test_gpu_kernel_stress_common.h"

#include <clio_ctp/util/gpu_api.h>

using TaskT = clio::run::MOD_NAME::GpuSubmitTask;

namespace {

/**
 * Stress kernel: one grid block per task slot. Thread 0 of each block
 * performs the Send + Wait; all other threads in the block return
 * immediately because IpcGpu2Cpu::ClientSend is gated on
 * `threadIdx.x == 0` (other lanes get an empty future).
 */
__global__ void ChiGpuKernelStress(clio::run::IpcManagerGpuInfo gpu_info,
                                    ctp::ipc::FullPtr<TaskT> *task_handles,
                                    clio::run::u32 num_tasks) {
  CLIO_GPU_INIT(gpu_info, /*ipc_ptr=*/nullptr);
  if (threadIdx.x != 0) return;
  clio::run::u32 slot = blockIdx.x;
  if (slot >= num_tasks) return;
  // Use the kernel-scope `g_ipc_manager_ptr` (declared by CLIO_GPU_INIT)
  // rather than the CLIO_IPC macro: NVCC compiles the kernel body in both
  // the host and device passes, and the host-pass expansion of CLIO_IPC
  // resolves the global `g_ipc_manager` symbol — which the macro shadows
  // with a `clio::run::gpu::IpcManager&` local. Reaching through the typed
  // pointer dodges that name collision and compiles cleanly in both passes.
  auto fp = task_handles[slot];
  auto fut = g_ipc_manager_ptr->Send(fp);
  fut.Wait();
  (void)g_ipc_manager;
}

}  // namespace

// NVCC parses every host TU in two passes; the host fixture below uses
// clio::run::IpcManager APIs (GetGpuIpcManager, AllocateAndRegisterGpuBackend,
// PlaceTaskSlots) that are CTP_IS_HOST-only. Gate the TEST_CASE body so
// only the host pass instantiates it. The __global__ kernel above remains
// visible to both passes — NVCC needs the host stub for the launch glue.
#if !CTP_IS_DEVICE_PASS

TEST_CASE("GPU producer-only stress: kernel submits N tasks",
          "[gpu2cpu][stress]") {
  using namespace chi_test_gpu_stress;
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  const clio::run::u32 gpu_id = 0;

  // 1. Allocate + register a pinned-host backend big enough for all slots.
  char *base = nullptr;
  ctp::ipc::AllocatorId alloc_id = ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost, kBackendBytes,
      &base);
  REQUIRE(!alloc_id.IsNull());
  REQUIRE(base != nullptr);

  // 2. Lay out N (Task + FutureShm) pairs.
  auto handles = PlaceTaskSlots(base, alloc_id, gpu_id);
  REQUIRE(handles.size() == kNumTasks);

  // 3. Stage the FullPtr<TaskT> array in pinned host so the kernel sees it.
  ctp::ipc::FullPtr<TaskT> *task_handle_dev =
      ctp::GpuApi::MallocHost<ctp::ipc::FullPtr<TaskT>>(kNumTasks);
  REQUIRE(task_handle_dev != nullptr);
  for (clio::run::u32 i = 0; i < kNumTasks; ++i) task_handle_dev[i] = handles[i];

  // 4. Launch: one block per task, 32 threads (warp scheduler == lane 0).
  clio::run::IpcManagerGpuInfo info = ipc->GetGpuIpcManager()->GetGpuInfo(gpu_id);
  REQUIRE(info.gpu2cpu_queue != nullptr);

  std::fprintf(stderr,
               "[STRESS] launching grid=%u blocks, 32 threads/block\n",
               kNumTasks);
  ChiGpuKernelStress<<<kNumTasks, 32>>>(info, task_handle_dev, kNumTasks);
  ctp::GpuApi::Synchronize();

  // 5. Verify every slot saw the chimod's formula.
  clio::run::u32 first_bad = VerifyResults(handles, gpu_id);
  REQUIRE(first_bad == kNumTasks);
  std::fprintf(stderr, "[STRESS] all %u tasks completed correctly\n",
               kNumTasks);

  // 6. Cleanup.
  ctp::GpuApi::FreeHost(reinterpret_cast<char *>(task_handle_dev));
  ipc->FreeGpuBackend(gpu_id, alloc_id);
}

#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()

#else  // CUDA/ROCm not enabled (or SYCL is)

int main() { return 0; }

#endif
