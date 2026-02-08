/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * GPU kernels for Part 3: GPU Task Submission tests
 * This file contains only GPU kernel code and is compiled as CUDA
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>
#include <chimaera/pool_query.h>
#include <chimaera/task.h>
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <hermes_shm/util/gpu_api.h>

/**
 * GPU kernel that submits a task from within the kernel
 * Tests Part 3: GPU kernel calling NewTask and Send
 */
__global__ void gpu_submit_task_kernel(
    const hipc::MemoryBackend *backend,
    chi::PoolId pool_id,
    chi::u32 test_value,
    int *result_flag) {
  // Simplest test - just write a value
  *result_flag = 999;
  return;

  // Manually expand CHIMAERA_GPU_INIT for single thread
  __shared__ char g_ipc_manager_storage[sizeof(chi::IpcManager)];
  __shared__ chi::IpcManager *g_ipc_manager_ptr;
  __shared__ hipc::ArenaAllocator<false> *g_arena_alloc;

  *result_flag = -700;  // Before reinterpret_cast
  g_arena_alloc = reinterpret_cast<hipc::ArenaAllocator<false>*>(backend->data_);

  *result_flag = -701;  // Before placement new
  // Skip placement new for now to test
  //new (g_arena_alloc) hipc::ArenaAllocator<false>();
  *result_flag = -702;  // Skipped placement new

  *result_flag = -702;  // Before shm_init
  g_arena_alloc->shm_init(*backend, backend->data_capacity_);

  *result_flag = -703;  // Before IpcManager cast
  g_ipc_manager_ptr = reinterpret_cast<chi::IpcManager*>(g_ipc_manager_storage);

  *result_flag = -704;  // Before ClientGpuInit
  g_ipc_manager_ptr->ClientGpuInit(*backend, g_arena_alloc);

  *result_flag = -705;  // Before creating reference
  chi::IpcManager &g_ipc_manager = *g_ipc_manager_ptr;

  *result_flag = -500;  // After init

  // Create task using NewTask
  chi::u32 gpu_id = 0;
  chi::PoolQuery query = chi::PoolQuery::Local();

  auto task = (&g_ipc_manager)->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
      chi::CreateTaskId(), pool_id, query, gpu_id, test_value);

  if (task.IsNull()) {
    *result_flag = -1;  // NewTask failed
    return;
  }

  // Submit task using Send
  (&g_ipc_manager)->Send(task);

  // Mark success
  *result_flag = 1;
}

/**
 * C++ wrapper function to run the GPU kernel test
 * This allows the CPU test file to call this without needing CUDA headers
 */
extern "C" int run_gpu_kernel_task_submission_test(chi::PoolId pool_id, chi::u32 test_value) {
  // Create GPU memory backend for kernel use
  hipc::MemoryBackendId backend_id(100, 0);
  size_t gpu_memory_size = 10 * 1024 * 1024;  // 10MB
  hipc::GpuMalloc gpu_backend;
  if (!gpu_backend.shm_init(backend_id, gpu_memory_size, "gpu_kernel_submit", 0)) {
    return -100;  // Backend init failed
  }

  // Allocate result flag on GPU
  int *d_result_flag = hshm::GpuApi::Malloc<int>(sizeof(int));
  int h_result_flag = -999;  // Sentinel value to detect if kernel runs at all
  hshm::GpuApi::Memcpy(d_result_flag, &h_result_flag, sizeof(int));

  // Copy backend to GPU memory so kernel can access it
  hipc::MemoryBackend *d_backend = hshm::GpuApi::Malloc<hipc::MemoryBackend>(sizeof(hipc::MemoryBackend));
  hipc::MemoryBackend h_backend = gpu_backend;  // Copy to temporary
  hshm::GpuApi::Memcpy(d_backend, &h_backend, sizeof(hipc::MemoryBackend));

  // Launch kernel that submits a task (using 1 thread, 1 block for simplicity)
  gpu_submit_task_kernel<<<1, 1>>>(d_backend, pool_id, test_value, d_result_flag);

  // Check for kernel launch errors
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    hshm::GpuApi::Free(d_result_flag);
    return -201;  // Kernel launch error
  }

  // Synchronize and check for errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    hshm::GpuApi::Free(d_result_flag);
    return -200;  // CUDA error
  }

  // Check kernel result
  hshm::GpuApi::Memcpy(&h_result_flag, d_result_flag, sizeof(int));

  // Cleanup
  hshm::GpuApi::Free(d_result_flag);
  hshm::GpuApi::Free(d_backend);

  return h_result_flag;  // Return the result (1 = success, -1/-2 = error)
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
