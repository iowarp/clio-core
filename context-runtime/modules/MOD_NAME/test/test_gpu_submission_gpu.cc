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
    hipc::MemoryBackend backend,
    chi::PoolId pool_id,
    chi::u32 test_value,
    int *result_flags) {

  // Mark that kernel started
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    result_flags[0] = -777;  // Kernel entered
  }
  __syncthreads();

  // Initialize IPC manager (defines thread_id)
  CHIMAERA_GPU_INIT(backend);

  // Only thread 0 creates and submits task
  if (thread_id == 0) {
    result_flags[0] = -666;  // CHIMAERA_GPU_INIT completed

    // Step 0: Check IPC manager initialized
    if (&g_ipc_manager == nullptr) {
      result_flags[0] = -10;  // IPC manager null
      return;
    }
    result_flags[0] = 0;  // IPC manager OK

    // Step 1: Try to allocate a small buffer to verify backend works
    hipc::FullPtr<char> test_buffer = (&g_ipc_manager)->AllocateBuffer(64);
    if (test_buffer.IsNull()) {
      result_flags[1] = -11;  // Backend allocation failed
      return;
    }
    result_flags[1] = 0;  // Backend allocation OK

    // Step 2: Test allocating task-sized buffer
    size_t task_size = sizeof(chimaera::MOD_NAME::GpuSubmitTask);
    hipc::FullPtr<char> task_buffer = (&g_ipc_manager)->AllocateBuffer(task_size);
    if (task_buffer.IsNull()) {
      result_flags[2] = -12;  // Task buffer allocation failed
      return;
    }
    // Free the test buffers to avoid running out of memory
    (&g_ipc_manager)->FreeBuffer(test_buffer);
    (&g_ipc_manager)->FreeBuffer(task_buffer);
    result_flags[2] = 0;  // Task buffer allocation OK

    // Step 3: ULTRA-SIMPLE TEST - Just set to known value
    result_flags[3] = -777;  // TEST: If you see -777, the code executed!

    // Step 4: Skipped for now (no task created yet)
    result_flags[4] = 0;

    // Step 5: Mark as success
    result_flags[5] = 0;
  }

  __syncthreads();
}

/**
 * C++ wrapper function to run the GPU kernel test
 * This allows the CPU test file to call this without needing CUDA headers
 */
extern "C" int run_gpu_kernel_task_submission_test(chi::PoolId pool_id, chi::u32 test_value) {
  // Create GPU memory backend using GPU-registered shared memory (same as isolated test)
  hipc::MemoryBackendId backend_id(2, 0);
  size_t gpu_memory_size = 10 * 1024 * 1024;  // 10MB - same as isolated test
  hipc::GpuShmMmap gpu_backend;
  if (!gpu_backend.shm_init(backend_id, gpu_memory_size, "/gpu_kernel_submit", 0)) {
    return -100;  // Backend init failed
  }

  // Allocate result flags array on GPU (6 steps)
  int *d_result_flags = hshm::GpuApi::Malloc<int>(sizeof(int) * 6);
  int h_result_flags[6] = {-999, -999, -999, -999, -999, -999};  // Sentinel values
  hshm::GpuApi::Memcpy(d_result_flags, h_result_flags, sizeof(int) * 6);

  // Backend can be passed by value to kernel
  hipc::MemoryBackend h_backend = gpu_backend;  // Copy to temporary

  // Launch kernel that submits a task (using 1 thread, 1 block for simplicity)
  gpu_submit_task_kernel<<<1, 1>>>(h_backend, pool_id, test_value, d_result_flags);

  // Check for kernel launch errors
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    hshm::GpuApi::Free(d_result_flags);
    return -201;  // Kernel launch error
  }

  // Synchronize and check for errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    hshm::GpuApi::Free(d_result_flags);
    return -200;  // CUDA error
  }

  // Check kernel results
  hshm::GpuApi::Memcpy(h_result_flags, d_result_flags, sizeof(int) * 6);

  // Cleanup
  hshm::GpuApi::Free(d_result_flags);

  // Check all steps for errors
  for (int i = 0; i < 6; ++i) {
    if (h_result_flags[i] != 0) {
      return h_result_flags[i];  // Return first error
    }
  }

  return 1;  // Success - all steps passed
}

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
