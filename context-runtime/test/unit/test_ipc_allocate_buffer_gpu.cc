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
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Unit tests for GPU memory allocation in CHI_IPC
 * Tests GPU kernel memory allocation using BuddyAllocator
 * Only compiles when CUDA or HIP is enabled
 */

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/local_task_archives.h>
#include <chimaera/pool_query.h>
#include <chimaera/task.h>
#include <hermes_shm/memory/backend/gpu_malloc.h>
#include <hermes_shm/util/gpu_api.h>

#include <cstring>
#include <memory>
#include <vector>

#include "../simple_test.h"

namespace {

/**
 * Minimal GPU kernel to test basic execution (no CHIMAERA_GPU_INIT)
 */
__global__ void test_gpu_minimal_kernel(int *results) {
  int thread_id = threadIdx.x;
  results[thread_id] = thread_id + 100;  // Write a test value
}

/**
 * Test writing to backend.data_ without shm_init
 */
__global__ void test_gpu_backend_write_kernel(const hipc::MemoryBackend backend,
                                              int *results) {
  int thread_id = threadIdx.x;

  // Try to write a simple value to backend.data_
  if (thread_id == 0 && backend.data_ != nullptr) {
    char *test_ptr = backend.data_;
    test_ptr[0] = 42;                          // Simple write test
    results[0] = (test_ptr[0] == 42) ? 0 : 1;  // Verify
  }

  if (thread_id != 0) {
    results[thread_id] = 0;  // Other threads just pass
  }
}

/**
 * Test placement new on ArenaAllocator without shm_init
 */
__global__ void test_gpu_placement_new_kernel(const hipc::MemoryBackend backend,
                                              int *results) {
  int thread_id = threadIdx.x;

  if (thread_id == 0 && backend.data_ != nullptr) {
    // Try placement new without calling shm_init
    hipc::ArenaAllocator<false> *alloc =
        reinterpret_cast<hipc::ArenaAllocator<false> *>(backend.data_);
    new (alloc) hipc::ArenaAllocator<false>();
    results[0] = 0;  // Success if we got here
  } else {
    results[thread_id] = 0;
  }
}

/**
 * Test placement new + shm_init
 */
__global__ void test_gpu_shm_init_kernel(const hipc::MemoryBackend backend,
                                         int *results) {
  int thread_id = threadIdx.x;

  if (thread_id == 0 && backend.data_ != nullptr) {
    hipc::ArenaAllocator<false> *alloc =
        reinterpret_cast<hipc::ArenaAllocator<false> *>(backend.data_);
    new (alloc) hipc::ArenaAllocator<false>();
    results[0] = 1;  // Mark that we got past placement new
    alloc->shm_init(backend, backend.data_capacity_);
    results[0] = 0;  // Success if we got past shm_init
  } else {
    results[thread_id] = 0;
  }
}

/**
 * Test everything except IpcManager construction
 */
__global__ void test_gpu_alloc_no_ipc_kernel(const hipc::MemoryBackend backend,
                                             int *results) {
  __shared__ hipc::ArenaAllocator<false> *g_arena_alloc;
  int thread_id = threadIdx.x;

  if (thread_id == 0) {
    g_arena_alloc =
        reinterpret_cast<hipc::ArenaAllocator<false> *>(backend.data_);
    new (g_arena_alloc) hipc::ArenaAllocator<false>();
    g_arena_alloc->shm_init(backend, backend.data_capacity_);
  }
  __syncthreads();

  results[thread_id] = 0;  // Success
}

/**
 * Test just IpcManager construction in __shared__ memory
 */
__global__ void test_gpu_ipc_construct_kernel(int *results) {
  chi::IpcManager *ipc = chi::IpcManager::GetBlockIpcManager();
  int thread_id = threadIdx.x;
  __syncthreads();

  results[thread_id] = (ipc != nullptr) ? 0 : 1;
}

/**
 * Simple GPU kernel for testing CHIMAERA_GPU_INIT without allocation
 * Just verifies initialization succeeds
 */
__global__ void test_gpu_init_only_kernel(
    const hipc::MemoryBackend backend,
    int *results)  ///< Output: test results (0=pass, non-zero=fail)
{
  // Initialize IPC manager using the macro
  CHIMAERA_GPU_INIT(backend, nullptr);

  // Just report success if initialization didn't crash
  results[thread_id] = 0;
  __syncthreads();
}

/**
 * GPU kernel for testing CHIMAERA_GPU_INIT and AllocateBuffer
 * Each thread allocates a buffer, writes data, and verifies it
 */
__global__ void test_gpu_allocate_buffer_kernel(
    const hipc::MemoryBackend backend,
    int *results,             ///< Output: test results (0=pass, non-zero=fail)
    size_t *allocated_sizes,  ///< Output: sizes allocated per thread
    char **allocated_ptrs)    ///< Output: pointers allocated per thread
{
  // Initialize IPC manager using the macro
  CHIMAERA_GPU_INIT(backend, nullptr);

  // Each thread allocates a small buffer (64 bytes)
  size_t alloc_size = 64;

  // Allocate buffer using GPU path
  hipc::FullPtr<char> buffer = CHI_IPC->AllocateBuffer(alloc_size);

  // Store results
  if (buffer.IsNull()) {
    results[thread_id] = 1;  // Allocation failed
    allocated_sizes[thread_id] = 0;
    allocated_ptrs[thread_id] = nullptr;
  } else {
    // Write pattern to buffer
    char pattern = (char)(thread_id + 1);
    for (size_t i = 0; i < alloc_size; ++i) {
      buffer.ptr_[i] = pattern;
    }

    // Verify pattern
    bool pattern_ok = true;
    for (size_t i = 0; i < alloc_size; ++i) {
      if (buffer.ptr_[i] != pattern) {
        pattern_ok = false;
        break;
      }
    }

    results[thread_id] = pattern_ok ? 0 : 2;  // 2=verification failed
    allocated_sizes[thread_id] = alloc_size;
    allocated_ptrs[thread_id] = buffer.ptr_;
  }

  __syncthreads();
}

/**
 * GPU kernel for testing ToFullPtr on GPU
 * Allocates a buffer, gets its FullPtr, and verifies it works
 */
__global__ void test_gpu_to_full_ptr_kernel(
    const hipc::MemoryBackend backend,
    int *results)  ///< Output: test results (0=pass, non-zero=fail)
{
  // Initialize IPC manager in shared memory
  CHIMAERA_GPU_INIT(backend, nullptr);

  // Allocate a buffer
  size_t alloc_size = 512;
  hipc::FullPtr<char> buffer = CHI_IPC->AllocateBuffer(alloc_size);

  if (buffer.IsNull()) {
    results[thread_id] = 1;  // Allocation failed
    return;
  }

  // Write test data
  char test_value = (char)(thread_id + 42);
  for (size_t i = 0; i < alloc_size; ++i) {
    buffer.ptr_[i] = test_value;
  }

  // Get a ShmPtr and convert back to FullPtr
  hipc::ShmPtr<char> shm_ptr = buffer.shm_;

  // Convert back using ToFullPtr
  hipc::FullPtr<char> recovered = CHI_IPC->ToFullPtr(shm_ptr);

  if (recovered.IsNull()) {
    results[thread_id] = 3;  // ToFullPtr failed
    return;
  }

  // Verify the recovered pointer works
  bool data_ok = true;
  for (size_t i = 0; i < alloc_size; ++i) {
    if (recovered.ptr_[i] != test_value) {
      data_ok = false;
      break;
    }
  }

  results[thread_id] = data_ok ? 0 : 4;  // 4=recovered data mismatch
}

/**
 * GPU kernel for testing multiple independent allocations per thread
 * Each thread makes multiple allocations and verifies they're independent
 */
__global__ void test_gpu_multiple_allocs_kernel(
    const hipc::MemoryBackend backend,
    int *results)  ///< Output: test results (0=pass, non-zero=fail)
{
  // Initialize IPC manager in shared memory
  CHIMAERA_GPU_INIT(backend, nullptr);

  const int num_allocs = 4;
  size_t alloc_sizes[] = {256, 512, 1024, 2048};

  // Use local array for thread-local pointers
  char *local_ptrs[4];

  // Allocate multiple buffers
  for (int i = 0; i < num_allocs; ++i) {
    hipc::FullPtr<char> buffer =
        CHI_IPC->AllocateBuffer(alloc_sizes[i]);

    if (buffer.IsNull()) {
      results[thread_id] = 10 + i;  // Allocation i failed
      return;
    }

    local_ptrs[i] = buffer.ptr_;

    // Initialize with unique pattern
    char pattern = (char)(thread_id * num_allocs + i);
    for (size_t j = 0; j < alloc_sizes[i]; ++j) {
      local_ptrs[i][j] = pattern;
    }
  }

  // Verify all allocations
  for (int i = 0; i < num_allocs; ++i) {
    char expected = (char)(thread_id * num_allocs + i);
    for (size_t j = 0; j < alloc_sizes[i]; ++j) {
      if (local_ptrs[i][j] != expected) {
        results[thread_id] = 20 + i;  // Verification i failed
        return;
      }
    }
  }

  results[thread_id] = 0;  // All tests passed
}

/**
 * GPU kernel for testing NewTask from GPU
 * Tests that IpcManager::NewTask works from GPU kernel
 */
__global__ void test_gpu_new_task_kernel(const hipc::MemoryBackend backend,
                                         int *results) {
  // Initialize IPC manager (defines thread_id)
  CHIMAERA_GPU_INIT(backend, nullptr);

  // Only thread 0 creates task
  if (thread_id == 0) {
    // Create task using NewTask
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(1000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 0;
    chi::u32 test_value = 123;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                        task_id, pool_id, query, gpu_id, test_value);

    if (task.IsNull()) {
      results[0] = 1;  // NewTask failed
    } else {
      // Verify task was created correctly
      if (task->gpu_id_ == gpu_id && task->test_value_ == test_value) {
        results[0] = 0;  // Success
      } else {
        results[0] = 2;  // Task created but values wrong
      }
    }
  }

  __syncthreads();
}

/**
 * GPU kernel for testing task serialization/deserialization on GPU
 * Creates a task, uses GpuSaveTaskArchive to serialize it,
 * then GpuLoadTaskArchive to deserialize and verify
 */
__global__ void test_gpu_serialize_deserialize_kernel(
    const hipc::MemoryBackend backend, int *results) {
  // Initialize IPC manager (defines thread_id)
  CHIMAERA_GPU_INIT(backend, nullptr);

  // Only thread 0 tests serialization
  if (thread_id == 0) {
    // Create a task using NewTask
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(2000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 7;
    chi::u32 test_value = 456;

    auto original_task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
        task_id, pool_id, query, gpu_id, test_value);

    if (original_task.IsNull()) {
      results[0] = 1;  // NewTask failed
      __syncthreads();
      return;
    }

    // Allocate buffer for serialization
    size_t buffer_size = 1024;
    auto buffer_ptr = CHI_IPC->AllocateBuffer(buffer_size);

    if (buffer_ptr.IsNull()) {
      results[0] = 2;  // Buffer allocation failed
      __syncthreads();
      return;
    }

    // Serialize task using LocalSaveTaskArchive
    chi::LocalSaveTaskArchive save_ar(chi::LocalMsgType::kSerializeIn,
                                      buffer_ptr.ptr_, buffer_size);
    original_task->SerializeIn(save_ar);
    size_t serialized_size = save_ar.GetSize();

    // Create a new task to deserialize into
    auto loaded_task =
        CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>();

    if (loaded_task.IsNull()) {
      results[0] = 4;  // Second NewTask failed
      __syncthreads();
      return;
    }

    // Deserialize using LocalLoadTaskArchive
    chi::LocalLoadTaskArchive load_ar(buffer_ptr.ptr_, serialized_size);
    loaded_task->SerializeIn(load_ar);

    // Verify deserialized task matches original
    if (loaded_task->gpu_id_ == gpu_id &&
        loaded_task->test_value_ == test_value &&
        loaded_task->result_value_ == 0) {
      results[0] = 0;  // Success
    } else {
      results[0] = 3;  // Deserialization mismatch
    }
  }

  __syncthreads();
}

/**
 * GPU kernel for testing task serialization on GPU for CPU deserialization
 * Creates task, serializes with LocalSaveTaskArchive, ready for LocalTransfer
 * to CPU
 */
__global__ void test_gpu_serialize_for_cpu_kernel(
    const hipc::MemoryBackend backend, char *output_buffer, size_t *output_size,
    int *results) {
  // Initialize IPC manager (defines thread_id)
  CHIMAERA_GPU_INIT(backend, nullptr);

  // Only thread 0 serializes
  if (thread_id == 0) {
    // Create a task using NewTask
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(3000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 42;
    chi::u32 test_value = 99999;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                        task_id, pool_id, query, gpu_id, test_value);

    if (task.IsNull()) {
      results[0] = 1;  // NewTask failed
      *output_size = 0;
      __syncthreads();
      return;
    }

    // Serialize task using LocalSaveTaskArchive
    chi::LocalSaveTaskArchive save_ar(chi::LocalMsgType::kSerializeIn,
                                      output_buffer, 1024);
    task->SerializeIn(save_ar);

    // Store serialized size
    *output_size = save_ar.GetSize();
    results[0] = 0;  // Success
  }

  __syncthreads();
}

/**
 * GPU kernel that creates a task, serializes it into FutureShm via
 * MakeCopyFutureGpu, and returns the FutureShm ShmPtr for CPU deserialization.
 *
 * @param backend GPU memory backend for IPC allocation
 * @param d_future_shm_out Output: ShmPtr to FutureShm containing serialized
 * task
 * @param d_result Output: 0 on success, negative on error
 */
__global__ void test_gpu_make_copy_future_for_cpu_kernel(
    const hipc::MemoryBackend backend,
    hipc::ShmPtr<chi::FutureShm> *d_future_shm_out, int *d_result) {
  CHIMAERA_GPU_INIT(backend, nullptr);

  if (thread_id == 0) {
    // Create task on GPU
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(5000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 42;
    chi::u32 test_value = 99999;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                        task_id, pool_id, query, gpu_id, test_value);
    if (task.IsNull()) {
      *d_result = -1;  // NewTask failed
      return;
    }

    // Serialize task into FutureShm via MakeCopyFutureGpu
    auto future = CHI_IPC->MakeCopyFutureGpu(task);
    if (future.IsNull()) {
      *d_result = -2;  // MakeCopyFutureGpu failed
      return;
    }

    // Return the FutureShm ShmPtr so CPU can deserialize
    hipc::ShmPtr<chi::FutureShm> future_shm_ptr = future.GetFutureShmPtr();
    if (future_shm_ptr.IsNull()) {
      *d_result = -3;  // GetFutureShmPtr failed
      return;
    }
    *d_future_shm_out = future_shm_ptr;
    *d_result = 0;
  }

  __syncthreads();
}

/**
 * GPU kernel that reimplements IpcManager::Send on the GPU.
 * Creates a task, serializes it into FutureShm via MakeCopyFutureGpu,
 * enqueues the Future into the worker queue, and then blocks in
 * Future::Wait until the CPU sets FUTURE_COMPLETE.
 *
 * @param backend GPU memory backend for IPC allocation
 * @param worker_queue TaskQueue for enqueuing futures
 * @param d_result Output: 0 on success, negative on error
 */
__global__ void test_gpu_send_queue_wait_kernel(
    const hipc::MemoryBackend backend,
    chi::TaskQueue *worker_queue,
    int *d_result) {
  CHIMAERA_GPU_INIT(backend, worker_queue);

  if (thread_id == 0) {
    printf("GPU send_queue_wait: creating task\n");

    // 1. Create task on GPU
    chi::TaskId task_id = chi::CreateTaskId();
    chi::PoolId pool_id(6000, 0);
    chi::PoolQuery query = chi::PoolQuery::Local();
    chi::u32 gpu_id = 42;
    chi::u32 test_value = 77777;

    auto task = CHI_IPC->NewTask<chimaera::MOD_NAME::GpuSubmitTask>(
                        task_id, pool_id, query, gpu_id, test_value);
    if (task.IsNull()) {
      printf("GPU send_queue_wait: NewTask failed\n");
      *d_result = -1;
      return;
    }

    printf("GPU send_queue_wait: serializing into FutureShm\n");

    // 2. Serialize task into FutureShm via MakeCopyFutureGpu
    auto future = CHI_IPC->MakeCopyFutureGpu(task);
    if (future.IsNull()) {
      printf("GPU send_queue_wait: MakeCopyFutureGpu failed\n");
      *d_result = -2;
      return;
    }

    printf("GPU send_queue_wait: pushing to queue\n");

    // 3. Enqueue Future into worker queue lane 0
    auto &lane = worker_queue->GetLane(0, 0);
    chi::Future<chi::Task> task_future(future.GetFutureShmPtr());
    if (!lane.Push(task_future)) {
      printf("GPU send_queue_wait: Push failed\n");
      *d_result = -3;
      return;
    }

    printf("GPU send_queue_wait: waiting for FUTURE_COMPLETE\n");

    // 4. Block until CPU sets FUTURE_COMPLETE
    future.Wait();

    printf("GPU send_queue_wait: done\n");
    *d_result = 0;
  }

  __syncthreads();
}

/**
 * Helper function to run GPU kernel and check results
 * @param kernel_name Name of the kernel for error messages
 * @param backend GPU memory backend
 * @param block_size Number of GPU threads
 * @return true if all tests passed, false otherwise
 */
bool run_gpu_kernel_test(const std::string &kernel_name,
                         const hipc::MemoryBackend &backend, int block_size) {
  // Allocate result arrays on GPU
  int *d_results = hshm::GpuApi::Malloc<int>(sizeof(int) * block_size);

  // Initialize results to -1 (not run)
  std::vector<int> h_results(block_size, -1);
  hshm::GpuApi::Memcpy(d_results, h_results.data(), sizeof(int) * block_size);

  // Special test kernels
  if (kernel_name == "minimal") {
    test_gpu_minimal_kernel<<<1, block_size>>>(d_results);
  } else if (kernel_name == "backend_write") {
    test_gpu_backend_write_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "placement_new") {
    test_gpu_placement_new_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "shm_init") {
    test_gpu_shm_init_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "alloc_no_ipc") {
    test_gpu_alloc_no_ipc_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "ipc_construct") {
    test_gpu_ipc_construct_kernel<<<1, block_size>>>(d_results);
  } else if (kernel_name == "init_only") {
    test_gpu_init_only_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "allocate_buffer") {
    size_t *d_allocated_sizes =
        hshm::GpuApi::Malloc<size_t>(sizeof(size_t) * block_size);
    char **d_allocated_ptrs =
        hshm::GpuApi::Malloc<char *>(sizeof(char *) * block_size);

    test_gpu_allocate_buffer_kernel<<<1, block_size>>>(
        backend, d_results, d_allocated_sizes, d_allocated_ptrs);

    hshm::GpuApi::Free(d_allocated_sizes);
    hshm::GpuApi::Free(d_allocated_ptrs);
  } else if (kernel_name == "to_full_ptr") {
    test_gpu_to_full_ptr_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "multiple_allocs") {
    test_gpu_multiple_allocs_kernel<<<1, block_size>>>(backend, d_results);
  } else if (kernel_name == "new_task") {
    test_gpu_new_task_kernel<<<1, 1>>>(backend, d_results);
  } else if (kernel_name == "serialize_deserialize") {
    test_gpu_serialize_deserialize_kernel<<<1, 1>>>(backend, d_results);
  }

  // Synchronize to check for kernel errors
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    INFO("Kernel execution failed: " << cudaGetErrorString(sync_err));
    hshm::GpuApi::Free(d_results);
    return false;
  }

  // Copy results back
  cudaError_t memcpy_err =
      cudaMemcpy(h_results.data(), d_results, sizeof(int) * block_size,
                 cudaMemcpyDeviceToHost);
  if (memcpy_err != cudaSuccess) {
    INFO("Memcpy failed: " << cudaGetErrorString(memcpy_err));
    hshm::GpuApi::Free(d_results);
    return false;
  }
  hshm::GpuApi::Free(d_results);

  // Check results
  bool all_passed = true;
  for (int i = 0; i < block_size; ++i) {
    int expected = (kernel_name == "minimal") ? (i + 100) : 0;
    if (h_results[i] != expected) {
      INFO(kernel_name << " failed for thread " << i << ": result="
                       << h_results[i] << ", expected=" << expected);
      all_passed = false;
    }
  }

  return all_passed;
}

}  // namespace

TEST_CASE("GPU IPC AllocateBuffer basic functionality",
          "[gpu][ipc][allocate_buffer]") {
  // Create GPU memory backend
  hipc::MemoryBackendId backend_id(2, 0);     // Use ID 2.0 for GPU backend
  size_t gpu_memory_size = 10 * 1024 * 1024;  // 10MB GPU memory

  hipc::GpuShmMmap gpu_backend;
  REQUIRE(gpu_backend.shm_init(backend_id, gpu_memory_size, "/gpu_test", 0));

  SECTION("GPU kernel minimal (no macro)") {
    int block_size = 32;
    bool passed = run_gpu_kernel_test("minimal", gpu_backend, block_size);
    if (!passed) {
      INFO("Basic GPU kernel execution failed - hardware/driver issue?");
    }
    REQUIRE(passed);
  }

  SECTION("GPU kernel backend write") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("backend_write", gpu_backend, block_size));
  }

  SECTION("GPU kernel placement new") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("placement_new", gpu_backend, block_size));
  }

  SECTION("GPU kernel shm_init") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("shm_init", gpu_backend, block_size));
  }

  SECTION("GPU kernel alloc without IpcManager") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("alloc_no_ipc", gpu_backend, block_size));
  }

  SECTION("GPU kernel IpcManager construct") {
    int block_size = 32;
    REQUIRE(run_gpu_kernel_test("ipc_construct", gpu_backend, block_size));
  }

  SECTION("GPU kernel init only") {
    int block_size = 32;  // Warp size
    REQUIRE(run_gpu_kernel_test("init_only", gpu_backend, block_size));
  }

  SECTION("GPU kernel allocate buffer") {
    int block_size = 32;  // Warp size
    REQUIRE(run_gpu_kernel_test("allocate_buffer", gpu_backend, block_size));
  }

  SECTION("GPU kernel NewTask") {
    INFO("Testing IpcManager::NewTask on GPU");
    REQUIRE(run_gpu_kernel_test("new_task", gpu_backend, 1));
  }

  SECTION("GPU kernel serialize/deserialize") {
    INFO("Testing GPU task serialization and deserialization");
    REQUIRE(run_gpu_kernel_test("serialize_deserialize", gpu_backend, 1));
  }

  SECTION("GPU serialize -> CPU deserialize") {
    INFO(
        "Testing GPU task serialization -> LocalTransfer -> CPU "
        "deserialization");

    // Allocate pinned host buffer for transfer (LocalTransfer requires pinned
    // memory)
    size_t buffer_size = 1024;
    char *h_buffer = nullptr;
    cudaError_t err = cudaMallocHost(&h_buffer, buffer_size);
    REQUIRE(err == cudaSuccess);

    // Allocate GPU buffer
    char *d_buffer = hshm::GpuApi::Malloc<char>(buffer_size);
    size_t *d_output_size = hshm::GpuApi::Malloc<size_t>(sizeof(size_t));
    int *d_results = hshm::GpuApi::Malloc<int>(sizeof(int));

    // Run GPU kernel to serialize task using LocalSaveTaskArchive
    test_gpu_serialize_for_cpu_kernel<<<1, 1>>>(gpu_backend, d_buffer,
                                                d_output_size, d_results);

    err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Check GPU serialization result
    int h_result = -1;
    hshm::GpuApi::Memcpy(&h_result, d_results, sizeof(int));
    REQUIRE(h_result == 0);

    // Get serialized size
    size_t h_output_size = 0;
    hshm::GpuApi::Memcpy(&h_output_size, d_output_size, sizeof(size_t));
    INFO("Serialized task size: " + std::to_string(h_output_size) + " bytes");

    // LocalTransfer: Copy serialized data from GPU to pinned host memory
    hshm::GpuApi::Memcpy(h_buffer, d_buffer, h_output_size);

    // Deserialize on CPU using LocalLoadTaskArchive
    std::vector<char> cpu_buffer(h_buffer, h_buffer + h_output_size);
    chi::LocalLoadTaskArchive load_ar(cpu_buffer);

    // Create a task to deserialize into
    chimaera::MOD_NAME::GpuSubmitTask cpu_task;
    cpu_task.SerializeIn(load_ar);

    // Debug output
    INFO("Deserialized values: gpu_id=" + std::to_string(cpu_task.gpu_id_) +
         ", test_value=" + std::to_string(cpu_task.test_value_) +
         ", result_value=" + std::to_string(cpu_task.result_value_));

    // Verify deserialized task values
    REQUIRE(cpu_task.gpu_id_ == 42);
    REQUIRE(cpu_task.test_value_ == 99999);
    REQUIRE(cpu_task.result_value_ == 0);

    INFO(
        "SUCCESS: GPU serialized task -> LocalTransfer -> CPU deserialized "
        "correctly!");

    // Cleanup
    cudaFreeHost(h_buffer);
    hshm::GpuApi::Free(d_buffer);
    hshm::GpuApi::Free(d_output_size);
    hshm::GpuApi::Free(d_results);
  }

  // TODO: Fix these tests
  // SECTION("GPU kernel ToFullPtr") {
  //   int block_size = 32;
  //   REQUIRE(run_gpu_kernel_test("to_full_ptr", gpu_backend, block_size));
  // }

  // SECTION("GPU kernel multiple allocations") {
  //   int block_size = 32;
  //   REQUIRE(run_gpu_kernel_test("multiple_allocs", gpu_backend, block_size));
  // }

  SECTION("GPU MakeCopyFuture -> CPU Deserialize") {
    INFO(
        "Testing GPU task serialization into FutureShm, then CPU "
        "deserialization");

    // Allocate GPU output buffers
    auto *d_future_shm_ptr = hshm::GpuApi::Malloc<hipc::ShmPtr<chi::FutureShm>>(
        sizeof(hipc::ShmPtr<chi::FutureShm>));
    int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));

    // Initialize output buffers
    hipc::ShmPtr<chi::FutureShm> h_null_ptr;
    h_null_ptr.SetNull();
    hshm::GpuApi::Memcpy(d_future_shm_ptr, &h_null_ptr,
                         sizeof(hipc::ShmPtr<chi::FutureShm>));
    int h_result_init = -999;
    hshm::GpuApi::Memcpy(d_result, &h_result_init, sizeof(int));

    // MakeCopyFutureGpu needs extra stack for serialization
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    // Launch kernel: creates task and serializes into FutureShm
    test_gpu_make_copy_future_for_cpu_kernel<<<1, 1>>>(
        gpu_backend, d_future_shm_ptr, d_result);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      INFO("CUDA error: " << cudaGetErrorString(err));
    }
    REQUIRE(err == cudaSuccess);

    // Verify kernel succeeded
    int h_result = -999;
    hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));
    INFO("GPU kernel result: " << h_result);
    REQUIRE(h_result == 0);

    // Retrieve FutureShm ShmPtr from GPU
    hipc::ShmPtr<chi::FutureShm> h_future_shm_ptr;
    hshm::GpuApi::Memcpy(&h_future_shm_ptr, d_future_shm_ptr,
                         sizeof(hipc::ShmPtr<chi::FutureShm>));
    REQUIRE(!h_future_shm_ptr.IsNull());

    // Resolve ShmPtr to raw pointer using backend base address + offset
    chi::FutureShm *future_shm = reinterpret_cast<chi::FutureShm *>(
        reinterpret_cast<char *>(gpu_backend.data_) +
        h_future_shm_ptr.off_.load());
    REQUIRE(future_shm != nullptr);

    // Verify serialized data exists in copy_space
    size_t input_size = future_shm->input_.total_written_.load();
    INFO("Serialized size: " << input_size << " bytes");
    REQUIRE(input_size > 0);
    REQUIRE(future_shm->flags_.Any(chi::FutureShm::FUTURE_COPY_FROM_CLIENT));

    // Deserialize on CPU from FutureShm copy_space
    std::vector<char> cpu_buffer(future_shm->copy_space,
                                 future_shm->copy_space + input_size);
    chi::LocalLoadTaskArchive load_ar(cpu_buffer);
    chimaera::MOD_NAME::GpuSubmitTask deserialized_task;
    deserialized_task.SerializeIn(load_ar);

    // Verify deserialized task matches original values
    INFO("Deserialized: gpu_id="
         << deserialized_task.gpu_id_
         << ", test_value=" << deserialized_task.test_value_
         << ", result_value=" << deserialized_task.result_value_);
    REQUIRE(deserialized_task.gpu_id_ == 42);
    REQUIRE(deserialized_task.test_value_ == 99999);
    REQUIRE(deserialized_task.result_value_ == 0);

    // Cleanup
    hshm::GpuApi::Free(d_future_shm_ptr);
    hshm::GpuApi::Free(d_result);
  }

  SECTION("GPU Send -> Queue -> Wait") {
    INFO("Testing GPU task creation, queue enqueue, and Future::Wait");

    // Create queue backend (GPU-accessible host memory)
    hipc::MemoryBackendId queue_backend_id(3, 0);
    size_t queue_memory_size = 64 * 1024 * 1024;
    hipc::GpuShmMmap queue_backend;
    REQUIRE(queue_backend.shm_init(queue_backend_id, queue_memory_size,
                                   "/gpu_queue_test", 0));

    // Create ArenaAllocator on queue backend
    auto *queue_allocator = reinterpret_cast<hipc::ArenaAllocator<false> *>(
        queue_backend.data_);
    new (queue_allocator) hipc::ArenaAllocator<false>();
    queue_allocator->shm_init(queue_backend, queue_backend.data_capacity_);

    // Create TaskQueue (1 group, 1 lane per group, depth 256)
    auto gpu_queue = queue_allocator->template NewObj<chi::TaskQueue>(
        queue_allocator, 1, 1, 256);
    REQUIRE(!gpu_queue.IsNull());

    // Allocate GPU result buffer
    int *d_result = hshm::GpuApi::Malloc<int>(sizeof(int));
    int h_result_init = -999;
    hshm::GpuApi::Memcpy(d_result, &h_result_init, sizeof(int));

    // Extra stack for serialization
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    // Launch kernel async (kernel will block in Future::Wait)
    test_gpu_send_queue_wait_kernel<<<1, 1>>>(
        gpu_backend, gpu_queue.ptr_, d_result);

    // CPU polls queue until a Future is available (no cudaDeviceSynchronize)
    auto &lane = gpu_queue.ptr_->GetLane(0, 0);
    chi::Future<chi::Task> popped_future;
    while (!lane.Pop(popped_future)) {
      // Spin until GPU pushes the future
    }
    INFO("Popped future from queue");

    // Resolve FutureShm pointer using data backend base address
    hipc::ShmPtr<chi::FutureShm> future_shm_ptr =
        popped_future.GetFutureShmPtr();
    REQUIRE(!future_shm_ptr.IsNull());
    chi::FutureShm *future_shm = reinterpret_cast<chi::FutureShm *>(
        reinterpret_cast<char *>(gpu_backend.data_) +
        future_shm_ptr.off_.load());

    // Verify FUTURE_COPY_FROM_CLIENT flag and serialized data
    REQUIRE(future_shm->flags_.Any(chi::FutureShm::FUTURE_COPY_FROM_CLIENT));
    size_t input_size = future_shm->input_.total_written_.load();
    INFO("Serialized size: " << input_size << " bytes");
    REQUIRE(input_size > 0);

    // Deserialize on CPU and verify task values
    std::vector<char> cpu_buffer(future_shm->copy_space,
                                 future_shm->copy_space + input_size);
    chi::LocalLoadTaskArchive load_ar(cpu_buffer);
    chimaera::MOD_NAME::GpuSubmitTask deserialized_task;
    deserialized_task.SerializeIn(load_ar);

    INFO("Deserialized: gpu_id=" << deserialized_task.gpu_id_
         << ", test_value=" << deserialized_task.test_value_
         << ", result_value=" << deserialized_task.result_value_);
    REQUIRE(deserialized_task.gpu_id_ == 42);
    REQUIRE(deserialized_task.test_value_ == 77777);
    REQUIRE(deserialized_task.result_value_ == 0);

    // Set FUTURE_COMPLETE to unblock the GPU kernel's Future::Wait
    future_shm->flags_.SetBits(chi::FutureShm::FUTURE_COMPLETE);

    // Wait for kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      INFO("CUDA error: " << cudaGetErrorString(err));
    }
    REQUIRE(err == cudaSuccess);

    // Verify kernel result
    int h_result = -999;
    hshm::GpuApi::Memcpy(&h_result, d_result, sizeof(int));
    INFO("GPU kernel result: " << h_result);
    REQUIRE(h_result == 0);

    // Cleanup
    hshm::GpuApi::Free(d_result);
  }
}

// TODO: Fix per-thread allocations test
/*TEST_CASE("GPU IPC per-thread allocations", "[gpu][ipc][per_thread]") {
  // Create GPU memory backend with larger size for multiple threads
  hipc::MemoryBackendId backend_id(3, 0);
  size_t gpu_memory_size = 50 * 1024 * 1024;  // 50MB for more threads

  hipc::GpuShmMmap gpu_backend;
  REQUIRE(gpu_backend.shm_init(backend_id, gpu_memory_size, "/gpu_test_mt", 0));

  SECTION("GPU kernel with 64 threads") {
    int block_size = 64;
    REQUIRE(run_gpu_kernel_test("allocate_buffer", gpu_backend, block_size));
  }

  SECTION("GPU kernel with 128 threads") {
    int block_size = 128;
    REQUIRE(run_gpu_kernel_test("allocate_buffer", gpu_backend, block_size));
  }
}*/

SIMPLE_TEST_MAIN()

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
