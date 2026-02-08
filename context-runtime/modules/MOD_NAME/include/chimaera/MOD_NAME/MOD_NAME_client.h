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

#ifndef MOD_NAME_CLIENT_H_
#define MOD_NAME_CLIENT_H_

#include <chimaera/chimaera.h>

#include "MOD_NAME_tasks.h"

/**
 * Client API for MOD_NAME
 *
 * Provides async methods for external programs to submit tasks to the runtime.
 * All methods return Future objects - call Wait() to block for completion.
 * Task cleanup is automatic when Future goes out of scope after Wait().
 */

namespace chimaera::MOD_NAME {

class Client : public chi::ContainerClient {
 public:
  /** Default constructor */
  Client() = default;

  /** Constructor with pool ID */
  explicit Client(const chi::PoolId& pool_id) { Init(pool_id); }

  /**
   * Create the container (asynchronous)
   * @param pool_query Pool routing information
   * @param pool_name Unique name for the pool (user-provided)
   * @param custom_pool_id Explicit pool ID for the pool being created
   * @return Future for the CreateTask
   */
  chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery& pool_query,
                                       const std::string& pool_name,
                                       const chi::PoolId& custom_pool_id) {
    auto* ipc_manager = CHI_IPC;

    // CreateTask is a GetOrCreatePoolTask, which must be handled by admin pool
    // Pass 'this' as client pointer for PostWait callback
    auto task = ipc_manager->NewTask<CreateTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,  // Send to admin pool for GetOrCreatePool processing
        pool_query,
        CreateParams::chimod_lib_name,  // chimod name from CreateParams
        pool_name,                      // user-provided pool name
        custom_pool_id,                 // target pool ID to create
        this                            // Client pointer for PostWait
    );

    return ipc_manager->Send(task);
  }

  /**
   * Execute custom operation (asynchronous)
   * @param pool_query Pool routing information
   * @param input_data Input data for the operation
   * @param operation_id Operation identifier
   * @return Future for the CustomTask
   */
  chi::Future<CustomTask> AsyncCustom(const chi::PoolQuery& pool_query,
                                       const std::string& input_data,
                                       chi::u32 operation_id) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<CustomTask>(
        chi::CreateTaskId(), pool_id_, pool_query, input_data, operation_id);

    return ipc_manager->Send(task);
  }

  /**
   * Execute CoMutex test (asynchronous)
   * @param pool_query Pool routing information
   * @param test_id Test identifier
   * @param hold_duration_ms Duration to hold the mutex in milliseconds
   * @return Future for the CoMutexTestTask
   */
  chi::Future<CoMutexTestTask> AsyncCoMutexTest(
      const chi::PoolQuery& pool_query,
      chi::u32 test_id, chi::u32 hold_duration_ms) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<CoMutexTestTask>(
        chi::CreateTaskId(), pool_id_, pool_query, test_id, hold_duration_ms);

    return ipc_manager->Send(task);
  }

  /**
   * Execute CoRwLock test (asynchronous)
   * @param pool_query Pool routing information
   * @param test_id Test identifier
   * @param is_writer Whether this is a writer (true) or reader (false)
   * @param hold_duration_ms Duration to hold the lock in milliseconds
   * @return Future for the CoRwLockTestTask
   */
  chi::Future<CoRwLockTestTask> AsyncCoRwLockTest(
      const chi::PoolQuery& pool_query,
      chi::u32 test_id, bool is_writer, chi::u32 hold_duration_ms) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<CoRwLockTestTask>(
        chi::CreateTaskId(), pool_id_, pool_query, test_id, is_writer,
        hold_duration_ms);

    return ipc_manager->Send(task);
  }

  /**
   * Submit Wait test task (asynchronous)
   * Tests recursive task.Wait() functionality with specified depth
   * @param pool_query Pool routing information
   * @param depth Number of recursive calls to make
   * @param test_id Test identifier for tracking
   * @return Future for the WaitTestTask
   */
  chi::Future<WaitTestTask> AsyncWaitTest(const chi::PoolQuery& pool_query,
                                           chi::u32 depth,
                                           chi::u32 test_id) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<WaitTestTask>(
        chi::CreateTaskId(), pool_id_, pool_query, depth, test_id);

    return ipc_manager->Send(task);
  }

  /**
   * Submit TestLargeOutput task (asynchronous)
   * Tests large output streaming functionality with 1MB output
   * @param pool_query Pool routing information
   * @return Future for the TestLargeOutputTask
   */
  chi::Future<TestLargeOutputTask> AsyncTestLargeOutput(const chi::PoolQuery& pool_query) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<TestLargeOutputTask>(
        chi::CreateTaskId(), pool_id_, pool_query);

    return ipc_manager->Send(task);
  }

  /**
   * Submit GpuSubmit task (asynchronous)
   * Tests GPU task submission functionality (Part 3)
   * @param pool_query Pool routing information
   * @param gpu_id GPU ID that submitted the task
   * @param test_value Test value to verify correct execution
   * @return Future for the GpuSubmitTask
   */
  chi::Future<GpuSubmitTask> AsyncGpuSubmit(const chi::PoolQuery& pool_query,
                                            chi::u32 gpu_id,
                                            chi::u32 test_value) {
    auto* ipc_manager = CHI_IPC;

    auto task = ipc_manager->NewTask<GpuSubmitTask>(
        chi::CreateTaskId(), pool_id_, pool_query, gpu_id, test_value);

    return ipc_manager->Send(task);
  }
};

}  // namespace chimaera::MOD_NAME

#endif  // MOD_NAME_CLIENT_H_
