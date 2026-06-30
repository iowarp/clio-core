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

#ifndef MOD_NAME_RUNTIME_H_
#define MOD_NAME_RUNTIME_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/container.h>
#include <clio_runtime/comutex.h>
#include <clio_runtime/corwlock.h>
#include "MOD_NAME_tasks.h"
#include "autogen/MOD_NAME_methods.h"
#include "MOD_NAME_client.h"

namespace clio::run::MOD_NAME {

// Forward declarations (CustomTask only, CreateTask is a using alias in MOD_NAME_tasks.h)
struct CustomTask;
struct CoMutexTestTask;
struct CoRwLockTestTask;
struct WaitTestTask;
struct TestLargeOutputTask;
struct GpuSubmitTask;
struct SubtaskTestTask;

/**
 * Runtime implementation for MOD_NAME container
 */
class Runtime : public clio::run::Container {
public:
  // CreateParams type used by CLIO_TASK_CC macro for lib_name access
  using CreateParams = clio::run::MOD_NAME::CreateParams;

private:
  // Container-specific state
  clio::run::u32 create_count_ = 0;
  clio::run::u32 custom_count_ = 0;

  // Client for making calls to this ChiMod
  Client client_;

  // Static synchronization objects for testing
  static clio::run::CoMutex test_comutex_;
  static clio::run::CoRwLock test_corwlock_;

public:
  /**
   * Constructor
   */
  Runtime() = default;

  /**
   * Destructor
   */
  virtual ~Runtime() = default;


  /**
   * Initialize container with pool information
   */
  void Init(const clio::run::PoolId& pool_id, const std::string& pool_name,
            clio::run::u32 container_id = 0) override;

  /**
   * Execute a method on a task
   */
  clio::run::TaskResume Run(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task_ptr) override;

  //===========================================================================
  // Method implementations
  //===========================================================================

  /**
   * Handle Create task
   * Returns TaskResume for consistency with Run method
   */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);

  /**
   * Handle Custom task
   * Returns TaskResume for consistency with Run method
   */
  clio::run::TaskResume Custom(clio::run::shared_ptr<CustomTask> &task);

  /**
   * Handle CoMutexTest task
   * Returns TaskResume for consistency with Run method
   */
  clio::run::TaskResume CoMutexTest(clio::run::shared_ptr<CoMutexTestTask> &task);

  /**
   * Handle CoRwLockTest task
   * Returns TaskResume for consistency with Run method
   */
  clio::run::TaskResume CoRwLockTest(clio::run::shared_ptr<CoRwLockTestTask> &task);

  /**
   * Handle WaitTest task
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume WaitTest(clio::run::shared_ptr<WaitTestTask> &task);

  /**
   * Handle TestLargeOutput task
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume TestLargeOutput(clio::run::shared_ptr<TestLargeOutputTask> &task);

  /**
   * Handle GpuSubmit task (GPU-compatible task for Part 3 testing)
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume GpuSubmit(clio::run::shared_ptr<GpuSubmitTask> &task);

  clio::run::TaskResume SubtaskTest(clio::run::shared_ptr<SubtaskTestTask> &task);

  /**
   * Handle ManyToOneSum task — the aggregate task's value_ already holds the
   * summed batch input (folded by AggregateIn); echo it into sum_, which is
   * broadcast back to every batched submitter.
   */
  clio::run::TaskResume ManyToOneSum(clio::run::shared_ptr<ManyToOneSumTask> &task);

  /**
   * Handle Monitor task - return msgpack-encoded test data
   * Part of the unified kMonitor:9 interface
   */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);

  /**
   * Handle Destroy task - Alias for DestroyPool (DestroyTask = DestroyPoolTask)
   * Returns TaskResume for consistency with Run method
   */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);

  /**
   * Get remaining work count for this container
   * Template implementation returns 0 (no work tracking)
   */
  clio::run::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Task Serialization Methods
  //===========================================================================

  /**
   * Serialize task parameters for network transfer (unified method)
   */
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive& archive, clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Deserialize task parameters into an existing task from network transfer
   */
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Allocate and deserialize task parameters from network transfer
   */
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive& archive) override;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   */
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive& archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   */
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive& archive) override;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   */
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive& archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method,
                                        clio::run::shared_ptr<clio::run::Task> &orig_task_ptr, bool deep) override;

  /**
   * Create a new task of the specified method type
   */
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task>& replica_task) override;
  void AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task>& member_task) override;

};

} // namespace clio::run::MOD_NAME

#endif // MOD_NAME_RUNTIME_H_