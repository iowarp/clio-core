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

#ifndef SIMPLE_MOD_RUNTIME_H_
#define SIMPLE_MOD_RUNTIME_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/container.h>
#include "simple_mod_tasks.h"
#include "simple_mod_client.h"

namespace external_test::simple_mod {

// Simple mod local queue indices
enum SimpleModQueueIndex {
  kMetadataQueue = 0,  // Queue for metadata operations
};

/**
 * Runtime implementation for Simple Mod container
 * 
 * Minimal ChiMod for testing external development patterns.
 * Demonstrates basic runtime structure for external ChiMod development.
 */
class Runtime : public clio::run::Container {
public:
  // CreateParams type used by CLIO_TASK_CC macro for lib_name access
  using CreateParams = external_test::simple_mod::CreateParams;

private:
  // Container-specific state
  clio::run::u32 create_count_ = 0;
  
  // Client for making calls to this ChiMod
  Client client_;

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
   * @param pool_id The unique ID of this pool
   * @param pool_name The semantic name of this pool (user-provided)
   * @param container_id The container ID
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
   * Handle Create task - Initialize the Simple Mod container
   */
  void Create(clio::run::shared_ptr<CreateTask> &task);

  /**
   * Handle Destroy task - Destroy the Simple Mod container
   */
  void Destroy(clio::run::shared_ptr<DestroyTask> &task);

  /**
   * Handle Flush task - Flush simple mod operations
   */
  void Flush(clio::run::shared_ptr<FlushTask> &task);

  /**
   * Get remaining work count for this simple mod container
   */
  clio::run::u64 GetWorkRemaining() const override;

  //===========================================================================
  // Container Virtual Methods (automatically generated in autogen/)
  //===========================================================================

  /**
   * Serialize task parameters for network transfer (auto-generated)
   * @param method The method ID
   * @param archive SaveTaskArchive for serialization
   * @param task_ptr The task to serialize
   */
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Deserialize task parameters into an existing task from network transfer
   * (auto-generated)
   */
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Allocate and deserialize task parameters from network transfer
   * (auto-generated)
   */
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive& archive) override;

  /**
   * Deserialize task input parameters into an existing task for local transfer
   * (auto-generated)
   */
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive& archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Allocate and deserialize task input parameters for local transfer
   * (auto-generated)
   */
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive& archive) override;

  /**
   * Serialize task for local transfer (auto-generated)
   */
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive& archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Create a new copy of a task for distributed execution (auto-generated)
   */
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method,
                                        clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
                                        bool deep) override;

  /**
   * Create a new task of the specified method type (auto-generated)
   */
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task>& replica_task) override;

};

}  // namespace external_test::simple_mod

#endif  // SIMPLE_MOD_RUNTIME_H_