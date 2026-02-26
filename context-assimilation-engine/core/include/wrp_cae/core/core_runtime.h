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

#ifndef WRP_CAE_CORE_RUNTIME_H_
#define WRP_CAE_CORE_RUNTIME_H_

#include <chimaera/chimaera.h>
#include <wrp_cae/core/core_tasks.h>
#include <wrp_cae/core/core_client.h>
#include <memory>

// Forward declaration for CTE client
namespace wrp_cte::core {
  class Client;
}

namespace wrp_cae::core {

class Runtime : public chi::Container {
 public:
  // CreateParams type used by CHI_TASK_CC macro for lib_name access
  using CreateParams = wrp_cae::core::CreateParams;

  Runtime() = default;
  ~Runtime() override = default;

  // Virtual methods implemented in autogen/core_lib_exec.cc
  chi::TaskResume Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr, chi::RunContext& rctx) override;
  chi::u64 GetWorkRemaining() const override;
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive, hipc::FullPtr<chi::Task> task_ptr) override;
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive,
                hipc::FullPtr<chi::Task> task_ptr) override;
  hipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive& archive) override;
  void LocalLoadTask(chi::u32 method, chi::LocalLoadTaskArchive& archive,
                     hipc::FullPtr<chi::Task> task_ptr) override;
  hipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::LocalLoadTaskArchive& archive) override;
  void LocalSaveTask(chi::u32 method, chi::LocalSaveTaskArchive& archive, hipc::FullPtr<chi::Task> task_ptr) override;
  hipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method, hipc::FullPtr<chi::Task> orig_task_ptr, bool deep) override;
  hipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;
  void Aggregate(chi::u32 method, hipc::FullPtr<chi::Task> orig_task,
                 const hipc::FullPtr<chi::Task>& replica_task) override;
  void DelTask(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;
  /**
   * Initialize container with pool information (REQUIRED)
   * This is called by the framework before Create is called
   */
  void Init(const chi::PoolId& pool_id, const std::string& pool_name,
            chi::u32 container_id = 0) override;


  /**
   * Monitor container state (Method::kMonitor)
   */
  chi::TaskResume Monitor(hipc::FullPtr<MonitorTask> task, chi::RunContext &rctx);

  /**
   * Create the container (Method::kCreate)
   * This method creates queues and sets up container resources
   * NOTE: Container is already initialized via Init() before Create is called
   */
  chi::TaskResume Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx);

  /**
   * Destroy the container (Method::kDestroy)
   */
  chi::TaskResume Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext& ctx) {
    HLOG(kInfo, "Core container destroyed for pool: {} (ID: {})",
          pool_name_, pool_id_);
    co_return;
  }

  /**
   * ParseOmni - Parse OMNI YAML file and schedule assimilation tasks (Method::kParseOmni)
   * This is a coroutine that uses co_await for async assimilator operations.
   * @return TaskResume for coroutine suspension/resumption
   */
  chi::TaskResume ParseOmni(hipc::FullPtr<ParseOmniTask> task, chi::RunContext& ctx);

  /**
   * ProcessHdf5Dataset - Process a single HDF5 dataset (Method::kProcessHdf5Dataset)
   * Used for distributed processing where each dataset task is routed to a specific node.
   * @return TaskResume for coroutine suspension/resumption
   */
  chi::TaskResume ProcessHdf5Dataset(hipc::FullPtr<ProcessHdf5DatasetTask> task, chi::RunContext& ctx);

 private:
  Client client_;
  std::shared_ptr<wrp_cte::core::Client> cte_client_;
};

}  // namespace wrp_cae::core

#endif  // WRP_CAE_CORE_RUNTIME_H_
