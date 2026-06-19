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

#ifndef CLIO_CAE_CORE_RUNTIME_H_
#define CLIO_CAE_CORE_RUNTIME_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_cae/core/core_tasks.h>
#include <clio_cae/core/core_client.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

// Forward declaration for CTE client
namespace clio::cte::core {
  class Client;
}

namespace clio::cae::core {

class Runtime : public chi::Container {
 public:
  // CreateParams type used by CLIO_TASK_CC macro for lib_name access
  using CreateParams = clio::cae::core::CreateParams;

  Runtime() = default;
  ~Runtime() override = default;

  // Virtual methods implemented in autogen/core_lib_exec.cc
  chi::TaskResume Run(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr, chi::RunContext& rctx) override;
  chi::u64 GetWorkRemaining() const override;
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive, ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  ctp::ipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive& archive) override;
  void LocalLoadTask(chi::u32 method, chi::DefaultLoadArchive& archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  ctp::ipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::DefaultLoadArchive& archive) override;
  void LocalSaveTask(chi::u32 method, chi::DefaultSaveArchive& archive, ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  ctp::ipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task_ptr, bool deep) override;
  ctp::ipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;
  void AggregateOut(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task,
                 const ctp::ipc::FullPtr<chi::Task>& replica_task) override;
  void DelTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  /**
   * Initialize container with pool information (REQUIRED)
   * This is called by the framework before Create is called
   */
  void Init(const chi::PoolId& pool_id, const std::string& pool_name,
            chi::u32 container_id = 0) override;


  /**
   * Monitor container state (Method::kMonitor)
   */
  chi::TaskResume Monitor(ctp::ipc::FullPtr<MonitorTask> task, chi::RunContext &rctx);

  /**
   * Create the container (Method::kCreate)
   * This method creates queues and sets up container resources
   * NOTE: Container is already initialized via Init() before Create is called
   */
  chi::TaskResume Create(ctp::ipc::FullPtr<CreateTask> task, chi::RunContext& ctx);

  /**
   * Destroy the container (Method::kDestroy)
   */
  chi::TaskResume Destroy(ctp::ipc::FullPtr<DestroyTask> task, chi::RunContext& ctx) {
    HLOG(kInfo, "Core container destroyed for pool: {} (ID: {})",
          pool_name_, pool_id_);
#ifdef __NVCOMPILER
    chi::RunContext& rctx = ctx;
#endif
    CLIO_TASK_BODY_BEGIN
    CLIO_CO_RETURN;
    CLIO_TASK_BODY_END
  }

  /**
   * ParseOmni - Parse OMNI YAML file and schedule assimilation tasks (Method::kParseOmni)
   * This is a coroutine that uses co_await for async assimilator operations.
   * @return TaskResume for coroutine suspension/resumption
   */
  chi::TaskResume ParseOmni(ctp::ipc::FullPtr<ParseOmniTask> task, chi::RunContext& ctx);

  /**
   * ProcessHdf5Dataset - Process a single HDF5 dataset (Method::kProcessHdf5Dataset)
   * Used for distributed processing where each dataset task is routed to a specific node.
   * @return TaskResume for coroutine suspension/resumption
   */
  chi::TaskResume ProcessHdf5Dataset(ctp::ipc::FullPtr<ProcessHdf5DatasetTask> task, chi::RunContext& ctx);

  /**
   * ExportData - Export all blobs in a CTE tag to a file (Method::kExportData)
   * Supports "hdf5" and "binary" formats.
   * @return TaskResume for coroutine suspension/resumption
   */
  chi::TaskResume ExportData(ctp::ipc::FullPtr<ExportDataTask> task, chi::RunContext& ctx);

  /**
   * CTE interceptor handlers (Method::kPutBlob / kGetBlob / kGetOrCreateTag).
   * Each forwards the inbound task to the configured next CTE pool. No
   * labeling/intelligence yet — just passthrough so a client pointed at
   * the CAE pool transparently lands data in CTE behind it.
   */
  chi::TaskResume PutBlob(ctp::ipc::FullPtr<PutBlobTask> task,
                          chi::RunContext &ctx);
  chi::TaskResume GetBlob(ctp::ipc::FullPtr<GetBlobTask> task,
                          chi::RunContext &ctx);
  chi::TaskResume GetOrCreateTag(ctp::ipc::FullPtr<GetOrCreateTagTask> task,
                                 chi::RunContext &ctx);
  chi::TaskResume SemanticSearch(ctp::ipc::FullPtr<SemanticSearchTask> task,
                                 chi::RunContext &ctx);

  /**
   * Resolve PoolQuery::Dynamic() → PoolQuery::Local() for the interceptor
   * methods so they run on the receiving container without re-routing.
   */
  chi::PoolQuery ScheduleTask(
      const ctp::ipc::FullPtr<chi::Task> &task) override;

 private:
  /** Resolve the next pool ID (CTE core) we should forward to. */
  chi::PoolId ResolveNextPoolId() const;

  Client client_;
  std::shared_ptr<clio::cte::core::Client> cte_client_;
  chi::PoolId next_pool_id_;  // CTE core pool when CAE is the interceptor

  // Transparent labeling config snapshotted from CreateParams at Create
  // time. Read-only afterwards, so no synchronization is needed for the
  // PutBlob fast path.
  std::vector<LabelMatch> label_matches_;
  std::unordered_map<std::string, std::string> label_prompts_;
  std::string label_endpoint_;

  // tag_id → tag_name cache populated by GetOrCreateTag forwards.
  // PutBlobTask carries tag_id but not tag_name; matching against
  // LabelMatch::tag_re_ needs the name, so we remember it on the way
  // through. Read under shared lock from PutBlob, written under
  // exclusive lock from GetOrCreateTag.
  std::unordered_map<clio::cte::core::TagId, std::string> tag_names_;
  std::mutex tag_names_mu_;
};

}  // namespace clio::cae::core

#endif  // CLIO_CAE_CORE_RUNTIME_H_
