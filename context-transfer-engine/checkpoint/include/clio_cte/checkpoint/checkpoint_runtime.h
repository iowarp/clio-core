/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CHECKPOINT_CHECKPOINT_RUNTIME_H_
#define CLIO_CTE_CHECKPOINT_CHECKPOINT_RUNTIME_H_

#include <memory>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/checkpoint/checkpoint_tasks.h>

namespace clio::cte::checkpoint {

/**
 * Checkpoint chimod runtime -- the FAULT HANDLER behind vector.Copy's lazy
 * copies. The core dispatches a GetBlobTask here when a Get or Put touches
 * a blob that a fault-handled tag does not hold. Context::fault_params_
 * names the SOURCE tag; this handler
 *
 *   1. reads the whole source blob (kNoFault, so nothing recurses),
 *   2. materialises it into the COPY tag with a kNoFault put,
 *   3. serves the originally requested range into the caller's buffer
 *      (size 0 = a put-side materialise-only fault: steps 1-2 alone).
 *
 * A blob absent from the source too falls back to the core's pre-fault
 * behaviour by forwarding the original get with kNoFault -- create_on_get
 * and error semantics stay exactly as they were.
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = CheckpointConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override {
    return clio::cte::core::CoreInterposer::GetTaskStats(task);
  }

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);

  // ---- Container virtuals (defined in autogen/checkpoint_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method,
                                             clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method,
                                           clio::run::shared_ptr<clio::run::Task> &orig,
                                           bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

 private:
  /** Lazily bind the next-pool client (compose next_pool_id / core). */
  clio::cte::core::Client *GetNextClient();

  /** A core get as a RAW task (never the client wrapper's SHM fast path --
   *  see the definition for why that path cannot run in a worker). */
  clio::run::Future<clio::cte::core::GetBlobTask> SendGet(
      const clio::cte::core::TagId &tag_id, const std::string &blob_name,
      clio::run::u64 offset, clio::run::u64 size, clio::run::u32 flags,
      ctp::ipc::ShmPtr<> blob_data, const clio::cte::core::Context &context);

  CheckpointConfig config_;
  std::unique_ptr<clio::cte::core::Client> next_client_;
};

}  // namespace clio::cte::checkpoint

#endif  // CLIO_CTE_CHECKPOINT_CHECKPOINT_RUNTIME_H_
