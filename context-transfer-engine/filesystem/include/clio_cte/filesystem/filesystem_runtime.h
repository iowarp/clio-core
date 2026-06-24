/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_
#define CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/filesystem/filesystem_client.h>
#include <clio_cte/filesystem/filesystem_tasks.h>

namespace clio::cte::filesystem {

/**
 * Filesystem chimod runtime. A thin interface over the CTE core: every op
 * maps a path to a CTE tag and an offset range to 1 MiB page-blobs (exactly
 * like the libfuse adapter), driving the CTE core client. The one addition is
 * per-file logical-size metadata so getattr is exact and truncate/append work.
 */
class Runtime : public clio::run::Container {
 public:
  using CreateParams = FilesystemConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Method handlers ----
  clio::run::TaskResume Create(ctp::ipc::FullPtr<CreateTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Destroy(ctp::ipc::FullPtr<DestroyTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Monitor(ctp::ipc::FullPtr<MonitorTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Open(ctp::ipc::FullPtr<OpenTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Close(ctp::ipc::FullPtr<CloseTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Read(ctp::ipc::FullPtr<ReadTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Write(ctp::ipc::FullPtr<WriteTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Append(ctp::ipc::FullPtr<AppendTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Getattr(ctp::ipc::FullPtr<GetattrTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Truncate(ctp::ipc::FullPtr<TruncateTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Unlink(ctp::ipc::FullPtr<UnlinkTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Mkdir(ctp::ipc::FullPtr<MkdirTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Rmdir(ctp::ipc::FullPtr<RmdirTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Rename(ctp::ipc::FullPtr<RenameTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Link(ctp::ipc::FullPtr<LinkTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume Readdir(ctp::ipc::FullPtr<ReaddirTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume StatSize(ctp::ipc::FullPtr<StatSizeTask> task, clio::run::RunContext &rctx);
  // ---- deferred-append pipeline ----
  clio::run::TaskResume AppendSequence(ctp::ipc::FullPtr<AppendSequenceTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume AppendCollect(ctp::ipc::FullPtr<AppendCollectTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume AppendPlan(ctp::ipc::FullPtr<AppendPlanTask> task, clio::run::RunContext &rctx);
  clio::run::TaskResume AppendExecution(ctp::ipc::FullPtr<AppendExecutionTask> task, clio::run::RunContext &rctx);

  // ---- Container virtuals (defined in autogen/filesystem_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method, ctp::ipc::FullPtr<clio::run::Task> task_ptr,
                      clio::run::RunContext &rctx) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     ctp::ipc::FullPtr<clio::run::Task> task_ptr) override;
  ctp::ipc::FullPtr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     ctp::ipc::FullPtr<clio::run::Task> task_ptr) override;
  void AggregateOut(clio::run::u32 method, ctp::ipc::FullPtr<clio::run::Task> orig_task,
                    const ctp::ipc::FullPtr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method, ctp::ipc::FullPtr<clio::run::Task> agg_task,
                   const ctp::ipc::FullPtr<clio::run::Task> &member_task) override;
  void DelTask(clio::run::u32 method, ctp::ipc::FullPtr<clio::run::Task> task_ptr) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                ctp::ipc::FullPtr<clio::run::Task> task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                ctp::ipc::FullPtr<clio::run::Task> task_ptr) override;
  ctp::ipc::FullPtr<clio::run::Task> AllocLoadTask(clio::run::u32 method,
                                             clio::run::LoadTaskArchive &archive) override;
  ctp::ipc::FullPtr<clio::run::Task> NewCopyTask(clio::run::u32 method,
                                           ctp::ipc::FullPtr<clio::run::Task> orig,
                                           bool deep) override;
  ctp::ipc::FullPtr<clio::run::Task> NewTask(clio::run::u32 method) override;

 private:
  // CTE core client this filesystem sits over (set at Create from next_pool_id_).
  clio::cte::core::Client cte_;
  clio::run::PoolId next_pool_id_ = clio::run::PoolId::GetNull();
  // Client bound to THIS filesystem pool, for self-submitted pipeline tasks
  // (periodic AppendSequence, AppendCollect, AppendExecution).
  Client self_;
  // Staging tag for append data blobs. Append writes the bytes here (NOT under
  // the file's tag) so GetTagSize(file_tag) reports the true file tail,
  // unpolluted by still-unmerged staged appends. Resolved once at Create.
  clio::cte::core::TagId staging_tag_id_ = clio::cte::core::TagId::GetNull();

  // ---- per-file logical-size metadata + handle table ----
  struct FileInfo {
    clio::cte::core::TagId tag_id_;
    std::string path_;
    std::atomic<clio::run::u64> size_{0};  // logical size
  };
  std::mutex meta_mu_;
  std::unordered_map<clio::run::u64, std::shared_ptr<FileInfo>> handles_;  // handle -> file
  std::unordered_map<std::string, std::shared_ptr<FileInfo>> by_path_;
  std::atomic<clio::run::u64> next_handle_{1};

  // ---- deferred-append pipeline state ----
  // Per-node logical append counter (orders appends sharing a UTC tick).
  std::atomic<clio::run::u64> append_logical_{0};
  // Pending appends placed locally, awaiting the periodic AppendSequence drain.
  // Multi-producer (worker threads running Append), single-consumer
  // (AppendSequence) — guarded by a mutex rather than a fixed-capacity ring so
  // a burst of appends can never be silently dropped.
  struct PendingAppend {
    clio::cte::core::TagId tag_id_;
    AppendEntry entry_;
  };
  std::mutex append_mu_;
  std::vector<PendingAppend> append_pending_;
  bool append_seq_started_ = false;  // periodic AppendSequence kicked off once
};

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_
