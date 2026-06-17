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
class Runtime : public chi::Container {
 public:
  using CreateParams = FilesystemConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Method handlers ----
  chi::TaskResume Create(ctp::ipc::FullPtr<CreateTask> task, chi::RunContext &ctx);
  chi::TaskResume Destroy(ctp::ipc::FullPtr<DestroyTask> task, chi::RunContext &ctx);
  chi::TaskResume Monitor(ctp::ipc::FullPtr<MonitorTask> task, chi::RunContext &ctx);
  chi::TaskResume Open(ctp::ipc::FullPtr<OpenTask> task, chi::RunContext &ctx);
  chi::TaskResume Close(ctp::ipc::FullPtr<CloseTask> task, chi::RunContext &ctx);
  chi::TaskResume Read(ctp::ipc::FullPtr<ReadTask> task, chi::RunContext &ctx);
  chi::TaskResume Write(ctp::ipc::FullPtr<WriteTask> task, chi::RunContext &ctx);
  chi::TaskResume Append(ctp::ipc::FullPtr<AppendTask> task, chi::RunContext &ctx);
  chi::TaskResume Getattr(ctp::ipc::FullPtr<GetattrTask> task, chi::RunContext &ctx);
  chi::TaskResume Truncate(ctp::ipc::FullPtr<TruncateTask> task, chi::RunContext &ctx);
  chi::TaskResume Unlink(ctp::ipc::FullPtr<UnlinkTask> task, chi::RunContext &ctx);
  chi::TaskResume Mkdir(ctp::ipc::FullPtr<MkdirTask> task, chi::RunContext &ctx);
  chi::TaskResume Rmdir(ctp::ipc::FullPtr<RmdirTask> task, chi::RunContext &ctx);
  chi::TaskResume Rename(ctp::ipc::FullPtr<RenameTask> task, chi::RunContext &ctx);
  chi::TaskResume Link(ctp::ipc::FullPtr<LinkTask> task, chi::RunContext &ctx);
  chi::TaskResume Readdir(ctp::ipc::FullPtr<ReaddirTask> task, chi::RunContext &ctx);
  chi::TaskResume StatSize(ctp::ipc::FullPtr<StatSizeTask> task, chi::RunContext &ctx);

  // ---- Container virtuals (defined in autogen/filesystem_lib_exec.cc) ----
  void Init(const chi::PoolId &pool_id, const std::string &pool_name,
            chi::u32 container_id = 0) override;
  chi::TaskResume Run(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr,
                      chi::RunContext &rctx) override;
  chi::u64 GetWorkRemaining() const override;
  void LocalLoadTask(chi::u32 method, chi::DefaultLoadArchive &archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  ctp::ipc::FullPtr<chi::Task> LocalAllocLoadTask(
      chi::u32 method, chi::DefaultLoadArchive &archive) override;
  void LocalSaveTask(chi::u32 method, chi::DefaultSaveArchive &archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  void Aggregate(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task,
                 const ctp::ipc::FullPtr<chi::Task> &replica_task) override;
  void DelTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  void SaveTask(chi::u32 method, chi::SaveTaskArchive &archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  void LoadTask(chi::u32 method, chi::LoadTaskArchive &archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;
  ctp::ipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method,
                                             chi::LoadTaskArchive &archive) override;
  ctp::ipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method,
                                           ctp::ipc::FullPtr<chi::Task> orig,
                                           bool deep) override;
  ctp::ipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;

 private:
  // CTE core client this filesystem sits over (set at Create from next_pool_id_).
  clio::cte::core::Client cte_;
  chi::PoolId next_pool_id_ = chi::PoolId::GetNull();

  // ---- per-file logical-size metadata + handle table ----
  struct FileInfo {
    clio::cte::core::TagId tag_id_;
    std::string path_;
    std::atomic<chi::u64> size_{0};  // logical size
  };
  std::mutex meta_mu_;
  std::unordered_map<chi::u64, std::shared_ptr<FileInfo>> handles_;  // handle -> file
  std::unordered_map<std::string, std::shared_ptr<FileInfo>> by_path_;
  std::atomic<chi::u64> next_handle_{1};
};

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_
