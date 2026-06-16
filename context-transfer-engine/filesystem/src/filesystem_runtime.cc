/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Filesystem chimod method handlers. SCAFFOLD (#552): the bodies below are
 * stubs. Each is meant to be implemented exactly like the corresponding
 * libfuse adapter op (context-transfer-engine/adapter/libfuse/fuse_cte.{h,cc}),
 * but driving the CTE core client `cte_` instead of CLIO_CTE_CLIENT, and
 * maintaining per-file logical size in `handles_`/`by_path_`:
 *
 *   Open      -> resolve/create tag (cte_.AsyncGetOrCreateTag), assign a
 *                handle, seed FileInfo{tag,size=GetTagSize}.
 *   Read      -> page-loop cte_.AsyncGetBlob over [offset,offset+size).
 *   Write     -> page-loop cte_.AsyncPutBlob; new_size = max(size, offset+len).
 *   Append    -> Write at FileInfo.size_, then advance it.
 *   Getattr   -> tag exists? dir? FileInfo.size_ (logical).
 *   Truncate  -> set FileInfo.size_; free trailing page-blobs (cte_.AsyncDelBlob).
 *   Unlink    -> cte_.AsyncDelTag.
 *   Mkdir/Rmdir/Readdir/Rename -> sentinel-tag + TagQuery, like libfuse.
 */
#include <clio_cte/filesystem/filesystem_runtime.h>

namespace clio::cte::filesystem {

#define CLIO_FS_TODO(task)            \
  do {                                \
    (task)->return_code_ = ENOSYS;    \
    (void)ctx;                        \
    CLIO_CO_RETURN;                   \
  } while (0)

chi::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task,
                                chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  // TODO(#552): read FilesystemConfig (next_pool_id_), bind cte_ to it.
  // next_pool_id_ = task->params_.next_pool_id_; cte_ = core::Client(next_pool_id_);
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Destroy(ctp::ipc::FullPtr<DestroyTask> task,
                                 chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Monitor(ctp::ipc::FullPtr<MonitorTask> task,
                                 chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  task->SetReturnCode(0);
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Open(ctp::ipc::FullPtr<OpenTask> task,
                              chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Close(ctp::ipc::FullPtr<CloseTask> task,
                               chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Read(ctp::ipc::FullPtr<ReadTask> task,
                              chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Write(ctp::ipc::FullPtr<WriteTask> task,
                               chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Append(ctp::ipc::FullPtr<AppendTask> task,
                                chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Getattr(ctp::ipc::FullPtr<GetattrTask> task,
                                 chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Truncate(ctp::ipc::FullPtr<TruncateTask> task,
                                  chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Unlink(ctp::ipc::FullPtr<UnlinkTask> task,
                                chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Mkdir(ctp::ipc::FullPtr<MkdirTask> task,
                               chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Rmdir(ctp::ipc::FullPtr<RmdirTask> task,
                               chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Rename(ctp::ipc::FullPtr<RenameTask> task,
                                chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::Readdir(ctp::ipc::FullPtr<ReaddirTask> task,
                                 chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}
chi::TaskResume Runtime::StatSize(ctp::ipc::FullPtr<StatSizeTask> task,
                                  chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_TODO(task);
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::filesystem
