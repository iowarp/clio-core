/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_
#define CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_

#include <clio_cte/core/core_client.h>
#include <clio_cte/filesystem/filesystem_tasks.h>

namespace clio::cte::filesystem {

/**
 * Filesystem client — the single API every interceptor (POSIX, STDIO,
 * libfuse, HDF5 VFD, MPI-IO) calls. Inherits the full CTE core client (so all
 * blob/tag/target operations remain available) and adds POSIX-shaped
 * filesystem operations routed through the filesystem chimod, which owns the
 * path->tag mapping, page-blob I/O, and per-file logical-size metadata.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;
  explicit Client(const clio::run::PoolId &fs_pool_id) { Init(fs_pool_id); }

#if CTP_IS_HOST
  /** Create/initialize the filesystem container over a CTE core pool. */
  clio::run::Future<CreateTask> AsyncCreate(const clio::run::PoolQuery &pool_query,
                                      const std::string &pool_name,
                                      const clio::run::PoolId &custom_pool_id,
                                      const FilesystemConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        FilesystemConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    return ipc->Send(task);
  }

  clio::run::Future<OpenTask> AsyncOpen(const std::string &path, clio::run::u32 flags,
                                  clio::run::u32 mode = 0644) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<OpenTask>(clio::run::CreateTaskId(), pool_id_,
                                       clio::run::PoolQuery::Local(), path, flags,
                                       mode);
    return ipc->Send(task);
  }

  clio::run::Future<CloseTask> AsyncClose(clio::run::u64 handle) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CloseTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), handle);
    return ipc->Send(task);
  }

  clio::run::Future<ReadTask> AsyncRead(clio::run::u64 handle, clio::run::u64 offset,
                                  clio::run::u64 size, ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReadTask>(clio::run::CreateTaskId(), pool_id_,
                                       clio::run::PoolQuery::Local(), handle, offset,
                                       size, data);
    return ipc->Send(task);
  }

  clio::run::Future<WriteTask> AsyncWrite(clio::run::u64 handle, clio::run::u64 offset,
                                    clio::run::u64 size, ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<WriteTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), handle, offset,
                                        size, data);
    return ipc->Send(task);
  }

  clio::run::Future<AppendTask> AsyncAppend(clio::run::u64 handle, clio::run::u64 size,
                                      ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendTask>(clio::run::CreateTaskId(), pool_id_,
                                         clio::run::PoolQuery::Local(), handle, size,
                                         data);
    return ipc->Send(task);
  }

  clio::run::Future<GetattrTask> AsyncGetattr(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<GetattrTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<TruncateTask> AsyncTruncate(const std::string &path,
                                          clio::run::u64 new_size) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<TruncateTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path,
                                           new_size);
    return ipc->Send(task);
  }

  clio::run::Future<UnlinkTask> AsyncUnlink(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<UnlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                         clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<UtimensTask> AsyncUtimens(const std::string &path,
                                              clio::run::u64 atime_ns,
                                              clio::run::u64 mtime_ns,
                                              clio::run::u32 flags) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<UtimensTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), path,
                                          atime_ns, mtime_ns, flags);
    return ipc->Send(task);
  }

  clio::run::Future<ChownTask> AsyncChown(const std::string &path,
                                          clio::run::u32 uid,
                                          clio::run::u32 gid,
                                          clio::run::u32 mode = 0xFFFFFFFFu) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ChownTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), path, uid,
                                        gid, mode);
    return ipc->Send(task);
  }

  // chmod reuses the ChownTask (per-file mode override) with uid/gid left
  // unchanged, avoiding a separate RPC method for a single stored field.
  clio::run::Future<ChownTask> AsyncChmod(const std::string &path,
                                          clio::run::u32 mode) {
    return AsyncChown(path, 0xFFFFFFFFu, 0xFFFFFFFFu, mode);
  }

  clio::run::Future<MkdirTask> AsyncMkdir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<MkdirTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RmdirTask> AsyncRmdir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RmdirTask>(clio::run::CreateTaskId(), pool_id_,
                                        clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RenameTask> AsyncRename(const std::string &src,
                                      const std::string &dst) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RenameTask>(clio::run::CreateTaskId(), pool_id_,
                                         clio::run::PoolQuery::Local(), src, dst);
    return ipc->Send(task);
  }

  clio::run::Future<LinkTask> AsyncLink(const std::string &target,
                                  const std::string &link) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<LinkTask>(clio::run::CreateTaskId(), pool_id_,
                                       clio::run::PoolQuery::Local(), target, link);
    return ipc->Send(task);
  }

  clio::run::Future<SymlinkTask> AsyncSymlink(const std::string &target,
                                              const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SymlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), target,
                                          path);
    return ipc->Send(task);
  }

  clio::run::Future<ReadlinkTask> AsyncReadlink(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReadlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<SetxattrTask> AsyncSetxattr(const std::string &path,
                                                const std::string &name,
                                                const std::string &value,
                                                clio::run::u32 flags) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SetxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path,
                                           name, value, flags);
    return ipc->Send(task);
  }

  clio::run::Future<GetxattrTask> AsyncGetxattr(const std::string &path,
                                                const std::string &name) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<GetxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path,
                                           name);
    return ipc->Send(task);
  }

  clio::run::Future<ListxattrTask> AsyncListxattr(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ListxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                            clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RemovexattrTask> AsyncRemovexattr(const std::string &path,
                                                      const std::string &name) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RemovexattrTask>(clio::run::CreateTaskId(),
                                              pool_id_,
                                              clio::run::PoolQuery::Local(),
                                              path, name);
    return ipc->Send(task);
  }

  // ---- deferred-append pipeline ----
  /** Kick off (or tick) the periodic local pending-append drain. */
  clio::run::Future<AppendSequenceTask> AsyncAppendSequence(
      double period_us = 0.0,
      const clio::run::PoolQuery &pool_query = clio::run::PoolQuery::Local()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendSequenceTask>(clio::run::CreateTaskId(), pool_id_,
                                                 pool_query);
    if (period_us > 0.0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }
    return ipc->Send(task);
  }

  /** Collect one tag's pending appends at its sequencer (ManyToOne batch). */
  clio::run::Future<AppendCollectTask> AsyncAppendCollect(
      const clio::cte::core::TagId &tag_id,
      const std::vector<AppendEntry> &entries, const clio::run::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendCollectTask>(clio::run::CreateTaskId(), pool_id_,
                                                pool_query, tag_id, entries);
    return ipc->Send(task);
  }

  /** Plan + dispatch one tag's batch (suspendable; submitted by AppendCollect). */
  clio::run::Future<AppendPlanTask> AsyncAppendPlan(
      const clio::cte::core::TagId &tag_id,
      const std::vector<AppendEntry> &entries, const clio::run::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendPlanTask>(clio::run::CreateTaskId(), pool_id_,
                                             pool_query, tag_id, entries);
    return ipc->Send(task);
  }

  /** Apply a slice of the merge plan (GetBlob->PutBlob->DelBlob). */
  clio::run::Future<AppendExecutionTask> AsyncAppendExecution(
      const clio::cte::core::TagId &tag_id,
      const clio::cte::core::TagId &staging_tag_id,
      const std::vector<AppendPlanStep> &steps, const clio::run::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendExecutionTask>(
        clio::run::CreateTaskId(), pool_id_, pool_query, tag_id, staging_tag_id, steps);
    return ipc->Send(task);
  }

  clio::run::Future<ReaddirTask> AsyncReaddir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReaddirTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  clio::run::Future<StatSizeTask> AsyncStatSize(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<StatSizeTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Local(), path);
    return ipc->Send(task);
  }
#endif  // CTP_IS_HOST
};

// Process-wide filesystem client singleton (reuses the clio_cte export macro).
// Declared inside the namespace so it resolves as
// clio::cte::filesystem::g_fs_client (matching CLIO_CFS_CLIENT below and the
// definition in filesystem_client.cc), exactly like core's g_cte_client.
CLIO_CTE_DEFINE_GLOBAL_PTR_VAR_H(clio::cte::filesystem::Client, g_fs_client);

/** Initialize the filesystem client singleton (creates/binds the pool). */
bool CLIO_CFS_CLIENT_INIT(const std::string &config_path = "",
                          const clio::run::PoolQuery &pool_query =
                              clio::run::PoolQuery::Dynamic());

}  // namespace clio::cte::filesystem

#define CLIO_CFS_CLIENT                                  \
  (&(*CTP_GET_GLOBAL_PTR_VAR(clio::cte::filesystem::Client, \
                             clio::cte::filesystem::g_fs_client)))

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_
