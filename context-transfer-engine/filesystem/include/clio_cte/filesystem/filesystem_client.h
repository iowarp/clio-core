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
  explicit Client(const chi::PoolId &fs_pool_id) { Init(fs_pool_id); }

#if CTP_IS_HOST
  /** Create/initialize the filesystem container over a CTE core pool. */
  chi::Future<CreateTask> AsyncCreate(const chi::PoolQuery &pool_query,
                                      const std::string &pool_name,
                                      const chi::PoolId &custom_pool_id,
                                      const FilesystemConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, pool_query,
        FilesystemConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    return ipc->Send(task);
  }

  chi::Future<OpenTask> AsyncOpen(const std::string &path, chi::u32 flags,
                                  chi::u32 mode = 0644) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<OpenTask>(chi::CreateTaskId(), pool_id_,
                                       chi::PoolQuery::Local(), path, flags,
                                       mode);
    return ipc->Send(task);
  }

  chi::Future<CloseTask> AsyncClose(chi::u64 handle) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CloseTask>(chi::CreateTaskId(), pool_id_,
                                        chi::PoolQuery::Local(), handle);
    return ipc->Send(task);
  }

  chi::Future<ReadTask> AsyncRead(chi::u64 handle, chi::u64 offset,
                                  chi::u64 size, ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReadTask>(chi::CreateTaskId(), pool_id_,
                                       chi::PoolQuery::Local(), handle, offset,
                                       size, data);
    return ipc->Send(task);
  }

  chi::Future<WriteTask> AsyncWrite(chi::u64 handle, chi::u64 offset,
                                    chi::u64 size, ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<WriteTask>(chi::CreateTaskId(), pool_id_,
                                        chi::PoolQuery::Local(), handle, offset,
                                        size, data);
    return ipc->Send(task);
  }

  chi::Future<AppendTask> AsyncAppend(chi::u64 handle, chi::u64 size,
                                      ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendTask>(chi::CreateTaskId(), pool_id_,
                                         chi::PoolQuery::Local(), handle, size,
                                         data);
    return ipc->Send(task);
  }

  chi::Future<GetattrTask> AsyncGetattr(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<GetattrTask>(chi::CreateTaskId(), pool_id_,
                                          chi::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  chi::Future<TruncateTask> AsyncTruncate(const std::string &path,
                                          chi::u64 new_size) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<TruncateTask>(chi::CreateTaskId(), pool_id_,
                                           chi::PoolQuery::Local(), path,
                                           new_size);
    return ipc->Send(task);
  }

  chi::Future<UnlinkTask> AsyncUnlink(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<UnlinkTask>(chi::CreateTaskId(), pool_id_,
                                         chi::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  chi::Future<MkdirTask> AsyncMkdir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<MkdirTask>(chi::CreateTaskId(), pool_id_,
                                        chi::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  chi::Future<RmdirTask> AsyncRmdir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RmdirTask>(chi::CreateTaskId(), pool_id_,
                                        chi::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  chi::Future<RenameTask> AsyncRename(const std::string &src,
                                      const std::string &dst) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RenameTask>(chi::CreateTaskId(), pool_id_,
                                         chi::PoolQuery::Local(), src, dst);
    return ipc->Send(task);
  }

  chi::Future<LinkTask> AsyncLink(const std::string &target,
                                  const std::string &link) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<LinkTask>(chi::CreateTaskId(), pool_id_,
                                       chi::PoolQuery::Local(), target, link);
    return ipc->Send(task);
  }

  // ---- deferred-append pipeline ----
  /** Kick off (or tick) the periodic local pending-append drain. */
  chi::Future<AppendSequenceTask> AsyncAppendSequence(
      double period_us = 0.0,
      const chi::PoolQuery &pool_query = chi::PoolQuery::Local()) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendSequenceTask>(chi::CreateTaskId(), pool_id_,
                                                 pool_query);
    if (period_us > 0.0) {
      task->SetPeriod(period_us, chi::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }
    return ipc->Send(task);
  }

  /** Collect one tag's pending appends at its sequencer (ManyToOne batch). */
  chi::Future<AppendCollectTask> AsyncAppendCollect(
      const clio::cte::core::TagId &tag_id,
      const std::vector<AppendEntry> &entries, const chi::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendCollectTask>(chi::CreateTaskId(), pool_id_,
                                                pool_query, tag_id, entries);
    return ipc->Send(task);
  }

  /** Plan + dispatch one tag's batch (suspendable; submitted by AppendCollect). */
  chi::Future<AppendPlanTask> AsyncAppendPlan(
      const clio::cte::core::TagId &tag_id,
      const std::vector<AppendEntry> &entries, const chi::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendPlanTask>(chi::CreateTaskId(), pool_id_,
                                             pool_query, tag_id, entries);
    return ipc->Send(task);
  }

  /** Apply a slice of the merge plan (GetBlob->PutBlob->DelBlob). */
  chi::Future<AppendExecutionTask> AsyncAppendExecution(
      const clio::cte::core::TagId &tag_id,
      const clio::cte::core::TagId &staging_tag_id,
      const std::vector<AppendPlanStep> &steps, const chi::PoolQuery &pool_query) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AppendExecutionTask>(
        chi::CreateTaskId(), pool_id_, pool_query, tag_id, staging_tag_id, steps);
    return ipc->Send(task);
  }

  chi::Future<ReaddirTask> AsyncReaddir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReaddirTask>(chi::CreateTaskId(), pool_id_,
                                          chi::PoolQuery::Local(), path);
    return ipc->Send(task);
  }

  chi::Future<StatSizeTask> AsyncStatSize(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<StatSizeTask>(chi::CreateTaskId(), pool_id_,
                                           chi::PoolQuery::Local(), path);
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
                          const chi::PoolQuery &pool_query =
                              chi::PoolQuery::Dynamic());

}  // namespace clio::cte::filesystem

#define CLIO_CFS_CLIENT                                  \
  (&(*CTP_GET_GLOBAL_PTR_VAR(clio::cte::filesystem::Client, \
                             clio::cte::filesystem::g_fs_client)))

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_
