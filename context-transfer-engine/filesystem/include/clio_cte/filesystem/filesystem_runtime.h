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
#include <clio_cte/filesystem/shm_fs_cache.h>

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

  /**
   * Per-task cost estimate for the scheduler.
   *
   * Three fields, three consumers:
   *   io_size_    bytes the task moves — used for large-I/O routing.
   *   compute_    estimated CPU microseconds. This is the ONLY feature
   *               Container::InferCpuTime multiplies its learned per-method
   *               coefficient by, so leaving it 0 collapses that model to a
   *               constant with no dependence on request size — which is what
   *               it did here before this override existed (clio-fs had no
   *               GetTaskStats at all, so every task reported all zeros).
   *   wall_time_  estimated wall microseconds, seeded at the ~500 MB/s the
   *               other chimods use; InferWallClockTime learns the coefficient.
   *
   * The seeds below come from measured single-thread costs on this filesystem
   * (clio_cte_reliability_bench, 1 client thread): stat ~31us, readdir ~255us
   * for 10 entries and ~592us for 100, rename ~568us. They only need to be the
   * right ORDER of magnitude — ReinforceCpuModel/ReinforceWallModel converge
   * the coefficient from the first few completions.
   */
  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override {
    clio::run::TaskStat stat;
    if (task == nullptr) {
      return stat;
    }
    // Payload copy runs at roughly 10 GB/s, i.e. ~10 KB per CPU microsecond.
    constexpr float kBytesPerComputeUs = 10000.0f;
    constexpr float kBytesPerWallUs = 500.0f;  // ~500 MB/s, house convention
    switch (task->method_) {
      case Method::kRead: {
        const auto *t = static_cast<const ReadTask *>(task);
        stat.io_size_ = t->size_;
        stat.compute_ =
            static_cast<size_t>(t->size_ / kBytesPerComputeUs) + 2;
        stat.wall_time_ = static_cast<float>(t->size_) / kBytesPerWallUs;
        return stat;
      }
      case Method::kWrite: {
        const auto *t = static_cast<const WriteTask *>(task);
        stat.io_size_ = t->size_;
        stat.compute_ =
            static_cast<size_t>(t->size_ / kBytesPerComputeUs) + 2;
        stat.wall_time_ = static_cast<float>(t->size_) / kBytesPerWallUs;
        return stat;
      }
      case Method::kAppend: {
        const auto *t = static_cast<const AppendTask *>(task);
        stat.io_size_ = t->size_;
        stat.compute_ =
            static_cast<size_t>(t->size_ / kBytesPerComputeUs) + 2;
        stat.wall_time_ = static_cast<float>(t->size_) / kBytesPerWallUs;
        return stat;
      }
      case Method::kReaddir:
        // A listing is a trigram regex query over the tag index plus one
        // result marshalled per entry — CPU-bound, and by far the most
        // expensive metadata call. The entry count is not known until the
        // query returns, so this seeds the fixed part.
        stat.compute_ = 200;
        stat.wall_time_ = 250.0f;
        return stat;
      case Method::kRename:
        // Two index searches (self + descendants) plus the re-key.
        stat.compute_ = 400;
        stat.wall_time_ = 500.0f;
        return stat;
      case Method::kStatSize:
      case Method::kGetattr:
        // Resolved through the O(1) name hash when the path is cached.
        stat.compute_ = 20;
        stat.wall_time_ = 30.0f;
        return stat;
      case Method::kOpen:
      case Method::kClose:
      case Method::kMkdir:
      case Method::kRmdir:
      case Method::kUnlink:
      case Method::kTruncate:
      case Method::kLink:
      case Method::kSymlink:
      case Method::kReadlink:
      case Method::kUtimens:
      case Method::kChown:
      case Method::kSetxattr:
      case Method::kGetxattr:
      case Method::kListxattr:
      case Method::kRemovexattr:
        // Single metadata mutation/lookup against the tag map.
        stat.compute_ = 15;
        stat.wall_time_ = 25.0f;
        return stat;
      default:
        return stat;
    }
  }

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume Open(clio::run::shared_ptr<OpenTask> &task);
  clio::run::TaskResume Close(clio::run::shared_ptr<CloseTask> &task);
  clio::run::TaskResume MultiCreate(clio::run::shared_ptr<MultiCreateTask> &task);
  clio::run::TaskResume AdvanceSize(clio::run::shared_ptr<AdvanceSizeTask> &task);
  clio::run::TaskResume Read(clio::run::shared_ptr<ReadTask> &task);
  clio::run::TaskResume Write(clio::run::shared_ptr<WriteTask> &task);
  clio::run::TaskResume Append(clio::run::shared_ptr<AppendTask> &task);
  clio::run::TaskResume Getattr(clio::run::shared_ptr<GetattrTask> &task);
  clio::run::TaskResume Truncate(clio::run::shared_ptr<TruncateTask> &task);
  clio::run::TaskResume Unlink(clio::run::shared_ptr<UnlinkTask> &task);
  clio::run::TaskResume Mkdir(clio::run::shared_ptr<MkdirTask> &task);
  clio::run::TaskResume Rmdir(clio::run::shared_ptr<RmdirTask> &task);
  clio::run::TaskResume Rename(clio::run::shared_ptr<RenameTask> &task);
  clio::run::TaskResume Link(clio::run::shared_ptr<LinkTask> &task);
  clio::run::TaskResume Symlink(clio::run::shared_ptr<SymlinkTask> &task);
  clio::run::TaskResume Readlink(clio::run::shared_ptr<ReadlinkTask> &task);
  clio::run::TaskResume Setxattr(clio::run::shared_ptr<SetxattrTask> &task);
  clio::run::TaskResume Getxattr(clio::run::shared_ptr<GetxattrTask> &task);
  clio::run::TaskResume Listxattr(clio::run::shared_ptr<ListxattrTask> &task);
  clio::run::TaskResume Removexattr(clio::run::shared_ptr<RemovexattrTask> &task);
  clio::run::TaskResume Utimens(clio::run::shared_ptr<UtimensTask> &task);
  clio::run::TaskResume Chown(clio::run::shared_ptr<ChownTask> &task);
  clio::run::TaskResume Readdir(clio::run::shared_ptr<ReaddirTask> &task);
  clio::run::TaskResume StatSize(clio::run::shared_ptr<StatSizeTask> &task);
  // ---- deferred-append pipeline ----
  clio::run::TaskResume AppendSequence(clio::run::shared_ptr<AppendSequenceTask> &task);
  clio::run::TaskResume AppendCollect(clio::run::shared_ptr<AppendCollectTask> &task);
  clio::run::TaskResume AppendPlan(clio::run::shared_ptr<AppendPlanTask> &task);
  clio::run::TaskResume AppendExecution(clio::run::shared_ptr<AppendExecutionTask> &task);

  // ---- Container virtuals (defined in autogen/filesystem_lib_exec.cc) ----
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
  // Global store for per-file extended attributes. Each file's xattrs live in
  // ONE serialized blob under this tag, named by the file's packed tag id
  // (decimal string). Kept OUT of the file's own tag so xattrs never inflate
  // GetTagSize (i.e. the reported st_size). Resolved once at Create.
  clio::cte::core::TagId xattr_tag_id_ = clio::cte::core::TagId::GetNull();

  // ---- per-file logical-size metadata + handle table ----
  struct FileInfo {
    clio::cte::core::TagId tag_id_;
    std::string path_;
    std::atomic<clio::run::u64> size_{0};  // logical size
    // utimens overrides (ns; 0 = not set, defer to the core tag's timestamp).
    // Guarded by meta_mu_. Cleared by a later write/truncate (content change
    // re-establishes the natural mtime) so the override never goes stale.
    clio::run::u64 set_atime_{0};
    clio::run::u64 set_mtime_{0};
    clio::run::u64 set_ctime_{0};
    // chown overrides. 0xFFFFFFFF = never chown'd (getattr defers to the
    // adapter's getuid()/getgid() default). Guarded by meta_mu_.
    clio::run::u32 set_uid_{0xFFFFFFFFu};
    clio::run::u32 set_gid_{0xFFFFFFFFu};
    // chmod / create mode override (permission bits). 0xFFFFFFFF = no stored
    // mode (getattr synthesizes 0644 for files, 0755 for dirs). Persists across
    // writes (unlike timestamps). Guarded by meta_mu_.
    clio::run::u32 set_mode_{0xFFFFFFFFu};
  };
  std::mutex meta_mu_;
  std::unordered_map<clio::run::u64, std::shared_ptr<FileInfo>> handles_;
  std::unordered_map<std::string, std::shared_ptr<FileInfo>> by_path_;
  // Tag-keyed index over the same FileInfo objects (see AdvanceSizeTask):
  // maintained wherever by_path_ gains or loses a file entry.
  std::unordered_map<clio::run::u64, std::shared_ptr<FileInfo>> by_tag_;
  std::atomic<clio::run::u64> next_handle_{1};

  // ---- issue #817: shared-memory attribute mirror ----
  // Lets a client resolve path -> {tag id, logical size} itself and then read
  // the file's page blobs straight out of the RAM bdev segment, with no round
  // trip at all. Pure cache: every mirror call is best-effort and happens
  // AFTER the authoritative update, so the mirror can only ever lag.
  ShmFsCache shm_fs_cache_;

  /**
   * Publish a path's current attributes.
   *
   * @param path absolute path (the map key, and the CTE tag name).
   * @param fi the authoritative entry, already updated.
   * @param extra_flags refusal flags to OR in (e.g. kShmFilePendingAppend).
   *
   * Caller may hold meta_mu_ or not: the mirror is independent of it, and the
   * SHM map has its own per-slot seqlock.
   */
  void MirrorFile(const std::string &path, const FileInfo &fi,
                  clio::run::u32 extra_flags = 0);

  /** Drop a path from the mirror (unlink/rename/rmdir). */
  void MirrorErase(const std::string &path) {
    shm_fs_cache_.ErasePath(path);
  }

  /**
   * Re-publish a path with refusal flags set, so in-flight clients stop
   * fast-pathing it. Used where the authoritative state is about to change in
   * a way that invalidates cached pages (truncate, rename) but the path lives
   * on. Erasing would work too; this keeps the tag binding for the next op.
   */
  void MirrorDir(const std::string &path,
                 const clio::cte::core::TagId &tag_id, bool complete);
  void MirrorRefuse(const std::string &path);

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
