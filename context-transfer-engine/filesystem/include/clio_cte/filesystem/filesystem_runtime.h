/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_
#define CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/filesystem/filesystem_client.h>
#include <clio_cte/filesystem/filesystem_tasks.h>
#include <clio_cte/filesystem/fs_meta_log.h>
#include <clio_cte/stream/stream_client.h>
#include <clio_cte/stream/stream_runtime.h>
#include <clio_cte/filesystem/fs_shard.h>
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
  clio::run::TaskResume ShardOp(clio::run::shared_ptr<ShardOpTask> &task);

  // ---- Container virtuals (defined in autogen/filesystem_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  /** Recovery start (`clio_run restart`): Create replays the metadata log. */
  void Restart(const clio::run::PoolId &pool_id, const std::string &pool_name,
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


  // ---- namespace-shard value types (public: the .cc's codec uses them) ----

  /** Attributes a stat reports, as carried between shards. */
  struct FsAttr {
    clio::run::u64 id_ = 0;                  ///< packed id (inode number)
    clio::run::u32 type_ = 0;                ///< kFsType*
    clio::run::u64 size_ = 0;                ///< logical size (symlink: target length)
    clio::run::u32 nlink_ = 1;               ///< hard-link count
    clio::run::u32 mode_ = 0xFFFFFFFFu;      ///< permission bits, all-ones = default
    clio::run::u32 uid_ = 0xFFFFFFFFu;       ///< owner, all-ones = default
    clio::run::u32 gid_ = 0xFFFFFFFFu;       ///< group, all-ones = default
    clio::run::u64 atime_ = 0;               ///< ns
    clio::run::u64 mtime_ = 0;               ///< ns
    clio::run::u64 ctime_ = 0;               ///< ns
  };

  /** Request of one ShardOp (a superset; each op reads what it needs). */
  struct FsReq {
    std::string dir_, leaf_, str_, str2_;
    clio::run::u64 id_ = 0, a_ = 0, b_ = 0;
    clio::run::u32 type_ = 0, flags_ = 0;
    clio::run::u32 mode_ = 0xFFFFFFFFu, uid_ = 0xFFFFFFFFu, gid_ = 0xFFFFFFFFu;
  };

  /** Response of one ShardOp. */
  struct FsResp {
    clio::run::u32 rc_ = 0;        ///< errno-style result
    clio::run::u64 id_ = 0;        ///< entry id (lookup / insert winner)
    clio::run::u32 type_ = 0;      ///< entry type
    clio::run::u64 old_id_ = 0;    ///< replaced / removed entry id
    clio::run::u32 old_type_ = 0;  ///< replaced / removed entry type
    clio::run::u32 created_ = 0;   ///< insert created a new entry
    clio::run::u64 handle_ = 0;    ///< open handle
    FsAttr attr_;                  ///< stat result
    std::string str_;              ///< symlink target / xattr value / blob
  };

 private:
  // CTE core client this filesystem sits over (set at Create from next_pool_id_).
  clio::cte::core::Client cte_;
  clio::run::PoolId next_pool_id_ = clio::run::PoolId::GetNull();
  // Client bound to THIS filesystem pool, for self-submitted tasks (the
  // append pipeline, and ShardOps to the owners of other namespace state).
  Client self_;
  // Global store for per-file extended attributes. Each file's xattrs live in
  // ONE serialized blob under this tag, named by the file's packed tag id
  // (decimal string). Kept OUT of the file's own tag so xattrs never inflate
  // GetTagSize (i.e. the reported st_size). Resolved once at Create.
  clio::cte::core::TagId xattr_tag_id_ = clio::cte::core::TagId::GetNull();
  // Every file is a stream (clio::cte::stream): the stream pool owns its
  // logical size and merges its deferred appends. A file's stream home is
  // its inode home, so size reads are in-process (StreamLocal).
  clio::cte::stream::Client stream_;
  /** This node's stream container (resolved in Create; empty if absent). */
  clio::run::ContainerHold stream_hold_;

  // ---- inodes this container is home for ----
  struct FileInfo {
    clio::cte::core::TagId tag_id_;
    std::string path_;                       ///< a current name (mirror only)
    // Everything below is guarded by meta_mu_.
    clio::run::u32 type_ = kFsTypeFile;      ///< file or symlink
    clio::run::u32 nlink_ = 1;
    clio::run::u32 mode_ = 0xFFFFFFFFu;      ///< all-ones = adapter default
    clio::run::u32 uid_ = 0xFFFFFFFFu;
    clio::run::u32 gid_ = 0xFFFFFFFFu;
    clio::run::u64 atime_ = 0, mtime_ = 0, ctime_ = 0;
    std::string symlink_;                    ///< symlink target
    clio::run::u32 open_count_ = 0;          ///< live handles
    bool orphan_ = false;     ///< last name gone while open: purge at close
    bool has_xattr_ = false;  ///< an xattr blob may exist
    bool dirty_ = false;      ///< times changed since last logged
  };
  std::mutex meta_mu_;  ///< guards handles_, by_tag_ and FileInfo fields
  std::unordered_map<clio::run::u64, std::shared_ptr<FileInfo>> handles_;
  std::unordered_map<clio::run::u64, std::shared_ptr<FileInfo>> by_tag_;
  std::atomic<clio::run::u64> next_handle_{1};

  // ---- directories this container owns (hash of the dir path) ----
  /** Entry states: a pending entry is invisible, a leaving one is visible;
   *  both make other mutations of that name wait (or fail EBUSY). */
  enum : clio::run::u32 { kEntLive = 0, kEntPending = 1, kEntLeaving = 2 };
  struct Dentry {
    clio::cte::core::TagId id_;
    clio::run::u32 type_ = kFsTypeFile;
    clio::run::u32 state_ = kEntLive;
  };
  struct DirState {
    clio::cte::core::TagId id_;
    std::map<std::string, Dentry> ents_;     ///< sorted: stable readdir
    clio::run::u32 mode_ = 0xFFFFFFFFu, uid_ = 0xFFFFFFFFu, gid_ = 0xFFFFFFFFu;
    clio::run::u64 atime_ = 0, mtime_ = 0, ctime_ = 0;
    bool moving_ = false;    ///< a rename is moving this subtree
    bool retiring_ = false;  ///< an rmdir is removing it
  };
  std::mutex ns_mu_;  ///< guards dirs_ and every DirState (taken before meta_mu_)
  std::unordered_map<std::string, std::shared_ptr<DirState>> dirs_;

  // ---- persistence of this container's shard ----
  FsMetaLog log_;
  std::string log_path_;
  bool is_restart_ = false;           ///< set by Restart() before Create
  clio::run::u32 next_minor_ = 1;     ///< next id minor (guarded by meta_mu_)
  clio::run::u32 minted_hi_ = 0;      ///< logged reservation (guarded by meta_mu_)
  clio::run::u32 num_containers_ = 0; ///< pool size, fetched lazily

  // ---- deferred data purge (unlinked files' pages + xattrs) ----
  struct PurgeItem {
    clio::cte::core::TagId id_;
    bool data_ = false;
    bool xattr_ = false;
  };
  std::mutex purge_mu_;
  std::vector<PurgeItem> purge_pending_;
  bool purge_started_ = false;
  // Serializes xattr read-modify-write per inode (ids in flight).
  std::mutex xattr_mu_;
  std::unordered_set<clio::run::u64> xattr_busy_;

  // ---- shard ownership + transport (fs_namespace.cc) ----
  /** @return number of containers in this pool (== nodes). */
  clio::run::u32 NumContainers();
  /** @return container owning directory `dir`. */
  clio::run::u32 DirOwner(const std::string &dir);
  /** @return container that is home for inode `packed`. */
  clio::run::u32 InodeOwner(clio::run::u64 packed);
  /**
   * Run ShardOp `op` on container `target`: inline when it is this
   * container, otherwise as a task sent to it.
   * @param target owner container
   * @param op FsShardOp code
   * @param req request
   * @param resp filled with the response (rc_ = EIO when unreachable)
   */
  clio::run::TaskResume CallShard(clio::run::u32 target, clio::run::u32 op,
                                  const FsReq &req, FsResp &resp);
  /** Execute ShardOp `op` against this container's state. */
  clio::run::TaskResume ExecShardOp(clio::run::u32 op, const FsReq &req,
                                    FsResp &resp);

  // ---- directory state (fs_namespace.cc; caller holds ns_mu_ where noted) ----
  /** Lookup `leaf` in `dir` (ns_mu_ held). @return 0, ENOENT or ENOTDIR */
  int LookupLocked(const std::string &dir, const std::string &leaf,
                   Dentry *out);
  /** Insert / replace an entry (see kShardInsert). @return rc or kFsRetry */
  int InsertEntry(const FsReq &req, FsResp &resp);
  /** Remove an entry (see kShardRemove). @return rc or kFsRetry */
  int RemoveEntry(const FsReq &req, FsResp &resp);
  /** Create a directory's state (mkdir, stage two). */
  int DirCreate(const FsReq &req);
  /** Remove an empty directory's state (rmdir, stage two). */
  int DirRetire(const FsReq &req);
  /** Get or set a directory's attributes. */
  int DirAttrOp(const FsReq &req, FsResp &resp);
  /** Serialize (and mark moving) a directory's state for a rename. */
  int DirExport(const FsReq &req, FsResp &resp);
  /** Install a directory's state under its new path. */
  int DirImport(const FsReq &req);
  /** Drop a directory's state (after it moved) or clear its moving mark. */
  int DirDropOrUnmark(const FsReq &req, bool drop);
  /** Fill `attr` from a directory state (ns_mu_ held). */
  static void DirAttrLocked(const DirState &ds, FsAttr *attr);
  /** Stamp a directory's mtime+ctime after an entry change (ns_mu_ held). */
  static void TouchDirLocked(DirState &ds);

  // ---- inodes (fs_namespace.cc) ----
  /** Mint a new, never-reused inode id homed here. */
  clio::cte::core::TagId MintId();
  /** Create the inode for a new file/symlink (meta_mu_ NOT held). */
  std::shared_ptr<FileInfo> NewInode(const clio::cte::core::TagId &id,
                                     clio::run::u32 type, clio::run::u32 mode,
                                     const std::string &symlink,
                                     const std::string &path);
  /** @return the inode `packed` (nullptr if not homed here). */
  std::shared_ptr<FileInfo> FindInode(clio::run::u64 packed);
  /** Fill `attr` from an inode (meta_mu_ held). */
  void InodeAttrLocked(const FileInfo &fi, FsAttr *attr);
  /** Open inode: allocate a handle (O_TRUNC is applied by the caller). */
  int InodeOpenLocal(const FsReq &req, FsResp &resp);
  /**
   * Add `delta` (req.a_) to nlink; queues the purge when the last link and
   * handle go. req.str_ names the link: a new name (delta > 0) becomes the
   * inode's primary name, a removed one (delta < 0) stops being it.
   */
  int InodeNlink(const FsReq &req, FsResp &resp);
  /** chmod / chown / utimens on an inode. */
  int InodeSetAttr(const FsReq &req, FsResp &resp);
  /** Set an inode's logical size and trim its pages. */
  clio::run::TaskResume InodeTruncate(const FsReq &req, FsResp &resp);
  /** getxattr/setxattr/listxattr/removexattr on an inode. */
  clio::run::TaskResume InodeXattr(const FsReq &req, FsResp &resp);
  /** Destroy inode `packed` now or when its last handle closes. */
  void DropInodeLocked(const std::shared_ptr<FileInfo> &fi);
  /** Queue the data purge of a destroyed inode. */
  void QueuePurge(const PurgeItem &item);
  /** Drain queued purges (periodic ShardOp body). */
  clio::run::TaskResume PurgeDrain();
  /**
   * Delete, on THIS container only, the pages of every id in `req.str_`
   * (packed u64s). One such request per container replaces a cluster-wide
   * broadcast per deleted file.
   */
  clio::run::TaskResume PurgeLocal(const FsReq &req);
  /** Start the periodic purge drain once (call from a task body). */
  void EnsurePurgeDrain();

  // ---- CTE tag-name publisher (async mirror of the namespace) ----
  // Every file and directory is also a CTE tag named "$tagid{parent}/leaf",
  // so CTE search (TagQuery / BlobQuery / SemanticSearch, the indexer) sees
  // paths. The directory owner stays authoritative; name changes are queued
  // here and broadcast in batches by the periodic drain.
  std::mutex tn_mu_;             ///< guards tn_batch_ (taken after ns_mu_)
  std::string tn_batch_;         ///< EncodeTagNameOp records not yet sent
  bool catchup_pending_ = false; ///< restart: rebuild names on this node
  int catchup_attempts_ = 0;
  std::unordered_set<clio::run::u32> catchup_missing_;  ///< peers not yet heard
  /**
   * Queue one name operation for broadcast.
   * @param op operation
   * @param id tag id (the file/dir id)
   * @param name stored name ("$tagid{parent}/leaf")
   * @param name2 new stored name (rename only)
   */
  void PublishName(clio::cte::core::TagNameOp op,
                   const clio::cte::core::TagId &id, const std::string &name,
                   const std::string &name2 = std::string());
  /** Broadcast queued name operations (drain body). */
  clio::run::TaskResume FlushNames();
  /** Encode every live entry this container owns as kAddName records. */
  std::string EncodeShardNames();

  // ---- file sizes (owned by the stream pool) ----
  /**
   * Logical size of an inode this container is home for (no network).
   * @param fi the inode
   * @return symlink target length, or the file's stream size
   */
  clio::run::u64 FileSize(const FileInfo &fi);
  /**
   * Change a file's size at its stream (this container is its home).
   * @param tag file tag
   * @param op size operation
   * @param value operand
   * @param old_size receives the size before (may be null)
   * @param new_size receives the size after (may be null)
   * @param rc receives the return code
   */
  clio::run::TaskResume FileSizeOp(clio::cte::core::TagId tag,
                                   clio::cte::stream::StreamSizeOp op,
                                   clio::run::u64 value,
                                   clio::run::u64 *old_size,
                                   clio::run::u64 *new_size, clio::run::u32 *rc);
  /** Restart catch-up: pull every peer's names into this node (drain body). */
  clio::run::TaskResume CatchUpNames();

  // ---- persistence (fs_namespace.cc) ----
  /** Encode a directory state record. */
  static std::string EncDirPut(const std::string &path, const DirState &ds);
  /** Encode an inode record. */
  static std::string EncInode(const FileInfo &fi);
  /** Append a directory state record (ns_mu_ held). */
  void LogDirPut(const std::string &path, const DirState &ds);
  /** Append an entry put/delete record (ns_mu_ held). */
  void LogEnt(bool put, const std::string &dir, const std::string &leaf,
              const Dentry &e, const DirState &ds);
  /** Append an inode record (meta_mu_ held). */
  void LogInode(const FileInfo &fi);

  // ---- inode records in CTE (fs_inode.cc) ----
  // Each inode's persistent attributes live in a small blob (kInodeBlob) of
  // the file's own tag: CTE persists and replicates it, the cache chimod
  // keeps coherent copies on reading nodes, and any node can stat the file
  // from it. The inode's home keeps a write-through copy (FileInfo) and is
  // the only writer; open-handle state stays home-only.
  /**
   * Mark an inode's record for storing (meta_mu_ held). Every handler that
   * can dirty an inode calls FlushInodes() before it replies.
   * @param fi the inode
   */
  void MarkInodeDirtyLocked(const FileInfo &fi);
  /** Store every dirty inode record (serialized per inode). */
  clio::run::TaskResume FlushInodes();
  /**
   * Make sure an inode this container homes is in memory: after a restart
   * inodes load lazily from their records.
   * @param packed inode id
   */
  clio::run::TaskResume EnsureInode(clio::run::u64 packed);
  /**
   * Stat an inode from its record (any container; served from the local
   * cached copy when there is one).
   * @param packed inode id
   * @param resp receives attr_ and str_ (symlink target); rc_ ENOENT if the
   *        record does not exist
   */
  clio::run::TaskResume ReadInodeRecord(clio::run::u64 packed, FsResp &resp);
  /**
   * Encode an inode record.
   * @param fi inode
   * @param size logical size to record
   * @return record bytes
   */
  static std::string EncInodeRec(const FileInfo &fi, clio::run::u64 size);
  /**
   * Decode an inode record.
   * @param rec record bytes
   * @param fi receives the attributes
   * @param size receives the recorded size
   * @return false if malformed
   */
  static bool DecInodeRec(const std::string &rec, FileInfo *fi,
                          clio::run::u64 *size);
  std::unordered_set<clio::run::u64> inode_dirty_;    ///< meta_mu_
  std::unordered_set<clio::run::u64> inode_storing_;  ///< meta_mu_
  /** Apply one replayed log record. */
  void ApplyLogRecord(FsLogRec type, const std::string &payload);
  /**
   * Open the log; on a restart replay and compact it, on a fresh start
   * discard it (the core's data is not being recovered either).
   * @param log_path log file ("" = volatile namespace)
   * @param replay true on a restart
   */
  void RecoverShard(const std::string &log_path, bool replay);
  /** Rewrite the log as a snapshot of live state (takes ns_mu_, meta_mu_). */
  void CompactLog();

  // ---- public-handler building blocks (filesystem_runtime.cc) ----
  /**
   * Resolve `path` to its entry (runs on the owner of its parent dir).
   * @param path absolute path
   * @param out receives the entry
   * @return 0, ENOENT or ENOTDIR
   */
  int ResolveLocal(const std::string &path, Dentry *out);
  /** Stat an entry wherever its attributes live (dir state or inode home). */
  clio::run::TaskResume StatEntry(const std::string &path, const Dentry &e,
                                  FsResp &resp);
  /** Rename a non-directory (entry move + nlink bookkeeping). */
  clio::run::TaskResume RenameFile(const std::string &src,
                                   const std::string &dst, const Dentry &se,
                                   int &rc);
  /** Rename a directory (moves every directory state of the subtree). */
  clio::run::TaskResume RenameDir(const std::string &src,
                                  const std::string &dst, const Dentry &se,
                                  int &rc);
  /**
   * Drop a replaced/unlinked entry's link on its inode home.
   * @param packed inode id
   * @param name the name that went away (the inode stops mirroring under
   *        it); empty for a link that had no name of its own
   */
  clio::run::TaskResume UnlinkInode(clio::run::u64 packed,
                                    const std::string &name);

  // ---- issue #817: shared-memory attribute mirror (single-node only) ----
  ShmFsCache shm_fs_cache_;
  /** Publish a file's attributes under `path` (meta_mu_ held). */
  void MirrorFile(const std::string &path, const FileInfo &fi,
                  clio::run::u32 extra_flags = 0);
  /** Drop a path from the mirror (unlink/rename/rmdir). */
  void MirrorErase(const std::string &path) { shm_fs_cache_.ErasePath(path); }
  /** Publish a directory (complete = its full listing is mirrored). */
  void MirrorDir(const std::string &path,
                 const clio::cte::core::TagId &tag_id, bool complete);
  /** Re-publish a path refusing the client fast path. */
  void MirrorRefuse(const std::string &path);

};

/** ShardOp codes (see Runtime::ExecShardOp). */
enum FsShardOp : clio::run::u32 {
  kShardLookup = 1,      ///< (dir, leaf) -> id, type
  kShardInsert = 2,      ///< insert / replace an entry, optionally minting its inode
  kShardRemove = 3,      ///< remove an entry
  kShardDirCreate = 4,   ///< create a directory's state
  kShardDirRetire = 5,   ///< remove an empty directory's state
  kShardDirAttr = 6,     ///< get / set a directory's attributes
  kShardDirExport = 7,   ///< serialize + mark a directory for a rename
  kShardDirImport = 8,   ///< install a moved directory's state
  kShardDirDrop = 9,     ///< drop a moved directory's old state
  kShardDirUnmark = 10,  ///< abort a move: clear the mark
  kShardInodeStat = 11,  ///< stat an inode
  kShardInodeOpen = 12,  ///< open an inode (handle)
  kShardInodeNlink = 13, ///< add to an inode's link count
  kShardInodeSetAttr = 14,  ///< chmod / chown / utimens
  kShardInodeTruncate = 15, ///< truncate an inode
  kShardInodeXattr = 16,    ///< xattr get/set/list/remove
  kShardPurgeDrain = 17,    ///< periodic: purge dead inodes' data
  kShardPurgeLocal = 18,    ///< drop this container's pages of a batch of ids
  kShardRepublish = 19,     ///< send this container's names to node req.a_
};

/** Insert flags (FsReq::flags_ of kShardInsert). */
enum : clio::run::u32 {
  kInsExcl = 1u,         ///< fail EEXIST when the name exists
  kInsReplace = 2u,      ///< replace an existing non-directory (rename)
  kInsReplaceDir = 4u,   ///< replace an existing (retired) directory entry
  kInsFailBusy = 8u,     ///< EBUSY instead of waiting on a busy name/dir
  kInsNewInode = 16u,    ///< mint + create the inode here (create/symlink)
  kInsPending = 32u,     ///< insert invisible (mkdir reservation)
  kInsCommit = 64u,      ///< make a pending entry live
  kInsNoTagName = 128u,  ///< do not publish the name (rename publishes once)
};

/** Remove flags (FsReq::flags_ of kShardRemove). */
enum : clio::run::u32 {
  kRmNonDir = 1u,        ///< EISDIR if the entry is a directory
  kRmDirOnly = 2u,       ///< ENOTDIR if the entry is not a directory
  kRmMarkLeaving = 4u,   ///< only mark the entry leaving (two-phase remove)
  kRmRestore = 8u,       ///< make a leaving entry live again (abort)
  kRmFailBusy = 16u,     ///< EBUSY instead of waiting on a busy name
  kRmNoTagName = 32u,    ///< do not publish the removal (rename publishes once)
};

/** xattr sub-ops (FsReq::type_ of kShardInodeXattr). */
enum : clio::run::u32 {
  kXattrGet = 1u, kXattrSet = 2u, kXattrList = 3u, kXattrRemove = 4u,
};
/** FsReq::flags_ bit of kShardInodeXattr: the id is a directory (no inode). */
static constexpr clio::run::u32 kXattrNoInode = 1u << 8;

/** Internal "try again" result of an entry mutation that must wait. */
static constexpr int kFsRetry = -1;

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_
