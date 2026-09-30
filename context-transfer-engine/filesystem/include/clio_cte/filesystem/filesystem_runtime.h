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

#ifndef CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_
#define CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_

#include <atomic>
#include <chrono>
#include <map>
#include <set>
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
#include <clio_cte/stream/stream_client.h>
#include <clio_cte/stream/stream_runtime.h>
#include <clio_cte/filesystem/fs_shard.h>
#include <clio_cte/filesystem/fs_dir_block.h>
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
  /**
   * fsync this container's namespace log (Method::kSyncMeta).
   * @param task sync task; rc 5 (EIO) if the fsync failed
   */
  clio::run::TaskResume SyncMeta(clio::run::shared_ptr<SyncMetaTask> &task);
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
  /** Recovering start (a plain `clio_run start`): Create replays the metadata log. */
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


  // ---- namespace value types (public: the .cc codecs use them) ----

  /** Attributes a stat reports, as carried between containers. */
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
  /** How long a node serves a cached directory block or inode attributes
   *  without asking the home (ms), measured from when it asked. */
  static constexpr clio::run::u64 kCacheLeaseMs = 10000;
  /** Extra the home waits past a lease (ms): the holder's clock started it
   *  earlier than the home's did. */
  static constexpr clio::run::u64 kCacheLeaseSlackMs = 2000;
  /** Steady-clock milliseconds (lease times). @return now */
  static clio::run::u64 SteadyMs() {
    return static_cast<clio::run::u64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  struct FsReq {
    std::string dir_, leaf_, str_, str2_;
    clio::run::u64 id_ = 0, a_ = 0, b_ = 0;
    clio::run::u32 type_ = 0, flags_ = 0;
    clio::run::u32 mode_ = 0xFFFFFFFFu, uid_ = 0xFFFFFFFFu, gid_ = 0xFFFFFFFFu;
    clio::run::u64 dir_id_ = 0;   ///< packed id of the directory addressed
    clio::run::u32 block_ = 0;    ///< directory block addressed
  };

  /** Response of one ShardOp. */
  struct FsResp {
    clio::run::u32 rc_ = 0;        ///< errno-style result (or kFsRedirect)
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
  // clio-fs's own small records (per-container id reservations and the
  // inodes unlinked while open), one blob each. Resolved once at Create.
  clio::cte::core::TagId sys_tag_id_ = clio::cte::core::TagId::GetNull();
  // Every file is a stream (clio::cte::stream): the stream pool owns its
  // logical size and merges its deferred appends. A file's stream home is
  // its inode home, so size reads are in-process (StreamLocal).
  clio::cte::stream::Client stream_;
  /** This node's stream container (resolved in Create; empty if absent). */
  clio::run::ContainerHold stream_hold_;
  /** Entries a directory block holds before it splits. */
  clio::run::u32 split_entries_ = 1024;
  /** Next holder registration number (blocks and inodes). */
  std::atomic<clio::run::u64> reg_seq_{1};
  /** When this container started (steady ms): a lease granted by an earlier
   *  incarnation runs out by started_ms_ + kCacheLeaseMs + slack. */
  clio::run::u64 started_ms_ = 0;

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
    /** Containers caching its attrs -> their registration number. */
    std::map<clio::run::u32, clio::run::u64> holders_;
    /** Loaded from its record: who caches it is unknown (the home restarted
     *  or moved), so the next store pushes to every container. */
    bool holders_unknown_ = false;
    /** When each holder's lease on the cached attributes runs out. */
    std::map<clio::run::u32, clio::run::u64> holder_lease_ms_;
  };
  std::mutex meta_mu_;  ///< guards handles_, by_tag_ and FileInfo fields
  std::unordered_map<clio::run::u64, std::shared_ptr<FileInfo>> handles_;
  std::unordered_map<clio::run::u64, std::shared_ptr<FileInfo>> by_tag_;
  std::atomic<clio::run::u64> next_handle_{1};

  // ---- directory blocks (fs_dirs.cc) ----
  /** Identifies one block: (directory id, block index). */
  struct BlockKey {
    clio::run::u64 dir_ = 0;
    clio::run::u32 index_ = 0;
    bool operator==(const BlockKey &o) const {
      return dir_ == o.dir_ && index_ == o.index_;
    }
  };
  /** Hash of a BlockKey. */
  struct BlockKeyHash {
    size_t operator()(const BlockKey &k) const {
      return static_cast<size_t>(FsMix64(k.dir_ ^ (static_cast<clio::run::u64>(
                                                       k.index_) << 40)));
    }
  };
  /**
   * One block as this container knows it: the authoritative copy when this
   * container is the block's home, otherwise a cached copy kept current by
   * the home's pushes.
   */
  struct BlockSlot {
    DirBlock blk_;
    bool home_ = false;                ///< this container is the home
    clio::run::u32 home_id_ = 0;       ///< the home (cached copies)
    // ---- home only (ns_mu_) ----
    /** Containers caching a copy -> their registration number (a failed
     *  push only drops a holder that has not re-registered since). */
    std::map<clio::run::u32, clio::run::u64> holders_;
    std::vector<DirDelta> unpushed_;   ///< changes not yet pushed
    clio::run::u64 durable_version_ = 0;  ///< written + pushed up to here
    bool persist_dirty_ = false;       ///< durable image changed
    bool committing_ = false;          ///< a commit is writing/pushing
    bool splitting_ = false;           ///< a split is moving entries out
    /** Loaded from the blob: who caches it is unknown (the home restarted
     *  or moved), so the next commit pushes a snapshot to every container
     *  and registers those that answer. */
    bool holders_unknown_ = false;
    /** Home: when each holder's lease runs out (steady ms, see
     *  kCacheLeaseMs); a change is acknowledged only once every holder got
     *  it or its lease has run out. */
    std::map<clio::run::u32, clio::run::u64> holder_lease_ms_;
    /** Copy: served without asking the home until then (steady ms). */
    clio::run::u64 lease_until_ms_ = 0;
  };
  std::mutex ns_mu_;  ///< guards blocks_, loading_, early_ and every slot
  std::unordered_map<BlockKey, std::shared_ptr<BlockSlot>, BlockKeyHash> blocks_;
  /** Blocks being loaded here (a push that arrives meanwhile is kept). */
  std::unordered_set<BlockKey, BlockKeyHash> loading_;
  /** Pushes that arrived while their block was loading. */
  std::unordered_map<BlockKey, std::vector<DirDelta>, BlockKeyHash> early_;

  // ---- inode attribute cache (inodes homed elsewhere) ----
  /** A cached stat of an inode homed on another container. */
  struct InodeCacheEnt {
    FsAttr attr_;
    std::string symlink_;
    clio::run::u32 home_ = 0;  ///< the home that registered this copy
    clio::run::u64 lease_until_ms_ = 0;  ///< served without asking until
  };
  std::mutex icache_mu_;  ///< guards icache_ and iloading_
  std::unordered_map<clio::run::u64, InodeCacheEnt> icache_;
  /** Inodes being fetched here, with the newest push that arrived meanwhile
   *  (home_ == ~0u when none did). */
  std::unordered_map<clio::run::u64, InodeCacheEnt> iloading_;

  // ---- system records ----
  bool is_restart_ = false;           ///< set by Restart() before Create
  clio::run::u32 next_minor_ = 1;     ///< next id minor (guarded by meta_mu_)
  clio::run::u32 minted_hi_ = 0;      ///< durably reserved up to (meta_mu_)
  bool reserving_ = false;            ///< a reservation write is in flight
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
  bool orphans_dirty_ = false;  ///< orphan record must be rewritten (purge_mu_)
  // Serializes xattr read-modify-write per inode (ids in flight).
  std::mutex xattr_mu_;
  std::unordered_set<clio::run::u64> xattr_busy_;

  // ---- ownership + transport (fs_namespace.cc) ----
  /** @return number of containers in this pool (== nodes). */
  clio::run::u32 NumContainers();
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
  /**
   * Send ShardOp `op` to container `target` without waiting (fan-out).
   * @param target container (may be this one)
   * @param op FsShardOp code
   * @param req request
   * @return the task's future
   */
  clio::run::Future<ShardOpTask> SendShard(clio::run::u32 target,
                                           clio::run::u32 op, const FsReq &req);
  /**
   * Decode the response of a completed SendShard.
   * @param f completed future
   * @param resp receives the response (rc_ = EIO when it failed)
   * @return false if the target was unreachable or answered garbage
   */
  static bool ReadShardResp(clio::run::Future<ShardOpTask> &f, FsResp *resp);
  /** Execute ShardOp `op` against this container's state. */
  clio::run::TaskResume ExecShardOp(clio::run::u32 op, const FsReq &req,
                                    FsResp &resp);
  /** Execute an inode ShardOp (the kShardInode* range). */
  clio::run::TaskResume ExecInodeOp(clio::run::u32 op, const FsReq &req,
                                    FsResp &resp, int &rc);

  // ---- directory blocks: homes and caches (fs_dirs.cc) ----
  /**
   * The container that owns block (dir, k): the owner of its CTE blob, or,
   * while that node is dead, the successor serving it.
   * @param dir packed directory id
   * @param k block index
   * @return home container
   */
  clio::run::u32 BlockHome(clio::run::u64 dir, clio::run::u32 k);
  /**
   * This container's copy of block (dir, k): the authoritative one when this
   * container is its home (read from the CTE on first use), else a cached
   * copy fetched from the home, which registers this container for pushes.
   * @param dir packed directory id
   * @param k block index
   * @param out receives the slot
   * @param rc 0, ENOENT (no such block/directory) or EIO
   */
  clio::run::TaskResume LoadBlock(clio::run::u64 dir, clio::run::u32 k,
                                  std::shared_ptr<BlockSlot> &out, int &rc);
  /**
   * Read block (dir, k) from its CTE blob (the home's first use).
   * @param dir packed directory id
   * @param k block index
   * @param out receives the block
   * @param rc 0, ENOENT (no blob) or EIO
   */
  clio::run::TaskResume ReadBlockBlob(clio::run::u64 dir, clio::run::u32 k,
                                      DirBlock *out, int &rc);
  /**
   * Write a block's durable image to its CTE blob (on a non-volatile tier
   * when the deployment has one).
   * @param dir packed directory id
   * @param k block index
   * @param image EncodeDirBlock(block, true)
   * @param rc 0 or EIO
   */
  clio::run::TaskResume WriteBlockBlob(clio::run::u64 dir, clio::run::u32 k,
                                       const std::string &image, int &rc);
  /**
   * Home side of a fetch: register the caller as a holder and return the
   * block (req.dir_id_, req.block_; req.a_ = the caller's container).
   */
  clio::run::TaskResume ServeBlockFetch(const FsReq &req, FsResp &resp);
  /**
   * Load a block this container is home of from its blob (materializing
   * "/"), jumping its version past every earlier incarnation's.
   * @param dir directory id
   * @param k block index
   * @param slot the new slot to fill
   * @param lrc 0 or an errno
   */
  clio::run::TaskResume LoadHomeBlock(clio::run::u64 dir, clio::run::u32 k,
                                      std::shared_ptr<BlockSlot> slot,
                                      int &lrc);
  /**
   * Fetch a copy of a block from its home, registering as a holder and
   * taking a lease on the copy.
   * @param dir directory id
   * @param k block index
   * @param home the block's home container
   * @param slot the new slot to fill
   * @param lrc 0, kFsRedirect or an errno
   */
  clio::run::TaskResume FetchBlock(clio::run::u64 dir, clio::run::u32 k,
                                   clio::run::u32 home,
                                   std::shared_ptr<BlockSlot> slot, int &lrc);
  /**
   * The containers a home block's next push goes to (ns_mu_ held).
   * @param slot the home block
   * @param resync true: every other container (holders unknown after a
   *        reload), with leases no shorter than this incarnation's horizon
   * @param holders out: containers to push to
   * @param leases out: their lease ends (steady ms), parallel to holders
   */
  void PushTargetsLocked(const BlockSlot &slot, bool resync,
                         std::vector<clio::run::u32> *holders,
                         std::vector<clio::run::u64> *leases);
  /**
   * Look a block up in this container's cache.
   * @param key the block
   * @param home the block's current home
   * @param out set when the cached slot can be used as is (the home's own,
   *        or a copy from that home within its lease)
   * @param expired set to a copy from that home whose lease ran out
   * @return true when out was set
   */
  bool FindCachedBlock(const BlockKey &key, clio::run::u32 home,
                       std::shared_ptr<BlockSlot> *out,
                       std::shared_ptr<BlockSlot> *expired);
  /**
   * Deregister holders a push found gone. Only the registration the push
   * went to is removed (a holder that fetched again meanwhile stays), and a
   * lease is forgotten only once it has run out.
   * @param gone holders PushToHolders gave up on
   * @param pushed the registrations (holder -> seq) the push went to
   * @param holders the block's / inode's current registrations
   * @param leases the matching lease ends (steady ms)
   */
  static void DropGoneHolders(
      const std::vector<clio::run::u32> &gone,
      const std::map<clio::run::u32, clio::run::u64> &pushed,
      std::map<clio::run::u32, clio::run::u64> *holders,
      std::map<clio::run::u32, clio::run::u64> *leases);
  /**
   * Publish a block LoadBlock just loaded (or failed to): clear the loading
   * mark, apply pushes that raced the fetch, and cache it on success.
   * @param key the block
   * @param slot the loaded slot
   * @param lrc the load's result (the slot is cached only when 0)
   * @param out set to slot on success
   */
  void InstallLoadedBlock(const BlockKey &key, std::shared_ptr<BlockSlot> slot,
                          int lrc, std::shared_ptr<BlockSlot> *out);
  /** FsResp::id_ of a fetch whose req.b_ matched the home's version. */
  static constexpr clio::run::u64 kFetchUnchanged = 1;
  /**
   * Renew the lease on a cached (non-home) block whose lease ran out: ask
   * the home whether it is still at our version and take its image if not.
   * @param slot the cached block
   * @param dir directory id
   * @param k block index
   * @param home the block's home container
   * @param rc 0 (copy current, lease renewed), kFsRedirect (not the home
   *           any more) or an errno (unreachable: never serve the copy)
   */
  clio::run::TaskResume RevalidateBlock(std::shared_ptr<BlockSlot> slot,
                                        clio::run::u64 dir, clio::run::u32 k,
                                        clio::run::u32 home, int &rc);
  /**
   * Holder side of a push: apply the deltas in req.str_ to cached copies.
   * @return 0, or ENOENT when some block is not cached here (the home then
   *         stops pushing it here)
   */
  int ApplyBlockPush(const FsReq &req);
  /**
   * Record a change to a home block (ns_mu_ held): bumps its version and
   * queues the delta for the next push.
   * @param slot the home block
   * @param delta the change (ops/header; versions, depth, sealed filled here)
   * @param persisted true if the change alters the block's durable image
   * @return the block's new version
   */
  clio::run::u64 RecordChangeLocked(BlockSlot &slot, DirDelta delta,
                                    bool persisted);
  /**
   * Make version `version` of a home block durable and pushed to every
   * holder, batching whatever else changed meanwhile (group commit).
   * @param slot home block
   * @param version version the caller needs committed
   * @param rc 0 or EIO (the blob write failed)
   */
  clio::run::TaskResume CommitBlock(std::shared_ptr<BlockSlot> slot,
                                    clio::run::u64 version, int &rc);
  /**
   * Send a push (shard op `op`, request `req`) to every holder of a cached
   * block or inode in parallel and wait for them. Holders that no longer
   * cache it, or that could not be reached before their lease ran out, are
   * returned in `gone`. A holder is never given up on while its lease is
   * valid: it could still serve the old copy.
   * @param op kShardBlockPush or kShardInodePush
   * @param req the push
   * @param holders containers caching it
   * @param leases each holder's lease end (steady ms), parallel to holders
   * @param gone out: holders to deregister
   */
  clio::run::TaskResume PushToHolders(
      clio::run::u32 op, const FsReq &req,
      const std::vector<clio::run::u32> &holders,
      const std::vector<clio::run::u64> &leases,
      std::vector<clio::run::u32> *gone);
  /** Insert / replace an entry in a home block (ns_mu_ held). */
  int InsertEntryLocked(BlockSlot &slot, const FsReq &req, FsResp &resp);
  /** Remove / mark / restore an entry in a home block (ns_mu_ held). */
  int RemoveEntryLocked(BlockSlot &slot, const FsReq &req, FsResp &resp);
  /**
   * Home side of an entry mutation (kShardInsert / kShardRemove): waits out
   * busy entries, commits, splits the block when it grew too big.
   */
  clio::run::TaskResume EntryMutation(clio::run::u32 op, const FsReq &req,
                                      FsResp &resp);
  /** Split a home block that outgrew split_entries_. */
  clio::run::TaskResume SplitBlock(std::shared_ptr<BlockSlot> slot);
  /** Install a block handed over by a split (this container is its home). */
  clio::run::TaskResume InstallBlock(const FsReq &req, FsResp &resp);
  /** Create a new directory's block 0 (mkdir stage two; repair too). */
  clio::run::TaskResume DirCreate(const FsReq &req, FsResp &resp);
  /** Seal (req.a_ = 1: fails ENOTEMPTY unless empty) or unseal a block. */
  clio::run::TaskResume SealBlock(const FsReq &req, FsResp &resp);
  /** Delete a sealed block and tell its holders to forget it. */
  clio::run::TaskResume DropBlock(const FsReq &req, FsResp &resp);
  /** Get or set a directory's attributes / parent pointer (block 0's home). */
  clio::run::TaskResume DirAttrOp(const FsReq &req, FsResp &resp);
  /**
   * Seal every block of a directory (rmdir), or unseal them.
   * @param dir packed directory id
   * @param seal true to seal (fails ENOTEMPTY and unseals on a non-empty
   *        block), false to unseal
   * @param rc 0, ENOTEMPTY or EIO
   */
  clio::run::TaskResume SealDir(clio::run::u64 dir, bool seal, int &rc);
  /** Delete every (sealed) block of a directory. */
  clio::run::TaskResume DropDir(clio::run::u64 dir);
  /**
   * Every block index of a directory (walks the split tree through the
   * cache).
   * @param dir packed directory id
   * @param out receives the indices
   * @param rc 0, ENOENT or EIO
   */
  clio::run::TaskResume DirBlocks(clio::run::u64 dir,
                                  std::vector<clio::run::u32> *out, int &rc);
  /**
   * Snapshot of a directory's listing from this node's block copies, under
   * one lock (see CollectDir). Pending entries are hidden; a leaving entry
   * whose inode is live under another name (a rename in progress) is too.
   * @param dir directory id
   * @param blocks the directory's block indices
   * @param slots the loaded copy of each block, in the same order
   * @param out receives (name, entry) pairs sorted by name
   * @param newest receives the newest block mtime
   * @return false if a copy was replaced meanwhile (the caller retries)
   */
  bool SnapshotDir(clio::run::u64 dir,
                   const std::vector<clio::run::u32> &blocks,
                   const std::vector<std::shared_ptr<BlockSlot>> &slots,
                   std::vector<std::pair<std::string, DirEntry>> *out,
                   clio::run::u64 *newest);
  /**
   * List a directory (every block, through the cache).
   * @param dir packed directory id
   * @param out receives (name, entry), sorted by name, pending ones left out
   * @param newest receives the newest block mtime
   * @param rc 0, ENOENT or EIO
   */
  clio::run::TaskResume CollectDir(
      clio::run::u64 dir,
      std::vector<std::pair<std::string, DirEntry>> *out,
      clio::run::u64 *newest, int &rc);
  /** Stat a directory (block 0's header + effective times). */
  clio::run::TaskResume DirStat(clio::run::u64 dir, FsResp &resp);
  /**
   * The block holding `leaf` in `dir`, walking down the split tree.
   * @param dir packed directory id
   * @param leaf entry name
   * @param slot receives the block
   * @param rc 0, ENOENT or EIO
   */
  clio::run::TaskResume WalkToBlock(clio::run::u64 dir, const std::string &leaf,
                                    std::shared_ptr<BlockSlot> &slot, int &rc);
  /**
   * Look up `leaf` in `dir` through the cache.
   * @param dir packed directory id
   * @param leaf entry name
   * @param out receives the entry
   * @param rc 0, ENOENT or EIO
   */
  clio::run::TaskResume LookupEntry(clio::run::u64 dir, const std::string &leaf,
                                    DirEntry &out, int &rc);
  /**
   * Resolve an absolute path through the cache.
   * @param path normalized absolute path
   * @param ent receives the entry ("/" resolves to the root)
   * @param parent receives the parent directory's id (0 for "/")
   * @param rc 0, ENOENT, ENOTDIR or EIO
   */
  clio::run::TaskResume ResolvePath(const std::string &path, DirEntry &ent,
                                    clio::run::u64 &parent, int &rc);
  /**
   * Run an entry mutation on the home of the block holding (dir, leaf),
   * re-walking when a split moved the name meanwhile.
   * @param op kShardInsert or kShardRemove
   * @param req request (dir_id_ and leaf_ set; block_ filled here)
   * @param resp response
   */
  clio::run::TaskResume EntryOp(clio::run::u32 op, FsReq req, FsResp &resp);
  /**
   * Refuse to move directory `moving` under `dst_parent` when that would
   * put it inside its own subtree, or when an ancestor is itself moving.
   * @param moving packed id of the directory being moved
   * @param dst_parent packed id of the destination's parent
   * @param rc 0, EINVAL (into its own subtree) or EBUSY (retry later)
   */
  clio::run::TaskResume CheckMoveTarget(clio::run::u64 moving,
                                        clio::run::u64 dst_parent, int &rc);
  /**
   * Encode every live entry of the blocks this container homes as kAddName
   * records (restart catch-up; lists this node's block blobs in the CTE).
   */
  clio::run::TaskResume EncodeHomeNames(std::string *out);

  // ---- inodes (fs_namespace.cc) ----
  /**
   * Make sure at least `margin` ids are durably reserved beyond the next
   * one (writes the reservation record when not).
   */
  clio::run::TaskResume EnsureIdReserve(clio::run::u32 margin);
  /** Mint a new, never-reused inode id homed here (0 when none reserved). */
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
  /** Rewrite this container's orphan record (inodes unlinked while open). */
  clio::run::TaskResume StoreOrphans();
  /**
   * Restart: read this container's id reservation and orphan records,
   * start minting past the reservation, and destroy the orphans (handles do
   * not survive a restart).
   */
  clio::run::TaskResume LoadSysRecords();

  // ---- CTE tag-name publisher (async mirror of the namespace) ----
  // Every file and directory is also a CTE tag named "$tagid{parent}/leaf",
  // so CTE search (TagQuery / BlobQuery / SemanticSearch, the indexer) sees
  // paths. The block homes stay authoritative; name changes are queued here
  // and broadcast in batches by the periodic drain.
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
  /** Restart catch-up: pull every peer's names into this node (drain body). */
  clio::run::TaskResume CatchUpNames();

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

  // ---- inode records in CTE + the attribute cache (fs_inode.cc) ----
  // Each inode's persistent attributes live in a small blob (kInodeBlob) of
  // the file's own tag: CTE persists and replicates it. The inode's home is
  // the only writer; every other container that stats the inode caches its
  // attributes and receives each change as a push before it is acknowledged.
  /**
   * Mark an inode's record for storing (meta_mu_ held). Every handler that
   * can dirty an inode calls FlushInodes() before it replies.
   * @param fi the inode
   */
  void MarkInodeDirtyLocked(const FileInfo &fi);
  /**
   * Record an inode change (meta_mu_ held): its record is stored before the
   * change is acknowledged, and an orphan (unlinked while open) is kept in
   * this container's orphan record so a restart can destroy it.
   * @param fi the inode
   */
  void LogInode(const FileInfo &fi);
  /** Set once a non-volatile placement of an inode record failed but a RAM
   *  one succeeded: the deployment has no persistent tier to put them on. */
  bool inode_volatile_only_ = false;
  /**
   * Store one inode record in its CTE tag (non-volatile tier first, RAM if
   * no tier accepts it).
   * @param packed the inode's packed tag id
   * @param rec the encoded record (EncInodeRec)
   * @param rc the PutBlob return code (0 on success)
   */
  clio::run::TaskResume StoreInodeRec(clio::run::u64 packed,
                                      const std::string &rec, int &rc);
  /** Store every dirty inode record and push it to its holders. */
  clio::run::TaskResume FlushInodes();
  /**
   * Make sure an inode this container homes is in memory: after a restart
   * inodes load lazily from their records.
   * @param packed inode id
   */
  clio::run::TaskResume EnsureInode(clio::run::u64 packed);
  /**
   * Stat an inode: the home's own copy, else this container's cached copy,
   * else a fetch from the home (which registers this container for pushes).
   * @param packed inode id
   * @param resp receives attr_ and str_ (symlink target); rc_ ENOENT if the
   *        inode does not exist
   */
  clio::run::TaskResume StatInode(clio::run::u64 packed, FsResp &resp);
  /** Holder side of an inode push (req.id_, attrs in req.str_; b_=1 drop). */
  int ApplyInodePush(const FsReq &req);
  /**
   * Encode an inode record.
   * @param fi inode
   * @param size logical size to record
   * @param writer the container storing the record
   * @return record bytes
   */
  static std::string EncInodeRec(const FileInfo &fi, clio::run::u64 size,
                                 clio::run::u32 writer);
  /**
   * Decode an inode record.
   * @param rec record bytes
   * @param fi receives the attributes
   * @param size receives the recorded size
   * @param writer receives the container that stored it (kNoRecWriter for a
   *        record from before the field existed)
   * @return false if malformed
   */
  static bool DecInodeRec(const std::string &rec, FileInfo *fi,
                          clio::run::u64 *size, clio::run::u32 *writer);
  /** DecInodeRec's writer for records that do not carry one. */
  static constexpr clio::run::u32 kNoRecWriter = 0xFFFFFFFFu;
  /**
   * After a restart, reconcile every stream this node's stream container
   * restored from its log with its file's inode record, before the
   * filesystem serves anything: a file served elsewhere while this node was
   * down (a truncate there) must not come back with the size this node's
   * stream log remembers.
   */
  clio::run::TaskResume ReconcileRestoredStreams();
  std::unordered_set<clio::run::u64> inode_dirty_;    ///< meta_mu_
  std::unordered_set<clio::run::u64> inode_storing_;  ///< meta_mu_

  // ---- public-handler building blocks (filesystem_runtime.cc) ----
  /**
   * Stat an entry wherever its attributes live (directory or inode).
   * @param parent packed id of the directory holding it (0 for "/")
   * @param leaf its name there (to repair a directory lost to a crash)
   * @param e the entry
   * @param resp receives the attributes
   */
  clio::run::TaskResume StatEntry(clio::run::u64 parent,
                                  const std::string &leaf, const DirEntry &e,
                                  FsResp &resp);
  /** Rename a non-directory (entry move + nlink bookkeeping). */
  clio::run::TaskResume RenameFile(const std::string &src,
                                   const std::string &dst, clio::run::u64 sp,
                                   clio::run::u64 dp, const DirEntry &se,
                                   int &rc);
  /** Rename a directory: one entry moves; its subtree stays where it is. */
  clio::run::TaskResume RenameDir(const std::string &src,
                                  const std::string &dst, clio::run::u64 sp,
                                  clio::run::u64 dp, const DirEntry &se,
                                  int &rc);
  /**
   * Remove an empty directory: seal its blocks, drop them, remove its entry.
   * @param parent packed id of its parent
   * @param leaf its name
   * @param expect its id (0 = whatever is there)
   * @param rc 0, ENOTEMPTY, ENOENT, ENOTDIR, EBUSY or EIO
   */
  clio::run::TaskResume RemoveDir(clio::run::u64 parent,
                                  const std::string &leaf,
                                  clio::run::u64 expect, int &rc);
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
  kShardInsert = 2,      ///< insert / replace an entry, optionally minting its inode
  kShardRemove = 3,      ///< remove an entry
  kShardDirCreate = 4,   ///< create a directory's block 0
  kShardBlockSeal = 5,   ///< seal (rmdir) or unseal one block
  kShardDirAttr = 6,     ///< get / set a directory's attributes
  kShardBlockFetch = 7,  ///< send a block and register the caller for pushes
  kShardBlockPush = 8,   ///< apply a home's deltas to cached blocks
  kShardBlockInstall = 9,   ///< take over a block split off another home
  kShardBlockDrop = 10,  ///< delete a sealed block (rmdir)
  kShardInodeStat = 11,  ///< stat an inode (req.b_ != 0: register req.a_ for pushes)
  kShardInodeOpen = 12,  ///< open an inode (handle)
  kShardInodeNlink = 13, ///< add to an inode's link count
  kShardInodeSetAttr = 14,  ///< chmod / chown / utimens
  kShardInodeTruncate = 15, ///< truncate an inode
  kShardInodeXattr = 16,    ///< xattr get/set/list/remove
  kShardPurgeDrain = 17,    ///< periodic: purge dead inodes' data
  kShardPurgeLocal = 18,    ///< drop this container's pages of a batch of ids
  kShardRepublish = 19,     ///< send this container's names to node req.a_
  kShardInodePush = 20,     ///< apply an inode home's attribute push
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
/** A mutation reached a block that no longer holds the name (it split or
 *  its home moved): the caller re-walks and resends. */
static constexpr int kFsRedirect = 0x10000;

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_
