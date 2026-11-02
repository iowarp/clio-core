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
#include <clio_cte/filesystem/fs_canonical.h>
#include <clio_cte/filesystem/fs_inode.h>
#include <clio_cte/filesystem/fs_inode_table.h>
#include <clio_cte/filesystem/fs_meta_log.h>
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
    clio::run::u64 parent_ = 0;              ///< directories: Inode::parent_
    std::string parent_key_;                 ///< directories: Inode::parent_key_
    clio::run::u32 volume_flags_ = 0;        ///< root: Inode::volume_flags_
  };

  /**
   * Append attributes to an encoding (shard replies and inode pushes).
   * @param e encoder
   * @param a attributes
   */
  static void EncodeAttr(FsEnc &e, const FsAttr &a);
  /**
   * Read what EncodeAttr wrote.
   * @param d decoder
   * @param a receives the attributes
   * @return false if malformed
   */
  static bool DecodeAttr(FsDec &d, FsAttr *a);

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
  /** Set once Create has bound cte_. The container is reachable before its
   *  Create runs (PoolManager registers it first); metadata I/O before then
   *  went to the null pool and never completed (#1125). */
  std::atomic<bool> cte_ready_{false};
  /** core GetBlobSize's rc when the tag or blob does not exist (any other
   *  non-zero rc is a failure to answer, not an answer). */
  static constexpr clio::run::u32 kCteBlobNotFoundRc = 1;
  /** How long metadata I/O waits for cte_ before failing with EIO. */
  static constexpr clio::run::u64 kCteReadyWaitMs = 120000;
  /** Re-check period while waiting for cte_. */
  static constexpr double kCteReadyPollUs = 1000.0;

  /**
   * Wait (bounded) until Create has bound the CTE client.
   * @param ok false if it is still unbound after kCteReadyWaitMs
   */
  clio::run::TaskResume AwaitCteReady(bool &ok);
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
  /** Rows an inode-table block holds before it splits. */
  clio::run::u32 split_entries_ = 1024;
  /** Names are compared case-insensitively (FilesystemConfig; fixed for the
   *  volume's life by the root inode's record). */
  bool case_insensitive_ = false;

  /**
   * The canonical name of one path component on this volume.
   * @param name the component as spelled
   * @return the inode-table key
   */
  std::string KeyOf(const std::string &name) const {
    return FsCanonicalName(name, case_insensitive_);
  }
  /**
   * The canonical form of a whole path on this volume.
   * @param path the path as spelled
   * @return the path every spelling of it maps to
   */
  std::string KeyOfPath(const std::string &path) const {
    return FsCanonicalPath(path, case_insensitive_);
  }
  /** Next holder registration number (blocks and inodes). */
  std::atomic<clio::run::u64> reg_seq_{1};
  /** When this container started (steady ms): a lease granted by an earlier
   *  incarnation runs out by started_ms_ + kCacheLeaseMs + slack. */
  clio::run::u64 started_ms_ = 0;

  // ---- inodes this container is home for ----
  /**
   * An inode resident on its home: its persisted attributes (the Inode
   * base, what its record holds) plus state that lives only in memory.
   * Everything is guarded by meta_mu_.
   */
  struct InodeState : Inode {
    /** The path the SHM mirror publishes it under (a name it was opened or
     *  created by); empty when it has none. Not persisted. */
    std::string mirror_path_;
    clio::run::u32 open_count_ = 0;  ///< live handles
    bool dirty_ = false;      ///< times changed since last logged
    /** Containers caching its attrs -> their registration number. */
    std::map<clio::run::u32, clio::run::u64> holders_;
    /** Loaded from its record: who caches it is unknown (the home restarted
     *  or moved), so the next store pushes to every container. */
    bool holders_unknown_ = false;
    /** When each holder's lease on the cached attributes runs out. */
    std::map<clio::run::u32, clio::run::u64> holder_lease_ms_;
    /** Wall time (ns) of the last explicit mtime set (utimens); a size push
     *  for a write before it keeps that mtime. Not persisted. */
    clio::run::u64 mtime_set_ns_ = 0;
  };
  std::mutex meta_mu_;  ///< guards handles_, inodes_ and InodeState fields
  std::unordered_map<clio::run::u64, std::shared_ptr<InodeState>> handles_;
  std::unordered_map<clio::run::u64, std::shared_ptr<InodeState>> inodes_;
  std::atomic<clio::run::u64> next_handle_{1};

  // ---- inode-table blocks (fs_inode_table.cc) ----
  /** Identifies one block: (directory id, block index). */
  struct TableBlockKey {
    clio::run::u64 dir_ = 0;
    clio::run::u32 index_ = 0;
    bool operator==(const TableBlockKey &o) const {
      return dir_ == o.dir_ && index_ == o.index_;
    }
  };
  /** Hash of a TableBlockKey. */
  struct TableBlockKeyHash {
    size_t operator()(const TableBlockKey &k) const {
      return static_cast<size_t>(FsMix64(k.dir_ ^ (static_cast<clio::run::u64>(
                                                       k.index_) << 40)));
    }
  };
  /**
   * One block as this container knows it: the authoritative copy when this
   * container is the block's home, otherwise a cached copy kept current by
   * the home's pushes.
   */
  struct TableBlockSlot {
    InodeTableBlock blk_;
    bool home_ = false;                ///< this container is the home
    clio::run::u32 home_id_ = 0;       ///< the home (cached copies)
    // ---- home only (ns_mu_) ----
    /** Containers caching a copy -> their registration number (a failed
     *  push only drops a holder that has not re-registered since). */
    std::map<clio::run::u32, clio::run::u64> holders_;
    std::vector<TableDelta> unpushed_;   ///< changes not yet pushed
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
    /** Home, directory journal: the journal holds no image this block's
     *  deltas build on (new, split off, materialized, or a journal write
     *  failed), so the next commit journals the whole block. */
    bool journal_image_ = true;
  };
  std::mutex ns_mu_;  ///< guards blocks_, loading_, early_ and every slot
  std::unordered_map<TableBlockKey, std::shared_ptr<TableBlockSlot>, TableBlockKeyHash> blocks_;
  /** IpcManager::HeldDeadSeq() when home blocks were last trusted. A change
   *  is a rejoin notice: a peer held this node dead and may have failed its
   *  homes over, so clean home blocks are dropped and reloaded. */
  std::atomic<clio::run::u64> seen_held_dead_seq_{0};
  /** Drop every clean home block (reload on next use); see above. */
  void DropHomeBlocksAfterFailover();
  /** Blocks being loaded here (a push that arrives meanwhile is kept),
   *  each with the step its loader is on -- named in a slow waiter's
   *  warning, so a load that never returns says where it is stuck. */
  std::unordered_map<TableBlockKey, const char *, TableBlockKeyHash> loading_;

  /**
   * Record which step the loader of `key` is on (static string).
   * @param key the block being loaded
   * @param stage what the loader is waiting on
   */
  void SetLoadStage(const TableBlockKey &key, const char *stage) {
    std::lock_guard<std::mutex> g(ns_mu_);
    auto it = loading_.find(key);
    if (it != loading_.end()) it->second = stage;
  }
  /** Pushes that arrived while their block was loading. */
  std::unordered_map<TableBlockKey, std::vector<TableDelta>, TableBlockKeyHash> early_;

  // ---- inode-table journal (fs_table_journal.cc) ----
  /** True when directory blocks persist in dir_log_ (metadata_log_path is
   *  set) instead of CTE blobs. */
  bool dir_journal_ = false;
  /** Set once OpenDirJournal has decided the journal (and replayed it): home
   *  block loads wait for it, as must anything a peer asks before Create. */
  std::atomic<bool> dir_ready_{false};
  /** Node-local journal of the blocks this container is home for. */
  FsMetaLog dir_log_;
  /** Orders appends against a compaction's snapshot + rewrite. Taken
   *  BEFORE ns_mu_, never while holding it. */
  std::mutex dir_log_mu_;
  /** Durable images replayed from the journal and not loaded yet (a home
   *  block moves to blocks_ on first use). Guarded by ns_mu_. */
  std::unordered_map<TableBlockKey, std::string, TableBlockKeyHash> journaled_;
  /** Steady ms of the journal's last periodic fsync. */
  clio::run::u64 dir_log_synced_ms_ = 0;

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
  /**
   * An inode's home container, before failover (its id's home bits, else
   * its hash).
   * @param packed inode id
   * @return the home container id
   */
  clio::run::u32 InodeHome(clio::run::u64 packed);
  /** @return container that is home for inode `packed`. */
  clio::run::u32 InodeOwner(clio::run::u64 packed);
  /**
   * Run ShardOp `op` on container `target`: inline when it is this
   * container, otherwise as a task sent to it.
   * @param target owner container
   * @param op FsShardOp code
   * @param req request
   * @param resp filled with the response (rc_ = EIO when unreachable)
   * @param node_lost if non-null, set true when the call failed because the
   *        target's node was lost with it in flight (the caller may resolve
   *        the target again: failover names the stand-in once it is dead)
   */
  clio::run::TaskResume CallShard(clio::run::u32 target, clio::run::u32 op,
                                  const FsReq &req, FsResp &resp,
                                  bool *node_lost = nullptr);
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

  // ---- inode-table blocks: homes and caches (fs_inode_table.cc) ----
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
                                  std::shared_ptr<TableBlockSlot> &out, int &rc);
  /**
   * Read block (dir, k) from its CTE blob (the home's first use).
   * @param dir packed directory id
   * @param k block index
   * @param out receives the block
   * @param rc 0, ENOENT (no blob) or EIO
   */
  clio::run::TaskResume ReadBlockBlob(clio::run::u64 dir, clio::run::u32 k,
                                      InodeTableBlock *out, int &rc);
  /**
   * Write a block's durable image to its CTE blob (on a non-volatile tier
   * when the deployment has one).
   * @param dir packed directory id
   * @param k block index
   * @param image EncodeTableBlock(block, true)
   * @param rc 0 or EIO
   */
  clio::run::TaskResume WriteBlockBlob(clio::run::u64 dir, clio::run::u32 k,
                                       const std::string &image, int &rc);
  /**
   * Read block (dir, k) from its CTE blob only (no journal lookup).
   * @param dir packed directory id
   * @param k block index
   * @param out receives the block
   * @param rc 0, ENOENT (no such blob) or EIO (any other failure)
   */
  clio::run::TaskResume ReadBlockCte(clio::run::u64 dir, clio::run::u32 k,
                                     InodeTableBlock *out, int &rc);
  /**
   * Store a block image in its CTE blob, padded to a power of two, on a
   * non-volatile tier when one accepts it (else RAM). With the directory
   * journal on this is the replicated mirror a stand-in reads (#1263).
   * @param dir packed directory id
   * @param k block index
   * @param image EncodeTableBlock(block, true)
   * @param rc 0, ENOSPC (store full) or EIO
   */
  clio::run::TaskResume PutBlockCte(clio::run::u64 dir, clio::run::u32 k,
                                    const std::string &image, int &rc);
  /**
   * Open this container's directory journal, then replay it (restart) or
   * empty it (fresh start). Leaves dir_journal_ false when `path` is empty
   * or cannot be opened, so blocks stay in CTE blobs.
   * @param path journal file (metadata_log_path + ".dirs." + container id)
   */
  void OpenDirJournal(const std::string &path);
  /**
   * Rebuild journaled_ from the journal: images, then each block's deltas
   * in order, then the durable form (pending entries dropped, leaving ones
   * live, unsealed), as a crash aborts the operations that set them.
   * @return number of records replayed
   */
  size_t ReplayDirJournal();
  /**
   * Rewrite the journal as one image per block this container is home for.
   * Caller holds dir_log_mu_ and not ns_mu_.
   */
  void CompactDirJournalLocked();
  /**
   * Append one record to the directory journal, compacting it when it has
   * grown past kDirJournalCompactBytes. Never call with ns_mu_ held.
   * @param type record type
   * @param payload encoded record
   * @return 0, ENOSPC or EIO
   */
  int JournalAppend(FsLogRec type, const std::string &payload);
  /** Blocks rebuilt while replaying the directory journal. */
  using TableBlockState = std::unordered_map<TableBlockKey, InodeTableBlock, TableBlockKeyHash>;
  /**
   * Apply one kBlockDeltas record during replay.
   * @param p payload (EncodeTableDeltas)
   * @param state blocks replayed so far
   * @return number of deltas that could not be applied
   */
  static size_t ReplayBlockDeltas(const std::string &p, TableBlockState *state);
  /**
   * Journal the deletion of block (dir, k) (rmdir) and, in the same step,
   * forget its replayed image and its sealed slot.
   * @param dir packed directory id
   * @param k block index
   * @return 0, ENOSPC or EIO
   */
  int JournalDropBlock(clio::run::u64 dir, clio::run::u32 k);
  /**
   * At the start of a commit (ns_mu_ held): take what this commit must make
   * durable. CTE blobs: the image when the block changed durably. Journal:
   * the whole image for a block with none journaled yet, else the deltas.
   * @param slot the block (home)
   * @param deltas the changes this commit pushes
   * @param image receives the CTE blob image (empty: none)
   * @param jtype receives the journal record type
   * @param jrec receives the journal record (empty: none)
   */
  void TakePersistLocked(TableBlockSlot &slot, const std::vector<TableDelta> &deltas,
                         std::string *image, FsLogRec *jtype,
                         std::string *jrec);
  /**
   * Hand the replayed image of block (dir, k) to its home's first load.
   * @param dir packed directory id
   * @param k block index
   * @param out receives the block
   * @return 0, ENOENT (no such block) or EIO (undecodable image)
   */
  int TakeJournaledBlock(clio::run::u64 dir, clio::run::u32 k, InodeTableBlock *out);
  /**
   * fsync the directory journal.
   * @return 0 or EIO
   */
  int SyncDirJournal();
  /** fsync the directory journal if it has unsynced records and the last
   *  sync is older than kDirJournalSyncMs (called from the drain tick). */
  void SyncDirJournalIfDue();
  /**
   * Blocks this container is home for according to its journal: replayed
   * but not loaded, and loaded home slots.
   * @param out receives the keys
   */
  void JournaledHomeKeys(std::vector<TableBlockKey> *out);
  /** Journal size that triggers a compaction (bytes). */
  static constexpr clio::run::u64 kDirJournalCompactBytes = 64ull << 20;
  /** Longest a journal record waits for its fsync on the drain tick (ms). */
  static constexpr clio::run::u64 kDirJournalSyncMs = 1000;
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
                                      std::shared_ptr<TableBlockSlot> slot,
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
                                   std::shared_ptr<TableBlockSlot> slot, int &lrc);
  /**
   * The containers a home block's next push goes to (ns_mu_ held).
   * @param slot the home block
   * @param resync true: every other container (holders unknown after a
   *        reload), with leases no shorter than this incarnation's horizon
   * @param holders out: containers to push to
   * @param leases out: their lease ends (steady ms), parallel to holders
   */
  /** Hold an abandoned, still in-flight push until it settles. */
  void ParkPush(clio::run::Future<ShardOpTask> &&f);
  std::mutex parked_mu_;
  std::vector<clio::run::Future<ShardOpTask>> parked_pushes_;
  void PushTargetsLocked(const TableBlockSlot &slot, bool resync,
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
  bool FindCachedBlock(const TableBlockKey &key, clio::run::u32 home,
                       std::shared_ptr<TableBlockSlot> *out,
                       std::shared_ptr<TableBlockSlot> *expired);
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
  void InstallLoadedBlock(const TableBlockKey &key, std::shared_ptr<TableBlockSlot> slot,
                          int lrc, std::shared_ptr<TableBlockSlot> *out);
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
  clio::run::TaskResume RevalidateBlock(std::shared_ptr<TableBlockSlot> slot,
                                        clio::run::u64 dir, clio::run::u32 k,
                                        clio::run::u32 home, int &rc);
  /**
   * Holder side of a push: apply the deltas in req.str_ to cached copies.
   * @return 0, or ENOENT when some block is not cached here (the home then
   *         stops pushing it here)
   */
  int ApplyTablePush(const FsReq &req);
  /**
   * Record a change to a home block (ns_mu_ held): bumps its version and
   * queues the delta for the next push.
   * @param slot the home block
   * @param delta the change (ops/header; versions, depth, sealed filled here)
   * @param persisted true if the change alters the block's durable image
   * @return the block's new version
   */
  clio::run::u64 RecordChangeLocked(TableBlockSlot &slot, TableDelta delta,
                                    bool persisted);
  /**
   * Make version `version` of a home block durable and pushed to every
   * holder, batching whatever else changed meanwhile (group commit).
   * @param slot home block
   * @param version version the caller needs committed
   * @param rc 0 or EIO (the blob write failed)
   */
  clio::run::TaskResume CommitBlock(std::shared_ptr<TableBlockSlot> slot,
                                    clio::run::u64 version, int &rc);
  /**
   * Send a push (shard op `op`, request `req`) to every holder of a cached
   * block or inode in parallel and wait for them. Holders that no longer
   * cache it, or that could not be reached before their lease ran out, are
   * returned in `gone`. A holder is never given up on while its lease is
   * valid: it could still serve the old copy.
   * @param op kTablePush or kShardInodePush
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
  /**
   * Insert / replace a row in a home block (ns_mu_ held). req.leaf_ is the
   * caller's spelling; the row is keyed by its canonical name.
   */
  int InsertRowLocked(TableBlockSlot &slot, const FsReq &req, FsResp &resp);
  /** Remove / mark / restore a row in a home block (ns_mu_ held). */
  int RemoveRowLocked(TableBlockSlot &slot, const FsReq &req, FsResp &resp);
  /**
   * Change the spelling of a row in a home block (ns_mu_ held): a rename
   * between two spellings of one name (req.leaf_ the new spelling, req.id_
   * the inode the row must still name).
   */
  int RespellRowLocked(TableBlockSlot &slot, const FsReq &req, FsResp &resp);
  /**
   * Home side of a row mutation (kTableInsert / kTableRemove /
   * kTableRespell): waits out busy rows, commits, splits the block when it
   * grew too big.
   */
  clio::run::TaskResume RowMutation(clio::run::u32 op, const FsReq &req,
                                    FsResp &resp);
  /** Split a home block that outgrew split_entries_. */
  clio::run::TaskResume SplitBlock(std::shared_ptr<TableBlockSlot> slot);
  /** Install a block handed over by a split (this container is its home). */
  clio::run::TaskResume InstallBlock(const FsReq &req, FsResp &resp);
  /** Create a new directory's table block 0 (mkdir stage two; repair too). */
  clio::run::TaskResume CreateTable(const FsReq &req, FsResp &resp);
  /** Seal (req.a_ = 1: fails ENOTEMPTY unless empty) or unseal a block. */
  clio::run::TaskResume SealBlock(const FsReq &req, FsResp &resp);
  /** Delete a sealed block and tell its holders to forget it. */
  clio::run::TaskResume DropBlock(const FsReq &req, FsResp &resp);
  /**
   * Seal every block of a directory (rmdir), or unseal them.
   * @param dir packed directory id
   * @param seal true to seal (fails ENOTEMPTY and unseals on a non-empty
   *        block), false to unseal
   * @param rc 0, ENOTEMPTY or EIO
   */
  clio::run::TaskResume SealTable(clio::run::u64 dir, bool seal, int &rc);
  /** Delete every (sealed) block of a directory. */
  clio::run::TaskResume DropTable(clio::run::u64 dir);
  /**
   * Every block index of a directory (walks the split tree through the
   * cache).
   * @param dir packed directory id
   * @param out receives the indices
   * @param rc 0, ENOENT or EIO
   */
  clio::run::TaskResume TableBlocks(clio::run::u64 dir,
                                    std::vector<clio::run::u32> *out, int &rc);
  /**
   * Snapshot of a directory's listing from this node's block copies, under
   * one lock (see CollectDir). Pending rows are hidden; a leaving row whose
   * inode is live under another name (a rename in progress) is too.
   * @param dir directory id
   * @param blocks the directory's block indices
   * @param slots the loaded copy of each block, in the same order
   * @param out receives (listed name, row) pairs in key order
   * @param newest receives the newest block mtime
   * @return false if a copy was replaced meanwhile (the caller retries)
   */
  bool SnapshotDir(clio::run::u64 dir,
                   const std::vector<clio::run::u32> &blocks,
                   const std::vector<std::shared_ptr<TableBlockSlot>> &slots,
                   std::vector<std::pair<std::string, InodeRow>> *out,
                   clio::run::u64 *newest);
  /**
   * List a directory (every block, through the cache).
   * @param dir packed directory id
   * @param out receives (listed name, row) in key order, pending ones left out
   * @param newest receives the newest block mtime
   * @param rc 0, ENOENT or EIO
   */
  clio::run::TaskResume CollectDir(
      clio::run::u64 dir,
      std::vector<std::pair<std::string, InodeRow>> *out,
      clio::run::u64 *newest, int &rc);
  /**
   * Stat a directory: its inode record's attributes, with mtime/ctime moved
   * up to its table's newest row change (see DirTimes).
   * @param dir packed directory id
   * @param resp receives attr_; rc_ ENOENT when its record or its table is
   *        missing
   */
  clio::run::TaskResume StatDir(clio::run::u64 dir, FsResp &resp);
  /**
   * The block holding canonical name `key` in `dir`, walking down the split
   * tree.
   * @param dir packed directory id
   * @param key canonical name (KeyOf)
   * @param slot receives the block
   * @param rc 0, ENOENT or EIO
   */
  clio::run::TaskResume WalkToBlock(clio::run::u64 dir, const std::string &key,
                                    std::shared_ptr<TableBlockSlot> &slot, int &rc);
  /**
   * Look up a name of `dir` through the cache.
   * @param dir packed directory id
   * @param name the name, as spelled (any spelling of it is found)
   * @param out receives the row
   * @param rc 0, ENOENT or EIO
   */
  clio::run::TaskResume LookupRow(clio::run::u64 dir, const std::string &name,
                                  InodeRow &out, int &rc);
  /**
   * Resolve an absolute path through the cache.
   * @param path normalized absolute path
   * @param ent receives the row ("/" resolves to the root)
   * @param parent receives the parent directory's id (0 for "/")
   * @param rc 0, ENOENT, ENOTDIR or EIO
   */
  clio::run::TaskResume ResolvePath(const std::string &path, InodeRow &ent,
                                    clio::run::u64 &parent, int &rc);
  /**
   * Run a row mutation on the home of the block holding (dir, KeyOf(leaf)),
   * re-walking when a split moved the key meanwhile.
   * @param op kTableInsert, kTableRemove or kTableRespell
   * @param req request (dir_id_ and leaf_ set; block_ filled here)
   * @param resp response
   */
  clio::run::TaskResume TableOp(clio::run::u32 op, FsReq req, FsResp &resp);
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
   * @param margin ids that must stay reserved past the next one
   * @param ok OUT false when the reservation could not be recorded (no id
   *        past the recorded one may then be minted)
   */
  clio::run::TaskResume EnsureIdReserve(clio::run::u32 margin, bool &ok);
  /**
   * Mint a new, never-reused inode id homed on `home` (null when none is
   * reserved). The major names this container as the minter, so ids minted
   * for one home by different containers never collide (#1158).
   * @param home the container that will serve the inode
   * @return the id, or null
   */
  clio::cte::core::TagId MintIdFor(clio::run::u32 home);
  /**
   * Mint an id homed on `home`, extending the durable id reservation first
   * when it has run out.
   * @param home the container that will serve the inode
   * @param id OUT the id (null when rc != 0)
   * @param rc OUT 0, or EIO when the reservation could not be recorded
   */
  clio::run::TaskResume MintIdReady(clio::run::u32 home,
                                    clio::cte::core::TagId &id, int &rc);
  /**
   * Store the record of a NEW inode before its name is published (#1158).
   * The inode's home (FsInodeHomeFor) is usually not the container that
   * inserts the name; it loads the inode from this record on first use
   * (EnsureInode). Stored first, a published name never leads to a missing
   * inode. When the home is this container the inode is made resident too.
   * @param ino the new inode: tag_id_, type_, mode_, symlink_ and (for a
   *        directory) parent_ / parent_key_ set; the times are set here
   * @param mirror_path the path it is created under (SHM mirror key)
   * @param rc OUT 0, or ENOSPC / EIO when the record could not be stored
   */
  clio::run::TaskResume CreateInodeRecord(Inode ino,
                                          const std::string &mirror_path,
                                          int &rc);
  /**
   * Undo CreateInodeRecord for an id whose name was never published (the
   * name was taken meanwhile, or the insert failed).
   * @param id the inode's id
   */
  clio::run::TaskResume DiscardInodeRecord(const clio::cte::core::TagId &id);
  /**
   * Create an inode and its first name: mint an id homed by the path (or
   * adopt the client-minted r.id_), store the inode's record, then insert
   * the row on its block's home. A name taken meanwhile yields the existing
   * row (EEXIST with kInsExcl) and the record is discarded.
   * @param r dir_id_, dir_ and leaf_ set; id_ a client-minted id or 0
   * @param type kFsTypeFile, kFsTypeSymlink or kFsTypeDir
   * @param mode permission bits
   * @param symlink target (symlinks)
   * @param flags kInsExcl, kInsPending (mkdir) or 0
   * @param er OUT the insert result; created_ != 0 when the name is new
   */
  clio::run::TaskResume CreateEntry(FsReq r, clio::run::u32 type,
                                    clio::run::u32 mode,
                                    const std::string &symlink,
                                    clio::run::u32 flags, FsResp &er);
  /**
   * The root directory's record, created by its home the first time the
   * volume is used: it carries the volume's flags (kVolume*) from then on.
   * @param rc OUT 0, or ENOSPC / EIO when it could not be stored
   */
  clio::run::TaskResume MaterializeRoot(int &rc);
  /**
   * Refuse a configuration that disagrees with the volume (the root's
   * record), on the root's home. Logs why.
   * @param ok OUT false when the volume was created with other flags
   */
  clio::run::TaskResume CheckVolumeFlags(bool &ok);
  /** @return the inode `packed` (nullptr if not homed here). */
  std::shared_ptr<InodeState> FindInode(clio::run::u64 packed);
  /** Fill `attr` from an inode (meta_mu_ held). */
  void InodeAttrLocked(const InodeState &fi, FsAttr *attr);

  /**
   * Set a file's mtime/ctime for a size push (AdvanceSize): the client's
   * write time when it sent one (never moving mtime backwards), else the
   * current time if the push grew the file or reserved an append.
   * @param task the push, after its size op ran (old_size_ is set)
   * @param fi the file, under meta_mu_
   */
  void StampSizePush(const AdvanceSizeTask &task, InodeState *fi);
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
  /**
   * After this node changed an inode homed elsewhere (chmod/chown/utimens),
   * replace its cached attributes with the home's reply, so the next stat
   * here sees the change instead of waiting for the home's async push.
   * @param packed the inode
   * @param attr the attributes the home returned
   */
  void RefreshCachedInode(clio::run::u64 packed, const FsAttr &attr);
  /** Set an inode's logical size and trim its pages. */
  clio::run::TaskResume InodeTruncate(const FsReq &req, FsResp &resp);
  /** getxattr/setxattr/listxattr/removexattr on an inode. */
  clio::run::TaskResume InodeXattr(const FsReq &req, FsResp &resp);
  /** Destroy inode `packed` now or when its last handle closes. */
  void DropInodeLocked(const std::shared_ptr<InodeState> &fi);
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
  /** Consecutive failed tag-name broadcasts; the batch is re-queued for up
   *  to kNameFlushRetries of them (#1182). */
  clio::run::u32 tn_flush_failures_ = 0;
  static constexpr clio::run::u32 kNameFlushRetries = 30;
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
  clio::run::u64 FileSize(const InodeState &fi);
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
  // Each inode's persistent attributes live in one xattr key (kInodeKey) of
  // the file's own tag: CTE persists and replicates it. The inode's home is
  // the only writer; every other container that stats the inode caches its
  // attributes and receives each change as a push before it is acknowledged.
  /**
   * Mark an inode's record for storing (meta_mu_ held). Every handler that
   * can dirty an inode calls FlushInodes() before it replies.
   * @param fi the inode
   */
  void MarkInodeDirtyLocked(const InodeState &fi);
  /**
   * Record an inode change (meta_mu_ held): its record is stored before the
   * change is acknowledged, and an orphan (unlinked while open) is kept in
   * this container's orphan record so a restart can destroy it.
   * @param fi the inode
   */
  void LogInode(const InodeState &fi);
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
  /**
   * Where an inode's record lives: its owner's CTE container (Local when the
   * owner is this container). EnsureInode reads it there.
   * @param packed the inode's packed id
   */
  clio::run::PoolQuery InodeRecQuery(clio::run::u64 packed);
  clio::run::TaskResume StoreInodeRec(clio::run::u64 packed,
                                      const std::string &rec, int &rc);
  /**
   * Store every dirty inode record and push it to its holders. Each dirty
   * inode is tried once per call: one that cannot be stored (a full store)
   * stays dirty for a later flush instead of being retried in a loop that
   * never returns.
   * @param err out (optional): 0, or the errno of the first record dirtied
   *        after @p since that could not be stored (ENOSPC for a full
   *        store, else EIO)
   * @param since DirtySeq() when the caller started its change; 0 flushes
   *        every dirty record (the periodic drain). Otherwise only records
   *        dirtied after it are stored, waited on and reported: an earlier
   *        record is another operation's change (one record stuck on a full
   *        store failed every unlink on its container, #1151, and one slow
   *        store made every Close queue behind it, #1149)
   */
  clio::run::TaskResume FlushInodes(int *err = nullptr,
                                    clio::run::u64 since = 0);
  /**
   * The current inode-dirty sequence number: every MarkInodeDirtyLocked
   * after this call stamps its inode with a larger one.
   * @return the sequence number
   */
  clio::run::u64 DirtySeq();
  /**
   * Whether an inode was dropped (nlink 0, no open) and its purge has not
   * yet deleted its record. EnsureInode must not load such a record: it
   * still says nlink 1, and an open that resolved the name just before a
   * rename replaced it brought the file back with no stream size, so a
   * read through it returned 0 bytes (#1150).
   * @param packed inode id
   * @return true while the inode is between drop and purge
   */
  bool IsDying(clio::run::u64 packed);
  /**
   * Whether an inode's latest change came after a caller's DirtySeq()
   * (meta_mu_ held).
   * @param packed inode id
   * @param since the caller's DirtySeq() when it started
   * @return true when the inode was dirtied after `since`
   */
  bool DirtiedAfterLocked(clio::run::u64 packed, clio::run::u64 since);
  /** Inodes dropped and queued for purge, until the purge ran (meta_mu_). */
  std::unordered_set<clio::run::u64> dying_;
  /**
   * Whether FlushInodes should skip a dirty record this round (meta_mu_
   * held): its last store failed under kInodeStoreRetryMs ago and the
   * caller did not change it. Every namespace op flushes, so without this
   * a record stuck on a full store was retried by each one.
   * @param packed inode id
   * @param since the caller's DirtySeq() at the start of its change
   * @return true to leave it for a later flush
   */
  bool StoreBackingOff(clio::run::u64 packed, clio::run::u64 since);
  /** How long a record whose store failed waits before another caller's
   *  flush retries it. */
  static constexpr clio::run::u64 kInodeStoreRetryMs = 1000;
  /** A Close slower than this logs how long each of its phases took. */
  static constexpr clio::run::u64 kSlowCloseMs = 5000;
  /** SteadyMs() of each dirty record's last failed store (meta_mu_). */
  std::unordered_map<clio::run::u64, clio::run::u64> inode_store_failed_ms_;
  /**
   * Make sure an inode this container homes is in memory: after a restart
   * inodes load lazily from their records.
   * @param packed inode id
   * @param err if not null, receives EIO when the record could not be read
   *        (its owner's node is down with no copy reachable, #1166; a lost
   *        block; a malformed record) -- as opposed to no record existing,
   *        which leaves it 0. A caller that finds no inode afterwards must
   *        answer EIO, not ENOENT, when it is set.
   */
  clio::run::TaskResume EnsureInode(clio::run::u64 packed, int *err = nullptr);
  /**
   * The inode record this home's first live successor stored while it
   * served the inode (this node down or cut off), if any.
   * @param packed inode
   * @param rec receives the record (empty: none)
   * @param from receives the container holding it
   */
  clio::run::TaskResume FetchStandInRecord(clio::run::u64 packed,
                                           std::string *rec,
                                           clio::run::u32 *from);
  /** Whether inode record a was written after b (ctime/mtime; an
   *  unreadable b loses). */
  static bool RecordNewer(const std::string &a, const std::string &b);
  /**
   * Stat an inode: the home's own copy, else this container's cached copy,
   * else a fetch from the home (which registers this container for pushes).
   * @param packed inode id
   * @param resp receives attr_ and str_ (symlink target); rc_ ENOENT if the
   *        inode does not exist
   */
  clio::run::TaskResume StatInode(clio::run::u64 packed, FsResp &resp);
  /**
   * Copy an inode's attributes into a Getattr reply.
   * @param a the attributes (StatRow / StatInode)
   * @param task the reply to fill (exists_ = 1)
   */
  static void FillGetattr(const FsAttr &a, GetattrTask *task);
  /** Re-resolves of a name whose inode vanished under a racing rename. */
  static constexpr int kNameRaceRetries = 8;
  /**
   * Open's first stage: create-or-find the name's entry (O_CREAT) or look
   * it up. Finishes the task itself (done = true) when the open ends here:
   * a missing name (handle 0), an error, or a directory.
   * @param task the Open task
   * @param name dir_id_ / dir_ / leaf_ of the name
   * @param er out: id_, type_, created_ of the entry
   * @param done out: true when the task's reply is final
   */
  clio::run::TaskResume OpenName(OpenTask *task, const FsReq &name,
                                 FsResp &er, bool &done);
  /** Holder side of an inode push (req.id_, attrs in req.str_; b_=1 drop). */
  int ApplyInodePush(const FsReq &req);
  /**
   * After a restart, reconcile every stream this node's stream container
   * restored from its log with its file's inode record, before the
   * filesystem serves anything: a file served elsewhere while this node was
   * down (a truncate there) must not come back with the size this node's
   * stream log remembers.
   */
  clio::run::TaskResume ReconcileRestoredStreams();
  std::unordered_set<clio::run::u64> inode_dirty_;    ///< meta_mu_
  /** Sequence number of each dirty inode's latest change (meta_mu_). */
  std::unordered_map<clio::run::u64, clio::run::u64> inode_dirty_seq_;
  clio::run::u64 dirty_seq_ = 0;  ///< meta_mu_; last number handed out
  std::unordered_set<clio::run::u64> inode_storing_;  ///< meta_mu_

  // ---- public-handler building blocks (filesystem_runtime.cc) ----
  /**
   * Stat the inode a row names.
   * @param parent packed id of the directory holding the row (0 for "/")
   * @param name its name there (to repair a directory lost to a crash)
   * @param e the row
   * @param resp receives the attributes
   */
  clio::run::TaskResume StatRow(clio::run::u64 parent,
                                const std::string &name, const InodeRow &e,
                                FsResp &resp);
  /**
   * Rename a non-directory (row move + nlink bookkeeping).
   * @param src source path
   * @param dst destination path
   * @param sp packed id of the source's directory
   * @param dp packed id of the destination's directory
   * @param se the source row
   * @param rc OUT 0 or an errno (EBUSY: crossed another rename, retry)
   */
  clio::run::TaskResume RenameFile(const std::string &src,
                                   const std::string &dst, clio::run::u64 sp,
                                   clio::run::u64 dp, const InodeRow &se,
                                   int &rc);
  /**
   * Rename a directory: one row moves; its subtree stays where it is (its
   * table is keyed by the directory's id, not its path).
   * @param src source path
   * @param dst destination path
   * @param sp packed id of the source's directory
   * @param dp packed id of the destination's directory
   * @param se the source row
   * @param rc OUT 0 or an errno (EBUSY: crossed another rename, retry)
   */
  clio::run::TaskResume RenameDir(const std::string &src,
                                  const std::string &dst, clio::run::u64 sp,
                                  clio::run::u64 dp, const InodeRow &se,
                                  int &rc);
  /**
   * Rename between two spellings of one name (same directory, same key):
   * the row stays, only its spelling changes.
   * @param src source path
   * @param dst destination path (same key as src)
   * @param dir packed id of the directory
   * @param se the row
   * @param rc OUT 0 or an errno
   */
  clio::run::TaskResume RespellName(const std::string &src,
                                    const std::string &dst, clio::run::u64 dir,
                                    const InodeRow &se, int &rc);
  /**
   * Remove an empty directory: seal its blocks, drop them, remove its row,
   * then drop its inode.
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
  void MirrorFile(const std::string &path, const InodeState &fi,
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
  kTableInsert = 2,      ///< insert / replace a row
  kTableRemove = 3,      ///< remove / mark / restore a row
  kTableCreate = 4,      ///< create a directory's table block 0
  kTableSeal = 5,        ///< seal (rmdir) or unseal one block
  kTableRespell = 6,     ///< change the spelling of a row
  kTableFetch = 7,       ///< send a block and register the caller for pushes
  kTablePush = 8,        ///< apply a home's deltas to cached blocks
  kTableInstall = 9,     ///< take over a block split off another home
  kTableDrop = 10,       ///< delete a sealed block (rmdir)
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

/** Insert flags (FsReq::flags_ of kTableInsert). */
enum : clio::run::u32 {
  kInsExcl = 1u,         ///< fail EEXIST when the name exists
  kInsReplace = 2u,      ///< replace an existing non-directory (rename)
  kInsReplaceDir = 4u,   ///< replace an existing (retired) directory row
  kInsFailBusy = 8u,     ///< EBUSY instead of waiting on a busy name/dir
  kInsNewInode = 16u,    ///< the caller stored the new inode's record (req.id_)
  kInsPending = 32u,     ///< insert invisible (mkdir reservation)
  kInsCommit = 64u,      ///< make a pending row live
  kInsNoTagName = 128u,  ///< do not publish the name (rename publishes once)
};

/** Remove flags (FsReq::flags_ of kTableRemove). */
enum : clio::run::u32 {
  kRmNonDir = 1u,        ///< EISDIR if the row names a directory
  kRmDirOnly = 2u,       ///< ENOTDIR if the row does not name a directory
  kRmMarkLeaving = 4u,   ///< only mark the row leaving (two-phase remove)
  kRmRestore = 8u,       ///< make a leaving row live again (abort)
  kRmFailBusy = 16u,     ///< EBUSY instead of waiting on a busy name
  kRmNoTagName = 32u,    ///< do not publish the removal (rename publishes once)
};

/** Attribute-change flags (FsReq::flags_ of kShardInodeSetAttr). */
enum : clio::run::u32 {
  kSetAtime = 1u,        ///< atime := req.a_
  kSetMtime = 2u,        ///< mtime := req.b_
  kSetAtimeNow = 4u,     ///< atime := now (the home's clock)
  kSetMtimeNow = 8u,     ///< mtime := now (the home's clock)
  kSetUid = 16u,         ///< uid := req.uid_
  kSetGid = 32u,         ///< gid := req.gid_
  kSetMode = 64u,        ///< mode := req.mode_
  /** A read: advance atime by the relatime rule, never ctime (see
   *  InodeSetAttr). Set from the UtimensTask flag kUtimensAccess. */
  kAccessTouch = 512u,
  kAccessStrict = 1024u, ///< with kAccessTouch: strictatime, always move
  /** A directory moved: parent_ := req.b_, parent_key_ := req.str_. */
  kSetParent = 2048u,
};

/** Table-create flags (FsReq::flags_ of kTableCreate). */
enum : clio::run::u32 {
  /** Create block 0 only if it is missing (a crash between rmdir's halves
   *  left the parent's row but not the table). */
  kCreateRepair = 1u,
};

/** xattr sub-ops (FsReq::type_ of kShardInodeXattr). */
enum : clio::run::u32 {
  kXattrGet = 1u, kXattrSet = 2u, kXattrList = 3u, kXattrRemove = 4u,
};

/** Internal "try again" result of an entry mutation that must wait. */
static constexpr int kFsRetry = -1;
/** A mutation reached a block that no longer holds the name (it split or
 *  its home moved): the caller re-walks and resends. */
static constexpr int kFsRedirect = 0x10000;

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_RUNTIME_H_
