/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Task definitions for the filesystem chimod — the on-the-wire contract for
 * every filesystem operation the interceptors delegate to.
 *
 * Serialization convention (mirrors the CTE core tasks, which is what the
 * runtime archives actually dispatch to): SerializeIn carries IN/INOUT fields
 * (plus any bulk-in payload) and MUST call Task::SerializeIn(ar) first;
 * SerializeOut carries OUT/INOUT fields (plus any bulk-out payload) and MUST
 * call Task::SerializeOut(ar) first. The archive operators only ever call
 * SerializeIn/SerializeOut — naming them anything else silently falls back to
 * the base Task serializer and OUT fields never travel back to the client.
 *
 * Handle model: a file handle is an opaque u64 the runtime assigns at Open;
 * it indexes a server-side table of {CTE tag, logical size}. Paths are mapped
 * to CTE tags exactly like the libfuse adapter; offsets map to 1 MiB
 * page-blobs ("0","1",...). The logical size kept per file makes getattr
 * exact and lets truncate/append work (CTE blobs alone can't express them).
 */
#ifndef CLIO_CTE_FILESYSTEM_FILESYSTEM_TASKS_H_
#define CLIO_CTE_FILESYSTEM_FILESYSTEM_TASKS_H_

#include <string>
#include <clio_ctp/util/msan.h>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/filesystem/autogen/filesystem_methods.h>

namespace clio::cte::filesystem {

/** Monitor reuses the admin task type (same as core/compressor chimods). */
using MonitorTask = clio::run::admin::MonitorTask;

/** 1 MiB page size, matching the libfuse adapter's blob paging. */
GLOBAL_CROSS_CONST clio::run::u64 kFsPageSize = 1024 * 1024;

/**
 * Page-blob name for a byte offset (stringified page index).
 *
 * Lives in the shared header, not in the runtime's .cc, because since issue
 * #817 a CLIENT resolves page blobs itself on the fast path. Two copies of
 * this rule would let the reader and the writer disagree about which blob a
 * byte range lives in -- the same reasoning that keeps MakeShmBlobKey single
 * sourced in the CTE core.
 */
inline std::string PageName(clio::run::u64 off) {
  return std::to_string(off / kFsPageSize);
}

/**
 * Well-known default pool id/name for the filesystem chimod, used by the
 * interceptor adapters (libfuse, POSIX, ...) to create-or-bind a single
 * shared filesystem pool over the default CTE core pool — exactly the way
 * clio::cte::core uses kCtePoolId/kCtePoolName.
 */
static constexpr clio::run::PoolId kCfsPoolId(560, 0);
static constexpr const char *kCfsPoolName = "clio_cte_filesystem";

/**
 * Container creation params. next_pool_id_ is the CTE core pool this
 * filesystem sits over (where its tags/blobs actually live).
 */
struct FilesystemConfig {
  static constexpr const char* chimod_lib_name = "clio_cte_filesystem";

  clio::run::PoolId next_pool_id_;  ///< CTE core pool id (e.g. 512.0)
  /**
   * Node-local file persisting this container's slice of the namespace
   * (directory entries/attributes and the inodes it is home for). Empty =
   * the namespace is volatile (lost on restart). "~" and $VARS expand; the
   * container id is appended so co-located containers never share a file.
   */
  std::string metadata_log_path_;
  /**
   * Stream pool that owns every file's logical size and merges deferred
   * appends (clio::cte::stream). Created with defaults (over next_pool_id_,
   * logging next to metadata_log_path_) if the deployment did not compose it.
   */
  clio::run::PoolId stream_pool_id_;
  /**
   * Entries a directory block holds before it splits in two (YAML
   * `dir_split_entries`). Small directories stay one block on one node; a
   * large one spreads over the cluster in blocks of about this many entries.
   */
  clio::run::u32 dir_split_entries_;

  FilesystemConfig()
      : next_pool_id_(clio::run::PoolId::GetNull()),
        stream_pool_id_(565, 0),
        dir_split_entries_(1024) {}
  FilesystemConfig(const clio::run::PoolId &pool_id, const FilesystemConfig &other)
      : next_pool_id_(other.next_pool_id_),
        metadata_log_path_(other.metadata_log_path_),
        stream_pool_id_(other.stream_pool_id_),
        dir_split_entries_(other.dir_split_entries_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_, metadata_log_path_, stream_pool_id_, dir_split_entries_);
  }

  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) {
      return;
    }
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      // yaml-cpp is a prebuilt .so: the scalars its scanner just
      // produced carry no MSan shadow, so every key lookup and
      // .as<>() below reads memory it has no record of. One walk
      // here covers the whole tree.
      ctp::MsanUnpoisonYaml(node);
      if (node["next_pool_id"]) {
        next_pool_id_ = clio::run::PoolId::FromString(
            node["next_pool_id"].as<std::string>());
      }
      if (node["metadata_log_path"]) {
        metadata_log_path_ = node["metadata_log_path"].as<std::string>();
      }
      if (node["stream_pool_id"]) {
        stream_pool_id_ = clio::run::PoolId::FromString(
            node["stream_pool_id"].as<std::string>());
      }
      if (node["dir_split_entries"]) {
        dir_split_entries_ = node["dir_split_entries"].as<clio::run::u32>();
      }
    } catch (...) {
      // best-effort
    }
  }
};

using CreateTask = clio::run::admin::GetOrCreatePoolTask<FilesystemConfig>;

/** DestroyTask — teardown (mirrors compressor). */
struct DestroyTask : public clio::run::Task {
  DestroyTask() : clio::run::Task() {}
  explicit DestroyTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}
  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(other.template Cast<clio::run::Task>()); (void)other; }
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** Open: resolve/create a file's tag; returns a handle + current size. */
struct OpenTask : public clio::run::Task {
  IN clio::run::priv::string path_;
  IN clio::run::u32 flags_;
  IN clio::run::u32 mode_;
  OUT clio::run::u64 handle_;
  OUT clio::run::u64 size_;
  OUT clio::run::u32 created_;  // 1 if the file was newly created
  // Packed TagId (major<<32|minor) of the file's tag, so adapters can drive
  // page-blob I/O (sieve writes, RYW reads) through the CTE client directly.
  OUT clio::run::u64 tag_packed_;

  OpenTask()
      : clio::run::Task(), path_(CTP_MALLOC), flags_(0), mode_(0644),
        handle_(0), size_(0), created_(0), tag_packed_(0) {}
  explicit OpenTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                    const clio::run::PoolQuery &pool_query, const std::string &path,
                    clio::run::u32 flags, clio::run::u32 mode)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kOpen),
        path_(CTP_MALLOC, path), flags_(flags), mode_(mode), handle_(0),
        size_(0), created_(0), tag_packed_(0) {}
  void Copy(const ctp::ipc::FullPtr<OpenTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; flags_ = o->flags_; mode_ = o->mode_;
    handle_ = o->handle_; size_ = o->size_; created_ = o->created_;
    tag_packed_ = o->tag_packed_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, flags_, mode_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(handle_, size_, created_, tag_packed_);
  }
};

/** Close: release a handle (drains pending writes server-side). */
struct CloseTask : public clio::run::Task {
  IN clio::run::u64 handle_;
  // Advance the file's logical size to at least this before releasing the
  // handle (0 = no advance). Carried on CLOSE — not a path-keyed truncate —
  // because it must survive a rename racing a detached close: the handle's
  // server-side FileInfo tracks the file identity across renames.
  IN clio::run::u64 advance_size_;
  CloseTask() : clio::run::Task(), handle_(0), advance_size_(0) {}
  explicit CloseTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                     const clio::run::PoolQuery &pool_query, clio::run::u64 handle,
                     clio::run::u64 advance_size = 0)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kClose),
        handle_(handle), advance_size_(advance_size) {}
  void Copy(const ctp::ipc::FullPtr<CloseTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    handle_ = o->handle_; advance_size_ = o->advance_size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(handle_, advance_size_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * MultiCreate (batched, sieve-flushed file creation): create N files in ONE
 * task. Each entry carries the path, the client-MINTED packed TagId (top-bit
 * minor partition; the tag chain adopts it via preferred_id), and the create
 * mode. create(2) itself returns without any task — the adapter's flusher
 * ships these every tick, exactly like the write sieve ships pages.
 */
struct MultiCreateEnt {
  std::string path_;
  clio::run::u64 tag_packed_ = 0;
  clio::run::u32 mode_ = 0644;
};

inline std::string EncodeMultiCreate(const std::vector<MultiCreateEnt> &ents) {
  std::string out;
  auto put32 = [&](clio::run::u32 v) {
    out.append(reinterpret_cast<const char *>(&v), sizeof(v));
  };
  auto put64 = [&](clio::run::u64 v) {
    out.append(reinterpret_cast<const char *>(&v), sizeof(v));
  };
  put32(static_cast<clio::run::u32>(ents.size()));
  for (const auto &e : ents) {
    put32(static_cast<clio::run::u32>(e.path_.size()));
    out.append(e.path_);
    put64(e.tag_packed_);
    put32(e.mode_);
  }
  return out;
}

inline bool DecodeMultiCreate(const char *data, size_t len,
                              std::vector<MultiCreateEnt> *out) {
  size_t off = 0;
  auto get32 = [&](clio::run::u32 *v) {
    if (off + sizeof(*v) > len) return false;
    std::memcpy(v, data + off, sizeof(*v));
    off += sizeof(*v);
    return true;
  };
  auto get64 = [&](clio::run::u64 *v) {
    if (off + sizeof(*v) > len) return false;
    std::memcpy(v, data + off, sizeof(*v));
    off += sizeof(*v);
    return true;
  };
  clio::run::u32 n = 0;
  if (!get32(&n)) return false;
  out->clear();
  out->reserve(n);
  for (clio::run::u32 i = 0; i < n; ++i) {
    MultiCreateEnt e;
    clio::run::u32 plen = 0;
    if (!get32(&plen) || off + plen > len) return false;
    e.path_.assign(data + off, plen);
    off += plen;
    if (!get64(&e.tag_packed_) || !get32(&e.mode_)) return false;
    out->push_back(std::move(e));
  }
  return true;
}

struct MultiCreateTask : public clio::run::Task {
  IN clio::run::priv::string packed_;  // EncodeMultiCreate payload
  OUT clio::run::u32 num_ok_;
  OUT clio::run::u32 first_rc_;
  /** Every entry that failed: (u32 index into the batch, u32 errno) pairs,
   *  little-endian. The client owes each failure to that file's fsync or
   *  close. */
  OUT clio::run::priv::string failed_;
  MultiCreateTask()
      : clio::run::Task(), packed_(CTP_MALLOC), num_ok_(0), first_rc_(0),
        failed_(CTP_MALLOC) {}
  explicit MultiCreateTask(const clio::run::TaskId &task_id,
                           const clio::run::PoolId &pool_id,
                           const clio::run::PoolQuery &pool_query,
                           const std::string &packed)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kMultiCreate),
        packed_(CTP_MALLOC, packed), num_ok_(0), first_rc_(0),
        failed_(CTP_MALLOC) {}
  void Copy(const ctp::ipc::FullPtr<MultiCreateTask> &o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    packed_ = o->packed_; num_ok_ = o->num_ok_; first_rc_ = o->first_rc_;
    failed_ = o->failed_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(packed_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(num_ok_, first_rc_, failed_);
  }
};

/**
 * AdvanceSize: raise a file's logical size, keyed by TAG rather than path or
 * handle. This is the rename-proof size push for sieve-minted files: a
 * path-keyed truncate delivered after the app renamed the file RESURRECTED
 * the old name (cfs Truncate materializes missing paths by design), and the
 * ghost then hijacked later renames. The tag names the file's identity; the
 * handler publishes under whatever path the file currently has.
 */
struct AdvanceSizeTask : public clio::run::Task {
  IN clio::run::u64 tag_packed_;
  /** Raise mode: the new size (never lowers it). Reserve mode: a length. */
  IN clio::run::u64 size_;
  /**
   * Nonzero = RESERVE: atomically grow the file by size_ bytes and return the
   * previous size in old_size_. This is the O_APPEND offset for a writer on a
   * node whose kernel cannot know the file's current end (another node may
   * have appended since); the reservation is linearized at the home.
   */
  IN clio::run::u32 reserve_;
  OUT clio::run::u64 old_size_;
  AdvanceSizeTask()
      : clio::run::Task(), tag_packed_(0), size_(0), reserve_(0),
        old_size_(0) {}
  explicit AdvanceSizeTask(const clio::run::TaskId &task_id,
                           const clio::run::PoolId &pool_id,
                           const clio::run::PoolQuery &pool_query,
                           clio::run::u64 tag_packed, clio::run::u64 size,
                           clio::run::u32 reserve = 0)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kAdvanceSize),
        tag_packed_(tag_packed), size_(size), reserve_(reserve),
        old_size_(0) {}
  void Copy(const ctp::ipc::FullPtr<AdvanceSizeTask> &o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    tag_packed_ = o->tag_packed_; size_ = o->size_; reserve_ = o->reserve_;
    old_size_ = o->old_size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(tag_packed_, size_, reserve_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(old_size_);
  }
};

/**
 * SyncMeta: fsync the namespace log (directory entries, directory state,
 * orphan inodes) so creates, renames and unlinks survive power loss.
 * Broadcast: every container fsyncs the log of the namespace shard it owns.
 */
struct SyncMetaTask : public clio::run::Task {
  SyncMetaTask() : clio::run::Task() {}
  explicit SyncMetaTask(const clio::run::TaskId &task_id,
                        const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kSyncMeta) {}
  void Copy(const ctp::ipc::FullPtr<SyncMetaTask> &o) {
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
  }
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** Read: page-loop GetBlob over [offset, offset+size). */
struct ReadTask : public clio::run::Task {
  IN clio::run::u64 handle_;
  IN clio::run::u64 offset_;
  IN clio::run::u64 size_;
  INOUT ctp::ipc::ShmPtr<> data_;
  OUT clio::run::u64 bytes_read_;
  ReadTask()
      : clio::run::Task(), handle_(0), offset_(0), size_(0),
        data_(ctp::ipc::ShmPtr<>::GetNull()), bytes_read_(0) {}
  explicit ReadTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                    const clio::run::PoolQuery &pool_query, clio::run::u64 handle,
                    clio::run::u64 offset, clio::run::u64 size, ctp::ipc::ShmPtr<> data)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kRead),
        handle_(handle), offset_(offset), size_(size), data_(data),
        bytes_read_(0) {}
  void Copy(const ctp::ipc::FullPtr<ReadTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    handle_ = o->handle_; offset_ = o->offset_; size_ = o->size_;
    data_ = o->data_; bytes_read_ = o->bytes_read_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(handle_, offset_, size_, data_);
    ar.bulk(data_, size_, BULK_EXPOSE);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(bytes_read_);
    ar.bulk(data_, size_, BULK_XFER);
  }
};

/** Write: page-loop PutBlob; advances the file's logical size. */
struct WriteTask : public clio::run::Task {
  IN clio::run::u64 handle_;
  IN clio::run::u64 offset_;
  IN clio::run::u64 size_;
  IN ctp::ipc::ShmPtr<> data_;
  OUT clio::run::u64 bytes_written_;
  OUT clio::run::u64 new_size_;  // logical size after the write
  WriteTask()
      : clio::run::Task(), handle_(0), offset_(0), size_(0),
        data_(ctp::ipc::ShmPtr<>::GetNull()), bytes_written_(0), new_size_(0) {}
  explicit WriteTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                     const clio::run::PoolQuery &pool_query, clio::run::u64 handle,
                     clio::run::u64 offset, clio::run::u64 size, ctp::ipc::ShmPtr<> data)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kWrite),
        handle_(handle), offset_(offset), size_(size), data_(data),
        bytes_written_(0), new_size_(0) {}
  void Copy(const ctp::ipc::FullPtr<WriteTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    handle_ = o->handle_; offset_ = o->offset_; size_ = o->size_;
    data_ = o->data_; bytes_written_ = o->bytes_written_;
    new_size_ = o->new_size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(handle_, offset_, size_, data_);
    ar.bulk(data_, size_, BULK_XFER);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(bytes_written_, new_size_);
  }
};


/** Getattr: exists / is-dir / logical size for a path. */
struct GetattrTask : public clio::run::Task {
  IN clio::run::priv::string path_;
  OUT clio::run::u32 exists_;
  OUT clio::run::u32 is_dir_;
  OUT clio::run::u64 size_;
  OUT clio::run::u64 ino_;     // stable inode = packed TagId (0 when nonexistent)
  OUT clio::run::u64 ctime_;   // tag change-time (ns); 0 if unknown
  OUT clio::run::u64 mtime_;   // tag modify-time (ns); 0 if unknown
  OUT clio::run::u64 atime_;   // tag access-time (ns); 0 if unknown
  OUT clio::run::u32 is_symlink_;  // 1 if the entry is a symlink (S_IFLNK)
  OUT clio::run::u32 uid_;  // chown'd owner uid; 0xFFFFFFFF = defer to default
  OUT clio::run::u32 gid_;  // chown'd owner gid; 0xFFFFFFFF = defer to default
  OUT clio::run::u32 mode_;  // chmod'd/created mode bits; 0xFFFFFFFF = default
  // POSIX link count (canonical name + hard-link aliases), computed
  // server-side so getattr costs ONE round trip — the adapter previously
  // issued a separate alias query per regular-file stat.
  OUT clio::run::u32 nlink_;
  GetattrTask()
      : clio::run::Task(), path_(CTP_MALLOC), exists_(0), is_dir_(0), size_(0),
        ino_(0), ctime_(0), mtime_(0), atime_(0), is_symlink_(0),
        uid_(0xFFFFFFFFu), gid_(0xFFFFFFFFu), mode_(0xFFFFFFFFu), nlink_(1) {}
  explicit GetattrTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query, const std::string &path)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetattr),
        path_(CTP_MALLOC, path), exists_(0), is_dir_(0), size_(0), ino_(0),
        ctime_(0), mtime_(0), atime_(0), is_symlink_(0),
        uid_(0xFFFFFFFFu), gid_(0xFFFFFFFFu), mode_(0xFFFFFFFFu), nlink_(1) {}
  void Copy(const ctp::ipc::FullPtr<GetattrTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; exists_ = o->exists_; is_dir_ = o->is_dir_;
    size_ = o->size_; ino_ = o->ino_; ctime_ = o->ctime_;
    mtime_ = o->mtime_; atime_ = o->atime_; is_symlink_ = o->is_symlink_;
    uid_ = o->uid_; gid_ = o->gid_; mode_ = o->mode_; nlink_ = o->nlink_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(exists_, is_dir_, size_, ino_, ctime_, mtime_, atime_, is_symlink_,
       uid_, gid_, mode_, nlink_);
  }
};

/** Truncate: set logical size; free trailing page-blobs. */
struct TruncateTask : public clio::run::Task {
  IN clio::run::priv::string path_;
  IN clio::run::u64 new_size_;
  /** Nonzero = ftruncate through an OPEN handle: resolve by this TAG and
   *  never materialize the path. A path-keyed truncate deliberately creates
   *  missing files (POSIX truncate(2)), but an fd-truncate of a file
   *  unlinked-while-open RESURRECTED the deleted name through that same
   *  materialization (generic/070's undeletable ghosts). */
  IN clio::run::u64 tag_packed_;
  /** The caller-known FULL extent of prior writes (the FUSE adapter's
   *  hiwater). The chimod-tracked size lags sieve-direct writes (they bypass
   *  the Write handler), so trimming by it left REAL page-blobs beyond a
   *  shrink alive — holes then read the old bytes back (fsx). The trim
   *  covers up to max(tracked, old_extent_). */
  IN clio::run::u64 old_extent_;
  TruncateTask()
      : clio::run::Task(), path_(CTP_MALLOC), new_size_(0), tag_packed_(0),
        old_extent_(0) {}
  explicit TruncateTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query,
                        const std::string &path, clio::run::u64 new_size,
                        clio::run::u64 tag_packed = 0,
                        clio::run::u64 old_extent = 0)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kTruncate),
        path_(CTP_MALLOC, path), new_size_(new_size),
        tag_packed_(tag_packed), old_extent_(old_extent) {}
  void Copy(const ctp::ipc::FullPtr<TruncateTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; new_size_ = o->new_size_; tag_packed_ = o->tag_packed_;
    old_extent_ = o->old_extent_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, new_size_, tag_packed_, old_extent_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** UtimensTask flag: a read of the file. The owner advances atime by the
 *  relatime rule and leaves mtime and ctime alone; other bits are ignored. */
GLOBAL_CROSS_CONST clio::run::u32 kUtimensAccess = 16u;
/** With kUtimensAccess: strictatime -- every read moves atime. */
GLOBAL_CROSS_CONST clio::run::u32 kUtimensAccessStrict = 32u;

/** Utimens: set a file's atime/mtime (ns). flags bit0=set atime, bit1=set
 *  mtime; a cleared bit means UTIME_OMIT (leave that stamp). ctime always
 *  bumps (except for kUtimensAccess). UTIME_NOW is resolved to a concrete
 *  ns value by the adapter. */
struct UtimensTask : public clio::run::Task {
  IN clio::run::priv::string path_;
  IN clio::run::u64 atime_ns_;
  IN clio::run::u64 mtime_ns_;
  IN clio::run::u32 flags_;  // bit0: set atime, bit1: set mtime
  UtimensTask()
      : clio::run::Task(), path_(CTP_MALLOC), atime_ns_(0), mtime_ns_(0),
        flags_(0) {}
  explicit UtimensTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query, const std::string &path,
                       clio::run::u64 atime_ns, clio::run::u64 mtime_ns,
                       clio::run::u32 flags)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kUtimens),
        path_(CTP_MALLOC, path), atime_ns_(atime_ns), mtime_ns_(mtime_ns),
        flags_(flags) {}
  void Copy(const ctp::ipc::FullPtr<UtimensTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; atime_ns_ = o->atime_ns_; mtime_ns_ = o->mtime_ns_;
    flags_ = o->flags_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, atime_ns_, mtime_ns_, flags_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * Chown: set the owner uid/gid of the file at `path_`. A field equal to
 * 0xFFFFFFFF (the POSIX uid_t/gid_t == (uid_t)-1 convention) means "leave this
 * field unchanged". The override is stored per-file and surfaced by getattr.
 */
struct ChownTask : public clio::run::Task {
  IN clio::run::priv::string path_;
  IN clio::run::u32 uid_;  // new uid, or 0xFFFFFFFF to leave unchanged
  IN clio::run::u32 gid_;  // new gid, or 0xFFFFFFFF to leave unchanged
  IN clio::run::u32 mode_;  // new mode bits, or 0xFFFFFFFF to leave unchanged
  ChownTask()
      : clio::run::Task(), path_(CTP_MALLOC), uid_(0xFFFFFFFFu),
        gid_(0xFFFFFFFFu), mode_(0xFFFFFFFFu) {}
  explicit ChownTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                     const clio::run::PoolQuery &pool_query, const std::string &path,
                     clio::run::u32 uid, clio::run::u32 gid,
                     clio::run::u32 mode = 0xFFFFFFFFu)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kChown),
        path_(CTP_MALLOC, path), uid_(uid), gid_(gid), mode_(mode) {}
  void Copy(const ctp::ipc::FullPtr<ChownTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; uid_ = o->uid_; gid_ = o->gid_; mode_ = o->mode_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, uid_, gid_, mode_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** A single-path task body shared by unlink/mkdir/rmdir. */
#define CLIO_FS_PATH_TASK(NAME, METHOD)                                        \
  struct NAME : public clio::run::Task {                                             \
    IN clio::run::priv::string path_;                                                \
    NAME() : clio::run::Task(), path_(CTP_MALLOC) {}                                 \
    explicit NAME(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,      \
                  const clio::run::PoolQuery &pool_query, const std::string &path)   \
        : clio::run::Task(task_id, pool_id, pool_query, METHOD),                     \
          path_(CTP_MALLOC, path) {}                                           \
    void Copy(const ctp::ipc::FullPtr<NAME>& o) {                              \
      clio::run::Task::Copy(o.template Cast<clio::run::Task>());               \
      path_ = o->path_;                                                        \
    }                                                                          \
    template <typename Ar> void SerializeIn(Ar &ar) {                          \
      Task::SerializeIn(ar); ar(path_);                                        \
    }                                                                          \
    template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); } \
  }

CLIO_FS_PATH_TASK(UnlinkTask, Method::kUnlink);
CLIO_FS_PATH_TASK(MkdirTask, Method::kMkdir);
CLIO_FS_PATH_TASK(RmdirTask, Method::kRmdir);
#undef CLIO_FS_PATH_TASK

/** Rename: move src -> dst (re-tags / re-keys blobs). */
struct RenameTask : public clio::run::Task {
  IN clio::run::priv::string src_;
  IN clio::run::priv::string dst_;
  RenameTask() : clio::run::Task(), src_(CTP_MALLOC), dst_(CTP_MALLOC) {}
  explicit RenameTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                      const clio::run::PoolQuery &pool_query, const std::string &src,
                      const std::string &dst)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kRename),
        src_(CTP_MALLOC, src), dst_(CTP_MALLOC, dst) {}
  void Copy(const ctp::ipc::FullPtr<RenameTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    src_ = o->src_; dst_ = o->dst_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(src_, dst_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * Link: create a hard link `link_` to the existing file `target_`. Both names
 * end up bound to the same CTE tag id (a tag-level alias), so they share all
 * data. Returns errno-style codes in return_code_ (0 / ENOENT / EIO).
 */
struct LinkTask : public clio::run::Task {
  IN clio::run::priv::string target_;  // existing file path
  IN clio::run::priv::string link_;    // new link path
  LinkTask() : clio::run::Task(), target_(CTP_MALLOC), link_(CTP_MALLOC) {}
  explicit LinkTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                    const clio::run::PoolQuery &pool_query, const std::string &target,
                    const std::string &link)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kLink),
        target_(CTP_MALLOC, target), link_(CTP_MALLOC, link) {}
  void Copy(const ctp::ipc::FullPtr<LinkTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    target_ = o->target_; link_ = o->link_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(target_, link_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * Symlink: create a symbolic link at `path_` whose target string is `target_`.
 * The symlink is a CTE tag at `path_` carrying a reserved marker blob that
 * stores the target bytes. Returns errno-style codes (0 / EEXIST / EIO).
 */
struct SymlinkTask : public clio::run::Task {
  IN clio::run::priv::string target_;  // link target string (contents)
  IN clio::run::priv::string path_;    // path of the new symlink
  SymlinkTask() : clio::run::Task(), target_(CTP_MALLOC), path_(CTP_MALLOC) {}
  explicit SymlinkTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query,
                       const std::string &target, const std::string &path)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kSymlink),
        target_(CTP_MALLOC, target), path_(CTP_MALLOC, path) {}
  void Copy(const ctp::ipc::FullPtr<SymlinkTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    target_ = o->target_; path_ = o->path_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(target_, path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * Readlink: read the target string of the symlink at `path_`. Returns the
 * bytes in `target_`; errno-style codes (0 / ENOENT / EINVAL) in return_code_.
 */
struct ReadlinkTask : public clio::run::Task {
  IN clio::run::priv::string path_;    // path of the symlink
  OUT clio::run::priv::string target_;  // resolved target string
  ReadlinkTask() : clio::run::Task(), path_(CTP_MALLOC), target_(CTP_MALLOC) {}
  explicit ReadlinkTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query, const std::string &path)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kReadlink),
        path_(CTP_MALLOC, path), target_(CTP_MALLOC) {}
  void Copy(const ctp::ipc::FullPtr<ReadlinkTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; target_ = o->target_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(target_);
  }
};

/**
 * Setxattr: set extended attribute `name_` to `value_` on the file at `path_`.
 * `value_` may contain NUL bytes (raw byte string). `flags_` carries
 * XATTR_CREATE (1) / XATTR_REPLACE (2) semantics. Returns errno-style codes
 * (0 / EEXIST / ENODATA / ENOENT / EIO) in return_code_.
 */
struct SetxattrTask : public clio::run::Task {
  IN clio::run::priv::string path_;   // file path
  IN clio::run::priv::string name_;   // attribute name
  IN clio::run::priv::string value_;  // attribute value bytes (may hold NULs)
  IN clio::run::u32 flags_;           // XATTR_CREATE=1 / XATTR_REPLACE=2
  SetxattrTask()
      : clio::run::Task(), path_(CTP_MALLOC), name_(CTP_MALLOC),
        value_(CTP_MALLOC), flags_(0) {}
  explicit SetxattrTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query,
                        const std::string &path, const std::string &name,
                        const std::string &value, clio::run::u32 flags)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kSetxattr),
        path_(CTP_MALLOC, path), name_(CTP_MALLOC, name),
        value_(CTP_MALLOC, value), flags_(flags) {}
  void Copy(const ctp::ipc::FullPtr<SetxattrTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; name_ = o->name_; value_ = o->value_; flags_ = o->flags_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, name_, value_, flags_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * Getxattr: read extended attribute `name_` of the file at `path_`. On success
 * `found_`=1 and `value_` holds the raw bytes; `found_`=0 means the attribute
 * is absent (adapter maps to -ENODATA). return_code_ is errno-style (0 /
 * ENOENT).
 */
struct GetxattrTask : public clio::run::Task {
  IN clio::run::priv::string path_;   // file path
  IN clio::run::priv::string name_;   // attribute name
  OUT clio::run::priv::string value_;  // attribute value bytes (may hold NULs)
  OUT clio::run::u32 found_;           // 1 if the attribute exists
  GetxattrTask()
      : clio::run::Task(), path_(CTP_MALLOC), name_(CTP_MALLOC),
        value_(CTP_MALLOC), found_(0) {}
  explicit GetxattrTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query,
                        const std::string &path, const std::string &name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetxattr),
        path_(CTP_MALLOC, path), name_(CTP_MALLOC, name), value_(CTP_MALLOC),
        found_(0) {}
  void Copy(const ctp::ipc::FullPtr<GetxattrTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; name_ = o->name_; value_ = o->value_; found_ = o->found_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, name_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(value_, found_);
  }
};

/**
 * Listxattr: list all extended attribute names of the file at `path_`. `names_`
 * is the NUL-separated concatenation of names, each NUL-terminated (the POSIX
 * listxattr wire format). return_code_ is errno-style (0 / ENOENT).
 */
struct ListxattrTask : public clio::run::Task {
  IN clio::run::priv::string path_;   // file path
  OUT clio::run::priv::string names_;  // NUL-terminated name list
  ListxattrTask() : clio::run::Task(), path_(CTP_MALLOC), names_(CTP_MALLOC) {}
  explicit ListxattrTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                         const clio::run::PoolQuery &pool_query,
                         const std::string &path)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kListxattr),
        path_(CTP_MALLOC, path), names_(CTP_MALLOC) {}
  void Copy(const ctp::ipc::FullPtr<ListxattrTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; names_ = o->names_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(names_);
  }
};

/**
 * Removexattr: remove extended attribute `name_` from the file at `path_`.
 * Returns errno-style codes (0 / ENODATA / ENOENT / EIO) in return_code_.
 */
struct RemovexattrTask : public clio::run::Task {
  IN clio::run::priv::string path_;   // file path
  IN clio::run::priv::string name_;   // attribute name
  RemovexattrTask()
      : clio::run::Task(), path_(CTP_MALLOC), name_(CTP_MALLOC) {}
  explicit RemovexattrTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                           const clio::run::PoolQuery &pool_query,
                           const std::string &path, const std::string &name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kRemovexattr),
        path_(CTP_MALLOC, path), name_(CTP_MALLOC, name) {}
  void Copy(const ctp::ipc::FullPtr<RemovexattrTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; name_ = o->name_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, name_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** Readdir: list direct children of a directory. */
struct ReaddirTask : public clio::run::Task {
  IN clio::run::priv::string path_;
  OUT clio::run::priv::vector<clio::run::priv::string> entries_;
  OUT clio::run::priv::vector<clio::run::u64> inos_;  // packed TagId per entry (parallel)
  ReaddirTask()
      : clio::run::Task(), path_(CTP_MALLOC), entries_(CTP_MALLOC),
        inos_(CTP_MALLOC) {}
  explicit ReaddirTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query, const std::string &path)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kReaddir),
        path_(CTP_MALLOC, path), entries_(CTP_MALLOC), inos_(CTP_MALLOC) {}
  void Copy(const ctp::ipc::FullPtr<ReaddirTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; entries_ = o->entries_; inos_ = o->inos_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(entries_, inos_);
  }
};

/** StatSize: lightweight logical-size lookup. */
struct StatSizeTask : public clio::run::Task {
  IN clio::run::priv::string path_;
  OUT clio::run::u32 exists_;
  OUT clio::run::u64 size_;
  StatSizeTask() : clio::run::Task(), path_(CTP_MALLOC), exists_(0), size_(0) {}
  explicit StatSizeTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query, const std::string &path)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kStatSize),
        path_(CTP_MALLOC, path), exists_(0), size_(0) {}
  void Copy(const ctp::ipc::FullPtr<StatSizeTask>& o) {
    // Base fields first (pool id, method, query, flags): a forwarded copy
    // without them reached SendIn with a null pool and crashed the node.
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    path_ = o->path_; exists_ = o->exists_; size_ = o->size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(exists_, size_);
  }
};

/**
 * ShardOp (internal): one operation on namespace state owned by the target
 * container -- a directory's entries/attributes (routed by hash of the
 * directory path) or an inode (routed to its home). A public filesystem task
 * runs where its first piece of state lives and sends a ShardOp for any
 * other piece, so no container ever holds state it does not own. The
 * request and response are FsEnc-encoded (see filesystem_runtime.cc); the
 * task itself is just two byte strings.
 */
struct ShardOpTask : public clio::run::Task {
  IN clio::run::u32 op_;               ///< FsShardOp code
  IN clio::run::priv::string req_;     ///< encoded request
  OUT clio::run::priv::string resp_;   ///< encoded response
  ShardOpTask() : clio::run::Task(), op_(0), req_(CTP_MALLOC), resp_(CTP_MALLOC) {}
  explicit ShardOpTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query,
                       clio::run::u32 op, const std::string &req)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kShardOp),
        op_(op), req_(CTP_MALLOC, req), resp_(CTP_MALLOC) {}
  void Copy(const ctp::ipc::FullPtr<ShardOpTask> &o) {
    clio::run::Task::Copy(o.template Cast<clio::run::Task>());
    op_ = o->op_; req_ = o->req_; resp_ = o->resp_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(op_, req_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(resp_);
  }
};







}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_TASKS_H_
