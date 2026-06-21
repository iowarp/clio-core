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

  FilesystemConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}
  FilesystemConfig(const clio::run::PoolId &pool_id, const FilesystemConfig &other)
      : next_pool_id_(other.next_pool_id_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_);
  }

  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) {
      return;
    }
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      if (node["next_pool_id"]) {
        next_pool_id_ = clio::run::PoolId::FromString(
            node["next_pool_id"].as<std::string>());
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
  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) { (void)other; }
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

  OpenTask()
      : clio::run::Task(), path_(CTP_MALLOC), flags_(0), mode_(0644),
        handle_(0), size_(0), created_(0) {}
  explicit OpenTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                    const clio::run::PoolQuery &pool_query, const std::string &path,
                    clio::run::u32 flags, clio::run::u32 mode)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kOpen),
        path_(CTP_MALLOC, path), flags_(flags), mode_(mode), handle_(0),
        size_(0), created_(0) {}
  void Copy(const ctp::ipc::FullPtr<OpenTask>& o) {
    path_ = o->path_; flags_ = o->flags_; mode_ = o->mode_;
    handle_ = o->handle_; size_ = o->size_; created_ = o->created_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, flags_, mode_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(handle_, size_, created_);
  }
};

/** Close: release a handle (drains pending writes server-side). */
struct CloseTask : public clio::run::Task {
  IN clio::run::u64 handle_;
  CloseTask() : clio::run::Task(), handle_(0) {}
  explicit CloseTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                     const clio::run::PoolQuery &pool_query, clio::run::u64 handle)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kClose),
        handle_(handle) {}
  void Copy(const ctp::ipc::FullPtr<CloseTask>& o) { handle_ = o->handle_; }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(handle_);
  }
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

/** Append: write at the current logical size, then advance it. */
struct AppendTask : public clio::run::Task {
  IN clio::run::u64 handle_;
  IN clio::run::u64 size_;
  IN ctp::ipc::ShmPtr<> data_;
  OUT clio::run::u64 offset_;          // where the data landed (old logical size)
  OUT clio::run::u64 bytes_written_;
  OUT clio::run::u64 new_size_;
  AppendTask()
      : clio::run::Task(), handle_(0), size_(0),
        data_(ctp::ipc::ShmPtr<>::GetNull()), offset_(0), bytes_written_(0),
        new_size_(0) {}
  explicit AppendTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                      const clio::run::PoolQuery &pool_query, clio::run::u64 handle,
                      clio::run::u64 size, ctp::ipc::ShmPtr<> data)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kAppend),
        handle_(handle), size_(size), data_(data), offset_(0),
        bytes_written_(0), new_size_(0) {}
  void Copy(const ctp::ipc::FullPtr<AppendTask>& o) {
    handle_ = o->handle_; size_ = o->size_; data_ = o->data_;
    offset_ = o->offset_; bytes_written_ = o->bytes_written_;
    new_size_ = o->new_size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(handle_, size_, data_);
    ar.bulk(data_, size_, BULK_XFER);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(offset_, bytes_written_, new_size_);
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
  GetattrTask()
      : clio::run::Task(), path_(CTP_MALLOC), exists_(0), is_dir_(0), size_(0),
        ino_(0), ctime_(0) {}
  explicit GetattrTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query, const std::string &path)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetattr),
        path_(CTP_MALLOC, path), exists_(0), is_dir_(0), size_(0), ino_(0),
        ctime_(0) {}
  void Copy(const ctp::ipc::FullPtr<GetattrTask>& o) {
    path_ = o->path_; exists_ = o->exists_; is_dir_ = o->is_dir_;
    size_ = o->size_; ino_ = o->ino_; ctime_ = o->ctime_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(exists_, is_dir_, size_, ino_, ctime_);
  }
};

/** Truncate: set logical size; free trailing page-blobs. */
struct TruncateTask : public clio::run::Task {
  IN clio::run::priv::string path_;
  IN clio::run::u64 new_size_;
  TruncateTask() : clio::run::Task(), path_(CTP_MALLOC), new_size_(0) {}
  explicit TruncateTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query,
                        const std::string &path, clio::run::u64 new_size)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kTruncate),
        path_(CTP_MALLOC, path), new_size_(new_size) {}
  void Copy(const ctp::ipc::FullPtr<TruncateTask>& o) {
    path_ = o->path_; new_size_ = o->new_size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_, new_size_);
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
    void Copy(const ctp::ipc::FullPtr<NAME>& o) { path_ = o->path_; }          \
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
    target_ = o->target_; link_ = o->link_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(target_, link_);
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
    path_ = o->path_; exists_ = o->exists_; size_ = o->size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(exists_, size_);
  }
};

// ===========================================================================
// Deferred-append pipeline
//
// Appends are NOT applied to the tail synchronously. Instead Append stamps a
// (UTC, logical) order, writes the bytes as a standalone "data blob" under the
// file's tag, and queues a pending entry. A periodic AppendSequence drains the
// per-node queue and, per tag, submits an AppendCollect routed ManyToOne to the
// tag's sequencer. There the batched members' entries are combined (AggregateIn)
// into one global, timestamp-sorted batch (the "plan" phase): it reads the file
// tail (GetTagSize minus this batch's still-staged data), lays each data blob
// out into 1 MiB file pages, and dispatches AppendExecution slices (<=16 MiB
// each) that GetBlob->PutBlob->DelBlob the data into the file pages.
// ===========================================================================

/** One pending append: a staged data blob waiting to be merged into a file. */
struct AppendEntry {
  std::string data_blob_id_;     ///< name of the staged blob (under file tag)
  clio::run::u64 data_blob_size_ = 0;  ///< staged blob length in bytes
  clio::run::u64 utc_ns_ = 0;          ///< wallclock at placement (primary sort key)
  clio::run::u64 logical_ = 0;         ///< per-node logical counter (tiebreak)
  template <class Ar> void serialize(Ar &ar) {
    ar(data_blob_id_, data_blob_size_, utc_ns_, logical_);
  }
};

/** One merge step: copy [size_] bytes of a data blob into a file page blob. */
struct AppendPlanStep {
  clio::run::u64 file_page_ = 0;       ///< destination file page index (blob name)
  std::string data_blob_id_;     ///< source staged blob
  clio::run::u64 off_in_page_ = 0;     ///< destination offset within the file page
  clio::run::u64 off_in_data_ = 0;     ///< source offset within the data blob
  clio::run::u64 size_ = 0;            ///< bytes copied by this step (<= data size)
  clio::run::u64 data_blob_size_ = 0;  ///< full data blob size (the final step for a
                                 ///< blob has size_==data_blob_size_ so its
                                 ///< DelBlob can't race a partial copy)
  template <class Ar> void serialize(Ar &ar) {
    ar(file_page_, data_blob_id_, off_in_page_, off_in_data_, size_,
       data_blob_size_);
  }
};

/** AppendSequence: periodic local trigger to drain the pending-append queue. */
struct AppendSequenceTask : public clio::run::Task {
  AppendSequenceTask() : clio::run::Task() {}
  explicit AppendSequenceTask(const clio::run::TaskId &task_id,
                              const clio::run::PoolId &pool_id,
                              const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kAppendSequence) {}
  void Copy(const ctp::ipc::FullPtr<AppendSequenceTask> &o) {
    Task::Copy(o.template Cast<Task>());
  }
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * AppendCollect: ManyToOne collective per tag. Each node submits its pending
 * entries for the tag; AggregateIn concatenates them at the sequencer; the
 * single aggregate run sorts + plans + dispatches the merge.
 */
struct AppendCollectTask : public clio::run::Task {
  IN clio::cte::core::TagId tag_id_;
  IN std::vector<AppendEntry> entries_;
  OUT clio::run::u64 new_size_;  ///< file logical size after the batch is applied
  AppendCollectTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()), new_size_(0) {}
  explicit AppendCollectTask(const clio::run::TaskId &task_id,
                             const clio::run::PoolId &pool_id,
                             const clio::run::PoolQuery &pool_query,
                             const clio::cte::core::TagId &tag_id,
                             const std::vector<AppendEntry> &entries)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kAppendCollect),
        tag_id_(tag_id), entries_(entries), new_size_(0) {}
  void Copy(const ctp::ipc::FullPtr<AppendCollectTask> &o) {
    Task::Copy(o.template Cast<Task>());
    tag_id_ = o->tag_id_; entries_ = o->entries_; new_size_ = o->new_size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(tag_id_, entries_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(new_size_);
  }
  /** ManyToOne: fold a batched member's pending entries into this aggregate. */
  void AggregateIn(const ctp::ipc::FullPtr<clio::run::Task> &member_base) {
    auto m = member_base.template Cast<AppendCollectTask>();
    entries_.insert(entries_.end(), m->entries_.begin(), m->entries_.end());
  }
};

/**
 * AppendPlan: a REGULAR (suspendable) task that does the heavy planning work
 * for one tag's batch. Submitted by the synchronous AppendCollect aggregate
 * (the ManyToOne synthetic aggregate task can't itself suspend). Sorts the
 * batch, reads the file tail, builds the page-merge plan, and dispatches
 * AppendExecution slices.
 */
struct AppendPlanTask : public clio::run::Task {
  IN clio::cte::core::TagId tag_id_;
  IN std::vector<AppendEntry> entries_;
  AppendPlanTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()) {}
  explicit AppendPlanTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const clio::cte::core::TagId &tag_id,
                          const std::vector<AppendEntry> &entries)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kAppendPlan),
        tag_id_(tag_id), entries_(entries) {}
  void Copy(const ctp::ipc::FullPtr<AppendPlanTask> &o) {
    Task::Copy(o.template Cast<Task>());
    tag_id_ = o->tag_id_; entries_ = o->entries_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(tag_id_, entries_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** AppendExecution: apply a slice of the merge plan (GetBlob->PutBlob->DelBlob). */
struct AppendExecutionTask : public clio::run::Task {
  IN clio::cte::core::TagId tag_id_;          // destination file tag
  IN clio::cte::core::TagId staging_tag_id_;  // source staged data blobs
  IN std::vector<AppendPlanStep> steps_;
  AppendExecutionTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()),
        staging_tag_id_(clio::cte::core::TagId::GetNull()) {}
  explicit AppendExecutionTask(const clio::run::TaskId &task_id,
                               const clio::run::PoolId &pool_id,
                               const clio::run::PoolQuery &pool_query,
                               const clio::cte::core::TagId &tag_id,
                               const clio::cte::core::TagId &staging_tag_id,
                               const std::vector<AppendPlanStep> &steps)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kAppendExecution),
        tag_id_(tag_id), staging_tag_id_(staging_tag_id), steps_(steps) {}
  void Copy(const ctp::ipc::FullPtr<AppendExecutionTask> &o) {
    Task::Copy(o.template Cast<Task>());
    tag_id_ = o->tag_id_; staging_tag_id_ = o->staging_tag_id_;
    steps_ = o->steps_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(tag_id_, staging_tag_id_, steps_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_TASKS_H_
