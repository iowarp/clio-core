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

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/filesystem/autogen/filesystem_methods.h>

namespace clio::cte::filesystem {

/** Monitor reuses the admin task type (same as core/compressor chimods). */
using MonitorTask = clio::run::admin::MonitorTask;

/** 1 MiB page size, matching the libfuse adapter's blob paging. */
GLOBAL_CROSS_CONST chi::u64 kFsPageSize = 1024 * 1024;

/**
 * Well-known default pool id/name for the filesystem chimod, used by the
 * interceptor adapters (libfuse, POSIX, ...) to create-or-bind a single
 * shared filesystem pool over the default CTE core pool — exactly the way
 * clio::cte::core uses kCtePoolId/kCtePoolName.
 */
static constexpr chi::PoolId kCfsPoolId(560, 0);
static constexpr const char *kCfsPoolName = "clio_cte_filesystem";

/**
 * Container creation params. next_pool_id_ is the CTE core pool this
 * filesystem sits over (where its tags/blobs actually live).
 */
struct FilesystemConfig {
  static constexpr const char* chimod_lib_name = "clio_cte_filesystem";

  chi::PoolId next_pool_id_;  ///< CTE core pool id (e.g. 512.0)

  FilesystemConfig() : next_pool_id_(chi::PoolId::GetNull()) {}
  FilesystemConfig(const chi::PoolId &pool_id, const FilesystemConfig &other)
      : next_pool_id_(other.next_pool_id_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_);
  }

  void LoadConfig(const chi::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) {
      return;
    }
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      if (node["next_pool_id"]) {
        next_pool_id_ = chi::PoolId::FromString(
            node["next_pool_id"].as<std::string>());
      }
    } catch (...) {
      // best-effort
    }
  }
};

using CreateTask = clio::run::admin::GetOrCreatePoolTask<FilesystemConfig>;

/** DestroyTask — teardown (mirrors compressor). */
struct DestroyTask : public chi::Task {
  DestroyTask() : chi::Task() {}
  explicit DestroyTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query)
      : chi::Task(task_id, pool_id, pool_query, Method::kDestroy) {}
  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) { (void)other; }
  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** Open: resolve/create a file's tag; returns a handle + current size. */
struct OpenTask : public chi::Task {
  IN chi::priv::string path_;
  IN chi::u32 flags_;
  IN chi::u32 mode_;
  OUT chi::u64 handle_;
  OUT chi::u64 size_;
  OUT chi::u32 created_;  // 1 if the file was newly created

  OpenTask()
      : chi::Task(), path_(CTP_MALLOC), flags_(0), mode_(0644),
        handle_(0), size_(0), created_(0) {}
  explicit OpenTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query, const std::string &path,
                    chi::u32 flags, chi::u32 mode)
      : chi::Task(task_id, pool_id, pool_query, Method::kOpen),
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
struct CloseTask : public chi::Task {
  IN chi::u64 handle_;
  CloseTask() : chi::Task(), handle_(0) {}
  explicit CloseTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                     const chi::PoolQuery &pool_query, chi::u64 handle)
      : chi::Task(task_id, pool_id, pool_query, Method::kClose),
        handle_(handle) {}
  void Copy(const ctp::ipc::FullPtr<CloseTask>& o) { handle_ = o->handle_; }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(handle_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/** Read: page-loop GetBlob over [offset, offset+size). */
struct ReadTask : public chi::Task {
  IN chi::u64 handle_;
  IN chi::u64 offset_;
  IN chi::u64 size_;
  INOUT ctp::ipc::ShmPtr<> data_;
  OUT chi::u64 bytes_read_;
  ReadTask()
      : chi::Task(), handle_(0), offset_(0), size_(0),
        data_(ctp::ipc::ShmPtr<>::GetNull()), bytes_read_(0) {}
  explicit ReadTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query, chi::u64 handle,
                    chi::u64 offset, chi::u64 size, ctp::ipc::ShmPtr<> data)
      : chi::Task(task_id, pool_id, pool_query, Method::kRead),
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
struct WriteTask : public chi::Task {
  IN chi::u64 handle_;
  IN chi::u64 offset_;
  IN chi::u64 size_;
  IN ctp::ipc::ShmPtr<> data_;
  OUT chi::u64 bytes_written_;
  OUT chi::u64 new_size_;  // logical size after the write
  WriteTask()
      : chi::Task(), handle_(0), offset_(0), size_(0),
        data_(ctp::ipc::ShmPtr<>::GetNull()), bytes_written_(0), new_size_(0) {}
  explicit WriteTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                     const chi::PoolQuery &pool_query, chi::u64 handle,
                     chi::u64 offset, chi::u64 size, ctp::ipc::ShmPtr<> data)
      : chi::Task(task_id, pool_id, pool_query, Method::kWrite),
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
struct AppendTask : public chi::Task {
  IN chi::u64 handle_;
  IN chi::u64 size_;
  IN ctp::ipc::ShmPtr<> data_;
  OUT chi::u64 offset_;          // where the data landed (old logical size)
  OUT chi::u64 bytes_written_;
  OUT chi::u64 new_size_;
  AppendTask()
      : chi::Task(), handle_(0), size_(0),
        data_(ctp::ipc::ShmPtr<>::GetNull()), offset_(0), bytes_written_(0),
        new_size_(0) {}
  explicit AppendTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                      const chi::PoolQuery &pool_query, chi::u64 handle,
                      chi::u64 size, ctp::ipc::ShmPtr<> data)
      : chi::Task(task_id, pool_id, pool_query, Method::kAppend),
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
struct GetattrTask : public chi::Task {
  IN chi::priv::string path_;
  OUT chi::u32 exists_;
  OUT chi::u32 is_dir_;
  OUT chi::u64 size_;
  GetattrTask()
      : chi::Task(), path_(CTP_MALLOC), exists_(0), is_dir_(0), size_(0) {}
  explicit GetattrTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query, const std::string &path)
      : chi::Task(task_id, pool_id, pool_query, Method::kGetattr),
        path_(CTP_MALLOC, path), exists_(0), is_dir_(0), size_(0) {}
  void Copy(const ctp::ipc::FullPtr<GetattrTask>& o) {
    path_ = o->path_; exists_ = o->exists_; is_dir_ = o->is_dir_;
    size_ = o->size_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(exists_, is_dir_, size_);
  }
};

/** Truncate: set logical size; free trailing page-blobs. */
struct TruncateTask : public chi::Task {
  IN chi::priv::string path_;
  IN chi::u64 new_size_;
  TruncateTask() : chi::Task(), path_(CTP_MALLOC), new_size_(0) {}
  explicit TruncateTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                        const chi::PoolQuery &pool_query,
                        const std::string &path, chi::u64 new_size)
      : chi::Task(task_id, pool_id, pool_query, Method::kTruncate),
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
  struct NAME : public chi::Task {                                             \
    IN chi::priv::string path_;                                                \
    NAME() : chi::Task(), path_(CTP_MALLOC) {}                                 \
    explicit NAME(const chi::TaskId &task_id, const chi::PoolId &pool_id,      \
                  const chi::PoolQuery &pool_query, const std::string &path)   \
        : chi::Task(task_id, pool_id, pool_query, METHOD),                     \
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
struct RenameTask : public chi::Task {
  IN chi::priv::string src_;
  IN chi::priv::string dst_;
  RenameTask() : chi::Task(), src_(CTP_MALLOC), dst_(CTP_MALLOC) {}
  explicit RenameTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                      const chi::PoolQuery &pool_query, const std::string &src,
                      const std::string &dst)
      : chi::Task(task_id, pool_id, pool_query, Method::kRename),
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
struct LinkTask : public chi::Task {
  IN chi::priv::string target_;  // existing file path
  IN chi::priv::string link_;    // new link path
  LinkTask() : chi::Task(), target_(CTP_MALLOC), link_(CTP_MALLOC) {}
  explicit LinkTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query, const std::string &target,
                    const std::string &link)
      : chi::Task(task_id, pool_id, pool_query, Method::kLink),
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
struct ReaddirTask : public chi::Task {
  IN chi::priv::string path_;
  OUT chi::priv::vector<chi::priv::string> entries_;
  ReaddirTask()
      : chi::Task(), path_(CTP_MALLOC), entries_(CTP_MALLOC) {}
  explicit ReaddirTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query, const std::string &path)
      : chi::Task(task_id, pool_id, pool_query, Method::kReaddir),
        path_(CTP_MALLOC, path), entries_(CTP_MALLOC) {}
  void Copy(const ctp::ipc::FullPtr<ReaddirTask>& o) {
    path_ = o->path_; entries_ = o->entries_;
  }
  template <typename Ar> void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(path_);
  }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(entries_);
  }
};

/** StatSize: lightweight logical-size lookup. */
struct StatSizeTask : public chi::Task {
  IN chi::priv::string path_;
  OUT chi::u32 exists_;
  OUT chi::u64 size_;
  StatSizeTask() : chi::Task(), path_(CTP_MALLOC), exists_(0), size_(0) {}
  explicit StatSizeTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                        const chi::PoolQuery &pool_query, const std::string &path)
      : chi::Task(task_id, pool_id, pool_query, Method::kStatSize),
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

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_TASKS_H_
