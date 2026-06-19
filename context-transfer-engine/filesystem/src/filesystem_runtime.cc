/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Filesystem chimod method handlers. Each op is implemented like the libfuse
 * adapter (paths -> CTE tags, offsets -> 1 MiB page-blobs "0","1",...) driving
 * a CTE core client (`cte_`, bound to next_pool_id_ at Create), plus per-file
 * logical-size metadata so getattr is exact and truncate/append work.
 */
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clio_cte/filesystem/filesystem_runtime.h>

namespace clio::cte::filesystem {

namespace {
/** UTC wallclock nanoseconds (system_clock) — the primary append order key. */
inline chi::u64 NowUtcNs() {
  return static_cast<chi::u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

/**
 * Reserved name for a staged append data blob, under the file's own tag. The
 * "_append/" prefix can't collide with the numeric page-blob names ("0","1",
 * ...); node id keeps it unique across nodes, the logical counter within a node.
 */
inline std::string MakeDataBlobId(chi::u32 node_id, chi::u64 logical) {
  return "_append/" + std::to_string(node_id) + "." + std::to_string(logical);
}
/**
 * Reserved child tag that marks an explicit (possibly empty) directory.
 *
 * With hierarchical tag names a directory no longer needs a trailing-slash
 * sentinel tag (which now collides with the file of the same path). Instead a
 * directory is simply "a tag that has at least one child": intermediate dirs
 * already have children (the files/dirs under them), and an *empty* directory
 * created by mkdir gets this one hidden marker child so it is still detectable.
 * The marker is filtered out of readdir and hidden from getattr.
 */
constexpr const char *kDirMarker = ".__clio_dir__";

/** Page name for a byte offset (1 MiB pages, stringified index — libfuse). */
inline std::string PageName(chi::u64 off) {
  return std::to_string(off / kFsPageSize);
}

/** Strip a single trailing '/' from a path (keeping a bare root "/"). */
inline std::string StripTrailingSlash(std::string p) {
  if (p.size() > 1 && p.back() == '/') p.pop_back();
  return p;
}
/** Escape regex metacharacters for an exact TagQuery match (from libfuse). */
inline std::string EscapeExact(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      out += '\\';
    }
    out += c;
  }
  return out;
}
}  // namespace

chi::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task,
                                chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  FilesystemConfig cfg = task->GetParams();
  next_pool_id_ = cfg.next_pool_id_;
  if (!next_pool_id_.IsNull()) {
    cte_ = clio::cte::core::Client(next_pool_id_);
  }
  // Bind a client to our own pool for self-submitted pipeline tasks. Use the
  // assigned pool id from the CreateTask (pool_id_ isn't reliable yet here),
  // matching how the CTE core initializes its self-client.
  self_.Init(task->new_pool_id_);
  HLOG(kInfo, "filesystem: Create over CTE core pool {}",
       next_pool_id_.ToString());
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
  std::string path = task->path_.str();

  // Did the tag already exist (so we can report created_)?
  bool existed = false;
  {
    auto q = cte_.AsyncTagQuery(EscapeExact(path), 1, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(q);
    existed = (q->GetReturnCode() == 0 && !q->results_.empty());
  }

  // Honor O_CREAT: a plain open of a missing file must fail. Report it by
  // returning handle_=0 (the adapters map that to ENOENT) rather than
  // silently creating the tag.
  if (!existed && !(task->flags_ & O_CREAT)) {
    task->handle_ = 0;
    task->size_ = 0;
    task->created_ = 0;
    task->return_code_ = 0;
    (void)ctx;
    CLIO_CO_RETURN;
  }

  // Resolve / create the tag for this path.
  auto t = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                    chi::PoolQuery::Local());
  CLIO_CO_AWAIT(t);
  if (t->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }
  clio::cte::core::TagId tag_id = t->tag_id_;

  // Current physical size (best-effort baseline for the logical size).
  chi::u64 size = 0;
  {
    auto s = cte_.AsyncGetTagSize(tag_id, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(s);
    if (s->GetReturnCode() == 0) {
      size = s->tag_size_;
    }
  }

  // Register the handle + per-file logical size (source of truth henceforth).
  chi::u64 handle = next_handle_.fetch_add(1);
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    std::shared_ptr<FileInfo> fi;
    if (it != by_path_.end()) {
      fi = it->second;
      size = fi->size_.load();  // keep the live logical size if already open
    } else {
      fi = std::make_shared<FileInfo>();
      fi->tag_id_ = tag_id;
      fi->path_ = path;
      fi->size_.store(size);
      by_path_[path] = fi;
    }
    handles_[handle] = fi;
  }

  task->handle_ = handle;
  task->size_ = size;
  task->created_ = existed ? 0u : 1u;
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Close(ctp::ipc::FullPtr<CloseTask> task,
                               chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    handles_.erase(task->handle_);
  }
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// Look up the FileInfo for a handle (nullptr if unknown). Brief lock only.
#define CLIO_FS_LOOKUP(fi, handle)                       \
  std::shared_ptr<FileInfo> fi;                          \
  do {                                                   \
    std::lock_guard<std::mutex> g(meta_mu_);             \
    auto it = handles_.find(handle);                     \
    if (it != handles_.end()) fi = it->second;           \
  } while (0)

chi::TaskResume Runtime::Read(ctp::ipc::FullPtr<ReadTask> task,
                              chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_LOOKUP(fi, task->handle_);
  if (!fi) {
    task->return_code_ = EBADF;
    CLIO_CO_RETURN;
  }
  clio::cte::core::TagId tag_id = fi->tag_id_;
  chi::u64 file_size = fi->size_.load();

  auto *ipc = CLIO_IPC;
  // Read directly into the task's payload — GetBlob copies into the ShmPtr we
  // hand it, so point it at the right sub-region of data_ each page instead of
  // staging through a freshly-allocated buffer.
  ctp::ipc::ShmPtr<char> data_base = task->data_.template Cast<char>();
  char *dst = ipc->ToFullPtr<char>(data_base).ptr_;

  chi::u64 offset = task->offset_;
  chi::u64 want = task->size_;
  if (offset >= file_size) {
    task->bytes_read_ = 0;
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  if (offset + want > file_size) {
    want = file_size - offset;  // clamp to logical EOF
  }

  // Pre-zero the whole region. Bytes within the logical file size that were
  // never physically written — holes, or a page-blob shorter than the request
  // (e.g. after ftruncate-grow) — must read back as zeros. GetBlob writes only
  // the bytes that exist and returns rc==0 even for a short/empty read, so it
  // can't be relied on to zero the gap; pre-zeroing makes hole reads correct.
  std::memset(dst, 0, want);

  chi::u64 done = 0;
  chi::u64 cur = offset;
  while (done < want) {
    chi::u64 page_off = cur % kFsPageSize;
    chi::u64 to_read = std::min(kFsPageSize - page_off, want - done);
    auto g = cte_.AsyncGetBlob(tag_id, PageName(cur), page_off, to_read,
                               /*flags*/ 0u,
                               (data_base + done).template Cast<void>(),
                               chi::PoolQuery::Local());
    CLIO_CO_AWAIT(g);
    // A miss/short read just leaves the pre-zeroed bytes as zeros (a hole is
    // not an error), so the return code is intentionally ignored here.
    done += to_read;
    cur += to_read;
  }
  task->bytes_read_ = done;
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Write(ctp::ipc::FullPtr<WriteTask> task,
                               chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_LOOKUP(fi, task->handle_);
  if (!fi) {
    task->return_code_ = EBADF;
    CLIO_CO_RETURN;
  }
  clio::cte::core::TagId tag_id = fi->tag_id_;

  // Write straight out of the task's payload — PutBlob reads from the ShmPtr we
  // pass, so hand it the right sub-region of data_ per page rather than copying
  // into a freshly-allocated staging buffer first.
  ctp::ipc::ShmPtr<char> data_base = task->data_.template Cast<char>();

  chi::u64 want = task->size_;
  chi::u64 done = 0;
  chi::u64 cur = task->offset_;
  bool ok = true;
  while (done < want) {
    chi::u64 page_off = cur % kFsPageSize;
    chi::u64 to_write = std::min(kFsPageSize - page_off, want - done);
    auto p = cte_.AsyncPutBlob(tag_id, PageName(cur), page_off, to_write,
                               (data_base + done).template Cast<void>(),
                               /*score*/ -1.0f, clio::cte::core::Context(),
                               /*flags*/ 0u, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() != 0) { ok = false; break; }
    done += to_write;
    cur += to_write;
  }

  // Advance logical size to max(size, offset + bytes_written).
  chi::u64 end = task->offset_ + done;
  chi::u64 old = fi->size_.load();
  while (end > old && !fi->size_.compare_exchange_weak(old, end)) {
  }
  task->bytes_written_ = done;
  task->new_size_ = fi->size_.load();
  task->return_code_ = ok ? 0 : EIO;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Append(ctp::ipc::FullPtr<AppendTask> task,
                                chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_LOOKUP(fi, task->handle_);
  if (!fi) {
    task->return_code_ = EBADF;
    CLIO_CO_RETURN;
  }
  clio::cte::core::TagId tag_id = fi->tag_id_;
  chi::u64 want = task->size_;

  // Deferred append (local placement). Stamp a global order — wallclock UTC
  // (primary) + per-node logical counter (tiebreak) — and write the bytes as a
  // standalone "data blob" under the file's tag. The actual merge into the file
  // tail happens later in the AppendSequence -> AppendCollect -> AppendExecution
  // pipeline, so this path does no read-modify-write of the tail and needs no
  // global coordination.
  chi::u64 logical = append_logical_.fetch_add(1) + 1;
  chi::u64 utc_ns = NowUtcNs();
  chi::u32 node_id = CLIO_IPC->GetNodeId();
  std::string data_blob_id = MakeDataBlobId(node_id, logical);

  auto p = cte_.AsyncPutBlob(tag_id, data_blob_id, 0, want, task->data_, -1.0f,
                             clio::cte::core::Context(), 0u,
                             chi::PoolQuery::Local());
  CLIO_CO_AWAIT(p);
  if (p->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }

  // Queue the pending entry; start the periodic drain on first use.
  bool need_start = false;
  {
    std::lock_guard<std::mutex> g(append_mu_);
    append_pending_.push_back(
        PendingAppend{tag_id, AppendEntry{data_blob_id, want, utc_ns, logical}});
    if (!append_seq_started_) {
      append_seq_started_ = true;
      need_start = true;
    }
  }
  if (need_start) {
    // Periodic local drain (1 ms). The future is intentionally discarded — a
    // periodic task is auto-rescheduled by the worker and never "completes".
    self_.AsyncAppendSequence(/*period_us=*/1000.0, chi::PoolQuery::Local());
  }

  // Optimistically advance the tracked logical size. For a single writer this
  // is exact; with concurrent appends across nodes the precise offset is only
  // settled when the batch is sequenced (eventual consistency), so offset_ is
  // best-effort.
  chi::u64 newsz = fi->size_.fetch_add(want) + want;
  task->offset_ = newsz - want;
  task->bytes_written_ = want;
  task->new_size_ = newsz;
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Getattr(ctp::ipc::FullPtr<GetattrTask> task,
                                 chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  std::string path = task->path_.str();

  // The empty-directory marker is an internal tag; hide it from stat().
  {
    std::string base = path;
    auto slash = base.find_last_of('/');
    if (slash != std::string::npos) base = base.substr(slash + 1);
    if (base == kDirMarker) {
      task->exists_ = 0; task->is_dir_ = 0; task->size_ = 0;
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }

  // Live logical size wins if the file is currently tracked (open files only;
  // directories are never tracked here).
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    if (it != by_path_.end()) {
      task->exists_ = 1;
      task->is_dir_ = 0;
      task->size_ = it->second->size_.load();
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }

  // Directory? Any tag with at least one direct child is a directory (real
  // children for populated dirs, the hidden marker child for empty ones).
  std::string dir = StripTrailingSlash(path);
  {
    std::string child_re = "^" + EscapeExact(dir) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(child_re, 1, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      task->exists_ = 1; task->is_dir_ = 1; task->size_ = 0;
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }

  // Regular file: an exact tag with no children. Fall back to physical size.
  auto q = cte_.AsyncTagQuery(EscapeExact(path), 1, chi::PoolQuery::Local());
  CLIO_CO_AWAIT(q);
  if (q->GetReturnCode() == 0 && !q->results_.empty()) {
    task->exists_ = 1; task->is_dir_ = 0;
    auto tag = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                        chi::PoolQuery::Local());
    CLIO_CO_AWAIT(tag);
    auto s = cte_.AsyncGetTagSize(tag->tag_id_, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(s);
    task->size_ = (s->GetReturnCode() == 0) ? s->tag_size_ : 0;
  } else {
    task->exists_ = 0; task->is_dir_ = 0; task->size_ = 0;
  }
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Truncate(ctp::ipc::FullPtr<TruncateTask> task,
                                  chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  std::string path = task->path_.str();
  chi::u64 new_size = task->new_size_;

  // Resolve the tag + old logical size and set the new logical size. Capture
  // everything under the lock, then release it BEFORE any co_await (never hold
  // a std::mutex across a coroutine suspension).
  clio::cte::core::TagId tag_id;
  chi::u64 old_size = 0;
  bool tracked = false;
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    if (it != by_path_.end()) {
      tag_id = it->second->tag_id_;
      old_size = it->second->size_.load();
      it->second->size_.store(new_size);
      tracked = !tag_id.IsNull();
    }
  }

  // Untracked path (truncate without a prior open): resolve the tag + its
  // current size, then record a tracking entry.
  if (!tracked) {
    auto t = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                      chi::PoolQuery::Local());
    CLIO_CO_AWAIT(t);
    if (t->GetReturnCode() == 0) {
      tag_id = t->tag_id_;
      auto s = cte_.AsyncGetTagSize(tag_id, chi::PoolQuery::Local());
      CLIO_CO_AWAIT(s);
      if (s->GetReturnCode() == 0) {
        old_size = s->tag_size_;
      }
    }
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    if (it == by_path_.end()) {
      auto fi = std::make_shared<FileInfo>();
      fi->tag_id_ = tag_id;
      fi->path_ = path;
      fi->size_.store(new_size);
      by_path_[path] = fi;
    } else {
      it->second->tag_id_ = tag_id;
      it->second->size_.store(new_size);
    }
  }

  // Shrink: free the page-blob data beyond new_size so the truncated bytes are
  // really gone (a later grow then reads zeros). Grow needs no blob work — the
  // Read handler zero-fills the extended region.
  if (!tag_id.IsNull() && new_size < old_size) {
    chi::u64 boundary_page = new_size / kFsPageSize;
    chi::u64 boundary_off = new_size % kFsPageSize;
    // Trim the boundary page to its surviving prefix (frees the tail).
    auto tb = cte_.AsyncTruncateBlob(tag_id, std::to_string(boundary_page),
                                     boundary_off, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(tb);
    // Delete whole pages beyond the boundary, up to the old last page.
    chi::u64 last_page = (old_size == 0) ? 0 : (old_size - 1) / kFsPageSize;
    for (chi::u64 p = boundary_page + 1; p <= last_page; ++p) {
      auto d = cte_.AsyncDelBlob(tag_id, std::to_string(p),
                                 chi::PoolQuery::Local());
      CLIO_CO_AWAIT(d);
    }
  }

  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Unlink(ctp::ipc::FullPtr<UnlinkTask> task,
                                chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());

  // Refuse to unlink a directory (a tag with children) — that is rmdir's job.
  {
    std::string child_re = "^" + EscapeExact(path) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(child_re, 1, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      task->return_code_ = EISDIR;
      CLIO_CO_RETURN;
    }
  }

  // DelTag is hierarchy-aware: a hard-link (alias) path unlinks only that name;
  // the canonical path removes the file and all its remaining links + blobs.
  auto d = cte_.AsyncDelTag(path, chi::PoolQuery::Local());
  CLIO_CO_AWAIT(d);
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    by_path_.erase(path);
  }
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Mkdir(ctp::ipc::FullPtr<MkdirTask> task,
                               chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());

  // EEXIST if the path already exists as a directory (has a child) or a file
  // (an exact tag).
  {
    std::string child_re = "^" + EscapeExact(path) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(child_re, 1, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      task->return_code_ = EEXIST;
      CLIO_CO_RETURN;
    }
  }
  {
    auto q = cte_.AsyncTagQuery(EscapeExact(path), 1, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      task->return_code_ = EEXIST;
      CLIO_CO_RETURN;
    }
  }

  // Create the directory by giving it one hidden marker child. This also
  // creates the directory's own hierarchical tag (the parent chain), so the
  // directory becomes detectable as "a tag with a child".
  auto t = cte_.AsyncGetOrCreateTag(path + "/" + kDirMarker,
                                    clio::cte::core::TagId::GetNull(),
                                    chi::PoolQuery::Local());
  CLIO_CO_AWAIT(t);
  task->return_code_ = (t->GetReturnCode() == 0) ? 0 : EIO;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Rmdir(ctp::ipc::FullPtr<RmdirTask> task,
                               chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());

  // List direct children. A directory must have at least one child (the marker,
  // for an empty dir); any NON-marker child means the directory is not empty.
  std::string child_re = "^" + EscapeExact(path) + "/[^/]+$";
  auto q = cte_.AsyncTagQuery(child_re, 0, chi::PoolQuery::Local());
  CLIO_CO_AWAIT(q);
  if (q->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }
  bool is_dir = false;
  const size_t prefix = path.size() + 1;  // strip "<path>/"
  for (const auto &full : q->results_) {
    is_dir = true;
    std::string base = full.size() > prefix ? full.substr(prefix) : std::string();
    if (base != kDirMarker) {
      task->return_code_ = ENOTEMPTY;
      CLIO_CO_RETURN;
    }
  }
  if (!is_dir) {
    task->return_code_ = ENOENT;  // not a directory (or doesn't exist)
    CLIO_CO_RETURN;
  }

  // Empty directory: recursive DelTag removes its tag and the marker child.
  auto d = cte_.AsyncDelTag(path, chi::PoolQuery::Local());
  CLIO_CO_AWAIT(d);
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Rename(ctp::ipc::FullPtr<RenameTask> task,
                                chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  std::string src = task->src_.str();
  std::string dst = task->dst_.str();
  if (src == dst) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  // Resolve the source tag (its name is the path). Renaming the tag keeps its
  // TagId, so every page-blob (keyed by TagId) moves with it — no data copy.
  clio::cte::core::TagId src_tag;
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(src);
    if (it != by_path_.end()) {
      src_tag = it->second->tag_id_;
    }
  }
  if (src_tag.IsNull()) {
    auto q = cte_.AsyncTagQuery(EscapeExact(src), 1, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      auto t = cte_.AsyncGetOrCreateTag(src, clio::cte::core::TagId::GetNull(),
                                        chi::PoolQuery::Local());
      CLIO_CO_AWAIT(t);
      if (t->GetReturnCode() == 0) {
        src_tag = t->tag_id_;
      }
    }
  }
  if (src_tag.IsNull()) {
    task->return_code_ = ENOENT;
    CLIO_CO_RETURN;
  }

  // POSIX rename overwrites an existing destination: drop dst's tag first.
  {
    auto d = cte_.AsyncDelTag(dst, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(d);
  }
  // Rename the tag in place (keeps TagId + blobs).
  auto r = cte_.AsyncRenameTag(src, dst, src_tag, chi::PoolQuery::Local());
  CLIO_CO_AWAIT(r);
  if (r->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }

  // Move the per-file logical-size metadata entry to the new path.
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(src);
    if (it != by_path_.end()) {
      it->second->path_ = dst;
      by_path_[dst] = it->second;
      by_path_.erase(it);
    }
  }
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Link(ctp::ipc::FullPtr<LinkTask> task,
                              chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  std::string target = StripTrailingSlash(task->target_.str());
  std::string link = StripTrailingSlash(task->link_.str());

  // A hard link must not land on an existing name.
  {
    auto q = cte_.AsyncTagQuery(EscapeExact(link), 1, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      task->return_code_ = EEXIST;
      CLIO_CO_RETURN;
    }
  }

  // Bind `link` as an alias of `target`'s tag. GetOrCreateTagAlias resolves the
  // target by path, creates `link`'s parent chain, and binds the relative key
  // for `link` to the target's tag id — so both paths share the same data.
  // found_ == 0 means the target did not exist.
  auto a = cte_.AsyncGetOrCreateTagAlias(target, link, chi::PoolQuery::Local());
  CLIO_CO_AWAIT(a);
  if (a->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }
  task->return_code_ = (a->found_ == 1) ? 0 : ENOENT;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Readdir(ctp::ipc::FullPtr<ReaddirTask> task,
                                 chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  // Direct children: tags whose resolved name is "<dir>/<name>" with no
  // further slash. Returns full resolved paths; the adapter strips the prefix.
  std::string dir = task->path_.str();
  if (dir.empty() || dir.back() != '/') dir += '/';
  std::string regex = "^" + EscapeExact(dir) + "[^/]+$";
  auto q = cte_.AsyncTagQuery(regex, 0, chi::PoolQuery::Local());
  CLIO_CO_AWAIT(q);
  task->entries_ = chi::priv::vector<chi::priv::string>(CTP_MALLOC);
  if (q->GetReturnCode() == 0) {
    const size_t prefix = dir.size();
    for (auto &name : q->results_) {
      // Hide the internal empty-directory marker.
      std::string base = name.size() > prefix ? name.substr(prefix) : name;
      if (base == kDirMarker) continue;
      task->entries_.push_back(chi::priv::string(CTP_MALLOC, name));
    }
  }
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::StatSize(ctp::ipc::FullPtr<StatSizeTask> task,
                                  chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  std::string path = task->path_.str();
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    if (it != by_path_.end()) {
      task->exists_ = 1;
      task->size_ = it->second->size_.load();
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }
  auto qy = cte_.AsyncTagQuery(EscapeExact(path), 1, chi::PoolQuery::Local());
  CLIO_CO_AWAIT(qy);
  if (qy->GetReturnCode() == 0 && !qy->results_.empty()) {
    task->exists_ = 1;
    auto tag = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                        chi::PoolQuery::Local());
    CLIO_CO_AWAIT(tag);
    auto s = cte_.AsyncGetTagSize(tag->tag_id_, chi::PoolQuery::Local());
    CLIO_CO_AWAIT(s);
    task->size_ = (s->GetReturnCode() == 0) ? s->tag_size_ : 0;
  } else {
    task->exists_ = 0;
    task->size_ = 0;
  }
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Deferred-append pipeline handlers
// ===========================================================================

chi::TaskResume Runtime::AppendSequence(
    ctp::ipc::FullPtr<AppendSequenceTask> task, chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  // Drain the per-node pending queue, then group entries by tag.
  std::vector<PendingAppend> drained;
  {
    std::lock_guard<std::mutex> g(append_mu_);
    drained.swap(append_pending_);
  }
  if (drained.empty()) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  std::unordered_map<clio::cte::core::TagId, std::vector<AppendEntry>> by_tag;
  for (auto &pa : drained) {
    by_tag[pa.tag_id_].push_back(pa.entry_);
  }

  // One AppendCollect per tag, routed ManyToOne so every node's batch for the
  // same tag aggregates at that tag's sequencer (the leader chosen by the
  // container hash). batch_key = the tag id keeps distinct tags' collectives
  // separate on the same leader.
  std::vector<chi::Future<AppendCollectTask>> futs;
  futs.reserve(by_tag.size());
  for (auto &kv : by_tag) {
    const clio::cte::core::TagId &tag = kv.first;
    chi::u32 chash = static_cast<chi::u32>(
        std::hash<clio::cte::core::TagId>()(tag));
    chi::u64 bkey = (static_cast<chi::u64>(tag.major_) << 32) | tag.minor_;
    auto q = chi::PoolQuery::ManyToOne(chash, bkey, /*batch_for_ns=*/50000);
    futs.push_back(self_.AsyncAppendCollect(tag, kv.second, q));
  }
  for (auto &f : futs) {
    CLIO_CO_AWAIT(f);
  }
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::AppendCollect(
    ctp::ipc::FullPtr<AppendCollectTask> task, chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  // This runs ONCE per batch (the ManyToOne aggregate). task->entries_ already
  // holds every node's pending entries for this tag (combined via AggregateIn).
  clio::cte::core::TagId tag_id = task->tag_id_;
  std::vector<AppendEntry> entries(task->entries_.begin(),
                                   task->entries_.end());
  // Global order: UTC first, logical counter as tiebreak.
  std::sort(entries.begin(), entries.end(),
            [](const AppendEntry &a, const AppendEntry &b) {
              if (a.utc_ns_ != b.utc_ns_) return a.utc_ns_ < b.utc_ns_;
              return a.logical_ < b.logical_;
            });

  chi::u64 batch_total = 0;
  for (auto &e : entries) batch_total += e.data_blob_size_;

  // Tail of the file. GetTagSize (broadcast) sums all blobs under the tag,
  // which still includes this batch's staged data blobs — subtract them to get
  // the true file content size to append at. (Concurrent batches for the same
  // tag are serialized by ManyToOne batching; cross-batch size tracking is a
  // separate hardening issue.)
  chi::u64 cur_size = 0;
  {
    auto s = cte_.AsyncGetTagSize(tag_id, chi::PoolQuery::Broadcast());
    CLIO_CO_AWAIT(s);
    chi::u64 total = (s->GetReturnCode() == 0) ? s->tag_size_ : 0;
    cur_size = (total >= batch_total) ? (total - batch_total) : 0;
  }

  // Build the merge plan: lay each data blob out contiguously starting at
  // cur_size, splitting on 1 MiB file-page boundaries.
  std::vector<AppendPlanStep> plan;
  chi::u64 file_off = cur_size;
  for (auto &e : entries) {
    chi::u64 remaining = e.data_blob_size_;
    chi::u64 doff = 0;
    while (remaining > 0) {
      chi::u64 page = file_off / kFsPageSize;
      chi::u64 page_off = file_off % kFsPageSize;
      chi::u64 step = std::min(kFsPageSize - page_off, remaining);
      plan.push_back(AppendPlanStep{page, e.data_blob_id_, page_off, doff, step,
                                    e.data_blob_size_});
      file_off += step;
      doff += step;
      remaining -= step;
    }
  }

  // Split into AppendExecution slices of up to 16 MiB. A data blob is never
  // split across two slices, so exactly one execution task DelBlobs it (no
  // double-free race); a single blob larger than 16 MiB forms its own slice.
  constexpr chi::u64 kMaxExecBytes = 16ull * 1024 * 1024;
  std::vector<chi::Future<AppendExecutionTask>> futs;
  chi::u32 spread = 0;
  size_t i = 0;
  while (i < plan.size()) {
    std::vector<AppendPlanStep> slice;
    chi::u64 bytes = 0;
    while (i < plan.size()) {
      slice.push_back(plan[i]);
      bytes += plan[i].size_;
      ++i;
      // Stop at a data-blob boundary once we've reached the size target.
      bool at_blob_boundary =
          (i >= plan.size()) ||
          (plan[i].data_blob_id_ != slice.back().data_blob_id_);
      if (bytes >= kMaxExecBytes && at_blob_boundary) break;
    }
    auto q = chi::PoolQuery::DirectHash(spread++);
    futs.push_back(self_.AsyncAppendExecution(tag_id, slice, q));
  }
  for (auto &f : futs) {
    CLIO_CO_AWAIT(f);
  }

  task->new_size_ = file_off;
  task->return_code_ = 0;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::AppendExecution(
    ctp::ipc::FullPtr<AppendExecutionTask> task, chi::RunContext &ctx) {
  CLIO_TASK_BODY_BEGIN
  clio::cte::core::TagId tag_id = task->tag_id_;
  auto *ipc = CLIO_IPC;
  const size_t n = task->steps_.size();
  bool ok = true;

  // Allocate one staging buffer per step up front so buffers/futures stay
  // index-aligned with steps_. If any allocation fails, unwind and bail.
  std::vector<ctp::ipc::FullPtr<char>> bufs;
  bufs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    ctp::ipc::FullPtr<char> b = ipc->AllocateBuffer(task->steps_[i].size_);
    if (b.IsNull()) { ok = false; break; }
    bufs.push_back(b);
  }
  if (!ok) {
    for (auto &b : bufs) ipc->FreeBuffer(b);
    task->return_code_ = ENOMEM;
    CLIO_CO_RETURN;
  }

  // 1) GetBlob each step's data slice in parallel.
  std::vector<chi::Future<clio::cte::core::GetBlobTask>> gets;
  gets.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const AppendPlanStep &s = task->steps_[i];
    gets.push_back(cte_.AsyncGetBlob(tag_id, s.data_blob_id_, s.off_in_data_,
                                     s.size_, 0u,
                                     bufs[i].shm_.template Cast<void>(),
                                     chi::PoolQuery::Local()));
  }

  // 2) As each Get completes, PutBlob the bytes into the destination file page.
  std::vector<chi::Future<clio::cte::core::PutBlobTask>> puts;
  puts.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    CLIO_CO_AWAIT(gets[i]);  // a short/hole read just leaves zeros
    const AppendPlanStep &s = task->steps_[i];
    puts.push_back(cte_.AsyncPutBlob(
        tag_id, std::to_string(s.file_page_), s.off_in_page_, s.size_,
        bufs[i].shm_.template Cast<void>(), -1.0f, clio::cte::core::Context(),
        0u, chi::PoolQuery::Local()));
  }
  for (auto &p : puts) {
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() != 0) ok = false;
  }
  for (auto &b : bufs) ipc->FreeBuffer(b);

  // 3) DelBlob each distinct staged data blob (once), then await all. Because a
  // data blob's steps are confined to one execution task, this is the only task
  // that deletes it — no double-free.
  std::unordered_set<std::string> seen;
  std::vector<chi::Future<clio::cte::core::DelBlobTask>> dels;
  for (const auto &s : task->steps_) {
    if (seen.insert(s.data_blob_id_).second) {
      dels.push_back(
          cte_.AsyncDelBlob(tag_id, s.data_blob_id_, chi::PoolQuery::Local()));
    }
  }
  for (auto &d : dels) {
    CLIO_CO_AWAIT(d);
  }

  task->return_code_ = ok ? 0 : EIO;
  (void)ctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

#undef CLIO_FS_LOOKUP

}  // namespace clio::cte::filesystem

// Define ChiMod entry points (alloc/new/name/destroy) so the runtime's module
// manager can dlopen and instantiate this chimod.
CLIO_TASK_CC(clio::cte::filesystem::Runtime)
