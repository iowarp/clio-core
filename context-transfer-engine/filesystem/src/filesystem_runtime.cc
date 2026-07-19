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
#include <utility>
#include <vector>

#include <clio_cte/filesystem/filesystem_runtime.h>

namespace clio::cte::filesystem {

namespace {
/** UTC wallclock nanoseconds (system_clock) — the primary append order key. */
inline clio::run::u64 NowUtcNs() {
  return static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

/**
 * Reserved name for a staged append data blob, under the file's own tag. The
 * "_append/" prefix can't collide with the numeric page-blob names ("0","1",
 * ...); node id keeps it unique across nodes, the logical counter within a node.
 */
inline std::string MakeDataBlobId(clio::run::u32 node_id, clio::run::u64 logical) {
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
inline std::string PageName(clio::run::u64 off) {
  return std::to_string(off / kFsPageSize);
}

/** Strip a single trailing '/' from a path (keeping a bare root "/"). */
inline std::string StripTrailingSlash(std::string p) {
  if (p.size() > 1 && p.back() == '/') p.pop_back();
  return p;
}
/** Parent directory of an absolute path ("/a/b/c" -> "/a/b", "/a" -> "/"). */
inline std::string ParentDir(const std::string &p) {
  std::string s = StripTrailingSlash(p);
  auto slash = s.find_last_of('/');
  if (slash == std::string::npos || slash == 0) return "/";
  return s.substr(0, slash);
}
// A blob name used only to stamp a tag's timestamps: it is never a real page
// (pages are named by number), so a TruncateBlob of it finds the blob missing
// and just bumps the tag's mtime/ctime without touching any data.
inline const char *TsTouchBlob() { return "__clio_ts_touch__"; }

// Reserved blob under a symlink's tag holding the link target string. Its
// presence (non-empty) is what marks a tag as a symlink (S_IFLNK) at getattr.
inline const char *SymlinkMarker() { return "__clio_symlink__"; }

// Bump a directory's mtime/ctime after a child is added or removed (POSIX
// updates the parent dir's times on create/unlink/rename/mkdir/rmdir/link).
// Resolves the dir's tag and stamps it via a no-op truncate of a sentinel blob
// (see TsTouchBlob). Uses the local cte_ client; must be used in a task body.
#define CLIO_FS_TOUCH_DIR(dirpath)                                          \
  do {                                                                       \
    auto _tp = cte_.AsyncGetOrCreateTag(                                     \
        (dirpath), clio::cte::core::TagId::GetNull(),                        \
        clio::run::PoolQuery::Dynamic());                                    \
    CLIO_CO_AWAIT(_tp);                                                      \
    if (_tp->GetReturnCode() == 0) {                                         \
      auto _tt = cte_.AsyncTruncateBlob(_tp->tag_id_, TsTouchBlob(), 0,      \
                                        clio::run::PoolQuery::Dynamic());    \
      CLIO_CO_AWAIT(_tt);                                                    \
    }                                                                        \
  } while (0)
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
/** Anchored exact-match pattern for a single path. Anchoring with ^...$ makes
 *  the query an exact match (not a substring/`regex_search`), which is what
 *  getattr/lookup/rename want; it also lets the CTE core serve it from the
 *  O(1) tag-name hash instead of compiling a std::regex per call (#680). */
inline std::string ExactRe(const std::string &s) {
  return "^" + EscapeExact(s) + "$";
}

// ---- cluster-wide routing (issue #685) -------------------------------------
// Every CTE-core operation is issued with PoolQuery::Dynamic(): the CTE core's
// Monitor (Runtime::ScheduleTask) resolves each Dynamic task to the right
// concrete routing for its method — creates/lookups converge on one owner
// container cluster-wide, blob ops hash to their owning container, and queries
// fan out — so a path created on node A is visible from node B. The runtime is
// the single source of routing truth; the filesystem no longer hand-rolls a
// per-path DirectHash here.
/** Stable inode = packed TagId ((major<<32)|minor). A tag's id is fixed for its
 *  lifetime and unique, so this is a stable, collision-free inode; hard-link
 *  aliases share the TagId and thus correctly share an inode. 0 maps to 1 since
 *  st_ino==0 means "no inode". The CTE core already packs the same way in
 *  TagQueryTask::result_ids_, so getattr and readdir agree. */
inline clio::run::u64 InoFromPacked(clio::run::u64 packed) { return packed ? packed : 1; }
inline clio::run::u64 InoFromTag(const clio::cte::core::TagId &t) {
  return InoFromPacked((static_cast<clio::run::u64>(t.major_) << 32) |
                       static_cast<clio::run::u64>(t.minor_));
}

// ---- extended-attribute (xattr) blob (de)serialization ----
// A file's xattrs are stored as ONE blob under the global xattr tag, named by
// the file's packed tag id (decimal). Payload = repeated records
// [u32 name_len][name][u32 val_len][val], little-endian u32 via memcpy (values
// may hold NULs). Empty/missing blob == no xattrs.
inline std::string XattrKey(const clio::cte::core::TagId &t) {
  return std::to_string((static_cast<clio::run::u64>(t.major_) << 32) |
                        static_cast<clio::run::u64>(t.minor_));
}
inline void PutU32(std::string &out, clio::run::u32 v) {
  char b[4];
  std::memcpy(b, &v, 4);  // host is little-endian on all supported targets
  out.append(b, 4);
}
inline std::string SerializeXattrs(
    const std::vector<std::pair<std::string, std::string>> &xa) {
  std::string out;
  for (const auto &kv : xa) {
    PutU32(out, static_cast<clio::run::u32>(kv.first.size()));
    out.append(kv.first.data(), kv.first.size());
    PutU32(out, static_cast<clio::run::u32>(kv.second.size()));
    out.append(kv.second.data(), kv.second.size());
  }
  return out;
}
inline std::vector<std::pair<std::string, std::string>> DeserializeXattrs(
    const char *data, size_t len) {
  std::vector<std::pair<std::string, std::string>> xa;
  size_t pos = 0;
  while (pos + 4 <= len) {
    clio::run::u32 nlen = 0;
    std::memcpy(&nlen, data + pos, 4);
    pos += 4;
    if (pos + nlen > len) break;
    std::string name(data + pos, nlen);
    pos += nlen;
    if (pos + 4 > len) break;
    clio::run::u32 vlen = 0;
    std::memcpy(&vlen, data + pos, 4);
    pos += 4;
    if (pos + vlen > len) break;
    std::string val(data + pos, vlen);
    pos += vlen;
    xa.emplace_back(std::move(name), std::move(val));
  }
  return xa;
}
}  // namespace

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
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

  // Resolve the global append-staging tag (shared by all files). Append data
  // blobs live here, so they don't inflate any file's GetTagSize. Flat tag
  // name (no leading '/') => not hierarchical.
  {
    auto st = cte_.AsyncGetOrCreateTag("_clio_append_staging",
                                       clio::cte::core::TagId::GetNull(),
                                       clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(st);
    if (st->GetReturnCode() == 0) {
      staging_tag_id_ = st->tag_id_;
    }
  }

  // Resolve the global xattr-store tag (shared by all files). Each file's
  // xattrs live in ONE serialized blob here, named by the file's packed tag id,
  // so they never inflate any file's GetTagSize (st_size). Flat tag name.
  {
    auto xt = cte_.AsyncGetOrCreateTag("_clio_xattr_store",
                                       clio::cte::core::TagId::GetNull(),
                                       clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(xt);
    if (xt->GetReturnCode() == 0) {
      xattr_tag_id_ = xt->tag_id_;
    }
  }
  HLOG(kInfo, "filesystem: Create over CTE core pool {}",
       next_pool_id_.ToString());
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Open(clio::run::shared_ptr<OpenTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = task->path_.str();

  // Did the tag already exist (so we can report created_)?
  bool existed = false;
  {
    auto q = cte_.AsyncTagQuery(ExactRe(path), 1, clio::run::PoolQuery::Dynamic());
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
    CLIO_CO_RETURN;
  }

  // Resolve / create the tag for this path.
  auto t = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                    clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(t);
  if (t->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }
  clio::cte::core::TagId tag_id = t->tag_id_;

  // Current physical size (best-effort baseline for the logical size).
  clio::run::u64 size = 0;
  {
    auto s = cte_.AsyncGetTagSize(tag_id, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(s);
    if (s->GetReturnCode() == 0) {
      size = s->tag_size_;
    }
  }

  // Register the handle + per-file logical size (source of truth henceforth).
  clio::run::u64 handle = next_handle_.fetch_add(1);
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    std::shared_ptr<FileInfo> fi;
    // Reuse the cached entry ONLY if it still tracks the tag this path resolves
    // to now. A by_path_ entry can outlive its tag (path key does not follow an
    // ancestor rename; a delete via the renamed path clears a different key), so
    // an entry left over from a since-deleted file would otherwise be reused with
    // its DEAD tag_id_ — every read/write on the freshly O_CREAT'd file would hit
    // the old, gone tag (generic/023). When the cached tag differs from the
    // freshly resolved one, bind a new FileInfo to the correct tag, replacing the
    // stale mapping (any still-open handle keeps its own shared_ptr, so an
    // open-unlink-recreate on the same name stays correctly separated).
    if (it != by_path_.end() && it->second->tag_id_ == tag_id) {
      fi = it->second;
      size = fi->size_.load();  // keep the live logical size if already open
    } else {
      fi = std::make_shared<FileInfo>();
      fi->tag_id_ = tag_id;
      fi->path_ = path;
      fi->size_.store(size);
      by_path_[path] = fi;
    }
    // A fresh O_CREAT carries the caller's mode (cp/install rely on this to
    // make copied binaries executable — getattr otherwise synthesizes 0644).
    if (!existed) fi->set_mode_ = task->mode_ & 07777u;
    handles_[handle] = fi;
  }

  task->handle_ = handle;
  task->size_ = size;
  task->created_ = existed ? 0u : 1u;
  // Creating a new file updates its parent directory's mtime/ctime.
  if (!existed) {
    CLIO_FS_TOUCH_DIR(ParentDir(path));
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Close(clio::run::shared_ptr<CloseTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    handles_.erase(task->handle_);
  }
  task->return_code_ = 0;
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

clio::run::TaskResume Runtime::Read(clio::run::shared_ptr<ReadTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_LOOKUP(fi, task->handle_);
  if (!fi) {
    task->return_code_ = EBADF;
    CLIO_CO_RETURN;
  }
  clio::cte::core::TagId tag_id = fi->tag_id_;
  clio::run::u64 file_size = fi->size_.load();

  auto *ipc = CLIO_IPC;
  // Read directly into the task's payload — GetBlob copies into the ShmPtr we
  // hand it, so point it at the right sub-region of data_ each page instead of
  // staging through a freshly-allocated buffer.
  ctp::ipc::ShmPtr<char> data_base = task->data_.template Cast<char>();
  char *dst = ipc->ToFullPtr<char>(data_base).ptr_;

  clio::run::u64 offset = task->offset_;
  clio::run::u64 want = task->size_;
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

  clio::run::u64 done = 0;
  clio::run::u64 cur = offset;
  while (done < want) {
    clio::run::u64 page_off = cur % kFsPageSize;
    clio::run::u64 to_read = std::min(kFsPageSize - page_off, want - done);
    auto g = cte_.AsyncGetBlob(tag_id, PageName(cur), page_off, to_read,
                               /*flags*/ 0u,
                               (data_base + done).template Cast<void>(),
                               clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(g);
    // A miss/short read just leaves the pre-zeroed bytes as zeros (a hole is
    // not an error), so the return code is intentionally ignored here.
    done += to_read;
    cur += to_read;
  }
  task->bytes_read_ = done;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Write(clio::run::shared_ptr<WriteTask> &task) {
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

  clio::run::u64 want = task->size_;
  clio::run::u64 done = 0;
  clio::run::u64 cur = task->offset_;
  bool ok = true;
  while (done < want) {
    clio::run::u64 page_off = cur % kFsPageSize;
    clio::run::u64 to_write = std::min(kFsPageSize - page_off, want - done);
    auto p = cte_.AsyncPutBlob(tag_id, PageName(cur), page_off, to_write,
                               (data_base + done).template Cast<void>(),
                               /*score*/ -1.0f, clio::cte::core::Context(),
                               /*flags*/ 0u, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() != 0) { ok = false; break; }
    done += to_write;
    cur += to_write;
  }

  // Advance logical size to max(size, offset + bytes_written).
  clio::run::u64 end = task->offset_ + done;
  clio::run::u64 old = fi->size_.load();
  while (end > old && !fi->size_.compare_exchange_weak(old, end)) {
  }
  // A write re-establishes the natural mtime/ctime, so drop any utimens
  // overrides (fast unlocked check keeps this off the hot path when unused).
  if (fi->set_atime_ || fi->set_mtime_ || fi->set_ctime_) {
    std::lock_guard<std::mutex> g(meta_mu_);
    fi->set_atime_ = 0; fi->set_mtime_ = 0; fi->set_ctime_ = 0;
  }
  task->bytes_written_ = done;
  task->new_size_ = fi->size_.load();
  task->return_code_ = ok ? 0 : EIO;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Append(clio::run::shared_ptr<AppendTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_FS_LOOKUP(fi, task->handle_);
  if (!fi) {
    task->return_code_ = EBADF;
    CLIO_CO_RETURN;
  }
  clio::cte::core::TagId tag_id = fi->tag_id_;
  clio::run::u64 want = task->size_;

  // Deferred append (local placement). Stamp a global order — wallclock UTC
  // (primary) + per-node logical counter (tiebreak) — and write the bytes as a
  // standalone "data blob" under the file's tag. The actual merge into the file
  // tail happens later in the AppendSequence -> AppendCollect -> AppendExecution
  // pipeline, so this path does no read-modify-write of the tail and needs no
  // global coordination.
  clio::run::u64 logical = append_logical_.fetch_add(1) + 1;
  clio::run::u64 utc_ns = NowUtcNs();
  clio::run::u32 node_id = CLIO_IPC->GetNodeId();
  std::string data_blob_id = MakeDataBlobId(node_id, logical);

  // Stage the bytes under the shared staging tag (NOT the file tag), so the
  // file's GetTagSize stays equal to its merged content (the true tail).
  auto p = cte_.AsyncPutBlob(staging_tag_id_, data_blob_id, 0, want,
                             task->data_, -1.0f, clio::cte::core::Context(), 0u,
                             clio::run::PoolQuery::Dynamic());
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
    self_.AsyncAppendSequence(/*period_us=*/1000.0, clio::run::PoolQuery::Local());
  }

  // Optimistically advance the tracked logical size. For a single writer this
  // is exact; with concurrent appends across nodes the precise offset is only
  // settled when the batch is sequenced (eventual consistency), so offset_ is
  // best-effort.
  clio::run::u64 newsz = fi->size_.fetch_add(want) + want;
  task->offset_ = newsz - want;
  task->bytes_written_ = want;
  task->new_size_ = newsz;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Getattr(clio::run::shared_ptr<GetattrTask> &task) {
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
    bool tracked = false;
    clio::run::u64 live_size = 0;
    clio::cte::core::TagId h_tag = clio::cte::core::TagId::GetNull();
    clio::run::u64 ov_atime = 0, ov_mtime = 0, ov_ctime = 0;
    clio::run::u32 ov_uid = 0xFFFFFFFFu, ov_gid = 0xFFFFFFFFu;
    clio::run::u32 ov_mode = 0xFFFFFFFFu;
    {
      std::lock_guard<std::mutex> g(meta_mu_);
      auto it = by_path_.find(path);
      if (it != by_path_.end()) {
        tracked = true;
        live_size = it->second->size_.load();
        h_tag = it->second->tag_id_;
        ov_atime = it->second->set_atime_;
        ov_mtime = it->second->set_mtime_;
        ov_ctime = it->second->set_ctime_;
        ov_uid = it->second->set_uid_;
        ov_gid = it->second->set_gid_;
        ov_mode = it->second->set_mode_;
      }
    }
    if (tracked) {
      // Validate the cached entry against core BEFORE trusting it. A by_path_ key
      // is the absolute path *string*, so the entry does NOT follow the file when
      // an ancestor directory is renamed, and a delete through the post-rename
      // path clears a different key. Such a stale entry (a real, now-deleted tag
      // id) would otherwise make getattr report a long-gone file as still
      // present — the kernel then routes a fresh create() to open() and gets
      // ENOENT, permanently wedging that name (generic/023 tree/regu, tree/tree:
      // `mkdir d; echo>d/bar; rename d e; rm e/bar; rmdir e; mkdir d; echo>d/bar`
      // failed because getattr(/d/bar) still resolved through the orphaned
      // "/d/bar" entry). GetTagSize returns nonzero once the tag is gone, so on
      // failure we drop the stale entry and fall through to the authoritative
      // path resolution below. A genuinely open-but-unlinked file keeps its tag
      // alive, so its GetTagSize still succeeds and it stays tracked.
      bool stale = false;
      bool have_ts = false;
      clio::run::u64 g_ctime = 0, g_mtime = 0, g_atime = 0;
      if (!h_tag.IsNull()) {
        auto s = cte_.AsyncGetTagSize(h_tag, clio::run::PoolQuery::Dynamic());
        CLIO_CO_AWAIT(s);
        if (s->GetReturnCode() == 0) {
          g_ctime = s->ctime_;
          g_mtime = s->mtime_;
          g_atime = s->atime_;
          have_ts = true;
        } else {
          stale = true;
        }
      }
      if (stale) {
        std::lock_guard<std::mutex> g(meta_mu_);
        // Only drop it if it still maps to the same dead tag — a concurrent
        // reopen may have rebound this path to a fresh, live FileInfo.
        auto it = by_path_.find(path);
        if (it != by_path_.end() && it->second->tag_id_ == h_tag) {
          by_path_.erase(it);
        }
        // fall through to normal resolution (do not CLIO_CO_RETURN here)
      } else {
        task->exists_ = 1;
        task->is_dir_ = 0;
        task->size_ = live_size;
        task->ino_ = InoFromTag(h_tag);
        // Timestamps (ctime/mtime/atime) from the tag.
        if (have_ts) {
          task->ctime_ = g_ctime;
          task->mtime_ = g_mtime;
          task->atime_ = g_atime;
        }
        // utimens atime/mtime overrides win over the natural tag timestamps until
        // the next write/truncate clears them (utimens can set those to
        // arbitrary, even past, values).
        if (ov_mtime != 0) task->mtime_ = ov_mtime;
        if (ov_atime != 0) task->atime_ = ov_atime;
        // ctime, however, can ONLY advance: POSIX ties it to the last metadata
        // change and forbids setting it to an arbitrary value. The utimens
        // override records the ctime bump at utimens time, but a later metadata
        // op (e.g. adding/removing a hard link) advances the tag's live
        // last_changed_ beyond it — so take the max. Letting the frozen override
        // win instead masked those real ctime changes (generic/755: ctime
        // unchanged after unlinking a hard link).
        if (ov_ctime > task->ctime_) task->ctime_ = ov_ctime;
        // chown owner overrides (0xFFFFFFFF if never chown'd => adapter defaults
        // to getuid()/getgid()).
        task->uid_ = ov_uid;
        task->gid_ = ov_gid;
        // chmod/create mode override (0xFFFFFFFF => adapter synthesizes 0644).
        task->mode_ = ov_mode;
        task->return_code_ = 0;
        CLIO_CO_RETURN;
      }
    }
  }

  // Directory? A tag with any direct child is a directory. Two-step probe
  // (#680): (1) the hidden marker child of a mkdir'd dir via an O(1) exact
  // lookup -- this covers every FUSE directory and, crucially, avoids an
  // O(children) child scan on a populated dir (getattr is the stat hot path;
  // t_mtab keeps 1000s of files in one dir, so the scan made generic/100 ~5x
  // slower). (2) Fallback for a dir created IMPLICITLY as the parent of a
  // descendant file (no marker): match any child. This is cheap for files and
  // empty dirs (no children => empty result); only markerless populated dirs
  // pay the scan, which the FUSE path never produces (it always mkdir's).
  std::string dir = StripTrailingSlash(path);
  {
    bool is_directory = false;
    {
      auto qm = cte_.AsyncTagQuery(ExactRe(dir + "/" + kDirMarker), 1,
                                   clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(qm);
      is_directory = (qm->GetReturnCode() == 0 && !qm->results_.empty());
    }
    if (!is_directory) {
      std::string child_re = "^" + EscapeExact(dir) + "/[^/]+$";
      auto qc = cte_.AsyncTagQuery(child_re, 1, clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(qc);
      is_directory = (qc->GetReturnCode() == 0 && !qc->results_.empty());
    }
    if (is_directory) {
      task->exists_ = 1; task->is_dir_ = 1; task->size_ = 0;
      // Resolve the directory's own tag to give it a stable inode.
      auto tag = cte_.AsyncGetOrCreateTag(
          dir, clio::cte::core::TagId::GetNull(), clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(tag);
      task->ino_ = InoFromTag(tag->tag_id_);
      // Directories are tags too, so they carry ctime/mtime/atime. Surface them
      // (one extra GetTagSize) so e.g. mv-into-dir shows up as a dir mtime/ctime
      // change (generic/309). Child-modifying ops bump the parent dir's tag.
      auto ds = cte_.AsyncGetTagSize(tag->tag_id_, clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(ds);
      if (ds->GetReturnCode() == 0) {
        task->ctime_ = ds->ctime_;
        task->mtime_ = ds->mtime_;
        task->atime_ = ds->atime_;
      }
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }

  // Regular file: an exact tag with no children. Fall back to physical size.
  auto q = cte_.AsyncTagQuery(ExactRe(path), 1, clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(q);
  if (q->GetReturnCode() == 0 && !q->results_.empty()) {
    task->exists_ = 1; task->is_dir_ = 0;
    // The exact query already resolved the tag id (result_ids_), so read the
    // size/timestamps directly instead of a redundant GetOrCreateTag round-trip
    // -- one fewer sequential RPC on the stat hot path (#680).
    clio::run::u64 packed = q->result_ids_.empty() ? 0 : q->result_ids_[0];
    clio::cte::core::TagId tid(static_cast<clio::run::u32>(packed >> 32),
                               static_cast<clio::run::u32>(packed & 0xffffffffULL));
    auto s = cte_.AsyncGetTagSize(tid, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(s);
    task->size_ = (s->GetReturnCode() == 0) ? s->tag_size_ : 0;
    task->ino_ = InoFromPacked(packed);
    if (s->GetReturnCode() == 0) {
      task->ctime_ = s->ctime_;
      task->mtime_ = s->mtime_;
      task->atime_ = s->atime_;
    }
    // Symlink probe: only in the exact-file branch (one extra RPC per file
    // stat). A tag carrying the reserved marker blob is a symlink; its size is
    // the target length (S_IFLNK). Non-symlinks pay one GetBlobSize miss.
    auto sm = cte_.AsyncGetBlobSize(tid, SymlinkMarker(), clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(sm);
    if (sm->GetReturnCode() == 0 && sm->size_ > 0) {
      task->is_symlink_ = 1;
      task->size_ = sm->size_;
    }
  } else {
    task->exists_ = 0; task->is_dir_ = 0; task->size_ = 0;
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Truncate(clio::run::shared_ptr<TruncateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = task->path_.str();
  clio::run::u64 new_size = task->new_size_;

  // Resolve the tag + old logical size and set the new logical size. Capture
  // everything under the lock, then release it BEFORE any co_await (never hold
  // a std::mutex across a coroutine suspension).
  clio::cte::core::TagId tag_id;
  clio::run::u64 old_size = 0;
  bool tracked = false;
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    if (it != by_path_.end()) {
      tag_id = it->second->tag_id_;
      old_size = it->second->size_.load();
      it->second->size_.store(new_size);
      tracked = !tag_id.IsNull();
      // A truncate re-establishes natural mtime/ctime — drop utimens overrides.
      it->second->set_atime_ = 0;
      it->second->set_mtime_ = 0;
      it->second->set_ctime_ = 0;
    }
  }

  // Untracked path (truncate without a prior open): resolve the tag + its
  // current size, then record a tracking entry.
  if (!tracked) {
    auto t = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                      clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(t);
    if (t->GetReturnCode() == 0) {
      tag_id = t->tag_id_;
      auto s = cte_.AsyncGetTagSize(tag_id, clio::run::PoolQuery::Dynamic());
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
    clio::run::u64 boundary_page = new_size / kFsPageSize;
    clio::run::u64 boundary_off = new_size % kFsPageSize;
    // Trim the boundary page to its surviving prefix (frees the tail). This also
    // bumps the tag's mtime/ctime (truncate is a modification).
    auto tb = cte_.AsyncTruncateBlob(tag_id, std::to_string(boundary_page),
                                     boundary_off, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(tb);
    // Delete whole pages beyond the boundary, up to the old last page.
    clio::run::u64 last_page = (old_size == 0) ? 0 : (old_size - 1) / kFsPageSize;
    for (clio::run::u64 p = boundary_page + 1; p <= last_page; ++p) {
      auto d = cte_.AsyncDelBlob(tag_id, std::to_string(p), clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(d);
    }
  } else if (!tag_id.IsNull()) {
    // Grow (or same-size) truncate does no blob work, but POSIX still updates
    // mtime and ctime. Reserve no storage — stamp the tag's timestamps by
    // truncating the first page strictly beyond the new EOF, which never holds
    // data (writes only create pages up to EOF, and shrink deletes past it), so
    // TruncateBlob finds it missing and only bumps mtime/ctime.
    clio::run::u64 touch_page = new_size / kFsPageSize + 1;
    auto tb = cte_.AsyncTruncateBlob(tag_id, std::to_string(touch_page),
                                     0, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(tb);
  }

  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Unlink(clio::run::shared_ptr<UnlinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());

  // Refuse to unlink a directory (a tag with children) — that is rmdir's job.
  {
    std::string child_re = "^" + EscapeExact(path) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(child_re, 1, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      task->return_code_ = EISDIR;
      CLIO_CO_RETURN;
    }
  }

  // DelTag is hierarchy-aware: a hard-link (alias) path unlinks only that name;
  // for the canonical name it promotes a surviving alias so the file lives until
  // its last link is removed. posix_unlink=true selects POSIX unlink semantics
  // (#680) instead of the core's cascade-delete-all-aliases behavior.
  auto d = cte_.AsyncDelTag(path, clio::run::PoolQuery::Dynamic(),
                            /*posix_unlink=*/true);
  CLIO_CO_AWAIT(d);
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    by_path_.erase(path);
  }
  CLIO_FS_TOUCH_DIR(ParentDir(path));  // child removed => parent mtime/ctime
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Mkdir(clio::run::shared_ptr<MkdirTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());

  // EEXIST if the path already exists as a directory (has a child) or a file
  // (an exact tag).
  {
    std::string child_re = "^" + EscapeExact(path) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(child_re, 1, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      task->return_code_ = EEXIST;
      CLIO_CO_RETURN;
    }
  }
  {
    auto q = cte_.AsyncTagQuery(ExactRe(path), 1, clio::run::PoolQuery::Dynamic());
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
                                    clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(t);
  if (t->GetReturnCode() == 0) {
    CLIO_FS_TOUCH_DIR(ParentDir(path));  // new subdir => parent mtime/ctime
    task->return_code_ = 0;
  } else {
    task->return_code_ = EIO;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Rmdir(clio::run::shared_ptr<RmdirTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());

  // List direct children. A directory must have at least one child (the marker,
  // for an empty dir); any NON-marker child means the directory is not empty.
  std::string child_re = "^" + EscapeExact(path) + "/[^/]+$";
  auto q = cte_.AsyncTagQuery(child_re, 0, clio::run::PoolQuery::Dynamic());
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
  auto d = cte_.AsyncDelTag(path, clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(d);
  CLIO_FS_TOUCH_DIR(ParentDir(path));  // subdir removed => parent mtime/ctime
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Rename(clio::run::shared_ptr<RenameTask> &task) {
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
    auto q = cte_.AsyncTagQuery(ExactRe(src), 1, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      auto t = cte_.AsyncGetOrCreateTag(src, clio::cte::core::TagId::GetNull(),
                                        clio::run::PoolQuery::Dynamic());
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

  // Classify source as a directory iff it has at least one child entry (a dir
  // always has at least its hidden marker child; a regular file has none).
  bool src_is_dir = false;
  {
    std::string re = "^" + EscapeExact(src) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(re, 1, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(q);
    src_is_dir = (q->GetReturnCode() == 0 && !q->results_.empty());
  }

  // Classify the destination and enforce POSIX rename-overwrite rules before
  // touching anything (issue #597 B1, generic/245). Replacing a destination is
  // only legal file-over-file or dir-over-empty-dir; everything else is an error
  // and must leave both operands untouched.
  //   dst_state: 0 = absent, 1 = file, 2 = empty dir, 3 = non-empty dir.
  int dst_state = 0;
  {
    std::string re = "^" + EscapeExact(dst) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(re, 0, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      const size_t prefix = dst.size() + 1;  // strip "<dst>/"
      dst_state = 2;                          // has a child => directory
      for (const auto &full : q->results_) {
        std::string base =
            full.size() > prefix ? full.substr(prefix) : std::string();
        if (base != kDirMarker) { dst_state = 3; break; }  // real child
      }
    } else {
      // No children: a regular file if an exact tag exists, else nonexistent.
      auto qf =
          cte_.AsyncTagQuery(ExactRe(dst), 1, clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(qf);
      dst_state =
          (qf->GetReturnCode() == 0 && !qf->results_.empty()) ? 1 : 0;
    }
  }
  if (dst_state != 0) {
    if (dst_state == 2 || dst_state == 3) {  // destination is a directory
      if (!src_is_dir) {
        task->return_code_ = EISDIR;  // non-dir onto a directory
        CLIO_CO_RETURN;
      }
      if (dst_state == 3) {
        task->return_code_ = ENOTEMPTY;  // directory onto a non-empty directory
        CLIO_CO_RETURN;
      }
      // dir onto an empty dir: allowed (the empty dst is replaced below).
    } else {  // destination is a regular file
      if (src_is_dir) {
        task->return_code_ = ENOTDIR;  // directory onto a non-dir
        CLIO_CO_RETURN;
      }
      // file onto file: allowed (overwrite below).
    }
  }

  // POSIX rename overwrites an existing destination: unlink dst's name (POSIX
  // semantics, #680 -- a surviving hard link to dst must live on).
  {
    auto d = cte_.AsyncDelTag(dst, clio::run::PoolQuery::Dynamic(),
                              /*posix_unlink=*/true);
    CLIO_CO_AWAIT(d);
  }
  // Rename the tag in place (keeps TagId + blobs).
  auto r = cte_.AsyncRenameTag(src, dst, src_tag, clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(r);
  if (r->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }

  // Move the per-file logical-size metadata entry to the new path.
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    // The destination's old tag was just deleted and its name rebound to the
    // source's tag, so any cached FileInfo for `dst` (e.g. left over from an
    // earlier write to the overwritten file) now points at a dead tag. Drop it
    // first, unconditionally, so getattr re-resolves `dst` from scratch. Without
    // this, renaming a symlink over a previously-written regular file kept
    // reporting the stale regular-file type/size via the tracked getattr branch
    // (generic/023 symb/regu case).
    by_path_.erase(dst);
    auto it = by_path_.find(src);
    if (it != by_path_.end()) {
      it->second->path_ = dst;
      by_path_[dst] = it->second;
      by_path_.erase(it);
    }
  }
  // A rename changes both the source and destination directories (generic/309).
  CLIO_FS_TOUCH_DIR(ParentDir(src));
  std::string dst_parent = ParentDir(dst);
  if (dst_parent != ParentDir(src)) {
    CLIO_FS_TOUCH_DIR(dst_parent);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Link(clio::run::shared_ptr<LinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string target = StripTrailingSlash(task->target_.str());
  std::string link = StripTrailingSlash(task->link_.str());

  // A hard link must not land on an existing name.
  {
    auto q = cte_.AsyncTagQuery(ExactRe(link), 1, clio::run::PoolQuery::Dynamic());
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
  auto a = cte_.AsyncGetOrCreateTagAlias(target, link, clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(a);
  if (a->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }
  if (a->found_ == 1) {
    CLIO_FS_TOUCH_DIR(ParentDir(link));  // new link => parent dir mtime/ctime
    task->return_code_ = 0;
  } else {
    task->return_code_ = ENOENT;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Symlink(clio::run::shared_ptr<SymlinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string target = task->target_.str();
  std::string path = StripTrailingSlash(task->path_.str());

  // A symlink must not land on an existing name.
  {
    auto q = cte_.AsyncTagQuery(ExactRe(path), 1, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      task->return_code_ = EEXIST;
      CLIO_CO_RETURN;
    }
  }

  // Create the symlink's tag at `path` (exactly like a regular file's tag).
  clio::cte::core::TagId tag_id = clio::cte::core::TagId::GetNull();
  {
    auto t = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                      clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(t);
    if (t->GetReturnCode() != 0) {
      task->return_code_ = EIO;
      CLIO_CO_RETURN;
    }
    tag_id = t->tag_id_;
  }

  // Store the target string in the reserved marker blob under the tag.
  auto *ipc = CLIO_IPC;
  clio::run::u64 len = target.size();
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(len);
  if (buf.IsNull()) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }
  std::memcpy(buf.ptr_, target.data(), len);
  auto p = cte_.AsyncPutBlob(tag_id, SymlinkMarker(), 0, len,
                             buf.shm_.template Cast<void>(), -1.0f,
                             clio::cte::core::Context(), 0u,
                             clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(p);
  bool ok = (p->GetReturnCode() == 0);
  ipc->FreeBuffer(buf);
  if (!ok) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }

  CLIO_FS_TOUCH_DIR(ParentDir(path));  // new symlink => parent dir mtime/ctime
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Readlink(clio::run::shared_ptr<ReadlinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());

  // Resolve the tag at `path`.
  auto q = cte_.AsyncTagQuery(ExactRe(path), 1, clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(q);
  if (q->GetReturnCode() != 0 || q->results_.empty()) {
    task->return_code_ = ENOENT;
    CLIO_CO_RETURN;
  }
  clio::run::u64 packed = q->result_ids_.empty() ? 0 : q->result_ids_[0];
  clio::cte::core::TagId tid(static_cast<clio::run::u32>(packed >> 32),
                             static_cast<clio::run::u32>(packed & 0xffffffffULL));

  // The marker blob holds the target string; its size is the target length.
  clio::run::u64 len = 0;
  {
    auto s = cte_.AsyncGetBlobSize(tid, SymlinkMarker(),
                                   clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(s);
    if (s->GetReturnCode() == 0) {
      len = s->size_;
    }
  }
  if (len == 0) {
    task->return_code_ = EINVAL;  // not a symlink (no marker)
    CLIO_CO_RETURN;
  }

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(len);
  if (buf.IsNull()) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }
  auto g = cte_.AsyncGetBlob(tid, SymlinkMarker(), 0, len, 0u,
                             buf.shm_.template Cast<void>(),
                             clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(g);
  bool ok = (g->GetReturnCode() == 0);
  if (ok) {
    task->target_ = clio::run::priv::string(
        CTP_MALLOC, std::string(buf.ptr_, len));
  }
  ipc->FreeBuffer(buf);
  task->return_code_ = ok ? 0 : EIO;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ---- extended attributes (xattr) ----
// All handlers use a read-modify-write over the file's single xattr blob, which
// lives under xattr_tag_id_ (NOT the file's own tag) keyed by the file's packed
// tag id, so xattrs never inflate the file's reported size.

// Resolve `pathvar` to a file tag id `tidvar`; ENOENT-return if it is absent.
#define CLIO_XATTR_RESOLVE_TAG(pathvar, tidvar)                              \
  clio::cte::core::TagId tidvar = clio::cte::core::TagId::GetNull();         \
  {                                                                          \
    auto _q = cte_.AsyncTagQuery(ExactRe(pathvar), 1,                        \
                                 clio::run::PoolQuery::Dynamic());           \
    CLIO_CO_AWAIT(_q);                                                       \
    if (_q->GetReturnCode() != 0 || _q->results_.empty()) {                 \
      task->return_code_ = ENOENT;                                          \
      CLIO_CO_RETURN;                                                       \
    }                                                                       \
    clio::run::u64 _packed =                                                \
        _q->result_ids_.empty() ? 0 : _q->result_ids_[0];                   \
    tidvar = clio::cte::core::TagId(                                        \
        static_cast<clio::run::u32>(_packed >> 32),                         \
        static_cast<clio::run::u32>(_packed & 0xffffffffULL));              \
  }

// Load the file's xattr map into `xavar` (empty if no blob / read miss).
#define CLIO_XATTR_LOAD(tidvar, xavar)                                       \
  std::vector<std::pair<std::string, std::string>> xavar;                    \
  {                                                                          \
    std::string _key = XattrKey(tidvar);                                     \
    clio::run::u64 _len = 0;                                                 \
    auto _s = cte_.AsyncGetBlobSize(xattr_tag_id_, _key,                     \
                                    clio::run::PoolQuery::Dynamic());        \
    CLIO_CO_AWAIT(_s);                                                       \
    if (_s->GetReturnCode() == 0) { _len = _s->size_; }                      \
    if (_len > 0) {                                                          \
      auto *_ipc = CLIO_IPC;                                                 \
      ctp::ipc::FullPtr<char> _buf = _ipc->AllocateBuffer(_len);             \
      if (!_buf.IsNull()) {                                                  \
        auto _g = cte_.AsyncGetBlob(xattr_tag_id_, _key, 0, _len, 0u,        \
                                    _buf.shm_.template Cast<void>(),         \
                                    clio::run::PoolQuery::Dynamic());        \
        CLIO_CO_AWAIT(_g);                                                   \
        if (_g->GetReturnCode() == 0) {                                      \
          xavar = DeserializeXattrs(_buf.ptr_, _len);                        \
        }                                                                    \
        _ipc->FreeBuffer(_buf);                                             \
      }                                                                      \
    }                                                                        \
  }

// Reserialize `xavar` and write it back over the file's xattr blob (replace).
#define CLIO_XATTR_STORE(tidvar, xavar)                                      \
  {                                                                          \
    std::string _payload = SerializeXattrs(xavar);                           \
    std::string _key = XattrKey(tidvar);                                     \
    auto *_ipc = CLIO_IPC;                                                   \
    clio::run::u64 _len = _payload.size();                                   \
    ctp::ipc::FullPtr<char> _buf = _ipc->AllocateBuffer(_len ? _len : 1);    \
    if (_buf.IsNull()) { task->return_code_ = EIO; CLIO_CO_RETURN; }         \
    if (_len) { std::memcpy(_buf.ptr_, _payload.data(), _len); }             \
    auto _p = cte_.AsyncPutBlob(xattr_tag_id_, _key, 0, _len,                \
                                _buf.shm_.template Cast<void>(), -1.0f,       \
                                clio::cte::core::Context(),                   \
                                clio::cte::core::kCtePutReplace,              \
                                clio::run::PoolQuery::Dynamic());            \
    CLIO_CO_AWAIT(_p);                                                       \
    bool _ok = (_p->GetReturnCode() == 0);                                   \
    _ipc->FreeBuffer(_buf);                                                  \
    if (!_ok) { task->return_code_ = EIO; CLIO_CO_RETURN; }                  \
  }

clio::run::TaskResume Runtime::Setxattr(clio::run::shared_ptr<SetxattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());
  std::string name = task->name_.str();
  std::string value = task->value_.str();
  CLIO_XATTR_RESOLVE_TAG(path, tid);
  CLIO_XATTR_LOAD(tid, xa);

  bool present = false;
  for (auto &kv : xa) {
    if (kv.first == name) {
      present = true;
      // XATTR_CREATE: fail if the attribute already exists.
      if ((task->flags_ & 0x1u) != 0) {  // XATTR_CREATE
        task->return_code_ = EEXIST;
        CLIO_CO_RETURN;
      }
      kv.second = value;  // upsert (replace)
      break;
    }
  }
  if (!present) {
    // XATTR_REPLACE: fail if the attribute does not already exist.
    if ((task->flags_ & 0x2u) != 0) {  // XATTR_REPLACE
      task->return_code_ = ENODATA;
      CLIO_CO_RETURN;
    }
    xa.emplace_back(name, value);
  }

  CLIO_XATTR_STORE(tid, xa);
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Getxattr(clio::run::shared_ptr<GetxattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());
  std::string name = task->name_.str();
  CLIO_XATTR_RESOLVE_TAG(path, tid);
  CLIO_XATTR_LOAD(tid, xa);

  task->found_ = 0;
  for (const auto &kv : xa) {
    if (kv.first == name) {
      task->value_ = clio::run::priv::string(CTP_MALLOC, kv.second);
      task->found_ = 1;
      break;
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Listxattr(clio::run::shared_ptr<ListxattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());
  CLIO_XATTR_RESOLVE_TAG(path, tid);
  CLIO_XATTR_LOAD(tid, xa);

  std::string names;
  for (const auto &kv : xa) {
    names.append(kv.first);
    names.push_back('\0');
  }
  task->names_ = clio::run::priv::string(CTP_MALLOC, names);
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Removexattr(clio::run::shared_ptr<RemovexattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());
  std::string name = task->name_.str();
  CLIO_XATTR_RESOLVE_TAG(path, tid);
  CLIO_XATTR_LOAD(tid, xa);

  bool erased = false;
  for (auto it = xa.begin(); it != xa.end(); ++it) {
    if (it->first == name) {
      xa.erase(it);
      erased = true;
      break;
    }
  }
  if (!erased) {
    task->return_code_ = ENODATA;
    CLIO_CO_RETURN;
  }

  CLIO_XATTR_STORE(tid, xa);
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

#undef CLIO_XATTR_RESOLVE_TAG
#undef CLIO_XATTR_LOAD
#undef CLIO_XATTR_STORE

clio::run::TaskResume Runtime::Utimens(clio::run::shared_ptr<UtimensTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());
  // flags: bit0 = explicit atime, bit1 = explicit mtime, bit2 = atime UTIME_NOW,
  // bit3 = mtime UTIME_NOW. A field with no bit set is UTIME_OMIT (left alone).
  // UTIME_NOW is resolved HERE (server) so the value shares the tag clock.
  const bool a_set = (task->flags_ & 0x1u) != 0;
  const bool m_set = (task->flags_ & 0x2u) != 0;
  const bool a_now = (task->flags_ & 0x4u) != 0;
  const bool m_now = (task->flags_ & 0x8u) != 0;
  const clio::run::u64 now = clio::cte::core::GetCurrentTimeNs();

  // A directory isn't tracked in by_path_ (that map holds files, which getattr
  // reports as regular). For a dir, just bump its tag's ctime/mtime via a touch
  // (no exact-value override); files below get per-path overrides.
  {
    std::string child_re = "^" + EscapeExact(path) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(child_re, 1, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      CLIO_FS_TOUCH_DIR(path);
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }

  // Regular file: resolve its tag (utimens may target a not-yet-opened file),
  // then record the requested atime/mtime as overrides that getattr honors.
  // ctime always advances (metadata changed).
  clio::cte::core::TagId tag_id;
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    if (it != by_path_.end()) tag_id = it->second->tag_id_;
  }
  if (tag_id.IsNull()) {
    auto t = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                      clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(t);
    if (t->GetReturnCode() != 0) { task->return_code_ = EIO; CLIO_CO_RETURN; }
    tag_id = t->tag_id_;
  }
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    std::shared_ptr<FileInfo> fi;
    if (it != by_path_.end()) {
      fi = it->second;
    } else {
      fi = std::make_shared<FileInfo>();
      fi->tag_id_ = tag_id;
      fi->path_ = path;
      by_path_[path] = fi;
    }
    if (a_now) fi->set_atime_ = now;
    else if (a_set) fi->set_atime_ = task->atime_ns_;
    if (m_now) fi->set_mtime_ = now;
    else if (m_set) fi->set_mtime_ = task->mtime_ns_;
    fi->set_ctime_ = now;  // utimens always advances ctime
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Chown(clio::run::shared_ptr<ChownTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::string path = StripTrailingSlash(task->path_.str());

  // A directory isn't tracked in by_path_ (that map holds files). For a dir,
  // just bump its tag's ctime/mtime via a touch (owner-tracking on dirs is not
  // needed for the target tests); files below get per-path owner overrides.
  {
    std::string child_re = "^" + EscapeExact(path) + "/[^/]+$";
    auto q = cte_.AsyncTagQuery(child_re, 1, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(q);
    if (q->GetReturnCode() == 0 && !q->results_.empty()) {
      CLIO_FS_TOUCH_DIR(path);
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }

  // Regular file: resolve its tag (chown may target a not-yet-opened file).
  clio::cte::core::TagId tag_id;
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    if (it != by_path_.end()) tag_id = it->second->tag_id_;
  }
  if (tag_id.IsNull()) {
    auto t = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                      clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(t);
    if (t->GetReturnCode() != 0) { task->return_code_ = EIO; CLIO_CO_RETURN; }
    tag_id = t->tag_id_;
  }
  // Fetch the current on-tag size BEFORE taking meta_mu_ (no RPC under the
  // lock). If we must CREATE a new FileInfo below, we seed its size_ from this
  // so getattr's tracked branch reports the real file size (not 0) for a file
  // that already has data.
  clio::run::u64 cur_size = 0;
  {
    auto s = cte_.AsyncGetTagSize(tag_id, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(s);
    if (s->GetReturnCode() == 0) cur_size = s->tag_size_;
  }
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = by_path_.find(path);
    std::shared_ptr<FileInfo> fi;
    if (it != by_path_.end()) {
      fi = it->second;  // already tracked: do NOT overwrite its live size_.
    } else {
      fi = std::make_shared<FileInfo>();
      fi->tag_id_ = tag_id;
      fi->path_ = path;
      fi->size_.store(cur_size);  // seed size so getattr doesn't report 0
      by_path_[path] = fi;
    }
    // 0xFFFFFFFF means "leave this field unchanged" (POSIX (uid_t)-1).
    if (task->uid_ != 0xFFFFFFFFu) fi->set_uid_ = task->uid_;
    if (task->gid_ != 0xFFFFFFFFu) fi->set_gid_ = task->gid_;
    // chmod rides the same task (uid/gid unchanged): store the permission bits.
    if (task->mode_ != 0xFFFFFFFFu) fi->set_mode_ = task->mode_ & 07777u;
    fi->set_ctime_ = clio::cte::core::GetCurrentTimeNs();  // chown/chmod advances ctime
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Readdir(clio::run::shared_ptr<ReaddirTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Direct children: tags whose resolved name is "<dir>/<name>" with no
  // further slash. Returns full resolved paths; the adapter strips the prefix.
  std::string dir = task->path_.str();
  if (dir.empty() || dir.back() != '/') dir += '/';
  std::string regex = "^" + EscapeExact(dir) + "[^/]+$";
  auto q = cte_.AsyncTagQuery(regex, 0, clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(q);
  task->entries_ = clio::run::priv::vector<clio::run::priv::string>(CTP_MALLOC);
  task->inos_ = clio::run::priv::vector<clio::run::u64>(CTP_MALLOC);
  if (q->GetReturnCode() == 0) {
    const size_t prefix = dir.size();
    // q->result_ids_ is index-aligned with q->results_; carry each child's
    // packed TagId through as its inode so readdir d_ino matches stat st_ino.
    for (size_t i = 0; i < q->results_.size(); ++i) {
      const std::string &name = q->results_[i];
      // Hide the internal empty-directory marker.
      std::string base = name.size() > prefix ? name.substr(prefix) : name;
      if (base == kDirMarker) continue;
      task->entries_.push_back(clio::run::priv::string(CTP_MALLOC, name));
      clio::run::u64 packed = i < q->result_ids_.size() ? q->result_ids_[i] : 0;
      task->inos_.push_back(InoFromPacked(packed));
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::StatSize(clio::run::shared_ptr<StatSizeTask> &task) {
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
  auto qy = cte_.AsyncTagQuery(ExactRe(path), 1, clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(qy);
  if (qy->GetReturnCode() == 0 && !qy->results_.empty()) {
    task->exists_ = 1;
    auto tag = cte_.AsyncGetOrCreateTag(path, clio::cte::core::TagId::GetNull(),
                                        clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(tag);
    auto s = cte_.AsyncGetTagSize(tag->tag_id_, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(s);
    task->size_ = (s->GetReturnCode() == 0) ? s->tag_size_ : 0;
  } else {
    task->exists_ = 0;
    task->size_ = 0;
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Deferred-append pipeline handlers
// ===========================================================================

clio::run::TaskResume Runtime::AppendSequence(
    clio::run::shared_ptr<AppendSequenceTask> &task) {
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
  // separate on the same leader. Submitted and awaited one tag at a time so
  // that at most one subtask future is ever outstanding in this coroutine — see
  // the use-after-free note in AppendExecution: the CPU await path cannot tell
  // which of several in-flight futures a sibling completion belongs to.
  for (auto &kv : by_tag) {
    const clio::cte::core::TagId &tag = kv.first;
    clio::run::u32 chash = static_cast<clio::run::u32>(
        std::hash<clio::cte::core::TagId>()(tag));
    clio::run::u64 bkey = (static_cast<clio::run::u64>(tag.major_) << 32) | tag.minor_;
    auto q = clio::run::PoolQuery::ManyToOne(chash, bkey, /*batch_for_ns=*/50000);
    auto f = self_.AsyncAppendCollect(tag, kv.second, q);
    CLIO_CO_AWAIT(f);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AppendCollect(
    clio::run::shared_ptr<AppendCollectTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Runs ONCE per batch as the ManyToOne aggregate; task->entries_ holds every
  // node's pending entries for this tag (combined via AggregateIn). The actual
  // merge needs to suspend (it reads the file tail and drives PutBlobs), which a
  // ManyToOne aggregate cannot express directly, so it is delegated to a regular
  // AppendPlan task. We AWAIT that task rather than fire-and-forget it, which is
  // essential for two reasons:
  //   1. Lifetime: AppendPlan's parent RunContext is THIS aggregate's. If we
  //      returned immediately, EndTask -> OnAggregateComplete would free the
  //      aggregate (and its RunContext) while AppendPlan was still running, so
  //      AppendPlan's completion would dereference a freed parent (a UAF in
  //      IpcCpu2Self::RuntimeSend). Awaiting keeps the parent alive until the
  //      merge finishes.
  //   2. Serialization: the BatchManager holds this group's in-flight claim
  //      until the aggregate completes. Awaiting the merge keeps the claim held
  //      for the whole merge, so no second batch for the same tag can start
  //      while this one is still writing the tail — concurrent merges would both
  //      plan from the same GetTagSize and clobber each other.
  // Only one subtask future (AppendPlan) is ever outstanding here, so the
  // single-outstanding-future rule documented in AppendExecution holds.
  std::vector<AppendEntry> entries(task->entries_.begin(),
                                   task->entries_.end());
  auto f =
      self_.AsyncAppendPlan(task->tag_id_, entries, clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(f);
  task->new_size_ = 0;  // settled by the merge; members don't read it
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AppendPlan(clio::run::shared_ptr<AppendPlanTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Regular (suspendable) task: sort the batch, read the file tail, build the
  // 1 MiB-page merge plan, and dispatch AppendExecution slices.
  clio::cte::core::TagId tag_id = task->tag_id_;
  std::vector<AppendEntry> entries(task->entries_.begin(),
                                   task->entries_.end());
  // Global order: UTC first, logical counter as tiebreak.
  std::sort(entries.begin(), entries.end(),
            [](const AppendEntry &a, const AppendEntry &b) {
              if (a.utc_ns_ != b.utc_ns_) return a.utc_ns_ < b.utc_ns_;
              return a.logical_ < b.logical_;
            });

  // Tail of the file = its merged content size. Append data blobs live under
  // the separate staging tag, so GetTagSize(file_tag) is exactly the tail.
  // Because at most one AppendCollect per tag runs at a time (BatchManager
  // serialization) and each fully merges + deletes its staged blobs before
  // completing, this read is stable: the previous batch's bytes are already in
  // the file pages and no other batch is mutating it.
  clio::run::u64 cur_size = 0;
  {
    auto s = cte_.AsyncGetTagSize(tag_id, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(s);
    cur_size = (s->GetReturnCode() == 0) ? s->tag_size_ : 0;
  }

  // Build the merge plan: lay each data blob out contiguously starting at
  // cur_size, splitting on 1 MiB file-page boundaries.
  std::vector<AppendPlanStep> plan;
  clio::run::u64 file_off = cur_size;
  for (auto &e : entries) {
    clio::run::u64 remaining = e.data_blob_size_;
    clio::run::u64 doff = 0;
    while (remaining > 0) {
      clio::run::u64 page = file_off / kFsPageSize;
      clio::run::u64 page_off = file_off % kFsPageSize;
      clio::run::u64 step = std::min(kFsPageSize - page_off, remaining);
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
  constexpr clio::run::u64 kMaxExecBytes = 16ull * 1024 * 1024;
  clio::run::u32 spread = 0;
  size_t i = 0;
  while (i < plan.size()) {
    std::vector<AppendPlanStep> slice;
    clio::run::u64 bytes = 0;
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
    // Dispatch and await one slice at a time. Slices of a single batch write
    // disjoint file regions, but they MUST NOT be in flight simultaneously:
    // the CPU await path keeps only one coroutine handle, so an out-of-order
    // slice completion would resume this coroutine onto the wrong awaiter and
    // free a still-queued task's FutureShm (the use-after-free documented in
    // AppendExecution). Concurrency across the system still comes from distinct
    // tags' batches, which run as independent top-level AppendPlan tasks.
    auto q = clio::run::PoolQuery::DirectHash(spread++);
    auto f = self_.AsyncAppendExecution(tag_id, staging_tag_id_, slice, q);
    CLIO_CO_AWAIT(f);
  }

  (void)file_off;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AppendExecution(
    clio::run::shared_ptr<AppendExecutionTask> &task) {
  CLIO_TASK_BODY_BEGIN
  clio::cte::core::TagId tag_id = task->tag_id_;        // destination file tag
  clio::cte::core::TagId staging = task->staging_tag_id_;  // source staged blobs
  auto *ipc = CLIO_IPC;
  const size_t n = task->steps_.size();
  bool ok = true;

  // Apply each step strictly sequentially: AT MOST ONE subtask future may be
  // outstanding at a time. The runtime's CPU await path stores only the
  // coroutine handle (RunContext::coro_handle_) — it does NOT record *which*
  // future the coroutine is suspended on. Any sibling subtask completion
  // resumes the coroutine, and await_resume() then unconditionally Destroy()s
  // whatever future it is currently awaiting. So if several subtasks were in
  // flight at once, an out-of-order completion would run await_resume() on a
  // DIFFERENT, still-running task — freeing that task's FutureShm while it is
  // still queued in a worker lane, a heap-use-after-free (observed as garbage
  // pool ids in Worker::ProcessNewTask). Issuing get -> await -> put -> await
  // one step at a time keeps exactly one future live, so only that future's
  // completion can ever resume us. System-wide concurrency still comes from
  // distinct tags' batches, which run as independent top-level merge tasks.
  for (size_t i = 0; i < n && ok; ++i) {
    const AppendPlanStep &s = task->steps_[i];
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(s.size_);
    if (buf.IsNull()) {
      ok = false;
      break;
    }

    // Read this step's staged data slice (a short/hole read leaves zeros).
    auto g = cte_.AsyncGetBlob(staging, s.data_blob_id_, s.off_in_data_,
                               s.size_, 0u, buf.shm_.template Cast<void>(),
                               clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(g);

    // Write the bytes into the destination file page. PutBlobs are necessarily
    // serialized: multiple steps can target the SAME 1 MiB page blob at
    // different offsets, and concurrent PutBlobs to one blob race in the core's
    // extend/modify path (corrupting the block list).
    auto p = cte_.AsyncPutBlob(
        tag_id, std::to_string(s.file_page_), s.off_in_page_, s.size_,
        buf.shm_.template Cast<void>(), -1.0f, clio::cte::core::Context(), 0u,
        clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() != 0) ok = false;

    ipc->FreeBuffer(buf);
  }

  // DelBlob each distinct staged data blob exactly once, one at a time (same
  // single-outstanding-future rule as the read/write loop above). Because a
  // data blob's steps are confined to one execution task, this is the only
  // task that deletes it — no double-free.
  std::unordered_set<std::string> seen;
  for (const auto &s : task->steps_) {
    if (seen.insert(s.data_blob_id_).second) {
      auto d = cte_.AsyncDelBlob(staging, s.data_blob_id_,
                                 clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(d);
    }
  }

  task->return_code_ = ok ? 0 : EIO;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

#undef CLIO_FS_LOOKUP

}  // namespace clio::cte::filesystem

// Define ChiMod entry points (alloc/new/name/destroy) so the runtime's module
// manager can dlopen and instantiate this chimod.
CLIO_TASK_CC(clio::cte::filesystem::Runtime)
