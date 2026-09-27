/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Filesystem chimod method handlers. Paths name entries of a hash-sharded
 * namespace (fs_shard.h, fs_namespace.cc); file data lives in the CTE core as
 * 1 MiB page-blobs ("0","1",...) under the file's id, driven through a core
 * client (`cte_`, bound to next_pool_id_ at Create).
 *
 * Every handler runs on the container its client routed it to -- the owner
 * of the path's parent directory, of a directory, or of an inode -- and
 * reaches any other piece of state through CallShard on that state's owner.
 */
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <clio_ctp/util/config_parse.h>
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
 * Reserved name for a staged append data blob. The "_append/" prefix can't
 * collide with the numeric page-blob names ("0","1", ...); node id keeps it
 * unique across nodes, the logical counter within a node.
 */
inline std::string MakeDataBlobId(clio::run::u32 node_id, clio::run::u64 logical) {
  return "_append/" + std::to_string(node_id) + "." + std::to_string(logical);
}

/** Stable inode number: the packed id (0 maps to 1: st_ino 0 = no inode). */
inline clio::run::u64 InoFromPacked(clio::run::u64 packed) {
  return packed ? packed : 1;
}

// ---- extended-attribute (xattr) blob (de)serialization ----
// A file's xattrs are stored as ONE blob under the global xattr tag, named by
// the file's packed id (decimal). Payload = repeated records
// [u32 name_len][name][u32 val_len][val] (values may hold NULs).
inline std::string XattrKey(clio::run::u64 packed) {
  return std::to_string(packed);
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

/** SetAttr flag bits shared with fs_namespace.cc (kShardInodeSetAttr). */
enum : clio::run::u32 {
  kSetAtime = 1u, kSetMtime = 2u, kSetAtimeNow = 4u, kSetMtimeNow = 8u,
  kSetUid = 16u, kSetGid = 32u, kSetMode = 64u, kAttrRepair = 128u,
  kSetCtimeOnly = 256u,
};

/** Random backoff (us) for a two-party operation that lost a race. */
inline double BackoffUs(int attempt) {
  thread_local std::mt19937 rng(std::random_device{}());
  const int cap = std::min(2000, 50 << std::min(attempt, 5));
  return static_cast<double>(std::uniform_int_distribution<int>(20, cap)(rng));
}
}  // namespace

// ===========================================================================
// Lifecycle
// ===========================================================================

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  FilesystemConfig cfg = task->GetParams();
  next_pool_id_ = cfg.next_pool_id_;
  if (!next_pool_id_.IsNull()) {
    cte_ = clio::cte::core::Client(next_pool_id_);
  }
  // Bind a client to our own pool for self-submitted tasks. Use the
  // assigned pool id from the CreateTask (pool_id_ isn't reliable yet here).
  self_.Init(task->new_pool_id_);

  // Global append-staging tag (shared by all files): append data blobs live
  // here so they don't inflate any file's GetTagSize.
  {
    auto st = cte_.AsyncGetOrCreateTag("_clio_append_staging",
                                       clio::cte::core::TagId::GetNull(),
                                       clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(st);
    if (st->GetReturnCode() == 0) staging_tag_id_ = st->tag_id_;
  }
  // Global xattr-store tag: each file's xattrs live in ONE blob here, named
  // by the file's packed id.
  {
    auto xt = cte_.AsyncGetOrCreateTag("_clio_xattr_store",
                                       clio::cte::core::TagId::GetNull(),
                                       clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(xt);
    if (xt->GetReturnCode() == 0) xattr_tag_id_ = xt->tag_id_;
  }

  // This container's slice of the namespace: replay it from its own log.
  {
    std::string lp;
    if (!cfg.metadata_log_path_.empty()) {
      lp = ctp::ConfigParse::ExpandPath(cfg.metadata_log_path_) + "." +
           std::to_string(container_id_);
    }
    RecoverShard(lp, is_restart_);
  }

  // issue #817: shared-memory attribute mirror. Node-local derived state that
  // clients treat as authoritative, so it is only safe when this node owns
  // the whole namespace (a single node). CLIO_CFS_SHM_MIRROR_MULTINODE=1
  // forces it on for deployments whose files are never shared across nodes.
  {
    size_t capacity = 256 * 1024;
    if (const char *env = clio::run::env::GetCompat("CFS_SHM_FILE_CAPACITY")) {
      char *end = nullptr;
      unsigned long long v = std::strtoull(env, &end, 10);
      if (end != env && v > 0) capacity = static_cast<size_t>(v);
    }
    auto *ipc = CLIO_IPC;
    const bool multi_node = ipc != nullptr && ipc->GetNumHosts() > 1;
    const char *force = clio::run::env::GetCompat("CFS_SHM_MIRROR_MULTINODE");
    const bool forced = force != nullptr && *force == '1';
    if (multi_node && !forced) {
      HLOG(kInfo, "filesystem: shared-memory attribute cache disabled on a "
           "{}-node deployment", ipc->GetNumHosts());
    } else if (shm_fs_cache_.Create(capacity, task->new_pool_id_)) {
      HLOG(kInfo, "filesystem: shared-memory attribute cache enabled "
           "(files={})", capacity);
    }
  }
  HLOG(kInfo, "filesystem: container {} over CTE pool {} (namespace hash "
       "sharded across {} containers)", container_id_,
       next_pool_id_.ToString(), NumContainers());
  // Never claim the root complete: after a restart it has children the
  // mirror has never seen.
  MirrorDir("/", FsRootId(), /*complete=*/false);
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
  // "next_pool": the pool this filesystem stores page blobs through (the top
  // of the interposition chain). Data-path clients (the FUSE adapter writes
  // pages directly) must target it too, or their bytes bypass replication.
  if (task->query_ == "next_pool") {
    task->results_[container_id_] =
        std::to_string(next_pool_id_.major_) + "." +
        std::to_string(next_pool_id_.minor_);
  }
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::EnsurePurgeDrain() {
  {
    std::lock_guard<std::mutex> g(purge_mu_);
    if (purge_started_) return;
    purge_started_ = true;
  }
  // Periodic (20 ms): drains the data of destroyed inodes off the unlink
  // path. A periodic task never completes, so the future is dropped.
  self_.AsyncShardOp(kShardPurgeDrain, std::string(),
                     clio::run::PoolQuery::Local(), 20000.0);
}

// ===========================================================================
// issue #817: shared-memory attribute mirror (single node only)
// ===========================================================================

void Runtime::MirrorFile(const std::string &path, const FileInfo &fi,
                         clio::run::u32 extra_flags) {
  if (!shm_fs_cache_.IsEnabled()) return;
  // A file with several names is refreshed only under its primary name;
  // refuse the fast path for the others so they never serve a stale size.
  if (path != fi.path_ || fi.nlink_ > 1) extra_flags |= kShmFileNoFastPath;
  ShmFileRecord rec;
  rec.tag_id_ = fi.tag_id_;
  rec.size_ = fi.size_.load();
  rec.ino_ = InoFromPacked(FsPack(fi.tag_id_));
  rec.ov_atime_ns_ = fi.atime_;
  rec.ov_mtime_ns_ = fi.mtime_;
  rec.ov_ctime_ns_ = fi.ctime_;
  rec.mode_ = fi.mode_;
  rec.uid_ = fi.uid_;
  rec.gid_ = fi.gid_;
  rec.flags_ = kShmFileExists | extra_flags;
  if (fi.type_ == kFsTypeSymlink) rec.flags_ |= kShmFileNoFastPath;
  shm_fs_cache_.PutFile(path, rec);
}

void Runtime::MirrorDir(const std::string &path,
                        const clio::cte::core::TagId &tag_id, bool complete) {
  if (!shm_fs_cache_.IsEnabled()) return;
  ShmFileRecord rec;
  rec.tag_id_ = tag_id;
  rec.size_ = 0;
  rec.ino_ = InoFromPacked(FsPack(tag_id));
  rec.ov_atime_ns_ = 0;
  rec.ov_mtime_ns_ = 0;
  rec.ov_ctime_ns_ = 0;
  rec.mode_ = kShmFileNoOverride;
  rec.uid_ = kShmFileNoOverride;
  rec.gid_ = kShmFileNoOverride;
  rec.flags_ = kShmFileExists | kShmFileIsDir | (complete ? kShmDirComplete : 0u);
  shm_fs_cache_.PutFile(path, rec);
}

void Runtime::MirrorRefuse(const std::string &path) {
  if (!shm_fs_cache_.IsEnabled()) return;
  ShmFileRecord rec;
  if (!shm_fs_cache_.TryGetFile(path, &rec)) return;
  rec.flags_ |= kShmFileNoFastPath;
  shm_fs_cache_.PutFile(path, rec);
}

// ===========================================================================
// Shared building blocks
// ===========================================================================

int Runtime::ResolveLocal(const std::string &path, Dentry *out) {
  std::lock_guard<std::mutex> g(ns_mu_);
  return LookupLocked(FsParentDir(path), FsLeaf(path), out);
}

clio::run::TaskResume Runtime::StatEntry(const std::string &path,
                                         const Dentry &e, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  FsReq r;
  r.id_ = FsPack(e.id_);
  if (e.type_ == kFsTypeDir) {
    r.dir_ = path;
    // The parent's LIVE entry is authoritative: a missing listing behind it
    // (crash between rmdir's halves) is recreated rather than reported gone.
    if (e.state_ == kEntLive) r.flags_ = kAttrRepair;
    CLIO_CO_AWAIT(CallShard(DirOwner(path), kShardDirAttr, r, resp));
  } else {
    CLIO_CO_AWAIT(CallShard(InodeOwner(r.id_), kShardInodeStat, r, resp));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::UnlinkInode(clio::run::u64 packed,
                                           const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  FsReq r;
  r.id_ = packed;
  r.str_ = name;
  r.a_ = static_cast<clio::run::u64>(static_cast<clio::run::i64>(-1));
  FsResp resp;
  CLIO_CO_AWAIT(CallShard(InodeOwner(packed), kShardInodeNlink, r, resp));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Resolve `path` on the owner of its parent directory into `ent` / `erc`
 * (0, ENOENT, ENOTDIR, or EIO when that owner is unreachable). The root
 * resolves to itself.
 */
#define CLIO_FS_RESOLVE(pathv, ent, erc)                                      \
  Dentry ent;                                                                 \
  int erc = 0;                                                                \
  do {                                                                        \
    if ((pathv) == "/") {                                                     \
      ent.id_ = FsRootId();                                                   \
      ent.type_ = kFsTypeDir;                                                 \
      break;                                                                  \
    }                                                                         \
    FsReq _lr;                                                                \
    _lr.dir_ = FsParentDir(pathv);                                            \
    _lr.leaf_ = FsLeaf(pathv);                                                \
    FsResp _lp;                                                               \
    CLIO_CO_AWAIT(CallShard(DirOwner(_lr.dir_), kShardLookup, _lr, _lp));     \
    erc = static_cast<int>(_lp.rc_);                                          \
    ent.id_ = FsUnpack(_lp.id_);                                              \
    ent.type_ = _lp.type_;                                                    \
  } while (0)

// ===========================================================================
// Open / close / size
// ===========================================================================

clio::run::TaskResume Runtime::Open(clio::run::shared_ptr<OpenTask> &task) {
  CLIO_TASK_BODY_BEGIN
  EnsurePurgeDrain();
  const std::string path = FsNormPath(task->path_.str());
  task->handle_ = 0;
  task->size_ = 0;
  task->created_ = 0;
  if (path == "/") {
    task->return_code_ = EISDIR;
    CLIO_CO_RETURN;
  }
  FsReq r;
  r.dir_ = FsParentDir(path);
  r.leaf_ = FsLeaf(path);
  FsResp er;
  if (task->flags_ & O_CREAT) {
    // Create-or-open in ONE shard op on the directory's owner: of several
    // racing creators (any node) exactly one sees created_=1, which is what
    // O_EXCL keys off.
    r.type_ = kFsTypeFile;
    r.mode_ = task->mode_ & 07777u;
    r.flags_ = kInsNewInode | ((task->flags_ & O_EXCL) ? kInsExcl : 0u);
    CLIO_CO_AWAIT(CallShard(DirOwner(r.dir_), kShardInsert, r, er));
  } else {
    CLIO_CO_AWAIT(CallShard(DirOwner(r.dir_), kShardLookup, r, er));
    if (er.rc_ == ENOENT) {  // handle_ = 0 means ENOENT to the client
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
  }
  if (er.rc_ != 0) {
    task->return_code_ = er.rc_;
    CLIO_CO_RETURN;
  }
  if (er.type_ == kFsTypeDir) {
    task->return_code_ = EISDIR;
    CLIO_CO_RETURN;
  }
  // Second stage on the inode's home (the same container unless the file
  // was renamed or linked here from a directory another node owns).
  FsReq o;
  o.id_ = er.id_;
  o.dir_ = path;  // the name it is opened by (mirror key)
  o.flags_ = er.created_ ? 0u : (task->flags_ & O_TRUNC);
  FsResp orr;
  CLIO_CO_AWAIT(CallShard(InodeOwner(o.id_), kShardInodeOpen, o, orr));
  if (orr.rc_ == ENOENT) {  // entry without an inode: a crash leftover
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  task->handle_ = orr.handle_;
  task->size_ = orr.attr_.size_;
  task->created_ = er.created_;
  task->tag_packed_ = er.id_;
  task->return_code_ = orr.rc_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AdvanceSize(
    clio::run::shared_ptr<AdvanceSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::shared_ptr<FileInfo> fi = FindInode(task->tag_packed_);
  if (fi == nullptr) {
    task->return_code_ = ENOENT;
    CLIO_CO_RETURN;
  }
  // A size push means the file was WRITTEN (the adapter only sends one for a
  // handle that wrote): it is the file's new mtime.
  if (task->reserve_ != 0) {
    task->old_size_ = fi->size_.fetch_add(task->size_);
  } else {
    clio::run::u64 want = task->size_;
    clio::run::u64 old = fi->size_.load();
    while (want > old && !fi->size_.compare_exchange_weak(old, want)) {
    }
    task->old_size_ = old;
  }
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    const clio::run::u64 now = clio::cte::core::GetWallTimeNs();
    fi->mtime_ = fi->ctime_ = now;
    LogInode(*fi);  // the logical size is durable once close returns
    if (!fi->path_.empty()) MirrorFile(fi->path_, *fi);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiCreate(
    clio::run::shared_ptr<MultiCreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Batched file creation (sieve create): each entry adopts its
  // client-minted id on the owner of its directory. Per-entry failures don't
  // stop the batch; the first is reported.
  task->num_ok_ = 0;
  task->first_rc_ = 0;
  std::string packed = task->packed_.str();
  std::vector<MultiCreateEnt> ents;
  if (!DecodeMultiCreate(packed.data(), packed.size(), &ents)) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  for (const auto &e : ents) {
    const std::string path = FsNormPath(e.path_);
    FsReq r;
    r.dir_ = FsParentDir(path);
    r.leaf_ = FsLeaf(path);
    r.id_ = e.tag_packed_;
    r.type_ = kFsTypeFile;
    r.mode_ = e.mode_ & 07777u;
    r.flags_ = kInsNewInode | kInsExcl;
    FsResp er;
    CLIO_CO_AWAIT(CallShard(DirOwner(r.dir_), kShardInsert, r, er));
    if (er.rc_ != 0) {
      if (task->first_rc_ == 0) task->first_rc_ = er.rc_;
      continue;
    }
    task->num_ok_++;
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Close(clio::run::shared_ptr<CloseTask> &task) {
  CLIO_TASK_BODY_BEGIN
  EnsurePurgeDrain();
  std::lock_guard<std::mutex> g(meta_mu_);
  auto it = handles_.find(task->handle_);
  if (it != handles_.end()) {
    std::shared_ptr<FileInfo> fi = it->second;
    handles_.erase(it);
    if (fi->open_count_ > 0) fi->open_count_--;
    // Sieve-written files advance their logical size at close: the handle's
    // FileInfo tracks the file across renames.
    if (task->advance_size_ != 0) {
      clio::run::u64 want = task->advance_size_;
      clio::run::u64 old = fi->size_.load();
      while (want > old && !fi->size_.compare_exchange_weak(old, want)) {
      }
      fi->dirty_ = true;
    }
    if (fi->orphan_ && fi->open_count_ == 0) {
      DropInodeLocked(fi);
    } else if (fi->dirty_) {
      fi->dirty_ = false;
      LogInode(*fi);
      if (!fi->path_.empty()) MirrorFile(fi->path_, *fi);
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// Look up the FileInfo for a handle (nullptr if unknown). Brief lock only.
// The critical section is one hash lookup; sharding or a shared_mutex here
// measured neutral-to-worse (see git history of this file).
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
  // Read directly into the task's payload via the PRIVATE-memory GetBlob path
  // (issue #823): no staging buffer, no copy.
  ctp::ipc::ShmPtr<char> data_base = task->data_.template Cast<char>();
  char *dst = ipc->ToFullPtr<char>(data_base).ptr_;

  clio::run::u64 offset = task->offset_;
  clio::run::u64 want = task->size_;
  if (offset >= file_size) {
    task->bytes_read_ = 0;
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  if (offset + want > file_size) want = file_size - offset;  // logical EOF
  // Holes (never-written bytes within the logical size) read as zeros.
  std::memset(dst, 0, want);

  clio::run::u64 done = 0;
  clio::run::u64 cur = offset;
  while (done < want) {
    clio::run::u64 page_off = cur % kFsPageSize;
    clio::run::u64 to_read = std::min(kFsPageSize - page_off, want - done);
    auto g = cte_.AsyncGetBlob(tag_id, PageName(cur), page_off, to_read,
                               /*flags*/ 0u, dst + done,
                               clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(g);
    // A miss (rc 1) is a hole; any other code (a page on an unreachable node)
    // must fail the read, not read as zeros.
    const int grc = static_cast<int>(g->GetReturnCode());
    if (grc != 0 && grc != 1) {
      task->bytes_read_ = 0;
      task->return_code_ = EIO;
      CLIO_CO_RETURN;
    }
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
  auto *ipc = CLIO_IPC;
  ctp::ipc::ShmPtr<char> data_base = task->data_.template Cast<char>();
  const char *src = ipc->ToFullPtr<char>(data_base).ptr_;

  clio::run::u64 want = task->size_;
  clio::run::u64 done = 0;
  clio::run::u64 cur = task->offset_;
  bool ok = true;
  while (done < want) {
    clio::run::u64 page_off = cur % kFsPageSize;
    clio::run::u64 to_write = std::min(kFsPageSize - page_off, want - done);
    // DOUBLING preallocation (2x the written extent, floor 8 KiB, cap 64 KiB)
    // keeps runs of small appends in place without reserving a flat 64 KiB
    // per page (which made a checkout's RAM tier ~3x its data).
    static constexpr clio::run::u64 kFsPreallocCap = 64ull * 1024;
    static const clio::run::u64 flat_prealloc = [] {
      const char *e = std::getenv("CLIO_CFS_PREALLOC");
      return e != nullptr ? std::strtoull(e, nullptr, 10) : 0ULL;
    }();
    const clio::run::u64 prealloc =
        flat_prealloc != 0
            ? flat_prealloc
            : std::min<clio::run::u64>(
                  kFsPreallocCap,
                  std::max<clio::run::u64>(2 * (page_off + to_write), 8192));
    auto p = cte_.AsyncPutBlob(tag_id, PageName(cur), page_off, to_write,
                               src + done, /*score*/ -1.0f,
                               clio::cte::core::Context::Preallocate(prealloc),
                               /*flags*/ 0u, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() != 0) { ok = false; break; }
    done += to_write;
    cur += to_write;
  }
  clio::run::u64 end = task->offset_ + done;
  clio::run::u64 old = fi->size_.load();
  while (end > old && !fi->size_.compare_exchange_weak(old, end)) {
  }
  {
    // Published AFTER every PutBlob completed, so the mirror can lag the
    // file but never lead it. The record is logged at close.
    std::lock_guard<std::mutex> g(meta_mu_);
    const clio::run::u64 now = clio::cte::core::GetWallTimeNs();
    fi->mtime_ = fi->ctime_ = now;
    fi->dirty_ = true;
    if (!fi->path_.empty()) MirrorFile(fi->path_, *fi);
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
  // Deferred append: stamp a global order (UTC + per-node counter) and stage
  // the bytes; the AppendSequence -> AppendCollect -> AppendExecution
  // pipeline merges them into the file tail later.
  clio::run::u64 logical = append_logical_.fetch_add(1) + 1;
  clio::run::u64 utc_ns = NowUtcNs();
  clio::run::u32 node_id = CLIO_IPC->GetNodeId();
  std::string data_blob_id = MakeDataBlobId(node_id, logical);
  auto p = cte_.AsyncPutBlob(staging_tag_id_, data_blob_id, 0, want,
                             task->data_, -1.0f, clio::cte::core::Context(), 0u,
                             clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(p);
  if (p->GetReturnCode() != 0) {
    task->return_code_ = EIO;
    CLIO_CO_RETURN;
  }
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
    // Periodic local drain (1 ms); a periodic task never completes.
    self_.AsyncAppendSequence(/*period_us=*/1000.0, clio::run::PoolQuery::Local());
  }
  clio::run::u64 newsz = fi->size_.fetch_add(want) + want;
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    const clio::run::u64 now = clio::cte::core::GetWallTimeNs();
    fi->mtime_ = fi->ctime_ = now;
    fi->dirty_ = true;
    if (!fi->path_.empty()) MirrorFile(fi->path_, *fi, kShmFilePendingAppend);
  }
  task->offset_ = newsz - want;
  task->bytes_written_ = want;
  task->new_size_ = newsz;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Stat / truncate
// ===========================================================================

clio::run::TaskResume Runtime::Getattr(clio::run::shared_ptr<GetattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  task->exists_ = 0;
  task->is_dir_ = 0;
  task->size_ = 0;
  CLIO_FS_RESOLVE(path, ent, erc);
  if (erc == ENOENT || erc == ENOTDIR) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  if (erc != 0) {
    task->return_code_ = erc;
    CLIO_CO_RETURN;
  }
  FsResp sr;
  CLIO_CO_AWAIT(StatEntry(path, ent, sr));
  if (sr.rc_ == ENOENT) {  // entry without its state: a crash leftover
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  if (sr.rc_ != 0) {
    task->return_code_ = sr.rc_;
    CLIO_CO_RETURN;
  }
  const FsAttr &a = sr.attr_;
  task->exists_ = 1;
  task->is_dir_ = a.type_ == kFsTypeDir ? 1u : 0u;
  task->is_symlink_ = a.type_ == kFsTypeSymlink ? 1u : 0u;
  task->size_ = a.size_;
  task->ino_ = InoFromPacked(a.id_);
  task->atime_ = a.atime_;
  task->mtime_ = a.mtime_;
  task->ctime_ = a.ctime_;
  task->uid_ = a.uid_;
  task->gid_ = a.gid_;
  task->mode_ = a.mode_;
  task->nlink_ = a.nlink_;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::StatSize(clio::run::shared_ptr<StatSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  task->exists_ = 0;
  task->size_ = 0;
  CLIO_FS_RESOLVE(path, ent, erc);
  if (erc == 0) {
    FsResp sr;
    CLIO_CO_AWAIT(StatEntry(path, ent, sr));
    if (sr.rc_ == 0) {
      task->exists_ = 1;
      task->size_ = sr.attr_.size_;
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Truncate(clio::run::shared_ptr<TruncateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  FsReq r;
  r.a_ = task->new_size_;
  r.b_ = task->old_extent_;
  std::string path;
  if (task->tag_packed_ != 0) {
    // ftruncate through an open handle: by inode, never by name (the name
    // may be gone -- unlinked while open -- and must not be resurrected).
    r.id_ = task->tag_packed_;
  } else {
    path = FsNormPath(task->path_.str());
    CLIO_FS_RESOLVE(path, ent, erc);
    if (erc != 0) {
      task->return_code_ = erc;
      CLIO_CO_RETURN;
    }
    if (ent.type_ == kFsTypeDir) {
      task->return_code_ = EISDIR;
      CLIO_CO_RETURN;
    }
    r.id_ = FsPack(ent.id_);
    MirrorRefuse(path);  // pages are about to change under cached readers
  }
  FsResp tr;
  CLIO_CO_AWAIT(CallShard(InodeOwner(r.id_), kShardInodeTruncate, r, tr));
  // An fd-truncate of an inode already destroyed is a no-op (the file is
  // gone for everyone but this descriptor's stale view).
  task->return_code_ =
      (tr.rc_ == ENOENT && task->tag_packed_ != 0) ? 0 : tr.rc_;
  if (task->return_code_ != 0) {
    HLOG(kWarning, "filesystem: truncate of {} (id {}) to {} failed rc={}",
         path, r.id_, r.a_, task->return_code_);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Namespace mutations
// ===========================================================================

clio::run::TaskResume Runtime::Unlink(clio::run::shared_ptr<UnlinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  EnsurePurgeDrain();
  const std::string path = FsNormPath(task->path_.str());
  if (path == "/") {
    task->return_code_ = EISDIR;
    CLIO_CO_RETURN;
  }
  FsReq r;
  r.dir_ = FsParentDir(path);
  r.leaf_ = FsLeaf(path);
  r.flags_ = kRmNonDir;
  FsResp rr;
  CLIO_CO_AWAIT(CallShard(DirOwner(r.dir_), kShardRemove, r, rr));
  if (rr.rc_ != 0) {
    task->return_code_ = rr.rc_;
    CLIO_CO_RETURN;
  }
  MirrorErase(path);
  // The name is gone; drop its link on the inode (destroyed at nlink 0, or
  // at its last close if still open).
  CLIO_CO_AWAIT(UnlinkInode(rr.old_id_, path));
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Mkdir(clio::run::shared_ptr<MkdirTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  if (path == "/") {
    task->return_code_ = EEXIST;
    CLIO_CO_RETURN;
  }
  // 1. Reserve the name (invisible) on the parent's owner: exactly one of
  //    several racing mkdirs (any node) gets it.
  FsReq r;
  r.dir_ = FsParentDir(path);
  r.leaf_ = FsLeaf(path);
  r.type_ = kFsTypeDir;
  r.flags_ = kInsExcl | kInsPending;
  FsResp rr;
  const clio::run::u32 parent_owner = DirOwner(r.dir_);
  CLIO_CO_AWAIT(CallShard(parent_owner, kShardInsert, r, rr));
  if (rr.rc_ != 0) {
    task->return_code_ = rr.rc_;
    CLIO_CO_RETURN;
  }
  // 2. Create the directory's own state on ITS owner.
  FsReq c;
  c.dir_ = path;
  c.id_ = rr.id_;
  FsResp cr;
  CLIO_CO_AWAIT(CallShard(DirOwner(path), kShardDirCreate, c, cr));
  // 3. Publish (or roll back) the reservation.
  r.id_ = rr.id_;
  r.flags_ = cr.rc_ == 0 ? kInsCommit : 0u;
  FsResp fr;
  if (cr.rc_ == 0) {
    CLIO_CO_AWAIT(CallShard(parent_owner, kShardInsert, r, fr));
  } else {
    CLIO_CO_AWAIT(CallShard(parent_owner, kShardRemove, r, fr));
  }
  task->return_code_ = cr.rc_ != 0 ? cr.rc_ : fr.rc_;
  if (task->return_code_ == 0) {
    MirrorRefuse(r.dir_);
    MirrorDir(path, FsUnpack(rr.id_), /*complete=*/true);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Rmdir(clio::run::shared_ptr<RmdirTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  if (path == "/") {
    task->return_code_ = EBUSY;
    CLIO_CO_RETURN;
  }
  // 1. Mark the entry leaving (still visible; nobody else may change it).
  FsReq r;
  r.dir_ = FsParentDir(path);
  r.leaf_ = FsLeaf(path);
  r.flags_ = kRmDirOnly | kRmMarkLeaving;
  FsResp mr;
  const clio::run::u32 parent_owner = DirOwner(r.dir_);
  CLIO_CO_AWAIT(CallShard(parent_owner, kShardRemove, r, mr));
  if (mr.rc_ != 0) {
    task->return_code_ = mr.rc_;
    CLIO_CO_RETURN;
  }
  // 2. Retire the directory's state on its owner iff it is empty. A create
  //    into it serializes on that owner, so it either lands first (and this
  //    fails ENOTEMPTY) or finds the directory gone.
  FsReq d;
  d.dir_ = path;
  d.id_ = mr.old_id_;
  FsResp dr;
  CLIO_CO_AWAIT(CallShard(DirOwner(path), kShardDirRetire, d, dr));
  // 3. Remove the entry, or restore it.
  r.id_ = mr.old_id_;
  r.flags_ = dr.rc_ == 0 ? 0u : kRmRestore;
  FsResp fr;
  CLIO_CO_AWAIT(CallShard(parent_owner, kShardRemove, r, fr));
  task->return_code_ = dr.rc_ != 0 ? dr.rc_ : fr.rc_;
  if (task->return_code_ == 0) MirrorErase(path);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RenameFile(const std::string &src,
                                          const std::string &dst,
                                          const Dentry &se, int &rc) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 id = FsPack(se.id_);
  FsReq s;
  s.dir_ = FsParentDir(src);
  s.leaf_ = FsLeaf(src);
  s.id_ = id;
  s.flags_ = kRmMarkLeaving | kRmFailBusy;
  const clio::run::u32 o1 = DirOwner(s.dir_);
  FsResp sr;
  CLIO_CO_AWAIT(CallShard(o1, kShardRemove, s, sr));
  if (sr.rc_ != 0) {
    rc = static_cast<int>(sr.rc_);
    CLIO_CO_RETURN;
  }
  // Hold an extra link while the name moves, so a crash between the two
  // halves leaves an over-counted (leaked) file, never a dangling entry.
  FsReq n;
  n.id_ = id;
  n.a_ = 1;
  n.str_ = dst;
  FsResp nr;
  CLIO_CO_AWAIT(CallShard(InodeOwner(id), kShardInodeNlink, n, nr));
  FsReq ins;
  ins.dir_ = FsParentDir(dst);
  ins.leaf_ = FsLeaf(dst);
  ins.id_ = id;
  ins.type_ = se.type_;
  ins.flags_ = kInsReplace | kInsFailBusy;
  FsResp ir;
  CLIO_CO_AWAIT(CallShard(DirOwner(ins.dir_), kShardInsert, ins, ir));
  const bool same = ir.rc_ == 0 && ir.old_id_ == id;
  s.flags_ = (ir.rc_ != 0 || same) ? kRmRestore : 0u;
  FsResp fr;
  CLIO_CO_AWAIT(CallShard(o1, kShardRemove, s, fr));
  // Drop the handoff link. A failed or no-op rename keeps the source as the
  // inode's name; a completed one leaves the destination (set above).
  CLIO_CO_AWAIT(UnlinkInode(id, (ir.rc_ != 0 || same) ? dst : src));
  if (ir.rc_ != 0) {
    rc = static_cast<int>(ir.rc_);
    CLIO_CO_RETURN;
  }
  if (!same && ir.old_id_ != 0) {
    CLIO_CO_AWAIT(UnlinkInode(ir.old_id_, dst));  // the replaced destination
  }
  rc = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

namespace {
/** Child directory names listed in an exported directory blob. */
std::vector<std::string> ExportedSubdirs(const std::string &blob) {
  std::vector<std::string> out;
  FsDec d(blob.data(), blob.size());
  clio::run::u64 u64v = 0;
  clio::run::u32 u32v = 0, n = 0;
  if (!d.U64(&u64v) || !d.U32(&u32v) || !d.U32(&u32v) || !d.U32(&u32v) ||
      !d.U64(&u64v) || !d.U64(&u64v) || !d.U64(&u64v) || !d.U32(&n)) {
    return out;
  }
  for (clio::run::u32 i = 0; i < n; ++i) {
    std::string leaf;
    clio::run::u32 type = 0;
    if (!d.Str(&leaf) || !d.U64(&u64v) || !d.U32(&type)) break;
    if (type == kFsTypeDir) out.push_back(leaf);
  }
  return out;
}
}  // namespace

clio::run::TaskResume Runtime::RenameDir(const std::string &src,
                                         const std::string &dst,
                                         const Dentry &se, int &rc) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 id = FsPack(se.id_);
  FsReq s;
  s.dir_ = FsParentDir(src);
  s.leaf_ = FsLeaf(src);
  s.id_ = id;
  s.flags_ = kRmMarkLeaving | kRmFailBusy | kRmDirOnly;
  const clio::run::u32 o1 = DirOwner(s.dir_);
  FsResp sr;
  CLIO_CO_AWAIT(CallShard(o1, kShardRemove, s, sr));
  if (sr.rc_ != 0) {
    rc = static_cast<int>(sr.rc_);
    CLIO_CO_RETURN;
  }
  // Freeze the whole subtree: every directory state under src is exported
  // and marked moving on its owner. A directory already being moved (a
  // crossing rename) fails the export, and this rename backs off -- which
  // is what makes two crossing renames unable to form a cycle.
  std::vector<std::pair<std::string, std::string>> moved;  // (path, blob)
  rc = 0;
  {
    std::vector<std::string> todo{src};
    while (!todo.empty() && rc == 0) {
      std::string d = todo.back();
      todo.pop_back();
      FsReq ex;
      ex.dir_ = d;
      FsResp er;
      CLIO_CO_AWAIT(CallShard(DirOwner(d), kShardDirExport, ex, er));
      if (er.rc_ == ENOENT) continue;  // listing lost to a crash: empty dir
      if (er.rc_ != 0) {
        rc = static_cast<int>(er.rc_);
        break;
      }
      for (const std::string &c : ExportedSubdirs(er.str_)) {
        todo.push_back(FsJoin(d, c));
      }
      moved.emplace_back(d, std::move(er.str_));
    }
  }
  // Destination: absent, or an EMPTY directory that is retired first.
  if (rc == 0) {
    FsReq l;
    l.dir_ = FsParentDir(dst);
    l.leaf_ = FsLeaf(dst);
    FsResp lr;
    CLIO_CO_AWAIT(CallShard(DirOwner(l.dir_), kShardLookup, l, lr));
    if (lr.rc_ == 0 && lr.type_ != kFsTypeDir) {
      rc = ENOTDIR;
    } else if (lr.rc_ == 0) {
      FsReq ret;
      ret.dir_ = dst;
      ret.id_ = lr.id_;
      FsResp rr;
      CLIO_CO_AWAIT(CallShard(DirOwner(dst), kShardDirRetire, ret, rr));
      rc = static_cast<int>(rr.rc_);
    } else if (lr.rc_ != ENOENT) {
      rc = static_cast<int>(lr.rc_);
    }
  }
  // Install every state under its new path, then switch the entry.
  size_t imported = 0;
  for (; rc == 0 && imported < moved.size(); ++imported) {
    FsReq im;
    im.dir_ = dst + moved[imported].first.substr(src.size());
    im.str_ = moved[imported].second;
    FsResp ir;
    CLIO_CO_AWAIT(CallShard(DirOwner(im.dir_), kShardDirImport, im, ir));
    rc = static_cast<int>(ir.rc_);
    if (rc != 0) break;
  }
  if (rc == 0) {
    FsReq ins;
    ins.dir_ = FsParentDir(dst);
    ins.leaf_ = FsLeaf(dst);
    ins.id_ = id;
    ins.type_ = kFsTypeDir;
    ins.flags_ = kInsReplace | kInsReplaceDir | kInsFailBusy;
    FsResp ir;
    CLIO_CO_AWAIT(CallShard(DirOwner(ins.dir_), kShardInsert, ins, ir));
    rc = static_cast<int>(ir.rc_);
  }
  // Commit: drop the old states + the source entry. Abort: drop the new
  // states, unfreeze the old ones, restore the source entry.
  for (size_t i = 0; i < moved.size(); ++i) {
    FsReq x;
    x.dir_ = moved[i].first;
    FsResp xr;
    CLIO_CO_AWAIT(CallShard(DirOwner(x.dir_),
                            rc == 0 ? kShardDirDrop : kShardDirUnmark, x, xr));
    if (rc != 0 && i < imported) {
      FsReq y;
      y.dir_ = dst + moved[i].first.substr(src.size());
      FsResp yr;
      CLIO_CO_AWAIT(CallShard(DirOwner(y.dir_), kShardDirDrop, y, yr));
    }
  }
  s.flags_ = rc == 0 ? 0u : kRmRestore;
  FsResp fr;
  CLIO_CO_AWAIT(CallShard(o1, kShardRemove, s, fr));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Rename(clio::run::shared_ptr<RenameTask> &task) {
  CLIO_TASK_BODY_BEGIN
  EnsurePurgeDrain();
  const std::string src = FsNormPath(task->src_.str());
  const std::string dst = FsNormPath(task->dst_.str());
  if (src == dst) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  if (src == "/" || dst == "/") {
    task->return_code_ = EBUSY;
    CLIO_CO_RETURN;
  }
  if (dst.size() > src.size() && dst.compare(0, src.size(), src) == 0 &&
      dst[src.size()] == '/') {
    task->return_code_ = EINVAL;  // a directory into its own subtree
    CLIO_CO_RETURN;
  }
  int rc = EBUSY;
  bool is_dir = false;
  // Two-party operation: on contention with a crossing rename/link (EBUSY)
  // everything is rolled back and retried after a random backoff.
  for (int attempt = 0; attempt < 400 && rc == EBUSY; ++attempt) {
    if (attempt > 0) CLIO_CO_AWAIT(clio::run::yield(BackoffUs(attempt)));
    CLIO_FS_RESOLVE(src, se, erc);
    if (erc != 0) {
      rc = erc;
      break;
    }
    is_dir = se.type_ == kFsTypeDir;
    if (is_dir) {
      CLIO_CO_AWAIT(RenameDir(src, dst, se, rc));
    } else {
      CLIO_CO_AWAIT(RenameFile(src, dst, se, rc));
    }
  }
  if (rc == 0) {
    MirrorErase(src);
    MirrorErase(dst);
    MirrorRefuse(dst);
    MirrorRefuse(FsParentDir(dst));
    if (is_dir) shm_fs_cache_.BumpNsGen();
  }
  task->return_code_ = rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Link(clio::run::shared_ptr<LinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string target = FsNormPath(task->target_.str());
  const std::string link = FsNormPath(task->link_.str());
  CLIO_FS_RESOLVE(target, te, erc);
  if (erc != 0) {
    task->return_code_ = erc;
    CLIO_CO_RETURN;
  }
  if (te.type_ == kFsTypeDir) {
    task->return_code_ = EPERM;  // no hard links to directories
    CLIO_CO_RETURN;
  }
  const clio::run::u64 id = FsPack(te.id_);
  // Count the link first: a crash before the name lands over-counts (a
  // leak), never leaves a name whose inode can be destroyed under it.
  FsReq n;
  n.id_ = id;
  n.a_ = 1;
  FsResp nr;
  CLIO_CO_AWAIT(CallShard(InodeOwner(id), kShardInodeNlink, n, nr));
  if (nr.rc_ != 0) {
    task->return_code_ = nr.rc_;
    CLIO_CO_RETURN;
  }
  FsReq ins;
  ins.dir_ = FsParentDir(link);
  ins.leaf_ = FsLeaf(link);
  ins.id_ = id;
  ins.type_ = te.type_;
  ins.flags_ = kInsExcl;
  FsResp ir;
  CLIO_CO_AWAIT(CallShard(DirOwner(ins.dir_), kShardInsert, ins, ir));
  if (ir.rc_ != 0) {
    CLIO_CO_AWAIT(UnlinkInode(id, std::string()));
    task->return_code_ = ir.rc_;
    CLIO_CO_RETURN;
  }
  MirrorRefuse(target);
  MirrorRefuse(ins.dir_);
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Symlink(clio::run::shared_ptr<SymlinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  FsReq r;
  r.dir_ = FsParentDir(path);
  r.leaf_ = FsLeaf(path);
  r.type_ = kFsTypeSymlink;
  r.mode_ = 0777u;
  r.str_ = task->target_.str();
  r.flags_ = kInsExcl | kInsNewInode;
  MirrorRefuse(r.dir_);  // before the name can exist (libfuse's post-op stat)
  FsResp rr;
  CLIO_CO_AWAIT(CallShard(DirOwner(r.dir_), kShardInsert, r, rr));
  task->return_code_ = rr.rc_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Readlink(clio::run::shared_ptr<ReadlinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  CLIO_FS_RESOLVE(path, ent, erc);
  if (erc != 0) {
    task->return_code_ = erc;
    CLIO_CO_RETURN;
  }
  if (ent.type_ != kFsTypeSymlink) {
    task->return_code_ = EINVAL;
    CLIO_CO_RETURN;
  }
  FsResp sr;
  CLIO_CO_AWAIT(StatEntry(path, ent, sr));
  if (sr.rc_ == 0) {
    task->target_ = clio::run::priv::string(CTP_MALLOC, sr.str_);
  }
  task->return_code_ = sr.rc_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Attributes
// ===========================================================================

/**
 * Apply attribute request `req` (flags + values; id/dir filled here) to the
 * entry `path` resolves to: a directory's state on its owner, or an inode
 * on its home. Sets `rcv`.
 */
#define CLIO_FS_SETATTR(pathv, req, rcv)                                      \
  do {                                                                        \
    CLIO_FS_RESOLVE(pathv, _e, _erc);                                         \
    if (_erc != 0) {                                                          \
      rcv = _erc;                                                             \
      break;                                                                  \
    }                                                                         \
    (req).id_ = FsPack(_e.id_);                                               \
    FsResp _ar;                                                               \
    if (_e.type_ == kFsTypeDir) {                                             \
      (req).dir_ = (pathv);                                                   \
      (req).flags_ |= kSetCtimeOnly | kAttrRepair;                            \
      CLIO_CO_AWAIT(CallShard(DirOwner(pathv), kShardDirAttr, req, _ar));     \
    } else {                                                                  \
      CLIO_CO_AWAIT(                                                          \
          CallShard(InodeOwner((req).id_), kShardInodeSetAttr, req, _ar));    \
    }                                                                         \
    rcv = static_cast<int>(_ar.rc_);                                          \
  } while (0)

clio::run::TaskResume Runtime::Utimens(clio::run::shared_ptr<UtimensTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  FsReq r;
  // Same bit layout as the task: bit0/1 explicit atime/mtime, bit2/3 NOW
  // (resolved on the owner so the stamp shares its clock).
  r.flags_ = task->flags_ & (kSetAtime | kSetMtime | kSetAtimeNow | kSetMtimeNow);
  r.a_ = task->atime_ns_;
  r.b_ = task->mtime_ns_;
  int rc = 0;
  CLIO_FS_SETATTR(path, r, rc);
  task->return_code_ = rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Chown(clio::run::shared_ptr<ChownTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  FsReq r;
  if (task->uid_ != 0xFFFFFFFFu) { r.flags_ |= kSetUid; r.uid_ = task->uid_; }
  if (task->gid_ != 0xFFFFFFFFu) { r.flags_ |= kSetGid; r.gid_ = task->gid_; }
  if (task->mode_ != 0xFFFFFFFFu) { r.flags_ |= kSetMode; r.mode_ = task->mode_; }
  int rc = 0;
  CLIO_FS_SETATTR(path, r, rc);
  task->return_code_ = rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Readdir(clio::run::shared_ptr<ReaddirTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // The directory's whole listing lives on this container (the client
  // routed by the directory's own path): one local scan, no fan-out.
  const std::string dir = FsNormPath(task->path_.str());
  task->entries_ = clio::run::priv::vector<clio::run::priv::string>(CTP_MALLOC);
  task->inos_ = clio::run::priv::vector<clio::run::u64>(CTP_MALLOC);
  std::vector<std::pair<std::string, clio::run::u64>> listing;
  int rc = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    Dentry probe;
    LookupLocked(dir, std::string(), &probe);  // materializes "/" if needed
    auto it = dirs_.find(dir);
    if (it == dirs_.end()) {
      rc = ENOENT;
    } else {
      listing.reserve(it->second->ents_.size());
      for (const auto &kv : it->second->ents_) {
        if (kv.second.state_ == kEntPending) continue;
        listing.emplace_back(kv.first, FsPack(kv.second.id_));
      }
    }
  }
  task->entries_.reserve(listing.size());
  task->inos_.reserve(listing.size());
  for (const auto &kv : listing) {
    task->entries_.push_back(
        clio::run::priv::string(CTP_MALLOC, FsJoin(dir, kv.first)));
    task->inos_.push_back(InoFromPacked(kv.second));
  }
  task->return_code_ = rc;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Extended attributes
// ===========================================================================

clio::run::TaskResume Runtime::InodeXattr(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  resp.rc_ = 0;
  const bool no_inode = (req.flags_ & kXattrNoInode) != 0;
  std::shared_ptr<FileInfo> fi;
  if (!no_inode) {
    fi = FindInode(req.id_);
    if (fi == nullptr) {
      resp.rc_ = ENOENT;
      CLIO_CO_RETURN;
    }
  }
  // Serialize this id's read-modify-write (cooperative wait, no lock held
  // across the awaits below).
  for (;;) {
    {
      std::lock_guard<std::mutex> g(xattr_mu_);
      if (xattr_busy_.insert(req.id_).second) break;
    }
    CLIO_CO_AWAIT(clio::run::yield(20));
  }
  const std::string key = XattrKey(req.id_);
  std::vector<std::pair<std::string, std::string>> xa;
  {
    auto s = cte_.AsyncGetBlobSize(xattr_tag_id_, key,
                                   clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(s);
    const clio::run::u64 len = s->GetReturnCode() == 0 ? s->size_ : 0;
    if (len > 0) {
      auto *ipc = CLIO_IPC;
      ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(len);
      if (!buf.IsNull()) {
        auto g = cte_.AsyncGetBlob(xattr_tag_id_, key, 0, len, 0u,
                                   buf.shm_.template Cast<void>(),
                                   clio::run::PoolQuery::Dynamic());
        CLIO_CO_AWAIT(g);
        if (g->GetReturnCode() == 0) xa = DeserializeXattrs(buf.ptr_, len);
        ipc->FreeBuffer(buf);
      }
    }
  }
  bool store = false;
  const clio::run::u32 xflags = req.flags_ & 0xFFu;
  auto it = std::find_if(xa.begin(), xa.end(),
                         [&](const auto &kv) { return kv.first == req.str_; });
  switch (req.type_) {
    case kXattrGet:
      resp.created_ = it != xa.end() ? 1u : 0u;  // "found"
      if (it != xa.end()) resp.str_ = it->second;
      break;
    case kXattrList:
      for (const auto &kv : xa) {
        resp.str_.append(kv.first);
        resp.str_.push_back('\0');
      }
      break;
    case kXattrSet:
      if (it != xa.end() && (xflags & 0x1u)) {         // XATTR_CREATE
        resp.rc_ = EEXIST;
      } else if (it == xa.end() && (xflags & 0x2u)) {  // XATTR_REPLACE
        resp.rc_ = ENODATA;
      } else {
        if (it != xa.end()) it->second = req.str2_;
        else xa.emplace_back(req.str_, req.str2_);
        store = true;
      }
      break;
    case kXattrRemove:
      if (it == xa.end()) {
        resp.rc_ = ENODATA;
      } else {
        xa.erase(it);
        store = true;
      }
      break;
    default:
      resp.rc_ = EINVAL;
      break;
  }
  if (store) {
    const std::string payload = SerializeXattrs(xa);
    if (payload.empty()) {
      // No xattrs remain: delete the blob (a size-0 replace put fails).
      auto d = cte_.AsyncDelBlob(xattr_tag_id_, key,
                                 clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(d);
    } else {
      auto *ipc = CLIO_IPC;
      ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(payload.size());
      if (buf.IsNull()) {
        resp.rc_ = EIO;
      } else {
        std::memcpy(buf.ptr_, payload.data(), payload.size());
        auto p = cte_.AsyncPutBlob(xattr_tag_id_, key, 0, payload.size(),
                                   buf.shm_.template Cast<void>(), -1.0f,
                                   clio::cte::core::Context(),
                                   clio::cte::core::kCtePutReplace,
                                   clio::run::PoolQuery::Dynamic());
        CLIO_CO_AWAIT(p);
        if (p->GetReturnCode() != 0) resp.rc_ = EIO;
        ipc->FreeBuffer(buf);
      }
    }
    if (fi != nullptr && resp.rc_ == 0) {
      std::lock_guard<std::mutex> g(meta_mu_);
      fi->has_xattr_ = true;
      fi->ctime_ = clio::cte::core::GetWallTimeNs();
      LogInode(*fi);
    }
  }
  {
    std::lock_guard<std::mutex> g(xattr_mu_);
    xattr_busy_.erase(req.id_);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Run xattr sub-op `subop` on `pathv` (name/value/flags from the task) on
 * the owner of the entry's attributes; leaves the response in `xr`.
 */
#define CLIO_FS_XATTR(pathv, subop, namev, valuev, flagsv, xr)                \
  FsResp xr;                                                                  \
  do {                                                                        \
    CLIO_FS_RESOLVE(pathv, _e, _erc);                                         \
    if (_erc != 0) {                                                          \
      xr.rc_ = static_cast<clio::run::u32>(_erc);                             \
      break;                                                                  \
    }                                                                         \
    FsReq _xq;                                                                \
    _xq.id_ = FsPack(_e.id_);                                                 \
    _xq.type_ = (subop);                                                      \
    _xq.str_ = (namev);                                                       \
    _xq.str2_ = (valuev);                                                     \
    _xq.flags_ = (flagsv) & 0xFFu;                                            \
    clio::run::u32 _owner = InodeOwner(_xq.id_);                              \
    if (_e.type_ == kFsTypeDir) {                                             \
      /* A directory has no inode home: its xattrs are serialized on the */  \
      /* owner of its entry, which every request for this name reaches. */   \
      _xq.flags_ |= kXattrNoInode;                                            \
      _owner = DirOwner(FsParentDir(pathv));                                  \
    }                                                                         \
    CLIO_CO_AWAIT(CallShard(_owner, kShardInodeXattr, _xq, xr));              \
  } while (0)

clio::run::TaskResume Runtime::Setxattr(clio::run::shared_ptr<SetxattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  CLIO_FS_XATTR(path, kXattrSet, task->name_.str(), task->value_.str(),
                task->flags_, xr);
  task->return_code_ = xr.rc_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Getxattr(clio::run::shared_ptr<GetxattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  CLIO_FS_XATTR(path, kXattrGet, task->name_.str(), std::string(), 0u, xr);
  task->found_ = xr.created_;
  if (xr.rc_ == 0 && xr.created_) {
    task->value_ = clio::run::priv::string(CTP_MALLOC, xr.str_);
  }
  task->return_code_ = xr.rc_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Listxattr(clio::run::shared_ptr<ListxattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  CLIO_FS_XATTR(path, kXattrList, std::string(), std::string(), 0u, xr);
  task->names_ = clio::run::priv::string(CTP_MALLOC, xr.str_);
  task->return_code_ = xr.rc_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Removexattr(
    clio::run::shared_ptr<RemovexattrTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  CLIO_FS_XATTR(path, kXattrRemove, task->name_.str(), std::string(), 0u, xr);
  task->return_code_ = xr.rc_;
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
  // same tag aggregates at that tag's sequencer. Awaited one tag at a time:
  // the CPU await path supports a single outstanding subtask future.
  for (auto &kv : by_tag) {
    const clio::cte::core::TagId &tag = kv.first;
    clio::run::u32 chash = static_cast<clio::run::u32>(FsMix64(FsPack(tag)));
    auto q = clio::run::PoolQuery::ManyToOne(chash, FsPack(tag),
                                             /*batch_for_ns=*/50000);
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
  // Runs ONCE per batch as the ManyToOne aggregate. The merge must suspend,
  // which the aggregate cannot, so it is delegated to AppendPlan and AWAITED:
  // that keeps this aggregate (AppendPlan's parent) alive and holds the
  // batch claim so no second batch for the tag merges concurrently.
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
  clio::cte::core::TagId tag_id = task->tag_id_;
  std::vector<AppendEntry> entries(task->entries_.begin(),
                                   task->entries_.end());
  std::sort(entries.begin(), entries.end(),
            [](const AppendEntry &a, const AppendEntry &b) {
              if (a.utc_ns_ != b.utc_ns_) return a.utc_ns_ < b.utc_ns_;
              return a.logical_ < b.logical_;
            });
  // Tail = the file's merged content size (staged bytes live elsewhere); at
  // most one batch per tag merges at a time, so this read is stable.
  clio::run::u64 cur_size = 0;
  {
    auto s = cte_.AsyncGetTagSize(tag_id, clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(s);
    cur_size = (s->GetReturnCode() == 0) ? s->tag_size_ : 0;
  }
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
  // Slices of up to 16 MiB; a data blob never spans two slices, so exactly
  // one execution task deletes it. Awaited one at a time (single
  // outstanding future).
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
      bool at_blob_boundary =
          (i >= plan.size()) ||
          (plan[i].data_blob_id_ != slice.back().data_blob_id_);
      if (bytes >= kMaxExecBytes && at_blob_boundary) break;
    }
    auto q = clio::run::PoolQuery::DirectHash(spread++);
    auto f = self_.AsyncAppendExecution(tag_id, staging_tag_id_, slice, q);
    CLIO_CO_AWAIT(f);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AppendExecution(
    clio::run::shared_ptr<AppendExecutionTask> &task) {
  CLIO_TASK_BODY_BEGIN
  clio::cte::core::TagId tag_id = task->tag_id_;
  clio::cte::core::TagId staging = task->staging_tag_id_;
  auto *ipc = CLIO_IPC;
  const size_t n = task->steps_.size();
  bool ok = true;
  // Strictly sequential: at most one subtask future may be outstanding (the
  // CPU await path does not record which future a coroutine waits on).
  for (size_t i = 0; i < n && ok; ++i) {
    const AppendPlanStep &s = task->steps_[i];
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(s.size_);
    if (buf.IsNull()) {
      ok = false;
      break;
    }
    auto g = cte_.AsyncGetBlob(staging, s.data_blob_id_, s.off_in_data_,
                               s.size_, 0u, buf.shm_.template Cast<void>(),
                               clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(g);
    auto p = cte_.AsyncPutBlob(
        tag_id, std::to_string(s.file_page_), s.off_in_page_, s.size_,
        buf.shm_.template Cast<void>(), -1.0f, clio::cte::core::Context(), 0u,
        clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() != 0) ok = false;
    ipc->FreeBuffer(buf);
  }
  std::unordered_set<std::string> seen;
  for (const auto &s : task->steps_) {
    if (seen.insert(s.data_blob_id_).second) {
      auto d = cte_.AsyncDelBlob(staging, s.data_blob_id_,
                                 clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(d);
    }
  }
  // The tail lives in the file's own pages again: re-publish without the
  // pending-append refusal.
  if (ok) {
    std::shared_ptr<FileInfo> fi = FindInode(FsPack(tag_id));
    if (fi != nullptr) {
      std::lock_guard<std::mutex> g(meta_mu_);
      if (!fi->path_.empty()) MirrorFile(fi->path_, *fi);
    }
  }
  task->return_code_ = ok ? 0 : EIO;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

#undef CLIO_FS_LOOKUP
#undef CLIO_FS_RESOLVE
#undef CLIO_FS_SETATTR
#undef CLIO_FS_XATTR

}  // namespace clio::cte::filesystem

// Define ChiMod entry points (alloc/new/name/destroy) so the runtime's module
// manager can dlopen and instantiate this chimod.
CLIO_TASK_CC(clio::cte::filesystem::Runtime)
