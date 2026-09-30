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
  kSetCtimeOnly = 256u, kAccessTouch = 512u, kAccessStrict = 1024u,
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

  // Every file's logical size and deferred appends belong to the stream
  // pool. Deployments compose it before this pool; create it with defaults
  // (over the same chain, logging next to the namespace log) otherwise.
  stream_.Init(cfg.stream_pool_id_);
  {
    clio::cte::stream::StreamConfig sc;
    sc.next_pool_id_ = next_pool_id_;
    if (!cfg.metadata_log_path_.empty()) {
      sc.log_path_ =
          ctp::ConfigParse::ExpandPath(cfg.metadata_log_path_) + ".stream";
    }
    auto sp = stream_.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                  clio::cte::stream::kStreamPoolName,
                                  cfg.stream_pool_id_, sc);
    CLIO_CO_AWAIT(sp);
    if (sp->GetReturnCode() != 0) {
      HLOG(kError, "filesystem: stream pool {} unavailable (rc {}); file "
           "sizes cannot be tracked", cfg.stream_pool_id_,
           sp->GetReturnCode());
    }
    auto *pm = CLIO_POOL_MANAGER;
    clio::run::ContainerHold h =
        pm->GetRealOrStaticContainer(cfg.stream_pool_id_).get();
    if (h && dynamic_cast<clio::cte::stream::Runtime *>(&*h) != nullptr) {
      stream_hold_ = h;
    } else {
      HLOG(kError, "filesystem: no local stream container for pool {}",
           cfg.stream_pool_id_);
    }
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
  // clio-fs's own records (id reservations, orphans): one tag, one blob per
  // container and record.
  {
    auto st = cte_.AsyncGetOrCreateTag("_clio_fs_sys",
                                       clio::cte::core::TagId::GetNull(),
                                       clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(st);
    if (st->GetReturnCode() == 0) sys_tag_id_ = st->tag_id_;
  }
  split_entries_ = std::max<clio::run::u32>(cfg.dir_split_entries_, 16);
  // The namespace itself lives in CTE blobs (directory blocks, inode
  // records) and loads on demand; only this container's id reservation and
  // orphan list are read up front.
  CLIO_CO_AWAIT(LoadSysRecords());
  if (is_restart_) CLIO_CO_AWAIT(ReconcileRestoredStreams());
  // Mirror the namespace into CTE tag names (async). The root first; on a
  // restart this node missed broadcasts while down, so drop its published
  // names, re-add the ones this container owns, and pull the rest from peers.
  PublishName(clio::cte::core::TagNameOp::kSetRoot, FsRootId(), "/");
  if (is_restart_) {
    std::string reset;
    clio::cte::core::EncodeTagNameOp(
        &reset, clio::cte::core::TagNameOp::kResetPublished, FsRootId(),
        clio::cte::core::GetWallTimeNs(), std::string());
    clio::cte::core::EncodeTagNameOp(
        &reset, clio::cte::core::TagNameOp::kSetRoot, FsRootId(),
        clio::cte::core::GetWallTimeNs(), "/");
    {
      std::string own;
      CLIO_CO_AWAIT(EncodeHomeNames(&own));
      reset += own;
    }
    auto u = cte_.AsyncUpdateTagNames(reset, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(u);
    const clio::run::u32 n = NumContainers();
    for (clio::run::u32 c = 0; c < n; ++c) {
      if (c != container_id_) catchup_missing_.insert(c);
    }
    catchup_pending_ = !catchup_missing_.empty();
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
  EnsurePurgeDrain();  // also broadcasts the queued tag names
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReconcileRestoredStreams() {
  CLIO_TASK_BODY_BEGIN
  if (!stream_hold_) CLIO_CO_RETURN;
  auto &stream = static_cast<clio::cte::stream::Runtime &>(*stream_hold_);
  const std::vector<clio::cte::core::TagId> tags = stream.UnverifiedStreams();
  const auto t0 = std::chrono::steady_clock::now();
  size_t loaded = 0;
  for (const clio::cte::core::TagId &tag : tags) {
    const clio::run::u64 packed = FsPack(tag);
    if (InodeOwner(packed) != container_id_) continue;
    // Loading the inode reconciles its stream with the record (see
    // EnsureInode) and releases it.
    CLIO_CO_AWAIT(EnsureInode(packed));
    ++loaded;
  }
  stream.ReleaseRestored();  // the rest have no record to reconcile with
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
  HLOG(kInfo, "filesystem: reconciled {} of {} restored file sizes in {} ms",
       loaded, tags.size(), ms);
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
  rec.size_ = FileSize(fi);
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

clio::run::TaskResume Runtime::StatEntry(clio::run::u64 parent,
                                         const std::string &leaf,
                                         const DirEntry &e, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (e.type_ != kFsTypeDir) {
    CLIO_CO_AWAIT(StatInode(e.id_, resp));
    CLIO_CO_RETURN;
  }
  CLIO_CO_AWAIT(DirStat(e.id_, resp));
  if (resp.rc_ == ENOENT && e.state_ == kDirEntLive && parent != 0) {
    // The parent's live entry is authoritative: a listing missing behind it
    // (crash between rmdir's halves) is recreated rather than reported gone.
    FsReq r;
    r.dir_id_ = e.id_;
    r.b_ = parent;
    r.leaf_ = leaf;
    r.flags_ = 128u;  // kAttrRepair
    FsResp cr;
    CLIO_CO_AWAIT(CallShard(BlockHome(e.id_, 0), kShardDirCreate, r, cr));
    CLIO_CO_AWAIT(DirStat(e.id_, resp));
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
 * Resolve `path` through the local cache into `ent` (its entry), `par` (its
 * parent directory's id) and `erc` (0, ENOENT, ENOTDIR or EIO).
 */
#define CLIO_FS_RESOLVE(pathv, ent, par, erc)                                 \
  DirEntry ent;                                                               \
  clio::run::u64 par = 0;                                                     \
  int erc = 0;                                                                \
  CLIO_CO_AWAIT(ResolvePath((pathv), ent, par, erc))

/**
 * Resolve the directory that will hold `pathv`'s entry into `dent` / `derc`
 * (ENOTDIR when it is not a directory).
 */
#define CLIO_FS_PARENT(pathv, dent, derc)                                     \
  CLIO_FS_RESOLVE(FsParentDir(pathv), dent, dent##_par, derc);                \
  (void)dent##_par;                                                           \
  if (derc == 0 && dent.type_ != kFsTypeDir) derc = ENOTDIR

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
  CLIO_FS_PARENT(path, pe, perc);
  if (perc != 0) {
    // handle_ = 0 means ENOENT to the client. A create whose parent is gone
    // says so outright: a caller that forgets to check the handle must not
    // think it created a file in a removed directory.
    task->return_code_ =
        (perc == ENOENT && (task->flags_ & O_CREAT) == 0) ? 0 : perc;
    CLIO_CO_RETURN;
  }
  FsReq r;
  r.dir_id_ = pe.id_;
  r.dir_ = FsParentDir(path);
  r.leaf_ = FsLeaf(path);
  FsResp er;
  if (task->flags_ & O_CREAT) {
    // Create-or-open in ONE mutation on the home of the name's block: of
    // several racing creators (any node) exactly one sees created_=1, which
    // is what O_EXCL keys off.
    r.type_ = kFsTypeFile;
    r.mode_ = task->mode_ & 07777u;
    r.flags_ = kInsNewInode | ((task->flags_ & O_EXCL) ? kInsExcl : 0u);
    CLIO_CO_AWAIT(EntryOp(kShardInsert, r, er));
  } else {
    DirEntry e;
    int lrc = 0;
    CLIO_CO_AWAIT(LookupEntry(pe.id_, r.leaf_, e, lrc));
    if (lrc == ENOENT) {
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
    er.rc_ = static_cast<clio::run::u32>(lrc);
    er.id_ = e.id_;
    er.type_ = e.type_;
  }
  if (er.rc_ != 0) {
    task->return_code_ = er.rc_;
    CLIO_CO_RETURN;
  }
  if (er.type_ == kFsTypeDir) {
    task->return_code_ = EISDIR;
    CLIO_CO_RETURN;
  }
  // Second stage on the inode's home (where it was created; a rename or
  // link can put its name in a block another node homes).
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

clio::run::u64 Runtime::FileSize(const FileInfo &fi) {
  if (fi.type_ == kFsTypeSymlink) return fi.symlink_.size();
  if (!stream_hold_) return 0;
  // This container is the inode's home and therefore its stream's home: the
  // size is read from the co-located stream container, no task, no network.
  auto &stream = static_cast<clio::cte::stream::Runtime &>(*stream_hold_);
  clio::run::u64 size = 0;
  stream.LocalSize(fi.tag_id_, &size);
  return size;
}

clio::run::TaskResume Runtime::FileSizeOp(clio::cte::core::TagId tag,
                                          clio::cte::stream::StreamSizeOp op,
                                          clio::run::u64 value,
                                          clio::run::u64 *old_size,
                                          clio::run::u64 *new_size,
                                          clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  auto f = stream_.AsyncSizeOp(tag, container_id_, op, value);
  CLIO_CO_AWAIT(f);
  *rc = f->GetReturnCode();
  if (old_size != nullptr) *old_size = f->old_size_;
  if (new_size != nullptr) *new_size = f->new_size_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SyncMeta(
    clio::run::shared_ptr<SyncMetaTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // The namespace lives in CTE blobs now (directory blocks, inode records);
  // fsync/fsyncdir make them durable with SyncTag on the tags involved. This
  // container only has to sync its own system records.
  if (!sys_tag_id_.IsNull()) {
    auto st = cte_.AsyncSyncTag(sys_tag_id_, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(st);
    task->return_code_ = st->GetReturnCode() == 0 ? 0 : 5;  // EIO
  } else {
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AdvanceSize(
    clio::run::shared_ptr<AdvanceSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // Clients send it here (Local); it runs on the inode's live home.
    const clio::run::u32 owner = InodeOwner(task->tag_packed_);
    if (owner != container_id_) {
      auto f = self_.AsyncAdvanceSizeAt(
          clio::run::PoolQuery::DirectId(
              static_cast<clio::run::ContainerId>(owner)),
          task->tag_packed_, task->size_, task->reserve_);
      CLIO_CO_AWAIT(f);
      task->old_size_ = f->old_size_;
      task->return_code_ = f->GetReturnCode();
      CLIO_CO_RETURN;
    }
  }
  CLIO_CO_AWAIT(EnsureInode(task->tag_packed_));
  std::shared_ptr<FileInfo> fi = FindInode(task->tag_packed_);
  if (fi == nullptr) {
    task->return_code_ = ENOENT;
    CLIO_CO_RETURN;
  }
  // A size push means the file was WRITTEN (the adapter only sends one for a
  // handle that wrote): it is the file's new mtime. The size itself lives in
  // the file's stream (durable in the stream log once this returns).
  {
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(FileSizeOp(fi->tag_id_,
                             task->reserve_ != 0
                                 ? clio::cte::stream::StreamSizeOp::kReserve
                                 : clio::cte::stream::StreamSizeOp::kMax,
                             task->size_, &task->old_size_, nullptr, &rc));
    if (rc != 0) {
      task->return_code_ = EIO;
      CLIO_CO_RETURN;
    }
  }
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    const clio::run::u64 now = clio::cte::core::GetWallTimeNs();
    fi->mtime_ = fi->ctime_ = now;
    LogInode(*fi);
    if (!fi->path_.empty()) MirrorFile(fi->path_, *fi);
  }
  CLIO_CO_AWAIT(FlushInodes());  // the record carries the new size and mtime
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiCreate(
    clio::run::shared_ptr<MultiCreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Batched file creation (sieve create): each entry adopts its
  // client-minted id on the home of its name's block. Per-entry failures
  // don't stop the batch; the first is reported.
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
    CLIO_FS_PARENT(path, pe, perc);
    FsResp er;
    if (perc == 0) {
      FsReq r;
      r.dir_id_ = pe.id_;
      r.dir_ = FsParentDir(path);
      r.leaf_ = FsLeaf(path);
      r.id_ = e.tag_packed_;
      r.type_ = kFsTypeFile;
      r.mode_ = e.mode_ & 07777u;
      r.flags_ = kInsNewInode | kInsExcl;
      CLIO_CO_AWAIT(EntryOp(kShardInsert, r, er));
    } else {
      er.rc_ = static_cast<clio::run::u32>(perc);
    }
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
  std::shared_ptr<FileInfo> fi;
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = handles_.find(task->handle_);
    if (it != handles_.end()) {
      fi = it->second;
      handles_.erase(it);
    }
  }
  if (fi == nullptr) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  // Sieve-written files advance their logical size at close: the handle's
  // FileInfo tracks the file across renames.
  if (task->advance_size_ != 0) {
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(FileSizeOp(fi->tag_id_, clio::cte::stream::StreamSizeOp::kMax,
                             task->advance_size_, nullptr, nullptr, &rc));
  }
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    if (fi->open_count_ > 0) fi->open_count_--;
    if (task->advance_size_ != 0) fi->dirty_ = true;
    if (fi->orphan_ && fi->open_count_ == 0) {
      DropInodeLocked(fi);
    } else if (fi->dirty_) {
      fi->dirty_ = false;
      LogInode(*fi);
      if (!fi->path_.empty()) MirrorFile(fi->path_, *fi);
    }
  }
  CLIO_CO_AWAIT(FlushInodes());
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
  clio::run::u64 file_size = FileSize(*fi);

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
  bool no_space = false;  // the store is full: ENOSPC, not EIO
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
    if (p->GetReturnCode() != 0) {
      ok = false;
      no_space = clio::cte::core::PutRcIsNoSpace(p->GetReturnCode());
      break;
    }
    done += to_write;
    cur += to_write;
  }
  clio::run::u64 end = task->offset_ + done;
  clio::run::u64 new_size = 0;
  {
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(FileSizeOp(tag_id, clio::cte::stream::StreamSizeOp::kMax,
                             end, nullptr, &new_size, &rc));
    if (rc != 0) ok = false;
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
  task->new_size_ = new_size;
  task->return_code_ = ok ? 0 : (no_space ? ENOSPC : EIO);
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
  CLIO_FS_RESOLVE(path, ent, par, erc);
  if (erc == ENOENT || erc == ENOTDIR) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  if (erc != 0) {
    task->return_code_ = erc;
    CLIO_CO_RETURN;
  }
  FsResp sr;
  CLIO_CO_AWAIT(StatEntry(par, FsLeaf(path), ent, sr));
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
  CLIO_FS_RESOLVE(path, ent, par, erc);
  if (erc == 0) {
    FsResp sr;
    CLIO_CO_AWAIT(StatEntry(par, FsLeaf(path), ent, sr));
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
    CLIO_FS_RESOLVE(path, ent, par, erc);
    (void)par;
    if (erc != 0) {
      task->return_code_ = erc;
      CLIO_CO_RETURN;
    }
    if (ent.type_ == kFsTypeDir) {
      task->return_code_ = EISDIR;
      CLIO_CO_RETURN;
    }
    r.id_ = ent.id_;
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
  CLIO_FS_PARENT(path, pe, perc);
  if (perc != 0) {
    task->return_code_ = perc;
    CLIO_CO_RETURN;
  }
  FsReq r;
  r.dir_id_ = pe.id_;
  r.leaf_ = FsLeaf(path);
  r.flags_ = kRmNonDir;
  FsResp rr;
  CLIO_CO_AWAIT(EntryOp(kShardRemove, r, rr));
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
  CLIO_FS_PARENT(path, pe, perc);
  if (perc != 0) {
    task->return_code_ = perc;
    CLIO_CO_RETURN;
  }
  // 1. Reserve the name (invisible) on its block's home: exactly one of
  //    several racing mkdirs (any node) gets it, and mints the id.
  FsReq r;
  r.dir_id_ = pe.id_;
  r.leaf_ = FsLeaf(path);
  r.type_ = kFsTypeDir;
  r.flags_ = kInsExcl | kInsPending;
  FsResp rr;
  CLIO_CO_AWAIT(EntryOp(kShardInsert, r, rr));
  if (rr.rc_ != 0) {
    task->return_code_ = rr.rc_;
    CLIO_CO_RETURN;
  }
  // 2. Create the new directory's block 0 on ITS home.
  FsReq c;
  c.dir_id_ = rr.id_;
  c.b_ = pe.id_;
  c.leaf_ = r.leaf_;
  FsResp cr;
  CLIO_CO_AWAIT(CallShard(BlockHome(rr.id_, 0), kShardDirCreate, c, cr));
  // 3. Publish (or roll back) the reservation.
  r.id_ = rr.id_;
  r.flags_ = cr.rc_ == 0 ? kInsCommit : 0u;
  FsResp fr;
  CLIO_CO_AWAIT(EntryOp(cr.rc_ == 0 ? kShardInsert : kShardRemove, r, fr));
  task->return_code_ = cr.rc_ != 0 ? cr.rc_ : fr.rc_;
  if (cr.rc_ == 0 && fr.rc_ != 0) {
    // Lost the name: nothing can reach the new directory, drop its block.
    CLIO_CO_AWAIT(DropDir(rr.id_));
  }
  if (task->return_code_ == 0) {
    MirrorRefuse(FsParentDir(path));
    MirrorDir(path, FsUnpack(rr.id_), /*complete=*/true);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RemoveDir(clio::run::u64 parent,
                                         const std::string &leaf,
                                         clio::run::u64 expect, int &rc) {
  CLIO_TASK_BODY_BEGIN
  // 1. Mark the entry leaving (still visible; nobody else may change it).
  FsReq r;
  r.dir_id_ = parent;
  r.leaf_ = leaf;
  r.id_ = expect;
  r.flags_ = kRmDirOnly | kRmMarkLeaving;
  FsResp mr;
  CLIO_CO_AWAIT(EntryOp(kShardRemove, r, mr));
  if (mr.rc_ != 0) {
    rc = static_cast<int>(mr.rc_);
    CLIO_CO_RETURN;
  }
  // 2. Seal every block iff all are empty. A create into it lands on some
  //    block's home first (and this fails ENOTEMPTY) or finds it sealed.
  const clio::run::u64 dir = mr.old_id_;
  int src = 0;
  CLIO_CO_AWAIT(SealDir(dir, true, src));
  if (src == 0) CLIO_CO_AWAIT(DropDir(dir));
  // 3. Remove the entry, or restore it.
  r.id_ = dir;
  r.flags_ = src == 0 ? 0u : kRmRestore;
  FsResp fr;
  CLIO_CO_AWAIT(EntryOp(kShardRemove, r, fr));
  rc = src != 0 ? src : static_cast<int>(fr.rc_);
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
  CLIO_FS_PARENT(path, pe, perc);
  if (perc != 0) {
    task->return_code_ = perc;
    CLIO_CO_RETURN;
  }
  int rc = 0;
  CLIO_CO_AWAIT(RemoveDir(pe.id_, FsLeaf(path), 0, rc));
  task->return_code_ = rc;
  if (rc == 0) MirrorErase(path);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RenameFile(const std::string &src,
                                          const std::string &dst,
                                          clio::run::u64 sp, clio::run::u64 dp,
                                          const DirEntry &se, int &rc) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 id = se.id_;
  FsReq s;
  s.dir_id_ = sp;
  s.leaf_ = FsLeaf(src);
  s.id_ = id;
  s.flags_ = kRmMarkLeaving | kRmFailBusy;
  FsResp sr;
  CLIO_CO_AWAIT(EntryOp(kShardRemove, s, sr));
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
  ins.dir_id_ = dp;
  ins.leaf_ = FsLeaf(dst);
  ins.id_ = id;
  ins.type_ = se.type_;
  ins.flags_ = kInsReplace | kInsFailBusy | kInsNoTagName;
  FsResp ir;
  CLIO_CO_AWAIT(EntryOp(kShardInsert, ins, ir));
  const bool same = ir.rc_ == 0 && ir.old_id_ == id;
  s.flags_ = (ir.rc_ != 0 || same) ? kRmRestore : kRmNoTagName;
  FsResp fr;
  CLIO_CO_AWAIT(EntryOp(kShardRemove, s, fr));
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
  if (!same) {
    // ONE name change for the whole move (the per-home steps publish none).
    const std::string to =
        clio::cte::core::MakeTagRefName(FsUnpack(dp), FsLeaf(dst));
    if (ir.old_id_ != 0) {
      PublishName(clio::cte::core::TagNameOp::kRemoveName,
                  FsUnpack(ir.old_id_), to);
    }
    PublishName(clio::cte::core::TagNameOp::kRename, FsUnpack(id),
                clio::cte::core::MakeTagRefName(FsUnpack(sp), FsLeaf(src)),
                to);
  }
  rc = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RenameDir(const std::string &src,
                                         const std::string &dst,
                                         clio::run::u64 sp, clio::run::u64 dp,
                                         const DirEntry &se, int &rc) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 id = se.id_;
  // 1. Freeze the source entry (a crossing move backs off on it).
  FsReq s;
  s.dir_id_ = sp;
  s.leaf_ = FsLeaf(src);
  s.id_ = id;
  s.flags_ = kRmMarkLeaving | kRmFailBusy | kRmDirOnly;
  FsResp sr;
  CLIO_CO_AWAIT(EntryOp(kShardRemove, s, sr));
  if (sr.rc_ != 0) {
    rc = static_cast<int>(sr.rc_);
    CLIO_CO_RETURN;
  }
  // 2. Never into its own subtree, never under an ancestor that is moving.
  CLIO_CO_AWAIT(CheckMoveTarget(id, dp, rc));
  // 3. The destination: absent, or an empty directory (sealed now, dropped
  //    once replaced).
  clio::run::u64 victim = 0;
  if (rc == 0) {
    DirEntry de;
    int lrc = 0;
    CLIO_CO_AWAIT(LookupEntry(dp, FsLeaf(dst), de, lrc));
    if (lrc == 0 && de.type_ != kFsTypeDir) {
      rc = ENOTDIR;
    } else if (lrc == 0) {
      CLIO_CO_AWAIT(SealDir(de.id_, true, rc));
      if (rc == 0) victim = de.id_;
    } else if (lrc != ENOENT) {
      rc = lrc;
    }
  }
  // 4. Move the one entry. The subtree does not move: its blocks are keyed
  //    by directory id, not path.
  FsResp ir;
  if (rc == 0) {
    FsReq ins;
    ins.dir_id_ = dp;
    ins.leaf_ = FsLeaf(dst);
    ins.id_ = id;
    ins.type_ = kFsTypeDir;
    ins.flags_ = kInsReplace | kInsReplaceDir | kInsFailBusy | kInsNoTagName;
    CLIO_CO_AWAIT(EntryOp(kShardInsert, ins, ir));
    rc = static_cast<int>(ir.rc_);
  }
  if (rc == 0) {
    FsReq pa;
    pa.dir_id_ = id;
    pa.id_ = dp;
    pa.leaf_ = FsLeaf(dst);
    pa.flags_ = 2048u;  // kSetParent
    FsResp par;
    CLIO_CO_AWAIT(CallShard(BlockHome(id, 0), kShardDirAttr, pa, par));
    if (victim != 0) CLIO_CO_AWAIT(DropDir(victim));
    const std::string to =
        clio::cte::core::MakeTagRefName(FsUnpack(dp), FsLeaf(dst));
    if (victim != 0) {
      PublishName(clio::cte::core::TagNameOp::kRemoveName, FsUnpack(victim),
                  to);
    }
    PublishName(clio::cte::core::TagNameOp::kRename, FsUnpack(id),
                clio::cte::core::MakeTagRefName(FsUnpack(sp), FsLeaf(src)),
                to);
  } else if (victim != 0) {
    int urc = 0;
    CLIO_CO_AWAIT(SealDir(victim, false, urc));
  }
  // 5. Finish or undo the source.
  s.flags_ = rc == 0 ? kRmNoTagName : kRmRestore;
  FsResp fr;
  CLIO_CO_AWAIT(EntryOp(kShardRemove, s, fr));
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
    CLIO_FS_RESOLVE(src, se, sp, erc);
    if (erc != 0) {
      rc = erc;
      break;
    }
    CLIO_FS_PARENT(dst, dpe, derc);
    if (derc != 0) {
      rc = derc;
      break;
    }
    is_dir = se.type_ == kFsTypeDir;
    if (is_dir) {
      CLIO_CO_AWAIT(RenameDir(src, dst, sp, dpe.id_, se, rc));
    } else {
      CLIO_CO_AWAIT(RenameFile(src, dst, sp, dpe.id_, se, rc));
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
  CLIO_FS_RESOLVE(target, te, tp, erc);
  (void)tp;
  if (erc != 0) {
    task->return_code_ = erc;
    CLIO_CO_RETURN;
  }
  if (te.type_ == kFsTypeDir) {
    task->return_code_ = EPERM;  // no hard links to directories
    CLIO_CO_RETURN;
  }
  CLIO_FS_PARENT(link, le, lerc);
  if (lerc != 0) {
    task->return_code_ = lerc;
    CLIO_CO_RETURN;
  }
  const clio::run::u64 id = te.id_;
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
  ins.dir_id_ = le.id_;
  ins.leaf_ = FsLeaf(link);
  ins.id_ = id;
  ins.type_ = te.type_;
  ins.flags_ = kInsExcl;
  FsResp ir;
  CLIO_CO_AWAIT(EntryOp(kShardInsert, ins, ir));
  if (ir.rc_ != 0) {
    CLIO_CO_AWAIT(UnlinkInode(id, std::string()));
    task->return_code_ = ir.rc_;
    CLIO_CO_RETURN;
  }
  MirrorRefuse(target);
  MirrorRefuse(FsParentDir(link));
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Symlink(clio::run::shared_ptr<SymlinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  CLIO_FS_PARENT(path, pe, perc);
  if (perc != 0) {
    task->return_code_ = perc;
    CLIO_CO_RETURN;
  }
  FsReq r;
  r.dir_id_ = pe.id_;
  r.dir_ = FsParentDir(path);
  r.leaf_ = FsLeaf(path);
  r.type_ = kFsTypeSymlink;
  r.mode_ = 0777u;
  r.str_ = task->target_.str();
  r.flags_ = kInsExcl | kInsNewInode;
  MirrorRefuse(r.dir_);  // before the name can exist (libfuse's post-op stat)
  FsResp rr;
  CLIO_CO_AWAIT(EntryOp(kShardInsert, r, rr));
  task->return_code_ = rr.rc_;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Readlink(clio::run::shared_ptr<ReadlinkTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const std::string path = FsNormPath(task->path_.str());
  CLIO_FS_RESOLVE(path, ent, par, erc);
  if (erc != 0) {
    task->return_code_ = erc;
    CLIO_CO_RETURN;
  }
  if (ent.type_ != kFsTypeSymlink) {
    task->return_code_ = EINVAL;
    CLIO_CO_RETURN;
  }
  FsResp sr;
  CLIO_CO_AWAIT(StatEntry(par, FsLeaf(path), ent, sr));
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
    CLIO_FS_RESOLVE(pathv, _e, _p, _erc);                                     \
    (void)_p;                                                                 \
    if (_erc != 0) {                                                          \
      rcv = _erc;                                                             \
      break;                                                                  \
    }                                                                         \
    (req).id_ = _e.id_;                                                       \
    FsResp _ar;                                                               \
    if (_e.type_ == kFsTypeDir) {                                             \
      (req).dir_id_ = _e.id_;                                                 \
      (req).flags_ |= kSetCtimeOnly;                                          \
      CLIO_CO_AWAIT(                                                          \
          CallShard(BlockHome(_e.id_, 0), kShardDirAttr, req, _ar));          \
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
  r.flags_ = task->flags_ &
             (kSetAtime | kSetMtime | kSetAtimeNow | kSetMtimeNow);
  if (task->flags_ & kUtimensAccess) {
    r.flags_ = kAccessTouch |
               ((task->flags_ & kUtimensAccessStrict) ? kAccessStrict : 0u);
  }
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
  // Every block of the directory, from this container's cache (a block not
  // cached yet is fetched once from its home, which then keeps it current).
  const std::string dir = FsNormPath(task->path_.str());
  task->entries_ = clio::run::priv::vector<clio::run::priv::string>(CTP_MALLOC);
  task->inos_ = clio::run::priv::vector<clio::run::u64>(CTP_MALLOC);
  CLIO_FS_RESOLVE(dir, de, dpar, rc);
  (void)dpar;
  if (rc == 0 && de.type_ != kFsTypeDir) rc = ENOTDIR;
  std::vector<std::pair<std::string, DirEntry>> listing;
  clio::run::u64 newest = 0;
  if (rc == 0) CLIO_CO_AWAIT(CollectDir(de.id_, &listing, &newest, rc));
  task->entries_.reserve(listing.size());
  task->inos_.reserve(listing.size());
  for (const auto &kv : listing) {
    task->entries_.push_back(
        clio::run::priv::string(CTP_MALLOC, FsJoin(dir, kv.first)));
    task->inos_.push_back(InoFromPacked(kv.second.id_));
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
    CLIO_FS_RESOLVE(pathv, _e, _p, _erc);                                     \
    (void)_p;                                                                 \
    if (_erc != 0) {                                                          \
      xr.rc_ = static_cast<clio::run::u32>(_erc);                             \
      break;                                                                  \
    }                                                                         \
    FsReq _xq;                                                                \
    _xq.id_ = _e.id_;                                                         \
    _xq.type_ = (subop);                                                      \
    _xq.str_ = (namev);                                                       \
    _xq.str2_ = (valuev);                                                     \
    _xq.flags_ = (flagsv) & 0xFFu;                                            \
    clio::run::u32 _owner = InodeOwner(_xq.id_);                              \
    if (_e.type_ == kFsTypeDir) {                                             \
      /* A directory has no inode home: its xattrs are serialized on the */  \
      /* home of its block 0, which every request for it reaches. */         \
      _xq.flags_ |= kXattrNoInode;                                            \
      _owner = BlockHome(_e.id_, 0);                                          \
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





#undef CLIO_FS_LOOKUP
#undef CLIO_FS_RESOLVE
#undef CLIO_FS_PARENT
#undef CLIO_FS_SETATTR
#undef CLIO_FS_XATTR

}  // namespace clio::cte::filesystem

// Define ChiMod entry points (alloc/new/name/destroy) so the runtime's module
// manager can dlopen and instantiate this chimod.
CLIO_TASK_CC(clio::cte::filesystem::Runtime)
