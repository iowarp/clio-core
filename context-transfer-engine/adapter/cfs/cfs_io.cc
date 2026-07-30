/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Implementation of the shared CTE adapter I/O core (see cfs_io.h).
 */
#include "cfs_io.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace clio::cae {

CTP_DEFINE_GLOBAL_PTR_VAR_CC(CfsIo, g_cfs_io);

bool CfsIo::EnsureClient() {
  if (client_ready_) {
    return true;
  }
  client_ready_ = clio::cte::filesystem::CLIO_CFS_CLIENT_INIT();
  if (!client_ready_) {
    HLOG(kError, "CfsIo: failed to initialize the filesystem client");
  }
  return client_ready_;
}

bool CfsIo::QueryGetattr(const std::string &path, bool *exists,
                         clio::run::u64 *size) {
  if (!EnsureClient()) {
    return false;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncGetattr(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    return false;
  }
  *exists = (t->exists_ != 0);
  *size = t->size_;
  return true;
}

bool CfsIo::QuerySize(const std::string &path, clio::run::u64 *size) {
  bool exists = false;
  if (!QueryGetattr(path, &exists, size)) {
    return false;
  }
  if (!exists) {
    *size = 0;
  }
  return true;
}

ssize_t CfsIo::TryReadShm(const std::string &path, clio::run::u64 off,
                          void *buf, size_t count) {
  // Attach lazily, and RETRY when not yet attached, rather than latching the
  // result of one attempt at init. A client can legitimately come up before
  // its pool has been composed (or before the chimod has registered its cache
  // root), and a one-shot attach at init would leave that process on the RPC
  // path forever -- silently, since a disabled cache and a working one look
  // identical from the outside. Retrying costs a scan of a <=32-entry
  // directory, against an RPC that costs ~100 us.
  auto *cfs = CLIO_CFS_CLIENT;
  if (cfs == nullptr || (!cfs->HasShmCache() && !cfs->AttachShmCache())) {
    return -1;
  }
  auto *cte = CLIO_CTE_CLIENT;
  if (cte == nullptr || (!cte->HasShmCache() && !cte->AttachShmCache())) {
    return -1;  // page payloads come from the core cache; both are required
  }

  clio::cte::filesystem::ShmFileRecord rec;
  if (!cfs->TryGetFileRecordShm(path, &rec) || !rec.IsFastPathable()) {
    return -1;
  }

  // Clamp to the LOGICAL size. This is the number the chimod owns and the one
  // POSIX read() semantics are defined against; the tag's physical byte total
  // would be wrong after a sparse write or an ftruncate-grow.
  //
  // Reading a stale (too small) size here cannot produce a short read for a
  // properly ordered caller: the runtime publishes the new size at the end of
  // the Write handler, so it lands before the write() that grew the file
  // returns to its caller.
  if (off >= rec.size_) {
    return 0;  // at or past EOF
  }
  clio::run::u64 want = count;
  if (off + want > rec.size_) {
    want = rec.size_ - off;
  }

  char *dst = static_cast<char *>(buf);
  clio::run::u64 done = 0;
  clio::run::u64 cur = off;
  while (done < want) {
    clio::run::u64 page_off = cur % clio::cte::filesystem::kFsPageSize;
    clio::run::u64 to_read =
        std::min(clio::cte::filesystem::kFsPageSize - page_off, want - done);
    // issue #872: a deferred write to this page may still be in flight; the
    // registry await is one relaxed load when nothing is pending. (The
    // pre-existing mirror-republish lag documented in DoRead applies here
    // exactly as it does to drained WriteTask writes.)
    clio::cte::core::Client::AwaitPendingPuts(
        rec.tag_id_, clio::cte::filesystem::PageName(cur));
    if (!cte->TryReadBlobShm(rec.tag_id_, clio::cte::filesystem::PageName(cur),
                             dst + done, static_cast<size_t>(to_read),
                             static_cast<size_t>(page_off))) {
      // Abandon the WHOLE request rather than mixing sources. A hole reads as
      // zeros and a missing page is indistinguishable from an uncached one, so
      // the only safe reading of a failure here is "let the runtime answer".
      return -1;
    }
    done += to_read;
    cur += to_read;
  }

  // Say so, once, the first time a read is actually served from shared
  // memory. "The fast path is on" is otherwise unobservable from outside --
  // a disabled cache and a working one differ only in latency -- and the
  // failure this whole change was chasing was precisely a fast path that was
  // silently off everywhere. Operators need something to grep for.
  static std::once_flag announced;
  std::call_once(announced, [] {
    HLOG(kInfo,
         "clio-fs: serving reads from shared memory (zero-IPC fast path "
         "active)");
  });
  return static_cast<ssize_t>(done);
}

ssize_t CfsIo::DoRead(clio::run::u64 handle, const std::string &path,
                      clio::run::u64 off, void *buf, size_t count) {
  if (count == 0) {
    return 0;
  }
  // Read-your-own-writes. A queued write is not visible to either path until
  // its task runs -- the SHM mirror still carries the old size and the old
  // page bytes -- so a read that overlaps one must wait for it. This is the
  // per-client pending-write set the #783 design anticipated; it only became
  // necessary once writes stopped blocking.
  //
  // KNOWN GAP (read side, not addressed here): DrainIfOverlap waits for the
  // write's task to COMPLETE, but the runtime republishes the SHM read mirror
  // AFTER the authoritative update, so for a brief window a completed write is
  // durable yet the mirror the fast path reads still shows the prior bytes.
  // A single write outruns this; a burst of rewrites to ONE page keeps the
  // mirror a republish behind, so a fast-path RYOW read mid-burst can return
  // the next-to-last value. fsync/reopen always sees the last. The fix is to
  // gate the fast path on the drained write's placement_gen (bump-and-compare
  // as TryReadShm already does for relocation) and fall back to RPC when the
  // mirror has not caught up; the payload path proper is unaffected.
  DrainIfOverlap(path, off, count);
  // issue #817: try the zero-IPC path first. It either answers completely or
  // declines, and declining costs one hash lookup.
  ssize_t fast = TryReadShm(path, off, buf, count);
  if (fast >= 0) {
    return fast;
  }
  // issue #872: the runtime cannot see this client's Defer registry, so a
  // task-path read must not race in-flight deferred puts for these pages.
  // With the record we await exactly the overlapped pages; without it (no
  // tag known) fall back to a full drain — rare, since a file that was ever
  // deferred-written has a fast-pathable record.
  {
    auto *cfs_cl = CLIO_CFS_CLIENT;
    clio::cte::filesystem::ShmFileRecord rrec;
    if (cfs_cl != nullptr && cfs_cl->TryGetFileRecordShm(path, &rrec) &&
        rrec.IsFastPathable()) {
      AwaitDeferPages(rrec.tag_id_, off, count);
    } else {
      clio::cte::core::Client::AwaitPutsUntilSpace(0);
    }
  }
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
  if (shm.IsNull()) {
    errno = ENOMEM;
    return -1;
  }
  ctp::ipc::ShmPtr<> shm_ptr(shm.shm_);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRead(handle, off, count, shm_ptr);
  t.Wait();
  ssize_t ret;
  if (t->GetReturnCode() == 0) {
    size_t got = static_cast<size_t>(t->bytes_read_);
    if (got > 0) {
      std::memcpy(buf, shm.ptr_, got);
    }
    ret = static_cast<ssize_t>(got);
  } else {
    errno = EIO;
    ret = -1;
  }
  ipc->FreeBuffer(shm);
  return ret;
}

// ---- issue #817: the async write window -----------------------------------

bool CfsIo::AsyncWritesEnabled() {
  // ON by default (issue #817). write(2) hands the bytes to the chimod and
  // returns; fsync/close drain and report.
  //
  // Measured on 128 x 64 KiB overwrite + fsync, five runs each, medians:
  // queued 5.77 ms vs blocking 6.86 ms. A modest ~13% with overlapping
  // ranges -- the honest summary is "no worse, probably a little better",
  // not a headline.
  //
  // A NOTE ON HOW EASY THIS IS TO MEASURE WRONG: the first pass over a fresh
  // file allocates every block and a second pass does not, so timing one mode
  // on a new file and the other on the warmed one compares allocation against
  // overwrite. Doing exactly that produced a confident, entirely spurious
  // "async is 2.3x SLOWER" -- with the two arms differing by 40% even when
  // both ran the SAME code path. The benchmark now warms first and times only
  // overwrites; keep it that way.
  //
  // Set CLIO_CFS_ASYNC_WRITES=0 to fall back to blocking writes.
  static const bool v = [] {
    if (const char *e = std::getenv("CLIO_CFS_ASYNC_WRITES")) {
      return !(std::string(e) == "0" || std::string(e) == "false");
    }
    return true;
  }();
  return v;
}

bool CfsIo::DeferWritesEnabled() {
  // issue #872: route the write PAYLOAD through the CTE Defer registry
  // (AsyncPutBlobDefer pages the caller's buffer client-side, staging the
  // bytes into SHM during write(2)); only the logical-size advance still
  // crosses to the chimod, as a metadata-only AdvanceSizeTask. Requires the
  // #817 SHM record for the file (tag id) — files without a fast-pathable
  // record fall back to the WriteTask path transparently.
  //
  // Consistency: in-process readers are covered exactly (the read paths await
  // the registry's per-page pending puts). Cross-process readers can
  // transiently observe the advanced size before the page bytes land — the
  // AdvanceSizeTask cannot know when the client's puts complete. The
  // single-daemon deployment (one FUSE/interceptor process per node) is
  // unaffected; multi-writer-multi-reader setups should set CLIO_CFS_DEFER=0.
  static const bool v = [] {
    if (const char *e = std::getenv("CLIO_CFS_DEFER")) {
      return !(std::string(e) == "0" || std::string(e) == "false");
    }
    return true;
  }();
  return v;
}

void CfsIo::AwaitDeferPages(const clio::cte::core::TagId &tag,
                            clio::run::u64 off, clio::run::u64 size) {
  clio::run::u64 cur = off;
  const clio::run::u64 end = off + size;
  while (cur < end) {
    clio::cte::core::Client::AwaitPendingPuts(
        tag, clio::cte::filesystem::PageName(cur));
    cur += clio::cte::filesystem::kFsPageSize -
           (cur % clio::cte::filesystem::kFsPageSize);
  }
}

void CfsIo::RetireEntry(PathWrites *pw, PendingWrite &p) {
  if (p.deferred) {
    // Payload first (the registry waits and error-counts the puts), then the
    // metadata advance. Defer-put failures are process-global counters, not
    // attributable to one entry; sample the delta and latch EIO like a failed
    // WriteTask would.
    static std::atomic<clio::run::u64> seen_errs{0};
    AwaitDeferPages(p.tag, p.off, p.size);
    p.meta_fut.Wait();
    clio::run::u64 errs = clio::cte::core::Client::DeferErrorCount();
    clio::run::u64 prev = seen_errs.exchange(errs);
    if ((errs != prev || p.meta_fut->GetReturnCode() != 0) && pw->sticky == 0) {
      pw->sticky = EIO;
    }
    return;
  }
  p.fut.Wait();
  if (p.fut->GetReturnCode() != 0 && pw->sticky == 0) {
    pw->sticky = EIO;
  }
  CLIO_IPC->FreeBuffer(p.buf);
}

// The window is a BACK-PRESSURE bound, not an error condition: a writer that
// reaches it blocks on the oldest outstanding write until it drains, and then
// proceeds. Nothing here can make a write(2) fail.
//
// 64 MiB of staging comes out of the client's SHM allocator (256 MiB by
// default, see IpcManager::IncreaseClientShm), so the two must be sized
// together -- a window larger than the allocator would guarantee the
// allocation failure that DoWrite has to drain its way out of.
clio::run::u64 CfsIo::MaxInflightBytes() {
  static const clio::run::u64 v = [] {
    if (const char *e = std::getenv("CLIO_CFS_WRITE_WINDOW_BYTES")) {
      char *end = nullptr;
      unsigned long long n = std::strtoull(e, &end, 10);
      if (end != e && n > 0) return static_cast<clio::run::u64>(n);
    }
    return static_cast<clio::run::u64>(64ULL * 1024 * 1024);
  }();
  return v;
}

// 8192 tasks. At 4 KiB -- the granularity a POSIX application actually
// writes at -- the count bound would otherwise bind long before the byte
// bound (8192 x 4 KiB = 32 MiB, still under the 64 MiB window), so which of
// the two limits a workload meets now depends on its write size rather than
// on an arbitrarily small task count.
size_t CfsIo::MaxInflightWrites() {
  static const size_t v = [] {
    if (const char *e = std::getenv("CLIO_CFS_WRITE_WINDOW_COUNT")) {
      char *end = nullptr;
      unsigned long long n = std::strtoull(e, &end, 10);
      if (end != e && n > 0) return static_cast<size_t>(n);
    }
    return static_cast<size_t>(8192);
  }();
  return v;
}

std::shared_ptr<PathWrites> CfsIo::PendingFor(const std::string &path,
                                              bool create) {
  std::lock_guard<std::mutex> g(pw_mu_);
  auto it = pending_writes_.find(path);
  if (it != pending_writes_.end()) {
    return it->second;
  }
  if (!create) {
    return nullptr;
  }
  auto pw = std::make_shared<PathWrites>();
  pending_writes_[path] = pw;
  return pw;
}

void CfsIo::ReapAndBound(PathWrites *pw) {
  auto *ipc = CLIO_IPC;
  std::lock_guard<std::mutex> g(pw->mu);

  // Retire anything already finished. Front to back: the queue is in
  // submission order, and a completed write behind an outstanding one still
  // has to keep its slot so `bytes` and the drain order stay coherent.
  // (Deferred entries look at the metadata future; their payload puts are
  // awaited inside RetireEntry, which may block briefly if a put straggles.)
  size_t done = 0;
  while (done < pw->q.size() &&
         (pw->q[done].deferred ? pw->q[done].meta_fut.IsComplete()
                               : pw->q[done].fut.IsComplete())) {
    PendingWrite &p = pw->q[done];
    RetireEntry(pw, p);
    pw->bytes -= p.size;
    ++done;
  }
  if (done > 0) {
    pw->q.erase(pw->q.begin(), pw->q.begin() + static_cast<long>(done));
  }
  (void)ipc;

  // Still over the window: block on the oldest until we are under it. This is
  // the back-pressure -- without it a writer outruns the runtime and the
  // staging buffers grow without bound.
  while (pw->bytes > MaxInflightBytes() || pw->q.size() > MaxInflightWrites()) {
    PendingWrite &p = pw->q.front();
    RetireEntry(pw, p);
    pw->bytes -= p.size;
    pw->q.erase(pw->q.begin());
  }
  // No return value: a latched error belongs to fsync/close, and the caller
  // (DoWrite) must not report or clear it.
}

ctp::ipc::FullPtr<char> CfsIo::AllocateStaging(const std::string &path,
                                               size_t count) {
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
  if (!shm.IsNull()) {
    return shm;
  }

  // The client's SHM allocator is out of room, and the most likely reason is
  // this window: every queued write pins a staging buffer until it retires.
  // Draining gives those buffers straight back, so the write that would have
  // failed with ENOMEM waits instead and then proceeds -- the same trade the
  // window bound already makes, just triggered by the allocator rather than
  // by a count. Try this path first, then every other path's window, because
  // a process writing many files spreads its staging across all of them.
  if (auto pw = PendingFor(path, /*create=*/false)) {
    DrainWindow(pw.get());
    shm = ipc->AllocateBuffer(count);
    if (!shm.IsNull()) {
      return shm;
    }
  }

  std::vector<std::shared_ptr<PathWrites>> all;
  {
    // Copy out under the map lock: DrainWindow waits on tasks, and pw_mu_ is
    // never held across a Wait.
    std::lock_guard<std::mutex> g(pw_mu_);
    all.reserve(pending_writes_.size());
    for (auto &kv : pending_writes_) {
      all.push_back(kv.second);
    }
  }
  for (auto &pw : all) {
    DrainWindow(pw.get());
  }
  // Whatever is left is not ours to free -- the allocator is genuinely out.
  return ipc->AllocateBuffer(count);
}

void CfsIo::DrainWindow(PathWrites *pw) {
  std::lock_guard<std::mutex> g(pw->mu);
  for (auto &p : pw->q) {
    RetireEntry(pw, p);
  }
  pw->q.clear();
  pw->bytes = 0;
}

int CfsIo::DrainPath(const std::string &path) {
  auto pw = PendingFor(path, /*create=*/false);
  if (!pw) {
    return 0;
  }
  DrainWindow(pw.get());
  // PEEK, do not consume. Most callers here (stat, lseek(SEEK_END),
  // ftruncate) drain only to see a coherent size and DISCARD this return
  // value -- if draining also cleared the latch, an intervening stat(2) would
  // silently eat the failure report that fsync/close owes the application.
  // Now that write(2) never reports a latched error itself (see DoWrite),
  // fsync and close are the only places it can surface, so nothing else may
  // consume it.
  std::lock_guard<std::mutex> g(pw->mu);
  return pw->sticky;
}

int CfsIo::DrainPathAndConsume(const std::string &path) {
  auto pw = PendingFor(path, /*create=*/false);
  if (!pw) {
    return 0;
  }
  DrainWindow(pw.get());
  // Report a latched error exactly once, to the fsync/close that asks first.
  std::lock_guard<std::mutex> g(pw->mu);
  int err = pw->sticky;
  pw->sticky = 0;
  return err;
}

void CfsIo::ForgetIfIdle(const std::string &path) {
  std::lock_guard<std::mutex> g(pw_mu_);
  auto it = pending_writes_.find(path);
  if (it == pending_writes_.end()) {
    return;
  }
  // Only drop a window that is genuinely finished: an empty queue with no
  // error still owed to someone. Otherwise a long-lived process (an
  // interceptor lives as long as the application) accumulates one entry per
  // file it ever wrote.
  std::lock_guard<std::mutex> g2(it->second->mu);
  if (it->second->q.empty() && it->second->sticky == 0) {
    pending_writes_.erase(it);
  }
}

void CfsIo::DrainIfOverlap(const std::string &path, clio::run::u64 off,
                           clio::run::u64 len) {
  auto pw = PendingFor(path, /*create=*/false);
  if (!pw) {
    return;
  }
  bool overlap = false;
  {
    std::lock_guard<std::mutex> g(pw->mu);
    for (const auto &p : pw->q) {
      if (off < p.off + p.size && p.off < off + len) {
        overlap = true;
        break;
      }
    }
  }
  if (overlap) {
    // Drain everything, not just the overlapping entries: the queue is
    // ordered, and waiting on a later write while an earlier one is still
    // outstanding would leave the file in a state no single read reflects.
    // The sticky error is deliberately NOT consumed here -- a read must not
    // swallow the report owed to the next write/fsync/close.
    std::lock_guard<std::mutex> g(pw->mu);
    for (auto &p : pw->q) {
      RetireEntry(pw.get(), p);
    }
    pw->q.clear();
    pw->bytes = 0;
  }
}

ssize_t CfsIo::DoWrite(clio::run::u64 handle, const std::string &path,
                       clio::run::u64 off, const void *buf, size_t count,
                       bool sync) {
  if (count == 0) {
    return 0;
  }

  // issue #872: deferred data path. When the file's SHM record gives us the
  // tag, page the caller's buffer straight into AsyncPutBlobDefer — the
  // registry stages the bytes during this call (the buffer is reusable on
  // return), backpressures against real SHM capacity, and gives the read
  // paths per-page read-after-write consistency. Only the logical-size
  // advance still crosses to the chimod, as a metadata-only task parked in
  // the same per-path window so every drain/ordering rule stays intact.
  if (!sync && AsyncWritesEnabled() && DeferWritesEnabled()) {
    auto *cfs_cl = CLIO_CFS_CLIENT;
    auto *cte = CLIO_CTE_CLIENT;
    clio::cte::filesystem::ShmFileRecord rec;
    if (cfs_cl != nullptr && cte != nullptr &&
        (cfs_cl->HasShmCache() || cfs_cl->AttachShmCache()) &&
        cfs_cl->TryGetFileRecordShm(path, &rec) && rec.IsFastPathable()) {
      const char *src = static_cast<const char *>(buf);
      clio::run::u64 done = 0;
      clio::run::u64 cur = off;
      bool ok = true;
      while (done < count) {
        clio::run::u64 page_off = cur % clio::cte::filesystem::kFsPageSize;
        clio::run::u64 to_write = std::min(
            clio::cte::filesystem::kFsPageSize - page_off, count - done);
        static constexpr clio::run::u64 kFsPreallocBytes = 64ull * 1024;
        int rc = cte->AsyncPutBlobDefer(
            rec.tag_id_, clio::cte::filesystem::PageName(cur), page_off,
            to_write, src + done, /*score*/ -1.0f,
            clio::cte::core::Context::Preallocate(kFsPreallocBytes),
            /*flags*/ 0u, clio::run::PoolQuery::Dynamic());
        if (rc != 0) {
          ok = false;
          break;
        }
        done += to_write;
        cur += to_write;
      }
      if (ok) {
        // Same rationale as TryReadShm's announce: an engaged and a
        // silently-fallen-back defer path are otherwise indistinguishable.
        static std::once_flag announced;
        std::call_once(announced, [] {
          HLOG(kInfo,
               "clio-fs: deferred write path active (payload via "
               "AsyncPutBlobDefer, issue #872)");
        });
        auto pw = PendingFor(path, /*create=*/true);
        PendingWrite p;
        p.meta_fut = CLIO_CFS_CLIENT->AsyncAdvanceSize(handle, off + count);
        p.off = off;
        p.size = count;
        p.deferred = true;
        p.tag = rec.tag_id_;
        {
          std::lock_guard<std::mutex> g(pw->mu);
          pw->q.push_back(std::move(p));
          pw->bytes += count;
        }
        ReapAndBound(pw.get());
        return static_cast<ssize_t>(count);
      }
      // A mid-buffer Defer failure (staging exhausted with nothing left to
      // await) falls through to the WriteTask path, whose full-range write
      // makes the partial pages moot (same bytes, rewritten).
    }
  }

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm = AllocateStaging(path, count);
  if (shm.IsNull()) {
    errno = ENOMEM;
    return -1;
  }
  // The copy is mandatory either way: the caller's buffer is theirs to reuse
  // the moment write(2) returns.
  std::memcpy(shm.ptr_, buf, count);
  ctp::ipc::ShmPtr<> shm_ptr(shm.shm_);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncWrite(handle, off, count, shm_ptr);

  if (sync || !AsyncWritesEnabled()) {
    t.Wait();
    ssize_t ret;
    if (t->GetReturnCode() == 0) {
      ret = static_cast<ssize_t>(t->bytes_written_);
    } else {
      errno = EIO;
      ret = -1;
    }
    ipc->FreeBuffer(shm);
    return ret;
  }

  // Asynchronous: hand the task off and return. The client waits for no other
  // write -- not disjoint, not overlapping.
  //
  // Writes are byte-range partial puts, so two writes to DIFFERENT bytes of one
  // page never conflict; only a write to bytes an in-flight write also touches
  // could reorder. And that does not arise in a real workload: a sequential
  // writer's offsets are disjoint, a random writer's are distinct, and an
  // application that issues two write()s to the SAME bytes with no fsync
  // between -- and depends on which wins -- has already lost that race to
  // itself. (The earlier code waited per PAGE, serializing every disjoint 4 KiB
  // write on a shared 1 MiB page: p50 371 us / 1298 IOPS where unwaited writes
  // run p50 4.8 us / 4367 IOPS.)
  //
  // The runtime does NOT order overlapping same-blob writes and a client wait
  // cannot make it: they race the #680 write token, and AsyncWrite's future
  // signals before the write durably commits, so even waiting on the prior
  // write's future leaves a window where an earlier write lands last (measured:
  // 512 rewrites of one range lose the last write ~30% of runs WITH a same-
  // range wait, ~100% WITHOUT). Closing that is a runtime change (#680), out of
  // scope here; an application needing strict last-writer-wins across
  // overlapping unsynced writes must fsync between them or use O_SYNC.
  auto pw = PendingFor(path, /*create=*/true);

  {
    std::lock_guard<std::mutex> g(pw->mu);
    PendingWrite p;
    p.fut = t;
    p.buf = shm;
    p.off = off;
    p.size = count;
    pw->q.push_back(std::move(p));
    pw->bytes += count;
  }
  // Reap + enforce the window AFTER accounting, so this write is included in
  // the bound rather than being the one that overshoots it. This can block --
  // that is the back-pressure -- but it cannot fail.
  ReapAndBound(pw.get());

  // ALWAYS SUCCEED. A queued write that failed leaves its errno latched on
  // the window for fsync/close to report; write(2) does not surface it, and
  // does not consume it. The kernel page cache behaves the same way: the
  // write(2) that hands bytes to writeback returns success, and a writeback
  // failure is reported to fsync/close -- reporting it on an unrelated later
  // write(2) would attribute the failure to bytes that are fine.
  //
  // The one thing this cannot do is make a failure disappear: if the tier is
  // full, the application still learns about it, at the next fsync or close
  // rather than at the write that happened to be in flight.
  return static_cast<ssize_t>(count);
}

int CfsIo::Open(const std::string &raw_path, int flags, int mode) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncOpen(path, static_cast<clio::run::u32>(flags),
                          static_cast<clio::run::u32>(mode));
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  if (t->handle_ == 0) {
    // Plain open of a missing file (chimod honors O_CREAT).
    errno = ENOENT;
    return -1;
  }
  clio::run::u64 size = t->size_;
  // O_TRUNC: drop the logical size to zero.
  if (flags & O_TRUNC) {
    auto tr = cfs->AsyncTruncate(path, 0);
    tr.Wait();
    size = 0;
  }
  std::lock_guard<std::mutex> g(mu_);
  int fd = next_fd_++;
  OpenFile of;
  of.handle = t->handle_;
  of.path = path;
  of.flags = flags;
  of.off = (flags & O_APPEND) ? size : 0;
  fds_[fd] = of;
  return fd;
}

ssize_t CfsIo::Read(int fd, void *buf, size_t count) {
  clio::run::u64 handle, off;
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    off = it->second.off;
    path = it->second.path;
  }
  ssize_t n = DoRead(handle, path, off, buf, count);
  if (n > 0) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it != fds_.end()) {
      it->second.off += static_cast<clio::run::u64>(n);
    }
  }
  return n;
}

ssize_t CfsIo::Write(int fd, const void *buf, size_t count) {
  clio::run::u64 handle, off;
  std::string path;
  int flags;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    off = it->second.off;
    path = it->second.path;
    flags = it->second.flags;
  }
  ssize_t n = DoWrite(handle, path, off, buf, count, IsSyncFd(flags));
  if (n > 0) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it != fds_.end()) {
      it->second.off += static_cast<clio::run::u64>(n);
    }
  }
  return n;
}

ssize_t CfsIo::Pread(int fd, void *buf, size_t count, off_t offset) {
  clio::run::u64 handle;
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    path = it->second.path;
  }
  return DoRead(handle, path, static_cast<clio::run::u64>(offset), buf, count);
}

ssize_t CfsIo::Pwrite(int fd, const void *buf, size_t count, off_t offset) {
  clio::run::u64 handle;
  std::string path;
  int flags;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    path = it->second.path;
    flags = it->second.flags;
  }
  return DoWrite(handle, path, static_cast<clio::run::u64>(offset), buf, count,
                 IsSyncFd(flags));
}

off_t CfsIo::Seek(int fd, off_t offset, int whence) {
  std::string path;
  clio::run::u64 cur;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    path = it->second.path;
    cur = it->second.off;
  }
  clio::run::u64 base = 0;
  switch (whence) {
    case SEEK_SET:
      base = 0;
      break;
    case SEEK_CUR:
      base = cur;
      break;
    case SEEK_END: {
      DrainPath(path);  // EOF must include queued writes (issue #817)
      clio::run::u64 size = 0;
      if (!QuerySize(path, &size)) {
        errno = EIO;
        return -1;
      }
      base = size;
      break;
    }
    default:
      errno = EINVAL;
      return -1;
  }
  off_t newoff = static_cast<off_t>(base) + offset;
  if (newoff < 0) {
    errno = EINVAL;
    return -1;
  }
  std::lock_guard<std::mutex> g(mu_);
  auto it = fds_.find(fd);
  if (it == fds_.end()) {
    errno = EBADF;
    return -1;
  }
  it->second.off = static_cast<clio::run::u64>(newoff);
  return newoff;
}

off_t CfsIo::Tell(int fd) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = fds_.find(fd);
  if (it == fds_.end()) {
    errno = EBADF;
    return -1;
  }
  return static_cast<off_t>(it->second.off);
}

off_t CfsIo::SizeFd(int fd) {
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    path = it->second.path;
  }
  // Queued writes have not advanced the logical size yet (issue #817).
  DrainPath(path);
  clio::run::u64 size = 0;
  if (!QuerySize(path, &size)) {
    return -1;
  }
  return static_cast<off_t>(size);
}

int CfsIo::Sync(int fd) {
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    path = it->second.path;
  }
  // Wait for every queued write on this file and report a latched failure
  // exactly once. Before #817 every write blocked, so this was a no-op; now
  // that write(2) always succeeds, fsync and close are the ONLY places a
  // queued write's failure can reach the application.
  int err = DrainPathAndConsume(path);
  if (err != 0) {
    errno = err;
    return -1;
  }
  return 0;
}

int CfsIo::FtruncateFd(int fd, off_t length) {
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    path = it->second.path;
  }
  return TruncatePath(std::string(kClioPrefix) + path, length);
}

int CfsIo::TruncatePath(const std::string &raw_path, off_t length) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  // Order matters: a queued write that landed AFTER the truncate would undo
  // it, so drain before resizing (issue #817).
  DrainPath(path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncTruncate(path, static_cast<clio::run::u64>(length));
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int CfsIo::RemovePath(const std::string &raw_path) {
  if (!EnsureClient()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  // Drain first: a write still queued against a deleted path would resurrect
  // the tag. Then drop the window so the path's entry does not leak.
  DrainPath(path);
  {
    std::lock_guard<std::mutex> g(pw_mu_);
    pending_writes_.erase(path);
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncUnlink(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int CfsIo::Close(int fd) {
  clio::run::u64 handle;
  std::string path;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    path = it->second.path;
    fds_.erase(it);
  }
  // Drain BEFORE releasing the chimod handle: a queued write names that
  // handle, and the runtime answers EBADF once it is gone (issue #817).
  // close(2) is also the last chance to report a latched write failure, which
  // is why its return code is not simply the Close task's.
  int werr = DrainPathAndConsume(path);
  ForgetIfIdle(path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncClose(handle);
  t.Wait();
  if (werr != 0) {
    errno = werr;
    return -1;
  }
  return (t->GetReturnCode() == 0) ? 0 : -1;
}

}  // namespace clio::cae
