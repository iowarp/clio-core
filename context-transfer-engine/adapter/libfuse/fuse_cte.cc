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

#include <atomic>
#include "fuse_cte.h"

#include <algorithm>
#include <climits>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <cstring>
#include <string>
#include <vector>
#include <cerrno>
#include <chrono>                 // errno
#include <cstdio>                 // snprintf, fprintf
#include <fcntl.h>                // O_CREAT, O_RDWR
#ifdef __linux__
#include <linux/falloc.h>         // FALLOC_FL_* (fallocate mode flags)
#endif

#ifndef _WIN32
#include <sys/statvfs.h>          // struct statvfs (statfs op)
#include <unistd.h>               // read, getuid, getgid
#ifdef __APPLE__
#include <sys/mount.h>            // struct statfs (macFUSE's statfs callback type)
#endif
#endif  // _WIN32
// The mount / process-entry glue (fuse_main, the apptainer --fusemount custom-io
// path, and the fuse_lowlevel/dlsym/mount/uio machinery it needs) now lives in
// fuse_cte_main.cc. This file holds only the operation callbacks.

#include "clio_runtime/clio_runtime.h"
#include "clio_cte/core/content_transfer_engine.h"
#include "clio_cte/core/core_client.h"  // CLIO_CTE_CLIENT + GetCapacity
#include "clio_cte/filesystem/filesystem_client.h"

// Bridge the POSIX type spellings the callbacks use to the concrete types
// each platform's FUSE layer expects in `struct fuse_operations`. On Linux
// (libfuse) the high-level API uses the POSIX types directly; on Windows the
// shim maps them to WinFsp's fuse_* types and supplies getuid/getgid/S_IF*.
#ifdef _WIN32
#include "fuse_win_compat.h"
#else
using cte_stat_t = struct stat;
using cte_off_t = off_t;
using cte_mode_t = mode_t;
using cte_timespec_t = struct timespec;
#ifdef __APPLE__
// macFUSE's statfs callback reports through Darwin's struct statfs, not
// statvfs (the two share f_bsize/f_blocks/f_bfree/f_bavail/f_files/f_ffree;
// f_frsize/f_favail/f_namemax exist only on statvfs — see cte_fuse_statfs).
using cte_statvfs_t = struct statfs;
// Darwin spells the nanosecond stat members st_*timespec. Macro-alias the
// Linux spellings AFTER all system headers so member accesses in the
// callbacks (and in the test TU that #includes this file) resolve.
#define st_atim st_atimespec
#define st_mtim st_mtimespec
#define st_ctim st_ctimespec
#else
using cte_statvfs_t = struct statvfs;
#endif
#endif

using namespace clio::cae::fuse;

// ============================================================================
// Helpers
// ============================================================================
//
// The libfuse adapter is now a thin shim over the filesystem chimod (issue
// #552): every FUSE callback resolves to a path/handle operation on the
// process-wide filesystem client (CLIO_CFS_CLIENT), which owns the path->tag
// mapping, page-blob I/O, and per-file logical-size metadata. Reads/writes
// are synchronous from FUSE's perspective (the client Waits on each op), so
// there is no per-fd pending-write queue here anymore.

namespace {
/** Per-open-file state: the chimod handle + the path it was opened on. */
struct CfsHandle {
  clio::run::u64 fh = 0;
  std::string path;
  // The file's CTE tag (null when unknown): lets data ops drive page blobs
  // through the CTE client directly — writes into the sieve buffer, reads
  // through the tiered RYW/SHM/RPC path — instead of a chimod task per op.
  clio::cte::core::TagId tag = clio::cte::core::TagId::GetNull();
};

// Sieve-direct data path (user directive: writes belong in the sieve
// buffer; only namespace ops should need chimod tasks). CLIO_FUSE_SIEVE=0
// falls back to the chimod Write/Read tasks wholesale.
bool SieveDataEnabled() {
  static const bool v = [] {
    const char *e = getenv("CLIO_FUSE_SIEVE");
    return e == nullptr || *e != '0';
  }();
  return v;
}

// Logical-size high-water for files with UNFLUSHED sieve writes: the chimod
// only learns the size at close (CloseTask::advance_size_), so getattr and
// read-clamping consult this in the meantime. Path-keyed; renames move the
// entry, the closer erases it once the size-carrying close has been sent.
std::mutex g_hw_mtx;
std::unordered_map<std::string, clio::run::u64> g_hiwater;

void HiwaterRaise(const std::string &path, clio::run::u64 end) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  auto &v = g_hiwater[path];
  if (end > v) v = end;
}
clio::run::u64 HiwaterFor(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  auto it = g_hiwater.find(path);
  return it == g_hiwater.end() ? 0 : it->second;
}
void HiwaterErase(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  g_hiwater.erase(path);
}
// Truncate invalidates any unflushed-write extent past the new size; without
// this, getattr's hiwater overlay kept reporting the pre-truncate size.
void HiwaterClamp(const std::string &path, clio::run::u64 size) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  auto it = g_hiwater.find(path);
  if (it == g_hiwater.end()) return;
  if (size == 0) {
    g_hiwater.erase(it);
  } else if (it->second > size) {
    it->second = size;
  }
}
void HiwaterRename(const std::string &from, const std::string &to) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  auto it = g_hiwater.find(from);
  if (it != g_hiwater.end()) {
    clio::run::u64 v = it->second;
    g_hiwater.erase(it);
    auto &dst = g_hiwater[to];
    if (v > dst) dst = v;
  }
}

// Drain every page-blob key of `path`'s file (sieve writes register under
// (tag, page-name), NOT under the cfs FileKey) up to `hiwater` bytes.
void DrainSievePages(const clio::cte::core::TagId &tag,
                     clio::run::u64 hiwater) {
  if (tag.IsNull() || hiwater == 0) return;
  for (clio::run::u64 off = 0; off < hiwater;
       off += clio::cte::filesystem::kFsPageSize) {
    clio::cte::core::Client::AwaitPendingPuts(
        tag, clio::cte::filesystem::PageName(off));
  }
}

CfsHandle *GetHandle(struct fuse_file_info *fi) {
  return reinterpret_cast<CfsHandle *>(fi->fh);
}

// Lazy closer (deferred writes x async close): a CloseTask that reaches the
// chimod while the file's deferred WriteTasks are still in flight races them
// — scheduler order decides whether a write lands on a closed handle and is
// DROPPED (this corrupted git index-pack output). release() therefore hands
// write-pending handles to this thread, which drains the file's writes and
// only then sends the (fire-and-forget) close. Handles with nothing pending
// skip the queue and close detached immediately — the common case for
// read-only opens.
// ==========================================================================
// Batched, sieve-flushed CREATION (user directive: creation joins the data
// sieve). An O_EXCL create mints the file's TagId client-side (top-bit minor
// partition — the server generator never mints there), records the creation,
// and returns immediately with a stable inode; the closer thread flushes
// accumulated creations every tick as ONE MultiCreateTask whose tag chains
// adopt the minted ids. Safe under O_EXCL because the kernel's just-finished
// authoritative negative LOOKUP precedes every CREATE it sends, and the
// kernel serializes same-path creation.
// ==========================================================================
struct PendingCreate {
  clio::cte::core::TagId tag;
  clio::run::u32 mode = 0644;
  clio::run::u64 ctime_ns = 0;  // synthesized attrs until the flush lands
};
std::mutex g_pc_mtx;
std::mutex g_flush_mtx;  // held across an ENTIRE flush (swap + Wait + retire)
std::unordered_map<std::string, PendingCreate> g_pending_creates;
std::vector<clio::cte::filesystem::MultiCreateEnt> g_create_queue;

clio::cte::core::TagId MintTagId() {
  static std::atomic<clio::run::u32> counter{1};
  auto *ipc = CLIO_IPC;
  clio::run::u32 node = ipc != nullptr ? ipc->GetNodeId() : 1;
  return clio::cte::core::TagId(
      node, 0x80000000u | counter.fetch_add(1, std::memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// Hard-link groups (issue #1007 follow-up): the kernel does NOT invalidate a
// link TARGET's cached attrs when a sibling name is linked or unlinked, so a
// 1 s attr TTL served stale nlink/ctime to a stat right after ln/rm
// (generic/002, generic/236). Links made through this mount are tracked
// here; link and unlink invalidate every sibling via fuse_invalidate_path.
// Links made by OTHER mounts stay subject to plain TTL staleness.
// ---------------------------------------------------------------------------
std::mutex g_lg_mtx;
std::unordered_map<std::string, std::shared_ptr<std::set<std::string>>>
    g_link_groups;

void LinkGroupJoin(const std::string &a, const std::string &b) {
  std::lock_guard<std::mutex> lk(g_lg_mtx);
  auto ita = g_link_groups.find(a);
  auto itb = g_link_groups.find(b);
  std::shared_ptr<std::set<std::string>> g;
  if (ita != g_link_groups.end() && itb != g_link_groups.end()) {
    g = ita->second;
    for (const auto &m : *itb->second) g->insert(m);
  } else if (ita != g_link_groups.end()) {
    g = ita->second;
  } else if (itb != g_link_groups.end()) {
    g = itb->second;
  } else {
    g = std::make_shared<std::set<std::string>>();
  }
  g->insert(a);
  g->insert(b);
  for (const auto &m : *g) g_link_groups[m] = g;
}

// Remove `p` from its group; returns the SIBLINGS whose attrs went stale.
std::vector<std::string> LinkGroupDrop(const std::string &p) {
  std::lock_guard<std::mutex> lk(g_lg_mtx);
  std::vector<std::string> out;
  auto it = g_link_groups.find(p);
  if (it == g_link_groups.end()) return out;
  auto g = it->second;
  g->erase(p);
  g_link_groups.erase(it);
  out.assign(g->begin(), g->end());
  if (g->size() <= 1) {
    for (const auto &m : *g) g_link_groups.erase(m);
  }
  return out;
}

void LinkGroupRename(const std::string &from, const std::string &to) {
  std::lock_guard<std::mutex> lk(g_lg_mtx);
  auto it = g_link_groups.find(from);
  if (it == g_link_groups.end()) return;
  auto g = it->second;
  g->erase(from);
  g->insert(to);
  g_link_groups.erase(it);
  g_link_groups[to] = g;
}

// True only once a real mount is serving requests. See InvalidatePath.
std::atomic<bool> g_fuse_session_live{false};

// Invalidate a path's kernel entry+attr (and data) cache. Safe from request
// context in write-through mode (no writeback pages to deadlock on).
//
// fuse_get_context() is only defined INSIDE a request handler. The handlers in
// this file are also called directly -- by test_fuse_ops, which mounts nothing
// -- and there libfuse has never populated a context. On macFUSE that returns
// a pointer to uninitialised storage, so `cx->fuse` is non-null GARBAGE, the
// null guard below passes, and fuse_invalidate_path dereferences it:
//
//   x0 = 0xee65bf959a3ab9e8                       <- wild mutex pointer
//   lr = libfuse3.4.dylib`fuse_invalidate_path+72
//   pc = libsystem_pthread.dylib`pthread_mutex_lock+12   EXC_BAD_ACCESS
//
// That is the fuse_ops SEGFAULT that has failed every macOS adapters run since
// at least 2026-08-21. It hit the "hard link" case first because link/unlink
// are the operations that invalidate siblings (see the note above cte_fuse_link)
// -- the earlier cases never reach here. The address differed every run and
// matched neither MallocScribble nor MallocPreScribble, which is what ruled out
// a freed-heap read and pointed at never-initialised memory.
//
// A null check cannot fix this: the pointer is garbage, not null. Gate on a
// flag that is set only by the real mount path instead, so a direct call with
// no session is a no-op rather than a wild dereference. Linux happened to
// survive it; that is luck, not correctness.
void InvalidatePath(const std::string &p) {
  if (!g_fuse_session_live.load(std::memory_order_acquire)) {
    return;
  }
  struct fuse_context *cx = fuse_get_context();
  if (cx != nullptr && cx->fuse != nullptr) {
    fuse_invalidate_path(cx->fuse, p.c_str());
  }
}

// ---------------------------------------------------------------------------
// Detached-utimens times overlay: the hook fires the stamp and returns, so a
// stat racing the in-flight task read the OLD times (generic/221 fstat right
// after futimens). The hook records the requested values here; getattr
// overlays them until the server view catches up (entries expire after 2 s —
// far beyond any task round trip).
// ---------------------------------------------------------------------------
struct TimesOverlay {
  clio::run::u64 atime_ns = 0;  // 0 = not set by the stamp
  clio::run::u64 mtime_ns = 0;
  clio::run::u64 ctime_ns = 0;  // always set (utimens advances ctime)
  std::chrono::steady_clock::time_point at;
};
std::mutex g_to_mtx;
std::unordered_map<std::string, TimesOverlay> g_times_overlay;

void TimesOverlaySet(const std::string &p, clio::run::u64 a,
                     clio::run::u64 m, clio::run::u64 c) {
  std::lock_guard<std::mutex> lk(g_to_mtx);
  g_times_overlay[p] = TimesOverlay{a, m, c, std::chrono::steady_clock::now()};
}

bool TimesOverlayGet(const std::string &p, TimesOverlay *out) {
  std::lock_guard<std::mutex> lk(g_to_mtx);
  auto it = g_times_overlay.find(p);
  if (it == g_times_overlay.end()) return false;
  if (std::chrono::steady_clock::now() - it->second.at >
      std::chrono::seconds(2)) {
    g_times_overlay.erase(it);
    return false;
  }
  *out = it->second;
  return true;
}

bool PendingCreateLookup(const std::string &path, PendingCreate *out) {
  std::lock_guard<std::mutex> lk(g_pc_mtx);
  auto it = g_pending_creates.find(path);
  if (it == g_pending_creates.end()) return false;
  *out = it->second;
  return true;
}

// Ship every queued creation as one MultiCreateTask and WAIT for it, then
// retire the pending entries (their mirror records now answer getattr).
void FlushCreates() {
  // g_flush_mtx spans swap -> Wait -> retire. Without it, a barrier caller
  // (rename's EnsureCreated) that arrives mid-flight sees an EMPTY queue and
  // sails on while the MultiCreate is still airborne; the late-landing create
  // then resurrects the just-renamed source path as a ghost (EEXIST on the
  // next O_EXCL create) and the rename misses the tag (zeros on read-back).
  std::lock_guard<std::mutex> fl(g_flush_mtx);
  std::vector<clio::cte::filesystem::MultiCreateEnt> batch;
  {
    std::lock_guard<std::mutex> lk(g_pc_mtx);
    if (g_create_queue.empty()) return;
    batch.swap(g_create_queue);
  }
  auto t = CLIO_CFS_CLIENT->AsyncMultiCreate(
      clio::cte::filesystem::EncodeMultiCreate(batch));
  t.Wait();
  // Retire ONLY after the server owns the metadata: a getattr between
  // retirement and the flush landing would miss both sources. Retire is
  // TAG-MATCHED: if the same path was re-created while this batch flew,
  // the registry holds the NEWER creation, which must survive.
  std::lock_guard<std::mutex> lk(g_pc_mtx);
  for (const auto &e : batch) {
    auto it = g_pending_creates.find(e.path_);
    if (it != g_pending_creates.end() &&
        ((static_cast<clio::run::u64>(it->second.tag.major_) << 32) |
         static_cast<clio::run::u64>(it->second.tag.minor_)) == e.tag_packed_) {
      g_pending_creates.erase(it);
    }
  }
}

// Barrier for ops that need the path authoritative server-side (rename,
// unlink, task-path getattr of a pending path): flush now, synchronously.
void EnsureCreated(const std::string &path) {
  // Loop: FlushCreates serializes behind any in-flight flush (g_flush_mtx),
  // so one more pass after it returns proves the entry retired — or flushes
  // it ourselves if it was re-queued.
  for (;;) {
    {
      std::lock_guard<std::mutex> lk(g_pc_mtx);
      if (g_pending_creates.find(path) == g_pending_creates.end()) return;
    }
    FlushCreates();
  }
}

struct PendingClose {
  enum Kind { kClose, kUtimens };
  Kind kind = kClose;
  clio::run::u64 fh = 0;
  std::string path;
  clio::run::u64 atime_ns = 0;  // kUtimens payload
  clio::run::u64 mtime_ns = 0;
  clio::run::u32 flags = 0;
  clio::cte::core::TagId tag = clio::cte::core::TagId::GetNull();  // kClose
  clio::run::u64 hiwater = 0;  // kClose: size to advance to (0 = none)
};
std::mutex g_closer_mtx;
std::condition_variable g_closer_cv;
std::deque<PendingClose> g_closer_q;
std::thread g_closer;
bool g_closer_started = false;
bool g_closer_stop = false;
bool g_closer_busy = false;  // CloserMain is executing a popped entry

bool CreateQueueNonEmpty() {
  std::lock_guard<std::mutex> lk(g_pc_mtx);
  return !g_create_queue.empty();
}

void CloserMain() {
#ifdef __linux__
  pthread_setname_np(pthread_self(), "cfs-closer");
#endif
  std::unique_lock<std::mutex> lk(g_closer_mtx);
  while (!g_closer_stop) {
    if (g_closer_q.empty()) {
      if (CreateQueueNonEmpty()) {
        // Tick-flush creations even when no closes are queued (the create
        // sieve's analog of the page flusher's 500 us cadence).
        g_closer_cv.wait_for(lk, std::chrono::microseconds(500),
                             [] { return g_closer_stop; });
        if (g_closer_stop) break;
        lk.unlock();
        FlushCreates();
        lk.lock();
        continue;
      }
      g_closer_cv.wait_for(lk, std::chrono::milliseconds(50), [] {
        return g_closer_stop || !g_closer_q.empty();
      });
      continue;
    }
    PendingClose pc = std::move(g_closer_q.front());
    g_closer_q.pop_front();
    g_closer_busy = true;
    lk.unlock();

    // Order the metadata op AFTER the file's in-flight writes land. A
    // fire-and-forget task racing them is how a detached close dropped
    // writes on a dead handle, and how a detached utimens republished the
    // mirror with a size read BEFORE a concurrent write advanced it —
    // regressing the published size and truncating subsequent reads
    // (git's .git/config came back garbage).
    clio::cte::core::Client::DeferAwaitKey(
        clio::cte::core::Client::DeferKeyHashName(pc.path));
    if (pc.kind == PendingClose::kClose) {
      DrainSievePages(pc.tag, pc.hiwater);
      if (pc.fh == 0) {
        // Minted-create file: no server handle. The creation must land
        // before the size push, and the push is TAG-VERIFIED: a FUSE
        // release can be delivered after the app already renamed the path
        // (release is asynchronous), and a path-keyed truncate then
        // RESURRECTS the old name as a ghost file — cfs Truncate
        // deliberately materializes missing paths — which a later rename
        // resolves instead of the real file (observed as git's config
        // reading zeros under a ghost inode). On mismatch the hiwater
        // overlay simply stays authoritative for the renamed name.
        EnsureCreated(pc.path);
        if (pc.hiwater != 0) {
          auto t = CLIO_CFS_CLIENT->AsyncAdvanceSize(
              (static_cast<clio::run::u64>(pc.tag.major_) << 32) |
                  static_cast<clio::run::u64>(pc.tag.minor_),
              pc.hiwater);
          t.Wait();
          HiwaterErase(pc.path);
        }
      } else if (pc.hiwater != 0) {
        // AWAITED: the hiwater entry may only be dropped once the chimod
        // has durably adopted the size, or a getattr in the gap would read
        // the stale pre-close size.
        auto t = CLIO_CFS_CLIENT->AsyncClose(pc.fh, pc.hiwater);
        t.Wait();
        HiwaterErase(pc.path);
      } else {
        CLIO_CFS_CLIENT->AsyncCloseDetached(pc.fh);
      }
    } else {
      EnsureCreated(pc.path);
      CLIO_CFS_CLIENT->AsyncUtimensDetached(pc.path, pc.atime_ns, pc.mtime_ns,
                                            pc.flags);
    }
    lk.lock();
    g_closer_busy = false;
    g_closer_cv.notify_all();
  }
}

// Per-file direct_io for WRITE-ONLY opens: in write-through mode the page
// cache hands us ONE 4 KiB FUSE WRITE per dirtied page (a clone's 2.1 GB of
// writes became ~525k ops); direct_io passes each write() through at its
// full size instead. Scoped to O_WRONLY because such fds cannot be mmap'd
// (the #597 direct_io+mmap hazard needs a readable fd) and the page cache
// keeps serving all read opens. Write-once-then-rename creators — compilers,
// git, archivers — are exactly this shape. CLIO_FUSE_DIRECT_WRONLY=0
// disables (a reader concurrently caching a file ANOTHER fd is direct-io
// writing can read stale until reopen; the write-once pattern never does).
bool DirectWronlyEnabled() {
  static const bool v = [] {
    const char *e = getenv("CLIO_FUSE_DIRECT_WRONLY");
    return e == nullptr || *e != '0';
  }();
  return v;
}

void MaybeDirectIo(struct fuse_file_info *fi) {
  if (DirectWronlyEnabled() && (fi->flags & O_ACCMODE) == O_WRONLY) {
    fi->direct_io = 1;
  }
}

// Synchronous barrier: process EVERY queued closer entry inline. Rename and
// unlink call this — they're rare, and a queued entry executing after the
// path moves is how every ghost/EEXIST class in the lock-cycle repro arose.
void CloserBarrier() {
  std::deque<PendingClose> q;
  {
    // Also wait out an entry CloserMain already popped: it is invisible to
    // the swap but still mutating this path's server state.
    std::unique_lock<std::mutex> lk(g_closer_mtx);
    g_closer_cv.wait(lk, [] { return !g_closer_busy; });
    q.swap(g_closer_q);
  }
  for (auto &pc : q) {
    clio::cte::core::Client::DeferAwaitKey(
        clio::cte::core::Client::DeferKeyHashName(pc.path));
    if (pc.kind == PendingClose::kClose) {
      DrainSievePages(pc.tag, pc.hiwater);
      if (pc.fh == 0) {
        EnsureCreated(pc.path);
        if (pc.hiwater != 0) {
          auto t = CLIO_CFS_CLIENT->AsyncAdvanceSize(
              (static_cast<clio::run::u64>(pc.tag.major_) << 32) |
                  static_cast<clio::run::u64>(pc.tag.minor_),
              pc.hiwater);
          t.Wait();
          HiwaterErase(pc.path);
        }
      } else {
        if (pc.hiwater != 0) {
          auto t = CLIO_CFS_CLIENT->AsyncClose(pc.fh, pc.hiwater);
          t.Wait();
          HiwaterErase(pc.path);
        } else {
          CLIO_CFS_CLIENT->AsyncCloseDetached(pc.fh);
        }
      }
    } else {
      EnsureCreated(pc.path);
      CLIO_CFS_CLIENT->AsyncUtimensDetached(pc.path, pc.atime_ns, pc.mtime_ns,
                                            pc.flags);
    }
  }
}

void EnqueueDrainOrdered(PendingClose pc) {
  {
    std::lock_guard<std::mutex> lk(g_closer_mtx);
    if (!g_closer_started) {
      g_closer_started = true;
      g_closer = std::thread(CloserMain);
      g_closer.detach();  // process-lifetime thread; the mount owns it
    }
    g_closer_q.push_back(std::move(pc));
  }
  g_closer_cv.notify_one();
}

void EnqueueClose(clio::run::u64 fh, std::string path) {
  PendingClose pc;
  pc.kind = PendingClose::kClose;
  pc.fh = fh;
  pc.path = std::move(path);
  EnqueueDrainOrdered(std::move(pc));
}
}  // namespace

// Announce that a real mount is about to serve requests. Only after this is
// fuse_get_context() meaningful; see InvalidatePath for what happens when it
// is trusted without a session.
void cte_fuse_mark_session_live() {
  g_fuse_session_live.store(true, std::memory_order_release);
}

// ============================================================================
// FUSE lifecycle
// ============================================================================

static void *cte_fuse_init(struct fuse_conn_info *conn,
                           struct fuse_config *cfg) {
  (void)conn;
  // Trust the inode numbers we report (st_ino in getattr, d_ino in readdir),
  // both derived from the tag id, instead of letting FUSE auto-generate them.
  // This makes stat and readdir agree on d_ino/st_ino (generic/637), and gives
  // hard-link aliases (which share a TagId) the same inode.
  cfg->use_ino = 1;
  // Keep the kernel page cache for file data (direct_io OFF). mmap on a FUSE
  // file is served generically by the page cache — there is no .mmap callback
  // in the high-level API; the kernel faults mapped pages through cte_fuse_read
  // and flushes dirty pages through cte_fuse_write. direct_io bypasses the page
  // cache, so the kernel returns ENODEV ("No such device") for any mmap (issue
  // #597). Writes stay write-through (no FUSE_CAP_WRITEBACK_CACHE), so each
  // write() still reaches the chimod synchronously and the exact logical size
  // is preserved; only mmap dirty pages flush lazily, which is inherent to mmap.
  cfg->direct_io = 0;
  // Kernel attribute/entry caching, default 1 second (what sshfs/NFS-style
  // filesystems ship). A zero-TTL cache sends EVERY path-component lookup and
  // EVERY stat to the chimod as its own round trip — measured ~0.3-0.8 ms
  // each, which made a kernel-tree checkout (95k files, deep paths) 20-30x
  // slower than ext4: the lookup storm alone dominated the clone.
  //
  // Coherence trade, stated plainly: metadata changed WITHOUT this FUSE
  // process seeing it (another mount/process on the same store, or nlink
  // after a hard link) can read stale for up to the TTL. Workloads that need
  // strict coherence run with CLIO_FUSE_ATTR_CACHE_S=0, which restores the
  // old every-op-goes-to-the-chimod behavior exactly.
  double attr_ttl = 1.0;
  if (const char *ttl_env = getenv("CLIO_FUSE_ATTR_CACHE_S")) {
    if (*ttl_env != '\0') attr_ttl = atof(ttl_env);
  }
  cfg->attr_timeout = attr_ttl;
  cfg->entry_timeout = attr_ttl;
  cfg->negative_timeout = attr_ttl;
  // Writeback cache: OFF by default. Enabling it batches write()s in the
  // kernel page cache, but the kernel then requires the filesystem to be a
  // bystander on file size — and the chimod's published size lags dirty
  // pages, so the kernel's open-time revalidation truncated its view of
  // files whose writes had not flushed yet. Measured: git clone corrupted
  // its pack ("invalid index-pack output") with writeback on, and completed
  // clean with it off, all other optimizations unchanged. Opt in with
  // CLIO_FUSE_WRITEBACK=1 only for single-writer workloads that fsync.
  bool writeback = false;
  if (const char *wb_env = getenv("CLIO_FUSE_WRITEBACK")) {
    writeback = (*wb_env == '1');
  }
  if (writeback && (conn->capable & FUSE_CAP_WRITEBACK_CACHE)) {
    conn->want |= FUSE_CAP_WRITEBACK_CACHE;
  }
  // Let the kernel clear suid/sgid/caps itself instead of asking us: without
  // this it sends a GETXATTR("security.capability") before EVERY write-out —
  // measured as exactly one extra round trip per WRITE (71k of 149k ops in a
  // clone's pack phase were these probes).
#ifdef FUSE_CAP_HANDLE_KILLPRIV_V2
  if (conn->capable & FUSE_CAP_HANDLE_KILLPRIV_V2) {
    conn->want |= FUSE_CAP_HANDLE_KILLPRIV_V2;
  }
#endif
  // Bigger transfers per op where the kernel allows it.
  if (conn->max_write < (1u << 20)) {
    conn->max_write = 1u << 20;
  }

  // This adapter always creates directories explicitly (mkdir plants the
  // marker), so the chimod's implicit-dir child scan — the costliest part
  // of a negative lookup — is provably unnecessary here. Opt out before the
  // embedded runtime starts. Respect an explicit user setting.
  if (getenv("CLIO_CFS_NO_IMPLICIT_DIRS") == nullptr) {
#ifndef _WIN32
    setenv("CLIO_CFS_NO_IMPLICIT_DIRS", "1", 0);
#else
    _putenv_s("CLIO_CFS_NO_IMPLICIT_DIRS", "1");
#endif
  }

  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (!success) {
    fprintf(stderr, "ERROR: CLIO_INIT failed\n");
    return nullptr;
  }
  // Create-or-bind the filesystem chimod pool (which also brings up the CTE
  // core pool it sits over). Every FUSE op below routes through CLIO_CFS_CLIENT.
  if (!clio::cte::filesystem::CLIO_CFS_CLIENT_INIT()) {
    fprintf(stderr, "ERROR: filesystem client init failed\n");
    return nullptr;
  }
  // Bind the CTE core client to the same clio_cte_core pool (idempotent) so
  // statfs can query real capacity via GetCapacity.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "WARNING: CTE core client init failed; statfs capacity=0\n");
  }
  return nullptr;
}

#ifndef _WIN32
// Client bootstrap for alternate front ends (the low-level adapter): same
// sequence as cte_fuse_init's tail. Safe to call once from any init hook.
int cte_fuse_bootstrap_clients() {
  if (getenv("CLIO_CFS_NO_IMPLICIT_DIRS") == nullptr) {
    setenv("CLIO_CFS_NO_IMPLICIT_DIRS", "1", 0);
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    fprintf(stderr, "ERROR: CLIO_INIT failed\n");
    return -1;
  }
  if (!clio::cte::filesystem::CLIO_CFS_CLIENT_INIT()) {
    fprintf(stderr, "ERROR: filesystem client init failed\n");
    return -1;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "WARNING: CTE core client init failed; statfs capacity=0\n");
  }
  return 0;
}
#endif  // !_WIN32

static void cte_fuse_destroy(void *private_data) {
  (void)private_data;
  clio::run::CLIO_RUNTIME_FINALIZE();
}

// ============================================================================
// Metadata
// ============================================================================

// Decode a tag timestamp (stored as the two's-complement bits of an i64
// nanoseconds-since-epoch value in a u64 field) into a POSIX timespec. Using
// signed floor division makes pre-epoch (negative) times round-trip correctly
// — an unsigned divide turns e.g. Jan 1 1960 into a huge positive year
// (generic/258). For normal post-epoch times the value and result are
// identical to the old unsigned path (remainder is non-negative). tv_nsec is
// always normalized into [0, 1e9).
// NsecT is templated so this binds to the platform's timespec tv_nsec type:
// `long` on Linux/libfuse, `int64_t` on Windows/WinFsp (struct fuse_timespec).
template <typename NsecT>
static inline void NsBitsToTimespec(clio::run::u64 bits, time_t &sec,
                                    NsecT &nsec) {
  int64_t ns = static_cast<int64_t>(bits);
  int64_t s = ns / 1000000000LL;
  int64_t rem = ns % 1000000000LL;
  if (rem < 0) {
    s -= 1;
    rem += 1000000000LL;
  }
  sec = static_cast<time_t>(s);
  nsec = static_cast<NsecT>(rem);
}

// ---------------------------------------------------------------------------
// Windows compatibility: the cfs SYNC fast-path helpers (Read/Write/Flush/
// GetAttr) live in filesystem_client.h's Linux-only region. MSVC builds
// (only the wheels pipeline compiles this TU with WinFsp) route through the
// awaited async tasks — slower, but the SHM fast paths never existed there.
// ---------------------------------------------------------------------------
static inline int CfsFlushCompat(clio::cte::filesystem::Client *cfs,
                                 const std::string &p) {
#ifndef _WIN32
  return cfs->Flush(p);
#else
  (void)cfs;
  (void)p;
  return 0;  // write-behind is Linux-only; nothing buffered client-side
#endif
}

static inline void CfsGetAttrCompat(clio::cte::filesystem::Client *cfs,
                                    const std::string &p, bool *exists,
                                    clio::run::u64 *size) {
#ifndef _WIN32
  cfs->GetAttr(p, exists, size);
#else
  auto t = cfs->AsyncGetattr(p);
  t.Wait();
  *exists = (t->GetReturnCode() == 0 && t->exists_ != 0);
  *size = *exists ? t->size_ : 0;
#endif
}

static inline ssize_t CfsReadCompat(clio::cte::filesystem::Client *cfs,
                                    clio::run::u64 handle,
                                    const std::string &p, clio::run::u64 off,
                                    char *buf, size_t size) {
#ifndef _WIN32
  return cfs->Read(handle, p, off, buf, size);
#else
  (void)p;
  auto *ipc = CLIO_CPU_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(size);
  if (shm.IsNull()) return -1;
  auto t = cfs->AsyncRead(handle, off, size, shm.shm_.template Cast<void>());
  t.Wait();
  ssize_t got = t->GetReturnCode() == 0
                    ? static_cast<ssize_t>(t->bytes_read_)
                    : -1;
  if (got > 0) std::memcpy(buf, shm.ptr_, static_cast<size_t>(got));
  ipc->FreeBuffer(shm);
  return got;
#endif
}

static inline ssize_t CfsWriteCompat(clio::cte::filesystem::Client *cfs,
                                     clio::run::u64 handle,
                                     const std::string &p, clio::run::u64 off,
                                     const char *buf, size_t size) {
#ifndef _WIN32
  return cfs->Write(handle, p, off, buf, size);
#else
  (void)p;
  auto *ipc = CLIO_CPU_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(size);
  if (shm.IsNull()) return -1;
  std::memcpy(shm.ptr_, buf, size);
  auto t = cfs->AsyncWrite(handle, off, size, shm.shm_.template Cast<void>());
  t.Wait();
  ssize_t wrote = t->GetReturnCode() == 0
                      ? static_cast<ssize_t>(t->bytes_written_)
                      : -1;
  ipc->FreeBuffer(shm);
  return wrote;
#endif
}

static int cte_fuse_getattr_stat_inner(const char *path, cte_stat_t *stbuf,
                                 struct fuse_file_info *fi) {
  (void)fi;
  memset(stbuf, 0, sizeof(*stbuf));

  std::string p(path);

  // Root is always a directory.
  if (p == "/") {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
    stbuf->st_ino = 1;  // fixed root inode
    stbuf->st_uid = getuid();
    stbuf->st_gid = getgid();
    return 0;
  }

  // PENDING-CREATE stat: a minted file whose MultiCreate has not flushed
  // yet has no record anywhere server-side; its identity lives here.
  {
    PendingCreate pc;
    if (PendingCreateLookup(p, &pc)) {
      stbuf->st_uid = getuid();
      stbuf->st_gid = getgid();
      stbuf->st_ino = static_cast<ino_t>(
          (static_cast<clio::run::u64>(pc.tag.major_) << 32) |
          static_cast<clio::run::u64>(pc.tag.minor_));
      stbuf->st_mode = S_IFREG | (pc.mode & 07777u);
      stbuf->st_nlink = 1;
      stbuf->st_size = static_cast<cte_off_t>(HiwaterFor(p));
      stbuf->st_blksize = static_cast<decltype(stbuf->st_blksize)>(4096);
      stbuf->st_blocks = static_cast<decltype(stbuf->st_blocks)>(
          (static_cast<uint64_t>(stbuf->st_size) + 511) / 512);
      NsBitsToTimespec(pc.ctime_ns, stbuf->st_ctim.tv_sec,
                       stbuf->st_ctim.tv_nsec);
      NsBitsToTimespec(pc.ctime_ns, stbuf->st_mtim.tv_sec,
                       stbuf->st_mtim.tv_nsec);
      NsBitsToTimespec(pc.ctime_ns, stbuf->st_atim.tv_sec,
                       stbuf->st_atim.tv_nsec);
      return 0;
    }
  }

  // MIRROR-FIRST stat (user directive): build the full stat from the SHM
  // mirrors — file record (identity, size, overrides) + tag record (natural
  // timestamps) — with no task at all. Records that are absent, refused
  // (hardlink targets, pending appends), or missing their tag record fall
  // through to the chimod task; a tombstone answers ENOENT immediately.
  auto *cfs = CLIO_CFS_CLIENT;
  {
    auto *cte0 = CLIO_CTE_CLIENT;
    clio::cte::filesystem::ShmFileRecord mrec;
    // TryGetFileRecordShm does NOT self-attach: without this the whole
    // mirror-first path silently returns false on every call until some
    // OTHER path (the read fast path) attaches — which a checkout-shaped
    // workload never does early. Attach is one-time and cheap to re-check.
    if (cfs != nullptr && cte0 != nullptr &&
        (cfs->HasShmCache() || cfs->AttachShmCache()) &&
        cfs->TryGetFileRecordShm(p, &mrec)) {
      if (!mrec.Exists()) {
        return -ENOENT;  // tombstone: authoritative absence, no task
      }
      if (!(mrec.flags_ & (clio::cte::filesystem::kShmFileNoFastPath |
                           clio::cte::filesystem::kShmFilePendingAppend |
                           clio::cte::filesystem::kShmFileIsDir))) {
        clio::cte::core::ShmTagRecord trec;
        clio::cte::core::TagId mtag(mrec.tag_id_.major_, mrec.tag_id_.minor_);
        if (cte0->TryGetTagRecordShm(mtag, &trec)) {
          stbuf->st_uid = (mrec.uid_ != clio::cte::filesystem::kShmFileNoOverride)
                              ? static_cast<uid_t>(mrec.uid_) : getuid();
          stbuf->st_gid = (mrec.gid_ != clio::cte::filesystem::kShmFileNoOverride)
                              ? static_cast<gid_t>(mrec.gid_) : getgid();
          stbuf->st_ino = static_cast<ino_t>(mrec.ino_);
          const bool m_have_mode =
              (mrec.mode_ != clio::cte::filesystem::kShmFileNoOverride);
          const unsigned int m_perm = mrec.mode_ & 07777u;
          if (mrec.IsDir()) {
            stbuf->st_mode = S_IFDIR | (m_have_mode ? m_perm : 0755u);
            stbuf->st_nlink = 2;
            stbuf->st_size = 0;
          } else {
            stbuf->st_mode = S_IFREG | (m_have_mode ? m_perm : 0644u);
            stbuf->st_nlink = 1;  // hardlinked records are refused above
            clio::run::u64 sz = mrec.size_;
            clio::run::u64 pend = clio::cte::core::Client::DeferMaxPendingEnd(
                clio::cte::core::Client::DeferKeyHashName(p));
            if (pend > sz) sz = pend;
            clio::run::u64 hw = HiwaterFor(p);
            if (hw > sz) sz = hw;
            stbuf->st_size = static_cast<cte_off_t>(sz);
          }
          stbuf->st_blksize =
              static_cast<decltype(stbuf->st_blksize)>(4096);
          stbuf->st_blocks = static_cast<decltype(stbuf->st_blocks)>(
              (static_cast<uint64_t>(stbuf->st_size) + 511) / 512);
          // Natural times from the tag record; utimens overrides win for
          // atime/mtime, ctime can only advance (same rules as the task
          // path).
          clio::run::u64 m_ct = trec.last_changed_;
          clio::run::u64 m_mt = trec.last_modified_;
          clio::run::u64 m_at = trec.last_read_;
          if (mrec.ov_mtime_ns_ != 0) m_mt = mrec.ov_mtime_ns_;
          if (mrec.ov_atime_ns_ != 0) m_at = mrec.ov_atime_ns_;
          if (mrec.ov_ctime_ns_ > m_ct) m_ct = mrec.ov_ctime_ns_;
          if (m_ct != 0)
            NsBitsToTimespec(m_ct, stbuf->st_ctim.tv_sec,
                             stbuf->st_ctim.tv_nsec);
          if (m_mt != 0)
            NsBitsToTimespec(m_mt, stbuf->st_mtim.tv_sec,
                             stbuf->st_mtim.tv_nsec);
          if (m_at != 0)
            NsBitsToTimespec(m_at, stbuf->st_atim.tv_sec,
                             stbuf->st_atim.tv_nsec);
          return 0;
        }
      }
    }
  }

  // COMPLETE-DIR negative (issue #1007 follow-up): the path has NO mirror
  // record, but its parent directory's record says the mirror reflects the
  // ENTIRE child set — a miss is then authoritative ENOENT with no task.
  // Negative lookups (one ahead of every create) cost 291 us each through
  // the chimod vs ~10 us here — the largest remaining checkout cost. Only
  // when TryGetFileRecordShm finds NOTHING: a present-but-refused record
  // (kShmFileNoFastPath, pending appends) means the file exists and needs
  // the task path.
  {
    auto *cte1 = CLIO_CTE_CLIENT;
    clio::cte::filesystem::ShmFileRecord self;
    if (cfs != nullptr && cte1 != nullptr && !cfs->MirrorSaturated() &&
        !cfs->TryGetFileRecordShm(p, &self)) {
      auto slash = p.find_last_of('/');
      if (slash != std::string::npos) {
        std::string parent = (slash == 0) ? "/" : p.substr(0, slash);
        clio::cte::filesystem::ShmFileRecord prec;
        bool have = cfs->TryGetFileRecordShm(parent, &prec);
        if (have && prec.Exists() &&
            (prec.flags_ & clio::cte::filesystem::kShmFileIsDir) &&
            (prec.flags_ & clio::cte::filesystem::kShmDirComplete) &&
            !(prec.flags_ & clio::cte::filesystem::kShmFileNoFastPath)) {
          // NoFastPath on a dir = demoted (symlink/hard-link child): its
          // negatives are no longer authoritative. MirrorRefuse ORs the bit
          // into the existing record, so Complete may still be set.
          return -ENOENT;
        }
      }
    }
  }

  // Delegate to the filesystem chimod: it owns exists/is-dir/logical-size.
  auto t = cfs->AsyncGetattr(p);
  t.Wait();
  if (t->GetReturnCode() != 0 || t->exists_ == 0) {
    return -ENOENT;
  }
  // Owner: a prior chown recorded an override (uid_/gid_ != 0xFFFFFFFF);
  // otherwise report the mounting user's uid/gid (files carry no stored owner).
  stbuf->st_uid =
      (t->uid_ != 0xFFFFFFFFu) ? static_cast<uid_t>(t->uid_) : getuid();
  stbuf->st_gid =
      (t->gid_ != 0xFFFFFFFFu) ? static_cast<gid_t>(t->gid_) : getgid();
  stbuf->st_ino = static_cast<ino_t>(t->ino_);  // stable inode = packed TagId
  // A chmod/create recorded permission bits (t->mode_ != 0xFFFFFFFF) win over
  // the synthesized defaults; keep only the low 12 bits (perms + setuid/gid/sticky).
  const bool have_mode = (t->mode_ != 0xFFFFFFFFu);
  const unsigned int perm = t->mode_ & 07777u;
  if (t->is_dir_) {
    stbuf->st_mode = S_IFDIR | (have_mode ? perm : 0755u);
    stbuf->st_nlink = 2;
  } else if (t->is_symlink_) {
    stbuf->st_mode = S_IFLNK | 0777;
    stbuf->st_nlink = 1;
    stbuf->st_size = static_cast<cte_off_t>(t->size_);  // target length
  } else {
    stbuf->st_mode = S_IFREG | (have_mode ? perm : 0644u);
    // POSIX link count, computed by the chimod inside the Getattr task —
    // the separate alias round trip this replaced ran on EVERY regular-file
    // stat (2+ times per file on a checkout).
    stbuf->st_nlink = static_cast<nlink_t>(t->nlink_);
    // cte_off_t is off_t on Linux; the WinFsp shim maps it for Windows.
    stbuf->st_size = static_cast<cte_off_t>(t->size_);
    // A deferred write that extended the file may not have advanced the
    // published size yet, but the write(2) it came from already returned —
    // stat must see it. High-water pending end, no drain (same rule as the
    // cfs client's GetAttr).
    clio::run::u64 pend = clio::cte::core::Client::DeferMaxPendingEnd(
        clio::cte::core::Client::DeferKeyHashName(p));
    if (pend > static_cast<clio::run::u64>(stbuf->st_size)) {
      stbuf->st_size = static_cast<cte_off_t>(pend);
    }
    // Sieve-path writes advance the chimod size only at close: overlay the
    // local hiwater in the meantime.
    clio::run::u64 hw = HiwaterFor(p);
    if (hw > static_cast<clio::run::u64>(stbuf->st_size)) {
      stbuf->st_size = static_cast<cte_off_t>(hw);
    }
  }
  // Report the 512-byte block count backing the file so stat(2) st_blocks is
  // non-zero for files that hold data (generic/615 asserts a buffered/direct
  // write shows allocated blocks). Derived from the logical size; directories
  // report 0. st_blksize advertises a sensible I/O unit for tools.
  stbuf->st_blksize = static_cast<decltype(stbuf->st_blksize)>(4096);
  stbuf->st_blocks = static_cast<decltype(stbuf->st_blocks)>(
      (static_cast<uint64_t>(stbuf->st_size) + 511) / 512);
  // Timestamps come from the tag as ns since the epoch (0 means the chimod had
  // no value, so leave that field at the epoch): ctime = last metadata change
  // (last_changed_), mtime = last content change (last_modified_), atime = last
  // access (last_read_). All three are surfaced from the same GetTagSize query.
  if (t->ctime_ != 0) {
    NsBitsToTimespec(t->ctime_, stbuf->st_ctim.tv_sec, stbuf->st_ctim.tv_nsec);
  }
  // Fall back to ctime when mtime is unknown, so a valid file never reports
  // mtime at the epoch while it has a real ctime (merged from #680).
  clio::run::u64 mtime_ns = (t->mtime_ != 0) ? t->mtime_ : t->ctime_;
  if (mtime_ns != 0) {
    NsBitsToTimespec(mtime_ns, stbuf->st_mtim.tv_sec, stbuf->st_mtim.tv_nsec);
  }
  if (t->atime_ != 0) {
    NsBitsToTimespec(t->atime_, stbuf->st_atim.tv_sec, stbuf->st_atim.tv_nsec);
  }
  return 0;
}

int cte_fuse_getattr_stat(const char *path, cte_stat_t *stbuf,
                          struct fuse_file_info *fi) {
  int rc = cte_fuse_getattr_stat_inner(path, stbuf, fi);
  if (rc != 0) return rc;
  TimesOverlay ov;
  if (TimesOverlayGet(std::string(path), &ov)) {
    if (ov.atime_ns != 0)
      NsBitsToTimespec(ov.atime_ns, stbuf->st_atim.tv_sec,
                       stbuf->st_atim.tv_nsec);
    if (ov.mtime_ns != 0)
      NsBitsToTimespec(ov.mtime_ns, stbuf->st_mtim.tv_sec,
                       stbuf->st_mtim.tv_nsec);
    // ctime only ADVANCES.
    clio::run::u64 cur = static_cast<clio::run::u64>(
        static_cast<int64_t>(stbuf->st_ctim.tv_sec) * 1000000000LL +
        stbuf->st_ctim.tv_nsec);
    if (ov.ctime_ns > cur)
      NsBitsToTimespec(ov.ctime_ns, stbuf->st_ctim.tv_sec,
                       stbuf->st_ctim.tv_nsec);
  }
  return 0;
}

#ifdef __APPLE__
// macFUSE's high-level getattr callback fills a fuse_darwin_attr, not a
// struct stat. Compute into a struct stat (== cte_stat_t here) and translate.
static void CopyStatToDarwinAttr(const struct stat &st,
                                 struct fuse_darwin_attr *attr) {
  memset(attr, 0, sizeof(*attr));
  attr->ino = st.st_ino;
  attr->mode = st.st_mode;
  attr->nlink = st.st_nlink;
  attr->uid = st.st_uid;
  attr->gid = st.st_gid;
  attr->rdev = st.st_rdev;
  attr->size = st.st_size;
  attr->blocks = st.st_blocks;
  attr->blksize = st.st_blksize;
  attr->flags = st.st_flags;
  attr->atimespec = st.st_atimespec;
  attr->mtimespec = st.st_mtimespec;
  attr->ctimespec = st.st_ctimespec;
  attr->btimespec = st.st_birthtimespec;
}

static int cte_fuse_getattr(const char *path, struct fuse_darwin_attr *attr,
                            struct fuse_file_info *fi) {
  struct stat stbuf;
  int rc = cte_fuse_getattr_stat(path, &stbuf, fi);
  if (rc != 0) return rc;
  CopyStatToDarwinAttr(stbuf, attr);
  return 0;
}
#else
// Linux (struct stat) and Windows (WinFsp stat via cte_stat_t).
static int cte_fuse_getattr(const char *path, cte_stat_t *stbuf,
                            struct fuse_file_info *fi) {
  return cte_fuse_getattr_stat(path, stbuf, fi);
}
#endif

int cte_fuse_utimens(const char *path, const cte_timespec_t tv[2],
                            struct fuse_file_info *fi) {
  (void)fi;
  // Translate the POSIX (atime, mtime) timespec pair into the chimod's flag
  // encoding: bit0/bit1 = explicit atime/mtime, bit2/bit3 = UTIME_NOW (resolved
  // server-side so it shares the tag clock). UTIME_OMIT leaves a field alone.
  clio::run::u32 flags = 0;
  clio::run::u64 atime_ns = 0, mtime_ns = 0;
#if defined(UTIME_NOW) && defined(UTIME_OMIT)
  if (tv != nullptr) {
    if (tv[0].tv_nsec == UTIME_NOW) {
      flags |= 0x4u;
    } else if (tv[0].tv_nsec != UTIME_OMIT) {
      flags |= 0x1u;
      // Signed arithmetic so pre-epoch times don't wrap (generic/258); stored
      // as the two's-complement bits, decoded symmetrically in NsBitsToTimespec.
      atime_ns = static_cast<clio::run::u64>(
          static_cast<int64_t>(tv[0].tv_sec) * 1000000000LL +
          static_cast<int64_t>(tv[0].tv_nsec));
    }
    if (tv[1].tv_nsec == UTIME_NOW) {
      flags |= 0x8u;
    } else if (tv[1].tv_nsec != UTIME_OMIT) {
      flags |= 0x2u;
      mtime_ns = static_cast<clio::run::u64>(
          static_cast<int64_t>(tv[1].tv_sec) * 1000000000LL +
          static_cast<int64_t>(tv[1].tv_nsec));
    }
  } else {
    // NULL tv means "set both to now".
    flags |= 0x4u | 0x8u;
  }
#else
  // Platform without UTIME_NOW/OMIT: treat as set-both-to-now.
  (void)tv;
  flags |= 0x4u | 0x8u;
#endif
  auto *cfs = CLIO_CFS_CLIENT;
  // Fire-and-forget by default: the kernel's writeback SETATTR stamps mtime
  // on EVERY dirtied file at flush time, and awaiting each one put a full
  // round trip per file on checkout-shaped workloads. A utimens on a missing
  // path is silently dropped in detached mode (the next getattr tells the
  // truth); CLIO_FUSE_ASYNC_UTIMENS=0 restores the awaited call.
  static const bool async_utimens = [] {
    const char *e = getenv("CLIO_FUSE_ASYNC_UTIMENS");
    return e == nullptr || *e != '0';
  }();
  if (async_utimens) {
    std::string p(path);
    if (clio::cte::core::Client::DeferKeyPending(
            clio::cte::core::Client::DeferKeyHashName(p))) {
      // Writes in flight: the stamp must EXECUTE after they land (see
      // CloserMain) — but this caller need not wait for that.
      PendingClose pc;
      pc.kind = PendingClose::kUtimens;
      pc.path = p;
      pc.atime_ns = atime_ns;
      pc.mtime_ns = mtime_ns;
      pc.flags = flags;
      EnqueueDrainOrdered(std::move(pc));
    } else {
      EnsureCreated(p);
      cfs->AsyncUtimensDetached(p, atime_ns, mtime_ns, flags);
    }
    {
      clio::run::u64 now_ns = static_cast<clio::run::u64>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count());
      clio::run::u64 a = (flags & 0x4u) ? now_ns
                         : (flags & 0x1u) ? atime_ns : 0;
      clio::run::u64 m = (flags & 0x8u) ? now_ns
                         : (flags & 0x2u) ? mtime_ns : 0;
      TimesOverlaySet(p, a, m, now_ns);
    }
    return 0;
  }
  EnsureCreated(std::string(path));
  auto t = cfs->AsyncUtimens(std::string(path), atime_ns, mtime_ns, flags);
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;
}

// chmod records the permission bits as a per-file override in the chimod
// (surfaced by getattr), mirroring chown. Storing the mode lets +x stick, so a
// binary copied onto the fs can be executed (generic/452) and chmod-then-proceed
// callers (e.g. mount's mtab updater in generic/089) still succeed. AsyncChmod
// resolves the path and returns ENOENT for a missing file, so no separate
// existence probe is needed.
int cte_fuse_chmod(const char *path, cte_mode_t mode,
                          struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  std::string p(path);
  // PENDING-CREATE chmod: the file's identity still lives client-side (git
  // chmods config.lock right after the minted create). Rewrite the mode in
  // the registry AND the queued batch entry so the flush carries it; if the
  // batch entry already swapped into an in-flight flush, fall through to the
  // barrier + task path.
  {
    std::lock_guard<std::mutex> lk(g_pc_mtx);
    auto it = g_pending_creates.find(p);
    if (it != g_pending_creates.end()) {
      it->second.mode = static_cast<clio::run::u32>(mode) & 07777u;
      for (auto &e : g_create_queue) {
        if (e.path_ == p) {
          e.mode_ = it->second.mode;
          return 0;
        }
      }
    }
  }
  EnsureCreated(p);
  // Probe existence first: the chmod path resolves via GetOrCreateTag, so a
  // missing target would otherwise be silently created. chmod(2) must ENOENT.
  auto g = cfs->AsyncGetattr(p);
  g.Wait();
  if (g->GetReturnCode() != 0 || g->exists_ == 0) return -ENOENT;
  auto t = cfs->AsyncChmod(p, static_cast<clio::run::u32>(mode) & 07777u);
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

// chown records a per-file owner uid/gid override in the chimod, surfaced by
// getattr (files carry no stored POSIX owner otherwise). A uid/gid of
// (uid_t)-1 == 0xFFFFFFFF means "leave that field unchanged" (POSIX), which is
// exactly the chimod's "unchanged" sentinel, so no translation is needed.
int cte_fuse_chown(const char *path, uid_t uid, gid_t gid,
                          struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  EnsureCreated(std::string(path));
  auto t = cfs->AsyncChown(std::string(path),
                           static_cast<clio::run::u32>(uid),
                           static_cast<clio::run::u32>(gid));
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

// ============================================================================
// Directory operations
// ============================================================================

#ifdef __APPLE__
using ClioFuseFillDirT = fuse_darwin_fill_dir_t;
#else
using ClioFuseFillDirT = fuse_fill_dir_t;
#endif

int cte_fuse_readdir_flush_guard() {
  // Pending minted creates are invisible to the chimod's listing until their
  // batch flushes; readdir callers (rm -r, ls) must see every child or they
  // act on a partial view (generic/070: rm -r left "non-empty" dirs whose
  // remaining children it was never shown).
  if (CreateQueueNonEmpty()) {
    FlushCreates();
  }
  return 0;
}

int cte_fuse_readdir(const char *path, void *buf,
                            ClioFuseFillDirT filler, cte_off_t offset,
                            struct fuse_file_info *fi,
                            enum fuse_readdir_flags flags) {
  cte_fuse_readdir_flush_guard();
  (void)offset;
  (void)fi;
  (void)flags;

  std::string p(path);

  filler(buf, ".", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
  filler(buf, "..", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));

  // Delegate listing to the filesystem chimod. It returns the full tag paths
  // of the directory's children; strip the directory prefix to get basenames.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncReaddir(p);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    return 0;
  }
  size_t prefix_len = p.size();
  if (!p.empty() && p.back() != '/') prefix_len++;
  // entries_ and inos_ are index-aligned (the chimod builds them together).
  for (size_t i = 0; i < t->entries_.size(); ++i) {
    std::string full = t->entries_[i].str();
    std::string name = full.size() > prefix_len ? full.substr(prefix_len) : full;
    // A child sentinel directory may come back as "<dir>/<name>/"; drop the
    // trailing slash so it shows as a plain directory entry.
    if (!name.empty() && name.back() == '/') name.pop_back();
    if (name.empty()) continue;
    // Supply only d_ino (the child's tag-derived inode) so getdents agrees with
    // a subsequent stat (generic/637). Leave st_mode = 0 (DT_UNKNOWN): the entry
    // type is not reliably known here, so the kernel issues a getattr to resolve
    // it — setting a wrong d_type would mislead `rm -rf`/`find`.
    cte_stat_t st;
    memset(&st, 0, sizeof(st));
    st.st_ino = i < t->inos_.size() ? static_cast<ino_t>(t->inos_[i]) : 0;
    // d_ino = 0 means "deleted" to glibc's readdir(3), which silently DROPS
    // the entry — children whose ino lookup missed became invisible to ls
    // and rm -r while the server still listed them (generic/070's
    // undeletable "non-empty" dirs). Any nonzero value keeps the entry
    // visible; a name hash stays stable across listings.
    if (st.st_ino == 0) {
      clio::run::u64 h = 1469598103934665603ull;
      for (unsigned char c : full) h = (h ^ c) * 1099511628211ull;
      st.st_ino = static_cast<ino_t>(h | 1);
    }
#ifdef __APPLE__
    // macFUSE's fill-dir callback consumes a fuse_darwin_attr, not a
    // struct stat — translate (same mapping getattr uses).
    struct fuse_darwin_attr attr;
    CopyStatToDarwinAttr(st, &attr);
    filler(buf, name.c_str(), &attr, 0, static_cast<fuse_fill_dir_flags>(0));
#else
    filler(buf, name.c_str(), &st, 0, static_cast<fuse_fill_dir_flags>(0));
#endif
  }
  return 0;
}

int cte_fuse_mkdir(const char *path, cte_mode_t mode) {
  (void)mode;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncMkdir(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // errno-style (0/EEXIST/EIO)
  return rc == 0 ? 0 : -rc;
}

int cte_fuse_rmdir(const char *path) {
  cte_fuse_readdir_flush_guard();  // children may still be pending creates
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRmdir(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/ENOTEMPTY/ENOENT/EIO
  return rc == 0 ? 0 : -rc;
}

// ============================================================================
// File lifecycle
// ============================================================================

// O_DIRECT needs no special handling: we deliberately do NOT set the per-file
// direct_io flag. Our read/write handlers already work at arbitrary offsets and
// sizes, so an O_DIRECT open flows through the exact same buffered page-cache
// path as a regular open — it is never rejected. (Enabling per-file direct_io
// would make the kernel return ENODEV for any mmap of an O_DIRECT fd, breaking
// programs that both O_DIRECT and mmap the same file, e.g. fsx -Z; see #597.)

// Honor O_TRUNC: the chimod open resolves the tag but does not truncate, so an
// open/creat of an existing file with O_TRUNC would keep its old page-blobs
// (leaving stale data an app expects to be gone — e.g. reads of a re-created
// file's holes). Clear it to zero length here, which frees those blobs.
// Declared here as well as in the header: the header's export block is
// Linux-only (#ifndef _WIN32), and MSVC compiles this TU for the wheels.
int cte_fuse_truncate(const char *path, cte_off_t size,
                      struct fuse_file_info *fi);

static inline void MaybeTruncateOnOpen(clio::cte::filesystem::Client *cfs,
                                       const std::string &p, int flags) {
  if (flags & O_TRUNC) {
    // The FULL truncate hook, not a raw AsyncTruncate: O_TRUNC on reopen
    // must drain the file's deferred pipelines (closer AND sieve pages)
    // before the truncate lands, or a still-pending write from the last
    // open session lands AFTER it and resurrects the old bytes —
    // generic/729's dio write@0 + fsync + close + O_TRUNC reopen read the
    // previous phase's page back out of the "empty" file.
    (void)cfs;
    cte_fuse_truncate(p.c_str(), 0, nullptr);
  }
}

int cte_fuse_create(const char *path, cte_mode_t mode,
                           struct fuse_file_info *fi) {
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  // SIEVE-CREATE fast path (O_EXCL only): mint the TagId, record the
  // creation for the flusher's MultiCreate batch, and return immediately —
  // no task. The kernel's authoritative negative LOOKUP just preceded this
  // CREATE, and the kernel serializes same-path creation, so exclusivity
  // holds without asking the chimod.
  if (SieveDataEnabled() && (fi->flags & O_EXCL) && CLIO_CTE_CLIENT != nullptr) {
    auto *handle = new CfsHandle();
    handle->fh = 0;  // no server handle; data ops key off the tag
    handle->path = p;
    handle->tag = MintTagId();
    {
      PendingCreate pc;
      pc.tag = handle->tag;
      pc.mode = static_cast<clio::run::u32>(mode);
      pc.ctime_ns = static_cast<clio::run::u64>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count());
      std::lock_guard<std::mutex> lk(g_pc_mtx);
      g_pending_creates[p] = pc;
      clio::cte::filesystem::MultiCreateEnt e;
      e.path_ = p;
      e.tag_packed_ =
          (static_cast<clio::run::u64>(handle->tag.major_) << 32) |
          static_cast<clio::run::u64>(handle->tag.minor_);
      e.mode_ = pc.mode;
      g_create_queue.push_back(std::move(e));
    }
    g_closer_cv.notify_one();
    {
      // Ensure the flusher thread exists even before the first close.
      std::lock_guard<std::mutex> lk(g_closer_mtx);
      if (!g_closer_started) {
        g_closer_started = true;
        g_closer = std::thread(CloserMain);
        g_closer.detach();
      }
    }
    fi->fh = reinterpret_cast<uint64_t>(handle);
    MaybeDirectIo(fi);
    return 0;
  }
  auto t = cfs->AsyncOpen(p, O_CREAT | O_RDWR, static_cast<clio::run::u32>(mode));
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  handle->tag = clio::cte::core::TagId(
      static_cast<clio::run::u32>(t->tag_packed_ >> 32),
      static_cast<clio::run::u32>(t->tag_packed_ & 0xffffffffULL));
  fi->fh = reinterpret_cast<uint64_t>(handle);
  MaybeDirectIo(fi);
  MaybeTruncateOnOpen(cfs, p, fi->flags);
  if (fi->flags & O_TRUNC) HiwaterClamp(p, 0);
  return 0;
}

int cte_fuse_open(const char *path, struct fuse_file_info *fi) {
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  // A minted create that has not flushed yet is invisible to the chimod; a
  // reopen inside that window must resolve locally or it would ENOENT.
  {
    PendingCreate pc;
    if (PendingCreateLookup(p, &pc)) {
      auto *handle = new CfsHandle();
      handle->fh = 0;
      handle->path = p;
      handle->tag = pc.tag;
      fi->fh = reinterpret_cast<uint64_t>(handle);
      MaybeDirectIo(fi);
      if (fi->flags & O_TRUNC) HiwaterClamp(p, 0);
      return 0;
    }
  }
  // The chimod honors O_CREAT: a plain open of a missing file returns
  // handle==0 so we can surface ENOENT.
  auto t = cfs->AsyncOpen(p, static_cast<clio::run::u32>(fi->flags), 0644);
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;
  if (t->handle_ == 0) return -ENOENT;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  handle->tag = clio::cte::core::TagId(
      static_cast<clio::run::u32>(t->tag_packed_ >> 32),
      static_cast<clio::run::u32>(t->tag_packed_ & 0xffffffffULL));
  fi->fh = reinterpret_cast<uint64_t>(handle);
  MaybeDirectIo(fi);
  MaybeTruncateOnOpen(cfs, p, fi->flags);
  if (fi->flags & O_TRUNC) HiwaterClamp(p, 0);
  return 0;
}

// Writes are DEFERRED through the cfs client's write-behind pipeline (see
// cte_fuse_write): fsync is the durability point that drains them; flush
// (close(2)) deliberately does NOT drain — the kernel page cache makes the
// same trade — but it does surface any failure a completed deferred write
// already latched against the file.
int cte_fuse_flush(const char *path, struct fuse_file_info *fi) {
  auto *handle = GetHandle(fi);
  const std::string p = handle ? handle->path : std::string(path ? path : "");
  if (p.empty()) return 0;
  int err = clio::cte::core::Client::DeferTakeKeyError(
      clio::cte::core::Client::DeferKeyHashName(p));
  return err != 0 ? -err : 0;
}

int cte_fuse_fsync(const char *path, int /*datasync*/,
                          struct fuse_file_info *fi) {
  auto *handle = GetHandle(fi);
  const std::string p = handle ? handle->path : std::string(path ? path : "");
  if (p.empty()) return 0;
  // Sieve-path durability: drain the file's page-blob keys. The logical
  // size still travels on close; a stat between fsync and close is served
  // by the hiwater overlay in getattr.
  if (handle != nullptr) {
    EnsureCreated(p);
    DrainSievePages(handle->tag, HiwaterFor(p));
  }
  auto *cfs = CLIO_CFS_CLIENT;
  if (CfsFlushCompat(cfs, p) != 0) return -errno;
  return 0;
}

int cte_fuse_release(const char *path, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return 0;
  auto *cfs = CLIO_CFS_CLIENT;
  // Fire-and-forget by default: every write was already awaited (or flushed
  // by the kernel before release under writeback caching), so Close's only
  // effect is server-side handle bookkeeping — and release() runs once per
  // file, which made its round trip ~30% of the per-file cost on a kernel
  // checkout. release's return code is not delivered to close(2) by FUSE
  // anyway. CLIO_FUSE_ASYNC_CLOSE=0 restores the awaited close.
  static const bool async_close = [] {
    const char *e = getenv("CLIO_FUSE_ASYNC_CLOSE");
    return e == nullptr || *e != '0';
  }();
  int rc = 0;
  const clio::run::u64 hiwater = HiwaterFor(handle->path);
  const bool minted = (handle->fh == 0);
  if (async_close) {
    // Ordering: the close (and the size push it carries) must EXECUTE after
    // the file's in-flight writes land — cfs-path writes under FileKey,
    // sieve writes under the page keys — but release() need not wait: the
    // lazy closer does, off the caller's path.
    if (minted || hiwater != 0 ||
        clio::cte::core::Client::DeferKeyPending(
            clio::cte::core::Client::DeferKeyHashName(handle->path))) {
      PendingClose pc;
      pc.kind = PendingClose::kClose;
      pc.fh = handle->fh;
      pc.path = handle->path;
      pc.tag = handle->tag;
      pc.hiwater = hiwater;
      EnqueueDrainOrdered(std::move(pc));
    } else {
      cfs->AsyncCloseDetached(handle->fh);
    }
  } else {
    clio::cte::core::Client::DeferAwaitKey(
        clio::cte::core::Client::DeferKeyHashName(handle->path));
    DrainSievePages(handle->tag, hiwater);
    if (minted) {
      EnsureCreated(handle->path);
      if (hiwater != 0) {
        auto tt = cfs->AsyncAdvanceSize(
            (static_cast<clio::run::u64>(handle->tag.major_) << 32) |
                static_cast<clio::run::u64>(handle->tag.minor_),
            hiwater);
        tt.Wait();
        HiwaterErase(handle->path);
      }
    } else {
      auto t = cfs->AsyncClose(handle->fh, hiwater);
      t.Wait();
      if (hiwater != 0) HiwaterErase(handle->path);
      rc = (t->GetReturnCode() == 0) ? 0 : -EIO;
    }
  }
  delete handle;
  fi->fh = 0;
  return rc;
}

// ============================================================================
// Read / Write — delegated to the chimod's page-based I/O
// ============================================================================

int cte_fuse_read(const char *path, char *buf, size_t size,
                         cte_off_t offset, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return -EBADF;

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);
  if (size == 0) return 0;

  // SIEVE-DIRECT read: page-wise AsyncGetBlobDefer — sieve/pending bytes
  // served from their staging (RYW, no wait), settled bytes from the SHM
  // mirror (zero IPC), RPC as the fallback. EOF clamps against the chimod
  // size raised by the local hiwater (bytes acknowledged to write(2) but
  // not yet carried to the chimod by close).
  auto *cte = CLIO_CTE_CLIENT;
  auto *cfs = CLIO_CFS_CLIENT;
  if (SieveDataEnabled() && cte != nullptr && !handle->tag.IsNull()) {
    clio::run::u64 fsize = 0;
    PendingCreate pc_probe;
    if (!PendingCreateLookup(handle->path, &pc_probe)) {
      // A pending minted create's whole size story is local; only ask the
      // chimod once the file exists server-side.
      bool exists = false;
      CfsGetAttrCompat(cfs, handle->path, &exists, &fsize);
    }
    clio::run::u64 hw = HiwaterFor(handle->path);
    if (hw > fsize) fsize = hw;
    clio::run::u64 off = static_cast<clio::run::u64>(offset);
    if (off >= fsize) return 0;
    clio::run::u64 want = std::min<clio::run::u64>(size, fsize - off);
    std::memset(buf, 0, want);  // holes / short pages read as zeros
    clio::run::u64 done = 0;
    while (done < want) {
      clio::run::u64 cur = off + done;
      clio::run::u64 page_off = cur % clio::cte::filesystem::kFsPageSize;
      clio::run::u64 n = std::min<clio::run::u64>(
          clio::cte::filesystem::kFsPageSize - page_off, want - done);
      auto g = cte->AsyncGetBlobDefer(
          handle->tag, clio::cte::filesystem::PageName(cur), page_off, n,
          buf + done);
      g.Wait();
      done += n;
    }
    return static_cast<int>(want);
  }
  // cfs tiered read fallback.
  ssize_t got = CfsReadCompat(cfs, handle->fh, handle->path,
                              static_cast<clio::run::u64>(offset), buf, size);
  if (got < 0) return -EIO;
  return static_cast<int>(got);
}

int cte_fuse_write(const char *path, const char *buf, size_t size,
                          cte_off_t offset, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return -EBADF;

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);
  if (size == 0) return 0;

  // SIEVE-DIRECT (user directive): the write is a memcpy into the CTE
  // sieve's page buffer — no task at all until a 64 KiB page fills and
  // ships. Read-your-writes comes from the sieve's registered extents; the
  // logical size rides the hiwater map until close carries it to the
  // chimod. Falls back to the cfs deferred WriteTask pipeline when the tag
  // is unknown or CLIO_FUSE_SIEVE=0.
  auto *cte = CLIO_CTE_CLIENT;
  if (SieveDataEnabled() && cte != nullptr && !handle->tag.IsNull()) {
    size_t done = 0;
    while (done < size) {
      clio::run::u64 cur = static_cast<clio::run::u64>(offset) + done;
      clio::run::u64 page_off = cur % clio::cte::filesystem::kFsPageSize;
      clio::run::u64 n = std::min<clio::run::u64>(
          clio::cte::filesystem::kFsPageSize - page_off, size - done);
      int rc = cte->AsyncPutBlobDefer(
          handle->tag, clio::cte::filesystem::PageName(cur), page_off, n,
          buf + done);
      if (rc != 0) return -EIO;
      done += n;
    }
    HiwaterRaise(handle->path, static_cast<clio::run::u64>(offset) + size);
    return static_cast<int>(size);
  }
  // Deferred write-behind through the cfs client (staging pool, RYW
  // registration, fsync drain) — one submit, no wait.
  auto *cfs = CLIO_CFS_CLIENT;
  ssize_t wrote = CfsWriteCompat(cfs, handle->fh, handle->path,
                                 static_cast<clio::run::u64>(offset), buf,
                                 size);
  if (wrote < 0) return -errno;
  return static_cast<int>(wrote);
}

// ============================================================================
// Unlink / Truncate
// ============================================================================

int cte_fuse_unlink(const char *path) {
  auto *cfs = CLIO_CFS_CLIENT;
  // Deferred writes racing the unlink would land on a deleted file and latch
  // spurious errors; drain first (no-op unless this file has writes in
  // flight, which an unlink-after-write pattern rarely does).
  CloserBarrier();
  clio::cte::core::Client::DeferAwaitKey(
      clio::cte::core::Client::DeferKeyHashName(std::string(path)));
  EnsureCreated(std::string(path));
  HiwaterErase(std::string(path));
  auto t = cfs->AsyncUnlink(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/EISDIR/EIO
  if (rc == 0) {
    for (const auto &sib : LinkGroupDrop(std::string(path))) {
      InvalidatePath(sib);  // their nlink/ctime changed under the TTL
    }
  }
  return rc == 0 ? 0 : -rc;
}

int cte_fuse_truncate(const char *path, cte_off_t size,
                             struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  // Order against in-flight deferred writes: a truncate applied before a
  // pending write lands would resurrect the truncated range (or vice versa).
  // Sieve-path writes live under the file's page-blob keys; drain them too
  // (tag resolved via the SHM mirror, best-effort) and clamp the hiwater
  // overlay so stat reflects the truncation immediately.
  clio::cte::core::TagId tr_tag = clio::cte::core::TagId::GetNull();
  clio::run::u64 tr_old_extent = 0;
  {
    const std::string p(path);
    // A queued asynchronous close carries the file's pre-truncate hiwater and
    // pushes it via AdvanceSize, which only ever RAISES the size — landing
    // after this truncate it resurrects the truncated length (CI fuse_ops
    // truncate: wrote 1000, truncated to 100, stat said 1000). Drain the
    // closer first, exactly like rename and unlink.
    CloserBarrier();
    // Resolve the tag from EVERY identity source, not just the mirror: an
    // ftruncate arrives with the open handle (fi), and a minted file whose
    // MultiCreate has not flushed has its tag only in the pending table — the
    // mirror-only lookup skipped the sieve drain for both, and pages flushed
    // AFTER the truncate resurrected the dropped bytes (xfstests fsx).
    if (fi != nullptr) {
      auto *h = GetHandle(fi);
      if (h != nullptr) tr_tag = h->tag;
    }
    if (tr_tag.IsNull()) {
      PendingCreate pc;
      if (PendingCreateLookup(p, &pc)) tr_tag = pc.tag;
    }
    if (tr_tag.IsNull()) {
      clio::cte::filesystem::ShmFileRecord rec;
      if (cfs->TryGetFileRecordShm(p, &rec) && !rec.tag_id_.IsNull()) {
        tr_tag = clio::cte::core::TagId(rec.tag_id_.major_, rec.tag_id_.minor_);
      }
    }
    if (tr_tag.IsNull()) {
      // Authoritative fallback (generic/729 residue): a refused/absent
      // mirror record left the tag unresolved, the sieve drain silently
      // skipped, and a pending write from the file's previous open session
      // landed AFTER the truncate. Ask the server; escape the path for an
      // exact-match regex.
      auto *cte_q = CLIO_CTE_CLIENT;
      if (cte_q != nullptr) {
        std::string re = "^";
        for (char c : p) {
          if (std::strchr("\\^$.|?*+()[]{}", c) != nullptr) re += '\\';
          re += c;
        }
        re += "$";
        auto q = cte_q->AsyncTagQuery(re, 1);
        q.Wait();
        if (q->GetReturnCode() == 0 && !q->result_ids_.empty()) {
          clio::run::u64 packed = q->result_ids_[0];
          tr_tag = clio::cte::core::TagId(
              static_cast<clio::run::u32>(packed >> 32),
              static_cast<clio::run::u32>(packed & 0xffffffffULL));
        }
      }
    }
    EnsureCreated(p);
    clio::cte::core::Client::DeferAwaitKey(
        clio::cte::core::Client::DeferKeyHashName(p));
    if (!tr_tag.IsNull()) {
      DrainSievePages(tr_tag, HiwaterFor(p));
    }
    tr_old_extent = HiwaterFor(p);
    HiwaterClamp(p, static_cast<clio::run::u64>(size));
  }
  auto t = cfs->AsyncTruncate(
      std::string(path), static_cast<clio::run::u64>(size),
      tr_tag.IsNull() ? 0ULL
                      : ((static_cast<clio::run::u64>(tr_tag.major_) << 32) |
                         static_cast<clio::run::u64>(tr_tag.minor_)),
      tr_old_extent);
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

#ifdef __linux__
// Write `len` zero bytes at `off` through an open handle, chunked so a large
// range doesn't need one giant SHM buffer. Used by ZERO_RANGE. Returns 0 or a
// negative errno.
static int cte_fuse_write_zeros(struct fuse_file_info *fi, cte_off_t off,
                                cte_off_t len) {
  // Zeros MUST take the exact same path as write(2) — with the sieve-direct
  // adapter that is AsyncPutBlobDefer on the file's TAG, not the cfs handle
  // pipeline. The old cfs->AsyncWrite(handle->fh, ...) route (a) wrote via a
  // handle that is 0 for minted files and (b) created a SECOND data path
  // whose bytes the sieve's read-your-writes extents then overrode —
  // generic/008's zeroed first byte read back as its old value.
  constexpr size_t kChunk = 64 * 1024;
  std::unique_ptr<char[]> zbuf(new char[kChunk]());
  for (cte_off_t done = 0; done < len;) {
    const size_t n = static_cast<size_t>(
        ((len - done) < static_cast<cte_off_t>(kChunk)) ? (len - done)
                                                        : kChunk);
    int wrote = cte_fuse_write(nullptr, zbuf.get(), n, off + done, fi);
    if (wrote < 0) return wrote;
    if (wrote == 0) return -EIO;
    done += wrote;
  }
  return 0;
}

// fallocate — page-blobs are created lazily on write and holes read back as
// zeros, so we never reserve storage ahead of time. Supported modes:
//   * mode==0            : grow EOF to offset+length (never shrinks).
//   * FALLOC_FL_KEEP_SIZE: no-op success (nothing to reserve).
//   * FALLOC_FL_ZERO_RANGE: make [offset,offset+length) read as zeros by
//     writing zeros through the open handle (with KEEP_SIZE, restore EOF after
//     if the write extended it). This is a correct ZERO_RANGE for our
//     hole-reads-as-zero model and unblocks the fzero xfstests.
// Layout-shifting modes (punch hole, collapse/insert range) would need a
// chimod-level block-dealloc/shift op and still return EOPNOTSUPP.
static int cte_fuse_fallocate(const char *path, int mode, cte_off_t offset,
                              cte_off_t length, struct fuse_file_info *fi) {
  const int kSupportedModes = FALLOC_FL_KEEP_SIZE | FALLOC_FL_ZERO_RANGE;
  if (mode & ~kSupportedModes) {
    return -EOPNOTSUPP;  // punch/collapse/insert: layout-changing
  }
  if (offset < 0 || length <= 0) {
    return -EINVAL;
  }

  if (mode & FALLOC_FL_ZERO_RANGE) {
    auto *handle = GetHandle(fi);
    if (!handle) return -EBADF;
    // Original size, so KEEP_SIZE can restore EOF if the zero-write grows it.
    cte_stat_t st;
    int rc = cte_fuse_getattr_stat(path, &st, fi);
    if (rc != 0) return rc;
    rc = cte_fuse_write_zeros(fi, offset, length);
    if (rc != 0) return rc;
    if ((mode & FALLOC_FL_KEEP_SIZE) && (offset + length) > st.st_size) {
      // Restore EOF through the REAL truncate hook: it drains the deferred
      // write pipelines (path-keyed cfs writes AND sieve pages) before the
      // task. The raw AsyncTruncate raced the still-in-flight zero writes,
      // which then re-extended the file — fstat right after a
      // ZERO_RANGE|KEEP_SIZE reported the zero op's end as the size (fsx,
      // sieve-off mode).
      int trc = cte_fuse_truncate(path, st.st_size, fi);
      if (trc != 0) return trc;
    }
    // The zeros were written DAEMON-side: the kernel only drops its own
    // cached pages for PUNCH_HOLE, so a later mmap read of this range
    // served the stale pre-zero bytes (generic/075 fsx: every mismatching
    // byte was a should-be-zero). Invalidate the file's page cache.
    InvalidatePath(std::string(path));
    return 0;
  }

  if (mode & FALLOC_FL_KEEP_SIZE) {
    return 0;  // no size change requested, and there is nothing to reserve
  }

  // mode == 0: extend EOF to offset+length if the file is currently shorter.
  cte_stat_t st;
  int rc = cte_fuse_getattr_stat(path, &st, fi);
  if (rc != 0) return rc;
  cte_off_t need = offset + length;
  if (need <= st.st_size) return 0;  // already large enough; never shrink

  // Extend via the real truncate hook for the same pipeline-drain ordering
  // as the KEEP_SIZE branch above.
  int trc = cte_fuse_truncate(path, need, fi);
  return trc;
}
#endif  // __linux__

int cte_fuse_link(const char *from, const char *to) {
  EnsureCreated(std::string(from));
  EnsureCreated(std::string(to));
  // Hard link `to` -> existing file `from`. The chimod binds both names to the
  // same CTE tag (a tag-level alias), so they share all data.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncLink(std::string(from), std::string(to));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc == 0) {
    LinkGroupJoin(std::string(from), std::string(to));
    InvalidatePath(std::string(from));  // nlink/ctime changed under the TTL
  }
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes
}

int cte_fuse_symlink(const char *target, const char *path) {
  // Create a symlink at `path` pointing at `target`. The chimod stores the
  // target string in a reserved marker blob under `path`'s tag.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncSymlink(std::string(target), std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/EEXIST/EIO
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes
}

int cte_fuse_readlink(const char *path, char *buf, size_t size) {
  // Read the symlink target into `buf` (NUL-terminated). FUSE readlink returns
  // 0 on success (not the length).
  if (size == 0) {
    return -EINVAL;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncReadlink(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/ENOENT/EINVAL
  if (rc != 0) {
    return -rc;
  }
  std::string target = t->target_.str();
  size_t n = std::min(target.size(), size - 1);
  std::memcpy(buf, target.data(), n);
  buf[n] = '\0';
  return 0;
}

static int cte_fuse_setxattr_ensure(const std::string &p) { EnsureCreated(p); return 0; }
static int cte_fuse_setxattr(const char *path, const char *name,
                             const char *value, size_t size, int flags) {
  cte_fuse_setxattr_ensure(std::string(path));
  // Set xattr `name` on `path`. `value` is raw bytes (may contain NULs), so
  // preserve its length rather than treating it as a C string. `flags` carries
  // XATTR_CREATE(1) / XATTR_REPLACE(2), matching the runtime's bit checks.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncSetxattr(std::string(path), std::string(name),
                              std::string(value, size),
                              static_cast<unsigned int>(flags));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // EEXIST/ENODATA/ENOENT/EIO -> negative errno
}

static int cte_fuse_getxattr(const char *path, const char *name, char *value,
                             size_t size) {
  // Read xattr `name` of `path`. Return the value length (POSIX getxattr);
  // size==0 is a length query. Missing attribute -> -ENODATA.
  //
  // security.* / system.* short-circuit: the kernel probes
  // security.capability before every write-out and system.posix_acl_* on
  // permission checks. We never store either namespace (setxattr callers use
  // user.*), so answering locally saves a chimod round trip PER CREATE —
  // one of the four per-file waits that made kernel checkouts 20x slower
  // than ext4.
  // CLIO_FUSE_XATTR=0: answer -ENOSYS, which the kernel latches PER MOUNT —
  // it stops sending xattr ops entirely (a checkout issued 71k GETXATTR
  // probes, each a full FUSE transition). Default keeps xattrs working.
  static const bool xattr_enosys = [] {
    const char *e = getenv("CLIO_FUSE_XATTR");
    return e != nullptr && *e == '0';
  }();
  if (xattr_enosys) {
    return -ENOSYS;
  }
  if (strncmp(name, "security.", 9) == 0 || strncmp(name, "system.", 7) == 0) {
    return -ENODATA;
  }
  {
    // A pending minted create has no server record yet; it also has no
    // xattrs. Answer locally — the task path would say ENOENT.
    PendingCreate pc;
    if (PendingCreateLookup(std::string(path), &pc)) return -ENODATA;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncGetxattr(std::string(path), std::string(name));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc != 0) {
    return -rc;  // ENOENT (file absent)
  }
  if (t->found_ == 0) {
    return -ENODATA;  // attribute not present
  }
  std::string val = t->value_.str();
  size_t len = val.size();
  if (size == 0) {
    return static_cast<int>(len);  // length query
  }
  if (size < len) {
    return -ERANGE;
  }
  std::memcpy(value, val.data(), len);
  return static_cast<int>(len);
}

#ifdef __APPLE__
// macFUSE's xattr callbacks carry an extra `position` argument (resource-fork
// offset for the com.apple.ResourceFork attribute). We store xattrs whole, so
// only position 0 is meaningful; reject sub-range access like most non-HFS
// FUSE filesystems do.
static int cte_fuse_setxattr_darwin(const char *path, const char *name,
                                    const char *value, size_t size, int flags,
                                    uint32_t position) {
  if (position != 0) {
    return -EINVAL;
  }
  return cte_fuse_setxattr(path, name, value, size, flags);
}

static int cte_fuse_getxattr_darwin(const char *path, const char *name,
                                    char *value, size_t size,
                                    uint32_t position) {
  if (position != 0) {
    return -EINVAL;
  }
  return cte_fuse_getxattr(path, name, value, size);
}
#endif  // __APPLE__

static int cte_fuse_listxattr(const char *path, char *list, size_t size) {
  // Return the NUL-separated, NUL-terminated list of xattr names. size==0 is a
  // length query.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncListxattr(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc != 0) {
    return -rc;  // ENOENT
  }
  std::string names = t->names_.str();
  size_t len = names.size();
  if (size == 0) {
    return static_cast<int>(len);  // length query
  }
  if (size < len) {
    return -ERANGE;
  }
  std::memcpy(list, names.data(), len);
  return static_cast<int>(len);
}

static int cte_fuse_removexattr(const char *path, const char *name) {
  EnsureCreated(std::string(path));
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRemovexattr(std::string(path), std::string(name));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // ENODATA/ENOENT/EIO -> negative errno
}

#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1 << 0)  // from <linux/fs.h>; guarded to avoid header clash
#endif
#ifndef RENAME_EXCHANGE
#define RENAME_EXCHANGE (1 << 1)  // ditto; referenced by the unsupported-flag test
#endif

int cte_fuse_rename(const char *from, const char *to,
                           unsigned int flags) {
  auto *cfs = CLIO_CFS_CLIENT;
  // Deferred writes are keyed by PATH; a rename racing them would let the
  // writes land under the old name. Drain `from` first (git's
  // write-tmp-then-rename pattern hits this on every object file). Sieve
  // writes are keyed by (tag, page) and the tag survives the rename, so
  // they need no drain — but the hiwater overlay is path-keyed and moves.
  CloserBarrier();
  clio::cte::core::Client::DeferAwaitKey(
      clio::cte::core::Client::DeferKeyHashName(std::string(from)));
  EnsureCreated(std::string(from));
  {
    // The source's sieve pages (keyed by its tag) must land before the tag
    // graph mutates under the rename — especially when the destination's
    // old tag gets unlinked.
    const std::string fp(from);
    clio::cte::filesystem::ShmFileRecord frec;
    if (cfs->TryGetFileRecordShm(fp, &frec) && !frec.tag_id_.IsNull()) {
      DrainSievePages(clio::cte::core::TagId(frec.tag_id_.major_,
                                             frec.tag_id_.minor_),
                      HiwaterFor(fp));
    }
  }
  HiwaterRename(std::string(from), std::string(to));
  // RENAME_NOREPLACE: the rename must fail with EEXIST if `to` already exists.
  // Probe for the destination then fall through to a plain rename. This is the
  // standard high-level-FUSE approach (a tiny TOCTOU window vs a truly atomic
  // check, acceptable for a single-namespace rename). RENAME_EXCHANGE and
  // RENAME_WHITEOUT need chimod-level atomic swap / whiteout support and stay
  // EINVAL so callers fall back cleanly.
  if (flags & RENAME_NOREPLACE) {
    // A pending minted create is invisible to the chimod until its batch
    // flushes — probe the client-side registry FIRST or NOREPLACE onto a
    // just-created file wrongly succeeds (generic/024).
    {
      PendingCreate pc;
      if (PendingCreateLookup(std::string(to), &pc)) return -EEXIST;
    }
    auto g = cfs->AsyncGetattr(std::string(to));
    g.Wait();
    if (g->GetReturnCode() == 0 && g->exists_ != 0) return -EEXIST;
    flags &= ~static_cast<unsigned int>(RENAME_NOREPLACE);
  }
  if (flags != 0) {
    return -EINVAL;  // RENAME_EXCHANGE / RENAME_WHITEOUT unsupported
  }
  auto t = cfs->AsyncRename(std::string(from), std::string(to));
  t.Wait();
  if (t->GetReturnCode() == 0) LinkGroupRename(std::string(from), std::string(to));
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes (ENOENT/EIO)
}

// ============================================================================
// Main
// ============================================================================

// Report filesystem statistics. Total and remaining capacity are the real
// cluster-wide values, obtained from the CTE: GetCapacity sums the registered
// targets' total and remaining capacity per node, and a Broadcast aggregates
// that across the cluster (the task's AggregateOut sums per-node results).
// Reporting a non-zero capacity also matters operationally: a 0-block fs is
// hidden by `df` (which lists no path), which breaks tools that probe free
// space and xfstests' mount detection.
int cte_fuse_statfs(const char *path, cte_statvfs_t *stbuf) {
  (void)path;
  std::memset(stbuf, 0, sizeof(*stbuf));
  constexpr fsblkcnt_t kBlockSize = 4096;

  clio::run::u64 total_bytes = 0;
  clio::run::u64 remaining_bytes = 0;
  auto *cte = CLIO_CTE_CLIENT;
  if (cte != nullptr) {
    // Broadcast: total + remaining capacity across the whole cluster.
    auto t = cte->AsyncGetCapacity(clio::run::PoolQuery::Broadcast());
    t.Wait();
    if (t->return_code_ == 0) {
      total_bytes = t->total_capacity_;
      remaining_bytes = t->remaining_capacity_;
    }
  }
  fsblkcnt_t total_blocks = total_bytes / kBlockSize;
  fsblkcnt_t free_blocks = remaining_bytes / kBlockSize;

  stbuf->f_bsize = kBlockSize;
  stbuf->f_blocks = total_blocks;
  // Report real remaining space as both free and available (no reservation
  // distinction), so df shows used = total - remaining.
  stbuf->f_bfree = free_blocks;
  stbuf->f_bavail = free_blocks;
  stbuf->f_files = static_cast<fsfilcnt_t>(1) << 20;
  stbuf->f_ffree = static_cast<fsfilcnt_t>(1) << 20;
#ifdef __APPLE__
  // Darwin struct statfs has no f_frsize/f_favail/f_namemax; f_iosize is its
  // optimal-transfer-size analogue.
  stbuf->f_iosize = kBlockSize;
#else
  stbuf->f_frsize = kBlockSize;
  stbuf->f_favail = static_cast<fsfilcnt_t>(1) << 20;
  stbuf->f_namemax = 255;
#endif
  return 0;
}

// External linkage (declared in fuse_cte.h): the mount/process-entry glue in
// fuse_cte_main.cc points fuse_main()/fuse_new() at this table. The callbacks
// it references keep internal linkage — their addresses are captured here.
const struct fuse_operations cte_fuse_ops = {
    .getattr = cte_fuse_getattr,
    .readlink = cte_fuse_readlink,
    .mkdir = cte_fuse_mkdir,
    .unlink = cte_fuse_unlink,
    .rmdir = cte_fuse_rmdir,
    .symlink = cte_fuse_symlink,
    .rename = cte_fuse_rename,
    .link = cte_fuse_link,
    .chmod = cte_fuse_chmod,
    .chown = cte_fuse_chown,
    .truncate = cte_fuse_truncate,
    .open = cte_fuse_open,
    .read = cte_fuse_read,
    .write = cte_fuse_write,
    .statfs = cte_fuse_statfs,
    .flush = cte_fuse_flush,
    .release = cte_fuse_release,
    .fsync = cte_fuse_fsync,
#ifdef __APPLE__
    .setxattr = cte_fuse_setxattr_darwin,
    .getxattr = cte_fuse_getxattr_darwin,
#else
    .setxattr = cte_fuse_setxattr,
    .getxattr = cte_fuse_getxattr,
#endif
    .listxattr = cte_fuse_listxattr,
    .removexattr = cte_fuse_removexattr,
    .readdir = cte_fuse_readdir,
    .init = cte_fuse_init,
    .destroy = cte_fuse_destroy,
    .create = cte_fuse_create,
    .utimens = cte_fuse_utimens,
#ifdef __linux__
    .fallocate = cte_fuse_fallocate,
#endif
};
