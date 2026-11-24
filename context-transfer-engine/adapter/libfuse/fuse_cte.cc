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
#include <map>
#include <new>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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
#include "clio_runtime/evlat.h"  // CLIO_EVLAT rd_* / wr_* stage timers
#include "clio_cte/core/content_transfer_engine.h"
#include "clio_cte/core/core_client.h"  // CLIO_CFS_CLIENT->Cte() + GetCapacity
#include "clio_cte/filesystem/filesystem_client.h"
#include "clio_cte/filesystem/fs_inode.h"
#include "clio_cte/filesystem/fs_inode_table.h"  // directory offsets
#include "clio_cte/filesystem/fs_shard.h"
#include "clio_cte/stream/stream_client.h"  // deferred O_APPEND
#include "clio_cte/replication/replication_tasks.h"  // FlushTagTask (fsync barrier)
#include "fuse_cte_profile.h"  // CLIO_FUSE_PROFILE: per-callback timing
#include "fuse_readahead.h"  // CLIO_FUSE_READAHEAD_PAGES: detector, windows

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

// O_NOATIME is Linux-only. Elsewhere no open carries it, so a zero flag
// makes every handle take the normal atime-update path.
#ifndef O_NOATIME
#define O_NOATIME 0
#endif

// What an open of a file that vanished after the kernel's lookup returns:
// ESTALE makes the kernel look the path up again. Windows (WinFsp) has no
// ESTALE and no such retry, so the file is simply not there.
#ifdef ESTALE
static constexpr int kOpenVanishedErrno = ESTALE;
#else
static constexpr int kOpenVanishedErrno = ENOENT;
#endif
/** Re-sends of a page read that completed with the node-lost code. */
static constexpr int kReadNodeLostRetries = 20;
/** Pause between them (ms): node death is declared within ~10 s. */
static constexpr int kReadNodeLostRetryMs = 500;

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

// Defined below, outside the anonymous namespace; used by fsync's size step.
static clio::cte::stream::Client &StreamClient();
static clio::run::u64 PackTag(const clio::cte::core::TagId &tag);
// A handle's read-ahead cache (CLIO_FUSE_READAHEAD_PAGES), defined with the
// read path; RaInit reads the setting at mount.
struct RaCache;
static void RaInit();

namespace {
/** The attributes a stat is built from (FillStatFromAttrs): a Getattr
 *  reply's, or the ones the Open that created a file returned for it. */
struct StatAttrs {
  bool is_dir = false;
  bool is_symlink = false;
  clio::run::u64 size = 0;
  clio::run::u64 ino = 0;    // stable inode = packed TagId
  clio::run::u64 ctime = 0;  // ns; 0 if unknown
  clio::run::u64 mtime = 0;  // ns; 0 if unknown
  clio::run::u64 atime = 0;  // ns; 0 if unknown
  clio::run::u32 uid = 0xFFFFFFFFu;   // all-ones = the mounting user
  clio::run::u32 gid = 0xFFFFFFFFu;   // all-ones = the mounting group
  clio::run::u32 mode = 0xFFFFFFFFu;  // all-ones = the default mode
  clio::run::u32 nlink = 1;
};

/** Bumped by every truncate/chmod/chown/utimens/fallocate through this
 *  mount: attributes kept from before one are not served (TakeOpenAttrs). */
std::atomic<clio::run::u64> g_setattr_epoch{0};
/** Seconds the kernel may cache FILE attributes (cfg->attr_timeout); set at
 *  init. When positive, a read through an open handle reuses the size it
 *  last fetched for that long (see ReadClampSize). 0: every read asks. */
double g_file_attr_ttl_s = 0.0;
/** Pages kept fetched ahead of a sequential reader per open handle
 *  (CLIO_FUSE_READAHEAD_PAGES); set at init (RaInit). 0: read-ahead off. */
clio::run::u32 g_ra_pages = 0;

/** Per-open-file state: the chimod handle + the path it was opened on. */
struct CfsHandle {
  clio::run::u64 fh = 0;
  std::string path;
  // The file's CTE tag (null when unknown): lets data ops drive page blobs
  // through the CTE client directly (Client::PutBlob / GetBlob: the shm
  // cache when connected, else a task) instead of a chimod task per op.
  clio::cte::core::TagId tag = clio::cte::core::TagId::GetNull();
  // Guards `path`, which follows renames of the open file (see HandlePath).
  std::mutex path_mu;
  // Opened with O_APPEND (see AppendModeOf / cte_fuse_write).
  bool append = false;
  // Deferred appends accepted through this handle and not yet flushed: a
  // read, fstat, fsync or close through it first waits for their merge.
  std::atomic<bool> appended{false};
  // Opened with O_NOATIME, or atime already touched by a read through it.
  std::atomic<bool> atime_done{false};
  // Bytes were written through this handle since its last flush: only such
  // a flush has anything to publish (a scanner's or reader's close of a
  // freshly written file does not).
  std::atomic<bool> wrote{false};
  // The open-file record of the libfuse node it was opened on (null: not
  // registered; see RegisterOpenFile).
  std::shared_ptr<struct OpenNode> open_node;
  // The attributes the Open that CREATED this file returned with it. They
  // answer the one getattr(path, fi) the FUSE layer makes inside the same
  // create request, which otherwise costs a round trip to the inode's home
  // and registers this node as a holder of the inode. Set by KeepOpenAttrs,
  // taken (once) by TakeOpenAttrs; open_attr_ok publishes the other fields.
  StatAttrs open_attr;
  std::atomic<bool> open_attr_ok{false};
  std::thread::id open_attr_thread;                    // the creating thread
  std::chrono::steady_clock::time_point open_attr_at;  // when they were kept
  clio::run::u64 open_attr_epoch = 0;                  // g_setattr_epoch then
  // Size last fetched for a read through this handle, reused for
  // g_file_attr_ttl_s (ReadClampSize). rd_size_ns == 0: nothing kept.
  std::atomic<clio::run::u64> rd_size{0};
  std::atomic<int64_t> rd_size_ns{0};  // steady-clock ns when it was fetched
  std::atomic<clio::run::u64> rd_size_epoch{0};  // g_setattr_epoch then
  // Read-ahead cache (CLIO_FUSE_READAHEAD_PAGES): created by the handle's
  // first read when read-ahead is on, freed by RaRelease. Null otherwise.
  std::atomic<RaCache *> ra{nullptr};
};

// ---------------------------------------------------------------------------
// Case policy. On a case-insensitive volume every spelling of a path is one
// file, so every path-keyed cache below keys on the canonical path (the
// volume reports its policy at mount, cte_fuse_init). Values keep the
// spelling wherever a name goes back to the kernel or the runtime.
// ---------------------------------------------------------------------------
std::atomic<bool> g_case_insensitive{false};

/**
 * The key a path is cached under on this mount.
 * @param p path as spelled
 * @return its canonical form (the path itself on a case-sensitive volume)
 */
std::string CacheKey(const std::string &p) {
  return clio::cte::filesystem::FsCanonicalPath(
      p, g_case_insensitive.load(std::memory_order_relaxed));
}

/**
 * A map keyed by path that treats every spelling of a path as one key
 * (CacheKey). Iteration yields canonical keys.
 */
template <typename V>
class PathMap {
 public:
  using Map = std::unordered_map<std::string, V>;
  using iterator = typename Map::iterator;
  /** @return the value for `p`, default-constructed if absent */
  V &operator[](const std::string &p) { return m_[CacheKey(p)]; }
  /** @return the entry for `p`, or end() */
  iterator find(const std::string &p) { return m_.find(CacheKey(p)); }
  /** Insert unless present. @return (entry, inserted) */
  std::pair<iterator, bool> emplace(const std::string &p, const V &v) {
    return m_.emplace(CacheKey(p), v);
  }
  /** Erase `p`. @return entries erased (0 or 1) */
  size_t erase(const std::string &p) { return m_.erase(CacheKey(p)); }
  /** Erase at an iterator. @return the next iterator */
  iterator erase(iterator it) { return m_.erase(it); }
  iterator begin() { return m_.begin(); }
  iterator end() { return m_.end(); }
  size_t size() const { return m_.size(); }
  bool empty() const { return m_.empty(); }
  void clear() { m_.clear(); }

 private:
  Map m_;
};

/** A set of paths with the same key rule as PathMap. */
class PathSet {
 public:
  /** Add `p`. */
  void insert(const std::string &p) { s_.insert(CacheKey(p)); }
  /** Remove `p`. @return entries removed (0 or 1) */
  size_t erase(const std::string &p) { return s_.erase(CacheKey(p)); }
  /** @return 1 if `p` is present, else 0 */
  size_t count(const std::string &p) const { return s_.count(CacheKey(p)); }

 private:
  std::unordered_set<std::string> s_;
};

/** Orders spellings by their canonical form (hard-link groups, where one
 *  name may be named again by another spelling of it). */
struct CacheKeyLess {
  bool operator()(const std::string &a, const std::string &b) const {
    return CacheKey(a) < CacheKey(b);
  }
};

/** The files open on one libfuse node -- one kernel inode, one page cache. */
struct OpenNode {
  std::string path;  ///< the node's current path (follows renames)
  std::unordered_map<clio::run::u64, clio::run::u32> files;  ///< id -> opens
};

// Files open on this mount, per libfuse node (keyed by its current path).
// libfuse gives the kernel one inode per node, and a node is reached by a
// path, so files that are different on the server can share one kernel
// inode -- and one page cache -- here: after another node renames a new file
// over a name, and when a rename over a name open on this mount "hides" the
// node (libfuse renames it on the server to .fuse_hiddenXXX, which moves
// whatever file the name names NOW). A reader of one file then read pages a
// reader of the other had cached: one read(2) returned blocks of two
// different files (stress_safe_save across nodes). The records follow the
// nodes through renames (OpenFilesRenamed), so every open of a node sees the
// files already open on it.
std::mutex g_open_files_mu;
PathMap<std::shared_ptr<OpenNode>> g_open_files;
// Hidden names that DO exist on the server: hides that fell back to a
// server rename (HideOpenFile). Guarded by g_open_files_mu.
PathSet g_server_hidden;

/**
 * Register a new open of `h` on its node and keep the page cache from mixing
 * files: if another file is open on the same node, this descriptor bypasses
 * the page cache (direct_io), so each descriptor reads only its own file's
 * bytes.
 * @param h the new handle (tag and path set)
 * @param fi its fuse_file_info (direct_io may be set)
 */
static void RegisterOpenFile(CfsHandle *h, struct fuse_file_info *fi) {
  if (h->tag.IsNull()) return;
  const clio::run::u64 id =
      (static_cast<clio::run::u64>(h->tag.major_) << 32) | h->tag.minor_;
  std::lock_guard<std::mutex> g(g_open_files_mu);
  auto &node = g_open_files[h->path];
  if (node == nullptr) {
    node = std::make_shared<OpenNode>();
    node->path = h->path;
  }
  for (const auto &kv : node->files) {
    if (kv.first != id && kv.second > 0) {
      fi->direct_io = 1;
      break;
    }
  }
  ++node->files[id];
  h->open_node = node;
}

/**
 * Drop a handle's registration (at release).
 * @param h the handle being released
 */
static void UnregisterOpenFile(CfsHandle *h) {
  if (h->open_node == nullptr) return;
  const clio::run::u64 id =
      (static_cast<clio::run::u64>(h->tag.major_) << 32) | h->tag.minor_;
  std::lock_guard<std::mutex> g(g_open_files_mu);
  auto &node = h->open_node;
  auto ft = node->files.find(id);
  if (ft != node->files.end() && --ft->second == 0) node->files.erase(ft);
  if (node->files.empty()) {
    auto it = g_open_files.find(node->path);
    if (it != g_open_files.end() && it->second == node) g_open_files.erase(it);
  }
  node.reset();
}

/**
 * Move the open-file records along with libfuse's nodes after a successful
 * rename: the node at `from` (and, for a directory, every node below it) is
 * now reached by the new path. A node libfuse unhashed at `to` had no open
 * files (libfuse hides an open one first, which is itself a rename).
 * @param from the old path
 * @param to the new path
 */
static void OpenFilesRenamed(const std::string &from, const std::string &to) {
  std::lock_guard<std::mutex> g(g_open_files_mu);
  std::vector<std::shared_ptr<OpenNode>> moved;
  const std::string from_key = CacheKey(from);
  const std::string dir_prefix = from_key + "/";
  for (auto it = g_open_files.begin(); it != g_open_files.end();) {
    const std::string &k = it->first;  // canonical
    if (k == from_key || k.compare(0, dir_prefix.size(), dir_prefix) == 0) {
      moved.push_back(it->second);
      it = g_open_files.erase(it);
    } else {
      ++it;
    }
  }
  // A node's path keeps its own spelling below the moved prefix. Count
  // components, not bytes: folding can change a name's length.
  const size_t depth =
      static_cast<size_t>(std::count(from.begin(), from.end(), '/'));
  for (auto &node : moved) {
    size_t cut = 0, seen = 0;
    while (seen <= depth && cut != std::string::npos) {
      cut = node->path.find('/', cut + (seen == 0 ? 0 : 1));
      if (cut != std::string::npos) ++seen;
    }
    node->path = to + (cut == std::string::npos ? std::string()
                                                : node->path.substr(cut));
    g_open_files[node->path] = node;
  }
}

/**
 * Whether `path` is a name libfuse made up to hide an open file
 * (.fuse_hiddenXXXX). Such names exist only on this mount (HideOpenFile).
 * @param path a libfuse path (may be null)
 * @return true for a hidden name
 */
static bool IsHiddenName(const char *path) {
  if (path == nullptr) return false;
  const char *slash = strrchr(path, '/');
  const char *leaf = slash != nullptr ? slash + 1 : path;
  return strncmp(leaf, ".fuse_hidden", 12) == 0;
}

/**
 * The server path for a hidden name: the inode id form (FsIdStatPath) of a
 * file open on the hidden node.
 * @param path a hidden name (IsHiddenName)
 * @param out the id-form path
 * @return 0, or -ENOENT when no file is open on that node any more
 */
static int HiddenIdPath(const char *path, std::string *out) {
  std::lock_guard<std::mutex> g(g_open_files_mu);
  auto it = g_open_files.find(path);
  if (it == g_open_files.end()) return -ENOENT;
  for (const auto &kv : it->second->files) {
    if (kv.second > 0) {
      *out = clio::cte::filesystem::FsIdStatPath(kv.first);
      return 0;
    }
  }
  return -ENOENT;
}

/**
 * Map a libfuse path to the one to send the server: hidden names become the
 * id form of their open file, every other path is kept.
 * @param path in/out: the path to use (repointed into `out` when hidden)
 * @param out holds the hidden name's id-form path
 * @return 0, or -ENOENT for a hidden name with no open file
 */
static int MapHiddenPath(const char **path, std::string *out) {
  if (!IsHiddenName(*path)) return 0;
  const int rc = HiddenIdPath(*path, out);
  if (rc == 0) *path = out->c_str();
  return rc;
}

/**
 * The open file's CURRENT path.
 *
 * libfuse passes each op the path the file has NOW: after a rename, and
 * after the hidden rename (.fuse_hiddenXXXX) that keeps an unlinked-but-open
 * file alive. The name captured at open went stale then, and every lookup
 * keyed by it -- the size a read clamps to, the hiwater a write raises, the
 * size release publishes -- found nothing: an unlinked open file read back
 * 0 bytes and a renamed open file lost its size at close.
 * @param h    the open file
 * @param path the path libfuse supplied with this op (may be null)
 * @return the path to key this op's lookups by
 */
static std::string HandlePath(CfsHandle *h, const char *path) {
  std::lock_guard<std::mutex> lk(h->path_mu);
  if (path != nullptr && *path != '\0' && !IsHiddenName(path)) {
    if (h->path != path) h->path = path;
  } else if (!h->tag.IsNull()) {
    // The open file lost its last name (unlinked, or renamed over): libfuse
    // passes no path (hard_remove) or the local hidden name HideOpenFile
    // keeps for it. Key it by its inode from now on:
    // its old name may already belong to another file, whose path-keyed
    // state (sizes of queued writes, timestamps) this one must not touch.
    h->path = clio::cte::filesystem::FsIdStatPath(
        (static_cast<clio::run::u64>(h->tag.major_) << 32) | h->tag.minor_);
  }
  return h->path;
}

/**
 * The path an op on `fi` works on: libfuse's, or -- when libfuse has none
 * because the open file has no name any more -- its handle's inode key.
 * @param path libfuse's path (may be null)
 * @param fi the op's file info (may be null)
 * @return the path, or "" when there is neither
 */
static std::string OpPath(const char *path, struct fuse_file_info *fi) {
  if (IsHiddenName(path)) {
    std::string id_path;
    if (HiddenIdPath(path, &id_path) == 0) return id_path;
    path = nullptr;  // nothing open on the node: fall back to the handle
  }
  if (path != nullptr) return std::string(path);
  CfsHandle *h = fi != nullptr ? reinterpret_cast<CfsHandle *>(fi->fh)
                               : nullptr;
  return h != nullptr ? HandlePath(h, nullptr) : std::string();
}

/**
 * Score rule for the file data pages this adapter puts itself (the
 * filesystem module's `data_score` / `data_score_head_pages`). Fetched from
 * the filesystem container once at mount, before any write is served; off
 * (every page put with -1) when the module does not set `data_score`.
 */
clio::cte::core::PageScorePolicy g_page_score;

/**
 * True when this mount is one of several nodes sharing a namespace.
 *
 * Several fast paths here are sound only when this process sees every
 * mutation of the namespace: the asynchronous release publishes a file's
 * size after close(2) has returned. Neither holds when another node's mount
 * creates, reads or writes the same files, so they are disabled there.
 * Decided from the runtime's hostfile (more than one line); the environment
 * variable CLIO_FUSE_MULTINODE=0/1 overrides.
 * @return true on a multi-node deployment
 */
bool MultiNode() {
  static const bool v = [] {
    if (const char *e = getenv("CLIO_FUSE_MULTINODE")) {
      return *e == '1';
    }
    // Read the hostfile path straight from the server config: this runs
    // from cte_fuse_init, BEFORE CLIO_INIT has loaded the config manager,
    // and the answer is cached for the life of the mount.
    std::string hf;
    if (const char *conf = getenv("CLIO_SERVER_CONF")) {
      FILE *cf = fopen(conf, "r");
      if (cf != nullptr) {
        char cl[1024];
        while (fgets(cl, sizeof(cl), cf) != nullptr) {
          const char *k = strstr(cl, "hostfile:");
          if (k == nullptr || cl[0] == '#') continue;
          const char *v = k + strlen("hostfile:");
          while (*v == ' ' || *v == '\t' || *v == '"' || *v == '\'') ++v;
          std::string val(v);
          while (!val.empty() && strchr(" \t\r\n\"'", val.back())) {
            val.pop_back();
          }
          hf = val;
          break;
        }
        fclose(cf);
      }
    }
    if (hf.empty()) {
      auto *cfg = CLIO_CONFIG_MANAGER;
      if (cfg != nullptr) hf = cfg->GetHostfilePath();
    }
    if (hf.empty()) return false;
    FILE *f = fopen(hf.c_str(), "r");
    if (f == nullptr) return false;
    int hosts = 0;
    char line[512];
    while (fgets(line, sizeof(line), f) != nullptr) {
      for (char *c = line; *c != '\0'; ++c) {
        if (*c != ' ' && *c != '\t' && *c != '\n' && *c != '\r') {
          ++hosts;
          break;
        }
      }
    }
    fclose(f);
    return hosts > 1;
  }();
  return v;
}

/**
 * Whether fsync/close must force a replication durability barrier before
 * they can honor "once fsync returns, the bytes survive a crash".
 *
 * SYNCHRONOUS write-through (replication's replicate_period_ms_ == 0, or no
 * replication chained above the core pool at all) already gives fsync this
 * guarantee for free: PutBlob does not ack until the durable copy exists (or
 * there is no separate durable copy to wait for). ASYNCHRONOUS write-through
 * (replicate_period_ms_ > 0, the product default 50 ms) acks the primary
 * write immediately and copies it to the replica on the next periodic sweep
 * -- a crash inside that window would otherwise lose bytes fsync already
 * told the caller were safe.
 *
 * Decided ONCE, at mount time, with a single Monitor round trip to the data
 * pool (CLIO_CFS_CLIENT->Cte()'s pool_id_, the top of the write chain): asks the
 * pool whether it understands "replicate_period_ms" (only the replication
 * chimod does) and, if so, whether that period is nonzero. A bare core pool
 * (or any future chain module that does not recognise the query) leaves
 * `results_` empty and this returns false -- so the fsync/close path pays
 * NOTHING per call, and nothing extra at all, when replication is not in the
 * chain or is already synchronous.
 * @return true iff fsync must FlushTag before it can return success
 */
static bool NeedsReplicationFlushBarrier() {
  static const bool v = [] {
    auto *cte_c = CLIO_CFS_CLIENT->Cte();
    if (cte_c == nullptr) return false;
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<clio::cte::core::MonitorTask>(
        clio::run::CreateTaskId(), cte_c->pool_id_, clio::run::PoolQuery::Local(),
        std::string("replicate_period_ms"));
    auto fut = ipc->Send(task);
    fut.Wait();
    if (fut->GetReturnCode() != 0 || fut->results_.empty()) return false;
    const std::string &period_str = fut->results_.begin()->second;
    return std::strtol(period_str.c_str(), nullptr, 10) > 0;
  }();
  return v;
}

/**
 * Force the file's tag up to date on its persistent replica(s) -- the
 * async-write-through durability barrier NeedsReplicationFlushBarrier()
 * decided is necessary. No-op (and never called) when replication is absent
 * or already synchronous.
 * @param tag the file's tag id (its page blobs live under this key)
 * @return 0 on success, or a negative errno
 */
static int FlushReplicationBarrier(const clio::cte::core::TagId &tag) {
  if (tag.IsNull() || !NeedsReplicationFlushBarrier()) return 0;
  auto *cte_c = CLIO_CFS_CLIENT->Cte();
  auto *ipc = CLIO_CPU_IPC;
  auto task = ipc->NewTask<clio::cte::replication::FlushTagTask>(
      clio::run::CreateTaskId(), cte_c->pool_id_, clio::run::PoolQuery::Dynamic(),
      tag, /*replica=*/1, /*min_score=*/0.0f, clio::cte::core::Context());
  auto fut = ipc->Send(task);
  fut.Wait();
  // FlushTag is best-effort sweeping (see replication_runtime.cc): a
  // per-blob failure is reported but does not abort the sweep, so a nonzero
  // rc here means at least one blob could not be made durable. Surface it as
  // EIO rather than silently acking a durability guarantee that did not
  // actually hold.
  return fut->GetReturnCode() == 0 ? 0 : -EIO;
}

// Unsynced-write windows (issue #1133). With replication_factor 1 the bytes
// a node holds between two fsyncs (RAM tier, write-behind) die with it, and
// failover makes the next SyncTag succeed on the survivors. So each file
// written since its last fsync remembers when (wall clock) that window
// opened; fsync fails with EIO when a peer that has since died was last
// heard from after the window opened (SyncTagTask::liveness_change_ns_) --
// it may have accepted, and lost, some of those bytes. Writes made after
// it went silent went to live successors and fsync cleanly. Conservative:
// the node may have held none of them. Reported once, like Linux's errseq.
static std::mutex g_unsynced_mtx;
static std::unordered_map<clio::run::u64, clio::run::u64> g_unsynced;

/**
 * Note that `tag` was written and is not yet fsynced; the first write after
 * an fsync opens the window at the current wall-clock time.
 * @param tag the file's tag (null: nothing to track)
 */
static void UnsyncedNote(const clio::cte::core::TagId &tag) {
  if (tag.IsNull()) return;
  const clio::run::u64 k = PackTag(tag);
  std::lock_guard<std::mutex> lk(g_unsynced_mtx);
  if (g_unsynced.count(k) != 0) return;
  g_unsynced.emplace(
      k, static_cast<clio::run::u64>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count()));
}

/**
 * Close `tag`'s unsynced window (taken BEFORE the sync, so a write racing
 * it opens a new one).
 * @param tag the file's tag
 * @param opened_ns receives the wall-clock ns the window opened at
 * @return true if the tag had a window
 */
static bool UnsyncedTake(const clio::cte::core::TagId &tag,
                         clio::run::u64 *opened_ns) {
  const clio::run::u64 k = PackTag(tag);
  std::lock_guard<std::mutex> lk(g_unsynced_mtx);
  auto it = g_unsynced.find(k);
  if (it == g_unsynced.end()) return false;
  *opened_ns = it->second;
  g_unsynced.erase(it);
  return true;
}

/**
 * Make one CTE tag durable (the filesystem client's SyncTagDurable), and
 * apply this mount's unsynced-write window (#1133) on top.
 * @param tag the tag (null is a no-op)
 * @return 0, -ENOSPC when no persistent tier had room, or -EIO
 */
static int SyncOneTag(const clio::cte::core::TagId &tag) {
  using clio::cte::filesystem::Client;
  if (Client::FsyncDeferred()) return 0;
  if (CLIO_CFS_CLIENT->Cte() == nullptr || tag.IsNull()) return 0;
  clio::run::u64 opened_ns = 0;
  const bool unsynced = UnsyncedTake(tag, &opened_ns);
  bool lost_node = false;
  clio::run::u64 liveness_ns = 0;
  int rc = Client::SyncTagDurable(tag, &lost_node, &liveness_ns);
  if (rc != 0 || Client::FsyncDeferred()) return rc;
  // The live containers synced. A container whose node is down answers with
  // the lost-node code; once the cluster declares the node dead the sync
  // routes around it and succeeds. Either way, bytes written since the last
  // fsync that the node held are gone (#1133): fail if the window saw one.
  const bool moved = liveness_ns >= opened_ns;
  if (unsynced && (lost_node || moved)) {
    HLOG(kError, "clio_cte_fuse: fsync of tag {}.{}: a node {} while the "
         "file had unsynced writes; they may be lost, reporting EIO",
         tag.major_, tag.minor_,
         lost_node ? "is down" : "died");
    return -EIO;
  }
  if (lost_node) {
    HLOG(kWarning, "clio_cte_fuse: fsync of tag {}.{}: a container's node "
         "is down; the file had no unsynced writes", tag.major_, tag.minor_);
  }
  return 0;
}

/**
 * fsync(2)'s durability step, as the filesystem client's SyncDurable but
 * through this mount's SyncOneTag: the file's blobs, its size, then the
 * directory blocks naming it.
 * @param tag the file's tag (null: only the directory)
 * @param dir the directory whose entries must be durable
 * @return 0, -ENOSPC when no persistent tier had room, or -EIO
 */
static int SyncDurable(const clio::cte::core::TagId &tag,
                       const std::string &dir) {
  using clio::cte::filesystem::Client;
  int rc = SyncOneTag(tag);
  if (rc != 0) return rc;
  if (Client::FsyncDeferred()) return 0;
  rc = Client::SyncFileSize(tag);
  if (rc != 0) return rc;
  auto *cfs = CLIO_CFS_CLIENT;
  if (cfs == nullptr) return 0;
  rc = SyncOneTag(cfs->TagOfPath(dir, /*want_dir=*/true));
  if (rc != 0) return rc;
  return cfs->SyncMetaDurable();  // the directory journal
}

// Logical-size high-water for files written through direct page puts: the chimod
// only learns the size at close (CloseTask::advance_size_), so getattr and
// read-clamping consult this in the meantime. Path-keyed; renames move the
// entry, the closer erases it once the size-carrying close has been sent.
std::mutex g_hw_mtx;
PathMap<clio::run::u64> g_hiwater;
// Wall-clock time (ns) of the last write(2) this mount accepted per path,
// kept alongside the hiwater: page puts bypass the chimod, so the tag's mtime
// moves only when the size is published, and a stat right after write(2)
// reported the PRE-write mtime (make and rsync compare exactly that).
PathMap<clio::run::u64> g_wtime;
// A write time an explicit utimens took off the stat overlay (the caller's
// times win there), kept for the size push only: the push must still tell
// the chimod the bytes predate the utimens, or it stamps the time it lands
// and undoes the caller's mtime (cp -p / tar futimens before close).
PathMap<clio::run::u64> g_wtime_pre;

/** Current wall-clock time in nanoseconds since the epoch. */
static clio::run::u64 WallNowNs() {
  return static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

void TimesOverlayDropMtime(const std::string &p);

// Directory attribute cache. A directory's stat never comes from the SHM
// mirror (its times live in the directory block), so each one was a task to
// the runtime -- and a checkout stats directories constantly (397k of the
// 400k Getattr tasks of a 16k-file checkout). Directory attributes from the
// runtime are kept for the mount's attribute TTL, the staleness the kernel
// attribute cache already allows here, and dropped on every change this
// mount makes to the directory or its entries. On by default on single-node
// mounts only (there every change passes through this mount); multi-node
// mounts opt in with CLIO_FUSE_DIR_ATTR_CACHE_S and accept that staleness.
struct DirAttrEntry {
  cte_stat_t st;
  std::chrono::steady_clock::time_point expiry;
};
std::mutex g_dir_attr_mtx;
PathMap<DirAttrEntry> g_dir_attr;
double g_dir_attr_ttl_s = 0.0;  ///< set at init; 0 disables the cache
double g_entry_ttl_s = 0.0;  ///< kernel entry TTL, set at init (#1323)

/**
 * Serve a directory's attributes from the cache.
 * @param p the directory path
 * @param stbuf receives the attributes on a hit
 * @return true on a fresh hit
 */
// READDIR PREFETCH (opt-in, CLIO_FUSE_READDIR_PREFETCH=<window>): a listing
// of a large directory issues its children's getattrs concurrently, `window`
// in flight, and parks the results here. find/ls -l then stat every entry;
// without this each stat was its own serial round trip to the entry's owner
// node (IO500 find at 16 nodes: one pfind rank stat'ing 512k entries of
// mdtest-hard one by one took >12 min). Each result is consumed by the
// first getattr of that path and expires after kPrefetchTtlNs, so staleness
// is bounded by that window. Off by default: multi-node mounts otherwise
// keep exact close-to-open attributes.
// pfind/ls read the WHOLE listing before stat'ing anything, so the TTL must
// cover the full stat pass over a large directory (a 2 s TTL expired most of
// a 512k-entry prefetch before its first stat). CLIO_FUSE_READDIR_PREFETCH_TTL_S.
static int64_t PrefetchTtlNs() {
  static const int64_t ttl = [] {
    const char *e = getenv("CLIO_FUSE_READDIR_PREFETCH_TTL_S");
    const double s = (e != nullptr && *e != '\0') ? atof(e) : 60.0;
    return static_cast<int64_t>(s * 1e9);
  }();
  return ttl;
}
static std::mutex g_pf_mtx;
static std::unordered_map<std::string, std::pair<cte_stat_t, int64_t>> g_pf;
static int64_t PfNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
static int PrefetchWindow() {
  static const int w = [] {
    const char *e = getenv("CLIO_FUSE_READDIR_PREFETCH");
    return (e != nullptr && *e != '\0') ? atoi(e) : 0;
  }();
  return w;
}
static bool PrefetchTake(const std::string &p, cte_stat_t *st) {
  if (PrefetchWindow() <= 0) return false;
  std::lock_guard<std::mutex> lk(g_pf_mtx);
  auto it = g_pf.find(p);
  if (it == g_pf.end()) return false;
  const bool fresh = PfNowNs() - it->second.second < PrefetchTtlNs();
  if (fresh) *st = it->second.first;
  g_pf.erase(it);
  return fresh;
}
static void PrefetchDrop(const std::string &p) {
  if (PrefetchWindow() <= 0) return;
  std::lock_guard<std::mutex> lk(g_pf_mtx);
  g_pf.erase(p);
}

static bool DirAttrLookup(const std::string &p, cte_stat_t *stbuf) {
  if (g_dir_attr_ttl_s <= 0.0) return false;
  std::lock_guard<std::mutex> lk(g_dir_attr_mtx);
  auto it = g_dir_attr.find(p);
  if (it == g_dir_attr.end()) return false;
  if (std::chrono::steady_clock::now() >= it->second.expiry) {
    g_dir_attr.erase(it);
    return false;
  }
  *stbuf = it->second.st;
  return true;
}

/**
 * Remember a directory's attributes as the runtime just reported them.
 * @param p the directory path
 * @param st its attributes
 */
static void DirAttrStore(const std::string &p, const cte_stat_t &st) {
  if (g_dir_attr_ttl_s <= 0.0) return;
  const auto ttl = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(g_dir_attr_ttl_s));
  std::lock_guard<std::mutex> lk(g_dir_attr_mtx);
  if (g_dir_attr.size() > 65536) g_dir_attr.clear();  // bound it
  g_dir_attr[p] = DirAttrEntry{st, std::chrono::steady_clock::now() + ttl};
}

/** Drop every cached directory attribute (a rename moves whole subtrees). */
static void DirAttrClear() {
  std::lock_guard<std::mutex> lk(g_dir_attr_mtx);
  g_dir_attr.clear();
}

/**
 * Forget the cached attributes a change to \a p can make stale: \a p
 * itself (if it is a directory) and its parent (whose entries changed).
 * @param p the path being created, removed, renamed or changed
 */
static void DirAttrInvalidate(const std::string &p) {
  PrefetchDrop(p);
  if (g_dir_attr_ttl_s <= 0.0) return;
  std::lock_guard<std::mutex> lk(g_dir_attr_mtx);
  g_dir_attr.erase(p);
  const size_t slash = p.find_last_of('/');
  if (slash != std::string::npos) {
    g_dir_attr.erase(slash == 0 ? std::string("/") : p.substr(0, slash));
  }
}

void HiwaterRaise(const std::string &path, clio::run::u64 end) {
  {
    std::lock_guard<std::mutex> lk(g_hw_mtx);
    auto &v = g_hiwater[path];
    if (end > v) v = end;
    g_wtime[path] = WallNowNs();
    g_wtime_pre.erase(path);
  }
  // A write supersedes a pending utimens mtime (the 2 s overlay would
  // otherwise keep reporting the stamped time over the write).
  TimesOverlayDropMtime(path);
}

/**
 * Time of the last write(2) through this mount to `path` whose bytes may
 * not have reached the chimod yet.
 * @param path the file's current path
 * @return nanoseconds since the epoch, or 0 when nothing is outstanding
 */
clio::run::u64 WriteTimeFor(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  auto it = g_wtime.find(path);
  return it == g_wtime.end() ? 0 : it->second;
}
/**
 * Write time a size push carries for `path`: the last write(2), including
 * one an explicit utimens has since taken off the stat overlay.
 * @param path the file's current path
 * @return nanoseconds since the epoch, or 0 when nothing is outstanding
 */
clio::run::u64 PushTimeFor(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  auto it = g_wtime.find(path);
  if (it != g_wtime.end()) return it->second;
  auto pt = g_wtime_pre.find(path);
  return pt == g_wtime_pre.end() ? 0 : pt->second;
}
clio::run::u64 HiwaterFor(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  auto it = g_hiwater.find(path);
  return it == g_hiwater.end() ? 0 : it->second;
}
void HiwaterErase(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  g_hiwater.erase(path);
  g_wtime.erase(path);
  g_wtime_pre.erase(path);
}
/**
 * Drop a path's hiwater entry after `pushed` was adopted by the chimod --
 * unless a newer write raised it past that: a deferred close runs after the
 * application moved on, and erasing unconditionally lost the size of writes
 * made through a later handle (fsx saw the file shrink to the old size).
 * @param path the file's path
 * @param pushed the size the close just carried to the chimod
 */
void HiwaterEraseIfCovered(const std::string &path, clio::run::u64 pushed) {
  std::lock_guard<std::mutex> lk(g_hw_mtx);
  auto it = g_hiwater.find(path);
  if (it == g_hiwater.end() || it->second > pushed) return;
  g_hiwater.erase(it);
  g_wtime.erase(path);
  g_wtime_pre.erase(path);
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
  auto wt = g_wtime.find(from);
  if (wt != g_wtime.end()) {
    clio::run::u64 v = wt->second;
    g_wtime.erase(wt);
    auto &dst = g_wtime[to];
    if (v > dst) dst = v;
  }
  auto pt = g_wtime_pre.find(from);
  if (pt != g_wtime_pre.end()) {
    clio::run::u64 v = pt->second;
    g_wtime_pre.erase(pt);
    auto &dst = g_wtime_pre[to];
    if (v > dst) dst = v;
  }
}

CfsHandle *GetHandle(struct fuse_file_info *fi) {
  return reinterpret_cast<CfsHandle *>(fi->fh);
}

// Lazy closer: a release whose file this mount wrote carries the file's
// size (the hiwater) to the chimod off the caller's path. Handles with
// nothing to publish close detached immediately -- the common case for
// read-only opens.

/**
 * Map a filesystem task's return code to a FUSE result. The chimod answers
 * with errno values, but a task that failed in the runtime (an unreachable
 * node, a network timeout) carries a runtime code; passed through raw it
 * surfaced as a nonsense errno (mkdir: ERANGE). Those become EIO.
 * @param rc task return code
 * @return 0 or a negative errno
 */
static inline int FsErrno(int rc) {
  if (rc == 0) return 0;
  if (rc > 0 && rc < 4096) return -rc;
  return -EIO;
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
/** One hard-link group: its names as spelled, one per canonical name. */
using LinkGroup = std::set<std::string, CacheKeyLess>;
PathMap<std::shared_ptr<LinkGroup>> g_link_groups;

void LinkGroupJoin(const std::string &a, const std::string &b) {
  std::lock_guard<std::mutex> lk(g_lg_mtx);
  auto ita = g_link_groups.find(a);
  auto itb = g_link_groups.find(b);
  std::shared_ptr<LinkGroup> g;
  if (ita != g_link_groups.end() && itb != g_link_groups.end()) {
    g = ita->second;
    for (const auto &m : *itb->second) g->insert(m);
  } else if (ita != g_link_groups.end()) {
    g = ita->second;
  } else if (itb != g_link_groups.end()) {
    g = itb->second;
  } else {
    g = std::make_shared<LinkGroup>();
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

/**
 * The other names of `p`'s hard-link group (empty when it has none).
 * @param p a path
 * @return sibling paths, excluding `p`
 */
std::vector<std::string> LinkGroupSiblings(const std::string &p) {
  std::lock_guard<std::mutex> lk(g_lg_mtx);
  std::vector<std::string> out;
  auto it = g_link_groups.find(p);
  if (it == g_link_groups.end()) return out;
  for (const auto &m : *it->second) {
    if (CacheKey(m) != CacheKey(p)) out.push_back(m);
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
PathMap<TimesOverlay> g_times_overlay;

void TimesOverlaySet(const std::string &p, clio::run::u64 a,
                     clio::run::u64 m, clio::run::u64 c) {
  std::lock_guard<std::mutex> lk(g_to_mtx);
  g_times_overlay[p] = TimesOverlay{a, m, c, std::chrono::steady_clock::now()};
}

/**
 * Forget the mtime part of a path's utimens overlay (a later write owns it).
 * @param p the file's path
 */
void TimesOverlayDropMtime(const std::string &p) {
  std::lock_guard<std::mutex> lk(g_to_mtx);
  auto it = g_times_overlay.find(p);
  if (it != g_times_overlay.end()) it->second.mtime_ns = 0;
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

/** One entry of the lazy closer's queue. */
struct PendingClose {
  // kPublish: close-to-open size publish queued by flush (no server handle
  // involved); see cte_fuse_flush.
  enum Kind { kClose, kPublish };
  Kind kind = kClose;
  clio::run::u64 fh = 0;
  std::string path;
  clio::cte::core::TagId tag = clio::cte::core::TagId::GetNull();
  clio::run::u64 hiwater = 0;  // kClose: size to advance to (0 = none)
  // Time of the last write the hiwater covers, captured when the entry is
  // queued: by the time the closer runs it, an earlier entry's publish may
  // have dropped the path's write time, and the push would then stamp the
  // time it lands -- moving an mtime stat already reported (git re-hashed
  // every file of a checkout).
  clio::run::u64 wtime = 0;
};
std::mutex g_closer_mtx;
std::condition_variable g_closer_cv;
std::deque<PendingClose> g_closer_q;
std::thread g_closer;
bool g_closer_started = false;
bool g_closer_stop = false;
bool g_closer_busy = false;  // CloserMain is executing a popped entry
std::string g_closer_busy_path;  // the path of that entry (g_closer_busy)

// Failures of a background publish (kPublish), by path: the next flush or
// fsync of the path reports them, as a write-back error surfaces at fsync on
// a local filesystem.
std::mutex g_publish_err_mtx;
PathMap<int> g_publish_err;

/** Record a background publish failure for `path` (first one wins). */
void LatchPublishError(const std::string &path, int err) {
  std::lock_guard<std::mutex> lk(g_publish_err_mtx);
  g_publish_err.emplace(path, err);
}

/**
 * Take the background publish failure recorded for `path`, if any.
 * @param path the file's path
 * @return 0, or the negative errno to report
 */
int TakePublishError(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_publish_err_mtx);
  auto it = g_publish_err.find(path);
  if (it == g_publish_err.end()) return 0;
  const int e = it->second;
  g_publish_err.erase(it);
  return e;
}

/**
 * Publish one file for close-to-open from the closer: push the size its
 * page puts reached to the chimod. Its pages already landed: page puts are
 * synchronous. Errors are latched for the next flush/fsync of the path.
 * @param pc the kPublish entry
 */
void RunPublish(const PendingClose &pc) {
  if (pc.hiwater == 0 || pc.tag.IsNull()) return;
  auto t = CLIO_CFS_CLIENT->AsyncAdvanceSize(
      (static_cast<clio::run::u64>(pc.tag.major_) << 32) |
          static_cast<clio::run::u64>(pc.tag.minor_),
      pc.hiwater, pc.wtime);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    HLOG(kError, "clio_cte_fuse: publishing the size of {} failed (rc {}); "
         "reporting EIO", pc.path, t->GetReturnCode());
    LatchPublishError(pc.path, -EIO);
    return;
  }
  HiwaterEraseIfCovered(pc.path, pc.hiwater);
}

// Depth of RunCloserEntry on this thread. An entry can call back into a
// hook that drains the closer (a truncate): waiting for the closer there
// waited for this very entry, and every later close and truncate hung
// behind it.
thread_local int t_closer_entry_depth = 0;

/**
 * Execute one closer entry: a size publish, or a close carrying the size
 * this mount's page puts reached. Shared by the closer thread and the
 * synchronous barrier.
 * @param pc the entry
 */
void RunCloserEntry(const PendingClose &pc) {
  struct DepthGuard {
    DepthGuard() { ++t_closer_entry_depth; }
    ~DepthGuard() { --t_closer_entry_depth; }
  } depth_guard;
  if (pc.kind == PendingClose::kPublish) {
    RunPublish(pc);
    return;
  }
  if (pc.hiwater == 0) {
    CLIO_CFS_CLIENT->AsyncCloseDetached(pc.fh);
    return;
  }
  // AWAITED: the hiwater entry may only be dropped once the chimod has
  // durably adopted the size, or a getattr in the gap would read the stale
  // pre-close size. Erased only on success: a failed push (e.g. the inode's
  // home unreachable mid-partition) must leave the overlay to retry from.
  auto t = CLIO_CFS_CLIENT->AsyncClose(pc.fh, pc.hiwater);
  t.Wait();
  if (t->GetReturnCode() == 0) {
    HiwaterEraseIfCovered(pc.path, pc.hiwater);
  } else {
    HLOG(kError, "clio_cte_fuse: closer publishing the size of {} failed "
         "(rc {}); keeping the hiwater overlay for the next push",
         pc.path, t->GetReturnCode());
    LatchPublishError(pc.path, -EIO);
  }
}

void CloserMain() {
#ifdef __linux__
  pthread_setname_np(pthread_self(), "cfs-closer");
#endif
  std::unique_lock<std::mutex> lk(g_closer_mtx);
  while (!g_closer_stop) {
    if (g_closer_q.empty()) {
      g_closer_cv.wait_for(lk, std::chrono::milliseconds(50), [] {
        return g_closer_stop || !g_closer_q.empty();
      });
      continue;
    }
    PendingClose pc = std::move(g_closer_q.front());
    g_closer_q.pop_front();
    g_closer_busy = true;
    g_closer_busy_path = pc.path;
    lk.unlock();
    RunCloserEntry(pc);
    lk.lock();
    g_closer_busy = false;
    g_closer_busy_path.clear();
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

/**
 * Whether O_DIRECT opens bypass the kernel page cache. Off by default: an
 * O_DIRECT open normally takes the buffered path so a program can still mmap
 * the fd (#597). CLIO_FUSE_ODIRECT=1 honors O_DIRECT instead, for benchmarks
 * that must measure the filesystem rather than node memory (a timed reread
 * otherwise reads the page cache at memory speed).
 * @return true if O_DIRECT opens get per-file direct_io
 */
bool HonorODirect() {
  static const bool v = [] {
    const char *e = getenv("CLIO_FUSE_ODIRECT");
    return e != nullptr && *e == '1';
  }();
  return v;
}

void MaybeDirectIo(struct fuse_file_info *fi) {
  if (DirectWronlyEnabled() && (fi->flags & O_ACCMODE) == O_WRONLY) {
    fi->direct_io = 1;
  }
#ifdef O_DIRECT  // not defined on Windows, which builds this file too
  if (HonorODirect() && (fi->flags & O_DIRECT) != 0) {
    fi->direct_io = 1;
  }
#endif
}

// Synchronous barrier: process EVERY queued closer entry inline. Rename and
// unlink call this — they're rare, and a queued entry executing after the
// path moves is how every ghost/EEXIST class in the lock-cycle repro arose.
void CloserBarrier() {
  // Inside a closer entry: the entries before it have run, the ones after it
  // must run after it -- there is nothing this caller may wait for.
  if (t_closer_entry_depth > 0) return;
  std::deque<PendingClose> q;
  {
    // Also wait out an entry CloserMain already popped: it is invisible to
    // the swap but still mutating this path's server state.
    std::unique_lock<std::mutex> lk(g_closer_mtx);
    g_closer_cv.wait(lk, [] { return !g_closer_busy; });
    q.swap(g_closer_q);
  }
  for (auto &pc : q) RunCloserEntry(pc);
}

/**
 * Whether the asynchronous closer still holds an entry for `path` (queued,
 * or being executed). Local only: no task is sent.
 * @param path the path to look for
 * @return true if a close or utimens for `path` has not finished yet
 */
bool CloserPendingFor(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_closer_mtx);
  if (g_closer_busy && g_closer_busy_path == path) return true;
  for (const auto &pc : g_closer_q) {
    if (pc.path == path) return true;
  }
  return false;
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

namespace {
/** Set when cte_fuse_init could not bind the CLIO clients. */
std::atomic<bool> g_fuse_init_failed{false};

/**
 * Give up the mount after the clients failed to initialize. A FUSE init
 * callback cannot refuse a mount, so without this the mount stays live with
 * unbound clients and every request goes to pool 0.0, which the runtime holds
 * forever (a hang, not an error). Ending the session unmounts and makes
 * clio_cte_fuse exit non-zero (cte_fuse_init_failed()). Only a real mount has
 * a FUSE context; a direct call (test_fuse_ops) just records the failure.
 */
void FailMountInit() {
  g_fuse_init_failed.store(true, std::memory_order_release);
  if (!g_fuse_session_live.load(std::memory_order_acquire)) {
    return;
  }
  struct fuse_context *cx = fuse_get_context();
  if (cx != nullptr && cx->fuse != nullptr) {
    fuse_exit(cx->fuse);
  }
}
}  // namespace

bool cte_fuse_init_failed() {
  return g_fuse_init_failed.load(std::memory_order_acquire);
}

// ============================================================================
// FUSE lifecycle
// ============================================================================

/**
 * Ask the mounted volume for its case policy (its root reports it) and
 * adopt it for every path cache (CacheKey).
 * @return true for a case-insensitive volume
 */
static bool QueryVolumeCasePolicy() {
  auto t = CLIO_CFS_CLIENT->AsyncGetattr("/");
  t.Wait();
  const bool ci =
      t->GetReturnCode() == 0 &&
      (t->volume_flags_ & clio::cte::filesystem::kVolumeCaseInsensitive) != 0;
  g_case_insensitive.store(ci, std::memory_order_relaxed);
  return ci;
}

/**
 * Adopt the mounted volume's case policy. On a case-insensitive volume every
 * path cache keys on the canonical path (CacheKey), WinFsp treats the
 * spellings of a name as one file (one open node, one cache), and no
 * negative lookup is cached: creating "Foo" cannot invalidate a cached miss
 * on "foo".
 * @param conn the FUSE connection being initialized
 * @param cfg the FUSE configuration being initialized
 */
static void ApplyVolumeCasePolicy(struct fuse_conn_info *conn,
                                  struct fuse_config *cfg) {
  if (!QueryVolumeCasePolicy()) return;
  cfg->negative_timeout = 0;
#ifdef FUSE_CAP_CASE_INSENSITIVE
  conn->want |= FUSE_CAP_CASE_INSENSITIVE;
#else
  (void)conn;
#endif
  fprintf(stderr, "clio_cte_fuse: case-insensitive volume\n");
}

static void *cte_fuse_init(struct fuse_conn_info *conn,
                           struct fuse_config *cfg) {
  (void)conn;
  clio::cte::fuse_profile::Start();
  // Trust the inode numbers we report (st_ino in getattr, d_ino in readdir),
  // both derived from the tag id, instead of letting FUSE auto-generate them.
  // This makes stat and readdir agree on d_ino/st_ino (generic/637), and gives
  // hard-link aliases (which share a TagId) the same inode.
  cfg->use_ino = 1;
  // hard_remove stays OFF: libfuse then keeps an unlinked (or renamed-over)
  // open file reachable under a .fuse_hiddenXXX name. With it on, fstat of
  // such a file fails ESTALE -- the kernel's stat carries no file handle,
  // so libfuse needs a path and has none -- which breaks the open, unlink,
  // keep-using pattern of temp files. The price of hiding: a rename over a
  // name open on THIS mount is two server renames (hide, then replace), so
  // other nodes can see the name missing in between. Ops that do get a
  // null path are still handled (OpPath / HandlePath).
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
  // Multi-node mounts default to 0: with a TTL the kernel keeps the size it
  // last saw for up to the TTL, and neither stat nor open refreshes it, so a
  // reader on another node clamped reads to a pre-extension size (close-to-
  // open broken; the cross-node fsx model check diverged right after a
  // remote truncate). An explicit CLIO_FUSE_ATTR_CACHE_S opts back in.
  double attr_ttl = MultiNode() ? 0.0 : 1.0;
  if (const char *ttl_env = getenv("CLIO_FUSE_ATTR_CACHE_S")) {
    if (*ttl_env != '\0') attr_ttl = atof(ttl_env);
  }
  g_file_attr_ttl_s = attr_ttl > 0.0 ? attr_ttl : 0.0;
  // The ENTRY (name -> inode) cache is kept even on multi-node mounts. The
  // namespace is hash-sharded, so with a zero entry TTL every path component
  // of every syscall was its own round trip to that component's owner
  // (~0.7 ms each at 8 nodes): git's deep .git/objects paths made a single
  // clone >60x slower than on one node. A cached dentry cannot serve a stale
  // ANSWER here: this is the high-level API, which re-resolves the full path
  // at the owner for every getattr/open, so a name another node removed or
  // renamed fails ENOENT at once. Attributes (size) and negative entries
  // stay uncached on multi-node, so close-to-open and remote creates are
  // exact. CLIO_FUSE_ENTRY_CACHE_S overrides.
  double entry_ttl = MultiNode() ? 1.0 : attr_ttl;
  if (const char *ttl_env = getenv("CLIO_FUSE_ENTRY_CACHE_S")) {
    if (*ttl_env != '\0') entry_ttl = atof(ttl_env);
  }
  cfg->attr_timeout = attr_ttl;
  cfg->entry_timeout = entry_ttl;
  g_entry_ttl_s = entry_ttl;
  // Directory attribute cache (see DirAttrEntry). Single-node: on for
  // attr_ttl. Multi-node: off by default, opt in with
  // CLIO_FUSE_DIR_ATTR_CACHE_S. Why it is worth opting in: with attr_timeout
  // 0 and default_permissions the kernel re-fetches the attributes of every
  // ancestor directory on each path walk, so each syscall costs one getattr
  // per path component; this cache answers those from the adapter. Trade:
  // this mount's own changes invalidate the entry, but a change made through
  // another node (chmod/chown of a directory, entries added/removed changing
  // nlink/mtime, rmdir) can be seen stale for up to the TTL. File attributes
  // are never cached here. A set value applies on any mount; 0 disables and
  // negative values clamp to 0.
  g_dir_attr_ttl_s = MultiNode() ? 0.0 : attr_ttl;
  if (const char *dir_ttl_env = getenv("CLIO_FUSE_DIR_ATTR_CACHE_S")) {
    if (*dir_ttl_env != '\0') {
      g_dir_attr_ttl_s = std::max(0.0, atof(dir_ttl_env));
    }
  }
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
  // Read-ahead concurrency. The kernel sends read-ahead as "background"
  // requests and allows only max_background of them in flight per mount
  // (libfuse default 12), so with many readers each read is effectively one
  // synchronous round trip. CLIO_FUSE_MAX_BACKGROUND raises that limit (the
  // congestion threshold follows at 3/4). Unset: the libfuse default, as
  // before. Measured with 256, DRAM only, ior-easy-read: 2 nodes 9.0 -> 12.0
  // GiB/s, 8 nodes 24.3 -> 32.2 GiB/s; reads arrive as full 512 KiB requests
  // instead of ~280 KiB fragments. Asking for a larger read-ahead window
  // (conn->max_readahead) has no effect: the kernel keeps its own 512 KiB.
  if (const char *bg = getenv("CLIO_FUSE_MAX_BACKGROUND")) {
    const int v = atoi(bg);
    if (v > 0) {
      conn->max_background = static_cast<unsigned>(v);
      conn->congestion_threshold = static_cast<unsigned>(v * 3 / 4);
    }
  }
  RaInit();  // CLIO_FUSE_READAHEAD_PAGES (needs g_file_attr_ttl_s)
  HLOG(kInfo, "clio_cte_fuse: max_write {} max_readahead {} max_background {} "
       "congestion_threshold {}", conn->max_write, conn->max_readahead,
       conn->max_background, conn->congestion_threshold);

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
    fprintf(stderr, "ERROR: CLIO_INIT failed; giving up the mount\n");
    FailMountInit();
    return nullptr;
  }
  // Bind the composed filesystem pool (found by name) and the pools it is
  // configured with. Every FUSE op below routes through CLIO_CFS_CLIENT.
  if (!clio::cte::filesystem::CLIO_CFS_CLIENT_INIT()) {
    fprintf(stderr, "ERROR: filesystem client init failed; giving up the "
            "mount\n");
    FailMountInit();
    return nullptr;
  }
  // Every CTE call below (page blobs, capacity, fsync) goes through the
  // filesystem pool's own data chain, which CLIO_CFS_CLIENT_INIT bound from
  // the filesystem's config (CLIO_CFS_CLIENT->Cte()): no pool id is assumed.
  {
    const clio::run::PoolId data_pool = CLIO_CFS_CLIENT->Cte()->pool_id_;
    fprintf(stderr, "clio_cte_fuse: data pool %u.%u (filesystem chain)\n",
            data_pool.major_, data_pool.minor_);
  }
  // The write path puts page blobs itself, so it applies the
  // filesystem module's data-score rule itself; the module's yaml is the one
  // source of the setting.
  {
    clio::cte::core::PageScorePolicy policy;
    if (CLIO_CFS_CLIENT->GetPageScorePolicy(&policy)) {
      g_page_score = policy;
      if (policy.Enabled()) {
        fprintf(stderr, "clio_cte_fuse: file data score %.3f past the first "
                "%u page(s) of each file\n",
                static_cast<double>(policy.data_score_), policy.head_pages_);
      }
    }
  }
  ApplyVolumeCasePolicy(conn, cfg);
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
  if (QueryVolumeCasePolicy()) {
    fprintf(stderr, "clio_cte_fuse: case-insensitive volume\n");
  }
  return 0;
}

std::string cte_fuse_cache_key(const std::string &path) {
  return CacheKey(path);
}

bool cte_fuse_case_insensitive() {
  return g_case_insensitive.load(std::memory_order_relaxed);
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
// Windows compatibility: the cfs SYNC helpers (Read/Write/GetAttr) live in filesystem_client.h's Linux-only region. MSVC builds
// (only the wheels pipeline compiles this TU with WinFsp) route through the
// awaited async tasks — slower, but the SHM fast paths never existed there.
// ---------------------------------------------------------------------------
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

/**
 * Raise a stat's mtime/ctime to this mount's last unpublished write(2).
 * @param p     the file's path
 * @param stbuf the stat being filled; its times are only ever advanced
 */
static void OverlayWriteTime(const std::string &p, cte_stat_t *stbuf) {
  const clio::run::u64 wt = WriteTimeFor(p);
  if (wt == 0) return;
  const clio::run::u64 cur =
      static_cast<clio::run::u64>(stbuf->st_mtim.tv_sec) * 1000000000ULL +
      static_cast<clio::run::u64>(stbuf->st_mtim.tv_nsec);
  if (wt <= cur) return;
  stbuf->st_mtim.tv_sec = static_cast<time_t>(wt / 1000000000ULL);
  stbuf->st_mtim.tv_nsec = static_cast<long>(wt % 1000000000ULL);
  stbuf->st_ctim = stbuf->st_mtim;
}

/**
 * Fill a stat buffer from a file's attributes as the chimod reported them,
 * overlaying what this mount knows and the chimod does not yet: sizes of
 * sizes reached by page puts not yet published, and their write times.
 *
 * The size overlay must be read BEFORE the attributes were requested and
 * passed in as `hw_floor`: the lazy closer drops a path's hiwater as soon as
 * the chimod acknowledges the size push, so a Getattr answered just before
 * the push landed (size 0) and an overlay read just after it was dropped
 * (nothing) together reported a freshly written file as empty -- and the
 * kernel then served reads of it as EOF (xfstests generic/001 "corruption").
 * A hiwater dropped before the request was sent was already acknowledged, so
 * the reply carries that size.
 * @param a the attributes (a Getattr reply's, or a creating Open's)
 * @param p the path this mount's overlays are keyed by
 * @param stbuf the buffer to fill
 * @param hw_floor HiwaterFor(p) taken before the attributes were requested
 *        (0 when they come from no request)
 */
static void FillStatFromAttrs(const StatAttrs &a, const std::string &p,
                              cte_stat_t *stbuf, clio::run::u64 hw_floor) {
  const StatAttrs *t = &a;
  // Owner: a prior chown recorded an override (uid_/gid_ != 0xFFFFFFFF);
  // otherwise report the mounting user's uid/gid (files carry no stored owner).
  stbuf->st_uid =
      (t->uid != 0xFFFFFFFFu) ? static_cast<uid_t>(t->uid) : getuid();
  stbuf->st_gid =
      (t->gid != 0xFFFFFFFFu) ? static_cast<gid_t>(t->gid) : getgid();
  stbuf->st_ino = static_cast<ino_t>(t->ino);  // stable inode = packed TagId
  // A chmod/create recorded permission bits (t->mode_ != 0xFFFFFFFF) win over
  // the synthesized defaults; keep only the low 12 bits (perms + setuid/gid/sticky).
  const bool have_mode = (t->mode != 0xFFFFFFFFu);
  const unsigned int perm = t->mode & 07777u;
  if (t->is_dir) {
    stbuf->st_mode = S_IFDIR | (have_mode ? perm : 0755u);
    stbuf->st_nlink = 2;
  } else if (t->is_symlink) {
    stbuf->st_mode = S_IFLNK | 0777;
    stbuf->st_nlink = 1;
    stbuf->st_size = static_cast<cte_off_t>(t->size);  // target length
  } else {
    stbuf->st_mode = S_IFREG | (have_mode ? perm : 0644u);
    // POSIX link count, computed by the chimod inside the Getattr task —
    // the separate alias round trip this replaced ran on EVERY regular-file
    // stat (2+ times per file on a checkout).
    stbuf->st_nlink = static_cast<nlink_t>(t->nlink);
    // cte_off_t is off_t on Linux; the WinFsp shim maps it for Windows.
    stbuf->st_size = static_cast<cte_off_t>(t->size);
    // Direct page puts advance the chimod size only at close: overlay the
    // local hiwater in the meantime.
    const clio::run::u64 hw = std::max(hw_floor, HiwaterFor(p));
    if (hw > static_cast<clio::run::u64>(stbuf->st_size)) {
      stbuf->st_size = static_cast<cte_off_t>(hw);
    }
    OverlayWriteTime(p, stbuf);
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
  if (t->ctime != 0) {
    NsBitsToTimespec(t->ctime, stbuf->st_ctim.tv_sec, stbuf->st_ctim.tv_nsec);
  }
  // Fall back to ctime when mtime is unknown, so a valid file never reports
  // mtime at the epoch while it has a real ctime (merged from #680).
  clio::run::u64 mtime_ns = (t->mtime != 0) ? t->mtime : t->ctime;
  if (mtime_ns != 0) {
    NsBitsToTimespec(mtime_ns, stbuf->st_mtim.tv_sec, stbuf->st_mtim.tv_nsec);
  }
  if (t->atime != 0) {
    NsBitsToTimespec(t->atime, stbuf->st_atim.tv_sec, stbuf->st_atim.tv_nsec);
  }
}

/**
 * Fill a stat buffer from a Getattr reply (see FillStatFromAttrs).
 * @param t the Getattr reply (exists_ != 0)
 * @param p the path this mount's overlays are keyed by
 * @param stbuf the buffer to fill
 * @param hw_floor HiwaterFor(p) taken before the Getattr was sent
 */
static void FillStatFromGetattr(clio::cte::filesystem::GetattrTask *t,
                                const std::string &p, cte_stat_t *stbuf,
                                clio::run::u64 hw_floor) {
  StatAttrs a;
  a.is_dir = t->is_dir_ != 0;
  a.is_symlink = t->is_symlink_ != 0;
  a.size = t->size_;
  a.ino = t->ino_;
  a.ctime = t->ctime_;
  a.mtime = t->mtime_;
  a.atime = t->atime_;
  a.uid = t->uid_;
  a.gid = t->gid_;
  a.mode = t->mode_;
  a.nlink = t->nlink_;
  FillStatFromAttrs(a, p, stbuf, hw_floor);
}

/** How long attributes kept from a create may wait for the getattr of the
 *  same request. That getattr follows at once; this only keeps a frontend
 *  that never makes it from leaving them to a much later fstat. */
static constexpr std::chrono::milliseconds kOpenAttrMaxAge{20};

/**
 * Keep the attributes an Open returned for the file it just CREATED, for
 * the getattr(path, fi) that follows inside the same create request. Not
 * for a file that already existed (others may hold it open, with writes in
 * flight) and not when the reply carries none.
 * @param h the new open file
 * @param t the Open reply
 */
static void KeepOpenAttrs(CfsHandle *h, clio::cte::filesystem::OpenTask *t) {
  if (t->created_ == 0 || t->has_attr_ == 0) return;
  StatAttrs &a = h->open_attr;
  a.size = t->size_;
  a.ino = t->ino_;
  a.ctime = t->ctime_;
  a.mtime = t->mtime_;
  a.atime = t->atime_;
  a.uid = t->uid_;
  a.gid = t->gid_;
  a.mode = t->attr_mode_;
  a.nlink = t->nlink_;
  h->open_attr_thread = std::this_thread::get_id();
  h->open_attr_at = std::chrono::steady_clock::now();
  h->open_attr_epoch = g_setattr_epoch.load();
  h->open_attr_ok.store(true);
}

/**
 * Take the attributes KeepOpenAttrs kept, if they still describe the file:
 * at most once per handle, only on the thread that created the file (the
 * FUSE layer stats it there, in the same request), only right away, and
 * only if nothing was written, truncated or set through this mount since.
 * Whatever the answer, the handle keeps none afterwards.
 * @param h the open file
 * @param out receives the attributes
 * @return true when out may answer this getattr
 */
static bool TakeOpenAttrs(CfsHandle *h, StatAttrs *out) {
  if (!h->open_attr_ok.exchange(false)) return false;
  if (h->open_attr_thread != std::this_thread::get_id()) return false;
  if (std::chrono::steady_clock::now() - h->open_attr_at > kOpenAttrMaxAge) {
    return false;
  }
  if (h->open_attr_epoch != g_setattr_epoch.load()) return false;
  if (h->wrote.load() || h->appended.load()) return false;
  *out = h->open_attr;
  return true;
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

  if (DirAttrLookup(p, stbuf)) return 0;

  // A tombstone in the filesystem's path mirror answers ENOENT with no task.
  auto *cfs = CLIO_CFS_CLIENT;
  {
    clio::cte::filesystem::ShmFileRecord mrec;
    if (cfs != nullptr && (cfs->HasShmCache() || cfs->AttachShmCache()) &&
        cfs->TryGetFileRecordShm(p, &mrec) && !mrec.Exists()) {
      return -ENOENT;
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
    auto *cte1 = CLIO_CFS_CLIENT->Cte();
    clio::cte::filesystem::ShmFileRecord self;
    bool self_absent = false;
    // Only a PROVEN absence counts: a torn read of a record being rewritten
    // is a miss, not evidence (see TryGetFileRecordShm).
    if (cfs != nullptr && cte1 != nullptr && !cfs->MirrorSaturated() &&
        !cfs->TryGetFileRecordShm(p, &self, &self_absent) && self_absent) {
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

  if (PrefetchTake(p, stbuf)) return 0;
  // Delegate to the filesystem chimod: it owns exists/is-dir/logical-size.
  // The size overlay is read first (see FillStatFromAttrs).
  const clio::run::u64 hw_floor = HiwaterFor(p);
  auto t = cfs->AsyncGetattr(p);
  t.Wait();
  // A lookup that FAILED (its directory block's home unreachable, a timeout)
  // is EIO, never ENOENT: "no such file" for an existing file invites the
  // application to create it again over the real one.
  if (t->GetReturnCode() != 0) {
    return FsErrno(t->GetReturnCode());
  }
  if (t->exists_ == 0) {
    return -ENOENT;
  }
  FillStatFromGetattr(t.get(), p, stbuf, hw_floor);
  if (S_ISDIR(stbuf->st_mode)) DirAttrStore(p, *stbuf);
  return 0;
}

static int FlushAppends(CfsHandle *handle, const std::string &hp);

/**
 * Whether the path an open file is known by still names that file here:
 * the mount's mirror record for the path carries the handle's inode. Then
 * the path-keyed fast paths (mirror stat, mirror size) describe the right
 * file; after the name was renamed over or removed they describe another
 * file or none.
 * @param h the open file
 * @param hp its current path (HandlePath)
 * @return true when the mirror confirms the path still names h's inode
 */
static bool PathStillNamesHandle(CfsHandle *h, const std::string &hp) {
  auto *cfs = CLIO_CFS_CLIENT;
  clio::cte::filesystem::ShmFileRecord rec;
  if (cfs == nullptr || !(cfs->HasShmCache() || cfs->AttachShmCache()) ||
      !cfs->TryGetFileRecordShm(hp, &rec) || !rec.Exists()) {
    return false;
  }
  return rec.tag_id_.major_ == h->tag.major_ &&
         rec.tag_id_.minor_ == h->tag.minor_;
}

/**
 * Ask the chimod for an open file's attributes by its inode id -- not by a
 * path, which a concurrent rename may have pointed at another file (a
 * reader of a file being replaced by write-temp + rename saw ENOENT or
 * another file's size).
 * @param h the open file (its tag set)
 * @param hp its current path (unused: the lookup is by inode id)
 * @param t out: the reply
 * @return 0, -ENOENT (the inode is gone) or -EIO
 */
static int GetattrOpenHandle(CfsHandle *h, const std::string &hp,
                             clio::run::Future<
                                 clio::cte::filesystem::GetattrTask> *t) {
  (void)hp;
  *t = CLIO_CFS_CLIENT->AsyncGetattr(
      clio::cte::filesystem::FsIdStatPath(PackTag(h->tag)));
  t->Wait();
  if ((*t)->GetReturnCode() != 0) return -EIO;
  return (*t)->exists_ != 0 ? 0 : -ENOENT;
}

/**
 * The size a read through an open file clamps to (before this mount's own
 * overlays): the path's fast answer while the path still names the file,
 * else the file's own by inode id.
 * @param h the open file (its tag set)
 * @param hp its current path
 * @param size out: the file's logical size
 * @return 0, -ENOENT or -EIO
 */
static int OpenHandleSize(CfsHandle *h, const std::string &hp,
                          clio::run::u64 *size) {
  if (PathStillNamesHandle(h, hp)) {
    bool exists = false;
    CfsGetAttrCompat(CLIO_CFS_CLIENT, hp, &exists, size);
    // Re-check: a rename between the mirror check and the lookup may have
    // made the path answer for another file.
    if (exists && PathStillNamesHandle(h, hp)) return 0;
  }
  clio::run::Future<clio::cte::filesystem::GetattrTask> t;
  const int rc = GetattrOpenHandle(h, hp, &t);
  *size = rc == 0 ? t->size_ : 0;
  return rc;
}

/**
 * The size a read of [off, off+len) through `h` clamps to. With a file
 * attribute TTL (CLIO_FUSE_ATTR_CACHE_S > 0) the size fetched by an earlier
 * read of this handle is reused while it is younger than the TTL, no
 * truncate/setattr went through this mount since, and the read ends inside
 * it; a read reaching the kept size asks again, so end-of-file is always
 * decided on a fresh answer. Without a TTL every read asks, as before: on a
 * multi-node mount the lookup is a round trip to the file's home whenever
 * this node's mirror has no record of the file, and it was 28% of a 284 KiB
 * read (8 nodes, CLIO_EVLAT rd_size 262 us of rd_total 934 us).
 * @param h the open file
 * @param hp its current path
 * @param off read offset
 * @param len read length
 * @param size out: the file's logical size
 * @return 0, -ENOENT or -EIO
 */
static int ReadClampSize(CfsHandle *h, const std::string &hp,
                         clio::run::u64 off, clio::run::u64 len,
                         clio::run::u64 *size) {
  const double ttl = g_file_attr_ttl_s;
  const clio::run::u64 epoch = g_setattr_epoch.load(std::memory_order_relaxed);
  const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  if (ttl > 0.0) {
    const int64_t at = h->rd_size_ns.load(std::memory_order_acquire);
    const clio::run::u64 kept = h->rd_size.load(std::memory_order_relaxed);
    if (at != 0 && static_cast<double>(now_ns - at) < ttl * 1e9 &&
        h->rd_size_epoch.load(std::memory_order_relaxed) == epoch &&
        off + len <= kept) {
      *size = kept;
      return 0;
    }
  }
  const int rc = OpenHandleSize(h, hp, size);
  if (ttl > 0.0 && rc == 0) {
    h->rd_size.store(*size, std::memory_order_relaxed);
    h->rd_size_epoch.store(epoch, std::memory_order_relaxed);
    h->rd_size_ns.store(now_ns, std::memory_order_release);
  }
  return rc;
}

static int GetattrStatImpl(const char *path, cte_stat_t *stbuf,
                           struct fuse_file_info *fi);

int cte_fuse_getattr_stat(const char *path, cte_stat_t *stbuf,
                          struct fuse_file_info *fi) {
  // A getattr that took seconds, with its answer (#1169: a stat took 274 s
  // with two nodes down and no read-path diagnostic fired). Rate-limited.
  const auto t0 = std::chrono::steady_clock::now();
  const int rc = GetattrStatImpl(path, stbuf, fi);
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  if (ms > 2000.0) {
    static std::atomic<clio::run::u64> logged{0};
    const clio::run::u64 k = logged.fetch_add(1, std::memory_order_relaxed);
    if (k < 16 || k % 256 == 0) {
      HLOG(kWarning, "clio_cte_fuse: slow getattr of '{}': {} ms, rc {} ({} "
           "such)", path != nullptr ? path : "", ms, rc, k + 1);
    }
  }
  return rc;
}

static int GetattrStatImpl(const char *path, cte_stat_t *stbuf,
                           struct fuse_file_info *fi) {
  // fstat through a handle with deferred appends sees them.
  CfsHandle *handle = fi != nullptr ? GetHandle(fi) : nullptr;
  std::string hidden;
  if (handle == nullptr) {
    const int hrc = MapHiddenPath(&path, &hidden);
    if (hrc != 0) return hrc;
  }
  std::string key = path != nullptr ? std::string(path) : std::string();
  if (handle != nullptr) {
    key = HandlePath(handle, path);
    const int aerr = FlushAppends(handle, key);
    if (aerr != 0) return aerr;
  }
  // The first getattr through a handle uses up whatever its create kept,
  // whichever branch below answers it.
  StatAttrs kept;
  const bool have_kept = handle != nullptr && TakeOpenAttrs(handle, &kept);
  int rc = 0;
  if (handle != nullptr && !handle->tag.IsNull() &&
      !PathStillNamesHandle(handle, key)) {
    // fstat: the descriptor's own file, even if its name now leads
    // elsewhere (renamed over) or nowhere (unlinked while open).
    if (have_kept) {
      // The create that opened it returned these in this same request: no
      // task.
      memset(stbuf, 0, sizeof(*stbuf));
      FillStatFromAttrs(kept, key, stbuf, 0);
    } else {
      clio::run::Future<clio::cte::filesystem::GetattrTask> t;
      const clio::run::u64 hw_floor = HiwaterFor(key);
      rc = GetattrOpenHandle(handle, key, &t);
      if (rc == 0) {
        memset(stbuf, 0, sizeof(*stbuf));
        FillStatFromGetattr(t.get(), key, stbuf, hw_floor);
      }
    }
  } else {
    rc = cte_fuse_getattr_stat_inner(key.c_str(), stbuf, fi);
    if (rc == -ENOENT && handle != nullptr && !handle->tag.IsNull()) {
      // The name was renamed over between the mirror check and the lookup:
      // the descriptor's own file is still there.
      clio::run::Future<clio::cte::filesystem::GetattrTask> t;
      const clio::run::u64 hw_floor = HiwaterFor(key);
      rc = GetattrOpenHandle(handle, key, &t);
      if (rc == 0) {
        memset(stbuf, 0, sizeof(*stbuf));
        FillStatFromGetattr(t.get(), key, stbuf, hw_floor);
      }
    }
  }
  if (rc != 0) return rc;
  TimesOverlay ov;
  if (TimesOverlayGet(key, &ov)) {
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
// issue #1323: multi-node mounts keep a 1 s ENTRY cache, so after another
// node unlinks a name this kernel still holds a positive dentry for it. An
// open(O_CREAT) on that dentry revalidates it with GETATTR (not LOOKUP), and an
// ENOENT reply there fails the open outright: the kernel never falls back to
// LOOKUP + CREATE. Answer ESTALE ONCE for a path this mount resolved positive
// within the entry TTL: the VFS retries the walk with LOOKUP_REVAL, the forced
// LOOKUP gets the real ENOENT, and the open proceeds to CREATE (a plain stat
// ends in ENOENT as before). The one-shot erase keeps the REVAL retry from
// seeing ESTALE again.
static std::mutex g_pos_mtx;
static std::unordered_map<std::string, std::chrono::steady_clock::time_point>
    g_recent_pos;
static std::unordered_map<int64_t, std::chrono::steady_clock::time_point>
    g_estale_tid;  ///< last ESTALE per calling thread (guarded by g_pos_mtx)

static bool StaleOnVanished() {
#if defined(__linux__)
  return MultiNode() && g_entry_ttl_s > 0.0;
#else
  return false;
#endif
}

// Linux (struct stat) and Windows (WinFsp stat via cte_stat_t).
static int cte_fuse_getattr(const char *path, cte_stat_t *stbuf,
                            struct fuse_file_info *fi) {
  int rc = cte_fuse_getattr_stat(path, stbuf, fi);
  if (path == nullptr || fi != nullptr || !StaleOnVanished()) return rc;
  const auto now = std::chrono::steady_clock::now();
  const auto window = std::chrono::duration_cast<
      std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(g_entry_ttl_s * 2.0 + 0.5));
  std::lock_guard<std::mutex> lk(g_pos_mtx);
  if (rc == 0) {
    if (g_recent_pos.size() >= 65536) {
      for (auto it = g_recent_pos.begin(); it != g_recent_pos.end();) {
        it = (now - it->second > window) ? g_recent_pos.erase(it) : ++it;
      }
      if (g_recent_pos.size() >= 65536) g_recent_pos.clear();
    }
    g_recent_pos[path] = now;
  } else if (rc == -ENOENT) {
    auto it = g_recent_pos.find(path);
    if (it != g_recent_pos.end()) {
      bool fresh = now - it->second <= window;
      g_recent_pos.erase(it);
      // The VFS retries an ESTALE syscall exactly ONCE. A syscall that walks
      // two such paths (rename: source and target) would see a second ESTALE
      // on the retry and leak it to the application (dirs_rename_storm). Give
      // each calling thread at most one per second: its retry then always
      // sees the real answer.
      if (fresh) {
        const fuse_context *fc = fuse_get_context();
        const int64_t tid = fc != nullptr ? static_cast<int64_t>(fc->pid) : -1;
        auto tt = g_estale_tid.find(tid);
        if (tt != g_estale_tid.end() &&
            now - tt->second < std::chrono::seconds(1)) {
          fresh = false;
        } else {
          if (g_estale_tid.size() >= 4096) g_estale_tid.clear();
          g_estale_tid[tid] = now;
        }
      }
#if defined(__linux__)
      if (fresh) return -ESTALE;
#else
      (void)fresh;
#endif
    }
  }
  return rc;
}
#endif

int cte_fuse_utimens(const char *path_in, const cte_timespec_t tv[2],
                            struct fuse_file_info *fi) {
  g_setattr_epoch.fetch_add(1);  // attributes kept from a create are stale
  if (path_in != nullptr) DirAttrInvalidate(path_in);
  const std::string path_s = OpPath(path_in, fi);
  if (path_s.empty()) return -EBADF;
  const char *path = path_s.c_str();
  // An explicit utimens supersedes the write-time overlay (rsync/cp -a set
  // the source's mtime right after writing; the overlay must not win).
  {
    std::lock_guard<std::mutex> lk(g_hw_mtx);
    auto wt = g_wtime.find(path_s);
    if (wt != g_wtime.end()) {
      g_wtime_pre[path_s] = wt->second;
      g_wtime.erase(wt);
    }
  }
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
  // AWAITED by default. Fire-and-forget (CLIO_FUSE_ASYNC_UTIMENS=1) was
  // built for writeback-cache SETATTR storms, but writeback is off by
  // default, and a detached stamp REORDERS with the caller's next
  // operations: rsync -a / cp -a / tar restore a directory's mtime and a
  // later dir operation could land first and undo it (a second rsync
  // re-sent directories), and a missing path's stamp was dropped silently.
  static const bool async_utimens = [] {
    const char *e = getenv("CLIO_FUSE_ASYNC_UTIMENS");
    return e != nullptr && *e == '1' && !MultiNode();
  }();
  if (async_utimens) {
    std::string p(path);
    cfs->AsyncUtimensDetached(p, atime_ns, mtime_ns, flags);
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
  auto t = cfs->AsyncUtimens(std::string(path), atime_ns, mtime_ns, flags);
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return FsErrno(rc);
}

// chmod records the permission bits as a per-file override in the chimod
// (surfaced by getattr), mirroring chown. Storing the mode lets +x stick, so a
// binary copied onto the fs can be executed (generic/452) and chmod-then-proceed
// callers (e.g. mount's mtab updater in generic/089) still succeed. AsyncChmod
// resolves the path and returns ENOENT for a missing file, so no separate
// existence probe is needed.
int cte_fuse_chmod(const char *path, cte_mode_t mode,
                          struct fuse_file_info *fi) {
  g_setattr_epoch.fetch_add(1);  // attributes kept from a create are stale
  if (path != nullptr) DirAttrInvalidate(path);
  auto *cfs = CLIO_CFS_CLIENT;
  std::string p = OpPath(path, fi);
  if (p.empty()) return -EBADF;
  // Probe existence first: the chmod path resolves via GetOrCreateTag, so a
  // missing target would otherwise be silently created. chmod(2) must ENOENT.
  auto g = cfs->AsyncGetattr(p);
  g.Wait();
  if (g->GetReturnCode() != 0) return FsErrno(g->GetReturnCode());
  if (g->exists_ == 0) return -ENOENT;
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
  g_setattr_epoch.fetch_add(1);  // attributes kept from a create are stale
  if (path != nullptr) DirAttrInvalidate(path);
  auto *cfs = CLIO_CFS_CLIENT;
  const std::string p = OpPath(path, fi);
  if (p.empty()) return -EBADF;
  auto t = cfs->AsyncChown(p,
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

/**
 * File-type bits (S_IFMT) for a directory entry of the given filesystem type.
 * @param fs_type kFsType* as stored in the directory row
 * @return S_IFREG, S_IFDIR or S_IFLNK; 0 (DT_UNKNOWN) for anything else
 */
static cte_mode_t DirentMode(clio::run::u32 fs_type) {
  switch (fs_type) {
    case clio::cte::filesystem::kFsTypeFile: return S_IFREG;
    case clio::cte::filesystem::kFsTypeDir: return S_IFDIR;
    case clio::cte::filesystem::kFsTypeSymlink: return S_IFLNK;
    default: return 0;
  }
}

// ---------------------------------------------------------------------------
// Seekable, chunked listings.
//
// Every entry has a stable offset derived from its name (DirEntryOff in
// fs_inode_table.h): the same in every process and across opens, so
// telldir/seekdir work and a walker may hand a range of one directory to
// another process. The listing is ascending by offset and is fetched from
// the chimod in chunks of about kDirChunkEntries entries (ReaddirRange)
// instead of one reply holding the whole directory.
//
// Offsets the kernel sees: 0 = start, 1 after ".", 2 after "..", then for
// every entry the offset OF THE ENTRY THAT FOLLOWS IT (what telldir returns
// after reading it, and where a later readdir resumes), INT64_MAX after the
// last one.
// ---------------------------------------------------------------------------

namespace {
/** Entries asked for per ReaddirRange task. */
constexpr clio::run::u32 kDirChunkEntries = 8192;

/** One listed entry of a fetched chunk. */
struct DirChunkEntry {
  std::string name;         ///< basename as listed
  clio::run::u64 off = 0;   ///< the entry's own offset
  clio::run::u64 ino = 0;   ///< d_ino (never 0)
  clio::run::u32 type = 0;  ///< kFsType*
};

/** One fetched chunk: every listed entry with offset in [start, next). */
struct DirChunk {
  bool valid = false;
  clio::run::u64 start = 0;  ///< offset the chunk was fetched from
  clio::run::u64 next = clio::cte::filesystem::kDirOffEnd;  ///< next chunk
  std::vector<DirChunkEntry> entries;  ///< ascending by (off, name)
};

/**
 * Per-opendir state, kept in fuse_file_info::fh of a DIRECTORY handle.
 * The kernel keeps directory and file handles apart (opendir/readdir/
 * releasedir/fsyncdir versus open/read/.../release, and it sends no handle
 * with a directory's getattr or setattr), so this is never reached through
 * the CfsHandle casts of the file paths (GetHandle, OpPath).
 */
struct DirHandle {
  std::mutex mu;   ///< guards everything below
  DirChunk chunk;  ///< the chunk fetched last
  /** Equal offsets: the last entry emitted is followed by one with the SAME
   *  offset, so the kernel resumes at `resume_off` and `skip_count` entries
   *  of that offset are already out. 0 = no such resume pending. */
  clio::run::u64 resume_off = 0;
  size_t skip_count = 0;
};
}  // namespace

/**
 * Warm the stat cache for the entries of one fetched chunk (opt-in:
 * CLIO_FUSE_READDIR_PREFETCH sets the window of getattrs in flight).
 * @param dir the directory, for the slow-prefetch report
 * @param raw_entries entries the chunk held before filtering
 * @param fulls full path of every listed entry of the chunk
 */
static void PrefetchChunkStats(const std::string &dir, size_t raw_entries,
                               const std::vector<std::string> &fulls) {
  // Only chunks too large for the caller's own parallelism, and one
  // prefetch per mount at a time: pfind lists hundreds of directories at
  // once, and prefetching all of them (2000-entry mdtest-easy dirs too)
  // put ~65k getattrs in flight per node, saturated every runtime worker
  // and made a stat of the mount root take 61 s.
  static const size_t pf_min = [] {
    const char *e = getenv("CLIO_FUSE_READDIR_PREFETCH_MIN");
    return (e != nullptr && *e != '\0') ? static_cast<size_t>(atoll(e))
                                         : static_cast<size_t>(8192);
  }();
  static std::mutex pf_one;
  const int win = PrefetchWindow();
  if (win <= 0 || raw_entries < pf_min) return;
  std::unique_lock<std::mutex> pf_lk(pf_one, std::try_to_lock);
  if (!pf_lk.owns_lock()) return;
  auto *cfs = CLIO_CFS_CLIENT;
  using GetattrFuture = decltype(cfs->AsyncGetattr(dir));
  /** One prefetch in flight: its path, the size overlay read before it was
   *  sent (see FillStatFromAttrs), and its Getattr. */
  struct Inflight {
    std::string path;
    clio::run::u64 hw_floor;
    GetattrFuture fut;
  };
  std::vector<Inflight> inflight;
  inflight.reserve(static_cast<size_t>(win));
  auto drain = [&]() {
    for (auto &in : inflight) {
      in.fut.Wait();
      if (in.fut->GetReturnCode() != 0 || in.fut->exists_ == 0) continue;
      cte_stat_t pst;
      memset(&pst, 0, sizeof(pst));
      FillStatFromGetattr(in.fut.get(), in.path, &pst, in.hw_floor);
      std::lock_guard<std::mutex> lk(g_pf_mtx);
      g_pf[in.path] = {pst, PfNowNs()};
    }
    inflight.clear();
  };
  const int64_t pf_t0 = PfNowNs();
  for (const std::string &full : fulls) {
    const clio::run::u64 hw_floor = HiwaterFor(full);
    auto f = cfs->AsyncGetattr(full);
    inflight.push_back(Inflight{full, hw_floor, std::move(f)});
    if (inflight.size() >= static_cast<size_t>(win)) drain();
  }
  drain();
  if (raw_entries >= 4096) {
    HLOG(kWarning, "clio_cte_fuse: readdir prefetch of {} ({} entries) took {} ms",
         dir, raw_entries, (PfNowNs() - pf_t0) / 1000000);
  }
}

/**
 * Fetch the chunk of directory `p` that starts at offset `start`.
 * @param p directory path
 * @param start first offset wanted
 * @param c receives the chunk (invalid on failure)
 * @return 0 or a negative errno -- a failed listing is never an empty one
 */
static int FetchDirChunk(const std::string &p, clio::run::u64 start,
                         DirChunk *c) {
  c->valid = false;
  c->entries.clear();
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncReaddirRange(p, start, kDirChunkEntries);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    // Never report a listing that failed as an empty directory: rm -r
    // would unlink nothing and its rmdir would then fail ENOTEMPTY (#1029).
    HLOG(kError, "clio_cte_fuse: readdir of {} at offset {} failed (rc {})", p,
         start, t->GetReturnCode());
    return FsErrno(t->GetReturnCode());
  }
  const size_t n = t->entries_.size();
  if (t->offs_.size() != n) return -EIO;  // offsets are per entry
  // The chimod returns full paths; strip the directory prefix.
  size_t prefix_len = p.size();
  if (!p.empty() && p.back() != '/') prefix_len++;
  std::vector<std::string> fulls;
  c->entries.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    std::string full = t->entries_[i].str();
    std::string name = full.size() > prefix_len ? full.substr(prefix_len) : full;
    // A child sentinel directory may come back as "<dir>/<name>/"; drop the
    // trailing slash so it shows as a plain directory entry.
    if (!name.empty() && name.back() == '/') name.pop_back();
    if (name.empty()) continue;
    // libfuse keeps an unlinked-but-open file under .fuse_hiddenXXXX until
    // its last close -- and removes it only when the kernel's asynchronous
    // RELEASE arrives, after close(2) returned. The file has no name as far
    // as POSIX is concerned: listing it made `ls` show debris and let rm -r
    // trip over it.
    if (name.rfind(".fuse_hidden", 0) == 0) continue;
    DirChunkEntry e;
    e.off = t->offs_[i];
    e.type = i < t->types_.size() ? t->types_[i] : 0;
    e.ino = i < t->inos_.size() ? t->inos_[i] : 0;
    // d_ino = 0 means "deleted" to glibc's readdir(3), which silently DROPS
    // the entry — children whose ino lookup missed became invisible to ls
    // and rm -r while the server still listed them (generic/070's
    // undeletable "non-empty" dirs). Any nonzero value keeps the entry
    // visible; a name hash stays stable across listings.
    if (e.ino == 0) {
      clio::run::u64 h = 1469598103934665603ull;
      for (unsigned char ch : full) h = (h ^ ch) * 1099511628211ull;
      e.ino = h | 1;
    }
    e.name = std::move(name);
    c->entries.push_back(std::move(e));
    if (PrefetchWindow() > 0) {
      if (!full.empty() && full.back() == '/') full.pop_back();
      fulls.push_back(std::move(full));
    }
  }
  c->start = start;
  c->next = t->next_off_;
  c->valid = true;
  PrefetchChunkStats(p, n, fulls);
  return 0;
}

/**
 * Hand one entry to the kernel's fill callback.
 * @param buf the callback's buffer
 * @param filler the fill callback
 * @param e the entry
 * @param next_off offset to resume at after this entry (never 0)
 * @return the callback's result: non-zero when the reply buffer is full
 */
static int EmitDirent(void *buf, ClioFuseFillDirT filler,
                      const DirChunkEntry &e, clio::run::u64 next_off) {
  // Supply d_ino (the child's tag-derived inode) so getdents agrees with a
  // subsequent stat (generic/637), and d_type from the type stored in the
  // directory row (immutable for an inode). Without d_type a walker stats
  // every entry just to tell files from directories: pfind did 128,000
  // serial stats on one rank for one shared directory. An entry whose type
  // the chimod did not report stays DT_UNKNOWN (st_mode = 0).
  cte_stat_t st;
  memset(&st, 0, sizeof(st));
  st.st_mode = DirentMode(e.type);
  st.st_ino = static_cast<ino_t>(e.ino);
  const cte_off_t off = static_cast<cte_off_t>(next_off);
#ifdef __APPLE__
  // macFUSE's fill-dir callback consumes a fuse_darwin_attr, not a
  // struct stat — translate (same mapping getattr uses).
  struct fuse_darwin_attr attr;
  CopyStatToDarwinAttr(st, &attr);
  return filler(buf, e.name.c_str(), &attr, off,
                static_cast<fuse_fill_dir_flags>(0));
#else
  return filler(buf, e.name.c_str(), &st, off,
                static_cast<fuse_fill_dir_flags>(0));
#endif
}

/**
 * Emit the entries of `p` from offset `pos` on until the reply buffer is
 * full or the directory ends, fetching chunks as needed.
 * @param p directory path
 * @param dh listing state (its mutex held): an opendir handle, or a
 *        throwaway for a caller without one
 * @param pos first offset wanted (>= kDirOffFirst)
 * @param skip entries of offset `pos` to pass over: the ones this handle
 *        already emitted before the kernel resumed at `pos`
 * @param buf the fill callback's buffer
 * @param filler the fill callback
 * @return 0 or a negative errno
 */
static int EmitDirFrom(const std::string &p, DirHandle *dh,
                       clio::run::u64 pos, size_t skip, void *buf,
                       ClioFuseFillDirT filler) {
  using clio::cte::filesystem::kDirOffEnd;
  DirChunk &c = dh->chunk;
  for (;;) {
    if (!c.valid || pos < c.start || pos >= c.next) {
      if (const int rc = FetchDirChunk(p, pos, &c)) return rc;
    }
    const std::vector<DirChunkEntry> &v = c.entries;
    size_t i = static_cast<size_t>(
        std::lower_bound(v.begin(), v.end(), pos,
                         [](const DirChunkEntry &e, clio::run::u64 o) {
                           return e.off < o;
                         }) - v.begin());
    for (; skip > 0 && i < v.size() && v[i].off == pos; --skip) ++i;
    skip = 0;
    for (; i < v.size(); ++i) {
      // What the kernel resumes at after this entry: the NEXT entry's own
      // offset (a walker ends its range once telldir reaches the range's
      // end, so this must not be "own offset + 1").
      const clio::run::u64 next = i + 1 < v.size() ? v[i + 1].off : c.next;
      if (EmitDirent(buf, filler, v[i], next) != 0) return 0;  // buffer full
      if (next == v[i].off) {
        // The next entry collides with this one: the resume offset alone
        // cannot say how many of them are out.
        size_t first = i;
        while (first > 0 && v[first - 1].off == next) --first;
        dh->resume_off = next;
        dh->skip_count = i - first + 1;
      } else {
        dh->resume_off = 0;
        dh->skip_count = 0;
      }
    }
    if (c.next >= kDirOffEnd) return 0;  // end of directory
    if (c.next <= pos) {
      HLOG(kError, "clio_cte_fuse: readdir of {}: chunk at offset {} does "
           "not advance (next {})", p, pos, c.next);
      return -EIO;
    }
    pos = c.next;
  }
}

/**
 * opendir: allocate the listing state readdir keeps between calls.
 * @param path directory path (unused: the kernel resolved it)
 * @param fi receives the handle in fh
 * @return 0
 */
int cte_fuse_opendir(const char * /*path*/, struct fuse_file_info *fi) {
  fi->fh = reinterpret_cast<uint64_t>(new DirHandle());
  return 0;
}

/**
 * releasedir: free the state opendir allocated.
 * @param path directory path (unused)
 * @param fi the handle
 * @return 0
 */
int cte_fuse_releasedir(const char * /*path*/, struct fuse_file_info *fi) {
  if (fi != nullptr && fi->fh != 0) {
    delete reinterpret_cast<DirHandle *>(fi->fh);
    fi->fh = 0;
  }
  return 0;
}

/**
 * readdir: list a directory from `offset` on (see "Seekable, chunked
 * listings" above). Every fill call carries a non-zero offset, so libfuse
 * replies with what fits and calls again with the offset of the last entry
 * it took.
 * @param path directory path (null: removed while open -- it is empty)
 * @param buf the fill callback's buffer
 * @param filler the fill callback
 * @param offset 0 to start (and re-read) the listing, else an offset a fill
 *        call handed out, or any offset to seek to
 * @param fi the opendir handle (null or fh 0: no state is kept, every call
 *        fetches)
 * @param flags unused
 * @return 0 or a negative errno
 */
int cte_fuse_readdir(const char *path, void *buf,
                            ClioFuseFillDirT filler, cte_off_t offset,
                            struct fuse_file_info *fi,
                            enum fuse_readdir_flags flags) {
  using clio::cte::filesystem::kDirOffEnd;
  using clio::cte::filesystem::kDirOffFirst;
  (void)flags;
  const clio::run::u64 off =
      offset < 0 ? 0 : static_cast<clio::run::u64>(offset);
  if (off == 0 && filler(buf, ".", nullptr, 1,
                         static_cast<fuse_fill_dir_flags>(0)) != 0) {
    return 0;
  }
  if (off <= 1 && filler(buf, "..", nullptr, 2,
                         static_cast<fuse_fill_dir_flags>(0)) != 0) {
    return 0;
  }
  // hard_remove: a directory removed while open has no path; it is empty.
  if (path == nullptr || off >= kDirOffEnd) return 0;
  const std::string p(path);
  // A caller without a handle keeps nothing between calls: each call
  // fetches, and one that stops inside a run of equal offsets cannot resume
  // past it (its filler must take the whole run; the collectors that call
  // this without a handle take the whole directory in one call).
  DirHandle local;
  DirHandle *dh = (fi != nullptr && fi->fh != 0)
                      ? reinterpret_cast<DirHandle *>(fi->fh)
                      : &local;
  std::lock_guard<std::mutex> lk(dh->mu);
  if (off == 0) {
    // A (re)start re-reads the directory (rewinddir must show changes).
    dh->chunk.valid = false;
    dh->resume_off = 0;
    dh->skip_count = 0;
  }
  // Equal offsets: resuming exactly where this handle stopped inside a run
  // of entries that share an offset passes over the ones already out. Any
  // other arrival at that offset (a seek, another handle) gets them all.
  const size_t skip = (off >= kDirOffFirst && off == dh->resume_off)
                          ? dh->skip_count
                          : 0;
  return EmitDirFrom(p, dh, std::max(off, kDirOffFirst), skip, buf, filler);
}

/**
 * True when the last component of `path` exceeds NAME_MAX (255 bytes).
 * The kernel leaves component length to the filesystem; without this check
 * clio-fs accepted 256+ byte names that other POSIX filesystems (and tools
 * copying out of this one) reject with ENAMETOOLONG.
 * @param path absolute path inside the mount
 * @return true if the final name is too long
 */
static bool NameTooLong(const char *path) {
  const char *slash = strrchr(path, '/');
  const char *name = slash != nullptr ? slash + 1 : path;
  return strlen(name) > 255;
}

int cte_fuse_mkdir(const char *path, cte_mode_t mode) {
  DirAttrInvalidate(path);
  if (NameTooLong(path)) return -ENAMETOOLONG;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncMkdir(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // errno-style (0/EEXIST/EIO)
  if (rc != 0) return FsErrno(rc);
  // The kernel hands us the umask-applied mode; anything but the synthesized
  // default must be recorded (mkdir -m 700 / private temp dirs).
  const clio::run::u32 perm = static_cast<clio::run::u32>(mode) & 07777u;
  if (perm != 0755u) {
    auto c = cfs->AsyncChmod(std::string(path), perm);
    c.Wait();
  }
  return 0;
}

int cte_fuse_rmdir(const char *path) {
  DirAttrInvalidate(path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRmdir(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/ENOTEMPTY/ENOENT/EIO
  if (rc == ENOTEMPTY) {
    // #1029: say what the directory still holds that a listing hides
    // (.fuse_hidden names, pending/leaving entries) -- the server logs the
    // block contents for the same rmdir.
    auto l = cfs->AsyncReaddir(std::string(path));
    l.Wait();
    size_t shown = 0;
    if (l->GetReturnCode() == 0) {
      for (size_t i = 0; i < l->entries_.size(); ++i) {
        const std::string e = l->entries_[i].str();
        if (e.find("/.fuse_hidden") == std::string::npos) ++shown;
      }
    }
    if (shown == 0) {
      HLOG(kError, "clio_cte_fuse: rmdir {}: ENOTEMPTY but its listing "
           "(rc {}, {} raw entries) shows no child", path,
           l->GetReturnCode(), l->entries_.size());
    }
  }
  return FsErrno(rc);
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

/**
 * Apply an open's O_TRUNC to a file that already existed.
 *
 * A file the Open just created (`created`) is empty by construction, so its
 * O_TRUNC is skipped: the truncate is not free on a cluster -- it drains the
 * asynchronous closer (putting the PREVIOUS file's close on this open's
 * path) and sends a truncate to the inode's home, two dependent round trips
 * on every `open(O_CREAT|O_TRUNC)` of a new file (#1159).
 * @param cfs filesystem client (unused; the full truncate hook is used)
 * @param p the opened path
 * @param flags the open(2) flags
 * @param created nonzero when this Open created the file
 */
static inline void MaybeTruncateOnOpen(clio::cte::filesystem::Client *cfs,
                                       const std::string &p, int flags,
                                       clio::run::u32 created) {
  if (created != 0) {
    // A close of an earlier file by this name (removed on another node)
    // must still land before this file is used, as the truncate's closer
    // drain guaranteed. Rare, and checked locally.
    if ((flags & O_TRUNC) && CloserPendingFor(p)) CloserBarrier();
    return;
  }
  if (flags & O_TRUNC) {
    // The FULL truncate hook, not a raw AsyncTruncate: O_TRUNC on reopen
    // must drain the asynchronous closer before the truncate lands, or a
    // queued size push from the last open session lands AFTER it and
    // resurrects the old length —
    // generic/729's dio write@0 + fsync + close + O_TRUNC reopen read the
    // previous phase's page back out of the "empty" file.
    (void)cfs;
    cte_fuse_truncate(p.c_str(), 0, nullptr);
  }
}

int cte_fuse_create(const char *path, cte_mode_t mode,
                           struct fuse_file_info *fi) {
  if (path != nullptr) DirAttrInvalidate(path);
  if (NameTooLong(path)) return -ENAMETOOLONG;
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  // O_EXCL goes to the chimod, which decides it against every node's
  // creates (the kernel's negative lookup only covers this mount).
  auto t = cfs->AsyncOpen(p, O_CREAT | O_RDWR | (fi->flags & O_EXCL),
                          static_cast<clio::run::u32>(mode));
  t.Wait();
  // ENOENT: the parent is gone (e.g. an rmdir on another node won the race).
  if (t->GetReturnCode() != 0) return FsErrno(t->GetReturnCode());
  if (t->handle_ == 0) return -ENOENT;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  handle->tag = clio::cte::core::TagId(
      static_cast<clio::run::u32>(t->tag_packed_ >> 32),
      static_cast<clio::run::u32>(t->tag_packed_ & 0xffffffffULL));
  handle->append = (fi->flags & O_APPEND) != 0;
  handle->atime_done = (fi->flags & O_NOATIME) != 0;
  if (handle->append && MultiNode()) fi->direct_io = 1;  // offset is ours
  fi->fh = reinterpret_cast<uint64_t>(handle);
  RegisterOpenFile(handle, fi);
  MaybeDirectIo(fi);
  MaybeTruncateOnOpen(cfs, p, fi->flags, t->created_);
  if (fi->flags & O_TRUNC) HiwaterClamp(p, 0);
  // Last: the getattr(path, fi) the FUSE layer makes next, in this request,
  // is answered from the reply (only for a file this call created).
  KeepOpenAttrs(handle, t.get());
  return 0;
}

int cte_fuse_open(const char *path, struct fuse_file_info *fi) {
  // A hidden name is reached only through a dentry the kernel had not yet
  // moved off it: ESTALE makes the kernel look the real name up again.
  if (IsHiddenName(path)) return -kOpenVanishedErrno;
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  // The chimod honors O_CREAT: a plain open of a missing file returns
  // handle==0 so we can surface ENOENT.
  auto t = cfs->AsyncOpen(p, static_cast<clio::run::u32>(fi->flags), 0644);
  t.Wait();
  if (t->GetReturnCode() != 0) return FsErrno(t->GetReturnCode());
  if (t->handle_ == 0) {
    // The kernel only opens a file it just looked up: if it is gone now, a
    // rename over its name (or an unlink) won the race -- libfuse may even
    // have hidden and removed the old file under a .fuse_hidden name.
    // ESTALE makes the kernel look the path up again and open whatever it
    // names now (ENOENT from that lookup if nothing), where ENOENT failed
    // an open of a name that existed throughout (ext4 would open the old
    // file).
    return -kOpenVanishedErrno;
  }

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  handle->tag = clio::cte::core::TagId(
      static_cast<clio::run::u32>(t->tag_packed_ >> 32),
      static_cast<clio::run::u32>(t->tag_packed_ & 0xffffffffULL));
  handle->append = (fi->flags & O_APPEND) != 0;
  handle->atime_done = (fi->flags & O_NOATIME) != 0;
  if (handle->append && MultiNode()) fi->direct_io = 1;  // offset is ours
  fi->fh = reinterpret_cast<uint64_t>(handle);
  RegisterOpenFile(handle, fi);
  MaybeDirectIo(fi);
  MaybeTruncateOnOpen(cfs, p, fi->flags, t->created_);
  if (fi->flags & O_TRUNC) HiwaterClamp(p, 0);
  return 0;
}

/**
 * Land a file's deferred writes and publish its logical size to the
 * namespace home, synchronously. When replication is chained and
 * asynchronous, also forces the file's tag up to date on its persistent
 * replica(s) before returning -- see FlushReplicationBarrier.
 * @param handle the open file whose writes should become visible
 * @return 0, or a negative errno if the size (or the replication barrier)
 *         could not be published
 */
// ============================================================================
// O_APPEND
// ============================================================================

/** How an O_APPEND write finds its offset (CLIO_FUSE_APPEND). */
enum class AppendMode {
  kKernel,    ///< the kernel's cached size (one node: its size is exact)
  kReserve,   ///< reserve the end at the file's stream home, then write
  kDeferred,  ///< stage the bytes; the stream home merges them at the end
};

/**
 * The mount's O_APPEND mode: CLIO_FUSE_APPEND=kernel|reserve|deferred.
 * Default: kernel on one node, deferred on a multi-node mount (another
 * node's appends move the end past the kernel's cached size).
 * @return the mode
 */
static AppendMode AppendModeOf() {
  static const AppendMode mode = [] {
    const char *e = getenv("CLIO_FUSE_APPEND");
    const std::string v = e != nullptr ? e : "";
    if (v == "kernel") return AppendMode::kKernel;
    if (v == "reserve") return AppendMode::kReserve;
    if (v == "deferred") return AppendMode::kDeferred;
    return MultiNode() ? AppendMode::kDeferred : AppendMode::kKernel;
  }();
  return mode;
}

/** @return the stream pool client (file sizes and deferred appends). */
static clio::cte::stream::Client &StreamClient() {
  // The stream pool the filesystem is configured with (no fixed id).
  return *CLIO_CFS_CLIENT->Stream();
}

/**
 * Pack a tag id.
 * @param tag tag id
 * @return (major << 32) | minor
 */
static clio::run::u64 PackTag(const clio::cte::core::TagId &tag) {
  return (static_cast<clio::run::u64>(tag.major_) << 32) |
         static_cast<clio::run::u64>(tag.minor_);
}

// ============================================================================
// Sequential read-ahead (CLIO_FUSE_READAHEAD_PAGES)
// ============================================================================
//
// Opt-in. Each open handle that reads a file sequentially (RaSeqDetector,
// fuse_readahead.h) keeps the page its stream ends in and the next
// g_ra_pages pages fetched, or in flight, in buffers of its own, so the
// kernel's next read-ahead request is answered by a memcpy and the fetches
// of several pages -- each a round trip to another node on a multi-node
// mount -- overlap. Pages are fetched whole from the shm cache when it holds
// them, else with a GetBlob task (the read path's two paths), never past
// the size the read that started them clamped to.
//
// What a cached page may be stale against, and how each is bounded:
//  - writes, appends, truncates and fallocates through THIS mount, on any
//    handle: never. Each bumps a counter when it is done and before it is
//    acknowledged (RaInvalidateTag / RaInvalidateAll); a page fetched under
//    another counter value is not served. A fetch started while the write
//    was still running carries the old value, so it is dropped too.
//  - any setattr through this mount (g_setattr_epoch): never, same rule.
//  - deferred O_APPEND bytes this mount accepted but has not flushed: the
//    file's stream home merges them on its own schedule, so the cache is
//    bypassed for that file until they are flushed (RaAppendsPending).
//  - writes through OTHER nodes: a page is served for at most the file
//    attribute TTL (CLIO_FUSE_ATTR_CACHE_S) after its fetch was issued --
//    the same window in which the kernel already serves a stale size. With
//    that TTL 0 read-ahead is off whatever CLIO_FUSE_READAHEAD_PAGES says.
//
// Memory: at most g_ra_pages + 1 page buffers (1 MiB each) per open handle,
// counting pages whose fetch is still in flight; all freed at release.

/** Slots of the per-file invalidation counters; a file's tag hashes to one
 *  (two files sharing a slot only invalidate each other needlessly). */
static constexpr size_t kRaSlots = 1024;
/** Bumped after every write through this mount to a file of the slot. */
static std::atomic<clio::run::u64> g_ra_gen[kRaSlots];
/** Deferred appends accepted through this mount, per slot. */
static std::atomic<clio::run::u64> g_ra_append_seq[kRaSlots];
/** The g_ra_append_seq value known merged into the file, per slot. */
static std::atomic<clio::run::u64> g_ra_append_flushed[kRaSlots];
/** Bumped after every truncate / fallocate through this mount (any file). */
static std::atomic<clio::run::u64> g_ra_epoch{0};
/** 30 s waits a read gives a prefetch before fetching the bytes itself. */
static constexpr int kRaReadWaitSlices = 4;
/** 30 s waits release gives a prefetch that is still out. */
static constexpr int kRaReleaseWaitSlices = 20;

/**
 * The invalidation slot of a file.
 * @param tag the file's tag
 * @return index into the g_ra_* slot arrays
 */
static size_t RaSlot(const clio::cte::core::TagId &tag) {
  return static_cast<size_t>((PackTag(tag) * 0x9E3779B97F4A7C15ull) >> 54);
}

/**
 * Drop every read-ahead page of a file, on all handles: called when a write
 * to it through this mount is done, before the write is acknowledged.
 * @param tag the file's tag (null: nothing to do)
 */
static inline void RaInvalidateTag(const clio::cte::core::TagId &tag) {
  if (g_ra_pages == 0 || tag.IsNull()) return;
  g_ra_gen[RaSlot(tag)].fetch_add(1, std::memory_order_seq_cst);
}

/** Drop every read-ahead page of every file (truncate, fallocate: the file
 *  is not always known by tag there, and both are rare). */
static inline void RaInvalidateAll() {
  if (g_ra_pages == 0) return;
  g_ra_epoch.fetch_add(1, std::memory_order_seq_cst);
}

/** Invalidates a file's read-ahead pages when a write op returns, on every
 *  return path. */
struct RaTagGuard {
  clio::cte::core::TagId tag;  ///< the written file
  /** @param t the written file's tag */
  explicit RaTagGuard(const clio::cte::core::TagId &t) : tag(t) {}
  ~RaTagGuard() { RaInvalidateTag(tag); }
};

/** Invalidates all read-ahead pages when a truncate / fallocate returns. */
struct RaAllGuard {
  RaAllGuard() = default;
  ~RaAllGuard() { RaInvalidateAll(); }
};

/**
 * Note a deferred append this mount accepted: until it is flushed the file
 * changes whenever its stream home merges, so its reads bypass the cache.
 * @param tag the appended file
 */
static inline void RaAppendAccepted(const clio::cte::core::TagId &tag) {
  if (g_ra_pages == 0) return;
  g_ra_append_seq[RaSlot(tag)].fetch_add(1, std::memory_order_seq_cst);
}

/**
 * Taken BEFORE a flush of a file's deferred appends is sent: every append
 * counted in it was accepted before the flush, so the flush merges it.
 * @param tag the file
 * @return the mark to pass to RaAppendsFlushed
 */
static inline clio::run::u64 RaAppendMark(const clio::cte::core::TagId &tag) {
  if (g_ra_pages == 0) return 0;
  return g_ra_append_seq[RaSlot(tag)].load(std::memory_order_seq_cst);
}

/**
 * A flush of a file's deferred appends succeeded: the appends up to `mark`
 * are in the file, which changed -- drop its pages.
 * @param tag the file
 * @param mark RaAppendMark taken before the flush was sent
 */
static inline void RaAppendsFlushed(const clio::cte::core::TagId &tag,
                                    clio::run::u64 mark) {
  if (g_ra_pages == 0) return;
  std::atomic<clio::run::u64> &flushed = g_ra_append_flushed[RaSlot(tag)];
  clio::run::u64 cur = flushed.load(std::memory_order_seq_cst);
  while (cur < mark && !flushed.compare_exchange_weak(cur, mark)) {
  }
  RaInvalidateTag(tag);
}

/**
 * Whether deferred appends accepted through this mount may still be
 * unmerged for a file of this slot.
 * @param slot the file's slot (RaSlot)
 * @return true when the cache must be bypassed
 */
static inline bool RaAppendsPending(size_t slot) {
  return g_ra_append_seq[slot].load(std::memory_order_seq_cst) !=
         g_ra_append_flushed[slot].load(std::memory_order_seq_cst);
}

/** What a page was fetched under; it is served only while all still hold. */
struct RaToken {
  clio::run::u64 gen = 0;      ///< g_ra_gen of the file's slot
  clio::run::u64 setattr = 0;  ///< g_setattr_epoch
  clio::run::u64 epoch = 0;    ///< g_ra_epoch
  /** @return whether both tokens name the same state */
  bool operator==(const RaToken &o) const {
    return gen == o.gen && setattr == o.setattr && epoch == o.epoch;
  }
};

/**
 * The current invalidation state of a slot.
 * @param slot the file's slot
 * @return the token a page fetched now would carry
 */
static RaToken RaTokenNow(size_t slot) {
  RaToken t;
  t.gen = g_ra_gen[slot].load(std::memory_order_seq_cst);
  t.setattr = g_setattr_epoch.load(std::memory_order_seq_cst);
  t.epoch = g_ra_epoch.load(std::memory_order_seq_cst);
  return t;
}

/** @return steady-clock nanoseconds */
static int64_t RaNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/** One page of a handle's read-ahead cache. Guarded by RaCache::mu, except
 *  `fut` and the bytes of `buf`, which belong to whoever holds the page in
 *  kIssuing / kWaiting until it leaves that state. */
struct RaPage {
  enum class State {
    kIssuing,   ///< a thread is sending the fetch (no future yet)
    kInFlight,  ///< fetch sent; nobody is waiting on it
    kWaiting,   ///< a thread is waiting on the fetch
    kReady,     ///< buf holds the page's first `len` bytes
    kFailed     ///< the fetch failed: never served
  };
  clio::run::u64 page = 0;  ///< page number
  clio::run::u64 len = 0;   ///< bytes fetched: RaPageLen at issue
  std::unique_ptr<char[]> buf;  ///< kFsPageSize bytes; zeros past the data
  clio::run::Future<clio::cte::core::GetBlobTask> fut;  ///< the fetch
  State state = State::kIssuing;
  int pins = 0;         ///< threads using the page: it is not freed
  bool in_map = false;  ///< still in RaCache::pages (else a zombie)
  RaToken token;        ///< invalidation state when the fetch was issued
  int64_t issued_ns = 0;  ///< RaNowNs when the fetch was issued
};

/** The read-ahead cache of one open handle. */
struct RaCache {
  std::mutex mu;  ///< guards everything below; never held across a fetch
  std::condition_variable cv;  ///< a page left kIssuing / kWaiting / a pin
  size_t slot = 0;             ///< the file's invalidation slot
  RaSeqDetector seq;           ///< this handle's stream
  std::map<clio::run::u64, std::unique_ptr<RaPage>> pages;  ///< by page no.
  /// Pages dropped from `pages` whose buffer cannot be reused yet: pinned,
  /// or their fetch is still out (the runtime may yet write the buffer).
  std::vector<std::unique_ptr<RaPage>> zombies;
  std::vector<std::unique_ptr<char[]>> free_bufs;  ///< buffers not in use
  clio::run::u32 allocated = 0;  ///< buffers that exist (<= g_ra_pages + 1)
};

/**
 * Whether a page may still be served: nothing invalidated it and it is
 * younger than the file attribute TTL.
 * @param c the cache (mu held)
 * @param pg the page
 * @param now_ns RaNowNs
 * @return true when the page's bytes are current enough to serve
 */
static bool RaPageValid(const RaCache &c, const RaPage &pg, int64_t now_ns) {
  return pg.token == RaTokenNow(c.slot) &&
         static_cast<double>(now_ns - pg.issued_ns) < g_file_attr_ttl_s * 1e9;
}

/**
 * Whether a page's buffer can be reused now: nobody uses the page and its
 * fetch is over (a completed fetch nobody waited on is simply dropped).
 * @param pg the page (cache mu held)
 * @return true when the page can be freed
 */
static bool RaPageIdle(const RaPage &pg) {
  if (pg.pins != 0) return false;
  if (pg.state == RaPage::State::kReady || pg.state == RaPage::State::kFailed) {
    return true;
  }
  return pg.state == RaPage::State::kInFlight && pg.fut.IsComplete();
}

/**
 * Free an idle page: its buffer goes back to the cache's free list.
 * @param c the cache (mu held)
 * @param pg the page (RaPageIdle)
 */
static void RaFreePageLocked(RaCache &c, std::unique_ptr<RaPage> pg) {
  std::unique_ptr<char[]> buf = std::move(pg->buf);
  pg.reset();  // the fetch's future goes first: it is complete
  c.free_bufs.push_back(std::move(buf));
}

/**
 * Take a page out of the cache: freed now if idle, else kept as a zombie
 * until its fetch completes and its users are gone.
 * @param c the cache (mu held)
 * @param it the page's entry in c.pages
 */
static void RaRetireLocked(
    RaCache &c,
    std::map<clio::run::u64, std::unique_ptr<RaPage>>::iterator it) {
  std::unique_ptr<RaPage> pg = std::move(it->second);
  c.pages.erase(it);
  pg->in_map = false;
  if (RaPageIdle(*pg)) {
    RaFreePageLocked(c, std::move(pg));
  } else {
    c.zombies.push_back(std::move(pg));
  }
}

/**
 * Take `pg` out of the cache if it is still in it.
 * @param c the cache (mu held)
 * @param pg a pinned page of c
 */
static void RaRetirePageLocked(RaCache &c, RaPage *pg) {
  if (!pg->in_map) return;
  auto it = c.pages.find(pg->page);
  if (it != c.pages.end() && it->second.get() == pg) RaRetireLocked(c, it);
}

/**
 * Free every zombie that became idle.
 * @param c the cache (mu held)
 */
static void RaReapLocked(RaCache &c) {
  for (size_t i = 0; i < c.zombies.size();) {
    if (RaPageIdle(*c.zombies[i])) {
      std::unique_ptr<RaPage> pg = std::move(c.zombies[i]);
      c.zombies[i] = std::move(c.zombies.back());
      c.zombies.pop_back();
      RaFreePageLocked(c, std::move(pg));
    } else {
      ++i;
    }
  }
}

/**
 * Take every page out of the cache (a non-sequential read, release).
 * @param c the cache (mu held)
 */
static void RaRetireAllLocked(RaCache &c) {
  while (!c.pages.empty()) RaRetireLocked(c, c.pages.begin());
}

/**
 * Drop a pin. `pg` may be freed by this call.
 * @param c the cache (mu held)
 * @param pg the pinned page
 */
static void RaUnpinLocked(RaCache &c, RaPage *pg) {
  if (--pg->pins == 0 && !pg->in_map) {
    RaReapLocked(c);
    c.cv.notify_all();
  }
}

/**
 * A buffer for a new page, within the handle's budget of g_ra_pages + 1:
 * a free one, a new one, or that of the lowest idle page outside `keep`.
 * @param c the cache (mu held)
 * @param keep the pages wanted now (never evicted)
 * @return the buffer, or null when the budget is used by pages in use
 */
static std::unique_ptr<char[]> RaTakeBufLocked(RaCache &c,
                                               const RaWindow &keep) {
  if (c.free_bufs.empty() && c.allocated < g_ra_pages + 1) {
    char *raw = new (std::nothrow) char[clio::cte::filesystem::kFsPageSize];
    if (raw == nullptr) return nullptr;
    ++c.allocated;
    return std::unique_ptr<char[]>(raw);
  }
  if (c.free_bufs.empty()) {
    for (auto it = c.pages.begin(); it != c.pages.end(); ++it) {
      if (!keep.Has(it->first) && RaPageIdle(*it->second)) {
        RaRetireLocked(c, it);
        break;
      }
    }
  }
  if (c.free_bufs.empty()) return nullptr;
  std::unique_ptr<char[]> buf = std::move(c.free_bufs.back());
  c.free_bufs.pop_back();
  return buf;
}

/**
 * Wait for a page's fetch. The caller owns the wait (page in kWaiting,
 * pinned) and holds no lock.
 * @param h the open file (for the log)
 * @param pg the page
 * @param slices 30 s waits to give it
 * @param rc out: the fetch's return code when it completed
 * @return true when the fetch completed, false when it is still out
 */
static bool RaAwaitFetch(CfsHandle *h, RaPage *pg, int slices, int *rc) {
  for (int slice = 1; !pg->fut.Wait(30.0f); ++slice) {
    HLOG(kWarning, "clio_cte_fuse: read-ahead of tag {}.{} page {} (len {}) "
         "still outstanding after {} wait(s) of 30 s", h->tag.major_,
         h->tag.minor_, pg->page, pg->len, slice);
    if (slice >= slices) return false;
  }
  *rc = static_cast<int>(pg->fut->GetReturnCode());
  return true;
}

/**
 * Bring a pinned page to rest: wait for its fetch -- as the one waiter, or
 * for the thread that already is -- so no second fetch of it is issued.
 * @param c the cache
 * @param h the open file
 * @param pg the pinned page
 * @param lk c.mu, held on entry and return, released while waiting
 * @return true when the page is kReady (its bytes can be copied)
 */
static bool RaSettleLocked(RaCache &c, CfsHandle *h, RaPage *pg,
                           std::unique_lock<std::mutex> &lk) {
  namespace ev = clio::run::evlat;
  using State = RaPage::State;
  if (pg->state == State::kReady || pg->state == State::kFailed) {
    return pg->state == State::kReady;
  }
  const unsigned long long ev_t0 = ev::Now();
  bool out = false;  // our wait gave up with the fetch still out
  while (!out && pg->state != State::kReady && pg->state != State::kFailed) {
    if (pg->state == State::kInFlight) {
      pg->state = State::kWaiting;
      lk.unlock();
      int rc = -1;
      const bool done = RaAwaitFetch(h, pg, kRaReadWaitSlices, &rc);
      lk.lock();
      // 0 = read, 1 = a hole (the zeroed buffer is the answer), as in the
      // uncached read; anything else is not served from here.
      pg->state = !done ? State::kInFlight
                        : (rc == 0 || rc == 1) ? State::kReady
                                               : State::kFailed;
      out = !done;
      c.cv.notify_all();
    } else {
      c.cv.wait(lk, [pg] {
        return pg->state != State::kWaiting && pg->state != State::kIssuing;
      });
    }
  }
  ev::Span(ev::kRaWait, ev_t0, ev::Now());
  return pg->state == State::kReady;
}

/**
 * Serve one page part of a read from the handle's read-ahead cache: copy it
 * if the page is cached, waiting for its fetch if that is in flight. On any
 * doubt (stale, too short, failed) the page is dropped and the caller
 * fetches the part the normal way.
 * @param c the handle's cache (RaPrepare)
 * @param h the open file
 * @param cur file offset of the part
 * @param page_off its offset inside its page
 * @param n its length
 * @param dst where the bytes go
 * @return true when dst holds the part
 */
static bool RaServe(RaCache *c, CfsHandle *h, clio::run::u64 cur,
                    clio::run::u64 page_off, clio::run::u64 n, char *dst) {
  namespace ev = clio::run::evlat;
  std::unique_lock<std::mutex> lk(c->mu);
  const bool active = c->seq.Active();
  RaPage *pg = nullptr;
  auto it = c->pages.find(cur / clio::cte::filesystem::kFsPageSize);
  if (it != c->pages.end()) {
    if (RaPageValid(*c, *it->second, RaNowNs()) &&
        it->second->state != RaPage::State::kFailed &&
        RaCovers(it->second->len, page_off, n)) {
      pg = it->second.get();
    } else {
      RaRetireLocked(*c, it);
    }
  }
  if (pg == nullptr) {
    if (active) ev::Count(ev::kRaMiss, 1);
    return false;
  }
  ++pg->pins;
  const bool ok = RaSettleLocked(*c, h, pg, lk);
  if (ok) {
    lk.unlock();
    std::memcpy(dst, pg->buf.get() + page_off, n);
    lk.lock();
  } else {
    RaRetirePageLocked(*c, pg);
  }
  RaUnpinLocked(*c, pg);
  if (ok) {
    ev::Count(ev::kRaHit, 1);
  } else if (active) {
    ev::Count(ev::kRaMiss, 1);
  }
  return ok;
}

/**
 * Start the fetch of one page into the cache. The page is entered first
 * (kIssuing) so a concurrent reader waits for it instead of fetching it
 * again, and its token is taken before the fetch is sent.
 * @param c the cache
 * @param h the open file
 * @param page page number (not in c.pages)
 * @param len bytes to fetch (RaPageLen)
 * @param buf the page's buffer
 * @param lk c.mu, held on entry and return, released while sending
 */
static void RaIssueLocked(RaCache &c, CfsHandle *h, clio::run::u64 page,
                          clio::run::u64 len, std::unique_ptr<char[]> buf,
                          std::unique_lock<std::mutex> &lk) {
  auto owned = std::make_unique<RaPage>();
  RaPage *pg = owned.get();
  pg->page = page;
  pg->len = len;
  pg->buf = std::move(buf);
  pg->pins = 1;
  pg->in_map = true;
  pg->token = RaTokenNow(c.slot);
  pg->issued_ns = RaNowNs();
  c.pages[page] = std::move(owned);
  lk.unlock();
  // Holes and short pages read as zeros, as in the uncached read.
  std::memset(pg->buf.get(), 0, len);
  auto *cte = CLIO_CFS_CLIENT->Cte();
  const std::string name =
      clio::cte::filesystem::PageName(page * clio::cte::filesystem::kFsPageSize);
  if (cte->TryGetCached(h->tag, name, 0, len, pg->buf.get())) {
    lk.lock();
    pg->state = RaPage::State::kReady;  // served by the shm cache: no task
    c.cv.notify_all();
    RaUnpinLocked(c, pg);
    return;
  }
  auto fut = cte->AsyncGetBlob(h->tag, name, 0, len, 0, pg->buf.get());
  lk.lock();
  if (fut.IsNull()) {
    pg->state = RaPage::State::kFailed;  // no staging memory: not sent
    RaRetirePageLocked(c, pg);
  } else {
    pg->fut = std::move(fut);
    pg->state = RaPage::State::kInFlight;
  }
  c.cv.notify_all();
  RaUnpinLocked(c, pg);
}

/**
 * Make sure the pages of `w` are cached or being fetched, as far as the
 * handle's buffer budget allows. Does not wait for any fetch.
 * @param c the cache
 * @param h the open file
 * @param w the pages wanted
 * @param fsize the size this read clamps to
 * @param lk c.mu, held on entry and return
 */
static void RaEnsureAhead(RaCache &c, CfsHandle *h, const RaWindow &w,
                          clio::run::u64 fsize,
                          std::unique_lock<std::mutex> &lk) {
  if (w.Empty()) return;
  const int64_t now_ns = RaNowNs();
  for (auto it = c.pages.begin(); it != c.pages.end();) {
    auto cur = it++;
    if (!RaPageValid(c, *cur->second, now_ns)) RaRetireLocked(c, cur);
  }
  RaReapLocked(c);
  for (clio::run::u64 q = w.first; q <= w.last; ++q) {
    if (c.pages.count(q) != 0) continue;
    const clio::run::u64 len =
        RaPageLen(q, fsize, clio::cte::filesystem::kFsPageSize);
    if (len == 0) break;
    std::unique_ptr<char[]> buf = RaTakeBufLocked(c, w);
    if (buf == nullptr) break;
    RaIssueLocked(c, h, q, len, std::move(buf), lk);
  }
}

/**
 * The handle's read-ahead cache, created at its first read.
 * @param h the open file (tag set)
 * @return the cache
 */
static RaCache *RaCacheOf(CfsHandle *h) {
  RaCache *c = h->ra.load(std::memory_order_acquire);
  if (c != nullptr) return c;
  auto *fresh = new RaCache();
  fresh->slot = RaSlot(h->tag);
  if (h->ra.compare_exchange_strong(c, fresh, std::memory_order_acq_rel)) {
    return fresh;
  }
  delete fresh;  // another reader of this handle created it
  return c;
}

/**
 * Read-ahead step of a read: classify it against the handle's stream and,
 * when the stream is sequential, start fetching the pages ahead of it.
 * @param h the open file (tag set)
 * @param off the read's offset (inside the file)
 * @param want the read's clamped length (> 0)
 * @param fsize the size the read clamps to
 * @return the cache to serve the read's page parts from, or null when the
 *         read must take the normal path for all of them
 */
static RaCache *RaPrepare(CfsHandle *h, clio::run::u64 off,
                          clio::run::u64 want, clio::run::u64 fsize) {
  RaCache *c = RaCacheOf(h);
  std::unique_lock<std::mutex> lk(c->mu);
  const RaSeq cls =
      c->seq.Observe(off, want, clio::cte::filesystem::kFsPageSize);
  if (cls == RaSeq::kRandom || RaAppendsPending(c->slot)) {
    RaRetireAllLocked(*c);
    RaReapLocked(*c);
    return nullptr;
  }
  if (cls == RaSeq::kSequential && c->seq.Active()) {
    RaEnsureAhead(*c, h,
                  RaWindowFor(c->seq.End(), fsize, g_ra_pages,
                              clio::cte::filesystem::kFsPageSize),
                  fsize, lk);
  }
  return c;
}

/**
 * Free a handle's read-ahead cache at release. A fetch still out is waited
 * for: its buffer and future must outlive it. One that does not complete
 * within kRaReleaseWaitSlices is abandoned WITH its buffer (leaked, logged)
 * rather than freed under a runtime that may still write it.
 * @param h the handle being released
 */
static void RaRelease(CfsHandle *h) {
  RaCache *c = h->ra.exchange(nullptr, std::memory_order_acq_rel);
  if (c == nullptr) return;
  std::unique_lock<std::mutex> lk(c->mu);
  RaRetireAllLocked(*c);
  for (;;) {
    RaReapLocked(*c);
    if (c->zombies.empty()) break;
    RaPage *pg = c->zombies.back().get();
    if (pg->state != RaPage::State::kInFlight || pg->pins != 0) {
      // In another thread's hands (not expected at release): it notifies.
      c->cv.wait_for(lk, std::chrono::milliseconds(100));
      continue;
    }
    pg->state = RaPage::State::kWaiting;
    lk.unlock();
    int rc = 0;
    const bool done = RaAwaitFetch(h, pg, kRaReleaseWaitSlices, &rc);
    lk.lock();
    if (done) {
      pg->state = RaPage::State::kReady;
      continue;
    }
    HLOG(kError, "clio_cte_fuse: read-ahead of tag {}.{} page {} never "
         "completed; abandoning its {} byte buffer", h->tag.major_,
         h->tag.minor_, pg->page, clio::cte::filesystem::kFsPageSize);
    for (auto &z : c->zombies) {
      if (z.get() == pg) {
        (void)z.release();
        z = std::move(c->zombies.back());
        c->zombies.pop_back();
        break;
      }
    }
  }
  lk.unlock();
  delete c;
}

/**
 * Read CLIO_FUSE_READAHEAD_PAGES at mount (after the file attribute TTL is
 * known): read-ahead needs that TTL, which bounds how stale a cached page
 * can be against writes on other nodes.
 */
static void RaInit() {
  g_ra_pages = RaParsePages(getenv("CLIO_FUSE_READAHEAD_PAGES"));
  if (g_ra_pages == 0) return;
  if (g_file_attr_ttl_s <= 0.0) {
    HLOG(kWarning, "clio_cte_fuse: CLIO_FUSE_READAHEAD_PAGES={} ignored: the "
         "file attribute TTL is 0 (CLIO_FUSE_ATTR_CACHE_S; the default on a "
         "multi-node mount), and a read-ahead page is only served while "
         "younger than that TTL", g_ra_pages);
    g_ra_pages = 0;
    return;
  }
  HLOG(kInfo, "clio_cte_fuse: sequential read-ahead on: {} page(s) of {} "
       "bytes ahead per open handle (at most {} buffered), each served for "
       "at most {} ms after its fetch", g_ra_pages,
       clio::cte::filesystem::kFsPageSize, g_ra_pages + 1,
       g_file_attr_ttl_s * 1000.0);
}

static std::mutex g_unmerged_mu;  ///< guards g_unmerged_appends
/** Files with deferred appends this node accepted that may be unmerged. */
static std::set<clio::run::u64> g_unmerged_appends;

/**
 * Wait until this handle's deferred appends are merged into the file, then
 * stamp the file's mtime at its inode home. No-op without pending appends.
 * @param handle open file
 * @param hp the handle's current path (unused: the merge is keyed by tag)
 * @return 0 or -errno
 */
static int FlushAppends(CfsHandle *handle, const std::string &hp) {
  (void)hp;
  if (!handle->appended.exchange(false)) return 0;
  const clio::run::u64 packed = PackTag(handle->tag);
  const clio::run::u64 ra_mark = RaAppendMark(handle->tag);
  {
    // The flush merges every append accepted before it, from any handle;
    // one accepted later registers again once its write returns.
    std::lock_guard<std::mutex> g(g_unmerged_mu);
    g_unmerged_appends.erase(packed);
  }
  auto f = StreamClient().AsyncFlush(handle->tag,
                                     clio::cte::filesystem::FsIdHome(packed));
  f.Wait();
  if (f->GetReturnCode() != 0) {
    std::lock_guard<std::mutex> g(g_unmerged_mu);
    g_unmerged_appends.insert(packed);
    return -EIO;
  }
  RaAppendsFlushed(handle->tag, ra_mark);  // the merge changed the file
  auto t = CLIO_CFS_CLIENT->AsyncAdvanceSize(packed, f->size_);
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

/**
 * Deferred O_APPEND write: stage the bytes on this node; the file's stream
 * home appends them at the end in its next merge.
 * @param handle open O_APPEND file with a known tag whose id has a home
 * @param buf bytes
 * @param size byte count
 * @return bytes written or -errno
 */
static int DeferredAppend(CfsHandle *handle, const char *buf, size_t size) {
  const clio::run::u64 packed = PackTag(handle->tag);
  const clio::run::u32 rc = StreamClient().Append(
      handle->tag, clio::cte::filesystem::FsIdHome(packed), buf, size);
  if (rc != 0) return -EIO;
  RaAppendAccepted(handle->tag);  // reads bypass read-ahead until flushed
  handle->appended.store(true);
  {
    std::lock_guard<std::mutex> g(g_unmerged_mu);
    g_unmerged_appends.insert(packed);
  }
  return static_cast<int>(size);
}

/**
 * Merge every deferred append this node accepted for a file before a
 * truncate: write(2) already returned for them, so they come before the
 * truncate (as on ext4) -- merged after it, they would reappear past the
 * cut. No-op, without a round trip, for files with no such appends.
 * @param tag the file's tag
 * @return 0 or -EIO (the appends could not be merged: do not truncate)
 */
static int MergeAppendsBeforeTruncate(const clio::cte::core::TagId &tag) {
  const clio::run::u64 packed = PackTag(tag);
  {
    std::lock_guard<std::mutex> g(g_unmerged_mu);
    if (g_unmerged_appends.erase(packed) == 0) return 0;
  }
  const clio::run::u64 ra_mark = RaAppendMark(tag);
  auto f = StreamClient().AsyncFlush(tag,
                                     clio::cte::filesystem::FsIdHome(packed));
  f.Wait();
  if (f->GetReturnCode() == 0) {
    RaAppendsFlushed(tag, ra_mark);  // the merge changed the file
    return 0;
  }
  HLOG(kError, "clio_cte_fuse: appends to {}.{} could not be merged before "
       "a truncate (rc {})", tag.major_, tag.minor_, f->GetReturnCode());
  std::lock_guard<std::mutex> g(g_unmerged_mu);
  g_unmerged_appends.insert(packed);
  return -EIO;
}

/**
 * Close-to-open publish of a file, synchronously: merge its deferred
 * appends, push the size its page puts reached to the chimod, and wait for
 * the replication barrier. The pages themselves already landed: page puts
 * are synchronous (in the shm cache, or acknowledged by their task).
 * @param handle the open file
 * @param hp its current path
 * @return 0, or a negative errno if the size (or the replication barrier)
 *         could not be published
 */
static int PublishOnClose(CfsHandle *handle, const std::string &hp) {
  auto *cfs = CLIO_CFS_CLIENT;
  const clio::run::u64 hiwater = HiwaterFor(hp);
  const int aerr = FlushAppends(handle, hp);
  if (aerr != 0) return aerr;
  if (hiwater == 0 || handle->tag.IsNull()) return 0;
  auto t = cfs->AsyncAdvanceSize(
      (static_cast<clio::run::u64>(handle->tag.major_) << 32) |
          static_cast<clio::run::u64>(handle->tag.minor_),
      hiwater, PushTimeFor(hp));
  t.Wait();
  if (t->GetReturnCode() != 0) {
    HLOG(kError, "clio_cte_fuse: publishing the size of {} failed (rc {}); "
         "reporting EIO", hp, t->GetReturnCode());
    return -EIO;
  }
  // fsync's durability contract covers only what THIS handle wrote: the
  // barrier waits for the async replica sweep to catch up to the primary.
  // A no-op (see NeedsReplicationFlushBarrier) when replication is absent
  // or already synchronous.
  const int berr = FlushReplicationBarrier(handle->tag);
  if (berr != 0) {
    HLOG(kError, "clio_cte_fuse: replication barrier for {} failed ({})",
         hp, berr);
    return berr;
  }
  // The chimod now holds this size: drop the overlay, as RunPublish and the
  // closer do. Kept, it outlived a truncate from ANOTHER node (whose clamp
  // only touches its own overlay) and release() re-pushed the stale size.
  HiwaterEraseIfCovered(hp, hiwater);
  return 0;
}

/**
 * Whether close(2) publishes asynchronously on a single-node mount
 * (CLIO_FUSE_ASYNC_FLUSH, default on; 0 restores the awaited publish).
 * @return true when flush may queue the publish instead of awaiting it
 */
static bool AsyncFlushEnabled() {
  static const bool v = [] {
    const char *e = getenv("CLIO_FUSE_ASYNC_FLUSH");
    return e == nullptr || *e != '0';
  }();
  return v;
}

/**
 * close(2) on a single-node mount: report failures already known for the
 * file, and queue -- not await -- the close-to-open publish of what THIS
 * handle wrote. Close-to-open still holds because every operation that could
 * observe the file other than through this daemon's own overlays (hiwater
 * size) waits for the path's queued entries first: link, rename, unlink,
 * truncate and fsync do. A handle
 * that wrote nothing (a reader, or a virus scanner re-opening a fresh file)
 * publishes nothing.
 * @param handle the closing handle
 * @param p its current path
 * @return 0 or a negative errno already latched against the file
 */
static int FlushAsync(CfsHandle *handle, const std::string &p) {
  const int perr = TakePublishError(p);
  if (perr != 0) return perr;
  if (handle->wrote.exchange(false)) {
    PendingClose pc;
    pc.kind = PendingClose::kPublish;
    pc.path = p;
    pc.tag = handle->tag;
    pc.hiwater = HiwaterFor(p);
    pc.wtime = PushTimeFor(p);
    EnqueueDrainOrdered(std::move(pc));
    // The kernel caches each NAME separately (see the awaited path below).
    for (const auto &sib : LinkGroupSiblings(p)) InvalidatePath(sib);
  }
  return 0;
}

/**
 * close(2): publish what this handle wrote (its size; page puts are
 * synchronous, so their errors were already returned by write(2)).
 * @param path libfuse's path
 * @param fi the open file
 * @return 0 or a negative errno
 */
int cte_fuse_flush(const char *path, struct fuse_file_info *fi) {
  auto *handle = GetHandle(fi);
  const std::string p =
      handle ? HandlePath(handle, path) : std::string(path ? path : "");
  if (p.empty()) return 0;
  // A file with hard links takes the awaited path: its size and pages are
  // overlaid under THIS name only, so a stat or read through a sibling
  // right after close(2) saw the old file until the queued publish landed.
  if (handle != nullptr && AsyncFlushEnabled() && !MultiNode() &&
      !handle->appended.load() && LinkGroupSiblings(p).empty()) {
    return FlushAsync(handle, p);
  }
  if (handle != nullptr &&
      (MultiNode() || HiwaterFor(p) != 0 || handle->appended.load())) {
    // CLOSE-TO-OPEN: once close(2) returns, an open on ANY node -- or any
    // later operation on THIS one -- must see the file's bytes and size.
    // release() is delivered after close(2) has already returned (and is
    // asynchronous here), so the size push happens in flush, which
    // close(2) awaits. Only files this handle wrote pay for it. Leaving it
    // to the lazy closer let git's close -> link(tmp, obj) -> unlink(tmp)
    // outrun the size push: `git add` of a source tree left loose objects
    // empty or missing (git fsck "missing blob"), even on one node.
    int rc = PublishOnClose(handle, p);
    if (rc != 0) return rc;
    // The kernel caches each NAME separately: data and size written through
    // this name stay invisible through its hard links until their cached
    // pages and attrs are dropped (write via b, read via a saw the old file).
    for (const auto &sib : LinkGroupSiblings(p)) InvalidatePath(sib);
  }
  return 0;
}

int cte_fuse_fsync(const char *path, int /*datasync*/,
                          struct fuse_file_info *fi) {
  auto *handle = GetHandle(fi);
  const std::string p =
      handle ? HandlePath(handle, path) : std::string(path ? path : "");
  if (p.empty()) return 0;
  // A publish close(2) queued for this file must have run (and its outcome
  // be reported) before fsync can vouch for the file.
  if (CloserPendingFor(p)) CloserBarrier();
  {
    const int perr = TakePublishError(p);
    if (perr != 0) return perr;
  }
  // Publish the logical size. fsync is the durability point: a size that
  // only traveled on close was lost with the process (a crash after fsync
  // left the file's bytes stored but its size -- and on another node, its
  // visibility -- behind the unpublished hiwater overlay).
  if (handle != nullptr) {
    int rc = PublishOnClose(handle, p);
    if (rc != 0) return rc;
  }
  // Everything above landed the bytes on the store; fsync's contract is
  // that they are ON A PERSISTENT DEVICE when it returns.
  return SyncDurable(handle != nullptr ? handle->tag
                                       : clio::cte::core::TagId::GetNull(),
                     clio::cte::filesystem::FsParentDir(p));
}

/**
 * fsyncdir(2): make the namespace (directory entries) durable.
 * @param path directory path (its tag holds its entries' blocks)
 * @param datasync unused (entries have no separate data)
 * @param fi unused
 * @return 0 or a negative errno
 */
int cte_fuse_fsyncdir(const char *path, int /*datasync*/,
                      struct fuse_file_info * /*fi*/) {
  // A directory's entries live in blocks in its own tag: syncing it makes
  // creates, renames and unlinks under the directory durable (the
  // write-tmp, fsync, rename, fsync-dir pattern).
  return SyncDurable(clio::cte::core::TagId::GetNull(),
                     path != nullptr ? std::string(path) : std::string("/"));
}

int cte_fuse_release(const char *path, struct fuse_file_info *fi) {
  auto *handle = GetHandle(fi);
  if (!handle) return 0;
  const std::string rp = HandlePath(handle, path);  // adopt a rename
  auto *cfs = CLIO_CFS_CLIENT;
  // Normally flush (close(2)) already merged this handle's deferred appends;
  // a handle released without a flush must not drop them.
  (void)FlushAppends(handle, rp);
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
  if (async_close) {
    // The close carrying a size push runs off the caller's path, on the
    // lazy closer; a close with nothing to publish goes detached.
    if (hiwater != 0) {
      PendingClose pc;
      pc.kind = PendingClose::kClose;
      pc.fh = handle->fh;
      pc.path = handle->path;
      pc.tag = handle->tag;
      pc.hiwater = hiwater;
      pc.wtime = PushTimeFor(handle->path);
      EnqueueDrainOrdered(std::move(pc));
    } else {
      cfs->AsyncCloseDetached(handle->fh);
    }
  } else {
    auto t = cfs->AsyncClose(handle->fh, hiwater);
    t.Wait();
    rc = (t->GetReturnCode() == 0) ? 0 : -EIO;
    if (hiwater != 0) {
      if (rc == 0) {
        HiwaterEraseIfCovered(handle->path, hiwater);
      } else {
        LatchPublishError(handle->path, -EIO);
      }
    }
  }
  UnregisterOpenFile(handle);
  RaRelease(handle);  // waits out read-ahead fetches still in flight
  delete handle;
  fi->fh = 0;
  return rc;
}

// ============================================================================
// Read / Write — delegated to the chimod's page-based I/O
// ============================================================================

/** How reads move atime (CLIO_FUSE_ATIME, or the noatime/strictatime/
 *  relatime mount option; see fuse_cte_main.cc). */
enum class AtimeMode { kOff, kRelatime, kStrict };

/** @return the mount's atime mode (default relatime, as on Linux). */
static AtimeMode AtimeModeOf() {
  static const AtimeMode mode = [] {
    const char *e = std::getenv("CLIO_FUSE_ATIME");
    if (e == nullptr) return AtimeMode::kRelatime;
    if (std::strcmp(e, "0") == 0 || std::strcmp(e, "noatime") == 0) {
      return AtimeMode::kOff;
    }
    if (std::strcmp(e, "strict") == 0 || std::strcmp(e, "strictatime") == 0) {
      return AtimeMode::kStrict;
    }
    return AtimeMode::kRelatime;
  }();
  return mode;
}

/**
 * First read through an open file: let the file's owner advance atime by
 * the relatime rule (FUSE leaves atime to the filesystem; nothing moved it
 * before, generic/192). Once per open, so a streaming read pays one round
 * trip, and not at all for O_NOATIME opens.
 * @param handle the open file
 * @param hp its current path
 */
static void TouchAtimeOnRead(CfsHandle *handle, const std::string &hp) {
  const AtimeMode mode = AtimeModeOf();
  if (mode == AtimeMode::kOff || handle->atime_done.exchange(true)) return;
  clio::run::u32 flags = clio::cte::filesystem::kUtimensAccess;
  if (mode == AtimeMode::kStrict) {
    flags |= clio::cte::filesystem::kUtimensAccessStrict;
  }
  auto t = CLIO_CFS_CLIENT->AsyncUtimens(hp, 0, 0, flags);
  t.Wait();
}

/**
 * Read one range of one page blob into `dst`: from the shm cache when the
 * CTE client is connected and holds it, else a GetBlob task. The task is
 * waited in slices so a read whose completion never arrives is named in the
 * log; it still waits for the completion, since the runtime may yet write
 * into dst.
 * @param cte the CTE client of the file's data chain
 * @param tag the file's tag
 * @param page the page blob name
 * @param page_off offset inside the page
 * @param n bytes to read
 * @param dst destination
 * @return the GetBlob return code (0 read, 1 the page does not exist),
 *         ENOMEM when the task could not stage the read
 */
static int ReadPageOnce(clio::cte::core::Client *cte,
                        const clio::cte::core::TagId &tag,
                        const std::string &page, clio::run::u64 page_off,
                        clio::run::u64 n, char *dst) {
  if (cte->TryGetCached(tag, page, page_off, n, dst)) return 0;
  auto g = cte->AsyncGetBlob(tag, page, page_off, n, 0, dst);
  if (g.get() == nullptr) return ENOMEM;
  for (int slice = 1; !g.Wait(30.0f); ++slice) {
    HLOG(kWarning, "clio_cte_fuse: read of tag {}.{} page {} (off {} len {}) "
         "still outstanding after {} s", tag.major_, tag.minor_, page,
         page_off, n, slice * 30);
  }
  return static_cast<int>(g->GetReturnCode());
}

/**
 * Read one page range, retrying the transient answers of a page whose owner
 * was just declared dead: sent again it routes to the stand-in, which
 * serves it. Only after the retries is it an error.
 * @param cte the CTE client of the file's data chain
 * @param tag the file's tag
 * @param cur file offset of the range
 * @param page_off offset inside the page
 * @param n bytes to read
 * @param dst destination
 * @return the final GetBlob return code (0 read, 1 a hole, else an error)
 */
static int ReadPageRetrying(clio::cte::core::Client *cte,
                            const clio::cte::core::TagId &tag,
                            clio::run::u64 cur, clio::run::u64 page_off,
                            clio::run::u64 n, char *dst) {
  const std::string page = clio::cte::filesystem::PageName(cur);
  const auto read_t0 = std::chrono::steady_clock::now();
  int grc = 0;
  int attempt = 0;
  for (;; ++attempt) {
    grc = ReadPageOnce(cte, tag, page, page_off, n, dst);
    // Owner-down from a stand-in (#1280) can be as transient as node-lost:
    // the stand-in's liveness view lags the router's until it too declares
    // the owner dead and serves its copies.
    const bool transient =
        clio::cte::core::IsNodeLostRc(static_cast<clio::run::u32>(grc)) ||
        static_cast<clio::run::u32>(grc) == clio::cte::core::kBlobOwnerDownRc;
    if (!transient || attempt >= kReadNodeLostRetries) break;
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kReadNodeLostRetryMs));
  }
  if (attempt > 0) {
    // Where a slow failing read spent its time (#1169). Rate-limited: with
    // a node down every page read of that node lands here.
    static std::atomic<clio::run::u64> logged{0};
    const clio::run::u64 k = logged.fetch_add(1, std::memory_order_relaxed);
    if (k < 8 || k % 256 == 0) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - read_t0)
                            .count();
      HLOG(kWarning, "clio_cte_fuse: read of tag {}.{} page {} got the "
           "node-lost code {} time(s) over {} ms; final rc {} ({} such "
           "reads so far)", tag.major_, tag.minor_, page, attempt, ms, grc,
           k + 1);
    }
  }
  return grc;
}

/**
 * Log a read that took more than 2 s, with where it spent its time (#1169).
 * Rate-limited.
 * @param tag the file's tag
 * @param offset read offset
 * @param size read size
 * @param total_ms the read's time (ms)
 * @param size_ms time of its size lookup (ms)
 * @param pages pages read
 * @param rc the read's result
 */
static void LogSlowRead(const clio::cte::core::TagId &tag, cte_off_t offset,
                        size_t size, double total_ms, double size_ms,
                        clio::run::u64 pages, int rc) {
  if (total_ms < 2000.0) return;
  static std::atomic<clio::run::u64> logged{0};
  const clio::run::u64 k = logged.fetch_add(1, std::memory_order_relaxed);
  if (k < 8 || k % 256 == 0) {
    HLOG(kWarning, "clio_cte_fuse: slow read of tag {}.{} [{}, +{}): {} ms "
         "(size lookup {} ms, {} page read(s)); rc {} ({} such reads so far)",
         tag.major_, tag.minor_, offset, size, total_ms, size_ms, pages, rc,
         k + 1);
  }
}

/**
 * Read pages of a file through the CTE client (cache first, then tasks).
 * EOF clamps against the chimod size raised by the local hiwater (bytes
 * acknowledged to write(2) but not yet carried to the chimod by close).
 * @param handle the open file (its tag set)
 * @param hp its current path
 * @param cte the CTE client of the file's data chain
 * @param buf destination
 * @param size bytes requested
 * @param offset file offset
 * @return bytes read or a negative errno
 */
static int ReadDirect(CfsHandle *handle, const std::string &hp,
                      clio::cte::core::Client *cte, char *buf, size_t size,
                      cte_off_t offset) {
  namespace ev = clio::run::evlat;
  using clio::cte::filesystem::kFsPageSize;
  const auto read_t0 = std::chrono::steady_clock::now();
  auto ms_since_start = [&read_t0] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - read_t0)
        .count();
  };
  const unsigned long long ev_sz0 = ev::Now();
  // The descriptor's own file: after a rename over its name the path names
  // another file (or, mid-rename, none) and read 0 bytes.
  // The local size overlay is read BEFORE the size lookup: the closer drops
  // it as soon as the chimod acknowledges the size push, so read after a
  // lookup answered just before the push landed, neither would cover the
  // file's bytes and the read returned EOF (see FillStatFromAttrs).
  const clio::run::u64 hw_floor = HiwaterFor(hp);
  clio::run::u64 fsize = 0;
  const int src = ReadClampSize(
      handle, hp, static_cast<clio::run::u64>(offset), size, &fsize);
  const double size_ms = ms_since_start();
  if (src == -EIO) {
    LogSlowRead(handle->tag, offset, size, ms_since_start(), size_ms, 0, -EIO);
    return -EIO;
  }
  // The descriptor's own file is gone on the server: a silent EOF would
  // read as a truncated file. ESTALE, as NFS reports a vanished file.
  if (src == -ENOENT) return -kOpenVanishedErrno;
  fsize = std::max({fsize, hw_floor, HiwaterFor(hp)});
  ev::Span(ev::kRdSize, ev_sz0, ev::Now());
  const clio::run::u64 off = static_cast<clio::run::u64>(offset);
  if (off >= fsize) return 0;
  const clio::run::u64 want = std::min<clio::run::u64>(size, fsize - off);
  std::memset(buf, 0, want);  // holes / short pages read as zeros
  // Sequential read-ahead (CLIO_FUSE_READAHEAD_PAGES; off: one check).
  RaCache *ra = g_ra_pages != 0 ? RaPrepare(handle, off, want, fsize)
                                : nullptr;
  clio::run::u64 done = 0;
  clio::run::u64 ev_pages = 0;
  while (done < want) {
    const clio::run::u64 cur = off + done;
    const clio::run::u64 page_off = cur % kFsPageSize;
    const clio::run::u64 n =
        std::min<clio::run::u64>(kFsPageSize - page_off, want - done);
    if (ra != nullptr && RaServe(ra, handle, cur, page_off, n, buf + done)) {
      done += n;  // answered from the handle's read-ahead cache
      continue;
    }
    const unsigned long long ev_pg0 = ev::Now();
    ++ev_pages;
    const int grc =
        ReadPageRetrying(cte, handle->tag, cur, page_off, n, buf + done);
    ev::Span(ev::kRdPage, ev_pg0, ev::Now());
    // 0 = read, 1 = the page does not exist (a hole: the pre-zeroed buffer
    // is the right answer). Anything else -- above all the network-timeout
    // code a page on a DEAD node completes with -- is an I/O error, never
    // zeros with success.
    if (grc != 0 && grc != 1) {
      HLOG(kError, "clio_cte_fuse: read of tag {}.{} page {} failed rc {}; "
           "reporting EIO", handle->tag.major_, handle->tag.minor_,
           clio::cte::filesystem::PageName(cur), grc);
      LogSlowRead(handle->tag, offset, size, ms_since_start(), size_ms,
                  done / kFsPageSize + 1, -EIO);
      return -EIO;
    }
    done += n;
  }
  LogSlowRead(handle->tag, offset, size, ms_since_start(), size_ms,
              (want + kFsPageSize - 1) / kFsPageSize, 0);
  ev::Count(ev::kRdPages, ev_pages);
  ev::Count(ev::kRdBytes, (want + 512) / 1024);
  return static_cast<int>(want);
}

/**
 * read(2): page blobs through the CTE client when the file's tag is known,
 * else the filesystem client's read.
 * @param path libfuse's path
 * @param buf destination
 * @param size bytes requested
 * @param offset file offset
 * @param fi the open file
 * @return bytes read or a negative errno
 */
int cte_fuse_read(const char *path, char *buf, size_t size,
                         cte_off_t offset, struct fuse_file_info *fi) {
  namespace ev = clio::run::evlat;
  ev::ScopedSpan ev_total(ev::kRdTotal);
  auto *handle = GetHandle(fi);
  if (!handle) return -EBADF;
  const std::string hp = HandlePath(handle, path);
  TouchAtimeOnRead(handle, hp);

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);
  if (size == 0) return 0;
  // Read-your-appends: this handle's deferred appends land first.
  const int aerr = FlushAppends(handle, hp);
  if (aerr != 0) return aerr;

  auto *cte = CLIO_CFS_CLIENT->Cte();
  if (cte != nullptr && !handle->tag.IsNull()) {
    return ReadDirect(handle, hp, cte, buf, size, offset);
  }
  ssize_t got = CfsReadCompat(CLIO_CFS_CLIENT, handle->fh, hp,
                              static_cast<clio::run::u64>(offset), buf, size);
  if (got < 0) return -EIO;
  return static_cast<int>(got);
}

/**
 * write(2) through page blobs: each page range is a synchronous
 * Client::PutBlob -- a copy into the shm cache when the CTE client is
 * connected (no task, no wait), else a PutBlob task. Errors return here;
 * nothing is left to latch. The logical size rides the hiwater map until
 * close carries it to the chimod.
 * @param handle the open file (its tag set)
 * @param hp its current path
 * @param cte the CTE client of the file's data chain
 * @param buf source
 * @param size bytes
 * @param offset file offset
 * @return bytes written or a negative errno
 */
static int WriteDirect(CfsHandle *handle, const std::string &hp,
                       clio::cte::core::Client *cte, const char *buf,
                       size_t size, cte_off_t offset) {
  namespace ev = clio::run::evlat;
  using clio::cte::filesystem::kFsPageSize;
  size_t done = 0;
  while (done < size) {
    const clio::run::u64 cur = static_cast<clio::run::u64>(offset) + done;
    const clio::run::u64 page_off = cur % kFsPageSize;
    const clio::run::u64 n =
        std::min<clio::run::u64>(kFsPageSize - page_off, size - done);
    const unsigned long long ev_pg0 = ev::Now();
    const clio::run::u32 rc = cte->PutBlob(
        handle->tag, clio::cte::filesystem::PageName(cur), page_off, n,
        buf + done, g_page_score.ScoreOf(cur / kFsPageSize));
    ev::Span(ev::kWrPage, ev_pg0, ev::Now());
    if (rc != 0) {
      // Bytes already stored by earlier pages of this write are kept: the
      // file grows to them, and write(2) reports what was stored.
      if (done > 0) break;
      // A full store is ENOSPC (as on ext4); anything else is EIO.
      if (clio::cte::core::PutRcIsNoSpace(rc)) return -ENOSPC;
      return rc == ENOMEM ? -ENOMEM : -EIO;
    }
    done += n;
  }
  UnsyncedNote(handle->tag);
  HiwaterRaise(hp, static_cast<clio::run::u64>(offset) + done);
  return static_cast<int>(done);
}

/**
 * write(2): O_APPEND placement, then page blobs through the CTE client
 * (WriteDirect) when the file's tag is known, else the filesystem client.
 * @param path libfuse's path
 * @param buf source
 * @param size bytes
 * @param offset file offset (O_APPEND may move it)
 * @param fi the open file
 * @return bytes written or a negative errno
 */
int cte_fuse_write(const char *path, const char *buf, size_t size,
                          cte_off_t offset, struct fuse_file_info *fi) {
  namespace ev = clio::run::evlat;
  ev::ScopedSpan ev_total(ev::kWrTotal);
  auto *handle = GetHandle(fi);
  if (!handle) return -EBADF;
  const std::string hp = HandlePath(handle, path);

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);
  if (size == 0) return 0;
  // Read-ahead pages of this file (any handle) are dropped when this write
  // is done, on every return path below, before it is acknowledged.
  RaTagGuard ra_guard(handle->tag);
  handle->wrote.store(true, std::memory_order_relaxed);
  handle->open_attr_ok.store(false, std::memory_order_relaxed);

  // Page writes go straight to the file's page blobs (WriteDirect); the
  // filesystem client's write is the fallback when the tag is unknown.
  // O_APPEND across nodes: the kernel derived `offset` from ITS cached size,
  // which another node's appends have already moved past -- two nodes each
  // appending 200 records lost half of them. Either stage the bytes for the
  // file's stream home to append at the real end (deferred, no round trip
  // to the home per write), or reserve the end there first and write at it.
  if (handle->append && !handle->tag.IsNull() &&
      AppendModeOf() == AppendMode::kDeferred &&
      clio::cte::filesystem::FsIdHasHome(PackTag(handle->tag))) {
    UnsyncedNote(handle->tag);
    return DeferredAppend(handle, buf, size);
  }
  if (handle->append && !handle->tag.IsNull() &&
      AppendModeOf() != AppendMode::kKernel) {
    auto r = CLIO_CFS_CLIENT->AsyncReserveAppend(
        (static_cast<clio::run::u64>(handle->tag.major_) << 32) |
            static_cast<clio::run::u64>(handle->tag.minor_),
        static_cast<clio::run::u64>(size));
    r.Wait();
    if (r->GetReturnCode() != 0) return -EIO;
    offset = static_cast<cte_off_t>(r->old_size_);
  }
  auto *cte = CLIO_CFS_CLIENT->Cte();
  if (cte != nullptr && !handle->tag.IsNull()) {
    return WriteDirect(handle, hp, cte, buf, size, offset);
  }
  auto *cfs = CLIO_CFS_CLIENT;
  UnsyncedNote(handle->tag);
  ssize_t wrote = CfsWriteCompat(cfs, handle->fh, hp,
                                 static_cast<clio::run::u64>(offset), buf,
                                 size);
  if (wrote < 0) return -errno;
  return static_cast<int>(wrote);
}

// ============================================================================
// Unlink / Truncate
// ============================================================================

int cte_fuse_unlink(const char *path) {
  DirAttrInvalidate(path);
  // libfuse removes a hidden name when the file's last descriptor closes.
  // Unless the hide fell back to a server rename, the name never existed on
  // the server (HideOpenFile), which purges the file at its last close.
  if (IsHiddenName(path)) {
    std::lock_guard<std::mutex> g(g_open_files_mu);
    if (g_server_hidden.erase(path) == 0) return 0;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  // A queued close-to-open publish racing the unlink would land on a deleted
  // file; run the closer's entries first.
  CloserBarrier();
  HiwaterErase(std::string(path));
  auto t = cfs->AsyncUnlink(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/EISDIR/EIO
  if (rc == 0) {
    for (const auto &sib : LinkGroupDrop(std::string(path))) {
      InvalidatePath(sib);  // their nlink/ctime changed under the TTL
    }
  }
  return FsErrno(rc);
}

int cte_fuse_truncate(const char *path_in, cte_off_t size,
                             struct fuse_file_info *fi) {
  g_setattr_epoch.fetch_add(1);  // attributes kept from a create are stale
  RaAllGuard ra_guard;  // read-ahead pages are dropped again once it is done
  // ftruncate of an open file with no name left (hard_remove) truncates it
  // by its inode key -- common for unlinked temp files (databases).
  const std::string path_s = OpPath(path_in, fi);
  if (path_s.empty()) return -EBADF;
  const char *path = path_s.c_str();
  auto *cfs = CLIO_CFS_CLIENT;
  // Page writes are synchronous, so nothing of this file is in flight; the
  // tag is resolved for the appends merge and the truncate itself, and the
  // hiwater overlay is clamped so stat reflects the truncation immediately.
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
    // Resolve the tag from every identity source: an ftruncate arrives with
    // the open handle (fi); otherwise the mirror, then the name's owner.
    if (fi != nullptr) {
      auto *h = GetHandle(fi);
      if (h != nullptr) tr_tag = h->tag;
    }
    if (tr_tag.IsNull()) {
      clio::cte::filesystem::ShmFileRecord rec;
      if (cfs->TryGetFileRecordShm(p, &rec) && !rec.tag_id_.IsNull()) {
        tr_tag = clio::cte::core::TagId(rec.tag_id_.major_, rec.tag_id_.minor_);
      }
    }
    if (tr_tag.IsNull()) {
      // Authoritative fallback: a refused/absent mirror record leaves the
      // tag unresolved, and the appends merge below needs it. Ask the owner
      // of the name (its inode number IS the packed id).
      auto g = cfs->AsyncGetattr(p);
      g.Wait();
      if (g->GetReturnCode() == 0 && g->exists_ && !g->is_dir_ &&
          g->ino_ > 1) {
        tr_tag = clio::cte::filesystem::FsUnpack(g->ino_);
      }
    }
    if (!tr_tag.IsNull()) {
      const int aerr = MergeAppendsBeforeTruncate(tr_tag);
      if (aerr != 0) return aerr;
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
  const clio::run::u32 trc = t->GetReturnCode();
  if (trc == 0) return 0;
  return trc < 4096 ? -static_cast<int>(trc) : -EIO;
}

#ifdef __linux__
// Write `len` zero bytes at `off` through an open handle, chunked so a large
// range doesn't need one giant SHM buffer. Used by ZERO_RANGE. `path` is the
// op's path (null only when the file has no name). Returns 0 or a
// negative errno.
static int cte_fuse_write_zeros(const char *path, struct fuse_file_info *fi,
                                cte_off_t off, cte_off_t len) {
  // Zeros MUST take the exact same path as write(2) -- page puts on the
  // file's TAG, not the cfs handle pipeline: a second data path for the same
  // bytes let one overwrite the other (generic/008's zeroed first byte read
  // back as its old value).
  constexpr size_t kChunk = 64 * 1024;
  std::unique_ptr<char[]> zbuf(new char[kChunk]());
  for (cte_off_t done = 0; done < len;) {
    const size_t n = static_cast<size_t>(
        ((len - done) < static_cast<cte_off_t>(kChunk)) ? (len - done)
                                                        : kChunk);
    // Pass the op's path, not null: a null path tells cte_fuse_write the
    // file lost its last name, re-keying the handle (and the hiwater this
    // write raises) by inode, so a stat by path missed the growth
    // (generic/075: "Size error: expected 0x146ee stat 0x13226").
    int wrote = cte_fuse_write(path, zbuf.get(), n, off + done, fi);
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
/**
 * Bytes the store can still take, cluster-wide (the statfs figure).
 * @return remaining capacity, or ~0 when it cannot be determined
 */
static clio::run::u64 StoreRemainingBytes() {
  auto *cte = CLIO_CFS_CLIENT->Cte();
  if (cte == nullptr) return ~0ULL;
  auto t = cte->AsyncGetCapacity(clio::run::PoolQuery::Broadcast());
  t.Wait();
  if (t->return_code_ != 0) return ~0ULL;
  return t->remaining_capacity_;
}

static int cte_fuse_fallocate(const char *path, int mode, cte_off_t offset,
                              cte_off_t length, struct fuse_file_info *fi) {
  g_setattr_epoch.fetch_add(1);  // attributes kept from a create are stale
  RaAllGuard ra_guard;  // read-ahead pages are dropped again once it is done
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
    rc = cte_fuse_write_zeros(path, fi, offset, length);
    if (rc != 0) return rc;
    if ((mode & FALLOC_FL_KEEP_SIZE) && (offset + length) > st.st_size) {
      // Restore EOF through the REAL truncate hook: it runs the closer's
      // queued size pushes and clamps the hiwater overlay the zero writes
      // raised -- the raw AsyncTruncate left that overlay in place, so fstat
      // right after a ZERO_RANGE|KEEP_SIZE reported the zero op's end as the
      // size (fsx).
      int trc = cte_fuse_truncate(path, st.st_size, fi);
      if (trc != 0) return trc;
    }
    // The zeros were written DAEMON-side: the kernel only drops its own
    // cached pages for PUNCH_HOLE, so a later mmap read of this range
    // served the stale pre-zero bytes (generic/075 fsx: every mismatching
    // byte was a should-be-zero). Invalidate the file's page cache.
    if (path != nullptr) InvalidatePath(std::string(path));
    return 0;
  }

  // fallocate promises the space: a request the store cannot hold fails
  // now with ENOSPC (as on ext4), not at some later write. Pages are not
  // reserved, so concurrent writers can still use the space first.
  if (static_cast<clio::run::u64>(length) > StoreRemainingBytes()) {
    return -ENOSPC;
  }
  if (mode & FALLOC_FL_KEEP_SIZE) {
    return 0;  // no size change requested
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
  DirAttrInvalidate(to);
  if (NameTooLong(to)) return -ENAMETOOLONG;
  if (IsHiddenName(from) || IsHiddenName(to)) return -ENOENT;
  // The new name resolves through the store, not this daemon's per-path
  // overlays, so the source's queued close-to-open publish must land first
  // (git: close(tmp) -> link(tmp, obj) -> unlink(tmp) -> read obj).
  if (CloserPendingFor(std::string(from)) || CloserPendingFor(std::string(to))) {
    CloserBarrier();
  }
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
  return FsErrno(rc);  // chimod returns errno-style codes
}

int cte_fuse_symlink(const char *target, const char *path) {
  DirAttrInvalidate(path);
  if (NameTooLong(path)) return -ENAMETOOLONG;
  // Create a symlink at `path` pointing at `target`. The chimod stores the
  // target string in a reserved marker blob under `path`'s tag.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncSymlink(std::string(target), std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/EEXIST/EIO
  return FsErrno(rc);  // chimod returns errno-style codes
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
    return FsErrno(rc);
  }
  std::string target = t->target_.str();
  size_t n = std::min(target.size(), size - 1);
  std::memcpy(buf, target.data(), n);
  buf[n] = '\0';
  return 0;
}

static int cte_fuse_setxattr(const char *path, const char *name,
                             const char *value, size_t size, int flags) {
  std::string hidden;
  if (const int hrc = MapHiddenPath(&path, &hidden)) return hrc;
  // Set xattr `name` on `path`. `value` is raw bytes (may contain NULs), so
  // preserve its length rather than treating it as a C string. `flags` carries
  // XATTR_CREATE(1) / XATTR_REPLACE(2), matching the runtime's bit checks.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncSetxattr(std::string(path), std::string(name),
                              std::string(value, size),
                              static_cast<unsigned int>(flags));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return FsErrno(rc);  // EEXIST/ENODATA/ENOENT/EIO -> negative errno
}

static int cte_fuse_getxattr(const char *path, const char *name, char *value,
                             size_t size) {
  std::string hidden;
  if (const int hrc = MapHiddenPath(&path, &hidden)) return hrc;
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
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncGetxattr(std::string(path), std::string(name));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc != 0) {
    return FsErrno(rc);  // ENOENT (file absent)
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
  std::string hidden;
  if (const int hrc = MapHiddenPath(&path, &hidden)) return hrc;
  // Return the NUL-separated, NUL-terminated list of xattr names. size==0 is a
  // length query.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncListxattr(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc != 0) {
    return FsErrno(rc);  // ENOENT
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
  std::string hidden;
  if (const int hrc = MapHiddenPath(&path, &hidden)) return hrc;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRemovexattr(std::string(path), std::string(name));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return FsErrno(rc);  // ENODATA/ENOENT/EIO -> negative errno
}

#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1 << 0)  // from <linux/fs.h>; guarded to avoid header clash
#endif
#ifndef RENAME_EXCHANGE
#define RENAME_EXCHANGE (1 << 1)  // ditto; referenced by the unsupported-flag test
#endif

/**
 * libfuse's "hide": `from` is open on this mount and is being unlinked or
 * renamed over, so libfuse keeps the open file reachable as `to`
 * (.fuse_hiddenXXXX). Done as a server rename, that name was shared by
 * every node (each picks .fuse_hidden<node id><counter> from the same
 * counters), and the rename moved whatever file `from` named on the server
 * by then -- after another node's rename over `from`, a different file than
 * the one open here, which this node's kernel inode then read pages of.
 * The server already keeps an unlinked file alive while any node has it
 * open, so the hidden name needs no server state: `from` is unlinked there,
 * and `to` stays local, mapped to this node's open files (HiddenIdPath).
 * @param from the name being hidden
 * @param to libfuse's hidden name
 * @param rc out: 0, or a negative errno from the unlink
 * @return false when the hide must be a server rename instead
 */
static bool HideOpenFile(const char *from, const char *to, int *rc) {
  {
    std::lock_guard<std::mutex> g(g_open_files_mu);
    auto it = g_open_files.find(from);
    if (it == g_open_files.end()) return false;
  }
  *rc = cte_fuse_unlink(from);
  // Another node removed or replaced the name first: it is gone either way.
  if (*rc == -ENOENT) *rc = 0;
  if (*rc != 0) return true;
  OpenFilesRenamed(std::string(from), std::string(to));
  // Its handles now key their path-keyed state (queued sizes, write times)
  // by the inode (HandlePath): move what the rename handler put under `to`.
  std::string id_path;
  if (HiddenIdPath(to, &id_path) == 0) HiwaterRename(std::string(to), id_path);
  return true;
}

int cte_fuse_rename(const char *from, const char *to,
                           unsigned int flags) {
  DirAttrClear();  // a moved directory takes its subtree's paths with it
  if (NameTooLong(to)) return -ENAMETOOLONG;
  if (IsHiddenName(from)) return -ENOENT;  // only libfuse knows these names
  auto *cfs = CLIO_CFS_CLIENT;
  // Queued closer entries are keyed by PATH; one running after the rename
  // would publish under the old name. Run them first (git's
  // write-tmp-then-rename pattern hits this on every object file). Page
  // writes are keyed by (tag, page) and are synchronous, so they need
  // nothing -- but the hiwater overlay is path-keyed and moves.
  CloserBarrier();
  HiwaterRename(std::string(from), std::string(to));
  bool server_hide = false;
  if (IsHiddenName(to)) {
    int hrc = 0;
    if (HideOpenFile(from, to, &hrc)) return hrc;
    server_hide = true;  // falls through to the server rename below
  }
  // RENAME_NOREPLACE: the rename must fail with EEXIST if `to` already exists.
  // Probe for the destination then fall through to a plain rename. This is the
  // standard high-level-FUSE approach (a tiny TOCTOU window vs a truly atomic
  // check, acceptable for a single-namespace rename). RENAME_EXCHANGE and
  // RENAME_WHITEOUT need chimod-level atomic swap / whiteout support and stay
  // EINVAL so callers fall back cleanly.
  if (flags & RENAME_NOREPLACE) {
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
  if (t->GetReturnCode() == 0) {
    LinkGroupRename(std::string(from), std::string(to));
    OpenFilesRenamed(std::string(from), std::string(to));
    if (server_hide) {
      std::lock_guard<std::mutex> g(g_open_files_mu);
      g_server_hidden.insert(to);
    }
  }
  int rc = static_cast<int>(t->GetReturnCode());
  return FsErrno(rc);  // chimod returns errno-style codes (ENOENT/EIO)
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
namespace {
/** The store's capacity as statfs last reported it (see StatfsCapacity). */
struct StatfsCache {
  std::mutex mu;
  bool valid = false;        ///< an answer has been fetched
  bool refreshing = false;   ///< a caller is fetching a new one
  std::chrono::steady_clock::time_point at;  ///< when the answer was fetched
  clio::run::u64 total = 0;
  clio::run::u64 remaining = 0;
};
StatfsCache g_statfs_cache;

/**
 * How long a capacity answer serves statfs (ms). CLIO_FUSE_STATFS_TTL_MS
 * overrides the 1000 ms default; 0 asks the store on every call.
 * @return the time to live in milliseconds
 */
clio::run::u64 StatfsTtlMs() {
  static const clio::run::u64 v = [] {
    const char *e = getenv("CLIO_FUSE_STATFS_TTL_MS");
    return e != nullptr && e[0] != 0 ? std::strtoull(e, nullptr, 10) : 1000;
  }();
  return v;
}

/**
 * The store's total and remaining capacity for statfs, at most
 * StatfsTtlMs() old.
 *
 * Windows queries volume information about once per file written (a
 * Steam-style install of 15k files sent 9,100 queries), and each answer was
 * a capacity broadcast to every node (~120 us here). WinFsp's own volume
 * cache (VolumeInfoTimeout) does not absorb them -- writes invalidate it.
 * One caller refreshes an expired answer; the others keep serving the old
 * one meanwhile. fallocate's space check asks the store directly
 * (StoreRemainingBytes), where a stale "enough" would matter.
 * @param total receives the total capacity (bytes)
 * @param remaining receives the remaining capacity (bytes)
 */
void StatfsCapacity(clio::run::u64 *total, clio::run::u64 *remaining) {
  const auto now = std::chrono::steady_clock::now();
  const auto ttl = std::chrono::milliseconds(StatfsTtlMs());
  {
    std::lock_guard<std::mutex> g(g_statfs_cache.mu);
    const bool fresh = g_statfs_cache.valid && now - g_statfs_cache.at < ttl;
    if (fresh || (g_statfs_cache.valid && g_statfs_cache.refreshing)) {
      *total = g_statfs_cache.total;
      *remaining = g_statfs_cache.remaining;
      return;
    }
    g_statfs_cache.refreshing = true;
  }
  clio::run::u64 t_bytes = 0, r_bytes = 0;
  bool ok = false;
  auto *cte = CLIO_CFS_CLIENT->Cte();
  if (cte != nullptr) {
    // Broadcast: total + remaining capacity across the whole cluster.
    auto t = cte->AsyncGetCapacity(clio::run::PoolQuery::Broadcast());
    t.Wait();
    if (t->return_code_ == 0) {
      t_bytes = t->total_capacity_;
      r_bytes = t->remaining_capacity_;
      ok = true;
    }
  }
  std::lock_guard<std::mutex> g(g_statfs_cache.mu);
  g_statfs_cache.refreshing = false;
  if (ok) {
    g_statfs_cache.valid = StatfsTtlMs() != 0;
    g_statfs_cache.at = std::chrono::steady_clock::now();
    g_statfs_cache.total = t_bytes;
    g_statfs_cache.remaining = r_bytes;
  }
  *total = ok ? t_bytes : g_statfs_cache.total;
  *remaining = ok ? r_bytes : g_statfs_cache.remaining;
}
}  // namespace

int cte_fuse_statfs(const char *path, cte_statvfs_t *stbuf) {
  (void)path;
  std::memset(stbuf, 0, sizeof(*stbuf));
  constexpr fsblkcnt_t kBlockSize = 4096;

  clio::run::u64 total_bytes = 0;
  clio::run::u64 remaining_bytes = 0;
  StatfsCapacity(&total_bytes, &remaining_bytes);
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
#ifdef __APPLE__
// macFUSE: cte_fuse_readdir has two overloads here (fuse_cte.h declares the
// fuse_fill_dir_t one, this file defines the fuse_darwin_fill_dir_t one), so
// CLIO_FUSE_TIMED's &fn cannot pick one. A single, non-overloaded name can.
static int cte_fuse_readdir_darwin(const char *path, void *buf,
                                   ClioFuseFillDirT filler, cte_off_t offset,
                                   struct fuse_file_info *fi,
                                   enum fuse_readdir_flags flags) {
  return cte_fuse_readdir(path, buf, filler, offset, fi, flags);
}
#endif

const struct fuse_operations cte_fuse_ops = {
    .getattr = CLIO_FUSE_TIMED(getattr, cte_fuse_getattr),
    .readlink = CLIO_FUSE_TIMED(readlink, cte_fuse_readlink),
    .mkdir = CLIO_FUSE_TIMED(mkdir, cte_fuse_mkdir),
    .unlink = CLIO_FUSE_TIMED(unlink, cte_fuse_unlink),
    .rmdir = CLIO_FUSE_TIMED(rmdir, cte_fuse_rmdir),
    .symlink = CLIO_FUSE_TIMED(symlink, cte_fuse_symlink),
    .rename = CLIO_FUSE_TIMED(rename, cte_fuse_rename),
    .link = CLIO_FUSE_TIMED(link, cte_fuse_link),
    .chmod = CLIO_FUSE_TIMED(chmod, cte_fuse_chmod),
    .chown = CLIO_FUSE_TIMED(chown, cte_fuse_chown),
    .truncate = CLIO_FUSE_TIMED(truncate, cte_fuse_truncate),
    .open = CLIO_FUSE_TIMED(open, cte_fuse_open),
    .read = CLIO_FUSE_TIMED(read, cte_fuse_read),
    .write = CLIO_FUSE_TIMED(write, cte_fuse_write),
    .statfs = CLIO_FUSE_TIMED(statfs, cte_fuse_statfs),
    .flush = CLIO_FUSE_TIMED(flush, cte_fuse_flush),
    .release = CLIO_FUSE_TIMED(release, cte_fuse_release),
    .fsync = CLIO_FUSE_TIMED(fsync, cte_fuse_fsync),
#ifdef __APPLE__
    .setxattr = CLIO_FUSE_TIMED(setxattr, cte_fuse_setxattr_darwin),
    .getxattr = CLIO_FUSE_TIMED(getxattr, cte_fuse_getxattr_darwin),
#else
    .setxattr = CLIO_FUSE_TIMED(setxattr, cte_fuse_setxattr),
    .getxattr = CLIO_FUSE_TIMED(getxattr, cte_fuse_getxattr),
#endif
    .listxattr = CLIO_FUSE_TIMED(listxattr, cte_fuse_listxattr),
    .removexattr = CLIO_FUSE_TIMED(removexattr, cte_fuse_removexattr),
    .opendir = CLIO_FUSE_TIMED(opendir, cte_fuse_opendir),
#ifdef __APPLE__
    .readdir = CLIO_FUSE_TIMED(readdir, cte_fuse_readdir_darwin),
#else
    .readdir = CLIO_FUSE_TIMED(readdir, cte_fuse_readdir),
#endif
    .releasedir = CLIO_FUSE_TIMED(releasedir, cte_fuse_releasedir),
    .fsyncdir = CLIO_FUSE_TIMED(fsyncdir, cte_fuse_fsyncdir),
    .init = cte_fuse_init,
    .destroy = cte_fuse_destroy,
    .create = CLIO_FUSE_TIMED(create, cte_fuse_create),
    .utimens = CLIO_FUSE_TIMED(utimens, cte_fuse_utimens),
#ifdef __linux__
    .fallocate = CLIO_FUSE_TIMED(fallocate, cte_fuse_fallocate),
#endif
};
