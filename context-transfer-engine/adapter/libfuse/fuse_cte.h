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

#ifndef CLIO_CTE_ADAPTER_LIBFUSE_FUSE_CTE_H_
#define CLIO_CTE_ADAPTER_LIBFUSE_FUSE_CTE_H_

#ifdef CLIO_CTE_FUSE_ENABLED
#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 35
#endif
#include <fuse3/fuse.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>  // struct stat + S_IF* for fuse_cte.cc's getattr (Linux)
#include <unistd.h>    // getuid/getgid/read used by fuse_cte.cc on Linux
#endif

#include "clio_cte/core/core_client.h"
#include "clio_cte/core/core_tasks.h"

namespace clio::cae::fuse {

/** Default page size for blob I/O */
static constexpr size_t kDefaultPageSize = 1024 * 1024;  // 1 MB

/**
 * Max in-flight async PutBlobs per open file. When a write would push the
 * count past this, the write thread back-pressures by reaping completed
 * entries inline; if none have completed yet, it blocks on the oldest
 * entry until it does. Bounding the queue is also important to avoid
 * exhausting the daemon's FutureShm allocator under sustained writes.
 */
static constexpr size_t kMaxInFlightWrites = 8;

/**
 * In-flight async PutBlob — owns the clio Future and the SHM buffer
 * the put is reading from. Both must outlive the put (so the daemon can
 * still copy out of shm_buf when its handler runs). Stored in a
 * std::list on the handle so iterators stay stable across insertion and
 * erase — vector reallocation move-constructs Futures, which leaves the
 * moved-from source with a non-null future_shm_ that the destructor can
 * race with the in-flight handler.
 */
struct PendingWrite {
  clio::run::Future<clio::cte::core::PutBlobTask> task;
  ctp::ipc::FullPtr<char> shm_buf;
};

/**
 * CTE-backed filesystem helpers.
 *
 * Design:
 *   - Each file is a CTE Tag whose name is the absolute FUSE path.
 *   - Directories are implicit: a directory "/a/b" exists if any tag
 *     starts with "/a/b/".
 *   - readdir uses AsyncTagQuery with a regex to find direct children.
 *   - File data is stored as page-indexed blobs ("0", "1", "2", ...).
 *
 * No custom DirectoryTree or FsNode — all metadata lives in CTE.
 */

/** Per-open-file handle stored in fuse_file_info::fh */
struct FuseFileHandle {
  clio::cte::core::TagId tag_id;
  std::string path;
  int flags;
  /** Async PutBlob queue — back-pressured inside cte_fuse_write itself,
   *  final drain in cte_fuse_release. std::list to keep iterators stable. */
  std::mutex pending_mu;
  std::list<PendingWrite> pending_writes;
};

// ============================================================================
// CTE helper functions (async API wrappers)
// ============================================================================

// FPP-style FUSE adapter: every helper is pinned to PoolQuery::Local so the
// tag + blob metadata for rank R's file lives on rank R's local node.
// Default pool_queries in the client headers (Dynamic / Broadcast /
// DirectHash for new tags) would otherwise hash the tag to some peer
// node, after which CteTagExists(Local) wouldn't find it and IOR's open
// would fail with ENOENT. Explicit Local also bypasses the broadcast
// aggregation path on send_map_, which is where the 4n×256m hang lived.

/** Query CTE for the authoritative tag size */
static inline size_t CteGetTagSize(const clio::cte::core::TagId &tag_id) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncGetTagSize(tag_id, clio::run::PoolQuery::Local());
  task.Wait();
  if (task->GetReturnCode() != 0) return 0;
  return task->tag_size_;
}

/** Delete a CTE tag by name */
static inline void CteDelTag(const std::string &tag_name) {
  auto *cte_client = CLIO_CTE_CLIENT;
  auto task = cte_client->AsyncDelTag(tag_name, clio::run::PoolQuery::Local());
  task.Wait();
}

/** Get or create a CTE tag, returning its TagId. Returns null id on failure. */
static inline clio::cte::core::TagId CteGetOrCreateTag(const std::string &name) {
  auto *cte_client = CLIO_CTE_CLIENT;
  // pool_query=Local overrides ScheduleTask's default DirectHash routing for
  // new tags — without it the tag is hashed to a peer node and a subsequent
  // CteTagExists(Local) from the same FUSE adapter won't find it.
  auto task = cte_client->AsyncGetOrCreateTag(
      name, clio::cte::core::TagId::GetNull(), clio::run::PoolQuery::Local());
  task.Wait();
  if (task->GetReturnCode() != 0) return clio::cte::core::TagId::GetNull();
  return task->tag_id_;
}

/** Check if a tag exists by name using TagQuery with exact match */
static inline bool CteTagExists(const std::string &tag_name) {
  auto *cte_client = CLIO_CTE_CLIENT;
  // Escape regex special chars and do exact match
  std::string escaped;
  for (char c : tag_name) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      escaped += '\\';
    }
    escaped += c;
  }
  // pool_query=Local — the YAML pins this CTE pool to local-only and
  // every rank in FPP touches its own tag on its own node, so a broadcast
  // here is wasted fanout. It is also unsafe: the broadcast aggregation
  // path goes through send_map_ + completed_replicas_ and any single
  // missed/duplicate response leaves the originating future un-completed,
  // which is exactly the hang reproduced under 4n×256m load (one rank's
  // getattr parked in TagQuery.Wait, all daemon workers idle).
  auto task = cte_client->AsyncTagQuery(escaped, 1, clio::run::PoolQuery::Local());
  task.Wait();
  return task->GetReturnCode() == 0 && !task->results_.empty();
}

/**
 * Query CTE for the direct FILE children of a directory path.
 * For directory "/a/b", returns the basenames of tags "/a/b/<name>" that are
 * leaves (files). Subdirectories are intentionally excluded — they are reported
 * separately by CteListSubdirs.
 *
 * GetOrCreateTagChain materializes an intermediate tag for every path component
 * (creating "/a/b/sub/c.txt" also creates the tag "/a/b/sub"), so a naive
 * "^/a/b/[^/]+$" query would return the directory "sub" alongside the real
 * files. We instead query the whole subtree once and classify each immediate
 * component: a component is a directory if any tag exists *beneath* it
 * ("/a/b/<name>/..."), otherwise it is a file.
 */
static inline std::vector<std::string>
CteListDirectChildren(const std::string &dir_path) {
  auto *cte_client = CLIO_CTE_CLIENT;

  // Build regex: escape dir_path, then match the subtree (one-or-more chars
  // after the trailing slash, at any depth).
  std::string escaped;
  for (char c : dir_path) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      escaped += '\\';
    }
    escaped += c;
  }
  // Ensure trailing slash
  if (!escaped.empty() && escaped.back() != '/') escaped += '/';
  std::string regex = "^" + escaped + ".+";

  auto task = cte_client->AsyncTagQuery(regex, 0, clio::run::PoolQuery::Local());
  task.Wait();

  std::vector<std::string> basenames;
  if (task->GetReturnCode() != 0) return basenames;

  // Partition immediate components into leaf-file candidates (tag is exactly
  // "/dir/<name>") and directories (a tag "/dir/<name>/..." exists below).
  size_t prefix_len = dir_path.size();
  if (!dir_path.empty() && dir_path.back() != '/') prefix_len++;
  std::vector<std::string> file_candidates;
  std::vector<std::string> dir_names;
  for (const auto &full_path : task->results_) {
    if (full_path.size() <= prefix_len) continue;
    std::string remainder = full_path.substr(prefix_len);
    size_t slash_pos = remainder.find('/');
    if (slash_pos == std::string::npos) {
      file_candidates.push_back(remainder);
    } else {
      std::string subdir = remainder.substr(0, slash_pos);
      if (std::find(dir_names.begin(), dir_names.end(), subdir) ==
          dir_names.end()) {
        dir_names.push_back(subdir);
      }
    }
  }

  // Keep only leaf files (exclude any candidate that is actually a directory),
  // de-duplicated.
  for (const auto &name : file_candidates) {
    if (std::find(dir_names.begin(), dir_names.end(), name) != dir_names.end()) {
      continue;  // it is a directory, not a file
    }
    if (std::find(basenames.begin(), basenames.end(), name) ==
        basenames.end()) {
      basenames.push_back(name);
    }
  }
  return basenames;
}

/**
 * Find all unique immediate subdirectory names under dir_path.
 * For dir "/a", if tags "/a/b/c.txt" and "/a/b/d.txt" and "/a/e/f.txt" exist,
 * returns {"b", "e"}.
 */
static inline std::vector<std::string>
CteListSubdirs(const std::string &dir_path) {
  auto *cte_client = CLIO_CTE_CLIENT;

  // Match any tag that has at least two more path components after dir_path
  std::string escaped;
  for (char c : dir_path) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      escaped += '\\';
    }
    escaped += c;
  }
  if (!escaped.empty() && escaped.back() != '/') escaped += '/';
  // Match tags with at least one more slash after the child component
  std::string regex = "^" + escaped + "[^/]+/.*";

  auto task = cte_client->AsyncTagQuery(regex, 0, clio::run::PoolQuery::Local());
  task.Wait();

  // Extract unique immediate subdirectory names
  std::vector<std::string> subdirs;
  size_t prefix_len = dir_path.size();
  if (!dir_path.empty() && dir_path.back() != '/') prefix_len++;

  for (const auto &full_path : task->results_) {
    if (full_path.size() <= prefix_len) continue;
    std::string remainder = full_path.substr(prefix_len);
    size_t slash_pos = remainder.find('/');
    if (slash_pos != std::string::npos) {
      std::string subdir = remainder.substr(0, slash_pos);
      // Deduplicate
      if (std::find(subdirs.begin(), subdirs.end(), subdir) == subdirs.end()) {
        subdirs.push_back(subdir);
      }
    }
  }
  return subdirs;
}

/**
 * Check if a directory path has any tags underneath it.
 * A directory exists if any tag starts with "dir_path/".
 */
static inline bool CteDirExists(const std::string &dir_path) {
  auto *cte_client = CLIO_CTE_CLIENT;
  std::string escaped;
  for (char c : dir_path) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|') {
      escaped += '\\';
    }
    escaped += c;
  }
  if (!escaped.empty() && escaped.back() != '/') escaped += '/';
  std::string regex = "^" + escaped + ".*";
  auto task = cte_client->AsyncTagQuery(regex, 1, clio::run::PoolQuery::Local());
  task.Wait();
  return task->GetReturnCode() == 0 && !task->results_.empty();
}

/**
 * Page-based PutBlob: allocate SHM, copy data, async put, wait, free.
 *
 * Synchronous variant — blocks until the daemon completes the put.
 * Kept for callers that need a return code per-page. The async path used
 * by cte_fuse_write goes through CtePutBlobAsync below.
 */
static inline bool CtePutBlob(const clio::cte::core::TagId &tag_id,
                              const std::string &blob_name, const char *data,
                              size_t data_size, size_t blob_off) {
  auto *ipc_manager = CLIO_IPC;
  auto *cte_client = CLIO_CTE_CLIENT;
  ctp::ipc::FullPtr<char> shm_buf = ipc_manager->AllocateBuffer(data_size);
  if (shm_buf.IsNull()) return false;
  memcpy(shm_buf.ptr_, data, data_size);
  ctp::ipc::ShmPtr<> shm_ptr(shm_buf.shm_);
  auto task = cte_client->AsyncPutBlob(
      tag_id, blob_name, blob_off, data_size, shm_ptr,
      /*score*/ -1.0f, clio::cte::core::Context(), /*flags*/ 0u,
      clio::run::PoolQuery::Local());
  task.Wait();
  ipc_manager->FreeBuffer(shm_buf);
  return task->GetReturnCode() == 0;
}

/**
 * Async PutBlob with bounded in-flight queue + inline back-pressure.
 *
 * Submits AsyncPutBlob and parks the Future + SHM buffer on the open
 * file handle's pending_writes list. If the list now exceeds
 * kMaxInFlightWrites, the call back-pressures right here in write()
 * before returning — first scanning the whole list with non-blocking
 * Wait(0) to reap whatever has finished, then, if nothing has, blocking
 * on the oldest entry's Wait() until at least one slot frees up.
 *
 * Doing the reap in write() (as opposed to deferring to release()/
 * fsync()) keeps both the clio FutureShm allocator and the per-fd
 * SHM buffer footprint bounded under sustained writes, while still
 * letting the FUSE kernel pipeline up to kMaxInFlightWrites concurrent
 * chunks per fd before the writer stalls.
 */
static inline bool CtePutBlobAsync(struct FuseFileHandle *handle,
                                   const std::string &blob_name,
                                   const char *data, size_t data_size,
                                   size_t blob_off) {
  auto *ipc_manager = CLIO_IPC;
  auto *cte_client = CLIO_CTE_CLIENT;
  ctp::ipc::FullPtr<char> shm_buf = ipc_manager->AllocateBuffer(data_size);
  if (shm_buf.IsNull()) return false;
  memcpy(shm_buf.ptr_, data, data_size);
  ctp::ipc::ShmPtr<> shm_ptr(shm_buf.shm_);
  auto task = cte_client->AsyncPutBlob(
      handle->tag_id, blob_name, blob_off, data_size, shm_ptr,
      /*score*/ -1.0f, clio::cte::core::Context(), /*flags*/ 0u,
      clio::run::PoolQuery::Local());

  std::lock_guard<std::mutex> lk(handle->pending_mu);
  handle->pending_writes.push_back(
      PendingWrite{std::move(task), shm_buf});

  // Inline back-pressure. While over the cap, sweep for completed tasks
  // and reap them; if a full sweep finds none complete, block on the
  // oldest one (which is most likely to be next anyway).
  while (handle->pending_writes.size() > kMaxInFlightWrites) {
    bool reaped_any = false;
    auto it = handle->pending_writes.begin();
    while (it != handle->pending_writes.end()) {
      // Wait(0) polls without blocking. Returning true also marks the
      // Future consumed_, which lets ~Future free the FutureShm cleanly
      // when the std::list node is erased.
      if (it->task.Wait(0)) {
        ipc_manager->FreeBuffer(it->shm_buf);
        it = handle->pending_writes.erase(it);
        reaped_any = true;
      } else {
        ++it;
      }
    }
    if (reaped_any) continue;
    // Nothing complete yet — block on the oldest entry. Wait(-1) is the
    // indefinite blocking wait; it will consume the Future on completion.
    auto &front = handle->pending_writes.front();
    front.task.Wait();
    ipc_manager->FreeBuffer(front.shm_buf);
    handle->pending_writes.pop_front();
  }
  return true;
}

/**
 * Final drain — called only from cte_fuse_release to close out whatever
 * is left in the queue when the fd is closed (release() is the FUSE op
 * that runs at last close). The primary back-pressure happens inside
 * CtePutBlobAsync; this exists purely to avoid leaking SHM buffers and
 * Futures when the writer closes before the queue empties.
 *
 * Returns 0 on success, -EIO if any task reported a non-zero return.
 */
static inline int DrainPendingWrites(struct FuseFileHandle *handle) {
  std::list<PendingWrite> drain;
  {
    std::lock_guard<std::mutex> lk(handle->pending_mu);
    drain.swap(handle->pending_writes);
  }
  if (drain.empty()) return 0;
  auto *ipc_manager = CLIO_IPC;
  int rc = 0;
  for (auto &pw : drain) {
    pw.task.Wait();
    if (pw.task->GetReturnCode() != 0) rc = -EIO;
    ipc_manager->FreeBuffer(pw.shm_buf);
  }
  return rc;
}

/**
 * Page-based GetBlob: allocate SHM, async get, wait, copy out, free.
 */
static inline bool CteGetBlob(const clio::cte::core::TagId &tag_id,
                              const std::string &blob_name, char *data,
                              size_t data_size, size_t blob_off) {
  auto *ipc_manager = CLIO_IPC;
  auto *cte_client = CLIO_CTE_CLIENT;
  ctp::ipc::FullPtr<char> shm_buf = ipc_manager->AllocateBuffer(data_size);
  if (shm_buf.IsNull()) return false;
  ctp::ipc::ShmPtr<> shm_ptr(shm_buf.shm_);
  auto task = cte_client->AsyncGetBlob(
      tag_id, blob_name, blob_off, data_size, /*flags*/ 0u, shm_ptr,
      clio::run::PoolQuery::Local());
  task.Wait();
  bool ok = (task->GetReturnCode() == 0);
  if (ok) memcpy(data, shm_buf.ptr_, data_size);
  ipc_manager->FreeBuffer(shm_buf);
  return ok;
}

/** Escape a string for use as a literal in std::regex */
static inline std::string RegexEscape(const std::string &s) {
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

}  // namespace clio::cae::fuse

#ifdef CLIO_CTE_FUSE_ENABLED
// The FUSE operation table lives in fuse_cte.cc (defined at global scope, next
// to the callbacks it points at). fuse_cte_main.cc — the process-entry / mount
// glue — references it, so it is declared here with external linkage. Only the
// FUSE builds (which include <fuse3/fuse.h> above) see this declaration.
extern const struct fuse_operations cte_fuse_ops;
#endif  // CLIO_CTE_FUSE_ENABLED

#endif  // CLIO_CTE_ADAPTER_LIBFUSE_FUSE_CTE_H_
