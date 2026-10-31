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

/**
 * Inode-table blocks: homes, caches and the push protocol (see fs_inode_table.h
 * for the layout).
 *
 * The CTE stores every block; clio-fs caches them. Each block has one HOME
 * (the container owning its CTE blob) that applies every change, writes the
 * blob and pushes the change to each HOLDER (a container that fetched the
 * block) before it acknowledges the change. Holders therefore answer
 * lookups, stats and listings from memory, with no CTE access, and never
 * see a change later than the caller that made it.
 *
 * - Group commit: changes that arrive while a commit is in flight are
 *   written and pushed together by the next one, so a burst of creates in
 *   one directory costs a few blob writes and pushes, not one each.
 * - Liveness: a home waits only for holders whose nodes are alive; a node
 *   that dies is dropped from every holder set, and a restarted node starts
 *   with an empty cache.
 * - Failover: a block's home is the CTE's placement of its blob, including
 *   the successor that serves a dead owner's blobs. A cached copy remembers
 *   its home and is refetched when that changes.
 */
#include <cerrno>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <clio_cte/core/blob_placement.h>
#include <clio_cte/filesystem/filesystem_runtime.h>

namespace clio::cte::filesystem {

namespace {

/** Wall-clock nanoseconds (the timestamps stat reports). */
inline clio::run::u64 NowNs() { return clio::cte::core::GetWallTimeNs(); }

/** Wait between checks while another task commits the same block (us). */
constexpr double kCommitPollUs = 10.0;
/** Wait between checks while another task loads the same block (us). */
constexpr double kLoadPollUs = 20.0;
/** Version jump when a home loads a block from its blob: more changes
 *  than one incarnation of a home can make to one block. */
constexpr clio::run::u64 kIncarnationStride = 1ull << 32;
/** A block load, wait or commit slower than this is reported (seconds). */
constexpr double kSlowBlockOpS = 10.0;
/** How long a load keeps asking a block home that did not answer, waiting
 *  for failover to name its stand-in (s): node death is declared within
 *  ~10 s. Past it the load fails EIO. */
constexpr int kHomeFailoverWaitS = 15;
/** Pause between those attempts (us). */
constexpr double kHomeFailoverPollUs = 250000.0;

/** Encode a list of block indices. */
std::string EncIndices(const std::vector<clio::run::u32> &v) {
  std::string out;
  FsEnc e(&out);
  e.U32(static_cast<clio::run::u32>(v.size()));
  for (clio::run::u32 k : v) e.U32(k);
  return out;
}

/** Decode a list of block indices. */
std::vector<clio::run::u32> DecIndices(const std::string &s) {
  std::vector<clio::run::u32> out;
  FsDec d(s.data(), s.size());
  clio::run::u32 n = 0, k = 0;
  if (!d.U32(&n)) return out;
  for (clio::run::u32 i = 0; i < n && d.U32(&k); ++i) out.push_back(k);
  return out;
}

/** Context for writing a clio-fs metadata blob. */
clio::cte::core::Context MetaCtx(bool volatile_only) {
  clio::cte::core::Context ctx;
  // Storing the record is not a change to any file (no ctime bump), and it
  // belongs on a tier that survives a restart when there is one.
  ctx.op_flags_ |= clio::cte::core::Context::kMetaBlob;
  ctx.min_persistence_level_ = volatile_only ? 0 : 1;
  return ctx;
}

}  // namespace

// ===========================================================================
// Homes and blob I/O
// ===========================================================================

clio::run::u32 Runtime::BlockHome(clio::run::u64 dir, clio::run::u32 k) {
  const clio::run::u32 n = NumContainers();
  if (n <= 1) return 0;
  const clio::run::u32 owner =
      clio::cte::core::BlobHash(FsUnpack(dir), TableBlockName(k)) % n;
  return clio::cte::core::FailoverContainer(pool_id_, owner);
}

clio::run::TaskResume Runtime::ReadBlockBlob(clio::run::u64 dir,
                                             clio::run::u32 k, InodeTableBlock *out,
                                             int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  // Before the journal is replayed, "not in the journal" means nothing.
  while (!dir_ready_.load(std::memory_order_acquire)) {
    CLIO_CO_AWAIT(clio::run::yield(1000.0));
  }
  if (dir_journal_) {
    rc = TakeJournaledBlock(dir, k, out);
    if (rc == ENOENT) {
      // Nothing in this journal: either a stand-in for a dead home (the
      // block is in that node's journal and in its replicated CTE mirror,
      // #1263) or a block that does not exist. The CTE read tells them
      // apart: only "no such blob" is ENOENT.
      CLIO_CO_AWAIT(ReadBlockCte(dir, k, out, rc));
      CLIO_CO_RETURN;
    }
    if (rc != 0) CLIO_CO_RETURN;
    // A stand-in may have changed the block while this node was down; the
    // CTE hand-back delivered its copy. Take whichever is newer. A failed
    // CTE read never fails a read the journal answered.
    {
      auto cte_copy = std::make_shared<InodeTableBlock>();
      int cte_rc = 0;
      CLIO_CO_AWAIT(ReadBlockCte(dir, k, cte_copy.get(), cte_rc));
      if (cte_rc == 0 && cte_copy->version_ > out->version_) {
        *out = std::move(*cte_copy);
      }
    }
    rc = 0;
    CLIO_CO_RETURN;
  }
  CLIO_CO_AWAIT(ReadBlockCte(dir, k, out, rc));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReadBlockCte(clio::run::u64 dir,
                                            clio::run::u32 k,
                                            InodeTableBlock *out, int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  bool ready = false;
  CLIO_CO_AWAIT(AwaitCteReady(ready));
  if (!ready) {
    rc = EIO;
    CLIO_CO_RETURN;
  }
  const clio::cte::core::TagId tag = FsUnpack(dir);
  auto sz = cte_.AsyncGetBlobSize(tag, TableBlockName(k));
  CLIO_CO_AWAIT(sz);
  // Only "no such blob" means the block does not exist. Any other failure
  // (a lost node, a timeout, an unroutable request) must not read as an
  // empty directory: a create would then write a fresh block over the real
  // one.
  if (sz->GetReturnCode() != 0 && sz->GetReturnCode() != kCteBlobNotFoundRc) {
    rc = EIO;
    CLIO_CO_RETURN;
  }
  if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
    rc = ENOENT;
    CLIO_CO_RETURN;
  }
  std::string bytes(sz->size_, '\0');
  auto g = cte_.AsyncGetBlob(tag, TableBlockName(k), 0, bytes.size(), 0u,
                             bytes.data());
  CLIO_CO_AWAIT(g);
  if (g->GetReturnCode() != 0) {
    // Deleted between the two calls (an rmdir dropped it): it is gone.
    auto again = cte_.AsyncGetBlobSize(tag, TableBlockName(k));
    CLIO_CO_AWAIT(again);
    if (again->GetReturnCode() == kCteBlobNotFoundRc ||
        (again->GetReturnCode() == 0 && again->size_ == 0)) {
      rc = ENOENT;
      CLIO_CO_RETURN;
    }
  }
  // A block that shrank leaves stale bytes past its end; the codec stops at
  // the encoded length.
  if (g->GetReturnCode() != 0 ||
      !DecodeTableBlock(bytes.data(), bytes.size(), out) || out->dir_ != dir ||
      out->index_ != k) {
    if (g->GetReturnCode() == 0 && IsOldDirBlock(bytes.data(), bytes.size())) {
      HLOG(kError, "filesystem: directory {} was written by an older clio-fs "
           "(directory format CDB1); this volume must be recreated", dir);
    } else {
      HLOG(kError, "filesystem: inode-table block {}/{} unreadable", dir, k);
    }
    rc = EIO;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::WriteBlockBlob(clio::run::u64 dir,
                                              clio::run::u32 k,
                                              const std::string &image,
                                              int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  while (!dir_ready_.load(std::memory_order_acquire)) {
    CLIO_CO_AWAIT(clio::run::yield(1000.0));
  }
  if (dir_journal_) {
    rc = JournalAppend(FsLogRec::kBlockImage, image);
    if (rc != 0) CLIO_CO_RETURN;
  }
  // With the journal on, the CTE blob is the replicated mirror a stand-in
  // reads while this node is down (#1263); the change is acknowledged only
  // once both are written.
  CLIO_CO_AWAIT(PutBlockCte(dir, k, image, rc));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PutBlockCte(clio::run::u64 dir,
                                           clio::run::u32 k,
                                           const std::string &image,
                                           int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  bool ready = false;
  CLIO_CO_AWAIT(AwaitCteReady(ready));
  if (!ready) {
    rc = EIO;
    CLIO_CO_RETURN;
  }
  const clio::cte::core::TagId tag = FsUnpack(dir);
  // Pad to a power of two (>= 4 KiB). Every create grows the image a little;
  // a blob rewritten at a slightly larger size each time gains one more small
  // extent per write, and every rewrite then touches all of them (creates
  // slowed linearly with directory size). Padded, the blob only changes size
  // a few times and stays a few extents. The codec ignores the padding.
  size_t cap = 4096;
  while (cap < image.size()) cap <<= 1;
  std::string padded = image;
  padded.resize(cap, '\0');
  auto p = cte_.AsyncPutBlob(tag, TableBlockName(k), 0, padded.size(),
                             padded.data(), -1.0f,
                             MetaCtx(inode_volatile_only_), 0u,
                             clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(p);
  if (p->GetReturnCode() != 0 && !inode_volatile_only_) {
    // No non-volatile tier accepts it: keep it in RAM (lost on restart).
    p = cte_.AsyncPutBlob(tag, TableBlockName(k), 0, padded.size(),
                          padded.data(), -1.0f, MetaCtx(true), 0u,
                          clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() == 0) {
      HLOG(kWarning, "filesystem: no non-volatile CTE tier accepts metadata; "
           "directories do not survive a restart");
      inode_volatile_only_ = true;
    }
  }
  if (p->GetReturnCode() != 0) {
    HLOG(kError, "filesystem: writing inode-table block {}/{} failed (rc {})",
         dir, k, p->GetReturnCode());
    // A full store is ENOSPC, so a create or mkdir on a full filesystem says
    // so (as on ext4); anything else is an I/O error.
    rc = clio::cte::core::PutRcIsNoSpace(p->GetReturnCode()) ? ENOSPC : EIO;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Loading and caching
// ===========================================================================

clio::run::TaskResume Runtime::LoadBlock(clio::run::u64 dir, clio::run::u32 k,
                                         std::shared_ptr<TableBlockSlot> &out,
                                         int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  out.reset();
  auto wait_t0 = std::chrono::steady_clock::now();
  double next_report_s = kSlowBlockOpS;
  if (NumContainers() > 1) {
    const clio::run::u64 hd = CLIO_IPC->HeldDeadSeq();
    clio::run::u64 seen = seen_held_dead_seq_.load(std::memory_order_acquire);
    if (hd != seen &&
        seen_held_dead_seq_.compare_exchange_strong(seen, hd)) {
      DropHomeBlocksAfterFailover();
    }
  }
  for (int attempt = 0;; ++attempt) {
    const TableBlockKey key{dir, k};
    const clio::run::u32 home = BlockHome(dir, k);
    const bool mine = home == container_id_;
    if (mine && NumContainers() > 1 &&
        clio::cte::core::InMinority(pool_id_)) {
      // Split brain, home side: this node sees most of the pool dead, so it
      // is (probably) the one cut off, and the majority has a stand-in
      // serving -- and changing -- this block. Answering from our copy
      // reported a file created over there as ENOENT. Fail instead.
      rc = EIO;
      CLIO_CO_RETURN;
    }
    bool busy = false;
    bool busy_commit = false;
    const char *busy_stage = "";
    std::shared_ptr<TableBlockSlot> expired;  // a copy whose lease ran out
    if (FindCachedBlock(key, home, &out, &expired)) CLIO_CO_RETURN;
    if (expired != nullptr) {
      // Ask the home before serving it again: a change it could not push
      // here (a partition) must not be answered from the old copy.
      CLIO_CO_AWAIT(RevalidateBlock(expired, dir, k, home, rc));
      if (rc == 0) {
        out = expired;
        CLIO_CO_RETURN;
      }
      if (rc != kFsRedirect) {
        // Unreachable: never serve the old copy. But a home that just died
        // is not yet declared dead (~10 s); wait for failover to name its
        // stand-in, which serves the block from its replicated CTE copy
        // (#1263), rather than fail a lookup the cluster can answer.
        if (std::chrono::steady_clock::now() - wait_t0 <
            std::chrono::seconds(kHomeFailoverWaitS)) {
          rc = 0;
          CLIO_CO_AWAIT(clio::run::yield(kHomeFailoverPollUs));
          continue;
        }
        CLIO_CO_RETURN;
      }
      rc = 0;
      std::lock_guard<std::mutex> g(ns_mu_);
      auto it = blocks_.find(key);
      if (it != blocks_.end() && it->second == expired &&
          !expired->committing_) {
        blocks_.erase(it);  // the home moved: load it from the new one
      }
      continue;
    }
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      auto it = blocks_.find(key);
      if (it != blocks_.end()) {
        TableBlockSlot &s = *it->second;
        if (s.home_ == mine && (mine || s.home_id_ == home)) {
          out = it->second;  // revalidated by another task meanwhile
          CLIO_CO_RETURN;
        }
        // Its home moved (failover or return): this copy is not current.
        if (s.committing_) busy = busy_commit = true;
        else blocks_.erase(it);
      }
      const char *stage = "";
      if (!busy) {
        auto lit = loading_.find(key);
        busy = lit != loading_.end();
        if (busy) {
          stage = lit->second;
        } else {
          loading_.emplace(key, mine ? "home load" : "fetch from home");
        }
      }
      busy_stage = stage;
    }
    if (busy) {
      const double waited = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - wait_t0).count();
      if (waited > next_report_s) {
        HLOG(kWarning, "filesystem: waiting {} ms for inode-table block "
             "{}/{} (home {}, mine {}): {}{}",
             static_cast<clio::run::u64>(waited * 1000.0), dir, k, home, mine,
             busy_commit ? "a moved copy is still committing"
                         : "another task is loading it, at: ",
             busy_commit ? "" : busy_stage);
        next_report_s += kSlowBlockOpS;
      }
      CLIO_CO_AWAIT(clio::run::yield(kLoadPollUs));
      continue;
    }
    const auto load_t0 = std::chrono::steady_clock::now();
    auto slot = std::make_shared<TableBlockSlot>();
    int lrc = 0;
    if (mine) {
      CLIO_CO_AWAIT(LoadHomeBlock(dir, k, slot, lrc));
    } else {
      CLIO_CO_AWAIT(FetchBlock(dir, k, home, slot, lrc));
    }
    {
      const double took = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - load_t0).count();
      if (took > kSlowBlockOpS) {
        HLOG(kWarning, "filesystem: loading inode-table block {}/{} from {} "
             "took {} ms (rc {})", dir, k, mine ? "its blob" : "its home",
             static_cast<clio::run::u64>(took * 1000.0), lrc);
      }
    }
    InstallLoadedBlock(key, slot, lrc, &out);
    if (lrc == kFsRedirect && attempt < 50) {
      // The home and this node disagree about membership for a moment.
      CLIO_CO_AWAIT(clio::run::yield(1000.0));
      continue;
    }
    if (!mine && lrc == EIO && BlockHome(dir, k) != home && attempt < 3) {
      // The home died while it was asked, and failover has since moved the
      // block to its successor, which holds its replica: ask there instead
      // of failing a lookup the cluster can still answer.
      continue;
    }
    if (!mine && lrc == EIO &&
        std::chrono::steady_clock::now() - wait_t0 <
            std::chrono::seconds(kHomeFailoverWaitS)) {
      // The home did not answer but is not yet declared dead (~10 s): wait
      // for failover to move the block rather than fail now (#1263).
      CLIO_CO_AWAIT(clio::run::yield(kHomeFailoverPollUs));
      continue;
    }
    rc = lrc == kFsRedirect ? EIO : lrc;
    if (rc == 0 && slot->home_ &&
        (slot->persist_dirty_ || slot->holders_unknown_)) {
      int wrc = 0;
      CLIO_CO_AWAIT(CommitBlock(slot, slot->blk_.version_, wrc));
    }
    CLIO_CO_RETURN;
  }
  CLIO_TASK_BODY_END
}

void Runtime::DropHomeBlocksAfterFailover() {
  // Rejoin notice (decision stale_home_dir): a peer held this node dead,
  // so a stand-in may have served and changed blocks homed here (a one-way
  // partition never restarts this node, and a home trusts its copy with no
  // lease: a file created over there stayed invisible here forever). Drop
  // every CLEAN home block; the next use reloads it exactly as a restart
  // does -- journal and the handed-back CTE copy, newest version wins --
  // and resyncs its holders. Blocks with unwritten changes or work in
  // flight are kept: dropping them would lose acknowledged changes.
  size_t dropped = 0, kept = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    for (auto it = blocks_.begin(); it != blocks_.end();) {
      const TableBlockSlot &s = *it->second;
      if (!s.home_) {
        ++it;
        continue;
      }
      if (s.committing_ || s.splitting_ || s.persist_dirty_ ||
          !s.unpushed_.empty()) {
        ++kept;
        ++it;
        continue;
      }
      it = blocks_.erase(it);
      ++dropped;
    }
  }
  HLOG(kWarning, "filesystem: a peer held this node dead (rejoin notice); "
       "dropped {} home directory block(s) to reload them, kept {} with "
       "changes in flight", dropped, kept);
}

bool Runtime::FindCachedBlock(const TableBlockKey &key, clio::run::u32 home,
                              std::shared_ptr<TableBlockSlot> *out,
                              std::shared_ptr<TableBlockSlot> *expired) {
  const bool mine = home == container_id_;
  std::lock_guard<std::mutex> g(ns_mu_);
  auto it = blocks_.find(key);
  if (it == blocks_.end()) return false;
  TableBlockSlot &s = *it->second;
  if (s.home_ != mine || (!mine && s.home_id_ != home)) return false;
  if (mine || SteadyMs() < s.lease_until_ms_) {
    *out = it->second;
    return true;
  }
  *expired = it->second;
  return false;
}

void Runtime::DropGoneHolders(
    const std::vector<clio::run::u32> &gone,
    const std::map<clio::run::u32, clio::run::u64> &pushed,
    std::map<clio::run::u32, clio::run::u64> *holders,
    std::map<clio::run::u32, clio::run::u64> *leases) {
  const clio::run::u64 now = SteadyMs();
  for (clio::run::u32 h : gone) {
    // Only the registration this push went to: a holder that dropped its
    // copy and fetched again meanwhile stays registered.
    auto it = holders->find(h);
    auto pt = pushed.find(h);
    if (it == holders->end() || pt == pushed.end() ||
        it->second != pt->second) {
      continue;
    }
    holders->erase(it);
    auto lt = leases->find(h);
    if (lt != leases->end() && lt->second <= now) leases->erase(lt);
  }
}

void Runtime::InstallLoadedBlock(const TableBlockKey &key,
                                 std::shared_ptr<TableBlockSlot> slot, int lrc,
                                 std::shared_ptr<TableBlockSlot> *out) {
  std::lock_guard<std::mutex> g(ns_mu_);
  loading_.erase(key);
  auto ev = early_.find(key);
  if (lrc == 0) {
    if (ev != early_.end()) {
      // Pushes that arrived while the copy was in flight.
      for (const TableDelta &x : ev->second) {
        if (x.kind_ == TableDeltaKind::kOps &&
            x.new_version_ > slot->blk_.version_) {
          ApplyTableOps(&slot->blk_, x);
        }
      }
    }
    blocks_[key] = slot;
    *out = slot;
    // A home block now lives in blocks_; compaction takes it from there.
    if (slot->home_) journaled_.erase(key);
  }
  if (ev != early_.end()) early_.erase(ev);
}

/** CLIO_TRACE_DIRBLOCK=1: log every home directory-block load and commit
 *  (version, row count), to find where acknowledged entries disappear. */
static bool TraceDirBlock() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_TRACE_DIRBLOCK");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

clio::run::TaskResume Runtime::LoadHomeBlock(clio::run::u64 dir,
                                             clio::run::u32 k,
                                             std::shared_ptr<TableBlockSlot> slot,
                                             int &lrc) {
  CLIO_TASK_BODY_BEGIN
  lrc = 0;
  const TableBlockKey key{dir, k};
  SetLoadStage(key, "home load: reading the block blob (GetBlobSize/GetBlob)");
  CLIO_CO_AWAIT(ReadBlockBlob(dir, k, &slot->blk_, lrc));
  if (TraceDirBlock()) {
    HLOG(kWarning, "[DIRBLOCK] home load {}/{} on container {}: rc {} "
         "version {} rows {}", dir, k, container_id_, lrc,
         slot->blk_.version_, slot->blk_.rows_.size());
  }
  // Loaded from the journal: its deltas build on the image there.
  slot->journal_image_ = lrc != 0;
  if (lrc == ENOENT && dir == FsPack(FsRootId()) && k == 0) {
    // "/" exists by definition: its home materializes its table on first
    // use (its inode record is materialized by its own home, EnsureInode).
    slot->blk_.dir_ = dir;
    slot->blk_.version_ = 1;
    slot->blk_.mtime_ = NowNs();
    slot->persist_dirty_ = true;
    lrc = 0;
  }
  slot->home_ = true;
  slot->durable_version_ = slot->persist_dirty_ ? 0 : slot->blk_.version_;
  if (lrc == 0 && NumContainers() > 1) {
    // Copies cached elsewhere registered with an earlier incarnation of
    // this home and may be ahead of the blob (non-durable changes bump
    // the version too): resync them before anyone reads through us.
    slot->holders_unknown_ = true;
    slot->durable_version_ = 0;
    // Versions this incarnation hands out must exceed every version an
    // earlier one did, or a cached copy that is ahead of the blob would
    // look current to a revalidation. Persist the jump before serving,
    // so the next incarnation starts past it too.
    slot->blk_.version_ += kIncarnationStride;
    int wrc = 0;
    SetLoadStage(key, "home load: persisting the incarnation jump (PutBlob)");
    CLIO_CO_AWAIT(WriteBlockBlob(dir, k, EncodeTableBlock(slot->blk_, true),
                                 wrc));
    if (wrc != 0) {
      slot->persist_dirty_ = true;  // the next commit retries
      slot->journal_image_ = true;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FetchBlock(clio::run::u64 dir,
                                          clio::run::u32 k,
                                          clio::run::u32 home,
                                          std::shared_ptr<TableBlockSlot> slot,
                                          int &lrc) {
  CLIO_TASK_BODY_BEGIN
  FsReq fr;
  fr.dir_id_ = dir;
  fr.block_ = k;
  fr.a_ = container_id_;
  FsResp resp;
  const clio::run::u64 asked_ms = SteadyMs();
  CLIO_CO_AWAIT(CallShard(home, kTableFetch, fr, resp));
  lrc = static_cast<int>(resp.rc_);
  if (lrc == 0 && !DecodeTableBlock(resp.str_.data(), resp.str_.size(),
                                  &slot->blk_)) {
    lrc = EIO;
  }
  slot->home_id_ = home;
  // The lease runs from when we asked: the home granted it later.
  slot->lease_until_ms_ = asked_ms + kCacheLeaseMs;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RevalidateBlock(
    std::shared_ptr<TableBlockSlot> slot, clio::run::u64 dir, clio::run::u32 k,
    clio::run::u32 home, int &rc) {
  CLIO_TASK_BODY_BEGIN
  clio::run::u64 have = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    have = slot->blk_.version_;
  }
  FsReq fr;
  fr.dir_id_ = dir;
  fr.block_ = k;
  fr.a_ = container_id_;
  fr.b_ = have;  // "unchanged" if the home is still at this version
  FsResp resp;
  const clio::run::u64 asked_ms = SteadyMs();
  CLIO_CO_AWAIT(CallShard(home, kTableFetch, fr, resp));
  rc = static_cast<int>(resp.rc_);
  if (rc != 0) CLIO_CO_RETURN;
  InodeTableBlock nb;
  const bool changed = resp.id_ != kFetchUnchanged;
  if (changed && !DecodeTableBlock(resp.str_.data(), resp.str_.size(), &nb)) {
    rc = EIO;
    CLIO_CO_RETURN;
  }
  std::lock_guard<std::mutex> g(ns_mu_);
  // A push may have landed a newer version while we asked: keep it.
  if (changed && nb.version_ > slot->blk_.version_) {
    slot->blk_ = std::move(nb);
  }
  slot->lease_until_ms_ = asked_ms + kCacheLeaseMs;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ServeBlockFetch(const FsReq &req,
                                               FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (BlockHome(req.dir_id_, req.block_) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  std::shared_ptr<TableBlockSlot> slot;
  int rc = 0;
  CLIO_CO_AWAIT(LoadBlock(req.dir_id_, req.block_, slot, rc));
  if (rc != 0) {
    resp.rc_ = static_cast<clio::run::u32>(rc);
    CLIO_CO_RETURN;
  }
  std::lock_guard<std::mutex> g(ns_mu_);
  const clio::run::u32 holder = static_cast<clio::run::u32>(req.a_);
  if (holder != container_id_) {
    slot->holders_[holder] = reg_seq_++;
    slot->holder_lease_ms_[holder] =
        SteadyMs() + kCacheLeaseMs + kCacheLeaseSlackMs;
  }
  resp.rc_ = 0;
  if (req.b_ != 0 && req.b_ == slot->blk_.version_) {
    resp.id_ = kFetchUnchanged;  // a lease renewal: the copy is current
    CLIO_CO_RETURN;
  }
  // The in-memory state, uncommitted changes included: every later change
  // reaches the holder as a push with a higher version.
  resp.str_ = EncodeTableBlock(slot->blk_, false);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

int Runtime::ApplyTablePush(const FsReq &req) {
  std::vector<TableDelta> deltas;
  if (!DecodeTableDeltas(req.str_, &deltas)) return EINVAL;
  bool missing = false;
  std::lock_guard<std::mutex> g(ns_mu_);
  for (const TableDelta &x : deltas) {
    const TableBlockKey key{x.dir_, x.index_};
    if (loading_.count(key) != 0) {
      early_[key].push_back(x);  // applied once the fetched copy lands
      continue;
    }
    auto it = blocks_.find(key);
    if (it == blocks_.end() || it->second->home_) {
      if (x.kind_ != TableDeltaKind::kDrop) missing = true;
      continue;
    }
    InodeTableBlock &b = it->second->blk_;
    if (x.kind_ == TableDeltaKind::kDrop) {
      blocks_.erase(it);
    } else if (x.kind_ == TableDeltaKind::kSnapshot) {
      InodeTableBlock nb;
      if (DecodeTableBlock(x.snapshot_.data(), x.snapshot_.size(), &nb)) {
        b = std::move(nb);
      } else {
        blocks_.erase(it);
      }
    } else if (x.new_version_ > b.version_ && !ApplyTableOps(&b, x)) {
      blocks_.erase(it);  // missed a change: refetch on next use
    }
  }
  return missing ? ENOENT : 0;
}

// ===========================================================================
// Commit: write + push
// ===========================================================================

namespace {
/**
 * A delta carrying the whole block (holders replace their copy with it).
 * @param b the block, as the home holds it
 * @return the snapshot delta
 */
TableDelta SnapshotDelta(const InodeTableBlock &b) {
  TableDelta x;
  x.kind_ = TableDeltaKind::kSnapshot;
  x.dir_ = b.dir_;
  x.index_ = b.index_;
  x.base_version_ = b.version_;
  x.new_version_ = b.version_;
  x.depth_ = b.depth_;
  x.mtime_ = b.mtime_;
  x.sealed_ = b.sealed_;
  x.snapshot_ = EncodeTableBlock(b, false);
  return x;
}
}  // namespace

clio::run::u64 Runtime::RecordChangeLocked(TableBlockSlot &slot, TableDelta delta,
                                           bool persisted) {
  InodeTableBlock &b = slot.blk_;
  delta.kind_ = TableDeltaKind::kOps;
  delta.dir_ = b.dir_;
  delta.index_ = b.index_;
  delta.base_version_ = b.version_;
  delta.new_version_ = ++b.version_;
  delta.depth_ = b.depth_;
  delta.mtime_ = b.mtime_;
  delta.sealed_ = b.sealed_;
  slot.unpushed_.push_back(std::move(delta));
  if (persisted) slot.persist_dirty_ = true;
  return b.version_;
}

clio::run::TaskResume Runtime::PushToHolders(
    clio::run::u32 op, const FsReq &req,
    const std::vector<clio::run::u32> &holders,
    const std::vector<clio::run::u64> &leases,
    std::vector<clio::run::u32> *gone) {
  CLIO_TASK_BODY_BEGIN
  // (holder, lease end) still owed this push
  std::vector<std::pair<clio::run::u32, clio::run::u64>> todo;
  for (size_t i = 0; i < holders.size(); ++i) {
    todo.emplace_back(holders[i], i < leases.size() ? leases[i] : 0);
  }
  const auto start = std::chrono::steady_clock::now();
  bool warned = false;
  while (!todo.empty()) {
    const clio::run::u64 now = SteadyMs();
    std::vector<std::pair<clio::run::u32, clio::run::u64>> wait;
    std::vector<std::pair<size_t, clio::run::Future<ShardOpTask>>> fs;
    for (size_t i = 0; i < todo.size(); ++i) {
      if (now >= todo[i].second) {
        // Its lease ran out: it asks the home before using its copy again,
        // so the change reaches it whenever it comes back.
        gone->push_back(todo[i].first);
      } else if (!clio::cte::core::ContainerNodeAlive(pool_id_,
                                                      todo[i].first)) {
        wait.push_back(todo[i]);  // may only be cut off: wait out the lease
      } else {
        fs.emplace_back(i, SendShard(todo[i].first, op, req));
      }
    }
    for (auto &xf : fs) {
      const auto &h = todo[xf.first];
      // Wait for the push only while the holder's lease lasts. A peer whose
      // own filesystem pool is not composed yet (it restarted too, so it
      // caches nothing) holds the push in its deferred queue for up to
      // 120 s before dropping it; awaiting that blocked every commit -- and
      // so the restarted home's compose -- for ~128 s on a cluster-wide
      // crash restart (stress_crash_under_pressure). Past the lease the
      // holder must ask the home before using its copy, so it is gone.
      while (!xf.second.IsComplete() && SteadyMs() < h.second) {
        CLIO_CO_AWAIT(clio::run::yield(20000.0));
      }
      if (!xf.second.IsComplete()) {
        gone->push_back(h.first);
        ParkPush(std::move(xf.second));
        continue;
      }
      CLIO_CO_AWAIT(xf.second);
      FsResp resp;
      if (!ReadShardResp(xf.second, &resp)) {
        wait.push_back(h);  // unreachable: retry until it gets it or expires
      } else if (resp.rc_ == ENOENT) {
        gone->push_back(h.first);  // it no longer caches the block
      }
    }
    todo.swap(wait);
    if (todo.empty()) break;
    const double waited = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (!warned && waited > kSlowBlockOpS) {
      warned = true;
      HLOG(kWarning, "filesystem: {} cache holder(s) unreachable "
           "for {} ms; waiting for their leases to run out", todo.size(),
           static_cast<clio::run::u64>(waited * 1000.0));
    }
    CLIO_CO_AWAIT(clio::run::yield(100000.0));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::ParkPush(clio::run::Future<ShardOpTask> &&f) {
  // Keep an abandoned in-flight push alive until it settles, then drop it;
  // never destroy a future whose task is still out on the network.
  std::lock_guard<std::mutex> g(parked_mu_);
  auto &v = parked_pushes_;
  v.erase(std::remove_if(v.begin(), v.end(),
                         [](const clio::run::Future<ShardOpTask> &p) {
                           return p.IsComplete();
                         }),
          v.end());
  v.push_back(std::move(f));
}

void Runtime::PushTargetsLocked(const TableBlockSlot &slot, bool resync,
                                std::vector<clio::run::u32> *holders,
                                std::vector<clio::run::u64> *leases) {
  if (!resync) {
    for (const auto &hv : slot.holders_) {
      holders->push_back(hv.first);
      auto lt = slot.holder_lease_ms_.find(hv.first);
      leases->push_back(lt == slot.holder_lease_ms_.end() ? 0 : lt->second);
    }
    return;
  }
  // Whoever cached it from an earlier incarnation of this home holds a
  // lease that ends by this horizon at the latest.
  const clio::run::u64 horizon =
      started_ms_ + kCacheLeaseMs + kCacheLeaseSlackMs;
  for (clio::run::u32 c = 0; c < NumContainers(); ++c) {
    if (c == container_id_) continue;
    holders->push_back(c);
    auto lt = slot.holder_lease_ms_.find(c);
    leases->push_back(lt == slot.holder_lease_ms_.end()
                          ? horizon
                          : std::max(horizon, lt->second));
  }
}

clio::run::TaskResume Runtime::CommitBlock(std::shared_ptr<TableBlockSlot> slot,
                                           clio::run::u64 version, int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  for (;;) {
    std::vector<TableDelta> deltas;
    std::string image, jrec;
    FsLogRec jtype = FsLogRec::kBlockDeltas;
    std::string mirror;  // full image for the CTE copy of a journaled delta
    std::vector<clio::run::u32> holders;
    std::vector<clio::run::u64> leases;  // parallel to holders
    std::map<clio::run::u32, clio::run::u64> reg;  // registrations pushed to
    clio::run::u64 v = 0, dir = 0;
    clio::run::u32 k = 0;
    bool go = false, resync = false;
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      if (slot->durable_version_ >= version) break;
      if (!slot->committing_) {
        slot->committing_ = go = true;
        v = slot->blk_.version_;
        dir = slot->blk_.dir_;
        k = slot->blk_.index_;
        if (TraceDirBlock()) {
          HLOG(kWarning, "[DIRBLOCK] commit {}/{} on container {}: version "
               "{} rows {} deltas {} home {}", dir, k, container_id_, v,
               slot->blk_.rows_.size(), slot->unpushed_.size(),
               slot->home_);
        }
        deltas.swap(slot->unpushed_);
        TakePersistLocked(*slot, deltas, &image, &jtype, &jrec);
        // A journaled delta lives only in this node's journal: mirror the
        // block's whole current image to its replicated CTE blob too, so a
        // stand-in can serve it while this node is down (#1263).
        if (dir_journal_ && image.empty() && !jrec.empty()) {
          mirror = EncodeTableBlock(slot->blk_, true);
        }
        reg = slot->holders_;
        if (slot->holders_unknown_) {
          resync = true;
          slot->holders_unknown_ = false;
          deltas.assign(1, SnapshotDelta(slot->blk_));
        }
        PushTargetsLocked(*slot, resync, &holders, &leases);
      }
    }
    if (!go) {
      CLIO_CO_AWAIT(clio::run::yield(kCommitPollUs));
      continue;
    }
    int wrc = 0;
    const auto commit_t0 = std::chrono::steady_clock::now();
    if (!image.empty()) CLIO_CO_AWAIT(WriteBlockBlob(dir, k, image, wrc));
    if (wrc == 0 && !jrec.empty()) wrc = JournalAppend(jtype, jrec);
    bool mirror_failed = false;
    if (wrc == 0 && !mirror.empty()) {
      int mrc = 0;
      CLIO_CO_AWAIT(PutBlockCte(dir, k, mirror, mrc));
      if (mrc == ENOSPC) {
        // The change is durable in this node's journal; only the replicated
        // mirror a stand-in would read is behind. Failing the operation for
        // that made unlink/rmdir on a full store fail ENOSPC -- the one way
        // to free space. Succeed, and rewrite the mirror at the next commit
        // (persist_dirty_). The cost: a stand-in serving this block before
        // that sees it without this change.
        mirror_failed = true;
        HLOG(kWarning, "filesystem: inode-table block {}/{} change is "
             "journaled but its replicated mirror could not be written "
             "(store full); retried at the next commit", dir, k);
      } else {
        wrc = mrc;
      }
    }
    std::vector<clio::run::u32> gone;
    if (!deltas.empty() && !holders.empty()) {
      FsReq pr;
      pr.str_ = EncodeTableDeltas(deltas);
      CLIO_CO_AWAIT(PushToHolders(kTablePush, pr, holders, leases, &gone));
    }
    {
      const double took = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - commit_t0).count();
      if (took > kSlowBlockOpS) {
        HLOG(kWarning, "filesystem: committing inode-table block {}/{} took "
             "{} ms ({} holders{})", dir, k,
             static_cast<clio::run::u64>(took * 1000.0), holders.size(),
             resync ? ", resync" : "");
      }
    }
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      if (resync) {
        // Everyone who took the snapshot caches the block: register them,
        // with the lease the resync assumed (a copy from before this home's
        // restart may be trusted until that horizon). Without one they
        // counted as expired: the next change skipped them while they still
        // served their copy without asking.
        for (size_t i = 0; i < holders.size(); ++i) {
          const clio::run::u32 h = holders[i];
          if (std::find(gone.begin(), gone.end(), h) != gone.end()) continue;
          if (slot->holders_.count(h) == 0) slot->holders_[h] = reg_seq_++;
          clio::run::u64 &lease = slot->holder_lease_ms_[h];
          lease = std::max(lease, leases[i]);
        }
      }
      DropGoneHolders(gone, reg, &slot->holders_, &slot->holder_lease_ms_);
      if (wrc != 0) {
        slot->persist_dirty_ = true;  // the next commit retries
        slot->journal_image_ = true;  // these deltas never reached it
      } else if (mirror_failed) {
        slot->persist_dirty_ = true;  // rewrite the mirror next time
      }
      slot->durable_version_ = v;
      slot->committing_ = false;
    }
    if (wrc != 0) {
      rc = wrc;  // ENOSPC or EIO (WriteBlockBlob)
      break;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Row mutations (on the block's home)
// ===========================================================================

int Runtime::InsertRowLocked(TableBlockSlot &slot, const FsReq &req,
                             FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  InodeTableBlock &b = slot.blk_;
  if (b.sealed_) return ENOENT;  // an rmdir is removing the directory
  if (req.id_ == 0) return EINVAL;  // the caller names the inode
  const clio::cte::core::TagId dir_tag = FsUnpack(b.dir_);
  const std::string key = KeyOf(req.leaf_);
  const std::string tn = clio::cte::core::MakeTagRefName(dir_tag, req.leaf_);
  auto it = b.rows_.find(key);
  TableDelta dl;
  if ((f & kInsCommit) != 0) {
    if (it == b.rows_.end()) {
      // The reservation is gone: pending rows are never durable, so a home
      // that restarted or failed over (and back) lost it. The name is still
      // free, so take it now -- the check is as atomic here as the
      // reservation was.
      InodeRow e;
      e.id_ = req.id_;
      e.type_ = req.type_;
      e.spelling_ = RowSpelling(key, req.leaf_);
      e.state_ = kRowPending;
      it = b.rows_.emplace(key, e).first;
    } else if (it->second.id_ != req.id_) {
      return EEXIST;  // someone else took the name after we lost it
    } else if (it->second.state_ == kRowLive) {
      return 0;  // already committed (a retried commit)
    } else if (it->second.state_ != kRowPending) {
      return ENOENT;
    }
    it->second.state_ = kRowLive;
    b.mtime_ = NowNs();
    dl.ops_.push_back({key, true, it->second});
    RecordChangeLocked(slot, dl, true);
    PublishName(clio::cte::core::TagNameOp::kAddName, FsUnpack(req.id_),
                clio::cte::core::MakeTagRefName(dir_tag,
                                                RowName(key, it->second)));
    return 0;
  }
  if (it != b.rows_.end()) {
    InodeRow &e = it->second;
    if (e.state_ != kRowLive) return (f & kInsFailBusy) ? EBUSY : kFsRetry;
    resp.id_ = e.id_;
    resp.type_ = e.type_;
    if ((f & kInsExcl) != 0) return EEXIST;
    if ((f & (kInsReplace | kInsReplaceDir)) == 0) return 0;  // open existing
    if (e.id_ == req.id_) {  // rename onto another link of itself
      resp.old_id_ = req.id_;
      return 0;
    }
    if (e.type_ == kFsTypeDir && req.type_ != kFsTypeDir) return EISDIR;
    if (e.type_ != kFsTypeDir && req.type_ == kFsTypeDir) return ENOTDIR;
    if (e.type_ == kFsTypeDir && (f & kInsReplaceDir) == 0) return ENOTEMPTY;
    resp.old_id_ = e.id_;
    resp.old_type_ = e.type_;
    resp.str_ = RowName(key, e);  // the replaced name, as it was spelled
    const clio::run::u64 victim = e.id_;
    e.id_ = req.id_;
    e.type_ = req.type_;
    e.spelling_ = RowSpelling(key, req.leaf_);
    b.mtime_ = NowNs();
    dl.ops_.push_back({key, true, e});
    RecordChangeLocked(slot, dl, true);
    if ((f & kInsNoTagName) == 0) {
      PublishName(clio::cte::core::TagNameOp::kRemoveName, FsUnpack(victim),
                  clio::cte::core::MakeTagRefName(dir_tag, resp.str_));
      PublishName(clio::cte::core::TagNameOp::kAddName, FsUnpack(e.id_), tn);
    }
    resp.id_ = req.id_;
    resp.type_ = req.type_;
    resp.attr_.id_ = b.dir_;
    return 0;
  }
  InodeRow e;
  e.id_ = req.id_;
  e.type_ = req.type_;
  e.spelling_ = RowSpelling(key, req.leaf_);
  e.state_ = (f & kInsPending) ? kRowPending : kRowLive;
  b.rows_[key] = e;
  const bool live = e.state_ == kRowLive;
  if (live) b.mtime_ = NowNs();
  dl.ops_.push_back({key, true, e});
  RecordChangeLocked(slot, dl, live);
  if (live && (f & kInsNoTagName) == 0) {
    PublishName(clio::cte::core::TagNameOp::kAddName, FsUnpack(e.id_), tn);
  }
  resp.id_ = e.id_;
  resp.type_ = e.type_;
  resp.created_ = 1;
  resp.attr_.id_ = b.dir_;
  return 0;
}

int Runtime::RemoveRowLocked(TableBlockSlot &slot, const FsReq &req,
                             FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  InodeTableBlock &b = slot.blk_;
  const std::string key = KeyOf(req.leaf_);
  auto it = b.rows_.find(key);
  if (it == b.rows_.end()) return ENOENT;
  InodeRow &e = it->second;
  const bool mine = req.id_ != 0 && e.id_ == req.id_;
  TableDelta dl;
  if ((f & kRmRestore) != 0) {
    if (mine && e.state_ == kRowLeaving) {
      e.state_ = kRowLive;
      dl.ops_.push_back({key, true, e});
      RecordChangeLocked(slot, dl, false);
    }
    return 0;
  }
  if (req.id_ != 0 && !mine) return ENOENT;  // replaced underneath us
  if (e.state_ == kRowPending && mine) {  // abort a reservation
    b.rows_.erase(it);
    dl.ops_.push_back({key, false, InodeRow()});
    RecordChangeLocked(slot, dl, false);
    return 0;
  }
  // A leaving row may only be finished by the operation that marked it
  // (which names the id); everyone else waits for the outcome.
  if (e.state_ != kRowLive && !(e.state_ == kRowLeaving && mine &&
                                (f & kRmMarkLeaving) == 0)) {
    return (f & kRmFailBusy) ? EBUSY : kFsRetry;
  }
  if ((f & kRmNonDir) != 0 && e.type_ == kFsTypeDir) return EISDIR;
  if ((f & kRmDirOnly) != 0 && e.type_ != kFsTypeDir) return ENOTDIR;
  resp.old_id_ = e.id_;
  resp.old_type_ = e.type_;
  resp.attr_.id_ = b.dir_;
  resp.str_ = RowName(key, e);  // the removed name, as it was spelled
  if ((f & kRmMarkLeaving) != 0) {
    e.state_ = kRowLeaving;
    dl.ops_.push_back({key, true, e});
    RecordChangeLocked(slot, dl, false);
    return 0;
  }
  const clio::run::u64 gone = e.id_;
  b.rows_.erase(it);
  b.mtime_ = NowNs();
  dl.ops_.push_back({key, false, InodeRow()});
  RecordChangeLocked(slot, dl, true);
  if ((f & kRmNoTagName) == 0) {
    PublishName(clio::cte::core::TagNameOp::kRemoveName, FsUnpack(gone),
                clio::cte::core::MakeTagRefName(FsUnpack(b.dir_), resp.str_));
  }
  return 0;
}

int Runtime::RespellRowLocked(TableBlockSlot &slot, const FsReq &req,
                              FsResp &resp) {
  InodeTableBlock &b = slot.blk_;
  const std::string key = KeyOf(req.leaf_);
  auto it = b.rows_.find(key);
  if (it == b.rows_.end()) return ENOENT;
  InodeRow &e = it->second;
  if (req.id_ != 0 && e.id_ != req.id_) return ENOENT;  // replaced meanwhile
  if (e.state_ != kRowLive) {
    return (req.flags_ & kInsFailBusy) ? EBUSY : kFsRetry;
  }
  resp.id_ = e.id_;
  resp.type_ = e.type_;
  resp.attr_.id_ = b.dir_;
  resp.str_ = RowName(key, e);  // the spelling it had
  const std::string spelling = RowSpelling(key, req.leaf_);
  if (spelling == e.spelling_) return 0;  // already spelled so
  e.spelling_ = spelling;
  b.mtime_ = NowNs();
  TableDelta dl;
  dl.ops_.push_back({key, true, e});
  RecordChangeLocked(slot, dl, true);
  const clio::cte::core::TagId dir_tag = FsUnpack(b.dir_);
  PublishName(clio::cte::core::TagNameOp::kRename, FsUnpack(e.id_),
              clio::cte::core::MakeTagRefName(dir_tag, resp.str_),
              clio::cte::core::MakeTagRefName(dir_tag, req.leaf_));
  return 0;
}

clio::run::TaskResume Runtime::RowMutation(clio::run::u32 op,
                                           const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (BlockHome(req.dir_id_, req.block_) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  std::shared_ptr<TableBlockSlot> slot;
  int rc = 0;
  CLIO_CO_AWAIT(LoadBlock(req.dir_id_, req.block_, slot, rc));
  if (rc != 0) {
    resp.rc_ = static_cast<clio::run::u32>(rc);
    CLIO_CO_RETURN;
  }
  const clio::run::u64 h = TableHash(KeyOf(req.leaf_));
  clio::run::u64 before = 0, after = 0;
  for (;;) {
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      before = slot->blk_.version_;
      if (!BlockOwnsKey(slot->blk_, h)) {
        rc = kFsRedirect;  // it split: the caller re-walks
      } else if (slot->splitting_) {
        rc = kFsRetry;
      } else if (op == kTableInsert) {
        rc = InsertRowLocked(*slot, req, resp);
      } else if (op == kTableRemove) {
        rc = RemoveRowLocked(*slot, req, resp);
      } else {
        rc = RespellRowLocked(*slot, req, resp);
      }
      after = slot->blk_.version_;
    }
    if (rc != kFsRetry) break;
    CLIO_CO_AWAIT(clio::run::yield(50));
  }
  if (after != before) {
    int crc = 0;
    CLIO_CO_AWAIT(CommitBlock(slot, after, crc));
    if (crc != 0 && rc == 0) rc = crc;
  }
  resp.rc_ = static_cast<clio::run::u32>(rc);
  if (rc == 0 && op == kTableInsert) CLIO_CO_AWAIT(SplitBlock(slot));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Splits
// ===========================================================================

clio::run::TaskResume Runtime::SplitBlock(std::shared_ptr<TableBlockSlot> slot) {
  CLIO_TASK_BODY_BEGIN
  InodeTableBlock child;
  std::vector<std::string> moved;
  clio::run::u32 d = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    InodeTableBlock &b = slot->blk_;
    if (!slot->home_ || slot->splitting_ || b.sealed_ ||
        b.rows_.size() <= split_entries_ || b.depth_ >= kTableMaxDepth) {
      CLIO_CO_RETURN;
    }
    d = b.depth_;
    for (const auto &kv : b.rows_) {
      if (((TableHash(kv.first) >> d) & 1ULL) == 0) continue;
      // An operation in flight on this name finishes here first.
      if (kv.second.state_ != kRowLive) CLIO_CO_RETURN;
      moved.push_back(kv.first);
    }
    slot->splitting_ = true;
    child.dir_ = b.dir_;
    child.index_ = b.index_ + (1u << d);
    child.born_ = child.depth_ = d + 1;
    child.version_ = 1;
    child.mtime_ = b.mtime_;
    for (const std::string &leaf : moved) child.rows_[leaf] = b.rows_[leaf];
  }
  // The new block is written by its own home before the old one lets go:
  // until the old block's depth changes nobody routes to the new one, so a
  // crash in between leaves an unreachable orphan, never a lost row.
  FsReq ir;
  ir.dir_id_ = child.dir_;
  ir.block_ = child.index_;
  ir.str_ = EncodeTableBlock(child, false);
  FsResp irr;
  CLIO_CO_AWAIT(CallShard(BlockHome(child.dir_, child.index_),
                          kTableInstall, ir, irr));
  clio::run::u64 v = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    if (irr.rc_ == 0) {
      InodeTableBlock &b = slot->blk_;
      TableDelta dl;
      for (const std::string &leaf : moved) {
        b.rows_.erase(leaf);
        dl.ops_.push_back({leaf, false, InodeRow()});
      }
      b.depth_ = d + 1;
      v = RecordChangeLocked(*slot, dl, true);
    }
    slot->splitting_ = false;
  }
  if (v != 0) {
    int crc = 0;
    CLIO_CO_AWAIT(CommitBlock(slot, v, crc));
  } else {
    HLOG(kWarning, "filesystem: splitting inode-table block {}/{} failed "
         "(rc {}); it stays whole", child.dir_, child.index_ - (1u << d),
         irr.rc_);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::InstallBlock(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  auto slot = std::make_shared<TableBlockSlot>();
  if (!DecodeTableBlock(req.str_.data(), req.str_.size(), &slot->blk_)) {
    resp.rc_ = EINVAL;
    CLIO_CO_RETURN;
  }
  if (BlockHome(slot->blk_.dir_, slot->blk_.index_) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  slot->home_ = true;
  slot->persist_dirty_ = true;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    // A leftover from an earlier failed split of the same block is replaced.
    blocks_[TableBlockKey{slot->blk_.dir_, slot->blk_.index_}] = slot;
  }
  int rc = 0;
  CLIO_CO_AWAIT(CommitBlock(slot, slot->blk_.version_, rc));
  resp.rc_ = static_cast<clio::run::u32>(rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Directory lifecycle: mkdir, rmdir, attributes
// ===========================================================================

clio::run::TaskResume Runtime::CreateTable(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (BlockHome(req.dir_id_, 0) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  if ((req.flags_ & kCreateRepair) != 0) {
    // Only when truly missing (a crash between rmdir's halves left the
    // parent's row but not the table).
    std::shared_ptr<TableBlockSlot> have;
    int lrc = 0;
    CLIO_CO_AWAIT(LoadBlock(req.dir_id_, 0, have, lrc));
    if (lrc != ENOENT) {
      resp.rc_ = static_cast<clio::run::u32>(lrc);
      CLIO_CO_RETURN;
    }
  }
  auto slot = std::make_shared<TableBlockSlot>();
  InodeTableBlock &b = slot->blk_;
  b.dir_ = req.dir_id_;
  b.version_ = 1;
  b.mtime_ = NowNs();
  slot->home_ = true;
  slot->persist_dirty_ = true;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    blocks_[TableBlockKey{b.dir_, 0}] = slot;
  }
  int rc = 0;
  CLIO_CO_AWAIT(CommitBlock(slot, b.version_, rc));
  resp.rc_ = static_cast<clio::run::u32>(rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SealBlock(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (BlockHome(req.dir_id_, req.block_) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  std::shared_ptr<TableBlockSlot> slot;
  int rc = 0;
  CLIO_CO_AWAIT(LoadBlock(req.dir_id_, req.block_, slot, rc));
  if (rc != 0) {
    resp.rc_ = static_cast<clio::run::u32>(rc);
    CLIO_CO_RETURN;
  }
  clio::run::u64 v = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    InodeTableBlock &b = slot->blk_;
    resp.str_ = EncIndices(ChildBlocks(b));
    const bool seal = req.a_ != 0;
    if (seal && (!b.rows_.empty() || slot->splitting_)) {
      rc = ENOTEMPTY;  // pending rows (a mkdir inside) count too
      // #1029: rows a listing does NOT show (pending reservations,
      // leaving halves of a rename) or a split in flight make an rmdir fail
      // after rm -r removed everything it was shown. Name them.
      size_t hidden = 0;
      std::string names;
      for (const auto &kv : b.rows_) {
        if (kv.second.state_ == kRowLive &&
            kv.first.rfind(".fuse_hidden", 0) != 0) {
          continue;
        }
        ++hidden;
        if (names.size() < 512) {
          names += kv.first + "(state " +
                   std::to_string(static_cast<int>(kv.second.state_)) + ") ";
        }
      }
      if (hidden != 0 || slot->splitting_) {
        HLOG(kError, "filesystem: rmdir of dir {} block {} refused for "
             "{} unlisted entr(ies), splitting={}: {}", req.dir_id_,
             req.block_, hidden, slot->splitting_, names);
      }
    } else if (b.sealed_ != seal) {
      b.sealed_ = seal;
      v = RecordChangeLocked(*slot, TableDelta(), false);
    }
  }
  if (v != 0) {
    int crc = 0;
    CLIO_CO_AWAIT(CommitBlock(slot, v, crc));
  }
  resp.rc_ = static_cast<clio::run::u32>(rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DropBlock(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  std::vector<clio::run::u32> holders;
  std::vector<clio::run::u64> leases;
  std::shared_ptr<TableBlockSlot> sealed;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    auto it = blocks_.find(TableBlockKey{req.dir_id_, req.block_});
    if (it != blocks_.end() && it->second->home_) {
      if (!it->second->blk_.sealed_) {
        resp.rc_ = EBUSY;
        CLIO_CO_RETURN;
      }
      PushTargetsLocked(*it->second, false, &holders, &leases);
      sealed = it->second;
    }
  }
  // The sealed block stays cached until its blob is gone: a racing create
  // finds it sealed (ENOENT), and no load can bring back the blob's
  // unsealed image in the meantime.
  if (dir_journal_) {
    JournalDropBlock(req.dir_id_, req.block_);
  } else {
    auto d = cte_.AsyncDelBlob(FsUnpack(req.dir_id_),
                               TableBlockName(req.block_),
                               clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(d);
  }
  if (sealed != nullptr) {
    std::lock_guard<std::mutex> g(ns_mu_);
    auto it = blocks_.find(TableBlockKey{req.dir_id_, req.block_});
    if (it != blocks_.end() && it->second == sealed) blocks_.erase(it);
  }
  if (!holders.empty()) {
    TableDelta x;
    x.kind_ = TableDeltaKind::kDrop;
    x.dir_ = req.dir_id_;
    x.index_ = req.block_;
    std::vector<TableDelta> one;
    one.push_back(x);
    FsReq pr;
    pr.str_ = EncodeTableDeltas(one);
    std::vector<clio::run::u32> gone;
    CLIO_CO_AWAIT(PushToHolders(kTablePush, pr, holders, leases, &gone));
  }
  resp.rc_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SealTable(clio::run::u64 dir, bool seal,
                                         int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  std::vector<clio::run::u32> todo{0}, sealed;
  while (!todo.empty()) {
    const clio::run::u32 k = todo.back();
    todo.pop_back();
    FsReq r;
    r.dir_id_ = dir;
    r.block_ = k;
    r.a_ = seal ? 1 : 0;
    FsResp resp;
    CLIO_CO_AWAIT(CallShard(BlockHome(dir, k), kTableSeal, r, resp));
    if (resp.rc_ != 0 && seal) {
      rc = resp.rc_ == kFsRedirect ? EBUSY : static_cast<int>(resp.rc_);
      break;
    }
    sealed.push_back(k);
    for (clio::run::u32 c : DecIndices(resp.str_)) todo.push_back(c);
  }
  if (rc != 0) {
    // Not empty (or unreachable): undo the seals that took.
    for (clio::run::u32 k : sealed) {
      FsReq r;
      r.dir_id_ = dir;
      r.block_ = k;
      r.a_ = 0;
      FsResp resp;
      CLIO_CO_AWAIT(CallShard(BlockHome(dir, k), kTableSeal, r, resp));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DropTable(clio::run::u64 dir) {
  CLIO_TASK_BODY_BEGIN
  std::vector<clio::run::u32> blocks;
  int rc = 0;
  CLIO_CO_AWAIT(TableBlocks(dir, &blocks, rc));
  for (clio::run::u32 k : blocks) {
    FsReq r;
    r.dir_id_ = dir;
    r.block_ = k;
    FsResp resp;
    CLIO_CO_AWAIT(CallShard(BlockHome(dir, k), kTableDrop, r, resp));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Reads through the cache
// ===========================================================================

clio::run::TaskResume Runtime::TableBlocks(clio::run::u64 dir,
                                           std::vector<clio::run::u32> *out,
                                           int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  out->clear();
  std::vector<clio::run::u32> todo{0};
  std::unordered_set<clio::run::u32> seen;
  std::string shape;  // index:born-depth of every block, for the error below
  while (!todo.empty()) {
    const clio::run::u32 k = todo.back();
    todo.pop_back();
    if (!seen.insert(k).second) {
      // Each block has exactly one parent in the split tree; reaching one
      // twice means two blocks disagree about who owns which names.
      HLOG(kError, "filesystem: directory {} has a corrupt split tree "
           "(block {} reached twice): {}", dir, k, shape);
      rc = EIO;
      CLIO_CO_RETURN;
    }
    std::shared_ptr<TableBlockSlot> slot;
    CLIO_CO_AWAIT(LoadBlock(dir, k, slot, rc));
    if (rc != 0) CLIO_CO_RETURN;
    out->push_back(k);
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      const InodeTableBlock &b = slot->blk_;
      shape += std::to_string(b.index_) + ":" + std::to_string(b.born_) + "-" +
               std::to_string(b.depth_) + " ";
      for (clio::run::u32 c : ChildBlocks(b)) todo.push_back(c);
    }
    // Cached blocks load without suspending: let a large listing share the
    // worker.
    if (out->size() % 64 == 0) CLIO_CO_AWAIT(clio::run::yield(1.0));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CollectDir(
    clio::run::u64 dir, std::vector<std::pair<std::string, InodeRow>> *out,
    clio::run::u64 *newest, int &rc) {
  CLIO_TASK_BODY_BEGIN
  // A listing is one snapshot of this node's copies of every block, taken
  // under one lock. Every change reaches the copies before its operation is
  // acknowledged, so at any instant they hold a prefix of each rename's
  // steps (mark leaving -> insert new -> remove old): a file then shows under
  // its old name, its new name, or both -- never neither -- and SnapshotDir
  // drops the leaving duplicate. Reading block by block instead let a
  // concurrent rename vanish from, or double in, a listing.
  for (int attempt = 0;; ++attempt) {
    out->clear();
    *newest = 0;
    std::vector<clio::run::u32> blocks;
    CLIO_CO_AWAIT(TableBlocks(dir, &blocks, rc));
    if (rc != 0) CLIO_CO_RETURN;
    std::vector<std::shared_ptr<TableBlockSlot>> slots;
    for (clio::run::u32 k : blocks) {
      std::shared_ptr<TableBlockSlot> slot;
      CLIO_CO_AWAIT(LoadBlock(dir, k, slot, rc));
      if (rc != 0) CLIO_CO_RETURN;
      slots.push_back(std::move(slot));
    }
    if (SnapshotDir(dir, blocks, slots, out, newest)) break;
    if (attempt >= 8) {
      // Not "empty": an empty answer here made rm -r skip every child.
      HLOG(kError, "filesystem: listing directory {} kept racing block "
           "refetches; failing it EAGAIN", dir);
      out->clear();
      rc = EAGAIN;
      break;
    }
    // A copy was dropped for a refetch while we loaded the others: again.
    CLIO_CO_AWAIT(clio::run::yield(kLoadPollUs));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::SnapshotDir(
    clio::run::u64 dir, const std::vector<clio::run::u32> &blocks,
    const std::vector<std::shared_ptr<TableBlockSlot>> &slots,
    std::vector<std::pair<std::string, InodeRow>> *out,
    clio::run::u64 *newest) {
  std::lock_guard<std::mutex> g(ns_mu_);
  // Every copy must still be the cached one: a copy that missed a change is
  // dropped (and refetched) and no longer receives pushes.
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto it = blocks_.find(TableBlockKey{dir, blocks[i]});
    if (it == blocks_.end() || it->second != slots[i]) return false;
  }
  std::map<std::string, InodeRow> names;  // a name split across blocks: once
  std::unordered_set<clio::run::u64> live_ids;
  for (const auto &slot : slots) {
    *newest = std::max(*newest, slot->blk_.mtime_);
    for (const auto &kv : slot->blk_.rows_) {
      if (kv.second.state_ == kRowPending) continue;
      names.emplace(kv.first, kv.second);
      if (kv.second.state_ == kRowLive) live_ids.insert(kv.second.id_);
    }
  }
  for (const auto &kv : names) {
    // A leaving name whose inode is already live under another name is a
    // rename caught mid-way: list the file once, under its new name.
    if (kv.second.state_ == kRowLeaving &&
        live_ids.count(kv.second.id_) != 0) {
      continue;
    }
    out->emplace_back(RowName(kv.first, kv.second), kv.second);
  }
  return true;
}

clio::run::TaskResume Runtime::StatDir(clio::run::u64 dir, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  std::vector<clio::run::u32> blocks;
  int rc = 0;
  CLIO_CO_AWAIT(TableBlocks(dir, &blocks, rc));
  clio::run::u64 newest = 0;
  for (size_t i = 0; rc == 0 && i < blocks.size(); ++i) {
    std::shared_ptr<TableBlockSlot> slot;
    CLIO_CO_AWAIT(LoadBlock(dir, blocks[i], slot, rc));
    if (rc != 0) break;
    std::lock_guard<std::mutex> g(ns_mu_);
    newest = std::max(newest, slot->blk_.mtime_);
  }
  if (rc != 0) {
    resp = FsResp();
    resp.rc_ = static_cast<clio::run::u32>(rc);
    CLIO_CO_RETURN;
  }
  CLIO_CO_AWAIT(StatInode(dir, resp));
  if (resp.rc_ != 0) CLIO_CO_RETURN;
  // Creates and removes change the table, not the record (writing the
  // record on every create would serialize a directory's creates on its
  // inode home): a row change after the last attribute change is the
  // directory's mtime and ctime.
  FsAttr &a = resp.attr_;
  if (newest > a.ctime_) a.mtime_ = a.ctime_ = newest;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::WalkToBlock(clio::run::u64 dir,
                                           const std::string &key,
                                           std::shared_ptr<TableBlockSlot> &slot,
                                           int &rc) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 h = TableHash(key);
  clio::run::u32 k = 0;
  for (clio::run::u32 step = 0; step <= kTableMaxDepth + 1; ++step) {
    CLIO_CO_AWAIT(LoadBlock(dir, k, slot, rc));
    if (rc != 0) CLIO_CO_RETURN;
    clio::run::u32 next = 0;
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      next = NextTableBlock(slot->blk_, h);
    }
    if (next == k) CLIO_CO_RETURN;
    k = next;
  }
  rc = EIO;  // a split tree deeper than possible: corrupt block
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::LookupRow(clio::run::u64 dir,
                                         const std::string &name,
                                         InodeRow &out, int &rc) {
  CLIO_TASK_BODY_BEGIN
  const std::string key = KeyOf(name);
  std::shared_ptr<TableBlockSlot> slot;
  CLIO_CO_AWAIT(WalkToBlock(dir, key, slot, rc));
  if (rc != 0) CLIO_CO_RETURN;
  std::lock_guard<std::mutex> g(ns_mu_);
  auto it = slot->blk_.rows_.find(key);
  if (it == slot->blk_.rows_.end() || it->second.state_ == kRowPending) {
    rc = ENOENT;
  } else {
    out = it->second;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ResolvePath(const std::string &path,
                                           InodeRow &ent,
                                           clio::run::u64 &parent, int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  parent = 0;
  ent = InodeRow();
  ent.id_ = FsPack(FsRootId());
  ent.type_ = kFsTypeDir;
  if (path == "/") CLIO_CO_RETURN;
  if (path.rfind(kFsIdStatPrefix, 0) == 0) {
    // An open file named by its inode (FsIdStatPath): it may have no name
    // left at all. Ops on it act on the inode; there is no parent row.
    ent.id_ = std::strtoull(path.c_str() + sizeof(kFsIdStatPrefix) - 1,
                            nullptr, 10);
    ent.type_ = kFsTypeFile;
    ent.state_ = kRowLive;
    CLIO_CO_RETURN;
  }
  size_t pos = 1;
  while (pos <= path.size()) {
    size_t slash = path.find('/', pos);
    if (slash == std::string::npos) slash = path.size();
    const std::string comp = path.substr(pos, slash - pos);
    pos = slash + 1;
    if (comp.empty()) continue;
    if (ent.type_ != kFsTypeDir) {
      rc = ENOTDIR;
      CLIO_CO_RETURN;
    }
    parent = ent.id_;
    InodeRow next;
    CLIO_CO_AWAIT(LookupRow(parent, comp, next, rc));
    if (rc != 0) CLIO_CO_RETURN;
    ent = next;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::TableOp(clio::run::u32 op, FsReq req,
                                       FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  for (int attempt = 0; attempt < 200; ++attempt) {
    std::shared_ptr<TableBlockSlot> slot;
    int rc = 0;
    CLIO_CO_AWAIT(WalkToBlock(req.dir_id_, KeyOf(req.leaf_), slot, rc));
    if (rc != 0) {
      resp = FsResp();
      resp.rc_ = static_cast<clio::run::u32>(rc);
      CLIO_CO_RETURN;
    }
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      req.block_ = slot->blk_.index_;
    }
    bool lost = false;
    CLIO_CO_AWAIT(CallShard(BlockHome(req.dir_id_, req.block_), op, req, resp,
                            &lost));
    // The block's home died with the call in flight: once it is declared
    // dead BlockHome names its stand-in, which serves the block from its
    // replicated CTE copy (#1263). Walk again rather than fail the lookup
    // in the ~10 s before the death is declared; bounded by this loop.
    if (lost) {
      CLIO_CO_AWAIT(clio::run::yield(std::min(500000.0, 50000.0 * (attempt + 1))));
      continue;
    }
    if (resp.rc_ != static_cast<clio::run::u32>(kFsRedirect)) CLIO_CO_RETURN;
    {
      // Our copy said the name was here and the home says it is not: drop
      // the copy (a push may still be on its way) and walk again.
      std::lock_guard<std::mutex> g(ns_mu_);
      auto it = blocks_.find(TableBlockKey{req.dir_id_, req.block_});
      if (it != blocks_.end() && !it->second->home_ &&
          !it->second->committing_) {
        blocks_.erase(it);
      }
    }
    CLIO_CO_AWAIT(clio::run::yield(std::min(20000.0, 100.0 * (attempt + 1))));
  }
  resp.rc_ = EIO;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CheckMoveTarget(clio::run::u64 moving,
                                               clio::run::u64 dst_parent,
                                               int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  const clio::run::u64 root = FsPack(FsRootId());
  clio::run::u64 cur = dst_parent;
  for (int step = 0; step < 65536; ++step) {
    if (cur == moving) {
      rc = EINVAL;  // into its own subtree
      CLIO_CO_RETURN;
    }
    if (cur == root) CLIO_CO_RETURN;
    // The directory's record points back at the row that names it.
    FsResp sr;
    CLIO_CO_AWAIT(StatInode(cur, sr));
    if (sr.rc_ != 0) {
      rc = static_cast<int>(sr.rc_);
      CLIO_CO_RETURN;
    }
    const clio::run::u64 up_dir = sr.attr_.parent_;
    InodeRow up;
    CLIO_CO_AWAIT(LookupRow(up_dir, sr.attr_.parent_key_, up, rc));
    // An ancestor that is itself being moved (or removed) could close a
    // cycle with this move: back off and let it finish.
    if (rc != 0 || up.id_ != cur || up.state_ == kRowLeaving) {
      rc = EBUSY;
      CLIO_CO_RETURN;
    }
    cur = up_dir;
  }
  rc = EBUSY;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::EncodeHomeNames(std::string *out) {
  CLIO_TASK_BODY_BEGIN
  out->clear();
  std::vector<TableBlockKey> keys;
  if (dir_journal_) {
    JournaledHomeKeys(&keys);
  } else {
    auto list = cte_.AsyncListLocalBlobs("d\\.[0-9]+",
                                         clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(list);
    for (size_t i = 0; i < list->tag_ids_.size(); ++i) {
      keys.push_back(TableBlockKey{
          list->tag_ids_[i],
          static_cast<clio::run::u32>(
              std::strtoul(list->blob_names_[i].c_str() + 2, nullptr, 10))});
    }
  }
  const clio::run::u64 now = NowNs();
  for (const TableBlockKey &key : keys) {
    const clio::run::u64 dir = key.dir_;
    const clio::run::u32 k = key.index_;
    if (BlockHome(dir, k) != container_id_) continue;
    std::shared_ptr<TableBlockSlot> slot;
    int rc = 0;
    CLIO_CO_AWAIT(LoadBlock(dir, k, slot, rc));
    if (rc != 0) continue;
    std::lock_guard<std::mutex> g(ns_mu_);
    for (const auto &kv : slot->blk_.rows_) {
      if (kv.second.state_ == kRowPending) continue;
      clio::cte::core::EncodeTagNameOp(
          out, clio::cte::core::TagNameOp::kAddName, FsUnpack(kv.second.id_),
          now, clio::cte::core::MakeTagRefName(FsUnpack(dir),
                                               RowName(kv.first, kv.second)));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::filesystem
