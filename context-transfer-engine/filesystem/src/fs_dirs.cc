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
 * Directory blocks: homes, caches and the push protocol (see fs_dir_block.h
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
#include <chrono>
#include <cstring>
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
/** A block load, wait or commit slower than this is reported (seconds). */
constexpr double kSlowBlockOpS = 10.0;
/** How long a home keeps retrying a push to a live holder (s). */
constexpr double kPushGiveUpS = 60.0;

/** SetAttr flag bits shared with fs_namespace.cc (kShardDirAttr). */
enum : clio::run::u32 {
  kSetAtime = 1u, kSetMtime = 2u, kSetAtimeNow = 4u, kSetMtimeNow = 8u,
  kSetUid = 16u, kSetGid = 32u, kSetMode = 64u, kAttrRepair = 128u,
  kSetCtimeOnly = 256u, kAccessTouch = 512u,
  kSetParent = 2048u,  // move: new parent id (req.b_) and name (req.leaf_)
};

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
      clio::cte::core::BlobHash(FsUnpack(dir), DirBlockName(k)) % n;
  return clio::cte::core::FailoverContainer(pool_id_, owner);
}

clio::run::TaskResume Runtime::ReadBlockBlob(clio::run::u64 dir,
                                             clio::run::u32 k, DirBlock *out,
                                             int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  const clio::cte::core::TagId tag = FsUnpack(dir);
  auto sz = cte_.AsyncGetBlobSize(tag, DirBlockName(k));
  CLIO_CO_AWAIT(sz);
  if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
    rc = ENOENT;
    CLIO_CO_RETURN;
  }
  std::string bytes(sz->size_, '\0');
  auto g = cte_.AsyncGetBlob(tag, DirBlockName(k), 0, bytes.size(), 0u,
                             bytes.data());
  CLIO_CO_AWAIT(g);
  if (g->GetReturnCode() != 0) {
    // Deleted between the two calls (an rmdir dropped it): it is gone.
    auto again = cte_.AsyncGetBlobSize(tag, DirBlockName(k));
    CLIO_CO_AWAIT(again);
    if (again->GetReturnCode() != 0 || again->size_ == 0) {
      rc = ENOENT;
      CLIO_CO_RETURN;
    }
  }
  // A block that shrank leaves stale bytes past its end; the codec stops at
  // the encoded length.
  if (g->GetReturnCode() != 0 ||
      !DecodeDirBlock(bytes.data(), bytes.size(), out) || out->dir_ != dir ||
      out->index_ != k) {
    HLOG(kError, "filesystem: directory block {:x}/{} unreadable", dir, k);
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
  auto p = cte_.AsyncPutBlob(tag, DirBlockName(k), 0, padded.size(),
                             padded.data(), -1.0f,
                             MetaCtx(inode_volatile_only_), 0u,
                             clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(p);
  if (p->GetReturnCode() != 0 && !inode_volatile_only_) {
    // No non-volatile tier accepts it: keep it in RAM (lost on restart).
    p = cte_.AsyncPutBlob(tag, DirBlockName(k), 0, padded.size(),
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
    HLOG(kError, "filesystem: writing directory block {:x}/{} failed (rc {})",
         dir, k, p->GetReturnCode());
    rc = EIO;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Loading and caching
// ===========================================================================

clio::run::TaskResume Runtime::LoadBlock(clio::run::u64 dir, clio::run::u32 k,
                                         std::shared_ptr<BlockSlot> &out,
                                         int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  out.reset();
  auto wait_t0 = std::chrono::steady_clock::now();
  double next_report_s = kSlowBlockOpS;
  for (int attempt = 0;; ++attempt) {
    const BlockKey key{dir, k};
    const clio::run::u32 home = BlockHome(dir, k);
    const bool mine = home == container_id_;
    bool busy = false;
    bool busy_commit = false;
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      auto it = blocks_.find(key);
      if (it != blocks_.end()) {
        BlockSlot &s = *it->second;
        if (s.home_ == mine && (mine || s.home_id_ == home)) {
          out = it->second;
          CLIO_CO_RETURN;
        }
        // Its home moved (failover or return): this copy is not current.
        if (s.committing_) busy = busy_commit = true;
        else blocks_.erase(it);
      }
      if (!busy) {
        busy = loading_.count(key) != 0;
        if (!busy) loading_.insert(key);
      }
    }
    if (busy) {
      const double waited = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - wait_t0).count();
      if (waited > next_report_s) {
        HLOG(kWarning, "filesystem: waiting {:.0f} s for directory block "
             "{}/{} (home {}, mine {}): {}", waited, dir, k, home, mine,
             busy_commit ? "a moved copy is still committing"
                         : "another task is loading it");
        next_report_s += kSlowBlockOpS;
      }
      CLIO_CO_AWAIT(clio::run::yield(kLoadPollUs));
      continue;
    }
    const auto load_t0 = std::chrono::steady_clock::now();
    auto slot = std::make_shared<BlockSlot>();
    int lrc = 0;
    if (mine) {
      CLIO_CO_AWAIT(ReadBlockBlob(dir, k, &slot->blk_, lrc));
      if (lrc == ENOENT && dir == FsPack(FsRootId()) && k == 0) {
        // "/" exists by definition: its home materializes it on first use.
        slot->blk_.dir_ = dir;
        slot->blk_.version_ = 1;
        const clio::run::u64 now = NowNs();
        slot->blk_.hdr_.atime_ = slot->blk_.hdr_.mtime_ =
            slot->blk_.hdr_.ctime_ = now;
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
      }
    } else {
      FsReq fr;
      fr.dir_id_ = dir;
      fr.block_ = k;
      fr.a_ = container_id_;
      FsResp resp;
      CLIO_CO_AWAIT(CallShard(home, kShardBlockFetch, fr, resp));
      lrc = static_cast<int>(resp.rc_);
      if (lrc == 0 && !DecodeDirBlock(resp.str_.data(), resp.str_.size(),
                                      &slot->blk_)) {
        lrc = EIO;
      }
      slot->home_id_ = home;
    }
    {
      const double took = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - load_t0).count();
      if (took > kSlowBlockOpS) {
        HLOG(kWarning, "filesystem: loading directory block {}/{} from {} "
             "took {:.1f} s (rc {})", dir, k, mine ? "its blob" : "its home",
             took, lrc);
      }
    }
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      loading_.erase(key);
      auto ev = early_.find(key);
      if (lrc == 0) {
        if (ev != early_.end()) {
          for (const DirDelta &x : ev->second) {
            if (x.kind_ == DirDeltaKind::kOps &&
                x.new_version_ > slot->blk_.version_) {
              ApplyDirOps(&slot->blk_, x);
            }
          }
        }
        blocks_[key] = slot;
        out = slot;
      }
      if (ev != early_.end()) early_.erase(ev);
    }
    if (lrc == kFsRedirect && attempt < 50) {
      // The home and this node disagree about membership for a moment.
      CLIO_CO_AWAIT(clio::run::yield(1000.0));
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

clio::run::TaskResume Runtime::ServeBlockFetch(const FsReq &req,
                                               FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (BlockHome(req.dir_id_, req.block_) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  std::shared_ptr<BlockSlot> slot;
  int rc = 0;
  CLIO_CO_AWAIT(LoadBlock(req.dir_id_, req.block_, slot, rc));
  if (rc != 0) {
    resp.rc_ = static_cast<clio::run::u32>(rc);
    CLIO_CO_RETURN;
  }
  std::lock_guard<std::mutex> g(ns_mu_);
  const clio::run::u32 holder = static_cast<clio::run::u32>(req.a_);
  if (holder != container_id_) slot->holders_[holder] = reg_seq_++;
  // The in-memory state, uncommitted changes included: every later change
  // reaches the holder as a push with a higher version.
  resp.str_ = EncodeDirBlock(slot->blk_, false);
  resp.rc_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

int Runtime::ApplyBlockPush(const FsReq &req) {
  std::vector<DirDelta> deltas;
  if (!DecodeDirDeltas(req.str_, &deltas)) return EINVAL;
  bool missing = false;
  std::lock_guard<std::mutex> g(ns_mu_);
  for (const DirDelta &x : deltas) {
    const BlockKey key{x.dir_, x.index_};
    if (loading_.count(key) != 0) {
      early_[key].push_back(x);  // applied once the fetched copy lands
      continue;
    }
    auto it = blocks_.find(key);
    if (it == blocks_.end() || it->second->home_) {
      if (x.kind_ != DirDeltaKind::kDrop) missing = true;
      continue;
    }
    DirBlock &b = it->second->blk_;
    if (x.kind_ == DirDeltaKind::kDrop) {
      blocks_.erase(it);
    } else if (x.kind_ == DirDeltaKind::kSnapshot) {
      DirBlock nb;
      if (DecodeDirBlock(x.snapshot_.data(), x.snapshot_.size(), &nb)) {
        b = std::move(nb);
      } else {
        blocks_.erase(it);
      }
    } else if (x.new_version_ > b.version_ && !ApplyDirOps(&b, x)) {
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
DirDelta SnapshotDelta(const DirBlock &b) {
  DirDelta x;
  x.kind_ = DirDeltaKind::kSnapshot;
  x.dir_ = b.dir_;
  x.index_ = b.index_;
  x.base_version_ = b.version_;
  x.new_version_ = b.version_;
  x.depth_ = b.depth_;
  x.mtime_ = b.mtime_;
  x.sealed_ = b.sealed_;
  x.snapshot_ = EncodeDirBlock(b, false);
  return x;
}
}  // namespace

clio::run::u64 Runtime::RecordChangeLocked(BlockSlot &slot, DirDelta delta,
                                           bool persisted) {
  DirBlock &b = slot.blk_;
  delta.kind_ = DirDeltaKind::kOps;
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

clio::run::TaskResume Runtime::PushDeltas(
    const std::vector<clio::run::u32> &holders, const std::string &deltas,
    std::vector<clio::run::u32> *gone) {
  CLIO_TASK_BODY_BEGIN
  std::vector<clio::run::u32> todo = holders;
  const auto start = std::chrono::steady_clock::now();
  while (!todo.empty()) {
    std::vector<std::pair<clio::run::u32, clio::run::Future<ShardOpTask>>> fs;
    for (clio::run::u32 h : todo) {
      if (!clio::cte::core::ContainerNodeAlive(pool_id_, h)) {
        gone->push_back(h);  // a dead node's cache died with it
        continue;
      }
      FsReq r;
      r.str_ = deltas;
      fs.emplace_back(h, SendShard(h, kShardBlockPush, r));
    }
    todo.clear();
    for (auto &hf : fs) {
      CLIO_CO_AWAIT(hf.second);
      FsResp resp;
      if (!ReadShardResp(hf.second, &resp)) {
        todo.push_back(hf.first);  // unreachable but not declared dead yet
      } else if (resp.rc_ == ENOENT) {
        gone->push_back(hf.first);  // it no longer caches the block
      }
    }
    if (todo.empty()) break;
    const double waited = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (waited > kPushGiveUpS) {
      HLOG(kError, "filesystem: {} holder(s) unreachable for {} s but not "
           "declared dead; dropping them (their caches may be stale)",
           todo.size(), waited);
      gone->insert(gone->end(), todo.begin(), todo.end());
      break;
    }
    CLIO_CO_AWAIT(clio::run::yield(100000.0));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CommitBlock(std::shared_ptr<BlockSlot> slot,
                                           clio::run::u64 version, int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  for (;;) {
    std::vector<DirDelta> deltas;
    std::string image;
    std::vector<clio::run::u32> holders;
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
        deltas.swap(slot->unpushed_);
        if (slot->persist_dirty_) {
          image = EncodeDirBlock(slot->blk_, true);
          slot->persist_dirty_ = false;
        }
        reg = slot->holders_;
        if (slot->holders_unknown_) {
          resync = true;
          slot->holders_unknown_ = false;
          deltas.assign(1, SnapshotDelta(slot->blk_));
          for (clio::run::u32 c = 0; c < NumContainers(); ++c) {
            if (c != container_id_) holders.push_back(c);
          }
        } else {
          for (const auto &hv : reg) holders.push_back(hv.first);
        }
      }
    }
    if (!go) {
      CLIO_CO_AWAIT(clio::run::yield(kCommitPollUs));
      continue;
    }
    int wrc = 0;
    const auto commit_t0 = std::chrono::steady_clock::now();
    if (!image.empty()) CLIO_CO_AWAIT(WriteBlockBlob(dir, k, image, wrc));
    std::vector<clio::run::u32> gone;
    if (!deltas.empty() && !holders.empty()) {
      const std::string enc = EncodeDirDeltas(deltas);
      CLIO_CO_AWAIT(PushDeltas(holders, enc, &gone));
    }
    {
      const double took = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - commit_t0).count();
      if (took > kSlowBlockOpS) {
        HLOG(kWarning, "filesystem: committing directory block {}/{} took "
             "{:.1f} s ({} holders{})", dir, k, took, holders.size(),
             resync ? ", resync" : "");
      }
    }
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      if (resync) {
        // Everyone who took the snapshot caches the block: register them.
        for (clio::run::u32 h : holders) {
          if (std::find(gone.begin(), gone.end(), h) == gone.end() &&
              slot->holders_.count(h) == 0) {
            slot->holders_[h] = reg_seq_++;
          }
        }
      }
      for (clio::run::u32 h : gone) {
        // Only the registration this push went to: a holder that dropped its
        // copy and fetched again meanwhile stays registered.
        auto it = slot->holders_.find(h);
        if (it != slot->holders_.end() && it->second == reg[h]) {
          slot->holders_.erase(it);
        }
      }
      if (wrc != 0) slot->persist_dirty_ = true;  // the next commit retries
      slot->durable_version_ = v;
      slot->committing_ = false;
    }
    if (wrc != 0) {
      rc = EIO;
      break;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Entry mutations (on the block's home)
// ===========================================================================

int Runtime::InsertEntryLocked(BlockSlot &slot, const FsReq &req,
                               FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  DirBlock &b = slot.blk_;
  if (b.sealed_) return ENOENT;  // an rmdir is removing the directory
  const clio::cte::core::TagId dir_tag = FsUnpack(b.dir_);
  const std::string tn = clio::cte::core::MakeTagRefName(dir_tag, req.leaf_);
  auto it = b.ents_.find(req.leaf_);
  DirDelta dl;
  if ((f & kInsCommit) != 0) {
    if (it == b.ents_.end()) {
      // The reservation is gone: pending entries are never durable, so a
      // home that restarted or failed over (and back) lost it. The name is
      // still free, so take it now -- the check is as atomic here as the
      // reservation was.
      DirEntry e;
      e.id_ = req.id_;
      e.type_ = req.type_;
      e.state_ = kDirEntPending;
      it = b.ents_.emplace(req.leaf_, e).first;
    } else if (it->second.id_ != req.id_) {
      return EEXIST;  // someone else took the name after we lost it
    } else if (it->second.state_ == kDirEntLive) {
      return 0;  // already committed (a retried commit)
    } else if (it->second.state_ != kDirEntPending) {
      return ENOENT;
    }
    it->second.state_ = kDirEntLive;
    b.mtime_ = NowNs();
    dl.ops_.push_back({req.leaf_, true, it->second});
    RecordChangeLocked(slot, dl, true);
    PublishName(clio::cte::core::TagNameOp::kAddName, FsUnpack(req.id_), tn);
    return 0;
  }
  if (it != b.ents_.end()) {
    DirEntry &e = it->second;
    if (e.state_ != kDirEntLive) return (f & kInsFailBusy) ? EBUSY : kFsRetry;
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
    const clio::run::u64 victim = e.id_;
    e.id_ = req.id_;
    e.type_ = req.type_;
    b.mtime_ = NowNs();
    dl.ops_.push_back({req.leaf_, true, e});
    RecordChangeLocked(slot, dl, true);
    if ((f & kInsNoTagName) == 0) {
      PublishName(clio::cte::core::TagNameOp::kRemoveName, FsUnpack(victim),
                  tn);
      PublishName(clio::cte::core::TagNameOp::kAddName, FsUnpack(e.id_), tn);
    }
    resp.id_ = req.id_;
    resp.type_ = req.type_;
    resp.attr_.id_ = b.dir_;
    return 0;
  }
  DirEntry e;
  e.type_ = req.type_;
  e.state_ = (f & kInsPending) ? kDirEntPending : kDirEntLive;
  if ((f & kInsNewInode) != 0) {
    const clio::run::u64 want = req.id_;
    if (want != 0 &&
        (InodeOwner(want) != container_id_ || FindInode(want) != nullptr)) {
      return EEXIST;  // a client-minted id that is not ours or is taken
    }
    const clio::cte::core::TagId id = want != 0 ? FsUnpack(want) : MintId();
    if (id.IsNull()) return kFsRetry;  // reservation in flight
    e.id_ = FsPack(id);
    NewInode(id, req.type_, req.mode_, req.str_, FsJoin(req.dir_, req.leaf_));
  } else {
    const clio::cte::core::TagId id = req.id_ != 0 ? FsUnpack(req.id_) : MintId();
    if (id.IsNull()) return kFsRetry;
    e.id_ = FsPack(id);
  }
  b.ents_[req.leaf_] = e;
  const bool live = e.state_ == kDirEntLive;
  if (live) b.mtime_ = NowNs();
  dl.ops_.push_back({req.leaf_, true, e});
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

int Runtime::RemoveEntryLocked(BlockSlot &slot, const FsReq &req,
                               FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  DirBlock &b = slot.blk_;
  auto it = b.ents_.find(req.leaf_);
  if (it == b.ents_.end()) return ENOENT;
  DirEntry &e = it->second;
  const bool mine = req.id_ != 0 && e.id_ == req.id_;
  DirDelta dl;
  if ((f & kRmRestore) != 0) {
    if (mine && e.state_ == kDirEntLeaving) {
      e.state_ = kDirEntLive;
      dl.ops_.push_back({req.leaf_, true, e});
      RecordChangeLocked(slot, dl, false);
    }
    return 0;
  }
  if (req.id_ != 0 && !mine) return ENOENT;  // replaced underneath us
  if (e.state_ == kDirEntPending && mine) {  // abort a reservation
    b.ents_.erase(it);
    dl.ops_.push_back({req.leaf_, false, DirEntry()});
    RecordChangeLocked(slot, dl, false);
    return 0;
  }
  // A leaving entry may only be finished by the operation that marked it
  // (which names the id); everyone else waits for the outcome.
  if (e.state_ != kDirEntLive && !(e.state_ == kDirEntLeaving && mine &&
                                   (f & kRmMarkLeaving) == 0)) {
    return (f & kRmFailBusy) ? EBUSY : kFsRetry;
  }
  if ((f & kRmNonDir) != 0 && e.type_ == kFsTypeDir) return EISDIR;
  if ((f & kRmDirOnly) != 0 && e.type_ != kFsTypeDir) return ENOTDIR;
  resp.old_id_ = e.id_;
  resp.old_type_ = e.type_;
  resp.attr_.id_ = b.dir_;
  if ((f & kRmMarkLeaving) != 0) {
    e.state_ = kDirEntLeaving;
    dl.ops_.push_back({req.leaf_, true, e});
    RecordChangeLocked(slot, dl, false);
    return 0;
  }
  const DirEntry gone = e;
  b.ents_.erase(it);
  b.mtime_ = NowNs();
  dl.ops_.push_back({req.leaf_, false, DirEntry()});
  RecordChangeLocked(slot, dl, true);
  if ((f & kRmNoTagName) == 0) {
    PublishName(clio::cte::core::TagNameOp::kRemoveName, FsUnpack(gone.id_),
                clio::cte::core::MakeTagRefName(FsUnpack(b.dir_), req.leaf_));
  }
  return 0;
}

clio::run::TaskResume Runtime::EntryMutation(clio::run::u32 op,
                                             const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (BlockHome(req.dir_id_, req.block_) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  std::shared_ptr<BlockSlot> slot;
  int rc = 0;
  CLIO_CO_AWAIT(LoadBlock(req.dir_id_, req.block_, slot, rc));
  if (rc != 0) {
    resp.rc_ = static_cast<clio::run::u32>(rc);
    CLIO_CO_RETURN;
  }
  if (op == kShardInsert) CLIO_CO_AWAIT(EnsureIdReserve(1024));
  const clio::run::u64 h = DirNameHash(req.leaf_);
  clio::run::u64 before = 0, after = 0;
  for (;;) {
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      before = slot->blk_.version_;
      if (!BlockOwnsName(slot->blk_, h)) {
        rc = kFsRedirect;  // it split: the caller re-walks
      } else if (slot->splitting_) {
        rc = kFsRetry;
      } else {
        rc = op == kShardInsert ? InsertEntryLocked(*slot, req, resp)
                                : RemoveEntryLocked(*slot, req, resp);
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
  if (rc == 0 && op == kShardInsert) CLIO_CO_AWAIT(SplitBlock(slot));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Splits
// ===========================================================================

clio::run::TaskResume Runtime::SplitBlock(std::shared_ptr<BlockSlot> slot) {
  CLIO_TASK_BODY_BEGIN
  DirBlock child;
  std::vector<std::string> moved;
  clio::run::u32 d = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    DirBlock &b = slot->blk_;
    if (!slot->home_ || slot->splitting_ || b.sealed_ ||
        b.ents_.size() <= split_entries_ || b.depth_ >= kDirMaxDepth) {
      CLIO_CO_RETURN;
    }
    d = b.depth_;
    for (const auto &kv : b.ents_) {
      if (((DirNameHash(kv.first) >> d) & 1ULL) == 0) continue;
      // An operation in flight on this name finishes here first.
      if (kv.second.state_ != kDirEntLive) CLIO_CO_RETURN;
      moved.push_back(kv.first);
    }
    slot->splitting_ = true;
    child.dir_ = b.dir_;
    child.index_ = b.index_ + (1u << d);
    child.born_ = child.depth_ = d + 1;
    child.version_ = 1;
    child.mtime_ = b.mtime_;
    for (const std::string &leaf : moved) child.ents_[leaf] = b.ents_[leaf];
  }
  // The new block is written by its own home before the old one lets go:
  // until the old block's depth changes nobody routes to the new one, so a
  // crash in between leaves an unreachable orphan, never a lost entry.
  FsReq ir;
  ir.dir_id_ = child.dir_;
  ir.block_ = child.index_;
  ir.str_ = EncodeDirBlock(child, false);
  FsResp irr;
  CLIO_CO_AWAIT(CallShard(BlockHome(child.dir_, child.index_),
                          kShardBlockInstall, ir, irr));
  clio::run::u64 v = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    if (irr.rc_ == 0) {
      DirBlock &b = slot->blk_;
      DirDelta dl;
      for (const std::string &leaf : moved) {
        b.ents_.erase(leaf);
        dl.ops_.push_back({leaf, false, DirEntry()});
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
    HLOG(kWarning, "filesystem: splitting directory block {:x}/{} failed "
         "(rc {}); it stays whole", child.dir_, child.index_ - (1u << d),
         irr.rc_);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::InstallBlock(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  auto slot = std::make_shared<BlockSlot>();
  if (!DecodeDirBlock(req.str_.data(), req.str_.size(), &slot->blk_)) {
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
    blocks_[BlockKey{slot->blk_.dir_, slot->blk_.index_}] = slot;
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

clio::run::TaskResume Runtime::DirCreate(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (BlockHome(req.dir_id_, 0) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  if ((req.flags_ & kAttrRepair) != 0) {
    // Only when truly missing (a crash between rmdir's halves left the
    // parent's entry but not the listing).
    std::shared_ptr<BlockSlot> have;
    int lrc = 0;
    CLIO_CO_AWAIT(LoadBlock(req.dir_id_, 0, have, lrc));
    if (lrc != ENOENT) {
      resp.rc_ = static_cast<clio::run::u32>(lrc);
      CLIO_CO_RETURN;
    }
  }
  auto slot = std::make_shared<BlockSlot>();
  DirBlock &b = slot->blk_;
  b.dir_ = req.dir_id_;
  b.version_ = 1;
  b.hdr_.parent_ = req.b_;
  b.hdr_.leaf_ = req.leaf_;
  b.hdr_.mode_ = req.mode_;
  b.hdr_.uid_ = req.uid_;
  b.hdr_.gid_ = req.gid_;
  const clio::run::u64 now = NowNs();
  b.hdr_.atime_ = b.hdr_.mtime_ = b.hdr_.ctime_ = now;
  slot->home_ = true;
  slot->persist_dirty_ = true;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    blocks_[BlockKey{b.dir_, 0}] = slot;
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
  std::shared_ptr<BlockSlot> slot;
  int rc = 0;
  CLIO_CO_AWAIT(LoadBlock(req.dir_id_, req.block_, slot, rc));
  if (rc != 0) {
    resp.rc_ = static_cast<clio::run::u32>(rc);
    CLIO_CO_RETURN;
  }
  clio::run::u64 v = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    DirBlock &b = slot->blk_;
    resp.str_ = EncIndices(ChildBlocks(b));
    const bool seal = req.a_ != 0;
    if (seal && (!b.ents_.empty() || slot->splitting_)) {
      rc = ENOTEMPTY;  // pending entries (a mkdir inside) count too
    } else if (b.sealed_ != seal) {
      b.sealed_ = seal;
      v = RecordChangeLocked(*slot, DirDelta(), false);
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
  std::shared_ptr<BlockSlot> sealed;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    auto it = blocks_.find(BlockKey{req.dir_id_, req.block_});
    if (it != blocks_.end() && it->second->home_) {
      if (!it->second->blk_.sealed_) {
        resp.rc_ = EBUSY;
        CLIO_CO_RETURN;
      }
      for (const auto &hv : it->second->holders_) holders.push_back(hv.first);
      sealed = it->second;
    }
  }
  // The sealed block stays cached until its blob is gone: a racing create
  // finds it sealed (ENOENT), and no load can bring back the blob's
  // unsealed image in the meantime.
  auto d = cte_.AsyncDelBlob(FsUnpack(req.dir_id_), DirBlockName(req.block_),
                             clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(d);
  if (sealed != nullptr) {
    std::lock_guard<std::mutex> g(ns_mu_);
    auto it = blocks_.find(BlockKey{req.dir_id_, req.block_});
    if (it != blocks_.end() && it->second == sealed) blocks_.erase(it);
  }
  if (!holders.empty()) {
    DirDelta x;
    x.kind_ = DirDeltaKind::kDrop;
    x.dir_ = req.dir_id_;
    x.index_ = req.block_;
    std::vector<DirDelta> one;
    one.push_back(x);
    const std::string enc = EncodeDirDeltas(one);
    std::vector<clio::run::u32> gone;
    CLIO_CO_AWAIT(PushDeltas(holders, enc, &gone));
  }
  resp.rc_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DirAttrOp(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u32 f = req.flags_;
  if (BlockHome(req.dir_id_, 0) != container_id_) {
    resp.rc_ = kFsRedirect;
    CLIO_CO_RETURN;
  }
  std::shared_ptr<BlockSlot> slot;
  int rc = 0;
  CLIO_CO_AWAIT(LoadBlock(req.dir_id_, 0, slot, rc));
  if (rc != 0) {
    resp.rc_ = static_cast<clio::run::u32>(rc);
    CLIO_CO_RETURN;
  }
  clio::run::u64 v = 0;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    DirHeader &h = slot->blk_.hdr_;
    const clio::run::u64 now = NowNs();
    bool changed = false;
    if ((f & kAccessTouch) == 0) {  // directory reads do not track atime
      if (f & kSetAtimeNow) { h.atime_ = now; changed = true; }
      else if (f & kSetAtime) { h.atime_ = req.a_; changed = true; }
      if (f & kSetMtimeNow) { h.mtime_ = now; changed = true; }
      else if (f & kSetMtime) { h.mtime_ = req.b_; changed = true; }
      if (f & kSetUid) { h.uid_ = req.uid_; changed = true; }
      if (f & kSetGid) { h.gid_ = req.gid_; changed = true; }
      if (f & kSetMode) { h.mode_ = req.mode_ & 07777u; changed = true; }
      if (f & kSetParent) {
        h.parent_ = req.id_;
        h.leaf_ = req.leaf_;
        changed = true;
      }
      if (changed || (f & kSetCtimeOnly)) {
        h.ctime_ = now;
        DirDelta dl;
        dl.has_hdr_ = true;
        dl.hdr_ = h;
        v = RecordChangeLocked(*slot, dl, true);
      }
    }
  }
  if (v != 0) {
    int crc = 0;
    CLIO_CO_AWAIT(CommitBlock(slot, v, crc));
    rc = crc;
  }
  if (rc == 0) {
    CLIO_CO_AWAIT(DirStat(req.dir_id_, resp));
  } else {
    resp.rc_ = static_cast<clio::run::u32>(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SealDir(clio::run::u64 dir, bool seal,
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
    CLIO_CO_AWAIT(CallShard(BlockHome(dir, k), kShardBlockSeal, r, resp));
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
      CLIO_CO_AWAIT(CallShard(BlockHome(dir, k), kShardBlockSeal, r, resp));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DropDir(clio::run::u64 dir) {
  CLIO_TASK_BODY_BEGIN
  std::vector<clio::run::u32> blocks;
  int rc = 0;
  CLIO_CO_AWAIT(DirBlocks(dir, &blocks, rc));
  for (clio::run::u32 k : blocks) {
    FsReq r;
    r.dir_id_ = dir;
    r.block_ = k;
    FsResp resp;
    CLIO_CO_AWAIT(CallShard(BlockHome(dir, k), kShardBlockDrop, r, resp));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Reads through the cache
// ===========================================================================

clio::run::TaskResume Runtime::DirBlocks(clio::run::u64 dir,
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
    std::shared_ptr<BlockSlot> slot;
    CLIO_CO_AWAIT(LoadBlock(dir, k, slot, rc));
    if (rc != 0) CLIO_CO_RETURN;
    out->push_back(k);
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      const DirBlock &b = slot->blk_;
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
    clio::run::u64 dir, std::vector<std::pair<std::string, DirEntry>> *out,
    clio::run::u64 *newest, int &rc) {
  CLIO_TASK_BODY_BEGIN
  out->clear();
  *newest = 0;
  std::vector<clio::run::u32> blocks;
  CLIO_CO_AWAIT(DirBlocks(dir, &blocks, rc));
  if (rc != 0) CLIO_CO_RETURN;
  for (clio::run::u32 k : blocks) {
    std::shared_ptr<BlockSlot> slot;
    CLIO_CO_AWAIT(LoadBlock(dir, k, slot, rc));
    if (rc != 0) CLIO_CO_RETURN;
    std::lock_guard<std::mutex> g(ns_mu_);
    *newest = std::max(*newest, slot->blk_.mtime_);
    for (const auto &kv : slot->blk_.ents_) {
      if (kv.second.state_ == kDirEntPending) continue;
      out->emplace_back(kv.first, kv.second);
    }
  }
  std::sort(out->begin(), out->end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DirStat(clio::run::u64 dir, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  std::vector<clio::run::u32> blocks;
  int rc = 0;
  CLIO_CO_AWAIT(DirBlocks(dir, &blocks, rc));
  DirHeader hdr;
  clio::run::u64 newest = 0;
  for (size_t i = 0; rc == 0 && i < blocks.size(); ++i) {
    std::shared_ptr<BlockSlot> slot;
    CLIO_CO_AWAIT(LoadBlock(dir, blocks[i], slot, rc));
    if (rc != 0) break;
    std::lock_guard<std::mutex> g(ns_mu_);
    if (blocks[i] == 0) hdr = slot->blk_.hdr_;
    newest = std::max(newest, slot->blk_.mtime_);
  }
  resp.rc_ = static_cast<clio::run::u32>(rc);
  if (rc != 0) CLIO_CO_RETURN;
  FsAttr &a = resp.attr_;
  a.id_ = dir;
  a.type_ = kFsTypeDir;
  a.size_ = 0;
  a.nlink_ = 2;
  a.mode_ = hdr.mode_;
  a.uid_ = hdr.uid_;
  a.gid_ = hdr.gid_;
  a.atime_ = hdr.atime_;
  DirEffectiveTimes(hdr, newest, &a.mtime_, &a.ctime_);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::WalkToBlock(clio::run::u64 dir,
                                           const std::string &leaf,
                                           std::shared_ptr<BlockSlot> &slot,
                                           int &rc) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 h = DirNameHash(leaf);
  clio::run::u32 k = 0;
  for (clio::run::u32 step = 0; step <= kDirMaxDepth + 1; ++step) {
    CLIO_CO_AWAIT(LoadBlock(dir, k, slot, rc));
    if (rc != 0) CLIO_CO_RETURN;
    clio::run::u32 next = 0;
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      next = NextBlockForName(slot->blk_, h);
    }
    if (next == k) CLIO_CO_RETURN;
    k = next;
  }
  rc = EIO;  // a split tree deeper than possible: corrupt block
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::LookupEntry(clio::run::u64 dir,
                                           const std::string &leaf,
                                           DirEntry &out, int &rc) {
  CLIO_TASK_BODY_BEGIN
  std::shared_ptr<BlockSlot> slot;
  CLIO_CO_AWAIT(WalkToBlock(dir, leaf, slot, rc));
  if (rc != 0) CLIO_CO_RETURN;
  std::lock_guard<std::mutex> g(ns_mu_);
  auto it = slot->blk_.ents_.find(leaf);
  if (it == slot->blk_.ents_.end() || it->second.state_ == kDirEntPending) {
    rc = ENOENT;
  } else {
    out = it->second;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ResolvePath(const std::string &path,
                                           DirEntry &ent,
                                           clio::run::u64 &parent, int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  parent = 0;
  ent = DirEntry();
  ent.id_ = FsPack(FsRootId());
  ent.type_ = kFsTypeDir;
  if (path == "/") CLIO_CO_RETURN;
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
    DirEntry next;
    CLIO_CO_AWAIT(LookupEntry(parent, comp, next, rc));
    if (rc != 0) CLIO_CO_RETURN;
    ent = next;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::EntryOp(clio::run::u32 op, FsReq req,
                                       FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  for (int attempt = 0; attempt < 200; ++attempt) {
    std::shared_ptr<BlockSlot> slot;
    int rc = 0;
    CLIO_CO_AWAIT(WalkToBlock(req.dir_id_, req.leaf_, slot, rc));
    if (rc != 0) {
      resp = FsResp();
      resp.rc_ = static_cast<clio::run::u32>(rc);
      CLIO_CO_RETURN;
    }
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      req.block_ = slot->blk_.index_;
    }
    CLIO_CO_AWAIT(CallShard(BlockHome(req.dir_id_, req.block_), op, req, resp));
    if (resp.rc_ != static_cast<clio::run::u32>(kFsRedirect)) CLIO_CO_RETURN;
    {
      // Our copy said the name was here and the home says it is not: drop
      // the copy (a push may still be on its way) and walk again.
      std::lock_guard<std::mutex> g(ns_mu_);
      auto it = blocks_.find(BlockKey{req.dir_id_, req.block_});
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
    std::shared_ptr<BlockSlot> slot;
    CLIO_CO_AWAIT(LoadBlock(cur, 0, slot, rc));
    if (rc != 0) CLIO_CO_RETURN;
    DirHeader hdr;
    {
      std::lock_guard<std::mutex> g(ns_mu_);
      hdr = slot->blk_.hdr_;
    }
    DirEntry up;
    CLIO_CO_AWAIT(LookupEntry(hdr.parent_, hdr.leaf_, up, rc));
    // An ancestor that is itself being moved (or removed) could close a
    // cycle with this move: back off and let it finish.
    if (rc != 0 || up.id_ != cur || up.state_ == kDirEntLeaving) {
      rc = EBUSY;
      CLIO_CO_RETURN;
    }
    cur = hdr.parent_;
  }
  rc = EBUSY;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::EncodeHomeNames(std::string *out) {
  CLIO_TASK_BODY_BEGIN
  out->clear();
  auto list = cte_.AsyncListLocalBlobs("d\\.[0-9]+",
                                       clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(list);
  const clio::run::u64 now = NowNs();
  for (size_t i = 0; i < list->tag_ids_.size(); ++i) {
    const clio::run::u64 dir = list->tag_ids_[i];
    const clio::run::u32 k = static_cast<clio::run::u32>(
        std::strtoul(list->blob_names_[i].c_str() + 2, nullptr, 10));
    if (BlockHome(dir, k) != container_id_) continue;
    std::shared_ptr<BlockSlot> slot;
    int rc = 0;
    CLIO_CO_AWAIT(LoadBlock(dir, k, slot, rc));
    if (rc != 0) continue;
    std::lock_guard<std::mutex> g(ns_mu_);
    for (const auto &kv : slot->blk_.ents_) {
      if (kv.second.state_ == kDirEntPending) continue;
      clio::cte::core::EncodeTagNameOp(
          out, clio::cte::core::TagNameOp::kAddName, FsUnpack(kv.second.id_),
          now, clio::cte::core::MakeTagRefName(FsUnpack(dir), kv.first));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::filesystem
