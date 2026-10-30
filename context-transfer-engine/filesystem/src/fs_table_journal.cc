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
 * Directory journal: how a home persists its directory blocks when the
 * deployment names a metadata_log_path.
 *
 * The blocks live in memory (fs_inode_table.cc). Each commit appends what changed
 * -- the same TableDelta list it pushes to the nodes caching the block -- so
 * a create costs a few bytes in one local write(2), not a rewrite of the
 * whole block in a CTE blob. A block's first commit (new, split off, the
 * root materialized, or after a failed append) journals its whole image, so
 * every block's deltas build on an image earlier in the journal.
 *
 * Replay applies images and deltas in order; a delta not newer than the
 * block is one a compaction already folded in, and is skipped. Compaction
 * rewrites the journal as one image per block, under dir_log_mu_ so no
 * append lands between the snapshot and the rewrite.
 */
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/filesystem/filesystem_runtime.h>

namespace clio::cte::filesystem {

namespace {

/**
 * Encode a kBlockDrop payload.
 * @param dir packed directory id
 * @param k block index
 * @return payload
 */
std::string EncBlockDrop(clio::run::u64 dir, clio::run::u32 k) {
  std::string out;
  FsEnc e(&out);
  e.U64(dir);
  e.U32(k);
  return out;
}

/**
 * Decode a kBlockDrop payload.
 * @param p payload
 * @param dir receives the directory id
 * @param k receives the block index
 * @return false if malformed
 */
bool DecBlockDrop(const std::string &p, clio::run::u64 *dir,
                  clio::run::u32 *k) {
  FsDec d(p.data(), p.size());
  return d.U64(dir) && d.U32(k);
}

}  // namespace

size_t Runtime::ReplayBlockDeltas(const std::string &p, TableBlockState *state) {
  std::vector<TableDelta> ds;
  if (!DecodeTableDeltas(p, &ds)) return 1;
  size_t bad = 0;
  for (const TableDelta &x : ds) {
    auto it = state->find(TableBlockKey{x.dir_, x.index_});
    if (it == state->end()) {
      ++bad;  // no image to build on (a dropped block's late commit)
      continue;
    }
    // Not newer: a compaction's image already holds it.
    if (x.kind_ != TableDeltaKind::kOps ||
        x.new_version_ <= it->second.version_) {
      continue;
    }
    if (!ApplyTableOps(&it->second, x)) ++bad;
  }
  return bad;
}

void Runtime::OpenDirJournal(const std::string &path) {
  dir_journal_ = false;
  if (path.empty()) {
    HLOG(kInfo, "filesystem: no metadata_log_path; directory blocks persist "
         "in CTE blobs");
    dir_ready_.store(true, std::memory_order_release);
    return;
  }
  if (!dir_log_.Open(path)) {
    HLOG(kError, "filesystem: cannot open directory journal {}: {}; "
         "directory blocks persist in CTE blobs", path, std::strerror(errno));
    dir_ready_.store(true, std::memory_order_release);
    return;
  }
  dir_journal_ = true;
  // Fresh start: what an earlier run left describes data that is not being
  // recovered, so the compaction below empties it.
  const size_t n = is_restart_ ? ReplayDirJournal() : 0;
  size_t blocks = 0;
  {
    std::lock_guard<std::mutex> g(dir_log_mu_);
    CompactDirJournalLocked();
  }
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    blocks = journaled_.size();
  }
  dir_log_synced_ms_ = SteadyMs();
  dir_ready_.store(true, std::memory_order_release);
  HLOG(kInfo, "filesystem: directory journal {}: {} ({} records, {} blocks)",
       path, is_restart_ ? "replayed" : "fresh", n, blocks);
}

size_t Runtime::ReplayDirJournal() {
  TableBlockState state;
  size_t bad = 0;
  const size_t n = dir_log_.Replay(
      [&state, &bad](FsLogRec t, const std::string &p) {
        if (t == FsLogRec::kBlockImage) {
          InodeTableBlock b;
          if (!DecodeTableBlock(p.data(), p.size(), &b)) {
            ++bad;
            return;
          }
          const TableBlockKey key{b.dir_, b.index_};
          state[key] = std::move(b);
        } else if (t == FsLogRec::kBlockDeltas) {
          bad += ReplayBlockDeltas(p, &state);
        } else if (t == FsLogRec::kBlockDrop) {
          clio::run::u64 dir = 0;
          clio::run::u32 k = 0;
          if (DecBlockDrop(p, &dir, &k)) state.erase(TableBlockKey{dir, k});
          else ++bad;
        }
      });
  if (bad != 0) {
    HLOG(kWarning, "filesystem: directory journal: {} change(s) had no block "
         "to apply to (blocks dropped, or written after a failed append)",
         bad);
  }
  // The durable form: an operation in flight at the crash is aborted, so
  // pending entries go, leaving ones stay, and no block stays sealed.
  std::lock_guard<std::mutex> g(ns_mu_);
  journaled_.clear();
  for (const auto &kv : state) {
    journaled_[kv.first] = EncodeTableBlock(kv.second, true);
  }
  return n;
}

void Runtime::CompactDirJournalLocked() {
  std::vector<std::pair<FsLogRec, std::string>> recs;
  {
    std::lock_guard<std::mutex> g(ns_mu_);
    recs.reserve(journaled_.size() + blocks_.size());
    for (const auto &kv : journaled_) {
      recs.emplace_back(FsLogRec::kBlockImage, kv.second);
    }
    // After journaled_: a block being installed can sit in both, and the
    // later record wins on replay.
    for (const auto &kv : blocks_) {
      if (!kv.second->home_) continue;
      recs.emplace_back(FsLogRec::kBlockImage,
                        EncodeTableBlock(kv.second->blk_, true));
    }
  }
  if (!dir_log_.Rewrite(recs)) {
    HLOG(kError, "filesystem: compacting the directory journal failed; it "
         "keeps growing until a compaction succeeds");
  }
}

int Runtime::JournalAppend(FsLogRec type, const std::string &payload) {
  std::lock_guard<std::mutex> g(dir_log_mu_);
  if (!dir_log_.Append(type, payload)) {
    return errno == ENOSPC ? ENOSPC : EIO;
  }
  if (dir_log_.BytesSinceCompact() > kDirJournalCompactBytes) {
    CompactDirJournalLocked();
  }
  return 0;
}

int Runtime::JournalDropBlock(clio::run::u64 dir, clio::run::u32 k) {
  std::lock_guard<std::mutex> lg(dir_log_mu_);
  if (!dir_log_.Append(FsLogRec::kBlockDrop, EncBlockDrop(dir, k))) {
    return errno == ENOSPC ? ENOSPC : EIO;
  }
  // Forgotten in the same step as the append: a compaction in between
  // would snapshot the block and drop this record, bringing it back.
  const TableBlockKey key{dir, k};
  std::lock_guard<std::mutex> g(ns_mu_);
  journaled_.erase(key);
  auto it = blocks_.find(key);
  if (it != blocks_.end() && it->second->home_ && it->second->blk_.sealed_) {
    blocks_.erase(it);
  }
  return 0;
}

int Runtime::TakeJournaledBlock(clio::run::u64 dir, clio::run::u32 k,
                                InodeTableBlock *out) {
  std::string img;
  {
    // Copied, not moved: it leaves journaled_ only once the slot is in
    // blocks_ (InstallLoadedBlock), so a compaction never misses it.
    std::lock_guard<std::mutex> g(ns_mu_);
    auto it = journaled_.find(TableBlockKey{dir, k});
    if (it == journaled_.end()) return ENOENT;
    img = it->second;
  }
  if (!DecodeTableBlock(img.data(), img.size(), out) || out->dir_ != dir ||
      out->index_ != k) {
    HLOG(kError, "filesystem: journaled directory block {}/{} unreadable",
         dir, k);
    return EIO;
  }
  return 0;
}

void Runtime::TakePersistLocked(TableBlockSlot &slot,
                                const std::vector<TableDelta> &deltas,
                                std::string *image, FsLogRec *jtype,
                                std::string *jrec) {
  if (!dir_journal_) {
    if (slot.persist_dirty_) {
      *image = EncodeTableBlock(slot.blk_, true);
      slot.persist_dirty_ = false;
    }
    return;
  }
  // Every change is journaled, durable or not: each delta builds on the one
  // before it, and replay drops what a crash aborts (pending, leaving).
  slot.persist_dirty_ = false;
  if (slot.journal_image_) {
    *jtype = FsLogRec::kBlockImage;
    *jrec = EncodeTableBlock(slot.blk_, true);
    slot.journal_image_ = false;
  } else if (!deltas.empty()) {
    *jtype = FsLogRec::kBlockDeltas;
    *jrec = EncodeTableDeltas(deltas);
  }
}

int Runtime::SyncDirJournal() {
  if (!dir_journal_) return 0;
  return dir_log_.Sync() ? 0 : EIO;
}

void Runtime::SyncDirJournalIfDue() {
  if (!dir_journal_) return;
  const clio::run::u64 now = SteadyMs();
  if (now - dir_log_synced_ms_ < kDirJournalSyncMs) return;
  dir_log_synced_ms_ = now;
  if (dir_log_.Unsynced()) dir_log_.Sync();
}

void Runtime::JournaledHomeKeys(std::vector<TableBlockKey> *out) {
  out->clear();
  std::lock_guard<std::mutex> g(ns_mu_);
  for (const auto &kv : journaled_) out->push_back(kv.first);
  for (const auto &kv : blocks_) {
    if (kv.second->home_ && journaled_.count(kv.first) == 0) {
      out->push_back(kv.first);
    }
  }
}

}  // namespace clio::cte::filesystem
