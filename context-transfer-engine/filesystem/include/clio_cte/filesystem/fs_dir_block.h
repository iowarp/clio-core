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
 * Directory blocks: how clio-fs stores directories in the CTE.
 *
 * A directory is a CTE tag whose id is the directory's inode number. Its
 * entries live in "directory blocks", one CTE blob each ("d.<k>" in the
 * directory's tag), so the CTE persists, replicates and fails them over like
 * any other blob. The CTE places each block by hash(dir id, block name): a
 * small directory is one block on one node, a large one spreads over many.
 *
 * Extendible hashing over the low bits of hash64(name):
 *   - Block k holds the names whose low `born_` bits equal k.
 *   - Block 0 is born at depth 0 and owns every name at first.
 *   - A full block at depth d splits: the names with bit d set move to the
 *     new block k + 2^d (born at depth d+1); the old block's depth becomes
 *     d+1. Blocks never merge.
 *   - So a block's children are k + 2^d for d in [born_, depth_), and a
 *     name's block is found by walking down from block 0 (NextBlockForName):
 *     no bitmap, no directory of blocks, nothing to keep in sync.
 *
 * Block 0 also carries the directory's own header: attributes, and the
 * parent pointer (id + name) rename uses to refuse moving a directory into
 * its own subtree.
 *
 * Every block has one home (the container that owns its CTE blob). The home
 * is the only writer; every other node holding a cached copy receives each
 * change as a DirDelta before the change is acknowledged (see fs_dirs.cc).
 */
#ifndef CLIO_CTE_FILESYSTEM_FS_DIR_BLOCK_H_
#define CLIO_CTE_FILESYSTEM_FS_DIR_BLOCK_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/filesystem/fs_shard.h>

namespace clio::cte::filesystem {

/** Entry states: pending is reserved but invisible, leaving is visible but
 *  frozen (only the operation that marked it may finish or undo it). */
GLOBAL_CROSS_CONST clio::run::u32 kDirEntLive = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDirEntPending = 1;
GLOBAL_CROSS_CONST clio::run::u32 kDirEntLeaving = 2;

/** Deepest split a directory may reach (2^24 blocks). */
GLOBAL_CROSS_CONST clio::run::u32 kDirMaxDepth = 24;

/** One directory entry. */
struct DirEntry {
  clio::run::u64 id_ = 0;              ///< packed child id
  clio::run::u32 type_ = kFsTypeFile;  ///< kFsType*
  clio::run::u32 state_ = kDirEntLive;
};

/** The directory's own attributes (block 0 only). */
struct DirHeader {
  clio::run::u64 parent_ = 0;          ///< parent dir id; 0 for the root
  std::string leaf_;                   ///< this directory's name in parent_
  clio::run::u32 mode_ = 0xFFFFFFFFu;  ///< all-ones = adapter default
  clio::run::u32 uid_ = 0xFFFFFFFFu;
  clio::run::u32 gid_ = 0xFFFFFFFFu;
  clio::run::u64 atime_ = 0, mtime_ = 0, ctime_ = 0;  ///< ns
};

/** One directory block (see the file comment). */
struct DirBlock {
  clio::run::u64 dir_ = 0;      ///< packed directory id
  clio::run::u32 index_ = 0;    ///< k
  clio::run::u32 born_ = 0;     ///< depth the block was created at
  clio::run::u32 depth_ = 0;    ///< current depth (born_ + splits)
  clio::run::u64 version_ = 0;  ///< bumped by every change
  clio::run::u64 mtime_ = 0;    ///< ns of this block's last entry change
  bool sealed_ = false;         ///< an rmdir holds it: inserts fail ENOENT
  DirHeader hdr_;               ///< block 0 only
  std::map<std::string, DirEntry> ents_;  ///< sorted: stable readdir
};

/** Blob name of block k in the directory's tag. */
inline std::string DirBlockName(clio::run::u32 k) {
  return "d." + std::to_string(k);
}

/**
 * 64-bit hash of an entry name (FNV-1a, then mixed): its low bits pick the
 * block. Identical in every process and build.
 * @param leaf entry name
 * @return hash
 */
inline clio::run::u64 DirNameHash(const std::string &leaf) {
  clio::run::u64 h = 0xCBF29CE484222325ULL;
  for (unsigned char c : leaf) {
    h ^= c;
    h *= 0x100000001B3ULL;
  }
  return FsMix64(h);
}

/**
 * The block a name belongs to, one step down from `b`: `b` itself, or the
 * child the name moved to when `b` split.
 * @param b a block whose range contains the name
 * @param h DirNameHash of the name
 * @return b.index_ if `b` holds the name, else the child to look in next
 */
inline clio::run::u32 NextBlockForName(const DirBlock &b, clio::run::u64 h) {
  for (clio::run::u32 d = b.born_; d < b.depth_; ++d) {
    if ((h >> d) & 1ULL) return b.index_ + (1u << d);
  }
  return b.index_;
}

/**
 * Whether `b` itself holds the name with hash `h` (its range and no split
 * moved it away).
 * @param b block
 * @param h DirNameHash of the name
 * @return true if the name belongs in `b`
 */
inline bool BlockOwnsName(const DirBlock &b, clio::run::u64 h) {
  const clio::run::u64 mask =
      b.born_ >= 64 ? ~0ULL : ((1ULL << b.born_) - 1ULL);
  return (h & mask) == b.index_ && NextBlockForName(b, h) == b.index_;
}

/**
 * The blocks `b` split off.
 * @param b block
 * @return child block indices
 */
inline std::vector<clio::run::u32> ChildBlocks(const DirBlock &b) {
  std::vector<clio::run::u32> out;
  for (clio::run::u32 d = b.born_; d < b.depth_; ++d) {
    out.push_back(b.index_ + (1u << d));
  }
  return out;
}

/**
 * A directory's effective mtime/ctime from its header and all its blocks.
 * An explicit utimens (stamped in hdr.ctime_) wins until a later entry
 * change, which is what `tar` relies on when it restores directory times
 * after extracting their contents.
 * @param hdr block 0's header
 * @param newest_block_mtime max mtime_ over every block of the directory
 * @param mtime receives the effective mtime
 * @param ctime receives the effective ctime
 */
inline void DirEffectiveTimes(const DirHeader &hdr,
                              clio::run::u64 newest_block_mtime,
                              clio::run::u64 *mtime, clio::run::u64 *ctime) {
  *mtime = newest_block_mtime > hdr.ctime_ ? newest_block_mtime : hdr.mtime_;
  *ctime = newest_block_mtime > hdr.ctime_ ? newest_block_mtime : hdr.ctime_;
}

/** Number of entries that count toward the directory being non-empty. */
inline size_t DirBlockLiveCount(const DirBlock &b) {
  size_t n = 0;
  for (const auto &kv : b.ents_) {
    if (kv.second.state_ != kDirEntPending) ++n;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

/** Encode a header. */
inline void EncDirHeader(FsEnc &e, const DirHeader &h) {
  e.U64(h.parent_);
  e.Str(h.leaf_);
  e.U32(h.mode_);
  e.U32(h.uid_);
  e.U32(h.gid_);
  e.U64(h.atime_);
  e.U64(h.mtime_);
  e.U64(h.ctime_);
}

/** Decode a header. @return false on underflow */
inline bool DecDirHeader(FsDec &d, DirHeader *h) {
  return d.U64(&h->parent_) && d.Str(&h->leaf_) && d.U32(&h->mode_) &&
         d.U32(&h->uid_) && d.U32(&h->gid_) && d.U64(&h->atime_) &&
         d.U64(&h->mtime_) && d.U64(&h->ctime_);
}

/** Format tag of an encoded block. */
GLOBAL_CROSS_CONST clio::run::u32 kDirBlockMagic = 0x31424443u;  // "CDB1"

/**
 * Encode a block. `durable` gives the persisted image: pending entries are
 * left out and leaving ones stored live, because those states belong to an
 * operation in flight, which a crash aborts.
 * @param b block
 * @param durable true for the CTE blob, false for a cache transfer
 * @return bytes
 */
inline std::string EncodeDirBlock(const DirBlock &b, bool durable) {
  std::string out;
  FsEnc e(&out);
  e.U32(kDirBlockMagic);
  e.U64(b.dir_);
  e.U32(b.index_);
  e.U32(b.born_);
  e.U32(b.depth_);
  e.U64(b.version_);
  e.U64(b.mtime_);
  e.U32(b.sealed_ && !durable ? 1u : 0u);
  EncDirHeader(e, b.hdr_);
  clio::run::u32 n = 0;
  for (const auto &kv : b.ents_) {
    if (!durable || kv.second.state_ != kDirEntPending) ++n;
  }
  e.U32(n);
  for (const auto &kv : b.ents_) {
    if (durable && kv.second.state_ == kDirEntPending) continue;
    e.Str(kv.first);
    e.U64(kv.second.id_);
    e.U32(kv.second.type_);
    e.U32(durable ? kDirEntLive : kv.second.state_);
  }
  return out;
}

/**
 * Decode a block.
 * @param data bytes
 * @param len length
 * @param b receives the block
 * @return false if malformed
 */
inline bool DecodeDirBlock(const char *data, size_t len, DirBlock *b) {
  FsDec d(data, len);
  clio::run::u32 magic = 0, sealed = 0, n = 0;
  if (!d.U32(&magic) || magic != kDirBlockMagic || !d.U64(&b->dir_) ||
      !d.U32(&b->index_) || !d.U32(&b->born_) || !d.U32(&b->depth_) ||
      !d.U64(&b->version_) || !d.U64(&b->mtime_) || !d.U32(&sealed) ||
      !DecDirHeader(d, &b->hdr_) || !d.U32(&n)) {
    return false;
  }
  b->sealed_ = sealed != 0;
  b->ents_.clear();
  for (clio::run::u32 i = 0; i < n; ++i) {
    std::string leaf;
    DirEntry ent;
    if (!d.Str(&leaf) || !d.U64(&ent.id_) || !d.U32(&ent.type_) ||
        !d.U32(&ent.state_)) {
      return false;
    }
    b->ents_[leaf] = ent;
  }
  return b->depth_ >= b->born_ && b->depth_ <= kDirMaxDepth;
}

// ---------------------------------------------------------------------------
// Deltas: what a home pushes to the nodes caching a block
// ---------------------------------------------------------------------------

/** Kinds of delta. */
enum class DirDeltaKind : clio::run::u32 {
  kOps = 1,       ///< entry changes on top of base_version_
  kSnapshot = 2,  ///< the whole block (after a split, or to resync)
  kDrop = 3,      ///< the block is gone (rmdir): forget it
};

/** One entry change inside a kOps delta (absent entry = removal). */
struct DirDeltaOp {
  std::string leaf_;
  bool present_ = true;  ///< false: the entry was removed
  DirEntry ent_;
};

/** A change to one block, as pushed to its holders. */
struct DirDelta {
  DirDeltaKind kind_ = DirDeltaKind::kOps;
  clio::run::u64 dir_ = 0;
  clio::run::u32 index_ = 0;
  clio::run::u64 base_version_ = 0;  ///< holder must be at this version
  clio::run::u64 new_version_ = 0;
  clio::run::u32 depth_ = 0;         ///< the block's depth after the change
  clio::run::u64 mtime_ = 0;         ///< the block's mtime after the change
  bool sealed_ = false;
  bool has_hdr_ = false;             ///< header replaced
  DirHeader hdr_;
  std::vector<DirDeltaOp> ops_;
  std::string snapshot_;             ///< kSnapshot: EncodeDirBlock(.., false)
};

/**
 * Encode deltas for one holder (a batch).
 * @param deltas the deltas
 * @return bytes
 */
inline std::string EncodeDirDeltas(const std::vector<DirDelta> &deltas) {
  std::string out;
  FsEnc e(&out);
  e.U32(static_cast<clio::run::u32>(deltas.size()));
  for (const DirDelta &x : deltas) {
    e.U32(static_cast<clio::run::u32>(x.kind_));
    e.U64(x.dir_);
    e.U32(x.index_);
    e.U64(x.base_version_);
    e.U64(x.new_version_);
    e.U32(x.depth_);
    e.U64(x.mtime_);
    e.U32((x.sealed_ ? 1u : 0u) | (x.has_hdr_ ? 2u : 0u));
    if (x.has_hdr_) EncDirHeader(e, x.hdr_);
    e.U32(static_cast<clio::run::u32>(x.ops_.size()));
    for (const DirDeltaOp &op : x.ops_) {
      e.Str(op.leaf_);
      e.U32(op.present_ ? 1u : 0u);
      e.U64(op.ent_.id_);
      e.U32(op.ent_.type_);
      e.U32(op.ent_.state_);
    }
    e.Str(x.snapshot_);
  }
  return out;
}

/**
 * Decode a delta batch.
 * @param s bytes
 * @param out receives the deltas
 * @return false if malformed
 */
inline bool DecodeDirDeltas(const std::string &s, std::vector<DirDelta> *out) {
  FsDec d(s.data(), s.size());
  clio::run::u32 n = 0;
  if (!d.U32(&n)) return false;
  out->clear();
  for (clio::run::u32 i = 0; i < n; ++i) {
    DirDelta x;
    clio::run::u32 kind = 0, flags = 0, nops = 0;
    if (!d.U32(&kind) || !d.U64(&x.dir_) || !d.U32(&x.index_) ||
        !d.U64(&x.base_version_) || !d.U64(&x.new_version_) ||
        !d.U32(&x.depth_) || !d.U64(&x.mtime_) || !d.U32(&flags)) {
      return false;
    }
    x.kind_ = static_cast<DirDeltaKind>(kind);
    x.sealed_ = (flags & 1u) != 0;
    x.has_hdr_ = (flags & 2u) != 0;
    if (x.has_hdr_ && !DecDirHeader(d, &x.hdr_)) return false;
    if (!d.U32(&nops)) return false;
    for (clio::run::u32 j = 0; j < nops; ++j) {
      DirDeltaOp op;
      clio::run::u32 present = 0;
      if (!d.Str(&op.leaf_) || !d.U32(&present) || !d.U64(&op.ent_.id_) ||
          !d.U32(&op.ent_.type_) || !d.U32(&op.ent_.state_)) {
        return false;
      }
      op.present_ = present != 0;
      x.ops_.push_back(std::move(op));
    }
    if (!d.Str(&x.snapshot_)) return false;
    out->push_back(std::move(x));
  }
  return d.Done();
}

/**
 * Apply a kOps delta to a cached block.
 * @param b cached block
 * @param x delta (kOps)
 * @return false if `b` is not at x.base_version_ (the copy must be dropped
 *         and reloaded)
 */
inline bool ApplyDirOps(DirBlock *b, const DirDelta &x) {
  if (b->version_ != x.base_version_) return false;
  for (const DirDeltaOp &op : x.ops_) {
    if (op.present_) {
      b->ents_[op.leaf_] = op.ent_;
    } else {
      b->ents_.erase(op.leaf_);
    }
  }
  b->sealed_ = x.sealed_;
  b->depth_ = x.depth_;
  b->mtime_ = x.mtime_;
  if (x.has_hdr_) b->hdr_ = x.hdr_;
  b->version_ = x.new_version_;
  return true;
}

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FS_DIR_BLOCK_H_
