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
 * The inode table: how clio-fs maps names to inodes.
 *
 * The table is one logical map keyed by (parent directory id, canonical
 * name). Its value, an InodeRow, is one NAME of an inode: the inode's id,
 * its type, and the spelling the name was created with. A hard link is two
 * rows holding the same id. The table holds names only -- every attribute of
 * an inode, a directory's included, lives in the inode's record (see
 * filesystem_runtime.h).
 *
 * The canonical name of a spelling is the spelling itself on a case-sensitive
 * volume and its case fold on a case-insensitive one (FsCanonicalName). Every
 * spelling of a name therefore has one key, one block and one home.
 *
 * Each directory's rows are partitioned into blocks, one CTE blob each in the
 * directory's tag, so the CTE persists, replicates and fails them over like
 * any other blob. Extendible hashing over the low bits of TableHash(key):
 *   - Block k holds the keys whose low `born_` bits equal k.
 *   - Block 0 is born at depth 0 and owns every key at first.
 *   - A full block at depth d splits: the keys with bit d set move to the new
 *     block k + 2^d (born at depth d+1); the old block's depth becomes d+1.
 *     Blocks never merge.
 *   - A block's children are therefore k + 2^d for d in [born_, depth_), and
 *     a key's block is found by walking down from block 0 (NextTableBlock):
 *     no bitmap, no directory of blocks, nothing to keep in sync.
 *
 * Every block has one home (the container that owns its CTE blob). The home
 * is the only writer; every other node holding a cached copy receives each
 * change as a TableDelta before the change is acknowledged (see
 * fs_inode_table.cc).
 */
#ifndef CLIO_CTE_FILESYSTEM_FS_INODE_TABLE_H_
#define CLIO_CTE_FILESYSTEM_FS_INODE_TABLE_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/filesystem/fs_shard.h>

namespace clio::cte::filesystem {

/** Row states: pending is reserved but invisible, leaving is visible but
 *  frozen (only the operation that marked it may finish or undo it). */
GLOBAL_CROSS_CONST clio::run::u32 kRowLive = 0;
GLOBAL_CROSS_CONST clio::run::u32 kRowPending = 1;
GLOBAL_CROSS_CONST clio::run::u32 kRowLeaving = 2;

/** Deepest split a directory's table may reach (2^24 blocks). */
GLOBAL_CROSS_CONST clio::run::u32 kTableMaxDepth = 24;

/** One name of an inode. */
struct InodeRow {
  clio::run::u64 id_ = 0;              ///< packed id of the inode it names
  clio::run::u32 type_ = kFsTypeFile;  ///< kFsType* (the inode's; immutable)
  std::string spelling_;               ///< name as created; empty = the key
  clio::run::u32 state_ = kRowLive;    ///< kRow*
};

/**
 * The name a row is listed under: its spelling, or its key when the spelling
 * is the key itself.
 * @param key the row's canonical name
 * @param row the row
 * @return the name readdir reports
 */
inline const std::string &RowName(const std::string &key, const InodeRow &row) {
  return row.spelling_.empty() ? key : row.spelling_;
}

/**
 * The spelling to store in a row: empty when it equals the key, so the
 * common case (a canonical spelling) costs nothing.
 * @param key canonical name
 * @param spelling name as given by the caller
 * @return the value for InodeRow::spelling_
 */
inline std::string RowSpelling(const std::string &key,
                               const std::string &spelling) {
  return spelling == key ? std::string() : spelling;
}

/** One partition of one directory's rows (see the file comment). */
struct InodeTableBlock {
  clio::run::u64 dir_ = 0;      ///< packed id of the directory
  clio::run::u32 index_ = 0;    ///< k
  clio::run::u32 born_ = 0;     ///< depth the block was created at
  clio::run::u32 depth_ = 0;    ///< current depth (born_ + splits)
  clio::run::u64 version_ = 0;  ///< bumped by every change
  clio::run::u64 mtime_ = 0;    ///< ns of this block's last row change
  bool sealed_ = false;         ///< an rmdir holds it: inserts fail ENOENT
  std::map<std::string, InodeRow> rows_;  ///< canonical name -> row
};

/**
 * Blob name of block k in the directory's tag. The "d." prefix also decides
 * the block's placement (BlobHash of the name), so it is part of the format.
 * @param k block index
 * @return blob name
 */
inline std::string TableBlockName(clio::run::u32 k) {
  return "d." + std::to_string(k);
}

/**
 * 64-bit hash of a canonical name (FNV-1a, then mixed): its low bits pick the
 * block. Identical in every process and build.
 * @param key canonical name
 * @return hash
 */
inline clio::run::u64 TableHash(const std::string &key) {
  clio::run::u64 h = 0xCBF29CE484222325ULL;
  for (unsigned char c : key) {
    h ^= c;
    h *= 0x100000001B3ULL;
  }
  return FsMix64(h);
}

/**
 * The block a key belongs to, one step down from `b`: `b` itself, or the
 * child the key moved to when `b` split.
 * @param b a block whose range contains the key
 * @param h TableHash of the key
 * @return b.index_ if `b` holds the key, else the child to look in next
 */
inline clio::run::u32 NextTableBlock(const InodeTableBlock &b,
                                     clio::run::u64 h) {
  for (clio::run::u32 d = b.born_; d < b.depth_; ++d) {
    if ((h >> d) & 1ULL) return b.index_ + (1u << d);
  }
  return b.index_;
}

/**
 * Whether `b` is the block that holds keys hashing to `h` (it did not split
 * them away, and they fall in its range).
 * @param b block
 * @param h TableHash of the key
 * @return true when `b` owns the key
 */
inline bool BlockOwnsKey(const InodeTableBlock &b, clio::run::u64 h) {
  const clio::run::u64 mask =
      b.born_ >= 64 ? ~0ULL : ((1ULL << b.born_) - 1ULL);
  return (h & mask) == b.index_ && NextTableBlock(b, h) == b.index_;
}

/**
 * The blocks `b` split off.
 * @param b block
 * @return child block indices
 */
inline std::vector<clio::run::u32> ChildBlocks(const InodeTableBlock &b) {
  std::vector<clio::run::u32> out;
  for (clio::run::u32 d = b.born_; d < b.depth_; ++d) {
    out.push_back(b.index_ + (1u << d));
  }
  return out;
}

/**
 * Rows a listing would show (pending reservations excluded).
 * @param b block
 * @return count
 */
inline size_t TableBlockLiveCount(const InodeTableBlock &b) {
  size_t n = 0;
  for (const auto &kv : b.rows_) {
    if (kv.second.state_ != kRowPending) ++n;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

/** Format tag of an encoded block ("CIT1"). Blocks of the earlier directory
 *  format ("CDB1") are refused: the volume must be recreated. */
GLOBAL_CROSS_CONST clio::run::u32 kTableBlockMagic = 0x31544943u;
/** Format tag of the earlier directory-block format, recognized only to say
 *  why a block is refused. */
GLOBAL_CROSS_CONST clio::run::u32 kOldDirBlockMagic = 0x31424443u;

/**
 * Append one row to an encoding.
 * @param e encoder
 * @param key canonical name
 * @param row the row
 * @param state the state to record
 */
inline void EncRow(FsEnc &e, const std::string &key, const InodeRow &row,
                   clio::run::u32 state) {
  e.Str(key);
  e.U64(row.id_);
  e.U32(row.type_);
  e.Str(row.spelling_);
  e.U32(state);
}

/**
 * Read one row.
 * @param d decoder
 * @param key receives the canonical name
 * @param row receives the row
 * @return false if malformed
 */
inline bool DecRow(FsDec &d, std::string *key, InodeRow *row) {
  return d.Str(key) && d.U64(&row->id_) && d.U32(&row->type_) &&
         d.Str(&row->spelling_) && d.U32(&row->state_);
}

/**
 * Encode a block.
 * @param b the block
 * @param durable true for the persisted image: pending rows are dropped,
 *        leaving rows are recorded live and the seal is not recorded (a
 *        crash aborts the operations that set them)
 * @return the encoding
 */
inline std::string EncodeTableBlock(const InodeTableBlock &b, bool durable) {
  std::string out;
  FsEnc e(&out);
  e.U32(kTableBlockMagic);
  e.U64(b.dir_);
  e.U32(b.index_);
  e.U32(b.born_);
  e.U32(b.depth_);
  e.U64(b.version_);
  e.U64(b.mtime_);
  e.U32(b.sealed_ && !durable ? 1u : 0u);
  clio::run::u32 n = 0;
  for (const auto &kv : b.rows_) {
    if (!durable || kv.second.state_ != kRowPending) ++n;
  }
  e.U32(n);
  for (const auto &kv : b.rows_) {
    if (durable && kv.second.state_ == kRowPending) continue;
    EncRow(e, kv.first, kv.second, durable ? kRowLive : kv.second.state_);
  }
  return out;
}

/**
 * Decode a block. Trailing bytes (blob padding) are ignored.
 * @param data bytes
 * @param len length
 * @param b receives the block
 * @return false if malformed or of another format
 */
inline bool DecodeTableBlock(const char *data, size_t len, InodeTableBlock *b) {
  FsDec d(data, len);
  clio::run::u32 magic = 0, sealed = 0, n = 0;
  if (!d.U32(&magic) || magic != kTableBlockMagic || !d.U64(&b->dir_) ||
      !d.U32(&b->index_) || !d.U32(&b->born_) || !d.U32(&b->depth_) ||
      !d.U64(&b->version_) || !d.U64(&b->mtime_) || !d.U32(&sealed) ||
      !d.U32(&n)) {
    return false;
  }
  b->sealed_ = sealed != 0;
  b->rows_.clear();
  for (clio::run::u32 i = 0; i < n; ++i) {
    std::string key;
    InodeRow row;
    if (!DecRow(d, &key, &row)) return false;
    b->rows_[key] = std::move(row);
  }
  return b->depth_ >= b->born_ && b->depth_ <= kTableMaxDepth;
}

/**
 * Whether encoded bytes are a block of the earlier directory format.
 * @param data bytes
 * @param len length
 * @return true for a "CDB1" block
 */
inline bool IsOldDirBlock(const char *data, size_t len) {
  FsDec d(data, len);
  clio::run::u32 magic = 0;
  return d.U32(&magic) && magic == kOldDirBlockMagic;
}

// ---------------------------------------------------------------------------
// Deltas: what a home pushes to the nodes caching a block
// ---------------------------------------------------------------------------

enum class TableDeltaKind : clio::run::u32 {
  kOps = 1,       ///< row changes on top of base_version_
  kSnapshot = 2,  ///< the whole block (after a split, or to resync)
  kDrop = 3,      ///< the block is gone (rmdir): forget it
};

/** One row change. */
struct TableDeltaOp {
  std::string key_;      ///< canonical name
  bool present_ = true;  ///< false: the row was removed
  InodeRow row_;
};

/** A change to one block. */
struct TableDelta {
  TableDeltaKind kind_ = TableDeltaKind::kOps;
  clio::run::u64 dir_ = 0;
  clio::run::u32 index_ = 0;
  clio::run::u64 base_version_ = 0;  ///< holder must be at this version
  clio::run::u64 new_version_ = 0;
  clio::run::u32 depth_ = 0;         ///< the block's depth after the change
  clio::run::u64 mtime_ = 0;         ///< the block's mtime after the change
  bool sealed_ = false;
  std::vector<TableDeltaOp> ops_;
  std::string snapshot_;             ///< kSnapshot: EncodeTableBlock(.., false)
};

/**
 * Encode a delta batch.
 * @param deltas the deltas
 * @return bytes
 */
inline std::string EncodeTableDeltas(const std::vector<TableDelta> &deltas) {
  std::string out;
  FsEnc e(&out);
  e.U32(static_cast<clio::run::u32>(deltas.size()));
  for (const TableDelta &x : deltas) {
    e.U32(static_cast<clio::run::u32>(x.kind_));
    e.U64(x.dir_);
    e.U32(x.index_);
    e.U64(x.base_version_);
    e.U64(x.new_version_);
    e.U32(x.depth_);
    e.U64(x.mtime_);
    e.U32(x.sealed_ ? 1u : 0u);
    e.U32(static_cast<clio::run::u32>(x.ops_.size()));
    for (const TableDeltaOp &op : x.ops_) {
      e.U32(op.present_ ? 1u : 0u);
      EncRow(e, op.key_, op.row_, op.row_.state_);
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
inline bool DecodeTableDeltas(const std::string &s,
                              std::vector<TableDelta> *out) {
  FsDec d(s.data(), s.size());
  clio::run::u32 n = 0;
  if (!d.U32(&n)) return false;
  out->clear();
  for (clio::run::u32 i = 0; i < n; ++i) {
    TableDelta x;
    clio::run::u32 kind = 0, sealed = 0, nops = 0;
    if (!d.U32(&kind) || !d.U64(&x.dir_) || !d.U32(&x.index_) ||
        !d.U64(&x.base_version_) || !d.U64(&x.new_version_) ||
        !d.U32(&x.depth_) || !d.U64(&x.mtime_) || !d.U32(&sealed) ||
        !d.U32(&nops)) {
      return false;
    }
    x.kind_ = static_cast<TableDeltaKind>(kind);
    x.sealed_ = sealed != 0;
    for (clio::run::u32 j = 0; j < nops; ++j) {
      TableDeltaOp op;
      clio::run::u32 present = 0;
      if (!d.U32(&present) || !DecRow(d, &op.key_, &op.row_)) return false;
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
inline bool ApplyTableOps(InodeTableBlock *b, const TableDelta &x) {
  if (b->version_ != x.base_version_) return false;
  for (const TableDeltaOp &op : x.ops_) {
    if (op.present_) {
      b->rows_[op.key_] = op.row_;
    } else {
      b->rows_.erase(op.key_);
    }
  }
  b->sealed_ = x.sealed_;
  b->depth_ = x.depth_;
  b->mtime_ = x.mtime_;
  b->version_ = x.new_version_;
  return true;
}

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FS_INODE_TABLE_H_
