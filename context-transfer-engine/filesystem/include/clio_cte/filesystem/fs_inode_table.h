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

#include <algorithm>
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
// Directory offsets: where a name sits in a listing
//
// A listing is ordered by an offset derived from the name alone, so the
// offset of an entry is the same in every process and across opens, and a
// reader can seek to any offset (telldir/seekdir, or a walker handing a
// range of one directory to another process).
//
// The offset is the bit-reversed TableHash, shifted into 63 bits. A block
// holds the keys whose low `depth_` hash bits equal its index; reversed,
// those are the TOP bits, so every block covers one contiguous offset
// interval and the blocks of a directory tile the offset space. A listing in
// offset order therefore reads whole blocks, one after another.
// ---------------------------------------------------------------------------

/** Offset of the first real entry: 0 is "start", 1 and 2 follow "." and
 *  "..". */
GLOBAL_CROSS_CONST clio::run::u64 kDirOffFirst = 3;
/** Offset meaning "end of directory" (INT64_MAX: the largest off_t). */
GLOBAL_CROSS_CONST clio::run::u64 kDirOffEnd = 0x7FFFFFFFFFFFFFFFULL;

/**
 * Reverse the bits of a 64-bit value (bit 0 becomes bit 63).
 * @param x value
 * @return x with its bit order reversed
 */
inline clio::run::u64 BitReverse64(clio::run::u64 x) {
  x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
  x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
  x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
  x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x & 0x00FF00FF00FF00FFULL) << 8);
  x = ((x >> 16) & 0x0000FFFF0000FFFFULL) |
      ((x & 0x0000FFFF0000FFFFULL) << 16);
  return (x >> 32) | (x << 32);
}

/**
 * Listing offset of a key with hash `h`: the reversed hash in 63 bits, kept
 * clear of the reserved offsets at both ends.
 * @param h TableHash of the canonical name
 * @return offset in [kDirOffFirst, kDirOffEnd)
 */
inline clio::run::u64 DirOffOfHash(clio::run::u64 h) {
  const clio::run::u64 off = BitReverse64(h) >> 1;
  if (off < kDirOffFirst) return kDirOffFirst;
  if (off >= kDirOffEnd) return kDirOffEnd - 1;
  return off;
}

/**
 * Listing offset of a canonical name.
 * @param key canonical name (the row's key, not its spelling)
 * @return offset in [kDirOffFirst, kDirOffEnd)
 */
inline clio::run::u64 DirEntryOff(const std::string &key) {
  return DirOffOfHash(TableHash(key));
}

/**
 * The offset interval [lo, hi) of the keys whose low `depth` hash bits equal
 * `index`. With a block's depth_ it is the interval of the block's own rows;
 * with its born_ it is the interval of the block and everything it split off.
 * Every offset of such a key lies in the interval (the clamping in
 * DirOffOfHash moves an offset by at most 2, never across an interval edge:
 * intervals are at least 2^39 wide).
 * @param index block index (bits at and above `depth` are zero)
 * @param depth number of hash bits that select the interval (<= 24)
 * @param lo receives the first offset of the interval
 * @param hi receives one past the last offset (up to 2^63)
 */
inline void DirOffRange(clio::run::u32 index, clio::run::u32 depth,
                        clio::run::u64 *lo, clio::run::u64 *hi) {
  *lo = BitReverse64(index) >> 1;
  *hi = *lo + (1ULL << (63 - depth));
}

/** A block and everything it split off, named without loading it. */
struct TableSubtree {
  clio::run::u32 index_ = 0;  ///< the subtree's root block
  clio::run::u32 born_ = 0;   ///< depth that block was created at
};

/**
 * The subtrees `b` split off, in listing (offset) order. All of them follow
 * b's own rows: b keeps the keys whose hash bits [born_, depth_) are zero,
 * the child split at bit j holds those with bit j set and the bits below it
 * zero, and after the reversal a lower bit is more significant -- so the
 * deepest split comes first.
 * @param b block
 * @return child subtrees, ascending by offset
 */
inline std::vector<TableSubtree> ChildSubtreesByOffset(
    const InodeTableBlock &b) {
  std::vector<TableSubtree> out;
  for (clio::run::u32 d = b.depth_; d > b.born_; --d) {
    out.push_back(TableSubtree{b.index_ + (1u << (d - 1)), d});
  }
  return out;
}

/** One entry of a ranged listing. */
struct DirRangeEntry {
  clio::run::u64 off_ = 0;   ///< DirEntryOff of the entry's key
  std::string name_;         ///< the name as listed (RowName)
  clio::run::u64 id_ = 0;    ///< packed id of the inode it names
  clio::run::u32 type_ = 0;  ///< kFsType*
};

/**
 * Put entries in listing order: ascending offset, then name (names with the
 * same offset collided in the hash and always share a block).
 * @param v entries to sort in place
 */
inline void SortDirRange(std::vector<DirRangeEntry> *v) {
  std::sort(v->begin(), v->end(),
            [](const DirRangeEntry &a, const DirRangeEntry &b) {
              return a.off_ != b.off_ ? a.off_ < b.off_ : a.name_ < b.name_;
            });
}

/**
 * One chunk of a directory listing in offset order, built block by block.
 *
 * The caller asks Next() which block to load, loads it and hands it to
 * Take(), until Next() returns false; Finish() then sorts the chunk. Blocks
 * whose interval ends at or before the start offset are never asked for.
 * The chunk takes whole blocks until it holds at least `max_entries`
 * entries, then keeps going only to find the offset of the first entry after
 * it (blocks with no visible row are passed over), so the next chunk can
 * start exactly there.
 *
 * Pending rows are hidden. A leaving row is listed: a rename caught mid-way
 * can show a file under both names, or -- unlike a whole-directory snapshot
 * -- under neither, when its new name falls in a block read before the
 * rename and its old name in one read after. Telling the two apart needs
 * every block at once, which a ranged read does not have.
 */
class DirRangeScan {
 public:
  /**
   * @param start_off first offset wanted (below kDirOffFirst: from the start)
   * @param max_entries entries after which the chunk stops taking blocks
   */
  DirRangeScan(clio::run::u64 start_off, size_t max_entries)
      : start_off_(start_off), max_entries_(max_entries) {
    todo_.push_back(TableSubtree{0, 0});
  }

  /**
   * The next block to load.
   * @param sub receives the block (and the depth it must have been born at)
   * @return false when the chunk is complete
   */
  bool Next(TableSubtree *sub) {
    while (!done_ && !todo_.empty()) {
      *sub = todo_.back();
      todo_.pop_back();
      clio::run::u64 lo = 0, hi = 0;
      DirOffRange(sub->index_, sub->born_, &lo, &hi);
      if (hi > start_off_) return true;  // else: wholly before the start
    }
    return false;
  }

  /**
   * Account for the block Next() asked for.
   * @param sub what Next() returned
   * @param b the loaded block
   * @return false if the block is not the one its parent says it split off
   *         (a corrupt split tree)
   */
  bool Take(const TableSubtree &sub, const InodeTableBlock &b) {
    if (b.index_ != sub.index_ || b.born_ != sub.born_) return false;
    // Depth first, nearest offset on top of the stack.
    const std::vector<TableSubtree> kids = ChildSubtreesByOffset(b);
    for (auto it = kids.rbegin(); it != kids.rend(); ++it) {
      todo_.push_back(*it);
    }
    clio::run::u64 lo = 0, hi = 0;
    DirOffRange(b.index_, b.depth_, &lo, &hi);
    if (hi <= start_off_) return true;  // its own rows end before the start
    const bool full = entries_.size() >= max_entries_;
    for (const auto &kv : b.rows_) {
      if (kv.second.state_ == kRowPending) continue;
      const clio::run::u64 off = DirEntryOff(kv.first);
      if (off < start_off_) continue;
      if (full) {
        // Only where the next chunk starts is still wanted.
        next_off_ = std::min(next_off_, off);
        done_ = true;
      } else {
        entries_.push_back(DirRangeEntry{off, RowName(kv.first, kv.second),
                                         kv.second.id_, kv.second.type_});
      }
    }
    return true;
  }

  /** Sort the chunk into listing order (call once, after the last Take). */
  void Finish() { SortDirRange(&entries_); }

  /** @return the chunk's entries (in listing order after Finish) */
  std::vector<DirRangeEntry> &entries() { return entries_; }

  /** @return offset of the first entry after the chunk; kDirOffEnd when the
   *          chunk reaches the end of the directory */
  clio::run::u64 next_off() const { return next_off_; }

 private:
  clio::run::u64 start_off_;
  size_t max_entries_;
  std::vector<TableSubtree> todo_;       ///< subtrees not visited, nearest last
  std::vector<DirRangeEntry> entries_;
  clio::run::u64 next_off_ = kDirOffEnd;
  bool done_ = false;                    ///< next_off_ is known
};

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

/** How a logged delta fared against the block replay has built so far. */
enum class ReplayStep {
  kSkipped,   ///< not a row change newer than the block (already folded in)
  kApplied,   ///< applied; the block advanced to x.new_version_
  kMismatch,  ///< the block is not at x.base_version_: the log ends here
};

/**
 * Replay one delta read back from a log (the directory journal or a block
 * blob) onto the block it builds on. A delta not newer than the block is one
 * a compaction's image already holds, so it is skipped.
 * @param b the block as replay has it
 * @param x the delta
 * @return what happened to x
 */
inline ReplayStep ReplayTableDelta(InodeTableBlock *b, const TableDelta &x) {
  if (x.kind_ != TableDeltaKind::kOps || x.new_version_ <= b->version_) {
    return ReplayStep::kSkipped;
  }
  return ApplyTableOps(b, x) ? ReplayStep::kApplied : ReplayStep::kMismatch;
}

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FS_INODE_TABLE_H_
