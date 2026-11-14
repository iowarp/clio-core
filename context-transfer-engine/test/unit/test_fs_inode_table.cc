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
 * The inode table (fs_inode_table.h), inode records (fs_inode.h) and
 * canonical names (fs_canonical.h), exercised without a runtime: the codecs,
 * delta application, the split tree, and case folding. The split test grows
 * a directory the way the homes do -- insert, and split any block past a
 * threshold -- then checks that every name is reachable by walking from
 * block 0 and that each lives in exactly one block.
 */
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../../context-runtime/test/simple_test.h"
#include <clio_cte/filesystem/fs_canonical.h>
#include <clio_cte/filesystem/fs_inode.h>
#include <clio_cte/filesystem/fs_inode_table.h>

using namespace clio::cte::filesystem;

namespace {

/** A whole directory as a map index -> block (what the homes hold). */
using Blocks = std::map<clio::run::u32, InodeTableBlock>;

/**
 * Walk from block 0 to the block holding `key` (NextTableBlock).
 * @param blocks the directory
 * @param key canonical name
 * @return the block index, ~0u if the walk does not end
 */
clio::run::u32 Walk(const Blocks &blocks, const std::string &key) {
  const clio::run::u64 h = TableHash(key);
  clio::run::u32 k = 0;
  for (int step = 0; step < 64; ++step) {
    const clio::run::u32 next = NextTableBlock(blocks.at(k), h);
    if (next == k) return k;
    k = next;
  }
  return ~0u;
}

/**
 * Split block k the way SplitBlock does.
 * @param blocks the directory
 * @param k block to split
 */
void Split(Blocks &blocks, clio::run::u32 k) {
  InodeTableBlock &b = blocks.at(k);
  const clio::run::u32 d = b.depth_;
  InodeTableBlock child;
  child.dir_ = b.dir_;
  child.index_ = k + (1u << d);
  child.born_ = child.depth_ = d + 1;
  for (auto it = b.rows_.begin(); it != b.rows_.end();) {
    if ((TableHash(it->first) >> d) & 1ULL) {
      child.rows_[it->first] = it->second;
      it = b.rows_.erase(it);
    } else {
      ++it;
    }
  }
  b.depth_ = d + 1;
  blocks[child.index_] = child;
}

/**
 * Insert like a home: canonicalize, walk, insert, split past `limit`.
 * @param blocks the directory
 * @param name the name as spelled
 * @param ci case-insensitive volume
 * @param limit split threshold
 */
void Insert(Blocks &blocks, const std::string &name, bool ci, size_t limit) {
  const std::string key = FsCanonicalName(name, ci);
  const clio::run::u32 k = Walk(blocks, key);
  InodeRow e;
  e.id_ = TableHash(key) | 1;
  e.spelling_ = RowSpelling(key, name);
  blocks.at(k).rows_[key] = e;
  if (blocks.at(k).rows_.size() > limit) Split(blocks, k);
}

}  // namespace

TEST_CASE("inode_table_codec_roundtrip", "[fs][inodetable]") {
  InodeTableBlock b;
  b.dir_ = 0x4000000100000007ULL;
  b.index_ = 5;
  b.born_ = 2;
  b.depth_ = 4;
  b.version_ = 42;
  b.mtime_ = 123456789;
  b.sealed_ = true;
  b.rows_["a"] = InodeRow{11, kFsTypeFile, "", kRowLive};
  b.rows_["readme.md"] = InodeRow{14, kFsTypeFile, "README.md", kRowLive};
  b.rows_["pending"] = InodeRow{12, kFsTypeDir, "", kRowPending};
  b.rows_["leaving"] = InodeRow{13, kFsTypeSymlink, "Leaving", kRowLeaving};

  // A cache transfer keeps every state and spelling.
  InodeTableBlock c;
  const std::string xfer = EncodeTableBlock(b, false);
  REQUIRE(DecodeTableBlock(xfer.data(), xfer.size(), &c));
  REQUIRE(c.dir_ == b.dir_);
  REQUIRE(c.index_ == 5);
  REQUIRE(c.born_ == 2);
  REQUIRE(c.depth_ == 4);
  REQUIRE(c.version_ == 42);
  REQUIRE(c.mtime_ == 123456789);
  REQUIRE(c.sealed_);
  REQUIRE(c.rows_.size() == 4);
  REQUIRE(c.rows_["leaving"].state_ == kRowLeaving);
  REQUIRE(c.rows_["readme.md"].spelling_ == "README.md");
  REQUIRE(RowName("readme.md", c.rows_["readme.md"]) == "README.md");
  REQUIRE(RowName("a", c.rows_["a"]) == "a");

  // The durable image drops in-flight states: pending gone, leaving live,
  // never sealed.
  InodeTableBlock d;
  const std::string img = EncodeTableBlock(b, true);
  REQUIRE(DecodeTableBlock(img.data(), img.size(), &d));
  REQUIRE(d.rows_.size() == 3);
  REQUIRE(d.rows_.count("pending") == 0);
  REQUIRE(d.rows_["leaving"].state_ == kRowLive);
  REQUIRE(d.rows_["leaving"].spelling_ == "Leaving");
  REQUIRE_FALSE(d.sealed_);

  // A blob that shrank keeps stale bytes past its end: they are ignored.
  std::string padded = img + std::string(64, 'x');
  InodeTableBlock e;
  REQUIRE(DecodeTableBlock(padded.data(), padded.size(), &e));
  REQUIRE(e.rows_.size() == 3);

  // Garbage is refused.
  std::string bad = img;
  bad[0] ^= 0x55;
  REQUIRE_FALSE(DecodeTableBlock(bad.data(), bad.size(), &e));
  REQUIRE_FALSE(IsOldDirBlock(bad.data(), bad.size()));

  // A block of the earlier directory format is refused, and recognized.
  std::string old;
  FsEnc oe(&old);
  oe.U32(kOldDirBlockMagic);
  oe.U64(7);
  REQUIRE_FALSE(DecodeTableBlock(old.data(), old.size(), &e));
  REQUIRE(IsOldDirBlock(old.data(), old.size()));
}

TEST_CASE("inode_table_row_spelling", "[fs][inodetable]") {
  // The spelling is stored only when it is not the key itself.
  REQUIRE(RowSpelling("readme.md", "readme.md").empty());
  REQUIRE(RowSpelling("readme.md", "README.md") == "README.md");
}

TEST_CASE("inode_table_delta_apply", "[fs][inodetable]") {
  InodeTableBlock home;
  home.dir_ = 7;
  home.version_ = 10;
  InodeTableBlock holder = home;

  // Three changes, pushed as one batch of per-change deltas.
  std::vector<TableDelta> batch;
  for (int i = 0; i < 3; ++i) {
    TableDelta x;
    x.dir_ = 7;
    x.base_version_ = home.version_;
    x.new_version_ = ++home.version_;
    x.depth_ = home.depth_;
    x.mtime_ = 1000 + i;
    x.ops_.push_back({"f" + std::to_string(i), true,
                      InodeRow{100u + i, kFsTypeFile, "F" + std::to_string(i),
                               kRowLive}});
    home.rows_["f" + std::to_string(i)] = x.ops_.back().row_;
    batch.push_back(x);
  }
  // A removal and a split.
  {
    TableDelta x;
    x.dir_ = 7;
    x.base_version_ = home.version_;
    x.new_version_ = ++home.version_;
    x.depth_ = 1;
    x.mtime_ = 2000;
    x.ops_.push_back({"f1", false, InodeRow()});
    batch.push_back(x);
  }
  std::vector<TableDelta> got;
  REQUIRE(DecodeTableDeltas(EncodeTableDeltas(batch), &got));
  REQUIRE(got.size() == 4);
  for (const TableDelta &x : got) REQUIRE(ApplyTableOps(&holder, x));
  REQUIRE(holder.version_ == 14);
  REQUIRE(holder.rows_.size() == 2);
  REQUIRE(holder.rows_.count("f1") == 0);
  REQUIRE(holder.rows_["f2"].spelling_ == "F2");
  REQUIRE(holder.depth_ == 1);
  REQUIRE(holder.mtime_ == 2000);

  // A holder that missed a change refuses the next one (it must refetch).
  InodeTableBlock stale;
  stale.dir_ = 7;
  stale.version_ = 10;
  REQUIRE_FALSE(ApplyTableOps(&stale, got[1]));
}

TEST_CASE("inode_table_split_tree", "[fs][inodetable]") {
  for (bool ci : {false, true}) {
    Blocks blocks;
    blocks[0].dir_ = 1;
    const size_t kNames = 20000;
    const size_t kLimit = 64;
    std::vector<std::string> names;
    for (size_t i = 0; i < kNames; ++i) {
      names.push_back("File_" + std::to_string(i * 7919));
      Insert(blocks, names.back(), ci, kLimit);
    }
    // It split, and no block kept growing without bound.
    REQUIRE(blocks.size() > kNames / kLimit / 2);
    size_t total = 0;
    for (const auto &kv : blocks) {
      // Each block holds exactly the keys its range and depth say it does.
      for (const auto &e : kv.second.rows_) {
        REQUIRE(BlockOwnsKey(kv.second, TableHash(e.first)));
      }
      total += kv.second.rows_.size();
    }
    REQUIRE(total == kNames);
    // Every name is found by walking down from block 0 -- on a
    // case-insensitive volume under any spelling, listed as created.
    for (const std::string &n : names) {
      for (const std::string &asked : {n, FsFold(n), std::string("FILE_") +
                                                         n.substr(5)}) {
        const std::string key = FsCanonicalName(asked, ci);
        const clio::run::u32 k = Walk(blocks, key);
        REQUIRE(k != ~0u);
        const bool found = blocks.at(k).rows_.count(key) == 1;
        REQUIRE(found == (ci || asked == n));
        if (found) REQUIRE(RowName(key, blocks.at(k).rows_.at(key)) == n);
      }
    }
    // A missing name walks to a block that does not hold it.
    const clio::run::u32 k = Walk(blocks, "absent");
    REQUIRE(blocks.at(k).rows_.count("absent") == 0);
    // The split tree reaches every block from block 0 (what readdir and
    // rmdir traverse).
    std::set<clio::run::u32> seen;
    std::vector<clio::run::u32> todo{0};
    while (!todo.empty()) {
      const clio::run::u32 b = todo.back();
      todo.pop_back();
      REQUIRE(seen.insert(b).second);
      for (clio::run::u32 c : ChildBlocks(blocks.at(b))) todo.push_back(c);
    }
    REQUIRE(seen.size() == blocks.size());
  }
}

namespace {

/**
 * Grow a directory of `count` names, splitting past `limit`.
 * @param count number of names
 * @param limit split threshold
 * @return the directory's blocks
 */
Blocks GrowDir(size_t count, size_t limit) {
  Blocks blocks;
  blocks[0].dir_ = 1;
  for (size_t i = 0; i < count; ++i) {
    Insert(blocks, "f." + std::to_string(i * 2654435761ULL), false, limit);
  }
  return blocks;
}

/**
 * Every visible entry of a directory in listing order (the reference a
 * ranged read is checked against).
 * @param blocks the directory
 * @return entries sorted by (offset, name)
 */
std::vector<DirRangeEntry> ListAll(const Blocks &blocks) {
  std::vector<DirRangeEntry> all;
  for (const auto &kv : blocks) {
    for (const auto &e : kv.second.rows_) {
      if (e.second.state_ == kRowPending) continue;
      all.push_back(DirRangeEntry{DirEntryOff(e.first),
                                  RowName(e.first, e.second), e.second.id_,
                                  e.second.type_});
    }
  }
  SortDirRange(&all);
  return all;
}

/**
 * One ranged read, driven the way Runtime::CollectDirRange drives it.
 * @param blocks the directory
 * @param start first offset wanted
 * @param max entries after which the chunk stops taking blocks
 * @param next receives the offset the next chunk starts at
 * @param loaded receives the number of blocks the read loaded (may be null)
 * @return the chunk
 */
std::vector<DirRangeEntry> ReadRange(const Blocks &blocks,
                                     clio::run::u64 start, size_t max,
                                     clio::run::u64 *next,
                                     size_t *loaded = nullptr) {
  DirRangeScan scan(start, max);
  TableSubtree sub;
  size_t n = 0;
  while (scan.Next(&sub)) {
    ++n;
    REQUIRE(scan.Take(sub, blocks.at(sub.index_)));
  }
  scan.Finish();
  *next = scan.next_off();
  if (loaded != nullptr) *loaded = n;
  return scan.entries();
}

/**
 * Blocks of a directory in listing order: a block, then what it split off,
 * deepest split first.
 * @param blocks the directory
 * @param sub subtree to walk
 * @param out receives the block indices
 */
void WalkByOffset(const Blocks &blocks, const TableSubtree &sub,
                  std::vector<clio::run::u32> *out) {
  const InodeTableBlock &b = blocks.at(sub.index_);
  REQUIRE(b.born_ == sub.born_);
  out->push_back(sub.index_);
  for (const TableSubtree &c : ChildSubtreesByOffset(b)) {
    WalkByOffset(blocks, c, out);
  }
}

}  // namespace

TEST_CASE("dir_offset_mapping", "[fs][inodetable][diroff]") {
  REQUIRE(BitReverse64(0) == 0);
  REQUIRE(BitReverse64(1) == 0x8000000000000000ULL);
  REQUIRE(BitReverse64(0x8000000000000000ULL) == 1);
  REQUIRE(BitReverse64(0x00000000000000F0ULL) == 0x0F00000000000000ULL);
  for (clio::run::u64 x : {0x0123456789ABCDEFULL, 0xDEADBEEFCAFEF00DULL}) {
    REQUIRE(BitReverse64(BitReverse64(x)) == x);
  }
  // The reserved offsets at both ends are never an entry's.
  REQUIRE(kDirOffEnd == static_cast<clio::run::u64>(INT64_MAX));
  REQUIRE(DirOffOfHash(0) == kDirOffFirst);
  REQUIRE(DirOffOfHash(1ULL << 62) == kDirOffFirst);       // raw offset 1
  REQUIRE(DirOffOfHash(1ULL << 61) == kDirOffFirst);       // raw offset 2
  REQUIRE(DirOffOfHash(3ULL << 61) == kDirOffFirst);       // raw offset 3
  REQUIRE(DirOffOfHash(1ULL << 60) == 4);                  // raw offset 4
  REQUIRE(DirOffOfHash(~0ULL) == kDirOffEnd - 1);          // raw INT64_MAX
  REQUIRE(DirOffOfHash(~0ULL >> 1) == kDirOffEnd - 1);     // bit 63 is dropped
  REQUIRE(DirOffOfHash(~0ULL & ~(1ULL << 62)) == kDirOffEnd - 1);  // raw End-1
  REQUIRE(DirOffOfHash(~0ULL & ~(1ULL << 61)) == kDirOffEnd - 2);
  // Hash bit i lands on offset bit 62 - i: low hash bits order the listing.
  REQUIRE(DirOffOfHash(1) == 1ULL << 62);
  REQUIRE(DirOffOfHash(2) == 1ULL << 61);
  REQUIRE(DirEntryOff("name") == DirOffOfHash(TableHash("name")));

  // The interval of (index, depth): the keys whose low `depth` bits match.
  clio::run::u64 lo = 0, hi = 0;
  DirOffRange(0, 0, &lo, &hi);
  REQUIRE(lo == 0);
  REQUIRE(hi == 1ULL << 63);
  DirOffRange(1, 1, &lo, &hi);
  REQUIRE(lo == 1ULL << 62);
  REQUIRE(hi == 1ULL << 63);
  DirOffRange(0, 1, &lo, &hi);
  REQUIRE(lo == 0);
  REQUIRE(hi == 1ULL << 62);
  DirOffRange(2, 2, &lo, &hi);  // bits "10": reversed "01"
  REQUIRE(lo == 1ULL << 61);
  REQUIRE(hi == 1ULL << 62);
  DirOffRange((1u << kTableMaxDepth) - 1, kTableMaxDepth, &lo, &hi);
  REQUIRE(hi == 1ULL << 63);
  REQUIRE(hi - lo == 1ULL << (63 - kTableMaxDepth));
}

TEST_CASE("dir_offset_blocks_tile", "[fs][inodetable][diroff]") {
  const Blocks blocks = GrowDir(20000, 64);
  REQUIRE(blocks.size() > 100);
  // Every row's offset lies in its block's interval.
  for (const auto &kv : blocks) {
    clio::run::u64 lo = 0, hi = 0;
    DirOffRange(kv.second.index_, kv.second.depth_, &lo, &hi);
    for (const auto &e : kv.second.rows_) {
      const clio::run::u64 off = DirEntryOff(e.first);
      REQUIRE(off >= lo);
      REQUIRE(off < hi);
    }
  }
  // In listing order the blocks' intervals follow one another with no gap
  // and no overlap, from 0 to 2^63.
  std::vector<clio::run::u32> order;
  WalkByOffset(blocks, TableSubtree{0, 0}, &order);
  REQUIRE(order.size() == blocks.size());
  clio::run::u64 expect = 0;
  for (clio::run::u32 k : order) {
    clio::run::u64 lo = 0, hi = 0;
    DirOffRange(k, blocks.at(k).depth_, &lo, &hi);
    REQUIRE(lo == expect);
    expect = hi;
  }
  REQUIRE(expect == 1ULL << 63);
  // A subtree's interval is its block's plus everything it split off.
  for (const auto &kv : blocks) {
    clio::run::u64 lo = 0, hi = 0, slo = 0, shi = 0;
    DirOffRange(kv.second.index_, kv.second.depth_, &lo, &hi);
    DirOffRange(kv.second.index_, kv.second.born_, &slo, &shi);
    REQUIRE(slo == lo);
    clio::run::u64 end = hi;
    for (const TableSubtree &c : ChildSubtreesByOffset(kv.second)) {
      clio::run::u64 clo = 0, chi = 0;
      DirOffRange(c.index_, c.born_, &clo, &chi);
      REQUIRE(clo == end);
      end = chi;
    }
    REQUIRE(end == shi);
  }
}

TEST_CASE("dir_range_chunks", "[fs][inodetable][diroff]") {
  const Blocks blocks = GrowDir(20000, 64);
  const std::vector<DirRangeEntry> all = ListAll(blocks);
  REQUIRE(all.size() == 20000);
  // Chunk after chunk from the start: every entry once, in order, and each
  // chunk says exactly where the next one begins.
  for (size_t max : {size_t{1}, size_t{100}, size_t{8192}, size_t{50000}}) {
    size_t at = 0, chunks = 0;
    clio::run::u64 start = 0;
    while (start != kDirOffEnd) {
      clio::run::u64 next = 0;
      const std::vector<DirRangeEntry> c = ReadRange(blocks, start, max, &next);
      REQUIRE(!c.empty());
      // Whole blocks: at least `max` unless the directory ended, and never
      // more than one block over.
      REQUIRE((c.size() >= max || next == kDirOffEnd));
      REQUIRE(c.size() < max + 2 * 64);
      for (const DirRangeEntry &e : c) {
        REQUIRE(at < all.size());
        REQUIRE(e.off_ == all[at].off_);
        REQUIRE(e.name_ == all[at].name_);
        REQUIRE(e.id_ == all[at].id_);
        ++at;
      }
      REQUIRE(next > c.back().off_);
      REQUIRE(next == (at < all.size() ? all[at].off_ : kDirOffEnd));
      start = next;
      ++chunks;
    }
    REQUIRE(at == all.size());
    if (max >= all.size()) REQUIRE(chunks == 1);
  }
}

TEST_CASE("dir_range_seek", "[fs][inodetable][diroff]") {
  const Blocks blocks = GrowDir(20000, 64);
  const std::vector<DirRangeEntry> all = ListAll(blocks);
  // A read from any offset returns the entries at or after it, in order,
  // starting with the first such entry -- an entry's own offset, one past
  // it (inside a block), the reserved low offsets, and arbitrary cuts of
  // the offset space, as a walker splitting the directory would pick.
  std::vector<clio::run::u64> starts{0, 1, 2, 3, 4, kDirOffEnd - 1,
                                     kDirOffEnd};
  for (size_t i = 0; i < all.size(); i += 997) {
    starts.push_back(all[i].off_);
    starts.push_back(all[i].off_ + 1);
    starts.push_back(all[i].off_ - 1);
  }
  for (clio::run::u64 r = 1; r < 16; ++r) starts.push_back(r * (kDirOffEnd / 16));
  for (clio::run::u64 start : starts) {
    size_t at = 0;
    while (at < all.size() && all[at].off_ < start) ++at;
    clio::run::u64 next = 0;
    size_t loaded = 0;
    const std::vector<DirRangeEntry> c =
        ReadRange(blocks, start, 200, &next, &loaded);
    REQUIRE(c.size() <= all.size() - at);
    REQUIRE((c.size() >= 200 || next == kDirOffEnd));
    for (size_t i = 0; i < c.size(); ++i) {
      REQUIRE(c[i].off_ == all[at + i].off_);
      REQUIRE(c[i].name_ == all[at + i].name_);
    }
    const size_t after = at + c.size();
    REQUIRE(next == (after < all.size() ? all[after].off_ : kDirOffEnd));
    // Only the blocks from the start on are loaded, plus the path down to
    // them: far fewer than the directory holds.
    REQUIRE(loaded <= 200 / 16 + 2 * kTableMaxDepth);
  }
  // Ranges cut at arbitrary offsets partition the directory: what pfind's
  // ranks see when they split one directory by offset.
  size_t total = 0;
  for (clio::run::u64 r = 0; r < 16; ++r) {
    const clio::run::u64 lo = r * (kDirOffEnd / 16);
    const clio::run::u64 hi = r == 15 ? kDirOffEnd : (r + 1) * (kDirOffEnd / 16);
    clio::run::u64 start = lo;
    while (start < hi) {
      clio::run::u64 next = 0;
      const std::vector<DirRangeEntry> c = ReadRange(blocks, start, 300, &next);
      for (size_t i = 0; i < c.size(); ++i) {
        // The offset a reader is told after entry i is the next entry's: it
        // stops once that reaches its range's end.
        if (c[i].off_ >= hi) break;
        REQUIRE(c[i].off_ >= lo);
        ++total;
      }
      start = next;
    }
  }
  REQUIRE(total == all.size());
}

TEST_CASE("dir_range_edges", "[fs][inodetable][diroff]") {
  // An empty directory: no entries, and nothing after them.
  Blocks empty;
  empty[0].dir_ = 1;
  clio::run::u64 next = 0;
  REQUIRE(ReadRange(empty, 0, 8192, &next).empty());
  REQUIRE(next == kDirOffEnd);
  REQUIRE(ReadRange(empty, kDirOffEnd, 8192, &next).empty());
  REQUIRE(next == kDirOffEnd);

  // One unsplit block: everything in one chunk whatever the limit.
  Blocks one = GrowDir(50, 64);
  REQUIRE(one.size() == 1);
  REQUIRE(ReadRange(one, 0, 1, &next).size() == 50);
  REQUIRE(next == kDirOffEnd);

  // Pending rows are hidden, leaving rows listed. Blocks with no visible
  // row in the middle are passed over when looking for the next chunk's
  // start, and a chunk that ends before them still reaches the end of the
  // directory when nothing visible follows.
  Blocks blocks = GrowDir(2000, 64);
  std::vector<clio::run::u32> order;
  WalkByOffset(blocks, TableSubtree{0, 0}, &order);
  REQUIRE(order.size() > 8);
  for (size_t i = 2; i < 5; ++i) {
    for (auto &e : blocks.at(order[i]).rows_) e.second.state_ = kRowPending;
  }
  blocks.at(order[5]).rows_.begin()->second.state_ = kRowLeaving;
  const std::vector<DirRangeEntry> all = ListAll(blocks);
  const size_t head =
      blocks.at(order[0]).rows_.size() + blocks.at(order[1]).rows_.size();
  std::vector<DirRangeEntry> c = ReadRange(blocks, 0, head, &next);
  REQUIRE(c.size() == head);
  REQUIRE(next == all[head].off_);
  clio::run::u64 lo = 0, hi = 0;
  DirOffRange(order[5], blocks.at(order[5]).depth_, &lo, &hi);
  REQUIRE(next >= lo);  // the three hidden blocks were skipped
  REQUIRE(next < hi);
  size_t seen = c.size();
  while (next != kDirOffEnd) {
    c = ReadRange(blocks, next, 100, &next);
    seen += c.size();
  }
  REQUIRE(seen == all.size());
  for (size_t i = 5; i < order.size(); ++i) {
    for (auto &e : blocks.at(order[i]).rows_) e.second.state_ = kRowPending;
  }
  c = ReadRange(blocks, 0, head, &next);
  REQUIRE(c.size() == head);
  REQUIRE(next == kDirOffEnd);

  // A block that is not the one its parent split off is refused.
  DirRangeScan scan(0, 10);
  TableSubtree sub;
  REQUIRE(scan.Next(&sub));
  InodeTableBlock wrong;
  wrong.index_ = 1;
  wrong.born_ = wrong.depth_ = 1;
  REQUIRE_FALSE(scan.Take(sub, wrong));
}

TEST_CASE("dir_range_equal_offsets", "[fs][inodetable][diroff]") {
  // Names whose offsets are equal (here: clamped together at the low end)
  // stay in one block and list in name order, all of them from that offset.
  REQUIRE(DirOffOfHash(0) == DirOffOfHash(1ULL << 62));
  std::vector<DirRangeEntry> v{{7, "b", 1, 0}, {7, "a", 2, 0}, {5, "z", 3, 0}};
  SortDirRange(&v);
  REQUIRE(v[0].name_ == "z");
  REQUIRE(v[1].name_ == "a");
  REQUIRE(v[2].name_ == "b");
}

TEST_CASE("canonical_fold", "[fs][canonical]") {
  // ASCII.
  REQUIRE(FsFold("Attrib.ARCHIVE") == "attrib.archive");
  REQUIRE(FsFold("already_lower-09.txt") == "already_lower-09.txt");
  REQUIRE(FsFold("") == "");
  // Precomposed Latin: U+00C9 -> U+00E9.
  REQUIRE(FsFold("CAF\xC3\x89.txt") == "caf\xC3\xA9.txt");
  // Greek: capital sigma and final sigma both fold to small sigma.
  REQUIRE(FsFold("\xCE\xA3\xCE\x9F\xCE\xA6\xCE\x9F\xCE\xA3") ==
          FsFold("\xCF\x83\xCE\xBF\xCF\x86\xCE\xBF\xCF\x82"));
  // Simple folding keeps sharp s (no expansion to "ss"), and capital sharp
  // s folds to it.
  REQUIRE(FsFold("Stra\xC3\x9F" "e") == "stra\xC3\x9F" "e");
  REQUIRE(FsFold("STRASSE") != FsFold("Stra\xC3\x9F" "e"));
  REQUIRE(FsFold("\xE1\xBA\x9E") == "\xC3\x9F");
  // Kelvin sign folds to k.
  REQUIRE(FsFold("\xE2\x84\xAA") == "k");
  // Dotted capital I has no simple fold (Turkish rules are not applied).
  REQUIRE(FsFold("\xC4\xB0") == "\xC4\xB0");
  // No normalization: e + combining acute is not precomposed e-acute.
  REQUIRE(FsFold("Cafe\xCC\x81") == "cafe\xCC\x81");
  REQUIRE(FsFold("Cafe\xCC\x81") != FsFold("Caf\xC3\xA9"));
  // Supplementary plane: Deseret capital long I (U+10400) -> U+10428.
  REQUIRE(FsFold("\xF0\x90\x90\x80") == "\xF0\x90\x90\xA8");
  // Malformed UTF-8 folds only its ASCII bytes and keeps the rest.
  REQUIRE(FsFold(std::string("A\xFF" "B\xC3", 4)) ==
          std::string("a\xFF" "b\xC3", 4));
  REQUIRE(FsFold("\xED\xA0\x80Z") == "\xED\xA0\x80z");  // a surrogate
  // Folding is idempotent: a key never re-keys.
  for (const char *s : {"Attrib.ARCHIVE", "CAF\xC3\x89", "\xE1\xBA\x9E",
                        "\xCE\xA3", "\xE2\x84\xAA", "\xF0\x90\x90\x80"}) {
    REQUIRE(FsFold(FsFold(s)) == FsFold(s));
  }
}

TEST_CASE("canonical_names_and_paths", "[fs][canonical]") {
  // A case-sensitive volume keeps every spelling distinct.
  REQUIRE(FsCanonicalName("Foo", false) == "Foo");
  REQUIRE(FsCanonicalPath("/Games/AoE4/x", false) == "/Games/AoE4/x");
  // A case-insensitive one maps them to one key; '/' is never folded.
  REQUIRE(FsCanonicalName("Foo", true) == FsCanonicalName("fOO", true));
  REQUIRE(FsCanonicalPath("/Games/AOE4/X", true) ==
          FsCanonicalPath("/games/aoe4/x", true));
  REQUIRE(FsCanonicalPath("/Games/AOE4/X", true) == "/games/aoe4/x");
}

TEST_CASE("inode_record_codec", "[fs][inode]") {
  Inode a;
  a.type_ = kFsTypeDir;
  a.nlink_ = 1;
  a.mode_ = 0755;
  a.uid_ = 1000;
  a.gid_ = 100;
  a.atime_ = 1;
  a.mtime_ = 2;
  a.ctime_ = 3;
  a.has_xattr_ = true;
  a.parent_ = 0x4000000100000002ULL;
  a.parent_key_ = "games";
  a.volume_flags_ = kVolumeCaseInsensitive;
  a.fold_version_ = kFsFoldTableVersion;
  const std::string rec = EncodeInodeRecord(a, 0, 3, true);
  Inode b;
  clio::run::u64 size = 99;
  clio::run::u32 writer = 0;
  bool fresh = false;
  REQUIRE(DecodeInodeRecord(rec, &b, &size, &writer, &fresh));
  REQUIRE(b.type_ == kFsTypeDir);
  REQUIRE(b.mode_ == 0755);
  REQUIRE(b.uid_ == 1000);
  REQUIRE(b.gid_ == 100);
  REQUIRE(b.ctime_ == 3);
  REQUIRE(b.has_xattr_);
  REQUIRE_FALSE(b.orphan_);
  REQUIRE(b.parent_ == a.parent_);
  REQUIRE(b.parent_key_ == "games");
  REQUIRE(b.volume_flags_ == kVolumeCaseInsensitive);
  REQUIRE(b.fold_version_ == kFsFoldTableVersion);
  REQUIRE(size == 0);
  REQUIRE(writer == 3);
  REQUIRE(fresh);
  // A record of the earlier format ("CFI1") is refused.
  std::string old = rec;
  std::string magic_old;
  FsEnc(&magic_old).U32(0x31494643u);
  old.replace(0, 4, magic_old);
  REQUIRE_FALSE(DecodeInodeRecord(old, &b, &size, &writer, &fresh));
  // A truncated record is refused.
  REQUIRE_FALSE(DecodeInodeRecord(rec.substr(0, rec.size() - 2), &b, &size,
                                  &writer, &fresh));
}

SIMPLE_TEST_MAIN()
