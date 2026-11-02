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
