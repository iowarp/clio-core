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
 * Directory blocks (fs_dir_block.h): the codec, delta application and the
 * split tree, exercised without a runtime. The split test grows a directory
 * the way the homes do -- insert, and split any block past a threshold --
 * then checks that every name is reachable by walking from block 0 and that
 * each lives in exactly one block.
 */
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../../context-runtime/test/simple_test.h"
#include <clio_cte/filesystem/fs_dir_block.h>

using namespace clio::cte::filesystem;

namespace {

/** A whole directory as a map index -> block (what the homes hold). */
using Blocks = std::map<clio::run::u32, DirBlock>;

/** Walk from block 0 to the block holding `leaf` (NextBlockForName). */
clio::run::u32 Walk(const Blocks &blocks, const std::string &leaf) {
  const clio::run::u64 h = DirNameHash(leaf);
  clio::run::u32 k = 0;
  for (int step = 0; step < 64; ++step) {
    const clio::run::u32 next = NextBlockForName(blocks.at(k), h);
    if (next == k) return k;
    k = next;
  }
  return ~0u;
}

/** Split block k the way SplitBlock does. */
void Split(Blocks &blocks, clio::run::u32 k) {
  DirBlock &b = blocks.at(k);
  const clio::run::u32 d = b.depth_;
  DirBlock child;
  child.dir_ = b.dir_;
  child.index_ = k + (1u << d);
  child.born_ = child.depth_ = d + 1;
  for (auto it = b.ents_.begin(); it != b.ents_.end();) {
    if ((DirNameHash(it->first) >> d) & 1ULL) {
      child.ents_[it->first] = it->second;
      it = b.ents_.erase(it);
    } else {
      ++it;
    }
  }
  b.depth_ = d + 1;
  blocks[child.index_] = child;
}

/** Insert like a home: walk, insert, split past `limit`. */
void Insert(Blocks &blocks, const std::string &leaf, size_t limit) {
  const clio::run::u32 k = Walk(blocks, leaf);
  DirEntry e;
  e.id_ = DirNameHash(leaf) | 1;
  blocks.at(k).ents_[leaf] = e;
  if (blocks.at(k).ents_.size() > limit) Split(blocks, k);
}

}  // namespace

TEST_CASE("dir_block_codec_roundtrip", "[fs][dirblock]") {
  DirBlock b;
  b.dir_ = 0x4000000100000007ULL;
  b.index_ = 5;
  b.born_ = 2;
  b.depth_ = 4;
  b.version_ = 42;
  b.mtime_ = 123456789;
  b.sealed_ = true;
  b.hdr_.parent_ = 99;
  b.hdr_.leaf_ = "src";
  b.hdr_.mode_ = 0755;
  b.ents_["a"] = DirEntry{11, kFsTypeFile, kDirEntLive};
  b.ents_["pending"] = DirEntry{12, kFsTypeDir, kDirEntPending};
  b.ents_["leaving"] = DirEntry{13, kFsTypeSymlink, kDirEntLeaving};

  // A cache transfer keeps every state.
  DirBlock c;
  const std::string xfer = EncodeDirBlock(b, false);
  REQUIRE(DecodeDirBlock(xfer.data(), xfer.size(), &c));
  REQUIRE(c.dir_ == b.dir_);
  REQUIRE(c.index_ == 5);
  REQUIRE(c.born_ == 2);
  REQUIRE(c.depth_ == 4);
  REQUIRE(c.version_ == 42);
  REQUIRE(c.mtime_ == 123456789);
  REQUIRE(c.sealed_);
  REQUIRE(c.hdr_.leaf_ == "src");
  REQUIRE(c.ents_.size() == 3);
  REQUIRE(c.ents_["leaving"].state_ == kDirEntLeaving);

  // The durable image drops in-flight states: pending gone, leaving live,
  // never sealed.
  DirBlock d;
  const std::string img = EncodeDirBlock(b, true);
  REQUIRE(DecodeDirBlock(img.data(), img.size(), &d));
  REQUIRE(d.ents_.size() == 2);
  REQUIRE(d.ents_.count("pending") == 0);
  REQUIRE(d.ents_["leaving"].state_ == kDirEntLive);
  REQUIRE_FALSE(d.sealed_);

  // A blob that shrank keeps stale bytes past its end: they are ignored.
  std::string padded = img + std::string(64, 'x');
  DirBlock e;
  REQUIRE(DecodeDirBlock(padded.data(), padded.size(), &e));
  REQUIRE(e.ents_.size() == 2);

  // Garbage is refused.
  std::string bad = img;
  bad[0] ^= 0x55;
  REQUIRE_FALSE(DecodeDirBlock(bad.data(), bad.size(), &e));
}

TEST_CASE("dir_block_delta_apply", "[fs][dirblock]") {
  DirBlock home;
  home.dir_ = 7;
  home.version_ = 10;
  DirBlock holder = home;

  // Three changes, pushed as one batch of per-change deltas.
  std::vector<DirDelta> batch;
  for (int i = 0; i < 3; ++i) {
    DirDelta x;
    x.dir_ = 7;
    x.base_version_ = home.version_;
    x.new_version_ = ++home.version_;
    x.depth_ = home.depth_;
    x.mtime_ = 1000 + i;
    x.ops_.push_back({"f" + std::to_string(i), true, DirEntry{100u + i, 1, 0}});
    home.ents_["f" + std::to_string(i)] = x.ops_.back().ent_;
    batch.push_back(x);
  }
  // A removal and a header change.
  {
    DirDelta x;
    x.dir_ = 7;
    x.base_version_ = home.version_;
    x.new_version_ = ++home.version_;
    x.depth_ = 1;
    x.mtime_ = 2000;
    x.has_hdr_ = true;
    x.hdr_.mode_ = 0700;
    x.ops_.push_back({"f1", false, DirEntry()});
    batch.push_back(x);
  }
  std::vector<DirDelta> got;
  REQUIRE(DecodeDirDeltas(EncodeDirDeltas(batch), &got));
  REQUIRE(got.size() == 4);
  for (const DirDelta &x : got) REQUIRE(ApplyDirOps(&holder, x));
  REQUIRE(holder.version_ == 14);
  REQUIRE(holder.ents_.size() == 2);
  REQUIRE(holder.ents_.count("f1") == 0);
  REQUIRE(holder.hdr_.mode_ == 0700);
  REQUIRE(holder.depth_ == 1);
  REQUIRE(holder.mtime_ == 2000);

  // A holder that missed a change refuses the next one (it must refetch).
  DirBlock stale;
  stale.dir_ = 7;
  stale.version_ = 10;
  REQUIRE_FALSE(ApplyDirOps(&stale, got[1]));
}

TEST_CASE("dir_block_split_tree", "[fs][dirblock]") {
  Blocks blocks;
  blocks[0].dir_ = 1;
  const size_t kNames = 20000;
  const size_t kLimit = 64;
  std::vector<std::string> names;
  for (size_t i = 0; i < kNames; ++i) {
    names.push_back("file_" + std::to_string(i * 7919));
    Insert(blocks, names.back(), kLimit);
  }
  // It split, and no block kept growing without bound.
  REQUIRE(blocks.size() > kNames / kLimit / 2);
  size_t total = 0;
  for (const auto &kv : blocks) {
    // Each block holds exactly the names its range and depth say it does.
    for (const auto &e : kv.second.ents_) {
      REQUIRE(BlockOwnsName(kv.second, DirNameHash(e.first)));
    }
    total += kv.second.ents_.size();
  }
  REQUIRE(total == kNames);
  // Every name is found by walking down from block 0.
  for (const std::string &n : names) {
    const clio::run::u32 k = Walk(blocks, n);
    REQUIRE(k != ~0u);
    REQUIRE(blocks.at(k).ents_.count(n) == 1);
  }
  // A missing name walks to a block that does not hold it.
  const clio::run::u32 k = Walk(blocks, "absent");
  REQUIRE(blocks.at(k).ents_.count("absent") == 0);
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

TEST_CASE("dir_effective_times", "[fs][dirblock]") {
  DirHeader h;
  h.mtime_ = 500;   // tar restored an old mtime ...
  h.ctime_ = 1000;  // ... at time 1000
  clio::run::u64 m = 0, c = 0;
  DirEffectiveTimes(h, 900, &m, &c);  // no entry change since
  REQUIRE(m == 500);
  REQUIRE(c == 1000);
  DirEffectiveTimes(h, 2000, &m, &c);  // an entry changed afterwards
  REQUIRE(m == 2000);
  REQUIRE(c == 2000);
}

SIMPLE_TEST_MAIN()
