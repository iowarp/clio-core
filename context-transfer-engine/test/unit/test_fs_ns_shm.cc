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
 * The clio-fs shared-memory namespace segment (fs_ns_shm.h) on its own: no
 * runtime. One View formats the segment (the module's role), a second maps
 * it (a client's role); every operation is checked through both, and the op
 * log is checked against what each mutation must record.
 */

#include <atomic>
#include <chrono>
#include <set>
#include <thread>
#include <string>
#include <vector>

#include "../../../context-runtime/test/simple_test.h"
#include <clio_ctp/introspect/system_info.h>
#include <clio_cte/filesystem/fs_ns_shm.h>

namespace ns = clio::cte::filesystem::ns;
using clio::cte::filesystem::kFsTypeDir;
using clio::cte::filesystem::kFsTypeFile;
using clio::cte::filesystem::kFsTypeSymlink;

namespace {

constexpr ns::u64 kRootId = 1;
constexpr ns::u32 kMajor = 0x40000000u;

/** A formatted segment plus a client mapping of it. */
struct Seg {
  ns::View module;
  ns::View client;
  std::string name;

  /**
   * @param tag distinguishes segments of one test run
   * @param root_complete a fresh filesystem (root listing complete)
   * @param log_bytes op log size
   */
  Seg(const std::string &tag, bool root_complete,
      ns::u64 log_bytes = 1u << 20) {
    name = "clio_fsns_test_" + std::to_string(ctp::SystemInfo::GetPid()) +
           "_" + tag;
    ns::Limits lim;
    lim.max_inodes_ = 4096;
    lim.max_dentries_ = 4096;
    lim.max_targets_ = 64;
    lim.log_bytes_ = log_bytes;
    REQUIRE(module.Format(name, lim, false, kRootId, root_complete,
                          /*node_id=*/3, /*namespace_on=*/true));
    REQUIRE(module.NodeId() == 3u);
    REQUIRE(client.Attach(name));
    module.SetIdBlock(kMajor, 100, 100000);
  }
};

/** Drain the log, returning the op types in order. */
std::vector<ns::u32> Drain(ns::View *v, std::vector<std::string> *paths) {
  std::vector<ns::u32> types;
  ns::OpRec rec;
  std::string p;
  std::string p2;
  while (v->PeekOp(&rec, &p, &p2)) {
    types.push_back(rec.type_);
    if (paths != nullptr) paths->push_back(p + (p2.empty() ? "" : "|" + p2));
    v->PopOp(rec.seq_);
  }
  return types;
}

/** Names a readdir returns. */
std::set<std::string> Names(ns::View *v, const std::string &dir) {
  std::vector<ns::DirEnt> ents;
  REQUIRE(v->Readdir(dir, &ents) == 0);
  std::set<std::string> out;
  for (const auto &e : ents) out.insert(e.name);
  return out;
}

}  // namespace

TEST_CASE("ns shm - create lookup readdir across mappings", "[fs][ns]") {
  Seg s("create", true);
  ns::Attr a;
  bool created = false;
  REQUIRE(s.client.Mkdir("/d", 0750, 1, 2, 10) == 0);
  REQUIRE(s.client.Create("/d/f", 0640, 1, 2, true, 11, &a, &created) == 0);
  REQUIRE(created);
  REQUIRE(a.type == kFsTypeFile);
  REQUIRE((a.id >> 32) == kMajor);
  // The module's mapping sees the client's changes.
  ns::Attr b;
  REQUIRE(s.module.Lookup("/d/f", &b) == 0);
  REQUIRE(b.id == a.id);
  REQUIRE(b.mode == 0640u);
  REQUIRE(s.module.Lookup("/d", &b) == 0);
  REQUIRE(b.type == kFsTypeDir);
  REQUIRE((Names(&s.module, "/d") == std::set<std::string>{"f"}));
  // O_EXCL on an existing name, plain open returns it.
  REQUIRE(s.client.Create("/d/f", 0640, 1, 2, true, 12, &a, &created) ==
          EEXIST);
  REQUIRE(s.client.Create("/d/f", 0640, 1, 2, false, 12, &b, &created) == 0);
  REQUIRE_FALSE(created);
  REQUIRE(b.id == a.id);
  // A complete directory answers ENOENT for names it lacks.
  REQUIRE(s.client.Lookup("/d/nope", &b) == ENOENT);
  REQUIRE(s.client.Lookup("/d/f/x", &b) == ENOTDIR);
  std::vector<std::string> paths;
  REQUIRE((Drain(&s.module, &paths) ==
          std::vector<ns::u32>{ns::kOpMkdir, ns::kOpCreate}));
  REQUIRE((paths == std::vector<std::string>{"/d", "/d/f"}));
}

TEST_CASE("ns shm - ids come only from the module's block", "[fs][ns]") {
  Seg s("ids", true);
  s.module.SetIdBlock(kMajor, 5, 2);
  ns::Attr a;
  bool c = false;
  REQUIRE(s.client.Create("/a", 0644, 0, 0, true, 1, &a, &c) == 0);
  REQUIRE((a.id & 0xFFFFFFFFull) == 5u);
  REQUIRE(s.client.Create("/b", 0644, 0, 0, true, 1, &a, &c) == 0);
  REQUIRE((a.id & 0xFFFFFFFFull) == 6u);
  REQUIRE(s.client.Create("/c", 0644, 0, 0, true, 1, &a, &c) == EAGAIN);
  REQUIRE(s.client.Lookup("/c", &a) == ENOENT);  // nothing half-made
  REQUIRE(s.module.IdsLeft() == 0u);
  s.module.SetIdBlock(kMajor, 50, 10);
  REQUIRE(s.client.Create("/c", 0644, 0, 0, true, 1, &a, &c) == 0);
}

TEST_CASE("ns shm - unlink keeps a negative entry until applied",
          "[fs][ns]") {
  Seg s("unlink", true);
  ns::Attr a;
  bool c = false;
  REQUIRE(s.client.Create("/f", 0644, 0, 0, true, 1, &a, &c) == 0);
  REQUIRE(s.client.Unlink("/f", 2) == 0);
  REQUIRE(s.client.Lookup("/f", &a) == ENOENT);
  REQUIRE(s.client.Unlink("/f", 3) == ENOENT);
  REQUIRE(Names(&s.client, "/").empty());
  // Re-creating the name reuses the entry with a new inode.
  ns::Attr a2;
  REQUIRE(s.client.Create("/f", 0600, 0, 0, true, 4, &a2, &c) == 0);
  REQUIRE(a2.id != a.id);
  REQUIRE((Drain(&s.module, nullptr) ==
          std::vector<ns::u32>{ns::kOpCreate, ns::kOpUnlink, ns::kOpCreate}));
}

TEST_CASE("ns shm - incomplete directories miss and fill", "[fs][ns]") {
  Seg s("fill", false);  // restarted filesystem: root listing unknown
  ns::Attr a;
  bool c = false;
  REQUIRE(s.client.Lookup("/old", &a) == ns::kMiss);
  REQUIRE(s.client.Create("/old", 0644, 0, 0, true, 1, &a, &c) == ns::kMiss);
  // The module reports /old as an existing file.
  ns::Attr filled;
  filled.id = 0x4000000000000007ull;
  filled.type = kFsTypeFile;
  filled.size = 42;
  filled.nlink = 1;
  REQUIRE(s.client.FillEntry("/", "old", filled) == 0);
  REQUIRE(s.client.Lookup("/old", &a) == 0);
  REQUIRE(a.size == 42u);
  REQUIRE(s.client.Create("/old", 0644, 0, 0, true, 1, &a, &c) == EEXIST);
  // A name the module lacks: a negative fill lets the create proceed.
  REQUIRE(s.client.FillAbsent("/", "new") == 0);
  REQUIRE(s.client.Create("/new", 0644, 0, 0, true, 1, &a, &c) == 0);
  // A fill never replaces what the table holds (the unlink is pending).
  REQUIRE(s.client.Unlink("/old", 2) == 0);
  REQUIRE(s.client.FillEntry("/", "old", filled) == 0);
  REQUIRE(s.client.Lookup("/old", &a) == ENOENT);
  REQUIRE(s.client.Readdir("/", nullptr) == ns::kMiss);
  REQUIRE(s.client.MarkComplete("/") == 0);
  REQUIRE((Names(&s.client, "/") == std::set<std::string>{"new"}));
  REQUIRE(s.client.Lookup("/never", &a) == ENOENT);
}

TEST_CASE("ns shm - rename", "[fs][ns]") {
  Seg s("rename", true);
  ns::Attr a;
  ns::Attr b;
  bool c = false;
  REQUIRE(s.client.Mkdir("/x", 0755, 0, 0, 1) == 0);
  REQUIRE(s.client.Mkdir("/x/y", 0755, 0, 0, 1) == 0);
  REQUIRE(s.client.Create("/x/f", 0644, 0, 0, true, 1, &a, &c) == 0);
  REQUIRE(s.client.Create("/g", 0644, 0, 0, true, 1, &b, &c) == 0);
  REQUIRE(s.client.Rename("/x", "/x/y/z", false, 2) == EINVAL);
  REQUIRE(s.client.Rename("/x/f", "/g", true, 2) == EEXIST);
  REQUIRE(s.client.Rename("/x/f", "/x/y", false, 2) == EISDIR);
  REQUIRE(s.client.Rename("/x/y", "/g", false, 2) == ENOTDIR);
  // Replace /g: its inode goes, /x/f's moves.
  REQUIRE(s.client.Rename("/x/f", "/g", false, 2) == 0);
  ns::Attr r;
  REQUIRE(s.client.Lookup("/g", &r) == 0);
  REQUIRE(r.id == a.id);
  REQUIRE(s.client.Lookup("/x/f", &r) == ENOENT);
  REQUIRE_FALSE(s.client.StatId(b.id, &r));
  // A directory moves with its subtree.
  REQUIRE(s.client.Create("/x/y/deep", 0644, 0, 0, true, 3, &b, &c) == 0);
  REQUIRE(s.client.Rename("/x", "/moved", false, 4) == 0);
  REQUIRE(s.client.Lookup("/moved/y/deep", &r) == 0);
  REQUIRE(r.id == b.id);
  REQUIRE(s.client.Lookup("/x", &r) == ENOENT);
  REQUIRE((Names(&s.client, "/") == std::set<std::string>{"g", "moved"}));
}

TEST_CASE("ns shm - rmdir link symlink attrs truncate", "[fs][ns]") {
  Seg s("misc", true);
  ns::Attr a;
  ns::Attr r;
  bool c = false;
  REQUIRE(s.client.Mkdir("/d", 0755, 0, 0, 1) == 0);
  REQUIRE(s.client.Create("/d/f", 0644, 0, 0, true, 1, &a, &c) == 0);
  REQUIRE(s.client.Rmdir("/d", 2) == ENOTEMPTY);
  REQUIRE(s.client.Link("/d/f", "/h", 2) == 0);
  REQUIRE(s.client.Lookup("/h", &r) == 0);
  REQUIRE(r.id == a.id);
  REQUIRE(r.nlink == 2u);
  REQUIRE(s.client.Unlink("/d/f", 3) == 0);
  REQUIRE(s.client.Lookup("/h", &r) == 0);
  REQUIRE(r.nlink == 1u);
  REQUIRE(s.client.Rmdir("/d", 4) == 0);
  REQUIRE(s.client.Lookup("/d", &r) == ENOENT);
  REQUIRE(s.client.Symlink("/target/path", "/ln", 0, 0, 5) == 0);
  std::string t;
  REQUIRE(s.client.Readlink("/ln", &t) == 0);
  REQUIRE(t == "/target/path");
  REQUIRE(s.client.Readlink("/h", &t) == EINVAL);
  REQUIRE(s.client.Lookup("/ln", &r) == 0);
  REQUIRE(r.type == kFsTypeSymlink);
  REQUIRE(s.client.Chmod("/h", 0600, 6) == 0);
  REQUIRE(s.client.Chown("/h", 7, 0xFFFFFFFFu, 6) == 0);
  REQUIRE(s.client.Utimens("/h", 0, 99, 6) == 0);
  s.client.RaiseSize(a.id, 5000, 100);
  REQUIRE(s.client.Lookup("/h", &r) == 0);
  REQUIRE(r.mode == 0600u);
  REQUIRE(r.uid == 7u);
  REQUIRE(r.size == 5000u);
  REQUIRE(r.mtime == 100u);
  ns::u64 id = 0;
  ns::u64 old_size = 0;
  ns::u64 seq = 0;
  REQUIRE(s.client.Truncate("/h", 10, 7, &id, &old_size, &seq) == 0);
  REQUIRE(id == a.id);
  REQUIRE(old_size == 5000u);
  REQUIRE(seq == s.client.PushedSeq());
  REQUIRE_FALSE(s.client.WaitApplied(seq, 1.0));
  REQUIRE(s.client.Lookup("/h", &r) == 0);
  REQUIRE(r.size == 10u);
  REQUIRE((Drain(&s.module, nullptr) ==
          std::vector<ns::u32>{ns::kOpMkdir, ns::kOpCreate, ns::kOpLink,
                               ns::kOpUnlink, ns::kOpRmdir, ns::kOpSymlink,
                               ns::kOpChmod, ns::kOpChown, ns::kOpUtimens,
                               ns::kOpTruncate}));
}

TEST_CASE("ns shm - op log wraps and stays ordered", "[fs][ns]") {
  Seg s("wrap", true, 1u << 20);
  ns::Attr a;
  bool c = false;
  std::vector<std::string> seen;
  // ~8 MiB of records through a 1 MiB log: wraps many times.
  const std::string pad(3000, 'p');
  for (int i = 0; i < 10000; ++i) {
    const std::string path = "/" + pad.substr(0, 100 + i % 150) +
                             std::to_string(i);
    REQUIRE(s.client.Create(path, 0644, 0, 0, true, 1, &a, &c) == 0);
    // Each name goes again: the table (4096 entries) holds far fewer than
    // 10000 names, so this also checks applied negatives are reclaimed.
    REQUIRE(s.client.Unlink(path, 2) == 0);
    if (i % 50 == 49) {
      std::vector<std::string> p;
      Drain(&s.module, &p);
      seen.insert(seen.end(), p.begin(), p.end());
    }
  }
  std::vector<std::string> p;
  Drain(&s.module, &p);
  seen.insert(seen.end(), p.begin(), p.end());
  REQUIRE(seen.size() == 20000u);
  for (int i = 0; i < 10000; ++i) {
    const std::string path = "/" + pad.substr(0, 100 + i % 150) +
                             std::to_string(i);
    REQUIRE(seen[2 * i] == path);
    REQUIRE(seen[2 * i + 1] == path);
  }
}

TEST_CASE("ns shm - append generations stage, seal and merge", "[fs][ns]") {
  Seg s("append", true);
  const ns::u64 id = 0x4000000000000042ull;
  ns::u32 g0 = 0;
  ns::u32 g1 = 0;
  ns::u64 off = 0;
  REQUIRE(s.client.AppendIdle(id));
  // Two appends into generation 0 get consecutive blob offsets.
  REQUIRE(s.client.AppendStage(id, 100, &g0, &off) == 0);
  REQUIRE(off == 0u);
  REQUIRE(s.client.AppendStage(id, 50, &g1, &off) == 0);
  REQUIRE(g1 == g0);
  REQUIRE(off == 100u);
  REQUIRE_FALSE(s.client.AppendIdle(id));
  // Seal while one is still copying: inflight holds the merge back.
  REQUIRE(s.client.AppendStaged("/f", id, g0) == 0);
  ns::u32 sealed = 99;
  REQUIRE(s.module.AppendSeal(id, &sealed));
  REQUIRE(sealed == g0);
  ns::u64 len = 0;
  REQUIRE(s.module.AppendInflight(id, sealed, &len));
  // New appends go to the next generation, from offset 0.
  ns::u32 g2 = 0;
  REQUIRE(s.client.AppendStage(id, 7, &g2, &off) == 0);
  REQUIRE(g2 == g0 + 1);
  REQUIRE(off == 0u);
  REQUIRE(s.client.AppendStaged("/f", id, g1) == 0);
  REQUIRE_FALSE(s.module.AppendInflight(id, sealed, &len));
  REQUIRE(len == 150u);
  s.module.AppendMerged(id, sealed);
  REQUIRE(s.client.AppendStaged("/f", id, g2) == 0);
  REQUIRE_FALSE(s.client.AppendIdle(id));  // generation 1 still staged
  REQUIRE(s.module.AppendSeal(id, &sealed));
  REQUIRE(sealed == g2);
  REQUIRE_FALSE(s.module.AppendInflight(id, sealed, &len));
  REQUIRE(len == 7u);
  s.module.AppendMerged(id, sealed);
  REQUIRE(s.client.AppendIdle(id));
  REQUIRE_FALSE(s.module.AppendSeal(id, &sealed));  // nothing staged
  REQUIRE((Drain(&s.module, nullptr) ==
           std::vector<ns::u32>{ns::kOpAppend, ns::kOpAppend,
                                ns::kOpAppend}));
}

TEST_CASE("ns shm - append lock serializes one file's appends", "[fs][ns]") {
  Seg s("applock", true);
  const ns::u64 id = 0x4000000000000043ull;
  REQUIRE(s.client.AppendLock(id) == 0);
  std::atomic<int> got{0};
  std::thread other([&] {
    REQUIRE(s.module.AppendLock(id) == 0);
    got.store(1);
    s.module.AppendUnlock(id);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  REQUIRE(got.load() == 0);  // still held
  s.client.AppendUnlock(id);
  other.join();
  REQUIRE(got.load() == 1);
  REQUIRE(s.client.AppendIdle(id));
}

SIMPLE_TEST_MAIN()
