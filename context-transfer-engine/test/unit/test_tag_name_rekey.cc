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
 * Regression: a published directory rename whose destination parent has not
 * been named on this node yet must still move the whole subtree once the
 * parent arrives (core_runtime.cc: TnRename + pending_rekey_ +
 * UnparkNames). Before the fix only the renamed directory itself was
 * re-indexed, leaving its descendants under the old path for good.
 *
 * Batches are ordered by hand on ONE node via UpdateTagNames. Each test
 * case uses its own top-level names so the shared runtime's index never
 * carries state between cases.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::core::EncodeTagNameOp;
using clio::cte::core::MakeTagRefName;
using clio::cte::core::TagId;
using clio::cte::core::TagNameOp;

namespace {

/** Ids far above anything GetOrCreateTag hands out. */
constexpr clio::run::u32 kMajor = 0x7A000000u;

/** @return directory for test scratch files. */
std::string DataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Boots an embedded runtime with a single-RAM-tier CTE core, once. */
class TagNameFixture {
 public:
  TagNameFixture() {
    if (initialized_) return;
    const std::string conf = DataDir() + "/tag_name_rekey_config.yaml";
    const std::string log = DataDir() + "/tag_name_rekey_restart.bin";
    std::ofstream f(conf);
    REQUIRE(f.is_open());
    f << R"(
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0
    storage:
      - path: "ram::tag_name_rekey_ram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
    f.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", conf.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", log.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
    initialized_ = true;
  }

 private:
  static inline bool initialized_ = false;
};

/** Ids for one scenario: root is shared, the rest are per-scenario. */
struct Ids {
  TagId root, d, sub, deep, f3, f2, x;
};

/**
 * Build the ids of scenario `n` (ids never collide across scenarios).
 * @param n scenario number, 1-based
 */
Ids MakeIds(clio::run::u32 n) {
  Ids i;
  i.root = TagId(kMajor, 1);
  i.d = TagId(kMajor, n * 10 + 1);
  i.sub = TagId(kMajor, n * 10 + 2);
  i.deep = TagId(kMajor, n * 10 + 3);
  i.f3 = TagId(kMajor, n * 10 + 4);
  i.f2 = TagId(kMajor, n * 10 + 5);
  i.x = TagId(kMajor, n * 10 + 6);
  return i;
}

/**
 * Publish one batch through the core client and require success.
 * @param batch encoded UpdateTagNames records
 */
void Publish(const std::string &batch) {
  auto t = CLIO_CTE_CLIENT->AsyncUpdateTagNames(
      batch, clio::run::PoolQuery::Local());
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
}

/**
 * Batch A: root plus /<d>/sub/{deep/f3,f2} (seq 100..105).
 * @param ids scenario ids
 * @param d top-level directory name
 * @return encoded batch
 */
std::string BatchA(const Ids &ids, const std::string &d) {
  std::string b;
  EncodeTagNameOp(&b, TagNameOp::kSetRoot, ids.root, 100, "/");
  EncodeTagNameOp(&b, TagNameOp::kAddName, ids.d, 101,
                  MakeTagRefName(ids.root, d));
  EncodeTagNameOp(&b, TagNameOp::kAddName, ids.sub, 102,
                  MakeTagRefName(ids.d, "sub"));
  EncodeTagNameOp(&b, TagNameOp::kAddName, ids.deep, 103,
                  MakeTagRefName(ids.sub, "deep"));
  EncodeTagNameOp(&b, TagNameOp::kAddName, ids.f3, 104,
                  MakeTagRefName(ids.deep, "f3"));
  EncodeTagNameOp(&b, TagNameOp::kAddName, ids.f2, 105,
                  MakeTagRefName(ids.sub, "f2"));
  return b;
}

/**
 * Batch B: rename sub from /<d>/sub to <x>/moved (seq 200).
 * @param ids scenario ids
 * @return encoded batch
 */
std::string BatchB(const Ids &ids) {
  std::string b;
  EncodeTagNameOp(&b, TagNameOp::kRename, ids.sub, 200,
                  MakeTagRefName(ids.d, "sub"),
                  MakeTagRefName(ids.x, "moved"));
  return b;
}

/**
 * Batch C: name the destination parent /<xname> (seq 150, not stale).
 * @param ids scenario ids
 * @param xname top-level name for x
 * @return encoded batch
 */
std::string BatchC(const Ids &ids, const std::string &xname) {
  std::string b;
  EncodeTagNameOp(&b, TagNameOp::kAddName, ids.x, 150,
                  MakeTagRefName(ids.root, xname));
  return b;
}

/**
 * Read the tag search index the way `cte_search --tag-query` does.
 * @param re tag regex
 * @return the set of matching absolute names
 */
std::set<std::string> Query(const std::string &re) {
  auto q = CLIO_CTE_CLIENT->AsyncTagQuery(re, 0,
                                          clio::run::PoolQuery::Local());
  q.Wait();
  REQUIRE(q->GetReturnCode() == 0);
  return std::set<std::string>(q->results_.begin(), q->results_.end());
}

/**
 * Index keys belonging to one scenario (its two top-level names).
 * @param d old top-level name
 * @param x new top-level name
 */
std::set<std::string> ScenarioIndex(const std::string &d,
                                    const std::string &x) {
  return Query("^/(" + d + "|" + x + ")(/.*)?$");
}

}  // namespace

TEST_CASE("rename under a not-yet-named parent re-keys the subtree",
          "[cte][tagnames][rekey]") {
  TagNameFixture fx;
  const Ids ids = MakeIds(1);
  Publish(BatchA(ids, "d2a"));
  REQUIRE(ScenarioIndex("d2a", "xa") ==
          std::set<std::string>({"/d2a", "/d2a/sub", "/d2a/sub/deep",
                                 "/d2a/sub/deep/f3", "/d2a/sub/f2"}));
  Publish(BatchB(ids));  // parent x is unknown: the move must be deferred
  Publish(BatchC(ids, "xa"));
  const auto got = ScenarioIndex("d2a", "xa");
  // Without the pending_rekey_ branch in TnRename only /xa/moved appears and
  // /d2a/sub/{deep,deep/f3,f2} stay behind: the first two checks fail.
  REQUIRE(Query("^/d2a/sub(/.*)?$").empty());
  REQUIRE(got == std::set<std::string>({"/d2a", "/xa", "/xa/moved",
                                        "/xa/moved/deep",
                                        "/xa/moved/deep/f3",
                                        "/xa/moved/f2"}));
}

TEST_CASE("in-order rename (parent first) yields the same index",
          "[cte][tagnames][rekey]") {
  TagNameFixture fx;
  const Ids ids = MakeIds(2);
  Publish(BatchA(ids, "d2b"));
  Publish(BatchC(ids, "xb"));
  Publish(BatchB(ids));
  REQUIRE(Query("^/d2b/sub(/.*)?$").empty());
  REQUIRE(ScenarioIndex("d2b", "xb") ==
          std::set<std::string>({"/d2b", "/xb", "/xb/moved",
                                 "/xb/moved/deep", "/xb/moved/deep/f3",
                                 "/xb/moved/f2"}));
}

TEST_CASE("rename whose destination parent never arrives keeps old subtree",
          "[cte][tagnames][rekey]") {
  TagNameFixture fx;
  const Ids ids = MakeIds(3);
  Publish(BatchA(ids, "d2c"));
  Publish(BatchB(ids));  // x is never named
  // Nothing moved, nothing lost or duplicated; /xc/moved cannot resolve.
  REQUIRE(Query("^/xc(/.*)?$").empty());
  REQUIRE(ScenarioIndex("d2c", "xc") ==
          std::set<std::string>({"/d2c", "/d2c/sub", "/d2c/sub/deep",
                                 "/d2c/sub/deep/f3", "/d2c/sub/f2"}));
}

TEST_CASE("an ancestor rename while a move is parked still lands the move",
          "[cte][tagnames][rekey]") {
  // #1182: xnode_tag_names, 1 run in 3. The subtree move arrives before its
  // destination parent is named (parked, remembering the OLD absolute path
  // /d1d/sub); then an ancestor rename /d1d -> /d2d re-keys the index under
  // /d2d; then the parent arrives. The parked move used to look for
  // /d1d/sub, find nothing, and leave the subtree under /d2d/sub for good.
  TagNameFixture fx;
  const Ids ids = MakeIds(4);
  Publish(BatchA(ids, "d1d"));
  Publish(BatchB(ids));  // sub -> <x>/moved, x unknown: parked
  {
    std::string b;
    EncodeTagNameOp(&b, TagNameOp::kRename, ids.d, 160,
                    MakeTagRefName(ids.root, "d1d"),
                    MakeTagRefName(ids.root, "d2d"));
    Publish(b);  // the ancestor moves while the subtree move is parked
  }
  Publish(BatchC(ids, "xd"));
  REQUIRE(Query("^/d1d(/.*)?$").empty());
  REQUIRE(Query("^/d2d/sub(/.*)?$").empty());
  REQUIRE(ScenarioIndex("d2d", "xd") ==
          std::set<std::string>({"/d2d", "/xd", "/xd/moved",
                                 "/xd/moved/deep", "/xd/moved/deep/f3",
                                 "/xd/moved/f2"}));
}

SIMPLE_TEST_MAIN()
