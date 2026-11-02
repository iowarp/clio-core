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
 * Tag xattr sections (TagXattrTask): set / get / delete / list one key of a
 * tag, the create / replace / must-exist flags, size limits, list paging,
 * binary values, and the section dropping with its tag.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::core::TagId;

namespace {

/** Directory for the test's config file. */
std::string DataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Starts a single-node runtime with one RAM tier, once per process. */
class XattrFixture {
 public:
  XattrFixture() {
    conf_ = DataDir() + "/tag_xattr_config.yaml";
    std::ofstream f(conf_);
    f << R"(
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0
    targets:
      neighborhood: 1
    storage:
      - path: "ram::tag_xattr_dram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
    f.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", conf_.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  }
  ~XattrFixture() {
    std::error_code ec;
    fs::remove(conf_, ec);
  }

 private:
  std::string conf_;
};

XattrFixture &Fixture() {
  static XattrFixture fx;
  return fx;
}

/** Set a key; returns the task's return code. */
clio::run::u32 Set(const TagId &tag, const std::string &key,
                   const std::string &value, clio::run::u32 flags = 0) {
  auto f = CLIO_CTE_CLIENT->AsyncTagSetXattr(tag, key, value, flags);
  f.Wait();
  return f->GetReturnCode();
}

/** Get a key; returns found and fills *value. */
bool Get(const TagId &tag, const std::string &key, std::string *value) {
  auto f = CLIO_CTE_CLIENT->AsyncTagGetXattr(tag, key);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  if (f->found_) *value = f->value_.str();
  return f->found_ != 0;
}

/** Delete a key (empty = all); returns the task's return code. */
clio::run::u32 Del(const TagId &tag, const std::string &key,
                   clio::run::u32 flags = 0) {
  auto f = CLIO_CTE_CLIENT->AsyncTagDelXattr(tag, key, flags);
  f.Wait();
  return f->GetReturnCode();
}

/** List one page of keys after `cursor`; *more says if any remain. */
std::vector<std::string> List(const TagId &tag, const std::string &cursor,
                              clio::run::u32 max_keys, bool *more) {
  auto f = CLIO_CTE_CLIENT->AsyncTagListXattr(tag, cursor, max_keys);
  f.Wait();
  REQUIRE(f->GetReturnCode() == 0);
  *more = f->found_ != 0;
  return f->keys_;
}

}  // namespace

TEST_CASE("TagXattr - set get replace delete", "[cte][tag_xattr]") {
  Fixture();
  const TagId tag{7001, 1};
  std::string v;
  REQUIRE_FALSE(Get(tag, "a", &v));
  REQUIRE(Set(tag, "a", "one") == 0);
  REQUIRE(Get(tag, "a", &v));
  REQUIRE(v == "one");
  REQUIRE(Set(tag, "a", "two") == 0);  // upsert
  REQUIRE(Get(tag, "a", &v));
  REQUIRE(v == "two");
  REQUIRE(Del(tag, "a") == 0);
  REQUIRE_FALSE(Get(tag, "a", &v));
  REQUIRE(Del(tag, "a") == 0);  // absent is fine without must-exist
}

TEST_CASE("TagXattr - create replace must-exist flags", "[cte][tag_xattr]") {
  Fixture();
  const TagId tag{7001, 2};
  REQUIRE(Set(tag, "k", "v", clio::cte::core::kTagXattrReplace) ==
          clio::cte::core::kTagXattrMissingRc);
  REQUIRE(Set(tag, "k", "v", clio::cte::core::kTagXattrCreate) == 0);
  REQUIRE(Set(tag, "k", "w", clio::cte::core::kTagXattrCreate) ==
          clio::cte::core::kTagXattrExistsRc);
  REQUIRE(Set(tag, "k", "w", clio::cte::core::kTagXattrReplace) == 0);
  std::string v;
  REQUIRE(Get(tag, "k", &v));
  REQUIRE(v == "w");
  REQUIRE(Del(tag, "k", clio::cte::core::kTagXattrMustExist) == 0);
  REQUIRE(Del(tag, "k", clio::cte::core::kTagXattrMustExist) ==
          clio::cte::core::kTagXattrMissingRc);
}

TEST_CASE("TagXattr - binary values and limits", "[cte][tag_xattr]") {
  Fixture();
  const TagId tag{7001, 3};
  std::string bin(300, '\0');
  for (size_t i = 0; i < bin.size(); ++i) bin[i] = static_cast<char>(i);
  REQUIRE(Set(tag, "bin", bin) == 0);
  std::string v;
  REQUIRE(Get(tag, "bin", &v));
  REQUIRE(v == bin);
  REQUIRE(Set(tag, std::string(clio::cte::core::kTagXattrMaxKey + 1, 'k'),
              "v") == clio::cte::core::kTagXattrTooBigRc);
  REQUIRE(Set(tag, "big",
              std::string(clio::cte::core::kTagXattrMaxValue + 1, 'x')) ==
          clio::cte::core::kTagXattrTooBigRc);
  REQUIRE(Set(tag, "max",
              std::string(clio::cte::core::kTagXattrMaxValue, 'x')) == 0);
  REQUIRE(Set(tag, "", "v") == clio::cte::core::kTagXattrBadOpRc);
}

TEST_CASE("TagXattr - list pages in key order", "[cte][tag_xattr]") {
  Fixture();
  const TagId tag{7001, 4};
  for (int i = 9; i >= 0; --i) {
    REQUIRE(Set(tag, "key" + std::to_string(i), std::to_string(i)) == 0);
  }
  bool more = false;
  std::vector<std::string> all = List(tag, "", 0, &more);
  REQUIRE(all.size() == 10);
  REQUIRE_FALSE(more);
  REQUIRE(all.front() == "key0");
  REQUIRE(all.back() == "key9");
  std::vector<std::string> paged;
  std::string cursor;
  for (int pages = 0; pages < 10; ++pages) {
    auto page = List(tag, cursor, 3, &more);
    paged.insert(paged.end(), page.begin(), page.end());
    if (!more) break;
    cursor = page.back();
  }
  REQUIRE(paged == all);
}

TEST_CASE("TagXattr - sections are per tag; empty key clears", "[cte][tag_xattr]") {
  Fixture();
  const TagId a{7001, 5};
  const TagId b{7001, 6};
  REQUIRE(Set(a, "x", "a") == 0);
  REQUIRE(Set(a, "y", "a") == 0);
  REQUIRE(Set(b, "x", "b") == 0);
  std::string v;
  REQUIRE(Get(b, "x", &v));
  REQUIRE(v == "b");
  REQUIRE(Del(a, "") == 0);
  bool more = false;
  REQUIRE(List(a, "", 0, &more).empty());
  REQUIRE(Get(b, "x", &v));
  REQUIRE(v == "b");
}

TEST_CASE("TagXattr - deleting the tag drops its section", "[cte][tag_xattr]") {
  Fixture();
  auto *cte = CLIO_CTE_CLIENT;
  auto t = cte->AsyncGetOrCreateTag("tag_xattr_deleted");
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  const TagId tag = t->tag_id_;
  REQUIRE(Set(tag, "x", "1") == 0);
  auto d = cte->AsyncDelTag(tag);
  d.Wait();
  REQUIRE(d->GetReturnCode() == 0);
  std::string v;
  REQUIRE_FALSE(Get(tag, "x", &v));
}

namespace {
/** Get or create a tag by name; returns its id. */
TagId Tag(const std::string &name) {
  auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}
}  // namespace

TEST_CASE("TagXattr - a parent with a section outlives its last child",
          "[cte][tag_xattr]") {
  Fixture();
  // "/xkeep/child" materializes "/xkeep"; a section on it makes it state, not
  // a path placeholder, so deleting the child must not prune it (its delete
  // record would erase the section at the next restart).
  const TagId kept = Tag("/xkeep");
  const TagId child = Tag("/xkeep/child");
  REQUIRE(Set(kept, "~i", "record") == 0);
  auto d = CLIO_CTE_CLIENT->AsyncDelTag(child);
  d.Wait();
  REQUIRE(d->GetReturnCode() == 0);
  REQUIRE(Tag("/xkeep") == kept);
  std::string v;
  REQUIRE(Get(kept, "~i", &v));
  REQUIRE(v == "record");

  // Without state the placeholder still goes with its last child.
  const TagId plain = Tag("/xplain");
  const TagId pchild = Tag("/xplain/child");
  auto d2 = CLIO_CTE_CLIENT->AsyncDelTag(pchild);
  d2.Wait();
  REQUIRE(d2->GetReturnCode() == 0);
  REQUIRE_FALSE(Tag("/xplain") == plain);
}

SIMPLE_TEST_MAIN()
