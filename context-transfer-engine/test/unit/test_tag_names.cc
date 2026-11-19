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
 * Flat tag names: the core binds each name on the container it hashes to and
 * keeps each tag's name list on the tag's home. These cases drive every name
 * verb -- GetOrCreateTag, RenameTag, GetOrCreateTagAlias, DelTag (plain and
 * POSIX unlink), GetTagName, GetNumAliases, and the batched UpdateTagNames /
 * AsyncMultiRenameTag -- and check the bindings, the name lists and the
 * search index agree after each one. Each case uses its own names.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <clio_runtime/admin/admin_client.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "simple_test.h"

using clio::cte::core::EncodeTagNameOp;
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
    const std::string conf = DataDir() + "/tag_names_config.yaml";
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
      - path: "ram::tag_names_ram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
    f.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", conf.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
    initialized_ = true;
  }

 private:
  static inline bool initialized_ = false;
};

/** @return names matching `re` in the search index (cluster-wide). */
std::set<std::string> Query(const std::string &re) {
  auto q = CLIO_CTE_CLIENT->AsyncTagQuery(re, 0);
  q.Wait();
  REQUIRE(q->GetReturnCode() == 0);
  return std::set<std::string>(q->results_.begin(), q->results_.end());
}

/** @return the id `name` opens (creating it when absent). */
TagId Open(const std::string &name, bool *created = nullptr) {
  auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  if (created != nullptr) *created = t->created_ != 0;
  return t->tag_id_;
}

/** @return `id`'s canonical name, or "" when it has none. */
std::string NameOf(const TagId &id) {
  auto t = CLIO_CTE_CLIENT->AsyncGetTagName(id);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->found_ ? t->tag_name_.str() : std::string();
}

/** @return number of aliases of `id`, or -1 when its home does not know it. */
int AliasesOf(const TagId &id) {
  auto t = CLIO_CTE_CLIENT->AsyncGetNumAliases(id);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->found_ ? static_cast<int>(t->num_aliases_) : -1;
}

/** Run one batch and return its per-record result codes. */
std::vector<clio::run::u32> Batch(const std::string &ops) {
  auto t = CLIO_CTE_CLIENT->AsyncUpdateTagNames(ops);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  std::vector<clio::cte::core::TagNameOpRec> res;
  const std::string buf = t->results_.str();
  REQUIRE(clio::cte::core::DecodeTagNameOps(buf.data(), buf.size(), &res));
  std::vector<clio::run::u32> rc;
  for (const auto &r : res) rc.push_back(static_cast<clio::run::u32>(r.op_));
  return rc;
}

}  // namespace

TEST_CASE("rename keeps the id; the old name is free again",
          "[cte][tagnames]") {
  TagNameFixture fx;
  bool created = false;
  const TagId a = Open("tn_ren_a", &created);
  REQUIRE(created);
  auto r = CLIO_CTE_CLIENT->AsyncRenameTag("tn_ren_a", "tn_ren_b");
  r.Wait();
  REQUIRE(r->GetReturnCode() == 0);
  REQUIRE(r->tag_id_ == a);
  REQUIRE(Open("tn_ren_b", &created) == a);
  REQUIRE_FALSE(created);
  REQUIRE(NameOf(a) == "tn_ren_b");
  REQUIRE(Query("^tn_ren_.*$") == std::set<std::string>({"tn_ren_b"}));
  const TagId fresh = Open("tn_ren_a", &created);
  REQUIRE(created);
  REQUIRE_FALSE(fresh == a);
}

TEST_CASE("rename onto a taken name takes it from the other tag",
          "[cte][tagnames]") {
  TagNameFixture fx;
  const TagId a = Open("tn_over_a");
  const TagId b = Open("tn_over_b");
  auto r = CLIO_CTE_CLIENT->AsyncRenameTag("tn_over_a", "tn_over_b", a);
  r.Wait();
  REQUIRE(r->GetReturnCode() == 0);
  REQUIRE(Open("tn_over_b") == a);
  REQUIRE(NameOf(a) == "tn_over_b");
  REQUIRE(NameOf(b).empty());  // b lost its only name
  REQUIRE(Query("^tn_over_.*$") == std::set<std::string>({"tn_over_b"}));
}

TEST_CASE("aliases: unlink an alias, POSIX-unlink the canonical, delete",
          "[cte][tagnames]") {
  TagNameFixture fx;
  const TagId a = Open("tn_link_a");
  auto l = CLIO_CTE_CLIENT->AsyncGetOrCreateTagAlias(a, "tn_link_b");
  l.Wait();
  REQUIRE(l->found_ == 1);
  REQUIRE(l->tag_id_ == a);
  auto l2 = CLIO_CTE_CLIENT->AsyncGetOrCreateTagAlias("tn_link_a", "tn_link_c");
  l2.Wait();
  REQUIRE(l2->tag_id_ == a);
  REQUIRE(AliasesOf(a) == 2);
  REQUIRE(Open("tn_link_c") == a);

  // An alias already bound returns its tag, unchanged.
  const TagId other = Open("tn_link_other");
  auto l3 = CLIO_CTE_CLIENT->AsyncGetOrCreateTagAlias(other, "tn_link_b");
  l3.Wait();
  REQUIRE(l3->tag_id_ == a);
  REQUIRE(AliasesOf(other) == 0);

  // A missing target is reported, and binds nothing.
  auto l4 = CLIO_CTE_CLIENT->AsyncGetOrCreateTagAlias(
      TagId(kMajor, 999), "tn_link_dangling");
  l4.Wait();
  REQUIRE(l4->found_ == 0);
  REQUIRE(Query("^tn_link_dangling$").empty());

  // Deleting through an alias drops just that name.
  auto d1 = CLIO_CTE_CLIENT->AsyncDelTag("tn_link_c");
  d1.Wait();
  REQUIRE(d1->GetReturnCode() == 0);
  REQUIRE(AliasesOf(a) == 1);
  REQUIRE(Query("^tn_link_[abc]$") ==
          std::set<std::string>({"tn_link_a", "tn_link_b"}));

  // POSIX unlink of the canonical name promotes the surviving link.
  auto d2 = CLIO_CTE_CLIENT->AsyncDelTag(
      "tn_link_a", clio::run::PoolQuery::Dynamic(), true);
  d2.Wait();
  REQUIRE(d2->GetReturnCode() == 0);
  REQUIRE(NameOf(a) == "tn_link_b");
  REQUIRE(AliasesOf(a) == 0);

  // The last name goes, and the tag with it.
  auto d3 = CLIO_CTE_CLIENT->AsyncDelTag(
      "tn_link_b", clio::run::PoolQuery::Dynamic(), true);
  d3.Wait();
  REQUIRE(d3->GetReturnCode() == 0);
  REQUIRE(AliasesOf(a) == -1);
  REQUIRE(Query("^tn_link_[abc]$").empty());
  auto d4 = CLIO_CTE_CLIENT->AsyncDelTag("tn_link_b");
  d4.Wait();
  REQUIRE(d4->GetReturnCode() == 1);  // no such name any more
}

TEST_CASE("delete by id drops every name", "[cte][tagnames]") {
  TagNameFixture fx;
  const TagId a = Open("tn_delid_a");
  auto l = CLIO_CTE_CLIENT->AsyncGetOrCreateTagAlias(a, "tn_delid_b");
  l.Wait();
  auto d = CLIO_CTE_CLIENT->AsyncDelTag(a);
  d.Wait();
  REQUIRE(d->GetReturnCode() == 0);
  REQUIRE(Query("^tn_delid_.*$").empty());
  bool created = false;
  REQUIRE_FALSE(Open("tn_delid_b", &created) == a);
  REQUIRE(created);
}

TEST_CASE("batched name ops (clio-fs style <leaf>/<inode> names)",
          "[cte][tagnames][batch]") {
  TagNameFixture fx;
  const TagId f1(kMajor, 11), f2(kMajor, 12), f3(kMajor, 13);
  std::string ops;
  EncodeTagNameOp(&ops, TagNameOp::kAddName, f1, 0, "a.txt/11");
  EncodeTagNameOp(&ops, TagNameOp::kAddName, f2, 0, "b.txt/12");
  EncodeTagNameOp(&ops, TagNameOp::kAddName, f3, 0, "c.txt/13");
  EncodeTagNameOp(&ops, TagNameOp::kAddName, f1, 0, "hard.txt/11");  // a link
  REQUIRE(Batch(ops) == std::vector<clio::run::u32>({0, 0, 0, 0}));
  REQUIRE(NameOf(f1) == "a.txt/11");
  REQUIRE(AliasesOf(f1) == 1);
  REQUIRE(Query("^[a-z.]+/1[123]$") ==
          std::set<std::string>({"a.txt/11", "b.txt/12", "c.txt/13",
                                 "hard.txt/11"}));

  // mv a.txt d.txt; rm b.txt; rm the link; one batch.
  ops.clear();
  EncodeTagNameOp(&ops, TagNameOp::kRename, f1, 0, "a.txt/11", "d.txt/11");
  EncodeTagNameOp(&ops, TagNameOp::kRemoveName, f2, 0, "b.txt/12");
  EncodeTagNameOp(&ops, TagNameOp::kRemoveName, f1, 0, "hard.txt/11");
  REQUIRE(Batch(ops) == std::vector<clio::run::u32>({0, 0, 0}));
  REQUIRE(NameOf(f1) == "d.txt/11");
  REQUIRE(AliasesOf(f1) == 0);
  REQUIRE(NameOf(f2).empty());
  REQUIRE(Query("^[a-z.]+/1[123]$") ==
          std::set<std::string>({"d.txt/11", "c.txt/13"}));

  // Re-sending a batch is harmless (every step is idempotent).
  REQUIRE(Batch(ops).size() == 3);
  REQUIRE(Query("^[a-z.]+/1[123]$") ==
          std::set<std::string>({"d.txt/11", "c.txt/13"}));

  // The batched RenameTag wrapper.
  auto m = CLIO_CTE_CLIENT->AsyncMultiRenameTag(
      {std::make_tuple(f1, std::string("d.txt/11"), std::string("e.txt/11")),
       std::make_tuple(f3, std::string("c.txt/13"), std::string("f.txt/13"))});
  m.Wait();
  REQUIRE(m->applied_ == 2);
  REQUIRE(Query("^[a-z.]+/1[123]$") ==
          std::set<std::string>({"e.txt/11", "f.txt/13"}));
}

namespace {

/** Append one type-5 snapshot entry (name list of a tag) to `out`. */
void WriteType5(std::ofstream &out, const std::string &name, const TagId &id,
                const std::vector<std::string> &aliases) {
  const uint8_t entry_type = 5;
  const uint32_t name_len = static_cast<uint32_t>(name.size());
  const clio::run::u64 total = 0;
  const uint32_t n = static_cast<uint32_t>(aliases.size());
  out.write(reinterpret_cast<const char *>(&entry_type), sizeof(entry_type));
  out.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
  out.write(name.data(), name_len);
  out.write(reinterpret_cast<const char *>(&id), sizeof(id));
  out.write(reinterpret_cast<const char *>(&total), sizeof(total));
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  for (const auto &a : aliases) {
    const uint32_t len = static_cast<uint32_t>(a.size());
    out.write(reinterpret_cast<const char *>(&len), sizeof(len));
    out.write(a.data(), len);
  }
}

}  // namespace

TEST_CASE("a snapshot from before flat names restores every name",
          "[cte][tagnames][compat]") {
  TagNameFixture fx;
  namespace fs = std::filesystem;
  // An older binary wrote name lists only (no binding entries) and named
  // filesystem links hierarchically. The restored pool must open the old
  // flat names under their old ids and drop the hierarchical ones.
  const std::string log_path = DataDir() + "/tag_names_compat_meta.log";
  const TagId flat(1, 4243), dirent(1, 4244);
  {
    std::ofstream out(log_path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    WriteType5(out, "compat_flat", flat, {"compat_alias"});
    WriteType5(out, "$tagid{1.4243}/leaf", dirent, {});
    REQUIRE(out.good());
  }
  const clio::run::PoolId pool_id(516, 0);
  clio::run::PoolConfig cfg;
  cfg.mod_name_ = "clio_cte_core";
  cfg.pool_name_ = "clio_cte_compat_restored";
  cfg.pool_id_ = pool_id;
  cfg.pool_query_ = clio::run::PoolQuery::Local();
  cfg.restart_ = true;
  {
    std::ostringstream y;
    y << "targets:\n  neighborhood: 1\nstorage:\n"
      << "  - path: \"" << log_path + ".dat" << "\"\n"
      << "    bdev_type: \"file\"\n    capacity_limit: \"16MB\"\n"
      << "    persistence_level: \"temporary\"\n"
      << "performance:\n  metadata_log_path: " << log_path << "\n"
      << "  stat_targets_period_ms: 100\n";
    cfg.config_ = y.str();
  }
  clio::run::admin::Client admin(clio::run::kAdminPoolId);
  auto compose = admin.AsyncCompose(cfg);
  compose.Wait();
  const clio::run::u32 compose_rc = compose->GetReturnCode();
  clio::cte::core::Client c(pool_id);
  TagId got_flat, got_alias;
  bool created_flat = true, created_alias = true;
  std::string canonical;
  bool legacy_gone = false;
  if (compose_rc == 0) {
    auto o1 = c.AsyncGetOrCreateTag("compat_flat");
    o1.Wait();
    got_flat = o1->tag_id_;
    created_flat = o1->created_ != 0;
    auto o2 = c.AsyncGetOrCreateTag("compat_alias");
    o2.Wait();
    got_alias = o2->tag_id_;
    created_alias = o2->created_ != 0;
    auto nm = c.AsyncGetTagName(flat);
    nm.Wait();
    canonical = nm->found_ ? nm->tag_name_.str() : std::string();
    // Migration happens on the next tick: poll under a long deadline.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!legacy_gone && std::chrono::steady_clock::now() < deadline) {
      auto q = c.AsyncGetTagName(dirent);
      q.Wait();
      legacy_gone = q->found_ == 0;
      if (!legacy_gone) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    auto destroy = admin.AsyncDestroyPool(clio::run::PoolQuery::Local(), pool_id);
    destroy.Wait();
  }
  std::error_code ec;
  for (const auto &e : fs::directory_iterator(fs::path(log_path).parent_path(), ec)) {
    if (e.path().filename().string().rfind("tag_names_compat_meta.log", 0) == 0) {
      fs::remove(e.path(), ec);
    }
  }
  REQUIRE(compose_rc == 0);
  REQUIRE(got_flat == flat);
  REQUIRE_FALSE(created_flat);
  REQUIRE(got_alias == flat);
  REQUIRE_FALSE(created_alias);
  REQUIRE(canonical == "compat_flat");
  REQUIRE(legacy_gone);
}

SIMPLE_TEST_MAIN()
