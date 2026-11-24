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
 * Conditional PutBlob (Context::kPutIfAbsent / kPutIfVersion): create-once,
 * compare-and-swap, and a race of creators with exactly one winner.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::core::Context;
using clio::cte::core::TagId;

namespace {

/** @return directory for test scratch files. */
std::string DataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Fixture: runtime with a CTE core over a RAM tier. */
class CondPutFixture {
 public:
  CondPutFixture() {
    conf_ = DataDir() + "/cond_put_config.yaml";
    restart_log_ = DataDir() + "/cond_put_restart.bin";
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
      - path: "ram::cond_put_dram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
    f.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", conf_.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_RESTART_LOG", restart_log_.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
  }
  ~CondPutFixture() {
    std::error_code ec;
    fs::remove(conf_, ec);
    fs::remove(restart_log_, ec);
  }

 private:
  std::string conf_;
  std::string restart_log_;
};

/** Conditional put of `data`; returns (rc, version out). */
std::pair<clio::run::u32, clio::run::u64> Put(const TagId &tag,
                                              const std::string &name,
                                              const std::string &data,
                                              clio::run::u32 flags,
                                              clio::run::u64 version = 0) {
  auto *cte = CLIO_CTE_CLIENT;
  Context ctx;
  ctx.op_flags_ |= flags;
  ctx.version_ = version;
  auto f = cte->AsyncPutBlob(tag, name, 0, data.size(), data.data(), -1.0f,
                             ctx);
  f.Wait();
  return {f->GetReturnCode(), f->context_.version_};
}

/** Read the first `n` bytes of a blob. */
std::string Get(const TagId &tag, const std::string &name, size_t n) {
  auto *cte = CLIO_CTE_CLIENT;
  std::string out(n, '\0');
  auto g = cte->AsyncGetBlob(tag, name, 0, n, 0u, out.data());
  g.Wait();
  REQUIRE(g->GetReturnCode() == 0);
  return out;
}

/** A fresh tag. */
TagId MakeTag(const std::string &name) {
  auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}

}  // namespace

TEST_CASE("PutIfAbsent creates once", "[cte][condput]") {
  CondPutFixture fx;
  TagId tag = MakeTag("condput_absent");
  auto a = Put(tag, "b", "first", Context::kPutIfAbsent);
  REQUIRE(a.first == 0);
  REQUIRE(a.second != 0);
  auto b = Put(tag, "b", "other", Context::kPutIfAbsent);
  REQUIRE(b.first == clio::cte::core::kPutExistsRc);
  REQUIRE(b.second == a.second);  // the current version is reported
  REQUIRE(Get(tag, "b", 5) == "first");
}

TEST_CASE("PutIfVersion is compare-and-swap", "[cte][condput]") {
  CondPutFixture fx;
  TagId tag = MakeTag("condput_cas");
  auto v0 = Put(tag, "c", "v0", Context::kPutIfVersion, 0);  // absent = 0
  REQUIRE(v0.first == 0);
  auto v1 = Put(tag, "c", "v1", Context::kPutIfVersion, v0.second);
  REQUIRE(v1.first == 0);
  REQUIRE(v1.second > v0.second);
  auto stale = Put(tag, "c", "xx", Context::kPutIfVersion, v0.second);
  REQUIRE(stale.first == clio::cte::core::kPutVersionMismatchRc);
  REQUIRE(stale.second == v1.second);
  REQUIRE(Get(tag, "c", 2) == "v1");
  auto plain = Put(tag, "c", "v2", 0);  // unconditional still works
  REQUIRE(plain.first == 0);
  REQUIRE(Get(tag, "c", 2) == "v2");
}

TEST_CASE("Racing PutIfAbsent: exactly one winner", "[cte][condput]") {
  CondPutFixture fx;
  TagId tag = MakeTag("condput_race");
  for (int round = 0; round < 20; ++round) {
    const std::string name = "r" + std::to_string(round);
    std::atomic<int> wins{0}, exists{0}, other{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
      ts.emplace_back([&, t] {
        auto r = Put(tag, name, "w" + std::to_string(t), Context::kPutIfAbsent);
        if (r.first == 0) wins++;
        else if (r.first == clio::cte::core::kPutExistsRc) exists++;
        else other++;
      });
    }
    for (auto &th : ts) th.join();
    REQUIRE(wins.load() == 1);
    REQUIRE(exists.load() == 7);
    REQUIRE(other.load() == 0);
  }
}

SIMPLE_TEST_MAIN()
