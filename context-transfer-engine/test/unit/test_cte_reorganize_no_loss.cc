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
 * Issue #1097: ReorganizeBlob must never lose a blob's data. A move that
 * finds no room leaves the blob intact where it was; a move that succeeds
 * keeps the bytes.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::core::TagId;

namespace {

constexpr size_t kMiB = 1u << 20;

/** @return directory for test scratch files. */
std::string DataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Fixture: a runtime whose CTE has ONE small RAM tier. */
class ReorgFixture {
 public:
  ReorgFixture() {
    conf_ = DataDir() + "/reorg_no_loss_config.yaml";
    restart_log_ = DataDir() + "/reorg_no_loss_restart.bin";
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
      - path: "ram::reorg_small_dram"
        bdev_type: "ram"
        capacity_limit: "32MB"
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
  ~ReorgFixture() {
    std::error_code ec;
    fs::remove(conf_, ec);
    fs::remove(restart_log_, ec);
  }

 private:
  std::string conf_;
  std::string restart_log_;
};

/** A recognizable payload. */
std::string Payload(size_t n, char seed) {
  std::string v(n, '\0');
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<char>(seed + i % 97);
  return v;
}

/** Put `data` as blob `name`; returns the return code. */
clio::run::u32 Put(const TagId &tag, const std::string &name,
                   const std::string &data) {
  auto p = CLIO_CTE_CLIENT->AsyncPutBlob(tag, name, 0, data.size(),
                                         data.data(), 1.0f);
  p.Wait();
  return p->GetReturnCode();
}

/** Read `n` bytes of blob `name` (empty string on failure). */
std::string Get(const TagId &tag, const std::string &name, size_t n) {
  std::string out(n, '\0');
  auto g = CLIO_CTE_CLIENT->AsyncGetBlob(tag, name, 0, n, 0u, out.data());
  g.Wait();
  return g->GetReturnCode() == 0 ? out : std::string();
}

}  // namespace

TEST_CASE("ReorganizeBlob without room leaves the blob intact",
          "[cte][reorganize][1097]") {
  ReorgFixture fx;
  auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag("reorg_full");
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  const TagId tag = t->tag_id_;
  // Fill the only tier: a move needs room for a second copy, which is gone.
  const std::string a = Payload(12 * kMiB, 'a');
  const std::string b = Payload(12 * kMiB, 'k');
  REQUIRE(Put(tag, "a", a) == 0);
  REQUIRE(Put(tag, "b", b) == 0);
  for (float score : {0.2f, 0.9f, 0.1f}) {
    auto r = CLIO_CTE_CLIENT->AsyncReorganizeBlob(tag, "a", score);
    r.Wait();
    // Success or a refused move are both fine; losing the bytes is not.
    REQUIRE(Get(tag, "a", a.size()) == a);
    REQUIRE(Get(tag, "b", b.size()) == b);
  }
}

TEST_CASE("ReorganizeBlob with room moves and keeps the bytes",
          "[cte][reorganize][1097]") {
  ReorgFixture fx;
  auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag("reorg_room");
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  const TagId tag = t->tag_id_;
  const std::string c = Payload(4 * kMiB, 'c');
  REQUIRE(Put(tag, "c", c) == 0);
  auto r = CLIO_CTE_CLIENT->AsyncReorganizeBlob(tag, "c", 0.3f);
  r.Wait();
  REQUIRE(r->GetReturnCode() == 0);
  REQUIRE(Get(tag, "c", c.size()) == c);
}

SIMPLE_TEST_MAIN()
