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
 * MultiDelBlob / MultiDelTag: delete a set of blobs or tags in ONE task.
 *
 * - MultiDelBlob deletes exactly the listed blobs, counts the ones it
 *   found, reports a missing one (first_rc_ = 1), and leaves the rest of
 *   the tag readable with a matching tag size.
 * - MultiDelTag deletes every listed tag's blobs.
 * - Both free real space: a deliberately small 64 MiB tier is filled,
 *   emptied through one MultiDelTag, and filled again.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::core::MultiDelKey;
using clio::cte::core::TagId;

namespace {

/** Directory for the generated runtime config. */
std::string TestDataDir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/** Runtime + CTE client with one small RAM tier. */
class MultiDelFixture {
 public:
  std::string config_path_;
  bool initialized_ = false;

  MultiDelFixture() {
    config_path_ = TestDataDir() + "/multi_del_config.yaml";
    if (fs::exists(config_path_)) fs::remove(config_path_);
    std::ofstream f(config_path_);
    REQUIRE(f.is_open());
    f << R"(
runtime:
  num_threads: 2
  queue_depth: 1024

compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0
    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000
    storage:
      - path: "ram::multi_del_tier"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
)";
    f.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
    initialized_ = true;
  }

  ~MultiDelFixture() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
  }
};

MultiDelFixture *g_fixture = nullptr;

/** Create (or open) a tag and return its id. */
TagId OpenTag(const std::string &name) {
  auto t = CLIO_CTE_CLIENT->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  return t->tag_id_;
}

/** Put `n` blobs "b<i>" of `size` bytes (byte value = i) into `tag`. */
bool PutBlobs(const TagId &tag, int n, clio::run::u64 size) {
  std::string v(size, '\0');
  for (int i = 0; i < n; ++i) {
    std::memset(v.data(), i & 0xff, size);
    auto p = CLIO_CTE_CLIENT->AsyncPutBlob(tag, "b" + std::to_string(i), 0,
                                           size, v.data());
    p.Wait();
    if (p->GetReturnCode() != 0) return false;
  }
  return true;
}

/** Size of a blob, or -1 when it does not exist. */
long long BlobSize(const TagId &tag, const std::string &name) {
  auto s = CLIO_CTE_CLIENT->AsyncGetBlobSize(tag, name);
  s.Wait();
  return s->GetReturnCode() == 0 ? static_cast<long long>(s->size_) : -1;
}

}  // namespace

TEST_CASE("MultiDelBlob - deletes exactly the listed blobs in one task",
          "[multi_del][blob]") {
  REQUIRE(g_fixture != nullptr && g_fixture->initialized_);
  const clio::run::u64 size = 4096;
  const TagId tag = OpenTag("multi_del_blobs");
  REQUIRE(PutBlobs(tag, 200, size));
  std::vector<MultiDelKey> keys;
  for (int i = 0; i < 150; ++i) keys.push_back({tag, "b" + std::to_string(i)});
  keys.push_back({tag, "never_written"});
  auto d = CLIO_CTE_CLIENT->AsyncMultiDelBlob(keys);
  d.Wait();
  REQUIRE(d->num_ok_ == 150);
  REQUIRE(d->first_rc_ == 1);  // the missing blob
  for (int i = 0; i < 200; ++i) {
    const long long sz = BlobSize(tag, "b" + std::to_string(i));
    if (i < 150) {
      REQUIRE(sz < 0);
    } else {
      REQUIRE(sz == static_cast<long long>(size));
    }
  }
  std::string got(size, '\0');
  auto g = CLIO_CTE_CLIENT->AsyncGetBlob(tag, "b199", 0, size, 0, got.data());
  g.Wait();
  REQUIRE(g->GetReturnCode() == 0);
  REQUIRE(got == std::string(size, static_cast<char>(199)));
  auto ts = CLIO_CTE_CLIENT->AsyncGetTagSize(tag);
  ts.Wait();
  REQUIRE(ts->tag_size_ == 50 * size);
}

TEST_CASE("MultiDelTag - deletes every listed tag's blobs",
          "[multi_del][tag]") {
  REQUIRE(g_fixture != nullptr && g_fixture->initialized_);
  std::vector<TagId> tags;
  for (int t = 0; t < 3; ++t) {
    tags.push_back(OpenTag("multi_del_tag_" + std::to_string(t)));
    REQUIRE(PutBlobs(tags.back(), 20, 8192));
  }
  auto d = CLIO_CTE_CLIENT->AsyncMultiDelTag(tags);
  d.Wait();
  REQUIRE(d->GetReturnCode() == 0);
  REQUIRE(d->num_ok_ == 3);
  for (const TagId &t : tags) {
    for (int i = 0; i < 20; ++i) {
      REQUIRE(BlobSize(t, "b" + std::to_string(i)) < 0);
    }
  }
}

TEST_CASE("MultiDelTag - frees the tier for reuse", "[multi_del][space]") {
  REQUIRE(g_fixture != nullptr && g_fixture->initialized_);
  // 48 MiB of a 64 MiB tier, twice: the second fill only fits if the one
  // MultiDelTag in between returned every block.
  const TagId first = OpenTag("multi_del_fill_a");
  REQUIRE(PutBlobs(first, 48, 1024 * 1024));
  auto d = CLIO_CTE_CLIENT->AsyncMultiDelTag({first});
  d.Wait();
  REQUIRE(d->num_ok_ == 1);
  const TagId second = OpenTag("multi_del_fill_b");
  REQUIRE(PutBlobs(second, 48, 1024 * 1024));
}

int main(int argc, char **argv) {
  g_fixture = new MultiDelFixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
