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
 * GRAY-SCOTT DATA ORGANIZER TEST
 *
 * Exercises GrayScottDataOrganizer (organizer: "grayscott"):
 *  - the static policy (TargetScore / PageOf): checkpoint tags go cold and
 *    nothing else does by age; with a ReorganizeHint(step) the region
 *    pair about to be overwritten goes warm; everything else is left alone;
 *  - end to end: blobs in a gv_gs_ck* tag are demoted to the slow tier by the
 *    periodic DynamicReorganize round without any explicit call, while a blob
 *    that has been read stays where it is.
 *
 * What this test does NOT claim: a speedup. Measured on the paged Gray-Scott
 * bench the policy is free when idle (-0.3%) and net negative (+4.5%) in the
 * regime where checkpoints compete with live data for the fast tier, because
 * demotion-by-rescore is a migration and the freed slot refills at once. See
 * the organizer header for the table. The test guards the policy's
 * correctness and its idle cost, which are the properties worth keeping.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/data_organizer/data_organizer.h>
#include <clio_cte/core/data_organizer/grayscott_organizer.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include "simple_test.h"

namespace fs = std::filesystem;
using clio::cte::core::GrayScottDataOrganizer;
using clio::cte::core::OrganizerBlobStat;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}
static constexpr clio::run::u64 kBlobSize = 256 * 1024;
static constexpr int kOrganizerPeriodMs = 500;

/**
 * A vector page blob name AS AN ORGANIZER SEES IT.
 *
 * DeviceVector submits page names as a raw little-endian u32
 * (Context::kBlobNameRawInt32), so it is tempting to build one here with
 * memcpy -- and this helper used to. That is wrong: the CTE runs
 * NormalizeBlobName() on the way in, which renders the name DECIMAL in place
 * and clears the flag, so what reaches a BlobInfo and therefore an organizer
 * is decimal text. Encoding the raw form here made the test agree with a
 * PageOf() that had the same bug, which is how rule 2 shipped decoding every
 * real page number to garbage.
 *
 * @param pn page number
 * @return the decimal blob name the organizer will actually be handed
 */
static std::string PageName(clio::run::u32 pn) {
  return std::to_string(pn);
}

TEST_CASE("GrayScott organizer - static policy", "[cte][organizer][grayscott]") {
  const clio::cte::core::Timestamp now = 10'000'000'000ull;  // 10 s, ns
  OrganizerBlobStat s;
  s.score_ = 1.0f;
  s.last_modified_ = now - 5'000'000'000ull;   // written 5 s ago
  s.last_read_ = now - 1'000'000'000ull;       // read 1 s ago

  SECTION("PageOf decodes a decimal name and rejects anything else") {
    clio::run::u64 pn = 0;
    REQUIRE(GrayScottDataOrganizer::PageOf(PageName(37), pn));
    REQUIRE(pn == 37);
    REQUIRE_FALSE(GrayScottDataOrganizer::PageOf("hot_blob", pn));
  }
  SECTION("checkpoint tags go cold regardless of access") {
    s.tag_name_ = "gv_gs_ck3"; s.blob_name_ = PageName(1);
    REQUIRE(GrayScottDataOrganizer::TargetScore(s, now, 0, 8) == GrayScottDataOrganizer::kColdScore);
  }
  SECTION("a written-but-never-read blob is LEFT ALONE (paged reads never stamp last_read_)") {
    s.tag_name_ = "gv_grayscott"; s.blob_name_ = PageName(2); s.last_read_ = 0;
    s.last_modified_ = now - 30'000'000'000ull;   // 30 s old, still not a checkpoint
    REQUIRE(GrayScottDataOrganizer::TargetScore(s, now, 0, 8) < 0.0f);
  }
  SECTION("with a hint, only the pair being overwritten is demoted") {
    s.tag_name_ = "gv_grayscott";
    // 8 pages -> nz = 2: regions {0,1} are pages 0-3, {2,3} are pages 4-7.
    s.blob_name_ = PageName(5);   // region 2
    REQUIRE(GrayScottDataOrganizer::TargetScore(s, now, /*step 0*/ 1, 8) == GrayScottDataOrganizer::kWarmScore);
    s.blob_name_ = PageName(1);   // region 0: read this step, leave alone
    REQUIRE(GrayScottDataOrganizer::TargetScore(s, now, 1, 8) < 0.0f);
    REQUIRE(GrayScottDataOrganizer::TargetScore(s, now, /*step 1*/ 2, 8) == GrayScottDataOrganizer::kWarmScore);
  }
  SECTION("no hint, or a tag not shaped like the vector: leave alone") {
    s.tag_name_ = "gv_grayscott"; s.blob_name_ = PageName(5);
    REQUIRE(GrayScottDataOrganizer::TargetScore(s, now, 0, 8) < 0.0f);
    REQUIRE(GrayScottDataOrganizer::TargetScore(s, now, 1, 7) < 0.0f);
  }
}

class GrayScottOrganizerFixture {
 public:
  std::string config_path_, file_storage_path_;
  GrayScottOrganizerFixture() {
    config_path_ = chi_test_data_dir() + "/grayscott_organizer_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/grayscott_organizer_storage.bin";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  ~GrayScottOrganizerFixture() { Cleanup(); }
  void Cleanup() {
    for (const auto &p : {config_path_, file_storage_path_})
      if (fs::exists(p)) fs::remove(p);
  }
  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# DataOrganizer Test Configuration
# - 16MB DRAM (fast tier, score 1.0)
# - 64MB File (slow tier, score 0.2)
# - frecency organizer, 2 replicas, )" << kOrganizerPeriodMs << R"( ms period

runtime:
  num_threads: 2
  queue_depth: 1024
  first_busy_wait: 10000
  max_sleep: 50000

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
      # Fast tier: 16MB DRAM (score 1.0)
      - path: "ram::organizer_dram"
        bdev_type: "ram"
        capacity_limit: "16MB"
        score: 1.0

      # Slow tier: 64MB File (score 0.2)
      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "64MB"
        score: 0.2

    dpe:
      dpe_type: "max_bw"

    organizer: "grayscott"
    organizer_tasks: 2
    organizer_period_ms: )" << kOrganizerPeriodMs << R"(
)";
    config_file.close();
  }
};

/** Poll a blob's score until `pred` holds or the deadline passes. */
template <class Pred>
static float WaitForScore(clio::cte::core::Tag &tag, const std::string &name,
                          Pred pred, int max_ms) {
  float score = tag.GetBlobScore(name);
  for (int waited = 0; !pred(score) && waited < max_ms; waited += 250) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    score = tag.GetBlobScore(name);
  }
  return score;
}

TEST_CASE("GrayScott organizer - checkpoint tags are demoted, read data is not",
          "[cte][organizer][grayscott][e2e]") {
  GrayScottOrganizerFixture fx;
  REQUIRE(CLIO_CTE_CLIENT != nullptr);
  std::vector<char> data(kBlobSize, 'G');

  // A checkpoint tag: written at the fast tier's score, never read.
  clio::cte::core::Tag ckpt("gv_gs_ck0");
  for (clio::run::u32 pn = 0; pn < 4; ++pn) {
    ckpt.PutBlob(PageName(pn), data.data(), kBlobSize, 0, 1.0f);
  }
  // A live blob: written, then READ, so last_read_ is stamped.
  clio::cte::core::Tag live("gv_gs_live");
  live.PutBlob("plane", data.data(), kBlobSize, 0, 1.0f);
  { std::vector<char> buf(kBlobSize); live.GetBlob("plane", buf.data(), kBlobSize); }

  const float ck = WaitForScore(ckpt, PageName(1), [](float v) { return v <= 0.05f; },
                                12 * kOrganizerPeriodMs);
  INFO("checkpoint page score after organizer rounds: " << ck);
  REQUIRE(ck <= 0.05f);

  const float lv = WaitForScore(live, "plane", [](float) { return false; },
                                2 * kOrganizerPeriodMs);
  INFO("live blob score after organizer rounds: " << lv);
  REQUIRE(lv > 0.5f);
}

SIMPLE_TEST_MAIN()
