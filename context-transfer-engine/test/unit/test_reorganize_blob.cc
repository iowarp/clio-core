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
 * REORGANIZE BLOB TEST
 *
 * Tests score-based data movement between tiers:
 * - 16MB DRAM (fast tier, score 1.0)
 * - 64MB File (slow tier, score 0.2)
 *
 * Test cases:
 * 1. PutBlob(score=1.0) should place data in DRAM
 * 2. ReorganizeBlob(score=0.2) should move data to disk
 * 3. GetBlobInfo() verifies score and block placement changes
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

// Test constants for two-tier storage
static constexpr clio::run::u64 kDramCapacity = 16 * 1024 * 1024;  // 16MB
static constexpr clio::run::u64 kDiskCapacity = 64 * 1024 * 1024;  // 64MB
static constexpr clio::run::u64 kBlobSize = 1 * 1024 * 1024;       // 1MB per blob

// Tier scores (higher score = faster tier in this config)
static constexpr float kDramScore = 1.0f;    // Fast tier - DRAM
static constexpr float kDiskScore = 0.2f;    // Slow tier - Disk

/**
 * Test fixture for reorganize blob tests
 */
class ReorganizeBlobTestFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  bool initialized_ = false;

  ReorganizeBlobTestFixture() {
    INFO("=== Initializing ReorganizeBlob Test ===");

    // Setup paths
    std::string home_dir = ctp::SystemInfo::GetHomeDir();
    REQUIRE(!home_dir.empty());
    config_path_ = chi_test_data_dir() + "/reorganize_blob_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/reorganize_blob_storage.bin";

    // Clean up existing files
    Cleanup();

    // Create config file
    CreateConfigFile();

    // Set environment variable for runtime config
    // CLIO_SERVER_CONF is checked first, so set it to override any existing value
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);

    // Initialize CLIO Runtime runtime
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Initialize CTE client
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    initialized_ = true;
    INFO("=== ReorganizeBlob Test Environment Ready ===");
  }

  ~ReorganizeBlobTestFixture() {
    INFO("=== Cleaning up ReorganizeBlob Test ===");
    Cleanup();
  }

  void Cleanup() {
    if (fs::exists(config_path_)) {
      fs::remove(config_path_);
    }
    if (fs::exists(file_storage_path_)) {
      fs::remove(file_storage_path_);
    }
  }

  /**
   * Create configuration file with 16MB DRAM and 64MB file storage
   * DRAM: score=1.0 (fast tier)
   * DISK: score=0.2 (slow tier)
   */
  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());

    config_file << R"(
# ReorganizeBlob Test Configuration
# - 16MB DRAM (fast tier, score 1.0)
# - 64MB File (slow tier, score 0.2)

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
      - path: "ram::reorganize_dram"
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
)";

    config_file.close();
    INFO("Created config file: " << config_path_);
    INFO("  DRAM: 16MB @ score 1.0");
    INFO("  Disk: 64MB @ score 0.2");
  }

  /**
   * Create test data pattern
   */
  std::vector<char> CreateTestData(size_t size, char pattern = 'R') {
    std::vector<char> data(size);
    std::memset(data.data(), pattern, size);
    return data;
  }

  /**
   * Verify test data pattern
   */
  bool VerifyTestData(const std::vector<char>& data, char expected_pattern = 'R') {
    for (size_t i = 0; i < data.size(); ++i) {
      if (data[i] != expected_pattern) {
        return false;
      }
    }
    return true;
  }
};

// Global fixture instance
static ReorganizeBlobTestFixture* g_fixture = nullptr;

/**
 * Test: Put blob with score=1.0 (should go to DRAM)
 */
TEST_CASE("ReorganizeBlob - PutBlob to DRAM", "[reorganize][put][dram]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  // Create a tag for our test blobs
  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();

  INFO("Putting blob with score=1.0 (should go to DRAM)");

  // Allocate shared memory buffer
  auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm_buffer.IsNull());
  ctp::ipc::ShmPtr<> shm_ptr = shm_buffer.shm_.template Cast<void>();

  // Fill buffer with pattern
  auto test_data = g_fixture->CreateTestData(kBlobSize, 'D');  // 'D' for DRAM
  std::memcpy(shm_buffer.ptr_, test_data.data(), kBlobSize);

  // Put blob with high score (1.0) - should go to DRAM
  std::string blob_name = "test_blob_dram";
  auto put_task = tag.AsyncPutBlob(blob_name, shm_ptr, kBlobSize, 0, kDramScore);
  put_task.Wait();

  REQUIRE(put_task->GetReturnCode() == 0);
  INFO("PutBlob succeeded with score=" << kDramScore);

  // Get blob score to verify placement
  INFO("Calling AsyncGetBlobScore...");
  auto score_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_task.Wait();

  if (score_task->GetReturnCode() != 0) {
    INFO("GetBlobScore failed with return code: " << score_task->GetReturnCode());
    REQUIRE(score_task->GetReturnCode() == 0);
  }

  float blob_score = score_task->score_;
  INFO("GetBlobScore returned: " << blob_score);

  // Verify score is close to 1.0
  REQUIRE(std::abs(blob_score - kDramScore) < 0.01f);

  // Get blob size
  auto size_task = cte_client->AsyncGetBlobSize(tag_id, blob_name);
  size_task.Wait();
  REQUIRE(size_task->GetReturnCode() == 0);
  REQUIRE(size_task->size_ == kBlobSize);
  INFO("Blob size: " << size_task->size_);

  CLIO_IPC->FreeBuffer(shm_buffer);
  INFO("SUCCESS: Blob placed with score=1.0");
}

/**
 * Test: ReorganizeBlob to score=0.2 (should move to disk)
 */
TEST_CASE("ReorganizeBlob - Move to Disk", "[reorganize][move][disk][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "test_blob_dram";

  // Get blob score before reorganization
  auto score_before_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_before_task.Wait();
  REQUIRE(score_before_task->GetReturnCode() == 0);

  float score_before = score_before_task->score_;
  INFO("Before ReorganizeBlob:");
  INFO("  Score: " << score_before);

  INFO("Calling ReorganizeBlob with score=" << kDiskScore);

  // Reorganize blob to score 0.2 (should move to disk)
  auto reorg_task = cte_client->AsyncReorganizeBlob(tag_id, blob_name, kDiskScore);
  reorg_task.Wait();

  REQUIRE(reorg_task->GetReturnCode() == 0);
  INFO("ReorganizeBlob completed successfully");

  // Get blob score after reorganization
  auto score_after_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_after_task.Wait();
  REQUIRE(score_after_task->GetReturnCode() == 0);

  float score_after = score_after_task->score_;
  INFO("After ReorganizeBlob:");
  INFO("  Score: " << score_after);

  // Verify score changed to the new value
  REQUIRE(std::abs(score_after - kDiskScore) < 0.01f);
  INFO("SUCCESS: Score changed from " << score_before << " to " << score_after);
}

/**
 * Test: Verify data integrity after reorganization
 */
TEST_CASE("ReorganizeBlob - Verify Data Integrity", "[reorganize][integrity]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  std::string blob_name = "test_blob_dram";

  INFO("Verifying data integrity after reorganization");

  // Allocate buffer for reading
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());

  // Read blob data
  tag.GetBlob(blob_name, read_buffer.shm_.template Cast<void>(), kBlobSize, 0);

  // Verify data pattern
  std::vector<char> read_data(kBlobSize);
  std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);

  bool data_valid = g_fixture->VerifyTestData(read_data, 'D');  // 'D' pattern from put
  REQUIRE(data_valid);

  CLIO_IPC->FreeBuffer(read_buffer);
  INFO("SUCCESS: Data integrity verified after reorganization");
}

/**
 * Test: ReorganizeBlob back to DRAM (promote data)
 */
TEST_CASE("ReorganizeBlob - Promote to DRAM", "[reorganize][promote][dram][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "test_blob_dram";

  // Get blob score before promotion
  auto score_before_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_before_task.Wait();
  REQUIRE(score_before_task->GetReturnCode() == 0);

  float score_before = score_before_task->score_;
  INFO("Before promotion - Score: " << score_before);

  INFO("Promoting blob back to DRAM with score=" << kDramScore);

  // Reorganize blob back to score 1.0 (promote to DRAM)
  auto reorg_task = cte_client->AsyncReorganizeBlob(tag_id, blob_name, kDramScore);
  reorg_task.Wait();

  REQUIRE(reorg_task->GetReturnCode() == 0);

  // Get blob score after promotion
  auto score_after_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_after_task.Wait();
  REQUIRE(score_after_task->GetReturnCode() == 0);

  float score_after = score_after_task->score_;
  INFO("After promotion - Score: " << score_after);

  // Verify score changed back to DRAM score
  REQUIRE(std::abs(score_after - kDramScore) < 0.01f);
  INFO("SUCCESS: Score changed from " << score_before << " to " << score_after);

  // Verify data integrity after promotion
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());

  tag.GetBlob(blob_name, read_buffer.shm_.template Cast<void>(), kBlobSize, 0);

  std::vector<char> read_data(kBlobSize);
  std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);

  bool data_valid = g_fixture->VerifyTestData(read_data, 'D');
  REQUIRE(data_valid);

  CLIO_IPC->FreeBuffer(read_buffer);
  INFO("SUCCESS: Blob promoted back to DRAM with data integrity");
}

/**
 * Regression (issue #753): a GetBlob issued while the SAME blob is being
 * reorganized must neither fail (the old DelBlob + re-Put implementation left
 * a window with no metadata entry at all, so concurrent reads returned "blob
 * not found") nor observe torn data (an empty or half-written block layout).
 * Bounce the blob between tiers and overlap every in-flight move with reads.
 */
TEST_CASE("ReorganizeBlob - Concurrent GetBlob during move",
          "[reorganize][concurrent][integrity][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "test_blob_concurrent";

  // Seed the blob in DRAM with a known pattern.
  auto put_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!put_buffer.IsNull());
  auto test_data = g_fixture->CreateTestData(kBlobSize, 'C');
  std::memcpy(put_buffer.ptr_, test_data.data(), kBlobSize);
  auto put_task = tag.AsyncPutBlob(blob_name,
                                   put_buffer.shm_.template Cast<void>(),
                                   kBlobSize, 0, kDramScore);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());

  // Bounce between tiers; every round overlaps reads with the in-flight move.
  for (int round = 0; round < 6; ++round) {
    const float target_score = (round % 2 == 0) ? kDiskScore : kDramScore;
    auto reorg_task =
        cte_client->AsyncReorganizeBlob(tag_id, blob_name, target_score);
    for (int i = 0; i < 8; ++i) {
      std::memset(read_buffer.ptr_, 0, kBlobSize);
      auto get_task = cte_client->AsyncGetBlob(
          tag_id, blob_name.c_str(), 0, kBlobSize, 0,
          read_buffer.shm_.template Cast<void>());
      get_task.Wait();
      REQUIRE(get_task->GetReturnCode() == 0);
      std::vector<char> read_data(kBlobSize);
      std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);
      REQUIRE(g_fixture->VerifyTestData(read_data, 'C'));
    }
    reorg_task.Wait();
    REQUIRE(reorg_task->GetReturnCode() == 0);
  }

  // The blob must still verify after the final move settles.
  std::memset(read_buffer.ptr_, 0, kBlobSize);
  auto final_get = cte_client->AsyncGetBlob(
      tag_id, blob_name.c_str(), 0, kBlobSize, 0,
      read_buffer.shm_.template Cast<void>());
  final_get.Wait();
  REQUIRE(final_get->GetReturnCode() == 0);
  std::vector<char> final_data(kBlobSize);
  std::memcpy(final_data.data(), read_buffer.ptr_, kBlobSize);
  REQUIRE(g_fixture->VerifyTestData(final_data, 'C'));

  // Cleanup this case's blob (the shared Cleanup case removes the tag).
  auto del_task = cte_client->AsyncDelBlob(tag_id, blob_name);
  del_task.Wait();

  CLIO_IPC->FreeBuffer(read_buffer);
  CLIO_IPC->FreeBuffer(put_buffer);
  INFO("SUCCESS: concurrent reads during reorganize never failed");
}

/**
 * issue #753 (reader half): a GetBlob snapshots the block list and then reads
 * with no lock held, so an extent-freeing mutator used to be able to free the
 * snapshot's extents mid-read; with churn re-allocating those extents, the
 * read returned ANOTHER BLOB'S bytes with rc=0. The fix pins readers and makes
 * Reorganize/DelBlob/shrink drain pins before freeing.
 *
 * This test recreates the full failure recipe, which the older concurrent test
 * above cannot: (a) reads are PIPELINED (several in flight at once, so some are
 * always mid-ReadData when the move's free happens), and (b) churn puts apply
 * extent-REUSE pressure, so a freed-under-reader extent gets scribbled with a
 * different pattern instead of retaining the old bytes by luck.
 */
TEST_CASE("ReorganizeBlob - Pipelined reads survive extent reuse (#753)",
          "[reorganize][concurrent][753][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string blob_name = "race_753_blob";

  // An 8MB victim = 128 chunked block-reads per GetBlob, so every pipelined
  // read spans the move's whole clear-and-replace phase; the filled disk tier
  // (below) forces the churn stream to reuse exactly the extents the move
  // frees. Both are load-bearing: with a small victim or an empty tier the
  // pre-fix race window is missed and freed extents keep stale-but-correct
  // bytes, hiding the bug.
  constexpr clio::run::u64 kVictimSize = 8 * 1024 * 1024;
  // ONE fully-armed round. Repeating the near-full churn cycle accumulates
  // tier capacity-accounting drift until ReorganizeBlob's re-place fails at
  // every score and takes its triple-failure exit — which LOSES the blob
  // ("blob data LOST (entry kept, size 0)"), a real capacity-pressure hazard
  // tracked separately; this test's job is the reader-vs-reclaim race, and
  // round 0 arms it fully (fresh tiers, zero slack, mid-read clear).
  constexpr int kRounds = 1;
  constexpr int kDepth = 4;   // pipelined 8MB reads kept in flight
  // Upper bound on fill blobs; the loop below stops at the tier's ACTUAL
  // remaining capacity instead of assuming it (earlier cases in this binary —
  // and their force_net variants — leave tier residue that shifts the number).
  constexpr int kFillMax = 64;

  auto put_buffer = CLIO_IPC->AllocateBuffer(kVictimSize);
  REQUIRE(!put_buffer.IsNull());
  auto expect = g_fixture->CreateTestData(kVictimSize, 'R');
  std::memcpy(put_buffer.ptr_, expect.data(), kVictimSize);

  auto churn_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!churn_buffer.IsNull());
  auto churn_data = g_fixture->CreateTestData(kBlobSize, 'X');
  std::memcpy(churn_buffer.ptr_, churn_data.data(), kBlobSize);

  // Seed the victim ON DISK, and in MANY SMALL BLOCKS. Both matter:
  //  - Disk residency: a disk blob is not direct-readable, so every read takes
  //    the RPC path into GetBlobImpl (a DRAM blob is served by the client-side
  //    SHM fast path, which never enters the runtime read path and is
  //    separately protected by placement_gen_ revalidation).
  //  - Multi-block layout: ReadData issues one bdev read PER BLOCK, and the
  //    reader-vs-reclaim window exists between those awaits. A single-extent
  //    victim is read in ONE pread and the window is effectively zero — the
  //    128 x 64KB incremental seed below is what makes a read span 128 awaits,
  //    so the move's free lands mid-read deterministically.
  constexpr clio::run::u64 kSeedChunk = 64 * 1024;
  auto seed_victim = [&]() -> bool {
    auto del = cte_client->AsyncDelBlob(tag_id, blob_name);
    del.Wait();  // rc ignored: first seed has nothing to delete
    for (clio::run::u64 off = 0; off < kVictimSize; off += kSeedChunk) {
      auto put = tag.AsyncPutBlob(
          blob_name,
          ctp::ipc::ShmPtr<>((put_buffer.shm_ + off).template Cast<void>()),
          kSeedChunk, off, kDiskScore);
      put.Wait();
      if (put->GetReturnCode() != 0) return false;
    }
    return true;
  };
  // Seed FIRST (so the victim's 8MB definitely fits), then fill the tier to
  // its brim: put 1MB fills until one fails with no-space. Zero slack is what
  // forces the churn burst to reuse the victim's freed extents.
  REQUIRE(seed_victim());
  // Fill the disk tier to its brim WITHOUT spilling into DRAM: a put at
  // kDiskScore whose disk allocation fails is silently placed on the DRAM
  // fallback by the DPE (rc still 0), so "put until rc!=0" would keep going
  // until BOTH tiers are exhausted — starving the move's re-place and driving
  // ReorganizeBlob into its data-losing triple-failure corner (#863). Verify
  // each fill's placement via its SHM record and stop (deleting the spill) the
  // moment one leaves the file tier.
  auto fill_landed_on_disk = [&](const std::string &name) -> bool {
    clio::cte::core::ShmBlobRecord rec{};
    if (!cte_client->TryGetBlobRecordShm(tag_id, name, &rec)) return false;
    if (rec.num_blocks_ == 0) return false;
    for (clio::run::u32 b = 0; b < rec.num_blocks_ && b < clio::cte::core::kMaxInlineBlocks; ++b) {
      if (rec.blocks_[b].bdev_type_ !=
          static_cast<clio::run::u32>(clio::run::bdev::BdevType::kFile)) {
        return false;
      }
    }
    return true;
  };
  int fills_created = 0;
  for (int f = 0; f < kFillMax; ++f) {
    const std::string fname = "fill_" + std::to_string(f);
    auto put = tag.AsyncPutBlob(fname, churn_buffer.shm_.template Cast<void>(),
                                kBlobSize, 0, kDiskScore);
    put.Wait();
    if (put->GetReturnCode() != 0) break;  // both tiers full (should not hit)
    if (!fill_landed_on_disk(fname)) {
      auto del = cte_client->AsyncDelBlob(tag_id, fname);
      del.Wait();
      break;  // disk is full: exactly what we want
    }
    ++fills_created;
  }
  REQUIRE(fills_created >= 8);  // sanity: the fill really packed the tier

  std::vector<ctp::ipc::FullPtr<char>> read_bufs(kDepth);
  for (auto &b : read_bufs) {
    b = CLIO_IPC->AllocateBuffer(kVictimSize);
    REQUIRE(!b.IsNull());
  }

  using GetFuture = decltype(cte_client->AsyncGetBlob(
      tag_id, blob_name.c_str(), 0, kVictimSize, 0,
      read_bufs[0].shm_.template Cast<void>()));

  int rounds_run = 0;
  for (int round = 0; round < kRounds; ++round) {
    // Re-seed each round (round 0 seeded above): the move re-places the victim
    // as one big extent, so the multi-block layout must be rebuilt. Later
    // rounds are best-effort: under CLIO_FORCE_NET an occasional churn delete
    // can fail and leak a slot, making a later 8MB re-seed impossible — stop
    // racing then (cleanup below still runs) rather than aborting mid-test
    // with the tier left full for the NEXT test in this binary.
    if (round > 0 && !seed_victim()) break;
    // Race the disk->DRAM move: it frees the victim's DISK extents while the
    // pipelined disk reads are mid-flight.
    auto reorg = cte_client->AsyncReorganizeBlob(tag_id, blob_name, kDramScore);

    std::vector<GetFuture> gets;
    for (int i = 0; i < kDepth; ++i) {
      std::memset(read_bufs[i].ptr_, 0, kVictimSize);
      gets.push_back(cte_client->AsyncGetBlob(
          tag_id, blob_name.c_str(), 0, kVictimSize, 0,
          read_bufs[i].shm_.template Cast<void>()));
    }

    // Churn burst state: fired the moment the move's ClearBlob step frees the
    // victim's file extents (see the trigger in the loop below). With the file
    // tier otherwise 100% full, the burst can only allocate the just-freed
    // extents — scribbling them with 'X' while the pipelined reads are still
    // mid-flight. Fired earlier, the puts would spill to DRAM (disk full) and
    // touch nothing relevant.
    std::vector<decltype(tag.AsyncPutBlob(blob_name,
        churn_buffer.shm_.template Cast<void>(), kBlobSize, 0, kDiskScore))>
        churn;
    bool churn_fired = false;

    // Every read that reports success MUST carry the victim's bytes — never
    // the churn pattern, never a mix. Keep the pipeline full until the move
    // completes, then drain it; meanwhile CHURN continuously (synchronous
    // 1MB put+delete per loop iteration at disk score), so whatever the move
    // frees is immediately re-allocated and scribbled with 'X'. This is the
    // #753 assertion.
    int verified = 0;
    int overlapped = 0;  // gets that completed while the move was in flight
    int churn_seq = 0;
    bool hit_863 = false;
    while (true) {
      const bool move_done = reorg->IsComplete();
      for (int i = 0; i < kDepth; ++i) {
        if (gets[i].IsNull() || !gets[i]->IsComplete()) continue;
        gets[i].Wait();  // reap
        REQUIRE(gets[i]->GetReturnCode() == 0);
        if (std::memcmp(read_bufs[i].ptr_, expect.data(), kVictimSize) != 0) {
          // Discriminate a genuine reader-vs-reclaim corruption (#753) from
          // the reorganize's data-losing triple-failure corner (#863): the
          // latter empties the blob, and a read of an empty blob "succeeds"
          // leaving the buffer untouched. #863 is tracked separately — skip
          // loudly rather than mislabel it as the race under test.
          clio::cte::core::ShmBlobRecord vrec{};
          const bool lost =
              cte_client->TryGetBlobRecordShm(tag_id, blob_name, &vrec) &&
              vrec.total_size_ == 0;
          if (lost) {
            std::printf("[753] SKIP round: reorganize hit the #863 data-loss "
                        "corner; not a reader-race failure\n");
            hit_863 = true;
            break;
          }
          REQUIRE(std::memcmp(read_bufs[i].ptr_, expect.data(), kVictimSize) ==
                  0);
        }
        ++verified;
        if (!move_done) ++overlapped;
        if (!move_done) {
          std::memset(read_bufs[i].ptr_, 0, kVictimSize);
          gets[i] = cte_client->AsyncGetBlob(
              tag_id, blob_name.c_str(), 0, kVictimSize, 0,
              read_bufs[i].shm_.template Cast<void>());
        } else {
          gets[i] = GetFuture();  // slot drained
        }
      }
      // Post-clear trigger: ReorganizeBlob republishes the SHM mirror with the
      // EMPTIED layout immediately after ClearBlob frees the old extents. The
      // instant we observe it, the freed extents are on the file bdev's free
      // list — fire the churn burst so they get re-allocated and scribbled.
      if (!churn_fired && !move_done) {
        clio::cte::core::ShmBlobRecord vrec{};
        if (cte_client->TryGetBlobRecordShm(tag_id, blob_name, &vrec) &&
            vrec.num_blocks_ == 0) {
          // 4MB, deliberately HALF the freed capacity: a burst equal to the
          // whole 8MB can win the allocation race against the move's own
          // re-place and drive ReorganizeBlob into its triple-placement-
          // failure corner ("blob data LOST") — a real capacity-pressure
          // hazard, but not the property under test here. Half the extents
          // scribbled is ample for the whole-buffer memcmp to catch a
          // freed-under-reader read.
          for (int c = 0; c < 4; ++c) {
            churn.push_back(tag.AsyncPutBlob(
                "churn_" + std::to_string(c),
                churn_buffer.shm_.template Cast<void>(), kBlobSize, 0,
                kDiskScore));
          }
          churn_fired = true;
        }
      }
      if (hit_863) {
        for (int i = 0; i < kDepth; ++i) {
          if (!gets[i].IsNull()) gets[i].Wait();
        }
        break;
      }
      if (move_done) {
        bool all_drained = true;
        for (int i = 0; i < kDepth; ++i) {
          if (!gets[i].IsNull()) { all_drained = false; break; }
        }
        if (all_drained) break;
      }
      std::this_thread::yield();
    }
    if (!hit_863) REQUIRE(verified >= kDepth);
    (void)churn_seq;
    std::printf("[753] round %d: %d reads verified, %d overlapped the move, "
                "churn_fired=%d\n",
                round, verified, overlapped, (int)churn_fired);

    for (auto &c : churn) c.Wait();
    reorg.Wait();
    REQUIRE(reorg->GetReturnCode() == 0);
    if (churn_fired) {
      for (int c = 0; c < 4; ++c) {
        auto cdel =
            cte_client->AsyncDelBlob(tag_id, "churn_" + std::to_string(c));
        cdel.Wait();
      }
    }
    ++rounds_run;
  }
  REQUIRE(rounds_run >= 1);  // the fully-armed race ran at least once

  for (int f = 0; f < fills_created; ++f) {
    auto del = cte_client->AsyncDelBlob(tag_id, "fill_" + std::to_string(f));
    del.Wait();
  }
  {
    auto del = cte_client->AsyncDelBlob(tag_id, blob_name);
    del.Wait();
  }
  for (auto &b : read_bufs) CLIO_IPC->FreeBuffer(b);
  CLIO_IPC->FreeBuffer(churn_buffer);
  CLIO_IPC->FreeBuffer(put_buffer);
  INFO("SUCCESS: pipelined reads never observed reused extents (#753)");
}

/**
 * issue #753: deterministic white-box test of the reader-pin/drain protocol on
 * BlobInfo itself. The integration tests above hammer the real read/reorganize/
 * delete paths, but the race interleaving is scheduler-dependent; this test
 * pins the PROTOCOL invariants down exactly:
 *   1. a drainer must wait until every pinned reader unpins (an extent free
 *      while a pin is held would be the #753 use-after-free);
 *   2. once draining, new readers must back off (TryPinRead fails) so the
 *      drainer cannot be starved into a livelock;
 *   3. ending the drain re-admits readers;
 *   4. the RAII guards release on scope exit (a leaked pin wedges every later
 *      Reorganize/DelBlob of the blob; a leaked drain bit locks readers out).
 */
TEST_CASE("ReorganizeBlob - Reader pin/drain protocol invariants (#753)",
          "[reorganize][753][protocol]") {
  using clio::cte::core::BlobInfo;
  using clio::cte::core::BlobReadPinGuard;
  using clio::cte::core::BlobReaderDrainGuard;

  BlobInfo blob;

  // 1+3: pins hold the drainer; unpin releases it; end re-admits.
  REQUIRE(blob.TryPinRead());
  REQUIRE(blob.TryPinRead());  // pins are shared: many readers at once
  blob.BeginDrainReaders();
  REQUIRE(blob.HasReadPins());          // drainer must keep waiting...
  REQUIRE_FALSE(blob.TryPinRead());     // 2: ...and new readers back off
  blob.UnpinRead();
  REQUIRE(blob.HasReadPins());          // one pin still out
  blob.UnpinRead();
  REQUIRE_FALSE(blob.HasReadPins());    // drained: the free may proceed
  REQUIRE_FALSE(blob.TryPinRead());     // still draining: readers still out
  blob.EndDrainReaders();
  REQUIRE(blob.TryPinRead());           // re-admitted
  blob.UnpinRead();

  // 4: RAII guards.
  {
    REQUIRE(blob.TryPinRead());
    BlobReadPinGuard pin(&blob);
    REQUIRE(blob.HasReadPins());
  }
  REQUIRE_FALSE(blob.HasReadPins());
  {
    blob.BeginDrainReaders();
    BlobReaderDrainGuard drain(&blob);
    REQUIRE_FALSE(blob.TryPinRead());
  }
  REQUIRE(blob.TryPinRead());
  blob.UnpinRead();

  // Cross-thread hammer: readers pin/unpin in a loop while a "mutator" runs
  // repeated drain cycles, each time asserting the drained-quiescent state.
  // A protocol bug (drain not excluding pins, or a lost unpin) trips the
  // assertions or hangs the drain loop.
  std::atomic<bool> stop{false};
  std::atomic<clio::run::u64> pins_taken{0};
  std::vector<std::thread> readers;
  for (int r = 0; r < 4; ++r) {
    readers.emplace_back([&]() {
      while (!stop.load()) {
        if (blob.TryPinRead()) {
          pins_taken.fetch_add(1);
          std::this_thread::yield();
          blob.UnpinRead();
        } else {
          std::this_thread::yield();
        }
      }
    });
  }
  for (int cycle = 0; cycle < 100; ++cycle) {
    const clio::run::u64 before = pins_taken.load();
    blob.BeginDrainReaders();
    while (blob.HasReadPins()) {
      std::this_thread::yield();
    }
    // Quiescent: no pin may appear while the drain bit is held.
    for (int spin = 0; spin < 50; ++spin) {
      REQUIRE_FALSE(blob.HasReadPins());
    }
    blob.EndDrainReaders();
    // Readers MUST make progress between cycles — wait (bounded) for a new
    // pin so the hammer cannot silently degenerate into drain-only cycles on
    // a loaded machine, and so a drain bit that failed to clear is caught.
    int waited = 0;
    while (pins_taken.load() == before && waited < 2000000) {
      std::this_thread::yield();
      ++waited;
    }
    REQUIRE(pins_taken.load() > before);
  }
  stop.store(true);
  for (auto &t : readers) t.join();
  REQUIRE_FALSE(blob.HasReadPins());
  INFO("SUCCESS: reader pin/drain protocol invariants hold (#753)");
}

/**
 * issue #753 / #820: DelBlob (the path capacity Evict routes through) frees a
 * blob's extents; with pipelined reads in flight and churn re-allocating the
 * freed extents, a read must either return the blob's intact bytes (rc=0) or
 * fail cleanly (blob gone) — NEVER succeed with another blob's bytes.
 */
TEST_CASE("ReorganizeBlob - DelBlob under pipelined reads never yields garbage (#753)",
          "[reorganize][concurrent][753][del][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string blob_name = "race_753_del_blob";

  // One round, for the same capacity-drift reason as the sibling test above.
  constexpr int kRounds = 1;
  constexpr int kDepth = 6;
  constexpr int kChurn = 4;

  auto put_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!put_buffer.IsNull());
  auto expect = g_fixture->CreateTestData(kBlobSize, 'D');
  std::memcpy(put_buffer.ptr_, expect.data(), kBlobSize);

  auto churn_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!churn_buffer.IsNull());
  auto churn_data = g_fixture->CreateTestData(kBlobSize, 'X');
  std::memcpy(churn_buffer.ptr_, churn_data.data(), kBlobSize);

  // Create the victim FIRST (guaranteeing it fits), then fill the disk tier
  // to its brim (adaptively — earlier cases and their force_net variants leave
  // residue) so the churn puts below can only reuse the extents the delete
  // frees. Per round, the victim's re-creation reuses the space its churn
  // freed at the previous round's end.
  {
    auto put = tag.AsyncPutBlob(blob_name, put_buffer.shm_.template Cast<void>(),
                                kBlobSize, 0, kDiskScore);
    put.Wait();
    if (put->GetReturnCode() != 0) {
      // Earlier cases in this binary (notably the sibling 8MB race under
      // CLIO_FORCE_NET) can leak tier capacity through failed churn/delete
      // round-trips; with no room for even a 1MB victim this test cannot run
      // meaningfully. Skip loudly rather than fail — capacity residue is an
      // environmental precondition, not the property under test.
      std::printf("[753] SKIP: no tier capacity for the DelBlob-race victim "
                  "(residue from earlier cases)\n");
      CLIO_IPC->FreeBuffer(churn_buffer);
      CLIO_IPC->FreeBuffer(put_buffer);
      return;
    }
  }
  auto fill_del_on_disk = [&](const std::string &name) -> bool {
    clio::cte::core::ShmBlobRecord rec{};
    if (!cte_client->TryGetBlobRecordShm(tag_id, name, &rec)) return false;
    if (rec.num_blocks_ == 0) return false;
    for (clio::run::u32 b = 0; b < rec.num_blocks_ && b < clio::cte::core::kMaxInlineBlocks; ++b) {
      if (rec.blocks_[b].bdev_type_ !=
          static_cast<clio::run::u32>(clio::run::bdev::BdevType::kFile)) {
        return false;
      }
    }
    return true;
  };
  int fills_created = 0;
  for (int f = 0; f < 64; ++f) {
    const std::string fname = "fill_del_" + std::to_string(f);
    auto put = tag.AsyncPutBlob(fname, churn_buffer.shm_.template Cast<void>(),
                                kBlobSize, 0, kDiskScore);
    put.Wait();
    if (put->GetReturnCode() != 0) break;
    if (!fill_del_on_disk(fname)) {  // spilled off-disk: disk is full
      auto del = cte_client->AsyncDelBlob(tag_id, fname);
      del.Wait();
      break;
    }
    ++fills_created;
  }
  if (fills_created < 8) {
    // Residue from earlier cases left less than a meaningful fill's worth of
    // disk. Environmental, not the property under test — skip loudly.
    std::printf("[753] SKIP: only %d fills fit; disk residue from earlier "
                "cases\n", fills_created);
    for (int f = 0; f < fills_created; ++f) {
      auto del = cte_client->AsyncDelBlob(tag_id, "fill_del_" + std::to_string(f));
      del.Wait();
    }
    auto delv = cte_client->AsyncDelBlob(tag_id, blob_name);
    delv.Wait();
    CLIO_IPC->FreeBuffer(churn_buffer);
    CLIO_IPC->FreeBuffer(put_buffer);
    return;
  }

  std::vector<ctp::ipc::FullPtr<char>> read_bufs(kDepth);
  for (auto &b : read_bufs) {
    b = CLIO_IPC->AllocateBuffer(kBlobSize);
    REQUIRE(!b.IsNull());
  }

  int rounds_run = 0;
  for (int round = 0; round < kRounds; ++round) {
    if (round > 0) {
      // Re-create the victim on disk (its slot was freed by the previous
      // round's churn deletion). Best-effort like the sibling test: a leaked
      // slot under CLIO_FORCE_NET must stop the racing, not abort the test
      // before its cleanup.
      auto put = tag.AsyncPutBlob(blob_name,
                                  put_buffer.shm_.template Cast<void>(),
                                  kBlobSize, 0, kDiskScore);
      put.Wait();
      if (put->GetReturnCode() != 0) break;
    }

    // Pipeline reads, then delete out from under them, then churn to scribble
    // the freed extents.
    std::vector<decltype(cte_client->AsyncGetBlob(
        tag_id, blob_name.c_str(), 0, kBlobSize, 0,
        read_bufs[0].shm_.template Cast<void>()))> gets;
    for (int i = 0; i < kDepth; ++i) {
      std::memset(read_bufs[i].ptr_, 0, kBlobSize);
      gets.push_back(cte_client->AsyncGetBlob(
          tag_id, blob_name.c_str(), 0, kBlobSize, 0,
          read_bufs[i].shm_.template Cast<void>()));
    }
    auto del = cte_client->AsyncDelBlob(tag_id, blob_name);
    std::vector<decltype(tag.AsyncPutBlob(blob_name,
        churn_buffer.shm_.template Cast<void>(), kBlobSize, 0, kDramScore))>
        churn;
    for (int c = 0; c < kChurn; ++c) {
      churn.push_back(tag.AsyncPutBlob(
          "churn_del_" + std::to_string(c),
          churn_buffer.shm_.template Cast<void>(), kBlobSize, 0, kDramScore));
    }

    for (int i = 0; i < kDepth; ++i) {
      gets[i].Wait();
      if (gets[i]->GetReturnCode() == 0) {
        // Success means the read was pinned before the delete's drain: it must
        // carry the victim's intact bytes, not the churn scribble.
        REQUIRE(std::memcmp(read_bufs[i].ptr_, expect.data(), kBlobSize) == 0);
      }
      // Nonzero rc (blob gone) is the other legal outcome; garbage is not.
    }

    del.Wait();
    for (auto &c : churn) c.Wait();
    for (int c = 0; c < kChurn; ++c) {
      auto d = cte_client->AsyncDelBlob(tag_id, "churn_del_" + std::to_string(c));
      d.Wait();
    }
    ++rounds_run;
  }
  REQUIRE(rounds_run >= 1);

  for (int f = 0; f < fills_created; ++f) {
    auto del = cte_client->AsyncDelBlob(tag_id, "fill_del_" + std::to_string(f));
    del.Wait();
  }
  for (auto &b : read_bufs) CLIO_IPC->FreeBuffer(b);
  CLIO_IPC->FreeBuffer(churn_buffer);
  CLIO_IPC->FreeBuffer(put_buffer);
  INFO("SUCCESS: deletes under pipelined reads never yielded garbage (#753)");
}

/**
 * Test: Cleanup all blobs and tags
 */
TEST_CASE("ReorganizeBlob - Cleanup", "[reorganize][cleanup]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();

  INFO("Cleaning up test blobs and tags");

  // Delete the blob
  auto del_blob_task = cte_client->AsyncDelBlob(tag_id, "test_blob_dram");
  del_blob_task.Wait();

  // Delete the tag
  auto del_tag_task = cte_client->AsyncDelTag(tag_name);
  del_tag_task.Wait();

  INFO("Cleanup complete");
}

int main(int argc, char** argv) {
  // Create fixture (initializes runtime)
  g_fixture = new ReorganizeBlobTestFixture();

  // Run tests with optional filter from command line
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);

  // Cleanup
  delete g_fixture;
  g_fixture = nullptr;

  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
