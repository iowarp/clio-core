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
 * EVICTION TRIGGER TEST
 *
 * PutBlob makes room and retries once when placement fails, for puts whose
 * bytes are marked expendable. These cases cover:
 *
 *   1. a tier smaller than the workload keeps accepting writes
 *   2. data admitted after an eviction reads back byte-correct
 *   3. a write too large for an empty tier still fails, with a placement code
 *   4. growing one blob past the tier does not livelock on its own write lock
 *   5. a put without the flag never evicts, and is never evicted
 *   6. taking over an expendable blob is refused
 *   7. every EvictTask input field survives task duplication
 *
 * The tier is one small file-backed target. Assertions never assume an exact
 * usable capacity, only that writes keep succeeding and read back correctly.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

/* 4 MB blobs against a 64 MB tier: the workload outgrows the tier several
   times over, so eviction is on the critical path rather than incidental. */
static constexpr clio::run::u64 kBlobSize = 4 * 1024 * 1024;
static constexpr clio::run::u64 kTierBytes = 64 * 1024 * 1024;

/* The client-visible "could not place" band (PutBlob reports 10 + the raw
   allocator code). Kept here so a change to the band is a test failure. */
static constexpr int kPlaceRcFirst = 11;
static constexpr int kPlaceRcLast = 13;

class EvictTriggerFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  bool initialized_ = false;

  EvictTriggerFixture() {
    config_path_ = chi_test_data_dir() + "/evict_trigger_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/evict_trigger_storage.bin";
    Cleanup();
    CreateConfigFile();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    REQUIRE(WaitForTargets());
    initialized_ = true;
  }

  /* Block until the CTE has registered at least one storage target. Composing
     the bdev pools and registering them as CTE targets are separate steps, and
     a put issued in the gap fails against a tier that is about to exist. */
  bool WaitForTargets(int timeout_ms = 15000) {
    auto *cte_client = CLIO_CTE_CLIENT;
    if (cte_client == nullptr) return false;
    const int kStepMs = 100;
    for (int waited = 0; waited < timeout_ms; waited += kStepMs) {
      auto cap = cte_client->AsyncGetCapacity();
      cap.Wait();
      if (cap->GetReturnCode() == 0 && cap->total_capacity_ > 0) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kStepMs));
    }
    return false;
  }

  ~EvictTriggerFixture() { Cleanup(); }

  void Cleanup() {
    if (fs::exists(config_path_)) fs::remove(config_path_);
    if (fs::exists(file_storage_path_)) fs::remove(file_storage_path_);
  }

  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());
    config_file << R"(
# One file-backed tier, deliberately smaller than the workload so placement
# failure (and the make-room retry) is reached. File rather than RAM because a
# file bdev's usable capacity tracks its configured size closely, so the fill
# point does not depend on the host.
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
      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "64MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";
    config_file.close();
  }

  /* Write `size` bytes of `pattern` at `offset`. Returns PutBlob's return
     code rather than asserting, so the give-up cases can inspect it.
     `droppable` sets kCtePutDroppable. */
  int PutAt(const clio::cte::core::TagId &tag_id, const std::string &blob_name,
            clio::run::u64 offset, clio::run::u64 size, float score,
            char pattern, bool droppable) {
    auto *cte_client = CLIO_CTE_CLIENT;
    REQUIRE(cte_client != nullptr);
    auto shm_buffer = CLIO_IPC->AllocateBuffer(size);
    REQUIRE(!shm_buffer.IsNull());
    std::memset(shm_buffer.ptr_, pattern, size);
    const clio::run::u32 flags =
        droppable ? clio::cte::core::kCtePutDroppable : 0u;
    auto put_task = cte_client->AsyncPutBlob(
        tag_id, blob_name, offset, size, shm_buffer.shm_.template Cast<void>(),
        score, clio::cte::core::Context(), flags);
    put_task.Wait();
    const int rc = put_task->GetReturnCode();
    CLIO_IPC->FreeBuffer(shm_buffer);
    return rc;
  }
};

static EvictTriggerFixture *g_fixture = nullptr;

/* Read [offset, offset+size) and check every byte equals `pattern`. */
static bool BlobRegionMatches(clio::cte::core::Client *cte_client,
                              const clio::cte::core::TagId &tag_id,
                              const std::string &blob_name,
                              clio::run::u64 offset, clio::run::u64 size,
                              char pattern) {
  auto buf = CLIO_IPC->AllocateBuffer(size);
  if (buf.IsNull()) return false;
  std::memset(buf.ptr_, static_cast<char>(~pattern), size);
  auto get = cte_client->AsyncGetBlob(tag_id, blob_name, offset, size,
                                      /*flags=*/0,
                                      buf.shm_.template Cast<void>());
  get.Wait();
  bool ok = (get->GetReturnCode() == 0);
  if (ok) {
    const char *p = reinterpret_cast<const char *>(buf.ptr_);
    for (clio::run::u64 i = 0; i < size; ++i) {
      if (p[i] != pattern) {
        ok = false;
        break;
      }
    }
  }
  CLIO_IPC->FreeBuffer(buf);
  return ok;
}

/* Best-effort delete of "<prefix><i>" for i in [0, count). Already-evicted
   blobs are absent, so return codes are ignored. Cases share one tier, so a
   case that leaves it full makes the next one measure the wrong thing. */
static void DropBlobs(clio::cte::core::Client *cte_client,
                      const clio::cte::core::TagId &tag_id,
                      const std::string &prefix, int count) {
  for (int i = 0; i < count; ++i) {
    cte_client->AsyncDelBlob(tag_id, prefix + std::to_string(i)).Wait();
  }
}

/**
 * A workload several times the tier's size keeps being admitted.
 */
TEST_CASE("EvictTrigger - a full tier keeps accepting writes",
          "[evict_trigger][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("evict_trigger_fill_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  /* 3x the tier, so eviction has to run repeatedly rather than once. */
  const int kNumBlobs =
      static_cast<int>((kTierBytes / kBlobSize) * 3);
  for (int i = 0; i < kNumBlobs; ++i) {
    const std::string name = "fill_" + std::to_string(i);
    const char pattern = static_cast<char>('A' + (i % 26));
    const int rc = g_fixture->PutAt(tag_id, name, 0, kBlobSize, 0.5f, pattern,
                                     /*droppable=*/true);
    INFO("put " << name << " rc=" << rc);
    REQUIRE(rc == 0);
  }

  /* Only the most recent write is guaranteed resident; the rest may have been
     evicted to admit it. */
  const int last = kNumBlobs - 1;
  REQUIRE(BlobRegionMatches(cte_client, tag_id, "fill_" + std::to_string(last),
                            0, kBlobSize,
                            static_cast<char>('A' + (last % 26))));

  DropBlobs(cte_client, tag_id, "fill_", kNumBlobs);
}

/**
 * A single write larger than the whole tier cannot be satisfied by evicting
 * anything, so it still fails with a placement code.
 */
TEST_CASE("EvictTrigger - a write larger than the tier still fails",
          "[evict_trigger][noleak]") {
  REQUIRE(g_fixture != nullptr);
  clio::cte::core::Tag tag("evict_trigger_toobig_tag");

  const clio::run::u64 kTooBig = kTierBytes * 2;
  const int rc = g_fixture->PutAt(tag.GetTagId(), "too_big", 0, kTooBig, 0.5f, 'X',
                                   /*droppable=*/true);
  INFO("oversized put rc=" << rc);
  REQUIRE(rc >= kPlaceRcFirst);
  REQUIRE(rc <= kPlaceRcLast);
}

/**
 * Growing ONE blob past the tier. Every append after the tier fills triggers
 * eviction while that blob holds its own write lock, which eviction must skip.
 * A regression here hangs rather than failing an assertion, so the ctest
 * TIMEOUT is the real check.
 *
 * Also covers retry correctness: a partial extend keeps the blocks it placed,
 * so the retry must allocate only the remainder. Size and content checks prove
 * no extent was double-counted or lost.
 *
 * Growth stops once the blob owns the tier, since the only remaining candidate
 * is the blob itself.
 */
TEST_CASE("EvictTrigger - growing one blob past the tier does not livelock",
          "[evict_trigger][noleak]") {
  REQUIRE(g_fixture != nullptr);
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("evict_trigger_grow_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  /* A neighbour gives eviction something legal to take, so this exercises
     "skip the locked blob, take the other one". */
  REQUIRE(g_fixture->PutAt(tag_id, "victim", 0, kBlobSize, 0.1f, 'V',
                             /*droppable=*/true) == 0);

  /* Phase 1: grow to just under the tier. Reaching this means the neighbour
     was evicted, since victim + grower exceeds the tier. */
  const int kFitChunks = static_cast<int>(kTierBytes / kBlobSize) - 2;
  for (int i = 0; i < kFitChunks; ++i) {
    const char pattern = static_cast<char>('a' + (i % 26));
    const int rc = g_fixture->PutAt(tag_id, "grower", i * kBlobSize, kBlobSize,
                                    0.9f, pattern, /*droppable=*/true);
    INFO("append chunk " << i << " rc=" << rc);
    REQUIRE(rc == 0);
  }

  /* The blob spans exactly what was appended: no extent double-counted by a
     retried partial allocation, and none lost. */
  auto size_task = cte_client->AsyncGetBlobSize(tag_id, "grower");
  size_task.Wait();
  REQUIRE(size_task->GetReturnCode() == 0);
  REQUIRE(size_task->size_ ==
          static_cast<clio::run::u64>(kFitChunks) * kBlobSize);

  /* Every appended chunk reads back as written. */
  for (int i = 0; i < kFitChunks; ++i) {
    INFO("verify chunk " << i);
    REQUIRE(BlobRegionMatches(cte_client, tag_id, "grower",
                              static_cast<clio::run::u64>(i) * kBlobSize,
                              kBlobSize, static_cast<char>('a' + (i % 26))));
  }

  /* Phase 2: past the tier. The blob now owns nearly all of it, so the only
     candidate left is the blob being written, which is skipped. The put must
     fail rather than spin -- reaching this assertion is the livelock guard
     working. */
  int rc = 0;
  int extra = 0;
  for (; extra < 8; ++extra) {
    rc = g_fixture->PutAt(tag_id, "grower", (kFitChunks + extra) * kBlobSize,
                          kBlobSize, 0.9f, 'z', /*droppable=*/true);
    if (rc != 0) break;
  }
  INFO("growth stopped after " << extra << " extra chunk(s) with rc=" << rc);
  REQUIRE(rc >= kPlaceRcFirst);
  REQUIRE(rc <= kPlaceRcLast);

  /* grower owns nearly the whole tier; leaving it would starve later cases. */
  cte_client->AsyncDelBlob(tag_id, "grower").Wait();
  cte_client->AsyncDelBlob(tag_id, "victim").Wait();
}

/**
 * A put that does not declare its data expendable evicts nothing, so a full
 * tier refuses it and everything already written survives.
 *
 * Callers that store authoritative blobs rely on this: several fill until the
 * tier refuses and then operate on what they wrote.
 */
TEST_CASE("EvictTrigger - an ordinary put never evicts",
          "[evict_trigger][noleak]") {
  REQUIRE(g_fixture != nullptr);
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("evict_trigger_plain_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  /* Fill until the tier refuses. Without the flag this must terminate in a
     refusal rather than recycling forever. */
  const int kCap = static_cast<int>((kTierBytes / kBlobSize) * 3);
  int filled = 0;
  int rc = 0;
  for (; filled < kCap; ++filled) {
    rc = g_fixture->PutAt(tag_id, "plain_" + std::to_string(filled), 0,
                          kBlobSize, 0.5f, 'P', /*droppable=*/false);
    if (rc != 0) break;
  }
  INFO("plain fill refused after " << filled << " blob(s) with rc=" << rc);
  REQUIRE(rc >= kPlaceRcFirst);
  REQUIRE(rc <= kPlaceRcLast);
  REQUIRE(filled > 0);

  /* Everything that landed is still there. Nothing evicted it. */
  for (int i = 0; i < filled; ++i) {
    INFO("plain blob " << i << " must survive");
    REQUIRE(BlobRegionMatches(cte_client, tag_id,
                              "plain_" + std::to_string(i), 0, kBlobSize, 'P'));
  }

  DropBlobs(cte_client, tag_id, "plain_", filled);
}

/**
 * Eviction takes expendable blobs only. A droppable put that cannot be
 * satisfied out of other droppable blobs must fail rather than reclaim space
 * by destroying authoritative data.
 */
TEST_CASE("EvictTrigger - eviction never takes authoritative blobs",
          "[evict_trigger][noleak]") {
  REQUIRE(g_fixture != nullptr);
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("evict_trigger_mixed_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  /* Authoritative data at a LOW score, so a score-ranked eviction would reach
     for it first if it were eligible. */
  const int kKeepers = static_cast<int>(kTierBytes / kBlobSize) - 3;
  for (int i = 0; i < kKeepers; ++i) {
    const int rc = g_fixture->PutAt(tag_id, "keep_" + std::to_string(i), 0,
                                    kBlobSize, 0.1f, 'K', /*droppable=*/false);
    INFO("keeper " << i << " rc=" << rc);
    REQUIRE(rc == 0);
  }

  /* Push far more droppable data than the remainder holds. These keep
     succeeding by recycling among themselves, never by taking a keeper. */
  const int kCached = kKeepers * 2;
  for (int i = 0; i < kCached; ++i) {
    const int rc = g_fixture->PutAt(tag_id, "cache_" + std::to_string(i), 0,
                                    kBlobSize, 0.9f, 'C', /*droppable=*/true);
    INFO("droppable put " << i << " rc=" << rc);
    REQUIRE(rc == 0);
  }

  /* The most recent droppable write is resident and correct, so the recycling
     stored data rather than merely reporting success. */
  REQUIRE(BlobRegionMatches(cte_client, tag_id,
                            "cache_" + std::to_string(kCached - 1), 0,
                            kBlobSize, 'C'));

  /* Every authoritative blob is intact. */
  for (int i = 0; i < kKeepers; ++i) {
    INFO("keeper " << i << " must be untouched");
    REQUIRE(BlobRegionMatches(cte_client, tag_id, "keep_" + std::to_string(i),
                              0, kBlobSize, 'K'));
  }

  DropBlobs(cte_client, tag_id, "keep_", kKeepers);
  DropBlobs(cte_client, tag_id, "cache_", kCached);
}

/**
 * Taking over an expendable blob is refused; the reverse is allowed and leaves
 * the blob authoritative. See kCtePutDroppable.
 *
 * The second half is checked by making eviction try to take the blob.
 */
TEST_CASE("EvictTrigger - taking over a droppable blob is refused",
          "[evict_trigger][noleak]") {
  REQUIRE(g_fixture != nullptr);
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  clio::cte::core::Tag tag("evict_trigger_conflict_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  /* A cache copy, then an attempt to take ownership of the same name. */
  REQUIRE(g_fixture->PutAt(tag_id, "cache_blob", 0, kBlobSize, 0.5f, 'D',
                           /*droppable=*/true) == 0);
  const int rc = g_fixture->PutAt(tag_id, "cache_blob", 0, kBlobSize, 0.5f,
                                  'A', /*droppable=*/false);
  INFO("takeover rc=" << rc);
  REQUIRE(rc == static_cast<int>(clio::cte::core::kCteDroppabilityConflictRc));

  /* The cache copy still holds its own bytes and is still writable by its
     owner. */
  REQUIRE(BlobRegionMatches(cte_client, tag_id, "cache_blob", 0, kBlobSize,
                            'D'));
  REQUIRE(g_fixture->PutAt(tag_id, "cache_blob", 0, kBlobSize, 0.5f, 'E',
                           /*droppable=*/true) == 0);

  /* Reverse direction: allowed, and must leave the blob authoritative. */
  REQUIRE(g_fixture->PutAt(tag_id, "auth_blob", 0, kBlobSize, 0.5f, 'K',
                           /*droppable=*/false) == 0);
  REQUIRE(g_fixture->PutAt(tag_id, "auth_blob", 0, kBlobSize, 0.5f, 'L',
                           /*droppable=*/true) == 0);

  /* Push more droppable data than the tier holds. If the droppable put above
     had flipped the blob, this would consume it. */
  const int kPressure = static_cast<int>(kTierBytes / kBlobSize) * 2;
  for (int i = 0; i < kPressure; ++i) {
    REQUIRE(g_fixture->PutAt(tag_id, "press_" + std::to_string(i), 0, kBlobSize,
                             0.9f, 'C', /*droppable=*/true) == 0);
  }
  REQUIRE(BlobRegionMatches(cte_client, tag_id, "auth_blob", 0, kBlobSize,
                            'L'));

  DropBlobs(cte_client, tag_id, "press_", kPressure);
  cte_client->AsyncDelBlob(tag_id, "cache_blob").Wait();
  cte_client->AsyncDelBlob(tag_id, "auth_blob").Wait();
}

/**
 * Every EvictTask input field survives task duplication.
 *
 * Broadcast fan-out re-materialises the task per replica through
 * EvictTask::Copy, where a missing field silently reverts to its default.
 * Checked directly, since fan-out needs a multi-node pool to reach.
 */
TEST_CASE("EvictTrigger - EvictTask::Copy preserves every input field",
          "[evict_trigger][noleak]") {
  REQUIRE(g_fixture != nullptr);
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto *cte_client = CLIO_CTE_CLIENT;
  REQUIRE(ipc_manager != nullptr);
  REQUIRE(pool_manager != nullptr);
  REQUIRE(cte_client != nullptr);

  auto container = pool_manager->GetStaticContainer(cte_client->pool_id_).get();
  REQUIRE(container != nullptr);

  const float kScore = 0.25f;
  const clio::run::u64 kBytes = 4096;
  auto orig = ipc_manager->NewTask<clio::cte::core::EvictTask>(
      clio::run::CreateTaskId(), cte_client->pool_id_,
      clio::run::PoolQuery::Local(), kScore, kBytes,
      /*droppable_only=*/static_cast<clio::run::u32>(1));
  REQUIRE(!orig.IsNull());
  REQUIRE(orig->droppable_only_ == 1);

  clio::run::shared_ptr<clio::run::Task> task_ptr =
      orig.template Cast<clio::run::Task>();
  auto copied = container->NewCopyTask(clio::cte::core::Method::kEvict,
                                       task_ptr, /*deep=*/true);
  REQUIRE(!copied.IsNull());

  auto copied_evict = copied.template Cast<clio::cte::core::EvictTask>();
  REQUIRE(copied_evict->droppable_only_ == 1);
  REQUIRE(copied_evict->min_tier_score_ == kScore);
  REQUIRE(copied_evict->bytes_ == kBytes);

  copied.reset();
  orig.reset();
}

int main(int argc, char **argv) {
  g_fixture = new EvictTriggerFixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;
}
