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
 * Restart Integration Test
 *
 * Two-mode test program controlled by argv[1]:
 *   --put-blobs    Phase 1: create tag, put searchable blobs, flush
 *                  metadata+data, and verify keyword search
 *   --verify-blobs Phase 2: call RestartContainers, require blob recovery,
 *                  and verify the rebuilt keyword index returns the same
 *                  ranked results
 *
 * Designed to be orchestrated by test_restart.sh which starts/stops
 * the runtime between phases.
 */

#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_ctp/util/logging.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/clio_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static constexpr int kNumBlobs = 10;
static constexpr clio::run::u64 kBlobSize = 4096;
static const char* kTagName = "restart_test_tag";
static const char* kSearchTerm = "restartsearchtoken";

/**
 * Fill a fixed-size blob with a controlled keyword frequency.
 *
 * All blobs have the same token count, while later blob numbers contain more
 * copies of the search term. BM25 must therefore rank higher-numbered blobs
 * first, giving the restart test a deterministic expected order.
 *
 * @param buffer Destination buffer.
 * @param blob_number Zero-based blob number controlling keyword frequency.
 */
void FillSearchableBlob(char* buffer, int blob_number) {
  std::string payload;
  for (int i = 0; i <= blob_number; ++i) {
    payload += std::string(kSearchTerm) + " ";
  }
  for (int i = blob_number + 1; i < kNumBlobs; ++i) {
    payload += "backgroundword ";
  }
  memset(buffer, ' ', kBlobSize);
  memcpy(buffer, payload.data(), payload.size());
}

/**
 * Require the deterministic keyword-search result produced by PutBlobs.
 *
 * @param cte_client Client connected to the CTE pool.
 * @param phase Human-readable phase name used in diagnostics.
 * @return Zero when all blobs are returned in expected BM25 order.
 */
int VerifyKeywordSearch(clio::cte::core::Client& cte_client,
                        const char* phase) {
  auto search_task = cte_client.AsyncSemanticSearch(
      kTagName, "restart_blob_.*", kSearchTerm, kNumBlobs,
      clio::run::PoolQuery::Broadcast());
  search_task.Wait();
  if (search_task->GetReturnCode() != 0) {
    HLOG(kError, "{}: SemanticSearch failed, rc={}", phase,
         search_task->GetReturnCode());
    return 1;
  }
  if (search_task->results_.size() != kNumBlobs) {
    HLOG(kError, "{}: SemanticSearch returned {} results, expected {}", phase,
         search_task->results_.size(), kNumBlobs);
    return 1;
  }
  for (int i = 0; i < kNumBlobs; ++i) {
    const std::string expected_name =
        "restart_blob_" + std::to_string(kNumBlobs - i - 1);
    if (search_task->results_[i].blob_name_ != expected_name) {
      HLOG(kError, "{}: SemanticSearch result {} was '{}', expected '{}'",
           phase, i, search_task->results_[i].blob_name_, expected_name);
      return 1;
    }
  }
  HLOG(kInfo, "{}: SemanticSearch returned all {} blobs in expected order",
       phase, kNumBlobs);
  return 0;
}

/**
 * Store searchable blobs, verify the live index, and flush durable state.
 *
 * @return Zero on success.
 */
int PutBlobs() {
  // Connect to external runtime as client
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Phase 1: Failed to init client");
    return 1;
  }

  // Create CTE client bound to pool 512.0
  clio::cte::core::Client cte_client(clio::run::PoolId(512, 0));

  // Create or get tag
  auto tag_task = cte_client.AsyncGetOrCreateTag(kTagName);
  tag_task.Wait();
  clio::cte::core::TagId tag_id = tag_task->tag_id_;
  HLOG(kInfo, "Phase 1: Created tag '{}'", kTagName);

  // Put blobs with deterministic, distinct BM25 term frequencies.
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "restart_blob_" + std::to_string(i);

    // Allocate SHM buffer
    ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(kBlobSize);
    if (buf.IsNull()) {
      HLOG(kError, "Phase 1: Failed to allocate SHM buffer for blob {}", i);
      return 1;
    }

    FillSearchableBlob(buf.ptr_, i);

    // Convert to ShmPtr for API
    ctp::ipc::ShmPtr<> shm_ptr = buf.shm_.template Cast<void>();

    // Put blob
    auto put_task =
        cte_client.AsyncPutBlob(tag_id, blob_name, 0, kBlobSize, shm_ptr);
    put_task.Wait();

    if (put_task->GetReturnCode() != 0) {
      HLOG(kError, "Phase 1: PutBlob failed for blob {} rc={}", i,
           put_task->GetReturnCode());
      CLIO_IPC->FreeBuffer(buf);
      return 1;
    }
    CLIO_IPC->FreeBuffer(buf);
    HLOG(kInfo, "Phase 1: Put searchable blob '{}'", blob_name);
  }

  // Flush metadata (one-shot)
  HLOG(kInfo, "Phase 1: Flushing metadata...");
  auto flush_meta =
      cte_client.AsyncFlushMetadata(clio::run::PoolQuery::Local(), 0);
  flush_meta.Wait();
  HLOG(kInfo, "Phase 1: Metadata flush complete");

  // Flush data to the long-term file target.
  HLOG(kInfo, "Phase 1: Flushing data...");
  auto flush_data =
      cte_client.AsyncFlushData(clio::run::PoolQuery::Local(), 2, 0);
  flush_data.Wait();
  HLOG(kInfo, "Phase 1: Data flush complete");

  if (VerifyKeywordSearch(cte_client, "Phase 1") != 0) {
    return 1;
  }

  HLOG(kSuccess, "Phase 1: SUCCESS - {} blobs stored and flushed", kNumBlobs);
  return 0;
}

/**
 * Restart containers and require both blob and keyword-index recovery.
 *
 * @return Zero on success.
 */
int VerifyBlobs() {
  // Connect to external runtime as client
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "Phase 2: Failed to init client");
    return 1;
  }

  // Call RestartContainers via admin client
  HLOG(kInfo, "Phase 2: Calling RestartContainers...");
  clio::run::admin::Client admin_client(clio::run::kAdminPoolId);
  const auto restart_begin = std::chrono::steady_clock::now();
  auto restart_task =
      admin_client.AsyncRestartContainers(clio::run::PoolQuery::Local());
  restart_task.Wait();
  const double restart_rebuild_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - restart_begin)
          .count();
  HLOG(kInfo, "[RESTART_TEST] restart_rebuild_ms={}", restart_rebuild_ms);

  clio::run::u32 rc = restart_task->GetReturnCode();
  clio::run::u32 restarted = restart_task->containers_restarted_;
  HLOG(kInfo,
       "Phase 2: RestartContainers complete, rc={}, containers_restarted={}",
       rc, restarted);

  // Verify RestartContainers succeeded
  if (rc != 0) {
    HLOG(kError, "Phase 2: FAILED - RestartContainers returned error rc={}",
         rc);
    return 1;
  }
  if (restarted == 0) {
    HLOG(kError,
         "Phase 2: FAILED - No containers were restarted "
         "(restart config missing or unreadable)");
    return 1;
  }

  // Verify pool was recreated by connecting a CTE client
  clio::cte::core::Client cte_client(clio::run::PoolId(512, 0));

  // Verify we can create/get a tag on the restarted pool
  auto tag_task = cte_client.AsyncGetOrCreateTag(kTagName);
  tag_task.Wait();
  if (tag_task->GetReturnCode() != 0) {
    HLOG(kError,
         "Phase 2: FAILED - Could not create tag on restarted pool, rc={}",
         tag_task->GetReturnCode());
    return 1;
  }
  clio::cte::core::TagId tag_id = tag_task->tag_id_;
  HLOG(kInfo, "Phase 2: Tag '{}' accessible on restarted pool", kTagName);

  // Verify targets were re-registered by listing them
  auto targets_task =
      cte_client.AsyncListTargets(clio::run::PoolQuery::Local());
  targets_task.Wait();
  HLOG(kInfo, "Phase 2: ListTargets rc={}", targets_task->GetReturnCode());

  // Blob bytes are authoritative for rebuilding the in-memory keyword index.
  int recovered = 0;
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "restart_blob_" + std::to_string(i);

    ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(kBlobSize);
    if (buf.IsNull()) {
      HLOG(kError, "Phase 2: Failed to allocate buffer for '{}'", blob_name);
      return 1;
    }
    memset(buf.ptr_, 0, kBlobSize);
    ctp::ipc::ShmPtr<> shm_ptr = buf.shm_.template Cast<void>();

    auto get_task =
        cte_client.AsyncGetBlob(tag_id, blob_name, 0, kBlobSize, 0, shm_ptr);
    get_task.Wait();

    if (get_task->GetReturnCode() != 0) {
      HLOG(kError, "Phase 2: Failed to recover '{}', rc={}", blob_name,
           get_task->GetReturnCode());
      CLIO_IPC->FreeBuffer(buf);
      return 1;
    }

    std::vector<char> expected(kBlobSize);
    FillSearchableBlob(expected.data(), i);
    if (memcmp(buf.ptr_, expected.data(), kBlobSize) != 0) {
      HLOG(kError, "Phase 2: Recovered bytes differ for '{}'", blob_name);
      CLIO_IPC->FreeBuffer(buf);
      return 1;
    }
    CLIO_IPC->FreeBuffer(buf);
    ++recovered;
  }

  HLOG(kInfo, "Phase 2: Blob recovery: {}/{} recovered", recovered, kNumBlobs);

  if (VerifyKeywordSearch(cte_client, "Phase 2") != 0) {
    return 1;
  }

  HLOG(kSuccess,
       "Phase 2: SUCCESS - {} blobs and keyword index recovered after "
       "restarting {} container(s)",
       recovered, restarted);
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    HLOG(kError, "Usage: {} [--put-blobs|--verify-blobs]", argv[0]);
    return 1;
  }

  std::string mode(argv[1]);
  if (mode == "--put-blobs") {
    return PutBlobs();
  } else if (mode == "--verify-blobs") {
    return VerifyBlobs();
  } else {
    HLOG(kError, "Unknown mode: {}", mode);
    HLOG(kInfo, "Usage: {} [--put-blobs|--verify-blobs]", argv[0]);
    return 1;
  }
}
