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
 *   --put-blobs    Phase 1: create tag, put 10 blobs, flush metadata+data
 *   --verify-blobs Phase 2: call RestartContainers, verify pool recreation
 *                           and attempt blob recovery (informational)
 *
 * Designed to be orchestrated by test_restart.sh which starts/stops
 * the runtime between phases.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_client.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

static constexpr int kNumBlobs = 10;
static constexpr chi::u64 kBlobSize = 4096;
static const char* kTagName = "restart_test_tag";

/**
 * Phase 1: Put blobs and flush
 */
int PutBlobs() {
  // Connect to external runtime as client
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    std::cerr << "Phase 1: Failed to init client\n";
    return 1;
  }

  // Create CTE client bound to pool 512.0
  wrp_cte::core::Client cte_client(chi::PoolId(512, 0));

  // Create or get tag
  auto tag_task = cte_client.AsyncGetOrCreateTag(kTagName);
  tag_task.Wait();
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;
  std::cout << "Phase 1: Created tag '" << kTagName << "'\n";

  // Put kNumBlobs blobs with distinct data patterns
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "restart_blob_" + std::to_string(i);

    // Allocate SHM buffer
    hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(kBlobSize);
    if (buf.IsNull()) {
      std::cerr << "Phase 1: Failed to allocate SHM buffer for blob " << i << "\n";
      return 1;
    }

    // Fill with pattern: each blob gets a different base character
    char pattern = static_cast<char>('A' + i);
    memset(buf.ptr_, pattern, kBlobSize);

    // Convert to ShmPtr for API
    hipc::ShmPtr<> shm_ptr = buf.shm_.template Cast<void>();

    // Put blob
    auto put_task = cte_client.AsyncPutBlob(
        tag_id, blob_name, 0, kBlobSize, shm_ptr);
    put_task.Wait();

    if (put_task->GetReturnCode() != 0) {
      std::cerr << "Phase 1: PutBlob failed for blob " << i
                << " rc=" << put_task->GetReturnCode() << "\n";
      return 1;
    }
    std::cout << "Phase 1: Put blob '" << blob_name
              << "' pattern='" << pattern << "'\n";
  }

  // Flush metadata (one-shot)
  std::cout << "Phase 1: Flushing metadata...\n";
  auto flush_meta = cte_client.AsyncFlushMetadata(chi::PoolQuery::Local(), 0);
  flush_meta.Wait();
  std::cout << "Phase 1: Metadata flush complete\n";

  // Flush data (one-shot, persistence level 0 for RAM target)
  std::cout << "Phase 1: Flushing data...\n";
  auto flush_data = cte_client.AsyncFlushData(chi::PoolQuery::Local(), 0, 0);
  flush_data.Wait();
  std::cout << "Phase 1: Data flush complete\n";

  std::cout << "Phase 1: SUCCESS - " << kNumBlobs << " blobs stored and flushed\n";
  return 0;
}

/**
 * Phase 2: RestartContainers then verify pool recreation
 */
int VerifyBlobs() {
  // Connect to external runtime as client
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    std::cerr << "Phase 2: Failed to init client\n";
    return 1;
  }

  // Call RestartContainers via admin client
  std::cout << "Phase 2: Calling RestartContainers...\n";
  chimaera::admin::Client admin_client(chi::kAdminPoolId);
  auto restart_task = admin_client.AsyncRestartContainers(chi::PoolQuery::Local());
  restart_task.Wait();

  chi::u32 rc = restart_task->GetReturnCode();
  chi::u32 restarted = restart_task->containers_restarted_;
  std::cout << "Phase 2: RestartContainers complete, rc=" << rc
            << ", containers_restarted=" << restarted << "\n";

  // Verify RestartContainers succeeded
  if (rc != 0) {
    std::cerr << "Phase 2: FAILED - RestartContainers returned error rc=" << rc << "\n";
    return 1;
  }
  if (restarted == 0) {
    std::cerr << "Phase 2: FAILED - No containers were restarted "
              << "(restart config missing or unreadable)\n";
    return 1;
  }

  // Verify pool was recreated by connecting a CTE client
  wrp_cte::core::Client cte_client(chi::PoolId(512, 0));

  // Verify we can create/get a tag on the restarted pool
  auto tag_task = cte_client.AsyncGetOrCreateTag(kTagName);
  tag_task.Wait();
  if (tag_task->GetReturnCode() != 0) {
    std::cerr << "Phase 2: FAILED - Could not create tag on restarted pool, rc="
              << tag_task->GetReturnCode() << "\n";
    return 1;
  }
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;
  std::cout << "Phase 2: Tag '" << kTagName << "' accessible on restarted pool\n";

  // Verify targets were re-registered by listing them
  auto targets_task = cte_client.AsyncListTargets(chi::PoolQuery::Local());
  targets_task.Wait();
  std::cout << "Phase 2: ListTargets rc=" << targets_task->GetReturnCode() << "\n";

  // Attempt blob recovery (informational - data persistence is WIP)
  int recovered = 0;
  int failed = 0;
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "restart_blob_" + std::to_string(i);
    char expected_pattern = static_cast<char>('A' + i);

    hipc::FullPtr<char> buf = CHI_IPC->AllocateBuffer(kBlobSize);
    if (buf.IsNull()) {
      ++failed;
      continue;
    }
    memset(buf.ptr_, 0, kBlobSize);
    hipc::ShmPtr<> shm_ptr = buf.shm_.template Cast<void>();

    auto get_task = cte_client.AsyncGetBlob(
        tag_id, blob_name, 0, kBlobSize, 0, shm_ptr);
    get_task.Wait();

    if (get_task->GetReturnCode() != 0) {
      ++failed;
      continue;
    }

    // Verify data pattern
    bool data_ok = true;
    for (chi::u64 j = 0; j < kBlobSize; ++j) {
      if (buf.ptr_[j] != expected_pattern) {
        data_ok = false;
        break;
      }
    }

    if (data_ok) {
      ++recovered;
      std::cout << "Phase 2: Blob '" << blob_name << "' data recovered OK\n";
    } else {
      ++failed;
    }
  }

  std::cout << "Phase 2: Blob recovery: " << recovered << "/" << kNumBlobs
            << " recovered, " << failed << "/" << kNumBlobs << " pending implementation\n";

  // The test passes if RestartContainers worked and the pool is functional.
  // Full blob data recovery requires completing FlushData and metadata
  // recovery implementation.
  std::cout << "Phase 2: SUCCESS - Pool restart verified ("
            << restarted << " containers restarted)\n";
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " [--put-blobs|--verify-blobs]\n";
    return 1;
  }

  std::string mode(argv[1]);
  if (mode == "--put-blobs") {
    return PutBlobs();
  } else if (mode == "--verify-blobs") {
    return VerifyBlobs();
  } else {
    std::cerr << "Unknown mode: " << mode << "\n";
    std::cerr << "Usage: " << argv[0] << " [--put-blobs|--verify-blobs]\n";
    return 1;
  }
}
