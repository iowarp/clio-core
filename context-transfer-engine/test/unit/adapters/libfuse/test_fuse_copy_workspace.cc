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
 * FUSE ADAPTER — COPY /workspace TEST
 *
 * Walks /workspace (skipping build directories), copies every regular file
 * into CTE via the FUSE page-based helpers, then reads each file back and
 * verifies byte-for-byte equality with the original.
 *
 * This exercises the full FUSE data path at scale:
 *   CteGetOrCreateTag  -> tag creation (thousands of files)
 *   CtePutBlob         -> page-aligned writes (hundreds of MB)
 *   CteGetTagSize      -> metadata query
 *   CteGetBlob         -> page-aligned reads
 *   CteTagExists       -> tag existence check
 *   CteDelTag          -> cleanup
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "adapter/libfuse/fuse_cte.h"
#include "simple_test.h"

namespace fs = std::filesystem;
using namespace clio::cae::fuse;

// ============================================================================
// Constants
// ============================================================================

/** Root of the source tree to copy */
static const std::string kWorkspaceRoot = "/workspace";

/** Maximum individual file size we'll copy (skip huge files) */
static constexpr size_t kMaxFileSize = 4 * 1024 * 1024;  // 4 MB

/** Maximum total bytes to ingest (keeps RAM usage bounded) */
static constexpr size_t kMaxTotalBytes = 512 * 1024 * 1024;  // 512 MB

/** Directories to skip (build artifacts, git, caches) */
static const std::vector<std::string> kSkipDirs = {
    "build", "build_debug", "build_release", "build_socket",
    ".git", ".ppi-jarvis", ".jarvis-private", ".jarvis-shared",
    "__pycache__", "node_modules", ".cache", ".ssh-host",
};

// ============================================================================
// Helpers
// ============================================================================

/** Check whether a path component matches any skip directory */
static bool ShouldSkip(const fs::path &path) {
  for (const auto &component : path) {
    for (const auto &skip : kSkipDirs) {
      if (component == skip) return true;
    }
  }
  return false;
}

/** Read an entire file from disk into a byte vector */
static std::vector<char> ReadFileFromDisk(const fs::path &path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) return {};
  auto size = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0);
  std::vector<char> buf(size);
  ifs.read(buf.data(), static_cast<std::streamsize>(size));
  return buf;
}

/**
 * Write a buffer into CTE as page-indexed blobs under the given tag.
 * Mirrors the I/O path that cte_fuse_write() would take.
 */
static bool WriteFileToCte(const clio::cte::core::TagId &tag_id,
                           const char *data, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    size_t page = offset / kDefaultPageSize;
    size_t poff = offset % kDefaultPageSize;
    size_t chunk = std::min(kDefaultPageSize - poff, size - offset);
    if (!CtePutBlob(tag_id, std::to_string(page), data + offset, chunk, poff)) {
      return false;
    }
    offset += chunk;
  }
  return true;
}

/**
 * Read a file back from CTE page-by-page.
 * Mirrors the I/O path that cte_fuse_read() would take.
 */
static std::vector<char> ReadFileFromCte(const clio::cte::core::TagId &tag_id,
                                         size_t size) {
  std::vector<char> buf(size);
  size_t offset = 0;
  while (offset < size) {
    size_t page = offset / kDefaultPageSize;
    size_t poff = offset % kDefaultPageSize;
    size_t chunk = std::min(kDefaultPageSize - poff, size - offset);
    if (!CteGetBlob(tag_id, std::to_string(page), buf.data() + offset,
                    chunk, poff)) {
      return {};
    }
    offset += chunk;
  }
  return buf;
}

/** Collect eligible files under /workspace */
static std::vector<fs::path> CollectFiles() {
  std::vector<fs::path> files;
  size_t total = 0;
  std::error_code ec;

  for (auto it = fs::recursive_directory_iterator(
           kWorkspaceRoot, fs::directory_options::skip_permission_denied, ec);
       it != fs::recursive_directory_iterator(); ++it) {
    if (ec) { it.increment(ec); continue; }

    const auto &entry = *it;

    // Skip excluded directories
    if (entry.is_directory() && ShouldSkip(entry.path())) {
      it.disable_recursion_pending();
      continue;
    }

    if (!entry.is_regular_file(ec)) continue;
    if (ec) continue;

    auto fsize = entry.file_size(ec);
    if (ec || fsize == 0 || fsize > kMaxFileSize) continue;

    if (total + fsize > kMaxTotalBytes) break;

    files.push_back(entry.path());
    total += fsize;
  }

  return files;
}

// ============================================================================
// Test fixture
// ============================================================================

class CopyWorkspaceFixture {
 public:
  static inline bool g_initialized = false;

  CopyWorkspaceFixture() {
    if (!g_initialized) {
      bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      REQUIRE(success);

      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
      REQUIRE(success);

      auto *cte_client = CLIO_CTE_CLIENT;
      REQUIRE(cte_client != nullptr);
      cte_client->Init(clio::cte::core::kCtePoolId);

      clio::cte::core::CreateParams params;
      auto create_task = cte_client->AsyncCreate(
          clio::run::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
          clio::cte::core::kCtePoolId, params);
      create_task.Wait();
      REQUIRE(create_task->GetReturnCode() == 0);

      g_initialized = true;
      INFO("CTE runtime initialized for copy-workspace tests");
    }
  }
};

// ============================================================================
// Ingested file record
// ============================================================================

struct IngestedFile {
  fs::path disk_path;
  std::string tag_name;
  clio::cte::core::TagId tag_id;
  size_t size;
};

// ============================================================================
// Tests
// ============================================================================

TEST_CASE("FUSE Copy Workspace - Ingest and verify files",
          "[fuse][copy_workspace]") {
  auto *fixture = ctp::Singleton<CopyWorkspaceFixture>::GetInstance();
  (void)fixture;

  auto files = CollectFiles();
  REQUIRE(!files.empty());
  INFO("Collected " << files.size() << " files to copy");

  // Phase 1: Ingest — create tags and write page blobs
  std::vector<IngestedFile> ingested;

  size_t total_bytes = 0;
  for (const auto &path : files) {
    auto data = ReadFileFromDisk(path);
    if (data.empty()) continue;

    std::string tag_name = path.string();
    auto tag_id = CteGetOrCreateTag(tag_name);
    REQUIRE(!tag_id.IsNull());

    REQUIRE(WriteFileToCte(tag_id, data.data(), data.size()));
    ingested.push_back({path, tag_name, tag_id, data.size()});
    total_bytes += data.size();
  }

  REQUIRE(!ingested.empty());
  INFO("Ingested " << ingested.size() << " files, "
                    << total_bytes << " bytes total");

  // Phase 2: Verify — read back every file and compare byte-for-byte
  size_t verified = 0;
  for (const auto &entry : ingested) {
    // Check tag size matches
    size_t cte_size = CteGetTagSize(entry.tag_id);
    REQUIRE(cte_size == entry.size);

    // Read back from CTE
    auto cte_data = ReadFileFromCte(entry.tag_id, entry.size);
    REQUIRE(cte_data.size() == entry.size);

    // Read original from disk
    auto disk_data = ReadFileFromDisk(entry.disk_path);
    REQUIRE(disk_data.size() == entry.size);

    // Byte-for-byte comparison
    REQUIRE(memcmp(cte_data.data(), disk_data.data(), entry.size) == 0);
    ++verified;
  }

  INFO("Verified " << verified << " files byte-for-byte");
  REQUIRE(verified == ingested.size());

  // Phase 3: Spot-check tag existence via CteTagExists
  size_t spot_check = std::min(ingested.size(), static_cast<size_t>(20));
  for (size_t i = 0; i < spot_check; ++i) {
    REQUIRE(CteTagExists(ingested[i].tag_name));
  }
  INFO("Spot-checked " << spot_check << " tags via CteTagExists");

  // Phase 4: Cleanup — delete all tags
  size_t deleted = 0;
  for (const auto &entry : ingested) {
    CteDelTag(entry.tag_name);
    ++deleted;
  }
  INFO("Cleaned up " << deleted << " tags");

  // Spot-check that a few tags are actually gone
  size_t check_count = std::min(ingested.size(), static_cast<size_t>(10));
  for (size_t i = 0; i < check_count; ++i) {
    REQUIRE_FALSE(CteTagExists(ingested[i].tag_name));
  }
  INFO("Post-cleanup spot-check passed");
}

SIMPLE_TEST_MAIN()
