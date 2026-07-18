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
 * CTE → existing safe-bdev pool binding test (issue #543 follow-up).
 *
 * Proves that the Context Transfer Engine can use an ALREADY-EXISTING pool as
 * a storage target instead of always creating its own bdev. Specifically:
 *
 *   1. A RAM bdev member pool is composed in-process.
 *   2. A safe-bdev pool ("safe0") is composed OVER that member.
 *   3. A CTE pool is created, and its storage target is BOUND to the existing
 *      safe-bdev pool via AsyncRegisterTarget(..., attach_existing=1). This is
 *      exactly what the config-driven Create loop emits when a storage entry
 *      sets `existing_pool_id` (see core_runtime.cc Create()).
 *   4. PutBlob writes a known buffer; GetBlob reads it back; the bytes must
 *      match — proving CTE I/O flows through the safe-bdev target rather than a
 *      CTE-owned bdev.
 *
 * A second test asserts the config parser accepts `existing_pool_id` /
 * `existing_pool_module` and marks the target with HasExistingPool().
 */

#ifndef _WIN32
#include <unistd.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

using namespace std::chrono_literals;

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/safe_bdev/safe_bdev_client.h>
#include <clio_runtime/safe_bdev/safe_bdev_tasks.h>

#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_config.h>
#include <clio_cte/core/core_runtime.h>
#include <clio_cte/core/core_tasks.h>

namespace {

bool g_initialized = false;

// 4 MiB RAM member comfortably holds the safe-bdev reserved EC region.
constexpr clio::run::u64 kMemberRamSize = 4ULL * 1024ULL * 1024ULL;

/** Initialize Chimaera + CTE client once for the suite. */
void EnsureInit() {
  if (g_initialized) {
    return;
  }
  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  REQUIRE(success);
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(500ms);

  success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
  REQUIRE(success);
  g_initialized = true;
}

}  // namespace

// ===========================================================================
// Config parse: existing_pool_id makes path/bdev_type/capacity optional.
// ===========================================================================
TEST_CASE("CTE config parses existing_pool_id",
          "[cte][existing_pool][config]") {
  const std::string yaml =
      "storage:\n"
      "  - path: \"safe0\"\n"
      "    existing_pool_id: \"350.0\"\n"
      "    existing_pool_module: \"clio_safe_bdev\"\n"
      "    score: 1.0\n"
      "dpe:\n"
      "  dpe_type: \"max_bw\"\n";

  clio::cte::core::Config config;
  REQUIRE(config.LoadFromString(yaml));
  REQUIRE(config.storage_.devices_.size() == 1);

  const auto &dev = config.storage_.devices_[0];
  REQUIRE(dev.HasExistingPool());
  REQUIRE(dev.existing_pool_id_.major_ == 350);
  REQUIRE(dev.existing_pool_id_.minor_ == 0);
  REQUIRE(dev.existing_pool_module_ == "clio_safe_bdev");
  REQUIRE(dev.score_ == 1.0f);
  HLOG(kInfo,
       "CTE config: storage target bound to existing pool {}.{} (module={})",
       dev.existing_pool_id_.major_, dev.existing_pool_id_.minor_,
       dev.existing_pool_module_);
}

// ===========================================================================
// End-to-end: CTE binds an existing safe-bdev pool, PutBlob/GetBlob round-trips.
// ===========================================================================
TEST_CASE("CTE binds existing safe-bdev pool and round-trips a blob",
          "[cte][existing_pool][safe_bdev]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int salt = static_cast<int>(getpid() & 0xFFF);
  auto uniq = [&](const std::string &base, int idx) {
    return base + "_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  // --- (1) Compose a RAM member bdev. ---
  clio::run::PoolId member_id(static_cast<clio::run::u32>(7100 + salt), 0);
  clio::run::bdev::Client member_client(member_id);
  {
    auto t = member_client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                       uniq("cte_safe_member", 0), member_id,
                                       clio::run::bdev::BdevType::kRam,
                                       kMemberRamSize);
    t.Wait();
    member_client.pool_id_ = t->new_pool_id_;
    REQUIRE(t->GetReturnCode() == 0);
    member_id = member_client.pool_id_;
  }

  // --- (2) Compose a safe-bdev pool OVER that member. ---
  clio::run::PoolId safe_id(static_cast<clio::run::u32>(7300 + salt), 0);
  clio::run::safe_bdev::Client safe(safe_id);
  {
    std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
    members.emplace_back(uniq("cte_safe_member", 0), /*node_id=*/0, member_id);
    auto t = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                              uniq("cte_safe0", 0), safe_id,
                              /*max_failures=*/1, members);
    t.Wait();
    safe.pool_id_ = t->new_pool_id_;
    REQUIRE(t->GetReturnCode() == 0);
    safe_id = safe.pool_id_;
  }
  HLOG(kInfo, "CTE existing-pool test: safe-bdev pool ready at ({},{})",
       safe_id.major_, safe_id.minor_);

  // --- (3) Create a CTE pool and BIND its target to the existing safe-bdev
  //         pool (attach_existing=1) — no CTE-owned bdev is created. ---
  clio::run::PoolId cte_id(static_cast<clio::run::u32>(7500 + salt), 0);
  clio::cte::core::Client cte(cte_id);
  {
    clio::cte::core::CreateParams params;
    auto t = cte.AsyncCreate(clio::run::PoolQuery::Dynamic(), uniq("cte_existing", 0),
                             cte_id, params);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
  }

  {
    // attach_existing=1 binds the bdev client to safe_id WITHOUT AsyncCreate;
    // capacity 0 -> the handler uses the pool's GetStats remaining space.
    auto t = cte.AsyncRegisterTarget(
        uniq("safe0", 0), clio::run::bdev::BdevType::kFile, /*total_size=*/0,
        clio::run::PoolQuery::Local(), safe_id, clio::run::PoolQuery::Dynamic(),
        /*attach_existing=*/1);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    HLOG(kInfo,
         "CTE existing-pool test: CTE target bound to existing safe-bdev pool "
         "({},{}) [attach_existing=1]",
         safe_id.major_, safe_id.minor_);
  }

  // --- (4) PutBlob a known buffer, GetBlob it back, assert bytes match. ---
  clio::cte::core::TagId tag_id;
  {
    auto t = cte.AsyncGetOrCreateTag(uniq("cte_safe_tag", 0));
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    tag_id = t->tag_id_;
  }

  const clio::run::u64 kBlobSize = 256ULL * 1024ULL;  // 256 KiB (multi-chunk stripe)
  std::vector<char> expected(kBlobSize);
  for (size_t i = 0; i < kBlobSize; ++i) {
    expected[i] = static_cast<char>((i * 31 + 7) & 0xFF);
  }
  const std::string blob_name = uniq("cte_safe_blob", 0);

  // PutBlob.
  {
    auto wbuf = CLIO_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(wbuf.IsNull());
    memcpy(wbuf.ptr_, expected.data(), kBlobSize);
    auto t = cte.AsyncPutBlob(tag_id, blob_name, /*offset=*/0, kBlobSize,
                              wbuf.shm_.template Cast<void>(), /*score=*/1.0f);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    CLIO_IPC->FreeBuffer(wbuf);
  }

  // GetBlob.
  std::vector<char> got(kBlobSize, 0);
  {
    auto rbuf = CLIO_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(rbuf.IsNull());
    memset(rbuf.ptr_, 0, kBlobSize);
    auto t = cte.AsyncGetBlob(tag_id, blob_name, /*offset=*/0, kBlobSize,
                              /*flags=*/0, rbuf.shm_.template Cast<void>());
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    memcpy(got.data(), rbuf.ptr_, kBlobSize);
    CLIO_IPC->FreeBuffer(rbuf);
  }

  REQUIRE(got == expected);
  HLOG(kInfo,
       "CTE existing-pool test: blob '{}' ({} bytes) round-tripped through "
       "bound safe-bdev pool ({},{}) OK",
       blob_name, kBlobSize, safe_id.major_, safe_id.minor_);
}

SIMPLE_TEST_MAIN()
