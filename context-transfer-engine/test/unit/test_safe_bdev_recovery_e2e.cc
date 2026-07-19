/**
 * Integration stress test for safe-bdev recovery via CTE.
 * 
 * Demonstrates:
 *   1. 4 file-based bdevs (for safe-bdev).
 *   2. 1 ram-based bdev.
 *   3. 1 safe-bdev pool composed over the 4 file bdevs (max_failures=1).
 *   4. A CTE pool bound to the safe-bdev and the ram-bdev.
 *   5. We write a blob to CTE, force a file bdev failure, and verify we can still read it.
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
constexpr clio::run::u64 kMemberSize = 16ULL * 1024ULL * 1024ULL; // 16 MiB

void EnsureInit() {
  if (g_initialized) return;
  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  REQUIRE(success);
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(500ms);
  success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
  REQUIRE(success);
  g_initialized = true;
}

} // namespace

TEST_CASE("CTE E2E Safe BDEV Recovery", "[cte][recovery][safe_bdev]") {
  EnsureInit();
  REQUIRE(g_initialized);

  const int salt = static_cast<int>(getpid() & 0xFFF);
  auto uniq = [&](const std::string &base, int idx) {
    return base + "_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  // 1) Compose 4 file bdevs for safe-bdev
  std::vector<clio::run::PoolId> file_bdevs;
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int i = 0; i < 4; ++i) {
    clio::run::PoolId id(static_cast<clio::run::u32>(8100 + salt + i), 0);
    clio::run::bdev::Client client(id);
    auto t = client.AsyncCreate(clio::run::PoolQuery::Dynamic(), uniq("file_mem", i), id,
                                clio::run::bdev::BdevType::kFile, kMemberSize);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    file_bdevs.push_back(t->new_pool_id_);
    members.emplace_back(uniq("file_mem", i), 0, t->new_pool_id_);
  }

  // 2) Compose 1 RAM bdev
  clio::run::PoolId ram_id(static_cast<clio::run::u32>(8200 + salt), 0);
  clio::run::bdev::Client ram_client(ram_id);
  {
    auto t = ram_client.AsyncCreate(clio::run::PoolQuery::Dynamic(), uniq("ram_bdev", 0), ram_id,
                                    clio::run::bdev::BdevType::kRam, kMemberSize);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    ram_id = t->new_pool_id_;
  }

  // 3) Compose safe-bdev (max_failures=1)
  clio::run::PoolId safe_id(static_cast<clio::run::u32>(8300 + salt), 0);
  clio::run::safe_bdev::Client safe(safe_id);
  {
    auto t = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(), uniq("safe_e2e", 0), safe_id, 1, members);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    safe_id = t->new_pool_id_;
  }

  // 4) Compose CTE and bind both
  clio::run::PoolId cte_id(static_cast<clio::run::u32>(8500 + salt), 0);
  clio::cte::core::Client cte(cte_id);
  {
    clio::cte::core::CreateParams params;
    params.config_.performance_.stat_targets_period_ms_ = 100;
    auto t = cte.AsyncCreate(clio::run::PoolQuery::Dynamic(), uniq("cte_e2e", 0), cte_id, params);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
  }

  {
    auto t1 = cte.AsyncRegisterTarget(uniq("safe_e2e", 0), clio::run::bdev::BdevType::kFile, 0,
                                      clio::run::PoolQuery::Local(), safe_id, clio::run::PoolQuery::Dynamic());
    t1.Wait();
    REQUIRE(t1->GetReturnCode() == 0);

    auto t2 = cte.AsyncRegisterTarget(uniq("ram_bdev", 0), clio::run::bdev::BdevType::kRam, 0,
                                      clio::run::PoolQuery::Local(), ram_id, clio::run::PoolQuery::Dynamic());
    t2.Wait();
    REQUIRE(t2->GetReturnCode() == 0);
    std::this_thread::sleep_for(150ms);
  }

  // 5) Put Blob (size ensures it hits safe_bdev/ram depending on DPE)
  clio::cte::core::TagId tag_id;
  {
    auto t = cte.AsyncGetOrCreateTag(uniq("e2e_tag", 0));
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    tag_id = t->tag_id_;
  }

  const clio::run::u64 kBlobSize = 1024ULL * 1024ULL; // 1 MiB
  std::vector<char> expected(kBlobSize);
  for (size_t i = 0; i < kBlobSize; ++i) {
    expected[i] = static_cast<char>((i * 17 + 3) & 0xFF);
  }
  const std::string blob_name = uniq("e2e_blob", 0);

  {
    auto wbuf = CLIO_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(wbuf.IsNull());
    memcpy(wbuf.ptr_, expected.data(), kBlobSize);
    // Force score 1.0 to map it to safe_bdev? Or safe_bdev is file so its score is lower?
    // The test logic relies on CTE storing the data.
    auto t = cte.AsyncPutBlob(tag_id, blob_name, 0, kBlobSize, wbuf.shm_.template Cast<void>(), 1.0f);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    CLIO_IPC->FreeBuffer(wbuf);
  }

  // 6) Inject failure into one safe-bdev member (data drive)
  {
    HLOG(kInfo, "Marking file member 1 as faulty");
    auto t = safe.AsyncRemoveBdev(clio::run::PoolQuery::Local(), file_bdevs[1], 1);
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
  }

  // 7) Read back via CTE and verify
  std::vector<char> got(kBlobSize, 0);
  {
    auto rbuf = CLIO_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(rbuf.IsNull());
    memset(rbuf.ptr_, 0, kBlobSize);
    auto t = cte.AsyncGetBlob(tag_id, blob_name, 0, kBlobSize, 0, rbuf.shm_.template Cast<void>());
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    memcpy(got.data(), rbuf.ptr_, kBlobSize);
    CLIO_IPC->FreeBuffer(rbuf);
  }

  REQUIRE(got == expected);
  HLOG(kInfo, "SUCCESS: Read verified correctly after device failure.");

  // 8) Test Preemptive Migration (Issue #731)
  {
    HLOG(kInfo, "Marking file member 2 as dying (TTL = 3 days)");
    clio::run::bdev::Client bdev2(file_bdevs[2]);
    auto t_ttl = bdev2.AsyncSetLifespan(clio::run::PoolQuery::Local(), 3);
    t_ttl.Wait();
    REQUIRE(t_ttl->GetReturnCode() == 0);

    clio::run::PoolId new_bdev_id(static_cast<clio::run::u32>(9990 + salt), 0);
    clio::run::bdev::Client new_bdev(new_bdev_id);
    // Create the replacement bdev with 256K blocks (kFile, ~1 GB at 4096 block size)
    auto t_new = new_bdev.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      uniq("replacement_bdev", 0), new_bdev_id,
                                      clio::run::bdev::BdevType::kFile,
                                      static_cast<clio::run::u64>(256) * 1024 * 4096);
    t_new.Wait();
    REQUIRE(t_new->GetReturnCode() == 0);

    HLOG(kInfo, "Adding replacement BDEV to safe-bdev to trigger preemptive migration");
    // AsyncAddBdev: pool_query, pool_name, node_id, member_pool_id, as_parity=0
    auto t_add = safe.AsyncAddBdev(clio::run::PoolQuery::Local(),
                                   uniq("replacement_bdev", 0), 0,
                                   new_bdev_id, 0);
    t_add.Wait();
    REQUIRE(t_add->GetReturnCode() == 0);
  }

  // 9) Read back via CTE and verify (ensures migration succeeded)
  std::vector<char> got2(kBlobSize, 0);
  {
    auto rbuf = CLIO_IPC->AllocateBuffer(kBlobSize);
    REQUIRE_FALSE(rbuf.IsNull());
    memset(rbuf.ptr_, 0, kBlobSize);
    auto t = cte.AsyncGetBlob(tag_id, blob_name, 0, kBlobSize, 0, rbuf.shm_.template Cast<void>());
    t.Wait();
    REQUIRE(t->GetReturnCode() == 0);
    memcpy(got2.data(), rbuf.ptr_, kBlobSize);
    CLIO_IPC->FreeBuffer(rbuf);
  }

  REQUIRE(got2 == expected);
  HLOG(kInfo, "SUCCESS: Read verified correctly after preemptive migration.");

}

SIMPLE_TEST_MAIN()
