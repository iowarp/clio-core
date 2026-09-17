/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

// GetResidency: "are these bytes actually PRESENT in the tier?" (VFD_VOL_PLAN §1)
//
// The distinction under test is the one a client cannot draw for itself. A miss
// in the shared-memory mirror conflates two opposite situations:
//
//   * the bytes do not exist  -> a hole; zeros ARE the correct answer, and no
//                                round trip is needed to produce them
//   * the bytes exist but are not reachable from here (file/remote/GPU tier, or
//     transformed) -> the RPC path is required
//
// Collapsing both to "false" is why CfsIo's SHM fast path abandons a whole
// request when any single page misses. So the assertions that matter are not
// "the call succeeded" but that ABSENT and PRESENT-BUT-NOT-DIRECT are reported
// as different things, and that absence is not an error.

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

const char *kTargetName = "residency_target";
constexpr clio::run::u64 kTargetSize = 32ULL * 1024 * 1024;
constexpr size_t kBlobSize = 4096;

class Fixture {
 public:
  bool initialized_ = false;
  Fixture() {
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(941, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), kTargetName, bdev_pool_id,
        clio::run::bdev::BdevType::kRam, kTargetSize);
    create.Wait();
    auto reg = cte->AsyncRegisterTarget(
        kTargetName, clio::run::bdev::BdevType::kRam, kTargetSize,
        clio::run::PoolQuery::Local(), bdev_pool_id);
    reg.Wait();
    REQUIRE(reg->GetReturnCode() == 0);
    initialized_ = true;
  }
};

Fixture *g_fixture = nullptr;

}  // namespace

TEST_CASE("Residency - absent blob reports absent, not an error",
          "[cte][residency]") {
  auto *cte = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag{std::string("residency_absent_tag")};

  auto r = cte->AsyncGetResidency(tag.GetTagId(), "no_such_blob", 0, 512);
  r.Wait();

  // rc 0: "it is not there" is an ANSWER. Reporting absence as a failure would
  // put the caller back to guessing, which is the defect this op removes.
  REQUIRE(r->GetReturnCode() == 0);
  REQUIRE(r->exists_ == 0);
  REQUIRE(r->present_bytes_ == 0);
  REQUIRE(r->direct_readable_ == 0);
}

TEST_CASE("Residency - a RAM-tier blob is present and directly readable",
          "[cte][residency]") {
  auto *cte = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag{std::string("residency_present_tag")};
  std::vector<char> buf(kBlobSize, 'r');

  auto put = cte->AsyncPutBlob(tag.GetTagId(), std::string("blob0"), 0,
                               kBlobSize, buf.data());
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);

  auto r = cte->AsyncGetResidency(tag.GetTagId(), "blob0", 0, kBlobSize);
  r.Wait();
  REQUIRE(r->GetReturnCode() == 0);
  REQUIRE(r->exists_ == 1);
  REQUIRE(r->present_bytes_ == kBlobSize);
  // The target registered above is node-local RAM, which is exactly the case
  // the SHM path can serve.
  REQUIRE(r->direct_readable_ == 1);
}

TEST_CASE("Residency - a range past the stored extent is short, not absent",
          "[cte][residency]") {
  auto *cte = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag{std::string("residency_short_tag")};
  std::vector<char> buf(kBlobSize, 's');

  auto put = cte->AsyncPutBlob(tag.GetTagId(), std::string("blob0"), 0,
                               kBlobSize, buf.data());
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);

  // Ask for twice what was written. The blob exists, so this is not a hole in
  // the "no such blob" sense -- but only the stored prefix is present, and the
  // caller needs the length to know where the hole starts.
  auto over = cte->AsyncGetResidency(tag.GetTagId(), "blob0", 0, kBlobSize * 2);
  over.Wait();
  REQUIRE(over->GetReturnCode() == 0);
  REQUIRE(over->exists_ == 1);
  REQUIRE(over->present_bytes_ == kBlobSize);

  // Starting entirely past the stored bytes: present_bytes_ 0 while exists_
  // stays 1. That pair is what tells a caller "this blob is real, but nothing
  // backs the range you asked about" -- distinct from the absent case above.
  auto past = cte->AsyncGetResidency(tag.GetTagId(), "blob0", kBlobSize * 4,
                                     kBlobSize);
  past.Wait();
  REQUIRE(past->GetReturnCode() == 0);
  REQUIRE(past->exists_ == 1);
  REQUIRE(past->present_bytes_ == 0);

  // size 0 means "to the end of what is stored", not "nothing".
  auto rest = cte->AsyncGetResidency(tag.GetTagId(), "blob0", 1024, 0);
  rest.Wait();
  REQUIRE(rest->GetReturnCode() == 0);
  REQUIRE(rest->present_bytes_ == kBlobSize - 1024);
}

int main(int argc, char **argv) {
  Fixture fixture;
  g_fixture = &fixture;
  int result = SimpleTest::run_all_tests(argc > 1 ? argv[1] : "");
  SIMPLE_TEST_PROCESS_EXIT(result);
  if (SimpleTest::g_test_finalize) SimpleTest::g_test_finalize();
  return result;
}
