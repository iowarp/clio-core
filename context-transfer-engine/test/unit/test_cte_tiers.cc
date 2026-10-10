/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * test_cte_tiers — the cte_tiers report (core/util/cte_tiers_report.h)
 * against an in-process runtime: it must list a registered target with its
 * score and free space, and free space must fall as soon as data is stored
 * (the report refreshes the target stats instead of waiting for the sweep).
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>

#include "clio_cte/core/core_client.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"
#include "cte_tiers_report.h"
#include "simple_test.h"

using namespace std::chrono_literals;

namespace {

constexpr clio::run::u64 kRamTargetBytes = 64ULL * 1024 * 1024;
const char *kTargetName = "cte_tiers_test_target";
bool g_ready = false;

/** Register a 64 MiB RAM target into the pool CLIO_CTE_CLIENT talks to. */
bool RegisterRamTarget() {
  auto *cte = CLIO_CTE_CLIENT;
  clio::run::PoolId bdev_pool_id(931, 0);
  clio::run::bdev::Client bdev_client(bdev_pool_id);
  auto create = bdev_client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                        kTargetName, bdev_pool_id,
                                        clio::run::bdev::BdevType::kRam,
                                        kRamTargetBytes);
  create.Wait();
  auto reg = cte->AsyncRegisterTarget(kTargetName,
                                      clio::run::bdev::BdevType::kRam,
                                      kRamTargetBytes,
                                      clio::run::PoolQuery::Local(),
                                      bdev_pool_id);
  reg.Wait();
  return reg->GetReturnCode() == 0;
}

/** The report line naming `target`, or "" if there is none. */
std::string LineFor(const std::string &report, const std::string &target) {
  std::istringstream in(report);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find(target) != std::string::npos) return line;
  }
  return "";
}

/** The free_MB column (the last field) of a report line. */
double FreeMb(const std::string &line) {
  const size_t sp = line.find_last_of(' ');
  return std::stod(line.substr(sp + 1));
}

/** Free space summed over every target line (all lines after the header). */
double TotalFreeMb(const std::string &report) {
  std::istringstream in(report);
  std::string line;
  std::getline(in, line);  // header
  double total = 0;
  while (std::getline(in, line)) {
    if (!line.empty()) total += FreeMb(line);
  }
  return total;
}

}  // namespace

TEST_CASE("cte_tiers lists a registered target with score and free space",
          "[cte][cte_tiers]") {
  REQUIRE(g_ready);
  std::ostringstream out;
  REQUIRE(clio::cte::core::util::WriteTierReport(CLIO_CTE_CLIENT, out) == 0);
  const std::string report = out.str();
  REQUIRE(report.find("free_MB") != std::string::npos);  // header
  const std::string line = LineFor(report, kTargetName);
  INFO(report);
  REQUIRE(!line.empty());
  // An empty 64 MiB target: (almost) all of it free.
  const double free_mb = FreeMb(line);
  REQUIRE(free_mb > 60.0);
  REQUIRE(free_mb <= kRamTargetBytes / 1e6 + 0.1);
}

TEST_CASE("cte_tiers free space falls once data is stored",
          "[cte][cte_tiers]") {
  REQUIRE(g_ready);
  auto *cte = CLIO_CTE_CLIENT;
  std::ostringstream before;
  REQUIRE(clio::cte::core::util::WriteTierReport(cte, before) == 0);
  // Summed over all targets: an ambient config (~/.clio/clio.yaml) may add a
  // higher-scored tier that takes the put instead of this test's target.
  const double free_before = TotalFreeMb(before.str());

  constexpr size_t kBytes = 8 * 1024 * 1024;
  auto tag = cte->AsyncGetOrCreateTag("cte_tiers_test_tag");
  tag.Wait();
  REQUIRE(!tag->tag_id_.IsNull());
  auto buf = CLIO_IPC->AllocateBuffer(kBytes);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0x5A, kBytes);
  auto put = cte->AsyncPutBlob(tag->tag_id_, "blob", 0, kBytes,
                               buf.shm_.template Cast<void>(), 1.0f,
                               clio::cte::core::Context(), 0);
  put.Wait();
  REQUIRE(put->GetReturnCode() == 0);
  CLIO_IPC->FreeBuffer(buf);

  std::ostringstream after;
  REQUIRE(clio::cte::core::util::WriteTierReport(cte, after) == 0);
  const double free_after = TotalFreeMb(after.str());
  INFO(before.str() + after.str());
  REQUIRE(free_before - free_after >= kBytes / 1e6 - 0.5);
}

int main(int argc, char **argv) {
  if (clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::this_thread::sleep_for(300ms);
    g_ready = clio::cte::core::CLIO_CTE_CLIENT_INIT() && RegisterRamTarget();
  }
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return rc;
}
