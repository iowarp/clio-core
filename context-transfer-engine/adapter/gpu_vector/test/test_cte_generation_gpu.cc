/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Generational put/get: READINESS, not recency.
 *
 * The coherence question a multi-GPU run cannot otherwise answer is "has my
 * writer finished?". Nothing in the API answered it: a reader that ran ahead
 * got whatever existed, and with Context::create_on_get_ that is a
 * zero-filled blob returned as SUCCESS -- indistinguishable from real data,
 * so correctness silently became a question of which side ran first.
 *
 * A generational put stamps the blob; a generational get names the
 * generation it needs and IS NOT SERVED until the blob reaches it. The tests
 * below are written so that a get which ignored the generation would PASS
 * the data check and fail the ordering one -- the ordering is the claim.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace core = clio::cte::core;

#if !CTP_IS_DEVICE_PASS
namespace {
constexpr clio::run::u64 kBytes = 4096;

/** Wall-clock milliseconds since `t0`. */
double MsSince(const std::chrono::steady_clock::time_point &t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

TEST_CASE("cte: a generational get waits for its generation",
          "[cte][generation]") {
  {
    std::ofstream cfg("cte_generation.yaml");
    cfg << "networking:\n  port: 9431\n\n"
        << "runtime:\n  num_threads: 4\n  queue_depth: 4096\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n"
        << "    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"512MB\"\n\n"
        << "  - mod_name: clio_cte_core\n"
        << "    pool_name: cte_core\n    pool_query: local\n"
        << "    pool_id: \"512.0\"\n"
        << "    storage:\n"
        << "      - path: \"ram::gen_tier\"\n"
        << "        bdev_type: \"ram\"\n        capacity_limit: \"256MB\"\n"
        << "        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "cte_generation.yaml", 1);
  }
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
  SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE(core::CLIO_CTE_CLIENT_INIT());

  core::Client cte(core::kCtePoolId);
  auto tagf = cte.AsyncGetOrCreateTag(std::string("gen_test"));
  tagf.Wait();
  REQUIRE(tagf->GetReturnCode() == 0);
  const core::TagId tag = tagf->tag_id_;

  std::vector<char> wbuf(kBytes, 0), rbuf(kBytes, 0);

  auto put_gen = [&](const char *name, char fill, clio::run::u64 gen) {
    std::fill(wbuf.begin(), wbuf.end(), fill);
    core::Context ctx;
    ctx.generational_ = true;
    ctx.generation_ = gen;
    auto f = cte.AsyncPutBlob(tag, std::string(name), 0, kBytes, wbuf.data(),
                              0.5f, ctx, 0u,
                              clio::run::PoolQuery::Dynamic());
    f.Wait();
    return f.get()->GetReturnCode();
  };
  auto get_gen = [&](const char *name, clio::run::u64 gen,
                     bool create_on_get = false) {
    std::fill(rbuf.begin(), rbuf.end(), 0);
    core::Context ctx;
    ctx.create_on_get_ = create_on_get;
    if (gen != 0) {
      ctx.generational_ = true;
      ctx.generation_ = gen;
    }
    auto f = cte.AsyncGetBlob(tag, std::string(name), 0, kBytes, 0,
                              rbuf.data(),
                              clio::run::PoolQuery::Dynamic(), ctx);
    f.Wait();
    return f.get()->GetReturnCode();
  };

  int rc = 0;

  // 1. A get for a generation already published is served immediately.
  REQUIRE(put_gen("a", 'A', 5) == 0);
  {
    const auto t0 = std::chrono::steady_clock::now();
    const int g = get_gen("a", 5);
    const double ms = MsSince(t0);
    const bool ok = (g == 0) && (rbuf[0] == 'A') && (rbuf[kBytes - 1] == 'A');
    std::printf("  published gen 5, get gen 5     rc=%d %.1f ms -> %s\n", g,
                ms, ok ? "PASS" : "FAIL");
    if (!ok) rc = 1;
  }

  // 2. THE CLAIM: a get for a generation that has NOT been published must
  //    not return until a writer publishes it. The writer is deliberately
  //    late, so a get that ignored the generation would come back early --
  //    and with the right bytes, which is why the timing is the assertion.
  {
    std::atomic<bool> put_done{false};
    const auto t0 = std::chrono::steady_clock::now();
    std::thread writer([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(400));
      std::vector<char> w(kBytes, 'B');
      core::Context ctx;
      ctx.generational_ = true;
      ctx.generation_ = 9;
      auto f = cte.AsyncPutBlob(tag, std::string("b"), 0, kBytes, w.data(),
                                0.5f, ctx, 0u,
                                clio::run::PoolQuery::Dynamic());
      f.Wait();
      put_done.store(true);
    });
    const int g = get_gen("b", 9);
    const double ms = MsSince(t0);
    writer.join();
    // Served only after the writer, and with the writer's bytes.
    const bool ok = (g == 0) && put_done.load() && (ms >= 350.0) &&
                    (rbuf[0] == 'B') && (rbuf[kBytes - 1] == 'B');
    std::printf("  unpublished gen 9, late writer rc=%d %.1f ms -> %s\n", g,
                ms, ok ? "PASS" : "FAIL");
    if (!ok) {
      std::printf("    waited=%.1fms (want >=350) put_done=%d first=%c\n", ms,
                  (int)put_done.load(), rbuf[0]);
      rc = 1;
    }
  }

  // 3. An ORDINARY get of the same never-written blob returns success with
  //    an untouched buffer -- the create_on_get_ behaviour the generational
  //    flag exists to opt out of. This is what a reader used to get.
  {
    const int g = get_gen("never", 0, /*create_on_get=*/true);
    const bool ok = (g == 0) && (rbuf[0] == 0);
    std::printf("  ordinary get, never written    rc=%d -> %s%s\n", g,
                ok ? "PASS" : "FAIL",
                ok ? "  (zeros, reported as success)" : "");
    if (!ok) rc = 1;
  }

  // 4. A generation that never arrives must ERROR, not hang the worker.
  {
    const auto t0 = std::chrono::steady_clock::now();
    const int g = get_gen("never", 12345, /*create_on_get=*/true);
    const double ms = MsSince(t0);
    const bool ok = (g != 0);
    std::printf("  gen 12345 never published      rc=%d %.1f ms -> %s\n", g,
                ms, ok ? "PASS" : "FAIL");
    if (!ok) rc = 1;
  }

  REQUIRE(rc == 0);
}
#endif  // !CTP_IS_DEVICE_PASS

SIMPLE_TEST_MAIN()
