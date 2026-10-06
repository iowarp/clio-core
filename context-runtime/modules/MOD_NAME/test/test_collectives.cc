/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Single-node unit tests for the MOD_NAME collective methods (AllReduce and
 * Barrier) and the BatchManager machinery underneath them.
 *
 * The 4-node behaviour lives in the collective_bench docker test, which needs a
 * cluster and cannot run here. These tests cover everything that does NOT need
 * one, which is most of the code: the task types and their serialization, the
 * generated Run/NewTask/NewCopyTask/AggregateIn/AggregateOut switch arms, the
 * runtime handlers, and the BatchManager's group/flush/broadcast cycle.
 *
 * Two routing modes are used deliberately:
 *
 *   AllToOne  releases once a task from every container has arrived. On one
 *             node the pool has ONE container, so a single submission satisfies
 *             the barrier -- which exercises the count-based release path
 *             end to end without a cluster.
 *
 *   ManyToOne releases on a time window instead of a count, so several
 *             submissions from this one client land in the SAME batch. That is
 *             what makes real multi-member aggregation testable here:
 *             AggregateIn folds them together and the single result is
 *             broadcast back to every submitter. The sum is checked, so a
 *             batch that silently dropped or double-counted a member fails.
 */

#include "simple_test.h"

#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

namespace {

bool g_initialized = false;

/** Distinct pool id per test so one test's pool cannot serve another. */
clio::run::PoolId MakePoolId() {
  static clio::run::u32 counter = 0;
  auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
  return clio::run::PoolId(
      static_cast<clio::run::u32>((us & 0xFFFFF) + 4000 + (++counter) * 97), 0);
}

/** Bring up the runtime once and create a MOD_NAME pool to submit against. */
class CollectiveFixture {
 public:
  CollectiveFixture() : pool_id_(MakePoolId()) {
    if (!g_initialized) {
      bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      REQUIRE(ok);
      g_initialized = true;
      SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
      std::this_thread::sleep_for(100ms);
      REQUIRE(CLIO_RUNTIME_MANAGER != nullptr);
      REQUIRE(CLIO_IPC != nullptr);
      REQUIRE(CLIO_POOL_MANAGER != nullptr);
    }
    client_ = clio::run::MOD_NAME::Client(pool_id_);
    auto create = client_.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "test_collectives_pool", pool_id_);
    create.Wait();
    REQUIRE(create->return_code_ == 0);
    client_.pool_id_ = create->new_pool_id_;
    std::this_thread::sleep_for(100ms);
  }

  clio::run::MOD_NAME::Client &client() { return client_; }

 private:
  clio::run::PoolId pool_id_;
  clio::run::MOD_NAME::Client client_;
};

constexpr clio::run::u32 kHash = 0;
/**
 * ManyToOne batch window. The window opens on the FIRST arrival, so every
 * submission in a test's loop has to land inside it or the batch splits and the
 * expected total is wrong -- a flaky failure that would look like broken
 * aggregation. The submissions are non-blocking and microseconds apart, so this
 * is generous by three orders of magnitude on purpose: the assertions here are
 * about whether aggregation happened, and they should never be decided by how
 * loaded the runner is.
 */
constexpr clio::run::u64 kWindowNs = 500ull * 1000ull * 1000ull;  // 500ms

}  // namespace

// A single-node pool has one container, so one arrival satisfies the AllToOne
// count and the barrier releases. This is the count-based path -- if the
// release condition were wrong in the "everybody is here" direction, this hangs
// rather than fails, so the test timing out IS the failure signal.
TEST_CASE("collective_barrier_alltoone_single_node", "[MOD_NAME][collective]") {
  CollectiveFixture fixture;
  auto q = clio::run::PoolQuery::AllToOne(kHash, /*batch_key=*/1);
  auto f = fixture.client().AsyncBarrier(q);
  f.Wait();
  REQUIRE(f->return_code_ == 0);
}

// Same path with a payload: the aggregate echoes value_ into sum_ and the
// result is broadcast back. One member, so the total is its own contribution.
TEST_CASE("collective_allreduce_alltoone_single_node",
          "[MOD_NAME][collective]") {
  CollectiveFixture fixture;
  auto q = clio::run::PoolQuery::AllToOne(kHash, /*batch_key=*/2);
  auto f = fixture.client().AsyncAllReduce(q, 42);
  f.Wait();
  REQUIRE(f->return_code_ == 0);
  REQUIRE(f->sum_ == 42);
}

// THE aggregation test. Several submissions share one time-windowed batch, so
// AggregateIn actually folds multiple members together and the single result is
// broadcast 1->N. Every submitter must observe the SAME total -- the sum of all
// contributions, not its own -- which is what distinguishes a real collective
// from N independent tasks that each happened to succeed.
TEST_CASE("collective_allreduce_manytoone_aggregates",
          "[MOD_NAME][collective]") {
  CollectiveFixture fixture;
  constexpr clio::run::u32 kMembers = 4;
  // 1 + 2 + 3 + 4
  constexpr clio::run::u64 kExpected = kMembers * (kMembers + 1) / 2;

  std::vector<clio::run::Future<clio::run::MOD_NAME::AllReduceTask>> futures;
  futures.reserve(kMembers);
  for (clio::run::u32 i = 0; i < kMembers; ++i) {
    auto q = clio::run::PoolQuery::ManyToOne(kHash, /*batch_key=*/3, kWindowNs);
    futures.push_back(fixture.client().AsyncAllReduce(q, i + 1));
  }
  for (clio::run::u32 i = 0; i < kMembers; ++i) {
    futures[i].Wait();
    REQUIRE(futures[i]->return_code_ == 0);
    REQUIRE(futures[i]->sum_ == kExpected);
  }
}

// The barrier carries no payload, so batching it only has to not lose anyone:
// every member of the batch must be completed by the single aggregate's
// broadcast. A member dropped from the broadcast never completes and hangs.
TEST_CASE("collective_barrier_manytoone_batch", "[MOD_NAME][collective]") {
  CollectiveFixture fixture;
  constexpr clio::run::u32 kMembers = 4;
  std::vector<clio::run::Future<clio::run::MOD_NAME::BarrierTask>> futures;
  futures.reserve(kMembers);
  for (clio::run::u32 i = 0; i < kMembers; ++i) {
    auto q = clio::run::PoolQuery::ManyToOne(kHash, /*batch_key=*/4, kWindowNs);
    futures.push_back(fixture.client().AsyncBarrier(q));
  }
  for (clio::run::u32 i = 0; i < kMembers; ++i) {
    futures[i].Wait();
    REQUIRE(futures[i]->return_code_ == 0);
  }
}

// Successive collectives reusing one (hash, batch_key) must each run against a
// settled predecessor: the BatchManager allows one aggregate per key in flight
// and holds later arrivals until it completes. Reusing the key is what a real
// caller does in a loop, and a group that failed to reset would give the wrong
// total on the second round rather than fail outright.
TEST_CASE("collective_allreduce_repeated_same_key", "[MOD_NAME][collective]") {
  CollectiveFixture fixture;
  for (clio::run::u64 round = 1; round <= 5; ++round) {
    auto q = clio::run::PoolQuery::AllToOne(kHash, /*batch_key=*/5);
    auto f = fixture.client().AsyncAllReduce(q, round * 100);
    f.Wait();
    REQUIRE(f->return_code_ == 0);
    // Keyed to the round, so a stale group carried over from the previous one
    // shows up as a wrong total instead of passing by coincidence.
    REQUIRE(f->sum_ == round * 100);
  }
}

// Concurrent collectives on DIFFERENT batch keys must not merge: the key is
// part of the group identity, so two in-flight collectives stay separate and
// each returns its own total rather than their combined one.
TEST_CASE("collective_allreduce_distinct_keys_do_not_merge",
          "[MOD_NAME][collective]") {
  CollectiveFixture fixture;
  auto q6 = clio::run::PoolQuery::AllToOne(kHash, /*batch_key=*/6);
  auto q7 = clio::run::PoolQuery::AllToOne(kHash, /*batch_key=*/7);
  auto f6 = fixture.client().AsyncAllReduce(q6, 11);
  auto f7 = fixture.client().AsyncAllReduce(q7, 22);
  f6.Wait();
  f7.Wait();
  REQUIRE(f6->return_code_ == 0);
  REQUIRE(f7->return_code_ == 0);
  REQUIRE(f6->sum_ == 11);
  REQUIRE(f7->sum_ == 22);
}

SIMPLE_TEST_MAIN()
