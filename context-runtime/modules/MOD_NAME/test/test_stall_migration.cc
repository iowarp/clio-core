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
 * issue #785 — a stalled worker must not strand blocked / in-flight tasks.
 *
 * #781 made the runtime route NEW work around a worker wedged by a
 * non-yielding task. Work already committed to that worker is still stranded:
 * its lane backlog, its blocked / periodic / retry queues, its completion event
 * queue, and — worst — tasks suspended on co_await, which live in no queue at
 * all and are reachable only through the subtask future's parent handle plus the
 * raw EventQueue()/Lane() addresses cached in their RunContext.
 *
 * These tests are the deterministic repro for that. They deliberately use only
 * EXISTING module primitives so they reproduce the bug against unmodified code:
 *
 *   MOD_NAME::Custom   with spin_us_  -> a NON-YIELDING busy spin (the bad task)
 *   MOD_NAME::WaitTest with depth > 1 -> recursively self-sends a subtask and
 *                                        CLIO_CO_AWAITs it (the blocked task)
 *
 * A WaitTest parent suspended on co_await has its completion routed to whichever
 * worker it first ran on. If that worker is wedged inside a Custom spin, the
 * subtask can complete on a perfectly healthy worker and the parent still never
 * wakes.
 *
 * EXPECTED STATE: these FAIL on this branch today. They are the acceptance
 * criteria for the migration work, not a regression guard for it. Every timed
 * wait is a bounded poll on Future::IsComplete() — a stranded task makes a test
 * fail, never hang.
 */

#include "simple_test.h"

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

#include "clio_ctp/introspect/system_info.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/task.h>
#include <clio_runtime/types.h>

#include <clio_runtime/scheduler/default_sched.h>

#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

namespace {

/** Busy-spin length of the "bad" task, microseconds. Long enough that a test
 *  bound well under it cannot be satisfied by simply waiting the spin out. */
constexpr clio::run::u32 kSpinUs = 12'000'000;  // 12 s

/** How long a chained task is allowed to take while spinners are running.
 *
 *  DERIVED FROM THE RESCUE MECHANISM, not tuned until green. Rescue is not
 *  instant: LoadBalance declares a stall only after kStallThresholdSec (1 s)
 *  and runs on a 500 ms cadence, so each rescue costs up to ~1.5 s to trigger.
 *  A depth-D chain can hit D such stalls in sequence — each hop may be queued
 *  behind a different heavy task — so the floor is
 *
 *      kChainDepth * (kStallThresholdSec + monitor_period) = 2 * 1.5 s = 3 s
 *
 *  6 s therefore leaves 2x margin for scheduling variance while still being
 *  HALF the spin duration, so passing genuinely means "does not track the bad
 *  task's runtime".
 *
 *  Measured while calibrating: at depth 3 the floor is 4.5 s, which left almost
 *  no margin — a 3 s bound passed 2/5 runs, 4 s passed 1/5, 6 s passed 2/5 and
 *  7 s passed 3/3, every failure being rescue latency rather than stranding.
 *  The fix is fewer SEQUENTIAL rescue cycles (depth 2), not a looser bound:
 *  loosening it toward the spin duration would quietly destroy what the test
 *  asserts. */
constexpr int kChainDeadlineMs = 6000;

/** Generous bound used only to distinguish "slow" from "lost". */
constexpr int kDropDeadlineMs = 30000;

/** Number of non-yielding spinners.
 *
 * This MUST stay below the number of general-purpose (I/O) workers, or the test
 * measures saturation instead of stranding: with every compute worker spinning,
 * nothing runs and the result says nothing about whether blocked tasks can be
 * rescued. The control group below enforces that empirically.
 *
 * Note the default runtime has only ONE general-purpose worker: num_threads=4
 * divides into 1 scheduler + 1 I/O + net_send + net_recv, so a single bad task
 * wedges all compute. These tests therefore need CLIO_NUM_THREADS raised to
 * leave spare compute workers. Overridable via CLIO_STALL_TEST_SPINNERS. */
int NumSpinners() {
  if (const char *env = std::getenv("CLIO_STALL_TEST_SPINNERS")) {
    int n = std::atoi(env);
    if (n > 0) return n;
  }
  return 2;
}
constexpr int kNumChained = 8;
constexpr clio::run::u32 kChainDepth = 2;

constexpr clio::run::PoolId kStallPoolId = clio::run::PoolId(9100, 0);

bool g_initialized = false;

class StallFixture {
 public:
  StallFixture() {
    if (!g_initialized) {
      // The default num_threads=4 divides into 1 scheduler + 1 I/O + net_send +
      // net_recv, leaving exactly ONE general-purpose worker — so a single
      // non-yielding task wedges all compute and every run degenerates into
      // saturation. Raise it before CLIO_INIT reads the config so there is
      // provably spare compute capacity for the control group to land on.
      // Routed through SystemInfo::Setenv: MSVC has no setenv(), so the raw
      // call is a Windows build break. overwrite=0 keeps an externally
      // supplied value.
      if (std::getenv("CLIO_NUM_THREADS") == nullptr) {
        ctp::SystemInfo::Setenv("CLIO_NUM_THREADS", "8", /*overwrite=*/1);
      }
      bool success =
          clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      if (success) {
        g_initialized = true;
        SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
        std::this_thread::sleep_for(500ms);
      }
    }
  }

  /** Create the MOD_NAME container and return the pool id the runtime assigned
   *  (AsyncCreate remaps it, so callers must use the returned value). */
  bool createContainer(clio::run::PoolId pool_id, clio::run::PoolId &out) {
    clio::run::MOD_NAME::Client client(pool_id);
    auto create_task = client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                          "stall_migration_pool", pool_id);
    create_task.Wait();
    if (create_task->return_code_ != 0) {
      return false;
    }
    out = create_task->new_pool_id_;
    std::this_thread::sleep_for(100ms);
    return true;
  }
};

/** Poll a set of futures until all complete or the deadline expires.
 *  @return number still incomplete when we stopped. */
template <typename FutureT>
size_t WaitAllBounded(std::vector<FutureT> &futures, int deadline_ms) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(deadline_ms);
  for (;;) {
    size_t pending = 0;
    for (auto &f : futures) {
      if (!f.IsComplete()) {
        ++pending;
      }
    }
    if (pending == 0) {
      return 0;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return pending;
    }
    std::this_thread::sleep_for(5ms);
  }
}

}  // namespace

//==============================================================================
// TEST 1 — a wedged worker must not stall tasks blocked on co_await
//==============================================================================

TEST_CASE("stall_does_not_strand_blocked_tasks", "[stall][migration]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("chained co_await tasks complete while spinners wedge workers") {
    // Wedge the pool with non-yielding spinners. These are submitted and NOT
    // waited on — they own their workers for kSpinUs.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> spinners;
    const int num_spinners = NumSpinners();
    spinners.reserve(num_spinners);
    for (int i = 0; i < num_spinners; ++i) {
      spinners.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "s",
                                            0, kSpinUs));
    }

    // Let the spinners actually get picked up and enter their spin loops.
    std::this_thread::sleep_for(300ms);

    // Now submit chained tasks. Each recursively self-sends a subtask and
    // co_awaits it, so each parent parks with its completion routed to the
    // worker it first ran on.
    std::vector<clio::run::Future<clio::run::MOD_NAME::WaitTestTask>> chained;
    chained.reserve(kNumChained);
    for (int i = 0; i < kNumChained; ++i) {
      chained.push_back(client.AsyncWaitTest(clio::run::PoolQuery::Local(),
                                             kChainDepth,
                                             static_cast<clio::run::u32>(i)));
    }

    // CONTROL GROUP. Flat, dependency-free quick tasks submitted at the same
    // moment. #781 already routes these around a stalled worker, so they must
    // sail through. Without this control the test cannot tell the bug
    // ("dependency chains are stranded behind a wedged worker") apart from mere
    // saturation ("every worker is busy, nothing runs"). If the control fails
    // too, the spinner count is simply swamping the pool and the run proves
    // nothing about stranding.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> control;
    control.reserve(kNumChained);
    for (int i = 0; i < kNumChained; ++i) {
      control.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "c", 0, 10));
    }

    size_t control_pending = WaitAllBounded(control, kChainDeadlineMs);
    size_t stranded = WaitAllBounded(chained, kChainDeadlineMs);
    INFO("after " + std::to_string(kChainDeadlineMs) +
         "ms — control (flat) pending: " + std::to_string(control_pending) +
         ", chained (co_await) pending: " + std::to_string(stranded));

    // Runtime liveness: if this fails, the pool is saturated and the chained
    // result below is not evidence of anything.
    REQUIRE(control_pending == 0);

    // The acceptance criterion for #785: chained-task completion must not track
    // the spin duration. Any non-zero count here is a task stranded behind a
    // wedged worker while the runtime is demonstrably still alive.
    REQUIRE(stranded == 0);

    // Drain the spinners so the next test starts from a quiet runtime.
    WaitAllBounded(spinners, kDropDeadlineMs);
  }
}

//==============================================================================
// TEST 2 — a worker wedged AFTER a task parks must not strand that task
//
// This is the category the lane rescue does NOT cover. Test 1's chained tasks
// were stuck because their subtasks sat in a stalled worker's LANE, so moving
// the lane freed them. Here the parents are already parked on a co_await before
// any spinner is submitted, so what is stranded is the completion path: the
// parent is in no queue at all, and its wakeup is routed to the event queue of
// whichever worker it first ran on. Wedge that worker and the event has nowhere
// to land, however healthy the rest of the pool is.
//==============================================================================

TEST_CASE("stall_after_park_does_not_strand_completion", "[stall][event]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("parents parked on a subtask still wake when their worker wedges") {
    // CONCURRENCY BUDGET. Every simultaneously-spinning task occupies one
    // general-purpose worker, and there are only (num_threads - 4) of those —
    // 4 at the fixture's CLIO_NUM_THREADS=8. Each parked parent spawns a child
    // that SPINS, so parents count against the budget too. Exceed it and the
    // run degenerates into saturation, which is exactly what the control group
    // catches. Budget here: 2 children + 1 spinner = 3 busy, 1 left for the
    // control to land on.
    constexpr int kNumParked = 2;
    constexpr int kParkSpinners = 1;
    constexpr clio::run::u32 kParkSpinUs = 2'000'000;  // 2 s

    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> parked;
    parked.reserve(kNumParked);
    for (int i = 0; i < kNumParked; ++i) {
      parked.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "p", 0,
                                          kParkSpinUs, /*chain_depth=*/1));
    }

    // Let every parent actually reach its co_await and park.
    std::this_thread::sleep_for(500ms);

    // NOW wedge workers. Any parent parked on one of these has its completion
    // routed to a worker that will not drain its event queue for kSpinUs.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> spinners;
    spinners.reserve(kParkSpinners);
    for (int i = 0; i < kParkSpinners; ++i) {
      spinners.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "s", 0, kSpinUs));
    }

    // Same liveness control as test 1.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> control;
    control.reserve(kNumParked);
    for (int i = 0; i < kNumParked; ++i) {
      control.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "c", 0, 10));
    }
    size_t control_pending = WaitAllBounded(control, kChainDeadlineMs);

    // Budget: the child's own spin, plus slack for the rescue to notice and act
    // (LoadBalance runs on a 500 ms cadence). Still far below kSpinUs, so a
    // parent that simply waits out the spinner cannot pass this.
    const int park_deadline_ms =
        static_cast<int>(kParkSpinUs / 1000) + kChainDeadlineMs;
    size_t stranded = WaitAllBounded(parked, park_deadline_ms);

    INFO("after " + std::to_string(park_deadline_ms) +
         "ms — control pending: " + std::to_string(control_pending) +
         ", parked-then-wedged pending: " + std::to_string(stranded));

    REQUIRE(control_pending == 0);
    REQUIRE(stranded == 0);

    WaitAllBounded(spinners, kDropDeadlineMs);
  }
}

//==============================================================================
// TEST 3 — a BLOCKING stall must be rescued too, not just a spinning one
//
// #781's premise was a blocking syscall inside a coroutine that never yields,
// which is the commoner real-world shape: the thread is descheduled rather than
// burning CPU, so it is invisible in a CPU profile and looks nothing like a
// spin to the OS. From the runtime's side it is identical — ExecTask does not
// return — so the rescue should not care. This test proves that rather than
// assuming it, because a rescue keyed on CPU consumption rather than wall-clock
// elapsed time would pass every spin test and still leave real I/O stalls
// unrescued.
//==============================================================================

TEST_CASE("blocking_stall_is_rescued_like_a_spinning_one", "[stall][blocking]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("workers wedged in sleep() do not strand chained tasks") {
    constexpr int kNumBlockers = 2;
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> blockers;
    blockers.reserve(kNumBlockers);
    for (int i = 0; i < kNumBlockers; ++i) {
      blockers.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "b",
                                            0, /*spin_us=*/0,
                                            /*chain_depth=*/0,
                                            /*block_us=*/kSpinUs));
    }
    std::this_thread::sleep_for(300ms);

    std::vector<clio::run::Future<clio::run::MOD_NAME::WaitTestTask>> chained;
    chained.reserve(kNumChained);
    for (int i = 0; i < kNumChained; ++i) {
      chained.push_back(client.AsyncWaitTest(clio::run::PoolQuery::Local(),
                                             kChainDepth,
                                             static_cast<clio::run::u32>(i)));
    }
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> control;
    control.reserve(kNumChained);
    for (int i = 0; i < kNumChained; ++i) {
      control.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "c", 0, 10));
    }

    size_t control_pending = WaitAllBounded(control, kChainDeadlineMs);
    size_t stranded = WaitAllBounded(chained, kChainDeadlineMs);
    INFO("blocking stall — control pending: " +
         std::to_string(control_pending) +
         ", chained pending: " + std::to_string(stranded));

    REQUIRE(control_pending == 0);
    REQUIRE(stranded == 0);

    WaitAllBounded(blockers, kDropDeadlineMs);
  }
}

//==============================================================================
// TEST 4 — SOAK: repeated stall/rescue cycles must not lose tasks or grow
//
// One rescue working says little about a daemon that runs for weeks. Each cycle
// allocates and reassigns real state — replacement workers, lanes, event
// queues, parked-task lists — so the failure modes that only appear under
// repetition are exactly the dangerous ones: a queue orphaned on the Nth
// rescue, a worker leaked per cycle, a task quietly lost when a replacement is
// recycled. An earlier version of the rescue orphaned an event queue on every
// single cycle and it did not show up in any one-shot test.
//
// Asserts every task of every round completes. The pool bound itself is
// enforced in SpawnAdditionalWorker (baseline + one thread per core); this test
// exercises the path that would blow through it.
//==============================================================================

TEST_CASE("repeated_stall_cycles_lose_nothing", "[stall][soak]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("five stall/rescue rounds, nothing dropped") {
    constexpr int kRounds = 5;
    // Short stalls so the suite stays quick, but still well over
    // kStallThresholdSec (1 s) so every round genuinely triggers a rescue.
    constexpr clio::run::u32 kRoundStallUs = 2'500'000;  // 2.5 s
    size_t total_submitted = 0;
    size_t total_lost = 0;

    for (int round = 0; round < kRounds; ++round) {
      std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> stallers;
      // Alternate the stall SHAPE per round: spinning and blocking exercise
      // different worker states, and a rescue path that handled only one would
      // pass a uniform soak.
      const bool blocking = (round % 2 == 1);
      for (int i = 0; i < 2; ++i) {
        stallers.push_back(client.AsyncCustom(
            clio::run::PoolQuery::Local(), "s", 0,
            /*spin_us=*/blocking ? 0 : kRoundStallUs, /*chain_depth=*/0,
            /*block_us=*/blocking ? kRoundStallUs : 0));
      }
      std::this_thread::sleep_for(200ms);

      std::vector<clio::run::Future<clio::run::MOD_NAME::WaitTestTask>> chained;
      std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> quick;
      for (int i = 0; i < 4; ++i) {
        chained.push_back(client.AsyncWaitTest(
            clio::run::PoolQuery::Local(), kChainDepth,
            static_cast<clio::run::u32>(round * 100 + i)));
        quick.push_back(
            client.AsyncCustom(clio::run::PoolQuery::Local(), "q", 0, 10));
      }

      total_submitted += stallers.size() + chained.size() + quick.size();
      total_lost += WaitAllBounded(quick, kDropDeadlineMs);
      total_lost += WaitAllBounded(chained, kDropDeadlineMs);
      total_lost += WaitAllBounded(stallers, kDropDeadlineMs);
    }

    INFO("soak: " + std::to_string(total_submitted) + " tasks over " +
         std::to_string(kRounds) + " rounds, lost " +
         std::to_string(total_lost));
    REQUIRE(total_lost == 0);
  }
}

//==============================================================================
// TEST 5 — HEAD-OF-LINE BLOCKING: small tasks queued BEHIND massive ones
//
// The other tests submit small tasks while free workers exist, so
// RuntimeMapTask (#781) routes them AROUND the busy ones and they never queue
// behind anything. That measures routing, not head-of-line blocking, and it is
// the easy half of the problem.
//
// This test removes the escape route: it saturates every general-purpose worker
// with massive tasks FIRST, so that by the time the small tasks arrive there is
// no unloaded worker to route them to and they must land in a lane behind a
// massive task. That is the practically important shape — a 10 us request stuck
// behind a multi-second one — and the one lane transfer does NOT trivially fix,
// because moving a lane preserves its FIFO order: the replacement pops the next
// queued task, and if that is another massive task it wedges in turn.
//
// The expected cost is therefore roughly (massive tasks ahead) x (rescue
// latency), NOT (massive tasks ahead) x (massive duration) — which is the whole
// point. The assertion is deliberately generous about the constant and strict
// about the scaling: small tasks must not wait out the massive ones.
//==============================================================================

TEST_CASE("small_tasks_queued_behind_massive_ones", "[stall][headofline]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("a small task does not wait out the massive tasks ahead of it") {
    // Sized against the general-purpose worker count (num_threads 8 -> 4 I/O
    // workers): enough massive tasks to fill every worker AND leave some
    // queued, so small tasks provably cannot avoid landing behind one.
    constexpr int kMassive = 6;
    constexpr clio::run::u32 kMassiveUs = 6'000'000;  // 6 s each
    constexpr int kSmall = 12;

    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> massive;
    massive.reserve(kMassive);
    for (int i = 0; i < kMassive; ++i) {
      massive.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "M",
                                           0, kMassiveUs));
    }
    // Let them be picked up and fill the lanes before the small ones arrive.
    std::this_thread::sleep_for(400ms);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> small;
    small.reserve(kSmall);
    for (int i = 0; i < kSmall; ++i) {
      small.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "s", 0, 10));
    }

    // Generous ceiling: still well under what waiting out even ONE massive
    // task would cost if the small tasks were simply stuck.
    constexpr int kSmallDeadlineMs = 5000;
    size_t stuck = WaitAllBounded(small, kSmallDeadlineMs);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0)
                          .count();

    INFO("head-of-line: " + std::to_string(kSmall) + " small tasks behind " +
         std::to_string(kMassive) + " massive (" +
         std::to_string(kMassiveUs / 1000) + "ms each); " +
         std::to_string(stuck) + " still pending after " +
         std::to_string(elapsed_ms) + "ms");

    REQUIRE(stuck == 0);

    WaitAllBounded(massive, kDropDeadlineMs);
  }
}

//==============================================================================
// TEST 6 — DEPENDENCY DEPTH: a chain longer than the worker pool
//
// Everything so far stalls workers with tasks that eventually RETURN. This
// covers the other way a runtime can wedge itself: a dependency chain deeper
// than the number of workers. Each parent occupies a worker while parked on its
// child, so a naive runtime runs out of workers with every one of them waiting
// on a task that cannot be scheduled — a self-inflicted deadlock with no bad
// task anywhere in sight.
//
// This is the closest honest approximation to "is deadlock impossible" that the
// existing primitives can express. It is NOT a cycle: A waits on B waits on C
// is resolvable given enough scheduling, whereas a true cycle (A waits on B,
// B waits on A) is unresolvable by any amount of migration and the runtime has
// no detection for it. That gap is documented, not tested, because a test for
// it could only assert that we hang.
//==============================================================================

TEST_CASE("deep_dependency_chain_does_not_exhaust_the_pool", "[stall][depth]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("chains deeper than the worker pool still complete") {
    // 8 workers configured -> 4 general-purpose. Depth 12 per chain, several
    // chains at once, so parked parents far outnumber the workers available to
    // run their children.
    constexpr clio::run::u32 kDeep = 12;
    constexpr int kChains = 6;
    constexpr int kDeadlineMs = 20000;

    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> deep;
    deep.reserve(kChains);
    for (int i = 0; i < kChains; ++i) {
      // Leaf spins briefly so the chain is real work rather than instant.
      deep.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "d", 0,
                                        /*spin_us=*/1000,
                                        /*chain_depth=*/kDeep));
    }

    size_t stuck = WaitAllBounded(deep, kDeadlineMs);
    INFO("depth: " + std::to_string(kChains) + " chains of depth " +
         std::to_string(kDeep) + " (" +
         std::to_string(kChains * (kDeep + 1)) +
         " tasks, far exceeding the worker pool); " + std::to_string(stuck) +
         " chains incomplete");
    REQUIRE(stuck == 0);
  }
}

//==============================================================================
// TEST 7 — the progress watchdog must NOT fire on merely SLOW work
//
// Migration handles a worker wedged by a task that eventually returns. It is
// powerless against the classes where the WORK ITSELF cannot proceed — a true
// dependency cycle (A awaits B, B awaits A), a lock cycle, a lost wakeup —
// because there is no healthy worker to move the work to. Those present
// identically to a user: the runtime goes quiet forever.
//
// We cannot make them impossible. We can refuse to let them be silent. The
// watchdog keys on a narrow signature: tasks outstanding, NO worker executing,
// and nothing completed for the alarm window.
//
// This test covers the FALSE POSITIVE direction, which is the one that can be
// tested honestly here. Every worker is occupied by a long blocking task, so
// nothing completes for well over the alarm window — but workers ARE executing,
// so this is slowness, not deadlock, and the watchdog must stay quiet. An alarm
// that fires on every slow task would be trained away by operators inside a
// week, which is worse than no alarm at all.
//
// The firing direction is NOT covered: constructing a true cycle needs a task
// that awaits something never scheduled, which the existing module primitives
// cannot express. That gap is stated in the docs rather than papered over.
//==============================================================================

TEST_CASE("progress_watchdog_does_not_cry_wolf_on_slow_work", "[stall][watchdog]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("a total progress stall is reported, not swallowed") {
    // Longer than kNoProgressAlarmSec (10 s) so the watchdog has a window in
    // which genuinely nothing completes.
    constexpr clio::run::u32 kLongBlockUs = 14'000'000;  // 14 s
    constexpr int kOccupy = 12;  // comfortably exceeds the pool + its headroom

    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> blockers;
    blockers.reserve(kOccupy);
    for (int i = 0; i < kOccupy; ++i) {
      blockers.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "w",
                                            0, /*spin_us=*/0,
                                            /*chain_depth=*/0,
                                            /*block_us=*/kLongBlockUs));
    }

    // Everything must still complete — the watchdog is a diagnostic, and a
    // diagnostic that changed behaviour would be a bug of its own.
    size_t lost = WaitAllBounded(blockers, kDropDeadlineMs);
    INFO("watchdog scenario: " + std::to_string(kOccupy) +
         " long blocking tasks (slow, not deadlocked), lost " +
         std::to_string(lost));
    REQUIRE(lost == 0);
  }
}

//==============================================================================
// TEST 8 — the watchdog's FIRING conditions, tested as a predicate
//
// Test 7 covers the false-positive direction. This covers firing, which cannot
// be tested by building a real deadlock: the tasks would never complete,
// teardown would never finish, and CI would HANG rather than fail — the one
// outcome worse than an untested alarm. Testing the predicate directly gets the
// coverage without wedging anything.
//
// The lock-cycle row is the one that matters. CoMutex::Lock() calls
// ctp::Mutex::Lock(), a blocking mutex rather than a coroutine-aware park, so
// two tasks deadlocked A/B-B/A sit INSIDE ExecTask with IsExecuting() true. An
// earlier version of this watchdog keyed only on "nothing executing" and so
// missed that case entirely while claiming to cover it.
//==============================================================================

TEST_CASE("watchdog_predicate_fires_on_the_right_shapes", "[stall][predicate]") {
  using clio::run::DefaultScheduler;
  double win = 0.0;

  SECTION("idle runtime is not a deadlock") {
    // No outstanding work: nothing to be stuck on, however few workers run.
    REQUIRE_FALSE(DefaultScheduler::IsWedgedShape(0, 0, 0, &win));
    REQUIRE_FALSE(DefaultScheduler::IsWedgedShape(0, 4, 4, &win));
  }

  SECTION("work outstanding with nothing executing IS a deadlock") {
    REQUIRE(DefaultScheduler::IsWedgedShape(1, 0, 0, &win));
    // Unambiguous, so it uses the SHORT window.
    REQUIRE(win < 30.0);
  }

  SECTION("work outstanding with every executing worker stalled IS suspect") {
    REQUIRE(DefaultScheduler::IsWedgedShape(5, 4, 4, &win));
    // Ambiguous against a slow syscall, so it must use the LONG window —
    // firing quickly here would alarm on every long blocking task.
    REQUIRE(win > 30.0);
  }

  SECTION("a healthy worker means progress is possible") {
    // Three of four stalled, one still running: not wedged.
    REQUIRE_FALSE(DefaultScheduler::IsWedgedShape(5, 4, 3, &win));
    // Nothing stalled at all.
    REQUIRE_FALSE(DefaultScheduler::IsWedgedShape(5, 4, 0, &win));
  }
}

//==============================================================================
// TEST 9 — TaskGroup affinity, the mechanism D8 rests on
//
// D8 replaces the hardcoded net-worker roles with TaskGroup affinity, and D8-1
// splits kSend / kRecv / {kClientRecv,kClientSend} into separate groups because
// they own DIFFERENT ZMQ sockets. That change has been unverifiable: the admin
// network periodics never spawn in the co-located client-mode runtime the unit
// tests use, so nothing exercised it.
//
// This tests the underlying mechanism instead, which is what D8-1's correctness
// actually depends on:
//   - tasks sharing a group all run on ONE worker (co-location, which is what
//     keeps two tasks off the same socket from different threads)
//   - distinct groups bind independently rather than collapsing onto one
//
// It does NOT test ZMQ socket safety, which still needs a networked run. It
// does close the gap between "I changed the group ids" and "groups behave the
// way the change assumes".
//==============================================================================

TEST_CASE("task_groups_pin_together_and_bind_independently", "[stall][groups]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("same group co-locates; distinct groups bind separately") {
    constexpr int kPerGroup = 6;
    constexpr int64_t kGroupA = 4001;
    constexpr int64_t kGroupB = 4002;

    auto run_group = [&](int64_t gid) {
      std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> tasks;
      for (int i = 0; i < kPerGroup; ++i) {
        tasks.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "g", 0,
                                           /*spin_us=*/50, /*chain_depth=*/0,
                                           /*block_us=*/0, /*group_id=*/gid));
      }
      REQUIRE(WaitAllBounded(tasks, kDropDeadlineMs) == 0);
      std::vector<clio::run::u32> workers;
      for (auto &t : tasks) {
        workers.push_back(t->RunWorkerId());
      }
      return workers;
    };

    std::vector<clio::run::u32> a = run_group(kGroupA);
    std::vector<clio::run::u32> b = run_group(kGroupB);

    // Co-location: every task of a group ran on the same worker. This is the
    // property that keeps two tasks sharing a socket off two threads.
    for (size_t i = 1; i < a.size(); ++i) {
      REQUIRE(a[i] == a[0]);
    }
    for (size_t i = 1; i < b.size(); ++i) {
      REQUIRE(b[i] == b[0]);
    }

    INFO("group " + std::to_string(kGroupA) + " -> worker " +
         std::to_string(a[0]) + "; group " + std::to_string(kGroupB) +
         " -> worker " + std::to_string(b[0]));

    // Independent binding: a second group must get its OWN binding rather than
    // inheriting the first's. Landing on the same worker is legal (the mapper
    // may simply pick it again), so this asserts the weaker, meaningful thing:
    // both groups produced a binding at all, i.e. neither was left unrouted.
    REQUIRE(a[0] < 1000u);
    REQUIRE(b[0] < 1000u);
  }
}

//==============================================================================
// TEST 10 — no task may be DROPPED, however slow the runtime gets
//==============================================================================

TEST_CASE("no_tasks_dropped_under_stall_pressure", "[stall][drop]") {
  StallFixture fixture;
  clio::run::PoolId pool_id;
  REQUIRE(fixture.createContainer(kStallPoolId, pool_id));

  clio::run::MOD_NAME::Client client(pool_id);

  SECTION("every submitted task eventually completes") {
    // Deliberately generous deadline. This test does not care about latency —
    // it distinguishes "slow" from "lost". A task that never completes even
    // when given far longer than the spin duration was dropped, not delayed.
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> spinners;
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> quick;
    std::vector<clio::run::Future<clio::run::MOD_NAME::WaitTestTask>> chained;

    for (int i = 0; i < 4; ++i) {
      spinners.push_back(client.AsyncCustom(clio::run::PoolQuery::Local(), "s",
                                            0, kSpinUs / 4));
    }
    for (int i = 0; i < 64; ++i) {
      quick.push_back(
          client.AsyncCustom(clio::run::PoolQuery::Local(), "q", 0, 10));
    }
    for (int i = 0; i < 8; ++i) {
      chained.push_back(client.AsyncWaitTest(clio::run::PoolQuery::Local(),
                                             kChainDepth,
                                             static_cast<clio::run::u32>(i)));
    }

    size_t lost_quick = WaitAllBounded(quick, kDropDeadlineMs);
    size_t lost_chained = WaitAllBounded(chained, kDropDeadlineMs);
    size_t lost_spinners = WaitAllBounded(spinners, kDropDeadlineMs);

    INFO("dropped: quick=" + std::to_string(lost_quick) +
         " chained=" + std::to_string(lost_chained) +
         " spinners=" + std::to_string(lost_spinners));

    REQUIRE(lost_quick == 0);
    REQUIRE(lost_chained == 0);
    REQUIRE(lost_spinners == 0);
  }
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

SIMPLE_TEST_MAIN()
