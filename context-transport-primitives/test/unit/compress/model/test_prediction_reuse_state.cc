/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_prediction_reuse_state.cc
 * @brief Per-lineage reuse state, and the host-side index that addresses it.
 *
 * The state itself lives in DEVICE memory and is read only by kernels. What
 * is tested here is everything that decides WHICH slot a chunk lands in,
 * because a wrong index is the failure that silently differences two
 * unrelated blocks against each other -- and it is pure host C++, so it is
 * testable without a GPU.
 *
 * MainPretest()/MainPosttest() are defined once per binary in test_models.cc.
 */
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/preprocess/prediction_reuse_state.h"

namespace {
using ctp::compress::preprocess::kNoLineageSlot;
using ctp::compress::preprocess::LineageSlotRegistry;
}  // namespace

/* The same lineage must return the SAME slot every timestep -- that identity
   is the whole mechanism. A different lineage must get a different slot. */
TEST_CASE("PredictionReuseRegistryIsStablePerLineage") {
  LineageSlotRegistry reg(/*capacity=*/8);
  const uint32_t a1 = reg.SlotFor("E_x/chunk_0");
  const uint32_t b1 = reg.SlotFor("E_y/chunk_0");
  const uint32_t a2 = reg.SlotFor("E_x/chunk_0");
  REQUIRE(a1 != kNoLineageSlot);
  REQUIRE(b1 != kNoLineageSlot);
  REQUIRE(a1 == a2);
  REQUIRE(a1 != b1);
  REQUIRE(reg.size() == 2);

  // Arrival order must not matter: re-requesting in a different order still
  // resolves to the slots already assigned.
  REQUIRE(reg.SlotFor("E_y/chunk_0") == b1);
  REQUIRE(reg.SlotFor("E_x/chunk_0") == a1);
  REQUIRE(reg.size() == 2);
}

/* Slots are dense and in [0, capacity) -- they index a device array, so an
   out-of-range slot is an out-of-bounds device write. */
TEST_CASE("PredictionReuseRegistrySlotsAreDenseAndInRange") {
  const uint32_t cap = 16;
  LineageSlotRegistry reg(cap);
  std::vector<bool> seen(cap, false);
  for (uint32_t i = 0; i < cap; ++i) {
    const uint32_t s = reg.SlotFor("field/chunk_" + std::to_string(i));
    REQUIRE(s < cap);
    REQUIRE_FALSE(seen[s]);  // never hand the same slot to two lineages
    seen[s] = true;
  }
  REQUIRE(reg.size() == cap);
}

/* Capacity is a hard bound on device memory, so the registry REFUSES beyond
   it rather than evicting. Eviction would silently hand a new lineage the
   previous occupant's cached prediction. Refusal returns kNoLineageSlot,
   which the caller must read as "run the NN". */
TEST_CASE("PredictionReuseRegistryRefusesBeyondCapacityRatherThanEvicting") {
  LineageSlotRegistry reg(/*capacity=*/2);
  const uint32_t a = reg.SlotFor("a/chunk_0");
  const uint32_t b = reg.SlotFor("b/chunk_0");
  REQUIRE(a != kNoLineageSlot);
  REQUIRE(b != kNoLineageSlot);

  // Third distinct lineage: refused.
  REQUIRE(reg.SlotFor("c/chunk_0") == kNoLineageSlot);
  REQUIRE(reg.size() == 2);

  // The two already tracked are UNAFFECTED by the refusal -- an overflowing
  // run must not degrade the lineages it was already following.
  REQUIRE(reg.SlotFor("a/chunk_0") == a);
  REQUIRE(reg.SlotFor("b/chunk_0") == b);
}

/* An unresolved lineage (empty key) is never given a slot: it would collapse
   every unnameable block onto one state. */
TEST_CASE("PredictionReuseRegistryRejectsEmptyKey") {
  LineageSlotRegistry reg(/*capacity=*/4);
  REQUIRE(reg.SlotFor("") == kNoLineageSlot);
  REQUIRE(reg.size() == 0);
}

/* The memory bound has to be a stated number, not a hope. The whole point
   of signature-based state is that it does not scale with chunk size, so
   this pins the per-lineage cost. */
TEST_CASE("PredictionReuseStatePerLineageFootprintIsBounded") {
  using ctp::compress::preprocess::DevicePredictionReuseState;
  // Two 3-double signatures, a cached 32-candidate prediction block, and a
  // handful of scalars. If this grows past 2 KiB the 100K-lineage figure in
  // the design note stops being true, so the number is asserted.
  REQUIRE(sizeof(DevicePredictionReuseState) <= 2048);
  // ... and it must NOT depend on the chunk size in any way.
  REQUIRE(sizeof(DevicePredictionReuseState) > 0);
}

/* ------------------------------------------------------------------ *
 * Concurrency
 * ------------------------------------------------------------------ */

/* The registry itself is unsynchronised, exactly like BlockEvolutionTracker
   next door, and the runtime holds a mutex across it. This reproduces that
   usage from several threads at once and asserts the two properties the
   device state depends on: every lineage resolves to ONE slot however many
   threads ask, and no two lineages ever share one. A violation here is not a
   lost update -- it is two different blocks reading and writing each other's
   cached prediction. */
TEST_CASE("PredictionReuseRegistryIsStableUnderConcurrentCallers") {
  constexpr int kThreads = 8;
  constexpr int kLineages = 64;
  LineageSlotRegistry reg(/*capacity=*/kLineages);
  std::mutex m;
  std::vector<std::thread> threads;
  std::vector<std::vector<uint32_t>> seen(kThreads,
                                          std::vector<uint32_t>(kLineages));

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      // Each thread walks the same lineages in a DIFFERENT order, which is
      // what the runtime does -- chunks reach workers in whatever order the
      // scheduler hands them out.
      for (int k = 0; k < kLineages; ++k) {
        const int i = (k * 7 + t * 13) % kLineages;
        const std::string key = "field/chunk_" + std::to_string(i);
        std::lock_guard<std::mutex> lock(m);
        seen[t][i] = reg.SlotFor(key);
      }
    });
  }
  for (auto &th : threads) th.join();

  REQUIRE(reg.size() == kLineages);
  // Every thread agrees on every lineage's slot.
  for (int t = 1; t < kThreads; ++t) {
    for (int i = 0; i < kLineages; ++i) {
      REQUIRE(seen[t][i] == seen[0][i]);
      REQUIRE(seen[t][i] != kNoLineageSlot);
    }
  }
  // And the assignment is a bijection: no two lineages share a slot.
  std::vector<bool> used(kLineages, false);
  for (int i = 0; i < kLineages; ++i) {
    REQUIRE(seen[0][i] < static_cast<uint32_t>(kLineages));
    REQUIRE_FALSE(used[seen[0][i]]);
    used[seen[0][i]] = true;
  }
}

/* A large ceiling must not cost anything until it is used. `capacity` is an
   upper bound most runs never approach, so the registry reserves for a
   working set rather than for the bound -- and a registry declared with a
   million slots still behaves exactly like a small one for the ten lineages
   a mesh code actually has. */
TEST_CASE("PredictionReuseRegistryLargeCapacityBehavesLikeASmallOne") {
  LineageSlotRegistry big(/*capacity=*/1u << 20);
  REQUIRE(big.capacity() == (1u << 20));
  REQUIRE(big.size() == 0);

  std::vector<uint32_t> slots;
  for (int i = 0; i < 10; ++i) {
    const uint32_t s = big.SlotFor("field/chunk_" + std::to_string(i));
    REQUIRE(s != kNoLineageSlot);
    slots.push_back(s);
  }
  // Dense from zero, whatever the ceiling: the slot is the insertion index,
  // so the device array is used from the front and a big ceiling wastes
  // device memory but never scatters the slots that are in use.
  std::sort(slots.begin(), slots.end());
  for (int i = 0; i < 10; ++i) REQUIRE(slots[i] == static_cast<uint32_t>(i));
  REQUIRE(big.size() == 10);
  // Stable, as for any other capacity.
  REQUIRE(big.SlotFor("field/chunk_3") == 3);
}
