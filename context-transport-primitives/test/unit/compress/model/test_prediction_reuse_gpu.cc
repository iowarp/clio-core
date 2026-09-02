/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_prediction_reuse_gpu.cc
 * @brief The device half of prediction reuse: that the state really lives
 * on the GPU, that the kernels reach the same verdict the host tests assert,
 * and that a lineage's slots stay independent.
 *
 * These need a device and are tagged [gpu] so they can be selected on a node
 * that has one. The arithmetic itself is covered on the host in
 * test_prediction_reuse.cc -- what is checked here is the plumbing: kernel
 * launch, slot addressing, and the previous-vs-anchor update ordering as the
 * COMMIT KERNEL performs it rather than as the inline function does.
 *
 * MainPretest()/MainPosttest() are defined once per binary in test_models.cc.
 */
#include <cstdint>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"
#include "clio_ctp/compress/preprocess/prediction_reuse_gpu.h"
#include "clio_ctp/util/gpu_api.h"

namespace {
using ctp::compress::preprocess::DevicePredictionReuseState;
using ctp::compress::preprocess::kNoLineageSlot;
using ctp::compress::preprocess::LaunchReuseCommit;
using ctp::compress::preprocess::LaunchReuseDecision;
using ctp::compress::preprocess::ReuseStatesAlloc;
using ctp::compress::preprocess::ReuseStatesFree;
using ctp::compress::preprocess::ReuseStatesRead;
using ctp::compress::preprocess::PredictionReuseContext;
namespace ReuseDecision = ctp::compress::preprocess::ReuseDecision;

/** A DeviceFeatureStats on the device, carrying the three given features. */
ctp::DeviceFeatureStats *MakeStats(double h, double m, double d) {
  ctp::DeviceFeatureStats host{h, m, d};
  auto *p = ctp::GpuApi::Malloc<ctp::DeviceFeatureStats>(
      sizeof(ctp::DeviceFeatureStats));
  if (p == nullptr) return nullptr;
  ctp::GpuApi::Memcpy(p, &host, sizeof(host));
  return p;
}

/** Run one timestep through both kernels, as the inference path does. */
void Step(PredictionReuseContext *ctx, const ctp::DeviceFeatureStats *stats,
          long long t, bool pretend_model_ran) {
  ctx->timestep = t;
  REQUIRE(LaunchReuseDecision(*ctx, stats, nullptr));
  // The commit kernel decides "did the model run" from the flags the trigger
  // wrote, so a caller cannot force it; `pretend_model_ran` only chooses
  // whether we hand it a ranking to cache.
  std::vector<double> scores(32, 1.0);
  std::vector<int> order(32, 0);
  std::vector<float> f(32, 1.0F);
  auto *d_s = ctp::GpuApi::Malloc<double>(sizeof(double) * 32);
  auto *d_o = ctp::GpuApi::Malloc<int>(sizeof(int) * 32);
  auto *d_f = ctp::GpuApi::Malloc<float>(sizeof(float) * 32);
  ctp::GpuApi::Memcpy(d_s, scores.data(), sizeof(double) * 32);
  ctp::GpuApi::Memcpy(d_o, order.data(), sizeof(int) * 32);
  ctp::GpuApi::Memcpy(d_f, f.data(), sizeof(float) * 32);
  REQUIRE(LaunchReuseCommit(*ctx, stats, d_s, d_o, d_f, d_f, d_f, d_f, 32,
                               nullptr));
  ctp::GpuApi::Free(d_s);
  ctp::GpuApi::Free(d_o);
  ctp::GpuApi::Free(d_f);
  (void)pretend_model_ran;
}
}  // namespace

/* The state is device memory: allocation succeeds, reads come back, and it is
   zero-initialised so a fresh slot reads as a first observation. */
TEST_CASE("PredictionReuseGpuStateAllocatesAndZeroes", "[gpu]") {
  void *states = ReuseStatesAlloc(8);
  REQUIRE(states != nullptr);
  DevicePredictionReuseState s{};
  REQUIRE(ReuseStatesRead(states, 3, &s));
  REQUIRE(s.has_previous == 0);
  REQUIRE(s.has_anchor == 0);
  REQUIRE(s.has_prediction == 0);
  REQUIRE(s.cached_count == 0);
  ReuseStatesFree(states);
}

/* Writing one lineage's slot must not disturb another's. A slot
   collision is the failure that reuses one block's prediction for another. */
TEST_CASE("PredictionReuseGpuSlotsAreIndependent", "[gpu]") {
  void *states = ReuseStatesAlloc(4);
  REQUIRE(states != nullptr);
  auto *stats = MakeStats(4.0, 0.5, 0.25);
  REQUIRE(stats != nullptr);

  PredictionReuseContext ctx;
  ctx.states = states;
  ctx.slot = 1;
  Step(&ctx, stats, 0, true);

  DevicePredictionReuseState touched{}, untouched{};
  REQUIRE(ReuseStatesRead(states, 1, &touched));
  REQUIRE(ReuseStatesRead(states, 2, &untouched));
  REQUIRE(touched.has_previous == 1);
  REQUIRE(touched.previous.entropy == 4.0);
  REQUIRE(untouched.has_previous == 0);   // neighbour untouched
  REQUIRE(untouched.previous.entropy == 0.0);

  ctp::GpuApi::Free(stats);
  ReuseStatesFree(states);
}

/* The device reaches the same verdict as the host truth table: a first
   observation runs the model, an unchanged signature afterwards reuses. */
TEST_CASE("PredictionReuseGpuTriggerMatchesTheHostTruthTable", "[gpu]") {
  void *states = ReuseStatesAlloc(2);
  REQUIRE(states != nullptr);
  auto *stats = MakeStats(4.0, 0.5, 0.25);
  REQUIRE(stats != nullptr);

  PredictionReuseContext ctx;
  ctx.states = states;
  ctx.slot = 0;
  ctx.thresholds.step = 0.05;
  ctx.thresholds.anchor = 0.20;
  ctx.thresholds.refresh_interval = 1000;

  Step(&ctx, stats, 0, true);
  DevicePredictionReuseState s{};
  REQUIRE(ReuseStatesRead(states, 0, &s));
  REQUIRE((s.decision_flags & ReuseDecision::kFirstObservation) != 0u);
  REQUIRE(s.has_prediction == 1);
  REQUIRE(s.has_anchor == 1);
  REQUIRE(s.last_nn_timestep == 0);

  // Same data one timestep on: nothing moved, so reuse.
  Step(&ctx, stats, 1, false);
  REQUIRE(ReuseStatesRead(states, 0, &s));
  REQUIRE(s.decision_flags == ReuseDecision::kReuseCachedPrediction);
  REQUIRE(s.step_divergence == 0.0);
  REQUIRE(s.previous_timestep == 1);
  // The anchor did NOT move, because the model did not run.
  REQUIRE(s.last_nn_timestep == 0);

  ctp::GpuApi::Free(stats);
  ReuseStatesFree(states);
}

/* On the device: a big change runs the model, and only THEN does the
   anchor advance. This is the commit kernel's branch, not the inline
   function's -- it reads "did the model run" out of device memory. */
TEST_CASE("PredictionReuseGpuAnchorAdvancesOnlyWhenTheModelRuns", "[gpu]") {
  void *states = ReuseStatesAlloc(2);
  auto *quiet = MakeStats(4.0, 0.5, 0.25);
  auto *loud = MakeStats(40.0, 5.0, 2.5);
  REQUIRE(states != nullptr);
  REQUIRE(quiet != nullptr);
  REQUIRE(loud != nullptr);

  PredictionReuseContext ctx;
  ctx.states = states;
  ctx.slot = 0;
  ctx.thresholds.step = 0.05;
  ctx.thresholds.anchor = 0.20;
  ctx.thresholds.refresh_interval = 1000;

  Step(&ctx, quiet, 0, true);      // first observation -> runs
  Step(&ctx, quiet, 1, false);     // unchanged -> reuse, anchor frozen at t=0
  DevicePredictionReuseState s{};
  REQUIRE(ReuseStatesRead(states, 0, &s));
  REQUIRE(s.last_nn_timestep == 0);

  Step(&ctx, loud, 2, true);       // sudden change -> runs, anchor moves
  REQUIRE(ReuseStatesRead(states, 0, &s));
  REQUIRE((s.decision_flags & ReuseDecision::kStepDivergenceExceeded) != 0u);
  REQUIRE(s.last_nn_timestep == 2);
  REQUIRE(s.last_nn.entropy == 40.0);

  ctp::GpuApi::Free(quiet);
  ctp::GpuApi::Free(loud);
  ReuseStatesFree(states);
}

/* With both thresholds unreachable, the refresh is
   the only thing that can fire -- and it must. */
TEST_CASE("PredictionReuseGpuPeriodicRefreshFires", "[gpu]") {
  void *states = ReuseStatesAlloc(2);
  auto *stats = MakeStats(4.0, 0.5, 0.25);
  REQUIRE(states != nullptr);
  REQUIRE(stats != nullptr);

  PredictionReuseContext ctx;
  ctx.states = states;
  ctx.slot = 0;
  ctx.thresholds.step = 1.0;      // unreachable: divergence is bounded by 1
  ctx.thresholds.anchor = 1.0;
  ctx.thresholds.refresh_interval = 4;

  Step(&ctx, stats, 0, true);
  DevicePredictionReuseState s{};
  for (long long t = 1; t <= 3; ++t) {
    Step(&ctx, stats, t, false);
    REQUIRE(ReuseStatesRead(states, 0, &s));
    REQUIRE(s.decision_flags == ReuseDecision::kReuseCachedPrediction);
  }
  Step(&ctx, stats, 4, true);     // t - last_nn == 4 == interval
  REQUIRE(ReuseStatesRead(states, 0, &s));
  REQUIRE((s.decision_flags & ReuseDecision::kPeriodicRefresh) != 0u);
  REQUIRE(s.last_nn_timestep == 4);

  ctp::GpuApi::Free(stats);
  ReuseStatesFree(states);
}

/* A disabled context must be inert -- this is what guarantees that a chunk
   with no resolvable lineage, and every exploration chunk, runs the model. */
TEST_CASE("PredictionReuseGpuDisabledContextIsInert", "[gpu]") {
  auto *stats = MakeStats(4.0, 0.5, 0.25);
  REQUIRE(stats != nullptr);
  PredictionReuseContext off;               // states=nullptr, slot=kNoSlot
  REQUIRE_FALSE(LaunchReuseDecision(off, stats, nullptr));

  void *states = ReuseStatesAlloc(2);
  PredictionReuseContext no_slot;
  no_slot.states = states;
  no_slot.slot = kNoLineageSlot;
  REQUIRE_FALSE(LaunchReuseDecision(no_slot, stats, nullptr));

  ctp::GpuApi::Free(stats);
  ReuseStatesFree(states);
}
