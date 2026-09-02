/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file test_prediction_reuse.cc
 * @brief Step divergence, anchor divergence, path divergence and the reuse
 * decision.
 *
 * The arithmetic is `__host__ __device__` so the cases below run on the host,
 * where they are fast and debuggable; the kernel that calls it on the device
 * is covered separately by the GPU integration tests. What is checked here is
 * the part that is easy to get wrong and hard to see afterwards -- the
 * distinction between previous and last_nn state, and the order in which
 * state is written.
 *
 * MainPretest()/MainPosttest() are defined once per binary in test_models.cc.
 */
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/preprocess/prediction_reuse.h"

namespace {
using ctp::compress::preprocess::DevicePredictionReuseState;
using DeviceBlockState = ctp::compress::preprocess::DevicePredictionReuseState;
using ctp::compress::preprocess::DecidePredictionReuse;
using ctp::compress::preprocess::MustRunModel;
using ctp::compress::preprocess::SignatureDivergence;
using ctp::compress::preprocess::ChunkSignature;
using ctp::compress::preprocess::PredictionReuseThresholds;
namespace ReuseDecision = ctp::compress::preprocess::ReuseDecision;

ChunkSignature Sig(double h, double m, double d) {
  ChunkSignature s;
  s.entropy = h; s.mad = m; s.second_derivative = d;
  return s;
}

/** Fresh state, as the device array is zero-initialised. */
DevicePredictionReuseState Fresh() {
  DevicePredictionReuseState s{};
  return s;
}
}  // namespace

/* ------------------------------------------------------------------ *
 * Step divergence
 * ------------------------------------------------------------------ */

TEST_CASE("PredictionReuseDivergenceIdenticalIsZero") {
  const ChunkSignature a = Sig(4.0, 0.5, 0.25);
  REQUIRE(SignatureDivergence(a, a) == 0.0);
  // Including the all-zero signature, which is a real case: WarpX step00000
  // is an all-zero initial condition, and its measured features are 0/0/0.
  const ChunkSignature z = Sig(0.0, 0.0, 0.0);
  REQUIRE(SignatureDivergence(z, z) == 0.0);
}

TEST_CASE("PredictionReuseDivergenceGrowsWithChange") {
  const ChunkSignature base = Sig(4.0, 0.5, 0.25);
  const double small = SignatureDivergence(base, Sig(4.01, 0.5, 0.25));
  const double large = SignatureDivergence(base, Sig(40.0, 5.0, 2.5));
  REQUIRE(small > 0.0);
  REQUIRE(large > small);
}

/* Scale-free is the whole reason this is a RELATIVE measure. WarpX E/j carry
   MAD ~1e10 in physical units, and the model's own x_std for MAD is 0.106 --
   so a sigma-normalised divergence reports ~1e11 for a change of no
   consequence, and every threshold degenerates. Doubling must read the same
   whether the field is O(1) or O(1e10). */
TEST_CASE("PredictionReuseDivergenceIsScaleFree") {
  const double small_scale =
      SignatureDivergence(Sig(1.0, 1.0, 1.0), Sig(2.0, 2.0, 2.0));
  const double huge_scale =
      SignatureDivergence(Sig(1e10, 1e10, 1e10), Sig(2e10, 2e10, 2e10));
  REQUIRE(std::fabs(small_scale - huge_scale) < 1e-12);
}

/* Bounded in [0,1], so a threshold means the same thing on every field and
   cannot be defeated by an outlier. */
TEST_CASE("PredictionReuseDivergenceIsBounded") {
  REQUIRE(SignatureDivergence(Sig(0.0, 0.0, 0.0), Sig(1e30, 1e30, 1e30)) <= 1.0);
  REQUIRE(SignatureDivergence(Sig(-1e30, 1.0, 1.0), Sig(1e30, 1.0, 1.0)) <= 1.0);
  REQUIRE(SignatureDivergence(Sig(3.0, 0.1, 0.2), Sig(9.0, 0.3, 0.6)) >= 0.0);
}

/* A non-finite feature must not produce a small divergence -- that would read
   as "nothing changed" and reuse a stale prediction for a broken chunk. */
TEST_CASE("PredictionReuseDivergenceNonFiniteForcesMaximum") {
  const double nan_d = std::numeric_limits<double>::quiet_NaN();
  const double inf_d = std::numeric_limits<double>::infinity();
  REQUIRE(SignatureDivergence(Sig(4.0, 0.5, 0.25), Sig(nan_d, 0.5, 0.25)) == 1.0);
  REQUIRE(SignatureDivergence(Sig(4.0, 0.5, 0.25), Sig(4.0, inf_d, 0.25)) == 1.0);
  REQUIRE(SignatureDivergence(Sig(nan_d, 0.5, 0.25), Sig(4.0, 0.5, 0.25)) == 1.0);
}

/* ------------------------------------------------------------------ *
 * The decision truth table
 * ------------------------------------------------------------------ */

TEST_CASE("PredictionReuseTriggerFirstObservation") {
  DevicePredictionReuseState s = Fresh();
  PredictionReuseThresholds th;
  const uint32_t r = DecidePredictionReuse(&s, Sig(4.0, 0.5, 0.25), 0, th);
  REQUIRE((r & ReuseDecision::kFirstObservation) != 0u);
  REQUIRE(ctp::compress::preprocess::MustRunModel(r));
}

TEST_CASE("PredictionReuseTriggerMissingPreviousOrAnchorOrPrediction") {
  PredictionReuseThresholds th;
  // Seen before, but nothing cached to reuse -> must run.
  DevicePredictionReuseState s = Fresh();
  s.has_previous = 1; s.has_anchor = 1; s.has_prediction = 0;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  uint32_t r = DecidePredictionReuse(&s, Sig(4.0, 0.5, 0.25), 1, th);
  REQUIRE((r & ReuseDecision::kInvalidPrediction) != 0u);
  REQUIRE(ctp::compress::preprocess::MustRunModel(r));

  // Prediction cached but no anchor to measure drift from -> must run.
  s = Fresh();
  s.has_previous = 1; s.has_anchor = 0; s.has_prediction = 1;
  s.previous = Sig(4.0, 0.5, 0.25);
  r = DecidePredictionReuse(&s, Sig(4.0, 0.5, 0.25), 1, th);
  REQUIRE((r & ReuseDecision::kNoAnchorState) != 0u);
  REQUIRE(ctp::compress::preprocess::MustRunModel(r));
}

TEST_CASE("PredictionReuseTriggerReusesWhenBothDivergencesAreBelowThreshold") {
  PredictionReuseThresholds th;
  th.step = 0.05; th.anchor = 0.20; th.refresh_interval = 1000;
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  s.last_nn_timestep = 0;
  // ~0.25% change: far below both thresholds.
  const uint32_t r = DecidePredictionReuse(&s, Sig(4.02, 0.502, 0.2512), 1, th);
  REQUIRE(r == ReuseDecision::kReuseCachedPrediction);
  REQUIRE_FALSE(ctp::compress::preprocess::MustRunModel(r));
}

TEST_CASE("PredictionReuseTriggerStepAndAnchorThresholdsFireIndependently") {
  PredictionReuseThresholds th;
  th.step = 0.05; th.anchor = 0.20; th.refresh_interval = 1000;

  // Sudden change: step exceeded. Anchor equals previous here, so both fire;
  // what matters is that the STEP reason is present.
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  uint32_t r = DecidePredictionReuse(&s, Sig(12.0, 1.5, 0.75), 1, th);
  REQUIRE((r & ReuseDecision::kStepDivergenceExceeded) != 0u);

  // Slow drift: each step is small, but the distance from the ANCHOR is not.
  // This is the case step divergence alone cannot see, and the reason the two
  // states are kept separately (Rules 12-13).
  s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.last_nn = Sig(4.0, 0.5, 0.25);       // where the model last ran
  s.previous = Sig(6.0, 0.75, 0.375);    // where we were one step ago
  r = DecidePredictionReuse(&s, Sig(6.1, 0.76, 0.381), 9, th);
  REQUIRE((r & ReuseDecision::kStepDivergenceExceeded) == 0u);
  REQUIRE((r & ReuseDecision::kAnchorDivergenceExceeded) != 0u);
  REQUIRE(ctp::compress::preprocess::MustRunModel(r));
}

/* Threshold equality is a documented, single policy: `>` triggers, so D == T
   REUSES. Pinned because a silent flip changes every measured skip rate. */
TEST_CASE("PredictionReuseTriggerThresholdEqualityReuses") {
  PredictionReuseThresholds th;
  th.step = 0.5; th.anchor = 0.5; th.refresh_interval = 1000;
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  // (3 -> 9): |9-3|/(9+3) = 0.5 exactly, on all three features.
  s.previous = s.last_nn = Sig(3.0, 3.0, 3.0);
  const ChunkSignature now = Sig(9.0, 9.0, 9.0);
  REQUIRE(SignatureDivergence(s.previous, now) == 0.5);
  const uint32_t r = DecidePredictionReuse(&s, now, 1, th);
  REQUIRE(r == ReuseDecision::kReuseCachedPrediction);
}

TEST_CASE("PredictionReuseTriggerPeriodicRefresh") {
  PredictionReuseThresholds th;
  th.step = 1.0; th.anchor = 1.0;   // divergence can never fire
  th.refresh_interval = 5;
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  s.last_nn_timestep = 10;
  // The interval is counted in OBSERVATIONS of the lineage, not simulation
  // time: the model ran at observation 0, and this is observation 4.
  s.last_nn_observation = 0;
  s.observation_count = 4;
  const ChunkSignature now = Sig(4.0, 0.5, 0.25);
  // 4 - 0 < interval -> reuse, however far apart the timestamps are.
  REQUIRE(DecidePredictionReuse(&s, now, 14, th) ==
          ReuseDecision::kReuseCachedPrediction);
  REQUIRE(DecidePredictionReuse(&s, now, 140, th) ==
          ReuseDecision::kReuseCachedPrediction);
  // Observation 5 since the run: refresh (>= is the documented policy).
  s.observation_count = 5;
  const uint32_t r = DecidePredictionReuse(&s, now, 15, th);
  REQUIRE((r & ReuseDecision::kPeriodicRefresh) != 0u);
}

/* An out-of-order or repeated timestep means the series is not what
   the state describes. Run the model rather than reuse against it. */
TEST_CASE("PredictionReuseTriggerNonAdvancingTimestepRunsTheModel") {
  PredictionReuseThresholds th;
  th.step = 1.0; th.anchor = 1.0; th.refresh_interval = 1000;
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  s.previous_timestep = 10; s.last_nn_timestep = 10;
  const ChunkSignature now = Sig(4.0, 0.5, 0.25);
  REQUIRE((DecidePredictionReuse(&s, now, 9, th) &
           ReuseDecision::kTimestepNotAdvancing) != 0u);   // backwards
  REQUIRE((DecidePredictionReuse(&s, now, 10, th) &
           ReuseDecision::kTimestepNotAdvancing) != 0u);   // repeated
  REQUIRE((DecidePredictionReuse(&s, now, 11, th) &
           ReuseDecision::kTimestepNotAdvancing) == 0u);   // forwards is fine
}

/* The ordering bug this test exists to catch. If the trigger wrote
   `previous` before measuring, it would compare the current signature with
   itself and report zero divergence forever. Evaluate must therefore NOT
   modify the state at all; committing is a separate, explicit step. */
TEST_CASE("PredictionReuseTriggerEvaluateDoesNotWriteState") {
  PredictionReuseThresholds th;
  th.step = 0.05; th.anchor = 0.20; th.refresh_interval = 1000;
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  const ChunkSignature now = Sig(12.0, 1.5, 0.75);

  const uint32_t first = DecidePredictionReuse(&s, now, 1, th);
  REQUIRE((first & ReuseDecision::kStepDivergenceExceeded) != 0u);
  // Same call again must give the SAME answer. It would not if `previous` had
  // been advanced to `now` by the first call.
  const uint32_t again = DecidePredictionReuse(&s, now, 1, th);
  REQUIRE(again == first);
  REQUIRE(s.previous.entropy == 4.0);
}

/* Rules 10-11 -- the two commits are DIFFERENT. previous advances every
   timestep; last_nn only when the model ran. */
TEST_CASE("PredictionReuseCommitAdvancesPreviousAlwaysAnchorOnlyOnRun") {
  using ctp::compress::preprocess::CommitChunkObservation;
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  s.previous_timestep = 0; s.last_nn_timestep = 0;

  // Reused: previous advances, anchor must NOT.
  CommitChunkObservation(&s, Sig(5.0, 0.6, 0.3), 1, /*model_ran=*/false);
  REQUIRE(s.previous.entropy == 5.0);
  REQUIRE(s.previous_timestep == 1);
  REQUIRE(s.last_nn.entropy == 4.0);
  REQUIRE(s.last_nn_timestep == 0);

  // Model ran: both advance.
  CommitChunkObservation(&s, Sig(6.0, 0.7, 0.35), 2, /*model_ran=*/true);
  REQUIRE(s.previous.entropy == 6.0);
  REQUIRE(s.previous_timestep == 2);
  REQUIRE(s.last_nn.entropy == 6.0);
  REQUIRE(s.last_nn_timestep == 2);
  REQUIRE(s.has_anchor == 1);
}

/* ------------------------------------------------------------------ *
 * Anchor divergence is NOT summed step divergence
 * ------------------------------------------------------------------ */

/* An oscillation A->B->A->B->A returns to where the model last ran, so the
   ANCHOR divergence comes back to zero while the PATH keeps accumulating. */
TEST_CASE("PredictionReuseAnchorIsNotPathLength") {
  using ctp::compress::preprocess::CommitChunkObservation;
  const ChunkSignature A = Sig(4.0, 0.5, 0.25);
  const ChunkSignature B = Sig(8.0, 1.0, 0.50);
  PredictionReuseThresholds th;
  th.step = 1.0; th.anchor = 1.0; th.refresh_interval = 1000;  // never fire

  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = A;          // model last ran on A
  s.previous_timestep = s.last_nn_timestep = 0;

  long long t = 1;
  for (const ChunkSignature &sig : {B, A, B, A}) {
    DecidePredictionReuse(&s, sig, t, th);
    CommitChunkObservation(&s, sig, t, /*model_ran=*/false);
    ++t;
  }
  // Back at A: distance from the anchor is zero...
  REQUIRE(s.anchor_divergence == 0.0);
  // ...while the path travelled is not, and counts every leg.
  REQUIRE(s.path_divergence > 0.0);
  REQUIRE(s.path_divergence > s.anchor_divergence);
}

/* Path divergence is diagnostic and must not affect the decision. */
TEST_CASE("PredictionReusePathDivergenceDoesNotAffectTheTrigger") {
  using ctp::compress::preprocess::CommitChunkObservation;
  PredictionReuseThresholds th;
  th.step = 0.5; th.anchor = 0.5; th.refresh_interval = 1000;
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);

  // Accumulate a large path by oscillating, without ever exceeding a
  // threshold relative to previous or anchor.
  for (long long t = 1; t <= 20; ++t) {
    const ChunkSignature sig =
        (t % 2) ? Sig(4.4, 0.55, 0.275) : Sig(4.0, 0.5, 0.25);
    DecidePredictionReuse(&s, sig, t, th);
    CommitChunkObservation(&s, sig, t, /*model_ran=*/false);
  }
  REQUIRE(s.path_divergence > 0.5);   // well past both thresholds
  const uint32_t r = DecidePredictionReuse(&s, Sig(4.0, 0.5, 0.25), 21, th);
  REQUIRE(r == ReuseDecision::kReuseCachedPrediction);
}

/* ------------------------------------------------------------------ *
 * Layout edge cases
 * ------------------------------------------------------------------ */

/* CHUNK SIZE IS AN NN INPUT (slot 4), and the error bound is another (slot 3).
   A lineage whose chunk size changes -- a short final chunk, a reconfigured
   run -- would be handed a prediction the model made for a DIFFERENT input,
   and the signature cannot see it: entropy, MAD and the second derivative are
   all intensive, so halving the chunk changes none of them. Nothing else in
   this design catches that, which is why it is checked explicitly. */
TEST_CASE("PredictionReuseTriggerLayoutChangeForcesTheModelToRun") {
  PredictionReuseThresholds th;
  th.step = 1.0; th.anchor = 1.0; th.refresh_interval = 1000;  // never fire
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  s.cached_chunk_bytes = 8388608;
  s.cached_error_bound = 0.05;
  const ChunkSignature now = Sig(4.0, 0.5, 0.25);  // identical statistics

  // Same layout: reuse.
  REQUIRE(DecidePredictionReuse(&s, now, 1, th, 8388608, 0.05) ==
          ReuseDecision::kReuseCachedPrediction);

  // Half the chunk, identical statistics -- the model saw a different input 4.
  REQUIRE((DecidePredictionReuse(&s, now, 2, th, 4194304, 0.05) &
           ReuseDecision::kBlockLayoutChanged) != 0u);

  // Different error bound -- input 3.
  REQUIRE((DecidePredictionReuse(&s, now, 3, th, 8388608, 0.01) &
           ReuseDecision::kBlockLayoutChanged) != 0u);
}

/* A lineage that disappears and comes back must not be differenced against a
   stale `previous` as though no time had passed. The timestep guard is what
   makes that safe, and the divergence is still measured honestly. */
TEST_CASE("PredictionReuseTriggerLineageReturningAfterAGapStillAdvances") {
  using ctp::compress::preprocess::CommitChunkObservation;
  PredictionReuseThresholds th;
  th.step = 0.5; th.anchor = 0.5; th.refresh_interval = 1000;
  DevicePredictionReuseState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = Sig(4.0, 0.5, 0.25);
  s.previous_timestep = 5; s.last_nn_timestep = 5;
  s.cached_chunk_bytes = 1024; s.cached_error_bound = 0.05;

  // Returns 100 timesteps later, unchanged data: forward in time, so no
  // stall flag, and the divergence genuinely is zero.
  const uint32_t r =
      DecidePredictionReuse(&s, Sig(4.0, 0.5, 0.25), 105, th, 1024, 0.05);
  REQUIRE((r & ReuseDecision::kTimestepNotAdvancing) == 0u);
  REQUIRE(s.step_divergence == 0.0);
}

/* ------------------------------------------------------------------ *
 * Exploration is exempt, unconditionally
 * ------------------------------------------------------------------ */

/* An exploring run ranks and measures the FULL action space every timestep,
   however little the data moved. Reuse must therefore never be consulted for
   it -- not "rarely", not "only above a threshold". Divergence does not enter
   this decision at all, which is why the predicate does not take one. */
TEST_CASE("PredictionReuseNeverAppliesWhileExploring") {
  using ctp::compress::preprocess::PredictionReuseAllowed;

  // Exploring: refused, whichever way the feature switch is set.
  REQUIRE_FALSE(PredictionReuseAllowed(/*reuse_enabled=*/true,
                                       /*exploration_enabled=*/true));
  REQUIRE_FALSE(PredictionReuseAllowed(false, true));

  // Not exploring: the feature switch alone decides.
  REQUIRE(PredictionReuseAllowed(true, false));
  REQUIRE_FALSE(PredictionReuseAllowed(false, false));
}

/* The same statement from the other side: even a chunk whose statistics are
   BIT-IDENTICAL to the previous timestep -- divergence exactly zero, the most
   reusable case there is -- is still refused while exploring. This is the
   case a threshold-based reading of the rule would get wrong. */
TEST_CASE("PredictionReuseRefusedWhileExploringEvenAtZeroDivergence") {
  using ctp::compress::preprocess::PredictionReuseAllowed;
  const ChunkSignature same = Sig(4.0, 0.5, 0.25);
  REQUIRE(SignatureDivergence(same, same) == 0.0);   // maximally reusable

  DeviceBlockState s = Fresh();
  s.has_previous = s.has_anchor = s.has_prediction = 1;
  s.previous = s.last_nn = same;
  PredictionReuseThresholds th;
  // The trigger, consulted directly, would say reuse...
  REQUIRE(DecidePredictionReuse(&s, same, 1, th) ==
          ReuseDecision::kReuseCachedPrediction);
  // ...and it is never consulted, because exploration is enabled.
  REQUIRE_FALSE(PredictionReuseAllowed(true, /*exploration_enabled=*/true));
}

/* ------------------------------------------------------------------ *
 * The refresh counts OBSERVATIONS, not timesteps
 * ------------------------------------------------------------------ */

/* WarpX writes every 10 simulation steps; with the interval measured in
   simulation time every observation satisfied t - last_nn >= 10 and nothing
   was ever reused there, while Nyx (every step) refreshed once in 120. The
   cadence of a refresh must not depend on how often a simulation writes. */
TEST_CASE("PredictionReuseRefreshCountsObservationsNotSimulationTime") {
  using ctp::compress::preprocess::CommitChunkObservation;
  PredictionReuseThresholds th;
  th.step = 1.0; th.anchor = 1.0;   // divergence can never fire
  th.refresh_interval = 10;
  DeviceBlockState s = Fresh();
  const ChunkSignature same = Sig(4.0, 0.5, 0.25);

  // First observation at t=0 runs the model.
  REQUIRE(MustRunModel(DecidePredictionReuse(&s, same, 0, th)));
  CommitChunkObservation(&s, same, 0, /*model_ran=*/true);
  s.has_prediction = 1;   // the commit KERNEL sets this; the inline commit
                          // only advances the two signatures

  // WarpX cadence: t = 10, 20, ... 90. Nine further observations, each 10
  // simulation steps apart. None of them is the tenth observation since the
  // model ran, so ALL must reuse -- even though t - last_nn is 10, 20, ...
  for (long long t = 10; t <= 90; t += 10) {
    const uint32_t r = DecidePredictionReuse(&s, same, t, th);
    REQUIRE(r == ReuseDecision::kReuseCachedPrediction);
    CommitChunkObservation(&s, same, t, /*model_ran=*/false);
  }
  // The tenth observation since the run: refresh.
  const uint32_t r = DecidePredictionReuse(&s, same, 100, th);
  REQUIRE((r & ReuseDecision::kPeriodicRefresh) != 0u);
  CommitChunkObservation(&s, same, 100, /*model_ran=*/true);

  // The count resets when the model runs: the next nine reuse again.
  for (long long t = 110; t <= 190; t += 10) {
    REQUIRE(DecidePredictionReuse(&s, same, t, th) ==
            ReuseDecision::kReuseCachedPrediction);
    CommitChunkObservation(&s, same, t, /*model_ran=*/false);
  }
  REQUIRE((DecidePredictionReuse(&s, same, 200, th) &
           ReuseDecision::kPeriodicRefresh) != 0u);
}

/* The same rule from the other side: with consecutive timesteps the tenth
   observation is also t = 10, so the two readings agree there. That is the
   case the original formula was tested on, which is why it passed. */
TEST_CASE("PredictionReuseRefreshAgreesWithTimestepsWhenConsecutive") {
  using ctp::compress::preprocess::CommitChunkObservation;
  PredictionReuseThresholds th;
  th.step = 1.0; th.anchor = 1.0; th.refresh_interval = 5;
  DeviceBlockState s = Fresh();
  const ChunkSignature same = Sig(4.0, 0.5, 0.25);
  DecidePredictionReuse(&s, same, 0, th);
  CommitChunkObservation(&s, same, 0, true);
  s.has_prediction = 1;
  for (long long t = 1; t <= 4; ++t) {
    REQUIRE(DecidePredictionReuse(&s, same, t, th) ==
            ReuseDecision::kReuseCachedPrediction);
    CommitChunkObservation(&s, same, t, false);
  }
  REQUIRE((DecidePredictionReuse(&s, same, 5, th) &
           ReuseDecision::kPeriodicRefresh) != 0u);
}

/* ------------------------------------------------------------------ *
 * A cached prediction dies with the weights it came from
 * ------------------------------------------------------------------ */

/* In learn mode an SGD step changes the weights without changing any
   input, so an unchanged signature can still yield a changed prediction; on
   Nyx that cost 11% of stored bytes where inference mode cost 0.02%. The SGD
   call counter, which lives in device memory beside the weights, is the
   exact signal. A null counter disables the check, which is what an
   inference-only caller (SGD never runs) gets for free. */
TEST_CASE("PredictionReuseInvalidatedWhenWeightsChange") {
  using ctp::compress::preprocess::CommitChunkObservation;
  PredictionReuseThresholds th;
  th.step = 1.0; th.anchor = 1.0; th.refresh_interval = 1000;   // never fire
  const ChunkSignature same = Sig(4.0, 0.5, 0.25);
  DeviceBlockState s = Fresh();
  int sgd_calls = 3;   // stands in for w->sgd_call_count on the host

  DecidePredictionReuse(&s, same, 0, th, -1.0, -1.0, &sgd_calls);
  CommitChunkObservation(&s, same, 0, /*model_ran=*/true, &sgd_calls);
  s.has_prediction = 1;   // set by the commit KERNEL in production

  // Weights unchanged since the cache was made -> reuse.
  REQUIRE(DecidePredictionReuse(&s, same, 1, th, -1.0, -1.0, &sgd_calls) ==
          ReuseDecision::kReuseCachedPrediction);
  CommitChunkObservation(&s, same, 1, /*model_ran=*/false, &sgd_calls);

  // An SGD step landed. Same data, same signature, ZERO divergence -- and
  // the cache must still be refused, because it was made by a different
  // model.
  sgd_calls = 4;
  const uint32_t r = DecidePredictionReuse(&s, same, 2, th, -1.0, -1.0,
                                           &sgd_calls);
  REQUIRE((r & ReuseDecision::kModelChanged) != 0u);
  REQUIRE(s.step_divergence == 0.0);
  CommitChunkObservation(&s, same, 2, /*model_ran=*/true, &sgd_calls);

  // Re-cached under the new weights -> reuse resumes.
  REQUIRE(DecidePredictionReuse(&s, same, 3, th, -1.0, -1.0, &sgd_calls) ==
          ReuseDecision::kReuseCachedPrediction);

  // No counter supplied: the check is off, as for an inference-only caller.
  REQUIRE(DecidePredictionReuse(&s, same, 4, th) ==
          ReuseDecision::kReuseCachedPrediction);
}
