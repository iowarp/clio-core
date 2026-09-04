/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Online SGD on chunks the model has never seen -- inputs far outside the
 * shipped weights' feature bounds -- must still converge.
 *
 * The three LAMMPS fields, as the 2000-chunk learning run presented them:
 * mean statistics per field and the ratio the codec actually delivered. The
 * weight file's own bounds are mad <= 0.50 and second_deriv <= 1.006;
 * position sits at 365 and 467 training sigmas on those two inputs.
 *
 * What went wrong on the shipped rule: the trust region bounds the step at a
 * fixed 0.02 in PARAMETER space, and on these inputs a 0.02 move on W1 shifts
 * the ratio head by ~7 standardised units (0.5 on Nyx). Every step overshoots
 * the target, the error flips sign, and the prediction random-walks between
 * the 0.1 floor and the 100x cap -- measured std 39.9 around a mean of 25.4,
 * direction reversing on 45% of steps, over 666 position chunks.
 *
 * Two fixes address it, each env-toggleable: an output-space trust region
 * (CLIO_NEUROPRESS_SGD_OUT_DELTA, 0 = off) bounds the step by what it does to
 * the OUTPUT rather than to the parameters, and a soft feature bound
 * (CLIO_NEUROPRESS_INPUT_BOUND, 0 = off) pulls the far-out inputs back toward
 * the fitted range. Ablation, measured on an A100 with the shipped weights --
 * per field, the final prediction and the worst excursion over the second half
 * of the run:
 *
 *   arm                       position (1.06)    force (2.38)       velocity
 *   -----------------------   ----------------   ----------------   --------
 *   shipped rule (both off)   final 0.10,  94x   final 0.10,  42x   passes
 *   output bound only         final 0.32, 29.9x  final 2.64, 23.8x  passes
 *   input bound only          passes             final 1.18, 2.36x  passes
 *   both (the default now)    final 1.62, 1.94x  final 2.12, 2.18x  passes
 *
 * What is left after both fixes is not the learning rule. Bounded, position
 * and force sit 0.28 sigma apart across the model's ENTIRE input space
 * (entropy, MAD, second derivative) while their targets differ by 2.2x -- the
 * three statistics cannot separate them. The best single prediction for an
 * indistinguishable pair is the geometric mean of 1.06 and 2.38 = 1.59, which
 * leaves each field at exactly 1.50x, and the model lands on that compromise.
 * Closing it needs a per-lineage residual; that is a planned next step, and
 * the [.pending_lineage_residual] case below is its acceptance test.
 *
 * These tests pin the behaviour the fixes must deliver and the behaviour they
 * must not break.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/compress/model/neuropress_nn_gpu_kernels.h"
#include "clio_ctp/compress/model/neuropress_nn_predictor.h"

using ctp::compress::model::CandidateConfig;
using ctp::compress::model::CompressionFeatures;
using ctp::compress::model::DataFeatures;
using ctp::compress::model::MakeCompressionFeatures;
using ctp::compress::model::NeuroPressNNPredictor;
using ctp::compress::model::TrainingLabels;

namespace {

struct Field {
  const char *name;
  double entropy, mad, second_deriv;
  double actual_ratio;
};

// Mean statistics and delivered ratio per field, LAMMPS LJ melt, 4M atoms,
// float32 downcast, eb 1e-3, 666 chunks each (np-cusz/lmp_learn2k).
const Field kLammps[] = {
    {"velocity", 7.372, 1.53, 3.747, 2.30},
    {"position", 7.090, 38.93, 98.95, 1.06},
    {"force", 7.412, 33.07, 83.59, 2.38},
};

CompressionFeatures SampleFor(double entropy, double mad, double deriv) {
  DataFeatures data;
  data.chunk_size_bytes = 8.0 * 1024 * 1024;
  data.shannon_entropy = entropy;
  data.mad = mad;
  data.second_derivative_mean = deriv;
  data.data_type_float = 1.0;
  CandidateConfig candidate;
  candidate.base_id = 13;  // nvcomp-lz4, lossless
  candidate.preset_id = 2;
  return MakeCompressionFeatures(data, candidate);
}

// A chunk every continuous input of which sits inside the shipped bounds.
CompressionFeatures InDistributionSample() {
  return SampleFor(4.964, 0.005744, 0.01159);
}

double LogErr(double pred, double actual) {
  return std::fabs(std::log(std::max(pred, 1e-6) / actual));
}

// One field's trajectory, reduced to the numbers both interleaved cases judge.
struct FieldRun {
  const char *name;
  double actual;
  double first_pred, final_pred;
  double first_err, final_err;
  int first_cross;             // -1 if it never reached the target
  double worst_after_cross;    // log-error, second half of the run
  std::string traj;            // sampled trajectory, for the failure message
};

// Feeds the three LAMMPS-shaped samples round-robin through 300 SGD calls --
// exactly as the in-situ run interleaves x, v and f per frame -- and reduces
// each field's trajectory to a FieldRun. Returns false if the predictor
// declined to load (no CUDA device); the caller decides to SKIP.
bool RunInterleavedFields(std::vector<FieldRun> *out) {
  NeuroPressNNPredictor nn;
  if (!nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR)) return false;
  REQUIRE(nn.IsReady());

  constexpr int kFields = 3;
  constexpr int kRounds = 100;  // 300 SGD calls, 100 per field
  CompressionFeatures feats[kFields];
  std::vector<double> trace[kFields];
  for (int f = 0; f < kFields; ++f) {
    feats[f] = SampleFor(kLammps[f].entropy, kLammps[f].mad,
                         kLammps[f].second_deriv);
    trace[f].push_back(nn.Predict(feats[f]).compression_ratio);
  }

  // Lossless candidate: psnr < 0 withholds that head, decomp 0 withholds
  // that head, so the ratio and compress-time heads carry the error.
  for (int r = 0; r < kRounds; ++r) {
    for (int f = 0; f < kFields; ++f) {
      std::vector<CompressionFeatures> b(1, feats[f]);
      std::vector<TrainingLabels> l(
          1, TrainingLabels(static_cast<float>(kLammps[f].actual_ratio),
                            /*psnr=*/-1.0f, /*comp_time_ms=*/12.0f,
                            /*decomp_time_ms=*/0.0f));
      REQUIRE(nn.Train(b, l));
      trace[f].push_back(nn.Predict(feats[f]).compression_ratio);
    }
  }

  out->clear();
  for (int f = 0; f < kFields; ++f) {
    const double actual = kLammps[f].actual_ratio;
    const auto &t = trace[f];
    FieldRun run;
    run.name = kLammps[f].name;
    run.actual = actual;
    run.first_pred = t.front();
    run.final_pred = t.back();
    run.first_err = LogErr(t.front(), actual);
    run.final_err = LogErr(t.back(), actual);

    // Once the prediction has crossed the target it must stay near it. A
    // rule that overshoots swings back out past 2x on the far side; the
    // shipped rule swings all the way to the cap.
    run.first_cross = -1;
    for (size_t i = 1; i < t.size(); ++i) {
      if ((t[i] - actual) * (t[0] - actual) <= 0.0) {
        run.first_cross = int(i);
        break;
      }
    }
    // Judged over the second half of the run: a momentum-driven undershoot
    // right after the first crossing (velocity dips to 1.08 against 2.30
    // before settling at 2.30 exactly) is the transient, not the failure. A
    // rule that never settles is still caught: position under the shipped
    // rule spends its second half at 0.10 and 100.
    run.worst_after_cross = 0.0;
    if (run.first_cross >= 0) {
      for (size_t i = std::max<size_t>(run.first_cross, t.size() / 2);
           i < t.size(); ++i)
        run.worst_after_cross =
            std::max(run.worst_after_cross, LogErr(t[i], actual));
    }

    // The whole trajectory, so a failure says HOW it failed -- overshoot,
    // oscillation, or never arriving look identical in a final number.
    for (size_t i = 0; i < t.size(); ++i) {
      if (i < 40 || i + 5 >= t.size() || i % 10 == 0) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%s%.2f", i ? " " : "", t[i]);
        run.traj += buf;
      } else if (i == 40) {
        run.traj += " ...";
      }
    }
    out->push_back(run);
  }
  return true;
}

// The per-field verdict, with the two tolerances supplied by the caller: the
// acceptance case and the goal case differ only in how tight they are.
void CheckField(const FieldRun &r, double worst_limit, double final_limit) {
  INFO(r.name << ": first=" << r.first_pred << " final=" << r.final_pred
              << " actual=" << r.actual << " first_cross=" << r.first_cross
              << " worst_after_cross(x)=" << std::exp(r.worst_after_cross)
              << "\n  traj: " << r.traj);
  // CHECK, not REQUIRE: every field is reported, not just the first bad one.
  CHECK(std::isfinite(r.final_pred));
  CHECK(r.final_err < r.first_err);
  CHECK(r.first_cross >= 0);              // it does get there
  CHECK(r.worst_after_cross < worst_limit);
  CHECK(r.final_err < final_limit);
}

}  // namespace

TEST_CASE("Online SGD on out-of-distribution chunks stays bounded and improves "
          "every field (LAMMPS-shaped, three fields interleaved)",
          "[gpu]") {
  std::vector<FieldRun> runs;
  if (!RunInterleavedFields(&runs)) {
    SKIP("NeuroPress requires a CUDA device; Load() declined");
  }
  for (const auto &r : runs) {
    // The thresholds the two fixes must clear. Measured on an A100 with the
    // shipped weights: the shipped rule ends position at 0.10 against 1.06
    // and force at 0.10 against 2.38, with second-half excursions of 94x and
    // 42x; with both fixes on, position ends at 1.62 with a 1.94x excursion
    // and force at 2.12 with a 2.18x one. 2.25x and 1.6x sit just above the
    // 1.50x floor that the indistinguishable position/force pair imposes --
    // their targets differ 2.2x, so the best single prediction is the
    // geometric mean 1.59 and each field lands 1.50x away. So these pass on
    // the fixes and fail on the shipped rule by two orders of magnitude.
    // They are deliberately NOT the goal: that is the case below.
    CheckField(r, /*worst_limit=*/std::log(2.25), /*final_limit=*/std::log(1.6));
  }
}

TEST_CASE("Online SGD converges to each field's own target (LAMMPS-shaped; "
          "needs a per-lineage residual)",
          "[gpu][.pending_lineage_residual]") {
  // The goal, and expected to FAIL today -- hidden by default and registered
  // as its own ctest with WILL_FAIL so that its passing is what removes it.
  // The learning rule is no longer what stands in the way: after the output
  // bound and the input bound, position and force are 0.28 sigma apart over
  // the model's entire input space (entropy, MAD, second derivative) while
  // their targets differ 2.2x, so no function of those three statistics can
  // put both within 1.5x -- the geometric-mean compromise 1.59 leaves each at
  // exactly 1.50x. A per-lineage residual, which gives the model a per-field
  // term the shared statistics cannot supply, is what makes this reachable.
  std::vector<FieldRun> runs;
  if (!RunInterleavedFields(&runs)) {
    SKIP("NeuroPress requires a CUDA device; Load() declined");
  }
  for (const auto &r : runs) {
    CheckField(r, /*worst_limit=*/std::log(2.0), /*final_limit=*/std::log(1.5));
  }
}

TEST_CASE("The output-space bound leaves in-distribution learning intact",
          "[gpu]") {
  // Guard against over-constraining: on a chunk inside the bounds the step
  // already moves the output by ~0.5 sigma, which is the bound itself, so
  // learning must proceed at essentially the shipped rate. 50 steps toward
  // a target a third of the current prediction must close most of the gap.
  NeuroPressNNPredictor nn;
  if (!nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR)) {
    SKIP("NeuroPress requires a CUDA device; Load() declined");
  }
  auto features = InDistributionSample();
  const double before = nn.Predict(features).compression_ratio;
  const double target = before / 3.0;
  std::vector<CompressionFeatures> b(1, features);
  std::vector<TrainingLabels> l(
      1, TrainingLabels(static_cast<float>(target), -1.0f, 6.0f, 0.0f));
  for (int i = 0; i < 50; ++i) REQUIRE(nn.Train(b, l));
  const double after = nn.Predict(features).compression_ratio;
  INFO("before=" << before << " target=" << target << " after=" << after);
  REQUIRE(LogErr(after, target) < 0.3 * LogErr(before, target));
}

#if CTP_ENABLE_NEUROPRESS_GPU
TEST_CASE("NeuroPressSoftBoundSigma is the identity inside the bounds and "
          "logarithmic beyond them") {
  using ctp::compress::model::gpu::NeuroPressSoftBoundSigma;
  // The shipped file's mad bounds in sigma units: [0, 0.50] with
  // mean 0.18917, std 0.10615 -> lo = -1.782, hi = 2.928.
  const float lo = -1.782f, hi = 2.928f;
  for (float v : {-1.782f, -1.0f, 0.0f, 1.5f, 2.928f})
    REQUIRE(NeuroPressSoftBoundSigma(v, lo, hi) == v);
  REQUIRE(NeuroPressSoftBoundSigma(hi + 1e-6f, lo, hi) ==
          Catch::Approx(hi).margin(1e-5));
  float prev = NeuroPressSoftBoundSigma(hi, lo, hi);
  for (float v = hi + 0.5f; v < 500.0f; v *= 1.5f) {
    const float b = NeuroPressSoftBoundSigma(v, lo, hi);
    REQUIRE(b > prev);
    REQUIRE(b < v);
    prev = b;
  }
  // LAMMPS position's mad at 365 sigma lands near 8.8; velocity's 12.6 near
  // 5.3 -- still ordered, an order of magnitude nearer the fitted range.
  REQUIRE(NeuroPressSoftBoundSigma(365.0f, lo, hi) == Catch::Approx(8.82f).margin(0.05));
  REQUIRE(NeuroPressSoftBoundSigma(12.6f, lo, hi) == Catch::Approx(5.30f).margin(0.05));
  REQUIRE(NeuroPressSoftBoundSigma(-40.0f, lo, hi) ==
          Catch::Approx(lo - std::log1p(40.0f + lo)).margin(1e-4));
}
#endif
