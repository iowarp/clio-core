/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Clio's NeuroPressChunkDiag must mean what upstream's gpucompress_chunk_diag_t
 * means, so this compiles upstream's own gpucompress_diagnostics.cpp and diffs
 * the two records. Pins the contract, not two pipelines' values -- that Clio's
 * runtime fills its record correctly is test_neuropress_training_inputs.cc.
 */
#include "simple_test.h"

#include <cmath>
#include <cstring>
#include <string>

#include <clio_cte/compressor/neuropress_chunk_diag.h>

#include "gpucompress.h"
#include "api/internal.hpp"

namespace {

using clio::cte::compressor::NeuroPressChunkDiag;

/* Distinct and non-zero, so a zeroed or swapped field cannot match. */
constexpr int kPrimaryAction = 13;   // quantize half: 13/8 == 1
constexpr int kFinalAction = 22;     // a different action, as after a swap
constexpr double kErrorBound = 1e-3;
constexpr size_t kInputSize = 4u << 20;
constexpr float kEntropy = 6.75f;
constexpr float kMad = 0.4231f;
constexpr float kDeriv = 0.1177f;

gpucompress::ChunkDiagInput MakeUpstreamInput(bool explored) {
  gpucompress::ChunkDiagInput d{};
  d.nn_action = explored ? kFinalAction : kPrimaryAction;
  d.nn_original_action = kPrimaryAction;
  d.exploration_triggered = explored;
  d.sgd_fired = true;
  d.input_size = kInputSize;
  d.error_bound = kErrorBound;
  d.h_feat_entropy = kEntropy;
  d.h_feat_mad = kMad;
  d.h_feat_deriv = kDeriv;
  d.h_stats_valid = true;
  // Upstream's sentinel for "no oracle to regret against".
  d.regret = explored ? 0.25f : -1.0f;
  d.cost_model_error_pct = 0.42f;
  d.ratio_mape = 0.31f;
  d.comp_time_mape = 0.17f;
  d.top_actions = nullptr;
  d.top_actions_count = 0;
  d.predicted_costs = nullptr;
  return d;
}

/** Clio's record, filled from the SAME logical values. */
NeuroPressChunkDiag MakeClioRow(bool explored) {
  NeuroPressChunkDiag c;
  c.nn_action = explored ? kFinalAction : kPrimaryAction;
  c.nn_original_action = kPrimaryAction;
  c.exploration_triggered = explored ? 1 : 0;
  c.sgd_fired = 1;
  c.feat_action = kPrimaryAction;
  c.feat_entropy = kEntropy;
  c.feat_mad = kMad;
  c.feat_deriv = kDeriv;
  c.feat_eb_enc = static_cast<float>(kErrorBound);
  c.feat_ds_enc = static_cast<float>(kInputSize);
  c.regret = explored ? 0.25f : -1.0f;
  c.cost_model_error_pct = 0.42f;
  c.ratio_mape = 0.31f;
  c.comp_time_mape = 0.17f;
  return c;
}

}  // namespace

TEST_CASE("Clio's chunk record carries upstream's semantics",
          "[compressor][neuropress][diag][parity][693]") {
  for (int explored = 0; explored <= 1; ++explored) {
    const bool ex = (explored != 0);
    // Parenthesized: INFO streams, and << binds tighter than ?:.
    INFO((ex ? "exploration triggered" : "no exploration"));

    // ---- upstream ----
    gpucompress_reset_chunk_history();
    REQUIRE(gpucompress_get_chunk_history_count() == 0);
    const int slot = gpucompress::recordChunkDiagnostic(MakeUpstreamInput(ex));
    REQUIRE(slot >= 0);
    REQUIRE(gpucompress_get_chunk_history_count() == 1);

    gpucompress_chunk_diag_t up{};
    REQUIRE(gpucompress_get_chunk_diag(0, &up) == 0);

    // ---- Clio ----
    clio::cte::compressor::NeuroPressResetChunkHistory();
    REQUIRE(clio::cte::compressor::NeuroPressChunkHistoryCount() == 0);
    const int cslot =
        clio::cte::compressor::NeuroPressRecordChunkDiag(MakeClioRow(ex));
    REQUIRE(cslot >= 0);
    REQUIRE(clio::cte::compressor::NeuroPressChunkHistoryCount() == 1);

    NeuroPressChunkDiag cl;
    REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(0, &cl) == 0);

    // feat_action follows the ORIGINAL action on both sides, even when the
    // final differs.
    REQUIRE(up.feat_action == up.nn_original_action);
    REQUIRE(cl.feat_action == cl.nn_original_action);
    REQUIRE(up.feat_action == cl.feat_action);
    if (ex) {
      // The two really diverge, so the check above is discriminating.
      REQUIRE(up.nn_action != up.nn_original_action);
      REQUIRE(cl.nn_action != cl.nn_original_action);
    }
    REQUIRE(up.nn_action == cl.nn_action);
    REQUIRE(up.exploration_triggered == cl.exploration_triggered);
    REQUIRE(up.sgd_fired == cl.sgd_fired);

    // The raw configured bound: the field that was defaulted away.
    REQUIRE(static_cast<double>(up.feat_eb_enc) ==
            static_cast<double>(static_cast<float>(kErrorBound)));
    REQUIRE(static_cast<double>(cl.feat_eb_enc) ==
            static_cast<double>(up.feat_eb_enc));
    REQUIRE(static_cast<double>(up.feat_ds_enc) ==
            static_cast<double>(static_cast<float>(kInputSize)));
    REQUIRE(static_cast<double>(cl.feat_ds_enc) ==
            static_cast<double>(up.feat_ds_enc));
    REQUIRE(static_cast<double>(cl.feat_entropy) ==
            static_cast<double>(up.feat_entropy));
    REQUIRE(static_cast<double>(cl.feat_mad) == static_cast<double>(up.feat_mad));
    REQUIRE(static_cast<double>(cl.feat_deriv) ==
            static_cast<double>(up.feat_deriv));

    // ---- cost model and regret ----
    REQUIRE(static_cast<double>(cl.cost_model_error_pct) ==
            static_cast<double>(up.cost_model_error_pct));
    REQUIRE(static_cast<double>(cl.ratio_mape) ==
            static_cast<double>(up.ratio_mape));
    REQUIRE(static_cast<double>(cl.comp_time_mape) ==
            static_cast<double>(up.comp_time_mape));
    // -1 sentinel, not 0, on both sides.
    REQUIRE(static_cast<double>(cl.regret) == static_cast<double>(up.regret));
    if (!ex) {
      REQUIRE(static_cast<double>(up.regret) == -1.0);
      REQUIRE(static_cast<double>(cl.regret) == -1.0);
    }
  }
}

TEST_CASE("Chunk record accessors behave as upstream's do",
          "[compressor][neuropress][diag][parity][693]") {
  gpucompress_reset_chunk_history();
  clio::cte::compressor::NeuroPressResetChunkHistory();

  // Empty: a get is -1, not a zero-filled row mistakable for data.
  REQUIRE(gpucompress_get_chunk_history_count() == 0);
  REQUIRE(clio::cte::compressor::NeuroPressChunkHistoryCount() == 0);

  gpucompress_chunk_diag_t up{};
  NeuroPressChunkDiag cl;
  REQUIRE(gpucompress_get_chunk_diag(0, &up) == -1);
  REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(0, &cl) == -1);
  REQUIRE(gpucompress_get_chunk_diag(-1, &up) == -1);
  REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(-1, &cl) == -1);
  REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(0, nullptr) == -1);

  // Three rows; indices past the end still refuse.
  for (int i = 0; i < 3; ++i) {
    gpucompress::recordChunkDiagnostic(MakeUpstreamInput(false));
    clio::cte::compressor::NeuroPressRecordChunkDiag(MakeClioRow(false));
  }
  REQUIRE(gpucompress_get_chunk_history_count() == 3);
  REQUIRE(clio::cte::compressor::NeuroPressChunkHistoryCount() == 3);
  REQUIRE(gpucompress_get_chunk_diag(2, &up) == 0);
  REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(2, &cl) == 0);
  REQUIRE(gpucompress_get_chunk_diag(3, &up) == -1);
  REQUIRE(clio::cte::compressor::NeuroPressGetChunkDiag(3, &cl) == -1);

  // Reset clears on both.
  gpucompress_reset_chunk_history();
  clio::cte::compressor::NeuroPressResetChunkHistory();
  REQUIRE(gpucompress_get_chunk_history_count() == 0);
  REQUIRE(clio::cte::compressor::NeuroPressChunkHistoryCount() == 0);
}

SIMPLE_TEST_MAIN()
