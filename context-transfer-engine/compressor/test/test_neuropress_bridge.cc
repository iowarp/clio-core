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
 * @file test_neuropress_bridge.cc
 * @brief Cycle 3 (issue #693): NeuroPressCandidateStats bridges the ported
 * clio_ctp::compress::model predictors into the compressor chimod's own
 * CompressionStats, so dynamic selection can source GPU/NeuroPress
 * candidates instead of only the fixed CPU-only list.
 */
#include <cmath>
#include <set>
#include <string>

#include "clio_cte/compressor/models/neuropress_bridge.h"
#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "../../../context-runtime/test/simple_test.h"

using namespace clio::cte::compressor;      // NOLINT(build/namespaces)
using ctp::compress::model::NeuroPressNNPredictor;

#ifndef CLIO_CTP_NEUROPRESS_WEIGHTS_DIR
#error "CLIO_CTP_NEUROPRESS_WEIGHTS_DIR must be set by CMake"
#endif

TEST_CASE("NeuroPressCandidateStats ranks GPU candidates for compressible data",
          "[compressor][neuropress][dynamic][693]") {
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  // Highly compressible: near-zero entropy/MAD/curvature.
  auto stats = NeuroPressCandidateStats(
      nn, /*chunk_size=*/4 * 1024 * 1024, /*entropy=*/0.5, /*mad=*/0.02,
      /*second_derivative_mean=*/0.01, /*data_type_float=*/true);

  REQUIRE_FALSE(stats.empty());

  // Every returned wire id must resolve to a real, known library -- not
  // silently fall back to the "unknown id" default (zstd), which would mean
  // the base_id -> wire_id conversion is broken.
  for (const auto &s : stats) {
    std::string name = ctp::CompressionFactory::NameForWireId(s.compress_lib_);
    REQUIRE(name != "");
    REQUIRE(std::isfinite(s.compression_ratio_));
    REQUIRE(s.compression_ratio_ > 0);
    // Decompression time is the NN's own output (index 1), not a copy of
    // the compression time (index 0). It used to be aliased, which silently
    // zeroed out the w1 term of the cost model above.
    REQUIRE(std::isfinite(s.decompress_time_ms_));
    REQUIRE(s.decompress_time_ms_ > 0);
  }

  // At least one candidate must show the two times actually differing --
  // if every one matched, the aliasing bug would be back and the cost
  // model would be ranking on a duplicated term.
  bool saw_distinct_decomp_time = false;
  for (const auto &s : stats) {
    if (std::fabs(s.decompress_time_ms_ - s.compress_time_ms_) > 1e-6) {
      saw_distinct_decomp_time = true;
      break;
    }
  }
  REQUIRE(saw_distinct_decomp_time);

  // Best-first under NeuroPress's OWN cost model, not by ratio: the bridge
  // opts into RankingWeights::use_cost_model, so the order minimizes
  // cost = w0*ct + w1*dt + w2*size/(ratio*bw) (nn_gpu.cu). Asserting
  // descending ratio here would be asserting the pre-Cycle-3 ratio-only
  // policy, which deliberately no longer holds -- a codec that compresses
  // marginally better but runs far longer now correctly ranks lower.
  constexpr double kW0 = 1.0, kW1 = 1.0, kW2 = 1.0;
  constexpr double kBw = 5e6;  // bytes/ms, mirrors g_measured_bw_bytes_per_ms
  constexpr double kChunkSize = 4.0 * 1024 * 1024;
  auto cost_of = [&](const CompressionStats &s) {
    double ct = std::max(1.0, s.compress_time_ms_);
    double dt = (s.decompress_time_ms_ > 0.0)
                    ? std::max(1.0, s.decompress_time_ms_)
                    : ct;
    double ratio = std::min(100.0, s.compression_ratio_);
    return kW0 * ct + kW1 * dt +
           ((ratio > 0.0) ? kW2 * kChunkSize / (ratio * kBw) : 1e30);
  };
  for (size_t i = 1; i < stats.size(); ++i) {
    INFO("cost order violated at index " << i);
    REQUIRE(cost_of(stats[i - 1]) <= cost_of(stats[i]) + 1e-9);
  }

  // The GPU action space (nvcomp/cusz/ndzip, wire ids 11-24) must actually
  // be reachable through this bridge, not just through DefaultCandidates()
  // directly -- that's the whole point of Cycle 3.
  bool saw_gpu_candidate = false;
  for (const auto &s : stats) {
    if (s.compress_lib_ >= 11 && s.compress_lib_ <= 24) {
      saw_gpu_candidate = true;
      break;
    }
  }
  REQUIRE(saw_gpu_candidate);
}

TEST_CASE("NeuroPressCandidateStats is entropy-sensitive",
          "[compressor][neuropress][dynamic][693]") {
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  auto compressible = NeuroPressCandidateStats(
      nn, /*chunk_size=*/4 * 1024 * 1024, /*entropy=*/0.5, /*mad=*/0.02,
      /*second_derivative_mean=*/0.01, /*data_type_float=*/true);
  auto noisy = NeuroPressCandidateStats(
      nn, /*chunk_size=*/4 * 1024 * 1024, /*entropy=*/7.8, /*mad=*/0.9,
      /*second_derivative_mean=*/0.8, /*data_type_float=*/true);

  REQUIRE_FALSE(compressible.empty());
  REQUIRE_FALSE(noisy.empty());

  // The winning candidate's predicted ratio must actually respond to the
  // input statistics, not just return a constant.
  REQUIRE(compressible.front().compression_ratio_ >
          noisy.front().compression_ratio_);
}

TEST_CASE("NeuroPressCandidateStats never ranks outside its trained "
          "8-algorithm nvcomp action space",
          "[compressor][neuropress][dynamic][693]") {
  // No CPU library, and none of zfp-sycl/cuSZ/nDzip/cuSZp either -- none of
  // those were ever part of NeuroPress's trained action space (see
  // neuropress_bridge.cc), so a "prediction" for one would just be some
  // other algorithm's real prediction under an alias. Unconditional now,
  // not caller-configurable: this must hold regardless of whether the
  // source buffer happens to be device-resident.
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  auto stats = NeuroPressCandidateStats(
      nn, /*chunk_size=*/4 * 1024 * 1024, /*entropy=*/3.0, /*mad=*/0.3,
      /*second_derivative_mean=*/0.03, /*data_type_float=*/true);

  REQUIRE_FALSE(stats.empty());
  static const std::set<std::string> kTrainedNames = {
      "nvcomp-lz4",   "nvcomp-snappy",   "nvcomp-zstd", "nvcomp-gdeflate",
      "nvcomp-deflate", "nvcomp-ans",    "nvcomp-cascaded",
      "nvcomp-bitcomp"};
  for (const auto &s : stats) {
    std::string name = ctp::CompressionFactory::NameForWireId(s.compress_lib_);
    INFO("unexpected library outside NeuroPress's trained action space: "
        << name);
    REQUIRE(kTrainedNames.count(name) == 1);
  }
}

namespace {

/** NeuroPress action index (0-7) -> the library name the bridge reports. */
const char *NameForAlgoIdx(int algo_idx) {
  switch (algo_idx) {
    case 0: return "nvcomp-lz4";
    case 1: return "nvcomp-snappy";
    case 2: return "nvcomp-deflate";
    case 3: return "nvcomp-gdeflate";
    case 4: return "nvcomp-zstd";
    case 5: return "nvcomp-ans";
    case 6: return "nvcomp-cascaded";
    default: return "nvcomp-bitcomp";
  }
}

/** Recover decodeAction's index (algo + 8*quant + 16*shuffle) from what the
 *  bridge actually returned. Quantize and byte-shuffle ride in the packed
 *  preset word (see PackPreset in compressor_runtime.cc): bits 0-7 preset,
 *  8-15 shuffle element size, bit 24 quantize-enabled. */
int ActionOfStats(const CompressionStats &s) {
  std::string name = ctp::CompressionFactory::NameForWireId(s.compress_lib_);
  int algo = -1;
  for (int i = 0; i < 8; ++i) {
    if (name == NameForAlgoIdx(i)) { algo = i; break; }
  }
  const uint32_t bits = static_cast<uint32_t>(s.compress_preset_);
  const int quant = ((bits >> 24) & 1u) ? 1 : 0;
  const int shuffle = (((bits >> 8) & 0xFFu) != 0u) ? 1 : 0;
  return (algo < 0) ? -1 : algo + 8 * quant + 16 * shuffle;
}

/**
 * Predictor that saturates a chosen set of actions to an identical score and
 * leaves every other candidate strictly worse, so the bridge's ranking is
 * decided purely by the order it enumerates candidates in.
 *
 * Winners return the clamped extremes (1 ms times, 100x ratio), which is how
 * the real model produces ties: the policy clamps collapse a whole group of
 * candidates onto one cost.
 */
class TieStubPredictor : public ctp::compress::model::CompressionPredictor {
 public:
  explicit TieStubPredictor(std::set<int> winners)
      : winners_(std::move(winners)) {}

  bool Load(const std::string &) override { return true; }
  bool Save(const std::string &) override { return true; }
  bool IsReady() const override { return true; }
  ctp::compress::model::ModelType Type() const override {
    return ctp::compress::model::ModelType::kQTable;
  }

  ctp::compress::model::CompressionPrediction Predict(
      const ctp::compress::model::CompressionFeatures &f) override {
    const int base_id = static_cast<int>(f.library_config_id) / 10;
    int algo = -1;
    static const int kBaseIds[8] = {13, 14, 17, 16, 15, 18, 23, 24};
    for (int i = 0; i < 8; ++i) {
      if (kBaseIds[i] == base_id) { algo = i; break; }
    }
    const int action = algo + 8 * (f.quantize > 0.5 ? 1 : 0) +
                       16 * (f.byte_shuffle > 0.5 ? 1 : 0);
    ctp::compress::model::CompressionPrediction p;
    if (algo >= 0 && winners_.count(action)) {
      p.compression_ratio = 100.0;    // at the cap
      p.compression_time_ms = 1.0;    // at the floor
      p.decompression_time_ms = 1.0;  // at the floor
    } else {
      p.compression_ratio = 2.0;
      p.compression_time_ms = 50.0;
      p.decompression_time_ms = 50.0;
    }
    p.psnr_db = 0.0;
    return p;
  }

 private:
  std::set<int> winners_;
};

}  // namespace

TEST_CASE("NeuroPressCandidateStats breaks ties by lowest action index",
          "[compressor][neuropress][dynamic][693]") {
  // Ties are routine, not exotic: both sides clamp times to a 1 ms floor and
  // the ratio to a 100x cap before ranking (nn_gpu.cu:227-230), so on
  // compressible data a group of candidates lands on a bit-identical cost.
  // Position then decides the winner. Upstream ranks with a bitonic network
  // built from strict comparators, so it never swaps equal keys and the
  // LOWEST action index wins (decodeAction, internal.hpp:167-172, numbers an
  // action algo + 8*quant + 16*shuffle); Rank() here uses std::stable_sort,
  // so the FIRST ENUMERATED wins. The two agree only while this bridge
  // enumerates shuffle-outermost, quant, then algo.
  //
  // That ordering was previously argued from decodeAction rather than
  // measured, and it is not cheap to measure by accident: the real model's
  // ties all fall within a single algorithm's column, where both the correct
  // and an algo-outermost enumeration pick the same winner
  // (neuropress_tiebreak_parity.cu finds 32 real ties and not one of them
  // distinguishes the two). Only a tie whose lowest ACTION sits at a high
  // ALGORITHM index can, so the stub below manufactures exactly those.
  struct Case {
    std::set<int> tied;
    double error_bound;
    int expected;  // lowest action index in the set
    bool discriminating;
  };
  const std::vector<Case> cases = {
      // Lossless: only the 16 unquantized configs exist (0-7, 16-23).
      {{7, 16}, 0.0, 7, true},       // algo7 plain vs algo0 shuffled
      {{6, 16, 22}, 0.0, 6, true},
      {{3, 19}, 0.0, 3, false},      // control: same algo either way
      // With a bound, all 32 are reachable.
      {{1, 8}, 1e-3, 1, true},       // algo1 plain vs algo0 quantized
      {{15, 16}, 1e-3, 15, true},    // algo7 quantized vs algo0 shuffled
      {{7, 24}, 1e-3, 7, true},
      {{5, 13, 21, 29}, 1e-3, 5, false},  // control: one algo's whole column
  };

  int discriminating_cases = 0;
  for (const auto &c : cases) {
    TieStubPredictor stub(c.tied);
    auto stats = NeuroPressCandidateStats(
        stub, /*chunk_size=*/4 * 1024 * 1024, /*entropy=*/1.0, /*mad=*/0.1,
        /*second_derivative_mean=*/0.05, /*data_type_float=*/true,
        c.error_bound);
    REQUIRE_FALSE(stats.empty());
    if (c.discriminating) ++discriminating_cases;
    INFO("tie set winner mismatch at error_bound " << c.error_bound);
    REQUIRE(ActionOfStats(stats.front()) == c.expected);
  }

  // The cases marked discriminating are the ones that would resolve
  // differently under an algo-outermost enumeration. Without at least one,
  // this test would pass against either order and assert nothing.
  REQUIRE(discriminating_cases > 0);
}

SIMPLE_TEST_MAIN()
