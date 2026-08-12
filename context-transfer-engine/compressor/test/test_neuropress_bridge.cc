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

SIMPLE_TEST_MAIN()
