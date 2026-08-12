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
      /*second_derivative_mean=*/0.01, /*data_type_float=*/true,
      /*include_gpu=*/true);

  REQUIRE_FALSE(stats.empty());

  // Every returned wire id must resolve to a real, known library -- not
  // silently fall back to the "unknown id" default (zstd), which would mean
  // the base_id -> wire_id conversion is broken.
  for (const auto &s : stats) {
    std::string name = ctp::CompressionFactory::NameForWireId(s.compress_lib_);
    REQUIRE(name != "");
    REQUIRE(std::isfinite(s.compression_ratio_));
    REQUIRE(s.compression_ratio_ > 0);
  }

  // Best-first: sorted by predicted ratio (default RankingWeights is
  // ratio-only, so score == ratio).
  for (size_t i = 1; i < stats.size(); ++i) {
    REQUIRE(stats[i - 1].compression_ratio_ >= stats[i].compression_ratio_);
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
      /*second_derivative_mean=*/0.01, /*data_type_float=*/true, true);
  auto noisy = NeuroPressCandidateStats(
      nn, /*chunk_size=*/4 * 1024 * 1024, /*entropy=*/7.8, /*mad=*/0.9,
      /*second_derivative_mean=*/0.8, /*data_type_float=*/true, true);

  REQUIRE_FALSE(compressible.empty());
  REQUIRE_FALSE(noisy.empty());

  // The winning candidate's predicted ratio must actually respond to the
  // input statistics, not just return a constant.
  REQUIRE(compressible.front().compression_ratio_ >
          noisy.front().compression_ratio_);
}

TEST_CASE("NeuroPressCandidateStats respects include_gpu=false",
          "[compressor][neuropress][dynamic][693]") {
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  auto stats = NeuroPressCandidateStats(
      nn, /*chunk_size=*/4 * 1024 * 1024, /*entropy=*/3.0, /*mad=*/0.3,
      /*second_derivative_mean=*/0.03, /*data_type_float=*/true,
      /*include_gpu=*/false);

  REQUIRE_FALSE(stats.empty());
  for (const auto &s : stats) {
    REQUIRE_FALSE(s.compress_lib_ >= 11 && s.compress_lib_ <= 24);
  }
}

TEST_CASE("NeuroPressCandidateStats respects include_cpu=false, matching "
          "the original NeuroPress project's GPU-only action space",
          "[compressor][neuropress][dynamic][693]") {
  // A device-resident chunk (compressor_runtime.cc's EstCompressionStats
  // passes include_cpu=false for one) must never rank a CPU library --
  // Compress() would otherwise have to read the device pointer directly
  // on the host to try it, or stage a needless host copy to avoid that.
  NeuroPressNNPredictor nn;
  REQUIRE(nn.Load(CLIO_CTP_NEUROPRESS_WEIGHTS_DIR));
  REQUIRE(nn.IsReady());

  auto stats = NeuroPressCandidateStats(
      nn, /*chunk_size=*/4 * 1024 * 1024, /*entropy=*/3.0, /*mad=*/0.3,
      /*second_derivative_mean=*/0.03, /*data_type_float=*/true,
      /*include_gpu=*/true, /*include_cpu=*/false);

  REQUIRE_FALSE(stats.empty());
  for (const auto &s : stats) {
    // wire ids 11-24 are the GPU registry range (nvcomp/cusz/cuszp/ndzip);
    // every result must fall in it -- none of the CPU wire ids 0-10.
    INFO("unexpected non-GPU wire id in GPU-only ranking: " << s.compress_lib_);
    REQUIRE(s.compress_lib_ >= 11);
    REQUIRE(s.compress_lib_ <= 24);
  }
}

SIMPLE_TEST_MAIN()
