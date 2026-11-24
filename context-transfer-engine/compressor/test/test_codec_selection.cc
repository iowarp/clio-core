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
 * Codec selection consistency (issue #1189): the (library, preset) the
 * estimator scores must be the (library, preset) the executor runs, and a
 * compose-deployed compressor must be able to name its models.
 */

#include <string>
#include <utility>
#include <vector>

#include "../../../context-runtime/test/simple_test.h"
#include "clio_cte/compressor/compressor_runtime.h"

using namespace clio::cte::compressor;

namespace {

/** Every candidate the dynamic estimator scores (wire lib id, preset). */
const std::vector<std::pair<int, int>> kCandidates = {
    {10, kPresetBalanced},  // zstd
    {10, kPresetFast},      // zstd
    {4, kPresetFast},       // lz4
    {1, kPresetBest},       // bzip2
    {9, kPresetBalanced},   // zlib
};

}  // namespace

TEST_CASE("Codec selection: one preset encoding", "[compressor][1189]") {
  REQUIRE(PresetFromWire(kPresetFast) == ctp::CompressionPreset::FAST);
  REQUIRE(PresetFromWire(kPresetBalanced) == ctp::CompressionPreset::BALANCED);
  REQUIRE(PresetFromWire(kPresetBest) == ctp::CompressionPreset::BEST);
  // The Context default (2) is balanced, as the header documents.
  clio::cte::core::Context ctx;
  REQUIRE(PresetFromWire(ctx.compress_preset_) ==
          ctp::CompressionPreset::BALANCED);
}

TEST_CASE("Codec selection: features describe what executes",
          "[compressor][1189]") {
  clio::cte::core::Context ctx;
  for (const auto &[lib, preset] : kCandidates) {
    const CompressionFeatures f =
        MakeCodecFeatures(lib, preset, 1 << 20, 4.0, 1.0, 0.5, ctx);
    INFO("lib=" << lib << " preset=" << preset);
    // The preset one-hot names the preset the executor will run.
    const ctp::CompressionPreset runs = PresetFromWire(preset);
    REQUIRE(f.config_fast == (runs == ctp::CompressionPreset::FAST ? 1 : 0));
    REQUIRE(f.config_balanced ==
            (runs == ctp::CompressionPreset::BALANCED ? 1 : 0));
    REQUIRE(f.config_best == (runs == ctp::CompressionPreset::BEST ? 1 : 0));
    // The model id is GetLibraryId's scheme, and decodes back to the
    // library and preset the executor resolves from the wire values.
    const std::string name = ctp::CompressionFactory::NameForWireId(lib);
    REQUIRE(static_cast<int>(f.library_config_id) ==
            ctp::CompressionFactory::GetLibraryId(name, runs));
    auto [decoded_name, decoded_preset] =
        ctp::CompressionFactory::GetLibraryInfo(
            static_cast<int>(f.library_config_id));
    REQUIRE(decoded_name == name);
    REQUIRE(decoded_preset == runs);
  }
}

TEST_CASE("Codec selection: no decompress time unless predicted",
          "[compressor][1189]") {
  CompressionPrediction pred;
  pred.compression_time_ms = 7.0;
  REQUIRE(pred.decompression_time_ms == 0.0);
}

TEST_CASE("Codec selection: model paths come from compose YAML",
          "[compressor][1189]") {
  clio::run::PoolConfig pc;
  pc.config_ =
      "next_pool_id: \"512.0\"\n"
      "qtable_model_path: /models/qtable.bin\n"
      "linreg_model_path: /models/linreg.json\n"
      "distribution_model_path: /models/dist.json\n"
      "dnn_model_weights_path: /models/dnn.bin\n"
      "trace_folder_path: /traces\n";
  CompressorConfig cfg;
  cfg.LoadConfig(pc);
  REQUIRE(cfg.next_pool_id_ == clio::run::PoolId(512, 0));
  REQUIRE(cfg.qtable_model_path_ == "/models/qtable.bin");
  REQUIRE(cfg.linreg_model_path_ == "/models/linreg.json");
  REQUIRE(cfg.distribution_model_path_ == "/models/dist.json");
  REQUIRE(cfg.dnn_model_weights_path_ == "/models/dnn.bin");
  REQUIRE(cfg.trace_folder_path_ == "/traces");
}

SIMPLE_TEST_MAIN()
