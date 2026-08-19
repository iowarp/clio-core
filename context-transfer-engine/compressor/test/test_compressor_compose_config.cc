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
 * @file test_compressor_compose_config.cc
 * @brief CompressorConfig::LoadConfig against real compose YAML (issue #693).
 *
 * config_manager.cc stores each compose entry verbatim and leaves the parsing
 * to the module ("Store entire YAML node as config string for module-specific
 * parsing"), and CompressorConfig::LoadConfig is where that happens for this
 * chimod. Nothing covered it, and it had drifted: seven of the fifteen fields
 * the struct serializes were never read out of YAML at all. Five of those --
 * the predictor and trace paths -- are written into compress_compose.yaml by
 * jarvis_clio_core.clio_compress, so every model path configured through a
 * Jarvis pipeline was silently discarded. The seventh, neuropress_model_path,
 * is the switch that decides whether the NeuroPress model loads at all, which
 * made the six NeuroPress tunables that DID parse inert in any deployment.
 *
 * The regression test that matters here is not "does key X parse" one key at
 * a time -- it is that the set of YAML-reachable fields stays equal to the set
 * of serialized fields. A field added to the struct and to serialize() but not
 * to LoadConfig is exactly the bug above, and it is invisible: the config
 * loads, the pool starts, and the setting is ignored.
 */

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "clio_cte/compressor/compressor_tasks.h"

namespace {

using clio::cte::compressor::CompressorConfig;

/** Wrap a compose entry the way ConfigManager hands it to the module. */
clio::run::PoolConfig PoolConfigFrom(const std::string &yaml) {
  clio::run::PoolConfig pc;
  pc.config_ = yaml;
  return pc;
}

}  // namespace

TEST_CASE("LoadConfig reads a full compose entry", "[compressor][config][693]") {
  // Every key a deployment can set, with values deliberately different from
  // the struct defaults so a silently-ignored key fails rather than passing
  // by coincidence.
  const std::string yaml = R"(
mod_name: clio_cte_compressor
pool_name: cte_compressor
pool_query: local
pool_id: "512.0"
next_pool_id: "513.0"
tracking_enabled: false
qtable_model_path: /models/qtable.json
linreg_model_path: /models/linreg.json
distribution_model_path: /models/dist.json
dnn_model_weights_path: /models/dnn.json
trace_folder_path: /var/log/clio_compress
neuropress_model_path: /weights/neuropress
neuropress_online_learning_enabled: true
neuropress_mape_threshold: 0.12
neuropress_learning_rate: 0.25
neuropress_exploration_enabled: true
neuropress_exploration_threshold: 0.75
neuropress_exploration_k: 17
neuropress_best_mode: true
)";

  CompressorConfig config;
  config.LoadConfig(PoolConfigFrom(yaml));

  SECTION("interposition target") {
    REQUIRE(config.next_pool_id_ == clio::run::PoolId(513, 0));
  }

  SECTION("predictor and trace paths reach the module") {
    // The five clio_compress writes and nothing read until #693.
    REQUIRE(config.qtable_model_path_ == "/models/qtable.json");
    REQUIRE(config.linreg_model_path_ == "/models/linreg.json");
    REQUIRE(config.distribution_model_path_ == "/models/dist.json");
    REQUIRE(config.dnn_model_weights_path_ == "/models/dnn.json");
    REQUIRE(config.trace_folder_path_ == "/var/log/clio_compress");
  }

  SECTION("NeuroPress model path -- the master switch") {
    REQUIRE(config.neuropress_model_path_ == "/weights/neuropress");
  }

  SECTION("NeuroPress tunables") {
    REQUIRE(config.neuropress_online_learning_enabled_);
    REQUIRE(config.neuropress_mape_threshold_ == 0.12f);
    REQUIRE(config.neuropress_learning_rate_ == 0.25f);
    REQUIRE(config.neuropress_exploration_enabled_);
    REQUIRE(config.neuropress_exploration_threshold_ == 0.75f);
    REQUIRE(config.neuropress_exploration_k_ == 17);
    REQUIRE(config.neuropress_best_mode_);
  }

  SECTION("tracking can be turned off") {
    REQUIRE_FALSE(config.tracking_enabled_);
  }
}

TEST_CASE("LoadConfig leaves unset keys at their defaults",
          "[compressor][config][693]") {
  // A minimal entry must not disturb anything it does not mention: the
  // NeuroPress block stays off, so composing a compressor without a model is
  // still inference-free rather than accidentally learning.
  const std::string yaml = R"(
mod_name: clio_cte_compressor
pool_name: cte_compressor
pool_id: "512.0"
next_pool_id: "513.0"
)";

  CompressorConfig config;
  const CompressorConfig defaults;
  config.LoadConfig(PoolConfigFrom(yaml));

  REQUIRE(config.next_pool_id_ == clio::run::PoolId(513, 0));
  REQUIRE(config.neuropress_model_path_.empty());
  REQUIRE(config.qtable_model_path_.empty());
  REQUIRE(config.trace_folder_path_.empty());
  REQUIRE(config.neuropress_online_learning_enabled_ ==
          defaults.neuropress_online_learning_enabled_);
  REQUIRE(config.neuropress_exploration_enabled_ ==
          defaults.neuropress_exploration_enabled_);
  REQUIRE(config.neuropress_best_mode_ == defaults.neuropress_best_mode_);
  REQUIRE(config.neuropress_exploration_k_ ==
          defaults.neuropress_exploration_k_);
  REQUIRE(config.tracking_enabled_ == defaults.tracking_enabled_);
}

TEST_CASE("LoadConfig survives a malformed entry",
          "[compressor][config][693]") {
  // Parsing is best-effort by design -- a bad entry must not take the pool
  // down. This pins that it also does not half-apply: the config is left at
  // defaults rather than in a partly-parsed state.
  CompressorConfig config;
  const CompressorConfig defaults;
  REQUIRE_NOTHROW(config.LoadConfig(PoolConfigFrom("{{ not: yaml")));
  REQUIRE(config.neuropress_model_path_.empty());
  REQUIRE(config.neuropress_exploration_k_ ==
          defaults.neuropress_exploration_k_);

  // An empty config string is the normal case for a programmatically created
  // pool and must be a no-op, not an error.
  CompressorConfig empty_cfg;
  REQUIRE_NOTHROW(empty_cfg.LoadConfig(PoolConfigFrom("")));
  REQUIRE(empty_cfg.neuropress_model_path_.empty());
}

TEST_CASE("Every serialized field is reachable from compose YAML",
          "[compressor][config][693]") {
  // The guard against this bug recurring. Round-trip a config whose every
  // field is non-default THROUGH YAML and require it to come back intact.
  // A new field wired into serialize() but not LoadConfig fails here, because
  // its value will not survive the trip -- which is precisely the failure
  // mode that let five model paths go unread for as long as they did.
  const std::string yaml = R"(
next_pool_id: "7.3"
tracking_enabled: false
qtable_model_path: q
linreg_model_path: l
distribution_model_path: d
dnn_model_weights_path: n
trace_folder_path: t
neuropress_model_path: w
neuropress_online_learning_enabled: true
neuropress_mape_threshold: 0.5
neuropress_learning_rate: 0.5
neuropress_exploration_enabled: true
neuropress_exploration_threshold: 0.25
neuropress_exploration_k: 31
neuropress_best_mode: true
)";

  CompressorConfig loaded;
  loaded.LoadConfig(PoolConfigFrom(yaml));

  const CompressorConfig defaults;
  // Nothing may still hold its default: that would mean the key did not
  // reach the field.
  REQUIRE(loaded.next_pool_id_ != defaults.next_pool_id_);
  REQUIRE(loaded.tracking_enabled_ != defaults.tracking_enabled_);
  REQUIRE(loaded.qtable_model_path_ != defaults.qtable_model_path_);
  REQUIRE(loaded.linreg_model_path_ != defaults.linreg_model_path_);
  REQUIRE(loaded.distribution_model_path_ !=
          defaults.distribution_model_path_);
  REQUIRE(loaded.dnn_model_weights_path_ != defaults.dnn_model_weights_path_);
  REQUIRE(loaded.trace_folder_path_ != defaults.trace_folder_path_);
  REQUIRE(loaded.neuropress_model_path_ != defaults.neuropress_model_path_);
  REQUIRE(loaded.neuropress_online_learning_enabled_ !=
          defaults.neuropress_online_learning_enabled_);
  REQUIRE(loaded.neuropress_mape_threshold_ !=
          defaults.neuropress_mape_threshold_);
  REQUIRE(loaded.neuropress_learning_rate_ !=
          defaults.neuropress_learning_rate_);
  REQUIRE(loaded.neuropress_exploration_enabled_ !=
          defaults.neuropress_exploration_enabled_);
  REQUIRE(loaded.neuropress_exploration_threshold_ !=
          defaults.neuropress_exploration_threshold_);
  REQUIRE(loaded.neuropress_exploration_k_ !=
          defaults.neuropress_exploration_k_);
  REQUIRE(loaded.neuropress_best_mode_ != defaults.neuropress_best_mode_);
}
