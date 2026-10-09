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

#ifndef CLIO_CTE_DTSCHEDULE_DTSCHEDULE_TASKS_H_
#define CLIO_CTE_DTSCHEDULE_DTSCHEDULE_TASKS_H_

#include <algorithm>
#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/msan.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/dtschedule/autogen/dtschedule_methods.h>

#include <limits>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <yaml-cpp/yaml.h>

namespace clio::cte::dtschedule {

/**
 * The dtschedule chimod speaks the CTE core's vocabulary for data verbs.
 */
using Context = clio::cte::core::Context;
using TagId = clio::cte::core::TagId;

/**
 * Well-known default pool id/name for dtschedule module.
 */
static constexpr clio::run::PoolId kDtschedulePoolId(565, 0);
static constexpr const char *kDtschedulePoolName = "clio_cte_dtschedule";

/**
 * Tier score mapping from device name to numeric score.
 * Used to match blob_score to storage tiers (higher = faster).
 * Example: ram: 1.0, nvme: 0.7, ssd: 0.4, nfs: 0.1
 */
struct TierScore {
  std::string name_;   ///< Tier name (e.g., "ram", "nvme", "ssd")
  float score_;        ///< Score value (1.0 = fastest, 0.0 = slowest)

  TierScore(const std::string &name, float score)
      : name_(name), score_(score) {}

  bool operator<(const TierScore &other) const {
    return score_ > other.score_;  // Sort descending by score
  }
};

/**
 * Decision stats: counters for puts, compression decisions, bytes in/out.
 * Used for testing and per-node trace flushing.
 */
struct DecisionStats {
  uint64_t puts_ = 0;                 ///< Total PutBlob operations
  uint64_t compressed_ = 0;           ///< Blobs that were compressed
  uint64_t passthrough_ = 0;          ///< Blobs passed through (replica or already compressed)
  uint64_t skipped_small_ = 0;        ///< Blobs skipped (size < min)
  uint64_t bytes_in_ = 0;             ///< Total bytes written
  uint64_t bytes_out_ = 0;            ///< Total bytes after compression
  std::map<std::string, uint64_t> per_lib_count_;     ///< Count per codec lib
  std::map<std::string, uint64_t> per_scenario_count_;  ///< Count per scenario
  uint64_t consumer_tags_ = 0;        ///< Phase 4: Count of registered consumer tags
  uint64_t load_samples_ = 0;         ///< Phase 3: Count of load samples taken
  uint64_t copy_skipped_local_ = 0;   ///< Phase 4b: S2 copies skipped (C==P or C==O)
  uint64_t copy_refused_ = 0;         ///< Phase 4b: S2 copy rejected by core (rc != 0)
  uint64_t copies_pushed_ = 0;        ///< Phase 4b: S2 copies successfully pushed
  uint64_t s3_local_ = 0;             ///< Phase 4c: S3 local (C==P) compression ops
  uint64_t s3_local_copies_ = 0;      ///< Phase 4c: S3 local cache copies written at C
  uint32_t dag_files_ = 0;            ///< Phase 5: files in the loaded DAG spec
  uint32_t dag_nodes_ = 0;            ///< Phase 5: distinct nodes in the DAG spec
  uint64_t s3_sent_ = 0;              ///< Phase 4c: S3 remote CompressAt ops sent to C

  /**
   * Compute compression ratio (bytes_out / bytes_in), or 1.0 if no data.
   */
  /** Stored-bytes fraction (bytes_out / bytes_in); NOT the codec ratio,
   *  which is original/compressed everywhere else in the module. */
  double GetRatio() const {
    return bytes_in_ > 0 ? static_cast<double>(bytes_out_) / bytes_in_ : 1.0;
  }

  /**
   * Add another stats object to this one (used for aggregation).
   */
  void Add(const DecisionStats &other) {
    puts_ += other.puts_;
    compressed_ += other.compressed_;
    passthrough_ += other.passthrough_;
    skipped_small_ += other.skipped_small_;
    bytes_in_ += other.bytes_in_;
    bytes_out_ += other.bytes_out_;
    consumer_tags_ += other.consumer_tags_;
    load_samples_ += other.load_samples_;
    copy_skipped_local_ += other.copy_skipped_local_;
    copy_refused_ += other.copy_refused_;
    copies_pushed_ += other.copies_pushed_;
    s3_local_ += other.s3_local_;
    s3_local_copies_ += other.s3_local_copies_;
    dag_files_ += other.dag_files_;
    dag_nodes_ += other.dag_nodes_;
    s3_sent_ += other.s3_sent_;
    for (const auto &[lib, count] : other.per_lib_count_) {
      per_lib_count_[lib] += count;
    }
    for (const auto &[scenario, count] : other.per_scenario_count_) {
      per_scenario_count_[scenario] += count;
    }
  }
};

/**
 * Per-stage QoS overrides.
 *
 * A stage is a named blob filter (regex) that can override the global QoS
 * settings (max_error, objective, compression_preference) when the blob name
 * or tag name matches. Stages are processed in order; the first match applies.
 */
struct QosStage {
  std::regex match_;                            ///< Regex to match blob name or tag name
  double max_error_ = -1.0;                     ///< -1 = use global; else override
  std::string objective_;                       ///< "" = use global; else override
  std::vector<std::string> compression_preference_;  ///< Empty = use global

  QosStage() = default;

  /**
   * Parse stage from YAML: {match: '...', max_error: N, objective: '...', ...}
   */
  static QosStage FromYAML(const YAML::Node &node) {
    QosStage stage;
    try {
      if (node["match"]) {
        stage.match_ = std::regex(node["match"].as<std::string>());
      }
      if (node["max_error"]) {
        stage.max_error_ = node["max_error"].as<double>();
      }
      if (node["objective"]) {
        stage.objective_ = node["objective"].as<std::string>();
      }
      if (node["compression_preference"] && node["compression_preference"].IsSequence()) {
        for (const auto &item : node["compression_preference"]) {
          stage.compression_preference_.push_back(item.as<std::string>());
        }
      }
    } catch (const std::exception &e) {
      HLOG(kWarning, "dtschedule: Failed to parse QoS stage: {}", e.what());
    }
    return stage;
  }
};

/**
 * Knobs for ablation studies: can be changed at runtime via SetKnobs.
 */
struct Knobs {
  std::string ccm_ = "fixed:zstd:balanced";     ///< CCM strategy
  bool load_aware_ = true;                      ///< Enable load-aware offload
  bool workflow_aware_ = false;                 ///< Enable workflow awareness
  std::string force_scenario_ = "auto";         ///< Force specific scenario (auto|1|2|3)
  std::string decision_order_ = "joint";        ///< Decision order (joint|codec_first|tier_first)
  double ratio_noise_sigma_ = 0.0;              ///< Multiplicative noise on ratio

  /**
   * Serialize knobs to a hash for trace logging (used to detect knob changes).
   */
  uint64_t Hash() const {
    // Simple hash: concatenate string representations and hash
    std::string key = ccm_ + "|" + std::to_string(load_aware_) + "|" +
                      std::to_string(workflow_aware_) + "|" + force_scenario_ +
                      "|" + decision_order_ + "|" +
                      std::to_string(ratio_noise_sigma_);
    return std::hash<std::string>()(key);
  }
};

/**
 * Configuration for dtschedule, parsed from compose YAML.
 * Corresponds to DESIGN.md §7.
 */
struct DtscheduleConfig {
  static constexpr const char *chimod_lib_name = "clio_cte_dtschedule";

  clio::run::PoolId next_pool_id_;              ///< CTE core pool id (e.g. 512.0)
  clio::run::PoolId compressed_next_pool_id_;   ///< Pool for compressed blobs (default = next_pool_id)

  // QoS configuration (per DESIGN §7)
  std::string objective_ = "performance";       ///< performance | ratio
  double max_error_ = 1e-3;                     ///< Relative error bound (0 = lossless)
  std::vector<std::string> lossy_allowlist_;    ///< Regex patterns for lossy candidates
  std::vector<std::string> compression_preference_;  ///< Empty = all candidates
  double resample_error_ = 0.4;                 ///< Resample threshold
  double resample_chance_ = 1.0;                ///< Resample probability
  std::vector<QosStage> stages_;                ///< Per-stage QoS overrides (task 2)

  // CCM configuration
  std::string ccm_ = "fixed:zstd:balanced";     ///< CCM strategy (phase 1 only)
  std::string qtable_model_path_ = "";          ///< Path to Q-table model (unused phase 1)
  double ratio_noise_sigma_ = 0.0;              ///< Noise on predicted ratio

  // Load awareness (phases 3+)
  bool load_aware_ = true;
  int load_period_ms_ = 1000;
  double load_cap_ = 4.0;
  // Application hint: the writers' ranks wait for their own writes (blocking
  // I/O phases), so codec work on the writer's node does not delay its job.
  bool writers_block_ = false;

  // Workflow awareness (phases 4-5)
  std::string workflow_aware_ = "none";         ///< none | consumer | dag
  std::string dag_path_ = "";                   ///< Path to DAG JSON (unused phase 1)

  // DAG configuration (phase 5)
  struct DagConfig {
    bool colocate_fanin_ = true;            ///< All outputs of a task to same consumer
    uint32_t replicate_fanout_min_ = 4;     ///< Replicate when consumer count >= this
    uint32_t replicate_max_ = 8;            ///< Maximum copies to push (excluding owner)
  } dag_;                                    ///< DAG-specific configuration

  std::vector<uint32_t> load_peers_;            ///< Extra peers to poll for load (for tests)

  // Background compress-on-demote (consumer side): chunks a node has just
  // read are queued; while the owner's fastest tier is short of room they
  // are compressed on this node's idle CPU and rewritten to a lower tier,
  // freeing RAM so new writes keep landing in it.
  // Data placement authority for chunks >= min_compress_bytes:
  //   owner      -- the core's hash owner (bytes placed by the owner's DPE)
  //   dtschedule -- dtschedule picks the store node by cost (tier + network
  //                 across nodes; the consumer for scenarios 2/3) and records
  //                 the location when it differs from the hash owner.
  //   local      -- same routing, but always the writer's node (baseline).
  std::string placement_ = "owner";
  double demote_watermark_ = 0.0;   ///< Demote while fastest-tier free <
                                    ///< this fraction of its capacity (0 = off)
  int demote_queue_mb_ = 512;       ///< Max bytes of read chunks kept queued
  int demote_budget_mb_ = 64;       ///< Max bytes demoted per load tick

  // Scenario selection (phases 3-4)
  std::string force_scenario_ = "auto";         ///< auto | 1 | 2 | 3
  std::string decision_order_ = "joint";        ///< joint | codec_first | tier_first

  // Tier configuration
  std::vector<TierScore> tiers_;                ///< Sorted by score, descending
  double net_bw_gbps_ = 25.0;                   ///< Network bandwidth
  std::map<std::string, double> tier_bw_mbps_;  ///< Bandwidth (MB/ms) per tier
  clio::run::PoolId core_pool_id_;              ///< Pool id for consumer copy core writes

  // Tracing
  std::string trace_path_ = "";                 ///< CSV trace file base path (empty = off)
  bool trace_candidates_ = false;               ///< Also write <trace_path>.cand.<node>.csv
  int min_compress_bytes_ = 4096;               ///< Minimum size to compress
  // Chunks of one tag (file) that reuse a codec decision before it is made
  // again (features + ranking). 1 = decide every chunk. A decision is also
  // remade once it is older than load_period_ms.
  int decision_reuse_chunks_ = 32;
  // Runtime workers that run codec work concurrently: the cost model divides
  // compress/decompress time by it (the network and devices are shared).
  double cpu_parallelism_ = 4.0;

  DtscheduleConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}

  DtscheduleConfig(const clio::run::PoolId &pool_id, const DtscheduleConfig &other)
      : next_pool_id_(other.next_pool_id_),
        compressed_next_pool_id_(other.compressed_next_pool_id_),
        objective_(other.objective_),
        max_error_(other.max_error_),
        lossy_allowlist_(other.lossy_allowlist_),
        compression_preference_(other.compression_preference_),
        resample_error_(other.resample_error_),
        resample_chance_(other.resample_chance_),
        ccm_(other.ccm_),
        qtable_model_path_(other.qtable_model_path_),
        ratio_noise_sigma_(other.ratio_noise_sigma_),
        load_aware_(other.load_aware_),
        load_period_ms_(other.load_period_ms_),
        load_cap_(other.load_cap_),
        writers_block_(other.writers_block_),
        workflow_aware_(other.workflow_aware_),
        dag_path_(other.dag_path_),
        force_scenario_(other.force_scenario_),
        decision_order_(other.decision_order_),
        tiers_(other.tiers_),
        net_bw_gbps_(other.net_bw_gbps_),
        tier_bw_mbps_(other.tier_bw_mbps_),
        core_pool_id_(other.core_pool_id_),
        trace_path_(other.trace_path_),
        min_compress_bytes_(other.min_compress_bytes_) {
    (void)pool_id;
  }

  /**
   * Load configuration from PoolConfig.
   * Phase 1: just uses defaults; Phase 4+ will parse from pool_config.
   *
   * @param pool_config PoolConfig from compose (unused in Phase 1)
   */
  /**
   * Load configuration from YAML node.
   * Parses nested layout from compose YAML with top-level fallback.
   * Canonical nested layout:
   *   qos:
   *     objective, max_error, lossy_allowlist, compression_preference,
   *     resample_error, resample_chance, stages
   *   dag:
   *     colocate_fanin, replicate_fanout_min, replicate_max
   *   ccm, qtable_model_path, ratio_noise_sigma (top-level)
   *   load_aware, load_period_ms, load_cap (top-level)
   *   workflow_aware, dag_path (top-level)
   *   force_scenario, decision_order (top-level)
   *   tiers, net_bw_gbps, trace_path, min_compress_bytes (top-level)
   *
   * @param node YAML::Node from compose file
   */
  void LoadConfig(const YAML::Node &node) {
    if (!node || !node.IsMap()) {
      return;
    }

    // Parse nested QoS block if present
    if (node["qos"] && node["qos"].IsMap()) {
      const auto &qos = node["qos"];
      if (qos["objective"]) {
        objective_ = qos["objective"].as<std::string>();
      }
      if (qos["max_error"]) {
        max_error_ = qos["max_error"].as<double>();
      }
      if (qos["lossy_allowlist"] && qos["lossy_allowlist"].IsSequence()) {
        for (const auto &item : qos["lossy_allowlist"]) {
          lossy_allowlist_.push_back(item.as<std::string>());
        }
      }
      if (qos["compression_preference"] && qos["compression_preference"].IsSequence()) {
        for (const auto &item : qos["compression_preference"]) {
          compression_preference_.push_back(item.as<std::string>());
        }
      }
      if (qos["resample_error"]) {
        resample_error_ = qos["resample_error"].as<double>();
      }
      if (qos["resample_chance"]) {
        resample_chance_ = qos["resample_chance"].as<double>();
      }
      // Parse per-stage QoS overrides (task 2)
      if (qos["stages"] && qos["stages"].IsSequence()) {
        for (const auto &stage_node : qos["stages"]) {
          stages_.push_back(QosStage::FromYAML(stage_node));
        }
      }
    } else {
      // Fallback: parse QoS at top level
      if (node["objective"]) {
        objective_ = node["objective"].as<std::string>();
      }
      if (node["max_error"]) {
        max_error_ = node["max_error"].as<double>();
      }
      if (node["lossy_allowlist"] && node["lossy_allowlist"].IsSequence()) {
        for (const auto &item : node["lossy_allowlist"]) {
          lossy_allowlist_.push_back(item.as<std::string>());
        }
      }
      if (node["compression_preference"] && node["compression_preference"].IsSequence()) {
        for (const auto &item : node["compression_preference"]) {
          compression_preference_.push_back(item.as<std::string>());
        }
      }
      if (node["resample_error"]) {
        resample_error_ = node["resample_error"].as<double>();
      }
      if (node["resample_chance"]) {
        resample_chance_ = node["resample_chance"].as<double>();
      }
    }

    if (node["next_pool_id"]) {
      // "major.minor" as in the compose file (e.g. "512.0")
      std::string next_str = node["next_pool_id"].as<std::string>();
      size_t dot = next_str.find('.');
      if (dot != std::string::npos) {
        clio::run::u32 major = std::stoul(next_str.substr(0, dot));
        clio::run::u32 minor = std::stoul(next_str.substr(dot + 1));
        next_pool_id_ = clio::run::PoolId(major, minor);
      }
    }

    if (node["compressed_next_pool_id"]) {
      // "major.minor" pool for compressed blobs (default = next_pool_id)
      std::string cmp_str = node["compressed_next_pool_id"].as<std::string>();
      size_t dot = cmp_str.find('.');
      if (dot != std::string::npos) {
        clio::run::u32 major = std::stoul(cmp_str.substr(0, dot));
        clio::run::u32 minor = std::stoul(cmp_str.substr(dot + 1));
        compressed_next_pool_id_ = clio::run::PoolId(major, minor);
      }
    } else if (!next_pool_id_.IsNull()) {
      // Default: compressed blobs go to the same pool as raw blobs
      compressed_next_pool_id_ = next_pool_id_;
    }

    if (node["ccm"]) {
      ccm_ = node["ccm"].as<std::string>();
    }
    if (node["qtable_model_path"]) {
      qtable_model_path_ = node["qtable_model_path"].as<std::string>();
    }
    if (node["ratio_noise_sigma"]) {
      ratio_noise_sigma_ = node["ratio_noise_sigma"].as<double>();
    }

    if (node["load_aware"]) {
      load_aware_ = node["load_aware"].as<bool>();
    }
    if (node["load_period_ms"]) {
      load_period_ms_ = node["load_period_ms"].as<int>();
    }
    if (node["load_cap"]) {
      load_cap_ = node["load_cap"].as<double>();
    }
    if (node["writers_block"]) {
      writers_block_ = node["writers_block"].as<bool>();
    }

    if (node["workflow_aware"]) {
      workflow_aware_ = node["workflow_aware"].as<std::string>();
    }
    if (node["dag_path"]) {
      dag_path_ = node["dag_path"].as<std::string>();
    }

    // Parse DAG-specific configuration (phase 5)
    if (node["dag"] && node["dag"].IsMap()) {
      const auto &dag_node = node["dag"];
      if (dag_node["colocate_fanin"]) {
        dag_.colocate_fanin_ = dag_node["colocate_fanin"].as<bool>();
      }
      if (dag_node["replicate_fanout_min"]) {
        dag_.replicate_fanout_min_ = dag_node["replicate_fanout_min"].as<uint32_t>();
      }
      if (dag_node["replicate_max"]) {
        dag_.replicate_max_ = dag_node["replicate_max"].as<uint32_t>();
      }
    }

    if (node["force_scenario"]) {
      force_scenario_ = node["force_scenario"].as<std::string>();
    }
    if (node["decision_order"]) {
      decision_order_ = node["decision_order"].as<std::string>();
    }

    // Parse tier scores
    if (node["tiers"] && node["tiers"].IsMap()) {
      for (const auto &tier : node["tiers"]) {
        std::string tier_name = tier.first.as<std::string>();
        float score = tier.second.as<float>();
        tiers_.emplace_back(tier_name, score);
      }
      // Sort tiers by score (descending)
      std::sort(tiers_.begin(), tiers_.end());
    }

    if (node["net_bw_gbps"]) {
      net_bw_gbps_ = node["net_bw_gbps"].as<double>();
    }

    // Parse tier bandwidth (MB/ms) - default values if not specified
    if (node["tier_bw_mbps"] && node["tier_bw_mbps"].IsMap()) {
      for (const auto &tier_bw : node["tier_bw_mbps"]) {
        std::string tier_name = tier_bw.first.as<std::string>();
        double bw_mbps = tier_bw.second.as<double>();
        tier_bw_mbps_[tier_name] = bw_mbps;
      }
    }
    // Set defaults if not specified
    if (tier_bw_mbps_.empty()) {
      tier_bw_mbps_["ram"] = 20000.0;   // 20 GB/s
      tier_bw_mbps_["nvme"] = 3000.0;   // 3 GB/s
      tier_bw_mbps_["ssd"] = 500.0;     // 500 MB/s
      tier_bw_mbps_["nfs"] = 1000.0;    // 1 GB/s
    }

    // Parse core pool id for consumer copies (default 512.0)
    if (node["core_pool_id"]) {
      std::string core_str = node["core_pool_id"].as<std::string>();
      size_t dot = core_str.find('.');
      if (dot != std::string::npos) {
        clio::run::u32 major = std::stoul(core_str.substr(0, dot));
        clio::run::u32 minor = std::stoul(core_str.substr(dot + 1));
        core_pool_id_ = clio::run::PoolId(major, minor);
      }
    } else {
      // Default: use 512.0 (CTE core pool)
      core_pool_id_ = clio::run::PoolId(512, 0);
    }

    if (node["trace_path"]) {
      trace_path_ = node["trace_path"].as<std::string>();
    }
    if (node["trace_candidates"]) {
      trace_candidates_ = node["trace_candidates"].as<bool>();
    }
    if (node["min_compress_bytes"]) {
      // Saturate: a huge value is the documented way to disable selection.
      const long long raw = node["min_compress_bytes"].as<long long>();
      min_compress_bytes_ = static_cast<int>(
          std::min<long long>(raw, std::numeric_limits<int>::max()));
    }
    if (node["cpu_parallelism"]) {
      cpu_parallelism_ = std::max(1.0, node["cpu_parallelism"].as<double>());
    }
    if (node["decision_reuse_chunks"]) {
      decision_reuse_chunks_ =
          std::max(1, node["decision_reuse_chunks"].as<int>());
    }
    if (node["load_peers"]) {
      load_peers_ = node["load_peers"].as<std::vector<uint32_t>>();
    }
    if (node["placement"]) {
      placement_ = node["placement"].as<std::string>();
    }
    if (node["demote_watermark"]) {
      demote_watermark_ = node["demote_watermark"].as<double>();
    }
    if (node["demote_queue_mb"]) {
      demote_queue_mb_ = node["demote_queue_mb"].as<int>();
    }
    if (node["demote_budget_mb"]) {
      demote_budget_mb_ = node["demote_budget_mb"].as<int>();
    }
  }

  /**
   * Load configuration from PoolConfig (for compose mode).
   * When created via compose, the config is embedded in the PoolConfig.
   *
   * @param pool_config Pool configuration from compose
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    // Extract the config_ field from pool_config and load it as YAML
    if (!pool_config.config_.empty()) {
      try {
        YAML::Node node = YAML::Load(pool_config.config_);
        LoadConfig(node);
      } catch (const std::exception &e) {
        HLOG(kWarning, "dtschedule: Failed to parse config YAML: {}", e.what());
      }
    }
  }

  /**
   * Apply per-stage QoS overrides based on blob name (task 2).
   *
   * Returns: {stage_index, override_applied}
   * stage_index = -1 if no stage matched
   * override_applied = true if overrides were set on the output parameters
   *
   * @param blob_name Blob name to match against stages
   * @param[out] obj Objective override (empty if no override)
   * @param[out] max_err Max error override (-1 if no override)
   * @param[out] comp_pref Compression preference override (empty if no override)
   * @return pair<stage_index, was_overridden>
   */
  std::pair<int, bool> ApplyQosStage(const std::string &blob_name,
                                      std::string &obj,
                                      double &max_err,
                                      std::vector<std::string> &comp_pref) const {
    obj.clear();
    max_err = -1.0;
    comp_pref.clear();

    for (size_t i = 0; i < stages_.size(); ++i) {
      try {
        if (std::regex_search(blob_name, stages_[i].match_)) {
          bool overridden = false;
          if (stages_[i].max_error_ >= 0.0) {
            max_err = stages_[i].max_error_;
            overridden = true;
          }
          if (!stages_[i].objective_.empty()) {
            obj = stages_[i].objective_;
            overridden = true;
          }
          if (!stages_[i].compression_preference_.empty()) {
            comp_pref = stages_[i].compression_preference_;
            overridden = true;
          }
          return {static_cast<int>(i), overridden};
        }
      } catch (const std::exception &e) {
        HLOG(kWarning, "dtschedule: Stage {} regex match failed: {}", i, e.what());
      }
    }
    return {-1, false};
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(next_pool_id_, compressed_next_pool_id_, objective_, max_error_, lossy_allowlist_,
       compression_preference_, resample_error_, resample_chance_,
       ccm_, qtable_model_path_, ratio_noise_sigma_,
       load_aware_, load_period_ms_, load_cap_,
       workflow_aware_, dag_path_,
       force_scenario_, decision_order_,
       net_bw_gbps_, trace_path_, trace_candidates_, min_compress_bytes_,
       core_pool_id_);
    // Note: tiers_ and tier_bw_mbps_ not serialized; regenerated from config
  }
};

/**
 * Node load sample: CPU% and worker queue stats.
 * Used by PollNodeLoad to report system state.
 */
struct NodeLoadSample {
  uint64_t ts_ms_ = 0;               ///< Timestamp in milliseconds
  double cpu_util_ = 0.0;            ///< CPU utilization percentage
  uint32_t queued_tasks_ = 0;        ///< Tasks in worker queue
  /** Free bytes per configured tier on that node, in DtscheduleConfig::tiers_
   *  order (fastest first); empty when the node's targets were not read. */
  std::vector<uint64_t> tier_remaining_;

  NodeLoadSample() = default;
  NodeLoadSample(uint64_t ts, double cpu, uint32_t queued)
      : ts_ms_(ts), cpu_util_(cpu), queued_tasks_(queued) {}

  template <class Archive>
  void serialize(Archive &ar) {
    ar(ts_ms_, cpu_util_, queued_tasks_, tier_remaining_);
  }
};

// ============================================================================
// Task struct definitions for dtschedule-specific verbs
// ============================================================================

// ============================================================================
// Task struct definitions for dtschedule-specific verbs
// ============================================================================

/**
 * CreateTask: initialize dtschedule container.
 * Parameters: lib_name, pool_name, custom_pool_id, client, config
 */
/**
 * CreateTask uses the admin module's GetOrCreatePoolTask<DtscheduleConfig>.
 * This provides GetParams() for deserializing the configuration.
 */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<DtscheduleConfig>;

struct DestroyTask : public clio::run::Task {
  DestroyTask() : clio::run::Task() {}

  explicit DestroyTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
  }

  void Copy(const ctp::ipc::FullPtr<DestroyTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }

  template <typename Ar>
  void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

struct MonitorTask : public clio::run::Task {
  MonitorTask() : clio::run::Task() {}

  explicit MonitorTask(const clio::run::TaskId &task_id,
                       const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kMonitor) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
  }

  void Copy(const ctp::ipc::FullPtr<MonitorTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }

  template <typename Ar>
  void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

struct PollNodeLoadTask : public clio::run::Task {
  NodeLoadSample result_;

  PollNodeLoadTask() : clio::run::Task(), result_() {}

  explicit PollNodeLoadTask(const clio::run::TaskId &task_id,
                            const clio::run::PoolId &pool_id,
                            const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPollNodeLoad),
        result_() {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto &other = other_base.template Cast<PollNodeLoadTask>();
    result_ = other->result_;
  }

  void Copy(const ctp::ipc::FullPtr<PollNodeLoadTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    result_ = other->result_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }

  template <typename Ar>
  void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(result_);
  }
};

struct RegisterConsumerTask : public clio::run::Task {
  TagId tag_id_;
  uint32_t consumer_node_ = 0;

  RegisterConsumerTask() : clio::run::Task(), tag_id_(), consumer_node_(0) {}

  explicit RegisterConsumerTask(const clio::run::TaskId &task_id,
                                const clio::run::PoolId &pool_id,
                                const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kRegisterConsumer),
        tag_id_(), consumer_node_(0) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
  }

  void Copy(const ctp::ipc::FullPtr<RegisterConsumerTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    consumer_node_ = other->consumer_node_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, consumer_node_);
  }

  template <typename Ar>
  void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

struct CompressAtTask : public clio::run::Task {
  // IN parameters: blob data and metadata
  TagId tag_id_;
  clio::run::priv::string blob_name_;
  ctp::ipc::ShmPtr<> blob_data_;
  size_t size_ = 0;
  float score_ = -1.0f;
  uint32_t owner_node_ = 0;
  std::string lib_;  ///< Codec library name
  std::string preset_;  ///< Codec preset
  Context context_;  ///< CTE context (version, etc.)

  // OUT parameters: compression results
  size_t comp_size_ = 0;  ///< Compressed size
  double ctime_ms_ = 0.0;  ///< Compression time in ms
  double store_ms_ = 0.0;  ///< Storage time in ms

  CompressAtTask()
      : clio::run::Task(), tag_id_(), blob_name_(), blob_data_(), size_(0),
        score_(-1.0f), owner_node_(0), lib_(), preset_(), context_(),
        comp_size_(0), ctime_ms_(0.0), store_ms_(0.0) {}

  explicit CompressAtTask(const clio::run::TaskId &task_id,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kCompressAt),
        tag_id_(), blob_name_(), blob_data_(), size_(0),
        score_(-1.0f), owner_node_(0), lib_(), preset_(), context_(),
        comp_size_(0), ctime_ms_(0.0), store_ms_(0.0) {}

  ~CompressAtTask() {
#if !CTP_IS_DEVICE_PASS
    if (task_flags_.Any(TASK_DATA_OWNER) && !blob_data_.IsNull()) {
      auto *ipc_manager = CLIO_CPU_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(blob_data_.template Cast<char>());
      }
    }
#endif
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<CompressAtTask>();
    comp_size_ = other->comp_size_;
    ctime_ms_ = other->ctime_ms_;
    store_ms_ = other->store_ms_;
    context_ = other->context_;
  }

  void Copy(const ctp::ipc::FullPtr<CompressAtTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    blob_data_ = other->blob_data_;
    size_ = other->size_;
    score_ = other->score_;
    owner_node_ = other->owner_node_;
    lib_ = other->lib_;
    preset_ = other->preset_;
    context_ = other->context_;
    comp_size_ = other->comp_size_;
    ctime_ms_ = other->ctime_ms_;
    store_ms_ = other->store_ms_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, size_, score_, owner_node_, lib_, preset_,
       context_);
    // Emit bulk data after scalar fields
    ar.bulk(blob_data_, size_, BULK_XFER);
  }

  template <typename Ar>
  void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(comp_size_, ctime_ms_, store_ms_, context_);
  }
};

struct GetDecisionStatsTask : public clio::run::Task {
  DecisionStats result_;

  GetDecisionStatsTask() : clio::run::Task(), result_() {}

  explicit GetDecisionStatsTask(const clio::run::TaskId &task_id,
                                const clio::run::PoolId &pool_id,
                                const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetDecisionStats),
        result_() {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto &other = other_base.template Cast<GetDecisionStatsTask>();
    result_.Add(other->result_);
  }

  void Copy(const ctp::ipc::FullPtr<GetDecisionStatsTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    result_ = other->result_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }

  template <typename Ar>
  void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(result_.puts_, result_.compressed_, result_.skipped_small_,
       result_.bytes_in_, result_.bytes_out_, result_.consumer_tags_,
       result_.load_samples_, result_.copy_skipped_local_, result_.copy_refused_,
       result_.copies_pushed_);
  }
};

struct SetKnobsTask : public clio::run::Task {
  Knobs knobs_;

  SetKnobsTask() : clio::run::Task(), knobs_() {}

  explicit SetKnobsTask(const clio::run::TaskId &task_id,
                        const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kSetKnobs),
        knobs_() {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
  }

  void Copy(const ctp::ipc::FullPtr<SetKnobsTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    knobs_ = other->knobs_;
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar);
    ar(knobs_.ccm_, knobs_.load_aware_, knobs_.workflow_aware_,
       knobs_.force_scenario_, knobs_.decision_order_, knobs_.ratio_noise_sigma_);
  }

  template <typename Ar>
  void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};

/**
 * SampleLoadTask: periodic self-sampling of node load (CPU% and worker stats).
 * Used by the runtime to maintain a small ring buffer of load samples per node.
 * Phase 3: core implementation for local + remote load tracking.
 */
struct SampleLoadTask : public clio::run::Task {
  SampleLoadTask() : clio::run::Task() {}

  explicit SampleLoadTask(const clio::run::TaskId &task_id,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kSampleLoad) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
  }

  void Copy(const ctp::ipc::FullPtr<SampleLoadTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
  }

  template <typename Ar>
  void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }

  template <typename Ar>
  void SerializeOut(Ar &ar) { Task::SerializeOut(ar); }
};


}  // namespace clio::cte::dtschedule

#endif  // CLIO_CTE_DTSCHEDULE_DTSCHEDULE_TASKS_H_
