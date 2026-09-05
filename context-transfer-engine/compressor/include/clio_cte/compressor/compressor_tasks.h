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

// Copyright 2024 IOWarp contributors
#ifndef CLIO_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_
#define CLIO_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/compressor/autogen/compressor_methods.h>

namespace clio::cte::compressor {

/** Import Context from core for compression operations */
using Context = clio::cte::core::Context;
using CteOp = clio::cte::core::CteOp;
using Timestamp = std::chrono::steady_clock::time_point;

/**
 * CreateParams - Configuration for compressor container creation
 */
struct CompressorConfig {
  static constexpr const char* chimod_lib_name = "clio_cte_compressor";

  std::string qtable_model_path_;
  std::string linreg_model_path_;
  std::string distribution_model_path_;
  std::string dnn_model_weights_path_;
  // Directory of a trained NeuroPress NN's .nnwt weights (issue #693
  // 3). Loaded once at Create() time and consulted by EstCompressionStats()'s
  // dynamic-selection path (compressor_runtime.cc) whenever set.
  std::string neuropress_model_path_;
  // Master switch for ONLINE LEARNING (SGD Phase 1 + exploration Phase 2).
  // Off by default, mirroring NeuroPress's own
  // g_online_learning_enabled{false} (gpucompress_api.cpp), which gates its
  // whole learning block (gpucompress_compress.cpp) and must be turned on
  // explicitly via gpucompress_enable_online_learning(). Configuring
  // neuropress_model_path_ alone must give INFERENCE ONLY: a deployment
  // that just wants the trained model should not silently get a model whose
  // weights drift out from under it.
  bool neuropress_online_learning_enabled_ = false;
  // MAPE (mean absolute percentage error, weighted-cost) threshold that
  // gates online SGD after a real compress (issue #693). Mirrors
  // NeuroPress's own g_reinforce_mape_threshold / GPUCOMPRESS_MAPE_LOW_THRESH
  // (default 0.30 = 30%, gpucompress_api.cpp): training only fires when the
  // model's prediction for the chunk actually compressed was wrong by more
  // than this fraction, not on every chunk. This never writes .nnwt back to
  // disk -- it only adjusts the in-memory weights for this process's
  // lifetime, matching NeuroPress's own behavior (no runtime API persists a
  // trained model file either).
  float neuropress_mape_threshold_ = 0.30f;
  /**
   * Online-SGD learning rate. Mirrors NeuroPress's g_reinforce_lr
   * (gpucompress_api.cpp, default 0.01f), which is settable at runtime
   * through gpucompress_set_reinforcement(). Only the main SGD kernel
   * consumes it -- `lr_out = learning_rate * clip_scale` (nn_gpu.cu).
   * The deferred decompression-head pass takes the value but never uses it:
   * its step comes entirely from its own trust region.
   *
   * Values <= 0 are ignored rather than applied, matching upstream's
   * `if (learning_rate > 0.0f)` guard (gpucompress_learning.cpp).
   */
  float neuropress_learning_rate_ = 0.01f;
  // K-way exploration. Off by default, matching NeuroPress's own
  // g_exploration_enabled{false} -- an opt-in, not the normal route. When
  // enabled and error_pct (the same metric Phase 1 above uses) crosses this
  // HIGHER threshold (default 0.50, mirrors g_exploration_threshold /
  // GPUCOMPRESS_MAPE_HIGH_THRESH), up to neuropress_exploration_k_
  // alternative candidates are actually compressed (never stored) purely to
  // generate more real-outcome training samples.
  bool neuropress_exploration_enabled_ = false;
  float neuropress_exploration_threshold_ = 0.50f;
  int neuropress_exploration_k_ = 3;  // NeuroPress's own default K
  /**
   * Exhaustive-search ("best") mode, mirroring NeuroPress's g_best_mode /
   * gpucompress_set_best_mode(). Every chunk is explored with the full
   * remaining action space regardless of prediction error, and the cheapest
   * measured configuration is stored.
   *
   * This is a MEASUREMENT mode, not a faster one: it compresses each chunk
   * ~32 times and is correspondingly slower. Its purpose is to establish the
   * ceiling on selection quality reachable inside the action space, so a
   * model's real selections can be scored against the best available one.
   *
   * Both SGD phases are suppressed while it is on, exactly as upstream
   * suppresses them (`&& !g_best_mode.load()` on each): the training signal
   * is derived from the model's own prediction error, and a mode that
   * overrides the model's choice on every chunk would teach it from outcomes
   * it did not choose. Upstream calls that weight contamination.
   *
   * Requires neuropress_exploration_enabled_ -- this widens the exploration
   * gate, it does not open it.
   */
  bool neuropress_best_mode_ = false;
  /**
   * Static codec override. When non-empty, every chunk is compressed with
   * this library and NeuroPress's selection is not consulted at all -- no
   * inference, no learning, no exploration.
   *
   * This is the control condition for any claim about the selector. A
   * NeuroPress ratio only means something against the ratio a fixed codec
   * would have produced on the same bytes, and until now Clio had no way to
   * produce that number through the real write path: the only way to hold
   * the codec fixed was to not compress at all.
   *
   * Value is a canonical library name from CompressionFactory's registry
   * ("nvcomp-zstd", "zstd", "lz4", ...). An unknown name resolves to zstd,
   * matching WireIdForName's documented fallback. Note "zstd" is the CPU
   * codec and "nvcomp-zstd" is the GPU one -- upstream NeuroPress's
   * GPUCOMPRESS_ALGO_ZSTD builds an nvcomp::ZstdManager, so "nvcomp-zstd"
   * is its like-for-like counterpart, not "zstd".
   */
  std::string neuropress_static_lib_;
  /**
   * Byte-shuffle element size for the static codec, in bytes. 0 = no shuffle.
   *
   * Only meaningful alongside neuropress_static_lib_. Shuffling groups the
   * Nth byte of every element together, which makes the exponent and sign
   * bytes of a float array -- highly repetitive -- adjacent, and is often
   * worth more than the choice of codec on floating-point data.
   *
   * Upstream offers exactly one width: GPUCOMPRESS_PREPROC_SHUFFLE_4, and
   * its NN action space encodes shuffle as a single bit, so 4 is the
   * like-for-like setting. Clio's packed preset carries the width itself, so
   * 8 is also expressible here and worth trying on float64 -- but a run using
   * it is no longer comparable with upstream.
   */
  clio::run::u32 neuropress_static_shuffle_ = 0;
  /**
   * Apply linear quantization to the static codec's input, at the run's
   * error bound. Only meaningful alongside neuropress_static_lib_.
   *
   * Upstream's non-AUTO path carries quantize and shuffle as INDEPENDENT
   * preprocessing flags -- GPUCOMPRESS_PREPROC_QUANTIZE (0x10) and
   * GPUCOMPRESS_PREPROC_SHUFFLE_4 (0x02) -- and applies the first when the
   * flag is set AND the error bound is positive
   * (gpucompress_compress.cpp:434). This mirrors that: the bit is a REQUEST,
   * and the actual decision still needs error_bound > 0, checked where
   * upstream checks it.
   *
   * WHY IT EXISTS. Without it the static path can only express the 16
   * LOSSLESS actions of the 32-action space, so a "best fixed codec" control
   * is forced to compete lossless against a lossy selector. Measured on VPIC:
   * static ans+shuffle4 reached 1.235x while NeuroPress at eb 1e-3 reached
   * 4.825x -- a 4x gap that is an artifact of the control, not a result. The
   * per-chunk oracle picks a QUANTIZED action on most chunks of every
   * workload measured, so a control that cannot quantize is not a baseline.
   */
  bool neuropress_static_quantize_ = false;
  std::string trace_folder_path_;
  clio::run::PoolId next_pool_id_;  ///< Pool ID of the next module in the pipeline
                               ///< (e.g., CTE core at 513.0)
  /**
   * When true (default), the compressor tracks per-tag consumer node sets
   * via Decompress requests and uses them to route Compress placement
   * toward the most recent consumer of the same tag. When false, the
   * tracking map and PollConsumers periodic are bypassed and Compress
   * tasks fall back to pure DirectHash routing on the tag_id. Set false
   * for benchmarks where you want to isolate the cost of the tracking
   * mechanism itself, or for workloads with no clear producer-consumer
   * locality.
   */
  bool tracking_enabled_ = true;

  CompressorConfig() : next_pool_id_(clio::run::PoolId::GetNull()) {}

  CompressorConfig(const clio::run::PoolId &pool_id, const CompressorConfig &other)
      : qtable_model_path_(other.qtable_model_path_),
        linreg_model_path_(other.linreg_model_path_),
        distribution_model_path_(other.distribution_model_path_),
        dnn_model_weights_path_(other.dnn_model_weights_path_),
        neuropress_model_path_(other.neuropress_model_path_),
        neuropress_online_learning_enabled_(
            other.neuropress_online_learning_enabled_),
        neuropress_mape_threshold_(other.neuropress_mape_threshold_),
        neuropress_learning_rate_(other.neuropress_learning_rate_),
        neuropress_exploration_enabled_(other.neuropress_exploration_enabled_),
        neuropress_exploration_threshold_(
            other.neuropress_exploration_threshold_),
        neuropress_exploration_k_(other.neuropress_exploration_k_),
        neuropress_best_mode_(other.neuropress_best_mode_),
        neuropress_static_lib_(other.neuropress_static_lib_),
        neuropress_static_shuffle_(other.neuropress_static_shuffle_),
        neuropress_static_quantize_(other.neuropress_static_quantize_),
        trace_folder_path_(other.trace_folder_path_),
        next_pool_id_(other.next_pool_id_),
        tracking_enabled_(other.tracking_enabled_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    // next_pool_id_ MUST round-trip: it is the interposition-chain target
    // (issue #886) — omitting it silently rewired a programmatically
    // created compressor pool straight to the default core, bypassing any
    // interposer chained beneath it.
    ar(qtable_model_path_, linreg_model_path_, distribution_model_path_,
       dnn_model_weights_path_, neuropress_model_path_,
       neuropress_online_learning_enabled_,
       neuropress_mape_threshold_, neuropress_learning_rate_,
       neuropress_exploration_enabled_,
       neuropress_exploration_threshold_, neuropress_exploration_k_,
       neuropress_best_mode_, neuropress_static_lib_,
       neuropress_static_shuffle_, neuropress_static_quantize_,
       trace_folder_path_, next_pool_id_, tracking_enabled_);
  }

  /**
   * Load configuration from compose YAML.
   * Reads next_pool_id from the pool config.
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    // Parse next_pool_id from compose YAML config
    if (!pool_config.config_.empty()) {
      try {
        YAML::Node node = YAML::Load(pool_config.config_);
        if (node["next_pool_id"]) {
          std::string next_str = node["next_pool_id"].as<std::string>();
          // Parse "major.minor" format
          auto dot = next_str.find('.');
          if (dot != std::string::npos) {
            clio::run::u32 major = std::stoul(next_str.substr(0, dot));
            clio::run::u32 minor = std::stoul(next_str.substr(dot + 1));
            next_pool_id_ = clio::run::PoolId(major, minor);
          }
        }
        if (node["tracking_enabled"]) {
          tracking_enabled_ = node["tracking_enabled"].as<bool>();
        }
        // Predictor and trace paths. These are written into the compose file
        // by jarvis_clio_core.clio_compress and were, until now, read by
        // nobody: config_manager.cc hands the whole pool node over for
        // "module-specific parsing" and this is the only place that parsing
        // happens, so every model path configured through a pipeline was
        // silently dropped and the compressor ran with its defaults. Each is
        // empty by default and every consumer short-circuits on empty, so
        // parsing them changes nothing for a config that does not set them.
        if (node["qtable_model_path"]) {
          qtable_model_path_ = node["qtable_model_path"].as<std::string>();
        }
        if (node["linreg_model_path"]) {
          linreg_model_path_ = node["linreg_model_path"].as<std::string>();
        }
        if (node["distribution_model_path"]) {
          distribution_model_path_ =
              node["distribution_model_path"].as<std::string>();
        }
        if (node["dnn_model_weights_path"]) {
          dnn_model_weights_path_ =
              node["dnn_model_weights_path"].as<std::string>();
        }
        if (node["trace_folder_path"]) {
          trace_folder_path_ = node["trace_folder_path"].as<std::string>();
        }
        // The NeuroPress master switch (issue #693): the directory holding the
        // trained .nnwt weights. Everything else in this block is inert
        // without it -- compressor_runtime.cc only builds the predictor when
        // this is non-empty, so a compose file that set, say, an exploration
        // threshold but could not set this got a compressor with no model and
        // no indication of why.
        if (node["neuropress_model_path"]) {
          neuropress_model_path_ =
              node["neuropress_model_path"].as<std::string>();
        }
        if (node["neuropress_online_learning_enabled"]) {
          neuropress_online_learning_enabled_ =
              node["neuropress_online_learning_enabled"].as<bool>();
        }
        if (node["neuropress_mape_threshold"]) {
          neuropress_mape_threshold_ =
              node["neuropress_mape_threshold"].as<float>();
        }
        if (node["neuropress_learning_rate"]) {
          neuropress_learning_rate_ =
              node["neuropress_learning_rate"].as<float>();
        }
        if (node["neuropress_exploration_enabled"]) {
          neuropress_exploration_enabled_ =
              node["neuropress_exploration_enabled"].as<bool>();
        }
        if (node["neuropress_exploration_threshold"]) {
          neuropress_exploration_threshold_ =
              node["neuropress_exploration_threshold"].as<float>();
        }
        if (node["neuropress_exploration_k"]) {
          neuropress_exploration_k_ =
              node["neuropress_exploration_k"].as<int>();
        }
        // Static codec override -- the control condition. Deliberately
        // parsed alongside the NeuroPress knobs rather than above them: it
        // overrides all of them, and reading it here keeps that visible.
        if (node["neuropress_static_lib"]) {
          neuropress_static_lib_ =
              node["neuropress_static_lib"].as<std::string>();
        }
        if (node["neuropress_static_shuffle"]) {
          neuropress_static_shuffle_ =
              node["neuropress_static_shuffle"].as<clio::run::u32>();
        }
        if (node["neuropress_static_quantize"]) {
          neuropress_static_quantize_ =
              node["neuropress_static_quantize"].as<bool>();
        }
        // Exhaustive-search measurement mode. Widens the exploration gate
        // rather than opening it, so the runtime forces exploration on and K
        // to the full action space when this is set -- see Create() in
        // compressor_runtime.cc; no coupling is enforced here.
        if (node["neuropress_best_mode"]) {
          neuropress_best_mode_ = node["neuropress_best_mode"].as<bool>();
        }
      } catch (...) {
        // Config parsing is best-effort
      }
    }
  }
};

/**
 * CreateTask - Use GetOrCreatePoolTask for standard pool creation
 */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CompressorConfig>;

/**
 * DestroyTask - Cleanup the compressor container
 */
struct DestroyTask : public clio::run::Task {
  // No additional fields needed
  DestroyTask() : clio::run::Task() {}

  explicit DestroyTask(const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
                       const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDestroy) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<DestroyTask>());
  }

  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    // No additional fields to copy beyond clio::run::Task
  }

  template <typename Ar> void SerializeStart(Ar &ar) { task_serialize<Ar>(ar); }
  template <typename Ar> void SerializeEnd(Ar &ar) {}
};

/**
 * Target state - cached information about storage targets
 * Used by compressor to make intelligent compression/tiering decisions
 */
struct TargetState {
  std::string target_name_;      // Name of the target
  float target_score_;           // Target score (0-1, normalized log bandwidth)
  clio::run::u64 remaining_space_;     // Remaining allocatable space in bytes
  clio::run::u64 bytes_written_;       // Bytes written to target
  Timestamp last_updated_;       // When this state was last refreshed

  TargetState()
      : target_score_(0.0f), remaining_space_(0), bytes_written_(0),
        last_updated_(std::chrono::steady_clock::now()) {}

  TargetState(const std::string &name, float score, clio::run::u64 space, clio::run::u64 written)
      : target_name_(name), target_score_(score), remaining_space_(space),
        bytes_written_(written), last_updated_(std::chrono::steady_clock::now()) {}
};

using MonitorTask = clio::run::admin::MonitorTask;

/**
 * Compression telemetry data structure for performance monitoring
 * Tracks compression decisions and actual performance
 */
struct CompressionTelemetry {
  CteOp op_;                     // Operation type (kPutBlob or kGetBlob)
  int compress_lib_;             // Compression library used (0 = none)
  clio::run::u64 original_size_;       // Original data size in bytes
  clio::run::u64 compressed_size_;     // Compressed data size in bytes
  double compress_time_ms_;      // Actual compression time in milliseconds
  double decompress_time_ms_;    // Actual decompression time in milliseconds
  double psnr_db_;               // Actual PSNR for lossy compression
  Timestamp timestamp_;          // When operation occurred
  std::uint64_t logical_time_;   // Logical time for ordering

  CompressionTelemetry()
      : op_(CteOp::kPutBlob), compress_lib_(0), original_size_(0),
        compressed_size_(0), compress_time_ms_(0.0), decompress_time_ms_(0.0),
        psnr_db_(0.0), timestamp_(std::chrono::steady_clock::now()),
        logical_time_(0) {}

  CompressionTelemetry(CteOp op, int lib, clio::run::u64 orig_size, clio::run::u64 comp_size,
                       double comp_time, double decomp_time, double psnr,
                       const Timestamp &ts, std::uint64_t logical_time = 0)
      : op_(op), compress_lib_(lib), original_size_(orig_size),
        compressed_size_(comp_size), compress_time_ms_(comp_time),
        decompress_time_ms_(decomp_time), psnr_db_(psnr),
        timestamp_(ts), logical_time_(logical_time) {}

  // Calculate compression ratio
  double GetCompressionRatio() const {
    if (compressed_size_ == 0) return 1.0;
    return static_cast<double>(original_size_) / static_cast<double>(compressed_size_);
  }

  // Serialization support for cereal
  template <class Archive> void serialize(Archive &ar) {
    // Convert timestamps to duration counts for serialization
    auto ts_count = timestamp_.time_since_epoch().count();
    ar(op_, compress_lib_, original_size_, compressed_size_,
       compress_time_ms_, decompress_time_ms_, psnr_db_,
       ts_count, logical_time_);
    // Note: On deserialization, timestamps will be reconstructed from counts
    if (Archive::is_loading::value) {
      timestamp_ = Timestamp(Timestamp::duration(ts_count));
    }
  }
};

/**
 * DynamicScheduleTask - Analyzes data and determines optimal compression strategy
 * Then performs compression and calls PutBlob to store the data.
 * Has the same inputs as PutBlobTask for seamless integration.
 */
struct DynamicScheduleTask : public clio::run::Task {
  // Same inputs as PutBlobTask
  IN clio::cte::core::TagId tag_id_;        // Tag ID for blob grouping
  INOUT clio::run::priv::string blob_name_;     // Blob name (required)
  IN clio::run::u64 offset_;                    // Offset within blob
  IN clio::run::u64 size_;                      // Size of blob data
  IN ctp::ipc::ShmPtr<> blob_data_;           // Blob data (shared memory pointer)
  IN float score_;                        // Score 0-1 for placement decisions
  INOUT Context context_;                 // Context for compression control and statistics
  IN clio::run::u32 flags_;                     // Operation flags
  IN clio::run::PoolId core_pool_id_;           // Pool ID of core chimod for PutBlob

  // Output fields
  OUT float tier_score_;                  // Selected tier score (0-1, normalized)

  // SHM constructor
  DynamicScheduleTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()),
        blob_name_(CTP_MALLOC), offset_(0), size_(0),
        blob_data_(ctp::ipc::ShmPtr<>::GetNull()), score_(0.5f),
        context_(), flags_(0), core_pool_id_(clio::run::PoolId::GetNull()),
        tier_score_(0.0f) {}

  // Emplace constructor
  explicit DynamicScheduleTask(const clio::run::TaskId &task_id,
                               const clio::run::PoolId &pool_id,
                               const clio::run::PoolQuery &pool_query,
                               const clio::cte::core::TagId &tag_id,
                               const std::string &blob_name,
                               clio::run::u64 offset, clio::run::u64 size,
                               ctp::ipc::ShmPtr<> blob_data,
                               float score, const Context &context,
                               clio::run::u32 flags,
                               const clio::run::PoolId &core_pool_id)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDynamicSchedule),
        tag_id_(tag_id), blob_name_(CTP_MALLOC, blob_name),
        offset_(offset), size_(size), blob_data_(blob_data), score_(score),
        context_(context), flags_(flags), core_pool_id_(core_pool_id),
        tier_score_(0.0f) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<DynamicScheduleTask>());
  }

  void Copy(const ctp::ipc::FullPtr<DynamicScheduleTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    blob_data_ = other->blob_data_;
    score_ = other->score_;
    context_ = other->context_;
    flags_ = other->flags_;
    core_pool_id_ = other->core_pool_id_;
    tier_score_ = other->tier_score_;
  }

  /** Serialize */
  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(tag_id_, blob_name_, offset_, size_, score_, context_, flags_,
       core_pool_id_, tier_score_);
    ar.bulk(blob_data_, size_, BULK_XFER);
  }

  /** Deserialize */
  template <typename Ar>
  void SerializeEnd(Ar &ar) {
    ar(blob_name_, context_, tier_score_);
  }
};

/**
 * CompressTask - Performs compression and calls PutBlob to store the data.
 * Has the same inputs as PutBlobTask for seamless integration.
 */
struct CompressTask : public clio::run::Task {
  // Same inputs as PutBlobTask
  IN clio::cte::core::TagId tag_id_;        // Tag ID for blob grouping
  INOUT clio::run::priv::string blob_name_;     // Blob name (required)
  IN clio::run::u64 offset_;                    // Offset within blob
  IN clio::run::u64 size_;                      // Size of blob data
  IN ctp::ipc::ShmPtr<> blob_data_;           // Blob data (shared memory pointer)
  IN float score_;                        // Score 0-1 for placement decisions
  INOUT Context context_;                 // Context for compression control and statistics
  IN clio::run::u32 flags_;                     // Operation flags
  IN clio::run::PoolId core_pool_id_;           // Pool ID of core chimod for PutBlob
  /**
   * Compress and MEASURE, but do not store.
   *
   * Set by DynamicSchedule when exploration is on. Exploration only learns
   * which codec wins by compressing alternatives and comparing them against
   * the primary's measured cost, so the primary has to run -- but storing it
   * first means a winner is written over the top, and PutBlob overwrites
   * bytes without shortening a blob that was already longer. The rejected
   * candidate's footprint stays allocated (4-9x measured), and the two writes
   * leave two WAL records for one blob, which a shard-sequential replay can
   * apply out of order.
   *
   * With this set, Compress hands the bytes back through stored_* below and
   * the caller performs exactly one PutBlob once the winner is known -- which
   * is also what upstream effectively does, having no blob store: a winner
   * simply overwrites the primary in the caller's output buffer.
   */
  IN bool no_store_;

  // Output fields
  OUT float tier_score_;                  // Selected tier score (0-1, normalized)
  /**
   * no_store_ only. The compressed image (header + payload) and its length.
   *
   * OWNERSHIP TRANSFERS TO THE CALLER. stored_gpu_alloc_ is non-null when the
   * bytes live in device memory (Compress keeps its output on the device
   * whenever the input chunk is device-resident); the caller must then
   * FreeGpuBackend it, and FreeBuffer the host SHM otherwise. Compress's own
   * device-scratch guard deliberately releases its claim on this one
   * allocation so it is not freed out from under the caller.
   */
  OUT ctp::ipc::ShmPtr<> stored_data_;
  OUT clio::run::u64 stored_size_;
  OUT ctp::ipc::AllocatorId stored_gpu_alloc_;
  /**
   * Whether the caller must free stored_data_.
   *
   * False when the codec did not beat the input: the bytes to store are then
   * the caller's ORIGINAL buffer, handed straight back, and freeing it would
   * destroy memory this task never allocated.
   */
  OUT bool stored_owned_;

  // SHM constructor
  CompressTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()),
        blob_name_(CTP_MALLOC), offset_(0), size_(0),
        blob_data_(ctp::ipc::ShmPtr<>::GetNull()), score_(0.5f),
        context_(), flags_(0), core_pool_id_(clio::run::PoolId::GetNull()),
        no_store_(false), tier_score_(0.0f),
        stored_data_(ctp::ipc::ShmPtr<>::GetNull()), stored_size_(0),
        stored_gpu_alloc_(), stored_owned_(false) {}

  // Emplace constructor
  explicit CompressTask(const clio::run::TaskId &task_id,
                        const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query,
                        const clio::cte::core::TagId &tag_id,
                        const std::string &blob_name,
                        clio::run::u64 offset, clio::run::u64 size,
                        ctp::ipc::ShmPtr<> blob_data,
                        float score, const Context &context,
                        clio::run::u32 flags,
                        const clio::run::PoolId &core_pool_id,
                        bool no_store = false)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kCompress),
        tag_id_(tag_id), blob_name_(CTP_MALLOC, blob_name),
        offset_(offset), size_(size), blob_data_(blob_data), score_(score),
        context_(context), flags_(flags), core_pool_id_(core_pool_id),
        no_store_(no_store), tier_score_(0.0f),
        stored_data_(ctp::ipc::ShmPtr<>::GetNull()), stored_size_(0),
        stored_gpu_alloc_(), stored_owned_(false) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<CompressTask>());
  }

  void Copy(const ctp::ipc::FullPtr<CompressTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    blob_data_ = other->blob_data_;
    score_ = other->score_;
    context_ = other->context_;
    flags_ = other->flags_;
    core_pool_id_ = other->core_pool_id_;
    no_store_ = other->no_store_;
    tier_score_ = other->tier_score_;
    stored_data_ = other->stored_data_;
    stored_size_ = other->stored_size_;
    stored_gpu_alloc_ = other->stored_gpu_alloc_;
    stored_owned_ = other->stored_owned_;
  }

  /** Serialize */
  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(tag_id_, blob_name_, offset_, size_, score_, context_, flags_,
       core_pool_id_, no_store_, tier_score_);
    ar.bulk(blob_data_, size_, BULK_XFER);
  }

  /** Deserialize */
  template <typename Ar>
  void SerializeEnd(Ar &ar) {
    // stored_* are deliberately NOT serialized. They hand a buffer's OWNERSHIP
    // to the caller, which only means anything in-process -- a pointer and an
    // allocator id carried to another address space would name memory the
    // receiver cannot free. no_store_ is only ever set on the local
    // DynamicSchedule -> Compress call, so this never crosses a boundary.
    ar(blob_name_, context_, tier_score_);
  }
};

/**
 * DecompressTask - Calls GetBlob to retrieve data, then performs decompression.
 * Has the same inputs as GetBlobTask plus decompression output.
 */
struct DecompressTask : public clio::run::Task {
  // Same inputs as GetBlobTask
  IN clio::cte::core::TagId tag_id_;        // Tag ID for blob lookup
  IN clio::run::priv::string blob_name_;        // Blob name (required)
  IN clio::run::u64 offset_;                    // Offset within blob
  IN clio::run::u64 size_;                      // Size of data to retrieve (decompressed size)
  IN clio::run::u32 flags_;                     // Operation flags
  IN ctp::ipc::ShmPtr<> blob_data_;           // Output buffer for decompressed data
  IN clio::run::PoolId core_pool_id_;           // Pool ID of core chimod for GetBlob

  // Output fields
  OUT clio::run::u64 output_size_;              // Actual decompressed size
  OUT double decompress_time_ms_;         // Decompression time in milliseconds

  // SHM constructor
  DecompressTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()),
        blob_name_(CTP_MALLOC), offset_(0), size_(0), flags_(0),
        blob_data_(ctp::ipc::ShmPtr<>::GetNull()),
        core_pool_id_(clio::run::PoolId::GetNull()),
        output_size_(0), decompress_time_ms_(0.0) {}

  // Emplace constructor
  explicit DecompressTask(const clio::run::TaskId &task_id,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const clio::cte::core::TagId &tag_id,
                          const std::string &blob_name,
                          clio::run::u64 offset, clio::run::u64 size,
                          clio::run::u32 flags, ctp::ipc::ShmPtr<> blob_data,
                          const clio::run::PoolId &core_pool_id)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDecompress),
        tag_id_(tag_id), blob_name_(CTP_MALLOC, blob_name),
        offset_(offset), size_(size), flags_(flags), blob_data_(blob_data),
        core_pool_id_(core_pool_id),
        output_size_(0), decompress_time_ms_(0.0) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<DecompressTask>());
  }

  void Copy(const ctp::ipc::FullPtr<DecompressTask>& other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    flags_ = other->flags_;
    blob_data_ = other->blob_data_;
    core_pool_id_ = other->core_pool_id_;
    output_size_ = other->output_size_;
    decompress_time_ms_ = other->decompress_time_ms_;
  }

  /** Serialize */
  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(tag_id_, blob_name_, offset_, size_, flags_, core_pool_id_,
       output_size_, decompress_time_ms_);
    ar.bulk(blob_data_, size_, BULK_EXPOSE);
  }

  /** Deserialize */
  template <typename Ar>
  void SerializeEnd(Ar &ar) {
    ar(output_size_, decompress_time_ms_);
    ar.bulk(blob_data_, size_, BULK_XFER);
  }
};

/**
 * NodeLoadSample - Snapshot of a node's CPU utilization and worker load.
 * Returned as the OUT payload of a PollNodeLoadTask.
 */
struct NodeLoadSample {
  clio::run::u32 node_id_;          ///< Node ID being sampled
  float cpu_usage_pct_;       ///< AggregateOut CPU utilization (0-100)
  float worker_load_us_;      ///< Sum of WorkerStats::load_ across all workers (us)
  clio::run::u32 num_queued_tasks_; ///< Sum of queued tasks across all workers
  clio::run::u32 num_blocked_tasks_;///< Sum of blocked tasks across all workers
  clio::run::u32 num_workers_;      ///< Total worker count on this node

  NodeLoadSample()
      : node_id_(0), cpu_usage_pct_(0.0f), worker_load_us_(0.0f),
        num_queued_tasks_(0), num_blocked_tasks_(0), num_workers_(0) {}

  template <class Archive>
  void serialize(Archive &ar) {
    ar(node_id_, cpu_usage_pct_, worker_load_us_, num_queued_tasks_,
       num_blocked_tasks_, num_workers_);
  }
};

/**
 * PollNodeLoadTask - Query a node's CPU% and worker load.
 *
 * No inputs. The task is routed to a target node via PoolQuery::Physical(node_id)
 * and the runtime samples the local node's stats and writes them into the OUT
 * NodeLoadSample.
 */
struct PollNodeLoadTask : public clio::run::Task {
  OUT NodeLoadSample sample_;  ///< Sampled node load (filled by runtime)

  PollNodeLoadTask() : clio::run::Task(), sample_() {}

  explicit PollNodeLoadTask(const clio::run::TaskId &task_id,
                            const clio::run::PoolId &pool_id,
                            const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPollNodeLoad),
        sample_() {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<PollNodeLoadTask>());
  }

  void Copy(const ctp::ipc::FullPtr<PollNodeLoadTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    sample_ = other->sample_;
  }

  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(sample_);
  }

  template <typename Ar>
  void SerializeEnd(Ar &ar) {
    ar(sample_);
  }
};

/**
 * PollConsumersTask - Periodic task that, when fired, iterates the
 * compressor's tracked consumer list and dispatches PollNodeLoad to each
 * consumer node. Has no IN/OUT fields — it is a trigger.
 */
struct PollConsumersTask : public clio::run::Task {
  PollConsumersTask() : clio::run::Task() {}

  explicit PollConsumersTask(const clio::run::TaskId &task_id,
                             const clio::run::PoolId &pool_id,
                             const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPollConsumers) {}

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<PollConsumersTask>());
  }

  void Copy(const ctp::ipc::FullPtr<PollConsumersTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    (void)other;
  }

  template <typename Ar>
  void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
  }

  template <typename Ar>
  void SerializeEnd(Ar &ar) {}
};

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_
