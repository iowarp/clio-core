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
        trace_folder_path_(other.trace_folder_path_),
        next_pool_id_(other.next_pool_id_),
        tracking_enabled_(other.tracking_enabled_) {
    (void)pool_id;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(qtable_model_path_, linreg_model_path_, distribution_model_path_,
       dnn_model_weights_path_, trace_folder_path_, tracking_enabled_);
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

  void Copy(const ctp::ipc::FullPtr<DestroyTask>& other) {
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

  void Copy(const ctp::ipc::FullPtr<DynamicScheduleTask>& other) {
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

  // Output fields
  OUT float tier_score_;                  // Selected tier score (0-1, normalized)

  // SHM constructor
  CompressTask()
      : clio::run::Task(), tag_id_(clio::cte::core::TagId::GetNull()),
        blob_name_(CTP_MALLOC), offset_(0), size_(0),
        blob_data_(ctp::ipc::ShmPtr<>::GetNull()), score_(0.5f),
        context_(), flags_(0), core_pool_id_(clio::run::PoolId::GetNull()),
        tier_score_(0.0f) {}

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
                        const clio::run::PoolId &core_pool_id)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kCompress),
        tag_id_(tag_id), blob_name_(CTP_MALLOC, blob_name),
        offset_(offset), size_(size), blob_data_(blob_data), score_(score),
        context_(context), flags_(flags), core_pool_id_(core_pool_id),
        tier_score_(0.0f) {}

  void Copy(const ctp::ipc::FullPtr<CompressTask>& other) {
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

  void Copy(const ctp::ipc::FullPtr<DecompressTask>& other) {
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

  void Copy(const ctp::ipc::FullPtr<PollNodeLoadTask> &other) {
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

  void Copy(const ctp::ipc::FullPtr<PollConsumersTask> &other) {
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
