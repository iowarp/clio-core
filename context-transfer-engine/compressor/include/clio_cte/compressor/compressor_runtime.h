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
#ifndef CLIO_CTE_COMPRESSOR_COMPRESSOR_RUNTIME_H_
#define CLIO_CTE_COMPRESSOR_COMPRESSOR_RUNTIME_H_

#include <atomic>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/corwlock.h>
#include <clio_ctp/data_structures/ipc/ring_buffer.h>
#include <clio_ctp/introspect/system_info.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <clio_cte/compressor/compressor_tasks.h>
#include <clio_cte/compressor/compressor_client.h>
#include <clio_cte/compressor/models/compression_features.h>
#include <clio_cte/compressor/models/qtable_predictor.h>
#include <clio_cte/compressor/models/linreg_table_predictor.h>
#include <clio_cte/compressor/models/distribution_classifier.h>
#include <clio_cte/core/core_client.h>

#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
#include <clio_cte/compressor/models/dense_nn_predictor.h>
#endif

namespace clio::cte::compressor {

/**
 * Compression statistics predicted by AI models
 */
struct CompressionStats {
  int compress_lib_;           // Compression library ID
  int compress_preset_;        // Compression preset (0=balanced, 1=best, 2=default, 3=fast)
  double compression_ratio_;   // Predicted compression ratio
  double compress_time_ms_;    // Predicted compression time in milliseconds
  double decompress_time_ms_;  // Predicted decompression time in milliseconds
  double psnr_db_;             // Predicted PSNR for lossy compression (0 for lossless)

  CompressionStats()
      : compress_lib_(0), compress_preset_(2), compression_ratio_(1.0), compress_time_ms_(0.0),
        decompress_time_ms_(0.0), psnr_db_(0.0) {}

  CompressionStats(int lib, int preset, double ratio, double comp_time, double decomp_time, double psnr)
      : compress_lib_(lib), compress_preset_(preset), compression_ratio_(ratio), compress_time_ms_(comp_time),
        decompress_time_ms_(decomp_time), psnr_db_(psnr) {}
};

/**
 * CTE Compressor Runtime Container
 * Implements compression scheduling and execution
 */
class Runtime : public clio::run::Container {
public:
  using CreateParams = CompressorConfig; // Required for CLIO_TASK_CC (defined in compressor_tasks.h)

  Runtime() = default;
  ~Runtime() override = default;

private:
  // Client for this ChiMod
  Client client_;

  // Core client for target monitoring
  std::unique_ptr<clio::cte::core::Client> core_client_;

  /**
   * Create the container (Method::kCreate)
   * Initializes predictors and loads AI models
   */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);

  /**
   * Destroy the container (Method::kDestroy)
   * Cleanup resources and predictors
   */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);

  /**
   * Monitor container state (Method::kMonitor)
   * Polls core for target information and serializes results with msgpack
   */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);

  /**
   * Dynamic compression scheduling (Method::kDynamicSchedule)
   * Analyzes data and determines optimal compression strategy
   */
  clio::run::TaskResume DynamicSchedule(clio::run::shared_ptr<DynamicScheduleTask> &task);

  /**
   * Compress data (Method::kCompress)
   * Executes compression with specified library and parameters
   */
  clio::run::TaskResume Compress(clio::run::shared_ptr<CompressTask> &task);

  /**
   * Decompress data (Method::kDecompress)
   * Executes decompression with specified library and parameters
   */
  clio::run::TaskResume Decompress(clio::run::shared_ptr<DecompressTask> &task);

  /**
   * Sample this node's CPU utilization and aggregated worker load
   * (Method::kPollNodeLoad). Writes results into task->sample_.
   */
  clio::run::TaskResume PollNodeLoad(clio::run::shared_ptr<PollNodeLoadTask> &task);

  /**
   * Periodic task that iterates the tracked consumer list and dispatches
   * PollNodeLoad to each consumer node (Method::kPollConsumers).
   */
  clio::run::TaskResume PollConsumers(clio::run::shared_ptr<PollConsumersTask> &task);

  /**
   * Schedule a task by resolving Dynamic pool queries.
   */
  clio::run::PoolQuery ScheduleTask(const clio::run::shared_ptr<clio::run::Task> &task) override;

  // Autogen-provided methods
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive& archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  // Container virtual method implementations (defined in autogen/compressor_lib_exec.cc)
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
                                        bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task>& replica_task) override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(clio::run::u32 method,
                                               clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

private:
  // AI model predictors
  std::unique_ptr<QTablePredictor> qtable_predictor_;
  std::unique_ptr<LinRegTablePredictor> linreg_predictor_;
  // Note: DistributionClassifier is a template - use DistributionClassifierFactory for type-erased access

#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
  std::unique_ptr<DenseNNPredictor> nn_predictor_;
#endif

  // Compression telemetry ring buffer for performance monitoring
  using CompressionTelemetryLog = ctp::ipc::ring_buffer<CompressionTelemetry, CLIO_TASK_ALLOC_T>;
  ctp::ipc::ShmPtr<CompressionTelemetryLog> compression_telemetry_log_;
  std::atomic<std::uint64_t> compression_logical_time_;

  // Configuration
  CompressorConfig config_;

  // Target state cache for compression/tiering decisions
  std::unordered_map<std::string, TargetState> target_states_;
  std::mutex target_states_mutex_;

  // Maximum number of distinct consumer nodes tracked PER TAG. Per-tag
  // (rather than per-container) tracking lets the compressor route
  // future Compress traffic for tag T toward the most-recent reader of
  // T rather than the union of all consumers seen across all tags. The
  // bound applies per tag so a tag with more readers than this caps at
  // kMaxConsumersPerTag and silently drops the rest (kDebug-logged).
  static constexpr std::size_t kMaxConsumersPerTag = 32;

  // Per-tag consumer node-id sets (small unsorted vector per tag, size
  // capped at kMaxConsumersPerTag). Map entry is created on first
  // Decompress for the tag. Guarded by tag_consumers_lock_.
  //
  // Skipped entirely when CompressorConfig::tracking_enabled_ is false —
  // ScheduleTask then routes Compress via DirectHash(tag_id) and the
  // periodic PollConsumers becomes a no-op.
  std::unordered_map<clio::cte::core::TagId, std::vector<clio::run::u32>>
      tag_consumers_;
  clio::run::CoRwLock tag_consumers_lock_;

  // Previous CPU times sample, used by PollNodeLoad to compute CPU%.
  ctp::CpuTimes prev_cpu_times_;
  std::mutex cpu_times_mutex_;

  /**
   * Append node_id to tag_consumers_[tag_id] if not already present and
   * under the kMaxConsumersPerTag cap. No-op when
   * CompressorConfig::tracking_enabled_ is false. Acquires
   * tag_consumers_lock_ as a writer when an insert is needed.
   * @param tag_id Tag that the inbound Decompress is reading.
   * @param node_id Originating node ID of the Decompress request.
   */
  void RegisterConsumer(const clio::cte::core::TagId &tag_id, clio::run::u32 node_id);

  /**
   * Pick the best consumer node for placing future Compress traffic for
   * `tag_id`. Returns:
   *   - When tracking_enabled_=false OR no consumers known: invalid
   *     (caller should fall back to DirectHash).
   *   - When tracking_enabled_=true AND consumers known: the most
   *     recently registered consumer (back of the per-tag vector).
   * Acquires tag_consumers_lock_ as a reader.
   * @param tag_id Tag the Compress task is operating on.
   * @param node_id_out Out param: receives the chosen node ID on success.
   * @return true if a consumer was found, false otherwise.
   */
  bool PickConsumerForTag(const clio::cte::core::TagId &tag_id,
                          clio::run::u32 &node_id_out);

  /**
   * Estimate compression statistics using AI models
   * @param chunk Pointer to data chunk
   * @param chunk_size Size of chunk in bytes
   * @param context Compression context with parameters
   * @return Vector of compression statistics for candidate libraries
   */
  std::vector<CompressionStats> EstCompressionStats(
      const void* chunk, clio::run::u64 chunk_size, const Context& context);

  /**
   * Estimate workflow compression time for a specific tier
   * @param chunk_size Size of chunk in bytes
   * @param tier_bw Tier bandwidth in bytes/second
   * @param stats Compression statistics for library
   * @param context Compression context
   * @return Estimated time in milliseconds
   */
  double EstWorkflowCompressTime(
      clio::run::u64 chunk_size, double tier_bw, const CompressionStats& stats,
      const Context& context);

  /**
   * Find best compression for ratio optimization
   * @param chunk Pointer to data chunk
   * @param chunk_size Size of chunk
   * @param container_id Container ID for placement
   * @param stats Vector of compression statistics
   * @param context Compression context
   * @return Tuple of (tier_id, compress_lib, compress_preset, estimated_time, tier_score)
   */
  std::tuple<int, int, int, double, float> BestCompressRatio(
      const void* chunk, clio::run::u64 chunk_size, int container_id,
      const std::vector<CompressionStats>& stats, const Context& context);

  /**
   * Find best compression for time optimization
   * @param chunk Pointer to data chunk
   * @param chunk_size Size of chunk
   * @param container_id Container ID for placement
   * @param stats Vector of compression statistics
   * @param context Compression context
   * @return Tuple of (tier_id, compress_lib, compress_preset, estimated_time, tier_score)
   */
  std::tuple<int, int, int, double, float> BestCompressTime(
      const void* chunk, clio::run::u64 chunk_size, int container_id,
      const std::vector<CompressionStats>& stats, const Context& context);

  /**
   * Choose best compression based on context objective
   * @param context Compression context
   * @param chunk Pointer to data chunk
   * @param chunk_size Size of chunk
   * @param container_id Container ID for placement
   * @param stats Vector of compression statistics
   * @return Tuple of (tier_id, compress_lib, compress_preset, estimated_time, tier_score)
   */
  std::tuple<int, int, int, double, float> BestCompressForNode(
      const Context& context, const void* chunk, clio::run::u64 chunk_size,
      int container_id, const std::vector<CompressionStats>& stats);

  /**
   * Log compression telemetry for performance monitoring
   * @param telemetry Compression telemetry entry
   */
  void LogCompressionTelemetry(const CompressionTelemetry& telemetry);
};

} // namespace clio::cte::compressor

#endif // CLIO_CTE_COMPRESSOR_COMPRESSOR_RUNTIME_H_
