// Copyright 2024 IOWarp contributors
#ifndef WRP_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_
#define WRP_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_

#include <chimaera/chimaera.h>
#include <chimaera/task.h>
#include <chimaera/admin/admin_tasks.h>
#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/compressor/autogen/compressor_methods.h>

namespace wrp_cte::compressor {

/** Import Context from core for compression operations */
using Context = wrp_cte::core::Context;
using CteOp = wrp_cte::core::CteOp;
using Timestamp = std::chrono::steady_clock::time_point;

/**
 * CreateParams - Configuration for compressor container creation
 */
struct CompressorConfig {
  static constexpr const char* chimod_lib_name = "wrp_cte_compressor";

  std::string qtable_model_path_;
  std::string linreg_model_path_;
  std::string distribution_model_path_;
  std::string dnn_model_weights_path_;
  std::string trace_folder_path_;

  CompressorConfig() = default;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(qtable_model_path_, linreg_model_path_, distribution_model_path_,
       dnn_model_weights_path_, trace_folder_path_);
  }
};

/**
 * CreateTask - Use GetOrCreatePoolTask for standard pool creation
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CompressorConfig>;

/**
 * DestroyTask - Cleanup the compressor container
 */
struct DestroyTask : public chi::Task {
  // No additional fields needed
  DestroyTask() : chi::Task() {}

  explicit DestroyTask(const chi::TaskId &task_id, const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query)
      : chi::Task(task_id, pool_id, pool_query, Method::kDestroy) {}

  void Copy(const hipc::FullPtr<DestroyTask>& other) {
    // No additional fields to copy beyond chi::Task
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
  chi::u64 remaining_space_;     // Remaining allocatable space in bytes
  chi::u64 bytes_written_;       // Bytes written to target
  Timestamp last_updated_;       // When this state was last refreshed

  TargetState()
      : target_score_(0.0f), remaining_space_(0), bytes_written_(0),
        last_updated_(std::chrono::steady_clock::now()) {}

  TargetState(const std::string &name, float score, chi::u64 space, chi::u64 written)
      : target_name_(name), target_score_(score), remaining_space_(space),
        bytes_written_(written), last_updated_(std::chrono::steady_clock::now()) {}
};

/**
 * MonitorTask - Periodically poll core for target information
 * Updates cached target state (scores, capacities) for compression decisions
 */
struct MonitorTask : public chi::Task {
  IN chi::PoolId core_pool_id_;  // Pool ID of core chimod to monitor

  // SHM constructor
  MonitorTask() : chi::Task() {}

  // Emplace constructor
  explicit MonitorTask(const chi::TaskId &task_id,
                       const chi::PoolId &pool_id,
                       const chi::PoolQuery &pool_query,
                       const chi::PoolId &core_pool_id)
      : chi::Task(task_id, pool_id, pool_query, Method::kMonitor),
        core_pool_id_(core_pool_id) {}

  void Copy(const hipc::FullPtr<MonitorTask>& other) {
    core_pool_id_ = other->core_pool_id_;
  }

  template <typename Ar> void SerializeStart(Ar &ar) {
    task_serialize<Ar>(ar);
    ar(core_pool_id_);
  }
  template <typename Ar> void SerializeEnd(Ar &ar) {}
};

/**
 * Compression telemetry data structure for performance monitoring
 * Tracks compression decisions and actual performance
 */
struct CompressionTelemetry {
  CteOp op_;                     // Operation type (kPutBlob or kGetBlob)
  int compress_lib_;             // Compression library used (0 = none)
  chi::u64 original_size_;       // Original data size in bytes
  chi::u64 compressed_size_;     // Compressed data size in bytes
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

  CompressionTelemetry(CteOp op, int lib, chi::u64 orig_size, chi::u64 comp_size,
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
struct DynamicScheduleTask : public chi::Task {
  // Same inputs as PutBlobTask
  IN wrp_cte::core::TagId tag_id_;        // Tag ID for blob grouping
  INOUT chi::priv::string blob_name_;     // Blob name (required)
  IN chi::u64 offset_;                    // Offset within blob
  IN chi::u64 size_;                      // Size of blob data
  IN hipc::ShmPtr<> blob_data_;           // Blob data (shared memory pointer)
  IN float score_;                        // Score 0-1 for placement decisions
  INOUT Context context_;                 // Context for compression control and statistics
  IN chi::u32 flags_;                     // Operation flags
  IN chi::PoolId core_pool_id_;           // Pool ID of core chimod for PutBlob

  // Output fields
  OUT float tier_score_;                  // Selected tier score (0-1, normalized)

  // SHM constructor
  DynamicScheduleTask()
      : chi::Task(), tag_id_(wrp_cte::core::TagId::GetNull()),
        blob_name_(HSHM_MALLOC), offset_(0), size_(0),
        blob_data_(hipc::ShmPtr<>::GetNull()), score_(0.5f),
        context_(), flags_(0), core_pool_id_(chi::PoolId::GetNull()),
        tier_score_(0.0f) {}

  // Emplace constructor
  explicit DynamicScheduleTask(const chi::TaskId &task_id,
                               const chi::PoolId &pool_id,
                               const chi::PoolQuery &pool_query,
                               const wrp_cte::core::TagId &tag_id,
                               const std::string &blob_name,
                               chi::u64 offset, chi::u64 size,
                               hipc::ShmPtr<> blob_data,
                               float score, const Context &context,
                               chi::u32 flags,
                               const chi::PoolId &core_pool_id)
      : chi::Task(task_id, pool_id, pool_query, Method::kDynamicSchedule),
        tag_id_(tag_id), blob_name_(HSHM_MALLOC, blob_name),
        offset_(offset), size_(size), blob_data_(blob_data), score_(score),
        context_(context), flags_(flags), core_pool_id_(core_pool_id),
        tier_score_(0.0f) {}

  void Copy(const hipc::FullPtr<DynamicScheduleTask>& other) {
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
struct CompressTask : public chi::Task {
  // Same inputs as PutBlobTask
  IN wrp_cte::core::TagId tag_id_;        // Tag ID for blob grouping
  INOUT chi::priv::string blob_name_;     // Blob name (required)
  IN chi::u64 offset_;                    // Offset within blob
  IN chi::u64 size_;                      // Size of blob data
  IN hipc::ShmPtr<> blob_data_;           // Blob data (shared memory pointer)
  IN float score_;                        // Score 0-1 for placement decisions
  INOUT Context context_;                 // Context for compression control and statistics
  IN chi::u32 flags_;                     // Operation flags
  IN chi::PoolId core_pool_id_;           // Pool ID of core chimod for PutBlob

  // Output fields
  OUT float tier_score_;                  // Selected tier score (0-1, normalized)

  // SHM constructor
  CompressTask()
      : chi::Task(), tag_id_(wrp_cte::core::TagId::GetNull()),
        blob_name_(HSHM_MALLOC), offset_(0), size_(0),
        blob_data_(hipc::ShmPtr<>::GetNull()), score_(0.5f),
        context_(), flags_(0), core_pool_id_(chi::PoolId::GetNull()),
        tier_score_(0.0f) {}

  // Emplace constructor
  explicit CompressTask(const chi::TaskId &task_id,
                        const chi::PoolId &pool_id,
                        const chi::PoolQuery &pool_query,
                        const wrp_cte::core::TagId &tag_id,
                        const std::string &blob_name,
                        chi::u64 offset, chi::u64 size,
                        hipc::ShmPtr<> blob_data,
                        float score, const Context &context,
                        chi::u32 flags,
                        const chi::PoolId &core_pool_id)
      : chi::Task(task_id, pool_id, pool_query, Method::kCompress),
        tag_id_(tag_id), blob_name_(HSHM_MALLOC, blob_name),
        offset_(offset), size_(size), blob_data_(blob_data), score_(score),
        context_(context), flags_(flags), core_pool_id_(core_pool_id),
        tier_score_(0.0f) {}

  void Copy(const hipc::FullPtr<CompressTask>& other) {
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
struct DecompressTask : public chi::Task {
  // Same inputs as GetBlobTask
  IN wrp_cte::core::TagId tag_id_;        // Tag ID for blob lookup
  IN chi::priv::string blob_name_;        // Blob name (required)
  IN chi::u64 offset_;                    // Offset within blob
  IN chi::u64 size_;                      // Size of data to retrieve (decompressed size)
  IN chi::u32 flags_;                     // Operation flags
  IN hipc::ShmPtr<> blob_data_;           // Output buffer for decompressed data
  IN chi::PoolId core_pool_id_;           // Pool ID of core chimod for GetBlob

  // Output fields
  OUT chi::u64 output_size_;              // Actual decompressed size
  OUT double decompress_time_ms_;         // Decompression time in milliseconds

  // SHM constructor
  DecompressTask()
      : chi::Task(), tag_id_(wrp_cte::core::TagId::GetNull()),
        blob_name_(HSHM_MALLOC), offset_(0), size_(0), flags_(0),
        blob_data_(hipc::ShmPtr<>::GetNull()),
        core_pool_id_(chi::PoolId::GetNull()),
        output_size_(0), decompress_time_ms_(0.0) {}

  // Emplace constructor
  explicit DecompressTask(const chi::TaskId &task_id,
                          const chi::PoolId &pool_id,
                          const chi::PoolQuery &pool_query,
                          const wrp_cte::core::TagId &tag_id,
                          const std::string &blob_name,
                          chi::u64 offset, chi::u64 size,
                          chi::u32 flags, hipc::ShmPtr<> blob_data,
                          const chi::PoolId &core_pool_id)
      : chi::Task(task_id, pool_id, pool_query, Method::kDecompress),
        tag_id_(tag_id), blob_name_(HSHM_MALLOC, blob_name),
        offset_(offset), size_(size), flags_(flags), blob_data_(blob_data),
        core_pool_id_(core_pool_id),
        output_size_(0), decompress_time_ms_(0.0) {}

  void Copy(const hipc::FullPtr<DecompressTask>& other) {
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

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_COMPRESSOR_TASKS_H_
