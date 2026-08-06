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
#include <clio_cte/compressor/compressor_runtime.h>

#include <clio_ctp/serialize/msgpack_wrapper.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <string>
#include <tuple>
#include <vector>

#include "clio_runtime/work_orchestrator.h"
#include "clio_runtime/worker.h"
#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/data_stats.h"
#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/gpu_api.h"
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZP
#include "clio_ctp/compress/cuszp.h"
#endif

namespace {
// cuSZp's presets only reach an absolute error bound of 1e-4 (BEST). To sweep
// tighter bounds (e.g. 1e-6 / 1e-9) set CLIO_CUSZP_EB=<float>; it overrides the
// preset's bound at COMPRESS time only. Decompress needs no override — the
// cuSZp wrapper records the bound used in each blob's prefix and reconstructs
// from that. No-op unless the codec is cuSZp and the env var is set.
inline void ApplyCuszpErrorBoundOverride(ctp::Compressor *c,
                                         const std::string &lib) {
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_CUSZP
  if (lib != "cuszp") return;
  const char *e = std::getenv("CLIO_CUSZP_EB");
  if (e == nullptr) return;
  if (auto *cz = dynamic_cast<ctp::Cuszp *>(c)) {
    float eb = std::strtof(e, nullptr);
    if (eb > 0.0f) {
      cz->SetErrorBound(eb);
      HLOG(kInfo, "[cuszp] absolute error bound overridden to {}", eb);
    }
  }
#else
  (void)c; (void)lib;
#endif
}
}  // namespace

namespace clio::cte::compressor {

// Bring chi namespace items into scope for CLIO_CUR_WORKER macro
using clio::run::chi_cur_worker_key_;
using clio::run::Worker;

/**
 * Routing for the compressor's forward (de)compress store to the next pool
 * (cte_core). By default the compressor forwards with PoolQuery::Local(), so a
 * compressed page is stored on the SAME node that compressed it -- fine on one
 * node, but it never distributes across a multi-node cluster.
 *
 * When CLIO_CTE_COMPRESS_DISTRIBUTE=1, forward instead by HASHING (tag_id,
 * blob_name) to a container -- the SAME scheme cte_core uses for blobs
 * (Runtime::HashBlobToContainer, core_runtime.cc), so compressed pages fan out
 * across the distributed cte_core and a later GetBlob for the same page routes
 * back to the node that holds it. On a single node DirectHash resolves to the
 * only node, so this is transparent there; it is opt-in only to keep existing
 * single-node behavior byte-for-byte unchanged.
 *
 * PutBlob and GetBlob MUST use identical routing for a given blob_name, so all
 * forward sites call this one helper.
 */

// The regular tasks carry a shm string (.str()); the POD tasks a fixed_string
// (.c_str()). One overload pair lets the handler bodies be shared verbatim.
static inline std::string CompressorBlobName(
    const clio::cte::core::PutBlobTask &t) { return t.blob_name_.str(); }
static inline std::string CompressorBlobName(
    const clio::cte::core::GetBlobTask &t) { return t.blob_name_.str(); }
static inline std::string CompressorBlobName(
    const clio::cte::core::PodPutBlobTask &t) { return t.blob_name_.c_str(); }
static inline std::string CompressorBlobName(
    const clio::cte::core::PodGetBlobTask &t) { return t.blob_name_.c_str(); }

inline clio::run::PoolQuery ForwardQuery(const clio::cte::core::TagId &tag_id,
                                         const std::string &blob_name) {
  static const bool distribute = [] {
    const char *e = std::getenv("CLIO_CTE_COMPRESS_DISTRIBUTE");
    return e && e[0] == '1';
  }();
  if (!distribute) return clio::run::PoolQuery::Local();
  // Mirror Runtime::HashBlobToContainer exactly (core_runtime.cc:4794).
  std::hash<std::string> string_hasher;
  std::hash<clio::run::u32> u32_hasher;
  clio::run::u32 hash_value = u32_hasher(tag_id.major_);
  hash_value ^= u32_hasher(tag_id.minor_) + 0x9e3779b9 + (hash_value << 6) +
                (hash_value >> 2);
  hash_value ^= static_cast<clio::run::u32>(string_hasher(blob_name)) +
                0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
  return clio::run::PoolQuery::DirectHash(hash_value);
}

/**
 * Environment-variable compressor pin.
 *
 * When `CLIO_CTE_COMPRESS_LIB` is set, it FORCES every compression performed by
 * this module to use one specific library, overriding both the dynamic
 * predictor (DynamicSchedule) and any caller-supplied `context.compress_lib_`.
 * This is the operator-facing knob for "pin a GPU compressor" — set e.g.
 * `CLIO_CTE_COMPRESS_LIB=nvcomp-lz4` (a canonical name from CompressionFactory)
 * or a raw wire ID integer. `CLIO_CTE_COMPRESS_PRESET` optionally pins the
 * preset (`fast` | `balanced` | `best`, or 1 | 2 | 3); default balanced.
 *
 * The env is read once and cached (the pin is a deployment-time decision, not a
 * per-request one). Returns the pinned wire ID, or -1 when no valid pin is set.
 * `out_preset` receives the pinned preset integer (1/2/3) when the return is >=0.
 */
static int CompressorPinWireId(int* out_preset) {
  // Cache: -2 = not yet parsed, -1 = no pin, >=0 = pinned wire id.
  static int cached_wire = -2;
  static int cached_preset = 2;  // BALANCED
  if (cached_wire == -2) {
    cached_wire = -1;
    const char* lib = std::getenv("CLIO_CTE_COMPRESS_LIB");
    if (lib != nullptr && lib[0] != '\0') {
      // Accept a canonical name or a raw integer wire id.
      int wire = ctp::CompressionFactory::WireIdForName(lib);
      if (wire < 0) {
        char* end = nullptr;
        long parsed = std::strtol(lib, &end, 10);
        if (end != lib && *end == '\0') wire = static_cast<int>(parsed);
      }
      if (wire >= 0) {
        cached_wire = wire;
        const char* pre = std::getenv("CLIO_CTE_COMPRESS_PRESET");
        if (pre != nullptr && pre[0] != '\0') {
          if (std::strcmp(pre, "fast") == 0 || std::strcmp(pre, "1") == 0) {
            cached_preset = 1;
          } else if (std::strcmp(pre, "best") == 0 ||
                     std::strcmp(pre, "3") == 0) {
            cached_preset = 3;
          } else {
            cached_preset = 2;  // balanced / default
          }
        }
        HLOG(kInfo,
             "Compressor pinned via CLIO_CTE_COMPRESS_LIB: {} (wire={}, "
             "preset={})",
             ctp::CompressionFactory::NameForWireId(cached_wire), cached_wire,
             cached_preset);
      } else {
        HLOG(kWarning,
             "CLIO_CTE_COMPRESS_LIB='{}' is not a known compressor name or "
             "wire id; ignoring pin",
             lib);
      }
    }
  }
  if (cached_wire >= 0 && out_preset != nullptr) *out_preset = cached_preset;
  return cached_wire;
}

/**
 * Compression header prepended to compressed data for self-describing format.
 * This allows decompression without external metadata.
 */
struct CompressionHeader {
  static constexpr uint32_t kMagic = 0x43544543;  // "CTEC" in ASCII
  uint32_t magic_;            // Magic number to identify compressed data
  uint32_t compress_lib_;     // Compression library ID
  uint32_t compress_preset_;  // Compression preset
  uint64_t original_size_;    // Original uncompressed size
  // Exact compressed payload size (bytes after this header). Required by
  // frame-exact codecs like zstd/lz4: ZSTD_decompress needs the precise
  // compressed length, and the reader over-allocates its fetch buffer (it
  // does not know the compressed size a priori), so the trailing bytes are
  // garbage. Passing that over-estimate as the input size makes zstd fail.
  uint64_t compressed_size_;

  CompressionHeader()
      : magic_(kMagic),
        compress_lib_(0),
        compress_preset_(0),
        original_size_(0),
        compressed_size_(0) {}

  CompressionHeader(uint32_t lib, uint32_t preset, uint64_t orig_size,
                    uint64_t comp_size = 0)
      : magic_(kMagic),
        compress_lib_(lib),
        compress_preset_(preset),
        original_size_(orig_size),
        compressed_size_(comp_size) {}

  bool IsValid() const { return magic_ == kMagic; }
};
static_assert(sizeof(CompressionHeader) == 32,
              "CompressionHeader must be 32 bytes (added compressed_size_)");

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Load configuration from compose YAML (or direct CreateParams)
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target

  // Initialize the core client using next_pool_id from compose
  if (!config_.next_pool_id_.IsNull()) {
    core_client_ = std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }

  // Initialize atomic counters
  compression_logical_time_ = 0;

  // tag_consumers_ is lazily populated by RegisterConsumer; nothing to
  // preallocate here. The map is empty when tracking_enabled_=false.

  // Seed previous CPU times so PollNodeLoad's first delta is well-defined.
  prev_cpu_times_ = ctp::SystemInfo::GetCpuTimes();

  // Load Q-table model if configured (primary prediction method)
  if (!config_.qtable_model_path_.empty()) {
    try {
      HLOG(kDebug, "Loading Q-table model from: {}",
           config_.qtable_model_path_);
      qtable_predictor_ = std::make_unique<QTablePredictor>();
      if (qtable_predictor_->Load(config_.qtable_model_path_)) {
        HLOG(kDebug, "Q-table model loaded successfully with {} states",
             qtable_predictor_->GetNumStates());
      } else {
        HLOG(kWarning, "Failed to load Q-table model from: {}",
             config_.qtable_model_path_);
        qtable_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading Q-table model: {}", e.what());
      qtable_predictor_.reset();
    }
  }

  // Load LinReg table model if configured
  if (!config_.linreg_model_path_.empty()) {
    try {
      HLOG(kDebug, "Loading LinReg table model from: {}",
           config_.linreg_model_path_);
      linreg_predictor_ = std::make_unique<LinRegTablePredictor>();
      if (linreg_predictor_->Load(config_.linreg_model_path_)) {
        HLOG(kDebug, "LinReg table model loaded successfully");
      } else {
        HLOG(kWarning, "Failed to load LinReg table model from: {}",
             config_.linreg_model_path_);
        linreg_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading LinReg table model: {}", e.what());
      linreg_predictor_.reset();
    }
  }

  // Load distribution classifier if configured
  if (!config_.distribution_model_path_.empty()) {
    // Note: DistributionClassifier is template-based - use
    // DistributionClassifierFactory::Classify() directly No model loading
    // needed - the factory uses built-in mathematical classification
    HLOG(kDebug,
         "Distribution classifier available via factory (no model loading "
         "required)");
  }

#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
  // Load DNN model weights as fallback if Q-table not available
  if (!qtable_predictor_ && !config_.dnn_model_weights_path_.empty()) {
    try {
      HLOG(kDebug, "Loading DNN model weights from: {}",
           config_.dnn_model_weights_path_);
      nn_predictor_ = std::make_unique<DenseNNPredictor>();
      if (nn_predictor_->LoadWeights(config_.dnn_model_weights_path_)) {
        HLOG(kDebug, "DNN model loaded successfully");
      } else {
        HLOG(kWarning, "Failed to load DNN model weights from: {}",
             config_.dnn_model_weights_path_);
        nn_predictor_.reset();
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Exception while loading DNN model: {}", e.what());
      nn_predictor_.reset();
    }
  }
#endif  // CLIO_COMPRESSOR_ENABLE_DENSE_NN

  if (!qtable_predictor_ && !linreg_predictor_) {
    HLOG(kDebug,
         "No compression predictor configured, dynamic compression prediction "
         "disabled");
  }

  HLOG(kDebug,
       "CTE Compressor container created and initialized for pool: {} (ID: {})",
       pool_name_, pool_id_);

  // Spawn the periodic consumer-poll task (5s period). It iterates this
  // container's consumer list and dispatches PollNodeLoad to each node.
  client_.AsyncPollConsumers(clio::run::PoolQuery::Local(), 5000000);

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    // Reset predictors
    qtable_predictor_.reset();
    linreg_predictor_.reset();
    // No distribution_classifier_ to reset

#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
    nn_predictor_.reset();
#endif

    // Clear compression telemetry log if allocated
    // ShmPtr cleanup handled automatically

    HLOG(kDebug, "CTE Compressor container destroyed successfully");
  } catch (const std::exception& e) {
    HLOG(kError, "Exception during compressor destroy: {}", e.what());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::PoolQuery Runtime::ScheduleTask(const clio::run::shared_ptr<clio::run::Task> &task) {
  // Compress placement: consult per-tag consumer tracking (when enabled)
  // so the compressed copy lands on the node that most recently read
  // the tag. Falls through to DirectHash(tag_id) when tracking is off
  // or the tag has no known consumers yet — keeps placement
  // deterministic per tag without the tracking overhead.
  if (task->method_ == Method::kCompress) {
    auto& compress_task = task.template Cast<CompressTask>();
    clio::run::u32 consumer_node = 0;
    if (PickConsumerForTag(compress_task->tag_id_, consumer_node)) {
      return clio::run::PoolQuery::Physical(consumer_node);
    }
    // No consumer info — hash on tag_id so all blobs of the same tag
    // converge on the same container regardless of which node submits.
    clio::run::u32 hash = static_cast<clio::run::u32>(
        std::hash<clio::cte::core::TagId>{}(compress_task->tag_id_));
    return clio::run::PoolQuery::DirectHash(hash);
  }
  // Other Dynamic methods (Decompress, periodic ticks) resolve Local.
  return clio::run::PoolQuery::Local();
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (!core_client_) {
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  // Poll target states
  try {
    auto list_task = core_client_->AsyncListTargets();
    CLIO_CO_AWAIT(list_task);
    if (list_task->GetReturnCode() == 0) {
      std::lock_guard<std::mutex> lock(target_states_mutex_);
      for (auto &target_name : list_task->target_names_) {
        auto stat_task = core_client_->AsyncGetTargetInfo(target_name);
        CLIO_CO_AWAIT(stat_task);
        if (stat_task->GetReturnCode() == 0) {
          auto &state = target_states_[target_name];
          state.target_name_ = target_name;
          state.target_score_ = stat_task->target_score_;
          state.remaining_space_ = stat_task->remaining_space_;
          state.bytes_written_ = stat_task->bytes_written_;
        }
      }
    }
    // Serialize target_states_ to msgpack
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    pk.pack_map(target_states_.size());
    for (auto &[name, state] : target_states_) {
      pk.pack(name);
      pk.pack_map(4);
      pk.pack("score"); pk.pack(state.target_score_);
      pk.pack("remaining"); pk.pack(state.remaining_space_);
      pk.pack("written"); pk.pack(state.bytes_written_);
      pk.pack("name"); pk.pack(state.target_name_);
    }
    task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  } catch (const std::exception &e) {
    HLOG(kError, "Compressor::Monitor failed: {}", e.what());
  }
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ==============================================================================
// Compression Statistics Estimation
// ==============================================================================

std::vector<CompressionStats> Runtime::EstCompressionStats(
    const void* chunk, clio::run::u64 chunk_size, const Context& context) {
  std::vector<CompressionStats> results;

  // Determine data type from context
  // context.data_type_: 0 = char/uint8, 1 = float
  ctp::DataType data_type = (context.data_type_ == 1) ? ctp::DataType::FLOAT32
                                                       : ctp::DataType::UINT8;
  size_t type_size = ctp::DataStatisticsFactory::GetTypeSize(data_type);

  // Calculate number of elements (sample up to 64KB for efficiency)
  clio::run::u64 sample_bytes = std::min(chunk_size, static_cast<clio::run::u64>(65536));
  size_t num_elements = sample_bytes / type_size;
  if (num_elements == 0) {
    num_elements = 1;
  }

  // Calculate compression features using DataStatisticsFactory
  double entropy = ctp::DataStatisticsFactory::CalculateShannonEntropy(
      chunk, num_elements, data_type);
  double mad =
      ctp::DataStatisticsFactory::CalculateMAD(chunk, num_elements, data_type);
  double second_derivative_mean =
      ctp::DataStatisticsFactory::CalculateSecondDerivative(
          chunk, num_elements, data_type);

  // Determine candidate compression libraries and configs
  // Library IDs: BROTLI=0, BZIP2=1, Blosc2=2, FPZIP=3, LZ4=4, LZMA=5,
  //              SNAPPY=6, SZ3=7, ZFP=8, ZLIB=9, ZSTD=10
  // Config IDs: balanced=0, best=1, default=2, fast=3
  std::vector<std::pair<int, int>> candidate_lib_configs;
  if (context.dynamic_compress_ == 1) {
    // Static mode: use specified library with default config
    candidate_lib_configs.push_back({context.compress_lib_, 2});
  } else {
    // Dynamic mode: test common library/config combinations
    candidate_lib_configs = {
        {10, 0},  // ZSTD balanced
        {10, 3},  // ZSTD fast
        {4, 3},   // LZ4 fast
        {1, 1},   // BZIP2 best
        {9, 0},   // ZLIB balanced
    };
  }

  // Run predictions for each candidate library/config
  for (const auto& [lib_id, config_id] : candidate_lib_configs) {
    CompressionPrediction pred;

    // Use Q-table predictor if available (primary method)
    if (qtable_predictor_ && qtable_predictor_->IsReady()) {
      CompressionFeatures features;
      features.library_config_id = static_cast<double>(lib_id);
      features.chunk_size_bytes = static_cast<double>(chunk_size);
      features.shannon_entropy = entropy;
      features.mad = mad;
      features.second_derivative_mean = second_derivative_mean;
      // Set config encoding
      features.config_fast = (config_id == 3) ? 1 : 0;
      features.config_balanced = (config_id == 0) ? 1 : 0;
      features.config_best = (config_id == 1) ? 1 : 0;
      // Set data type encoding
      features.data_type_char = (context.data_type_ == 0) ? 1 : 0;
      features.data_type_float = (context.data_type_ == 1) ? 1 : 0;

      pred = qtable_predictor_->Predict(features);
    }
#ifdef CLIO_COMPRESSOR_ENABLE_DENSE_NN
    // Fallback to DNN if Q-table not available
    else if (nn_predictor_ && nn_predictor_->IsReady()) {
      CompressionFeatures features;
      features.library_config_id = static_cast<double>(lib_id);
      features.chunk_size_bytes = static_cast<double>(chunk_size);
      features.shannon_entropy = entropy;
      features.mad = mad;
      features.second_derivative_mean = second_derivative_mean;
      features.config_fast = (config_id == 3) ? 1 : 0;
      features.config_balanced = (config_id == 0) ? 1 : 0;
      features.config_best = (config_id == 1) ? 1 : 0;
      features.data_type_char = (context.data_type_ == 0) ? 1 : 0;
      features.data_type_float = (context.data_type_ == 1) ? 1 : 0;
      pred = nn_predictor_->Predict(features);
    }
#endif  // CLIO_COMPRESSOR_ENABLE_DENSE_NN
    else {
      // Heuristic fallback if no predictor available
      pred.compression_ratio = 2.0;
      pred.psnr_db = 0.0;
      pred.compression_time_ms = static_cast<double>(chunk_size) / 100000.0;
    }

    // Filter out compressions below PSNR threshold
    if (context.target_psnr_ > 0 && pred.psnr_db > 0 &&
        pred.psnr_db < context.target_psnr_) {
      continue;
    }

    // Add to results with library and preset
    results.emplace_back(lib_id, config_id, pred.compression_ratio,
                         pred.compression_time_ms, pred.compression_time_ms,
                         pred.psnr_db);
  }

  return results;
}

double Runtime::EstWorkflowCompressTime(clio::run::u64 chunk_size, double tier_bw,
                                        const CompressionStats& stats,
                                        const Context& context) {
  double compressed_size = chunk_size / stats.compression_ratio_;
  double transfer_time_ms = (compressed_size / tier_bw) * 1000.0;

  if (stats.psnr_db_ == 0.0) {
    // Lossless compression
    return stats.compress_time_ms_ + stats.decompress_time_ms_ +
           transfer_time_ms;
  } else {
    // Lossy compression - may need verification decompression
    double psnr_check_prob = static_cast<double>(context.psnr_chance_) / 100.0;
    return stats.compress_time_ms_ +
           (1.0 + psnr_check_prob) * stats.decompress_time_ms_ +
           transfer_time_ms;
  }
}

std::tuple<int, int, int, double, float> Runtime::BestCompressRatio(
    const void* chunk, clio::run::u64 chunk_size, int container_id,
    const std::vector<CompressionStats>& stats, const Context& context) {
  int best_tier = 0;
  int best_lib = 0;
  int best_preset = 2;  // Default: BALANCED
  double best_time = std::numeric_limits<double>::max();
  double best_ratio = 1.0;
  float best_tier_score = 0.0F;

  // Get target bandwidth from cached target states
  double tier_bw = 1e9;  // Default: 1 GB/s
  {
    std::lock_guard<std::mutex> lock(target_states_mutex_);
    if (!target_states_.empty()) {
      // Find target with highest score (best performance)
      float max_score = 0.0F;
      for (const auto& [name, state] : target_states_) {
        if (state.target_score_ > max_score) {
          max_score = state.target_score_;
          best_tier_score = max_score;
          // Estimate bandwidth from normalized log score
          // score = log(bw+1) / log(1000+1), solve for bw
          tier_bw = std::pow(1001.0, max_score) - 1.0;
          tier_bw = std::max(tier_bw, 1e6);   // At least 1 MB/s
          tier_bw = std::min(tier_bw, 1e10);  // Cap at 10 GB/s
        }
      }
    }
  }

  for (const auto& stat : stats) {
    // Calculate workflow time for this compression
    double est_time =
        EstWorkflowCompressTime(chunk_size, tier_bw, stat, context);

    // Choose compression with best ratio that meets time constraints
    if (stat.compression_ratio_ > best_ratio) {
      best_ratio = stat.compression_ratio_;
      best_lib = stat.compress_lib_;
      best_preset = stat.compress_preset_;
      best_time = est_time;
      best_tier = 0;
    }
  }

  return std::make_tuple(best_tier, best_lib, best_preset, best_time,
                         best_tier_score);
}

std::tuple<int, int, int, double, float> Runtime::BestCompressTime(
    const void* chunk, clio::run::u64 chunk_size, int container_id,
    const std::vector<CompressionStats>& stats, const Context& context) {
  int best_tier = 0;
  int best_lib = 0;
  int best_preset = 2;  // Default: BALANCED
  double best_time = std::numeric_limits<double>::max();
  float best_tier_score = 0.0F;

  // Get target bandwidth from cached target states
  double tier_bw = 1e9;  // Default: 1 GB/s
  {
    std::lock_guard<std::mutex> lock(target_states_mutex_);
    if (!target_states_.empty()) {
      // Find target with highest score (best performance)
      float max_score = 0.0F;
      for (const auto& [name, state] : target_states_) {
        if (state.target_score_ > max_score) {
          max_score = state.target_score_;
          best_tier_score = max_score;
          // Estimate bandwidth from normalized log score
          // score = log(bw+1) / log(1000+1), solve for bw
          tier_bw = std::pow(1001.0, max_score) - 1.0;
          tier_bw = std::max(tier_bw, 1e6);   // At least 1 MB/s
          tier_bw = std::min(tier_bw, 1e10);  // Cap at 10 GB/s
        }
      }
    }
  }

  // For each compression library and tier, calculate workflow time
  for (const auto& stat : stats) {
    double est_time =
        EstWorkflowCompressTime(chunk_size, tier_bw, stat, context);

    // Choose combination with best performance
    if (est_time < best_time) {
      best_time = est_time;
      best_lib = stat.compress_lib_;
      best_preset = stat.compress_preset_;
      best_tier = 0;
    }
  }

  return std::make_tuple(best_tier, best_lib, best_preset, best_time,
                         best_tier_score);
}

std::tuple<int, int, int, double, float> Runtime::BestCompressForNode(
    const Context& context, const void* chunk, clio::run::u64 chunk_size,
    int container_id, const std::vector<CompressionStats>& stats) {
  // Choose strategy based on context objective
  if (context.max_performance_) {
    // Objective: minimize time
    return BestCompressTime(chunk, chunk_size, container_id, stats, context);
  }
  // Objective: maximize compression ratio
  return BestCompressRatio(chunk, chunk_size, container_id, stats, context);
}

// ==============================================================================
// Task Execution Methods
// ==============================================================================

// Static atomic trace key counter for generating unique trace IDs
static std::atomic<clio::run::u64> g_trace_key_counter{1};

// Helper function to write trace log entry
static void WriteTraceLog(const std::string& trace_folder,
                          const std::string& log_name, clio::run::u32 container_id,
                          const std::string& entry) {
  if (trace_folder.empty()) return;

  try {
    std::string log_path =
        trace_folder + "/" + log_name + "." + std::to_string(container_id);
    std::ofstream log_file(log_path, std::ios::app);
    if (log_file.is_open()) {
      log_file << entry << std::endl;
      log_file.close();
    }
  } catch (const std::exception& e) {
    HLOG(kWarning, "Failed to write trace log: {}", e.what());
  }
}


// Codec pin resolution: the COMPOSE FILE's compress_lib wins; the
// CLIO_CTE_COMPRESS_LIB env remains as an operator override-of-last-resort.
static int PinWireFor(const CompressorConfig &cfg, int *out_preset) {
  if (!cfg.compress_lib_.empty()) {
    int w = ctp::CompressionFactory::WireIdForName(cfg.compress_lib_);
    if (w >= 0) {
      *out_preset = cfg.compress_preset_;
      return w;
    }
  }
  return CompressorPinWireId(out_preset);
}

clio::run::TaskResume Runtime::DynamicSchedule(
    clio::run::shared_ptr<DynamicScheduleTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract task parameters (same as PutBlobTask)
    clio::run::u64 chunk_size = task->size_;
    // Convert ShmPtr to raw pointer via FullPtr
    auto blob_fullptr =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    void* chunk_data = blob_fullptr.ptr_;
    Context& context = task->context_;

    // Initialize tracing if enabled
    auto start_time = std::chrono::high_resolution_clock::now();
    if (context.trace_) {
      context.trace_key_ = g_trace_key_counter.fetch_add(1);
      context.trace_node_ = static_cast<int>(CLIO_IPC->GetNodeId());
    }

    // Check if we have valid chunk data
    if (chunk_data == nullptr || chunk_size == 0) {
      HLOG(kWarning, "Invalid chunk data for dynamic scheduling");
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Operator pin: if CLIO_CTE_COMPRESS_LIB is set, bypass the predictor
    // entirely and compress with the pinned library. Keeps the dynamic path
    // deterministic under a pin (and avoids paying for stat estimation).
    {
      int pin_preset = 2;
      int pin_wire = PinWireFor(config_, &pin_preset);
      if (pin_wire >= 0) {
        context.compress_lib_ = pin_wire;
        context.compress_preset_ = pin_preset;
        auto compress_task = client_.AsyncCompress(
            clio::run::PoolQuery::Local(), task->tag_id_,
            task->blob_name_.str(), task->offset_, task->size_,
            task->blob_data_, task->score_, context, task->flags_,
            task->core_pool_id_);
        CLIO_CO_AWAIT(compress_task);
        task->context_ = compress_task->context_;
        task->tier_score_ = compress_task->tier_score_;
        task->return_code_ = compress_task->return_code_;
        CLIO_CO_RETURN;
      }
    }

    // Get compression stats
    auto stats = EstCompressionStats(chunk_data, chunk_size, context);

    if (stats.empty()) {
      // No valid compression available, disable compression
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    // Log predicted compression stats if tracing enabled
    if (context.trace_ && !stats.empty()) {
      for (const auto& stat : stats) {
        std::ostringstream log_entry;
        log_entry << context.trace_key_ << "," << stat.compress_lib_ << ","
                  << stat.compression_ratio_ << "," << stat.compress_time_ms_
                  << "," << stat.decompress_time_ms_ << "," << stat.psnr_db_;
        WriteTraceLog(config_.trace_folder_path_, "predicted_stats.log",
                      pool_id_.major_, log_entry.str());
      }
    }

    // Choose best compression strategy
    auto [best_tier, best_lib, best_preset, best_time, tier_score] =
        BestCompressForNode(context, chunk_data, chunk_size, container_id_,
                            stats);

    // Update context with selected compression library and preset
    context.compress_lib_ = best_lib;
    context.compress_preset_ = best_preset;
    task->tier_score_ = tier_score;

    // Log scheduling decision time if tracing enabled
    if (context.trace_) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration_ms =
          std::chrono::duration<double, std::milli>(end_time - start_time)
              .count();

      std::ostringstream log_entry;
      log_entry << context.trace_key_ << "," << duration_ms;
      WriteTraceLog(config_.trace_folder_path_, "sched_decision.log",
                    pool_id_.major_, log_entry.str());
    }

    // Now call Compress to perform compression and PutBlob
    auto compress_task = client_.AsyncCompress(
        clio::run::PoolQuery::Local(), task->tag_id_, task->blob_name_.str(),
        task->offset_, task->size_, task->blob_data_, task->score_, context,
        task->flags_, task->core_pool_id_);
    CLIO_CO_AWAIT(compress_task);

    // Copy results back
    task->context_ = compress_task->context_;
    task->tier_score_ = compress_task->tier_score_;
    task->return_code_ = compress_task->return_code_;

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in DynamicSchedule: {}", e.what());
    task->return_code_ = 1;
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Compress(clio::run::shared_ptr<CompressTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract task parameters (same as PutBlobTask)
    clio::run::u64 input_size = task->size_;
    Context& context = task->context_;

    // Validate inputs
    if (task->blob_data_.IsNull() || input_size == 0) {
      task->return_code_ = 1;  // Invalid input
      CLIO_CO_RETURN;
    }

    // Operator pin: CLIO_CTE_COMPRESS_LIB forces a specific compressor here,
    // overriding both the dynamic predictor and the caller's compress_lib_.
    // Applied at the single chokepoint where wire id -> factory lib happens, so
    // it governs every compression path (static, dynamic, explicit).
    {
      int pin_preset = 2;
      int pin_wire = PinWireFor(config_, &pin_preset);
      if (pin_wire >= 0) {
        context.compress_lib_ = pin_wire;
        context.compress_preset_ = pin_preset;
      }
    }

    // Initialize core client if needed (from compose next_pool_id or task param)
    if (!core_client_) {
      clio::run::PoolId core_id = !config_.next_pool_id_.IsNull()
          ? config_.next_pool_id_ : task->core_pool_id_;
      if (!core_id.IsNull()) {
        core_client_ = std::make_unique<clio::cte::core::Client>(core_id);
      }
    }

    // Get tier score for output
    float tier_score = 0.0F;
    {
      std::lock_guard<std::mutex> lock(target_states_mutex_);
      for (const auto& [name, state] : target_states_) {
        if (state.target_score_ > tier_score) {
          tier_score = state.target_score_;
        }
      }
    }
    task->tier_score_ = tier_score;

    // If no compression requested, just call PutBlob directly
    if (context.compress_lib_ <= 0) {
      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
          task->blob_data_, task->score_, context, task->flags_,
          ForwardQuery(task->tag_id_, task->blob_name_.str()));
      CLIO_CO_AWAIT(put_task);
      task->context_ = put_task->context_;
      task->return_code_ = put_task->return_code_;
      CLIO_CO_RETURN;
    }

    // Map the wire ID (CompressionHeader.compress_lib_) to a library name via
    // the shared registry in CompressionFactory (single source of truth; out-of-
    // range falls back to "zstd"). Note the wire ID is a separate namespace from
    // GetLibraryId's ML scheme (base_id*10 + preset, e.g. nvcomp-lz4 = 132).
    std::string library_name =
        ctp::CompressionFactory::NameForWireId(context.compress_lib_);

    // Map preset integer to enum
    ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
    if (context.compress_preset_ == 1) {
      preset = ctp::CompressionPreset::FAST;
    } else if (context.compress_preset_ == 3) {
      preset = ctp::CompressionPreset::BEST;
    }

    // Create compressor with specified preset
    auto compressor = ctp::CompressionFactory::GetPreset(library_name, preset);

    if (!compressor) {
      HLOG(kWarning, "Failed to create compressor for library: {}",
           library_name);
      task->return_code_ = 3;  // Compressor creation failed
      CLIO_CO_RETURN;
    }
    ApplyCuszpErrorBoundOverride(compressor.get(), library_name);

    auto compress_start = std::chrono::high_resolution_clock::now();

    // Allocate buffer for compressed data (worst case: original size + 5%
    // overhead)
    std::vector<char> compressed_buffer(input_size + (input_size / 20) + 1024);

    // Compress the data
    size_t compressed_size = compressed_buffer.size();
    // Convert ShmPtr to raw pointer via FullPtr
    auto input_fullptr =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    char* input_ptr = input_fullptr.ptr_;
    bool success = compressor->Compress(compressed_buffer.data(),
                                        compressed_size, input_ptr, input_size);

    auto compress_end = std::chrono::high_resolution_clock::now();
    double compress_time =
        std::chrono::duration<double, std::milli>(compress_end - compress_start)
            .count();

    // Check if compression succeeded and is beneficial
    // Include header size in the total stored size
    size_t header_size = sizeof(CompressionHeader);
    size_t total_stored_size = compressed_size + header_size;

    if (success && total_stored_size < input_size) {
      // Compression succeeded and reduced size (including header overhead)
      compressed_buffer.resize(compressed_size);

      // Update context with compression statistics
      context.actual_original_size_ = input_size;
      context.actual_compressed_size_ = total_stored_size;
      context.actual_compression_ratio_ =
          static_cast<double>(input_size) /
          static_cast<double>(total_stored_size);
      context.actual_compress_time_ms_ = compress_time;

      // Allocate shared memory for header + compressed data
      auto compressed_shm = CLIO_IPC->AllocateBuffer(total_stored_size);
      if (compressed_shm.IsNull()) {
        HLOG(kError, "Failed to allocate shared memory for compressed data");
        task->return_code_ = 4;  // Memory allocation failed
        CLIO_CO_RETURN;
      }

      // Write compression header
      CompressionHeader header(context.compress_lib_, context.compress_preset_,
                               input_size, compressed_size);
      std::memcpy(compressed_shm.ptr_, &header, header_size);

      // Write compressed data after header
      std::memcpy(compressed_shm.ptr_ + header_size, compressed_buffer.data(),
                  compressed_size);

      // Tell the runtime these bytes are no longer the caller's bytes, so it
      // can mark the blob authoritatively (issue #818). This is the ONLY place
      // that knows it for certain -- compress_lib_ is set on the not-beneficial
      // path below too, where the stored bytes are raw.
      context.transform_flags_ |= clio::cte::core::kBlobTransformed |
                                  clio::cte::core::kBlobTransformCompressed;

      // Call PutBlob with header + compressed data
      ctp::ipc::ShmPtr<> compressed_shm_ptr =
          compressed_shm.shm_.template Cast<void>();
      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_,
          total_stored_size, compressed_shm_ptr, task->score_, context,
          task->flags_, ForwardQuery(task->tag_id_, task->blob_name_.str()));
      CLIO_CO_AWAIT(put_task);

      // Free compressed data buffer
      CLIO_IPC->FreeBuffer(compressed_shm);

      // Log compression telemetry
      CompressionTelemetry telemetry(
          CteOp::kPutBlob, context.compress_lib_, input_size, total_stored_size,
          compress_time, 0.0, 0.0, std::chrono::steady_clock::now(),
          compression_logical_time_.fetch_add(1));
      LogCompressionTelemetry(telemetry);

      HLOG(kDebug,
           "Compression: {} bytes -> {} bytes (ratio: {:.2f}, time: {:.2f}ms)",
           input_size, total_stored_size,
           static_cast<double>(input_size) /
               static_cast<double>(total_stored_size),
           compress_time);

      task->context_ = context;
      task->return_code_ = put_task->return_code_;
    } else {
      // Compression failed or didn't reduce size - store original data
      HLOG(kDebug, "Compression not beneficial, storing original data");

      // Mark uncompressed BEFORE the put, not after. The bytes below are the
      // caller's original bytes, so the blob must not be recorded as carrying
      // the codec we merely attempted -- this used to be zeroed only after
      // PutBlob had already copied compress_lib_ onto the BlobInfo, which is
      // half of why compress_lib_ cannot be trusted as a transform signal
      // (issue #818). transform_flags_ is deliberately left unset here.
      context.compress_lib_ = 0;

      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
          task->blob_data_, task->score_, context, task->flags_,
          ForwardQuery(task->tag_id_, task->blob_name_.str()));
      CLIO_CO_AWAIT(put_task);

      task->context_ = put_task->context_;
      task->return_code_ = put_task->return_code_;
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in Compress: {}", e.what());
    task->return_code_ = 6;  // Exception occurred
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Decompress(clio::run::shared_ptr<DecompressTask> &task) {
  CLIO_TASK_BODY_BEGIN
  try {
    // Record the originating node (the consumer that issued this Decompress)
    // against this specific tag. pool_query_.ret_node_ was stamped by the
    // sender's IpcManager when the task was first resolved, so it carries
    // the original sender's node id even after a network hop. Per-tag
    // tracking lets ScheduleTask later route Compress for the same tag
    // toward this reader. No-op when tracking_enabled_=false.
    RegisterConsumer(task->tag_id_, task->pool_query_.GetReturnNode());

    // Extract task parameters (same as GetBlobTask)
    clio::run::u64 expected_size = task->size_;

    // Validate output buffer
    if (task->blob_data_.IsNull()) {
      task->return_code_ = 1;  // Invalid output buffer
      CLIO_CO_RETURN;
    }

    // Initialize core client if needed (from compose next_pool_id or task param)
    if (!core_client_) {
      clio::run::PoolId core_id = !config_.next_pool_id_.IsNull()
          ? config_.next_pool_id_ : task->core_pool_id_;
      if (!core_id.IsNull()) {
        core_client_ = std::make_unique<clio::cte::core::Client>(core_id);
      }
    }

    // Allocate temporary buffer to receive compressed data from GetBlob
    // We don't know the compressed size, so allocate expected_size as upper
    // bound
    auto temp_buffer = CLIO_IPC->AllocateBuffer(expected_size);
    if (temp_buffer.IsNull()) {
      task->return_code_ = 2;  // Memory allocation failed
      CLIO_CO_RETURN;
    }
    ctp::ipc::ShmPtr<> temp_buffer_ptr = temp_buffer.shm_.template Cast<void>();

    // Call GetBlob to retrieve the (potentially compressed) data. MUST use the
    // same routing as the PutBlob above so the read reaches the node that holds
    // this page in a distributed cte_core.
    auto get_task = core_client_->AsyncGetBlob(
        task->tag_id_, task->blob_name_.str(), task->offset_, expected_size,
        task->flags_, temp_buffer_ptr,
        ForwardQuery(task->tag_id_, task->blob_name_.str()));
    CLIO_CO_AWAIT(get_task);

    if (get_task->return_code_ != 0) {
      CLIO_IPC->FreeBuffer(temp_buffer);
      task->return_code_ = 10 + get_task->return_code_;  // GetBlob failed
      CLIO_CO_RETURN;
    }

    // Check for compression header
    auto* header = reinterpret_cast<CompressionHeader*>(temp_buffer.ptr_);
    size_t header_size = sizeof(CompressionHeader);

    if (header->IsValid()) {
      // Data is compressed - decompress it
      int compress_lib = static_cast<int>(header->compress_lib_);
      int compress_preset = static_cast<int>(header->compress_preset_);
      clio::run::u64 original_size = header->original_size_;

      // Map the wire ID to a library name via the shared registry (single
      // source of truth; out-of-range falls back to "zstd"). Wire ID is a
      // separate namespace from the factory's ML scheme (base_id*10+preset).
      std::string library_name =
          ctp::CompressionFactory::NameForWireId(compress_lib);

      // Map preset integer to enum
      ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
      if (compress_preset == 1) {
        preset = ctp::CompressionPreset::FAST;
      } else if (compress_preset == 3) {
        preset = ctp::CompressionPreset::BEST;
      }

      // Create decompressor
      auto decompressor =
          ctp::CompressionFactory::GetPreset(library_name, preset);
      if (!decompressor) {
        CLIO_IPC->FreeBuffer(temp_buffer);
        HLOG(kWarning, "Failed to create decompressor for library: {}",
             library_name);
        task->return_code_ = 3;  // Decompressor creation failed
        CLIO_CO_RETURN;
      }

      auto decompress_start = std::chrono::high_resolution_clock::now();

      // Get compressed data (after header). Prefer the exact compressed length
      // recorded at Put time (frame-exact codecs like zstd need it); fall back
      // to the over-estimate for legacy blobs written without the field.
      char* compressed_data = temp_buffer.ptr_ + header_size;
      size_t compressed_size = header->compressed_size_ != 0
                                   ? static_cast<size_t>(header->compressed_size_)
                                   : expected_size - header_size;

      // Decompress to output buffer
      auto output_fullptr =
          CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
      size_t decompressed_size = original_size;
      bool success =
          decompressor->Decompress(output_fullptr.ptr_, decompressed_size,
                                   compressed_data, compressed_size);

      auto decompress_end = std::chrono::high_resolution_clock::now();
      double decompress_time = std::chrono::duration<double, std::milli>(
                                   decompress_end - decompress_start)
                                   .count();

      CLIO_IPC->FreeBuffer(temp_buffer);

      if (success) {
        task->output_size_ = decompressed_size;
        task->decompress_time_ms_ = decompress_time;

        // Log decompression telemetry
        CompressionTelemetry telemetry(
            CteOp::kGetBlob, compress_lib, decompressed_size, compressed_size,
            0.0, decompress_time, 0.0, std::chrono::steady_clock::now(),
            compression_logical_time_.fetch_add(1));
        LogCompressionTelemetry(telemetry);

        HLOG(kDebug, "Decompression: {} bytes -> {} bytes (time: {:.2f}ms)",
             compressed_size, decompressed_size, decompress_time);

        task->return_code_ = 0;  // Success
      } else {
        HLOG(kError, "Decompression failed");
        task->output_size_ = 0;
        task->decompress_time_ms_ = 0.0;
        task->return_code_ = 5;  // Decompression failed
      }
    } else {
      // No compression header - data is uncompressed
      // Copy directly to output buffer
      auto output_fullptr =
          CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
      std::memcpy(output_fullptr.ptr_, temp_buffer.ptr_, expected_size);
      CLIO_IPC->FreeBuffer(temp_buffer);

      task->output_size_ = expected_size;
      task->decompress_time_ms_ = 0.0;
      task->return_code_ = 0;  // Success (no decompression needed)

      HLOG(kDebug, "GetBlob (no compression): {} bytes", expected_size);
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception in Decompress: {}", e.what());
    task->return_code_ = 6;  // Exception occurred
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// Compress a RAW core PutBlobTask (method kPutBlob=15) that landed at the
// compressor entrypoint -- e.g. a gpu_vector page eviction from device code.
// blob_data_ is a GPU/HBM pointer; cuSZp compresses it in place. The compressed
// bytes (with a CompressionHeader) are forwarded to the core pool under the same
// per-page name the core handler would have composed ("<name>_pi<gpu_page_idx_>").
template <typename PutT>
clio::run::TaskResume Runtime::CompressPutBlobImpl(
    clio::run::shared_ptr<PutT>& task) {
  try {
    clio::run::u64 input_size = task->size_;
    clio::cte::core::Context& context = task->context_;
    if (task->blob_data_.IsNull() || input_size == 0) {
      task->return_code_ = 1; CLIO_CO_RETURN;
    }
    { int pin_preset = 2; int pin_wire = PinWireFor(config_, &pin_preset);
      if (pin_wire >= 0) { context.compress_lib_ = pin_wire;
                           context.compress_preset_ = pin_preset; } }
    if (!core_client_ && !config_.next_pool_id_.IsNull())
      core_client_ = std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
    if (!core_client_) { task->return_code_ = 9; CLIO_CO_RETURN; }

    // Per-page name: what the core PutBlob handler composes from gpu_page_idx_.
    std::string name = CompressorBlobName(*task);
    if (task->gpu_family_idx_ != PutT::kNoFamilyIdx) {
      name += "_b" + std::to_string(task->gpu_family_idx_);
    }
    if (task->gpu_page_idx_ != PutT::kNoPageIdx)
      name += "_pi" + std::to_string(task->gpu_page_idx_);

    if (context.compress_lib_ <= 0) {  // no compression -> forward as-is
      auto pt = core_client_->AsyncPutBlob(task->tag_id_, name, task->offset_,
          input_size, task->blob_data_, task->score_, context, task->flags_,
          ForwardQuery(task->tag_id_, name));
      CLIO_CO_AWAIT(pt);
      task->return_code_ = pt->return_code_; CLIO_CO_RETURN;
    }

    std::string lib = ctp::CompressionFactory::NameForWireId(context.compress_lib_);
    ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
    if (context.compress_preset_ == 1) preset = ctp::CompressionPreset::FAST;
    else if (context.compress_preset_ == 3) preset = ctp::CompressionPreset::BEST;
    auto compressor = ctp::CompressionFactory::GetPreset(lib, preset);
    if (!compressor) { task->return_code_ = 3; CLIO_CO_RETURN; }
    ApplyCuszpErrorBoundOverride(compressor.get(), lib);

    std::vector<char> cbuf(input_size + (input_size / 20) + 1024);
    size_t csize = cbuf.size();
    auto in = CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    bool ok = compressor->Compress(cbuf.data(), csize, in.ptr_, input_size);
    size_t hsz = sizeof(CompressionHeader);
    size_t total = csize + hsz;

    HLOG(kInfo, "[CompressPutBlob] name={} off={} in={} csize={} ok={}",
         name, task->offset_, input_size, csize, ok);
    if (ok && total < input_size) {
      context.actual_original_size_ = input_size;
      context.actual_compressed_size_ = total;
      context.actual_compression_ratio_ = (double)input_size / (double)total;
      auto shm = CLIO_IPC->AllocateBuffer(total);
      if (shm.IsNull()) { task->return_code_ = 4; CLIO_CO_RETURN; }
      CompressionHeader hdr(context.compress_lib_, context.compress_preset_,
                            input_size, csize);
      std::memcpy(shm.ptr_, &hdr, hsz);
      std::memcpy(shm.ptr_ + hsz, cbuf.data(), csize);
      ctp::ipc::ShmPtr<> sp = shm.shm_.template Cast<void>();
      auto pt = core_client_->AsyncPutBlob(task->tag_id_, name, task->offset_,
          total, sp, task->score_, context, task->flags_,
          ForwardQuery(task->tag_id_, name));
      CLIO_CO_AWAIT(pt);
      CLIO_IPC->FreeBuffer(shm);
      task->context_ = context;
      task->return_code_ = pt->return_code_;
    } else {  // not beneficial -> store original (device blob) uncompressed
      auto pt = core_client_->AsyncPutBlob(task->tag_id_, name, task->offset_,
          input_size, task->blob_data_, task->score_, context, task->flags_,
          ForwardQuery(task->tag_id_, name));
      CLIO_CO_AWAIT(pt);
      task->return_code_ = pt->return_code_;
    }
  } catch (const std::exception& e) {
    HLOG(kError, "Exception in CompressPutBlob: {}", e.what());
    task->return_code_ = 6;
  }
  CLIO_CO_RETURN;
}

// Decompress a RAW core GetBlobTask (method kGetBlob=16) -- a gpu_vector page
// fault. Fetches the (compressed) blob from core under the "_pi" name and
// decompresses into the caller's HBM buffer. Mirrors Decompress(); the
// compressed size is over-estimated (cuSZp's stream is self-delimiting).
template <typename GetT>
clio::run::TaskResume Runtime::DecompressGetBlobImpl(
    clio::run::shared_ptr<GetT>& task) {
  try {
    clio::run::u64 expected_size = task->size_;
    if (task->blob_data_.IsNull()) { task->return_code_ = 1; CLIO_CO_RETURN; }
    if (!core_client_ && !config_.next_pool_id_.IsNull())
      core_client_ = std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
    if (!core_client_) { task->return_code_ = 9; CLIO_CO_RETURN; }

    std::string name = CompressorBlobName(*task);
    if (task->gpu_family_idx_ != GetT::kNoFamilyIdx) {
      name += "_b" + std::to_string(task->gpu_family_idx_);
    }
    if (task->gpu_page_idx_ != GetT::kNoPageIdx)
      name += "_pi" + std::to_string(task->gpu_page_idx_);

    size_t hsz = sizeof(CompressionHeader);
    const size_t fetch_size = expected_size + hsz + 1024;
    auto out = CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    const bool dev_out = ctp::IsDevicePointer(out.ptr_);

    // Fetch buffer for the compressed bytes. On the device-output fault path
    // allocate it in DEVICE memory: with an HBM bdev the core's read is then
    // D2D, and a GPU codec (nvcomp) decompresses device->device — the page
    // never touches host RAM. Falls back to a host SHM buffer when no GPU (or
    // VRAM) is available; every consumer below is pointer-type aware.
    char *dev_tmp = nullptr;
    ctp::ipc::FullPtr<char> tmp;
#if CTP_ENABLE_GPU
    // Device fetch buffers ONLY for GPU codecs. Two reasons:
    //  - for a host codec (zstd/lz4) the compressed frame is pulled D2H for
    //    the CPU decompressor anyway, so a device fetch buys nothing;
    //  - the per-fault GpuApi::Malloc here deadlocked the paged-inference
    //    fault path: under VRAM pressure cudaMalloc implicitly synchronizes
    //    the context, which can never complete while the faulting kernel is
    //    spin-waiting on THIS task. Host fetch + fresh-stream device write is
    //    the same pattern the uncompressed mem-bdev read path uses, which
    //    services spinning kernels correctly today.
    const bool pinned_gpu_codec =
        config_.compress_lib_.rfind("nvcomp", 0) == 0 ||
        config_.compress_lib_ == "cusz" || config_.compress_lib_ == "cuszp" ||
        config_.compress_lib_ == "zfp-sycl";
    if (dev_out && pinned_gpu_codec) {
      dev_tmp = ctp::GpuApi::Malloc<char>(fetch_size);
    }
#endif
    if (dev_tmp == nullptr) {
      tmp = CLIO_IPC->AllocateBuffer(fetch_size);
      if (tmp.IsNull()) { task->return_code_ = 2; CLIO_CO_RETURN; }
    }
    auto free_tmp = [&]() {
#if CTP_ENABLE_GPU
      if (dev_tmp != nullptr) { ctp::GpuApi::Free(dev_tmp); dev_tmp = nullptr; return; }
#endif
      CLIO_IPC->FreeBuffer(tmp);
    };
    ctp::ipc::ShmPtr<> tmpp;
    if (dev_tmp != nullptr) {
      tmpp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();  // raw device pointer
      tmpp.off_ = reinterpret_cast<clio::run::u64>(dev_tmp);
    } else {
      tmpp = tmp.shm_.template Cast<void>();
    }
    auto gt = core_client_->AsyncGetBlob(task->tag_id_, name, task->offset_,
        fetch_size, task->flags_, tmpp,
        ForwardQuery(task->tag_id_, name));
    CLIO_CO_AWAIT(gt);
    if (gt->return_code_ != 0) { free_tmp();
      task->return_code_ = 10 + gt->return_code_; CLIO_CO_RETURN; }

    // The header must be readable on the host; for a device fetch buffer pull
    // just the 32-byte header across.
    char *fetch_base = dev_tmp != nullptr ? dev_tmp : tmp.ptr_;
    CompressionHeader header_host;
#if CTP_ENABLE_GPU
    if (dev_tmp != nullptr) {
      void *hstream = ctp::GpuApi::CreateStream();
      ctp::GpuApi::MemcpyAsync(reinterpret_cast<char*>(&header_host), dev_tmp,
                               hsz, hstream);
      ctp::GpuApi::Synchronize(hstream);
      ctp::GpuApi::DestroyStream(hstream);
    } else
#endif
    {
      std::memcpy(&header_host, fetch_base, hsz);
    }
    auto* header = &header_host;
    HLOG(kInfo, "[DecompressGetBlob] name={} off={} req={} get_rc={} valid={}",
         name, task->offset_, expected_size, gt->return_code_,
         header->IsValid());
    // Copy `n` bytes into `dst`, which may be an HBM slot on the on-device fault
    // path. A raw cudaMemcpy on a thread_local stream (ctp::DeviceAwareMemcpy)
    // fails here with "invalid argument": this Decompress runs as a coroutine and
    // can resume on a different worker thread than the one that lazily created
    // that stream, so the stream belongs to a foreign CUDA context. Mirror the
    // proven mem-bdev device path (MemBdevTransport::LaunchReadBlocksGpu): create
    // a FRESH stream in the current context, copy on it, and synchronize -- no
    // co_await between create and sync, so no thread migration.
    auto write_out = [](char *dst, const char *src, size_t n) {
      if (ctp::IsDevicePointer(dst)) {
        // Host codec output must be pushed into the HBM slot with a device
        // copy. Mirror the mem-bdev GPU write path
        // (MemBdevTransport::LaunchReadBlocksGpu): a fresh stream in the current
        // context, copy, synchronize -- no co_await between, so this coroutine
        // stays on one thread and the stream/context stay consistent.
        void *stream = ctp::GpuApi::CreateStream();
        ctp::GpuApi::MemcpyAsync(dst, src, n, stream);
        ctp::GpuApi::Synchronize(stream);
        ctp::GpuApi::DestroyStream(stream);
      } else {
        std::memcpy(dst, src, n);
      }
    };
    if (header->IsValid()) {
      std::string lib = ctp::CompressionFactory::NameForWireId(header->compress_lib_);
      ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
      if (header->compress_preset_ == 1) preset = ctp::CompressionPreset::FAST;
      else if (header->compress_preset_ == 3) preset = ctp::CompressionPreset::BEST;
      auto dec = ctp::CompressionFactory::GetPreset(lib, preset);
      if (!dec) { free_tmp(); task->return_code_ = 3; CLIO_CO_RETURN; }
      char* cdata = fetch_base + hsz;
      // Use the EXACT compressed size recorded at Put time. cuSZp is self-
      // delimiting and tolerates an over-estimate, but zstd/lz4 require the
      // precise frame length or ZSTD_decompress fails ("Src size incorrect")
      // on the trailing garbage in the over-allocated fetch buffer. Fall back
      // to the (legacy) over-estimate only for blobs written before this field
      // existed (compressed_size_ == 0).
      size_t csize = header->compressed_size_ != 0
                         ? static_cast<size_t>(header->compressed_size_)
                         : expected_size;
      size_t dsize = header->original_size_;
      // On the on-device fault path task->blob_data_ is an HBM slot, so out.ptr_
      // is a DEVICE pointer. A GPU codec (nvcomp*/cusz) writes device output
      // zero-copy — hand it the slot directly so the decompress runs entirely
      // on the GPU (with an HBM bdev the compressed input is device memory too:
      // no byte of the page ever crosses PCIe). A HOST codec (zstd/lz4)
      // decompresses on the CPU and cannot write device memory — stage through
      // a host buffer and device-aware-copy it across. Host destinations keep
      // the original zero-copy path.
      const bool gpu_codec = lib.rfind("nvcomp", 0) == 0 || lib == "cusz" ||
                             lib == "cuszp" || lib == "zfp-sycl";
      // A host codec cannot read a device fetch buffer: pull the compressed
      // frame D2H first. (Only happens when blobs written with a host codec
      // are read back on the device fault path — mixed-codec stores.)
      std::vector<char> chost;
      if (dev_tmp != nullptr && !gpu_codec) {
        chost.resize(csize);
#if CTP_ENABLE_GPU
        void *cstream = ctp::GpuApi::CreateStream();
        ctp::GpuApi::MemcpyAsync(chost.data(), cdata, csize, cstream);
        ctp::GpuApi::Synchronize(cstream);
        ctp::GpuApi::DestroyStream(cstream);
#endif
        cdata = chost.data();
      }
      bool ok;
      if (ctp::IsDevicePointer(out.ptr_) && gpu_codec) {
        ok = dec->Decompress(out.ptr_, dsize, cdata, csize);
      } else if (ctp::IsDevicePointer(out.ptr_)) {
        std::vector<char> staging(dsize);
        ok = dec->Decompress(staging.data(), dsize, cdata, csize);
        if (ok) write_out(out.ptr_, staging.data(), dsize);
        // DEBUG(first few): prove the bytes actually landed at out.ptr_.
        } else {
        ok = dec->Decompress(out.ptr_, dsize, cdata, csize);
      }
      free_tmp();
      task->return_code_ = ok ? 0 : 5;
    } else {  // stored uncompressed -> copy through (out may be an HBM slot)
      write_out(out.ptr_, fetch_base, expected_size);
      free_tmp();
      task->return_code_ = 0;
    }
  } catch (const std::exception& e) {
    HLOG(kError, "Exception in DecompressGetBlob: {}", e.what());
    task->return_code_ = 6;
  }
  CLIO_CO_RETURN;
}

void Runtime::LogCompressionTelemetry(const CompressionTelemetry& telemetry) {
  // Log to compression telemetry buffer if available
  if (!compression_telemetry_log_.IsNull()) {
    // TODO: Fix ShmPtr API for telemetry logging
    // compression_telemetry_log_->Push(telemetry);
  }

  // Log to trace file if tracing is enabled
  if (!config_.trace_folder_path_.empty()) {
    std::ostringstream log_entry;
    log_entry << telemetry.logical_time_ << "," << telemetry.compress_lib_
              << "," << telemetry.original_size_ << ","
              << telemetry.compressed_size_ << ","
              << telemetry.compress_time_ms_ << ","
              << telemetry.decompress_time_ms_ << "," << telemetry.psnr_db_;

    std::string log_name = (telemetry.op_ == CteOp::kPutBlob)
                               ? "compress_stats.log"
                               : "decompress_stats.log";
    WriteTraceLog(config_.trace_folder_path_, log_name, pool_id_.major_,
                  log_entry.str());
  }
}

clio::run::u64 Runtime::GetWorkRemaining() const {
  // Return 0 - compressor has no persistent work queue
  return 0;
}

// ==============================================================================
// Consumer Tracking
// ==============================================================================

void Runtime::RegisterConsumer(const clio::cte::core::TagId &tag_id,
                               clio::run::u32 node_id) {
  // Tracking knob: when off, no per-tag bookkeeping happens and
  // ScheduleTask falls through to DirectHash on the tag_id. Use this to
  // measure the overhead of the tracking mechanism itself, or for
  // workloads with no producer-consumer locality.
  if (!config_.tracking_enabled_) {
    return;
  }

  // Fast path: lookup under reader lock. The per-tag vector grows only
  // (entries are never removed), so a stale read at worst sends one
  // duplicate registration through the writer path — which the writer
  // re-check absorbs.
  {
    clio::run::ScopedCoRwReadLock read_lock(tag_consumers_lock_);
    auto it = tag_consumers_.find(tag_id);
    if (it != tag_consumers_.end()) {
      for (clio::run::u32 existing : it->second) {
        if (existing == node_id) {
          return;  // Already registered for this tag.
        }
      }
    }
  }

  // Writer path: insert/grow under exclusive lock. Re-check first (another
  // writer may have raced us); cap at kMaxConsumersPerTag.
  clio::run::ScopedCoRwWriteLock write_lock(tag_consumers_lock_);
  auto &slots = tag_consumers_[tag_id];
  for (clio::run::u32 existing : slots) {
    if (existing == node_id) {
      return;
    }
  }
  if (slots.size() >= kMaxConsumersPerTag) {
    HLOG(kDebug,
         "Compressor: consumer slot full for tag ({} entries), dropping node {}",
         slots.size(), node_id);
    return;
  }
  slots.push_back(node_id);
  HLOG(kDebug,
       "Compressor: registered consumer node {} for tag (slot {}/{})",
       node_id, slots.size(), kMaxConsumersPerTag);
}

bool Runtime::PickConsumerForTag(const clio::cte::core::TagId &tag_id,
                                 clio::run::u32 &node_id_out) {
  if (!config_.tracking_enabled_) {
    return false;
  }
  clio::run::ScopedCoRwReadLock read_lock(tag_consumers_lock_);
  auto it = tag_consumers_.find(tag_id);
  if (it == tag_consumers_.end() || it->second.empty()) {
    return false;
  }
  // Most-recent reader heuristic: the latest pushed entry is the most
  // recent reader of the tag. A future improvement is to fold in the
  // PollConsumers load samples and pick the least-loaded known reader,
  // but the most-recent heuristic is cheap and exploits temporal
  // locality (read-then-recompute patterns).
  node_id_out = it->second.back();
  return true;
}

// ==============================================================================
// Node Load Sampling
// ==============================================================================

clio::run::TaskResume Runtime::PollNodeLoad(clio::run::shared_ptr<PollNodeLoadTask> &task) {
  CLIO_TASK_BODY_BEGIN
  NodeLoadSample sample;
  auto* ipc_manager = CLIO_IPC;
  sample.node_id_ = ipc_manager ? static_cast<clio::run::u32>(ipc_manager->GetNodeId())
                                : 0;

  // CPU utilization since the last sample. Mutex protects prev_cpu_times_
  // because PollNodeLoad may run concurrently across workers.
  ctp::CpuTimes cur = ctp::SystemInfo::GetCpuTimes();
  {
    std::lock_guard<std::mutex> lk(cpu_times_mutex_);
    sample.cpu_usage_pct_ =
        ctp::SystemInfo::ComputeCpuUtilization(prev_cpu_times_, cur);
    prev_cpu_times_ = cur;
  }

  // AggregateOut worker load across all workers on this node.
  auto* orchestrator = CLIO_WORK_ORCHESTRATOR;
  if (orchestrator) {
    std::size_t num_workers = orchestrator->GetWorkerCount();
    sample.num_workers_ = static_cast<clio::run::u32>(num_workers);
    for (std::size_t i = 0; i < num_workers; ++i) {
      clio::run::Worker* worker = orchestrator->GetWorker(static_cast<clio::run::u32>(i));
      if (!worker) {
        continue;
      }
      clio::run::WorkerStats stats = worker->GetWorkerStats();
      sample.worker_load_us_ += stats.load_;
      sample.num_queued_tasks_ += stats.num_queued_tasks_;
      sample.num_blocked_tasks_ += stats.num_blocked_tasks_;
    }
  }

  task->sample_ = sample;
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PollConsumers(clio::run::shared_ptr<PollConsumersTask> &task) {
  CLIO_TASK_BODY_BEGIN
  (void)task;
  // No-op when tracking is disabled.
  if (!config_.tracking_enabled_) {
    CLIO_CO_RETURN;
  }
  // Snapshot the union of consumers across all tags under the reader
  // lock so the periodic poll does not hold the lock while issuing
  // remote tasks. We dedupe to a single PollNodeLoad per node — readers
  // may appear in multiple tags' lists.
  std::vector<clio::run::u32> snapshot;
  {
    clio::run::ScopedCoRwReadLock read_lock(tag_consumers_lock_);
    std::unordered_set<clio::run::u32> dedup;
    for (const auto &kv : tag_consumers_) {
      for (clio::run::u32 node : kv.second) {
        if (dedup.insert(node).second) {
          snapshot.push_back(node);
        }
      }
    }
  }

  if (snapshot.empty()) {
    CLIO_CO_RETURN;
  }

  // Fan out one PollNodeLoad task per consumer node, then await each.
  std::vector<clio::run::Future<PollNodeLoadTask>> futures;
  futures.reserve(snapshot.size());
  for (clio::run::u32 node_id : snapshot) {
    futures.emplace_back(
        client_.AsyncPollNodeLoad(clio::run::PoolQuery::Physical(node_id)));
  }

  for (std::size_t i = 0; i < futures.size(); ++i) {
    auto& fut = futures[i];
    CLIO_CO_AWAIT(fut);
    if (fut->GetReturnCode() == 0) {
      const NodeLoadSample& s = fut->sample_;
      HLOG(kDebug,
           "Compressor: consumer node {} cpu={:.1f}% worker_load={:.1f}us "
           "queued={} blocked={} workers={}",
           snapshot[i], s.cpu_usage_pct_, s.worker_load_us_,
           s.num_queued_tasks_, s.num_blocked_tasks_, s.num_workers_);
    } else {
      HLOG(kDebug, "Compressor: PollNodeLoad to node {} failed (rc={})",
           snapshot[i], fut->GetReturnCode());
    }
  }

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}


// ---- Public entrypoints -----------------------------------------------------
// Regular (kPutBlob=15 / kGetBlob=16) come from host clients; the POD variants
// (kPodPutBlob=43 / kPodGetBlob=44) are what the gpu_vector's cache manager
// submits from DEVICE code. Both share one implementation.
clio::run::TaskResume Runtime::CompressPutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask>& task) {
  CLIO_CO_AWAIT(CompressPutBlobImpl(task));
  CLIO_CO_RETURN;
}
clio::run::TaskResume Runtime::CompressPodPutBlob(
    clio::run::shared_ptr<clio::cte::core::PodPutBlobTask>& task) {
  CLIO_CO_AWAIT(CompressPutBlobImpl(task));
  CLIO_CO_RETURN;
}
clio::run::TaskResume Runtime::DecompressGetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask>& task) {
  CLIO_CO_AWAIT(DecompressGetBlobImpl(task));
  CLIO_CO_RETURN;
}
// (debug wrapper counter lives in DecompressGetBlobImpl entry)
clio::run::TaskResume Runtime::DecompressPodGetBlob(
    clio::run::shared_ptr<clio::cte::core::PodGetBlobTask>& task) {
  CLIO_CO_AWAIT(DecompressGetBlobImpl(task));
  CLIO_CO_RETURN;
}

// ============================================================================
// Interposed core data verbs (issue #886): the compressor as a transparent
// interposer over the CTE core's task interface. Machinery (forwarding,
// batching, region iteration) comes from CoreInterposer + blob_batch.h —
// shared with the replication chimod; only the transform policy lives here.
// ============================================================================

bool Runtime::CompressIntoShm(clio::cte::core::Context &ctx, const char *src,
                              clio::run::u64 size,
                              ctp::ipc::FullPtr<char> *stored,
                              clio::run::u64 *stored_size) {
  std::string library_name =
      ctp::CompressionFactory::NameForWireId(ctx.compress_lib_);
  ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
  if (ctx.compress_preset_ == 1) {
    preset = ctp::CompressionPreset::FAST;
  } else if (ctx.compress_preset_ == 3) {
    preset = ctp::CompressionPreset::BEST;
  }
  auto compressor = ctp::CompressionFactory::GetPreset(library_name, preset);
  if (!compressor) {
    return false;
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<char> compressed(size + (size / 20) + 1024);
  size_t compressed_size = compressed.size();
  if (!compressor->Compress(compressed.data(), compressed_size,
                            const_cast<char *>(src), size)) {
    return false;
  }
  size_t header_size = sizeof(CompressionHeader);
  size_t total = compressed_size + header_size;
  if (total >= size) {
    return false;  // not beneficial — caller stores raw
  }
  auto shm = CLIO_IPC->AllocateBuffer(total);
  if (shm.IsNull()) {
    return false;
  }
  // Record the EXACT compressed payload size. Omitting it (the 4th arg
  // defaults to 0) makes every interposer-written blob look "legacy" to
  // DecompressGetBlobImpl, which then falls back to the over-allocated fetch
  // size as the frame length — precisely the trailing-garbage case that
  // compressed_size_ was added to fix, so zstd/lz4 reads fail. The two older
  // compress paths (Compress, CompressPutBlobImpl) already pass it.
  CompressionHeader header(ctx.compress_lib_, ctx.compress_preset_, size,
                           compressed_size);
  std::memcpy(shm.ptr_, &header, header_size);
  std::memcpy(shm.ptr_ + header_size, compressed.data(), compressed_size);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t0)
                  .count();
  ctx.actual_original_size_ = size;
  ctx.actual_compressed_size_ = total;
  ctx.actual_compression_ratio_ =
      static_cast<double>(size) / static_cast<double>(total);
  ctx.actual_compress_time_ms_ = ms;
  // The one place that KNOWS the stored bytes are no longer the caller's
  // bytes (issue #818 authoritative transform bit).
  ctx.transform_flags_ |= clio::cte::core::kBlobTransformed |
                          clio::cte::core::kBlobTransformCompressed;
  *stored = shm;
  *stored_size = total;
  return true;
}

int Runtime::DecompressStored(const char *stored, clio::run::u64 stored_size,
                              char *dst, clio::run::u64 dst_cap,
                              clio::run::u64 *out_size) {
  if (stored_size < sizeof(CompressionHeader)) {
    return 6;
  }
  const auto *header = reinterpret_cast<const CompressionHeader *>(stored);
  if (!header->IsValid() || header->original_size_ > dst_cap) {
    return 6;
  }
  std::string library_name = ctp::CompressionFactory::NameForWireId(
      static_cast<int>(header->compress_lib_));
  ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
  if (header->compress_preset_ == 1) {
    preset = ctp::CompressionPreset::FAST;
  } else if (header->compress_preset_ == 3) {
    preset = ctp::CompressionPreset::BEST;
  }
  auto decompressor = ctp::CompressionFactory::GetPreset(library_name, preset);
  if (!decompressor) {
    return 3;
  }
  size_t decompressed = header->original_size_;
  if (!decompressor->Decompress(dst, decompressed,
                                const_cast<char *>(stored) +
                                    sizeof(CompressionHeader),
                                stored_size - sizeof(CompressionHeader))) {
    return 5;
  }
  *out_size = decompressed;
  return 0;
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::cte::core::Context &ctx = task->context_;
    // Operator pin: CLIO_CTE_COMPRESS_LIB must govern EVERY compression path,
    // including this interposed one. Runtime::Compress applies the same pin,
    // but the interposer never routes through Compress/DynamicSchedule — it
    // calls CompressIntoShm directly — so without this the pin silently did
    // not reach interposed puts. That mattered: clients that own no codec
    // policy (the gpu_vector builds its page tasks with a default Context, so
    // compress_lib_ == 0) fell through the gate below and stored every page
    // RAW while still reporting success, which is exactly the "silently
    // passed data through" failure the compressed gpu_vector tests exist to
    // catch. Applied before the gate so partial/vectored/replica writes still
    // clear the codec and store raw.
    {
      int pin_preset = 2;
      int pin_wire = PinWireFor(config_, &pin_preset);
      if (pin_wire >= 0) {
        ctx.compress_lib_ = pin_wire;
        ctx.compress_preset_ = pin_preset;
      }
    }
    // Compression is defined for WHOLE-BLOB writes only: a partial or
    // vectored write cannot patch a compressed stream, so those (and
    // replica-addressed or emulated puts) forward with the codec request
    // cleared — raw bytes, never a recorded codec (issue #818 rule).
    const bool whole_blob = task->segments_.empty() && task->offset_ == 0;
    if (ctx.replica_ != 0 || ctx.compress_lib_ <= 0 || ctx.emulate_ ||
        !whole_blob) {
      if (ctx.compress_lib_ > 0 && !whole_blob) {
        ctx.compress_lib_ = 0;
      }
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                             task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    auto src_full =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    ctp::ipc::FullPtr<char> stored;
    clio::run::u64 stored_size = 0;
    if (src_full.ptr_ != nullptr &&
        CompressIntoShm(ctx, src_full.ptr_, task->size_, &stored,
                        &stored_size)) {
      if (!core_client_) {
        core_client_ =
            std::make_unique<clio::cte::core::Client>(CorePoolId());
      }
      // Re-issuing under an explicit NAME means the fresh task carries the
      // default gpu_page_idx_, so the core's own "_pi<idx>" suffixing
      // (core_runtime.cc) never runs. Compose it here or every page of a
      // gpu_vector blob would land on the base name and overwrite the last.
      std::string name = task->blob_name_.str();
      if (task->gpu_page_idx_ !=
          clio::cte::core::PutBlobTask::kNoPageIdx) {
        name += "_pi" + std::to_string(task->gpu_page_idx_);
      }
      auto put = core_client_->AsyncPutBlob(
          task->tag_id_, name, 0, stored_size,
          stored.shm_.template Cast<void>(), task->score_, ctx, task->flags_,
          ForwardQuery(task->tag_id_, name));
      CLIO_CO_AWAIT(put);
      CLIO_IPC->FreeBuffer(stored);
      task->context_ = put->context_;
      task->return_code_ = put->GetReturnCode();
    } else {
      // Failed or not beneficial: store the caller's raw bytes and make
      // sure the blob is never recorded as carrying the attempted codec.
      ctx.compress_lib_ = 0;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                             task.template Cast<clio::run::Task>()));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Serve the read as-is first: the untransformed case (and every replica-
  // addressed read) costs nothing extra, and the core reports the blob's
  // authoritative transform state OUT through the context either way.
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                         task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() != 0 || task->context_.replica_ != 0 ||
      !(task->context_.transform_flags_ &
        clio::cte::core::kBlobTransformCompressed)) {
    CLIO_CO_RETURN;
  }
  {
    // The forwarded read handed back CODEC bytes. Fetch the whole stored
    // blob, decompress ONCE, then slice every requested region out of the
    // original — which is what makes partial and VECTORED reads of
    // compressed blobs work through this interposer at all.
    if (!core_client_) {
      core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
    }
    clio::run::u64 stored_size = 0;
    {
      auto sz = core_client_->AsyncGetBlobSize(task->tag_id_,
                                               task->blob_name_.str());
      CLIO_CO_AWAIT(sz);
      if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
        task->return_code_ = 10 + sz->GetReturnCode();
        CLIO_CO_RETURN;
      }
      stored_size = sz->size_;
    }
    auto stored = CLIO_IPC->AllocateBuffer(stored_size);
    if (stored.IsNull()) {
      task->return_code_ = 2;
      CLIO_CO_RETURN;
    }
    {
      auto get = core_client_->AsyncGetBlob(
          task->tag_id_, task->blob_name_.str(), 0, stored_size,
          /*flags=*/0, stored.shm_.template Cast<void>(),
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() != 0) {
        CLIO_IPC->FreeBuffer(stored);
        task->return_code_ = 10 + get->GetReturnCode();
        CLIO_CO_RETURN;
      }
    }
    const auto *header =
        reinterpret_cast<const CompressionHeader *>(stored.ptr_);
    clio::run::u64 original_size =
        stored_size >= sizeof(CompressionHeader) ? header->original_size_ : 0;
    auto scratch = CLIO_IPC->AllocateBuffer(original_size);
    if (scratch.IsNull()) {
      CLIO_IPC->FreeBuffer(stored);
      task->return_code_ = 2;
      CLIO_CO_RETURN;
    }
    clio::run::u64 out_size = 0;
    int rc = DecompressStored(stored.ptr_, stored_size, scratch.ptr_,
                              original_size, &out_size);
    CLIO_IPC->FreeBuffer(stored);
    if (rc != 0) {
      CLIO_IPC->FreeBuffer(scratch);
      task->return_code_ = rc;
      CLIO_CO_RETURN;
    }
    // Copy each requested region out of the original bytes (shared
    // scalar-vs-vectored iteration, blob_batch.h). Regions beyond the
    // original size keep the core's short-read semantics (left untouched).
    bool region_ok = true;
    clio::cte::core::ForEachBlobRegion(
        *task, [&](const clio::cte::core::BlobRegion &r) {
          if (r.blob_off_ >= out_size) {
            return true;  // wholly past EOF: short read
          }
          clio::run::u64 n = r.size_;
          if (r.blob_off_ + n > out_size) {
            n = out_size - r.blob_off_;
          }
          auto dst =
              CLIO_IPC->ToFullPtr<char>(r.data_.template Cast<char>());
          if (dst.ptr_ == nullptr) {
            region_ok = false;
            return false;
          }
          std::memcpy(dst.ptr_, scratch.ptr_ + r.blob_off_, n);
          return true;
        });
    CLIO_IPC->FreeBuffer(scratch);
    // The caller now holds ORIGINAL bytes: clear the transform report so
    // nothing downstream tries to undo the codec again.
    task->context_.transform_flags_ &=
        ~(clio::cte::core::kBlobTransformed |
          clio::cte::core::kBlobTransformCompressed);
    task->return_code_ = region_ok ? 0 : 3;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                         task.template Cast<clio::run::Task>()));
  // A transformed blob's stored size is header+codec bytes; size-then-read
  // callers need the LOGICAL size. Probe the header with a tiny ranged get —
  // its OUT context carries the authoritative transform bit, so a raw blob
  // that merely looks like a header can never be misreported.
  if (task->GetReturnCode() == 0 && task->replica_ == 0 &&
      task->size_ >= sizeof(CompressionHeader)) {
    if (!core_client_) {
      core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
    }
    auto hdr_buf = CLIO_IPC->AllocateBuffer(sizeof(CompressionHeader));
    if (!hdr_buf.IsNull()) {
      auto get = core_client_->AsyncGetBlob(
          task->tag_id_, task->blob_name_.str(), 0, sizeof(CompressionHeader),
          /*flags=*/0, hdr_buf.shm_.template Cast<void>(),
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() == 0 &&
          (get->context_.transform_flags_ &
           clio::cte::core::kBlobTransformCompressed)) {
        const auto *header =
            reinterpret_cast<const CompressionHeader *>(hdr_buf.ptr_);
        if (header->IsValid()) {
          task->size_ = header->original_size_;
        }
      }
      CLIO_IPC->FreeBuffer(hdr_buf);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Batches carry a batch-wide Context (issue #886 follow-up) and get
  // SCALAR-EQUIVALENT semantics. No codec requested (or replica-addressed /
  // emulated): forward the batch intact — the chain below executes every
  // record with this context, records stay raw, amortization preserved.
  if (task->context_.replica_ != 0 || task->context_.compress_lib_ <= 0 ||
      task->context_.emulate_) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  // Codec requested: records transform INDIVIDUALLY (some compress, some
  // stay raw — not-beneficial or offset writes), so one shared batch cannot
  // describe the results. Decompose through OUR scalar handler, which
  // applies the exact scalar rules per record.
  task->num_ok_ = 0;
  task->first_rc_ = 0;
  {
    auto *ipc_manager = CLIO_CPU_IPC;
    clio::cte::core::MultiPutBatchView batch;
    if (!clio::cte::core::MultiPutBatchView::Attach(*task, &batch)) {
      task->SetReturnCode(batch.descs_.empty() ? 0 : 1);
      CLIO_CO_RETURN;
    }
    for (size_t bi = 0; bi < batch.size(); ++bi) {
      const auto &d = batch.descs_[bi];
      if (!batch.RecordValid(bi)) {
        if (task->first_rc_ == 0) task->first_rc_ = 2;
        continue;
      }
      auto sub = ipc_manager->NewTask<clio::cte::core::PutBlobTask>(
          clio::run::CreateTaskId(), task->pool_id_,
          clio::run::PoolQuery::Local(), d.tag_id_, d.blob_name_, d.offset_,
          d.size_, batch.RecordSlice(bi), /*score=*/-1.0f, task->context_,
          /*flags=*/0);
      sub.get()->BeginRunContext();
      CLIO_CO_AWAIT(PutBlob(sub));
      int rc = sub->GetReturnCode();
      if (rc == 0) {
        task->num_ok_++;
      } else if (task->first_rc_ == 0) {
        task->first_rc_ = rc;
      }
    }
  }
  task->SetReturnCode(task->first_rc_ == 0 ? 0 : task->first_rc_);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::compressor

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::cte::compressor::Runtime)
