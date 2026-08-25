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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
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
// Direct, for the exploration sweep's CompressLaunch/CompressFinish. The
// factory returns a base Compressor, which has no async path by design.
#include "clio_ctp/compress/nvcomp.h"
#include "clio_ctp/compress/data_stats.h"
#include "clio_ctp/compress/preprocess/byte_shuffle.h"
#include "clio_ctp/compress/preprocess/quantization.h"
#include "clio_ctp/compress/preprocess/data_stats_gpu.h"
#include "clio_ctp/compress/model/ranking.h"
#include "clio_ctp/util/logging.h"
#include "clio_cte/compressor/models/neuropress_bridge.h"
#include "clio_cte/compressor/neuropress_chunk_diag.h"
#include "clio_cte/compressor/neuropress_telemetry.h"
#include "clio_cte/compressor/neuropress_explore.h"

namespace clio::cte::compressor {

// Bring chi namespace items into scope for CLIO_CUR_WORKER macro
using clio::run::chi_cur_worker_key_;
using clio::run::Worker;

/**
 * Byte-shuffle element size packed into the free high bits of
 * compress_preset_.
 *
 * With error_bound=0 upstream's ranking masks every quantize action, so
 * byte-shuffle is the only extra dimension a lossless deployment can reach.
 *
 * Packed rather than added as a field: CompressionHeader is a fixed 24 bytes by
 * static_assert and is the on-disk format, so growing it would strand every
 * existing blob. compress_preset_ only ever holds 0-3, leaving the upper bits
 * free, and an older blob decodes to elem_size 0 -- "not shuffled".
 */
constexpr uint32_t kPresetMask = 0xFFu;
constexpr uint32_t kShuffleShift = 8;
constexpr uint32_t kShuffleMask = 0xFFu;

/**
 * Format version, in bits 16-23 of the same word.
 *
 * Upstream gates acceptance on an explicit version so a reader refuses a
 * layout it does not understand instead of misparsing it. There is no room for
 * a dedicated field here (24 bytes, on-disk format), but compress_preset_ uses
 * only bits 0-15, so bits 16-23 are free. An older blob decodes to version 0,
 * which is exactly right: it IS the original format.
 *
 * Bump kFormatVersion whenever the meaning of any header field, or of the bytes
 * that follow, changes. Readers accept up to their own version and reject newer.
 */
constexpr uint32_t kVersionShift = 16;
constexpr uint32_t kVersionMask = 0xFFu;
constexpr uint32_t kFormatVersion = 1;  // v0 = pre-versioning original layout

inline uint32_t PackPreset(uint32_t preset, uint32_t shuffle_elem_size) {
  return (preset & kPresetMask) |
         ((shuffle_elem_size & kShuffleMask) << kShuffleShift) |
         ((kFormatVersion & kVersionMask) << kVersionShift);
}
inline uint32_t UnpackPreset(uint32_t packed) { return packed & kPresetMask; }
inline uint32_t UnpackShuffle(uint32_t packed) {
  return (packed >> kShuffleShift) & kShuffleMask;
}
inline uint32_t UnpackVersion(uint32_t packed) {
  return (packed >> kVersionShift) & kVersionMask;
}

/**
 * @brief Analytical PSNR for linear quantization, verbatim from upstream.
 *
 * gpucompress_compress.cpp -- expected MSE for uniform error in
 * [-eb, eb] is eb^2/3, so PSNR = 10*log10(range^2 / MSE), capped at 120 dB.
 * Returns -1 for a degenerate range or bound, which is upstream's "PSNR
 * undefined" sentinel and the value that makes the SGD skip the head.
 */
inline double AnalyticalPsnr(double data_range, double error_bound) {
  if (data_range <= 0.0 || error_bound <= 0.0) return -1.0;
  const double mse_expected = (error_bound * error_bound) / 3.0;
  return std::min(10.0 * std::log10((data_range * data_range) / mse_expected),
                  120.0);
}

/**
 * Quantization state, in the last free byte of the same word.
 *
 * bit 24    : quantize applied
 * bits 25-26: precision code (0 = int8, 1 = int16, 2 = int32)
 *
 * Upstream keeps the equivalent in a quant_flags field, with room for it
 * because its header is 64 bytes. Clio's is 24 and fixed, so the flags ride
 * here and the four doubles that make the transform invertible go in a header
 * EXTENSION appended only when quantization ran -- see QuantHeaderExtension.
 * Type is not encoded because LINEAR is the only quantizer either side has.
 */
constexpr uint32_t kQuantShift = 24;
constexpr uint32_t kQuantEnabledBit = 1u;
constexpr uint32_t kQuantPrecisionShift = 1;
constexpr uint32_t kQuantPrecisionMask = 0x3u;

inline uint32_t PackQuant(bool enabled, int precision) {
  if (!enabled) return 0;
  uint32_t code = (precision == 8) ? 0u : (precision == 16) ? 1u : 2u;
  return (kQuantEnabledBit | (code << kQuantPrecisionShift)) << kQuantShift;
}
inline bool UnpackQuantEnabled(uint32_t packed) {
  return ((packed >> kQuantShift) & kQuantEnabledBit) != 0;
}
inline int UnpackQuantPrecision(uint32_t packed) {
  uint32_t code = (packed >> (kQuantShift + kQuantPrecisionShift)) &
                  kQuantPrecisionMask;
  return (code == 0) ? 8 : (code == 1) ? 16 : 32;
}

/**
 * The four doubles a reader needs to invert a quantization, appended
 * directly after the 24-byte core header and present ONLY when the quantize
 * bit is set. Same values upstream keeps in its extended v2 fields
 * (compression_header.h:63-66).
 *
 * This is why the format carries a version: a reader that does not know
 * about the extension would compute the wrong payload offset, and the
 * version gate makes that a clean rejection instead of silent corruption.
 */
struct QuantHeaderExtension {
  double error_bound;
  double scale;
  double data_min;
  double data_max;
};
static_assert(sizeof(QuantHeaderExtension) == 32,
              "QuantHeaderExtension is part of the on-disk format");


/**
 * Compression header prepended to compressed data for self-describing format.
 * This allows decompression without external metadata.
 */
struct CompressionHeader {
  static constexpr uint32_t kMagic = 0x43544543;  // "CTEC" in ASCII
  uint32_t magic_;            // Magic number to identify compressed data
  uint32_t compress_lib_;     // Compression library ID
  uint32_t compress_preset_;  // Compression preset | shuffle elem size << 8
  /**
   * Compressed payload length in bytes, excluding this header. 0 = "not
   * recorded", for blobs written before this field existed.
   *
   * Upstream carries the same field and bounds-checks it against the buffer it
   * was handed -- a bad value is an invalid header, never a guess. Without it
   * the length has to be derived from a blob-size RPC, falling back to the
   * caller's LOGICAL size when that fails, which over-reads past the real
   * stream into uninitialized memory (see Decompress()).
   *
   * uint32_t because it lives in the 4 padding bytes already wasted between
   * compress_preset_ and original_size_: the header is the on-disk format,
   * static_assert-ed at 24 bytes. A chunk large enough to overflow 32 bits is
   * far beyond this path, and IsValid() rejects that case anyway.
   */
  uint32_t compressed_size_;
  uint64_t original_size_;    // Original uncompressed size

  CompressionHeader()
      : magic_(kMagic),
        compress_lib_(0),
        compress_preset_(0),
        compressed_size_(0),
        original_size_(0) {}

  CompressionHeader(uint32_t lib, uint32_t preset, uint64_t orig_size,
                    uint64_t compressed_size)
      : magic_(kMagic),
        compress_lib_(lib),
        compress_preset_(preset),
        // Anything that does not fit records as "not recorded" rather than
        // truncating: a wrong length is worse than an absent one, since the
        // read side would trust it.
        compressed_size_(compressed_size <= UINT32_MAX
                             ? static_cast<uint32_t>(compressed_size)
                             : 0u),
        original_size_(orig_size) {}

  /**
   * @brief Magic matches AND the format is one this build understands.
   *
   * Version lives in compress_preset_'s bits 16-23 (see kVersionShift).
   * Rejecting a NEWER version is the point: without it, a future writer that
   * redefines a field would be silently misread by this binary rather than
   * refused. Older versions stay readable -- same rule as upstream's
   * `version >= 1 && version <= COMPRESSION_HEADER_VERSION`, except that
   * Clio's floor is 0, since blobs predating the field decode to 0 and that
   * is a layout this code still reads correctly.
   */
  bool IsValid() const {
    return magic_ == kMagic &&
           UnpackVersion(compress_preset_) <= kFormatVersion;
  }

  /**
   * @brief Payload length to feed the decompressor, or 0 if unusable.
   *
   * @param physical_size Total stored bytes (header + payload) as the caller
   *   knows them, used as the bound.
   *
   * Mirrors upstream's two checks: the addition must not wrap, and header +
   * payload must fit inside the bytes actually available. Returns 0 when the
   * field is absent (old blob) or fails either check, so the caller can fall
   * back to its existing size-query path instead of trusting a bad value.
   * The padding this field occupies was never explicitly zeroed before, so a
   * pre-existing blob can carry arbitrary bits here -- the bound check is
   * what makes reading it safe, not an assumption that it is 0.
   */
  size_t PayloadSize(size_t physical_size) const {
    if (compressed_size_ == 0) return 0;
    const size_t payload = static_cast<size_t>(compressed_size_);
    // Header bytes include the quantization extension when present, so the
    // bound check stays honest for a quantized blob.
    const size_t hdr = sizeof(CompressionHeader) +
                       (((compress_preset_ >> 24) & 1u)
                            ? sizeof(QuantHeaderExtension)
                            : 0);
    const size_t total = hdr + payload;
    if (total < payload) return 0;             // wraparound
    if (total > physical_size) return 0;       // does not fit what we have
    return payload;
  }
};
/**
 * @brief Header bytes for this blob: 24 core, plus 32 when quantized.
 *
 * Every offset into a stored blob must go through this rather than
 * sizeof(CompressionHeader), or a quantized blob's payload is read 32 bytes
 * early.
 */
inline size_t HeaderBytesFor(uint32_t packed_preset) {
  return sizeof(CompressionHeader) +
         (UnpackQuantEnabled(packed_preset) ? sizeof(QuantHeaderExtension)
                                            : 0);
}

static_assert(sizeof(CompressionHeader) == 24,
              "CompressionHeader must be 24 bytes -- it is the on-disk "
              "format; growing it strands every already-written blob");
static_assert(offsetof(CompressionHeader, original_size_) == 16,
              "compressed_size_ must occupy the former padding at offset 12");

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

  // Load NeuroPress NN model if configured (issue #693). Consulted
  // first in EstCompressionStats()'s dynamic-selection path -- see there.
  if (!config_.neuropress_model_path_.empty()) {
    try {
      HLOG(kDebug, "Loading NeuroPress NN model from: {}",
           config_.neuropress_model_path_);
      neuropress_predictor_ =
          std::make_unique<ctp::compress::model::NeuroPressNNPredictor>();
      neuropress_predictor_->SetLearningRate(config_.neuropress_learning_rate_);
      if (neuropress_predictor_->Load(config_.neuropress_model_path_)) {
        HLOG(kDebug, "NeuroPress NN model loaded successfully");
      } else {
        // NeuroPress has no CPU path -- upstream's network exists only as CUDA
        // kernels -- so a load failure means the requested model cannot run.
        // Fail the pool rather than fall back: a legacy-heuristic selection is
        // indistinguishable downstream from a NeuroPress one. Deployments that
        // do not configure NeuroPress are unaffected.
        HLOG(kError,
             "NeuroPress was requested (model path '{}') but could not be "
             "loaded. It has no CPU implementation, so it will not be "
             "silently replaced by another model -- failing CreateCompressor. "
             "Unset neuropress_model_path_ to run without it.",
             config_.neuropress_model_path_);
        neuropress_predictor_.reset();
        task->SetReturnCode(1);
        CLIO_CO_RETURN;
      }
    } catch (const std::exception& e) {
      HLOG(kError,
           "Exception while loading the requested NeuroPress NN model: {} -- "
           "failing rather than falling back to another model",
           e.what());
      neuropress_predictor_.reset();
      task->SetReturnCode(1);
      CLIO_CO_RETURN;
    }
  }

  // Static codec: the control condition. It overrides every selection knob,
  // so force them off HERE rather than trusting the config to be internally
  // consistent -- a run that reported "static nvcomp-zstd" while exploration
  // silently replaced the codec on some chunks would be a worthless control.
  // Symmetric with the best-mode block below: both normalise the config once
  // so the per-chunk path reads plain fields.
  if (!config_.neuropress_static_lib_.empty()) {
    const int wire =
        ctp::CompressionFactory::WireIdForName(config_.neuropress_static_lib_);
    const std::string resolved = ctp::CompressionFactory::NameForWireId(wire);
    config_.neuropress_online_learning_enabled_ = false;
    config_.neuropress_exploration_enabled_ = false;
    config_.neuropress_best_mode_ = false;
    HLOG(kWarning,
         "Static codec '{}' (resolved '{}', wire {}) is ON for pool '{}': "
         "every chunk uses this library and NeuroPress selection, learning "
         "and exploration are all disabled (byte shuffle: {}).",
         config_.neuropress_static_lib_, resolved, wire, pool_name_,
         config_.neuropress_static_shuffle_ == 0
             ? std::string("off")
             : std::to_string(config_.neuropress_static_shuffle_) + "-byte");
    if (resolved != config_.neuropress_static_lib_) {
      HLOG(kWarning,
           "  '{}' is not a registered library; it fell back to '{}'. Check "
           "the name against CompressionFactory's registry.",
           config_.neuropress_static_lib_, resolved);
    }
  }

  // Best mode brings its own exploration settings rather than requiring the
  // caller to assemble them, the same way gpucompress_set_best_mode() turns
  // exploration on and pins the K override to 31 rather than leaving either
  // to the caller. Applied once here so the per-chunk path reads plain
  // config, and so the effective settings are what a config dump reports.
  if (config_.neuropress_best_mode_) {
    config_.neuropress_exploration_enabled_ = true;
    config_.neuropress_exploration_k_ = 31;
    HLOG(kWarning,
         "NeuroPress best mode is ON for pool '{}': every chunk is compressed "
         "with all {} remaining configurations and stored as the smallest "
         "result. Selection is ratio-only and both SGD phases are off. This "
         "is a measurement mode and is roughly 32x slower than normal.",
         pool_name_, config_.neuropress_exploration_k_);
  }

  if (!qtable_predictor_ && !linreg_predictor_ && !neuropress_predictor_) {
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
    neuropress_predictor_.reset();
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

/** Defined with the selection log below; used here to skip work it alone reads.
 *  Declared in the same anonymous namespace as the definition, or this would
 *  be a different function and the call would be ambiguous. */
std::vector<CompressionStats> Runtime::EstCompressionStats(
    const void* chunk, clio::run::u64 chunk_size, const Context& context,
    bool* out_ranked_by_cost, double* out_entropy, double* out_mad,
    double* out_second_deriv, bool* out_neuropress_gpu_failed,
    const void** out_device_stats) {
  std::vector<CompressionStats> results;
  if (out_ranked_by_cost) *out_ranked_by_cost = false;
  if (out_neuropress_gpu_failed) *out_neuropress_gpu_failed = false;
  if (out_device_stats) *out_device_stats = nullptr;

  double entropy = 0.0, mad = 0.0, second_derivative_mean = 0.0;

  // Two models, two feature definitions, two ways of reading the same bytes.
  // NeuroPress decides when it is ready (neuropress_selection.cc); Clio's own
  // heuristics take the chunk otherwise, or when NeuroPress declines it.
  if (NeuroPressActive(context)) {
    std::vector<CompressionStats> neuropress_stats = NeuroPressRankChunk(
        chunk, chunk_size, context, &entropy, &mad, &second_derivative_mean,
        out_entropy, out_mad, out_second_deriv, out_neuropress_gpu_failed,
        out_device_stats);
    if (!neuropress_stats.empty()) {
      // Already best-first under NeuroPress's cost model; saying so is what
      // stops the caller re-selecting on ratio alone and discarding it.
      if (out_ranked_by_cost) *out_ranked_by_cost = true;
      return neuropress_stats;
    }
    // Declined. The statistics it computed are still the right ones for this
    // chunk, so the heuristics below reuse them rather than measuring again --
    // and the selection they produce is NOT NeuroPress's, which
    // NeuroPressRankChunk has already said out loud.
  } else {
    // Clio's own models were fit on these features, with the context's type
    // mapping rather than NeuroPress's unconditional float32.
    const ctp::DataType data_type = (context.data_type_ == 1)
                                        ? ctp::DataType::FLOAT32
                                        : ctp::DataType::UINT8;
    const size_t type_size = ctp::DataStatisticsFactory::GetTypeSize(data_type);
    const size_t num_elements = static_cast<size_t>(chunk_size / type_size);
    // A chunk smaller than one element has no statistics; scoring the zeros
    // left behind would rank it as perfectly compressible.
    if (!(num_elements > 0 &&
          ctp::ComputeCompressionFeatures(chunk, num_elements, data_type,
                                          &entropy, &mad,
                                          &second_derivative_mean))) {
      HLOG(kWarning,
           "EstCompressionStats: no usable statistics for this chunk "
           "(size={} elem_size={})",
           chunk_size, type_size);
    }
    if (out_entropy) *out_entropy = entropy;
    if (out_mad) *out_mad = mad;
    if (out_second_deriv) *out_second_deriv = second_derivative_mean;
  }

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

void Runtime::RecordDecompFeatures(
    const std::string& blob_key,
    const ctp::compress::model::CompressionFeatures& features) {
  std::lock_guard<std::mutex> lock(decomp_features_mutex_);
  auto it = decomp_features_.find(blob_key);
  if (it != decomp_features_.end()) {
    // Overwrite: the newest compression of this blob is what a subsequent
    // read will actually decompress.
    it->second.features = features;
    it->second.seq = decomp_feature_seq_++;
    return;
  }
  if (decomp_features_.size() >= kMaxDecompFeatureRecords) {
    // FIFO-evict the oldest. These are training hints for blobs that may
    // never be read back; dropping the stalest one loses a learning
    // opportunity, never correctness.
    auto oldest = decomp_features_.begin();
    for (auto cur = decomp_features_.begin(); cur != decomp_features_.end();
         ++cur) {
      if (cur->second.seq < oldest->second.seq) oldest = cur;
    }
    decomp_features_.erase(oldest);
  }
  decomp_features_.emplace(
      blob_key, DecompFeatureRecord{features, decomp_feature_seq_++});
}

void Runtime::LearnDecompTime(const std::string& blob_key,
                              double measured_ms) {
  if (!config_.neuropress_online_learning_enabled_ || measured_ms <= 0.0) {
    return;
  }
  if (!neuropress_predictor_ || !neuropress_predictor_->IsReady()) return;

  std::vector<ctp::compress::model::CompressionFeatures> batch_features;
  std::vector<double> batch_times;
  {
    std::lock_guard<std::mutex> lock(decomp_features_mutex_);
    auto it = decomp_features_.find(blob_key);
    if (it == decomp_features_.end()) return;  // never compressed here

    // Floor the measurement at 1 ms before it becomes a target. Upstream
    // stores `std::max(1.0f, ms)` (diagnostics_store.hpp:96-102) and its
    // learning path reads that clamped field, keeping the raw value only
    // for diagnostics. Sub-millisecond reads are routine for chunks of a
    // few MB, and feeding the raw value trains toward a target ~4x lower
    // in log space than upstream would use, dragging the head down until
    // it hits the +-5 weight clamp.
    it->second.measured_ms = std::max(1.0, measured_ms);

    // Train over EVERY record that has a measurement, not just this one and
    // not only once per record. gpucompress_batched_decomp_sgd() re-sweeps
    // the whole diagnostics store on each read and never consumes entries,
    // so a chunk read earlier is replayed into every later batch. Consuming
    // records (and gating on a batch of 8) meant a workload that read back
    // fewer than 8 blobs never trained the decompression head at all, since
    // the pending vectors were simply dropped at process exit.
    batch_features.reserve(decomp_features_.size());
    batch_times.reserve(decomp_features_.size());
    for (const auto& entry : decomp_features_) {
      if (entry.second.measured_ms <= 0.0) continue;
      batch_features.push_back(entry.second.features);
      batch_times.push_back(entry.second.measured_ms);
    }
  }
  if (batch_features.empty()) return;

  // One averaged update over the batch, matching upstream's
  // gpucompress_batched_decomp_sgd() -- the trust region scales to the mean
  // absolute error across chunks, not to whichever single blob was read last.
  bool trained =
      neuropress_predictor_->TrainDecompHead(batch_features, batch_times);
  HLOG(kDebug, "NeuroPress decomp-head SGD: batch={} trained={}",
       batch_features.size(), trained);
}


/* ---------------------------------------------------------------------------
 * End-to-end path trace -- COMPILE-TIME, off unless -DCLIO_NEUROPRESS_PATH_TRACE.
 * Compressor half of the trace in clio_vol.cc; same "[np-path]" prefix so one
 * grep spans connector and compressor. Undefined => expands to nothing, so
 * the compress path pays nothing.
 * --------------------------------------------------------------------------- */
#ifdef CLIO_NEUROPRESS_PATH_TRACE
#define CLIO_PATH_TRACE(...)                          \
  do {                                                \
    std::fprintf(stderr, "[np-path] " __VA_ARGS__);    \
    std::fprintf(stderr, "\n");                       \
    std::fflush(stderr);                              \
  } while (0)
#else
#define CLIO_PATH_TRACE(...) ((void)0)
#endif


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

    CLIO_PATH_TRACE("WRITE  compressor DynamicSchedule blob='%s' bytes=%llu "
                    "ptr=%s device=%d",
                    task->blob_name_.str().c_str(),
                    (unsigned long long)chunk_size,
                    chunk_data ? "ok" : "NULL",
                    chunk_data ? (ctp::IsDevicePointer(chunk_data) ? 1 : 0) : -1);

    // Check if we have valid chunk data
    if (chunk_data == nullptr || chunk_size == 0) {
      HLOG(kWarning, "Invalid chunk data for dynamic scheduling");
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Get compression stats. The three statistics are captured here rather
    // than from the SGD snapshot further down, which is only taken when
    // online learning is on -- these are what the selection actually ranked
    // on, in every mode.
    bool ranked_by_cost = false;
    double sel_entropy = 0.0, sel_mad = 0.0, sel_second_deriv = 0.0;
    bool neuropress_gpu_failed = false;
    // Statistics the selection ranked on; see out_device_stats.
    const void* sel_device_stats = nullptr;
    std::vector<CompressionStats> stats;
    if (!config_.neuropress_static_lib_.empty()) {
      // Control condition: one candidate, no inference. Deliberately does
      // NOT call EstCompressionStats -- running the model and discarding its
      // answer would put NN latency into a measurement whose entire point is
      // to exclude it.
      CompressionStats fixed{};
      fixed.compress_lib_ =
          ctp::CompressionFactory::WireIdForName(config_.neuropress_static_lib_);
      // Preset 2 (BALANCED) with no byte shuffle. The GPU codecs are
      // single_mode -- preset is ignored and the id always uses slot 2 -- and
      // upstream's non-AUTO algorithms apply no shuffle, so a shuffled
      // control would not be comparing the same transform.
      fixed.compress_preset_ = static_cast<int>(
          PackPreset(2, config_.neuropress_static_shuffle_));
      fixed.compression_ratio_ = 1.0;
      stats.push_back(fixed);
      ranked_by_cost = true;  // take stats.front() verbatim below
    } else {
      stats =
          EstCompressionStats(chunk_data, chunk_size, context, &ranked_by_cost,
                              &sel_entropy, &sel_mad, &sel_second_deriv,
                              &neuropress_gpu_failed, &sel_device_stats);
    }

    if (neuropress_gpu_failed) {
      // NeuroPress was asked for, the chunk was on the device, and its GPU
      // path could not decide. Upstream FAILS the write here -- a compress
      // error becomes worker_err = -1 and propagates out through H5Dwrite
      // (H5VLgpucompress.cu:2057, :2463, :3542) -- it does not store the
      // chunk by some other route. Storing it uncompressed, as the branch
      // below does, would be a silent substitution of a different outcome
      // for the one that was requested.
      HLOG(kError,
           "Compress: NeuroPress GPU selection failed for chunk of {} bytes; "
           "failing the write rather than storing it uncompressed",
           chunk_size);
      task->return_code_ = 4;
      CLIO_CO_RETURN;
    }

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

    // Choose best compression strategy. When NeuroPress ranked these, the
    // list is ALREADY best-first under its cost model (comp time + decomp
    // time + I/O); re-running BestCompressForNode would re-select on ratio
    // alone and throw that ordering away. Take element 0 instead, which is
    // what NeuroPress's own action selection does.
    int best_tier = 0, best_lib = 0, best_preset = 2;
    double best_time = 0.0;
    float tier_score = 0.0F;
    if (ranked_by_cost) {
      best_lib = stats.front().compress_lib_;
      best_preset = stats.front().compress_preset_;
      best_time = stats.front().compress_time_ms_;
    } else {
      std::tie(best_tier, best_lib, best_preset, best_time, tier_score) =
          BestCompressForNode(context, chunk_data, chunk_size, container_id_,
                              stats);
    }

    // Update context with selected compression library and preset
    context.compress_lib_ = best_lib;
    context.compress_preset_ = best_preset;
    // PRIMARY, not final: exploration below can replace this with a different
    // codec, and does. Reporting this as "selected" claimed gdeflate on 62
    // chunks whose stored headers on disk say bitcomp -- see the final hook at
    // the end of this function.
    CLIO_PATH_TRACE("WRITE  neuropress primary lib=%d (%s) preset=%d",
                    best_lib,
                    ctp::CompressionFactory::NameForWireId(best_lib).c_str(),
                    best_preset);
#ifdef CLIO_NEUROPRESS_PATH_TRACE
    const int np_primary_lib = best_lib;
    const int np_primary_preset = best_preset;
#endif
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

    // Snapshot the intrinsic data features NeuroPress's own
    // prediction was based on, for a possible SGD update below. Computed here
    // (chunk_data is certainly still valid -- the same pointer
    // EstCompressionStats just read) rather than after the await, since
    // nothing downstream guarantees the input blob outlives Compress().
    // Chunk-record slot, completed by the exploration block. -1 = none.
    int np_diag_slot = -1;
    bool neuropress_feat_valid = false;
    double neuropress_entropy = 0.0, neuropress_mad = 0.0,
           neuropress_second_deriv = 0.0;
    // Best mode needs these too. The features gate the exploration block
    // below (via neuropress_feat_valid), and best mode explores without
    // learning -- so keying their computation on the learning switch alone
    // left best mode with no features, hence no exploration, hence a run
    // that quietly returned an inference result.
    if ((config_.neuropress_online_learning_enabled_ ||
         config_.neuropress_best_mode_) &&
        NeuroPressActive(context)) {
      // FLOAT32 unconditionally, and NOT context.data_type_. Reaching here
      // means the enclosing guard already matched `neuropress_active` above
      // exactly, so inference computed these same three statistics as float32
      // -- see the comment there. Deriving the type from context.data_type_
      // instead made a caller that leaves it 0 (the default, e.g. the HDF5
      // VOL) train on uint8 statistics against features inference had read as
      // float32: MAD and entropy land hundreds of sigma apart, so every SGD
      // step corrected the model toward an input it never saw at inference.
      // That is precisely the failure the scope comment below warns about.
      ctp::DataType feat_type = ctp::DataType::FLOAT32;
      size_t feat_type_size = ctp::DataStatisticsFactory::GetTypeSize(feat_type);
      // Whole chunk -- MUST match EstCompressionStats' scope above, or the
      // features SGD trains on are not the features inference predicted
      // from, and the model learns against a different input than it saw.
      size_t feat_num_elements = static_cast<size_t>(chunk_size / feat_type_size);
      if (feat_num_elements == 0) feat_num_elements = 1;
      // Same guard as EstCompressionStats: on a device-resident chunk whose
      // stats cannot be computed the values are zeros, and training on them
      // would teach the model that chunk was trivially compressible.
      /* Same call as inference above, for the same reason: training on
         statistics read from a different interpretation of the bytes than the
         prediction was made from teaches the model against an input it never
         saw. On float64 those statistics were NaN, so every SGD step took a
         NaN gradient. */
      (void)feat_type;
      (void)feat_num_elements;
      if (sel_device_stats != nullptr) {
        /* The SGD kernel reads the statistics from this device buffer. The
           24 bytes come back only for the host consumers -- diagnostics,
           explore log, decomp head -- as upstream's "P4 fix" does. */
        neuropress_feat_valid = ctp::ReadDeviceFeatureStats(
            sel_device_stats, &neuropress_entropy, &neuropress_mad,
            &neuropress_second_deriv, nullptr);
      } else {
        /* Nothing on the device to reuse. Same call as inference, so both
           read the bytes the same way (float64 would otherwise give NaN). */
        neuropress_feat_valid = ctp::ComputeNeuroPressFeatures(
            chunk_data, chunk_size, task->context_.data_type_,
            &neuropress_entropy, &neuropress_mad, &neuropress_second_deriv);
      }
    }

    // Defer the store when exploration may replace this pick. Exploration
    // needs the primary MEASURED to have something to beat, but storing it
    // first means a winner is written over the top: PutBlob overwrites bytes
    // without shortening a longer blob, so the rejected candidate's footprint
    // stays allocated, and one blob ends up with two WAL records that a
    // shard-sequential replay can apply out of order. With the store deferred
    // there is exactly one put per blob, made once the winner is known.
    //
    // Only when exploration is actually on. With it off there is nothing that
    // could replace the pick, so the ordinary single-put path stands and this
    // costs nothing.
    const bool defer_store = config_.neuropress_exploration_enabled_;
    // Set when the exploration block below performs the one put. If it never
    // adopts anything, the primary still has to be stored -- see the fallback
    // after that block.
    bool stored_by_exploration = false;

    // Now call Compress to perform compression (and PutBlob unless deferred)
    auto compress_task = client_.AsyncCompress(
        clio::run::PoolQuery::Local(), task->tag_id_, task->blob_name_.str(),
        task->offset_, task->size_, task->blob_data_, task->score_, context,
        task->flags_, task->core_pool_id_, defer_store);
    CLIO_CO_AWAIT(compress_task);

    // Copy results back
    task->context_ = compress_task->context_;
    task->tier_score_ = compress_task->tier_score_;
    task->return_code_ = compress_task->return_code_;

    // The primary's image, held unstored while exploration runs. Freed by
    // PrimaryImage's destructor on every exit -- including the exception path,
    // which is why it is a guard and not a pair of locals.
    struct PrimaryImage {
      ctp::ipc::ShmPtr<> data = ctp::ipc::ShmPtr<>::GetNull();
      clio::run::u64 size = 0;
      ctp::ipc::AllocatorId gpu_alloc;
      bool owned = false;
      bool valid() const { return size != 0 && !data.IsNull(); }
      void release() {
        if (!owned) { data = ctp::ipc::ShmPtr<>::GetNull(); size = 0; return; }
        if (!gpu_alloc.IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, gpu_alloc);
          gpu_alloc = ctp::ipc::AllocatorId();
        } else if (!data.IsNull()) {
          CLIO_IPC->FreeBuffer(CLIO_IPC->ToFullPtr<char>(
              data.template Cast<char>()));
        }
        data = ctp::ipc::ShmPtr<>::GetNull();
        size = 0;
      }
      ~PrimaryImage() { release(); }
    } primary_image;
    if (defer_store) {
      primary_image.data = compress_task->stored_data_;
      primary_image.size = compress_task->stored_size_;
      primary_image.gpu_alloc = compress_task->stored_gpu_alloc_;
      primary_image.owned = compress_task->stored_owned_;
    }

    // Record what the model chose for this chunk, before the online-learning
    // and exploration blocks below can adopt an alternative. Not gated on
    // neuropress_feat_valid: that flag is only set when online learning or
    // best mode is on, and a selection is worth recording either way -- the
    // entropy/mad/curvature
    // columns are zero without that snapshot, but the chosen configuration is
    // always real.
    {
      const CompressionStats *logged_pred = nullptr;
      for (const auto &stat : stats) {
        if (stat.compress_lib_ == best_lib &&
            stat.compress_preset_ == best_preset) {
          logged_pred = &stat;
          break;
        }
      }
      // FNV-1a over the chunk this selection was made for, so a comparison
      // against another implementation can prove the two saw the same bytes
      // rather than assume it. A device-resident chunk is staged to host
      // first: that copy is pure diagnostic cost, but it only happens when
      // the log is switched on, and without it the interesting case -- a
      // GPU-resident write, which is the whole point of this path -- is
      // exactly the one that goes unverified.
      unsigned long long checksum = 0;
      if (SelectionLogEnabled() && chunk_data && chunk_size > 0) {
        std::vector<char> staged;
        const unsigned char *p =
            static_cast<const unsigned char *>(chunk_data);
        if (ctp::IsDevicePointer(chunk_data)) {
          staged.resize(chunk_size);
          ctp::DeviceAwareMemcpy(staged.data(), chunk_data, chunk_size);
          p = reinterpret_cast<const unsigned char *>(staged.data());
        }
        checksum = 14695981039346656037ull;  // 0xcbf29ce484222325
        for (size_t k = 0; k < chunk_size; ++k) {
          checksum ^= p[k];
          checksum *= 1099511628211ull;
        }
      }
      LogNeuroPressSelection(task->blob_name_.str(), chunk_size, sel_entropy,
                             sel_mad, sel_second_deriv, best_lib, best_preset,
                             logged_pred, context.actual_compression_ratio_,
                             context.actual_compress_time_ms_,
                             context.actual_psnr_db_, checksum);
    }

    // NeuroPress's online-learning loop, ported. Both phases below share one
    // error_pct (weighted-cost MAPE between what was predicted for the
    // algorithm actually used and what really happened) and differ only in
    // which threshold gates them. actual_compression_ratio_ <= 0 means the
    // codec produced no measurable result, so there is no ground truth to
    // train on.
    // Best mode reaches this block WITHOUT online learning. Exploration is
    // nested inside it, so gating purely on the learning switch made
    // `--best` silently degrade to plain inference: Create() turned
    // exploration on and pinned K to 31, the per-chunk path never looked,
    // and the run finished in inference time reporting an inference result.
    // Both SGD phases stay suppressed under best mode by their own
    // `!neuropress_best_mode_` guards below, so admitting it here buys the
    // exhaustive sweep and no training -- which is what best mode claims to
    // be. Mirror of e28e9315, which made exploration reachable without best
    // mode; this makes best mode reachable without learning.

    // Recorded on EVERY chunk NeuroPress selected for, not only when learning
    // is on -- upstream's recordChunkDiagnostic is likewise ungated
    // (gpucompress_compress.cpp), and a diagnostic that only exists while the
    // thing it diagnoses is enabled cannot serve as its control. The
    // cost-model fields are filled in by the learning block below.
    if (NeuroPressActive(context) && !stats.empty()) {
      const std::string np_name =
          ctp::CompressionFactory::NameForWireId(best_lib);
      int np_base = -1;
      for (const auto &e : ctp::compress::model::KnownCompressors()) {
        if (np_name == e.name) { np_base = e.base_id; break; }
      }
      if (np_base >= 0) {
        const uint32_t np_preset = static_cast<uint32_t>(best_preset);
        NeuroPressChunkDiag diag;
        diag.nn_original_action = ctp::compress::model::NeuroPressActionId(
            np_base, UnpackQuantEnabled(np_preset), UnpackShuffle(np_preset) != 0);
        diag.nn_action = diag.nn_original_action;
        // Upstream ties feat_action to nn_original_action
        // (gpucompress_diagnostics.cpp).
        diag.feat_action = diag.nn_original_action;
        // The selection's own features, so the record cannot disagree with
        // what was predicted from.
        diag.feat_entropy = static_cast<float>(sel_entropy);
        diag.feat_mad = static_cast<float>(sel_mad);
        diag.feat_deriv = static_cast<float>(sel_second_deriv);
        diag.feat_eb_enc = static_cast<float>(context.error_bound_);
        diag.feat_ds_enc = static_cast<float>(chunk_size);
        diag.predicted_ratio = static_cast<float>(stats.front().compression_ratio_);
        diag.actual_ratio = static_cast<float>(context.actual_compression_ratio_);
        diag.predicted_comp_time =
            static_cast<float>(stats.front().compress_time_ms_);
        diag.predicted_decomp_time =
            static_cast<float>(stats.front().decompress_time_ms_);
        diag.compression_ms =
            static_cast<float>(context.actual_compress_time_ms_);
        diag.decompression_ms = 0.0f;  // only a later read can fill this
        // One MAPE per predicted metric, plus the cost error, exactly as
        // upstream derives them (gpucompress_compress.cpp): clamp first --
        // times floored at 1 ms, ratio capped at 100x -- then compare.
        // Computed here rather than on the learning path so a frozen-weights
        // run has an accuracy to compare against.
        {
          const auto &f = stats.front();
          // Same ceiling the cost model and the kernel used; a hardcoded 100
          // here would report a MAPE against a differently-clamped ratio.
          const double kMapeCap = NeuroPressResolvedCostWeights().cap;
          const double pred_r = std::min(kMapeCap, f.compression_ratio_);
          const double pred_ct = std::max(1.0, f.compress_time_ms_);
          const double pred_dt = std::max(1.0, f.decompress_time_ms_);
          const double act_r = std::min(kMapeCap, context.actual_compression_ratio_);
          const double act_ct = std::max(1.0, context.actual_compress_time_ms_);
          // Decompression is not measured at write time; upstream substitutes
          // the prediction, which makes this term contribute nothing rather
          // than penalizing an unmeasured value.
          const double act_dt = pred_dt;
          diag.ratio_mape = static_cast<float>(
              (act_r > 0.0) ? std::fabs(act_r - pred_r) / act_r : 0.0);
          diag.comp_time_mape = static_cast<float>(
              (act_ct > 0.0) ? std::fabs(act_ct - pred_ct) / act_ct : 0.0);
          diag.decomp_time_mape = static_cast<float>(
              (act_dt > 0.0) ? std::fabs(act_dt - pred_dt) / act_dt : 0.0);

          // Same resolved weights the gate uses, so the reported cost and
          // the gated cost cannot disagree.
          const auto cw = NeuroPressResolvedCostWeights();
          const double ds = static_cast<double>(chunk_size);
          const double a_cost = cw.ct * act_ct + cw.dt * act_dt +
              ((act_r > 0.0) ? cw.io * ds / (act_r * cw.bw) : 0.0);
          const double p_cost = cw.ct * pred_ct + cw.dt * pred_dt +
              ((pred_r > 0.0) ? cw.io * ds / (pred_r * cw.bw) : 0.0);
          diag.actual_cost = static_cast<float>(a_cost);
          diag.predicted_cost = static_cast<float>(p_cost);
          diag.cost_model_error_pct = static_cast<float>(
              (a_cost > 0.0) ? std::fabs(a_cost - p_cost) / a_cost : 0.0);
        }
        // `stats` IS the ranking, so ids are read off it.
        int nrank = 0;
        for (const auto &r : stats) {
          if (nrank >= kNeuroPressRankingSlots) break;
          const std::string rn =
              ctp::CompressionFactory::NameForWireId(r.compress_lib_);
          int rb = -1;
          for (const auto &e : ctp::compress::model::KnownCompressors()) {
            if (rn == e.name) { rb = e.base_id; break; }
          }
          if (rb < 0) continue;
          const uint32_t rp = static_cast<uint32_t>(r.compress_preset_);
          diag.predicted_ranking[nrank++] =
              ctp::compress::model::NeuroPressActionId(
                  rb, UnpackQuantEnabled(rp), UnpackShuffle(rp) != 0);
        }
        diag.predicted_ranking_count = nrank;
        np_diag_slot = NeuroPressRecordChunkDiag(diag);
      }
    }
    if ((config_.neuropress_online_learning_enabled_ ||
         config_.neuropress_best_mode_) &&
        neuropress_feat_valid &&
        task->return_code_ == 0 && context.actual_compression_ratio_ > 0.0) {
      const CompressionStats* predicted = nullptr;
      for (const auto& stat : stats) {
        if (stat.compress_lib_ == best_lib &&
            stat.compress_preset_ == best_preset) {
          predicted = &stat;
          break;
        }
      }
      if (predicted) {
        // cost = w0*compress_time + w1*decompress_time +
        //        w2*chunk_size/(ratio*bandwidth) -- same formula and default
        // weights/bandwidth as NeuroPress's own g_rank_w0/w1/w2 (all 1.0) and
        // g_measured_bw_bytes_per_ms (5e6 = 5 GB/s).
        // Best mode ranks on RATIO ALONE. gpucompress_set_best_mode() zeroes
        // g_rank_w0/w1 and leaves w2 at 1.0 (gpucompress_learning.cpp), which
        // collapses the cost to size/(ratio*bw) -- so "best" means the
        // smallest output, not the fastest path to it. Leaving the latency
        // weights in would make the exhaustive search report a ceiling for a
        // different objective than the one it claims to measure.
        // The SAME weights the ranking used, so the model is gated on the
        // objective it was ranked by. Upstream keeps one set of globals for
        // both (g_rank_w0/w1/w2 feed nnForwardPass AND the cost at
        // gpucompress_compress.cpp:662-664); Clio had the
        // CLIO_NEUROPRESS_COST_W_* override reach the ranking only, so a
        // "ratio cost model" run selected on ratio while SGD still scored the
        // balanced cost.
        const auto kCw = NeuroPressResolvedCostWeights();
        // Best mode scores I/O alone: it ranks on what a configuration
        // SAVES, not what it costs to get there, so the two time weights
        // drop out rather than being weighed against bytes.
        const NeuroPressCost cost{config_.neuropress_best_mode_ ? 0.0 : kCw.ct,
                                  config_.neuropress_best_mode_ ? 0.0 : kCw.dt,
                                  kCw.io, kCw.bw, kCw.cap, chunk_size};
        // Decompress time is not measured at write time (only a later read
        // decompresses it) -- use the prediction for both sides, same as
        // NeuroPress's own primary_decomp_time_ms fallback, so this term
        // contributes ~0 error rather than penalizing an unmeasured value.
        double predicted_cost = cost(predicted->compress_time_ms_,
                                     predicted->decompress_time_ms_,
                                     predicted->compression_ratio_);
        double actual_cost = cost(context.actual_compress_time_ms_,
                                  predicted->decompress_time_ms_,
                                  context.actual_compression_ratio_);
        double error_pct = (actual_cost > 0.0)
            ? std::fabs(actual_cost - predicted_cost) / actual_cost
            : 0.0;

        // ---- Phase 1: "learn from PRIMARY result immediately" -- online
        // SGD on the real, just-measured outcome for the algorithm that was
        // ACTUALLY used, gated by g_reinforce_mape_threshold / default 30%.
        // In-memory only: Train() adjusts neuropress_predictor_'s live
        // weights for this process's lifetime and never calls Save() -- the
        // .nnwt file on disk stays untouched, exactly like every NeuroPress
        // runtime API (only gpucompress_load_nn/_reload_nn touch the file,
        // and both are read-only).
        std::string lib_name =
            ctp::CompressionFactory::NameForWireId(best_lib);
        int base_id = -1;
        for (const auto& entry : ctp::compress::model::KnownCompressors()) {
          if (lib_name == entry.name) {
            base_id = entry.base_id;
            break;
          }
        }
        if (base_id >= 0) {
          ctp::compress::model::DataFeatures data;
          data.chunk_size_bytes = static_cast<double>(chunk_size);
          data.shannon_entropy = neuropress_entropy;
          data.mad = neuropress_mad;
          data.second_derivative_mean = neuropress_second_deriv;
          data.data_type_char = (context.data_type_ == 1) ? 0.0 : 1.0;
          data.data_type_float = (context.data_type_ == 1) ? 1.0 : 0.0;

          // best_preset is PACKED (preset | shuffle_elem << 8). Feeding it
          // raw into preset_id corrupts LibraryConfigId() = base_id*10 +
          // preset_id, which is how the model recovers the algorithm: a
          // shuffled lz4 (2 | 4<<8 = 1026) yields 13*10 + 1026 = 1156, and
          // FeaturesTo8Input's 1156/10 = 115 misses every known base_id and
          // falls back to 115 % 8 = 3 (gdeflate). Every shuffled algorithm
          // collapsed onto two indices, so SGD trained the wrong head.
          // NeuroPress has no such hazard -- it carries the 0-31 action id
          // unchanged from selection into SGD (gpucompress_compress.cpp).
          ctp::compress::model::CandidateConfig candidate;
          candidate.base_id = base_id;
          candidate.preset_id =
              static_cast<int>(UnpackPreset(static_cast<uint32_t>(best_preset)));
          candidate.byte_shuffle =
              UnpackShuffle(static_cast<uint32_t>(best_preset)) != 0;
          // Both default to false/0 into FeaturesTo8Input slots 1 and 3, so
          // leaving them trained the lossless row at a bound of zero.
          candidate.quantize =
              UnpackQuantEnabled(static_cast<uint32_t>(best_preset));
          // Unconditional, not `quantize ? eb : 0`: nnSGDKernel writes
          // raw[3] with no quant test, and the 1e-7 sentinel is
          // inference-only (nn_gpu.cu).
          candidate.error_bound = context.error_bound_;
          candidate.library_name = lib_name;

          ctp::compress::model::CompressionFeatures chunk_features =
              ctp::compress::model::MakeCompressionFeatures(data, candidate);

          // Stash for the deferred decomp-head pass. Recorded on EVERY
          // compress, not just when the MAPE gate below trips: any blob may
          // be read back later, and that read is the only place a real
          // decompression time ever becomes available.
          RecordDecompFeatures(task->blob_name_.str(), chunk_features);


          // Withheld under best mode: that mode replaces the model's choice on
          // every chunk, so training on the outcome would teach the network
          // from a decision it did not make (upstream's
          // `&& !g_best_mode.load()`, gpucompress_compress.cpp).
          // EXPERIMENTAL, OFF BY DEFAULT -- a deliberate divergence from
          // upstream, which gates on the cost alone (error_pct, :705
          // gpucompress_compress.cpp) and never on a per-statistic MAPE.
          //
          // Why it exists: with balanced weights at 4 MiB the io term is
          // ~0.4% of the cost, so a ratio prediction can be wrong by 4-5x
          // while the cost error stays under the gate and the ratio head is
          // never corrected. Setting CLIO_NEUROPRESS_SGD_ON_RATIO=1 ORs a
          // ratio-MAPE gate onto the cost gate so that head trains too.
          // CLIO_NEUROPRESS_RATIO_MAPE_THRESH overrides its threshold,
          // defaulting to the same value as the cost gate.
          //
          // Read once: this is a per-chunk path on runtime worker threads,
          // where a getenv per call is both a syscall and a data race.
          struct RatioGateCfg { bool on; double thresh; };
          static const RatioGateCfg kRatioGate = [] {
            RatioGateCfg c{false, -1.0};
            const char *e = std::getenv("CLIO_NEUROPRESS_SGD_ON_RATIO");
            c.on = (e != nullptr && e[0] == '1');
            const char *t = std::getenv("CLIO_NEUROPRESS_RATIO_MAPE_THRESH");
            if (t && *t) {
              char *end = nullptr;
              const double v = std::strtod(t, &end);
              if (end != t && v >= 0.0) c.thresh = v;
            }
            return c;
          }();

          const double np_cost_thresh =
              static_cast<double>(config_.neuropress_mape_threshold_);
          const bool np_cost_gate = error_pct > np_cost_thresh;

          // Same clamps the reported ratio_mape uses, so the gate and the
          // column agree.
          double np_ratio_mape = 0.0;
          {
            const double cap = NeuroPressResolvedCostWeights().cap;
            const double pr = std::min(cap, predicted->compression_ratio_);
            const double ar = std::min(cap, context.actual_compression_ratio_);
            if (ar > 0.0) np_ratio_mape = std::fabs(ar - pr) / ar;
          }
          const bool np_ratio_gate =
              kRatioGate.on &&
              np_ratio_mape > (kRatioGate.thresh >= 0.0 ? kRatioGate.thresh
                                                        : np_cost_thresh);

          const bool np_will_train =
              (np_cost_gate || np_ratio_gate) &&
              !config_.neuropress_best_mode_;
          NeuroPressUpdateChunkDiagSgd(np_diag_slot, np_will_train);
          if (np_will_train) {
            std::vector<ctp::compress::model::CompressionFeatures> features = {
                chunk_features};
            // PSNR label for the PRIMARY sample. A lossless primary trains
            // toward 120 dB, not the -1.0 that withholds the gradient.
            //
            // Upstream is asymmetric here, and the asymmetry is deliberate on
            // both sides:
            //   primary  -- seeded to a literal 120.0 and overwritten only
            //               when quantization ran (gpucompress_compress.cpp:
            //               `explored_samples.push_back({..., 120.0, ...})`
            //               at :697-700, read into primary_sgd[0].actual_psnr
            //               at :711).
            //   explored -- left at -1.0 for a lossless slot, with upstream's
            //               own note that 0.0 cannot be used because the
            //               kernel remaps <=0 to 120 (:895-899). Clio matches
            //               that separately, at the explore_labels site.
            //
            // context.actual_psnr_db_ stays -1.0 and is NOT reused here: it
            // feeds LogNeuroPressSelection above, which is upstream's
            // DIAGNOSTICS value (`primary_actual_psnr`, :388, reported as
            // di.actual_psnr at :1100) -- a different variable from the SGD
            // sample's. Conflating the two was the defect: the PSNR head's
            // error backpropagates through every shared layer, so withholding
            // it moved 5513 of 13576 weights per step relative to upstream.
            // Upstream's net effect, written as one expression: the sample is
            // seeded 120.0 and replaced ONLY by a valid positive analytical
            // PSNR, so `>0 ? it : 120.0` is the same function. Reading the
            // sign of actual_psnr_db_ rather than a quantize flag also keeps
            // the degenerate-range case right -- AnalyticalPsnr returns -1 for
            // a zero range, and upstream's `if (psnr > 0.0)` leaves 120.0
            // standing in exactly that case too.
            const double primary_sgd_psnr =
                (context.actual_psnr_db_ > 0.0) ? context.actual_psnr_db_
                                                : 120.0;
            std::vector<ctp::compress::model::TrainingLabels> labels = {
                ctp::compress::model::TrainingLabels(
                    static_cast<float>(context.actual_compression_ratio_),
                    static_cast<float>(primary_sgd_psnr),
                    static_cast<float>(context.actual_compress_time_ms_),
                    /*decompress_time=*/0.0f)};

            // Device-resident statistics when the selection had them, so the
            // SGD kernel reads entropy/MAD/second-derivative on-device exactly
            // as nnSGDKernel does. Null falls back to the host fields already
            // in `features`.
            bool trained = neuropress_predictor_->TrainDeviceStats(
                features, labels, sel_device_stats);
            HLOG(kDebug,
                 "NeuroPress SGD: lib={} preset={} error_pct={} "
                 "threshold={} trained={}",
                 lib_name, best_preset, error_pct,
                 config_.neuropress_mape_threshold_, trained);
          }
        }

        // ---- Phase 2: learn from exploration results separately. When error
        // crosses the higher exploration threshold, compress the SAME chunk
        // with up to K alternative candidates (next-best predicted, skipping
        // the one already used). This generates real-outcome training samples,
        // and any alternative cheaper than the primary REPLACES the stored
        // result (see the adoption block after the loop). Off by default.
        //
        // Each candidate runs on its OWN CUDA stream, as upstream's
        // ExploreSlot does: the slot is opened before its preprocessing so
        // quantize -> shuffle -> compress queue as one dependent chain that
        // never waits, and all K are collected together in phase 2 below (see
        // the OpenSlot/CompressLaunch/CompressFinish notes there). The
        // concurrency is the device's, not the host's -- one thread, K
        // streams. Measured at K=8 on 4 MiB chunks: 11.10 ms of summed
        // per-slot kernel time inside 1.85 ms of wall time, against 9.24 ms
        // for the same work through Compress(), which takes the thread's
        // single cached stream and synchronizes before returning.
        if (config_.neuropress_exploration_enabled_ &&
            (config_.neuropress_best_mode_ ||
             error_pct > static_cast<double>(
                             config_.neuropress_exploration_threshold_))) {
          // K bounds the RANKED WINDOW scanned, not the number measured, and
          // an ineligible slot inside that window is dropped rather than
          // replaced from further down. Both are upstream's
          // (gpucompress_compress.cpp): it walks top_actions[1..K] and
          // `continue`s past a slot it cannot run, so exploring K slots can
          // yield fewer than K samples. Collecting K ELIGIBLE alternatives
          // instead would explore configurations the model ranked below the
          // window upstream looks at.
          //
          // The quantize skip is what makes that matter here. On the device
          // path all 32 candidates reach `stats` -- RankKernel masks the
          // quantize ones to -INFINITY on a lossless run, which sinks them to
          // the bottom but leaves them selectable -- so without this skip a
          // lossless run explores each masked action as its plain lossless
          // twin, measuring all 16 real configurations TWICE and feeding the
          // duplicates to the phase-2 SGD batch as if they were independent
          // samples.
          // Positions are counted rather than indexed from 1: upstream can
          // assume top_actions[0] is the primary, but here the primary is
          // stats.front() only on the cost-ranked path -- the legacy
          // heuristics can pick from anywhere in the list.
          std::vector<const CompressionStats*> alternatives;
          int examined = 0;
          for (const auto& stat : stats) {
            if (stat.compress_lib_ == best_lib &&
                stat.compress_preset_ == best_preset) {
              continue;
            }
            if (examined >= config_.neuropress_exploration_k_) break;
            ++examined;
            if (UnpackQuantEnabled(
                    static_cast<uint32_t>(stat.compress_preset_)) &&
                !(context.error_bound_ > 0.0)) {
              continue;
            }
            alternatives.push_back(&stat);
          }

          std::vector<ctp::compress::model::CompressionFeatures>
              explore_features;
          std::vector<ctp::compress::model::TrainingLabels> explore_labels;
          // Per-sample cost, kept parallel to the two vectors above purely so
          // the batch can be ordered and truncated the way upstream's SGD
          // phase 2 does before training. See the sort below.
          std::vector<double> explore_costs;
          double best_cost = actual_cost;  // seeded with the primary's own

          // Device shuffle scratch for the explored candidates; released
          // together once the loop is done.
          std::vector<ctp::ipc::AllocatorId> explore_gpu_scratch;

          // Best explored alternative so far, if any beat the primary.
          // Upstream rewrites the output buffer in place each time it finds
          // a cheaper action (gpucompress_compress.cpp+ "Write winner to
          // output"), so successive winners simply overwrite one another and
          // the final buffer holds the cheapest. Clio's blob is already
          // persisted by AsyncCompress before exploration runs, so the same
          // observable outcome is reached by remembering the best and
          // re-storing once after the loop -- upstream's intermediate writes
          // land in a local buffer nothing can observe.
          ExploreWinner winner;

          std::vector<std::unique_ptr<ExploreSlot>> slots;
          slots.reserve(alternatives.size());

          // The three steps below are the SWEEP's, not the two
          // learning phases' -- this whole sweep runs inside learning
          // Phase 2. They are numbered on their own axis so a
          // reference to "Phase 2" keeps meaning upstream's.
          // ---- Sweep step 1: prepare every slot (serial) ----
          for (const auto* alt : alternatives) {
            std::string alt_name =
                ctp::CompressionFactory::NameForWireId(alt->compress_lib_);
            // Packed, like every other compress_preset_ that came out of
            // the ranking -- unpack before comparing, or a shuffled FAST
            // alternative (1 | 4<<8 = 1025) silently reads as BALANCED.
            const uint32_t alt_preset_id =
                UnpackPreset(static_cast<uint32_t>(alt->compress_preset_));
            const bool alt_wants_quant =
                UnpackQuantEnabled(static_cast<uint32_t>(alt->compress_preset_));
            const uint32_t alt_shuffle =
                UnpackShuffle(static_cast<uint32_t>(alt->compress_preset_));
            ctp::CompressionPreset alt_preset =
                ctp::CompressionPreset::BALANCED;
            if (alt_preset_id == 1) {
              alt_preset = ctp::CompressionPreset::FAST;
            } else if (alt_preset_id == 3) {
              alt_preset = ctp::CompressionPreset::BEST;
            }
            auto alt_compressor =
                ctp::CompressionFactory::GetPreset(alt_name, alt_preset);
            if (!alt_compressor) continue;

            // The slot -- and therefore its stream -- has to exist BEFORE the
            // preprocessing below, because quantize and shuffle are queued on
            // that same stream so the whole per-slot chain (quantize ->
            // shuffle -> compress) runs without waiting on anything, exactly
            // as upstream queues it (gpucompress_compress.cpp).
            auto slot = std::make_unique<ExploreSlot>();
            void* alt_stream = nullptr;
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
            slot->gpu = dynamic_cast<ctp::NvComp*>(alt_compressor.get());
            if (slot->gpu != nullptr && ctp::NvComp::OpenSlot(&slot->async)) {
              alt_stream = slot->async.stream;
            } else {
              // No stream means the synchronous fallback below, so the
              // preprocessing must keep its own waiting behaviour.
              slot->gpu = nullptr;
            }
#endif

            // Same device-pointer safety net Runtime::Compress() uses: a
            // CPU-only alternative can't read a device pointer directly.
            std::vector<char> alt_device_staging;
            char* alt_input = ctp::CompressionFactory::StageInputIfNeeded(
                static_cast<char*>(chunk_data), chunk_size,
                alt->compress_lib_, alt_device_staging);

            // Apply the alternative's OWN quantization first, exactly as the
            // primary does and as upstream does per explored slot
            // (gpucompress_compress.cpp: quantize, update
            // alt_compress_size, THEN shuffle the quantized buffer).
            // Measuring an unquantized buffer while crediting a quantized
            // action would mislabel the sample -- the same defect the
            // shuffle handling below was fixed for.
            size_t alt_compress_size = chunk_size;
            bool alt_applied_quant = false;
            ctp::compress::preprocess::DeviceQuantizeParams alt_quant_params;
            // Same gates as the primary: preproc bit + positive bound, with
            // the buffer treated as float32 unconditionally (upstream's
            // gpucompress_compress.cpp and :827).
            const bool alt_want_quant =
                alt_wants_quant && context.error_bound_ > 0.0 &&
                chunk_size >= sizeof(float) &&
                (chunk_size % sizeof(float)) == 0;
            std::vector<char> alt_quant_staging;
            if (alt_want_quant && ctp::IsDevicePointer(alt_input)) {
              char* alt_q_buf = nullptr;
              ctp::ipc::AllocatorId alt_q_alloc =
                  CLIO_IPC->AllocateAndRegisterGpuBackend(
                      /*gpu_id=*/0,
                      clio::run::gpu::IpcManager::MemKind::kDeviceMem,
                      chunk_size, &alt_q_buf);
              size_t alt_q_bytes = 0;
              if (!alt_q_alloc.IsNull()) {
                explore_gpu_scratch.push_back(alt_q_alloc);
                if (ctp::compress::preprocess::QuantizeDevice(
                        alt_input, chunk_size / sizeof(float),
                        context.error_bound_, alt_q_buf, &alt_q_bytes,
                        &alt_quant_params, alt_stream)) {
                  alt_input = alt_q_buf;
                  alt_compress_size = alt_q_bytes;
                  alt_applied_quant = true;
                }
              }
            } else if (alt_want_quant) {
              // Host-resident alternative: honor the action here too, or the
              // sample is credited as quantized while measuring unquantized
              // bytes.
              auto alt_hq = ctp::compress::preprocess::Quantize<float>(
                  reinterpret_cast<const float*>(alt_input),
                  chunk_size / sizeof(float), context.error_bound_);
              if (!alt_hq.quantized_.empty()) {
                alt_quant_staging.assign(alt_hq.quantized_.begin(),
                                         alt_hq.quantized_.end());
                alt_input = alt_quant_staging.data();
                alt_compress_size = alt_quant_staging.size();
                alt_applied_quant = true;
                alt_quant_params.effective_error_bound =
                    alt_hq.effective_error_bound_;
                alt_quant_params.scale = alt_hq.scale_;
                alt_quant_params.data_min = alt_hq.data_min_;
                alt_quant_params.data_max = alt_hq.data_max_;
                alt_quant_params.precision = alt_hq.precision_;
                alt_quant_params.bound_achievable = alt_hq.bound_achievable_;
              }
            }

            // Apply the alternative's OWN byte-shuffle before measuring it.
            // Upstream reconstructs the full preprocessing for each explored
            // action. Compressing unshuffled bytes would measure a different
            // action than the one being credited, and the shuffle dimension
            // would never receive any exploration signal.
            std::vector<char> alt_shuffle_staging;
            uint32_t alt_applied_shuffle = 0;
            if (alt_shuffle != 0) {
              if (ctp::IsDevicePointer(alt_input)) {
                char* alt_shuf_buf = nullptr;
                ctp::ipc::AllocatorId alt_shuf_alloc =
                    CLIO_IPC->AllocateAndRegisterGpuBackend(
                        /*gpu_id=*/0,
                        clio::run::gpu::IpcManager::MemKind::kDeviceMem,
                        alt_compress_size, &alt_shuf_buf);
                if (!alt_shuf_alloc.IsNull()) {
                  if (ctp::compress::preprocess::ByteShuffleDevice(
                          alt_input, alt_shuf_buf, alt_compress_size,
                          alt_shuffle, alt_stream)) {
                    alt_input = alt_shuf_buf;
                    alt_applied_shuffle = alt_shuffle;
                  }
                  // Exploration output is never stored, so the scratch can be
                  // released as soon as this candidate is measured -- but the
                  // compress below still reads it, so free after that.
                  explore_gpu_scratch.push_back(alt_shuf_alloc);
                }
              } else {
                alt_shuffle_staging.resize(alt_compress_size);
                if (ctp::compress::preprocess::ByteShuffle(
                        reinterpret_cast<const uint8_t*>(alt_input),
                        alt_compress_size, alt_shuffle,
                        reinterpret_cast<uint8_t*>(
                            alt_shuffle_staging.data()))) {
                  alt_input = alt_shuffle_staging.data();
                  alt_applied_shuffle = alt_shuffle;
                }
              }
            }

            size_t alt_worst_case = alt_compress_size + (alt_compress_size / 20) + 1024;

            // Compress into DEVICE memory when the input is device-resident, as
            // upstream does. A host destination makes nvcomp stage the result
            // D2H inside the measured window, putting an explored sample's time
            // on a different scale from the primary's -- and the two are
            // compared directly, both for the winner and as training labels.
            std::vector<char> alt_output;
            char *alt_out_ptr = nullptr;
            ctp::ipc::AllocatorId alt_out_alloc;
            if (ctp::IsDevicePointer(alt_input)) {
              alt_out_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
                  /*gpu_id=*/0,
                  clio::run::gpu::IpcManager::MemKind::kDeviceMem,
                  alt_worst_case, &alt_out_ptr);
              if (!alt_out_alloc.IsNull()) {
                explore_gpu_scratch.push_back(alt_out_alloc);
              } else {
                alt_out_ptr = nullptr;
              }
            }
            if (!alt_out_ptr) {
              alt_output.resize(alt_worst_case);
              alt_out_ptr = alt_output.data();
            }
            slot->alt = alt;
            slot->name = alt_name;
            slot->preset_id = alt_preset_id;
            slot->applied_shuffle = alt_applied_shuffle;
            slot->applied_quant = alt_applied_quant;
            slot->quant_params = alt_quant_params;
            slot->compressor = std::move(alt_compressor);
            slot->device_staging = std::move(alt_device_staging);
            slot->quant_staging = std::move(alt_quant_staging);
            slot->shuffle_staging = std::move(alt_shuffle_staging);
            slot->output = std::move(alt_output);
            slot->input = alt_input;
            slot->compress_size = alt_compress_size;
            slot->out_ptr = alt_out_ptr;
            slot->capacity = alt_worst_case;

            // Launch on the slot's stream, behind its own preprocessing. This
            // returns as soon as the work is queued, so the next iteration's
            // quantize/shuffle overlaps this candidate's codec kernels.
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
            if (slot->gpu != nullptr) {
              slot->launched = slot->gpu->CompressLaunch(
                  slot->out_ptr, slot->capacity, slot->input,
                  slot->compress_size, &slot->async);
              if (!slot->launched) slot->gpu = nullptr;
            }
#endif
            slots.push_back(std::move(slot));
          }

          // ---- Sweep step 2: every slot is already in flight; collect it ----
          CollectExploreSlots(slots);

          std::vector<ExploreRow> explore_rows;
          int explore_rank = 0;

          // ---- Sweep step 3: score the slots in rank order (serial) ----
          for (auto& sp : slots) {
            ExploreSlot& slot_ref = *sp;
            const CompressionStats* alt = slot_ref.alt;
            const std::string& alt_name = slot_ref.name;
            const uint32_t alt_preset_id = slot_ref.preset_id;
            const uint32_t alt_applied_shuffle = slot_ref.applied_shuffle;
            const bool alt_applied_quant = slot_ref.applied_quant;
            const ctp::compress::preprocess::DeviceQuantizeParams&
                alt_quant_params = slot_ref.quant_params;
            char* alt_out_ptr = slot_ref.out_ptr;
            const bool alt_ok = slot_ref.ok;
            const size_t alt_compressed_size = slot_ref.compressed_size;
            const double alt_time_ms = slot_ref.time_ms;
            if (!alt_ok || alt_compressed_size == 0) continue;

            double alt_ratio = static_cast<double>(chunk_size) /
                               static_cast<double>(alt_compressed_size);
            // Decompression time is the PRIMARY's prediction, held constant
            // across every alternative -- upstream passes the same pred_dt
            // to compute_cost for each explored slot
            // (gpucompress_compress.cpp, "decomp_time uses NN
            // prediction (no decomp at write)"), because nothing is
            // decompressed at write time so no alternative has a measured
            // value either. Using each candidate's OWN predicted dt instead
            // let a difference the exploration never measured move the
            // ranking, and skewed the regret figure derived from it.
            double alt_cost =
                cost(alt_time_ms, predicted->decompress_time_ms_, alt_ratio);
            if (alt_cost < best_cost) {
              best_cost = alt_cost;
              // Adopt it. Upstream's only condition here is that the winner
              // fits the caller's output buffer; Clio additionally requires
              // it to still beat storing the bytes raw, because unlike
              // upstream Clio HAS a raw path (Compress()'s "not beneficial"
              // branch) and must not replace a raw blob with something
              // larger than the original.
              const size_t winner_total =
                  alt_compressed_size + sizeof(CompressionHeader);
              if (winner_total < chunk_size) {
                winner.have = true;
                // The row for THIS candidate has not been pushed yet, so its
                // index is the current size. A later winner overwrites this,
                // which is what makes only the final one carry the flag.
                winner.row = static_cast<int>(explore_rows.size());
                // Pull the winner's bytes back AFTER the measurement, so the
                // copy never lands inside the timed window.
                winner.payload.resize(alt_compressed_size);
                if (ctp::IsDevicePointer(alt_out_ptr)) {
                  ctp::DeviceAwareMemcpy(winner.payload.data(), alt_out_ptr,
                                         alt_compressed_size);
                } else {
                  std::memcpy(winner.payload.data(), alt_out_ptr,
                              alt_compressed_size);
                }
                winner.lib = alt->compress_lib_;
                winner.preset_id = alt_preset_id;
                winner.shuffle = alt_applied_shuffle;
                winner.ratio = alt_ratio;
                winner.time_ms = alt_time_ms;
                // Carry the quantization state too. Without it the adopted
                // blob's header would say "not quantized" while its payload
                // IS quantized -- unrecoverable on read.
                winner.quant = alt_applied_quant;
                winner.quant_params = alt_quant_params;
              }
            }

            if (ExploreLogEnabled()) {
              explore_rows.push_back(ExploreRow{
                  alt_name, alt_preset_id, alt_applied_quant,
                  alt_applied_shuffle, alt->compression_ratio_,
                  alt->compress_time_ms_, alt_ratio, alt_time_ms,
                  alt_applied_quant
                      ? AnalyticalPsnr(alt_quant_params.data_max -
                                           alt_quant_params.data_min,
                                       alt_quant_params.effective_error_bound)
                      : -1.0,
                  alt_cost, explore_rank});
            }
            ++explore_rank;

            int alt_base_id = -1;
            for (const auto& entry :
                ctp::compress::model::KnownCompressors()) {
              if (alt_name == entry.name) {
                alt_base_id = entry.base_id;
                break;
              }
            }
            if (alt_base_id < 0) continue;

            ctp::compress::model::DataFeatures alt_data;
            alt_data.chunk_size_bytes = static_cast<double>(chunk_size);
            alt_data.shannon_entropy = neuropress_entropy;
            alt_data.mad = neuropress_mad;
            alt_data.second_derivative_mean = neuropress_second_deriv;
            alt_data.data_type_char = (context.data_type_ == 1) ? 0.0 : 1.0;
            alt_data.data_type_float = (context.data_type_ == 1) ? 1.0 : 0.0;

            // Packed, same as the primary above -- unpack both halves.
            ctp::compress::model::CandidateConfig alt_candidate;
            alt_candidate.base_id = alt_base_id;
            alt_candidate.preset_id = static_cast<int>(
                UnpackPreset(static_cast<uint32_t>(alt->compress_preset_)));
            // Credit what was ACTUALLY applied: a declined shuffle (wrong
            // size multiple, failed allocation) means the measurement above
            // belongs to the unshuffled variant.
            alt_candidate.byte_shuffle = alt_applied_shuffle != 0;
            alt_candidate.quantize = alt_applied_quant;
            // The configured bound, not `applied ? eb : 0` -- see the primary
            // sample above. nnSGDKernel writes raw[3] with no quant test and
            // its caller passes cfg.error_bound for every explored sample
            // alike (gpucompress_compress.cpp).
            alt_candidate.error_bound = context.error_bound_;
            alt_candidate.library_name = alt_name;

            explore_features.push_back(
                ctp::compress::model::MakeCompressionFeatures(alt_data,
                                                               alt_candidate));
            // PSNR per explored slot, as upstream computes it
            // (gpucompress_compress.cpp, analytical_psnr from the slot's
            // own quantization range). -1 for a lossless alternative, which
            // makes the SGD withhold the PSNR gradient rather than train it
            // toward 120 dB.
            const double alt_psnr =
                alt_applied_quant
                    ? AnalyticalPsnr(
                          alt_quant_params.data_max - alt_quant_params.data_min,
                          alt_quant_params.effective_error_bound)
                    : -1.0;
            explore_labels.emplace_back(static_cast<float>(alt_ratio),
                                        static_cast<float>(alt_psnr),
                                        static_cast<float>(alt_time_ms),
                                        0.0f);
            explore_costs.push_back(alt_cost);
          }

          for (size_t ri = 0; ri < explore_rows.size(); ++ri) {
            const ExploreRow& row = explore_rows[ri];
            LogNeuroPressExplore(
                task->blob_name_.str(), chunk_size, row.rank, row.lib,
                row.preset_id, row.quant, row.shuffle, row.pred_ratio,
                row.pred_ct, row.ratio, row.ct_ms, row.psnr, row.cost,
                actual_cost, static_cast<int>(ri) == winner.row);
          }

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
          // Release each slot's stream, events and any temporary output. Done
          // AFTER scoring, because the winner's payload is copied out of
          // out_ptr up there and a slot that wrote into its own temporary
          // still owns those bytes until CompressFinish has delivered them.
          for (auto& sp : slots) {
            if (sp->gpu != nullptr) ctp::NvComp::ReleaseSlot(&sp->async);
          }
#endif
          slots.clear();

          for (const auto& scratch : explore_gpu_scratch) {
            if (!scratch.IsNull()) {
              CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, scratch);
            }
          }
          explore_gpu_scratch.clear();

          // ---- Adopt the winner (upstream's "Write winner to output") ----
          // When exploration finds a cheaper action than the one that was
          // actually used, upstream does NOT merely learn from it: it
          // replaces the stored result, rewriting the header (algorithm,
          // shuffle size, original and compressed sizes), the payload, the
          // reported output size, actual_ratio, algo_to_use and preproc_to_use
          // (gpucompress_compress.cpp). The exploration is a real
          // second chance at the write, not just a source of training data.
          //
          // Clio's primary blob is already persisted by the time we get here,
          // so adopting means storing the winner over it -- a whole-blob put
          // at the same offset, exactly what the primary itself did. The
          // Context is updated the same way upstream updates its diagnostics
          // state, so what the blob records matches what it now contains.
          if (winner.have) {
            const size_t hdr_size =
                sizeof(CompressionHeader) +
                (winner.quant ? sizeof(QuantHeaderExtension) : 0);
            const size_t winner_total = winner.payload.size() + hdr_size;
            CompressionHeader winner_header(
                static_cast<uint32_t>(winner.lib),
                PackPreset(winner.preset_id, winner.shuffle) |
                    PackQuant(winner.quant, winner.quant_params.precision),
                chunk_size, winner.payload.size());
            QuantHeaderExtension winner_ext{};
            if (winner.quant) {
              winner_ext.error_bound =
                  winner.quant_params.effective_error_bound;
              winner_ext.scale = winner.quant_params.scale;
              winner_ext.data_min = winner.quant_params.data_min;
              winner_ext.data_max = winner.quant_params.data_max;
            }

            auto winner_shm = CLIO_IPC->AllocateBuffer(winner_total);
            if (winner_shm.IsNull()) {
              HLOG(kWarning,
                   "NeuroPress explore: winner found but SHM allocation "
                   "failed; keeping the primary's stored result");
            } else {
              std::memcpy(winner_shm.ptr_, &winner_header,
                          sizeof(CompressionHeader));
              if (winner.quant) {
                std::memcpy(winner_shm.ptr_ + sizeof(CompressionHeader),
                            &winner_ext, sizeof(winner_ext));
              }
              std::memcpy(winner_shm.ptr_ + hdr_size, winner.payload.data(),
                          winner.payload.size());

              Context winner_ctx = context;
              winner_ctx.compress_lib_ = winner.lib;
              winner_ctx.compress_preset_ = static_cast<int>(
                  PackPreset(winner.preset_id, winner.shuffle) |
                  PackQuant(winner.quant, winner.quant_params.precision));
              winner_ctx.transform_flags_ |=
                  clio::cte::core::kBlobTransformed |
                  clio::cte::core::kBlobTransformCompressed;

              auto winner_put = core_client_->AsyncPutBlob(
                  task->tag_id_, task->blob_name_.str(), task->offset_,
                  winner_total, winner_shm.shm_.template Cast<void>(),
                  task->score_, winner_ctx, task->flags_,
                  clio::run::PoolQuery::Local());
              CLIO_CO_AWAIT(winner_put);
              const int winner_rc = winner_put->return_code_;
              CLIO_IPC->FreeBuffer(winner_shm);

              if (winner_rc != 0) {
                // The primary's blob is still intact -- a failed overwrite
                // leaves the earlier whole-blob put in place -- so this is a
                // missed improvement, not a lost blob.
                HLOG(kWarning,
                     "NeuroPress explore: winner put failed (rc={}); keeping "
                     "the primary's stored result",
                     winner_rc);
              } else {
                stored_by_exploration = true;
                // Supersede the primary's payload-log row. That row was
                // written from Compress() before this sweep existed, so
                // leaving it alone reports a chunk at the size the PRIMARY
                // produced while the tier holds the winner's -- always
                // larger, and only ever for the modes that explore, so a
                // comparison built on it ranks exploration below
                // configurations it actually beats. Last row for a blob wins.
                if (SelectionLogEnabled()) {
                  LogCompressedPayload(
                      task->blob_name_.str(), winner.payload.data(),
                      winner.payload.size(), /*on_device=*/false,
                      winner_total < chunk_size, winner.time_ms, "adopted");
                }
                task->context_ = winner_ctx;
                task->context_.actual_original_size_ = chunk_size;
                task->context_.actual_compressed_size_ = winner_total;
                task->context_.actual_compression_ratio_ = winner.ratio;
                task->context_.actual_compress_time_ms_ = winner.time_ms;
                // The stored bytes are now the winner's, so the features the
                // deferred decomp head will join a later read against must
                // describe the winner, not the primary.
                ctp::compress::model::DataFeatures win_data;
                win_data.chunk_size_bytes = static_cast<double>(chunk_size);
                win_data.shannon_entropy = neuropress_entropy;
                win_data.mad = neuropress_mad;
                win_data.second_derivative_mean = neuropress_second_deriv;
                win_data.data_type_char = (context.data_type_ == 1) ? 0.0 : 1.0;
                win_data.data_type_float = (context.data_type_ == 1) ? 1.0 : 0.0;
                int win_base_id = -1;
                std::string win_name =
                    ctp::CompressionFactory::NameForWireId(winner.lib);
                for (const auto& entry :
                     ctp::compress::model::KnownCompressors()) {
                  if (win_name == entry.name) {
                    win_base_id = entry.base_id;
                    break;
                  }
                }
                if (win_base_id >= 0) {
                  ctp::compress::model::CandidateConfig win_candidate;
                  win_candidate.base_id = win_base_id;
                  win_candidate.preset_id = static_cast<int>(winner.preset_id);
                  win_candidate.byte_shuffle = winner.shuffle != 0;
                  win_candidate.quantize = winner.quant;
                  // Feeds the deferred decomp head, whose upstream counterpart
                  // reads DeferredDecompSample::error_bound_enc from the
                  // diagnostics field `feat_eb_enc = d.error_bound`, itself
                  // set from cfg.error_bound unconditionally
                  // (gpucompress_diagnostics.cpp, gpucompress_compress.cpp).
                  win_candidate.error_bound = context.error_bound_;
                  win_candidate.library_name = win_name;
                  RecordDecompFeatures(
                      task->blob_name_.str(),
                      ctp::compress::model::MakeCompressionFeatures(
                          win_data, win_candidate));
                }
                HLOG(kDebug,
                     "NeuroPress explore: adopted {} (ratio={} time={}ms "
                     "quant={} shuffle={}) over the primary",
                     win_name, winner.ratio, winner.time_ms,
                     winner.quant ? winner.quant_params.precision : 0,
                     winner.shuffle);
              }
            }
          }

          // Withheld under best mode, same reason phase 1 is: these samples
          // come from a sweep the model did not choose, so training on them
          // contaminates the weights with outcomes it never predicted
          // (upstream's `&& !g_best_mode.load()`).
          if (!explore_features.empty() && !config_.neuropress_best_mode_) {
            // Order by ascending cost and keep at most the cheapest 7, matching
            // upstream's SGD phase 2. Its index 0 is the primary, which phase 1
            // already learned from, so alternatives cap at NN_MAX_SGD_SAMPLES-1.
            // The cap is architectural: upstream's sample array is fixed-size
            // and its kernel accepts no more. Clio's alternatives are already
            // primary-free, so the whole vector is eligible. SGD accumulates
            // over the batch, so this ordering is part of the result.
            constexpr size_t kMaxExploreSgdSamples = 7;  // NN_MAX_SGD_SAMPLES-1
            std::vector<size_t> order(explore_features.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(),
                             [&](size_t a, size_t b) {
                               return explore_costs[a] < explore_costs[b];
                             });
            if (order.size() > kMaxExploreSgdSamples) {
              order.resize(kMaxExploreSgdSamples);
            }
            std::vector<ctp::compress::model::CompressionFeatures> sorted_feats;
            std::vector<ctp::compress::model::TrainingLabels> sorted_labels;
            sorted_feats.reserve(order.size());
            sorted_labels.reserve(order.size());
            for (size_t i : order) {
              sorted_feats.push_back(explore_features[i]);
              sorted_labels.push_back(explore_labels[i]);
            }
            explore_features.swap(sorted_feats);
            explore_labels.swap(sorted_labels);

            // Same chunk, so the same device statistics -- exploration varies
            // the ACTION, not the data. Upstream's phase-2 SGD reuses its
            // d_stats_ptr across the explored samples for the same reason
            // (gpucompress_compress.cpp).
            bool explore_trained = neuropress_predictor_->TrainDeviceStats(
                explore_features, explore_labels, sel_device_stats);
            // Regret: how much worse the primary's real cost was than the
            // best alternative found. 0 if the primary was already best.
            double regret = (best_cost > 0.0)
                ? (actual_cost - best_cost) / best_cost
                : 0.0;
            HLOG(kDebug,
                 "NeuroPress explore: k={} error_pct={} threshold={} "
                 "trained={} regret={}",
                 explore_features.size(), error_pct,
                 config_.neuropress_exploration_threshold_, explore_trained,
                 regret);

            // `context` holds the adopted winner: what was written.
            if (np_diag_slot >= 0) {
              int final_action = -1;
              const std::string fn =
                  ctp::CompressionFactory::NameForWireId(context.compress_lib_);
              for (const auto &entry :
                   ctp::compress::model::KnownCompressors()) {
                if (fn == entry.name) {
                  const uint32_t fp =
                      static_cast<uint32_t>(context.compress_preset_);
                  final_action = ctp::compress::model::NeuroPressActionId(
                      entry.base_id, UnpackQuantEnabled(fp),
                      UnpackShuffle(fp) != 0);
                  break;
                }
              }
              NeuroPressUpdateChunkDiagExploration(
                  np_diag_slot, final_action, /*triggered=*/true,
                  static_cast<float>(regret));
            }
          }
        }
      }
    }

#ifdef CLIO_NEUROPRESS_PATH_TRACE
    // The decision that actually reaches storage. `context` is a reference to
    // task->context_, and an adopted exploration winner is committed by
    // assigning task->context_ wholesale -- so reading it here, after the
    // exploration block, is the only place that agrees with the CTEC header
    // the chunk is written with.
    {
      const int np_final_lib = context.compress_lib_;
      const int np_final_preset = context.compress_preset_;
      // Blob name included so this can be joined per chunk against the codec
      // the READ side recovers from the stored CTEC header. Aggregate counts
      // can agree while individual assignments differ.
      CLIO_PATH_TRACE(
          "WRITE  neuropress FINAL blob='%s' lib=%d (%s) preset=%d %s",
          task->blob_name_.str().c_str(),
          np_final_lib,
          ctp::CompressionFactory::NameForWireId(np_final_lib).c_str(),
          np_final_preset,
          (np_final_lib == np_primary_lib && np_final_preset == np_primary_preset)
              ? "(primary kept)"
              : "(EXPLORATION OVERRODE THE PRIMARY)");
    }
#endif

    // The one put, when exploration did not make it. Reaching here with a
    // deferred image means either exploration never ran for this chunk (below
    // its threshold) or it ran and nothing beat the primary -- in both cases
    // the primary's bytes are the answer and nothing has stored them yet.
    if (defer_store && !stored_by_exploration && primary_image.valid()) {
      auto primary_put = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_,
          primary_image.size, primary_image.data, task->score_,
          task->context_, task->flags_, clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(primary_put);
      if (primary_put->return_code_ != 0) {
        // Nothing else stored this blob, so unlike a failed exploration put
        // there is no earlier copy to fall back on: the chunk is simply not
        // in the tier and the caller has to know.
        HLOG(kError,
             "DynamicSchedule: deferred store of '{}' failed (rc={}); the "
             "chunk was compressed but is NOT in the tier",
             task->blob_name_.str(), primary_put->return_code_);
        task->return_code_ = primary_put->return_code_;
      }
    }
    primary_image.release();

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

    // Initialize core client if needed (from compose next_pool_id or task param)
    if (!core_client_) {
      clio::run::PoolId core_id = !config_.next_pool_id_.IsNull()
          ? config_.next_pool_id_ : task->core_pool_id_;
      if (!core_id.IsNull()) {
        core_client_ = std::make_unique<clio::cte::core::Client>(core_id);
      }
    }
    // Neither this chimod's own compose config nor the caller supplied a
    // core pool to store into -- every use of core_client_ below would
    // dereference a null unique_ptr. Fail the task instead of crashing the
    // whole runtime process on what is really a caller/deployment
    // misconfiguration (e.g. a compressor pool created without
    // CompressorConfig.next_pool_id_, called via the AsyncPutBlob
    // convenience override instead of AsyncDynamicSchedule/AsyncCompress'
    // explicit core_pool_id parameter).
    if (!core_client_) {
      HLOG(kError,
           "Compress: no core pool available (compose next_pool_id_ unset "
           "and caller passed no explicit core_pool_id) -- cannot store "
           "the result");
      task->return_code_ = 7;  // No core pool available
      CLIO_CO_RETURN;
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
          clio::run::PoolQuery::Local());
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
    // compress_preset_ carries the byte-shuffle element size in its high
    // bits (see PackPreset) -- unpack before mapping to the preset enum, or
    // a shuffled candidate reads as a bogus preset.
    const uint32_t packed_preset =
        static_cast<uint32_t>(context.compress_preset_);
    const uint32_t preset_id = UnpackPreset(packed_preset);
    const uint32_t shuffle_elem = UnpackShuffle(packed_preset);
    // The selector's quantize bit. Set by the ranking when it picks a
    // quantize action; the actual decision to quantize also needs a
    // positive error bound, checked below where upstream checks it.
    const bool quantize_requested = UnpackQuantEnabled(packed_preset);

    ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
    if (preset_id == 1) {
      preset = ctp::CompressionPreset::FAST;
    } else if (preset_id == 3) {
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

    // Core header now; grows by the quantization extension below if the
    // quantize action actually runs. Every offset into the stored blob has
    // to use THIS, not sizeof(CompressionHeader).
    size_t header_size = sizeof(CompressionHeader);
    // Worst-case compressed size: original size + 5% overhead.
    size_t worst_case_size = input_size + (input_size / 20) + 1024;

    // Convert ShmPtr to raw pointer via FullPtr
    auto input_fullptr =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    char* input_ptr = input_fullptr.ptr_;

    // GPU-native libraries (nvcomp/cusz/cuszp/ndzip) accept a device pointer
    // directly -- they stage their own H2D as needed. Everything else is a
    // plain host codec that would read a device pointer directly and crash,
    // so stage a D2H copy ourselves first. NeuroPress's own selection keeps
    // this a safety net rather than the normal route: EstCompressionStats
    // excludes CPU candidates for a device-resident buffer, so only the
    // legacy heuristic fallback (or an explicit/static compress_lib_) can
    // still land here with a device pointer + CPU library.
    std::vector<char> device_staging;
    input_ptr = ctp::CompressionFactory::StageInputIfNeeded(
        input_ptr, input_size, context.compress_lib_, device_staging);

    // Byte-shuffle preprocessing, when the ranked candidate asked for it.
    // Groups each element's Nth byte together so a codec sees runs of
    // similar magnitude -- NeuroPress's own shuffle dimension. Applied AFTER
    // any device staging above, since ByteShuffle is host code, and the
    // element size is recorded in the header so Decompress can invert it.
    std::vector<char> shuffle_staging;
    char *shuffle_device_buf = nullptr;
    ctp::ipc::AllocatorId shuffle_device_alloc;
    // Declared up here, alongside the shuffle scratch, so a single guard
    // owns both device allocations.
    ctp::ipc::AllocatorId device_output_alloc_id;
    ctp::ipc::AllocatorId quant_device_alloc;
    uint32_t applied_shuffle = 0;

    // Bytes the CODEC sees. Diverges from input_size the moment quantization
    // runs, because quantizing float32 to int8/int16 shrinks the buffer
    // before the codec ever looks at it. Upstream keeps the same distinction
    // (compress_input_size vs input_size, gpucompress_compress.cpp).
    size_t compress_input_size = input_size;
    bool applied_quant = false;
    std::vector<char> quant_staging;  // host quantize path only
    ctp::compress::preprocess::DeviceQuantizeParams quant_params;

    // Releases both on EVERY exit from this scope. The frees used to live
    // only in the success branch, so an incompressible chunk (the
    // not-beneficial path), a failed SHM allocation, or a thrown exception
    // each leaked input_size bytes of device memory plus the compressed
    // output buffer -- and incompressible chunks are exactly the workload
    // that takes that branch repeatedly, so it drained the GPU. NeuroPress
    // frees on every exit path too (gpucompress_compress.cpp and
    // the sibling checks at :486-519).
    struct DeviceScratchGuard {
      ctp::ipc::AllocatorId *ids[3];
      ~DeviceScratchGuard() {
        for (auto *id : ids) {
          if (id && !id->IsNull()) {
            CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, *id);
            *id = ctp::ipc::AllocatorId();
          }
        }
      }
    } device_scratch{{&shuffle_device_alloc, &device_output_alloc_id,
                     &quant_device_alloc}};

    // ---- Quantization, BEFORE the shuffle ----
    // Upstream's order is quantize then shuffle, and the read side inverts it
    // in reverse -- unshuffle, then dequantize. Getting the order wrong
    // round-trips to garbage, so both sides are written against that pairing.
    //
    // Gated as upstream gates it: a positive error bound, which is also the
    // condition under which its ranking stops masking quantize actions. No
    // data_type_ gate -- upstream treats the buffer as float32 unconditionally,
    // and so does EstCompressionStats whenever NeuroPress is ranking. Requiring
    // a declared float type here would let quantize candidates be ranked and
    // chosen but never applied. Non-float bytes read as float32 can produce a
    // non-finite range, which makes the quantizer decline cleanly.
    const bool want_quant = quantize_requested && context.error_bound_ > 0.0 &&
                            input_size >= sizeof(float) &&
                            (input_size % sizeof(float)) == 0;
    if (want_quant && ctp::IsDevicePointer(input_ptr)) {
      char *quant_buf = nullptr;
      quant_device_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          input_size, &quant_buf);
      size_t quant_bytes = 0;
      if (!quant_device_alloc.IsNull() &&
          ctp::compress::preprocess::QuantizeDevice(
              input_ptr, input_size / sizeof(float), context.error_bound_,
              quant_buf, &quant_bytes, &quant_params)) {
        input_ptr = quant_buf;          // still device-resident
        compress_input_size = quant_bytes;
        applied_quant = true;
        HLOG(kDebug,
             "NeuroPress quantize: {} -> {} bytes (precision={} eb={} "
             "effective_eb={} achievable={})",
             input_size, quant_bytes, quant_params.precision,
             context.error_bound_, quant_params.effective_error_bound,
             quant_params.bound_achievable);
      } else if (!quant_device_alloc.IsNull()) {
        CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, quant_device_alloc);
        quant_device_alloc = ctp::ipc::AllocatorId();
      }
      // On failure nothing is recorded and the chunk is compressed
      // losslessly -- the data is still correct, just not the action that
      // was ranked.
    } else if (want_quant) {
      // Host-resident chunk. StageInputIfNeeded may have staged a device
      // buffer down for a CPU codec, and a caller can hand us host memory
      // directly, so the quantize action has to be honored here too --
      // otherwise it is ranked and chosen but never applied, exactly the
      // mismatch the device branch above exists to avoid. The host
      // quantizer shares its arithmetic with the device kernels and was
      // shown to produce byte-identical output, so a blob written on either
      // path decodes on either path.
      auto host_q = ctp::compress::preprocess::Quantize<float>(
          reinterpret_cast<const float *>(input_ptr),
          input_size / sizeof(float), context.error_bound_);
      if (!host_q.quantized_.empty()) {
        quant_staging.assign(host_q.quantized_.begin(),
                             host_q.quantized_.end());
        input_ptr = quant_staging.data();
        compress_input_size = quant_staging.size();
        applied_quant = true;
        quant_params.effective_error_bound = host_q.effective_error_bound_;
        quant_params.scale = host_q.scale_;
        quant_params.data_min = host_q.data_min_;
        quant_params.data_max = host_q.data_max_;
        quant_params.precision = host_q.precision_;
        quant_params.bound_achievable = host_q.bound_achievable_;
        HLOG(kDebug,
             "NeuroPress quantize (host): {} -> {} bytes (precision={} eb={})",
             input_size, compress_input_size, host_q.precision_,
             context.error_bound_);
      }
    }

    if (applied_quant) {
      header_size += sizeof(QuantHeaderExtension);
    }

    if (shuffle_elem != 0) {
      if (ctp::IsDevicePointer(input_ptr)) {
        // Shuffle ON the device. Staging down to shuffle on the host would
        // also leave input_ptr pointing at host memory, which flips
        // output_on_device below and sends the compressed output through
        // host memory too -- a D2H/H2D round trip per chunk that undoes the
        // whole device-resident path. NeuroPress shuffles on-device for the
        // same reason (byte_shuffle_kernels.cu).
        shuffle_device_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
            /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
            compress_input_size, &shuffle_device_buf);
        if (!shuffle_device_alloc.IsNull() &&
            ctp::compress::preprocess::ByteShuffleDevice(
                input_ptr, shuffle_device_buf, compress_input_size, shuffle_elem)) {
          input_ptr = shuffle_device_buf;  // still on the device
          applied_shuffle = shuffle_elem;
        } else if (!shuffle_device_alloc.IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, shuffle_device_alloc);
          shuffle_device_alloc = ctp::ipc::AllocatorId();
          shuffle_device_buf = nullptr;
        }
      } else {
        shuffle_staging.resize(compress_input_size);
        if (ctp::compress::preprocess::ByteShuffle(
                reinterpret_cast<const uint8_t *>(input_ptr), compress_input_size,
                shuffle_elem,
                reinterpret_cast<uint8_t *>(shuffle_staging.data()))) {
          input_ptr = shuffle_staging.data();
          applied_shuffle = shuffle_elem;
        }
      }
      // On failure input_ptr is untouched and applied_shuffle stays 0, so
      // the header records "not shuffled" and the read side does nothing.
    }

    // input_ptr is still device-resident only when StageInputIfNeeded left
    // it alone, i.e. the codec is GPU-native. Runtime::Compress always runs
    // co-located with core_client_ (see above), so PutBlob below never
    // stages this output either -- it flows straight to
    // MemBdevTransport::WriteBlocks, which is already GPU-aware. Keeping the
    // compressed OUTPUT on-device too avoids compressor->Compress()'s own
    // internal D2H (see nvcomp.h's ToDeviceInput/out_is_device) plus this
    // function's own host memcpy into the SHM buffer below -- down to the
    // one D2H copy the bdev write does regardless.
    bool output_on_device = ctp::IsDevicePointer(input_ptr);

    std::vector<char> compressed_buffer;
    char *device_output = nullptr;
    if (output_on_device) {
      device_output_alloc_id = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          header_size + worst_case_size, &device_output);
      if (device_output_alloc_id.IsNull()) {
        output_on_device = false;  // Fall back to the host buffer below.
      }
    }
    if (!output_on_device) {
      compressed_buffer.resize(worst_case_size);
    }
    // Compressed bytes land after the header's spot so the device path can
    // fill the header in-place afterward without a second allocation.
    char *compress_dst =
        output_on_device ? (device_output + header_size)
                          : compressed_buffer.data();

    // Time ONLY the compress call. NeuroPress brackets exactly this with
    // CUDA events (gpucompress_compress.cpp) and its offline labels were
    // measured the same way, so including the output allocation and the
    // D2H/H2D staging above would feed the model a systematically inflated
    // comp_time -- biasing both the log1p target for output 0 and the
    // error_pct that gates whether training fires at all.
    size_t compressed_size = worst_case_size;
    auto compress_start = std::chrono::high_resolution_clock::now();
    // compress_input_size, not input_size: after quantization the codec sees
    // the narrowed buffer, which is the whole point of the transform.
    bool success = compressor->Compress(compress_dst, compressed_size,
                                        input_ptr, compress_input_size);

    auto compress_end = std::chrono::high_resolution_clock::now();
    double compress_time =
        std::chrono::duration<double, std::milli>(compress_end - compress_start)
            .count();
    // Device time for the codec launch alone, when CLIO_CODEC_KERNEL_TIMING
    // is on. compress_time above is host wall clock around the whole
    // Compress() call and also covers staging, allocation and the output
    // copy, so it is not comparable with another implementation's kernel
    // time; this is.
    const double compress_kernel_ms = ctp::LastCodecKernelMs();

    // Check if compression succeeded and is beneficial (include header size
    // in the total stored size)
    size_t total_stored_size = compressed_size + header_size;

    if (success && SelectionLogEnabled()) {
      LogCompressedPayload(task->blob_name_.str(), compress_dst,
                           compressed_size, output_on_device,
                           total_stored_size < input_size,
                           compress_kernel_ms, "primary");
    }

    CLIO_PATH_TRACE("WRITE  codec ran lib=%d in=%zu out=%zu kept=%d",
                    context.compress_lib_, (size_t)input_size,
                    (size_t)total_stored_size,
                    (success && total_stored_size < input_size) ? 1 : 0);

    if (success && total_stored_size < input_size) {
      // Update context with compression statistics
      context.actual_original_size_ = input_size;
      context.actual_compressed_size_ = total_stored_size;
      // Ratio is measured against the CODEC's output, not the stored total.
      // Upstream computes actual_ratio = input_size / compressed_size
      // (gpucompress_compress.cpp) with compressed_size being the
      // payload alone -- its own 64-byte header is excluded. Including
      // Clio's 24-byte header biased every training label slightly low, and
      // the bias grows as the chunk shrinks. actual_compressed_size_ below
      // still reports the true stored size, which is what telemetry wants.
      context.actual_compression_ratio_ =
          static_cast<double>(input_size) /
          static_cast<double>(compressed_size);
      // Prefer the CODEC KERNEL time over host wall clock. This is what the
      // model was trained against and what upstream ranks on: NeuroPress
      // brackets exactly its `compressor->compress(...)` launch with CUDA
      // events (gpucompress_compress.cpp) and feeds THAT into
      // compute_cost. compress_time here is wall clock around the whole
      // Compress() call, which also covers ToDeviceInput staging, a possible
      // cudaMalloc, configure_compression, the stream sync and the output
      // copy -- systematically larger, and it propagates into three
      // decisions: the exploration winner, the SGD comp-time target, and the
      // error_pct that gates whether SGD or exploration fire at all.
      // Falls back to wall clock if the event bracket did not produce a
      // reading (LastCodecKernelMs() returns -1 when it could not run).
      context.actual_compress_time_ms_ =
          (compress_kernel_ms >= 0.0) ? compress_kernel_ms : compress_time;

      // PSNR is DEFINED only when quantization ran. Upstream seeds
      // primary_actual_psnr = -1.0 ("lossless -> skip PSNR MAPE",
      // gpucompress_compress.cpp:388) and overwrites it only inside
      // `if (d_quantized && quant_result.isValid())` (:653-658).
      //
      // That variable is upstream's DIAGNOSTICS value -- it is reported as
      // di.actual_psnr (:1100) and nothing else. It is NOT what reaches the
      // SGD: the phase-1 sample gets a literal 120.0 (:697-700, :711). This
      // field therefore feeds the selection log only, and the SGD site
      // re-derives its own label -- see primary_sgd_psnr above. Reusing this
      // value for training was the defect: it withheld the PSNR gradient on
      // every lossless chunk, and because that error backpropagates through
      // the shared trunk it moved 5513 of 13576 weights per step away from
      // upstream, not just the one head.
      if (applied_quant) {
        const double psnr = AnalyticalPsnr(
            quant_params.data_max - quant_params.data_min,
            quant_params.effective_error_bound);
        if (psnr > 0.0) context.actual_psnr_db_ = psnr;
      } else {
        context.actual_psnr_db_ = -1.0;
      }

      // Record the shuffle that was ACTUALLY applied, not the one requested:
      // if ByteShuffle declined (unsupported element size, or a size that is
      // not a whole number of elements) the bytes are unshuffled and the
      // read side must not try to invert it.
      // original_size_ is the TRUE original byte count, not the quantized
      // one: the read side sizes its output buffer from it and dequantizes
      // back to exactly that many bytes. compressed_size is the codec's
      // output. The quantize bit and precision ride in the packed word, and
      // the four doubles that make the transform invertible go in the
      // extension written right after this header.
      CompressionHeader header(
          context.compress_lib_,
          PackPreset(preset_id, applied_shuffle) |
              PackQuant(applied_quant, quant_params.precision),
          input_size, compressed_size);
      QuantHeaderExtension quant_ext{};
      if (applied_quant) {
        quant_ext.error_bound = quant_params.effective_error_bound;
        quant_ext.scale = quant_params.scale;
        quant_ext.data_min = quant_params.data_min;
        quant_ext.data_max = quant_params.data_max;
      }
      ctp::ipc::ShmPtr<> compressed_shm_ptr;
      ctp::ipc::FullPtr<char> compressed_shm;  // Only used off the device path.

      if (output_on_device) {
        // Header goes in the room compress_dst was offset past above --
        // this is the only host touch for the whole compressed buffer, and
        // it's 24 bytes, not the payload.
        // Core and extension are copied SEPARATELY: header_size is 56 for a
        // quantized blob, and copying that many bytes out of the 24-byte
        // struct would read past it.
        ctp::GpuApi::Memcpy(device_output,
                            reinterpret_cast<const char *>(&header),
                            sizeof(CompressionHeader));
        if (applied_quant) {
          ctp::GpuApi::Memcpy(device_output + sizeof(CompressionHeader),
                              reinterpret_cast<const char *>(&quant_ext),
                              sizeof(quant_ext));
        }
        compressed_shm_ptr.alloc_id_ = device_output_alloc_id;
        compressed_shm_ptr.off_ = reinterpret_cast<clio::run::u64>(device_output);
      } else {
        compressed_buffer.resize(compressed_size);

        // Allocate shared memory for header + compressed data
        compressed_shm = CLIO_IPC->AllocateBuffer(total_stored_size);
        if (compressed_shm.IsNull()) {
          HLOG(kError, "Failed to allocate shared memory for compressed data");
          task->return_code_ = 4;  // Memory allocation failed
          CLIO_CO_RETURN;
        }
        std::memcpy(compressed_shm.ptr_, &header, sizeof(CompressionHeader));
        if (applied_quant) {
          std::memcpy(compressed_shm.ptr_ + sizeof(CompressionHeader),
                      &quant_ext, sizeof(quant_ext));
        }
        std::memcpy(compressed_shm.ptr_ + header_size, compressed_buffer.data(),
                    compressed_size);
        compressed_shm_ptr = compressed_shm.shm_.template Cast<void>();
      }

      // Tell the runtime these bytes are no longer the caller's bytes, so it
      // can mark the blob authoritatively (issue #818). This is the ONLY place
      // that knows it for certain -- compress_lib_ is set on the not-beneficial
      // path below too, where the stored bytes are raw.
      context.transform_flags_ |= clio::cte::core::kBlobTransformed |
                                  clio::cte::core::kBlobTransformCompressed;

      int stored_put_rc = 0;
      if (task->no_store_) {
        // Hand the image to the caller instead of storing it, so exploration
        // can compare candidates and write the winner exactly once. See
        // CompressTask::no_store_ for why storing here first is the thing
        // being avoided.
        //
        // Ownership moves with the pointer. On the device path the buffer is
        // one of the three allocations device_scratch frees on every exit, so
        // release its claim by nulling the id it watches -- otherwise the
        // guard would free the bytes the caller is about to store. On the host
        // path simply skip the FreeBuffer below.
        task->stored_data_ = compressed_shm_ptr;
        task->stored_size_ = total_stored_size;
        task->stored_owned_ = true;
        if (output_on_device) {
          task->stored_gpu_alloc_ = device_output_alloc_id;
          device_output_alloc_id = ctp::ipc::AllocatorId();
        } else {
          task->stored_gpu_alloc_ = ctp::ipc::AllocatorId();
        }
        task->context_ = context;
        task->return_code_ = 0;
      } else {
        // Call PutBlob with header + compressed data
        auto put_task = core_client_->AsyncPutBlob(
            task->tag_id_, task->blob_name_.str(), task->offset_,
            total_stored_size, compressed_shm_ptr, task->score_, context,
            task->flags_, clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put_task);
        stored_put_rc = put_task->return_code_;

        // Device allocations belong to device_scratch above, which releases
        // them on every exit; only the host SHM buffer is freed here.
        if (!output_on_device) {
          CLIO_IPC->FreeBuffer(compressed_shm);
        }
      }

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
      // The no-store branch already set its own return code; put_task only
      // exists on the storing branch.
      if (!task->no_store_) task->return_code_ = stored_put_rc;
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

      if (task->no_store_) {
        // Same deferral as the compressed path, and for the same reason: an
        // exploration winner must not be written over an already-stored blob.
        // The bytes here are the CALLER'S original buffer, so hand it back
        // unowned -- freeing it would destroy memory this task never
        // allocated. Without this branch the raw image was stored anyway and
        // a later winner left the full uncompressed footprint behind, which
        // is exactly the case the regression test caught.
        task->stored_data_ = task->blob_data_;
        task->stored_size_ = task->size_;
        task->stored_gpu_alloc_ = ctp::ipc::AllocatorId();
        task->stored_owned_ = false;
        task->context_ = context;
        task->return_code_ = 0;
        CLIO_CO_RETURN;
      }

      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
          task->blob_data_, task->score_, context, task->flags_,
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put_task);

      task->context_ = put_task->context_;
      task->return_code_ = put_task->return_code_;

      // Record what the codec REALLY achieved, even though we discarded its
      // output. A codec that ran and produced a 1.02x ratio in 8 ms is a
      // perfectly good training sample -- and it is the sample the model
      // most needs, since high-entropy chunks are where it mispredicts.
      // Leaving the Context defaults in place (ratio 1.0, time 0.0) taught
      // it that the chosen algorithm achieves exactly 1.0x in 0 ms, and
      // because cost() maps a 0 ratio to 1e30 the MAPE gate then fired on
      // essentially every such chunk. NeuroPress always has real numbers
      // here: it computes actual_ratio = input_size / compressed_size
      // unconditionally (gpucompress_compress.cpp) and has no
      // not-beneficial branch at all.
      //
      // A genuine codec FAILURE is different -- there is no measurement, so
      // leave the ratio at 0 and let the learning gate skip the chunk.
      if (success) {
        task->context_.actual_original_size_ = input_size;
        task->context_.actual_compressed_size_ = total_stored_size;
        task->context_.actual_compression_ratio_ =
            static_cast<double>(input_size) /
            static_cast<double>(compressed_size);
        task->context_.actual_compress_time_ms_ = compress_time;
      } else {
        task->context_.actual_compression_ratio_ = 0.0;
        task->context_.actual_compress_time_ms_ = 0.0;
      }
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

    // Extract task parameters (same as GetBlobTask). task->size_ is the
    // caller-declared LOGICAL (original, uncompressed) size -- compression
    // is only ever kept when it makes the stored blob strictly smaller (see
    // Compress()'s "total_stored_size < input_size" check), so it is a safe
    // UPPER bound, but it is not the actual number of bytes physically
    // stored. Using it directly as compressed_size below over-reads past the
    // true compressed stream into the destination buffer's uninitialized
    // tail, which corrupts the decompressor's input; LZ4 in particular then
    // returns a negative status that silently underflows the unsigned
    // output_size and gets reported as success with a garbage size.
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
    // See the matching guard in Compress(): without a resolved core pool,
    // the unconditional core_client_->AsyncGetBlob() below null-derefs.
    if (!core_client_) {
      HLOG(kError,
           "Decompress: no core pool available (compose next_pool_id_ unset "
           "and caller passed no explicit core_pool_id) -- cannot fetch the "
           "blob to decompress");
      task->return_code_ = 3;  // No core pool available
      CLIO_CO_RETURN;
    }

    // Ask core directly (bypassing this class's own GetBlobSize override,
    // which deliberately reports the LOGICAL size) for the blob's actual
    // PHYSICAL size -- the true number of bytes to read and hand to the
    // decompressor. Falls back to the logical-size upper bound if the query
    // fails, so a size-lookup problem degrades to the old (possibly
    // over-reading) behavior rather than aborting the read outright.
    if (core_client_) {
      auto size_task = core_client_->AsyncGetBlobSize(
          task->tag_id_, task->blob_name_.str(), clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(size_task);
      if (size_task->return_code_ == 0 && size_task->size_ > 0) {
        expected_size = size_task->size_;
      }
    }

    // Allocate temporary buffer to receive compressed data from GetBlob,
    // sized to the blob's actual physical size (see above).
    auto temp_buffer = CLIO_IPC->AllocateBuffer(expected_size);
    if (temp_buffer.IsNull()) {
      task->return_code_ = 2;  // Memory allocation failed
      CLIO_CO_RETURN;
    }
    ctp::ipc::ShmPtr<> temp_buffer_ptr = temp_buffer.shm_.template Cast<void>();

    // Call GetBlob to retrieve the (potentially compressed) data
    auto get_task = core_client_->AsyncGetBlob(
        task->tag_id_, task->blob_name_.str(), task->offset_, expected_size,
        task->flags_, temp_buffer_ptr, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(get_task);

    if (get_task->return_code_ != 0) {
      CLIO_IPC->FreeBuffer(temp_buffer);
      task->return_code_ = 10 + get_task->return_code_;  // GetBlob failed
      CLIO_CO_RETURN;
    }

    // Is this blob compressed? Ask the CORE, do not guess from the bytes.
    //
    // Clio stores incompressible data raw with no header (Compress()'s "not
    // beneficial" branch), so this cannot be decided by testing the first four
    // bytes against the magic -- a user payload that happens to begin with it
    // would be parsed as a header. GetBlob reports the blob's authoritative
    // transform state through the context (issue #818), the same signal
    // GetBlobSize and the interposer already use. The magic is then an
    // integrity check on a blob that is supposed to carry a header, which is
    // what it is upstream.
    const bool blob_is_compressed =
        (get_task->context_.transform_flags_ &
         clio::cte::core::kBlobTransformCompressed) != 0;

    CLIO_PATH_TRACE("READ   compressor Decompress blob='%s' physical=%llu "
                    "logical=%llu stored_compressed=%d -> %s",
                    task->blob_name_.str().c_str(),
                    (unsigned long long)expected_size,
                    (unsigned long long)task->size_,
                    blob_is_compressed ? 1 : 0,
                    blob_is_compressed ? "inverting codec" : "passthrough");

    auto* header = reinterpret_cast<CompressionHeader*>(temp_buffer.ptr_);
    size_t header_size = sizeof(CompressionHeader);

    if (blob_is_compressed && expected_size >= header_size &&
        !header->IsValid()) {
      // Marked compressed but the header does not check out: the blob is
      // damaged, or was written by a build whose format this one does not
      // understand (IsValid() also gates on the format version). Failing is
      // the only safe answer -- upstream returns INVALID_HEADER here.
      HLOG(kError,
           "Decompress: blob '{}' is marked compressed but its header is not "
           "valid (magic/version mismatch) -- refusing to guess",
           task->blob_name_.str());
      CLIO_IPC->FreeBuffer(temp_buffer);
      task->return_code_ = 6;  // Invalid/unreadable header
      CLIO_CO_RETURN;
    }

    if (blob_is_compressed) {
      // Data is compressed - decompress it
      int compress_lib = static_cast<int>(header->compress_lib_);
      // High bits carry the byte-shuffle element size (see PackPreset). A
      // blob written before shuffling existed has zeros there, so it decodes
      // as "not shuffled" and this stays backward compatible.
      const uint32_t packed_preset = header->compress_preset_;
      int compress_preset = static_cast<int>(UnpackPreset(packed_preset));
      const uint32_t stored_shuffle = UnpackShuffle(packed_preset);
      clio::run::u64 original_size = header->original_size_;

      // A quantized blob carries a 32-byte extension after the core header,
      // holding the four doubles that make the transform invertible. Read it
      // before computing the payload offset -- everything downstream depends
      // on header_size being right.
      const bool stored_quant = UnpackQuantEnabled(packed_preset);
      ctp::compress::preprocess::DeviceQuantizeParams stored_quant_params;
      if (stored_quant) {
        header_size += sizeof(QuantHeaderExtension);
        if (expected_size < header_size) {
          HLOG(kError,
               "Decompress: blob '{}' claims quantization but is too small "
               "to hold the header extension",
               task->blob_name_.str());
          CLIO_IPC->FreeBuffer(temp_buffer);
          task->return_code_ = 6;
          CLIO_CO_RETURN;
        }
        QuantHeaderExtension ext{};
        std::memcpy(&ext, temp_buffer.ptr_ + sizeof(CompressionHeader),
                    sizeof(ext));
        stored_quant_params.effective_error_bound = ext.error_bound;
        stored_quant_params.scale = ext.scale;
        stored_quant_params.data_min = ext.data_min;
        stored_quant_params.data_max = ext.data_max;
        stored_quant_params.precision = UnpackQuantPrecision(packed_preset);
      }

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

      // Get compressed data (after header). Prefer the length the WRITER
      // recorded in the header over deriving it from expected_size: the
      // latter came from an AsyncGetBlobSize RPC that silently falls back to
      // the caller's LOGICAL (uncompressed) size, which is only an upper
      // bound -- feeding that to the decompressor over-reads past the real
      // stream into the buffer's uninitialized tail. NeuroPress reads its
      // own header.compressed_size the same way and bounds-checks it
      // (gpucompress_compress.cpp); PayloadSize() applies the same
      // two checks and returns 0 if the field is absent or implausible, in
      // which case we keep the old derivation.
      char* compressed_data = temp_buffer.ptr_ + header_size;
      const size_t recorded_size = header->PayloadSize(expected_size);
      size_t compressed_size =
          (recorded_size > 0) ? recorded_size : (expected_size - header_size);

      // Decompress to output buffer
      auto output_fullptr =
          CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());

      // A quantized blob decompresses to the NARROWED buffer, not to the
      // original bytes: the codec's output is int8/int16/int32 values that
      // only become float32 again after dequantization. So the codec writes
      // into a scratch of the quantized size, the unshuffle inverts on that
      // scratch, and dequantize is what finally fills the caller's buffer.
      // Upstream is shaped the same way (gpucompress_compress.cpp:
      // decompress, then unshuffle, then dequantize, each into its own
      // buffer).
      const size_t quant_elems =
          stored_quant ? (original_size / sizeof(float)) : 0;
      const size_t quant_bytes =
          stored_quant
              ? quant_elems * ctp::compress::preprocess::PrecisionToBytes(
                                  stored_quant_params.precision)
              : 0;

      char *codec_dst = output_fullptr.ptr_;
      size_t decompressed_size = original_size;
      ctp::ipc::AllocatorId quant_scratch_alloc;
      char *quant_scratch = nullptr;
      if (stored_quant) {
        quant_scratch_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
            /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
            quant_bytes, &quant_scratch);
        if (quant_scratch_alloc.IsNull()) {
          HLOG(kError,
               "Decompress: could not allocate the dequantization scratch "
               "({} bytes) for blob '{}'",
               quant_bytes, task->blob_name_.str());
          CLIO_IPC->FreeBuffer(temp_buffer);
          task->return_code_ = 2;
          CLIO_CO_RETURN;
        }
        codec_dst = quant_scratch;
        decompressed_size = quant_bytes;
      }

      bool success =
          decompressor->Decompress(codec_dst, decompressed_size,
                                   compressed_data, compressed_size);

      // Proof-of-execution trace, off unless
      // CLIO_NEUROPRESS_DECOMPRESS_TRACE is set. A caller only ever sees a
      // return code, and a path that served the bytes from somewhere else
      // returns 0 just as happily -- which is precisely what the HDF5 VOL
      // does when a dropped cache tag sends the read to the uncompressed
      // copy. This reports from the codec call itself.
      {
        static const bool trace = [] {
          const char *e = std::getenv("CLIO_NEUROPRESS_DECOMPRESS_TRACE");
          return e && *e;
        }();
        if (trace) {
          static std::atomic<long> calls{0};
          std::fprintf(stderr,
                       "[clio-decompress] call #%ld blob=%s lib=%s "
                       "compressed=%zu -> %zu ok=%d kernel_ms=%.6f\n",
                       calls.fetch_add(1) + 1, task->blob_name_.str().c_str(),
                       library_name.c_str(),
                       static_cast<size_t>(compressed_size),
                       static_cast<size_t>(decompressed_size),
                       success ? 1 : 0, ctp::LastCodecKernelMs());
          std::fflush(stderr);
        }
      }

      CLIO_IPC->FreeBuffer(temp_buffer);

      // Invert the byte-shuffle the write side applied. Must happen before
      // the caller ever sees the buffer -- a shuffled image is not the
      // user's data, it just happens to be the same length.
      if (success && stored_shuffle != 0) {
        if (ctp::IsDevicePointer(codec_dst)) {
          // Unshuffle ON the device, into a scratch device buffer and back.
          // Staging down to the host would work but costs a D2H/H2D round
          // trip per chunk for a GPU consumer decompressing into its own
          // memory -- exactly the case this path exists to serve.
          char *scratch = nullptr;
          ctp::ipc::AllocatorId scratch_alloc =
              CLIO_IPC->AllocateAndRegisterGpuBackend(
                  /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
                  decompressed_size, &scratch);
          if (scratch_alloc.IsNull() ||
              !ctp::compress::preprocess::ByteUnshuffleDevice(
                  codec_dst, scratch, decompressed_size,
                  stored_shuffle)) {
            HLOG(kError,
                 "Decompress: device byte-unshuffle failed (elem={} size={}) "
                 "-- the returned buffer would be shuffled garbage, failing "
                 "instead",
                 stored_shuffle, decompressed_size);
            success = false;
          } else {
            ctp::GpuApi::Memcpy(codec_dst, scratch,
                                decompressed_size);
          }
          if (!scratch_alloc.IsNull()) {
            CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, scratch_alloc);
          }
        } else {
          std::vector<char> unshuffled(decompressed_size);
          if (ctp::compress::preprocess::ByteUnshuffle(
                  reinterpret_cast<const uint8_t *>(codec_dst),
                  decompressed_size, stored_shuffle,
                  reinterpret_cast<uint8_t *>(unshuffled.data()))) {
            std::memcpy(codec_dst, unshuffled.data(),
                        decompressed_size);
          } else {
            HLOG(kError,
                 "Decompress: byte-unshuffle failed (elem={} size={}) -- the "
                 "returned buffer would be shuffled garbage, failing instead",
                 stored_shuffle, decompressed_size);
            success = false;
          }
        }
      }

      // Dequantize LAST, inverting the write side's quantize-then-shuffle
      // (upstream inverts in the same order, gpucompress_compress.cpp:
      // 1253-1296). This is what finally produces the caller's float32
      // bytes, so the reported output size becomes the original size again.
      if (success && stored_quant) {
        if (!ctp::compress::preprocess::DequantizeDevice(
                codec_dst, quant_elems, stored_quant_params,
                output_fullptr.ptr_)) {
          HLOG(kError,
               "Decompress: dequantization failed for blob '{}' (elems={} "
               "precision={}) -- the returned buffer would be quantized "
               "integers, failing instead",
               task->blob_name_.str(), quant_elems,
               stored_quant_params.precision);
          success = false;
        } else {
          decompressed_size = original_size;
        }
      }
      if (!quant_scratch_alloc.IsNull()) {
        CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, quant_scratch_alloc);
      }

      // Stop the clock AFTER the unshuffle, not before it. Upstream's
      // CUDA-event bracket (H5VLgpucompress.cu:3302-3311) wraps
      // gpucompress_decompress_gpu(), which performs the byte-unshuffle and
      // dequantize plus their scratch allocations inside the timed region.
      // Ending it at the codec call made Clio's decomp-head target
      // systematically low for every shuffled blob -- and shuffled actions
      // are the ones the model picks most often.
      auto decompress_end = std::chrono::high_resolution_clock::now();
      double decompress_time = std::chrono::duration<double, std::milli>(
                                   decompress_end - decompress_start)
                                   .count();

      if (success) {
        task->output_size_ = decompressed_size;
        task->decompress_time_ms_ = decompress_time;

        // Show the RECONSTRUCTED VALUES, not just that a call returned 0.
        // Same env gate as the codec-level trace above; this one fires after
        // unshuffle and dequantize, so it reports the buffer the caller is
        // actually about to receive. Sizes alone cannot distinguish a real
        // reconstruction from a zero-filled buffer of the right length, and a
        // zeroed buffer is what a silently-failed codec leaves behind.
        //
        // Scans the WHOLE buffer, not a sample. The fields this path carries
        // are sparse -- a Gray-Scott V field is a few percent non-zero, packed
        // into the slabs that hold the reacting region -- so both a head sample
        // and an interior window read legitimate zeros and cry "blank" over a
        // perfect reconstruction. A full count is the only sample that cannot
        // lie, and per-chunk totals can be summed and checked against the
        // source, which is what makes this proof rather than reassurance.
        // One D2H copy per chunk, on a path that is already opt-in.
        {
          static const bool vtrace = [] {
            const char *e = std::getenv("CLIO_NEUROPRESS_DECOMPRESS_TRACE");
            return e && *e;
          }();
          if (vtrace && decompressed_size >= sizeof(float)) {
            const size_t nfloat = decompressed_size / sizeof(float);
            std::vector<float> all(nfloat);
            // output_fullptr may be device memory (the VOL stages GPU-side),
            // so this cannot be a plain dereference.
            ctp::DeviceAwareMemcpy(reinterpret_cast<char *>(all.data()),
                                   output_fullptr.ptr_,
                                   nfloat * sizeof(float));
            size_t nz = 0;
            float lo = all[0], hi = all[0];
            double sum = 0.0;
            for (float f : all) {
              if (f != 0.0f) ++nz;
              lo = std::min(lo, f);
              hi = std::max(hi, f);
              sum += f;
            }
            std::fprintf(stderr,
                         "[clio-decompress]   -> produced %zu bytes = %zu "
                         "floats; nonzero=%zu min=%.6f max=%.6f mean=%.6g %s\n",
                         static_cast<size_t>(decompressed_size), nfloat, nz,
                         lo, hi, sum / (double)nfloat,
                         nz == 0 ? "(all-zero chunk)" : "");
            std::fflush(stderr);
          }
        }

        // Deferred decomp-head learning: this is the ONLY point a real
        // decompression time exists. Join it back to the features the
        // original Compress predicted from and train that one head.
        LearnDecompTime(task->blob_name_.str(), decompress_time);

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
      // Device-aware, not std::memcpy: the caller's destination is a CUDA-IPC
      // device buffer whenever a GPU consumer reads into its own memory, and
      // a host memcpy into it segfaults. The compressed branch above already
      // routes through the device-aware helpers; this one did not, so every
      // raw-stored blob read into device memory crashed. Same class of bug as
      // the earlier host-preprocessing-on-a-device-pointer faults.
      ctp::DeviceAwareMemcpy(output_fullptr.ptr_, temp_buffer.ptr_,
                             expected_size);
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
  // compress_preset_ is PACKED (preset | shuffle_elem << 8) whenever the
  // selection came from DynamicSchedule. Unpack before comparing, or a
  // shuffled FAST candidate (1 | 4<<8 = 1025) silently falls through to
  // BALANCED -- and, worse, the packed value used to be copied verbatim
  // into the header while no shuffle was ever applied, so Runtime::
  // Decompress would invert a shuffle that never happened.
  const uint32_t requested_preset =
      UnpackPreset(static_cast<uint32_t>(ctx.compress_preset_));
  const uint32_t requested_shuffle =
      UnpackShuffle(static_cast<uint32_t>(ctx.compress_preset_));
  const bool requested_quant =
      UnpackQuantEnabled(static_cast<uint32_t>(ctx.compress_preset_));
  ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
  if (requested_preset == 1) {
    preset = ctp::CompressionPreset::FAST;
  } else if (requested_preset == 3) {
    preset = ctp::CompressionPreset::BEST;
  }
  auto compressor = ctp::CompressionFactory::GetPreset(library_name, preset);
  if (!compressor) {
    return false;
  }
  auto t0 = std::chrono::high_resolution_clock::now();

  // Honor the shuffle the selector asked for. This path is host-side (src is
  // a host SHM buffer), so the host routines are the right ones -- but the
  // blob it writes shares one magic and one header layout with the blobs
  // Runtime::Compress writes, and either read path may consume either blob.
  // Applying the shuffle here is what makes the two writers agree.
  const char *compress_src = src;
  size_t compress_size = size;

  // Quantize FIRST, then shuffle -- the same order Runtime::Compress and
  // upstream use (gpucompress_compress.cpp). This path is host-side
  // (src is a host SHM buffer), so the host quantizer applies; it shares its
  // arithmetic with the device kernels, so a blob written here decodes
  // identically to one written by the device path. Without this the quantize
  // bit was silently dropped: the data stayed lossless and the header said
  // so, self-consistent but not the action that was selected.
  std::vector<char> quant_staging;
  bool applied_quant = false;
  ctp::compress::preprocess::QuantizationResult quant_result;
  if (requested_quant && ctx.error_bound_ > 0.0 && size >= sizeof(float) &&
      (size % sizeof(float)) == 0) {
    quant_result = ctp::compress::preprocess::Quantize<float>(
        reinterpret_cast<const float *>(src), size / sizeof(float),
        ctx.error_bound_);
    if (!quant_result.quantized_.empty()) {
      quant_staging.assign(quant_result.quantized_.begin(),
                           quant_result.quantized_.end());
      compress_src = quant_staging.data();
      compress_size = quant_staging.size();
      applied_quant = true;
    }
  }

  std::vector<char> shuffle_staging;
  uint32_t applied_shuffle = 0;
  if (requested_shuffle != 0) {
    shuffle_staging.resize(compress_size);
    if (ctp::compress::preprocess::ByteShuffle(
            reinterpret_cast<const uint8_t *>(compress_src), compress_size,
            requested_shuffle,
            reinterpret_cast<uint8_t *>(shuffle_staging.data()))) {
      compress_src = shuffle_staging.data();
      applied_shuffle = requested_shuffle;
    }
    // On failure compress_src is untouched and applied_shuffle stays 0, so
    // the header records "not shuffled" and the read side does nothing.
  }

  std::vector<char> compressed(size + (size / 20) + 1024);
  size_t compressed_size = compressed.size();
  if (!compressor->Compress(compressed.data(), compressed_size,
                            const_cast<char *>(compress_src), compress_size)) {
    return false;
  }
  size_t header_size = sizeof(CompressionHeader) +
                       (applied_quant ? sizeof(QuantHeaderExtension) : 0);
  size_t total = compressed_size + header_size;
  if (total >= size) {
    return false;  // not beneficial — caller stores raw
  }
  auto shm = CLIO_IPC->AllocateBuffer(total);
  if (shm.IsNull()) {
    return false;
  }
  // Record the shuffle that was ACTUALLY applied, not the one requested --
  // a declined shuffle (wrong size multiple, allocation failure) must read
  // back as "not shuffled" or the read side inverts a transform that never
  // ran. Same rule Runtime::Compress follows.
  CompressionHeader header(
      ctx.compress_lib_,
      PackPreset(requested_preset, applied_shuffle) |
          PackQuant(applied_quant, quant_result.precision_),
      size, compressed_size);
  QuantHeaderExtension quant_ext{};
  if (applied_quant) {
    quant_ext.error_bound = quant_result.effective_error_bound_;
    quant_ext.scale = quant_result.scale_;
    quant_ext.data_min = quant_result.data_min_;
    quant_ext.data_max = quant_result.data_max_;
  }
  // Core and extension copied separately -- header_size is 56 for a
  // quantized blob and the struct is only 24.
  std::memcpy(shm.ptr_, &header, sizeof(CompressionHeader));
  if (applied_quant) {
    std::memcpy(shm.ptr_ + sizeof(CompressionHeader), &quant_ext,
                sizeof(quant_ext));
  }
  std::memcpy(shm.ptr_ + header_size, compressed.data(), compressed_size);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - t0)
                  .count();
  ctx.actual_original_size_ = size;
  ctx.actual_compressed_size_ = total;
  ctx.actual_compression_ratio_ =
      static_cast<double>(size) / static_cast<double>(total);
  ctx.actual_compress_time_ms_ = ms;
  // PSNR is defined only when quantization ran; -1 otherwise, so the SGD
  // withholds the head's gradient (see Runtime::Compress).
  ctx.actual_psnr_db_ =
      applied_quant ? AnalyticalPsnr(quant_result.data_max_ -
                                         quant_result.data_min_,
                                     quant_result.effective_error_bound_)
                    : -1.0;
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
  // NOT a sniff: the only caller (Runtime::GetBlob) has already confirmed
  // kBlobTransformCompressed on the blob before getting here, so this is an
  // integrity check on something that is supposed to carry a header --
  // exactly what upstream's header.isValid() is
  // (gpucompress_compress.cpp). A failure means damage or an unknown
  // format version, and refusing is the only safe answer.
  const auto *header = reinterpret_cast<const CompressionHeader *>(stored);
  if (!header->IsValid() || header->original_size_ > dst_cap) {
    return 6;
  }
  std::string library_name = ctp::CompressionFactory::NameForWireId(
      static_cast<int>(header->compress_lib_));

  // Proof-of-execution trace, off unless CLIO_NEUROPRESS_DECOMPRESS_TRACE is
  // set. A caller can only see a return code, which a path that quietly
  // served the data from somewhere else would also produce -- and in the
  // HDF5 VOL flow that is exactly what happens, because a dropped cache tag
  // sends reads to the uncompressed copy without decompressing anything.
  // This says, from inside the codec path, that it really ran.
  {
    static const bool trace = [] {
      const char *e = std::getenv("CLIO_NEUROPRESS_DECOMPRESS_TRACE");
      return e && *e;
    }();
    if (trace) {
      static std::atomic<long> calls{0};
      std::fprintf(stderr,
                   "[clio-decompress] call #%ld lib=%s(%d) preset=%u "
                   "stored=%llu -> original=%llu\n",
                   calls.fetch_add(1) + 1, library_name.c_str(),
                   static_cast<int>(header->compress_lib_),
                   static_cast<unsigned>(header->compress_preset_),
                   static_cast<unsigned long long>(stored_size),
                   static_cast<unsigned long long>(header->original_size_));
      std::fflush(stderr);
    }
  }
  // Same packed layout Runtime::Compress writes -- unpack both halves. This
  // path used to compare the raw field, so a shuffled blob matched neither
  // 1 nor 3 and silently used BALANCED, and the shuffle was never inverted:
  // the caller got byte-planes back and no error.
  const uint32_t stored_preset = UnpackPreset(header->compress_preset_);
  const uint32_t stored_shuffle = UnpackShuffle(header->compress_preset_);
  const bool stored_quant = UnpackQuantEnabled(header->compress_preset_);
  const size_t hdr_bytes =
      sizeof(CompressionHeader) +
      (stored_quant ? sizeof(QuantHeaderExtension) : 0);
  if (stored_size < hdr_bytes) {
    return 6;
  }
  QuantHeaderExtension stored_ext{};
  if (stored_quant) {
    std::memcpy(&stored_ext, stored + sizeof(CompressionHeader),
                sizeof(stored_ext));
  }
  ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
  if (stored_preset == 1) {
    preset = ctp::CompressionPreset::FAST;
  } else if (stored_preset == 3) {
    preset = ctp::CompressionPreset::BEST;
  }
  auto decompressor = ctp::CompressionFactory::GetPreset(library_name, preset);
  if (!decompressor) {
    return 3;
  }
  // Writer-recorded payload length when present, else the old derivation.
  // This path already aborts on a failed size query rather than guessing, so
  // it was never exposed to the over-read Runtime::Decompress could hit --
  // but reading the same field here keeps the two readers agreeing on what a
  // blob means, which is the property that broke when only one of them
  // understood the shuffle bits.
  const size_t recorded_size = header->PayloadSize(stored_size);
  const size_t payload_size =
      (recorded_size > 0) ? recorded_size : (stored_size - hdr_bytes);

  // A quantized blob decompresses to the NARROWED integers, so the codec
  // writes into a staging buffer and dequantization is what finally fills
  // the caller's dst. Same shape as Runtime::Decompress and as upstream
  // (gpucompress_compress.cpp).
  const size_t quant_elems =
      stored_quant ? (header->original_size_ / sizeof(float)) : 0;
  const size_t quant_bytes =
      stored_quant ? quant_elems * ctp::compress::preprocess::PrecisionToBytes(
                                       UnpackQuantPrecision(
                                           header->compress_preset_))
                   : 0;
  std::vector<char> quant_staging;
  char *codec_dst = dst;
  size_t decompressed = header->original_size_;
  if (stored_quant) {
    quant_staging.resize(quant_bytes);
    codec_dst = quant_staging.data();
    decompressed = quant_bytes;
  }

  if (!decompressor->Decompress(codec_dst, decompressed,
                                const_cast<char *>(stored) + hdr_bytes,
                                payload_size)) {
    return 5;
  }

  // Invert the byte-shuffle before the caller sees the buffer. dst is a host
  // buffer on this path (the interposer stages through SHM), so the host
  // routines apply. Failing is the only safe outcome: returning shuffled
  // bytes is silent corruption, since they are the right LENGTH.
  if (stored_shuffle != 0) {
    std::vector<char> unshuffled(decompressed);
    if (!ctp::compress::preprocess::ByteUnshuffle(
            reinterpret_cast<const uint8_t *>(codec_dst), decompressed,
            stored_shuffle, reinterpret_cast<uint8_t *>(unshuffled.data()))) {
      HLOG(kError,
           "DecompressStored: byte-unshuffle failed (elem={} size={}) -- the "
           "returned buffer would be shuffled garbage, failing instead",
           stored_shuffle, decompressed);
      return 5;
    }
    std::memcpy(codec_dst, unshuffled.data(), decompressed);
  }

  // Dequantize LAST, inverting quantize-then-shuffle.
  if (stored_quant) {
    ctp::compress::preprocess::QuantizationResult qr;
    qr.quantized_.assign(quant_staging.begin(), quant_staging.end());
    qr.precision_ = UnpackQuantPrecision(header->compress_preset_);
    qr.scale_ = stored_ext.scale;
    qr.data_min_ = stored_ext.data_min;
    qr.data_max_ = stored_ext.data_max;
    qr.effective_error_bound_ = stored_ext.error_bound;
    qr.num_elements_ = quant_elems;
    std::vector<float> restored =
        ctp::compress::preprocess::Dequantize<float>(qr);
    if (restored.size() != quant_elems) {
      HLOG(kError,
           "DecompressStored: dequantization failed (elems={} precision={}) "
           "-- the returned buffer would be quantized integers, failing "
           "instead",
           quant_elems, qr.precision_);
      return 5;
    }
    std::memcpy(dst, restored.data(), restored.size() * sizeof(float));
    decompressed = header->original_size_;
  }

  *out_size = decompressed;
  return 0;
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::cte::core::Context &ctx = task->context_;
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
      auto put = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), 0, stored_size,
          stored.shm_.template Cast<void>(), task->score_, ctx, task->flags_,
          clio::run::PoolQuery::Local());
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
  {
    const bool compressed = (task->context_.transform_flags_ &
                             clio::cte::core::kBlobTransformCompressed) != 0;
    CLIO_PATH_TRACE("READ   compressor GetBlob blob='%s' rc=%d replica=%d "
                    "stored_compressed=%d -> %s",
                    task->blob_name_.str().c_str(), (int)task->GetReturnCode(),
                    (int)task->context_.replica_, compressed ? 1 : 0,
                    compressed ? "decompressing" : "passthrough");
  }
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
    CLIO_PATH_TRACE("READ   codec inverted stored=%llu -> original=%llu rc=%d",
                    (unsigned long long)stored_size,
                    (unsigned long long)out_size, rc);
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
