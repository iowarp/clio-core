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
#include <clio_cte/compressor/neuropress_path_trace.h>

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

/** Byte-shuffle element size packed into the free high bits of compress_preset_. */
constexpr uint32_t kPresetMask = 0xFFu;
constexpr uint32_t kShuffleShift = 8;
constexpr uint32_t kShuffleMask = 0xFFu;

/** Format version, in bits 16-23 of the same word. */
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

/** Analytical PSNR for linear quantization, verbatim from upstream. */
/** Say which reason the quantizer declined a chunk for. */
/** True when CLIO_NEUROPRESS_REQUIRE_DEVICE demands that every chunk reach the compressor in ... */
/** One place to say why a host-resident NeuroPress transform is refused. */
inline void RefuseHostPreprocess(const char *what, size_t bytes) {
  HLOG(kError,
       "NeuroPress {} is CUDA-only and this {}-byte buffer is HOST-resident, "
       "so the transform cannot be applied. The CPU implementations were "
       "removed deliberately. Hand the compressor device memory (an in-situ "
       "adapter, or the LAMMPS driver's --order device); "
       "CLIO_NEUROPRESS_REQUIRE_DEVICE=0 disables NeuroPress preprocessing "
       "for a caller that genuinely wants plain host compression.",
       what, bytes);
}

/** True when CLIO_NEUROPRESS_STAGE_H2D asks a HOST-resident chunk to be copied up to the devi... */
inline bool NeuroPressStageH2D() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_NEUROPRESS_STAGE_H2D");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return on;
}

/** Hand Compress the copy DynamicSchedule already staged, instead of the host
 *  buffer it came from. DEFAULT ON; 0 restores the second staging. */
inline bool NeuroPressReuseStagedH2D() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_NEUROPRESS_REUSE_STAGED_H2D");
    return e == nullptr || (*e != '\0' && *e != '0');
  }();
  return on;
}

inline bool NeuroPressRequireDevice() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_NEUROPRESS_REQUIRE_DEVICE");
    // DEFAULT ON.
    return !(e != nullptr && (*e == '0' || *e == '\0'));
  }();
  return on;
}

inline void ReportQuantizeRefusal(
    ctp::compress::preprocess::QuantizeRefusal reason, double requested,
    size_t chunk_bytes) {
  const char *why =
      ctp::compress::preprocess::QuantizeRefusalName(reason);
  CLIO_PATH_TRACE("WRITE  QuantizeDevice FAILED %llu bytes eb=%g reason=%s",
                  (unsigned long long)chunk_bytes, requested, why);
  HLOG(kDebug,
       "NeuroPress quantize: failed on a {}-byte chunk at eb={} ({}); storing "
       "it losslessly",
       chunk_bytes, requested, why);
  // One bit per reason, so a second reason still gets its own line.
  static std::atomic<uint32_t> seen{0};
  const uint32_t bit = 1u << static_cast<int>(reason);
  if (seen.fetch_or(bit) & bit) return;
  HLOG(kWarning,
       "NeuroPress quantize: quantization at eb={} FAILED ({}), so that chunk "
       "is stored LOSSLESSLY. This is not a property of the data. Further "
       "failures for this reason are logged at debug level only.",
       requested, why);
}

/** Element width the quantizer reads for Context::data_type_; 0 = none. */
inline size_t QuantElemBytes(int data_type) {
  return data_type == 1 ? 4 : data_type == 2 ? 8 : 0;
}

/** A quantize request on data that is not float32/float64 is not applied:
 *  its bytes have no values to bound. */
inline void WarnQuantizeUntyped(int data_type) {
  static std::atomic<bool> warned{false};
  if (warned.exchange(true)) return;
  HLOG(kWarning,
       "NeuroPress quantize: requested for data_type_={}, which is neither "
       "float32 (1) nor float64 (2); such chunks are stored losslessly",
       data_type);
}

inline double AnalyticalPsnr(double data_range, double error_bound) {
  if (data_range <= 0.0 || error_bound <= 0.0) return -1.0;
  const double mse_expected = (error_bound * error_bound) / 3.0;
  return std::min(10.0 * std::log10((data_range * data_range) / mse_expected),
                  120.0);
}

/** Quantization state, in the last free byte of the same word. */
constexpr uint32_t kQuantShift = 24;
constexpr uint32_t kQuantEnabledBit = 1u;
constexpr uint32_t kQuantPrecisionShift = 1;
constexpr uint32_t kQuantPrecisionMask = 0x3u;
constexpr uint32_t kQuantEscapeBit = 1u << 3;
constexpr uint32_t kQuantFloat64Bit = 1u << 4;
/** Format of a blob using escapes, float64 or 64-bit indices, so a build
 *  without those decoders refuses it instead of misreading it. */
constexpr uint32_t kFormatVersionQuantExt = 2;

inline uint32_t PackQuant(
    bool enabled, const ctp::compress::preprocess::DeviceQuantizeParams &p) {
  if (!enabled) return 0;
  uint32_t code = (p.precision == 8)    ? 0u
                  : (p.precision == 16) ? 1u
                  : (p.precision == 64) ? 3u
                                        : 2u;
  uint32_t bits = kQuantEnabledBit | (code << kQuantPrecisionShift);
  if (p.escapes) bits |= kQuantEscapeBit;
  if (p.elem_bytes == 8) bits |= kQuantFloat64Bit;
  return bits << kQuantShift;
}
/** The stored preset word: PackPreset plus the quantization applied. */
inline uint32_t PackStored(
    uint32_t preset, uint32_t shuffle_elem_size, bool quant,
    const ctp::compress::preprocess::DeviceQuantizeParams &p) {
  uint32_t packed = PackPreset(preset, shuffle_elem_size) | PackQuant(quant, p);
  if (quant && (p.escapes || p.elem_bytes == 8 || p.precision == 64)) {
    packed = (packed & ~(kVersionMask << kVersionShift)) |
             (kFormatVersionQuantExt << kVersionShift);
  }
  return packed;
}
inline bool UnpackQuantEnabled(uint32_t packed) {
  return ((packed >> kQuantShift) & kQuantEnabledBit) != 0;
}
inline int UnpackQuantPrecision(uint32_t packed) {
  uint32_t code = (packed >> (kQuantShift + kQuantPrecisionShift)) &
                  kQuantPrecisionMask;
  return (code == 0) ? 8 : (code == 1) ? 16 : (code == 2) ? 32 : 64;
}
/** Width, escapes and element type of a stored quantized blob. */
inline void UnpackQuantLayout(
    uint32_t packed, ctp::compress::preprocess::DeviceQuantizeParams *p) {
  const uint32_t bits = packed >> kQuantShift;
  p->precision = UnpackQuantPrecision(packed);
  p->escapes = (bits & kQuantEscapeBit) != 0;
  p->elem_bytes = (bits & kQuantFloat64Bit) ? 8 : 4;
}

/** The four doubles a reader needs to invert a quantization, appended directly after the 24-b... */
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
  /** Compressed payload length in bytes, excluding this header. */
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
        // Anything that does not fit records as "not recorded" rather than truncating: a wrong lengt...
        compressed_size_(compressed_size <= UINT32_MAX
                             ? static_cast<uint32_t>(compressed_size)
                             : 0u),
        original_size_(orig_size) {}

  /** Magic matches AND the format is one this build understands. */
  bool IsValid() const {
    return magic_ == kMagic &&
           UnpackVersion(compress_preset_) <= kFormatVersionQuantExt;
  }

  /** Payload length to feed the decompressor, or 0 if unusable. */
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
/** Header bytes for this blob: 24 core, plus 32 when quantized. */
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

/** Bring up prediction reuse: off by default, CLIO_NEUROPRESS_REUSE_PREDICTIONS=1 opts in, and never while exploring. */
void Runtime::InitPredictionReuse() {
  auto env_num = [](const char *name, double fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char *end = nullptr;
    const double parsed = std::strtod(v, &end);
    return end == v ? fallback : parsed;
  };
  // An exploring run never consults reuse (PredictionReuseAllowed), so do not
  // reserve the device state for it either -- and say so in the log.
  if (config_.neuropress_exploration_enabled_) {
    HLOG(kInfo,
         "NeuroPress prediction reuse is OFF for this run because exploration "
         "is enabled: an exploring run ranks and measures the full action "
         "space every timestep, whatever the divergence between them");
    return;
  }

  // Opt in, not out: absent means off. Only an explicit 1/true/yes/on
  // enables it; anything else runs the model for every chunk.
  {
    const char *v = std::getenv("CLIO_NEUROPRESS_REUSE_PREDICTIONS");
    const bool on =
        v != nullptr &&
        (*v == '1' || *v == 't' || *v == 'T' || *v == 'y' || *v == 'Y' ||
         ((*v == 'o' || *v == 'O') && (v[1] == 'n' || v[1] == 'N')));
    if (!on) {
      HLOG(kInfo,
           "NeuroPress prediction reuse is OFF (the default); set "
           "CLIO_NEUROPRESS_REUSE_PREDICTIONS=1 to enable it");
      return;
    }
  }

  /** Opt IN, not out: the device-decided path is the shipped behaviour and stays the default. */
  {
    const char *v = std::getenv("CLIO_NEUROPRESS_REUSE_HOST_DECIDE");
    np_reuse_host_decide_ = (v != nullptr && v[0] == '1');
  }

  np_reuse_thresholds_.step =
      env_num("CLIO_NEUROPRESS_REUSE_STEP_THRESHOLD", np_reuse_thresholds_.step);
  np_reuse_thresholds_.anchor = env_num("CLIO_NEUROPRESS_REUSE_ANCHOR_THRESHOLD",
                                           np_reuse_thresholds_.anchor);
  np_reuse_thresholds_.refresh_interval = static_cast<long long>(env_num(
      "CLIO_NEUROPRESS_REUSE_REFRESH_STEPS",
      static_cast<double>(np_reuse_thresholds_.refresh_interval)));

  /** Distinct (field, chunk index) pairs the run may track. */
  constexpr double kMaxLineageSlots = 1048576.0;  // ~1 GiB of device state
  const double requested =
      env_num("CLIO_NEUROPRESS_REUSE_MAX_LINEAGES", 262144);
  double clamped = requested;
  if (!(clamped >= 1.0)) clamped = 1.0;
  if (clamped > kMaxLineageSlots) clamped = kMaxLineageSlots;
  if (clamped != requested) {
    HLOG(kWarning,
         "CLIO_NEUROPRESS_REUSE_MAX_LINEAGES={} is out of range; using {}",
         requested, clamped);
  }
  const uint32_t capacity = static_cast<uint32_t>(clamped);
  np_reuse_states_ =
      ctp::compress::preprocess::ReuseStatesAlloc(capacity);
  if (np_reuse_states_ == nullptr) {
    // Report and stay off rather than fail the pool: this is an optimisation,
    // and a run that cannot have it should still compress.
    HLOG(kWarning,
         "NeuroPress prediction reuse: {} lineage slots could "
         "not be allocated on the device; continuing with it OFF, so the "
         "model runs for every chunk as usual",
         capacity);
    return;
  }
  np_reuse_registry_ = std::make_unique<
      ctp::compress::preprocess::LineageSlotRegistry>(capacity);
  np_reuse_enabled_ = true;
  HLOG(kInfo,
       "NeuroPress prediction reuse ON, decided on the {}: step={} "
       "anchor={} refresh={} capacity={} lineages ({} MiB on the device)",
       np_reuse_host_decide_ ? "HOST before the forward pass"
                             : "DEVICE after it,",
       np_reuse_thresholds_.step, np_reuse_thresholds_.anchor,
       np_reuse_thresholds_.refresh_interval, capacity,
       (static_cast<size_t>(capacity) *
        sizeof(ctp::compress::preprocess::DevicePredictionReuseState)) /
           (1024 * 1024));
}

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
        InitPredictionReuse();
      } else {
        // NeuroPress has no CPU path -- upstream's network exists only as CUDA kernels -- so a load ...
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

  // Static codec: the control condition.
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
         "and exploration are all disabled (byte shuffle: {}, quantize: {}).",
         config_.neuropress_static_lib_, resolved, wire, pool_name_,
         config_.neuropress_static_shuffle_ == 0
             ? std::string("off")
             : std::to_string(config_.neuropress_static_shuffle_) + "-byte",
         config_.neuropress_static_quantize_ ? "on (needs eb>0)" : "off");
    if (resolved != config_.neuropress_static_lib_) {
      HLOG(kWarning,
           "  '{}' is not a registered library; it fell back to '{}'. Check "
           "the name against CompressionFactory's registry.",
           config_.neuropress_static_lib_, resolved);
    }
    // Upstream exposes ONE width: GPUCOMPRESS_PREPROC_SHUFFLE_4, and its NN encodes shuffle as a...
    if (config_.neuropress_static_quantize_ &&
        (resolved == "cusz" || resolved == "cuszp")) {
      HLOG(kWarning,
           "Static codec '{}' quantizes internally at its own error bound; "
           "ignoring neuropress_static_quantize so the bound is not applied "
           "twice (upstream ignores preprocessing flags for these too).",
           resolved);
      config_.neuropress_static_quantize_ = false;
    }
    if (config_.neuropress_static_shuffle_ != 0 &&
        config_.neuropress_static_shuffle_ != 4) {
      HLOG(kWarning,
           "  byte shuffle {} diverges from upstream, which offers only "
           "4-byte (GPUCOMPRESS_PREPROC_SHUFFLE_4). Results from this run are "
           "not comparable with NeuroPress.",
           config_.neuropress_static_shuffle_);
    }
  }

  // Best mode brings its own exploration settings rather than requiring the caller to assemble...
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

/** Resolve one blob name to a reuse slot. */
ctp::compress::preprocess::PredictionReuseContext Runtime::PredictionReuseContextFor(
    const std::string &blob_name) {
  ctp::compress::preprocess::PredictionReuseContext ctx;
  if (!np_reuse_enabled_ || np_reuse_states_ == nullptr ||
      !np_reuse_registry_) {
    return ctx;
  }
  /** Exploration is never served a reused prediction. */
  if (!ctp::compress::preprocess::PredictionReuseAllowed(
          np_reuse_enabled_, config_.neuropress_exploration_enabled_)) {
    return ctx;
  }

  const auto lineage = ctp::compress::preprocess::ParseBlobLineage(blob_name);
  if (!lineage.resolved) return ctx;

  uint32_t slot;
  {
    std::lock_guard<std::mutex> lock(np_reuse_mutex_);
    slot = np_reuse_registry_->SlotFor(lineage.key);
  }
  if (slot == ctp::compress::preprocess::kNoLineageSlot) return ctx;

  ctx.states = np_reuse_states_;
  ctx.slot = slot;
  ctx.timestep = lineage.timestep;
  ctx.thresholds = np_reuse_thresholds_;
  return ctx;
}

/** The host slot for a lineage, created on first use. */
Runtime::NpHostReuseSlot *Runtime::NpHostReuseSlotFor(uint32_t slot) {
  if (slot == ctp::compress::preprocess::kNoLineageSlot) return nullptr;
  std::lock_guard<std::mutex> lock(np_reuse_mutex_);
  if (slot >= np_reuse_host_.size()) np_reuse_host_.resize(slot + 1);
  auto &p = np_reuse_host_[slot];
  if (!p) p = std::make_unique<NpHostReuseSlot>();
  return p.get();
}

/** Defined with the selection log below; used here to skip work it alone reads. */
std::vector<CompressionStats> Runtime::EstCompressionStats(
    const void* chunk, clio::run::u64 chunk_size, const Context& context,
    bool* out_ranked_by_cost, double* out_entropy, double* out_mad,
    double* out_second_deriv, bool* out_neuropress_gpu_failed,
    const void** out_device_stats,
    const ctp::compress::preprocess::PredictionReuseContext* reuse,
    ctp::compress::preprocess::PredictionReuseOutcome* out_outcome) {
  std::vector<CompressionStats> results;
  if (out_ranked_by_cost) *out_ranked_by_cost = false;
  if (out_neuropress_gpu_failed) *out_neuropress_gpu_failed = false;
  if (out_device_stats) *out_device_stats = nullptr;

  double entropy = 0.0, mad = 0.0, second_derivative_mean = 0.0;

  // Two models, two feature definitions, two ways of reading the same bytes.
  if (NeuroPressActive(context)) {
    std::vector<CompressionStats> neuropress_stats = NeuroPressRankChunk(
        chunk, chunk_size, context, &entropy, &mad, &second_derivative_mean,
        out_entropy, out_mad, out_second_deriv, out_neuropress_gpu_failed,
        out_device_stats, reuse, out_outcome);
    if (!neuropress_stats.empty()) {
      // Already best-first under NeuroPress's cost model; saying so is what
      // stops the caller re-selecting on ratio alone and discarding it.
      if (out_ranked_by_cost) *out_ranked_by_cost = true;
      return neuropress_stats;
    }
    // Declined. Its statistics are still right for this chunk, so the
    // heuristics below reuse them rather than measuring again.
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
    // FIFO-evict the oldest.
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

    // Upstream's label floor is 0.01 ms (nn_gpu.cu:775), not the 1 ms
    // prediction clamp. CLIO_NEUROPRESS_DT_LABEL_FLOOR_MS overrides.
    static const double kLabelFloorMs = [] {
      const char* e = std::getenv("CLIO_NEUROPRESS_DT_LABEL_FLOOR_MS");
      const double v = (e != nullptr && *e != '\0') ? std::atof(e) : 0.01;
      return (v > 0.0) ? v : 0.01;
    }();
    it->second.measured_ms = std::max(kLabelFloorMs, measured_ms);

    // Train over EVERY record that has a measurement, not just this one and not only once per re...
    batch_features.reserve(decomp_features_.size());
    batch_times.reserve(decomp_features_.size());
    for (const auto& entry : decomp_features_) {
      if (entry.second.measured_ms <= 0.0) continue;
      batch_features.push_back(entry.second.features);
      batch_times.push_back(entry.second.measured_ms);
    }
  }
  if (batch_features.empty()) return;

  // One averaged update over the batch, matching upstream's gpucompress_batched_decomp_sgd() -...
  bool trained =
      neuropress_predictor_->TrainDecompHead(batch_features, batch_times);
  HLOG(kDebug, "NeuroPress decomp-head SGD: batch={} trained={}",
       batch_features.size(), trained);
}


/** CLIO_PATH_TRACE and NpWhere() now live in neuropress_path_trace.h, which documents the sev... */


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

    // Strict device residency, when the caller asked for it.
    const void *host_chunk_src =
        (chunk_data != nullptr && !ctp::IsDevicePointer(chunk_data))
            ? chunk_data
            : nullptr;

    // Per-chunk phase log (CLIO_NEUROPRESS_PHASE_LOG).
    const bool phase_log = PhaseLogEnabled();
    ChunkPhases phases;
    bool phases_selected = false;
    double ds_h2d_ms = 0.0;
    double ds_h2d_start_ns = -1.0;   // for the elapsed-staging union
    auto written_time = start_time;
    std::chrono::steady_clock::time_point explore_t0;
    bool explore_ran = false;
    double explore_inner_ms = 0.0;
    auto ms_since = [](std::chrono::steady_clock::time_point t) {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t).count();
    };

    // SetNull(), not the default constructor.
    ctp::ipc::AllocatorId h2d_alloc;
    h2d_alloc.SetNull();
    struct H2dGuard {
      ctp::ipc::AllocatorId *id;
      ~H2dGuard() {
        if (id && !id->IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, *id);
          id->SetNull();
        }
      }
    } h2d_guard{&h2d_alloc};
    if (NeuroPressStageH2D() && chunk_data != nullptr &&
        !ctp::IsDevicePointer(chunk_data)) {
      char *staged = nullptr;
      h2d_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          chunk_size, &staged);
      if (h2d_alloc.IsNull()) {
        HLOG(kError,
             "CLIO_NEUROPRESS_STAGE_H2D: could not allocate {} bytes of device "
             "memory to stage blob '{}' up; failing rather than falling back "
             "to a host path that no longer exists.",
             (unsigned long long)chunk_size, task->blob_name_.str());
        context.compress_lib_ = 0;
        context.dynamic_compress_ = 0;
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }
      // DeviceAwareMemcpy, NOT GpuApi::Memcpy.
      const auto h2d_t0 = std::chrono::steady_clock::now();
      ds_h2d_start_ns = static_cast<double>(
          h2d_t0.time_since_epoch().count());
      ctp::DeviceAwareMemcpy(staged, chunk_data, chunk_size);
      ds_h2d_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - h2d_t0).count();
      chunk_data = staged;
      // Same convention Compress uses to hand a device-resident output back (see where compressed_...
      CLIO_PATH_TRACE("WRITE  staged H2D %llu bytes -> device",
                      (unsigned long long)chunk_size);
    }

    if (NeuroPressRequireDevice() && !ctp::IsDevicePointer(chunk_data)) {
      HLOG(kError,
           "CLIO_NEUROPRESS_REQUIRE_DEVICE is set and blob '{}' ({} bytes) "
           "arrived in HOST memory. Quantization, byte shuffle and codec "
           "selection would all silently take their host paths. Refusing. "
           "Hand the compressor a device pointer (an in-situ adapter, or the "
           "LAMMPS driver's --order device); or set CLIO_NEUROPRESS_STAGE_H2D=1 "
           "to copy the chunk up and run the CUDA kernels on it, as upstream's "
           "host entry point does -- at the cost of an H2D per chunk.",
           task->blob_name_.str(), (unsigned long long)chunk_size);
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Get compression stats.
    bool ranked_by_cost = false;
    double sel_entropy = 0.0, sel_mad = 0.0, sel_second_deriv = 0.0;
    /* Outlives the inference call: the outcome copy is asynchronous and is
       only readable after the synchronize inside it. */
    ctp::compress::preprocess::PredictionReuseContext np_reuse_ctx;
    ctp::compress::preprocess::PredictionReuseOutcome np_reuse_outcome;
    bool neuropress_gpu_failed = false;
    // Statistics the selection ranked on; see out_device_stats.
    const void* sel_device_stats = nullptr;
    std::vector<CompressionStats> stats;
    if (!config_.neuropress_static_lib_.empty()) {
      // Control condition: one candidate, no inference.
      CompressionStats fixed{};
      fixed.compress_lib_ =
          ctp::CompressionFactory::WireIdForName(config_.neuropress_static_lib_);
      // Preset 2 (BALANCED): the GPU codecs are single_mode, so the preset is ignored and the id a...
      fixed.compress_preset_ = static_cast<int>(
          PackPreset(2, config_.neuropress_static_shuffle_) |
          (config_.neuropress_static_quantize_ ? (kQuantEnabledBit << kQuantShift)
                                                : 0u));
      fixed.compression_ratio_ = 1.0;
      stats.push_back(fixed);
      ranked_by_cost = true;  // take stats.front() verbatim below
      // No selection ran, so there are no selection spans to take -- but this
      // chunk still stages, preprocesses, compresses and stores. Open the row
      // anyway, or a fixed-codec run reports no write phases at all and its
      // H2D staging cannot be measured (or subtracted) the way a dynamic
      // run's can.
      phases_selected = phase_log;
    } else {
      /** Prediction reuse for THIS chunk. */
      np_reuse_ctx = PredictionReuseContextFor(task->blob_name_.str());
      /* NN inputs 4 and 3. Without these a cache made for one chunk size
         would be replayed for another, and the signature cannot detect it. */
      np_reuse_ctx.chunk_bytes = static_cast<double>(chunk_size);
      np_reuse_ctx.error_bound = context.error_bound_;
      const bool np_reuse_on =
          np_reuse_ctx.slot != ctp::compress::preprocess::kNoLineageSlot;
      stats =
          EstCompressionStats(chunk_data, chunk_size, context, &ranked_by_cost,
                              &sel_entropy, &sel_mad, &sel_second_deriv,
                              &neuropress_gpu_failed, &sel_device_stats,
                              np_reuse_on ? &np_reuse_ctx : nullptr,
                              np_reuse_on ? &np_reuse_outcome : nullptr);
      phases_selected = phase_log && TakeSelectionPhases(&phases);
    }

    if (neuropress_gpu_failed) {
      // NeuroPress was asked for, the chunk was on the device, and its GPU path could not decide.
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

    // Choose best compression strategy.
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
    // PRIMARY, not final: exploration below can replace this with a different codec, and does.
    CLIO_PATH_TRACE("WRITE  neuropress primary lib=%d (%s) preset=%d",
                    best_lib,
                    ctp::CompressionFactory::NameForWireId(best_lib).c_str(),
                    best_preset);
    // Kept unconditionally now that the trace is a runtime switch: two int copies, and the FINAL...
    const int np_primary_lib = best_lib;
    const int np_primary_preset = best_preset;
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

    // Snapshot the intrinsic data features NeuroPress's own prediction was based on, for a possi...
    int np_diag_slot = -1;
    bool neuropress_feat_valid = false;
    double neuropress_entropy = 0.0, neuropress_mad = 0.0,
           neuropress_second_deriv = 0.0;
    // Best mode needs these too.
    if ((config_.neuropress_online_learning_enabled_ ||
         config_.neuropress_best_mode_) &&
        NeuroPressActive(context)) {
      // FLOAT32 unconditionally, and NOT context.data_type_.
      ctp::DataType feat_type = ctp::DataType::FLOAT32;
      size_t feat_type_size = ctp::DataStatisticsFactory::GetTypeSize(feat_type);
      // Whole chunk -- MUST match EstCompressionStats' scope above, or the features SGD trains on ...
      size_t feat_num_elements = static_cast<size_t>(chunk_size / feat_type_size);
      if (feat_num_elements == 0) feat_num_elements = 1;
      // Same guard as EstCompressionStats: on a device-resident chunk whose stats cannot be comput...
      /** Same call as inference above, for the same reason: training on statistics read from a diff... */
      (void)feat_type;
      (void)feat_num_elements;
      if (sel_device_stats != nullptr) {
        /** The SGD kernel reads the statistics from this device buffer. */
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

    // Defer the store when exploration may replace this pick.
    const bool defer_store = config_.neuropress_exploration_enabled_;
    // Set when the exploration block below performs the one put.
    bool stored_by_exploration = false;

    // HAND COMPRESS THE COPY SELECTION ALREADY STAGED, rather than the host buffer it was made f...
    ctp::ipc::ShmPtr<> compress_input = task->blob_data_;
    if (NeuroPressReuseStagedH2D() && !h2d_alloc.IsNull() &&
        chunk_data != nullptr) {
      compress_input.alloc_id_ = h2d_alloc;
      compress_input.off_ = reinterpret_cast<clio::run::u64>(chunk_data);
    }

    // Now call Compress to perform compression (and PutBlob unless deferred)
    if (phases_selected) OpenCompressPhases(task->blob_name_.str());
    auto compress_task = client_.AsyncCompress(
        clio::run::PoolQuery::Local(), task->tag_id_, task->blob_name_.str(),
        task->offset_, task->size_, compress_input, task->score_, context,
        task->flags_, task->core_pool_id_, defer_store);
    CLIO_CO_AWAIT(compress_task);
    written_time = std::chrono::high_resolution_clock::now();

    // Copy results back
    task->context_ = compress_task->context_;
    task->tier_score_ = compress_task->tier_score_;
    task->return_code_ = compress_task->return_code_;

    CLIO_PATH_TRACE(
        "4 primary  %s ran lib=%d (%s) -- MEASURED ratio=%.2f ct=%.3f ms "
        "(vs PREDICTED ratio=%.2f ct=%.3f ms) rc=%d",
        NpWhere(chunk_data), context.compress_lib_,
        // lib 0 is "stored raw, no codec kept", NOT a codec.
        context.compress_lib_ == 0
            ? "STORED RAW"
            : ctp::CompressionFactory::NameForWireId(context.compress_lib_).c_str(),
        context.actual_compression_ratio_, context.actual_compress_time_ms_,
        stats.empty() ? -1.0 : stats.front().compression_ratio_,
        stats.empty() ? -1.0 : stats.front().compress_time_ms_,
        // GetReturnCode(), not return_code_: the field is a ctp::ipc::atomic<u32> and handing that o...
        (int)task->GetReturnCode());

    // The primary's image, held unstored while exploration runs.
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

    // Record what the model chose for this chunk, before the online-learning and exploration blo...
    {
      const CompressionStats *logged_pred = nullptr;
      for (const auto &stat : stats) {
        if (stat.compress_lib_ == best_lib &&
            stat.compress_preset_ == best_preset) {
          logged_pred = &stat;
          break;
        }
      }
      // FNV-1a over the chunk this selection was made for, so a comparison against another impleme...
      unsigned long long checksum = 0;
      if (SelectionLogEnabled() && chunk_data && chunk_size > 0) {
        std::vector<char> staged;
        // Prefer the host bytes staging was fed, when there were any: the device copy holds the same...
        const unsigned char *p = static_cast<const unsigned char *>(
            host_chunk_src != nullptr ? host_chunk_src : chunk_data);
        if (host_chunk_src == nullptr && ctp::IsDevicePointer(chunk_data)) {
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

    // NeuroPress's online-learning loop, ported.

    // Recorded on EVERY chunk NeuroPress selected for, not only when learning is on -- upstream'...
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
        // One MAPE per predicted metric, plus the cost error, exactly as upstream derives them...
        {
          const auto &f = stats.front();
          // Same ceiling the cost model and the kernel used; a hardcoded 100
          // here would report a MAPE against a differently-clamped ratio.
          const double kMapeCap = NeuroPressResolvedCostWeights().cap;
          const double pred_r = std::min(kMapeCap, f.compression_ratio_);
          const double pred_ct =
              std::max(NeuroPressCost::kMinTimeMs, f.compress_time_ms_);
          const double pred_dt =
              std::max(NeuroPressCost::kMinTimeMs, f.decompress_time_ms_);
          const double act_r = std::min(kMapeCap, context.actual_compression_ratio_);
          const double act_ct = std::max(NeuroPressCost::kMinTimeMs,
                                         context.actual_compress_time_ms_);
          // Decompression is not measured at write time; upstream substitutes the prediction, which ma...
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
        // cost = w0*compress_time + w1*decompress_time + w2*chunk_size/(ratio*bandwidth) -- same for...
        const auto kCw = NeuroPressResolvedCostWeights();
        // Best mode scores I/O alone: it ranks on what a configuration SAVES, not what it costs to g...
        const NeuroPressCost cost{config_.neuropress_best_mode_ ? 0.0 : kCw.ct,
                                  config_.neuropress_best_mode_ ? 0.0 : kCw.dt,
                                  kCw.io, kCw.bw, kCw.cap, chunk_size};
        // Decompress time is not measured at write time (only a later read decompresses it) -- use t...
        double predicted_cost = cost(predicted->compress_time_ms_,
                                     predicted->decompress_time_ms_,
                                     predicted->compression_ratio_);
        double actual_cost = cost(context.actual_compress_time_ms_,
                                  predicted->decompress_time_ms_,
                                  context.actual_compression_ratio_);
        double error_pct = (actual_cost > 0.0)
            ? std::fabs(actual_cost - predicted_cost) / actual_cost
            : 0.0;

        // CLIO_NEUROPRESS_SGD_GATE=raw scores the SGD gate without the time
        // floor; by default it scores the ranked cost, as upstream.
        static const bool kRawSgdGate = [] {
          const char *e = std::getenv("CLIO_NEUROPRESS_SGD_GATE");
          return (e != nullptr && std::strcmp(e, "raw") == 0);
        }();
        double sgd_error_pct = error_pct;
        if (kRawSgdGate) {
          const double raw_pred = cost.Raw(predicted->compress_time_ms_,
                                           predicted->decompress_time_ms_,
                                           predicted->compression_ratio_);
          const double raw_act = cost.Raw(context.actual_compress_time_ms_,
                                          predicted->decompress_time_ms_,
                                          context.actual_compression_ratio_);
          sgd_error_pct = (raw_act > 0.0)
              ? std::fabs(raw_act - raw_pred) / raw_act
              : 0.0;
        }

        // ---- Phase 1: "learn from PRIMARY result immediately" -- online SGD on the real, just-meas...
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

          // best_preset is PACKED (preset | shuffle_elem << 8).
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
          // Unconditional, not `quantize ?
          candidate.error_bound = context.error_bound_;
          candidate.library_name = lib_name;

          ctp::compress::model::CompressionFeatures chunk_features =
              ctp::compress::model::MakeCompressionFeatures(data, candidate);

          // Stash for the deferred decomp-head pass.
          RecordDecompFeatures(task->blob_name_.str(), chunk_features);


          // Withheld under best mode: that mode replaces the model's choice on every chunk, so trainin...
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
          const bool np_cost_gate = sgd_error_pct > np_cost_thresh;

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
            // PSNR label for the PRIMARY sample.
            const double primary_sgd_psnr =
                (context.actual_psnr_db_ > 0.0) ? context.actual_psnr_db_
                                                : 120.0;
            std::vector<ctp::compress::model::TrainingLabels> labels = {
                ctp::compress::model::TrainingLabels(
                    static_cast<float>(context.actual_compression_ratio_),
                    static_cast<float>(primary_sgd_psnr),
                    static_cast<float>(context.actual_compress_time_ms_),
                    // 0 when MEASURE_DT did not measure it
                    static_cast<float>(
                        (context.actual_decompress_time_ms_ > 0.0)
                            ? context.actual_decompress_time_ms_
                            : 0.0))};

            // This chunk's own statistics, from the host features (see below).
            const auto sgd_t0 = std::chrono::steady_clock::now();
            // NOT sel_device_stats. That is a per-thread scratch buffer, and this
            // chunk suspended on its compression (CLIO_CO_AWAIT above) while other
            // chunks' selections on the same worker overwrote it: an SGD step
            // reading it trained on ANOTHER chunk's features (97.8% of steps on
            // AI, CUDA_INFERENCE_AUDIT.md P0). `features` carries this chunk's own
            // statistics, read from that buffer right after selection, before
            // any suspension; nullptr makes the kernel use them.
            bool trained = neuropress_predictor_->TrainDeviceStats(
                features, labels, nullptr);
            phases.sgd_ms += ms_since(sgd_t0);
            if (trained) ++phases.sgd_updates;
            /** kModelChanged for the host-decided reuse path. */
            if (trained) {
              np_sgd_epoch_.fetch_add(1, std::memory_order_relaxed);
            }
            HLOG(kDebug,
                 "NeuroPress SGD: lib={} preset={} error_pct={} "
                 "threshold={} trained={}",
                 lib_name, best_preset, sgd_error_pct,
                 config_.neuropress_mape_threshold_, trained);
          }
        }

        // ---- Phase 2: learn from exploration results separately.
        {
          const double np_thresh =
              static_cast<double>(config_.neuropress_exploration_threshold_);
          CLIO_PATH_TRACE(
              "5 gate     error_pct=%.6f vs threshold=%.6f -> %s",
              error_pct, np_thresh,
              !config_.neuropress_exploration_enabled_
                  ? "SKIP -- exploration disabled"
              : config_.neuropress_best_mode_
                  ? "EXPLORE -- best mode, gate bypassed"
              : (error_pct > np_thresh)
                  ? "EXPLORE"
                  : "SKIP -- the prediction was good enough (strict >)");
        }
        if (config_.neuropress_exploration_enabled_ &&
            (config_.neuropress_best_mode_ ||
             error_pct > static_cast<double>(
                             config_.neuropress_exploration_threshold_))) {
          explore_t0 = std::chrono::steady_clock::now();
          explore_ran = true;
          // K bounds the RANKED WINDOW scanned, not the number measured, and an ineligible slot inside...
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
          // Per-sample cost, kept parallel to the two vectors above purely so the batch can be ordered...
          std::vector<double> explore_costs;
          std::vector<double> explore_pred_costs;

          // ---- The PRIMARY's decompression time, measured the same way the alternatives' will be.
          const double primary_dt_ms = context.actual_decompress_time_ms_;
          double primary_rank_cost = actual_cost;
          if (primary_dt_ms >= 0.0) {
            primary_rank_cost = cost(context.actual_compress_time_ms_,
                                     primary_dt_ms,
                                     context.actual_compression_ratio_);
          }
          double best_cost = primary_rank_cost;  // seeded with the primary's own

          // Device shuffle scratch for the explored candidates; released
          // together once the loop is done.
          std::vector<ctp::ipc::AllocatorId> explore_gpu_scratch;
          // The NN often ranks a lossless action first, so the primary path never calls QuantizeDevice...
          bool alt_refusal_reported = false;

          // Best explored alternative so far, if any beat the primary.
          ExploreWinner winner;

          std::vector<std::unique_ptr<ExploreSlot>> slots;
          slots.reserve(alternatives.size());

          // Codec launches in flight at once.

          // Default 4, diverging from upstream deliberately.

          // The cost model ranks on those times, so the inflation can hand the
          // sweep to the wrong codec. 4 buys that for ~19% of sweep wall clock.
          static const size_t kExploreStreams = [] {
            const char *e = std::getenv("CLIO_NEUROPRESS_EXPLORE_STREAMS");
            if (e == nullptr || *e == '\0') return size_t{4};
            const long v = std::atol(e);
            return v > 0 ? static_cast<size_t>(v) : size_t{0};  // 0 = all
          }();

          CLIO_PATH_TRACE(
              "6 explore  sweep opening: K=%d, %zu alternative(s) to MEASURE, "
              "%zu concurrent stream(s); the NN is NOT re-run -- each carries "
              "the prediction step 2 already made for it",
              config_.neuropress_exploration_k_, alternatives.size(),
              kExploreStreams);
          // The three steps below are the SWEEP's, not the two learning phases' -- this whole sweep ru...
          for (const auto* alt : alternatives) {
            std::string alt_name =
                ctp::CompressionFactory::NameForWireId(alt->compress_lib_);
            // Packed, like every other compress_preset_ that came out of the ranking -- unpack before co...
            const uint32_t alt_preset_id =
                UnpackPreset(static_cast<uint32_t>(alt->compress_preset_));
            const bool alt_wants_quant =
                UnpackQuantEnabled(static_cast<uint32_t>(alt->compress_preset_));
            // Set when this candidate asks for a transform that is CUDA-only and its buffer is host-resi...
            bool alt_skip = false;
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

            // The slot -- and therefore its stream -- has to exist BEFORE the preprocessing below, becau...
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
            // The ORIGINAL bytes, before this candidate's quantize/shuffle replace alt_input.
            const void* alt_orig_device = alt_input;

            // Apply the alternative's OWN quantization first, exactly as the primary does and as upstrea...
            size_t alt_compress_size = chunk_size;
            bool alt_applied_quant = false;
            ctp::compress::preprocess::DeviceQuantizeParams alt_quant_params;
            // Same gates as the primary: preproc bit + positive bound, with the buffer treated as float3...
            const size_t alt_quant_elem = QuantElemBytes(context.data_type_);
            const bool alt_want_quant =
                alt_wants_quant && context.error_bound_ > 0.0 &&
                alt_quant_elem != 0 && chunk_size >= alt_quant_elem &&
                (chunk_size % alt_quant_elem) == 0;
            std::vector<char> alt_quant_staging;
            if (alt_want_quant && ctp::IsDevicePointer(alt_input)) {
              char* alt_q_buf = nullptr;
              ctp::ipc::AllocatorId alt_q_alloc =
                  CLIO_IPC->AllocateAndRegisterGpuBackend(
                      /*gpu_id=*/0,
                      clio::run::gpu::IpcManager::MemKind::kDeviceMem,
                      ctp::compress::preprocess::QuantizeCapacity(
                          chunk_size / alt_quant_elem, alt_quant_elem),
                      &alt_q_buf);
              size_t alt_q_bytes = 0;
              if (!alt_q_alloc.IsNull()) {
                explore_gpu_scratch.push_back(alt_q_alloc);
                if (ctp::compress::preprocess::QuantizeDevice(
                        alt_input, chunk_size / alt_quant_elem, alt_quant_elem,
                        context.error_bound_, alt_q_buf, &alt_q_bytes,
                        &alt_quant_params, alt_stream)) {
                  alt_input = alt_q_buf;
                  alt_compress_size = alt_q_bytes;
                  alt_applied_quant = true;
                } else if (!alt_refusal_reported) {
                  alt_refusal_reported = true;
                  ReportQuantizeRefusal(alt_quant_params.refusal,
                                        context.error_bound_, chunk_size);
                }
              }
            } else if (alt_want_quant) {
              // Host-resident alternative, and the transform is CUDA-only.
              RefuseHostPreprocess("quantization", chunk_size);
              alt_skip = true;
            }

            // Apply the alternative's OWN byte-shuffle before measuring it.
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
                  // Exploration output is never stored, so the scratch can be released as soon as this candida...
                  explore_gpu_scratch.push_back(alt_shuf_alloc);
                }
              } else {
                // Same reasoning as the quantize branch above.
                RefuseHostPreprocess("byte shuffle", alt_compress_size);
                alt_skip = true;
              }
            }

            if (alt_skip) continue;  // see alt_skip's declaration

            size_t alt_worst_case = alt_compress_size + (alt_compress_size / 20) + 1024;

            // Compress into DEVICE memory when the input is device-resident, as upstream does.
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
            // Only useful when it really is device memory: ComputeQualityDevice
            // refuses a host pointer outright rather than measuring on the CPU.
            slot->orig_device =
                ctp::IsDevicePointer(alt_orig_device) ? alt_orig_device : nullptr;
            slot->orig_bytes = chunk_size;
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

            // Launch on the slot's stream, behind its own preprocessing.
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
            if (slot->gpu != nullptr) {
              slot->launched = slot->gpu->CompressLaunch(
                  slot->out_ptr, slot->capacity, slot->input,
                  slot->compress_size, &slot->async);
              if (!slot->launched) slot->gpu = nullptr;
            }
#endif
            slots.push_back(std::move(slot));

            // Batched: drain once the in-flight count reaches the limit, so
            // the next launches start against an idle device.
            if (kExploreStreams != 0) {
              size_t in_flight = 0;
              for (const auto& sp : slots) {
                if (!sp->collected) ++in_flight;
              }
              if (in_flight >= kExploreStreams) CollectExploreSlots(slots);
            }
          }

          // ---- Sweep step 2: collect whatever is still in flight ----
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
            // Decompression time is the PRIMARY's prediction, held constant across every alternative -- ...
            const double alt_dt = (slot_ref.decomp_time_ms >= 0.0)
                                      ? slot_ref.decomp_time_ms
                                      : predicted->decompress_time_ms_;
            CLIO_PATH_TRACE(
                "6 explore  %s rank=%d %-16s q=%d sh=%u | PREDICTED "
                "ratio=%.2f ct=%.3f | MEASURED ratio=%.2f ct=%.3f dt=%.3f%s",
                NpWhere(alt_out_ptr), explore_rank, alt_name.c_str(),
                alt_applied_quant ? 1 : 0, (unsigned)alt_applied_shuffle,
                alt->compression_ratio_, alt->compress_time_ms_, alt_ratio,
                alt_time_ms, alt_dt,
                (slot_ref.decomp_time_ms >= 0.0)
                    ? ""
                    : " (dt = the PRIMARY's prediction, not measured)");
            double alt_cost = cost(alt_time_ms, alt_dt, alt_ratio);
            if (alt_cost < best_cost) {
              best_cost = alt_cost;
              // Adopt it.
              const size_t winner_total =
                  alt_compressed_size + sizeof(CompressionHeader);
              if (winner_total < chunk_size) {
                winner.have = true;
                // The row for THIS candidate has not been pushed yet, so its index is the current size....
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
                winner.dt_ms = slot_ref.decomp_time_ms;
                // Carry the quantization state too.
                winner.quant = alt_applied_quant;
                winner.quant_params = alt_quant_params;
                CLIO_PATH_TRACE(
                    "7 adopt    %s rank=%d %s q=%d sh=%u ratio=%.2f "
                    "cost=%.6f beats the running best -- REPLACES the primary "
                    "(payload D2H'd to host for the store)",
                    NpWhere(alt_out_ptr), explore_rank, alt_name.c_str(),
                    alt_applied_quant ? 1 : 0, (unsigned)alt_applied_shuffle,
                    alt_ratio, alt_cost);
              }
            }

            if (ExploreLogEnabled()) {
              explore_rows.push_back(ExploreRow{
                  alt_name, alt_preset_id, alt_applied_quant,
                  alt_applied_shuffle, alt->compression_ratio_,
                  alt->compress_time_ms_, alt->decompress_time_ms_, alt_ratio,
                  alt_time_ms,
                  alt_applied_quant
                      ? AnalyticalPsnr(alt_quant_params.data_max -
                                           alt_quant_params.data_min,
                                       alt_quant_params.effective_error_bound)
                      : -1.0,
                  alt_cost, explore_rank, slot_ref.decomp_time_ms,
                  slot_ref.quality, slot_ref.have_quality,
                  alt_quant_params.refusal});
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
            // Credit what was ACTUALLY applied: a declined shuffle (wrong size multiple, failed allocati...
            alt_candidate.byte_shuffle = alt_applied_shuffle != 0;
            alt_candidate.quantize = alt_applied_quant;
            // The configured bound, not `applied ?
            alt_candidate.error_bound = context.error_bound_;
            alt_candidate.library_name = alt_name;

            explore_features.push_back(
                ctp::compress::model::MakeCompressionFeatures(alt_data,
                                                               alt_candidate));
            // PSNR per explored slot, as upstream computes it (gpucompress_compress.cpp, analytical_psnr...
            const double alt_psnr =
                alt_applied_quant
                    ? AnalyticalPsnr(
                          alt_quant_params.data_max - alt_quant_params.data_min,
                          alt_quant_params.effective_error_bound)
                    : -1.0;
            // The measured dt, so the trust region bounds head 1 too.
            explore_labels.emplace_back(static_cast<float>(alt_ratio),
                                        static_cast<float>(alt_psnr),
                                        static_cast<float>(alt_time_ms),
                                        static_cast<float>(
                                            (slot_ref.decomp_time_ms >= 0.0)
                                                ? slot_ref.decomp_time_ms
                                                : 0.0));
            explore_costs.push_back(alt_cost);
            explore_pred_costs.push_back(cost(alt->compress_time_ms_,
                                              alt->decompress_time_ms_,
                                              alt->compression_ratio_));
          }

          // Input 3 as the MODEL sees it.
          auto eb_for_log = [&](bool quant) {
            return quant ? context.error_bound_ : 1e-7;
          };
          // The model's own pick, logged alongside the alternatives so every action for a chunk sits i...
          if (ExploreLogEnabled()) {
            const uint32_t p_packed = static_cast<uint32_t>(best_preset);
            // The measurement Runtime::Compress took, which this function already awaited (line 1388)....
            ctp::compress::preprocess::QualityMetrics primary_qm;
            const bool have_primary_qm =
                TakePrimaryQuality(task->blob_name_.str(), &primary_qm);
            // Parked by Compress when it declined the primary's quantize.
            ctp::compress::preprocess::QuantizeRefusal primary_refusal =
                ctp::compress::preprocess::QuantizeRefusal::kNone;
            TakePrimaryQuantizeRefusal(task->blob_name_.str(),
                                       &primary_refusal);
            LogNeuroPressExplore(
                task->blob_name_.str(), chunk_size, /*rank=*/-1,
                ctp::CompressionFactory::NameForWireId(best_lib),
                UnpackPreset(p_packed), UnpackQuantEnabled(p_packed),
                UnpackShuffle(p_packed),
                predicted->compression_ratio_, predicted->compress_time_ms_,
                predicted->decompress_time_ms_,
                context.actual_compression_ratio_,
                context.actual_compress_time_ms_, context.actual_psnr_db_,
                primary_rank_cost, primary_rank_cost,
                /*adopted=*/!winner.have, /*is_primary=*/true, primary_dt_ms,
                neuropress_entropy, neuropress_mad, neuropress_second_deriv,
                eb_for_log(UnpackQuantEnabled(p_packed)), primary_refusal,
                have_primary_qm ? &primary_qm : nullptr);
          }
          for (size_t ri = 0; ri < explore_rows.size(); ++ri) {
            const ExploreRow& row = explore_rows[ri];
            LogNeuroPressExplore(
                task->blob_name_.str(), chunk_size, row.rank, row.lib,
                row.preset_id, row.quant, row.shuffle, row.pred_ratio,
                row.pred_ct, row.pred_dt, row.ratio, row.ct_ms, row.psnr,
                row.cost,
                // primary_rank_cost, NOT actual_cost: the baseline every one of these rows was actually rank...
                primary_rank_cost, static_cast<int>(ri) == winner.row,
                /*is_primary=*/false, row.dt_ms,
                neuropress_entropy, neuropress_mad, neuropress_second_deriv,
                eb_for_log(row.quant), row.refusal,
                row.have_quality ? &row.quality : nullptr);
          }

#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
          // Release each slot's stream, events and any temporary output.
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

          // ---- Adopt the winner (upstream's "Write winner to output") ---- When exploration finds a ...
          if (winner.have) {
            const size_t hdr_size =
                sizeof(CompressionHeader) +
                (winner.quant ? sizeof(QuantHeaderExtension) : 0);
            const size_t winner_total = winner.payload.size() + hdr_size;
            CompressionHeader winner_header(
                static_cast<uint32_t>(winner.lib),
                PackStored(winner.preset_id, winner.shuffle, winner.quant,
                           winner.quant_params),
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
                  PackStored(winner.preset_id, winner.shuffle, winner.quant,
                             winner.quant_params));
              winner_ctx.transform_flags_ |=
                  clio::cte::core::kBlobTransformed |
                  clio::cte::core::kBlobTransformCompressed;

              const auto winner_put_t0 = std::chrono::high_resolution_clock::now();
              const auto winner_put_s0 = std::chrono::steady_clock::now();
              auto winner_put = core_client_->AsyncPutBlob(
                  task->tag_id_, task->blob_name_.str(), task->offset_,
                  winner_total, winner_shm.shm_.template Cast<void>(),
                  task->score_, winner_ctx, task->flags_,
                  clio::run::PoolQuery::Local());
              CLIO_CO_AWAIT(winner_put);
              written_time = std::chrono::high_resolution_clock::now();
              if (phases_selected) {
                ChunkPhases put_io;
                put_io.io_ms = std::chrono::duration<double, std::milli>(
                                   written_time - winner_put_t0).count();
                put_io.io_start_ns = static_cast<double>(
                    winner_put_s0.time_since_epoch().count());
                explore_inner_ms += put_io.io_ms;
                AddCompressPhases(task->blob_name_.str(), put_io);
              }
              const int winner_rc = winner_put->return_code_;
              CLIO_IPC->FreeBuffer(winner_shm);

              if (winner_rc != 0) {
                // The primary's blob is still intact -- a failed overwrite leaves the earlier whole-blob put...
                HLOG(kWarning,
                     "NeuroPress explore: winner put failed (rc={}); keeping "
                     "the primary's stored result",
                     winner_rc);
              } else {
                stored_by_exploration = true;
                // Supersede the primary's payload-log row.
                if (SelectionLogEnabled()) {
                  LogCompressedPayload(
                      task->blob_name_.str(), winner.payload.data(),
                      winner.payload.size(), /*on_device=*/false,
                      winner_total < chunk_size, winner.time_ms, "adopted");
                }
                // THE SELECTION LOG IS WRITTEN BEFORE THE SWEEP RUNS, so its row names the model's pick and ...
                if (SelectionLogEnabled()) {
                  LogNeuroPressSelection(
                      task->blob_name_.str(), chunk_size, neuropress_entropy,
                      neuropress_mad, neuropress_second_deriv, winner.lib,
                      static_cast<int>(
                          PackStored(winner.preset_id, winner.shuffle,
                                     winner.quant, winner.quant_params)),
                      /*predicted=*/nullptr, winner.ratio, winner.time_ms,
                      /*actual_psnr=*/-1.0,
                      // The chunk's checksum is on the primary's row; the
                      // input bytes are the same one, so it is not repeated.
                      /*checksum=*/0ull, "adopted");
                }
                task->context_ = winner_ctx;
                task->context_.actual_original_size_ = chunk_size;
                task->context_.actual_compressed_size_ = winner_total;
                task->context_.actual_compression_ratio_ = winner.ratio;
                task->context_.actual_compress_time_ms_ = winner.time_ms;
                // The phase log parked the PRIMARY's kernel time; the bytes
                // on the tier are the winner's, so the log must name one
                // codec and time the same one.
                SetCompressMs(task->blob_name_.str(), winner.time_ms);
                // ...and dt, or the context reports the PRIMARY's
                // decompression time next to the WINNER's codec name.
                task->context_.actual_decompress_time_ms_ = winner.dt_ms;
                // The stored bytes are now the winner's, so the features the deferred decomp head will join ...
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
                  // Feeds the deferred decomp head, whose upstream counterpart reads DeferredDecompSample::err...
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

          // Withheld under best mode, same reason phase 1 is: these samples come from a sweep the mode...
          if (!explore_features.empty() && !config_.neuropress_best_mode_) {
            // Order by ascending cost and keep at most the cheapest 7, matching upstream's SGD phase 2....
            constexpr size_t kMaxExploreSgdSamples = 7;  // NN_MAX_SGD_SAMPLES-1
            std::vector<size_t> order(explore_features.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(),
                             [&](size_t a, size_t b) {
                               return explore_costs[a] < explore_costs[b];
                             });
            // CLIO_NEUROPRESS_EXPLORE_SGD_HARD=N (default 3, 0 = upstream): keep the
            // 7-N cheapest, then the N most mispredicted, |log(pred/measured)|.
            static const size_t kHardSamples = [] {
              const char *e = std::getenv("CLIO_NEUROPRESS_EXPLORE_SGD_HARD");
              const long v = (e != nullptr && *e != '\0') ? std::atol(e) : 3;
              return static_cast<size_t>(
                  std::clamp<long>(v, 0, static_cast<long>(kMaxExploreSgdSamples)));
            }();
            if (kHardSamples > 0 && order.size() > kMaxExploreSgdSamples &&
                explore_pred_costs.size() == explore_costs.size()) {
              const size_t keep_cheap = kMaxExploreSgdSamples - kHardSamples;
              auto miss = [&](size_t i) {
                return (explore_costs[i] > 0.0 && explore_pred_costs[i] > 0.0)
                           ? std::fabs(std::log(explore_pred_costs[i] /
                                                explore_costs[i]))
                           : 0.0;
              };
              std::stable_sort(order.begin() + static_cast<long>(keep_cheap),
                               order.end(), [&](size_t a, size_t b) {
                                 return miss(a) > miss(b);
                               });
            }
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

            // Same chunk, so the same statistics -- exploration varies the ACTION, not the data.
            phases.explored = static_cast<int>(alternatives.size());
            const auto explore_sgd_t0 = std::chrono::steady_clock::now();
            // nullptr for the same reason as the primary step: the device
            // buffer may belong to another chunk by now; the host features
            // carry this chunk's statistics.
            bool explore_trained = neuropress_predictor_->TrainDeviceStats(
                explore_features, explore_labels, nullptr);
            const double explore_sgd_ms = ms_since(explore_sgd_t0);
            phases.sgd_ms += explore_sgd_ms;
            explore_inner_ms += explore_sgd_ms;
            if (explore_trained) ++phases.sgd_updates;
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

    if (explore_ran) {
      phases.explore_ms = std::max(0.0, ms_since(explore_t0) - explore_inner_ms);
    }

    // The decision that actually reaches storage.
    {
      const int np_final_lib = context.compress_lib_;
      const int np_final_preset = context.compress_preset_;
      // Blob name included so this can be joined per chunk against the codec the READ side recover...
      CLIO_PATH_TRACE(
          "WRITE  neuropress FINAL blob='%s' lib=%d (%s) preset=%d %s",
          task->blob_name_.str().c_str(),
          np_final_lib,
          ctp::CompressionFactory::NameForWireId(np_final_lib).c_str(),
          np_final_preset,
          np_final_lib == 0
              ? "(STORED RAW -- no codec kept)"
          : (np_final_lib == np_primary_lib &&
             np_final_preset == np_primary_preset)
              ? "(primary kept)"
              : "(EXPLORATION OVERRODE THE PRIMARY)");
    }

    // The one put, when exploration did not make it.
    if (defer_store && !stored_by_exploration && primary_image.valid()) {
      const auto primary_put_t0 = std::chrono::high_resolution_clock::now();
      const auto primary_put_s0 = std::chrono::steady_clock::now();
      auto primary_put = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_,
          primary_image.size, primary_image.data, task->score_,
          task->context_, task->flags_, clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(primary_put);
      written_time = std::chrono::high_resolution_clock::now();
      if (phases_selected) {
        ChunkPhases put_io;
        put_io.io_ms = std::chrono::duration<double, std::milli>(
                           written_time - primary_put_t0).count();
        put_io.io_start_ns = static_cast<double>(
            primary_put_s0.time_since_epoch().count());
        AddCompressPhases(task->blob_name_.str(), put_io);
      }
      if (primary_put->return_code_ != 0) {
        // Nothing else stored this blob, so unlike a failed exploration put there is no earlier copy...
        HLOG(kError,
             "DynamicSchedule: deferred store of '{}' failed (rc={}); the "
             "chunk was compressed but is NOT in the tier",
             task->blob_name_.str(), primary_put->return_code_);
        task->return_code_ = primary_put->return_code_;
      }
    }
    primary_image.release();

    // Always take the parked entry; log only a chunk that reached the tier.
    if (phases_selected &&
        TakeCompressPhases(task->blob_name_.str(), &phases) &&
        task->return_code_ == 0) {
      phases.h2d_ms = std::max(phases.h2d_ms, 0.0) + ds_h2d_ms;
      // Earliest of the two staging points, so the interval covers both.
      if (ds_h2d_start_ns >= 0.0 &&
          (phases.h2d_start_ns < 0.0 || ds_h2d_start_ns < phases.h2d_start_ns)) {
        phases.h2d_start_ns = ds_h2d_start_ns;
      }
      LogChunkPhases(task->blob_name_.str(), "write", chunk_size,
                     task->context_.compress_lib_, phases,
                     std::chrono::duration<double, std::milli>(
                         written_time - start_time).count(),
                     task->context_.compress_lib_ != 0
                         ? task->context_.actual_compressed_size_
                         : chunk_size);
    }

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
    // Neither this chimod's own compose config nor the caller supplied a core pool to store into...
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
    // The selector's quantize bit.
    const bool quantize_requested = UnpackQuantEnabled(packed_preset);

    ctp::CompressionPreset preset = ctp::CompressionPreset::BALANCED;
    if (preset_id == 1) {
      preset = ctp::CompressionPreset::FAST;
    } else if (preset_id == 3) {
      preset = ctp::CompressionPreset::BEST;
    }

    // Create compressor with specified preset
    const auto factory_t0 = std::chrono::steady_clock::now();
    auto compressor = ctp::CompressionFactory::GetPreset(library_name, preset);
    // THE REQUESTED BOUND, not the preset's. GetPreset builds a lossy codec
    // from a three-value menu (FAST 1e-2 / BALANCED 1e-3 / BEST 1e-4); the
    // caller's actual error bound never reached it, so any request off that
    // menu silently ran at the nearest preset while the verifier checked the
    // number that was asked for. A lossless codec returns false and is
    // unaffected.
    if (compressor && context.error_bound_ > 0.0) {
      compressor->SetErrorBound(context.error_bound_);
    }
    ChunkPhases compress_phases;  // phase log
    compress_phases.factory_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - factory_t0).count();
    auto park_compress_phases = [&](double io_ms) {
      if (!PhaseLogEnabled()) return;
      compress_phases.io_ms = io_ms;
      compress_phases.preproc_ms = context.actual_preproc_time_ms_;
      compress_phases.h2d_ms = context.actual_h2d_time_ms_;
      // h2d_start_ns is already set where the copy happened, if it happened.
      AddCompressPhases(task->blob_name_.str(), compress_phases);
    };

    if (!compressor) {
      HLOG(kWarning, "Failed to create compressor for library: {}",
           library_name);
      task->return_code_ = 3;  // Compressor creation failed
      CLIO_CO_RETURN;
    }

    // Core header now; grows by the quantization extension below if the quantize action actually...
    size_t header_size = sizeof(CompressionHeader);
    // Worst-case compressed size: original size + 5% overhead.
    size_t worst_case_size = input_size + (input_size / 20) + 1024;
    // Ask the codec what IT needs.
    if (compressor != nullptr) {
      const size_t codec_wants = compressor->MaxCompressedSize(input_size);
      if (codec_wants > worst_case_size) worst_case_size = codec_wants;
    }

    // Convert ShmPtr to raw pointer via FullPtr
    auto input_fullptr =
        CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());
    char* input_ptr = input_fullptr.ptr_;

    // Upstream's route for a host-resident caller, applied HERE as well as in DynamicSchedule.
    ctp::ipc::AllocatorId compress_h2d_alloc;
    struct CompressH2dGuard {
      ctp::ipc::AllocatorId *id;
      ~CompressH2dGuard() {
        if (id && !id->IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, *id);
          *id = ctp::ipc::AllocatorId();
        }
      }
    } compress_h2d_guard{&compress_h2d_alloc};
    if (NeuroPressStageH2D() && input_ptr != nullptr &&
        !ctp::IsDevicePointer(input_ptr)) {
      char *staged_in = nullptr;
      compress_h2d_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          input_size, &staged_in);
      if (compress_h2d_alloc.IsNull()) {
        HLOG(kError,
             "CLIO_NEUROPRESS_STAGE_H2D: could not allocate {} bytes to stage "
             "this chunk up for compression; failing rather than falling back "
             "to a host path that no longer exists.",
             (unsigned long long)input_size);
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }
      {
        // Same instrument as the codec and preprocessing kernels.
        const double h2d_start_ns = static_cast<double>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        if (compress_phases.h2d_start_ns < 0.0 ||
            h2d_start_ns < compress_phases.h2d_start_ns) {
          compress_phases.h2d_start_ns = h2d_start_ns;
        }
#if CTP_ENABLE_CUDA
        ctp::CodecKernelTimer _kt(nullptr, &context.actual_h2d_time_ms_);
#endif
        ctp::GpuApi::Memcpy(staged_in, input_ptr, input_size);
      }
      input_ptr = staged_in;
    }

    CLIO_PATH_TRACE("WRITE  Compress ENTRY blob='%s' bytes=%llu device=%d",
                    task->blob_name_.str().c_str(),
                    (unsigned long long)input_size,
                    ctp::IsDevicePointer(input_ptr) ? 1 : 0);

    // GPU-native libraries (nvcomp/cusz/cuszp/ndzip) accept a device pointer directly -- they st...
    std::vector<char> device_staging;
    // Short-circuited on the trace flag: unlike the other hoisted locals this one costs a...
    const bool np_dev_pre_stage =
        NpTraceEnabled() && ctp::IsDevicePointer(input_ptr);
    // The ORIGINAL bytes for the PRIMARY, captured before quantize (:2985) and shuffle (:3047) r...
    const void* primary_orig_device = nullptr;
    input_ptr = ctp::CompressionFactory::StageInputIfNeeded(
        input_ptr, input_size, context.compress_lib_, device_staging);
    // Only when it really is device memory: ComputeQualityDevice refuses a host pointer rather t...
    primary_orig_device =
        ctp::IsDevicePointer(input_ptr) ? input_ptr : nullptr;
    CLIO_PATH_TRACE(
        "WRITE  StageInputIfNeeded lib=%d (%s) device_in=%d device_out=%d %s",
        context.compress_lib_,
        ctp::CompressionFactory::NameForWireId(context.compress_lib_).c_str(),
        np_dev_pre_stage ? 1 : 0, ctp::IsDevicePointer(input_ptr) ? 1 : 0,
        (np_dev_pre_stage && !ctp::IsDevicePointer(input_ptr))
            ? "*** HOST FALLBACK: full D2H of the payload ***"
            : "(no copy)");

    // Byte-shuffle preprocessing, when the ranked candidate asked for it.
    std::vector<char> shuffle_staging;
    char *shuffle_device_buf = nullptr;
    ctp::ipc::AllocatorId shuffle_device_alloc;
    // Declared up here, alongside the shuffle scratch, so a single guard
    // owns both device allocations.
    ctp::ipc::AllocatorId device_output_alloc_id;
    ctp::ipc::AllocatorId quant_device_alloc;
    uint32_t applied_shuffle = 0;

    // Bytes the CODEC sees.
    size_t compress_input_size = input_size;
    bool applied_quant = false;
    std::vector<char> quant_staging;  // host quantize path only
    ctp::compress::preprocess::DeviceQuantizeParams quant_params;

    // Releases both on EVERY exit from this scope.
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

    // ---- Quantization, BEFORE the shuffle ---- Upstream's order is quantize then shuffle, and ...
    const size_t quant_elem = QuantElemBytes(context.data_type_);
    if (quantize_requested && context.error_bound_ > 0.0 && quant_elem == 0) {
      WarnQuantizeUntyped(context.data_type_);
    }
    const bool want_quant = quantize_requested && context.error_bound_ > 0.0 &&
                            quant_elem != 0 && input_size >= quant_elem &&
                            (input_size % quant_elem) == 0;
    if (want_quant && ctp::IsDevicePointer(input_ptr)) {
      char *quant_buf = nullptr;
      const size_t quant_elems = input_size / quant_elem;
      quant_device_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          ctp::compress::preprocess::QuantizeCapacity(quant_elems, quant_elem),
          &quant_buf);
      size_t quant_bytes = 0;
      // Same CUDA-event instrument as the codec, so the two halves of clio_s are comparable.
      // REVERTED, AND WHY IT MUST STAY REVERTED UNTIL THE PATH SYNCHRONISES
      // ITSELF: recording on the legacy default stream implicitly synchronises
      // every blocking stream in the context, and the preprocessing path
      // depends on that barrier. Moving these events to the kernel's own
      // DeviceStatsStream() removed it and corrupted data -- AI lost 1832 of
      // 3608 blobs on one rep, 1 of 3608 on another, silently and
      // non-deterministically. The measurement defect is real (the bracket
      // times a device-wide drain, 312x on a 4-worker repro) but it is
      // load-bearing; fix the missing synchronisation first, then move it.
      bool _q_ok;
      {
#if CTP_ENABLE_CUDA
        ctp::CodecKernelTimer _kt(nullptr, &context.actual_preproc_time_ms_);
#endif
        _q_ok = !quant_device_alloc.IsNull() &&
            ctp::compress::preprocess::QuantizeDevice(
                input_ptr, quant_elems, quant_elem, context.error_bound_,
                quant_buf, &quant_bytes, &quant_params);
      }
      if (_q_ok) {
        input_ptr = quant_buf;          // still device-resident
        compress_input_size = quant_bytes;
        applied_quant = true;
        CLIO_PATH_TRACE(
            "WRITE  QuantizeDevice (CUDA) blob='%s' %llu -> %llu bytes prec=%d "
            "eb=%g effective_eb=%g escaped=%llu device=1",
            task->blob_name_.str().c_str(),
            (unsigned long long)input_size, (unsigned long long)quant_bytes,
            (int)quant_params.precision, context.error_bound_,
            quant_params.effective_error_bound,
            (unsigned long long)quant_params.escape_count);
        HLOG(kDebug,
             "NeuroPress quantize: {} -> {} bytes (precision={} eb={} "
             "effective_eb={} escaped={})",
             input_size, quant_bytes, quant_params.precision,
             context.error_bound_, quant_params.effective_error_bound,
             quant_params.escape_count);
      } else {
        if (quant_device_alloc.IsNull()) {
          quant_params.refusal =
              ctp::compress::preprocess::QuantizeRefusal::kDeviceError;
        }
        ReportQuantizeRefusal(quant_params.refusal, context.error_bound_,
                              input_size);
        // For the explore row, written by DynamicSchedule after this returns.
        RecordPrimaryQuantizeRefusal(task->blob_name_.str(),
                                     quant_params.refusal);
        if (!quant_device_alloc.IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, quant_device_alloc);
          quant_device_alloc = ctp::ipc::AllocatorId();
        }
      }
      // On failure nothing is recorded and the chunk is compressed losslessly -- the data is still...
    } else if (want_quant) {
      // Host-resident chunk.
      RefuseHostPreprocess("quantization", input_size);
      context.compress_lib_ = 0;
      context.dynamic_compress_ = 0;
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    if (applied_quant) {
      header_size += sizeof(QuantHeaderExtension);
    }

    if (shuffle_elem != 0) {
      if (ctp::IsDevicePointer(input_ptr)) {
        // Shuffle ON the device.
        shuffle_device_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
            /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
            compress_input_size, &shuffle_device_buf);
        bool _s_ok;
        {
#if CTP_ENABLE_CUDA
          // Left on the default stream for the reason at the quantize bracket.
          ctp::CodecKernelTimer _kt(nullptr, &context.actual_preproc_time_ms_);
#endif
          _s_ok = !shuffle_device_alloc.IsNull() &&
              ctp::compress::preprocess::ByteShuffleDevice(
                  input_ptr, shuffle_device_buf, compress_input_size, shuffle_elem);
        }
        if (_s_ok) {
          input_ptr = shuffle_device_buf;  // still on the device
          applied_shuffle = shuffle_elem;
          CLIO_PATH_TRACE(
              "WRITE  ByteShuffleDevice (CUDA) %llu bytes elem=%u device=1",
              (unsigned long long)compress_input_size, (unsigned)shuffle_elem);
        } else if (!shuffle_device_alloc.IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, shuffle_device_alloc);
          shuffle_device_alloc = ctp::ipc::AllocatorId();
          shuffle_device_buf = nullptr;
        }
      } else {
        RefuseHostPreprocess("byte shuffle", compress_input_size);
        context.compress_lib_ = 0;
        context.dynamic_compress_ = 0;
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }
      // On failure input_ptr is untouched and applied_shuffle stays 0, so
      // the header records "not shuffled" and the read side does nothing.
    }

    // input_ptr is still device-resident only when StageInputIfNeeded left it alone, i.e.
    bool output_on_device = ctp::IsDevicePointer(input_ptr);
    CLIO_PATH_TRACE("WRITE  codec input device=%d -> output buffer %s",
                    output_on_device ? 1 : 0,
                    output_on_device ? "DEVICE (no D2H before the bdev write)"
                                     : "*** HOST (compressed bytes copied "
                                       "through host SHM) ***");

    // Escape slots make a quantized chunk larger than the input it replaced.
    if (compress_input_size > input_size) {
      worst_case_size = std::max(
          {worst_case_size,
           compress_input_size + compress_input_size / 20 + 1024,
           compressor->MaxCompressedSize(compress_input_size)});
    }

    std::vector<char> compressed_buffer;
    char *device_output = nullptr;
    if (output_on_device) {
      device_output_alloc_id = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          header_size + worst_case_size, &device_output);
      if (device_output_alloc_id.IsNull()) {
        // GPU OOM is exactly when the caller needs to know: a device-resident chunk is about to be c...
        HLOG(kError,
             "Compress: GPU output allocation failed ({} bytes); the chunk is "
             "device-resident but its compressed output falls back to HOST "
             "memory",
             header_size + worst_case_size);
        output_on_device = false;
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

    // Time ONLY the compress call.
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
    // Device time for the codec launch alone, when CLIO_CODEC_KERNEL_TIMING is on.
    const double compress_kernel_ms = ctp::LastCodecKernelMs();
    compress_phases.compress_ms =
        (compress_kernel_ms >= 0.0) ? compress_kernel_ms : compress_time;

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
      context.actual_compression_ratio_ =
          static_cast<double>(input_size) /
          static_cast<double>(compressed_size);
      // Prefer the CODEC KERNEL time over host wall clock.
      context.actual_compress_time_ms_ =
          (compress_kernel_ms >= 0.0) ? compress_kernel_ms : compress_time;

      // Decompress what we just produced, back, and time the codec call alone.
#if CTP_ENABLE_COMPRESS && CTP_ENABLE_NVCOMP
      if (MeasureExploreDecompTime()) {
        double dt_ms = -1.0;
        if (ctp::NvComp::DecompressMeasureAnyPtr(compress_dst, compressed_size,
                                                 compress_input_size, &dt_ms)) {
          context.actual_decompress_time_ms_ = dt_ms;
        }
      }
      // MEASURED reconstruction quality for the PRIMARY, here for the same reason the dt measureme...
      if (MeasureExploreQuality() && primary_orig_device != nullptr) {
        ctp::compress::preprocess::QualityMetrics qm;
        if (MeasureStoredChunkQuality(
                primary_orig_device, input_size, compress_dst, compressed_size,
                compress_input_size, applied_shuffle,
                applied_quant ? &quant_params : nullptr, &qm)) {
          // Parked for the explore row DynamicSchedule writes after it awaits this call, so the primar...
          RecordPrimaryQuality(task->blob_name_.str(), qm);
          // Its own log, not Context: Context is serialized (core_tasks.h), so carrying it there would...
          LogMeasuredQuality(task->blob_name_.str(), input_size,
                             applied_shuffle, applied_quant, qm);
        }
      }
#endif

      // PSNR is DEFINED only when quantization ran.
      if (applied_quant) {
        const double psnr = AnalyticalPsnr(
            quant_params.data_max - quant_params.data_min,
            quant_params.effective_error_bound);
        if (psnr > 0.0) context.actual_psnr_db_ = psnr;
      } else {
        context.actual_psnr_db_ = -1.0;
      }

      // Record the shuffle that was ACTUALLY applied, not the one requested: if ByteShuffle declin...
      CompressionHeader header(
          context.compress_lib_,
          PackStored(preset_id, applied_shuffle, applied_quant,
                     quant_params),
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
        // Header goes in the room compress_dst was offset past above -- this is the only host touch ...
        ctp::DeviceAwareMemcpy(device_output,
                               reinterpret_cast<const char *>(&header),
                               sizeof(CompressionHeader));
        if (applied_quant) {
          ctp::DeviceAwareMemcpy(device_output + sizeof(CompressionHeader),
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
        // Hand the image to the caller instead of storing it, so exploration can compare candidates ...
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
        park_compress_phases(0.0);  // the put happens in DynamicSchedule
      } else {
        CLIO_PATH_TRACE(
            "WRITE  PutBlob -> tier blob='%s' stored=%llu bytes, handed over "
            "from %s memory",
            task->blob_name_.str().c_str(),
            (unsigned long long)total_stored_size,
            output_on_device ? "DEVICE" : "HOST");
        // Call PutBlob with header + compressed data
        const auto put_t0 = std::chrono::steady_clock::now();
        auto put_task = core_client_->AsyncPutBlob(
            task->tag_id_, task->blob_name_.str(), task->offset_,
            total_stored_size, compressed_shm_ptr, task->score_, context,
            task->flags_, clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put_task);
        compress_phases.io_start_ns =
            static_cast<double>(put_t0.time_since_epoch().count());
        park_compress_phases(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - put_t0)
                                 .count());
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
      // Compression failed or didn't reduce size - store original data.
      if (success) {
        HLOG(kWarning,
             "Compression not beneficial for blob '{}': {} bytes in -> {} "
             "bytes out with lib {} ({}), which is not smaller, so the "
             "ORIGINAL bytes were stored and compress_lib_ was reset to 0. "
             "The blob is in the tier UNCOMPRESSED.",
             task->blob_name_.str(), (size_t)input_size,
             (size_t)total_stored_size, context.compress_lib_,
             ctp::CompressionFactory::NameForWireId(context.compress_lib_));
      } else {
        HLOG(kWarning,
             "Compression FAILED for blob '{}' ({} bytes, lib {} ({})); the "
             "ORIGINAL bytes were stored instead and compress_lib_ was reset "
             "to 0. The blob is in the tier UNCOMPRESSED.",
             task->blob_name_.str(), (size_t)input_size, context.compress_lib_,
             ctp::CompressionFactory::NameForWireId(context.compress_lib_));
      }

      // Mark uncompressed BEFORE the put, not after. The bytes below are the
      // caller's original bytes, so the blob must not be recorded as carrying
      // the codec we merely attempted -- this used to be zeroed only after
      // PutBlob had already copied compress_lib_ onto the BlobInfo, which is
      // half of why compress_lib_ cannot be trusted as a transform signal
      // (issue #818). transform_flags_ is deliberately left unset here.
      context.compress_lib_ = 0;

      if (task->no_store_) {
        // Same deferral as the compressed path, and for the same reason: an exploration winner must ...
        task->stored_data_ = task->blob_data_;
        task->stored_size_ = task->size_;
        task->stored_gpu_alloc_ = ctp::ipc::AllocatorId();
        task->stored_owned_ = false;
        task->context_ = context;
        task->return_code_ = 0;
        park_compress_phases(0.0);
        CLIO_CO_RETURN;
      }

      const auto raw_put_t0 = std::chrono::steady_clock::now();
      auto put_task = core_client_->AsyncPutBlob(
          task->tag_id_, task->blob_name_.str(), task->offset_, task->size_,
          task->blob_data_, task->score_, context, task->flags_,
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put_task);
      compress_phases.io_start_ns =
          static_cast<double>(raw_put_t0.time_since_epoch().count());
      park_compress_phases(std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - raw_put_t0)
                               .count());

      task->context_ = put_task->context_;
      task->return_code_ = put_task->return_code_;

      // Record what the codec REALLY achieved, even though we discarded its output.
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

    const auto read_t0 = std::chrono::steady_clock::now();
    auto ms_since = [](std::chrono::steady_clock::time_point t) {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t).count();
    };
    ChunkPhases read_phases;
    read_phases.io_ms = 0.0;

    // Extract task parameters (same as GetBlobTask).
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

    // Ask core directly (bypassing this class's own GetBlobSize override, which deliberately rep...
    if (core_client_) {
      const auto size_t0 = std::chrono::steady_clock::now();
      auto size_task = core_client_->AsyncGetBlobSize(
          task->tag_id_, task->blob_name_.str(), clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(size_task);
      read_phases.io_ms += ms_since(size_t0);
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
    const auto get_t0 = std::chrono::steady_clock::now();
    auto get_task = core_client_->AsyncGetBlob(
        task->tag_id_, task->blob_name_.str(), task->offset_, expected_size,
        task->flags_, temp_buffer_ptr, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(get_task);
    read_phases.io_ms += ms_since(get_t0);

    if (get_task->return_code_ != 0) {
      CLIO_IPC->FreeBuffer(temp_buffer);
      task->return_code_ = 10 + get_task->return_code_;  // GetBlob failed
      CLIO_CO_RETURN;
    }

    // Is this blob compressed?
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
      // Marked compressed but the header does not check out: the blob is damaged, or was written b...
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
      // High bits carry the byte-shuffle element size (see PackPreset).
      const uint32_t packed_preset = header->compress_preset_;
      int compress_preset = static_cast<int>(UnpackPreset(packed_preset));
      const uint32_t stored_shuffle = UnpackShuffle(packed_preset);
      clio::run::u64 original_size = header->original_size_;

      // A quantized blob carries a 32-byte extension after the core header, holding the four doubl...
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
        UnpackQuantLayout(packed_preset, &stored_quant_params);
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
      const auto factory_t0 = std::chrono::steady_clock::now();
      auto decompressor =
          ctp::CompressionFactory::GetPreset(library_name, preset);
      read_phases.factory_ms = ms_since(factory_t0);
      if (!decompressor) {
        CLIO_IPC->FreeBuffer(temp_buffer);
        HLOG(kWarning, "Failed to create decompressor for library: {}",
             library_name);
        task->return_code_ = 3;  // Decompressor creation failed
        CLIO_CO_RETURN;
      }

      auto decompress_start = std::chrono::high_resolution_clock::now();

      // Get compressed data (after header).
      char* compressed_data = temp_buffer.ptr_ + header_size;
      const size_t recorded_size = header->PayloadSize(expected_size);
      size_t compressed_size =
          (recorded_size > 0) ? recorded_size : (expected_size - header_size);

      // Decompress to output buffer
      auto output_fullptr =
          CLIO_IPC->ToFullPtr<char>(task->blob_data_.template Cast<char>());

      // A quantized blob decompresses to the NARROWED buffer, not to the original bytes: the codec...
      const size_t quant_elems =
          stored_quant ? (original_size / stored_quant_params.elem_bytes) : 0;
      const size_t quant_bytes =
          stored_quant ? ctp::compress::preprocess::QuantizedBytes(
                             quant_elems, stored_quant_params)
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

      ctp::LastCodecKernelMs() = -1.0;  // not a stale compress time
      bool success =
          decompressor->Decompress(codec_dst, decompressed_size,
                                   compressed_data, compressed_size);
      const double decompress_kernel_ms = ctp::LastCodecKernelMs();

      // Proof-of-execution trace, off unless CLIO_NEUROPRESS_DECOMPRESS_TRACE is set.
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

      // Invert the byte-shuffle the write side applied.
      if (success && stored_shuffle != 0) {
        if (ctp::IsDevicePointer(codec_dst)) {
          // Unshuffle ON the device, into a scratch device buffer and back.
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
            // DeviceAwareMemcpy, NOT GpuApi::Memcpy.
            ctp::DeviceAwareMemcpy(codec_dst, scratch, decompressed_size);
          }
          if (!scratch_alloc.IsNull()) {
            CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, scratch_alloc);
          }
        } else {
          // The blob records a shuffle, the inverse is CUDA-only, and the codec wrote into HOST memory...
          char *stage = nullptr;
          ctp::ipc::AllocatorId stage_alloc =
              CLIO_IPC->AllocateAndRegisterGpuBackend(
                  /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
                  decompressed_size, &stage);
          char *unshuf = nullptr;
          ctp::ipc::AllocatorId unshuf_alloc;
          if (!stage_alloc.IsNull()) {
            unshuf_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
                /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
                decompressed_size, &unshuf);
          }
          if (stage_alloc.IsNull() || unshuf_alloc.IsNull()) {
            HLOG(kError,
                 "Decompress: could not allocate the {}-byte device scratch "
                 "needed to unshuffle blob '{}' into host memory",
                 decompressed_size, task->blob_name_.str());
            success = false;
          } else {
            ctp::DeviceAwareMemcpy(stage, codec_dst, decompressed_size);
            if (!ctp::compress::preprocess::ByteUnshuffleDevice(
                    stage, unshuf, decompressed_size, stored_shuffle)) {
              HLOG(kError,
                   "Decompress: staged byte-unshuffle failed (elem={} "
                   "size={}) -- the returned buffer would be shuffled "
                   "garbage, failing instead",
                   stored_shuffle, decompressed_size);
              success = false;
            } else {
              ctp::DeviceAwareMemcpy(codec_dst, unshuf, decompressed_size);
            }
          }
          if (!unshuf_alloc.IsNull()) {
            CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, unshuf_alloc);
          }
          if (!stage_alloc.IsNull()) {
            CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, stage_alloc);
          }
        }
      }

      // Dequantize LAST, inverting the write side's quantize-then-shuffle (upstream inverts in the...
      if (success && stored_quant) {
        // DequantizeDevice writes its floats from a CUDA kernel, so the destination has to be DEVICE...
        char *deq_dst = output_fullptr.ptr_;
        char *deq_scratch = nullptr;
        ctp::ipc::AllocatorId deq_scratch_alloc;
        if (!ctp::IsDevicePointer(output_fullptr.ptr_)) {
          deq_scratch_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
              /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
              original_size, &deq_scratch);
          if (deq_scratch_alloc.IsNull()) {
            HLOG(kError,
                 "Decompress: could not allocate the dequantization output "
                 "scratch ({} bytes) for blob '{}'",
                 original_size, task->blob_name_.str());
            success = false;
          } else {
            deq_dst = deq_scratch;
          }
        }
        if (success && !ctp::compress::preprocess::DequantizeDevice(
                codec_dst, quant_elems, stored_quant_params, deq_dst)) {
          HLOG(kError,
               "Decompress: dequantization failed for blob '{}' (elems={} "
               "precision={}) -- the returned buffer would be quantized "
               "integers, failing instead",
               task->blob_name_.str(), quant_elems,
               stored_quant_params.precision);
          success = false;
        } else if (success) {
          if (deq_scratch != nullptr) {
            ctp::DeviceAwareMemcpy(output_fullptr.ptr_, deq_scratch,
                                   original_size);
          }
          decompressed_size = original_size;
        }
        if (!deq_scratch_alloc.IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, deq_scratch_alloc);
        }
      }
      if (!quant_scratch_alloc.IsNull()) {
        CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, quant_scratch_alloc);
      }

      // Stop the clock AFTER the unshuffle, not before it.
      auto decompress_end = std::chrono::high_resolution_clock::now();
      double decompress_time = std::chrono::duration<double, std::milli>(
                                   decompress_end - decompress_start)
                                   .count();

      if (success) {
        task->output_size_ = decompressed_size;
        task->decompress_time_ms_ = decompress_time;
        if (PhaseLogEnabled()) {
          // SYMMETRY WITH THE WRITE PATH (compress_ms, above): the phase log
          // records the CODEC KERNEL time when the codec measured one, so
          // that compress_ms and decompress_ms are the same KIND of quantity
          // and a figure may put them side by side. The host wall over this
          // whole region -- per-chunk manager construction, the H2D of the
          // compressed bytes, the D2H of the output, the allocations and the
          // inverse preprocessing -- is not lost: LogChunkPhases derives
          // other_ms as wall minus the measured components, so it lands
          // there, exactly as the write path's non-kernel work does.
          //
          // Measured on VPIC at 4 MiB: the wall is 3.10 ms per chunk and the
          // kernel 0.39 ms, so 87% of what this column used to report was
          // not decompression.
          read_phases.decompress_ms = (decompress_kernel_ms >= 0.0)
                                          ? decompress_kernel_ms
                                          : decompress_time;
          LogChunkPhases(task->blob_name_.str(), "read", original_size,
                         compress_lib, read_phases, ms_since(read_t0),
                         expected_size);
        }

        // Show the RECONSTRUCTED VALUES, not just that a call returned 0.
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

        // Deferred decomp-head learning: this is the ONLY point a real decompression time exists.
        // Label with the codec kernel time, as compress and exploration do;
        // CLIO_NEUROPRESS_DT_LABEL=wall uses the wall clock.
        static const bool kDtLabelWall = [] {
          const char *e = std::getenv("CLIO_NEUROPRESS_DT_LABEL");
          return e != nullptr && std::strcmp(e, "wall") == 0;
        }();
        LearnDecompTime(task->blob_name_.str(),
                        (!kDtLabelWall && decompress_kernel_ms >= 0.0)
                            ? decompress_kernel_ms
                            : decompress_time);

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
      // Device-aware, not std::memcpy: the caller's destination is a CUDA-IPC device buffer whenev...
      const auto copy_t0 = std::chrono::steady_clock::now();
      ctp::DeviceAwareMemcpy(output_fullptr.ptr_, temp_buffer.ptr_,
                             expected_size);
      CLIO_IPC->FreeBuffer(temp_buffer);
      if (PhaseLogEnabled()) {
        read_phases.factory_ms = 0.0;
        // A blob stored raw was never compressed, so no codec runs here and
        // there is no decompression to time. Charging this copy to
        // decompress_ms made a chunk no codec had touched carry up to 1.7 ms
        // of "Decompress Time" in figure 5. The copy is still accounted:
        // other_ms is the wall minus what was measured.
        read_phases.decompress_ms = 0.0;
        LogChunkPhases(task->blob_name_.str(), "read", expected_size,
                       /*lib=*/0, read_phases, ms_since(read_t0),
                       expected_size);
      }

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
  // compress_preset_ is PACKED (preset | shuffle_elem << 8) whenever the selection came from D...
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
  // The requested bound, as at the primary compress site above.
  if (ctx.error_bound_ > 0.0) {
    compressor->SetErrorBound(ctx.error_bound_);
  }
  auto t0 = std::chrono::high_resolution_clock::now();

  // Honor the shuffle the selector asked for.
  const char *compress_src = src;
  size_t compress_size = size;

  // Quantize FIRST, then shuffle -- the same order Runtime::Compress and upstream use (gpucomp...
  bool applied_quant = false;
  uint32_t applied_shuffle = 0;
  ctp::compress::preprocess::DeviceQuantizeParams quant_params{};

  // Quantize and byte shuffle are CUDA-ONLY -- the CPU implementations were removed deliberate...
  const size_t quant_elem = QuantElemBytes(ctx.data_type_);
  if (requested_quant && ctx.error_bound_ > 0.0 && quant_elem == 0) {
    WarnQuantizeUntyped(ctx.data_type_);
  }
  const bool want_quant = requested_quant && ctx.error_bound_ > 0.0 &&
                          quant_elem != 0 && size >= quant_elem &&
                          (size % quant_elem) == 0;

  // Function scope: the transformed bytes live in these until the codec has read them, which i...
  ctp::ipc::AllocatorId in_alloc, quant_alloc, shuffle_alloc;
  in_alloc.SetNull();
  quant_alloc.SetNull();
  shuffle_alloc.SetNull();
  struct DevScratch {
    ctp::ipc::AllocatorId *ids[3];
    ~DevScratch() {
      for (auto *id : ids) {
        if (id && !id->IsNull()) {
          CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, *id);
          id->SetNull();
        }
      }
    }
  } dev_scratch{{&in_alloc, &quant_alloc, &shuffle_alloc}};

  if (want_quant || requested_shuffle != 0) {
    const char *dev_cur = src;
    size_t dev_bytes = size;

    if (!ctp::IsDevicePointer(src)) {
      char *dev_in = nullptr;
      in_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem, size,
          &dev_in);
      if (in_alloc.IsNull() || dev_in == nullptr) {
        RefuseHostPreprocess(want_quant ? "quantization" : "byte shuffle",
                             size);
        return false;
      }
      ctp::DeviceAwareMemcpy(dev_in, src, size);
      dev_cur = dev_in;
    }

    if (want_quant) {
      char *quant_buf = nullptr;
      quant_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          ctp::compress::preprocess::QuantizeCapacity(size / quant_elem,
                                                      quant_elem),
          &quant_buf);
      size_t quant_bytes = 0;
      if (!quant_alloc.IsNull() &&
          ctp::compress::preprocess::QuantizeDevice(
              dev_cur, size / quant_elem, quant_elem, ctx.error_bound_,
              quant_buf, &quant_bytes, &quant_params)) {
        dev_cur = quant_buf;
        dev_bytes = quant_bytes;
        applied_quant = true;
        CLIO_PATH_TRACE(
            "WRITE  QuantizeDevice (CUDA, interpose) %llu -> %llu bytes "
            "prec=%d eb=%g effective_eb=%g device=1",
            (unsigned long long)size, (unsigned long long)quant_bytes,
            (int)quant_params.precision, ctx.error_bound_,
            quant_params.effective_error_bound);
      } else {
        // The chunk still compresses, losslessly, and the header records that.
        if (quant_alloc.IsNull()) {
          quant_params.refusal =
              ctp::compress::preprocess::QuantizeRefusal::kDeviceError;
        }
        ReportQuantizeRefusal(quant_params.refusal, ctx.error_bound_, size);
      }
    }

    if (requested_shuffle != 0) {
      char *shuffle_buf = nullptr;
      shuffle_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          dev_bytes, &shuffle_buf);
      if (!shuffle_alloc.IsNull() &&
          ctp::compress::preprocess::ByteShuffleDevice(
              dev_cur, shuffle_buf, dev_bytes, requested_shuffle)) {
        dev_cur = shuffle_buf;
        applied_shuffle = requested_shuffle;
        CLIO_PATH_TRACE(
            "WRITE  ByteShuffleDevice (CUDA, interpose) %llu bytes elem=%u "
            "device=1",
            (unsigned long long)dev_bytes, (unsigned)requested_shuffle);
      }
      // On failure applied_shuffle stays 0, so the header records "not
      // shuffled" and the read side does nothing.
    }

    if (applied_quant || applied_shuffle != 0) {
      compress_src = dev_cur;   // STILL ON THE DEVICE
      compress_size = dev_bytes;
    }
  }

  // The only transfer that is ever needed, and only when the codec is a CPU one: a GPU-native ...
  std::vector<char> codec_staging;
  compress_src = ctp::CompressionFactory::StageInputIfNeeded(
      const_cast<char *>(compress_src), compress_size, ctx.compress_lib_,
      codec_staging);

  const size_t codec_in = std::max<size_t>(size, compress_size);
  std::vector<char> compressed(codec_in + (codec_in / 20) + 1024);
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
  // Record the shuffle that was ACTUALLY applied, not the one requested -- a declined shuffle ...
  CompressionHeader header(
      ctx.compress_lib_,
      PackStored(requested_preset, applied_shuffle, applied_quant,
                 quant_params),
      size, compressed_size);
  QuantHeaderExtension quant_ext{};
  if (applied_quant) {
    quant_ext.error_bound = quant_params.effective_error_bound;
    quant_ext.scale = quant_params.scale;
    quant_ext.data_min = quant_params.data_min;
    quant_ext.data_max = quant_params.data_max;
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
      applied_quant ? AnalyticalPsnr(quant_params.data_max -
                                         quant_params.data_min,
                                     quant_params.effective_error_bound)
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
  // NOT a sniff: the only caller (Runtime::GetBlob) has already confirmed kBlobTransformCompre...
  const auto *header = reinterpret_cast<const CompressionHeader *>(stored);
  if (!header->IsValid() || header->original_size_ > dst_cap) {
    return 6;
  }
  std::string library_name = ctp::CompressionFactory::NameForWireId(
      static_cast<int>(header->compress_lib_));

  // Proof-of-execution trace, off unless CLIO_NEUROPRESS_DECOMPRESS_TRACE is set.
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
  // Same packed layout Runtime::Compress writes -- unpack both halves.
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
  const size_t recorded_size = header->PayloadSize(stored_size);
  const size_t payload_size =
      (recorded_size > 0) ? recorded_size : (stored_size - hdr_bytes);

  // A quantized blob decompresses to the NARROWED integers, so the codec writes into a staging...
  ctp::compress::preprocess::DeviceQuantizeParams layout;
  UnpackQuantLayout(header->compress_preset_, &layout);
  const size_t quant_elems =
      stored_quant ? (header->original_size_ / layout.elem_bytes) : 0;
  const size_t quant_bytes =
      stored_quant
          ? ctp::compress::preprocess::QuantizedBytes(quant_elems, layout)
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

  // Invert the byte-shuffle and the quantization before the caller sees the buffer.
  if (stored_shuffle != 0) {
    char *up = nullptr, *uo = nullptr;
    ctp::ipc::AllocatorId up_alloc, uo_alloc;
    up_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
        /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        decompressed, &up);
    if (!up_alloc.IsNull()) {
      uo_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          decompressed, &uo);
    }
    bool ok = !up_alloc.IsNull() && !uo_alloc.IsNull();
    if (ok) {
      ctp::DeviceAwareMemcpy(up, codec_dst, decompressed);
      ok = ctp::compress::preprocess::ByteUnshuffleDevice(up, uo, decompressed,
                                                          stored_shuffle);
      if (ok) ctp::DeviceAwareMemcpy(codec_dst, uo, decompressed);
    }
    if (!uo_alloc.IsNull()) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, uo_alloc);
    if (!up_alloc.IsNull()) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, up_alloc);
    if (!ok) {
      HLOG(kError,
           "DecompressStored: staged byte-unshuffle failed (elem={} size={})",
           stored_shuffle, decompressed);
      return 5;
    }
  }

  // Dequantize LAST, inverting quantize-then-shuffle.
  if (stored_quant) {
    char *qin = nullptr, *qout = nullptr;
    ctp::ipc::AllocatorId qin_alloc, qout_alloc;
    qin_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
        /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        quant_bytes, &qin);
    if (!qin_alloc.IsNull()) {
      qout_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          header->original_size_, &qout);
    }
    bool ok = !qin_alloc.IsNull() && !qout_alloc.IsNull();
    if (ok) {
      ctp::DeviceAwareMemcpy(qin, codec_dst, quant_bytes);
      // Rebuild the writer's parameters from the header extension, the same
      // four fields Runtime::Decompress reads.
      ctp::compress::preprocess::DeviceQuantizeParams qp = layout;
      qp.effective_error_bound = stored_ext.error_bound;
      qp.scale = stored_ext.scale;
      qp.data_min = stored_ext.data_min;
      qp.data_max = stored_ext.data_max;
      ok = ctp::compress::preprocess::DequantizeDevice(qin, quant_elems, qp,
                                                       qout);
      if (ok) {
        ctp::DeviceAwareMemcpy(dst, qout, header->original_size_);
        decompressed = header->original_size_;
      }
    }
    if (!qout_alloc.IsNull()) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, qout_alloc);
    if (!qin_alloc.IsNull()) CLIO_IPC->FreeGpuBackend(/*gpu_id=*/0, qin_alloc);
    if (!ok) {
      HLOG(kError,
           "DecompressStored: staged dequantization failed (elems={})",
           quant_elems);
      return 5;
    }
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
