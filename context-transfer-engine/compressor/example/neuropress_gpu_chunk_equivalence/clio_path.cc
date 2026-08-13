/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file clio_path.cc
 * @brief The Clio-NeuroPress side of the callback trace.
 *
 * Each stage calls the SAME function Clio's compressor runtime calls, in the
 * order the runtime calls it:
 *
 *   statistics   ctp::ComputeDeviceStatsResident      compressor_runtime.cc:647
 *   inference    NeuroPressCandidateStatsDevice       compressor_runtime.cc:704
 *   diagnostics  ctp::ReadDeviceFeatureStats          compressor_runtime.cc:714
 *   quantization ctp::...::QuantizeDevice             compressor_runtime.cc:2378
 *   shuffle      ctp::...::ByteShuffleDevice          compressor_runtime.cc:2444
 *   compression  CompressionFactory preset->Compress  compressor_runtime.cc:2507
 *
 * Note the ORDER of the diagnostics read: on the device path Clio pulls the
 * three features to the host AFTER inference, not before, because doing it
 * before would synchronize the stream and reinstate the stall the device path
 * exists to remove. Native's harness read is a probe with no production
 * counterpart at all; Clio's is a real production read. The trace records both
 * as the `diagnostics` stage and the comparison treats the FEATURE VALUES as
 * the thing that must match -- not the fact of the copy, which legitimately
 * differs.
 */

#include "clio_path.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/compress/model/neuropress_nn_predictor.h>
#include <clio_ctp/compress/model/ranking.h>
#include <clio_ctp/compress/preprocess/byte_shuffle.h>
#include <clio_ctp/compress/preprocess/data_stats.h>
#include <clio_ctp/compress/preprocess/data_stats_gpu.h>
#include <clio_ctp/compress/preprocess/quantization.h>

#include <clio_cte/compressor/models/neuropress_bridge.h>

#include "device_probe.h"

namespace npeq {

namespace {

std::unique_ptr<ctp::compress::model::NeuroPressNNPredictor> g_predictor;

/** Bit layout of CompressionStats::compress_preset_ (PackPreset/PackQuant). */
constexpr uint32_t kShuffleShift = 8;
constexpr uint32_t kQuantEnabledBit = 1u << 24;

/** Map a CompressionFactory wire id back to the model's base id. */
int BaseIdForWireId(int wire_id) {
  const std::string name = ctp::CompressionFactory::NameForWireId(wire_id);
  for (const auto &entry : ctp::compress::model::KnownCompressors()) {
    if (entry.name == name) return entry.base_id;
  }
  return -1;
}

/**
 * Recover the upstream action id from one ranked CompressionStats.
 *
 * Not a guess: the bridge encodes shuffle and quantize into compress_preset_'s
 * high bits on the way out (neuropress_bridge.cc:365-368) and the algorithm
 * into compress_lib_, and NeuroPressActionId is the same function the bridge
 * used to order the candidate list in the first place.
 */
int ActionIdFor(const clio::cte::compressor::CompressionStats &stat) {
  const int base_id = BaseIdForWireId(stat.compress_lib_);
  if (base_id < 0) return -1;
  const uint32_t bits = static_cast<uint32_t>(stat.compress_preset_);
  const bool shuffle = ((bits >> kShuffleShift) & 0xFFu) != 0;
  const bool quant = (bits & kQuantEnabledBit) != 0;
  return ctp::compress::model::NeuroPressActionId(base_id, quant, shuffle);
}

const char *AlgoName(int algo) {
  static const char *kNames[] = {"lz4",  "snappy", "deflate",  "gdeflate",
                                 "zstd", "ans",    "cascaded", "bitcomp"};
  if (algo < 0 || algo > 7) return "unknown";
  return kNames[algo];
}

}  // namespace

bool ClioInitPredictor(const std::string &weights_dir, std::string *error) {
  if (g_predictor && g_predictor->IsReady()) return true;
  g_predictor =
      std::make_unique<ctp::compress::model::NeuroPressNNPredictor>();
  if (!g_predictor->Load(weights_dir)) {
    if (error) *error = "NeuroPressNNPredictor::Load failed for " + weights_dir;
    g_predictor.reset();
    return false;
  }
  if (!g_predictor->IsReady()) {
    if (error) *error = "NeuroPressNNPredictor loaded but is not ready";
    g_predictor.reset();
    return false;
  }
  if (!g_predictor->GpuInferenceActive()) {
    // Not fatal, but it means Clio would be inferring on a host port while
    // native infers on the GPU -- which is itself a divergence worth seeing
    // rather than silently tolerating.
    if (error) *error = "NeuroPressNNPredictor is ready but GPU inference is "
                        "NOT active";
    return false;
  }
  return true;
}

void ClioShutdownPredictor() { g_predictor.reset(); }

ClioChunkResult ClioRunChunk(const void *device_input, size_t bytes,
                             double error_bound, int chunk_id) {
  ClioChunkResult out;
  auto &recorder = Recorder::Instance();

  if (!g_predictor || !g_predictor->IsReady()) {
    out.error = "ClioInitPredictor was not called";
    return out;
  }
  if (ClassifyPointer(device_input) != PointerKind::kDevice) {
    out.error = "input chunk is not device-resident";
    return out;
  }

  const size_t num_elements = bytes / sizeof(float);
  void *stream = ctp::DeviceStatsStream();
  const void *device_stats = nullptr;

  // ---------------- statistics ----------------
  {
    StageScope stage(Stage::kStatistics, device_input, bytes);
    stage->note = "ctp::ComputeDeviceStatsResident (compressor_runtime.cc:647)";
    stage->stream = stream;
    device_stats = ctp::ComputeDeviceStatsResident(
        device_input, num_elements, ctp::DataType::FLOAT32, stream);
    if (device_stats == nullptr) {
      stage.Fail("ComputeDeviceStatsResident returned null");
      out.error = "clio statistics failed";
      return out;
    }
    stage.SetOutput(device_stats, sizeof(ctp::DeviceFeatureStats));
  }

  // ---------------- inference / ranking / selection ----------------
  std::vector<clio::cte::compressor::CompressionStats> ranked;
  {
    StageScope stage(Stage::kInference, device_stats,
                     sizeof(ctp::DeviceFeatureStats));
    stage->note =
        "NeuroPressCandidateStatsDevice -> PredictBatchDeviceStats "
        "(compressor_runtime.cc:704); the kernel reads the statistics out of "
        "device memory";
    stage->stream = stream;
    bool infer_failed = false;
    ranked = clio::cte::compressor::NeuroPressCandidateStatsDevice(
        *g_predictor, static_cast<clio::run::u64>(bytes), device_stats, stream,
        /*data_type_float=*/true, error_bound, /*min_psnr=*/0.0,
        &infer_failed);
    if (ranked.empty()) {
      stage.Fail(infer_failed ? "device inference failed"
                              : "no candidates were produced");
      out.error = "clio inference produced no ranking";
      return out;
    }
    stage->inference.valid = true;
    stage->inference.num_candidates = static_cast<int>(ranked.size());
    for (const auto &stat : ranked) {
      const int action = ActionIdFor(stat);
      if (action < 0 || action >= 32) continue;
      stage->inference.present[action] = true;
      stage->inference.ratio[action] =
          static_cast<float>(stat.compression_ratio_);
      stage->inference.comp_ms[action] =
          static_cast<float>(stat.compress_time_ms_);
      stage->inference.decomp_ms[action] =
          static_cast<float>(stat.decompress_time_ms_);
      stage->inference.psnr_db[action] = static_cast<float>(stat.psnr_db_);
    }
    stage.SetOutput(nullptr, 0);
  }

  {
    StageScope stage(Stage::kRanking, nullptr, 0);
    stage->note =
        "the order NeuroPressCandidateStatsDevice returned, best first; slot "
        "order IS action order by construction (neuropress_bridge.cc:224-265)";
    stage->ranking.valid = true;
    for (const auto &stat : ranked) {
      const int action = ActionIdFor(stat);
      if (action >= 0) stage->ranking.order.push_back(action);
    }
    // Clio's ranked list carries the SCORE, not upstream's raw cost, and the
    // two are related by score = -cost. Recording the cost is therefore exact,
    // not a rescaling -- but the ranked stats do not expose the score at all
    // (CompressionStats has no field for it), so per-action costs are left
    // absent rather than invented. The ORDER is the comparable artifact and it
    // is recorded in full.
    stage.SetOutput(nullptr, 0);
  }

  const auto &winner = ranked.front();
  out.action = ActionIdFor(winner);
  out.algo = out.action >= 0 ? (out.action % 8) : -1;
  const uint32_t winner_bits = static_cast<uint32_t>(winner.compress_preset_);
  const uint32_t shuffle_elem = (winner_bits >> kShuffleShift) & 0xFFu;
  const bool quantize_requested = (winner_bits & kQuantEnabledBit) != 0;
  out.shuffle_selected = shuffle_elem != 0;
  out.quantize_selected = quantize_requested && error_bound > 0.0;

  {
    StageScope stage(Stage::kSelection, nullptr, 0);
    stage->note = "ranked.front() -- the candidate the cost model put first";
    stage->selection.valid = true;
    stage->selection.action = out.action;
    stage->selection.algo = out.algo;
    stage->selection.algo_name = AlgoName(out.algo);
    stage->selection.quantize = quantize_requested;
    stage->selection.shuffle = out.shuffle_selected;
    stage.SetOutput(nullptr, 0);
  }

  // ---------------- diagnostics (features) ----------------
  //
  // Production position: AFTER the inference, on a stream that is now idle.
  {
    StageScope stage(Stage::kDiagnostics, device_stats,
                     sizeof(ctp::DeviceFeatureStats));
    stage->note =
        "ctp::ReadDeviceFeatureStats (compressor_runtime.cc:714) -- a REAL "
        "production 24-byte D->H, taken after inference so it costs no extra "
        "stall. Clio has no counterpart of gpucompress_chunk_diag_t";
    const bool ok = ctp::ReadDeviceFeatureStats(
        device_stats, &out.entropy, &out.mad, &out.second_derivative, stream);
    stage->stats.valid = ok;
    stage->stats.entropy = out.entropy;
    stage->stats.mad = out.mad;
    stage->stats.second_derivative = out.second_derivative;
    stage->diagnostics.valid = true;
    stage->diagnostics.native_record_present = false;
    stage->diagnostics.nn_action = out.action;
    stage->diagnostics.nn_original_action = out.action;
    stage->diagnostics.feat_entropy = static_cast<float>(out.entropy);
    stage->diagnostics.feat_mad = static_cast<float>(out.mad);
    stage->diagnostics.feat_deriv =
        static_cast<float>(out.second_derivative);
    stage->diagnostics.predicted_ratio =
        static_cast<float>(winner.compression_ratio_);
    stage->diagnostics.predicted_comp_time =
        static_cast<float>(winner.compress_time_ms_);
    stage->diagnostics.predicted_decomp_time =
        static_cast<float>(winner.decompress_time_ms_);
    stage->diagnostics.predicted_psnr =
        static_cast<float>(winner.psnr_db_);
    if (!ok) stage.Fail("ReadDeviceFeatureStats failed");
    stage.SetOutput(nullptr, 0);
  }

  // ---------------- quantization ----------------
  const void *pipeline_buffer = device_input;
  size_t pipeline_bytes = bytes;

  // The runtime's own gate, copied: want_quant requires the bit AND a positive
  // bound AND a whole number of float32 elements (compressor_runtime.cc:2368).
  const bool want_quant = quantize_requested && error_bound > 0.0 &&
                          bytes >= sizeof(float) &&
                          (bytes % sizeof(float)) == 0;
  if (want_quant) {
    StageScope stage(Stage::kQuantization, pipeline_buffer, pipeline_bytes);
    stage->note =
        "ctp::compress::preprocess::QuantizeDevice "
        "(compressor_runtime.cc:2378)";
    void *quant_buf = nullptr;
    if (cudaMalloc(&quant_buf, bytes) != cudaSuccess) {
      stage.Fail("cudaMalloc for the quantized buffer failed");
      stage.SetOutput(nullptr, 0);
    } else {
      size_t quant_bytes = 0;
      ctp::compress::preprocess::DeviceQuantizeParams params;
      const bool ok = ctp::compress::preprocess::QuantizeDevice(
          pipeline_buffer, num_elements, error_bound, quant_buf, &quant_bytes,
          &params);
      if (!ok || quant_bytes == 0) {
        stage.Fail("QuantizeDevice declined");
        cudaFree(quant_buf);
        stage.SetOutput(nullptr, 0);
      } else {
        out.quantized_device = quant_buf;
        out.quantized_bytes = quant_bytes;
        pipeline_buffer = quant_buf;
        pipeline_bytes = quant_bytes;
        stage->quant.valid = true;
        stage->quant.precision = params.precision;
        stage->quant.error_bound = params.error_bound;
        stage->quant.effective_error_bound = params.effective_error_bound;
        stage->quant.scale = params.scale;
        stage->quant.data_min = params.data_min;
        stage->quant.data_max = params.data_max;
        stage->quant.bound_achievable = params.bound_achievable;

        // Same round trip as the native side, through Clio's own dequantizer,
        // against the REQUESTED bound.
        void *restored = nullptr;
        if (cudaMalloc(&restored, num_elements * sizeof(float)) ==
            cudaSuccess) {
          recorder.PushHarness();
          if (ctp::compress::preprocess::DequantizeDevice(
                  quant_buf, num_elements, params, restored)) {
            out.dequantized_device = restored;
            out.dequantized_bytes = num_elements * sizeof(float);
            out.bound_checked = CountBoundViolations(
                static_cast<const float *>(device_input),
                static_cast<const float *>(restored), num_elements,
                error_bound, &out.bound_violations);
          } else {
            cudaFree(restored);
          }
          recorder.PopHarness();
        }
        stage.SetOutput(quant_buf, quant_bytes);
      }
    }
  } else {
    recorder.RecordSkipped(
        Stage::kQuantization,
        quantize_requested
            ? "candidate selects quantize but the runtime's want_quant gate is "
              "not satisfied (compressor_runtime.cc:2368)"
            : "candidate does not select quantization");
  }

  // ---------------- shuffle ----------------
  if (shuffle_elem != 0) {
    StageScope stage(Stage::kShuffle, pipeline_buffer, pipeline_bytes);
    stage->note =
        "ctp::compress::preprocess::ByteShuffleDevice on the "
        "post-quantization buffer (compressor_runtime.cc:2444)";
    void *shuffle_buf = nullptr;
    if (cudaMalloc(&shuffle_buf, pipeline_bytes) != cudaSuccess) {
      stage.Fail("cudaMalloc for the shuffled buffer failed");
      stage.SetOutput(nullptr, 0);
    } else if (!ctp::compress::preprocess::ByteShuffleDevice(
                   pipeline_buffer, shuffle_buf, pipeline_bytes,
                   shuffle_elem)) {
      stage.Fail("ByteShuffleDevice declined");
      cudaFree(shuffle_buf);
      stage.SetOutput(nullptr, 0);
    } else {
      out.shuffled_device = shuffle_buf;
      out.shuffled_bytes = pipeline_bytes;
      pipeline_buffer = shuffle_buf;
      stage.SetOutput(shuffle_buf, pipeline_bytes);
    }
  } else {
    recorder.RecordSkipped(Stage::kShuffle,
                           "candidate does not select byte shuffle");
  }

  // ---------------- compression ----------------
  {
    const std::string library_name =
        ctp::CompressionFactory::NameForWireId(winner.compress_lib_);
    const int preset_id = static_cast<int>(winner_bits & 0xFFu);
    auto compressor = ctp::CompressionFactory::GetPreset(
        library_name, static_cast<ctp::CompressionPreset>(preset_id));

    StageScope stage(Stage::kCompression, pipeline_buffer, pipeline_bytes);
    stage->note =
        "CompressionFactory preset->Compress, the codec call the runtime "
        "brackets (compressor_runtime.cc:2507); output stays on the device";
    stage->compression.algo_name = library_name;
    if (compressor == nullptr) {
      stage.Fail("CompressionFactory could not build " + library_name);
      stage.SetOutput(nullptr, 0);
      out.error = "clio codec unavailable: " + library_name;
      return out;
    }
    // Same sizing the runtime uses (compressor_runtime.cc:2283).
    const size_t worst_case = pipeline_bytes + (pipeline_bytes / 20) + 1024;
    void *device_output = nullptr;
    if (cudaMalloc(&device_output, worst_case) != cudaSuccess) {
      stage.Fail("cudaMalloc for the compressed output failed");
      stage.SetOutput(nullptr, 0);
      out.error = "clio output allocation failed";
      return out;
    }
    size_t compressed_size = worst_case;
    const bool ok = compressor->Compress(device_output, compressed_size,
                                         const_cast<void *>(pipeline_buffer),
                                         pipeline_bytes);
    if (!ok) {
      stage.Fail("Compress returned false");
      cudaFree(device_output);
      stage.SetOutput(nullptr, 0);
      out.error = "clio compression failed";
      return out;
    }
    out.payload_device = device_output;
    out.payload_bytes = compressed_size;
    stage->compression.valid = true;
    stage->compression.compressed_bytes = compressed_size;
    stage.SetOutput(device_output, compressed_size);
  }

  out.ok = true;
  (void)chunk_id;
  return out;
}

void ClioReleaseChunk(ClioChunkResult *result) {
  if (result == nullptr) return;
  if (result->quantized_device != nullptr) {
    cudaFree(const_cast<void *>(result->quantized_device));
    result->quantized_device = nullptr;
  }
  if (result->dequantized_device != nullptr) {
    cudaFree(const_cast<void *>(result->dequantized_device));
    result->dequantized_device = nullptr;
  }
  if (result->shuffled_device != nullptr) {
    cudaFree(const_cast<void *>(result->shuffled_device));
    result->shuffled_device = nullptr;
  }
  if (result->payload_device != nullptr) {
    cudaFree(const_cast<void *>(result->payload_device));
    result->payload_device = nullptr;
  }
}

}  // namespace npeq
