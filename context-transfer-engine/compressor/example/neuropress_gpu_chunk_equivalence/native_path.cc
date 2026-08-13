/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file native_path.cc
 * @brief The native NeuroPress side of the callback trace.
 *
 * Every stage below calls upstream's OWN entry point. Nothing here
 * reimplements a NeuroPress algorithm; where a number is reported it was
 * produced by the library, not by this file.
 *
 * Two things are worth understanding before reading:
 *
 * 1. WHY THE STAGES ARE CALLED SEPARATELY. Upstream's AUTO path is two public
 *    phases -- `gpucompress_infer_gpu` then
 *    `gpucompress_compress_with_action_gpu` (gpucompress_compress.cpp:154-233)
 *    -- and the second phase runs quantize, shuffle and the codec internally
 *    without handing back the intermediate device buffers. There is no way to
 *    observe the shuffled bytes through the public API. So the harness runs the
 *    production compress call AND, separately, calls upstream's own
 *    `quantize_simple` / `byte_shuffle_simple` with the arguments the compress
 *    path uses, purely to capture the intermediate buffers for comparison.
 *    These are labelled STAGE PROBE in the trace. They are the same
 *    deterministic functions on the same input, so their output is what ran
 *    inside the production call -- but the label is there because the
 *    distinction is real and hiding it would be dishonest. What the production
 *    call itself did is separately visible: CUPTI records the kernels it
 *    launched, inside the `compression` entry.
 *
 * 2. WHY DIAGNOSTICS ARE READ, NOT RECONSTRUCTED. `gpucompress_chunk_diag_t` is
 *    upstream's own per-chunk record, written by `recordChunkDiagnostic()` at
 *    the end of the compress call. The harness reads it through the public
 *    `gpucompress_get_chunk_diag`. It is not approximated.
 */

#include "native_path.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <gpucompress.h>

#include "api/internal.hpp"
#include "nn/nn_weights.h"
#include "preprocessing/byte_shuffle.cuh"
#include "preprocessing/quantization.cuh"
#include "stats/auto_stats_gpu.h"

#include "device_probe.h"

namespace npeq {

namespace {

bool g_initialized = false;

const char *AlgoName(int algo) {
  static const char *kNames[] = {"lz4",  "snappy", "deflate",  "gdeflate",
                                 "zstd", "ans",    "cascaded", "bitcomp"};
  if (algo < 0 || algo > 7) return "unknown";
  return kNames[algo];
}

/**
 * Read the three selection features out of the device AutoStatsGPU.
 *
 * Upstream lays entropy / mad_normalized / deriv_normalized out contiguously
 * precisely so they can be fetched in ONE 24-byte D->H copy
 * (stats_kernel.cu:441-446); this does the same rather than three copies. It
 * is a harness read -- upstream's AUTO path never brings these to the host,
 * and deliberately zeroes them in the reported stats (gpucompress_compress.cpp
 * :323-325) -- so it is bracketed as such by the caller.
 */
bool ReadStatsFeatures(const AutoStatsGPU *d_stats, double *entropy,
                       double *mad, double *deriv) {
  if (d_stats == nullptr) return false;
  double buf[3] = {0.0, 0.0, 0.0};
  const cudaError_t rc =
      cudaMemcpy(buf, &d_stats->entropy, sizeof(buf), cudaMemcpyDeviceToHost);
  if (rc != cudaSuccess) return false;
  *entropy = buf[0];
  *mad = buf[1];
  *deriv = buf[2];
  return true;
}

}  // namespace

bool NativeInit(const std::string &weights_path, std::string *error) {
  if (g_initialized) return true;
  const gpucompress_error_t rc = gpucompress_init(weights_path.c_str());
  if (rc != GPUCOMPRESS_SUCCESS) {
    if (error) *error = std::string("gpucompress_init: ") +
                        gpucompress_error_string(rc);
    return false;
  }
  if (!gpucompress_nn_is_loaded()) {
    if (error) *error = "gpucompress_init succeeded but no .nnwt is loaded";
    return false;
  }
  // Match the Clio side's configuration exactly. Clio's example leaves online
  // learning off (the library default on both sides), and exploration would
  // make the recorded action a function of measured timings rather than of the
  // chunk -- which is not a property two implementations can be expected to
  // reproduce, and not what this harness is measuring.
  gpucompress_disable_online_learning();
  gpucompress_set_exploration(0);
  gpucompress_set_selection_mode(GPUCOMPRESS_SELECT_NN);
  g_initialized = true;
  return true;
}

void NativeShutdown() {
  if (!g_initialized) return;
  gpucompress_cleanup();
  g_initialized = false;
}

NativeChunkResult NativeRunChunk(const void *device_input, size_t bytes,
                                 double error_bound, int chunk_id) {
  NativeChunkResult out;
  auto &recorder = Recorder::Instance();

  if (!g_initialized) {
    out.error = "NativeInit was not called";
    return out;
  }
  if (ClassifyPointer(device_input) != PointerKind::kDevice) {
    out.error = "input chunk is not device-resident";
    return out;
  }

  gpucompress_config_t cfg = gpucompress_default_config();
  cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
  cfg.error_bound = error_bound;
  cfg.cuda_device = 0;
  cfg.cuda_stream = nullptr;
  // Quantization is part of the action space upstream decodes, so the
  // preprocessing bitmask is not what gates it -- decodeAction is. Leave the
  // config's own mask alone and let the selected action decide, which is what
  // the AUTO path does.

  CompContext *ctx = gpucompress::acquireCompContext();
  if (ctx == nullptr) {
    out.error = "acquireCompContext returned null";
    return out;
  }

  // Diagnostics history is per-process; reset it so the record read back below
  // belongs to this chunk and no other.
  gpucompress_reset_chunk_history();

  const AutoStatsGPU *d_stats = nullptr;

  // ---------------- statistics ----------------
  {
    StageScope stage(Stage::kStatistics, device_input, bytes);
    stage->note = "gpucompress::runStatsKernelsNoSync (stats_kernel.cu)";
    stage->stream = ctx->stream;
    d_stats = gpucompress::runStatsKernelsNoSync(device_input, bytes,
                                                 ctx->stream, ctx);
    if (d_stats == nullptr) {
      stage.Fail("runStatsKernelsNoSync returned null");
      gpucompress::releaseCompContext(ctx);
      out.error = "native statistics failed";
      return out;
    }
    cudaStreamSynchronize(ctx->stream);
    stage.SetOutput(d_stats, sizeof(AutoStatsGPU));
  }

  TraceEntry *diag_entry = nullptr;

  // ---------------- inference / ranking / selection ----------------
  //
  // One call produces all three, because upstream produces all three in one
  // call: gpucompress_infer_gpu returns the winning action, the ranked action
  // list (out_top_actions) and the per-config predicted costs. Splitting it
  // into three trace entries reflects the three DECISIONS, not three calls --
  // the trace says so in each entry's note.
  int action = -1;
  float pred_ratio = 0.0f, pred_ct = 0.0f, pred_dt = 0.0f, pred_psnr = 0.0f;
  float pred_rmse = 0.0f, pred_max_err = 0.0f, pred_mae = 0.0f, pred_ssim = 0.0f;
  std::vector<int> top_actions(NN_NUM_CONFIGS, -1);
  std::vector<float> predicted_costs(NN_NUM_CONFIGS, 0.0f);
  std::vector<NNDebugPerConfig> per_config(NN_NUM_CONFIGS);
  gpucompress_stats_t native_stats{};

  {
    StageScope stage(Stage::kInference, d_stats, sizeof(AutoStatsGPU));
    stage->note =
        "gpucompress_infer_gpu -> runNNFusedInferenceCtx; stats stay on the "
        "device, the kernel reads d_stats_ptr directly";
    stage->stream = ctx->stream;
    const gpucompress_error_t rc = gpucompress_infer_gpu(
        device_input, bytes, &cfg, &native_stats, ctx, &action, &pred_ratio,
        &pred_ct, &pred_dt, &pred_psnr, top_actions.data(),
        predicted_costs.data(), &pred_rmse, &pred_max_err, &pred_mae,
        &pred_ssim, per_config.data());
    if (rc != GPUCOMPRESS_SUCCESS || action < 0) {
      stage.Fail(std::string("gpucompress_infer_gpu: ") +
                 gpucompress_error_string(rc));
      gpucompress::releaseCompContext(ctx);
      out.error = "native inference failed";
      return out;
    }
    stage->inference.valid = true;
    stage->inference.num_candidates = NN_NUM_CONFIGS;
    for (int a = 0; a < NN_NUM_CONFIGS && a < 32; ++a) {
      stage->inference.present[a] = true;
      stage->inference.ratio[a] = per_config[a].ratio;
      stage->inference.comp_ms[a] = per_config[a].comp_time;
      stage->inference.decomp_ms[a] = per_config[a].decomp_time;
      stage->inference.psnr_db[a] = per_config[a].psnr;
    }
    stage.SetOutput(nullptr, 0);
  }

  {
    StageScope stage(Stage::kRanking, nullptr, 0);
    stage->note =
        "out_top_actions / out_predicted_costs from the same "
        "gpucompress_infer_gpu call -- upstream's own ranking outputs, not a "
        "re-derivation";
    stage->ranking.valid = true;
    for (int i = 0; i < NN_NUM_CONFIGS && i < 32; ++i) {
      if (top_actions[i] >= 0) stage->ranking.order.push_back(top_actions[i]);
    }
    for (int a = 0; a < NN_NUM_CONFIGS && a < 32; ++a) {
      stage->ranking.cost_present[a] = true;
      stage->ranking.cost[a] = per_config[a].cost;
    }
    stage.SetOutput(nullptr, 0);
  }

  const gpucompress::DecodedAction decoded = gpucompress::decodeAction(action);
  out.action = action;
  out.algo = decoded.algorithm;
  out.quantize_selected = decoded.use_quantization && error_bound > 0.0;
  out.shuffle_selected = decoded.shuffle_size > 0;

  {
    StageScope stage(Stage::kSelection, nullptr, 0);
    stage->note = "decodeAction(action) -- internal.hpp:164";
    stage->selection.valid = true;
    stage->selection.action = action;
    stage->selection.algo = decoded.algorithm;
    stage->selection.algo_name = AlgoName(decoded.algorithm);
    stage->selection.quantize = decoded.use_quantization;
    stage->selection.shuffle = decoded.shuffle_size > 0;
    stage.SetOutput(nullptr, 0);
  }

  // ---------------- diagnostics ----------------
  //
  // Placed HERE, after selection, to sit where Clio's production read sits
  // (compressor_runtime.cc:714). It could not be placed where NATIVE produces
  // the equivalent, because native produces no such thing at this point in the
  // pipeline: its AUTO path never brings the features to the host at all and
  // deliberately zeroes them in the stats it reports
  // (gpucompress_compress.cpp:323-325). That asymmetry is a REAL and already
  // recorded observability difference, and the comparison says so explicitly
  // rather than this position implying the two are equivalent.
  //
  // What native does produce is a per-chunk diagnostics RECORD, written at the
  // end of the compress call. That is folded into this same entry below, once
  // the compress call has run.
  {
    StageScope stage(Stage::kDiagnostics, d_stats, sizeof(AutoStatsGPU));
    stage->note =
        "STAGE PROBE: 24-byte D->H of "
        "AutoStatsGPU{entropy,mad_normalized,deriv_normalized}, the layout "
        "upstream fetches in one copy (stats_kernel.cu:441-446). Tagged "
        "TEST_HARNESS_TRANSFER: upstream's AUTO path makes no such read";
    recorder.PushHarness();
    const bool ok = ReadStatsFeatures(d_stats, &out.entropy, &out.mad,
                                      &out.second_derivative);
    recorder.PopHarness();
    stage->stats.valid = ok;
    stage->stats.entropy = out.entropy;
    stage->stats.mad = out.mad;
    stage->stats.second_derivative = out.second_derivative;
    if (!ok) stage.Fail("D->H of AutoStatsGPU failed");
    diag_entry = stage.get();
    stage.SetOutput(nullptr, 0);
  }

  // ---------------- quantization (stage probe) ----------------
  //
  // Gate copied from the compress path verbatim: PREPROC bit from the action
  // AND cfg.error_bound > 0.0 (gpucompress_compress.cpp:434). A selected
  // quantize action with a zero bound does NOT quantize, on either side.
  const void *pipeline_buffer = device_input;
  size_t pipeline_bytes = bytes;

  if (decoded.use_quantization && error_bound > 0.0) {
    StageScope stage(Stage::kQuantization, pipeline_buffer, pipeline_bytes);
    stage->note =
        "STAGE PROBE: quantize_simple, the same call the compress path makes "
        "internally (gpucompress_compress.cpp:434-455), run separately because "
        "the public API returns no intermediate buffer";
    stage->stream = ctx->stream;

    QuantizationConfig qcfg;
    qcfg.type = QuantizationType::LINEAR;
    qcfg.precision = QuantizationPrecision::AUTO;
    qcfg.error_bound = error_bound;
    qcfg.num_elements = bytes / sizeof(float);
    qcfg.element_size = sizeof(float);

    QuantizationResult qres = quantize_simple(
        const_cast<void *>(pipeline_buffer), qcfg.num_elements,
        sizeof(float), qcfg, ctx->d_range_min, ctx->d_range_max, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    if (qres.d_quantized == nullptr || qres.quantized_bytes == 0) {
      stage.Fail("quantize_simple produced no output");
    } else {
      // The buffer may belong to the context pool (owns_output == false), in
      // which case a later call would overwrite it. Take our own copy so the
      // comparison is against the bytes THIS chunk produced.
      void *keep = nullptr;
      if (cudaMalloc(&keep, qres.quantized_bytes) == cudaSuccess &&
          cudaMemcpy(keep, qres.d_quantized, qres.quantized_bytes,
                     cudaMemcpyDeviceToDevice) == cudaSuccess) {
        out.quantized_device = keep;
        out.quantized_bytes = qres.quantized_bytes;
        pipeline_buffer = keep;
        pipeline_bytes = qres.quantized_bytes;
      } else {
        stage.Fail("could not retain the quantized buffer");
      }
      if (qres.owns_output && qres.d_quantized != nullptr) {
        cudaFree(qres.d_quantized);
      }
      stage->quant.valid = true;
      stage->quant.precision = qres.actual_precision;
      stage->quant.error_bound = error_bound;
      stage->quant.effective_error_bound = qres.error_bound;
      stage->quant.scale = qres.scale_factor;
      stage->quant.data_min = qres.data_min;
      stage->quant.data_max = qres.data_max;
      stage->quant.bound_achievable = true;

      // Round trip through upstream's OWN dequantizer and check the guarantee
      // the caller actually asked for. Byte-equality with Clio says the two
      // agree; it does not say either of them honours the bound.
      if (out.quantized_device != nullptr) {
        QuantizationResult meta = qres;
        meta.d_quantized = const_cast<void *>(out.quantized_device);
        meta.owns_output = false;
        recorder.PushHarness();
        void *d_restored =
            dequantize_simple(meta.d_quantized, meta, ctx->stream);
        cudaStreamSynchronize(ctx->stream);
        if (d_restored != nullptr) {
          out.dequantized_device = d_restored;
          out.dequantized_bytes = qcfg.num_elements * sizeof(float);
          out.bound_checked = CountBoundViolations(
              static_cast<const float *>(device_input),
              static_cast<const float *>(d_restored), qcfg.num_elements,
              error_bound, &out.bound_violations);
        }
        recorder.PopHarness();
      }
    }
    stage.SetOutput(out.quantized_device, out.quantized_bytes);
  } else {
    recorder.RecordSkipped(
        Stage::kQuantization,
        decoded.use_quantization
            ? "action selects quantize but error_bound <= 0, so the compress "
              "path does not run it (gpucompress_compress.cpp:434)"
            : "action does not select quantization");
  }

  // ---------------- shuffle (stage probe) ----------------
  if (decoded.shuffle_size > 0) {
    StageScope stage(Stage::kShuffle, pipeline_buffer, pipeline_bytes);
    stage->note =
        "STAGE PROBE: byte_shuffle_simple on the post-quantization buffer, as "
        "the compress path does (gpucompress_compress.cpp:457-470)";
    stage->stream = ctx->stream;
    uint8_t *shuffled = byte_shuffle_simple(
        const_cast<void *>(pipeline_buffer), pipeline_bytes,
        static_cast<unsigned>(decoded.shuffle_size),
        gpucompress::SHUFFLE_CHUNK_SIZE, ctx->stream);
    if (shuffled == nullptr) {
      stage.Fail("byte_shuffle_simple returned null");
      stage.SetOutput(nullptr, 0);
    } else {
      out.shuffled_device = shuffled;
      out.shuffled_bytes = pipeline_bytes;
      pipeline_buffer = shuffled;
      stage.SetOutput(shuffled, pipeline_bytes);
    }
  } else {
    recorder.RecordSkipped(Stage::kShuffle,
                           "action does not select byte shuffle");
  }

  // ---------------- compression (the real production call) ----------------
  size_t output_cap = gpucompress_max_compressed_size(bytes);
  void *d_output = nullptr;
  if (cudaMalloc(&d_output, output_cap) != cudaSuccess) {
    gpucompress::releaseCompContext(ctx);
    out.error = "could not allocate the native output buffer";
    return out;
  }

  {
    StageScope stage(Stage::kCompression, device_input, bytes);
    stage->note =
        "gpucompress_compress_with_action_gpu -- the real production call. It "
        "re-runs quantize and shuffle internally; the kernels it launched are "
        "listed here";
    size_t output_size = output_cap;
    const gpucompress_error_t rc = gpucompress_compress_with_action_gpu(
        device_input, bytes, d_output, &output_size, &cfg, &native_stats,
        /*stream_arg=*/nullptr, action, pred_ratio, pred_ct, pred_dt,
        pred_psnr, top_actions.data(), predicted_costs.data(),
        /*stage1_nn_ms=*/0.0f, /*stage1_stats_ms=*/0.0f,
        const_cast<AutoStatsGPU *>(d_stats), /*out_diag_slot=*/nullptr,
        pred_rmse, pred_max_err, pred_mae, pred_ssim);
    if (rc != GPUCOMPRESS_SUCCESS) {
      stage.Fail(std::string("gpucompress_compress_with_action_gpu: ") +
                 gpucompress_error_string(rc));
      cudaFree(d_output);
      gpucompress::releaseCompContext(ctx);
      out.error = "native compression failed";
      return out;
    }
    out.compressed_device = d_output;
    out.compressed_bytes = output_size;
    // GPUCOMPRESS_HEADER_SIZE is 64 and the codec payload starts right after
    // it. The payload is what can be compared against Clio, whose container is
    // a different shape by design (see D15-1).
    if (output_size > GPUCOMPRESS_HEADER_SIZE) {
      out.payload_device =
          static_cast<const char *>(d_output) + GPUCOMPRESS_HEADER_SIZE;
      out.payload_bytes = output_size - GPUCOMPRESS_HEADER_SIZE;
    }
    stage->compression.valid = true;
    stage->compression.algo_name = AlgoName(decoded.algorithm);
    stage->compression.compressed_bytes = output_size;
    stage.SetOutput(d_output, output_size);
  }

  // ---------------- fold in the native per-chunk diagnostics record ----------
  //
  // Now that the compress call has run, upstream's own record exists. Read it
  // through the public API and attach it to the diagnostics entry.
  if (diag_entry != nullptr && gpucompress_get_chunk_history_count() > 0) {
    gpucompress_chunk_diag_t diag{};
    if (gpucompress_get_chunk_diag(0, &diag) == 0) {
      auto &d = diag_entry->diagnostics;
      d.valid = true;
      d.native_record_present = true;
      d.nn_action = diag.nn_action;
      d.nn_original_action = diag.nn_original_action;
      d.exploration_triggered = diag.exploration_triggered;
      d.sgd_fired = diag.sgd_fired;
      d.feat_entropy = diag.feat_entropy;
      d.feat_mad = diag.feat_mad;
      d.feat_deriv = diag.feat_deriv;
      d.predicted_ratio = diag.predicted_ratio;
      d.predicted_comp_time = diag.predicted_comp_time;
      d.predicted_decomp_time = diag.predicted_decomp_time;
      d.predicted_psnr = diag.predicted_psnr;
      d.predicted_ranking_count = diag.predicted_ranking_count;
      for (int i = 0; i < 32 && i < diag.predicted_ranking_count; ++i) {
        d.predicted_ranking[i] = diag.predicted_ranking[i];
      }
    }
  }

  gpucompress::releaseCompContext(ctx);
  out.ok = true;
  (void)chunk_id;
  return out;
}

void NativeReleaseChunk(NativeChunkResult *result) {
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
  if (result->compressed_device != nullptr) {
    cudaFree(const_cast<void *>(result->compressed_device));
    result->compressed_device = nullptr;
    result->payload_device = nullptr;
  }
}

}  // namespace npeq
