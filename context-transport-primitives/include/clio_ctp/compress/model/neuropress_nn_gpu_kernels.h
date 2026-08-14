/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file neuropress_nn_gpu_kernels.h
 * @brief C++-callable interface to the CUDA-compiled NeuroPress GPU kernels.
 *
 * This header has no CUDA-specific types or includes, so it is safe to
 * include from a plain g++-compiled translation unit (neuropress_nn_predictor.cc).
 * The actual kernels live in neuropress_nn_gpu_kernels.cu, compiled by nvcc
 * into a separate static library, matching NeuroPress's own design (src/nn/
 * nn_gpu.cu): the NN's weights, inference forward pass, and SGD training all
 * live and execute on the device -- the host never touches the decision data,
 * only the small per-call inputs/outputs.
 */

#ifndef CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_GPU_KERNELS_H_
#define CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_GPU_KERNELS_H_

#include <cstddef>

namespace ctp::compress::model::gpu {

/** @brief Opaque device-resident weight handle (defined in the .cu TU). */
struct NeuroPressGpuWeights;

/**
 * @brief Allocate device weight storage and upload from host-parsed .nnwt
 * data (same flattened layout NeuroPressNNPredictor::Load() already parses:
 * weights[weight_offsets[layer] + out*fan_in + in], biases[bias_offsets[layer]
 * + out], 5 layers: [8,64],[64,64],[64,64],[64,64],[64,8]).
 *
 * @return Non-null handle on success, nullptr on CUDA allocation failure.
 */
NeuroPressGpuWeights *NeuroPressGpuLoad(const float *weights, size_t weights_len,
                                        const float *biases, size_t biases_len,
                                        const float *x_means, const float *x_stds,
                                        const float *y_means, const float *y_stds);

/** @brief Free a handle returned by NeuroPressGpuLoad(). No-op on nullptr. */
void NeuroPressGpuFree(NeuroPressGpuWeights *w);

/**
 * @brief Download the current device weights/biases back to host (for
 * Save() -- the .nnwt file format is a host/CPU concern, unrelated to
 * where inference/training actually executes).
 */
void NeuroPressGpuDownloadWeights(NeuroPressGpuWeights *w, float *weights_out,
                                  float *biases_out);

/**
 * @brief Push host weights/biases back to the device copy -- the inverse of
 * NeuroPressGpuDownloadWeights(), same host layout.
 *
 * Needed by the deferred decompression-head update
 * (NeuroPressNNPredictor::TrainDecompHead), which does its arithmetic on the
 * host so there is exactly ONE implementation of that ported math to audit,
 * then syncs the result to the device copy inference actually reads.
 */
void NeuroPressGpuUploadWeights(NeuroPressGpuWeights *w, const float *weights,
                                const float *biases);

/**
 * @brief Batched inference: one kernel launch scores every candidate,
 * mirroring NeuroPress's nnFusedInferenceKernel. raw_inputs is row-major
 * [num_candidates x 8], each row [algo_id, quant, shuffle, error_bound,
 * chunk_size, entropy, mad, second_derivative] (NeuroPressNNPredictor's
 * FeaturesTo8Input() order, pre-standardization -- standardization and the
 * log1p/expm1 inverse-transform both happen on-device).
 *
 * @param out_* Each sized [num_candidates], already inverse-transformed
 *   into real units (ratio, ms, dB) -- callers don't need to know about
 *   the log1p training convention.
 *
 * @return false if any CUDA step failed, in which case the out_* buffers
 *   hold nothing meaningful. Callers MUST NOT rank on them: zero-filled
 *   outputs clamp to ratio 0.1 / 1 ms for every candidate, which looks like
 *   a real (if pessimistic) opinion and leaves the winner to sort order.
 *   NeuroPress returns -1 here and its caller reports
 *   GPUCOMPRESS_ERROR_NN_NOT_LOADED (nn_gpu.cu).
 */
bool NeuroPressGpuInferBatch(NeuroPressGpuWeights *w, const float *raw_inputs,
                             int num_candidates, float *out_comp_time_ms,
                             float *out_decomp_time_ms, float *out_ratio,
                             float *out_psnr_db);

/**
 * @brief Same inference, reading the data features from DEVICE memory.
 *
 * The variant above takes a fully host-built [num_candidates][8] matrix, which
 * means the chunk's entropy/MAD/second-derivative must already have been
 * brought to the host. Upstream never does that: gpucompress_infer_gpu hands
 * runNNFusedInferenceCtx an `AutoStatsGPU*` and the kernel reads it on-device
 * ("Stats remain on GPU", gpucompress_compress.cpp).
 *
 * This entry point restores that shape. It is the one to use whenever the
 * chunk itself is device-resident, which is exactly the situation upstream is
 * always in.
 *
 * @param device_stats Device pointer from ctp::ComputeDeviceStatsResident().
 * @param action_ids Host array [num_candidates] of UPSTREAM ACTION INDICES
 *   (`algo + 8*quantize + 16*shuffle`, decodeAction's encoding). The kernel
 *   decodes each one to rebuild inputs 0-2 exactly as upstream rebuilds them
 *   from its thread index (nn_gpu.cu) -- nothing about the
 *   configuration is assembled on the host. Uploaded asynchronously on
 *   `stream`; never waited on.
 * @param error_bound The chunk's RAW bound, as a scalar. The 1e-7 lossless
 *   sentinel is applied in-kernel for unquantized actions, where upstream
 *   applies it (nn_gpu.cu).
 * @param chunk_size_bytes Input 4, passed as a scalar kernel argument.
 * @param stream The stream the statistics were enqueued on -- passing a
 *   different one breaks the chaining and reads the stats before they exist.
 *   Use ctp::DeviceStatsStream().
 *
 * @return false if any CUDA step failed; same contract as the variant above,
 *   the out_* buffers must not be ranked on.
 */
/**
 * @brief Cost-model weights for on-device ranking.
 *
 * Mirrors the fields of RankingWeights that the cost model actually reads
 * (predictor.h:193-197). Passed as a struct so the kernel launch does not grow
 * a parameter list nobody can read.
 */
struct GpuRankParams {
  double data_size_bytes = 0.0;
  double w_compress_time = 1.0;
  double w_decompress_time = 1.0;
  double w_io = 1.0;
  double bandwidth_bytes_per_ms = 5e6;

  /**
   * The two mask inputs, applied in-kernel exactly as nn_gpu.cu does:
   *   quantize actions are masked when error_bound <= 0
   *   any action is masked when min_psnr > 0 and its predicted PSNR is below it
   * A masked action scores -INFINITY, so it loses to every unmasked one but is
   * still ranked -- upstream always returns an action, even when every action
   * is masked. Leave min_psnr at 0 to disable the PSNR mask, which is
   * upstream's own default (g_min_psnr_db, gpucompress_api.cpp).
   */
  double error_bound = 0.0;
  double min_psnr = 0.0;
};

bool NeuroPressGpuInferBatchDeviceStats(
    NeuroPressGpuWeights *w, const void *device_stats, const int *action_ids,
    int num_candidates, float chunk_size_bytes, float error_bound,
    void *stream, float *out_comp_time_ms, float *out_decomp_time_ms,
    float *out_ratio, float *out_psnr_db, const GpuRankParams *rank = nullptr,
    int *out_order = nullptr, double *out_scores = nullptr);

/**
 * @brief One deferred decompression-time observation.
 *
 * Mirrors upstream's DeferredDecompSample: the CONFIGURATION travels as an
 * action index and is decoded in-kernel, not spelled out field by field, so
 * the SGD kernel rebuilds its inputs exactly as the inference kernel does.
 * error_bound_enc is the RAW bound -- the 1e-7 lossless sentinel is applied at
 * inference only, and both of upstream's SGD paths feed the raw value
 * (nn_gpu.cu).
 */
struct NeuroPressGpuDecompSample {
  int action;
  float error_bound_enc;
  float data_size_enc;
  float entropy;
  float mad;
  float second_derivative;
  float actual_decomp_ms;
};

/**
 * @brief Deferred head-only SGD for the decompression-time output, on-device.
 *
 * Updates ONLY w5 row 1 and b5[1]; the trunk is read-only. One block of 64
 * threads, as upstream launches it (nn_gpu.cu). Replaces a host-side
 * read-modify-write that downloaded every parameter, edited two, and uploaded
 * them all back.
 *
 * @return false if the handle is null, there are no samples, or every sample
 *   was gated out as noise -- in which case no weight was touched.
 */
bool NeuroPressGpuTrainDecompHead(NeuroPressGpuWeights *w,
                                  const NeuroPressGpuDecompSample *samples,
                                  int num_samples);

/** @brief One real observed outcome to train on (device-uploaded per call). */
struct NeuroPressGpuSGDSample {
  float raw_input[8];  // same 8-dim layout as NeuroPressGpuInferBatch's rows
  float actual_ratio;
  float actual_comp_time_ms;    // <=0 -> skip this output's gradient
  float actual_decomp_time_ms;  // <=0 -> skip this output's gradient
  float actual_psnr_db;         // <0 -> skip (not applicable); 0 -> lossless
};

/**
 * @brief One online SGD step: forward + backward + weight update, entirely
 * on-device, mirroring NeuroPress's nnSGDKernel (log1p-clamped targets,
 * noise gating, uncertainty weighting, PCGrad-lite, per-output gradient
 * clipping, trust-region step sizing, EMA + anti-flip damping, weight
 * clamping). State (log_var, EMA gradient, call count) persists in the
 * device weight handle across calls.
 *
 * @param num_samples Capped internally to 8 (NeuroPress's own
 *   NN_MAX_SGD_SAMPLES), matching upstream's own per-call batch limit.
 * @return true if the weight update was applied; false if not (a
 *   non-finite step/gradient was detected and safely skipped).
 *
 * @param learning_rate Online-SGD rate, upstream's g_reinforce_lr. Consumed
 *   at exactly one place, `lr_out = learning_rate * clip_scale`
 *   (nn_gpu.cu), matching runNNSGD's own parameter.
 */
bool NeuroPressGpuTrain(NeuroPressGpuWeights *w,
                        const NeuroPressGpuSGDSample *samples,
                        int num_samples, float learning_rate);

}  // namespace ctp::compress::model::gpu

#endif  // CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_GPU_KERNELS_H_
