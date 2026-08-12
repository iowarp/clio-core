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
 */
void NeuroPressGpuInferBatch(NeuroPressGpuWeights *w, const float *raw_inputs,
                             int num_candidates, float *out_comp_time_ms,
                             float *out_decomp_time_ms, float *out_ratio,
                             float *out_psnr_db);

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
 */
bool NeuroPressGpuTrain(NeuroPressGpuWeights *w,
                        const NeuroPressGpuSGDSample *samples,
                        int num_samples);

}  // namespace ctp::compress::model::gpu

#endif  // CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_GPU_KERNELS_H_
