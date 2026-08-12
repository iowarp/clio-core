/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file neuropress_nn_gpu_kernels.cu
 * @brief CUDA kernels for NeuroPress inference and online SGD training.
 *
 * Ports NeuroPress's nn_gpu.cu (nnFusedInferenceKernel, nnSGDKernel) to
 * Clio's own weight layout. Weights, EMA gradient state, and uncertainty
 * log-variance all live in device memory for the lifetime of the handle --
 * the host never touches the decision data, matching the original design.
 * Per-sample activations and gradient accumulators are cached in a
 * persistent device scratch buffer (allocated once, not per-call) rather
 * than shared memory, trading a little device memory for a simpler,
 * easier-to-verify kernel; the numerical algorithm itself is unchanged from
 * upstream (same clamps, same constants, same order of operations).
 */

#include "clio_ctp/compress/model/neuropress_nn_gpu_kernels.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

namespace ctp::compress::model::gpu {

namespace {
constexpr int kInputDim = 8;
constexpr int kHiddenDim = 64;
constexpr int kOutputDim = 8;
constexpr int kMaxSamples = 8;  // NeuroPress's NN_MAX_SGD_SAMPLES

constexpr int kW1 = kHiddenDim * kInputDim;   // 512
constexpr int kW234 = kHiddenDim * kHiddenDim;  // 4096 each (W2,W3,W4)
constexpr int kW5 = kOutputDim * kHiddenDim;   // 512
constexpr int kParamCount =
    kW1 + kHiddenDim +          // w1,b1
    kW234 + kHiddenDim +        // w2,b2
    kW234 + kHiddenDim +        // w3,b3
    kW234 + kHiddenDim +        // w4,b4
    kW5 + kOutputDim;           // w5,b5  == 13576, matches NeuroPress's SGD_REGION

// Offsets into a flat kParamCount-length weights-then-biases buffer.
constexpr int kOffW1 = 0;
constexpr int kOffB1 = kOffW1 + kW1;
constexpr int kOffW2 = kOffB1 + kHiddenDim;
constexpr int kOffB2 = kOffW2 + kW234;
constexpr int kOffW3 = kOffB2 + kHiddenDim;
constexpr int kOffB3 = kOffW3 + kW234;
constexpr int kOffW4 = kOffB3 + kHiddenDim;
constexpr int kOffB4 = kOffW4 + kW234;
constexpr int kOffW5 = kOffB4 + kHiddenDim;
constexpr int kOffB5 = kOffW5 + kW5;
static_assert(kOffB5 + kOutputDim == kParamCount, "offset layout mismatch");
}  // namespace

/** @brief Device-resident state. Persists for the handle's lifetime. */
struct NeuroPressGpuWeights {
  float params[kParamCount];        // w1,b1,w2,b2,w3,b3,w4,b4,w5,b5
  float x_means[kInputDim];
  float x_stds[kInputDim];
  float y_means[kOutputDim];
  float y_stds[kOutputDim];

  // Online-learning state (never serialized to .nnwt, matches upstream).
  float log_var[kOutputDim];
  float ema[kParamCount];
  int sgd_call_count;

  // Persistent scratch for Train() -- avoids per-call cudaMalloc.
  float act_x[kMaxSamples][kInputDim];
  float act_z1[kMaxSamples][kHiddenDim], act_h1[kMaxSamples][kHiddenDim];
  float act_z2[kMaxSamples][kHiddenDim], act_h2[kMaxSamples][kHiddenDim];
  float act_z3[kMaxSamples][kHiddenDim], act_h3[kMaxSamples][kHiddenDim];
  float act_z4[kMaxSamples][kHiddenDim], act_h4[kMaxSamples][kHiddenDim];
  float act_y[kMaxSamples][kOutputDim];
  float d5_clamped[kMaxSamples][kOutputDim];
  float d5_raw[kMaxSamples][kOutputDim];
  float combined[kParamCount];
  float out_grad[kParamCount];
  float dz4_all[kOutputDim][kHiddenDim];
};

NeuroPressGpuWeights *NeuroPressGpuLoad(const float *weights, size_t weights_len,
                                        const float *biases, size_t biases_len,
                                        const float *x_means, const float *x_stds,
                                        const float *y_means, const float *y_stds) {
  // Layout sanity: caller's flattened weights_ (13312 = 512+3*4096+512) plus
  // biases_ (264 = 4*64+8) must add up to exactly kParamCount (13576) when
  // interleaved below. If the .nnwt format ever changes shape this catches
  // it instead of silently corrupting device memory.
  if (weights_len + biases_len != static_cast<size_t>(kParamCount)) {
    return nullptr;
  }

  NeuroPressGpuWeights *device_w = nullptr;
  if (cudaMalloc(&device_w, sizeof(NeuroPressGpuWeights)) != cudaSuccess) {
    return nullptr;
  }
  if (cudaMemset(device_w, 0, sizeof(NeuroPressGpuWeights)) != cudaSuccess) {
    cudaFree(device_w);
    return nullptr;
  }

  // Host-side: interleave weights_[]/biases_[] (Clio's own layout, 5
  // separate weight matrices + 5 separate bias vectors) into this kernel's
  // single flat params[] buffer (w1,b1,w2,b2,...), matching the offsets
  // above. weights_len/biases_len callers pass the FULL flattened arrays;
  // per-layer sizes are fixed by the architecture (8->64->64->64->64->8).
  float host_params[kParamCount];
  size_t w_off[5] = {0, kW1, kW1 + kW234, kW1 + 2 * kW234, kW1 + 3 * kW234};
  size_t b_off[5] = {0, kHiddenDim, 2 * kHiddenDim, 3 * kHiddenDim,
                     4 * kHiddenDim};
  int dst_w_off[5] = {kOffW1, kOffW2, kOffW3, kOffW4, kOffW5};
  int dst_b_off[5] = {kOffB1, kOffB2, kOffB3, kOffB4, kOffB5};
  int w_sizes[5] = {kW1, kW234, kW234, kW234, kW5};
  int b_sizes[5] = {kHiddenDim, kHiddenDim, kHiddenDim, kHiddenDim, kOutputDim};
  for (int layer = 0; layer < 5; ++layer) {
    std::memcpy(host_params + dst_w_off[layer], weights + w_off[layer],
                sizeof(float) * static_cast<size_t>(w_sizes[layer]));
    std::memcpy(host_params + dst_b_off[layer], biases + b_off[layer],
                sizeof(float) * static_cast<size_t>(b_sizes[layer]));
  }

  bool ok = true;
  ok &= cudaMemcpy(&device_w->params, host_params, sizeof(host_params),
                   cudaMemcpyHostToDevice) == cudaSuccess;
  ok &= cudaMemcpy(&device_w->x_means, x_means, sizeof(float) * kInputDim,
                   cudaMemcpyHostToDevice) == cudaSuccess;
  ok &= cudaMemcpy(&device_w->x_stds, x_stds, sizeof(float) * kInputDim,
                   cudaMemcpyHostToDevice) == cudaSuccess;
  ok &= cudaMemcpy(&device_w->y_means, y_means, sizeof(float) * kOutputDim,
                   cudaMemcpyHostToDevice) == cudaSuccess;
  ok &= cudaMemcpy(&device_w->y_stds, y_stds, sizeof(float) * kOutputDim,
                   cudaMemcpyHostToDevice) == cudaSuccess;
  if (!ok) {
    cudaFree(device_w);
    return nullptr;
  }
  return device_w;
}

void NeuroPressGpuFree(NeuroPressGpuWeights *w) {
  if (w) cudaFree(w);
}

void NeuroPressGpuDownloadWeights(NeuroPressGpuWeights *w, float *weights_out,
                                  float *biases_out) {
  if (!w) return;
  float host_params[kParamCount];
  cudaMemcpy(host_params, &w->params, sizeof(host_params),
            cudaMemcpyDeviceToHost);
  int src_w_off[5] = {kOffW1, kOffW2, kOffW3, kOffW4, kOffW5};
  int src_b_off[5] = {kOffB1, kOffB2, kOffB3, kOffB4, kOffB5};
  size_t dst_w_off[5] = {0, kW1, kW1 + kW234, kW1 + 2 * kW234,
                         kW1 + 3 * kW234};
  size_t dst_b_off[5] = {0, kHiddenDim, 2 * kHiddenDim, 3 * kHiddenDim,
                         4 * kHiddenDim};
  int w_sizes[5] = {kW1, kW234, kW234, kW234, kW5};
  int b_sizes[5] = {kHiddenDim, kHiddenDim, kHiddenDim, kHiddenDim, kOutputDim};
  for (int layer = 0; layer < 5; ++layer) {
    std::memcpy(weights_out + dst_w_off[layer], host_params + src_w_off[layer],
               sizeof(float) * static_cast<size_t>(w_sizes[layer]));
    std::memcpy(biases_out + dst_b_off[layer], host_params + src_b_off[layer],
               sizeof(float) * static_cast<size_t>(b_sizes[layer]));
  }
}

void NeuroPressGpuUploadWeights(NeuroPressGpuWeights *w, const float *weights,
                                const float *biases) {
  if (!w || !weights || !biases) return;
  // Read-modify-write: params[] also holds nothing else, but the device copy
  // is the live one, so start from it rather than zero-filling anything this
  // mapping does not cover.
  float host_params[kParamCount];
  cudaMemcpy(host_params, &w->params, sizeof(host_params),
            cudaMemcpyDeviceToHost);
  int dst_w_off[5] = {kOffW1, kOffW2, kOffW3, kOffW4, kOffW5};
  int dst_b_off[5] = {kOffB1, kOffB2, kOffB3, kOffB4, kOffB5};
  size_t src_w_off[5] = {0, kW1, kW1 + kW234, kW1 + 2 * kW234,
                         kW1 + 3 * kW234};
  size_t src_b_off[5] = {0, kHiddenDim, 2 * kHiddenDim, 3 * kHiddenDim,
                         4 * kHiddenDim};
  int w_sizes[5] = {kW1, kW234, kW234, kW234, kW5};
  int b_sizes[5] = {kHiddenDim, kHiddenDim, kHiddenDim, kHiddenDim, kOutputDim};
  for (int layer = 0; layer < 5; ++layer) {
    std::memcpy(host_params + dst_w_off[layer], weights + src_w_off[layer],
               sizeof(float) * static_cast<size_t>(w_sizes[layer]));
    std::memcpy(host_params + dst_b_off[layer], biases + src_b_off[layer],
               sizeof(float) * static_cast<size_t>(b_sizes[layer]));
  }
  cudaMemcpy(&w->params, host_params, sizeof(host_params),
            cudaMemcpyHostToDevice);
}

// ============================================================================
// Inference: one block per candidate, thread t owns hidden neuron t.
// ============================================================================
__global__ void InferKernel(const NeuroPressGpuWeights *__restrict__ w,
                            const float *__restrict__ raw_inputs,
                            float *__restrict__ out_comp_time,
                            float *__restrict__ out_decomp_time,
                            float *__restrict__ out_ratio,
                            float *__restrict__ out_psnr) {
  int cand = blockIdx.x;
  int t = threadIdx.x;

  __shared__ float s_x[kInputDim];
  __shared__ float s_h1[kHiddenDim], s_h2[kHiddenDim], s_h3[kHiddenDim],
      s_h4[kHiddenDim];
  __shared__ float s_y[kOutputDim];

  if (t < kInputDim) {
    float std_val = w->x_stds[t];
    if (std_val < 1e-8f) std_val = 1e-8f;
    s_x[t] = (raw_inputs[cand * kInputDim + t] - w->x_means[t]) / std_val;
  }
  __syncthreads();

  float sum = w->params[kOffB1 + t];
  for (int i = 0; i < kInputDim; ++i) sum += w->params[kOffW1 + t * kInputDim + i] * s_x[i];
  s_h1[t] = fmaxf(0.0f, sum);
  __syncthreads();

  sum = w->params[kOffB2 + t];
  for (int i = 0; i < kHiddenDim; ++i) sum += w->params[kOffW2 + t * kHiddenDim + i] * s_h1[i];
  s_h2[t] = fmaxf(0.0f, sum);
  __syncthreads();

  sum = w->params[kOffB3 + t];
  for (int i = 0; i < kHiddenDim; ++i) sum += w->params[kOffW3 + t * kHiddenDim + i] * s_h2[i];
  s_h3[t] = fmaxf(0.0f, sum);
  __syncthreads();

  sum = w->params[kOffB4 + t];
  for (int i = 0; i < kHiddenDim; ++i) sum += w->params[kOffW4 + t * kHiddenDim + i] * s_h3[i];
  s_h4[t] = fmaxf(0.0f, sum);
  __syncthreads();

  if (t < kOutputDim) {
    float o = w->params[kOffB5 + t];
    for (int i = 0; i < kHiddenDim; ++i) o += w->params[kOffW5 + t * kHiddenDim + i] * s_h4[i];
    s_y[t] = o;
  }
  __syncthreads();

  if (t == 0) {
    float comp_time = expm1f(s_y[0] * w->y_stds[0] + w->y_means[0]);
    float decomp_time = expm1f(s_y[1] * w->y_stds[1] + w->y_means[1]);
    float ratio = expm1f(s_y[2] * w->y_stds[2] + w->y_means[2]);
    float psnr = s_y[3] * w->y_stds[3] + w->y_means[3];
    out_comp_time[cand] = fmaxf(1.0f, comp_time);
    out_decomp_time[cand] = decomp_time;
    out_ratio[cand] = fmaxf(0.001f, fminf(100.0f, ratio));
    out_psnr[cand] = psnr;
  }
}

void NeuroPressGpuInferBatch(NeuroPressGpuWeights *w, const float *raw_inputs,
                             int num_candidates, float *out_comp_time_ms,
                             float *out_decomp_time_ms, float *out_ratio,
                             float *out_psnr_db) {
  if (!w || num_candidates <= 0) return;

  float *d_in = nullptr, *d_ct = nullptr, *d_dt = nullptr, *d_r = nullptr,
        *d_p = nullptr;
  size_t in_bytes = sizeof(float) * static_cast<size_t>(num_candidates) * kInputDim;
  size_t out_bytes = sizeof(float) * static_cast<size_t>(num_candidates);
  cudaMalloc(&d_in, in_bytes);
  cudaMalloc(&d_ct, out_bytes);
  cudaMalloc(&d_dt, out_bytes);
  cudaMalloc(&d_r, out_bytes);
  cudaMalloc(&d_p, out_bytes);

  cudaMemcpy(d_in, raw_inputs, in_bytes, cudaMemcpyHostToDevice);
  InferKernel<<<num_candidates, kHiddenDim>>>(w, d_in, d_ct, d_dt, d_r, d_p);
  cudaMemcpy(out_comp_time_ms, d_ct, out_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(out_decomp_time_ms, d_dt, out_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(out_ratio, d_r, out_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(out_psnr_db, d_p, out_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_ct);
  cudaFree(d_dt);
  cudaFree(d_r);
  cudaFree(d_p);
}

// ============================================================================
// SGD: single block <<<1, kHiddenDim>>>, mirrors nnSGDKernel's algorithm.
// ============================================================================
__device__ __forceinline__ void ForwardOneLayer(
    const float *__restrict__ w_layer, const float *__restrict__ b_layer,
    const float *in, int fan_in, int t, float &z_out, float &h_out) {
  float sum = b_layer[t];
  for (int i = 0; i < fan_in; ++i) sum += w_layer[t * fan_in + i] * in[i];
  z_out = sum;
  h_out = fmaxf(0.0f, sum);
}

__global__ void SGDKernel(NeuroPressGpuWeights *w,
                          const NeuroPressGpuSGDSample *__restrict__ samples,
                          int num_samples, bool *out_applied) {
  int t = threadIdx.x;  // 0..63
  __shared__ float s_reduce[kHiddenDim];

  // ---- Phase 1: per-sample forward pass + target/error computation ----
  for (int si = 0; si < num_samples; ++si) {
    if (t < kInputDim) {
      float std_val = w->x_stds[t];
      if (std_val < 1e-8f) std_val = 1e-8f;
      w->act_x[si][t] = (samples[si].raw_input[t] - w->x_means[t]) / std_val;
    }
    __syncthreads();

    float z, h;
    ForwardOneLayer(&w->params[kOffW1], &w->params[kOffB1], w->act_x[si],
                    kInputDim, t, z, h);
    w->act_z1[si][t] = z;
    w->act_h1[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW2], &w->params[kOffB2], w->act_h1[si],
                    kHiddenDim, t, z, h);
    w->act_z2[si][t] = z;
    w->act_h2[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW3], &w->params[kOffB3], w->act_h2[si],
                    kHiddenDim, t, z, h);
    w->act_z3[si][t] = z;
    w->act_h3[si][t] = h;
    __syncthreads();

    ForwardOneLayer(&w->params[kOffW4], &w->params[kOffB4], w->act_h3[si],
                    kHiddenDim, t, z, h);
    w->act_z4[si][t] = z;
    w->act_h4[si][t] = h;
    __syncthreads();

    if (t < kOutputDim) {
      float o = w->params[kOffB5 + t];
      for (int i = 0; i < kHiddenDim; ++i)
        o += w->params[kOffW5 + t * kHiddenDim + i] * w->act_h4[si][i];
      w->act_y[si][t] = o;
    }
    __syncthreads();

    if (t == 0) {
      const NeuroPressGpuSGDSample &s = samples[si];
      float d5[kOutputDim];

      float clamped_ratio = fmaxf(0.5f, fminf(s.actual_ratio, 10000.0f));
      float y_std2 = fmaxf(w->y_stds[2], 1e-8f);
      d5[2] = w->act_y[si][2] - (log1pf(clamped_ratio) - w->y_means[2]) / y_std2;

      if (s.actual_comp_time_ms > 0.0f) {
        float clamped = fmaxf(0.01f, fminf(s.actual_comp_time_ms, 5000.0f));
        float y_std0 = fmaxf(w->y_stds[0], 1e-8f);
        d5[0] = w->act_y[si][0] - (log1pf(clamped) - w->y_means[0]) / y_std0;
      } else {
        d5[0] = 0.0f;
      }
      if (s.actual_decomp_time_ms > 0.0f) {
        float clamped = fmaxf(0.01f, fminf(s.actual_decomp_time_ms, 5000.0f));
        float y_std1 = fmaxf(w->y_stds[1], 1e-8f);
        d5[1] = w->act_y[si][1] - (log1pf(clamped) - w->y_means[1]) / y_std1;
      } else {
        d5[1] = 0.0f;
      }
      if (s.actual_psnr_db >= 0.0f) {
        float psnr_val = (s.actual_psnr_db == 0.0f) ? 120.0f : s.actual_psnr_db;
        float clamped_psnr = fminf(psnr_val, 120.0f);
        float y_std3 = fmaxf(w->y_stds[3], 1e-8f);
        d5[3] = w->act_y[si][3] - (clamped_psnr - w->y_means[3]) / y_std3;
      } else {
        d5[3] = 0.0f;
      }
      for (int o = 4; o < kOutputDim; ++o) d5[o] = 0.0f;

      constexpr float kNoiseGateThresh = 0.10f;
      if (fabsf(d5[0]) < kNoiseGateThresh) d5[0] = 0.0f;

      for (int o = 0; o < kOutputDim; ++o) w->d5_raw[si][o] = d5[o];

      constexpr float kSgdErrorDelta = 0.5f;
      for (int o = 0; o < kOutputDim; ++o)
        d5[o] = fmaxf(-kSgdErrorDelta, fminf(d5[o], kSgdErrorDelta));
      for (int o = 0; o < kOutputDim; ++o) w->d5_clamped[si][o] = d5[o];
    }
    __syncthreads();
  }

  // ---- Phase 1.5: uncertainty weighting (Kendall et al., 2018) ----
  constexpr float kUwLr = 0.01f;
  constexpr float kUwLogVarMin = -2.0f;
  constexpr float kUwLogVarMax = 4.0f;
  if (t < kOutputDim) {
    float lv = w->log_var[t];
    float precision = expf(fmaxf(-20.0f, fminf(20.0f, -lv)));
    float raw_mse = 0.0f;
    for (int si = 0; si < num_samples; ++si) {
      float e = w->d5_raw[si][t];
      raw_mse += e * e;
    }
    raw_mse /= static_cast<float>(num_samples);
    float grad_lv = 0.5f * (1.0f - precision * raw_mse);
    lv -= kUwLr * grad_lv;
    lv = fmaxf(kUwLogVarMin, fminf(lv, kUwLogVarMax));
    w->log_var[t] = lv;
    float uw = expf(fmaxf(-20.0f, fminf(20.0f, -0.5f * lv)));
    for (int si = 0; si < num_samples; ++si) w->d5_clamped[si][t] *= uw;
  }
  __syncthreads();

  // ---- Phase 2: per-output backward passes with PCGrad-lite ----
  constexpr float kGradClipThreshold = 0.1f;
  constexpr float kPcgradCosThresh = -0.1f;
  constexpr float kDefaultLearningRate = 0.01f;  // NeuroPress's g_reinforce_lr default

  for (int i = t; i < kParamCount; i += kHiddenDim) w->combined[i] = 0.0f;
  __syncthreads();

  for (int target_out = 0; target_out < kOutputDim; ++target_out) {
    for (int i = t; i < kParamCount; i += kHiddenDim) w->out_grad[i] = 0.0f;
    __syncthreads();

    for (int si = 0; si < num_samples; ++si) {
      // Step 1: L4 backward delta for ALL outputs, normalized to unit vectors.
      for (int o = 0; o < kOutputDim; ++o) {
        float es = w->d5_clamped[si][o];
        float dh4_t = w->params[kOffW5 + o * kHiddenDim + t] * es;
        float v = (w->act_z4[si][t] > 0.0f) ? dh4_t : 0.0f;
        w->dz4_all[o][t] = v;
      }
      __syncthreads();
      for (int o = 0; o < kOutputDim; ++o) {
        float local = w->dz4_all[o][t] * w->dz4_all[o][t];
        s_reduce[t] = local;
        __syncthreads();
        for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
          if (t < s) s_reduce[t] += s_reduce[t + s];
          __syncthreads();
        }
        float norm = sqrtf(s_reduce[0]) + 1e-6f;
        w->dz4_all[o][t] /= norm;
        __syncthreads();
      }

      // Step 2: PCGrad projection for target_out against every other output.
      float my_dz4 = w->dz4_all[target_out][t];
      for (int j = 0; j < kOutputDim; ++j) {
        if (j == target_out) continue;
        float local_dot = my_dz4 * w->dz4_all[j][t];
        s_reduce[t] = local_dot;
        __syncthreads();
        for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
          if (t < s) s_reduce[t] += s_reduce[t + s];
          __syncthreads();
        }
        float cos_ij = s_reduce[0];
        if (cos_ij < kPcgradCosThresh) my_dz4 -= cos_ij * w->dz4_all[j][t];
        __syncthreads();
      }
      float err_mag = fabsf(w->d5_clamped[si][target_out]);
      float dz4 = my_dz4 * err_mag;

      // Step 3: W5/b5 gradient uses the ORIGINAL (unprojected) error.
      float error_signal = w->d5_clamped[si][target_out];
      w->out_grad[kOffW5 + target_out * kHiddenDim + t] +=
          error_signal * w->act_h4[si][t];
      if (t == 0) w->out_grad[kOffB5 + target_out] += error_signal;

      // Step 4/4b: L4 gradient, backward L4->L3 (using PROJECTED dz4).
      for (int i = 0; i < kHiddenDim; ++i)
        w->out_grad[kOffW4 + t * kHiddenDim + i] += dz4 * w->act_h3[si][i];
      w->out_grad[kOffB4 + t] += dz4;

      // Broadcast dz4[0..63] through shared memory so every thread can sum
      // weights_[*, t] * dz4[*] for the L4->L3 backward step.
      __shared__ float s_dz4[kHiddenDim];
      s_dz4[t] = dz4;
      __syncthreads();
      float dh3_t = 0.0f;
      for (int j = 0; j < kHiddenDim; ++j)
        dh3_t += w->params[kOffW4 + j * kHiddenDim + t] * s_dz4[j];
      float dz3 = (w->act_z3[si][t] > 0.0f) ? dh3_t : 0.0f;

      // Step 5/5b: L3 gradient, backward L3->L2.
      for (int i = 0; i < kHiddenDim; ++i)
        w->out_grad[kOffW3 + t * kHiddenDim + i] += dz3 * w->act_h2[si][i];
      w->out_grad[kOffB3 + t] += dz3;
      __shared__ float s_dz3[kHiddenDim];
      s_dz3[t] = dz3;
      __syncthreads();
      float dh2_t = 0.0f;
      for (int j = 0; j < kHiddenDim; ++j)
        dh2_t += w->params[kOffW3 + j * kHiddenDim + t] * s_dz3[j];
      float dz2 = (w->act_z2[si][t] > 0.0f) ? dh2_t : 0.0f;

      // Step 6/6b: L2 gradient, backward L2->L1, then L1 gradient.
      for (int i = 0; i < kHiddenDim; ++i)
        w->out_grad[kOffW2 + t * kHiddenDim + i] += dz2 * w->act_h1[si][i];
      w->out_grad[kOffB2 + t] += dz2;
      __shared__ float s_dz2[kHiddenDim];
      s_dz2[t] = dz2;
      __syncthreads();
      float dh1_t = 0.0f;
      for (int j = 0; j < kHiddenDim; ++j)
        dh1_t += w->params[kOffW2 + j * kHiddenDim + t] * s_dz2[j];
      float dz1 = (w->act_z1[si][t] > 0.0f) ? dh1_t : 0.0f;

      for (int i = 0; i < kInputDim; ++i)
        w->out_grad[kOffW1 + t * kInputDim + i] += dz1 * w->act_x[si][i];
      w->out_grad[kOffB1 + t] += dz1;
      __syncthreads();
    }  // per-sample

    // Average over samples, compute this output's gradient norm (only the
    // params it actually touches: shared L1-L4 + its own W5/b5 row).
    float inv_n = 1.0f / static_cast<float>(num_samples);
    float local_norm_sq = 0.0f;
    for (int i = 0; i < kInputDim; ++i) {
      int idx = kOffW1 + t * kInputDim + i;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    { int idx = kOffB1 + t; w->out_grad[idx] *= inv_n; local_norm_sq += w->out_grad[idx] * w->out_grad[idx]; }
    for (int i = 0; i < kHiddenDim; ++i) {
      int idx = kOffW2 + t * kHiddenDim + i;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    { int idx = kOffB2 + t; w->out_grad[idx] *= inv_n; local_norm_sq += w->out_grad[idx] * w->out_grad[idx]; }
    for (int i = 0; i < kHiddenDim; ++i) {
      int idx = kOffW3 + t * kHiddenDim + i;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    { int idx = kOffB3 + t; w->out_grad[idx] *= inv_n; local_norm_sq += w->out_grad[idx] * w->out_grad[idx]; }
    for (int i = 0; i < kHiddenDim; ++i) {
      int idx = kOffW4 + t * kHiddenDim + i;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    { int idx = kOffB4 + t; w->out_grad[idx] *= inv_n; local_norm_sq += w->out_grad[idx] * w->out_grad[idx]; }
    {
      int idx = kOffW5 + target_out * kHiddenDim + t;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }
    if (t == target_out) {
      int idx = kOffB5 + target_out;
      w->out_grad[idx] *= inv_n;
      local_norm_sq += w->out_grad[idx] * w->out_grad[idx];
    }

    s_reduce[t] = local_norm_sq;
    __syncthreads();
    for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
      if (t < s) s_reduce[t] += s_reduce[t + s];
      __syncthreads();
    }
    float out_norm = sqrtf(s_reduce[0]) + 1e-8f;
    float clip_scale = (out_norm > kGradClipThreshold) ? (kGradClipThreshold / out_norm) : 1.0f;
    float lr_out = kDefaultLearningRate * clip_scale;

    for (int i = 0; i < kInputDim; ++i)
      w->combined[kOffW1 + t * kInputDim + i] += lr_out * w->out_grad[kOffW1 + t * kInputDim + i];
    w->combined[kOffB1 + t] += lr_out * w->out_grad[kOffB1 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW2 + t * kHiddenDim + i] += lr_out * w->out_grad[kOffW2 + t * kHiddenDim + i];
    w->combined[kOffB2 + t] += lr_out * w->out_grad[kOffB2 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW3 + t * kHiddenDim + i] += lr_out * w->out_grad[kOffW3 + t * kHiddenDim + i];
    w->combined[kOffB3 + t] += lr_out * w->out_grad[kOffB3 + t];
    for (int i = 0; i < kHiddenDim; ++i)
      w->combined[kOffW4 + t * kHiddenDim + i] += lr_out * w->out_grad[kOffW4 + t * kHiddenDim + i];
    w->combined[kOffB4 + t] += lr_out * w->out_grad[kOffB4 + t];
    {
      int idx = kOffW5 + target_out * kHiddenDim + t;
      w->combined[idx] += lr_out * w->out_grad[idx];
    }
    if (t == target_out) {
      int idx = kOffB5 + target_out;
      w->combined[idx] += lr_out * w->out_grad[idx];
    }
    __syncthreads();
  }  // per target_out

  // ---- Trust-region step, anti-flip damping, EMA smoothing, weight update ----
  constexpr float kEmaDecay = 0.85f;
  constexpr float kTrustK = 0.08f;
  constexpr float kMaxStep = 0.02f;
  constexpr float kMinStep = 1e-4f;
  constexpr float kAntiFlipDamp = 0.5f;
  constexpr float kWClamp = 10.0f;

  // Gradient norm (sum over all params, block-wide reduction via strided
  // per-thread partial sums since kParamCount > kHiddenDim).
  float local_norm_sq = 0.0f;
  for (int i = t; i < kParamCount; i += kHiddenDim) local_norm_sq += w->combined[i] * w->combined[i];
  s_reduce[t] = local_norm_sq;
  __syncthreads();
  for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
    if (t < s) s_reduce[t] += s_reduce[t + s];
    __syncthreads();
  }
  float g_norm = sqrtf(s_reduce[0]) + 1e-8f;
  float inv_norm = 1.0f / g_norm;
  for (int i = t; i < kParamCount; i += kHiddenDim) w->combined[i] *= inv_norm;
  __syncthreads();

  __shared__ float s_avg_err;
  if (t == 0) {
    float sum_err = 0.0f;
    int count = 0;
    for (int si = 0; si < num_samples; ++si) {
      for (int o = 0; o < kOutputDim; ++o) {
        float e = fabsf(w->d5_raw[si][o]);
        if (e > 0.0f) { sum_err += e; ++count; }
      }
    }
    s_avg_err = (count > 0) ? sum_err / static_cast<float>(count) : 0.0f;
  }
  __syncthreads();
  float step = fmaxf(kMinStep, fminf(kMaxStep, kTrustK * s_avg_err));

  bool warmed_up = (w->sgd_call_count > 3);
  if (warmed_up) {
    float local_dot = 0.0f;
    for (int i = t; i < kParamCount; i += kHiddenDim) local_dot += w->combined[i] * w->ema[i];
    s_reduce[t] = local_dot;
    __syncthreads();
    for (int s = kHiddenDim / 2; s > 0; s >>= 1) {
      if (t < s) s_reduce[t] += s_reduce[t + s];
      __syncthreads();
    }
    // s_reduce[0] holds the block-wide dot product after the reduction
    // above; every thread reads the same value, so `step` (a per-thread
    // local) stays consistent across the whole block.
    if (s_reduce[0] < 0.0f) step *= kAntiFlipDamp;
  }

  for (int i = t; i < kParamCount; i += kHiddenDim)
    if (!isfinite(w->combined[i])) w->combined[i] = 0.0f;
  __syncthreads();

  float ema_new = 1.0f - kEmaDecay;
  for (int i = t; i < kParamCount; i += kHiddenDim)
    w->ema[i] = kEmaDecay * w->ema[i] + ema_new * w->combined[i];
  __syncthreads();

  if (t == 0) w->sgd_call_count += 1;

  bool finite_ok = isfinite(step) && isfinite(g_norm);
  if (finite_ok) {
    // Output 1 (decompression time) is owned by the deferred head-only pass
    // (NeuroPressNNPredictor::TrainDecompHead), fed by real measured times a
    // later read supplies -- so this pass must not write its head weights.
    // Mirrors nnSGDKernel's `if (out == 1) continue;` / `t != 1`
    // (nn_gpu.cu). Output 1's error still reaches the shared trunk, exactly
    // as upstream; only w5 row 1 and b5[1] are withheld.
    const int skip_w5_begin = kOffW5 + 1 * kHiddenDim;
    const int skip_w5_end = skip_w5_begin + kHiddenDim;
    const int skip_b5 = kOffB5 + 1;
    for (int i = t; i < kParamCount; i += kHiddenDim) {
      if ((i >= skip_w5_begin && i < skip_w5_end) || i == skip_b5) continue;
      float p = w->params[i] - step * w->ema[i];
      w->params[i] = fmaxf(-kWClamp, fminf(kWClamp, p));
    }
  }
  if (t == 0) *out_applied = finite_ok;
}

bool NeuroPressGpuTrain(NeuroPressGpuWeights *w,
                        const NeuroPressGpuSGDSample *samples,
                        int num_samples) {
  if (!w || num_samples <= 0) return false;
  if (num_samples > kMaxSamples) num_samples = kMaxSamples;

  NeuroPressGpuSGDSample *d_samples = nullptr;
  bool *d_applied = nullptr;
  cudaMalloc(&d_samples, sizeof(NeuroPressGpuSGDSample) * static_cast<size_t>(num_samples));
  cudaMalloc(&d_applied, sizeof(bool));
  cudaMemcpy(d_samples, samples,
            sizeof(NeuroPressGpuSGDSample) * static_cast<size_t>(num_samples),
            cudaMemcpyHostToDevice);

  SGDKernel<<<1, kHiddenDim>>>(w, d_samples, num_samples, d_applied);

  bool applied = false;
  cudaMemcpy(&applied, d_applied, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_samples);
  cudaFree(d_applied);
  return applied;
}

}  // namespace ctp::compress::model::gpu
