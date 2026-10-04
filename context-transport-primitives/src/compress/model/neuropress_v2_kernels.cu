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

/**
 * @file neuropress_v2_kernels.cu
 * @brief The GPU side of NeuroPress v2: inputs from device statistics, the
 * network, and the ranking of every setting, in one single-block kernel.
 */

#include "neuropress_v2_kernels.h"

#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_INF

#include <cmath>

#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

namespace ctp::compress::model::v2 {

namespace {

constexpr int kThreads = 256;  ///< one block; >= kMaxWidth

/**
 * Thread 0: the four standardised inputs from the chunk's statistics, with
 * the training definitions (see neuropress_v2_predictor.h).
 */
__device__ void BuildInputs(const float *p, const NetDesc &d,
                            const ctp::DeviceFeatureStats *st,
                            double chunk_bytes, float *x_raw, float *x_std) {
  double range = st->value_max - st->value_min;
  if (!(range > 0.0)) range = 1.0;
  x_raw[0] = static_cast<float>(log2(chunk_bytes));
  x_raw[1] = static_cast<float>(st->entropy);
  x_raw[2] = static_cast<float>(log10(st->mad / range + 1e-12));
  x_raw[3] = static_cast<float>(log10(st->second_derivative / range + 1e-12));
  for (int i = 0; i < 4; ++i) {
    x_std[i] = (x_raw[i] - p[d.xm_off + i]) / p[d.xs_off + i];
  }
}

/**
 * All layers, block-wide: activations ping-pong between two shared buffers.
 * @return the buffer holding the (still standardised) outputs
 */
__device__ const float *RunLayers(const float *p, const NetDesc &d,
                                  float (*buf)[kMaxWidth]) {
  int in = 0;
  for (int l = 0; l < d.n_layers; ++l) {
    const int n_in = d.dims[l];
    const int n_out = d.dims[l + 1];
    const float *w = p + d.w_off[l];
    const float *b = p + d.b_off[l];
    for (int o = threadIdx.x; o < n_out; o += blockDim.x) {
      float s = b[o];
      for (int i = 0; i < n_in; ++i) s += w[o * n_in + i] * buf[in][i];
      buf[1 - in][o] = (l + 1 < d.n_layers) ? fmaxf(0.0f, s) : s;
    }
    __syncthreads();
    in = 1 - in;
  }
  return buf[in];
}

/** Single block: inputs, network, per-setting outcome and cost, ordering. */
__global__ void RankKernel(const float *p, NetDesc d,
                           const unsigned char *available,
                           const ctp::DeviceFeatureStats *st, RankArgs a,
                           RankOut *out) {
  __shared__ float buf[2][kMaxWidth];
  __shared__ double cost[kMaxSettings];
  if (threadIdx.x == 0) BuildInputs(p, d, st, a.chunk_bytes, out->x, buf[0]);
  __syncthreads();
  const float *y = RunLayers(p, d, buf);
  for (int s = threadIdx.x; s < d.n_settings; s += blockDim.x) {
    float v[3];
    for (int k = 0; k < 3; ++k) {
      const int o = 3 * s + k;
      v[k] = expf(y[o] * p[d.ys_off + o] + p[d.ym_off + o]);
    }
    out->ct[s] = v[0];
    out->dt[s] = v[1];
    out->ratio[s] = v[2];
    double c = a.w_ct * v[0] + a.w_dt * v[1] +
               a.w_io * a.chunk_bytes / (static_cast<double>(v[2]) *
                                         a.bw_bytes_per_ms);
    if (!available[s] || !isfinite(c) || !(v[2] > 0.0f)) c = CUDART_INF;
    cost[s] = c;
    out->cost[s] = c;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Insertion sort, cheapest first; ties keep the lower index first.
    for (int s = 0; s < d.n_settings; ++s) {
      int j = s;
      while (j > 0 && cost[out->order[j - 1]] > cost[s]) {
        out->order[j] = out->order[j - 1];
        --j;
      }
      out->order[j] = s;
    }
  }
}

/** @return this thread's device copy of a RankOut (allocated once). */
RankOut *ThreadOut() {
  static thread_local RankOut *d_out = nullptr;
  if (d_out == nullptr && cudaMalloc(&d_out, sizeof(RankOut)) != cudaSuccess) {
    d_out = nullptr;
  }
  return d_out;
}

}  // namespace

bool Upload(const void *host, size_t bytes, void **dev) {
  *dev = nullptr;
  if (cudaMalloc(dev, bytes) != cudaSuccess) return false;
  if (cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(*dev);
    *dev = nullptr;
    return false;
  }
  return true;
}

bool CopyToDevice(void *dev, size_t offset_bytes, const void *host,
                  size_t bytes) {
  return cudaMemcpy(static_cast<char *>(dev) + offset_bytes, host, bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

void FreeDevice(void *dev) {
  if (dev != nullptr) cudaFree(dev);
}

bool RankOnDevice(const float *d_params, const NetDesc &desc,
                  const unsigned char *d_available, const void *device_stats,
                  const RankArgs &args, void *stream, RankOut *out) {
  if (desc.n_settings > kMaxSettings || desc.n_layers > kMaxLayers) {
    return false;
  }
  for (int l = 0; l <= desc.n_layers; ++l) {
    if (desc.dims[l] > kMaxWidth) return false;
  }
  RankOut *d_out = ThreadOut();
  if (d_out == nullptr || device_stats == nullptr) return false;
  auto s = static_cast<cudaStream_t>(stream);
  RankKernel<<<1, kThreads, 0, s>>>(
      d_params, desc, d_available,
      static_cast<const ctp::DeviceFeatureStats *>(device_stats), args, d_out);
  if (cudaGetLastError() != cudaSuccess) return false;
  if (cudaMemcpyAsync(out, d_out, sizeof(RankOut), cudaMemcpyDeviceToHost,
                      s) != cudaSuccess) {
    return false;
  }
  return cudaStreamSynchronize(s) == cudaSuccess;
}

}  // namespace ctp::compress::model::v2
