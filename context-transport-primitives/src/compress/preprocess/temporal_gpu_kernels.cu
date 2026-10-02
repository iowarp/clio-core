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
 * @file temporal_gpu_kernels.cu
 * @brief CUDA implementation of the temporal prediction primitives declared
 *        in clio_ctp/compress/preprocess/temporal.h.
 *
 * Each reconstruction formula has exactly one implementation (the prediction
 * in PredictKernel, the sum pred + dec in AddKernel / BoundCheckKernel), used
 * by both the encoder and the decoder, so the two round identically.
 */

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "clio_ctp/compress/preprocess/temporal.h"

namespace ctp::compress::preprocess {

namespace {

constexpr int kThreads = 256;

/** Grid size for a grid-stride kernel over n elements. */
int Blocks(size_t n) {
  const size_t b = (n + kThreads - 1) / kThreads;
  return static_cast<int>(b > 65535u * 8u ? 65535u * 8u : (b ? b : 1));
}

/** Launch check plus a sync of the caller's stream. */
bool Finish(cudaStream_t s) {
  return cudaGetLastError() == cudaSuccess &&
         cudaStreamSynchronize(s) == cudaSuccess;
}

__global__ void PredictKernel(const float *lo, const float *hi, float w,
                              float *out, size_t n) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    out[i] = lo[i] + w * (hi[i] - lo[i]);
  }
}

__global__ void AddKernel(const float *a, const float *b, float sign,
                          float *out, size_t n) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    out[i] = sign > 0.0f ? a[i] + b[i] : a[i] - b[i];
  }
}

__global__ void BoundCheckKernel(const float *x, const float *pred,
                                 const float *dec, float ebd, float *recon,
                                 uint32_t *esc_idx, float *esc_val,
                                 unsigned long long *count, size_t n) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    float r = pred != nullptr ? pred[i] + dec[i] : dec[i];
    if (!(fabsf(r - x[i]) < ebd)) {  // also true for NaN
      r = x[i];
      const unsigned long long p = atomicAdd(count, 1ULL);
      esc_idx[p] = static_cast<uint32_t>(i);
      esc_val[p] = x[i];
    }
    recon[i] = r;
  }
}

__global__ void ScatterKernel(const uint32_t *idx, const float *val,
                              uint64_t m, float *out) {
  for (uint64_t j = blockIdx.x * static_cast<uint64_t>(blockDim.x) +
                    threadIdx.x;
       j < m; j += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    out[idx[j]] = val[j];
  }
}

/** log2(2|q| + 1) bits for leftover r at quantization step 2 eb. */
__device__ double CodeBits(double r, double inv_step) {
  const double q = fabs(rint(r * inv_step));
  return log2(2.0 * q + 1.0);
}

/** 3-D Lorenzo prediction of m at (x, y, z), zero outside the grid. */
__device__ double Lorenzo(const float *m, size_t x, size_t y, size_t z,
                          size_t nx, size_t ny) {
  auto at = [&](size_t dz, size_t dy, size_t dx) -> double {
    if (x < dx || y < dy || z < dz) return 0.0;
    return m[((z - dz) * ny + (y - dy)) * nx + (x - dx)];
  };
  return at(0, 0, 1) + at(0, 1, 0) + at(1, 0, 0) - at(0, 1, 1) -
         at(1, 0, 1) - at(1, 1, 0) + at(1, 1, 1);
}

/** Accumulates [temporal bits, spatial bits, temporal zeros, spatial zeros]. */
__global__ void ProbeKernel(const float *prev, const float *mid,
                            const float *next, size_t nx, size_t ny,
                            size_t nz, double inv_step, double *acc) {
  double s[4] = {0.0, 0.0, 0.0, 0.0};
  const size_t n = nx * ny * nz;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    const double rt = mid[i] - 0.5 * (static_cast<double>(prev[i]) + next[i]);
    const double rs = mid[i] - Lorenzo(mid, x, y, z, nx, ny);
    const double bt = CodeBits(rt, inv_step), bs = CodeBits(rs, inv_step);
    s[0] += bt;
    s[1] += bs;
    s[2] += bt == 0.0;
    s[3] += bs == 0.0;
  }
  for (int k = 0; k < 4; ++k) {
    for (int off = 16; off > 0; off >>= 1) {
      s[k] += __shfl_down_sync(0xffffffffu, s[k], off);
    }
    if ((threadIdx.x & 31) == 0) atomicAdd(&acc[k], s[k]);
  }
}

/** One device counter per host thread, allocated on first use. */
unsigned long long *ThreadCounter() {
  thread_local unsigned long long *counter = nullptr;
  if (counter == nullptr &&
      cudaMalloc(&counter, sizeof(unsigned long long)) != cudaSuccess) {
    counter = nullptr;
  }
  return counter;
}

}  // namespace

bool TemporalPredictDevice(const float *lo, const float *hi, float w,
                           float *out, size_t n, void *stream) {
  auto s = static_cast<cudaStream_t>(stream);
  PredictKernel<<<Blocks(n), kThreads, 0, s>>>(lo, hi, w, out, n);
  return Finish(s);
}

bool AddDevice(const float *a, const float *b, float sign, float *out,
               size_t n, void *stream) {
  auto s = static_cast<cudaStream_t>(stream);
  AddKernel<<<Blocks(n), kThreads, 0, s>>>(a, b, sign, out, n);
  return Finish(s);
}

bool BoundCheckDevice(const float *x, const float *pred, const float *dec,
                      double eb, float *recon, uint32_t *esc_idx,
                      float *esc_val, uint64_t *esc_count, size_t n,
                      void *stream) {
  if (n >= (static_cast<size_t>(1) << 32) || esc_count == nullptr) {
    return false;
  }
  unsigned long long *counter = ThreadCounter();
  if (counter == nullptr) return false;
  auto s = static_cast<cudaStream_t>(stream);
  float ebd = static_cast<float>(eb);  // largest float32 not above eb
  if (static_cast<double>(ebd) > eb) ebd = nextafterf(ebd, 0.0f);
  if (cudaMemsetAsync(counter, 0, sizeof(*counter), s) != cudaSuccess) {
    return false;
  }
  BoundCheckKernel<<<Blocks(n), kThreads, 0, s>>>(
      x, pred, dec, ebd, recon, esc_idx, esc_val, counter, n);
  unsigned long long host = 0;
  if (cudaMemcpyAsync(&host, counter, sizeof(host), cudaMemcpyDeviceToHost,
                      s) != cudaSuccess ||
      !Finish(s)) {
    return false;
  }
  *esc_count = host;
  return true;
}

bool ScatterDevice(const uint32_t *idx, const float *val, uint64_t m,
                   float *out, void *stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (m == 0) return true;
  ScatterKernel<<<Blocks(m), kThreads, 0, s>>>(idx, val, m, out);
  return Finish(s);
}

bool TemporalProbeDevice(const float *prev, const float *mid,
                         const float *next, size_t nx, size_t ny, size_t nz,
                         double eb, TemporalProbeResult *out, void *stream) {
  const size_t n = nx * ny * nz;
  if (n == 0 || out == nullptr || !(eb > 0.0)) return false;
  thread_local double *acc = nullptr;  // four accumulators per host thread
  if (acc == nullptr && cudaMalloc(&acc, 4 * sizeof(double)) != cudaSuccess) {
    acc = nullptr;
    return false;
  }
  auto s = static_cast<cudaStream_t>(stream);
  if (cudaMemsetAsync(acc, 0, 4 * sizeof(double), s) != cudaSuccess) {
    return false;
  }
  ProbeKernel<<<Blocks(n), kThreads, 0, s>>>(prev, mid, next, nx, ny, nz,
                                             0.5 / eb, acc);
  double host[4] = {0.0, 0.0, 0.0, 0.0};
  if (cudaMemcpyAsync(host, acc, sizeof(host), cudaMemcpyDeviceToHost, s) !=
          cudaSuccess ||
      !Finish(s)) {
    return false;
  }
  const double inv_n = 1.0 / static_cast<double>(n);
  out->temporal_bits = host[0] * inv_n;
  out->spatial_bits = host[1] * inv_n;
  out->temporal_zero = host[2] * inv_n;
  out->spatial_zero = host[3] * inv_n;
  return true;
}

}  // namespace ctp::compress::preprocess
