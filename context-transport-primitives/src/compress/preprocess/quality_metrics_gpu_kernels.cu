/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file quality_metrics_gpu_kernels.cu
 * @brief Device reduction for the measured quality metrics.
 *
 * A restatement of upstream NeuroPress's traceQualityKernel
 * (src/hdf5/H5VLgpucompress.cu), kept structurally identical -- float
 * accumulators, [9][256] shared tile, grid-stride loop, block reduction, then
 * cross-block atomics -- so a parity test compares two implementations of the
 * same algorithm rather than two different algorithms.
 *
 * Derivation is NOT duplicated here: the nine sums go back to the host and
 * through QualityFromAccumulators, the same function the host reference uses.
 */

#include <cuda_runtime.h>

#include <cstddef>

#include "clio_ctp/compress/preprocess/quality_metrics.h"
#include "clio_ctp/compress/preprocess/quality_metrics_gpu.h"
#include "clio_ctp/util/gpu_api.h"

namespace ctp::compress::preprocess {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 256;   // upstream's cap; keeps the atomic fan-in small

/* float atomics for min/max via CAS -- CUDA has no atomicMax(float*). */
__device__ static void AtomicMaxF(float *addr, float val) {
  int *ai = reinterpret_cast<int *>(addr);
  int assumed, old = *ai;
  do {
    assumed = old;
    if (__int_as_float(assumed) >= val) return;
    old = atomicCAS(ai, assumed, __float_as_int(val));
  } while (assumed != old);
}
__device__ static void AtomicMinF(float *addr, float val) {
  int *ai = reinterpret_cast<int *>(addr);
  int assumed, old = *ai;
  do {
    assumed = old;
    if (__int_as_float(assumed) <= val) return;
    old = atomicCAS(ai, assumed, __float_as_int(val));
  } while (assumed != old);
}

/**
 * @param shift origin for slots 4-8. Subtracting it before squaring is what
 *   keeps the variance computable in float: E[x^2]-E[x]^2 on unshifted data
 *   cancels to zero for a low-variance field, which is how upstream reports
 *   SSIM exactly 1.0 on a constant field carrying real noise.
 */
__global__ void QualityKernel(const float *__restrict__ d_orig,
                              const float *__restrict__ d_decoded, int n,
                              float shift, float *__restrict__ out) {
  __shared__ float s[9][kThreads];
  const int t = threadIdx.x;
  // Slots 2 and 3 seed to +/-FLT_MAX so a min/max over an empty stride is
  // neutral; the other seven seed to zero.
  float v[9] = {0.f, 0.f, 3.4e38f, -3.4e38f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int i = blockIdx.x * blockDim.x + t; i < n;
       i += gridDim.x * blockDim.x) {
    const float x = d_orig[i], y = d_decoded[i], d = x - y;
    const float dx = x - shift, dy = y - shift;
    v[0] += d * d;
    v[1] = fmaxf(v[1], fabsf(d));
    v[2] = fminf(v[2], x);
    v[3] = fmaxf(v[3], x);
    v[4] += dx;
    v[5] += dx * dx;
    v[6] += dy;
    v[7] += dy * dy;
    v[8] += dx * dy;
  }
  for (int k = 0; k < 9; ++k) s[k][t] = v[k];
  __syncthreads();
  for (int half = blockDim.x / 2; half > 0; half >>= 1) {
    if (t < half) {
      s[0][t] += s[0][t + half];
      s[1][t] = fmaxf(s[1][t], s[1][t + half]);
      s[2][t] = fminf(s[2][t], s[2][t + half]);
      s[3][t] = fmaxf(s[3][t], s[3][t + half]);
      for (int k = 4; k < 9; ++k) s[k][t] += s[k][t + half];
    }
    __syncthreads();
  }
  if (t == 0) {
    atomicAdd(&out[0], s[0][0]);
    AtomicMaxF(&out[1], s[1][0]);
    AtomicMinF(&out[2], s[2][0]);
    AtomicMaxF(&out[3], s[3][0]);
    for (int k = 4; k < 9; ++k) atomicAdd(&out[k], s[k][0]);
  }
}

}  // namespace

bool ComputeQualityDevice(const void *d_orig, const void *d_decoded,
                          std::size_t n, void *stream, QualityMetrics *out) {
  if (!d_orig || !d_decoded || !out || n == 0) return false;
  // REFUSE HOST MEMORY. There is no staging path and no host implementation
  // on purpose: a quality number computed off the CPU reads identically in
  // the output to one computed on the GPU, so a fallback here would be
  // invisible. Same contract as CLIO_NEUROPRESS_REQUIRE_DEVICE.
  if (!ctp::IsDevicePointer(d_orig) || !ctp::IsDevicePointer(d_decoded)) {
    return false;
  }
  // The kernel indexes with int, as upstream's does. Refuse rather than
  // silently measuring a prefix.
  if (n > static_cast<std::size_t>(0x7FFFFFFF)) return false;

  cudaStream_t s = static_cast<cudaStream_t>(stream);
  float *d_acc = nullptr;
  if (cudaMalloc(&d_acc, 9 * sizeof(float)) != cudaSuccess) return false;

  // The shift is the first original element: one 4-byte read, and any value
  // inside the data conditions the subtraction equally well.
  float shift = 0.f;
  bool ok = cudaMemcpyAsync(&shift, d_orig, sizeof(float),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess &&
            cudaStreamSynchronize(s) == cudaSuccess;

  const float init[9] = {0.f, 0.f, 3.4e38f, -3.4e38f, 0.f, 0.f, 0.f, 0.f, 0.f};
  ok = ok && cudaMemcpyAsync(d_acc, init, sizeof(init), cudaMemcpyHostToDevice,
                             s) == cudaSuccess;
  if (ok) {
    const int ni = static_cast<int>(n);
    int blocks = (ni + kThreads - 1) / kThreads;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    QualityKernel<<<blocks, kThreads, 0, s>>>(
        static_cast<const float *>(d_orig),
        static_cast<const float *>(d_decoded), ni, shift, d_acc);
    ok = cudaGetLastError() == cudaSuccess;
  }
  float h[9] = {0};
  if (ok) {
    ok = cudaMemcpyAsync(h, d_acc, sizeof(h), cudaMemcpyDeviceToHost, s) ==
         cudaSuccess;
  }
  if (ok) ok = cudaStreamSynchronize(s) == cudaSuccess;
  cudaFree(d_acc);
  if (!ok) return false;

  QualityAccumulators a;
  a.sq_err = h[0];
  a.max_abs_err = h[1];
  a.min_x = h[2];
  a.max_x = h[3];
  a.sum_x = h[4];
  a.sum_xx = h[5];
  a.sum_y = h[6];
  a.sum_yy = h[7];
  a.sum_xy = h[8];
  a.shift = shift;
  *out = QualityFromAccumulators(a, n);
  return true;
}

}  // namespace ctp::compress::preprocess
