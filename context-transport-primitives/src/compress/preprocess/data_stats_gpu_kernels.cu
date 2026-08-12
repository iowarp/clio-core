/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_ctp/compress/preprocess/data_stats_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ctp {
namespace {

constexpr int kHistBins = 256;
constexpr int kBlockSize = 256;

/**
 * Pass 1: byte histogram (for entropy) + typed value sum (for mean, needed
 * by pass 2's MAD) + abs second-derivative sum, all in one grid-stride pass
 * over the buffer. Independent per-block partial results are combined with
 * atomics -- this buffer is a feature-extraction sample (tens of KB), not a
 * hot loop that needs a tree reduction.
 */
template <typename T>
__global__ void StatsPass1Kernel(const T *data, size_t num_elements,
                                  unsigned int *histogram, double *sum_out,
                                  double *sum_abs_d2_out) {
  __shared__ unsigned int local_hist[kHistBins];
  for (int i = threadIdx.x; i < kHistBins; i += blockDim.x) local_hist[i] = 0;
  __shared__ double block_sum[kBlockSize];
  __shared__ double block_d2[kBlockSize];
  double thread_sum = 0.0;
  double thread_d2 = 0.0;
  __syncthreads();

  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
  size_t num_bytes = num_elements * sizeof(T);
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_bytes;
       i += stride) {
    atomicAdd(&local_hist[bytes[i]], 1u);
  }

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
       i += stride) {
    thread_sum += static_cast<double>(data[i]);
  }

  // Second derivative: d2[i] = data[i+1] - 2*data[i] + data[i-1], i in
  // [1, num_elements-2] -- matches DataStatistics<T>::CalculateSecondDerivative.
  if (num_elements >= 3) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
         i + 1 < num_elements; i += stride) {
      double d2 = static_cast<double>(data[i + 1]) -
                  2.0 * static_cast<double>(data[i]) +
                  static_cast<double>(data[i - 1]);
      thread_d2 += fabs(d2);
    }
  }

  block_sum[threadIdx.x] = thread_sum;
  block_d2[threadIdx.x] = thread_d2;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
      block_d2[threadIdx.x] += block_d2[threadIdx.x + s];
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < kHistBins; i += blockDim.x) {
    if (local_hist[i]) atomicAdd(&histogram[i], local_hist[i]);
  }
  if (threadIdx.x == 0) {
    atomicAdd(sum_out, block_sum[0]);
    atomicAdd(sum_abs_d2_out, block_d2[0]);
  }
}

/** Pass 2: sum of |x - mean|, once the mean from pass 1 is known. */
template <typename T>
__global__ void StatsPass2Kernel(const T *data, size_t num_elements,
                                  double mean, double *sum_abs_dev_out) {
  __shared__ double block_sum[kBlockSize];
  double thread_sum = 0.0;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
       i += stride) {
    thread_sum += fabs(static_cast<double>(data[i]) - mean);
  }
  block_sum[threadIdx.x] = thread_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(sum_abs_dev_out, block_sum[0]);
}

template <typename T>
bool ComputeDeviceStatsTyped(const T *data, size_t num_elements,
                              double *out_entropy, double *out_mad,
                              double *out_second_derivative) {
  if (num_elements == 0) {
    *out_entropy = 0.0;
    *out_mad = 0.0;
    *out_second_derivative = 0.0;
    return true;
  }

  unsigned int *d_hist = nullptr;
  double *d_scalars = nullptr;  // [sum, sum_abs_d2, sum_abs_dev]

  // Every CUDA step is checked. Returning true on failure used to hand the
  // caller an entropy computed from an UNINITIALIZED stack histogram with
  // mad = 0 -- i.e. the "perfectly compressible" corner of the feature
  // space -- for a chunk nothing is known about. NeuroPress propagates the
  // failure at every stage (stats_kernel.cu:307-354 returns nullptr, and
  // gpucompress_compress.cpp:285 bails on it); ComputeCompressionFeatures's
  // contract already promises the same, so it must actually be able to fail.
  bool ok = cudaMalloc(&d_hist, kHistBins * sizeof(unsigned int)) ==
                cudaSuccess &&
            cudaMemset(d_hist, 0, kHistBins * sizeof(unsigned int)) ==
                cudaSuccess &&
            cudaMalloc(&d_scalars, 3 * sizeof(double)) == cudaSuccess &&
            cudaMemset(d_scalars, 0, 3 * sizeof(double)) == cudaSuccess;

  int grid = static_cast<int>(std::min<size_t>(
      (num_elements + kBlockSize - 1) / kBlockSize, 1024));
  if (grid < 1) grid = 1;

  double h_sum_and_d2[2] = {0.0, 0.0};
  unsigned int h_hist[kHistBins] = {0};
  double h_sum_abs_dev = 0.0;
  double mean = 0.0;

  if (ok) {
    StatsPass1Kernel<T><<<grid, kBlockSize>>>(data, num_elements, d_hist,
                                               d_scalars, d_scalars + 1);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaMemcpy(h_sum_and_d2, d_scalars, 2 * sizeof(double),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  if (ok) {
    mean = h_sum_and_d2[0] / static_cast<double>(num_elements);
    StatsPass2Kernel<T>
        <<<grid, kBlockSize>>>(data, num_elements, mean, d_scalars + 2);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaMemcpy(h_hist, d_hist, kHistBins * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(&h_sum_abs_dev, d_scalars + 2, sizeof(double),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  cudaFree(d_hist);
  cudaFree(d_scalars);
  if (!ok) return false;

  size_t num_bytes = num_elements * sizeof(T);
  double entropy = 0.0;
  for (int i = 0; i < kHistBins; i++) {
    if (h_hist[i] > 0) {
      double p = static_cast<double>(h_hist[i]) / static_cast<double>(num_bytes);
      entropy += -p * std::log2(p);
    }
  }

  *out_entropy = entropy;
  *out_mad = h_sum_abs_dev / static_cast<double>(num_elements);
  *out_second_derivative = (num_elements > 2)
      ? h_sum_and_d2[1] / static_cast<double>(num_elements - 2)
      : 0.0;
  return true;
}

}  // namespace

bool ComputeDeviceStats(const void *device_data, size_t num_elements,
                         DataType type, double *out_entropy, double *out_mad,
                         double *out_second_derivative) {
  switch (type) {
    case DataType::UINT8:
      return ComputeDeviceStatsTyped<uint8_t>(
          static_cast<const uint8_t *>(device_data), num_elements,
          out_entropy, out_mad, out_second_derivative);
    case DataType::INT32:
      return ComputeDeviceStatsTyped<int32_t>(
          static_cast<const int32_t *>(device_data), num_elements,
          out_entropy, out_mad, out_second_derivative);
    case DataType::FLOAT32:
      return ComputeDeviceStatsTyped<float>(
          static_cast<const float *>(device_data), num_elements, out_entropy,
          out_mad, out_second_derivative);
    case DataType::DOUBLE64:
      return ComputeDeviceStatsTyped<double>(
          static_cast<const double *>(device_data), num_elements,
          out_entropy, out_mad, out_second_derivative);
    default:
      return false;
  }
}

}  // namespace ctp


// ===========================================================================
// Device byte-shuffle / unshuffle (issue #693).
//
// Lives in this translation unit rather than its own: adding a second
// separately-device-linked .cu to this RDC-enabled static library made
// __cudaRegisterLinkedBinary segfault during static init, before main().
// Same registration/RDC interaction that broke the demo target earlier.
// ===========================================================================
#include "clio_ctp/compress/preprocess/byte_shuffle.h"  // kShuffleChunkBytes

namespace ctp::compress::preprocess {

namespace {

/**
 * Byte planes are built WITHIN each kShuffleChunkBytes block, never across
 * the whole buffer -- NeuroPress splits the input first
 * (byte_shuffle_kernels.cu's createDeviceChunkArrays) and each of its blocks
 * computes `num_elements = chunk_size / ElementSize` for its OWN chunk. A
 * global plane layout produces different bytes for anything above 256 KiB.
 *
 * One thread block per chunk, grid-strided so a large buffer does not need
 * one block per chunk resident at once. Within a chunk the write side is
 * coalesced (out[b*n + elem] is contiguous across threads) at the cost of a
 * strided read -- the same trade upstream makes.
 */
template <unsigned ElemSize>
__global__ void ShuffleKernel(const uint8_t *__restrict__ in,
                              uint8_t *__restrict__ out, size_t num_bytes,
                              size_t chunk_bytes) {
  const size_t num_chunks = (num_bytes + chunk_bytes - 1) / chunk_bytes;
  for (size_t c = blockIdx.x; c < num_chunks; c += gridDim.x) {
    const size_t base = c * chunk_bytes;
    const size_t remain = num_bytes - base;
    const size_t chunk = remain < chunk_bytes ? remain : chunk_bytes;
    const size_t n = chunk / ElemSize;
    const uint8_t *ci = in + base;
    uint8_t *co = out + base;
    for (size_t elem = threadIdx.x; elem < n; elem += blockDim.x) {
#pragma unroll
      for (unsigned b = 0; b < ElemSize; ++b) {
        co[b * n + elem] = ci[elem * ElemSize + b];
      }
    }
    // Trailing partial element of THIS chunk, copied verbatim
    // (byte_shuffle_kernels.cu:59-65). Only the last chunk can have one,
    // since kShuffleChunkBytes is a multiple of every supported ElemSize.
    for (size_t i = threadIdx.x + n * ElemSize; i < chunk; i += blockDim.x) {
      co[i] = ci[i];
    }
  }
}

template <unsigned ElemSize>
__global__ void UnshuffleKernel(const uint8_t *__restrict__ in,
                                uint8_t *__restrict__ out, size_t num_bytes,
                                size_t chunk_bytes) {
  const size_t num_chunks = (num_bytes + chunk_bytes - 1) / chunk_bytes;
  for (size_t c = blockIdx.x; c < num_chunks; c += gridDim.x) {
    const size_t base = c * chunk_bytes;
    const size_t remain = num_bytes - base;
    const size_t chunk = remain < chunk_bytes ? remain : chunk_bytes;
    const size_t n = chunk / ElemSize;
    const uint8_t *ci = in + base;
    uint8_t *co = out + base;
    for (size_t elem = threadIdx.x; elem < n; elem += blockDim.x) {
#pragma unroll
      for (unsigned b = 0; b < ElemSize; ++b) {
        co[elem * ElemSize + b] = ci[b * n + elem];
      }
    }
    for (size_t i = threadIdx.x + n * ElemSize; i < chunk; i += blockDim.x) {
      co[i] = ci[i];
    }
  }
}

/** Shared validation + launch geometry for both directions. Returns false
 *  if the request is not shuffleable; otherwise fills the launch config. */
bool PrepareLaunch(const uint8_t *in, uint8_t *out, size_t num_bytes,
                   size_t elem_size, int *blocks, int *threads) {
  if (!in || !out || num_bytes == 0) return false;
  if (elem_size != 2 && elem_size != 4 && elem_size != 8) return false;
  if (num_bytes < elem_size) return false;
  *threads = 256;
  const size_t num_chunks =
      (num_bytes + kShuffleChunkBytes - 1) / kShuffleChunkBytes;
  *blocks = static_cast<int>(num_chunks > 65535 ? 65535 : num_chunks);
  return true;
}

bool FinishLaunch(cudaStream_t stream) {
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

}  // namespace

bool ByteShuffleDevice(const void *device_in, void *device_out,
                       size_t num_bytes, size_t elem_size) {
  const uint8_t *in = static_cast<const uint8_t *>(device_in);
  uint8_t *out = static_cast<uint8_t *>(device_out);
  int blocks = 0, threads = 0;
  if (!PrepareLaunch(in, out, num_bytes, elem_size, &blocks, &threads)) {
    return false;
  }
  cudaStream_t stream = 0;
  const size_t cb = kShuffleChunkBytes;
  if (elem_size == 2) {
    ShuffleKernel<2><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  } else if (elem_size == 4) {
    ShuffleKernel<4><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  } else {
    ShuffleKernel<8><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  }
  return FinishLaunch(stream);
}

bool ByteUnshuffleDevice(const void *device_in, void *device_out,
                         size_t num_bytes, size_t elem_size) {
  const uint8_t *in = static_cast<const uint8_t *>(device_in);
  uint8_t *out = static_cast<uint8_t *>(device_out);
  int blocks = 0, threads = 0;
  if (!PrepareLaunch(in, out, num_bytes, elem_size, &blocks, &threads)) {
    return false;
  }
  cudaStream_t stream = 0;
  const size_t cb = kShuffleChunkBytes;
  if (elem_size == 2) {
    UnshuffleKernel<2><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  } else if (elem_size == 4) {
    UnshuffleKernel<4><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  } else {
    UnshuffleKernel<8><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  }
  return FinishLaunch(stream);
}

}  // namespace ctp::compress::preprocess
