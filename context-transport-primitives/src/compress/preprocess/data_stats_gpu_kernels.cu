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
  cudaMalloc(&d_hist, kHistBins * sizeof(unsigned int));
  cudaMemset(d_hist, 0, kHistBins * sizeof(unsigned int));
  cudaMalloc(&d_scalars, 3 * sizeof(double));
  cudaMemset(d_scalars, 0, 3 * sizeof(double));

  int grid = static_cast<int>(std::min<size_t>(
      (num_elements + kBlockSize - 1) / kBlockSize, 1024));
  if (grid < 1) grid = 1;

  StatsPass1Kernel<T><<<grid, kBlockSize>>>(data, num_elements, d_hist,
                                             d_scalars, d_scalars + 1);

  double h_sum_and_d2[2] = {0.0, 0.0};
  cudaMemcpy(h_sum_and_d2, d_scalars, 2 * sizeof(double),
             cudaMemcpyDeviceToHost);
  double mean = h_sum_and_d2[0] / static_cast<double>(num_elements);

  StatsPass2Kernel<T>
      <<<grid, kBlockSize>>>(data, num_elements, mean, d_scalars + 2);

  unsigned int h_hist[kHistBins];
  double h_sum_abs_dev = 0.0;
  cudaMemcpy(h_hist, d_hist, kHistBins * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_sum_abs_dev, d_scalars + 2, sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(d_hist);
  cudaFree(d_scalars);

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
