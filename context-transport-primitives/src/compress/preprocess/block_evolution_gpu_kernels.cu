/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file block_evolution_gpu_kernels.cu
 * @brief Per-block temporal evolution on device-resident buffers.
 *
 * Deliberately shaped like data_stats_gpu_kernels.cu next door: one fused
 * grid-stride reduction, per-thread leaked scratch so the per-chunk path has
 * no allocator traffic, and a one-thread finalize kernel so the square roots
 * and the division happen on device rather than in a host expression built
 * from three separately copied scalars.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include "clio_ctp/compress/preprocess/block_evolution_gpu.h"

namespace ctp {
namespace {

constexpr int kBlockSize = 256;
/** Same grid ceiling data_stats_gpu_kernels.cu uses: enough blocks to fill
    the device, few enough that the final atomicAdd fan-in stays small. */
constexpr int kMaxGrid = 1024;

/** Device-side POD result. Mirrors the public struct's numeric fields; the
    status is decided on the host, from these counts. */
struct DeviceBlockEvolution {
  double absolute_change;
  double normalized_change;
  double b1_norm;
  double b2_norm;
  unsigned long long elements_compared;
  unsigned long long nonfinite_skipped;
};

/**
 * One pass, three sums.
 *
 *   sums[0] = sum (b2-b1)^2      sums[1] = sum b1^2      sums[2] = sum b2^2
 *
 * Both elements are read once and all three accumulators updated from
 * registers. Splitting this into three kernels would triple the memory
 * traffic of a reduction that is entirely bandwidth-bound.
 *
 * NaN/Inf: a pair is skipped when EITHER side is non-finite, and counted.
 * Skipping is not cosmetic -- a single NaN admitted into any accumulator
 * makes all three sums NaN, so the block's evolution, and every comparison
 * against it, would be lost rather than degraded. The count travels with the
 * result so a caller can see the norms are partial.
 *
 * Accumulation is double regardless of input type. For float32 input at
 * 61,440 elements a float accumulator loses roughly the last 3 decimal
 * digits of the sum of squares, which is the same order as the differences
 * this metric is meant to resolve between quiet blocks.
 */
template <typename T>
__global__ void BlockEvolutionSumsKernel(const T *__restrict__ b1,
                                          const T *__restrict__ b2,
                                          size_t num_elements, double *sums,
                                          unsigned long long *counts) {
  __shared__ double s_d2[kBlockSize];
  __shared__ double s_n1[kBlockSize];
  __shared__ double s_n2[kBlockSize];
  __shared__ unsigned long long s_bad[kBlockSize];

  double t_d2 = 0.0, t_n1 = 0.0, t_n2 = 0.0;
  unsigned long long t_bad = 0;

  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_elements; i += stride) {
    const double x1 = static_cast<double>(b1[i]);
    const double x2 = static_cast<double>(b2[i]);
    if (!isfinite(x1) || !isfinite(x2)) {
      ++t_bad;
      continue;
    }
    const double d = x2 - x1;
    t_d2 += d * d;
    t_n1 += x1 * x1;
    t_n2 += x2 * x2;
  }

  s_d2[threadIdx.x] = t_d2;
  s_n1[threadIdx.x] = t_n1;
  s_n2[threadIdx.x] = t_n2;
  s_bad[threadIdx.x] = t_bad;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_d2[threadIdx.x] += s_d2[threadIdx.x + s];
      s_n1[threadIdx.x] += s_n1[threadIdx.x + s];
      s_n2[threadIdx.x] += s_n2[threadIdx.x + s];
      s_bad[threadIdx.x] += s_bad[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&sums[0], s_d2[0]);
    atomicAdd(&sums[1], s_n1[0]);
    atomicAdd(&sums[2], s_n2[0]);
    atomicAdd(counts, s_bad[0]);
  }
}

/**
 * Square roots and the normalized ratio, on device.
 *
 * One thread: this is 3 sqrts and a divide. It runs here rather than on the
 * host purely so the result crosses PCIe once, as a finished struct, instead
 * of as three raw sums the host then has to combine -- the same reason
 * FinalizeFeatureStatsKernel exists in data_stats_gpu_kernels.cu.
 *
 * All-zero blocks land here as 0/(0+0+eps) = 0, which is the correct reading:
 * a block that was zero and stayed zero did not evolve.
 */
__global__ void FinalizeBlockEvolutionKernel(const double *sums,
                                             const unsigned long long *counts,
                                             size_t num_elements,
                                             double epsilon,
                                             DeviceBlockEvolution *out) {
  const double d = sqrt(sums[0]);
  const double n1 = sqrt(sums[1]);
  const double n2 = sqrt(sums[2]);
  out->absolute_change = d;
  out->b1_norm = n1;
  out->b2_norm = n2;
  out->normalized_change = d / (n1 + n2 + epsilon);
  out->nonfinite_skipped = *counts;
  out->elements_compared = num_elements - *counts;
}

/**
 * Per-thread scratch: stream + the three sums + the bad counter + the result.
 *
 * Leaked on purpose, exactly as DeviceStatsScratch is next door: freeing a
 * device allocation from a thread-exit or static destructor races the CUDA
 * runtime's own teardown. Total is 64 bytes per thread that ever measures.
 */
struct EvoScratch {
  cudaStream_t stream = nullptr;
  double *d_sums = nullptr;
  unsigned long long *d_counts = nullptr;
  DeviceBlockEvolution *d_out = nullptr;
  bool ok = false;
};

EvoScratch &Scratch() {
  static thread_local EvoScratch *s = [] {
    auto *p = new EvoScratch();
    p->ok = cudaStreamCreate(&p->stream) == cudaSuccess &&
            cudaMalloc(&p->d_sums, 3 * sizeof(double)) == cudaSuccess &&
            cudaMalloc(&p->d_counts, sizeof(unsigned long long)) ==
                cudaSuccess &&
            cudaMalloc(&p->d_out, sizeof(DeviceBlockEvolution)) == cudaSuccess;
    return p;
  }();
  return *s;
}

template <typename T>
bool LaunchTyped(const T *b1, const T *b2, size_t num_elements,
                 double epsilon, EvoScratch &s, cudaStream_t stream) {
  if (cudaMemsetAsync(s.d_sums, 0, 3 * sizeof(double), stream) != cudaSuccess ||
      cudaMemsetAsync(s.d_counts, 0, sizeof(unsigned long long), stream) !=
          cudaSuccess) {
    return false;
  }
  int grid = static_cast<int>(std::min<size_t>(
      (num_elements + kBlockSize - 1) / kBlockSize, kMaxGrid));
  if (grid < 1) grid = 1;

  BlockEvolutionSumsKernel<T><<<grid, kBlockSize, 0, stream>>>(
      b1, b2, num_elements, s.d_sums, s.d_counts);
  FinalizeBlockEvolutionKernel<<<1, 1, 0, stream>>>(
      s.d_sums, s.d_counts, num_elements, epsilon, s.d_out);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool ComputeBlockEvolutionDevice(const void *d_prev, const void *d_curr,
                                 size_t num_elements, DataType type,
                                 double epsilon, void *stream,
                                 BlockEvolution *out) {
  if (out == nullptr) return false;
  *out = BlockEvolution{};
  if (d_prev == nullptr || d_curr == nullptr || num_elements == 0) {
    return false;
  }

  EvoScratch &s = Scratch();
  if (!s.ok) return false;
  cudaStream_t st =
      stream ? static_cast<cudaStream_t>(stream) : s.stream;

  bool launched = false;
  switch (type) {
    case DataType::FLOAT32:
      launched = LaunchTyped(static_cast<const float *>(d_prev),
                             static_cast<const float *>(d_curr), num_elements,
                             epsilon, s, st);
      break;
    case DataType::DOUBLE64:
      launched = LaunchTyped(static_cast<const double *>(d_prev),
                             static_cast<const double *>(d_curr), num_elements,
                             epsilon, s, st);
      break;
    case DataType::INT32:
      launched = LaunchTyped(static_cast<const int32_t *>(d_prev),
                             static_cast<const int32_t *>(d_curr),
                             num_elements, epsilon, s, st);
      break;
    case DataType::UINT8:
      launched = LaunchTyped(static_cast<const uint8_t *>(d_prev),
                             static_cast<const uint8_t *>(d_curr),
                             num_elements, epsilon, s, st);
      break;
    default:
      return false;
  }
  if (!launched) return false;

  // The ONLY host transfer: 48 bytes of finished result. The blocks
  // themselves -- 61,440 B each on a LAMMPS chunk, megabytes on a Nyx fab --
  // never move.
  DeviceBlockEvolution host{};
  if (cudaMemcpyAsync(&host, s.d_out, sizeof(host), cudaMemcpyDeviceToHost,
                      st) != cudaSuccess ||
      cudaStreamSynchronize(st) != cudaSuccess) {
    return false;
  }

  out->absolute_change = host.absolute_change;
  out->normalized_change = host.normalized_change;
  out->b1_norm = host.b1_norm;
  out->b2_norm = host.b2_norm;
  out->elements_compared = host.elements_compared;
  out->nonfinite_skipped = host.nonfinite_skipped;
  out->status = (host.elements_compared == 0)
                    ? BlockEvolutionStatus::kAllNonFinite
                    : BlockEvolutionStatus::kOk;
  return out->status == BlockEvolutionStatus::kOk;
}

namespace detail {

void *EvoDeviceAlloc(size_t bytes) {
  void *p = nullptr;
  if (bytes == 0 || cudaMalloc(&p, bytes) != cudaSuccess) {
    // Clear the sticky error: a failed retention must not be misattributed to
    // the next unrelated CUDA call the simulation makes.
    (void)cudaGetLastError();
    return nullptr;
  }
  return p;
}

void EvoDeviceFree(void *ptr) {
  if (ptr) cudaFree(ptr);
}

bool EvoDeviceCopyAsync(void *dst, const void *src, size_t bytes,
                        void *stream) {
  if (dst == nullptr || src == nullptr || bytes == 0) return false;
  EvoScratch &s = Scratch();
  cudaStream_t st = stream ? static_cast<cudaStream_t>(stream) : s.stream;
  // cudaMemcpyDefault, not DeviceToDevice: the tracker retains host chunks
  // through this same call on a build where the source may be either.
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, st) ==
         cudaSuccess;
}

bool EvoDeviceSync(void *stream) {
  EvoScratch &s = Scratch();
  cudaStream_t st = stream ? static_cast<cudaStream_t>(stream) : s.stream;
  return cudaStreamSynchronize(st) == cudaSuccess;
}

}  // namespace detail
}  // namespace ctp
