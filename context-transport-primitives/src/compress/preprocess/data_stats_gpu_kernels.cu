/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#include "clio_ctp/compress/preprocess/data_stats_gpu.h"
#include "clio_ctp/compress/preprocess/byte_shuffle.h"  // kShuffleChunkBytes

#include <cuda_runtime.h>
#include <cstdio>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

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
  // Per-WARP privatized histograms, and a 4-bytes-at-a-time read, both taken
  // from histogramKernelVec4 (entropy_kernel.cu). That variant is not
  // an upstream curiosity: launchEntropyKernelsAsync PICKS it whenever
  // `num_bytes >= 1024 && (ptr % 4) == 0` (:238), which is every chunk this
  // path sees -- a 4 MiB float buffer is both. A single shared histogram read
  // byte-at-a-time, as this did, issues four times the memory transactions
  // and puts every thread in the block on one set of 256 shared counters.
  //
  // The COUNTS are unaffected: each byte is still counted exactly once, so
  // entropy is bit-identical either way. That is why the dataset parity
  // harness could not have caught this -- it compares numbers, and the
  // numbers were always right.
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  __shared__ unsigned int s_hist[kWarpsPerBlock][kHistBins];
  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int lane_id = static_cast<int>(threadIdx.x) % 32;
  for (int b = lane_id; b < kHistBins; b += 32) s_hist[warp_id][b] = 0;

  __shared__ double block_sum[kBlockSize];
  __shared__ double block_d2[kBlockSize];
  double thread_sum = 0.0;
  double thread_d2 = 0.0;
  __syncthreads();

  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
  size_t num_bytes = num_elements * sizeof(T);
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Same dispatch condition upstream uses, evaluated per block. It depends
  // only on the base pointer and the length, so it is uniform -- no divergence.
  if (num_bytes >= 1024 &&
      (reinterpret_cast<uintptr_t>(bytes) % 4) == 0) {
    const size_t num_words = num_bytes / 4;
    const uint32_t *data32 = reinterpret_cast<const uint32_t *>(bytes);
    for (size_t i = gid; i < num_words; i += stride) {
      const uint32_t w = data32[i];
      atomicAdd(&s_hist[warp_id][(w >> 0) & 0xFFu], 1u);
      atomicAdd(&s_hist[warp_id][(w >> 8) & 0xFFu], 1u);
      atomicAdd(&s_hist[warp_id][(w >> 16) & 0xFFu], 1u);
      atomicAdd(&s_hist[warp_id][(w >> 24) & 0xFFu], 1u);
    }
    // Trailing bytes, counted by ONE block so they are not counted per block
    // (entropy_kernel.cu).
    if (blockIdx.x == 0) {
      for (size_t i = num_words * 4 + threadIdx.x; i < num_bytes;
           i += blockDim.x) {
        atomicAdd(&s_hist[warp_id][bytes[i]], 1u);
      }
    }
  } else {
    for (size_t i = gid; i < num_bytes; i += stride) {
      atomicAdd(&s_hist[warp_id][bytes[i]], 1u);
    }
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

  // Fold the per-warp histograms together before the global atomic, so each
  // bin costs one global atomic per block rather than one per warp.
  for (int b = threadIdx.x; b < kHistBins; b += blockDim.x) {
    unsigned int total = 0;
    for (int w = 0; w < kWarpsPerBlock; ++w) total += s_hist[w][b];
    if (total) atomicAdd(&histogram[b], total);
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

/**
 * Pass 2, device-mean variant: reads the mean out of the scalar buffer on the
 * GPU instead of taking it as a host argument.
 *
 * Mirrors madPass2Kernel (stats_kernel.cu), which computes
 * `stats->sum / stats->num_elements` into a __shared__ slot once per block.
 * That one line is what lets upstream keep both passes on one stream: the
 * host-argument form forces a D2H, a host divide and a relaunch between them.
 */
template <typename T>
__global__ void StatsPass2DevKernel(const T *data, size_t num_elements,
                                     const double *__restrict__ sum_in,
                                     double *sum_abs_dev_out) {
  __shared__ double s_mean;
  if (threadIdx.x == 0) {
    s_mean = *sum_in / static_cast<double>(num_elements);
  }
  __syncthreads();
  const double mean = s_mean;

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

/**
 * Shannon entropy from the byte histogram, on the GPU.
 *
 * Structure copied from entropyFromHistogramKernel (entropy_kernel.cu):
 * one block, one bin per thread, then a shared-memory tree reduction. The
 * reduction ORDER is part of the port -- a serial host loop and a tree sum
 * over 256 doubles need not agree in the last ulp, and the whole point is to
 * land on the value upstream's kernel produces, not merely a correct one.
 */
__global__ void EntropyFromHistKernel(const unsigned int *__restrict__ histogram,
                                       size_t total_count,
                                       double *__restrict__ entropy_out) {
  __shared__ double s_partial[kBlockSize];
  const int tid = threadIdx.x;
  double partial = 0.0;
  for (int bin = tid; bin < kHistBins; bin += blockDim.x) {
    unsigned int count = histogram[bin];
    if (count > 0) {
      double p = static_cast<double>(count) / static_cast<double>(total_count);
      partial -= p * log2(p);
    }
  }
  s_partial[tid] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) s_partial[tid] += s_partial[tid + s];
    __syncthreads();
  }
  if (tid == 0) *entropy_out = s_partial[0];
}

/**
 * Normalize the accumulated sums into the two remaining features.
 * Mirrors finalizeStatsOnlyKernel (stats_kernel.cu), including its
 * `n > 2` guard on the second derivative. Entropy is not touched here --
 * EntropyFromHistKernel writes it straight into the struct, the same way
 * upstream's entropy kernel writes into `&d_stats->entropy`.
 */
/* Convert a float64 chunk to float32 in place of reinterpreting it.
 *
 * The distinction is the entire bug this exists to fix: reinterpreting keeps
 * the bits and changes their meaning, so each double is read as two float32
 * words and the low word -- pure mantissa -- lands on IEEE-754's reserved
 * exponent==255 (a NaN) about one time in 256. Converting keeps the meaning
 * and changes the bits, which is what the model was normalised against.
 *
 * Native float64 statistics were the obvious alternative and are wrong here:
 * MAD and the second derivative come out identical either way (they measure
 * values, so precision is irrelevant -- 37.731005 vs 37.731004 on a real
 * chunk), but entropy is a BYTE histogram and shifts by 0.58 bits (6.8747 vs
 * 6.2985) because a double spends eight bytes where a float spends four. The
 * model has only ever seen the four-byte distribution. */
__global__ void DowncastF64ToF32Kernel(const double *__restrict__ in,
                                        float *__restrict__ out,
                                        size_t num_elements) {
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements;
       i += stride) {
    out[i] = static_cast<float>(in[i]);
  }
}

__global__ void FinalizeFeatureStatsKernel(const double *__restrict__ scalars,
                                            size_t num_elements,
                                            DeviceFeatureStats *__restrict__ out) {
  if (threadIdx.x != 0) return;
  out->mad = (num_elements > 0)
                 ? scalars[2] / static_cast<double>(num_elements)
                 : 0.0;
  out->second_derivative =
      (num_elements > 2) ? scalars[1] / static_cast<double>(num_elements - 2)
                         : 0.0;
}

/**
 * Per-thread device scratch, allocated once and reused.
 *
 * Upstream preallocates the equivalent in its CompContext (ctx->d_stats,
 * ctx->d_histogram, ctx->d_stats_workspace) precisely so the per-chunk path
 * contains no allocator traffic; ComputeDeviceStatsTyped below still does
 * four cudaMalloc/cudaFree per chunk, which is the other half of what made
 * the two pipelines different shapes.
 *
 * Leaked on purpose, like EmaBuffer() in neuropress_nn_gpu_kernels.cu: freeing
 * a device allocation from a thread-exit or static destructor races the CUDA
 * runtime's own teardown.
 */
struct DeviceStatsScratch {
  cudaStream_t stream = nullptr;
  unsigned int *d_hist = nullptr;
  double *d_scalars = nullptr;  // [sum, sum_abs_d2, sum_abs_dev]
  DeviceFeatureStats *d_stats = nullptr;
  /* Narrowed copy of a float64 chunk, grown on demand and reused. See
     ComputeDeviceStatsResidentF32From64: the model is normalised on float32
     statistics, so a double chunk is CONVERTED before it is measured. */
  float *d_narrow = nullptr;
  size_t narrow_capacity = 0;
  /* QuantizeDevice's range keys and pass counters, four u64 reused per chunk
     rather than an allocator round trip each. */
  unsigned long long *d_range = nullptr;
  bool ok = false;
};

DeviceStatsScratch &Scratch() {
  static thread_local DeviceStatsScratch *s = [] {
    auto *p = new DeviceStatsScratch();
    p->ok = cudaStreamCreate(&p->stream) == cudaSuccess &&
            cudaMalloc(&p->d_hist, kHistBins * sizeof(unsigned int)) ==
                cudaSuccess &&
            cudaMalloc(&p->d_scalars, 3 * sizeof(double)) == cudaSuccess &&
            cudaMalloc(&p->d_stats, sizeof(DeviceFeatureStats)) == cudaSuccess &&
            cudaMalloc(&p->d_range, 4 * sizeof(unsigned long long)) ==
                cudaSuccess;
    return p;
  }();
  return *s;
}

/** Device-resident stats for one element type. No host round trip. */
template <typename T>
bool ComputeDeviceStatsResidentTyped(const T *data, size_t num_elements,
                                      DeviceStatsScratch &s,
                                      cudaStream_t stream) {
  const size_t num_bytes = num_elements * sizeof(T);
  if (cudaMemsetAsync(s.d_hist, 0, kHistBins * sizeof(unsigned int), stream) !=
          cudaSuccess ||
      cudaMemsetAsync(s.d_scalars, 0, 3 * sizeof(double), stream) !=
          cudaSuccess ||
      cudaMemsetAsync(s.d_stats, 0, sizeof(DeviceFeatureStats), stream) !=
          cudaSuccess) {
    return false;
  }

  int grid = static_cast<int>(std::min<size_t>(
      (num_elements + kBlockSize - 1) / kBlockSize, 1024));
  if (grid < 1) grid = 1;

  // Same four stages upstream runs, in the same order, all on one stream:
  // pass 1 (histogram + sum + second derivative), entropy from the histogram,
  // pass 2 (MAD, mean read on-device), finalize.
  StatsPass1Kernel<T><<<grid, kBlockSize, 0, stream>>>(
      data, num_elements, s.d_hist, s.d_scalars, s.d_scalars + 1);
  EntropyFromHistKernel<<<1, kBlockSize, 0, stream>>>(s.d_hist, num_bytes,
                                                      &s.d_stats->entropy);
  StatsPass2DevKernel<T><<<grid, kBlockSize, 0, stream>>>(
      data, num_elements, s.d_scalars, s.d_scalars + 2);
  FinalizeFeatureStatsKernel<<<1, 1, 0, stream>>>(s.d_scalars, num_elements,
                                                  s.d_stats);
  return cudaGetLastError() == cudaSuccess;
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
  // failure at every stage (stats_kernel.cu returns nullptr, and
  // gpucompress_compress.cpp bails on it); ComputeCompressionFeatures's
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

void *DeviceStatsStream() {
  DeviceStatsScratch &s = Scratch();
  return s.ok ? static_cast<void *>(s.stream) : nullptr;
}

/* Device-resident float64: convert to float32 on the GPU, then measure that.
 *
 * Same treatment the host path applies, so a chunk gets the same features
 * wherever it happens to live -- otherwise the model's input would depend on
 * whether the application wrote from host or device memory, which is not a
 * property of the data at all.
 *
 * The narrowed buffer is scratch, grown on demand and reused across chunks;
 * it costs half the chunk's size in device memory. Returns the same opaque
 * stats pointer the float32 path returns, so callers are unchanged. */
const void *ComputeDeviceStatsResidentF32From64(const void *device_data,
                                                  size_t num_doubles,
                                                  void *stream) {
  DeviceStatsScratch &s = Scratch();
  if (!s.ok || device_data == nullptr || num_doubles == 0) return nullptr;
  cudaStream_t st = stream ? static_cast<cudaStream_t>(stream) : s.stream;

  if (s.narrow_capacity < num_doubles) {
    if (s.d_narrow != nullptr) cudaFree(s.d_narrow);
    s.d_narrow = nullptr;
    s.narrow_capacity = 0;
    if (cudaMalloc(&s.d_narrow, num_doubles * sizeof(float)) != cudaSuccess) {
      /* Out of device memory for the scratch: report failure rather than fall
         back to reinterpreting, which is the defect this replaces. */
      s.d_narrow = nullptr;
      return nullptr;
    }
    s.narrow_capacity = num_doubles;
  }

  const int block = 256;
  const int grid = static_cast<int>(
      (num_doubles + block - 1) / block > 65535
          ? 65535
          : (num_doubles + block - 1) / block);
  DowncastF64ToF32Kernel<<<grid, block, 0, st>>>(
      static_cast<const double *>(device_data), s.d_narrow, num_doubles);
  if (cudaGetLastError() != cudaSuccess) return nullptr;

  return ComputeDeviceStatsResident(static_cast<const void *>(s.d_narrow),
                                    num_doubles, DataType::FLOAT32, stream);
}

const void *ComputeDeviceStatsResident(const void *device_data,
                                        size_t num_elements, DataType type,
                                        void *stream) {
  DeviceStatsScratch &s = Scratch();
  if (!s.ok || device_data == nullptr) return nullptr;
  cudaStream_t st = stream ? static_cast<cudaStream_t>(stream) : s.stream;

  // A chunk with no elements has no statistics. The zeroing memset in the
  // typed helper would leave the struct at all zeros, which is a real point
  // in the feature space ("perfectly compressible"), so refuse instead --
  // upstream refuses the same case outright (gpucompress_compress.cpp).
  if (num_elements == 0) return nullptr;

  bool ok = false;
  switch (type) {
    case DataType::UINT8:
      ok = ComputeDeviceStatsResidentTyped<uint8_t>(
          static_cast<const uint8_t *>(device_data), num_elements, s, st);
      break;
    case DataType::INT32:
      ok = ComputeDeviceStatsResidentTyped<int32_t>(
          static_cast<const int32_t *>(device_data), num_elements, s, st);
      break;
    case DataType::FLOAT32:
      ok = ComputeDeviceStatsResidentTyped<float>(
          static_cast<const float *>(device_data), num_elements, s, st);
      break;
    case DataType::DOUBLE64:
      ok = ComputeDeviceStatsResidentTyped<double>(
          static_cast<const double *>(device_data), num_elements, s, st);
      break;
    default:
      return nullptr;
  }
  return ok ? static_cast<const void *>(s.d_stats) : nullptr;
}

bool ReadDeviceFeatureStats(const void *device_stats, double *out_entropy,
                            double *out_mad, double *out_second_derivative,
                            void *stream) {
  if (!device_stats || !out_entropy || !out_mad || !out_second_derivative) {
    return false;
  }
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  DeviceFeatureStats h{};
  if (cudaMemcpyAsync(&h, device_stats, sizeof(h), cudaMemcpyDeviceToHost,
                      st) != cudaSuccess) {
    return false;
  }
  // Stream-scoped, not device-wide: this runs concurrently with other
  // workers' compressions and must not serialize them. Upstream likewise
  // ends its inference phase with cudaStreamSynchronize(stream)
  // (nn_gpu.cu), never cudaDeviceSynchronize.
  if (cudaStreamSynchronize(st) != cudaSuccess) return false;
  *out_entropy = h.entropy;
  *out_mad = h.mad;
  *out_second_derivative = h.second_derivative;
  return true;
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
namespace ctp::compress::preprocess {

namespace {

/**
 * Byte planes are built WITHIN each kShuffleChunkBytes block, never across
 * the whole buffer -- NeuroPress splits the input first
 * (byte_shuffle_kernels.cu's createDeviceChunkArrays) and each of its blocks
 * computes `num_elements = chunk_size / ElementSize` for its OWN chunk. A
 * global plane layout produces different bytes for anything above 256 KiB.
 *
 * A 2-D grid: blockIdx.y walks the chunks, and the x blocks split ONE chunk's
 * elements between them. It used to be one block per chunk, which left an
 * 8 MiB buffer 32 blocks on a 108-SM A100 and a 2 MiB one 8 -- 12% achieved
 * occupancy, under 3% of DRAM bandwidth, ~150 us per chunk. Only the work
 * decomposition changed, so every byte lands where it did. Within a chunk the
 * write side is coalesced (out[b*n + elem] is contiguous across threads) at
 * the cost of a strided read -- the same trade upstream makes.
 */
template <unsigned ElemSize>
__global__ void ShuffleKernel(const uint8_t *__restrict__ in,
                              uint8_t *__restrict__ out, size_t num_bytes,
                              size_t chunk_bytes) {
  const size_t num_chunks = (num_bytes + chunk_bytes - 1) / chunk_bytes;
  const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t c = blockIdx.y; c < num_chunks; c += gridDim.y) {
    const size_t base = c * chunk_bytes;
    const size_t remain = num_bytes - base;
    const size_t chunk = remain < chunk_bytes ? remain : chunk_bytes;
    const size_t n = chunk / ElemSize;
    const uint8_t *ci = in + base;
    uint8_t *co = out + base;
    for (size_t elem = first; elem < n; elem += stride) {
#pragma unroll
      for (unsigned b = 0; b < ElemSize; ++b) {
        co[b * n + elem] = ci[elem * ElemSize + b];
      }
    }
    // Trailing partial element of THIS chunk, copied verbatim
    // (byte_shuffle_kernels.cu:59-65). Only the last chunk can have one,
    // since kShuffleChunkBytes is a multiple of every supported ElemSize.
    for (size_t i = first + n * ElemSize; i < chunk; i += stride) {
      co[i] = ci[i];
    }
  }
}

template <unsigned ElemSize>
__global__ void UnshuffleKernel(const uint8_t *__restrict__ in,
                                uint8_t *__restrict__ out, size_t num_bytes,
                                size_t chunk_bytes) {
  const size_t num_chunks = (num_bytes + chunk_bytes - 1) / chunk_bytes;
  const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t c = blockIdx.y; c < num_chunks; c += gridDim.y) {
    const size_t base = c * chunk_bytes;
    const size_t remain = num_bytes - base;
    const size_t chunk = remain < chunk_bytes ? remain : chunk_bytes;
    const size_t n = chunk / ElemSize;
    const uint8_t *ci = in + base;
    uint8_t *co = out + base;
    for (size_t elem = first; elem < n; elem += stride) {
#pragma unroll
      for (unsigned b = 0; b < ElemSize; ++b) {
        co[elem * ElemSize + b] = ci[b * n + elem];
      }
    }
    for (size_t i = first + n * ElemSize; i < chunk; i += stride) {
      co[i] = ci[i];
    }
  }
}

/**
 * Shared validation + launch geometry for both directions: y = one block row
 * per chunk (at most 65535, grid-strided past that), x = enough blocks to give
 * each of a full chunk's elements its own thread.
 *
 * @param in        device input
 * @param out       device output
 * @param num_bytes buffer length
 * @param elem_size 2, 4 or 8
 * @param grid      filled with the launch grid
 * @param threads   filled with the block size
 * @return false if the request is not shuffleable
 */
bool PrepareLaunch(const uint8_t *in, uint8_t *out, size_t num_bytes,
                   size_t elem_size, dim3 *grid, int *threads) {
  if (!in || !out || num_bytes == 0) return false;
  if (elem_size != 2 && elem_size != 4 && elem_size != 8) return false;
  // Sub-element buffers are copied verbatim, not rejected -- same as the
  // host routines and as NeuroPress's byte_shuffle_simple.
  *threads = 256;
  const size_t num_chunks =
      (num_bytes + kShuffleChunkBytes - 1) / kShuffleChunkBytes;
  const size_t per_chunk =
      std::min(num_bytes, kShuffleChunkBytes) / elem_size;
  const size_t x = (per_chunk + *threads - 1) / *threads;
  *grid = dim3(static_cast<unsigned>(x < 1 ? 1 : x),
               static_cast<unsigned>(num_chunks > 65535 ? 65535 : num_chunks));
  return true;
}

bool FinishLaunch(cudaStream_t stream) {
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

}  // namespace

bool ByteShuffleDevice(const void *device_in, void *device_out,
                       size_t num_bytes, size_t elem_size, void *stream_in) {
  const uint8_t *in = static_cast<const uint8_t *>(device_in);
  uint8_t *out = static_cast<uint8_t *>(device_out);
  dim3 blocks;
  int threads = 0;
  if (!PrepareLaunch(in, out, num_bytes, elem_size, &blocks, &threads)) {
    return false;
  }
  // A caller-supplied stream also means "do not wait": whatever consumes the
  // shuffled bytes is queued behind this on the same stream, so the ordering
  // is already guaranteed and a sync here would only serialize the sweep.
  const bool caller_stream = (stream_in != nullptr);
  cudaStream_t stream =
      caller_stream ? static_cast<cudaStream_t>(stream_in)
                    : static_cast<cudaStream_t>(DeviceStatsStream());
  const size_t cb = kShuffleChunkBytes;
  if (elem_size == 2) {
    ShuffleKernel<2><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  } else if (elem_size == 4) {
    ShuffleKernel<4><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  } else {
    ShuffleKernel<8><<<blocks, threads, 0, stream>>>(in, out, num_bytes, cb);
  }
  if (caller_stream) return cudaGetLastError() == cudaSuccess;
  return FinishLaunch(stream);
}

bool ByteUnshuffleDevice(const void *device_in, void *device_out,
                         size_t num_bytes, size_t elem_size) {
  const uint8_t *in = static_cast<const uint8_t *>(device_in);
  uint8_t *out = static_cast<uint8_t *>(device_out);
  dim3 blocks;
  int threads = 0;
  if (!PrepareLaunch(in, out, num_bytes, elem_size, &blocks, &threads)) {
    return false;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(DeviceStatsStream());
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

// ===========================================================================
// Device quantization (issue #693): the lossy half of NeuroPress's action
// space. Lives here for the same reason the shuffle kernels do -- adding a
// separately-device-linked .cu to this RDC-enabled static library breaks
// __cudaRegisterLinkedBinary at static init.
//
// Every element is either on the grid, checked through the decoder's own
// arithmetic, or stored bit-exact in its slot (escape mode), so a chunk is
// never refused and never exceeds the bound.
// ===========================================================================
#include "clio_ctp/compress/preprocess/quantization.h"

#include <cstring>

namespace ctp::compress::preprocess {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// Largest double below 2^63, so the int64 cast of a checked index is defined.
constexpr double kInt64Max = 9223372036854774784.0;

/** The decoder: one correctly rounded fma, then the element type. */
template <typename T>
__device__ __forceinline__ T Decode(double q, double inv_scale, double offset) {
  return static_cast<T>(fma(q, inv_scale, offset));
}

__device__ __forceinline__ int32_t BitsOf(float f) {
  return static_cast<int32_t>(__float_as_uint(f));
}
__device__ __forceinline__ int64_t BitsOf(double d) {
  return static_cast<int64_t>(__double_as_longlong(d));
}
__device__ __forceinline__ float FromBits(int32_t b) {
  return __uint_as_float(static_cast<unsigned int>(b));
}
__device__ __forceinline__ double FromBits(int64_t b) {
  return __longlong_as_double(static_cast<long long>(b));
}

/** Grid encoder. fail: 1 = index outside the width, 2 = bound missed. */
template <typename InT, typename OutT>
__global__ void QuantizeKernel(const InT *__restrict__ in,
                               OutT *__restrict__ out, size_t n, double scale,
                               double offset, double lo, double hi,
                               double inv_scale, double eb_check,
                               unsigned long long *fail) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    const double x = static_cast<double>(in[i]);
    const double q = round((x - offset) * scale);
    if (!(q >= lo && q <= hi)) { atomicExch(fail, 1ull); return; }
    out[i] = static_cast<OutT>(q);
    const double z = static_cast<double>(Decode<InT>(q, inv_scale, offset));
    if (!(fabs(z - x) <= eb_check)) { atomicExch(fail, 2ull); return; }
  }
}

/**
 * Escape-mode encoder, one bitmap byte per iteration: an element with
 * |x| < limit whose index fits and whose decode passes keeps its index;
 * any other element keeps its own bits and gets its map bit set.
 */
template <typename InT, typename SlotT>
__global__ void EscapeQuantizeKernel(const InT *__restrict__ in,
                                     SlotT *__restrict__ slots,
                                     uint8_t *__restrict__ map, size_t n,
                                     double limit, double scale,
                                     double offset, double lo, double hi,
                                     double inv_scale, double eb_check,
                                     unsigned long long *escaped) {
  size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t rows = (n + 7) / 8;
  for (; j < rows; j += stride) {
    unsigned int bits = 0, count = 0;
    const size_t end = (8 * j + 8 < n) ? 8 * j + 8 : n;
    for (size_t i = 8 * j; i < end; ++i) {
      const double x = static_cast<double>(in[i]);
      bool ok = fabs(x) < limit;
      if (ok) {
        const double q = round((x - offset) * scale);
        ok = q >= lo && q <= hi &&
             fabs(static_cast<double>(Decode<InT>(q, inv_scale, offset)) -
                  x) <= eb_check;
        if (ok) slots[i] = static_cast<SlotT>(q);
      }
      if (!ok) {
        slots[i] = BitsOf(in[i]);
        bits |= 1u << (i - 8 * j);
        ++count;
      }
    }
    map[j] = static_cast<uint8_t>(bits);
    if (count != 0) atomicAdd(escaped, static_cast<unsigned long long>(count));
  }
}

template <typename SlotT, typename OutT>
__global__ void DequantizeKernel(const SlotT *__restrict__ in,
                                 OutT *__restrict__ out, size_t n,
                                 double inv_scale, double offset) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    out[i] = Decode<OutT>(static_cast<double>(in[i]), inv_scale, offset);
  }
}

template <typename SlotT, typename OutT>
__global__ void EscapeDequantizeKernel(const SlotT *__restrict__ in,
                                       const uint8_t *__restrict__ map,
                                       OutT *__restrict__ out, size_t n,
                                       double inv_scale, double offset) {
  size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t rows = (n + 7) / 8;
  for (; j < rows; j += stride) {
    const unsigned int bits = map[j];
    const size_t end = (8 * j + 8 < n) ? 8 * j + 8 : n;
    for (size_t i = 8 * j; i < end; ++i) {
      out[i] = ((bits >> (i - 8 * j)) & 1u)
                   ? FromBits(in[i])
                   : Decode<OutT>(static_cast<double>(in[i]), inv_scale,
                                  offset);
    }
  }
}

/**
 * Monotonic value->unsigned key, so integer min/max order values correctly:
 * for x >= 0 the IEEE bit pattern already orders, for x < 0 it orders in
 * reverse, and this maps both into one increasing space.
 */
__device__ __forceinline__ unsigned int ValueKey(float f) {
  unsigned int b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ unsigned long long ValueKey(double d) {
  const unsigned long long s = 0x8000000000000000ull;
  unsigned long long b = static_cast<unsigned long long>(__double_as_longlong(d));
  return (b & s) ? ~b : (b | s);
}
double KeyToValue(unsigned int k) {
  unsigned int b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  float f;
  std::memcpy(&f, &b, sizeof(f));
  return static_cast<double>(f);
}
double KeyToValue(unsigned long long k) {
  const unsigned long long s = 0x8000000000000000ull;
  unsigned long long b = (k & s) ? (k & ~s) : ~k;
  double d;
  std::memcpy(&d, &b, sizeof(d));
  return d;
}

__device__ __forceinline__ void AtomicMinKey(unsigned int *a, unsigned int v) {
  atomicMin(a, v);
}
__device__ __forceinline__ void AtomicMaxKey(unsigned int *a, unsigned int v) {
  atomicMax(a, v);
}
__device__ __forceinline__ void AtomicMinKey(unsigned long long *a,
                                            unsigned long long v) {
  unsigned long long cur = *a;
  while (v < cur) {
    const unsigned long long seen = atomicCAS(a, cur, v);
    if (seen == cur) break;
    cur = seen;
  }
}
__device__ __forceinline__ void AtomicMaxKey(unsigned long long *a,
                                            unsigned long long v) {
  unsigned long long cur = *a;
  while (v > cur) {
    const unsigned long long seen = atomicCAS(a, cur, v);
    if (seen == cur) break;
    cur = seen;
  }
}

/**
 * Min/max over the elements with |x| < limit (NaN never qualifies), block
 * reduction + one atomic pair per block; `skipped` is set if any did not.
 */
template <typename T, typename K>
__global__ void RangeKernel(const T *__restrict__ in, size_t n, double limit,
                            K *out_min, K *out_max,
                            unsigned long long *skipped) {
  __shared__ K s_min[kBlockSize];
  __shared__ K s_max[kBlockSize];
  K lo = static_cast<K>(~static_cast<K>(0)), hi = 0;
  bool skip = false;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    if (!(fabs(static_cast<double>(in[i])) < limit)) {
      skip = true;
      continue;
    }
    const K k = ValueKey(in[i]);
    lo = k < lo ? k : lo;
    hi = k > hi ? k : hi;
  }
  if (skip) atomicExch(skipped, 1ull);
  s_min[threadIdx.x] = lo;
  s_max[threadIdx.x] = hi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const K a = s_min[threadIdx.x + s], b = s_max[threadIdx.x + s];
      if (a < s_min[threadIdx.x]) s_min[threadIdx.x] = a;
      if (b > s_max[threadIdx.x]) s_max[threadIdx.x] = b;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    AtomicMinKey(out_min, s_min[0]);
    AtomicMaxKey(out_max, s_max[0]);
  }
}

int Blocks(size_t count) {
  const size_t b = (count + 255) / 256;
  return static_cast<int>(b < 1 ? 1 : (b > 65535 ? 65535 : b));
}

/**
 * Range of the elements below `limit` on the caller's stream, in the
 * per-thread scratch. *found is false when no element qualified.
 */
bool DeviceRange(const void *d_in, size_t n, bool f64, double limit,
                 cudaStream_t stream, double *lo, double *hi, bool *found,
                 bool *skipped) {
  // A sticky error from an unrelated earlier call (IsDevicePointer on a host
  // pointer) must not be blamed on this launch.
  cudaGetLastError();
  DeviceStatsScratch &sc = Scratch();
  unsigned long long *d = sc.d_range;
  bool owns = false;
  if (d == nullptr) {
    if (cudaMalloc(&d, 4 * sizeof(unsigned long long)) != cudaSuccess) {
      return false;
    }
    owns = true;
  }
  const int grid = static_cast<int>(
      std::min<size_t>((n + kBlockSize - 1) / kBlockSize, 1024));
  unsigned long long h[3] = {~0ull, 0ull, 0ull};
  if (!f64) {
    const unsigned int k[2] = {0xFFFFFFFFu, 0u};
    std::memcpy(h, k, sizeof(k));
  }
  bool ok = cudaMemcpyAsync(d, h, sizeof(h), cudaMemcpyHostToDevice,
                            stream) == cudaSuccess;
  if (ok) {
    if (f64) {
      RangeKernel<double, unsigned long long><<<grid, kBlockSize, 0, stream>>>(
          static_cast<const double *>(d_in), n, limit, d, d + 1, d + 2);
    } else {
      auto *k = reinterpret_cast<unsigned int *>(d);
      RangeKernel<float, unsigned int><<<grid, kBlockSize, 0, stream>>>(
          static_cast<const float *>(d_in), n, limit, k, k + 1, d + 2);
    }
    ok = cudaGetLastError() == cudaSuccess &&
         cudaMemcpyAsync(h, d, sizeof(h), cudaMemcpyDeviceToHost, stream) ==
             cudaSuccess &&
         cudaStreamSynchronize(stream) == cudaSuccess;
  }
  if (owns) cudaFree(d);
  if (!ok) return false;
  *skipped = h[2] != 0;
  if (f64) {
    *found = h[0] <= h[1];
    *lo = KeyToValue(h[0]);
    *hi = KeyToValue(h[1]);
  } else {
    unsigned int k[2];
    std::memcpy(k, h, sizeof(k));
    *found = k[0] <= k[1];
    *lo = KeyToValue(k[0]);
    *hi = KeyToValue(k[1]);
  }
  if (!*found) *lo = *hi = 0.0;
  return true;
}

/**
 * Half the float32 spacing at |x| -- the most a (float) cast can move a double
 * of that magnitude. From nextafterf, not a fixed ratio: the real quantity
 * doubles at every binade boundary.
 */
double HalfUlpFloat32(double x) {
  // Beyond the float range the cast is undefined; the decoder's cast then
  // returns the source itself, so nothing needs reserving.
  if (!(std::fabs(x) <= std::numeric_limits<float>::max())) return 0.0;
  const float f = static_cast<float>(std::fabs(x));
  if (!std::isfinite(f)) return 0.0;
  const float nxt = std::nextafterf(f, std::numeric_limits<float>::infinity());
  return 0.5 * (static_cast<double>(nxt) - static_cast<double>(f));
}

/** Full float64 spacing at |x|; infinite when x is. */
double UlpFloat64(double x) {
  x = std::fabs(x);
  if (!std::isfinite(x)) return kInf;
  return std::nextafter(x, kInf) - x;
}

struct Grid {
  double offset = 0.0, delta = 1.0, scale = 1.0, inv_scale = 1.0;
};

/**
 * Step whose decoded values land within 0.95*eb of their source:
 *   float32  rounding gives delta/2 and the cast at most ULP/2; where ULP is
 *            the larger, the cast returns the source itself (|z-x| <= delta).
 *   float64  no final cast, so reserve 12 ULP for the double arithmetic.
 * The per-element check, not this bound, is what the guarantee rests on.
 */
bool MakeGrid(double lo, double hi, double eb, bool f64, Grid *g) {
  const double m = std::min(std::max(std::fabs(lo), std::fabs(hi)) + eb,
                            std::numeric_limits<double>::max());
  double d;
  if (hi - lo == 0.0) {
    d = 1.0;  // every value IS lo
  } else if (!std::isfinite(eb)) {
    d = 1e300;
  } else if (f64) {
    d = 2.0 * (0.95 * eb - 12.0 * UlpFloat64(m));
  } else {
    d = std::max(2.0 * (0.95 * eb - HalfUlpFloat32(m)), 0.95 * eb);
  }
  if (!(d > 0.0)) return false;
  d = std::min(d, 1e300);
  const double s = 1.0 / d;
  const double inv = 1.0 / s;
  if (!(s > 0.0 && std::isfinite(s) && inv > 0.0 && std::isfinite(inv))) {
    return false;
  }
  *g = Grid{lo, d, s, inv};
  return true;
}

int PrecisionFor(double qmax, bool f64) {
  if (qmax <= 127.0) return 8;
  if (qmax <= 32767.0) return 16;
  if (qmax <= 2147483647.0) return 32;
  if (f64 && qmax <= kInt64Max) return 64;
  return 0;
}

template <typename OutT>
double IndexMax() {
  return sizeof(OutT) == 8 ? kInt64Max
                           : static_cast<double>(std::numeric_limits<OutT>::max());
}

template <typename InT, typename OutT>
void LaunchGrid(const void *in, void *out, size_t n, const Grid &g,
                double eb_check, unsigned long long *flag, cudaStream_t st) {
  QuantizeKernel<InT, OutT><<<Blocks(n), 256, 0, st>>>(
      static_cast<const InT *>(in), static_cast<OutT *>(out), n, g.scale,
      g.offset, static_cast<double>(std::numeric_limits<OutT>::min()),
      IndexMax<OutT>(), g.inv_scale, eb_check, flag);
}

template <typename InT>
void LaunchGridTyped(int precision, const void *in, void *out, size_t n,
                     const Grid &g, double eb_check, unsigned long long *flag,
                     cudaStream_t st) {
  switch (precision) {
    case 8:  LaunchGrid<InT, int8_t>(in, out, n, g, eb_check, flag, st); break;
    case 16: LaunchGrid<InT, int16_t>(in, out, n, g, eb_check, flag, st); break;
    case 32: LaunchGrid<InT, int32_t>(in, out, n, g, eb_check, flag, st); break;
    default: LaunchGrid<InT, int64_t>(in, out, n, g, eb_check, flag, st); break;
  }
}

/** One quantize pass; *counter receives the kernel's flag or escape count. */
bool RunPass(bool f64, bool escape, int precision, const void *in, void *out,
             size_t n, double limit, const Grid &g, double eb_check,
             cudaStream_t st, unsigned long long *counter) {
  cudaGetLastError();
  DeviceStatsScratch &sc = Scratch();
  unsigned long long *d = sc.d_range;
  bool owns = false;
  if (d == nullptr) {
    if (cudaMalloc(&d, 4 * sizeof(unsigned long long)) != cudaSuccess) {
      return false;
    }
    owns = true;
  }
  unsigned long long *c = d + 3;
  unsigned long long h = 0;
  bool ok = cudaMemcpyAsync(c, &h, sizeof(h), cudaMemcpyHostToDevice, st) ==
            cudaSuccess;
  if (ok) {
    if (escape) {
      const int blocks = Blocks((n + 7) / 8);
      const double lo = f64 ? -9223372036854775808.0 : -2147483648.0;
      const double hi = f64 ? kInt64Max : 2147483647.0;
      if (f64) {
        auto *s = static_cast<int64_t *>(out);
        EscapeQuantizeKernel<double, int64_t><<<blocks, 256, 0, st>>>(
            static_cast<const double *>(in), s,
            reinterpret_cast<uint8_t *>(s + n), n, limit, g.scale, g.offset,
            lo, hi, g.inv_scale, eb_check, c);
      } else {
        auto *s = static_cast<int32_t *>(out);
        EscapeQuantizeKernel<float, int32_t><<<blocks, 256, 0, st>>>(
            static_cast<const float *>(in), s,
            reinterpret_cast<uint8_t *>(s + n), n, limit, g.scale, g.offset,
            lo, hi, g.inv_scale, eb_check, c);
      }
    } else if (f64) {
      LaunchGridTyped<double>(precision, in, out, n, g, eb_check, c, st);
    } else {
      LaunchGridTyped<float>(precision, in, out, n, g, eb_check, c, st);
    }
    ok = cudaGetLastError() == cudaSuccess &&
         cudaMemcpyAsync(&h, c, sizeof(h), cudaMemcpyDeviceToHost, st) ==
             cudaSuccess &&
         cudaStreamSynchronize(st) == cudaSuccess;
  }
  if (owns) cudaFree(d);
  *counter = h;
  return ok;
}

}  // namespace

bool QuantizeDevice(const void *device_in, size_t num_elements,
                    size_t elem_bytes, double error_bound, void *device_out,
                    size_t *out_bytes, DeviceQuantizeParams *out_params,
                    void *stream_in) {
  if (out_params == nullptr) return false;
  *out_params = DeviceQuantizeParams{};
  out_params->refusal = QuantizeRefusal::kInvalidArgument;
  if (!device_in || !device_out || !out_bytes || num_elements == 0 ||
      (elem_bytes != 4 && elem_bytes != 8) || !(error_bound > 0.0)) {
    return false;
  }
  out_params->refusal = QuantizeRefusal::kDeviceError;
  const bool f64 = elem_bytes == 8;
  const size_t n = num_elements;
  // Same per-thread stream the rest of this path uses; upstream's
  // quantize_simple likewise takes a stream and waits only on it.
  cudaStream_t st = (stream_in != nullptr)
                        ? static_cast<cudaStream_t>(stream_in)
                        : static_cast<cudaStream_t>(DeviceStatsStream());
  // Four doubles below eb: a double-rounded |z - x| that passes this is
  // within eb exactly.
  double eb_check = error_bound;
  for (int k = 0; k < 4; ++k) eb_check = std::nextafter(eb_check, 0.0);

  double lo = 0.0, hi = 0.0;
  bool found = false, skipped = false;
  if (!DeviceRange(device_in, n, f64, kInf, st, &lo, &hi, &found, &skipped)) {
    return false;
  }
  Grid g;
  int precision = 0;
  if (found && MakeGrid(lo, hi, error_bound, f64, &g)) {
    precision = PrecisionFor(std::ceil((hi - lo) / g.delta), f64);
  }
  unsigned long long counter = 0;
  if (precision != 0 && !skipped &&
      !RunPass(f64, false, precision, device_in, device_out, n, kInf, g,
               eb_check, st, &counter)) {
    return false;
  }
  const bool escapes = precision == 0 || skipped || counter != 0;
  unsigned long long escaped = 0;
  if (escapes) {
    // When the finite values fit a grid, only the others escape. Otherwise
    // the grid spans the elements below 2^(floor(log2 eb)+24), above which
    // float32 spacing already exceeds eb, and that span always fits int32;
    // float64 keeps a margin for its 12-ULP reserve, which int64 holds.
    double limit = kInf;
    if (precision == 0) {
      if (std::isfinite(error_bound)) {
        limit = std::ldexp(1.0, std::ilogb(error_bound) + (f64 ? 47 : 24));
      }
      if (!DeviceRange(device_in, n, f64, limit, st, &lo, &hi, &found,
                       &skipped)) {
        return false;
      }
      if (!MakeGrid(lo, hi, error_bound, f64, &g)) {
        g = Grid{lo, 1.0, 1.0, 1.0};
      }
    }
    precision = f64 ? 64 : 32;
    if (!RunPass(f64, true, precision, device_in, device_out, n, limit, g,
                 eb_check, st, &escaped)) {
      return false;
    }
  }

  out_params->error_bound = error_bound;
  out_params->effective_error_bound = 0.5 * g.delta;
  out_params->scale = g.scale;
  out_params->data_min = g.offset;
  out_params->data_max = hi;
  out_params->precision = precision;
  out_params->elem_bytes = static_cast<int>(elem_bytes);
  out_params->escapes = escapes;
  out_params->escape_count = escaped;
  out_params->bound_achievable = true;
  *out_bytes = QuantizedBytes(n, *out_params);
  const size_t used = n * PrecisionToBytes(precision) + (escapes ? (n + 7) / 8 : 0);
  if (*out_bytes > used &&
      (cudaMemsetAsync(static_cast<char *>(device_out) + used, 0,
                       *out_bytes - used, st) != cudaSuccess ||
       cudaStreamSynchronize(st) != cudaSuccess)) {
    return false;
  }
  out_params->refusal = QuantizeRefusal::kNone;
  return true;
}

bool DequantizeDevice(const void *device_in, size_t num_elements,
                      const DeviceQuantizeParams &params, void *device_out) {
  cudaStream_t st = static_cast<cudaStream_t>(DeviceStatsStream());
  if (!device_in || !device_out || num_elements == 0 ||
      !(params.scale > 0.0) || !std::isfinite(params.scale) ||
      (params.elem_bytes != 4 && params.elem_bytes != 8)) {
    return false;
  }
  const bool f64 = params.elem_bytes == 8;
  const double inv = 1.0 / params.scale;
  if (!std::isfinite(inv)) return false;
  const double off = params.data_min;
  const size_t n = num_elements;
  const size_t width = PrecisionToBytes(params.precision);
  cudaGetLastError();

  if (params.escapes) {
    if (width != static_cast<size_t>(params.elem_bytes)) return false;
    const auto *map = static_cast<const uint8_t *>(device_in) + n * width;
    const int blocks = Blocks((n + 7) / 8);
    if (f64) {
      EscapeDequantizeKernel<int64_t, double><<<blocks, 256, 0, st>>>(
          static_cast<const int64_t *>(device_in), map,
          static_cast<double *>(device_out), n, inv, off);
    } else {
      EscapeDequantizeKernel<int32_t, float><<<blocks, 256, 0, st>>>(
          static_cast<const int32_t *>(device_in), map,
          static_cast<float *>(device_out), n, inv, off);
    }
  } else if (f64) {
    auto *o = static_cast<double *>(device_out);
    switch (width) {
      case 1: DequantizeKernel<int8_t, double><<<Blocks(n), 256, 0, st>>>(
                  static_cast<const int8_t *>(device_in), o, n, inv, off); break;
      case 2: DequantizeKernel<int16_t, double><<<Blocks(n), 256, 0, st>>>(
                  static_cast<const int16_t *>(device_in), o, n, inv, off); break;
      case 4: DequantizeKernel<int32_t, double><<<Blocks(n), 256, 0, st>>>(
                  static_cast<const int32_t *>(device_in), o, n, inv, off); break;
      default: DequantizeKernel<int64_t, double><<<Blocks(n), 256, 0, st>>>(
                  static_cast<const int64_t *>(device_in), o, n, inv, off); break;
    }
  } else {
    auto *o = static_cast<float *>(device_out);
    switch (width) {
      case 1: DequantizeKernel<int8_t, float><<<Blocks(n), 256, 0, st>>>(
                  static_cast<const int8_t *>(device_in), o, n, inv, off); break;
      case 2: DequantizeKernel<int16_t, float><<<Blocks(n), 256, 0, st>>>(
                  static_cast<const int16_t *>(device_in), o, n, inv, off); break;
      case 4: DequantizeKernel<int32_t, float><<<Blocks(n), 256, 0, st>>>(
                  static_cast<const int32_t *>(device_in), o, n, inv, off); break;
      default: DequantizeKernel<int64_t, float><<<Blocks(n), 256, 0, st>>>(
                  static_cast<const int64_t *>(device_in), o, n, inv, off); break;
    }
  }
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaStreamSynchronize(st) == cudaSuccess;
}

}  // namespace ctp::compress::preprocess
