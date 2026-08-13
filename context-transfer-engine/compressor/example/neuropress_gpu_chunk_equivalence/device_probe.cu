/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file device_probe.cu
 * @brief Device kernels behind device_probe.h.
 *
 * Deliberately free of Clio and NeuroPress includes: clio_cte/compressor
 * headers do not compile under nvcc (see the sibling neuropress_gpu_direct
 * example, which splits for the same reason), and NeuroPress's internals drag
 * in nvcomp. Kernels only.
 */

#include "device_probe.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace npeq {

namespace {

constexpr int kBlock = 256;
constexpr int kMaxBlocks = 1024;

#define NPEQ_CUDA_OK(expr)                                                   \
  do {                                                                       \
    cudaError_t rc_ = (expr);                                                \
    if (rc_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "[device_probe] %s:%d %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(rc_));                                 \
      return false;                                                          \
    }                                                                        \
  } while (0)

/** splitmix64 finalizer -- strong avalanche, no state. */
__device__ __forceinline__ unsigned long long Mix64(unsigned long long h) {
  h ^= h >> 30;
  h *= 0xBF58476D1CE4E5B9ULL;
  h ^= h >> 27;
  h *= 0x94D049BB133111EBULL;
  h ^= h >> 31;
  return h;
}

/**
 * Per-word hash summed across threads.
 *
 * The sum is what makes this order-independent (integer addition is
 * associative and commutative, so block scheduling cannot change the answer);
 * mixing the word with its own index is what keeps it sensitive to a
 * permutation of the data.
 */
__global__ void HashKernel(const unsigned char *__restrict__ data, size_t bytes,
                           unsigned long long *__restrict__ out) {
  const size_t num_words = bytes / 8;
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  unsigned long long acc = 0;
  const unsigned long long *words =
      reinterpret_cast<const unsigned long long *>(data);
  for (size_t i = idx; i < num_words; i += stride) {
    acc += Mix64(words[i] ^ (0x9E3779B97F4A7C15ULL * (i + 1)));
  }

  // Trailing bytes that do not fill a word, folded in once by a single thread
  // so the tail contributes exactly as much as any other position.
  if (idx == 0) {
    unsigned long long tail = 0;
    for (size_t b = num_words * 8; b < bytes; ++b) {
      tail = (tail << 8) | data[b];
    }
    if (bytes % 8 != 0) {
      acc += Mix64(tail ^ (0x9E3779B97F4A7C15ULL * (num_words + 1)));
    }
  }

  __shared__ unsigned long long shared[kBlock];
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, shared[0]);
}

/** Device-side accumulator for CompareDeviceBuffers. */
struct DeviceCompare {
  unsigned long long differing;
  unsigned long long first_diff;
};

__global__ void CompareKernel(const unsigned char *__restrict__ a,
                              const unsigned char *__restrict__ b, size_t bytes,
                              DeviceCompare *__restrict__ out) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  unsigned long long local_diff = 0;
  unsigned long long local_first = ~0ULL;
  for (size_t i = idx; i < bytes; i += stride) {
    if (a[i] != b[i]) {
      ++local_diff;
      if (i < local_first) local_first = i;
    }
  }
  if (local_diff != 0) {
    atomicAdd(&out->differing, local_diff);
    atomicMin(&out->first_diff, local_first);
  }
}

__global__ void BoundKernel(const float *__restrict__ a,
                            const float *__restrict__ b, size_t n, double bound,
                            unsigned long long *__restrict__ out) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (size_t i = idx; i < n; i += stride) {
    const double d = fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (!(d <= bound)) ++local;  // NaN-safe: a NaN difference counts as a miss
  }
  if (local != 0) atomicAdd(out, local);
}

/** Deterministic per-index pseudo-random in [0,1). No RNG state anywhere. */
__device__ __forceinline__ float IndexRandom(size_t i, unsigned int seed) {
  unsigned long long h = Mix64(static_cast<unsigned long long>(i) * 0x2545F491ULL +
                               seed * 0x9E3779B9ULL);
  return static_cast<float>((h >> 40) & 0xFFFFFF) / 16777216.0f;
}

__global__ void FillKernel(float *__restrict__ buf, size_t n, int regime,
                           unsigned int seed) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = idx; i < n; i += stride) {
    float v = 0.0f;
    switch (regime) {
      case 0:  // kConstant
        v = 1.5f;
        break;
      case 1:  // kLinearRamp
        v = static_cast<float>(i) * 1e-3f;
        break;
      case 2:  // kSmoothWave
        v = 100.0f * sinf(static_cast<float>(i) * 1e-4f);
        break;
      case 3:  // kStepped
        v = static_cast<float>((i / 4096) % 64) * 0.5f;
        break;
      case 4:  // kNoisyWave
        v = 100.0f * sinf(static_cast<float>(i) * 1e-4f) +
            0.5f * (IndexRandom(i, seed) - 0.5f);
        break;
      case 5:  // kHighEntropy
        v = (IndexRandom(i, seed) - 0.5f) * 2.0e6f;
        break;
      case 6:  // kSmallMagnitude
        v = (IndexRandom(i, seed) - 0.5f) * 2.0e-6f;
        break;
      default: {  // kMixed -- alternating smooth and noisy bands
        const bool noisy = ((i / 65536) % 2) == 1;
        v = noisy ? (IndexRandom(i, seed) - 0.5f) * 2.0e6f
                  : 100.0f * sinf(static_cast<float>(i) * 1e-4f);
        break;
      }
    }
    buf[i] = v;
  }
}

int BlocksFor(size_t work) {
  const size_t want = (work + kBlock - 1) / kBlock;
  if (want == 0) return 1;
  return want > kMaxBlocks ? kMaxBlocks : static_cast<int>(want);
}

}  // namespace

PointerKind ClassifyPointer(const void *ptr) {
  if (ptr == nullptr) return PointerKind::kUnknown;
  cudaPointerAttributes attr{};
  const cudaError_t rc = cudaPointerGetAttributes(&attr, ptr);
  if (rc != cudaSuccess) {
    // Not an error: an ordinary host allocation the driver never saw reports
    // cudaErrorInvalidValue on some versions. Clear it so the next real CUDA
    // call is not blamed for it.
    cudaGetLastError();
    return PointerKind::kHost;
  }
  switch (attr.type) {
    case cudaMemoryTypeDevice: return PointerKind::kDevice;
    case cudaMemoryTypeManaged: return PointerKind::kManaged;
    case cudaMemoryTypeHost: return PointerKind::kHost;
    default: return PointerKind::kHost;
  }
}

bool HashDeviceBuffer(const void *device_ptr, size_t bytes, uint64_t *out_hash) {
  if (device_ptr == nullptr || bytes == 0 || out_hash == nullptr) return false;
  unsigned long long *d_out = nullptr;
  NPEQ_CUDA_OK(cudaMalloc(&d_out, sizeof(unsigned long long)));
  cudaError_t rc = cudaMemset(d_out, 0, sizeof(unsigned long long));
  if (rc == cudaSuccess) {
    HashKernel<<<BlocksFor(bytes / 8 + 1), kBlock>>>(
        static_cast<const unsigned char *>(device_ptr), bytes, d_out);
    rc = cudaDeviceSynchronize();
  }
  unsigned long long host = 0;
  if (rc == cudaSuccess) {
    rc = cudaMemcpy(&host, d_out, sizeof(host), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_out);
  if (rc != cudaSuccess) {
    std::fprintf(stderr, "[device_probe] hash failed: %s\n",
                 cudaGetErrorString(rc));
    return false;
  }
  *out_hash = host;
  return true;
}

ByteCompareResult CompareDeviceBuffers(const void *a, const void *b,
                                       size_t bytes, size_t element_size) {
  ByteCompareResult result;
  result.total_bytes = bytes;
  if (a == nullptr || b == nullptr || bytes == 0) return result;

  DeviceCompare *d_out = nullptr;
  if (cudaMalloc(&d_out, sizeof(DeviceCompare)) != cudaSuccess) return result;
  DeviceCompare init{0, ~0ULL};
  cudaError_t rc =
      cudaMemcpy(d_out, &init, sizeof(init), cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) {
    CompareKernel<<<BlocksFor(bytes), kBlock>>>(
        static_cast<const unsigned char *>(a),
        static_cast<const unsigned char *>(b), bytes, d_out);
    rc = cudaDeviceSynchronize();
  }
  DeviceCompare host{0, ~0ULL};
  if (rc == cudaSuccess) {
    // The ONLY host transfer this comparison makes: 16 bytes of verdict, not
    // the buffers. Tagged TEST_HARNESS_TRANSFER by the caller.
    rc = cudaMemcpy(&host, d_out, sizeof(host), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_out);
  if (rc != cudaSuccess) {
    std::fprintf(stderr, "[device_probe] compare failed: %s\n",
                 cudaGetErrorString(rc));
    return result;
  }

  result.valid = true;
  result.differing_bytes = static_cast<size_t>(host.differing);
  result.identical_bytes = bytes - result.differing_bytes;
  if (host.first_diff != ~0ULL) {
    result.first_differing_byte = static_cast<size_t>(host.first_diff);
    result.first_differing_element =
        element_size > 0 ? result.first_differing_byte / element_size
                         : result.first_differing_byte;
  }
  return result;
}

namespace {

/** Min/max reduction. Integer-ordered floats would be faster; clarity wins. */
__global__ void RangeKernel(const float *__restrict__ data, size_t n,
                            float *__restrict__ out_min,
                            float *__restrict__ out_max) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  float lo = CUDART_INF_F;
  float hi = -CUDART_INF_F;
  for (size_t i = idx; i < n; i += stride) {
    const float v = data[i];
    lo = fminf(lo, v);
    hi = fmaxf(hi, v);
  }
  __shared__ float s_lo[kBlock];
  __shared__ float s_hi[kBlock];
  s_lo[threadIdx.x] = lo;
  s_hi[threadIdx.x] = hi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_lo[threadIdx.x] = fminf(s_lo[threadIdx.x], s_lo[threadIdx.x + s]);
      s_hi[threadIdx.x] = fmaxf(s_hi[threadIdx.x], s_hi[threadIdx.x + s]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMin(reinterpret_cast<int *>(out_min),
              __float_as_int(s_lo[0]) ^
                  ((__float_as_int(s_lo[0]) >> 31) | 0x80000000));
    atomicMax(reinterpret_cast<int *>(out_max),
              __float_as_int(s_hi[0]) ^
                  ((__float_as_int(s_hi[0]) >> 31) | 0x80000000));
  }
}

/** Undo the monotonic float->int mapping RangeKernel atomics used. */
float DecodeOrdered(int key) {
  const int f = key ^ (((~key) >> 31) | 0x80000000);
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

}  // namespace

bool ComputeDeviceRange(const float *device_ptr, size_t num_elements,
                        double *out_min, double *out_max) {
  if (device_ptr == nullptr || num_elements == 0) return false;
  int *d_pair = nullptr;
  NPEQ_CUDA_OK(cudaMalloc(&d_pair, 2 * sizeof(int)));
  // Ordered-int sentinels: +inf and -inf under the same mapping.
  const int init[2] = {0x7FFFFFFF, static_cast<int>(0x80000000)};
  cudaError_t rc =
      cudaMemcpy(d_pair, init, sizeof(init), cudaMemcpyHostToDevice);
  if (rc == cudaSuccess) {
    RangeKernel<<<BlocksFor(num_elements), kBlock>>>(
        device_ptr, num_elements, reinterpret_cast<float *>(d_pair),
        reinterpret_cast<float *>(d_pair + 1));
    rc = cudaDeviceSynchronize();
  }
  int host[2] = {0, 0};
  if (rc == cudaSuccess) {
    rc = cudaMemcpy(host, d_pair, sizeof(host), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_pair);
  if (rc != cudaSuccess) return false;
  *out_min = static_cast<double>(DecodeOrdered(host[0]));
  *out_max = static_cast<double>(DecodeOrdered(host[1]));
  return true;
}

bool FetchDeviceWindow(const void *device_ptr, size_t offset, size_t bytes,
                       void *host_out) {
  if (device_ptr == nullptr || host_out == nullptr || bytes == 0) return false;
  const cudaError_t rc =
      cudaMemcpy(host_out, static_cast<const unsigned char *>(device_ptr) + offset,
                 bytes, cudaMemcpyDeviceToHost);
  return rc == cudaSuccess;
}

bool CountBoundViolations(const float *a, const float *b, size_t num_elements,
                          double bound, unsigned long long *out_violations) {
  if (a == nullptr || b == nullptr || out_violations == nullptr) return false;
  unsigned long long *d_out = nullptr;
  NPEQ_CUDA_OK(cudaMalloc(&d_out, sizeof(unsigned long long)));
  cudaError_t rc = cudaMemset(d_out, 0, sizeof(unsigned long long));
  if (rc == cudaSuccess) {
    BoundKernel<<<BlocksFor(num_elements), kBlock>>>(a, b, num_elements, bound,
                                                     d_out);
    rc = cudaDeviceSynchronize();
  }
  unsigned long long host = 0;
  if (rc == cudaSuccess) {
    rc = cudaMemcpy(&host, d_out, sizeof(host), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_out);
  if (rc != cudaSuccess) return false;
  *out_violations = host;
  return true;
}

const char *ChunkRegimeName(ChunkRegime regime) {
  switch (regime) {
    case ChunkRegime::kConstant: return "constant";
    case ChunkRegime::kLinearRamp: return "linear-ramp";
    case ChunkRegime::kSmoothWave: return "smooth-wave";
    case ChunkRegime::kStepped: return "stepped";
    case ChunkRegime::kNoisyWave: return "noisy-wave";
    case ChunkRegime::kHighEntropy: return "high-entropy";
    case ChunkRegime::kSmallMagnitude: return "small-magnitude";
    case ChunkRegime::kMixed: return "mixed-bands";
  }
  return "unknown";
}

bool FillChunk(float *device_buf, size_t num_elements, ChunkRegime regime,
               uint32_t seed) {
  if (device_buf == nullptr || num_elements == 0) return false;
  FillKernel<<<BlocksFor(num_elements), kBlock>>>(
      device_buf, num_elements, static_cast<int>(regime), seed);
  NPEQ_CUDA_OK(cudaDeviceSynchronize());
  return true;
}

}  // namespace npeq
