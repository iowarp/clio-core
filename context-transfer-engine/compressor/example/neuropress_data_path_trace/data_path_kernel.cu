/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file data_path_kernel.cu
 * @brief Device-side generation and verification for the data-path trace.
 *
 * Kernels only, no Clio includes -- clio_cte/compressor headers do not compile
 * under nvcc, the same split the sibling examples use.
 */

#include <cuda_runtime.h>

#include <cstdio>

namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ unsigned long long Mix64(unsigned long long h) {
  h ^= h >> 30;
  h *= 0xBF58476D1CE4E5B9ULL;
  h ^= h >> 27;
  h *= 0x94D049BB133111EBULL;
  h ^= h >> 31;
  return h;
}

/**
 * Two regimes, because the data path FORKS on whether compression paid off.
 *
 * regime 0 (compressible): piecewise-constant plateaus. Low byte entropy, so
 *   the codec wins and the chunk is stored compressed -- which is the path the
 *   residency question is about.
 * regime 1 (incompressible): smooth carrier plus per-index jitter. Float32
 *   mantissas make this high-entropy at BYTE level, the codec comes out larger
 *   than the input, and the runtime discards its output and stores the original
 *   (compressor_runtime.cc:2685). That is a different data path and worth being
 *   able to trace deliberately rather than stumbling into.
 *
 * Both are deterministic in the element index, so a chunk id always yields the
 * same bytes.
 */
__global__ void GenKernel(float *__restrict__ buf, size_t n, unsigned int seed,
                          int regime) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = idx; i < n; i += stride) {
    if (regime == 0) {
      buf[i] = static_cast<float>(((i / 4096) + seed) % 64) * 0.5f;
    } else {
      const unsigned long long h =
          Mix64(i * 0x2545F491ULL + seed * 0x9E3779B9ULL);
      const float jitter =
          static_cast<float>((h >> 40) & 0xFFFF) / 65536.0f - 0.5f;
      buf[i] = 50.0f * sinf(static_cast<float>(i) * 1e-4f) + 0.25f * jitter;
    }
  }
}

__global__ void MismatchKernel(const float *__restrict__ a,
                               const float *__restrict__ b, size_t n,
                               unsigned long long *__restrict__ out) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (size_t i = idx; i < n; i += stride) {
    if (a[i] != b[i]) ++local;
  }
  if (local != 0) atomicAdd(out, local);
}

int BlocksFor(size_t work) {
  const size_t want = (work + kBlock - 1) / kBlock;
  if (want == 0) return 1;
  return want > 1024 ? 1024 : static_cast<int>(want);
}

}  // namespace

extern "C" bool NpPathGenerateChunk(float *device_buf, size_t num_elements,
                                    unsigned int seed, int regime) {
  if (device_buf == nullptr || num_elements == 0) return false;
  GenKernel<<<BlocksFor(num_elements), kBlock>>>(device_buf, num_elements, seed,
                                                 regime);
  return cudaDeviceSynchronize() == cudaSuccess;
}

extern "C" bool NpPathCountMismatches(const float *a, const float *b,
                                      size_t num_elements,
                                      unsigned long long *out) {
  if (a == nullptr || b == nullptr || out == nullptr) return false;
  unsigned long long *d_out = nullptr;
  if (cudaMalloc(&d_out, sizeof(*d_out)) != cudaSuccess) return false;
  cudaError_t rc = cudaMemset(d_out, 0, sizeof(*d_out));
  if (rc == cudaSuccess) {
    MismatchKernel<<<BlocksFor(num_elements), kBlock>>>(a, b, num_elements, d_out);
    rc = cudaDeviceSynchronize();
  }
  unsigned long long host = 0;
  if (rc == cudaSuccess) {
    rc = cudaMemcpy(&host, d_out, sizeof(host), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_out);
  if (rc != cudaSuccess) return false;
  *out = host;
  return true;
}
