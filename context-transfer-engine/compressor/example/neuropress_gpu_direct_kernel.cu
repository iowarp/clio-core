/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_gpu_direct_kernel.cu
 * @brief Data generation for the direct GPU-buffer compression example.
 *
 * Own translation unit with NO Clio includes: clio_cte/compressor headers do
 * not compile cleanly under nvcc, so device code and Clio client code stay
 * in separate TUs linked through a plain declaration -- the same split used
 * for core_client.h/.cc and the HDF5 VOL demo.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

/**
 * Five statistical regimes cycling per chunk, so NeuroPress has genuinely
 * different data to reason about rather than one operating point. Written
 * directly into the device buffer -- the host never sees this data.
 */
__global__ void FillRegimes(float *buf, size_t num_elems,
                            size_t elems_per_chunk) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_elems) return;
  size_t chunk = i / elems_per_chunk;
  size_t local = i % elems_per_chunk;
  switch (chunk % 5) {
    case 0:  // constant -- trivially compressible
      buf[i] = 42.0f;
      break;
    case 1: {  // smooth sine -- low entropy
      float p = 6.28318f * static_cast<float>(local) /
                static_cast<float>(elems_per_chunk);
      buf[i] = sinf(p * 4.0f);
      break;
    }
    case 2:  // stepped ramp -- moderate, byte planes differ a lot
      buf[i] = static_cast<float>((local / 512) % 1024) * 0.25f;
      break;
    case 3: {  // smooth + small noise
      uint32_t x = static_cast<uint32_t>(i) * 747796405u + 2891336453u;
      x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
      float n = static_cast<float>(x & 0xFFFF) / 65535.0f - 0.5f;
      float p = 6.28318f * static_cast<float>(local) /
                static_cast<float>(elems_per_chunk);
      buf[i] = sinf(p * 2.0f) + 0.05f * n;
      break;
    }
    default: {  // hash noise -- effectively incompressible
      uint32_t x = static_cast<uint32_t>(i) * 2654435761u + 0x9E3779B9u;
      x ^= x >> 15; x *= 0x85EBCA6Bu; x ^= x >> 13;
      buf[i] = static_cast<float>(x) / 4294967295.0f;
      break;
    }
  }
}

__global__ void CompareBuffers(const float *a, const float *b,
                               size_t num_elems, unsigned long long *mismatch) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= num_elems) return;
  if (a[i] != b[i]) atomicAdd(mismatch, 1ull);
}

}  // namespace

void LaunchFillRegimes(float *d_buf, size_t num_elems, size_t elems_per_chunk) {
  const int threads = 256;
  const int blocks = static_cast<int>((num_elems + threads - 1) / threads);
  FillRegimes<<<blocks, threads>>>(d_buf, num_elems, elems_per_chunk);
  cudaDeviceSynchronize();
}

/** Verify on-device: the data never has to come back to the host at all. */
unsigned long long CountMismatchesOnDevice(const float *d_a, const float *d_b,
                                           size_t num_elems) {
  unsigned long long *d_count = nullptr;
  if (cudaMalloc(&d_count, sizeof(unsigned long long)) != cudaSuccess) {
    return ~0ull;
  }
  cudaMemset(d_count, 0, sizeof(unsigned long long));
  const int threads = 256;
  const int blocks = static_cast<int>((num_elems + threads - 1) / threads);
  CompareBuffers<<<blocks, threads>>>(d_a, d_b, num_elems, d_count);
  cudaDeviceSynchronize();
  unsigned long long h_count = ~0ull;
  cudaMemcpy(&h_count, d_count, sizeof(h_count), cudaMemcpyDeviceToHost);
  cudaFree(d_count);
  return h_count;
}
