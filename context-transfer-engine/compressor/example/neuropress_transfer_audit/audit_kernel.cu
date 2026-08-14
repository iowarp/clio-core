/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file audit_kernel.cu
 * @brief Device-side generation for the large-chunk transfer audit.
 *
 * Kernels only, no Clio includes -- clio_cte/compressor headers do not compile
 * under nvcc, the same split every sibling example uses.
 *
 * The payload is generated ON THE DEVICE and never has a host original. That
 * is a requirement of the audit, not a convenience: if the data were produced
 * on the host and uploaded, the very first H2D would be a payload-sized
 * transfer and the question "which transfers are redundant" would be answered
 * trivially and wrongly.
 */

#include <cuda_runtime.h>

namespace {

constexpr int kBlock = 256;

/**
 * Piecewise-constant plateaus: low byte entropy, so the codec wins and the
 * blob is stored COMPRESSED. That matters for this audit because the
 * not-beneficial path stores the original bytes instead, which adds a
 * payload-sized D2H that is a property of compression failing rather than of
 * the data path (compressor_runtime.cc:2685).
 *
 * The plateau is sized in ELEMENTS and scaled with the chunk so that a 256 MiB
 * chunk gets the same number of distinct plateaus as a 16 MiB one -- otherwise
 * the entropy, and therefore the algorithm the model picks, would drift with
 * chunk size and the comparison against the small-chunk run would not hold.
 */
__global__ void GenKernel(float *__restrict__ buf, size_t n, size_t plateau,
                          unsigned int seed) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = idx; i < n; i += stride) {
    buf[i] = static_cast<float>(((i / plateau) + seed) % 64) * 0.5f;
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
  return want > 4096 ? 4096 : static_cast<int>(want);
}

}  // namespace

extern "C" bool NpAuditGenerateChunk(float *device_buf, size_t num_elements,
                                     unsigned int seed) {
  if (device_buf == nullptr || num_elements == 0) return false;
  // 16384 plateaus regardless of chunk size; see GenKernel.
  size_t plateau = num_elements / 16384;
  if (plateau == 0) plateau = 1;
  GenKernel<<<BlocksFor(num_elements), kBlock>>>(device_buf, num_elements,
                                                 plateau, seed);
  return cudaDeviceSynchronize() == cudaSuccess;
}

extern "C" bool NpAuditCountMismatches(const float *a, const float *b,
                                       size_t num_elements,
                                       unsigned long long *out) {
  if (a == nullptr || b == nullptr || out == nullptr) return false;
  unsigned long long *d_out = nullptr;
  if (cudaMalloc(&d_out, sizeof(*d_out)) != cudaSuccess) return false;
  cudaError_t rc = cudaMemset(d_out, 0, sizeof(*d_out));
  if (rc == cudaSuccess) {
    MismatchKernel<<<BlocksFor(num_elements), kBlock>>>(a, b, num_elements,
                                                        d_out);
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
