/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_e2e_kernel.cu
 * @brief Device-side dataset generator for the NeuroPress GPU->HDF5 e2e run.
 *
 * Held in its own translation unit for the same reason
 * neuropress_gpu_demo_kernel.cu is: the clio_cte/compressor headers the
 * driver needs do not compile cleanly under nvcc, so the kernel is built as
 * a small shared CUDA library and the driver stays a plain host TU.
 *
 * The dataset deliberately SWEEPS compressibility across its chunks rather
 * than repeating one pattern. A uniform dataset would have every chunk land
 * on the same configuration, and a selection comparison over 256 identical
 * decisions proves almost nothing -- it would pass just as happily against a
 * model stuck on a constant. Sweeping means entropy, MAD and curvature all
 * move across the run, the network is asked a genuinely different question
 * per chunk, and the comparison against native has something to disagree
 * about.
 *
 * Every value is a pure function of its global element index, with no RNG
 * state and no dependence on launch geometry, so the buffer is bit-identical
 * on every run and on both sides of the comparison.
 */

#include <cuda_runtime.h>

namespace {

/** Integer hash (Thomas Wang's 32-bit mix). Deterministic and stateless, so
 *  the "noise" is reproducible to the bit on any device or launch shape. */
__device__ __forceinline__ unsigned int HashU32(unsigned int x) {
  x = (x ^ 61u) ^ (x >> 16);
  x *= 9u;
  x = x ^ (x >> 4);
  x *= 0x27d4eb2du;
  x = x ^ (x >> 15);
  return x;
}

/** Hash to a float in [-1, 1). */
__device__ __forceinline__ float HashUnit(unsigned int x) {
  return (static_cast<float>(HashU32(x) & 0xFFFFFFu) / 8388608.0f) - 1.0f;
}

/**
 * One element of chunk `chunk` at within-chunk index `i`.
 *
 * The chunk index drives a noise amplitude ramp, so early chunks are almost
 * constant (the compressors' easy case), middle chunks are smooth fields
 * with mild texture, and late chunks approach incompressible noise.
 */
__device__ __forceinline__ float ElementValue(unsigned int chunk,
                                              unsigned int i,
                                              unsigned int num_chunks) {
  const float t = static_cast<float>(chunk) /
                  static_cast<float>(num_chunks > 1 ? num_chunks - 1 : 1);

  // Smooth, highly compressible base field: a slow sinusoid plus a coarse
  // staircase, so even the noisiest chunks keep some exploitable structure.
  const float phase = static_cast<float>(i) * 0.0001f +
                      static_cast<float>(chunk) * 0.37f;
  const float smooth = __sinf(phase) + 0.5f * __sinf(phase * 0.021f);
  const float stair = static_cast<float>((i >> 12) & 0x3F) * 0.03125f;

  // Amplitude ramp: t^2 keeps the low end genuinely flat rather than
  // spending half the run already noisy.
  const float noise_amp = t * t;
  const float noise = HashUnit(i * 2654435761u + chunk * 40503u);

  return (1.0f - noise_amp) * (smooth + stair) + noise_amp * noise * 8.0f;
}

__global__ void GenerateKernel(float *__restrict__ out, unsigned long long n,
                               unsigned int elems_per_chunk,
                               unsigned int num_chunks) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    const unsigned int chunk =
        static_cast<unsigned int>(idx / elems_per_chunk);
    const unsigned int i = static_cast<unsigned int>(idx % elems_per_chunk);
    out[idx] = ElementValue(chunk, i, num_chunks);
  }
}

}  // namespace

/**
 * @brief Fill @p d_out with the e2e dataset, entirely on-device.
 *
 * @param d_out           Device buffer of @p num_elems floats.
 * @param num_elems       Total element count (num_chunks * elems_per_chunk).
 * @param elems_per_chunk Elements in one chunk (chunk_bytes / sizeof(float)).
 * @param num_chunks      Number of chunks the dataset is split into.
 */
void GenerateNeuroPressE2EDataset(float *d_out, unsigned long long num_elems,
                                  unsigned int elems_per_chunk,
                                  unsigned int num_chunks) {
  const int threads = 256;
  // Grid-stride loop, so the launch shape never changes the values written.
  int blocks = 1024;
  GenerateKernel<<<blocks, threads>>>(d_out, num_elems, elems_per_chunk,
                                      num_chunks);
}
