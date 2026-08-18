/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file grayscott_kernels.cuh
 * @brief Device helpers and kernels for the 3D Gray-Scott workload (#693).
 *
 * Separate from grayscott_sim.h so the parity harness can call these kernels
 * directly, side by side with NeuroPress's gs_init_kernel / gs_step_kernel,
 * without going through the host-side Simulation wrapper. Everything here is
 * namespaced, so including this and upstream's gray_scott_gpu.cuh in one
 * translation unit is well defined.
 *
 * Every numeric choice below is load-bearing for parity -- the divide-by-6
 * Laplacian normalization, the splitmix64 constants, the 12^3 cube bounds and
 * the per-step seed mix all have to match upstream bit-for-bit, because a
 * float trajectory diverges visibly after a few hundred steps if any of them
 * drifts. Change them only alongside a parity run.
 */

#ifndef CLIO_CTE_COMPRESSOR_GENERATOR_GRAYSCOTT_GRAYSCOTT_KERNELS_CUH_
#define CLIO_CTE_COMPRESSOR_GENERATOR_GRAYSCOTT_GRAYSCOTT_KERNELS_CUH_

#include <cuda_runtime.h>

namespace clio::cte::compressor::grayscott {

/** Threads per block for both kernels. */
constexpr int kBlockSize = 256;

/**
 * @brief Grid size for a grid-stride launch over `n` cells.
 *
 * Capped at 65535 blocks. Both kernels stride, so the cap changes how much
 * work each thread does but never which thread owns which cell -- results are
 * independent of the launch geometry.
 */
inline int GridSize(size_t n) {
  size_t blocks = (n + kBlockSize - 1) / kBlockSize;
  return blocks < 65535 ? static_cast<int>(blocks) : 65535;
}

/** @brief Flat index into the L^3 grid, x fastest (ZYX order). */
__device__ __host__ inline size_t Index(int x, int y, int z, int L) {
  return static_cast<size_t>(x) + static_cast<size_t>(y) * L +
         static_cast<size_t>(z) * L * L;
}

/** @brief Periodic wrap for a coordinate at most one step out of range.
 *  Branch-select rather than modulo -- the neighbours below are never more
 *  than one cell outside, so the general case would be wasted work. */
__device__ inline int Wrap(int v, int L) {
  return (v < 0) ? v + L : ((v >= L) ? v - L : v);
}

/** @brief splitmix64 hashed to [0, 1), using the top 24 bits of the mixed
 *  word. Reproducible across runs and machines, unlike cuRAND state, which is
 *  why the workload can be compared against upstream at all. */
__device__ inline float Rand01(unsigned long long seed) {
  seed ^= seed >> 33;
  seed *= 0xff51afd7ed558ccdULL;
  seed ^= seed >> 33;
  seed *= 0xc4ceb9fe1a85ec53ULL;
  seed ^= seed >> 33;
  return static_cast<float>(seed & 0xFFFFFFu) / static_cast<float>(0x1000000u);
}

/**
 * @brief Initial condition: U=1, V=0 everywhere, with a 12^3 cube of
 * U=0.25 / V=0.33 centred on the grid, then noise added to U.
 */
__global__ void GrayScottInitKernel(float *u, float *v, int L, float noise,
                                    unsigned long long seed);

/**
 * @brief One forward-Euler step: fused 6-point Laplacian, reaction, update.
 */
__global__ void GrayScottStepKernel(const float *u_in, const float *v_in,
                                    float *u_out, float *v_out, int L,
                                    float Du, float Dv, float F, float k,
                                    float dt, float noise,
                                    unsigned long long noise_seed);

/**
 * @brief 2D initial condition: U=1 everywhere, V=0 except a 6x6 square at the
 * centre where V=1.
 *
 * The pre-existing harnesses build this on the host and memcpy it in. Doing it
 * in a kernel is bit-identical -- 1.0f and 0.0f are exact -- and keeps the
 * field GPU-resident from the first instruction, which is the property the
 * NeuroPress workload is measuring.
 */
__global__ void GrayScottInit2DKernel(float *u, float *v, int nx, int ny);

/**
 * @brief One 2D forward-Euler step: 4-point Laplacian, UNNORMALIZED.
 *
 * Mirrors GsStepKernel in adapter/kvhdf5/test/e2e/gray_scott_gpu_test.cu
 * exactly, including the missing /4 on the Laplacian and the `dt * (...)`
 * association. Those are not defects to clean up here: they define the
 * trajectory the existing e2e tests assert against, so "tidying" them would
 * change every expected value in the tree.
 */
__global__ void GrayScottStep2DKernel(const float *u_in, const float *v_in,
                                      float *u_out, float *v_out, int nx,
                                      int ny, float Du, float Dv, float F,
                                      float k, float dt);

}  // namespace clio::cte::compressor::grayscott

#endif  // CLIO_CTE_COMPRESSOR_GENERATOR_GRAYSCOTT_GRAYSCOTT_KERNELS_CUH_
