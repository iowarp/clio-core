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
 * @file grayscott_sim.cu
 * @brief Kernels and host driver for the 3D Gray-Scott workload (#693).
 *
 * Ported from NeuroPress's src/gray-scott/. See grayscott_kernels.cuh on why
 * the arithmetic here is fixed by parity rather than free to tidy.
 */

#include "grayscott_kernels.cuh"
#include "grayscott_sim.h"

#include <cstdio>
#include <utility>

namespace clio::cte::compressor::grayscott {

__global__ void GrayScottInitKernel(float *u, float *v, int L, float noise,
                                    unsigned long long seed) {
  const size_t n = static_cast<size_t>(L) * L * L;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    int x = static_cast<int>(i % L);
    int y = static_cast<int>((i / L) % L);
    int z = static_cast<int>(i / (static_cast<size_t>(L) * L));

    float ui = 1.0f;
    float vi = 0.0f;

    // Centre perturbation: a 12-cell cube, i.e. [half-6, half+6) per axis.
    // This seeds the whole pattern -- without it the field stays uniform and
    // every codec reports the same trivial ratio.
    const int half = L / 2;
    if (x >= half - 6 && x < half + 6 && y >= half - 6 && y < half + 6 &&
        z >= half - 6 && z < half + 6) {
      ui = 0.25f;
      vi = 0.33f;
    }

    ui += noise * (Rand01(seed ^ static_cast<unsigned long long>(i)) * 2.0f -
                   1.0f);

    u[i] = ui;
    v[i] = vi;
  }
}

__global__ void GrayScottStepKernel(const float *u_in, const float *v_in,
                                    float *u_out, float *v_out, int L,
                                    float Du, float Dv, float F, float k,
                                    float dt, float noise,
                                    unsigned long long noise_seed) {
  const size_t n = static_cast<size_t>(L) * L * L;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    int x = static_cast<int>(i % L);
    int y = static_cast<int>((i / L) % L);
    int z = static_cast<int>(i / (static_cast<size_t>(L) * L));

    const float uc = u_in[i];
    const float vc = v_in[i];

    const size_t xm = Index(Wrap(x - 1, L), y, z, L);
    const size_t xp = Index(Wrap(x + 1, L), y, z, L);
    const size_t ym = Index(x, Wrap(y - 1, L), z, L);
    const size_t yp = Index(x, Wrap(y + 1, L), z, L);
    const size_t zm = Index(x, y, Wrap(z - 1, L), L);
    const size_t zp = Index(x, y, Wrap(z + 1, L), L);

    // Averaged, not summed: the /6 folds the stencil normalization into the
    // Laplacian so Du/Dv keep the reference implementation's meaning.
    const float lap_u = (u_in[xm] + u_in[xp] + u_in[ym] + u_in[yp] + u_in[zm] +
                         u_in[zp] - 6.0f * uc) / 6.0f;
    const float lap_v = (v_in[xm] + v_in[xp] + v_in[ym] + v_in[yp] + v_in[zm] +
                         v_in[zp] - 6.0f * vc) / 6.0f;

    const float uvv = uc * vc * vc;
    float du = Du * lap_u - uvv + F * (1.0f - uc);
    const float dv = Dv * lap_v + uvv - (F + k) * vc;

    du += noise *
          (Rand01(noise_seed ^ static_cast<unsigned long long>(i)) * 2.0f -
           1.0f);

    u_out[i] = uc + du * dt;
    v_out[i] = vc + dv * dt;
  }
}


// ---------------------------------------------------------------------------
// 2D model (kClio2D)
// ---------------------------------------------------------------------------

__global__ void GrayScottInit2DKernel(float *u, float *v, int nx, int ny) {
  const size_t n = static_cast<size_t>(nx) * ny;
  const int cx = nx / 2;
  const int cy = ny / 2;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const int x = static_cast<int>(i % nx);
    const int y = static_cast<int>(i / nx);
    // [c-3, c+3) per axis -- a 6x6 square, matching the host loops in
    // gray_scott_gpu_test.cu.
    const bool seed = (x >= cx - 3 && x < cx + 3 && y >= cy - 3 && y < cy + 3);
    u[i] = 1.0f;
    v[i] = seed ? 1.0f : 0.0f;
  }
}

__global__ void GrayScottStep2DKernel(const float *u_in, const float *v_in,
                                      float *u_out, float *v_out, int nx,
                                      int ny, float Du, float Dv, float F,
                                      float k, float dt) {
  const size_t n = static_cast<size_t>(nx) * ny;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    const int x = static_cast<int>(i % nx);
    const int y = static_cast<int>(i / nx);
    const int xm = (x == 0) ? (nx - 1) : (x - 1);
    const int xp = (x == nx - 1) ? 0 : (x + 1);
    const int ym = (y == 0) ? (ny - 1) : (y - 1);
    const int yp = (y == ny - 1) ? 0 : (y + 1);

    const float uc = u_in[i];
    const float vc = v_in[i];

    // Raw 4-point Laplacian: no division. See the header -- this is the
    // existing harnesses' trajectory, not an oversight to correct.
    const size_t row = static_cast<size_t>(y) * nx;
    const float lap_u = u_in[row + xm] + u_in[row + xp] +
                        u_in[static_cast<size_t>(ym) * nx + x] +
                        u_in[static_cast<size_t>(yp) * nx + x] - 4.0f * uc;
    const float lap_v = v_in[row + xm] + v_in[row + xp] +
                        v_in[static_cast<size_t>(ym) * nx + x] +
                        v_in[static_cast<size_t>(yp) * nx + x] - 4.0f * vc;

    const float uvv = uc * vc * vc;
    u_out[i] = uc + dt * (Du * lap_u - uvv + F * (1.0f - uc));
    v_out[i] = vc + dt * (Dv * lap_v + uvv - (F + k) * vc);
  }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

SimSettings SimSettings::NeuroPress3D(int L) {
  SimSettings s;
  s.model = Model::kNeuroPress3D;
  s.nx = s.ny = s.nz = L;
  s.Du = 0.05f;
  s.Dv = 0.1f;
  s.F = 0.04f;
  s.k = 0.06075f;
  s.dt = 0.2f;
  s.noise = 0.0f;
  return s;
}

SimSettings SimSettings::Clio2D(int nx, int ny) {
  SimSettings s;
  s.model = Model::kClio2D;
  s.nx = nx;
  s.ny = ny;
  s.nz = 1;
  s.Du = 0.16f;
  s.Dv = 0.08f;
  s.F = 0.055f;
  s.k = 0.062f;
  s.dt = 1.0f;
  s.noise = 0.0f;  // the 2D model has no noise term at all
  return s;
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

struct Simulation::Impl {
  SimSettings settings;
  float *d_u = nullptr;
  float *d_v = nullptr;
  float *d_u_scratch = nullptr;
  float *d_v_scratch = nullptr;
  size_t n = 0;
  size_t nbytes = 0;
  int step = 0;
  bool valid = false;
};

namespace {

bool Check(cudaError_t err, const char *what) {
  if (err == cudaSuccess) return true;
  std::fprintf(stderr, "grayscott: %s failed: %s\n", what,
               cudaGetErrorString(err));
  return false;
}

}  // namespace

Simulation::Simulation(const SimSettings &settings) : impl_(new Impl) {
  impl_->settings = settings;
  impl_->n = static_cast<size_t>(settings.nx) * settings.ny * settings.nz;
  impl_->nbytes = impl_->n * sizeof(float);

  // All four or none: a partially allocated sim would run and produce
  // garbage rather than fail, which is the worse outcome for a benchmark.
  const bool ok =
      Check(cudaMalloc(&impl_->d_u, impl_->nbytes), "cudaMalloc(u)") &&
      Check(cudaMalloc(&impl_->d_v, impl_->nbytes), "cudaMalloc(v)") &&
      Check(cudaMalloc(&impl_->d_u_scratch, impl_->nbytes),
            "cudaMalloc(u scratch)") &&
      Check(cudaMalloc(&impl_->d_v_scratch, impl_->nbytes),
            "cudaMalloc(v scratch)");
  impl_->valid = ok;
  if (!ok) {
    cudaFree(impl_->d_u);
    cudaFree(impl_->d_v);
    cudaFree(impl_->d_u_scratch);
    cudaFree(impl_->d_v_scratch);
    impl_->d_u = impl_->d_v = impl_->d_u_scratch = impl_->d_v_scratch = nullptr;
  }
}

Simulation::~Simulation() {
  if (impl_ != nullptr) {
    cudaFree(impl_->d_u);
    cudaFree(impl_->d_v);
    cudaFree(impl_->d_u_scratch);
    cudaFree(impl_->d_v_scratch);
    delete impl_;
  }
}

bool Simulation::Valid() const { return impl_ != nullptr && impl_->valid; }

bool Simulation::Init() {
  if (!Valid()) return false;
  const SimSettings &s = impl_->settings;
  const int grid = GridSize(impl_->n);
  if (s.model == Model::kClio2D) {
    GrayScottInit2DKernel<<<grid, kBlockSize>>>(impl_->d_u, impl_->d_v, s.nx,
                                                s.ny);
  } else {
    GrayScottInitKernel<<<grid, kBlockSize>>>(
        impl_->d_u, impl_->d_v, s.nx, s.noise,
        static_cast<unsigned long long>(s.seed));
  }
  if (!Check(cudaGetLastError(), "init kernel")) return false;
  impl_->step = 0;
  return true;
}

bool Simulation::Run(int steps) {
  if (!Valid()) return false;
  if (steps <= 0) return true;
  const SimSettings &s = impl_->settings;
  const int grid = GridSize(impl_->n);

  for (int i = 0; i < steps; ++i) {
    if (s.model == Model::kClio2D) {
      GrayScottStep2DKernel<<<grid, kBlockSize>>>(
          impl_->d_u, impl_->d_v, impl_->d_u_scratch, impl_->d_v_scratch, s.nx,
          s.ny, s.Du, s.Dv, s.F, s.k, s.dt);
    } else {
      // Mixing the absolute step index (not i) keeps the noise stream a
      // function of simulation time, so Run(10) and Run(5) twice agree.
      const unsigned long long noise_seed =
          static_cast<unsigned long long>(s.seed) ^
          (static_cast<unsigned long long>(impl_->step + i) << 20);

      GrayScottStepKernel<<<grid, kBlockSize>>>(
          impl_->d_u, impl_->d_v, impl_->d_u_scratch, impl_->d_v_scratch, s.nx,
          s.Du, s.Dv, s.F, s.k, s.dt, s.noise, noise_seed);
    }

    // Swap, never copy: the freshly written scratch becomes current.
    std::swap(impl_->d_u, impl_->d_u_scratch);
    std::swap(impl_->d_v, impl_->d_v_scratch);
  }

  if (!Check(cudaGetLastError(), "step kernel")) return false;
  impl_->step += steps;
  return true;
}

float *Simulation::DeviceU() const { return Valid() ? impl_->d_u : nullptr; }
float *Simulation::DeviceV() const { return Valid() ? impl_->d_v : nullptr; }
size_t Simulation::NumElems() const { return impl_ != nullptr ? impl_->n : 0; }
size_t Simulation::NumBytes() const {
  return impl_ != nullptr ? impl_->nbytes : 0;
}
int Simulation::Step() const { return impl_ != nullptr ? impl_->step : 0; }

}  // namespace clio::cte::compressor::grayscott
