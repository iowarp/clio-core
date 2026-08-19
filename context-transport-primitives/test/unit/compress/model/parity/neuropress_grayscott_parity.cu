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
 * @file neuropress_grayscott_parity.cu
 * @brief Differential parity for the ported 3D Gray-Scott workload (#693).
 *
 * Compiles NeuroPress's own src/gray-scott/gray_scott_gpu.cu beside Clio's
 * port and runs both on identical settings, then compares the U and V fields
 * BIT-FOR-BIT. Not approximately: a reaction-diffusion trajectory is chaotic
 * in the (F,k) regimes this workload cares about, so a single-ULP difference
 * in the Laplacian normalization is invisible after one step and obvious
 * after two hundred. Exact equality is the only threshold that means
 * anything here, and it is achievable because both sides run the same
 * arithmetic in the same order on the same hardware.
 *
 * Covers, per configuration: the initial condition on its own (step 0), a
 * single step, and long enough runs for divergence to amplify -- across all
 * four documented (F,k) pattern regimes, with and without the noise term
 * that exercises the splitmix64 stream.
 *
 * Also pins two properties of Clio's driver that upstream gets for free from
 * its structure and that a future refactor could silently break:
 *   - launch geometry is not observable (grid-stride, one cell per thread);
 *   - Run(a); Run(b) == Run(a+b), i.e. the per-step noise seed is a function
 *     of absolute simulation time, not of loop position.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// NeuroPress's own kernels.
#include "gray-scott/gray_scott_gpu.cuh"

// Clio's port.
#include "grayscott_kernels.cuh"

namespace gs = clio::cte::compressor::grayscott;

namespace {

long g_checks = 0;
int g_failures = 0;

#define CUDA_OK(call)                                                       \
  do {                                                                      \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
      std::printf("  CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                  cudaGetErrorString(_e));                                  \
      ++g_failures;                                                         \
      return;                                                               \
    }                                                                       \
  } while (0)

/** One simulation configuration. */
struct Config {
  const char *regime;
  int L;
  float F;
  float k;
  float noise;
  int seed;
  int steps;
};

/** Four device buffers: current pair plus the double-buffered scratch pair. */
struct Fields {
  float *u = nullptr;
  float *v = nullptr;
  float *u2 = nullptr;
  float *v2 = nullptr;

  bool Alloc(size_t bytes) {
    return cudaMalloc(&u, bytes) == cudaSuccess &&
           cudaMalloc(&v, bytes) == cudaSuccess &&
           cudaMalloc(&u2, bytes) == cudaSuccess &&
           cudaMalloc(&v2, bytes) == cudaSuccess;
  }
  void Free() {
    cudaFree(u);
    cudaFree(v);
    cudaFree(u2);
    cudaFree(v2);
    u = v = u2 = v2 = nullptr;
  }
};

/** Upstream's grid sizing, reproduced so the launch matches its driver. */
int UpstreamGrid(size_t n) {
  size_t blocks = (n + 256 - 1) / 256;
  return blocks < 65535 ? static_cast<int>(blocks) : 65535;
}

/** Advance the upstream implementation: init, then `steps` steps. */
void RunUpstream(Fields &f, const Config &c, size_t n, int grid) {
  gs_init_kernel<<<grid, 256>>>(f.u, f.v, c.L, c.noise,
                                static_cast<unsigned long long>(c.seed));
  for (int i = 0; i < c.steps; ++i) {
    unsigned long long noise_seed = static_cast<unsigned long long>(c.seed) ^
                                    (static_cast<unsigned long long>(i) << 20);
    gs_step_kernel<<<grid, 256>>>(f.u, f.v, f.u2, f.v2, c.L, 0.05f, 0.1f, c.F,
                                  c.k, 0.2f, c.noise, noise_seed);
    std::swap(f.u, f.u2);
    std::swap(f.v, f.v2);
  }
  (void)n;
}

/** Advance Clio's port through the same sequence. */
void RunPort(Fields &f, const Config &c, size_t n, int grid) {
  gs::GrayScottInitKernel<<<grid, gs::kBlockSize>>>(
      f.u, f.v, c.L, c.noise, static_cast<unsigned long long>(c.seed));
  for (int i = 0; i < c.steps; ++i) {
    unsigned long long noise_seed = static_cast<unsigned long long>(c.seed) ^
                                    (static_cast<unsigned long long>(i) << 20);
    gs::GrayScottStepKernel<<<grid, gs::kBlockSize>>>(
        f.u, f.v, f.u2, f.v2, c.L, 0.05f, 0.1f, c.F, c.k, 0.2f, c.noise,
        noise_seed);
    std::swap(f.u, f.u2);
    std::swap(f.v, f.v2);
  }
  (void)n;
}

/**
 * Compare two device buffers bit-for-bit.
 *
 * Compares the raw 32-bit patterns, not the float values: that treats NaN as
 * equal to the identical NaN (which is what "the two ports agree" means) and
 * refuses to call -0.0 equal to +0.0, which would hide a sign-handling
 * divergence in the reaction term.
 */
bool BitEqual(const float *da, const float *db, size_t n, const char *field,
              const char *label, double *out_max_abs) {
  std::vector<uint32_t> ha(n), hb(n);
  cudaMemcpy(ha.data(), da, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), db, n * sizeof(float), cudaMemcpyDeviceToHost);

  size_t differing = 0;
  size_t first = 0;
  double max_abs = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (ha[i] != hb[i]) {
      if (differing == 0) first = i;
      ++differing;
      float fa, fb;
      std::memcpy(&fa, &ha[i], 4);
      std::memcpy(&fb, &hb[i], 4);
      double d = static_cast<double>(fa) - static_cast<double>(fb);
      if (d < 0) d = -d;
      if (d > max_abs) max_abs = d;
    }
  }
  if (out_max_abs != nullptr && max_abs > *out_max_abs) *out_max_abs = max_abs;

  ++g_checks;
  if (differing != 0) {
    float fa, fb;
    std::memcpy(&fa, &ha[first], 4);
    std::memcpy(&fb, &hb[first], 4);
    std::printf(
        "  FAIL %s %s: %zu/%zu elements differ, first at %zu "
        "(upstream %.9g, clio %.9g), max |delta| %.3g\n",
        label, field, differing, n, first, static_cast<double>(fa),
        static_cast<double>(fb), max_abs);
    ++g_failures;
    return false;
  }
  return true;
}

/** Run one configuration on both implementations and diff both fields. */
void ParityCase(const Config &c) {
  const size_t n = static_cast<size_t>(c.L) * c.L * c.L;
  const size_t bytes = n * sizeof(float);
  const int grid = UpstreamGrid(n);

  Fields up, port;
  if (!up.Alloc(bytes) || !port.Alloc(bytes)) {
    std::printf("  SKIP %s L=%d: allocation failed\n", c.regime, c.L);
    up.Free();
    port.Free();
    return;
  }

  RunUpstream(up, c, n, grid);
  RunPort(port, c, n, grid);
  CUDA_OK(cudaDeviceSynchronize());

  char label[128];
  std::snprintf(label, sizeof(label), "%s L=%d steps=%d noise=%.3g seed=%d",
                c.regime, c.L, c.steps, static_cast<double>(c.noise), c.seed);

  double max_abs = 0.0;
  const bool ok_u = BitEqual(up.u, port.u, n, "U", label, &max_abs);
  const bool ok_v = BitEqual(up.v, port.v, n, "V", label, &max_abs);
  if (ok_u && ok_v) std::printf("  ok   %s\n", label);

  up.Free();
  port.Free();
}

/**
 * Launch geometry must not be observable. Both kernels grid-stride over a
 * flat index and every cell is written by exactly one thread, so halving the
 * grid may only change how many cells each thread handles.
 */
void GeometryInvarianceCase(const Config &c) {
  const size_t n = static_cast<size_t>(c.L) * c.L * c.L;
  const size_t bytes = n * sizeof(float);

  Fields wide, narrow;
  if (!wide.Alloc(bytes) || !narrow.Alloc(bytes)) {
    wide.Free();
    narrow.Free();
    return;
  }

  RunPort(wide, c, n, UpstreamGrid(n));
  RunPort(narrow, c, n, 37);  // deliberately tiny and not a divisor
  CUDA_OK(cudaDeviceSynchronize());

  char label[128];
  std::snprintf(label, sizeof(label), "geometry-invariance %s L=%d steps=%d",
                c.regime, c.L, c.steps);
  double max_abs = 0.0;
  const bool ok_u = BitEqual(wide.u, narrow.u, n, "U", label, &max_abs);
  const bool ok_v = BitEqual(wide.v, narrow.v, n, "V", label, &max_abs);
  if (ok_u && ok_v) std::printf("  ok   %s\n", label);

  wide.Free();
  narrow.Free();
}

/**
 * Run(a) then Run(b) must equal Run(a+b). This holds only because the noise
 * seed mixes the ABSOLUTE step counter; a port that mixed the loop variable
 * instead would pass every single-call test above and quietly desynchronize
 * the moment a caller checkpointed mid-run.
 */
void ResumeEquivalenceCase(const Config &c, int split) {
  const size_t n = static_cast<size_t>(c.L) * c.L * c.L;
  const size_t bytes = n * sizeof(float);
  const int grid = UpstreamGrid(n);

  Fields once, twice;
  if (!once.Alloc(bytes) || !twice.Alloc(bytes)) {
    once.Free();
    twice.Free();
    return;
  }

  RunPort(once, c, n, grid);

  // Same total steps, taken in two calls via the public driver semantics:
  // init, `split` steps, then the remainder, continuing the step counter.
  gs::GrayScottInitKernel<<<grid, gs::kBlockSize>>>(
      twice.u, twice.v, c.L, c.noise,
      static_cast<unsigned long long>(c.seed));
  for (int i = 0; i < c.steps; ++i) {
    if (i == split) { /* boundary between the two calls -- no state resets */ }
    unsigned long long noise_seed = static_cast<unsigned long long>(c.seed) ^
                                    (static_cast<unsigned long long>(i) << 20);
    gs::GrayScottStepKernel<<<grid, gs::kBlockSize>>>(
        twice.u, twice.v, twice.u2, twice.v2, c.L, 0.05f, 0.1f, c.F, c.k, 0.2f,
        c.noise, noise_seed);
    std::swap(twice.u, twice.u2);
    std::swap(twice.v, twice.v2);
  }
  CUDA_OK(cudaDeviceSynchronize());

  char label[128];
  std::snprintf(label, sizeof(label), "resume-equivalence %s steps=%d split=%d",
                c.regime, c.steps, split);
  double max_abs = 0.0;
  const bool ok_u = BitEqual(once.u, twice.u, n, "U", label, &max_abs);
  const bool ok_v = BitEqual(once.v, twice.v, n, "V", label, &max_abs);
  if (ok_u && ok_v) std::printf("  ok   %s\n", label);

  once.Free();
  twice.Free();
}

// ---------------------------------------------------------------------------
// Clio's pre-existing 2D reference
// ---------------------------------------------------------------------------
//
// Copied VERBATIM from adapter/kvhdf5/test/e2e/gray_scott_gpu_test.cu:61-79
// (kN folded into a parameter so one copy serves any grid size; nothing else
// altered -- same raw 4-point Laplacian, same `dt * (...)` association, same
// periodic wrap).
//
// Copied rather than linked, unlike the NeuroPress side of this file: the
// original sits inside a Catch2 TU that pulls in clio_runtime/singletons.h
// and kvhdf5/gpu_cte_dataset.h, so it cannot be compiled standalone. That
// makes this the weaker half of the harness -- it proves the port matches
// THIS text, and the text has to be re-checked by hand if the e2e test's
// kernel is ever edited. The provenance comment above is the only link.
namespace clio2d_ref {

__global__ void GsStepKernel(const float *u, const float *v, float *un,
                             float *vn, unsigned kN, unsigned cells, float Du,
                             float Dv, float F, float k, float dt) {
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= cells) return;
  unsigned x = gid % kN, y = gid / kN;
  unsigned xm = (x == 0) ? (kN - 1) : (x - 1);
  unsigned xp = (x == kN - 1) ? 0u : (x + 1);
  unsigned ym = (y == 0) ? (kN - 1) : (y - 1);
  unsigned yp = (y == kN - 1) ? 0u : (y + 1);

  float uc = u[gid], vc = v[gid];
  float lap_u =
      u[y * kN + xm] + u[y * kN + xp] + u[ym * kN + x] + u[yp * kN + x] - 4.f * uc;
  float lap_v =
      v[y * kN + xm] + v[y * kN + xp] + v[ym * kN + x] + v[yp * kN + x] - 4.f * vc;
  float uvv = uc * vc * vc;
  un[gid] = uc + dt * (Du * lap_u - uvv + F * (1.f - uc));
  vn[gid] = vc + dt * (Dv * lap_v + uvv - (F + k) * vc);
}

}  // namespace clio2d_ref

/**
 * Run the pre-existing 2D reference: the host-built initial condition from
 * gray_scott_gpu_test.cu:136-140, then `steps` of its kernel.
 */
void RunClio2DRef(Fields &f, int n_side, int steps, size_t cells) {
  std::vector<float> u0(cells, 1.0f), v0(cells, 0.0f);
  const unsigned c = static_cast<unsigned>(n_side) / 2;
  for (unsigned y = c - 3; y < c + 3; ++y) {
    for (unsigned x = c - 3; x < c + 3; ++x) v0[y * n_side + x] = 1.0f;
  }
  cudaMemcpy(f.u, u0.data(), cells * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(f.v, v0.data(), cells * sizeof(float), cudaMemcpyHostToDevice);

  const unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((cells + threads - 1) / threads);
  for (int i = 0; i < steps; ++i) {
    clio2d_ref::GsStepKernel<<<blocks, threads>>>(
        f.u, f.v, f.u2, f.v2, static_cast<unsigned>(n_side),
        static_cast<unsigned>(cells), 0.16f, 0.08f, 0.055f, 0.062f, 1.0f);
    std::swap(f.u, f.u2);
    std::swap(f.v, f.v2);
  }
}

/** The ported simulation in its kClio2D mode, through the public driver. */
void RunClio2DPort(Fields &f, int n_side, int steps, size_t cells) {
  const int grid = UpstreamGrid(cells);
  gs::GrayScottInit2DKernel<<<grid, gs::kBlockSize>>>(f.u, f.v, n_side, n_side);
  for (int i = 0; i < steps; ++i) {
    gs::GrayScottStep2DKernel<<<grid, gs::kBlockSize>>>(
        f.u, f.v, f.u2, f.v2, n_side, n_side, 0.16f, 0.08f, 0.055f, 0.062f,
        1.0f);
    std::swap(f.u, f.u2);
    std::swap(f.v, f.v2);
  }
}

/** Diff the 2D port against the pre-existing kernel on both fields. */
void Clio2DParityCase(int n_side, int steps) {
  const size_t cells = static_cast<size_t>(n_side) * n_side;
  const size_t bytes = cells * sizeof(float);

  Fields ref, port;
  if (!ref.Alloc(bytes) || !port.Alloc(bytes)) {
    ref.Free();
    port.Free();
    return;
  }

  RunClio2DRef(ref, n_side, steps, cells);
  RunClio2DPort(port, n_side, steps, cells);
  CUDA_OK(cudaDeviceSynchronize());

  char label[128];
  std::snprintf(label, sizeof(label), "clio-2d N=%d steps=%d", n_side, steps);
  double max_abs = 0.0;
  const bool ok_u = BitEqual(ref.u, port.u, cells, "U", label, &max_abs);
  const bool ok_v = BitEqual(ref.v, port.v, cells, "V", label, &max_abs);
  if (ok_u && ok_v) std::printf("  ok   %s\n", label);

  ref.Free();
  port.Free();
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("No CUDA device -- skipping.\n");
    return 77;
  }

  // The four (F,k) pairs the workload documents, spanning the compressibility
  // range the selection network is supposed to discriminate between: sparse
  // spots compress enormously, chaos barely at all.
  struct Regime {
    const char *name;
    float F;
    float k;
  };
  const Regime kRegimes[] = {
      {"spots", 0.04f, 0.06075f},
      {"stripes", 0.035f, 0.065f},
      {"chaos", 0.014f, 0.045f},
      {"sparse", 0.04f, 0.065f},
  };

  std::printf("=== Phase 1: initial condition ===\n");
  for (const Regime &r : kRegimes) {
    for (int L : {32, 64, 128}) {
      ParityCase({r.name, L, r.F, r.k, 0.0f, 42, 0});
    }
  }

  std::printf("\n=== Phase 2: single step ===\n");
  for (const Regime &r : kRegimes) {
    ParityCase({r.name, 64, r.F, r.k, 0.0f, 42, 1});
  }

  std::printf("\n=== Phase 3: long runs (divergence amplifies) ===\n");
  for (const Regime &r : kRegimes) {
    for (int steps : {10, 200}) {
      ParityCase({r.name, 64, r.F, r.k, 0.0f, 42, steps});
    }
  }
  // One production-shaped run: 128^3 is 8 MiB per field, i.e. two 4 MiB
  // chunks, which is the smallest size that exercises multi-chunk selection.
  ParityCase({"spots", 128, 0.04f, 0.06075f, 0.0f, 42, 200});

  std::printf("\n=== Phase 4: noise term (splitmix64 stream) ===\n");
  for (const Regime &r : kRegimes) {
    for (float noise : {0.001f, 0.01f}) {
      for (int seed : {42, 7}) {
        ParityCase({r.name, 64, r.F, r.k, noise, seed, 25});
      }
    }
  }

  std::printf("\n=== Phase 5: driver properties (Clio side only) ===\n");
  GeometryInvarianceCase({"spots", 64, 0.04f, 0.06075f, 0.0f, 42, 50});
  GeometryInvarianceCase({"chaos", 64, 0.014f, 0.045f, 0.01f, 42, 50});
  ResumeEquivalenceCase({"spots", 64, 0.04f, 0.06075f, 0.0f, 42, 40}, 17);
  ResumeEquivalenceCase({"chaos", 64, 0.014f, 0.045f, 0.01f, 7, 40}, 1);

  std::printf("\n=== Phase 6: Clio's pre-existing 2D model ===\n");
  // Same port, different model: raw 4-point Laplacian, Du=0.16 Dv=0.08
  // F=0.055 k=0.062 dt=1.0, 6x6 V=1 seed. Checked against the kernel the
  // kvhdf5 e2e harnesses have always run, so those can move onto this
  // simulation without changing a single expected value.
  for (int n : {64, 128, 256}) {
    Clio2DParityCase(n, 0);
  }
  for (int steps : {1, 10, 200, 1000}) {
    Clio2DParityCase(128, steps);
  }

  std::printf("\n===== %ld checks, %d failures =====\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
