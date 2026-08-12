/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Gray-Scott reaction-diffusion simulation for the NeuroPress GPU demo, in
 * its own nvcc-compiled translation unit with NO Clio includes. clio_cte/
 * compressor headers behave differently when pulled through nvcc (see
 * neuropress_gpu_demo.cc, a plain host TU) -- the fix pattern already
 * established in core_client.h/.cc this project: keep CUDA device code and
 * Clio client code in separate TUs, linked only through a plain function
 * declaration.
 *
 * Same reaction-diffusion model and parameters as
 * context-transfer-engine/adapter/kvhdf5/test/e2e/gray_scott_gpu_test.cu --
 * this demo represents a real simulation producing GPU-resident data one
 * timestep at a time, not a synthetic fill pattern. Every step's evolved u
 * field is copied device-to-device directly into that step's chunk slot in
 * the caller's output buffer; the host never touches the simulation state.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace {

struct GsParams {
  float Du, Dv, F, k, dt;
};

/** One Gray-Scott step: one thread per cell, periodic BCs. */
__global__ void GsStepKernel(const float *u, const float *v, float *un,
                             float *vn, unsigned w, unsigned h, GsParams p) {
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned cells = w * h;
  if (gid >= cells) return;
  unsigned x = gid % w, y = gid / w;
  unsigned xm = (x == 0) ? (w - 1) : (x - 1);
  unsigned xp = (x == w - 1) ? 0u : (x + 1);
  unsigned ym = (y == 0) ? (h - 1) : (y - 1);
  unsigned yp = (y == h - 1) ? 0u : (y + 1);

  float uc = u[gid], vc = v[gid];
  float lap_u = u[y * w + xm] + u[y * w + xp] + u[ym * w + x] +
                u[yp * w + x] - 4.f * uc;
  float lap_v = v[y * w + xm] + v[y * w + xp] + v[ym * w + x] +
                v[yp * w + x] - 4.f * vc;
  float uvv = uc * vc * vc;
  un[gid] = uc + p.dt * (p.Du * lap_u - uvv + p.F * (1.f - uc));
  vn[gid] = vc + p.dt * (p.Dv * lap_v + uvv - (p.F + p.k) * vc);
}

}  // namespace

/** Runs `num_steps` of Gray-Scott on a grid_w x grid_h grid, writing each
    step's evolved u field into d_snapshots[step * grid_w * grid_h ...] --
    device-to-device, so d_snapshots ends up holding num_steps consecutive
    simulation snapshots entirely on the GPU. */
void RunGrayScottSimulation(float *d_snapshots, unsigned grid_w,
                            unsigned grid_h, unsigned num_steps) {
  const size_t cells = static_cast<size_t>(grid_w) * grid_h;
  const size_t bytes = cells * sizeof(float);

  float *u_curr = nullptr, *u_next = nullptr;
  float *v_curr = nullptr, *v_next = nullptr;
  cudaMalloc(&u_curr, bytes);
  cudaMalloc(&u_next, bytes);
  cudaMalloc(&v_curr, bytes);
  cudaMalloc(&v_next, bytes);

  // Classic Gray-Scott IC: u=1 everywhere, v=1 in a small centre square.
  // Built on the host (a few KiB) and copied in once -- this is initial
  // condition data, not the simulation output the demo is about.
  std::vector<float> u0(cells, 1.0f), v0(cells, 0.0f);
  unsigned cy = grid_h / 2, cx = grid_w / 2;
  for (unsigned y = cy - 3; y < cy + 3; ++y) {
    for (unsigned x = cx - 3; x < cx + 3; ++x) v0[y * grid_w + x] = 1.0f;
  }
  cudaMemcpy(u_curr, u0.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(v_curr, v0.data(), bytes, cudaMemcpyHostToDevice);

  const GsParams params{0.16f, 0.08f, 0.055f, 0.062f, 1.0f};
  const int threads = 256;
  const int blocks = static_cast<int>((cells + threads - 1) / threads);

  for (unsigned step = 0; step < num_steps; ++step) {
    GsStepKernel<<<blocks, threads>>>(u_curr, v_curr, u_next, v_next, grid_w,
                                      grid_h, params);
    std::swap(u_curr, u_next);
    std::swap(v_curr, v_next);
    // Persist this timestep's evolved u field into its chunk slot -- the
    // same "explicit persistence" boundary gray_scott_gpu_test.cu documents:
    // compute stays pure CUDA, I/O is a distinct, explicit step.
    cudaMemcpy(d_snapshots + static_cast<size_t>(step) * cells, u_curr, bytes,
               cudaMemcpyDeviceToDevice);
  }

  cudaDeviceSynchronize();
  cudaFree(u_curr);
  cudaFree(u_next);
  cudaFree(v_curr);
  cudaFree(v_next);
}
