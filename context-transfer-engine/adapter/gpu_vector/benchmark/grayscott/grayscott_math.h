/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The Gray-Scott science, once, for every substrate and every backend.
 *
 * Dependency-free on purpose -- no clio, no CUDA runtime, no SYCL -- because
 * the MPI, NVSHMEM, BaM and Kokkos baselines all need the same initial
 * condition and the same reaction term, and a baseline that links this
 * project is not a baseline. It compiles under nvcc, clang-CUDA, DPC++ and a
 * plain host compiler.
 *
 * What is deliberately NOT here: how a plane is reached. The paged path
 * gathers its stencil from held pages, the baselines from raw device
 * pointers, and that difference is exactly what the benchmark measures.
 */
#ifndef CLIO_GV_BENCH_GRAYSCOTT_MATH_H_
#define CLIO_GV_BENCH_GRAYSCOTT_MATH_H_

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GS_MATH_FN __host__ __device__ inline
#else
#define GS_MATH_FN inline
#endif

namespace clio_gs {

using gs_u64 = unsigned long long;

/** True inside the centred cube that seeds the pattern. */
GS_MATH_FN bool InSeedCube(gs_u64 x, gs_u64 y, gs_u64 z, gs_u64 nx, gs_u64 ny,
                           gs_u64 nz) {
  return (x > nx / 3 && x < 2 * nx / 3 && y > ny / 3 && y < 2 * ny / 3 &&
          z > nz / 3 && z < 2 * nz / 3);
}

/** Initial condition: v seeded in a centred cube, u elsewhere. Deterministic,
 *  so every configuration starts from the identical field. */
GS_MATH_FN float InitU(gs_u64 x, gs_u64 y, gs_u64 z, gs_u64 nx, gs_u64 ny,
                       gs_u64 nz) {
  return InSeedCube(x, y, z, nx, ny, nz) ? 0.5f : 1.0f;
}

GS_MATH_FN float InitV(gs_u64 x, gs_u64 y, gs_u64 z, gs_u64 nx, gs_u64 ny,
                       gs_u64 nz) {
  return InSeedCube(x, y, z, nx, ny, nz) ? 0.25f : 0.0f;
}

/**
 * The reaction-diffusion update for one cell, given its laplacians.
 *
 * Split from the stencil GATHER on purpose: the gather is what differs
 * between substrates (held pages vs raw device pointers vs Kokkos Views),
 * and the arithmetic is what must not.
 */
GS_MATH_FN void ReactDiffuse(float u, float v, float lu, float lv, float Du,
                             float Dv, float F, float K, float dt, float *un,
                             float *vn) {
  const float uvv = u * v * v;
  *un = u + dt * (Du * lu - uvv + F * (1.0f - u));
  *vn = v + dt * (Dv * lv + uvv - (F + K) * v);
}

}  // namespace clio_gs

#endif  // CLIO_GV_BENCH_GRAYSCOTT_MATH_H_
