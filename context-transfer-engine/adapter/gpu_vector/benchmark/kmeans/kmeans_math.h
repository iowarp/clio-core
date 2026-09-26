/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The kmeans science, once, for every substrate and every backend.
 *
 * WHY THIS FILE HAS NO DEPENDENCIES
 *
 * The paged benchmark and the MPI baseline must agree on the data and on the
 * arithmetic, or the comparison between them means nothing -- but the
 * baseline deliberately links NOTHING from clio ("a baseline that shares a
 * substrate with the thing it is benchmarked against is not a baseline").
 * So the shared part cannot be a clio header. What is left, and what lives
 * here, is pure per-point math over raw pointers: no clio, no CUDA runtime,
 * no SYCL, no allocation, no thread indexing.
 *
 * It compiles under all four compilers this tree uses: nvcc (the MPI and BaM
 * baselines are built with `nvcc -x cu`), clang-CUDA, DPC++, and a plain host
 * compiler.
 *
 * Everything ABOVE this -- how points are stored, how threads are indexed,
 * how partial sums meet -- is what each substrate is actually being measured
 * on, and is deliberately NOT shared.
 *
 * These were four copies of the same twenty lines: the paged bench, the MPI
 * baseline, the NVSHMEM baseline and the BaM baseline, each with a comment
 * saying "IDENTICAL to the paged bench" and no way to enforce it.
 */
#ifndef CLIO_GV_BENCH_KMEANS_MATH_H_
#define CLIO_GV_BENCH_KMEANS_MATH_H_

/** Callable from host and device under every compiler here. Deliberately
 *  local rather than CTP_INLINE_CROSS_FUN: the baselines do not include clio
 *  headers at all, and must not start now. */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define KM_MATH_FN __host__ __device__ inline
#else
#define KM_MATH_FN inline
#endif

namespace clio_km {

using km_u32 = unsigned int;
using km_u64 = unsigned long long;

/**
 * Deterministic synthetic coordinate: cluster-structured, so the assignment
 * step does real work instead of every point landing on one centroid.
 *
 * Generated from the GLOBAL element index, which is what lets a paged run, a
 * sharded MPI run and a staged-tile baseline all cluster the same cloud
 * without shipping an input file.
 */
KM_MATH_FN float PointVal(km_u64 idx, km_u32 dims, km_u32 k) {
  const km_u64 point = idx / dims;
  const km_u64 dim = idx % dims;
  const km_u64 cluster = point % k;            // ground-truth cluster
  const float centre = static_cast<float>(cluster) * 8.0f;
  // Deterministic jitter in [-1, 1]; a cheap hash so it is reproducible on
  // host and device without carrying RNG state.
  const km_u64 h =
      (point * 6364136223846793005ull + dim * 1442695040888963407ull);
  const float jitter =
      static_cast<float>(static_cast<km_u32>(h >> 40)) * (2.0f / 16777216.0f) -
      1.0f;
  return centre + jitter;
}

/**
 * Nearest centroid to `pt`, by squared distance over `dims`.
 *
 * `pt` and `cent` are raw pointers on purpose: the paged path hands in a
 * pointer into a held page frame, the baselines hand in one into plain device
 * memory, and this must not be able to tell the difference.
 */
template <typename PtT, typename CentT>
KM_MATH_FN km_u32 NearestCentroid(const PtT &pt, const CentT &cent,
                                  km_u32 dims, km_u32 k) {
  float best = 3.4e38f;
  km_u32 bestk = 0;
  for (km_u32 c = 0; c < k; ++c) {
    float d = 0.0f;
    for (km_u32 i = 0; i < dims; ++i) {
      const float x = pt[i] - cent[c * dims + i];
      d += x * x;
    }
    if (d < best) {
      best = d;
      bestk = c;
    }
  }
  return bestk;
}

/** centroid = sum / count for cluster `c`, leaving an empty cluster where it
 *  was. The caller supplies the thread-to-cluster mapping. */
KM_MATH_FN void UpdateCentroid(float *cent, const float *sums,
                               const unsigned *counts, km_u32 dims, km_u32 c) {
  const unsigned n = counts[c];
  if (n == 0) return;
  for (km_u32 i = 0; i < dims; ++i) {
    cent[c * dims + i] = sums[c * dims + i] / static_cast<float>(n);
  }
}

}  // namespace clio_km

#endif  // CLIO_GV_BENCH_KMEANS_MATH_H_
