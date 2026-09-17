/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The PME science, once, for every substrate and every backend.
 *
 * Dependency-free -- no clio, no CUDA runtime, no SYCL -- so the MPI,
 * NVSHMEM, BaM and Kokkos baselines can share it without linking this
 * project. Compiles under nvcc, clang-CUDA, DPC++ and a host compiler.
 *
 * What is NOT here is how a mesh plane is reached; that is what the
 * benchmark measures.
 */
#ifndef CLIO_GV_BENCH_GMX_MATH_H_
#define CLIO_GV_BENCH_GMX_MATH_H_

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GMX_MATH_FN __host__ __device__ inline
#else
#define GMX_MATH_FN inline
#endif

namespace clio_gmx {

using gmx_u64 = unsigned long long;

/** Charge is accumulated in FIXED POINT so the sum is order-independent and
 *  the gates can demand bit equality. */
static constexpr double kFxScale = 16777216.0;   // 2^24

/** Deterministic pseudo-random atom cloud: same atoms for every path. */
GMX_MATH_FN gmx_u64 Lcg(gmx_u64 s) {
  return s * 6364136223846793005ull + 1442695040888963407ull;
}

GMX_MATH_FN float Frac01(gmx_u64 s) {
  return static_cast<float>((s >> 40) & 0xFFFFFF) / 16777216.0f;
}

/**
 * Round-to-nearest-even, double -> long long -- the fixed-point conversion
 * every substrate must agree on to the bit.
 *
 * NOT llrint. That is what the CUDA sources called, and DPC++ has no device
 * implementation of it: the SYCL build failed with
 * "error: no libcall available for llrint" at device link. `rint` IS
 * available on both (CUDA's math header and SYCL's builtins), and it carries
 * the SAME rounding mode -- round-to-nearest, ties-to-even -- so the cast
 * below is bit-identical to llrint's result over the range these charges
 * occupy. Anything cheaper (x < 0 ? x - 0.5 : x + 0.5) rounds ties AWAY from
 * zero and would silently break the digit-exact gates.
 */
GMX_MATH_FN long long FxRound(double x) {
#if defined(__CUDA_ARCH__)
  return llrint(x);
#else
  return static_cast<long long>(rint(x));
#endif
}

/** Cardinal cubic B-spline weights for fractional offset t in [0,1): the
 *  four grid nodes i0..i0+3 with i0 = floor(x) - 1 get M4(t+1..t-2). */
GMX_MATH_FN void Spline4(float t, float w[4]) {
  const float t2 = t * t, t3 = t2 * t;
  w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) / 6.0f;   // node i0,   dist 1+t
  w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) / 6.0f;       // node i0+1, dist t
  w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) / 6.0f;  // dist 1-t
  w[3] = t3 / 6.0f;                                    // node i0+3, dist 2-t
}

}  // namespace clio_gmx

#endif  // CLIO_GV_BENCH_GMX_MATH_H_
