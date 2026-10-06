/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The LBANN MLP's deterministic initialisation, once, for every substrate
 * and every backend.
 *
 * Dependency-free -- no clio, no CUDA runtime, no SYCL -- so the MPI,
 * NVSHMEM, BaM and Kokkos baselines can share it. Every substrate must start
 * from byte-identical weights or the loss curves are not comparable, which
 * is why this is one copy rather than four.
 */
#ifndef CLIO_GV_BENCH_LBANN_MATH_H_
#define CLIO_GV_BENCH_LBANN_MATH_H_

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LB_MATH_FN __host__ __device__ inline
#else
#define LB_MATH_FN inline
#endif

namespace clio_lb {

using lb_u64 = unsigned long long;

LB_MATH_FN lb_u64 Lcg(lb_u64 s) {
  return s * 6364136223846793005ull + 1442695040888963407ull;
}

/**
 * Symmetric [-1, 1) from an ALREADY-HASHED value: the weight initialiser.
 *
 * Takes the LCG output, it does not apply the LCG itself -- callers hash the
 * index first and reuse the state. Copied byte-for-byte from the original,
 * because this decides every initial weight and a different scale factor
 * would give a different model that still looked plausible.
 */
LB_MATH_FN float Sym01(lb_u64 s) {
  return (static_cast<float>((s >> 40) & 0xFFFFFF) / 8388608.0f) - 1.0f;
}

}  // namespace clio_lb

#endif  // CLIO_GV_BENCH_LBANN_MATH_H_
