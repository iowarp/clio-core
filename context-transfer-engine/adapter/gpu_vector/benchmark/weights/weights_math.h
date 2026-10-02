/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The synthetic-weights science, once, for every substrate and every backend.
 *
 * Dependency-free -- no clio, no CUDA runtime, no SYCL -- so the MPI,
 * NVSHMEM, BaM and Kokkos baselines can share it without linking this
 * project. Compiles under nvcc, clang-CUDA, DPC++ and a host compiler.
 *
 * These generators decide the COMPRESSION RATIO of the dataset, which is the
 * independent variable the weights benchmark sweeps. Every substrate must
 * generate byte-identical data or the residency numbers are not comparable,
 * so having one copy is not tidiness -- it is the experiment's control.
 */
#ifndef CLIO_GV_BENCH_WEIGHTS_MATH_H_
#define CLIO_GV_BENCH_WEIGHTS_MATH_H_

#if defined(__CUDACC__) || defined(__HIPCC__)
#define WT_MATH_FN __host__ __device__ inline
#else
#define WT_MATH_FN inline
#endif

namespace clio_wt {

using wt_u32 = unsigned int;
using wt_u64 = unsigned long long;

// The DATA's flat-block granularity, deliberately FIXED and independent of
// the vector's page size. PageIsFlat/Weight decide compressibility on this
// granule, so tying it to the runtime page size would change the generated
// bytes whenever the page size changed -- and a page-size sweep would then be
// comparing different datasets, not different paging.
constexpr wt_u64 kFlatGranuleBytes = 64 * 1024;
constexpr wt_u64 kFlatGranuleElems = kFlatGranuleBytes / sizeof(wt_u32);

/**
 * Is this page one of the highly compressible ones?
 *
 * Hashed rather than striped so the compressible pages are scattered through
 * the model: a contiguous compressible half would let the tier hold a solid
 * run of it and flatter residency for reasons that have nothing to do with
 * the ratio.
 */
WT_MATH_FN bool PageIsFlat(wt_u64 page, wt_u32 pct) {
  wt_u32 h = static_cast<wt_u32>(page * 2654435761u);
  h ^= h >> 15;
  return (h % 100u) < pct;
}

/**
 * A weight value. Runs of kRun identical values: structured weight data
 * repeats locally, and it is that repetition -- not the value distribution --
 * that a byte codec exploits. kRun sets the compression ratio; 8 lands near
 * 2x, which is where a real K-quant model sits.
 */
WT_MATH_FN wt_u32 Weight(wt_u64 i) {
  constexpr wt_u64 kRun = 8;
  wt_u32 r = static_cast<wt_u32>((i / kRun) * 2654435761u);
  r ^= r >> 13;
  return (r & 0x3F3F3F3Fu) |
         (static_cast<wt_u32>((i / 4096) % 13) * 0x40404040u);
}

/** As Weight(i), except a flat page is a single repeated value: what a byte
 *  codec collapses. */
WT_MATH_FN wt_u32 Weight(wt_u64 i, wt_u32 flat_pct) {
  if (PageIsFlat(i / kFlatGranuleElems, flat_pct)) {
    return 0x01010101u;
  }
  return Weight(i);
}

/** The activation each weight is multiplied by. Integer, so the accumulation
 *  is order-independent and the checksum gate can demand exactness. */
WT_MATH_FN wt_u32 Activation(wt_u64 i) {
  return static_cast<wt_u32>((i % 7) + 1);
}

}  // namespace clio_wt

#endif  // CLIO_GV_BENCH_WEIGHTS_MATH_H_
