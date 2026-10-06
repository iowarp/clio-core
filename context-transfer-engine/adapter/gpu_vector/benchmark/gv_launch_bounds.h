/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Launch bounds for the paged gpu_vector benchmark kernels.
 *
 * WHY THESE KERNELS NEED IT. The paged benches are device coroutines, and
 * NVPTX has no tail calls, so CoroSplit merges every resume segment into
 * one function and the register allocator takes the LIVENESS UNION across
 * all suspend points. The result is near-independent of what the kernel
 * actually computes: measured with cuobjdump, both the kmeans and the
 * lammps_md coroutine kernels land on exactly REG=192.
 *
 * 192 registers is 65536/192 = 341 resident threads per SM, i.e. ONE block
 * per SM at 256 threads and 12.5% occupancy -- against REG=32 and 100% for
 * the CTE-free baselines compiled from the same workloads. Past one block
 * per SM the extra blocks are not concurrent, they are QUEUED, and a
 * partial final wave idles most of the device: measured on kmeans at 32 GB,
 * 128 blocks ran 1.79x slower than MPI purely because its second wave
 * carried 20 blocks on 108 SMs.
 *
 * WHAT THE NUMBERS MEAN. The first is the maximum block size the kernel
 * will ever be launched with; the second is the occupancy target, which is
 * what tells ptxas how many blocks' worth of registers to fit per SM. It is
 * the same budget cmake/ClioCoroRegCap.cmake enforces from the outside
 * (CLIO_CORO_REGS_PER_SM / (CLIO_CORO_REF_THREADS * CLIO_CORO_TARGET_BLOCKS)
 * = 65536/(256*4) = 64), so the two agree by construction rather than by
 * coincidence.
 *
 * MEASURED EFFECT, isolated on lammps_md's launch TU with the regcap pass
 * stripped so only the annotation varies:
 *
 *     GV_LB_BLOCKS=4  ->  REG=64    STACK:16  LOCAL:0
 *     GV_LB_BLOCKS=0  ->  REG=192   STACK:16  LOCAL:0
 *
 * The cut is real and costs no local memory. It is NOT a guarantee:
 * __launch_bounds__ lowers to .maxntid/.minnctapersm and ptxas may exceed
 * the request when a kernel cannot be made to fit. It is also
 * architecture-relative -- it pins an OCCUPANCY target, which equals 64
 * registers only on a 65536-register SM.
 *
 * THE CONSTRAINT THIS IMPOSES. GV_LB_THREADS is a hard maximum: launching
 * any annotated kernel with more than this many threads FAILS at runtime.
 * Every paged bench defaults to 256 threads and takes --threads, so a sweep
 * that raises --threads above GV_LB_THREADS must raise this with it (and
 * accept the correspondingly smaller register budget, since the budget is
 * regs_per_sm / (threads * blocks)).
 *
 * GV_LB_BLOCKS=0 compiles the annotations out, for an A/B against the
 * unbounded build or against the coro_regcap pass alone.
 */
#ifndef CLIO_GV_BENCH_GV_LAUNCH_BOUNDS_H_
#define CLIO_GV_BENCH_GV_LAUNCH_BOUNDS_H_

#ifndef GV_LB_THREADS
#define GV_LB_THREADS 256
#endif
#ifndef GV_LB_BLOCKS
#define GV_LB_BLOCKS 4
#endif

#if defined(__CUDACC__) || defined(__CUDA__) || defined(__NVCC__)
#if GV_LB_BLOCKS > 0
#define GV_LAUNCH_BOUNDS __launch_bounds__(GV_LB_THREADS, GV_LB_BLOCKS)
#else
#define GV_LAUNCH_BOUNDS
#endif
#else
/* Non-CUDA compilations (the SYCL launch TUs share these headers) have no
 * __launch_bounds__; the occupancy question there is the device compiler's. */
#define GV_LAUNCH_BOUNDS
#endif

#endif  // CLIO_GV_BENCH_GV_LAUNCH_BOUNDS_H_
