/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Geometry and constants shared by the MD device code, the launch seam and
 * the host driver.
 *
 * Deliberately free of anything device-only: the host driver includes this
 * but must NEVER include md_kernels.h, because DPC++ member-checks a whole
 * translation unit and a TU holding both the coroutines and gpu_vector.h's
 * host-only Vector cannot compile. That constraint is why this file exists
 * separately from md_kernels.h at all.
 */
#ifndef CLIO_GV_BENCH_MD_COMMON_H_
#define CLIO_GV_BENCH_MD_COMMON_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/types.h>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

/** Coroutine frames live in the lane; integrator coroutines are small. */
/**
 * Per-thread coroutine-frame lane. SMALLER IS FASTER, and not for the
 * memory: lanes are strided by this value, so a warp reading one frame
 * local touches 32 separate cache lines. At 4096 a 64-thread block's
 * frames span 256 KB and blow past L1; trimming the frame (see
 * kMaxNlGuards) lets a block's whole frame working set stay cached.
 */
// 2048 until the slab clipping added e0/e1/s_lo/s_hi to four coroutines and
// pushed SentinelCoro to 2064 -- reported exactly, by the device fatal
// channel, as "103 coro-frame ... need=2064". 2560 leaves headroom; the cost
// is lane memory, blocks * threads * bytes (64 x 256 x 2560 = 40 MB).
static constexpr u32 kYieldLaneBytes = 2560;
/** Elements per atom in x and v (float4 packing). */
static constexpr u32 kStride = 4;
/** MD_PROF=1 phase attribution inside the force coroutine, cycles, thread
 *  0 of each block: [0] stencil holds [1] f hold+zero [2] list guards
 *  [3] pair loop [4] whole row loop [5] rows processed. */
/**
 * Launch bounds for the MD kernels. Every one is launched with a.threads
 * (default 256), so the max-threads half is exact. The second number is the
 * occupancy target: it tells ptxas how many blocks per SM to allocate
 * registers for, which is the same budget coro_regcap enforces from outside.
 *
 * MD_LB_BLOCKS=0 compiles them out, for an A/B against the pass alone.
 */
#ifndef MD_LB_THREADS
#define MD_LB_THREADS 256
#endif
#ifndef MD_LB_BLOCKS
#define MD_LB_BLOCKS 4
#endif
#if MD_LB_BLOCKS > 0
#define MD_LAUNCH_BOUNDS __launch_bounds__(MD_LB_THREADS, MD_LB_BLOCKS)
#else
#define MD_LAUNCH_BOUNDS
#endif

static constexpr int kMaxNlGuards = 4;
static constexpr u32 kSpanGuards = 12;
static constexpr u32 kMinSlotsX = 2 * kSpanGuards + 4;
static constexpr u32 kMinSlotsF = 4;
static constexpr u32 kMinSlotsNl = static_cast<u32>(kMaxNlGuards) + 2;

struct Slab {
  u64 lo, hi;          // element range this node owns
  u64 pg_lo, pg_hi;    // pages overlapping it
};
CTP_INLINE_CROSS_FUN Slab SlabOf(u32 nb, u32 cap, u32 z0, u32 z1, u64 epp) {
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  Slab s;
  s.lo = static_cast<u64>(z0) * nb * row_elems;
  s.hi = static_cast<u64>(z1) * nb * row_elems;
  s.pg_lo = s.lo / epp;
  s.pg_hi = (s.hi + epp - 1) / epp;
  return s;
}


#endif  // CLIO_GV_BENCH_MD_COMMON_H_
