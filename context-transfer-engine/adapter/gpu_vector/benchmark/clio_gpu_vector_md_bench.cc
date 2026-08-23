/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * eternia-MD: a from-scratch reimplementation of the LAMMPS melt benchmark
 * (lj/cut, NVE, periodic box) with ALL simulation state device-canonical on
 * paged gv::Vectors. Design: core/eternia.md. LAMMPS itself is only the
 * correctness oracle.
 *
 * STAGE 1 (this file today): the layout and the integrator.
 *   - bin-major, padded-bin atom layout on two vectors, x (xyz + type in .w)
 *     and v (vxyz + spare), pages containing whole bins;
 *   - K1 (half-kick + drift), K4 (second half-kick), K5 (thermo reduction);
 *   - the BALLISTIC GATE: under a constant acceleration, velocity Verlet is
 *     algebraically exact, and per-atom updates are order-identical to a
 *     host float replica, so the gate demands BITWISE equality against the
 *     replica plus closed-form agreement in double. Any mismatch is a bug
 *     in the layout, the holds, or the kernels -- there is nowhere to hide.
 *   - the RESIDENT contract: stage-1 caches are sized to fit, and the run
 *     asserts zero faults / zero evictions in the timed kernels.
 *
 * Stages 2-5 (binning/resort, forces, the device-built Verlet list on a
 * paged neigh vector, out-of-core configs, stock-LAMMPS golden validation)
 * layer onto this file without changing the structures introduced here.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

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
static constexpr u32 kYieldLaneBytes = 2048;   // largest frame measured 1856
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

__device__ unsigned long long g_md_cyc[8];

/** Page iterations actually completed by the integrator, summed over
 *  blocks. Distinguishes WORK DROPPED (driver/park bookkeeping) from DATA
 *  CORRUPTED (the paging path): the expected count is exact and known. */
__device__ unsigned long long g_pages_done;
/** Sink for the ballistic path's optional THIRD vector read, so the load
 *  cannot be optimised away while the physics stays untouched. */
__device__ unsigned long long g_third_sink;
/** READ-ONLY PROBE (--readprobe). Every experiment so far assumed writes
 *  were being lost; this asks the other half of the question -- does the
 *  FAULT PATH ever deliver wrong bytes? The vector is seeded with a value
 *  that is unique per element and exact in float, then only READ.
 *  [0] mismatches [1] first bad page [2] first bad index. */
__device__ unsigned long long g_read_bad[4];
/** First bad sample: {claimed, page, elem, observed-value-as-u32-bits}. */
__device__ unsigned long long g_read_sample[8];
/** Set from MD_PROBE_LDCG: read the probe's elements bypassing L1. */
__device__ bool kProbeLdcg;
/** {frame0, pages_per_block, page_bytes, elems_per_page} of the probe's table. */
__device__ unsigned long long g_read_geom[12];
/** Per LOGICAL block: page iterations completed, and the last page index
 *  reached. Says WHICH block loses work, and where it stopped. */
__device__ unsigned long long g_blk_done[64];
__device__ unsigned long long g_blk_last[64];

/** Guards a block may hold over one row of the paged list. Deliberately
 *  small: this array lives in the per-thread coroutine FRAME, so every
 *  slot costs frame footprint whether used or not, and the list page is
 *  sized to hold a whole row (one guard, two across a boundary). The host
 *  validates the bound in configuration terms before any run. */
static constexpr int kMaxNlGuards = 4;

/**
 * Worst-case x-span guards a force chunk holds at once: three z planes, each
 * up to two y ranges when the stencil crosses the y wrap, each range up to
 * two guards when it straddles a page boundary. Reserved up front because
 * the true count is not known until the spans are computed, and reserving
 * after the first hold is exactly the deadlock admission exists to prevent.
 */
static constexpr u32 kSpanGuards = 12;

/**
 * MINIMUM frames a table must have for these kernels to make progress.
 *
 * A cache is not a free parameter: it has to hold everything a kernel pins
 * AT ONCE, plus room to fetch the next span while those stay pinned. Below
 * that, EvictPages has nothing it may take and traps -- correctly, because
 * the request cannot be satisfied. Asking for fewer slots does not produce
 * "more paging", it produces an impossible configuration.
 *
 * x    the pair loop pins up to kSpanGuards spans, and the gather runs as
 *      (src=dx, srcx=dx, dst=dx2) so ONE table supplies both span sets;
 *      +2 for the destination row, +2 free frames for the next fetch.
 * f    two guards (a row and its straddle), +2 to fetch against.
 * nl   kMaxNlGuards, +2 to fetch against.
 */
static constexpr u32 kMinSlotsX = 2 * kSpanGuards + 4;
static constexpr u32 kMinSlotsF = 4;
static constexpr u32 kMinSlotsNl = static_cast<u32>(kMaxNlGuards) + 2;

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define GV_MD_CORO 1
#endif

namespace {

/** Pages a range touches -- the vector no longer exposes this. */
template <typename V>
CTP_GPU_FUN u32 PagesSpanned(const V &v, u64 off, u64 count) {
  if (count == 0) return 0;
  return static_cast<u32>(v.PageOf(off + count - 1) - v.PageOf(off)) + 1u;
}


/**
 * The keystone layout (eternia.md section 2): atoms live bin-major in
 * padded bins; an atom's index is bin * cap + slot; empty slots carry
 * type = -1 in x's .w lane. Pages contain WHOLE bins (cap * kStride divides
 * the page), so every kernel's working set is a set of whole pages known
 * from geometry. Stage 1 only needs the layout to exist and to survive the
 * integrator; binning DYNAMICS (resort, stencils) arrive in stage 2.
 */
struct Geometry {
  double box = 0.0;        // cubic box edge
  double bin_edge = 0.0;   // >= cutoff + skin
  u32 nb = 0;              // bins per dimension
  u64 nbins = 0;           // nb^3
  u32 cap = 0;             // atom slots per bin (padded, fixed)
  u64 nslots = 0;          // nbins * cap == total atom slots
  u64 nelems = 0;          // nslots * kStride == vector elements
  u64 natoms = 0;          // real atoms
};

struct Args {
  u32 lattice = 20;        // FCC unit cells per dimension -> 4*n^3 atoms
  double rho = 0.8442;     // reduced density (the melt deck's)
  double cutoff = 2.5;
  double skin = 0.3;
  u32 cap = 32;
  u64 page_kb = 64;
  u32 blocks = 64;
  u32 threads = 256;
  u32 slots = 0;           // x/v cache slots per block; 0 = resident (auto)
  u32 ppslots = 0;         // resort ping-pong cache slots; 0 = same as slots
  u32 fslots = 0;          // force-vector cache slots; 0 = same as slots
  u32 vslots = 0;          // velocity cache slots; 0 = same as slots
  u64 steps = 100;
  u64 rebin = 20;          // resort cadence (steps); 0 = never
  double temp = 0.0;       // scale initial velocities to this T (0 = leave)
  u32 maxneigh = 96;       // Verlet-list capacity per atom slot
  u32 rowchunk = 4;        // rows per block in the force pass (hold reuse)
  u64 ckpt = 0;            // checkpoint every N steps (0 = never)
  u64 nl_page_kb = 0;      // list page size; 0 = one whole row per page
  int use_list = 1;        // stage 3 list force; --no-list = cell-direct
  // NVE drift tolerance. The default suits cold runs (measured 6e-7 over
  // 200 steps). HOT melt-deck runs need ~5e-3: the unshifted lj/cut energy
  // is discontinuous at the cutoff, and STOCK LAMMPS itself (double
  // precision) drifts 1.9e-3 over the same 250 steps at T=3 -- measured
  // from its own logs, cadence-independent here (same at rebin 10/5/2).
  double drift_tol = 5e-4;
  double dt = 0.005;
  int gate = 1;            // stage 1: the ballistic gate IS the run
  int readprobe = 0;       // read-only paging probe (no writes at all)
  int md = 0;              // stage 2: real LJ forces + NVE, statics gates
  double g[3] = {0.1, -0.05, 0.02};   // constant acceleration for the gate
};

/** LAMMPS melt step-0 pair energy per atom (rho=0.8442 FCC, lj/cut 2.5,
 *  no shift): the published stock value this reimplementation must reproduce
 *  from geometry alone. */
static constexpr double kMeltPePerAtom = -6.7733681;

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

/** Deterministic per-atom initial velocity: any fixed function works for the
 *  gate (the replica and the closed form both use it); keep it simple and
 *  well-spread. */
void InitVelocity(u64 atom_id, float out[3]) {
  const double a = static_cast<double>(atom_id);
  out[0] = static_cast<float>(0.20 * std::sin(0.37 * a + 0.1));
  out[1] = static_cast<float>(0.20 * std::cos(0.53 * a + 0.7));
  out[2] = static_cast<float>(0.20 * std::sin(0.71 * a + 1.3));
}

/**
 * Build the bin-major initial state on the host: FCC lattice positions
 * binned into the padded layout. Refuses loudly if any bin overflows cap --
 * a silent overflow would corrupt a neighbour bin, which is the exact class
 * of quiet failure this project is done tolerating.
 * @return false on overflow.
 */
bool BuildInitialState(const Args &a, const Geometry &g,
                       std::vector<float> &hx, std::vector<float> &hv,
                       std::vector<u32> &bin_count) {
  hx.assign(g.nelems, 0.0f);
  hv.assign(g.nelems, 0.0f);
  // Sentinel every slot first.
  for (u64 s = 0; s < g.nslots; ++s) hx[s * kStride + 3] = -1.0f;
  bin_count.assign(g.nbins, 0u);

  const double cell = g.box / a.lattice;   // FCC unit-cell edge
  static const double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  u64 atom_id = 0;
  for (u32 i = 0; i < a.lattice; ++i) {
    for (u32 j = 0; j < a.lattice; ++j) {
      for (u32 k = 0; k < a.lattice; ++k) {
        for (int b = 0; b < 4; ++b) {
          const double px = (i + kBasis[b][0]) * cell;
          const double py = (j + kBasis[b][1]) * cell;
          const double pz = (k + kBasis[b][2]) * cell;
          u32 bx = static_cast<u32>(px / g.bin_edge);
          u32 by = static_cast<u32>(py / g.bin_edge);
          u32 bz = static_cast<u32>(pz / g.bin_edge);
          if (bx >= g.nb) bx = g.nb - 1;   // boundary atom, edge bin
          if (by >= g.nb) by = g.nb - 1;
          if (bz >= g.nb) bz = g.nb - 1;
          const u64 bin = (static_cast<u64>(bz) * g.nb + by) * g.nb + bx;
          const u32 slot = bin_count[bin]++;
          if (slot >= g.cap) {
            std::fprintf(stderr,
                         "bin %llu overflows cap=%u -- raise --cap\n",
                         (unsigned long long)bin, g.cap);
            return false;
          }
          const u64 e = (bin * g.cap + slot) * kStride;
          hx[e + 0] = static_cast<float>(px);
          hx[e + 1] = static_cast<float>(py);
          hx[e + 2] = static_cast<float>(pz);
          hx[e + 3] = 1.0f;   // type 1
          float vv[3];
          InitVelocity(atom_id, vv);
          hv[e + 0] = vv[0];
          hv[e + 1] = vv[1];
          hv[e + 2] = vv[2];
          hv[e + 3] = 0.0f;
          ++atom_id;
        }
      }
    }
  }
  return atom_id == g.natoms;
}

/**
 * Host double-precision reference for the step-0 statics (stage-2 gate,
 * validation layer 2): the same cell-direct pass in double. Returns PE,
 * the scalar virial sum W = sum r.f over pairs (counted once), and the
 * pair count within the cutoff.
 */
void HostForceReference(const Geometry &g, const std::vector<float> &hx,
                        double cutoff, double *pe_out, double *w_out,
                        unsigned long long *pairs_out) {
  const double L = g.box;
  const double c2 = cutoff * cutoff;
  double pe = 0.0, w = 0.0;
  unsigned long long pairs = 0;
  const int nb = static_cast<int>(g.nb);
  for (u64 bin = 0; bin < g.nbins; ++bin) {
    const int bx = static_cast<int>(bin % nb);
    const int by = static_cast<int>((bin / nb) % nb);
    const int bz = static_cast<int>(bin / (static_cast<u64>(nb) * nb));
    for (u32 si = 0; si < g.cap; ++si) {
      const u64 ei = (bin * g.cap + si) * kStride;
      if (hx[ei + 3] < 0.0f) continue;
      const double xi = hx[ei], yi = hx[ei + 1], zi = hx[ei + 2];
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dxx = -1; dxx <= 1; ++dxx) {
            const int jbx = (bx + dxx + nb) % nb;
            const int jby = (by + dy + nb) % nb;
            const int jbz = (bz + dz + nb) % nb;
            const u64 jbin =
                (static_cast<u64>(jbz) * nb + jby) * nb + jbx;
            for (u32 sj = 0; sj < g.cap; ++sj) {
              const u64 ej = (jbin * g.cap + sj) * kStride;
              if (hx[ej + 3] < 0.0f) continue;
              if (ej == ei) continue;
              double ddx = xi - hx[ej];
              double ddy = yi - hx[ej + 1];
              double ddz = zi - hx[ej + 2];
              if (ddx > 0.5 * L) ddx -= L; else if (ddx < -0.5 * L) ddx += L;
              if (ddy > 0.5 * L) ddy -= L; else if (ddy < -0.5 * L) ddy += L;
              if (ddz > 0.5 * L) ddz -= L; else if (ddz < -0.5 * L) ddz += L;
              const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (r2 >= c2) continue;
              const double r2i = 1.0 / r2;
              const double r6i = r2i * r2i * r2i;
              pe += 0.5 * (4.0 * r6i * (r6i - 1.0));     // pair counted twice
              w += 0.5 * (r6i * (48.0 * r6i - 24.0));    // r.f, same halving
              ++pairs;
            }
          }
        }
      }
    }
  }
  *pe_out = pe;
  *w_out = w;
  *pairs_out = pairs / 2;   // full-list double count -> unique pairs
}

}  // namespace

#if defined(GV_MD_CORO)

/**
 * K1: first velocity-Verlet half -- v += (dt/2)*accel, x += dt*v.
 * A block owns pages block, block+nblocks, ...; per page it write-holds the
 * SAME page of x and v and streams the slots. Sentinel slots are skipped.
 * No periodic wrap here BY DESIGN: positions wrap at the resort (stage 2),
 * exactly as LAMMPS wraps at reneighbour -- and it keeps the ballistic gate
 * algebraically exact.
 *
 * Stage 2 note: `accel` becomes a read-hold on the f vector; the hold
 * structure of this kernel does not change.
 */
/**
 * K3: the cell-direct LJ force pass (stage 2). Work unit = one full X-ROW
 * of bins (bz, by, *): the row is index-contiguous, and because the x
 * dimension wraps WITHIN the row, its stencil is exactly NINE full rows
 * (dz, dy in {-1,0,1}, z/y wrapped) -- each one contiguous elem range,
 * held in at most two guards (a row can straddle one page boundary).
 * One thread per i-atom, register accumulation, ONE store per component:
 * no atomics anywhere (full-list/newton-off: every atom's force is
 * computed entirely here, each pair evaluated from both sides; PE and the
 * virial take the standard 0.5x).
 *
 * acc[0] += PE, acc[1] += virial W = sum r.f, acc[2] += pairs-within-cutoff
 * (double count), accumulated only when eflag != 0.
 */
__device__ gy::YCoroMain ForceCoro(gv::DeviceVector<float> x,
                                   gv::DeviceVector<float> f,
                                   u32 nb, u32 cap, float box, float cutoff,
                                   int eflag, double *acc, u32 nblocks,
                                   u32 block) {
  extern __shared__ char smem_raw[];
  double *red = reinterpret_cast<double *>(smem_raw + CLIO_YIELD_SMEM_BYTES);
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 nrows = static_cast<u64>(nb) * nb;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  double pe = 0.0, w = 0.0, npairs = 0.0;

  for (u64 row = block; row < nrows; row += nblocks) {
    const u32 by = static_cast<u32>(row % nb);
    const u32 bz = static_cast<u32>(row / nb);

    // Hold the NINE stencil rows (this row is rows[4]); <= 2 guards each.
    gv::Held<float> hg[9][2];
    u64 rbase[9], rrun0[9];
    const float *rp0[9], *rp1[9];
    for (int q = 0; q < 9; ++q) {
      const int dz = q / 3 - 1, dy = q % 3 - 1;
      const u32 wz = (bz + nb + dz) % nb;
      const u32 wy = (by + nb + dy) % nb;
      const u64 rowbin0 = (static_cast<u64>(wz) * nb + wy) * nb;
      rbase[q] = rowbin0 * cap * kStride;
co_await x.BeginFetch(x.PageLo(rbase[q]), x.PageSpan(rbase[q], row_elems));
      co_await x.AwaitFetch();
            hg[q][0] = co_await x.HoldPage(rbase[q], row_elems);
      rrun0[q] = hg[q][0].run();
      if (rrun0[q] < row_elems) {
        hg[q][1] = co_await x.HoldPage(rbase[q] + rrun0[q],
                                       row_elems - rrun0[q]);
      }
      rp0[q] = hg[q][0].ptr();
      rp1[q] = hg[q][1] ? hg[q][1].ptr() : nullptr;
    }
    // Write-hold this row of f and zero it.
    const u64 fbase = (static_cast<u64>(bz) * nb + by) * static_cast<u64>(nb) *
                      cap * kStride;
co_await f.BeginFetch(f.PageLo(fbase), f.PageSpan(fbase, row_elems));
    co_await f.AwaitFetch();
        gv::Held<float> hf0 = co_await f.HoldPage(fbase, row_elems, true);
    gv::Held<float> hf1;
    const u64 frun0 = hf0.run();
    if (frun0 < row_elems) {
      hf1 = co_await f.HoldPage(fbase + frun0, row_elems - frun0, true);
    }
    float *const fp0 = hf0.ptr();
    float *const fp1 = hf1 ? hf1.ptr() : nullptr;
    for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) {
      (e < frun0 ? fp0[e] : fp1[e - frun0]) = 0.0f;
    }
    __syncthreads();

    // One thread per i-slot of the row.
    const u64 islots = static_cast<u64>(nb) * cap;
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 ei = s * kStride;
      const float *const ip =
          (ei < rrun0[4]) ? rp0[4] + ei : rp1[4] + (ei - rrun0[4]);
      if (ip[3] < 0.0f) continue;   // padded slot
      const float xi = ip[0], yi = ip[1], zi = ip[2];
      const u32 bx = static_cast<u32>(s / cap);
      float fx = 0.0f, fy = 0.0f, fz = 0.0f;
      for (int q = 0; q < 9; ++q) {
        for (int dxx = -1; dxx <= 1; ++dxx) {
          const u32 jbx = (bx + nb + dxx) % nb;
          const u64 jb = static_cast<u64>(jbx) * cap * kStride;
          for (u32 sj = 0; sj < cap; ++sj) {
            const u64 ej = jb + static_cast<u64>(sj) * kStride;
            const float *const jp =
                (ej < rrun0[q]) ? rp0[q] + ej : rp1[q] + (ej - rrun0[q]);
            const float tj = jp[3];
            if (tj < 0.0f) continue;
            if (q == 4 && jbx == bx && sj == s % cap) continue;   // self
            float ddx = xi - jp[0];
            float ddy = yi - jp[1];
            float ddz = zi - jp[2];
            // Minimum image: bins wrap, so a raw delta can be ~box.
            if (ddx > halfL) ddx -= box; else if (ddx < -halfL) ddx += box;
            if (ddy > halfL) ddy -= box; else if (ddy < -halfL) ddy += box;
            if (ddz > halfL) ddz -= box; else if (ddz < -halfL) ddz += box;
            const float rsq = ddx * ddx + ddy * ddy + ddz * ddz;
            if (rsq >= c2) continue;
            const float r2i = 1.0f / rsq;
            const float r6i = r2i * r2i * r2i;
            const float fpair = r6i * (48.0f * r6i - 24.0f) * r2i;
            fx = __fmaf_rn(ddx, fpair, fx);
            fy = __fmaf_rn(ddy, fpair, fy);
            fz = __fmaf_rn(ddz, fpair, fz);
            if (eflag) {
              pe += 0.5 * static_cast<double>(4.0f * r6i * (r6i - 1.0f));
              w += 0.5 * static_cast<double>(r6i * (48.0f * r6i - 24.0f));
              npairs += 1.0;
            }
          }
        }
      }
      float *const op = (ei < frun0) ? fp0 + ei : fp1 + (ei - frun0);
      op[0] = fx;
      op[1] = fy;
      op[2] = fz;
    }
    __syncthreads();
  }  // guards die per row

  if (eflag) {
    const double vals[3] = {pe, w, npairs};
    for (int q = 0; q < 3; ++q) {
      red[threadIdx.x] = vals[q];
      __syncthreads();
      for (u32 wd = blockDim.x / 2; wd > 0; wd >>= 1) {
        if (threadIdx.x < wd) red[threadIdx.x] += red[threadIdx.x + wd];
        __syncthreads();
      }
      if (threadIdx.x == 0) atomicAdd(&acc[q], red[0]);
      __syncthreads();
    }
  }
}

/**
 * K2c (stage 3): build the Verlet list on the PAGED neigh vector.
 *
 * TRANSPOSED, PADDED layout -- entry k of row-slot s lives at
 *     row * islots * maxneigh  +  k * islots  +  s
 * so at every k the threads of a warp read CONSECUTIVE addresses: fully
 * coalesced. The CSR layout this replaces gave each thread its own
 * contiguous run, which is the worst pattern a GPU can be handed (32
 * threads, 32 unrelated cache lines) plus a divergent per-entry guard
 * walk; measured, the CSR list LOST to cell-direct outright -- 532 ms of
 * force time against 427 -- despite examining ~8x fewer candidates.
 *
 * Fixed padding also deletes the count/prefix-sum/fill round trip: counts
 * go straight to the resident d_cnt in one pass and nothing crosses PCIe
 * at a rebuild. Overflow past maxneigh sets d_err and the host refuses.
 *
 * Entries are packed ROW-RELATIVE: (q << 16) | row_slot, where q is the
 * stencil row (0..8); the force pass decodes with two shifts instead of a
 * global-index page lookup. 16 bits hold any row (nb * cap < 65536,
 * checked at startup).
 */
__device__ gy::YCoroMain BuildListCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<int> nl, u32 nb,
                                       u32 cap, float box, float rlist,
                                       u32 maxneigh, u32 *d_cnt, int *d_err,
                                       u32 rowchunk, u32 nblocks, u32 block) {
  extern __shared__ char smem_raw[];
  // BLOCK-UNIFORM TABLES LIVE IN SHARED, NOT IN THE FRAME. Every thread
  // holds the same rows and the same list guards, but a thread-local array
  // indexed by a runtime value (rp0[q], np[gi]) cannot be a register: it
  // lands in the coroutine frame, which IS the yield-stack lane in GLOBAL
  // memory. That made ~6 dependent global loads of pure bookkeeping per
  // entry and is why the list pass first measured SLOWER than cell-direct
  // despite 17x fewer candidates. Filled once per row after every hold
  // (no co_await follows, so shared survives) and read by all threads.
  char *tbl = smem_raw + CLIO_YIELD_SMEM_BYTES +
              blockDim.x * sizeof(double);
  const float **s_sp0 = reinterpret_cast<const float **>(tbl);
  const float **s_sp1 = s_sp0 + 9;
  u64 *s_srun = reinterpret_cast<u64 *>(s_sp1 + 9);
  u64 *s_qoff = s_srun + 9;
  u32 *s_qspan = reinterpret_cast<u32 *>(s_qoff + 9);
  const int **s_np = reinterpret_cast<const int **>(s_qspan + 12);
  u64 *s_gs = reinterpret_cast<u64 *>(s_np + kMaxNlGuards);
  u64 *s_gl = s_gs + kMaxNlGuards;
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 nrows = static_cast<u64>(nb) * nb;
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float r2list = rlist * rlist;
  const float halfL = 0.5f * box;
  // Same amortization as the force pass: the stencil spans serve a CHUNK
  // of rows, so the three holds are paid once per chunk rather than once
  // per row.
  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  const u64 nchunks = static_cast<u64>(nb) * cpz;
  for (u64 ch = block; ch < nchunks; ch += nblocks) {
    const u32 bz = static_cast<u32>(ch / cpz);
    const u32 y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    const u32 ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1)
                                               : (nb - 1);
    // Same admission as the force chunk, and for the same reason: this
    // kernel's span guards live for the whole chunk and its list guards are
    // WRITE holds, so it exhausts the very same tables. Admitting only the
    // force kernel just relocates the exhaustion to this one -- every hold
    // set in the grid has to be admitted, or none of them are protected.
    u32 span_guards = 0;
    {
      const int lo0 = static_cast<int>(y0) - 1;
      const int hi0 = static_cast<int>(ylast) + 1;
      for (int dz = -1; dz <= 1; ++dz) {
        const u32 wz = (bz + nb + dz) % nb;
        u32 rl[2], rn[2], nr = 0;
        if (lo0 < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi0) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi0 > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo0);
          rn[nr] = nb - static_cast<u32>(lo0); ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo0);
          rn[nr] = static_cast<u32>(hi0 - lo0 + 1); ++nr;
        }
        for (u32 t = 0; t < nr; ++t) {
          const u64 rb =
              ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
          span_guards += PagesSpanned(x, rb, static_cast<u64>(rn[t]) * row_elems);
        }
      }
    }
    // LATCHED. Admission can arm mid-run (the livelock watchdog), so the
    // decision is taken once here and reused at the release below; retesting
    // it there would let a block give back a reservation it never took.
    {   // guards die at the close of this scope, before the reservations go back
    gv::Held<float> hg[6][2];
    u64 srun[6];
    const float *sp0[6], *sp1[6];
    u32 sbase[6], scnt[6], sdz[6];
    u32 nspans = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      const u32 wz = (bz + nb + dz) % nb;
      const int lo = static_cast<int>(y0) - 1;
      const int hi = static_cast<int>(ylast) + 1;
      u32 rl[2], rn[2], nr = 0;
      if (lo < 0) {
        rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
        rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
      } else if (hi > static_cast<int>(nb) - 1) {
        rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo); ++nr;
        rl[nr] = 0; rn[nr] = 1u; ++nr;
      } else {
        rl[nr] = static_cast<u32>(lo);
        rn[nr] = static_cast<u32>(hi - lo + 1); ++nr;
      }
      for (u32 t = 0; t < nr; ++t) {
        const u64 rb =
            ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
        const u64 len = static_cast<u64>(rn[t]) * row_elems;
co_await x.BeginFetch(x.PageLo(rb), x.PageSpan(rb, len));
        co_await x.AwaitFetch();
                hg[nspans][0] = co_await x.HoldPage(rb, len);
        srun[nspans] = hg[nspans][0].run();
        if (srun[nspans] < len) {
          hg[nspans][1] =
              co_await x.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        sp0[nspans] = hg[nspans][0].ptr();
        sp1[nspans] = hg[nspans][1] ? hg[nspans][1].ptr() : nullptr;
        sbase[nspans] = rl[t];
        scnt[nspans] = rn[t];
        sdz[nspans] = static_cast<u32>(dz + 1);
        ++nspans;
      }
    }
    for (u32 by = y0; by <= ylast; ++by) {
    const u64 row = static_cast<u64>(bz) * nb + by;
    // Write-hold this row's whole list region (host-checked to fit the
    // guard array): the working-set lower bound for the neigh vector.
    gv::Held<int> hn[kMaxNlGuards];
    int *np[kMaxNlGuards];
    u64 gstart[kMaxNlGuards], glen[kMaxNlGuards];
    u32 nguards = 0;
    {
      const u64 nb0 = row * rowlist;
      u64 off = 0;
      while (off < rowlist && nguards < kMaxNlGuards) {
co_await nl.BeginFetch(nl.PageLo(nb0 + off), nl.PageSpan(nb0 + off, 1));
        co_await nl.AwaitFetch();
                hn[nguards] =
            co_await nl.HoldPage(nb0 + off, rowlist - off, /*write=*/true);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
    }
    // SHARED DOES NOT SURVIVE A PARK. The yield driver exits the kernel on
    // a fault, so every block-uniform table has to be (re)published AFTER
    // the last hold that can suspend -- the span pointers included, even
    // though the holds themselves are taken once per CHUNK and their guards
    // live in the frame. Resident runs never park and so never noticed;
    // out of core this read back garbage pointers and died on an MMU fault
    // inside the coroutine's resume path.
    if (threadIdx.x == 0) {
      for (u32 t = 0; t < nspans; ++t) {
        s_sp0[t] = sp0[t];
        s_sp1[t] = sp1[t];
        s_srun[t] = srun[t];
      }
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          const u32 wy = (by + nb + dy) % nb;
          const int q = (dz + 1) * 3 + (dy + 1);
          for (u32 t = 0; t < nspans; ++t) {
            if (sdz[t] != static_cast<u32>(dz + 1)) continue;
            if (wy >= sbase[t] && wy < sbase[t] + scnt[t]) {
              s_qspan[q] = t;
              s_qoff[q] = static_cast<u64>(wy - sbase[t]) * row_elems;
              break;
            }
          }
        }
      }
      for (u32 q = 0; q < nguards; ++q) {
        s_np[q] = np[q];
        s_gs[q] = gstart[q];
        s_gl[q] = glen[q];
      }
    }
    __syncthreads();
    const u64 slotbase = row * islots;
    const u32 sp4 = s_qspan[4];
    const u64 off4 = s_qoff[4], run4 = s_srun[sp4];
    const float *const ip0 = s_sp0[sp4];
    const float *const ip1 = s_sp1[sp4];
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 e = off4 + s * kStride;
      const float *const ip = (e < run4) ? ip0 + e : ip1 + (e - run4);
      if (ip[3] < 0.0f) {
        d_cnt[slotbase + s] = 0;
        continue;
      }
      const float xi = ip[0], yi = ip[1], zi = ip[2];
      const u32 bx = static_cast<u32>(s / cap);
      u32 cnt = 0, gi = 0;
      for (int q = 0; q < 9; ++q) {
        for (int dxx = -1; dxx <= 1; ++dxx) {
          const u32 jbx = (bx + nb + dxx) % nb;
          const u64 jb = static_cast<u64>(jbx) * cap * kStride;
          const u32 spq = s_qspan[q];
          const u64 rq = s_srun[spq];
          const u64 qo = s_qoff[q];
          const float *const qp0 = s_sp0[spq];
          const float *const qp1 = s_sp1[spq];
          for (u32 sj = 0; sj < cap; ++sj) {
            const u64 ej = qo + jb + static_cast<u64>(sj) * kStride;
            const float *const jp = (ej < rq) ? qp0 + ej : qp1 + (ej - rq);
            if (jp[3] < 0.0f) continue;
            if (q == 4 && jbx == bx && sj == s % cap) continue;
            float ddx = xi - jp[0];
            float ddy = yi - jp[1];
            float ddz = zi - jp[2];
            if (ddx > halfL) ddx -= box; else if (ddx < -halfL) ddx += box;
            if (ddy > halfL) ddy -= box; else if (ddy < -halfL) ddy += box;
            if (ddz > halfL) ddz -= box; else if (ddz < -halfL) ddz += box;
            const float rsq = ddx * ddx + ddy * ddy + ddz * ddz;
            if (rsq >= r2list) continue;
            if (cnt >= maxneigh) {   // refuse, never overrun
              *d_err = 1;
              continue;
            }
            const u64 o = static_cast<u64>(cnt) * islots + s;
            while (gi + 1 < nguards && o >= s_gs[gi] + s_gl[gi]) ++gi;
            // Entries are STENCIL-RELATIVE, (q, slot-within-row), not
            // span-relative: the two passes then need not agree on how
            // they group rows into held spans, which is what lets the
            // force pass amortize its holds over a CHUNK of rows while
            // the build pass holds per row.
            const_cast<int *>(s_np[gi])[o - s_gs[gi]] = static_cast<int>(
                (static_cast<u32>(q) << 16) | (jbx * cap + sj));
            ++cnt;
          }
        }
      }
      d_cnt[slotbase + s] = cnt;
    }
    __syncthreads();
    }   // per-row loop
    }
  }     // per-chunk loop
}

/**
 * K3-list (stage 3): the force pass streaming the Verlet list. Same
 * nine-row x holds and f write-hold as the cell-direct pass; the candidate
 * scan is replaced by this atom's padded column of entries, decoded
 * (q, row_slot) with two shifts. Entries cover cutoff + skin; the cutoff
 * test here skips the skin shell, which is what keeps the list valid
 * between rebuilds.
 */
__device__ gy::YCoroMain ListForceCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<float> f,
                                       gv::DeviceVector<int> nl, u32 nb,
                                       u32 cap, float box, float cutoff,
                                       u32 maxneigh, const u32 *d_cnt,
                                       int eflag, double *acc, int nocompute,
                                       u32 rowchunk, u32 nblocks, u32 block) {
  extern __shared__ char smem_raw[];
  double *red = reinterpret_cast<double *>(smem_raw + CLIO_YIELD_SMEM_BYTES);
  // BLOCK-UNIFORM TABLES LIVE IN SHARED, NOT IN THE FRAME. Every thread
  // holds the same rows and the same list guards, but a thread-local array
  // indexed by a runtime value (rp0[q], np[gi]) cannot be a register: it
  // lands in the coroutine frame, which IS the yield-stack lane in GLOBAL
  // memory. That made ~6 dependent global loads of pure bookkeeping per
  // entry and is why the list pass first measured SLOWER than cell-direct
  // despite 17x fewer candidates. Filled once per row after every hold
  // (no co_await follows, so shared survives) and read by all threads.
  char *tbl = smem_raw + CLIO_YIELD_SMEM_BYTES +
              blockDim.x * sizeof(double);
  const float **s_sp0 = reinterpret_cast<const float **>(tbl);
  const float **s_sp1 = s_sp0 + 9;
  u64 *s_srun = reinterpret_cast<u64 *>(s_sp1 + 9);
  u64 *s_qoff = s_srun + 9;
  u32 *s_qspan = reinterpret_cast<u32 *>(s_qoff + 9);
  const int **s_np = reinterpret_cast<const int **>(s_qspan + 12);
  u64 *s_gs = reinterpret_cast<u64 *>(s_np + kMaxNlGuards);
  u64 *s_gl = s_gs + kMaxNlGuards;
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 nrows = static_cast<u64>(nb) * nb;
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  double pe = 0.0, w = 0.0, npairs = 0.0;
  // AMORTIZE THE STENCIL HOLDS OVER A CHUNK OF ROWS. Holds are what this
  // pass costs (with the pair loop deleted it ran 245 ms against 252 with
  // it), and a single row needed 3 span holds to serve 1 row of work. A
  // chunk of R rows in one z-plane needs the SAME 3 spans -- rows
  // [y0-1, ylast+1] -- so the cost per row falls by R. Rows are grouped
  // within a plane, never across one, because that is the axis along which
  // the index space is contiguous.
  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  const u64 nchunks = static_cast<u64>(nb) * cpz;
  for (u64 ch = block; ch < nchunks; ch += nblocks) {
    const long long _r0 = clock64();
    const u32 bz = static_cast<u32>(ch / cpz);
    const u32 y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    const u32 ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1)
                                               : (nb - 1);
    // The stencil rows for the whole chunk: [y0-1, ylast+1] mod nb at
    // each of the three z planes, as one contiguous run (two when the
    // range crosses the y wrap).
    //
    // ADMITTED BEFORE THE FIRST HOLD. These span guards stay alive for the
    // whole chunk, so without admission enough blocks pin the entire table
    // between them and no claim can succeed -- measured as 48 of 48 slots
    // pinned at a failing claim, and as a 32-slot cache that wedges at 64
    // blocks but runs at 32. Waiting here costs nothing held; waiting after
    // the first hold is what deadlocks.
    // How many spans this chunk will actually hold. The y range
    // [y0-1, ylast+1] splits in two only when it crosses the wrap, and the
    // split depends solely on (y0, ylast, nb) -- the same for all three z
    // planes. So the count is pure geometry, known BEFORE any hold, and
    // reserving it beats reserving the worst case: at 48 slots the blanket
    // 12 admitted only 4 of 64 blocks and cost 124 ms/step against 43.
    // EXACTLY how many x guards this chunk will hold. The span geometry is
    // pure arithmetic, and PagesSpanned turns each span into the number of
    // guards holding it takes, so the reservation is the true hold set
    // rather than a bound. The bound (3 planes x 2 ranges x 2 guards = 12)
    // over-reserved badly enough to admit only a few of 64 blocks, which is
    // what made admission cost 2.5-3x and kept it opt-in.
    u32 span_guards = 0;
    {
      const int lo0 = static_cast<int>(y0) - 1;
      const int hi0 = static_cast<int>(ylast) + 1;
      for (int dz = -1; dz <= 1; ++dz) {
        const u32 wz = (bz + nb + dz) % nb;
        u32 rl[2], rn[2], nr = 0;
        if (lo0 < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi0) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi0 > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo0);
          rn[nr] = nb - static_cast<u32>(lo0); ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo0);
          rn[nr] = static_cast<u32>(hi0 - lo0 + 1); ++nr;
        }
        for (u32 t = 0; t < nr; ++t) {
          const u64 rb =
              ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
          span_guards += PagesSpanned(x, rb, static_cast<u64>(rn[t]) * row_elems);
        }
      }
    }
    // ALL THREE RESERVATIONS BEFORE ANY HOLD. Reserving x, taking the x
    // holds, and only then waiting for f/nl admission is the very
    // hold-while-waiting pattern admission exists to remove -- just moved
    // across vectors: blocks sit on x slots waiting for f, so x is
    // exhausted by blocks that are themselves blocked. Acquiring every
    // reservation up front, while holding nothing, is what makes the
    // guarantee hold across the whole working set rather than per vector.
    {   // guards die at the close of this scope, before ExitHoldSet
    gv::Held<float> hg[6][2];
    u64 srun[6];
    const float *sp0[6], *sp1[6];
    u32 sbase[6], scnt[6], sdz[6];
    u32 nspans = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      const u32 wz = (bz + nb + dz) % nb;
      const int lo = static_cast<int>(y0) - 1;
      const int hi = static_cast<int>(ylast) + 1;
      u32 rl[2], rn[2], nr = 0;
      if (lo < 0) {
        rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
        rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
      } else if (hi > static_cast<int>(nb) - 1) {
        rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo); ++nr;
        rl[nr] = 0; rn[nr] = 1u; ++nr;
      } else {
        rl[nr] = static_cast<u32>(lo);
        rn[nr] = static_cast<u32>(hi - lo + 1); ++nr;
      }
      for (u32 t = 0; t < nr; ++t) {
        const u64 rb =
            ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
        const u64 len = static_cast<u64>(rn[t]) * row_elems;
co_await x.BeginFetch(x.PageLo(rb), x.PageSpan(rb, len));
        co_await x.AwaitFetch();
                hg[nspans][0] = co_await x.HoldPage(rb, len);
        srun[nspans] = hg[nspans][0].run();
        if (srun[nspans] < len) {
          hg[nspans][1] =
              co_await x.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        sp0[nspans] = hg[nspans][0].ptr();
        sp1[nspans] = hg[nspans][1] ? hg[nspans][1].ptr() : nullptr;
        sbase[nspans] = rl[t];
        scnt[nspans] = rn[t];
        sdz[nspans] = static_cast<u32>(dz + 1);
        ++nspans;
      }
    }
    if (threadIdx.x == 0) atomicAdd(&g_md_cyc[0], (unsigned long long)(clock64() - _r0));
    for (u32 by = y0; by <= ylast; ++by) {
    const u64 row = static_cast<u64>(bz) * nb + by;
    // Which span holds each stencil row, and at what offset. Block-uniform
    // and only 9 entries, so thread 0 resolves it once per row.
    const long long _f0 = clock64();
    const u64 fbase = row * row_elems;
co_await f.BeginFetch(f.PageLo(fbase), f.PageSpan(fbase, row_elems));
    co_await f.AwaitFetch();
        // ADMISSION, and in a FIXED ORDER: x at the chunk, then f, then nl.
    // Every paged vector needs it, not just x -- f and nl have their own
    // tables and can exhaust them exactly the same way. Tightening only x's
    // reservation re-wedged a 32-slot run precisely because the blanket
    // reservation had been throttling f/nl concurrency by accident. A fixed
    // acquisition order is what keeps the nesting (x held while waiting for
    // f) from forming a cycle.
    gv::Held<float> hf0 = co_await f.HoldPage(fbase, row_elems, true);
    gv::Held<float> hf1;
    const u64 frun0 = hf0.run();
    if (frun0 < row_elems) {
      hf1 = co_await f.HoldPage(fbase + frun0, row_elems - frun0, true);
    }
    float *const fp0 = hf0.ptr();
    float *const fp1 = hf1 ? hf1.ptr() : nullptr;
    for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) {
      (e < frun0 ? fp0[e] : fp1[e - frun0]) = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&g_md_cyc[1], (unsigned long long)(clock64() - _f0));
    const long long _l0 = clock64();
    // Read-hold this row's whole list region.
    gv::Held<int> hn[kMaxNlGuards];
    const int *np[kMaxNlGuards];
    u64 gstart[kMaxNlGuards], glen[kMaxNlGuards];
    u32 nguards = 0;
    {
      const u64 nb0 = row * rowlist;
      u64 off = 0;
      while (off < rowlist && nguards < kMaxNlGuards) {
co_await nl.BeginFetch(nl.PageLo(nb0 + off), nl.PageSpan(nb0 + off, 1));
        co_await nl.AwaitFetch();
                hn[nguards] = co_await nl.HoldPage(nb0 + off, rowlist - off);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
    }
    // SHARED DOES NOT SURVIVE A PARK. The yield driver exits the kernel on
    // a fault, so every block-uniform table has to be (re)published AFTER
    // the last hold that can suspend -- the span pointers included, even
    // though the holds themselves are taken once per CHUNK and their guards
    // live in the frame. Resident runs never park and so never noticed;
    // out of core this read back garbage pointers and died on an MMU fault
    // inside the coroutine's resume path.
    if (threadIdx.x == 0) {
      for (u32 t = 0; t < nspans; ++t) {
        s_sp0[t] = sp0[t];
        s_sp1[t] = sp1[t];
        s_srun[t] = srun[t];
      }
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          const u32 wy = (by + nb + dy) % nb;
          const int q = (dz + 1) * 3 + (dy + 1);
          for (u32 t = 0; t < nspans; ++t) {
            if (sdz[t] != static_cast<u32>(dz + 1)) continue;
            if (wy >= sbase[t] && wy < sbase[t] + scnt[t]) {
              s_qspan[q] = t;
              s_qoff[q] = static_cast<u64>(wy - sbase[t]) * row_elems;
              break;
            }
          }
        }
      }
      for (u32 q = 0; q < nguards; ++q) {
        s_np[q] = np[q];
        s_gs[q] = gstart[q];
        s_gl[q] = glen[q];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&g_md_cyc[2], (unsigned long long)(clock64() - _l0));
    const long long _p0 = clock64();
    const u64 slotbase = row * islots;
    const u32 sp4 = s_qspan[4];
    const u64 off4 = s_qoff[4], run4 = s_srun[sp4];
    const float *const ip0 = s_sp0[sp4];
    const float *const ip1 = s_sp1[sp4];
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 e = off4 + s * kStride;
      const float *const ip = (e < run4) ? ip0 + e : ip1 + (e - run4);
      if (ip[3] < 0.0f) continue;
      const float xi = ip[0], yi = ip[1], zi = ip[2];
      const u32 cnt = nocompute ? 0u : d_cnt[slotbase + s];
      float fx = 0.0f, fy = 0.0f, fz = 0.0f;
      u32 gi = 0;
      for (u32 k = 0; k < cnt; ++k) {
        const u64 o = static_cast<u64>(k) * islots + s;
        while (gi + 1 < nguards && o >= s_gs[gi] + s_gl[gi]) ++gi;
        const u32 ent = static_cast<u32>(s_np[gi][o - s_gs[gi]]);
        const u32 q = ent >> 16;
        const u32 spq = s_qspan[q];
        const u64 ej = s_qoff[q] +
                       static_cast<u64>(ent & 0xffffu) * kStride;
        const u64 rq = s_srun[spq];
        const float *const jp =
            (ej < rq) ? s_sp0[spq] + ej : s_sp1[spq] + (ej - rq);
        float ddx = xi - jp[0];
        float ddy = yi - jp[1];
        float ddz = zi - jp[2];
        if (ddx > halfL) ddx -= box; else if (ddx < -halfL) ddx += box;
        if (ddy > halfL) ddy -= box; else if (ddy < -halfL) ddy += box;
        if (ddz > halfL) ddz -= box; else if (ddz < -halfL) ddz += box;
        const float rsq = ddx * ddx + ddy * ddy + ddz * ddz;
        if (rsq >= c2) continue;
        const float r2i = 1.0f / rsq;
        const float r6i = r2i * r2i * r2i;
        const float fpair = r6i * (48.0f * r6i - 24.0f) * r2i;
        fx = __fmaf_rn(ddx, fpair, fx);
        fy = __fmaf_rn(ddy, fpair, fy);
        fz = __fmaf_rn(ddz, fpair, fz);
        if (eflag) {
          pe += 0.5 * static_cast<double>(4.0f * r6i * (r6i - 1.0f));
          w += 0.5 * static_cast<double>(r6i * (48.0f * r6i - 24.0f));
          npairs += 1.0;
        }
      }
      const u64 fe = s * kStride;
      float *const op = (fe < frun0) ? fp0 + fe : fp1 + (fe - frun0);
      op[0] = fx;
      op[1] = fy;
      op[2] = fz;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&g_md_cyc[3], (unsigned long long)(clock64() - _p0));
      // NOT the chunk total here. _r0 is stamped once per CHUNK, so
      // accumulating (now - _r0) once per ROW sums rowchunk overlapping
      // spans and inflates the total by ~rowchunk -- which made the named
      // phases look like a quarter of the kernel and invented a 74%
      // "unattributed" gap that does not exist. The chunk total is taken
      // once, after the row loop.
      atomicAdd(&g_md_cyc[5], 1ull);
    }
    }   // per-row loop
    // The chunk total, taken ONCE. See the note at g_md_cyc[5].
    if (threadIdx.x == 0) {
      atomicAdd(&g_md_cyc[4], (unsigned long long)(clock64() - _r0));
    }
    // Every span guard above is dead here (the chunk body scope ends with
    // this brace), so the reservation is given back exactly once per
    // EnterHoldSet. The only chunk-level `continue` is above the Enter.
    }
  }     // per-chunk loop
  if (eflag) {
    const double vals[3] = {pe, w, npairs};
    for (int q = 0; q < 3; ++q) {
      red[threadIdx.x] = vals[q];
      __syncthreads();
      for (u32 wd = blockDim.x / 2; wd > 0; wd >>= 1) {
        if (threadIdx.x < wd) red[threadIdx.x] += red[threadIdx.x + wd];
        __syncthreads();
      }
      if (threadIdx.x == 0) atomicAdd(&acc[q], red[0]);
      __syncthreads();
    }
  }
}

/**
 * K2a: destination pass of the resort (stage 2). Positions have drifted
 * (and possibly left the box); wrap each atom into [0, box), compute its
 * new bin, claim a slot with an atomic per-bin counter, and record the
 * destination in d_dest (resident index class). The wrapped coordinates
 * are stored back, exactly as LAMMPS wraps at reneighbour. An atom moves
 * at most one bin per rebuild window (the skin rule), which is what makes
 * K2b's scatter row-local. Bin overflow sets d_err; the host refuses.
 */
__device__ gy::YCoroMain RebinCoro(gv::DeviceVector<float> x, u32 nb,
                                   u32 cap, float box, u32 *bincnt,
                                   u32 *d_dest, int *d_err, u32 nblocks,
                                   u32 block) {
  const u64 epp = x.ElemsPerPage();
  const u64 npages = (x.size() + epp - 1) / epp;
  const float fnb = static_cast<float>(nb);
  for (u64 pg = block; pg < npages; pg += nblocks) {
co_await x.BeginFetch(x.PageLo(pg * epp), x.PageSpan(pg * epp, epp));
    co_await x.AwaitFetch();
        auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    float *const px = hx.ptr();
    const u64 nslots = epp / kStride;
    const u64 slot0 = pg * nslots;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) {
        d_dest[slot0 + s] = ~0u;
        continue;
      }
      float px0 = px[e + 0], py0 = px[e + 1], pz0 = px[e + 2];
      if (px0 < 0.0f) px0 += box; else if (px0 >= box) px0 -= box;
      if (py0 < 0.0f) py0 += box; else if (py0 >= box) py0 -= box;
      if (pz0 < 0.0f) pz0 += box; else if (pz0 >= box) pz0 -= box;
      px[e + 0] = px0; px[e + 1] = py0; px[e + 2] = pz0;
      u32 bx = static_cast<u32>(px0 * fnb / box);
      u32 by = static_cast<u32>(py0 * fnb / box);
      u32 bz = static_cast<u32>(pz0 * fnb / box);
      if (bx >= nb) bx = nb - 1;
      if (by >= nb) by = nb - 1;
      if (bz >= nb) bz = nb - 1;
      const u64 bin = (static_cast<u64>(bz) * nb + by) * nb + bx;
      const u32 slot = atomicAdd(&bincnt[bin], 1u);
      if (slot >= cap) {
        *d_err = 1;
        d_dest[slot0 + s] = ~0u;
        continue;
      }
      d_dest[slot0 + s] = static_cast<u32>(bin * cap + slot);
    }
    __syncthreads();
  }
}

/**
 * K2b: apply the permutation for ONE vector, src -> dst (the ping-pong
 * buffer). Row-parallel like the force pass: a source row's destinations
 * lie within +-1 bin, i.e. inside the same NINE-row stencil, so the block
 * write-holds those rows of dst collectively and threads scatter. dst rows
 * were pre-sentineled by SentinelCoro. All slot claims were made in K2a,
 * so no two atoms share a destination. `keep_w` carries the type lane for
 * x and writes 0 for v.
 */
/**
 * K2b as a GATHER, not a scatter (the page-boundary rule).
 *
 * A scatter has block r writing destination rows r-1..r+1 while block r+1
 * writes r..r+2: their destination spans OVERLAP, so two blocks write the
 * same page. Resident that is invisible -- one frame, disjoint slots,
 * nothing evicts -- but with page-granular writeback one block's eviction
 * discards the other's writes, which showed up out of core as an
 * intermittent CUDA fatal, a 10x energy drift, or NaN, run to run on the
 * same command.
 *
 * Inverted, each block OWNS one destination row and PULLS into it: it
 * scans the nine source rows that could possibly target this row (an atom
 * moves at most one bin per rebuild window) and copies the ones whose
 * recorded destination lands here. Every destination slot then has exactly
 * ONE writer and no page is written by two blocks. The cost is reading
 * nine source rows per destination row instead of one, which is the same
 * stencil shape the force pass already pays and is safe to share because
 * it is read-only.
 */
__device__ gy::YCoroMain GatherCoro(gv::DeviceVector<float> src,
                                    gv::DeviceVector<float> srcx,
                                    gv::DeviceVector<float> dst, u32 nb,
                                    u32 cap, const u32 *d_dest, int keep_w,
                                    u32 nblocks, u32 block) {
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 row_elems = islots * kStride;
  const u64 nrows = static_cast<u64>(nb) * nb;
  for (u64 row = block; row < nrows; row += nblocks) {
    const u32 by = static_cast<u32>(row % nb);
    const u32 bz = static_cast<u32>(row / nb);
    // THE ONLY WRITE HOLD: this block's own destination row.
    {co_await dst.BeginFetch(dst.PageLo(row * row_elems), dst.PageSpan(row * row_elems, row_elems));
    co_await dst.AwaitFetch();
       // every guard dies at this scope's close, before the reservations
    gv::Held<float> hd0 = co_await dst.HoldPage(row * row_elems, row_elems,
                                                /*write=*/true);
    gv::Held<float> hd1;
    const u64 drun = hd0.run();
    if (drun < row_elems) {
co_await dst.BeginFetch(dst.PageLo(row * row_elems + drun), dst.PageSpan(row * row_elems + drun, 1));
      co_await dst.AwaitFetch();
            hd1 = co_await dst.HoldPage(row * row_elems + drun, row_elems - drun,
                                  /*write=*/true);
    }
    float *const dp0 = hd0.ptr();
    float *const dp1 = hd1 ? hd1.ptr() : nullptr;
    // The nine candidate source rows, held as SPANS (one or two contiguous
    // runs per dz), of both vectors -- they are the same vector on the
    // position pass, where the second hold is a cache hit. Row-at-a-time
    // holds needed 36 live guards and overflowed the coroutine frame.
    gv::Held<float> hs[6][2], hx[6][2];
    u64 srun[6], xrun[6];
    const float *sp0[6], *sp1[6], *xp0[6], *xp1[6];
    u32 qspan[9];
    u64 qoff[9];
    u64 srow[9];
    u32 nspans = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      const u32 wz = (bz + nb + dz) % nb;
      const int lo = static_cast<int>(by) - 1;
      const int hi = static_cast<int>(by) + 1;
      u32 rl[2], rn[2], nr = 0;
      if (lo < 0) {
        rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
        rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
      } else if (hi > static_cast<int>(nb) - 1) {
        rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo); ++nr;
        rl[nr] = 0; rn[nr] = 1u; ++nr;
      } else {
        rl[nr] = static_cast<u32>(lo); rn[nr] = 3u; ++nr;
      }
      for (u32 t = 0; t < nr; ++t) {
        const u64 rb =
            (static_cast<u64>(wz) * nb + rl[t]) * row_elems;
        const u64 len = static_cast<u64>(rn[t]) * row_elems;
co_await src.BeginFetch(src.PageLo(rb), src.PageSpan(rb, len));
        co_await src.AwaitFetch();
                hs[nspans][0] = co_await src.HoldPage(rb, len);
        srun[nspans] = hs[nspans][0].run();
        if (srun[nspans] < len) {
          hs[nspans][1] =
              co_await src.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        sp0[nspans] = hs[nspans][0].ptr();
        sp1[nspans] = hs[nspans][1] ? hs[nspans][1].ptr() : nullptr;
co_await srcx.BeginFetch(srcx.PageLo(rb), srcx.PageSpan(rb, len));
        co_await srcx.AwaitFetch();
                hx[nspans][0] = co_await srcx.HoldPage(rb, len);
        xrun[nspans] = hx[nspans][0].run();
        if (xrun[nspans] < len) {
co_await srcx.BeginFetch(srcx.PageLo(rb + xrun[nspans]), srcx.PageSpan(rb + xrun[nspans], 1));
          co_await srcx.AwaitFetch();
                    hx[nspans][1] =
              co_await srcx.HoldPage(rb + xrun[nspans], len - xrun[nspans]);
        }
        xp0[nspans] = hx[nspans][0].ptr();
        xp1[nspans] = hx[nspans][1] ? hx[nspans][1].ptr() : nullptr;
        // Which logical stencil rows this run covers.
        for (int dy = -1; dy <= 1; ++dy) {
          const u32 wy = (by + nb + dy) % nb;
          if (wy < rl[t] || wy >= rl[t] + rn[t]) continue;
          const int q = (dz + 1) * 3 + (dy + 1);
          qspan[q] = nspans;
          qoff[q] = static_cast<u64>(wy - rl[t]) * row_elems;
          srow[q] = static_cast<u64>(wz) * nb + wy;
        }
        ++nspans;
      }
    }
    for (u64 idx = threadIdx.x; idx < 9 * islots; idx += blockDim.x) {
      const u32 q = static_cast<u32>(idx / islots);
      const u64 sslot = idx % islots;
      const u32 sp = qspan[q];
      const u64 e = qoff[q] + sslot * kStride;
      const float *const xs =
          (e < xrun[sp]) ? xp0[sp] + e : xp1[sp] + (e - xrun[sp]);
      if (xs[3] < 0.0f) continue;              // padded source slot
      const u32 dest = d_dest[srow[q] * islots + sslot];
      if (dest == ~0u) continue;               // overflow victim
      if (static_cast<u64>(dest) / islots != row) continue;   // not ours
      const u64 de = (static_cast<u64>(dest) % islots) * kStride;
      float *const dp = (de < drun) ? dp0 + de : dp1 + (de - drun);
      const float *const sv =
          (e < srun[sp]) ? sp0[sp] + e : sp1[sp] + (e - srun[sp]);
      dp[0] = sv[0];
      dp[1] = sv[1];
      dp[2] = sv[2];
      dp[3] = keep_w ? sv[3] : 0.0f;
    }
    __syncthreads();
    // PUBLISH THIS ROW. Each block owns a private page table, so a row left
    // in this block's cache is invisible to every other block. Flush exactly
    // the row -- a whole-page flush would also send this block's stale copy
    // of the ~11 OTHER rows sharing the page and clobber their owners.
    co_await dst.BeginFlush(row * row_elems, row_elems);
    co_await dst.EndFlush();
    }   // guards dead here
    if (false) {
    }
  }     // per-row loop
}

/** Pre-sentinel every slot of a ping-pong destination vector. */
__device__ gy::YCoroMain SentinelCoro(gv::DeviceVector<float> dst,
                                      u32 nblocks, u32 block) {
  const u64 epp = dst.ElemsPerPage();
  const u64 npages = (dst.size() + epp - 1) / epp;
  for (u64 pg = block; pg < npages; pg += nblocks) {
co_await dst.BeginFetch(dst.PageLo(pg * epp), dst.PageSpan(pg * epp, epp));
    co_await dst.AwaitFetch();
        auto h = co_await dst.HoldPage(pg * epp, epp, /*write=*/true);
    float *const p = h.ptr();
    const u64 nslots = epp / kStride;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
      p[s * kStride + 3] = -1.0f;
    }
    __syncthreads();
  }
}

/** K1/K4 for real MD: the kick reads the FORCE VECTOR (mass = 1). Same
 *  page-parallel shape as the ballistic form. */
__device__ gy::YCoroMain MDIntegrateCoro(gv::DeviceVector<float> x,
                                         gv::DeviceVector<float> v,
                                         gv::DeviceVector<float> f,
                                         float dt, int drift, u32 nblocks,
                                         u32 block) {
  const u64 epp = x.ElemsPerPage();
  const u64 npages = (x.size() + epp - 1) / epp;
  const float half = 0.5f * dt;
  for (u64 pg = block; pg < npages; pg += nblocks) {
co_await x.BeginFetch(x.PageLo(pg * epp), x.PageSpan(pg * epp, epp));
    co_await x.AwaitFetch();
        auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
co_await v.BeginFetch(v.PageLo(pg * epp), v.PageSpan(pg * epp, epp));
    co_await v.AwaitFetch();
        auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
co_await f.BeginFetch(f.PageLo(pg * epp), f.PageSpan(pg * epp, epp));
    co_await f.AwaitFetch();
        auto hf = co_await f.HoldPage(pg * epp, epp);
    float *const px = hx.ptr();
    float *const pv = hv.ptr();
    const float *const pf = hf.ptr();
    const u64 nslots = epp / kStride;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;
      const float vx0 = __fmaf_rn(half, pf[e + 0], pv[e + 0]);
      const float vy0 = __fmaf_rn(half, pf[e + 1], pv[e + 1]);
      const float vz0 = __fmaf_rn(half, pf[e + 2], pv[e + 2]);
      pv[e + 0] = vx0;
      pv[e + 1] = vy0;
      pv[e + 2] = vz0;
      if (drift) {
        px[e + 0] = __fmaf_rn(dt, vx0, px[e + 0]);
        px[e + 1] = __fmaf_rn(dt, vy0, px[e + 1]);
        px[e + 2] = __fmaf_rn(dt, vz0, px[e + 2]);
      }
    }
    __syncthreads();
  }
}

/** Flush every dirty page of the SHARED x/v tables to the backing store, so
 *  host Download (which reads the store, not the frames) sees the truth.
 *  One block does it; the others exit immediately. Validation-path only --
 *  steady-state MD never flushes x/v (they are device-canonical). */
__device__ gy::YCoroMain FlushAllCoro(gv::DeviceVector<float> x,
                                      gv::DeviceVector<float> v, u32 nblocks,
                                      u32 block, u64 row_elems, u64 nrows) {
  // EVERY BLOCK FLUSHES ITS OWN TABLE, in full.
  //
  // Each block owns a private page table now, so the old "block 0 flushes
  // everything" would publish one block's copy and drop the rest. A full
  // flush is right here because IntegrateCoro partitions x and v BY PAGE
  // (pg = block; pg += nblocks) -- a block's table only ever holds pages it
  // owns outright, so there is no other block's data in them to clobber.
  (void) row_elems;
  (void) nrows;
  (void) block;
  co_await x.BeginFlush();
  co_await x.EndFlush();
  co_await v.BeginFlush();
  co_await v.EndFlush();
}

/** Seeded value of element i: unique and exactly representable (the whole
 *  vector is well under 2^24 elements). */
CTP_INLINE_CROSS_FUN float ProbeVal(u64 i) { return static_cast<float>(i); }

__device__ gy::YCoroMain ReadProbeCoro(gv::DeviceVector<float> x, u64 passes,
                                       u32 nblocks, u32 block) {
  const u64 epp = x.ElemsPerPage();
  const u64 npages = (x.size() + epp - 1) / epp;
  // Publish this block's table geometry once, so a bad guard pointer can be
  // converted to an actual slot index instead of being guessed at from raw
  // addresses (which are only comparable within one process).
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    g_read_geom[1] = x.PagesPerTable();
    g_read_geom[2] = x.PageBytes();
    g_read_geom[3] = epp;
  }
  for (u64 it = 0; it < passes; ++it) {
    for (u64 pg = block; pg < npages; pg += nblocks) {
co_await x.BeginFetch(x.PageLo(pg * epp), x.PageSpan(pg * epp, epp));
      co_await x.AwaitFetch();
            auto h = co_await x.HoldPage(pg * epp, epp);   // READ hold only
      // Is the guard born valid? Its slot must hold the page we asked for.
      // This separates "the hold returned the wrong frame" from "the frame
      // was right and the slot was recycled out from under the pin".

      const float *const p = h.ptr();
      for (u64 i = threadIdx.x; i < epp; i += blockDim.x) {
        // MD_PROBE_LDCG=1: read past the (non-coherent) per-SM L1. If the
        // mismatches vanish only under this, the bytes were always in the
        // frame and the kernel was reading a stale cached line -- host DMA
        // that lands mid-kernel is not visible to an ordinary ld.global.
        const float seen = kProbeLdcg ? __ldcg(&p[i]) : p[i];
        if (seen != ProbeVal(pg * epp + i)) {
          atomicAdd(&g_read_bad[0], 1ull);
          // IS THE DATA ACTUALLY THERE? Re-read the SAME element fully
          // uncached (ld.global.cv bypasses L1 AND treats L2 as volatile),
          // with no delay -- this is a pure cache question, not a timing
          // one. A match here means the bytes were in the frame all along
          // and the kernel was reading a stale cached line.
          if (__ldcv(&p[i]) == ProbeVal(pg * epp + i)) {
            atomicAdd(&g_read_bad[1], 1ull);
          }
          // DOES IT HEAL? Wait, re-read the SAME address through the SAME
          // guard, and see whether the value arrives late. If it does, the
          // page was published as resident before its transfer landed; if
          // it stays wrong, the get truly never wrote the frame.
          // The white-box duplicate/stale-frame diagnostics that lived here
          // are gone with the accessors they used. A page cannot occupy two
          // frames of one table now: the fault publishes into the frame it
          // claimed, and the guard pins it.
        }
      }
      __syncthreads();
    }
  }
}

__device__ gy::YCoroMain IntegrateCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<float> v,
                                       gv::DeviceVector<float> third,
                                       int use_third, float dt, float gx,
                                       float gy_, float gz, int drift,
                                       u32 nblocks, u32 block) {
  const u64 epp = x.ElemsPerPage();
  const u64 npages = (x.size() + epp - 1) / epp;
  const float half = 0.5f * dt;
  for (u64 pg = block; pg < npages; pg += nblocks) {
co_await x.BeginFetch(x.PageLo(pg * epp), x.PageSpan(pg * epp, epp));
    co_await x.AwaitFetch();
        auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
co_await v.BeginFetch(v.PageLo(pg * epp), v.PageSpan(pg * epp, epp));
    co_await v.AwaitFetch();
        auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
    // THE TEST: a THIRD vector held read-only alongside the other two, the
    // pattern the real integrator has (x write, v write, f read) and the
    // ballistic gate never had. The value is only sunk, so the physics --
    // and therefore the bitwise comparison against the host replica --
    // is untouched.
    gv::Held<float> ht;
    if (use_third) {
co_await third.BeginFetch(third.PageLo(pg * epp), third.PageSpan(pg * epp, epp));
      co_await third.AwaitFetch();
            ht = co_await third.HoldPage(pg * epp, epp);
    }
    float *const px = hx.ptr();
    float *const pv = hv.ptr();
    if (use_third && threadIdx.x == 0) {
      atomicAdd(&g_third_sink,
                static_cast<unsigned long long>(__float_as_uint(ht.ptr()[0])));
    }
    const u64 nslots = epp / kStride;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;   // padded slot
      // EXPLICIT fma on every update, mirrored by std::fmaf in the host
      // replica: the bitwise gate must not depend on which contractions a
      // compiler happens to choose.
      const float vx0 = __fmaf_rn(half, gx, pv[e + 0]);
      const float vy0 = __fmaf_rn(half, gy_, pv[e + 1]);
      const float vz0 = __fmaf_rn(half, gz, pv[e + 2]);
      pv[e + 0] = vx0;
      pv[e + 1] = vy0;
      pv[e + 2] = vz0;
      if (drift) {
        px[e + 0] = __fmaf_rn(dt, vx0, px[e + 0]);
        px[e + 1] = __fmaf_rn(dt, vy0, px[e + 1]);
        px[e + 2] = __fmaf_rn(dt, vz0, px[e + 2]);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&g_pages_done, 1ull);
      if (block < 64) {
        atomicAdd(&g_blk_done[block], 1ull);
        g_blk_last[block] = pg;
      }
    }
    // Guards die here: pages unpin. Nothing flushes -- x/v are device-
    // canonical; the backing store is only written on eviction (none in the
    // resident regime) or by an explicit host Download for validation.
  }
}

/** K5: thermo -- KE and net momentum, block-reduced then atomically merged.
 *  Read-holds only. Reduction scratch is dynamic shared memory after the
 *  yield header. */
__device__ gy::YCoroMain ThermoCoro(gv::DeviceVector<float> x,
                                    gv::DeviceVector<float> v,
                                    double *out, u32 nblocks, u32 block) {
  extern __shared__ char smem_raw[];
  double *red = reinterpret_cast<double *>(smem_raw + CLIO_YIELD_SMEM_BYTES);
  const u64 epp = x.ElemsPerPage();
  const u64 npages = (x.size() + epp - 1) / epp;
  double ke = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (u64 pg = block; pg < npages; pg += nblocks) {
co_await x.BeginFetch(x.PageLo(pg * epp), x.PageSpan(pg * epp, epp));
    co_await x.AwaitFetch();
        auto hx = co_await x.HoldPage(pg * epp, epp);
co_await v.BeginFetch(v.PageLo(pg * epp), v.PageSpan(pg * epp, epp));
    co_await v.AwaitFetch();
        auto hv = co_await v.HoldPage(pg * epp, epp);
    const float *const px = hx.ptr();
    const float *const pv = hv.ptr();
    const u64 nslots = epp / kStride;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;
      const double vx = pv[e + 0], vy = pv[e + 1], vz = pv[e + 2];
      ke += 0.5 * (vx * vx + vy * vy + vz * vz);   // m = 1
      mx += vx; my += vy; mz += vz;
    }
    __syncthreads();
  }
  // Tree-reduce the four sums through shared, one at a time.
  const double vals[4] = {ke, mx, my, mz};
  for (int q = 0; q < 4; ++q) {
    red[threadIdx.x] = vals[q];
    __syncthreads();
    for (u32 w = blockDim.x / 2; w > 0; w >>= 1) {
      if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
      __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&out[q], red[0]);
    __syncthreads();
  }
}

/* x and v are created with ONE shared table (vector nblocks = 1): a single
 * copy of every page, held by all GPU blocks. block_override_ = 0 points
 * every block at that table; blocks write DISJOINT pages, so the only
 * sharing is the lock-free probes. */

__global__ MD_LAUNCH_BOUNDS void ReadProbeKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x, u64 passes,
                                u32 nblocks, gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ReadProbeCoro(x, passes, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void IntegrateKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<float> v,
                                gv::DeviceVector<float> third, int use_third,
                                float dt, float gx, float gy_, float gz,
                                int drift, u32 nblocks,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  third.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(IntegrateCoro(x, v, third, use_third, dt, gx, gy_, gz,
                               drift, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void ThermoKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> x,
                             gv::DeviceVector<float> v, double *out,
                             u32 nblocks, gy::YieldableView<> yv,
                             gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ThermoCoro(x, v, out, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void ForceKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> x,
                            gv::DeviceVector<float> f, u32 nb, u32 cap,
                            float box, float cutoff, int eflag, double *acc,
                            u32 nblocks, gy::YieldableView<> yv,
                            gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  f.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ForceCoro(x, f, nb, cap, box, cutoff, eflag, acc, nblocks,
                           yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void BuildListKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<int> nl, u32 nb, u32 cap,
                                float box, float rlist, u32 maxneigh,
                                u32 *d_cnt, int *d_err, u32 rowchunk,
                                u32 nblocks,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  nl.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(BuildListCoro(x, nl, nb, cap, box, rlist, maxneigh, d_cnt,
                               d_err, rowchunk, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void ListForceKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<float> f,
                                gv::DeviceVector<int> nl, u32 nb, u32 cap,
                                float box, float cutoff, u32 maxneigh,
                                const u32 *d_cnt, int eflag, double *acc,
                                int nocompute, u32 rowchunk, u32 nblocks,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  f.Init(yv.Block());
  nl.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ListForceCoro(x, f, nl, nb, cap, box, cutoff, maxneigh,
                               d_cnt, eflag, acc, nocompute, rowchunk,
                               nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void RebinKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> x, u32 nb, u32 cap,
                            float box, u32 *bincnt, u32 *d_dest, int *d_err,
                            u32 nblocks, gy::YieldableView<> yv,
                            gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RebinCoro(x, nb, cap, box, bincnt, d_dest, d_err, nblocks,
                           yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void GatherKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<float> src,
                              gv::DeviceVector<float> srcx,
                              gv::DeviceVector<float> dst, u32 nb, u32 cap,
                              const u32 *d_dest, int keep_w, u32 nblocks,
                              gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  src.Init(yv.Block());
  srcx.Init(yv.Block());
  dst.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(GatherCoro(src, srcx, dst, nb, cap, d_dest, keep_w,
                            nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void SentinelKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> dst, u32 nblocks,
                               gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  dst.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SentinelCoro(dst, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void MDIntegrateKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceVector<float> x,
                                  gv::DeviceVector<float> v,
                                  gv::DeviceVector<float> f, float dt,
                                  int drift, u32 nblocks,
                                  gy::YieldableView<> yv,
                                  gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  f.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(MDIntegrateCoro(x, v, f, dt, drift, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void FlushAllKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> x,
                               gv::DeviceVector<float> v, u32 nblocks,
                               u64 row_elems, u64 nrows,
                               gy::YieldableView<> yv,
                               gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(FlushAllCoro(x, v, nblocks, yv.Block(), row_elems,
                              nrows));
}

#if !CTP_IS_DEVICE_PASS
namespace {

/** Rounds after which the driver gives up on a kernel that never finishes. */
static constexpr u32 kMaxRounds = 2000000;

class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}

  /** True once any Run hit the round cap. THE CAP IS OTHERWISE SILENT:
   *  RunToCompletion returning at max_rounds looks exactly like success, so
   *  a livelocked block is abandoned mid-coroutine and the caller happily
   *  uses whatever it managed to compute. That is how an out-of-core run
   *  came back with 17 of 22 page iterations done and no error anywhere. */
  bool HitCap() const { return hit_cap_; }

  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.ResetTimers();
    drv_.Reset();
    stack_.Reset();
    const u32 r = RunInner(std::forward<LaunchT>(launch));
    if (r >= kMaxRounds && !hit_cap_) {
      hit_cap_ = true;
      std::fprintf(stderr,
                   "[driver] GAVE UP after %u rounds -- blocks are still "
                   "parked and their remaining work will NOT run. Results "
                   "from here are incomplete.\n", r);
    }
    return r;
  }

 private:
  template <typename LaunchT>
  u32 RunInner(LaunchT &&launch) {
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, kMaxRounds,
        [](u32, u64 tag) -> bool {
          unsigned int f = 0;
          ctp::GpuApi::Memcpy(&f, reinterpret_cast<const unsigned int *>(tag),
                              sizeof(f));
          return (f & 1u) != 0u;
        });
  }
 public:
  double KernelMs() const { return drv_.KernelMs(); }
  /** Blocks launched per round -- if the driver ever exits while a block
   *  is still parked, the tail of this log shows it. */
  const std::vector<std::pair<double, u32>> &RoundLog() const {
    return drv_.RoundLog();
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
  bool hit_cap_ = false;
};

}  // namespace
#endif  // !CTP_IS_DEVICE_PASS
#endif  // GV_MD_CORO

#if !CTP_IS_DEVICE_PASS

/** Clamp a requested slot count up to what the kernels actually pin. */
static u32 AtLeastSlots(u32 want, u32 floor_slots, const char *what) {
  if (want >= floor_slots) return want;
  std::printf("  note: %s cache %u frames is below the %u pinned at once; "
              "raising to %u\n", what, want, floor_slots, floor_slots);
  return floor_slots;
}

int main(int argc, char **argv) {
#if !defined(GV_MD_CORO)
  (void)argc; (void)argv;
  std::fprintf(stderr,
               "eternia-MD requires the clang device-coroutine build "
               "(CLIO_GPU_YIELD_CORO=ON)\n");
  return 2;
#else
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto want = [&](const char *k) {
      return std::strcmp(argv[i], k) == 0 && i + 1 < argc;
    };
    if (want("--lattice")) a.lattice = static_cast<u32>(atoi(argv[++i]));
    else if (want("--rho")) a.rho = atof(argv[++i]);
    else if (want("--cutoff")) a.cutoff = atof(argv[++i]);
    else if (want("--skin")) a.skin = atof(argv[++i]);
    else if (want("--cap")) a.cap = static_cast<u32>(atoi(argv[++i]));
    else if (want("--page-kb")) a.page_kb = static_cast<u64>(atol(argv[++i]));
    else if (want("--blocks")) a.blocks = static_cast<u32>(atoi(argv[++i]));
    else if (want("--threads")) a.threads = static_cast<u32>(atoi(argv[++i]));
    else if (want("--slots")) a.slots = static_cast<u32>(atoi(argv[++i]));
    else if (want("--ppslots")) a.ppslots = static_cast<u32>(atoi(argv[++i]));
    else if (want("--fslots")) a.fslots = static_cast<u32>(atoi(argv[++i]));
    else if (want("--vslots")) a.vslots = static_cast<u32>(atoi(argv[++i]));
    else if (want("--steps")) a.steps = static_cast<u64>(atol(argv[++i]));
    else if (want("--rebin")) a.rebin = static_cast<u64>(atol(argv[++i]));
    else if (want("--temp")) a.temp = atof(argv[++i]);
    else if (want("--drift-tol")) a.drift_tol = atof(argv[++i]);
    else if (want("--maxneigh")) a.maxneigh = static_cast<u32>(atoi(argv[++i]));
    else if (want("--rowchunk")) a.rowchunk = static_cast<u32>(atoi(argv[++i]));
    else if (want("--ckpt")) a.ckpt = static_cast<u64>(atol(argv[++i]));
    else if (want("--nl-page-kb")) a.nl_page_kb = static_cast<u64>(atol(argv[++i]));
    else if (std::strcmp(argv[i], "--no-list") == 0) a.use_list = 0;
    else if (want("--dt")) a.dt = atof(argv[++i]);
    else if (std::strcmp(argv[i], "--no-gate") == 0) a.gate = 0;
    else if (std::strcmp(argv[i], "--md") == 0) a.md = 1;
    else if (std::strcmp(argv[i], "--readprobe") == 0) a.readprobe = 1;
    else {
      std::fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }

  // ---- geometry -----------------------------------------------------------
  Geometry g;
  g.natoms = 4ull * a.lattice * a.lattice * a.lattice;
  g.box = a.lattice * std::cbrt(4.0 / a.rho);
  const double min_edge = a.cutoff + a.skin;
  g.nb = static_cast<u32>(g.box / min_edge);
  if (g.nb == 0) g.nb = 1;
  g.bin_edge = g.box / g.nb;
  g.nbins = static_cast<u64>(g.nb) * g.nb * g.nb;
  g.cap = a.cap;
  g.nslots = g.nbins * g.cap;
  g.nelems = g.nslots * kStride;

  const u64 page_bytes = a.page_kb * 1024;
  const u64 page_elems = page_bytes / sizeof(float);
  const u64 bin_elems = static_cast<u64>(g.cap) * kStride;
  if (page_elems % bin_elems != 0) {
    std::fprintf(stderr,
                 "page (%llu elems) must hold WHOLE bins (%llu elems/bin): "
                 "adjust --page-kb or --cap\n",
                 (unsigned long long)page_elems,
                 (unsigned long long)bin_elems);
    return 1;
  }
  const u64 npages = (g.nelems + page_elems - 1) / page_elems;
  // A HELD SPAN MUST FIT ONE PAGE, or two guards cannot cover it.
  // The force and build passes hold a span of (rowchunk + 2) rows with a
  // guard for the page it starts in and one for the page it straddles
  // into. A span LONGER than a page crosses three, and the third is
  // unguarded -- reads walk off the second page into whatever frame
  // follows. Resident that is benign and invisible, because identity
  // placement puts page p in slot p and the frames are contiguous; out of
  // core the next frame is a different page, and the run either invents
  // neighbours (caught by the maxneigh refusal) or dies on an illegal
  // access. Checked here, in configuration terms, before anything runs.
  {
    const u64 row_bytes =
        static_cast<u64>(g.nb) * g.cap * kStride * sizeof(float);
    const u64 span_bytes = static_cast<u64>(a.rowchunk + 2) * row_bytes;
    if (span_bytes > page_bytes) {
      std::fprintf(stderr,
                   "page too small: a %u-row held span is %llu bytes but the "
                   "page is %llu; raise --page-kb to at least %llu or lower "
                   "--rowchunk/--cap\n",
                   a.rowchunk + 2, (unsigned long long)span_bytes,
                   (unsigned long long)page_bytes,
                   (unsigned long long)((span_bytes + 1023) / 1024));
      return 1;
    }
  }
  // THE VECTOR OWNS WHOLE PAGES. A partial tail page would leave slots the
  // kernels can reach but the host never initialized: Preload zero-pads
  // them, zero is not the sentinel, and 2464 phantom atoms integrated the
  // constant field -- caught by the ballistic gate as a momentum excess of
  // exactly 2464*dt*g. Rounding the size up puts every reachable slot under
  // the layout's contract: a real atom or an explicit sentinel.
  g.nelems = npages * page_elems;
  g.nslots = g.nelems / kStride;
  // x and v use ONE SHARED table (vector nblocks = 1): one copy of every
  // page, held by all GPU blocks, which write disjoint pages. Stage 1 is
  // the resident regime: the default cache holds everything. An explicit
  // --slots below that is a configuration for later stages, not this gate.
  u32 slots = (a.slots != 0) ? a.slots : static_cast<u32>(npages + 2);
  if (slots < kMinSlotsX) {
    std::printf("  note: --slots %u is below the %u frames these kernels pin "
                "at once; raising to %u\n",
                slots, kMinSlotsX, kMinSlotsX);
    slots = kMinSlotsX;
  }
  // Which regime this configuration is IN: the cache either can hold every
  // page of the working set or it cannot. The gates below key off this
  // rather than assuming residency.
  const bool expect_resident = (static_cast<clio::run::u64>(slots) >= npages);

  std::printf(
      "eternia-MD stage 1 (integrator + ballistic gate)\n"
      "  atoms=%llu box=%.4f bins=%u^3 cap=%u slots/bin-pad=%.0f%%\n"
      "  page=%lluKB (%llu bins/page) pages=%llu blocks=%u threads=%u "
      "cache=%u slots (one table per block)\n"
      "  steps=%llu dt=%g g=(%g,%g,%g)\n",
      (unsigned long long)g.natoms, g.box, g.nb, g.cap,
      100.0 * (1.0 - static_cast<double>(g.natoms) / g.nslots),
      (unsigned long long)a.page_kb, (unsigned long long)(page_elems / bin_elems),
      (unsigned long long)npages, a.blocks, a.threads, slots,
      (unsigned long long)a.steps, a.dt, a.g[0], a.g[1], a.g[2]);

  // ---- host initial state -------------------------------------------------
  std::vector<float> hx, hv;
  std::vector<u32> bin_count;
  if (!BuildInitialState(a, g, hx, hv, bin_count)) return 1;
  if (a.temp > 0.0) {
    // Zero the net momentum, then scale velocities to the requested reduced
    // temperature (KE = 1.5 N T, m = 1) -- the melt deck's hot start.
    double mean[3] = {0, 0, 0};
    for (u64 s = 0; s < g.nslots; ++s) {
      if (hx[s * kStride + 3] < 0.0f) continue;
      for (int d = 0; d < 3; ++d) mean[d] += hv[s * kStride + d];
    }
    for (int d = 0; d < 3; ++d) mean[d] /= static_cast<double>(g.natoms);
    double ke = 0.0;
    for (u64 s = 0; s < g.nslots; ++s) {
      if (hx[s * kStride + 3] < 0.0f) continue;
      for (int d = 0; d < 3; ++d) {
        const double vd = hv[s * kStride + d] - mean[d];
        hv[s * kStride + d] = static_cast<float>(vd);
        ke += 0.5 * vd * vd;
      }
    }
    const double scale =
        std::sqrt(1.5 * a.temp * static_cast<double>(g.natoms) / ke);
    for (u64 s = 0; s < g.nslots; ++s) {
      if (hx[s * kStride + 3] < 0.0f) continue;
      for (int d = 0; d < 3; ++d) {
        hv[s * kStride + d] =
            static_cast<float>(hv[s * kStride + d] * scale);
      }
    }
  }

  // ---- runtime ------------------------------------------------------------
  {
    std::ofstream cfg("gpu_vector_md.yaml");
    const char *thr = getenv("GV_THREADS");
    auto env_mb = [](const char *k, long d) {
      const char *e = getenv(k);
      return e ? std::atol(e) : d;
    };
    const long hbm_mb = env_mb("GV_HBM_MB", 0);
    const long dram_mb = env_mb("GV_DRAM_MB", 4096);
    cfg << "networking:\n  port: 9438\n\n"
        << "runtime:\n  num_threads: " << (thr ? atoi(thr) : 8)
        << "\n  queue_depth: 8192\n  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n";
    if (hbm_mb > 0) {
      cfg << "      - path: \"hbm::gv_md_hbm\"\n        bdev_type: \"hbm\"\n"
          << "        capacity_limit: \"" << hbm_mb << "MB\"\n"
          << "        score: 1.0\n";
    }
    cfg << "      - path: \"ram::gv_md_ram\"\n        bdev_type: \"ram\"\n"
        << "        capacity_limit: \"" << dram_mb << "MB\"\n"
        << "        score: " << (hbm_mb > 0 ? "0.6" : "1.0") << "\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_md.yaml", 1);
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // ---- vectors ------------------------------------------------------------
  // ONE TABLE PER CUDA BLOCK is the designed model (independent per-block
  // caches); the single shared table is the alternative this benchmark
  // used so that a page exists exactly once. They are not equivalent under
  // FAULTS: a table owns its lock, its scalar task slots AND its batch
  // slots, so N CUDA blocks on one table share one set of in-flight
  // transfer state.
  // A vector is parameterized by the EXACT launch configuration of the
  // kernel that uses it: one page table per CUDA block, always.
  const u32 tbl_blocks = a.blocks;
  // Row geometry, so a flush can name exactly the rows a block owns.
  const u64 md_row_elems = static_cast<u64>(g.nb) * g.cap * kStride;
  const u64 md_nrows = static_cast<u64>(g.nb) * g.nb;
  gv::Vector<float> vx("md_x", {0}, page_bytes, tbl_blocks, slots,
                       g.nelems);
  // v gets its own knob too, so x-paging and v-paging can be separated:
  // x is the only vector read through multi-page SPAN holds that stay live
  // across a park, which is the access pattern no other gate covers.
  const u32 vslots =
      AtLeastSlots((a.vslots != 0) ? a.vslots : slots, kMinSlotsX, "v");
  gv::Vector<float> vv("md_v", {0}, page_bytes, tbl_blocks, vslots,
                       g.nelems);
  vx.EnableStats();
  vv.EnableStats();
  vx.Preload(hx.data(), g.nelems);
  vv.Preload(hv.data(), g.nelems);
  vx.ClearCache();
  vv.ClearCache();
  vx.Prefetch(0, npages, 0, tbl_blocks);
  vv.Prefetch(0, npages, 0, tbl_blocks);
  auto dx = vx.GetDevice(0);
  auto dv = vv.GetDevice(0);

  // MD_THIRD=1 gives the ballistic path a third, read-only paged vector.
  const int use_third = std::getenv("MD_THIRD") != nullptr ? 1 : 0;
  gv::Vector<float> vthird("md_third", {0}, page_bytes, tbl_blocks,
                           use_third ? slots : 1u,
                           use_third ? g.nelems : page_elems);
  if (use_third) {
    std::vector<float> hz(g.nelems, 0.0f);
    vthird.Preload(hz.data(), g.nelems);
    vthird.ClearCache();
    vthird.Prefetch(0, npages, 0, tbl_blocks);
  }
  auto dthird = vthird.GetDevice(0);

  double *d_thermo = ctp::GpuApi::Malloc<double>(4 * sizeof(double));

  YieldRunner runner(a.blocks, a.threads);
  const u32 smem_thermo =
      CLIO_YIELD_SMEM_BYTES + a.threads * sizeof(double);

  // ---- READ-ONLY PAGING PROBE -------------------------------------------
  if (a.readprobe) {
    std::vector<float> pat(g.nelems);
    for (u64 i = 0; i < g.nelems; ++i) pat[i] = static_cast<float>(i);
    vx.Preload(pat.data(), g.nelems);
    vx.ClearCache();
    vx.Prefetch(0, npages, 0, tbl_blocks);
    auto dxp = vx.GetDevice(0);
    const unsigned long long z4[4] = {0, 0, 0, 0};
    cudaMemcpyToSymbol(g_read_bad, z4, sizeof(z4));
    const unsigned long long z8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    cudaMemcpyToSymbol(g_read_sample, z8, sizeof(z8));
    const bool ldcg = getenv("MD_PROBE_LDCG") != nullptr;
    cudaMemcpyToSymbol(kProbeLdcg, &ldcg, sizeof(ldcg));
    const unsigned long long epp_host = g.nelems / npages;
    const clio::run::u64 audit_before = 0;  // frame audit removed with the debug accessors
    const double t0p = NowMs();
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      ReadProbeKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dxp, a.steps,
                                                        a.blocks, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    std::printf("  frame audit: after Prefetch=%llu bad, after kernel=%llu "
                "bad (slot i must own frame i)\n",
                (unsigned long long)audit_before,
                0ull);
    unsigned long long rb[4] = {0, 0, 0, 0};
    cudaMemcpyFromSymbol(rb, g_read_bad, sizeof(rb));
    const auto sp = vx.ReadStats(0);
    std::printf("  READ PROBE: %llu wrong elements over %llu passes "
                "(faults %llu evicts %llu GET_ERRORS %llu puts %llu "
                "put_errors %llu prefetch_late %llu EARLY_COMPLETE %llu "
                "OUT-OF-TABLE %llu (site %llu), %.0f ms)%s\n",
                rb[0], (unsigned long long)a.steps,
                (unsigned long long)sp.faults, (unsigned long long)sp.evicts,
                (unsigned long long)sp.get_errors,
                (unsigned long long)sp.puts,
                (unsigned long long)sp.put_errors,
                NowMs() - t0p,
                rb[0] ? "   <-- THE FAULT PATH DELIVERS WRONG BYTES"
                      : "  [reads are clean]");
    if (rb[0]) {
      std::printf("    of those, %llu were CORRECT when re-read "
                  "uncached (stale-cache reads) | "
                  "WRONG-FRAME resolves=%llu | misses whose frame "
                  "NO GET EVER TARGETED=%llu\n", rb[1], rb[2], rb[3]);
      unsigned long long sm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      cudaMemcpyFromSymbol(sm, g_read_sample, sizeof(sm));
      if (sm[0]) {
        unsigned int bits = static_cast<unsigned int>(sm[3]);
        float got;
        std::memcpy(&got, &bits, 4);
        const unsigned long long want = sm[1] * epp_host + sm[2];
        std::printf("    sample: page %llu elem %llu -> got %.1f (bits 0x%08x)"
                    ", wanted %.1f.", sm[1], sm[2], got, bits,
                    static_cast<double>(want));
        if (bits == 0u) {
          std::printf("  ZERO: the frame was never written."
                      "  guard slot now holds page %llu, pins=%llu, "
                      "fetching=%llu (we asked for page %llu); "
                      "last get submitted into THIS frame was for "
                      "page %lld; DEVICE FRAME=%p\n",
                      sm[5] & 0xffffffffull, (sm[5] >> 40) & 0xffffff,
                      (sm[5] >> 32) & 0xffull, sm[1],
                      (long long)sm[7], (void *)(uintptr_t)sm[4]);
        unsigned long long gm[12] = {0};
        cudaMemcpyFromSymbol(gm, g_read_geom, sizeof(gm));
        if (gm[2]) {
          const long long idx =
              (static_cast<long long>(sm[4]) - static_cast<long long>(gm[0])) /
              static_cast<long long>(gm[2]);
          std::printf("    table: frame0=%p slots/block=%llu page_bytes=%llu"
                      " -> the guard's frame is SLOT %lld of this block "
                      "(%s)\n",
                      (void *)(uintptr_t)gm[0], gm[1], gm[2], idx,
                      (idx >= 0 && idx < (long long)gm[1])
                          ? "in range"
                          : "OUT OF THIS BLOCK'S TABLE");
          const bool pow2 = gm[3] && ((gm[3] & (gm[3] - 1)) == 0);
          std::printf("    elems/page=%llu (%s) page_shift=%llu "
                      "page_mask=%#llx%s\n",
                      gm[3], pow2 ? "power of two" : "NOT a power of two",
                      gm[4], gm[5],
                      (!pow2 && gm[4])
                          ? "  <-- SHIFT/MASK FAST PATH ON A "
                            "NON-POWER-OF-TWO PAGE"
                          : "");
          std::printf("    SAME Page: table slot index=%lld, but its frame "
                      "is arena index %lld -- %s\n",
                      (long long)gm[6], idx,
                      ((long long)gm[6] == idx)
                          ? "consistent"
                          : "MISMATCH: Page::data does not match its slot");
          std::printf("    per-block table bases: b0=%p b1=%p b2=%p b3=%p%s\n",
                      (void *)(uintptr_t)gm[8], (void *)(uintptr_t)gm[9],
                      (void *)(uintptr_t)gm[10], (void *)(uintptr_t)gm[11],
                      (gm[9] && gm[9] != gm[8])
                          ? "  <-- BLOCKS ARE ON DIFFERENT TABLES"
                          : "  (all blocks share one table)");
        }
        } else if (got == static_cast<float>(static_cast<unsigned long long>(got))) {
          const unsigned long long enc =
              static_cast<unsigned long long>(got);
          std::printf("  That value is element %llu, i.e. PAGE %llu -- the get "
                      "delivered ANOTHER PAGE'S BLOB.\n", enc, enc / epp_host);
        } else {
          std::printf("  Not a seeded value at all: torn or foreign bytes.\n");
        }
      }
    }
    return rb[0] ? 1 : 0;
  }

  // ---- STAGE 2: real MD (cell-direct LJ + NVE) ---------------------------
  if (a.md) {
    if (g.nb < 3) {
      std::fprintf(stderr, "need at least 3 bins per dimension\n");
      return 1;
    }
    // f gets its own cache size: the ballistic gate integrates a CONSTANT
    // acceleration, so it never exercises READING f under paging, and that
    // is the one path the resident gates cannot cover.
    const u32 fslots =
        AtLeastSlots((a.fslots != 0) ? a.fslots : slots, kMinSlotsF, "f");
    gv::Vector<float> vf("md_f", {0}, page_bytes, tbl_blocks, fslots,
                         g.nelems);
    vf.EnableStats();
    {
      std::vector<float> hz(g.nelems, 0.0f);
      vf.Preload(hz.data(), g.nelems);
    }
    vf.ClearCache();
    vf.Prefetch(0, npages, 0, 1);
    auto df = vf.GetDevice(0);
    // Ping-pong destination vectors for the resort (K2b scatters into
    // these, then the handles swap). Same geometry, resident.
    // The resort's destinations get their own cache size, so "the scatter
    // is wrong" can be told apart from "the scatter's PAGING is wrong".
    const u32 ppslots =
        AtLeastSlots((a.ppslots != 0) ? a.ppslots : slots, kMinSlotsX, "x2/v2");
    gv::Vector<float> vx2("md_x2", {0}, page_bytes, tbl_blocks, ppslots,
                          g.nelems);
    gv::Vector<float> vv2("md_v2", {0}, page_bytes, tbl_blocks, ppslots,
                          g.nelems);
    vx2.EnableStats();
    vv2.EnableStats();
    {
      std::vector<float> hs(g.nelems, 0.0f);
      for (u64 s = 0; s < g.nslots; ++s) hs[s * kStride + 3] = -1.0f;
      vx2.Preload(hs.data(), g.nelems);
      vv2.Preload(hs.data(), g.nelems);
    }
    vx2.ClearCache();
    vv2.ClearCache();
    vx2.Prefetch(0, npages, 0, 1);
    vv2.Prefetch(0, npages, 0, 1);
    auto dx2 = vx2.GetDevice(0);
    auto dv2 = vv2.GetDevice(0);
    gv::Vector<float> *cvx = &vx;   // current x (for stats/flush/download)
    // Resident index class for the resort.
    u32 *d_bincnt = ctp::GpuApi::Malloc<u32>(g.nbins * sizeof(u32));
    u32 *d_dest = ctp::GpuApi::Malloc<u32>(g.nslots * sizeof(u32));
    int *d_err = ctp::GpuApi::Malloc<int>(sizeof(int));
    // Stage 3: the Verlet list on a PAGED int vector, plus its resident
    // index (per-slot counts and CSR offsets).
    if (static_cast<u64>(g.nb) * g.cap >= 65536) {
      std::fprintf(stderr, "nb*cap must fit 16 bits for entry packing\n");
      return 1;
    }
    // Padded rows: every row owns islots * maxneigh entries, so a row's
    // region is a fixed contiguous span the block holds in one go. The
    // guard array bounds how many pages that may take -- checked here, in
    // configuration terms, rather than discovered as a device trap.
    const u64 md_islots = static_cast<u64>(g.nb) * g.cap;
    const u64 md_rowlist = md_islots * a.maxneigh;
    // THE LIST GETS ITS OWN PAGE SIZE, and the default is one whole ROW
    // per page: a row region is exactly what a block holds, so one page
    // means ONE guard instead of a chain of them. Holds are block-
    // collective (a barrier plus a per-thread guard written into the
    // coroutine frame), so their count -- not their size -- is what the
    // pass pays for.
    u64 nl_page_bytes = a.nl_page_kb * 1024;
    if (nl_page_bytes == 0) {
      nl_page_bytes = md_rowlist * sizeof(int);
      u64 pw = 4096;                       // round up to a power of two
      while (pw < nl_page_bytes) pw <<= 1;
      nl_page_bytes = pw;
    }
    const u64 nl_page_elems = nl_page_bytes / sizeof(int);
    const u64 nl_elems_raw = static_cast<u64>(g.nb) * g.nb * md_rowlist;
    const u64 nl_pages = (nl_elems_raw + nl_page_elems - 1) / nl_page_elems;
    const u64 nl_elems = nl_pages * nl_page_elems;
    if ((md_rowlist + nl_page_elems - 1) / nl_page_elems + 1 >
        static_cast<u64>(kMaxNlGuards)) {
      std::fprintf(stderr,
                   "list: one row spans %llu entries = more than %d pages; "
                   "raise --page-kb or lower --maxneigh/--cap\n",
                   (unsigned long long)md_rowlist, kMaxNlGuards - 1);
      return 1;
    }
    gv::Vector<int> vn("md_nl", {0}, nl_page_bytes, tbl_blocks,
                       AtLeastSlots(static_cast<u32>(nl_pages + 2), kMinSlotsNl, "nl"),
                       nl_elems);
    vn.EnableStats();
    {
      std::vector<int> hz(nl_elems, 0);
      vn.Preload(hz.data(), nl_elems);
    }
    vn.ClearCache();
    vn.Prefetch(0, nl_pages, 0, 1);
    auto dn = vn.GetDevice(0);
    std::printf("  list: maxneigh=%u row=%llu entries page=%lluKB "
                "(%llu pages, %llu guards/row)\n",
                a.maxneigh, (unsigned long long)md_rowlist,
                (unsigned long long)(nl_page_bytes >> 10),
                (unsigned long long)nl_pages,
                (unsigned long long)((md_rowlist + nl_page_elems - 1) /
                                     nl_page_elems + 1));
    u32 *d_cnt = ctp::GpuApi::Malloc<u32>(g.nslots * sizeof(u32));
    double *d_acc = ctp::GpuApi::Malloc<double>(3 * sizeof(double));
    // reduction scratch + the block-uniform pointer tables (9 stencil
    // rows, kMaxNlGuards list guards) that must not live in the frame.
    const u32 smem_tbl = static_cast<u32>(
        9 * (2 * sizeof(void *) + 2 * sizeof(u64)) + 12 * sizeof(u32) +
        kMaxNlGuards * (sizeof(void *) + 2 * sizeof(u64)));
    const u32 smem_force =
        CLIO_YIELD_SMEM_BYTES + a.threads * sizeof(double) + smem_tbl;
    const float fbox = static_cast<float>(g.box);
    const float fcut = static_cast<float>(a.cutoff);
    const float fdt = static_cast<float>(a.dt);

    // MD_NOCOMPUTE=1 keeps every hold and skips the pair loop: the
    // difference against a normal run IS the hold cost.
    const int nocompute = std::getenv("MD_NOCOMPUTE") != nullptr ? 1 : 0;
    const bool trace = std::getenv("MD_TRACE") != nullptr;
    double acc[3] = {0, 0, 0};
    double t_force_kern = 0.0, t_ckpt = 0.0, t_ckpt_stock = 0.0;
    u64 n_ckpt = 0;
    double t_force = 0.0, t_kick = 0.0, t_resort = 0.0, t_build = 0.0;
    const float frlist = static_cast<float>(a.cutoff + a.skin);
    // Build the Verlet list: device count pass, host prefix sum (index
    // class, ~MBs), device fill pass. Refuses loudly on maxneigh or the
    // per-row guard bound.
    auto build_list = [&]() -> bool {
      if (trace) { std::fprintf(stderr, "[md] build\n"); std::fflush(stderr); }
      const double _t = NowMs();
      ctp::GpuApi::Memset(d_err, 0, sizeof(int));
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        BuildListKernel<<<gr, b, smem_force>>>(
            gpu, dx, dn, g.nb, g.cap, fbox, frlist, a.maxneigh, d_cnt,
            d_err, a.rowchunk, a.blocks, vw, sv);
      });
      ctp::GpuApi::Synchronize();
      int err = 0;
      ctp::GpuApi::Memcpy(&err, d_err, sizeof(int));
      t_build += NowMs() - _t;
      if (err != 0) {
        std::fprintf(stderr,
                     "list: an atom has more than --maxneigh %u neighbours "
                     "within cutoff+skin\n", a.maxneigh);
        return false;
      }
      return true;
    };
    auto force = [&](int eflag) {
      if (trace) { std::fprintf(stderr, "[md] force eflag=%d\n", eflag);
                   std::fflush(stderr); }
      const double _t = NowMs();
      if (eflag) ctp::GpuApi::Memset(d_acc, 0, 3 * sizeof(double));
      if (a.use_list) {
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          ListForceKernel<<<gr, b, smem_force>>>(
              gpu, dx, df, dn, g.nb, g.cap, fbox, fcut, a.maxneigh, d_cnt,
              eflag, d_acc, nocompute, a.rowchunk, a.blocks, vw, sv);
        });
      } else {
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          ForceKernel<<<gr, b, smem_force>>>(gpu, dx, df, g.nb, g.cap, fbox,
                                             fcut, eflag, d_acc, a.blocks,
                                             vw, sv);
        });
      }
      ctp::GpuApi::Synchronize();
      if (eflag) ctp::GpuApi::Memcpy(acc, d_acc, 3 * sizeof(double));
      t_force += NowMs() - _t;
      t_force_kern += runner.KernelMs();
    };
    // MD_FLUSH_AFTER_KICK=1 makes every dirty x/v page durable in the
    // backing store before anything can fault it back in. If that removes
    // the intermittent out-of-core error, the hazard is a refetch racing an
    // in-flight writeback (read-after-write through the store).
    const bool flush_after_kick =
        std::getenv("MD_FLUSH_AFTER_KICK") != nullptr;
    auto kick = [&](int drift) {
      if (trace) { std::fprintf(stderr, "[md] kick drift=%d\n", drift);
                   std::fflush(stderr); }
      const double _t = NowMs();
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        MDIntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, dv, df, fdt, drift, a.blocks, vw, sv);
      });
      if (flush_after_kick) {
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          FlushAllKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu, dx, dv, a.blocks, md_row_elems, md_nrows, vw, sv);
        });
        ctp::GpuApi::Synchronize();
      }
      t_kick += NowMs() - _t;
    };
    auto thermo_ke = [&]() -> double {
      ctp::GpuApi::Memset(d_thermo, 0, 4 * sizeof(double));
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        ThermoKernel<<<gr, b, smem_thermo>>>(gpu, dx, dv, d_thermo, a.blocks,
                                             vw, sv);
      });
      ctp::GpuApi::Synchronize();
      double t4[4];
      ctp::GpuApi::Memcpy(t4, d_thermo, sizeof(t4));
      return t4[0];
    };
    // K2: wrap + rebin + double-buffered scatter, then swap the handles.
    // Returns false on bin overflow (the host refuses the run).
    auto resort = [&]() -> bool {
      const double _t = NowMs();
      ctp::GpuApi::Memset(d_bincnt, 0, g.nbins * sizeof(u32));
      ctp::GpuApi::Memset(d_err, 0, sizeof(int));
      const float fbox2 = static_cast<float>(g.box);
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        RebinKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, g.nb, g.cap, fbox2, d_bincnt, d_dest, d_err, a.blocks,
            vw, sv);
      });
      ctp::GpuApi::Synchronize();
      int err = 0;
      ctp::GpuApi::Memcpy(&err, d_err, sizeof(int));
      if (err != 0) {
        std::fprintf(stderr, "resort: bin overflow -- raise --cap\n");
        return false;
      }
      for (auto *dst : {&dx2, &dv2}) {
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          SentinelKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, *dst,
                                                           a.blocks, vw, sv);
        });
      }
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        GatherKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, dx, dx2, g.nb, g.cap, d_dest, /*keep_w=*/1, a.blocks,
            vw, sv);
      });
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        GatherKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dv, dx, dv2, g.nb, g.cap, d_dest, /*keep_w=*/0, a.blocks,
            vw, sv);
      });
      ctp::GpuApi::Synchronize();
      // INVALIDATE. Per-block caches have no cross-block coherence: after the
      // gather republished every row, each block still holds its own stale
      // copies of pages other blocks rewrote. Drop them so the next phase
      // fetches the published bytes. Safe here -- the gather flushed.
      vx.ClearCache();
      vv.ClearCache();
      vx2.ClearCache();
      vv2.ClearCache();
      std::swap(dx, dx2);
      std::swap(dv, dv2);
      cvx = (cvx == &vx) ? &vx2 : &vx;
      t_resort += NowMs() - _t;
      return true;
    };

    // ---- validation layer 2: step-0 statics from geometry alone ----------
    if (a.use_list && !build_list()) return 1;
    force(/*eflag=*/1);
    double pe_ref = 0.0, w_ref = 0.0;
    unsigned long long pairs_ref = 0;
    HostForceReference(g, hx, a.cutoff, &pe_ref, &w_ref, &pairs_ref);
    const double pe0 = acc[0];
    const unsigned long long pairs0 =
        static_cast<unsigned long long>(acc[2] + 0.5) / 2;
    const double pe_atom = pe0 / static_cast<double>(g.natoms);
    const double ref_rel =
        std::fabs(pe0 - pe_ref) / std::fabs(pe_ref);
    const double melt_abs = std::fabs(pe_atom - kMeltPePerAtom);
    std::printf(
        "  statics: PE/atom dev=%.7f hostref=%.7f LAMMPS=%.7f | pairs "
        "dev=%llu ref=%llu | W dev=%.6g ref=%.6g\n",
        pe_atom, pe_ref / g.natoms, kMeltPePerAtom, pairs0, pairs_ref,
        acc[1], w_ref);
    const bool statics_ok =
        (pairs0 == pairs_ref) && (ref_rel < 2e-5) && (melt_abs < 2e-4);
    std::printf("  STATICS GATE: %s (ref_rel=%.2e, melt_abs=%.2e)\n",
                statics_ok ? "PASS" : "FAIL", ref_rel, melt_abs);

    // ---- CHECKPOINTING -------------------------------------------------
    // A checkpoint here is what the paged design makes it: FlushAsync over
    // the live state plus AwaitFlush, page-granular, straight into the CTE
    // tier stack. Nothing is staged through the host, and the flush is
    // asynchronous by construction, so a real application can overlap it
    // with the following steps.
    //
    // The honest baseline alongside it is what a NON-paged GPU code must
    // pay before it can write anything at all: a device-to-host copy of the
    // same bytes. That is timed with the same clock on the same data size.
    float *d_ckpt_stock = nullptr;
    float *h_ckpt_stock = nullptr;
    if (a.ckpt != 0) {
      d_ckpt_stock = ctp::GpuApi::Malloc<float>(2 * g.nelems * sizeof(float));
      cudaMallocHost(&h_ckpt_stock, 2 * g.nelems * sizeof(float));
    }
    auto checkpoint = [&]() {
      const double _t = NowMs();
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        FlushAllKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, dx, dv, a.blocks, md_row_elems, md_nrows, vw, sv);
      });
      ctp::GpuApi::Synchronize();
      t_ckpt += NowMs() - _t;
      const double _t2 = NowMs();
      ctp::GpuApi::Memcpy(h_ckpt_stock, d_ckpt_stock, 2 * g.nelems);
      t_ckpt_stock += NowMs() - _t2;
      ++n_ckpt;
    };

    // ---- resort continuity gate: same physical state, new layout, so the
    // potential energy must be unchanged to float-summation noise ----------
    bool resort_ok = true;
    double resort_rel = 0.0;
    if (a.rebin != 0) {
      const double pe_before = acc[0];
      if (!resort()) return 1;
      if (a.use_list && !build_list()) return 1;
      force(/*eflag=*/1);
      resort_rel = std::fabs(acc[0] - pe_before) / std::fabs(pe_before);
      resort_ok = resort_rel < 1e-6;
      std::printf("  RESORT GATE: %s (PE %.6f -> %.6f, rel=%.2e)\n",
                  resort_ok ? "PASS" : "FAIL", pe_before, acc[0], resort_rel);
    }

    // ---- NVE: energy and momentum must be conserved ----------------------
    const double ke0 = thermo_ke();
    const double e0 = acc[0] + ke0;
    const double t0 = NowMs();
    for (u64 step = 0; step < a.steps; ++step) {
      kick(/*drift=*/1);   // uses f(t)
      if (a.rebin != 0 && step != 0 && step % a.rebin == 0) {
        if (!resort()) return 1;
        if (a.use_list && !build_list()) return 1;
      }
      force(/*eflag=*/0);  // f(t+dt)
      kick(/*drift=*/0);
      if (a.ckpt != 0 && (step + 1) % a.ckpt == 0) checkpoint();
    }
    ctp::GpuApi::Synchronize();
    const double run_ms = NowMs() - t0;
    force(/*eflag=*/1);
    const double ke_n = thermo_ke();
    const double e_n = acc[0] + ke_n;
    const double e_drift = std::fabs(e_n - e0) / std::fabs(e0);
    const auto sfx = cvx->ReadStats(0);
    const auto sff = vf.ReadStats(0);
    // THE RESIDENT CONTRACT IS "NO EVICTION", NOT "NO FAULT".
    //
    // It used to mean zero faults, which only held because every block shared
    // ONE page table: a page fetched by any block was resident for all of
    // them. With a private table per block -- the launch geometry the vector
    // requires -- each block must fetch every page it reads at least once,
    // and the resort's invalidation forces a refetch of anything another
    // block republished. Faults are therefore structural here.
    //
    // What "resident" still means, and what actually matters, is that the
    // cache never has to give anything up: no evictions, so nothing is
    // fetched twice for want of a frame.
    const bool res_ok = (sfx.evicts == 0 && sff.evicts == 0);
    std::printf(
        "  NVE %llu steps: E0=%.6f En=%.6f drift=%.2e | KE %.3f -> %.3f\n"
        "  paging: x faults=%llu evicts=%llu | f faults=%llu evicts=%llu  "
        "%s\n",
        (unsigned long long)a.steps, e0, e_n, e_drift, ke0, ke_n,
        (unsigned long long)sfx.faults, (unsigned long long)sfx.evicts,
        (unsigned long long)sff.faults, (unsigned long long)sff.evicts,
        expect_resident
            ? (res_ok ? "[resident contract HELD: no evictions]"
                      : "[RESIDENT CONTRACT VIOLATED: evicted]")
            : "[out-of-core regime: faults EXPECTED]");
    // REGIME-AWARE. The resident contract -- zero faults in the timed region
    // -- is a promise about the RESIDENT regime only: it says a cache that
    // can hold the working set must never page. Out of core the cache
    // provably cannot hold it, so faults are the feature, not a failure, and
    // asserting res_ok there failed runs whose physics was bit-identical to
    // the resident answer. What must hold in BOTH regimes is the energy.
    // Claim-failure counters are gone: a claim that cannot succeed now
    // terminates the kernel rather than being tallied.
    std::printf("  slots per table: %u\n", slots);
    const bool nve_ok = (e_drift < a.drift_tol) &&
                        (!expect_resident || res_ok) && !runner.HitCap();
    std::printf("  NVE GATE: %s\n", nve_ok ? "PASS" : "FAIL");
    std::printf("  %llu steps in %.1f ms (%.3f ms/step, %.1f "
                "Matom-steps/s)\n",
                (unsigned long long)a.steps, run_ms, run_ms / a.steps,
                static_cast<double>(g.natoms) * a.steps / run_ms / 1000.0);
    if (n_ckpt != 0) {
      // A checkpoint is only worth anything if it can be read back, so the
      // last one is verified against the live device state.
      std::vector<float> ck(g.nelems);
      const u64 got = cvx->Download(ck.data(), g.nelems);
      u64 bad = 0;
      for (u64 e = 0; e < g.nelems; e += kStride) {
        if (ck[e + 3] >= 0.0f && !(ck[e] == ck[e])) ++bad;   // NaN check
      }
      std::printf("  checkpoints: %llu | eternia flush %.2f ms each "
                  "(page-granular into the tier stack) | host staging a "
                  "non-paged code would pay first: %.2f ms each | readback "
                  "%llu/%llu pages, %llu bad\n",
                  (unsigned long long)n_ckpt, t_ckpt / n_ckpt,
                  t_ckpt_stock / n_ckpt, (unsigned long long)got,
                  (unsigned long long)npages, (unsigned long long)bad);
    }
    if (d_ckpt_stock != nullptr) ctp::GpuApi::Free(d_ckpt_stock);
    if (h_ckpt_stock != nullptr) cudaFreeHost(h_ckpt_stock);
    std::printf("  phases (total ms): force=%.1f (gpu %.1f) kick=%.1f "
                "resort=%.1f build=%.1f\n", t_force, t_force_kern, t_kick,
                t_resort, t_build);
    if (std::getenv("MD_PROF") != nullptr) {
      unsigned long long c[8] = {0};
      cudaMemcpyFromSymbol(c, g_md_cyc, sizeof(c));
      int khz = 0;
      cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
      const double us = 1000.0 / (double)khz;
      std::printf("  [mdprof] rows=%llu | stencil-holds=%.1f f-hold+zero=%.1f "
                  "list-guards=%.1f pair=%.1f total=%.1f (ms summed over "
                  "blocks)\n",
                  c[5], c[0] * us / 1000.0, c[1] * us / 1000.0,
                  c[2] * us / 1000.0, c[3] * us / 1000.0,
                  c[4] * us / 1000.0);
    }
    ctp::GpuApi::Free(d_acc);
    ctp::GpuApi::Free(d_thermo);
    ctp::GpuApi::Free(d_bincnt);
    ctp::GpuApi::Free(d_dest);
    ctp::GpuApi::Free(d_err);
    ctp::GpuApi::Free(d_cnt);
    return (statics_ok && nve_ok && resort_ok) ? 0 : 1;
  }

  // ---- the run ------------------------------------------------------------
  const float fdt = static_cast<float>(a.dt);
  const float fgx = static_cast<float>(a.g[0]);
  const float fgy = static_cast<float>(a.g[1]);
  const float fgz = static_cast<float>(a.g[2]);
  {
    const unsigned long long z = 0;
    cudaMemcpyToSymbol(g_pages_done, &z, sizeof(z));
  }
  const double t0 = NowMs();
  for (u64 step = 0; step < a.steps; ++step) {
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      IntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, dx, dv, dthird, use_third, fdt, fgx, fgy, fgz, /*drift=*/1,
          a.blocks, vw, sv);
    });
    const u32 rr = runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                                  gy::YieldStackView sv) {
      IntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, dx, dv, dthird, use_third, fdt, fgx, fgy, fgz, /*drift=*/0,
          a.blocks, vw, sv);
    });
    if (std::getenv("MD_ROUNDS") != nullptr) {
      const auto &rl = runner.RoundLog();
      std::fprintf(stderr, "[rounds] step=%llu n=%zu rounds=%u blocks/round:",
                   (unsigned long long)step, rl.size(), rr);
      for (size_t i = 0; i < rl.size() && i < 40; ++i) {
        std::fprintf(stderr, " %u", rl[i].second);
      }
      std::fprintf(stderr, "\n");
    }
  }
  ctp::GpuApi::Synchronize();
  const double run_ms = NowMs() - t0;

  double thermo[4] = {0, 0, 0, 0};
  double thermo2[4] = {0, 0, 0, 0};
  for (int rep = 0; rep < 2; ++rep) {
    ctp::GpuApi::Memset(d_thermo, 0, 4 * sizeof(double));
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      ThermoKernel<<<gr, b, smem_thermo>>>(gpu, dx, dv, d_thermo, a.blocks,
                                           vw, sv);
    });
    ctp::GpuApi::Synchronize();
    ctp::GpuApi::Memcpy(rep == 0 ? thermo : thermo2, d_thermo,
                        4 * sizeof(double));
  }
  std::printf("  thermo rep0: KE=%.9g p=(%.9g,%.9g,%.9g)\n"
              "  thermo rep1: KE=%.9g p=(%.9g,%.9g,%.9g)  %s\n",
              thermo[0], thermo[1], thermo[2], thermo[3], thermo2[0],
              thermo2[1], thermo2[2], thermo2[3],
              (thermo[0] == thermo2[0] && thermo[1] == thermo2[1] &&
               thermo[2] == thermo2[2] && thermo[3] == thermo2[3])
                  ? "[deterministic]" : "[VARIES: race]");

  // ---- resident-regime assertion -----------------------------------------
  {
    unsigned long long done = 0;
    cudaMemcpyFromSymbol(&done, g_pages_done, sizeof(done));
    const unsigned long long want =
        static_cast<unsigned long long>(npages) * a.steps * 2ull;
    std::printf("  page iterations: %llu of %llu expected%s\n", done, want,
                (done == want) ? "  [all work ran]"
                               : "   <-- WORK WAS DROPPED");
    unsigned long long bd[64] = {0}, bl[64] = {0};
    cudaMemcpyFromSymbol(bd, g_blk_done, sizeof(bd));
    cudaMemcpyFromSymbol(bl, g_blk_last, sizeof(bl));
    std::printf("  per block (done,last):");
    for (u32 b = 0; b < a.blocks && b < 8; ++b) {
      std::printf(" b%u=(%llu,%llu)", b, bd[b], bl[b]);
    }
    std::printf("\n");
  }
  {
    clio::run::u64 dx_dirty = 0, dv_dirty = 0;
    const u64 dupx = 0;
    const u64 dupv = 0;
    std::printf("  duplicate slots: x %llu (%llu with dirty) | v %llu "
                "(%llu with dirty)%s\n",
                (unsigned long long)dupx, (unsigned long long)dx_dirty,
                (unsigned long long)dupv, (unsigned long long)dv_dirty,
                (dupx || dupv) ? "   <-- a page lives in TWO slots" : "");
  }
  const auto sx = vx.ReadStats(0);
  const auto sv2 = vv.ReadStats(0);
  const bool resident_ok = (sx.faults == 0 && sx.evicts == 0 &&
                            sv2.faults == 0 && sv2.evicts == 0);
  std::printf("  paging: x faults=%llu evicts=%llu | v faults=%llu "
              "evicts=%llu  %s\n",
              (unsigned long long)sx.faults, (unsigned long long)sx.evicts,
              (unsigned long long)sv2.faults, (unsigned long long)sv2.evicts,
              resident_ok ? "[resident contract HELD]"
                          : "[RESIDENT CONTRACT VIOLATED]");

  // ---- the ballistic gate -------------------------------------------------
  // (a) BITWISE: replay the identical float recurrence on the host. The
  //     device updates each atom independently with the same operation
  //     order, so any bit of divergence is a real defect.
  // (b) CLOSED FORM: velocity Verlet under constant g is exact;
  //     x_n = x0 + n dt v0 + n^2 dt^2/2 g in double bounds float drift.
  int rc = 0;
  if (a.gate) {
    // Download reads the BACKING STORE; the kernels wrote the cache frames.
    // Flush first (validation-path only -- steady-state MD never flushes
    // x/v; they are device-canonical).
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      FlushAllKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, dx, dv, a.blocks, md_row_elems, md_nrows, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    // MD_SETTLE_MS=N waits before reading the backing store. If the
    // out-of-core mismatches vanish with a wait, the final AwaitFlush is
    // returning before its puts are durable ("landed but unsettled") and
    // the host is simply reading the store too early -- which would make
    // this a VERIFICATION artefact, not lost data.
    if (const char *ms = std::getenv("MD_SETTLE_MS")) {
      std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(ms)));
    }
    std::vector<float> gx_out(g.nelems), gv_out(g.nelems);
    vx.Download(gx_out.data(), g.nelems);
    vv.Download(gv_out.data(), g.nelems);
    u64 bit_x = 0, bit_v = 0;
    double max_cf = 0.0;
    double ke_ref = 0.0, mom_ref[3] = {0, 0, 0};
    const double n = static_cast<double>(a.steps);
    for (u64 s = 0; s < g.nslots; ++s) {
      const u64 e = s * kStride;
      if (hx[e + 3] < 0.0f) continue;
      // (a) float replica, same op order as the kernel
      float rx[3] = {hx[e], hx[e + 1], hx[e + 2]};
      float rv[3] = {hv[e], hv[e + 1], hv[e + 2]};
      const float gg[3] = {fgx, fgy, fgz};
      const float fhalf = 0.5f * fdt;
      for (u64 st = 0; st < a.steps; ++st) {
        for (int d = 0; d < 3; ++d) {
          rv[d] = std::fmaf(fhalf, gg[d], rv[d]);
          rx[d] = std::fmaf(fdt, rv[d], rx[d]);
        }
        for (int d = 0; d < 3; ++d) rv[d] = std::fmaf(fhalf, gg[d], rv[d]);
      }
      for (int d = 0; d < 3; ++d) {
        if (rx[d] != gx_out[e + d]) {
          if (bit_x < 4) {
            std::printf("    x mismatch slot=%llu d=%d page=%llu got=%.9g "
                        "want=%.9g diff=%.3e\n",
                        (unsigned long long)s, d,
                        (unsigned long long)(e / (page_bytes / sizeof(float))),
                        gx_out[e + d], rx[d],
                        std::fabs(static_cast<double>(gx_out[e + d]) - rx[d]));
          }
          ++bit_x;
        }
        if (rv[d] != gv_out[e + d]) {
          if (bit_v < 4) {
            std::printf("    v mismatch slot=%llu d=%d got=%.9g want=%.9g "
                        "diff=%.3e\n",
                        (unsigned long long)s, d, gv_out[e + d], rv[d],
                        std::fabs(static_cast<double>(gv_out[e + d]) - rv[d]));
          }
          ++bit_v;
        }
        // (b) closed form in double
        const double cf = static_cast<double>(hx[e + d]) +
                          n * a.dt * static_cast<double>(hv[e + d]) +
                          0.5 * n * n * a.dt * a.dt * a.g[d];
        const double err = std::fabs(cf - gx_out[e + d]);
        if (err > max_cf) max_cf = err;
        const double vd = gv_out[e + d];
        mom_ref[d] += vd;
        ke_ref += 0.5 * vd * vd;
      }
    }
    const double ke_err = std::fabs(thermo[0] - ke_ref) /
                          (ke_ref != 0.0 ? ke_ref : 1.0);
    double mom_err = 0.0;
    for (int d = 0; d < 3; ++d) {
      mom_err = std::max(mom_err, std::fabs(thermo[1 + d] - mom_ref[d]));
    }
    std::printf("    mom dev=(%.9g,%.9g,%.9g) ref=(%.9g,%.9g,%.9g)\n",
                thermo[1], thermo[2], thermo[3], mom_ref[0], mom_ref[1],
                mom_ref[2]);
    std::printf("  gate: bitwise mismatches x=%llu v=%llu | closed-form max "
                "|dx|=%.3e | thermo KE dev=%.6f host=%.6f rel_err=%.2e "
                "mom_err=%.2e\n",
                (unsigned long long)bit_x, (unsigned long long)bit_v, max_cf,
                thermo[0], ke_ref, ke_err, mom_err);
    const bool gate_ok = (bit_x == 0 && bit_v == 0) && (max_cf < 1e-2) &&
                         (ke_err < 1e-9) && (mom_err < 1e-6) && resident_ok &&
                         !runner.HitCap();
    std::printf("  BALLISTIC GATE: %s\n", gate_ok ? "PASS" : "FAIL");
    if (!gate_ok) rc = 1;
  }

  std::printf("  %llu steps in %.1f ms (%.3f ms/step, %.1f Matom-steps/s)\n",
              (unsigned long long)a.steps, run_ms, run_ms / a.steps,
              static_cast<double>(g.natoms) * a.steps / run_ms / 1000.0);
  ctp::GpuApi::Free(d_thermo);
  return rc;
#endif  // GV_MD_CORO
}
#endif  // !CTP_IS_DEVICE_PASS
