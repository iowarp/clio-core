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
__device__ unsigned long long g_md_cyc[8];

/** Guards a block may hold over one row of the paged list. Deliberately
 *  small: this array lives in the per-thread coroutine FRAME, so every
 *  slot costs frame footprint whether used or not, and the list page is
 *  sized to hold a whole row (one guard, two across a boundary). The host
 *  validates the bound in configuration terms before any run. */
static constexpr int kMaxNlGuards = 4;

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define GV_MD_CORO 1
#endif

namespace {

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
  u64 steps = 100;
  u64 rebin = 20;          // resort cadence (steps); 0 = never
  double temp = 0.0;       // scale initial velocities to this T (0 = leave)
  u32 maxneigh = 96;       // Verlet-list capacity per atom slot
  u32 rowchunk = 4;        // rows per block in the force pass (hold reuse)
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
    if (threadIdx.x == 0) {
      for (u32 t = 0; t < nspans; ++t) {
        s_sp0[t] = sp0[t];
        s_sp1[t] = sp1[t];
        s_srun[t] = srun[t];
      }
    }
    __syncthreads();
    for (u32 by = y0; by <= ylast; ++by) {
    const u64 row = static_cast<u64>(bz) * nb + by;
    if (threadIdx.x == 0) {
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
    }
    __syncthreads();
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
        hn[nguards] =
            co_await nl.HoldPage(nb0 + off, rowlist - off, /*write=*/true);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
    }
    if (threadIdx.x == 0) {
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
    if (threadIdx.x == 0) {
      for (u32 t = 0; t < nspans; ++t) {
        s_sp0[t] = sp0[t];
        s_sp1[t] = sp1[t];
        s_srun[t] = srun[t];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&g_md_cyc[0], (unsigned long long)(clock64() - _r0));
    for (u32 by = y0; by <= ylast; ++by) {
    const u64 row = static_cast<u64>(bz) * nb + by;
    // Which span holds each stencil row, and at what offset. Block-uniform
    // and only 9 entries, so thread 0 resolves it once per row.
    if (threadIdx.x == 0) {
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
    }
    __syncthreads();
    const long long _f0 = clock64();
    const u64 fbase = row * row_elems;
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
        hn[nguards] = co_await nl.HoldPage(nb0 + off, rowlist - off);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
    }
    if (threadIdx.x == 0) {
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
      atomicAdd(&g_md_cyc[4], (unsigned long long)(clock64() - _r0));
      atomicAdd(&g_md_cyc[5], 1ull);
    }
    }   // per-row loop
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
  const u64 epp = x.h_->elems_per_page_;
  const u64 npages = (x.h_->size_ + epp - 1) / epp;
  const float fnb = static_cast<float>(nb);
  for (u64 pg = block; pg < npages; pg += nblocks) {
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
__device__ gy::YCoroMain ScatterCoro(gv::DeviceVector<float> src,
                                     gv::DeviceVector<float> srcx,
                                     gv::DeviceVector<float> dst, u32 nb,
                                     u32 cap, const u32 *d_dest, int keep_w,
                                     u32 nblocks, u32 block) {
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 nrows = static_cast<u64>(nb) * nb;
  for (u64 row = block; row < nrows; row += nblocks) {
    const u32 by = static_cast<u32>(row % nb);
    const u32 bz = static_cast<u32>(row / nb);
    // Hold the NINE destination stencil rows for writing.
    gv::Held<float> hg[9][2];
    u64 rbase[9], rrun0[9];
    float *rp0[9], *rp1[9];
    for (int q = 0; q < 9; ++q) {
      const int dz = q / 3 - 1, dy = q % 3 - 1;
      const u32 wz = (bz + nb + dz) % nb;
      const u32 wy = (by + nb + dy) % nb;
      rbase[q] = ((static_cast<u64>(wz) * nb + wy) * nb) * cap * kStride;
      hg[q][0] = co_await dst.HoldPage(rbase[q], row_elems, true);
      rrun0[q] = hg[q][0].run();
      if (rrun0[q] < row_elems) {
        hg[q][1] =
            co_await dst.HoldPage(rbase[q] + rrun0[q], row_elems - rrun0[q],
                                  true);
      }
      rp0[q] = hg[q][0].ptr();
      rp1[q] = hg[q][1] ? hg[q][1].ptr() : nullptr;
    }
    // Source row of the vector being moved, and of x (for the sentinel).
    const u64 sbase = row * static_cast<u64>(nb) * cap * kStride;
    auto hs0 = co_await src.HoldPage(sbase, row_elems);
    gv::Held<float> hs1;
    const u64 srun0 = hs0.run();
    if (srun0 < row_elems) {
      hs1 = co_await src.HoldPage(sbase + srun0, row_elems - srun0);
    }
    auto hxs0 = co_await srcx.HoldPage(sbase, row_elems);
    gv::Held<float> hxs1;
    const u64 xrun0 = hxs0.run();
    if (xrun0 < row_elems) {
      hxs1 = co_await srcx.HoldPage(sbase + xrun0, row_elems - xrun0);
    }
    const float *const sp0 = hs0.ptr();
    const float *const sp1 = hs1 ? hs1.ptr() : nullptr;
    const float *const xp0 = hxs0.ptr();
    const float *const xp1 = hxs1 ? hxs1.ptr() : nullptr;
    const u64 islots = static_cast<u64>(nb) * cap;
    const u64 slotbase = row * islots;
    const u64 rowbin0 = row * nb;   // first linear bin of this source row
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 e = s * kStride;
      const float tw = (e < xrun0) ? xp0[e + 3] : xp1[e + 3 - xrun0];
      if (tw < 0.0f) continue;
      const u32 dest = d_dest[slotbase + s];
      if (dest == ~0u) continue;   // overflow victim; host already refused
      const u64 dbin = dest / cap;
      // Locate dbin inside the 9-row stencil.
      const u32 dbz = static_cast<u32>(dbin / (static_cast<u64>(nb) * nb));
      const u32 dby = static_cast<u32>((dbin / nb) % nb);
      const int ddz = (dbz == (bz + nb - 1) % nb) ? -1
                      : (dbz == bz)               ? 0
                                                  : 1;
      const int ddy = (dby == (by + nb - 1) % nb) ? -1
                      : (dby == by)               ? 0
                                                  : 1;
      const int q = (ddz + 1) * 3 + (ddy + 1);
      const u64 de = (dbin - (rbase[q] / (cap * kStride))) * cap * kStride +
                     (dest % cap) * kStride;
      float *const dp =
          (de < rrun0[q]) ? rp0[q] + de : rp1[q] + (de - rrun0[q]);
      const float *const spp = (e < srun0) ? sp0 + e : sp1 + (e - srun0);
      dp[0] = spp[0];
      dp[1] = spp[1];
      dp[2] = spp[2];
      dp[3] = keep_w ? spp[3] : 0.0f;
      (void)rowbin0;
    }
    __syncthreads();
  }
}

/** Pre-sentinel every slot of a ping-pong destination vector. */
__device__ gy::YCoroMain SentinelCoro(gv::DeviceVector<float> dst,
                                      u32 nblocks, u32 block) {
  const u64 epp = dst.h_->elems_per_page_;
  const u64 npages = (dst.h_->size_ + epp - 1) / epp;
  for (u64 pg = block; pg < npages; pg += nblocks) {
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
  const u64 epp = x.h_->elems_per_page_;
  const u64 npages = (x.h_->size_ + epp - 1) / epp;
  const float half = 0.5f * dt;
  for (u64 pg = block; pg < npages; pg += nblocks) {
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
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
                                      gv::DeviceVector<float> v, u32 block) {
  if (block == 0) {
    x.FlushAsync(0, x.h_->size_);
    co_await x.AwaitFlush();
    v.FlushAsync(0, v.h_->size_);
    co_await v.AwaitFlush();
  }
}

__device__ gy::YCoroMain IntegrateCoro(gv::DeviceVector<float> x,
                                       gv::DeviceVector<float> v,
                                       float dt, float gx, float gy_,
                                       float gz, int drift, u32 nblocks,
                                       u32 block) {
  const u64 epp = x.h_->elems_per_page_;
  const u64 npages = (x.h_->size_ + epp - 1) / epp;
  const float half = 0.5f * dt;
  for (u64 pg = block; pg < npages; pg += nblocks) {
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
    float *const px = hx.ptr();
    float *const pv = hv.ptr();
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
  const u64 epp = x.h_->elems_per_page_;
  const u64 npages = (x.h_->size_ + epp - 1) / epp;
  double ke = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (u64 pg = block; pg < npages; pg += nblocks) {
    auto hx = co_await x.HoldPage(pg * epp, epp);
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

__global__ void IntegrateKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<float> v, float dt,
                                float gx, float gy_, float gz, int drift,
                                u32 nblocks, gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  v.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(IntegrateCoro(x, v, dt, gx, gy_, gz, drift, nblocks,
                               yv.Block()));
}

__global__ void ThermoKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> x,
                             gv::DeviceVector<float> v, double *out,
                             u32 nblocks, gy::YieldableView<> yv,
                             gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  v.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ThermoCoro(x, v, out, nblocks, yv.Block()));
}

__global__ void ForceKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> x,
                            gv::DeviceVector<float> f, u32 nb, u32 cap,
                            float box, float cutoff, int eflag, double *acc,
                            u32 nblocks, gy::YieldableView<> yv,
                            gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  f.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ForceCoro(x, f, nb, cap, box, cutoff, eflag, acc, nblocks,
                           yv.Block()));
}

__global__ void BuildListKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<int> nl, u32 nb, u32 cap,
                                float box, float rlist, u32 maxneigh,
                                u32 *d_cnt, int *d_err, u32 rowchunk,
                                u32 nblocks,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  nl.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(BuildListCoro(x, nl, nb, cap, box, rlist, maxneigh, d_cnt,
                               d_err, rowchunk, nblocks, yv.Block()));
}

__global__ void ListForceKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<float> f,
                                gv::DeviceVector<int> nl, u32 nb, u32 cap,
                                float box, float cutoff, u32 maxneigh,
                                const u32 *d_cnt, int eflag, double *acc,
                                int nocompute, u32 rowchunk, u32 nblocks,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  f.block_override_ = 0;
  nl.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ListForceCoro(x, f, nl, nb, cap, box, cutoff, maxneigh,
                               d_cnt, eflag, acc, nocompute, rowchunk,
                               nblocks, yv.Block()));
}

__global__ void RebinKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> x, u32 nb, u32 cap,
                            float box, u32 *bincnt, u32 *d_dest, int *d_err,
                            u32 nblocks, gy::YieldableView<> yv,
                            gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RebinCoro(x, nb, cap, box, bincnt, d_dest, d_err, nblocks,
                           yv.Block()));
}

__global__ void ScatterKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<float> src,
                              gv::DeviceVector<float> srcx,
                              gv::DeviceVector<float> dst, u32 nb, u32 cap,
                              const u32 *d_dest, int keep_w, u32 nblocks,
                              gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  src.block_override_ = 0;
  srcx.block_override_ = 0;
  dst.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ScatterCoro(src, srcx, dst, nb, cap, d_dest, keep_w,
                             nblocks, yv.Block()));
}

__global__ void SentinelKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> dst, u32 nblocks,
                               gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  dst.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SentinelCoro(dst, nblocks, yv.Block()));
}

__global__ void MDIntegrateKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceVector<float> x,
                                  gv::DeviceVector<float> v,
                                  gv::DeviceVector<float> f, float dt,
                                  int drift, u32 nblocks,
                                  gy::YieldableView<> yv,
                                  gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  v.block_override_ = 0;
  f.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(MDIntegrateCoro(x, v, f, dt, drift, nblocks, yv.Block()));
}

__global__ void FlushAllKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> x,
                               gv::DeviceVector<float> v,
                               gy::YieldableView<> yv,
                               gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.block_override_ = 0;
  v.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(FlushAllCoro(x, v, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS
namespace {

class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.ResetTimers();
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000,
        [](u32, u64 tag) -> bool {
          unsigned int f = 0;
          ctp::GpuApi::Memcpy(&f, reinterpret_cast<const unsigned int *>(tag),
                              sizeof(f));
          return (f & 1u) != 0u;
        });
  }
  double KernelMs() const { return drv_.KernelMs(); }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

}  // namespace
#endif  // !CTP_IS_DEVICE_PASS
#endif  // GV_MD_CORO

#if !CTP_IS_DEVICE_PASS
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
    else if (want("--steps")) a.steps = static_cast<u64>(atol(argv[++i]));
    else if (want("--rebin")) a.rebin = static_cast<u64>(atol(argv[++i]));
    else if (want("--temp")) a.temp = atof(argv[++i]);
    else if (want("--drift-tol")) a.drift_tol = atof(argv[++i]);
    else if (want("--maxneigh")) a.maxneigh = static_cast<u32>(atoi(argv[++i]));
    else if (want("--rowchunk")) a.rowchunk = static_cast<u32>(atoi(argv[++i]));
    else if (want("--nl-page-kb")) a.nl_page_kb = static_cast<u64>(atol(argv[++i]));
    else if (std::strcmp(argv[i], "--no-list") == 0) a.use_list = 0;
    else if (want("--dt")) a.dt = atof(argv[++i]);
    else if (std::strcmp(argv[i], "--no-gate") == 0) a.gate = 0;
    else if (std::strcmp(argv[i], "--md") == 0) a.md = 1;
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
  const u32 slots =
      (a.slots != 0) ? a.slots : static_cast<u32>(npages + 2);

  std::printf(
      "eternia-MD stage 1 (integrator + ballistic gate)\n"
      "  atoms=%llu box=%.4f bins=%u^3 cap=%u slots/bin-pad=%.0f%%\n"
      "  page=%lluKB (%llu bins/page) pages=%llu blocks=%u threads=%u "
      "cache=%u slots (one shared table)\n"
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
  gv::Vector<float> vx("md_x", {0}, page_bytes, /*nblocks=*/1, slots,
                       g.nelems);
  gv::Vector<float> vv("md_v", {0}, page_bytes, /*nblocks=*/1, slots,
                       g.nelems);
  vx.EnableStats();
  vv.EnableStats();
  vx.Preload(hx.data(), g.nelems);
  vv.Preload(hv.data(), g.nelems);
  vx.ClearCache();
  vv.ClearCache();
  vx.Prefetch(0, npages, 0, 1);
  vv.Prefetch(0, npages, 0, 1);
  auto dx = vx.GetDevice(0);
  auto dv = vv.GetDevice(0);

  double *d_thermo = ctp::GpuApi::Malloc<double>(4 * sizeof(double));

  YieldRunner runner(a.blocks, a.threads);
  const u32 smem_thermo =
      CLIO_YIELD_SMEM_BYTES + a.threads * sizeof(double);

  // ---- STAGE 2: real MD (cell-direct LJ + NVE) ---------------------------
  if (a.md) {
    if (g.nb < 3) {
      std::fprintf(stderr, "need at least 3 bins per dimension\n");
      return 1;
    }
    gv::Vector<float> vf("md_f", {0}, page_bytes, /*nblocks=*/1, slots,
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
    gv::Vector<float> vx2("md_x2", {0}, page_bytes, 1, slots, g.nelems);
    gv::Vector<float> vv2("md_v2", {0}, page_bytes, 1, slots, g.nelems);
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
    gv::Vector<int> vn("md_nl", {0}, nl_page_bytes, 1,
                       static_cast<u32>(nl_pages + 2), nl_elems);
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
    double acc[3] = {0, 0, 0};
    double t_force_kern = 0.0;
    double t_force = 0.0, t_kick = 0.0, t_resort = 0.0, t_build = 0.0;
    const float frlist = static_cast<float>(a.cutoff + a.skin);
    // Build the Verlet list: device count pass, host prefix sum (index
    // class, ~MBs), device fill pass. Refuses loudly on maxneigh or the
    // per-row guard bound.
    auto build_list = [&]() -> bool {
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
    auto kick = [&](int drift) {
      const double _t = NowMs();
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        MDIntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, dv, df, fdt, drift, a.blocks, vw, sv);
      });
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
        ScatterKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, dx, dx2, g.nb, g.cap, d_dest, /*keep_w=*/1, a.blocks,
            vw, sv);
      });
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        ScatterKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dv, dx, dv2, g.nb, g.cap, d_dest, /*keep_w=*/0, a.blocks,
            vw, sv);
      });
      ctp::GpuApi::Synchronize();
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
    }
    ctp::GpuApi::Synchronize();
    const double run_ms = NowMs() - t0;
    force(/*eflag=*/1);
    const double ke_n = thermo_ke();
    const double e_n = acc[0] + ke_n;
    const double e_drift = std::fabs(e_n - e0) / std::fabs(e0);
    const auto sfx = cvx->ReadStats(0);
    const auto sff = vf.ReadStats(0);
    const bool res_ok = (sfx.faults == 0 && sfx.evicts == 0 &&
                         sff.faults == 0 && sff.evicts == 0);
    std::printf(
        "  NVE %llu steps: E0=%.6f En=%.6f drift=%.2e | KE %.3f -> %.3f\n"
        "  paging: x faults=%llu evicts=%llu | f faults=%llu evicts=%llu  "
        "%s\n",
        (unsigned long long)a.steps, e0, e_n, e_drift, ke0, ke_n,
        (unsigned long long)sfx.faults, (unsigned long long)sfx.evicts,
        (unsigned long long)sff.faults, (unsigned long long)sff.evicts,
        res_ok ? "[resident contract HELD]" : "[RESIDENT CONTRACT VIOLATED]");
    const bool nve_ok = (e_drift < a.drift_tol) && res_ok;
    std::printf("  NVE GATE: %s\n", nve_ok ? "PASS" : "FAIL");
    std::printf("  %llu steps in %.1f ms (%.3f ms/step, %.1f "
                "Matom-steps/s)\n",
                (unsigned long long)a.steps, run_ms, run_ms / a.steps,
                static_cast<double>(g.natoms) * a.steps / run_ms / 1000.0);
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
  const double t0 = NowMs();
  for (u64 step = 0; step < a.steps; ++step) {
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      IntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, dx, dv, fdt, fgx, fgy, fgz, /*drift=*/1, a.blocks, vw, sv);
    });
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      IntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, dx, dv, fdt, fgx, fgy, fgz, /*drift=*/0, a.blocks, vw, sv);
    });
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
      FlushAllKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dx, dv, vw, sv);
    });
    ctp::GpuApi::Synchronize();
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
                         (ke_err < 1e-9) && (mom_err < 1e-6) && resident_ok;
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
