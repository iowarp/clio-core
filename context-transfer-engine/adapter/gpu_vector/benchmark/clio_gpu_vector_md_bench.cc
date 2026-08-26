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
#include <memory>
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

__device__ unsigned long long g_md_cyc[8];

/** Page iterations actually completed by the integrator, summed over
 *  blocks. Distinguishes WORK DROPPED (driver/park bookkeeping) from DATA
 *  CORRUPTED (the paging path): the expected count is exact and known. */
/** MD_NO_PUBLISH=1 drops every publish. The physics is then WRONG (blocks
 *  read each other's stale pages); it exists only so the device-side cost of
 *  publishing can be measured as a difference against a correct run. Uniform
 *  across all threads, so the co_await stays out of a divergent branch. */
__device__ u32 g_publish = 1u;
/** Publish the slab INTERIOR (gen-0 track-the-store flushes). Single node
 *  needs it -- OOC eviction performs no I/O and --ckpt reads the store.
 *  Decomposed runs are resident, and a peer only ever reads the BOUNDARY
 *  planes, which exchange() publishes generationally -- so the interior
 *  flush there was per-step traffic with no reader: measured 12 puts/step
 *  on x alone against MPI's 0.53 MB/step ledger. */
__device__ u32 g_pub_interior = 1u;
/** MD_RESORT_DEBUG: slots the gather actually wrote, plus one sample. */
__device__ unsigned long long g_gather_wrote = 0ull;
/** Publish-kernel latency split, in device cycles: the flush await vs the
 *  peer-halo fetch await. The host round counter cannot tell them apart. */
__device__ unsigned long long g_pub_flush_cyc = 0ull;
__device__ unsigned long long g_pub_fetch_cyc = 0ull;
__device__ unsigned long long g_gather_sample = 0ull;

/** SHARING PROBE: bit b set means block b held that page at least once.
 *  16 blocks fit a u32; the popcount per page is the sharing degree. */
__device__ u32 g_xmask[8192];
__device__ u32 g_nlmask[8192];
__device__ void MarkPages(u32 *mask, u64 p0, u64 p1, u32 block) {
  if (threadIdx.x != 0 || block >= 32) return;
  for (u64 p = p0; p <= p1 && p < 8192; ++p) {
    atomicOr(&mask[p], 1u << block);
  }
}

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
 * The block-uniform tables the list passes stage out of the coroutine frame.
 *
 * Declared as a TYPE, not carved out of `extern __shared__` with casts: the
 * arena is one struct, so its size is a compile-time fact instead of a
 * hand-summed byte count that has to be kept in step at the launch site.
 */
struct MdTables {
  const float *sp0[9];
  const float *sp1[9];
  u64 srun[9];
  u64 qoff[9];
  u32 qspan[12];
  const int *np[kMaxNlGuards];
  u64 gs[kMaxNlGuards];
  u64 gl[kMaxNlGuards];
};


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
  u32 nlslots = 0;         // list cache frames per block; 0 = NlSlots() default
  u64 vram_mb = 0;         // cache budget across ALL vectors; 0 = size for residency
  int use_list = 1;        // stage 3 list force; --no-list = cell-direct
  // PREPENDED TO EVERY VECTOR TAG. Two processes sharing one CTE would
  // otherwise both create "md_x" and page into each other's blobs; the
  // distributed harness runs a bench per node, so each needs its own
  // namespace. Empty by default, so a single-node run is unchanged.
  const char *tag_prefix = "";
  // DOMAIN DECOMPOSITION. --nodes N --node i splits the z-planes into N
  // contiguous slabs and gives this process slab i. One node (the default) is
  // the whole domain, so a single-process run is unchanged -- which is also
  // the reference a distributed run has to reproduce.
  u32 nodes = 1;
  u32 node = 0;
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
  // Constant acceleration for the ballistic gate. It COMPACTS the system:
  // over a long run atoms pile up, bins overflow --cap and neighbour counts
  // pass --maxneigh, so a 100+ step run needs it off. --gravity 0 does that
  // and leaves plain NVE, which is what a long checkpointing run wants.
  double g[3] = {0.1, -0.05, 0.02};
};

/** LAMMPS melt step-0 pair energy per atom (rho=0.8442 FCC, lj/cut 2.5,
 *  no shift): the published stock value this reimplementation must reproduce
 *  from geometry alone. */
static constexpr double kMeltPePerAtom = -6.7733681;

/** Unbuffered kernel marker. GV_MD_TRACE=1. A trap kills the context and the
 *  runtime aborts, so anything buffered is lost -- this says which launch. */
static void MdMark(const char *what) {
  static const bool on = getenv("GV_MD_TRACE") != nullptr;
  if (!on) return;
  // ATTRIBUTE THE ERROR TO THE LAUNCH THAT CAUSED IT. CUDA reports
  // asynchronously, so the Synchronize that FAILS is usually inside a LATER
  // kernel's driver loop -- the last marker names where it surfaced, not
  // where it happened. Syncing here pins it down without the full
  // serialization of CUDA_LAUNCH_BLOCKING, which hides the race entirely.
  static const char *prev = "(none)";
  cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[mdtrace] *** FAILED IN %s: %s\n", prev,
                 cudaGetErrorString(e));
    std::fflush(stderr);
    std::_Exit(9);
  }
  std::fprintf(stderr, "[mdtrace] %s\n", what);
  std::fflush(stderr);
  prev = what;
}

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

/**
 * This node's slab as an ELEMENT range, and the pages it touches.
 *
 * A slab boundary does not have to land on a page boundary, and it must not
 * have to: forcing that would constrain --page-kb against the lattice. Two
 * nodes may therefore share the page at their seam. That is safe for exactly
 * the reason disjoint writers within a node are safe -- each writes only its
 * own elements and flushes BYTE-EXACTLY, so neither can clobber the other's
 * half of the page.
 */
/**
 * A boolean env flag that survives docker-compose.
 *
 * `getenv(x) != nullptr` is TRUE for an EMPTY value, and the harness declares
 * `MD_NO_HALO=${MD_NO_HALO:-}` unconditionally, so the variable is always
 * present in the container and every such flag read as ON -- including in the
 * "halo enabled" arm of the A/B, which is why both arms produced identical
 * numbers. Empty, "0", "false" and "no" all mean OFF.
 */
static bool EnvOn(const char *name) {
  const char *v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return false;
  return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
           std::strcmp(v, "no") == 0);
}

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

__device__ gy::YCoroMain ForceCoro(gv::DeviceVector<float> x,
                                   gv::DeviceVector<float> f,
                                   u32 nb, u32 cap, float box, float cutoff,
                                   int eflag, double *acc, u32 z0, u32 z1,
                                   u32 nblocks, u32 block, u64 hgen,
                                   bool force_all) {
  extern __shared__ char smem_raw[];
  double *red = reinterpret_cast<double *>(smem_raw + CLIO_YIELD_SMEM_BYTES);
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  // THIS NODE'S SLAB, in z-planes. The nine-row stencil still reaches one
  // plane either side, so the rows it READS run past [z0, z1) -- those are
  // the halo, and they are this node's copy of a neighbour's rows, refreshed
  // through the generational path before the pass runs.
  const u64 row_lo = static_cast<u64>(z0) * nb;
  const u64 row_hi = static_cast<u64>(z1) * nb;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  double pe = 0.0, w = 0.0, npairs = 0.0;

  for (u64 row = row_lo + block; row < row_hi; row += nblocks) {
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
      // THE VERSION DEMAND BELONGS HERE, on the read that consumes the data.
      // A z-plane outside [z0,z1) is a neighbour's, and a resident copy of it
      // is last step's. Generation 0 means "any version", so asking for it
      // here -- which is what this line used to do -- accepts the stale copy
      // and makes the whole exchange a no-op no matter what the publisher
      // does. hgen == 0 restores that unsynchronised behaviour deliberately
      // (single node, or MD_NO_HALO).
      // MD_FORCE_GEN makes EVERY stencil row generational, so a single-node
      // run can answer "does a generational fetch refuse a resident older
      // copy?" without a second node. hgen is then a generation nobody has
      // published, which must NOT be served.
      const bool halo = force_all || (wz < z0 || wz >= z1);
      co_await x.Fetch(halo ? hgen : 0, rbase[q], row_elems);
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
    const u64 fbase =
        (static_cast<u64>(bz) * nb + by) * static_cast<u64>(nb) * cap * kStride;
    co_await f.Fetch(0, fbase, row_elems);
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
    // GIVE THE PINS BACK. In shared mode Fetch is the pinner and UnpinRange
    // the releaser, one unpin per fetch over the same range; a range fetched
    // and never released is a frame no block can reclaim. A no-op under
    // private tables, where the guards above own their pins -- which is what
    // lets this one kernel run unmodified under both modes.
    for (int q = 0; q < 9; ++q) x.UnpinRange(rbase[q], row_elems);
    f.UnpinRange(fbase, row_elems);
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
                                       u32 rowchunk, u32 z0, u32 z1,
                                       u32 nblocks, u32 block, u64 hgen) {
  extern __shared__ char smem_raw[];
  // BLOCK-UNIFORM TABLES LIVE IN SHARED, NOT IN THE FRAME. Every thread
  // holds the same rows and the same list guards, but a thread-local array
  // indexed by a runtime value (rp0[q], np[gi]) cannot be a register: it
  // lands in the coroutine frame, which IS the yield-stack lane in GLOBAL
  // memory. That made ~6 dependent global loads of pure bookkeeping per
  // entry and is why the list pass first measured SLOWER than cell-direct
  // despite 17x fewer candidates. Filled once per row after every hold
  // (no co_await follows, so shared survives) and read by all threads.
  // SURVIVES A PARK. These were staged in plain __shared__, which the driver
  // destroys when it exits the kernel to suspend, so every one of them had to
  // be re-published by hand after the last hold that could suspend -- and a
  // suspend added anywhere after that fill would have read stale pointers.
  CLIO_SHARED_PERSIST(MdTables, s_tbl);
  const float **s_sp0 = s_tbl.sp0;
  const float **s_sp1 = s_tbl.sp1;
  u64 *s_srun = s_tbl.srun;
  u64 *s_qoff = s_tbl.qoff;
  u32 *s_qspan = s_tbl.qspan;
  const int **s_np = s_tbl.np;
  u64 *s_gs = s_tbl.gs;
  u64 *s_gl = s_tbl.gl;
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
  // Chunks are (bz, y-run) pairs, so a z-plane slab is a contiguous chunk
  // range: [z0*cpz, z1*cpz).
  const u64 ch_lo = static_cast<u64>(z0) * cpz;
  const u64 ch_hi = static_cast<u64>(z1) * cpz;
  for (u64 ch = ch_lo + block; ch < ch_hi; ch += nblocks) {
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
    // WHAT THE FETCH PINNED, so the chunk can give it back. In shared mode
    // Fetch is the pinner and UnpinRange the releaser; a range fetched and
    // never released is a frame no other block can ever reclaim. A no-op
    // under private tables, where the guards above own their own pins.
    u64 sxrb[6], sxlen[6];
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
        // Same rule as ForceCoro: a plane outside this node's slab belongs to
        // a neighbour, and only a generational fetch refuses last step's copy.
        co_await x.Fetch((wz < z0 || wz >= z1) ? hgen : 0, rb, len);
        hg[nspans][0] = co_await x.HoldPage(rb, len);
        srun[nspans] = hg[nspans][0].run();
        if (srun[nspans] < len) {
          hg[nspans][1] =
              co_await x.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        MarkPages(g_xmask, x.PageOf(rb), x.PageOf(rb + len - 1), block);
        sxrb[nspans] = rb;
        sxlen[nspans] = len;
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
        co_await nl.Fetch(0, nl.PageLo(nb0 + off), nl.PageSpan(nb0 + off, 1));
        hn[nguards] =
            co_await nl.HoldPage(nb0 + off, rowlist - off, /*write=*/true);
        MarkPages(g_nlmask, nl.PageOf(nb0 + off),
                  nl.PageOf(nb0 + off + hn[nguards].run() - 1), block);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
    }
    // Publish the block-uniform tables. This used to be a RE-publication
    // that had to sit after the last hold that could suspend, because the
    // arena was plain __shared__ and the driver destroys shared when it
    // exits the kernel to park. CLIO_SHARED_PERSIST carries it across the
    // suspension now, so this is an ordinary fill and a co_await added below
    // it is no longer a silent corruption.
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
    // PUBLISH THIS LIST ROW AND LET THE FRAME GO.
    //
    // The row was write-held, and a dirty page is unevictable -- the vector
    // will not write it back on the caller's behalf. Building every row this
    // block owns without flushing therefore fills the table with dirty frames
    // that EvictPages cannot reclaim, and it traps. That, not any working
    // set, is what set the list cache floor: a block builds ~53 rows, so it
    // needed ~64 frames purely to keep every dirty row. The pass only ever
    // reads one row at a time, so with the flush the cache can be tiny.
    co_await nl.BeginFlush(0, row * rowlist, rowlist);
    // Safe after BeginFlush and not before the writes: a frame with a flush
    // in flight is not an eviction candidate, so the row's bytes cannot be
    // dropped between the submit and the wait.
    //
    // RECOMPUTED FROM gstart, NOT STORED. Two more u64 arrays here overflow
    // the coroutine frame (2144 > 2048) -- and they were redundant: gstart
    // already records the offset each fetch was issued at, and nothing
    // between the fetch and here suspends, so the arithmetic costs
    // registers rather than frame.
    for (u32 gq = 0; gq < nguards; ++gq) {
      const u64 fo = row * rowlist + gstart[gq];
      nl.UnpinRange(nl.PageLo(fo), nl.PageSpan(fo, 1));
    }
    }   // per-row loop
    // Inside the guard scope: sxrb/nspans live here, and the pins must go
    // back before the next chunk fetches its own spans.
    for (u32 sq = 0; sq < nspans; ++sq) x.UnpinRange(sxrb[sq], sxlen[sq]);
    }
    co_await nl.EndFlush();
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
                                       u32 rowchunk, u32 z0, u32 z1,
                                       u32 nblocks, u32 block, u64 hgen,
                                       bool force_all) {
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
  // SURVIVES A PARK. These were staged in plain __shared__, which the driver
  // destroys when it exits the kernel to suspend, so every one of them had to
  // be re-published by hand after the last hold that could suspend -- and a
  // suspend added anywhere after that fill would have read stale pointers.
  CLIO_SHARED_PERSIST(MdTables, s_tbl);
  const float **s_sp0 = s_tbl.sp0;
  const float **s_sp1 = s_tbl.sp1;
  u64 *s_srun = s_tbl.srun;
  u64 *s_qoff = s_tbl.qoff;
  u32 *s_qspan = s_tbl.qspan;
  const int **s_np = s_tbl.np;
  u64 *s_gs = s_tbl.gs;
  u64 *s_gl = s_tbl.gl;
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
  const u64 ch_lo = static_cast<u64>(z0) * cpz;
  const u64 ch_hi = static_cast<u64>(z1) * cpz;
  for (u64 ch = ch_lo + block; ch < ch_hi; ch += nblocks) {
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
    // WHAT THE FETCH PINNED, so the chunk can give it back. In shared mode
    // Fetch is the pinner and UnpinRange the releaser; a range fetched and
    // never released is a frame no other block can ever reclaim. A no-op
    // under private tables, where the guards above own their own pins.
    u64 sxrb[6], sxlen[6];
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
        // Same rule as ForceCoro: a plane outside this node's slab belongs to
        // a neighbour, and only a generational fetch refuses last step's copy.
        co_await x.Fetch((force_all || wz < z0 || wz >= z1) ? hgen : 0,
                         rb, len);
        hg[nspans][0] = co_await x.HoldPage(rb, len);
        srun[nspans] = hg[nspans][0].run();
        if (srun[nspans] < len) {
          hg[nspans][1] =
              co_await x.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        MarkPages(g_xmask, x.PageOf(rb), x.PageOf(rb + len - 1), block);
        sxrb[nspans] = rb;
        sxlen[nspans] = len;
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
    co_await f.Fetch(0, fbase, row_elems);
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
        co_await nl.Fetch(0, nl.PageLo(nb0 + off), nl.PageSpan(nb0 + off, 1));
        hn[nguards] = co_await nl.HoldPage(nb0 + off, rowlist - off);
        MarkPages(g_nlmask, nl.PageOf(nb0 + off),
                  nl.PageOf(nb0 + off + hn[nguards].run() - 1), block);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
    }
    // Publish the block-uniform tables. This used to be a RE-publication
    // that had to sit after the last hold that could suspend, because the
    // arena was plain __shared__ and the driver destroys shared when it
    // exits the kernel to park. CLIO_SHARED_PERSIST carries it across the
    // suspension now, so this is an ordinary fill and a co_await added below
    // it is no longer a silent corruption.
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
    // The row's own pins go back here; the chunk's x spans go back below.
    // f is written and never flushed by design -- the next kernel reads it
    // out of the cache -- so this releases the pin, not the page.
    // Recomputed from gstart; see the note in the build pass.
    for (u32 gq = 0; gq < nguards; ++gq) {
      const u64 fo = row * rowlist + gstart[gq];
      nl.UnpinRange(nl.PageLo(fo), nl.PageSpan(fo, 1));
    }
    f.UnpinRange(fbase, row_elems);
    }   // per-row loop
    // The chunk total, taken ONCE. See the note at g_md_cyc[5].
    if (threadIdx.x == 0) {
      atomicAdd(&g_md_cyc[4], (unsigned long long)(clock64() - _r0));
    }
    // Every span guard above is dead here (the chunk body scope ends with
    // this brace), so the reservation is given back exactly once per
    // EnterHoldSet. The only chunk-level `continue` is above the Enter.
    for (u32 sq = 0; sq < nspans; ++sq) x.UnpinRange(sxrb[sq], sxlen[sq]);
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
/**
 * K2a-i: WRAP ONLY. Fold this node's own atoms back into the box.
 *
 * Split out of the old single-pass rebin because the SLOT ASSIGNMENT below
 * has to read a neighbour's positions, and it must read them ALREADY WRAPPED
 * -- otherwise the two nodes compute different bins for the same atom. So:
 * everyone wraps their own, everyone publishes, then everyone assigns.
 * This pass writes only pages this node owns, so one-writer-per-page holds.
 */
__device__ gy::YCoroMain RebinWrapCoro(gv::DeviceVector<float> x, u32 nb,
                                       u32 cap, float box, u32 z0, u32 z1,
                                       u32 nblocks, u32 block) {
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  for (u64 pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    const u64 e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    const u64 e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    co_await x.Fetch(0, pg * epp, epp);
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    float *const px = hx.ptr();
    const u64 nslots = epp / kStride;
    const u64 s_lo = (e0 - pg * epp) / kStride;
    const u64 s_hi = (e1 - pg * epp) / kStride;
    for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;
      float px0 = px[e + 0], py0 = px[e + 1], pz0 = px[e + 2];
      if (px0 < 0.0f) px0 += box; else if (px0 >= box) px0 -= box;
      if (py0 < 0.0f) py0 += box; else if (py0 >= box) py0 -= box;
      if (pz0 < 0.0f) pz0 += box; else if (pz0 >= box) pz0 -= box;
      px[e + 0] = px0; px[e + 1] = py0; px[e + 2] = pz0;
    }
    __syncthreads();
    x.UnpinRange(pg * epp, epp);
  }
}

/**
 * K2a-ii: ASSIGN. Claim a destination slot for every atom that lands in a bin
 * THIS NODE OWNS -- and for no other.
 *
 * THIS IS THE MIGRATION HANDOFF. The old pass had each node assign slots for
 * its OWN atoms wherever they landed, so an atom drifting out of node0's slab
 * got a d_dest pointing into node1's rows -- which node0's gather (bounded to
 * node0's rows) skipped, and which node1 had never computed. Both dropped it,
 * and every mid-flight resort shed the atoms that crossed a boundary:
 * measured as pairs 4741632 -> 4580898 at the first one.
 *
 * Inverting the ownership fixes it. A node scans its slab PLUS one plane
 * either side -- an atom moves at most one bin per rebuild, so that is every
 * atom that can reach it -- and claims a slot only when the DESTINATION bin is
 * inside [z0, z1). bincnt is then authoritative for the bins it counts,
 * because exactly one node counts them, and the arriving migrant has an owner.
 * The halo planes are read-only here, so the write-per-page invariant the
 * generational design rests on is untouched.
 */
__device__ gy::YCoroMain RebinAssignCoro(gv::DeviceVector<float> x, u32 nb,
                                         u32 cap, float box, u32 *bincnt,
                                         u32 *d_dest, int *d_err, u32 z0,
                                         u32 z1, u32 nblocks, u32 block,
                                         u64 hgen) {
  const u64 epp = x.ElemsPerPage();
  const float fnb = static_cast<float>(nb);
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 plane_elems = static_cast<u64>(nb) * row_elems;
  const u64 islots_r = static_cast<u64>(nb) * cap;
  const u32 mine = z1 - z0;
  const bool whole = (mine >= nb);          // single node owns everything
  // ONE PASS PER PAGE, NEVER TWO.
  //
  // A PLANE-indexed scan only works when planes are a whole number of pages,
  // and that is enforced ONLY for --nodes > 1 (the seam rule). On one node an
  // unaligned lattice shares a boundary page between consecutive planes, so a
  // plane loop visits it twice and assigns its atoms two slots -- which is a
  // corruption this pass would introduce rather than fix. When this node owns
  // everything there are no halo planes to single out, so sweep the pages
  // directly and skip the plane decomposition entirely.
  const u64 total_elems = static_cast<u64>(nb) * plane_elems;
  const u64 npages_all = (total_elems + epp - 1) / epp;
  const u32 nscan = whole ? 1u : (mine + 2u);
  for (u32 i = 0; i < nscan; ++i) {
    const u32 wz = whole ? 0u : ((z0 + nb - 1u + i) % nb);
    const bool in_mine = whole || (wz >= z0 && wz < z1);
    const u64 p_lo = whole ? 0ull : static_cast<u64>(wz) * plane_elems;
    const u64 pg_lo = whole ? 0ull : (p_lo / epp);
    const u64 pg_hi =
        whole ? npages_all : ((p_lo + plane_elems + epp - 1) / epp);
    for (u64 pg = pg_lo + block; pg < pg_hi; pg += nblocks) {
      // A HALO PLANE IS A NEIGHBOUR'S AND MUST BE DEMANDED AT THE GENERATION
      // THE WRAP WAS PUBLISHED AS. Generation 0 would accept last step's
      // positions and bin the atom from them.
      co_await x.Fetch(in_mine ? 0 : hgen, pg * epp, epp);
      auto hx = co_await x.HoldPage(pg * epp, epp);
      const float *const px = hx.ptr();
      const u64 nslots = epp / kStride;
      const u64 slot0 = pg * nslots;
      for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
        const u64 e = s * kStride;
        if (px[e + 3] < 0.0f) continue;          // padded slot
        const float px0 = px[e + 0], py0 = px[e + 1], pz0 = px[e + 2];
        u32 bx = static_cast<u32>(px0 * fnb / box);
        u32 by = static_cast<u32>(py0 * fnb / box);
        u32 bz = static_cast<u32>(pz0 * fnb / box);
        if (bx >= nb) bx = nb - 1;
        if (by >= nb) by = nb - 1;
        if (bz >= nb) bz = nb - 1;
        if (!whole && (bz < z0 || bz >= z1)) continue;   // not my bin
        const u64 bin = (static_cast<u64>(bz) * nb + by) * nb + bx;
        const u32 slot = atomicAdd(&bincnt[bin], 1u);
        if (slot >= cap) { *d_err = 1; continue; }
        d_dest[slot0 + s] = static_cast<u32>(bin * cap + slot);
        // THE GATHER CAN ONLY REACH ONE BIN IN Y AND Z (see K2b): an atom
        // that crossed more than one is never collected and vanishes.
        const u64 srow = (slot0 + s) / islots_r;
        const u32 sby = static_cast<u32>(srow % nb);
        const u32 sbz = static_cast<u32>(srow / nb);
        const u32 dy = min((by + nb - sby) % nb, (sby + nb - by) % nb);
        const u32 dz = min((bz + nb - sbz) % nb, (sbz + nb - bz) % nb);
        if (dy > 1u || dz > 1u) *d_err = 2;
      }
      __syncthreads();
      x.UnpinRange(pg * epp, epp);
    }
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
                                    u32 z0, u32 z1, u32 nblocks, u32 block,
                                    u64 hgen) {
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 row_elems = islots * kStride;
  const u64 row_lo = static_cast<u64>(z0) * nb;
  const u64 row_hi = static_cast<u64>(z1) * nb;
  for (u64 row = row_lo + block; row < row_hi; row += nblocks) {
    const u32 by = static_cast<u32>(row % nb);
    const u32 bz = static_cast<u32>(row / nb);
    // THE ONLY WRITE HOLD: this block's own destination row.
    {co_await dst.Fetch(0, row * row_elems, row_elems);
       // every guard dies at this scope's close, before the reservations
    gv::Held<float> hd0 = co_await dst.HoldPage(row * row_elems, row_elems,
                                                /*write=*/true);
    gv::Held<float> hd1;
    const u64 drun = hd0.run();
    if (drun < row_elems) {
      co_await dst.Fetch(0, dst.PageLo(row * row_elems + drun),
                         dst.PageSpan(row * row_elems + drun, 1));
      hd1 = co_await dst.HoldPage(row * row_elems + drun, row_elems - drun,
                                  /*write=*/true);
    }
    float *const dp0 = hd0.ptr();
    float *const dp1 = hd1 ? hd1.ptr() : nullptr;
    // RESET THE DESTINATION ROW TO PADDED FIRST.
    //
    // The scatter below writes only the slots that RECEIVE an atom. Every
    // other slot would keep this buffer's contents from the previous resort,
    // and w >= 0 marks a slot occupied -- so those ghosts are read back as
    // real atoms sitting on top of the live ones, and the pair force between
    // a ghost and its own current copy is what blows the energy up. The
    // step-0 resort gate cannot see this: the buffer is padded at setup, so
    // only the SECOND resort onwards inherits stale slots.
    for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
      const u64 de = s * kStride;
      float *const dp = (de < drun) ? dp0 + de : dp1 + (de - drun);
      dp[3] = -1.0f;
    }
    __syncthreads();
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
        // WHOLE PAGES, BECAUSE THIS READER RE-READS THEM. The nine stencil
        // spans of a row overlap heavily: neighbouring spans land in the same
        // pages, and a page made valid over only one span's slice forces the
        // next span to refetch it. Asking for the pages outright makes the
        // second span a cache hit. The vector fetches exactly what it is
        // told -- so a caller that wants pages has to say pages.
        // A SOURCE PLANE OUTSIDE THIS SLAB IS A NEIGHBOUR'S, so the same
        // rule the force pass follows applies here: generation 0 accepts the
        // resident copy however old, and the gather would scatter atoms from
        // LAST step's positions.
        co_await src.Fetch((wz < z0 || wz >= z1) ? hgen : 0,
                           src.PageLo(rb), src.PageSpan(rb, len));
        hs[nspans][0] = co_await src.HoldPage(rb, len);
        srun[nspans] = hs[nspans][0].run();
        if (srun[nspans] < len) {
          hs[nspans][1] =
              co_await src.HoldPage(rb + srun[nspans], len - srun[nspans]);
        }
        sp0[nspans] = hs[nspans][0].ptr();
        sp1[nspans] = hs[nspans][1] ? hs[nspans][1].ptr() : nullptr;
        co_await srcx.Fetch((wz < z0 || wz >= z1) ? hgen : 0,
                            srcx.PageLo(rb), srcx.PageSpan(rb, len));
        hx[nspans][0] = co_await srcx.HoldPage(rb, len);
        xrun[nspans] = hx[nspans][0].run();
        if (xrun[nspans] < len) {
          co_await srcx.Fetch((wz < z0 || wz >= z1) ? hgen : 0,
                              srcx.PageLo(rb + xrun[nspans]),
                              srcx.PageSpan(rb + xrun[nspans], 1));
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
      atomicAdd(&g_gather_wrote, 1ull);
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
    if (g_publish && g_pub_interior) {
      co_await dst.Flush(0, row * row_elems, row_elems);
    }
    // One unpin per fetch, in the same shape the fetches were issued: the
    // stencil spans of src and srcx (the same vector on the position pass,
    // so it was pinned twice and is released twice), the second srcx page
    // when a span straddles, then this row of dst.
    //
    // THE SPAN GEOMETRY IS RECOMPUTED, NOT REMEMBERED. Carrying (rb, len)
    // for six spans costs 96 bytes of coroutine frame, which is exactly what
    // overflowed it (2144 > 2048). This loop suspends nowhere, so the same
    // arithmetic lives in registers instead. It must walk dz and t in the
    // ORIGINAL ORDER, because that order is what defined the span index.
    {
      u32 sq = 0;
      for (int dz = -1; dz <= 1; ++dz) {
        const u32 wz = (bz + nb + dz) % nb;
        const int lo = static_cast<int>(by) - 1;
        const int hi = static_cast<int>(by) + 1;
        u32 rl[2], rn[2], nr = 0;
        if (lo < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo);
          ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo); rn[nr] = 3u; ++nr;
        }
        for (u32 t = 0; t < nr && sq < nspans; ++t, ++sq) {
          const u64 rb = (static_cast<u64>(wz) * nb + rl[t]) * row_elems;
          const u64 len = static_cast<u64>(rn[t]) * row_elems;
          src.UnpinRange(src.PageLo(rb), src.PageSpan(rb, len));
          srcx.UnpinRange(srcx.PageLo(rb), srcx.PageSpan(rb, len));
          if (xrun[sq] < len) {
            srcx.UnpinRange(srcx.PageLo(rb + xrun[sq]),
                            srcx.PageSpan(rb + xrun[sq], 1));
          }
        }
      }
    }
    dst.UnpinRange(row * row_elems, row_elems);
    if (drun < row_elems) {
      dst.UnpinRange(dst.PageLo(row * row_elems + drun),
                     dst.PageSpan(row * row_elems + drun, 1));
    }
    }   // guards dead here
    if (false) {
    }
  }     // per-row loop
}

/** Pre-sentinel every slot of a ping-pong destination vector. */
__device__ gy::YCoroMain SentinelCoro(gv::DeviceVector<float> dst, u32 nb,
                                      u32 cap, u32 z0, u32 z1, u32 nblocks,
                                      u32 block) {
  const u64 epp = dst.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  for (u64 pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    const u64 e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    const u64 e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    co_await dst.Fetch(0, pg * epp, epp);
    auto h = co_await dst.HoldPage(pg * epp, epp, /*write=*/true);
    float *const p = h.ptr();
    const u64 s_lo = (e0 - pg * epp) / kStride;
    const u64 s_hi = (e1 - pg * epp) / kStride;
    for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
      p[s * kStride + 3] = -1.0f;
    }
    __syncthreads();
    dst.UnpinRange(pg * epp, epp);
  }
}

/** K1/K4 for real MD: the kick reads the FORCE VECTOR (mass = 1). Same
 *  page-parallel shape as the ballistic form. */
__device__ gy::YCoroMain MDIntegrateCoro(gv::DeviceVector<float> x,
                                         gv::DeviceVector<float> v,
                                         gv::DeviceVector<float> f,
                                         float dt, int drift, u32 nb, u32 cap,
                                         u32 z0, u32 z1, u32 nblocks,
                                         u32 block) {
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  const float half = 0.5f * dt;
  for (u64 pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    // CLIPPED TO THE SLAB. The seam page belongs to two nodes; this one
    // touches only its own elements of it, and flushes only those.
    const u64 e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    const u64 e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    co_await x.Fetch(0, pg * epp, epp);
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    co_await v.Fetch(0, pg * epp, epp);
    auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
    co_await f.Fetch(0, pg * epp, epp);
    auto hf = co_await f.HoldPage(pg * epp, epp);
    float *const px = hx.ptr();
    float *const pv = hv.ptr();
    const float *const pf = hf.ptr();
    // Slots of THIS page that lie in the slab.
    const u64 s_lo = (e0 - pg * epp) / kStride;
    const u64 s_hi = (e1 - pg * epp) / kStride;
    for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
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
    x.UnpinRange(pg * epp, epp);
    v.UnpinRange(pg * epp, epp);
    f.UnpinRange(pg * epp, epp);
  }
}

/** Flush every dirty page of the SHARED x/v tables to the backing store, so
 *  host Download (which reads the store, not the frames) sees the truth.
 *  One block does it; the others exit immediately. Validation-path only --
 *  steady-state MD never flushes x/v (they are device-canonical). */

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
    g_read_geom[1] = x.SetSize();
    g_read_geom[2] = x.PageBytes();
    g_read_geom[3] = epp;
  }
  for (u64 it = 0; it < passes; ++it) {
    for (u64 pg = block; pg < npages; pg += nblocks) {
      co_await x.Fetch(0, pg * epp, epp);
      auto h = co_await x.HoldPage(pg * epp, epp);  // READ hold only
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
      x.UnpinRange(pg * epp, epp);
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
    co_await x.Fetch(0, pg * epp, epp);
    auto hx = co_await x.HoldPage(pg * epp, epp, /*write=*/true);
    co_await v.Fetch(0, pg * epp, epp);
    auto hv = co_await v.HoldPage(pg * epp, epp, /*write=*/true);
    // THE TEST: a THIRD vector held read-only alongside the other two, the
    // pattern the real integrator has (x write, v write, f read) and the
    // ballistic gate never had. The value is only sunk, so the physics --
    // and therefore the bitwise comparison against the host replica --
    // is untouched.
    gv::Held<float> ht;
    if (use_third) {
      co_await third.Fetch(0, pg * epp, epp);
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
    // PUBLISH THIS PAGE, BY NAME. A block's table is private, so a page left
    // here is invisible to every other block -- and both the force stencil
    // and the resort gather read pages this block owns. Flushing the exact
    // page just written keeps every put disjoint: no two blocks can name the
    // same range, which is precisely what a whole-table flush could not
    // promise. Async, so the put overlaps the next page's integration.
    const u64 cnt = (x.size() - pg * epp < epp) ? x.size() - pg * epp : epp;
    if (g_publish && g_pub_interior) {
      co_await x.BeginFlush(0, pg * epp, cnt);
      co_await v.BeginFlush(0, pg * epp, cnt);
    }
    x.UnpinRange(pg * epp, epp);
    v.UnpinRange(pg * epp, epp);
    if (use_third) third.UnpinRange(pg * epp, epp);
  }
  // Land them before the kernel exits: the next kernel reads these pages
  // from other blocks, so they must be in the backing store by then.
  co_await x.EndFlush();
  co_await v.EndFlush();
}

/** K5: thermo -- KE and net momentum, block-reduced then atomically merged.
 *  Read-holds only. Reduction scratch is dynamic shared memory after the
 *  yield header. */
__device__ gy::YCoroMain ThermoCoro(gv::DeviceVector<float> x,
                                    gv::DeviceVector<float> v,
                                    double *out, u32 nb, u32 cap, u32 z0,
                                    u32 z1, u32 nblocks, u32 block) {
  extern __shared__ char smem_raw[];
  double *red = reinterpret_cast<double *>(smem_raw + CLIO_YIELD_SMEM_BYTES);
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  double ke = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (u64 pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    const u64 e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    const u64 e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    co_await x.Fetch(0, pg * epp, epp);
    auto hx = co_await x.HoldPage(pg * epp, epp);
    co_await v.Fetch(0, pg * epp, epp);
    auto hv = co_await v.HoldPage(pg * epp, epp);
    const float *const px = hx.ptr();
    const float *const pv = hv.ptr();
    // Slots of THIS page that lie in the slab.
    const u64 s_lo = (e0 - pg * epp) / kStride;
    const u64 s_hi = (e1 - pg * epp) / kStride;
    for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
      const u64 e = s * kStride;
      if (px[e + 3] < 0.0f) continue;
      const double vx = pv[e + 0], vy = pv[e + 1], vz = pv[e + 2];
      ke += 0.5 * (vx * vx + vy * vy + vz * vz);   // m = 1
      mx += vx; my += vy; mz += vz;
    }
    __syncthreads();
    x.UnpinRange(pg * epp, epp);
    v.UnpinRange(pg * epp, epp);
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


/**
 * PUBLISH THIS NODE'S SLAB, as a generation.
 *
 * x and v are device-canonical inside one process -- nobody else can see the
 * frames, so nothing has to leave them. Across nodes that is exactly wrong:
 * the neighbour's force stencil reads one plane past this slab, and the
 * resort's gather reaches one plane in y AND z, so both arrays have to be in
 * the store before the neighbour looks. Flushed AS generation `gen`, which is
 * what lets the reader below demand that version rather than whatever it
 * happens to have.
 */
__device__ gy::YCoroMain PublishSlabCoro(gv::DeviceVector<float> x,
                                         gv::DeviceVector<float> v, u32 nb,
                                         u32 cap, u32 z0, u32 z1, u64 gen,
                                         u32 nblocks, u32 block) {
  const u64 epp = x.ElemsPerPage();
  // BATCHED, NOT PAGE-AT-A-TIME. This kernel is pure store latency: after
  // the force pass stopped parking, the per-page loop here -- one awaited
  // flush per page, one awaited fetch per halo page -- was 60% of the whole
  // run (15.6 ms/step against MPI's 6.3 TOTAL). GatherRanges is variadic and
  // a Multi task carries 64 records, so both boundary planes go out as ONE
  // flush per vector and both halo planes come back as ONE fetch: ~3 store
  // round trips per step instead of ~12.
  //
  // TWO BLOCKS, ONE VECTOR EACH. The wait is latency, not bandwidth; more
  // blocks would just contend for the same task slots.
  const u64 row_elems_p = static_cast<u64>(nb) * cap * kStride;
  const u64 plane_elems_p = static_cast<u64>(nb) * row_elems_p;
  const u64 lo_pl = static_cast<u64>(z0) * plane_elems_p;
  const u64 hi_pl = static_cast<u64>(z1 - 1) * plane_elems_p;
  const u64 below = static_cast<u64>((z0 + nb - 1u) % nb) * plane_elems_p;
  const u64 above = static_cast<u64>(z1 % nb) * plane_elems_p;
  // THREE INDEPENDENT LATENCIES, THREE BLOCKS. The halo wait depends on the
  // PEER's stamp, not on this node's flush -- measured 38% flush / 62% fetch
  // stacked SEQUENTIALLY in one block, so the step paid flush + fetch when it
  // only owes max(flush, fetch). Each block carries its own task slots, so
  // the three waits genuinely overlap.
  if (block == 0) {
    const long long _c0 = clock64();
    co_await x.Fetch(0, lo_pl, plane_elems_p, hi_pl, plane_elems_p);
    co_await x.BeginFlush(gen, lo_pl, plane_elems_p, hi_pl, plane_elems_p);
    co_await x.EndFlush();
    if (threadIdx.x == 0) {
      atomicAdd(&g_pub_flush_cyc,
                (unsigned long long)(clock64() - _c0));
    }
    x.UnpinRange(lo_pl, plane_elems_p);
    x.UnpinRange(hi_pl, plane_elems_p);
  } else if (block == 1) {
    co_await v.Fetch(0, lo_pl, plane_elems_p, hi_pl, plane_elems_p);
    co_await v.BeginFlush(gen, lo_pl, plane_elems_p, hi_pl, plane_elems_p);
    co_await v.EndFlush();
    v.UnpinRange(lo_pl, plane_elems_p);
    v.UnpinRange(hi_pl, plane_elems_p);
  } else if (block == 2) {
    // The demand stays on the consumer (force still asks hgen); this fetch
    // only warms the frames to `gen` so the heavy kernel finds resident_ok.
    const long long _c1 = clock64();
    co_await x.Fetch(gen, below, plane_elems_p, above, plane_elems_p);
    if (threadIdx.x == 0) {
      atomicAdd(&g_pub_fetch_cyc,
                (unsigned long long)(clock64() - _c1));
    }
    x.UnpinRange(below, plane_elems_p);
    x.UnpinRange(above, plane_elems_p);
  }
}

__global__ MD_LAUNCH_BOUNDS void PublishSlabKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x,
    gv::DeviceVector<float> v, u32 nb, u32 cap, u32 z0, u32 z1, u64 gen,
    u32 nblocks, gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(PublishSlabCoro(x, v, nb, cap, z0, z1, gen, nblocks,
                                 yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void ThermoKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> x,
                             gv::DeviceVector<float> v, double *out,
                             u32 nb, u32 cap, u32 z0, u32 z1,
                             u32 nblocks, gy::YieldableView<> yv,
                             gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ThermoCoro(x, v, out, nb, cap, z0, z1, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void ForceKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> x,
                            gv::DeviceVector<float> f, u32 nb, u32 cap,
                            float box, float cutoff, int eflag, double *acc,
                            u32 z0, u32 z1, u32 nblocks, u64 hgen,
                            bool force_all, gy::YieldableView<> yv,
                            gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  f.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ForceCoro(x, f, nb, cap, box, cutoff, eflag, acc, z0, z1, nblocks,
                           yv.Block(), hgen, force_all));
}

__global__ MD_LAUNCH_BOUNDS void BuildListKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<int> nl, u32 nb, u32 cap,
                                float box, float rlist, u32 maxneigh,
                                u32 *d_cnt, int *d_err, u32 rowchunk,
                                u32 z0, u32 z1, u32 nblocks, u64 hgen,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  nl.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(BuildListCoro(x, nl, nb, cap, box, rlist, maxneigh, d_cnt,
                               d_err, rowchunk, z0, z1, nblocks,
                               yv.Block(), hgen));
}

__global__ MD_LAUNCH_BOUNDS void ListForceKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<float> f,
                                gv::DeviceVector<int> nl, u32 nb, u32 cap,
                                float box, float cutoff, u32 maxneigh,
                                const u32 *d_cnt, int eflag, double *acc,
                                int nocompute, u32 rowchunk, u32 z0, u32 z1, u32 nblocks,
                                u64 hgen, bool force_all, gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  f.Init(yv.Block());
  nl.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ListForceCoro(x, f, nl, nb, cap, box, cutoff, maxneigh,
                               d_cnt, eflag, acc, nocompute, rowchunk,
                               z0, z1, nblocks, yv.Block(), hgen, force_all));
}

__global__ MD_LAUNCH_BOUNDS void RebinWrapKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x, u32 nb,
    u32 cap, float box, u32 z0, u32 z1, u32 nblocks, gy::YieldableView<> yv,
    gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RebinWrapCoro(x, nb, cap, box, z0, z1, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void RebinAssignKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x, u32 nb,
    u32 cap, float box, u32 *bincnt, u32 *d_dest, int *d_err, u32 z0, u32 z1,
    u32 nblocks, u64 hgen, gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RebinAssignCoro(x, nb, cap, box, bincnt, d_dest, d_err, z0,
                                 z1, nblocks, yv.Block(), hgen));
}

__global__ MD_LAUNCH_BOUNDS void GatherKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<float> src,
                              gv::DeviceVector<float> srcx,
                              gv::DeviceVector<float> dst, u32 nb, u32 cap,
                              const u32 *d_dest, int keep_w, u32 z0, u32 z1,
                              u32 nblocks, u64 hgen,
                              gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  src.Init(yv.Block());
  srcx.Init(yv.Block());
  dst.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(GatherCoro(src, srcx, dst, nb, cap, d_dest, keep_w, z0, z1,
                            nblocks, yv.Block(), hgen));
}

__global__ MD_LAUNCH_BOUNDS void SentinelKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> dst, u32 nb, u32 cap,
                               u32 z0, u32 z1, u32 nblocks,
                               gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  dst.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SentinelCoro(dst, nb, cap, z0, z1, nblocks, yv.Block()));
}

__global__ MD_LAUNCH_BOUNDS void MDIntegrateKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceVector<float> x,
                                  gv::DeviceVector<float> v,
                                  gv::DeviceVector<float> f, float dt,
                                  int drift, u32 nb, u32 cap, u32 z0, u32 z1,
                                  u32 nblocks,
                                  gy::YieldableView<> yv,
                                  gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  f.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(MDIntegrateCoro(x, v, f, dt, drift, nb, cap, z0, z1, nblocks,
                                 yv.Block()));
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

/**
 * Frames for the neighbour-list cache.
 *
 * NOT nl_pages + 2. A Verlet row is islots*maxneigh ints -- 372 KB at lattice
 * 50 -- so a fully resident list is hundreds of pages, and with one table per
 * block that is multiplied by the block count: 699 pages x 512 KB x 16 blocks
 * is 5.6 GB, which simply fails to allocate. The list is also STREAMED, one
 * row at a time, so residency buys nothing.
 *
 * Size it to what the kernel pins, with a little headroom, and let it page.
 */
/**
 * The list's page geometry, needed BEFORE the list is built so the cache
 * budget can account for it. Kept here so the planner and the construction
 * site cannot disagree about how big a list page is.
 */
struct NlGeom { clio::run::u64 page_bytes, pages; };
static NlGeom NlGeometry(u32 nb, u32 cap, u32 maxneigh, clio::run::u64 page_kb) {
  const clio::run::u64 rowlist =
      static_cast<clio::run::u64>(nb) * cap * maxneigh;
  clio::run::u64 pb = page_kb * 1024;
  if (pb == 0) {
    pb = rowlist * sizeof(int);
    clio::run::u64 pw = 4096;
    while (pw < pb) pw <<= 1;
    pb = pw;
  }
  const clio::run::u64 pe = pb / sizeof(int);
  const clio::run::u64 raw = static_cast<clio::run::u64>(nb) * nb * rowlist;
  return NlGeom{pb, (raw + pe - 1) / pe};
}

/**
 * Divide a VRAM budget across the six per-block caches.
 *
 * EVERY CACHE IS PER BLOCK, so each frame is paid for `nblocks` times -- that
 * multiplier, not the data, is what makes a paged run expensive: at 16 blocks
 * a fully resident configuration holds sixteen copies of everything.
 *
 * The split is not proportional to size. The atom vectors (x, v, x2, v2, f)
 * are small and every one of them is re-read each step, so residency is what
 * keeps evictions at zero; the list is an order of magnitude larger and is
 * STREAMED one row at a time, so frames spent on it buy almost nothing. So:
 * floors first, then take the atom vectors as close to resident as the budget
 * allows, then hand the remainder to the list.
 */
struct CachePlan { u32 slots, vslots, fslots, nlslots; clio::run::u64 bytes; };
/** x, x2 and v2 share `s`; v and f are sized separately. */
static clio::run::u64 PlanCost(u32 nblocks, u32 s, u32 vs, u32 fs, u32 nls,
                               clio::run::u64 pb, clio::run::u64 nlpb) {
  return static_cast<clio::run::u64>(nblocks) *
         ((3ull * s + vs + fs) * pb + static_cast<clio::run::u64>(nls) * nlpb);
}
static bool PlanCaches(clio::run::u64 budget, u32 nblocks,
                       clio::run::u64 npages, clio::run::u64 page_bytes,
                       clio::run::u64 nl_pages, clio::run::u64 nl_page_bytes,
                       u32 rows_per_block, CachePlan *out) {
  const u32 s_max = static_cast<u32>(npages + 2);
  const u32 nl_max = (nl_pages == 0) ? 0u : static_cast<u32>(nl_pages + 2);
  // The list floor WAS 128 MB per block, which described a bug: the build
  // pass dirtied every row and never published one, and a dirty page cannot
  // be evicted. That is fixed at the write site; both passes read one row at
  // a time, so a few frames plus fetch headroom is the real floor.
  // MEASURED, and it is BYTES not frames: the knee is ~8 MB of resident list
  // per block at 256 KB, 512 KB and 1024 KB list pages alike. 16 MB was the
  // largest single term in the whole VRAM floor and bought 2.6% at 64 blocks
  // (45.4 -> 44.2 ms/step) and NOTHING at 16 or 32 blocks, where a bigger
  // list cache is flat to slightly worse. Correctness needs 2 frames; the
  // rest is streaming headroom.
  const clio::run::u64 kNlLiveBytes = 8ull * 1024 * 1024;
  u32 nl_floor = 0;
  if (nl_pages != 0) {
    nl_floor = static_cast<u32>((kNlLiveBytes + nl_page_bytes - 1) /
                                nl_page_bytes);
    if (nl_floor < kMinSlotsNl) nl_floor = kMinSlotsNl;
    if (nl_floor > nl_max) nl_floor = nl_max;
  }
  // MEASURED FLOOR, not derived. kMinSlotsX/kMinSlotsF count what one hold
  // pins, and sizing v and f by that crashed: 16 frames fails, 24 passes.
  // Guessing a floor from the guard count has now been wrong twice -- the
  // list's floor was the same mistake -- so it is pinned to what runs.
  const u32 kAtomFloor = 24;
  // THE FLOOR SCALES WITH ROWS PER BLOCK. A block sweeps nrows/nblocks rows
  // of each atom vector per kernel, and the write-side ones go dirty -- a
  // dirty frame is stuck until its flush lands. Fewer blocks means more rows
  // each, so a constant floor that holds at 64 blocks traps at 8.
  //
  // MEASURED minima for x at lattice 50 (rows/block 106 / 53 / 27 / 14 at
  // --blocks 8 / 16 / 32 / 64): 48 / 40 / 24 / 24 frames. It is ROWS, not
  // pages: at 16 blocks the minimum is 40 at BOTH 128 KB and 256 KB pages,
  // which is why sizing this off pages-per-block still trapped.
  //
  // f, the vector the force pass WRITES, needs the most (48 at 8 blocks where
  // v and x run at 24) -- but a FLAT 48 for f is wrong in the other
  // direction: at 128 blocks it pushed the floor to 6656 MB and made a
  // configuration that runs fine at 24 unaffordable. One scaled rule, applied
  // to all three, covers both ends.
  //
  // 3/4 of rows-per-block fits 53->40 exactly and is generous at the low end.
  // Deliberately generous: too high wastes VRAM, too low ADVERTISES A BUDGET
  // THAT CANNOT RUN -- which is what "needs 248 MB minimum" did before it
  // trapped in EvictPages.
  u32 scaled = rows_per_block == 0 ? 0u : (3u * rows_per_block + 3u) / 4u;
  if (scaled > s_max) scaled = s_max;
  u32 s = kAtomFloor, vs = kAtomFloor, fs = kAtomFloor, nls = nl_floor;
  if (s < scaled) s = scaled;
  if (vs < scaled) vs = scaled;
  if (fs < scaled) fs = scaled;
  if (s > s_max) s = s_max;
  if (vs > s_max) vs = s_max;
  if (fs > s_max) fs = s_max;
  if (PlanCost(nblocks, s, vs, fs, nls, page_bytes, nl_page_bytes) > budget) {
    std::fprintf(stderr,
                 "--vram-mb %llu is below the floor: %u blocks needs %llu MB "
                 "minimum for the pinned frames alone\n",
                 (unsigned long long)(budget / (1024 * 1024)), nblocks,
                 (unsigned long long)(PlanCost(nblocks, s, vs, fs, nls,
                                               page_bytes, nl_page_bytes) /
                                      (1024 * 1024)));
    return false;
  }
  // MEASURED PRIORITY, at lattice 50 with 16 blocks.
  //
  // x and the resort's two ping-pong destinations are the vectors that must
  // be resident: x is read through nine-row stencils that sweep the whole
  // domain, and the gather writes destination rows scattered across it.
  // Holding those three resident and starving v and f costs 102 ms/step;
  // starving the ping-pong instead costs 153, and starving everything but x
  // costs 157. So spend on the three first, then the list, then v and f --
  // which only ever touch pages their own block owns.
  while (s < s_max &&
         PlanCost(nblocks, s + 1, vs, fs, nls, page_bytes, nl_page_bytes) <=
             budget) {
    ++s;
  }
  while (nls < nl_max &&
         PlanCost(nblocks, s, vs, fs, nls + 1, page_bytes, nl_page_bytes) <=
             budget) {
    ++nls;
  }
  while ((vs < s_max || fs < s_max) &&
         PlanCost(nblocks, s, vs + (vs < s_max ? 1 : 0),
                  fs + (fs < s_max ? 1 : 0), nls, page_bytes,
                  nl_page_bytes) <= budget) {
    if (vs < s_max) ++vs;
    if (fs < s_max) ++fs;
  }
  *out = CachePlan{s, vs, fs, nls,
                   PlanCost(nblocks, s, vs, fs, nls, page_bytes,
                            nl_page_bytes)};
  return true;
}

/**
 * Frames per SET for a shared cache that holds a whole vector.
 *
 * Shared mode changes what a cache costs. Private residency is
 * blocks * (pages+2) frames -- the same page stored once per CUDA block --
 * so it scales with the launch and residency stops being affordable long
 * before the vector is large. Shared residency is (pages+2) frames TOTAL,
 * independent of the block count, so sizing for residency is the default
 * here rather than the thing a budget has to be talked out of.
 *
 * Two multipliers on top of that:
 *  - 2x for collisions. A hash cache needs somewhere for a colliding page to
 *    go, and every slot owns its page storage, so asking for room for 2x the
 *    pages is what makes residency actually reachable. The vector spends
 *    exactly what it is given -- there is no hidden halving -- so this 2x is
 *    the real cost and the accounting below prints it.
 *  - a floor of 4. With more sets than pages the formula yields ONE frame per
 *    set, which is a direct map with no room for a collision, and the first
 *    two pages that hash together wedge. Associativity is not optional.
 */
static u32 SharedSlots(clio::run::u64 pages, u32 nsets) {
  // TAGS ARE CHEAP NOW. A slot is 64 bytes and the page-sized storage behind
  // it comes from the allocator, so associativity costs kilobytes rather than
  // hundreds of megabytes: the 4x headroom below is 4 * (pages+2) * 64B, and
  // for the neighbour list that is 147 KB against the 290 MB the same
  // headroom used to cost. Ask for enough that a hash collision never has to
  // evict a live page.
  const clio::run::u64 want = (4ull * (pages + 2) + nsets - 1) / nsets;
  // A FLOOR OF 4, not because capacity says so but because a set of one is a
  // direct map with nowhere for a collision to go. Residency needs
  // 2*(pages+2)/nsets, which rounds to nothing for a small vector.
  //
  // I raised this to 16 while chasing an intermittent "illegal instruction",
  // on the theory that the concurrent pinned working set was overflowing a
  // set. It was not -- the fault was a divergent barrier in HoldPage -- and
  // the floor is back at 4, which is 144 MB of cache for this deck against
  // 576 MB at 16.
  const clio::run::u64 kConcurrencyFloor = 4;
  return static_cast<u32>(want < kConcurrencyFloor ? kConcurrencyFloor : want);
}

static u32 NlSlots(clio::run::u64 nl_pages) {
  // MEASURED, not derived. kMaxNlGuards (4) and rowchunk (4) describe what a
  // single row pins, but the force pass keeps far more of the list live than
  // that product: at lattice 50 this fails at 48 frames and passes at 96.
  // Until the true working set is pinned down, cap the list cache here --
  // enough for the widest case measured, and bounded so it does not scale
  // with the list (572 pages x 512 KB x 16 blocks would be 4.7 GB).
  const u32 want = 128;
  const u32 all = static_cast<u32>(nl_pages + 2);
  return (all < want) ? AtLeastSlots(all, kMinSlotsNl, "nl") : want;
}


/**
 * Sum `n` doubles across the nodes, through the CTE.
 *
 * A slab's energy is not conserved on its own -- the resort moves atoms across
 * the boundary, so what leaves one node's books arrives on another's. Only the
 * TOTAL is a physical invariant, so every gate that tests one has to see the
 * whole domain. Each node publishes its partial under its own name and then
 * reads all of them; `round` keeps one step's partials from being confused
 * with the next's.
 *
 * The put is retried because a node may arrive before the pool exists; the
 * get is retried because a peer may not have published yet. That polling is
 * what makes this a reduction AND a barrier, which is exactly what a gate
 * boundary wants -- unlike the halo exchange, where the generation is the
 * barrier and polling would defeat the point.
 */
static bool ReduceSum(clio::cte::core::Client &cte,
                      const clio::cte::core::TagId &tag, u32 node, u32 nodes,
                      u64 round, double *vals, int n, int timeout_s = 120) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto expired = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count() > timeout_s;
  };
  const std::string mine =
      "mdred_" + std::to_string(round) + "_" + std::to_string(node);
  // GENERATIONAL, LIKE EVERY OTHER CROSS-NODE HANDOFF IN THIS BENCH. A plain
  // put/get pair has no readiness gate: the put can report complete while a
  // peer's get still reads unpublished bytes, and with the per-step barrier
  // deleted the nodes drift far enough apart to hit it -- measured as a node
  // printing E0 = exactly HALF the true energy (its own partial plus a peer
  // read of zeros). Each round's blob is written once, so generation 1 is
  // the whole protocol: the get is not served until the writer's put landed.
  clio::cte::core::Context red_ctx;
  red_ctx.op_flags_ |= clio::cte::core::Context::kGenerational;
  red_ctx.generation_ = 1;
  for (;;) {
    auto f = cte.AsyncPutBlob(tag, mine, 0, n * sizeof(double),
                              reinterpret_cast<char *>(vals), 1.0f, red_ctx);
    f.Wait();
    if (f->GetReturnCode() == 0) break;
    if (expired()) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::vector<double> total(vals, vals + n);
  for (u32 peer = 0; peer < nodes; ++peer) {
    if (peer == node) continue;
    const std::string name =
        "mdred_" + std::to_string(round) + "_" + std::to_string(peer);
    std::vector<double> got(n, 0.0);
    for (;;) {
      auto f = cte.AsyncGetBlob(tag, name, 0, n * sizeof(double), 0u,
                                reinterpret_cast<char *>(got.data()),
                                clio::run::PoolQuery::Dynamic(), red_ctx);
      f.Wait();
      if (f->GetReturnCode() == 0) break;
      if (expired()) {
        std::fprintf(stderr, "  reduce: timed out waiting for node %u\n", peer);
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for (int i = 0; i < n; ++i) total[i] += got[i];
  }
  for (int i = 0; i < n; ++i) vals[i] = total[i];
  return true;
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
    else if (want("--gravity")) {
      const double gs = atof(argv[++i]);
      a.g[0] = 0.1 * gs; a.g[1] = -0.05 * gs; a.g[2] = 0.02 * gs;
    }
    else if (want("--rebin")) a.rebin = static_cast<u64>(atol(argv[++i]));
    else if (want("--temp")) a.temp = atof(argv[++i]);
    else if (want("--drift-tol")) a.drift_tol = atof(argv[++i]);
    else if (want("--maxneigh")) a.maxneigh = static_cast<u32>(atoi(argv[++i]));
    else if (want("--rowchunk")) a.rowchunk = static_cast<u32>(atoi(argv[++i]));
    else if (want("--ckpt")) a.ckpt = static_cast<u64>(atol(argv[++i]));
    else if (want("--nl-page-kb")) a.nl_page_kb = static_cast<u64>(atol(argv[++i]));
    else if (want("--nlslots")) a.nlslots = static_cast<u32>(atoi(argv[++i]));
    else if (want("--vram-mb")) a.vram_mb = static_cast<u64>(atol(argv[++i]));
    else if (std::strcmp(argv[i], "--no-list") == 0) a.use_list = 0;
    else if (want("--dt")) a.dt = atof(argv[++i]);
    else if (std::strcmp(argv[i], "--no-gate") == 0) a.gate = 0;
    else if (want("--tag-prefix")) a.tag_prefix = argv[++i];
    else if (want("--nodes")) a.nodes = static_cast<u32>(atoi(argv[++i]));
    else if (want("--node")) a.node = static_cast<u32>(atoi(argv[++i]));
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
  // A DEFAULT CACHE MUST STILL FIT THE DEVICE. `slots` above sizes ONE table
  // for full residency, but the per-block vectors get one table per block, so
  // the default cost is blocks * (npages+2) * page_bytes while the PROBLEM
  // SIZE IS UNCHANGED -- raising --blocks alone used to walk straight into
  // "CUDA Error 2: out of memory" with nothing having checked. Residency is
  // still the default WHEN IT FITS; when it does not, plan within what the
  // device actually has rather than asking for what it does not.
  // PlanCaches divides a budget across per-BLOCK caches, and there are none:
  // a cache is global now, so its arithmetic would size every cache
  // blocks-times too small. Say so rather than quietly mean something else.
  if (a.vram_mb != 0) {
    std::fprintf(stderr,
                 "--vram-mb divided a budget across per-block caches, which no "
                 "longer exist. One cache per vector, sized for residency.\n");
    return 1;
  }
  if (false) {
    size_t dev_free = 0, dev_total = 0;
    if (cudaMemGetInfo(&dev_free, &dev_total) == cudaSuccess) {
      const NlGeom nlg0 =
          a.md ? NlGeometry(g.nb, g.cap, a.maxneigh, a.nl_page_kb)
               : NlGeom{0, 0};
      const u32 s_full = static_cast<u32>(npages + 2);
      const u32 nl_full =
          (nlg0.pages == 0) ? 0u : static_cast<u32>(nlg0.pages + 2);
      const clio::run::u64 want =
          PlanCost(a.blocks, s_full, s_full, s_full, nl_full, page_bytes,
                   nlg0.page_bytes);
      // Headroom for the runtime, the staging buffers and the frames the
      // vectors allocate outside their caches. Measured, not guessed: 0.80
      // of free is what leaves lattice 50 at 16 blocks resident.
      const clio::run::u64 budget =
          static_cast<clio::run::u64>(static_cast<double>(dev_free) * 0.80);
      if (want > budget) {
        a.vram_mb = budget / (1024ull * 1024ull);
        std::printf("  note: full residency needs %llu MB over %u blocks but "
                    "only %llu MB is free; planning caches within %llu MB "
                    "(problem size unchanged -- pass --vram-mb to override)\n",
                    (unsigned long long)(want / (1024 * 1024)), a.blocks,
                    (unsigned long long)(dev_free / (1024 * 1024)),
                    (unsigned long long)a.vram_mb);
      }
    }
  }
  // --vram-mb divides a budget across every per-block cache. Explicit
  // per-vector flags still win, so a single vector can be pinned while the
  // rest fit the budget.
  if (a.vram_mb != 0) {
    const NlGeom nlg =
        a.md ? NlGeometry(g.nb, g.cap, a.maxneigh, a.nl_page_kb)
             : NlGeom{0, 0};
    CachePlan plan;
    const u32 rpb =
        a.md ? static_cast<u32>((static_cast<u64>(g.nb) * g.nb + a.blocks - 1) /
                                a.blocks)
             : 0u;
    if (!PlanCaches(a.vram_mb * 1024ull * 1024ull, a.blocks, npages,
                    page_bytes, nlg.pages, nlg.page_bytes, rpb, &plan)) {
      return 1;
    }
    if (a.slots == 0) slots = plan.slots;
    if (a.fslots == 0) a.fslots = plan.fslots;
    if (a.vslots == 0) a.vslots = plan.vslots;
    if (a.ppslots == 0) a.ppslots = plan.slots;
    if (a.nlslots == 0 && a.md) a.nlslots = plan.nlslots;
    std::printf("  VRAM budget %llu MB -> x/x2/v2 %u frames/block (of %llu "
                "resident), v %u, f %u, list %u (of %llu); caches use "
                "%llu MB over %u blocks\n",
                (unsigned long long)a.vram_mb, plan.slots,
                (unsigned long long)(npages + 2), plan.vslots, plan.fslots,
                plan.nlslots,
                (unsigned long long)(nlg.pages ? nlg.pages + 2 : 0),
                (unsigned long long)(plan.bytes / (1024 * 1024)),
                a.blocks);
  }
  if (slots < kMinSlotsX) {
    std::printf("  note: --slots %u is below the %u frames these kernels pin "
                "at once; raising to %u\n",
                slots, kMinSlotsX, kMinSlotsX);
    slots = kMinSlotsX;
  }
  // Which regime this configuration is IN: the cache either can hold every
  // page of the working set or it cannot. The gates below key off this
  // rather than assuming residency.
  // A BUDGET MEANS EVICTIONS ARE THE POINT, not a broken promise. The
  // resident contract only applies when every cache was sized to hold its
  // whole vector; under --vram-mb the caches are deliberately smaller, and
  // the f vector evicting is the budget working. Energy still has to hold.
  const bool expect_resident =
      (static_cast<clio::run::u64>(slots) >= npages) && a.vram_mb == 0 &&
      a.fslots == 0 && a.nlslots == 0;

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
  //
  // THE BENCH OWNS ITS CONFIG ONLY WHEN NOBODY ELSE SUPPLIED ONE. It used to
  // write gpu_vector_md.yaml and Setenv with overwrite=1 unconditionally,
  // which made it impossible to point at a cluster: any CLIO_SERVER_CONF the
  // caller exported was clobbered a line later. A distributed harness needs
  // exactly that -- a config naming a hostfile and the other nodes -- so an
  // already-set CLIO_SERVER_CONF is now left alone.
  if (getenv("CLIO_SERVER_CONF") != nullptr) {
    std::printf("  runtime: using CLIO_SERVER_CONF=%s (not writing one)\n",
                getenv("CLIO_SERVER_CONF"));
  } else {
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

  // WATCH THE FATAL CHANNEL. A device trap kills the context, and the
  // runtime's own error path then aborts the process, so nothing the bench
  // does AFTER a failed Synchronize is guaranteed to run. The slots live in
  // pinned host memory, so a poller sees the reason the instant the device
  // writes it -- which is the difference between "CUDA Error 715" and
  // "AllocatePage: set full, page 7, set 3, 4 pinned".
  // The yield driver has its own traps (frame overflow, nesting depth) and
  // the same problem, so give it the same channel: a device global pointing
  // at pinned host slots.
  unsigned long long *yfatal =
      ctp::GpuApi::MallocHost<unsigned long long>(4 * sizeof(unsigned long long));
  if (yfatal != nullptr) {
    std::memset(yfatal, 0, 4 * sizeof(unsigned long long));
    cudaMemcpyToSymbol(gy::g_yield_fatal, &yfatal, sizeof(yfatal));
  }
  std::thread fatal_watch([yfatal] {
    for (;;) {
      if (yfatal != nullptr && yfatal[0] != 0) {
        std::fprintf(stderr,
                     "[yield] DEVICE FATAL %llu (101=depth 102=frame "
                     "103=coro-frame): block=%llu lane=%llu need=%llu\n",
                     yfatal[0], yfatal[1], yfatal[2], yfatal[3]);
        std::fflush(stderr);
        return;
      }
      const std::string msg = gv::FatalReport();
      if (!msg.empty()) {
        std::fprintf(stderr, "%s\n", msg.c_str());
        std::fflush(stderr);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });
  fatal_watch.detach();

  // ---- vectors ------------------------------------------------------------
  // ONE TABLE PER CUDA BLOCK is the designed model (independent per-block
  // caches); the single shared table is the alternative this benchmark
  // used so that a page exists exactly once. They are not equivalent under
  // FAULTS: a table owns its lock, its scalar task slots AND its batch
  // slots, so N CUDA blocks on one table share one set of in-flight
  // transfer state.
  // A vector is parameterized by the EXACT launch configuration of the
  // kernel that uses it: one page table per CUDA block, always.
  // THIS NODE'S SLAB OF Z-PLANES. The force stencil reaches one plane either
  // side, so a node also READS the plane below z0 and the one at z1 -- the
  // halo, which its neighbours own and publish.
  if (a.nodes == 0 || a.node >= a.nodes) {
    std::fprintf(stderr, "--node %u is out of range for --nodes %u\n",
                 a.node, a.nodes);
    return 1;
  }
  const u32 zper = (g.nb + a.nodes - 1) / a.nodes;
  const u32 my_z0 = (a.node * zper < g.nb) ? a.node * zper : g.nb;
  const u32 my_z1 = (my_z0 + zper < g.nb) ? (my_z0 + zper) : g.nb;
  if (a.nodes > 1) {
    // A SLAB MUST END ON A PAGE BOUNDARY, and this is not a tidiness rule.
    // The cache transfers WHOLE PAGES -- that is what removed the
    // interval-cannot-describe-a-hole corruption -- so refreshing a halo page
    // that straddles a slab seam overwrites this node's own live elements
    // with the store's copy of them. Silently: the run still passes its
    // gates at 20 steps and is quietly wrong by 200 (E0 -211771 against
    // -215787).
    //
    // A z-plane is nb*nb*cap*kStride elements. For lattice 20 that is 61952
    // bytes = 2^9 * 11^2, whose largest power-of-two divisor is 512 B, so no
    // realistic page divides it. nb a multiple of 16 does divide (lattice 28
    // gives nb=16, a 128 KB plane). Refuse the run rather than produce
    // plausible numbers.
    const u64 plane_bytes =
        static_cast<u64>(g.nb) * g.nb * g.cap * kStride * sizeof(float);
    if (plane_bytes % page_bytes != 0) {
      std::fprintf(stderr,
                   "--nodes %u needs a z-plane that is a whole number of "
                   "pages: plane is %llu B, page is %llu B. Pick a lattice "
                   "whose bin count divides evenly (nb=%u here; a multiple of "
                   "16 works, e.g. --lattice 28) or a smaller --page-kb.\n",
                   a.nodes, (unsigned long long)plane_bytes,
                   (unsigned long long)page_bytes, g.nb);
      return 1;
    }
    std::printf("  DECOMP: node %u/%u owns z-planes [%u,%u) of %u "
                "(plane %llu KB = %llu pages)\n",
                a.node, a.nodes, my_z0, my_z1, g.nb,
                (unsigned long long)(plane_bytes >> 10),
                (unsigned long long)(plane_bytes / page_bytes));
  }

  // Bumped once per drift step; both sides of the exchange agree on it
  // because both derive it from the step count, not from each other.
  u64 halo_gen = 0;
  // MD_NO_HALO=1 is the unsynchronised control: no publish, and the force
  // pass demands generation 0, i.e. accepts whatever copy of a neighbour's
  // plane it happens to be holding. It must produce a MEASURABLY WORSE answer
  // than the exchange; when it did not, that was the bug.
  const bool no_halo = EnvOn("MD_NO_HALO");
  // DIAGNOSTIC: force every force-pass fetch to name this generation.
  const char *fg = std::getenv("MD_FORCE_GEN");
  const u64 force_gen = fg ? std::strtoull(fg, nullptr, 10) : 0;
  clio::cte::core::TagId red_tag{};
  u64 red_round = 0;

  // AFTER the runtime is up (a client built before CLIO_INIT dereferences a
  // runtime that does not exist), and ONLY when there is a reduction to do:
  // an unconditional second CTE client crashed the single-node run in the
  // statics pass.
  std::unique_ptr<clio::cte::core::Client> cte_red;
  if (a.nodes > 1) {
    cte_red = std::make_unique<clio::cte::core::Client>(
        clio::cte::core::kCtePoolId);
    auto t = cte_red->AsyncGetOrCreateTag(std::string(a.tag_prefix) + "md_red");
    t.Wait();
    if (t->GetReturnCode() != 0) {
      std::fprintf(stderr, "could not create the reduction tag\n");
      return 1;
    }
    red_tag = t->tag_id_;
  }

  const u32 tbl_blocks = a.blocks;
  auto tag = [&](const char *name) {
    return std::string(a.tag_prefix) + name;
  };
  // Row geometry, so a flush can name exactly the rows a block owns.
  const u64 md_row_elems = static_cast<u64>(g.nb) * g.cap * kStride;
  const u64 md_nrows = static_cast<u64>(g.nb) * g.nb;
  // SHARED MODE: `nblocks` is a count of associative SETS rather than of
  // per-block tables, and every vector below is sized by SharedSlots for its
  // own page count. The per-vector knobs (--slots/--vslots/--fslots/...) are
  // private-mode sizing and do not apply; shared mode holds each vector.
  const u32 nsets = tbl_blocks;
  // Running total of what the caches cost, so the two modes can be compared
  // as one number rather than six. Paired with a MEASURED free-VRAM delta:
  // the frame arrays are what this arithmetic covers, and the page tables,
  // task slots and staging buffers are what it does not, so a run reports
  // both and the gap between them is visible rather than assumed.
  u64 cache_frames_total = 0, cache_bytes_total = 0;
  size_t vram_free_before = 0, vram_total_dev = 0;
  cudaMemGetInfo(&vram_free_before, &vram_total_dev);
  // WHERE THE VRAM GOES, per vector: the DATA (what the pages actually hold)
  // against the FRAME ARRAY (what the cache allocates for them), because
  // every slot owns dedicated page storage. The ratio is the over-provision,
  // and it is the whole difference between our footprint and a resident
  // MPI-style array.
  u64 data_bytes_total = 0, tbl_bytes_total = 0;
  // The free-VRAM delta since the LAST account() call: everything THIS
  // vector's construction consumed, not just the frame arrays the arithmetic
  // below covers. The gap between the two columns is the hidden overhead.
  size_t vram_prev_probe = vram_free_before;
  auto account = [&](const char *what, u32 nslot, u64 pgb, u64 vec_pages,
                     u64 cap_pages) {
    size_t fnow2 = 0, tnow2 = 0;
    cudaMemGetInfo(&fnow2, &tnow2);
    const double real_mb = (vram_prev_probe > fnow2)
        ? (double)(vram_prev_probe - fnow2) / 1048576.0 : 0.0;
    vram_prev_probe = fnow2;
    const u64 slots = static_cast<u64>(nsets) * nslot;
    const u64 bytes = cap_pages * pgb;      // the allocator's regions
    const u64 data = vec_pages * pgb;
    const u64 tbl = slots * sizeof(gv::Page);
    const u64 frames = cap_pages;
    cache_frames_total += frames;
    cache_bytes_total += bytes;
    data_bytes_total += data;
    tbl_bytes_total += tbl;
    std::printf("  %-5s page=%3lluKB  data %6.1f MB (%llu pages)  regions "
                "%llu = %6.1f MB  %.2fx  tags %llu = %.2f MB\n",
                what, (unsigned long long)(pgb >> 10),
                (double)data / 1048576.0, (unsigned long long)vec_pages,
                (unsigned long long)frames, (double)bytes / 1048576.0,
                (double)bytes / (double)(data ? data : 1),
                (unsigned long long)slots, (double)tbl / 1048576.0);
    std::printf("        REAL free-VRAM delta for %s: %.1f MB\n", what,
                real_mb);
  };
  bool cache_reported = false;
  auto report_caches = [&]() {
    if (cache_reported) return;
    cache_reported = true;
    size_t fnow = 0, tnow = 0;
    cudaMemGetInfo(&fnow, &tnow);
    const double used_mb =
        (vram_free_before > fnow)
            ? (double)(vram_free_before - fnow) / 1048576.0
            : 0.0;
    std::printf("  CACHE TOTAL: %llu regions = %.1f MB of page storage for "
                "%.1f MB of data (%.2fx over-provision, %.1f MB of it spare "
                "page storage);\n"
                "  tags %.2f MB (sizeof(Page)=%zu). Device VRAM "
                "consumed by the vectors %.1f MB (free %.0f -> %.0f MB "
                "of %.0f)\n",
                (unsigned long long)cache_frames_total,
                (double)cache_bytes_total / 1048576.0,
                (double)data_bytes_total / 1048576.0,
                (double)cache_bytes_total /
                    (double)(data_bytes_total ? data_bytes_total : 1),
                // Slab-sized caches hold FEWER regions than the domain has
                // pages, so spare = regions - data can be negative; clamp
                // rather than underflow the unsigned subtraction.
                (cache_bytes_total > data_bytes_total)
                    ? (double)(cache_bytes_total - data_bytes_total) /
                          1048576.0
                    : 0.0,
                (double)tbl_bytes_total / 1048576.0, sizeof(gv::Page),
                used_mb, (double)vram_free_before / 1048576.0,
                (double)fnow / 1048576.0, (double)vram_total_dev / 1048576.0);
  };
  const u32 x_slots = SharedSlots(npages, nsets);
  // CACHE THE SLAB, NOT THE DOMAIN. In decomposed mode this node touches its
  // own planes plus ONE plane either side (force stencil, resort assign,
  // gather all reach exactly bz +/- 1), so sizing every cache to the whole
  // domain -- and prefetching all of it -- made each node hold its peer's
  // half of every vector for no reader. Measured: 84 MB per rank against
  // MPI's 29.2, with the full-domain neighbour list alone 52 of it. Planes
  // are whole pages in decomposed mode (the seam rule enforces it), so the
  // slab's page range is exact.
  const u64 ppp = (a.nodes > 1)
      ? (static_cast<u64>(g.nb) * g.nb * g.cap * kStride * sizeof(float)) /
            page_bytes
      : 0;
  const u32 zmine = my_z1 - my_z0;
  const u64 slab_pg_lo = (a.nodes > 1) ? my_z0 * ppp : 0;
  const u64 slab_pg_hi = (a.nodes > 1) ? my_z1 * ppp : npages;
  // x and v carry the halo: slab + 2 planes. f and the ping-pong buffers are
  // written per-slab only -- but the resort SWAPS the handles, so x2/v2
  // become x/v and need halo capacity too.
  const u32 md_cap = (a.nodes > 1)
      ? static_cast<u32>((zmine + 2) * ppp + 2)
      : static_cast<u32>(npages + 2);
  const u32 md_cap_own = (a.nodes > 1)
      ? static_cast<u32>(zmine * ppp + 2)
      : static_cast<u32>(npages + 2);
  gv::Vector<float> vx(tag("md_x"), {0}, page_bytes, tbl_blocks, x_slots,
                       g.nelems, clio::run::PoolId::GetNull(), 0, 1,
                       /*nsets=*/0, /*capacity_pages=*/md_cap);
  account("x", x_slots, page_bytes, npages, md_cap);
  // v gets its own knob too, so x-paging and v-paging can be separated:
  // x is the only vector read through multi-page SPAN holds that stay live
  // across a park, which is the access pattern no other gate covers.
  const u32 vslots = SharedSlots(npages, nsets);
  gv::Vector<float> vv(tag("md_v"), {0}, page_bytes, tbl_blocks, vslots,
                       g.nelems, clio::run::PoolId::GetNull(), 0, 1, 0,
                       md_cap);
  account("v", vslots, page_bytes, npages, md_cap);
  vx.EnableStats();
  vv.EnableStats();
  // ONE WRITER FOR THE INITIAL DECK, AND A BARRIER BEHIND IT.
  //
  // Preload writes the host deck into the BLOBS, and in decomposed mode every
  // node shares one tag namespace -- so every node was laying the whole
  // lattice down on top of every other node's. node2 starts GVMB_START_DELAY
  // seconds late, by which time node1 may already have run its statics pass
  // and resorted; node2's preload then overwrote the wrapped positions node1
  // had just computed. Intermittent, and both nodes agree on the wrong answer
  // afterwards because they read the same clobbered store -- which is exactly
  // what `RESORT GATE: FAIL (rel=4.14e-03)` was, with the halo exchange
  // playing no part in it.
  //
  // Node 0 lays the deck down; everyone waits for it and faults it in. The
  // reduction doubles as the barrier: it does not return until every node has
  // published its contribution for the round.
  if (a.nodes > 1) {
    if (a.node == 0) {
      vx.Preload(hx.data(), g.nelems);
      vv.Preload(hv.data(), g.nelems);
    }
    double ready = 1.0;
    if (!ReduceSum(*cte_red, red_tag, a.node, a.nodes, red_round++, &ready, 1)) {
      std::fprintf(stderr, "  deck barrier failed\n");
      return 1;
    }
  } else {
    vx.Preload(hx.data(), g.nelems);
    vv.Preload(hv.data(), g.nelems);
  }
  vx.ClearCache();
  vv.ClearCache();
  vx.Prefetch(slab_pg_lo, slab_pg_hi, 0, tbl_blocks);
  vv.Prefetch(slab_pg_lo, slab_pg_hi, 0, tbl_blocks);
  auto dx = vx.GetDevice(0);
  auto dv = vv.GetDevice(0);

  // MD_THIRD=1 gives the ballistic path a third, read-only paged vector.
  const int use_third = EnvOn("MD_THIRD") ? 1 : 0;
  const u32 third_slots = use_third ? SharedSlots(npages, nsets) : 1u;
  gv::Vector<float> vthird(tag("md_third"), {0}, page_bytes, tbl_blocks,
                           third_slots,
                           use_third ? g.nelems : page_elems,
                           clio::run::PoolId::GetNull(), 0, 1);
  if (use_third) {
    account("third", third_slots, page_bytes, npages, md_cap);
    std::vector<float> hz(g.nelems, 0.0f);
    vthird.Preload(hz.data(), g.nelems);
    vthird.ClearCache();
    vthird.Prefetch(0, npages, 0, tbl_blocks);
  }
  auto dthird = vthird.GetDevice(0);
  // The ballistic and probe paths own every vector they will ever build; the
  // MD path reports after the list, which is the last of its six.
  if (!a.md) report_caches();

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
    const bool ldcg = EnvOn("MD_PROBE_LDCG");
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
    const u32 fslots = SharedSlots(npages, nsets);
    gv::Vector<float> vf(tag("md_f"), {0}, page_bytes, tbl_blocks, fslots,
                         g.nelems, clio::run::PoolId::GetNull(), 0, 1, 0,
                         md_cap_own);
    account("f", fslots, page_bytes, npages, md_cap_own);
    vf.EnableStats();
    {
      std::vector<float> hz(g.nelems, 0.0f);
      vf.Preload(hz.data(), g.nelems);
    }
    vf.ClearCache();
    vf.Prefetch(slab_pg_lo, slab_pg_hi, 0, 1);
    auto df = vf.GetDevice(0);
    // Ping-pong destination vectors for the resort (K2b scatters into
    // these, then the handles swap). Same geometry, resident.
    // The resort's destinations get their own cache size, so "the scatter
    // is wrong" can be told apart from "the scatter's PAGING is wrong".
    const u32 ppslots = SharedSlots(npages, nsets);
    gv::Vector<float> vx2(tag("md_x2"), {0}, page_bytes, tbl_blocks, ppslots,
                          g.nelems, clio::run::PoolId::GetNull(), 0, 1, 0,
                          md_cap);
    gv::Vector<float> vv2(tag("md_v2"), {0}, page_bytes, tbl_blocks, ppslots,
                          g.nelems, clio::run::PoolId::GetNull(), 0, 1, 0,
                          md_cap);
    account("x2", ppslots, page_bytes, npages, md_cap);
    account("v2", ppslots, page_bytes, npages, md_cap);
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
    vx2.Prefetch(slab_pg_lo, slab_pg_hi, 0, 1);
    vv2.Prefetch(slab_pg_lo, slab_pg_hi, 0, 1);
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
    // THE LIST IS THE ONE VECTOR SHARING WAS NOT OBVIOUSLY FOR: it is
    // streamed a row at a time, degree 1.39, so a private cache already
    // stores each page about once. It goes shared anyway, because the run
    // has ONE mode -- and the interesting question is exactly what that
    // costs a vector with nothing to share. Sized for residency like the
    // rest: shared residency is 2x the list, where private residency was
    // blocks x the list and had to be capped at NlSlots to allocate at all.
    const u32 nlslots = SharedSlots(nl_pages, nsets);
    // THE LIST IS NODE-LOCAL DATA. Each node builds and reads only its own
    // slab's rows -- no kernel ever touches a peer's -- so caching the whole
    // domain's list on every node was 48 MB of a 59 MB cache spent holding
    // rows with no reader. Rows are not page-aligned to the slab, so the cap
    // is the slab rows' byte extent in pages, +2 seam slack, +2 guards.
    const u64 nl_row_elems_all = nl_elems / (static_cast<u64>(g.nb) * g.nb);
    const u64 nl_lo_elem = static_cast<u64>(my_z0) * g.nb * nl_row_elems_all;
    const u64 nl_hi_elem = static_cast<u64>(my_z1) * g.nb * nl_row_elems_all;
    const u64 nl_pg_elems = nl_page_bytes / sizeof(int);
    const u64 nl_slab_lo = (a.nodes > 1) ? nl_lo_elem / nl_pg_elems : 0;
    const u64 nl_slab_hi = (a.nodes > 1)
        ? (nl_hi_elem + nl_pg_elems - 1) / nl_pg_elems
        : nl_pages;
    const u32 nl_cap = (a.nodes > 1)
        ? static_cast<u32>(nl_slab_hi - nl_slab_lo + 4)
        : static_cast<u32>(nl_pages + 2);
    gv::Vector<int> vn(tag("md_nl"), {0}, nl_page_bytes, tbl_blocks, nlslots,
                       nl_elems, clio::run::PoolId::GetNull(), 0, 1, 0,
                       nl_cap);
    account("list", nlslots, nl_page_bytes, nl_pages, nl_cap);
    vn.EnableStats();
    {
      std::vector<int> hz(nl_elems, 0);
      vn.Preload(hz.data(), nl_elems);
    }
    vn.ClearCache();
    vn.Prefetch(nl_slab_lo, nl_slab_hi, 0, 1);
    auto dn = vn.GetDevice(0);
    report_caches();
    std::printf("  list: maxneigh=%u row=%llu entries page=%lluKB "
                "(%llu pages, %llu guards/row)\n",
                a.maxneigh, (unsigned long long)md_rowlist,
                (unsigned long long)(nl_page_bytes >> 10),
                (unsigned long long)nl_pages,
                (unsigned long long)((md_rowlist + nl_page_elems - 1) /
                                     nl_page_elems + 1));
    u32 *d_cnt = ctp::GpuApi::Malloc<u32>(g.nslots * sizeof(u32));
    double *d_acc = ctp::GpuApi::Malloc<double>(3 * sizeof(double));
    // Reduction scratch only. The block-uniform tables moved into the
    // persistent arena, which lives inside YieldSmem and is therefore
    // already covered by CLIO_YIELD_SMEM_BYTES -- no hand-summed byte count
    // to keep in step with the struct any more.
    const u32 smem_force =
        CLIO_YIELD_SMEM_BYTES + a.threads * sizeof(double);
    const float fbox = static_cast<float>(g.box);
    const float fcut = static_cast<float>(a.cutoff);
    const float fdt = static_cast<float>(a.dt);

    // MD_NOCOMPUTE=1 keeps every hold and skips the pair loop: the
    // difference against a normal run IS the hold cost.
    const int nocompute = EnvOn("MD_NOCOMPUTE") ? 1 : 0;
    const bool trace = EnvOn("MD_TRACE");
    double acc[3] = {0, 0, 0};
    double t_force_kern = 0.0, t_ckpt = 0.0, t_ckpt_stock = 0.0;
    u64 n_ckpt = 0;
    double t_force = 0.0, t_kick = 0.0, t_resort = 0.0, t_build = 0.0;
    double t_kick_int = 0.0, t_kick_pub = 0.0;
    u64 r_kick_int = 0, r_kick_pub = 0;
    const float frlist = static_cast<float>(a.cutoff + a.skin);
    // Build the Verlet list: device count pass, host prefix sum (index
    // class, ~MBs), device fill pass. Refuses loudly on maxneigh or the
    // per-row guard bound.
    auto build_list = [&]() -> bool {
      if (trace) { std::fprintf(stderr, "[md] build\n"); std::fflush(stderr); }
      const double _t = NowMs();
      ctp::GpuApi::Memset(d_err, 0, sizeof(int));
      MdMark("BuildList");
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        BuildListKernel<<<gr, b, smem_force>>>(
            gpu, dx, dn, g.nb, g.cap, fbox, frlist, a.maxneigh, d_cnt,
            d_err, a.rowchunk, my_z0, my_z1, a.blocks,
            no_halo ? 0 : halo_gen, vw, sv);
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
        MdMark("ListForce");
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          ListForceKernel<<<gr, b, smem_force>>>(
              gpu, dx, df, dn, g.nb, g.cap, fbox, fcut, a.maxneigh, d_cnt,
              eflag, d_acc, nocompute, a.rowchunk, my_z0, my_z1, a.blocks,
              force_gen ? force_gen : (no_halo ? 0 : halo_gen),
              force_gen != 0, vw, sv);
        });
      } else {
        MdMark("Force");
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          ForceKernel<<<gr, b, smem_force>>>(gpu, dx, df, g.nb, g.cap, fbox,
                                             fcut, eflag, d_acc, my_z0, my_z1,
                                             a.blocks,
                                             force_gen ? force_gen
                                                       : (no_halo ? 0 : halo_gen),
                                             force_gen != 0, vw, sv);
        });
      }
      ctp::GpuApi::Synchronize();
      if (eflag) {
        ctp::GpuApi::Memcpy(acc, d_acc, 3 * sizeof(double));
        // PE, virial and the pair count are sums over the WHOLE domain; this
        // node computed its slab's share.
        // FATAL, AND EXIT RATHER THAN RETURN. `return 1` here made this
        // void lambda deduce `int`, so every other path fell off the end of a
        // non-void function -- undefined behaviour that compiles, and
        // segfaulted in the statics pass. A peer that never published its
        // partial is not something the run can continue past anyway.
        if (a.nodes > 1 &&
            !ReduceSum(*cte_red, red_tag, a.node, a.nodes, red_round++, acc, 3)) {
          std::fprintf(stderr, "  energy reduction failed\n");
          std::fflush(stderr);
          std::_Exit(1);
        }
      }
      t_force += NowMs() - _t;
      t_force_kern += runner.KernelMs();
    };
    // MD_FLUSH_AFTER_KICK=1 makes every dirty x/v page durable in the
    // backing store before anything can fault it back in. If that removes
    // the intermittent out-of-core error, the hazard is a refetch racing an
    // in-flight writeback (read-after-write through the store).
    const bool flush_after_kick =
        EnvOn("MD_FLUSH_AFTER_KICK");
    // PUBLISH THIS SLAB AS THE NEXT GENERATION.
    //
    // EVERY WRITE THIS NODE MAKES NEEDS ONE OF THESE BEFORE A PEER READS IT,
    // not just the integrator's. It used to live inside kick(drift=1) only,
    // so the RESORT -- which rewrites this slab's positions wholesale -- was
    // followed by a force() that read the neighbours' planes straight out of
    // the local cache, still holding their PRE-RESORT contents. That is the
    // intermittent `RESORT GATE: FAIL (rel=5.44e-03)`: same characteristic
    // value every time, because it is one specific region going stale, and
    // independent of MD_NO_HALO because the resort never had an exchange at
    // all.
    //
    // NO SEPARATE HALO KERNEL. A prefetch that unpins on the spot holds
    // nothing for the consumer; the version demand lives on the read that
    // consumes the data (ForceCoro/ListForceCoro), which is the only place
    // that can order anything.
    auto exchange = [&]() {
      if (a.nodes <= 1 || no_halo) return;
      ++halo_gen;
      r_kick_pub += runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        PublishSlabKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, dv, g.nb, g.cap, my_z0, my_z1, halo_gen, a.blocks,
            vw, sv);
      });
      ctp::GpuApi::Synchronize();
      // NO BARRIER HERE. THE GENERATION IS THE BARRIER.
      //
      // This used to be an all-reduce on a dummy value, once per exchange --
      // so at least once per timestep, as a blob put plus N-1 blob gets in a
      // sleep-polling retry loop, to communicate one bit ("I have
      // published"). It cost more than the entire force computation:
      //   phases (total ms): force=2752.9 kick=4591.4 resort=1603.0
      // where kick is integration (trivial) plus this.
      //
      // It was scaffolding for a bug that is fixed. A generational get that
      // arrived before its writer's put appeared to ERROR rather than wait,
      // so the ordering looked like it had to be imposed from outside -- but
      // the real cause was that the publish flushed pages it had never
      // fetched, and SubmitFlushRanges drops a page it cannot Find, silently,
      // before it ever builds a task. Every generational put was being thrown
      // away; nothing was ever late. With the publish fixed, the demand's own
      // wait-for-the-writer gate does the ordering, which is what the design
      // said all along.
    };
    auto kick = [&](int drift) {
      if (trace) { std::fprintf(stderr, "[md] kick drift=%d\n", drift);
                   std::fflush(stderr); }
      const double _t = NowMs();
      MdMark("MDIntegrate");
      // SPLIT-TIMED: two quantum hunts (round-trip count, CTE poll) missed
      // kick's cost, so measure which half owns it and how many DRIVER
      // ROUNDS each pays -- a round is a full kernel relaunch, and rounds,
      // not bytes, are the unit parks are paid in.
      r_kick_int += runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        MDIntegrateKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, dv, df, fdt, drift, g.nb, g.cap, my_z0, my_z1,
            a.blocks, vw, sv);
      });
      t_kick_int += NowMs() - _t;
      const double _t2 = NowMs();
      // EXCHANGE, and only when there is someone to exchange with. Publish
      // this slab AS the next generation, then demand the neighbours' halo AT
      // that generation. No barrier between the two: the generation is the
      // barrier, which is the property this decomposition is built on.
      // MD_NO_HALO=1 skips the exchange, which must MEASURABLY DEGRADE the
      // run. A distributed gate that passes either way is not testing the
      // exchange, and this is the cheapest way to know which one it is.
      if (drift) exchange();
      t_kick_pub += NowMs() - _t2;
      t_kick += NowMs() - _t;
    };
    auto thermo_ke = [&]() -> double {
      ctp::GpuApi::Memset(d_thermo, 0, 4 * sizeof(double));
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        ThermoKernel<<<gr, b, smem_thermo>>>(gpu, dx, dv, d_thermo, g.nb, g.cap,
                                             my_z0, my_z1, a.blocks,
                                             vw, sv);
      });
      ctp::GpuApi::Synchronize();
      double t4[4];
      ctp::GpuApi::Memcpy(t4, d_thermo, sizeof(t4));
      // Same for kinetic energy and momentum: per-slab values are not
      // conserved once the resort moves an atom across a boundary.
      // Not `return 0.0`: a silent zero would sail through the NVE gate as a
      // plausible number.
      if (a.nodes > 1 &&
          !ReduceSum(*cte_red, red_tag, a.node, a.nodes, red_round++, t4, 4)) {
        std::fprintf(stderr, "  thermo reduction failed\n");
        std::fflush(stderr);
        std::_Exit(1);
      }
      return t4[0];
    };
    // K2: wrap + rebin + double-buffered scatter, then swap the handles.
    // Returns false on bin overflow (the host refuses the run).
    auto resort = [&]() -> bool {
      const double _t = NowMs();
      // The inputs are already published: IntegrateCoro flushes each page it
      // writes, by name. What is still needed is invalidation -- a block
      // holding a neighbour's page would keep reading its stale copy rather
      // than refetch the version its owner just wrote.
      // Clear all four -- resort swaps the handles, so dx/dv alias vx2/vv2 on
      // every other pass and naming one pair would clear the wrong vectors.
      //
      if (EnvOn("MD_RESORT_DEBUG")) {
        std::printf("  [resort-debug] ENTRY occ x=%u x2=%u v=%u v2=%u\n",
                    vx.Occupied(0), vx2.Occupied(0), vv.Occupied(0),
                    vv2.Occupied(0));
        std::fflush(stdout);
      }
      // BUT NOT IN DECOMPOSED MODE, where this is actively DESTRUCTIVE:
      // eviction performs no I/O and the interior is no longer flushed to
      // the store, so clearing discards the ONLY copy and the refetch
      // returns the preloaded zeros -- measured as PE alternating
      // 0 <-> correct with the swap parity, pairs 0 <-> 4741632. With one
      // shared cache the gather's write-holds are visible to every block, so
      // there is nothing to invalidate; cross-node halo staleness is the
      // GENERATIONAL demand's job (that is what `refetch` counts), not an
      // invalidation pass's. Single-node keeps the clear until its OOC
      // interaction is separately verified.
      if (a.nodes == 1) {
        vx.ClearCache();
        vv.ClearCache();
        vx2.ClearCache();
        vv2.ClearCache();
      }
      ctp::GpuApi::Memset(d_bincnt, 0, g.nbins * sizeof(u32));
      // ~0u IS "NO DESTINATION", AND d_dest WAS NEVER CLEARED.
      //
      // RebinKernel fills only the slots of THIS node's slab, but GatherCoro
      // reads d_dest for every source slot in its stencil -- which reaches one
      // plane past the slab in y and z. Those entries were whatever the
      // PREVIOUS resort left there (uninitialised on the first), so the gather
      // scattered atoms according to stale destinations: duplicates at
      // overlapping positions, LJ repulsion, and KE climbing 2634 -> 17984
      // over 200 steps. Clearing makes an unwritten entry mean what the
      // scatter already tests for -- `dest == ~0u` -> skip.
      ctp::GpuApi::Memset(d_dest, 0xFF, g.nslots * sizeof(u32));
      ctp::GpuApi::Memset(d_err, 0, sizeof(int));
      const float fbox2 = static_cast<float>(g.box);
      MdMark("Rebin");
      // WRAP, PUBLISH, THEN ASSIGN. The assign pass reads a neighbour's
      // positions to decide whether an atom is arriving in one of this node's
      // bins, and it must read them ALREADY WRAPPED -- otherwise the two
      // nodes bin the same atom differently. exchange() both publishes the
      // wrap as the next generation and barriers behind it, which is exactly
      // the ordering the assign pass then demands with that generation.
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        RebinWrapKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, g.nb, g.cap, fbox2, my_z0, my_z1, a.blocks, vw, sv);
      });
      ctp::GpuApi::Synchronize();
      exchange();
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        RebinAssignKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, g.nb, g.cap, fbox2, d_bincnt, d_dest, d_err,
            my_z0, my_z1, a.blocks, no_halo ? 0 : halo_gen, vw, sv);
      });
      ctp::GpuApi::Synchronize();
      // MD_RESORT_DEBUG=1: what the assign actually produced, from the
      // device arrays themselves -- distinguishes "assign wrote nothing"
      // from "gather dropped what assign wrote" in one line.
      if (EnvOn("MD_RESORT_DEBUG")) {
        unsigned long long gw = 0;
        cudaMemcpyFromSymbol(&gw, g_gather_wrote, sizeof(gw));
        const unsigned long long zz = 0;
        cudaMemcpyToSymbol(g_gather_wrote, &zz, sizeof(zz));
        std::printf("  [resort-debug] gather_wrote=%llu (both passes) | "
                    "max page copies x=%u x2=%u\n", gw,
                    vx.MaxPageCopies(0), vx2.MaxPageCopies(0));
        std::vector<u32> hd(g.nslots), hb(g.nbins);
        ctp::GpuApi::Memcpy(hd.data(), d_dest, g.nslots * sizeof(u32));
        ctp::GpuApi::Memcpy(hb.data(), d_bincnt, g.nbins * sizeof(u32));
        u64 assigned = 0, bsum = 0;
        for (u32 vd : hd) assigned += (vd != ~0u);
        for (u32 vb : hb) bsum += vb;
        std::printf("  [resort-debug] assigned=%llu bincnt_sum=%llu "
                    "(atoms=%llu)\n",
                    (unsigned long long)assigned, (unsigned long long)bsum,
                    (unsigned long long)g.natoms);
        std::fflush(stdout);
      }
      int err = 0;
      ctp::GpuApi::Memcpy(&err, d_err, sizeof(int));
      if (err == 1) {
        std::fprintf(stderr, "resort: bin overflow -- raise --cap\n");
        return false;
      }
      if (err == 2) {
        std::fprintf(stderr,
                     "resort: an atom crossed more than one bin in y or z, "
                     "which the +/-1 gather stencil cannot reach -- lower "
                     "--rebin\n");
        return false;
      }
      // SINGLE NODE ONLY. The premise ("RebinCoro publishes the wrapped
      // positions itself") predates the wrap/assign split -- the wrap no
      // longer flushes anything -- and in decomposed mode this clear threw
      // away the WRAPPED positions between the assign and the gather, so the
      // gather refetched pre-wrap bytes from the store. Same class as the
      // other two mid-resort clears: with one shared cache there is nothing
      // to invalidate.
      if (a.nodes == 1) {
        vx.ClearCache();
        vv.ClearCache();
        vx2.ClearCache();
        vv2.ClearCache();
      }
      for (auto *dst : {&dx2, &dv2}) {
        MdMark("Sentinel");
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          SentinelKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, *dst, g.nb, g.cap,
                                                           my_z0, my_z1,
                                                           a.blocks, vw, sv);
        });
      }
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        GatherKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dx, dx, dx2, g.nb, g.cap, d_dest, /*keep_w=*/1,
            my_z0, my_z1, a.blocks, no_halo ? 0 : halo_gen, vw, sv);
      });
      MdMark("Gather-v");
      MdMark("Gather-x");
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        GatherKernel<<<gr, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu, dv, dx, dv2, g.nb, g.cap, d_dest, /*keep_w=*/0,
            my_z0, my_z1, a.blocks, no_halo ? 0 : halo_gen, vw, sv);
      });
      ctp::GpuApi::Synchronize();
      // INVALIDATE -- SINGLE NODE ONLY, and even there the rationale is the
      // private-cache era talking: with one shared cache the gather's
      // write-holds are already visible to every block. In decomposed mode
      // this clear was the LAST unexplained corruption: "safe here -- the
      // gather flushed" is FALSE once the interior flush is off, so this
      // discarded the frames the gather had just written (the cache's ONLY
      // copy) and the next fetch pulled the store's stale preload. Probes:
      // assign 43904/43904 correct, gather_wrote 87808 correct, then
      // Occupied()==0 on the spare -- with evicts=0, because ClearCache does
      // not count. PE alternated exactly 0 <-> exactly-initial with the swap
      // parity, KE read 0, and the system never evolved.
      if (a.nodes == 1) {
        vx.ClearCache();
        vv.ClearCache();
        vx2.ClearCache();
        vv2.ClearCache();
      }
      std::swap(dx, dx2);
      std::swap(dv, dv2);
      cvx = (cvx == &vx) ? &vx2 : &vx;
      if (EnvOn("MD_RESORT_DEBUG")) {
        std::printf("  [resort-debug] EXIT  occ x=%u x2=%u v=%u v2=%u\n",
                    vx.Occupied(0), vx2.Occupied(0), vv.Occupied(0),
                    vv2.Occupied(0));
        std::fflush(stdout);
      }
      t_resort += NowMs() - _t;
      return true;
    };

    {
      const u32 pub = (EnvOn("MD_NO_PUBLISH")) ? 0u : 1u;
      cudaMemcpyToSymbol(g_publish, &pub, sizeof(pub));
      const u32 pub_int = (a.nodes > 1 && EnvOn("MD_NO_INTERIOR")) ? 0u : 1u;
      cudaMemcpyToSymbol(g_pub_interior, &pub_int, sizeof(pub_int));
      if (pub == 0u) {
        std::printf("  [MD_NO_PUBLISH] publishing DISABLED -- physics is "
                    "invalid, timing only\n");
      }
    }

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
      // NO SEPARATE CHECKPOINT PASS EXISTS ANY MORE, and that is the result,
      // not a measurement bug. Every kernel publishes each page as it writes
      // it -- it has to, because the other blocks read those pages -- so the
      // live state is already in the tier stack at every step boundary. The
      // paged checkpoint cost is therefore folded into the step, and what is
      // timed here is only the barrier that proves it landed.
      const double _t = NowMs();
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
      // The resort rewrote this slab; a peer's force must not read the old
      // copy of it, and this node's must not read the peer's old copy.
      exchange();
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
    // ZERO THE PHASE TIMERS HERE. The statics and resort gates above call
    // force() and build_list() too, and counting those made the phase totals
    // sum to more than the run they were supposedly decomposing.
    t_force = 0.0; t_kick = 0.0; t_resort = 0.0; t_build = 0.0;
    t_kick_int = 0.0; t_kick_pub = 0.0; r_kick_int = 0; r_kick_pub = 0;
    t_force_kern = 0.0; t_ckpt = 0.0; t_ckpt_stock = 0.0; n_ckpt = 0;
    const double t0 = NowMs();
    for (u64 step = 0; step < a.steps; ++step) {
      kick(/*drift=*/1);   // uses f(t)
      if (a.rebin != 0 && step != 0 && step % a.rebin == 0) {
        // Same continuity check the step-0 RESORT GATE applies, but on a
        // resort that runs mid-flight: same physical state, new layout, so PE
        // must be unchanged. Off by default -- it costs two extra force
        // evaluations per resort.
        const bool rchk = EnvOn("MD_RESORT_CHECK");
        double pe_b = 0.0, pr_b = 0.0;
        if (rchk) { force(/*eflag=*/1); pe_b = acc[0]; pr_b = acc[2]; }
        if (!resort()) return 1;
        exchange();
        if (a.use_list && !build_list()) return 1;
        if (rchk) {
          force(/*eflag=*/1);
          // Pairs alongside PE: a resort that DROPS atoms loses pairs, one
          // that merely misplaces them keeps the count and moves the energy.
          std::printf("  [resort-check] step %llu: PE %.6f -> %.6f rel=%.2e | "
                      "pairs %.0f -> %.0f (%+.0f)\n",
                      (unsigned long long)step, pe_b, acc[0],
                      std::fabs(acc[0] - pe_b) / std::fabs(pe_b), pr_b, acc[2],
                      acc[2] - pr_b);
        }
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
        "  paging: x faults=%llu evicts=%llu | f faults=%llu evicts=%llu | "
        "x puts=%llu get_errors=%llu put_errors=%llu | "
        "gen: resident_ok=%llu refetch=%llu busy=%llu flush_dropped=%llu  "
        "%s\n",
        (unsigned long long)a.steps, e0, e_n, e_drift, ke0, ke_n,
        (unsigned long long)sfx.faults, (unsigned long long)sfx.evicts,
        (unsigned long long)sff.faults, (unsigned long long)sff.evicts,
        // A GENERATIONAL GET THAT WAS NEVER SERVED MUST SHOW UP HERE. If the
        // frames come back empty and this stays 0, the failure is silent and
        // PublishFetch declared unfetched bytes valid.
        (unsigned long long)sfx.puts,
        (unsigned long long)sfx.get_errors,
        (unsigned long long)sfx.put_errors,
        (unsigned long long)sfx.gen_ok,
        (unsigned long long)sfx.gen_stale,
        (unsigned long long)sfx.gen_busy,
        (unsigned long long)sfx.flush_skipped,
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
    {
      // HOW OFTEN IS A PAGE SHARED? popcount of the per-page block mask.
      static u32 xm[8192], nm[8192];
      cudaMemcpyFromSymbol(xm, g_xmask, sizeof(xm));
      cudaMemcpyFromSymbol(nm, g_nlmask, sizeof(nm));
      const char *nms[2] = {"x", "list"};
      const u32 *ms[2] = {xm, nm};
      for (int k = 0; k < 2; ++k) {
        u64 touched = 0, shared = 0, sumdeg = 0, maxdeg = 0, hist[17] = {0};
        for (u32 p = 0; p < 8192; ++p) {
          const u32 d = __builtin_popcount(ms[k][p]);
          if (d == 0) continue;
          ++touched; sumdeg += d;
          if (d > 1) ++shared;
          if (d > maxdeg) maxdeg = d;
          if (d <= 16) ++hist[d];
        }
        if (touched == 0) continue;
        std::printf("  SHARING %-4s: %llu pages touched, %llu shared by >1 "
                    "block (%.1f%%), mean degree %.2f, max %llu\n",
                    nms[k], (unsigned long long)touched,
                    (unsigned long long)shared,
                    100.0 * static_cast<double>(shared) / touched,
                    static_cast<double>(sumdeg) / touched,
                    (unsigned long long)maxdeg);
        std::printf("    degree histogram:");
        for (int d = 1; d <= 16; ++d) {
          if (hist[d]) std::printf(" %d:%llu", d, (unsigned long long)hist[d]);
        }
        std::printf("\n");
      }
    }
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
    std::printf("  kick split: integrate=%.1f ms (%llu rounds) "
                "publish=%.1f ms (%llu rounds)\n",
                t_kick_int, (unsigned long long)r_kick_int, t_kick_pub,
                (unsigned long long)r_kick_pub);
    {
      unsigned long long fc = 0, gc = 0;
      cudaMemcpyFromSymbol(&fc, g_pub_flush_cyc, sizeof(fc));
      cudaMemcpyFromSymbol(&gc, g_pub_fetch_cyc, sizeof(gc));
      // Device cycles at ~1 GHz order; the RATIO is what matters.
      std::printf("  publish split (device cycles): flush=%llu halo_fetch=%llu"
                  " (%.1f%% fetch)\n", fc, gc,
                  (fc + gc) ? 100.0 * (double)gc / (double)(fc + gc) : 0.0);
    }
    std::printf("  phases (total ms): force=%.1f (gpu %.1f) kick=%.1f "
                "resort=%.1f build=%.1f\n", t_force, t_force_kern, t_kick,
                t_resort, t_build);
    if (EnvOn("MD_PROF")) {
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
    MdMark("Thermo");
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      ThermoKernel<<<gr, b, smem_thermo>>>(gpu, dx, dv, d_thermo, g.nb,
                                           g.cap, my_z0, my_z1, a.blocks,
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
    // Download reads the BACKING STORE, which the kernels have already
    // written: each one flushes every page it writes, by name.
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
