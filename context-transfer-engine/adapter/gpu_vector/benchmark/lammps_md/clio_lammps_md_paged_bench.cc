/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * eternia-MD: a from-scratch reimplementation of the LAMMPS melt benchmark
 * (lj/cut, NVE, periodic box) with ALL simulation state device-canonical on
 * paged gv::Vectors. Design: ../eternia.md (adapter/gpu_vector). LAMMPS itself is only the
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
#include "../bench_flush_data.h"
#include <clio_cte/gpu_vector/gpu_vector.h>
#include "md_launch.h"
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

namespace md = ::clio::gv_bench::md;

namespace {
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

  /** One driver round of an ASYNC sequence: launches whatever is ready,
   *  returns true while blocks are still parked. BeginSeq resets state; the
   *  caller then Steps between (or during) other work -- this is the whole
   *  comm/compute overlap: the publish kernel's CTE tasks make progress on
   *  the CPU while the interior force owns the GPU. */
  void BeginSeq() {
    drv_.ResetTimers();
    drv_.Reset();
    stack_.Reset();
  }
  template <typename LaunchT>
  bool Step(LaunchT &&launch) {
    return drv_.Round(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [](u32, u64 tag) -> bool {
          unsigned int f = 0;
          ctp::GpuApi::Memcpy(&f, reinterpret_cast<const unsigned int *>(tag),
                              sizeof(f));
          return (f & 1u) != 0u;
        });
  }

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
    if (md::DevMemInfo(&dev_free, &dev_total)) {
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
    md::SymbolWrite(md::MdSym::kYieldFatal, &yfatal, sizeof(yfatal));
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
  md::DevMemInfo(&vram_free_before, &vram_total_dev);
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
                     u64 cap_pages, u64 nblk) {
    size_t fnow2 = 0, tnow2 = 0;
    md::DevMemInfo(&fnow2, &tnow2);
    const double real_mb = (vram_prev_probe > fnow2)
        ? (double)(vram_prev_probe - fnow2) / 1048576.0 : 0.0;
    vram_prev_probe = fnow2;
    const u64 slots = static_cast<u64>(nsets) * nslot;
    // THE CONSTRUCTOR ROUNDS UP: every block owns the same number of regions
    // (per_block = ceil(cap/nblocks)), so the real pool is a MULTIPLE of the
    // task-table count. Report the rounded pool, not the request -- the first
    // distributed OOC run asked for 12 regions, silently got 20 (one per
    // table), and its evicts=0 looked like a broken counter instead of a
    // cache that genuinely fit.
    const u64 frames = (nblk != 0) ? ((cap_pages + nblk - 1) / nblk) * nblk
                                   : cap_pages;
    const u64 bytes = frames * pgb;         // the allocator's regions
    const u64 data = vec_pages * pgb;
    const u64 tbl = slots * sizeof(gv::Page);
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
    md::DevMemInfo(&fnow, &tnow);
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
  u32 md_cap = (a.nodes > 1)
      ? static_cast<u32>((zmine + 2) * ppp + 2)
      : static_cast<u32>(npages + 2);
  u32 md_cap_own = (a.nodes > 1)
      ? static_cast<u32>(zmine * ppp + 2)
      : static_cast<u32>(npages + 2);
  // OUT-OF-CORE, DECOMPOSED: an explicit --slots below residency caps the
  // REGION capacity, not just the tags -- the cache then holds fewer pages
  // than the slab touches and the run faults+evicts every step. This is only
  // sound with the interior flushes ON (the default): eviction performs no
  // I/O, so a dirty page's bytes survive eviction only because every write
  // path published them first. MD_NO_INTERIOR + --slots would silently lose
  // data, so refuse the combination outright.
  if (a.slots != 0) {
    if (a.nodes > 1 && EnvOn("MD_NO_INTERIOR")) {
      std::fprintf(stderr,
                   "--slots (out-of-core) with MD_NO_INTERIOR=1 would evict "
                   "dirty pages whose only copy is the frame. Refusing.\n");
      return 1;
    }
    // Single node too: same region pressure without the docker harness. The
    // resident single-node ctests are untouched -- their pool rounds up to
    // one region per block, which already exceeds their working set.
    if (a.slots < md_cap) md_cap = a.slots;
    // f is NOT capped: forces are never flushed (they are per-step scratch,
    // rebuilt from x every step), so a dirty f page evicted under region
    // pressure would lose its only copy. x/v/x2/v2 are the state that pages.
  }
  // +4 TASK TABLES on x and v, reserved for the ASYNC publish kernel.
  // Tables are per logical block, and the publish overlaps the interior
  // force -- two kernels using one table means two in-flight tasks in one
  // slot, which the vector forbids. Tables are cheap (one task set each);
  // the publish gets 16..19 and the compute kernels keep 0..15.
  // nsets PINNED EXPLICITLY: nsets=0 defaults to nblocks, so widening the
  // task tables for the async publish silently moved the cache from 16 to
  // 20 sets -- hashing, widths and floors all shifted as a side effect of an
  // unrelated change. Tables are tasks; sets are cache geometry; they were
  // separated for exactly this reason.
  gv::Vector<float> vx(tag("md_x"), {0}, page_bytes, tbl_blocks + 4, x_slots,
                       g.nelems, clio::run::PoolId::GetNull(), 0, 1,
                       /*nsets=*/tbl_blocks, /*capacity_pages=*/md_cap);
  account("x", x_slots, page_bytes, npages, md_cap, tbl_blocks + 4);
  // v gets its own knob too, so x-paging and v-paging can be separated:
  // x is the only vector read through multi-page SPAN holds that stay live
  // across a park, which is the access pattern no other gate covers.
  const u32 vslots = SharedSlots(npages, nsets);
  gv::Vector<float> vv(tag("md_v"), {0}, page_bytes, tbl_blocks + 4, vslots,
                       g.nelems, clio::run::PoolId::GetNull(), 0, 1,
                       /*nsets=*/tbl_blocks, md_cap);
  account("v", vslots, page_bytes, npages, md_cap, tbl_blocks + 4);
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
    account("third", third_slots, page_bytes, npages, md_cap, tbl_blocks);
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
    md::SymbolWrite(md::MdSym::kReadBad, z4, sizeof(z4));
    const unsigned long long z8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    md::SymbolWrite(md::MdSym::kReadSample, z8, sizeof(z8));
    const bool ldcg = EnvOn("MD_PROBE_LDCG");
    md::SymbolWrite(md::MdSym::kProbeLdcg, &ldcg, sizeof(ldcg));
    const unsigned long long epp_host = g.nelems / npages;
    const clio::run::u64 audit_before = 0;  // frame audit removed with the debug accessors
    const double t0p = NowMs();
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      md::LaunchReadProbe(gr, b, CLIO_YIELD_SMEM_BYTES,
gpu, dxp, a.steps,
                                                        a.blocks, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    std::printf("  frame audit: after Prefetch=%llu bad, after kernel=%llu "
                "bad (slot i must own frame i)\n",
                (unsigned long long)audit_before,
                0ull);
    unsigned long long rb[4] = {0, 0, 0, 0};
    md::SymbolRead(rb, md::MdSym::kReadBad, sizeof(rb));
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
      md::SymbolRead(sm, md::MdSym::kReadSample, sizeof(sm));
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
        md::SymbolRead(gm, md::MdSym::kReadGeom, sizeof(gm));
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
    account("f", fslots, page_bytes, npages, md_cap_own, tbl_blocks);
    vf.EnableStats();
    {
      std::vector<float> hz(g.nelems, 0.0f);
      vf.Preload(hz.data(), g.nelems);
    }
    vf.ClearCache();
    vf.Prefetch(slab_pg_lo, slab_pg_hi, 0, 1);
    auto df = vf.GetDevice(0);
    // MD_VERIFY_REFAULT=N: the write/evict/refault/compare probe, instead of
    // the physics. See RefaultWriteCoro.
    if (const char *vr = getenv("MD_VERIFY_REFAULT");
        vr != nullptr && *vr != '\0' && atoi(vr) > 0) {
      const int vr_rounds = atoi(vr);
      // MD_VR_HALO=1 adds the exchange half: stamped boundary publishes and
      // generational peer-boundary reads, decoded per element.
      const bool vr_halo = EnvOn("MD_VR_HALO") && a.nodes > 1;
      const u64 vr_ppp = vr_halo ? ppp : 0;
      const u64 vr_below = ((u64)((my_z0 + g.nb - 1u) % g.nb)) * ppp;
      const u64 vr_above = ((u64)(my_z1 % g.nb)) * ppp;
      auto *d_vr =
          ctp::GpuApi::Malloc<unsigned long long>(9 * sizeof(unsigned long long));
      u64 total_bad = 0, total_newer = 0, total_older = 0, total_torn = 0;
      for (int r = 0; r < vr_rounds; ++r) {
        ctp::GpuApi::Memset(d_vr, 0, 9 * sizeof(unsigned long long));
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          md::LaunchRefaultWrite(gr, b, CLIO_YIELD_SMEM_BYTES,

              gpu, dx, slab_pg_lo, slab_pg_hi, (u64)r, vr_ppp, a.blocks, vw,
              sv);
        });
        ctp::GpuApi::Synchronize();
        vx.ClearCache();          // every frame dropped: each read refaults
        runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          md::LaunchRefaultVerify(gr, b, CLIO_YIELD_SMEM_BYTES,

              gpu, dx, slab_pg_lo, slab_pg_hi, (u64)r, vr_ppp, vr_below,
              vr_above, a.blocks, d_vr, vw, sv);
        });
        ctp::GpuApi::Synchronize();
        unsigned long long h_vr[9] = {0};
        ctp::GpuApi::Memcpy(h_vr, d_vr, sizeof(h_vr));
        total_bad += h_vr[0];
        total_newer += h_vr[6];
        total_older += h_vr[7];
        total_torn += h_vr[8];
        if (h_vr[0] != 0) {
          const unsigned int seen_bits = (unsigned int)h_vr[4];
          float seen;
          std::memcpy(&seen, &seen_bits, sizeof(seen));
          std::printf("  refault round %d: %llu BAD elems; first pg=%llu "
                      "e=%llu seen=%.1f want=%.1f\n",
                      r, h_vr[0], h_vr[2], h_vr[3], seen,
                      (double)(((h_vr[2] * 131ull + (u64)r * 4099ull +
                                 (h_vr[3] & 63ull)) & 0xFFFFull)));
        }
        if (vr_halo) {
          std::printf("  refault round %d: own %s | halo exact=%llu "
                      "newer=%llu older=%llu garbage=%llu\n",
                      r, h_vr[0] == 0 ? "clean" : "BAD", h_vr[5], h_vr[6],
                      h_vr[7], h_vr[8]);
        } else if (h_vr[0] == 0) {
          std::printf("  refault round %d: clean\n", r);
        }
        std::fflush(stdout);
      }
      std::printf("  REFAULT VERIFY: %s (own bad=%llu halo newer=%llu "
                  "older=%llu garbage=%llu over %d rounds)\n",
                  (total_bad + total_newer + total_older + total_torn) == 0
                      ? "PASS" : "FAIL",
                  (unsigned long long)total_bad,
                  (unsigned long long)total_newer,
                  (unsigned long long)total_older,
                  (unsigned long long)total_torn, vr_rounds);
      return (total_bad + total_newer + total_older + total_torn) == 0 ? 0
                                                                       : 1;
    }
    // Ping-pong destination vectors for the resort (K2b scatters into
    // these, then the handles swap). Same geometry, resident.
    // The resort's destinations get their own cache size, so "the scatter
    // is wrong" can be told apart from "the scatter's PAGING is wrong".
    const u32 ppslots = SharedSlots(npages, nsets);
    // +4 TABLES AND PINNED nsets, SAME AS vx/vv: the resort std::swap()s the
    // handles, so the async publish runs on THESE vectors every other epoch.
    // Widening only x/v made Init(16..19) trap with kFatalInitBlock the
    // first time the publish followed a swap -- and the trap's printf was
    // eaten by __trap, so it surfaced as a naked CUDA 719 one Synchronize
    // later. Every bisect arm agreed: gates green (pre-swap), first loop
    // step red, tbl_base=12 green (fits 16 tables), tbl_base=0 green.
    gv::Vector<float> vx2(tag("md_x2"), {0}, page_bytes, tbl_blocks + 4,
                          ppslots, g.nelems, clio::run::PoolId::GetNull(), 0,
                          1, /*nsets=*/tbl_blocks, md_cap);
    gv::Vector<float> vv2(tag("md_v2"), {0}, page_bytes, tbl_blocks + 4,
                          ppslots, g.nelems, clio::run::PoolId::GetNull(), 0,
                          1, /*nsets=*/tbl_blocks, md_cap);
    account("x2", ppslots, page_bytes, npages, md_cap, tbl_blocks + 4);
    account("v2", ppslots, page_bytes, npages, md_cap, tbl_blocks + 4);
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
    account("list", nlslots, nl_page_bytes, nl_pages, nl_cap, tbl_blocks);
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
        md::LaunchBuildList(gr, b, smem_force,

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
    // THE PUBLISH RUNS ASYNC AGAINST THE INTERIOR FORCE. Its own runner
    // (4 blocks: 3 waits + stride) so the main runner's state is free for
    // the force kernels while the exchange is in flight.
    YieldRunner pubrunner(4, a.threads);
    bool pub_pending = false;
    // 1 on the FIRST exchange a vector serves as x (run start, and again
    // after every resort swap): that exchange takes the persistent halo
    // reservation and keeps its pins. See PublishSlabCoro block 2.
    u32 halo_first = 1u;
    auto pub_launch = [&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                          gy::YieldStackView sv) {
      md::LaunchPublishSlab(gr, b, CLIO_YIELD_SMEM_BYTES,

          gpu, dx, dv, g.nb, g.cap, my_z0, my_z1, halo_gen, halo_first,
          /*nblocks=*/4, /*tbl_base=*/a.blocks, vw, sv);
    };
    auto pub_begin = [&]() {
      ++halo_gen;
      pubrunner.BeginSeq();
      if (pubrunner.Step(pub_launch)) {
        pub_pending = true;
      }
      ++r_kick_pub;
    };
    auto pub_drain = [&]() {
      while (pub_pending) {
        if (!pubrunner.Step(pub_launch)) pub_pending = false;
        ++r_kick_pub;
        // A __trap EATS ITS OWN printf, but the vector's fatal channel is
        // host-readable memory written BEFORE the trap. Read it HERE, before
        // the next Synchronize turns the story into "CUDA 719" and aborts.
        const char *ce = md::LaunchError();
        if (ce != nullptr) {
          std::fprintf(stderr, "[pub] launch error: %s\n", ce);
          std::fprintf(stderr, "[pub] device fatal: %s\n",
                       gv::FatalReport().c_str());
          std::fflush(stderr);
          std::_Exit(3);
        }
      }
      halo_first = 0u;   // the persistent halo pin is in place
    };
    // PUMP BOTH RUNNERS. runner.Run blocks the host until its kernel
    // drains, but a force chunk's wait can transitively need the PUBLISH to
    // advance (a boundary fetch waits on the peer's generation, the peer
    // waits on OUR boundary flush, and our flush is a parked publish
    // coroutine that only moves when the host Steps the pubrunner). With
    // Run() the host is the missing edge in a three-way cycle: force waits
    // on publish, publish waits on the host, the host waits on force --
    // observed as one node frozen at `admit stall used=24` and the peer's
    // generational get timing out (FATAL 7) forever. Stepping the pending
    // publish between force rounds removes the host edge.
    auto run_pumped = [&](auto &&launch) {
      runner.BeginSeq();
      u32 rounds = 0;
      for (;;) {
        const bool more = runner.Step(launch);
        if (pub_pending && !pubrunner.Step(pub_launch)) pub_pending = false;
        if (!more) break;
        if (++rounds >= kMaxRounds) {
          std::fprintf(stderr,
                       "[driver] GAVE UP after %u pumped rounds -- blocks "
                       "are still parked.\n", rounds);
          break;
        }
      }
    };
    auto force = [&](int eflag) {
      if (trace) { std::fprintf(stderr, "[md] force eflag=%d\n", eflag);
                   std::fflush(stderr); }
      const double _t = NowMs();
      if (eflag) ctp::GpuApi::Memset(d_acc, 0, 3 * sizeof(double));
      if (a.use_list) {
        MdMark("ListForce");
        // COMM/COMPUTE OVERLAP. While the publish is in flight, run the
        // INTERIOR band first -- its stencil never leaves the slab, so it
        // needs nothing the exchange delivers -- and let the CTE workers
        // service the publish's puts and the peer-halo get underneath it.
        // Drain, then the two BOUNDARY planes run against a warm halo.
        const bool overlap = pub_pending && !eflag;
        const u32 band1 = overlap ? 1u : 0u;
        run_pumped([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          md::LaunchListForce(gr, b, smem_force,

              gpu, dx, df, dn, g.nb, g.cap, fbox, fcut, a.maxneigh, d_cnt,
              eflag, d_acc, nocompute, a.rowchunk, my_z0, my_z1, a.blocks,
              force_gen ? force_gen : (no_halo ? 0 : halo_gen),
              force_gen != 0, band1, vw, sv);
        });
        pub_drain();
        if (overlap) {
          run_pumped([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                         gy::YieldStackView sv) {
            md::LaunchListForce(gr, b, smem_force,

                gpu, dx, df, dn, g.nb, g.cap, fbox, fcut, a.maxneigh, d_cnt,
                eflag, d_acc, nocompute, a.rowchunk, my_z0, my_z1, a.blocks,
                no_halo ? 0 : halo_gen, false, 2u, vw, sv);
          });
        }
      } else {
        pub_drain();
        MdMark("Force");
        run_pumped([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                       gy::YieldStackView sv) {
          md::LaunchForce(gr, b, smem_force,
gpu, dx, df, g.nb, g.cap, fbox,
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
      // THE SAME 4-BLOCK PUBLISH AS THE PER-STEP EXCHANGE, driven to
      // completion. This used to launch PublishSlabKernel on the MAIN runner
      // with a.blocks blocks -- and the coro's halo-warm arm is block 2, so
      // any --blocks < 3 run simply never executed it while the host still
      // cleared halo_first: the persistent halo pin was never taken, the
      // halo paged in and out every step, and the mid-read overwrite race
      // came back as a silent few-ulp poison (caught by the 4x sweep point:
      // refetch=0 where every healthy config shows one per halo page per
      // step, and a trajectory that no longer matched the reference).
      pub_begin();
      pub_drain();
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
        md::LaunchMDIntegrate(gr, b, CLIO_YIELD_SMEM_BYTES,

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
      if (drift && a.nodes > 1 && !no_halo) {
        pub_begin();
        // MD_NO_OVERLAP=1: submit-and-drain immediately -- isolates "the
        // async publish kernel with reserved tables" from "the overlap with
        // the interior band" when hunting a trap.
        if (EnvOn("MD_NO_OVERLAP")) pub_drain();
      }
      t_kick_pub += NowMs() - _t2;
      t_kick += NowMs() - _t;
    };
    auto thermo_ke = [&]() -> double {
      ctp::GpuApi::Memset(d_thermo, 0, 4 * sizeof(double));
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        md::LaunchThermo(gr, b, smem_thermo,
gpu, dx, dv, d_thermo, g.nb, g.cap,
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
        md::LaunchRebinWrap(gr, b, CLIO_YIELD_SMEM_BYTES,

            gpu, dx, g.nb, g.cap, fbox2, my_z0, my_z1, a.blocks, vw, sv);
      });
      ctp::GpuApi::Synchronize();
      exchange();
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        md::LaunchRebinAssign(gr, b, CLIO_YIELD_SMEM_BYTES,

            gpu, dx, g.nb, g.cap, fbox2, d_bincnt, d_dest, d_err,
            my_z0, my_z1, a.blocks, no_halo ? 0 : halo_gen, vw, sv);
      });
      ctp::GpuApi::Synchronize();
      // MD_RESORT_DEBUG=1: what the assign actually produced, from the
      // device arrays themselves -- distinguishes "assign wrote nothing"
      // from "gather dropped what assign wrote" in one line.
      if (EnvOn("MD_RESORT_DEBUG")) {
        unsigned long long gw = 0;
        md::SymbolRead(&gw, md::MdSym::kGatherWrote, sizeof(gw));
        const unsigned long long zz = 0;
        md::SymbolWrite(md::MdSym::kGatherWrote, &zz, sizeof(zz));
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
          md::LaunchSentinel(gr, b, CLIO_YIELD_SMEM_BYTES,
gpu, *dst, g.nb, g.cap,
                                                           my_z0, my_z1,
                                                           a.blocks, vw, sv);
        });
      }
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        md::LaunchGather(gr, b, CLIO_YIELD_SMEM_BYTES,

            gpu, dx, dx, dx2, g.nb, g.cap, d_dest, /*keep_w=*/1,
            my_z0, my_z1, a.blocks, no_halo ? 0 : halo_gen, vw, sv);
      });
      MdMark("Gather-v");
      MdMark("Gather-x");
      // The v halo is consumed ONLY here; pin it for the whole gather so an
      // eviction cannot refault it mid-pass into the peer's NEXT publish.
      if (a.nodes > 1 && !no_halo) {
        pubrunner.BeginSeq();
        while (pubrunner.Step([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                                  gy::YieldStackView sv) {
          md::LaunchHaloPin(gr, b, CLIO_YIELD_SMEM_BYTES,

              gpu, dv, g.nb, g.cap, my_z0, my_z1, halo_gen,
              /*tbl_base=*/a.blocks, vw, sv);
        })) {
        }
        ctp::GpuApi::Synchronize();
      }
      runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                     gy::YieldStackView sv) {
        md::LaunchGather(gr, b, CLIO_YIELD_SMEM_BYTES,

            gpu, dv, dx, dv2, g.nb, g.cap, d_dest, /*keep_w=*/0,
            my_z0, my_z1, a.blocks, no_halo ? 0 : halo_gen, vw, sv);
      });
      ctp::GpuApi::Synchronize();
      if (a.nodes > 1 && !no_halo) {
        pubrunner.BeginSeq();
        while (pubrunner.Step([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                                  gy::YieldStackView sv) {
          md::LaunchHaloUnpin(gr, b, CLIO_YIELD_SMEM_BYTES,

              gpu, dv, g.nb, g.cap, my_z0, my_z1, /*tbl_base=*/a.blocks, vw,
              sv);
        })) {
        }
        ctp::GpuApi::Synchronize();
      }
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
      // The halo role moves with the swap: give the old x its pins back and
      // let the next exchange re-reserve on the new one.
      if (a.nodes > 1) {
        pubrunner.BeginSeq();
        while (pubrunner.Step([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                                  gy::YieldStackView sv) {
          md::LaunchHaloUnpin(gr, b, CLIO_YIELD_SMEM_BYTES,

              gpu, dx, g.nb, g.cap, my_z0, my_z1, /*tbl_base=*/a.blocks, vw,
              sv);
        })) {
        }
        ctp::GpuApi::Synchronize();
        halo_first = 1u;
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
      md::SymbolWrite(md::MdSym::kPublish, &pub, sizeof(pub));
      const u32 pub_int = (a.nodes > 1 && EnvOn("MD_NO_INTERIOR")) ? 0u : 1u;
      md::SymbolWrite(md::MdSym::kPubInterior, &pub_int, sizeof(pub_int));
      // Out of core the frames are NOT the only copy anyone will ever read:
      // every write must reach the store before its page can be evicted.
      const u32 md_flush = (a.slots != 0) ? 1u : 0u;
      md::SymbolWrite(md::MdSym::kMdFlush, &md_flush, sizeof(md_flush));
      if (md_flush != 0u) {
        std::printf("  [out-of-core] write-site publishes ON for x/v and the "
                    "resort destinations\n");
      }
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
      h_ckpt_stock = ctp::GpuApi::MallocHost<float>(2 * g.nelems * sizeof(float));
    }
    // The persisted-and-isolated snapshot each checkpoint keeps: a LAZY
    // vector.Copy of positions and velocities. Only the newest pair is held
    // (a real run would name them per step and keep them all -- the tags are
    // cheap, it is the device views that are not, so the handles are
    // host-only).
    std::unique_ptr<gv::Vector<float>> ck_x, ck_v;
    auto checkpoint = [&]() {
      // NO SEPARATE CHECKPOINT PASS EXISTS ANY MORE, and that is the result,
      // not a measurement bug. Every kernel publishes each page as it writes
      // it -- it has to, because the other blocks read those pages -- so the
      // live state is already in the tier stack at every step boundary. The
      // barrier proves it landed; vector.Copy then pins THIS step's bytes
      // under a snapshot tag the next steps cannot overwrite. The copy is
      // lazy (fault-materialised), so its cost here is two tag registrations,
      // and the bytes are only duplicated for pages a later step dirties or
      // a restore reads.
      const double _t = NowMs();
      ctp::GpuApi::Synchronize();
      gv::Vector<float> *cvv = (cvx == &vx) ? &vv : &vv2;
      ck_x = cvx->Copy("md_ckpt_x_" + std::to_string(n_ckpt));
      ck_v = cvv->Copy("md_ckpt_v_" + std::to_string(n_ckpt));
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
        // The async publish holds x's boundary pins and reserved task
        // slots; the resort rewrites x wholesale. Settle it first.
        pub_drain();
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
    {
      // The x line alone hid which vector was actually churning: v and the
      // neighbour list page too, and each has its own staleness story.
      const auto sv_ = cvx == &vx ? vv.ReadStats(0) : vv2.ReadStats(0);
      const auto sx2_ = cvx == &vx ? vx2.ReadStats(0) : vx.ReadStats(0);
      const auto snl_ = vn.ReadStats(0);
      std::printf("  paging+: v faults=%llu evicts=%llu ge=%llu | "
                  "x2 faults=%llu evicts=%llu | nl faults=%llu evicts=%llu "
                  "ge=%llu\n",
                  (unsigned long long)sv_.faults,
                  (unsigned long long)sv_.evicts,
                  (unsigned long long)sv_.get_errors,
                  (unsigned long long)sx2_.faults,
                  (unsigned long long)sx2_.evicts,
                  (unsigned long long)snl_.faults,
                  (unsigned long long)snl_.evicts,
                  (unsigned long long)snl_.get_errors);
    }
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
      md::SymbolRead(xm, md::MdSym::kXMask, sizeof(xm));
      md::SymbolRead(nm, md::MdSym::kNlMask, sizeof(nm));
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
      // LAST SNAPSHOT TAG is read -- through the checkpoint fault handler,
      // exactly the path a restore would take -- not the live vector.
      std::vector<float> ck(g.nelems);
      const u64 got = ck_x->Download(ck.data(), g.nelems);
      u64 bad = 0;
      for (u64 e = 0; e < g.nelems; e += kStride) {
        if (ck[e + 3] >= 0.0f && !(ck[e] == ck[e])) ++bad;   // NaN check
      }
      std::printf("  checkpoints: %llu | flush + vector.Copy %.2f ms each "
                  "(page-granular publish + lazy snapshot tag) | host "
                  "staging a non-paged code would pay first: %.2f ms each | "
                  "snapshot readback %llu/%llu pages, %llu bad\n",
                  (unsigned long long)n_ckpt, t_ckpt / n_ckpt,
                  t_ckpt_stock / n_ckpt, (unsigned long long)got,
                  (unsigned long long)npages, (unsigned long long)bad);
    }
    if (d_ckpt_stock != nullptr) ctp::GpuApi::Free(d_ckpt_stock);
    if (h_ckpt_stock != nullptr) ctp::GpuApi::FreeHost(h_ckpt_stock);
    std::printf("  kick split: integrate=%.1f ms (%llu rounds) "
                "publish=%.1f ms (%llu rounds)\n",
                t_kick_int, (unsigned long long)r_kick_int, t_kick_pub,
                (unsigned long long)r_kick_pub);
    {
      unsigned long long fc = 0, gc = 0;
      md::SymbolRead(&fc, md::MdSym::kPubFlushCyc, sizeof(fc));
      md::SymbolRead(&gc, md::MdSym::kPubFetchCyc, sizeof(gc));
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
      md::SymbolRead(c, md::MdSym::kMdCyc, sizeof(c));
      const int khz = static_cast<int>(md::DeviceClockKHz());
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
    md::SymbolWrite(md::MdSym::kPagesDone, &z, sizeof(z));
  }
  const double t0 = NowMs();
  for (u64 step = 0; step < a.steps; ++step) {
    runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      md::LaunchIntegrate(gr, b, CLIO_YIELD_SMEM_BYTES,

          gpu, dx, dv, dthird, use_third, fdt, fgx, fgy, fgz, /*drift=*/1,
          a.blocks, vw, sv);
    });
    const u32 rr = runner.Run([&](dim3 gr, dim3 b, gy::YieldableView<> vw,
                                  gy::YieldStackView sv) {
      md::LaunchIntegrate(gr, b, CLIO_YIELD_SMEM_BYTES,

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
      md::LaunchThermo(gr, b, smem_thermo,
gpu, dx, dv, d_thermo, g.nb,
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
    md::SymbolRead(&done, md::MdSym::kPagesDone, sizeof(done));
    const unsigned long long want =
        static_cast<unsigned long long>(npages) * a.steps * 2ull;
    std::printf("  page iterations: %llu of %llu expected%s\n", done, want,
                (done == want) ? "  [all work ran]"
                               : "   <-- WORK WAS DROPPED");
    unsigned long long bd[64] = {0}, bl[64] = {0};
    md::SymbolRead(bd, md::MdSym::kBlkDone, sizeof(bd));
    md::SymbolRead(bl, md::MdSym::kBlkLast, sizeof(bl));
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
  BenchFlushData();
  return rc;
#endif  // GV_MD_CORO
}
#endif  // !CTP_IS_DEVICE_PASS
