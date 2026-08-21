/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * eternia-MD, NVSHMEM edition: the SAME melt deck (lj/cut, NVE, periodic box,
 * bin-major padded layout, minimum image, device-built Verlet list) with the
 * one substitution that matters -- simulation state lives in the NVSHMEM
 * SYMMETRIC HEAP, partitioned across PEs, instead of on paged gv::Vectors.
 *
 * Why this file exists
 * --------------------
 * `clio_gpu_vector_md_bench` answers "what does a GPU MD code cost when its
 * capacity ceiling is the CTE tier stack (HBM -> RAM -> NVMe) rather than
 * VRAM?" The honest question next to it is: "what does the OTHER way of
 * escaping the VRAM ceiling cost?" That other way is scale-out -- put the
 * box on N GPUs and reach for a neighbour's memory over NVLink/IB. NVSHMEM
 * is how that is written today.
 *
 *   md_bench (paged gv::Vector)          md_nvshmem_bench (symmetric heap)
 *     capacity = the tier stack            capacity = npes * VRAM
 *     miss  -> page fault -> tier          remote row -> getmem from a peer
 *     1 rank, 0 bytes over PCIe/step       N PEs, halo bytes over the fabric
 *
 * Everything except the memory substrate is deliberately IDENTICAL to the
 * eternia bench, because the comparison is only worth anything if the
 * physics, the layout, the work decomposition and the gates are the same:
 *
 *   - atoms bin-major in PADDED bins, an atom's index is (bin, slot), empty
 *     slots carry type = -1 in x's .w lane;
 *   - work unit = a CHUNK of x-rows of bins within one z-plane, so the
 *     stencil for the chunk is three z-planes x (rowchunk + 2) y-rows,
 *     each a CONTIGUOUS run of elements;
 *   - list entries packed stencil-relative as (q << 16) | slot-within-row,
 *     transposed and padded so a warp reads consecutive addresses;
 *   - the same gates: step-0 statics against a host double reference AND
 *     the published stock LAMMPS PE/atom, resort continuity, NVE drift, and
 *     the ballistic gate (bitwise against a host float replica).
 *
 * The substitution, precisely
 * ---------------------------
 * The bin grid is cut into contiguous z-plane slabs, one slab per PE, and
 * x/v/f/list are symmetric allocations sized to the largest slab. Where the
 * eternia kernel writes
 *
 *     Held<float> h = co_await x.HoldPage(base, len);   // may fault to a tier
 *
 * this kernel writes
 *
 *     const float *p = StageSpan(...);   // local pointer, or a block-
 *                                        // collective getmem from the
 *                                        // owning PE into block scratch
 *
 * That per-block scratch buffer is the structural analogue of the page
 * cache, and it is reported as such. There is no fault path, no eviction,
 * no tier: a row is either in this PE's slab or it is copied from the peer
 * that owns it, every time it is needed.
 *
 * What this costs, and why it is the point
 * ----------------------------------------
 * Nothing here is out of core. Per-PE VRAM must hold a whole slab of x, v,
 * f, the Verlet list and the ping-pong resort buffers; the run refuses at
 * startup with numbers when it will not fit. The ledger at the end reports
 * the halo bytes moved per step and the fraction of the force pass spent
 * staging them -- the fabric traffic that buys the capacity, against the
 * eternia bench's page traffic that buys the same capacity a different way.
 *
 * Single-GPU note: with one PE every row is local, so `--force-remote`
 * routes even self-owned rows through the staging path (a self-PE getmem).
 * That exercises the whole remote code path on one GPU and prices the
 * staging copy itself; it is not a substitute for a real multi-GPU run.
 *
 * Build: needs NVSHMEM (CLIO_CORE_ENABLE_NVSHMEM=ON) and is compiled by
 * NVCC with -rdc=true, independent of the tree's clang CUDA path, because
 * the NVSHMEM device library requires relocatable device code. It links
 * NOTHING from clio: the baseline must not be able to hide behind the
 * runtime it is being compared to.
 *
 * Run recipe (mirrors the eternia bench's deck):
 *   1 PE:   clio_md_nvshmem_bench --md --lattice 20 --steps 50 --rebin 10 \
 *             --temp 3.0 --cap 48 --blocks 128 --threads 64
 *   N PEs:  mpirun -n 4 clio_md_nvshmem_bench --md --lattice 40 ...
 *           (MPI bootstrap; built when MPI is found, else single-PE UID)
 */

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <string>
#include <vector>

#if defined(MD_NVSHMEM_USE_MPI)
#include <mpi.h>
#endif

using u32 = unsigned int;
using u64 = unsigned long long;

/** Elements per atom in x, v and f (float4 packing: xyz + type/spare). */
static constexpr u32 kStride = 4;
/** Stencil rows a chunk can need per z-plane: [y0-1, ylast+1] is one run,
 *  two when it crosses the y wrap, so three planes give at most six. */
static constexpr int kMaxSpans = 6;
/** LAMMPS melt step-0 pair energy per atom (rho=0.8442 FCC, lj/cut 2.5, no
 *  shift): the published stock value this reimplementation must reproduce
 *  from geometry alone -- the same constant the eternia bench gates on. */
static constexpr double kMeltPePerAtom = -6.7733681;

/** Device comm ledger, indexed by kCtr*: staging is the whole cost of
 *  distribution, so it is counted rather than estimated. */
enum CtrIdx {
  kCtrRemoteBytes = 0,   // bytes copied from another PE
  kCtrLocalBytes = 1,    // bytes copied from this PE (--force-remote)
  kCtrRemoteSpans = 2,   // getmem calls to another PE
  kCtrLocalSpans = 3,    // spans resolved to a bare local pointer
  kCtrMigrants = 4,      // atoms that changed PE at a resort
  kCtrNumCtrs = 8
};

#define MD_CUDA_CHECK(call)                                                  \
  do {                                                                       \
    const cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(_e));                                  \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

namespace {

struct Args {
  u32 lattice = 20;        // FCC unit cells per dimension -> 4*n^3 atoms
  double rho = 0.8442;     // reduced density (the melt deck's)
  double cutoff = 2.5;
  double skin = 0.3;
  u32 cap = 32;            // atom slots per bin (padded, fixed)
  u32 blocks = 128;
  u32 threads = 64;
  u64 steps = 100;
  u64 ckpt = 0;            // checkpoint every N steps (0 = never)
  std::string ckpt_dir;    // if set, checkpoints are made DURABLE here
  u64 rebin = 20;          // resort cadence (steps); 0 = never
  double temp = 0.0;       // scale initial velocities to this T (0 = leave)
  u32 maxneigh = 96;       // Verlet-list capacity per atom slot
  u32 rowchunk = 4;        // rows per block per chunk (staging reuse)
  int use_list = 1;        // --no-list = cell-direct stencil scan
  int force_remote = 0;    // stage even self-owned rows through getmem
  double drift_tol = 5e-4;
  double dt = 0.005;
  int gate = 1;            // the ballistic gate IS the default run
  int md = 0;              // --md: real LJ forces + NVE + statics gates
  double g[3] = {0.1, -0.05, 0.02};   // constant acceleration for the gate
};

/** Geometry: identical to the eternia bench's, plus the z-plane split. */
struct Geometry {
  double box = 0.0;
  double bin_edge = 0.0;
  u32 nb = 0;              // bins per dimension
  u64 nbins = 0;           // nb^3
  u32 cap = 0;
  u64 natoms = 0;
  u64 row_elems = 0;       // one y-row of bins, in elements
  u64 plane_elems = 0;     // one z-plane, in elements
};

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

/** Deterministic per-atom initial velocity -- the eternia bench's function,
 *  bit for bit, so the two codes start from the same state. */
void InitVelocity(u64 atom_id, float out[3]) {
  const double a = static_cast<double>(atom_id);
  out[0] = static_cast<float>(0.20 * std::sin(0.37 * a + 0.1));
  out[1] = static_cast<float>(0.20 * std::cos(0.53 * a + 0.7));
  out[2] = static_cast<float>(0.20 * std::sin(0.71 * a + 1.3));
}

/**
 * Build the WHOLE bin-major initial state on the host. Every PE builds all
 * of it (it is the same deterministic lattice) and then uploads only its own
 * slab; the full arrays stay on the host as the reference the gates compare
 * against. Refuses loudly on bin overflow.
 */
bool BuildInitialState(const Args &a, const Geometry &g,
                       std::vector<float> &hx, std::vector<float> &hv) {
  const u64 nslots = g.nbins * g.cap;
  hx.assign(nslots * kStride, 0.0f);
  hv.assign(nslots * kStride, 0.0f);
  for (u64 s = 0; s < nslots; ++s) hx[s * kStride + 3] = -1.0f;
  std::vector<u32> bin_count(g.nbins, 0u);

  const double cell = g.box / a.lattice;
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
          if (bx >= g.nb) bx = g.nb - 1;
          if (by >= g.nb) by = g.nb - 1;
          if (bz >= g.nb) bz = g.nb - 1;
          const u64 bin = (static_cast<u64>(bz) * g.nb + by) * g.nb + bx;
          const u32 slot = bin_count[bin]++;
          if (slot >= g.cap) {
            std::fprintf(stderr, "bin %llu overflows cap=%u -- raise --cap\n",
                         bin, g.cap);
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
 * Host double-precision reference for the step-0 statics: the same
 * cell-direct pass in double, over the whole box. Returns PE, the scalar
 * virial W = sum r.f (pairs counted once) and the pair count within cutoff.
 * Byte-identical in intent to the eternia bench's reference, so a
 * disagreement between the two codes is a disagreement about the substrate,
 * never about the physics.
 */
void HostForceReference(const Geometry &g, const std::vector<float> &hx,
                        double cutoff, double *pe_out, double *w_out,
                        u64 *pairs_out) {
  const double L = g.box;
  const double c2 = cutoff * cutoff;
  double pe = 0.0, w = 0.0;
  u64 pairs = 0;
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
            const u64 jbin = (static_cast<u64>(jbz) * nb + jby) * nb + jbx;
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
              pe += 0.5 * (4.0 * r6i * (r6i - 1.0));
              w += 0.5 * (r6i * (48.0 * r6i - 24.0));
              ++pairs;
            }
          }
        }
      }
    }
  }
  *pe_out = pe;
  *w_out = w;
  *pairs_out = pairs / 2;
}

}  // namespace

/**
 * The decomposition, as the kernels see it. Small, by value, in every
 * kernel signature: the whole distributed story is these seven fields plus
 * two tiny lookup tables.
 */
struct Decomp {
  u32 nb = 0;
  u32 cap = 0;
  u32 npes = 1;
  u32 mype = 0;
  u32 nplanes = 0;         // planes THIS pe owns
  u32 z0 = 0;              // first global plane this pe owns
  int force_remote = 0;    // stage self-owned rows too (1-GPU exercise)
  const u32 *plane_pe = nullptr;   // [nb]   global plane -> owning pe
  const u32 *plane_z0 = nullptr;   // [npes] pe -> its first global plane
};

/** Global plane -> (owning pe, element offset of that plane in the owner's
 *  symmetric slab). The only address translation in the whole file. */
__device__ __forceinline__ u64 PlaneOff(const Decomp &d, u32 bz,
                                        u64 plane_elems, int *pe) {
  const u32 p = d.plane_pe[bz];
  *pe = static_cast<int>(p);
  return static_cast<u64>(bz - d.plane_z0[p]) * plane_elems;
}

/**
 * Resolve the stencil of a CHUNK of rows into at most six contiguous spans,
 * copying the ones this PE does not own into block scratch.
 *
 * This is the function the whole comparison turns on: it is where the
 * eternia bench would co_await HoldPage and possibly fault to a tier, and
 * where a distributed code instead pays a fabric copy. Block-collective --
 * every thread of the block must call it, and it leaves the block
 * synchronized with all spans readable.
 *
 * The BLOCKING block-scoped getmem is used deliberately in preference to
 * the _nbi form: NVSHMEM exposes no block-scoped quiet, so completing an
 * nbi block transfer would mean a quiet from every thread, and over a P2P
 * transport the blocking form is already a cooperative block-wide copy.
 */
__device__ int StageStencil(const Decomp &d, const float *xsym, float *scratch,
                            u32 bz, u32 y0, u32 ylast, u64 row_elems,
                            u64 plane_elems, const float **sp, u32 *sbase,
                            u32 *scnt, u32 *sdz, unsigned long long *ctr) {
  const u32 nb = d.nb;
  int nspans = 0;
  u64 cursor = 0;
  for (int dz = -1; dz <= 1; ++dz) {
    const u32 wz = (bz + nb + dz) % nb;
    int pe = 0;
    const u64 poff = PlaneOff(d, wz, plane_elems, &pe);
    // The chunk's stencil in y is [y0-1, ylast+1], one contiguous run of
    // rows unless it crosses the wrap, in which case it is two.
    const int lo = static_cast<int>(y0) - 1;
    const int hi = static_cast<int>(ylast) + 1;
    u32 rl[2], rn[2];
    int nr = 0;
    if (hi - lo + 1 >= static_cast<int>(nb)) {
      rl[nr] = 0; rn[nr] = nb; ++nr;                    // whole plane
    } else if (lo < 0) {
      rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
      rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
    } else if (hi > static_cast<int>(nb) - 1) {
      rl[nr] = static_cast<u32>(lo); rn[nr] = nb - static_cast<u32>(lo); ++nr;
      rl[nr] = 0; rn[nr] = 1u; ++nr;
    } else {
      rl[nr] = static_cast<u32>(lo);
      rn[nr] = static_cast<u32>(hi - lo + 1); ++nr;
    }
    for (int t = 0; t < nr; ++t) {
      const u64 off = poff + static_cast<u64>(rl[t]) * row_elems;
      const u64 len = static_cast<u64>(rn[t]) * row_elems;
      const bool local = (pe == static_cast<int>(d.mype)) && !d.force_remote;
      if (local) {
        sp[nspans] = xsym + off;
        if (threadIdx.x == 0) atomicAdd(&ctr[kCtrLocalSpans], 1ull);
      } else {
        float *dst = scratch + cursor;
        // Symmetric addressing: the SOURCE is this PE's own address of the
        // symmetric object; NVSHMEM translates it into the peer's heap.
        nvshmemx_getmem_block(dst, xsym + off, len * sizeof(float), pe);
        sp[nspans] = dst;
        cursor += len;
        if (threadIdx.x == 0) {
          if (pe == static_cast<int>(d.mype)) {
            atomicAdd(&ctr[kCtrLocalBytes], len * sizeof(float));
          } else {
            atomicAdd(&ctr[kCtrRemoteBytes], len * sizeof(float));
          }
          atomicAdd(&ctr[kCtrRemoteSpans], 1ull);
        }
      }
      sbase[nspans] = rl[t];
      scnt[nspans] = rn[t];
      sdz[nspans] = static_cast<u32>(dz + 1);
      ++nspans;
    }
  }
  __syncthreads();
  return nspans;
}

/**
 * Publish the nine stencil-row pointers for row `by` into shared memory.
 * Block-uniform bookkeeping belongs in shared, not in per-thread arrays
 * indexed by a runtime value -- the same lesson the eternia kernels learned
 * (there the alternative was the coroutine frame in global memory; here it
 * is local memory, which is the same cache line problem one level down).
 *
 * Note what is NOT needed here and IS needed there: nothing in this kernel
 * can suspend, so shared memory published once stays valid. Distribution
 * costs a copy; paging costs a suspension. That is the trade in one line.
 */
__device__ __forceinline__ void PublishRow(const float **s_qptr, u32 by, u32 nb,
                                           int nspans, const float **sp,
                                           const u32 *sbase, const u32 *scnt,
                                           const u32 *sdz, u64 row_elems) {
  if (threadIdx.x == 0) {
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        const u32 wy = (by + nb + dy) % nb;
        const int q = (dz + 1) * 3 + (dy + 1);
        for (int t = 0; t < nspans; ++t) {
          if (sdz[t] != static_cast<u32>(dz + 1)) continue;
          if (wy >= sbase[t] && wy < sbase[t] + scnt[t]) {
            s_qptr[q] = sp[t] + static_cast<u64>(wy - sbase[t]) * row_elems;
            break;
          }
        }
      }
    }
  }
  __syncthreads();
}

/** Per-block scratch, in elements: three planes' worth of the chunk's
 *  stencil rows. The structural analogue of a page cache slot count. */
__host__ __device__ __forceinline__ u64 ScratchElems(u32 rowchunk, u32 nb,
                                                     u64 row_elems) {
  const u32 rows = (rowchunk + 2 < nb) ? (rowchunk + 2) : nb;
  return 3ull * rows * row_elems;
}

/**
 * K3 (cell-direct): the LJ force pass with no list -- scan the 27 stencil
 * bins directly. Same arithmetic, same minimum image, same halving of PE
 * and virial as the eternia bench; full list / newton off, so every atom's
 * force is computed entirely by its owning block and there are no atomics
 * on f anywhere.
 */
__global__ void ForceKernel(Decomp d, const float *x, float *f, float box,
                            float cutoff, int eflag, double *acc,
                            float *scratch, u64 row_elems, u64 plane_elems,
                            u32 rowchunk, unsigned long long *ctr) {
  extern __shared__ char smem[];
  double *red = reinterpret_cast<double *>(smem);
  const float **s_qptr =
      reinterpret_cast<const float **>(smem + blockDim.x * sizeof(double));
  const u32 nb = d.nb, cap = d.cap;
  const u64 islots = static_cast<u64>(nb) * cap;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  float *const myscratch =
      scratch + blockIdx.x * ScratchElems(rowchunk, nb, row_elems);
  double pe = 0.0, w = 0.0, npairs = 0.0;

  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  const u64 nchunks = static_cast<u64>(d.nplanes) * cpz;
  for (u64 ch = blockIdx.x; ch < nchunks; ch += gridDim.x) {
    const u32 lz = static_cast<u32>(ch / cpz);
    const u32 bz = d.z0 + lz;
    const u32 y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    const u32 ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1) : (nb - 1);

    const float *sp[kMaxSpans];
    u32 sbase[kMaxSpans], scnt[kMaxSpans], sdz[kMaxSpans];
    const int nspans = StageStencil(d, x, myscratch, bz, y0, ylast, row_elems,
                                    plane_elems, sp, sbase, scnt, sdz, ctr);

    for (u32 by = y0; by <= ylast; ++by) {
      PublishRow(s_qptr, by, nb, nspans, sp, sbase, scnt, sdz, row_elems);
      // f is local by construction: a PE computes forces only for its own
      // atoms, so nothing is ever written to a peer.
      float *const fp = f + (static_cast<u64>(lz) * nb + by) * row_elems;
      for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) fp[e] = 0.0f;
      __syncthreads();
      const float *const ip_row = s_qptr[4];
      for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
        const float *const ip = ip_row + s * kStride;
        if (ip[3] < 0.0f) continue;   // padded slot
        const float xi = ip[0], yi = ip[1], zi = ip[2];
        const u32 bx = static_cast<u32>(s / cap);
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int q = 0; q < 9; ++q) {
          const float *const qp = s_qptr[q];
          for (int dxx = -1; dxx <= 1; ++dxx) {
            const u32 jbx = (bx + nb + dxx) % nb;
            const u64 jb = static_cast<u64>(jbx) * cap * kStride;
            for (u32 sj = 0; sj < cap; ++sj) {
              const float *const jp = qp + jb + static_cast<u64>(sj) * kStride;
              if (jp[3] < 0.0f) continue;
              if (q == 4 && jbx == bx && sj == s % cap) continue;   // self
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
          }
        }
        float *const op = fp + s * kStride;
        op[0] = fx;
        op[1] = fy;
        op[2] = fz;
      }
      __syncthreads();
    }
  }
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
 * K2c: build the Verlet list for this PE's rows. TRANSPOSED and PADDED --
 * entry k of row-slot s at row * islots * maxneigh + k * islots + s -- so
 * a warp reads consecutive addresses at every k, and entries are packed
 * STENCIL-relative as (q << 16) | (jbx * cap + sj), which the force pass
 * decodes with two shifts against its own staged spans. Both properties are
 * carried over unchanged from the eternia bench; a list that is laid out
 * differently would not be the same benchmark.
 */
__global__ void BuildListKernel(Decomp d, const float *x, int *nl, float rlist,
                                u32 maxneigh, u32 *cnt, int *err,
                                float *scratch, u64 row_elems, u64 plane_elems,
                                u32 rowchunk, float box,
                                unsigned long long *ctr) {
  extern __shared__ char smem[];
  const float **s_qptr = reinterpret_cast<const float **>(smem);
  const u32 nb = d.nb, cap = d.cap;
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float r2list = rlist * rlist;
  const float halfL = 0.5f * box;
  float *const myscratch =
      scratch + blockIdx.x * ScratchElems(rowchunk, nb, row_elems);

  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  const u64 nchunks = static_cast<u64>(d.nplanes) * cpz;
  for (u64 ch = blockIdx.x; ch < nchunks; ch += gridDim.x) {
    const u32 lz = static_cast<u32>(ch / cpz);
    const u32 bz = d.z0 + lz;
    const u32 y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    const u32 ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1) : (nb - 1);

    const float *sp[kMaxSpans];
    u32 sbase[kMaxSpans], scnt[kMaxSpans], sdz[kMaxSpans];
    const int nspans = StageStencil(d, x, myscratch, bz, y0, ylast, row_elems,
                                    plane_elems, sp, sbase, scnt, sdz, ctr);

    for (u32 by = y0; by <= ylast; ++by) {
      PublishRow(s_qptr, by, nb, nspans, sp, sbase, scnt, sdz, row_elems);
      const u64 lrow = static_cast<u64>(lz) * nb + by;
      int *const np = nl + lrow * rowlist;
      const u64 slotbase = lrow * islots;
      const float *const ip_row = s_qptr[4];
      for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
        const float *const ip = ip_row + s * kStride;
        if (ip[3] < 0.0f) {
          cnt[slotbase + s] = 0;
          continue;
        }
        const float xi = ip[0], yi = ip[1], zi = ip[2];
        const u32 bx = static_cast<u32>(s / cap);
        u32 n = 0;
        for (int q = 0; q < 9; ++q) {
          const float *const qp = s_qptr[q];
          for (int dxx = -1; dxx <= 1; ++dxx) {
            const u32 jbx = (bx + nb + dxx) % nb;
            const u64 jb = static_cast<u64>(jbx) * cap * kStride;
            for (u32 sj = 0; sj < cap; ++sj) {
              const float *const jp = qp + jb + static_cast<u64>(sj) * kStride;
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
              if (n >= maxneigh) {   // refuse, never overrun
                *err = 1;
                continue;
              }
              np[static_cast<u64>(n) * islots + s] = static_cast<int>(
                  (static_cast<u32>(q) << 16) | (jbx * cap + sj));
              ++n;
            }
          }
        }
        cnt[slotbase + s] = n;
      }
      __syncthreads();
    }
  }
}

/** K3-list: the force pass streaming this atom's padded column of entries.
 *  Entries cover cutoff + skin; the cutoff test here skips the skin shell,
 *  which is what keeps the list valid between rebuilds. */
__global__ void ListForceKernel(Decomp d, const float *x, float *f,
                                const int *nl, const u32 *cnt, float box,
                                float cutoff, u32 maxneigh, int eflag,
                                double *acc, float *scratch, u64 row_elems,
                                u64 plane_elems, u32 rowchunk, int nocompute,
                                unsigned long long *ctr) {
  extern __shared__ char smem[];
  double *red = reinterpret_cast<double *>(smem);
  const float **s_qptr =
      reinterpret_cast<const float **>(smem + blockDim.x * sizeof(double));
  const u32 nb = d.nb, cap = d.cap;
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  float *const myscratch =
      scratch + blockIdx.x * ScratchElems(rowchunk, nb, row_elems);
  double pe = 0.0, w = 0.0, npairs = 0.0;

  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  const u64 nchunks = static_cast<u64>(d.nplanes) * cpz;
  for (u64 ch = blockIdx.x; ch < nchunks; ch += gridDim.x) {
    const u32 lz = static_cast<u32>(ch / cpz);
    const u32 bz = d.z0 + lz;
    const u32 y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    const u32 ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1) : (nb - 1);

    const float *sp[kMaxSpans];
    u32 sbase[kMaxSpans], scnt[kMaxSpans], sdz[kMaxSpans];
    const int nspans = StageStencil(d, x, myscratch, bz, y0, ylast, row_elems,
                                    plane_elems, sp, sbase, scnt, sdz, ctr);

    for (u32 by = y0; by <= ylast; ++by) {
      PublishRow(s_qptr, by, nb, nspans, sp, sbase, scnt, sdz, row_elems);
      const u64 lrow = static_cast<u64>(lz) * nb + by;
      float *const fp = f + lrow * row_elems;
      for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) fp[e] = 0.0f;
      __syncthreads();
      const int *const np = nl + lrow * rowlist;
      const u64 slotbase = lrow * islots;
      const float *const ip_row = s_qptr[4];
      for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
        const float *const ip = ip_row + s * kStride;
        if (ip[3] < 0.0f) continue;
        const float xi = ip[0], yi = ip[1], zi = ip[2];
        const u32 n = nocompute ? 0u : cnt[slotbase + s];
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (u32 k = 0; k < n; ++k) {
          const u32 ent = static_cast<u32>(np[static_cast<u64>(k) * islots + s]);
          const float *const jp =
              s_qptr[ent >> 16] + static_cast<u64>(ent & 0xffffu) * kStride;
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
        float *const op = fp + s * kStride;
        op[0] = fx;
        op[1] = fy;
        op[2] = fz;
      }
      __syncthreads();
    }
  }
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
 * K2a: wrap positions into the box, compute each atom's new bin, and claim
 * a slot in it. THE DISTRIBUTED PART: an atom whose new bin lies in another
 * PE's slab claims its slot with a REMOTE atomic on that PE's bin counter,
 * which is exactly the migration the eternia bench does not have to write
 * (there the atom stays in the same address space and only its page moves).
 */
__global__ void RebinKernel(Decomp d, float *x, float box, u32 *bincnt,
                            int *dest_pe, u32 *dest_slot, int *err,
                            u64 local_slots, unsigned long long *ctr) {
  const u32 nb = d.nb, cap = d.cap;
  const float fnb = static_cast<float>(nb);
  for (u64 s = blockIdx.x * blockDim.x + threadIdx.x; s < local_slots;
       s += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 e = s * kStride;
    if (x[e + 3] < 0.0f) {
      dest_pe[s] = -1;
      continue;
    }
    float px = x[e + 0], py = x[e + 1], pz = x[e + 2];
    if (px < 0.0f) px += box; else if (px >= box) px -= box;
    if (py < 0.0f) py += box; else if (py >= box) py -= box;
    if (pz < 0.0f) pz += box; else if (pz >= box) pz -= box;
    x[e + 0] = px; x[e + 1] = py; x[e + 2] = pz;
    u32 bx = static_cast<u32>(px * fnb / box);
    u32 by = static_cast<u32>(py * fnb / box);
    u32 bz = static_cast<u32>(pz * fnb / box);
    if (bx >= nb) bx = nb - 1;
    if (by >= nb) by = nb - 1;
    if (bz >= nb) bz = nb - 1;
    const int pe = static_cast<int>(d.plane_pe[bz]);
    const u32 lz = bz - d.plane_z0[pe];
    const u64 lbin = (static_cast<u64>(lz) * nb + by) * nb + bx;
    u32 slot;
    if (pe == static_cast<int>(d.mype)) {
      slot = atomicAdd(&bincnt[lbin], 1u);
    } else {
      slot = nvshmem_uint32_atomic_fetch_add(&bincnt[lbin], 1u, pe);
      atomicAdd(&ctr[kCtrMigrants], 1ull);
    }
    if (slot >= cap) {
      *err = 1;
      dest_pe[s] = -1;
      continue;
    }
    dest_pe[s] = pe;
    dest_slot[s] = static_cast<u32>(lbin * cap + slot);
  }
}

/** Pre-sentinel every slot of a ping-pong destination array. */
__global__ void SentinelKernel(float *dst, u64 local_slots) {
  for (u64 s = blockIdx.x * blockDim.x + threadIdx.x; s < local_slots;
       s += static_cast<u64>(gridDim.x) * blockDim.x) {
    dst[s * kStride + 3] = -1.0f;
  }
}

/**
 * K2b: apply the permutation for ONE array. A local destination is a plain
 * store; a destination on another PE is four element puts into that PE's
 * symmetric copy. Element puts rather than a 16-byte putmem on purpose:
 * putmem wants a device buffer for the source, and migrants are a thin
 * boundary population -- the honest cost of this design is that they are
 * scattered, not that they are large.
 */
__global__ void ScatterKernel(Decomp d, const float *src, const float *srcx,
                              float *dst, const int *dest_pe,
                              const u32 *dest_slot, int keep_w,
                              u64 local_slots) {
  int put_any = 0;
  for (u64 s = blockIdx.x * blockDim.x + threadIdx.x; s < local_slots;
       s += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 e = s * kStride;
    if (srcx[e + 3] < 0.0f) continue;
    const int pe = dest_pe[s];
    if (pe < 0) continue;   // overflow victim; the host has already refused
    const u64 de = static_cast<u64>(dest_slot[s]) * kStride;
    const float w = keep_w ? src[e + 3] : 0.0f;
    if (pe == static_cast<int>(d.mype)) {
      dst[de + 0] = src[e + 0];
      dst[de + 1] = src[e + 1];
      dst[de + 2] = src[e + 2];
      dst[de + 3] = w;
    } else {
      nvshmem_float_p(&dst[de + 0], src[e + 0], pe);
      nvshmem_float_p(&dst[de + 1], src[e + 1], pe);
      nvshmem_float_p(&dst[de + 2], src[e + 2], pe);
      nvshmem_float_p(&dst[de + 3], w, pe);
      put_any = 1;
    }
  }
  // COMPLETE THE PUTS HERE, not by trusting the host barrier. Over a P2P
  // transport these are plain stores and kernel completion settles them, but
  // over a proxied transport they are in flight until quieted, and the next
  // thing that happens is a peer reading this exact destination array. Quiet
  // is per-thread, so only the threads that issued a put pay for it.
  if (put_any) nvshmem_quiet();
}

/** K1/K4 for real MD: velocity Verlet, mass 1, force read from f. Purely
 *  local -- every PE integrates the atoms in its own slab. */
__global__ void MDIntegrateKernel(float *x, float *v, const float *f, float dt,
                                  int drift, u64 local_slots) {
  const float half = 0.5f * dt;
  for (u64 s = blockIdx.x * blockDim.x + threadIdx.x; s < local_slots;
       s += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 e = s * kStride;
    if (x[e + 3] < 0.0f) continue;
    const float vx = __fmaf_rn(half, f[e + 0], v[e + 0]);
    const float vy = __fmaf_rn(half, f[e + 1], v[e + 1]);
    const float vz = __fmaf_rn(half, f[e + 2], v[e + 2]);
    v[e + 0] = vx; v[e + 1] = vy; v[e + 2] = vz;
    if (drift) {
      x[e + 0] = __fmaf_rn(dt, vx, x[e + 0]);
      x[e + 1] = __fmaf_rn(dt, vy, x[e + 1]);
      x[e + 2] = __fmaf_rn(dt, vz, x[e + 2]);
    }
  }
}

/** The ballistic-gate integrator: constant acceleration, explicit fma on
 *  every update so the host float replica can be compared BITWISE. */
__global__ void IntegrateKernel(float *x, float *v, float dt, float gx,
                                float gy, float gz, int drift,
                                u64 local_slots) {
  const float half = 0.5f * dt;
  for (u64 s = blockIdx.x * blockDim.x + threadIdx.x; s < local_slots;
       s += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 e = s * kStride;
    if (x[e + 3] < 0.0f) continue;
    const float vx = __fmaf_rn(half, gx, v[e + 0]);
    const float vy = __fmaf_rn(half, gy, v[e + 1]);
    const float vz = __fmaf_rn(half, gz, v[e + 2]);
    v[e + 0] = vx; v[e + 1] = vy; v[e + 2] = vz;
    if (drift) {
      x[e + 0] = __fmaf_rn(dt, vx, x[e + 0]);
      x[e + 1] = __fmaf_rn(dt, vy, x[e + 1]);
      x[e + 2] = __fmaf_rn(dt, vz, x[e + 2]);
    }
  }
}

/** K5: local KE and net momentum; the host sums them across PEs. */
__global__ void ThermoKernel(const float *x, const float *v, double *out,
                             u64 local_slots) {
  extern __shared__ char smem[];
  double *red = reinterpret_cast<double *>(smem);
  double ke = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (u64 s = blockIdx.x * blockDim.x + threadIdx.x; s < local_slots;
       s += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 e = s * kStride;
    if (x[e + 3] < 0.0f) continue;
    const double vx = v[e + 0], vy = v[e + 1], vz = v[e + 2];
    ke += 0.5 * (vx * vx + vy * vy + vz * vz);   // m = 1
    mx += vx; my += vy; mz += vz;
  }
  const double vals[4] = {ke, mx, my, mz};
  for (int q = 0; q < 4; ++q) {
    red[threadIdx.x] = vals[q];
    __syncthreads();
    for (u32 wd = blockDim.x / 2; wd > 0; wd >>= 1) {
      if (threadIdx.x < wd) red[threadIdx.x] += red[threadIdx.x + wd];
      __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&out[q], red[0]);
    __syncthreads();
  }
}

namespace {

/**
 * Cross-PE sums of a handful of host doubles: thermo scalars, gate verdicts,
 * the comm ledger. Deliberately NOT an NVSHMEM collective when MPI is in the
 * build -- these are host scalars on a control path, nothing about them wants
 * the symmetric heap, and NVSHMEM's collectives are unavailable in "limited
 * MPG" configurations (more than one PE per GPU), which is exactly the shape
 * a workstation has to use to test the decomposition at all. The DATA path is
 * untouched: staging, migrant atomics and migrant puts are all NVSHMEM.
 */
struct Reducer {
  double *src = nullptr;
  double *dst = nullptr;
  static constexpr int kMax = 8;

  void Init() {
#if !defined(MD_NVSHMEM_USE_MPI)
    src = static_cast<double *>(nvshmem_malloc(kMax * sizeof(double)));
    dst = static_cast<double *>(nvshmem_malloc(kMax * sizeof(double)));
#endif
  }
  void Fini() {
    if (src) nvshmem_free(src);
    if (dst) nvshmem_free(dst);
  }
  /** Sum n doubles across all PEs, in place on the host buffer. */
  void Sum(double *host, int n) {
#if defined(MD_NVSHMEM_USE_MPI)
    MPI_Allreduce(MPI_IN_PLACE, host, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    MD_CUDA_CHECK(cudaMemcpy(src, host, n * sizeof(double),
                             cudaMemcpyHostToDevice));
    nvshmem_double_sum_reduce(NVSHMEM_TEAM_WORLD, dst, src, n);
    MD_CUDA_CHECK(cudaMemcpy(host, dst, n * sizeof(double),
                             cudaMemcpyDeviceToHost));
#endif
  }
};

}  // namespace

int main(int argc, char **argv) {
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
    else if (want("--blocks")) a.blocks = static_cast<u32>(atoi(argv[++i]));
    else if (want("--threads")) a.threads = static_cast<u32>(atoi(argv[++i]));
    else if (want("--steps")) a.steps = static_cast<u64>(atol(argv[++i]));
    else if (want("--ckpt")) a.ckpt = static_cast<u64>(atol(argv[++i]));
    else if (want("--ckpt-dir")) a.ckpt_dir = argv[++i];
    else if (want("--rebin")) a.rebin = static_cast<u64>(atol(argv[++i]));
    else if (want("--temp")) a.temp = atof(argv[++i]);
    else if (want("--drift-tol")) a.drift_tol = atof(argv[++i]);
    else if (want("--maxneigh")) a.maxneigh = static_cast<u32>(atoi(argv[++i]));
    else if (want("--rowchunk")) a.rowchunk = static_cast<u32>(atoi(argv[++i]));
    else if (want("--dt")) a.dt = atof(argv[++i]);
    else if (std::strcmp(argv[i], "--no-list") == 0) a.use_list = 0;
    else if (std::strcmp(argv[i], "--force-remote") == 0) a.force_remote = 1;
    else if (std::strcmp(argv[i], "--no-gate") == 0) a.gate = 0;
    else if (std::strcmp(argv[i], "--md") == 0) a.md = 1;
    else {
      std::fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }

  // ---- bootstrap ----------------------------------------------------------
  // MPI when it is compiled in (so `mpirun -n N` just works and each rank
  // takes one GPU), otherwise a single-PE unique-ID bootstrap so the
  // benchmark still runs on a workstation with one card.
  int mype = 0, npes = 1;
#if defined(MD_NVSHMEM_USE_MPI)
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mype);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  int ndev = 0;
  MD_CUDA_CHECK(cudaGetDeviceCount(&ndev));
  MD_CUDA_CHECK(cudaSetDevice(mype % (ndev > 0 ? ndev : 1)));
  {
    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &comm;
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
      std::fprintf(stderr, "nvshmem MPI bootstrap failed\n");
      return 1;
    }
  }
#else
  // CUDA CONTEXT BEFORE INIT. nvshmemx_init_attr reports success without a
  // current device but the symmetric heap is never created, and the first
  // nvshmem_malloc then fails with "API called before initialization has
  // completed" -- a failure that looks like a bootstrap bug and is not one.
  MD_CUDA_CHECK(cudaSetDevice(0));
  {
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_get_uniqueid(&uid);
    nvshmemx_set_attr_uniqueid_args(0, 1, &uid, &attr);
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) {
      std::fprintf(stderr, "nvshmem uniqueid bootstrap failed\n");
      return 1;
    }
  }
#endif
  mype = nvshmem_my_pe();
  npes = nvshmem_n_pes();
  const bool root = (mype == 0);

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
  g.row_elems = static_cast<u64>(g.nb) * g.cap * kStride;
  g.plane_elems = static_cast<u64>(g.nb) * g.row_elems;

  if (a.md && g.nb < 3) {
    if (root) std::fprintf(stderr, "need at least 3 bins per dimension\n");
    return 1;
  }
  if (static_cast<u64>(g.nb) * g.cap >= 65536) {
    if (root) std::fprintf(stderr, "nb*cap must fit 16 bits for entry packing\n");
    return 1;
  }
  if (static_cast<u32>(npes) > g.nb) {
    if (root) {
      std::fprintf(stderr,
                   "%d PEs but only %u z-planes: a PE would own nothing\n",
                   npes, g.nb);
    }
    return 1;
  }
  // THE STENCIL MUST NOT WRAP ONTO THE OWNER. With one or two planes per PE
  // a chunk's three-plane stencil can span more than the neighbours, which
  // is legal here (every plane is addressable from every PE) but makes the
  // decomposition meaningless -- refuse rather than report a number nobody
  // should quote.
  if (g.nb / static_cast<u32>(npes) < 3 && npes > 1) {
    if (root) {
      std::fprintf(stderr,
                   "each PE needs at least 3 z-planes (%u planes / %d PEs); "
                   "raise --lattice or lower the PE count\n", g.nb, npes);
    }
    return 1;
  }

  // Even z-plane split; the symmetric allocation is sized to the largest
  // slab, because every PE's heap must be identical.
  std::vector<u32> h_plane_z0(npes), h_plane_pe(g.nb);
  u32 maxplanes = 0;
  for (int p = 0; p < npes; ++p) {
    const u32 z0 = static_cast<u32>(static_cast<u64>(p) * g.nb / npes);
    const u32 z1 = static_cast<u32>(static_cast<u64>(p + 1) * g.nb / npes);
    h_plane_z0[p] = z0;
    for (u32 z = z0; z < z1; ++z) h_plane_pe[z] = static_cast<u32>(p);
    maxplanes = std::max(maxplanes, z1 - z0);
  }
  const u32 myz0 = h_plane_z0[mype];
  const u32 myz1 = (mype + 1 < npes)
                       ? h_plane_z0[mype + 1]
                       : g.nb;
  const u32 mynplanes = myz1 - myz0;

  const u64 local_slots =
      static_cast<u64>(maxplanes) * g.nb * g.nb * g.cap;
  const u64 local_elems = local_slots * kStride;
  const u64 islots = static_cast<u64>(g.nb) * g.cap;
  const u64 rowlist = islots * a.maxneigh;
  const u64 nl_elems = static_cast<u64>(maxplanes) * g.nb * rowlist;
  const u64 scratch_elems =
      static_cast<u64>(a.blocks) * ScratchElems(a.rowchunk, g.nb, g.row_elems);

  // ---- footprint, refused up front rather than discovered as an OOM ------
  const double xvf_mb = 3.0 * local_elems * sizeof(float) / 1048576.0;
  const double pp_mb = 2.0 * local_elems * sizeof(float) / 1048576.0;
  const double nl_mb = nl_elems * sizeof(int) / 1048576.0;
  const double idx_mb =
      (local_slots * (sizeof(u32) + sizeof(int) + sizeof(u32)) +
       static_cast<u64>(maxplanes) * g.nb * g.nb * sizeof(u32)) /
      1048576.0;
  const double scratch_mb = scratch_elems * sizeof(float) / 1048576.0;
  const double total_mb =
      xvf_mb + (a.md ? pp_mb : 0.0) + (a.md && a.use_list ? nl_mb : 0.0) +
      (a.md ? idx_mb : 0.0) + (a.md ? scratch_mb : 0.0);
  if (root) {
    std::printf(
        "eternia-MD / NVSHMEM baseline\n"
        "  atoms=%llu box=%.4f bins=%u^3 cap=%u  PEs=%d (planes/PE=%u..%u)\n"
        "  blocks=%u threads=%u rowchunk=%u list=%s staging=%s\n"
        "  per-PE VRAM: x/v/f %.1f MB + resort %.1f MB + list %.1f MB + "
        "index %.1f MB + stencil scratch %.2f MB = %.1f MB\n"
        "  (NOTHING here is out of core: the capacity ceiling is %d x VRAM, "
        "not a tier stack)\n",
        g.natoms, g.box, g.nb, g.cap, npes, g.nb / static_cast<u32>(npes),
        maxplanes, a.blocks, a.threads, a.rowchunk,
        a.use_list ? "verlet" : "cell-direct",
        a.force_remote ? "ALL rows staged (--force-remote)"
                       : "remote rows only",
        xvf_mb, a.md ? pp_mb : 0.0, (a.md && a.use_list) ? nl_mb : 0.0,
        a.md ? idx_mb : 0.0, a.md ? scratch_mb : 0.0, total_mb, npes);
  }
  {
    size_t freeb = 0, totb = 0;
    MD_CUDA_CHECK(cudaMemGetInfo(&freeb, &totb));
    if (total_mb * 1048576.0 > 0.85 * static_cast<double>(freeb)) {
      if (root) {
        std::fprintf(stderr,
                     "refusing: %.1f MB of per-PE state against %.1f MB free "
                     "VRAM. This baseline has no tier to spill to -- lower "
                     "--lattice/--cap/--maxneigh or add PEs.\n",
                     total_mb, freeb / 1048576.0);
      }
      return 1;
    }
  }

  // ---- host initial state -------------------------------------------------
  std::vector<float> hx, hv;
  if (!BuildInitialState(a, g, hx, hv)) return 1;
  if (a.temp > 0.0) {
    // Zero the net momentum, then scale to the requested reduced temperature
    // (KE = 1.5 N T, m = 1) -- the melt deck's hot start, same as eternia.
    const u64 nslots = g.nbins * g.cap;
    double mean[3] = {0, 0, 0};
    for (u64 s = 0; s < nslots; ++s) {
      if (hx[s * kStride + 3] < 0.0f) continue;
      for (int d = 0; d < 3; ++d) mean[d] += hv[s * kStride + d];
    }
    for (int d = 0; d < 3; ++d) mean[d] /= static_cast<double>(g.natoms);
    double ke = 0.0;
    for (u64 s = 0; s < nslots; ++s) {
      if (hx[s * kStride + 3] < 0.0f) continue;
      for (int d = 0; d < 3; ++d) {
        const double vd = hv[s * kStride + d] - mean[d];
        hv[s * kStride + d] = static_cast<float>(vd);
        ke += 0.5 * vd * vd;
      }
    }
    const double scale =
        std::sqrt(1.5 * a.temp * static_cast<double>(g.natoms) / ke);
    for (u64 s = 0; s < nslots; ++s) {
      if (hx[s * kStride + 3] < 0.0f) continue;
      for (int d = 0; d < 3; ++d) {
        hv[s * kStride + d] = static_cast<float>(hv[s * kStride + d] * scale);
      }
    }
  }

  // ---- symmetric allocation ----------------------------------------------
  float *dx = static_cast<float *>(nvshmem_malloc(local_elems * sizeof(float)));
  float *dv = static_cast<float *>(nvshmem_malloc(local_elems * sizeof(float)));
  float *df = static_cast<float *>(nvshmem_malloc(local_elems * sizeof(float)));
  if (dx == nullptr || dv == nullptr || df == nullptr) {
    std::fprintf(stderr, "nvshmem_malloc failed for x/v/f\n");
    return 1;
  }
  Reducer red;
  red.Init();

  // Upload this PE's slab. The tail planes of a short slab are sentineled so
  // every reachable slot is either a real atom or explicitly empty -- the
  // same contract the eternia bench enforces by rounding its vector up to
  // whole pages.
  {
    std::vector<float> sx(local_elems, 0.0f), sv(local_elems, 0.0f);
    for (u64 s = 0; s < local_slots; ++s) sx[s * kStride + 3] = -1.0f;
    for (u32 lz = 0; lz < mynplanes; ++lz) {
      const u64 gsrc = static_cast<u64>(myz0 + lz) * g.plane_elems;
      const u64 ldst = static_cast<u64>(lz) * g.plane_elems;
      std::memcpy(&sx[ldst], &hx[gsrc], g.plane_elems * sizeof(float));
      std::memcpy(&sv[ldst], &hv[gsrc], g.plane_elems * sizeof(float));
    }
    MD_CUDA_CHECK(cudaMemcpy(dx, sx.data(), local_elems * sizeof(float),
                             cudaMemcpyHostToDevice));
    MD_CUDA_CHECK(cudaMemcpy(dv, sv.data(), local_elems * sizeof(float),
                             cudaMemcpyHostToDevice));
  }
  MD_CUDA_CHECK(cudaMemset(df, 0, local_elems * sizeof(float)));

  // Decomposition tables and the comm ledger (plain device memory: they are
  // read-only machinery, never communicated).
  Decomp d;
  d.nb = g.nb;
  d.cap = g.cap;
  d.npes = static_cast<u32>(npes);
  d.mype = static_cast<u32>(mype);
  d.nplanes = mynplanes;
  d.z0 = myz0;
  d.force_remote = a.force_remote;
  {
    u32 *t0 = nullptr, *t1 = nullptr;
    MD_CUDA_CHECK(cudaMalloc(&t0, g.nb * sizeof(u32)));
    MD_CUDA_CHECK(cudaMalloc(&t1, npes * sizeof(u32)));
    MD_CUDA_CHECK(cudaMemcpy(t0, h_plane_pe.data(), g.nb * sizeof(u32),
                             cudaMemcpyHostToDevice));
    MD_CUDA_CHECK(cudaMemcpy(t1, h_plane_z0.data(), npes * sizeof(u32),
                             cudaMemcpyHostToDevice));
    d.plane_pe = t0;
    d.plane_z0 = t1;
  }
  unsigned long long *d_ctr = nullptr;
  MD_CUDA_CHECK(cudaMalloc(&d_ctr, kCtrNumCtrs * sizeof(unsigned long long)));
  MD_CUDA_CHECK(cudaMemset(d_ctr, 0, kCtrNumCtrs * sizeof(unsigned long long)));
  float *d_scratch = nullptr;
  MD_CUDA_CHECK(cudaMalloc(&d_scratch, scratch_elems * sizeof(float)));
  double *d_thermo = nullptr;
  MD_CUDA_CHECK(cudaMalloc(&d_thermo, 4 * sizeof(double)));

  const u32 smem_thermo = a.threads * sizeof(double);
  const u32 smem_force =
      a.threads * sizeof(double) + 9 * sizeof(void *);
  const u32 smem_build = 9 * sizeof(void *);
  const float fbox = static_cast<float>(g.box);
  const float fcut = static_cast<float>(a.cutoff);
  const float frlist = static_cast<float>(a.cutoff + a.skin);
  const float fdt = static_cast<float>(a.dt);

  auto sync_all = [&]() {
    MD_CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
  };

  // ---- STAGE 2/3: real MD -------------------------------------------------
  if (a.md) {
    float *dx2 = static_cast<float *>(
        nvshmem_malloc(local_elems * sizeof(float)));
    float *dv2 = static_cast<float *>(
        nvshmem_malloc(local_elems * sizeof(float)));
    // The bin counters are symmetric because a migrating atom claims its
    // slot with a remote atomic on the OWNER's counter.
    u32 *d_bincnt = static_cast<u32 *>(nvshmem_malloc(
        static_cast<u64>(maxplanes) * g.nb * g.nb * sizeof(u32)));
    int *d_dest_pe = nullptr;
    u32 *d_dest_slot = nullptr, *d_cnt = nullptr;
    int *d_err = nullptr;
    double *d_acc = nullptr;
    int *d_nl = nullptr;
    MD_CUDA_CHECK(cudaMalloc(&d_dest_pe, local_slots * sizeof(int)));
    MD_CUDA_CHECK(cudaMalloc(&d_dest_slot, local_slots * sizeof(u32)));
    MD_CUDA_CHECK(cudaMalloc(&d_cnt, local_slots * sizeof(u32)));
    MD_CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));
    MD_CUDA_CHECK(cudaMalloc(&d_acc, 3 * sizeof(double)));
    if (a.use_list) {
      MD_CUDA_CHECK(cudaMalloc(&d_nl, nl_elems * sizeof(int)));
      MD_CUDA_CHECK(cudaMemset(d_nl, 0, nl_elems * sizeof(int)));
    }
    if (dx2 == nullptr || dv2 == nullptr || d_bincnt == nullptr) {
      std::fprintf(stderr, "nvshmem_malloc failed for the resort buffers\n");
      return 1;
    }

    const u32 lin_blocks = a.blocks;
    double acc[3] = {0, 0, 0};
    double t_force = 0.0, t_kick = 0.0, t_resort = 0.0, t_build = 0.0;
    const int nocompute = std::getenv("MD_NOCOMPUTE") != nullptr ? 1 : 0;

    auto build_list = [&]() -> bool {
      const double t = NowMs();
      MD_CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
      sync_all();   // every PE's positions must be settled before staging
      BuildListKernel<<<a.blocks, a.threads, smem_build>>>(
          d, dx, d_nl, frlist, a.maxneigh, d_cnt, d_err, d_scratch,
          g.row_elems, g.plane_elems, a.rowchunk, fbox, d_ctr);
      MD_CUDA_CHECK(cudaGetLastError());
      sync_all();
      int err = 0;
      MD_CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(int),
                               cudaMemcpyDeviceToHost));
      double e = err;
      red.Sum(&e, 1);
      t_build += NowMs() - t;
      if (e != 0.0) {
        if (root) {
          std::fprintf(stderr,
                       "list: an atom has more than --maxneigh %u neighbours "
                       "within cutoff+skin\n", a.maxneigh);
        }
        return false;
      }
      return true;
    };
    auto force = [&](int eflag) {
      const double t = NowMs();
      if (eflag) MD_CUDA_CHECK(cudaMemset(d_acc, 0, 3 * sizeof(double)));
      sync_all();
      if (a.use_list) {
        ListForceKernel<<<a.blocks, a.threads, smem_force>>>(
            d, dx, df, d_nl, d_cnt, fbox, fcut, a.maxneigh, eflag, d_acc,
            d_scratch, g.row_elems, g.plane_elems, a.rowchunk, nocompute,
            d_ctr);
      } else {
        ForceKernel<<<a.blocks, a.threads, smem_force>>>(
            d, dx, df, fbox, fcut, eflag, d_acc, d_scratch, g.row_elems,
            g.plane_elems, a.rowchunk, d_ctr);
      }
      MD_CUDA_CHECK(cudaGetLastError());
      sync_all();
      if (eflag) {
        MD_CUDA_CHECK(cudaMemcpy(acc, d_acc, 3 * sizeof(double),
                                 cudaMemcpyDeviceToHost));
        red.Sum(acc, 3);
      }
      t_force += NowMs() - t;
    };
    auto kick = [&](int drift) {
      const double t = NowMs();
      MDIntegrateKernel<<<lin_blocks, a.threads>>>(dx, dv, df, fdt, drift,
                                                   local_slots);
      MD_CUDA_CHECK(cudaGetLastError());
      MD_CUDA_CHECK(cudaDeviceSynchronize());
      t_kick += NowMs() - t;
    };
    auto thermo_ke = [&]() -> double {
      MD_CUDA_CHECK(cudaMemset(d_thermo, 0, 4 * sizeof(double)));
      ThermoKernel<<<lin_blocks, a.threads, smem_thermo>>>(dx, dv, d_thermo,
                                                           local_slots);
      MD_CUDA_CHECK(cudaGetLastError());
      MD_CUDA_CHECK(cudaDeviceSynchronize());
      double t4[4];
      MD_CUDA_CHECK(cudaMemcpy(t4, d_thermo, sizeof(t4),
                               cudaMemcpyDeviceToHost));
      red.Sum(t4, 4);
      return t4[0];
    };
    // K2: wrap + rebin (remote atomics for migrants) + scatter (remote puts)
    // + swap. Barriers bracket every phase because a peer may be writing
    // into this PE's destination array.
    auto resort = [&]() -> bool {
      const double t = NowMs();
      MD_CUDA_CHECK(cudaMemset(d_bincnt, 0, static_cast<u64>(maxplanes) *
                                                g.nb * g.nb * sizeof(u32)));
      MD_CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
      SentinelKernel<<<lin_blocks, a.threads>>>(dx2, local_slots);
      SentinelKernel<<<lin_blocks, a.threads>>>(dv2, local_slots);
      sync_all();
      RebinKernel<<<lin_blocks, a.threads>>>(d, dx, fbox, d_bincnt, d_dest_pe,
                                             d_dest_slot, d_err, local_slots,
                                             d_ctr);
      MD_CUDA_CHECK(cudaGetLastError());
      sync_all();
      int err = 0;
      MD_CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(int),
                               cudaMemcpyDeviceToHost));
      double e = err;
      red.Sum(&e, 1);
      if (e != 0.0) {
        if (root) std::fprintf(stderr, "resort: bin overflow -- raise --cap\n");
        return false;
      }
      ScatterKernel<<<lin_blocks, a.threads>>>(d, dx, dx, dx2, d_dest_pe,
                                               d_dest_slot, /*keep_w=*/1,
                                               local_slots);
      ScatterKernel<<<lin_blocks, a.threads>>>(d, dv, dx, dv2, d_dest_pe,
                                               d_dest_slot, /*keep_w=*/0,
                                               local_slots);
      MD_CUDA_CHECK(cudaGetLastError());
      sync_all();
      std::swap(dx, dx2);
      std::swap(dv, dv2);
      t_resort += NowMs() - t;
      return true;
    };

    // ---- validation layer 2: step-0 statics from geometry alone ----------
    if (a.use_list && !build_list()) return 1;
    force(/*eflag=*/1);
    double pe_ref = 0.0, w_ref = 0.0;
    u64 pairs_ref = 0;
    HostForceReference(g, hx, a.cutoff, &pe_ref, &w_ref, &pairs_ref);
    const double pe0 = acc[0];
    const u64 pairs0 = static_cast<u64>(acc[2] + 0.5) / 2;
    const double pe_atom = pe0 / static_cast<double>(g.natoms);
    const double ref_rel = std::fabs(pe0 - pe_ref) / std::fabs(pe_ref);
    const double melt_abs = std::fabs(pe_atom - kMeltPePerAtom);
    const bool statics_ok =
        (pairs0 == pairs_ref) && (ref_rel < 2e-5) && (melt_abs < 2e-4);
    if (root) {
      std::printf(
          "  statics: PE/atom dev=%.7f hostref=%.7f LAMMPS=%.7f | pairs "
          "dev=%llu ref=%llu | W dev=%.6g ref=%.6g\n"
          "  STATICS GATE: %s (ref_rel=%.2e, melt_abs=%.2e)\n",
          pe_atom, pe_ref / g.natoms, kMeltPePerAtom, pairs0, pairs_ref,
          acc[1], w_ref, statics_ok ? "PASS" : "FAIL", ref_rel, melt_abs);
    }

    // ---- resort continuity: same physical state, new layout, so PE must
    // be unchanged to float-summation noise (and, with npes > 1, atoms have
    // just crossed PE boundaries) ------------------------------------------
    bool resort_ok = true;
    double resort_rel = 0.0;
    if (a.rebin != 0) {
      const double pe_before = acc[0];
      if (!resort()) return 1;
      if (a.use_list && !build_list()) return 1;
      force(/*eflag=*/1);
      resort_rel = std::fabs(acc[0] - pe_before) / std::fabs(pe_before);
      resort_ok = resort_rel < 1e-6;
      if (root) {
        std::printf("  RESORT GATE: %s (PE %.6f -> %.6f, rel=%.2e)\n",
                    resort_ok ? "PASS" : "FAIL", pe_before, acc[0],
                    resort_rel);
      }
    }

    // ---- NVE ------------------------------------------------------------
    const double ke0 = thermo_ke();
    const double e0 = acc[0] + ke0;
    MD_CUDA_CHECK(cudaMemset(d_ctr, 0,
                             kCtrNumCtrs * sizeof(unsigned long long)));
    sync_all();
    const double t0 = NowMs();
    // ---- CHECKPOINT ------------------------------------------------------
    // What a non-persistent substrate must do to survive a crash: get the
    // simulation state off the GPU. x and v only -- forces and the Verlet
    // list are derived and a restart rebuilds them, which is what LAMMPS
    // restart files carry too.
    //
    // The destination is PINNED HOST DRAM, deliberately: the paged vector
    // this is compared against persists into a CTE tier stack whose only
    // configured target is a RAM tier, so staging to host DRAM lands the
    // bytes in the same storage class. Anything else compares two
    // destinations, not two substrates.
    float *h_ckpt = nullptr;
    double t_ckpt = 0.0;
    double t_ckpt_io = 0.0;
    u64 n_ckpt = 0;
    const u64 ckpt_elems = 2 * local_elems;
    if (a.ckpt != 0) {
      MD_CUDA_CHECK(cudaMallocHost(&h_ckpt, ckpt_elems * sizeof(float)));
    }
    auto checkpoint = [&]() {
      const double _t = NowMs();
      MD_CUDA_CHECK(cudaMemcpy(h_ckpt, dx, local_elems * sizeof(float),
                               cudaMemcpyDeviceToHost));
      MD_CUDA_CHECK(cudaMemcpy(h_ckpt + local_elems, dv,
                               local_elems * sizeof(float),
                               cudaMemcpyDeviceToHost));
      t_ckpt += NowMs() - _t;
      // DURABLE HALF. Staging to host DRAM is only the PCIe hop; a
      // checkpoint that a crash cannot eat has to reach storage. Written as
      // one file per checkpoint (restart-file shaped) and fsync'd, because
      // without the fsync this measures the page cache, not the device.
      if (!a.ckpt_dir.empty()) {
        const double _w = NowMs();
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/ckpt_%llu.bin",
                      a.ckpt_dir.c_str(), (unsigned long long)n_ckpt);
        int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
          const char *p = reinterpret_cast<const char *>(h_ckpt);
          size_t left = ckpt_elems * sizeof(float);
          while (left > 0) {
            ssize_t w = ::write(fd, p, left);
            if (w <= 0) break;
            p += w; left -= static_cast<size_t>(w);
          }
          ::fsync(fd);
          ::close(fd);
        }
        t_ckpt_io += NowMs() - _w;
      }
      ++n_ckpt;
    };
    for (u64 step = 0; step < a.steps; ++step) {
      kick(/*drift=*/1);
      if (a.rebin != 0 && step != 0 && step % a.rebin == 0) {
        if (!resort()) return 1;
        if (a.use_list && !build_list()) return 1;
      }
      force(/*eflag=*/0);
      kick(/*drift=*/0);
     if (a.ckpt != 0 && (step + 1) % a.ckpt == 0) checkpoint();
    }
    sync_all();
    const double run_ms = NowMs() - t0;
    force(/*eflag=*/1);
    const double ke_n = thermo_ke();
    const double e_n = acc[0] + ke_n;
    const double e_drift = std::fabs(e_n - e0) / std::fabs(e0);
    const bool nve_ok = (e_drift < a.drift_tol);

    // Checkpoint ledger. Reported per run so the with/without comparison is
    // a difference in one measured quantity rather than an inference.
    if (n_ckpt != 0 && root) {
      const double mb = static_cast<double>(ckpt_elems) * sizeof(float) /
                        (1024.0 * 1024.0);
      std::printf(
          "  checkpoints: %llu x %.1f MB = %.2f GB | stage D2H %.1f ms each "
          "(%.2f GB/s) | durable write %.1f ms each (%.2f GB/s) | total "
          "%.1f ms = %.1f%% on top of the %.1f ms run%s\n",
          (unsigned long long)n_ckpt, mb,
          mb * static_cast<double>(n_ckpt) / 1024.0, t_ckpt / n_ckpt,
          (mb / 1024.0) / (t_ckpt / n_ckpt / 1000.0),
          n_ckpt ? t_ckpt_io / n_ckpt : 0.0,
          t_ckpt_io > 0.0 ? (mb / 1024.0) / (t_ckpt_io / n_ckpt / 1000.0) : 0.0,
          t_ckpt + t_ckpt_io, 100.0 * (t_ckpt + t_ckpt_io) / run_ms, run_ms,
          a.ckpt_dir.empty() ? "  [DRAM ONLY -- NOT durable]" : "");
    }
    if (h_ckpt != nullptr) cudaFreeHost(h_ckpt);

    // ---- the comm ledger: what distribution actually cost ---------------
    unsigned long long ctr[kCtrNumCtrs] = {0};
    MD_CUDA_CHECK(cudaMemcpy(ctr, d_ctr, sizeof(ctr), cudaMemcpyDeviceToHost));
    double cd[5] = {static_cast<double>(ctr[kCtrRemoteBytes]),
                    static_cast<double>(ctr[kCtrLocalBytes]),
                    static_cast<double>(ctr[kCtrRemoteSpans]),
                    static_cast<double>(ctr[kCtrLocalSpans]),
                    static_cast<double>(ctr[kCtrMigrants])};
    red.Sum(cd, 5);
    if (root) {
      std::printf(
          "  NVE %llu steps: E0=%.6f En=%.6f drift=%.2e | KE %.3f -> %.3f\n"
          "  NVE GATE: %s\n"
          "  %llu steps in %.1f ms (%.3f ms/step, %.1f Matom-steps/s)\n",
          a.steps, e0, e_n, e_drift, ke0, ke_n, nve_ok ? "PASS" : "FAIL",
          a.steps, run_ms, run_ms / a.steps,
          static_cast<double>(g.natoms) * a.steps / run_ms / 1000.0);
      const double per_step_mb = (cd[0] + cd[1]) / a.steps / 1048576.0;
      std::printf(
          "  comm ledger (all PEs): staged %.2f GB total, %.2f MB/step "
          "(%.1f%% of it from a PEER) | spans %.0f staged / %.0f resolved "
          "local | migrants %.0f\n"
          "  substrate: symmetric heap, %d PE(s); every byte above crossed "
          "the fabric or was copied to dodge it -- the eternia bench moves "
          "its equivalent through the CTE tier stack instead\n",
          (cd[0] + cd[1]) / 1073741824.0, per_step_mb,
          (cd[0] + cd[1]) > 0.0 ? 100.0 * cd[0] / (cd[0] + cd[1]) : 0.0,
          cd[2], cd[3], cd[4], npes);
      std::printf("  phases (total ms): force=%.1f kick=%.1f resort=%.1f "
                  "build=%.1f\n", t_force, t_kick, t_resort, t_build);
    }

    nvshmem_free(dx2);
    nvshmem_free(dv2);
    nvshmem_free(d_bincnt);
    MD_CUDA_CHECK(cudaFree(d_dest_pe));
    MD_CUDA_CHECK(cudaFree(d_dest_slot));
    MD_CUDA_CHECK(cudaFree(d_cnt));
    MD_CUDA_CHECK(cudaFree(d_err));
    MD_CUDA_CHECK(cudaFree(d_acc));
    if (d_nl) MD_CUDA_CHECK(cudaFree(d_nl));
    red.Fini();
    nvshmem_free(dx);
    nvshmem_free(dv);
    nvshmem_free(df);
    nvshmem_finalize();
#if defined(MD_NVSHMEM_USE_MPI)
    MPI_Finalize();
#endif
    return (statics_ok && nve_ok && resort_ok) ? 0 : 1;
  }

  // ---- the ballistic run --------------------------------------------------
  const float fgx = static_cast<float>(a.g[0]);
  const float fgy = static_cast<float>(a.g[1]);
  const float fgz = static_cast<float>(a.g[2]);
  sync_all();
  const double t0 = NowMs();
  for (u64 step = 0; step < a.steps; ++step) {
    IntegrateKernel<<<a.blocks, a.threads>>>(dx, dv, fdt, fgx, fgy, fgz,
                                             /*drift=*/1, local_slots);
    IntegrateKernel<<<a.blocks, a.threads>>>(dx, dv, fdt, fgx, fgy, fgz,
                                             /*drift=*/0, local_slots);
  }
  MD_CUDA_CHECK(cudaGetLastError());
  sync_all();
  const double run_ms = NowMs() - t0;

  double thermo[4] = {0, 0, 0, 0};
  MD_CUDA_CHECK(cudaMemset(d_thermo, 0, 4 * sizeof(double)));
  ThermoKernel<<<a.blocks, a.threads, smem_thermo>>>(dx, dv, d_thermo,
                                                     local_slots);
  MD_CUDA_CHECK(cudaDeviceSynchronize());
  MD_CUDA_CHECK(cudaMemcpy(thermo, d_thermo, sizeof(thermo),
                           cudaMemcpyDeviceToHost));
  red.Sum(thermo, 4);

  int rc = 0;
  if (a.gate) {
    // Replay the identical float recurrence on the host for THIS PE's slab
    // and demand bitwise equality, exactly as the eternia bench does; the
    // mismatch counts are then summed across PEs.
    std::vector<float> gx_out(local_elems), gv_out(local_elems);
    MD_CUDA_CHECK(cudaMemcpy(gx_out.data(), dx, local_elems * sizeof(float),
                             cudaMemcpyDeviceToHost));
    MD_CUDA_CHECK(cudaMemcpy(gv_out.data(), dv, local_elems * sizeof(float),
                             cudaMemcpyDeviceToHost));
    double bit_x = 0, bit_v = 0, max_cf = 0.0;
    double ke_ref = 0.0, mom_ref[3] = {0, 0, 0};
    const double n = static_cast<double>(a.steps);
    const float fhalf = 0.5f * fdt;
    const float gg[3] = {fgx, fgy, fgz};
    for (u32 lz = 0; lz < mynplanes; ++lz) {
      const u64 gbase = static_cast<u64>(myz0 + lz) * g.plane_elems;
      const u64 lbase = static_cast<u64>(lz) * g.plane_elems;
      for (u64 o = 0; o < g.plane_elems; o += kStride) {
        if (hx[gbase + o + 3] < 0.0f) continue;
        float rx[3] = {hx[gbase + o], hx[gbase + o + 1], hx[gbase + o + 2]};
        float rv[3] = {hv[gbase + o], hv[gbase + o + 1], hv[gbase + o + 2]};
        for (u64 st = 0; st < a.steps; ++st) {
          for (int c = 0; c < 3; ++c) {
            rv[c] = std::fmaf(fhalf, gg[c], rv[c]);
            rx[c] = std::fmaf(fdt, rv[c], rx[c]);
          }
          for (int c = 0; c < 3; ++c) rv[c] = std::fmaf(fhalf, gg[c], rv[c]);
        }
        for (int c = 0; c < 3; ++c) {
          if (rx[c] != gx_out[lbase + o + c]) bit_x += 1.0;
          if (rv[c] != gv_out[lbase + o + c]) bit_v += 1.0;
          const double cf = static_cast<double>(hx[gbase + o + c]) +
                            n * a.dt * static_cast<double>(hv[gbase + o + c]) +
                            0.5 * n * n * a.dt * a.dt * a.g[c];
          max_cf = std::max(max_cf, std::fabs(cf - gx_out[lbase + o + c]));
          const double vd = gv_out[lbase + o + c];
          mom_ref[c] += vd;
          ke_ref += 0.5 * vd * vd;
        }
      }
    }
    double gate[7] = {bit_x, bit_v, max_cf, ke_ref,
                      mom_ref[0], mom_ref[1], mom_ref[2]};
    // max_cf is summed rather than maxed; it is a bound either way, and a
    // sum over PEs can only make the gate stricter.
    red.Sum(gate, 7);
    const double ke_err = std::fabs(thermo[0] - gate[3]) /
                          (gate[3] != 0.0 ? gate[3] : 1.0);
    double mom_err = 0.0;
    for (int c = 0; c < 3; ++c) {
      mom_err = std::max(mom_err, std::fabs(thermo[1 + c] - gate[4 + c]));
    }
    const bool gate_ok = (gate[0] == 0.0 && gate[1] == 0.0) &&
                         (gate[2] < 1e-2) && (ke_err < 1e-9) &&
                         (mom_err < 1e-6);
    if (root) {
      std::printf(
          "  gate: bitwise mismatches x=%.0f v=%.0f | closed-form max "
          "|dx|=%.3e | thermo KE dev=%.6f host=%.6f rel_err=%.2e "
          "mom_err=%.2e\n  BALLISTIC GATE: %s\n",
          gate[0], gate[1], gate[2], thermo[0], gate[3], ke_err, mom_err,
          gate_ok ? "PASS" : "FAIL");
    }
    if (!gate_ok) rc = 1;
  }
  if (root) {
    std::printf("  %llu steps in %.1f ms (%.3f ms/step, %.1f Matom-steps/s)\n",
                a.steps, run_ms, run_ms / a.steps,
                static_cast<double>(g.natoms) * a.steps / run_ms / 1000.0);
  }

  red.Fini();
  nvshmem_free(dx);
  nvshmem_free(dv);
  nvshmem_free(df);
  nvshmem_finalize();
#if defined(MD_NVSHMEM_USE_MPI)
  MPI_Finalize();
#endif
  return rc;
}
