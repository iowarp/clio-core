/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * eternia-MD, MPI edition: the same melt deck and the same kernels as
 * `clio_md_nvshmem_bench`, with the ONE variable changed that the pair of
 * them exists to isolate -- how a rank reaches a row of bins it does not own.
 *
 *   md_nvshmem_bench   one-sided, in-kernel: a block pulls the stencil rows
 *                      it needs with nvshmemx_getmem_block, on demand, from
 *                      the symmetric heap of the PE that owns them.
 *   md_mpi_bench       two-sided, pre-staged: MPI cannot be called from a
 *                      kernel, so the boundary planes are exchanged with
 *                      MPI_Sendrecv BEFORE the launch and the kernel reads
 *                      them out of a halo already sitting in its own slab.
 *
 * Everything else is deliberately identical -- float precision, bin-major
 * padded layout, ghost-free minimum image, full list / newton off, the
 * transposed-padded Verlet list with (q << 16) | slot entries, the chunked
 * row decomposition, and all four gates. The pair loops are copied verbatim
 * rather than shared, matching this directory's house style of
 * self-contained benchmarks; the gates are what prove they still agree
 * (identical step-0 PE and an identical exact pair count across all three
 * benches is a strong equivalence check, and any drift shows up there).
 *
 * WHY THIS IS NOT A FAIR FIGHT, STATED UP FRONT
 * --------------------------------------------
 * The OpenMPI on this machine reports mpi_built_with_cuda_support:false, so
 * every halo byte goes device -> host -> MPI -> host -> device. That is the
 * classic path, it is what a great many production codes still do, and it is
 * precisely the cost NVSHMEM/GPUDirect exist to remove. The ledger reports
 * the bounce explicitly so the number is never mistaken for a fabric cost.
 *
 * THE STRUCTURAL DIFFERENCE, WHICH IS THE POINT
 * ---------------------------------------------
 * One-sided RMA let the NVSHMEM version be lazy and fine-grained: a block
 * pulls exactly the spans it needs, when it needs them, and migration is a
 * remote atomic plus four remote puts issued from inside the resort kernel.
 * Two-sided messaging cannot do either. It forces:
 *   - a bulk halo exchange of whole boundary PLANES before every force and
 *     every list build, whether or not a given block would have touched them;
 *   - an explicit pack / exchange / unpack for atom migration, because there
 *     is no way to atomically claim a slot in a peer's bin from a kernel.
 * So the MPI version moves more bytes but moves them in fewer, larger,
 * better-pipelined transfers. Which wins is an empirical question, and that
 * is the question this file exists to answer.
 *
 * Addressing: each rank holds an EXTENDED slab of (nplanes + 2) z-planes --
 * [halo_lo | its own planes | halo_hi] -- so once the halo has landed, plane
 * resolution is pure pointer arithmetic and the kernels contain no
 * communication of any kind.
 *
 * Run recipe (mirrors the other two benches):
 *   mpirun -n 1 clio_md_mpi_bench --md --lattice 40 --steps 100 --rebin 10 \
 *       --temp 3.0 --cap 48 --blocks 529 --threads 128 --rowchunk 1 \
 *       --drift-tol 5e-3
 *   mpirun -n 4 clio_md_mpi_bench --md --lattice 40 ...
 *
 * At one rank every plane is owned and no halo is ever read, so this bench
 * must reproduce the NVSHMEM bench's one-PE time to within noise. That
 * agreement is the check that the transport really is the only variable.
 */

#include <mpi.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using u32 = unsigned int;
using u64 = unsigned long long;

static constexpr u32 kStride = 4;
static constexpr int kMaxSpans = 6;
static constexpr double kMeltPePerAtom = -6.7733681;

/** Device ledger. The MPI version counts what it PACKS; the bytes that
 *  actually cross MPI are counted on the host, where they happen. */
enum CtrIdx {
  kCtrMigrants = 0,      // atoms handed to a neighbouring rank
  kCtrMigrantOverflow = 1,
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
  u32 lattice = 20;
  double rho = 0.8442;
  double cutoff = 2.5;
  double skin = 0.3;
  u32 cap = 32;
  u32 blocks = 128;
  u32 threads = 64;
  u64 steps = 100;
  u64 ckpt = 0;            // checkpoint every N steps (0 = never)
  u64 rebin = 20;
  double temp = 0.0;
  u32 maxneigh = 96;
  u32 rowchunk = 4;
  int use_list = 1;
  int force_halo = 0;      // exchange the halo even at one rank
  double drift_tol = 5e-4;
  double dt = 0.005;
  int gate = 1;
  int md = 0;
  double g[3] = {0.1, -0.05, 0.02};
};

struct Geometry {
  double box = 0.0;
  double bin_edge = 0.0;
  u32 nb = 0;
  u64 nbins = 0;
  u32 cap = 0;
  u64 natoms = 0;
  u64 row_elems = 0;
  u64 plane_elems = 0;
};

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

void InitVelocity(u64 atom_id, float out[3]) {
  const double a = static_cast<double>(atom_id);
  out[0] = static_cast<float>(0.20 * std::sin(0.37 * a + 0.1));
  out[1] = static_cast<float>(0.20 * std::cos(0.53 * a + 0.7));
  out[2] = static_cast<float>(0.20 * std::sin(0.71 * a + 1.3));
}

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
          hx[e + 3] = 1.0f;
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
 * The decomposition as the kernels see it. Note what is MISSING relative to
 * the NVSHMEM sibling: no peer table, no symmetric base pointers. After the
 * halo exchange this rank's extended slab is self-contained, so the kernels
 * need nothing but their own geometry.
 */
struct Decomp {
  u32 nb = 0;
  u32 cap = 0;
  u32 npes = 1;
  u32 mype = 0;
  u32 nplanes = 0;     // planes this rank owns
  u32 z0 = 0;          // first global plane this rank owns
  int use_halo = 0;    // npes > 1, or --force-halo at one rank
  u32 halo_lo = 0;     // global plane held in extended slot 0
  u32 halo_hi = 0;     // global plane held in extended slot nplanes + 1
};

/**
 * Global plane -> element offset inside the EXTENDED slab
 * [halo_lo | owned planes | halo_hi].
 *
 * Owned planes win over halo planes, which is what makes a one-rank run
 * free: with npes == 1 every plane including the wrap-around neighbours is
 * owned, nothing is ever read out of the halo, and the exchange is skipped
 * entirely unless --force-halo asks for it.
 */
__device__ __forceinline__ u64 PlaneOff(const Decomp &d, u32 bz,
                                        u64 plane_elems) {
  if (d.use_halo) {
    if (bz == d.halo_lo) return 0;
    if (bz == d.halo_hi) return (d.nplanes + 1ull) * plane_elems;
  }
  return (static_cast<u64>(bz - d.z0) + 1ull) * plane_elems;
}

/**
 * Resolve the stencil of a chunk of rows into at most six contiguous spans.
 * The NVSHMEM sibling's StageStencil, with the communication removed: the
 * halo has already landed, so this is pointer arithmetic and nothing else.
 * Kept block-collective in shape (and it still ends synchronized) so the two
 * kernels stay line-for-line comparable.
 */
__device__ int ResolveStencil(const Decomp &d, const float *x, u32 bz, u32 y0,
                              u32 ylast, u64 row_elems, u64 plane_elems,
                              const float **sp, u32 *sbase, u32 *scnt,
                              u32 *sdz) {
  const u32 nb = d.nb;
  int nspans = 0;
  for (int dz = -1; dz <= 1; ++dz) {
    const u32 wz = (bz + nb + dz) % nb;
    const u64 poff = PlaneOff(d, wz, plane_elems);
    const int lo = static_cast<int>(y0) - 1;
    const int hi = static_cast<int>(ylast) + 1;
    u32 rl[2], rn[2];
    int nr = 0;
    if (hi - lo + 1 >= static_cast<int>(nb)) {
      rl[nr] = 0; rn[nr] = nb; ++nr;
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
      sp[nspans] = x + poff + static_cast<u64>(rl[t]) * row_elems;
      sbase[nspans] = rl[t];
      scnt[nspans] = rn[t];
      sdz[nspans] = static_cast<u32>(dz + 1);
      ++nspans;
    }
  }
  __syncthreads();
  return nspans;
}

/** Publish the nine stencil-row pointers for row `by` into shared memory.
 *  Identical to the NVSHMEM sibling. */
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

/** K3 cell-direct. Pair arithmetic copied verbatim from the NVSHMEM bench. */
__global__ void ForceKernel(Decomp d, const float *x, float *f, float box,
                            float cutoff, int eflag, double *acc,
                            u64 row_elems, u64 plane_elems, u32 rowchunk) {
  extern __shared__ char smem[];
  double *red = reinterpret_cast<double *>(smem);
  const float **s_qptr =
      reinterpret_cast<const float **>(smem + blockDim.x * sizeof(double));
  const u32 nb = d.nb, cap = d.cap;
  const u64 islots = static_cast<u64>(nb) * cap;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
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
    const int nspans = ResolveStencil(d, x, bz, y0, ylast, row_elems,
                                      plane_elems, sp, sbase, scnt, sdz);

    for (u32 by = y0; by <= ylast; ++by) {
      PublishRow(s_qptr, by, nb, nspans, sp, sbase, scnt, sdz, row_elems);
      float *const fp = f + (static_cast<u64>(lz) * nb + by) * row_elems;
      for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) fp[e] = 0.0f;
      __syncthreads();
      const float *const ip_row = s_qptr[4];
      for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
        const float *const ip = ip_row + s * kStride;
        if (ip[3] < 0.0f) continue;
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
              if (q == 4 && jbx == bx && sj == s % cap) continue;
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

/** K2c: build the Verlet list for this rank's rows. */
__global__ void BuildListKernel(Decomp d, const float *x, int *nl, float rlist,
                                u32 maxneigh, u32 *cnt, int *err,
                                u64 row_elems, u64 plane_elems, u32 rowchunk,
                                float box) {
  extern __shared__ char smem[];
  const float **s_qptr = reinterpret_cast<const float **>(smem);
  const u32 nb = d.nb, cap = d.cap;
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float r2list = rlist * rlist;
  const float halfL = 0.5f * box;

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
    const int nspans = ResolveStencil(d, x, bz, y0, ylast, row_elems,
                                      plane_elems, sp, sbase, scnt, sdz);

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
              if (n >= maxneigh) {
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

/** K3-list: the force pass streaming the padded Verlet list. */
__global__ void ListForceKernel(Decomp d, const float *x, float *f,
                                const int *nl, const u32 *cnt, float box,
                                float cutoff, u32 maxneigh, int eflag,
                                double *acc, u64 row_elems, u64 plane_elems,
                                u32 rowchunk, int nocompute) {
  extern __shared__ char smem[];
  double *red = reinterpret_cast<double *>(smem);
  const float **s_qptr =
      reinterpret_cast<const float **>(smem + blockDim.x * sizeof(double));
  const u32 nb = d.nb, cap = d.cap;
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
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
    const int nspans = ResolveStencil(d, x, bz, y0, ylast, row_elems,
                                      plane_elems, sp, sbase, scnt, sdz);

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
 * K2a: wrap, re-bin, and claim a slot -- but only for atoms that stay on
 * this rank. THE TRANSPORT DIFFERENCE, in one kernel: the NVSHMEM version
 * claimed a slot in the owner's bin with a remote atomic and put the atom
 * straight there. Two-sided messaging has no such verb, so an atom leaving
 * the slab is PACKED into a per-direction outbox here and unpacked by the
 * receiving rank after an MPI exchange. Migrants are recorded as
 * (global bin, x4, v4) because the destination slot cannot be known until
 * the owner claims it.
 */
__global__ void RebinKernel(Decomp d, float *x, const float *v, float box,
                            u32 *bincnt, int *dest_pe, u32 *dest_slot,
                            int *err, u64 local_slots, float *out_lo,
                            float *out_hi, u32 *nout_lo, u32 *nout_hi,
                            u32 out_cap, unsigned long long *ctr) {
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
    const u32 rel = bz - d.z0;
    if (rel < d.nplanes) {                    // stays here
      const u64 lbin = (static_cast<u64>(rel) * nb + by) * nb + bx;
      const u32 slot = atomicAdd(&bincnt[lbin], 1u);
      if (slot >= cap) {
        *err = 1;
        dest_pe[s] = -1;
        continue;
      }
      dest_pe[s] = static_cast<int>(d.mype);
      dest_slot[s] = static_cast<u32>(lbin * cap + slot);
      continue;
    }
    // Leaves the slab. An atom moves at most one bin per rebuild window (the
    // skin rule), so the destination is always an ADJACENT rank -- down when
    // it fell off the low face, up otherwise.
    const bool down = (bz == (d.z0 + d.nb - 1) % d.nb);
    u32 *const nout = down ? nout_lo : nout_hi;
    float *const out = down ? out_lo : out_hi;
    const u32 idx = atomicAdd(nout, 1u);
    if (idx >= out_cap) {
      atomicAdd(&ctr[kCtrMigrantOverflow], 1ull);
      *err = 2;
      dest_pe[s] = -1;
      continue;
    }
    const u64 gbin = (static_cast<u64>(bz) * nb + by) * nb + bx;
    float *const rec = out + static_cast<u64>(idx) * 9;
    rec[0] = static_cast<float>(gbin);   // exact: gbin < 2^24 is enforced host-side
    rec[1] = px; rec[2] = py; rec[3] = pz; rec[4] = x[e + 3];
    rec[5] = v[e + 0]; rec[6] = v[e + 1]; rec[7] = v[e + 2];
    rec[8] = 0.0f;
    dest_pe[s] = -1;                     // not written by the local scatter
    atomicAdd(&ctr[kCtrMigrants], 1ull);
  }
}

/** Unpack received migrants: claim a local slot and write both arrays. */
__global__ void UnpackKernel(Decomp d, const float *in, u32 n, u32 *bincnt,
                             float *dstx, float *dstv, int *err) {
  const u32 nb = d.nb, cap = d.cap;
  for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const float *const rec = in + static_cast<u64>(i) * 9;
    const u64 gbin = static_cast<u64>(rec[0]);
    const u32 bz = static_cast<u32>(gbin / (static_cast<u64>(nb) * nb));
    const u32 by = static_cast<u32>((gbin / nb) % nb);
    const u32 bx = static_cast<u32>(gbin % nb);
    const u32 rel = bz - d.z0;
    if (rel >= d.nplanes) {   // should be impossible; refuse rather than corrupt
      *err = 3;
      continue;
    }
    const u64 lbin = (static_cast<u64>(rel) * nb + by) * nb + bx;
    const u32 slot = atomicAdd(&bincnt[lbin], 1u);
    if (slot >= cap) {
      *err = 1;
      continue;
    }
    const u64 de = (lbin * cap + slot) * kStride;
    dstx[de + 0] = rec[1]; dstx[de + 1] = rec[2];
    dstx[de + 2] = rec[3]; dstx[de + 3] = rec[4];
    dstv[de + 0] = rec[5]; dstv[de + 1] = rec[6];
    dstv[de + 2] = rec[7]; dstv[de + 3] = 0.0f;
  }
}

/** K2b: apply the permutation for atoms that stayed. All stores are local;
 *  unlike the NVSHMEM sibling there is no remote-put branch, because the
 *  atoms that would have taken it are in the outboxes instead. */
__global__ void ScatterKernel(const float *src, const float *srcx, float *dst,
                              const int *dest_pe, const u32 *dest_slot,
                              int keep_w, u64 local_slots) {
  for (u64 s = blockIdx.x * blockDim.x + threadIdx.x; s < local_slots;
       s += static_cast<u64>(gridDim.x) * blockDim.x) {
    const u64 e = s * kStride;
    if (srcx[e + 3] < 0.0f) continue;
    if (dest_pe[s] < 0) continue;
    const u64 de = static_cast<u64>(dest_slot[s]) * kStride;
    dst[de + 0] = src[e + 0];
    dst[de + 1] = src[e + 1];
    dst[de + 2] = src[e + 2];
    dst[de + 3] = keep_w ? src[e + 3] : 0.0f;
  }
}

__global__ void SentinelKernel(float *dst, u64 local_slots) {
  for (u64 s = blockIdx.x * blockDim.x + threadIdx.x; s < local_slots;
       s += static_cast<u64>(gridDim.x) * blockDim.x) {
    dst[s * kStride + 3] = -1.0f;
  }
}

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
    ke += 0.5 * (vx * vx + vy * vy + vz * vz);
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
    else if (want("--rebin")) a.rebin = static_cast<u64>(atol(argv[++i]));
    else if (want("--temp")) a.temp = atof(argv[++i]);
    else if (want("--drift-tol")) a.drift_tol = atof(argv[++i]);
    else if (want("--maxneigh")) a.maxneigh = static_cast<u32>(atoi(argv[++i]));
    else if (want("--rowchunk")) a.rowchunk = static_cast<u32>(atoi(argv[++i]));
    else if (want("--dt")) a.dt = atof(argv[++i]);
    else if (std::strcmp(argv[i], "--no-list") == 0) a.use_list = 0;
    else if (std::strcmp(argv[i], "--force-halo") == 0) a.force_halo = 1;
    else if (std::strcmp(argv[i], "--no-gate") == 0) a.gate = 0;
    else if (std::strcmp(argv[i], "--md") == 0) a.md = 1;
    else {
      std::fprintf(stderr, "unknown arg %s\n", argv[i]);
      return 1;
    }
  }

  MPI_Init(&argc, &argv);
  int mype = 0, npes = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mype);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  int ndev = 0;
  MD_CUDA_CHECK(cudaGetDeviceCount(&ndev));
  MD_CUDA_CHECK(cudaSetDevice(mype % (ndev > 0 ? ndev : 1)));
  const bool root = (mype == 0);

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
  // Migrant records carry the destination bin as a float. Exact only while
  // the bin index fits the 24-bit mantissa; checked rather than assumed.
  if (g.nbins >= (1ull << 24)) {
    if (root) {
      std::fprintf(stderr,
                   "%llu bins exceeds the 2^24 exactly-representable range "
                   "used by the migrant record\n", g.nbins);
    }
    return 1;
  }
  if (g.nb / static_cast<u32>(npes) < 3 && npes > 1) {
    if (root) {
      std::fprintf(stderr,
                   "each rank needs at least 3 z-planes (%u planes / %d "
                   "ranks); raise --lattice or lower the rank count\n",
                   g.nb, npes);
    }
    return 1;
  }

  std::vector<u32> h_plane_z0(npes + 1);
  for (int p = 0; p <= npes; ++p) {
    h_plane_z0[p] = static_cast<u32>(static_cast<u64>(p) * g.nb / npes);
  }
  const u32 myz0 = h_plane_z0[mype];
  const u32 mynplanes = h_plane_z0[mype + 1] - myz0;
  const int rank_down = (mype - 1 + npes) % npes;
  const int rank_up = (mype + 1) % npes;

  // The EXTENDED slab: [halo_lo | owned planes | halo_hi].
  const u32 ext_planes = mynplanes + 2;
  const u64 ext_elems = static_cast<u64>(ext_planes) * g.plane_elems;
  const u64 own_off = g.plane_elems;                 // owned data starts here
  const u64 local_slots = static_cast<u64>(mynplanes) * g.nb * g.nb * g.cap;
  const u64 local_elems = local_slots * kStride;
  const u64 islots = static_cast<u64>(g.nb) * g.cap;
  const u64 rowlist = islots * a.maxneigh;
  const u64 nl_elems = static_cast<u64>(mynplanes) * g.nb * rowlist;
  const u32 out_cap = static_cast<u32>(g.nb) * g.nb * g.cap;   // one plane

  const double xvf_mb =
      (ext_elems + 2.0 * local_elems) * sizeof(float) / 1048576.0;
  const double pp_mb = 2.0 * local_elems * sizeof(float) / 1048576.0;
  const double nl_mb = nl_elems * sizeof(int) / 1048576.0;
  const double total_mb = xvf_mb + (a.md ? pp_mb : 0.0) +
                          (a.md && a.use_list ? nl_mb : 0.0);
  if (root) {
    std::printf(
        "eternia-MD / MPI baseline (two-sided, host-staged halo)\n"
        "  atoms=%llu box=%.4f bins=%u^3 cap=%u  ranks=%d (planes/rank=%u)\n"
        "  blocks=%u threads=%u rowchunk=%u list=%s halo=%s\n"
        "  per-rank VRAM: x(+2 halo planes)/v/f %.1f MB + resort %.1f MB + "
        "list %.1f MB = %.1f MB\n",
        g.natoms, g.box, g.nb, g.cap, npes, mynplanes, a.blocks, a.threads,
        a.rowchunk, a.use_list ? "verlet" : "cell-direct",
        (npes > 1 || a.force_halo) ? "exchanged every step"
                                   : "not needed (1 rank)",
        xvf_mb, a.md ? pp_mb : 0.0, (a.md && a.use_list) ? nl_mb : 0.0,
        total_mb);
  }

  std::vector<float> hx, hv;
  if (!BuildInitialState(a, g, hx, hv)) return 1;
  if (a.temp > 0.0) {
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

  // x lives in the extended slab; v and f only ever cover owned planes.
  float *dx_ext = nullptr, *dv = nullptr, *df = nullptr;
  MD_CUDA_CHECK(cudaMalloc(&dx_ext, ext_elems * sizeof(float)));
  MD_CUDA_CHECK(cudaMalloc(&dv, local_elems * sizeof(float)));
  MD_CUDA_CHECK(cudaMalloc(&df, local_elems * sizeof(float)));
  float *dx = dx_ext + own_off;
  {
    std::vector<float> sx(ext_elems, 0.0f), sv(local_elems, 0.0f);
    for (u64 s = 0; s < ext_elems / kStride; ++s) sx[s * kStride + 3] = -1.0f;
    for (u32 lz = 0; lz < mynplanes; ++lz) {
      const u64 gsrc = static_cast<u64>(myz0 + lz) * g.plane_elems;
      std::memcpy(&sx[own_off + static_cast<u64>(lz) * g.plane_elems],
                  &hx[gsrc], g.plane_elems * sizeof(float));
      std::memcpy(&sv[static_cast<u64>(lz) * g.plane_elems], &hv[gsrc],
                  g.plane_elems * sizeof(float));
    }
    MD_CUDA_CHECK(cudaMemcpy(dx_ext, sx.data(), ext_elems * sizeof(float),
                             cudaMemcpyHostToDevice));
    MD_CUDA_CHECK(cudaMemcpy(dv, sv.data(), local_elems * sizeof(float),
                             cudaMemcpyHostToDevice));
  }
  MD_CUDA_CHECK(cudaMemset(df, 0, local_elems * sizeof(float)));

  Decomp d;
  d.nb = g.nb;
  d.cap = g.cap;
  d.npes = static_cast<u32>(npes);
  d.mype = static_cast<u32>(mype);
  d.nplanes = mynplanes;
  d.z0 = myz0;
  d.use_halo = (npes > 1) || a.force_halo;
  d.halo_lo = (myz0 + g.nb - 1) % g.nb;
  d.halo_hi = (myz0 + mynplanes) % g.nb;

  unsigned long long *d_ctr = nullptr;
  MD_CUDA_CHECK(cudaMalloc(&d_ctr, kCtrNumCtrs * sizeof(unsigned long long)));
  MD_CUDA_CHECK(cudaMemset(d_ctr, 0, kCtrNumCtrs * sizeof(unsigned long long)));
  double *d_thermo = nullptr;
  MD_CUDA_CHECK(cudaMalloc(&d_thermo, 4 * sizeof(double)));

  // Pinned staging for the halo. This is the whole substrate difference:
  // NVSHMEM read the peer's memory directly from a kernel; here the bytes
  // must land in host memory before MPI can be told about them.
  float *h_send_lo = nullptr, *h_send_hi = nullptr;
  float *h_recv_lo = nullptr, *h_recv_hi = nullptr;
  MD_CUDA_CHECK(cudaMallocHost(&h_send_lo, g.plane_elems * sizeof(float)));
  MD_CUDA_CHECK(cudaMallocHost(&h_send_hi, g.plane_elems * sizeof(float)));
  MD_CUDA_CHECK(cudaMallocHost(&h_recv_lo, g.plane_elems * sizeof(float)));
  MD_CUDA_CHECK(cudaMallocHost(&h_recv_hi, g.plane_elems * sizeof(float)));

  double t_halo = 0.0, halo_bytes = 0.0;
  u64 n_halo = 0;
  const bool need_halo = (npes > 1) || a.force_halo;

  /**
   * Exchange the two boundary planes of x. My FIRST owned plane is what the
   * down neighbour needs as its halo_hi; my LAST is what the up neighbour
   * needs as its halo_lo.
   */
  auto exchange_halo = [&]() {
    if (!need_halo) return;
    const double t = NowMs();
    const u64 pb = g.plane_elems * sizeof(float);
    const float *const first = dx;
    const float *const last =
        dx + static_cast<u64>(mynplanes - 1) * g.plane_elems;
    MD_CUDA_CHECK(cudaMemcpy(h_send_hi, first, pb, cudaMemcpyDeviceToHost));
    MD_CUDA_CHECK(cudaMemcpy(h_send_lo, last, pb, cudaMemcpyDeviceToHost));
    MPI_Sendrecv(h_send_lo, static_cast<int>(g.plane_elems), MPI_FLOAT,
                 rank_up, 11, h_recv_lo, static_cast<int>(g.plane_elems),
                 MPI_FLOAT, rank_down, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_hi, static_cast<int>(g.plane_elems), MPI_FLOAT,
                 rank_down, 12, h_recv_hi, static_cast<int>(g.plane_elems),
                 MPI_FLOAT, rank_up, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MD_CUDA_CHECK(cudaMemcpy(dx_ext, h_recv_lo, pb, cudaMemcpyHostToDevice));
    MD_CUDA_CHECK(cudaMemcpy(dx_ext + static_cast<u64>(mynplanes + 1) *
                                          g.plane_elems,
                             h_recv_hi, pb, cudaMemcpyHostToDevice));
    t_halo += NowMs() - t;
    halo_bytes += 2.0 * pb;
    ++n_halo;
  };

  const u32 smem_thermo = a.threads * sizeof(double);
  const u32 smem_force = a.threads * sizeof(double) + 9 * sizeof(void *);
  const u32 smem_build = 9 * sizeof(void *);
  const float fbox = static_cast<float>(g.box);
  const float fcut = static_cast<float>(a.cutoff);
  const float frlist = static_cast<float>(a.cutoff + a.skin);
  const float fdt = static_cast<float>(a.dt);

  auto sum_doubles = [&](double *host, int n) {
    MPI_Allreduce(MPI_IN_PLACE, host, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  };

  if (a.md) {
    float *dx2_ext = nullptr, *dv2 = nullptr;
    MD_CUDA_CHECK(cudaMalloc(&dx2_ext, ext_elems * sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&dv2, local_elems * sizeof(float)));
    float *dx2 = dx2_ext + own_off;
    u32 *d_bincnt = nullptr, *d_dest_slot = nullptr, *d_cnt = nullptr;
    int *d_dest_pe = nullptr, *d_err = nullptr, *d_nl = nullptr;
    double *d_acc = nullptr;
    float *d_out_lo = nullptr, *d_out_hi = nullptr, *d_in = nullptr;
    u32 *d_nout_lo = nullptr, *d_nout_hi = nullptr;
    MD_CUDA_CHECK(cudaMalloc(&d_bincnt, static_cast<u64>(mynplanes) * g.nb *
                                            g.nb * sizeof(u32)));
    MD_CUDA_CHECK(cudaMalloc(&d_dest_pe, local_slots * sizeof(int)));
    MD_CUDA_CHECK(cudaMalloc(&d_dest_slot, local_slots * sizeof(u32)));
    MD_CUDA_CHECK(cudaMalloc(&d_cnt, local_slots * sizeof(u32)));
    MD_CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));
    MD_CUDA_CHECK(cudaMalloc(&d_acc, 3 * sizeof(double)));
    MD_CUDA_CHECK(cudaMalloc(&d_out_lo, static_cast<u64>(out_cap) * 9 *
                                            sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&d_out_hi, static_cast<u64>(out_cap) * 9 *
                                            sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&d_in, static_cast<u64>(out_cap) * 9 *
                                        sizeof(float)));
    MD_CUDA_CHECK(cudaMalloc(&d_nout_lo, sizeof(u32)));
    MD_CUDA_CHECK(cudaMalloc(&d_nout_hi, sizeof(u32)));
    if (a.use_list) {
      MD_CUDA_CHECK(cudaMalloc(&d_nl, nl_elems * sizeof(int)));
      MD_CUDA_CHECK(cudaMemset(d_nl, 0, nl_elems * sizeof(int)));
    }
    std::vector<float> h_mig(static_cast<u64>(out_cap) * 9);
    std::vector<float> h_mig_in(static_cast<u64>(out_cap) * 9);

    double acc[3] = {0, 0, 0};
    double t_force = 0.0, t_kick = 0.0, t_resort = 0.0, t_build = 0.0;
    double mig_bytes = 0.0;
    const int nocompute = std::getenv("MD_NOCOMPUTE") != nullptr ? 1 : 0;

    auto build_list = [&]() -> bool {
      const double t = NowMs();
      MD_CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
      BuildListKernel<<<a.blocks, a.threads, smem_build>>>(
          d, dx_ext, d_nl, frlist, a.maxneigh, d_cnt, d_err, g.row_elems,
          g.plane_elems, a.rowchunk, fbox);
      MD_CUDA_CHECK(cudaGetLastError());
      MD_CUDA_CHECK(cudaDeviceSynchronize());
      int err = 0;
      MD_CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(int),
                               cudaMemcpyDeviceToHost));
      double e = err;
      sum_doubles(&e, 1);
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
      if (a.use_list) {
        ListForceKernel<<<a.blocks, a.threads, smem_force>>>(
            d, dx_ext, df, d_nl, d_cnt, fbox, fcut, a.maxneigh, eflag, d_acc,
            g.row_elems, g.plane_elems, a.rowchunk, nocompute);
      } else {
        ForceKernel<<<a.blocks, a.threads, smem_force>>>(
            d, dx_ext, df, fbox, fcut, eflag, d_acc, g.row_elems,
            g.plane_elems, a.rowchunk);
      }
      MD_CUDA_CHECK(cudaGetLastError());
      MD_CUDA_CHECK(cudaDeviceSynchronize());
      if (eflag) {
        MD_CUDA_CHECK(cudaMemcpy(acc, d_acc, 3 * sizeof(double),
                                 cudaMemcpyDeviceToHost));
        sum_doubles(acc, 3);
      }
      t_force += NowMs() - t;
    };
    auto kick = [&](int drift) {
      const double t = NowMs();
      MDIntegrateKernel<<<a.blocks, a.threads>>>(dx, dv, df, fdt, drift,
                                                 local_slots);
      MD_CUDA_CHECK(cudaGetLastError());
      MD_CUDA_CHECK(cudaDeviceSynchronize());
      t_kick += NowMs() - t;
    };
    auto thermo_ke = [&]() -> double {
      MD_CUDA_CHECK(cudaMemset(d_thermo, 0, 4 * sizeof(double)));
      ThermoKernel<<<a.blocks, a.threads, smem_thermo>>>(dx, dv, d_thermo,
                                                         local_slots);
      MD_CUDA_CHECK(cudaGetLastError());
      MD_CUDA_CHECK(cudaDeviceSynchronize());
      double t4[4];
      MD_CUDA_CHECK(cudaMemcpy(t4, d_thermo, sizeof(t4),
                               cudaMemcpyDeviceToHost));
      sum_doubles(t4, 4);
      return t4[0];
    };
    /**
     * K2 with two-sided migration: rebin locally and pack leavers, exchange
     * the outboxes with both neighbours, unpack into claimed slots. Three
     * MPI messages per rebuild against the NVSHMEM version's zero (it did
     * the whole thing with remote atomics and puts from inside the kernel).
     */
    auto resort = [&]() -> bool {
      const double t = NowMs();
      MD_CUDA_CHECK(cudaMemset(d_bincnt, 0, static_cast<u64>(mynplanes) *
                                                g.nb * g.nb * sizeof(u32)));
      MD_CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
      MD_CUDA_CHECK(cudaMemset(d_nout_lo, 0, sizeof(u32)));
      MD_CUDA_CHECK(cudaMemset(d_nout_hi, 0, sizeof(u32)));
      SentinelKernel<<<a.blocks, a.threads>>>(dx2, local_slots);
      SentinelKernel<<<a.blocks, a.threads>>>(dv2, local_slots);
      RebinKernel<<<a.blocks, a.threads>>>(
          d, dx, dv, fbox, d_bincnt, d_dest_pe, d_dest_slot, d_err,
          local_slots, d_out_lo, d_out_hi, d_nout_lo, d_nout_hi, out_cap,
          d_ctr);
      MD_CUDA_CHECK(cudaGetLastError());
      MD_CUDA_CHECK(cudaDeviceSynchronize());
      int err = 0;
      MD_CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(int),
                               cudaMemcpyDeviceToHost));
      double e = err;
      sum_doubles(&e, 1);
      if (e != 0.0) {
        if (root) {
          std::fprintf(stderr,
                       "resort: bin overflow or migrant-buffer overflow "
                       "-- raise --cap\n");
        }
        return false;
      }
      ScatterKernel<<<a.blocks, a.threads>>>(dx, dx, dx2, d_dest_pe,
                                             d_dest_slot, 1, local_slots);
      ScatterKernel<<<a.blocks, a.threads>>>(dv, dx, dv2, d_dest_pe,
                                             d_dest_slot, 0, local_slots);
      MD_CUDA_CHECK(cudaGetLastError());

      if (npes > 1) {
        u32 n_lo = 0, n_hi = 0;
        MD_CUDA_CHECK(cudaMemcpy(&n_lo, d_nout_lo, sizeof(u32),
                                 cudaMemcpyDeviceToHost));
        MD_CUDA_CHECK(cudaMemcpy(&n_hi, d_nout_hi, sizeof(u32),
                                 cudaMemcpyDeviceToHost));
        // Down-going atoms leave through the low face and land on rank_down;
        // up-going ones on rank_up. Counts first, then payload.
        u32 r_from_up = 0, r_from_down = 0;
        MPI_Sendrecv(&n_lo, 1, MPI_UNSIGNED, rank_down, 21, &r_from_up, 1,
                     MPI_UNSIGNED, rank_up, 21, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(&n_hi, 1, MPI_UNSIGNED, rank_up, 22, &r_from_down, 1,
                     MPI_UNSIGNED, rank_down, 22, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        const u32 nrecv = r_from_up + r_from_down;
        if (nrecv > out_cap) {
          std::fprintf(stderr, "migrant inbox overflow (%u > %u)\n", nrecv,
                       out_cap);
          return false;
        }
        if (n_lo) {
          MD_CUDA_CHECK(cudaMemcpy(h_mig.data(), d_out_lo,
                                   static_cast<u64>(n_lo) * 9 * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        }
        MPI_Sendrecv(h_mig.data(), static_cast<int>(n_lo * 9), MPI_FLOAT,
                     rank_down, 23, h_mig_in.data(),
                     static_cast<int>(r_from_up * 9), MPI_FLOAT, rank_up, 23,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n_hi) {
          MD_CUDA_CHECK(cudaMemcpy(h_mig.data(), d_out_hi,
                                   static_cast<u64>(n_hi) * 9 * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        }
        MPI_Sendrecv(h_mig.data(), static_cast<int>(n_hi * 9), MPI_FLOAT,
                     rank_up, 24, h_mig_in.data() + r_from_up * 9,
                     static_cast<int>(r_from_down * 9), MPI_FLOAT, rank_down,
                     24, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mig_bytes += static_cast<double>(n_lo + n_hi) * 9 * sizeof(float);
        if (nrecv) {
          MD_CUDA_CHECK(cudaMemcpy(d_in, h_mig_in.data(),
                                   static_cast<u64>(nrecv) * 9 * sizeof(float),
                                   cudaMemcpyHostToDevice));
          UnpackKernel<<<a.blocks, a.threads>>>(d, d_in, nrecv, d_bincnt, dx2,
                                                dv2, d_err);
          MD_CUDA_CHECK(cudaGetLastError());
        }
      }
      MD_CUDA_CHECK(cudaDeviceSynchronize());
      MD_CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(int),
                               cudaMemcpyDeviceToHost));
      e = err;
      sum_doubles(&e, 1);
      if (e != 0.0) {
        if (root) std::fprintf(stderr, "resort: unpack refused (err)\n");
        return false;
      }
      std::swap(dx_ext, dx2_ext);
      std::swap(dx, dx2);
      std::swap(dv, dv2);
      t_resort += NowMs() - t;
      return true;
    };

    exchange_halo();
    if (a.use_list && !build_list()) return 1;
    force(1);
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

    bool resort_ok = true;
    if (a.rebin != 0) {
      const double pe_before = acc[0];
      if (!resort()) return 1;
      exchange_halo();
      if (a.use_list && !build_list()) return 1;
      force(1);
      const double rel = std::fabs(acc[0] - pe_before) / std::fabs(pe_before);
      resort_ok = rel < 1e-6;
      if (root) {
        std::printf("  RESORT GATE: %s (PE %.6f -> %.6f, rel=%.2e)\n",
                    resort_ok ? "PASS" : "FAIL", pe_before, acc[0], rel);
      }
    }

    const double ke0 = thermo_ke();
    const double e0 = acc[0] + ke0;
    t_halo = 0.0; halo_bytes = 0.0; n_halo = 0; mig_bytes = 0.0;
    MD_CUDA_CHECK(cudaMemset(d_ctr, 0,
                             kCtrNumCtrs * sizeof(unsigned long long)));
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = NowMs();
    // ---- CHECKPOINT ------------------------------------------------------
    // What a non-persistent substrate must do to survive a crash: get the
    // simulation state off the GPU. x and v only -- forces and the Verlet
    // list are derived and a restart rebuilds them, which is what LAMMPS
    // restart files carry too.
    //
    // The destination is PINNED HOST DRAM, deliberately: the paged vector
    // this is being compared against persists into a CTE tier stack whose
    // only configured target is a RAM tier, so staging to host DRAM lands
    // the bytes in the same storage class. Anything else would be comparing
    // two destinations, not two substrates.
    float *h_ckpt = nullptr;
    double t_ckpt = 0.0;
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
      ++n_ckpt;
    };
    for (u64 step = 0; step < a.steps; ++step) {
      kick(1);
      if (a.rebin != 0 && step != 0 && step % a.rebin == 0) {
        if (!resort()) return 1;
        exchange_halo();
        if (a.use_list && !build_list()) return 1;
      }
      exchange_halo();   // positions moved in the kick
      force(0);
      kick(0);
     if (a.ckpt != 0 && (step + 1) % a.ckpt == 0) checkpoint();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double run_ms = NowMs() - t0;
    exchange_halo();
    force(1);
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
          "  checkpoints: %llu x %.1f MB = %.2f GB staged to host DRAM | "
          "%.1f ms each (%.2f GB/s) | %.1f ms total = %.1f%% on top of the "
          "%.1f ms run\n",
          (unsigned long long)n_ckpt, mb,
          mb * static_cast<double>(n_ckpt) / 1024.0, t_ckpt / n_ckpt,
          (mb / 1024.0) / (t_ckpt / n_ckpt / 1000.0), t_ckpt,
          100.0 * t_ckpt / run_ms, run_ms);
    }
    if (h_ckpt != nullptr) cudaFreeHost(h_ckpt);

    unsigned long long ctr[kCtrNumCtrs] = {0};
    MD_CUDA_CHECK(cudaMemcpy(ctr, d_ctr, sizeof(ctr), cudaMemcpyDeviceToHost));
    double led[4] = {halo_bytes, mig_bytes,
                     static_cast<double>(ctr[kCtrMigrants]), t_halo};
    sum_doubles(led, 4);
    if (root) {
      std::printf(
          "  NVE %llu steps: E0=%.6f En=%.6f drift=%.2e | KE %.3f -> %.3f\n"
          "  NVE GATE: %s\n"
          "  %llu steps in %.1f ms (%.3f ms/step, %.1f Matom-steps/s)\n",
          a.steps, e0, e_n, e_drift, ke0, ke_n, nve_ok ? "PASS" : "FAIL",
          a.steps, run_ms, run_ms / a.steps,
          static_cast<double>(g.natoms) * a.steps / run_ms / 1000.0);
      std::printf(
          "  comm ledger (all ranks): halo %.2f GB in %llu exchanges "
          "(%.2f MB/step) | migrants %.0f atoms, %.2f MB | halo time %.1f ms "
          "summed over ranks = %.1f%% of a rank's run\n"
          "  substrate: two-sided MPI, host-staged (D2H -> MPI -> H2D); "
          "every halo byte crosses PCIe TWICE because this OpenMPI is not "
          "CUDA-aware\n",
          led[0] / 1073741824.0, n_halo, led[0] / a.steps / 1048576.0, led[2],
          led[1] / 1048576.0, led[3], 100.0 * (led[3] / npes) / run_ms);
      std::printf("  phases (total ms): force=%.1f kick=%.1f resort=%.1f "
                  "build=%.1f halo=%.1f\n", t_force, t_kick, t_resort, t_build,
                  t_halo);
    }
    MPI_Finalize();
    return (statics_ok && nve_ok && resort_ok) ? 0 : 1;
  }

  // ---- the ballistic run --------------------------------------------------
  const float fgx = static_cast<float>(a.g[0]);
  const float fgy = static_cast<float>(a.g[1]);
  const float fgz = static_cast<float>(a.g[2]);
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  for (u64 step = 0; step < a.steps; ++step) {
    IntegrateKernel<<<a.blocks, a.threads>>>(dx, dv, fdt, fgx, fgy, fgz, 1,
                                             local_slots);
    IntegrateKernel<<<a.blocks, a.threads>>>(dx, dv, fdt, fgx, fgy, fgz, 0,
                                             local_slots);
  }
  MD_CUDA_CHECK(cudaGetLastError());
  MD_CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  const double run_ms = NowMs() - t0;

  double thermo[4] = {0, 0, 0, 0};
  MD_CUDA_CHECK(cudaMemset(d_thermo, 0, 4 * sizeof(double)));
  ThermoKernel<<<a.blocks, a.threads, smem_thermo>>>(dx, dv, d_thermo,
                                                     local_slots);
  MD_CUDA_CHECK(cudaDeviceSynchronize());
  MD_CUDA_CHECK(cudaMemcpy(thermo, d_thermo, sizeof(thermo),
                           cudaMemcpyDeviceToHost));
  sum_doubles(thermo, 4);

  int rc = 0;
  if (a.gate) {
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
    sum_doubles(gate, 7);
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
  MPI_Finalize();
  return rc;
}
