/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * eternia-MD, Aurora baseline editions (MPI / oneCCL / Intel SHMEM).
 *
 * The CUDA MPI edition's decomposition, ported: each rank holds an EXTENDED
 * slab of (nplanes + 2) z-planes -- [halo_lo | its own planes | halo_hi] --
 * so once the halo has landed, plane resolution is pointer arithmetic and
 * the kernels contain no communication. The halo planes and the migrant
 * outboxes of the periodic resort move through the substrate (gv_comm.h):
 * GPU-aware MPI_Sendrecv, ccl::send/recv, or an ISHMEM put into the peer's
 * symmetric slab. Migrant COUNTS are one host integer each way in every
 * edition (control traffic, not the data plane). Float precision, bin-major
 * padded layout, ghost-free minimum image, full list / newton off, the
 * transposed-padded Verlet list with (q << 16) | slot entries, the chunked
 * row decomposition, and all four gates are the CUDA editions', so the
 * step-0 PE and the exact pair count are the equivalence check across
 * substrates and machines.
 *
 * Checkpoints (--ckpt N, --ckpt-dir D) stage x and v to pinned host memory
 * and, with a directory, write and fsync one restart-shaped file each:
 * the "direct" arms of the persistence study.
 *
 * Run recipe:
 *   mpiexec -n 4 --ppn 1 clio_lammps_md_<sub>_bench --md --lattice 40
 *       --steps 20 --rebin 10 --temp 3.0 --cap 48 --blocks 512 --threads 128
 * Without --md: the ballistic run and its bitwise gate.
 */

#include "gv_comm.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using gvc::u32;
using gvc::u64;

namespace {

constexpr u32 kStride = 4;
constexpr int kMaxSpans = 6;
constexpr double kMeltPePerAtom = -6.7733681;
constexpr int kCtrMigrants = 0;
constexpr int kCtrMigrantOverflow = 1;
constexpr int kCtrNumCtrs = 8;

struct Args {
  u32 lattice = 20;
  double rho = 0.8442;
  double cutoff = 2.5;
  double skin = 0.3;
  u32 cap = 32;
  u32 blocks = 128;
  u32 threads = 64;
  u64 steps = 100;
  u64 ckpt = 0;
  std::string ckpt_dir;
  u64 rebin = 20;
  double temp = 0.0;
  u32 maxneigh = 96;
  u32 rowchunk = 4;
  int use_list = 1;
  int force_halo = 0;
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

void InitVelocity(u64 atom_id, float out[3]) {
  const double a = static_cast<double>(atom_id);
  out[0] = static_cast<float>(0.20 * std::sin(0.37 * a + 0.1));
  out[1] = static_cast<float>(0.20 * std::cos(0.53 * a + 0.7));
  out[2] = static_cast<float>(0.20 * std::sin(0.71 * a + 1.3));
}

/** The FCC melt deck, bin-major padded, identical on every substrate. */
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
          hv[e + 0] = vv[0]; hv[e + 1] = vv[1]; hv[e + 2] = vv[2];
          hv[e + 3] = 0.0f;
          ++atom_id;
        }
      }
    }
  }
  return atom_id == g.natoms;
}

/** Host double-precision reference for the step-0 statics. */
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

/** The decomposition as the kernels see it. */
struct Decomp {
  u32 nb = 0;
  u32 cap = 0;
  u32 npes = 1;
  u32 mype = 0;
  u32 nplanes = 0;
  u32 z0 = 0;
  int use_halo = 0;
  u32 halo_lo = 0;
  u32 halo_hi = 0;
  u32 hi_slot = 0;     // extended-slab slot of halo_hi: max_planes + 1
};

/** Global plane -> element offset inside the EXTENDED slab. Owned planes
 *  win over halo planes, which is what makes a one-rank run free. */
inline u64 PlaneOff(const Decomp &d, u32 bz, u64 plane_elems) {
  if (d.use_halo) {
    if (bz == d.halo_lo) return 0;
    if (bz == d.halo_hi) return static_cast<u64>(d.hi_slot) * plane_elems;
  }
  return (static_cast<u64>(bz - d.z0) + 1ull) * plane_elems;
}

/** Resolve the stencil of a chunk of rows into at most six contiguous
 *  spans of the extended slab. Ends group-synchronised. */
inline int ResolveStencil(const Decomp &d, const float *x, u32 bz, u32 y0,
                          u32 ylast, u64 row_elems, u64 plane_elems,
                          const float **sp, u32 *sbase, u32 *scnt, u32 *sdz,
                          sycl::nd_item<1> it) {
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
  sycl::group_barrier(it.get_group());
  return nspans;
}

/** Publish the nine stencil-row pointers for row `by` into local memory. */
inline void PublishRow(const float **s_qptr, u32 by, u32 nb, int nspans,
                       const float **sp, const u32 *sbase, const u32 *scnt,
                       const u32 *sdz, u64 row_elems, sycl::nd_item<1> it) {
  if (it.get_local_id(0) == 0) {
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
  sycl::group_barrier(it.get_group());
}

/** Group-reduce three doubles into acc[0..2] (eflag accumulators). */
inline void ReduceAdd3(double *red, const double vals[3], double *acc,
                       sycl::nd_item<1> it) {
  const u32 tid = it.get_local_id(0);
  const u32 n = it.get_local_range(0);
  for (int q = 0; q < 3; ++q) {
    red[tid] = vals[q];
    sycl::group_barrier(it.get_group());
    for (u32 wd = n / 2; wd > 0; wd >>= 1) {
      if (tid < wd) red[tid] += red[tid + wd];
      sycl::group_barrier(it.get_group());
    }
    if (tid == 0) gvc::AtomicAdd(&acc[q], red[0]);
    sycl::group_barrier(it.get_group());
  }
}

/** Everything the force kernels take, so the launches stay one line. */
struct ForceArgs {
  Decomp d;
  const float *x;
  float *f;
  const int *nl;
  const u32 *cnt;
  float box, cutoff;
  u32 maxneigh;
  int eflag;
  double *acc;
  u64 row_elems, plane_elems;
  u32 rowchunk;
  int nocompute;
  u32 blocks, threads;
};

/** Minimum-image LJ pair for the force loops. Returns false outside cutoff. */
inline bool LjPair(float xi, float yi, float zi, const float *jp, float halfL,
                   float box, float c2, float &fx, float &fy, float &fz,
                   double &pe, double &w, double &npairs, int eflag) {
  float ddx = xi - jp[0];
  float ddy = yi - jp[1];
  float ddz = zi - jp[2];
  if (ddx > halfL) ddx -= box; else if (ddx < -halfL) ddx += box;
  if (ddy > halfL) ddy -= box; else if (ddy < -halfL) ddy += box;
  if (ddz > halfL) ddz -= box; else if (ddz < -halfL) ddz += box;
  const float rsq = ddx * ddx + ddy * ddy + ddz * ddz;
  if (rsq >= c2) return false;
  const float r2i = 1.0f / rsq;
  const float r6i = r2i * r2i * r2i;
  const float fpair = r6i * (48.0f * r6i - 24.0f) * r2i;
  fx = sycl::fma(ddx, fpair, fx);
  fy = sycl::fma(ddy, fpair, fy);
  fz = sycl::fma(ddz, fpair, fz);
  if (eflag) {
    pe += 0.5 * static_cast<double>(4.0f * r6i * (r6i - 1.0f));
    w += 0.5 * static_cast<double>(r6i * (48.0f * r6i - 24.0f));
    npairs += 1.0;
  }
  return true;
}

/** K3 cell-direct force over this rank's rows. */
void ForceKernel(sycl::queue &q, ForceArgs a) {
  const size_t g = static_cast<size_t>(a.blocks) * a.threads;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<double, 1> red(a.threads, h);
     sycl::local_accessor<const float *, 1> qptr(9, h);
     h.parallel_for(sycl::nd_range<1>(g, a.threads), [=](sycl::nd_item<1> it) {
       const float **s_qptr = &qptr[0];
       const Decomp &d = a.d;
       const u32 nb = d.nb, cap = d.cap, tid = it.get_local_id(0);
       const u64 islots = static_cast<u64>(nb) * cap;
       const float c2 = a.cutoff * a.cutoff;
       const float halfL = 0.5f * a.box;
       double pe = 0.0, w = 0.0, npairs = 0.0;
       const u32 cpz = (nb + a.rowchunk - 1) / a.rowchunk;
       const u64 nchunks = static_cast<u64>(d.nplanes) * cpz;
       for (u64 ch = it.get_group(0); ch < nchunks; ch += a.blocks) {
         const u32 lz = static_cast<u32>(ch / cpz);
         const u32 bz = d.z0 + lz;
         const u32 y0 = static_cast<u32>(ch % cpz) * a.rowchunk;
         if (y0 >= nb) continue;
         const u32 ylast = (y0 + a.rowchunk - 1 < nb) ? (y0 + a.rowchunk - 1)
                                                      : (nb - 1);
         const float *sp[kMaxSpans];
         u32 sbase[kMaxSpans], scnt[kMaxSpans], sdz[kMaxSpans];
         const int nspans = ResolveStencil(d, a.x, bz, y0, ylast, a.row_elems,
                                           a.plane_elems, sp, sbase, scnt,
                                           sdz, it);
         for (u32 by = y0; by <= ylast; ++by) {
           PublishRow(s_qptr, by, nb, nspans, sp, sbase, scnt, sdz,
                      a.row_elems, it);
           float *const fp =
               a.f + (static_cast<u64>(lz) * nb + by) * a.row_elems;
           for (u64 e = tid; e < a.row_elems; e += a.threads) fp[e] = 0.0f;
           sycl::group_barrier(it.get_group());
           const float *const ip_row = s_qptr[4];
           for (u64 s = tid; s < islots; s += a.threads) {
             const float *const ip = ip_row + s * kStride;
             if (ip[3] < 0.0f) continue;
             const float xi = ip[0], yi = ip[1], zi = ip[2];
             const u32 bx = static_cast<u32>(s / cap);
             float fx = 0.0f, fy = 0.0f, fz = 0.0f;
             for (int qq = 0; qq < 9; ++qq) {
               const float *const qp = s_qptr[qq];
               for (int dxx = -1; dxx <= 1; ++dxx) {
                 const u32 jbx = (bx + nb + dxx) % nb;
                 const u64 jb = static_cast<u64>(jbx) * cap * kStride;
                 for (u32 sj = 0; sj < cap; ++sj) {
                   const float *const jp =
                       qp + jb + static_cast<u64>(sj) * kStride;
                   if (jp[3] < 0.0f) continue;
                   if (qq == 4 && jbx == bx && sj == s % cap) continue;
                   LjPair(xi, yi, zi, jp, halfL, a.box, c2, fx, fy, fz, pe,
                          w, npairs, a.eflag);
                 }
               }
             }
             float *const op = fp + s * kStride;
             op[0] = fx; op[1] = fy; op[2] = fz;
           }
           sycl::group_barrier(it.get_group());
         }
       }
       if (a.eflag) {
         const double vals[3] = {pe, w, npairs};
         ReduceAdd3(&red[0], vals, a.acc, it);
       }
     });
   }).wait();
}

/** K2c: build the Verlet list for this rank's rows. */
void BuildListKernel(sycl::queue &q, ForceArgs a, int *nl, float rlist,
                     u32 *cnt, int *err) {
  const size_t g = static_cast<size_t>(a.blocks) * a.threads;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<const float *, 1> qptr(9, h);
     h.parallel_for(sycl::nd_range<1>(g, a.threads), [=](sycl::nd_item<1> it) {
       const float **s_qptr = &qptr[0];
       const Decomp &d = a.d;
       const u32 nb = d.nb, cap = d.cap, tid = it.get_local_id(0);
       const u64 islots = static_cast<u64>(nb) * cap;
       const u64 rowlist = islots * a.maxneigh;
       const float r2list = rlist * rlist;
       const float halfL = 0.5f * a.box;
       const u32 cpz = (nb + a.rowchunk - 1) / a.rowchunk;
       const u64 nchunks = static_cast<u64>(d.nplanes) * cpz;
       for (u64 ch = it.get_group(0); ch < nchunks; ch += a.blocks) {
         const u32 lz = static_cast<u32>(ch / cpz);
         const u32 bz = d.z0 + lz;
         const u32 y0 = static_cast<u32>(ch % cpz) * a.rowchunk;
         if (y0 >= nb) continue;
         const u32 ylast = (y0 + a.rowchunk - 1 < nb) ? (y0 + a.rowchunk - 1)
                                                      : (nb - 1);
         const float *sp[kMaxSpans];
         u32 sbase[kMaxSpans], scnt[kMaxSpans], sdz[kMaxSpans];
         const int nspans = ResolveStencil(d, a.x, bz, y0, ylast, a.row_elems,
                                           a.plane_elems, sp, sbase, scnt,
                                           sdz, it);
         for (u32 by = y0; by <= ylast; ++by) {
           PublishRow(s_qptr, by, nb, nspans, sp, sbase, scnt, sdz,
                      a.row_elems, it);
           const u64 lrow = static_cast<u64>(lz) * nb + by;
           int *const np = nl + lrow * rowlist;
           const u64 slotbase = lrow * islots;
           const float *const ip_row = s_qptr[4];
           for (u64 s = tid; s < islots; s += a.threads) {
             const float *const ip = ip_row + s * kStride;
             if (ip[3] < 0.0f) { cnt[slotbase + s] = 0; continue; }
             const float xi = ip[0], yi = ip[1], zi = ip[2];
             const u32 bx = static_cast<u32>(s / cap);
             u32 n = 0;
             for (int qq = 0; qq < 9; ++qq) {
               const float *const qp = s_qptr[qq];
               for (int dxx = -1; dxx <= 1; ++dxx) {
                 const u32 jbx = (bx + nb + dxx) % nb;
                 const u64 jb = static_cast<u64>(jbx) * cap * kStride;
                 for (u32 sj = 0; sj < cap; ++sj) {
                   const float *const jp =
                       qp + jb + static_cast<u64>(sj) * kStride;
                   if (jp[3] < 0.0f) continue;
                   if (qq == 4 && jbx == bx && sj == s % cap) continue;
                   float ddx = xi - jp[0];
                   float ddy = yi - jp[1];
                   float ddz = zi - jp[2];
                   if (ddx > halfL) ddx -= a.box; else if (ddx < -halfL) ddx += a.box;
                   if (ddy > halfL) ddy -= a.box; else if (ddy < -halfL) ddy += a.box;
                   if (ddz > halfL) ddz -= a.box; else if (ddz < -halfL) ddz += a.box;
                   const float rsq = ddx * ddx + ddy * ddy + ddz * ddz;
                   if (rsq >= r2list) continue;
                   if (n >= a.maxneigh) { *err = 1; continue; }
                   np[static_cast<u64>(n) * islots + s] = static_cast<int>(
                       (static_cast<u32>(qq) << 16) | (jbx * cap + sj));
                   ++n;
                 }
               }
             }
             cnt[slotbase + s] = n;
           }
           sycl::group_barrier(it.get_group());
         }
       }
     });
   }).wait();
}

/** K3-list: the force pass streaming the padded Verlet list. */
void ListForceKernel(sycl::queue &q, ForceArgs a) {
  const size_t g = static_cast<size_t>(a.blocks) * a.threads;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<double, 1> red(a.threads, h);
     sycl::local_accessor<const float *, 1> qptr(9, h);
     h.parallel_for(sycl::nd_range<1>(g, a.threads), [=](sycl::nd_item<1> it) {
       const float **s_qptr = &qptr[0];
       const Decomp &d = a.d;
       const u32 nb = d.nb, cap = d.cap, tid = it.get_local_id(0);
       const u64 islots = static_cast<u64>(nb) * cap;
       const u64 rowlist = islots * a.maxneigh;
       const float c2 = a.cutoff * a.cutoff;
       const float halfL = 0.5f * a.box;
       double pe = 0.0, w = 0.0, npairs = 0.0;
       const u32 cpz = (nb + a.rowchunk - 1) / a.rowchunk;
       const u64 nchunks = static_cast<u64>(d.nplanes) * cpz;
       for (u64 ch = it.get_group(0); ch < nchunks; ch += a.blocks) {
         const u32 lz = static_cast<u32>(ch / cpz);
         const u32 bz = d.z0 + lz;
         const u32 y0 = static_cast<u32>(ch % cpz) * a.rowchunk;
         if (y0 >= nb) continue;
         const u32 ylast = (y0 + a.rowchunk - 1 < nb) ? (y0 + a.rowchunk - 1)
                                                      : (nb - 1);
         const float *sp[kMaxSpans];
         u32 sbase[kMaxSpans], scnt[kMaxSpans], sdz[kMaxSpans];
         const int nspans = ResolveStencil(d, a.x, bz, y0, ylast, a.row_elems,
                                           a.plane_elems, sp, sbase, scnt,
                                           sdz, it);
         for (u32 by = y0; by <= ylast; ++by) {
           PublishRow(s_qptr, by, nb, nspans, sp, sbase, scnt, sdz,
                      a.row_elems, it);
           const u64 lrow = static_cast<u64>(lz) * nb + by;
           float *const fp = a.f + lrow * a.row_elems;
           for (u64 e = tid; e < a.row_elems; e += a.threads) fp[e] = 0.0f;
           sycl::group_barrier(it.get_group());
           const int *const np = a.nl + lrow * rowlist;
           const u64 slotbase = lrow * islots;
           const float *const ip_row = s_qptr[4];
           for (u64 s = tid; s < islots; s += a.threads) {
             const float *const ip = ip_row + s * kStride;
             if (ip[3] < 0.0f) continue;
             const float xi = ip[0], yi = ip[1], zi = ip[2];
             const u32 n = a.nocompute ? 0u : a.cnt[slotbase + s];
             float fx = 0.0f, fy = 0.0f, fz = 0.0f;
             for (u32 k = 0; k < n; ++k) {
               const u32 ent =
                   static_cast<u32>(np[static_cast<u64>(k) * islots + s]);
               const float *const jp =
                   s_qptr[ent >> 16] + static_cast<u64>(ent & 0xffffu) * kStride;
               LjPair(xi, yi, zi, jp, halfL, a.box, c2, fx, fy, fz, pe, w,
                      npairs, a.eflag);
             }
             float *const op = fp + s * kStride;
             op[0] = fx; op[1] = fy; op[2] = fz;
           }
           sycl::group_barrier(it.get_group());
         }
       }
       if (a.eflag) {
         const double vals[3] = {pe, w, npairs};
         ReduceAdd3(&red[0], vals, a.acc, it);
       }
     });
   }).wait();
}

/** Launch a grid-stride kernel over [0, n). */
template <typename Body>
void GridStride(sycl::queue &q, u32 blocks, u32 threads, u64 n, Body body) {
  const size_t g = static_cast<size_t>(blocks) * threads;
  q.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
     for (u64 t = it.get_global_id(0); t < n; t += g) body(t);
   }).wait();
}

/** The resort's device state. */
struct Resort {
  u32 *bincnt;
  int *dest_pe;
  u32 *dest_slot;
  int *err;
  float *out_lo, *out_hi;
  u32 *nout_lo, *nout_hi;
  u32 out_cap;
  unsigned long long *ctr;
};

/**
 * K2a: wrap, re-bin, and claim a slot for atoms that stay; PACK the leavers
 * into a per-direction outbox (global bin, x4, v4) for the substrate to
 * carry, as the CUDA MPI edition does.
 */
void RebinKernel(sycl::queue &q, u32 blocks, u32 threads, Decomp d, float *x,
                 const float *v, float box, Resort r, u64 local_slots) {
  const u32 nb = d.nb, cap = d.cap;
  const float fnb = static_cast<float>(nb);
  GridStride(q, blocks, threads, local_slots, [=](u64 s) {
    const u64 e = s * kStride;
    if (x[e + 3] < 0.0f) { r.dest_pe[s] = -1; return; }
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
    if (rel < d.nplanes) {
      const u64 lbin = (static_cast<u64>(rel) * nb + by) * nb + bx;
      sycl::atomic_ref<u32, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          bc(r.bincnt[lbin]);
      const u32 slot = bc.fetch_add(1u);
      if (slot >= cap) { *r.err = 1; r.dest_pe[s] = -1; return; }
      r.dest_pe[s] = static_cast<int>(d.mype);
      r.dest_slot[s] = static_cast<u32>(lbin * cap + slot);
      return;
    }
    const bool down = (bz == (d.z0 + d.nb - 1) % d.nb);
    u32 *const nout = down ? r.nout_lo : r.nout_hi;
    float *const out = down ? r.out_lo : r.out_hi;
    sycl::atomic_ref<u32, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        no(*nout);
    const u32 idx = no.fetch_add(1u);
    if (idx >= r.out_cap) {
      gvc::AtomicAdd(&r.ctr[kCtrMigrantOverflow], 1ull);
      *r.err = 2; r.dest_pe[s] = -1; return;
    }
    const u64 gbin = (static_cast<u64>(bz) * nb + by) * nb + bx;
    float *const rec = out + static_cast<u64>(idx) * 9;
    rec[0] = static_cast<float>(gbin);
    rec[1] = px; rec[2] = py; rec[3] = pz; rec[4] = x[e + 3];
    rec[5] = v[e + 0]; rec[6] = v[e + 1]; rec[7] = v[e + 2];
    rec[8] = 0.0f;
    r.dest_pe[s] = -1;
    gvc::AtomicAdd(&r.ctr[kCtrMigrants], 1ull);
  });
}

/** Unpack received migrants: claim a local slot and write both arrays. */
void UnpackKernel(sycl::queue &q, u32 blocks, u32 threads, Decomp d,
                  const float *in, u32 n, u32 *bincnt, float *dstx,
                  float *dstv, int *err) {
  if (n == 0) return;
  const u32 nb = d.nb, cap = d.cap;
  GridStride(q, blocks, threads, n, [=](u64 i) {
    const float *const rec = in + i * 9;
    const u64 gbin = static_cast<u64>(rec[0]);
    const u32 bz = static_cast<u32>(gbin / (static_cast<u64>(nb) * nb));
    const u32 by = static_cast<u32>((gbin / nb) % nb);
    const u32 bx = static_cast<u32>(gbin % nb);
    const u32 rel = bz - d.z0;
    if (rel >= d.nplanes) { *err = 3; return; }
    const u64 lbin = (static_cast<u64>(rel) * nb + by) * nb + bx;
    sycl::atomic_ref<u32, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        bc(bincnt[lbin]);
    const u32 slot = bc.fetch_add(1u);
    if (slot >= cap) { *err = 1; return; }
    const u64 de = (lbin * cap + slot) * kStride;
    dstx[de + 0] = rec[1]; dstx[de + 1] = rec[2];
    dstx[de + 2] = rec[3]; dstx[de + 3] = rec[4];
    dstv[de + 0] = rec[5]; dstv[de + 1] = rec[6];
    dstv[de + 2] = rec[7]; dstv[de + 3] = 0.0f;
  });
}

/** K2b: apply the permutation for atoms that stayed. */
void ScatterKernel(sycl::queue &q, u32 blocks, u32 threads, const float *src,
                   const float *srcx, float *dst, const int *dest_pe,
                   const u32 *dest_slot, int keep_w, u64 local_slots) {
  GridStride(q, blocks, threads, local_slots, [=](u64 s) {
    const u64 e = s * kStride;
    if (srcx[e + 3] < 0.0f) return;
    if (dest_pe[s] < 0) return;
    const u64 de = static_cast<u64>(dest_slot[s]) * kStride;
    dst[de + 0] = src[e + 0];
    dst[de + 1] = src[e + 1];
    dst[de + 2] = src[e + 2];
    dst[de + 3] = keep_w ? src[e + 3] : 0.0f;
  });
}

void SentinelKernel(sycl::queue &q, u32 blocks, u32 threads, float *dst,
                    u64 local_slots) {
  GridStride(q, blocks, threads, local_slots,
             [=](u64 s) { dst[s * kStride + 3] = -1.0f; });
}

/** Velocity-Verlet half-kick (+ drift) with forces or with constant g. */
void IntegrateKernel(sycl::queue &q, u32 blocks, u32 threads, float *x,
                     float *v, const float *f, float dt, float gx, float gy,
                     float gz, int drift, u64 local_slots) {
  const float half = 0.5f * dt;
  GridStride(q, blocks, threads, local_slots, [=](u64 s) {
    const u64 e = s * kStride;
    if (x[e + 3] < 0.0f) return;
    const float ax = f ? f[e + 0] : gx;
    const float ay = f ? f[e + 1] : gy;
    const float az = f ? f[e + 2] : gz;
    const float vx = sycl::fma(half, ax, v[e + 0]);
    const float vy = sycl::fma(half, ay, v[e + 1]);
    const float vz = sycl::fma(half, az, v[e + 2]);
    v[e + 0] = vx; v[e + 1] = vy; v[e + 2] = vz;
    if (drift) {
      x[e + 0] = sycl::fma(dt, vx, x[e + 0]);
      x[e + 1] = sycl::fma(dt, vy, x[e + 1]);
      x[e + 2] = sycl::fma(dt, vz, x[e + 2]);
    }
  });
}

/** KE and momentum of the owned slots, into out[0..3]. */
void ThermoKernel(sycl::queue &q, u32 blocks, u32 threads, const float *x,
                  const float *v, double *out, u64 local_slots) {
  const size_t g = static_cast<size_t>(blocks) * threads;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<double, 1> red(threads, h);
     h.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
       double ke = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
       for (u64 s = it.get_global_id(0); s < local_slots; s += g) {
         const u64 e = s * kStride;
         if (x[e + 3] < 0.0f) continue;
         const double vx = v[e + 0], vy = v[e + 1], vz = v[e + 2];
         ke += 0.5 * (vx * vx + vy * vy + vz * vz);
         mx += vx; my += vy; mz += vz;
       }
       const double vals[4] = {ke, mx, my, mz};
       const u32 tid = it.get_local_id(0);
       for (int qq = 0; qq < 4; ++qq) {
         red[tid] = vals[qq];
         sycl::group_barrier(it.get_group());
         for (u32 wd = threads / 2; wd > 0; wd >>= 1) {
           if (tid < wd) red[tid] += red[tid + wd];
           sycl::group_barrier(it.get_group());
         }
         if (tid == 0) gvc::AtomicAdd(&out[qq], red[0]);
         sycl::group_barrier(it.get_group());
       }
     });
   }).wait();
}

/** Parse the command line; false on an unknown argument. */
bool ParseArgs(int argc, char **argv, Args &a) {
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
    else if (std::strcmp(argv[i], "--force-halo") == 0) a.force_halo = 1;
    else if (std::strcmp(argv[i], "--no-gate") == 0) a.gate = 0;
    else if (std::strcmp(argv[i], "--md") == 0) a.md = 1;
    else {
      std::fprintf(stderr, "unknown arg %s\n", argv[i]);
      return false;
    }
  }
  return true;
}

/** Zero-mean the velocities and scale to --temp, as the CUDA editions. */
void Thermalise(const Args &a, const Geometry &g, const std::vector<float> &hx,
                std::vector<float> &hv) {
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
  const double scale = std::sqrt(1.5 * a.temp * static_cast<double>(g.natoms) / ke);
  for (u64 s = 0; s < nslots; ++s) {
    if (hx[s * kStride + 3] < 0.0f) continue;
    for (int d = 0; d < 3; ++d) {
      hv[s * kStride + d] = static_cast<float>(hv[s * kStride + d] * scale);
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  Args a;
  if (!ParseArgs(argc, argv, a)) return 1;
  gvc::Comm comm;
  comm.Init(&argc, &argv);
  const int mype = comm.rank, npes = comm.nranks;
  sycl::queue &q = comm.q;
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

  auto refuse = [&](const char *why) {
    if (root) std::fprintf(stderr, "%s\n", why);
    comm.Finalize();
    return 1;
  };
  if (a.md && g.nb < 3) return refuse("need at least 3 bins per dimension");
  if (static_cast<u64>(g.nb) * g.cap >= 65536) {
    return refuse("nb*cap must fit 16 bits for entry packing");
  }
  if (g.nbins >= (1ull << 24)) {
    return refuse("bins exceed the 2^24 exactly-representable migrant range");
  }
  if (g.nb / static_cast<u32>(npes) < 3 && npes > 1) {
    return refuse("each rank needs at least 3 z-planes; raise --lattice");
  }

  std::vector<u32> h_plane_z0(npes + 1);
  for (int p = 0; p <= npes; ++p) {
    h_plane_z0[p] = static_cast<u32>(static_cast<u64>(p) * g.nb / npes);
  }
  const u32 myz0 = h_plane_z0[mype];
  const u32 mynplanes = h_plane_z0[mype + 1] - myz0;
  const int rank_down = (mype - 1 + npes) % npes;
  const int rank_up = (mype + 1) % npes;
  // THE EXTENDED SLAB IS THE SAME SHAPE ON EVERY RANK: [halo_lo | slots
  // 1..max_planes (this rank owns the first nplanes of them) | halo_hi at
  // slot max_planes + 1]. Plane counts are uneven when nb % npes != 0, and
  // a halo_hi at (nplanes + 1) then sits at a different offset per rank --
  // under ISHMEM a put lands at the SENDER's offset on the peer, which
  // overwrote the peer's last owned plane (1.5% of pairs missing at
  // step 0), and unequal symmetric allocation sizes corrupt the heap.
  u32 max_planes = 0;
  for (int p = 0; p < npes; ++p) {
    max_planes = std::max(max_planes, h_plane_z0[p + 1] - h_plane_z0[p]);
  }
  const u32 hi_slot = max_planes + 1;
  const u32 ext_planes = max_planes + 2;
  const u64 ext_elems = static_cast<u64>(ext_planes) * g.plane_elems;
  const u64 own_off = g.plane_elems;
  const u64 local_slots = static_cast<u64>(mynplanes) * g.nb * g.nb * g.cap;
  const u64 local_elems = local_slots * kStride;
  const u64 islots = static_cast<u64>(g.nb) * g.cap;
  const u64 rowlist = islots * a.maxneigh;
  const u64 nl_elems = static_cast<u64>(mynplanes) * g.nb * rowlist;
  const u32 out_cap = static_cast<u32>(g.nb) * g.nb * g.cap;

  const double xvf_mb =
      (ext_elems + 2.0 * local_elems) * sizeof(float) / 1048576.0;
  const double pp_mb = 2.0 * local_elems * sizeof(float) / 1048576.0;
  const double nl_mb = nl_elems * sizeof(int) / 1048576.0;
  const double total_mb = xvf_mb + (a.md ? pp_mb : 0.0) +
                          (a.md && a.use_list ? nl_mb : 0.0);
  if (root) {
    std::printf(
        "eternia-MD / %s baseline (SYCL)\n"
        "  atoms=%llu box=%.4f bins=%u^3 cap=%u  ranks=%d (planes/rank=%u)\n"
        "  blocks=%u threads=%u rowchunk=%u list=%s halo=%s\n"
        "  per-rank VRAM: x(+2 halo planes)/v/f %.1f MB + resort %.1f MB + "
        "list %.1f MB = %.1f MB\n",
        gvc::Comm::Name(), g.natoms, g.box, g.nb, g.cap, npes, mynplanes,
        a.blocks, a.threads, a.rowchunk, a.use_list ? "verlet" : "cell-direct",
        (npes > 1 || a.force_halo) ? "exchanged every step"
                                   : "not needed (1 rank)",
        xvf_mb, a.md ? pp_mb : 0.0, (a.md && a.use_list) ? nl_mb : 0.0,
        total_mb);
  }

  std::vector<float> hx, hv;
  if (!BuildInitialState(a, g, hx, hv)) return refuse("deck build failed");
  if (a.temp > 0.0) Thermalise(a, g, hx, hv);

  // x lives in the extended slab (symmetric: peers put halo planes into it);
  // v and f only ever cover owned planes.
  float *dx_ext = comm.Alloc<float>(ext_elems);
  float *dv = comm.AllocLocal<float>(local_elems);
  float *df = comm.AllocLocal<float>(local_elems);
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
    q.memcpy(dx_ext, sx.data(), ext_elems * sizeof(float));
    q.memcpy(dv, sv.data(), local_elems * sizeof(float));
    q.memset(df, 0, local_elems * sizeof(float));
    q.wait();
  }

  Decomp d;
  d.nb = g.nb; d.cap = g.cap;
  d.npes = static_cast<u32>(npes); d.mype = static_cast<u32>(mype);
  d.nplanes = mynplanes; d.z0 = myz0;
  d.use_halo = (npes > 1) || a.force_halo;
  d.halo_lo = (myz0 + g.nb - 1) % g.nb;
  d.halo_hi = (myz0 + mynplanes) % g.nb;
  d.hi_slot = hi_slot;

  unsigned long long *d_ctr = comm.AllocLocal<unsigned long long>(kCtrNumCtrs);
  q.memset(d_ctr, 0, kCtrNumCtrs * sizeof(unsigned long long)).wait();
  double *d_thermo = comm.AllocLocal<double>(4);

  double t_halo = 0.0, halo_bytes = 0.0;
  u64 n_halo = 0;
  const bool need_halo = (npes > 1) || a.force_halo;
  // Exchange the two boundary planes of x: my LAST owned plane is the up
  // neighbour's halo_lo; my FIRST is the down neighbour's halo_hi.
  auto exchange_halo = [&](float *xe) {
    if (!need_halo) return;
    const double t = gvc::NowMs();
    const float *const first = xe + own_off;
    const float *const last =
        xe + own_off + static_cast<u64>(mynplanes - 1) * g.plane_elems;
    comm.Sendrecv(last, rank_up, xe, rank_down, g.plane_elems);
    comm.Sendrecv(first, rank_down, xe + static_cast<u64>(hi_slot) * g.plane_elems,
                  rank_up, g.plane_elems);
    t_halo += gvc::NowMs() - t;
    halo_bytes += 2.0 * g.plane_elems * sizeof(float);
    ++n_halo;
  };

  const float fbox = static_cast<float>(g.box);
  const float fcut = static_cast<float>(a.cutoff);
  const float frlist = static_cast<float>(a.cutoff + a.skin);
  const float fdt = static_cast<float>(a.dt);

  if (a.md) {
    float *dx2_ext = comm.Alloc<float>(ext_elems);
    float *dv2 = comm.AllocLocal<float>(local_elems);
    float *dx2 = dx2_ext + own_off;
    Resort r;
    r.bincnt = comm.AllocLocal<u32>(static_cast<u64>(mynplanes) * g.nb * g.nb);
    r.dest_pe = comm.AllocLocal<int>(local_slots);
    r.dest_slot = comm.AllocLocal<u32>(local_slots);
    r.err = comm.AllocLocal<int>(1);
    r.out_lo = comm.AllocLocal<float>(static_cast<u64>(out_cap) * 9);
    r.out_hi = comm.AllocLocal<float>(static_cast<u64>(out_cap) * 9);
    r.nout_lo = comm.AllocLocal<u32>(1);
    r.nout_hi = comm.AllocLocal<u32>(1);
    r.out_cap = out_cap;
    r.ctr = d_ctr;
    // Two symmetric inboxes: what arrives from up, what arrives from down.
    float *d_in_up = comm.Alloc<float>(static_cast<u64>(out_cap) * 9);
    float *d_in_dn = comm.Alloc<float>(static_cast<u64>(out_cap) * 9);
    u32 *d_cnt = comm.AllocLocal<u32>(local_slots);
    double *d_acc = comm.AllocLocal<double>(3);
    int *d_nl = nullptr;
    if (a.use_list) {
      d_nl = comm.AllocLocal<int>(nl_elems);
      q.memset(d_nl, 0, nl_elems * sizeof(int)).wait();
    }

    double acc[3] = {0, 0, 0};
    double t_force = 0.0, t_kick = 0.0, t_resort = 0.0, t_build = 0.0;
    double mig_bytes = 0.0;
    const int nocompute = std::getenv("MD_NOCOMPUTE") != nullptr ? 1 : 0;
    auto fargs = [&](int eflag) {
      ForceArgs fa;
      fa.d = d; fa.x = dx_ext; fa.f = df; fa.nl = d_nl; fa.cnt = d_cnt;
      fa.box = fbox; fa.cutoff = fcut; fa.maxneigh = a.maxneigh;
      fa.eflag = eflag; fa.acc = d_acc; fa.row_elems = g.row_elems;
      fa.plane_elems = g.plane_elems; fa.rowchunk = a.rowchunk;
      fa.nocompute = nocompute; fa.blocks = a.blocks; fa.threads = a.threads;
      return fa;
    };
    auto read_err = [&]() -> bool {
      int err = 0;
      q.memcpy(&err, r.err, sizeof(int)).wait();
      double e = err;
      comm.HostSumN(&e, 1);
      return e != 0.0;
    };

    auto build_list = [&]() -> bool {
      const double t = gvc::NowMs();
      q.memset(r.err, 0, sizeof(int)).wait();
      BuildListKernel(q, fargs(0), d_nl, frlist, d_cnt, r.err);
      const bool bad = read_err();
      t_build += gvc::NowMs() - t;
      if (bad && root) {
        std::fprintf(stderr, "list: an atom has more than --maxneigh %u "
                             "neighbours within cutoff+skin\n", a.maxneigh);
      }
      return !bad;
    };
    auto force = [&](int eflag) {
      const double t = gvc::NowMs();
      if (eflag) q.memset(d_acc, 0, 3 * sizeof(double)).wait();
      if (a.use_list) ListForceKernel(q, fargs(eflag));
      else ForceKernel(q, fargs(eflag));
      if (eflag) {
        q.memcpy(acc, d_acc, 3 * sizeof(double)).wait();
        comm.HostSumN(acc, 3);
      }
      t_force += gvc::NowMs() - t;
    };
    auto kick = [&](int drift) {
      const double t = gvc::NowMs();
      IntegrateKernel(q, a.blocks, a.threads, dx, dv, df, fdt, 0, 0, 0, drift,
                      local_slots);
      t_kick += gvc::NowMs() - t;
    };
    auto thermo_ke = [&]() -> double {
      q.memset(d_thermo, 0, 4 * sizeof(double)).wait();
      ThermoKernel(q, a.blocks, a.threads, dx, dv, d_thermo, local_slots);
      double t4[4];
      q.memcpy(t4, d_thermo, sizeof(t4)).wait();
      comm.HostSumN(t4, 4);
      return t4[0];
    };
    // K2 with two-sided migration: rebin locally and pack leavers, exchange
    // the outboxes with both neighbours, unpack into claimed slots.
    auto resort = [&]() -> bool {
      const double t = gvc::NowMs();
      q.memset(r.bincnt, 0,
               static_cast<u64>(mynplanes) * g.nb * g.nb * sizeof(u32));
      q.memset(r.err, 0, sizeof(int));
      q.memset(r.nout_lo, 0, sizeof(u32));
      q.memset(r.nout_hi, 0, sizeof(u32));
      q.wait();
      SentinelKernel(q, a.blocks, a.threads, dx2, local_slots);
      SentinelKernel(q, a.blocks, a.threads, dv2, local_slots);
      RebinKernel(q, a.blocks, a.threads, d, dx, dv, fbox, r, local_slots);
      if (read_err()) {
        if (root) std::fprintf(stderr, "resort: bin or migrant-buffer "
                                       "overflow -- raise --cap\n");
        return false;
      }
      ScatterKernel(q, a.blocks, a.threads, dx, dx, dx2, r.dest_pe,
                    r.dest_slot, 1, local_slots);
      ScatterKernel(q, a.blocks, a.threads, dv, dx, dv2, r.dest_pe,
                    r.dest_slot, 0, local_slots);
      if (npes > 1) {
        u32 n_lo = 0, n_hi = 0;
        q.memcpy(&n_lo, r.nout_lo, sizeof(u32));
        q.memcpy(&n_hi, r.nout_hi, sizeof(u32));
        q.wait();
        // Counts first (host), then the payload on the substrate: what
        // leaves through my low face lands on rank_down's "from up" inbox.
        const u32 r_from_up = comm.HostExchange(n_lo, rank_down, rank_up);
        const u32 r_from_down = comm.HostExchange(n_hi, rank_up, rank_down);
        if (r_from_up > out_cap || r_from_down > out_cap) {
          std::fprintf(stderr, "migrant inbox overflow\n");
          return false;
        }
        comm.SendrecvN(r.out_lo, rank_down, static_cast<u64>(n_lo) * 9,
                       d_in_up, rank_up, static_cast<u64>(r_from_up) * 9);
        comm.SendrecvN(r.out_hi, rank_up, static_cast<u64>(n_hi) * 9,
                       d_in_dn, rank_down, static_cast<u64>(r_from_down) * 9);
        mig_bytes += static_cast<double>(n_lo + n_hi) * 9 * sizeof(float);
        UnpackKernel(q, a.blocks, a.threads, d, d_in_up, r_from_up, r.bincnt,
                     dx2, dv2, r.err);
        UnpackKernel(q, a.blocks, a.threads, d, d_in_dn, r_from_down,
                     r.bincnt, dx2, dv2, r.err);
      }
      if (read_err()) {
        if (root) std::fprintf(stderr, "resort: unpack refused (err)\n");
        return false;
      }
      std::swap(dx_ext, dx2_ext);
      std::swap(dx, dx2);
      std::swap(dv, dv2);
      t_resort += gvc::NowMs() - t;
      return true;
    };

    exchange_halo(dx_ext);
    if (a.use_list && !build_list()) return refuse("list build failed");
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
      if (!resort()) return refuse("resort failed");
      exchange_halo(dx_ext);
      if (a.use_list && !build_list()) return refuse("list build failed");
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
    q.memset(d_ctr, 0, kCtrNumCtrs * sizeof(unsigned long long)).wait();
    comm.Barrier();
    const double t0 = gvc::NowMs();
    // CHECKPOINT: x and v to pinned host DRAM, and durably to --ckpt-dir
    // as one fsync'd restart-shaped file each, as the CUDA MPI edition.
    float *h_ckpt = nullptr;
    double t_ckpt = 0.0, t_ckpt_io = 0.0;
    u64 n_ckpt = 0;
    const u64 ckpt_elems = 2 * local_elems;
    if (a.ckpt != 0) h_ckpt = sycl::malloc_host<float>(ckpt_elems, q);
    auto checkpoint = [&]() {
      const double _t = gvc::NowMs();
      q.memcpy(h_ckpt, dx, local_elems * sizeof(float));
      q.memcpy(h_ckpt + local_elems, dv, local_elems * sizeof(float));
      q.wait();
      t_ckpt += gvc::NowMs() - _t;
      if (!a.ckpt_dir.empty()) {
        const double _w = gvc::NowMs();
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/ckpt_r%d_%llu.bin",
                      a.ckpt_dir.c_str(), mype, (unsigned long long)n_ckpt);
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
        t_ckpt_io += gvc::NowMs() - _w;
      }
      ++n_ckpt;
    };
    for (u64 step = 0; step < a.steps; ++step) {
      kick(1);
      if (a.rebin != 0 && step != 0 && step % a.rebin == 0) {
        if (!resort()) return refuse("resort failed");
        exchange_halo(dx_ext);
        if (a.use_list && !build_list()) return refuse("list build failed");
      }
      exchange_halo(dx_ext);
      force(0);
      kick(0);
      if (a.ckpt != 0 && (step + 1) % a.ckpt == 0) checkpoint();
    }
    comm.Barrier();
    const double run_ms = gvc::NowMs() - t0;
    exchange_halo(dx_ext);
    force(1);
    const double ke_n = thermo_ke();
    const double e_n = acc[0] + ke_n;
    const double e_drift = std::fabs(e_n - e0) / std::fabs(e0);
    const bool nve_ok = (e_drift < a.drift_tol);

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
    if (h_ckpt != nullptr) sycl::free(h_ckpt, q);

    unsigned long long ctr[kCtrNumCtrs] = {0};
    q.memcpy(ctr, d_ctr, sizeof(ctr)).wait();
    double led[4] = {halo_bytes, mig_bytes,
                     static_cast<double>(ctr[kCtrMigrants]), t_halo};
    comm.HostSumN(led, 4);
    const bool ok = statics_ok && nve_ok && resort_ok;
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
          "summed over ranks = %.1f%% of a rank's run\n",
          led[0] / 1073741824.0, n_halo, led[0] / a.steps / 1048576.0, led[2],
          led[1] / 1048576.0, led[3], 100.0 * (led[3] / npes) / run_ms);
      std::printf("  phases (total ms): force=%.1f kick=%.1f resort=%.1f "
                  "build=%.1f halo=%.1f\n", t_force, t_kick, t_resort, t_build,
                  t_halo);
      std::printf("LAMMPS_MD %s: atoms=%llu lattice=%u steps=%llu ranks=%d "
                  "ms=%.1f ms_per_step=%.3f halo_ms=%.1f E0=%.6f En=%.6f "
                  "drift=%.2e ckpt_n=%llu ckpt_ms=%.1f\n",
                  gvc::Comm::Name(), g.natoms, a.lattice, a.steps, npes,
                  run_ms, run_ms / a.steps, t_halo, e0, e_n, e_drift,
                  (unsigned long long)n_ckpt, t_ckpt + t_ckpt_io);
      std::printf("LAMMPS_MD %s: %s\n", gvc::Comm::Name(),
                  ok ? "ALL GATES PASS" : "GATE FAILURE");
    }
    const int rc = comm.Verdict(ok ? 0 : 1);
    comm.Finalize();
    return rc;
  }

  // ---- the ballistic run --------------------------------------------------
  const float fgx = static_cast<float>(a.g[0]);
  const float fgy = static_cast<float>(a.g[1]);
  const float fgz = static_cast<float>(a.g[2]);
  comm.Barrier();
  const double t0 = gvc::NowMs();
  for (u64 step = 0; step < a.steps; ++step) {
    IntegrateKernel(q, a.blocks, a.threads, dx, dv, nullptr, fdt, fgx, fgy,
                    fgz, 1, local_slots);
    IntegrateKernel(q, a.blocks, a.threads, dx, dv, nullptr, fdt, fgx, fgy,
                    fgz, 0, local_slots);
  }
  comm.Barrier();
  const double run_ms = gvc::NowMs() - t0;

  double thermo[4] = {0, 0, 0, 0};
  q.memset(d_thermo, 0, 4 * sizeof(double)).wait();
  ThermoKernel(q, a.blocks, a.threads, dx, dv, d_thermo, local_slots);
  q.memcpy(thermo, d_thermo, sizeof(thermo)).wait();
  comm.HostSumN(thermo, 4);

  int rc = 0;
  if (a.gate) {
    std::vector<float> gx_out(local_elems), gv_out(local_elems);
    q.memcpy(gx_out.data(), dx, local_elems * sizeof(float));
    q.memcpy(gv_out.data(), dv, local_elems * sizeof(float));
    q.wait();
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
    comm.HostSumN(gate, 7);
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
    std::printf("LAMMPS_MD %s: %s\n", gvc::Comm::Name(),
                rc == 0 ? "ALL GATES PASS" : "GATE FAILURE");
  }
  rc = comm.Verdict(rc);
  comm.Finalize();
  return rc;
}
