/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * lammps_md paged kernels, MACRO (Duff's device) form.
 *
 * Same workload as md_kernels.h with suspension expressed through
 * yield_stack.h's macros rather than C++20 device coroutines, which clang
 * cannot compile for SPIR-V. See kmeans_macros_kernels.h for the rationale.
 *
 * WHAT GOES IN THE FRAME, AND WHAT DOES NOT. Declarations placed BEFORE
 * CLIO_YBEGIN re-run on every entry, including every resume, so anything
 * derived purely from the parameters (row_elems, c2, halfL, the scratch
 * handle) stays an ordinary local and costs no frame space. Only state that
 * is MUTATED and read across a suspension needs CLIO_YLOCAL -- loop
 * induction variables, accumulators, and every page guard.
 *
 * This file is the densest of the six: ForceMacro alone keeps eighteen page
 * guards (hg[9][2]) plus four nine-element side arrays live across a chain of
 * suspensions inside a loop that itself suspends.
 */
#ifndef CLIO_GV_BENCH_MD_MACROS_KERNELS_H_
#define CLIO_GV_BENCH_MD_MACROS_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

#include "md_kernels.h"

/* md_kernels.h keeps everything at GLOBAL scope (the gv/gy aliases come from
 * md_common.h), so these do too -- the Macro-suffixed names cannot collide
 * with the Coro-suffixed ones, and the plain helpers are already visible. */

using VecF = gv::DeviceVector<float>;
using VecI = gv::DeviceVector<int>;
using HeldF = gv::Held<float>;
/** CLIO_YLOCAL takes one macro argument, so array types need names. */
using HeldF9x2 = HeldF[9][2];
using U64x9 = u64[9];
using CFPtr9 = const float *[9];
using HeldF6x2 = HeldF[6][2];
using U64x6 = u64[6];
using CFPtr6 = const float *[6];
using U32x9 = u32[9];
using U32x2 = u32[2];
using U32x6 = u32[6];
using HeldI = gv::Held<int>;
using HeldIxN = HeldI[kMaxNlGuards];
using IPtrN = int *[kMaxNlGuards];
using U64xN = u64[kMaxNlGuards];
using CIPtrN = const int *[kMaxNlGuards];

/** Macro-form AdmitSpans: wait until the admission pool has room. */
CTP_GPU_FUN inline void AdmitSpansMacro(u32 need, u32 pool, u32 slack) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u32, rounds, 0u);
  // Declared in the frame rather than as an automatic: the resume label sits
  // after it in the loop body, and a jump may not cross an initialisation.
  CLIO_YLOCAL_INIT(bool, wait, false);
  CLIO_YBEGIN();
  if (need != 0u) {
    for (;;) {
      wait = false;
      if (threadIdx.x == 0) {
        wait = AdmitWaits(need, pool, slack);
        if (wait && ++rounds == 100000u) {
          rounds = 0;
          MD_DEV_PRINTF(
              "[gpu_vector] admit stall: need=%u pool=%u slack=%u used=%u\n",
              need, pool, slack, *(volatile u32 *)&MdG().adm_used);
        }
      }
      __syncthreads();
      if (!__syncthreads_or(wait ? 1 : 0)) break;
      CLIO_YIELD_STACK();
    }
  }
  CLIO_YEND();
}

/** Macro-form ForceCoro: the direct O(N^2)-per-stencil force pass. */
CTP_GPU_FUN inline void ForceMacro(VecF x, VecF f, u32 nb, u32 cap, float box,
                                   float cutoff, int eflag, double *acc,
                                   u32 z0, u32 z1, u32 nblocks, u32 block,
                                   u64 hgen, bool force_all) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, row, 0);
  CLIO_YLOCAL_INIT(int, q, 0);
  CLIO_YLOCAL_INIT(u32, by, 0);
  CLIO_YLOCAL_INIT(u32, bz, 0);
  CLIO_YLOCAL_INIT(bool, halo, false);
  CLIO_YLOCAL(HeldF9x2, hg);
  CLIO_YLOCAL(U64x9, rbase);
  CLIO_YLOCAL(U64x9, rrun0);
  CLIO_YLOCAL(CFPtr9, rp0);
  CLIO_YLOCAL(CFPtr9, rp1);
  CLIO_YLOCAL_INIT(u64, fbase, 0);
  CLIO_YLOCAL_INIT(u64, frun0, 0);
  CLIO_YLOCAL(HeldF, hf0);
  CLIO_YLOCAL(HeldF, hf1);
  // Accumulate across rows, so across suspensions.
  CLIO_YLOCAL_INIT(double, pe, 0.0);
  CLIO_YLOCAL_INIT(double, w, 0.0);
  CLIO_YLOCAL_INIT(double, npairs, 0.0);
  // Derived only from parameters: re-run on every entry, so no frame cost.
  MD_RED_SCRATCH(red);
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 row_lo = static_cast<u64>(z0) * nb;
  const u64 row_hi = static_cast<u64>(z1) * nb;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  CLIO_YBEGIN();
  for (row = row_lo + block; row < row_hi; row += nblocks) {
    by = static_cast<u32>(row % nb);
    bz = static_cast<u32>(row / nb);
    for (q = 0; q < 9; ++q) {
      // Compute-only scope: a resume jumps to the case labels BELOW it, so
      // nothing declared here may be needed afterwards. rbase[q] and halo are
      // -- CLIO_YCALL re-executes its call on resume -- so both live in the
      // frame and only the scratch stays local.
      {
        const int dz = q / 3 - 1, dy = q % 3 - 1;
        const u32 wz = (bz + nb + dz) % nb;
        const u32 wy = (by + nb + dy) % nb;
        const u64 rowbin0 = (static_cast<u64>(wz) * nb + wy) * nb;
        rbase[q] = rowbin0 * cap * kStride;
        halo = force_all || (wz < z0 || wz >= z1);
      }
      CLIO_YCALL(x.MFetch(halo ? hgen : 0, rbase[q], row_elems));
      CLIO_YCALL(x.MHoldPage(&hg[q][0], rbase[q], row_elems));
      rrun0[q] = hg[q][0].run();
      if (rrun0[q] < row_elems) {
        CLIO_YCALL(x.MHoldPage(&hg[q][1], rbase[q] + rrun0[q],
                               row_elems - rrun0[q]));
      }
      rp0[q] = hg[q][0].ptr();
      rp1[q] = hg[q][1] ? hg[q][1].ptr() : nullptr;
    }
    fbase =
        (static_cast<u64>(bz) * nb + by) * static_cast<u64>(nb) * cap * kStride;
    CLIO_YCALL(f.MFetch(0, fbase, row_elems));
    CLIO_YCALL(f.MHoldPage(&hf0, fbase, row_elems, true));
    frun0 = hf0.run();
    if (frun0 < row_elems) {
      CLIO_YCALL(f.MHoldPage(&hf1, fbase + frun0, row_elems - frun0, true));
    }
    {
      float *const fp0 = hf0.ptr();
      float *const fp1 = hf1 ? hf1.ptr() : nullptr;
      for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) {
        (e < frun0 ? fp0[e] : fp1[e - frun0]) = 0.0f;
      }
      __syncthreads();
      const u64 islots = static_cast<u64>(nb) * cap;
      for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
        const u64 ei = s * kStride;
        const float *const ip =
            (ei < rrun0[4]) ? rp0[4] + ei : rp1[4] + (ei - rrun0[4]);
        if (ip[3] < 0.0f) continue;   // padded slot
        const float xi = ip[0], yi = ip[1], zi = ip[2];
        const u32 bx = static_cast<u32>(s / cap);
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int qq = 0; qq < 9; ++qq) {
          for (int dxx = -1; dxx <= 1; ++dxx) {
            const u32 jbx = (bx + nb + dxx) % nb;
            const u64 jb = static_cast<u64>(jbx) * cap * kStride;
            for (u32 sj = 0; sj < cap; ++sj) {
              const u64 ej = jb + static_cast<u64>(sj) * kStride;
              const float *const jp =
                  (ej < rrun0[qq]) ? rp0[qq] + ej : rp1[qq] + (ej - rrun0[qq]);
              const float tj = jp[3];
              if (tj < 0.0f) continue;
              if (qq == 4 && jbx == bx && sj == s % cap) continue;   // self
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
        float *const op = (ei < frun0) ? fp0 + ei : fp1 + (ei - frun0);
        op[0] = fx;
        op[1] = fy;
        op[2] = fz;
      }
      __syncthreads();
      for (int qq = 0; qq < 9; ++qq) x.UnpinRange(rbase[qq], row_elems);
      f.UnpinRange(fbase, row_elems);
    }
  }  // guards die per row
  if (eflag) {
    const double vals[3] = {pe, w, npairs};
    for (int qq = 0; qq < 3; ++qq) {
      red[threadIdx.x] = vals[qq];
      __syncthreads();
      for (u32 wd = blockDim.x / 2; wd > 0; wd >>= 1) {
        if (threadIdx.x < wd) red[threadIdx.x] += red[threadIdx.x + wd];
        __syncthreads();
      }
      if (threadIdx.x == 0) atomicAdd(&acc[qq], red[0]);
      __syncthreads();
    }
  }
  CLIO_YEND();
}

/** Macro-form SentinelCoro: stamp the padded slots of this block's slab. */
CTP_GPU_FUN inline void SentinelMacro(VecF dst, u32 nb, u32 cap, u32 z0,
                                      u32 z1, u32 nblocks, u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, pg, 0);
  CLIO_YLOCAL_INIT(u64, e0, 0);
  CLIO_YLOCAL_INIT(u64, e1, 0);
  CLIO_YLOCAL(HeldF, h);
  const u64 epp = dst.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  CLIO_YBEGIN();
  for (pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    CLIO_YCALL(dst.MFetch(0, pg * epp, epp));
    CLIO_YCALL(dst.MHoldPage(&h, pg * epp, epp, /*write=*/true));
    // Braced: a CLIO_YCALL case label follows in this block.
    {
      float *const p = h.ptr();
      const u64 s_lo = (e0 - pg * epp) / kStride;
      const u64 s_hi = (e1 - pg * epp) / kStride;
      for (u64 s = s_lo + threadIdx.x; s < s_hi; s += blockDim.x) {
        p[s * kStride + 3] = -1.0f;
      }
    }
    __syncthreads();
    if (MdG().md_flush) {
      CLIO_YCALL(dst.MBeginFlush(0, e0, e1 - e0));
    }
    dst.UnpinRange(pg * epp, epp);
  }
  if (MdG().md_flush) {
    CLIO_YCALL(dst.MEndFlush());
  }
  CLIO_YEND();
}

/** Macro-form HaloPinCoro: admit and pin the two halo planes. */
CTP_GPU_FUN inline void HaloPinMacro(VecF x, u32 nb, u32 cap, u32 z0, u32 z1,
                                     u64 gen, u32 block) {
  CLIO_YFRAME();
  // Read back by CLIO_YCALL when it re-executes its call on resume.
  CLIO_YLOCAL_INIT(u64, below, 0);
  CLIO_YLOCAL_INIT(u64, above, 0);
  CLIO_YLOCAL_INIT(u64, plane_elems_p, 0);
  CLIO_YLOCAL_INIT(u32, need, 0u);
  CLIO_YBEGIN();
  if (block == 0) {
    {
      const u64 row_elems_p = static_cast<u64>(nb) * cap * kStride;
      plane_elems_p = static_cast<u64>(nb) * row_elems_p;
      below = static_cast<u64>((z0 + nb - 1u) % nb) * plane_elems_p;
      above = static_cast<u64>(z1 % nb) * plane_elems_p;
      need = PagesSpanned(x, below, plane_elems_p) +
             PagesSpanned(x, above, plane_elems_p);
    }
    CLIO_YCALL(AdmitSpansMacro(need, x.Regions(), 0u));
    CLIO_YCALL(x.MFetch(gen, below, plane_elems_p, above, plane_elems_p));
  }
  CLIO_YEND();
}

/** Macro-form RefaultWriteCoro. */
CTP_GPU_FUN inline void RefaultWriteMacro(VecF x, u64 pg_lo, u64 pg_hi,
                                          u64 round, u64 ppp, u32 nblocks,
                                          u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, pg, 0);
  CLIO_YLOCAL(HeldF, h);
  const u64 epp = x.ElemsPerPage();
  CLIO_YBEGIN();
  for (pg = pg_lo + block; pg < pg_hi; pg += nblocks) {
    CLIO_YCALL(x.MFetch(0, pg * epp, epp));
    CLIO_YCALL(x.MHoldPage(&h, pg * epp, epp, /*write=*/true));
    {
      float *p = h.ptr();
      for (u64 e = threadIdx.x; e < epp; e += blockDim.x) {
        p[e] = RefaultPattern(pg, round, e);
      }
    }
    __syncthreads();
    CLIO_YCALL(x.MBeginFlush(0, pg * epp, epp));
    x.UnpinRange(pg * epp, epp);
  }
  CLIO_YCALL(x.MEndFlush());
  if (ppp != 0 && block == 0) {
    CLIO_YCALL(x.MBeginFlush(round + 1, pg_lo * epp, ppp * epp,
                             (pg_hi - ppp) * epp, ppp * epp));
    CLIO_YCALL(x.MEndFlush());
  }
  CLIO_YEND();
}

/** Macro-form HaloUnpinCoro. No suspension of its own, but it keeps the
 *  frame shape so callers can reach it through CLIO_YCALL like the rest. */
CTP_GPU_FUN inline void HaloUnpinMacro(VecF x, u32 nb, u32 cap, u32 z0,
                                       u32 z1, u32 block) {
  CLIO_YFRAME();
  CLIO_YBEGIN();
  if (block == 0) {
    const u64 row_elems_p = static_cast<u64>(nb) * cap * kStride;
    const u64 plane_elems_p = static_cast<u64>(nb) * row_elems_p;
    const u64 below = static_cast<u64>((z0 + nb - 1u) % nb) * plane_elems_p;
    const u64 above = static_cast<u64>(z1 % nb) * plane_elems_p;
    const u32 need = PagesSpanned(x, below, plane_elems_p) +
                     PagesSpanned(x, above, plane_elems_p);
    x.UnpinRange(below, plane_elems_p);
    x.UnpinRange(above, plane_elems_p);
    ReleaseSpans(need);
  }
  CLIO_YEND();
}

/** Macro-form RebinWrapCoro: wrap positions back into the box. */
CTP_GPU_FUN inline void RebinWrapMacro(VecF x, u32 nb, u32 cap, float box,
                                       u32 z0, u32 z1, u32 nblocks,
                                       u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, pg, 0);
  CLIO_YLOCAL_INIT(u64, e0, 0);
  CLIO_YLOCAL_INIT(u64, e1, 0);
  CLIO_YLOCAL(HeldF, hx);
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  CLIO_YBEGIN();
  for (pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    CLIO_YCALL(x.MFetch(0, pg * epp, epp));
    CLIO_YCALL(x.MHoldPage(&hx, pg * epp, epp, /*write=*/true));
    // Braced: the conditional MBeginFlush below puts a case label in this
    // block, and a jump to it may not cross these initialisations.
    {
      float *const px = hx.ptr();
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
    }
    __syncthreads();
    if (MdG().md_flush) {
      CLIO_YCALL(x.MBeginFlush(0, e0, e1 - e0));
    }
    x.UnpinRange(pg * epp, epp);
  }
  if (MdG().md_flush) {
    CLIO_YCALL(x.MEndFlush());
  }
  CLIO_YEND();
}

/** Macro-form ReadProbeCoro: the streaming read-verification probe. */
CTP_GPU_FUN inline void ReadProbeMacro(VecF x, u64 passes, u32 nblocks,
                                       u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, it, 0);
  CLIO_YLOCAL_INIT(u64, pg, 0);
  CLIO_YLOCAL(HeldF, h);
  const u64 epp = x.ElemsPerPage();
  const u64 npages = (x.size() + epp - 1) / epp;
  CLIO_YBEGIN();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    MdG().read_geom[1] = x.SetSize();
    MdG().read_geom[2] = x.PageBytes();
    MdG().read_geom[3] = epp;
  }
  for (it = 0; it < passes; ++it) {
    for (pg = block; pg < npages; pg += nblocks) {
      CLIO_YCALL(x.MFetch(0, pg * epp, epp));
      CLIO_YCALL(x.MHoldPage(&h, pg * epp, epp));  // READ hold only
      const float *const p = h.ptr();
      for (u64 i = threadIdx.x; i < epp; i += blockDim.x) {
        const float seen = MdG().probe_ldcg ? ProbeLoadCG(&p[i]) : p[i];
        if (seen != ProbeVal(pg * epp + i)) {
          atomicAdd(&MdG().read_bad[0], 1ull);
          if (ProbeLoadCV(&p[i]) == ProbeVal(pg * epp + i)) {
            atomicAdd(&MdG().read_bad[1], 1ull);
          }
        }
      }
      __syncthreads();
      x.UnpinRange(pg * epp, epp);
    }
  }
  CLIO_YEND();
}

/** Macro-form ThermoCoro: kinetic energy and momentum over the slab. */
CTP_GPU_FUN inline void ThermoMacro(VecF x, VecF v, double *out, u32 nb,
                                    u32 cap, u32 z0, u32 z1, u32 nblocks,
                                    u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, pg, 0);
  CLIO_YLOCAL_INIT(u64, e0, 0);
  CLIO_YLOCAL_INIT(u64, e1, 0);
  CLIO_YLOCAL(HeldF, hx);
  CLIO_YLOCAL(HeldF, hv);
  // Accumulate across pages, so across suspensions.
  CLIO_YLOCAL_INIT(double, ke, 0.0);
  CLIO_YLOCAL_INIT(double, mx, 0.0);
  CLIO_YLOCAL_INIT(double, my, 0.0);
  CLIO_YLOCAL_INIT(double, mz, 0.0);
  MD_RED_SCRATCH(red);
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  CLIO_YBEGIN();
  for (pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    CLIO_YCALL(x.MFetch(0, pg * epp, epp));
    CLIO_YCALL(x.MHoldPage(&hx, pg * epp, epp));
    CLIO_YCALL(v.MFetch(0, pg * epp, epp));
    CLIO_YCALL(v.MHoldPage(&hv, pg * epp, epp));
    const float *const px = hx.ptr();
    const float *const pv = hv.ptr();
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
  {
    const double vals[4] = {ke, mx, my, mz};
    for (int q4 = 0; q4 < 4; ++q4) {
      red[threadIdx.x] = vals[q4];
      __syncthreads();
      for (u32 w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
        __syncthreads();
      }
      if (threadIdx.x == 0) atomicAdd(&out[q4], red[0]);
      __syncthreads();
    }
  }
  CLIO_YEND();
}

/** Macro-form RebinAssignCoro: assign each atom to its destination bin. */
CTP_GPU_FUN inline void RebinAssignMacro(VecF x, u32 nb, u32 cap, float box,
                                         u32 *bincnt, u32 *d_dest, int *d_err,
                                         u32 z0, u32 z1, u32 nblocks,
                                         u32 block, u64 hgen) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u32, i, 0u);
  CLIO_YLOCAL_INIT(u64, pg, 0);
  // wz/in_mine/p_lo/pg_lo/pg_hi are set in the OUTER loop body and read by the
  // inner loop's condition and by CLIO_YCALL when it re-executes on resume, so
  // the resume jump would otherwise cross their initialisation.
  CLIO_YLOCAL_INIT(u32, wz, 0u);
  CLIO_YLOCAL_INIT(bool, in_mine, false);
  CLIO_YLOCAL_INIT(u64, p_lo, 0);
  CLIO_YLOCAL_INIT(u64, pg_lo, 0);
  CLIO_YLOCAL_INIT(u64, pg_hi, 0);
  CLIO_YLOCAL(HeldF, hx);
  const u64 epp = x.ElemsPerPage();
  const float fnb = static_cast<float>(nb);
  const u64 row_elems = static_cast<u64>(nb) * cap * kStride;
  const u64 plane_elems = static_cast<u64>(nb) * row_elems;
  const u64 islots_r = static_cast<u64>(nb) * cap;
  const u32 mine = z1 - z0;
  const bool whole = (mine >= nb);          // single node owns everything
  const u64 total_elems = static_cast<u64>(nb) * plane_elems;
  const u64 npages_all = (total_elems + epp - 1) / epp;
  const u32 nscan = whole ? 1u : (mine + 2u);
  CLIO_YBEGIN();
  for (i = 0; i < nscan; ++i) {
    wz = whole ? 0u : ((z0 + nb - 1u + i) % nb);
    in_mine = whole || (wz >= z0 && wz < z1);
    p_lo = whole ? 0ull : static_cast<u64>(wz) * plane_elems;
    pg_lo = whole ? 0ull : (p_lo / epp);
    pg_hi = whole ? npages_all : ((p_lo + plane_elems + epp - 1) / epp);
    for (pg = pg_lo + block; pg < pg_hi; pg += nblocks) {
      CLIO_YCALL(x.MFetch(in_mine ? 0 : hgen, pg * epp, epp));
      CLIO_YCALL(x.MHoldPage(&hx, pg * epp, epp));
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
        const u64 srow = (slot0 + s) / islots_r;
        const u32 sby = static_cast<u32>(srow % nb);
        const u32 sbz = static_cast<u32>(srow / nb);
        const u32 dy = MdMinU32((by + nb - sby) % nb, (sby + nb - by) % nb);
        const u32 dz = MdMinU32((bz + nb - sbz) % nb, (sbz + nb - bz) % nb);
        if (dy > 1u || dz > 1u) *d_err = 2;
      }
      __syncthreads();
      x.UnpinRange(pg * epp, epp);
    }
  }
  CLIO_YEND();
}

/** Macro-form IntegrateCoro: the velocity-Verlet half-kick (and drift). */
CTP_GPU_FUN inline void IntegrateMacro(VecF x, VecF v, VecF third,
                                       int use_third, float dt, float gx,
                                       float gy_, float gz, int drift,
                                       u64 pg_lo, u64 pg_hi, u32 nblocks,
                                       u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, pg, 0);
  CLIO_YLOCAL_INIT(u64, cnt, 0);
  CLIO_YLOCAL(HeldF, hx);
  CLIO_YLOCAL(HeldF, hv);
  CLIO_YLOCAL(HeldF, ht);
  const u64 epp = x.ElemsPerPage();
  const float half = 0.5f * dt;
  CLIO_YBEGIN();
  // THIS NODE'S SLAB ONLY, [pg_lo, pg_hi): every page is published by name
  // into one store shared by every node, so integrating the whole lattice
  // overwrites the neighbour's slab. Single node passes [0, npages).
  for (pg = pg_lo + block; pg < pg_hi; pg += nblocks) {
    CLIO_YCALL(x.MFetch(0, pg * epp, epp));
    CLIO_YCALL(x.MHoldPage(&hx, pg * epp, epp, /*write=*/true));
    CLIO_YCALL(v.MFetch(0, pg * epp, epp));
    CLIO_YCALL(v.MHoldPage(&hv, pg * epp, epp, /*write=*/true));
    if (use_third) {
      CLIO_YCALL(third.MFetch(0, pg * epp, epp));
      CLIO_YCALL(third.MHoldPage(&ht, pg * epp, epp));
    }
    // Braced: the conditional MBeginFlush below places case labels in this
    // block, so nothing initialised here may sit above them.
    {
      float *const px = hx.ptr();
      float *const pv = hv.ptr();
      if (use_third && threadIdx.x == 0) {
        atomicAdd(
            &MdG().third_sink,
            static_cast<unsigned long long>(__float_as_uint(ht.ptr()[0])));
      }
      const u64 nslots = epp / kStride;
      for (u64 s = threadIdx.x; s < nslots; s += blockDim.x) {
        const u64 e = s * kStride;
        if (px[e + 3] < 0.0f) continue;   // padded slot
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
        atomicAdd(&MdG().pages_done, 1ull);
        if (block < 64) {
          atomicAdd(&MdG().blk_done[block], 1ull);
          MdG().blk_last[block] = pg;
        }
      }
    }
    cnt = (x.size() - pg * epp < epp) ? x.size() - pg * epp : epp;
    if (MdG().publish && MdG().pub_interior) {
      CLIO_YCALL(x.MBeginFlush(0, pg * epp, cnt));
      CLIO_YCALL(v.MBeginFlush(0, pg * epp, cnt));
    }
    x.UnpinRange(pg * epp, epp);
    v.UnpinRange(pg * epp, epp);
    if (use_third) third.UnpinRange(pg * epp, epp);
  }
  CLIO_YCALL(x.MEndFlush());
  CLIO_YCALL(v.MEndFlush());
  CLIO_YEND();
}

/** Macro-form MDIntegrateCoro: the force half-kick over this block's slab. */
CTP_GPU_FUN inline void MDIntegrateMacro(VecF x, VecF v, VecF f, float dt,
                                         int drift, u32 nb, u32 cap, u32 z0,
                                         u32 z1, u32 nblocks, u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, pg, 0);
  CLIO_YLOCAL_INIT(u64, e0, 0);
  CLIO_YLOCAL_INIT(u64, e1, 0);
  CLIO_YLOCAL(HeldF, hx);
  CLIO_YLOCAL(HeldF, hv);
  CLIO_YLOCAL(HeldF, hf);
  const u64 epp = x.ElemsPerPage();
  const Slab sl = SlabOf(nb, cap, z0, z1, epp);
  const float half = 0.5f * dt;
  CLIO_YBEGIN();
  for (pg = sl.pg_lo + block; pg < sl.pg_hi; pg += nblocks) {
    e0 = (pg * epp > sl.lo) ? pg * epp : sl.lo;
    e1 = ((pg + 1) * epp < sl.hi) ? (pg + 1) * epp : sl.hi;
    if (e1 <= e0) continue;
    CLIO_YCALL(x.MFetch(0, pg * epp, epp));
    CLIO_YCALL(x.MHoldPage(&hx, pg * epp, epp, /*write=*/true));
    CLIO_YCALL(v.MFetch(0, pg * epp, epp));
    CLIO_YCALL(v.MHoldPage(&hv, pg * epp, epp, /*write=*/true));
    CLIO_YCALL(f.MFetch(0, pg * epp, epp));
    CLIO_YCALL(f.MHoldPage(&hf, pg * epp, epp));
    // Braced: the conditional MBeginFlush below adds case labels here.
    {
      float *const px = hx.ptr();
      float *const pv = hv.ptr();
      const float *const pf = hf.ptr();
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
    }
    __syncthreads();
    if (MdG().md_flush) {
      if (drift) {
        CLIO_YCALL(x.MBeginFlush(0, e0, e1 - e0));
      }
      CLIO_YCALL(v.MBeginFlush(0, e0, e1 - e0));
    }
    x.UnpinRange(pg * epp, epp);
    v.UnpinRange(pg * epp, epp);
    f.UnpinRange(pg * epp, epp);
  }
  if (MdG().md_flush) {
    if (drift) {
      CLIO_YCALL(x.MEndFlush());
    }
    CLIO_YCALL(v.MEndFlush());
  }
  CLIO_YEND();
}

/** Macro-form PublishSlabCoro: publish the owned planes and pin the halo.
 *
 *  The three arms are selected by block id. A resume jumps straight to a case
 *  label inside whichever arm suspended, skipping the `if` -- which is correct
 *  because `block` is uniform for the life of the launch. Everything each arm
 *  needs after its first suspension therefore lives in the frame. */
CTP_GPU_FUN inline void PublishSlabMacro(VecF x, VecF v, u32 nb, u32 cap,
                                         u32 z0, u32 z1, u64 gen,
                                         u32 halo_first, u32 nblocks,
                                         u32 block) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u32, pub_need, 0u);
  CLIO_YLOCAL_INIT(long long, c0, 0);
  CLIO_YLOCAL_INIT(long long, c1, 0);
  const u64 epp = x.ElemsPerPage();
  const u64 row_elems_p = static_cast<u64>(nb) * cap * kStride;
  const u64 plane_elems_p = static_cast<u64>(nb) * row_elems_p;
  const u64 lo_pl = static_cast<u64>(z0) * plane_elems_p;
  const u64 hi_pl = static_cast<u64>(z1 - 1) * plane_elems_p;
  const u64 below = static_cast<u64>((z0 + nb - 1u) % nb) * plane_elems_p;
  const u64 above = static_cast<u64>(z1 % nb) * plane_elems_p;
  (void)epp;
  CLIO_YBEGIN();
  if (block == 0) {
    pub_need = PagesSpanned(x, lo_pl, plane_elems_p) +
               PagesSpanned(x, hi_pl, plane_elems_p);
    CLIO_YCALL(AdmitSpansMacro(pub_need, x.Regions(), 0u));
    c0 = clock64();
    CLIO_YCALL(x.MFetch(0, lo_pl, plane_elems_p, hi_pl, plane_elems_p));
    CLIO_YCALL(x.MBeginFlush(gen, lo_pl, plane_elems_p, hi_pl, plane_elems_p));
    CLIO_YCALL(x.MEndFlush());
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().pub_flush_cyc,
                (unsigned long long)(clock64() - c0));
    }
    x.UnpinRange(lo_pl, plane_elems_p);
    x.UnpinRange(hi_pl, plane_elems_p);
    ReleaseSpans(pub_need);
  } else if (block == 1) {
    CLIO_YCALL(v.MFetch(0, lo_pl, plane_elems_p, hi_pl, plane_elems_p));
    CLIO_YCALL(v.MBeginFlush(gen, lo_pl, plane_elems_p, hi_pl, plane_elems_p));
    CLIO_YCALL(v.MEndFlush());
    v.UnpinRange(lo_pl, plane_elems_p);
    v.UnpinRange(hi_pl, plane_elems_p);
  } else if (block == 2) {
    pub_need = PagesSpanned(x, below, plane_elems_p) +
               PagesSpanned(x, above, plane_elems_p);
    if (halo_first != 0u) {
      CLIO_YCALL(AdmitSpansMacro(pub_need, x.Regions(), 0u));
    }
    c1 = clock64();
    CLIO_YCALL(x.MFetch(gen, below, plane_elems_p, above, plane_elems_p));
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().pub_fetch_cyc,
                (unsigned long long)(clock64() - c1));
    }
    if (halo_first == 0u) {
      x.UnpinRange(below, plane_elems_p);
      x.UnpinRange(above, plane_elems_p);
    }
  }
  CLIO_YEND();
}

/** Macro-form RefaultVerifyCoro. */
CTP_GPU_FUN inline void RefaultVerifyMacro(VecF x, u64 pg_lo, u64 pg_hi,
                                           u64 round, u64 ppp, u64 below_pg,
                                           u64 above_pg, u32 nblocks,
                                           u32 block,
                                           unsigned long long *d_out) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, pg, 0);
  CLIO_YLOCAL_INIT(u32, half, 0u);
  CLIO_YLOCAL_INIT(u64, base, 0);
  CLIO_YLOCAL_INIT(u64, hpg, 0);
  CLIO_YLOCAL(HeldF, h);
  const u64 epp = x.ElemsPerPage();
  CLIO_YBEGIN();
  for (pg = pg_lo + block; pg < pg_hi; pg += nblocks) {
    CLIO_YCALL(x.MFetch(0, pg * epp, epp));
    CLIO_YCALL(x.MHoldPage(&h, pg * epp, epp));
    const float *p = h.ptr();
    for (u64 e = threadIdx.x; e < epp; e += blockDim.x) {
      const float want = RefaultPattern(pg, round, e);
      if (p[e] != want) {
        atomicAdd(&d_out[0], 1ull);
        if (atomicCAS((unsigned long long *)&d_out[1], 0ull, 1ull) == 0ull) {
          d_out[2] = pg;
          d_out[3] = e;
          d_out[4] = (unsigned long long)__float_as_uint(p[e]);
        }
      }
    }
    __syncthreads();
    x.UnpinRange(pg * epp, epp);
  }
  if (ppp != 0 && block == 0) {
    CLIO_YCALL(x.MFetch(round + 1, below_pg * epp, ppp * epp,
                        above_pg * epp, ppp * epp));
    for (half = 0; half < 2; ++half) {
      base = (half == 0 ? below_pg : above_pg);
      for (hpg = base; hpg < base + ppp; ++hpg) {
        CLIO_YCALL(x.MHoldPage(&h, hpg * epp, epp));
        const float *p = h.ptr();
        for (u64 e = threadIdx.x; e < epp; e += blockDim.x) {
          const int r = RefaultDecode(hpg, e, p[e]);
          if (r == (int)round) {
            atomicAdd(&d_out[5], 1ull);
          } else if (r > (int)round) {
            atomicAdd(&d_out[6], 1ull);
          } else if (r >= 0) {
            atomicAdd(&d_out[7], 1ull);
          } else {
            atomicAdd(&d_out[8], 1ull);
          }
        }
        __syncthreads();
      }
    }
    x.UnpinRange(below_pg * epp, ppp * epp);
    x.UnpinRange(above_pg * epp, ppp * epp);
  }
  CLIO_YEND();
}

/** Macro-form GatherCoro: compact each source row into its destination bins.
 *
 *  The heaviest frame in the file: 24 page guards (hs/hx) plus the per-span
 *  side tables and the nine-way q-> span map, every one of them written before
 *  a suspension and read after it. rl/rn/nr and wz are frame-resident for the
 *  same reason the compiler would otherwise refuse the resume label -- they
 *  are set in the dz loop and consumed inside the t loop, which suspends. */
CTP_GPU_FUN inline void GatherMacro(VecF src, VecF srcx, VecF dst, u32 nb,
                                    u32 cap, const u32 *d_dest, int keep_w,
                                    u32 z0, u32 z1, u32 nblocks, u32 block,
                                    u64 hgen) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, row, 0);
  CLIO_YLOCAL_INIT(u32, by, 0u);
  CLIO_YLOCAL_INIT(u32, bz, 0u);
  CLIO_YLOCAL_INIT(u64, drun, 0);
  CLIO_YLOCAL(HeldF, hd0);
  CLIO_YLOCAL(HeldF, hd1);
  CLIO_YLOCAL(HeldF6x2, hs);
  CLIO_YLOCAL(HeldF6x2, hx);
  CLIO_YLOCAL(U64x6, srun);
  CLIO_YLOCAL(U64x6, xrun);
  CLIO_YLOCAL(CFPtr6, sp0);
  CLIO_YLOCAL(CFPtr6, sp1);
  CLIO_YLOCAL(CFPtr6, xp0);
  CLIO_YLOCAL(CFPtr6, xp1);
  CLIO_YLOCAL(U32x9, qspan);
  CLIO_YLOCAL(U64x9, qoff);
  CLIO_YLOCAL(U64x9, srow);
  CLIO_YLOCAL_INIT(u32, nspans, 0u);
  CLIO_YLOCAL_INIT(int, dz, 0);
  CLIO_YLOCAL_INIT(u32, t, 0u);
  CLIO_YLOCAL_INIT(u32, nr, 0u);
  CLIO_YLOCAL(U32x2, rl);
  CLIO_YLOCAL(U32x2, rn);
  CLIO_YLOCAL_INIT(u64, rb, 0);
  CLIO_YLOCAL_INIT(u64, len, 0);
  CLIO_YLOCAL_INIT(u32, wz, 0u);
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 row_elems = islots * kStride;
  const u64 row_lo = static_cast<u64>(z0) * nb;
  const u64 row_hi = static_cast<u64>(z1) * nb;
  CLIO_YBEGIN();
  for (row = row_lo + block; row < row_hi; row += nblocks) {
    by = static_cast<u32>(row % nb);
    bz = static_cast<u32>(row / nb);
    // THE ONLY WRITE HOLD: this block's own destination row.
    CLIO_YCALL(dst.MFetch(0, row * row_elems, row_elems));
    CLIO_YCALL(dst.MHoldPage(&hd0, row * row_elems, row_elems,
                             /*write=*/true));
    drun = hd0.run();
    if (drun < row_elems) {
      CLIO_YCALL(dst.MFetch(0, dst.PageLo(row * row_elems + drun),
                            dst.PageSpan(row * row_elems + drun, 1)));
      CLIO_YCALL(dst.MHoldPage(&hd1, row * row_elems + drun,
                               row_elems - drun, /*write=*/true));
    }
    {
      float *const dp0 = hd0.ptr();
      float *const dp1 = hd1 ? hd1.ptr() : nullptr;
      for (u64 s = threadIdx.x; s < islots; s += blockDim.x) {
        const u64 de = s * kStride;
        float *const dp = (de < drun) ? dp0 + de : dp1 + (de - drun);
        dp[3] = -1.0f;
      }
    }
    __syncthreads();
    nspans = 0;
    for (dz = -1; dz <= 1; ++dz) {
      {
        wz = (bz + nb + dz) % nb;
        const int lo = static_cast<int>(by) - 1;
        const int hi = static_cast<int>(by) + 1;
        nr = 0;
        if (lo < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo);
          rn[nr] = nb - static_cast<u32>(lo); ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo); rn[nr] = 3u; ++nr;
        }
      }
      for (t = 0; t < nr; ++t) {
        rb = (static_cast<u64>(wz) * nb + rl[t]) * row_elems;
        len = static_cast<u64>(rn[t]) * row_elems;
        CLIO_YCALL(src.MFetch((wz < z0 || wz >= z1) ? hgen : 0,
                              src.PageLo(rb), src.PageSpan(rb, len)));
        CLIO_YCALL(src.MHoldPage(&hs[nspans][0], rb, len));
        srun[nspans] = hs[nspans][0].run();
        if (srun[nspans] < len) {
          CLIO_YCALL(src.MHoldPage(&hs[nspans][1], rb + srun[nspans],
                                   len - srun[nspans]));
        }
        sp0[nspans] = hs[nspans][0].ptr();
        sp1[nspans] = hs[nspans][1] ? hs[nspans][1].ptr() : nullptr;
        CLIO_YCALL(srcx.MFetch((wz < z0 || wz >= z1) ? hgen : 0,
                               srcx.PageLo(rb), srcx.PageSpan(rb, len)));
        CLIO_YCALL(srcx.MHoldPage(&hx[nspans][0], rb, len));
        xrun[nspans] = hx[nspans][0].run();
        if (xrun[nspans] < len) {
          CLIO_YCALL(srcx.MFetch((wz < z0 || wz >= z1) ? hgen : 0,
                                 srcx.PageLo(rb + xrun[nspans]),
                                 srcx.PageSpan(rb + xrun[nspans], 1)));
          CLIO_YCALL(srcx.MHoldPage(&hx[nspans][1], rb + xrun[nspans],
                                    len - xrun[nspans]));
        }
        xp0[nspans] = hx[nspans][0].ptr();
        xp1[nspans] = hx[nspans][1] ? hx[nspans][1].ptr() : nullptr;
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
    {
      float *const dp0 = hd0.ptr();
      float *const dp1 = hd1 ? hd1.ptr() : nullptr;
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
        atomicAdd(&MdG().gather_wrote, 1ull);
        const u64 de = (static_cast<u64>(dest) % islots) * kStride;
        float *const dp = (de < drun) ? dp0 + de : dp1 + (de - drun);
        const float *const sv =
            (e < srun[sp]) ? sp0[sp] + e : sp1[sp] + (e - srun[sp]);
        dp[0] = sv[0];
        dp[1] = sv[1];
        dp[2] = sv[2];
        dp[3] = keep_w ? sv[3] : 0.0f;
      }
    }
    __syncthreads();
    if (MdG().publish && MdG().pub_interior) {
      CLIO_YCALL(dst.MFlush(0, row * row_elems, row_elems));
    }
    {
      u32 sq = 0;
      for (int dz2 = -1; dz2 <= 1; ++dz2) {
        const u32 wz2 = (bz + nb + dz2) % nb;
        const int lo = static_cast<int>(by) - 1;
        const int hi = static_cast<int>(by) + 1;
        u32 rl2[2], rn2[2], nr2 = 0;
        if (lo < 0) {
          rl2[nr2] = 0; rn2[nr2] = static_cast<u32>(hi) + 1u; ++nr2;
          rl2[nr2] = nb - 1u; rn2[nr2] = 1u; ++nr2;
        } else if (hi > static_cast<int>(nb) - 1) {
          rl2[nr2] = static_cast<u32>(lo);
          rn2[nr2] = nb - static_cast<u32>(lo); ++nr2;
          rl2[nr2] = 0; rn2[nr2] = 1u; ++nr2;
        } else {
          rl2[nr2] = static_cast<u32>(lo); rn2[nr2] = 3u; ++nr2;
        }
        for (u32 t2 = 0; t2 < nr2 && sq < nspans; ++t2, ++sq) {
          const u64 rb2 = (static_cast<u64>(wz2) * nb + rl2[t2]) * row_elems;
          const u64 len2 = static_cast<u64>(rn2[t2]) * row_elems;
          src.UnpinRange(src.PageLo(rb2), src.PageSpan(rb2, len2));
          srcx.UnpinRange(srcx.PageLo(rb2), srcx.PageSpan(rb2, len2));
          if (xrun[sq] < len2) {
            srcx.UnpinRange(srcx.PageLo(rb2 + xrun[sq]),
                            srcx.PageSpan(rb2 + xrun[sq], 1));
          }
        }
      }
    }
    if (MdG().md_flush) {
      CLIO_YCALL(dst.MBeginFlush(0, row * row_elems, row_elems));
    }
    dst.UnpinRange(row * row_elems, row_elems);
    if (drun < row_elems) {
      dst.UnpinRange(dst.PageLo(row * row_elems + drun),
                     dst.PageSpan(row * row_elems + drun, 1));
    }
    // Guards are frame-resident, so clear them explicitly at row end.
    hd0 = {}; hd1 = {};
    for (u32 i6 = 0; i6 < 6; ++i6) {
      hs[i6][0] = {}; hs[i6][1] = {};
      hx[i6][0] = {}; hx[i6][1] = {};
    }
  }
  if (MdG().md_flush) {
    CLIO_YCALL(dst.MEndFlush());
  }
  CLIO_YEND();
}

/** Macro-form BuildListCoro: build the Verlet neighbour list for a chunk.
 *
 *  Keeps the CLIO_SHARED_PERSIST(MdTables) arena exactly as the coroutine
 *  does -- that arena is copied out on park and back on resume by the yield
 *  driver, so it needs no change for the macro mechanism. What does change is
 *  that every span table and neighbour-list guard becomes frame-resident. */
CTP_GPU_FUN inline void BuildListMacro(VecF x, VecI nl, u32 nb, u32 cap,
                                       float box, float rlist, u32 maxneigh,
                                       u32 *d_cnt, int *d_err, u32 rowchunk,
                                       u32 z0, u32 z1, u32 nblocks, u32 block,
                                       u64 hgen) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, ch, 0);
  CLIO_YLOCAL_INIT(u32, bz, 0u);
  CLIO_YLOCAL_INIT(u32, y0, 0u);
  CLIO_YLOCAL_INIT(u32, ylast, 0u);
  CLIO_YLOCAL_INIT(u32, span_guards, 0u);
  CLIO_YLOCAL(HeldF6x2, hg);
  CLIO_YLOCAL(U64x6, srun);
  CLIO_YLOCAL(CFPtr6, sp0);
  CLIO_YLOCAL(CFPtr6, sp1);
  CLIO_YLOCAL(U32x6, sbase);
  CLIO_YLOCAL(U32x6, scnt);
  CLIO_YLOCAL(U32x6, sdz);
  CLIO_YLOCAL(U64x6, sxrb);
  CLIO_YLOCAL(U64x6, sxlen);
  CLIO_YLOCAL_INIT(u32, nspans, 0u);
  CLIO_YLOCAL_INIT(int, dz, 0);
  CLIO_YLOCAL_INIT(u32, t, 0u);
  CLIO_YLOCAL_INIT(u32, nr, 0u);
  CLIO_YLOCAL(U32x2, rl);
  CLIO_YLOCAL(U32x2, rn);
  CLIO_YLOCAL_INIT(u32, wz, 0u);
  CLIO_YLOCAL_INIT(u64, rb, 0);
  CLIO_YLOCAL_INIT(u64, len, 0);
  CLIO_YLOCAL_INIT(u32, by, 0u);
  CLIO_YLOCAL_INIT(u64, row, 0);
  CLIO_YLOCAL(HeldIxN, hn);
  CLIO_YLOCAL(IPtrN, np);
  CLIO_YLOCAL(U64xN, gstart);
  CLIO_YLOCAL(U64xN, glen);
  CLIO_YLOCAL_INIT(u32, nguards, 0u);
  CLIO_YLOCAL_INIT(u64, off, 0);
  CLIO_YLOCAL_INIT(u64, nb0, 0);
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
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float r2list = rlist * rlist;
  const float halfL = 0.5f * box;
  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  const u64 ch_lo = static_cast<u64>(z0) * cpz;
  const u64 ch_hi = static_cast<u64>(z1) * cpz;
  CLIO_YBEGIN();
  for (ch = ch_lo + block; ch < ch_hi; ch += nblocks) {
    bz = static_cast<u32>(ch / cpz);
    y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1) : (nb - 1);
    span_guards = 0;
    {
      const int lo0 = static_cast<int>(y0) - 1;
      const int hi0 = static_cast<int>(ylast) + 1;
      for (int dz0 = -1; dz0 <= 1; ++dz0) {
        const u32 wz0 = (bz + nb + dz0) % nb;
        u32 rl0[2], rn0[2], nr0 = 0;
        if (lo0 < 0) {
          rl0[nr0] = 0; rn0[nr0] = static_cast<u32>(hi0) + 1u; ++nr0;
          rl0[nr0] = nb - 1u; rn0[nr0] = 1u; ++nr0;
        } else if (hi0 > static_cast<int>(nb) - 1) {
          rl0[nr0] = static_cast<u32>(lo0);
          rn0[nr0] = nb - static_cast<u32>(lo0); ++nr0;
          rl0[nr0] = 0; rn0[nr0] = 1u; ++nr0;
        } else {
          rl0[nr0] = static_cast<u32>(lo0);
          rn0[nr0] = static_cast<u32>(hi0 - lo0 + 1); ++nr0;
        }
        for (u32 t0 = 0; t0 < nr0; ++t0) {
          const u64 rb0 =
              ((static_cast<u64>(wz0) * nb + rl0[t0]) * nb) * cap * kStride;
          span_guards +=
              PagesSpanned(x, rb0, static_cast<u64>(rn0[t0]) * row_elems);
        }
      }
    }
    CLIO_YCALL(AdmitSpansMacro(span_guards, x.Regions(), kAdmitSlackChunks));
    nspans = 0;
    for (dz = -1; dz <= 1; ++dz) {
      {
        wz = (bz + nb + dz) % nb;
        const int lo = static_cast<int>(y0) - 1;
        const int hi = static_cast<int>(ylast) + 1;
        nr = 0;
        if (lo < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo);
          rn[nr] = nb - static_cast<u32>(lo); ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo);
          rn[nr] = static_cast<u32>(hi - lo + 1); ++nr;
        }
      }
      for (t = 0; t < nr; ++t) {
        rb = ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
        len = static_cast<u64>(rn[t]) * row_elems;
        CLIO_YCALL(x.MFetch((wz < z0 || wz >= z1) ? hgen : 0, rb, len));
        CLIO_YCALL(x.MHoldPage(&hg[nspans][0], rb, len));
        srun[nspans] = hg[nspans][0].run();
        if (srun[nspans] < len) {
          CLIO_YCALL(x.MHoldPage(&hg[nspans][1], rb + srun[nspans],
                                 len - srun[nspans]));
        }
        MarkPages(MdG().xmask, x.PageOf(rb), x.PageOf(rb + len - 1), block);
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
    for (by = y0; by <= ylast; ++by) {
      row = static_cast<u64>(bz) * nb + by;
      nguards = 0;
      nb0 = row * rowlist;
      off = 0;
      while (off < rowlist && nguards < (u32)kMaxNlGuards) {
        CLIO_YCALL(nl.MFetch(0, nl.PageLo(nb0 + off),
                             nl.PageSpan(nb0 + off, 1)));
        CLIO_YCALL(nl.MHoldPage(&hn[nguards], nb0 + off, rowlist - off,
                                /*write=*/true));
        MarkPages(MdG().nlmask, nl.PageOf(nb0 + off),
                  nl.PageOf(nb0 + off + hn[nguards].run() - 1), block);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
      if (threadIdx.x == 0) {
        for (u32 tt = 0; tt < nspans; ++tt) {
          s_sp0[tt] = sp0[tt];
          s_sp1[tt] = sp1[tt];
          s_srun[tt] = srun[tt];
        }
        for (int dz2 = -1; dz2 <= 1; ++dz2) {
          for (int dy = -1; dy <= 1; ++dy) {
            const u32 wy = (by + nb + dy) % nb;
            const int q = (dz2 + 1) * 3 + (dy + 1);
            for (u32 tt = 0; tt < nspans; ++tt) {
              if (sdz[tt] != static_cast<u32>(dz2 + 1)) continue;
              if (wy >= sbase[tt] && wy < sbase[tt] + scnt[tt]) {
                s_qspan[q] = tt;
                s_qoff[q] = static_cast<u64>(wy - sbase[tt]) * row_elems;
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
      {
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
                const_cast<int *>(s_np[gi])[o - s_gs[gi]] = static_cast<int>(
                    (static_cast<u32>(q) << 16) | (jbx * cap + sj));
                ++cnt;
              }
            }
          }
          d_cnt[slotbase + s] = cnt;
        }
      }
      __syncthreads();
      CLIO_YCALL(nl.MBeginFlush(0, row * rowlist, rowlist));
      for (u32 gq = 0; gq < nguards; ++gq) {
        const u64 fo = row * rowlist + gstart[gq];
        nl.UnpinRange(nl.PageLo(fo), nl.PageSpan(fo, 1));
      }
      for (u32 gq = 0; gq < (u32)kMaxNlGuards; ++gq) hn[gq] = {};
    }   // per-row loop
    for (u32 sq = 0; sq < nspans; ++sq) x.UnpinRange(sxrb[sq], sxlen[sq]);
    for (u32 i6 = 0; i6 < 6; ++i6) { hg[i6][0] = {}; hg[i6][1] = {}; }
    ReleaseSpans(span_guards);
    CLIO_YCALL(nl.MEndFlush());
  }     // per-chunk loop
  CLIO_YEND();
}

/** Macro-form ListForceCoro: the force pass streaming the Verlet list. */
CTP_GPU_FUN inline void ListForceMacro(VecF x, VecF f, VecI nl, u32 nb,
                                       u32 cap, float box, float cutoff,
                                       u32 maxneigh, const u32 *d_cnt,
                                       int eflag, double *acc, int nocompute,
                                       u32 rowchunk, u32 z0, u32 z1,
                                       u32 nblocks, u32 block, u64 hgen,
                                       bool force_all, u32 band) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, ch, 0);
  CLIO_YLOCAL_INIT(long long, r0, 0);
  CLIO_YLOCAL_INIT(long long, f0c, 0);
  CLIO_YLOCAL_INIT(long long, l0, 0);
  CLIO_YLOCAL_INIT(long long, p0, 0);
  CLIO_YLOCAL_INIT(u32, bz, 0u);
  CLIO_YLOCAL_INIT(u32, y0, 0u);
  CLIO_YLOCAL_INIT(u32, ylast, 0u);
  CLIO_YLOCAL_INIT(u32, span_guards, 0u);
  CLIO_YLOCAL(HeldF6x2, hg);
  CLIO_YLOCAL(U64x6, srun);
  CLIO_YLOCAL(CFPtr6, sp0);
  CLIO_YLOCAL(CFPtr6, sp1);
  CLIO_YLOCAL(U32x6, sbase);
  CLIO_YLOCAL(U32x6, scnt);
  CLIO_YLOCAL(U32x6, sdz);
  CLIO_YLOCAL(U64x6, sxrb);
  CLIO_YLOCAL(U64x6, sxlen);
  CLIO_YLOCAL_INIT(u32, nspans, 0u);
  CLIO_YLOCAL_INIT(int, dz, 0);
  CLIO_YLOCAL_INIT(u32, t, 0u);
  CLIO_YLOCAL_INIT(u32, nr, 0u);
  CLIO_YLOCAL(U32x2, rl);
  CLIO_YLOCAL(U32x2, rn);
  CLIO_YLOCAL_INIT(u32, wz, 0u);
  CLIO_YLOCAL_INIT(u64, rb, 0);
  CLIO_YLOCAL_INIT(u64, len, 0);
  CLIO_YLOCAL_INIT(u32, by, 0u);
  CLIO_YLOCAL_INIT(u64, row, 0);
  CLIO_YLOCAL_INIT(u64, fbase, 0);
  CLIO_YLOCAL_INIT(u64, frun0, 0);
  CLIO_YLOCAL(HeldF, hf0);
  CLIO_YLOCAL(HeldF, hf1);
  CLIO_YLOCAL(HeldIxN, hn);
  CLIO_YLOCAL(CIPtrN, np);
  CLIO_YLOCAL(U64xN, gstart);
  CLIO_YLOCAL(U64xN, glen);
  CLIO_YLOCAL_INIT(u32, nguards, 0u);
  CLIO_YLOCAL_INIT(u64, off, 0);
  CLIO_YLOCAL_INIT(u64, nb0, 0);
  CLIO_YLOCAL_INIT(double, pe, 0.0);
  CLIO_YLOCAL_INIT(double, w, 0.0);
  CLIO_YLOCAL_INIT(double, npairs, 0.0);
  MD_RED_SCRATCH(red);
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
  const u64 islots = static_cast<u64>(nb) * cap;
  const u64 rowlist = islots * maxneigh;
  const float c2 = cutoff * cutoff;
  const float halfL = 0.5f * box;
  const u32 cpz = (nb + rowchunk - 1) / rowchunk;
  const u64 ch_lo = static_cast<u64>(z0) * cpz;
  const u64 ch_hi = static_cast<u64>(z1) * cpz;
  CLIO_YBEGIN();
  for (ch = ch_lo + block; ch < ch_hi; ch += nblocks) {
    r0 = clock64();
    bz = static_cast<u32>(ch / cpz);
    {
      const bool on_boundary = (bz == z0 || bz + 1u == z1);
      if (band == 1u && on_boundary) continue;
      if (band == 2u && !on_boundary) continue;
    }
    y0 = static_cast<u32>(ch % cpz) * rowchunk;
    if (y0 >= nb) continue;
    ylast = (y0 + rowchunk - 1 < nb) ? (y0 + rowchunk - 1) : (nb - 1);
    span_guards = 0;
    {
      const int lo0 = static_cast<int>(y0) - 1;
      const int hi0 = static_cast<int>(ylast) + 1;
      for (int dz0 = -1; dz0 <= 1; ++dz0) {
        const u32 wz0 = (bz + nb + dz0) % nb;
        u32 rl0[2], rn0[2], nr0 = 0;
        if (lo0 < 0) {
          rl0[nr0] = 0; rn0[nr0] = static_cast<u32>(hi0) + 1u; ++nr0;
          rl0[nr0] = nb - 1u; rn0[nr0] = 1u; ++nr0;
        } else if (hi0 > static_cast<int>(nb) - 1) {
          rl0[nr0] = static_cast<u32>(lo0);
          rn0[nr0] = nb - static_cast<u32>(lo0); ++nr0;
          rl0[nr0] = 0; rn0[nr0] = 1u; ++nr0;
        } else {
          rl0[nr0] = static_cast<u32>(lo0);
          rn0[nr0] = static_cast<u32>(hi0 - lo0 + 1); ++nr0;
        }
        for (u32 t0 = 0; t0 < nr0; ++t0) {
          const u64 rb0 =
              ((static_cast<u64>(wz0) * nb + rl0[t0]) * nb) * cap * kStride;
          span_guards +=
              PagesSpanned(x, rb0, static_cast<u64>(rn0[t0]) * row_elems);
        }
      }
    }
    CLIO_YCALL(AdmitSpansMacro(span_guards, x.Regions(), kAdmitSlackChunks));
    nspans = 0;
    for (dz = -1; dz <= 1; ++dz) {
      {
        wz = (bz + nb + dz) % nb;
        const int lo = static_cast<int>(y0) - 1;
        const int hi = static_cast<int>(ylast) + 1;
        nr = 0;
        if (lo < 0) {
          rl[nr] = 0; rn[nr] = static_cast<u32>(hi) + 1u; ++nr;
          rl[nr] = nb - 1u; rn[nr] = 1u; ++nr;
        } else if (hi > static_cast<int>(nb) - 1) {
          rl[nr] = static_cast<u32>(lo);
          rn[nr] = nb - static_cast<u32>(lo); ++nr;
          rl[nr] = 0; rn[nr] = 1u; ++nr;
        } else {
          rl[nr] = static_cast<u32>(lo);
          rn[nr] = static_cast<u32>(hi - lo + 1); ++nr;
        }
      }
      for (t = 0; t < nr; ++t) {
        rb = ((static_cast<u64>(wz) * nb + rl[t]) * nb) * cap * kStride;
        len = static_cast<u64>(rn[t]) * row_elems;
        CLIO_YCALL(x.MFetch((force_all || wz < z0 || wz >= z1) ? hgen : 0,
                            rb, len));
        CLIO_YCALL(x.MHoldPage(&hg[nspans][0], rb, len));
        srun[nspans] = hg[nspans][0].run();
        if (srun[nspans] < len) {
          CLIO_YCALL(x.MHoldPage(&hg[nspans][1], rb + srun[nspans],
                                 len - srun[nspans]));
        }
        MarkPages(MdG().xmask, x.PageOf(rb), x.PageOf(rb + len - 1), block);
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
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().md_cyc[0], (unsigned long long)(clock64() - r0));
    }
    for (by = y0; by <= ylast; ++by) {
      row = static_cast<u64>(bz) * nb + by;
      f0c = clock64();
      fbase = row * row_elems;
      CLIO_YCALL(f.MFetch(0, fbase, row_elems));
      CLIO_YCALL(f.MHoldPage(&hf0, fbase, row_elems, true));
      frun0 = hf0.run();
      if (frun0 < row_elems) {
        CLIO_YCALL(f.MHoldPage(&hf1, fbase + frun0, row_elems - frun0, true));
      }
      {
        float *const fp0 = hf0.ptr();
        float *const fp1 = hf1 ? hf1.ptr() : nullptr;
        for (u64 e = threadIdx.x; e < row_elems; e += blockDim.x) {
          (e < frun0 ? fp0[e] : fp1[e - frun0]) = 0.0f;
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        atomicAdd(&MdG().md_cyc[1], (unsigned long long)(clock64() - f0c));
      }
      l0 = clock64();
      nguards = 0;
      nb0 = row * rowlist;
      off = 0;
      while (off < rowlist && nguards < (u32)kMaxNlGuards) {
        CLIO_YCALL(nl.MFetch(0, nl.PageLo(nb0 + off),
                             nl.PageSpan(nb0 + off, 1)));
        CLIO_YCALL(nl.MHoldPage(&hn[nguards], nb0 + off, rowlist - off));
        MarkPages(MdG().nlmask, nl.PageOf(nb0 + off),
                  nl.PageOf(nb0 + off + hn[nguards].run() - 1), block);
        np[nguards] = hn[nguards].ptr();
        gstart[nguards] = off;
        glen[nguards] = hn[nguards].run();
        off += hn[nguards].run();
        ++nguards;
      }
      if (threadIdx.x == 0) {
        for (u32 tt = 0; tt < nspans; ++tt) {
          s_sp0[tt] = sp0[tt];
          s_sp1[tt] = sp1[tt];
          s_srun[tt] = srun[tt];
        }
        for (int dz2 = -1; dz2 <= 1; ++dz2) {
          for (int dy = -1; dy <= 1; ++dy) {
            const u32 wy = (by + nb + dy) % nb;
            const int q = (dz2 + 1) * 3 + (dy + 1);
            for (u32 tt = 0; tt < nspans; ++tt) {
              if (sdz[tt] != static_cast<u32>(dz2 + 1)) continue;
              if (wy >= sbase[tt] && wy < sbase[tt] + scnt[tt]) {
                s_qspan[q] = tt;
                s_qoff[q] = static_cast<u64>(wy - sbase[tt]) * row_elems;
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
      if (threadIdx.x == 0) {
        atomicAdd(&MdG().md_cyc[2], (unsigned long long)(clock64() - l0));
      }
      p0 = clock64();
      {
        float *const fp0 = hf0.ptr();
        float *const fp1 = hf1 ? hf1.ptr() : nullptr;
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
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        atomicAdd(&MdG().md_cyc[3], (unsigned long long)(clock64() - p0));
        atomicAdd(&MdG().md_cyc[5], 1ull);
      }
      for (u32 gq = 0; gq < nguards; ++gq) {
        const u64 fo = row * rowlist + gstart[gq];
        nl.UnpinRange(nl.PageLo(fo), nl.PageSpan(fo, 1));
      }
      f.UnpinRange(fbase, row_elems);
      hf0 = {}; hf1 = {};
      for (u32 gq = 0; gq < (u32)kMaxNlGuards; ++gq) hn[gq] = {};
    }   // per-row loop
    if (threadIdx.x == 0) {
      atomicAdd(&MdG().md_cyc[4], (unsigned long long)(clock64() - r0));
    }
    for (u32 sq = 0; sq < nspans; ++sq) x.UnpinRange(sxrb[sq], sxlen[sq]);
    for (u32 i6 = 0; i6 < 6; ++i6) { hg[i6][0] = {}; hg[i6][1] = {}; }
    ReleaseSpans(span_guards);
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
  CLIO_YEND();
}

#endif  // CLIO_GV_BENCH_MD_MACROS_KERNELS_H_
