/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * grayscott paged kernels, MACRO (Duff's device) form.
 *
 * Same workload as grayscott_kernels.h with suspension expressed through
 * yield_stack.h's macros rather than C++20 device coroutines, which clang
 * cannot compile for SPIR-V. See kmeans_macros_kernels.h for the rationale.
 *
 * THE ONE RULE, at its densest here. StepMacro holds EIGHT pages across a
 * chain of sixteen suspensions, and `interior`/`zm`/`zp`/`gzm`/`gzp` are all
 * computed at the top of the loop body and read after every one of those
 * fetches. As initialised automatics they would also sit between the switch
 * and its case labels, which the compiler rejects outright -- so every one of
 * them is a frame local. Values born and consumed inside the compute loop
 * (`u`, `v`, `lu`, `lv`, `uvv`) stay ordinary registers.
 */
#ifndef CLIO_GV_BENCH_GRAYSCOTT_MACROS_KERNELS_H_
#define CLIO_GV_BENCH_GRAYSCOTT_MACROS_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

#include "grayscott_kernels.h"

namespace clio::gv_bench::grayscott_macros {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;
// Plain helpers beside the coroutines (InitU, InitV, the dense bodies).
using namespace ::clio::gv_bench::grayscott;

using VecF = gv::DeviceVector<float>;
using HeldF = gv::Held<float>;

/** Macro-form SeedCoro. */
CTP_GPU_FUN inline void SeedMacro(VecF vec, u64 plane, u64 nx, u64 ny, u64 nz,
                                  u64 z0, u64 z1, u64 ubase, u64 vbase) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, z, z0);
  CLIO_YLOCAL(HeldF, h);
  CLIO_YBEGIN();
  for (; z < z1; ++z) {
    CLIO_YCALL(vec.MFetch(0, ubase + z * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&h, ubase + z * plane, plane, /*write=*/true));
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      h[ubase + z * plane + i] = InitU(i % nx, i / nx, z, nx, ny, nz);
    }
    CLIO_YCALL(vec.MBeginFlush(1, ubase + z * plane, plane));
    vec.UnpinRange(ubase + z * plane, plane);

    CLIO_YCALL(vec.MFetch(0, vbase + z * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&h, vbase + z * plane, plane, /*write=*/true));
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      h[vbase + z * plane + i] = InitV(i % nx, i / nx, z, nx, ny, nz);
    }
    CLIO_YCALL(vec.MBeginFlush(1, vbase + z * plane, plane));
    vec.UnpinRange(vbase + z * plane, plane);
  }
  CLIO_YCALL(vec.MEndFlush());
  CLIO_YEND();
}

/** Macro-form StepCoro: the seven-point stencil over a paged 3-D field. */
CTP_GPU_FUN inline void StepMacro(VecF vec, u64 plane, u64 nx, u64 ny, u64 nz,
                                  u64 z0, u64 z1, u64 nlo, u64 nhi, u64 gen,
                                  u64 ubase, u64 vbase, u64 unext, u64 vnext,
                                  float Du, float Dv, float F, float K,
                                  float dt) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, z, z0);
  CLIO_YLOCAL_INIT(u64, zm, 0);
  CLIO_YLOCAL_INIT(u64, zp, 0);
  CLIO_YLOCAL_INIT(u64, gzm, 0);
  CLIO_YLOCAL_INIT(u64, gzp, 0);
  CLIO_YLOCAL_INIT(bool, interior, false);
  CLIO_YLOCAL(HeldF, uzm);
  CLIO_YLOCAL(HeldF, uz);
  CLIO_YLOCAL(HeldF, uzp);
  CLIO_YLOCAL(HeldF, vzm);
  CLIO_YLOCAL(HeldF, vz);
  CLIO_YLOCAL(HeldF, vzp);
  CLIO_YLOCAL(HeldF, unx);
  CLIO_YLOCAL(HeldF, vnx);
  CLIO_YBEGIN();
  for (; z < z1; ++z) {
    gy::YieldPublishCursor(z + 1);
    interior = (z > 0 && z + 1 < nz);
    zm = interior ? (z - 1) : z;
    zp = interior ? (z + 1) : z;
    gzm = (zm < nlo || zm >= nhi) ? gen : 0;
    gzp = (zp < nlo || zp >= nhi) ? gen : 0;
    CLIO_YCALL(vec.MFetch(gzm, ubase + zm * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&uzm, ubase + zm * plane, plane));
    CLIO_YCALL(vec.MFetch(0, ubase + z * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&uz, ubase + z * plane, plane));
    CLIO_YCALL(vec.MFetch(gzp, ubase + zp * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&uzp, ubase + zp * plane, plane));
    CLIO_YCALL(vec.MFetch(gzm, vbase + zm * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&vzm, vbase + zm * plane, plane));
    CLIO_YCALL(vec.MFetch(0, vbase + z * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&vz, vbase + z * plane, plane));
    CLIO_YCALL(vec.MFetch(gzp, vbase + zp * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&vzp, vbase + zp * plane, plane));
    CLIO_YCALL(vec.MFetch(0, unext + z * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&unx, unext + z * plane, plane, /*write=*/true));
    CLIO_YCALL(vec.MFetch(0, vnext + z * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&vnx, vnext + z * plane, plane, /*write=*/true));
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      const u64 x = i % nx, y = i / nx;
      const float u = uz[ubase + z * plane + i];
      const float v = vz[vbase + z * plane + i];
      float lu, lv;
      if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || !interior) {
        lu = 0.0f; lv = 0.0f;      // fixed boundary
      } else {
        lu = uz[ubase + z * plane + i - 1] +
             uz[ubase + z * plane + i + 1] +
             uz[ubase + z * plane + i - nx] +
             uz[ubase + z * plane + i + nx] +
             uzm[ubase + zm * plane + i] +
             uzp[ubase + zp * plane + i] - 6.0f * u;
        lv = vz[vbase + z * plane + i - 1] +
             vz[vbase + z * plane + i + 1] +
             vz[vbase + z * plane + i - nx] +
             vz[vbase + z * plane + i + nx] +
             vzm[vbase + zm * plane + i] +
             vzp[vbase + zp * plane + i] - 6.0f * v;
      }
      const float uvv = u * v * v;
      unx[unext + z * plane + i] = u + dt * (Du * lu - uvv + F * (1.0f - u));
      vnx[vnext + z * plane + i] = v + dt * (Dv * lv + uvv - (F + K) * v);
    }
    __syncthreads();
    CLIO_YCALL(vec.MBeginFlush(gen + 1, unext + z * plane, plane));
    CLIO_YCALL(vec.MBeginFlush(gen + 1, vnext + z * plane, plane));
    vec.UnpinRange(ubase + zm * plane, plane);
    vec.UnpinRange(ubase + z * plane, plane);
    vec.UnpinRange(ubase + zp * plane, plane);
    vec.UnpinRange(vbase + zm * plane, plane);
    vec.UnpinRange(vbase + z * plane, plane);
    vec.UnpinRange(vbase + zp * plane, plane);
    vec.UnpinRange(unext + z * plane, plane);
    vec.UnpinRange(vnext + z * plane, plane);
    if (interior) {
      uzm = {};
      vzm = {};
    }
  }
  uzm = {}; uz = {}; uzp = {};
  vzm = {}; vz = {}; vzp = {};
  unx = {}; vnx = {};
  CLIO_YCALL(vec.MEndFlush());
  CLIO_YEND();
}

/** Macro-form SumCoro. */
CTP_GPU_FUN inline void SumMacro(VecF vec, u64 plane, u64 z0, u64 z1,
                                 u64 vbase, double *out) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, z, z0);
  CLIO_YLOCAL(HeldF, h);
  CLIO_YBEGIN();
  for (; z < z1; ++z) {
    CLIO_YCALL(vec.MFetch(0, vbase + z * plane, plane));
    CLIO_YCALL(vec.MHoldPage(&h, vbase + z * plane, plane));
    double acc = 0.0;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      acc += static_cast<double>(h[vbase + z * plane + i]);
    }
    atomicAdd(out, acc);
    __syncthreads();
    vec.UnpinRange(vbase + z * plane, plane);
  }
  CLIO_YEND();
}

}  // namespace clio::gv_bench::grayscott_macros

#endif  // CLIO_GV_BENCH_GRAYSCOTT_MACROS_KERNELS_H_
