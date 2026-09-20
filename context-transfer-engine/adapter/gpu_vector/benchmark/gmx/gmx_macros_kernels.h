/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * gmx paged kernels, MACRO (Duff's device) form.
 *
 * Same workload as gmx_kernels.h with suspension expressed through
 * yield_stack.h's macros rather than C++20 device coroutines, which clang
 * cannot compile for SPIR-V. See kmeans_macros_kernels.h for the rationale.
 *
 * THE ONE RULE, and where it bites hardest here: GatherMacro holds FOUR pages
 * at once and fetches them in an inner loop, so both the loop index `j` and
 * the `hz[4]` array span suspensions and must live in the frame. `acc` does
 * too -- it accumulates across every bin. Everything born and consumed between
 * two yields (the spline weights, `phi`, `row`) stays an ordinary register.
 */
#ifndef CLIO_GV_BENCH_GMX_MACROS_KERNELS_H_
#define CLIO_GV_BENCH_GMX_MACROS_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

#include "gmx_kernels.h"

namespace clio::gv_bench::gmx_macros {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;
// The plain helpers beside the coroutines -- Spline4, FxRound, kFxScale and
// the dense bodies -- are ordinary device code and reused as-is.
using namespace ::clio::gv_bench::gmx;

using MeshV = gv::DeviceVector<unsigned long long>;
using HeldU64 = gv::Held<unsigned long long>;
/** CLIO_YLOCAL takes one macro argument, so the array type needs a name. */
using HeldU64x4 = HeldU64[4];

/** Macro-form SpreadCoro. */
CTP_GPU_FUN inline void SpreadMacro(MeshV mesh, const float *ax,
                                    const float *ay, const float *az,
                                    const long long *aq, const u32 *bin_start,
                                    u64 K, u64 plane, u64 z0, u64 z1) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, z, z0);
  CLIO_YLOCAL(HeldU64, h);
  CLIO_YBEGIN();
  for (; z < z1; ++z) {
    CLIO_YCALL(mesh.MFetch(0, z * plane, plane));
    CLIO_YCALL(mesh.MHoldPage(&h, z * plane, plane, /*write=*/true));
    for (int db = -3; db <= 0; ++db) {
      const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
      const u32 a0 = bin_start[b];
      const u32 a1 = bin_start[b + 1];
      for (u32 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
        const float x = ax[a], y = ay[a], zz = az[a];
        const int ix0 = static_cast<int>(floorf(x)) - 1;
        const int iy0 = static_cast<int>(floorf(y)) - 1;
        const int dzw = static_cast<int>((z + K - b) % K);
        float wx[4], wy[4], wz[4];
        Spline4(x - floorf(x), wx);
        Spline4(y - floorf(y), wy);
        Spline4(zz - floorf(zz), wz);
        long long qz4[4];
        {
          long long run = 0;
          for (int j = 0; j < 3; ++j) {
            qz4[j] = static_cast<long long>(
                FxRound(static_cast<double>(aq[a]) * wz[j]));
            run += qz4[j];
          }
          qz4[3] = aq[a] - run;
        }
        const long long qz = qz4[dzw];
        long long xy_run = 0;
        for (int jy = 0; jy < 4; ++jy) {
          const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                           static_cast<int>(K));
          for (int jx = 0; jx < 4; ++jx) {
            const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                            static_cast<int>(K));
            const long long v =
                (jy == 3 && jx == 3)
                    ? qz - xy_run
                    : static_cast<long long>(FxRound(
                          static_cast<double>(qz) * wy[jy] * wx[jx]));
            xy_run += v;
            atomicAdd(reinterpret_cast<unsigned long long *>(
                          &h[z * plane + gy_ * K + gx]),
                      static_cast<unsigned long long>(v));
          }
        }
      }
      __syncthreads();
    }
    CLIO_YCALL(mesh.MBeginFlush(0, z * plane, plane));
    mesh.UnpinRange(z * plane, plane);
  }
  CLIO_YCALL(mesh.MEndFlush());
  CLIO_YEND();
}

/** Macro-form SumCoro. */
CTP_GPU_FUN inline void SumMacro(MeshV mesh, u64 K, u64 plane, u64 z0, u64 z1,
                                 unsigned long long *out) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, z, z0);
  CLIO_YLOCAL(HeldU64, h);
  CLIO_YBEGIN();
  for (; z < z1; ++z) {
    CLIO_YCALL(mesh.MFetch(0, z * plane, plane));
    CLIO_YCALL(mesh.MHoldPage(&h, z * plane, plane));
    unsigned long long q = 0, ck = 0;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      const unsigned long long v = h[z * plane + i];
      q += v;
      ck += v * (2ull * (z * plane + i) + 1ull);
    }
    atomicAdd(&out[0], q);
    atomicAdd(&out[1], ck);
    __syncthreads();
    mesh.UnpinRange(z * plane, plane);
  }
  CLIO_YEND();
}

/** Macro-form GatherCoro. */
CTP_GPU_FUN inline void GatherMacro(MeshV mesh, const float *ax,
                                    const float *ay, const float *az,
                                    const long long *aq, const u32 *bin_start,
                                    u64 K, u64 plane, u64 b0, u64 b1,
                                    unsigned long long *out) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, b, b0);
  // Spans the inner fetch loop's suspensions.
  CLIO_YLOCAL_INIT(int, j, 0);
  // Computed before MFetch and read after it, so it spans that suspension --
  // and as an initialized automatic it would also make the case label
  // unreachable ("cannot jump from switch statement to this case label"),
  // which is the compiler catching the same rule from the other side.
  CLIO_YLOCAL_INIT(u64, zj, 0);
  CLIO_YLOCAL(HeldU64x4, hz);
  CLIO_YLOCAL_INIT(unsigned long long, acc, 0ull);
  CLIO_YBEGIN();
  for (; b < b1; ++b) {
    for (j = 0; j < 4; ++j) {
      zj = (b + static_cast<u64>(j)) % K;
      CLIO_YCALL(mesh.MFetch(0, zj * plane, plane));
      CLIO_YCALL(mesh.MHoldPage(&hz[j], zj * plane, plane));
    }
    const u32 a0 = bin_start[b];
    const u32 a1 = bin_start[b + 1];
    for (u32 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
      const float x = ax[a], y = ay[a], zz = az[a];
      const int ix0 = static_cast<int>(floorf(x)) - 1;
      const int iy0 = static_cast<int>(floorf(y)) - 1;
      float wx[4], wy[4], wzS[4];
      Spline4(x - floorf(x), wx);
      Spline4(y - floorf(y), wy);
      Spline4(zz - floorf(zz), wzS);
      double phi = 0.0;
      for (int jz = 0; jz < 4; ++jz) {
        const u64 z = (b + static_cast<u64>(jz)) % K;
        double pl = 0.0;
        for (int jy = 0; jy < 4; ++jy) {
          const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                           static_cast<int>(K));
          double row = 0.0;
          for (int jx = 0; jx < 4; ++jx) {
            const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                            static_cast<int>(K));
            const long long v = static_cast<long long>(
                hz[jz][z * plane + gy_ * K + gx]);
            row += static_cast<double>(v) * wx[jx];
          }
          pl += row * wy[jy];
        }
        phi += pl * wzS[jz];
      }
      acc += static_cast<unsigned long long>(static_cast<long long>(
          FxRound(phi * (static_cast<double>(aq[a]) / kFxScale))));
    }
    __syncthreads();
    for (int jj = 0; jj < 4; ++jj) {
      const u64 z = (b + static_cast<u64>(jj)) % K;
      hz[jj] = {};
      mesh.UnpinRange(z * plane, plane);
    }
  }
  atomicAdd(out, acc);
  CLIO_YEND();
}

/** Macro-form ZeroCoro. */
CTP_GPU_FUN inline void ZeroMacro(MeshV mesh, u64 plane, u64 z0, u64 z1) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, z, z0);
  CLIO_YLOCAL(HeldU64, h);
  CLIO_YBEGIN();
  for (; z < z1; ++z) {
    CLIO_YCALL(mesh.MFetch(0, z * plane, plane));
    CLIO_YCALL(mesh.MHoldPage(&h, z * plane, plane, /*write=*/true));
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) h[z * plane + i] = 0;
    __syncthreads();
    CLIO_YCALL(mesh.MBeginFlush(0, z * plane, plane));
    mesh.UnpinRange(z * plane, plane);
  }
  CLIO_YCALL(mesh.MEndFlush());
  CLIO_YEND();
}

}  // namespace clio::gv_bench::gmx_macros

#endif  // CLIO_GV_BENCH_GMX_MACROS_KERNELS_H_
