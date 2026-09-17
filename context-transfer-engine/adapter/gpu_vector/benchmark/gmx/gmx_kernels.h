/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The PME spread/gather device code -- ONE copy, compiled by CUDA and SYCL.
 *
 * Same split as kmeans and grayscott: the workload is here, the launches are
 * in cuda/ and sycl/. Nothing in this file is backend-specific -- no warp
 * intrinsics, no __shared__, no cooperative groups -- so the device-side
 * spellings that remain are supplied for SYCL by
 * clio_ctp/util/sycl_cuda_compat.h.
 *
 * CTP_GPU_FUN, not plain inline: under CUDA these must be __device__ or the
 * host pass compiles them as host functions and every co_await on a
 * device-only verb fails to resolve.
 */
#ifndef CLIO_GV_BENCH_GMX_KERNELS_H_
#define CLIO_GV_BENCH_GMX_KERNELS_H_

#include "gmx_math.h"

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::gmx {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;

using ::clio_gmx::Frac01;
using ::clio_gmx::FxRound;
using ::clio_gmx::kFxScale;
using ::clio_gmx::Lcg;
using ::clio_gmx::Spline4;


/**
 * Spread this block's planes. For plane z the contributing atoms are those
 * whose spline base bin b satisfies b <= z <= b+3, i.e. bins z-3..z (mod K).
 * The plane is fetched, written by THIS BLOCK ONLY, published, released.
 * Partial sums are durable at every unpin: if the plane is evicted between
 * two owner visits (it is not -- one visit per plane -- but the rule is what
 * matters), the refault reads the published partials back and accumulation
 * continues, which is the write-site-publish contract the md workload
 * bled for.
 */
CTP_GPU_FUN inline gy::YCoroMain SpreadCoro(gv::DeviceVector<unsigned long long> mesh,
                                    const float *ax, const float *ay,
                                    const float *az, const long long *aq,
                                    const u32 *bin_start, u64 K, u64 plane,
                                    u64 z0, u64 z1) {
  for (u64 z = z0; z < z1; ++z) {
    co_await mesh.Fetch(0, z * plane, plane);
    auto h = co_await mesh.HoldPage(z * plane, plane, /*write=*/true);
    // Four source bins feed plane z; threads stride the atoms of each bin.
    for (int db = -3; db <= 0; ++db) {
      const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
      const u32 a0 = bin_start[b];
      const u32 a1 = bin_start[b + 1];
      for (u32 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
        const float x = ax[a], y = ay[a], zz = az[a];
        const int ix0 = static_cast<int>(floorf(x)) - 1;
        const int iy0 = static_cast<int>(floorf(y)) - 1;
        // Which of the atom's four z-nodes is THIS plane? The bin choice
        // already guarantees (z - b) mod K lands in 0..3.
        const int dzw = static_cast<int>((z + K - b) % K);
        float wx[4], wy[4], wz[4];
        Spline4(x - floorf(x), wx);
        Spline4(y - floorf(y), wy);
        Spline4(zz - floorf(zz), wz);
        // EXACTLY-CONSERVING SPLIT. Rounding each of the 64 pieces
        // independently leaves a per-atom residue (measured: 809 fixed-point
        // units over 200k atoms), so the LAST piece at each level absorbs
        // the remainder: the four z-pieces sum to q exactly, and the 16
        // xy-pieces of each z-piece sum to it exactly. Deterministic, so
        // the dense reference computes the identical values.
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
    // PUBLISH AT THE WRITE SITE, then release. One writer per page makes
    // this ordering sound; eviction after the unpin costs a refault, never
    // data.
    co_await mesh.BeginFlush(0, z * plane, plane);
    mesh.UnpinRange(z * plane, plane);
  }
  co_await mesh.EndFlush();
}


/** Mesh checksum + exact charge total, striding planes across blocks. */
CTP_GPU_FUN inline gy::YCoroMain SumCoro(gv::DeviceVector<unsigned long long> mesh,
                                 u64 K, u64 plane, u64 z0, u64 z1,
                                 unsigned long long *out) {
  for (u64 z = z0; z < z1; ++z) {
    co_await mesh.Fetch(0, z * plane, plane);
    auto h = co_await mesh.HoldPage(z * plane, plane);
    unsigned long long q = 0, ck = 0;
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) {
      const unsigned long long v = h[z * plane + i];
      q += v;
      // Position-dependent mixing so a value landing on the WRONG grid
      // point cannot cancel: checksum(v at i) != checksum(v at j).
      ck += v * (2ull * (z * plane + i) + 1ull);
    }
    atomicAdd(&out[0], q);
    atomicAdd(&out[1], ck);
    __syncthreads();
    mesh.UnpinRange(z * plane, plane);
  }
}


/**
 * The force-stage read pattern: interpolate the mesh back at every atom
 * position (the potential/force gather of PME, minus the convolution the
 * FFT would have applied). A block owns atom bins; an atom's stencil spans
 * planes iz0..iz0+3, held as a sliding window exactly like the md force
 * stencil. Accumulation is fixed point again, so the result is bit-equal to
 * the dense path.
 */
CTP_GPU_FUN inline gy::YCoroMain GatherCoro(gv::DeviceVector<unsigned long long> mesh,
                                    const float *ax, const float *ay,
                                    const float *az, const long long *aq,
                                    const u32 *bin_start, u64 K, u64 plane,
                                    u64 b0, u64 b1, unsigned long long *out) {
  gv::Held<unsigned long long> hz[4];
  unsigned long long acc = 0;
  for (u64 b = b0; b < b1; ++b) {
    // Atoms in bin b have iz0 == b: hold planes b..b+3 (mod K).
    for (int j = 0; j < 4; ++j) {
      const u64 z = (b + static_cast<u64>(j)) % K;
      co_await mesh.Fetch(0, z * plane, plane);
      hz[j] = co_await mesh.HoldPage(z * plane, plane);
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
      // q_a * phi(x_a), requantized: exact and order-independent.
      acc += static_cast<unsigned long long>(static_cast<long long>(
          FxRound(phi * (static_cast<double>(aq[a]) / kFxScale))));
    }
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
      const u64 z = (b + static_cast<u64>(j)) % K;
      hz[j] = {};
      mesh.UnpinRange(z * plane, plane);
    }
  }
  atomicAdd(out, acc);
}


/** Zero this block's planes and publish, so a fault after eviction reads
 *  zeros rather than "blob not found". */
CTP_GPU_FUN inline gy::YCoroMain ZeroCoro(gv::DeviceVector<unsigned long long> mesh,
                                  u64 plane, u64 z0, u64 z1) {
  for (u64 z = z0; z < z1; ++z) {
    co_await mesh.Fetch(0, z * plane, plane);
    auto h = co_await mesh.HoldPage(z * plane, plane, /*write=*/true);
    for (u64 i = threadIdx.x; i < plane; i += blockDim.x) h[z * plane + i] = 0;
    __syncthreads();
    co_await mesh.BeginFlush(0, z * plane, plane);
    mesh.UnpinRange(z * plane, plane);
  }
  co_await mesh.EndFlush();
}


// ---------------- DENSE REFERENCE (plain cudaMalloc, no paging) -----------


CTP_GPU_FUN inline void DenseSpreadBody(unsigned long long *mesh, const float *ax,
                                  const float *ay, const float *az,
                                  const long long *aq, const u32 *bin_start,
                                  u64 K, u64 plane, u64 zper) {
  const u64 z0 = static_cast<u64>(blockIdx.x) * zper;
  const u64 z1 = (z0 + zper < K) ? (z0 + zper) : K;
  for (u64 z = z0; z < z1; ++z) {
    for (int db = -3; db <= 0; ++db) {
      const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
      const u32 a0 = bin_start[b];
      const u32 a1 = bin_start[b + 1];
      for (u32 a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
        const float x = ax[a], y = ay[a], zz = az[a];
        const int ix0 = static_cast<int>(floorf(x)) - 1;
        const int iy0 = static_cast<int>(floorf(y)) - 1;
        // Which of the atom's four z-nodes is THIS plane? The bin choice
        // already guarantees (z - b) mod K lands in 0..3.
        const int dzw = static_cast<int>((z + K - b) % K);
        float wx[4], wy[4], wz[4];
        Spline4(x - floorf(x), wx);
        Spline4(y - floorf(y), wy);
        Spline4(zz - floorf(zz), wz);
        // EXACTLY-CONSERVING SPLIT. Rounding each of the 64 pieces
        // independently leaves a per-atom residue (measured: 809 fixed-point
        // units over 200k atoms), so the LAST piece at each level absorbs
        // the remainder: the four z-pieces sum to q exactly, and the 16
        // xy-pieces of each z-piece sum to it exactly. Deterministic, so
        // the dense reference computes the identical values.
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
            atomicAdd(&mesh[z * plane + gy_ * K + gx],
                      static_cast<unsigned long long>(v));
          }
        }
      }
      __syncthreads();
    }
  }
}

CTP_GPU_FUN inline void DenseSumBody(const unsigned long long *mesh, u64 n,
                               unsigned long long *out) {
  unsigned long long q = 0, ck = 0;
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    q += mesh[i];
    ck += mesh[i] * (2ull * i + 1ull);
  }
  atomicAdd(&out[0], q);
  atomicAdd(&out[1], ck);
}

CTP_GPU_FUN inline void DenseGatherBody(const unsigned long long *mesh,
                                  const float *ax, const float *ay,
                                  const float *az, const long long *aq,
                                  const u32 *bin_start, u64 K, u64 plane,
                                  u64 bper, unsigned long long *out) {
  const u64 b0 = static_cast<u64>(blockIdx.x) * bper;
  const u64 b1 = (b0 + bper < K) ? (b0 + bper) : K;
  unsigned long long acc = 0;
  for (u64 b = b0; b < b1; ++b) {
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
            const long long v =
                static_cast<long long>(mesh[z * plane + gy_ * K + gx]);
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
  }
  atomicAdd(out, acc);
}

}  // namespace clio::gv_bench::gmx

#endif  // CLIO_GV_BENCH_GMX_KERNELS_H_
