/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The seven launches the LBANN MLP needs, backend-neutral.
 *
 * THE SEAM, for the reason documented in kmeans_launch.h: a CUDA kernel is a
 * __global__ function at namespace scope; a SYCL kernel is a LAMBDA INSIDE
 * ITS LAUNCHER, and DPC++ member-checks the whole translation unit. The
 * workload is in lbann_kernels.h, once.
 *
 * Seven of them because a training step is seven phases, each owning a
 * different slice of the weight vector -- that structure is the workload's,
 * not the port's.
 */
#ifndef CLIO_GV_BENCH_LBANN_LAUNCH_H_
#define CLIO_GV_BENCH_LBANN_LAUNCH_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::lbann {

using ::clio::run::u32;
using ::clio::run::u64;
using DevF32 = ::clio::cte::gpu_vector::DeviceVector<float>;
using View = ::clio::run::gpu::YieldableView<>;
using StackView = ::clio::run::gpu::YieldStackView;
using GpuInfo = ::clio::run::IpcManagerGpuInfo;

/** Per-block device state the SYCL backend allocates once; no-op on CUDA. */
void InitBackend(u32 max_blocks, const GpuInfo &info);

void LaunchFwd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w1_off, const float *b1v, u64 I, u64 H, u64 B, const float *x, float *a1, u64 hper, u64 rpp, u64 rbase, u64 rend, u64 gen,
                 View vw, StackView sv, float **tab, u64 tab_stride);

void LaunchFwd2(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, const float *b2v, u64 H, u64 O, u64 B, const float *a1, const float *y, float *d2, double *loss_parts, u64 oper, u64 rpp, u64 rbase, u64 rend, u64 gen,
                 View vw, StackView sv);

void LaunchW2Resolve(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                     u64 w2_off, u64 H, u64 O, u64 rpp, u64 gen, float **tab,
                     View vw, StackView sv, u64 o_lo);
void LaunchW2Release(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                     u64 w2_off, u64 H, u64 O, u64 rpp, View vw,
                     StackView sv, u64 o_lo);
void LaunchBwd1PartialTab(u32 blocks, u32 threads, float *const *tab,
                          u64 rpp, u64 o_lo, u64 o_hi, u64 H, u64 B,
                          const float *d2, float *out);
void LaunchFwd2Tab(u32 blocks, u32 threads, float *const *tab, u64 rpp,
                   u64 o_lo, u64 o_hi, u64 H, u64 O, u64 B, const float *b2v,
                   const float *a1, const float *y, float *d2,
                   double *loss_parts);
void LaunchBwd1Combine(u32 blocks, u32 threads, const float *parts,
                       u64 nodes, u64 HB, const float *a1, float *d1);
void LaunchBwd1Tab(u32 blocks, u32 threads, float *const *tab, u64 np,
                   u64 rpp, u64 O, u64 H, u64 B, u64 h0, u64 h1,
                   const float *a1, const float *d2, float *d1);
void LaunchBwd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float *d1, u64 hper, u64 rpp, u64 rbase, u64 rend, u64 o0, u64 o1, u64 gen,
                 View vw, StackView sv, float **tab, u64 tab_stride);

void LaunchUpd2(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, float *b2v, u64 H, u64 O, u64 B, const float *a1, const float *d2, float lr, u64 oper, u64 rpp, u64 rbase, u64 rend, u64 gen,
                 View vw, StackView sv);

void LaunchUpd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w1_off, float *b1v, u64 I, u64 H, u64 B, const float *x, const float *d1, float lr, u64 hper, u64 rpp, u64 rbase, u64 rend, u64 gen,
                 View vw, StackView sv, float **tab, u64 tab_stride);

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk,
                 View vw, StackView sv, u64 base, u32 publish);

void LaunchDigest(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk, unsigned long long *out,
                 View vw, StackView sv);

/** Max |paged - dense| over the weights, scaled by 1e9. The distributed
 *  gate: a bit-exact digest cannot survive a cross-node summation order. */
void LaunchMaxDiff(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk, const float *ref,
                 unsigned long long *out, u64 rlo, u64 rhi, u64 gen, View vw,
                 StackView sv);


/* ---- dense reference: the whole model resident, no paging ---- */

void LaunchDenseSeed(float *w, u64 n);

void LaunchDenseFwd1(u32 blocks, u32 threads, const float *w, u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, float *a1, u64 hper);

void LaunchDenseFwd2(u32 blocks, u32 threads, const float *w, u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *y, float *d2, double *lp, u64 oper);

void LaunchDenseBwd1(u32 blocks, u32 threads, const float *w, u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float *d1, u64 hper, u64 rpp);

void LaunchDenseUpd2(u32 blocks, u32 threads, float *w, u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float lr, u64 oper);

void LaunchDenseUpd1(u32 blocks, u32 threads, float *w, u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, const float *d1, float lr, u64 hper);

void LaunchDenseDigest(const float *w, u64 n, unsigned long long *out);

}  // namespace clio::gv_bench::lbann

#endif  // CLIO_GV_BENCH_LBANN_LAUNCH_H_
