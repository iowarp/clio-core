/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * LBANN launches, SYCL. Compiled with -fsycl; the ONLY file in this benchmark
 * that is. Nothing here is workload -- every body lives in
 * ../lbann_kernels.h, shared verbatim with the CUDA side.
 *
 * This TU also owns the yield device globals (CLIO_SYCL_KERNEL_TU) and
 * SyclYieldStackReset: exactly one -fsycl TU per linked program may.
 */

#define CLIO_SYCL_KERNEL_TU 1

#include "../lbann_launch.h"

#include "../lbann_kernels.h"

#include <sycl/sycl.hpp>

namespace clio::gv_bench::lbann {

namespace {

/** The yieldable prologue -- the same statements as the CUDA LB_KERNEL
 *  macro, inside a lambda instead of a __global__ function. */
template <typename MakeCoro>
void SubmitYieldable(dim3 grid, dim3 block, DevF32 w, View vw, StackView sv,
                     MakeCoro make) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t global = static_cast<size_t>(grid.x) * block.x;
  q.parallel_for(
       sycl::nd_range<1>{sycl::range<1>(global), sycl::range<1>(block.x)},
       [=](sycl::nd_item<1>) {
         DevF32 dev = w;
         dev.Init(vw.Block());
         ::clio::run::gpu::YieldTlsPublish(sv, vw.Y(), vw.Block());
         __syncthreads();
         CLIO_YCORO_RUN(make(dev, vw.Block()));
       })
      .wait();
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  ::clio::run::gpu::SyclInitBlockIpcManagers(max_blocks, info);
}

void LaunchFwd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, float *a1, u64 hper, u64 rpp, u64 rbase, u64 rend,
                 View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, w, vw, sv,
                  [=](DevF32 dev, u32 blk_) {
                    const u64 blk = static_cast<u64>(blk_);
                    const bool bias0 = (blk == 0);
                    (void)bias0;
                    return Fwd1Coro(dev, w1_off, b1_off, I, H, B, x, a1, rbase + blk * hper,
                          ((rbase + (blk + 1) * hper) < rend) ? (rbase + (blk + 1) * hper) : rend, rpp);
                  });
}

void LaunchFwd2(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *y, float *d2, double *loss_parts, u64 oper, u64 rpp, u64 rbase, u64 rend,
                 View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, w, vw, sv,
                  [=](DevF32 dev, u32 blk_) {
                    const u64 blk = static_cast<u64>(blk_);
                    const bool bias0 = (blk == 0);
                    (void)bias0;
                    return Fwd2Coro(dev, w2_off, b2_off, H, O, B, a1, y, d2, loss_parts, rbase + blk * oper,
                          ((rbase + (blk + 1) * oper) < rend) ? (rbase + (blk + 1) * oper) : rend, rpp);
                  });
}

void LaunchBwd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float *d1, u64 hper, u64 rpp, u64 rbase, u64 rend, u64 o0, u64 o1, u64 gen,
                 View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, w, vw, sv,
                  [=](DevF32 dev, u32 blk_) {
                    const u64 blk = static_cast<u64>(blk_);
                    const bool bias0 = (blk == 0);
                    (void)bias0;
                    return Bwd1Coro(dev, w2_off, H, O, B, a1, d2, d1, rbase + blk * hper,
                          ((rbase + (blk + 1) * hper) < rend) ? (rbase + (blk + 1) * hper) : rend, rpp, o0, o1, gen);
                  });
}

void LaunchUpd2(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float lr, u64 oper, u64 rpp, u64 rbase, u64 rend, u64 gen,
                 View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, w, vw, sv,
                  [=](DevF32 dev, u32 blk_) {
                    const u64 blk = static_cast<u64>(blk_);
                    const bool bias0 = (blk == 0);
                    (void)bias0;
                    return Upd2Coro(dev, w2_off, b2_off, H, O, B, a1, d2, lr, rbase + blk * oper,
                          ((rbase + (blk + 1) * oper) < rend) ? (rbase + (blk + 1) * oper) : rend, rpp, gen, bias0);
                  });
}

void LaunchUpd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, const float *d1, float lr, u64 hper, u64 rpp, u64 rbase, u64 rend,
                 View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, w, vw, sv,
                  [=](DevF32 dev, u32 blk_) {
                    const u64 blk = static_cast<u64>(blk_);
                    const bool bias0 = (blk == 0);
                    (void)bias0;
                    return Upd1Coro(dev, w1_off, b1_off, I, H, B, x, d1, lr, rbase + blk * hper,
                          ((rbase + (blk + 1) * hper) < rend) ? (rbase + (blk + 1) * hper) : rend, rpp, bias0);
                  });
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk,
                 View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, w, vw, sv,
                  [=](DevF32 dev, u32 blk_) {
                    const u64 blk = static_cast<u64>(blk_);
                    const bool bias0 = (blk == 0);
                    (void)bias0;
                    return SeedCoro(dev, n, blk * eper,
                          ((blk + 1) * eper < n) ? (blk + 1) * eper : n, chunk);
                  });
}

void LaunchMaxDiff(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk, const float *ref,
                 unsigned long long *out, View vw, StackView sv) {
  (void)info;
  SubmitYieldable(grid, block, w, vw, sv,
                  [=](DevF32 dev, u32 blk_) {
                    const u64 blk = static_cast<u64>(blk_);
                    const bool bias0 = (blk == 0);
                    (void)bias0;
                    return MaxDiffCoro(dev, blk * eper,
                          ((blk + 1) * eper < n) ? (blk + 1) * eper : n, chunk, ref, out);
                  });
}

void LaunchDigest(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk, unsigned long long *out,
                 View vw, StackView sv) {
  (void)info;   // stamped once by InitBackend, not per launch
  SubmitYieldable(grid, block, w, vw, sv,
                  [=](DevF32 dev, u32 blk_) {
                    const u64 blk = static_cast<u64>(blk_);
                    const bool bias0 = (blk == 0);
                    (void)bias0;
                    return DigestCoro(dev, blk * eper,
                          ((blk + 1) * eper < n) ? (blk + 1) * eper : n, chunk, out);
                  });
}


namespace {
/** Plain (non-yieldable) submission, in CUDA's (grid, block) shape. */
template <typename BodyT>
void SubmitPlain(u32 blocks, u32 threads, BodyT body) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t global = static_cast<size_t>(blocks) * threads;
  q.parallel_for(
       sycl::nd_range<1>{sycl::range<1>(global), sycl::range<1>(threads)},
       [=](sycl::nd_item<1>) { body(); })
      .wait();
}
}  // namespace

void LaunchDenseSeed(float *w, u64 n) {
  SubmitPlain(64, 256, [=]() { DenseSeed(w, n); });
}

void LaunchDenseFwd1(u32 blocks, u32 threads, const float *w, u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, float *a1, u64 hper) {
  SubmitPlain(blocks, threads, [=]() { DenseFwd1(w, w1_off, b1_off, I, H, B, x, a1, hper); });
}

void LaunchDenseFwd2(u32 blocks, u32 threads, const float *w, u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *y, float *d2, double *lp, u64 oper) {
  SubmitPlain(blocks, threads, [=]() { DenseFwd2(w, w2_off, b2_off, H, O, B, a1, y, d2, lp, oper); });
}

void LaunchDenseBwd1(u32 blocks, u32 threads, const float *w, u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float *d1, u64 hper, u64 rpp) {
  SubmitPlain(blocks, threads, [=]() { DenseBwd1(w, w2_off, H, O, B, a1, d2, d1, hper, rpp); });
}

void LaunchDenseUpd2(u32 blocks, u32 threads, float *w, u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float lr, u64 oper) {
  SubmitPlain(blocks, threads, [=]() { DenseUpd2(w, w2_off, b2_off, H, O, B, a1, d2, lr, oper); });
}

void LaunchDenseUpd1(u32 blocks, u32 threads, float *w, u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, const float *d1, float lr, u64 hper) {
  SubmitPlain(blocks, threads, [=]() { DenseUpd1(w, w1_off, b1_off, I, H, B, x, d1, lr, hper); });
}

void LaunchDenseDigest(const float *w, u64 n, unsigned long long *out) {
  SubmitPlain(64, 256, [=]() { DenseDigest(w, n, out); });
}

}  // namespace clio::gv_bench::lbann

namespace clio::run::gpu {

/** The out-of-line half of YieldStack::Reset; see yield_stack.h. */
void SyclYieldStackReset(const YieldStackView &view, clio::run::u32 nlanes,
                         char *smem_base) {
  auto &q = ctp::GpuApi::SyclQueue();
  YieldStackView v = view;
  q.parallel_for(sycl::range<1>(nlanes), [=](sycl::id<1> i) {
     auto *h = reinterpret_cast<YieldLaneHeader *>(
         v.base_ + static_cast<clio::run::u64>(i[0]) * v.bytes_per_lane_);
     h->sp_ = sizeof(YieldLaneHeader);   // the header is not frame space
     h->live_depth_ = 0;
     h->cur_depth_ = 0;
     h->error_ = kYieldErrNone;
     h->coro_resume_ = 0;
     h->coro_top_ = 0;
     h->coro_park_ = 0;
   }).wait();
  char *base = smem_base;
  q.copy(&base, g_yield_smem_dg, 1).wait();
}

}  // namespace clio::run::gpu
