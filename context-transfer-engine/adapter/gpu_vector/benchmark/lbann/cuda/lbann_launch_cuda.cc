/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * LBANN launches, CUDA. Compiled by clang-CUDA. Nothing here is workload:
 * every body lives in ../lbann_kernels.h.
 */

#include "../lbann_launch.h"
#include "../../gv_launch_bounds.h"

#include "../lbann_kernels.h"

namespace clio::gv_bench::lbann {

namespace {

/** The prologue every yieldable kernel repeats. Same text as the SYCL side's
 *  lambda; only the submission differs. */
#define LB_KERNEL(NAME, CORO_CALL, ...)                                       \
  __global__ GV_LAUNCH_BOUNDS void NAME##Kernel(GpuInfo info, DevF32 w, __VA_ARGS__, View yv,  \
                               StackView ys) {                                \
    CLIO_GPU_INIT(info, nullptr);                                             \
    w.Init(yv.Block());                                                       \
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());                              \
    __syncthreads();                                                          \
    /* One block per node does the replicated bias update: -= is not */       \
    /* idempotent, so every block running it would apply it gridDim times. */ \
    const bool bias0 = (yv.Block() == 0);                                     \
    (void)bias0;                                                              \
    CLIO_YCORO_RUN(CORO_CALL);                                                \
  }

LB_KERNEL(Fwd1,
          Fwd1Coro(w, w1_off, b1_off, I, H, B, x, a1, rbase + static_cast<u64>(yv.Block()) * hper,
                          ((rbase + (static_cast<u64>(yv.Block()) + 1) * hper) < rend) ? (rbase + (static_cast<u64>(yv.Block()) + 1) * hper) : rend, rpp),
          u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, float *a1, u64 hper, u64 rpp, u64 rbase, u64 rend)

LB_KERNEL(Fwd2,
          Fwd2Coro(w, w2_off, b2_off, H, O, B, a1, y, d2, loss_parts, rbase + static_cast<u64>(yv.Block()) * oper,
                          ((rbase + (static_cast<u64>(yv.Block()) + 1) * oper) < rend) ? (rbase + (static_cast<u64>(yv.Block()) + 1) * oper) : rend, rpp),
          u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *y, float *d2, double *loss_parts, u64 oper, u64 rpp, u64 rbase, u64 rend)

LB_KERNEL(Bwd1,
          Bwd1Coro(w, w2_off, H, O, B, a1, d2, d1, rbase + static_cast<u64>(yv.Block()) * hper,
                          ((rbase + (static_cast<u64>(yv.Block()) + 1) * hper) < rend) ? (rbase + (static_cast<u64>(yv.Block()) + 1) * hper) : rend, rpp, o0, o1, gen),
          u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float *d1, u64 hper, u64 rpp, u64 rbase, u64 rend, u64 o0, u64 o1, u64 gen)

LB_KERNEL(Upd2,
          Upd2Coro(w, w2_off, b2_off, H, O, B, a1, d2, lr, rbase + static_cast<u64>(yv.Block()) * oper,
                          ((rbase + (static_cast<u64>(yv.Block()) + 1) * oper) < rend) ? (rbase + (static_cast<u64>(yv.Block()) + 1) * oper) : rend, rpp, gen, bias0),
          u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float lr, u64 oper, u64 rpp, u64 rbase, u64 rend, u64 gen)

LB_KERNEL(Upd1,
          Upd1Coro(w, w1_off, b1_off, I, H, B, x, d1, lr, rbase + static_cast<u64>(yv.Block()) * hper,
                          ((rbase + (static_cast<u64>(yv.Block()) + 1) * hper) < rend) ? (rbase + (static_cast<u64>(yv.Block()) + 1) * hper) : rend, rpp, bias0),
          u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, const float *d1, float lr, u64 hper, u64 rpp, u64 rbase, u64 rend)

LB_KERNEL(Seed,
          SeedCoro(w, n, static_cast<u64>(yv.Block()) * eper,
                          ((static_cast<u64>(yv.Block()) + 1) * eper < n) ? (static_cast<u64>(yv.Block()) + 1) * eper : n, chunk),
          u64 n, u64 eper, u64 chunk)

LB_KERNEL(MaxDiff,
          MaxDiffCoro(w, static_cast<u64>(yv.Block()) * eper,
                          ((static_cast<u64>(yv.Block()) + 1) * eper < n) ? (static_cast<u64>(yv.Block()) + 1) * eper : n, chunk, ref, out, rlo, rhi, gen),
          u64 n, u64 eper, u64 chunk, const float *ref, unsigned long long *out, u64 rlo, u64 rhi, u64 gen)

LB_KERNEL(Digest,
          DigestCoro(w, static_cast<u64>(yv.Block()) * eper,
                          ((static_cast<u64>(yv.Block()) + 1) * eper < n) ? (static_cast<u64>(yv.Block()) + 1) * eper : n, chunk, out),
          u64 n, u64 eper, u64 chunk, unsigned long long *out)


/* __global__ wrappers for the dense reference bodies. The bodies are
 * CTP_GPU_FUN device functions in ../lbann_kernels.h, shared with SYCL;
 * only these launch stubs are CUDA's. */
__global__ GV_LAUNCH_BOUNDS void DenseSeedKernel(float *w, u64 n) {
  DenseSeed(w, n);
}

__global__ GV_LAUNCH_BOUNDS void DenseFwd1Kernel(const float *w, u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, float *a1, u64 hper) {
  DenseFwd1(w, w1_off, b1_off, I, H, B, x, a1, hper);
}

__global__ GV_LAUNCH_BOUNDS void DenseFwd2Kernel(const float *w, u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *y, float *d2, double *lp, u64 oper) {
  DenseFwd2(w, w2_off, b2_off, H, O, B, a1, y, d2, lp, oper);
}

__global__ GV_LAUNCH_BOUNDS void DenseBwd1Kernel(const float *w, u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float *d1, u64 hper, u64 rpp) {
  DenseBwd1(w, w2_off, H, O, B, a1, d2, d1, hper, rpp);
}

__global__ GV_LAUNCH_BOUNDS void DenseUpd2Kernel(float *w, u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float lr, u64 oper) {
  DenseUpd2(w, w2_off, b2_off, H, O, B, a1, d2, lr, oper);
}

__global__ GV_LAUNCH_BOUNDS void DenseUpd1Kernel(float *w, u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, const float *d1, float lr, u64 hper) {
  DenseUpd1(w, w1_off, b1_off, I, H, B, x, d1, lr, hper);
}

__global__ GV_LAUNCH_BOUNDS void DenseDigestKernel(const float *w, u64 n, unsigned long long *out) {
  DenseDigest(w, n, out);
}

}  // namespace

void InitBackend(u32 max_blocks, const GpuInfo &info) {
  // CUDA's per-block IpcManager is __shared__ storage, born fresh at every
  // launch and initialized by CLIO_GPU_INIT.
  (void)max_blocks;
  (void)info;
}

void LaunchFwd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, float *a1, u64 hper, u64 rpp, u64 rbase, u64 rend,
                 View vw, StackView sv) {
  Fwd1Kernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, w, w1_off, b1_off, I, H, B, x, a1, hper, rpp, rbase, rend, vw,
                                                  sv);
}

void LaunchFwd2(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *y, float *d2, double *loss_parts, u64 oper, u64 rpp, u64 rbase, u64 rend,
                 View vw, StackView sv) {
  Fwd2Kernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, w, w2_off, b2_off, H, O, B, a1, y, d2, loss_parts, oper, rpp, rbase, rend, vw,
                                                  sv);
}

void LaunchBwd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float *d1, u64 hper, u64 rpp, u64 rbase, u64 rend, u64 o0, u64 o1, u64 gen,
                 View vw, StackView sv) {
  Bwd1Kernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, w, w2_off, H, O, B, a1, d2, d1, hper, rpp, rbase, rend, o0, o1, gen, vw,
                                                  sv);
}

void LaunchUpd2(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float lr, u64 oper, u64 rpp, u64 rbase, u64 rend, u64 gen,
                 View vw, StackView sv) {
  Upd2Kernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, w, w2_off, b2_off, H, O, B, a1, d2, lr, oper, rpp, rbase, rend, gen, vw,
                                                  sv);
}

void LaunchUpd1(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, const float *d1, float lr, u64 hper, u64 rpp, u64 rbase, u64 rend,
                 View vw, StackView sv) {
  Upd1Kernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, w, w1_off, b1_off, I, H, B, x, d1, lr, hper, rpp, rbase, rend, vw,
                                                  sv);
}

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk,
                 View vw, StackView sv) {
  SeedKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, w, n, eper, chunk, vw,
                                                  sv);
}

void LaunchMaxDiff(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk, const float *ref,
                 unsigned long long *out, u64 rlo, u64 rhi, u64 gen, View vw,
                 StackView sv) {
  MaxDiffKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, w, n, eper, chunk,
                                                        ref, out, rlo, rhi, gen, vw, sv);
}

void LaunchDigest(dim3 grid, dim3 block, const GpuInfo &info, DevF32 w,
                 u64 n, u64 eper, u64 chunk, unsigned long long *out,
                 View vw, StackView sv) {
  DigestKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, w, n, eper, chunk, out, vw,
                                                  sv);
}


void LaunchDenseSeed(float *w, u64 n) {
  DenseSeedKernel<<<64, 256>>>(w, n);
}

void LaunchDenseFwd1(u32 blocks, u32 threads, const float *w, u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, float *a1, u64 hper) {
  DenseFwd1Kernel<<<blocks, threads>>>(w, w1_off, b1_off, I, H, B, x, a1, hper);
}

void LaunchDenseFwd2(u32 blocks, u32 threads, const float *w, u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *y, float *d2, double *lp, u64 oper) {
  DenseFwd2Kernel<<<blocks, threads>>>(w, w2_off, b2_off, H, O, B, a1, y, d2, lp, oper);
}

void LaunchDenseBwd1(u32 blocks, u32 threads, const float *w, u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float *d1, u64 hper, u64 rpp) {
  DenseBwd1Kernel<<<blocks, threads>>>(w, w2_off, H, O, B, a1, d2, d1, hper, rpp);
}

void LaunchDenseUpd2(u32 blocks, u32 threads, float *w, u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2, float lr, u64 oper) {
  DenseUpd2Kernel<<<blocks, threads>>>(w, w2_off, b2_off, H, O, B, a1, d2, lr, oper);
}

void LaunchDenseUpd1(u32 blocks, u32 threads, float *w, u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x, const float *d1, float lr, u64 hper) {
  DenseUpd1Kernel<<<blocks, threads>>>(w, w1_off, b1_off, I, H, B, x, d1, lr, hper);
}

void LaunchDenseDigest(const float *w, u64 n, unsigned long long *out) {
  DenseDigestKernel<<<64, 256>>>(w, n, out);
}

}  // namespace clio::gv_bench::lbann
