/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * LBANN science kernel -- MLP training with the WEIGHTS out of core -- over
 * a GPU vector.
 *
 * WHY THIS SHAPE. LBANN's paged-vector story was always the weights: the
 * model outgrows VRAM while each training step still touches all of it
 * (forward reads W, backward reads W again and then updates it in place).
 * That is the "re-read everything every pass, then WRITE it" pattern -- the
 * weights benchmark covers the read half; this adds the update half, which
 * is what makes eviction dangerous: an unpublished SGD update undone by a
 * refault is exactly the frozen-physics failure the md workload had, in
 * model-training clothes (loss keeps falling, weights quietly revert).
 *
 * STRUCTURE. Two layers, x -> ReLU(W1 x + b1) -> W2 a1 + b2, MSE loss, SGD.
 * One paged vector holds [W1 | b1 | W2 | b2]; activations, gradients and
 * the batch are small and stay resident. Each step runs five phases, each a
 * kernel launch (the inter-layer barrier), all reading or writing the paged
 * weights through Fetch/HoldPage and publishing every update at the write
 * site:
 *
 *   fwd1  reads W1 rows      (block owns h-rows; pages slide)
 *   fwd2  reads W2 rows      (block owns o-rows)
 *   bwd1  reads W2 rows      (PRE-update, classic backprop ordering)
 *   upd2  W2 -= lr dW2       (block owns o-rows: ONE WRITER PER PAGE)
 *   upd1  W1 -= lr dW1       (block owns h-rows: ONE WRITER PER PAGE)
 *
 * DETERMINISM BY CONSTRUCTION. Every output element is computed by exactly
 * one thread with a fixed-order sequential sum -- no atomics anywhere on
 * the training path -- so the paged run must match a dense in-VRAM
 * reference BIT FOR BIT: the per-step losses are compared as doubles for
 * equality, and the final weights as an order-independent integer sum of
 * their float bit patterns. Any stale page, lost update or wrong-order
 * flush is a hard gate failure, not a slightly-different loss curve.
 *
 * OUT OF CORE: --cap M caps the weight cache at M pages. fwd/bwd sweep the
 * whole model every step, so cap < model pages means every step refaults
 * and re-publishes -- Fetch before every hold, Flush before every unpin,
 * the contract this benchmark exists to certify under pressure.
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include "../bench_flush_data.h"
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#endif

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define GV_LB_CORO 1
#endif

CTP_INLINE_CROSS_FUN u64 Lcg(u64 s) { return s * 6364136223846793005ull + 1442695040888963407ull; }
CTP_INLINE_CROSS_FUN float Sym01(u64 s) {
  return (static_cast<float>((s >> 40) & 0xFFFFFF) / 8388608.0f) - 1.0f;
}

#if defined(GV_LB_CORO)

/**
 * fwd1: a1[h,b] = relu(sum_i W1[h,i] x[b,i] + b1[h]).
 *
 * A block owns a contiguous h-range. W1 is row-major (rows of I floats), so
 * an h-range is a contiguous element range and pages slide through it the
 * way grayscott's planes do. ONE GUARD AT A TIME: the row loop fetches the
 * page under the current row, computes every row on that page, releases it.
 */
__device__ gy::YCoroMain Fwd1Coro(gv::DeviceVector<float> w, u64 w1_off,
                                  u64 b1_off, u64 I, u64 H, u64 B,
                                  const float *x, float *a1, u64 h0, u64 h1,
                                  u64 rows_per_page) {
  for (u64 hp = h0; hp < h1; hp += rows_per_page) {
    const u64 page_lo = w1_off + hp * I;
    const u64 hend = (hp + rows_per_page < h1) ? hp + rows_per_page : h1;
    const u64 count = (hend - hp) * I;
    co_await w.Fetch(0, page_lo, count);
    auto hw = co_await w.HoldPage(page_lo, count);
    co_await w.Fetch(0, b1_off + hp, hend - hp);
    auto hb = co_await w.HoldPage(b1_off + hp, hend - hp);
    // One thread per (h, b) output element; the i-sum is sequential.
    const u64 nout = (hend - hp) * B;
    for (u64 t = threadIdx.x; t < nout; t += blockDim.x) {
      const u64 h = hp + t / B;
      const u64 b = t % B;
      float acc = hb[b1_off + h];
      for (u64 i = 0; i < I; ++i) {
        acc += hw[w1_off + h * I + i] * x[b * I + i];
      }
      a1[h * B + b] = acc > 0.0f ? acc : 0.0f;
    }
    __syncthreads();
    w.UnpinRange(page_lo, count);
    w.UnpinRange(b1_off + hp, hend - hp);
  }
}

/** fwd2 + output gradient: z2[o,b], d2[o,b] = 2 (z2 - y) / (B*O), and the
 *  per-(o-range) loss partial, summed once by thread 0 in fixed order. */
__device__ gy::YCoroMain Fwd2Coro(gv::DeviceVector<float> w, u64 w2_off,
                                  u64 b2_off, u64 H, u64 O, u64 B,
                                  const float *a1, const float *y, float *d2,
                                  double *loss_parts, u64 o0, u64 o1,
                                  u64 rows_per_page) {
  for (u64 op = o0; op < o1; op += rows_per_page) {
    const u64 page_lo = w2_off + op * H;
    const u64 oend = (op + rows_per_page < o1) ? op + rows_per_page : o1;
    const u64 count = (oend - op) * H;
    co_await w.Fetch(0, page_lo, count);
    auto hw = co_await w.HoldPage(page_lo, count);
    co_await w.Fetch(0, b2_off + op, oend - op);
    auto hb = co_await w.HoldPage(b2_off + op, oend - op);
    const u64 nout = (oend - op) * B;
    for (u64 t = threadIdx.x; t < nout; t += blockDim.x) {
      const u64 o = op + t / B;
      const u64 b = t % B;
      float acc = hb[b2_off + o];
      for (u64 h = 0; h < H; ++h) {
        acc += hw[w2_off + o * H + h] * a1[h * B + b];
      }
      const float diff = acc - y[b * O + o];
      d2[o * B + b] = 2.0f * diff / static_cast<float>(B * O);
      // Stash the squared error where the deterministic reducer finds it.
      loss_parts[o * B + b] =
          static_cast<double>(diff) * static_cast<double>(diff);
    }
    __syncthreads();
    w.UnpinRange(page_lo, count);
    w.UnpinRange(b2_off + op, oend - op);
  }
}

/** bwd1: d1[h,b] = relu'(a1) * sum_o W2[o,h] d2[o,b]. Reads W2 PRE-update
 *  (launched before upd2), one thread per (h,b), o-sum sequential. The
 *  block again owns o-ROWS of W2 pages for the sliding holds, but every
 *  block needs every row, so blocks stride the row-pages and accumulate
 *  into d1 with a fixed per-element owner: block(h) = h range. */
__device__ gy::YCoroMain Bwd1Coro(gv::DeviceVector<float> w, u64 w2_off,
                                  u64 H, u64 O, u64 B, const float *a1,
                                  const float *d2, float *d1, u64 h0, u64 h1,
                                  u64 rows_per_page) {
  // This block owns d1 rows h0..h1 and must read ALL of W2 for them. Pages
  // slide over the whole of W2; the o-sum stays sequential per element by
  // accumulating across page visits in registers is impossible (o spans
  // pages), so d1 is built in a local sweep: zero first, then += per page.
  for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
    d1[(h0 + t / B) * B + t % B] = 0.0f;
  }
  __syncthreads();
  for (u64 op = 0; op < O; op += rows_per_page) {
    const u64 oend = (op + rows_per_page < O) ? op + rows_per_page : O;
    const u64 page_lo = w2_off + op * H;
    const u64 count = (oend - op) * H;
    co_await w.Fetch(0, page_lo, count);
    auto hw = co_await w.HoldPage(page_lo, count);
    for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
      const u64 h = h0 + t / B;
      const u64 b = t % B;
      float acc = d1[h * B + b];
      for (u64 o = op; o < oend; ++o) {
        acc += hw[w2_off + o * H + h] * d2[o * B + b];
      }
      d1[h * B + b] = acc;
    }
    __syncthreads();
    w.UnpinRange(page_lo, count);
  }
  for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
    const u64 h = h0 + t / B;
    const u64 b = t % B;
    if (a1[h * B + b] <= 0.0f) d1[h * B + b] = 0.0f;
  }
}

/** upd2: W2[o,h] -= lr sum_b d2[o,b] a1[h,b]; b2 likewise. Block owns
 *  o-rows: ONE WRITER PER PAGE, publish at the write site. */
__device__ gy::YCoroMain Upd2Coro(gv::DeviceVector<float> w, u64 w2_off,
                                  u64 b2_off, u64 H, u64 O, u64 B,
                                  const float *a1, const float *d2, float lr,
                                  u64 o0, u64 o1, u64 rows_per_page) {
  for (u64 op = o0; op < o1; op += rows_per_page) {
    const u64 page_lo = w2_off + op * H;
    const u64 oend = (op + rows_per_page < o1) ? op + rows_per_page : o1;
    const u64 count = (oend - op) * H;
    co_await w.Fetch(0, page_lo, count);
    auto hw = co_await w.HoldPage(page_lo, count, /*write=*/true);
    co_await w.Fetch(0, b2_off + op, oend - op);
    auto hb = co_await w.HoldPage(b2_off + op, oend - op, /*write=*/true);
    const u64 nout = (oend - op) * H;
    for (u64 t = threadIdx.x; t < nout; t += blockDim.x) {
      const u64 o = op + t / H;
      const u64 h = t % H;
      float g = 0.0f;
      for (u64 b = 0; b < B; ++b) {
        g += d2[o * B + b] * a1[h * B + b];
      }
      hw[w2_off + o * H + h] -= lr * g;
    }
    for (u64 t = threadIdx.x; t < (oend - op); t += blockDim.x) {
      const u64 o = op + t;
      float g = 0.0f;
      for (u64 b = 0; b < B; ++b) g += d2[o * B + b];
      hb[b2_off + o] -= lr * g;
    }
    __syncthreads();
    co_await w.BeginFlush(0, page_lo, count);
    co_await w.BeginFlush(0, b2_off + op, oend - op);
    w.UnpinRange(page_lo, count);
    w.UnpinRange(b2_off + op, oend - op);
  }
  co_await w.EndFlush();
}

/** upd1: W1[h,i] -= lr d1[h,b] x[b,i]; b1 likewise. Block owns h-rows. */
__device__ gy::YCoroMain Upd1Coro(gv::DeviceVector<float> w, u64 w1_off,
                                  u64 b1_off, u64 I, u64 H, u64 B,
                                  const float *x, const float *d1, float lr,
                                  u64 h0, u64 h1, u64 rows_per_page) {
  for (u64 hp = h0; hp < h1; hp += rows_per_page) {
    const u64 page_lo = w1_off + hp * I;
    const u64 hend = (hp + rows_per_page < h1) ? hp + rows_per_page : h1;
    const u64 count = (hend - hp) * I;
    co_await w.Fetch(0, page_lo, count);
    auto hw = co_await w.HoldPage(page_lo, count, /*write=*/true);
    co_await w.Fetch(0, b1_off + hp, hend - hp);
    auto hb = co_await w.HoldPage(b1_off + hp, hend - hp, /*write=*/true);
    const u64 nout = (hend - hp) * I;
    for (u64 t = threadIdx.x; t < nout; t += blockDim.x) {
      const u64 h = hp + t / I;
      const u64 i = t % I;
      float g = 0.0f;
      for (u64 b = 0; b < B; ++b) {
        g += d1[h * B + b] * x[b * I + i];
      }
      hw[w1_off + h * I + i] -= lr * g;
    }
    for (u64 t = threadIdx.x; t < (hend - hp); t += blockDim.x) {
      const u64 h = hp + t;
      float g = 0.0f;
      for (u64 b = 0; b < B; ++b) g += d1[h * B + b];
      hb[b1_off + h] -= lr * g;
    }
    __syncthreads();
    co_await w.BeginFlush(0, page_lo, count);
    co_await w.BeginFlush(0, b1_off + hp, hend - hp);
    w.UnpinRange(page_lo, count);
    w.UnpinRange(b1_off + hp, hend - hp);
  }
  co_await w.EndFlush();
}

/** Seed the weights deterministically and publish them. */
__device__ gy::YCoroMain SeedCoro(gv::DeviceVector<float> w, u64 n, u64 e0,
                                  u64 e1, u64 chunk) {
  for (u64 lo = e0; lo < e1; lo += chunk) {
    const u64 hi = (lo + chunk < e1) ? lo + chunk : e1;
    co_await w.Fetch(0, lo, hi - lo);
    auto h = co_await w.HoldPage(lo, hi - lo, /*write=*/true);
    for (u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      h[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + i)) * 0.05f;
    }
    __syncthreads();
    co_await w.BeginFlush(0, lo, hi - lo);
    w.UnpinRange(lo, hi - lo);
  }
  co_await w.EndFlush();
}

/** Order-independent integer digest of the weights: sum of bit patterns. */
__device__ gy::YCoroMain DigestCoro(gv::DeviceVector<float> w, u64 e0, u64 e1,
                                    u64 chunk, unsigned long long *out) {
  unsigned long long acc = 0;
  for (u64 lo = e0; lo < e1; lo += chunk) {
    const u64 hi = (lo + chunk < e1) ? lo + chunk : e1;
    co_await w.Fetch(0, lo, hi - lo);
    auto h = co_await w.HoldPage(lo, hi - lo);
    for (u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      acc += static_cast<unsigned long long>(__float_as_uint(h[i])) *
             (2ull * i + 1ull);
    }
    __syncthreads();
    w.UnpinRange(lo, hi - lo);
  }
  atomicAdd(out, acc);
}

#define LB_KERNEL(NAME, CORO_CALL, ...)                                       \
  __global__ void NAME(clio::run::IpcManagerGpuInfo info,                     \
                       gv::DeviceVector<float> w, __VA_ARGS__,                \
                       gy::YieldableView<> yv, gy::YieldStackView ys) {       \
    CLIO_GPU_INIT(info, nullptr);                                             \
    w.Init(yv.Block());                                                       \
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());                              \
    __syncthreads();                                                          \
    CLIO_YCORO_RUN(CORO_CALL);                                                \
  }

LB_KERNEL(Fwd1Kernel,
          Fwd1Coro(w, w1_off, b1_off, I, H, B, x, a1,
                   static_cast<u64>(yv.Block()) * hper,
                   ((static_cast<u64>(yv.Block()) + 1) * hper < H)
                       ? (static_cast<u64>(yv.Block()) + 1) * hper : H,
                   rpp),
          u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x,
          float *a1, u64 hper, u64 rpp)
LB_KERNEL(Fwd2Kernel,
          Fwd2Coro(w, w2_off, b2_off, H, O, B, a1, y, d2, loss_parts,
                   static_cast<u64>(yv.Block()) * oper,
                   ((static_cast<u64>(yv.Block()) + 1) * oper < O)
                       ? (static_cast<u64>(yv.Block()) + 1) * oper : O,
                   rpp),
          u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1,
          const float *y, float *d2, double *loss_parts, u64 oper, u64 rpp)
LB_KERNEL(Bwd1Kernel,
          Bwd1Coro(w, w2_off, H, O, B, a1, d2, d1,
                   static_cast<u64>(yv.Block()) * hper,
                   ((static_cast<u64>(yv.Block()) + 1) * hper < H)
                       ? (static_cast<u64>(yv.Block()) + 1) * hper : H,
                   rpp),
          u64 w2_off, u64 H, u64 O, u64 B, const float *a1, const float *d2,
          float *d1, u64 hper, u64 rpp)
LB_KERNEL(Upd2Kernel,
          Upd2Coro(w, w2_off, b2_off, H, O, B, a1, d2, lr,
                   static_cast<u64>(yv.Block()) * oper,
                   ((static_cast<u64>(yv.Block()) + 1) * oper < O)
                       ? (static_cast<u64>(yv.Block()) + 1) * oper : O,
                   rpp),
          u64 w2_off, u64 b2_off, u64 H, u64 O, u64 B, const float *a1,
          const float *d2, float lr, u64 oper, u64 rpp)
LB_KERNEL(Upd1Kernel,
          Upd1Coro(w, w1_off, b1_off, I, H, B, x, d1, lr,
                   static_cast<u64>(yv.Block()) * hper,
                   ((static_cast<u64>(yv.Block()) + 1) * hper < H)
                       ? (static_cast<u64>(yv.Block()) + 1) * hper : H,
                   rpp),
          u64 w1_off, u64 b1_off, u64 I, u64 H, u64 B, const float *x,
          const float *d1, float lr, u64 hper, u64 rpp)
LB_KERNEL(SeedKernel,
          SeedCoro(w, n, static_cast<u64>(yv.Block()) * eper,
                   ((static_cast<u64>(yv.Block()) + 1) * eper < n)
                       ? (static_cast<u64>(yv.Block()) + 1) * eper : n,
                   chunk),
          u64 n, u64 eper, u64 chunk)
LB_KERNEL(DigestKernel,
          DigestCoro(w, static_cast<u64>(yv.Block()) * eper,
                     ((static_cast<u64>(yv.Block()) + 1) * eper < n)
                         ? (static_cast<u64>(yv.Block()) + 1) * eper : n,
                     chunk, out),
          u64 n, u64 eper, u64 chunk, unsigned long long *out)

// -------------------- DENSE REFERENCE (same loops, plain memory) ----------

__global__ void DenseSeed(float *w, u64 n) {
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    w[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + i)) * 0.05f;
  }
}

__global__ void DenseFwd1(const float *w, u64 w1_off, u64 b1_off, u64 I,
                          u64 H, u64 B, const float *x, float *a1, u64 hper) {
  const u64 h0 = static_cast<u64>(blockIdx.x) * hper;
  const u64 h1 = (h0 + hper < H) ? h0 + hper : H;
  for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
    const u64 h = h0 + t / B;
    const u64 b = t % B;
    float acc = w[b1_off + h];
    for (u64 i = 0; i < I; ++i) acc += w[w1_off + h * I + i] * x[b * I + i];
    a1[h * B + b] = acc > 0.0f ? acc : 0.0f;
  }
}

__global__ void DenseFwd2(const float *w, u64 w2_off, u64 b2_off, u64 H,
                          u64 O, u64 B, const float *a1, const float *y,
                          float *d2, double *loss_parts, u64 oper) {
  const u64 o0 = static_cast<u64>(blockIdx.x) * oper;
  const u64 o1 = (o0 + oper < O) ? o0 + oper : O;
  for (u64 t = threadIdx.x; t < (o1 - o0) * B; t += blockDim.x) {
    const u64 o = o0 + t / B;
    const u64 b = t % B;
    float acc = w[b2_off + o];
    for (u64 h = 0; h < H; ++h) acc += w[w2_off + o * H + h] * a1[h * B + b];
    const float diff = acc - y[b * O + o];
    d2[o * B + b] = 2.0f * diff / static_cast<float>(B * O);
    loss_parts[o * B + b] =
        static_cast<double>(diff) * static_cast<double>(diff);
  }
}

__global__ void DenseBwd1(const float *w, u64 w2_off, u64 H, u64 O, u64 B,
                          const float *a1, const float *d2, float *d1,
                          u64 hper, u64 rpp) {
  const u64 h0 = static_cast<u64>(blockIdx.x) * hper;
  const u64 h1 = (h0 + hper < H) ? h0 + hper : H;
  for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
    d1[(h0 + t / B) * B + t % B] = 0.0f;
  }
  __syncthreads();
  // SAME page-shaped o-blocking as the paged path, so the float sums
  // associate identically and the comparison can be bit-exact.
  for (u64 op = 0; op < O; op += rpp) {
    const u64 oend = (op + rpp < O) ? op + rpp : O;
    for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
      const u64 h = h0 + t / B;
      const u64 b = t % B;
      float acc = d1[h * B + b];
      for (u64 o = op; o < oend; ++o) {
        acc += w[w2_off + o * H + h] * d2[o * B + b];
      }
      d1[h * B + b] = acc;
    }
    __syncthreads();
  }
  for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
    const u64 h = h0 + t / B;
    const u64 b = t % B;
    if (a1[h * B + b] <= 0.0f) d1[h * B + b] = 0.0f;
  }
}

__global__ void DenseUpd2(float *w, u64 w2_off, u64 b2_off, u64 H, u64 O,
                          u64 B, const float *a1, const float *d2, float lr,
                          u64 oper) {
  const u64 o0 = static_cast<u64>(blockIdx.x) * oper;
  const u64 o1 = (o0 + oper < O) ? o0 + oper : O;
  for (u64 t = threadIdx.x; t < (o1 - o0) * H; t += blockDim.x) {
    const u64 o = o0 + t / H;
    const u64 h = t % H;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[o * B + b] * a1[h * B + b];
    w[w2_off + o * H + h] -= lr * g;
  }
  for (u64 t = threadIdx.x; t < (o1 - o0); t += blockDim.x) {
    const u64 o = o0 + t;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[o * B + b];
    w[b2_off + o] -= lr * g;
  }
}

__global__ void DenseUpd1(float *w, u64 w1_off, u64 b1_off, u64 I, u64 H,
                          u64 B, const float *x, const float *d1, float lr,
                          u64 hper) {
  const u64 h0 = static_cast<u64>(blockIdx.x) * hper;
  const u64 h1 = (h0 + hper < H) ? h0 + hper : H;
  for (u64 t = threadIdx.x; t < (h1 - h0) * I; t += blockDim.x) {
    const u64 h = h0 + t / I;
    const u64 i = t % I;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[h * B + b] * x[b * I + i];
    w[w1_off + h * I + i] -= lr * g;
  }
  for (u64 t = threadIdx.x; t < (h1 - h0); t += blockDim.x) {
    const u64 h = h0 + t;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[h * B + b];
    w[b1_off + h] -= lr * g;
  }
}

__global__ void DenseDigest(const float *w, u64 n, unsigned long long *out) {
  unsigned long long acc = 0;
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    acc += static_cast<unsigned long long>(__float_as_uint(w[i])) *
           (2ull * i + 1ull);
  }
  atomicAdd(out, acc);
}

#endif  // GV_LB_CORO

#if !CTP_IS_DEVICE_PASS

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

#if defined(GV_LB_CORO)
class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000, gv::ResumeWhenComplete);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};
#endif

}  // namespace

int main(int argc, char **argv) {
  u32 blocks = 8, threads = 256, cap = 0;
  u64 page_kb = 64, I = 256, H = 4096, O = 64, B = 64, steps = 5;
  float lr = 0.01f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--cap") cap = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--in") I = next();
    else if (a == "--hidden") H = next();
    else if (a == "--out") O = next();
    else if (a == "--batch") B = next();
    else if (a == "--steps") steps = next();
    else if (a == "--lr" && i + 1 < argc) lr = std::strtof(argv[++i], nullptr);
    else if (a == "--help") {
      std::printf("usage: %s [--blocks N] [--threads N] [--cap PAGES] "
                  "[--page-kb N] [--in N] [--hidden N] [--out N] [--batch N] "
                  "[--steps N] [--lr F]\n", argv[0]);
      return 0;
    }
  }

#if !defined(GV_LB_CORO)
  std::fprintf(stderr, "LBANN ERROR: built without C++20 device coroutines.\n");
  return 2;
#else
  const u64 page_bytes = page_kb * 1024;
  const u64 elems_per_page = page_bytes / sizeof(float);
  // Rows must tile pages exactly so an h-range is a whole-page range and
  // the one-writer-per-page rule holds by construction.
  if (elems_per_page % I != 0 || elems_per_page % H != 0) {
    std::fprintf(stderr, "LBANN ERROR: a %lluKB page must hold whole rows of "
                 "W1 (I=%llu) and W2 (H=%llu).\n",
                 (unsigned long long)page_kb, (unsigned long long)I,
                 (unsigned long long)H);
    return 2;
  }
  const u64 w1_n = H * I, w2_n = O * H;
  const u64 w1_off = 0;
  const u64 b1_off = w1_n;                       // b1 lives page-aligned after W1
  const u64 b1_pad = elems_per_page;             // one page for b1
  const u64 w2_off = w1_n + b1_pad;
  const u64 b2_off = w2_off + w2_n;
  const u64 b2_pad = elems_per_page;
  const u64 n = w2_off + w2_n + b2_pad;
  const u64 npages = (n + elems_per_page - 1) / elems_per_page;
  const u64 rpp1 = elems_per_page / I;           // W1 rows per page
  const u64 rpp2 = elems_per_page / H;           // W2 rows per page
  if (H % blocks != 0 || O % blocks != 0 ||
      (H / blocks) % rpp1 != 0 || (O / blocks) % (rpp2 ? rpp2 : 1) != 0) {
    std::fprintf(stderr, "LBANN ERROR: blocks must evenly split H and O into "
                 "whole pages (H=%llu O=%llu rpp1=%llu rpp2=%llu blocks=%u).\n",
                 (unsigned long long)H, (unsigned long long)O,
                 (unsigned long long)rpp1, (unsigned long long)rpp2, blocks);
    return 2;
  }
  const u64 hper = H / blocks, oper = O / blocks;

  {
    std::ofstream cfg("gv_lbann_bench.yaml");
    cfg << "networking:\n  port: 9449\n\n"
        << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
        << "  first_busy_wait: 10000000\n\n"
        << "gpu:\n  queue_depth: 8192\n\n"
        << "compose:\n"
        << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
        << "    pool_query: local\n    pool_id: \"301.0\"\n"
        << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
        << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
        << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
        << "      - path: \"ram::gv_lb_ram\"\n        bdev_type: \"ram\"\n"
        << "        capacity_limit: \"4GB\"\n        score: 1.0\n"
        << "    dpe:\n      dpe_type: \"max_bw\"\n";
    cfg.close();
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_lbann_bench.yaml", 1);
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "LBANN ERROR: runtime init failed\n");
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "LBANN ERROR: cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  std::printf("MLP training with paged weights\n"
              "  layers %llu -> %llu -> %llu  batch=%llu  steps=%llu lr=%.3f\n"
              "  weights %.1f MB in %llu pages of %lluKB  cache=%s\n"
              "  blocks=%u threads=%u\n",
              (unsigned long long)I, (unsigned long long)H,
              (unsigned long long)O, (unsigned long long)B,
              (unsigned long long)steps, lr,
              static_cast<double>(n * sizeof(float)) / 1048576.0,
              (unsigned long long)npages, (unsigned long long)page_kb,
              cap == 0 ? "resident" : (std::to_string(cap) + " pages").c_str(),
              blocks, threads);

  // ---- Batch and targets: deterministic, resident. -----------------------
  std::vector<float> hxv(B * I), hyv(B * O);
  for (u64 i = 0; i < B * I; ++i) hxv[i] = Sym01(Lcg(0xA02BDBF7BB3C0A7ull + i));
  for (u64 i = 0; i < B * O; ++i) hyv[i] = Sym01(Lcg(0x6C62272E07BB0142ull + i));
  auto *d_x = ctp::GpuApi::Malloc<float>(B * I * sizeof(float));
  auto *d_y = ctp::GpuApi::Malloc<float>(B * O * sizeof(float));
  ctp::GpuApi::Memcpy(d_x, hxv.data(), B * I * sizeof(float));
  ctp::GpuApi::Memcpy(d_y, hyv.data(), B * O * sizeof(float));
  auto *d_a1 = ctp::GpuApi::Malloc<float>(H * B * sizeof(float));
  auto *d_d1 = ctp::GpuApi::Malloc<float>(H * B * sizeof(float));
  auto *d_d2 = ctp::GpuApi::Malloc<float>(O * B * sizeof(float));
  auto *d_lp = ctp::GpuApi::Malloc<double>(O * B * sizeof(double));
  auto *d_dg = ctp::GpuApi::Malloc<unsigned long long>(
      2 * sizeof(unsigned long long));

  std::vector<double> loss_ref(steps), loss_got(steps);
  auto host_loss = [&]() {
    std::vector<double> lp(O * B);
    ctp::GpuApi::Memcpy(lp.data(), d_lp, O * B * sizeof(double));
    double s = 0.0;                 // fixed order: one deterministic sum
    for (u64 i = 0; i < O * B; ++i) s += lp[i];
    return s / static_cast<double>(B * O);
  };

  // ---- Dense reference training. -----------------------------------------
  auto *d_wref = ctp::GpuApi::Malloc<float>(n * sizeof(float));
  DenseSeed<<<64, 256>>>(d_wref, n);
  ctp::GpuApi::Synchronize();
  const double t_ref0 = NowMs();
  for (u64 s = 0; s < steps; ++s) {
    DenseFwd1<<<blocks, threads>>>(d_wref, w1_off, b1_off, I, H, B, d_x, d_a1,
                                   hper);
    DenseFwd2<<<blocks, threads>>>(d_wref, w2_off, b2_off, H, O, B, d_a1, d_y,
                                   d_d2, d_lp, oper);
    DenseBwd1<<<blocks, threads>>>(d_wref, w2_off, H, O, B, d_a1, d_d2, d_d1,
                                   hper, rpp2 ? rpp2 : 1);
    DenseUpd2<<<blocks, threads>>>(d_wref, w2_off, b2_off, H, O, B, d_a1,
                                   d_d2, lr, oper);
    DenseUpd1<<<blocks, threads>>>(d_wref, w1_off, b1_off, I, H, B, d_x, d_d1,
                                   lr, hper);
    ctp::GpuApi::Synchronize();
    loss_ref[s] = host_loss();
  }
  const double t_ref = NowMs() - t_ref0;
  ctp::GpuApi::Memset(d_dg, 0, 2 * sizeof(unsigned long long));
  DenseDigest<<<64, 256>>>(d_wref, n, &d_dg[0]);
  ctp::GpuApi::Synchronize();

  // ---- Paged training. ---------------------------------------------------
  const u64 eper = ((npages + blocks - 1) / blocks) * elems_per_page;
  gv::Vector<float> w("gv_lbann_w", {0}, page_bytes, blocks, 24, n,
                      clio::run::PoolId::GetNull(), 0, 1, 0,
                      cap == 0 ? static_cast<u32>(npages + 2) : cap);
  w.EnableStats();
  auto dw = w.GetDevice(0);
  YieldRunner runner(blocks, threads);
  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                 gy::YieldStackView sv) {
    SeedKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dw, n, eper,
                                                elems_per_page, vw, sv);
  });
  ctp::GpuApi::Synchronize();

  const double t0 = NowMs();
  for (u64 s = 0; s < steps; ++s) {
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      Fwd1Kernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dw, w1_off, b1_off, I,
                                                  H, B, d_x, d_a1, hper, rpp1,
                                                  vw, sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      Fwd2Kernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dw, w2_off, b2_off, H,
                                                  O, B, d_a1, d_y, d_d2, d_lp,
                                                  oper, rpp2 ? rpp2 : 1, vw,
                                                  sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      Bwd1Kernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dw, w2_off, H, O, B,
                                                  d_a1, d_d2, d_d1, hper,
                                                  rpp2 ? rpp2 : 1, vw, sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      Upd2Kernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dw, w2_off, b2_off, H,
                                                  O, B, d_a1, d_d2, lr, oper,
                                                  rpp2 ? rpp2 : 1, vw, sv);
    });
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      Upd1Kernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dw, w1_off, b1_off, I,
                                                  H, B, d_x, d_d1, lr, hper,
                                                  rpp1, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    loss_got[s] = host_loss();
  }
  const double t_paged = NowMs() - t0;
  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                 gy::YieldStackView sv) {
    DigestKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dw, n, eper,
                                                  elems_per_page, &d_dg[1],
                                                  vw, sv);
  });
  ctp::GpuApi::Synchronize();
  unsigned long long dg[2] = {0, 0};
  ctp::GpuApi::Memcpy(dg, d_dg, sizeof(dg));

  const auto st = w.ReadStats(0);
  std::printf("  paging: faults=%llu evicts=%llu puts=%llu get_errors=%llu "
              "put_errors=%llu\n",
              (unsigned long long)st.faults, (unsigned long long)st.evicts,
              (unsigned long long)st.puts, (unsigned long long)st.get_errors,
              (unsigned long long)st.put_errors);
  std::printf("  %llu steps: paged %.1f ms/step, dense %.1f ms/step\n",
              (unsigned long long)steps, t_paged / steps, t_ref / steps);

  int rc = 0;
  bool loss_ok = true;
  for (u64 s = 0; s < steps; ++s) {
    if (loss_got[s] != loss_ref[s]) {
      std::printf("  LOSS GATE: step %llu paged %.17g != dense %.17g\n",
                  (unsigned long long)s, loss_got[s], loss_ref[s]);
      loss_ok = false;
    }
  }
  if (loss_ok) {
    std::printf("  LOSS GATE: PASS (all %llu steps bit-equal; final loss "
                "%.6f -> %.6f)\n",
                (unsigned long long)steps, loss_ref[0],
                loss_ref[steps - 1]);
  } else {
    rc = 1;
  }
  if (dg[0] != dg[1]) {
    std::printf("  WEIGHT GATE: FAIL (digest paged %llu != dense %llu)\n",
                dg[1], dg[0]);
    rc = 1;
  } else {
    std::printf("  WEIGHT GATE: PASS (final weights bit-equal to dense "
                "reference)\n");
  }
  std::printf("%s\n", rc == 0 ? "LBANN BENCH: ALL GATES PASS"
                              : "LBANN BENCH: GATE FAILURE");
  BenchFlushData();
  return rc;
#endif  // GV_LB_CORO
}
#endif  // !CTP_IS_DEVICE_PASS
