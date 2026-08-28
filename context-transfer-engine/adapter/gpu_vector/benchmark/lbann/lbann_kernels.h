/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The LBANN MLP device code -- ONE copy, compiled by CUDA and by SYCL.
 *
 * Same split as the other workloads: the workload is here, the launches are
 * in cuda/ and sycl/. Nothing here is backend-specific -- no warp
 * intrinsics, no __shared__, no cooperative groups.
 *
 * CTP_GPU_FUN, not plain inline: under CUDA these must be __device__ or the
 * host pass compiles them as host functions and every co_await on a
 * device-only verb fails to resolve.
 */
#ifndef CLIO_GV_BENCH_LBANN_KERNELS_H_
#define CLIO_GV_BENCH_LBANN_KERNELS_H_

#include "lbann_math.h"

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::lbann {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;

using ::clio_lb::Lcg;
using ::clio_lb::Sym01;


/**
 * fwd1: a1[h,b] = relu(sum_i W1[h,i] x[b,i] + b1[h]).
 *
 * A block owns a contiguous h-range. W1 is row-major (rows of I floats), so
 * an h-range is a contiguous element range and pages slide through it the
 * way grayscott's planes do. ONE GUARD AT A TIME: the row loop fetches the
 * page under the current row, computes every row on that page, releases it.
 */
CTP_GPU_FUN inline gy::YCoroMain Fwd1Coro(gv::DeviceVector<float> w, u64 w1_off,
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
CTP_GPU_FUN inline gy::YCoroMain Fwd2Coro(gv::DeviceVector<float> w, u64 w2_off,
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
CTP_GPU_FUN inline gy::YCoroMain Bwd1Coro(gv::DeviceVector<float> w, u64 w2_off,
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
CTP_GPU_FUN inline gy::YCoroMain Upd2Coro(gv::DeviceVector<float> w, u64 w2_off,
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
CTP_GPU_FUN inline gy::YCoroMain Upd1Coro(gv::DeviceVector<float> w, u64 w1_off,
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
CTP_GPU_FUN inline gy::YCoroMain SeedCoro(gv::DeviceVector<float> w, u64 n, u64 e0,
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
CTP_GPU_FUN inline gy::YCoroMain DigestCoro(gv::DeviceVector<float> w, u64 e0, u64 e1,
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

// -------------------- DENSE REFERENCE (same loops, plain memory) ----------

CTP_GPU_FUN inline void DenseSeed(float *w, u64 n) {
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    w[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + i)) * 0.05f;
  }
}

CTP_GPU_FUN inline void DenseFwd1(const float *w, u64 w1_off, u64 b1_off, u64 I,
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

CTP_GPU_FUN inline void DenseFwd2(const float *w, u64 w2_off, u64 b2_off, u64 H,
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

CTP_GPU_FUN inline void DenseBwd1(const float *w, u64 w2_off, u64 H, u64 O, u64 B,
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

CTP_GPU_FUN inline void DenseUpd2(float *w, u64 w2_off, u64 b2_off, u64 H, u64 O,
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

CTP_GPU_FUN inline void DenseUpd1(float *w, u64 w1_off, u64 b1_off, u64 I, u64 H,
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

CTP_GPU_FUN inline void DenseDigest(const float *w, u64 n, unsigned long long *out) {
  unsigned long long acc = 0;
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    acc += static_cast<unsigned long long>(__float_as_uint(w[i])) *
           (2ull * i + 1ull);
  }
  atomicAdd(out, acc);
}


}  // namespace clio::gv_bench::lbann

#endif  // CLIO_GV_BENCH_LBANN_KERNELS_H_
