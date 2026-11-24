/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * lbann paged kernels, MACRO (Duff's device) form.
 *
 * Same workload as lbann_kernels.h with suspension expressed through
 * yield_stack.h's macros rather than C++20 device coroutines, which clang
 * cannot compile for SPIR-V. See kmeans_macros_kernels.h for the rationale.
 *
 * THE ONE RULE. Every `const u64 page_lo/count/hend/oend` in the coroutine
 * bodies is computed BEFORE the fetch and read AFTER it, so each becomes a
 * frame local here. That is not a stylistic choice: an initialized automatic
 * in that position also makes the resume label unreachable, and the compiler
 * says so ("cannot jump from switch statement to this case label").
 *
 * MaxDiffMacro's early `co_return` becomes a guarded body rather than a bare
 * return: CLIO_YEND pops the frame, so returning around it would leak a frame
 * for the rest of the launch.
 */
#ifndef CLIO_GV_BENCH_LBANN_MACROS_KERNELS_H_
#define CLIO_GV_BENCH_LBANN_MACROS_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

#include "lbann_kernels.h"

namespace clio::gv_bench::lbann_macros {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;
// Plain helpers beside the coroutines (Sym01, Lcg, the dense bodies).
using namespace ::clio::gv_bench::lbann;

using WV = gv::DeviceVector<float>;
using HeldF = gv::Held<float>;

/** Macro-form Fwd1Coro. */
CTP_GPU_FUN inline void Fwd1Macro(WV w, u64 w1_off, const float *b1v, u64 I,
                                  u64 H, u64 B, const float *x, float *a1,
                                  u64 h0, u64 h1, u64 rows_per_page, u64 gen) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, hp, h0);
  CLIO_YLOCAL_INIT(u64, page_lo, 0);
  CLIO_YLOCAL_INIT(u64, hend, 0);
  CLIO_YLOCAL_INIT(u64, count, 0);
  CLIO_YLOCAL(HeldF, hw);
  CLIO_YBEGIN();
  for (; hp < h1; hp += rows_per_page) {
    page_lo = w1_off + hp * I;
    hend = (hp + rows_per_page < h1) ? hp + rows_per_page : h1;
    count = (hend - hp) * I;
    CLIO_YCALL(w.MFetch(gen, page_lo, count));
    CLIO_YCALL(w.MHoldPage(&hw, page_lo, count));
    const u64 nout = (hend - hp) * B;
    for (u64 t = threadIdx.x; t < nout; t += blockDim.x) {
      const u64 h = hp + t / B;
      const u64 b = t % B;
      float acc = b1v[h];
      for (u64 i = 0; i < I; ++i) {
        acc += hw[w1_off + h * I + i] * x[b * I + i];
      }
      a1[h * B + b] = acc > 0.0f ? acc : 0.0f;
    }
    __syncthreads();
    w.UnpinRange(page_lo, count);
  }
  CLIO_YEND();
}

/** Macro-form Fwd2Coro. */
CTP_GPU_FUN inline void Fwd2Macro(WV w, u64 w2_off, const float *b2v, u64 H,
                                  u64 O, u64 B, const float *a1,
                                  const float *y, float *d2,
                                  double *loss_parts, u64 o0, u64 o1,
                                  u64 rows_per_page, u64 gen) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, op, o0);
  CLIO_YLOCAL_INIT(u64, page_lo, 0);
  CLIO_YLOCAL_INIT(u64, oend, 0);
  CLIO_YLOCAL_INIT(u64, count, 0);
  CLIO_YLOCAL(HeldF, hw);
  CLIO_YBEGIN();
  for (; op < o1; op += rows_per_page) {
    page_lo = w2_off + op * H;
    oend = (op + rows_per_page < o1) ? op + rows_per_page : o1;
    count = (oend - op) * H;
    CLIO_YCALL(w.MFetch(gen, page_lo, count));
    CLIO_YCALL(w.MHoldPage(&hw, page_lo, count));
    const u64 nout = (oend - op) * B;
    for (u64 t = threadIdx.x; t < nout; t += blockDim.x) {
      const u64 o = op + t / B;
      const u64 b = t % B;
      float acc = b2v[o];
      for (u64 h = 0; h < H; ++h) {
        acc += hw[w2_off + o * H + h] * a1[h * B + b];
      }
      const float diff = acc - y[b * O + o];
      d2[o * B + b] = 2.0f * diff / static_cast<float>(B * O);
      loss_parts[o * B + b] =
          static_cast<double>(diff) * static_cast<double>(diff);
    }
    __syncthreads();
    w.UnpinRange(page_lo, count);
  }
  CLIO_YEND();
}

/** Macro-form Bwd1Coro. */
CTP_GPU_FUN inline void Bwd1Macro(WV w, u64 w2_off, u64 H, u64 O, u64 B,
                                  const float *a1, const float *d2, float *d1,
                                  u64 h0, u64 h1, u64 rows_per_page, u64 o0,
                                  u64 o1, u64 gen) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, op, 0);
  CLIO_YLOCAL_INIT(u64, oend, 0);
  CLIO_YLOCAL_INIT(u64, page_lo, 0);
  CLIO_YLOCAL_INIT(u64, count, 0);
  CLIO_YLOCAL(HeldF, hw);
  CLIO_YBEGIN();
  for (u64 t = threadIdx.x; t < (h1 - h0) * B; t += blockDim.x) {
    d1[(h0 + t / B) * B + t % B] = 0.0f;
  }
  __syncthreads();
  for (op = 0; op < O; op += rows_per_page) {
    oend = (op + rows_per_page < O) ? op + rows_per_page : O;
    page_lo = w2_off + op * H;
    count = (oend - op) * H;
    (void)o0; (void)o1;
    CLIO_YCALL(w.MFetch(gen, page_lo, count));
    CLIO_YCALL(w.MHoldPage(&hw, page_lo, count));
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
  CLIO_YEND();
}

/** Macro-form Upd2Coro. */
CTP_GPU_FUN inline void Upd2Macro(WV w, u64 w2_off, float *b2v, u64 H, u64 O,
                                  u64 B, const float *a1, const float *d2,
                                  float lr, u64 o0, u64 o1, u64 rows_per_page,
                                  u64 gen, bool do_bias) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, op, o0);
  CLIO_YLOCAL_INIT(u64, page_lo, 0);
  CLIO_YLOCAL_INIT(u64, oend, 0);
  CLIO_YLOCAL_INIT(u64, count, 0);
  CLIO_YLOCAL(HeldF, hw);
  CLIO_YBEGIN();
  for (; op < o1; op += rows_per_page) {
    page_lo = w2_off + op * H;
    oend = (op + rows_per_page < o1) ? op + rows_per_page : o1;
    count = (oend - op) * H;
    CLIO_YCALL(w.MFetch(gen - 1, page_lo, count));
    CLIO_YCALL(w.MHoldPage(&hw, page_lo, count, /*write=*/true));
    // Braced: a CLIO_YCALL case label follows in this block, and jumping to
    // it must not cross an initialised automatic.
    {
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
    }
    __syncthreads();
    CLIO_YCALL(w.MBeginFlush(gen, page_lo, count));
    w.UnpinRange(page_lo, count);
  }
  if (do_bias) {
    for (u64 o = threadIdx.x; o < O; o += blockDim.x) {
      float g = 0.0f;
      for (u64 b = 0; b < B; ++b) g += d2[o * B + b];
      b2v[o] -= lr * g;
    }
    __syncthreads();
  }
  CLIO_YCALL(w.MEndFlush());
  CLIO_YEND();
}

/** Macro-form Upd1Coro. */
CTP_GPU_FUN inline void Upd1Macro(WV w, u64 w1_off, float *b1v, u64 I, u64 H,
                                  u64 B, const float *x, const float *d1,
                                  float lr, u64 h0, u64 h1, u64 rows_per_page,
                                  u64 gen, bool do_bias) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, hp, h0);
  CLIO_YLOCAL_INIT(u64, page_lo, 0);
  CLIO_YLOCAL_INIT(u64, hend, 0);
  CLIO_YLOCAL_INIT(u64, count, 0);
  CLIO_YLOCAL(HeldF, hw);
  CLIO_YBEGIN();
  for (; hp < h1; hp += rows_per_page) {
    page_lo = w1_off + hp * I;
    hend = (hp + rows_per_page < h1) ? hp + rows_per_page : h1;
    count = (hend - hp) * I;
    CLIO_YCALL(w.MFetch(gen - 1, page_lo, count));
    CLIO_YCALL(w.MHoldPage(&hw, page_lo, count, /*write=*/true));
    // Braced, for the same reason as Upd2Macro above.
    {
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
    }
    __syncthreads();
    CLIO_YCALL(w.MBeginFlush(gen, page_lo, count));
    w.UnpinRange(page_lo, count);
  }
  if (do_bias) {
    for (u64 h = threadIdx.x; h < H; h += blockDim.x) {
      float g = 0.0f;
      for (u64 b = 0; b < B; ++b) g += d1[h * B + b];
      b1v[h] -= lr * g;
    }
    __syncthreads();
  }
  CLIO_YCALL(w.MEndFlush());
  CLIO_YEND();
}

/** Macro-form SeedCoro. */
CTP_GPU_FUN inline void SeedMacro(WV w, u64 n, u64 e0, u64 e1, u64 chunk) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, lo, e0);
  CLIO_YLOCAL_INIT(u64, hi, 0);
  CLIO_YLOCAL(HeldF, h);
  CLIO_YBEGIN();
  for (; lo < e1; lo += chunk) {
    hi = (lo + chunk < e1) ? lo + chunk : e1;
    CLIO_YCALL(w.MFetch(0, lo, hi - lo));
    CLIO_YCALL(w.MHoldPage(&h, lo, hi - lo, /*write=*/true));
    for (u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      h[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + i)) * 0.05f;
    }
    __syncthreads();
    CLIO_YCALL(w.MBeginFlush(1, lo, hi - lo));
    w.UnpinRange(lo, hi - lo);
  }
  CLIO_YCALL(w.MEndFlush());
  CLIO_YEND();
}

/** Macro-form DigestCoro. */
CTP_GPU_FUN inline void DigestMacro(WV w, u64 e0, u64 e1, u64 chunk,
                                    unsigned long long *out) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, lo, e0);
  CLIO_YLOCAL_INIT(u64, hi, 0);
  CLIO_YLOCAL_INIT(unsigned long long, acc, 0ull);
  CLIO_YLOCAL(HeldF, h);
  CLIO_YBEGIN();
  for (; lo < e1; lo += chunk) {
    hi = (lo + chunk < e1) ? lo + chunk : e1;
    CLIO_YCALL(w.MFetch(0, lo, hi - lo));
    CLIO_YCALL(w.MHoldPage(&h, lo, hi - lo));
    for (u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      acc += static_cast<unsigned long long>(__float_as_uint(h[i])) *
             (2ull * i + 1ull);
    }
    __syncthreads();
    w.UnpinRange(lo, hi - lo);
  }
  atomicAdd(out, acc);
  CLIO_YEND();
}

/** Macro-form MaxDiffCoro. The coroutine's early co_return becomes a guarded
 *  body: CLIO_YEND pops the frame, so a bare return would leak one. */
CTP_GPU_FUN inline void MaxDiffMacro(WV w, u64 e0, u64 e1, u64 chunk,
                                     const float *ref, unsigned long long *out,
                                     u64 rlo, u64 rhi, u64 gen) {
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(u64, lo, 0);
  CLIO_YLOCAL_INIT(u64, hi, 0);
  CLIO_YLOCAL_INIT(u64, b0, 0);
  CLIO_YLOCAL_INIT(u64, b1, 0);
  CLIO_YLOCAL_INIT(unsigned long long, acc, 0ull);
  CLIO_YLOCAL(HeldF, h);
  CLIO_YBEGIN();
  b0 = e0 < rlo ? rlo : e0;
  b1 = e1 > rhi ? rhi : e1;
  if (b0 < b1) {
    for (lo = b0; lo < b1; lo += chunk) {
      hi = (lo + chunk < b1) ? lo + chunk : b1;
      CLIO_YCALL(w.MFetch(gen, lo, hi - lo));
      CLIO_YCALL(w.MHoldPage(&h, lo, hi - lo));
      for (u64 i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float d = h[i] - ref[i];
        const float a = d < 0.0f ? -d : d;
        const unsigned long long q =
            static_cast<unsigned long long>(a * 1e9f);
        if (q > acc) acc = q;
      }
      __syncthreads();
      w.UnpinRange(lo, hi - lo);
    }
    unsigned long long old = *out;
    while (acc > old) {
      const unsigned long long prev = atomicCAS(out, old, acc);
      if (prev == old) break;    // we won
      old = prev;
    }
  }
  CLIO_YEND();
}

}  // namespace clio::gv_bench::lbann_macros

#endif  // CLIO_GV_BENCH_LBANN_MACROS_KERNELS_H_
