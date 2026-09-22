/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * MLP training, Aurora baseline editions (MPI / oneCCL / Intel SHMEM).
 *
 * MODEL-parallel, rank <-> the paged bench's block. A rank owns h-rows of W1
 * and o-rows of W2 in device memory; per step it computes its a1 rows
 * (allgathered), its z2/d2 rows (allgathered), its o-partial of the backprop
 * sum (allgathered and combined IN RANK ORDER, which preserves the paged
 * bench's page-blocked float ordering), and updates its own shards in place.
 * One work-item per output element with fixed-order sums throughout, so the
 * per-step LOSS and the final WEIGHT DIGEST must be BIT-EQUAL to the
 * in-process dense reference at any rank count. The allgathers are the
 * substrate's (gv_comm.h), on device buffers. Links nothing from clio.
 *
 * Run recipe: mpiexec -n 4 --ppn 1 clio_lbann_<sub>_bench --hidden 4096
 *             --steps 5
 */

#include "gv_comm.h"
#include "../lbann/lbann_math.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using gvc::u32;
using gvc::u64;
using clio_lb::Lcg;
using clio_lb::Sym01;

namespace {

/** Launch a grid-stride kernel: `nthreads` total work-items in groups of
 *  `threads`, body(t) called for t in [0, n) with stride nthreads. */
template <typename Body>
void GridStride(sycl::queue &q, u32 blocks, u32 threads, u64 n, Body body) {
  const size_t g = static_cast<size_t>(blocks) * threads;
  q.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
     for (u64 t = it.get_global_id(0); t < n; t += g) body(t);
   }).wait();
}

/** a1[h,b] over h in [h0,h1): w1 points at row h0 of W1, b1 at bias h0. */
void Fwd1(sycl::queue &q, u32 bl, u32 th, const float *w1, const float *b1,
          u64 h0, u64 h1, u64 I, u64 B, const float *x, float *a1) {
  GridStride(q, bl, th, (h1 - h0) * B, [=](u64 t) {
    const u64 h = h0 + t / B;
    const u64 b = t % B;
    float acc = b1[h - h0];
    for (u64 i = 0; i < I; ++i) acc += w1[(h - h0) * I + i] * x[b * I + i];
    a1[h * B + b] = acc > 0.0f ? acc : 0.0f;
  });
}

void Fwd2(sycl::queue &q, u32 bl, u32 th, const float *w2, const float *b2,
          u64 o0, u64 o1, u64 H, u64 O, u64 B, const float *a1,
          const float *y, float *d2, double *loss_parts) {
  GridStride(q, bl, th, (o1 - o0) * B, [=](u64 t) {
    const u64 o = o0 + t / B;
    const u64 b = t % B;
    float acc = b2[o - o0];
    for (u64 h = 0; h < H; ++h) acc += w2[(o - o0) * H + h] * a1[h * B + b];
    const float diff = acc - y[b * O + o];
    d2[o * B + b] = 2.0f * diff / static_cast<float>(B * O);
    loss_parts[o * B + b] =
        static_cast<double>(diff) * static_cast<double>(diff);
  });
}

/** This rank's o-rows' PARTIAL of d1, page-blocked in o exactly like the
 *  paged bwd1 so the float order matches when partials are combined in
 *  ascending rank (= ascending o) order. */
void Bwd1Partial(sycl::queue &q, u32 bl, u32 th, const float *w2, u64 o0,
                 u64 o1, u64 H, u64 B, const float *d2, float *d1_part,
                 u64 rpp) {
  GridStride(q, bl, th, H * B, [=](u64 t) {
    const u64 h = t / B;
    const u64 b = t % B;
    float acc = 0.0f;
    for (u64 op = o0; op < o1; op += rpp) {
      const u64 oend = (op + rpp < o1) ? op + rpp : o1;
      float blk = 0.0f;
      for (u64 o = op; o < oend; ++o) blk += w2[(o - o0) * H + h] * d2[o * B + b];
      acc += blk;
    }
    d1_part[h * B + b] = acc;
  });
}

/** Combine rank partials IN RANK ORDER (deterministic), apply relu mask. */
void Bwd1Combine(sycl::queue &q, u32 bl, u32 th, const float *parts,
                 u64 nranks, u64 H, u64 B, const float *a1, float *d1) {
  const u64 nout = H * B;
  GridStride(q, bl, th, nout, [=](u64 t) {
    float acc = 0.0f;
    for (u64 r = 0; r < nranks; ++r) acc += parts[r * nout + t];
    d1[t] = (a1[t] <= 0.0f) ? 0.0f : acc;
  });
}

void Upd2(sycl::queue &q, u32 bl, u32 th, float *w2, float *b2, u64 o0,
          u64 o1, u64 H, u64 B, const float *a1, const float *d2, float lr) {
  GridStride(q, bl, th, (o1 - o0) * H, [=](u64 t) {
    const u64 o = o0 + t / H;
    const u64 h = t % H;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[o * B + b] * a1[h * B + b];
    w2[(o - o0) * H + h] -= lr * g;
  });
  GridStride(q, bl, th, o1 - o0, [=](u64 t) {
    const u64 o = o0 + t;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d2[o * B + b];
    b2[o - o0] -= lr * g;
  });
}

void Upd1(sycl::queue &q, u32 bl, u32 th, float *w1, float *b1, u64 h0,
          u64 h1, u64 I, u64 B, const float *x, const float *d1, float lr) {
  GridStride(q, bl, th, (h1 - h0) * I, [=](u64 t) {
    const u64 h = h0 + t / I;
    const u64 i = t % I;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[h * B + b] * x[b * I + i];
    w1[(h - h0) * I + i] -= lr * g;
  });
  GridStride(q, bl, th, h1 - h0, [=](u64 t) {
    const u64 h = h0 + t;
    float g = 0.0f;
    for (u64 b = 0; b < B; ++b) g += d1[h * B + b];
    b1[h - h0] -= lr * g;
  });
}

/** Order-independent integer digest over LOGICAL element ids. */
void Digest(sycl::queue &q, const float *w, u64 gbase, u64 n,
            unsigned long long *out) {
  GridStride(q, 64, 256, n, [=](u64 i) {
    const unsigned long long bits = sycl::bit_cast<unsigned>(w[i]);
    gvc::AtomicAdd(out, bits * (2ull * (gbase + i) + 1ull));
  });
}

/** Seed a weight shard on the host from the LOGICAL-id generator. */
void SeedShard(sycl::queue &q, float *dst, u64 id0, u64 n) {
  std::vector<float> t(static_cast<size_t>(n));
  for (u64 i = 0; i < n; ++i) {
    t[i] = Sym01(Lcg(0xB5297A4D3F84D5B5ull + id0 + i)) * 0.05f;
  }
  q.memcpy(dst, t.data(), n * sizeof(float)).wait();
}

/** Read the four digests of a parameter set, summed. */
unsigned long long DigestAll(sycl::queue &q, const float *w1, const float *b1,
                             const float *w2, const float *b2, u64 w1_id0,
                             u64 w1_n, u64 b1_id0, u64 b1_n, u64 w2_id0,
                             u64 w2_n, u64 b2_id0, u64 b2_n) {
  unsigned long long *d = sycl::malloc_device<unsigned long long>(1, q);
  q.memset(d, 0, sizeof(unsigned long long)).wait();
  Digest(q, w1, w1_id0, w1_n, d);
  Digest(q, b1, b1_id0, b1_n, d);
  Digest(q, w2, w2_id0, w2_n, d);
  Digest(q, b2, b2_id0, b2_n, d);
  unsigned long long h = 0;
  q.memcpy(&h, d, sizeof(h)).wait();
  sycl::free(d, q);
  return h;
}

}  // namespace

int main(int argc, char **argv) {
  gvc::Comm comm;
  comm.Init(&argc, &argv);
  const int rank = comm.rank, nranks = comm.nranks;
  sycl::queue &q = comm.q;

  u32 blocks = 8, threads = 256;
  u64 I = 256, H = 4096, O = 64, B = 64, steps = 5, rpp = 16;
  float lr = 0.01f;
  bool no_ref = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--in") I = next();
    else if (a == "--hidden") H = next();
    else if (a == "--out") O = next();
    else if (a == "--batch") B = next();
    else if (a == "--steps") steps = next();
    else if (a == "--rpp") rpp = next();
    else if (a == "--lr" && i + 1 < argc) lr = std::strtof(argv[++i], nullptr);
    else if (a == "--no-ref") no_ref = true;
  }
  if (H % nranks != 0 || O % nranks != 0) {
    if (rank == 0) {
      std::fprintf(stderr, "LBANN %s: H and O must divide the rank count\n",
                   gvc::Comm::Name());
    }
    comm.Finalize();
    return 2;
  }
  const u64 hper = H / nranks, oper = O / nranks;
  const u64 h0 = rank * hper, h1 = h0 + hper;
  const u64 o0 = rank * oper, o1 = o0 + oper;
  const u64 w1_n = H * I, w2_n = O * H;

  if (rank == 0) {
    std::printf("MLP training, %s edition (SYCL): %llu -> %llu -> %llu, "
                "batch=%llu, steps=%llu, %d ranks (model-parallel), "
                "%.1f MB of parameters per rank\n",
                gvc::Comm::Name(), (unsigned long long)I,
                (unsigned long long)H, (unsigned long long)O,
                (unsigned long long)B, (unsigned long long)steps, nranks,
                (hper * I + hper + oper * H + oper) * 4.0 / 1048576.0);
  }

  // Batch and targets, deterministic and replicated.
  std::vector<float> hxv(B * I), hyv(B * O);
  for (u64 i = 0; i < B * I; ++i) hxv[i] = Sym01(Lcg(0xA02BDBF7BB3C0A7ull + i));
  for (u64 i = 0; i < B * O; ++i) hyv[i] = Sym01(Lcg(0x6C62272E07BB0142ull + i));
  float *d_x = sycl::malloc_device<float>(B * I, q);
  float *d_y = sycl::malloc_device<float>(B * O, q);
  q.memcpy(d_x, hxv.data(), B * I * sizeof(float)).wait();
  q.memcpy(d_y, hyv.data(), B * O * sizeof(float)).wait();

  // Shards, seeded by LOGICAL element id so every substrate starts from
  // byte-identical weights.
  float *d_w1 = sycl::malloc_device<float>(hper * I, q);
  float *d_b1 = sycl::malloc_device<float>(hper, q);
  float *d_w2 = sycl::malloc_device<float>(oper * H, q);
  float *d_b2 = sycl::malloc_device<float>(oper, q);
  SeedShard(q, d_w1, h0 * I, hper * I);
  SeedShard(q, d_b1, w1_n + h0, hper);
  SeedShard(q, d_w2, w1_n + H + o0 * H, oper * H);
  SeedShard(q, d_b2, w1_n + H + w2_n + o0, oper);

  // The gathered activations: symmetric under ISHMEM (fcollect targets).
  float *d_a1 = comm.Alloc<float>(H * B);
  float *d_a1_own = comm.Alloc<float>(hper * B);
  float *d_d2 = comm.Alloc<float>(O * B);
  float *d_d2_own = comm.Alloc<float>(oper * B);
  double *d_lp = comm.Alloc<double>(O * B);
  double *d_lp_own = comm.Alloc<double>(oper * B);
  float *d_d1p = comm.Alloc<float>(static_cast<u64>(nranks) * H * B);
  float *d_d1p_own = comm.Alloc<float>(H * B);
  float *d_d1 = sycl::malloc_device<float>(H * B, q);
  std::vector<double> h_lp(O * B);
  std::vector<double> loss(steps);

  comm.Barrier();
  const double t0 = gvc::NowMs();
  double t_comm = 0.0;
  for (u64 s = 0; s < steps; ++s) {
    // fwd1 own rows -> allgather a1. The kernels write the OWN slice into
    // the gathered layout; the substrate gathers from a contiguous own
    // buffer, so copy the slice out and gather it back in rank order.
    Fwd1(q, blocks, threads, d_w1, d_b1, h0, h1, I, B, d_x, d_a1);
    q.memcpy(d_a1_own, d_a1 + h0 * B, hper * B * sizeof(float)).wait();
    double c0 = gvc::NowMs();
    comm.Allgather(d_a1_own, d_a1, hper * B);
    t_comm += gvc::NowMs() - c0;
    // fwd2 own rows -> allgather d2 and the loss parts.
    Fwd2(q, blocks, threads, d_w2, d_b2, o0, o1, H, O, B, d_a1, d_y, d_d2,
         d_lp);
    q.memcpy(d_d2_own, d_d2 + o0 * B, oper * B * sizeof(float));
    q.memcpy(d_lp_own, d_lp + o0 * B, oper * B * sizeof(double));
    q.wait();
    c0 = gvc::NowMs();
    comm.Allgather(d_d2_own, d_d2, oper * B);
    comm.Allgather(d_lp_own, d_lp, oper * B);
    t_comm += gvc::NowMs() - c0;
    q.memcpy(h_lp.data(), d_lp, O * B * sizeof(double)).wait();
    double l = 0.0;
    for (u64 i = 0; i < O * B; ++i) l += h_lp[i];
    loss[s] = l / static_cast<double>(B * O);
    // bwd1: own o-partial -> allgather -> combine IN RANK ORDER.
    Bwd1Partial(q, blocks, threads, d_w2, o0, o1, H, B, d_d2, d_d1p_own, rpp);
    c0 = gvc::NowMs();
    comm.Allgather(d_d1p_own, d_d1p, H * B);
    t_comm += gvc::NowMs() - c0;
    Bwd1Combine(q, blocks, threads, d_d1p, nranks, H, B, d_a1, d_d1);
    // updates on own shards (d2/a1/d1 are full everywhere).
    Upd2(q, blocks, threads, d_w2, d_b2, o0, o1, H, B, d_a1, d_d2, lr);
    Upd1(q, blocks, threads, d_w1, d_b1, h0, h1, I, B, d_x, d_d1, lr);
  }
  comm.Barrier();
  const double ms = gvc::NowMs() - t0;

  // Weight digest over the LOGICAL ids (order-independent integer sum).
  const unsigned long long dg_loc =
      DigestAll(q, d_w1, d_b1, d_w2, d_b2, h0 * I, hper * I, w1_n + h0, hper,
                w1_n + H + o0 * H, oper * H, w1_n + H + w2_n + o0, oper);
  const unsigned long long dg = comm.HostSum(dg_loc);

  int rc = 0;
  if (rank == 0) {
    std::printf("  %llu steps in %.1f ms (comm %.1f ms)\n",
                (unsigned long long)steps, ms, t_comm);
    std::printf("LBANN %s: in=%llu hidden=%llu out=%llu batch=%llu steps=%llu "
                "ranks=%d ms=%.1f comm_ms=%.1f ms_per_step=%.2f "
                "weight_digest=%llu loss_final=%.6f\n",
                gvc::Comm::Name(), (unsigned long long)I,
                (unsigned long long)H, (unsigned long long)O,
                (unsigned long long)B, (unsigned long long)steps, nranks, ms,
                t_comm, ms / steps, dg, loss[steps - 1]);
    if (no_ref) {
      std::printf("  LOSS GATE: skipped (--no-ref); loss %.6f -> %.6f\n",
                  loss[0], loss[steps - 1]);
    } else {
      // Dense in-process reference: the same kernels run single-shard, with
      // the SAME ASSOCIATION as the distributed path (one partial per
      // rank-range, combined in rank order).
      float *r_w1 = sycl::malloc_device<float>(w1_n, q);
      float *r_b1 = sycl::malloc_device<float>(H, q);
      float *r_w2 = sycl::malloc_device<float>(w2_n, q);
      float *r_b2 = sycl::malloc_device<float>(O, q);
      SeedShard(q, r_w1, 0, w1_n);
      SeedShard(q, r_b1, w1_n, H);
      SeedShard(q, r_w2, w1_n + H, w2_n);
      SeedShard(q, r_b2, w1_n + H + w2_n, O);
      std::vector<double> loss_ref(steps);
      for (u64 s = 0; s < steps; ++s) {
        Fwd1(q, blocks, threads, r_w1, r_b1, 0, H, I, B, d_x, d_a1);
        Fwd2(q, blocks, threads, r_w2, r_b2, 0, O, H, O, B, d_a1, d_y, d_d2,
             d_lp);
        q.memcpy(h_lp.data(), d_lp, O * B * sizeof(double)).wait();
        double l = 0.0;
        for (u64 i = 0; i < O * B; ++i) l += h_lp[i];
        loss_ref[s] = l / static_cast<double>(B * O);
        for (int r = 0; r < nranks; ++r) {
          Bwd1Partial(q, blocks, threads, r_w2 + static_cast<u64>(r) * oper * H,
                      static_cast<u64>(r) * oper,
                      static_cast<u64>(r + 1) * oper, H, B, d_d2,
                      d_d1p + static_cast<u64>(r) * H * B, rpp);
        }
        Bwd1Combine(q, blocks, threads, d_d1p, nranks, H, B, d_a1, d_d1);
        Upd2(q, blocks, threads, r_w2, r_b2, 0, O, H, B, d_a1, d_d2, lr);
        Upd1(q, blocks, threads, r_w1, r_b1, 0, H, I, B, d_x, d_d1, lr);
      }
      const unsigned long long dg_ref =
          DigestAll(q, r_w1, r_b1, r_w2, r_b2, 0, w1_n, w1_n, H, w1_n + H,
                    w2_n, w1_n + H + w2_n, O);
      bool ok = true;
      for (u64 s = 0; s < steps; ++s) {
        if (loss[s] != loss_ref[s]) {
          std::printf("  LOSS GATE: step %llu %.17g != %.17g\n",
                      (unsigned long long)s, loss[s], loss_ref[s]);
          ok = false;
        }
      }
      if (ok) {
        std::printf("  LOSS GATE: PASS (all steps bit-equal; %.6f -> %.6f)\n",
                    loss_ref[0], loss_ref[steps - 1]);
      } else {
        rc = 1;
      }
      if (dg != dg_ref) {
        std::printf("  WEIGHT GATE: FAIL (digest %llu != %llu)\n", dg, dg_ref);
        rc = 1;
      } else {
        std::printf("  WEIGHT GATE: PASS (bit-equal to dense reference)\n");
      }
      sycl::free(r_w1, q); sycl::free(r_b1, q);
      sycl::free(r_w2, q); sycl::free(r_b2, q);
    }
    std::printf("LBANN %s: %s\n", gvc::Comm::Name(),
                rc == 0 ? "ALL GATES PASS" : "GATE FAILURE");
  }
  rc = comm.Verdict(rc);
  comm.Free(d_a1); comm.Free(d_a1_own); comm.Free(d_d2); comm.Free(d_d2_own);
  comm.Free(d_lp); comm.Free(d_lp_own); comm.Free(d_d1p); comm.Free(d_d1p_own);
  sycl::free(d_d1, q); sycl::free(d_x, q); sycl::free(d_y, q);
  sycl::free(d_w1, q); sycl::free(d_b1, q); sycl::free(d_w2, q);
  sycl::free(d_b2, q);
  comm.Finalize();
  return rc;
}
