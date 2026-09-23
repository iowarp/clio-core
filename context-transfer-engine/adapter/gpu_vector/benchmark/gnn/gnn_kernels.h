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
 * Distributed GraphSAGE forward over a paged feature matrix: the device code,
 * one copy.
 *
 * THE MODEL. Two layers of GraphSAGE-mean with fixed, seeded weights:
 *
 *   agg1[i]  = mean_{j in N(i)} x[j]
 *   h1[i]    = ReLU( x[i] @ Ws1 + agg1[i] @ Wn1 )          [F -> F]
 *   agg2[i]  = mean_{j in N(i)} h1[j]
 *   out[i]   = h1[i] @ Ws2 + agg2[i] @ Wn2                 [F -> C]
 *
 * THE DECOMPOSITION. The vector holds two regions of P pages each -- X (the
 * input features) and H (the layer-1 embeddings) -- with rpp rows of F floats
 * per page. Each node OWNS a contiguous run of pages [own_lo, own_hi) in both
 * regions: it seeds those X pages, writes those H pages and produces logits
 * for their rows. A neighbour j of an owned vertex can live on ANY node, and
 * that is the whole point: the vector's pages are CTE blobs in one shared tag,
 * so faulting a peer's page IS the cross-node exchange. There is no separate
 * halo step.
 *
 * WHY GENERATIONS, AND ONLY ON PEER PAGES. Nothing orders node A's seed before
 * node B's first read of it, so a peer page is fetched with a DEMANDED
 * generation: X is published as generation 1 by SeedCoro and H as generation
 * 2 by layer 1 of LayerCoro, and a fetch naming generation G polls until the publish
 * lands -- the generation is the barrier. Own pages take generation 0 ("any
 * version"): a page's generation is stamped by the fetch that delivers it, so
 * a page this node wrote and never re-fetched sits at 0 forever, and
 * demanding one of it stalls the block ("gen stall: page N at gen 0 want G").
 * Own pages are current by construction -- the kernel that wrote them ended
 * with EndFlush before the next one launched.
 *
 * DETERMINISM. One thread owns one output element and sums its terms in a
 * fixed order (edge order, then feature order); nothing uses atomics. So an
 * element's value does not depend on which node or block computed it, and a
 * run at any node count is BIT-IDENTICAL to the single-node one. That is what
 * lets the distributed gate be an equality test on an integer digest rather
 * than a tolerance.
 */
#ifndef CLIO_GV_BENCH_GNN_KERNELS_H_
#define CLIO_GV_BENCH_GNN_KERNELS_H_

#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

namespace clio::gv_bench::gnn {

namespace gv = ::clio::cte::gpu_vector;
namespace gy = ::clio::run::gpu;
using ::clio::run::u32;
using ::clio::run::u64;

/** Generation SeedCoro publishes X as, and layer 1 demands of peer X. */
constexpr u64 kGenFeatures = 1;
/** Generation layer 1 publishes H as, and layer 2 demands of peer H. */
constexpr u64 kGenHidden = 2;

/**
 * Everything a GNN launch needs besides the vector, as one POD.
 *
 * Trivially copyable on purpose: it crosses every suspend point of every
 * coroutine below, and the clio-coroc lowering byte-copies such values into
 * the frame.
 */
struct GnnParams {
  u64 xbase;           /**< element offset of region X */
  u64 hbase;           /**< element offset of region H */
  u64 epp;             /**< elements per page (= rpp * F) */
  u64 rpp;             /**< rows per page */
  u64 own_lo;          /**< first page this node owns */
  u64 own_hi;          /**< one past the last page this node owns */
  u64 pages_per_block; /**< owned pages each block processes */
  u32 F;               /**< feature width (= hidden width) */
  u32 C;               /**< classes (layer-2 output width) */
  u32 no_remote;       /**< negative control: drop peer-owned neighbours */
  u32 pad_;
  const u64 *indptr;   /**< CSR row pointers, N+1 entries */
  const u64 *indices;  /**< CSR column indices */
  const float *ws1;    /**< F x F, row-major [f][k] */
  const float *wn1;    /**< F x F */
  const float *ws2;    /**< F x C */
  const float *wn2;    /**< F x C */
  float *scratch;      /**< nblocks x F neighbour-mean accumulators */
  float *out;          /**< owned rows x C logits */
};

/**
 * SplitMix64 finaliser: the one hash both the host and the device use, so the
 * features the device seeds and the ones the host reference recomputes are
 * the same numbers.
 *
 * @param z value to mix
 * @return a well-mixed 64-bit value
 */
CTP_CROSS_FUN inline u64 Mix64(u64 z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/**
 * Feature x[row][f], in [-1, 1). 24 random bits scaled by a power of two, so
 * the value is exact in float and identical on host and device.
 *
 * @param row vertex id
 * @param f   feature column
 * @param F   feature width
 * @return the feature value
 */
CTP_CROSS_FUN inline float FeatureAt(u64 row, u64 f, u64 F) {
  const u64 bits = (Mix64(row * F + f) >> 40) & 0xFFFFFFull;
  return static_cast<float>(bits) * (2.0f / 16777216.0f) - 1.0f;
}

/** Block `blk`'s owned page range [p0, p1); empty when it has none. */
CTP_GPU_FUN inline void BlockPages(const GnnParams &p, u32 blk, u64 &p0,
                                   u64 &p1) {
  p0 = p.own_lo + static_cast<u64>(blk) * p.pages_per_block;
  p1 = p0 + p.pages_per_block;
  if (p1 > p.own_hi) p1 = p.own_hi;
  if (p0 > p1) p0 = p1;
}

/** Whether page `pg` is owned by a peer node rather than this one. */
CTP_GPU_FUN inline bool IsRemote(const GnnParams &p, u64 pg) {
  return pg < p.own_lo || pg >= p.own_hi;
}

/**
 * Seed this block's X pages with FeatureAt and publish them as generation
 * kGenFeatures, which is what a peer's layer-1 fetch demands.
 *
 * @param vec the paged vector
 * @param p   launch parameters
 * @param blk this block's LOGICAL index (yv.Block())
 */
#if CLIO_HAS_YCORO
CTP_GPU_FUN inline gy::YCoroMain SeedCoro(gv::DeviceVector<float> vec,
                                          GnnParams p, u32 blk) {
  u64 p0 = 0, p1 = 0;
  BlockPages(p, blk, p0, p1);
  for (u64 pg = p0; pg < p1; ++pg) {
    const u64 off = p.xbase + pg * p.epp;
    co_await vec.Fetch(0, off, p.epp);
    auto h = co_await vec.HoldPage(off, p.epp, /*write=*/true);
    for (u64 e = threadIdx.x; e < p.epp; e += blockDim.x) {
      const u64 row = pg * p.rpp + e / p.F;
      h[off + e] = FeatureAt(row, e % p.F, p.F);
    }
    __syncthreads();
    co_await vec.BeginFlush(kGenFeatures, off, p.epp);
    vec.UnpinRange(off, p.epp);
  }
  // Peers fault on these pages as soon as they launch layer 1, and so do the
  // other blocks of this node, so the seed must be durable before returning.
  co_await vec.EndFlush();
}
#endif  // CLIO_HAS_YCORO

/**
 * One GraphSAGE layer over this block's owned pages.
 *
 *   layer 1:  H[i]   = ReLU( X[i] Ws1 + mean_{j in N(i)} X[j] Wn1 )
 *             written into region H, published as generation kGenHidden
 *   layer 2:  out[i] = H[i] Ws2 + mean_{j in N(i)} H[j] Wn2
 *             written into the plain device buffer p.out (owned rows only)
 *
 * ONE COROUTINE FOR BOTH LAYERS, and the neighbour loop is inline rather than
 * a helper coroutine: the two layers differ only in which region they read,
 * where they write and whether they clamp, and a nested user coroutine would
 * add a yield-stack level none of the other paged benchmarks use (and one the
 * clio-coroc porter does not know how to rewrite).
 *
 * Under the negative control a peer-owned neighbour is skipped but still
 * counted in the divisor, so dropping the cross-node reads MUST change the
 * answer.
 *
 * @param vec   the paged vector
 * @param p     launch parameters
 * @param blk   this block's LOGICAL index (yv.Block())
 * @param layer 1 or 2
 */
#if CLIO_HAS_YCORO
CTP_GPU_FUN inline gy::YCoroMain LayerCoro(gv::DeviceVector<float> vec,
                                           GnnParams p, u32 blk, u32 layer) {
  u64 p0 = 0, p1 = 0;
  BlockPages(p, blk, p0, p1);
  const bool first = (layer == 1);
  const u64 in_base = first ? p.xbase : p.hbase;
  const u64 gen = first ? kGenFeatures : kGenHidden;
  const u32 width = first ? p.F : p.C;
  const float *ws = first ? p.ws1 : p.ws2;
  const float *wn = first ? p.wn1 : p.wn2;
  float *acc = p.scratch + static_cast<u64>(blk) * p.F;
  const u64 row_lo = p.own_lo * p.rpp;
  gv::Held<float> self, out, nb;
  for (u64 pg = p0; pg < p1; ++pg) {
    gy::YieldPublishCursor(pg + 1);
    const u64 soff = in_base + pg * p.epp;
    const u64 hoff = p.hbase + pg * p.epp;
    co_await vec.Fetch(0, soff, p.epp);
    self = co_await vec.HoldPage(soff, p.epp);
    if (first) {
      co_await vec.Fetch(0, hoff, p.epp);
      out = co_await vec.HoldPage(hoff, p.epp, /*write=*/true);
    }
    for (u64 r = 0; r < p.rpp; ++r) {
      const u64 i = pg * p.rpp + r;
      for (u32 f = threadIdx.x; f < p.F; f += blockDim.x) acc[f] = 0.0f;
      __syncthreads();
      // ---- neighbour sum, in edge order --------------------------------
      const u64 e0 = p.indptr[i];
      const u64 e1 = p.indptr[i + 1];
      for (u64 e = e0; e < e1; ++e) {
        const u64 j = p.indices[e];
        const u64 jp = j / p.rpp;
        const bool remote = IsRemote(p, jp);
        if (remote && p.no_remote != 0) continue;
        const u64 noff = in_base + jp * p.epp;
        co_await vec.Fetch(remote ? gen : 0, noff, p.epp);
        nb = co_await vec.HoldPage(noff, p.epp);
        for (u32 f = threadIdx.x; f < p.F; f += blockDim.x) {
          acc[f] += nb[in_base + j * p.F + f];
        }
        __syncthreads();
        nb = {};
        vec.UnpinRange(noff, p.epp);
      }
      if (e1 > e0) {
        const float deg = static_cast<float>(e1 - e0);
        for (u32 f = threadIdx.x; f < p.F; f += blockDim.x) {
          acc[f] = acc[f] / deg;
        }
      }
      __syncthreads();
      // ---- combine: one thread per output column, fixed order ----------
      for (u32 k = threadIdx.x; k < width; k += blockDim.x) {
        float s = 0.0f;
        for (u32 f = 0; f < p.F; ++f) {
          s += self[in_base + i * p.F + f] * ws[f * width + k];
        }
        for (u32 f = 0; f < p.F; ++f) {
          s += acc[f] * wn[f * width + k];
        }
        if (first) {
          out[p.hbase + i * p.F + k] = s > 0.0f ? s : 0.0f;
        } else {
          p.out[(i - row_lo) * p.C + k] = s;
        }
      }
      // The next row zeroes `acc`, which this row's combine is still reading.
      __syncthreads();
    }
    self = {};
    vec.UnpinRange(soff, p.epp);
    if (first) {
      out = {};
      co_await vec.BeginFlush(kGenHidden, hoff, p.epp);
      vec.UnpinRange(hoff, p.epp);
    }
  }
  // Layer 2 -- on this node and on every peer -- reads the rows layer 1 just
  // wrote, so they must be durable before this kernel returns.
  if (first) {
    co_await vec.EndFlush();
  }
}
#endif  // CLIO_HAS_YCORO

}  // namespace clio::gv_bench::gnn

#endif  // CLIO_GV_BENCH_GNN_KERNELS_H_
