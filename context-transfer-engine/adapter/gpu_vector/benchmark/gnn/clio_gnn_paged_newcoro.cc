#if CTP_ENABLE_SYCL
#define CLIO_SYCL_KERNEL_TU 1
#endif
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
 * DISTRIBUTED GNN: a 2-layer GraphSAGE forward over a paged feature matrix,
 * sharded across nodes by vertex ownership.
 *
 * The other GNN tooling in this directory (the burst bench, gnn_aggregate) is
 * single-node by construction: it reads a prepared dataset from local disk
 * and owns the whole matrix. This one is the multi-node edition. Features are
 * synthesised from a hash so every node, and the single-node reference, agree
 * on the input without shipping a dataset into the containers.
 *
 * THE EXCHANGE. Each node owns a contiguous run of pages and computes the
 * rows in it; a vertex's neighbours are drawn from the WHOLE graph, so about
 * (nodes-1)/nodes of its edges point at a peer's rows. Those rows are read by
 * faulting the peer's page through the shared tag, at a demanded generation
 * (see gnn_kernels.h). Layer 2 reads layer 1's embeddings the same way, so the
 * run exchanges twice: raw features, then hidden state.
 *
 * WHAT IS GATED (by test/distributed_workloads/run_workloads_distributed.sh):
 *   logit_digest   an order-independent integer digest of every logit's bit
 *                  pattern, reduced across nodes. Bit-exact against the
 *                  single-node run at any node count (see DETERMINISM in
 *                  gnn_kernels.h), so the gate is an equality test.
 *   remote_edges   edges whose far end a peer owns. 0 on one node; the
 *                  witness that the decomposition actually crosses nodes.
 *   GNN_NO_REMOTE  negative control: skip peer-owned neighbours. The digest
 *                  must move, or the cross-node reads were not load-bearing.
 * and, independently of node count, every node checks its own logits against
 * a host float reference (ref_err), so a run where all node counts are
 * equally wrong still fails.
 *
 *   clio_gnn_paged_bench [--vertices N] [--features F] [--classes C]
 *                        [--rows-per-page R] [--blocks B] [--slots S]
 *                        [--nodes N --node I]
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

static constexpr u32 kYieldLaneBytes = 1024;

#include "../gv_launch_bounds.h"
#include "gnn_kernels.h"
#include "gnn_launch.h"

namespace gn = clio::gv_bench::gnn;

// HOST DRIVER ONLY BELOW THIS LINE: gv::Vector and the CTE client are
// compiled out of the CUDA device pass.
#if CTP_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

/* =====================================================
 * THE WORKLOAD, on the new coroutine API. Ordinary return
 * types, ordinary locals, one marker per suspending call.
 * The `, clio::co::Ctx &_cy` on each signature and at each
 * call site is appended by clio-coroc, not written here.
 * ===================================================== */
namespace clio::gv_bench::gnn {

/**
 * Seed this block's X pages with FeatureAt and publish them as generation
 * kGenFeatures, which is what a peer's layer-1 fetch demands.
 *
 * @param vec the paged vector
 * @param p   launch parameters
 * @param blk this block's LOGICAL index (yv.Block())
 */
CTP_GPU_FUN CLIO_COROC_INLINE void SeedCoro(gv::DeviceVector<float> vec,
                                          GnnParams p, u32 blk) {
  u64 p0 = 0;
  u64 p1 = 0;
  BlockPages(p, blk, p0, p1);
  for (u64 pg = p0; pg < p1; ++pg) {
    const u64 off = p.xbase + pg * p.epp;
    CO_AWAIT(vec.CoFetch(0, off, p.epp));
    auto h = CO_AWAIT(vec.CoHoldPage(off, p.epp, /*write=*/true));
    for (u64 e = threadIdx.x; e < p.epp; e += blockDim.x) {
      const u64 row = pg * p.rpp + e / p.F;
      h[off + e] = FeatureAt(row, e % p.F, p.F);
    }
    __syncthreads();
    CO_AWAIT(vec.CoBeginFlush(kGenFeatures, off, p.epp));
    vec.UnpinRange(off, p.epp);
  }
  // Peers fault on these pages as soon as they launch layer 1, and so do the
  // other blocks of this node, so the seed must be durable before returning.
  CO_AWAIT(vec.CoEndFlush());
}

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
CTP_GPU_FUN CLIO_COROC_INLINE void LayerCoro(gv::DeviceVector<float> vec,
                                           GnnParams p, u32 blk, u32 layer) {
  u64 p0 = 0;
  u64 p1 = 0;
  BlockPages(p, blk, p0, p1);
  const bool first = (layer == 1);
  const u64 in_base = first ? p.xbase : p.hbase;
  const u64 gen = first ? kGenFeatures : kGenHidden;
  const u32 width = first ? p.F : p.C;
  const float *ws = first ? p.ws1 : p.ws2;
  const float *wn = first ? p.wn1 : p.wn2;
  float *acc = p.scratch + static_cast<u64>(blk) * p.F;
  const u64 row_lo = p.own_lo * p.rpp;
  gv::PageRef<float> self;
  gv::PageRef<float> out;
  gv::PageRef<float> nb;
  for (u64 pg = p0; pg < p1; ++pg) {
    gy::YieldPublishCursor(pg + 1);
    const u64 soff = in_base + pg * p.epp;
    const u64 hoff = p.hbase + pg * p.epp;
    CO_AWAIT(vec.CoFetch(0, soff, p.epp));
    self = CO_AWAIT(vec.CoHoldPage(soff, p.epp, /*write=*/false));
    if (first) {
      CO_AWAIT(vec.CoFetch(0, hoff, p.epp));
      out = CO_AWAIT(vec.CoHoldPage(hoff, p.epp, /*write=*/true));
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
        CO_AWAIT(vec.CoFetch(remote ? gen : 0, noff, p.epp));
        nb = CO_AWAIT(vec.CoHoldPage(noff, p.epp, /*write=*/false));
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
      CO_AWAIT(vec.CoBeginFlush(kGenHidden, hoff, p.epp));
      vec.UnpinRange(hoff, p.epp);
    }
  }
  // Layer 2 -- on this node and on every peer -- reads the rows layer 1 just
  // wrote, so they must be durable before this kernel returns.
  if (first) {
    CO_AWAIT(vec.CoEndFlush());
  }
}

}  // namespace clio::gv_bench::gnn

namespace clio::gv_bench::gnn {

namespace {

__global__ GV_LAUNCH_BOUNDS void SeedKernel(GpuInfo info, DevF32 vec,
                                            GnnParams p, View yv,
                                            StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  __syncthreads();
  CLIO_COROC_RUN(yv, ys, SeedCoro(_cy, vec, p, yv.Block()));
}

__global__ GV_LAUNCH_BOUNDS void LayerKernel(GpuInfo info, DevF32 vec,
                                             GnnParams p, u32 layer, View yv,
                                             StackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  vec.Init(yv.Block());
  __syncthreads();
  CLIO_COROC_RUN(yv, ys, LayerCoro(_cy, vec, p, yv.Block(), layer));
}

}  // namespace

void LaunchSeed(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                const GnnParams &p, View vw, StackView sv) {
  SeedKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, vec, p, vw, sv);
}

void LaunchLayer(dim3 grid, dim3 block, const GpuInfo &info, DevF32 vec,
                 const GnnParams &p, u32 layer, View vw, StackView sv) {
  LayerKernel<<<grid, block, CLIO_YIELD_SMEM_BYTES>>>(info, vec, p, layer, vw,
                                                      sv);
}

}  // namespace clio::gv_bench::gnn

#if !CTP_IS_DEVICE_PASS

// Cross-node reduction; uses the CTE client, so inside the guard.
#include "../bench_dist.h"

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

class YieldRunner {
 public:
  YieldRunner(unsigned nb, unsigned nt)
      : drv_(nb, nt), stack_(nb, nt, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    // Both resets are required: RunToCompletion does not reset, so a reused
    // runner whose driver still reads "done" skips the launch entirely.
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> v) {
          launch(g, b, v, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000, gv::ResumeWhenComplete);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};

/** Command line. */
struct Opts {
  u64 vertices = 8192;
  u32 features = 64;
  u32 classes = 16;
  u64 rows_per_page = 64;
  u32 blocks = 8;
  u32 threads = 256;
  u32 slots = 64;
  u32 nodes = 1;
  u32 node = 0;
};

/**
 * Parse argv into `o`.
 *
 * @return 0 to continue, 1 if --help was printed, 2 on a bad argument
 */
int ParseArgs(int argc, char **argv, Opts &o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> unsigned long long {
      return (i + 1 < argc) ? std::stoull(argv[++i]) : 0ull;
    };
    if (a == "--vertices") o.vertices = next();
    else if (a == "--features") o.features = static_cast<u32>(next());
    else if (a == "--classes") o.classes = static_cast<u32>(next());
    else if (a == "--rows-per-page") o.rows_per_page = next();
    else if (a == "--blocks") o.blocks = static_cast<u32>(next());
    else if (a == "--threads") o.threads = static_cast<u32>(next());
    else if (a == "--slots") o.slots = static_cast<u32>(next());
    else if (a == "--nodes") o.nodes = static_cast<u32>(next());
    else if (a == "--node") o.node = static_cast<u32>(next());
    // --repeat is accepted for deck compatibility with the other paged
    // benches; the forward is run once (its generations are single-use).
    else if (a == "--repeat") (void)next();
    else if (a == "--help") {
      std::printf("usage: %s [--vertices N] [--features F] [--classes C] "
                  "[--rows-per-page R]\n       [--blocks B] [--threads T] "
                  "[--slots S] [--nodes N --node I]\n", argv[0]);
      return 1;
    } else {
      std::fprintf(stderr, "GNN ERROR: unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (o.nodes == 0 || o.node >= o.nodes) {
    std::fprintf(stderr, "GNN ERROR: --node %u out of range for --nodes %u\n",
                 o.node, o.nodes);
    return 2;
  }
  if (o.rows_per_page == 0 || o.vertices % o.rows_per_page != 0) {
    std::fprintf(stderr, "GNN ERROR: --vertices %llu must be a multiple of "
                 "--rows-per-page %llu (a page must hold whole rows of one "
                 "owner)\n", (unsigned long long)o.vertices,
                 (unsigned long long)o.rows_per_page);
    return 2;
  }
  if (o.features == 0 || o.classes == 0 || o.blocks == 0) {
    std::fprintf(stderr, "GNN ERROR: features, classes and blocks must be "
                 "nonzero\n");
    return 2;
  }
  return 0;
}

/**
 * The synthetic graph, identical on every node: vertex i has 3..8 neighbours
 * drawn uniformly from the whole vertex set, so edges ignore ownership.
 *
 * @param n       vertex count
 * @param indptr  out: CSR row pointers (n+1)
 * @param indices out: CSR column indices
 */
void BuildGraph(u64 n, std::vector<u64> &indptr, std::vector<u64> &indices) {
  indptr.assign(n + 1, 0);
  indices.clear();
  for (u64 i = 0; i < n; ++i) {
    const u64 deg = 3 + gn::Mix64(i ^ 0xD1B54A32D192ED03ull) % 6;
    for (u64 e = 0; e < deg; ++e) {
      indices.push_back(gn::Mix64((i << 4) + e + 0xA0761D6478BD642Full) % n);
    }
    indptr[i + 1] = indices.size();
  }
}

/**
 * A seeded weight matrix, rows x cols, scaled so activations stay O(1).
 *
 * @param stream distinct id per matrix so they are independent
 * @param rows   input width
 * @param cols   output width
 * @return row-major weights
 */
std::vector<float> BuildWeights(u64 stream, u32 rows, u32 cols) {
  std::vector<float> w(static_cast<size_t>(rows) * cols);
  const float scale = 1.0f / std::sqrt(static_cast<float>(rows));
  for (size_t k = 0; k < w.size(); ++k) {
    w[k] = gn::FeatureAt(stream, k, w.size()) * scale;
  }
  return w;
}

/** The four weight matrices, in the order GnnParams names them. */
struct Weights {
  std::vector<float> ws1, wn1, ws2, wn2;
};

/**
 * Host float reference of the forward pass for rows [row_lo, row_hi).
 *
 * Needs layer 1 for every row (neighbours of owned rows can be anywhere), so
 * it computes H for the whole graph and logits for the owned rows only.
 *
 * @return row-major logits, (row_hi - row_lo) x C
 */
std::vector<float> HostReference(const Opts &o, const std::vector<u64> &indptr,
                                 const std::vector<u64> &indices,
                                 const Weights &w, u64 row_lo, u64 row_hi) {
  const u64 n = o.vertices;
  const u32 F = o.features, C = o.classes;
  std::vector<float> x(n * F), h(n * F), agg(F);
  for (u64 i = 0; i < n; ++i) {
    for (u32 f = 0; f < F; ++f) x[i * F + f] = gn::FeatureAt(i, f, F);
  }
  auto mean = [&](const std::vector<float> &src, u64 i) {
    std::fill(agg.begin(), agg.end(), 0.0f);
    for (u64 e = indptr[i]; e < indptr[i + 1]; ++e) {
      for (u32 f = 0; f < F; ++f) agg[f] += src[indices[e] * F + f];
    }
    const u64 deg = indptr[i + 1] - indptr[i];
    if (deg != 0) {
      for (u32 f = 0; f < F; ++f) agg[f] /= static_cast<float>(deg);
    }
  };
  for (u64 i = 0; i < n; ++i) {
    mean(x, i);
    for (u32 k = 0; k < F; ++k) {
      float s = 0.0f;
      for (u32 f = 0; f < F; ++f) s += x[i * F + f] * w.ws1[f * F + k];
      for (u32 f = 0; f < F; ++f) s += agg[f] * w.wn1[f * F + k];
      h[i * F + k] = s > 0.0f ? s : 0.0f;
    }
  }
  std::vector<float> out((row_hi - row_lo) * C);
  for (u64 i = row_lo; i < row_hi; ++i) {
    mean(h, i);
    for (u32 c = 0; c < C; ++c) {
      float s = 0.0f;
      for (u32 f = 0; f < F; ++f) s += h[i * F + f] * w.ws2[f * C + c];
      for (u32 f = 0; f < F; ++f) s += agg[f] * w.wn2[f * C + c];
      out[(i - row_lo) * C + c] = s;
    }
  }
  return out;
}

/**
 * Order-independent digest of a logit block: a wrapping sum of a hash of
 * (global element index, bit pattern). Integer addition commutes, so the
 * per-node partials reduce to exactly the single-node value.
 *
 * @param out     row-major logits for rows starting at row_lo
 * @param row_lo  global id of the first row
 * @param C       classes per row
 */
unsigned long long Digest(const std::vector<float> &out, u64 row_lo, u32 C) {
  unsigned long long d = 0;
  for (size_t k = 0; k < out.size(); ++k) {
    u32 bits = 0;
    std::memcpy(&bits, &out[k], sizeof(bits));
    const u64 idx = row_lo * C + k;
    d += gn::Mix64((idx << 32) ^ bits);
  }
  return d;
}

/**
 * Write a single-host runtime config, UNLESS the caller already exported
 * CLIO_SERVER_CONF. A harness-supplied cluster config must win: overwriting
 * it would stand each node up as its own single-host runtime on one port.
 *
 * @param data_mb logical size of the vector, to size the RAM tier
 */
void EnsureServerConf(double data_mb) {
  if (getenv("CLIO_SERVER_CONF") != nullptr) {
    std::printf("  runtime: using CLIO_SERVER_CONF=%s (not writing one)\n",
                getenv("CLIO_SERVER_CONF"));
    return;
  }
  const unsigned long long ram_mb =
      static_cast<unsigned long long>(data_mb) + 512ull;
  std::ofstream cfg("gv_gnn_bench.yaml");
  cfg << "networking:\n  port: 9445\n\n"
      << "runtime:\n  num_threads: 8\n  queue_depth: 8192\n"
      << "  first_busy_wait: 10000000\n\n"
      << "gpu:\n  queue_depth: 8192\n\n"
      << "compose:\n"
      << "  - mod_name: clio_bdev\n    pool_name: \"ram::chi_default_bdev\"\n"
      << "    pool_query: local\n    pool_id: \"301.0\"\n"
      << "    bdev_type: ram\n    capacity: \"1GB\"\n\n"
      << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n"
      << "    pool_query: local\n    pool_id: \"512.0\"\n    storage:\n"
      << "      - path: \"ram::gv_gnn_ram\"\n        bdev_type: \"ram\"\n"
      << "        capacity_limit: \"" << ram_mb << "MB\"\n"
      << "        score: 0.5\n"
      << "    dpe:\n      dpe_type: \"max_bw\"\n";
  cfg.close();
  ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gv_gnn_bench.yaml", 1);
}

/** Copy a host vector to a fresh device allocation. */
template <typename T>
T *ToDevice(const std::vector<T> &h) {
  T *d = ctp::GpuApi::Malloc<T>(h.size() * sizeof(T));
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(d),
                      reinterpret_cast<const char *>(h.data()),
                      h.size() * sizeof(T));
  return d;
}

/**
 * Compare device logits against the host reference.
 *
 * @return the largest |got - want| / (1 + |want|)
 */
double MaxRelErr(const std::vector<float> &got, const std::vector<float> &want) {
  double worst = 0.0;
  for (size_t k = 0; k < got.size(); ++k) {
    const double d = std::fabs(static_cast<double>(got[k]) - want[k]);
    const double r = d / (1.0 + std::fabs(static_cast<double>(want[k])));
    if (!(r <= worst)) worst = r;  // also catches NaN
  }
  return worst;
}

/** Edges of rows [row_lo, row_hi) whose far end is outside that range. */
unsigned long long RemoteEdges(const std::vector<u64> &indptr,
                               const std::vector<u64> &indices, u64 row_lo,
                               u64 row_hi) {
  unsigned long long r = 0;
  for (u64 e = indptr[row_lo]; e < indptr[row_hi]; ++e) {
    if (indices[e] < row_lo || indices[e] >= row_hi) ++r;
  }
  return r;
}

/** The problem every node agrees on, plus this node's share of it. */
struct Problem {
  Opts o;
  bool no_remote = false;
  u64 F = 0, C = 0, rpp = 0;
  u64 npages = 0;       /**< pages per region */
  u64 epp = 0;          /**< elements per page */
  u64 page_bytes = 0;
  u64 n_elems = 0;      /**< regions X and H */
  double data_mb = 0.0;
  clio_bench_dist::Shard own;
  u64 row_lo = 0, row_hi = 0;
  std::vector<u64> indptr, indices;
  Weights w;
};

/**
 * Parse the command line and derive the shared problem.
 *
 * @return 0 to run, -1 if --help was printed, otherwise an exit code
 */
int MakeProblem(int argc, char **argv, Problem &pr) {
  if (const int rc = ParseArgs(argc, argv, pr.o); rc != 0) {
    return rc == 1 ? -1 : rc;
  }
  const char *const no_remote_env = getenv("GNN_NO_REMOTE");
  pr.no_remote = no_remote_env != nullptr && no_remote_env[0] != '\0';
  const Opts &o = pr.o;
  pr.F = o.features;
  pr.C = o.classes;
  pr.rpp = o.rows_per_page;
  pr.npages = o.vertices / pr.rpp;
  pr.epp = pr.rpp * pr.F;
  pr.page_bytes = pr.epp * sizeof(float);
  pr.n_elems = 2 * pr.npages * pr.epp;
  pr.data_mb =
      static_cast<double>(pr.n_elems * sizeof(float)) / (1024.0 * 1024.0);
  pr.own = clio_bench_dist::ShardOf(pr.npages, o.node, o.nodes);
  if (pr.own.count() == 0) {
    std::fprintf(stderr, "GNN ERROR: --nodes %u leaves node %u no pages of "
                 "%llu\n", o.nodes, o.node, (unsigned long long)pr.npages);
    return 2;
  }
  pr.row_lo = pr.own.begin * pr.rpp;
  pr.row_hi = pr.own.end * pr.rpp;
  BuildGraph(o.vertices, pr.indptr, pr.indices);
  const u32 F = o.features, C = o.classes;
  pr.w = Weights{BuildWeights(1ull << 40, F, F), BuildWeights(2ull << 40, F, F),
                 BuildWeights(3ull << 40, F, C), BuildWeights(4ull << 40, F, C)};
  std::printf("GNN (GraphSAGE-mean, 2 layers) over a GPU vector\n"
              "  N=%llu F=%llu C=%llu edges=%zu rows/page=%llu pages=%llu x 2 "
              "regions (%lluKB pages, %.1fMB)\n"
              "  node %u of %u owns pages [%llu,%llu) blocks=%u slots=%u%s\n",
              (unsigned long long)o.vertices, (unsigned long long)pr.F,
              (unsigned long long)pr.C, pr.indices.size(),
              (unsigned long long)pr.rpp, (unsigned long long)pr.npages,
              (unsigned long long)(pr.page_bytes >> 10), pr.data_mb, o.node,
              o.nodes, (unsigned long long)pr.own.begin,
              (unsigned long long)pr.own.end, o.blocks, o.slots,
              pr.no_remote ? "  [GNN_NO_REMOTE: peer neighbours dropped]" : "");
  return 0;
}

/**
 * Bring up the runtime and CTE client, and (multi-node only) the tag the
 * digest reduction publishes its partials under.
 *
 * @param pr      the problem
 * @param cte_red out: reduction client, left null on one node
 * @param red_tag out: reduction tag
 * @return true on success
 */
bool InitRuntime(const Problem &pr,
                 std::unique_ptr<clio::cte::core::Client> &cte_red,
                 clio::cte::core::TagId &red_tag) {
  EnsureServerConf(pr.data_mb);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "GNN ERROR: runtime init failed\n");
    return false;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "GNN ERROR: cte client init failed\n");
    return false;
  }
  if (pr.o.nodes <= 1) return true;
  cte_red = std::make_unique<clio::cte::core::Client>(
      clio::cte::core::kCtePoolId);
  auto t = cte_red->AsyncGetOrCreateTag("gv_gnn_red");
  t.Wait();
  if (t->GetReturnCode() != 0) {
    std::fprintf(stderr, "GNN ERROR: could not create reduction tag\n");
    return false;
  }
  red_tag = t->tag_id_;
  return true;
}

/** Upload the graph and weights and allocate the device buffers. */
gn::GnnParams UploadParams(const Problem &pr) {
  gn::GnnParams p{};
  p.xbase = 0;
  p.hbase = pr.npages * pr.epp;
  p.epp = pr.epp;
  p.rpp = pr.rpp;
  p.own_lo = pr.own.begin;
  p.own_hi = pr.own.end;
  p.pages_per_block = (pr.own.count() + pr.o.blocks - 1) / pr.o.blocks;
  p.F = static_cast<u32>(pr.F);
  p.C = static_cast<u32>(pr.C);
  p.no_remote = pr.no_remote ? 1u : 0u;
  p.indptr = ToDevice(pr.indptr);
  p.indices = ToDevice(pr.indices);
  p.ws1 = ToDevice(pr.w.ws1);
  p.wn1 = ToDevice(pr.w.wn1);
  p.ws2 = ToDevice(pr.w.ws2);
  p.wn2 = ToDevice(pr.w.wn2);
  p.scratch = ctp::GpuApi::Malloc<float>(pr.o.blocks * pr.F * sizeof(float));
  p.out = ctp::GpuApi::Malloc<float>((pr.row_hi - pr.row_lo) * pr.C *
                                     sizeof(float));
  return p;
}

/** Release everything UploadParams allocated. */
void FreeParams(gn::GnnParams &p) {
  ctp::GpuApi::Free(const_cast<u64 *>(p.indptr));
  ctp::GpuApi::Free(const_cast<u64 *>(p.indices));
  ctp::GpuApi::Free(const_cast<float *>(p.ws1));
  ctp::GpuApi::Free(const_cast<float *>(p.wn1));
  ctp::GpuApi::Free(const_cast<float *>(p.ws2));
  ctp::GpuApi::Free(const_cast<float *>(p.wn2));
  ctp::GpuApi::Free(p.scratch);
  ctp::GpuApi::Free(p.out);
}

}  // namespace

int main(int argc, char **argv) {
  Problem pr;
  if (const int rc = MakeProblem(argc, argv, pr); rc != 0) {
    return rc < 0 ? 0 : rc;
  }
  const Opts &o = pr.o;
  std::unique_ptr<clio::cte::core::Client> cte_red;
  clio::cte::core::TagId red_tag{};
  if (!InitRuntime(pr, cte_red, red_tag)) return 1;
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // ONE SHARED TAG across the nodes: a peer's page is read by faulting it,
  // which only works if every node names the same blobs.
  gv::Vector<float> vec("gv_gnn", {0}, pr.page_bytes, o.blocks,
                        o.slots < 24u ? 24u : o.slots, pr.n_elems);
  vec.EnableStats();
  auto dev = vec.GetDevice(0);
  YieldRunner runner(o.blocks, o.threads);
  gn::GnnParams p = UploadParams(pr);

  runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
    gn::LaunchSeed(g, b, gpu, dev, p, vw, sv);
  });
  ctp::GpuApi::Synchronize();
  vec.ResetStats();
  const double t0 = NowMs();
  for (u32 layer = 1; layer <= 2; ++layer) {
    runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                   gy::YieldStackView sv) {
      gn::LaunchLayer(g, b, gpu, dev, p, layer, vw, sv);
    });
    ctp::GpuApi::Synchronize();
  }
  const double ms = NowMs() - t0;

  std::vector<float> got((pr.row_hi - pr.row_lo) * pr.C);
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(got.data()),
                      reinterpret_cast<const char *>(p.out),
                      got.size() * sizeof(float));
  unsigned long long red[2] = {
      Digest(got, pr.row_lo, static_cast<u32>(pr.C)),
      RemoteEdges(pr.indptr, pr.indices, pr.row_lo, pr.row_hi)};
  // The control deliberately computes a different function, so the reference
  // (which models the real one) does not apply to it.
  double ref_err = 0.0;
  if (!pr.no_remote) {
    ref_err = MaxRelErr(got, HostReference(o, pr.indptr, pr.indices, pr.w,
                                           pr.row_lo, pr.row_hi));
  }
  if (o.nodes > 1 &&
      !clio_bench_dist::ReduceSumU64(*cte_red, red_tag, o.node, o.nodes, 0,
                                     red, 2, "gnnred")) {
    std::fprintf(stderr, "GNN ERROR: digest reduction failed\n");
    return 1;
  }
  const auto st = vec.ReadStats(0);
  const bool ref_ok = ref_err <= 1e-4;
  std::fprintf(stderr,
               "GNN mode=paged nodes=%u node=%u N=%llu F=%llu C=%llu "
               "rpp=%llu pages=%llu blocks=%u slots=%u ms=%.1f "
               "logit_digest=%llu remote_edges=%llu ref_err=%.3g ref=%s "
               "faults=%llu evicts=%llu get_errors=%llu\n",
               o.nodes, o.node, (unsigned long long)o.vertices,
               (unsigned long long)pr.F, (unsigned long long)pr.C,
               (unsigned long long)pr.rpp, (unsigned long long)pr.npages,
               o.blocks, o.slots, ms, red[0], red[1], ref_err,
               pr.no_remote ? "skipped" : (ref_ok ? "PASS" : "FAIL"),
               (unsigned long long)st.faults, (unsigned long long)st.evicts,
               (unsigned long long)st.get_errors);

  FreeParams(p);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return (pr.no_remote || ref_ok) ? 0 : 1;
}

#endif  // !CTP_IS_DEVICE_PASS
