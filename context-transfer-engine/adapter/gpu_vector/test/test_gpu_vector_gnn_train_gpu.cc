/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GNN TRAINING over a compressed / streamed feature matrix (Eternia).
 *
 * Model: SIGN-style 1-hop GraphSAGE -- the node features are mean-aggregated over
 * each node's 1-hop neighbourhood once (gnn_agg.py -> agg_features.f32), then a
 * trainable 2-layer MLP classifier (F -> H -> C) is learned by FULL-BATCH gradient
 * descent with softmax cross-entropy over the labelled nodes. The aggregated
 * feature matrix A (N x F) is the large object; Eternia stores it losslessly
 * (zstd) and streams it a window at a time each epoch.
 *
 * Why full-batch: the gradient is a SUM over all nodes, so it can be accumulated
 * WHILE streaming A window-by-window -- resident memory stays O(window + weights),
 * independent of N. One weight update per epoch. Because the per-window grad kernel
 * uses one thread per weight (summing its window's nodes in order) and windows are
 * issued in order on one stream, accumulation is deterministic -> given lossless
 * (bit-identical) features, the streamed-Eternia run and an in-core resident run
 * produce BIT-IDENTICAL weights, loss and accuracy every epoch.
 *
 * Two methods, same training:
 *   - IN-CORE: A resident in HBM (peak GPU = whole A). OOMs once A > HBM.
 *   - ETERNIA: A stored zstd-compressed, streamed through a bounded window
 *     (peak GPU = window). Runs at any A size.
 * Real A (ogbn-arxiv/products aggregated), tiled to reach a requested size.
 *
 * Env: CLIO_GNN_DATA, CLIO_GNN_TRAIN_NODES (tiled node count; 0=native),
 *      CLIO_GNN_EPOCHS (default 30), CLIO_GNN_LR (default 0.5),
 *      CLIO_GNN_HIDDEN (default 64), CLIO_GNN_PAGE_ROWS, CLIO_GNN_WINDOW,
 *      CLIO_GNN_HBM_MIB / CLIO_GNN_DRAM_MIB, CLIO_GNN_CSV.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>

#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_ctp/util/gpu_api.h>

#include "gnn_dataset.h"
#include "../gnn/gnn_aggregate_impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "../benchmark/bench_memcpy_probe.h"

using namespace std::chrono_literals;

namespace {

bool g_initialized = false;
// Phase timers for CLIO_GNN_KTIME (see the epoch loop).
bool g_ktime = false;
double t_gather = 0, t_fwd = 0, t_gw1 = 0, t_gw2 = 0;
inline clio::run::PoolId StoragePool() { return clio::run::PoolId(600, 0); }

constexpr int kMaxH = 256;
// ogbn-papers100M has 172 classes, so this must exceed 64 (arxiv 40 / products 47).
constexpr int kMaxC = 192;
// Every kValStride-th node is held out as validation: excluded from the loss and
// from the gradient, scored separately. Deterministic (index-based), so it does
// not disturb the bit-exactness of the in-core vs streamed comparison.
constexpr unsigned long long kValStride = 10;

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

/** Wire ids whose codecs are error-bounded rather than exact. */
inline bool IsLossyCodec(int wire_id) {
  return wire_id == 7 || wire_id == 8 || wire_id == 18 || wire_id == 20;
}

void EnsureInit() {
  // An externally supplied CLIO_SERVER_CONF WINS. This test composes its own
  // tiers (2 GiB HBM + 20 GiB RAM, no NVMe) and used to setenv over whatever
  // the caller had set, which meant a papers100M run configured with a RAM
  // tier plus a 128 GiB NVMe tier silently got the test's 22 GiB of RAM tiers
  // instead -- so nothing ever spilled to disk, the store grew in memory, and
  // the process was heading for an OOM at ~26% of the matrix. The defaults
  // below are for standalone ctest runs, not for callers who know better.
  if (const char *ext = std::getenv("CLIO_SERVER_CONF")) {
    std::ifstream probe(ext);
    if (probe.good()) {
      std::fprintf(stderr, "[TRAIN] using caller's CLIO_SERVER_CONF=%s\n", ext);
      if (g_initialized) return;
      REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
      std::this_thread::sleep_for(1s);
      g_initialized = true;
      return;
    }
  }
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10602;
  std::string cfg = "/tmp/gpu_vec_gnn_train_" + std::to_string(port) + ".yaml";
  int hbm_mib = std::getenv("CLIO_GNN_HBM_MIB") ? std::atoi(std::getenv("CLIO_GNN_HBM_MIB")) : 2048;
  int dram_mib = std::getenv("CLIO_GNN_DRAM_MIB") ? std::atoi(std::getenv("CLIO_GNN_DRAM_MIB")) : 20480;
  std::string dram_kind = std::getenv("CLIO_GNN_DRAM_TYPE")
                              ? std::getenv("CLIO_GNN_DRAM_TYPE")
                              : std::string("ram");
  int nvme_mib = std::getenv("CLIO_GNN_NVME_MIB") ? std::atoi(std::getenv("CLIO_GNN_NVME_MIB")) : 0;
  std::string nvme_path = std::getenv("CLIO_GNN_NVME_PATH")
                              ? std::getenv("CLIO_GNN_NVME_PATH")
                              : std::string("/tmp/gnn_nvme_tier_") +
                                    std::to_string(port) + ".dat";
  {
    std::ofstream f(cfg);
    f << "networking:\n  port: " << port << "\n\n"
      << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
      << "compose:\n"
      << "  - mod_name: clio_bdev\n    pool_name: \"hbm::chi_default_bdev\"\n"
      << "    pool_query: local\n    pool_id: \"301.0\"\n    bdev_type: hbm\n    capacity: \"128MB\"\n\n"
      << "  - mod_name: clio_cte_compressor\n    pool_name: cte_compressor\n    pool_query: local\n"
      << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
      << "  - mod_name: clio_cte_core\n    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
      << "    storage:\n"
      // SCORE IS "PREFER LOWER", AND THE COMPARISON IS <= THE BLOB SCORE.
      // MaxBwDpe splits targets into low_score (target_score_ <= blob_score)
      // = preferred and the rest = fallback, then ranks WITHIN a group by
      // bandwidth. Pages are put with score 0.5f, so a tier scored exactly
      // 0.5 lands in the SAME preferred group as kHBM and can outrank it on
      // bandwidth -- measured as "TIER SPLIT: kHBM 0MiB, DRAM 8192MiB" with a
      // healthy 5120 MiB HBM tier sitting empty. Every tier below kHBM must
      // therefore be STRICTLY above the blob score.
      //
      // MaxBwDpe partitions targets into
      // low_score (target_score_ <= blob_score) = PREFERRED and high_score =
      // fallback, so the fast tier needs the LOW number. This was written the
      // other way round (hbm 1.0, ram 0.5), which put kHBM permanently in the
      // fallback group: measured "TIER SPLIT: kHBM 0MiB, DRAM 2MiB" -- a
      // configured, scored, adequately sized HBM tier that received nothing.
      // That in turn disabled the device-to-device page fault path, because
      // the batched decoder only takes it for blobs that live on a device
      // tier, so every fault fell back to host staging.
      << "      - path: \"hbm::cte_hbm_tier\"\n        bdev_type: \"hbm\"\n        capacity_limit: \""
      << hbm_mib << "MB\"\n        score: 0.0\n"
      // The host tier's TYPE matters for a fair comparison. "ram" is pageable
      // host memory and the spill path measured 8.9 GB/s on it, against ~25
      // GB/s for PCIe 4.0 x16 -- so an uncompressed run that overflows to it
      // is penalised by the allocation type, not only by not fitting. "pinned"
      // (cudaMallocHost) is the honest opponent; CLIO_GNN_DRAM_TYPE selects it.
      << "      - path: \"" << dram_kind << "::cte_dram_tier\"\n        bdev_type: \""
      << dram_kind << "\"\n        capacity_limit: \""
      << dram_mib << "MB\"\n        score: 0.6\n";
    // OPTIONAL NVMe TIER, identical in both the raw and the compressed run.
    // Without it, overflow lands in host DRAM, and on this machine that is
    // FREE: spilling 3 GiB of a 4 GiB matrix cost 4.4% (12.25s resident vs
    // 12.79s at 75% spilled), because the fetch hides behind the gather. A
    // compressed run cannot beat a free spill -- it pays real decode work to
    // avoid a cost that is not there. Compression only pays when the overflow
    // goes somewhere genuinely slower, which is also the realistic case.
    if (nvme_mib > 0) {
      f << "      - path: \"" << nvme_path << "\"\n        bdev_type: \"file\"\n"
        << "        capacity_limit: \"" << nvme_mib << "MB\"\n        score: 1.0\n";
    }
    f
      << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

void ReadMeta(const std::string &dir, clio::run::u64 &N, int &F, clio::run::u64 &E, int &C) {
  std::ifstream f(dir + "/meta.txt"); REQUIRE(f.good());
  long long n, e; int ff, cc; f >> n >> ff >> e >> cc;
  N = (clio::run::u64)n; F = ff; E = (clio::run::u64)e; C = cc;
}

}  // namespace

/** Per-lane yield frame. Coroutine frames are compiler-laid-out and hold the
 *  whole live state plus resume/destroy pointers, so they need an order of
 *  magnitude more than the hand-packed macro frames (a few u64s) ever did. */
static constexpr clio::run::u32 kYieldLaneBytes = 4096;

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

/** Copy one held run into the window scratch. The caller has just held
 *  (lo+i, n-i), so the run lives entirely on the held page. */
CTP_GPU_FUN void GnnCopyRun(const float *src, clio::run::u64 run,
                            float *dst) {
  // BULK COPY off the guard's base pointer, not element-by-element. The
  // guard's ptr() IS the first held element -- valid for exactly the guard's
  // lifetime -- so the run becomes a flat memcpy. Four floats per thread
  // lets the compiler emit 128-bit loads and stores.
  const clio::run::u64 quads = run >> 2;
  for (clio::run::u64 q = threadIdx.x; q < quads; q += blockDim.x) {
    float4 val;
    val.x = src[q * 4 + 0];
    val.y = src[q * 4 + 1];
    val.z = src[q * 4 + 2];
    val.w = src[q * 4 + 3];
    reinterpret_cast<float4 *>(dst)[q] = val;
  }
  for (clio::run::u64 k = (quads << 2) + threadIdx.x; k < run;
       k += blockDim.x) {
    dst[k] = src[k];
  }
}

/** Per-block slice of [lo,hi). Keyed on the LOGICAL block, never blockIdx.x:
 *  the driver relaunches a COMPACTED grid, so blockIdx.x stops matching the
 *  logical block after the first suspend. */
CTP_GPU_FUN void GnnSlice(clio::run::u64 &lo, clio::run::u64 hi,
                          float *&scratch, clio::run::u32 nblocks,
                          clio::run::u32 block, clio::run::u64 &n) {
  const clio::run::u64 total = hi - lo;
  const clio::run::u64 chunk = (total + nblocks - 1) / nblocks;
  const clio::run::u64 base = (clio::run::u64)block * chunk;
  n = base >= total ? 0 : min(chunk, total - base);
  lo += base;
  scratch += base;
}

/**
 * The gather, as a REAL C++20 device coroutine.
 *
 * co_await on HoldPage instead of the deleted CLIO_Y* macro state machine.
 * The compiler hoists everything live across a suspend into the coroutine
 * frame, so `i` and `run` are ordinary locals, and the loop reads as a loop.
 * The kernel is never parked spinning: a suspend returns from the kernel
 * entirely and the host driver resumes the frame.
 */
__device__ gy::YCoroMain GnnGatherCoro(gv::DeviceVector<float> v,
                                       clio::run::u64 lo, clio::run::u64 hi,
                                       float *scratch, clio::run::u32 nblocks,
                                       clio::run::u32 block) {
  clio::run::u64 n = 0;
  GnnSlice(lo, hi, scratch, nblocks, block, n);
  for (clio::run::u64 i = 0; i < n;) {
    co_await v.BeginFetch(v.PageLo(lo + i), v.PageSpan(lo + i, 1));
    co_await v.AwaitFetch();
    auto h = co_await v.HoldPage(lo + i, n - i);
    GnnCopyRun(h.ptr(), h.run(), scratch + i);
    i += h.run();
  }
}

__global__ void GnnGatherKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> v, clio::run::u64 lo,
                                clio::run::u64 hi, float *scratch,
                                clio::run::u32 nblocks, gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(GnnGatherCoro(v, lo, hi, scratch, nblocks, yv.Block()));
}

/** Forward + backward for the window's nodes (one thread per node). Writes per-node
 *  h1, dz1, dz2 to window buffers; accumulates loss/correct/count. */
/**
 * Warp-aggregated counter increment.
 *
 * The training kernel counts labelled nodes, correct predictions and the two
 * validation tallies, and it was doing one atomicAdd PER NODE to each -- four
 * global atomics on four fixed addresses from every one of ~33M threads. Every
 * lane in a warp contends for the same cache line, so the hardware serialises
 * them.
 *
 * __ballot_sync gives the warp's participating lanes in one instruction; lane
 * 0 adds their population count once. Same totals, 32x fewer atomics, and the
 * ORDER of an integer counter's increments does not affect its value, so this
 * is exactly equivalent rather than approximately so.
 */
__device__ __forceinline__ void WarpAddOne(int *addr) {
  const unsigned mask = __activemask();
  const int leader = __ffs(mask) - 1;
  const int cnt = __popc(mask);
  if ((threadIdx.x & 31) == leader) atomicAdd(addr, cnt);
}

/**
 * Nodes handled per thread.
 *
 * ONE, measured. The kernel is written to block over kNB nodes so that W1 is
 * loaded once per (f, h-tile) and reused, which is the obvious remaining lever
 * once occupancy, local memory, feature re-reads and atomics have each been
 * ruled out. It does not pay -- kNB>1 costs more in registers and in the
 * per-node h1 storage than the W1 reuse returns:
 *
 *   kNB=1 kHT=32  0.44 s  REG:83  STACK:1792
 *   kNB=2 kHT=32  0.47 s  REG:128 STACK:2864
 *   kNB=2 kHT=16  0.48 s  REG:79  STACK:2864
 *   kNB=4 kHT=16  0.49 s  REG:98  STACK:4944
 *   kNB=4 kHT=8   0.54 s  REG:72  STACK:4944
 *
 * The blocking is left in and set to 1: it costs nothing at kNB=1 and the
 * numbers above are the reason not to raise it.
 */
constexpr int kNB = 1;

__global__ void TrainNodeKernel(const float *A, clio::run::u64 nn, clio::run::u64 node_lo,
                                const long long *labels, int F, int H, int C,
                                const float *W1, const float *b1, const float *W2,
                                const float *b2, float *h1_out, float *dz1_out,
                                float *dz2_out, double *loss_buf, int *correct, int *count,
                                int *val_correct, int *val_count) {
  // REGISTER BLOCKING OVER NODES.
  //
  // With one node per thread, every thread streams all H*F of W1 for its own
  // node. Lanes in a warp read the same W1 element at the same time, so it
  // costs one broadcast per warp rather than per thread -- but still H*F per
  // 32 nodes, and that traffic is what forward is left standing on after
  // occupancy, local memory, feature re-reads and atomics were each tested and
  // ruled out.
  //
  // Giving each thread kNB nodes loads W1 once per (f, h-tile) and applies it
  // to all of them, so the same traffic serves kNB*32 nodes. The nodes a
  // thread owns are strided by blockDim, so for a given f the warp still reads
  // A at the same addresses it did before -- the access pattern is unchanged,
  // only the reuse is new.
  //
  // Arithmetic is untouched: acc[j][k] accumulates W1[(h0+k)*F+f]*a_j[f] over
  // ascending f exactly as the single-node form did.
  const clio::run::u64 base =
      (clio::run::u64)blockIdx.x * blockDim.x * kNB + threadIdx.x;
  clio::run::u64 nid[kNB];
  bool live[kNB];
  long long y[kNB];
#pragma unroll
  for (int j = 0; j < kNB; ++j) {
    nid[j] = base + (clio::run::u64)j * blockDim.x;
    live[j] = nid[j] < nn;
    if (live[j]) {
      y[j] = labels[node_lo + nid[j]];
      loss_buf[nid[j]] = 0.0;
      // ZERO ONLY WHAT WILL NOT BE WRITTEN. This used to clear dz2_out, h1_out
      // and dz1_out for EVERY node and then overwrite all three for the ones
      // that train, i.e. 2H+C wasted stores per live node. A node that trains
      // writes all of them; only the two early-out paths (unlabelled, and the
      // held-out validation stride) need the zeros, and those clear their own.
      const bool val = ((node_lo + nid[j]) % kValStride) == (kValStride - 1);
      if (y[j] < 0 || val) {
        for (int c = 0; c < C; ++c) dz2_out[nid[j] * C + c] = 0.0f;
        for (int h = 0; h < H; ++h) {
          h1_out[nid[j] * H + h] = 0.0f;
          dz1_out[nid[j] * H + h] = 0.0f;
        }
      }
      if (y[j] < 0) live[j] = false;
    }
  }

  float h1[kNB][kMaxH];
  constexpr int kHT = 32;
  // Layer 1, h tiled into registers. See kNB above for why the node blocking
  // is 1, and the notes below for what has been ruled out.
  //
  // TRIED: staging the feature rows through shared memory so the global reads
  // coalesce. A is [node][f], so a[f] is F*4 = 512 bytes apart across the warp
  // -- 32 transactions per instruction with four useful bytes each -- which
  // looks like an obvious 8x waste. Loading an f-chunk for the block first
  // (padded to avoid bank conflicts) makes no difference at all: 0.41 s either
  // way. The rows are re-read only H/kHT = 2 times and the cache absorbs them.
  for (int h0 = 0; h0 < H; h0 += kHT) {
    const int hn = min(kHT, H - h0);
    float acc[kNB][kHT];
#pragma unroll
    for (int j = 0; j < kNB; ++j)
#pragma unroll
      for (int k = 0; k < kHT; ++k) acc[j][k] = (k < hn) ? b1[h0 + k] : 0.f;
    for (int f = 0; f < F; ++f) {
      float w[kHT];
#pragma unroll
      for (int k = 0; k < kHT; ++k) w[k] = (k < hn) ? W1[(h0 + k) * F + f] : 0.f;
#pragma unroll
      for (int j = 0; j < kNB; ++j) {
        if (!live[j]) continue;
        const float av = A[nid[j] * (clio::run::u64)F + f];
#pragma unroll
        for (int k = 0; k < kHT; ++k) acc[j][k] += w[k] * av;
      }
    }
#pragma unroll
    for (int j = 0; j < kNB; ++j)
#pragma unroll
      for (int k = 0; k < kHT; ++k) {
        if (k < hn) h1[j][h0 + k] = acc[j][k] > 0.f ? acc[j][k] : 0.f;
      }
  }

  // The rest is per node: layer 2, softmax, loss and the backward pass. W2 is
  // only C*H floats, so there is nothing to gain from blocking it.
#pragma unroll 1
  for (int j = 0; j < kNB; ++j) {
    if (!live[j]) continue;
    const clio::run::u64 n = nid[j];
    float z2[kMaxC], maxz = -1e30f;
    constexpr int kCT = 8;
    for (int c0 = 0; c0 < C; c0 += kCT) {
      const int cn = min(kCT, C - c0);
      float acc[kCT];
#pragma unroll
      for (int k = 0; k < kCT; ++k) acc[k] = (k < cn) ? b2[c0 + k] : 0.f;
      for (int h = 0; h < H; ++h) {
        const float hv = h1[j][h];
#pragma unroll
        for (int k = 0; k < kCT; ++k) {
          if (k < cn) acc[k] += W2[(c0 + k) * H + h] * hv;
        }
      }
#pragma unroll
      for (int k = 0; k < kCT; ++k) {
        if (k < cn) { z2[c0 + k] = acc[k]; if (acc[k] > maxz) maxz = acc[k]; }
      }
    }
    float sum = 0.f;
    for (int c = 0; c < C; ++c) { z2[c] = expf(z2[c] - maxz); sum += z2[c]; }
    int pred = 0; float pmax = -1.f;   // p aliases z2
    for (int c = 0; c < C; ++c) { z2[c] = z2[c] / sum; if (z2[c] > pmax) { pmax = z2[c]; pred = c; } }
    if (((node_lo + n) % kValStride) == (kValStride - 1)) {
      if (pred == (int)y[j]) WarpAddOne(val_correct);
      WarpAddOne(val_count);
      continue;
    }
    loss_buf[n] = -log((double)fmaxf(z2[y[j]], 1e-30f));
    if (pred == (int)y[j]) WarpAddOne(correct);
    WarpAddOne(count);
    for (int c = 0; c < C; ++c) { z2[c] = z2[c] - (c == (int)y[j] ? 1.f : 0.f); dz2_out[n * C + c] = z2[c]; }
    for (int h0 = 0; h0 < H; h0 += kHT) {
      const int hn = min(kHT, H - h0);
      float acc[kHT];
#pragma unroll
      for (int k = 0; k < kHT; ++k) acc[k] = 0.f;
      for (int c = 0; c < C; ++c) {
        const float zv = z2[c];
#pragma unroll
        for (int k = 0; k < kHT; ++k) {
          if (k < hn) acc[k] += W2[c * H + h0 + k] * zv;
        }
      }
#pragma unroll
      for (int k = 0; k < kHT; ++k) {
        if (k < hn) {
          const float hv = h1[j][h0 + k];
          dz1_out[n * H + h0 + k] = (hv > 0.f) ? acc[k] : 0.f;
          h1_out[n * H + h0 + k] = hv;
        }
      }
    }
  }
}

/** Grad accumulation (one thread per weight element; sums the window's nodes in
 *  order and += into the global double grad -- deterministic across windows because
 *  the launches serialize on one stream). */
/** Number of node-range splits for the dW1 reduction. See GradW1Kernel. */
constexpr int kGradSplits = 32;

/**
 * Accumulator type for the weight gradients.
 *
 * DOUBLE by default, which is what the gradient kernels have always used and
 * is the only reason they cost what they do: GradW1Kernel runs at 247 GFLOP/s
 * against this GPU's MEASURED FP64 ceiling of 240 GFLOP/s (a pure register-FMA
 * benchmark on an RTX 4070 Laptop, 36 SMs), so it is AT the hardware limit and
 * cannot be made faster while it stays FP64.
 *
 * -DCLIO_GNN_GRAD_FP32 switches the accumulators to float. That is a numerical
 * change to the model, not a tuning knob, which is why it is off by default:
 * with kGradSplits=32 each thread sums nn/32 terms, so the accumulation depth
 * is ~16k and the expected relative error is around 8e-6 -- immaterial for SGD
 * but a hundred times the float representation error, and it is not my call to
 * make on someone else's training run. Both the in-core and the streamed path
 * use whichever is selected, so the bit-exactness assertion holds either way.
 */
#if defined(CLIO_GNN_GRAD_FP32)
using GradAcc = float;
#else
using GradAcc = double;
#endif

/**
 * dW1[h][f] += sum_n dz1[n][h] * A[n][f]: TILED and SPLIT over n.
 *
 * Two separate problems, and each fix alone made things worse:
 *
 *  - One thread per (h,f) is only H*F = 8192 threads. Splitting the node
 *    range kGradSplits ways fixes the occupancy.
 *  - Each thread reads A[n*F+f] for its own f, so A is fetched once per h --
 *    64 passes over the feature matrix, the same defect fixed in the forward
 *    pass. Staging A and dz1 in shared memory cuts that to H/kTH passes.
 *
 * Tiling WITHOUT the split was measured SLOWER than neither (1.37s vs 1.12s)
 * because it launched 32 blocks; that is why it was reverted earlier. With
 * the split the grid is (F/kTF, H/kTH, kGradSplits) = 1024 blocks, so the
 * traffic reduction finally has the threads to pay for it.
 *
 * ORDER IS PRESERVED EXACTLY: each thread still walks its split's n-range in
 * ascending order (chunks ascend, and r ascends within a chunk), and the
 * reduce folds splits in fixed order. No atomics. So this produces the same
 * doubles as the untiled split version, which produced the same values as the
 * unsplit one.
 */
__global__ void GradW1Kernel(const float *A, const float *dz1, clio::run::u64 nn,
                             int F, int H, double *part, double *partb) {
  constexpr int kTH = 16, kTF = 16, kTN = 64;
  // The shared tiles hold DOUBLE, not float. The inner loop is
  // s += (double)sD * (double)sA, and on this GPU cvt.f64.f32 issues to the
  // same FP64 unit as the FMA, at the same 1/64 rate -- so a float tile costs
  // TWO conversions per useful multiply-add, i.e. three FP64-unit ops to do
  // one, which is what pinned the kernel near 20% of the FP64 ceiling after
  // occupancy, traffic and ILP had each been fixed and each moved it a few
  // percent. Converting once at load time instead makes it 0.125 conversions
  // per FMA rather than 2.
  //
  // Exact, not approximate: a float-to-double conversion is lossless, and the
  // product of two floats needs 48 mantissa bits so the double multiply was
  // already exact. Same values, same order, same result.
  __shared__ GradAcc sA[kTN][kTF];
  __shared__ GradAcc sD[kTN][kTH];
  const int f0 = blockIdx.x * kTF;
  const int h0 = blockIdx.y * kTH;
  const int sp = blockIdx.z;
  const int tf = threadIdx.x % kTF;
  const int th = threadIdx.x / kTF;
  const clio::run::u64 chunk = (nn + kGradSplits - 1) / kGradSplits;
  const clio::run::u64 lo = (clio::run::u64)sp * chunk;
  const clio::run::u64 hi = min(lo + chunk, nn);
  // FOUR ACCUMULATORS, not one. A single `s` makes every FP64 FMA depend on
  // the previous one, so the loop runs at the latency of the FMA pipeline
  // rather than its throughput -- which is why this kernel sat at ~20% of the
  // FP64 ceiling even after it had enough threads and its traffic cut. Four
  // independent chains give the scheduler four FMAs to keep in flight.
  //
  // This DOES change the summation order (s0 takes r=0,4,8..., s1 takes
  // r=1,5,9..., combined pairwise at the end) and so the last bits of the
  // gradient can differ from the single-accumulator form. It is not a
  // precision downgrade: everything stays FP64, and splitting one long
  // sequential sum into four shorter ones then combining pairwise is if
  // anything better conditioned. It stays deterministic, and both the in-core
  // and the streamed path run this same kernel, so they still agree exactly.
  GradAcc s0 = 0, s1 = 0, s2 = 0, s3 = 0, sb = 0;
  for (clio::run::u64 n0 = lo; n0 < hi; n0 += kTN) {
    const int lim = (int)min((clio::run::u64)kTN, hi - n0);
    for (int i = threadIdx.x; i < kTN * kTF; i += blockDim.x) {
      const int r = i / kTF, c = i % kTF;
      sA[r][c] = (r < lim && f0 + c < F)
                     ? (GradAcc)A[(n0 + r) * (clio::run::u64)F + f0 + c] : (GradAcc)0;
    }
    for (int i = threadIdx.x; i < kTN * kTH; i += blockDim.x) {
      const int r = i / kTH, c = i % kTH;
      sD[r][c] = (r < lim && h0 + c < H)
                     ? (GradAcc)dz1[(n0 + r) * (clio::run::u64)H + h0 + c] : (GradAcc)0;
    }
    __syncthreads();
    int r = 0;
    for (; r + 3 < lim; r += 4) {
      s0 += sD[r][th] * sA[r][tf];
      s1 += sD[r + 1][th] * sA[r + 1][tf];
      s2 += sD[r + 2][th] * sA[r + 2][tf];
      s3 += sD[r + 3][th] * sA[r + 3][tf];
    }
    for (; r < lim; ++r) s0 += sD[r][th] * sA[r][tf];
    // db1 sums dz1 alone: only the first f-tile accumulates it, or every
    // f-tile would contribute its own copy.
    if (blockIdx.x == 0 && tf == 0) {
      for (int r = 0; r < lim; ++r) sb += sD[r][th];
    }
    __syncthreads();
  }
  const GradAcc s = (s0 + s1) + (s2 + s3);
  if (h0 + th < H && f0 + tf < F) {
    part[(clio::run::u64)sp * H * F + (h0 + th) * F + f0 + tf] = s;
  }
  if (blockIdx.x == 0 && tf == 0 && h0 + th < H) {
    partb[(clio::run::u64)sp * H + h0 + th] = sb;
  }
}

/** Fold GradW1Kernel's per-split partials in fixed order. */
__global__ void GradW1ReduceKernel(const double *part, const double *partb,
                                   int F, int H, double *dW1, double *db1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < H * F) {
    double s = 0.0;
    for (int sp = 0; sp < kGradSplits; ++sp) s += part[(clio::run::u64)sp * H * F + idx];
    dW1[idx] += s;
  }
  if (idx < H) {
    double sb = 0.0;
    for (int sp = 0; sp < kGradSplits; ++sp) sb += partb[(clio::run::u64)sp * H + idx];
    db1[idx] += sb;
  }
}

/**
 * dW2[c][h] += sum_n dz2[n][c] * h1[n][h]: tiled, split over n, DOUBLE tiles.
 *
 * Same three fixes as GradW1Kernel and for the same reasons -- see there. The
 * one that mattered was staging the tiles as double: cvt.f64.f32 issues to the
 * FP64 unit at 1/64 rate on this GPU, so converting inside the inner loop
 * costs two such ops per useful multiply-add.
 */
__global__ void GradW2Kernel(const float *h1, const float *dz2, clio::run::u64 nn,
                             int H, int C, double *part, double *partb) {
  constexpr int kTH = 16, kTC = 8, kTN = 64;
  __shared__ GradAcc sH[kTN][kTH];
  __shared__ GradAcc sD[kTN][kTC];
  const int h0 = blockIdx.x * kTH;
  const int c0 = blockIdx.y * kTC;
  const int sp = blockIdx.z;
  const int th = threadIdx.x % kTH;
  const int tc = threadIdx.x / kTH;
  const clio::run::u64 chunk = (nn + kGradSplits - 1) / kGradSplits;
  const clio::run::u64 lo = (clio::run::u64)sp * chunk;
  const clio::run::u64 hi = min(lo + chunk, nn);
  GradAcc s0 = 0, s1 = 0, s2 = 0, s3 = 0, sb = 0;
  for (clio::run::u64 n0 = lo; n0 < hi; n0 += kTN) {
    const int lim = (int)min((clio::run::u64)kTN, hi - n0);
    for (int i = threadIdx.x; i < kTN * kTH; i += blockDim.x) {
      const int r = i / kTH, c = i % kTH;
      sH[r][c] = (r < lim && h0 + c < H)
                     ? (GradAcc)h1[(n0 + r) * (clio::run::u64)H + h0 + c] : (GradAcc)0;
    }
    for (int i = threadIdx.x; i < kTN * kTC; i += blockDim.x) {
      const int r = i / kTC, c = i % kTC;
      sD[r][c] = (r < lim && c0 + c < C)
                     ? (GradAcc)dz2[(n0 + r) * (clio::run::u64)C + c0 + c] : (GradAcc)0;
    }
    __syncthreads();
    int r = 0;
    for (; r + 3 < lim; r += 4) {
      s0 += sD[r][tc] * sH[r][th];
      s1 += sD[r + 1][tc] * sH[r + 1][th];
      s2 += sD[r + 2][tc] * sH[r + 2][th];
      s3 += sD[r + 3][tc] * sH[r + 3][th];
    }
    for (; r < lim; ++r) s0 += sD[r][tc] * sH[r][th];
    if (blockIdx.x == 0 && th == 0) {
      for (int k = 0; k < lim; ++k) sb += sD[k][tc];
    }
    __syncthreads();
  }
  const GradAcc s = (s0 + s1) + (s2 + s3);
  if (c0 + tc < C && h0 + th < H) {
    part[(clio::run::u64)sp * C * H + (c0 + tc) * H + h0 + th] = s;
  }
  if (blockIdx.x == 0 && th == 0 && c0 + tc < C) {
    partb[(clio::run::u64)sp * C + c0 + tc] = sb;
  }
}

/** Fold GradW2Kernel's per-split partials in fixed order. */
__global__ void GradW2ReduceKernel(const double *part, const double *partb,
                                   int H, int C, double *dW2, double *db2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < C * H) {
    double s = 0.0;
    for (int sp = 0; sp < kGradSplits; ++sp) s += part[(clio::run::u64)sp * C * H + idx];
    dW2[idx] += s;
  }
  if (idx < C) {
    double sb = 0.0;
    for (int sp = 0; sp < kGradSplits; ++sp) sb += partb[(clio::run::u64)sp * C + idx];
    db2[idx] += sb;
  }
}
/** Deterministic loss reduction: one thread sums the window's per-node losses in
 *  order and adds to the global loss (windows serialize on the stream -> the total
 *  is order-deterministic -> bit-identical across the in-core and streamed runs). */
/** Blocks for the loss reduction. */
constexpr int kLossBlocks = 256;

/**
 * Per-block partial sums of the per-node loss.
 *
 * This replaces a <<<1,1>>> kernel that summed all nn doubles on a SINGLE
 * THREAD -- a dependent chain of 524288 global loads with nothing to hide the
 * latency behind. It sat inside the forward phase timer, which is why forward
 * appeared immovable at ~0.41s no matter what was done to TrainNodeKernel:
 * bisecting the training kernel in isolation showed it costs 11.0 ms/window
 * against the 51 ms/window the phase was reporting, and this is the other 40.
 *
 * Deterministic: each thread sums a contiguous ascending range, the block
 * folds those with a fixed-shape tree, and LossFinalKernel folds the blocks in
 * index order. No atomics.
 */
__global__ void LossPartialKernel(const double *loss_buf, clio::run::u64 nn,
                                  double *part) {
  __shared__ double sm[256];
  const clio::run::u64 nthreads = (clio::run::u64)gridDim.x * blockDim.x;
  const clio::run::u64 chunk = (nn + nthreads - 1) / nthreads;
  const clio::run::u64 idx = blockIdx.x * (clio::run::u64)blockDim.x + threadIdx.x;
  const clio::run::u64 lo = idx * chunk;
  const clio::run::u64 hi = min(lo + chunk, nn);
  double s = 0.0;
  for (clio::run::u64 n = lo; n < hi; ++n) s += loss_buf[n];
  sm[threadIdx.x] = s;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if ((int)threadIdx.x < st) sm[threadIdx.x] += sm[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.x] = sm[0];
}

/** Fold the per-block partials in index order. */
__global__ void LossFinalKernel(const double *part, int nblocks, double *d_loss) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double s = 0.0;
  for (int i = 0; i < nblocks; ++i) s += part[i];
  d_loss[0] += s;
}

#if !CTP_IS_DEVICE_PASS

namespace {
using gnn_test::EnvI64;

/**
 * Run a yieldable kernel to completion.
 *
 * The driver and its lane stack are built PER CALL, and that is not free:
 * YieldStack cudaMallocs nblocks*lanes*bytes_per_lane (2 MiB at 32x256x256),
 * Yieldable allocates its own block state, and cudaFree synchronizes the
 * device, so every window pays an allocate/free/sync cycle inside the gather
 * phase.
 *
 * HOISTING THEM OUT OF THE LOOP IS 45% OF THE GATHER (0.11 s -> 0.06 s at
 * 1 GiB) AND IS WRONG. Reusing the objects across runs -- calling
 * YieldStack::Reset() and letting RunToCompletion call Yieldable::Reset() --
 * makes the streamed run stop matching the in-core one: max|dloss| 5.6e-05,
 * CLOSE instead of BIT-EXACT. Isolating the two shows it is the DRIVER, not
 * the stack: rebuilding the stack per call and reusing only Yieldable still
 * fails, and still shows the 0.06 s gather, so the cost and the bug are the
 * same object.
 *
 * Yieldable::Reset() zeroes resume_point_, status_, wait_tag_ and the pending
 * list and uploads them, which looks complete, so something else survives a
 * run. That is core-runtime state, not the vector's and not this test's, so it
 * is reported rather than worked around here. Until it is fixed the objects
 * have to be rebuilt each call.
 */
template <typename LaunchT>
clio::run::u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  const unsigned nthreads =
      (unsigned)EnvI64("CLIO_GNN_GATHER_THREADS", 256);
  gy::Yieldable<> drv(nblocks, nthreads);
  gy::YieldStack stack(nblocks, nthreads, kYieldLaneBytes);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000,
      gv::ResumeWhenComplete);
}

/**
 * Driver and lane stack built ONCE and reused across every window.
 *
 * Building them per call costs an allocate/free pair each time -- YieldStack
 * cudaMallocs nblocks*lanes*bytes_per_lane (2 MiB at 32x256x256), Yieldable
 * allocates its own block state, and cudaFree SYNCHRONIZES THE DEVICE -- so a
 * 64-window run paid 64 of those inside the gather. Reusing them is worth 45%
 * of the gather.
 *
 * BOTH Resets are mandatory, and forgetting the driver's is silent. Yieldable
 * calls Reset() from its CONSTRUCTOR, not from RunToCompletion, so a reused
 * driver still has num_pending_ == 0 from the previous run: Round() returns
 * false immediately, the kernel never launches, and the gather quietly returns
 * whatever was already in the scratch buffer. It shows up as a fast run whose
 * loss is merely CLOSE (max|dloss| 5.6e-05) rather than BIT-EXACT.
 */
class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  clio::run::u32 Run(LaunchT &&launch) {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/200000,
      gv::ResumeWhenComplete);
  }
 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};
}  // namespace

TEST_CASE("gpu_vector: GNN training over a compressed/streamed feature matrix "
          "matches an in-core baseline (bit-exact) and runs beyond HBM",
          "[gpu_vector][compress][gnn][train][bench]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const std::string dir = std::getenv("CLIO_GNN_DATA") ? std::getenv("CLIO_GNN_DATA")
                                                       : "/workspace/data/ogb/arxiv";
  // Real dataset when gnn_agg.py has produced the aggregated matrix, else the
  // shared synthetic fallback -- see gnn_dataset.h. Training on synthetic data
  // still proves what this test asserts (the streamed run tracks the in-core
  // run exactly); only the accuracy numbers need the real thing.
  // STREAM MODE. At papers100M the aggregated matrix is 53 GiB, so the host
  // copy this test normally makes (ds.feat, then A0 on top of it) is not
  // merely wasteful, it is impossible -- two copies would be 106 GiB. With
  // CLIO_GNN_INGEST_FILE set the matrix is never held in host memory at all:
  // it is paged from the file straight into the CTE, and the shape comes from
  // the environment instead of from a loaded array.
  const char *ingest_file = std::getenv("CLIO_GNN_INGEST_FILE");
  gnn_test::Dataset ds;
  if (ingest_file == nullptr) {
    ds = gnn_test::LoadOrSynthDataset(dir, "TRAIN", "agg_features.f32");
  } else {
    ds.N = EnvI64("CLIO_GNN_NODES", 0);
    ds.F = (int)EnvI64("CLIO_GNN_DIM", 0);
    ds.C = (int)EnvI64("CLIO_GNN_CLASSES", 0);
    ds.real = true;
    REQUIRE(ds.N > 0);
    REQUIRE(ds.F > 0);
    REQUIRE(ds.C > 0);
    // Labels are N x 8 bytes (papers100M: 888 MiB) -- small enough to hold.
    if (const char *lf = std::getenv("CLIO_GNN_LABELS_FILE")) {
      REQUIRE(gnn_test::ReadFileQuiet(lf, ds.labels));
    } else {
      ds.labels.resize((size_t)(ds.N * (std::int64_t)sizeof(std::int64_t)));
      auto *lw = reinterpret_cast<std::int64_t *>(ds.labels.data());
      for (std::int64_t i = 0; i < ds.N; ++i) lw[i] = i % ds.C;
    }
  }
  clio::run::u64 N0 = (clio::run::u64)ds.N, E0 = (clio::run::u64)ds.E;
  int F = ds.F, C = ds.C;
  REQUIRE(N0 > 0);
  REQUIRE(C <= kMaxC);
  std::fprintf(stderr, "[TRAIN] source=%s N=%llu F=%d C=%d\n",
               ingest_file != nullptr ? "STREAMED (external ingest)"
                                      : ds.SourceName(),
               (unsigned long long)N0, F, C);
  const int H = std::getenv("CLIO_GNN_HIDDEN") ? std::atoi(std::getenv("CLIO_GNN_HIDDEN")) : 64;
  REQUIRE(H <= kMaxH);
  const int epochs = std::getenv("CLIO_GNN_EPOCHS") ? std::atoi(std::getenv("CLIO_GNN_EPOCHS")) : 30;
  const float lr = std::getenv("CLIO_GNN_LR") ? (float)std::atof(std::getenv("CLIO_GNN_LR")) : 0.5f;
  // CLIO_GNN_MINIBATCH=1 switches from full-batch GD (one update per epoch) to
  // mini-batch SGD (one update per streamed window). Default 0 keeps the exact
  // full-batch behaviour the bit-exactness results were measured with.
  const bool kMinibatch = std::getenv("CLIO_GNN_MINIBATCH") &&
                          std::atoi(std::getenv("CLIO_GNN_MINIBATCH")) != 0;
  const clio::run::u32 window = std::getenv("CLIO_GNN_WINDOW") ? (clio::run::u32)std::atoi(std::getenv("CLIO_GNN_WINDOW")) : 4;
  const clio::run::u64 page_rows = std::getenv("CLIO_GNN_PAGE_ROWS") ? (clio::run::u64)std::atoi(std::getenv("CLIO_GNN_PAGE_ROWS")) : 512;
  const clio::run::u64 page_size = page_rows * (clio::run::u64)F * sizeof(float);
  // REFERENCE CEILING for the gather path: a bare H2D cudaMemcpy at exactly
  // this page size, measured before the runtime starts. The gap between this
  // and the achieved gather rate is the vector's overhead.
  const MemcpyProbe mcp = ProbeMemcpyBandwidth(static_cast<size_t>(page_size));
  const clio::run::u64 epp = page_size / sizeof(float);
  REQUIRE(epp % (clio::run::u64)F == 0);
  clio::run::u64 want = std::getenv("CLIO_GNN_TRAIN_NODES") ? (clio::run::u64)std::atoll(std::getenv("CLIO_GNN_TRAIN_NODES")) : 0;
  if (want == 0) want = N0;
  clio::run::u64 K = (want + page_rows - 1) / page_rows; K = (K / window) * window; if (K == 0) K = window;
  const clio::run::u64 N = K * page_rows;
  const clio::run::u64 total_elems = N * (clio::run::u64)F;
  const clio::run::u64 dataset_bytes = total_elems * sizeof(float);

  size_t gfree = 0, gtot = 0; cudaMemGetInfo(&gfree, &gtot);
  // Device-wide free memory BEFORE this test allocates anything. The GPU is
  // exclusive to this job, so (g_gpu_free0 - free_now) is this process's total
  // device footprint -- not just the feature window, but weights, labels, page
  // cache and runtime buffers too. This is the number to quote as "peak GPU".
  const size_t g_gpu_free0 = gfree;
  size_t g_peak_gpu = 0;
  auto track_peak = [&]() {
    size_t fnow = 0, tnow = 0;
    if (cudaMemGetInfo(&fnow, &tnow) == cudaSuccess && g_gpu_free0 > fnow) {
      size_t used = g_gpu_free0 - fnow;
      if (used > g_peak_gpu) g_peak_gpu = used;
    }
  };
  std::fprintf(stderr,
      "\n============== GNN TRAINING (Eternia) ==============\n"
      "[TRAIN] A=%lluMiB (%llu nodes x %d-d agg, %s tiled)  F=%d H=%d C=%d  "
      "epochs=%d lr=%.3f  GPU free=%zuMiB  window=%u pages (peak %lluMiB)\n",
      (unsigned long long)(dataset_bytes >> 20), (unsigned long long)N, F,
      ingest_file != nullptr ? "streamed"
                             : (ds.real ? "real" : "SYNTHETIC"), F, H, C,
      epochs, lr, gfree >> 20, window, (unsigned long long)(2ull * window * page_size >> 20));

  // ---- Load aggregated features + labels (host), tiled ----
  // A0 exists only in the non-stream path; see the note at the load above.
  std::vector<float> A0;
  if (ingest_file == nullptr) {
    A0.resize(N0 * (clio::run::u64)F);
    REQUIRE(ds.feat.size() >= A0.size() * sizeof(float));
    std::memcpy(A0.data(), ds.feat.data(), A0.size() * sizeof(float));
    std::vector<char>().swap(ds.feat);  // drop the loader's copy immediately
  }
  std::vector<long long> lab0(N0);
  REQUIRE(ds.labels.size() >= N0 * sizeof(long long));
  std::memcpy(lab0.data(), ds.labels.data(), N0 * sizeof(long long));
  const clio::run::u64 file_elems = N0 * (clio::run::u64)F;

  // Resident tiled labels.
  long long *d_lab = nullptr; d_lab = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_lab)>>(N * sizeof(long long));
  { std::vector<long long> tile(std::min<clio::run::u64>(N, 1u << 22));
    for (clio::run::u64 off = 0; off < N; off += tile.size()) {
      clio::run::u64 n = std::min<clio::run::u64>(tile.size(), N - off);
      for (clio::run::u64 i = 0; i < n; ++i) tile[i] = lab0[(off + i) % N0];
      ctp::GpuApi::Memcpy(d_lab + off, tile.data(), n * sizeof(long long));
    } }

  // ---- Shared init weights (seeded; identical for both runs) ----
  std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 0.1f);
  std::vector<float> W1_0(H * F), b1_0(H, 0.f), W2_0(C * H), b2_0(C, 0.f);
  for (auto &w : W1_0) w = nd(rng); for (auto &w : W2_0) w = nd(rng);

  // Device weights + grads + window buffers.
  const clio::run::u64 max_nn = (clio::run::u64)window * page_rows;
  float *dW1, *db1, *dW2, *db2; double *gW1, *gb1, *gW2, *gb2;
  // Per-split partials for the dW1 reduction (kGradSplits ways).
  double *gw1_part = nullptr, *gb1_part = nullptr;
  double *gw2_part = nullptr, *gb2_part = nullptr;
  double *loss_part = nullptr;

  float *h1_buf, *dz1_buf, *dz2_buf, *d_scratch;
  dW1 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(dW1)>>(H * F * sizeof(float));
  db1 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(db1)>>(H * sizeof(float));
  dW2 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(dW2)>>(C * H * sizeof(float));
  db2 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(db2)>>(C * sizeof(float));
  gW1 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(gW1)>>(H * F * sizeof(double));
  gw1_part = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(gw1_part)>>((size_t)kGradSplits * H * F * sizeof(double));
  gb1_part = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(gb1_part)>>((size_t)kGradSplits * H * sizeof(double));
  gw2_part = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(gw2_part)>>((size_t)kGradSplits * C * H * sizeof(double));
  gb2_part = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(gb2_part)>>((size_t)kGradSplits * C * sizeof(double));
  loss_part = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(loss_part)>>(kLossBlocks * sizeof(double));
  gb1 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(gb1)>>(H * sizeof(double));
  gW2 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(gW2)>>(C * H * sizeof(double));
  gb2 = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(gb2)>>(C * sizeof(double));
  h1_buf = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(h1_buf)>>(max_nn * H * sizeof(float));
  dz1_buf = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(dz1_buf)>>(max_nn * H * sizeof(float));
  dz2_buf = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(dz2_buf)>>(max_nn * C * sizeof(float));
  d_scratch = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_scratch)>>(max_nn * F * sizeof(float));
  double *loss_buf; loss_buf = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(loss_buf)>>(max_nn * sizeof(double));
  double *d_loss; int *d_correct, *d_count, *d_vcorrect, *d_vcount;
  d_loss = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_loss)>>(sizeof(double));
  d_correct = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_correct)>>(sizeof(int));
  d_count = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_count)>>(sizeof(int));
  d_vcorrect = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_vcorrect)>>(sizeof(int));
  d_vcount = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_vcount)>>(sizeof(int));

  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  // One epoch over a window provider. get_win(win_idx, first_node, nn) fills
  // d_scratch (nn*F). Returns final (loss,acc). Deterministic accumulation.
  auto run_epoch = [&](std::function<void(clio::run::u64, clio::run::u64, clio::run::u64)> gather,
                       double &out_loss, double &out_acc, clio::run::u64 &out_count,
                       double &out_vacc) {
    ctp::GpuApi::Memset(gW1, 0, H * F * sizeof(double)); ctp::GpuApi::Memset(gb1, 0, H * sizeof(double));
    ctp::GpuApi::Memset(gW2, 0, C * H * sizeof(double)); ctp::GpuApi::Memset(gb2, 0, C * sizeof(double));
    ctp::GpuApi::Memset(d_loss, 0, sizeof(double)); ctp::GpuApi::Memset(d_correct, 0, sizeof(int));
    ctp::GpuApi::Memset(d_count, 0, sizeof(int));
    ctp::GpuApi::Memset(d_vcorrect, 0, sizeof(int)); ctp::GpuApi::Memset(d_vcount, 0, sizeof(int));

    // Apply one SGD step from the currently accumulated gradients, normalised by
    // `nrm` contributing nodes, then (in mini-batch mode) clear them for the next
    // batch. Weights are tiny (H*F + C*H floats), so the host round-trip is cheap.
    auto apply_step = [&](int nrm, bool clear_grads) {
      std::vector<double> aW1(H * F), ab1(H), aW2(C * H), ab2(C);
      ctp::GpuApi::Memcpy(aW1.data(), gW1, H * F * sizeof(double));
      ctp::GpuApi::Memcpy(ab1.data(), gb1, H * sizeof(double));
      ctp::GpuApi::Memcpy(aW2.data(), gW2, C * H * sizeof(double));
      ctp::GpuApi::Memcpy(ab2.data(), gb2, C * sizeof(double));
      std::vector<float> aw1(H * F), ab1f(H), aw2(C * H), ab2f(C);
      ctp::GpuApi::Memcpy(aw1.data(), dW1, H * F * sizeof(float));
      ctp::GpuApi::Memcpy(ab1f.data(), db1, H * sizeof(float));
      ctp::GpuApi::Memcpy(aw2.data(), dW2, C * H * sizeof(float));
      ctp::GpuApi::Memcpy(ab2f.data(), db2, C * sizeof(float));
      float ainv = 1.0f / (float)std::max<int>(1, nrm);
      for (int i = 0; i < H * F; ++i) aw1[i] -= lr * (float)(aW1[i] * ainv);
      for (int i = 0; i < H; ++i) ab1f[i] -= lr * (float)(ab1[i] * ainv);
      for (int i = 0; i < C * H; ++i) aw2[i] -= lr * (float)(aW2[i] * ainv);
      for (int i = 0; i < C; ++i) ab2f[i] -= lr * (float)(ab2[i] * ainv);
      ctp::GpuApi::Memcpy(dW1, aw1.data(), H * F * sizeof(float));
      ctp::GpuApi::Memcpy(db1, ab1f.data(), H * sizeof(float));
      ctp::GpuApi::Memcpy(dW2, aw2.data(), C * H * sizeof(float));
      ctp::GpuApi::Memcpy(db2, ab2f.data(), C * sizeof(float));
      if (clear_grads) {
        ctp::GpuApi::Memset(gW1, 0, H * F * sizeof(double)); ctp::GpuApi::Memset(gb1, 0, H * sizeof(double));
        ctp::GpuApi::Memset(gW2, 0, C * H * sizeof(double)); ctp::GpuApi::Memset(gb2, 0, C * sizeof(double));
      }
    };
    int prev_cnt = 0;   // labelled nodes seen before the current batch

    for (clio::run::u64 w = 0; w < K; w += window) {
      clio::run::u64 first = w * page_rows;
      clio::run::u64 nn = (clio::run::u64)window * page_rows;
      // PHASE TIMING (CLIO_GNN_KTIME=1): synchronize around each phase so
      // wall clock is attributed to gather / forward / gradient rather than
      // inferred from totals. Off by default -- the syncs serialize the
      // pipeline and would distort the number being measured.
      const bool ktime = g_ktime;
      double tp0 = 0.0;
      if (ktime) { ctp::GpuApi::Synchronize(); tp0 = NowSec(); }
      gather(w, first, nn);   // fills d_scratch with nn*F floats
      if (ktime) { ctp::GpuApi::Synchronize(); t_gather += NowSec() - tp0; tp0 = NowSec(); }
      int tpb = 128;
      // TRIED AND REVERTED: templating this kernel on the per-thread array
      // bounds so a 64/40 run would not carry kMaxH/kMaxC (256/192) worth of
      // local scratch. The instantiation does not survive this build's device
      // linking -- the launch fails with "invalid device function" -- and the
      // failure is silent unless the error below is checked.
      TrainNodeKernel<<<(int)((nn + (clio::run::u64)tpb * kNB - 1) / (tpb * kNB)), tpb>>>(
          d_scratch, nn, first, d_lab, F, H, C, dW1, db1, dW2, db2,
          h1_buf, dz1_buf, dz2_buf, loss_buf, d_correct, d_count,
          d_vcorrect, d_vcount);
      // A launch that fails leaves loss_buf untouched, and the test then
      // compares two runs of zeros and calls them BIT-EXACT. Check it.
      {
        const cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
          std::fprintf(stderr, "[TRAIN] TrainNodeKernel launch FAILED: %s\n",
                       cudaGetErrorString(le));
          REQUIRE(le == cudaSuccess);
        }
      }
      LossPartialKernel<<<kLossBlocks, 256>>>(loss_buf, nn, loss_part);
      LossFinalKernel<<<1, 1>>>(loss_part, kLossBlocks, d_loss);
      if (ktime) { ctp::GpuApi::Synchronize(); t_fwd += NowSec() - tp0; tp0 = NowSec(); }
      track_peak();
      GradW1Kernel<<<dim3((F + 15) / 16, (H + 15) / 16, kGradSplits), 256>>>(
          d_scratch, dz1_buf, nn, F, H, gw1_part, gb1_part);
      GradW1ReduceKernel<<<(H * F + tpb - 1) / tpb, tpb>>>(
          gw1_part, gb1_part, F, H, gW1, gb1);
      if (ktime) { ctp::GpuApi::Synchronize(); t_gw1 += NowSec() - tp0; tp0 = NowSec(); }
      GradW2Kernel<<<dim3((H + 15) / 16, (C + 7) / 8, kGradSplits), 128>>>(
          h1_buf, dz2_buf, nn, H, C, gw2_part, gb2_part);
      GradW2ReduceKernel<<<(C * H + tpb - 1) / tpb, tpb>>>(
          gw2_part, gb2_part, H, C, gW2, gb2);
      if (ktime) { ctp::GpuApi::Synchronize(); t_gw2 += NowSec() - tp0; }
      if (kMinibatch) {
        // Mini-batch SGD: one weight update per window instead of one per epoch,
        // so an epoch performs ceil(K/window) updates rather than 1. Normalise by
        // the labelled nodes in THIS batch (d_count accumulates across the epoch).
        ctp::GpuApi::Synchronize();
        int cnt_now = 0;
        ctp::GpuApi::Memcpy(&cnt_now, d_count, sizeof(int));
        apply_step(cnt_now - prev_cnt, /*clear_grads=*/true);
        prev_cnt = cnt_now;
      }
    }
    ctp::GpuApi::Synchronize();
    // SGD update on host (weights are small).
    std::vector<double> hW1(H * F), hb1(H), hW2(C * H), hb2(C); double hloss; int hcorr, hcnt;
    ctp::GpuApi::Memcpy(hW1.data(), gW1, H * F * sizeof(double));
    ctp::GpuApi::Memcpy(hb1.data(), gb1, H * sizeof(double));
    ctp::GpuApi::Memcpy(hW2.data(), gW2, C * H * sizeof(double));
    ctp::GpuApi::Memcpy(hb2.data(), gb2, C * sizeof(double));
    ctp::GpuApi::Memcpy(&hloss, d_loss, sizeof(double));
    ctp::GpuApi::Memcpy(&hcorr, d_correct, sizeof(int));
    ctp::GpuApi::Memcpy(&hcnt, d_count, sizeof(int));
    int hvcorr = 0, hvcnt = 0;
    ctp::GpuApi::Memcpy(&hvcorr, d_vcorrect, sizeof(int));
    ctp::GpuApi::Memcpy(&hvcnt, d_vcount, sizeof(int));
    std::vector<float> W1(H * F), b1(H), W2(C * H), b2(C);
    ctp::GpuApi::Memcpy(W1.data(), dW1, H * F * sizeof(float));
    ctp::GpuApi::Memcpy(b1.data(), db1, H * sizeof(float));
    ctp::GpuApi::Memcpy(W2.data(), dW2, C * H * sizeof(float));
    ctp::GpuApi::Memcpy(b2.data(), db2, C * sizeof(float));
    float inv = 1.0f / (float)std::max<int>(1, hcnt);
    if (kMinibatch) {
      // Updates were already applied per window; do not apply an extra epoch-level
      // step (that would double-count and use stale, cleared gradients).
      out_loss = hloss / std::max<int>(1, hcnt);
      out_acc = (double)hcorr / std::max<int>(1, hcnt);
      out_count = (clio::run::u64)hcnt;
      out_vacc = hvcnt ? (double)hvcorr / (double)hvcnt : 0.0;
      return;
    }
    for (int i = 0; i < H * F; ++i) W1[i] -= lr * (float)(hW1[i] * inv);
    for (int i = 0; i < H; ++i) b1[i] -= lr * (float)(hb1[i] * inv);
    for (int i = 0; i < C * H; ++i) W2[i] -= lr * (float)(hW2[i] * inv);
    for (int i = 0; i < C; ++i) b2[i] -= lr * (float)(hb2[i] * inv);
    ctp::GpuApi::Memcpy(dW1, W1.data(), H * F * sizeof(float));
    ctp::GpuApi::Memcpy(db1, b1.data(), H * sizeof(float));
    ctp::GpuApi::Memcpy(dW2, W2.data(), C * H * sizeof(float));
    ctp::GpuApi::Memcpy(db2, b2.data(), C * sizeof(float));
    out_loss = hloss / std::max<int>(1, hcnt); out_acc = (double)hcorr / std::max<int>(1, hcnt);
    out_count = (clio::run::u64)hcnt;
    out_vacc = hvcnt ? (double)hvcorr / (double)hvcnt : 0.0;
  };

  auto reset_weights = [&]() {
    ctp::GpuApi::Memcpy(dW1, W1_0.data(), H * F * sizeof(float));
    ctp::GpuApi::Memcpy(db1, b1_0.data(), H * sizeof(float));
    ctp::GpuApi::Memcpy(dW2, W2_0.data(), C * H * sizeof(float));
    ctp::GpuApi::Memcpy(db2, b2_0.data(), C * sizeof(float));
  };

  // =========================== IN-CORE baseline ===========================
  std::vector<double> base_loss(epochs), base_acc(epochs), base_vacc(epochs);
  std::string base_status = (ingest_file != nullptr) ? "N/A(stream)" : "OOM";
  double base_time = -1.0;
  {
    size_t fb = 0, tb = 0; cudaMemGetInfo(&fb, &tb);
    float *d_all = nullptr; cudaError_t al = cudaErrorMemoryAllocation;
    // In stream mode there is no host copy to upload from, so the in-core
    // baseline is not merely too big -- it is unavailable. Report it as such
    // rather than pretending it OOM'd on GPU memory.
    if (A0.empty()) {
      al = cudaErrorMemoryAllocation;
    } else if (dataset_bytes + (size_t(384) << 20) < fb) {
      // RAW cudaMalloc on purpose: this probe WANTS a graceful OOM (the
      // in-core baseline is optional), and GpuApi::Malloc fails fatally.
      al = cudaMalloc(&d_all, dataset_bytes);
    }
    if (al != cudaSuccess) {
      cudaGetLastError();
      if (ingest_file != nullptr) {
        std::fprintf(stderr,
                     "[TRAIN] IN-CORE: SKIPPED -- stream mode keeps no host "
                     "copy to upload, so there is no in-core run to compare "
                     "against. This is NOT an out-of-memory result.\n");
      } else
      std::fprintf(stderr, "[TRAIN] IN-CORE: *** OOM *** cannot place %lluMiB A resident.\n",
                   (unsigned long long)(dataset_bytes >> 20));
    } else {
      for (clio::run::u64 off = 0; off < total_elems; off += file_elems) {
        clio::run::u64 n = std::min(file_elems, total_elems - off);
        ctp::GpuApi::Memcpy(d_all + off, A0.data(), n * sizeof(float));
      }
      reset_weights();
      double t0 = NowSec();
      for (int e = 0; e < epochs; ++e) {
        clio::run::u64 cnt;
        run_epoch([&](clio::run::u64, clio::run::u64 first, clio::run::u64 nn) {
          ctp::GpuApi::Memcpy(d_scratch, d_all + first * F, nn * F * sizeof(float));
        }, base_loss[e], base_acc[e], cnt, base_vacc[e]);
      }
      base_time = NowSec() - t0; base_status = "OK";
      ctp::GpuApi::Free(d_all);
      std::fprintf(stderr, "[TRAIN] IN-CORE: epoch0 loss=%.6f acc=%.4f -> epoch%d loss=%.6f acc=%.4f val_acc=%.4f (%.2fs)\n",
                   base_loss[0], base_acc[0], epochs - 1, base_loss[epochs - 1], base_acc[epochs - 1],
                   base_vacc[epochs - 1], base_time);
    }
  }

  // =========================== ETERNIA (streamed) ===========================
  const char *tag = "gnn_train";
  clio::cte::core::Client core; core.Init(clio::cte::core::kCtePoolId);
  auto tagf = core.AsyncGetOrCreateTag(tag); tagf.Wait(); REQUIRE(tagf->GetReturnCode() == 0);
  auto tag_id = tagf->tag_id_;
  // The Vector resolves its tag by NAME, so redirecting it after an
  // in-process aggregation means changing this string, not just tag_id.
  std::string vec_tag = tag;
  clio::cte::core::Client comp; comp.Init(StoragePool());
  // Pin the codec explicitly. With only dynamic_compress_ set the predictor is
  // free to choose a pass-through, which stores the matrix uncompressed while
  // every assertion below still passes -- the same trap the Option A test fell
  // into once CLIO_CTE_COMPRESS_LIB stopped being read.
  const int comp_lib = (int)EnvI64("CLIO_GNN_COMPRESS_LIB", 10);     // ZSTD
  const int comp_preset = (int)EnvI64("CLIO_GNN_COMPRESS_PRESET", 2);  // BALANCED
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.compress_lib_ = comp_lib;
  ctx.compress_preset_ = comp_preset;
  clio::run::bdev::Client hbm_bdev(clio::run::PoolId(512, 1)), dram_bdev(clio::run::PoolId(513, 1));
  auto h0 = hbm_bdev.AsyncGetStats(); h0.Wait(); auto r0 = dram_bdev.AsyncGetStats(); r0.Wait();
  clio::run::u64 hbm0 = h0->remaining_size_, dram0 = r0->remaining_size_;

  std::vector<float> pagebuf(epp);
  std::FILE *ing = nullptr;
  if (ingest_file != nullptr) {
    ing = std::fopen(ingest_file, "rb");
    REQUIRE(ing != nullptr);
    std::fprintf(stderr, "[TRAIN] streaming pages from %s\n", ingest_file);
  }
  double store_t0 = NowSec();
  for (clio::run::u64 p = 0; p < K; ++p) {
    if (ing != nullptr) {
      // Read one page straight from the stream; nothing else is retained.
      const size_t want = (size_t)(epp * sizeof(float));
      const size_t got = std::fread(pagebuf.data(), 1, want, ing);
      if (got < want) {
        std::memset(reinterpret_cast<char *>(pagebuf.data()) + got, 0,
                    want - got);
      }
    } else {
      // Tile A into this page (host buffer; zstd is CPU-side).
      clio::run::u64 start = p * epp;
      for (clio::run::u64 i = 0; i < epp; ++i) pagebuf[i] = A0[(start + i) % file_elems];
    }
    ctp::ipc::ShmPtr<> dp; dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(pagebuf.data());
    // The vector's fault path names a page by its number in decimal.
    // The old "<tag>_b0_pi<n>" scheme is from the pre-rewrite vector; storing
    // under it would leave every page a miss with no visible error.
    std::string name = std::to_string(p);
    auto pf = comp.AsyncPutBlob(tag_id, name, 0ull, page_size, dp, 0.5f, ctx, 0, clio::run::PoolQuery::Local());
    pf.Wait(); REQUIRE(pf->GetReturnCode() == 0);
  }
  if (ing != nullptr) std::fclose(ing);

  // IN-PROCESS AGGREGATION. With CLIO_GNN_CSR_FILE the file streamed above is
  // the RAW feature matrix, and the aggregated matrix the trainer wants is
  // produced right here, from the CTE into the CTE.
  //
  // This exists because the aggregation tool is a daemon client and this test
  // cannot be one -- a process that faults on the GPU has to host the runtime,
  // so it has its own CTE. Aggregating externally would mean exporting the
  // result (53 GiB at papers100M) and re-ingesting it, i.e. exactly the flat
  // copy the whole pipeline is built to avoid.
  if (const char *csr_path = std::getenv("CLIO_GNN_CSR_FILE")) {
    std::vector<char> csr;
    REQUIRE(gnn_test::ReadFileQuiet(csr_path, csr));
    const std::int64_t *hdr = reinterpret_cast<const std::int64_t *>(csr.data());
    REQUIRE((clio::run::u64)hdr[0] == N0);
    const std::int64_t *ip = hdr + 2;
    const std::int64_t *ix = hdr + 2 + (N0 + 1);
    std::fprintf(stderr,
                 "[TRAIN] aggregating in-process: N=%llu E=%lld (CSR %s)\n",
                 (unsigned long long)N0, (long long)hdr[1], csr_path);

    // The store above already wrote the RAW matrix under `tag`; aggregation
    // reads that and writes `tag_agg`, which the vector then opens.
    const auto tag_id_raw = tag_id;
    auto agg_tf = core.AsyncGetOrCreateTag(std::string(tag) + "_agg");
    agg_tf.Wait();
    REQUIRE(agg_tf->GetReturnCode() == 0);

    gnn_agg::Params ap;
    ap.page_bytes = page_size;
    ap.dim = F;
    ap.nodes = (std::int64_t)N0;
    ap.block_rows = EnvI64("CLIO_GNN_AGG_BLOCK_ROWS", 0);
    ap.codec = comp_lib;
    ap.preset = comp_preset;
    const double agg_t0 = NowSec();
    REQUIRE(gnn_agg::Aggregate(comp, tag_id, agg_tf->tag_id_, ip, ix, ap));
    std::fprintf(stderr, "[TRAIN] aggregation done in %.1fs\n",
                 NowSec() - agg_t0);
    // Self-check: at papers100M nothing outside this process can recompute A
    // to diff against, so sampled rows are recomputed from the stored raw
    // pages and compared. Cheap relative to the aggregation itself.
    const int nver = (int)EnvI64("CLIO_GNN_VERIFY_ROWS", 32);
    if (nver > 0) {
      REQUIRE(gnn_agg::VerifyRows(comp, tag_id_raw, agg_tf->tag_id_, ip, ix, ap,
                                  nver, 1e-3, 0x9E3779B97F4A7C15ull));
      // Prove the checker can fail. Comparing the RAW matrix against the
      // aggregate formula must mismatch on any graph with edges; if this
      // reports OK the verifier is not actually checking anything, and the
      // reassurance it gives at papers100M would be worthless.
      if (EnvI64("CLIO_GNN_VERIFY_SELFTEST", 1) != 0) {
        const bool caught = !gnn_agg::VerifyRows(comp, tag_id_raw, tag_id_raw,
                                                 ip, ix, ap, nver, 1e-3,
                                                 0x9E3779B97F4A7C15ull);
        std::fprintf(stderr, "[TRAIN] verifier self-test (raw vs formula): %s\n",
                     caught ? "correctly MISMATCHED" : "*** DID NOT CATCH ***");
        REQUIRE(caught);
      }
    }
    tag_id = agg_tf->tag_id_;
    vec_tag = std::string(tag) + "_agg";
  }

  double store_dt = NowSec() - store_t0;
  auto h1s = hbm_bdev.AsyncGetStats(); h1s.Wait(); auto r1s = dram_bdev.AsyncGetStats(); r1s.Wait();
  clio::run::u64 hbm_used = hbm0 >= h1s->remaining_size_ ? hbm0 - h1s->remaining_size_ : 0;
  clio::run::u64 dram_used = dram0 >= r1s->remaining_size_ ? dram0 - r1s->remaining_size_ : 0;
  clio::run::u64 stored = hbm_used + dram_used;
  double ratio = stored ? (double)dataset_bytes / (double)stored : 0.0;
  std::fprintf(stderr, "[TRAIN] ETERNIA: stored A %lluMiB -> %lluMiB zstd (%.3fx) in %.2fs\n",
               (unsigned long long)(dataset_bytes >> 20), (unsigned long long)(stored >> 20), ratio, store_dt);
  // WHICH TIER actually holds the bytes. The requirement is kHBM whenever it
  // fits, and a config that merely DECLARES an HBM tier does not establish
  // that: the DPE ranks targets by a predicted bandwidth model, so a scored
  // HBM tier has been measured receiving zero blobs. Print the split so a run
  // that quietly landed in DRAM cannot be reported as an HBM result.
  std::fprintf(stderr, "[TRAIN] TIER SPLIT: kHBM %lluMiB, DRAM %lluMiB%s\n",
               (unsigned long long)(hbm_used >> 20),
               (unsigned long long)(dram_used >> 20),
               hbm_used == 0 ? "   <-- VIOLATION: nothing landed in kHBM" : "");

  // The gather runs WIDE. It used to be one block of 32 threads, which copied
  // the whole matrix through a single warp on a single SM and cost 3.2x over
  // an in-core baseline -- more than paging and compression combined, and the
  // reason a fully resident run measured SLOWER than a paged one (the copy
  // dominated; the faults were noise).
  //
  // TOTAL cache is held at 2*window pages regardless of the block count, so a
  // wider gather is not silently also a bigger cache. Page caches are per
  // block, so blocks are capped at `window`: each block needs >=2 pages (one
  // held, one in flight) for the fault path to make progress.
  const clio::run::u32 want_blocks =
      (clio::run::u32)EnvI64("CLIO_GNN_GATHER_BLOCKS", 32);
  const clio::run::u32 gather_blocks =
      std::max<clio::run::u32>(1, std::min<clio::run::u32>(want_blocks, window));
  // PAGES PER BLOCK IS WHAT MAKES PAGING BATCHED. For an encoded tag the
  // fault path claims a RUN of pages with one multi-get
  // (BeginFetchRunLocked), and the run can never be longer than the slots the
  // block owns. Sizing this at 2 -- which is what holding the TOTAL cache
  // fixed while widening the gather produced -- caps every request at one
  // page, and one page per request is the whole cost of a compressed run:
  // 30.5ms of task round trip for a 0.159ms decode.
  const clio::run::u32 ppb = (clio::run::u32)EnvI64(
      "CLIO_GNN_PAGES_PER_BLOCK",
      std::max<clio::run::u32>(2, (2 * window) / gather_blocks));
  // WINDOW MUST COVER blocks x run_length PAGES, or the run-fetch over-reads.
  //
  // For an encoded tag a fault fetches a RUN of consecutive pages
  // (BeginFetchRunLocked, min(kPodMultiMax, pages_per_block) long). The gather
  // splits a window across the blocks, so a block owns window/blocks
  // consecutive pages. If the run is longer than that slice, the tail of every
  // run belongs to OTHER blocks -- they never read it, and those blocks fetch
  // the same pages again into their own tables. Measured at window=32 with 8
  // blocks: each page fetched 16 times (8 blocks x 2 epochs), 87.3% of all
  // fetches wasted. Sizing window >= blocks * run takes it to 0.0%.
  {
    const clio::run::u32 run_len =
        std::min<clio::run::u32>(clio::cte::core::kPodMultiMax, ppb);
    const clio::run::u32 need = gather_blocks * run_len;
    if (comp_lib != 0 && (clio::run::u32)window < need) {
      std::fprintf(stderr,
                   "[TRAIN] WARNING: window=%d pages < blocks*run=%u -- the "
                   "run-fetch will over-read and other blocks will re-fetch "
                   "the same pages. Set CLIO_GNN_WINDOW >= %u.\n",
                   window, need, need);
    }
  }
  std::fprintf(stderr,
               "[TRAIN] gather: %u blocks x %lld threads, %u pages/block "
               "(total cache %u pages = %lluMiB)\n",
               gather_blocks, (long long)EnvI64("CLIO_GNN_GATHER_THREADS", 256),
               ppb, gather_blocks * ppb,
               (unsigned long long)((gather_blocks * (clio::run::u64)ppb *
                                     page_size) >> 20));
  gv::Vector<float> vec(vec_tag, {0}, page_size, gather_blocks,
                        /*pages_per_block=*/ppb, K * epp, StoragePool(),
                        comp_lib, comp_preset);
  vec.EnableStats();
  // Tell the device side how big each stored page is, so an encoded page is
  // fetched with its true (compressed) length rather than the logical one.
  for (int attempt = 0; attempt < 0; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::vector<double> et_loss(epochs), et_acc(epochs), et_vacc(epochs);
  reset_weights();
  YieldRunner gather_runner(gather_blocks,
                            (unsigned)EnvI64("CLIO_GNN_GATHER_THREADS", 256));

  g_ktime = EnvI64("CLIO_GNN_KTIME", 0) != 0;
  t_gather = t_fwd = t_gw1 = t_gw2 = 0;
  double et_t0 = NowSec();
  for (int e = 0; e < epochs; ++e) {
    clio::run::u64 cnt;
    // Stream A windows through the vector each epoch: prefetch (decompress) each
    // window into HBM slots, gather its ELEMENTS [first*F, (first+nn)*F) into
    // d_scratch, then run the same train kernels as the in-core path.
    run_epoch([&](clio::run::u64 wi, clio::run::u64 first, clio::run::u64 nn) {
      (void)wi;  // pages are faulted on demand now; no separate prefetch step
      gather_runner.Run([&](dim3 g, dim3 b, gy::YieldableView<> vw,
                            gy::YieldStackView sv) {
        GnnGatherKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu_info, vec.GetDevice(0), first * (clio::run::u64)F,
            (first + nn) * (clio::run::u64)F, d_scratch, gather_blocks, vw, sv);
      });
      ctp::GpuApi::Synchronize();
    }, et_loss[e], et_acc[e], cnt, et_vacc[e]);
    std::fprintf(stderr, "[TRAIN]   eternia e%02d loss=%.6f acc=%.4f val_acc=%.4f (%.1fs elapsed)\n",
                 e, et_loss[e], et_acc[e], et_vacc[e], NowSec() - et_t0);
  }
  {
    // The vector's own paging counters. `faults` against the number of pages
    // the workload actually reads is the honest measure of whether the page
    // stream is relevant; prefetch_late says whether a run-fetch landed before
    // the block walked onto it, which is the difference between prefetching
    // and re-fetching.
    auto vs = vec.ReadStats(0);
    const unsigned long long need = (unsigned long long)K * (unsigned long long)epochs;
    std::fprintf(stderr,
        "[TRAIN] PAGING: faults=%llu (workload needs %llu -> %.2fx) evicts=%llu puts=%llu "
        "get_err=%llu put_err=%llu\n",
        (unsigned long long)vs.faults, need,
        need ? (double)vs.faults / (double)need : 0.0,
        (unsigned long long)vs.evicts, (unsigned long long)vs.puts,
        (unsigned long long)vs.get_errors, (unsigned long long)vs.put_errors);
  }

  if (g_ktime) {
    std::fprintf(stderr,
        "[TRAIN] KTIME: gather=%.2fs forward=%.2fs gradW1=%.2fs gradW2=%.2fs\n",
        t_gather, t_fwd, t_gw1, t_gw2);
  }
  double et_time = NowSec() - et_t0;
  const clio::run::u64 peak_win = (clio::run::u64)window * page_size;
  std::fprintf(stderr,
      "[TRAIN] ETERNIA: epoch0 loss=%.6f acc=%.4f -> epoch%d loss=%.6f acc=%.4f val_acc=%.4f (%.2fs)  "
      "peak GPU window=%lluMiB memcpy_pin_gbps=%.2f memcpy_page_gbps=%.2f\n",
      et_loss[0], et_acc[0], epochs - 1,
      et_loss[epochs - 1], et_acc[epochs - 1], et_vacc[epochs - 1], et_time,
      (unsigned long long)(peak_win >> 20), mcp.pinned_gbps,
      mcp.pageable_gbps);

  // ---- Did the GPU actually read the right bytes? --------------------
  //
  // In stream mode there is no in-core baseline, so a PASS here otherwise means
  // only "the run completed" -- which is how a read path that returned wrong
  // bytes 2 runs in 3 was once reported as a successful papers100M result.
  // This samples pages and pulls each one BOTH ways, through the GPU vector's
  // fault path and through an ordinary CPU GetBlob, and requires agreement.
  //
  // KNOW WHAT THIS DOES NOT CATCH. It reads one page per launch, so the cache
  // is never under pressure and nothing is evicted while a lane still holds it.
  // Checked directly: with the known-bad at() read path it still reports 32/32
  // agreeing, because that bug needs a full sweep to surface. This catches
  // wrong page mapping, a page never stored, and gross corruption -- not
  // concurrency. The bit-exactness assertion against the in-core baseline is
  // what covers the sweep, and it is the check that actually found at().
  {
    const int nver = (int)EnvI64("CLIO_GNN_VERIFY_PAGES", 8);
    if (nver > 0 && K > 0) {
      ctp::ipc::FullPtr<char> cpu = CLIO_IPC->AllocateBuffer((size_t)page_size);
      REQUIRE(!cpu.IsNull());
      std::vector<float> gpu_rows((size_t)epp);
      std::uint64_t st = 0x243F6A8885A308D3ull;
      int checked = 0, bad = 0;
      for (int i = 0; i < nver; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        const clio::run::u64 pg = st % K;
        const clio::run::u64 lo = pg * epp;

        RunYieldable(1, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                            gy::YieldStackView sv) {
          GnnGatherKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
              gpu_info, vec.GetDevice(0), lo, lo + epp, d_scratch, 1, vw, sv);
        });
        ctp::GpuApi::Synchronize();
        ctp::GpuApi::Memcpy(gpu_rows.data(), d_scratch,
                            (size_t)(epp * sizeof(float)));

        const std::string nm = std::to_string(pg);
        auto gf = comp.AsyncGetBlob(tag_id, nm, (clio::run::u64)0, page_size, 0,
                                    cpu.shm_.template Cast<void>(),
                                    clio::run::PoolQuery::Local());
        gf.Wait();
        if (gf->GetReturnCode() != 0) continue;  // page never stored
        ++checked;
        if (std::memcmp(gpu_rows.data(), cpu.ptr_, (size_t)page_size) != 0) {
          ++bad;
        }
      }
      CLIO_IPC->FreeBuffer(cpu);
      std::fprintf(stderr,
                   "[TRAIN] read verify: %d/%d sampled pages agree between the "
                   "GPU vector and a CPU GetBlob%s\n",
                   checked - bad, checked, bad ? "  *** MISMATCH ***" : "");
      REQUIRE(bad == 0);
    }
  }

  std::fprintf(stderr,
      "[TRAIN] PEAK GPU (whole process, incl. labels + page cache + runtime): "
      "%lluMiB  (feature window alone: %lluMiB)\n",
      (unsigned long long)(g_peak_gpu >> 20),
      (unsigned long long)(2ull * window * page_size >> 20));

  // ---- Compare curves (bit-exact where in-core ran) ----
  std::string exact = "n/a(OOM)";
  if (base_status == "OK") {
    double maxdl = 0, maxda = 0, maxdv = 0;
    for (int e = 0; e < epochs; ++e) {
      maxdl = std::max(maxdl, std::fabs(base_loss[e] - et_loss[e]));
      maxda = std::max(maxda, std::fabs(base_acc[e] - et_acc[e]));
      maxdv = std::max(maxdv, std::fabs(base_vacc[e] - et_vacc[e]));
    }
    exact = (maxdl == 0.0 && maxda == 0.0 && maxdv == 0.0) ? "BIT-EXACT" : "CLOSE";
    std::fprintf(stderr, "[TRAIN] VERIFY vs in-core: max|dloss|=%.3e max|dacc|=%.3e max|dvacc|=%.3e -> %s\n",
                 maxdl, maxda, maxdv, exact.c_str());
    std::fprintf(stderr, "[TRAIN] per-epoch (in-core || eternia):\n");
    for (int e = 0; e < epochs; ++e)
      std::fprintf(stderr, "   e%02d  loss %.6f acc %.4f  ||  loss %.6f acc %.4f\n",
                   e, base_loss[e], base_acc[e], et_loss[e], et_acc[e]);
    // A LOSSY codec cannot be bit-exact, by construction. Demanding it here
    // just makes the test unusable with cuSZp, so the bar depends on the
    // codec: lossless must reproduce the in-core run exactly, lossy must
    // TRACK it -- the training curve is what matters, not the last bit of a
    // feature. Tolerance is on the metrics rather than the features because
    // that is the property anyone actually relies on.
    if (IsLossyCodec(comp_lib)) {
      const double tol = EnvI64("CLIO_GNN_LOSSY_TOL_E6", 1000) / 1e6;  // 1e-3
      std::fprintf(stderr,
                   "[TRAIN] lossy codec %d: requiring curves to track within "
                   "%.1e (not bit-exactness)\n", comp_lib, tol);
      REQUIRE(maxdl < tol);
      REQUIRE(maxda < tol);
      REQUIRE(maxdv < tol);
    } else {
      REQUIRE(exact == "BIT-EXACT");
    }
  } else {
    if (ingest_file != nullptr)
      std::fprintf(stderr, "[TRAIN] in-core SKIPPED (stream mode) -> trained the "
                   "%lluMiB matrix straight from the vector. Final train_acc=%.4f "
                   "val_acc=%.4f\n",
                   (unsigned long long)(dataset_bytes >> 20), et_acc[epochs - 1],
                   et_vacc[epochs - 1]);
    else
    std::fprintf(stderr, "[TRAIN] in-core OOM'd -> Eternia is the ONLY method that TRAINED this "
                 "%lluMiB feature matrix. Final train_acc=%.4f val_acc=%.4f\n",
                 (unsigned long long)(dataset_bytes >> 20), et_acc[epochs - 1], et_vacc[epochs - 1]);
    // Only meaningful with more than one epoch: at epochs==1 this compares
    // et_acc[0] with itself and fails for a perfectly valid configuration.
    if (epochs > 1) {
      REQUIRE(et_acc[epochs - 1] > et_acc[0]);  // training made progress
    }
  }
  std::fprintf(stderr, "====================================================\n");

  const char *csv = std::getenv("CLIO_GNN_CSV");
  if (csv) {
    std::ifstream probe(csv); bool empty = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    probe.close(); std::ofstream out(csv, std::ios::app);
    const char *lib_e = std::getenv("CLIO_CTE_COMPRESS_LIB");
    const char *pre_e = std::getenv("CLIO_CTE_COMPRESS_PRESET");
    if (out) { if (empty) out << "data,lib,preset,A_mib,nodes,H,epochs,base_status,base_final_acc,"
                                 "base_final_vacc,eternia_final_acc,eternia_final_vacc,stored_mib,ratio,"
                                 "peak_gpu_mib,store_s,base_epoch_s,eternia_epoch_s,bit_exact\n";
      out << dir << "," << (lib_e ? lib_e : "default") << "," << (pre_e ? pre_e : "default") << ","
          << (dataset_bytes >> 20) << "," << N << "," << H << "," << epochs << "," << base_status << ","
          << (base_status == "OK" ? base_acc[epochs - 1] : -1.0) << ","
          << (base_status == "OK" ? base_vacc[epochs - 1] : -1.0) << ","
          << et_acc[epochs - 1] << "," << et_vacc[epochs - 1] << ","
          << (stored >> 20) << "," << ratio << "," << (peak_win >> 20) << "," << store_dt << ","
          << (base_status == "OK" ? base_time / epochs : -1.0) << "," << et_time / epochs << ","
          << exact << "\n"; }
  }

  ctp::GpuApi::Free(d_vcorrect); ctp::GpuApi::Free(d_vcount);
  ctp::GpuApi::Free(d_lab); ctp::GpuApi::Free(dW1); ctp::GpuApi::Free(db1); ctp::GpuApi::Free(dW2); ctp::GpuApi::Free(db2);
  ctp::GpuApi::Free(gW1); ctp::GpuApi::Free(gb1); ctp::GpuApi::Free(gW2); ctp::GpuApi::Free(gb2);
  ctp::GpuApi::Free(h1_buf); ctp::GpuApi::Free(dz1_buf); ctp::GpuApi::Free(dz2_buf); ctp::GpuApi::Free(d_scratch);
  ctp::GpuApi::Free(loss_buf); ctp::GpuApi::Free(d_loss); ctp::GpuApi::Free(d_correct); ctp::GpuApi::Free(d_count);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif
