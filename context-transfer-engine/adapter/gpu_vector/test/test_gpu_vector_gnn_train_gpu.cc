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

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;

/**
 * Gather the element range [lo,hi) of the paged vector into `scratch`.
 *
 * YIELDABLE, and it has to be: every page in the range is a fault that the
 * runtime must service, and the blocking HoldPage deadlocks against itself
 * there -- the kernel would hold the SM waiting for a fetch that only a later
 * launch could perform. This replaces the old PrefetchPagesSync + read_range
 * pair; on-demand faulting through HoldPageYield does the same work without a
 * separate prefetch call (which the rewrite removed).
 *
 * Runs on `nblocks` blocks, each taking a contiguous slice of [lo,hi) keyed on
 * yv.Block() -- NOT blockIdx.x, which stops matching the logical block once the
 * driver relaunches a compacted grid. Every lane runs the loop because the
 * fault path is block-collective, and a block whose slice is empty must still
 * fall through to CLIO_YEND rather than return early.
 */
__global__ void GnnGatherKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> v, clio::run::u64 lo,
                                clio::run::u64 hi, float *scratch,
                                clio::run::u32 nblocks, gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, i, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  // Slice off yv.Block(), never blockIdx.x: the driver relaunches a COMPACTED
  // grid, so blockIdx.x stops matching the logical block after the first yield.
  const clio::run::u64 total = hi - lo;
  const clio::run::u64 chunk = (total + nblocks - 1) / nblocks;
  const clio::run::u64 base = yv.Block() * chunk;
  const clio::run::u64 n = base >= total ? 0 : min(chunk, total - base);
  lo += base;
  scratch += base;

  CLIO_YBEGIN();
  for (; i < n; i += run) {
    CLIO_YCALL(v.HoldPageYield(lo + i, n - i, &run));
    // BULK COPY off the page pointer, not element-by-element.
    //
    // at() re-derives the page base and the in-page index on EVERY element.
    // The page is already resolved and pinned by the HoldPageYield above, so
    // TryHoldRawConst hands back the base directly (read-only, so it does not
    // dirty the page the way HoldRaw would) and the run becomes a flat memcpy.
    // Copying 4 floats per thread lets the compiler emit 128-bit loads and
    // stores, which is what turns this from pointer-chasing into bandwidth.
    {
      clio::run::u64 probe = 0;
      const float *src = v.TryHoldRawConst(lo + i, n - i, &probe);
      if (src != nullptr && probe >= run) {
        float *dst = scratch + i;
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
      } else {
        // Miss on the probe: fall back to the scalar accessor. at() is
        // read-only; operator[] would mark the page dirty, which blocks the
        // run-fetch from claiming slots and forces single-page paging.
        for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
          scratch[i + k] = v.at(lo + i + k);
        }
      }
    }
  }
  CLIO_YEND();
}

/** Forward + backward for the window's nodes (one thread per node). Writes per-node
 *  h1, dz1, dz2 to window buffers; accumulates loss/correct/count. */
__global__ void TrainNodeKernel(const float *A, clio::run::u64 nn, clio::run::u64 node_lo,
                                const long long *labels, int F, int H, int C,
                                const float *W1, const float *b1, const float *W2,
                                const float *b2, float *h1_out, float *dz1_out,
                                float *dz2_out, double *loss_buf, int *correct, int *count,
                                int *val_correct, int *val_count) {
  clio::run::u64 n = blockIdx.x * (clio::run::u64)blockDim.x + threadIdx.x;
  if (n >= nn) return;
  long long y = labels[node_lo + n];
  loss_buf[n] = 0.0;
  for (int c = 0; c < C; ++c) dz2_out[n * C + c] = 0.0f;
  for (int h = 0; h < H; ++h) { h1_out[n * H + h] = 0.0f; dz1_out[n * H + h] = 0.0f; }
  if (y < 0) return;
  const float *a = A + n * (clio::run::u64)F;
  // PER-THREAD SCRATCH IS THE COST HERE. This kernel had five arrays --
  // z1, h1, z2, p, dz2 -- at kMaxH/kMaxC, which is 1088 floats (4352 B) of
  // LOCAL memory per thread whatever H and C are. Three of them are
  // redundant, and removing them changes no arithmetic:
  //   z1 : only ever used as (z1[h] > 0), and h1 = max(z1,0), so h1[h] > 0
  //        is the same predicate exactly.
  //   p  : derived from z2 and z2 is dead afterwards, so p aliases z2.
  //   dz2: derived from p, and the loss is read BEFORE it, so it aliases z2
  //        too.
  // 1088 floats -> 448.
  float h1[kMaxH];
  // LOOP ORDER: f OUTSIDE, h inside. Written the natural way -- h outer,
  // accumulating one h at a time -- this re-reads the node's whole feature row
  // a[0..F) once PER h, so each 512-byte row is fetched H=64 times: 16 GiB per
  // window, measured at ~170 GB/s, i.e. saturating HBM to move the same bytes
  // over and over. Hoisting f outside reads each element ONCE and updates all
  // H accumulators from it.
  //
  // The arithmetic is IDENTICAL, not merely equivalent: h1[h] still accumulates
  // W1[h*F+f]*a[f] over f in ascending order, exactly as before, so every sum
  // is formed from the same terms in the same sequence. W1 is only H*F floats
  // and stays in cache across the f loop.
  for (int h = 0; h < H; ++h) h1[h] = b1[h];
  for (int f = 0; f < F; ++f) {
    const float av = a[f];
    for (int h = 0; h < H; ++h) h1[h] += W1[h * F + f] * av;
  }
  for (int h = 0; h < H; ++h) h1[h] = h1[h] > 0.f ? h1[h] : 0.f;
  float z2[kMaxC], maxz = -1e30f;
  for (int c = 0; c < C; ++c) {
    float s = b2[c];
    for (int h = 0; h < H; ++h) s += W2[c * H + h] * h1[h];
    z2[c] = s; if (s > maxz) maxz = s;
  }
  float sum = 0.f;
  for (int c = 0; c < C; ++c) { z2[c] = expf(z2[c] - maxz); sum += z2[c]; }
  int pred = 0; float pmax = -1.f;   // p aliases z2
  for (int c = 0; c < C; ++c) { z2[c] = z2[c] / sum; if (z2[c] > pmax) { pmax = z2[c]; pred = c; } }
  // Held-out validation node: score it, but contribute neither loss nor gradient.
  // dz1_out/dz2_out/h1_out were zeroed above, so returning here keeps its grad at 0.
  if (((node_lo + n) % kValStride) == (kValStride - 1)) {
    if (pred == (int)y) atomicAdd(val_correct, 1);
    atomicAdd(val_count, 1);
    return;
  }
  loss_buf[n] = -log((double)fmaxf(z2[y], 1e-30f));
  if (pred == (int)y) atomicAdd(correct, 1);
  atomicAdd(count, 1);
  // dz2 aliases z2 (which currently holds p); the loss above already read it.
  for (int c = 0; c < C; ++c) { z2[c] = z2[c] - (c == (int)y ? 1.f : 0.f); dz2_out[n * C + c] = z2[c]; }
  for (int h = 0; h < H; ++h) {
    float s = 0.f;
    for (int c = 0; c < C; ++c) s += W2[c * H + h] * z2[c];
    dz1_out[n * H + h] = (h1[h] > 0.f) ? s : 0.f;   // h1>0 iff z1>0
    h1_out[n * H + h] = h1[h];
  }
}

/** Grad accumulation (one thread per weight element; sums the window's nodes in
 *  order and += into the global double grad -- deterministic across windows because
 *  the launches serialize on one stream). */
/** Number of node-range splits for the dW1 reduction. See GradW1Kernel. */
constexpr int kGradSplits = 32;

/**
 * dW1[h][f] += sum_n dz1[n][h] * A[n][f], split over n.
 *
 * The natural form gives one thread the whole reduction for its (h,f), which
 * is only H*F = 8192 threads -- far too few to fill this GPU, and it measured
 * ~11% of FP64 peak while being 9.0s of a 22.6s run. Splitting the node range
 * kGradSplits ways multiplies the thread count by 32 without changing the
 * arithmetic each thread does.
 *
 * STILL FP64 and STILL DETERMINISTIC. Each thread sums a contiguous ascending
 * n-range in double, and GradW1ReduceKernel folds the partials in fixed split
 * order. No atomics: an atomicAdd over doubles would make the order
 * run-dependent, and the test asserts the streamed run reproduces the in-core
 * run exactly. The order differs from the unsplit version, but both the
 * in-core and the vector path run this same kernel, so they still agree
 * bit-for-bit.
 *
 * Tiling this through shared memory was tried instead and is SLOWER (1.37s vs
 * 1.12s at 1 GiB): the kernel is arithmetic-bound, not bandwidth-bound, so
 * staging buys nothing. Its global reads are already coalesced -- consecutive
 * idx give consecutive f, so a warp reads 32 contiguous floats of A.
 */
__global__ void GradW1Kernel(const float *A, const float *dz1, clio::run::u64 nn,
                             int F, int H, double *part, double *partb) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= H * F) return;
  const int sp = blockIdx.y;
  const int h = idx / F, f = idx % F;
  const clio::run::u64 chunk = (nn + kGradSplits - 1) / kGradSplits;
  const clio::run::u64 lo = (clio::run::u64)sp * chunk;
  const clio::run::u64 hi = min(lo + chunk, nn);
  double s = 0.0;
  for (clio::run::u64 n = lo; n < hi; ++n) {
    s += (double)dz1[n * H + h] * (double)A[n * F + f];
  }
  part[(clio::run::u64)sp * H * F + idx] = s;
  if (f == 0) {
    double sb = 0.0;
    for (clio::run::u64 n = lo; n < hi; ++n) sb += (double)dz1[n * H + h];
    partb[(clio::run::u64)sp * H + h] = sb;
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
 * dW2[c][h] += sum_n dz2[n][c] * h1[n][h], split over n like GradW1Kernel.
 *
 * C*H is only 2560 threads here, and a shared-memory tiled version launched
 * just 20 blocks. Splitting the node range gives 32x the threads for the same
 * per-thread arithmetic, and beats the tiled form (which was itself an
 * improvement on the naive one). Same determinism argument: contiguous
 * ascending n-ranges in double, folded in fixed split order, no atomics.
 */
__global__ void GradW2Kernel(const float *h1, const float *dz2, clio::run::u64 nn,
                             int H, int C, double *part, double *partb) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * H) return;
  const int sp = blockIdx.y;
  const int c = idx / H, h = idx % H;
  const clio::run::u64 chunk = (nn + kGradSplits - 1) / kGradSplits;
  const clio::run::u64 lo = (clio::run::u64)sp * chunk;
  const clio::run::u64 hi = min(lo + chunk, nn);
  double s = 0.0;
  for (clio::run::u64 n = lo; n < hi; ++n) {
    s += (double)dz2[n * C + c] * (double)h1[n * H + h];
  }
  part[(clio::run::u64)sp * C * H + idx] = s;
  if (h == 0) {
    double sb = 0.0;
    for (clio::run::u64 n = lo; n < hi; ++n) sb += (double)dz2[n * C + c];
    partb[(clio::run::u64)sp * C + c] = sb;
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
__global__ void LossReduceKernel(const double *loss_buf, clio::run::u64 nn, double *d_loss) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double s = 0.0;
  for (clio::run::u64 n = 0; n < nn; ++n) s += loss_buf[n];
  d_loss[0] += s;
}

#if !CTP_IS_DEVICE_PASS

namespace {
using gnn_test::EnvI64;

/** Run a yieldable kernel to completion (re-launch until no block is pending). */
template <typename LaunchT>
clio::run::u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  const unsigned nthreads =
      (unsigned)EnvI64("CLIO_GNN_GATHER_THREADS", 256);
  gy::Yieldable<> drv(nblocks, nthreads);
  gy::YieldStack stack(nblocks, nthreads, 256);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000);
}
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
  long long *d_lab = nullptr; REQUIRE(cudaMalloc(&d_lab, N * sizeof(long long)) == cudaSuccess);
  { std::vector<long long> tile(std::min<clio::run::u64>(N, 1u << 22));
    for (clio::run::u64 off = 0; off < N; off += tile.size()) {
      clio::run::u64 n = std::min<clio::run::u64>(tile.size(), N - off);
      for (clio::run::u64 i = 0; i < n; ++i) tile[i] = lab0[(off + i) % N0];
      cudaMemcpy(d_lab + off, tile.data(), n * sizeof(long long), cudaMemcpyHostToDevice);
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
  float *h1_buf, *dz1_buf, *dz2_buf, *d_scratch;
  REQUIRE(cudaMalloc(&dW1, H * F * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&db1, H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dW2, C * H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&db2, C * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gW1, H * F * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gw1_part,
                     (size_t)kGradSplits * H * F * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gb1_part,
                     (size_t)kGradSplits * H * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gw2_part,
                     (size_t)kGradSplits * C * H * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gb2_part,
                     (size_t)kGradSplits * C * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gb1, H * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gW2, C * H * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gb2, C * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&h1_buf, max_nn * H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dz1_buf, max_nn * H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dz2_buf, max_nn * C * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_scratch, max_nn * F * sizeof(float)) == cudaSuccess);
  double *loss_buf; REQUIRE(cudaMalloc(&loss_buf, max_nn * sizeof(double)) == cudaSuccess);
  double *d_loss; int *d_correct, *d_count, *d_vcorrect, *d_vcount;
  REQUIRE(cudaMalloc(&d_loss, sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_correct, sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_count, sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_vcorrect, sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_vcount, sizeof(int)) == cudaSuccess);

  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  // One epoch over a window provider. get_win(win_idx, first_node, nn) fills
  // d_scratch (nn*F). Returns final (loss,acc). Deterministic accumulation.
  auto run_epoch = [&](std::function<void(clio::run::u64, clio::run::u64, clio::run::u64)> gather,
                       double &out_loss, double &out_acc, clio::run::u64 &out_count,
                       double &out_vacc) {
    cudaMemset(gW1, 0, H * F * sizeof(double)); cudaMemset(gb1, 0, H * sizeof(double));
    cudaMemset(gW2, 0, C * H * sizeof(double)); cudaMemset(gb2, 0, C * sizeof(double));
    cudaMemset(d_loss, 0, sizeof(double)); cudaMemset(d_correct, 0, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    cudaMemset(d_vcorrect, 0, sizeof(int)); cudaMemset(d_vcount, 0, sizeof(int));

    // Apply one SGD step from the currently accumulated gradients, normalised by
    // `nrm` contributing nodes, then (in mini-batch mode) clear them for the next
    // batch. Weights are tiny (H*F + C*H floats), so the host round-trip is cheap.
    auto apply_step = [&](int nrm, bool clear_grads) {
      std::vector<double> aW1(H * F), ab1(H), aW2(C * H), ab2(C);
      cudaMemcpy(aW1.data(), gW1, H * F * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(ab1.data(), gb1, H * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(aW2.data(), gW2, C * H * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(ab2.data(), gb2, C * sizeof(double), cudaMemcpyDeviceToHost);
      std::vector<float> aw1(H * F), ab1f(H), aw2(C * H), ab2f(C);
      cudaMemcpy(aw1.data(), dW1, H * F * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(ab1f.data(), db1, H * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(aw2.data(), dW2, C * H * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(ab2f.data(), db2, C * sizeof(float), cudaMemcpyDeviceToHost);
      float ainv = 1.0f / (float)std::max<int>(1, nrm);
      for (int i = 0; i < H * F; ++i) aw1[i] -= lr * (float)(aW1[i] * ainv);
      for (int i = 0; i < H; ++i) ab1f[i] -= lr * (float)(ab1[i] * ainv);
      for (int i = 0; i < C * H; ++i) aw2[i] -= lr * (float)(aW2[i] * ainv);
      for (int i = 0; i < C; ++i) ab2f[i] -= lr * (float)(ab2[i] * ainv);
      cudaMemcpy(dW1, aw1.data(), H * F * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(db1, ab1f.data(), H * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(dW2, aw2.data(), C * H * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(db2, ab2f.data(), C * sizeof(float), cudaMemcpyHostToDevice);
      if (clear_grads) {
        cudaMemset(gW1, 0, H * F * sizeof(double)); cudaMemset(gb1, 0, H * sizeof(double));
        cudaMemset(gW2, 0, C * H * sizeof(double)); cudaMemset(gb2, 0, C * sizeof(double));
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
      TrainNodeKernel<<<(int)((nn + tpb - 1) / tpb), tpb>>>(
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
      LossReduceKernel<<<1, 1>>>(loss_buf, nn, d_loss);
      if (ktime) { ctp::GpuApi::Synchronize(); t_fwd += NowSec() - tp0; tp0 = NowSec(); }
      track_peak();
      GradW1Kernel<<<dim3((H * F + tpb - 1) / tpb, kGradSplits), tpb>>>(
          d_scratch, dz1_buf, nn, F, H, gw1_part, gb1_part);
      GradW1ReduceKernel<<<(H * F + tpb - 1) / tpb, tpb>>>(
          gw1_part, gb1_part, F, H, gW1, gb1);
      if (ktime) { ctp::GpuApi::Synchronize(); t_gw1 += NowSec() - tp0; tp0 = NowSec(); }
      GradW2Kernel<<<dim3((C * H + tpb - 1) / tpb, kGradSplits), tpb>>>(
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
        cudaMemcpy(&cnt_now, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        apply_step(cnt_now - prev_cnt, /*clear_grads=*/true);
        prev_cnt = cnt_now;
      }
    }
    ctp::GpuApi::Synchronize();
    // SGD update on host (weights are small).
    std::vector<double> hW1(H * F), hb1(H), hW2(C * H), hb2(C); double hloss; int hcorr, hcnt;
    cudaMemcpy(hW1.data(), gW1, H * F * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb1.data(), gb1, H * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hW2.data(), gW2, C * H * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb2.data(), gb2, C * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hloss, d_loss, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hcorr, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hcnt, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    int hvcorr = 0, hvcnt = 0;
    cudaMemcpy(&hvcorr, d_vcorrect, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hvcnt, d_vcount, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<float> W1(H * F), b1(H), W2(C * H), b2(C);
    cudaMemcpy(W1.data(), dW1, H * F * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1.data(), db1, H * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W2.data(), dW2, C * H * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2.data(), db2, C * sizeof(float), cudaMemcpyDeviceToHost);
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
    cudaMemcpy(dW1, W1.data(), H * F * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db1, b1.data(), H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW2, W2.data(), C * H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db2, b2.data(), C * sizeof(float), cudaMemcpyHostToDevice);
    out_loss = hloss / std::max<int>(1, hcnt); out_acc = (double)hcorr / std::max<int>(1, hcnt);
    out_count = (clio::run::u64)hcnt;
    out_vacc = hvcnt ? (double)hvcorr / (double)hvcnt : 0.0;
  };

  auto reset_weights = [&]() {
    cudaMemcpy(dW1, W1_0.data(), H * F * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db1, b1_0.data(), H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW2, W2_0.data(), C * H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db2, b2_0.data(), C * sizeof(float), cudaMemcpyHostToDevice);
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
        cudaMemcpy(d_all + off, A0.data(), n * sizeof(float), cudaMemcpyHostToDevice);
      }
      reset_weights();
      double t0 = NowSec();
      for (int e = 0; e < epochs; ++e) {
        clio::run::u64 cnt;
        run_epoch([&](clio::run::u64, clio::run::u64 first, clio::run::u64 nn) {
          cudaMemcpy(d_scratch, d_all + first * F, nn * F * sizeof(float), cudaMemcpyDeviceToDevice);
        }, base_loss[e], base_acc[e], cnt, base_vacc[e]);
      }
      base_time = NowSec() - t0; base_status = "OK";
      cudaFree(d_all);
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
    // The vector's fault path asks for "p<page_num>" (PageBlobName in page.h).
    // The old "<tag>_b0_pi<n>" scheme is from the pre-rewrite vector; storing
    // under it would leave every page a miss with no visible error.
    std::string name = "p" + std::to_string(p);
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
                        comp_lib, comp_preset, clio::cte::core::kCtePoolId);
  vec.EnableStats();
  // Tell the device side how big each stored page is, so an encoded page is
  // fetched with its true (compressed) length rather than the logical one.
  for (int attempt = 0; attempt < 50 && vec.PublishStoredSizes() < K; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::vector<double> et_loss(epochs), et_acc(epochs), et_vacc(epochs);
  reset_weights();
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
      RunYieldable(gather_blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
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
        "prefetches=%llu pf_hits=%llu pf_late=%llu pf_dropped=%llu "
        "verify_lost=%llu get_err=%llu put_err=%llu\n",
        (unsigned long long)vs.faults, need,
        need ? (double)vs.faults / (double)need : 0.0,
        (unsigned long long)vs.evicts, (unsigned long long)vs.puts,
        (unsigned long long)vs.prefetches,
        (unsigned long long)vs.prefetch_hits, (unsigned long long)vs.prefetch_late,
        (unsigned long long)vs.pf_dropped, (unsigned long long)vs.verify_lost,
        (unsigned long long)vs.get_errors, (unsigned long long)vs.put_errors);
  }
  {
    // Per-page fetch counts (CLIO_FAULT_HIST=1). The workload reads each page
    // exactly `epochs` times, so any page fetched more than that was evicted
    // before it was used -- this says WHICH pages churn, not just how many.
    auto fh = vec.ReadFaultHist(0);
    if (!fh.empty()) {
      unsigned long long tot = 0, over = 0, zero = 0, mx = 0;
      for (auto v : fh) {
        tot += v;
        if (v == 0) ++zero;
        if (v > (unsigned)epochs) over += (v - (unsigned)epochs);
        if (v > mx) mx = v;
      }
      std::fprintf(stderr,
          "[TRAIN] FAULT HIST: pages=%zu never_fetched=%llu total_fetches=%llu "
          "wasted=%llu (%.1f%%) max_per_page=%llu ideal=%llu\n",
          fh.size(), zero, tot, over,
          tot ? 100.0 * (double)over / (double)tot : 0.0, mx,
          (unsigned long long)fh.size() * (unsigned long long)epochs);
    }
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
      "peak GPU window=%lluMiB\n", et_loss[0], et_acc[0], epochs - 1,
      et_loss[epochs - 1], et_acc[epochs - 1], et_vacc[epochs - 1], et_time,
      (unsigned long long)(peak_win >> 20));

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
  // concurrency. The bit-exactness assertion in run_pipeline_test.sh is what
  // covers the sweep, and it is the check that actually found at().
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
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaMemcpy(gpu_rows.data(), d_scratch,
                           (size_t)(epp * sizeof(float)),
                           cudaMemcpyDeviceToHost) == cudaSuccess);

        char nm[32];
        clio::cte::gpu_vector::PageBlobName(pg, nm);
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

  cudaFree(d_vcorrect); cudaFree(d_vcount);
  cudaFree(d_lab); cudaFree(dW1); cudaFree(db1); cudaFree(dW2); cudaFree(db2);
  cudaFree(gW1); cudaFree(gb1); cudaFree(gW2); cudaFree(gb2);
  cudaFree(h1_buf); cudaFree(dz1_buf); cudaFree(dz2_buf); cudaFree(d_scratch);
  cudaFree(loss_buf); cudaFree(d_loss); cudaFree(d_correct); cudaFree(d_count);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif
