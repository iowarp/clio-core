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
      << "      - path: \"hbm::cte_hbm_tier\"\n        bdev_type: \"hbm\"\n        capacity_limit: \""
      << hbm_mib << "MB\"\n        score: 1.0\n"
      << "      - path: \"ram::cte_dram_tier\"\n        bdev_type: \"ram\"\n        capacity_limit: \""
      << dram_mib << "MB\"\n        score: 0.5\n"
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
 * Single block by construction, so yv.Block() is 0; every lane still runs the
 * loop because the fault path is block-collective.
 */
__global__ void GnnGatherKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> v, clio::run::u64 lo,
                                clio::run::u64 hi, float *scratch,
                                gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  CLIO_YKERNEL_ENTER(yv, ys);
  CLIO_YFRAME();
  CLIO_YLOCAL_INIT(clio::run::u64, i, 0);
  CLIO_YLOCAL_INIT(clio::run::u64, run, 0);
  const clio::run::u64 n = hi - lo;

  CLIO_YBEGIN();
  for (; i < n; i += run) {
    CLIO_YCALL(v.HoldPageYield(lo + i, n - i, &run));
    for (clio::run::u64 k = threadIdx.x; k < run; k += blockDim.x) {
      scratch[i + k] = v[lo + i + k];
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
  float z1[kMaxH], h1[kMaxH];
  for (int h = 0; h < H; ++h) {
    float s = b1[h];
    for (int f = 0; f < F; ++f) s += W1[h * F + f] * a[f];
    z1[h] = s; h1[h] = s > 0.f ? s : 0.f;
  }
  float z2[kMaxC], maxz = -1e30f;
  for (int c = 0; c < C; ++c) {
    float s = b2[c];
    for (int h = 0; h < H; ++h) s += W2[c * H + h] * h1[h];
    z2[c] = s; if (s > maxz) maxz = s;
  }
  float sum = 0.f;
  for (int c = 0; c < C; ++c) { z2[c] = expf(z2[c] - maxz); sum += z2[c]; }
  int pred = 0; float pmax = -1.f; float p[kMaxC];
  for (int c = 0; c < C; ++c) { p[c] = z2[c] / sum; if (p[c] > pmax) { pmax = p[c]; pred = c; } }
  // Held-out validation node: score it, but contribute neither loss nor gradient.
  // dz1_out/dz2_out/h1_out were zeroed above, so returning here keeps its grad at 0.
  if (((node_lo + n) % kValStride) == (kValStride - 1)) {
    if (pred == (int)y) atomicAdd(val_correct, 1);
    atomicAdd(val_count, 1);
    return;
  }
  loss_buf[n] = -log((double)fmaxf(p[y], 1e-30f));
  if (pred == (int)y) atomicAdd(correct, 1);
  atomicAdd(count, 1);
  float dz2[kMaxC];
  for (int c = 0; c < C; ++c) { dz2[c] = p[c] - (c == (int)y ? 1.f : 0.f); dz2_out[n * C + c] = dz2[c]; }
  for (int h = 0; h < H; ++h) {
    float s = 0.f;
    for (int c = 0; c < C; ++c) s += W2[c * H + h] * dz2[c];
    dz1_out[n * H + h] = (z1[h] > 0.f) ? s : 0.f;
    h1_out[n * H + h] = h1[h];
  }
}

/** Grad accumulation (one thread per weight element; sums the window's nodes in
 *  order and += into the global double grad -- deterministic across windows because
 *  the launches serialize on one stream). */
__global__ void GradW1Kernel(const float *A, const float *dz1, clio::run::u64 nn,
                             int F, int H, double *dW1, double *db1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= H * F) return;
  int h = idx / F, f = idx % F;
  double s = 0.0;
  for (clio::run::u64 n = 0; n < nn; ++n) s += (double)dz1[n * H + h] * (double)A[n * F + f];
  dW1[idx] += s;
  if (f == 0) { double sb = 0.0; for (clio::run::u64 n = 0; n < nn; ++n) sb += (double)dz1[n * H + h]; db1[h] += sb; }
}
__global__ void GradW2Kernel(const float *h1, const float *dz2, clio::run::u64 nn,
                             int H, int C, double *dW2, double *db2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * H) return;
  int c = idx / H, h = idx % H;
  double s = 0.0;
  for (clio::run::u64 n = 0; n < nn; ++n) s += (double)dz2[n * C + c] * (double)h1[n * H + h];
  dW2[idx] += s;
  if (h == 0) { double sb = 0.0; for (clio::run::u64 n = 0; n < nn; ++n) sb += (double)dz2[n * C + c]; db2[c] += sb; }
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
  gy::Yieldable<> drv(nblocks, 32);
  gy::YieldStack stack(nblocks, 32, 256);
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
  float *h1_buf, *dz1_buf, *dz2_buf, *d_scratch;
  REQUIRE(cudaMalloc(&dW1, H * F * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&db1, H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dW2, C * H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&db2, C * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&gW1, H * F * sizeof(double)) == cudaSuccess);
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
      gather(w, first, nn);   // fills d_scratch with nn*F floats
      int tpb = 128;
      TrainNodeKernel<<<(int)((nn + tpb - 1) / tpb), tpb>>>(
          d_scratch, nn, first, d_lab, F, H, C, dW1, db1, dW2, db2,
          h1_buf, dz1_buf, dz2_buf, loss_buf, d_correct, d_count,
          d_vcorrect, d_vcount);
      LossReduceKernel<<<1, 1>>>(loss_buf, nn, d_loss);
      track_peak();
      GradW1Kernel<<<(H * F + tpb - 1) / tpb, tpb>>>(d_scratch, dz1_buf, nn, F, H, gW1, gb1);
      GradW2Kernel<<<(C * H + tpb - 1) / tpb, tpb>>>(h1_buf, dz2_buf, nn, H, C, gW2, gb2);
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

  // One block, a cache of 2*window pages, spilling through the compressor.
  gv::Vector<float> vec(vec_tag, {0}, page_size, /*nblocks=*/1,
                        /*pages_per_block=*/2 * window, K * epp, StoragePool(),
                        comp_lib, comp_preset, clio::cte::core::kCtePoolId);
  vec.EnableStats();
  // Tell the device side how big each stored page is, so an encoded page is
  // fetched with its true (compressed) length rather than the logical one.
  for (int attempt = 0; attempt < 50 && vec.PublishStoredSizes() < K; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::vector<double> et_loss(epochs), et_acc(epochs), et_vacc(epochs);
  reset_weights();
  double et_t0 = NowSec();
  for (int e = 0; e < epochs; ++e) {
    clio::run::u64 cnt;
    // Stream A windows through the vector each epoch: prefetch (decompress) each
    // window into HBM slots, gather its ELEMENTS [first*F, (first+nn)*F) into
    // d_scratch, then run the same train kernels as the in-core path.
    run_epoch([&](clio::run::u64 wi, clio::run::u64 first, clio::run::u64 nn) {
      (void)wi;  // pages are faulted on demand now; no separate prefetch step
      RunYieldable(1, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                          gy::YieldStackView sv) {
        GnnGatherKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
            gpu_info, vec.GetDevice(0), first * (clio::run::u64)F,
            (first + nn) * (clio::run::u64)F, d_scratch, vw, sv);
      });
      ctp::GpuApi::Synchronize();
    }, et_loss[e], et_acc[e], cnt, et_vacc[e]);
    std::fprintf(stderr, "[TRAIN]   eternia e%02d loss=%.6f acc=%.4f val_acc=%.4f (%.1fs elapsed)\n",
                 e, et_loss[e], et_acc[e], et_vacc[e], NowSec() - et_t0);
  }
  double et_time = NowSec() - et_t0;
  const clio::run::u64 peak_win = (clio::run::u64)window * page_size;
  std::fprintf(stderr,
      "[TRAIN] ETERNIA: epoch0 loss=%.6f acc=%.4f -> epoch%d loss=%.6f acc=%.4f val_acc=%.4f (%.2fs)  "
      "peak GPU window=%lluMiB\n", et_loss[0], et_acc[0], epochs - 1,
      et_loss[epochs - 1], et_acc[epochs - 1], et_vacc[epochs - 1], et_time,
      (unsigned long long)(peak_win >> 20));
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
    REQUIRE(exact == "BIT-EXACT");
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
