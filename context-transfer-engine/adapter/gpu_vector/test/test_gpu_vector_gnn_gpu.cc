/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GraphSAGE-mean forward over a LOSSLESSLY (zstd) compressed node-feature matrix,
 * stored through the CTE compressor(600)->core(512) stack, on a REAL GNN dataset
 * (ogbn-arxiv), checked BIT-FOR-BIT against an uncompressed baseline.
 *
 * This is the lossless counterpart to test_gpu_vector_kmeans_real_gpu.cc. That
 * test uses cuSZp (LOSSY, GPU codec) and checks inertia PARITY, because a lossy
 * perturbation cannot reproduce the raw result exactly. Here the codec is zstd
 * (LOSSLESS, CPU codec), so the correctness statement is the strongest possible:
 * the decompressed feature bytes are IDENTICAL to the originals, hence the GNN
 * forward output is bit-for-bit identical (memcmp == 0 / max-abs-diff == 0.0).
 *
 * zstd is a CPU codec: the compressor runs Compress()/Decompress() on the HOST
 * over ToFullPtr(blob_data_). So (unlike the cuSZp path, which hands the codec a
 * device pointer) we feed the compressor HOST buffers:
 *   Store: AsyncPutBlob(pool 600) with ShmPtr{alloc_id=null, off_=(u64)host_ptr}.
 *   Read : AsyncGetBlob(pool 600) into a host buffer, then cudaMemcpy H2D.
 * Single-node/single-process (CLIO_INIT server in-proc) => raw host pointers are
 * valid across the client->runtime boundary.
 *
 * The codec is pinned ON THE BLOB CONTEXT (compress_lib_/compress_preset_).
 * It used to be pinned with CLIO_CTE_COMPRESS_LIB, but nothing reads that env
 * any more: with only dynamic_compress_ set the predictor picked a
 * pass-through, every component reported 1.000x, and the test still passed --
 * because bit-exactness is exactly what a pass-through gives you. Hence the
 * ratio assert at the end. Override with CLIO_GNN_COMPRESS_LIB/_PRESET.
 *
 * Workload (2-layer GraphSAGE-mean, fixed seeded random weights):
 *   agg1  = mean_{j in N(i)} x[j]
 *   h1    = ReLU( x @ Wself1 + agg1 @ Wneigh1 )      [F -> H]
 *   agg2  = mean_{j in N(i)} h1[j]
 *   logit = h1 @ Wself2 + agg2 @ Wneigh2             [H -> C]
 * The SpMM-mean aggregation runs on the GPU over the CSR graph. Aggregation and
 * the dense combine are BOTH deterministic (one thread owns one output element,
 * fixed-order reduction, NO atomics), so identical device input bytes => bitwise
 * identical logits. That is what makes the bit-exact assertion meaningful.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
// compressor_client.h pulls in host-only task types that do not compile in the
// nvcc device pass; it is only used by the host TEST_CASE below.
#if !CTP_IS_DEVICE_PASS
#include <clio_cte/compressor/compressor_client.h>
#endif

#include <clio_ctp/util/gpu_api.h>

#include "gnn_dataset.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

// GraphSAGE hidden dimension (layer-1 output). Layer-2 output = num classes.
constexpr int kHidden = 256;

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10594;
  std::string cfg = "/tmp/gpu_vec_gnn_" + std::to_string(port) + ".yaml";
  // Single-node Option-A layout: a default hbm bdev (301) + the compressor
  // (600, forwarding to 512) + cte_core (512) backed by ONE host-RAM tier. The
  // zstd-compressed feature pages are host bytes, so a RAM tier is the natural
  // sink; no HBM/DRAM split is needed here (that is the capacity story).
  const char *ram_env = std::getenv("CLIO_GNN_RAM_MIB");
  int ram_mib = ram_env ? std::atoi(ram_env) : 4096;
  {
    std::ofstream f(cfg);
    f << "networking:\n  port: " << port << "\n\n"
      << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
      << "compose:\n"
      << "  - mod_name: clio_bdev\n"
      << "    pool_name: \"hbm::chi_default_bdev\"\n"
      << "    pool_query: local\n    pool_id: \"301.0\"\n"
      << "    bdev_type: hbm\n    capacity: \"128MB\"\n\n"
      << "  - mod_name: clio_cte_compressor\n"
      << "    pool_name: cte_compressor\n    pool_query: local\n"
      << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
      << "  - mod_name: clio_cte_core\n"
      << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
      << "    storage:\n"
      << "      - path: \"ram::cte_ram_tier\"\n"
      << "        bdev_type: \"ram\"\n        capacity_limit: \""
      << ram_mib << "MB\"\n        score: 1.0\n"
      << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  std::fprintf(stderr, "[GNN] compose=%s\n", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

#if !CTP_IS_DEVICE_PASS
using gnn_test::CodecName;
using gnn_test::EnvI64;

// Kept as a named wrapper: the per-component section below reports which file
// it could not open, which the quiet shared loader deliberately does not.
bool ReadFile(const std::string &path, std::vector<char> &out) {
  if (gnn_test::ReadFileQuiet(path, out)) return true;
  std::fprintf(stderr, "[GNN] cannot open %s\n", path.c_str());
  return false;
}
#endif  // !CTP_IS_DEVICE_PASS

}  // namespace

// ===========================================================================
//  GPU kernels (compiled in the device pass). Deterministic: one thread owns
//  one output element, fixed-order reduction, no atomics.
// ===========================================================================

// Mean aggregation over the CSR graph: agg[i, d] = mean_{j in N(i)} feat[j, d].
// One thread per (node, dim) element via grid-stride over N*D.
__global__ void GnnAggMeanKernel(const float *feat, int D,
                                 const std::int64_t *indptr,
                                 const std::int64_t *indices, std::int64_t N,
                                 float *agg) {
  std::int64_t total = N * (std::int64_t)D;
  for (std::int64_t idx =
           (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (std::int64_t)gridDim.x * blockDim.x) {
    std::int64_t i = idx / D;
    int d = (int)(idx - i * D);
    std::int64_t beg = indptr[i], end = indptr[i + 1];
    float acc = 0.0f;
    for (std::int64_t k = beg; k < end; ++k) {
      acc += feat[indices[k] * (std::int64_t)D + d];
    }
    std::int64_t deg = end - beg;
    agg[idx] = (deg > 0) ? acc / (float)deg : 0.0f;
  }
}

// GraphSAGE combine: out[i, o] = self[i,:]@Wself[:,o] + neigh[i,:]@Wneigh[:,o],
// optionally ReLU. Weights are row-major [Din x Dout]. One thread per (node,out)
// element via grid-stride over N*Dout.
__global__ void GnnCombineKernel(const float *self, const float *neigh, int Din,
                                 int Dout, const float *Wself,
                                 const float *Wneigh, std::int64_t N, float *out,
                                 int relu) {
  std::int64_t total = N * (std::int64_t)Dout;
  for (std::int64_t idx =
           (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (std::int64_t)gridDim.x * blockDim.x) {
    std::int64_t i = idx / Dout;
    int o = (int)(idx - i * Dout);
    const float *s = &self[i * (std::int64_t)Din];
    const float *g = &neigh[i * (std::int64_t)Din];
    float acc = 0.0f;
    for (int f = 0; f < Din; ++f) {
      acc += s[f] * Wself[f * Dout + o] + g[f] * Wneigh[f * Dout + o];
    }
    if (relu && acc < 0.0f) acc = 0.0f;
    out[idx] = acc;
  }
}

#if !CTP_IS_DEVICE_PASS

namespace {

void LaunchAgg(const float *feat, int D, const std::int64_t *indptr,
               const std::int64_t *indices, std::int64_t N, float *agg) {
  const int threads = 256;
  std::int64_t total = N * (std::int64_t)D;
  int blocks = (int)std::min<std::int64_t>(65535, (total + threads - 1) / threads);
  GnnAggMeanKernel<<<blocks, threads>>>(feat, D, indptr, indices, N, agg);
}

void LaunchCombine(const float *self, const float *neigh, int Din, int Dout,
                   const float *Wself, const float *Wneigh, std::int64_t N,
                   float *out, int relu) {
  const int threads = 256;
  std::int64_t total = N * (std::int64_t)Dout;
  int blocks = (int)std::min<std::int64_t>(65535, (total + threads - 1) / threads);
  GnnCombineKernel<<<blocks, threads>>>(self, neigh, Din, Dout, Wself, Wneigh, N,
                                        out, relu);
}

}  // namespace

TEST_CASE("gpu_vector: GraphSAGE forward over a lossless-zstd compressed "
          "ogbn-arxiv feature matrix is bit-identical to the baseline",
          "[gpu_vector][compress][gnn][zstd][real]") {
  EnsureInit();

  const std::string data_dir = std::getenv("CLIO_GNN_DATA")
                                   ? std::getenv("CLIO_GNN_DATA")
                                   : "/workspace/data/ogb/arxiv";
  const char *lib_env = std::getenv("CLIO_CTE_COMPRESS_LIB");
  const char *preset_env = std::getenv("CLIO_CTE_COMPRESS_PRESET");
  std::fprintf(stderr, "[GNN] codec pin: lib=%s preset=%s data=%s\n",
               lib_env ? lib_env : "(none)", preset_env ? preset_env : "balanced",
               data_dir.c_str());

  // Real ogbn-arxiv when gnn_prep.py has been run, else a synthetic graph of
  // the same shape -- see gnn_dataset.h for why that substitution is sound.
  gnn_test::Dataset ds = gnn_test::LoadOrSynthDataset(data_dir, "GNN");
  const std::int64_t N = ds.N, E = ds.E;
  const int F = ds.F, C = ds.C;
  const bool real_data = ds.real;
  REQUIRE(N > 0); REQUIRE(F > 0); REQUIRE(C > 0);

  const int H = std::getenv("CLIO_GNN_HIDDEN") ? std::atoi(std::getenv("CLIO_GNN_HIDDEN"))
                                               : kHidden;
  std::fprintf(stderr, "[GNN] source=%s N=%lld F=%d E=%lld C=%d H=%d\n",
               ds.SourceName(), (long long)N, F, (long long)E, C, H);

  const std::int64_t feat_elems = ds.FeatElems();
  const float *h_feat = ds.Feat();
  const std::int64_t logical_bytes = ds.FeatBytes();
  const std::int64_t csrE = ds.Csr()[1];
  const std::int64_t *h_indptr = ds.Indptr();
  const std::int64_t *h_indices = ds.Indices();
  REQUIRE(ds.Csr()[0] == N);
  REQUIRE(h_indptr[N] == csrE);

  // ---- CSR to device (shared by both runs) ----
  std::int64_t *d_indptr = nullptr, *d_indices = nullptr;
  REQUIRE(cudaMalloc(&d_indptr, (N + 1) * sizeof(std::int64_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_indices, csrE * sizeof(std::int64_t)) == cudaSuccess);
  cudaMemcpy(d_indptr, h_indptr, (N + 1) * sizeof(std::int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, csrE * sizeof(std::int64_t), cudaMemcpyHostToDevice);

  // ---- fixed seeded random weights (shared by both runs) ----
  auto make_weights = [](int din, int dout, unsigned seed) {
    std::mt19937 rng(seed);
    float s = 1.0f / std::sqrt((float)din);
    std::uniform_real_distribution<float> dist(-s, s);
    std::vector<float> w((size_t)din * dout);
    for (auto &x : w) x = dist(rng);
    return w;
  };
  std::vector<float> hWself1 = make_weights(F, H, 1);
  std::vector<float> hWneigh1 = make_weights(F, H, 2);
  std::vector<float> hWself2 = make_weights(H, C, 3);
  std::vector<float> hWneigh2 = make_weights(H, C, 4);
  float *dWself1, *dWneigh1, *dWself2, *dWneigh2;
  REQUIRE(cudaMalloc(&dWself1, hWself1.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dWneigh1, hWneigh1.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dWself2, hWself2.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dWneigh2, hWneigh2.size() * sizeof(float)) == cudaSuccess);
  cudaMemcpy(dWself1, hWself1.data(), hWself1.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dWneigh1, hWneigh1.data(), hWneigh1.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dWself2, hWself2.data(), hWself2.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dWneigh2, hWneigh2.data(), hWneigh2.size() * sizeof(float), cudaMemcpyHostToDevice);

  // ---- device scratch shared by both forward passes ----
  float *d_feat = nullptr, *d_agg1 = nullptr, *d_h1 = nullptr, *d_agg2 = nullptr,
        *d_logits = nullptr;
  REQUIRE(cudaMalloc(&d_feat, feat_elems * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_agg1, feat_elems * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_h1, N * (std::int64_t)H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_agg2, N * (std::int64_t)H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_logits, N * (std::int64_t)C * sizeof(float)) == cudaSuccess);

  auto forward = [&](std::vector<float> &out_logits) -> double {
    ctp::GpuApi::Synchronize();
    double t0 = NowSec();
    LaunchAgg(d_feat, F, d_indptr, d_indices, N, d_agg1);
    LaunchCombine(d_feat, d_agg1, F, H, dWself1, dWneigh1, N, d_h1, /*relu=*/1);
    LaunchAgg(d_h1, H, d_indptr, d_indices, N, d_agg2);
    LaunchCombine(d_h1, d_agg2, H, C, dWself2, dWneigh2, N, d_logits, /*relu=*/0);
    ctp::GpuApi::Synchronize();
    double dt = NowSec() - t0;
    out_logits.resize((size_t)(N * (std::int64_t)C));
    cudaMemcpy(out_logits.data(), d_logits,
               N * (std::int64_t)C * sizeof(float), cudaMemcpyDeviceToHost);
    return dt;
  };

  // =====================================================================
  //  BASELINE: features resident in HBM, no compression.
  // =====================================================================
  cudaMemcpy(d_feat, h_feat, logical_bytes, cudaMemcpyHostToDevice);
  // Warmup: the first launch pays PTX->SASS JIT (arch 90 PTX on sm_120), which
  // would otherwise be charged entirely to the baseline. Discard it so the two
  // reported forward times reflect steady state (they run identical work).
  { std::vector<float> warm; forward(warm); }
  std::vector<float> logits_base;
  double fwd_base_ms = forward(logits_base) * 1e3;
  std::fprintf(stderr, "[GNN] baseline forward: %.2f ms\n", fwd_base_ms);

  // =====================================================================
  //  COMPRESSED: store the feature matrix page-by-page through the zstd-
  //  pinned compressor from HOST buffers, then read the pages back
  //  (decompress on host) -> H2D -> identical forward.
  // =====================================================================
  clio::cte::core::Client core;
  core.Init(clio::cte::core::kCtePoolId);
  auto tagf = core.AsyncGetOrCreateTag("gnn_arxiv");
  tagf.Wait();
  REQUIRE(tagf->GetReturnCode() == 0);
  auto tag_id = tagf->tag_id_;

  clio::cte::compressor::Client comp(StoragePool());
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.data_type_ = 1;  // float32
  // Pin the codec ON THE BLOB CONTEXT. This test used to pin it with
  // CLIO_CTE_COMPRESS_LIB, but nothing reads that env any more -- with only
  // dynamic_compress_ set, the predictor picked a pass-through and every
  // component reported a flat 1.000x ratio while still passing the bit-exact
  // assert. A lossless test that compresses nothing still "passes", so the
  // ratio has to be pinned and then checked (see the < 1.0 guard below).
  ctx.compress_lib_ = (int)EnvI64("CLIO_GNN_COMPRESS_LIB", 10);  // 10 = ZSTD
  ctx.compress_preset_ = (int)EnvI64("CLIO_GNN_COMPRESS_PRESET", 2);  // BALANCED

  // Page = whole rows, ~CLIO_GNN_PAGE_KIB KiB (default 256), a multiple of F*4.
  const char *pk_env = std::getenv("CLIO_GNN_PAGE_KIB");
  std::int64_t page_kib = pk_env ? std::atoll(pk_env) : 256;
  std::int64_t row_bytes = (std::int64_t)F * sizeof(float);
  std::int64_t rows_per_page = std::max<std::int64_t>(1, (page_kib * 1024) / row_bytes);
  std::int64_t npages = (N + rows_per_page - 1) / rows_per_page;
  std::fprintf(stderr,
      "[GNN] store: %lld pages x %lld rows (%lld B/page) via zstd (host bufs)\n",
      (long long)npages, (long long)rows_per_page,
      (long long)(rows_per_page * row_bytes));

  auto page_name = [](std::int64_t p) { return "feat_p" + std::to_string(p); };

  // ---- store loop (timed) ----
  double store_t0 = NowSec();
  for (std::int64_t p = 0; p < npages; ++p) {
    std::int64_t r0 = p * rows_per_page;
    std::int64_t rows = std::min(rows_per_page, N - r0);
    std::int64_t bytes = rows * row_bytes;
    ctp::ipc::ShmPtr<> dp;
    dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(h_feat + r0 * (std::int64_t)F);
    auto pf = comp.AsyncPutBlob(tag_id, page_name(p), (clio::run::u64)0,
                                (clio::run::u64)bytes, dp, 0.5f, ctx, 0,
                                clio::run::PoolQuery::Local());
    pf.Wait();
    REQUIRE(pf->GetReturnCode() == 0);
  }
  double store_dt = NowSec() - store_t0;

  // ---- exact stored size per page (core GetBlobSize == bytes actually
  //      stored: compressed+24B header, or raw when not beneficial) ----
  std::vector<std::int64_t> stored_sz(npages);
  std::int64_t stored_bytes = 0;
  for (std::int64_t p = 0; p < npages; ++p) {
    auto sf = core.AsyncGetBlobSize(tag_id, page_name(p));
    sf.Wait();
    REQUIRE(sf->GetReturnCode() == 0);
    stored_sz[p] = (std::int64_t)sf->size_;
    stored_bytes += stored_sz[p];
  }
  double ratio = stored_bytes ? (double)logical_bytes / (double)stored_bytes : 0.0;

  // ---- read loop (timed): decompress on host -> H2D into d_feat ----
  cudaMemset(d_feat, 0, logical_bytes);
  std::vector<char> stage((size_t)(rows_per_page * row_bytes));
  double read_t0 = NowSec();
  for (std::int64_t p = 0; p < npages; ++p) {
    std::int64_t r0 = p * rows_per_page;
    std::int64_t rows = std::min(rows_per_page, N - r0);
    std::int64_t bytes = rows * row_bytes;
    ctp::ipc::ShmPtr<> dp;
    dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(stage.data());
    // Pass the EXACT stored size: the compressor's Decompress uses it as the
    // compressed byte count (minus its 24B header), so an over-estimate would
    // feed zstd trailing garbage and fail. GetBlobSize gives it exactly.
    auto gf = comp.AsyncGetBlob(tag_id, page_name(p), (clio::run::u64)0,
                                (clio::run::u64)stored_sz[p], 0, dp,
                                clio::run::PoolQuery::Local());
    gf.Wait();
    REQUIRE(gf->GetReturnCode() == 0);
    REQUIRE((std::int64_t)gf->output_size_ == bytes);
    cudaMemcpy(d_feat + r0 * (std::int64_t)F, stage.data(), bytes,
               cudaMemcpyHostToDevice);
  }
  double read_dt = NowSec() - read_t0;

  // ---- lossless proof #1: decompressed device bytes == original bytes ----
  std::vector<float> feat_rt((size_t)feat_elems);
  cudaMemcpy(feat_rt.data(), d_feat, logical_bytes, cudaMemcpyDeviceToHost);
  int feat_cmp = std::memcmp(feat_rt.data(), h_feat, (size_t)logical_bytes);
  std::fprintf(stderr, "[GNN] feature round-trip memcmp = %d (0 == bit-exact)\n",
               feat_cmp);
  REQUIRE(feat_cmp == 0);

  // ---- compressed-path forward ----
  // Warmup again: a long CPU-bound level-19 store idles the GPU, which then
  // downclocks (DVFS); the first post-store launch pays the ramp-up. Discard
  // one pass so the timed compressed forward reflects the same steady state as
  // the baseline (the computation is provably identical — see the bit-exact
  // check below).
  { std::vector<float> warm; forward(warm); }
  std::vector<float> logits_comp;
  double fwd_comp_ms = forward(logits_comp) * 1e3;
  std::fprintf(stderr, "[GNN] compressed forward done (%.2f ms)\n", fwd_comp_ms);

  // ---- lossless proof #2: logits bit-identical ----
  REQUIRE(logits_comp.size() == logits_base.size());
  int logit_cmp = std::memcmp(logits_comp.data(), logits_base.data(),
                              logits_base.size() * sizeof(float));
  double max_abs = 0.0;
  for (size_t i = 0; i < logits_base.size(); ++i)
    max_abs = std::max(max_abs, (double)std::fabs(logits_comp[i] - logits_base[i]));

  // ---- per-component lossless ratios (features / indptr / indices / labels).
  //      Store each component in the SAME page-sized chunks as the feature
  //      matrix. Chunking is not just for consistency: zstd BEST (level 19)
  //      optimal-parsing is superlinear, and a single ~18 MiB sorted-int64
  //      blob (CSR indices) takes >800 s, whereas the same data in 256 KiB
  //      chunks compresses in seconds — so paged stores keep every preset
  //      tractable while giving the same ratio to <0.1%. ----
  const std::int64_t chunk_bytes = rows_per_page * row_bytes;  // == feature page
  auto store_paged = [&](const char *prefix, const void *ptr,
                         std::int64_t bytes) -> std::int64_t {
    const char *base = reinterpret_cast<const char *>(ptr);
    std::int64_t stored = 0, off = 0, ci = 0;
    while (off < bytes) {
      std::int64_t n = std::min(chunk_bytes, bytes - off);
      std::string nm = std::string(prefix) + "_c" + std::to_string(ci++);
      ctp::ipc::ShmPtr<> dp;
      dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
      dp.off_ = reinterpret_cast<clio::run::u64>(base + off);
      auto pf = comp.AsyncPutBlob(tag_id, nm, (clio::run::u64)0,
                                  (clio::run::u64)n, dp, 0.5f, ctx, 0,
                                  clio::run::PoolQuery::Local());
      pf.Wait();
      if (pf->GetReturnCode() != 0) return -1;
      auto sf = core.AsyncGetBlobSize(tag_id, nm);
      sf.Wait();
      if (sf->GetReturnCode() != 0) return -1;
      stored += (std::int64_t)sf->size_;
      off += n;
    }
    return stored;
  };
  auto store_one = [&](const char *name, const void *ptr, std::int64_t bytes) {
    return store_paged(name, ptr, bytes);
  };
  auto comp_t0 = NowSec();
  (void)comp_t0;
  std::int64_t indptr_bytes = (N + 1) * (std::int64_t)sizeof(std::int64_t);
  std::int64_t indices_bytes = csrE * (std::int64_t)sizeof(std::int64_t);
  std::int64_t s_indptr = -1, s_indices = -1, s_labels = -1, label_sz = 0;
  std::vector<char> label_bytes;
  // The per-component breakdown is opt-out (CLIO_GNN_COMPONENTS=0) because zstd
  // BEST (level 19) optimal-parse is a pathological worst case on the sorted
  // int64 CSR indices: even in 256 KiB chunks the ~18 MiB indices array takes
  // several minutes, while the (float) feature matrix — the headline — stores
  // fine. So for the BEST sweep we measure the feature ratio and skip components.
  const char *comp_env = std::getenv("CLIO_GNN_COMPONENTS");
  bool do_components = !comp_env || comp_env[0] != '0';
  if (do_components) {
    std::fprintf(stderr, "[GNN] storing per-component blobs (paged)...\n");
    s_indptr = store_one("comp_indptr", h_indptr, indptr_bytes);
    std::fprintf(stderr, "[GNN]   indptr stored (%.2fs elapsed)\n", NowSec() - comp_t0);
    s_indices = store_one("comp_indices", h_indices, indices_bytes);
    std::fprintf(stderr, "[GNN]   indices stored (%.2fs elapsed)\n", NowSec() - comp_t0);
    label_bytes = ds.labels;  // synthesised too, so the table is never short a row
    if (!label_bytes.empty()) {
      label_sz = (std::int64_t)label_bytes.size();
      s_labels = store_one("comp_labels", label_bytes.data(), label_sz);
    }
    std::fprintf(stderr, "[GNN]   labels stored (%.2fs elapsed)\n", NowSec() - comp_t0);
  } else {
    std::fprintf(stderr, "[GNN] per-component breakdown skipped "
                         "(CLIO_GNN_COMPONENTS=0)\n");
  }

  auto rr = [](std::int64_t logical, std::int64_t stored) {
    return stored > 0 ? (double)logical / (double)stored : 0.0;
  };

  // ---- results block ----
  double comp_mbs = store_dt > 0 ? (logical_bytes / 1048576.0) / store_dt : 0.0;
  double decomp_mbs = read_dt > 0 ? (logical_bytes / 1048576.0) / read_dt : 0.0;
  std::fprintf(stderr,
      "\n================= GNN LOSSLESS-ZSTD RESULTS =================\n"
      "  dataset      : %s  N=%lld F=%d E=%lld C=%d\n"
      "  codec        : %s  preset=%s  (LOSSLESS)\n"
      "  features     : logical=%.2f MiB  stored=%.2f MiB  ratio=%.3fx\n",
      real_data ? "ogbn-arxiv" : "SYNTHETIC (not ogbn-arxiv)",
      (long long)N, F, (long long)E, (long long)C,
      CodecName(ctx.compress_lib_), preset_env ? preset_env : "balanced",
      logical_bytes / 1048576.0, stored_bytes / 1048576.0, ratio);
  if (do_components) {
    std::fprintf(stderr,
        "    per-component lossless ratio (paged):\n"
        "      indptr   : %.2f MiB -> %.2f MiB  (%.3fx)\n"
        "      indices  : %.2f MiB -> %.2f MiB  (%.3fx)\n"
        "      labels   : %.2f MiB -> %.2f MiB  (%.3fx)\n",
        indptr_bytes / 1048576.0, s_indptr / 1048576.0, rr(indptr_bytes, s_indptr),
        indices_bytes / 1048576.0, s_indices / 1048576.0, rr(indices_bytes, s_indices),
        label_sz / 1048576.0, s_labels / 1048576.0, rr(label_sz, s_labels));
  } else {
    std::fprintf(stderr, "    per-component : skipped (CLIO_GNN_COMPONENTS=0)\n");
  }
  std::fprintf(stderr,
      "  throughput   : compress=%.1f MB/s (%.3fs)  decompress=%.1f MB/s (%.3fs)\n"
      "  forward (ms) : baseline=%.2f  compressed=%.2f\n"
      "  bit-exact    : feature memcmp=%d  logit memcmp=%d  max_abs_diff=%.1f\n"
      "  RESULT       : %s\n"
      "============================================================\n",
      comp_mbs, store_dt, decomp_mbs, read_dt,
      fwd_base_ms, fwd_comp_ms,
      feat_cmp, logit_cmp, max_abs,
      (feat_cmp == 0 && logit_cmp == 0 && max_abs == 0.0) ? "PASS (bit-exact)"
                                                          : "FAIL");

  REQUIRE(logit_cmp == 0);
  REQUIRE(max_abs == 0.0);
  // The codec must actually have compressed something. Bit-exactness alone is
  // satisfied by a pass-through, which is exactly what this test silently did
  // when its codec pin stopped being read: every component reported 1.000x and
  // it still passed. Pass-through lands at (or just under) 1.000x; real
  // ogbn-arxiv float features are the worst genuine case at ~1.078x, so a 1.01x
  // floor separates the two without being tight on real data.
  REQUIRE(ratio > 1.01);

  cudaFree(d_indptr); cudaFree(d_indices);
  cudaFree(dWself1); cudaFree(dWneigh1); cudaFree(dWself2); cudaFree(dWneigh2);
  cudaFree(d_feat); cudaFree(d_agg1); cudaFree(d_h1); cudaFree(d_agg2);
  cudaFree(d_logits);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif
