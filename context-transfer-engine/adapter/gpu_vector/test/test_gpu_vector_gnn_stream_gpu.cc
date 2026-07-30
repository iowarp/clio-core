/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Option B: the SAME GraphSAGE-mean forward as test_gpu_vector_gnn_gpu.cc, but the
 * ogbn-arxiv feature matrix is sourced through the REAL gv::Vector<float> +
 * SequentialTransaction streaming abstraction with LOSSLESS zstd -- exercising the
 * compressor's CPU-codec D2H/H2D staging (compressor_runtime.cc, Option B) on the
 * on-device streaming path, not just the host-staged store of Option A.
 *
 * Flow: store the (row-padded) feature matrix page-by-page through the zstd-pinned
 * compressor from HOST buffers, then stream every page back with a
 * SequentialTransaction sweep -- each window is decompressed by the compressor
 * (host zstd) and H2D-staged into the resident HBM window, then a gather kernel
 * scatters it into the full feature buffer. The 2-layer GraphSAGE forward then runs
 * on that streamed buffer and is checked BIT-FOR-BIT against an uncompressed
 * baseline (lossless => identical). This proves the workload runs through the real
 * vector abstraction with zstd, with an exact correctness guarantee.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_cte/gpu_vector/transaction.h>

#include <clio_ctp/util/gpu_api.h>

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
  int port = port_env ? std::atoi(port_env) : 10596;
  std::string cfg = "/tmp/gpu_vec_gnn_stream_" + std::to_string(port) + ".yaml";
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
  std::fprintf(stderr, "[GNN-STREAM] compose=%s\n", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

bool ReadFile(const std::string &path, std::vector<char> &out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "[GNN-STREAM] cannot open %s\n", path.c_str()); return false; }
  std::streamoff n = f.tellg();
  f.seekg(0);
  out.resize((size_t)n);
  f.read(out.data(), n);
  return (bool)f;
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

// Scatter a resident window [lo,hi) (flat float indices) into the FULL feature
// buffer at its global offset. Single block (this vector is single-block).
__global__ void GnnScatterKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceView<float> view, clio::run::u64 lo,
                                 clio::run::u64 hi, float *d_feat) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  v.read_range(lo, hi, [d_feat](clio::run::u64 i, float val) { d_feat[i] = val; });
  (void)g_ipc_manager;
}

__global__ void GnnAggMeanKernel(const float *feat, int D,
                                 const std::int64_t *indptr,
                                 const std::int64_t *indices, std::int64_t N,
                                 float *agg) {
  std::int64_t total = N * (std::int64_t)D;
  for (std::int64_t idx = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (std::int64_t)gridDim.x * blockDim.x) {
    std::int64_t i = idx / D;
    int d = (int)(idx - i * D);
    std::int64_t beg = indptr[i], end = indptr[i + 1];
    float acc = 0.0f;
    for (std::int64_t k = beg; k < end; ++k)
      acc += feat[indices[k] * (std::int64_t)D + d];
    std::int64_t deg = end - beg;
    agg[idx] = (deg > 0) ? acc / (float)deg : 0.0f;
  }
}

__global__ void GnnCombineKernel(const float *self, const float *neigh, int Din,
                                 int Dout, const float *Wself,
                                 const float *Wneigh, std::int64_t N, float *out,
                                 int relu) {
  std::int64_t total = N * (std::int64_t)Dout;
  for (std::int64_t idx = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total; idx += (std::int64_t)gridDim.x * blockDim.x) {
    std::int64_t i = idx / Dout;
    int o = (int)(idx - i * Dout);
    const float *s = &self[i * (std::int64_t)Din];
    const float *g = &neigh[i * (std::int64_t)Din];
    float acc = 0.0f;
    for (int f = 0; f < Din; ++f)
      acc += s[f] * Wself[f * Dout + o] + g[f] * Wneigh[f * Dout + o];
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

TEST_CASE("gpu_vector: GraphSAGE over a zstd Vector<float> stream is bit-identical "
          "to the baseline",
          "[gpu_vector][compress][gnn][zstd][stream][real]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;
  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  const std::string data_dir = std::getenv("CLIO_GNN_DATA")
                                   ? std::getenv("CLIO_GNN_DATA")
                                   : "/workspace/data/ogb/arxiv";

  std::int64_t N = 0, E = 0; int F = 0, C = 0;
  { std::ifstream mf(data_dir + "/meta.txt"); REQUIRE((bool)mf); mf >> N >> F >> E >> C; }
  REQUIRE(N > 0); REQUIRE(F > 0); REQUIRE(C > 0);
  const int H = kHidden;

  std::vector<char> feat_bytes;
  REQUIRE(ReadFile(data_dir + "/features.f32", feat_bytes));
  const std::int64_t feat_elems = N * (std::int64_t)F;
  REQUIRE((std::int64_t)feat_bytes.size() == feat_elems * (std::int64_t)sizeof(float));
  const float *h_feat = reinterpret_cast<const float *>(feat_bytes.data());

  std::vector<char> csr_bytes;
  REQUIRE(ReadFile(data_dir + "/graph.csr", csr_bytes));
  const std::int64_t *csr = reinterpret_cast<const std::int64_t *>(csr_bytes.data());
  REQUIRE(csr[0] == N);
  const std::int64_t csrE = csr[1];
  const std::int64_t *h_indptr = csr + 2;
  const std::int64_t *h_indices = csr + 2 + (N + 1);
  REQUIRE(h_indptr[N] == csrE);

  // ---- streaming page geometry ----
  const clio::run::u32 window = 4;
  const char *pk_env = std::getenv("CLIO_GNN_PAGE_KIB");
  const clio::run::u64 page_size =
      (pk_env ? (clio::run::u64)std::atoi(pk_env) : 256ULL) * 1024;
  const clio::run::u64 epp = page_size / sizeof(float);
  // K pages must cover N*F floats and be a multiple of the window; pad the tail
  // with zeros (the forward only reads the first N*F elements).
  clio::run::u64 K = (feat_elems + epp - 1) / epp;
  K = ((K + window - 1) / window) * window;
  const clio::run::u64 total_elems = K * epp;
  std::fprintf(stderr,
      "[GNN-STREAM] N=%lld F=%d E=%lld C=%d H=%d  pages=%llu x %lluKiB "
      "(window=%u, padded %lld->%llu floats)\n",
      (long long)N, F, (long long)E, (long long)C, H,
      (unsigned long long)K, (unsigned long long)(page_size >> 10), window,
      (long long)feat_elems, (unsigned long long)total_elems);

  std::vector<float> h_pad(total_elems, 0.0f);
  std::memcpy(h_pad.data(), h_feat, feat_elems * sizeof(float));

  // ---- CSR + weights + scratch (shared) ----
  std::int64_t *d_indptr = nullptr, *d_indices = nullptr;
  REQUIRE(cudaMalloc(&d_indptr, (N + 1) * sizeof(std::int64_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_indices, csrE * sizeof(std::int64_t)) == cudaSuccess);
  cudaMemcpy(d_indptr, h_indptr, (N + 1) * sizeof(std::int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, csrE * sizeof(std::int64_t), cudaMemcpyHostToDevice);

  auto mkw = [](int din, int dout, unsigned seed) {
    std::mt19937 rng(seed); float s = 1.0f / std::sqrt((float)din);
    std::uniform_real_distribution<float> dist(-s, s);
    std::vector<float> w((size_t)din * dout); for (auto &x : w) x = dist(rng); return w;
  };
  std::vector<float> hW1s = mkw(F, H, 1), hW1n = mkw(F, H, 2),
                     hW2s = mkw(H, C, 3), hW2n = mkw(H, C, 4);
  float *dW1s, *dW1n, *dW2s, *dW2n;
  REQUIRE(cudaMalloc(&dW1s, hW1s.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dW1n, hW1n.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dW2s, hW2s.size() * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dW2n, hW2n.size() * sizeof(float)) == cudaSuccess);
  cudaMemcpy(dW1s, hW1s.data(), hW1s.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dW1n, hW1n.data(), hW1n.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dW2s, hW2s.data(), hW2s.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dW2n, hW2n.data(), hW2n.size() * sizeof(float), cudaMemcpyHostToDevice);

  float *d_feat = nullptr, *d_agg1 = nullptr, *d_h1 = nullptr, *d_agg2 = nullptr,
        *d_logits = nullptr;
  REQUIRE(cudaMalloc(&d_feat, total_elems * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_agg1, feat_elems * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_h1, N * (std::int64_t)H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_agg2, N * (std::int64_t)H * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_logits, N * (std::int64_t)C * sizeof(float)) == cudaSuccess);

  auto forward = [&](std::vector<float> &out) {
    LaunchAgg(d_feat, F, d_indptr, d_indices, N, d_agg1);
    LaunchCombine(d_feat, d_agg1, F, H, dW1s, dW1n, N, d_h1, 1);
    LaunchAgg(d_h1, H, d_indptr, d_indices, N, d_agg2);
    LaunchCombine(d_h1, d_agg2, H, C, dW2s, dW2n, N, d_logits, 0);
    ctp::GpuApi::Synchronize();
    out.resize((size_t)(N * (std::int64_t)C));
    cudaMemcpy(out.data(), d_logits, N * (std::int64_t)C * sizeof(float),
               cudaMemcpyDeviceToHost);
  };

  // ---- baseline (resident) ----
  cudaMemcpy(d_feat, h_pad.data(), total_elems * sizeof(float), cudaMemcpyHostToDevice);
  { std::vector<float> warm; forward(warm); }
  std::vector<float> logits_base; forward(logits_base);

  // ---- store the padded feature pages through the zstd-pinned compressor from
  //      HOST buffers, under the names the Vector's cache manager reads ----
  const char *tag = "gnn_stream";
  clio::cte::core::Client core; core.Init(clio::cte::core::kCtePoolId);
  auto tagf = core.AsyncGetOrCreateTag(tag); tagf.Wait();
  REQUIRE(tagf->GetReturnCode() == 0);
  auto tag_id = tagf->tag_id_;
  clio::cte::core::Client comp; comp.Init(StoragePool());
  clio::cte::core::Context ctx; ctx.dynamic_compress_ = 1; ctx.data_type_ = 1;

  double store_t0 = NowSec();
  for (clio::run::u32 p = 0; p < K; ++p) {
    ctp::ipc::ShmPtr<> dp;
    dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(h_pad.data() + (clio::run::u64)p * epp);
    std::string name = std::string(tag) + "_b0_pi" + std::to_string(p);
    auto pf = comp.AsyncPutBlob(tag_id, name, (clio::run::u64)0, page_size, dp,
                                0.5f, ctx, 0, clio::run::PoolQuery::Local());
    pf.Wait();
    REQUIRE(pf->GetReturnCode() == 0);
  }
  double store_dt = NowSec() - store_t0;

  // ---- stream every page back through gv::Vector<float> + SequentialTransaction
  //      (zstd decompress + H2D staging in the compressor) into d_feat ----
  gv::Vector<float> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                        /*gpu_pages_per_block=*/2 * window,
                        /*host_pages_per_block=*/0, page_size,
                        /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                        /*manager_threads_per_block=*/32,
                        /*allow_cold_miss_fault=*/false,
                        /*storage_pool_id=*/StoragePool());

  cudaMemset(d_feat, 0, total_elems * sizeof(float));
  double stream_t0 = NowSec();
  gv::SequentialTransaction<float> sweep(vec, /*first_page=*/0, /*npages=*/K);
  sweep.Iterate([&](clio::run::u64 lo, clio::run::u64 hi, gv::DeviceView<float> v,
                    cudaStream_t s) {
    GnnScatterKernel<<<1, 32, 0, s>>>(gpu_info, v, lo, hi, d_feat);
  });
  ctp::GpuApi::Synchronize();
  double stream_dt = NowSec() - stream_t0;

  // ---- lossless proof #1: streamed device bytes == original ----
  std::vector<float> feat_rt((size_t)feat_elems);
  cudaMemcpy(feat_rt.data(), d_feat, feat_elems * sizeof(float), cudaMemcpyDeviceToHost);
  int feat_cmp = std::memcmp(feat_rt.data(), h_feat, (size_t)(feat_elems * sizeof(float)));
  std::fprintf(stderr, "[GNN-STREAM] streamed feature memcmp = %d (0 == bit-exact)\n",
               feat_cmp);
  REQUIRE(feat_cmp == 0);

  // ---- forward on the streamed features + bit-exact vs baseline ----
  { std::vector<float> warm; forward(warm); }
  std::vector<float> logits_comp; forward(logits_comp);
  REQUIRE(logits_comp.size() == logits_base.size());
  int logit_cmp = std::memcmp(logits_comp.data(), logits_base.data(),
                              logits_base.size() * sizeof(float));
  double max_abs = 0.0;
  for (size_t i = 0; i < logits_base.size(); ++i)
    max_abs = std::max(max_abs, (double)std::fabs(logits_comp[i] - logits_base[i]));

  double store_mbs = store_dt > 0 ? (feat_elems * sizeof(float) / 1048576.0) / store_dt : 0.0;
  double stream_mbs = stream_dt > 0 ? (feat_elems * sizeof(float) / 1048576.0) / stream_dt : 0.0;
  std::fprintf(stderr,
      "\n============ GNN zstd Vector<T> STREAM RESULTS ============\n"
      "  dataset   : ogbn-arxiv  N=%lld F=%d E=%lld C=%d\n"
      "  path      : gv::Vector<float> + SequentialTransaction (zstd, staged)\n"
      "  store     : %.1f MB/s (%.3fs, %llu pages)\n"
      "  stream    : %.1f MB/s (%.3fs, window=%u pages resident)\n"
      "  bit-exact : feature memcmp=%d  logit memcmp=%d  max_abs_diff=%.1f\n"
      "  RESULT    : %s\n"
      "==========================================================\n",
      (long long)N, F, (long long)E, (long long)C,
      store_mbs, store_dt, (unsigned long long)K,
      stream_mbs, stream_dt, window,
      feat_cmp, logit_cmp, max_abs,
      (feat_cmp == 0 && logit_cmp == 0 && max_abs == 0.0) ? "PASS (bit-exact)"
                                                          : "FAIL");
  REQUIRE(logit_cmp == 0);
  REQUIRE(max_abs == 0.0);

  cudaFree(d_indptr); cudaFree(d_indices);
  cudaFree(dW1s); cudaFree(dW1n); cudaFree(dW2s); cudaFree(dW2n);
  cudaFree(d_feat); cudaFree(d_agg1); cudaFree(d_h1); cudaFree(d_agg2); cudaFree(d_logits);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif
