/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * GNN CAPACITY comparison: a GraphSAGE neighbor-aggregation + graph readout over
 * a node-feature matrix LARGER THAN GPU MEMORY, the traditional in-core way vs.
 * through the lossless-zstd compressed / tiered GPU vector (Eternia).
 *
 * The capacity point (same as the k-means capacity test, now for a GNN): a
 * traditional in-core GPU GNN must hold the ENTIRE node-feature matrix resident in
 * HBM, so its peak GPU memory grows with the graph and it OOMs once the features
 * exceed HBM. Eternia instead stores the features losslessly compressed (zstd)
 * across a tier stack (HBM->DRAM) and streams a small fixed window through HBM,
 * so its peak GPU memory is O(window) -- constant -- regardless of feature size.
 * Same workload, same real features, but only Eternia runs once they exceed HBM.
 *
 * Workload (memory-bound core of a GNN, with an O(F) output so it is bit-exact
 * verifiable and does not itself need O(N) resident memory): the aggregated
 * neighbour-feature GRAPH READOUT
 *      pool[f] = sum over nodes v of  deg(v) * feat[v][f]
 * which equals sum over all edges (u->v) of feat[u][f] -- the total GraphSAGE
 * sum-aggregation of the graph, pooled to a graph-level embedding (the forward of
 * a graph-classification GNN). deg(v) is the real node degree; feat is the real
 * ogbn-products feature matrix, TILED to reach the requested logical size.
 *
 * Determinism / correctness: the readout is summed one feature per thread over
 * nodes in ascending id order (double accumulator, no atomics), so the traditional
 * resident path and the Eternia windowed-streaming path perform the SAME float
 * operations in the SAME order -> the pooled vectors are BIT-IDENTICAL whenever
 * both run (i.e. at sizes that fit). Past the OOM crossover only Eternia runs.
 *
 * Env: CLIO_GNN_DATA (dir with features.f32 + graph.csr + meta.txt),
 *      CLIO_GNN_CAP_NODES (logical node count; tiled), CLIO_GNN_PAGE_ROWS
 *      (nodes/page, default 2560), CLIO_GNN_WINDOW (pages, default 4),
 *      CLIO_GNN_HBM_MIB / CLIO_GNN_DRAM_MIB (tier caps), CLIO_GNN_CSV.
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
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

bool g_initialized = false;
inline clio::run::PoolId StoragePool() { return clio::run::PoolId(600, 0); }

int g_F = 100;  // feature dim (set from meta.txt)

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10598;
  std::string cfg = "/tmp/gpu_vec_gnn_cap_" + std::to_string(port) + ".yaml";
  const char *hbm_env = std::getenv("CLIO_GNN_HBM_MIB");
  const char *dram_env = std::getenv("CLIO_GNN_DRAM_MIB");
  int hbm_mib = hbm_env ? std::atoi(hbm_env) : 2048;     // small HBM tier -> spill
  int dram_mib = dram_env ? std::atoi(dram_env) : 20480;  // DRAM overflow tier
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
      << "      - path: \"hbm::cte_hbm_tier\"\n"
      << "        bdev_type: \"hbm\"\n        capacity_limit: \""
      << hbm_mib << "MB\"\n        score: 1.0\n"
      << "      - path: \"ram::cte_dram_tier\"\n"
      << "        bdev_type: \"ram\"\n        capacity_limit: \""
      << dram_mib << "MB\"\n        score: 0.5\n"
      << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

/** Read meta.txt "N F E C". */
void ReadMeta(const std::string &dir, clio::run::u64 &N, int &F,
              clio::run::u64 &E, int &C) {
  std::ifstream f(dir + "/meta.txt");
  REQUIRE(f.good());
  long long n, e; int ff, cc;
  f >> n >> ff >> e >> cc;
  N = (clio::run::u64)n; F = ff; E = (clio::run::u64)e; C = cc;
}

/** Load the flat features.f32 (N0*F float32) into host `out`. */
void LoadFeatures(const std::string &dir, clio::run::u64 N0, int F,
                  std::vector<float> &out) {
  std::ifstream f(dir + "/features.f32", std::ios::binary);
  REQUIRE(f.good());
  out.resize(N0 * (clio::run::u64)F);
  f.read(reinterpret_cast<char *>(out.data()),
         (std::streamsize)(out.size() * sizeof(float)));
  REQUIRE((bool)f);
}

/** Node degrees from graph.csr: int64 N,E, indptr[N+1], indices[E].
 *  deg[v] = indptr[v+1]-indptr[v]. Returns real N0 degrees. */
void LoadDegrees(const std::string &dir, clio::run::u64 N0,
                 std::vector<int> &deg) {
  std::ifstream f(dir + "/graph.csr", std::ios::binary);
  REQUIRE(f.good());
  std::int64_t hdr[2];
  f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
  REQUIRE((clio::run::u64)hdr[0] == N0);
  std::vector<std::int64_t> indptr(N0 + 1);
  f.read(reinterpret_cast<char *>(indptr.data()),
         (std::streamsize)((N0 + 1) * sizeof(std::int64_t)));
  REQUIRE((bool)f);
  deg.resize(N0);
  for (clio::run::u64 v = 0; v < N0; ++v)
    deg[v] = (int)(indptr[v + 1] - indptr[v]);
}

/** Fill host float `dst[count]` for logical range [start,start+count) by tiling
 *  a `file_elems`-long source (wrap-around). */
void FillTiled(const std::vector<float> &src, clio::run::u64 file_elems,
               clio::run::u64 start, clio::run::u64 count, float *dst) {
  clio::run::u64 s = start % file_elems, done = 0;
  while (done < count) {
    clio::run::u64 n = std::min(count - done, file_elems - s);
    std::memcpy(dst + done, src.data() + s, n * sizeof(float));
    done += n; s = (s + n) % file_elems;
  }
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

/** Gather the resident vector window [lo,hi) (float elems) into contiguous
 *  scratch (identical to the k-means capacity gather). */
__global__ void GnnGatherKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceView<float> view, clio::run::u64 lo,
                                clio::run::u64 hi, float *scratch) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  v.read_range(lo, hi, [scratch, lo](clio::run::u64 i, float val) {
    scratch[i - lo] = val;
  });
  (void)g_ipc_manager;
}

/** Deterministic graph readout accumulation over the nodes [node_lo, node_lo+nn)
 *  whose features are contiguous in `scratch` (nn * F floats). One thread per
 *  feature f sums deg(v)*feat[v][f] over v in ASCENDING order into pool[f]
 *  (double, no atomics). Called once per window with pool persisting across
 *  windows -> the concatenation over windows sums all nodes in id order. F is
 *  passed so the same kernel serves any feature dim. */
__global__ void GnnReadoutKernel(const float *scratch, clio::run::u64 node_lo,
                                 clio::run::u64 nn, const int *deg_all, int F,
                                 double *pool) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;
  double acc = pool[f];
  for (clio::run::u64 i = 0; i < nn; ++i) {
    double w = (double)deg_all[node_lo + i];
    acc += w * (double)scratch[i * (clio::run::u64)F + f];
  }
  pool[f] = acc;
}

/** Traditional resident readout: features for all N nodes are resident in
 *  `d_feat` (row-major N*F). Same per-feature ascending-order sum -> bit-exact
 *  with the windowed path. */
__global__ void GnnReadoutResidentKernel(const float *d_feat, clio::run::u64 N,
                                         const int *deg_all, int F,
                                         double *pool) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;
  double acc = 0.0;
  for (clio::run::u64 v = 0; v < N; ++v)
    acc += (double)deg_all[v] * (double)d_feat[v * (clio::run::u64)F + f];
  pool[f] = acc;
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: GNN capacity comparison (traditional in-core vs Eternia "
          "zstd streaming) on a feature matrix larger than HBM",
          "[gpu_vector][compress][gnn][capacity][bench]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const std::string dir = std::getenv("CLIO_GNN_DATA")
                              ? std::getenv("CLIO_GNN_DATA")
                              : "/workspace/data/ogb/products";
  clio::run::u64 N0 = 0, E0 = 0; int F = 100, C = 47;
  ReadMeta(dir, N0, F, E0, C);
  g_F = F;

  const clio::run::u32 window =
      std::getenv("CLIO_GNN_WINDOW") ? (clio::run::u32)std::atoi(std::getenv("CLIO_GNN_WINDOW")) : 4;
  const clio::run::u64 page_rows =
      std::getenv("CLIO_GNN_PAGE_ROWS") ? (clio::run::u64)std::atoi(std::getenv("CLIO_GNN_PAGE_ROWS")) : 2560;
  const clio::run::u64 page_size = page_rows * (clio::run::u64)F * sizeof(float);
  const clio::run::u64 epp = page_size / sizeof(float);   // = page_rows * F
  REQUIRE(epp % (clio::run::u64)F == 0);
  // Logical node count (tiled up from the real N0). Rounded to a whole number of
  // window-strides so the SequentialTransaction sweep tiles cleanly.
  clio::run::u64 want_nodes =
      std::getenv("CLIO_GNN_CAP_NODES") ? (clio::run::u64)std::atoll(std::getenv("CLIO_GNN_CAP_NODES")) : N0;
  clio::run::u64 K = (want_nodes + page_rows - 1) / page_rows;   // pages
  K = (K / window) * window; if (K == 0) K = window;
  const clio::run::u64 N = K * page_rows;                        // logical nodes
  const clio::run::u64 total_elems = N * (clio::run::u64)F;
  const clio::run::u64 dataset_bytes = total_elems * sizeof(float);

  size_t gpu_free = 0, gpu_total = 0;
  cudaMemGetInfo(&gpu_free, &gpu_total);
  std::fprintf(stderr,
      "\n============== GNN CAPACITY COMPARISON ==============\n"
      "[GCAP] features=%lluMiB (%llu nodes x %d-d, real ogbn tiled)  GPU=%zuMiB "
      "total %zuMiB free  page=%llu rows (%lluKiB)  window=%u pages (peak "
      "%lluMiB)\n",
      (unsigned long long)(dataset_bytes >> 20), (unsigned long long)N, F,
      gpu_total >> 20, gpu_free >> 20, (unsigned long long)page_rows,
      (unsigned long long)(page_size >> 10), window,
      (unsigned long long)((clio::run::u64)2 * window * page_size >> 20));

  // Real features + degrees, held once, tiled to size.
  std::vector<float> feat; LoadFeatures(dir, N0, F, feat);
  std::vector<int> deg0;   LoadDegrees(dir, N0, deg0);
  const clio::run::u64 file_elems = N0 * (clio::run::u64)F;

  // Resident tiled degree array (the "graph"; small vs features). deg_all[v] =
  // deg0[v mod N0]. Kept resident on both paths (identical weight sequence).
  int *d_deg = nullptr;
  REQUIRE(cudaMalloc(&d_deg, N * sizeof(int)) == cudaSuccess);
  {
    std::vector<int> tile(std::min<clio::run::u64>(N, 1u << 22));
    for (clio::run::u64 off = 0; off < N; off += tile.size()) {
      clio::run::u64 n = std::min<clio::run::u64>(tile.size(), N - off);
      for (clio::run::u64 i = 0; i < n; ++i) tile[i] = deg0[(off + i) % N0];
      cudaMemcpy(d_deg + off, tile.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    }
  }

  double *d_pool = nullptr;
  REQUIRE(cudaMalloc(&d_pool, (size_t)F * sizeof(double)) == cudaSuccess);
  const int tpb = 128, fblocks = (F + tpb - 1) / tpb;

  // =====================================================================
  //  METHOD 1 -- TRADITIONAL in-core: whole feature matrix resident.
  // =====================================================================
  std::string trad_status = "OOM";
  double trad_runtime = -1.0;
  std::vector<double> pool_trad(F, 0.0);
  bool trad_ran = false;
  {
    size_t free_b = 0, total_b = 0; cudaMemGetInfo(&free_b, &total_b);
    float *d_all = nullptr;
    cudaError_t alloc = cudaErrorMemoryAllocation;
    if (dataset_bytes + (size_t(384) << 20) < free_b)
      alloc = cudaMalloc(&d_all, dataset_bytes);
    if (alloc != cudaSuccess) {
      cudaGetLastError();
      std::fprintf(stderr,
          "[GCAP] TRADITIONAL: *** OOM *** cannot place %lluMiB feature matrix in "
          "%zuMiB free HBM -> in-core GNN CANNOT run this graph.\n",
          (unsigned long long)(dataset_bytes >> 20), free_b >> 20);
    } else {
      for (clio::run::u64 off = 0; off < total_elems; off += file_elems) {
        clio::run::u64 n = std::min(file_elems, total_elems - off);
        cudaMemcpy(d_all + off, feat.data(), n * sizeof(float), cudaMemcpyHostToDevice);
      }
      cudaMemset(d_pool, 0, (size_t)F * sizeof(double));
      double t0 = NowSec();
      GnnReadoutResidentKernel<<<fblocks, tpb>>>(d_all, N, d_deg, F, d_pool);
      ctp::GpuApi::Synchronize();
      trad_runtime = NowSec() - t0;
      cudaMemcpy(pool_trad.data(), d_pool, (size_t)F * sizeof(double), cudaMemcpyDeviceToHost);
      trad_status = "OK"; trad_ran = true;
      cudaFree(d_all);
      std::fprintf(stderr,
          "[GCAP] TRADITIONAL: OK in %.3fs (pinned %lluMiB of HBM). pool[0]=%.6e\n",
          trad_runtime, (unsigned long long)(dataset_bytes >> 20), pool_trad[0]);
    }
  }

  // =====================================================================
  //  METHOD 2 -- ETERNIA: store zstd-compressed + tiered, stream a window.
  // =====================================================================
  const char *tag = "gnn_cap";
  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
  clio::cte::core::Client core; core.Init(clio::cte::core::kCtePoolId);
  auto tagf = core.AsyncGetOrCreateTag(tag); tagf.Wait();
  REQUIRE(tagf->GetReturnCode() == 0);
  auto tag_id = tagf->tag_id_;
  clio::cte::core::Client comp; comp.Init(StoragePool());
  clio::cte::core::Context ctx; ctx.dynamic_compress_ = 1; ctx.compress_preset_ = 2;

  clio::run::bdev::Client hbm_bdev(clio::run::PoolId(512, 1));
  clio::run::bdev::Client dram_bdev(clio::run::PoolId(513, 1));
  auto h0 = hbm_bdev.AsyncGetStats(); h0.Wait();
  auto r0 = dram_bdev.AsyncGetStats(); r0.Wait();
  clio::run::u64 hbm_rem0 = h0->remaining_size_, dram_rem0 = r0->remaining_size_;

  auto free_mib = []{ size_t f=0,t=0; cudaMemGetInfo(&f,&t); return (long long)(f>>20); };
  const long long mem_base = free_mib();

  // Store one page at a time from a HOST buffer -- zstd is a CPU codec, so it
  // compresses host memory directly (no device staging needed on the store side).
  std::vector<float> pagebuf_h(epp);
  double store_t0 = NowSec();
  for (clio::run::u64 p = 0; p < K; ++p) {
    FillTiled(feat, file_elems, p * epp, epp, pagebuf_h.data());
    ctp::ipc::ShmPtr<> dp; dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(pagebuf_h.data());
    std::string name = std::string(tag) + "_b0_pi" + std::to_string(p);
    auto pf = comp.AsyncPutBlob(tag_id, name, (clio::run::u64)0, page_size, dp,
                                0.5f, ctx, 0, clio::run::PoolQuery::Local());
    pf.Wait();
    REQUIRE(pf->GetReturnCode() == 0);
  }
  double store_dt = NowSec() - store_t0;

  auto h1 = hbm_bdev.AsyncGetStats(); h1.Wait();
  auto r1 = dram_bdev.AsyncGetStats(); r1.Wait();
  clio::run::u64 hbm_used = (hbm_rem0 >= h1->remaining_size_) ? hbm_rem0 - h1->remaining_size_ : 0;
  clio::run::u64 dram_used = (dram_rem0 >= r1->remaining_size_) ? dram_rem0 - r1->remaining_size_ : 0;
  clio::run::u64 stored = hbm_used + dram_used;
  double ratio = stored ? (double)dataset_bytes / (double)stored : 0.0;
  std::fprintf(stderr,
      "[GCAP] ETERNIA: stored %lluMiB in %.2fs -> %lluMiB zstd (%.3fx)  split "
      "HBM=%lluMiB + DRAM=%lluMiB\n", (unsigned long long)(dataset_bytes >> 20),
      store_dt, (unsigned long long)(stored >> 20), ratio,
      (unsigned long long)(hbm_used >> 20), (unsigned long long)(dram_used >> 20));

  gv::Vector<float> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                        /*gpu_pages_per_block=*/2 * window,
                        /*host_pages_per_block=*/0, page_size,
                        /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                        /*manager_threads_per_block=*/32,
                        /*allow_cold_miss_fault=*/false,
                        /*storage_pool_id=*/StoragePool());
  const clio::run::u64 win_elems = (clio::run::u64)window * epp;
  float *d_scratch = nullptr;
  REQUIRE(cudaMalloc(&d_scratch, win_elems * sizeof(float)) == cudaSuccess);
  const long long mem_working = free_mib();

  cudaMemset(d_pool, 0, (size_t)F * sizeof(double));
  double km_t0 = NowSec();
  gv::SequentialTransaction<float> sweep(vec, /*first_page=*/0, /*npages=*/K);
  sweep.Iterate([&](clio::run::u64 lo, clio::run::u64 hi,
                    gv::DeviceView<float> v, cudaStream_t s) {
    GnnGatherKernel<<<1, 32, 0, s>>>(gpu_info, v, lo, hi, d_scratch);
    const clio::run::u64 node_lo = lo / (clio::run::u64)F;
    const clio::run::u64 nn = (hi - lo) / (clio::run::u64)F;
    GnnReadoutKernel<<<fblocks, tpb, 0, s>>>(d_scratch, node_lo, nn, d_deg, F, d_pool);
  });
  ctp::GpuApi::Synchronize();
  double km_dt = NowSec() - km_t0;
  std::vector<double> pool_eternia(F, 0.0);
  cudaMemcpy(pool_eternia.data(), d_pool, (size_t)F * sizeof(double), cudaMemcpyDeviceToHost);

  const clio::run::u64 peak_win_b = win_elems * sizeof(float);
  std::fprintf(stderr,
      "[GCAP] ETERNIA: OK, readout in %.3fs  peak GPU window=%lluMiB (%.0fx "
      "smaller than the %lluMiB features)  pool[0]=%.6e\n",
      km_dt, (unsigned long long)(peak_win_b >> 20),
      (double)dataset_bytes / (double)peak_win_b,
      (unsigned long long)(dataset_bytes >> 20), pool_eternia[0]);
  std::fprintf(stderr,
      "[MEM] free HBM base=%lldMiB -> working=%lldMiB (Eternia working set "
      "%lldMiB incl. %lluMiB stream window + %lluMiB degrees)\n",
      mem_base, mem_working, mem_base - mem_working,
      (unsigned long long)(peak_win_b >> 20),
      (unsigned long long)((N * sizeof(int)) >> 20));

  // ---- Bit-exact verify (only where traditional also ran) ----
  std::string exact = "n/a(OOM)";
  if (trad_ran) {
    int mism = std::memcmp(pool_trad.data(), pool_eternia.data(),
                           (size_t)F * sizeof(double));
    double maxrel = 0.0;
    for (int f = 0; f < F; ++f) {
      double a = pool_trad[f], b = pool_eternia[f];
      double denom = std::max(1e-300, std::fabs(a));
      maxrel = std::max(maxrel, std::fabs(a - b) / denom);
    }
    exact = (mism == 0) ? "BIT-EXACT" : "DIFF";
    std::fprintf(stderr,
        "[GCAP] VERIFY vs traditional: memcmp=%d max_rel=%.3e -> %s\n",
        mism, maxrel, exact.c_str());
    REQUIRE(mism == 0);   // lossless + identical summation order => bit-identical
  } else {
    std::fprintf(stderr,
        "[GCAP] VERIFY: traditional OOM'd -> no in-core baseline to compare; "
        "Eternia is the ONLY method that ran this %lluMiB feature matrix.\n",
        (unsigned long long)(dataset_bytes >> 20));
  }
  std::fprintf(stderr, "====================================================\n");

  // ---- CSV row (CLIO_GNN_CSV=<path>, appended; header if new) ----
  {
    char row[512];
    std::snprintf(row, sizeof(row),
        "%llu,%llu,%llu,%u,%s,%.3f,%.3f,%.3f,%llu,%.3f,%llu,%llu,%llu,%s",
        (unsigned long long)(dataset_bytes >> 20), (unsigned long long)N,
        (unsigned long long)page_rows, window, trad_status.c_str(),
        trad_runtime, store_dt, km_dt, (unsigned long long)(stored >> 20), ratio,
        (unsigned long long)(hbm_used >> 20), (unsigned long long)(dram_used >> 20),
        (unsigned long long)(peak_win_b >> 20), exact.c_str());
    const char *hdr =
        "features_mib,nodes,page_rows,window_pages,traditional_status,"
        "traditional_s,eternia_store_s,eternia_readout_s,eternia_stored_mib,"
        "eternia_ratio,hbm_used_mib,dram_used_mib,eternia_peak_gpu_mib,bit_exact";
    std::fprintf(stdout, "[CSV],%s\n[CSV],%s\n", hdr, row); std::fflush(stdout);
    const char *csv_path = std::getenv("CLIO_GNN_CSV");
    if (csv_path) {
      std::ifstream probe(csv_path);
      const bool empty = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
      probe.close();
      std::ofstream out(csv_path, std::ios::app);
      if (out) { if (empty) out << hdr << "\n"; out << row << "\n"; }
    }
  }

  cudaFree(d_deg); cudaFree(d_pool); cudaFree(d_scratch);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif
