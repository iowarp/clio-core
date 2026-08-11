/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Capacity comparison: k-means on a dataset LARGER THAN GPU MEMORY, the
 * traditional in-core way vs. through the compressed/tiered GPU vector (Eternia).
 *
 * The point Eternia makes is capacity decoupling: a traditional GPU k-means (e.g.
 * FAISS GpuIndexFlat) must hold the ENTIRE point set resident in HBM, so its peak
 * GPU memory grows with the dataset and it OOMs once the data exceeds HBM. Eternia
 * instead stores the points compressed across a tier stack (HBM->DRAM->...) and
 * streams a small fixed-size window through HBM for each k-means sweep, so its peak
 * GPU memory is O(window) -- constant -- regardless of dataset size. Same workload,
 * same real data (SIFT), but only Eternia runs once the data is bigger than HBM.
 *
 * This program runs BOTH methods at one dataset size and prints a comparison:
 *   - Traditional: cudaMalloc the whole dataset (what an in-core GPU k-means needs
 *     to do), then run k-means. If the allocation fails -> OOM, reported, skipped.
 *   - Eternia: store the dataset compressed one page at a time, then run k-means as
 *     a windowed streaming sweep. Reports compression, peak-window bytes, inertia.
 *
 * The real SIFT base set (1M x 128-d) is held ONCE in host memory (~512 MB) and
 * TILED to reach the requested logical size (CLIO_KM_PAGES pages), so arbitrarily
 * large datasets can be built from the real vectors without a larger download.
 *
 * NOTE: this is a capacity/where-it-runs comparison, NOT a correctness test -- at
 * the OOM crossover there is no traditional baseline to check against. Correctness
 * (compressed clustering == uncompressed) is verified in
 * test_gpu_vector_kmeans_real_gpu.cc at sizes that fit.
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

// Whether the optional deeper tiers were configured (env-gated in EnsureInit).
inline bool nvme_env_present() {
  return std::getenv("CLIO_KM_NVME_MIB") && std::getenv("CLIO_KM_NVME_DIR");
}
inline bool pfs_env_present() {
  return std::getenv("CLIO_KM_PFS_MIB") && std::getenv("CLIO_KM_PFS_DIR");
}

constexpr int kDims = 128;
constexpr int kClusters = 256;

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
  std::string cfg = "/tmp/gpu_vec_kmeans_cap_" + std::to_string(port) + ".yaml";
  const char *hbm_env = std::getenv("CLIO_KM_HBM_MIB");
  const char *dram_env = std::getenv("CLIO_KM_DRAM_MIB");
  int hbm_mib = hbm_env ? std::atoi(hbm_env) : 2048;    // small HBM tier -> spill
  int dram_mib = dram_env ? std::atoi(dram_env) : 12288; // DRAM overflow tier
  // Optional deeper tiers: local NVMe scratch then Lustre/PFS (file bdevs).
  // Set CLIO_KM_NVME_MIB + CLIO_KM_NVME_DIR and/or CLIO_KM_PFS_MIB +
  // CLIO_KM_PFS_DIR to force the compressed matrix to cascade HBM->DRAM->NVMe->PFS
  // once HBM and DRAM fill. Scores descend so max_bw fills faster tiers first.
  const char *nvme_env = std::getenv("CLIO_KM_NVME_MIB");
  const char *nvme_dir = std::getenv("CLIO_KM_NVME_DIR");
  const char *pfs_env = std::getenv("CLIO_KM_PFS_MIB");
  const char *pfs_dir = std::getenv("CLIO_KM_PFS_DIR");
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
      << dram_mib << "MB\"\n        score: 0.75\n";
    if (nvme_env && nvme_dir) {
      f << "      - path: \"" << nvme_dir << "/cte_nvme_tier\"\n"
        << "        bdev_type: \"file\"\n        capacity_limit: \""
        << std::atoi(nvme_env) << "MB\"\n        score: 0.5\n";
    }
    if (pfs_env && pfs_dir) {
      f << "      - path: \"" << pfs_dir << "/cte_pfs_tier\"\n"
        << "        bdev_type: \"file\"\n        capacity_limit: \""
        << std::atoi(pfs_env) << "MB\"\n        score: 0.25\n";
    }
    f << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

/** Load the whole SIFT .fvecs base into a compact [N0*kDims] host array,
 *  normalized to [0,1]. Returns the number of vectors loaded (N0). */
clio::run::u64 LoadSiftAll(const std::string &path, std::vector<float> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "[CAP] cannot open %s\n", path.c_str()); return 0; }
  f.seekg(0, std::ios::end);
  const std::streamoff bytes = f.tellg();
  f.seekg(0, std::ios::beg);
  const clio::run::u64 rec = 4 + (clio::run::u64)kDims * 4;  // int32 dim + kDims f32
  const clio::run::u64 n0 = (clio::run::u64)bytes / rec;
  out.resize(n0 * (clio::run::u64)kDims);
  for (clio::run::u64 p = 0; p < n0; ++p) {
    std::int32_t dim = 0;
    f.read(reinterpret_cast<char *>(&dim), sizeof(dim));
    if (!f || dim != kDims) { std::fprintf(stderr, "[CAP] bad record %llu dim=%d\n",
                                           (unsigned long long)p, dim); return 0; }
    f.read(reinterpret_cast<char *>(out.data() + p * kDims), sizeof(float) * kDims);
    for (int d = 0; d < kDims; ++d) out[p * kDims + d] *= (1.0f / 255.0f);
  }
  return n0;
}

/** Fill host `dst[count]` for the logical float range [start_elem, start_elem+count)
 *  by TILING the SIFT array of `file_elems` floats (wrap-around). */
void FillTiled(const std::vector<float> &sift, clio::run::u64 file_elems,
               clio::run::u64 start_elem, clio::run::u64 count, float *dst) {
  clio::run::u64 s = start_elem % file_elems, done = 0;
  while (done < count) {
    clio::run::u64 n = std::min(count - done, file_elems - s);
    std::memcpy(dst + done, sift.data() + s, n * sizeof(float));
    done += n; s = (s + n) % file_elems;
  }
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

__global__ void KmGatherKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceView<float> view, clio::run::u64 lo,
                               clio::run::u64 hi, float *scratch) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  dev::vector<float> v(view, g_ipc_manager_ptr);
  v.read_range(lo, hi, [scratch, lo](clio::run::u64 i, float val) {
    scratch[i - lo] = val;
  });
  (void)g_ipc_manager;
}

__global__ void KmAssignKernel(const float *scratch, clio::run::u64 npts,
                               const float *centroids, double *sums, int *counts,
                               double *inertia) {
  for (clio::run::u64 p = blockIdx.x * (clio::run::u64)blockDim.x + threadIdx.x;
       p < npts; p += (clio::run::u64)gridDim.x * blockDim.x) {
    const float *pt = &scratch[p * kDims];
    int best = 0;
    float best_d2 = 3.0e38f;
    for (int c = 0; c < kClusters; ++c) {
      float d2 = 0.0f;
      for (int d = 0; d < kDims; ++d) {
        float diff = pt[d] - centroids[c * kDims + d];
        d2 += diff * diff;
      }
      if (d2 < best_d2) { best_d2 = d2; best = c; }
    }
    for (int d = 0; d < kDims; ++d)
      atomicAdd(&sums[best * kDims + d], (double)pt[d]);
    atomicAdd(&counts[best], 1);
    atomicAdd(inertia, (double)best_d2);
  }
}

#if !CTP_IS_DEVICE_PASS

static void UpdateCentroids(const std::vector<double> &sums,
                            const std::vector<int> &counts,
                            std::vector<float> &centroids) {
  for (int c = 0; c < kClusters; ++c)
    if (counts[c] > 0)
      for (int d = 0; d < kDims; ++d)
        centroids[c * kDims + d] =
            (float)(sums[c * kDims + d] / (double)counts[c]);
}

TEST_CASE("gpu_vector: k-means capacity comparison (traditional in-core vs "
          "Eternia streaming) on a dataset larger than HBM",
          "[gpu_vector][compress][kmeans][capacity][bench]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  // HBM working-window size (pages). MegaMmap Fig-8 analog: shrinking this is the
  // GPU equivalent of lowering per-node DRAM -- the resident footprint the vector
  // keeps in real HBM. The vector double-buffers, so peak window = 2*window pages.
  const clio::run::u32 window =
      std::getenv("CLIO_KM_WINDOW")
          ? (clio::run::u32)std::atoi(std::getenv("CLIO_KM_WINDOW")) : 4;
  // Compression ablation: 1 = Eternia (cuSZp), 0 = tiering only (MegaMmap-style,
  // store raw through the tiers). Lets us isolate compression's capacity benefit.
  const int do_compress =
      std::getenv("CLIO_KM_COMPRESS") ? std::atoi(std::getenv("CLIO_KM_COMPRESS")) : 1;
  const char *pk_env = std::getenv("CLIO_KM_PAGE_KIB");
  const clio::run::u64 page_size =
      (pk_env ? (clio::run::u64)std::atoi(pk_env) : 1024ULL) * 1024;  // 1 MiB pages
  const clio::run::u64 epp = page_size / sizeof(float);
  REQUIRE(epp % kDims == 0);
  const char *kp_env = std::getenv("CLIO_KM_PAGES");
  clio::run::u32 K = kp_env ? (clio::run::u32)std::atoi(kp_env) : 4096;  // 4 GiB
  K = (K / window) * window;
  if (K == 0) K = window;
  const clio::run::u64 total_elems = (clio::run::u64)K * epp;
  const clio::run::u64 npoints = total_elems / kDims;
  const int iters = std::getenv("CLIO_KM_ITERS")
                        ? std::atoi(std::getenv("CLIO_KM_ITERS")) : 4;
  const clio::run::u64 dataset_bytes = total_elems * sizeof(float);
  const std::string data_path =
      std::getenv("CLIO_KM_DATA") ? std::getenv("CLIO_KM_DATA")
                                  : "/workspace/data/sift/sift/sift_base.fvecs";

  size_t gpu_free = 0, gpu_total = 0;
  cudaMemGetInfo(&gpu_free, &gpu_total);
  std::fprintf(stderr,
      "\n================ CAPACITY COMPARISON ================\n"
      "[CAP] dataset=%lluMiB (%llu pts x %d-d SIFT, tiled)  GPU=%zuMiB total, "
      "%zuMiB free  iters=%d  window=%u pages (%lluMiB)  mode=%s\n",
      (unsigned long long)(dataset_bytes >> 20), (unsigned long long)npoints,
      kDims, gpu_total >> 20, gpu_free >> 20, iters, window,
      (unsigned long long)((clio::run::u64)2 * window * page_size >> 20),
      do_compress ? "ETERNIA(compress+tier)" : "TIER-ONLY(no compress)");

  // Real SIFT held once, tiled to the requested size.
  std::vector<float> sift;
  const clio::run::u64 n0 = LoadSiftAll(data_path, sift);
  REQUIRE(n0 > 0);
  const clio::run::u64 file_elems = n0 * (clio::run::u64)kDims;

  // Shared centroid init: first kClusters tiled points.
  std::vector<float> init(kClusters * kDims);
  FillTiled(sift, file_elems, 0, (clio::run::u64)kClusters * kDims, init.data());

  float *d_centroids = nullptr; double *d_sums = nullptr, *d_inertia = nullptr;
  int *d_counts = nullptr;
  REQUIRE(cudaMalloc(&d_centroids, kClusters * kDims * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_sums, kClusters * kDims * sizeof(double)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_counts, kClusters * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_inertia, sizeof(double)) == cudaSuccess);
  std::vector<double> h_sums(kClusters * kDims);
  std::vector<int> h_counts(kClusters);

  // Whole-process GPU peak: max over samples of (total - free) via cudaMemGetInfo,
  // sampled every window during store + k-means. Captures the CUDA context
  // (~300 MiB floor), cuSZp's own context, the HBM-resident compressed cache
  // (CLIO_KM_HBM_MIB cap) and the streaming window -- i.e. the TRUE resident GPU
  // footprint, NOT just the window buffer. Reset per method so each method's peak
  // reflects its own residency; peak stays BOUNDED (~HBM_cap+overhead) and flat vs
  // dataset for Eternia, but grows == dataset for the in-core baseline.
  size_t g_total = 0, g_free_min = SIZE_MAX;
  auto sample_peak = [&]{ size_t f=0,t=0; if (cudaMemGetInfo(&f,&t)==cudaSuccess){
      g_total=t; if (f<g_free_min) g_free_min=f; } };
  auto peak_mib = [&]{ return (g_total>g_free_min)?(long long)((g_total-g_free_min)>>20):0; };

  // =====================================================================
  //  METHOD 1 -- TRADITIONAL in-core GPU k-means: whole dataset resident.
  // =====================================================================
  std::string trad_status = "OOM";   // for the CSV row
  double trad_runtime = -1.0;
  double trad_inertia = 0.0;         // in-core k-means inertia (parity baseline)
  long long trad_peak_mib = -1;      // whole-process peak where it ran; -1 if OOM
  {
    g_free_min = SIZE_MAX;  // per-method peak window
    std::fprintf(stderr,
        "[CAP] TRADITIONAL: needs %lluMiB resident in HBM (peak GPU mem = "
        "dataset)...\n", (unsigned long long)(dataset_bytes >> 20));
    // Only attempt the allocation when it plausibly fits (leave headroom for the
    // runtime's own GPU state). A doomed multi-GB cudaMalloc can abort the process
    // rather than return an error cleanly, so gate on the live free-memory query.
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    float *d_all = nullptr;
    cudaError_t alloc = cudaErrorMemoryAllocation;
    if (dataset_bytes + (size_t(256) << 20) < free_b) {
      alloc = cudaMalloc(&d_all, dataset_bytes);
    }
    if (alloc != cudaSuccess) {
      cudaGetLastError();  // clear sticky error
      std::fprintf(stderr,
          "[CAP] TRADITIONAL: *** OOM *** cannot place %lluMiB dataset in %zuMiB "
          "of free HBM -> traditional in-core k-means CANNOT run this dataset.\n",
          (unsigned long long)(dataset_bytes >> 20), free_b >> 20);
    } else {
      // Fill the resident dataset from the tiled SIFT set, in file-sized chunks.
      for (clio::run::u64 off = 0; off < total_elems; off += file_elems) {
        clio::run::u64 n = std::min(file_elems, total_elems - off);
        cudaMemcpy(d_all + off, sift.data(), n * sizeof(float), cudaMemcpyHostToDevice);
      }
      sample_peak();  // whole dataset now resident
      std::vector<float> cen(init);
      double prev = 0.0, t0 = NowSec();
      for (int it = 0; it < iters; ++it) {
        cudaMemcpy(d_centroids, cen.data(), cen.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_sums, 0, kClusters * kDims * sizeof(double));
        cudaMemset(d_counts, 0, kClusters * sizeof(int));
        cudaMemset(d_inertia, 0, sizeof(double));
        int threads = 256;
        int blocks = (int)std::min<clio::run::u64>(65535, (npoints + threads - 1) / threads);
        KmAssignKernel<<<blocks, threads>>>(d_all, npoints, d_centroids, d_sums, d_counts, d_inertia);
        ctp::GpuApi::Synchronize();
        sample_peak();
        cudaMemcpy(h_sums.data(), d_sums, h_sums.size() * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&prev, d_inertia, sizeof(double), cudaMemcpyDeviceToHost);
        UpdateCentroids(h_sums, h_counts, cen);
      }
      double dt = NowSec() - t0;
      trad_status = "OK"; trad_runtime = dt; trad_peak_mib = peak_mib(); trad_inertia = prev;
      std::fprintf(stderr,
          "[CAP] TRADITIONAL: OK, ran in %.2fs, final inertia=%.1f  peak GPU="
          "%lldMiB (whole dataset resident + context).\n", dt, prev, trad_peak_mib);
      cudaFree(d_all);
    }
  }

  // =====================================================================
  //  METHOD 2 -- ETERNIA: store compressed + tiered, stream a fixed window.
  // =====================================================================
  const char *tag = "kmeans_cap";
  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);
  clio::cte::core::Client core; core.Init(clio::cte::core::kCtePoolId);
  auto tagf = core.AsyncGetOrCreateTag(tag); tagf.Wait();
  REQUIRE(tagf->GetReturnCode() == 0);
  auto tag_id = tagf->tag_id_;
  // Compress ablation routes storage differently: with compression, writes go
  // through the compressor pool (600 -> compresses -> forwards to cte_core 512);
  // tiering-only writes go STRAIGHT to cte_core (512) so pages are stored raw
  // through the same tier stack (this is the MegaMmap-equivalent path -- tiering
  // without compression). Routing through 600 always compresses (the cuSZp pin
  // ignores ctx.dynamic_compress_), so the pool choice is what actually ablates.
  const clio::run::PoolId storage_pool =
      do_compress ? clio::run::PoolId(600, 0) : clio::run::PoolId(512, 0);
  clio::cte::core::Client comp; comp.Init(storage_pool);
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = do_compress;   // 0 = tiering only (no cuSZp)
  ctx.compress_preset_ = 2;

  const bool have_nvme = nvme_env_present();
  const bool have_pfs = pfs_env_present();
  clio::run::bdev::Client hbm_bdev(clio::run::PoolId(512, 1));
  clio::run::bdev::Client dram_bdev(clio::run::PoolId(513, 1));
  clio::run::bdev::Client nvme_bdev(clio::run::PoolId(514, 1));
  clio::run::bdev::Client pfs_bdev(clio::run::PoolId(515, 1));
  auto h0 = hbm_bdev.AsyncGetStats(); h0.Wait();
  auto r0 = dram_bdev.AsyncGetStats(); r0.Wait();
  clio::run::u64 hbm_rem0 = h0->remaining_size_, dram_rem0 = r0->remaining_size_;
  clio::run::u64 nvme_rem0 = 0, pfs_rem0 = 0;
  if (have_nvme) { auto n0=nvme_bdev.AsyncGetStats(); n0.Wait(); nvme_rem0=n0->remaining_size_; }
  if (have_pfs)  { auto p0=pfs_bdev.AsyncGetStats();  p0.Wait(); pfs_rem0=p0->remaining_size_; }

  std::fprintf(stderr,
      "[CAP] ETERNIA: storing %lluMiB compressed across HBM+DRAM tiers (peak GPU "
      "mem = one %lluKiB window, NOT the dataset)...\n",
      (unsigned long long)(dataset_bytes >> 20),
      (unsigned long long)((clio::run::u64)2 * window * page_size >> 10));

  // [MEM] GPU-memory probes: is the "hbm" storage tier real device memory, and
  // what is the TRUE resident GPU footprint (not just the gather scratch)?
  auto free_mib = []{ size_t f=0,t=0; cudaMemGetInfo(&f,&t); return (long long)(f>>20); };
  const long long mem_base = free_mib();   // free HBM before storing anything
  g_free_min = SIZE_MAX;                    // per-method peak window (Eternia)
  sample_peak();                            // baseline: context + HBM tier already resident

  std::vector<float> pagebuf_h(epp);
  float *pagebuf = nullptr;
  REQUIRE(cudaMalloc(&pagebuf, page_size) == cudaSuccess);
  double store_t0 = NowSec();
  for (clio::run::u32 p = 0; p < K; ++p) {
    FillTiled(sift, file_elems, (clio::run::u64)p * epp, epp, pagebuf_h.data());
    REQUIRE(cudaMemcpy(pagebuf, pagebuf_h.data(), page_size,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    ctp::GpuApi::Synchronize();
    ctp::ipc::ShmPtr<> dp; dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(pagebuf);
    std::string name = std::string(tag) + "_b0_pi" + std::to_string(p);
    // Blob score 1.0: every vector page is equally "the working set" and wants
    // the fastest tier, so all tiers are in the DPE's preferred group and fill
    // in configured-score order HBM->DRAM->NVMe->PFS (cascading on overflow).
    auto pf = comp.AsyncPutBlob(tag_id, name, (clio::run::u64)0, page_size, dp,
                                1.0f, ctx, 0, clio::run::PoolQuery::Local());
    pf.Wait();
    REQUIRE(pf->GetReturnCode() == 0);
    if ((p & 0x3f) == 0) sample_peak();  // HBM compressed cache fills up as we store
  }
  cudaFree(pagebuf);
  sample_peak();
  double store_dt = NowSec() - store_t0;
  const long long mem_after_store = free_mib();  // if hbm tier were real HBM, this drops ~= stored

  auto h1 = hbm_bdev.AsyncGetStats(); h1.Wait();
  auto r1 = dram_bdev.AsyncGetStats(); r1.Wait();
  clio::run::u64 hbm_used = (hbm_rem0 >= h1->remaining_size_) ? hbm_rem0 - h1->remaining_size_ : 0;
  clio::run::u64 dram_used = (dram_rem0 >= r1->remaining_size_) ? dram_rem0 - r1->remaining_size_ : 0;
  clio::run::u64 nvme_used = 0, pfs_used = 0;
  if (have_nvme) { auto n1=nvme_bdev.AsyncGetStats(); n1.Wait();
                   nvme_used = (nvme_rem0 >= n1->remaining_size_) ? nvme_rem0 - n1->remaining_size_ : 0; }
  if (have_pfs)  { auto p1=pfs_bdev.AsyncGetStats();  p1.Wait();
                   pfs_used = (pfs_rem0 >= p1->remaining_size_) ? pfs_rem0 - p1->remaining_size_ : 0; }
  clio::run::u64 stored = hbm_used + dram_used + nvme_used + pfs_used;
  double ratio = stored ? (double)dataset_bytes / (double)stored : 0.0;
  std::fprintf(stderr,
      "[CAP] ETERNIA: stored in %.2fs -> %lluMiB compressed (%.2fx)  split: "
      "HBM=%lluMiB + DRAM=%lluMiB + NVMe=%lluMiB + PFS=%lluMiB\n", store_dt,
      (unsigned long long)(stored >> 20), ratio,
      (unsigned long long)(hbm_used >> 20), (unsigned long long)(dram_used >> 20),
      (unsigned long long)(nvme_used >> 20), (unsigned long long)(pfs_used >> 20));

  gv::Vector<float> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                        /*gpu_pages_per_block=*/2 * window,
                        /*host_pages_per_block=*/0, page_size,
                        /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                        /*manager_threads_per_block=*/32,
                        /*allow_cold_miss_fault=*/false,
                        /*storage_pool_id=*/storage_pool);
  const clio::run::u64 win_elems = (clio::run::u64)window * epp;
  float *d_scratch = nullptr;
  REQUIRE(cudaMalloc(&d_scratch, win_elems * sizeof(float)) == cudaSuccess);

  const long long mem_working = free_mib();  // after vector page-cache + scratch allocated
  std::fprintf(stderr,
      "[MEM] free HBM: base=%lldMiB -> after store %lldMiB (store used %lldMiB of GPU) "
      "-> after vector+scratch %lldMiB (working set %lldMiB).  gather=<<<1,32>>>  "
      "assign=<<<%d blocks,256>>>  d_scratch=%lluMiB  vec_cache=2x%u pages=%lluMiB\n",
      mem_base, mem_after_store, mem_base - mem_after_store, mem_working,
      mem_base - mem_working,
      (int)(( (win_elems / kDims) + 255) / 256),
      (unsigned long long)(win_elems * sizeof(float) >> 20),
      window, (unsigned long long)((clio::run::u64)2 * window * page_size >> 20));

  std::vector<float> cen(init);
  double prev = 0.0, km_t0 = NowSec();
  for (int it = 0; it < iters; ++it) {
    cudaMemcpy(d_centroids, cen.data(), cen.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sums, 0, kClusters * kDims * sizeof(double));
    cudaMemset(d_counts, 0, kClusters * sizeof(int));
    cudaMemset(d_inertia, 0, sizeof(double));
    gv::SequentialTransaction<float> sweep(vec, /*first_page=*/0, /*npages=*/K);
    sweep.Iterate([&](clio::run::u64 lo, clio::run::u64 hi,
                      gv::DeviceView<float> v, cudaStream_t s) {
      KmGatherKernel<<<1, 32, 0, s>>>(gpu_info, v, lo, hi, d_scratch);
      const clio::run::u64 npts = (hi - lo) / kDims;
      const int threads = 256;
      const int blocks = (int)((npts + threads - 1) / threads);
      KmAssignKernel<<<blocks, threads, 0, s>>>(d_scratch, npts, d_centroids,
                                                d_sums, d_counts, d_inertia);
      sample_peak();  // window + scratch + HBM cache all resident during the sweep
    });
    ctp::GpuApi::Synchronize();
    sample_peak();
    cudaMemcpy(h_sums.data(), d_sums, h_sums.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&prev, d_inertia, sizeof(double), cudaMemcpyDeviceToHost);
    clio::run::u64 assigned = 0;
    for (int c = 0; c < kClusters; ++c) assigned += (clio::run::u64)h_counts[c];
    UpdateCentroids(h_sums, h_counts, cen);
    REQUIRE(assigned == npoints);
  }
  double km_dt = NowSec() - km_t0;
  const long long eternia_peak_mib = peak_mib();  // whole-process, incl HBM cache+context
  std::fprintf(stderr,
      "[CAP] ETERNIA: OK, k-means ran in %.2fs, final inertia=%.1f  peak GPU="
      "%lldMiB (whole-process: HBM cache+cuSZp+window+context; %.1fx smaller than "
      "the %lluMiB dataset).\n",
      km_dt, prev, eternia_peak_mib,
      eternia_peak_mib > 0 ? (double)(dataset_bytes >> 20) / (double)eternia_peak_mib : 0.0,
      (unsigned long long)(dataset_bytes >> 20));
  const double eternia_inertia = prev;
  std::fprintf(stderr, "====================================================\n");

  // =====================================================================
  //  Track-A extra baselines (CLIO_KM_TRACKA=1): same k-means, same init,
  //  same iters, whole-process peak measured -- so every method is directly
  //  comparable (throughput, time/iter, inertia, peak_gpu) on ONE A100.
  //   METHOD 3 UVM     : cudaMallocManaged whole dataset, in-core kernel
  //                      (driver pages in/out -> oversubscription past HBM).
  //   METHOD 4 STAGED  : dataset in host RAM, stream fixed windows H2D and
  //                      run the in-core kernel per window (explicit staging,
  //                      no compression). Peak stays ~window like Eternia, but
  //                      it moves UNCOMPRESSED bytes and is host-RAM capped.
  // =====================================================================
  const bool want_tracka = std::getenv("CLIO_KM_TRACKA") &&
                           std::atoi(std::getenv("CLIO_KM_TRACKA")) != 0;
  std::string uvm_status = "SKIP"; double uvm_km = -1.0, uvm_inertia = 0.0; long long uvm_peak = -1;
  std::string stg_status = "SKIP"; double stg_km = -1.0, stg_inertia = 0.0; long long stg_peak = -1;
  if (want_tracka) {
    // ---- METHOD 3: UVM ----
    // UVM oversubscription runs a full k-means sweep over managed memory; once the
    // dataset exceeds HBM the driver thrashes pages host<->device and a single
    // sweep becomes impractically slow. Gate on CLIO_KM_UVM_MAX_GIB (default 40 =
    // HBM) so large sizes are recorded as OVERSUB_IMPRACTICAL rather than hanging.
    const long long uvm_max_gib = std::getenv("CLIO_KM_UVM_MAX_GIB")
        ? std::atoll(std::getenv("CLIO_KM_UVM_MAX_GIB")) : 40;
    if ((long long)(dataset_bytes >> 30) > uvm_max_gib) {
      uvm_status = "OVERSUB_IMPRACTICAL";
      std::fprintf(stderr, "[CAP] UVM: skipped at %lluGiB (> %lldGiB HBM: full "
                   "k-means sweep thrashes managed pages, impractically slow)\n",
                   (unsigned long long)(dataset_bytes >> 30), uvm_max_gib);
    } else {
    uvm_status = "OOM";
    float *m_all = nullptr;
    cudaError_t a = cudaMallocManaged(&m_all, dataset_bytes);
    if (a != cudaSuccess) {
      cudaGetLastError();
      std::fprintf(stderr, "[CAP] UVM: *** alloc failed *** %lluMiB managed\n",
                   (unsigned long long)(dataset_bytes >> 20));
    } else {
      for (clio::run::u64 off = 0; off < total_elems; off += file_elems) {
        clio::run::u64 n = std::min(file_elems, total_elems - off);
        std::memcpy(m_all + off, sift.data(), n * sizeof(float));
      }
      g_free_min = SIZE_MAX; sample_peak();
      std::vector<float> ucen(init); double uprev = 0.0, ut0 = NowSec();
      for (int it = 0; it < iters; ++it) {
        cudaMemcpy(d_centroids, ucen.data(), ucen.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_sums, 0, kClusters * kDims * sizeof(double));
        cudaMemset(d_counts, 0, kClusters * sizeof(int));
        cudaMemset(d_inertia, 0, sizeof(double));
        int threads = 256;
        int blocks = (int)std::min<clio::run::u64>(65535, (npoints + threads - 1) / threads);
        KmAssignKernel<<<blocks, threads>>>(m_all, npoints, d_centroids, d_sums, d_counts, d_inertia);
        ctp::GpuApi::Synchronize(); sample_peak();
        cudaMemcpy(h_sums.data(), d_sums, h_sums.size() * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&uprev, d_inertia, sizeof(double), cudaMemcpyDeviceToHost);
        UpdateCentroids(h_sums, h_counts, ucen);
      }
      uvm_km = NowSec() - ut0; uvm_inertia = uprev; uvm_status = "OK"; uvm_peak = peak_mib();
      std::fprintf(stderr, "[CAP] UVM: OK, ran in %.2fs inertia=%.1f peak GPU=%lldMiB "
                   "(managed oversubscription)\n", uvm_km, uvm_inertia, uvm_peak);
      cudaFree(m_all);
    }
    }  // end UVM size gate
    // ---- METHOD 4: STAGED (host->device window streaming, uncompressed) ----
    const clio::run::u64 win_e = (clio::run::u64)window * epp;
    float *d_win = nullptr; float *h_win = nullptr;
    if (cudaMalloc(&d_win, win_e * sizeof(float)) == cudaSuccess &&
        cudaHostAlloc((void**)&h_win, win_e * sizeof(float), cudaHostAllocDefault) == cudaSuccess) {
      g_free_min = SIZE_MAX; sample_peak();
      std::vector<float> scen(init); double sprev = 0.0, st0 = NowSec();
      const clio::run::u64 npages_full = total_elems / win_e;  // full windows
      for (int it = 0; it < iters; ++it) {
        cudaMemcpy(d_centroids, scen.data(), scen.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_sums, 0, kClusters * kDims * sizeof(double));
        cudaMemset(d_counts, 0, kClusters * sizeof(int));
        cudaMemset(d_inertia, 0, sizeof(double));
        for (clio::run::u64 w = 0; w < npages_full; ++w) {
          FillTiled(sift, file_elems, w * win_e, win_e, h_win);
          cudaMemcpy(d_win, h_win, win_e * sizeof(float), cudaMemcpyHostToDevice);
          const clio::run::u64 npw = win_e / kDims;
          int threads = 256; int blocks = (int)((npw + threads - 1) / threads);
          KmAssignKernel<<<blocks, threads>>>(d_win, npw, d_centroids, d_sums, d_counts, d_inertia);
        }
        ctp::GpuApi::Synchronize(); sample_peak();
        cudaMemcpy(h_sums.data(), d_sums, h_sums.size() * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&sprev, d_inertia, sizeof(double), cudaMemcpyDeviceToHost);
        UpdateCentroids(h_sums, h_counts, scen);
      }
      stg_km = NowSec() - st0; stg_inertia = sprev; stg_status = "OK"; stg_peak = peak_mib();
      std::fprintf(stderr, "[CAP] STAGED: OK, ran in %.2fs inertia=%.1f peak GPU=%lldMiB "
                   "(host->device window streaming, uncompressed)\n", stg_km, stg_inertia, stg_peak);
      cudaFree(d_win); cudaFreeHost(h_win);
    }
    // ---- Per-method Track-A CSV (CLIO_KM_TRACKA_CSV) ----
    const char *ta = std::getenv("CLIO_KM_TRACKA_CSV");
    if (ta) {
      const long long dmib = (long long)(dataset_bytes >> 20);
      auto tput = [&](double s){ return s > 0 ? (double)dmib * iters / s : 0.0; };
      auto tpi = [&](double s){ return iters > 0 ? s / iters : 0.0; };
      std::ifstream probe(ta);
      const bool empty = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
      probe.close();
      std::ofstream out(ta, std::ios::app);
      if (out) {
        if (empty) out << "method,size_gib,ran,peak_gpu_mib,throughput_mibps,"
                          "time_per_iter_s,inertia,iters\n";
        const double gib = (double)dmib / 1024.0;
        char b[256];
        std::snprintf(b, sizeof(b), "eternia,%.0f,ran,%lld,%.1f,%.3f,%.1f,%d\n",
                      gib, eternia_peak_mib, tput(km_dt), tpi(km_dt), eternia_inertia, iters);
        out << b;
        std::snprintf(b, sizeof(b), "traditional_incore,%.0f,%s,%lld,%.1f,%.3f,%.1f,%d\n",
                      gib, trad_status == "OK" ? "ran" : "OOM", trad_peak_mib,
                      tput(trad_runtime), tpi(trad_runtime), trad_inertia, iters);
        out << b;  // incore inertia is the lossless baseline; compare Eternia vs this
        std::snprintf(b, sizeof(b), "uvm,%.0f,%s,%lld,%.1f,%.3f,%.1f,%d\n",
                      gib, uvm_status == "OK" ? "ran" : uvm_status.c_str(), uvm_peak,
                      tput(uvm_km), tpi(uvm_km), uvm_inertia, iters);
        out << b;
        std::snprintf(b, sizeof(b), "staged,%.0f,%s,%lld,%.1f,%.3f,%.1f,%d\n",
                      gib, stg_status == "OK" ? "ran" : stg_status.c_str(), stg_peak,
                      tput(stg_km), tpi(stg_km), stg_inertia, iters);
        out << b;
      }
    }
  }

  // ---- Machine-readable CSV row (CLIO_KM_CSV=<path>, appended; header if new) ----
  // One unified schema covers the A1/A2/A3 experiments (all this harness), keyed by
  // dataset size, window, and compress mode. Also echoed to stdout as a CSV, line.
  {
    const char *hbm_cap_env = std::getenv("CLIO_KM_HBM_MIB");
    const char *dram_cap_env = std::getenv("CLIO_KM_DRAM_MIB");
    const long long hbm_cap = hbm_cap_env ? std::atoll(hbm_cap_env) : 2048;
    const long long dram_cap = dram_cap_env ? std::atoll(dram_cap_env) : 12288;
    char row[640];
    std::snprintf(row, sizeof(row),
        "%llu,%llu,%u,%llu,%s,%d,%lld,%lld,%s,%.2f,%lld,%.2f,%.2f,%llu,%.2f,%llu,%llu,%llu,%llu,%lld",
        (unsigned long long)(dataset_bytes >> 20),   // dataset_mib
        (unsigned long long)(page_size >> 10),        // page_kib
        window,                                        // window_pages
        (unsigned long long)((clio::run::u64)2 * window * page_size >> 20),  // window_mib(2x buffered)
        do_compress ? "eternia" : "tier_only",         // mode
        iters,                                         // iters
        hbm_cap,                                        // hbm_cap_mib (logged config)
        dram_cap,                                       // dram_cap_mib (logged config)
        trad_status.c_str(),                           // traditional_status
        trad_runtime,                                  // traditional_runtime_s (-1 if OOM)
        trad_peak_mib,                                  // traditional_peak_gpu_mib (-1 if OOM)
        store_dt,                                       // eternia_store_s
        km_dt,                                          // eternia_km_runtime_s
        (unsigned long long)(stored >> 20),            // eternia_stored_mib
        ratio,                                          // eternia_ratio
        (unsigned long long)(hbm_used >> 20),          // hbm_used_mib
        (unsigned long long)(dram_used >> 20),         // dram_used_mib
        (unsigned long long)(nvme_used >> 20),         // nvme_used_mib
        (unsigned long long)(pfs_used >> 20),          // pfs_used_mib
        eternia_peak_mib);                             // eternia_peak_gpu_mib (whole-process)
    const char *hdr =
        "dataset_mib,page_kib,window_pages,window_mib,mode,iters,hbm_cap_mib,dram_cap_mib,"
        "traditional_status,traditional_runtime_s,traditional_peak_gpu_mib,"
        "eternia_store_s,eternia_km_runtime_s,eternia_stored_mib,"
        "eternia_ratio,hbm_used_mib,dram_used_mib,nvme_used_mib,pfs_used_mib,eternia_peak_gpu_mib";
    std::fprintf(stdout, "[CSV],%s\n[CSV],%s\n", hdr, row);
    std::fflush(stdout);
    const char *csv_path = std::getenv("CLIO_KM_CSV");
    if (csv_path) {
      std::ifstream probe(csv_path);
      const bool empty = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
      probe.close();
      std::ofstream out(csv_path, std::ios::app);
      if (out) { if (empty) out << hdr << "\n"; out << row << "\n"; }
    }
  }

  cudaFree(d_centroids); cudaFree(d_sums); cudaFree(d_counts);
  cudaFree(d_inertia); cudaFree(d_scratch);
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif
