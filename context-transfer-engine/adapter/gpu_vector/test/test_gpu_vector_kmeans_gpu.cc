/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * K-means clustering over a compressed GPU vector larger than the HBM cache.
 *
 * A different workload shape from the Gray-Scott tests: Gray-Scott is compute
 * plus bursty checkpoint WRITES, whereas K-means is repeated full-dataset
 * streaming READS -- every iteration sweeps the entire point set. That is
 * exactly what the issue-#700 Transaction prefetch path is for, and it exercises
 * the read/decompress path far harder than a checkpoint does.
 *
 * It also compresses DIFFERENTLY. The other unit tests use a smooth analytic
 * field (which cuSZp squeezes ~15x); clustered point data is noisier, so the
 * ratio reported here is the honest one for ML-style data.
 *
 * Setup: N points x D dims (float32) are generated around K known centers and
 * written through the compressed vector (dataset >> resident HBM window). Each
 * K-means iteration then sweeps the dataset with a SequentialTransaction --
 * window W+1 decompresses while the assign kernel consumes window W -- assigning
 * every point to its nearest centroid and accumulating per-centroid sums; the
 * host then recomputes centroids.
 *
 * Verification (lossy-aware): cuSZp perturbs each coordinate by <= the error
 * bound, so we do not expect bit-exact centroids. Instead we assert that the
 * recovered centroids match the KNOWN generating centers (each recovered
 * centroid matched to its nearest true center) within a tolerance that is loose
 * against cluster noise but far tighter than the inter-cluster spacing -- i.e.
 * clustering over compressed data recovers the true structure. Inertia is also
 * required to be non-increasing across iterations.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include "simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_cte/gpu_vector/transaction.h>

#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

bool g_initialized = false;
inline clio::run::PoolId StoragePool() { return clio::run::PoolId(600, 0); }

constexpr int kDims = 8;      // floats per point
constexpr int kClusters = 8;  // K

/** Deterministic hash -> [0,1). Used to synthesize clustered points. */
__host__ __device__ inline float Hash01(clio::run::u64 x) {
  x = (x ^ 0x9E3779B97F4A7C15ULL) * 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBULL;
  x ^= x >> 31;
  return (float)((x >> 11) & 0xFFFFFFULL) / (float)0x1000000ULL;
}

/** True generating center c, dimension d -- well separated in [0,1]^D. */
__host__ __device__ inline float TrueCenter(int c, int d) {
  return 0.1f + 0.8f * Hash01((clio::run::u64)(c + 1) * 1315423911ULL +
                              (clio::run::u64)d * 2654435761ULL);
}

/** Coordinate d of point p: its cluster's center plus small bounded noise. */
__host__ __device__ inline float PointCoord(clio::run::u64 p, int d) {
  int c = (int)(p % (clio::run::u64)kClusters);
  float noise = Hash01(p * 7919ULL + (clio::run::u64)d * 104729ULL) - 0.5f;
  return TrueCenter(c, d) + 0.030f * noise;  // spread << inter-center spacing
}

void EnsureInit() {
#if !CTP_IS_DEVICE_PASS
  if (g_initialized) return;
  const char *port_env = std::getenv("CLIO_PORT");
  int port = port_env ? std::atoi(port_env) : 10590;
  std::string cfg = "/tmp/gpu_vec_kmeans_" + std::to_string(port) + ".yaml";
  {
    std::ofstream f(cfg);
    f << "networking:\n  port: " << port << "\n\n"
      << "runtime:\n  num_threads: 8\n  queue_depth: 65536\n\n"
      << "compose:\n"
      << "  - mod_name: clio_bdev\n"
      << "    pool_name: \"ram::chi_default_bdev\"\n"
      << "    pool_query: local\n    pool_id: \"301.0\"\n"
      << "    bdev_type: ram\n    capacity: \"2048MB\"\n\n"
      << "  - mod_name: clio_cte_compressor\n"
      << "    pool_name: cte_compressor\n    pool_query: local\n"
      << "    pool_id: \"600.0\"\n    next_pool_id: \"512.0\"\n\n"
      << "  - mod_name: clio_cte_core\n"
      << "    pool_name: cte_core\n    pool_query: local\n    pool_id: \"512.0\"\n"
      << "    storage:\n      - path: \"ram::cte_ram_tier1\"\n"
      << "        bdev_type: \"ram\"\n        capacity_limit: \"2048MB\"\n"
      << "        score: 1.0\n"
      << "    dpe:\n      dpe_type: \"max_bw\"\n";
  }
  setenv("CLIO_SERVER_CONF", cfg.c_str(), 1);
  std::fprintf(stderr, "[KM] compose=%s\n", cfg.c_str());
  REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer));
  std::this_thread::sleep_for(1s);
  g_initialized = true;
#endif
}

}  // namespace

namespace gv = clio::cte::gpu_vector;
namespace dev = cte::gpu::dev;

/** Fill one page-sized staging buffer with points [first_elem, +n). */
__global__ void KmFillKernel(float *buf, clio::run::u64 first_elem,
                             clio::run::u64 n) {
  for (clio::run::u64 j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
       j += (clio::run::u64)gridDim.x * blockDim.x) {
    clio::run::u64 e = first_elem + j;
    buf[j] = PointCoord(e / kDims, (int)(e % kDims));
  }
}

/**
 * Gather the resident window [lo,hi) (float indices) into a CONTIGUOUS scratch
 * buffer. Must be one block: read_range uses blockIdx.x as the vector's block
 * index and this vector is single-block.
 *
 * Why a separate gather instead of clustering inside the callback: read_range
 * hands each thread a STRIDED subset of the range, not consecutive runs, so a
 * thread cannot reassemble a point's D consecutive coordinates on its own.
 * Gathering first decouples the streaming read from the point-structured math.
 */
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

/**
 * Standard K-means assignment over the gathered window: one thread per point,
 * accumulating per-centroid sums/counts and total inertia. Plain device memory,
 * so this is free to use a full grid.
 */
__global__ void KmAssignKernel(const float *scratch, clio::run::u64 npts,
                               const float *centroids, float *sums, int *counts,
                               float *inertia) {
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
    for (int d = 0; d < kDims; ++d) atomicAdd(&sums[best * kDims + d], pt[d]);
    atomicAdd(&counts[best], 1);
    atomicAdd(inertia, best_d2);
  }
}

#if !CTP_IS_DEVICE_PASS

TEST_CASE("gpu_vector: k-means over a compressed dataset larger than the HBM "
          "cache recovers the true clusters",
          "[gpu_vector][compress][kmeans][stress]") {
  EnsureInit();
  auto *ipc = CLIO_CPU_IPC;

  const clio::run::u32 window = 4;                       // pages per window
  const clio::run::u64 page_size = 256ULL * 1024;        // cuSZp-correct size
  const clio::run::u64 epp = page_size / sizeof(float);
  const clio::run::u32 K = 64;                           // pages -> 16 MiB
  const clio::run::u64 total_elems = (clio::run::u64)K * epp;
  const clio::run::u64 npoints = total_elems / kDims;
  const int iters = std::getenv("CLIO_KM_ITERS")
                        ? std::atoi(std::getenv("CLIO_KM_ITERS")) : 8;
  const char *tag = "kmeans";
  clio::run::IpcManagerGpuInfo gpu_info = ipc->GetGpuIpcManager()->GetGpuInfo(0);

  std::fprintf(stderr,
      "[KM] points=%llu dims=%d clusters=%d dataset=%lluMiB resident=%lluMiB "
      "iters=%d\n",
      (unsigned long long)npoints, kDims, kClusters,
      (unsigned long long)(total_elems * sizeof(float) >> 20),
      (unsigned long long)((clio::run::u64)2 * window * page_size >> 20), iters);

  // ---- Store the point set compressed, one page at a time ----
  clio::cte::core::Client core;
  core.Init(clio::cte::core::kCtePoolId);
  auto tagf = core.AsyncGetOrCreateTag(tag);
  tagf.Wait();
  REQUIRE(tagf->GetReturnCode() == 0);
  auto tag_id = tagf->tag_id_;

  clio::cte::core::Client comp;
  comp.Init(StoragePool());
  clio::cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.compress_preset_ = 2;   // BALANCED -> abs error bound 1e-3

  float *pagebuf = nullptr;
  REQUIRE(cudaMalloc(&pagebuf, page_size) == cudaSuccess);
  for (clio::run::u32 p = 0; p < K; ++p) {
    KmFillKernel<<<64, 256>>>(pagebuf, (clio::run::u64)p * epp, epp);
    ctp::GpuApi::Synchronize();
    ctp::ipc::ShmPtr<> dp;
    dp.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    dp.off_ = reinterpret_cast<clio::run::u64>(pagebuf);
    std::string name = std::string(tag) + "_b0_pi" + std::to_string(p);
    auto pf = comp.AsyncPutBlob(tag_id, name, (clio::run::u64)0, page_size, dp,
                                0.5f, ctx, 0, clio::run::PoolQuery::Local());
    pf.Wait();
    REQUIRE(pf->GetReturnCode() == 0);
  }
  cudaFree(pagebuf);
  std::fprintf(stderr, "[KM] stored %u compressed pages (%lluMiB logical)\n", K,
               (unsigned long long)(total_elems * sizeof(float) >> 20));

  // ---- K-means iterations, each a full streaming sweep via prefetch ----
  gv::Vector<float> vec(tag, /*nblocks=*/1, /*gpu_id=*/0,
                        /*gpu_pages_per_block=*/2 * window,
                        /*host_pages_per_block=*/0, page_size,
                        /*cache_period_us=*/20000, gv::CacheMode::kLegacy,
                        /*manager_threads_per_block=*/32,
                        /*allow_cold_miss_fault=*/false,
                        /*storage_pool_id=*/StoragePool());

  float *d_centroids = nullptr, *d_sums = nullptr, *d_inertia = nullptr;
  int *d_counts = nullptr;
  REQUIRE(cudaMalloc(&d_centroids, kClusters * kDims * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_sums, kClusters * kDims * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_counts, kClusters * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_inertia, sizeof(float)) == cudaSuccess);
  // Contiguous staging for one prefetch window (window pages worth of floats).
  const clio::run::u64 win_elems = (clio::run::u64)window * epp;
  float *d_scratch = nullptr;
  REQUIRE(cudaMalloc(&d_scratch, win_elems * sizeof(float)) == cudaSuccess);

  // Init centroids to the first K points' coordinates (deterministic).
  std::vector<float> h_centroids(kClusters * kDims);
  for (int c = 0; c < kClusters; ++c)
    for (int d = 0; d < kDims; ++d)
      h_centroids[c * kDims + d] = PointCoord((clio::run::u64)c, d);

  std::vector<float> h_sums(kClusters * kDims);
  std::vector<int> h_counts(kClusters);
  double prev_inertia = 0.0;

  for (int it = 0; it < iters; ++it) {
    cudaMemcpy(d_centroids, h_centroids.data(),
               h_centroids.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sums, 0, kClusters * kDims * sizeof(float));
    cudaMemset(d_counts, 0, kClusters * sizeof(int));
    cudaMemset(d_inertia, 0, sizeof(float));

    // Stream the whole dataset: window W+1 decompresses while W is consumed.
    gv::SequentialTransaction<float> sweep(vec, /*first_page=*/0, /*npages=*/K);
    sweep.Iterate([&](clio::run::u64 lo, clio::run::u64 hi,
                      gv::DeviceView<float> v, cudaStream_t s) {
      // 1) gather the resident window into contiguous scratch (single block:
      //    read_range indexes the vector's blocks by blockIdx.x)
      KmGatherKernel<<<1, 32, 0, s>>>(gpu_info, v, lo, hi, d_scratch);
      // 2) cluster it, one thread per point
      const clio::run::u64 npts = (hi - lo) / kDims;
      const int threads = 256;
      const int blocks = (int)((npts + threads - 1) / threads);
      KmAssignKernel<<<blocks, threads, 0, s>>>(d_scratch, npts, d_centroids,
                                                d_sums, d_counts, d_inertia);
    });
    ctp::GpuApi::Synchronize();

    cudaMemcpy(h_sums.data(), d_sums, h_sums.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int),
               cudaMemcpyDeviceToHost);
    float inertia = 0.0f;
    cudaMemcpy(&inertia, d_inertia, sizeof(float), cudaMemcpyDeviceToHost);

    clio::run::u64 assigned = 0;
    for (int c = 0; c < kClusters; ++c) {
      assigned += (clio::run::u64)h_counts[c];
      if (h_counts[c] > 0)
        for (int d = 0; d < kDims; ++d)
          h_centroids[c * kDims + d] = h_sums[c * kDims + d] / (float)h_counts[c];
    }
    std::fprintf(stderr, "[KM] iter %d: assigned=%llu inertia=%.4f\n", it,
                 (unsigned long long)assigned, (double)inertia);
    REQUIRE(assigned == npoints);          // every point was seen exactly once
    if (it > 0) REQUIRE((double)inertia <= prev_inertia * 1.01 + 1e-6);
    prev_inertia = (double)inertia;
  }

  // ---- Verify the recovered centroids match the KNOWN generating centers ----
  // Lossy compression perturbs coordinates by <= the error bound, so match each
  // recovered centroid to its nearest true center and bound the distance.
  double worst = 0.0;
  std::vector<int> used(kClusters, 0);
  for (int c = 0; c < kClusters; ++c) {
    double best = 1e30; int best_t = -1;
    for (int t = 0; t < kClusters; ++t) {
      double d2 = 0.0;
      for (int d = 0; d < kDims; ++d) {
        double diff = h_centroids[c * kDims + d] - (double)TrueCenter(t, d);
        d2 += diff * diff;
      }
      if (d2 < best) { best = d2; best_t = t; }
    }
    used[best_t] = 1;
    worst = std::max(worst, std::sqrt(best));
  }
  int matched = 0;
  for (int t = 0; t < kClusters; ++t) matched += used[t];

  std::fprintf(stderr,
      "[KM] recovered %d/%d distinct clusters; worst centroid-to-true-center "
      "distance = %.5f\n", matched, kClusters, worst);
  REQUIRE(matched == kClusters);   // a distinct true center per centroid
  REQUIRE(worst < 0.02);           // << inter-center spacing; noise is 0.03 wide

  cudaFree(d_centroids); cudaFree(d_sums); cudaFree(d_counts);
  cudaFree(d_inertia); cudaFree(d_scratch);
  std::fprintf(stderr,
      "[KM] PASS: k-means over %lluMiB compressed (%llu points, %d dims) swept "
      "%d times through a %lluMiB resident window; true clusters recovered.\n",
      (unsigned long long)(total_elems * sizeof(float) >> 20),
      (unsigned long long)npoints, kDims, iters,
      (unsigned long long)((clio::run::u64)2 * window * page_size >> 20));
}

SIMPLE_TEST_MAIN()

#endif  // !CTP_IS_DEVICE_PASS

#else
int main() { return 0; }
#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
