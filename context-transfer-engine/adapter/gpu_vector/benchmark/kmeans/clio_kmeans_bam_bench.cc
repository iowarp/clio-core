/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * k-means, BaM edition: the capacity sibling rather than a transport one.
 *
 * Single rank, no MPI. The POINT SET lives in pinned host DRAM behind BaM's
 * GPU page cache and the assignment kernel reads it through bam_ptr -- the
 * same "array bigger than VRAM, faulted in page-granularly from inside the
 * kernel" contract the paged vector offers, provided by the upstream
 * academic system instead. Centroids, sums and counts are tiny and stay in
 * device memory, exactly as in the paged bench.
 *
 * SAME SCIENCE: point generator, assignment and update are copied from the
 * paged bench, and the access order is sequential in point index so a
 * faulted page serves whole warps -- the same courtesy the paged bench's
 * streaming read shows its cache.
 *
 * GATES: COUNT (exact, every iteration) and CSUM (--check-csum V against
 * the paged bench's value, relative tolerance).
 *
 * Run recipe:
 *   clio_kmeans_bam_bench --data-mb 256 --cache-mb 64
 */

#include <cuda_runtime.h>

#include <bam/array.cuh>
#include <bam/page_cache.cuh>
#include <bam/page_cache_host.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using u32 = unsigned int;
using u64 = unsigned long long;

#define KM_CUDA_CHECK(x)                                                     \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);              \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__host__ __device__ inline float PointVal(u64 idx, u32 dims, u32 k) {
  const u64 point = idx / dims;
  const u64 dim = idx % dims;
  const u64 cluster = point % k;
  const float centre = static_cast<float>(cluster) * 8.0f;
  const u64 h = (point * 6364136223846793005ull + dim * 1442695040888963407ull);
  const float jitter =
      static_cast<float>(static_cast<u32>(h >> 40)) * (2.0f / 16777216.0f) -
      1.0f;
  return centre + jitter;
}

/**
 * Assignment through BaM. One thread per point, striding; the point's dims
 * are read through a bam_ptr that re-resolves only when the index leaves the
 * current page, which is the pattern the md sibling measured as the
 * difference between usable and 3x-slow.
 */
__global__ void AssignKernel(bam::ArrayDevice<float> pts, u64 npts, u32 dims,
                             u32 k, const float *cent, float *sums,
                             unsigned *counts) {
  for (u64 p = blockIdx.x * blockDim.x + threadIdx.x; p < npts;
       p += static_cast<u64>(gridDim.x) * blockDim.x) {
    bam::bam_ptr<float> ptr(&pts);
    float best = 3.4e38f;
    u32 bestk = 0;
    // Two passes over the dims (distance per cluster, then the sum update)
    // would double the faults; instead cache the point in registers/local.
    float pv[64];
    const u32 d_eff = dims > 64 ? 64u : dims;
    for (u32 i = 0; i < d_eff; ++i) pv[i] = ptr.at(p * dims + i);
    for (u32 c = 0; c < k; ++c) {
      float d = 0.0f;
      for (u32 i = 0; i < d_eff; ++i) {
        const float x = pv[i] - cent[c * dims + i];
        d += x * x;
      }
      if (d < best) { best = d; bestk = c; }
    }
    for (u32 i = 0; i < d_eff; ++i) {
      atomicAdd(&sums[bestk * dims + i], pv[i]);
    }
    atomicAdd(&counts[bestk], 1u);
  }
}

__global__ void UpdateKernel(float *cent, const float *sums,
                             const unsigned *counts, u32 dims, u32 k) {
  const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  const unsigned n = counts[c];
  if (n == 0) return;
  for (u32 i = 0; i < dims; ++i) {
    cent[c * dims + i] = sums[c * dims + i] / static_cast<float>(n);
  }
}

static double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now()
                                                       .time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  u32 blocks = 64, threads = 256, dims = 32, k = 16, iters = 4;
  u64 data_mb = 256, cache_mb = 64, bam_page_kb = 64;
  double check_csum = 0.0, check_tol = 1e-4;
  bool do_check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--dims") dims = static_cast<u32>(next());
    else if (a == "--clusters") k = static_cast<u32>(next());
    else if (a == "--iters") iters = static_cast<u32>(next());
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--cache-mb") cache_mb = next();
    else if (a == "--bam-page-kb") bam_page_kb = next();
    else if (a == "--check-csum" && i + 1 < argc) {
      check_csum = std::strtod(argv[++i], nullptr);
      do_check = true;
    } else if (a == "--check-tol" && i + 1 < argc) {
      check_tol = std::strtod(argv[++i], nullptr);
    }
  }
  if (dims > 64) {
    std::fprintf(stderr, "KMEANS BAM: --dims capped at 64 (register cache)\n");
    return 2;
  }
  KM_CUDA_CHECK(cudaSetDevice(0));

  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  const u64 npts = total_elems / dims;
  const u64 nelems = npts * dims;
  const u64 page_bytes = bam_page_kb * 1024;

  std::printf("k-means, BaM edition\n"
              "  points=%llu dims=%u k=%u iters=%u  data=%.1f MB behind a "
              "%llu MB / %lluKB-page BaM cache\n",
              (unsigned long long)npts, dims, k, iters,
              static_cast<double>(nelems * sizeof(float)) / 1048576.0,
              (unsigned long long)cache_mb, (unsigned long long)bam_page_kb);

  // BaM cache over pinned host DRAM (the emulated-controller path exists
  // too; host memory is the apples-to-apples sibling of the vector's DRAM
  // tier).
  bam::PageCacheConfig cfg{};
  cfg.page_size = static_cast<size_t>(page_bytes);
  cfg.num_pages = static_cast<size_t>((cache_mb * 1024 * 1024) / page_bytes);
  cfg.num_queues = 1;
  cfg.queue_depth = 1024;
  cfg.backend = bam::BackendType::kHostMemory;
  cfg.nvme_dev = nullptr;
  std::unique_ptr<bam::PageCache> cache(new bam::PageCache(cfg));
  // WHOLE PAGES ONLY: BaM's host_read_page copies full pages with no clamp,
  // so the backing array is rounded up (see the md sibling for the crash
  // this avoids).
  const u64 bam_elems = ((nelems * sizeof(float) + page_bytes - 1) /
                         page_bytes) * (page_bytes / sizeof(float));
  std::unique_ptr<bam::Array<float>> pts(
      new bam::Array<float>(bam_elems, *cache));
  {
    std::vector<float> host(bam_elems, 0.0f);
    for (u64 i = 0; i < nelems; ++i) host[i] = PointVal(i, dims, k);
    pts->load_from_host(host.data(), bam_elems);
  }

  std::vector<float> h_cent(static_cast<size_t>(k) * dims);
  for (u64 i = 0; i < static_cast<u64>(k) * dims; ++i) {
    h_cent[i] = PointVal(i, dims, k);
  }
  float *d_cent = nullptr, *d_sums = nullptr;
  unsigned *d_counts = nullptr;
  KM_CUDA_CHECK(cudaMalloc(&d_cent, h_cent.size() * sizeof(float)));
  KM_CUDA_CHECK(cudaMalloc(&d_sums, h_cent.size() * sizeof(float)));
  KM_CUDA_CHECK(cudaMalloc(&d_counts, k * sizeof(unsigned)));
  KM_CUDA_CHECK(cudaMemcpy(d_cent, h_cent.data(),
                           h_cent.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

  int rc = 0;
  const double t0 = NowMs();
  for (u32 it = 0; it < iters; ++it) {
    KM_CUDA_CHECK(cudaMemset(d_sums, 0, h_cent.size() * sizeof(float)));
    KM_CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(unsigned)));
    AssignKernel<<<blocks, threads>>>(pts->device(), npts, dims, k, d_cent,
                                      d_sums, d_counts);
    KM_CUDA_CHECK(cudaGetLastError());
    KM_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned> h_counts(k);
    KM_CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, k * sizeof(unsigned),
                             cudaMemcpyDeviceToHost));
    u64 sum_counts = 0;
    for (u32 c = 0; c < k; ++c) sum_counts += h_counts[c];
    if (sum_counts != npts) {
      std::printf("  COUNT GATE: FAIL at iter %u (%llu != %llu)\n", it,
                  (unsigned long long)sum_counts, (unsigned long long)npts);
      rc = 1;
    }
    UpdateKernel<<<(k + 63) / 64, 64>>>(d_cent, d_sums, d_counts, dims, k);
    KM_CUDA_CHECK(cudaDeviceSynchronize());
  }
  const double ms = NowMs() - t0;

  KM_CUDA_CHECK(cudaMemcpy(h_cent.data(), d_cent,
                           h_cent.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
  double csum = 0.0;
  for (float f : h_cent) csum += static_cast<double>(f);

  if (rc == 0) std::printf("  COUNT GATE: PASS (all %u iterations)\n", iters);
  std::printf("  %u iters in %.1f ms  centroid_checksum=%.6f\n", iters, ms,
              csum);
  if (do_check) {
    const double rel = std::fabs(csum - check_csum) /
                       (std::fabs(check_csum) > 0 ? std::fabs(check_csum)
                                                  : 1.0);
    if (rel > check_tol) {
      std::printf("  CSUM GATE: FAIL (%.6f vs %.6f, rel %.2e > %.0e)\n", csum,
                  check_csum, rel, check_tol);
      rc = 1;
    } else {
      std::printf("  CSUM GATE: PASS (rel %.2e)\n", rel);
    }
  }
  std::printf("%s\n", rc == 0 ? "KMEANS BAM: ALL GATES PASS"
                              : "KMEANS BAM: GATE FAILURE");
  return rc;
}
