/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * k-means, NVSHMEM edition: the one-sided sibling of the MPI baseline.
 *
 * SAME SCIENCE, DIFFERENT DATA PLANE. Point generator, assignment kernel and
 * centroid update are copied from the paged bench; each PE owns a
 * global-index shard of the deterministic cloud ON THE SYMMETRIC HEAP, and
 * the per-cluster partial sums meet through NVSHMEM's host collectives
 * (nvshmem_float_sum_reduce / nvshmem_uint_sum_reduce) instead of
 * host-staged MPI_Allreduce -- the reduction happens on DEVICE buffers with
 * no host bounce, which is the one variable this pair isolates.
 *
 * Links nothing from clio, like every baseline here.
 *
 * GATES: COUNT (exact, every iteration) and CSUM (--check-csum V against the
 * paged bench's value, relative tolerance for the usual atomicAdd order
 * reasons).
 *
 * Run recipe:
 *   mpirun -n 4 clio_kmeans_nvshmem_bench --data-mb 256 --iters 4   (MPI boot)
 *   clio_kmeans_nvshmem_bench --data-mb 256                (single-PE, no MPI)
 */

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cuda_runtime.h>

#if defined(MD_NVSHMEM_USE_MPI)
#include <mpi.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__global__ void SeedKernel(float *pts, u64 base_idx, u64 n, u32 dims, u32 k) {
  for (u64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<u64>(gridDim.x) * blockDim.x) {
    pts[i] = PointVal(base_idx + i, dims, k);
  }
}

__global__ void AssignKernel(const float *pts, u64 npts, u32 dims, u32 k,
                             const float *cent, float *sums,
                             unsigned *counts) {
  for (u64 p = blockIdx.x * blockDim.x + threadIdx.x; p < npts;
       p += static_cast<u64>(gridDim.x) * blockDim.x) {
    const float *pt = pts + p * dims;
    float best = 3.4e38f;
    u32 bestk = 0;
    for (u32 c = 0; c < k; ++c) {
      float d = 0.0f;
      for (u32 i = 0; i < dims; ++i) {
        const float x = pt[i] - cent[c * dims + i];
        d += x * x;
      }
      if (d < best) { best = d; bestk = c; }
    }
    for (u32 i = 0; i < dims; ++i) {
      atomicAdd(&sums[bestk * dims + i], pt[i]);
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
#if defined(MD_NVSHMEM_USE_MPI)
  MPI_Init(&argc, &argv);
  {
    int wr = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &wr);
    int ndev = 0;
    KM_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    KM_CUDA_CHECK(cudaSetDevice(ndev > 0 ? wr % ndev : 0));
    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &comm;
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0) {
      std::fprintf(stderr, "nvshmem MPI bootstrap failed\n");
      return 1;
    }
  }
#else
  // CUDA CONTEXT BEFORE INIT: without a current device the init reports
  // success but the symmetric heap is never created (see the md sibling).
  KM_CUDA_CHECK(cudaSetDevice(0));
  {
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_get_uniqueid(&uid);
    nvshmemx_set_attr_uniqueid_args(0, 1, &uid, &attr);
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) {
      std::fprintf(stderr, "nvshmem uniqueid bootstrap failed\n");
      return 1;
    }
  }
#endif
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();

  u32 blocks = 64, threads = 256, dims = 32, k = 16, iters = 4;
  u64 data_mb = 256;
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
    else if (a == "--check-csum" && i + 1 < argc) {
      check_csum = std::strtod(argv[++i], nullptr);
      do_check = true;
    } else if (a == "--check-tol" && i + 1 < argc) {
      check_tol = std::strtod(argv[++i], nullptr);
    }
  }

  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  const u64 npts = total_elems / dims;
  const u64 per = npts / npes;
  const u64 p0 = static_cast<u64>(mype) * per;
  const u64 p1 = (mype == npes - 1) ? npts : p0 + per;
  const u64 my_pts = p1 - p0;
  const u64 my_elems = my_pts * dims;

  if (mype == 0) {
    std::printf("k-means, NVSHMEM edition\n"
                "  points=%llu dims=%u k=%u iters=%u  PEs=%d (%.1f MB/PE)\n",
                (unsigned long long)npts, dims, k, iters, npes,
                static_cast<double>(my_elems * sizeof(float)) / 1048576.0);
  }

  // The shard lives on the symmetric heap; the reduction scratch too, which
  // is what lets the collectives run on device buffers with no host bounce.
  float *d_pts = static_cast<float *>(nvshmem_malloc(my_elems *
                                                     sizeof(float)));
  float *d_sums =
      static_cast<float *>(nvshmem_malloc(static_cast<size_t>(k) * dims *
                                          sizeof(float)));
  float *d_sums_out =
      static_cast<float *>(nvshmem_malloc(static_cast<size_t>(k) * dims *
                                          sizeof(float)));
  unsigned *d_counts =
      static_cast<unsigned *>(nvshmem_malloc(k * sizeof(unsigned)));
  unsigned *d_counts_out =
      static_cast<unsigned *>(nvshmem_malloc(k * sizeof(unsigned)));
  if (!d_pts || !d_sums || !d_sums_out || !d_counts || !d_counts_out) {
    std::fprintf(stderr, "nvshmem_malloc failed\n");
    return 1;
  }
  SeedKernel<<<256, 256>>>(d_pts, p0 * dims, my_elems, dims, k);
  KM_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_cent(static_cast<size_t>(k) * dims);
  for (u64 i = 0; i < static_cast<u64>(k) * dims; ++i) {
    h_cent[i] = PointVal(i, dims, k);
  }
  float *d_cent = nullptr;
  KM_CUDA_CHECK(cudaMalloc(&d_cent, h_cent.size() * sizeof(float)));
  KM_CUDA_CHECK(cudaMemcpy(d_cent, h_cent.data(),
                           h_cent.size() * sizeof(float),
                           cudaMemcpyHostToDevice));

  int rc = 0;
  nvshmem_barrier_all();
  const double t0 = NowMs();
  for (u32 it = 0; it < iters; ++it) {
    KM_CUDA_CHECK(cudaMemset(d_sums, 0,
                             static_cast<size_t>(k) * dims * sizeof(float)));
    KM_CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(unsigned)));
    AssignKernel<<<blocks, threads>>>(d_pts, my_pts, dims, k, d_cent, d_sums,
                                      d_counts);
    KM_CUDA_CHECK(cudaDeviceSynchronize());
    // Device-buffer collectives where the runtime allows them. NVSHMEM's
    // host reduction API refuses limited-MPG runs (several PEs sharing one
    // GPU -- exactly what a workstation smoke test is), so under an MPI
    // bootstrap the reduction rides MPI host-staged, the same call the md
    // sibling's Sum() makes. On a real one-PE-per-GPU cluster build without
    // MPI, the nvshmem reduce path below is taken.
#if defined(MD_NVSHMEM_USE_MPI)
    if (npes > 1) {
      std::vector<float> hs(static_cast<size_t>(k) * dims);
      std::vector<unsigned> hc(k);
      KM_CUDA_CHECK(cudaMemcpy(hs.data(), d_sums, hs.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
      KM_CUDA_CHECK(cudaMemcpy(hc.data(), d_counts, k * sizeof(unsigned),
                               cudaMemcpyDeviceToHost));
      MPI_Allreduce(MPI_IN_PLACE, hs.data(), static_cast<int>(hs.size()),
                    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, hc.data(), static_cast<int>(k),
                    MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
      KM_CUDA_CHECK(cudaMemcpy(d_sums_out, hs.data(),
                               hs.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
      KM_CUDA_CHECK(cudaMemcpy(d_counts_out, hc.data(), k * sizeof(unsigned),
                               cudaMemcpyHostToDevice));
    } else {
      KM_CUDA_CHECK(cudaMemcpy(d_sums_out, d_sums,
                               static_cast<size_t>(k) * dims * sizeof(float),
                               cudaMemcpyDeviceToDevice));
      KM_CUDA_CHECK(cudaMemcpy(d_counts_out, d_counts, k * sizeof(unsigned),
                               cudaMemcpyDeviceToDevice));
    }
#else
    nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, d_sums_out, d_sums,
                             static_cast<size_t>(k) * dims);
    nvshmem_uint_sum_reduce(NVSHMEM_TEAM_WORLD, d_counts_out, d_counts, k);
#endif
    nvshmem_barrier_all();
    // COUNT GATE, exact.
    std::vector<unsigned> h_counts(k);
    KM_CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts_out,
                             k * sizeof(unsigned), cudaMemcpyDeviceToHost));
    u64 sum_counts = 0;
    for (u32 c = 0; c < k; ++c) sum_counts += h_counts[c];
    if (sum_counts != npts) {
      if (mype == 0) {
        std::printf("  COUNT GATE: FAIL at iter %u (%llu != %llu)\n", it,
                    (unsigned long long)sum_counts, (unsigned long long)npts);
      }
      rc = 1;
    }
    UpdateKernel<<<(k + 63) / 64, 64>>>(d_cent, d_sums_out, d_counts_out,
                                        dims, k);
    KM_CUDA_CHECK(cudaDeviceSynchronize());
  }
  const double ms = NowMs() - t0;

  KM_CUDA_CHECK(cudaMemcpy(h_cent.data(), d_cent,
                           h_cent.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
  double csum = 0.0;
  for (float f : h_cent) csum += static_cast<double>(f);

  if (mype == 0) {
    if (rc == 0) std::printf("  COUNT GATE: PASS (all %u iterations)\n", iters);
    std::printf("  %u iters in %.1f ms  centroid_checksum=%.6f\n", iters, ms,
                csum);
    if (do_check) {
      const double rel = std::fabs(csum - check_csum) /
                         (std::fabs(check_csum) > 0 ? std::fabs(check_csum)
                                                    : 1.0);
      if (rel > check_tol) {
        std::printf("  CSUM GATE: FAIL (%.6f vs %.6f, rel %.2e > %.0e)\n",
                    csum, check_csum, rel, check_tol);
        rc = 1;
      } else {
        std::printf("  CSUM GATE: PASS (rel %.2e)\n", rel);
      }
    }
    std::printf("%s\n", rc == 0 ? "KMEANS NVSHMEM: ALL GATES PASS"
                                : "KMEANS NVSHMEM: GATE FAILURE");
  }
  nvshmem_free(d_pts);
  nvshmem_free(d_sums);
  nvshmem_free(d_sums_out);
  nvshmem_free(d_counts);
  nvshmem_free(d_counts_out);
  nvshmem_finalize();
#if defined(MD_NVSHMEM_USE_MPI)
  MPI_Finalize();
#endif
  return rc;
}
