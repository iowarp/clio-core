/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * k-means, MPI edition: the scale-out baseline for the paged k-means bench.
 *
 * SAME SCIENCE, DIFFERENT DATA PLANE. The point generator, the assignment
 * kernel and the centroid update are copied from the paged bench (this
 * directory's house style: self-contained benchmarks whose gates prove they
 * still agree). What changes is where the points live and how partial sums
 * meet: each rank generates its GLOBAL-INDEX shard of the deterministic
 * point cloud into plain device memory, assigns locally, and the per-cluster
 * sums and counts are combined with MPI_Allreduce. Every rank then computes
 * the identical centroid update, so the model is replicated and only the
 * reductions cross the wire -- classic data-parallel Lloyd.
 *
 * Like its md sibling, this links NOTHING from clio: a baseline that shares
 * a substrate with the thing it is benchmarked against is not a baseline.
 *
 * GATES:
 *   COUNT   sum of cluster counts == total points, every iteration, exact --
 *           a lost or double-counted point cannot hide.
 *   CSUM    the final centroid checksum, printed always; --check-csum V
 *           compares against a reference (the paged bench's value) with the
 *           documented relative tolerance -- atomicAdd + allreduce make float
 *           summation order layout-dependent, so bit equality is not
 *           expected between substrates.
 *
 * Run recipe:
 *   mpirun -n 4 clio_kmeans_mpi_bench --data-mb 256 --iters 4
 */

#include <mpi.h>

#include <cuda_runtime.h>

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
      MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                        \
  } while (0)

/** Deterministic synthetic coordinate -- IDENTICAL to the paged bench, so
 *  every substrate clusters the same data. */
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

/** The paged bench's assignment step, verbatim science. */
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

/** centroid = sum / count, leaving an empty cluster where it was. */
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
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  {
    int ndev = 0;
    KM_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    KM_CUDA_CHECK(cudaSetDevice(ndev > 0 ? rank % ndev : 0));
  }

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
    } else if (a == "--help") {
      if (rank == 0) {
        std::printf("usage: mpirun -n N %s [--blocks N] [--threads N] "
                    "[--dims N] [--clusters N] [--iters N] [--data-mb N] "
                    "[--check-csum V] [--check-tol R]\n", argv[0]);
      }
      MPI_Finalize();
      return 0;
    }
  }

  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  const u64 npts = total_elems / dims;
  // Contiguous point shards; the LAST rank absorbs the remainder so every
  // point is owned exactly once.
  const u64 per = npts / nranks;
  const u64 p0 = static_cast<u64>(rank) * per;
  const u64 p1 = (rank == nranks - 1) ? npts : p0 + per;
  const u64 my_pts = p1 - p0;
  const u64 my_elems = my_pts * dims;

  if (rank == 0) {
    std::printf("k-means, MPI edition\n"
                "  points=%llu dims=%u k=%u iters=%u  ranks=%d "
                "(%.1f MB/rank)\n",
                (unsigned long long)npts, dims, k, iters, nranks,
                static_cast<double>(my_elems * sizeof(float)) / 1048576.0);
  }

  float *d_pts = nullptr;
  KM_CUDA_CHECK(cudaMalloc(&d_pts, my_elems * sizeof(float)));
  SeedKernel<<<256, 256>>>(d_pts, p0 * dims, my_elems, dims, k);
  KM_CUDA_CHECK(cudaDeviceSynchronize());

  // Initial centroids: the first k points -- identical on every rank and to
  // the paged bench, computed straight from the generator.
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

  std::vector<float> h_sums(h_cent.size());
  std::vector<unsigned> h_counts(k);
  int rc = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  for (u32 it = 0; it < iters; ++it) {
    KM_CUDA_CHECK(cudaMemset(d_sums, 0, h_cent.size() * sizeof(float)));
    KM_CUDA_CHECK(cudaMemset(d_counts, 0, k * sizeof(unsigned)));
    AssignKernel<<<blocks, threads>>>(d_pts, my_pts, dims, k, d_cent, d_sums,
                                      d_counts);
    KM_CUDA_CHECK(cudaDeviceSynchronize());
    // THE DATA PLANE UNDER TEST: partial sums cross the wire host-staged
    // (this MPI reports no CUDA support, and the bounce is part of what the
    // baseline measures, as in the md sibling).
    KM_CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sums,
                             h_sums.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
    KM_CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                             k * sizeof(unsigned), cudaMemcpyDeviceToHost));
    MPI_Allreduce(MPI_IN_PLACE, h_sums.data(), static_cast<int>(h_sums.size()),
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_counts.data(), static_cast<int>(k),
                  MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    // COUNT GATE, exact: a lost or doubled point cannot hide in a float.
    u64 sum_counts = 0;
    for (u32 c = 0; c < k; ++c) sum_counts += h_counts[c];
    if (sum_counts != npts) {
      if (rank == 0) {
        std::printf("  COUNT GATE: FAIL at iter %u (%llu != %llu)\n", it,
                    (unsigned long long)sum_counts, (unsigned long long)npts);
      }
      rc = 1;
    }
    KM_CUDA_CHECK(cudaMemcpy(d_sums, h_sums.data(),
                             h_sums.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    KM_CUDA_CHECK(cudaMemcpy(d_counts, h_counts.data(),
                             k * sizeof(unsigned), cudaMemcpyHostToDevice));
    UpdateKernel<<<(k + 63) / 64, 64>>>(d_cent, d_sums, d_counts, dims, k);
    KM_CUDA_CHECK(cudaDeviceSynchronize());
  }
  const double ms = NowMs() - t0;

  KM_CUDA_CHECK(cudaMemcpy(h_cent.data(), d_cent,
                           h_cent.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
  double csum = 0.0;
  for (float f : h_cent) csum += static_cast<double>(f);

  if (rank == 0) {
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
    std::printf("%s\n", rc == 0 ? "KMEANS MPI: ALL GATES PASS"
                                : "KMEANS MPI: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return rc;
}
