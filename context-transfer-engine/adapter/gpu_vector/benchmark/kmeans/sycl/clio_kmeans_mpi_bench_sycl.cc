/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * k-means, MPI edition on SYCL: the scale-out baseline for the paged bench,
 * with a SYCL data plane instead of a CUDA one.
 *
 * SAME SCIENCE, SAME GATES, SAME OUTPUT FORMAT as clio_kmeans_mpi_bench.cc --
 * and the science is not copied, it is INCLUDED: both files get PointVal,
 * NearestCentroid and UpdateCentroid from ../kmeans_math.h, so the two
 * baselines cannot drift apart the way four hand-copied versions of the same
 * twenty lines were free to.
 *
 * What is genuinely different, and is the whole reason this file exists:
 *   - kernels are lambdas on a sycl::queue, not __global__ + <<<>>>;
 *   - allocation is sycl::malloc_device, not cudaMalloc;
 *   - device selection is a sycl::device from the GPU list, not cudaSetDevice.
 *
 * Like its CUDA sibling this links NOTHING from clio: a baseline that shares
 * a substrate with the thing it is benchmarked against is not a baseline.
 * That is also why it takes the science from a dependency-free header rather
 * than from the paged benchmark's kernels.
 *
 * GATES (identical to the CUDA baseline):
 *   COUNT   sum of cluster counts == total points, every iteration, exact.
 *   CSUM    final centroid checksum; --check-csum V compares against a
 *           reference with a RELATIVE tolerance -- atomicAdd and allreduce
 *           make float summation order layout-dependent, so bit equality is
 *           not expected between substrates.
 *
 * Run recipe:
 *   mpirun -n 2 clio_kmeans_mpi_bench_sycl --data-mb 256 --iters 4
 */

#include <mpi.h>

#include <sycl/sycl.hpp>

#include "../kmeans_math.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using u32 = unsigned int;
using u64 = unsigned long long;

using clio_km::NearestCentroid;
using clio_km::PointVal;
using clio_km::UpdateCentroid;

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

/** Device-scope atomic add, the SYCL spelling of the baselines' atomicAdd. */
template <typename T>
inline void AtomicAdd(T *p, T v) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
      r(*p);
  r.fetch_add(v);
}

/**
 * A grid-stride range, in CUDA's shape.
 *
 * The CUDA baseline launches <<<blocks, threads>>> and strides by
 * gridDim.x*blockDim.x. Submitting the same geometry here -- rather than a
 * flat range sized to the data -- keeps the two baselines comparable: the
 * same number of work-items does the same number of atomicAdds in the same
 * grouping, which is what makes the checksums line up as closely as they do.
 */
template <typename BodyT>
void GridStride(sycl::queue &q, u32 blocks, u32 threads, BodyT body) {
  const size_t global = static_cast<size_t>(blocks) * threads;
  q.parallel_for(sycl::nd_range<1>{sycl::range<1>(global),
                                   sycl::range<1>(threads)},
                 [=](sycl::nd_item<1> it) {
                   body(static_cast<u64>(it.get_global_id(0)),
                        static_cast<u64>(it.get_global_range(0)));
                 })
      .wait();
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

  // One GPU per rank, round-robin, mirroring cudaSetDevice(rank % ndev).
  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (devices.empty()) {
    if (rank == 0) std::printf("KMEANS MPI SYCL: no GPU devices\n");
    MPI_Finalize();
    return 1;
  }
  sycl::queue q{devices[rank % static_cast<int>(devices.size())]};

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
    std::printf("k-means, MPI edition (SYCL)\n"
                "  points=%llu dims=%u k=%u iters=%u  ranks=%d "
                "(%.1f MB/rank)  device=%s\n",
                (unsigned long long)npts, dims, k, iters, nranks,
                static_cast<double>(my_elems * sizeof(float)) / 1048576.0,
                q.get_device().get_info<sycl::info::device::name>().c_str());
  }

  float *d_pts = sycl::malloc_device<float>(my_elems, q);
  if (d_pts == nullptr) {
    std::printf("KMEANS MPI SYCL: rank %d point allocation failed\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  {
    const u64 base_idx = p0 * dims;
    const u64 n = my_elems;
    GridStride(q, 256, 256, [=](u64 tid, u64 stride) {
      for (u64 i = tid; i < n; i += stride) {
        d_pts[i] = PointVal(base_idx + i, dims, k);
      }
    });
  }

  // Initial centroids: the first k points -- identical on every rank and to
  // the paged bench, computed straight from the generator.
  std::vector<float> h_cent(static_cast<size_t>(k) * dims);
  for (u64 i = 0; i < static_cast<u64>(k) * dims; ++i) {
    h_cent[i] = PointVal(i, dims, k);
  }
  float *d_cent = sycl::malloc_device<float>(h_cent.size(), q);
  float *d_sums = sycl::malloc_device<float>(h_cent.size(), q);
  unsigned *d_counts = sycl::malloc_device<unsigned>(k, q);
  q.memcpy(d_cent, h_cent.data(), h_cent.size() * sizeof(float)).wait();

  std::vector<float> h_sums(h_cent.size());
  std::vector<unsigned> h_counts(k);
  int rc = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = NowMs();
  for (u32 it = 0; it < iters; ++it) {
    q.memset(d_sums, 0, h_cent.size() * sizeof(float)).wait();
    q.memset(d_counts, 0, k * sizeof(unsigned)).wait();
    {
      const u64 n = my_pts;
      const float *pts = d_pts;
      const float *cent = d_cent;
      float *sums = d_sums;
      unsigned *counts = d_counts;
      GridStride(q, blocks, threads, [=](u64 tid, u64 stride) {
        for (u64 p = tid; p < n; p += stride) {
          const float *pt = pts + p * dims;
          const u32 bestk = NearestCentroid(pt, cent, dims, k);
          for (u32 i = 0; i < dims; ++i) {
            AtomicAdd(&sums[bestk * dims + i], pt[i]);
          }
          AtomicAdd(&counts[bestk], 1u);
        }
      });
    }
    // THE DATA PLANE UNDER TEST: partial sums cross the wire host-staged.
    // Same choice as the CUDA baseline -- this MPI reports no device-buffer
    // support, and the bounce is part of what the baseline measures.
    q.memcpy(h_sums.data(), d_sums, h_sums.size() * sizeof(float)).wait();
    q.memcpy(h_counts.data(), d_counts, k * sizeof(unsigned)).wait();
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
    q.memcpy(d_sums, h_sums.data(), h_sums.size() * sizeof(float)).wait();
    q.memcpy(d_counts, h_counts.data(), k * sizeof(unsigned)).wait();
    {
      float *cent = d_cent;
      const float *sums = d_sums;
      const unsigned *counts = d_counts;
      GridStride(q, (k + 63) / 64, 64, [=](u64 tid, u64) {
        const u32 c = static_cast<u32>(tid);
        if (c < k) UpdateCentroid(cent, sums, counts, dims, c);
      });
    }
  }
  const double ms = NowMs() - t0;

  q.memcpy(h_cent.data(), d_cent, h_cent.size() * sizeof(float)).wait();
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
    std::printf("%s\n", rc == 0 ? "KMEANS MPI SYCL: ALL GATES PASS"
                                : "KMEANS MPI SYCL: GATE FAILURE");
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  sycl::free(d_pts, q);
  sycl::free(d_cent, q);
  sycl::free(d_sums, q);
  sycl::free(d_counts, q);
  MPI_Finalize();
  return rc;
}
