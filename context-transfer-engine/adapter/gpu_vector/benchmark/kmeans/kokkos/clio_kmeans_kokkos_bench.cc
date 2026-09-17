/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * k-means, Kokkos edition: the performance-portability baseline.
 *
 * WHAT THIS ROW IS FOR. The family already has plain CUDA + MPI, NVSHMEM,
 * BaM and the paged gv::Vector. Kokkos is the fourth kind of answer to "how
 * do I reach data from a kernel": not a different transport, but a different
 * PROGRAMMING MODEL -- one source, `Kokkos::View` for storage, and a backend
 * chosen at build time. It is the thing a portability-minded application
 * would reach for instead of the vector, so it is the thing the vector should
 * be measured against.
 *
 * ONE SOURCE, TWO BACKENDS. This file is compiled unchanged against a
 * CUDA-backend Kokkos and a SYCL-backend Kokkos; the CMake builds whichever
 * installs it finds. That is Kokkos's whole claim, and exercising it here
 * costs nothing extra.
 *
 * SAME SCIENCE, SAME GATES, SAME OUTPUT SHAPE as clio_kmeans_mpi_bench.cc,
 * and deliberately the same communication too -- host-staged MPI_Allreduce of
 * the partial sums -- so the ONLY difference between those two rows is the
 * device data plane: Kokkos Views and parallel_for versus cudaMalloc and
 * <<<>>>. The science is not copied, it is INCLUDED from ../kmeans_math.h,
 * which is dependency-free precisely so baselines can use it.
 *
 * Like every other baseline here it links NOTHING from clio.
 *
 * GATES:
 *   COUNT   sum of cluster counts == total points, every iteration, exact.
 *   CSUM    final centroid checksum; --check-csum V compares with a RELATIVE
 *           tolerance -- atomic_add and allreduce make float summation order
 *           layout-dependent, so bit equality is not expected across
 *           substrates or even across backends.
 *
 * Run recipe:
 *   mpirun -n 2 clio_kmeans_kokkos_bench --data-mb 256 --iters 4
 */

#include <mpi.h>

#include <Kokkos_Core.hpp>

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

using DevSpace = Kokkos::DefaultExecutionSpace;
using FloatView = Kokkos::View<float *, DevSpace>;
using UintView = Kokkos::View<unsigned *, DevSpace>;

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
  // `blocks`/`threads` are accepted so the CLI matches its siblings exactly,
  // but Kokkos owns the launch geometry -- surrendering that decision IS the
  // programming model, and overriding it would measure something Kokkos does
  // not do in practice.
  (void)blocks;
  (void)threads;

  // EVERY VIEW MUST DIE BEFORE Kokkos::finalize(). The scope below exists
  // for exactly that: Kokkos aborts with
  //   "allocation <name> is being deallocated after Kokkos::finalize"
  // if a View (or a mirror of one) outlives it, which a `return` from inside
  // the scope would cause. rc is declared outside so the value survives.
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
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
      std::printf("k-means, Kokkos edition\n"
                  "  points=%llu dims=%u k=%u iters=%u  ranks=%d "
                  "(%.1f MB/rank)  backend=%s\n",
                  (unsigned long long)npts, dims, k, iters, nranks,
                  static_cast<double>(my_elems * sizeof(float)) / 1048576.0,
                  DevSpace::name());
    }

    FloatView d_pts("pts", my_elems);
    {
      const u64 base_idx = p0 * dims;
      const u32 d = dims, kk = k;
      Kokkos::parallel_for(
          "kmeans_seed", Kokkos::RangePolicy<DevSpace>(0, my_elems),
          KOKKOS_LAMBDA(const u64 i) {
            d_pts(i) = PointVal(base_idx + i, d, kk);
          });
      Kokkos::fence();
    }

    // Initial centroids: the first k points -- identical on every rank and to
    // every other substrate, computed straight from the generator.
    std::vector<float> h_cent(static_cast<size_t>(k) * dims);
    for (u64 i = 0; i < static_cast<u64>(k) * dims; ++i) {
      h_cent[i] = PointVal(i, dims, k);
    }
    FloatView d_cent("cent", h_cent.size());
    FloatView d_sums("sums", h_cent.size());
    UintView d_counts("counts", k);
    {
      auto h = Kokkos::create_mirror_view(d_cent);
      for (size_t i = 0; i < h_cent.size(); ++i) h(i) = h_cent[i];
      Kokkos::deep_copy(d_cent, h);
    }

    std::vector<float> h_sums(h_cent.size());
    std::vector<unsigned> h_counts(k);
    auto m_sums = Kokkos::create_mirror_view(d_sums);
    auto m_counts = Kokkos::create_mirror_view(d_counts);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = NowMs();
    for (u32 it = 0; it < iters; ++it) {
      Kokkos::deep_copy(d_sums, 0.0f);
      Kokkos::deep_copy(d_counts, 0u);
      {
        const u32 d = dims, kk = k;
        Kokkos::parallel_for(
            "kmeans_assign", Kokkos::RangePolicy<DevSpace>(0, my_pts),
            KOKKOS_LAMBDA(const u64 p) {
              // A View is not a pointer, so hand NearestCentroid a shim that
              // indexes it. The arithmetic is then provably the other
              // baselines' -- it is the same function.
              struct PtAt {
                const FloatView &v;
                u64 base;
                KOKKOS_INLINE_FUNCTION float operator[](u32 i) const {
                  return v(base + i);
                }
              };
              struct CentAt {
                const FloatView &v;
                KOKKOS_INLINE_FUNCTION float operator[](u32 i) const {
                  return v(i);
                }
              };
              const PtAt pt{d_pts, p * d};
              const CentAt cent{d_cent};
              const u32 bestk = NearestCentroid(pt, cent, d, kk);
              for (u32 i = 0; i < d; ++i) {
                Kokkos::atomic_add(&d_sums(bestk * d + i), d_pts(p * d + i));
              }
              Kokkos::atomic_add(&d_counts(bestk), 1u);
            });
        Kokkos::fence();
      }
      // THE DATA PLANE UNDER TEST: partial sums cross the wire host-staged,
      // the same choice the CUDA and SYCL baselines make, so the rows differ
      // only in how the device side is expressed.
      Kokkos::deep_copy(m_sums, d_sums);
      Kokkos::deep_copy(m_counts, d_counts);
      for (size_t i = 0; i < h_sums.size(); ++i) h_sums[i] = m_sums(i);
      for (u32 c = 0; c < k; ++c) h_counts[c] = m_counts(c);
      MPI_Allreduce(MPI_IN_PLACE, h_sums.data(),
                    static_cast<int>(h_sums.size()), MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, h_counts.data(), static_cast<int>(k),
                    MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
      // COUNT GATE, exact: a lost or doubled point cannot hide in a float.
      u64 sum_counts = 0;
      for (u32 c = 0; c < k; ++c) sum_counts += h_counts[c];
      if (sum_counts != npts) {
        if (rank == 0) {
          std::printf("  COUNT GATE: FAIL at iter %u (%llu != %llu)\n", it,
                      (unsigned long long)sum_counts,
                      (unsigned long long)npts);
        }
        rc = 1;
      }
      for (size_t i = 0; i < h_sums.size(); ++i) m_sums(i) = h_sums[i];
      for (u32 c = 0; c < k; ++c) m_counts(c) = h_counts[c];
      Kokkos::deep_copy(d_sums, m_sums);
      Kokkos::deep_copy(d_counts, m_counts);
      {
        const u32 d = dims;
        Kokkos::parallel_for(
            "kmeans_update", Kokkos::RangePolicy<DevSpace>(0, k),
            KOKKOS_LAMBDA(const u32 c) {
              const unsigned n = d_counts(c);
              if (n == 0) return;
              for (u32 i = 0; i < d; ++i) {
                d_cent(c * d + i) = d_sums(c * d + i) / static_cast<float>(n);
              }
            });
        Kokkos::fence();
      }
    }
    const double ms = NowMs() - t0;

    {
      auto h = Kokkos::create_mirror_view(d_cent);
      Kokkos::deep_copy(h, d_cent);
      for (size_t i = 0; i < h_cent.size(); ++i) h_cent[i] = h(i);
    }
    double csum = 0.0;
    for (float f : h_cent) csum += static_cast<double>(f);

    if (rank == 0) {
      if (rc == 0) {
        std::printf("  COUNT GATE: PASS (all %u iterations)\n", iters);
      }
      std::printf("  %u iters in %.1f ms  centroid_checksum=%.6f\n", iters, ms,
                  csum);
      if (do_check) {
        const double rel =
            std::fabs(csum - check_csum) /
            (std::fabs(check_csum) > 0 ? std::fabs(check_csum) : 1.0);
        if (rel > check_tol) {
          std::printf("  CSUM GATE: FAIL (%.6f vs %.6f, rel %.2e > %.0e)\n",
                      csum, check_csum, rel, check_tol);
          rc = 1;
        } else {
          std::printf("  CSUM GATE: PASS (rel %.2e)\n", rel);
        }
      }
      std::printf("%s\n", rc == 0 ? "KMEANS KOKKOS: ALL GATES PASS"
                                  : "KMEANS KOKKOS: GATE FAILURE");
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }   // every View and mirror is destroyed here, before finalize
  Kokkos::finalize();
  MPI_Finalize();
  return rc;
}
