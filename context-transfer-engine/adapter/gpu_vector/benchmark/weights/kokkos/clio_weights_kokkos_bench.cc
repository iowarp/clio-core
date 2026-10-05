/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * weights, Kokkos edition: the performance-portability baseline.
 *
 * WHAT THIS ROW IS FOR. Kokkos is not another transport but another
 * PROGRAMMING MODEL -- one source over Kokkos::View, backend chosen at build
 * time. It is what a portability-minded application would reach for instead
 * of the paged vector, so it is what the vector should be measured against.
 *
 * ONE SOURCE, WHICHEVER BACKEND KOKKOS WAS BUILT FOR (-DKokkos_ROOT).
 *
 * SAME SHARDING AND THE SAME GATE as clio_weights_mpi_bench.cc: a contiguous
 * slice of the model per rank, partial sums combined with MPI_Allreduce, so
 * the only difference between those rows is how the device side is
 * expressed. The generator comes from ../weights_math.h -- shared, not
 * copied, which in this benchmark is the experiment's control rather than
 * tidiness: those generators decide the dataset's compression ratio, the
 * independent variable being swept.
 *
 * THE GATE IS EXACT, AND CAN BE. Accumulation is integer, so it commutes:
 * the answer does not depend on how Kokkos orders its reduction, on the rank
 * count, or on the backend. Unlike grayscott's float checksum there is no
 * tolerance here and none is needed -- a mismatch is a real mismatch.
 *
 * Like every baseline here it links NOTHING from clio.
 *
 * Run recipe:
 *   mpirun -n 2 clio_weights_kokkos_bench --pages 64 --flat-pct 25
 */

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "../weights_math.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using u32 = unsigned int;
using u64 = unsigned long long;

using clio_wt::Activation;
using clio_wt::Weight;

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

using DevSpace = Kokkos::DefaultExecutionSpace;
using U32View = Kokkos::View<u32 *, DevSpace>;

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  u32 blocks = 64, threads = 256, passes = 3, flat_pct = 0;
  u64 pages = 64, page_kb = 64, data_mb = 0;   // data_mb overrides pages
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--passes") passes = static_cast<u32>(next());
    else if (a == "--flat-pct") flat_pct = static_cast<u32>(next());
    else if (a == "--pages") pages = next();
    else if (a == "--page-kb") page_kb = next();
    // --data-mb for CLI parity with clio_weights_mpi_bench, which sizes that
    // way; --pages/--page-kb for parity with the paged bench. Matching sizes
    // is what makes the rows comparable at all.
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--help") {
      if (rank == 0) {
        std::printf("usage: mpirun -n N %s [--blocks N] [--threads N] "
                    "[--passes N] [--flat-pct P] [--pages N] [--page-kb N] "
                    "[--data-mb N]\n",
                    argv[0]);
      }
      MPI_Finalize();
      return 0;
    }
  }
  // Accepted for CLI parity; Kokkos owns the launch geometry.
  (void)blocks;
  (void)threads;

  // Every View must die before Kokkos::finalize(); hence the scope.
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const u64 page_elems = (page_kb * 1024) / sizeof(u32);
    const u64 n_total = (data_mb != 0)
                            ? (data_mb * 1024ull * 1024ull) / sizeof(u32)
                            : pages * page_elems;
    // Contiguous shards; the LAST rank absorbs the remainder so every element
    // is owned exactly once.
    const u64 per = n_total / nranks;
    const u64 base = static_cast<u64>(rank) * per;
    const u64 end = (rank == nranks - 1) ? n_total : base + per;
    const u64 n = end - base;

    if (rank == 0) {
      std::printf("weights, Kokkos edition\n"
                  "  elements=%llu (%.1f MB) flat_pct=%u passes=%u ranks=%d "
                  "backend=%s\n",
                  (unsigned long long)n_total,
                  static_cast<double>(n_total * sizeof(u32)) / 1048576.0,
                  flat_pct, passes, nranks, DevSpace::name());
    }

    U32View w("w", n);
    {
      const u32 fp = flat_pct;
      Kokkos::parallel_for(
          "wt_seed", Kokkos::RangePolicy<DevSpace>(0, n),
          KOKKOS_LAMBDA(const u64 i) { w(i) = Weight(base + i, fp); });
      Kokkos::fence();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = NowMs();
    unsigned long long got = 0;
    for (u32 p = 0; p < passes; ++p) {
      unsigned long long local = 0;
      Kokkos::parallel_reduce(
          "wt_sum", Kokkos::RangePolicy<DevSpace>(0, n),
          KOKKOS_LAMBDA(const u64 i, unsigned long long &acc) {
            acc += static_cast<unsigned long long>(w(i)) * Activation(base + i);
          },
          local);
      Kokkos::fence();
      MPI_Allreduce(&local, &got, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                    MPI_COMM_WORLD);
    }
    const double ms = NowMs() - t0;

    // The reference, computed on the host from the same generator: integer,
    // so this is an EXACT expectation, not a tolerance.
    unsigned long long want = 0;
    for (u64 i = 0; i < n_total; ++i) {
      want += static_cast<unsigned long long>(Weight(i, flat_pct)) *
              Activation(i);
    }

    if (rank == 0) {
      std::printf("  %u passes in %.1f ms\n", passes, ms);
      if (got != want) {
        std::printf("  SUM GATE: FAIL (got=%llu want=%llu)\n", got, want);
        rc = 1;
      } else {
        std::printf("  SUM GATE: PASS (exact, %llu)\n", got);
      }
      std::printf("%s\n", rc == 0 ? "WEIGHTS KOKKOS: ALL GATES PASS"
                                  : "WEIGHTS KOKKOS: GATE FAILURE");
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }   // Views destroyed here, before finalize
  Kokkos::finalize();
  MPI_Finalize();
  return rc;
}
