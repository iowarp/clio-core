/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Gray-Scott, Kokkos edition: the performance-portability baseline.
 *
 * WHAT THIS ROW IS FOR. The family already has plain CUDA + MPI, NVSHMEM,
 * BaM and the paged gv::Vector. Kokkos is a different kind of answer to "how
 * does a kernel reach its data": not another transport, but another
 * PROGRAMMING MODEL -- one source over Kokkos::View, backend chosen at build
 * time. It is what a portability-minded application would reach for instead
 * of the vector, so it is what the vector should be measured against.
 *
 * ONE SOURCE, WHICHEVER BACKEND KOKKOS WAS BUILT FOR. Compiled unchanged
 * against a CUDA-backend or SYCL-backend Kokkos; the CMake uses whatever
 * -DKokkos_ROOT points at.
 *
 * SAME DECOMPOSITION, SAME HALO, SAME GATES as clio_grayscott_mpi_bench.cc:
 * a z-slab per rank with a one-plane halo exchanged by host-staged
 * MPI_Sendrecv before every step, so the ONLY difference between those two
 * rows is how the device side is expressed. The science is not copied --
 * InitU/InitV and ReactDiffuse come from ../grayscott_math.h, the
 * dependency-free header the other substrates already share.
 *
 * Like every baseline here it links NOTHING from clio.
 *
 * GATE:
 *   CSUM   sum(v) allreduced; --check-csum V compares against a reference
 *          (the paged bench's value) with a RELATIVE tolerance. The
 *          reduction order differs between substrates -- Kokkos owns its
 *          own -- so bit equality is not expected and the default tolerance
 *          is 1e-3, as in the MPI sibling.
 *
 * Run recipe:
 *   mpirun -n 2 clio_grayscott_kokkos_bench --data-mb 512 --steps 4
 */

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "../grayscott_math.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using u32 = unsigned int;
using u64 = unsigned long long;

using clio_gs::InitU;
using clio_gs::InitV;
using clio_gs::ReactDiffuse;

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

using DevSpace = Kokkos::DefaultExecutionSpace;
using FloatView = Kokkos::View<float *, DevSpace>;
using HostFloat = Kokkos::View<float *, Kokkos::HostSpace>;

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  u32 blocks = 64, threads = 256, steps = 4;
  u64 page_kb = 1024, data_mb = 2048;
  float Du = 0.2f, Dv = 0.1f, F = 0.02f, K = 0.048f, dt = 1.0f;
  double check_csum = 0.0, check_tol = 1e-3;
  bool do_check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--steps") steps = static_cast<u32>(next());
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--data-mb") data_mb = next();
    else if (a == "--check-csum" && i + 1 < argc) {
      check_csum = std::strtod(argv[++i], nullptr);
      do_check = true;
    } else if (a == "--check-tol" && i + 1 < argc) {
      check_tol = std::strtod(argv[++i], nullptr);
    } else if (a == "--help") {
      if (rank == 0) {
        std::printf("usage: mpirun -n N %s [--blocks N] [--threads N] "
                    "[--steps N] [--page-kb N] [--data-mb N] "
                    "[--check-csum V] [--check-tol R]\n", argv[0]);
      }
      MPI_Finalize();
      return 0;
    }
  }
  // Accepted for CLI parity with the siblings, but Kokkos owns the launch
  // geometry -- surrendering that decision IS the programming model.
  (void)blocks;
  (void)threads;

  // Every View must die before Kokkos::finalize(); hence the scope, and rc
  // declared outside it. See the kmeans sibling for the abort this avoids.
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    // SAME GRID DERIVATION as the paged bench: one page is one XY plane,
    // four regions share the dataset budget -- so checksums are comparable.
    const u64 plane = (page_kb * 1024) / sizeof(float);
    u64 nx = 1, ny = plane;
    while (nx * 2 <= ny) { nx *= 2; ny /= 2; }
    const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
    const u64 nz = total_elems / (4 * plane);

    const u64 per = nz / nranks;
    const u64 gz0 = static_cast<u64>(rank) * per;
    const u64 gz1 = (rank == nranks - 1) ? nz : gz0 + per;
    const u64 nzl = gz1 - gz0;
    const u64 ext = (nzl + 2) * plane;   // [halo_lo | own planes | halo_hi]

    if (rank == 0) {
      std::printf("Gray-Scott, Kokkos edition: %llux%llux%llu, %u steps, %d "
                  "ranks  backend=%s\n",
                  (unsigned long long)nx, (unsigned long long)ny,
                  (unsigned long long)nz, steps, nranks, DevSpace::name());
    }

    FloatView u("u", ext), v("v", ext), un("un", ext), vn("vn", ext);

    // Seed both buffers, exactly as the MPI baseline does.
    auto seed = [&](FloatView du, FloatView dv) {
      Kokkos::parallel_for(
          "gs_seed",
          Kokkos::MDRangePolicy<DevSpace, Kokkos::Rank<2>>({1, 0},
                                                           {nzl + 1, plane}),
          KOKKOS_LAMBDA(const u64 lz, const u64 i) {
            const u64 gzz = gz0 + lz - 1;
            du(lz * plane + i) = InitU(i % nx, i / nx, gzz, nx, ny, nz);
            dv(lz * plane + i) = InitV(i % nx, i / nx, gzz, nx, ny, nz);
          });
      Kokkos::fence();
    };
    seed(u, v);
    seed(un, vn);

    const int up = (rank + 1 < nranks) ? rank + 1 : MPI_PROC_NULL;
    const int dn = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    std::vector<float> h_send(plane), h_recv(plane);

    // THE DATA PLANE UNDER TEST: a one-plane halo, host-staged, the same
    // choice the CUDA baseline makes (this MPI reports no device-buffer
    // support, and the bounce is part of what the baseline measures).
    auto exchange = [&](FloatView fld) {
      auto host = Kokkos::create_mirror_view(fld);
      // Send my TOP own plane up, receive my LOW halo from below; then the
      // reverse.
      Kokkos::deep_copy(host, fld);
      for (u64 i = 0; i < plane; ++i) h_send[i] = host(nzl * plane + i);
      MPI_Sendrecv(h_send.data(), static_cast<int>(plane), MPI_FLOAT, up, 11,
                   h_recv.data(), static_cast<int>(plane), MPI_FLOAT, dn, 11,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (dn != MPI_PROC_NULL) {
        for (u64 i = 0; i < plane; ++i) host(i) = h_recv[i];
      }
      for (u64 i = 0; i < plane; ++i) h_send[i] = host(plane + i);
      MPI_Sendrecv(h_send.data(), static_cast<int>(plane), MPI_FLOAT, dn, 12,
                   h_recv.data(), static_cast<int>(plane), MPI_FLOAT, up, 12,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (up != MPI_PROC_NULL) {
        for (u64 i = 0; i < plane; ++i) host((nzl + 1) * plane + i) = h_recv[i];
      }
      Kokkos::deep_copy(fld, host);
    };

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = NowMs();
    for (u32 s = 0; s < steps; ++s) {
      exchange(u);
      exchange(v);
      {
        // `gz0` maps a local plane to its global z, so the fixed global
        // boundary (z == 0, z == nz-1) is honoured no matter how the slab
        // is cut -- the MPI baseline's rule, unchanged.
        FloatView cu = u, cv = v, cun = un, cvn = vn;
        Kokkos::parallel_for(
            "gs_step",
            Kokkos::MDRangePolicy<DevSpace, Kokkos::Rank<2>>({1, 0},
                                                             {nzl + 1, plane}),
            KOKKOS_LAMBDA(const u64 lz, const u64 i) {
              const u64 gzz = gz0 + lz - 1;
              const bool interior = (gzz > 0 && gzz + 1 < nz);
              const u64 z = lz * plane;
              const u64 zm = interior ? z - plane : z;
              const u64 zp = interior ? z + plane : z;
              const u64 x = i % nx, y = i / nx;
              const float uu = cu(z + i);
              const float vv = cv(z + i);
              float lu, lv;
              if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
                  !interior) {
                lu = 0.0f;
                lv = 0.0f;
              } else {
                lu = cu(z + i - 1) + cu(z + i + 1) + cu(z + i - nx) +
                     cu(z + i + nx) + cu(zm + i) + cu(zp + i) - 6.0f * uu;
                lv = cv(z + i - 1) + cv(z + i + 1) + cv(z + i - nx) +
                     cv(z + i + nx) + cv(zm + i) + cv(zp + i) - 6.0f * vv;
              }
              float unew, vnew;
              ReactDiffuse(uu, vv, lu, lv, Du, Dv, F, K, dt, &unew, &vnew);
              cun(z + i) = unew;
              cvn(z + i) = vnew;
            });
        Kokkos::fence();
      }
      auto tu = u; u = un; un = tu;
      auto tv = v; v = vn; vn = tv;
    }
    const double ms = NowMs() - t0;

    double local = 0.0;
    {
      FloatView cv = v;
      Kokkos::parallel_reduce(
          "gs_sum",
          Kokkos::MDRangePolicy<DevSpace, Kokkos::Rank<2>>({1, 0},
                                                           {nzl + 1, plane}),
          KOKKOS_LAMBDA(const u64 lz, const u64 i, double &acc) {
            acc += static_cast<double>(cv(lz * plane + i));
          },
          local);
      Kokkos::fence();
    }
    double csum = 0.0;
    MPI_Allreduce(&local, &csum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
      std::printf("  %u steps in %.1f ms  v_checksum=%.6f\n", steps, ms, csum);
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
      std::printf("%s\n", rc == 0 ? "GRAYSCOTT KOKKOS: ALL GATES PASS"
                                  : "GRAYSCOTT KOKKOS: GATE FAILURE");
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }   // every View and mirror is destroyed here, before finalize
  Kokkos::finalize();
  MPI_Finalize();
  return rc;
}
