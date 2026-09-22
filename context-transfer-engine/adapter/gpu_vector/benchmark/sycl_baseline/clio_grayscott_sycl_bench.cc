/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * Gray-Scott, Aurora baseline editions (MPI / oneCCL / Intel SHMEM).
 *
 * z-slab decomposition with a one-plane halo exchanged through the
 * substrate before every step (gv_comm.h Sendrecv: GPU-aware MPI_Sendrecv,
 * ccl::send/recv, or an ISHMEM put into the neighbour's symmetric halo
 * plane). Stencil, seeding and constants are the paged bench's, verbatim;
 * the fixed global boundary is honoured through the local->global plane
 * map. The checksum is sum(v), allreduced; compare against the paged bench
 * with --check-csum and a loose tolerance. Links nothing from clio.
 *
 * The field layout per rank is [halo_lo | nzl own planes | halo_hi]; under
 * ISHMEM the whole field is on the symmetric heap so a peer's put lands in
 * the halo plane at the same offset on every PE.
 *
 * Run recipe: mpiexec -n 4 --ppn 1 clio_grayscott_<sub>_bench
 *             --page-kb 1024 --data-mb 4096 --steps 4
 */

#include "gv_comm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

using gvc::u32;
using gvc::u64;

namespace {

/** Initial condition, verbatim from the paged bench: v seeded in a centred
 *  cube, u elsewhere, so every substrate starts from the identical field. */
inline float InitU(u64 x, u64 y, u64 z, u64 nx, u64 ny, u64 nz) {
  const bool in = (x > nx / 3 && x < 2 * nx / 3 && y > ny / 3 &&
                   y < 2 * ny / 3 && z > nz / 3 && z < 2 * nz / 3);
  return in ? 0.5f : 1.0f;
}
inline float InitV(u64 x, u64 y, u64 z, u64 nx, u64 ny, u64 nz) {
  const bool in = (x > nx / 3 && x < 2 * nx / 3 && y > ny / 3 &&
                   y < 2 * ny / 3 && z > nz / 3 && z < 2 * nz / 3);
  return in ? 0.25f : 0.0f;
}

/** Grid geometry of one rank's slab. `hi` is the extended-slab plane index
 *  of halo_hi: max over ranks of nzl, plus one, so every rank's buffer has
 *  the same shape (ISHMEM puts land at the sender's offset on the peer, and
 *  symmetric allocations must match in size). */
struct Slab {
  u64 plane, nx, ny, nz, nzl, gz0, hi;
};

/**
 * One Gray-Scott step for local planes [1, nzl+1) of the extended slab.
 * The formula is the paged bench's, verbatim; `gz0` maps a local plane to
 * its global z so the fixed global boundary is honoured however the slab
 * is cut. One work-group per plane (grid-stride), one work-item per cell.
 */
void Step(sycl::queue &q, u32 blocks, u32 threads, const float *u,
          const float *v, float *un, float *vn, Slab s, float Du, float Dv,
          float F, float K, float dt) {
  const size_t g = static_cast<size_t>(blocks) * threads;
  q.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
     const u64 lid = it.get_local_id(0), grp = it.get_group(0);
     for (u64 lz = 1 + grp; lz < s.nzl + 1; lz += blocks) {
       const u64 gzz = s.gz0 + lz - 1;
       const bool interior = (gzz > 0 && gzz + 1 < s.nz);
       const float *uz = u + lz * s.plane;
       const float *vz = v + lz * s.plane;
       // The plane above the last owned one is halo_hi, at its fixed slot.
       const u64 up = (lz == s.nzl) ? s.hi : lz + 1;
       const float *uzm = interior ? uz - s.plane : uz;
       const float *uzp = interior ? u + up * s.plane : uz;
       const float *vzm = interior ? vz - s.plane : vz;
       const float *vzp = interior ? v + up * s.plane : vz;
       float *unx = un + lz * s.plane;
       float *vnx = vn + lz * s.plane;
       for (u64 i = lid; i < s.plane; i += threads) {
         const u64 x = i % s.nx, y = i / s.nx;
         const float uu = uz[i];
         const float vv = vz[i];
         float lu, lv;
         if (x == 0 || x + 1 == s.nx || y == 0 || y + 1 == s.ny ||
             !interior) {
           lu = 0.0f; lv = 0.0f;
         } else {
           lu = uz[i - 1] + uz[i + 1] + uz[i - s.nx] + uz[i + s.nx] +
                uzm[i] + uzp[i] - 6.0f * uu;
           lv = vz[i - 1] + vz[i + 1] + vz[i - s.nx] + vz[i + s.nx] +
                vzm[i] + vzp[i] - 6.0f * vv;
         }
         const float uvv = uu * vv * vv;
         unx[i] = uu + dt * (Du * lu - uvv + F * (1.0f - uu));
         vnx[i] = vv + dt * (Dv * lv + uvv - (F + K) * vv);
       }
     }
   }).wait();
}

/** Seed the own planes of a slab pair from the global initial condition. */
void SeedSlab(sycl::queue &q, float *u, float *v, Slab s) {
  const u64 n = s.nzl * s.plane;
  const size_t g = 1024 * 256;   // grid-stride: n can exceed 32-bit ids
  q.parallel_for(sycl::nd_range<1>(g, 256), [=](sycl::nd_item<1> it) {
     for (u64 e = it.get_global_id(0); e < n; e += g) {
       const u64 lz = 1 + e / s.plane, i = e % s.plane;
       const u64 gzz = s.gz0 + lz - 1;
       u[lz * s.plane + i] = InitU(i % s.nx, i / s.nx, gzz, s.nx, s.ny, s.nz);
       v[lz * s.plane + i] = InitV(i % s.nx, i / s.nx, gzz, s.nx, s.ny, s.nz);
     }
   }).wait();
  // Halo planes start as zero, as cudaMalloc'd memory did not promise but
  // the exchange overwrites them before the first read anyway.
  q.memset(u, 0, s.plane * sizeof(float));
  q.memset(v, 0, s.plane * sizeof(float));
  q.memset(u + s.hi * s.plane, 0, s.plane * sizeof(float));
  q.memset(v + s.hi * s.plane, 0, s.plane * sizeof(float));
  q.wait();
}

/** sum(v) over the own planes, in double. */
double SumV(sycl::queue &q, const float *v, Slab s) {
  double *d_sum = sycl::malloc_device<double>(1, q);
  q.memset(d_sum, 0, sizeof(double)).wait();
  const u64 n = s.nzl * s.plane;
  const size_t g = 64 * 256;
  q.parallel_for(sycl::nd_range<1>(g, 256), [=](sycl::nd_item<1> it) {
     double acc = 0.0;
     for (u64 e = it.get_global_id(0); e < n; e += g) {
       acc += static_cast<double>(v[s.plane + e]);
     }
     gvc::AtomicAdd(d_sum, acc);
   }).wait();
  double h = 0.0;
  q.memcpy(&h, d_sum, sizeof(double)).wait();
  sycl::free(d_sum, q);
  return h;
}

}  // namespace

int main(int argc, char **argv) {
  gvc::Comm comm;
  comm.Init(&argc, &argv);
  const int rank = comm.rank, nranks = comm.nranks;
  sycl::queue &q = comm.q;

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
    }
  }
  // SAME GRID DERIVATION as the paged bench: one page is one XY plane, four
  // regions share the dataset budget -- so checksums are comparable.
  Slab s;
  s.plane = (page_kb * 1024) / sizeof(float);
  s.nx = 1; s.ny = s.plane;
  while (s.nx * 2 <= s.ny) { s.nx *= 2; s.ny /= 2; }
  const u64 total_elems = (data_mb * 1024ull * 1024ull) / sizeof(float);
  s.nz = total_elems / (4 * s.plane);
  const u64 per = s.nz / nranks;
  s.gz0 = static_cast<u64>(rank) * per;
  const u64 gz1 = (rank == nranks - 1) ? s.nz : s.gz0 + per;
  s.nzl = gz1 - s.gz0;
  // The last rank absorbs the remainder, so it has the most planes.
  const u64 nzl_max = s.nz - static_cast<u64>(nranks - 1) * per;
  s.hi = nzl_max + 1;
  const u64 ext = (nzl_max + 2) * s.plane;

  if (rank == 0) {
    std::printf("Gray-Scott, %s edition (SYCL): %llux%llux%llu, %u steps, %d "
                "ranks, %.1f MB/rank\n",
                gvc::Comm::Name(), (unsigned long long)s.nx,
                (unsigned long long)s.ny, (unsigned long long)s.nz, steps,
                nranks, 4.0 * ext * sizeof(float) / 1048576.0);
  }

  float *u = comm.Alloc<float>(ext), *v = comm.Alloc<float>(ext);
  float *un = comm.Alloc<float>(ext), *vn = comm.Alloc<float>(ext);
  SeedSlab(q, u, v, s);
  SeedSlab(q, un, vn, s);

  const int up = (rank + 1 < nranks) ? rank + 1 : -1;
  const int dn = (rank > 0) ? rank - 1 : -1;
  double t_comm = 0.0;
  auto exchange = [&](float *fld) {
    // My TOP own plane goes up into the upper neighbour's LOW halo; my
    // BOTTOM own plane goes down into the lower neighbour's HIGH halo.
    const double c0 = gvc::NowMs();
    comm.Sendrecv(fld + s.nzl * s.plane, up, fld, dn, s.plane);
    comm.Sendrecv(fld + s.plane, dn, fld + s.hi * s.plane, up, s.plane);
    t_comm += gvc::NowMs() - c0;
  };

  comm.Barrier();
  const double t0 = gvc::NowMs();
  for (u32 st = 0; st < steps; ++st) {
    exchange(u);
    exchange(v);
    Step(q, blocks, threads, u, v, un, vn, s, Du, Dv, F, K, dt);
    std::swap(u, un);
    std::swap(v, vn);
  }
  comm.Barrier();
  const double ms = gvc::NowMs() - t0;

  const double local = SumV(q, v, s);
  const double csum = comm.HostSum(local);

  int rc = 0;
  if (rank == 0) {
    std::printf("  %u steps in %.1f ms (comm %.1f ms)  v_checksum=%.6f\n",
                steps, ms, t_comm, csum);
    std::printf("GRAYSCOTT %s: nx=%llu ny=%llu nz=%llu steps=%u ranks=%d "
                "data_mb=%llu page_kb=%llu ms=%.1f comm_ms=%.1f "
                "ms_per_step=%.2f v_checksum=%.6f\n",
                gvc::Comm::Name(), (unsigned long long)s.nx,
                (unsigned long long)s.ny, (unsigned long long)s.nz, steps,
                nranks, (unsigned long long)data_mb,
                (unsigned long long)page_kb, ms, t_comm, ms / steps, csum);
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
    std::printf("GRAYSCOTT %s: %s\n", gvc::Comm::Name(),
                rc == 0 ? "ALL GATES PASS" : "GATE FAILURE");
  }
  rc = comm.Verdict(rc);
  comm.Free(u); comm.Free(v); comm.Free(un); comm.Free(vn);
  comm.Finalize();
  return rc;
}
