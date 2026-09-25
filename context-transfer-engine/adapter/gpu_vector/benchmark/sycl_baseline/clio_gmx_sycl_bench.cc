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
 * PME spread+gather, Aurora baseline editions (MPI / oneCCL / Intel SHMEM).
 *
 * The mesh is cut into z-plane slabs, one rank per slab, atoms replicated
 * (the mesh is the big object). Spread and gather are both decomposed BY
 * PLANE OWNER -- bins z-3..z feed plane z for both -- so NO halo is ever
 * exchanged; only three exact integer reductions cross the wire (charge
 * total, position-weighted mesh checksum, gather energy), on the
 * substrate's allreduce. Same fixed-point science as the paged bench and
 * the CUDA editions, so CONSERVATION and MESH gates are exact at any rank
 * count. Links nothing from clio.
 *
 * --passes N repeats the spread+gather over the same mesh (the mesh is
 * re-zeroed each pass), which is how the evaluation plan reaches a given
 * I/O volume for gmx without changing K.
 *
 * Run recipe: mpiexec -n 4 --ppn 1 clio_gmx_<sub>_bench --k 512 --atoms 2000000
 */

#include "gv_comm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using gvc::u32;
using gvc::u64;

namespace {

/** Q40.24 fixed point; identical constants and generators to the paged
 *  bench so every substrate spreads the same charges. */
constexpr double kFxScale = 16777216.0;

inline u64 Lcg(u64 s) {
  return s * 6364136223846793005ull + 1442695040888963407ull;
}
inline float Frac01(u64 s) {
  return static_cast<float>((s >> 40) & 0xFFFFFF) / 16777216.0f;
}
inline void Spline4(float t, float w[4]) {
  const float t2 = t * t, t3 = t2 * t;
  w[0] = (1.0f - 3.0f * t + 3.0f * t2 - t3) / 6.0f;
  w[1] = (4.0f - 6.0f * t2 + 3.0f * t3) / 6.0f;
  w[2] = (1.0f + 3.0f * t + 3.0f * t2 - 3.0f * t3) / 6.0f;
  w[3] = t3 / 6.0f;
}

/** The atom arrays and bin CSR, on the device. */
struct Atoms {
  const float *ax, *ay, *az;
  const long long *aq;
  const u32 *bin_start;
};

/**
 * Spread the contributions of bins z-3..z onto plane z of `dst` (one plane,
 * dst points at its base), one work-group per plane. The exactly-conserving
 * hierarchical split is the paged bench's, verbatim: the four z-pieces sum
 * to q exactly and the 16 xy-pieces of each z-piece sum to it exactly, so
 * conservation is an integer identity on every substrate.
 */
inline void SpreadPlane(unsigned long long *dst, u64 z, u64 K, Atoms at,
                        u64 lid, u64 nthreads, sycl::nd_item<1> it) {
  for (int db = -3; db <= 0; ++db) {
    const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
    const u32 a0 = at.bin_start[b];
    const u32 a1 = at.bin_start[b + 1];
    for (u32 a = a0 + lid; a < a1; a += nthreads) {
      const float x = at.ax[a], y = at.ay[a], zz = at.az[a];
      const int ix0 = static_cast<int>(sycl::floor(x)) - 1;
      const int iy0 = static_cast<int>(sycl::floor(y)) - 1;
      const int dzw = static_cast<int>((z + K - b) % K);
      float wx[4], wy[4], wz[4];
      Spline4(x - sycl::floor(x), wx);
      Spline4(y - sycl::floor(y), wy);
      Spline4(zz - sycl::floor(zz), wz);
      long long qz4[4];
      {
        long long run = 0;
        for (int j = 0; j < 3; ++j) {
          qz4[j] = static_cast<long long>(
              sycl::rint(static_cast<double>(at.aq[a]) * wz[j]));
          run += qz4[j];
        }
        qz4[3] = at.aq[a] - run;
      }
      const long long qz = qz4[dzw];
      long long xy_run = 0;
      for (int jy = 0; jy < 4; ++jy) {
        const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                         static_cast<int>(K));
        for (int jx = 0; jx < 4; ++jx) {
          const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                          static_cast<int>(K));
          const long long v =
              (jy == 3 && jx == 3)
                  ? qz - xy_run
                  : static_cast<long long>(sycl::rint(
                        static_cast<double>(qz) * wy[jy] * wx[jx]));
          xy_run += v;
          gvc::AtomicAdd(&dst[gy_ * K + gx],
                         static_cast<unsigned long long>(v));
        }
      }
    }
    sycl::group_barrier(it.get_group());
  }
}

/**
 * The gather, decomposed BY PLANE OWNER: the owner of plane z accumulates
 * the z-terms of the atoms in bins z-3..z, and the partials meet in an
 * exact integer reduction. Same per-plane requantisation as the CUDA
 * editions (exact against them, one unit off the paged bench's single
 * rounding of the four-term sum).
 */
inline unsigned long long GatherPlane(const unsigned long long *pl, u64 z,
                                      u64 K, Atoms at, u64 lid,
                                      u64 nthreads, sycl::nd_item<1> it) {
  unsigned long long acc = 0;
  for (int db = -3; db <= 0; ++db) {
    const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
    const u32 a0 = at.bin_start[b];
    const u32 a1 = at.bin_start[b + 1];
    for (u32 a = a0 + lid; a < a1; a += nthreads) {
      const float x = at.ax[a], y = at.ay[a], zz = at.az[a];
      const int ix0 = static_cast<int>(sycl::floor(x)) - 1;
      const int iy0 = static_cast<int>(sycl::floor(y)) - 1;
      const int dzw = static_cast<int>((z + K - b) % K);
      float wx[4], wy[4], wzS[4];
      Spline4(x - sycl::floor(x), wx);
      Spline4(y - sycl::floor(y), wy);
      Spline4(zz - sycl::floor(zz), wzS);
      double pl_sum = 0.0;
      for (int jy = 0; jy < 4; ++jy) {
        const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                         static_cast<int>(K));
        double row = 0.0;
        for (int jx = 0; jx < 4; ++jx) {
          const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                          static_cast<int>(K));
          row += static_cast<double>(
                     static_cast<long long>(pl[gy_ * K + gx])) * wx[jx];
        }
        pl_sum += row * wy[jy];
      }
      acc += static_cast<unsigned long long>(static_cast<long long>(
          sycl::rint(pl_sum * wzS[dzw] *
                     (static_cast<double>(at.aq[a]) / kFxScale))));
    }
    sycl::group_barrier(it.get_group());
  }
  return acc;
}

/** One work-group per plane, grid-stride over the slab's planes. */
void SpreadSlab(sycl::queue &q, u32 blocks, u32 threads,
                unsigned long long *slab, u64 z0, u64 z1, u64 K, u64 plane,
                Atoms at) {
  const size_t g = static_cast<size_t>(blocks) * threads;
  q.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
     const u64 lid = it.get_local_id(0);
     for (u64 z = z0 + it.get_group(0); z < z1; z += blocks) {
       SpreadPlane(slab + (z - z0) * plane, z, K, at, lid, threads, it);
     }
   }).wait();
}

void SumSlab(sycl::queue &q, u32 blocks, u32 threads,
             const unsigned long long *slab, u64 z0, u64 z1, u64 plane,
             unsigned long long *out) {
  const size_t g = static_cast<size_t>(blocks) * threads;
  q.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
     const u64 lid = it.get_local_id(0);
     for (u64 z = z0 + it.get_group(0); z < z1; z += blocks) {
       unsigned long long qq = 0, ck = 0;
       const unsigned long long *pl = slab + (z - z0) * plane;
       for (u64 i = lid; i < plane; i += threads) {
         qq += pl[i];
         ck += pl[i] * (2ull * (z * plane + i) + 1ull);
       }
       gvc::AtomicAdd(&out[0], qq);
       gvc::AtomicAdd(&out[1], ck);
     }
   }).wait();
}

void GatherSlab(sycl::queue &q, u32 blocks, u32 threads,
                const unsigned long long *slab, u64 z0, u64 z1, u64 K,
                u64 plane, Atoms at, unsigned long long *out) {
  const size_t g = static_cast<size_t>(blocks) * threads;
  q.parallel_for(sycl::nd_range<1>(g, threads), [=](sycl::nd_item<1> it) {
     const u64 lid = it.get_local_id(0);
     for (u64 z = z0 + it.get_group(0); z < z1; z += blocks) {
       const unsigned long long e =
           GatherPlane(slab + (z - z0) * plane, z, K, at, lid, threads, it);
       gvc::AtomicAdd(&out[2], e);
     }
   }).wait();
}

}  // namespace

int main(int argc, char **argv) {
  gvc::Comm comm;
  comm.Init(&argc, &argv);
  const int rank = comm.rank, nranks = comm.nranks;
  sycl::queue &q = comm.q;

  u32 blocks = 8, threads = 256, passes = 1;
  u64 K = 128, atoms = 200000;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--k") K = next();
    else if (a == "--atoms") atoms = next();
    else if (a == "--passes") passes = static_cast<u32>(next());
  }
  const u64 plane = K * K;

  // Deterministic atoms, z-binned CSR -- identical to the paged bench, and
  // replicated on every rank: the MESH is the big object, not the atoms.
  std::vector<float> hx(atoms), hy(atoms), hz(atoms);
  std::vector<long long> hq(atoms);
  std::vector<u32> bin_count(K + 1, 0);
  {
    u64 s = 0x9E3779B97F4A7C15ull;
    for (u64 a = 0; a < atoms; ++a) {
      s = Lcg(s); hx[a] = Frac01(s) * static_cast<float>(K);
      s = Lcg(s); hy[a] = Frac01(s) * static_cast<float>(K);
      s = Lcg(s); hz[a] = Frac01(s) * static_cast<float>(K);
      hq[a] = (a & 1) ? -(1ll << 24) : (1ll << 24);
    }
  }
  std::vector<u32> order(atoms);
  {
    std::vector<u32> bin(atoms);
    for (u64 a = 0; a < atoms; ++a) {
      const int iz0 = static_cast<int>(std::floor(hz[a])) - 1;
      bin[a] = static_cast<u32>((iz0 + static_cast<int>(K)) %
                                static_cast<int>(K));
      bin_count[bin[a] + 1]++;
    }
    for (u64 b = 0; b < K; ++b) bin_count[b + 1] += bin_count[b];
    std::vector<u32> cur(bin_count.begin(), bin_count.end() - 1);
    for (u64 a = 0; a < atoms; ++a) order[cur[bin[a]]++] = static_cast<u32>(a);
  }
  {
    std::vector<float> t(atoms);
    for (u64 a = 0; a < atoms; ++a) t[a] = hx[order[a]];
    hx.swap(t);
    for (u64 a = 0; a < atoms; ++a) t[a] = hy[order[a]];
    hy.swap(t);
    for (u64 a = 0; a < atoms; ++a) t[a] = hz[order[a]];
    hz.swap(t);
  }
  {
    std::vector<long long> t(atoms);
    for (u64 a = 0; a < atoms; ++a) t[a] = hq[order[a]];
    hq.swap(t);
  }
  long long q_total = 0;
  for (u64 a = 0; a < atoms; ++a) q_total += hq[a];

  const u64 per = K / nranks;
  const u64 z0 = static_cast<u64>(rank) * per;
  const u64 z1 = (rank == nranks - 1) ? K : z0 + per;

  if (rank == 0) {
    std::printf("PME spread+gather, %s edition (SYCL): mesh=%llu^3, "
                "atoms=%llu, %d ranks, %.1f MB of mesh per rank, %u pass(es)\n",
                gvc::Comm::Name(), (unsigned long long)K,
                (unsigned long long)atoms, nranks,
                (z1 - z0) * plane * 8.0 / 1048576.0, passes);
  }

  float *d_ax = sycl::malloc_device<float>(atoms, q);
  float *d_ay = sycl::malloc_device<float>(atoms, q);
  float *d_az = sycl::malloc_device<float>(atoms, q);
  long long *d_aq = sycl::malloc_device<long long>(atoms, q);
  u32 *d_bs = sycl::malloc_device<u32>(K + 1, q);
  q.memcpy(d_ax, hx.data(), atoms * sizeof(float));
  q.memcpy(d_ay, hy.data(), atoms * sizeof(float));
  q.memcpy(d_az, hz.data(), atoms * sizeof(float));
  q.memcpy(d_aq, hq.data(), atoms * sizeof(long long));
  q.memcpy(d_bs, bin_count.data(), (K + 1) * sizeof(u32));
  q.wait();
  const Atoms at{d_ax, d_ay, d_az, d_aq, d_bs};

  const u64 slab_n = (z1 - z0) * plane;
  unsigned long long *d_slab = sycl::malloc_device<unsigned long long>(slab_n, q);
  unsigned long long *d_out = comm.Alloc<unsigned long long>(3);

  comm.Barrier();
  const double t0 = gvc::NowMs();
  double t_spread = 0.0, t_gather = 0.0, t_comm = 0.0;
  unsigned long long tot[3] = {0, 0, 0};
  for (u32 p = 0; p < passes; ++p) {
    q.memset(d_slab, 0, slab_n * sizeof(unsigned long long));
    q.memset(d_out, 0, 3 * sizeof(unsigned long long));
    q.wait();
    const double a0 = gvc::NowMs();
    SpreadSlab(q, blocks, threads, d_slab, z0, z1, K, plane, at);
    const double a1 = gvc::NowMs();
    SumSlab(q, blocks, threads, d_slab, z0, z1, plane, d_out);
    GatherSlab(q, blocks, threads, d_slab, z0, z1, K, plane, at, d_out);
    const double a2 = gvc::NowMs();
    // THE DATA PLANE: three exact integer partials meet on the substrate.
    comm.AllreduceSum(d_out, 3);
    const double a3 = gvc::NowMs();
    t_spread += a1 - a0; t_gather += a2 - a1; t_comm += a3 - a2;
    q.memcpy(tot, d_out, sizeof(tot)).wait();
  }
  const double ms = gvc::NowMs() - t0;
  // E1 COMM LINE, EVERY rank: wall time inside this substrate's exchanges
  // (it includes waiting on the slowest peer), as a share of the timed run.
  // Same shape as the paged editions' "COMM" line; the harness takes the max.
  std::printf("COMM %s %s: rank %d comm_ms=%.1f of %.1f ms (%.1f%%)\n", "gmx",
              gvc::Comm::Name(), rank, t_comm, ms, ms > 0.0 ? 100.0 * t_comm / ms : 0.0);

  int rc = 0;
  if (rank == 0) {
    std::printf("  spread %.1f ms  gather+sum %.1f ms  comm %.1f ms  "
                "total %.1f ms\n", t_spread, t_gather, t_comm, ms);
    if (tot[0] != static_cast<unsigned long long>(q_total)) {
      std::printf("  CONSERVATION GATE: FAIL (%llu != %llu)\n", tot[0],
                  static_cast<unsigned long long>(q_total));
      rc = 1;
    } else {
      std::printf("  CONSERVATION GATE: PASS (exact)\n");
    }
    std::printf("  mesh_checksum=%llu  gather_energy=%llu\n", tot[1], tot[2]);
    std::printf("GMX %s: k=%llu atoms=%llu ranks=%d passes=%u ms=%.1f "
                "spread_ms=%.1f gather_ms=%.1f comm_ms=%.1f "
                "mesh_checksum=%llu gather_energy=%llu\n",
                gvc::Comm::Name(), (unsigned long long)K,
                (unsigned long long)atoms, nranks, passes, ms, t_spread,
                t_gather, t_comm, tot[1], tot[2]);
    std::printf("GMX %s: %s\n", gvc::Comm::Name(),
                rc == 0 ? "ALL GATES PASS" : "GATE FAILURE");
  }
  rc = comm.Verdict(rc);
  comm.Free(d_out);
  sycl::free(d_slab, q); sycl::free(d_ax, q); sycl::free(d_ay, q);
  sycl::free(d_az, q); sycl::free(d_aq, q); sycl::free(d_bs, q);
  comm.Finalize();
  return rc;
}
