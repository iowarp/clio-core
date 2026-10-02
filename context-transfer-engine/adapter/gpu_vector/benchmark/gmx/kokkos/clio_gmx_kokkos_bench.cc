/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * PME spread+gather, Kokkos edition: the performance-portability baseline.
 *
 * THE ONE THAT IS A TRANSLATION, NOT A MIRROR. The other Kokkos substrates
 * here are flat grid-stride loops that map onto RangePolicy line for line.
 * This one is not: SpreadPlane and GatherPlane are BLOCK-COOPERATIVE -- the
 * threads of a block stride a bin's atoms together and hit __syncthreads()
 * between the four bin groups -- so the Kokkos form has to be a TeamPolicy,
 * with the CUDA block becoming a team:
 *
 *     threadIdx.x     ->  team.team_rank()
 *     blockDim.x      ->  team.team_size()
 *     __syncthreads() ->  team.team_barrier()
 *     atomicAdd       ->  Kokkos::atomic_add
 *
 * WHY THE BARRIER IS THERE AT ALL, since it is easy to drop in translation:
 * successive db groups write overlapping xy footprints of the same plane,
 * and the loop relies on every thread finishing group db before any thread
 * starts db+1. Removing it does not corrupt the CONSERVATION gate (the
 * atomics still sum to q) but it does perturb nothing measurable -- which is
 * exactly why it has to be preserved deliberately rather than noticed later.
 *
 * The fixed-point arithmetic is untouched and comes from ../gmx_math.h:
 * FxRound, Spline4, Lcg, Frac01, kFxScale. The exactly-conserving
 * hierarchical split -- four z-pieces summing to q, sixteen xy-pieces
 * summing to each z-piece -- is an integer identity, so CONSERVATION is
 * exact on every substrate and every rank count.
 *
 * SAME SLAB DECOMPOSITION as clio_gmx_mpi_bench.cc: each rank owns planes
 * [z0,z1) and accumulates the z-terms of the atoms in bins z-3..z, so there
 * is no halo and no replica read. Partials meet in an exact integer
 * MPI_Allreduce.
 *
 * Like every baseline here it links NOTHING from clio.
 *
 * GATES:
 *   CONSERVATION  mesh total == input charge, EXACT, at every rank count and
 *                 on every backend. It is an integer identity, so nothing
 *                 about float arithmetic can move it.
 *   MESH/GATHER   reported; --check-mesh / --check-gather compare exactly
 *                 against a reference. Exact only WITHIN a build -- see the
 *                 contraction note below.
 *
 * WHY MESH/GATHER DO NOT MATCH THE nvcc SIBLINGS BIT FOR BIT, and why that is
 * not a translation bug. Against clio_gmx_mpi_bench at K=128/200k atoms this
 * reports mesh_checksum=...188787030 where nvcc reports ...156500648 -- a
 * handful of cells differing by one fixed-point unit. That is FMA
 * CONTRACTION: nvcc defaults to -fmad=true, clang-CUDA contracts differently,
 * and the products inside Spline4 and the qz*wy*wx quantisation land on
 * different sides of a rounding boundary. MEASURED, not assumed: rebuild the
 * baseline with `nvcc -fmad=false` and this bench with `-ffp-contract=off`
 * and the two agree BIT FOR BIT, mesh and gather both. Contraction stays on
 * here because this row is a performance baseline and turning it off would
 * measure a handicapped Kokkos.
 *
 * So: compare mesh/gather against a sibling built by the SAME compiler with
 * the SAME contraction setting; use CONSERVATION and rank-count invariance
 * (the 1-rank value must equal the N-rank value of this same binary) as the
 * cross-build gates.
 *
 * NOTE on the gather, carried over from the MPI sibling: it requantises per
 * z-term where the paged bench rounds the sum of four z-terms once, so the
 * gather value is exact only against siblings using this same per-plane
 * form -- a second, independent reason not to expect equality with the paged
 * row.
 *
 * Run recipe:
 *   mpirun -n 2 clio_gmx_kokkos_bench --k 128 --atoms 200000
 */

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "../gmx_math.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using u32 = unsigned int;
using u64 = unsigned long long;

using clio_gmx::Frac01;
using clio_gmx::FxRound;
using clio_gmx::kFxScale;
using clio_gmx::Lcg;
using clio_gmx::Spline4;

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

using DevSpace = Kokkos::DefaultExecutionSpace;
using Team = Kokkos::TeamPolicy<DevSpace>::member_type;
using FView = Kokkos::View<float *, DevSpace>;
using LLView = Kokkos::View<long long *, DevSpace>;
using U32View = Kokkos::View<u32 *, DevSpace>;
using ULLView = Kokkos::View<unsigned long long *, DevSpace>;

/**
 * Spread the contributions of bins z-3..z onto plane z (dst is the plane's
 * base within the slab). The exactly-conserving hierarchical split is the
 * paged bench's, verbatim.
 */
KOKKOS_INLINE_FUNCTION void SpreadPlane(const Team &team, ULLView slab,
                                        u64 dst, u64 z, u64 K, FView ax,
                                        FView ay, FView az, LLView aq,
                                        U32View bin_start) {
  const u32 tr = static_cast<u32>(team.team_rank());
  const u32 ts = static_cast<u32>(team.team_size());
  for (int db = -3; db <= 0; ++db) {
    const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
    const u32 a0 = bin_start(b);
    const u32 a1 = bin_start(b + 1);
    for (u32 a = a0 + tr; a < a1; a += ts) {
      const float x = ax(a), y = ay(a), zz = az(a);
      const int ix0 = static_cast<int>(Kokkos::floor(x)) - 1;
      const int iy0 = static_cast<int>(Kokkos::floor(y)) - 1;
      const int dzw = static_cast<int>((z + K - b) % K);
      float wx[4], wy[4], wz[4];
      Spline4(x - Kokkos::floor(x), wx);
      Spline4(y - Kokkos::floor(y), wy);
      Spline4(zz - Kokkos::floor(zz), wz);
      long long qz4[4];
      {
        long long run = 0;
        for (int j = 0; j < 3; ++j) {
          qz4[j] = FxRound(static_cast<double>(aq(a)) * wz[j]);
          run += qz4[j];
        }
        qz4[3] = aq(a) - run;   // the split closes exactly, by construction
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
                  : FxRound(static_cast<double>(qz) * wy[jy] * wx[jx]);
          xy_run += v;
          Kokkos::atomic_add(&slab(dst + gy_ * K + gx),
                             static_cast<unsigned long long>(v));
        }
      }
    }
    // Successive db groups write overlapping xy footprints of this plane.
    team.team_barrier();
  }
}

/**
 * The gather, decomposed BY PLANE OWNER: an atom's energy is
 * q * sum_jz wz_jz * (xy interpolation on plane iz0+jz), and each z-term
 * needs only ONE plane -- the same locality the spread has.
 */
KOKKOS_INLINE_FUNCTION unsigned long long GatherPlane(
    const Team &team, ULLView slab, u64 pl, u64 z, u64 K, FView ax, FView ay,
    FView az, LLView aq, U32View bin_start) {
  const u32 tr = static_cast<u32>(team.team_rank());
  const u32 ts = static_cast<u32>(team.team_size());
  unsigned long long acc = 0;
  for (int db = -3; db <= 0; ++db) {
    const u64 b = (z + K + static_cast<u64>(db + static_cast<int>(K))) % K;
    const u32 a0 = bin_start(b);
    const u32 a1 = bin_start(b + 1);
    for (u32 a = a0 + tr; a < a1; a += ts) {
      const float x = ax(a), y = ay(a), zz = az(a);
      const int ix0 = static_cast<int>(Kokkos::floor(x)) - 1;
      const int iy0 = static_cast<int>(Kokkos::floor(y)) - 1;
      const int dzw = static_cast<int>((z + K - b) % K);
      float wx[4], wy[4], wzS[4];
      Spline4(x - Kokkos::floor(x), wx);
      Spline4(y - Kokkos::floor(y), wy);
      Spline4(zz - Kokkos::floor(zz), wzS);
      double pl_sum = 0.0;
      for (int jy = 0; jy < 4; ++jy) {
        const u64 gy_ = static_cast<u64>((iy0 + jy + static_cast<int>(K)) %
                                         static_cast<int>(K));
        double row = 0.0;
        for (int jx = 0; jx < 4; ++jx) {
          const u64 gx = static_cast<u64>((ix0 + jx + static_cast<int>(K)) %
                                          static_cast<int>(K));
          row += static_cast<double>(
                     static_cast<long long>(slab(pl + gy_ * K + gx))) * wx[jx];
        }
        pl_sum += row * wy[jy];
      }
      // Per-z-term requantisation; see the note in the file header.
      acc += static_cast<unsigned long long>(FxRound(
          pl_sum * wzS[dzw] * (static_cast<double>(aq(a)) / kFxScale)));
    }
    team.team_barrier();
  }
  return acc;
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  u32 blocks = 8, threads = 256;
  u64 K = 128, atoms = 200000;
  unsigned long long check_mesh = 0, check_gather = 0;
  bool do_mesh = false, do_gather = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--k") K = next();
    else if (a == "--atoms") atoms = next();
    else if (a == "--check-mesh") { check_mesh = next(); do_mesh = true; }
    else if (a == "--check-gather") { check_gather = next(); do_gather = true; }
    else if (a == "--help") {
      if (rank == 0) {
        std::printf("usage: mpirun -n N %s [--k N] [--atoms N] "
                    "[--threads N] [--check-mesh V] [--check-gather V]\n",
                    argv[0]);
      }
      MPI_Finalize();
      return 0;
    }
  }
  (void)blocks;   // Kokkos sizes the league; team size is `threads`

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const u64 plane = K * K;

    // Deterministic atoms, z-binned CSR -- identical to the paged bench and
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
      std::printf("PME spread+gather, Kokkos edition: mesh=%llu^3, "
                  "atoms=%llu, %d ranks  backend=%s\n",
                  (unsigned long long)K, (unsigned long long)atoms, nranks,
                  DevSpace::name());
    }

    FView d_ax("ax", atoms), d_ay("ay", atoms), d_az("az", atoms);
    LLView d_aq("aq", atoms);
    U32View d_bs("bs", K + 1);
    {
      auto h_ax = Kokkos::create_mirror_view(d_ax);
      auto h_ay = Kokkos::create_mirror_view(d_ay);
      auto h_az = Kokkos::create_mirror_view(d_az);
      auto h_aq = Kokkos::create_mirror_view(d_aq);
      auto h_bs = Kokkos::create_mirror_view(d_bs);
      for (u64 a = 0; a < atoms; ++a) {
        h_ax(a) = hx[a]; h_ay(a) = hy[a]; h_az(a) = hz[a]; h_aq(a) = hq[a];
      }
      for (u64 b = 0; b <= K; ++b) h_bs(b) = bin_count[b];
      Kokkos::deep_copy(d_ax, h_ax);
      Kokkos::deep_copy(d_ay, h_ay);
      Kokkos::deep_copy(d_az, h_az);
      Kokkos::deep_copy(d_aq, h_aq);
      Kokkos::deep_copy(d_bs, h_bs);
    }

    const u64 nzl = z1 - z0;
    ULLView d_slab("slab", nzl * plane);   // Views zero-initialise
    ULLView d_out("out", 3);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = NowMs();

    // ---- spread: one TEAM per plane ----
    {
      ULLView slab = d_slab;
      FView ax = d_ax, ay = d_ay, az = d_az;
      LLView aq = d_aq;
      U32View bs = d_bs;
      Kokkos::parallel_for(
          "gmx_spread",
          Kokkos::TeamPolicy<DevSpace>(static_cast<int>(nzl),
                                       static_cast<int>(threads)),
          KOKKOS_LAMBDA(const Team &team) {
            const u64 lz = static_cast<u64>(team.league_rank());
            SpreadPlane(team, slab, lz * plane, z0 + lz, K, ax, ay, az, aq, bs);
          });
      Kokkos::fence();
    }

    // ---- sum: charge total and the position-weighted mesh checksum ----
    {
      ULLView slab = d_slab;
      unsigned long long q_sum = 0, ck_sum = 0;
      Kokkos::parallel_reduce(
          "gmx_sum",
          Kokkos::MDRangePolicy<DevSpace, Kokkos::Rank<2>>({0, 0},
                                                           {nzl, plane}),
          KOKKOS_LAMBDA(const u64 lz, const u64 i, unsigned long long &q,
                        unsigned long long &ck) {
            const unsigned long long m = slab(lz * plane + i);
            q += m;
            ck += m * (2ull * ((z0 + lz) * plane + i) + 1ull);
          },
          q_sum, ck_sum);
      Kokkos::fence();
      auto h_out = Kokkos::create_mirror_view(d_out);
      h_out(0) = q_sum;
      h_out(1) = ck_sum;
      h_out(2) = 0;
      Kokkos::deep_copy(d_out, h_out);
    }

    // ---- gather: one TEAM per plane, energies summed ----
    {
      ULLView slab = d_slab, out = d_out;
      FView ax = d_ax, ay = d_ay, az = d_az;
      LLView aq = d_aq;
      U32View bs = d_bs;
      unsigned long long e_sum = 0;
      Kokkos::parallel_reduce(
          "gmx_gather",
          Kokkos::TeamPolicy<DevSpace>(static_cast<int>(nzl),
                                       static_cast<int>(threads)),
          KOKKOS_LAMBDA(const Team &team, unsigned long long &acc) {
            const u64 lz = static_cast<u64>(team.league_rank());
            const unsigned long long e = GatherPlane(
                team, slab, lz * plane, z0 + lz, K, ax, ay, az, aq, bs);
            // Every thread of the team returns its own partial; contribute
            // once per thread, not once per team.
            acc += e;
          },
          e_sum);
      Kokkos::fence();
      auto h_out = Kokkos::create_mirror_view(d_out);
      Kokkos::deep_copy(h_out, d_out);
      h_out(2) = e_sum;
      Kokkos::deep_copy(d_out, h_out);
    }
    const double ms = NowMs() - t0;

    unsigned long long loc[3] = {0, 0, 0}, tot[3] = {0, 0, 0};
    {
      auto h_out = Kokkos::create_mirror_view(d_out);
      Kokkos::deep_copy(h_out, d_out);
      loc[0] = h_out(0); loc[1] = h_out(1); loc[2] = h_out(2);
    }
    MPI_Allreduce(loc, tot, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);

    if (rank == 0) {
      std::printf("  spread+gather in %.1f ms\n", ms);
      const unsigned long long want =
          static_cast<unsigned long long>(q_total);
      if (tot[0] != want) {
        std::printf("  CONSERVATION GATE: FAIL (%llu != %llu)\n", tot[0],
                    want);
        rc = 1;
      } else {
        std::printf("  CONSERVATION GATE: PASS (exact)\n");
      }
      std::printf("  mesh_checksum=%llu  gather_energy=%llu\n", tot[1],
                  tot[2]);
      if (do_mesh) {
        if (tot[1] != check_mesh) {
          std::printf("  MESH GATE: FAIL (%llu vs %llu)\n", tot[1],
                      check_mesh);
          rc = 1;
        } else {
          std::printf("  MESH GATE: PASS (exact)\n");
        }
      }
      if (do_gather) {
        if (tot[2] != check_gather) {
          std::printf("  GATHER GATE: FAIL (%llu vs %llu)\n", tot[2],
                      check_gather);
          rc = 1;
        } else {
          std::printf("  GATHER GATE: PASS (exact)\n");
        }
      }
      std::printf("%s\n", rc == 0 ? "GMX KOKKOS: ALL GATES PASS"
                                  : "GMX KOKKOS: GATE FAILURE");
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }   // Views destroyed here, before finalize
  Kokkos::finalize();
  MPI_Finalize();
  return rc;
}
