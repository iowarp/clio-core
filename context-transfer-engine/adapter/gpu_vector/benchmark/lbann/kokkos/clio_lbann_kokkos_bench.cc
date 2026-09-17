/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * MLP training, Kokkos edition: the performance-portability baseline.
 *
 * WHAT THIS ROW IS FOR. Kokkos is not another transport but another
 * PROGRAMMING MODEL -- one source over Kokkos::View, backend chosen at build
 * time. It is what a portability-minded application would reach for instead
 * of the paged vector.
 *
 * ONE SOURCE, WHICHEVER BACKEND KOKKOS WAS BUILT FOR (-DKokkos_ROOT).
 *
 * SAME MODEL PARALLELISM as clio_lbann_mpi_bench.cc: W1 sharded by hidden
 * row, W2 by output row, activations and deltas made whole again with
 * host-staged MPI_Allgather, and the d1 partials combined IN RANK ORDER so
 * the float summation order is deterministic. The initialisation comes from
 * ../lbann_math.h, shared with every other substrate, so all of them start
 * from byte-identical weights.
 *
 * Like every baseline here it links NOTHING from clio.
 *
 * GATES:
 *   LOSS     the per-step loss, against a dense single-shard reference run
 *            in-process on rank 0 with the same kernels.
 *   DIGEST   an integer digest of the final weights over their LOGICAL ids.
 *            Integer, so it commutes -- exact regardless of rank count,
 *            reduction order or backend.
 *
 * Run recipe:
 *   mpirun -n 2 clio_lbann_kokkos_bench --hidden 4096 --out 64 --steps 5
 */

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "../lbann_math.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using u32 = unsigned int;
using u64 = unsigned long long;

using clio_lb::Lcg;
using clio_lb::Sym01;

namespace {

double NowMs() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
             clock::now().time_since_epoch()).count();
}

using DevSpace = Kokkos::DefaultExecutionSpace;
using FView = Kokkos::View<float *, DevSpace>;
using DView = Kokkos::View<double *, DevSpace>;

constexpr u64 kSeedW = 0xB5297A4D3F84D5B5ull;
constexpr u64 kSeedX = 0xA02BDBF7BB3C0A7ull;
constexpr u64 kSeedY = 0x6C62272E07BB0142ull;

/** The weight initialiser, by LOGICAL element id -- the same ids every
 *  substrate uses, which is what makes the runs comparable. */
KOKKOS_INLINE_FUNCTION float W0(u64 logical) {
  return Sym01(Lcg(kSeedW + logical)) * 0.05f;
}

/** Copy a host vector into a device View. */
void Upload(FView dst, const std::vector<float> &src) {
  auto h = Kokkos::create_mirror_view(dst);
  for (size_t i = 0; i < src.size(); ++i) h(i) = src[i];
  Kokkos::deep_copy(dst, h);
}

void Download(std::vector<float> &dst, FView src, u64 off, u64 n) {
  auto h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(h, src);
  for (u64 i = 0; i < n; ++i) dst[off + i] = h(off + i);
}

/**
 * One training step over a shard [h0,h1) x [o0,o1).
 *
 * `gather_a1`, `gather_d2`, `gather_lp` and `gather_d1p` make the sharded
 * quantities whole. The dense reference passes no-ops for all four, because
 * a single shard is already whole -- which is how the reference reuses these
 * exact kernels instead of being a second implementation to trust.
 */
struct Net {
  FView w1, b1, w2, b2, a1, d1, d2, d1p, x, y;
  DView lp;
  u64 I, H, O, B, h0, h1, o0, o1, rpp;
  u64 nshard;   // how many partials Bwd1Combine sums (nranks, or 1)
  // THE REFERENCE MUST REASSOCIATE EXACTLY AS THE SHARDED PATH DOES.
  // (blk0+blk1)+(blk2+blk3) is not ((blk0+blk1)+blk2)+blk3 in floats, so a
  // reference that sums its whole o-range in one go does NOT reproduce what
  // N ranks compute -- it drifts by ~1 ulp per step, which the exact LOSS
  // gate then reports as a failure at 2 ranks and not at 1. So in ref_mode
  // the reference computes one partial per RANK-RANGE of `oper` outputs and
  // combines them in rank order, exactly like the distributed path.
  u64 oper;     // outputs per rank-range (ref_mode only)
  int ref_mode; // 1 = compute all nshard partials here, not just one
  float lr;
};

template <typename GatherA1, typename GatherD2, typename GatherLP,
          typename GatherD1P>
void TrainStep(const Net &n, u64 my_shard, GatherA1 ga1, GatherD2 gd2,
               GatherLP glp, GatherD1P gd1p) {
  const u64 I = n.I, H = n.H, O = n.O, B = n.B;
  const u64 h0 = n.h0, h1 = n.h1, o0 = n.o0, o1 = n.o1;
  const float lr = n.lr;
  FView w1 = n.w1, b1 = n.b1, w2 = n.w2, b2 = n.b2;
  FView a1 = n.a1, d1 = n.d1, d2 = n.d2, d1p = n.d1p, x = n.x, y = n.y;
  DView lp = n.lp;

  // ---- fwd1 over this rank's hidden rows ----
  Kokkos::parallel_for(
      "lb_fwd1", Kokkos::RangePolicy<DevSpace>(0, (h1 - h0) * B),
      KOKKOS_LAMBDA(const u64 t) {
        const u64 h = h0 + t / B, b = t % B;
        float acc = b1(h - h0);
        for (u64 i = 0; i < I; ++i) acc += w1((h - h0) * I + i) * x(b * I + i);
        a1(h * B + b) = acc > 0.0f ? acc : 0.0f;
      });
  Kokkos::fence();
  ga1();

  // ---- fwd2 over this rank's output rows ----
  Kokkos::parallel_for(
      "lb_fwd2", Kokkos::RangePolicy<DevSpace>(0, (o1 - o0) * B),
      KOKKOS_LAMBDA(const u64 t) {
        const u64 o = o0 + t / B, b = t % B;
        float acc = b2(o - o0);
        for (u64 h = 0; h < H; ++h) acc += w2((o - o0) * H + h) * a1(h * B + b);
        const float diff = acc - y(b * O + o);
        d2(o * B + b) = 2.0f * diff / static_cast<float>(B * O);
        lp(o * B + b) = static_cast<double>(diff) * static_cast<double>(diff);
      });
  Kokkos::fence();
  gd2();
  glp();

  // ---- bwd1: this rank's o-partial of d1, page-blocked in o exactly like
  //      the paged bwd1 so the float order matches on combine ----
  {
    const u64 rpp = n.rpp;
    // One partial per rank-range. The sharded path has exactly one (its own);
    // the reference walks all of them so the float association matches.
    const u64 nsp = n.ref_mode ? n.nshard : 1;
    for (u64 r = 0; r < nsp; ++r) {
      const u64 ro0 = n.ref_mode ? r * n.oper : o0;
      const u64 ro1r = n.ref_mode ? (r + 1) * n.oper : o1;
      const u64 ro1 = (ro1r < O) ? ro1r : O;
      const u64 wbase = n.ref_mode ? r * n.oper * H : 0;
      const u64 base = (n.ref_mode ? r : my_shard) * H * B;
      Kokkos::parallel_for(
          "lb_bwd1_partial", Kokkos::RangePolicy<DevSpace>(0, H * B),
          KOKKOS_LAMBDA(const u64 t) {
            const u64 h = t / B, b = t % B;
            float acc = 0.0f;
            for (u64 op = ro0; op < ro1; op += rpp) {
              const u64 oend = (op + rpp < ro1) ? op + rpp : ro1;
              float blk = 0.0f;
              for (u64 o = op; o < oend; ++o) {
                blk += w2(wbase + (o - ro0) * H + h) * d2(o * B + b);
              }
              acc += blk;
            }
            d1p(base + t) = acc;
          });
    }
    Kokkos::fence();
  }
  gd1p();

  // ---- combine partials IN RANK ORDER (deterministic), apply relu mask ----
  {
    const u64 nsh = n.nshard;
    Kokkos::parallel_for(
        "lb_bwd1_combine", Kokkos::RangePolicy<DevSpace>(0, H * B),
        KOKKOS_LAMBDA(const u64 t) {
          float acc = 0.0f;
          for (u64 r = 0; r < nsh; ++r) acc += d1p(r * H * B + t);
          d1(t) = (a1(t) <= 0.0f) ? 0.0f : acc;
        });
    Kokkos::fence();
  }

  // ---- SGD on this rank's shards ----
  Kokkos::parallel_for(
      "lb_upd2_w", Kokkos::RangePolicy<DevSpace>(0, (o1 - o0) * H),
      KOKKOS_LAMBDA(const u64 t) {
        const u64 o = o0 + t / H, h = t % H;
        float g = 0.0f;
        for (u64 b = 0; b < B; ++b) g += d2(o * B + b) * a1(h * B + b);
        w2((o - o0) * H + h) -= lr * g;
      });
  Kokkos::parallel_for(
      "lb_upd2_b", Kokkos::RangePolicy<DevSpace>(0, o1 - o0),
      KOKKOS_LAMBDA(const u64 t) {
        const u64 o = o0 + t;
        float g = 0.0f;
        for (u64 b = 0; b < B; ++b) g += d2(o * B + b);
        b2(o - o0) -= lr * g;
      });
  Kokkos::parallel_for(
      "lb_upd1_w", Kokkos::RangePolicy<DevSpace>(0, (h1 - h0) * I),
      KOKKOS_LAMBDA(const u64 t) {
        const u64 h = h0 + t / I, i = t % I;
        float g = 0.0f;
        for (u64 b = 0; b < B; ++b) g += d1(h * B + b) * x(b * I + i);
        w1((h - h0) * I + i) -= lr * g;
      });
  Kokkos::parallel_for(
      "lb_upd1_b", Kokkos::RangePolicy<DevSpace>(0, h1 - h0),
      KOKKOS_LAMBDA(const u64 t) {
        const u64 h = h0 + t;
        float g = 0.0f;
        for (u64 b = 0; b < B; ++b) g += d1(h * B + b);
        b1(h - h0) -= lr * g;
      });
  Kokkos::fence();
}

/** Integer digest of a weight shard over its LOGICAL ids. Bit-cast, not a
 *  numeric sum, so nothing reassociates: exact across rank counts. */
unsigned long long Digest(FView w, u64 gbase, u64 n) {
  unsigned long long acc = 0;
  Kokkos::parallel_reduce(
      "lb_digest", Kokkos::RangePolicy<DevSpace>(0, n),
      KOKKOS_LAMBDA(const u64 i, unsigned long long &a) {
        a += static_cast<unsigned long long>(
                 __builtin_bit_cast(unsigned int, w(i))) *
             (2ull * (gbase + i) + 1ull);
      },
      acc);
  Kokkos::fence();
  return acc;
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  u32 blocks = 8, threads = 256;
  u64 I = 256, H = 4096, O = 64, B = 64, steps = 5, rpp = 16;
  float lr = 0.01f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> u64 {
      return (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 10) : 0;
    };
    if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--in") I = next();
    else if (a == "--hidden") H = next();
    else if (a == "--out") O = next();
    else if (a == "--batch") B = next();
    else if (a == "--steps") steps = next();
    else if (a == "--rpp") rpp = next();
    else if (a == "--lr" && i + 1 < argc) lr = std::strtof(argv[++i], nullptr);
    else if (a == "--help") {
      if (rank == 0) {
        std::printf("usage: mpirun -n N %s [--in N] [--hidden N] [--out N] "
                    "[--batch N] [--steps N] [--rpp N] [--lr F]\n", argv[0]);
      }
      MPI_Finalize();
      return 0;
    }
  }
  (void)blocks;   // Kokkos owns the launch geometry
  (void)threads;

  if (H % nranks != 0 || O % nranks != 0) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "LBANN KOKKOS: H and O must divide the rank count\n");
    }
    MPI_Finalize();
    return 2;
  }

  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    const u64 hper = H / nranks, oper = O / nranks;
    const u64 h0 = rank * hper, h1 = h0 + hper;
    const u64 o0 = rank * oper, o1 = o0 + oper;
    const u64 w1_n = H * I, w2_n = O * H;

    if (rank == 0) {
      std::printf("MLP training, Kokkos edition: %llu -> %llu -> %llu, "
                  "batch=%llu, steps=%llu, %d ranks (model-parallel)  "
                  "backend=%s\n",
                  (unsigned long long)I, (unsigned long long)H,
                  (unsigned long long)O, (unsigned long long)B,
                  (unsigned long long)steps, nranks, DevSpace::name());
    }

    // Batch and targets: deterministic and replicated.
    std::vector<float> hxv(B * I), hyv(B * O);
    for (u64 i = 0; i < B * I; ++i) hxv[i] = Sym01(Lcg(kSeedX + i));
    for (u64 i = 0; i < B * O; ++i) hyv[i] = Sym01(Lcg(kSeedY + i));
    FView d_x("x", B * I), d_y("y", B * O);
    Upload(d_x, hxv);
    Upload(d_y, hyv);

    // Shards, seeded from the GLOBAL logical id so every substrate starts
    // from byte-identical weights whatever the shard layout.
    FView d_w1("w1", hper * I), d_b1("b1", hper);
    FView d_w2("w2", oper * H), d_b2("b2", oper);
    {
      std::vector<float> t(hper * I);
      for (u64 h = h0; h < h1; ++h)
        for (u64 i = 0; i < I; ++i) t[(h - h0) * I + i] = W0(h * I + i);
      Upload(d_w1, t);
      std::vector<float> tb(hper);
      for (u64 h = h0; h < h1; ++h) tb[h - h0] = W0(w1_n + h);
      Upload(d_b1, tb);
      std::vector<float> t2(oper * H);
      for (u64 o = o0; o < o1; ++o)
        for (u64 h = 0; h < H; ++h)
          t2[(o - o0) * H + h] = W0(w1_n + H + o * H + h);
      Upload(d_w2, t2);
      std::vector<float> tb2(oper);
      for (u64 o = o0; o < o1; ++o) tb2[o - o0] = W0(w1_n + H + w2_n + o);
      Upload(d_b2, tb2);
    }

    FView d_a1("a1", H * B), d_d1("d1", H * B), d_d2("d2", O * B);
    FView d_d1p("d1p", static_cast<u64>(nranks) * H * B);
    DView d_lp("lp", O * B);

    std::vector<float> h_gath(H * B), h_gath_o(O * B);
    std::vector<double> h_lp(O * B);
    std::vector<float> h_d1p(static_cast<size_t>(nranks) * H * B);
    std::vector<double> loss(steps);

    Net net{d_w1, d_b1, d_w2, d_b2, d_a1, d_d1, d_d2, d_d1p, d_x, d_y, d_lp,
            I,    H,    O,    B,    h0,   h1,   o0,   o1,    rpp,
            static_cast<u64>(nranks), oper, /*ref_mode=*/0, lr};

    // The four gathers. Host-staged, the same choice the CUDA baseline makes.
    auto ga1 = [&]() {
      Download(h_gath, d_a1, h0 * B, hper * B);
      MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_gath.data(),
                    static_cast<int>(hper * B), MPI_FLOAT, MPI_COMM_WORLD);
      Upload(d_a1, h_gath);
    };
    auto gd2 = [&]() {
      Download(h_gath_o, d_d2, o0 * B, oper * B);
      MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_gath_o.data(),
                    static_cast<int>(oper * B), MPI_FLOAT, MPI_COMM_WORLD);
      Upload(d_d2, h_gath_o);
    };
    auto glp = [&]() {
      auto hm = Kokkos::create_mirror_view(d_lp);
      Kokkos::deep_copy(hm, d_lp);
      for (u64 i = 0; i < oper * B; ++i) h_lp[o0 * B + i] = hm(o0 * B + i);
      MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_lp.data(),
                    static_cast<int>(oper * B), MPI_DOUBLE, MPI_COMM_WORLD);
    };
    auto gd1p = [&]() {
      auto hm = Kokkos::create_mirror_view(d_d1p);
      Kokkos::deep_copy(hm, d_d1p);
      for (u64 i = 0; i < H * B; ++i) {
        h_d1p[static_cast<u64>(rank) * H * B + i] = hm(rank * H * B + i);
      }
      MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_d1p.data(),
                    static_cast<int>(H * B), MPI_FLOAT, MPI_COMM_WORLD);
      auto hd = Kokkos::create_mirror_view(d_d1p);
      for (size_t i = 0; i < h_d1p.size(); ++i) hd(i) = h_d1p[i];
      Kokkos::deep_copy(d_d1p, hd);
    };

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = NowMs();
    for (u64 s = 0; s < steps; ++s) {
      TrainStep(net, static_cast<u64>(rank), ga1, gd2, glp, gd1p);
      double l = 0.0;
      for (u64 i = 0; i < O * B; ++i) l += h_lp[i];
      loss[s] = l / static_cast<double>(B * O);
    }
    const double ms = NowMs() - t0;

    unsigned long long dg_loc =
        Digest(d_w1, h0 * I, hper * I) + Digest(d_b1, w1_n + h0, hper) +
        Digest(d_w2, w1_n + H + o0 * H, oper * H) +
        Digest(d_b2, w1_n + H + w2_n + o0, oper);
    unsigned long long dg = 0;
    MPI_Allreduce(&dg_loc, &dg, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);

    if (rank == 0) {
      // DENSE IN-PROCESS REFERENCE: the same kernels, one shard, no MPI.
      // Reusing TrainStep rather than writing a second implementation is
      // the point -- a reference nobody can check is not a reference.
      FView r_w1("rw1", w1_n), r_b1("rb1", H);
      FView r_w2("rw2", w2_n), r_b2("rb2", O);
      {
        std::vector<float> t(w1_n);
        for (u64 i = 0; i < w1_n; ++i) t[i] = W0(i);
        Upload(r_w1, t);
        std::vector<float> tb(H);
        for (u64 h = 0; h < H; ++h) tb[h] = W0(w1_n + h);
        Upload(r_b1, tb);
        std::vector<float> t2(w2_n);
        for (u64 i = 0; i < w2_n; ++i) t2[i] = W0(w1_n + H + i);
        Upload(r_w2, t2);
        std::vector<float> t3(O);
        for (u64 o = 0; o < O; ++o) t3[o] = W0(w1_n + H + w2_n + o);
        Upload(r_b2, t3);
      }
      FView r_a1("ra1", H * B), r_d1("rd1", H * B), r_d2("rd2", O * B);
      // nranks partials now, not one: the reference mirrors the sharded
      // split so the float association matches.
      FView r_d1p("rd1p", static_cast<u64>(nranks) * H * B);
      DView r_lp("rlp", O * B);
      Net ref{r_w1, r_b1, r_w2, r_b2, r_a1, r_d1, r_d2, r_d1p, d_x, d_y,
              r_lp, I,   H,    O,    B,    0,    H,    0,     O,   rpp,
              static_cast<u64>(nranks), oper, /*ref_mode=*/1, lr};
      std::vector<double> rloss(steps);
      auto noop = []() {};
      std::vector<double> rlp_host(O * B);
      auto ref_lp = [&]() {
        auto hm = Kokkos::create_mirror_view(r_lp);
        Kokkos::deep_copy(hm, r_lp);
        for (u64 i = 0; i < O * B; ++i) rlp_host[i] = hm(i);
      };
      for (u64 s = 0; s < steps; ++s) {
        TrainStep(ref, 0, noop, noop, ref_lp, noop);
        double l = 0.0;
        for (u64 i = 0; i < O * B; ++i) l += rlp_host[i];
        rloss[s] = l / static_cast<double>(B * O);
      }
      const unsigned long long rdg =
          Digest(r_w1, 0, w1_n) + Digest(r_b1, w1_n, H) +
          Digest(r_w2, w1_n + H, w2_n) + Digest(r_b2, w1_n + H + w2_n, O);

      bool loss_ok = true;
      for (u64 s = 0; s < steps; ++s) {
        if (loss[s] != rloss[s]) loss_ok = false;
      }
      std::printf("  %llu steps in %.1f ms  loss %.6f -> %.6f\n",
                  (unsigned long long)steps, ms, loss[0], loss[steps - 1]);
      if (!loss_ok) {
        std::printf("  LOSS GATE: FAIL (sharded vs dense reference differ)\n");
        for (u64 s = 0; s < steps; ++s) {
          std::printf("    step %llu: %.9f vs %.9f\n",
                      (unsigned long long)s, loss[s], rloss[s]);
        }
        rc = 1;
      } else {
        std::printf("  LOSS GATE: PASS (all %llu steps bit-equal)\n",
                    (unsigned long long)steps);
      }
      if (dg != rdg) {
        std::printf("  DIGEST GATE: FAIL (%llu vs %llu)\n", dg, rdg);
        rc = 1;
      } else {
        std::printf("  DIGEST GATE: PASS (exact, %llu)\n", dg);
      }
      std::printf("%s\n", rc == 0 ? "LBANN KOKKOS: ALL GATES PASS"
                                  : "LBANN KOKKOS: GATE FAILURE");
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }   // Views destroyed here, before finalize
  Kokkos::finalize();
  MPI_Finalize();
  return rc;
}
