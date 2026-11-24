/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Collective latency comparison: Clio PoolQuery::AllToOne vs MPI.
 *
 * Four arms, all run by the same ranks over the same 4-node Docker cluster and
 * the same TCP network, so the numbers differ only in the collective machinery:
 *
 *   mpi_barrier      MPI_Barrier                       -- the reference barrier
 *   mpi_allreduce    MPI_Allreduce(1 x u64, MPI_SUM)   -- the reference reduce
 *   clio_barrier     MOD_NAME BarrierTask, AllToOne    -- our barrier
 *   clio_allreduce   MOD_NAME AllReduceTask, AllToOne  -- our reduce
 *
 * One rank per physical node (see mpi_hostfile), each attached as a client to
 * its OWN local clio daemon (CLIO_WITH_RUNTIME=0). MPI is used for the two MPI
 * arms and, in the clio arms, ONLY to align the start of a measurement phase
 * and to reduce the per-rank statistics at the end -- never inside a timed clio
 * region, so no MPI cost leaks into the clio numbers.
 *
 * Why AllToOne is the right analogue of an allreduce: routed AllToOne, a task
 * parks at the neighborhood leader until a task from EVERY container in the
 * pool has arrived (the pool has one container per node), at which point the
 * batch is folded into a single aggregate via AggregateIn, that aggregate runs
 * once, and its OUT is broadcast 1->N back to every participant. All
 * contribute, all block until the last one has, and all observe the same
 * combined result -- the defining properties of MPI_Allreduce. The barrier arm
 * is the same path with an empty task, so the difference between the two clio
 * arms isolates the cost of the reduction from the cost of the synchronization.
 *
 * The clio_allreduce arm also self-checks: every iteration's contribution is
 * keyed to the iteration number, so a batch that mixed two iterations, dropped
 * a member, or double-counted one would produce a sum that does not match the
 * closed form and is reported as a mismatch. A benchmark of a collective that
 * silently failed to be collective would just be measuring a fast no-op.
 *
 * Reported per arm: mean, p50, p99 and max of per-iteration latency. Each rank
 * computes its own statistics over its own samples; the printed value is the
 * MAX across ranks (the standard way to report a collective -- the operation is
 * not complete until the slowest participant returns).
 *
 * Env: COLL_BENCH_ITERS (default 1000), COLL_BENCH_WARMUP (default 100),
 *      COLL_BENCH_CSV (optional path for a machine-readable dump).
 * Exit code 0 == every arm ran and the allreduce self-check passed.
 */
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

namespace {

using Clock = std::chrono::steady_clock;

/** Pool used by the clio arms. Fixed so every rank names the same pool. */
constexpr clio::run::u32 kBenchPoolMajor = 9100;
/** Collective identity shared by all participants of a given arm. */
constexpr clio::run::u32 kContainerHash = 0;
constexpr clio::run::u64 kBarrierBatchKey = 1;
constexpr clio::run::u64 kAllReduceBatchKey = 2;

int EnvInt(const char *name, int def) {
  const char *e = std::getenv(name);
  if (e == nullptr || *e == '\0') return def;
  int v = std::atoi(e);
  // Accept 0 (a legitimate warmup count); only a malformed/negative value
  // falls back to the default.
  return v >= 0 ? v : def;
}

void Log(int rank, const std::string &msg) {
  std::fprintf(stderr, "[coll-bench rank%d] %s\n", rank, msg.c_str());
  std::fflush(stderr);
}

/** Per-rank latency statistics for one arm, in microseconds. */
struct Stats {
  double mean_us = 0.0;
  double p50_us = 0.0;
  double p99_us = 0.0;
  double max_us = 0.0;
};

/** Summarize a rank's per-iteration samples. Consumes (sorts) the vector. */
Stats Summarize(std::vector<double> &samples_us) {
  Stats s;
  if (samples_us.empty()) return s;
  double total = 0.0;
  for (double v : samples_us) total += v;
  s.mean_us = total / static_cast<double>(samples_us.size());
  std::sort(samples_us.begin(), samples_us.end());
  const size_t n = samples_us.size();
  // Nearest-rank percentiles; n>=1 so both indices are in range.
  s.p50_us = samples_us[(n * 50) / 100 < n ? (n * 50) / 100 : n - 1];
  s.p99_us = samples_us[(n * 99) / 100 < n ? (n * 99) / 100 : n - 1];
  s.max_us = samples_us[n - 1];
  return s;
}

/**
 * Reduce per-rank statistics to rank 0 by MAX. A collective's latency is the
 * slowest participant's latency, so max (not mean) is the honest summary.
 */
Stats ReduceMax(const Stats &local) {
  double in[4] = {local.mean_us, local.p50_us, local.p99_us, local.max_us};
  double out[4] = {0, 0, 0, 0};
  MPI_Reduce(in, out, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  Stats s;
  s.mean_us = out[0];
  s.p50_us = out[1];
  s.p99_us = out[2];
  s.max_us = out[3];
  return s;
}

/**
 * Run one arm: `warmup` untimed iterations, an alignment barrier, then `iters`
 * timed iterations. `op` receives the iteration index and returns false if that
 * iteration failed a correctness check.
 *
 * The alignment barrier is outside the timed region; nothing inside it touches
 * MPI, so an arm measures only the collective under test.
 */
template <typename Op>
Stats RunArm(int warmup, int iters, Op &&op, int *failures) {
  for (int i = 0; i < warmup; ++i) {
    if (!op(-1 - i)) ++(*failures);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    const bool ok = op(i);
    auto t1 = Clock::now();
    if (!ok) ++(*failures);
    samples.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return Summarize(samples);
}

/** Expected allreduce total for iteration `iter` over `size` ranks. */
std::uint64_t ExpectedSum(int iter, int size) {
  // Contribution of rank r at iteration i is (i+1)*1000 + (r+1), so the total
  // encodes BOTH the iteration and the full membership. A batch that mixed
  // iterations or lost a member cannot match this by accident.
  const std::uint64_t base =
      static_cast<std::uint64_t>(iter + 1) * 1000ull * static_cast<std::uint64_t>(size);
  const std::uint64_t members =
      static_cast<std::uint64_t>(size) * static_cast<std::uint64_t>(size + 1) / 2ull;
  return base + members;
}

std::uint64_t Contribution(int iter, int rank) {
  return static_cast<std::uint64_t>(iter + 1) * 1000ull +
         static_cast<std::uint64_t>(rank + 1);
}


/**
 * Absolute steady-clock nanoseconds. Every rank runs in a container on the SAME
 * host, and Docker does not virtualize CLOCK_MONOTONIC (no time namespace), so
 * these values are directly comparable across ranks. VerifyBarrier does not
 * assume that -- it measures the actual spread first and refuses to judge if
 * the clocks turn out not to be comparable.
 */
std::int64_t NowNsAbs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

/** Result of a barrier semantics check. */
struct BarrierCheck {
  bool conclusive = false;
  std::int64_t clock_spread_ns = 0;  /**< disagreement between rank clocks */
  double worst_margin_us = 0.0;      /**< min over ranks of (exit - last_enter) */
  double med_margin_us = 0.0;        /**< median release-propagation time */
  double max_margin_us = 0.0;        /**< slowest participant's release */
  int violations = 0;                /**< rounds where someone left too early */
  /**
   * Every (round, rank) margin. A margin is the RELEASE PROPAGATION time: how
   * long after the last participant arrived that participant got out. The min
   * is the fastest path through the release (typically the participant
   * co-located with the leader, which needs no network hop to be told); the max
   * is the slowest. Reporting only the min flatters the collective.
   */
  std::vector<double> margins_us;
};

/**
 * Does `op` actually behave as a barrier?
 *
 * The defining property is: NO participant may return before the LAST
 * participant has arrived. Return codes cannot show this -- a "barrier" that
 * simply completed each request independently would return 0 to everyone and
 * look perfectly healthy, which is precisely the failure this benchmark has
 * already seen once on the cross-node path.
 *
 * So force the question. Stagger the arrivals by `stagger_ms` (rotating which
 * rank is last each round, so no single rank's behaviour carries the result),
 * timestamp entry and exit on a clock all ranks share, and check every rank's
 * exit against the LATEST entry across ranks. A real barrier gives every rank a
 * non-negative margin; a fake one lets the early ranks out ~stagger_ms before
 * the last one even arrives, which is enormous next to the ~0.3ms the
 * collective itself takes and cannot be mistaken for noise.
 *
 * The same check is run against MPI_Barrier as a control: if the methodology
 * were broken it would fail there too.
 */
template <typename Op>
BarrierCheck VerifyBarrier(int rank, int size, int rounds, int stagger_ms,
                           Op &&op) {
  BarrierCheck res;

  // Are the ranks' clocks comparable at all? Line them up with an MPI barrier
  // and see how far apart their readings are. Anything under a millisecond is
  // far below the stagger and cannot manufacture or hide a violation.
  MPI_Barrier(MPI_COMM_WORLD);
  std::int64_t t_now = NowNsAbs();
  std::vector<std::int64_t> now_all(static_cast<size_t>(size), 0);
  MPI_Allgather(&t_now, 1, MPI_LONG_LONG, now_all.data(), 1, MPI_LONG_LONG,
                MPI_COMM_WORLD);
  res.clock_spread_ns = *std::max_element(now_all.begin(), now_all.end()) -
                        *std::min_element(now_all.begin(), now_all.end());
  if (res.clock_spread_ns > 5000000) {  // 5ms: clocks are not a shared timebase
    return res;
  }
  res.conclusive = true;
  res.worst_margin_us = 1e30;

  std::vector<std::int64_t> enters(static_cast<size_t>(size), 0);
  std::vector<std::int64_t> exits(static_cast<size_t>(size), 0);
  for (int round = 0; round < rounds; ++round) {
    // Rotate the arrival order so every rank takes a turn being last.
    const int slot = (rank + round) % size;
    std::this_thread::sleep_for(std::chrono::milliseconds(slot * stagger_ms));

    std::int64_t t_enter = NowNsAbs();
    op();
    std::int64_t t_exit = NowNsAbs();

    MPI_Allgather(&t_enter, 1, MPI_LONG_LONG, enters.data(), 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    MPI_Allgather(&t_exit, 1, MPI_LONG_LONG, exits.data(), 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD);

    const std::int64_t last_enter =
        *std::max_element(enters.begin(), enters.end());
    for (int r = 0; r < size; ++r) {
      const double margin_us =
          static_cast<double>(exits[static_cast<size_t>(r)] - last_enter) /
          1000.0;
      if (margin_us < res.worst_margin_us) res.worst_margin_us = margin_us;
      res.margins_us.push_back(margin_us);
      // Charge a violation only beyond the measured clock disagreement, so
      // clock noise can never be reported as a broken barrier.
      if (margin_us < -static_cast<double>(res.clock_spread_ns) / 1000.0) {
        ++res.violations;
      }
    }
  }
  if (!res.margins_us.empty()) {
    std::vector<double> sorted = res.margins_us;
    std::sort(sorted.begin(), sorted.end());
    res.med_margin_us = sorted[sorted.size() / 2];
    res.max_margin_us = sorted.back();
  }
  return res;
}

void PrintHeader(int size, int iters, int warmup) {
  std::printf("\n");
  std::printf("=== Collective latency: Clio PoolQuery::AllToOne vs MPI ===\n");
  std::printf("ranks (1 per node): %d   iterations: %d   warmup: %d\n", size,
              iters, warmup);
  std::printf("latency in microseconds, max across ranks\n\n");
  std::printf("%-16s %12s %12s %12s %12s\n", "arm", "mean", "p50", "p99",
              "max");
  std::printf("%-16s %12s %12s %12s %12s\n", "----------------", "-----------",
              "-----------", "-----------", "-----------");
}

void PrintRow(const char *name, const Stats &s) {
  std::printf("%-16s %12.2f %12.2f %12.2f %12.2f\n", name, s.mean_us, s.p50_us,
              s.p99_us, s.max_us);
}

void PrintRatio(const char *label, const Stats &ours, const Stats &theirs) {
  if (theirs.mean_us <= 0.0) {
    std::printf("%s: n/a (reference measured 0)\n", label);
    return;
  }
  std::printf("%s: %.1fx  (%.2f us vs %.2f us mean)\n", label,
              ours.mean_us / theirs.mean_us, ours.mean_us, theirs.mean_us);
}

void WriteCsv(const char *path, int size, int iters, const Stats &mpi_bar,
              const Stats &mpi_ar, const Stats &clio_bar,
              const Stats &clio_ar, const Stats &local_rtt,
              const Stats &remote_rtt) {
  std::FILE *f = std::fopen(path, "w");
  if (f == nullptr) {
    std::fprintf(stderr, "warning: could not open CSV path %s\n", path);
    return;
  }
  std::fprintf(f, "arm,ranks,iters,mean_us,p50_us,p99_us,max_us\n");
  const struct {
    const char *name;
    const Stats *s;
  } rows[] = {{"mpi_barrier", &mpi_bar},
              {"mpi_allreduce", &mpi_ar},
              {"clio_barrier", &clio_bar},
              {"clio_allreduce", &clio_ar},
              {"clio_local_rtt", &local_rtt},
              {"clio_remote_rtt", &remote_rtt}};
  for (const auto &r : rows) {
    std::fprintf(f, "%s,%d,%d,%.3f,%.3f,%.3f,%.3f\n", r.name, size, iters,
                 r.s->mean_us, r.s->p50_us, r.s->p99_us, r.s->max_us);
  }
  std::fclose(f);
  std::printf("\nCSV written to %s\n", path);
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int iters = EnvInt("COLL_BENCH_ITERS", 1000);
  const int warmup = EnvInt("COLL_BENCH_WARMUP", 100);
  int failures = 0;
  // Per-PHASE progress (never per-iteration): with four ranks blocking on each
  // other, a hang is only diagnosable if each rank says where it stopped.
  Log(rank, "start: " + std::to_string(size) + " ranks, iters=" +
                std::to_string(iters));

  // ---- MPI arms ------------------------------------------------------------
  // Run first: they need no runtime, so if the clio side cannot come up we
  // still have the reference numbers in the log.
  Stats mpi_barrier = RunArm(warmup, iters, [](int) {
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }, &failures);

  Log(rank, "mpi_barrier done");

  Stats mpi_allreduce = RunArm(warmup, iters, [rank, size](int iter) {
    std::uint64_t in = Contribution(iter, rank);
    std::uint64_t out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    // Warmup iterations use negative indices; only check the timed ones, whose
    // expected value is the same closed form the clio arm is held to.
    return iter < 0 || out == ExpectedSum(iter, size);
  }, &failures);

  Log(rank, "mpi_allreduce done");

  // ---- Attach to the local daemon -----------------------------------------
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(rank, "FAIL: CLIO_INIT(kClient) failed");
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 2;
  }

  // Rank 0 creates the pool; it is created across the whole cluster with one
  // container per node, which is exactly the membership the AllToOne barrier
  // counts. Every other rank waits for that to land before attaching.
  Log(rank, "attached to local daemon");

  const clio::run::PoolId pool_id(kBenchPoolMajor, 0);
  int create_rc = 0;
  if (rank == 0) {
    clio::run::MOD_NAME::Client creator(pool_id);
    auto create = creator.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "collective_bench_pool", pool_id);
    create.Wait();
    create_rc = static_cast<int>(create->return_code_);
    if (create_rc != 0) {
      Log(0, "FAIL: pool create rc=" + std::to_string(create_rc));
    }
  }
  if (rank == 0) Log(0, "pool create returned");
  MPI_Bcast(&create_rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (create_rc != 0) {
    MPI_Finalize();
    return 3;
  }
  // Let the new pool's metadata settle on every node before any rank routes to
  // it; the barrier alone only orders rank 0's create, not its propagation.
  std::this_thread::sleep_for(std::chrono::seconds(2));
  MPI_Barrier(MPI_COMM_WORLD);

  clio::run::MOD_NAME::Client client(pool_id);
  Log(rank, "pool ready");

  // ---- Do the collectives actually synchronize? -------------------------
  // Run BEFORE the timed arms: timing a collective that is not collective is
  // meaningless, and the return code alone cannot tell the difference.
  const int verify_rounds = EnvInt("COLL_BENCH_VERIFY_ROUNDS", 4);
  const int stagger_ms = EnvInt("COLL_BENCH_STAGGER_MS", 20);
  int barrier_violations = 0;
  if (verify_rounds > 0) {
    // MPI_Barrier first: a known-good barrier, so a failure here would mean the
    // CHECK is wrong rather than the collective.
    BarrierCheck mpi_chk = VerifyBarrier(rank, size, verify_rounds, stagger_ms,
                                         []() { MPI_Barrier(MPI_COMM_WORLD); });
    BarrierCheck clio_chk = VerifyBarrier(
        rank, size, verify_rounds, stagger_ms, [&client]() {
          auto q =
              clio::run::PoolQuery::AllToOne(kContainerHash, kBarrierBatchKey);
          auto f = client.AsyncBarrier(q);
          f.Wait();
        });
    // The allreduce is held to the same standard: it must synchronize, not just
    // return the right sum.
    BarrierCheck ar_chk = VerifyBarrier(
        rank, size, verify_rounds, stagger_ms, [&client]() {
          auto q = clio::run::PoolQuery::AllToOne(kContainerHash,
                                                  kAllReduceBatchKey);
          auto f = client.AsyncAllReduce(q, 1);
          f.Wait();
        });

    // Negative control: a plain LOCAL task is definitively not a barrier. If the
    // check above cannot catch this, then "0 violations" only means the check
    // is inert. This one is EXPECTED to be reported BROKEN and is deliberately
    // excluded from the exit code.
    BarrierCheck none_chk = VerifyBarrier(
        rank, size, verify_rounds, stagger_ms, [&client]() {
          auto f = client.AsyncCustom(clio::run::PoolQuery::Local(), "", 0);
          f.Wait();
        });

    if (rank == 0) {
      std::printf("\n=== Barrier semantics (staggered arrivals, %d rounds, "
                  "%dms stagger) ===\n",
                  verify_rounds, stagger_ms);
      std::printf("rank clock spread: %.1f us (must be small for the check to "
                  "mean anything)\n",
                  static_cast<double>(mpi_chk.clock_spread_ns) / 1000.0);
      const struct {
        const char *name;
        const BarrierCheck *c;
      } checks[] = {{"MPI_Barrier (control)", &mpi_chk},
                    {"clio barrier", &clio_chk},
                    {"clio allreduce", &ar_chk},
                    {"local task (neg ctrl)", &none_chk}};
      for (const auto &c : checks) {
        if (!c.c->conclusive) {
          std::printf("  %-22s INCONCLUSIVE (rank clocks not comparable)\n",
                      c.name);
          continue;
        }
        std::printf("  %-22s %s  release propagation min/med/max = "
                    "%+.1f / %+.1f / %+.1f us (%d violations)\n",
                    c.name, c.c->violations == 0 ? "HELD  " : "BROKEN",
                    c.c->worst_margin_us, c.c->med_margin_us,
                    c.c->max_margin_us, c.c->violations);
      }
      if (none_chk.conclusive && none_chk.violations == 0) {
        std::printf("  WARNING: the negative control did not register as "
                    "broken -- the check is not discriminating and the HELD "
                    "verdicts above mean nothing\n");
      }
      std::printf("\n");
      std::fflush(stdout);
    }
    barrier_violations =
        clio_chk.violations + ar_chk.violations + mpi_chk.violations;
    // A negative control that comes back clean means the detector is inert, so
    // the HELD verdicts prove nothing -- fail rather than report a false pass.
    if (none_chk.conclusive && none_chk.violations == 0) {
      ++barrier_violations;
    }
  }

  Log(rank, "barrier verification done; starting clio_barrier arm");

  // ---- Clio arms -----------------------------------------------------------
  // Every participant uses the same (container_hash, batch_key), which is what
  // makes their tasks one collective. A rank never has more than one request
  // outstanding (it waits before issuing the next), so a group can never
  // accumulate more members than the pool has containers and successive
  // iterations cannot merge into one batch.
  Stats clio_barrier = RunArm(warmup, iters, [&client](int) {
    auto q = clio::run::PoolQuery::AllToOne(kContainerHash, kBarrierBatchKey);
    auto f = client.AsyncBarrier(q);
    f.Wait();
    return f->return_code_ == 0;
  }, &failures);

  Log(rank, "clio_barrier done; starting clio_allreduce arm");

  int mismatches = 0;
  Stats clio_allreduce = RunArm(warmup, iters,
                                [&client, rank, size, &mismatches](int iter) {
    auto q = clio::run::PoolQuery::AllToOne(kContainerHash, kAllReduceBatchKey);
    auto f = client.AsyncAllReduce(q, Contribution(iter, rank));
    f.Wait();
    if (f->return_code_ != 0) return false;
    if (iter >= 0 && f->sum_ != ExpectedSum(iter, size)) {
      if (mismatches < 5) {
        Log(rank, "allreduce mismatch at iter " + std::to_string(iter) +
                      ": got " + std::to_string(f->sum_) + " expected " +
                      std::to_string(ExpectedSum(iter, size)));
      }
      ++mismatches;
      return false;
    }
    return true;
  }, &failures);

  Log(rank, "clio_allreduce done");

  // ---- Baseline arms: what a plain task costs on the same path ----------
  // The collective arms above are only interpretable against the cost of the
  // task machinery they are built on. These two are NOT collectives -- they are
  // ordinary Custom tasks with an empty body -- so they measure the floor:
  //
  //   clio_local_rtt   client -> its own daemon -> back (SHM, no network)
  //   clio_remote_rtt  client -> own daemon -> a PEER daemon -> back (one
  //                    network hop each way, the same hop a member takes to
  //                    reach the leader)
  //
  // Whatever clio_barrier costs ABOVE clio_remote_rtt is what the collective
  // machinery itself adds: parking in the BatchManager, waiting for the flusher
  // to notice, running the aggregate, and the 1->N completion broadcast.
  // Each rank targets its right-hand neighbour so every remote probe really
  // crosses the network (targeting a fixed node would make one rank's "remote"
  // probe local and skew the max-across-ranks number).
  const clio::run::u32 peer = static_cast<clio::run::u32>((rank + 1) % size);

  Stats clio_local_rtt = RunArm(warmup, iters, [&client](int) {
    auto f = client.AsyncCustom(clio::run::PoolQuery::Local(), "", 0);
    f.Wait();
    return f->return_code_ == 0;
  }, &failures);
  Log(rank, "clio_local_rtt done");

  Stats clio_remote_rtt = RunArm(warmup, iters, [&client, peer](int) {
    auto f = client.AsyncCustom(clio::run::PoolQuery::Physical(peer), "", 0);
    f.Wait();
    return f->return_code_ == 0;
  }, &failures);
  Log(rank, "clio_remote_rtt done");

  // ---- Report --------------------------------------------------------------
  Stats r_mpi_barrier = ReduceMax(mpi_barrier);
  Stats r_mpi_allreduce = ReduceMax(mpi_allreduce);
  Stats r_clio_barrier = ReduceMax(clio_barrier);
  Stats r_clio_allreduce = ReduceMax(clio_allreduce);
  Stats r_clio_local_rtt = ReduceMax(clio_local_rtt);
  Stats r_clio_remote_rtt = ReduceMax(clio_remote_rtt);

  int total_failures = 0;
  MPI_Reduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  int total_mismatches = 0;
  MPI_Reduce(&mismatches, &total_mismatches, 1, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  int total_barrier_violations = 0;
  MPI_Reduce(&barrier_violations, &total_barrier_violations, 1, MPI_INT,
             MPI_SUM, 0, MPI_COMM_WORLD);

  int exit_code = 0;
  if (rank == 0) {
    PrintHeader(size, iters, warmup);
    PrintRow("mpi_barrier", r_mpi_barrier);
    PrintRow("mpi_allreduce", r_mpi_allreduce);
    PrintRow("clio_barrier", r_clio_barrier);
    PrintRow("clio_allreduce", r_clio_allreduce);
    PrintRow("clio_local_rtt", r_clio_local_rtt);
    PrintRow("clio_remote_rtt", r_clio_remote_rtt);
    std::printf("\n");
    PrintRatio("clio_barrier   / mpi_barrier  ", r_clio_barrier,
               r_mpi_barrier);
    PrintRatio("clio_allreduce / mpi_allreduce", r_clio_allreduce,
               r_mpi_allreduce);
    std::printf("\n");
    // The decomposition: how much of a collective is just "a task went to
    // another node and came back", and how much is the collective itself.
    std::printf("breakdown (mean us):\n");
    std::printf("  plain local task round trip        %8.2f\n",
                r_clio_local_rtt.mean_us);
    std::printf("  plain remote task round trip       %8.2f  (+%.2f for the network hop)\n",
                r_clio_remote_rtt.mean_us,
                r_clio_remote_rtt.mean_us - r_clio_local_rtt.mean_us);
    std::printf("  collective barrier                 %8.2f  (+%.2f for the collective machinery)\n",
                r_clio_barrier.mean_us,
                r_clio_barrier.mean_us - r_clio_remote_rtt.mean_us);
    std::printf("  MPI barrier, for scale             %8.2f\n",
                r_mpi_barrier.mean_us);
    std::printf("\n");
    if (total_barrier_violations > 0) {
      std::printf("FAIL: %d barrier violations -- a participant returned before "
                  "the last one arrived, so this is not a barrier and its "
                  "timings describe nothing\n",
                  total_barrier_violations);
      exit_code = 6;
    } else if (total_mismatches > 0) {
      std::printf("FAIL: %d allreduce result mismatches -- the collective did "
                  "not combine correctly, so its timings are not meaningful\n",
                  total_mismatches);
      exit_code = 4;
    } else if (total_failures > 0) {
      std::printf("FAIL: %d failed iterations\n", total_failures);
      exit_code = 5;
    } else {
      std::printf("OK: all arms completed; allreduce results verified\n");
    }
    const char *csv = std::getenv("COLL_BENCH_CSV");
    if (csv != nullptr && *csv != '\0') {
      WriteCsv(csv, size, iters, r_mpi_barrier, r_mpi_allreduce,
               r_clio_barrier, r_clio_allreduce, r_clio_local_rtt,
               r_clio_remote_rtt);
    }
    std::fflush(stdout);
  }

  MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return exit_code;
}
