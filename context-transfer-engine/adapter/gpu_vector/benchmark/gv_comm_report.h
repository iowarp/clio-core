/*
 * E1's COMMUNICATION / COMPUTE SPLIT, from the GV_COMM_TIMING counters.
 *
 * Communication is what the kernel spends inside the vector's Fetch, Hold and
 * Flush calls; compute is the rest of each block's resident GPU time. Both are
 * GPU cycles summed over blocks, so their ratio is a share of GPU time that
 * does not depend on the clock rate. Parked time (a block waiting off the GPU
 * for the runtime to move pages) is not GPU time and is not in either; the
 * YieldRunner round count shows it. A snapshot is taken around each timed
 * kernel so setup and untimed kernels on the same vector are excluded.
 */
#ifndef CLIO_GV_BENCH_COMM_REPORT_H_
#define CLIO_GV_BENCH_COMM_REPORT_H_

#include <cstdio>

namespace clio_gv_bench {

struct CommAcc {
  unsigned long long fetch = 0, hold = 0, flush = 0, busy = 0;
  unsigned long long nfetch = 0, nhold = 0, nflush = 0;
  /** Wall ms in host-side exchanges (CTE all-gathers/reductions between
   *  kernels). The baselines' comm_ms includes their MPI/CCL/SHMEM
   *  exchanges, so Eternia's must include its own to be comparable. */
  double host_ms = 0.0;

  /** Account `ms` of host-side exchange time. */
  void AddHost(double ms) { host_ms += ms; }

  /** Add the counters that moved between two ReadStats() snapshots. */
  template <typename StatsT>
  void Add(const StatsT &a, const StatsT &b) {
    fetch += b.fetch_cyc - a.fetch_cyc;
    hold += b.hold_cyc - a.hold_cyc;
    flush += b.flush_cyc - a.flush_cyc;
    busy += b.busy_cyc - a.busy_cyc;
    nfetch += b.fetch_calls - a.fetch_calls;
    nhold += b.hold_calls - a.hold_calls;
    nflush += b.flush_calls - a.flush_calls;
  }

  /** One greppable line. `timed_ms` is the wall time the counters cover;
   *  comm_ms/compute_ms apportion it by the GPU-time shares. */
  void Print(const char *wl, double timed_ms) const {
    if (busy == 0) {
      std::printf("COMM %s: no GV_COMM_TIMING counters (untimed build)\n", wl);
      return;
    }
    const double b = static_cast<double>(busy);
    const double pf = 100.0 * fetch / b, ph = 100.0 * hold / b,
                 pl = 100.0 * flush / b;
    const double share = (pf + ph + pl) / 100.0;
    // GPU shares apportion only the time spent outside host exchanges.
    const double gpu_ms = timed_ms - host_ms;
    const double comm_ms = share * gpu_ms + host_ms;
    std::printf("COMM %s: comm=%.1f%% (fetch %.1f%% hold %.1f%% flush %.1f%%"
                " of GPU busy; host xchg %.1f ms) | est comm_ms=%.1f"
                " compute_ms=%.1f of %.1f ms | calls fetch=%llu hold=%llu"
                " flush=%llu\n",
                wl, 100.0 * comm_ms / timed_ms, pf, ph, pl, host_ms, comm_ms,
                timed_ms - comm_ms, timed_ms, nfetch, nhold, nflush);
  }
};

}  // namespace clio_gv_bench

#endif  // CLIO_GV_BENCH_COMM_REPORT_H_
