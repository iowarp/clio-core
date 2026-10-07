#!/usr/bin/env python3
"""The final paper-evaluation setup (decided 2026-10-07): one place for every parameter, read
by run_final_suite.sh, time_oracle_map.py, final_check.py, tiering_final.py and
plot_workload_summary.py. Transfer this file unchanged to the main testbed; there, only the
measured inputs are made again (the exhaustive search, the real disk speed, the time-oracle map).

    final_config.py --shell        print 'WORKLOAD DATASET W_CT,W_DT,W_IO COST_BW READS KMEANS' lines
    final_config.py --iterations   print ITERATIONS

Per workload: the dataset, the cost model (w_ct, w_dt, w_io) and the cost bandwidth (bytes per
ms, a tier speed) that NeuroPress and the best static codec select by. Run setup: 1 process,
1 chunk in flight, 1 write + READS reads each followed by one k-means iteration (KMEANS
clusters), page cache dropped before each timed read, NeuroPress learning on and exploration
off. Options: best static (the lowest total cost under the cost model), NeuroPress, cost oracle
(each chunk's lowest-cost setting) and time oracle (each chunk's lowest end-to-end time setting
at the workload's measured real disk speed, from the exhaustive search).
Tiering (offline model, tiering_final.py): single tier = the burst buffer; tiered =
write-through to the burst buffer (100 %) with DRAM and NVMe copies of TIERED_COPIES % of the
chunks, dealt round-robin in write order; the PFS only receives a background flush.
"""
import sys

# workload: (dataset, cost weights w_ct / w_dt / w_io, cost bandwidth in bytes per ms)
WORKLOADS = {
    "Nyx": ("nyx-multiphase-50g", (1.0, 40.0, 5.0), 1_000_000),
    "VPIC": ("vpic-slabs", (1.0, 40.0, 5.0), 500_000),
    "WarpX": ("ref-warpx-tune-final25g", (1.0, 20.0, 1.0), 500_000),
    "incflo": ("ref-incflo-tune-final25g", (1.0, 40.0, 5.0), 1_000_000),
}
READS = 10                 # timed reads, each followed by one k-means iteration
KMEANS = 8                 # k-means clusters
ITERATIONS = 3             # repeated runs per workload (error bars)
PROCS, INFLIGHT = 1, 1     # 1 process, 1 chunk in flight
# tier speeds (bytes per ms) and the tiered composition (% of the chunks with a copy)
TIER_BW = {"DRAM": 12e6, "NVMe": 1e6, "burst buffer": 0.5e6, "PFS": 0.25e6}
SINGLE_TIER = "burst buffer"
TIERED_COPIES = {"DRAM": 10, "NVMe": 20}     # the rest is read from the burst buffer
# a run whose read I/O per stored GB and read differs from the median of the same workload and
# option over the iterations by more than this fraction is flagged as disturbed
READ_SPEED_TOLERANCE = 0.08
# NeuroPress's own work per chunk / per training step, measured in learning runs (phases.csv)
SEL_MS = {"Nyx": 0.300, "VPIC": 0.352, "WarpX": 0.278, "incflo": 0.260}
TRAIN_MS = 0.15


def model_name(w):
    """@return the cost weights as text, e.g. '1/40/5'."""
    return "/".join(f"{x:g}" for x in w)


def bw_label(bw):
    """@return a bandwidth (bytes per ms) as run_workload_bench.sh tags it, e.g. '0.5'."""
    return f"{bw / 1e6:g}"


if __name__ == "__main__":
    if "--shell" in sys.argv:
        for wl, (ds, w, bw) in WORKLOADS.items():
            print(wl, ds, ",".join(f"{x:g}" for x in w), int(bw), READS, KMEANS)
    if "--iterations" in sys.argv:
        print(ITERATIONS)
