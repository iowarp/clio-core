#!/usr/bin/env bash
# The tiering experiments: 16 GB of data across two nodes, through an 8 GB
# HBM tier (4 GB per node) into an 8 GB filesystem tier on Flare, then the
# same on DAOS. One job each, in sequence, five-minute walltime.
#
#   TIERS="flare daos" BENCHES="kmeans grayscott weights" submit_tier_all_aurora.sh
#
# TIERS=stack runs the three-tier config, DRAM -> DAOS -> Flare in one
# runtime (pbs_newcoro_aurora_2n_tier.sh, BENCH_TIER=stack): 2 GB of DRAM,
# 4 GB of DAOS and 4.5 GB of Flare per node in front of 8 GB per node, so
# every tier fills and spills into the next, and the persistent pair can
# take the whole 8 GB when the benchmark persists everything at the end.
#
# The three data-volume benchmarks. --hbm-mb matches the HBM tier; each
# benchmark splits --data-mb (kmeans, grayscott) across the nodes or owns its
# --pages x --blocks per node (weights), so every figure below is 8 GB per
# node. Iteration counts are the minimum that exercises the spill both ways
# (write it out, then fault it back), sized for the 200 s cap.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIERS=${TIERS:-"flare daos"}
BENCHES=${BENCHES:-"kmeans grayscott weights"}
HBM_MB=${HBM_MB:-4096}
TIER_MB=${TIER_MB:-4608}
TOP_MB=${TOP_MB:-2048}
DAOS_MB=${DAOS_MB:-4096}
FLARE_MB=${FLARE_MB:-4608}
# 1 MB PAGES throughout. The file tier is latency-bound per write: the
# first kmeans run at 64 KB pages wrote ~45 MB/s per node to Flare (a
# synchronous ~1.4 ms per page) and was still loading its 8 GB when the
# 200 s cap hit, against 1.0-1.3 GB/s measured with dd. At 1 MB a page the
# same bytes are 16x fewer operations.
args_for() {
  case "$1" in
    kmeans)    echo "--data-mb 16384 --hbm-mb ${HBM_MB} --iters 1 --page-kb 1024" ;;
    grayscott) echo "--data-mb 16384 --hbm-mb ${HBM_MB} --steps 1 --repeat 1 --page-kb 1024" ;;
    weights)   echo "--blocks 64 --pages 128 --page-kb 1024 --hbm-mb ${HBM_MB} --repeat 1" ;;
    *)         echo "" ;;
  esac
}
for tier in ${TIERS}; do
  for b in ${BENCHES}; do
    JOB_SCRIPT=pbs_newcoro_aurora_2n_tier.sh \
    BENCH_EXTRA="BENCH_TIER=${tier},BENCH_HBM_MB=${HBM_MB},BENCH_TIER_MB=${TIER_MB},BENCH_TOP_MB=${TOP_MB},BENCH_DAOS_MB=${DAOS_MB},BENCH_FLARE_MB=${FLARE_MB}" \
      "${HERE}/submit_2n_aurora.sh" "${b}" "${b}_tier_${tier}" $(args_for "${b}")
  done
done
echo "ALL DONE"
