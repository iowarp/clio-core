#!/usr/bin/env bash
# E4 (tiering composition) at 4 nodes: one paged benchmark against the
# plan's five DRAM/DAOS/Lustre compositions, one job each, in sequence.
#
#   BENCHES="kmeans grayscott weights" COMPS="dram100 dram75 bal25 daos70 lustre70" \
#     submit_e4_aurora.sh
#
# Every composition splits the same per-node tier budget (TIER_BUDGET_MB,
# default 10240: the 8 GB/node deck plus a quarter, so no put is refused
# and every tier below the top fills and spills). A 0 share leaves that
# tier out of the config (pbs_newcoro_aurora_2n_tier.sh, BENCH_TIER=stack):
#
#   dram100   DRAM-only reference     100 /  0 /  0
#   dram75    DRAM-heavy               75 / 25 /  0
#   bal25     balanced                 25 / 50 / 25
#   daos70    DAOS-heavy               10 / 70 / 20
#   lustre70  Lustre-heavy             10 / 20 / 70
#
# The decks are the two-node tiering study's, 8 GB per node each (kmeans
# and grayscott split --data-mb across the nodes; weights owns its
# --pages x --blocks per node), 1 MB pages, a 4 GB HBM frame cache in front
# (--hbm-mb: the vector's own cache, not a CTE tier). Plan's E4 asks for
# 64 GB/node; this is the debugging-scale rung that must fit five minutes
# first. Job logs land in build-spike/pbs/<bench>_e4_<comp>.log.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHES=${BENCHES:-"kmeans grayscott weights"}
COMPS=${COMPS:-"dram100 dram75 bal25 daos70 lustre70"}
TIER_BUDGET_MB=${TIER_BUDGET_MB:-10240}
HBM_MB=${HBM_MB:-4096}
DATA_MB=${DATA_MB:-32768}
E4_CAP=${E4_CAP:-240}

# Shares in percent: dram daos flare.
shares_for() {
  case "$1" in
    dram100)  echo "100 0 0" ;;
    dram75)   echo "75 25 0" ;;
    bal25)    echo "25 50 25" ;;
    daos70)   echo "10 70 20" ;;
    lustre70) echo "10 20 70" ;;
    *)        echo "" ;;
  esac
}
args_for() {
  case "$1" in
    kmeans)    echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --iters 1 --page-kb 1024" ;;
    grayscott) echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --steps 1 --repeat 1 --page-kb 1024" ;;
    weights)   echo "--blocks 64 --pages 128 --page-kb 1024 --hbm-mb ${HBM_MB} --repeat 1" ;;
    *)         echo "" ;;
  esac
}
for b in ${BENCHES}; do
  for c in ${COMPS}; do
    read -r pd pa pf <<< "$(shares_for "${c}")"
    [ -n "${pd:-}" ] || { echo "unknown composition ${c}"; exit 2; }
    top=$(( TIER_BUDGET_MB * pd / 100 ))
    daos=$(( TIER_BUDGET_MB * pa / 100 ))
    flare=$(( TIER_BUDGET_MB * pf / 100 ))
    echo "== ${b} ${c}: dram ${top} MB, daos ${daos} MB, flare ${flare} MB per node"
    JOB_SCRIPT=pbs_newcoro_aurora_4n_tier.sh \
    BENCH_EXTRA="BENCH_TIER=stack,BENCH_LABEL=${c},BENCH_TOP_MB=${top},BENCH_DAOS_MB=${daos},BENCH_FLARE_MB=${flare},BENCH_CAP=${E4_CAP}" \
      "${HERE}/submit_2n_aurora.sh" "${b}" "${b}_e4_${c}" $(args_for "${b}")
  done
done
echo "E4 ALL DONE"
