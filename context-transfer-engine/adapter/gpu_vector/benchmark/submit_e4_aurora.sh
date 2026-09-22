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
# lbann is NOT in the default list: its edition trains a dense in-VRAM
# reference of the WHOLE model beside the paged one, so an 8 GB/node deck
# would need a 32 GB dense copy per node as well. It needs the baselines'
# --no-ref switch before it can take an E4 deck.
#
# The decks are the two-node tiering study's, 8 GB per node each (kmeans
# and grayscott split --data-mb across the nodes; weights owns its
# --pages x --blocks per node), 1 MB pages, a 4 GB HBM frame cache in front
# (--hbm-mb: the vector's own cache, not a CTE tier). Plan's E4 asks for
# 64 GB/node; this is the debugging-scale rung that must fit five minutes
# first. Job logs land in build-spike/pbs/<bench>_e4_<comp>.log.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHES=${BENCHES:-"kmeans grayscott weights gmx lammps_md"}
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
# gmx and lammps_md take their size through their geometry rather than a
# --data-mb, and both land on 8 GB/node at 4 nodes:
#
#   gmx        one page is one XY plane of u64 mesh points, so --page-kb
#              fixes K (page = K^2 * 8 bytes exactly). 20000 KB -> K =
#              1600, mesh 1600^3 * 8 = 32.8 GB global. --cap is in PAGES
#              here, not MB: 200 x 19.5 MB is about the 4 GB frame cache
#              the others get, and it clears the edition's own floor of
#              4 * blocks + 2.
#   lammps_md  ballistic mode (no --md, so no neighbour list): x and v are
#              binned at 512 bytes per bin and bins = floor(0.5713 *
#              lattice), measured from the stage-one deck (L=28 -> 16^3
#              bins, 2.0 MB per array). L=552 -> 315^3 bins -> 16 GB per
#              array globally, so x + v is 8 GB/node.
args_for() {
  case "$1" in
    kmeans)    echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --iters 1 --page-kb 1024" ;;
    grayscott) echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --steps 1 --repeat 1 --page-kb 1024" ;;
    weights)   echo "--blocks 64 --pages 128 --page-kb 1024 --hbm-mb ${HBM_MB} --repeat 1" ;;
    gmx)       echo "--page-kb 20000 --blocks 16 --cap 200 --repeat 1" ;;
    lammps_md) echo "--lattice 552 --steps 1 --page-kb 1024" ;;
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
