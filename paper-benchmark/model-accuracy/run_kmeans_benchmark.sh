#!/usr/bin/env bash
#===============================================================================
# run_kmeans_benchmark.sh -- the producer + k-means benchmark of one workload:
# the best single codec, NeuroPress and the oracle, all three selected by the
# same cost model (weights W at the PFS bandwidth COST_BW).
#
#   run_kmeans_benchmark.sh DATASET [W ...]   W = w_ct,w_dt,w_io
#                                             (default: 0,1,1 and 1,4,5)
#   FIG_DIR=DIR   where the figures go (default: compare_kmeans_runs.py's
#                 sim-tuning/<app>)
#
# 1. The exhaustive search: compress time, decompress time and ratio of all
#    45 settings on every chunk. Skipped when baselines/ has it for the same
#    input (baseline_store.py has); else run and kept there.
# 2. For each W: the oracle map, each chunk's lowest measured cost under W
#    at COST_BW (oracle_map.py).
# 3. run_exhaustive_e2e.sh: each setting as the single codec in its own full
#    run (1 write + READS k-means reads); the table
#    runs/DATASET_exhaustive_e2e.csv with the cost and rank of every setting
#    under every W.
# 4. For each W, tag km4<WL> (WL = w0-1-1, ...):
#      fixed   a link to the end-to-end run of the best single codec under W
#              (rank 1 of cost W in the table)
#      learn   NeuroPress learning, selecting with the weights W
#      oracle  each chunk stored with its setting in W's map
# 5. compare_kmeans_runs.py for each W: runs/DATASET_km4<WL>_compare.csv and
#    the figures.
# Every run: one PFS tier (TIERS=0.52e6:1, COST_BW=520000), READS=4,
# KMEANS=8, no selection log; one run at a time.
# Log: runs/DATASET_kmeans_benchmark.log.
#===============================================================================
set -euo pipefail
DS=$1; shift
WS=("$@"); [ ${#WS[@]} -eq 0 ] && WS=(0,1,1 1,4,5)
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/mnt/nvme0/v2-work/runs
PY=${PYTHON:-$HOME/np-venv/bin/python}
LOG=$RUNS/${DS}_kmeans_benchmark.log
export TIERS=${TIERS:-0.52e6:1} COST_BW=${COST_BW:-520000} READS=${READS:-4} \
       KMEANS=${KMEANS:-8} NO_SELECTION_LOG=1
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }

if pgrep -f bin/neuropress_field_replay > /dev/null; then
  echo "another neuropress_field_replay runs; no concurrent GPU work" >&2
  exit 1
fi
say "start $DS, weights ${WS[*]} (TIERS=$TIERS COST_BW=$COST_BW READS=$READS KMEANS=$KMEANS)"

# 1. exhaustive search, as for every stored baseline (4 tiers, 1 untimed read)
if "$PY" "$HERE/baseline_store.py" has "$DS" exhaustive > /dev/null 2>&1; then
  say "exhaustive search: stored baseline, not run"
else
  say "exhaustive search: run"
  env -u TIERS -u COST_BW -u READS -u KMEANS RUN_TAG=nolog \
    "$HERE/run_v2_workloads.sh" "$DS" exhaustive >> "$LOG" 2>&1
  rm -f "$RUNS/${DS}_exhaustive_nolog/chi_bdev.dat" "$RUNS/${DS}_exhaustive_nolog"/cte_tier.dat*
  "$PY" "$HERE/baseline_store.py" save "$DS" exhaustive >> "$LOG" 2>&1
fi

# 2. oracle maps
for W in "${WS[@]}"; do
  "$PY" "$HERE/oracle_map.py" "$DS" --w "$W" --bw "$COST_BW" \
    --out "$RUNS/${DS}_oracle_map_w${W//,/-}.csv" >> "$LOG" 2>&1
done

# 3. every setting end to end, then the table with every W
say "end-to-end single-codec runs: start"
"$HERE/run_exhaustive_e2e.sh" "$DS"
WARGS=(); for W in "${WS[@]}"; do WARGS+=(--w "$W"); done
"$PY" "$HERE/collect_single_codecs.py" "$DS" --bw "$COST_BW" "${WARGS[@]}" >> "$LOG" 2>&1
say "end-to-end single-codec runs: done, table runs/${DS}_exhaustive_e2e.csv"

# 4. + 5. the three options for each W
for W in "${WS[@]}"; do
  WL=w${W//,/-}; TAG=km4$WL
  read -r BEST NAME < <("$PY" - "$RUNS/${DS}_exhaustive_e2e.csv" "$W" <<'EOF'
import sys
import pandas as pd
t = pd.read_csv(sys.argv[1])
col = "rank_w" + "-".join(f"{float(x):g}" for x in sys.argv[2].split(","))
b = t.loc[t[col].idxmin()]
print(int(b.setting), b.method)
EOF
)
  ln -sfn "${DS}_fixed_km4s$BEST" "$RUNS/${DS}_fixed_$TAG"
  say "$WL: best single by cost = setting $BEST ($NAME); NeuroPress learning run"
  COST_W=$W RUN_TAG=$TAG "$HERE/run_v2_workloads.sh" "$DS" learn >> "$LOG" 2>&1
  rm -f "$RUNS/${DS}_learn_$TAG/chi_bdev.dat" "$RUNS/${DS}_learn_$TAG"/cte_tier.dat*
  say "$WL: oracle run"
  ORACLE_MAP=$RUNS/${DS}_oracle_map_$WL.csv RUN_TAG=$TAG \
    "$HERE/run_v2_workloads.sh" "$DS" oracle >> "$LOG" 2>&1
  rm -f "$RUNS/${DS}_oracle_$TAG/chi_bdev.dat" "$RUNS/${DS}_oracle_$TAG"/cte_tier.dat*
  PNG=()
  [ -n "${FIG_DIR:-}" ] && { mkdir -p "$FIG_DIR"; PNG=(--png "$FIG_DIR/v2_${DS}_kmeans_${READS}reads_$WL.png"); }
  "$PY" "$HERE/compare_kmeans_runs.py" "$DS" --tag "$TAG" --bw "$COST_BW" --w "$W" \
    --fixed-label "best single codec: $NAME (lowest cost)" --base-name "best single" \
    "${PNG[@]}" >> "$LOG" 2>&1
  say "$WL: compared, runs/${DS}_${TAG}_compare.csv"
done
say "KMEANS_BENCHMARK_DONE"
