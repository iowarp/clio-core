#!/usr/bin/env bash
#===============================================================================
# run_kmeans_parallel.sh -- the producer + k-means benchmark with PROCS
# processes and INFLIGHT chunks in flight in each: the best single codec,
# NeuroPress learning and the oracle, all selected by the same cost model.
#
#   run_kmeans_parallel.sh DATASET [W ...]   W = w_ct,w_dt,w_io
#                                            (default: 0,1,1 and 1,4,5)
#   PROCS=2       processes, started together; process i gets every PROCS-th
#                 chunk (DATASET-p<i>: links to the same files) and has its
#                 own Clio runtime (a client cannot pass GPU read buffers to
#                 a runtime in another process); NeuroPress learns in each
#                 process from its own chunks
#   INFLIGHT=8    chunk writes (CLIO_REPLAY_INFLIGHT) and reads
#                 (READ_INFLIGHT) in flight in each process
#
# The stored exhaustive search of DATASET (baselines/) is used, or run once
# first (all 45 settings on every chunk, 1 process, as run_kmeans_benchmark.sh).
# For each W:
#   best single  the candidate setting with the lowest total cost under W at
#                COST_BW in the exhaustive search (eval_v2_workloads
#                .candidate_settings), one fixed run
#   learn        NeuroPress learning, selecting with W
#   oracle       each chunk's lowest-cost candidate setting under W (the map
#                runs/DATASET_oracle_map_<WL>.csv, made again here)
# Runs: runs/DATASET-p<i>_<mode>_<TAG>, TAG = <TAG_PREFIX>p<PROCS>i<INFLIGHT><WL>;
# TAG_PREFIX is km<READS>, plus b<GB/s>g when COST_BW is not 520000 (0.52
# GB/s), e.g. km10b2gp2i8w1-10-8.25. TIERS follows COST_BW unless given.
# Each option's whole wall clock (start to exit of every process) goes to
# runs/DATASET_<TAG>_walls.csv. Then compare_parallel_runs.py: the table
# runs/DATASET_<TAG>_compare.csv and a figure, against REF_TAG (runs of this
# script with REF_PROCS processes, e.g. PROCS=1 INFLIGHT=1) when given, else
# against the serial runs km4<WL> of run_kmeans_benchmark.sh.
# Every run: one PFS tier (TIERS=0.52e6:1, COST_BW=520000), READS=4,
# KMEANS=8, no selection log. Log: runs/DATASET_kmeans_parallel.log.
#===============================================================================
set -euo pipefail
DS=$1; shift
WS=("$@"); [ ${#WS[@]} -eq 0 ] && WS=(0,1,1 1,4,5)
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/mnt/nvme0/v2-work
RUNS=$ROOT/runs
PY=${PYTHON:-$HOME/np-venv/bin/python}
PROCS=${PROCS:-2}; INFLIGHT=${INFLIGHT:-8}
# OPTIONS: which options to run (any of fixed learn oracle hcompress xgboost); runs of
# the others already present with the same tag stay in the comparison.
OPTIONS=${OPTIONS:-fixed learn oracle}
LOG=$RUNS/${DS}_kmeans_parallel.log
export COST_BW=${COST_BW:-520000} READS=${READS:-4} KMEANS=${KMEANS:-8} NO_SELECTION_LOG=1
export TIERS=${TIERS:-${COST_BW}:1}
BWL=$(awk -v b="$COST_BW" 'BEGIN{printf "%g", b / 1e6}')   # GB/s
BWSUF=""; [ "$COST_BW" != 520000 ] && BWSUF="b${BWL}g"
TAG_PREFIX=${TAG_PREFIX:-km${READS}$BWSUF}
export CLIO_REPLAY_INFLIGHT=$INFLIGHT READ_INFLIGHT=$INFLIGHT
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }

if pgrep -f bin/neuropress_field_replay > /dev/null; then
  echo "another neuropress_field_replay runs; no concurrent GPU work" >&2
  exit 1
fi
say "start $DS, weights ${WS[*]}, $PROCS processes x $INFLIGHT in flight"

# GOVERNOR (default performance): the CPU frequency governor while the runs
# go, restored on exit (needs sudo); GOVERNOR= (empty) leaves it alone.
GOV_FILES=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
OLD_GOV=$(cat "${GOV_FILES[0]}" 2>/dev/null || true)
if [ -n "${GOVERNOR-performance}" ] && [ -n "$OLD_GOV" ] && [ "$OLD_GOV" != "${GOVERNOR-performance}" ]; then
  echo "${GOVERNOR-performance}" | sudo tee "${GOV_FILES[@]}" > /dev/null &&
    trap 'echo "$OLD_GOV" | sudo tee "${GOV_FILES[@]}" > /dev/null' EXIT
  say "CPU governor ${GOVERNOR-performance} (was $OLD_GOV; restored on exit)"
fi

# the exhaustive search, once per input (as run_kmeans_benchmark.sh)
if "$PY" "$HERE/baseline_store.py" has "$DS" exhaustive > /dev/null 2>&1; then
  say "exhaustive search: stored baseline, not run"
else
  say "exhaustive search: run"
  env -u TIERS -u COST_BW -u READS -u KMEANS -u CLIO_REPLAY_INFLIGHT -u READ_INFLIGHT \
    RUN_TAG=nolog "$HERE/run_v2_workloads.sh" "$DS" exhaustive >> "$LOG" 2>&1
  rm -f "$RUNS/${DS}_exhaustive_nolog/chi_bdev.dat" "$RUNS/${DS}_exhaustive_nolog"/cte_tier.dat*
  "$PY" "$HERE/baseline_store.py" save "$DS" exhaustive >> "$LOG" 2>&1
fi

# the parts: every PROCS-th chunk, links to the same files
for ((i = 0; i < PROCS; i++)); do
  d=$ROOT/$DS-p$i/fields; rm -rf "$ROOT/$DS-p$i"; mkdir -p "$d"
  ls "$ROOT/$DS/fields" | awk -v n="$PROCS" -v i="$i" 'NR % n == i' | while read -r f; do
    ln -s "$(readlink -f "$ROOT/$DS/fields/$f")" "$d/$f"
  done
done

# one option: PROCS processes at once, then their stored data deleted
run_option() {   # MODE TAG [VAR=VALUE ...]
  local mode=$1 tag=$2; shift 2
  local pids=() t0
  t0=$(date +%s.%N)
  for ((i = 0; i < PROCS; i++)); do
    env "$@" RUN_TAG="$tag" "$HERE/run_v2_workloads.sh" "$DS-p$i" "$mode" >> "$LOG" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  local walls=$RUNS/${DS}_${tag}_walls.csv
  [ -f "$walls" ] || echo "mode,wall_s" > "$walls"
  echo "$mode,$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.3f", b-a}')" >> "$walls"
  for ((i = 0; i < PROCS; i++)); do
    rm -f "$RUNS/$DS-p${i}_${mode}_$tag/chi_bdev.dat" "$RUNS/$DS-p${i}_${mode}_$tag"/cte_tier.dat*
  done
  return $rc
}

for W in "${WS[@]}"; do
  WL=w${W//,/-}; TAG=${TAG_PREFIX}p${PROCS}i${INFLIGHT}$WL
  case " $OPTIONS " in *" fixed "*) rm -f "$RUNS/${DS}_${TAG}_walls.csv" ;; esac
  MAP=$RUNS/${DS}_oracle_map_$WL${BWSUF:+_$BWSUF}.csv
  "$PY" "$HERE/oracle_map.py" "$DS" --w "$W" --bw "$COST_BW" --out "$MAP" >> "$LOG" 2>&1
  read -r BEST NAME < <(cd "$HERE" && "$PY" - "$DS" "$W" "$COST_BW" <<'EOF'
import os, sys
import numpy as np
import eval_v2_workloads as ev
ds, w, bw = sys.argv[1], tuple(float(x) for x in sys.argv[2].split(",")), float(sys.argv[3])
names, store = ev.settings_list()
_, c = ev.load_truth(os.path.join("/mnt/nvme0/v2-work/baselines", ds, "exhaustive"),
                     len(names), store, w, bw)
ok = ~np.isnan(c).any(axis=1)
b = int(np.argmin(ev.for_selection(c[ok]).sum(axis=0)))
print(b, names[b])
EOF
)
  say "$WL: best single (exhaustive CSV) = setting $BEST ($NAME)"
  for opt in $OPTIONS; do
    case $opt in
      fixed)     run_option fixed "$TAG" FIXED_SETTING="$BEST"; say "$WL: best single done" ;;
      learn)     run_option learn "$TAG" COST_W="$W"; say "$WL: NeuroPress learning done" ;;
      oracle)    run_option oracle "$TAG" ORACLE_MAP="$MAP"; say "$WL: oracle done" ;;
      hcompress) run_option hcompress "$TAG" COST_W="$W"; say "$WL: HCompress done" ;;
      xgboost)   run_option xgboost "$TAG" COST_W="$W"; say "$WL: XGBoost done" ;;
      *) say "unknown option $opt (fixed|learn|oracle|hcompress|xgboost)"; exit 1 ;;
    esac
  done
  REF=(--serial-tag "km4$WL")
  # REF_TAG: one or more tag prefixes (comma-separated), each a configuration
  # of this script already run with the same weights
  [ -n "${REF_TAG:-}" ] && REF=(--ref-tag "$(echo "$REF_TAG" | sed "s/,/$WL,/g")$WL" \
                                --ref-procs "${REF_PROCS:-1}")
  "$PY" "$HERE/compare_parallel_runs.py" "$DS" --procs "$PROCS" --tag "$TAG" --w "$W" \
    --bw "$COST_BW" "${REF[@]}" --best "$NAME" ${FIG_DIR:+--fig-dir "$FIG_DIR"} >> "$LOG" 2>&1
  say "$WL: compared, runs/${DS}_${TAG}_compare.csv"
done
say "KMEANS_PARALLEL_DONE"
