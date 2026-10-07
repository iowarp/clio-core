#!/usr/bin/env bash
#===============================================================================
# run_final_suite.sh -- the final paper evaluation, every parameter from final_config.py:
# per workload and iteration, best static + NeuroPress learning + cost oracle
# (run_workload_bench.sh, 1 process x 1 chunk in flight, page cache dropped before each timed
# read) and the time oracle (run_v2_workloads.sh oracle with the time-oracle map); then the
# checks (final_check.py), the summary figure and the tiering figures.
#
#   run_final_suite.sh [WORKLOAD ...]        (default: every workload of final_config.py)
#
# The time-oracle map of a workload is made once, from its first iteration's best-static run
# (time_oracle_map.py: real disk speed fitted to that run), and used in every iteration.
# Run tags: km<READS>b<GB/s>gfin<iteration>p1i1w<weights> (time oracle: the same + "to").
# Results: FIG_ROOT/<workload>/iter<i>/ (compare / configs CSVs, figure), FIG_ROOT/final_*.csv,
# FIG_ROOT/summary_1x1_final.png, FIG_ROOT/tiering_summary.png, FIG_ROOT/<workload>/tiering_*;
# log: runs/final_suite.out.
# Nothing else may use the GPU or read /mnt/nvme0 while this runs.
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1
RUNS=/mnt/nvme0/v2-work/runs
PY=${PYTHON:-$HOME/np-venv/bin/python}
FIG_ROOT=${FIG_ROOT:-$HERE/../figures/new-workloads/sim-tuning/final}
LOG=$RUNS/final_suite.out
ITER=$(python3 final_config.py --iterations)
mapfile -t LINES < <(python3 final_config.py --shell)
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }
# a running benchmark binary, by process name (a command-line pattern would match shells)
wait_gpu() { while pgrep -x neuropress_fiel > /dev/null; do sleep 10; done; }
GOV=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
OLD_GOV=$(cat "${GOV[0]}" 2>/dev/null || true)
restore_gov() { [ -n "$OLD_GOV" ] && echo "$OLD_GOV" | sudo tee "${GOV[@]}" > /dev/null; }
trap restore_gov EXIT
mkdir -p "$FIG_ROOT"

for ((it = 1; it <= ITER; it++)); do
  for line in "${LINES[@]}"; do
    read -r WL DS W BW READS KMEANS <<< "$line"
    [ $# -gt 0 ] && [[ " $* " != *" $WL "* ]] && continue
    BWL=$(awk -v b="$BW" 'BEGIN{printf "%g", b / 1e6}')
    WL_TAG=w${W//,/-}
    TAG=km${READS}b${BWL}gfin${it}p1i1$WL_TAG
    WL_DIR=$FIG_ROOT/$(echo "$WL" | tr '[:upper:]' '[:lower:]')
    wait_gpu
    say "START $WL iteration $it (W=$W, cost bandwidth $BWL GB/s)"
    CLIO_REPLAY_DROP_CACHES=1 TAG_EXTRA=fin$it CONFIGS=1x1 OPTIONS="fixed learn oracle" KMEANS=$KMEANS \
      ./run_workload_bench.sh "$DS" "$W" "$BW" "$READS" "$WL_DIR/iter$it" >> "$LOG" 2>&1 \
      || say "$WL iteration $it: bench FAILED"
    MAP=$RUNS/${DS}_oracle_map_time_km${READS}b${BWL}gfin1p1i1$WL_TAG.csv
    if [ ! -f "$MAP" ]; then
      "$PY" time_oracle_map.py "$WL" --fixed-tag "$TAG" --out "$MAP" >> "$LOG" 2>&1 \
        || { say "$WL: time-oracle map FAILED"; continue; }
    fi
    wait_gpu
    echo performance | sudo tee "${GOV[@]}" > /dev/null
    say "START $WL iteration $it time oracle"
    env COST_BW="$BW" TIERS="$BW:1" READS="$READS" KMEANS="$KMEANS" NO_SELECTION_LOG=1 \
        CLIO_REPLAY_INFLIGHT=1 READ_INFLIGHT=1 CLIO_REPLAY_DROP_CACHES=1 ORACLE_MAP="$MAP" \
        RUN_TAG="${TAG}to" ./run_v2_workloads.sh "$DS-p0" oracle >> "$LOG" 2>&1 \
      || say "$WL iteration $it: time oracle FAILED"
    rm -f "$RUNS/$DS-p0_oracle_${TAG}to/chi_bdev.dat" "$RUNS/$DS-p0_oracle_${TAG}to"/cte_tier.dat*
    restore_gov
    say "DONE $WL iteration $it"
  done
done

# the measured summary first (the tiering figures use it as their anchor), then the tiering,
# then the checks again with the tiering agreement, then the summary figures
say "checks, tiering figures and summary figure"
"$PY" final_check.py --fig-root "$FIG_ROOT" >> "$LOG" 2>&1 || say "final_check (runs only) reports FAILED checks"
for line in "${LINES[@]}"; do
  read -r WL _ <<< "$line"
  "$PY" tiering_final.py "$WL" --fig-root "$FIG_ROOT" >> "$LOG" 2>&1 || say "$WL tiering figure FAILED"
done
"$PY" final_check.py --fig-root "$FIG_ROOT" >> "$LOG" 2>&1 || say "final_check reports FAILED checks"
"$PY" plot_workload_summary.py --final "$FIG_ROOT" >> "$LOG" 2>&1 || say "summary figure FAILED"
"$PY" tiering_summary.py --fig-root "$FIG_ROOT" >> "$LOG" 2>&1 || say "tiering summary figure FAILED"
say "FINAL_SUITE_DONE"
