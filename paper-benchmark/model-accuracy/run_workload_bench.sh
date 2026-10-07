#!/usr/bin/env bash
#===============================================================================
# run_workload_bench.sh -- the full parallel k-means benchmark of one workload,
# as done for Nyx and VPIC: best single codec, NeuroPress learning and oracle
# at 1x1, 2x8, 4x8 and 8x8 (processes x chunks in flight), each with a
# k-means iteration after every timed read; the k-means check after each
# configuration; then the compare CSVs and the configuration figure.
#
#   [CONFIGS="1x1 2x8 4x8 8x8"] [REFS=km10b1gp1i1,...] run_workload_bench.sh DATASET W COST_BW READS FIG_DIR
#
# W        cost weights w_ct,w_dt,w_io (e.g. 1,4,5)
# COST_BW  cost-model bandwidth in bytes per ms (1000000 = 1 GB/s)
# READS    timed reads per run (each followed by one k-means iteration)
# FIG_DIR  where the compare CSVs and configs_<DATASET>_w<W>.png go
# TAG_EXTRA (optional) is appended to the run tags (e.g. dc), so runs of another
# setup -- CLIO_REPLAY_DROP_CACHES=1: page cache dropped before each timed read
# -- stay apart from the others. PYTHON (default ~/np-venv/bin/python) runs the plot; GOVERNOR= (empty) leaves the
# CPU governor alone where sudo is not available (e.g. Delta), see run_kmeans_parallel.sh.
# Log: runs/DATASET_bench.log (steps, k-means checks).
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
DS=$1; W=$2; COST_BW=$3; READS=$4; FIG=$5
CONFIGS=${CONFIGS:-1x1 2x8 4x8 8x8}
RUNS=/mnt/nvme0/v2-work/runs
LOG=$RUNS/${DS}_bench.log
BWL=$(awk -v b="$COST_BW" 'BEGIN{printf "%g", b / 1e6}')
PREFIX=km${READS}b${BWL}g
[ "$COST_BW" = 520000 ] && PREFIX=km${READS}
PREFIX=$PREFIX${TAG_EXTRA:-}   # e.g. TAG_EXTRA=dc for runs with CLIO_REPLAY_DROP_CACHES=1
WL=w${W//,/-}
mkdir -p "$FIG"
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }
say "start $DS: W=$W bw=$BWL GB/s reads=$READS configs=$CONFIGS"
refs=${REFS:-}   # earlier configurations to compare with (tag prefixes, comma list)
for c in $CONFIGS; do
  p=${c%x*}; i=${c#*x}
  say "config ${p}x${i}"
  ( cd "$HERE" && env COST_BW=$COST_BW READS=$READS FIG_DIR=$FIG PROCS=$p INFLIGHT=$i TAG_PREFIX=$PREFIX \
      ${refs:+REF_TAG=$refs} ./run_kmeans_parallel.sh "$DS" "$W" ) \
      > "$RUNS/${DS}_${p}x${i}_bench.out" 2>&1 || say "config ${p}x${i}: run FAILED"
  "$HERE/check_kmeans.sh" "$DS" "${PREFIX}p${p}i${i}$WL" "$p" "$READS" >> "$LOG" 2>&1 \
    && say "config ${p}x${i}: k-means check ok" || say "config ${p}x${i}: k-means check FAILED"
  cp "$RUNS/${DS}_${PREFIX}p${p}i${i}${WL}_compare.csv" "$FIG/" 2>/dev/null
  refs=${refs:+$refs,}${PREFIX}p${p}i${i}
done
rm -f "$FIG"/v2_${DS}_${PREFIX}p*${WL}.png
( cd "$HERE" && "${PYTHON:-$HOME/np-venv/bin/python}" plot_parallel_configs.py "$DS" --w "${W//,/-}" \
    --prefix "$PREFIX" --reads "$READS" --bw "$COST_BW" --out "$FIG/configs_${DS}_${WL}.png" ) >> "$LOG" 2>&1
cp "$RUNS/${DS}_configs_${WL}.csv" "$FIG/" 2>/dev/null
say "BENCH_DONE $DS"
