#!/usr/bin/env bash
#===============================================================================
# run_single_codecs.sh -- manual check of the best single codec: a list of single codecs, each
# run alone (every chunk with that one setting, Clio's fixed mode) on every final workload, in
# exactly the option-A setup of the best-static runs (run_workload_bench.sh 1x1 path):
#   the workload's cost model and cost bandwidth (final_config.py, one tier), 1 process x 1 chunk
#   in flight, 1 write + READS timed reads (10), a k-means iteration (8 clusters) after each read,
#   page cache dropped before each timed read, CPU governor performance during the run.
# Then single_codecs.py collect: per workload and codec the end-to-end time, the compress and
# decompress time, the I/O time, the compression ratio and the cost under the workload's cost model
# (from the measured times), next to the option-A best static.
#
#   run_single_codecs.sh [run] [collect]          (default: run collect)
#   SC_SETTINGS  setting indices, in run order (default below: the option-A best statics first)
#   SC_WORKLOADS only these workloads (default: every final workload)
#   SC_DIR       results (default /mnt/nvme0/v2-work/single-codecs)
#   DRY_RUN=1    print the runs, start nothing
#   SC_DIR/STOP  (an empty file): the queue stops before its next run (the current run ends and is
#                cleaned up first); a finished run is skipped, so the queue can stop and start again
#
# Every run: the dataset's 1-process part (DATASET-p0: links to every chunk; the verified chunk
# count must be the dataset's file count); after every run, also a failed one, the stored compressed
# data (chi_bdev.dat, cte_tier.dat*) is deleted and the folder is checked for files over 100 MB. A
# verified run (rc 0, read check done, every chunk bit-exact, a page-cache drop before each timed
# read, READS k-means iterations without a failure) moves to SC_DIR/<workload>/s<setting>_<name>; a
# failed one to SC_DIR/failed/ and is tried once more. Log: SC_DIR/single_codecs.out.
# Nothing else may use the GPU or read /mnt/nvme0 meanwhile.
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1
ROOT=/mnt/nvme0/v2-work
RUNS=$ROOT/runs
export SC_DIR=${SC_DIR:-$ROOT/single-codecs}
PY=${PYTHON:-$HOME/np-venv/bin/python}
LOG=$SC_DIR/single_codecs.out
# 30 ndzip shuffle=bit, 29 ndzip (the option-A best statics), 2 ans shuffle=byte, 1 ans shuffle=bit,
# 35 spratio, 38 spspeed, 28 lz4 shuffle=byte, 32 snappy
SETTINGS=${SC_SETTINGS:-30 29 2 1 35 38 28 32}
STEPS=" ${*:-run collect} "
mkdir -p "$SC_DIR"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
# a running benchmark binary, by process name (a command-line pattern would match shells)
wait_gpu() { while pgrep -x neuropress_fiel > /dev/null; do sleep 10; done; }
GOV=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
OLD_GOV=$(cat "${GOV[0]}" 2>/dev/null || true)
# restores the governor only after this script changed it
GOV_SET=0
restore_gov() { [ "$GOV_SET" = 1 ] && [ -n "$OLD_GOV" ] && echo "$OLD_GOV" | sudo tee "${GOV[@]}" > /dev/null; GOV_SET=0; }
trap restore_gov EXIT
mapfile -t NAMES < <(cd "$HERE" && "$PY" -c "import eval_v2_workloads as ev; print('\n'.join(ev.settings_list()[0]))")
selected() { [ -z "${SC_WORKLOADS:-}" ] || [[ " $SC_WORKLOADS " == *" $1 "* ]]; }

# a complete run: the untimed read check passed and every chunk verified bit-exact
complete() {
  [ -f "$1/stdout.log" ] && grep -q "READ check" "$1/stdout.log" &&
    grep -q "VERIFIED: \([0-9]*\) of \1 " "$1/stdout.log"
}

# delete a run's stored compressed data and check that nothing large is left
clean() {
  rm -f "$1/chi_bdev.dat" "$1"/cte_tier.dat*
  local big
  big=$(find "$1" -type f -size +100M 2>/dev/null)
  [ -z "$big" ] || say "WARNING: files over 100 MB left in $1: $big"
}

# the 1-process part of a dataset: links to every chunk (as run_kmeans_parallel.sh with PROCS=1)
make_part() {
  local d=$ROOT/$1-p0/fields
  rm -rf "$ROOT/$1-p0"
  mkdir -p "$d"
  ls "$ROOT/$1/fields" | while read -r f; do ln -s "$(readlink -f "$ROOT/$1/fields/$f")" "$d/$f"; done
}

# run_one WORKLOAD DATASET W BW READS KMEANS SETTING: one fixed-setting run; when verified, moved
# to SC_DIR/<workload>/s<setting>_<name>
run_one() {
  local wl=$1 ds=$2 w=$3 bw=$4 reads=$5 km=$6 s=$7
  local name=${NAMES[$s]// /_}
  local bwl tag run dest
  bwl=$(awk -v b="$bw" 'BEGIN{printf "%g", b / 1e6}')
  tag=km${reads}b${bwl}gscp1i1w${w//,/-}s$s
  run=$RUNS/${ds}-p0_fixed_$tag
  dest=$SC_DIR/${wl,,}/s${s}_$name
  complete "$dest" && return 0
  if [ -f "$SC_DIR/STOP" ]; then say "STOP file found: the queue stops"; exit 0; fi
  if [ -n "${DRY_RUN:-}" ]; then
    echo "would run: $wl setting $s (${NAMES[$s]}), W=$w, cost bandwidth $bwl GB/s, $reads reads -> $dest"
    return 0
  fi
  wait_gpu
  make_part "$ds"
  rm -rf "$run"   # an earlier, stopped attempt
  echo performance | sudo tee "${GOV[@]}" > /dev/null && GOV_SET=1
  say "START $wl setting $s (${NAMES[$s]})"
  env COST_BW="$bw" TIERS="$bw:1" READS="$reads" KMEANS="$km" NO_SELECTION_LOG=1 \
      CLIO_REPLAY_INFLIGHT=1 READ_INFLIGHT=1 CLIO_REPLAY_DROP_CACHES=1 \
      FIXED_SETTING="$s" RUN_TAG="$tag" ./run_v2_workloads.sh "$ds-p0" fixed >> "$LOG" 2>&1
  local rc=$?
  restore_gov
  clean "$run"
  local drops want got kmok
  drops=$(grep -c "page cache dropped" "$run/stdout.log" 2>/dev/null)
  want=$(find "$ROOT/$ds/fields" -name '*.chunk' | wc -l)
  got=$(grep -o -m1 "VERIFIED: [0-9]* of [0-9]*" "$run/stdout.log" 2>/dev/null | awk '{print $4}')
  kmok=$(grep -c "^KMEANS iteration [0-9]*/$reads done: [0-9]* chunk(s) skipped, 0 failed" "$run/stdout.log" 2>/dev/null)
  if [ "$rc" -eq 0 ] && complete "$run" && [ "${drops:-0}" -ge "$reads" ] && [ "${got:-0}" -eq "$want" ] \
      && [ "${kmok:-0}" -eq "$reads" ]; then
    mkdir -p "$(dirname "$dest")"
    rm -rf "$dest"
    mv "$run" "$dest"
    say "DONE $wl setting $s (${NAMES[$s]}): $got of $want chunks bit-exact, page cache dropped $drops times, $kmok k-means iterations -> $dest"
  else
    local keep
    keep=$SC_DIR/failed/$(basename "$run")_$(date +%m%d-%H%M%S)
    mkdir -p "$SC_DIR/failed"
    mv "$run" "$keep"
    say "FAILED $wl setting $s (rc $rc, ${got:-0} of $want chunks verified, page cache dropped ${drops:-0} times, ${kmok:-0} k-means iterations); logs: $keep"
    return 1
  fi
}

# run_one, and once more when it fails (a codec can fail now and then: spspeed shuffle=bit)
run_retry() {
  run_one "$@" && return 0
  say "RETRY $1 setting $7"
  run_one "$@"
}

if [[ "$STEPS" == *" run "* ]]; then
  say "single codecs: settings $SETTINGS, results in $SC_DIR"
  for s in $SETTINGS; do
    while read -r WL DS W BW READS KM; do
      selected "$WL" || continue
      run_retry "$WL" "$DS" "$W" "$BW" "$READS" "$KM" "$s"
    done < <(python3 final_config.py --shell)
  done
fi
if [[ "$STEPS" == *" collect "* ]] && [ -z "${DRY_RUN:-}" ]; then
  wait_gpu
  "$PY" single_codecs.py collect >> "$LOG" 2>&1 || say "collect FAILED"
  left=$(find "$SC_DIR" "$RUNS" \( -name chi_bdev.dat -o -name 'cte_tier.dat*' \) 2>/dev/null)
  if [ -z "$left" ]; then say "check: no stored compressed data left"; else say "WARNING: stored data left: $left"; fi
  say "SINGLE_CODECS_DONE"
fi
