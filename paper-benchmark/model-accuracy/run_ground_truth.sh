#!/usr/bin/env bash
#===============================================================================
# run_ground_truth.sh -- per-chunk ground truth of the final workloads (final_config.py),
# measured once (the workloads are deterministic):
#   phase 1: the exhaustive search of each workload (run_v2_workloads.sh exhaustive: every
#            chunk compressed and decompressed with all 45 settings: codec times and ratio,
#            the per-chunk best of any cost model)
#   phase 2: one full run per candidate setting (run_v2_workloads.sh fixed: every chunk with
#            that one setting; 1 write + GT_READS timed reads), in the order of
#            GT_DIR/plan.txt (ground_truth.py plan: the most relevant settings of every
#            workload first): the real write, read and decompress time of every chunk
#   phase 3: ground_truth.py collect (tables, exhaustive vs the stored baseline) and a check
#            that no stored data is left anywhere
#
#   run_ground_truth.sh [PHASE ...]          (default: 1 2 3)
#   GT_DIR    results (default /mnt/nvme0/v2-work/ground-truth)
#   GT_READS  timed reads per run (default 2; the exhaustive search: 1)
#   DRY_RUN=1 print the runs, start nothing
#   GT_DIR/STOP (an empty file): the queue stops before its next run (a clean stop; the
#   current run ends and is cleaned up first)
#   tests on a small dataset: GT_WORKLOADS="Nyx" (only these workloads) and
#   GT_DATASETS="Nyx=nyx-multiphase-1g" (another dataset for a workload), with another GT_DIR
#
# Every run: the whole dataset (/mnt/nvme0/v2-work/<dataset>/fields; the verified chunk count
# must be the dataset's file count), 1 process, 1 chunk in flight, page cache dropped before each timed read
# (CLIO_REPLAY_DROP_CACHES=1; the count of drops in stdout.log is checked), a k-means
# iteration after each read, the workload's cost model and cost bandwidth (one tier), CPU
# governor performance during the run. After every run, also a failed one, its stored
# compressed data (chi_bdev.dat, cte_tier.dat*) is deleted and its folder is checked for files
# over 100 MB. A verified run's folder (CSV files and logs only) moves to
# GT_DIR/<workload>/exhaustive or GT_DIR/<workload>/s<setting>_<name>; a failed run's folder
# (logs only) moves to GT_DIR/failed/ and the run is tried once more (a codec can fail now and
# then: spspeed shuffle=bit wrote one corrupt chunk on 2026-10-07). A finished run is skipped, so
# the queue can stop and start again.
# Log: GT_DIR/ground_truth.out. Nothing else may use the GPU or read /mnt/nvme0 meanwhile.
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1
RUNS=/mnt/nvme0/v2-work/runs
export GT_DIR=${GT_DIR:-/mnt/nvme0/v2-work/ground-truth}
READS=${GT_READS:-2}
PY=${PYTHON:-$HOME/np-venv/bin/python}
LOG=$GT_DIR/ground_truth.out
PHASES=" ${*:-1 2 3} "
mkdir -p "$GT_DIR"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
# a running benchmark binary, by process name (a command-line pattern would match shells)
wait_gpu() { while pgrep -x neuropress_fiel > /dev/null; do sleep 10; done; }
GOV=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
OLD_GOV=$(cat "${GOV[0]}" 2>/dev/null || true)
restore_gov() { [ -n "$OLD_GOV" ] && echo "$OLD_GOV" | sudo tee "${GOV[@]}" > /dev/null; }
trap restore_gov EXIT
declare -A WL_LINE   # workload -> "dataset weights bandwidth kmeans"
while read -r WL DS W BW _ KM; do WL_LINE[$WL]="$DS $W $BW $KM"; done < <(python3 final_config.py --shell)
for x in ${GT_DATASETS:-}; do   # tests: another dataset for a workload
  read -r _ W BW KM <<< "${WL_LINE[${x%%=*}]}"
  WL_LINE[${x%%=*}]="${x#*=} $W $BW $KM"
done
selected() { [ -z "${GT_WORKLOADS:-}" ] || [[ " $GT_WORKLOADS " == *" $1 "* ]]; }

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

# run_one WORKLOAD MODE TAG DEST READS [VAR=value ...]: one run; when verified, moved to DEST
run_one() {
  local wl=$1 mode=$2 tag=$3 dest=$4 reads=$5
  shift 5
  local ds w bw km
  read -r ds w bw km <<< "${WL_LINE[$wl]}"
  local run=$RUNS/${ds}_${mode}_${tag}
  complete "$dest" && return 0
  if [ -f "$GT_DIR/STOP" ]; then say "STOP file found: the queue stops"; exit 0; fi
  if [ -n "${DRY_RUN:-}" ]; then
    echo "would run: $wl $mode $tag reads=$reads $* -> $dest"
    return 0
  fi
  wait_gpu
  rm -rf "$run"   # an earlier, stopped attempt
  echo performance | sudo tee "${GOV[@]}" > /dev/null
  say "START $wl $mode $tag"
  env "$@" COST_W="$w" COST_BW="$bw" TIERS="$bw:1" READS="$reads" KMEANS="$km" NO_SELECTION_LOG=1 \
      CLIO_REPLAY_INFLIGHT=1 READ_INFLIGHT=1 CLIO_REPLAY_DROP_CACHES=1 RUN_TAG="$tag" \
      ./run_v2_workloads.sh "$ds" "$mode" >> "$LOG" 2>&1
  local rc=$?
  restore_gov
  clean "$run"
  local drops want got
  drops=$(grep -c "page cache dropped" "$run/stdout.log" 2>/dev/null)
  want=$(find "/mnt/nvme0/v2-work/$ds/fields" -name '*.chunk' | wc -l)
  got=$(grep -o -m1 "VERIFIED: [0-9]* of [0-9]*" "$run/stdout.log" 2>/dev/null | awk '{print $4}')
  if [ "$rc" -eq 0 ] && complete "$run" && [ "${drops:-0}" -ge "$reads" ] && [ "${got:-0}" -eq "$want" ]; then
    mkdir -p "$(dirname "$dest")"
    rm -rf "$dest"
    mv "$run" "$dest"
    say "DONE $wl $mode $tag ($got of $want chunks bit-exact, page cache dropped $drops times) -> $dest"
  else
    local keep
    keep=$GT_DIR/failed/$(basename "$run")_$(date +%m%d-%H%M%S)
    mkdir -p "$GT_DIR/failed"
    mv "$run" "$keep"
    say "FAILED $wl $mode $tag (rc $rc, ${got:-0} of $want chunks verified, page cache dropped ${drops:-0} times); logs: $keep"
    return 1
  fi
}

# run_one, and once more when it fails
run_retry() {
  run_one "$@" && return 0
  say "RETRY $1 $2 $3"
  run_one "$@"
}

say "ground truth: phases${PHASES}, $READS timed reads per setting run, results in $GT_DIR"
if [[ "$PHASES" == *" 1 "* ]]; then
  for wl in WarpX incflo VPIC Nyx; do
    selected "$wl" || continue
    run_retry "$wl" exhaustive gtexh "$GT_DIR/${wl,,}/exhaustive" 1
  done
fi
if [[ "$PHASES" == *" 2 "* ]]; then
  [ -f "$GT_DIR/plan.txt" ] || "$PY" ground_truth.py plan --reads "$READS" >> "$LOG" 2>&1
  # the plan on its own descriptor: nothing in a run may read the loop's lines
  while read -r wl _ s name _ _ <&3; do
    selected "$wl" || continue
    run_retry "$wl" fixed "gt${READS}s$s" "$GT_DIR/${wl,,}/s${s}_$name" "$READS" FIXED_SETTING="$s"
  done 3< "$GT_DIR/plan.txt"
fi
if [[ "$PHASES" == *" 3 "* ]] && [ -z "${DRY_RUN:-}" ]; then
  wait_gpu
  "$PY" ground_truth.py collect >> "$LOG" 2>&1 || say "collect FAILED"
  left=$(find "$GT_DIR" "$RUNS" \( -name chi_bdev.dat -o -name 'cte_tier.dat*' \) 2>/dev/null)
  if [ -z "$left" ]; then say "check: no stored compressed data left"; else say "WARNING: stored data left: $left"; fi
  say "GROUND_TRUTH_DONE"
fi
