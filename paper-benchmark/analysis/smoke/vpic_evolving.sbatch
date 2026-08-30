#!/usr/bin/env bash
#SBATCH --job-name=np-vpic-1000
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:25:00
#SBATCH --output=/projects/bekn/imuradli/np-evolving/slurm-%j.out
#
# VPIC at the DEFAULT EVOLVING CONFIGURATION -- README.md, "Default Evolving
# Benchmark Configuration", selected by a 1,000-timestep hyperparameter study.
#
#   VPIC_CLEAN_DIV_INT = 10        (run_config_insitu.sh's own default)
#   VPIC_VTHE / VPIC_VTHEX          upstream, never overridden anywhere
#   VPIC_NPPC = 8, --ncell 126
#   1,000 steps                     <-- the ONE thing the smoke pass got wrong
#
# The smoke pass ran 200 steps (run_benchmark.sh's `quick` profile). At 200 the
# run is still in the noise phase: the first blob compresses 1.006x and the
# lossless median is 1.14. At 1,000 the instability has grown -- measured on
# `cby`, amplitude goes from +/-0.045 to +/-0.169 with 0 of 24 consecutive
# dumps bit-identical.
#
# Everything else is held at the smoke pass's value, INCLUDING --int 25, so
# step count is the only variable between the two populations and they can be
# compared directly. (run_benchmark.sh's `mid` profile pairs 1,000 steps with
# --int 12, which would also change the frame count from 40 to 83 and the
# payload from 5 to 10.4 GiB per cell.)
#
# WHY NOT A FIVE-MINUTE ALLOCATION: a cell measured 39 s at 200 steps, and both
# the simulation and the compression work scale with the step count, so a cell
# is ~3.5 min at 1,000. Four of them plus startup do not fit in five minutes,
# and a five-minute allocation would defer every cell forever rather than
# finishing any. 25 minutes with the same per-cell guard is the smallest
# allocation that can actually complete the set.
#
# RESULTS GO SOMEWHERE ELSE ON PURPOSE. These are a different configuration
# from the 200-step cells; pooling the two into one results directory would let
# collect.py average across a variable under test.
set -uo pipefail

REPO=/u/imuradli/clio-core
PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
# One provider for the whole krb5 family -- RUNPATH is not inherited
# transitively, so without this libkrb5.so.3 comes from /lib64 while
# libkrb5support.so.0 comes from conda and the deck dies at startup with
# "undefined symbol: krb5int_push_fscreatecon_for".
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# /scratch is 97% full; a cell needs ~16.5 GiB peak (5.5 GiB tier + 11 GiB
# bdev, both dropped after the cell) and the project allocation has 386 GB.
ROOT=${VPIC_ROOT:-/projects/bekn/imuradli/np-evolving}
RESULTS=$ROOT/results
STATUS=$ROOT/status.csv
mkdir -p "$RESULTS"
[ -f "$STATUS" ] || echo "workload,cell,state,detail" > "$STATUS"

NCELL=126 STEPS=1000 INT=25 CHUNK=8388608
BWNAME=nvme_5GBs BW=5e6 EB=1e-3 K=3
COST_MODELS="balance ratio"

say() { echo "[$(date +%H:%M:%S)] $*"; }
record() {
  if [ -f "$STATUS" ]; then
    grep -v "^$1,$2," "$STATUS" > "$STATUS.tmp" 2>/dev/null && mv -f "$STATUS.tmp" "$STATUS"
  fi
  printf '%s,%s,%s,"%s"\n' "$1" "$2" "$3" "$4" >> "$STATUS"
}

T0=$(date +%s)
left() {
  local l
  l=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null)
  # 10# on every field: squeue zero-pads and bash reads a leading zero as octal.
  case "$l" in
    *:*:*) echo $(( 10#${l%%:*} * 3600 + 10#$(echo "$l" | cut -d: -f2) * 60 + 10#$(echo "$l" | cut -d: -f3) )) ;;
    *:*)   echo $(( 10#${l%%:*} * 60 + 10#${l##*:} )) ;;
    *)     echo $(( 1500 - ($(date +%s) - T0) )) ;;
  esac
}

say "node=$(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | sed 's/^/       gpu: /'
say "VPIC ${NCELL}^3, $STEPS steps, dump every $INT -> $(( STEPS / INT )) frames, ~5 GiB/cell"
say "results=$RESULTS   free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -d ' ')"
echo

[ -x "$PB/vpic/weibel_clio.Linux" ] || { say "no deck binary"; exit 1; }

for mode in lossless lossy; do
  for model in $COST_MODELS; do
    tag="vpic_${mode}_${model}_${BWNAME}"
    if [ -f "$RESULTS/$tag/meta.json" ] &&
       grep -q '"rc":0' "$RESULTS/$tag/meta.json" 2>/dev/null; then
      say "-- $tag already complete, skipping"; continue
    fi
    if [ "$(left)" -lt 300 ]; then
      say "-- $(left) s left, not enough for $tag"
      record vpic "$tag" deferred "out of allocation"
      break 2
    fi
    eb=(); vfy=()
    if [ "$mode" = lossy ]; then eb=(--eb "$EB"); else vfy=(--verify); fi
    say "== $tag  ($(left) s left)"
    "$PB/vpic/run_config_insitu.sh" "explore-$model" \
        --ncell "$NCELL" --steps "$STEPS" --int "$INT" \
        --chunk "$CHUNK" --bw "$BW" "${eb[@]}" "${vfy[@]}" \
        --explore-k "$K" --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
    rc=${PIPESTATUS[0]}
    # 16.5 GiB per cell; keeping them would exhaust the allocation in three.
    rm -f "$RESULTS/$tag/chi_bdev.dat" "$RESULTS/$tag"/cte_tier.dat*
    rm -rf "$RESULTS/$tag"/rank*/chi_bdev.dat "$RESULTS/$tag"/rank*/cte_tier.dat*
    if [ "$rc" -eq 0 ]; then
      record vpic "$tag" ok "$(python3 -c "import json;print(json.load(open('$RESULTS/$tag/meta.json'))['wall_s'],'s')" 2>/dev/null)"
    else
      record vpic "$tag" FAILED "rc=$rc"
    fi
    say "   free after: $(df -BG --output=avail "$ROOT" | tail -1 | tr -d ' ')"
    echo
  done
done

say "== status"
column -s, -t "$STATUS" 2>/dev/null || cat "$STATUS"
echo
[ -x "$PB/collect.py" ] && { say "== collect"; "$PB/collect.py" "$RESULTS" 2>&1 | tail -30; }
