#!/usr/bin/env bash
#SBATCH --job-name=np-smoke-all
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/bekn/imuradli/np-smoke/slurm-%j.out
#
# The smoke pass of run_benchmark.sh -- "does every path still work" -- for
# every workload the tree defines, inside a FIVE MINUTE allocation.
#
# Two things differ from `run_benchmark.sh --smoke`, both forced by what is
# built on this machine rather than by preference:
#
#   1. Nyx takes the REPLAY route. run_benchmark.sh's config_for() prefers
#      run_config_insitu.sh whenever it is executable, and that route needs
#      libclio_nyx_insitu.so (not built) plus a Nyx carrying the in-situ
#      patch (this one carries only the raw-field-dump patch). The replay
#      route needs neither, and it is the route every existing result in
#      analysis/results was produced by, so the new cells are comparable.
#
#   2. Every cell is SKIPPED IF ALREADY DONE, so the script is idempotent and
#      a second five-minute allocation finishes whatever the first ran out of
#      time for. That is why two jobs are chained rather than one long one.
#
# The other three workloads are still attempted, through the exact entry point
# run_benchmark.sh would use, so their status is MEASURED here rather than
# asserted from reading CMake files. They fail in seconds; they cost the nyx
# cells nothing.
set -uo pipefail

REPO=/u/imuradli/clio-core
PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export NYX_BIN=/u/imuradli/np-build/nyx/Exec/HydroTests/nyx_HydroTests
# The conda env supplies clio's C++ and crypto libraries, and BOTH halves of
# the krb5 family have to come from it. RUNPATH is not enough: it is not
# inherited transitively, so with only the deck's RUNPATH the loader takes
# libgssapi_krb5 and libkrb5support from conda but libkrb5.so.3 from /lib64
# (something in the Cray MPI stack asks for it first) and the run dies at
# startup with
#   symbol lookup error: /lib64/libkrb5.so.3: undefined symbol:
#   krb5int_push_fscreatecon_for, version krb5support_0_MIT
# LD_LIBRARY_PATH is searched before the default system directories, so this
# makes the whole family resolve to one provider. libstdc++ needs no help --
# the deck is linked through analysis/smoke/gxx-conda and its RUNPATH covers
# that one.
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

ROOT=${SMOKE_ROOT:-/scratch/bekn/imuradli/np-smoke}
FIELDS=$ROOT/fields/nyx
RESULTS=$ROOT/results
STATUS=$ROOT/status.csv
mkdir -p "$ROOT" "$RESULTS"
[ -f "$STATUS" ] || echo "workload,cell,state,detail" > "$STATUS"

# run_benchmark.sh --smoke == PROFILE=quick, REPEATS=1, BW_POLICY=one.
NCELL=128 STEPS=200 PLOTINT=10 CHUNK=8388608
BWNAME=nvme_5GBs BW=5e6 EB=1e-3 K=3
COST_MODELS="balance ratio"

say() { echo "[$(date +%H:%M:%S)] $*"; }
record() {
  # Drop any earlier row for this (workload, cell): the chained jobs share one
  # status file, and a cell that was blocked before it was built would
  # otherwise keep its stale row beside the new one.
  if [ -f "$STATUS" ]; then
    grep -v "^$1,$2," "$STATUS" > "$STATUS.tmp" 2>/dev/null && mv -f "$STATUS.tmp" "$STATUS"
  fi
  printf '%s,%s,%s,"%s"\n' "$1" "$2" "$3" "$4" >> "$STATUS"
}

# Seconds of allocation left, so a cell is never started that cannot finish.
# scontrol is the only thing that knows; fall back to the wall clock.
T0=$(date +%s)
left() {
  local l
  l=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null)
  case "$l" in
    # 10# on every field: squeue zero-pads, and bash reads a leading-zero
    # number as OCTAL -- "4:09" died with "value too great for base", left()
    # returned nothing, and the guard below then compared an empty string.
    *:*:*) echo $(( 10#${l%%:*} * 3600 + 10#$(echo "$l" | cut -d: -f2) * 60 + 10#$(echo "$l" | cut -d: -f3) )) ;;
    *:*)   echo $(( 10#${l%%:*} * 60 + 10#${l##*:} )) ;;
    *)     echo $(( 300 - ($(date +%s) - T0) )) ;;
  esac
}

say "node=$(hostname)  job=${SLURM_JOB_ID:-none}  results=$RESULTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | sed 's/^/       gpu: /'
echo

# ---------------------------------------------------------------------------
# The three workloads whose driver this build does not contain. Attempted, not
# assumed: each is invoked exactly as run_benchmark.sh's config_for() would.
# ---------------------------------------------------------------------------
for w in warpx lammps; do
  cfg=$PB/$w/run_config.sh
  [ -x "$PB/$w/run_config_insitu.sh" ] && cfg=$PB/$w/run_config_insitu.sh
  say "== $w: $(basename "$cfg") explore-balance (smoke cell 1 of 4)"
  err=$( "$cfg" explore-balance --bw "$BW" --explore-k "$K" --chunk "$CHUNK" \
           --results "$RESULTS" --tag "${w}_lossless_balance_${BWNAME}" 2>&1 )
  rc=$?
  echo "$err" | sed 's/^/       /' | head -12
  if [ $rc -eq 0 ]; then
    record "$w" all "ok" "ran"
  else
    why=$(echo "$err" | grep -iE 'missing|no such|skipped|not found' | head -1 | tr -d '"' | cut -c1-160)
    # warpx/run_config.sh:60 dies under `set -e` on the `ls` that looks for
    # WARPX_BIN, BEFORE common.sh can print its "missing WarpX" message -- so
    # a bare rc=2 with no output is itself the diagnosis. Say so rather than
    # recording an empty reason.
    [ -z "$why" ] && why="exited rc=$rc with no message (see notes)"
    record "$w" all "blocked" "$why"
  fi
  echo
done

# ---------------------------------------------------------------------------
# Nyx, replay route. Phase 1 writes the field dumps; phase 2 replays them once
# per cell. The dumps are the expensive half and they are held across jobs.
# ---------------------------------------------------------------------------
if [ -f "$FIELDS/gen.json" ] && [ "$(find "$FIELDS" -name '*.f32' | wc -l)" -gt 0 ]; then
  say "== nyx: field dumps already present ($(du -sh "$FIELDS" | cut -f1)), skipping gen_fields"
else
  say "== nyx: gen_fields ${NCELL}^3, $STEPS steps, dump every $PLOTINT"
  if "$PB/nyx/gen_fields.sh" --ncell "$NCELL" --steps "$STEPS" \
        --plot-int "$PLOTINT" --out "$FIELDS" 2>&1 | sed 's/^/       /'; then
    record nyx gen_fields ok "$(find "$FIELDS" -name '*.f32' | wc -l) files"
  else
    record nyx gen_fields FAILED "see $FIELDS/nyx.log"
    say "nyx field generation failed -- no cells can run"; exit 1
  fi
fi
echo

for mode in lossless lossy; do
  for model in $COST_MODELS; do
    tag="nyx_${mode}_${model}_${BWNAME}"
    if [ -f "$RESULTS/$tag/meta.json" ] &&
       grep -q '"rc":0' "$RESULTS/$tag/meta.json" 2>/dev/null; then
      say "-- $tag already complete, skipping"; continue
    fi
    # 90 s is the measured cell time plus half again. Starting a cell that
    # cannot finish leaves a half-written store the next job would skip.
    if [ "$(left)" -lt 90 ]; then
      say "-- $(left) s left, not enough for $tag -- the chained job takes it"
      record nyx "$tag" deferred "out of allocation"
      break 2
    fi
    eb=(); [ "$mode" = lossy ] && eb=(--eb "$EB")
    say "== $tag  ($(left) s left)"
    env CLIO_NEUROPRESS_STAGE_H2D=1 \
        "$PB/nyx/run_config.sh" "explore-$model" \
          --fields "$FIELDS" --bw "$BW" "${eb[@]}" \
          --explore-k "$K" --chunk "$CHUNK" \
          --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
    rc=${PIPESTATUS[0]}
    # run_benchmark.sh drops these after every cell; one per cell would
    # multiply the campaign's footprint and the measurements are in the CSVs.
    rm -f "$RESULTS/$tag/chi_bdev.dat" "$RESULTS/$tag"/cte_tier.dat*
    if [ "$rc" -eq 0 ]; then
      record nyx "$tag" ok "$(python3 -c "import json;print(json.load(open('$RESULTS/$tag/meta.json'))['wall_s'],'s')" 2>/dev/null)"
    else
      record nyx "$tag" FAILED "rc=$rc"
    fi
    echo
  done
done

# ---------------------------------------------------------------------------
# VPIC, IN-SITU route -- the one run_benchmark.sh's config_for() prefers, and
# the only one that hands the compressor device memory. Unlike nyx's replay
# there is no shared phase-1: every cell RE-RUNS the simulation, so a cell
# costs a full VPIC run (~2 min at 126^3) and only two fit in five minutes.
# The chained jobs take the rest.
# ---------------------------------------------------------------------------
VP_NCELL=126 VP_STEPS=200 VP_INT=25
if [ ! -x "$PB/vpic/weibel_clio.Linux" ]; then
  say "== vpic: no deck binary -- build it with ./build_deck.sh --insitu"
  record vpic all blocked "no weibel_clio.Linux"
else
  for mode in lossless lossy; do
    for model in $COST_MODELS; do
      tag="vpic_${mode}_${model}_${BWNAME}"
      if [ -f "$RESULTS/$tag/meta.json" ] &&
         grep -q '"rc":0' "$RESULTS/$tag/meta.json" 2>/dev/null; then
        say "-- $tag already complete, skipping"; continue
      fi
      if [ "$(left)" -lt 180 ]; then
        say "-- $(left) s left, not enough for $tag -- the chained job takes it"
        record vpic "$tag" deferred "out of allocation"
        break 2
      fi
      eb=(); vfy=()
      if [ "$mode" = lossy ]; then eb=(--eb "$EB"); else vfy=(--verify); fi
      say "== $tag  ($(left) s left)"
      "$PB/vpic/run_config_insitu.sh" "explore-$model" \
          --ncell "$VP_NCELL" --steps "$VP_STEPS" --int "$VP_INT" \
          --chunk "$CHUNK" --bw "$BW" "${eb[@]}" "${vfy[@]}" \
          --explore-k "$K" --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
      rc=${PIPESTATUS[0]}
      rm -f "$RESULTS/$tag/chi_bdev.dat" "$RESULTS/$tag"/cte_tier.dat*
      if [ "$rc" -eq 0 ]; then
        record vpic "$tag" ok "$(python3 -c "import json;print(json.load(open('$RESULTS/$tag/meta.json'))['wall_s'],'s')" 2>/dev/null)"
      else
        record vpic "$tag" FAILED "rc=$rc"
      fi
      echo
    done
  done
fi

say "== status"
column -s, -t "$STATUS" 2>/dev/null || cat "$STATUS"
echo
if [ -x "$PB/collect.py" ]; then
  say "== collect"
  "$PB/collect.py" "$RESULTS" 2>&1 | tail -25
fi
