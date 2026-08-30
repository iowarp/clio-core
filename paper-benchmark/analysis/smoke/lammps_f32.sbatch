#!/usr/bin/env bash
#SBATCH --job-name=np-lammps-f32
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:05:00
#SBATCH --output=/projects/bekn/imuradli/np-lammps-f32/slurm-%j.out
#
# The LAMMPS smoke cells, at run_benchmark.sh's `quick` profile:
#   --box 40 (4*40^3 = 256,000 atoms) --steps 200 --gap 50 --chunk 4 MiB
#
# Back inside a FIVE MINUTE allocation, and this time it fits: LAMMPS is the
# cheap workload here. 256k atoms x 3 arrays (x/v/f) x 3 doubles is ~17.6 MiB
# per frame and 200/50+1 = 5 frames, so a cell moves ~88 MiB -- two orders of
# magnitude less than a VPIC cell, whose 5 GiB is what forced 25 minutes there.
#
# THE SAME WORKLOAD IN FLOAT32. Identical physics, identical atoms, identical
# trajectory -- LAMMPS still computes in double, and --f32 downcasts the
# gathered x/v/f to float32 on the way to the compressor. Element width is the
# ONLY thing that differs from the lammps_ cells, which makes this a controlled
# test of the confound the three workloads were chosen to break: nyx is float32
# and compresses 24-99x, vpic is float32 and compresses ~1.1x, lammps is
# float64 and compresses ~1.1x. If width mattered on its own, halving it here
# should move the ratio.
#
# NOTHING LEAVES THE GPU. --f32 used to be host-gather-only: it downcast
# during lammps_gather_atoms and staged H2D, which put the payload back across
# PCIe. The gather kernel is now generic in its destination element type
# (clio_lmp_device_gather_id_window_f32), so the narrowing happens on the
# device in the same store that was already writing the element. --f32 is
# therefore paired with --require-device here, and the run reads the same atom
# arrays, writes half the bytes and never crosses PCIe -- so unlike the host
# route, these TIMINGS ARE comparable with the float64 cells.
#
# W_VERIFY=inline for this workload: the driver always verifies and takes no
# flag, unlike nyx (--verify) and vpic (--verify / --check-bound).
set -uo pipefail

REPO=/u/imuradli/clio-core
PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
# One provider for the whole krb5 family; see vpic_evolving.sbatch for why
# RUNPATH alone is not enough under the Cray MPI stack.
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

ROOT=${LMP_ROOT:-/projects/bekn/imuradli/np-lammps-f32}
RESULTS=$ROOT/results
STATUS=$ROOT/status.csv
mkdir -p "$RESULTS"
[ -f "$STATUS" ] || echo "workload,cell,state,detail" > "$STATUS"

BOX=40 STEPS=200 GAP=50 CHUNK=4194304
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
  case "$l" in
    *:*:*) echo $(( 10#${l%%:*} * 3600 + 10#$(echo "$l" | cut -d: -f2) * 60 + 10#$(echo "$l" | cut -d: -f3) )) ;;
    *:*)   echo $(( 10#${l%%:*} * 60 + 10#${l##*:} )) ;;
    *)     echo $(( 300 - ($(date +%s) - T0) )) ;;
  esac
}

say "node=$(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | sed 's/^/       gpu: /'
say "LAMMPS LJ melt: box=$BOX ($(( 4 * BOX * BOX * BOX )) atoms), $STEPS steps, gap $GAP, chunk $CHUNK"
echo

[ -x "$BUILD/bin/neuropress_lammps_lib" ] || { say "driver missing"; exit 1; }

for mode in lossless lossy; do
  for model in $COST_MODELS; do
    tag="lammps32_${mode}_${model}_${BWNAME}"
    if [ -f "$RESULTS/$tag/meta.json" ] &&
       grep -q '"rc":0' "$RESULTS/$tag/meta.json" 2>/dev/null; then
      say "-- $tag already complete, skipping"; continue
    fi
    if [ "$(left)" -lt 45 ]; then
      say "-- $(left) s left, not enough for $tag"
      record lammps "$tag" deferred "out of allocation"; break 2
    fi
    eb=(); [ "$mode" = lossy ] && eb=(--eb "$EB")
    say "== $tag  ($(left) s left)"
    "$PB/lammps/run_config.sh" "explore-$model" \
        --box "$BOX" --steps "$STEPS" --gap "$GAP" --f32 --require-device \
        --chunk "$CHUNK" --bw "$BW" "${eb[@]}" \
        --explore-k "$K" --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
    rc=${PIPESTATUS[0]}
    rm -f "$RESULTS/$tag/chi_bdev.dat" "$RESULTS/$tag"/cte_tier.dat*
    if [ "$rc" -eq 0 ]; then
      record lammps "$tag" ok "$(python3 -c "import json;print(json.load(open('$RESULTS/$tag/meta.json'))['wall_s'],'s')" 2>/dev/null)"
    else
      record lammps "$tag" FAILED "rc=$rc"
    fi
    echo
  done
done

say "== status"; column -s, -t "$STATUS" 2>/dev/null || cat "$STATUS"
echo
[ -x "$PB/collect.py" ] && { say "== collect"; "$PB/collect.py" "$RESULTS" 2>&1 | tail -25; }
