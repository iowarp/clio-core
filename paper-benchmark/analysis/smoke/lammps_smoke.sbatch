#!/usr/bin/env bash
#SBATCH --job-name=np-lammps-smoke
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:05:00
#SBATCH --output=/projects/bekn/imuradli/np-lammps/slurm-%j.out
#
# The LAMMPS smoke cells, at run_benchmark.sh's `quick` profile:
#   --box 40 (4*40^3 = 256,000 atoms) --steps 200 --gap 50 --chunk 4 MiB
#
# Back inside a FIVE MINUTE allocation, and this time it fits: LAMMPS is the
# cheap workload here. 256k atoms x 3 arrays (x/v/f) x 3 doubles is ~17.6 MiB
# per frame and 200/50+1 = 5 frames, so a cell moves ~88 MiB -- two orders of
# magnitude less than a VPIC cell, whose 5 GiB is what forced 25 minutes there.
#
# --require-device is NOT optional and not a speed knob. It makes the driver
# gather each frame ON THE GPU through LAMMPS' Kokkos device views
# (lammps_device_view.cc) rather than through lammps_gather_atoms, so the
# pointer the compressor receives is device memory and quantization, byte
# shuffle and codec selection all take their CUDA paths. It is the whole reason
# this workload needed a Kokkos LAMMPS built from source rather than the site
# module, which ships PKG_KOKKOS=OFF and no build tree.
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

ROOT=${LMP_ROOT:-/projects/bekn/imuradli/np-lammps}
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
    tag="lammps_${mode}_${model}_${BWNAME}"
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
        --box "$BOX" --steps "$STEPS" --gap "$GAP" --require-device \
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
