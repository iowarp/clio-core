#!/usr/bin/env bash
#SBATCH --job-name=phase3
#SBATCH --account=CHANGEME   # override on the command line: sbatch -A <account>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#SBATCH --output=$WORK/phase3/%x-%j.out
#
# PHASE 3 -- the state-of-the-art baselines, on the chunks Phase 1 swept.
#
# Phase 1 and 1b are both confined to NeuroPress's 32-action nvcomp space,
# because kNeuroPressTrainedGpuBaseIds filters cusz/ndzip/cuszp out of the
# candidate set before ranking. Those three are therefore not "actions the
# model got wrong" -- they are outside the space entirely, and the model has
# no output slot for them. Upstream is identical: its own trace campaign
# measures ndzip as the per-chunk optimum on 15.6% of VPIC and 14.4% of Nyx
# chunks while NeuroPress selects it 0.0% of the time, by construction
# (NeuroPress/results.md:86).
#
# So this asks a question about the ACTION SPACE, not about the selector:
# what is NeuroPress giving up by not being able to reach these at all?
#
# Same chunks, same 8 MiB chunking, same eb 1e-3 as Phase 1 -- cusz and cuszp
# take that bound from preset BALANCED, ndzip is lossless and clears it
# trivially. Fidelity is VERIFIED, not assumed: the first cuSZ run in this
# campaign reported 9.387x with its bound never checked, and when checked it
# had exceeded eb on 118 of 120 chunks.
set -uo pipefail
. "$(dirname "$0")/env.sh"
H=$(dirname "$0")
R=$WORK/phase3/${WL}; mkdir -p "$R"
CHUNKS=${CHUNKS:-300}
CODECS=${CODECS:-"cusz cuszp ndzip"}

case "$WL" in
  nyx)   FIELDS=$NYX_FIELDS ;;
  vpic)  FIELDS=$VPIC_FIELDS ;;
  warpx) FIELDS=$WARPX_FIELDS ;;
  lammps) FIELDS=insitu ;;
  *) echo "unknown WL=$WL" >&2; exit 2 ;;
esac

echo "=== $(hostname) / $(nvidia-smi -L | head -1) ==="
echo "PHASE 3 baselines: WL=$WL chunks=$CHUNKS eb=1e-3  codecs: $CODECS"
echo "libso: $(stat -c %y $BUILD/bin/libclio_cte_compressor_runtime.so)"

for C in $CODECS; do
  echo
  echo "---- codec=$C ----"
  s=$(date +%s.%N)
  # LAMMPS is in-situ: it drives the solver rather than replaying .f32 files,
  # so it goes through its own runner with the codec pinned by STATIC_LIB.
  # That path has no --verify/--check-bound, so its fidelity is UNPROVEN and
  # the analyzer reports it as such rather than ranking it silently.
  if [ "$WL" = lammps ]; then
    env STATIC_LIB="$C" STATIC_SHUF=0 NP_LEARN=false NP_EXPLORE=false \
        EXPLORE_K=0 BEST=false \
        "$H/run_learn_lammps.sh" 100 $((CHUNKS+200)) 20 8388608 1e-3 0.10 \
        "$R/$C" 2>&1 | tail -14
  else
    "$H/run_static.sh" "$C" "$FIELDS" "$CHUNKS" 8388608 1e-3 "$R/$C" 2>&1 | tail -14
  fi
  echo "rc=${PIPESTATUS[0]} wall=$(awk -v a=$s -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')s"
  rm -f "$R/$C/chi_bdev.dat" "$R/$C"/cte_tier.dat*
done

echo
echo "=== baselines vs the action space ==="
python3 "$(dirname "$0")/analyze/phase3_analyze.py" "$R" "$WL"
