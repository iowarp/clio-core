#!/usr/bin/env bash
#SBATCH --job-name=phase1b
#SBATCH --account=CHANGEME   # override on the command line: sbatch -A <account>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#SBATCH --output=$WORK/phase1b/%x-%j.out
#
# PHASE 1b -- does ADAPTATION change the picture Phase 1 measured?
#
# Phase 1 scored a FROZEN model: best_mode suppresses both SGD phases, so the
# 38.7%/33.7%/6.0%/1.8% exact-match rates and the "every workload has a fixed
# action that beats NeuroPress" result describe weights that never moved.
# This runs the same 300 chunks with the SGD path live.
#
# NO SWEEP, and that is the whole point. Exploration is NOT merely a source of
# training samples: compressor_runtime.cc:2673 adopts the cheapest measured
# action over the primary on `if (winner.have)`, with no best_mode guard. Any
# explore-enabled arm therefore STORES the oracle's bytes, and its stored ratio
# measures the sweep, not the selector. With EXPLORE_K=0 what lands on the tier
# is the model's own pick, which is the only thing worth comparing to a fixed
# baseline.
#
# Two arms on identical chunks, differing in one bit:
#   adaptive  NP_LEARN=true   -- SGD phase 1 live, gated at MAPE 0.10
#   frozen    NP_LEARN=false  -- no SGD at all; the Phase 1 selector, no sweep
#
# Weights are in-memory only (compressor_runtime.cc:1861); model.nnwt on disk
# is never written. Ratio cost model, matching Phase 1 exactly.
set -uo pipefail
. "$(dirname "$0")/env.sh"
H=$(dirname "$0")
R=$WORK/phase1b/${WL}${TAG:-}; mkdir -p "$R"   # TAG isolates a re-run from the published results
CHUNKS=${CHUNKS:-300}
export W_CT=${W_CT:-0} W_DT=${W_DT:-0} W_IO=${W_IO:-1}   # overridable: CC2 scores a TIME-inclusive cost

case "$WL" in
  nyx)    FIELDS=$NYX_FIELDS ;;
  vpic)   FIELDS=$VPIC_FIELDS ;;
  warpx)  FIELDS=$WARPX_FIELDS ;;
  lammps) FIELDS=insitu ;;
  *) echo "unknown WL=$WL" >&2; exit 2 ;;
esac

echo "=== $(hostname) / $(nvidia-smi -L | head -1) ==="
echo "PHASE 1b: WL=$WL chunks=$CHUNKS  no sweep (EXPLORE_K=0)  ratio cost  MAPE 0.10"
echo "libso: $(stat -c %y $BUILD/bin/libclio_cte_compressor_runtime.so)"

for ARM in adaptive frozen; do
  case "$ARM" in
    adaptive) LEARN=true  ;;
    frozen)   LEARN=false ;;
  esac
  echo
  echo "---- arm=$ARM  NP_LEARN=$LEARN ----"
  s=$(date +%s.%N)
  COMMON=(NP_LEARN=$LEARN NP_EXPLORE=false EXPLORE_K=0 BEST=false
          CTP_LOG_LEVEL=info)
  if [ "$WL" = lammps ]; then
    env "${COMMON[@]}" "$H/run_learn_lammps.sh" 100 $((CHUNKS+200)) 20 \
        8388608 1e-3 0.10 "$R/$ARM" 2>&1 | tail -8
  else
    env "${COMMON[@]}" "$H/run_learn.sh" "$FIELDS" "$CHUNKS" \
        8388608 1e-3 0.10 "$R/$ARM" 2>&1 | tail -8
  fi
  echo "rc=${PIPESTATUS[0]} wall=$(awk -v a=$s -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')s"
  rm -f "$R/$ARM/chi_bdev.dat" "$R/$ARM"/cte_tier.dat*
done

echo
echo "=== adaptive vs frozen ==="
python3 "$(dirname "$0")/analyze/phase1b_analyze.py" "$R" "$WL"
