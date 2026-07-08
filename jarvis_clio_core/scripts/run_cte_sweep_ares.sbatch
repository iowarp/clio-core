#!/bin/bash
# Queue the single-node CLIO Core (CTE) performance sweep on Ares.
#
# Wraps `jarvis ppl run yaml .../ares/cte_sweep_1n.yaml` in a SLURM batch job:
# grabs one compute node, activates the iowarp conda env, (re)registers the
# repo, and runs the 48-run sweep (io_size {4k,16k,64k,128k} x threads {1,2,4,8}
# x 3 repeats, 60 s each, Put). Output lands in $HOME/cte_sweep_results/results.csv.
#
# Quick queue:
#   sbatch jarvis_clio_core/scripts/run_cte_sweep_ares.sbatch
#
# Override any path/env at submit time, e.g.:
#   sbatch --export=ALL,CONDA_ENV=iowarp,REPO=$HOME/clio-core-fork \
#          jarvis_clio_core/scripts/run_cte_sweep_ares.sbatch
#
# Watch it:   squeue -u $USER ; tail -f cte_sweep-<jobid>.out
#
#SBATCH --job-name=cte_sweep_1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# ConstrainCores=yes on Ares hands an ntasks=1 job a single core unless we ask
# for more; chimaera's 4 workers + up to 8 bench threads need real cores.
#SBATCH --cpus-per-task=40
#SBATCH --partition=compute
# 48 runs x (~60 s + runtime/CTE setup) ~= 1-1.5 h; 4 h is comfortable headroom.
#SBATCH --time=04:00:00
#SBATCH --output=cte_sweep-%j.out
#SBATCH --error=cte_sweep-%j.err

set -euo pipefail

# ---- knobs (override via --export=ALL,KEY=VAL) ----------------------------
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-iowarp}"
REPO="${REPO:-$HOME/clio-core-fork}"
PIPELINE="${PIPELINE:-$REPO/jarvis_clio_core/pipelines/ares/cte_sweep_1n.yaml}"
RESULTS="${RESULTS:-$HOME/cte_sweep_results/results.csv}"

echo "=== CTE sweep on $(hostname) — job ${SLURM_JOB_ID:-local} ==="
echo "conda: $CONDA_ENV ($CONDA_SH)"
echo "repo:  $REPO"
echo "yaml:  $PIPELINE"

# ---- activate the IOWarp/Jarvis conda env --------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# ---- register the repo with jarvis (idempotent) --------------------------
jarvis repo add "$REPO/jarvis_clio_core" 2>/dev/null || true

# ---- stale-state guard: free the runtime port from any prior run ---------
pkill -f 'clio_run' 2>/dev/null || true

# ---- run the 48-run sweep (runner auto-detects vars:+loop:) ---------------
jarvis ppl run yaml "$PIPELINE"

# ---- report ---------------------------------------------------------------
echo "=== done — results at $RESULTS ==="
if [ -f "$RESULTS" ]; then
  n=$(grep -c success "$RESULTS" 2>/dev/null || echo 0)
  echo "successful rows: $n / 48"
  head -1 "$RESULTS"
  grep success "$RESULTS" | head -3
  [ "$n" -ge 48 ] && echo "ALL 48 GREEN" || echo "INCOMPLETE — inspect $RESULTS / the .err log"
else
  echo "NO results.csv produced — check cte_sweep-${SLURM_JOB_ID:-local}.err"
fi
