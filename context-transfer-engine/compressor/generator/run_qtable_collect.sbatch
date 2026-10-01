#!/bin/bash
# Multi-node qtable data collector. One sbatch per workflow; N nodes
# inside the job, each running 1 task with 24 threads. Tasks split the
# (chunk × compressor) units round-robin via SLURM_PROCID/NPROCS.
#
#   sbatch --nodes=4 --export=WORKFLOW=montage,INPUT_DIR=/home/llogan/jarvis-runs/montage \
#          run_qtable_collect.sbatch
#
# Env overrides:
#   WORKFLOW             logical workflow id (also CSV filename stem)
#   INPUT_DIR            workflow output directory on NFS
#   OUTPUT_DIR           CSV destination          (default $HOME/qtable_csvs)
#   MAX_CHUNKS_PER_FILE  per-(file, chunk_size)   (default 200)
#   WORKERS              threads per node         (default $SLURM_CPUS_PER_TASK)
#   VENV                 python venv              (default $HOME/venv)
#
#SBATCH --job-name=qtable_collect
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --output=qtable_collect-%x-%j.out
#SBATCH --error=qtable_collect-%x-%j.err

set -euo pipefail

WORKFLOW="${WORKFLOW:?set WORKFLOW=<id>}"
INPUT_DIR="${INPUT_DIR:?set INPUT_DIR=<path>}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/qtable_csvs}"
MAX_CHUNKS_PER_FILE="${MAX_CHUNKS_PER_FILE:-200}"
WORKERS="${WORKERS:-$SLURM_CPUS_PER_TASK}"
VENV="${VENV:-$HOME/venv}"
FEATURES_ONLY="${FEATURES_ONLY:-0}"
FEATURES_FLAG=""
if [ "$FEATURES_ONLY" = "1" ]; then FEATURES_FLAG="--features-only"; fi

mkdir -p "$OUTPUT_DIR"
OUT_CSV="$OUTPUT_DIR/${WORKFLOW}.csv"

# readlink -f on $0 inside sbatch resolves to /var/spool/slurmd/... — the
# slurm-copied script, not the source. Use the canonical repo path
# (overridable via SCRIPT env if you check out elsewhere).
SCRIPT="${SCRIPT:-/mnt/common/llogan/iowarp/core/context-transfer-engine/compressor/generator/qtable_data_collect.py}"
if [ ! -f "$SCRIPT" ]; then
    echo "FATAL: script not found at $SCRIPT" >&2
    exit 2
fi

echo "workflow:   $WORKFLOW"
echo "input dir:  $INPUT_DIR"
echo "output csv: $OUT_CSV"
echo "nodes:      $SLURM_JOB_NUM_NODES"
echo "tasks:      $SLURM_NTASKS"
echo "workers/node: $WORKERS"
echo "total cores: $(( SLURM_NTASKS * WORKERS ))"
echo "max-chunks-per-file: $MAX_CHUNKS_PER_FILE"
echo

# blosc2's internal thread pool fights ours; pin it.
# spack libpressio if present so SZ3/FPZIP are available.
SETUP="ulimit -n 65536 ; export BLOSC_NTHREADS=1 ;
  if [ -f \$HOME/spack/share/spack/setup-env.sh ]; then
    . \$HOME/spack/share/spack/setup-env.sh ;
    spack load libpressio 2>/dev/null || true ;
  fi"

# Launch one task per node; SLURM_PROCID/NPROCS partition the work.
srun --mpi=none --ntasks=$SLURM_NTASKS --ntasks-per-node=1 \
    --cpu-bind=none --kill-on-bad-exit=1 \
    /bin/bash -c "$SETUP ; exec '$VENV/bin/python3' -u '$SCRIPT' \
        --workflow-id '$WORKFLOW' \
        --input-dir '$INPUT_DIR' \
        --output-csv '$OUT_CSV' \
        --workers $WORKERS \
        --max-chunks-per-file $MAX_CHUNKS_PER_FILE \
        $FEATURES_FLAG"

# Concatenate per-rank CSVs into the workflow CSV (header from rank 0,
# bodies from all). Per-rank files live next to OUT_CSV.
echo
echo "=== concatenating per-rank CSVs ==="
RANK_GLOB="${OUT_CSV%.csv}.rank*.csv"
RANK_FILES=( $RANK_GLOB )
if [ ${#RANK_FILES[@]} -gt 0 ]; then
    head -1 "${RANK_FILES[0]}" > "$OUT_CSV"
    for f in "${RANK_FILES[@]}"; do
        tail -n +2 "$f" >> "$OUT_CSV"
    done
    rm -f "${RANK_FILES[@]}"
fi
echo "wrote $OUT_CSV"
wc -l "$OUT_CSV"
