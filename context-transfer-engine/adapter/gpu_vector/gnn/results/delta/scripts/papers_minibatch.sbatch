#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J gnnpmb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --mem=224g
#SBATCH -t 05:00:00
#SBATCH -o /u/rpawar/gnnbench/logs/papers-minibatch-%j.out
#
# Unattended: re-prep ogbn-papers100M on node-local NVMe (the previous copy died
# with its allocation; the 60.3 GB raw zip survives on /work), then train it with
# MINI-BATCH SGD so the papers100M headline stops reporting the underfit
# majority-class-floor accuracy.
#
# Budget (measured on the earlier run): prep ~45 min, Eternia store ~159 s,
# ~278 s/epoch. page_rows=65536 window=2 -> 847 updates/epoch, so 10 epochs is
# already 8470 updates (arxiv needed ~1230 to reach 0.586 from 0.188).
# Ordering is deliberate: the 10-epoch run lands early, the 30-epoch matched
# control (directly comparable to the existing full-batch 30-epoch run) only if
# time remains.
set -uo pipefail
echo "HOST $(hostname)  START $(date)"

SIF=/u/rpawar/containers/deps-nvidia.sif
BIN=/u/rpawar/clio-core/build/bin
PAPERS=/tmp/gnn/data/papers100M
RES=/u/rpawar/gnnbench/results
CSV="$RES/gnn_papers_minibatch.csv"
DRAM_MIB=140000
mkdir -p "$RES"
APP="apptainer exec --nv -B /work,/tmp,/u,/projects $SIF"

echo "########## 1) PREP (expect ~45 min) ##########"
t0=$SECONDS
apptainer exec -B /work,/tmp,/u,/projects "$SIF" \
  bash /u/rpawar/gnnbench/prep_papers.sh 2>&1 | tail -25
echo "[time] prep took $((SECONDS-t0)) s"
if [ ! -s "$PAPERS/agg_features.f32" ]; then
  echo "!! PREP FAILED - agg_features.f32 missing; aborting"; exit 1
fi
cat "$PAPERS/meta.txt"

run() {   # $1=mb $2=epochs $3=port $4=label
  echo "########## $4 ##########"
  t=$SECONDS
  $APP bash -c "
    cd $BIN || exit 1
    unset CC CXX
    export LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1 CLIO_PORT=$3
    CLIO_CTE_COMPRESS_LIB=zstd CLIO_CTE_COMPRESS_PRESET=balanced \
    CLIO_GNN_DATA=$PAPERS CLIO_GNN_TRAIN_NODES=0 \
    CLIO_GNN_EPOCHS=$2 CLIO_GNN_LR=0.2 CLIO_GNN_HIDDEN=64 \
    CLIO_GNN_MINIBATCH=$1 CLIO_GNN_PAGE_ROWS=65536 CLIO_GNN_WINDOW=2 \
    CLIO_GNN_DRAM_MIB=$DRAM_MIB CLIO_GNN_HBM_MIB=2048 \
    CLIO_GNN_CSV=$CSV \
      ./test_gpu_vector_gnn_train 2>&1 \
      | grep -aE '^\[TRAIN\] (A=|IN-CORE|ETERNIA: (stored|epoch0)|in-core OOM)|eternia e|VERIFY|Passed:|Failed:'
  "
  echo "[time] $4 took $((SECONDS-t)) s"
  echo
}

# 10 epochs = ~8470 updates. Lands ~50 min after prep.
run 1 10 10911 "PAPERS100M MINI-BATCH 10 epochs (~8470 updates)"
echo "===== CSV so far ====="; cat "$CSV" 2>/dev/null

# Matched control: same 30 epochs as the committed full-batch run (acc 0.0626).
# Only worth starting if >2.6 h of walltime remains.
LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null)
echo "[sched] walltime left: $LEFT"
run 1 30 10912 "PAPERS100M MINI-BATCH 30 epochs (matched to full-batch control)"

echo "===== FINAL CSV ====="; cat "$CSV" 2>/dev/null
echo "DONE $(date)"
