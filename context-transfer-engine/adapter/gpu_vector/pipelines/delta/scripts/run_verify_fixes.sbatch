#!/bin/bash
#SBATCH --job-name=vfix
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=8 --mem=200g
#SBATCH --time=00:45:00
#SBATCH --output=/u/rpawar/gsbench/%x-%j.out
set -uo pipefail
# Verify the two workloads the pulled commits specifically fix, on the rebuilt (80d397c75) binaries:
#   weights -> "addressed pool 513.0 which is never created" (tip commit)
#   lbann   -> "serves durably stale replicas" + FlushData "destroyed every blob it could not place"
SIF=/u/rpawar/containers/deps-nvidia.sif; W=/u/rpawar/gsbench/bench_env_wrapper.sh
CSV=/u/rpawar/eternia_logs/verify_fixes_1n.csv
echo "workload,blocks,bench_ms,peak_gpu_mb,gates" > "$CSV"
peak4(){ for i in $(seq 1 $1); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -n|tail -1; sleep 0.5; done|sort -n|tail -1; }
L="srun --mpi=pmix -n 4 apptainer exec --nv --bind /u/rpawar $SIF $W"

# WEIGHTS: data-parallel, per-GPU 32GB at 4 ranks => data_mb=128000. Sweep blocks 20-80%.
echo "===== weights blocks sweep @data_mb=128000 (per-GPU 32GB) ====="
for BL in 22 43 65 86; do
  ($L bash -c "timeout 300 clio_weights_mpi_bench --data-mb 128000 --passes 3 --blocks $BL" >/tmp/wt_$BL.log 2>&1) & pid=$!
  pk=$(peak4 80); wait $pid; rc=$?
  ms=$(grep -aoE "[0-9.]+ ms" /tmp/wt_$BL.log|grep -aoE "[0-9.]+"|tail -1)
  ok=$(grep -c "WEIGHTS MPI: ALL GATES PASS" /tmp/wt_$BL.log)
  err=$(grep -aoE "pool 513|out of memory|CUDA error|GATE FAILURE" /tmp/wt_$BL.log|head -1)
  echo "weights,$BL,$ms,$pk,$ok" >> "$CSV"; echo "  weights blk=$BL peak=${pk}MB gates=$ok ms=$ms rc=$rc $err"
done

# LBANN: model-parallel MLP, hidden=3M ~ 32GB at 4 ranks, timeout-guarded, lr=1e-5.
echo "===== lbann blocks sweep @hidden=3000000 (timeout-guarded) ====="
for BL in 16 32 64 86; do
  ($L bash -c "timeout 300 clio_lbann_mpi_bench --in 4096 --hidden 3000000 --out 64 --batch 256 --steps 2 --lr 0.00001 --blocks $BL" >/tmp/lb_$BL.log 2>&1) & pid=$!
  pk=$(peak4 90); wait $pid; rc=$?
  ms=$(grep -aoE "[0-9.]+ *ms/step" /tmp/lb_$BL.log|grep -aoE "[0-9.]+"|tail -1)
  ok=$(grep -c "ALL GATES PASS" /tmp/lb_$BL.log)
  err=$(grep -aoE "stale|out of memory|timeout|CUDA error" /tmp/lb_$BL.log|head -1)
  echo "lbann,$BL,$ms,$pk,$ok" >> "$CSV"; echo "  lbann blk=$BL peak=${pk}MB gates=$ok ms=$ms rc=$rc $err"
done
echo "=== CSV ==="; cat "$CSV"; echo "=== VFIX_DONE ==="
