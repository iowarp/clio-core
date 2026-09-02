#!/bin/bash
#SBATCH --job-name=pcsweep
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=200g
#SBATCH --time=00:55:00
#SBATCH --output=/u/rpawar/gsbench/%x-%j.out
set -uo pipefail
# FULL PAGE-CACHE SWEEP (the last original-spec axis), on the pulled base 80d397c75.
# Paged single-GPU kmeans, 36GB working set (>= 32GB peak spec). 2D grid:
#   page cache 4,8,16,32 GB (spec 4-32GB) x blocks 22,43,65,86 (spec 20-80%).
# slots = cache_mb/(blocks*page_kb_MB) keeps TOTAL cache constant as blocks vary,
# isolating the cache axis. Cache-descending order so reliable big-cache cells
# land first; each cell timeout-guarded. Checksum stable across cache => correct.
SIF=/u/rpawar/containers/deps-nvidia.sif; SRC=/u/rpawar/eternia-4node; CU=/usr/local/cuda-12.6
CSV=/u/rpawar/eternia_logs/pagecache_sweep.csv
echo "cache_mb,blocks,slots,peak_gpu_mb,ms,gbps,faults,evicts,checksum,rc" > "$CSV"
peak(){ for i in $(seq 1 $1); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1; sleep 0.5; done|sort -n|tail -1; }
for CMB in 32768 16384 8192 4096; do
  for BL in 22 43 65 86; do
    SL=$(( CMB / (BL*1) )); [ "$SL" -lt 1 ] && SL=1
    TAG=${CMB}_${BL}; LOG=/tmp/pc_${TAG}.log
    echo "===== cache=${CMB}MB blocks=$BL slots=$SL ====="
    apptainer exec --nv --bind /u/rpawar "$SIF" bash -lc "
      export PATH=$SRC/build-clang/bin:$CU/bin:\$PATH LD_LIBRARY_PATH=$SRC/build-clang/bin:$CU/lib64:\$LD_LIBRARY_PATH CLIO_REPO_PATH=$SRC/build-clang/bin
      timeout 400 clio_kmeans_paged_bench --data-mb 36000 --slots $SL --blocks $BL --page-kb 1024 --hbm-mb 2048 --nvme-mb 48000 --nvme-path /tmp/gvpc_${TAG}.dat --iters 2
    " >"$LOG" 2>&1 & pid=$!
    pk=$(peak 120); wait $pid; rc=$?
    line=$(grep -a "KMEANS mode=paged" "$LOG" | tail -1)
    ms=$(echo "$line"|grep -aoE "ms=[0-9.]+"|cut -d= -f2)
    gb=$(echo "$line"|grep -aoE "GB/s=[0-9.]+"|cut -d= -f2)
    fa=$(echo "$line"|grep -aoE "faults=[0-9]+"|cut -d= -f2)
    ev=$(echo "$line"|grep -aoE "evicts=[0-9]+"|cut -d= -f2)
    ck=$(echo "$line"|grep -aoE "centroid_checksum=[0-9.]+"|cut -d= -f2)
    echo "$CMB,$BL,$SL,$pk,${ms:-},${gb:-},${fa:-},${ev:-},${ck:-},$rc" >> "$CSV"
    echo "  peak=${pk}MB ms=${ms:-NA} faults=${fa:-NA} rc=$rc"
    rm -f /tmp/gvpc_${TAG}.dat 2>/dev/null
  done
done
echo "=== CSV ==="; cat "$CSV"; echo "=== PCSWEEP_DONE ==="
