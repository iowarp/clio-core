#!/bin/bash
#SBATCH --job-name=lz4full
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=96g
#SBATCH --time=00:15:00
#SBATCH --output=/u/rpawar/sycl_nvcomp/compare/%x-%j.out
set -uo pipefail
SIF=/u/rpawar/containers/deps-nvidia.sif
C=/u/rpawar/sycl_nvcomp/compare
KRN=/u/rpawar/eternia-sycl-nvcomp/context-transport-primitives/src/compress/sycl_lz4_kernels.cc
apptainer exec --nv --bind /u/rpawar "$SIF" bash -lc '
  echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
  DPCPP=/opt/intel/dpcpp/bin/clang++
  echo "== build SYCL bench (host2host + GPU-only) =="
  $DPCPP -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
    -Xsycl-target-backend --cuda-gpu-arch=sm_80 \
    '"$KRN"' '"$C"'/sycl_file_bench.cc -o /tmp/sfb 2>&1 | grep -aiE "error" | head
  . /u/rpawar/nvcomp-venv/bin/activate
  for f in '"$C"'/data_32_hi.bin '"$C"'/data_64_mid.bin '"$C"'/data_256.bin '"$C"'/data_1024.bin; do
    mb=$(( $(stat -c%s "$f") / 1048576 ))
    reps=10; [ $mb -ge 256 ] && reps=5; [ $mb -ge 1024 ] && reps=3
    echo "==================== $(basename $f) (${mb}MB, reps=$reps) ===================="
    /tmp/sfb "$f" $reps
    python3 '"$C"'/nvcomp_bench.py "$f"
  done
' 2>&1 | grep -viE "WARNING: |INFO: |LOADER|Selected device|platform:|device:"
echo "=== LZ4FULL_DONE ==="
