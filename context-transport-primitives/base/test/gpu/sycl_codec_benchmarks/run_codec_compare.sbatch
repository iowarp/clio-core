#!/bin/bash
#SBATCH --job-name=codeccmp
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:15:00
#SBATCH --output=/u/rpawar/sycl_nvcomp/compare/%x-%j.out
set -uo pipefail
SIF=/u/rpawar/containers/deps-nvidia.sif; C=/u/rpawar/sycl_nvcomp/compare
K=/u/rpawar/eternia-sycl-nvcomp/context-transport-primitives/src/compress
apptainer exec --nv --bind /u/rpawar "$SIF" bash -lc '
  echo "== GPU =="; nvidia-smi --query-gpu=name --format=csv,noheader | head -1
  D=/opt/intel/dpcpp/bin/clang++; F="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80"
  $D -std=c++17 -O3 $F '"$K"'/sycl_snappy_kernels.cc '"$C"'/sycl_snappy_file_bench.cc -o /tmp/sfb_snappy 2>&1 | grep -aiE error | head
  $D -std=c++17 -O3 $F '"$K"'/sycl_ans_kernels.cc '"$C"'/sycl_ans_file_bench.cc -o /tmp/sfb_ans 2>&1 | grep -aiE error | head
  . /u/rpawar/nvcomp-venv/bin/activate
  echo "############ SNAPPY: sycl vs nvcomp ############"
  for f in '"$C"'/data_32_hi.bin '"$C"'/data_64_mid.bin; do
    echo "==== $(basename $f) ===="; /tmp/sfb_snappy "$f" 10; python3 '"$C"'/nvcomp_bench.py Snappy "$f"; done
  echo "############ ANS: sycl vs nvcomp ############"
  for f in '"$C"'/data_skew32.bin '"$C"'/data_32_hi.bin; do
    echo "==== $(basename $f) ===="; /tmp/sfb_ans "$f" 10; python3 '"$C"'/nvcomp_bench.py ANS "$f"; done
' 2>&1 | grep -viE "WARNING: |INFO: |LOADER|Selected device|platform:|device:"
echo "=== CODECCMP_DONE ==="
