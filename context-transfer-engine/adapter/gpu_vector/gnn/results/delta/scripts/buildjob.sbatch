#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4-interactive
#SBATCH -J gnnbuild
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH --mem=48g
#SBATCH -t 01:00:00
#SBATCH -o /u/rpawar/gnnbench/logs/build-%j.out
echo "BUILD on $(hostname) at $(date)"
echo "host driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"
echo "=============== CUDA sanity inside container ==============="
apptainer exec --nv -B /work,/tmp,/u /u/rpawar/containers/deps-nvidia.sif bash -c '
  unset CC CXX; export PATH=/usr/local/cuda-12.6/bin:$PATH
  nvcc -arch=sm_80 -o /tmp/cudatest /u/rpawar/gnnbench/cudatest.cu && /tmp/cudatest'
RC=$?
echo "CUDA SANITY RC=$RC"
if [ $RC -ne 0 ]; then echo "ABORTING: CUDA not functional in container"; exit 1; fi
echo "=============== BUILD ==============="
JOBS=16 apptainer exec --nv -B /work,/tmp,/u /u/rpawar/containers/deps-nvidia.sif \
  bash /u/rpawar/gnnbench/build.sh
echo "EXIT=$?"
