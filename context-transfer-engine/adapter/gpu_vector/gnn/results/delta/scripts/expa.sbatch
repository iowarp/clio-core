#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4-interactive
#SBATCH -J gnn_expa
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH -t 01:00:00
#SBATCH -o /u/rpawar/gnnbench/logs/expa-%j.out
echo "EXP-A on $(hostname) at $(date)"
apptainer exec --nv -B /work,/tmp,/u /u/rpawar/containers/deps-nvidia.sif \
  bash /u/rpawar/gnnbench/run_exp.sh a
echo "EXIT=$?"
