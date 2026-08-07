#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J gnnhold
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --mem=224g
#SBATCH -t 48:00:00
#SBATCH -o /u/rpawar/gnnbench/logs/holder-%j.out
echo "HOLDER on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /tmp | tail -1
mkdir -p /tmp/gnn
sleep 48h
