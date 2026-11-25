#!/bin/bash
#SBATCH -A bekn-delta-cpu
#SBATCH -p cpu
#SBATCH -J ssc_build
#SBATCH -N 1 -n 1 -c 64
#SBATCH --mem=120g
#SBATCH -t 02:00:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/build-%j.out
JOBS=64 apptainer exec -B /work,/tmp,/u /u/rpawar/containers/deps-nvidia.sif bash /u/rpawar/clio-core-ssc/ssc_work/build_clio.sh
