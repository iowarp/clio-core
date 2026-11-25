#!/bin/bash
#SBATCH -A bekn-delta-cpu
#SBATCH -p cpu
#SBATCH -J ssc_tests
#SBATCH -N 1 -n 1 -c 16 --mem=48g
#SBATCH -t 01:00:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/tests-%j.out
cd /u/rpawar/clio-core-ssc/build
apptainer exec -B /work,/tmp,/u /u/rpawar/containers/deps-nvidia.sif \
  ctest -R "cache|replic|indexer|compressor" --timeout 600 --output-on-failure 2>&1 | tail -80
