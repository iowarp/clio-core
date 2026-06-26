#!/bin/bash
# Scaling sweep for the Gray-Scott transfer benchmark.
set -e
cd /work
NVCC=/usr/local/cuda-12.6/bin/nvcc
if [ ! -x ./gs_bench ]; then
  $NVCC -O3 -gencode arch=compute_90,code=compute_90 -o gs_bench \
    clio_gray_scott_transfer_bench.cu
fi
printf '%-26s %-12s %10s %10s %12s %9s %12s\n' \
  "config" "case" "ms_total" "us/step" "GB/s(eff)" "ratio" "MB_moved"
for pbb in 1024 4096 16384; do
  for nb in 1024 8192 32768; do
    ./gs_bench "$nb" 128 "$pbb" 100 3 \
      | grep -vE '^#|^case' \
      | awk -v tag="nb=$nb,pbb=$pbb" '{printf "%-26s %s\n", tag, $0}'
  done
done
