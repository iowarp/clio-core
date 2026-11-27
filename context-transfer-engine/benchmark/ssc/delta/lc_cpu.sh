#!/bin/bash
# Learning curve, no-model arm (wknn10: v10's evidence, lowest merged cost per axis), CPU only.
#   bash delta/lc_cpu.sh     -> results/offline/lc_<node>_<app>_wknn10.jsonl
source /u/rpawar/clio-core-ssc/context-transfer-engine/benchmark/ssc/delta/env.sh
cd $HERE
declare -A DRAM=([1g]=1 [05g]=0.5 [4g]=4) DISK=([1g]=2 [05g]=1.5 [4g]=8)
for t in "1g app3" "05g app3" "1g gs" "4g gs" "1g app2"; do
  set -- $t; n=$1; app=$2; out=$RES/offline/lc_${n}_${app}_wknn10.jsonl
  [ -s $out ] && continue
  ctr python3 lc_subsets.py 5 | while read k mem; do
    SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} ctr python3 ablate_ssc.py $out.tmp --seeds 101-110 \
      --grid $RES/grid/grid_$n.jsonl --app $app --variants wknn10 --memory-phases $mem > /dev/null || exit 1
  done && mv $out.tmp $out && echo "$(date +%T) $n $app done"
done
