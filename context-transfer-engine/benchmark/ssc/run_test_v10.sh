#!/bin/bash
# FROZEN SSC = v10 (chosen on practice seeds 1-3: mean 0.90 vs v11 0.875). One run on test seeds 101-110.
cd "$(dirname "$0")"
L=/ssc-bench/live_results
V="const,rulesh,wknn10,none@qwen3:4b,v10@qwen3:4b,none@llama3.1:8b,v10@llama3.1:8b"
for n in "1g 1 2" "05g 0.5 1.5"; do set -- $n
  O=$L/ablation/test_v10_$1.jsonl; rm -f $O
  SSC_DRAM_GB=$2 SSC_DISK_GB=$3 python3 ablate_ssc.py $O --seeds 101-110 --grid $L/grid_v2/grid_$1.jsonl --variants "$V" > /dev/null 2>&1
  echo "=== TEST seeds 101-110, node $1"; python3 ablate_ssc.py --summary $O
done
