#!/bin/bash
# After the 0.5 GB grid: replay it, then a short live benchmark (~18 min) on the 1 GB node:
# qwen3-4B (thinking off) with SSC v11 vs without SSC, and a fixed safe baseline, seeds 201-202.
cd "$(dirname "$0")"
L=/ssc-bench/live_results
while ! grep -q "done 05g" $L/grid_v2/progress.log; do sleep 20; done
./replay_v2.sh $L/grid_v2/grid_05g.jsonl 0.5 1.5 $L/ablation/v2_05g_dev.jsonl > $L/ablation/v2_05g_dev.txt 2>&1
O=$L/short_v2; mkdir -p $O
G=$L/grid_v2/grid_1g.jsonl
for s in 201 202; do
  python3 e2e.py s1:qwen3:4b $O/qwen-ssc_s$s.jsonl --schedule seed:$s --ssc v11 --rules --grid $G > $O/qwen-ssc_s$s.log 2>&1
  python3 e2e.py s1:qwen3:4b $O/qwen-nossc_s$s.jsonl --schedule seed:$s --ssc none --rules --grid $G > $O/qwen-nossc_s$s.log 2>&1
  python3 e2e.py fixed:dram-nocache-noidx-lz4 $O/fixed_s$s.jsonl --schedule seed:$s --grid $G > $O/fixed_s$s.log 2>&1
  echo "$(date +%T) seed $s done" >> $O/progress.log
done
echo "$(date +%T) all done" >> $O/progress.log
