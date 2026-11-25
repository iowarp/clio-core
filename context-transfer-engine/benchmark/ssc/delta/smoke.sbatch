#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4-interactive
#SBATCH -J ssc_smoke
#SBATCH -N 1 -n 1 -c 16 --gpus=1 --mem=64g
#SBATCH -t 00:50:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/smoke-%j.out
set -x
source /u/rpawar/clio-core-ssc/context-transfer-engine/benchmark/ssc/delta/env.sh
LOGDIR=$RES/smoke_$JOB; mkdir -p $LOGDIR
df -h /tmp; nproc; nvidia-smi -L; free -g | head -2
cd $HERE
# 1. CLIO: fixed composition, two short windows
ctr python3 e2e.py fixed:dram-nocache-idx-lz4 $LOGDIR/fixed.jsonl --schedule gs_log:1,dl_ckpt:1,gs_output:1 2>&1 | tail -5
# 2. models
start_ollama 0 11500 0-7
for m in qwen3:4b qwen3:4b+think; do start_bridge $m 11500 $SSC_QUEUE_ROOT 8-9; done
sleep 5
ctr python3 e2e.py s1:qwen3:4b $LOGDIR/s1_none.jsonl --schedule gs_output:1,gs_log:1,gs_archive:1 --ssc none --rules 2>&1 | tail -5
ctr python3 e2e.py s1:qwen3:4b+think $LOGDIR/s2_none.jsonl --schedule gs_output:1,gs_log:1,gs_archive:1 --ssc none --rules 2>&1 | tail -5
grep -h decision_latency $LOGDIR/*.jsonl | python3 -c "
import sys,json
for l in sys.stdin: r=json.loads(l); print(r['policy'], r['phase'], r.get('answers'), round(r['decision_latency_s'],2))"
tail -3 $LOGDIR/bridge_*.log
echo SMOKE_DONE
