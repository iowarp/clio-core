#!/bin/bash
# One offline replay lane on a single GPU (no CLIO measurement unless the lane repairs a grid).
#   sbatch offline_lane.sbatch lanes/<lane>.sh
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J ssc_off
#SBATCH -N 1 -n 1 -c 16 --gpus=1 --mem=60g
#SBATCH -t 16:00:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/off-%j.out
source /u/rpawar/clio-core-ssc/context-transfer-engine/benchmark/ssc/delta/env.sh
LOGDIR=$RES/off_$JOB; mkdir -p $LOGDIR
G=$RES/grid; O=$RES/offline
declare -A DRAM=([1g]=1 [05g]=0.5 [4g]=4) DISK=([1g]=2 [05g]=1.5 [4g]=8)
CORES=$(taskset -cp $$ | awk '{print $NF}')
cd $HERE
off() {  # off <node> <app> <seeds> <variants>
  until [ -f $G/grid_$1.done ]; do sleep 60; done
  SSC_QUEUE_ROOT=$SSC_TMP/q SSC_DRAM_GB=${DRAM[$1]} SSC_DISK_GB=${DISK[$1]} \
    ctr python3 ablate_ssc.py $O/${1}_$2_j$JOB.jsonl --seeds $3 --grid $G/grid_$1.jsonl --app $2 --variants "$4" \
    >> $LOGDIR/off.log 2>&1
  local rc=$?; echo "$(date +%T) $1 $2 $3 $4 rc=$rc"
}
lane() { start_ollama 0 $((11500 + JOB % 300)) $CORES
         for m in "$@"; do start_bridge $m $((11500 + JOB % 300)) $SSC_TMP/q $CORES; done; }
NOMODEL="const,rulesh,wknn10,guards"; S1="none@qwen3:4b,v10@qwen3:4b,none@llama3.1:8b,v10@llama3.1:8b"
SWEEP="none@qwen3:1.7b,v10@qwen3:1.7b,none@qwen3:8b,v10@qwen3:8b,none@qwen3:14b,v10@qwen3:14b"
source $HERE/delta/$1
echo OFF_DONE
