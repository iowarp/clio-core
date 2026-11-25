#!/bin/bash
# Campaign 1: measured grids (3 node sizes, 10 phases, 3 repeats) + every offline replay.
# CLIO on cores 0-31 (grids only, sequential); one model lane per GPU on cores 32-63.
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J ssc_c1
#SBATCH -N 1 --exclusive --gpus=4 --mem=0
#SBATCH -t 14:00:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/c1-%j.out
source /u/rpawar/clio-core-ssc/context-transfer-engine/benchmark/ssc/delta/env.sh
LOGDIR=$RES/c1_$JOB; mkdir -p $LOGDIR $RES/grid $RES/offline
G=$RES/grid; O=$RES/offline
PH=gs_output,gs_analysis,gs_products,gs_log,gs_archive,gs_field_reread,gs_events,dl_samples,dl_ckpt,h5_particles
declare -A DRAM=([1g]=1 [05g]=0.5 [4g]=4) DISK=([1g]=2 [05g]=1.5 [4g]=8)
cd $HERE
nvidia-smi -L; lscpu | grep -E "Model name|^CPU\(s\)"

grids() {
  for n in 1g 05g 4g; do
    [ -f $G/grid_$n.done ] && continue
    SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c 0-31 \
      ctr python3 run_grid.py $G/grid_$n.jsonl --phases $PH --repeat 3 > $LOGDIR/grid_$n.log 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -s $G/grid_$n.jsonl ]; then touch $G/grid_$n.done; else touch $G/grid_$n.failed; fi
    echo "$(date +%T) grid $n rc=$rc rows=$(wc -l < $G/grid_$n.jsonl 2>/dev/null)"
  done
}
# off <lane> <node> <app> <seeds> <variants>
off() {
  local n=$2
  until [ -f $G/grid_$n.done ]; do [ -f $G/grid_$n.failed ] && return 1; sleep 60; done
  SSC_QUEUE_ROOT=$SSC_TMP/q$1 SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c ${CPUS[$1]} \
    ctr python3 ablate_ssc.py $O/${n}_$3_$1.jsonl --seeds $4 --grid $G/grid_$n.jsonl --app $3 --variants "$5" \
    >> $LOGDIR/off_$1.log 2>&1
  local rc=$?; echo "$(date +%T) lane $1 $n $3 $4 $5 rc=$rc"
}
lane() {   # lane <gpu> <models...>
  local g=$1; shift
  start_ollama $g $((11500 + g)) ${CPUS[$g]}
  for m in "$@"; do start_bridge $m $((11500 + g)) $SSC_TMP/q$g ${CPUS[$g]}; done
}
CPUS=([0]=32-39 [1]=40-47 [2]=48-55 [3]=56-63)
NOMODEL="const,rulesh,wknn10,guards"
S1="none@qwen3:4b,v10@qwen3:4b,none@llama3.1:8b,v10@llama3.1:8b"
SWEEP="none@qwen3:1.7b,v10@qwen3:1.7b,none@qwen3:8b,v10@qwen3:8b,none@qwen3:14b,v10@qwen3:14b"
GUARD="ng_none@qwen3:4b,ng_v10@qwen3:4b,ng_v10@llama3.1:8b"

grids & PIDS=($!)
lane 1 qwen3:4b llama3.1:8b qwen3:1.7b qwen3:8b qwen3:14b
lane 2 qwen3:4b+think
lane 3 qwen3:4b+think
lane 0 qwen3:14b+think
(  # lane 1: everything with thinking off, all node sizes and both applications
  off 1 1g gs 101-110 "$NOMODEL,$S1"; off 1 1g gs 101-110 "$SWEEP,$GUARD"; off 1 1g app2 101-110 "$NOMODEL,$S1"
  off 1 05g gs 101-110 "$NOMODEL,$S1"; off 1 05g app2 101-110 "$NOMODEL,$S1"
  off 1 4g gs 101-110 "$NOMODEL,$S1"; off 1 4g app2 101-110 "$NOMODEL,$S1"
  off 1 05g gs 101-110 "$SWEEP"
) & PIDS+=($!)
(  # lane 2: System 2 without SSC (the baseline the paper beats)
  off 2 1g gs 101-110 "none@qwen3:4b+think"; off 2 05g gs 101-105 "none@qwen3:4b+think"
) & PIDS+=($!)
(  # lane 3: System 2 with the SSC (reference), then System 2 on the second application
  off 3 1g gs 101-105 "v10@qwen3:4b+think"; off 3 1g app2 101-105 "none@qwen3:4b+think"
  off 3 4g gs 101-105 "none@qwen3:4b+think"
) & PIDS+=($!)
(  # lane 0: the strongest System 2 (14B, thinking) without SSC
  off 0 1g gs 101-105 "none@qwen3:14b+think"
) & PIDS+=($!)
wait "${PIDS[@]}"
echo C1_DONE
