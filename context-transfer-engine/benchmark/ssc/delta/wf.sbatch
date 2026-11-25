#!/bin/bash
# WfCommons application (third app): grids for its six phases at 1 / 0.5 GB DRAM (3 repeats),
# appended to the main grids once complete, plus its offline replays on the GPUs.
# CLIO on cores 0-31 (grids, sequential); one model lane per GPU on cores 32-63.
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J ssc_wf
#SBATCH -N 1 --exclusive --gpus=4 --mem=0
#SBATCH -t 09:00:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/wf-%j.out
source /u/rpawar/clio-core-ssc/context-transfer-engine/benchmark/ssc/delta/env.sh
LOGDIR=$RES/wf_$JOB; mkdir -p $LOGDIR
G=$RES/grid; O=$RES/offline
PH=mt_project,mt_fit,ep_fastq,ep_map,kg_vcf,kg_indiv
declare -A DRAM=([1g]=1 [05g]=0.5 [4g]=4) DISK=([1g]=2 [05g]=1.5 [4g]=8)
CPUS=([0]=32-39 [1]=40-47 [2]=48-55 [3]=56-63)
cd $HERE
grids() {
  for n in 1g 05g; do
    [ -f $G/grid_${n}_wf.done ] && continue
    SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c 0-31 \
      ctr python3 run_grid.py $G/grid_${n}_wf.jsonl --phases $PH --repeat 3 > $LOGDIR/grid_$n.log 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then cat $G/grid_${n}_wf.jsonl >> $G/grid_$n.jsonl; touch $G/grid_${n}_wf.done
    else touch $G/grid_${n}_wf.failed; fi
    echo "$(date +%T) wf grid $n rc=$rc rows=$(wc -l < $G/grid_${n}_wf.jsonl)"
  done
}
off() {  # off <lane> <node> <seeds> <variants>
  until [ -f $G/grid_$2_wf.done ]; do [ -f $G/grid_$2_wf.failed ] && return 1; sleep 60; done
  SSC_QUEUE_ROOT=$SSC_TMP/q$1 SSC_DRAM_GB=${DRAM[$2]} SSC_DISK_GB=${DISK[$2]} taskset -c ${CPUS[$1]} \
    ctr python3 ablate_ssc.py $O/${2}_app3_wf$1.jsonl --seeds $3 --grid $G/grid_$2.jsonl --app app3 --variants "$4" \
    >> $LOGDIR/off_$1.log 2>&1
  local rc=$?; echo "$(date +%T) lane $1 $2 app3 $3 $4 rc=$rc"
}
lane() { local g=$1; shift; start_ollama $g $((11500 + g)) ${CPUS[$g]}
         for m in "$@"; do start_bridge $m $((11500 + g)) $SSC_TMP/q$g ${CPUS[$g]}; done; }
NOMODEL="const,rulesh,wknn10,guards"; S1="none@qwen3:4b,v10@qwen3:4b,none@llama3.1:8b,v10@llama3.1:8b"
grids & PIDS=($!)
lane 0 qwen3:4b llama3.1:8b; lane 1 qwen3:4b+think; lane 2 qwen3:4b+think; lane 3 qwen3:4b+think
( off 0 1g 101-110 "$NOMODEL,$S1"; off 0 05g 101-110 "$NOMODEL,$S1" ) & PIDS+=($!)
( off 1 1g 101-105 "none@qwen3:4b+think" ) & PIDS+=($!)
( off 2 1g 106-110 "none@qwen3:4b+think" ) & PIDS+=($!)
( off 3 05g 101-105 "none@qwen3:4b+think" ) & PIDS+=($!)
wait "${PIDS[@]}"
echo WF_DONE
