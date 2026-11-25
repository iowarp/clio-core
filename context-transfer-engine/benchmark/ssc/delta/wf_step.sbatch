#!/bin/bash
# WfCommons campaign as a stream of 1-hour interactive jobs (the long-job queue was ~2,700 deep).
# Each job looks at what is left and does as much as fits before its deadline; every unit of work
# is resumable or marked done, so the next job continues where this one stopped.
#   CLIO side (cores 0-31, alone): grid 1 GB -> grid 0.5 GB -> live runs (1 GB, wseed 201-203)
#   GPU side (GPUs 1-3, cores 40-63): offline replay units, each started only if it can finish
#   Submit with delta/wf_stream.sh (one job at a time: the interactive QoS allows one per user).
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4-interactive
#SBATCH -J ssc_wfstep
#SBATCH -N 1 --exclusive --gpus=4 --mem=0
#SBATCH -t 01:00:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/wfstep-%j.out
source /u/rpawar/clio-core-ssc/context-transfer-engine/benchmark/ssc/delta/env.sh
LOGDIR=$RES/wfstep_$JOB; mkdir -p $LOGDIR
G=$RES/grid; O=$RES/offline; L=$RES/live; D=$RES/wf_done; mkdir -p $D
export SSC_DEADLINE=$(( $(date +%s) + 55 * 60 ))       # leave 5 min for shutdown
left() { echo $(( SSC_DEADLINE - $(date +%s) )); }
PH=mt_project,mt_fit,ep_fastq,ep_map,kg_vcf,kg_indiv
declare -A DRAM=([1g]=1 [05g]=0.5) DISK=([1g]=2 [05g]=1.5)
CPUS=([0]=32-39 [1]=40-47 [2]=48-55 [3]=56-63)
cd $HERE

grid_done() { [ -f $G/grid_$1_wf.done ]; }
grid_step() {   # measure what fits; mark done (and append to the main grid) once all 24 setups are in
  local n=$1
  SSC_SETUP_S=480 SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c 0-31 \
    ctr python3 run_grid.py $G/grid_${n}_wf.jsonl --phases $PH --repeat 3 >> $LOGDIR/grid_$n.log 2>&1
  local setups=$(python3 -c "import json;print(len({json.dumps(json.loads(l)['composition'],sort_keys=True) for l in open('$G/grid_${n}_wf.jsonl')}))" 2>/dev/null)
  echo "$(date +%T) grid $n: $setups/24 setups"
  if [ "$setups" = 24 ]; then cat $G/grid_${n}_wf.jsonl >> $G/grid_$n.jsonl; touch $G/grid_${n}_wf.done; fi
}

lane() { local g=$1; shift; start_ollama $g $((11500 + g)) ${CPUS[$g]}
         for m in "$@"; do start_bridge $m $((11500 + g)) $SSC_TMP/q$g ${CPUS[$g]}; done; }

# offline units: name node seeds variants est_seconds
UNITS=("s1_1g 1g 101-110 const,rulesh,wknn10,guards,none@qwen3:4b,v10@qwen3:4b,v10x@qwen3:4b,none@llama3.1:8b,v10@llama3.1:8b 2700")
for s in $(seq 101 110); do UNITS+=("s2_1g_$s 1g $s none@qwen3:4b+think 2100"); done
UNITS+=("s1_05g 05g 101-110 const,rulesh,wknn10,guards,none@qwen3:4b,v10@qwen3:4b,v10x@qwen3:4b,none@llama3.1:8b,v10@llama3.1:8b 2700")
for s in $(seq 101 105); do UNITS+=("s2_05g_$s 05g $s none@qwen3:4b+think 2100"); done
rmdir "${D:?}"/*.claim 2>/dev/null               # one job at a time: claims left by a killed job are stale
claim() { mkdir $D/$1.claim 2>/dev/null; }      # one lane per unit
gpu_lane() {    # gpu_lane <gpu>: run units until none fits
  local g=$1
  for u in "${UNITS[@]}"; do
    set -- $u
    local name=$1 n=$2 seeds=$3 vars=$4 est=$5
    [ -f $D/$name.done ] && continue
    grid_done $n || continue
    [ $(left) -lt $(( est + 120 )) ] && continue
    claim $name || continue
    SSC_QUEUE_ROOT=$SSC_TMP/q$g SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c ${CPUS[$g]} \
      timeout $(( $(left) - 60 )) ctr python3 ablate_ssc.py $O/${n}_app3_${name}.jsonl.tmp --seeds $seeds \
      --grid $G/grid_$n.jsonl --app app3 --variants "$vars" >> $LOGDIR/off_$g.log 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then mv $O/${n}_app3_${name}.jsonl.tmp $O/${n}_app3_${name}.jsonl; touch $D/$name.done
    else rm -f "${O:?}/${n}_app3_${name}.jsonl.tmp"; fi
    rmdir $D/$name.claim
    echo "$(date +%T) gpu $g unit $name rc=$rc"
  done
}

# live runs (1 GB, three WfCommons schedules); est seconds per arm (S2 blocking ~30 min)
live() {
  local s best worst
  read best worst < <(ctr python3 choose_fixed.py $G/grid_1g.jsonl $PH 2>/dev/null)
  for s in 201 202 203; do
    for a in "fixed-best 420 fixed:$best" "rules 420 heuristic2" "s1-ssc 480 s1:qwen3:4b --ssc v10 --rules" \
             "s1-nossc 480 s1:qwen3:4b --ssc none --rules" "s1-ssc-llama 480 s1:llama3.1:8b --ssc v10 --rules" \
             "oracle 420 oracle" "s2-nossc 2400 s1:qwen3:4b+think --ssc none --rules"; do
      set -- $a; local arm=$1 est=$2; shift 2
      local out=$L/1g_app3/${arm}_s${s}_r1.jsonl
      [ -s $out ] && continue
      [ $(left) -lt $(( est + 120 )) ] && continue
      mkdir -p $L/1g_app3
      SSC_QUEUE_ROOT=$SSC_TMP/q0 SSC_DRAM_GB=1 SSC_DISK_GB=2 taskset -c 0-31 timeout $(( $(left) - 60 )) \
        ctr python3 e2e.py "$@" $out.tmp --schedule wseed:$s --grid $G/grid_1g.jsonl > ${out%.jsonl}.log 2>&1
      local rc=$?
      [ $rc -eq 0 ] && mv "$out.tmp" "$out" || rm -f "${out:?}.tmp"
      echo "$(date +%T) live $arm wseed:$s rc=$rc"
    done
  done
}

PIDS=()
if ! grid_done 1g; then
  grid_step 1g                                   # GPUs idle: nothing to replay before the 1 GB grid
fi
if grid_done 1g; then                            # continue in the same job once the 1 GB grid is in
  lane 1 qwen3:4b llama3.1:8b qwen3:4b+think; lane 2 qwen3:4b+think; lane 3 qwen3:4b+think
  for g in 1 2 3; do gpu_lane $g & PIDS+=($!); done
  # CLIO side: live runs first (the paper's end-to-end result), then the 0.5 GB grid
  lane 0 qwen3:4b qwen3:4b+think llama3.1:8b; live
  grid_done 05g || grid_step 05g
  wait "${PIDS[@]}"
fi
# everything done?
all=1; for u in "${UNITS[@]}"; do set -- $u; [ -f $D/$1.done ] || all=0; done
for s in 201 202 203; do for a in fixed-best rules s1-ssc s1-nossc s1-ssc-llama oracle s2-nossc; do
  [ -s $L/1g_app3/${a}_s${s}_r1.jsonl ] || all=0; done; done
[ $all = 1 ] && touch $D/ALL_DONE && echo ALL_DONE
echo "$(date +%T) step done"
