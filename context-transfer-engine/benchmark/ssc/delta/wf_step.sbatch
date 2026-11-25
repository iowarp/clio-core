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
# One step at a time across both queues (interactive 1-h steps and a long regular-queue step may be
# pending together; whichever starts first works, a second one that starts meanwhile exits at once)
if ! mkdir $D/active.lock 2>/dev/null; then
  owner=$(cat $D/active.lock/job 2>/dev/null)
  if [ -n "$owner" ] && squeue -h -j "$owner" -t R 2>/dev/null | grep -q .; then
    echo "another step ($owner) is running; exiting"; exit 0
  fi
fi
echo $JOB > $D/active.lock/job
trap 'rm -f "${D:?}/active.lock/job"; rmdir "$D/active.lock" 2>/dev/null' EXIT
# STEP_MIN: minutes of work (55 for the 1-hour interactive QoS; set for longer regular-queue steps)
export SSC_DEADLINE=$(( $(date +%s) + ${STEP_MIN:-55} * 60 ))   # leave 5 min for shutdown
left() { echo $(( SSC_DEADLINE - $(date +%s) )); }
PH=mt_project,mt_fit,ep_fastq,ep_map,kg_vcf,kg_indiv
declare -A DRAM=([1g]=1 [05g]=0.5 [05gp]=0.5) DISK=([1g]=2 [05g]=1.5 [05gp]=2)
# 05gp: a second node setup, 0.5 GB DRAM + the second tier on the parallel file system (Lustre
# /work/hdd) instead of local NVMe. It also measures the Gray-Scott phases: they are the SSC's node
# experience, and a new setup has to rebuild it from its own measurements.
declare -A PHN=([1g]=$PH [05g]=$PH [05gp]=gs_output,gs_analysis,gs_products,gs_log,gs_archive,gs_field_reread,gs_events,$PH)
TIER=/work/hdd/bekn/rpawar/ssc_tier/$JOB        # scratch tier file (not results); removed after each grid step
CPUS=([0]=32-39 [1]=40-47 [2]=48-55 [3]=56-63)
cd $HERE

grid_done() { [ -f $G/grid_$1_wf.done ]; }
grid_step() {   # measure what fits; mark done (and append to the main grid) once all 24 setups are in
  local n=$1
  local extra=""; [ $n = 05gp ] && { mkdir -p $TIER; extra="SSC_DISK_DIR=$TIER"; }
  env $extra SSC_SETUP_S=$([ $n = 05gp ] && echo 900 || echo 480) SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} \
    taskset -c 0-31 ctr python3 run_grid.py $G/grid_${n}_wf.jsonl --phases ${PHN[$n]} --repeat 3 >> $LOGDIR/grid_$n.log 2>&1
  [ $n = 05gp ] && rm -f "${TIER:?}"/cte_disk_tier.dat* && rmdir "$TIER" 2>/dev/null
  local setups=$(python3 -c "import json;print(len({json.dumps(json.loads(l)['composition'],sort_keys=True) for l in open('$G/grid_${n}_wf.jsonl')}))" 2>/dev/null)
  echo "$(date +%T) grid $n: $setups/24 setups"
  if [ "$setups" = 24 ]; then cat $G/grid_${n}_wf.jsonl >> $G/grid_$n.jsonl; touch $G/grid_${n}_wf.done; fi
}

lane() { local g=$1; shift; start_ollama $g $((11500 + g)) ${CPUS[$g]}
         for m in "$@"; do start_bridge $m $((11500 + g)) $SSC_TMP/q$g ${CPUS[$g]}; done; }

# offline units: name node app seeds variants est_seconds
# Long System 2 units (one schedule, ~31 min) come first in each group so a fresh hour starts them;
# the short per-arm S1 units (~10 min) then fill the tail of the hour instead of leaving GPUs idle.
s1_units() {   # s1_units <prefix> <node> <app>: one unit per arm (no-model baselines together)
  UNITS+=("$1_nomodel $2 $3 101-110 const,rules,rulesh,wknn10,guards 300")
  for v in v10@qwen3:4b none@qwen3:4b v10@llama3.1:8b v10x@qwen3:4b none@llama3.1:8b; do
    UNITS+=("$1_$(echo $v | tr '@:.' '___') $2 $3 101-110 $v 720"); done
}
UNITS=()
for s in $(seq 101 110); do UNITS+=("s2_1g_$s 1g app3 $s none@qwen3:4b+think 1200"); done
s1_units s1_1g 1g app3
for s in $(seq 101 105); do UNITS+=("s2_05g_$s 05g app3 $s none@qwen3:4b+think 1200"); done
s1_units s1_05g 05g app3
UNITS+=("s1_05g_noguard 05g app3 101-110 ng_v10@qwen3:4b 720")      # is the guard the conservative part?
for s in $(seq 101 105); do UNITS+=("s2_05gp_$s 05gp app3 $s none@qwen3:4b+think 1200"); done
s1_units s1_05gp 05gp app3
UNITS+=("s1_05gp_noguard 05gp app3 101-110 ng_v10@qwen3:4b 720")
s1_units s1_05gp_gs 05gp gs
rmdir "${D:?}"/*.claim 2>/dev/null               # one job at a time: claims left by a killed job are stale
claim() { mkdir $D/$1.claim 2>/dev/null; }      # one lane per unit
gpu_lane() {    # gpu_lane <gpu>: run units until none fits
  local g=$1
  for u in "${UNITS[@]}"; do
    set -- $u
    local name=$1 n=$2 app=$3 seeds=$4 vars=$5 est=$6
    [ -f $D/$name.done ] && continue
    grid_done $n || continue
    [ $(left) -lt $(( est + 120 )) ] && continue
    claim $name || continue
    SSC_QUEUE_ROOT=$SSC_TMP/q$g SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c ${CPUS[$g]} \
      timeout $(( $(left) - 60 )) ctr python3 ablate_ssc.py $O/${n}_${app}_${name}.jsonl.tmp --seeds $seeds \
      --grid $G/grid_$n.jsonl --app $app --variants "$vars" >> $LOGDIR/off_$g.log 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then mv $O/${n}_${app}_${name}.jsonl.tmp $O/${n}_${app}_${name}.jsonl; touch $D/$name.done
    else rm -f "${O:?}/${n}_${app}_${name}.jsonl.tmp"; fi
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
             "oracle 420 oracle" "s1-ssc-async 480 s1:qwen3:4b --ssc v10 --rules --async-decide" \
             "s2-nossc-async 600 s1:qwen3:4b+think --ssc none --rules --async-decide" \
             "s2-nossc 2400 s1:qwen3:4b+think --ssc none --rules"; do
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
  for g in 1 2 3; do lane $g qwen3:4b llama3.1:8b qwen3:4b+think; done
  for g in 1 2 3; do gpu_lane $g & PIDS+=($!); done
  # CLIO side: the 0.5 GB and DRAM+PFS grids first (portability is what the paper needs most once
  # live schedule 201 is in), then the remaining live runs
  grid_done 05g || grid_step 05g
  grid_done 05g && { grid_done 05gp || grid_step 05gp; }
  grid_done 05gp && { lane 0 qwen3:4b qwen3:4b+think llama3.1:8b; live; }
  wait "${PIDS[@]}"
fi
# everything done?
all=1; for u in "${UNITS[@]}"; do set -- $u; [ -f $D/$1.done ] || all=0; done
for s in 201 202 203; do for a in fixed-best rules s1-ssc s1-nossc s1-ssc-llama oracle s1-ssc-async s2-nossc-async s2-nossc; do
  [ -s $L/1g_app3/${a}_s${s}_r1.jsonl ] || all=0; done; done
[ $all = 1 ] && touch $D/ALL_DONE && echo ALL_DONE
echo "$(date +%T) step done"
