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
declare -A DRAM=([1g]=1 [05g]=0.5 [05gp]=0.5 [4g]=4 [16g]=16 [16gp]=16) DISK=([1g]=2 [05g]=1.5 [05gp]=2 [4g]=8 [16g]=48 [16gp]=48)
# 16g / 16gp: the large-scale setups (16 GB DRAM + 48 GB NVMe or Lustre tier) with the phases at the
# sizes of the largest WfInstances instances (SSC_SCALE=large, workload.LARGE); all 11 phases in one grid
large() { [[ $1 == 16g* ]] && echo SSC_SCALE=large; }
# 05gp: a second node setup, 0.5 GB DRAM + the second tier on the parallel file system (Lustre
# /work/hdd) instead of local NVMe. It also measures the Gray-Scott phases: they are the SSC's node
# experience, and a new setup has to rebuild it from its own measurements.
declare -A PHN=([1g]=$PH [05g]=$PH [05gp]=gs_output,gs_analysis,gs_products,gs_log,gs_archive,gs_field_reread,gs_events,$PH
                [16g]=$PH,cy_season,sk_realign,sk_gvcf,sg_decon,sra_fastq [16gp]=$PH,cy_season,sk_realign,sk_gvcf,sg_decon,sra_fastq)
TIER=/work/hdd/bekn/rpawar/ssc_tier/$JOB        # scratch tier file (not results); removed after each grid step
CPUS=([0]=32-39 [1]=40-47 [2]=48-55 [3]=56-63)
cd $HERE

grid_done() { [ -f $G/grid_$1_wf.done ] || { [ $1 = 4g ] && [ -f $G/grid_4g.done ]; }; }   # 4g: GS/DL grid only
grid_step() {   # measure what fits; mark done (and append to the main grid) once all 24 setups are in
  local n=$1
  local extra=""; [[ $n == *gp ]] && { mkdir -p $TIER; extra="SSC_DISK_DIR=$TIER"; }
  local setup_s; case $n in 05gp) setup_s=900;; 16g) setup_s=1500;; 16gp) setup_s=2400;; *) setup_s=480;; esac
  env $extra $(large $n) SSC_SETUP_S=$setup_s SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} \
    taskset -c 0-31 ctr python3 run_grid.py $G/grid_${n}_wf.jsonl --phases ${PHN[$n]} --repeat 3 >> $LOGDIR/grid_$n.log 2>&1
  [[ $n == *gp ]] && rm -f "${TIER:?}"/cte_disk_tier.dat* && rmdir "$TIER" 2>/dev/null
  local setups=$(python3 -c "import json;print(len({json.dumps(json.loads(l)['composition'],sort_keys=True) for l in open('$G/grid_${n}_wf.jsonl')}))" 2>/dev/null)
  echo "$(date +%T) grid $n: $setups/24 setups"
  if [ "$setups" = 24 ]; then cat $G/grid_${n}_wf.jsonl >> $G/grid_$n.jsonl; touch $G/grid_${n}_wf.done; fi
}
# the WfCommons campaign adds four workflows (Cycles, SoyKB, Seismology, SRAsearch): their phases are
# measured on every node setup into grid_<n>_wf2 and appended to the node's grid once complete
WF2=cy_season,sk_realign,sk_gvcf,sg_decon,sra_fastq
wf2_done() { [ -f $G/grid_$1_wf2.done ]; }
ready() { grid_done $1 && { wf2_done $1 || [[ $1 == 16g* ]]; }; }
wf2_step() {
  local n=$1
  local extra=""; [ $n = 05gp ] && { mkdir -p $TIER; extra="SSC_DISK_DIR=$TIER"; }
  env $extra SSC_SETUP_S=$([ $n = 05gp ] && echo 900 || echo 480) SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} \
    taskset -c 0-31 ctr python3 run_grid.py $G/grid_${n}_wf2.jsonl --phases $WF2 --repeat 3 >> $LOGDIR/grid_${n}_wf2.log 2>&1
  [ $n = 05gp ] && rm -f "${TIER:?}"/cte_disk_tier.dat* && rmdir "$TIER" 2>/dev/null
  local setups=$(python3 -c "import json;print(len({json.dumps(json.loads(l)['composition'],sort_keys=True) for l in open('$G/grid_${n}_wf2.jsonl')}))" 2>/dev/null)
  echo "$(date +%T) grid $n wf2: $setups/24 setups"
  if [ "$setups" = 24 ]; then cat $G/grid_${n}_wf2.jsonl >> $G/grid_$n.jsonl; touch $G/grid_${n}_wf2.done; fi
}

lane() { local g=$1; shift; start_ollama $g $((11500 + g)) ${CPUS[$g]}
         for m in "$@"; do start_bridge $m $((11500 + g)) $SSC_TMP/q$g ${CPUS[$g]}; done; }

# offline units: name node app seeds variants est_seconds [memory-phases]
# All on the WfCommons campaign (app wf: seven workflows' phases on one node) with the SSC's experience
# from the other workflows (--experience wf, leave-one-workflow-out). SSC v12d as frozen.
UNITS=()
for n in 1g 05g 05gp; do
  UNITS+=("f_${n}_v12 $n wf 101-110 v12d@qwen3:4b,v12d@llama3.1:8b 900")
  UNITS+=("f_${n}_nomodel $n wf 101-110 const,rules,rulesh,wknn10,guards 400")
  UNITS+=("f_${n}_s1none $n wf 101-110 none@qwen3:4b 720")
  UNITS+=("f_${n}_cold $n wf 101-110 v10x@qwen3:4b 720")      # SSC without node experience
done
# model size (1 GB): every fast model with and without the SSC
for m in qwen3:1.7b qwen3:8b qwen3:14b; do UNITS+=("f_sweep_${m//[:.]/_} 1g wf 101-110 v12d@$m,none@$m 1100"); done
for s in 101 102 103 104 105; do UNITS+=("f_s2_1g_$s 1g wf $s none@qwen3:4b+think 1200"); done
# large scale: the headline setups
for n in 16g 16gp; do
  UNITS+=("f_${n}_v12 $n wf 101-110 v12d@qwen3:4b,v12d@llama3.1:8b 900")
  UNITS+=("f_${n}_nomodel $n wf 101-110 const,rules,rulesh,wknn10,guards 400")
  UNITS+=("f_${n}_s1none $n wf 101-110 none@qwen3:4b 720")
  UNITS+=("f_${n}_cold $n wf 101-110 v10x@qwen3:4b 720")
  for s in 101 102 103; do UNITS+=("f_s2_${n}_$s $n wf $s none@qwen3:4b+think 1200"); done
done
for n in 05g 05gp; do for s in 101 102 103; do UNITS+=("f_s2_${n}_$s $n wf $s none@qwen3:4b+think 1200"); done; done
# learning curve: the node has measured N of the 11 WfCommons phases (lc_subsets.py, seeded)
while read k mem; do [ $k = 11 ] && continue
  i=$(( ${i:-0} + 1 ))
  for n in 1g 05g; do UNITS+=("lc_${n}_wf_n${k}_$i $n wf 101-105 wknn10,v12d@qwen3:4b 420 $mem"); done
done < <(cd $HERE && ctr python3 lc_subsets.py 3 wf)
rmdir "${D:?}"/*.claim 2>/dev/null               # one job at a time: claims left by a killed job are stale
claim() { mkdir $D/$1.claim 2>/dev/null; }      # one lane per unit
gpu_lane() {    # gpu_lane <gpu>: run units until none fits
  local g=$1
  for u in "${UNITS[@]}"; do
    set -- $u
    local name=$1 n=$2 app=$3 seeds=$4 vars=$5 est=$6 mem=${7:+--memory-phases $7}
    local xp=""; [ $app = wf ] && xp="--experience wf"
    [ -f $D/$name.done ] && continue
    if [ $app = wf ]; then ready $n || continue; else grid_done $n || continue; fi
    [ $(left) -lt $(( est + 120 )) ] && continue
    claim $name || continue
    env $(large $n) SSC_QUEUE_ROOT=$SSC_TMP/q$g SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c ${CPUS[$g]} \
      timeout $(( $(left) - 60 )) ctr python3 ablate_ssc.py $O/${n}_${app}_${name}.jsonl.tmp --seeds $seeds \
      --grid $G/grid_$n.jsonl --app $app --variants "$vars" $mem $xp >> $LOGDIR/off_$g.log 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then mv $O/${n}_${app}_${name}.jsonl.tmp $O/${n}_${app}_${name}.jsonl; touch $D/$name.done
    else rm -f "${O:?}/${n}_${app}_${name}.jsonl.tmp"; fi
    rmdir $D/$name.claim
    echo "$(date +%T) gpu $g unit $name rc=$rc"
  done
}

# live runs (1 GB, the WfCommons campaign, schedules 301-303); est seconds per arm (S2 blocking ~30 min)
WFPH=mt_project,mt_fit,ep_fastq,ep_map,kg_vcf,kg_indiv,$WF2
LIVE_ARMS=("fixed-default 420 fixed:dram-nocache-noidx-none" "s1-ssc12 480 s1:qwen3:4b --ssc v12d --rules"
           "s1-ssc12-async 480 s1:qwen3:4b --ssc v12d --rules --async-decide" "fixed-best 420 fixed:BEST"
           "rules 420 heuristic2" "s1-nossc 480 s1:qwen3:4b --ssc none --rules" "oracle 420 oracle"
           "s2-nossc-async 600 s1:qwen3:4b+think --ssc none --rules --async-decide"
           "s2-nossc 2400 s1:qwen3:4b+think --ssc none --rules")
live() {
  local s best worst
  read best worst < <(ctr python3 choose_fixed.py $G/grid_1g.jsonl $WFPH 2>/dev/null)
  for s in 301 302 303; do
    for a in "${LIVE_ARMS[@]}"; do
      set -- ${a/BEST/$best}; local arm=$1 est=$2; shift 2
      local out=$L/1g_wf/${arm}_s${s}_r1.jsonl
      [ -s $out ] && continue
      [ $(left) -lt $(( est + 120 )) ] && continue
      mkdir -p $L/1g_wf
      SSC_QUEUE_ROOT=$SSC_TMP/q0 SSC_DRAM_GB=1 SSC_DISK_GB=2 taskset -c 0-31 timeout $(( $(left) - 60 )) \
        ctr python3 e2e.py "$@" $out.tmp --schedule fseed:$s --experience wf --grid $G/grid_1g.jsonl > ${out%.jsonl}.log 2>&1
      local rc=$?
      [ $rc -eq 0 ] && mv "$out.tmp" "$out" || rm -f "${out:?}.tmp"
      echo "$(date +%T) live $arm fseed:$s rc=$rc"
    done
  done
}

PIDS=()
# CLIO side first: the new workflows' grids (1 GB first: every unit waits for its node's grid), then
# the last DRAM+PFS setup and its new workflows, then live runs; GPU lanes start as soon as a grid is in
grid_done 1g && { wf2_done 1g || wf2_step 1g; }
if ready 1g; then
  for g in 1 2 3; do lane $g qwen3:4b llama3.1:8b qwen3:4b+think qwen3:1.7b qwen3:8b qwen3:14b; done
  for g in 1 2 3; do gpu_lane $g & PIDS+=($!); done
  wf2_done 05g || wf2_step 05g
  grid_done 05gp || grid_step 05gp
  grid_done 05gp && { wf2_done 05gp || wf2_step 05gp; }
  # live runs on the 1 GB setup, then the large-scale grids (each step uses what is left of its hour)
  ready 05gp && { lane 0 qwen3:4b qwen3:4b+think; live; }
  ready 05gp && { grid_done 16g || grid_step 16g; }
  grid_done 16g && { grid_done 16gp || grid_step 16gp; }
  wait "${PIDS[@]}"
fi
# everything done?
all=1; for u in "${UNITS[@]}"; do set -- $u; [ -f $D/$1.done ] || all=0; done
for s in 301 302 303; do for a in "${LIVE_ARMS[@]}"; do set -- $a
  [ -s $L/1g_wf/${1}_s${s}_r1.jsonl ] || all=0; done; done
[ $all = 1 ] && touch $D/ALL_DONE && echo ALL_DONE
echo "$(date +%T) step done"
