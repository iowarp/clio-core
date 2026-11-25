#!/bin/bash
# Campaign 2: live end-to-end runs on CLIO (needs the grids of campaign 1). Resumable: finished
# runs are skipped, so the same script can be resubmitted. Live runs are sequential on cores 0-31
# with the model on GPU 0 (offline replays run as separate 1-GPU jobs: offline_lane.sbatch).
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J ssc_c2
#SBATCH -N 1 --exclusive --gpus=4 --mem=0
#SBATCH -t 07:00:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/c2-%j.out
source /u/rpawar/clio-core-ssc/context-transfer-engine/benchmark/ssc/delta/env.sh
LOGDIR=$RES/c2_$JOB; mkdir -p $LOGDIR
G=$RES/grid; O=$RES/offline; L=$RES/live
declare -A DRAM=([1g]=1 [05g]=0.5 [4g]=4) DISK=([1g]=2 [05g]=1.5 [4g]=8)
CPUS=([0]=32-39 [1]=40-47 [2]=48-55 [3]=56-63)
cd $HERE
lane() { local g=$1; shift; start_ollama $g $((11500 + g)) ${CPUS[$g]}
         for m in "$@"; do start_bridge $m $((11500 + g)) $SSC_TMP/q$g ${CPUS[$g]}; done; }
off() {  # off <lane> <node> <app> <seeds> <variants>
  local n=$2
  SSC_QUEUE_ROOT=$SSC_TMP/q$1 SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c ${CPUS[$1]} \
    ctr python3 ablate_ssc.py $O/${n}_$3_c2l$1.jsonl --seeds $4 --grid $G/grid_$n.jsonl --app $3 --variants "$5" \
    >> $LOGDIR/off_$1.log 2>&1; echo "$(date +%T) lane $1 $n $3 $4 rc=$?"; }
# run <node> <arm> <schedule> <e2e args...>
run() {
  local n=$1 arm=$2 sch=$3; shift 3
  local d=$L/$n; [[ $sch == dseed* ]] && d=$L/${n}_app2; [[ $sch == wseed* ]] && d=$L/${n}_app3
  local out=$d/${arm}_s${sch#*:}_r1.jsonl
  [ -s $out ] && return
  mkdir -p $d
  SSC_QUEUE_ROOT=$SSC_TMP/q0 SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} taskset -c 0-31 \
    ctr python3 e2e.py "$@" $out.tmp --schedule $sch --grid $G/grid_$n.jsonl > ${out%.jsonl}.log 2>&1
  local rc=$?
  [ $rc -eq 0 ] && mv $out.tmp $out
  echo "$(date +%T) $n $arm $sch rc=$rc"
}
arms() {  # arms <node> <schedule> [app phases for choose_fixed]
  local n=$1 s=$2
  read best worst < <(ctr python3 choose_fixed.py $G/grid_$n.jsonl $3 2>/dev/null)
  run $n fixed-best $s fixed:$best
  run $n rules $s heuristic2
  run $n s1-ssc $s s1:qwen3:4b --ssc v10 --rules
  run $n s1-nossc $s s1:qwen3:4b --ssc none --rules
  run $n s1-ssc-llama $s s1:llama3.1:8b --ssc v10 --rules
  run $n oracle $s oracle
  run $n s1-ssc-async $s s1:qwen3:4b --ssc v10 --rules --async-decide
  run $n s2-nossc-async $s s1:qwen3:4b+think --ssc none --rules --async-decide
  run $n s2-nossc $s s1:qwen3:4b+think --ssc none --rules
}
lane 0 qwen3:4b qwen3:4b+think llama3.1:8b
case $1 in   # one block per job (each fits a short exclusive slot; runs are skipped once done)
  1g)   for s in 201 202 203 204 205; do arms 1g seed:$s; done ;;
  app2) for s in 201 202 203; do arms 1g dseed:$s dl_samples,dl_ckpt,h5_particles; done ;;
  05g)  for s in 201 202 203 204 205; do arms 05g seed:$s; done ;;
  app3) for s in 201 202 203; do arms 1g wseed:$s mt_project,mt_fit,ep_fastq,ep_map,kg_vcf,kg_indiv; done ;;
esac
echo C2_DONE
