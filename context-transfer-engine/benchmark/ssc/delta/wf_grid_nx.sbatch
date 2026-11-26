#!/bin/bash
# Finish the DRAM+PFS (05gp) grid on a NON-exclusive interactive node (1 GPU, 32 cores), and use the
# otherwise idle GPU for the SSC v12 practice evaluation (Gray-Scott practice seeds 1-3), because
# whole idle nodes were ~21 h away. Control: one setup already measured on an exclusive node is
# measured again here (WfCommons phases) into grid_05gp_control.jsonl, to quantify neighbour effects.
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4-interactive
#SBATCH -J ssc_wfstep
#SBATCH -N 1 -n 1 -c 32 --gpus=1 --mem=120g
#SBATCH -t 01:00:00
#SBATCH -o /u/rpawar/clio-core-ssc/ssc_work/logs/wfstep-%j.out
source /u/rpawar/clio-core-ssc/context-transfer-engine/benchmark/ssc/delta/env.sh
G=$RES/grid; D=$RES/wf_done; LOGDIR=$RES/wfstep_$JOB; mkdir -p $LOGDIR $D
if ! mkdir $D/active.lock 2>/dev/null; then
  owner=$(cat $D/active.lock/job 2>/dev/null)
  if [ -n "$owner" ] && squeue -h -j "$owner" -t R 2>/dev/null | grep -q .; then echo "another step ($owner) running"; exit 0; fi
fi
echo $JOB > $D/active.lock/job
trap 'rm -f "${D:?}/active.lock/job"; rmdir "$D/active.lock" 2>/dev/null' EXIT
export SSC_DEADLINE=$(( $(date +%s) + 55 * 60 ))
TIER=/work/hdd/bekn/rpawar/ssc_tier/$JOB; mkdir -p $TIER
PH=gs_output,gs_analysis,gs_products,gs_log,gs_archive,gs_field_reread,gs_events,mt_project,mt_fit,ep_fastq,ep_map,kg_vcf,kg_indiv
CORES=$(taskset -cp $$ | awk '{print $NF}')
cd $HERE
echo "non-exclusive node $(hostname), cores $CORES"
# 0. GPU side, in parallel: v12 practice replays (Gray-Scott only, seeds 1-3; v10 as reference)
LOGDIR=$LOGDIR start_ollama 0 $((11500 + JOB % 300)) $CORES
for m in qwen3:4b llama3.1:8b; do start_bridge $m $((11500 + JOB % 300)) $SSC_TMP/q $CORES; done
P=$RES/practice; mkdir -p $P
declare -A DRAM=([1g]=1 [05g]=0.5 [4g]=4) DISK=([1g]=2 [05g]=1.5 [4g]=8)
(
  for m in qwen3:4b llama3.1:8b; do for n in 1g 05g 4g; do
    [ $(( SSC_DEADLINE - $(date +%s) )) -lt 300 ] && break 2
    out=$P/${n}_gs_$(echo $m | tr ':.' '__').jsonl; [ -s $out ] && continue
    SSC_QUEUE_ROOT=$SSC_TMP/q SSC_DRAM_GB=${DRAM[$n]} SSC_DISK_GB=${DISK[$n]} \
      timeout $(( SSC_DEADLINE - $(date +%s) - 60 )) ctr python3 ablate_ssc.py $out.tmp --seeds 1-3 \
      --grid $G/grid_$n.jsonl --app gs --variants "v10@$m,v12r@$m,v12f@$m,v12@$m,v12d@$m" >> $LOGDIR/practice.log 2>&1 \
      && mv $out.tmp $out
    echo "$(date +%T) practice $n $m rc=$?"
  done; done
) & GPID=$!
# 1. control: re-measure an already-measured setup (WfCommons phases), ~3 min
SSC_DISK_DIR=$TIER SSC_DRAM_GB=0.5 SSC_DISK_GB=2 taskset -c $CORES ctr python3 run_grid.py $G/grid_05gp_control.jsonl \
  --only dram-nocache-noidx-none --phases mt_project,mt_fit,ep_fastq,ep_map,kg_vcf,kg_indiv --repeat 3 > $LOGDIR/control.log 2>&1
echo "$(date +%T) control rc=$?"
# 2. the remaining setups of the 05gp grid
SSC_DISK_DIR=$TIER SSC_SETUP_S=900 SSC_DRAM_GB=0.5 SSC_DISK_GB=2 taskset -c $CORES \
  ctr python3 run_grid.py $G/grid_05gp_wf.jsonl --phases $PH --repeat 3 >> $LOGDIR/grid_05gp.log 2>&1
setups=$(python3 -c "import json;print(len({json.dumps(json.loads(l)['composition'],sort_keys=True) for l in open('$G/grid_05gp_wf.jsonl')}))")
echo "$(date +%T) grid 05gp: $setups/24 setups (non-exclusive)"
if [ "$setups" = 24 ]; then cat $G/grid_05gp_wf.jsonl >> $G/grid_05gp.jsonl; touch $G/grid_05gp_wf.done; fi
rm -f "${TIER:?}"/cte_disk_tier.dat*; rmdir "$TIER" 2>/dev/null
wait $GPID
echo "$(date +%T) step done"
