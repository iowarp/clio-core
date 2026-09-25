#!/usr/bin/env bash
#PBS -l place=scatter
#PBS -l filesystems=home:flare
#PBS -A IOWarp
#PBS -j oe
#
# E5 AT SCALE: the four workloads side by side, each on its own quarter of the
# allocation (a 256-node job = four 64-node groups), every one with
# checkpointing on. kmeans and Gray-Scott run pbs_e5_aurora.sh (64 GB/node);
# gmx and lbann run pbs_e4_batch_aurora.sh on 8 GB/node decks sized here
# for the group.
#
#   E5S_OUT        output root (default flare llogan_e1/e5s_<jobid>)
#   E5S_KM         kmeans budgets (GB/node)          default "32 24 16 8 4"
#   E5S_GS         Gray-Scott budgets (GB/node)      default "32 24 16 8 4"
#   E5S_GMX        gmx budget divisors of the slab   default "1 2 4"
#   E5S_LB         lbann budget divisors             default "1 2 4 8"
#   E5S_KM_ITERS / E5S_GS_STEPS / E5S_GS_CKPT / E5S_GMX_PASSES / E5S_LB_STEPS
#   E5S_PERNODE_MB kmeans/Gray-Scott deck per node   default 65536
#   E5S_CAP        per-rank cap (s)                  default 1200
set -u
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
B=${ROOT}/context-transfer-engine/adapter/gpu_vector/benchmark
JOBTAG=${PBS_JOBID%%.*}
OUT=${E5S_OUT:-/lus/flare/projects/IOWarp/llogan_e1/e5s_${JOBTAG}}
mkdir -p "${OUT}"
export ROOT SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-/lus/flare/projects/IOWarp/llogan_e1/sycl_cache}
export CLIO_TASK_PROGRESS_INTERVAL_MS=0

# E5S_WL: which workloads to run, one equal group each (default all four).
WL=(${E5S_WL:-kmeans grayscott gmx lbann})
mapfile -t ALL < <(sort -u "${PBS_NODEFILE}")
N=$(( ${#ALL[@]} / ${#WL[@]} ))
declare -A GF
for g in "${!WL[@]}"; do
  printf '%s\n' "${ALL[@]:$(( g * N )):${N}}" > "${OUT}/nodes_${WL[$g]}"
  GF[${WL[$g]}]="${OUT}/nodes_${WL[$g]}"
done
CAP=${E5S_CAP:-1200}
echo "=== E5 at scale: ${#ALL[@]} nodes, groups of ${N}: ${WL[*]}, out ${OUT} ==="

# ---- gmx deck: K^3 x 8 B = 8 GiB x N, K a multiple of 16 and of N ----------
GX_K=$(awk -v n="${N}" 'BEGIN{k=exp(log(8*2^30*n/8)/3); m=16; while (m % n) m+=16; k=int((k+m-1)/m)*m; print k}')
GX_PKB=$(( GX_K * GX_K * 8 / 1024 ))
GX_SLAB=$(( GX_K / N ))
GX_BLK=${E5S_GMX_BLOCKS:-4}
GX_NEED=$(( 4 * GX_BLK < GX_SLAB + 3 ? 4 * GX_BLK + 2 : GX_SLAB + 5 ))
gx_cells=""
for d in ${E5S_GMX-1 2 4}; do
  c=$(( (GX_SLAB + 5) / d )); [ "${c}" -lt "${GX_NEED}" ] && c=${GX_NEED}
  gx_cells="${gx_cells} gmx+e5b:dram100:${GX_PKB}:${c}"
done
# ---- lbann deck: 32768 W1 rows (8 GB) per node; the page holds a whole W2
# row (H floats), so it is the larger of 1 MB and 128 KB x N --------------
LB_H=$(( 32768 * N ))
LB_PKB=$(( 128 * N )); [ "${LB_PKB}" -lt 1024 ] && LB_PKB=1024
LB_RES=$(( (32768 * 256 / LB_PKB) + (1024 / N * LB_H * 4 / 1024 / LB_PKB) + 16 ))
lb_cells=""
for d in ${E5S_LB-1 2 4 8}; do
  c=$(( LB_RES / d )); [ "${c}" -lt 128 ] && c=128
  lb_cells="${lb_cells} lbann+e5b:dram100:${LB_PKB}:${c}"
done
echo "    gmx: K=${GX_K} page ${GX_PKB} KB slab ${GX_SLAB} blocks ${GX_BLK} need ${GX_NEED} cells:${gx_cells}"
echo "    lbann: H=${LB_H} page ${LB_PKB} KB resident ~${LB_RES} frames cells:${lb_cells}"

km_cells=""; for b in ${E5S_KM-32 24 16 8 4}; do km_cells="${km_cells} kmeans:${b}"; done
gs_cells=""; for b in ${E5S_GS-32 24 16 8 4}; do gs_cells="${gs_cells} grayscott:${b}"; done

e5env="E5_RUNROOT=${OUT} E5_PERNODE_MB=${E5S_PERNODE_MB:-65536} E5_TIER_MB=200000 E5_CAP=${CAP} E5_RETRY=0"
[ -n "${GF[kmeans]:-}" ] && [ -n "${km_cells}" ] && env PBS_NODEFILE="${GF[kmeans]}" ${e5env} E5_CELLS="${km_cells# }" \
    E5_SFX_kmeans=${E5S_KM_SFX:-_x_ckpt2} E5_ITERS_kmeans=${E5S_KM_ITERS:-24} E5_KM_REPEAT=1 E5_KM_EXTRA="--ckpt-final" \
    bash "${B}/pbs_e5_aurora.sh" > "${OUT}/kmeans.log" 2>&1 &
[ -n "${GF[grayscott]:-}" ] && [ -n "${gs_cells}" ] && env PBS_NODEFILE="${GF[grayscott]}" ${e5env} E5_CELLS="${gs_cells# }" \
    E5_STEPS_grayscott=${E5S_GS_STEPS:-8} E5_GS_REPEAT=1 E5_GS_EXTRA="${E5S_GS_CKPT_ARGS:---ckpt-final}" \
    bash "${B}/pbs_e5_aurora.sh" > "${OUT}/grayscott.log" 2>&1 &
e4env="E4_OUTROOT=${OUT} BENCH_CAP=${CAP} DATA_MB=32768 TIER_BUDGET_MB=200000 HBM_MB=4096 BENCH_GROUP_N=${N}"
[ -n "${GF[gmx]:-}" ] && [ -n "${gx_cells}" ] && env PBS_NODEFILE="${GF[gmx]}" ${e4env} BENCH_CELLS="${gx_cells# }" \
    E5B_GMX_BLOCKS=${GX_BLK} E5B_GMX_PASSES=${E5S_GMX_PASSES:-12} \
    bash "${B}/pbs_e4_batch_aurora.sh" > "${OUT}/gmx.log" 2>&1 &
[ -n "${GF[lbann]:-}" ] && [ -n "${lb_cells}" ] && env PBS_NODEFILE="${GF[lbann]}" ${e4env} BENCH_CELLS="${lb_cells# }" \
    E5B_LB_HIDDEN=${LB_H} E5B_LB_STEPS=${E5S_LB_STEPS:-15} \
    bash "${B}/pbs_e4_batch_aurora.sh" > "${OUT}/lbann.log" 2>&1 &
wait
echo "E5S DONE"
