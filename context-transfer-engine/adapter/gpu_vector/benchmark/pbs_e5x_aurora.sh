#!/usr/bin/env bash
#PBS -l select=4
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A IOWarp
#PBS -j oe
#
# E5 for gmx, lbann and lammps_md, built ONLY on decks that already ran:
# gmx and lbann use the E1 decks (pbs_e1_scale_aurora.sh, Eternia arm only)
# with the cache shrunk by BENCH_E5_DIV; lammps_md uses the E4 deck
# (pbs_e4_batch_aurora.sh, ballistic gate) with --slots lowered.
#   E5X_GMX    "<blocks>:<div> ..."   (default "1024:1 64:2 32:4")
#   E5X_LBANN  "<div> ..."            (default "1 2 4 8")
#   E5X_MD     "<slots> ..." ("res" = the deck's resident default)
#   E5X_ITERS_gmx / E5X_ITERS_lbann  passes / steps (default 20 / 5)
set -u
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
B=${ROOT}/context-transfer-engine/adapter/gpu_vector/benchmark
OUT=${E5X_OUT:-/lus/flare/projects/IOWarp/llogan_e1/e5x}
mkdir -p "${OUT}"
export ROOT SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-/lus/flare/projects/IOWarp/llogan_e1/sycl_cache}
export CLIO_TASK_PROGRESS_INTERVAL_MS=0
common="BENCH_RUNROOT=${OUT} BENCH_PERNODE_MB=32000 BENCH_SUBS= BENCH_BL_SUFFIX=_c BENCH_CAP=${E5X_CAP:-600}"
for c in ${E5X_GMX-1024:1 64:2 32:4}; do
  echo "##### gmx blocks ${c%%:*} div ${c#*:}"
  env ${common} BENCH_WORKLOADS=gmx BENCH_BLOCKS=${c%%:*} BENCH_E5_DIV=${c#*:} \
      BENCH_ITERS=${E5X_ITERS_gmx:-20} BENCH_ET_SUFFIX_gmx=_x_ct4 bash "${B}/pbs_e1_scale_aurora.sh"
done
for d in ${E5X_LBANN-1 2 4 8}; do
  echo "##### lbann div ${d}"
  env ${common} BENCH_WORKLOADS=lbann BENCH_BLOCKS=1024 BENCH_E5_DIV=${d} \
      BENCH_ITERS=${E5X_ITERS_lbann:-5} BENCH_ET_SUFFIX_lbann=_x_ct10 bash "${B}/pbs_e1_scale_aurora.sh"
done
cells=""
for s in ${E5X_MD-res 32 8}; do
  if [ "${s}" = res ]; then cells="${cells} lammps_md:dram100"; else cells="${cells} lammps_md:dram100:1024:${s}"; fi
done
if [ -n "${cells}" ]; then
  echo "##### lammps_md cells:${cells}"
  env BENCH_CELLS="${cells# }" E4_OUTROOT=${OUT} BENCH_CAP=${E5X_CAP:-600} DATA_MB=32768 \
      TIER_BUDGET_MB=100000 HBM_MB=4096 BENCH_GROUP_N=4 bash "${B}/pbs_e4_batch_aurora.sh"
fi
echo "E5X DONE"
