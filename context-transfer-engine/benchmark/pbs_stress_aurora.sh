#!/usr/bin/env bash
#PBS -l select=16
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A IOWarp
#PBS -j oe
#
# clio_cte_vector_stress across the allocation: one rank per node, each
# embedding its runtime, the same runtime config the gpu_vector benchmarks
# use (bench_config.sh), fail-fast on the first failed rank.
#
#   STRESS_ARGS      benchmark arguments (default below: 8 GB per node of 1 MB
#                    pages, 32 threads, batches of 16, halo 1, 2 random reads
#                    per page, 6 steps, a checkpoint every 3)
#   STRESS_OUT       output root (default flare llogan_e1/stress_<jobid>)
#   STRESS_TIER_MB   DRAM tier per node (default 200000)
#   STRESS_CAP       per-rank cap in seconds (default 1500)
set -u
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
BENCH=${ROOT}/context-transfer-engine/adapter/gpu_vector/benchmark
# shellcheck source=../adapter/gpu_vector/benchmark/bench_config.sh
source "${BENCH}/bench_config.sh"
JOBTAG=${PBS_JOBID%%.*}
NRANKS=$(sort -u "${PBS_NODEFILE}" | wc -l)
OUT=${STRESS_OUT:-/lus/flare/projects/IOWarp/llogan_e1/stress_${JOBTAG}}
EXE=${STRESS_EXE:-${ROOT}/build-fresh/bin/clio_cte_vector_stress}
ARGS=${STRESS_ARGS:-"--pages-per-node 8192 --page-kb 1024 --threads 32 --batch 16 --halo 1 --stream 2 --steps 6 --ckpt-every 3"}
CAP=${STRESS_CAP:-1500}
mkdir -p "${OUT}"
sort -u "${PBS_NODEFILE}" > "${OUT}/hostfile"
BENCH_TIERS="ram::cte_stress_dram|ram|${STRESS_TIER_MB:-200000}MB|0.5" \
  bench_clio_yaml "${OUT}/clio.yaml" "${OUT}/hostfile" 9460
bench_check_binary "${EXE}" || exit 2
echo "=== cte vector stress: ${NRANKS} nodes, ${ARGS}, out ${OUT} ==="
start=$SECONDS
BENCH_RANK_EXE="${EXE}" BENCH_RANK_ARGS="${ARGS}" BENCH_RANK_DIR="${OUT}" \
BENCH_RANK_CAP="${CAP}" BENCH_RANK_N="${NRANKS}" CLIO_SERVER_CONF="${OUT}/clio.yaml" \
mpiexec -n "${NRANKS}" --ppn 1 --hosts "$(sort -u "${PBS_NODEFILE}" | paste -sd,)" \
        --envall --cpu-bind none bash -c '
  r=${PALS_RANKID:-${PMI_RANK:-0}}
  cd "$BENCH_RANK_DIR"
  ulimit -c 0
  timeout --foreground --signal=TERM --kill-after=10s "$BENCH_RANK_CAP" \
    stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS --nodes "$BENCH_RANK_N" --node "$r" \
    > "rank$r.log" 2>&1
  rc=$?
  echo "rank $r on $(hostname) exit=$rc" >> "rank$r.log"
  exit 0
' &
bench_watch_ranks "${OUT}" $! "${NRANKS}"
echo "elapsed $((SECONDS - start))s"
worst=0
for f in "${OUT}"/rank*.log; do
  rrc=$(grep -oE 'exit=[0-9]+$' "${f}" | tail -1 | cut -d= -f2); [ -z "${rrc}" ] && rrc=99
  [ "${rrc}" -gt "${worst}" ] && worst=${rrc}
done
echo "--- per-rank result lines"
grep -ah '^STRESS ' "${OUT}"/rank*.log | cut -c1-260 | head -"${NRANKS}"
echo "--- first errors"
grep -ah 'STRESS ERROR' "${OUT}"/rank*.log | head -20
echo "RESULT stress x${NRANKS}: worst exit=${worst}"
