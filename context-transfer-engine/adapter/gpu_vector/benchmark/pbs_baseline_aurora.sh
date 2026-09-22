#!/usr/bin/env bash
#PBS -l select=4
#PBS -l place=scatter
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A IOWarp
#PBS -j oe
#
# Run ONE Aurora baseline edition (MPI / oneCCL / Intel SHMEM, see
# sycl_baseline/gv_comm.h) across the allocation, one rank per node, one
# tile per rank -- the same shape the Eternia two-node and tiering jobs use,
# so a baseline number and an Eternia number differ only in the substrate.
#
#   BENCH_NAME   workload (kmeans, grayscott, gmx, lbann, lammps_md)
#   BENCH_SUB    mpi | ccl | ishmem, or several space-separated, run in
#                sequence inside the one allocation
#   BENCH_ARGS   the edition's own arguments (global sizes; each edition
#                splits them across ranks)
#   BENCH_CAP    per-rank timeout in seconds (default 240: the plan's
#                five-minute cap with room for launch and teardown). The
#                walltime is 15 minutes so one substrate hitting its cap
#                does not starve the ones after it.
#
# Nothing of the clio runtime is started: these editions link nothing from
# clio, which is what makes them baselines.
set -u

: "${BENCH_NAME:?set BENCH_NAME}"
: "${BENCH_SUB:?set BENCH_SUB (mpi|ccl|ishmem, space-separated for several)}"
: "${BENCH_ARGS:=}"
BENCH_CAP=${BENCH_CAP:-240}
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
NRANKS=$(sort -u "${PBS_NODEFILE}" | wc -l)

echo "=== ${BENCH_NAME} [${BENCH_SUB}] x${NRANKS} nodes: $(sort -u "$PBS_NODEFILE" | tr '\n' ' ') ==="
echo "args: ${BENCH_ARGS}"

# One tile per rank, as the Eternia jobs. The oneAPI tree the login shell
# exports is what the ranks need too (libccl.so by rpath, MPICH by rpath);
# ISHMEM is static. --envall carries the rest.
case "${BENCH_ZE_MASK:-0.0}" in
  none) ;;
  *)    export ZE_AFFINITY_MASK="${BENCH_ZE_MASK:-0.0}" ;;
esac
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
mkdir -p "$SYCL_CACHE_DIR"
# GPU-aware MPI for the MPI edition (MPICH --with-ze); oneCCL wants it too.
export MPIR_CVAR_ENABLE_GPU=${MPIR_CVAR_ENABLE_GPU:-1}
# ISHMEM's symmetric heap: every buffer a collective or a put names lives
# there, and the default is far below a per-node deck -- but the heap is
# RESERVED at init, so it competes with the plain-USM bulk data for the
# tile's 64 GB (kmeans at 32 GB/rank plus a 40 GB heap was
# UR_RESULT_ERROR_OUT_OF_RESOURCES). 16 GB by default; grayscott, whose
# fields are all symmetric, passes a larger value through the job's
# environment (BENCH_EXTRA="ISHMEM_SYMMETRIC_SIZE=...").
export ISHMEM_SYMMETRIC_SIZE=${ISHMEM_SYMMETRIC_SIZE:-17179869184}
export BENCH_RANK_ARGS="${BENCH_ARGS}"
export BENCH_RANK_CAP="${BENCH_CAP}"
CPU_BIND=${BENCH_CPU_BIND:-none}

# EVERY SUBSTRATE IN ONE ALLOCATION: the queue wait dwarfs these runs, so a
# job carries the whole substrate list and reports one RESULT line each.
worst=0
for sub in ${BENCH_SUB//:/ }; do
  EXE="${ROOT}/build-spike/clio_${BENCH_NAME}_${sub}_bench"
  RUNDIR="${ROOT}/build-spike/runbl_${sub}_${BENCH_NAME}"
  echo "--- ${BENCH_NAME}/${sub}: ${EXE} ---"
  if [ ! -x "${EXE}" ]; then
    echo "RESULT ${BENCH_NAME}/${sub}x${NRANKS}: NO-EXECUTABLE"; worst=2; continue
  fi
  mkdir -p "${RUNDIR}"
  rm -f "${RUNDIR}"/rank*.log
  export BENCH_RANK_EXE="${EXE}"
  export BENCH_RANK_DIR="${RUNDIR}"
  start=$SECONDS
  mpiexec -n "${NRANKS}" --ppn 1 --envall --cpu-bind "${CPU_BIND}" bash -c '
    r=${PALS_RANKID:-${PMI_RANK:-0}}
    cd "$BENCH_RANK_DIR"
    ulimit -c 0
    timeout --signal=TERM --kill-after=10s "$BENCH_RANK_CAP" \
      stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS > "rank$r.log" 2>&1
    rc=$?
    echo "rank $r on $(hostname) exit=$rc" >> "rank$r.log"
    # ALWAYS 0 HERE: PALS tears the job down the moment one rank exits
    # non-zero, and the others then never write their exit lines. The
    # result is derived from the per-rank lines instead.
    exit 0
  '
  mrc=$?
  echo "elapsed $((SECONDS - start))s, mpiexec exit=${mrc}"
  rc=0
  for r in $(seq 0 $((NRANKS - 1))); do
    echo "----- rank ${r} -----"
    tail -14 "${RUNDIR}/rank${r}.log" 2>/dev/null
    rrc=$(grep -oE "^rank ${r} on .* exit=[0-9]+" "${RUNDIR}/rank${r}.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
    [ -z "${rrc}" ] && rrc=99
    [ "${rrc}" -gt "${rc}" ] && rc=${rrc}
  done
  case "${rc}" in
    0)   echo "RESULT ${BENCH_NAME}/${sub}x${NRANKS}: OK" ;;
    124) echo "RESULT ${BENCH_NAME}/${sub}x${NRANKS}: TIMEOUT (a rank exceeded the ${BENCH_CAP}s cap)" ;;
    *)   echo "RESULT ${BENCH_NAME}/${sub}x${NRANKS}: FAILED rc=${rc}" ;;
  esac
  [ "${rc}" -gt "${worst}" ] && worst=${rc}
done
echo "RESULT ${BENCH_NAME}x${NRANKS}: $([ "${worst}" -eq 0 ] && echo OK || echo "FAILED rc=${worst}")"
exit "${worst}"
