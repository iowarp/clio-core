#!/usr/bin/env bash
#PBS -l place=scatter
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -A IOWarp
#PBS -j oe
#
# E1 AT PRODUCTION SCALE: one rung of the scaling ladder, all four editions,
# in one allocation.
#
# The node count comes from `qsub -l select=N` (submit_e1_scale_aurora.sh),
# because `prod` is a routing queue with a 256-node floor and the rung size
# is the experiment. 256 / 320 / 384 / 448 / 512 in steps of 64.
#
# WHY ONLY kmeans AND grayscott. The other editions cannot be cut at these
# counts: gmx needs its mesh side K divisible by the node count, lbann needs
# both layer widths divisible by it and by the block count, and lammps_md
# needs its bin count divisible by it AND a multiple of 64 for page
# alignment. None of those hold at 320 or 448 without absurd deck sizes.
# kmeans splits a point list and grayscott splits z-planes, so both divide
# evenly at every rung.
#
# WEAK SCALING, 4 GB PER NODE. The deck grows with the rung (data_mb =
# 4096 * nodes), so a flat curve is the result the claim predicts: each node
# exchanges with its neighbours, not with the cluster, so time per step
# should not grow with N. grayscott's nz works out to 1024 planes per node
# at every rung, which is what keeps it divisible.
#
# I/O IS STRUCTURALLY DISABLED for the Eternia arm: its config has a DRAM
# tier and nothing else, so there is no storage for a fault to reach. The
# frame cache is also sized to hold the node's whole share, which is what the
# plan means by a resident deck.
#
#   BENCH_NODES    the rung, for labelling (the real count comes from PBS)
#   BENCH_ITERS    iterations/steps per run (default 20)
#   BENCH_CAP      per-rank cap in seconds (default 900)
set -u

ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
ITERS=${BENCH_ITERS:-20}
CAP=${BENCH_CAP:-900}
NRANKS=$(sort -u "${PBS_NODEFILE}" | wc -l)
PERNODE_MB=${BENCH_PERNODE_MB:-4096}
DATA_MB=$(( PERNODE_MB * NRANKS ))
JOBTAG=${PBS_JOBID%%.*}

echo "=== E1 rung: ${NRANKS} nodes, ${PERNODE_MB} MB/node, ${DATA_MB} MB total, ${ITERS} iters ==="

export IGC_FunctionControl=3
export ZE_AFFINITY_MASK=${BENCH_ZE_MASK:-0.0}
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
mkdir -p "$SYCL_CACHE_DIR"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# ISHMEM's symmetric heap must hold every collective buffer; the shard is
# 4 GB, so 8 GB leaves room for the gathers on top of it.
export ISHMEM_SYMMETRIC_SIZE=${ISHMEM_SYMMETRIC_SIZE:-8589934592}
export MPIR_CVAR_ENABLE_GPU=1

# ---- one run, reported on its own RESULT line -----------------------------
# @param 1 label used in the RESULT line and the run directory
# @param 2 executable
# @param 3 argument string (the rank flags are appended)
# @param 4 1 if the binary takes --nodes/--node, 0 if it is an MPI baseline
run_one() {
  local label=$1 exe=$2 args=$3 ranked=$4
  local rundir="${ROOT}/build-spike/e1_${JOBTAG}_${label}"
  if [ ! -x "${exe}" ]; then
    echo "RESULT e1/${label}x${NRANKS}: NO-EXECUTABLE"
    return 2
  fi
  mkdir -p "${rundir}"
  rm -f "${rundir}"/rank*.log
  echo "--- ${label}: ${args} ---"
  local start=$SECONDS
  BENCH_RANK_EXE="${exe}" BENCH_RANK_ARGS="${args}" BENCH_RANK_DIR="${rundir}" \
  BENCH_RANK_CAP="${CAP}" BENCH_RANK_N="${NRANKS}" BENCH_RANK_RANKED="${ranked}" \
  mpiexec -n "${NRANKS}" --ppn 1 --envall --cpu-bind none bash -c '
    r=${PALS_RANKID:-${PMI_RANK:-0}}
    cd "$BENCH_RANK_DIR"
    ulimit -c 0
    extra=""
    [ "$BENCH_RANK_RANKED" = 1 ] && extra="--nodes $BENCH_RANK_N --node $r"
    timeout --signal=TERM --kill-after=10s "$BENCH_RANK_CAP" \
      stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS $extra \
      > "rank$r.log" 2>&1
    rc=$?
    echo "rank $r on $(hostname) exit=$rc" >> "rank$r.log"
    exit 0
  '
  local rc=0 r rrc
  echo "elapsed $((SECONDS - start))s"
  # Only ranks 0 and 1 are printed: at 512 nodes the full dump would bury the
  # RESULT lines. Every rank still contributes its exit code below.
  for r in 0 1; do
    echo "----- rank ${r} -----"
    grep -vE "LoadBalance|\[#78[15]|INFO|SUCCESS|WARNING" "${rundir}/rank${r}.log" \
      2>/dev/null | tail -18
  done
  for (( r = 0; r < NRANKS; ++r )); do
    rrc=$(grep -oE "^rank ${r} on .* exit=[0-9]+" "${rundir}/rank${r}.log" \
          2>/dev/null | tail -1 | grep -oE "[0-9]+$")
    [ -z "${rrc}" ] && rrc=99
    [ "${rrc}" -gt "${rc}" ] && rc=${rrc}
  done
  case "${rc}" in
    0)   echo "RESULT e1/${label}x${NRANKS}: OK" ;;
    124) echo "RESULT e1/${label}x${NRANKS}: TIMEOUT (a rank exceeded the ${CAP}s cap)" ;;
    *)   echo "RESULT e1/${label}x${NRANKS}: FAILED rc=${rc}" ;;
  esac
  return "${rc}"
}

# ---- the Eternia config: a DRAM tier and nothing else ----------------------
et_conf() {
  local rundir=$1 cap_mb=$2
  sort -u "${PBS_NODEFILE}" > "${rundir}/hostfile"
  cat > "${rundir}/clio_e1.yaml" <<EOF
networking:
  port: 9460
  hostfile: "${rundir}/hostfile"

# SWIM OFF. Failure detection is orthogonal to what this study measures and
# is actively harmful here. Its suspicion timeout is 60 s and expiry runs
# TriggerRecovery, which redistributes a live node's containers. At 256
# nodes the 256-way compose starves probe replies long enough to cross that
# threshold, so nodes that are merely busy get declared dead, their
# containers are redistributed, routing then answers Dne (container does not
# exist) and the cluster never converges -- the 256 rung sat for 900 s
# without completing one iteration. On a healthy batch allocation no node is
# going to fail mid-run; if one does, the job dies anyway.
swim:
  enabled: false

runtime:
  num_threads: 8
  queue_depth: 8192
  first_busy_wait: 10000000

gpu:
  queue_depth: 8192

compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "1GB"

  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "512.0"
    targets:
      neighborhood: 1
    storage:
      - path: "ram::gv_e1_dram"
        bdev_type: "ram"
        capacity_limit: "${cap_mb}MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
EOF
}

worst=0
for wl in kmeans grayscott; do
  case "${wl}" in
    kmeans)    base_args="--data-mb ${DATA_MB} --iters ${ITERS} --dims 32 --clusters 16 --blocks 1024 --threads 256" ;;
    grayscott) base_args="--data-mb ${DATA_MB} --steps ${ITERS} --page-kb 1024" ;;
  esac
  for sub in mpi ccl ishmem; do
    # NO `|| true` HERE. It makes $? the status of `true`, so every rung
    # summarised as OK however many cells failed -- which is exactly what
    # the 256 rung did with two of its eight cells broken. run_one already
    # never aborts the script (it returns, it does not exit), so the guard
    # was never needed.
    rc=0
    run_one "${wl}_${sub}" "${ROOT}/build-spike/clio_${wl}_${sub}_bench" \
            "${base_args}" 0 || rc=$?
    [ "${rc}" -gt "${worst}" ] && worst=${rc}
  done
  # The Eternia arm: 80 frames per block of 1 MB over 64 blocks is 5 GB of
  # frame cache against a 4 GB share, so the shard is resident.
  et_rundir="${ROOT}/build-spike/e1_${JOBTAG}_${wl}_eternia"
  mkdir -p "${et_rundir}"
  et_conf "${et_rundir}" $(( PERNODE_MB + 2048 ))
  export CLIO_SERVER_CONF="${et_rundir}/clio_e1.yaml"
  case "${wl}" in
    kmeans)    et_args="--data-mb ${DATA_MB} --hbm-mb ${PERNODE_MB} --iters ${ITERS} --page-kb 1024 --slots 80" ;;
    grayscott) et_args="--data-mb ${DATA_MB} --hbm-mb ${PERNODE_MB} --steps ${ITERS} --repeat 1 --page-kb 1024 --slots 80" ;;
  esac
  rc=0
  run_one "${wl}_eternia" "${ROOT}/build-spike/clio_${wl}_paged_newcoro_aot" \
          "${et_args}" 1 || rc=$?
  [ "${rc}" -gt "${worst}" ] && worst=${rc}
  unset CLIO_SERVER_CONF
done

echo "RESULT e1x${NRANKS}: $([ "${worst}" -eq 0 ] && echo OK || echo "FAILED rc=${worst}")"
echo "E1 RUNG DONE ${NRANKS}"
exit 0
