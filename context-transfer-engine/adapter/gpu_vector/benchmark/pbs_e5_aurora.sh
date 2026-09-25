#!/usr/bin/env bash
#PBS -l select=256
#PBS -l place=scatter
#PBS -l walltime=02:30:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -A IOWarp
#PBS -j oe
#
# E5 -- DYNAMIC MEMORY REDUCTION, over the E1 kernels, at full allocation
# width (one GPU tile per node, every node in one run per cell).
#
# The HBM BUDGET is the Eternia frame cache: slots x blocks x page. That is
# all the device memory the vector may hold; the deck lives in a per-node
# DRAM tier behind it, so every rung below the per-node deck is out of core
# and pages move through Fetch/Flush. Written data is flushed back (grayscott
# --ooc: each band's outputs are flushed asynchronously on release), so any
# page can be evicted and refetched.
#
#   E5_CELLS        "<workload>:<budget GB> ..." (default: kmeans and
#                   grayscott at 32 24 16 8 4)
#   E5_PERNODE_MB   deck per node (default 256000, the plan's 256 GB)
#   E5_TIER_MB      DRAM tier per node (default 450000)
#   E5_CAP          per-rank cap in seconds (default 1500)
#   E5_KM_REPEAT    kmeans timed repetitions (binary default 3, best reported)
#   E5_RETRY        reruns of a failed cell (default 0)
#   E5_ITERS_kmeans / E5_STEPS_grayscott   (default 8 / 8)
#   E5_SFX_kmeans / E5_SFX_grayscott       binary suffixes
#
# Energy: each rank reads its GPU's hwmon energy counters (microjoules)
# before and after the run; the cell line reports the sum over ranks.
set -u
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
JOBTAG=${PBS_JOBID%%.*}
NRANKS=$(sort -u "${PBS_NODEFILE}" | wc -l)
PERNODE_MB=${E5_PERNODE_MB:-256000}
DATA_MB=$(( PERNODE_MB * NRANKS ))
TIER_MB=${E5_TIER_MB:-450000}
CAP=${E5_CAP:-1500}
CELLS=${E5_CELLS:-"kmeans:32 kmeans:24 kmeans:16 kmeans:8 kmeans:4 grayscott:32 grayscott:24 grayscott:16 grayscott:8 grayscott:4"}

export ZE_AFFINITY_MASK=${BENCH_ZE_MASK:-0.0}
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# The #628 task-progress probe answers "Gone" for a replica whose response is
# still in flight (or not yet received) on a backlogged node; the origin then
# fails a healthy writeback with a network-timeout RC (E5 16 nodes: REFUSED
# writebacks, 8865932). The per-rank cap bounds real hangs instead.
export CLIO_TASK_PROGRESS_INTERVAL_MS=${CLIO_TASK_PROGRESS_INTERVAL_MS:-0}

echo "=== E5: ${NRANKS} nodes, ${PERNODE_MB} MB/node deck (${DATA_MB} MB), DRAM tier ${TIER_MB} MB/node ==="
echo "    cells: ${CELLS}"

# ---- per-cell runtime config: one DRAM tier, SWIM off (see E1) -----------
e5_conf() {
  local rundir=$1
  sort -u "${PBS_NODEFILE}" > "${rundir}/hostfile"
  cat > "${rundir}/clio_e5.yaml" <<EOF
networking:
  port: 9460
  hostfile: "${rundir}/hostfile"
  # >= node count: a Broadcast wider than neighborhood_size is split into
  # multi-node Range queries that the receiving node runs only locally (a
  # runtime bug), so at 64 nodes pool creation reached nodes 0 and 32 only
  # and 62 nodes had no CTE target. Sized past the allocation, every
  # broadcast becomes one single-node query per node (the 24-node path).
  neighborhood_size: 1024
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
      - path: "ram::gv_e5_dram"
        bdev_type: "ram"
        capacity_limit: "${TIER_MB}MB"
        score: 0.5
    dpe:
      dpe_type: "max_bw"
EOF
}

# ---- the deck for one cell ------------------------------------------------
# @param 1 workload  @param 2 budget in GB
# Sets EXE, ARGS, FC (IGC function control).
cell_deck() {
  local wl=$1 gb=$2
  case "${wl}" in
    kmeans)
      # 1024 work-groups; budget GB = slots x 1024 x 1 MB.
      EXE="${ROOT}/build-spike/clio_kmeans_paged_newcoro_aot${E5_SFX_kmeans:-_x_ct}"
      ARGS="--data-mb ${DATA_MB} --iters ${E5_ITERS_kmeans:-8} --page-kb 1024 --blocks 1024 --threads 256 --slots ${gb} --publish-seed${E5_KM_REPEAT:+ --repeat ${E5_KM_REPEAT}}"
      FC=3 ;;
    grayscott)
      # 512 work-groups so the 4 GB rung still has 8 slots per group
      # (--ooc needs 6); budget GB = slots x 512 x 1 MB.
      EXE="${ROOT}/build-spike/clio_grayscott_paged_newcoro_aot${E5_SFX_grayscott:-_x_fc2_ooc1}"
      ARGS="--data-mb ${DATA_MB} --steps ${E5_STEPS_grayscott:-8} --page-kb 1024 --blocks 512 --threads 256 --slots $(( gb * 2 )) --two-phase --ooc${E5_GS_REPEAT:+ --repeat ${E5_GS_REPEAT}}"
      FC=2 ;;
    *) EXE=""; ARGS=""; FC=3 ;;
  esac
}

# ---- one cell --------------------------------------------------------------
run_cell() {
  local wl=${1%%:*} gb=${1#*:}
  local label="${wl}_hbm${gb}"
  local rundir="${E5_RUNROOT:-${ROOT}/build-spike}/e5_${JOBTAG}_${label}"
  cell_deck "${wl}" "${gb}"
  if [ -z "${EXE}" ] || [ ! -x "${EXE}" ]; then
    echo "RESULT e5/${label}x${NRANKS}: NO-EXECUTABLE (${EXE})"
    return
  fi
  mkdir -p "${rundir}"; rm -f "${rundir}"/rank*.log
  e5_conf "${rundir}"
  echo "--- ${label}: ${ARGS} ---"
  local start=$SECONDS
  IGC_FunctionControl=${FC} CLIO_SERVER_CONF="${rundir}/clio_e5.yaml" \
  BENCH_RANK_EXE="${EXE}" BENCH_RANK_ARGS="${ARGS}" BENCH_RANK_DIR="${rundir}" \
  BENCH_RANK_CAP="${CAP}" BENCH_RANK_N="${NRANKS}" \
  mpiexec -n "${NRANKS}" --ppn 1 --envall --cpu-bind none bash -c '
    r=${PALS_RANKID:-${PMI_RANK:-0}}
    cd "$BENCH_RANK_DIR"
    ulimit -c 0
    en() { cat /sys/class/drm/card0/device/hwmon/hwmon*/energy1_input 2>/dev/null | tr "\n" " "; }
    e0=$(en)
    timeout --signal=TERM --kill-after=10s "$BENCH_RANK_CAP" \
      stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS --nodes "$BENCH_RANK_N" --node "$r" \
      > "rank$r.log" 2>&1
    rc=$?
    e1=$(en)
    echo "ENERGY_UJ before: ${e0} after: ${e1}" >> "rank$r.log"
    echo "rank $r on $(hostname) exit=$rc" >> "rank$r.log"
    exit 0
  '
  echo "elapsed $((SECONDS - start))s"
  for r in 0 1; do
    echo "----- rank ${r} -----"
    grep -aE "mode=paged|out of core|COMM |ERROR|FATAL|ENERGY_UJ|exit=" "${rundir}/rank${r}.log" |
      grep -v "INFO\|RouteTask" | tail -12
  done
  # Slowest rank's ms and the energy summed over ranks (first hwmon = card).
  local worst=0 msmax=0 ej=0 f rrc ms b a
  for f in "${rundir}"/rank*.log; do
    rrc=$(grep -oE "exit=[0-9]+$" "${f}" | tail -1 | cut -d= -f2)
    [ -z "${rrc}" ] && rrc=99
    [ "${rrc}" -gt "${worst}" ] && worst=${rrc}
    ms=$(grep -aoE "mode=paged.* ms=[0-9.]+" "${f}" | tail -1 | grep -oE " ms=[0-9.]+" | cut -d= -f2)
    [ -n "${ms}" ] && msmax=$(awk -v a="${msmax}" -v b="${ms}" 'BEGIN{print (b>a)?b:a}')
    read -r b _ <<< "$(grep -a "ENERGY_UJ" "${f}" | sed 's/.*before: //; s/ after:.*//')"
    read -r a _ <<< "$(grep -a "ENERGY_UJ" "${f}" | sed 's/.*after: //')"
    [ -n "${b:-}" ] && [ -n "${a:-}" ] && ej=$(awk -v e="${ej}" -v x="${a}" -v y="${b}" 'BEGIN{printf "%.0f", e + (x - y) / 1e6}')
  done
  LAST_RC=${worst}; LAST_DIR=${rundir}
  echo "E5CELL ${wl} hbm_gb=${gb} nodes=${NRANKS} ms=${msmax} energy_J=${ej} rc=${worst}"
  case "${worst}" in
    0)   echo "RESULT e5/${label}x${NRANKS}: OK" ;;
    124) echo "RESULT e5/${label}x${NRANKS}: TIMEOUT" ;;
    *)   echo "RESULT e5/${label}x${NRANKS}: FAILED rc=${worst}" ;;
  esac
}

# E5_RETRY: rerun a failed cell up to N more times (16 nodes: an intermittent
# single-rank device fault takes down a whole cell). A failed attempt's
# logs are kept as <rundir>.fail<k>.
for c in ${CELLS}; do
  run_cell "${c}"
  k=0
  while [ "${LAST_RC}" != 0 ] && [ "${k}" -lt "${E5_RETRY:-0}" ]; do
    k=$((k + 1)); mv "${LAST_DIR}" "${LAST_DIR}.fail${k}"
    echo "RETRY ${c} attempt $((k + 1))"
    run_cell "${c}"
  done
done
echo "E5 DONE"
exit 0
