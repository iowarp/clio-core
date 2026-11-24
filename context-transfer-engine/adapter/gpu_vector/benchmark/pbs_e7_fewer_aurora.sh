#!/usr/bin/env bash
#PBS -l place=scatter
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -l daos=daos_user
#PBS -q debug-scaling
#PBS -A IOWarp
#PBS -j oe
#
# E7 -- THE SAME WORKLOAD ON FEWER NODES.
#
# Every other study here grows the problem with the machine. This one does
# the opposite: the TOTAL footprint is fixed and the node count falls, so the
# per-node share rises and the question is whether the job still runs at all.
# That is the practical claim for tiering -- not that it is fast, but that it
# removes the "you need N nodes or you cannot start" floor.
#
#   480 GB total, in every rung:
#
#     32 nodes -> 15 GB/node      12 nodes -> 40 GB/node
#     16 nodes -> 30 GB/node       8 nodes -> 60 GB/node
#      6 nodes -> 80 GB/node       4 nodes -> 120 GB/node
#
# WHY 480 AND NOT 512. grayscott's z-planes must divide the node count, and
# nz = data_mb/4. 480 GB gives nz = 122880 = 96 x 1280, which divides 4, 6,
# 8, 12, 16 and 32 exactly. 512 GB gives nz = 131072 = 2^17, which does not
# divide 6 or 12, and the edition would refuse those two rungs.
#
# THE FRAME CACHE IS FIXED AT 8 GB PER NODE in every rung, so the only thing
# that changes is how far the deck overflows it: 1.9x at 32 nodes, 15x at 4.
#
# THE BASELINES CANNOT FOLLOW ALL THE WAY DOWN, and that is the point. They
# hold the shard in HBM, and a PVC tile has 64 GB, so a share above ~48 GB
# has nowhere to live. They run at 32, 16 and 12 nodes and are skipped below
# that with the reason printed -- skipped, not failed, because burning the
# per-rank cap on a run that cannot fit teaches nothing.
#
#   BENCH_TOTAL_MB  total footprint (default 491520 = 480 GB)
#   BENCH_STEPS     steps per run (default 2)
#   BENCH_CAP       per-rank cap in seconds (default 900)
#   BENCH_PPN       ranks per node, one per GPU (default 1). With more than
#                   one, rank r runs on tile 0 of GPU (r mod PPN) and the
#                   node's ranks share ONE clio runtime: every Eternia rank
#                   runs with CLIO_WITH_RUNTIME=1, the first to bind the port
#                   becomes the runtime and the rest attach as clients.
#   BENCH_CACHE_MB  Eternia frame cache PER RANK (default 8192)
set -u

ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
TOTAL_MB=${BENCH_TOTAL_MB:-491520}
STEPS=${BENCH_STEPS:-2}
CAP=${BENCH_CAP:-900}
CACHE_MB=${BENCH_CACHE_MB:-8192}
PPN=${BENCH_PPN:-1}
NNODES=$(sort -u "${PBS_NODEFILE}" | wc -l)
NRANKS=$(( NNODES * PPN ))
PERNODE_MB=$(( TOTAL_MB / NNODES ))
PERRANK_MB=$(( TOTAL_MB / NRANKS ))
JOBTAG=${PBS_JOBID%%.*}
DAOS_POOL=${DAOS_POOL:-IOWarp}
DAOS_CONT=${DAOS_CONT:-clio_tier}
FLARE_ROOT="/lus/flare/projects/IOWarp/clio_tier/${JOBTAG}"
# THE PERSISTENT SHARE, NOT THE TOTAL BUDGET, IS WHAT MUST COVER THE DATA.
# The first 32-node rung used a budget of 1.25x the share split 25/50/25,
# which leaves DAOS+Flare holding 0.9375x -- just under. The shutdown flush
# then declined blob after blob ("persistent tier(s) at level >= 1"), the
# full tiers left dirty frames unevictable, and the fault path livelocked:
# "ROUND CAP HIT after 2000000 rounds with 32 block(s) still suspended".
# One arithmetic slip, two symptoms. At 1.6x the persistent tiers hold 1.2x
# the share and there is room to write back.
BUDGET_MB=$(( PERNODE_MB * 8 / 5 ))
DRAM_MB=$(( BUDGET_MB / 4 ))
DAOS_MB=$(( BUDGET_MB / 2 ))
FLARE_MB=$(( BUDGET_MB - DRAM_MB - DAOS_MB ))
# 64 blocks x 1 MB pages, so slots = cache in MB / 64.
SLOTS=$(( CACHE_MB / 64 ))

echo "=== E7 rung: ${NNODES} nodes x ${PPN} GPUs = ${NRANKS} ranks, ${TOTAL_MB} MB total, ${PERNODE_MB} MB/node, ${PERRANK_MB} MB/rank ==="
echo "    tiers/node: dram ${DRAM_MB} + daos ${DAOS_MB} + flare ${FLARE_MB} MB"
echo "    frame cache ${CACHE_MB} MB per rank (${SLOTS} slots x 64 blocks x 1 MB), "\
"share/cache = ${PERRANK_MB}/${CACHE_MB}"

export IGC_FunctionControl=3
export ZE_AFFINITY_MASK=${BENCH_ZE_MASK:-0.0}
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
mkdir -p "$SYCL_CACHE_DIR"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export MPIR_CVAR_ENABLE_GPU=1
# ISHMEM's symmetric heap lives IN HBM, and a PVC tile has 64 GB, so asking
# for 64 GB fails outright: zeMemAllocDevice returns UNSUPPORTED_SIZE and
# ishmem_init dies with rc=139 before the benchmark starts (seen at the
# 32-node rung). Size it from the share instead -- 1.35x, which covers the
# collective buffers on top of the four grayscott arrays -- and cap it below
# the tile so the driver has somewhere to put everything else.
#
# THE CAP WAS 40 GB AND THAT IS EXACTLY THE 12-NODE SHARE, so the 12-node
# baseline asked for a heap the size of its own data with nothing left for
# the halo buffers and died before it started (rc=99, no exit line). 48 GB
# leaves 16 GB of the 64 GB tile for the driver and the collectives, which
# is the most that has run. A rung whose share still does not fit under it
# is the baseline floor this study is about, and it fails honestly.
ISHMEM_MB=$(( PERRANK_MB * 27 / 20 ))
[ "${ISHMEM_MB}" -gt 49152 ] && ISHMEM_MB=49152
[ "${ISHMEM_MB}" -lt 8192 ] && ISHMEM_MB=8192
export ISHMEM_SYMMETRIC_SIZE=${ISHMEM_SYMMETRIC_SIZE:-$(( ISHMEM_MB * 1024 * 1024 ))}
echo "    ishmem symmetric heap ${ISHMEM_MB} MB"

source /usr/share/lmod/lmod/init/bash
module use /soft/modulefiles
module load daos/base
daos cont query "${DAOS_POOL}" "${DAOS_CONT}" > /dev/null 2>&1 ||
  daos cont create --type=POSIX "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | tail -2
clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1
launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" 2>&1 | tail -1
MNT="/tmp/${DAOS_POOL}/${DAOS_CONT}"
mount | grep -qF "${MNT}" || { echo "dfuse NOT mounted"; exit 2; }
TDIR_DAOS="${MNT}/clio_tier/${JOBTAG}"
TDIR_FLARE="${FLARE_ROOT}"
mkdir -p "${TDIR_DAOS}" "${TDIR_FLARE}"

# @param 1 label  @param 2 exe  @param 3 args  @param 4 ranked (1 = --nodes/--node)
run_one() {
  local label=$1 exe=$2 args=$3 ranked=$4
  local rundir="${ROOT}/build-spike/e7_${JOBTAG}_${label}"
  if [ ! -x "${exe}" ]; then
    echo "RESULT e7/${label}x${NRANKS}: NO-EXECUTABLE"; return 2
  fi
  mkdir -p "${rundir}"; rm -f "${rundir}"/rank*.log
  echo "--- ${label}: ${args} ---"
  local start=$SECONDS
  BENCH_RANK_EXE="${exe}" BENCH_RANK_ARGS="${args}" BENCH_RANK_DIR="${rundir}" \
  BENCH_RANK_CAP="${CAP}" BENCH_RANK_N="${NRANKS}" BENCH_RANK_RANKED="${ranked}" \
  BENCH_RANK_PPN="${PPN}" \
  mpiexec -n "${NRANKS}" --ppn "${PPN}" --no-vni --envall --cpu-bind none bash -c '
    r=${PALS_RANKID:-${PMI_RANK:-0}}
    lr=${PALS_LOCAL_RANKID:-0}
    nd=$(( r / BENCH_RANK_PPN ))
    cd "$BENCH_RANK_DIR"
    ulimit -c 0
    # One GPU per rank: in FLAT mode tile 0 of GPU k is device 2k.
    if [ "$BENCH_RANK_PPN" -gt 1 ]; then
      export ZE_FLAT_DEVICE_HIERARCHY=FLAT
      export ZE_AFFINITY_MASK=$(( 2 * lr ))
    fi
    extra=""
    [ "$BENCH_RANK_RANKED" = 1 ] && extra="--nodes $BENCH_RANK_N --node $r"
    # The config names the NODE, not the rank: every rank on a node writes the
    # same one, and whichever binds the port first is that node'"'"'s runtime.
    [ -f clio_e7_template.yaml ] &&
      sed "s/__RANK__/$nd/g" clio_e7_template.yaml > "clio_e7_r$r.yaml" &&
      export CLIO_SERVER_CONF="$BENCH_RANK_DIR/clio_e7_r$r.yaml" &&
      export CLIO_WITH_RUNTIME=1
    echo "rank $r node $nd local $lr ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset} CLIO_WITH_RUNTIME=${CLIO_WITH_RUNTIME:-unset}" > "rank$r.log"
    timeout --signal=TERM --kill-after=10s "$BENCH_RANK_CAP" \
      stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS $extra \
      >> "rank$r.log" 2>&1
    rc=$?
    echo "rank $r on $(hostname) exit=$rc" >> "rank$r.log"
    exit 0
  '
  local rc=0 r rrc
  echo "elapsed $((SECONDS - start))s"
  for r in 0 1; do
    [ "${r}" -lt "${NRANKS}" ] || continue
    echo "----- rank ${r} -----"
    grep -vE "LoadBalance|\[#78[15]|INFO|SUCCESS|WARNING" "${rundir}/rank${r}.log" \
      2>/dev/null | tail -16
  done
  for (( r = 0; r < NRANKS; ++r )); do
    rrc=$(grep -oE "^rank ${r} on .* exit=[0-9]+" "${rundir}/rank${r}.log" \
          2>/dev/null | tail -1 | grep -oE "[0-9]+$")
    [ -z "${rrc}" ] && rrc=99
    [ "${rrc}" -gt "${rc}" ] && rc=${rrc}
  done
  case "${rc}" in
    0)   echo "RESULT e7/${label}x${NRANKS}: OK" ;;
    124) echo "RESULT e7/${label}x${NRANKS}: TIMEOUT (${CAP}s cap)" ;;
    *)   echo "RESULT e7/${label}x${NRANKS}: FAILED rc=${rc}" ;;
  esac
  return "${rc}"
}

BASE_ARGS="--data-mb ${TOTAL_MB} --steps ${STEPS} --page-kb 1024"

# ---- the baselines, where the shard still fits a 64 GB tile ---------------
if [ "${PERRANK_MB}" -le 49152 ]; then
  for sub in mpi ccl ishmem; do
    rc=0
    run_one "grayscott_${sub}" "${ROOT}/build-spike/clio_grayscott_${sub}_bench" \
            "${BASE_ARGS}" 0 || rc=$?
  done
else
  echo "RESULT e7/grayscott_baselinesx${NRANKS}: SKIPPED -- the shard is "\
"${PERRANK_MB} MB and a PVC tile holds 64 GB; an in-HBM baseline has nowhere "\
"to put it. This is the floor tiering is meant to remove."
fi

# ---- Eternia, every rung --------------------------------------------------
ET_DIR="${ROOT}/build-spike/e7_${JOBTAG}_grayscott_eternia"
mkdir -p "${ET_DIR}"
sort -u "${PBS_NODEFILE}" > "${ET_DIR}/hostfile"
cat > "${ET_DIR}/clio_e7_template.yaml" <<EOF
networking:
  port: 9460
  hostfile: "${ET_DIR}/hostfile"

# See pbs_e4_batch_aurora.sh: a wide compose starves SWIM's probe replies,
# its 60 s suspicion timeout fires and recovery moves a live node's
# containers, after which routing cannot find them.
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
      - path: "ram::gv_e7_dram"
        bdev_type: "ram"
        capacity_limit: "${DRAM_MB}MB"
        # SCORES MUST BE <= THE VECTOR'S BLOB SCORE (0.5, page.h) TO BE A FIRST
        # CHOICE: MaxBwDpe prefers targets scored at or below the blob and uses
        # the rest only as fallback. DRAM at 1.0 was fallback, so every page
        # went to DAOS first and DRAM only after DAOS and Flare filled.
        score: 0.5
      - path: "${TDIR_DAOS}/node__RANK__.dat"
        bdev_type: "file"
        persistence_level: "long_term"
        capacity_limit: "${DAOS_MB}MB"
        score: 0.3
      - path: "${TDIR_FLARE}/node__RANK__.dat"
        bdev_type: "file"
        persistence_level: "long_term"
        capacity_limit: "${FLARE_MB}MB"
        score: 0.1
    dpe:
      dpe_type: "max_bw"
EOF
rc=0
run_one "grayscott_eternia" \
        "${ROOT}/build-spike/clio_grayscott_paged_newcoro_aot" \
        "${BASE_ARGS} --hbm-mb ${CACHE_MB} --repeat 1 --slots ${SLOTS}" 1 || rc=$?

echo "--- tier files ---"
for d in "${TDIR_DAOS}" "${TDIR_FLARE}"; do du -sh "${d}" 2>&1; rm -rf "${d}"; done
clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1
echo "E7 RUNG DONE ${NRANKS}"
exit 0
