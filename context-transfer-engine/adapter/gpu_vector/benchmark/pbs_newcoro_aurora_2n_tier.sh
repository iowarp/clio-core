#!/usr/bin/env bash
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -l daos=daos_user
#PBS -q debug
#PBS -A IOWarp
#PBS -j oe
#
# TIERED two-node run of one paged benchmark: GPU HBM in front of a
# filesystem tier on Flare (Lustre) or DAOS (a POSIX container mounted with
# dfuse), with every tier smaller than the data so the spill is real.
#
# The two-node script (pbs_newcoro_aurora_2n.sh) reuses each benchmark's
# single-node config; this one WRITES the config, per rank, because the
# storage tiers are the experiment:
#
#   hbm::gv_tier_hbm         bdev_type hbm   capacity BENCH_HBM_MB   score 1.0
#   <TIER_DIR>/node<r>.dat   bdev_type file  capacity BENCH_TIER_MB  score 0.2
#
# PER RANK, not shared: a file bdev's pool name IS its path, so two runtimes
# given one path on a shared filesystem would write the same file. No host
# RAM tier, on purpose -- with one, the spill would land in DRAM and the
# filesystem would never see a byte. `long_term` persistence keeps the tier
# files until this script lists their sizes, then removes them.
#
#   BENCH_NAME, BENCH_ARGS   as for the other scripts; --hbm-mb in BENCH_ARGS
#                            should equal BENCH_HBM_MB (it sizes the vector's
#                            own frame cache)
#   BENCH_TIER               flare | daos
#   BENCH_HBM_MB             HBM tier capacity per node (default 4096)
#   BENCH_TIER_MB            file tier capacity per node (default 4608)
#   DAOS_POOL / DAOS_CONT    default IOWarp / clio_tier
set -u

: "${BENCH_NAME:?set BENCH_NAME}"
: "${BENCH_ARGS:=}"
BENCH_TIER=${BENCH_TIER:-flare}
BENCH_HBM_MB=${BENCH_HBM_MB:-4096}
BENCH_TIER_MB=${BENCH_TIER_MB:-4608}
DAOS_POOL=${DAOS_POOL:-IOWarp}
DAOS_CONT=${DAOS_CONT:-clio_tier}
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
EXE="${ROOT}/build-spike/${BENCH_EXE:-clio_${BENCH_NAME}_paged_newcoro_aot}"
RUNDIR=${RUNDIR:-${ROOT}/build-spike/runtier_${BENCH_TIER}_${BENCH_NAME}}
NRANKS=2
JOBTAG=${PBS_JOBID%%.*}

echo "=== ${BENCH_NAME} x${NRANKS} nodes, tier=${BENCH_TIER} hbm=${BENCH_HBM_MB}MB file=${BENCH_TIER_MB}MB per node ==="
echo "nodes: $(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
echo "exe:  ${EXE}"
echo "args: ${BENCH_ARGS} --nodes ${NRANKS} --node <rank>"
test -x "${EXE}" || { echo "NO EXECUTABLE -- build it first"; exit 2; }

mkdir -p "${RUNDIR}"
cd "${RUNDIR}"
ulimit -c unlimited
rm -f ./rank*.log ./hostfile ./clio_tier*.yaml
sort -u "${PBS_NODEFILE}" > hostfile

# ---- the filesystem tier -------------------------------------------------
case "${BENCH_TIER}" in
  flare)
    TIER_DIR="/lus/flare/projects/IOWarp/clio_tier/${JOBTAG}"
    ;;
  daos)
    # PBS runs this script in a non-login bash where `module` is not defined, so
    # a bare `module load` did nothing: daos then looked for the agent socket in
    # /var/run/daos_agent instead of the DAOS_AGENT_DRPC_DIR the module sets, and
    # launch-dfuse.sh was not on PATH. Initialise Lmod first, and say what took.
    source /usr/share/lmod/lmod/init/bash
    module use /soft/modulefiles
    module load daos/base
    echo "daos env: launch-dfuse.sh=$(command -v launch-dfuse.sh || echo MISSING) agent_dir=${DAOS_AGENT_DRPC_DIR:-unset} socket=$(ls "${DAOS_AGENT_DRPC_DIR:-/run/daos_agent_oneScratch}" 2>&1 | tr '\n' ' ')"
    if ! daos cont query "${DAOS_POOL}" "${DAOS_CONT}" > /dev/null 2>&1; then
      daos cont create --type=POSIX "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | tail -2
    fi
    clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1
    launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" 2>&1 | tail -2
    MNT="/tmp/${DAOS_POOL}/${DAOS_CONT}"
    mount | grep -qF "${MNT}" || { echo "dfuse NOT mounted at ${MNT}"; exit 2; }
    TIER_DIR="${MNT}/clio_tier/${JOBTAG}"
    ;;
  *) echo "BENCH_TIER must be flare or daos"; exit 2 ;;
esac
mkdir -p "${TIER_DIR}" || { echo "cannot create ${TIER_DIR}"; exit 2; }
echo "tier dir: ${TIER_DIR}"
df -h "${TIER_DIR}" | tail -1

# ---- per-rank config -----------------------------------------------------
# __RANK__ is substituted by each rank before it starts its runtime.
cat > clio_tier_template.yaml <<EOF
networking:
  port: 9460
  hostfile: "${RUNDIR}/hostfile"

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
    # Targets stay node-local (see pbs_newcoro_aurora_2n.sh): with the
    # default neighborhood a GPU page can be placed on the other node's HBM,
    # and that write ships the device frame through a synchronous staged
    # copy. The tiers below are per node by construction anyway.
    targets:
      neighborhood: 1
    storage:
      - path: "hbm::gv_tier_hbm"
        bdev_type: "hbm"
        capacity_limit: "${BENCH_HBM_MB}MB"
        score: 1.0
      - path: "${TIER_DIR}/node__RANK__.dat"
        bdev_type: "file"
        persistence_level: "long_term"
        capacity_limit: "${BENCH_TIER_MB}MB"
        score: 0.2
    dpe:
      dpe_type: "max_bw"
EOF
echo "--- config (rank template) ---"
grep -E "path:|capacity_limit|hostfile" clio_tier_template.yaml

export IGC_FunctionControl=3
case "${BENCH_ZE_MASK:-0.0}" in
  none) ;;
  *)    export ZE_AFFINITY_MASK="${BENCH_ZE_MASK:-0.0}" ;;
esac
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
mkdir -p "$SYCL_CACHE_DIR"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export BENCH_RANK_EXE="${EXE}"
export BENCH_RANK_ARGS="${BENCH_ARGS}"
export BENCH_RANK_N="${NRANKS}"
export BENCH_RANK_DIR="${RUNDIR}"

# 200 s: the dfuse launch and the tier listing need their share of the five
# minutes, and PBS reaping the job mid-sentence is what the cap prevents.
echo "--- run (200s cap per rank, ${NRANKS} ranks) ---"
start=$SECONDS
# --cpu-bind: see pbs_newcoro_aurora_2n.sh. The runtime's spinning threads
# must not share a narrow cpuset.
mpiexec -n "${NRANKS}" --ppn 1 --no-vni --envall --cpu-bind "${BENCH_CPU_BIND:-none}" bash -c '
  r=${PALS_RANKID:-${PMI_RANK:-0}}
  cd "$BENCH_RANK_DIR"
  sed "s/__RANK__/$r/g" clio_tier_template.yaml > "clio_tier_r$r.yaml"
  export CLIO_SERVER_CONF="$BENCH_RANK_DIR/clio_tier_r$r.yaml"
  timeout --signal=TERM --kill-after=10s 200 \
    stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS --nodes "$BENCH_RANK_N" --node "$r" \
    > "rank$r.log" 2>&1
  rc=$?
  echo "rank $r on $(hostname) exit=$rc" >> "rank$r.log"
  exit 0
'
mrc=$?
echo "--- elapsed $((SECONDS - start))s, mpiexec exit=${mrc} ---"
rc=${mrc}
for r in $(seq 0 $((NRANKS - 1))); do
  echo "----- rank ${r} -----"
  grep --line-buffered -vE "LoadBalance|\[#78[15]|INFO|SUCCESS|WARNING|HANGWATCH" "rank${r}.log" | tail -30
  rrc=$(grep -oE "^rank ${r} on .* exit=[0-9]+" "rank${r}.log" | tail -1 | grep -oE "[0-9]+$")
  [ -z "${rrc}" ] && rrc=99
  [ "${rrc}" -gt "${rc}" ] && rc=${rrc}
done

echo "--- tier files (what actually spilled to ${BENCH_TIER}) ---"
ls -l "${TIER_DIR}" 2>&1
du -sh "${TIER_DIR}" 2>&1
rm -rf "${TIER_DIR}"
[ "${BENCH_TIER}" = daos ] && clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1

case "${rc}" in
  0)   echo "RESULT ${BENCH_NAME}x${NRANKS}@${BENCH_TIER}: OK" ;;
  124) echo "RESULT ${BENCH_NAME}x${NRANKS}@${BENCH_TIER}: TIMEOUT (a rank exceeded the 200s cap)" ;;
  *)   echo "RESULT ${BENCH_NAME}x${NRANKS}@${BENCH_TIER}: FAILED rc=${rc}" ;;
esac
exit "${rc}"
