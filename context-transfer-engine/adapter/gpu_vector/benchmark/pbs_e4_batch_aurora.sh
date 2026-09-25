#!/usr/bin/env bash
#PBS -l select=16
#PBS -l place=scatter
#PBS -l walltime=00:60:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -l daos=daos_user
#PBS -q debug-scaling
#PBS -A IOWarp
#PBS -j oe
#
# MANY E4 CELLS IN ONE ALLOCATION.
#
# debug-scaling admits ONE running and ONE queued job per user
# (max_run = queued_jobs_threshold = 1), so cells submitted one per job are
# serialised by the queue, not by the machine: each 4-node cell waits 8-10
# minutes to run for 20-60 seconds. The queue does allow 256 nodes and an
# hour, so the way to run cells in parallel is to take a wide allocation and
# put several cells in it at once.
#
# This script splits its nodes into GROUPS OF FOUR and runs one cell per
# group concurrently, then takes the next cells as groups free up. Sixteen
# nodes run four cells at a time; a workload's whole five-composition row
# lands in one job instead of five.
#
# Each cell is a (workload, composition) pair. Groups share nothing: their
# own run directory, their own hostfile and config, their own tier files
# under their own job-and-cell path. The runtime's port is the same in every
# group, which is fine because the groups are disjoint sets of nodes.
#
#   BENCH_CELLS   space-separated <workload>[+<variant>]:<composition>[:<page-kb>[:<slots>]]
#                 e.g. "kmeans:dram100 grayscott+ckpt:bal25:1024:8". A
#                 +variant keeps the workload's binary but takes a different
#                 argument line from the table below, which is how E3's
#                 checkpoint arms are expressed.
#                 The third field overrides the deck's page size, which is
#                 what E2 sweeps, and the fourth its slots-per-block. Both
#                 go into the cell's log name so a sweep does not overwrite
#                 the E4 row. THE FOURTH FIELD IS WHAT MAKES E2 HONEST: the
#                 frame cache is slots x blocks x page, so sweeping the page
#                 alone sweeps the cache too; scaling slots inversely holds
#                 the cache constant in bytes. The editions spell that knob
#                 two ways -- kmeans, grayscott, weights and lammps_md take
#                 --slots (frames per block), lbann and gmx take --cap (frames
#                 for the whole grid) -- and the field is mapped to whichever
#                 the workload has, so a sweep reads the same either way. A
#                 fifth field is a free-form tag, which exists so REPEATS of
#                 one cell get distinct run directories and logs instead of
#                 overwriting each other.
#   BENCH_GROUP_N nodes per cell (default 4)
#   BENCH_CAP     per-rank cap in seconds (default 600)
#   TIER_BUDGET_MB, HBM_MB, DATA_MB  as submit_e4_aurora.sh
#
# The composition shares and the per-workload decks are the same table
# submit_e4_aurora.sh uses, kept here rather than sourced so the job script
# is self-contained on the compute node.
set -u
# GPU NotPresent faults on HOST addresses at 64 nodes (one rank dies, the
# rest livelock): per-page transfer buffers are malloc'd and freed, glibc
# returns large ones with munmap, and Level Zero's cached pinning of that
# address goes stale when the address is reused. Keep freed memory mapped:
# serve everything < 32 MB from the heap and never trim it.
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-33554432}
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-1099511627776}

: "${BENCH_CELLS:?set BENCH_CELLS}"
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
GROUP_N=${BENCH_GROUP_N:-4}
BENCH_CAP=${BENCH_CAP:-600}
TIER_BUDGET_MB=${TIER_BUDGET_MB:-10240}
BENCH_STEPS=${BENCH_STEPS:-2}
BENCH_CKPT_EVERY=${BENCH_CKPT_EVERY:-1}
HBM_MB=${HBM_MB:-4096}
DATA_MB=${DATA_MB:-32768}
DAOS_POOL=${DAOS_POOL:-IOWarp}
DAOS_CONT=${DAOS_CONT:-clio_tier}
JOBTAG=${PBS_JOBID%%.*}
FLARE_ROOT="/lus/flare/projects/IOWarp/clio_tier/${JOBTAG}"

# ---- the cell tables -------------------------------------------------------
# Shares in percent: dram daos flare. A tier at 0 is left out of the config.
shares_for() {
  case "$1" in
    dram100)  echo "100 0 0" ;;
    dram75)   echo "75 25 0" ;;
    bal25)    echo "25 50 25" ;;
    daos70)   echo "10 70 20" ;;
    lustre70) echo "10 20 70" ;;
    *)        echo "" ;;
  esac
}
args_for() {
  case "$1" in
    kmeans)    echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --iters 1 --page-kb 1024" ;;
    grayscott) echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --steps 1 --repeat 1 --page-kb 1024" ;;
    weights)   echo "--blocks 64 --pages 128 --page-kb 1024 --hbm-mb ${HBM_MB} --repeat 1" ;;
    gmx)       echo "--page-kb 20000 --blocks 16 --cap 200 --repeat 1" ;;
    lammps_md) echo "--lattice 534 --steps 1 --page-kb 1024" ;;
    lbann)     echo "--in 65536 --hidden 131072 --out 1024 --batch 64 --steps 1 --page-kb 1024 --blocks 64 --cap 4096 --no-ref" ;;
    # E3, persistence. Same deck and same 8 steps in every arm; what differs
    # is whether a checkpoint is taken and where it is allowed to settle.
    # `nockpt` is the floor, `ckpt` leaves each snapshot at blob score 1.0
    # (it stays in the fast tier -- the asynchronous case), and `ckptdrain`
    # demotes each finished snapshot out of it, which is the synchronous
    # case: the checkpoint is pushed down the stack before the run goes on.
    grayscott+nockpt)    echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --steps 8 --repeat 1 --page-kb 1024" ;;
    grayscott+ckpt)      echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --steps 8 --repeat 1 --page-kb 1024 --ckpt-every 2" ;;
    grayscott+ckptdrain) echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --steps 8 --repeat 1 --page-kb 1024 --ckpt-every 2 --ckpt-drain" ;;
    # E5 at scale: the deck and the step count come from the environment so
    # one arm can be calibrated and then swept. BENCH_STEPS steps, a snapshot
    # every BENCH_CKPT_EVERY of them.
    # E5 (HBM budget sweep) for the workloads without an E5 edition: a
    # 64 GB/node deck at 4 nodes; the budget is the 4th cell field (--cap in
    # frames for gmx/lbann, --slots per block for lammps_md).
    gmx+e5)       echo "--page-kb 78408 --blocks 8 --cap 427 --repeat 5" ;;
    lbann+e5)     echo "--in 65536 --hidden 1048576 --out 1024 --batch 64 --steps 1 --page-kb 1024 --blocks 64 --cap 32768 --no-ref" ;;
    lammps_md+e5) echo "--lattice 640 --steps 5 --page-kb 1024" ;;
    # E5 at scale (pbs_e5_scale_aurora.sh): 8 GB/node decks sized by the
    # caller, final-state checkpoint on; page (gmx plane) and cap come from
    # the cell's 3rd and 4th fields.
    gmx+e5b)      echo "--page-kb 20000 --blocks ${E5B_GMX_BLOCKS:-4} --repeat ${E5B_GMX_PASSES:-12} --no-dense --ckpt-final" ;;
    lbann+e5b)    echo "--in 65536 --hidden ${E5B_LB_HIDDEN:-131072} --out 1024 --batch 64 --steps ${E5B_LB_STEPS:-15} --page-kb 1024 --blocks 64 --no-ref --ckpt-final" ;;
    grayscott+scaled)    echo "--data-mb ${DATA_MB} --hbm-mb ${HBM_MB} --steps ${BENCH_STEPS:-2} --repeat 1 --page-kb 1024 --ckpt-every ${BENCH_CKPT_EVERY:-1}" ;;
    *)         echo "" ;;
  esac
}

# ---- DAOS, once for the whole allocation -----------------------------------
# Every group that needs a DAOS tier uses the same dfuse mount; mounting it
# per cell would race on the same mountpoint.
mount_daos() {
  source /usr/share/lmod/lmod/init/bash
  module use /soft/modulefiles
  module load daos/base
  if ! daos cont query "${DAOS_POOL}" "${DAOS_CONT}" > /dev/null 2>&1; then
    daos cont create --type=POSIX "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | tail -2
  fi
  clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1
  launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" 2>&1 | tail -2
  MNT="/tmp/${DAOS_POOL}/${DAOS_CONT}"
  mount | grep -qF "${MNT}" || { echo "dfuse NOT mounted at ${MNT}"; return 1; }
  return 0
}
NEED_DAOS=0
for cell in ${BENCH_CELLS}; do
  ctail=${cell#*:}
  read -r pd pa pf <<< "$(shares_for "${ctail%%:*}")"
  [ "${pa:-0}" -gt 0 ] && NEED_DAOS=1
done
MNT=""
if [ "${NEED_DAOS}" = 1 ]; then
  mount_daos || exit 2
fi

mapfile -t ALLNODES < <(sort -u "${PBS_NODEFILE}")
NGROUPS=$(( ${#ALLNODES[@]} / GROUP_N ))
echo "=== E4 batch: ${#ALLNODES[@]} nodes, ${NGROUPS} groups of ${GROUP_N}, $(echo ${BENCH_CELLS} | wc -w) cells ==="
[ "${NGROUPS}" -ge 1 ] || { echo "not enough nodes for one group"; exit 2; }

export IGC_FunctionControl=3
export ZE_AFFINITY_MASK="${BENCH_ZE_MASK:-0.0}"
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
mkdir -p "$SYCL_CACHE_DIR"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# ---- run one cell on one group --------------------------------------------
# @param 1 cell, as <workload>:<composition>
# @param 2 group index, which picks the node slice and the run directory
run_cell() {
  local cell=$1 gi=$2
  local wl wlv comp pkb tail
  wlv=${cell%%:*}
  wl=${wlv%%+*}
  tail=${cell#*:}
  comp=${tail%%:*}
  # Third field is the page size in KB, fourth the slots per block.
  local slots="" rtag=""
  if [ "${tail}" = "${comp}" ]; then
    pkb=""
  else
    pkb=${tail#*:}
    if [ "${pkb}" != "${pkb%:*}" ]; then
      slots=${pkb#*:}
      pkb=${pkb%%:*}
      if [ "${slots}" != "${slots%:*}" ]; then rtag=${slots#*:}; slots=${slots%%:*}; fi
    fi
  fi
  local exe="${ROOT}/build-spike/clio_${wl}_paged_newcoro_aot"
  local slug="${wlv#*+}_${comp}"
  [ "${wlv}" = "${wl}" ] && slug="${comp}"
  slug="${slug}${pkb:+_p${pkb}}${slots:+_s${slots}}${rtag:+_r${rtag}}"
  local rundir="${E4_OUTROOT:-${ROOT}/build-spike}/e4b_${JOBTAG}_${wl}_${slug}"
  local log="${E4_OUTROOT:-${ROOT}/build-spike/pbs}/${wl}_e4_${slug}.log"
  local pd pa pf top daos flare
  read -r pd pa pf <<< "$(shares_for "${comp}")"
  local args; args=$(args_for "${wlv}")
  # Replace the deck's own --page-kb rather than appending, so the edition
  # sees exactly one.
  if [ -n "${pkb}" ]; then
    args=$(echo "${args}" | sed -E "s/--page-kb [0-9]+/--page-kb ${pkb}/")
  fi
  if [ -n "${slots}" ]; then
    local knob="--slots"
    case "${wl}" in lbann|gmx) knob="--cap" ;; esac
    case "${args}" in
      *"${knob} "*) args=$(echo "${args}" | sed -E "s/${knob} [0-9]+/${knob} ${slots}/") ;;
      *)            args="${args} ${knob} ${slots}" ;;
    esac
  fi
  if [ -z "${pd:-}" ] || [ -z "${args}" ] || [ ! -x "${exe}" ]; then
    echo "RESULT ${wl}x${GROUP_N}@${slug}: FAILED rc=2 (unknown cell or no exe)" | tee -a "${log}"
    return
  fi
  top=$(( TIER_BUDGET_MB * pd / 100 ))
  daos=$(( TIER_BUDGET_MB * pa / 100 ))
  flare=$(( TIER_BUDGET_MB * pf / 100 ))

  mkdir -p "${rundir}"
  local hosts="" i
  for (( i = gi * GROUP_N; i < (gi + 1) * GROUP_N; ++i )); do
    hosts="${hosts}${hosts:+,}${ALLNODES[$i]}"
    echo "${ALLNODES[$i]}" >> "${rundir}/hostfile.tmp"
  done
  mv "${rundir}/hostfile.tmp" "${rundir}/hostfile"

  # Tier directories, per cell so two groups never share a file bdev.
  local tdir_daos="" tdir_flare="" storage=""
  storage="    storage:"
  if [ "${top}" -gt 0 ]; then
    storage="${storage}
      - path: \"ram::gv_tier_dram\"
        bdev_type: \"ram\"
        capacity_limit: \"${top}MB\"
        score: 1.0"
  fi
  if [ "${daos}" -gt 0 ]; then
    tdir_daos="${MNT}/clio_tier/${JOBTAG}/${wl}_${slug}"
    mkdir -p "${tdir_daos}"
    storage="${storage}
      - path: \"${tdir_daos}/node__RANK__.dat\"
        bdev_type: \"file\"
        persistence_level: \"long_term\"
        capacity_limit: \"${daos}MB\"
        score: 0.5"
  fi
  if [ "${flare}" -gt 0 ]; then
    tdir_flare="${FLARE_ROOT}/${wl}_${slug}"
    mkdir -p "${tdir_flare}"
    storage="${storage}
      - path: \"${tdir_flare}/node__RANK__.dat\"
        bdev_type: \"file\"
        persistence_level: \"long_term\"
        capacity_limit: \"${flare}MB\"
        score: 0.2"
  fi

  cat > "${rundir}/clio_tier_template.yaml" <<EOF
networking:
  port: 9460
  hostfile: "${rundir}/hostfile"
  # >= node count, as in the E1/E5 configs: a Broadcast wider than
  # neighborhood_size is split into Range queries that the receiving node
  # runs only locally, so at 64 nodes pool creation reached nodes 0 and 32
  # only (lbann at 64 nodes: writebacks REFUSED rc=11 on ranks 0 and 32).
  neighborhood_size: 1024

# SWIM OFF, for the same reason the E1 scaling config turns it off. Its
# suspicion timeout is 60 s and expiry runs TriggerRecovery, which moves a
# LIVE node's containers; a wide compose starves probe replies past that
# threshold and the cluster never forms. Seen at 256 nodes in E1 and again
# at 64 nodes here, where a grayscott cell died with "could not create
# reduction tag" after 36 minutes of send timeouts -- it never reached its
# first step, let alone its checkpoints. Four-node cells never hit it,
# which is why it went unnoticed while every batch was four nodes wide.
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
${storage}
    dpe:
      dpe_type: "max_bw"
EOF

  {
    echo "=== ${wl} @ ${slug} on ${hosts} (dram ${top} daos ${daos} flare ${flare} MB/node) ==="
    echo "args: ${args} --nodes ${GROUP_N} --node <rank>"
  } > "${log}"
  local start=$SECONDS
  BENCH_RANK_EXE="${exe}" BENCH_RANK_ARGS="${args}" BENCH_RANK_N="${GROUP_N}" \
  BENCH_RANK_DIR="${rundir}" BENCH_RANK_CAP="${BENCH_CAP}" \
  mpiexec -n "${GROUP_N}" --ppn 1 --hosts "${hosts}" --no-vni --envall \
          --cpu-bind none bash -c '
    r=${PALS_RANKID:-${PMI_RANK:-0}}
    cd "$BENCH_RANK_DIR"
    sed "s/__RANK__/$r/g" clio_tier_template.yaml > "clio_tier_r$r.yaml"
    export CLIO_SERVER_CONF="$BENCH_RANK_DIR/clio_tier_r$r.yaml"
    en() { cat /sys/class/drm/card0/device/hwmon/hwmon*/energy1_input 2>/dev/null | head -1; }
    e0=$(en)
    timeout --signal=TERM --kill-after=10s "${BENCH_RANK_CAP}" \
      stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS --nodes "$BENCH_RANK_N" --node "$r" \
      > "rank$r.log" 2>&1
    rc=$?
    e1=$(en)
    [ -n "$e0" ] && [ -n "$e1" ] && echo "ENERGY_UJ $((e1 - e0))" >> "rank$r.log"
    echo "rank $r on $(hostname) exit=$rc" >> "rank$r.log"
    exit 0
  ' >> "${log}" 2>&1
  local mrc=$? rc r rrc
  echo "--- elapsed $((SECONDS - start))s, mpiexec exit=${mrc} ---" >> "${log}"
  rc=${mrc}
  for (( r = 0; r < GROUP_N; ++r )); do
    echo "----- rank ${r} -----" >> "${log}"
    grep -vE "LoadBalance|\[#78[15]|INFO|SUCCESS|WARNING|HANGWATCH" \
         "${rundir}/rank${r}.log" 2>/dev/null | tail -30 >> "${log}"
    rrc=$(grep -oE "^rank ${r} on .* exit=[0-9]+" "${rundir}/rank${r}.log" 2>/dev/null |
          tail -1 | grep -oE "[0-9]+$")
    [ -z "${rrc}" ] && rrc=99
    [ "${rrc}" -gt "${rc}" ] && rc=${rrc}
  done
  {
    echo "--- tier files ---"
    for d in ${tdir_daos} ${tdir_flare}; do ls -l "${d}" 2>&1; rm -rf "${d}"; done
    case "${rc}" in
      0)   echo "RESULT ${wl}x${GROUP_N}@${slug}: OK" ;;
      124) echo "RESULT ${wl}x${GROUP_N}@${slug}: TIMEOUT (a rank exceeded the ${BENCH_CAP}s cap)" ;;
      *)   echo "RESULT ${wl}x${GROUP_N}@${slug}: FAILED rc=${rc}" ;;
    esac
  } >> "${log}"
  echo "CELL DONE ${wl}@${slug} rc=${rc} ($((SECONDS - start))s)"
}

# ---- dispatch: keep every group busy ---------------------------------------
declare -a GPID
for (( g = 0; g < NGROUPS; ++g )); do GPID[$g]=0; done
for cell in ${BENCH_CELLS}; do
  placed=0
  while [ "${placed}" = 0 ]; do
    for (( g = 0; g < NGROUPS; ++g )); do
      if [ "${GPID[$g]}" = 0 ] || ! kill -0 "${GPID[$g]}" 2>/dev/null; then
        run_cell "${cell}" "${g}" &
        GPID[$g]=$!
        placed=1
        break
      fi
    done
    [ "${placed}" = 0 ] && sleep 5
  done
done
wait
case "${NEED_DAOS}" in 1) clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1 ;; esac
rm -rf "${FLARE_ROOT}"
echo "E4 BATCH DONE"
