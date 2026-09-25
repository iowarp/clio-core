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
#   BENCH_PERNODE_MB  per-node share (default 4096; the plan's anchor is 32000)
#   BENCH_BLOCKS   space-separated grid sizes (work-groups of 256). EVERY
#                  substrate, Eternia included, runs each one, so a row is
#                  always a like-for-like grid. Default "1024".
#   BENCH_WORKLOADS default "kmeans grayscott"
#   BENCH_ET_SUFFIX suffix of the Eternia binary to run (e.g. "_x"); default none
#   BENCH_SLOTS    Eternia frames per block (default: enough 1 MB frames across
#                  64 blocks to hold the whole share plus 1 GB, so it is resident)
#
# The plan's own ladder (4 -> 64 nodes at 32 GB/node) runs in debug-scaling,
# which admits up to 256 nodes: qsub -q debug-scaling -l walltime=01:00:00.
set -u

ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
ITERS=${BENCH_ITERS:-20}
CAP=${BENCH_CAP:-900}
NRANKS=$(sort -u "${PBS_NODEFILE}" | wc -l)
PERNODE_MB=${BENCH_PERNODE_MB:-4096}
DATA_MB=$(( PERNODE_MB * NRANKS ))
JOBTAG=${PBS_JOBID%%.*}

BLOCKS_LIST=${BENCH_BLOCKS:-1024}
WORKLOADS=${BENCH_WORKLOADS:-"kmeans grayscott"}
# Frames per block that hold the whole share (plus 1 GB) at a given grid.
# Floor 24: grayscott's stencil holds ten planes per block and its edition
# raises anything below 24 to 24 anyway.
slots_for() {
  # 25% HEADROOM. Pages hash into sets, so a cache sized to the deck plus a
  # few percent overflows some sets and evicts; with edge-only publishing an
  # evicted dirty page is lost (grayscott refuses the run -- see its gate).
  local s=$(( (PERNODE_MB * 5 / 4 + $1 - 1) / $1 ))
  [ -n "${BENCH_SLOTS:-}" ] && s=${BENCH_SLOTS}
  [ "${s}" -lt 24 ] && s=24
  echo "${s}"
}

echo "=== E1 rung: ${NRANKS} nodes, ${PERNODE_MB} MB/node, ${DATA_MB} MB total, ${ITERS} iters ==="
echo "    grids: ${BLOCKS_LIST} (x256 threads), workloads: ${WORKLOADS}"

export IGC_FunctionControl=3
export ZE_AFFINITY_MASK=${BENCH_ZE_MASK:-0.0}
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
mkdir -p "$SYCL_CACHE_DIR"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# ISHMEM's symmetric heap is set per workload below: grayscott keeps its
# fields in the heap, so it needs the share plus halo room; kmeans keeps its
# points in plain device memory and needs only the collective buffers. A
# 64 GB tile refuses a heap near its size, so it is capped at 48 GB.
ishmem_heap_mb() {
  local mb
  case "$1" in
    grayscott) mb=$(( PERNODE_MB * 27 / 20 )) ;;
    *)         mb=$(( PERNODE_MB / 2 )) ;;
  esac
  [ "${mb}" -gt 49152 ] && mb=49152
  [ "${mb}" -lt 8192 ] && mb=8192
  echo "${mb}"
}
export MPIR_CVAR_ENABLE_GPU=1

# ---- one run, reported on its own RESULT line -----------------------------
# @param 1 label used in the RESULT line and the run directory
# @param 2 executable
# @param 3 argument string (the rank flags are appended)
# @param 4 1 if the binary takes --nodes/--node, 0 if it is an MPI baseline
run_one() {
  local label=$1 exe=$2 args=$3 ranked=$4
  local rundir="${BENCH_RUNROOT:-${ROOT}/build-spike}/e1_${JOBTAG}_${label}"
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
    en() { cat /sys/class/drm/card0/device/hwmon/hwmon*/energy1_input 2>/dev/null | tr "\n" " "; }
    e0=$(en)
    [ "$BENCH_RANK_RANKED" = 1 ] && extra="--nodes $BENCH_RANK_N --node $r"
    timeout --signal=TERM --kill-after=10s "$BENCH_RANK_CAP" \
      stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS $extra \
      > "rank$r.log" 2>&1
    rc=$?
    e1=$(en)
    echo "ENERGY_UJ before: ${e0} after: ${e1}" >> "rank$r.log"
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
  # E1 COMMUNICATION: every rank's COMM line (baselines: wall time in the
  # substrate's exchanges; Eternia: GPU time in Fetch/Hold/Flush), and the
  # slowest rank's -- a step waits for it.
  local cl
  cl=$(grep -h "^COMM" "${rundir}"/rank*.log 2>/dev/null)
  if [ -n "${cl}" ]; then
    echo "${cl}" | sed 's/^/    /' | cut -c1-240
    echo "COMMMAX e1/${label}x${NRANKS}: $(echo "${cl}" | grep -oE 'comm_ms=[0-9.]+' | cut -d= -f2 | sort -g | tail -1) ms (slowest rank)"
  fi
  # Energy summed over ranks (first hwmon counter = the card), in joules.
  local ej=0 f b a
  for f in "${rundir}"/rank*.log; do
    read -r b _ <<< "$(grep -a "ENERGY_UJ" "${f}" | sed 's/.*before: //; s/ after:.*//')"
    read -r a _ <<< "$(grep -a "ENERGY_UJ" "${f}" | sed 's/.*after: //')"
    [ -n "${b:-}" ] && [ -n "${a:-}" ] && ej=$(awk -v e="${ej}" -v x="${a}" -v y="${b}" 'BEGIN{printf "%.0f", e + (x - y) / 1e6}')
  done
  echo "ENERGY e1/${label}x${NRANKS}: ${ej} J"
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
  # >= node count: a Broadcast wider than neighborhood_size is split into
  # multi-node Range queries that the receiving node runs only locally (a
  # runtime bug), so at 64 nodes pool creation reached nodes 0 and 32 only
  # and 62 nodes had no CTE target. Sized past the allocation, every
  # broadcast becomes one single-node query per node (the 24-node path).
  neighborhood_size: 1024

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
        # <= the vector's blob score (0.5), or MaxBwDpe files the only tier
        # under fallback and places through its bandwidth model instead.
        score: 0.5
    dpe:
      dpe_type: "max_bw"
EOF
}

# WEAK-SCALED DECKS for gmx, lbann and lammps_md: each keeps the single-node
# deck's per-node load (gmx 16.8 GB of mesh, lbann ~17 GB of W1, lammps_md
# 16.4 M atoms) and satisfies the editions' divisibility rules at NRANKS.
wl_decks() {
  local wl=$1
  case "${wl}" in
    gmx)
      # K ~ 1280 * N^(1/3), a multiple of 16 (whole-KB square page) and of N.
      GX_K=$(awk -v n="${NRANKS}" 'BEGIN{k=int(1280*exp(log(n)/3)/16+0.5)*16; while (k % n) k+=16; print k}')
      GX_PKB=$(( GX_K * GX_K * 8 / 1024 ))
      GX_ATOMS=$(( 20000000 * NRANKS ))
      GX_CAP=$(( (GX_K + NRANKS - 1) / NRANKS + 16 )) ;;
    lbann)
      # WEAK SCALING WITH A FIXED INPUT: I = 262144 (a 1 MB W1 row), H/N W1
      # rows per node (16384: 16 GB of W1 per node, 16 rows per work-group
      # at 1024 work-groups), O = 4096 N (4 output rows per work-group). A
      # W2 row is H floats = 64 KB x N, so the page is the larger of 1 MB
      # and one W2 row; rows-per-page must divide the per-work-group row
      # counts (16 and 4). At N=24 that forces 12288 hidden rows per node
      # (see below) -- flagged.
      LB_I=262144; LB_O=$(( 4096 * NRANKS ))
      local hn=16384
      local w2row=$(( 65536 * NRANKS ))                 # bytes: H x 4 / 1024 per KB below
      local pkb=$(( w2row * hn / 16384 / 1024 ))        # KB of one W2 row at hn=16384
      [ "${pkb}" -lt 1024 ] && pkb=1024
      if (( pkb % 1024 != 0 )) || (( 16 % (pkb / 1024) != 0 )); then
        # No page holds whole rows of both layers (N=24: 1 MB and 1.5 MB).
        # Make the layers the same width instead: 12288 hidden rows per
        # node and I = H, so both rows are one page (1.125 MB at N=24);
        # 12 W1 rows per work-group. Flagged: ~13.5 GB of W1 per node.
        hn=12288; LB_I=$(( hn * NRANKS )); pkb=$(( LB_I * 4 / 1024 ))
        echo "    lbann: N=${NRANKS} -> I=H=${LB_I}, ${hn} hidden rows per node, page ${pkb} KB"
      fi
      LB_H=$(( hn * NRANKS )); LB_PKB=${pkb}
      local rpp1=$(( LB_PKB * 1024 / (LB_I * 4) ))
      local rpp2=$(( LB_PKB * 1024 / (LB_H * 4) )); [ "${rpp2}" -lt 1 ] && rpp2=1
      LB_RPP=${rpp2}
      # Cache: this node's W1 rows and ITS OWN W2 rows (bwd1 by partials).
      local w1p=$(( hn / rpp1 )); local w2p=$(( LB_O / NRANKS / rpp2 ))
      LB_CAP=$(( (w1p + w2p) * 5 / 4 ))
      # One set per work-group: size the sets from the grid actually used.
      local nb=${2:-1024}
      LB_SET=$(( (LB_CAP + nb - 1) / nb + 8 )) ;;
    lammps_md)
      MD_L=$(awk -v n="${NRANKS}" 'BEGIN{print int(160*exp(log(n)/3)+0.5)}')
      # bins per side as the edition computes them (cutoff 2.5 + skin 0.3)
      local nb; nb=$(awk -v l="${MD_L}" 'BEGIN{print int(l*exp(log(4/0.8442)/3)/2.8)}')
      local row=$(( nb * 48 * 16 ))
      MD_PKB=$(( row * 8 / 1024 )); MD_NLKB=$(( nb * 48 * 96 * 4 / 1024 ))
      echo "    lammps_md: lattice ${MD_L}, ${nb} bins/side, x page ${MD_PKB} KB (8 rows), list page ${MD_NLKB} KB" ;;
  esac
}

worst=0
for wl in ${WORKLOADS}; do
 for B in ${BLOCKS_LIST}; do
  SLOTS=$(slots_for "${B}")
  echo "=== ${wl} at ${B} work-groups x 256: eternia frame cache ${SLOTS} x ${B} x 1 MB = $(( SLOTS * B )) MB ==="
  wl_decks "${wl}" "${B}"
  # GMX FRAME BUDGET: the Eternia mesh cache holds at least one frame per
  # work-group and a frame is a whole K*K plane, which grows as N^(2/3). At
  # 16 nodes 1024 x 80 MB = 85 GB overflowed the 64 GB tile. Halve the grid
  # (for ALL four arms, so they still share it) until the frames fit 40 GB (52 GB failed to allocate at 64 nodes).
  if [ "${wl}" = gmx ]; then
    while [ $(( B * GX_PKB / 1024 )) -gt 40960 ] && [ "${B}" -gt 32 ]; do
      B=$(( B / 2 ))
    done
    wl_decks "${wl}" "${B}"
    echo "    gmx: ${B} work-groups (frames ${B} x $(( GX_PKB / 1024 )) MB)"
  fi
  # PER-WORKLOAD ITERATION COUNT: BENCH_ITERS_<wl> overrides BENCH_ITERS, so
  # every workload's timed region can be sized to minutes, not milliseconds
  # (a 5-step grayscott run times 0.3 s, where any one-off cost is the
  # result). Saved and restored around the arms so ITERS stays the default.
  _iv="BENCH_ITERS_${wl}"; ITERS_SAVE=${ITERS}; ITERS=${!_iv:-${ITERS}}
  # lbann's per-node W2 work grows with N (O and H both scale with N), so
  # above 16 nodes a fixed step count overruns the 1 h walltime (64 nodes:
  # ~11 s/step x 60 steps x 4 arms). Scale steps by 16/N, floor 15: every
  # arm still runs > 2 min.
  if [ "${wl}" = lbann ] && [ "${NRANKS}" -gt 16 ]; then
    ITERS=$(( ITERS * 16 / NRANKS )); [ "${ITERS}" -lt 15 ] && ITERS=15
  fi
  echo "    ${wl}: ${ITERS} iterations/steps/passes"
  case "${wl}" in
    kmeans)    base_args="--data-mb ${DATA_MB} --iters ${ITERS} --dims 32 --clusters 16 --blocks ${B} --threads 256" ;;
    grayscott) base_args="--data-mb ${DATA_MB} --steps ${ITERS} --page-kb 1024 --blocks ${B} --threads 256" ;;
    gmx)       base_args="--k ${GX_K} --atoms ${GX_ATOMS} --passes ${ITERS} --blocks ${B} --threads 256" ;;
    lbann)     base_args="--in ${LB_I} --hidden ${LB_H} --out ${LB_O} --batch 64 --lr ${LB_LR:-1e-7} --steps ${ITERS} --rpp ${LB_RPP} --blocks ${B} --threads 256 --no-ref" ;;
    lammps_md) base_args="--md --lattice ${MD_L} --cap 48 --temp 3.0 --steps 20 --rebin 10 --blocks 512 --threads 128 --drift-tol 1e-2" ;;
  esac
  export ISHMEM_SYMMETRIC_SIZE=$(( $(ishmem_heap_mb "${wl}") * 1024 * 1024 ))
  echo "    ${wl}: ishmem symmetric heap $(ishmem_heap_mb "${wl}") MB"
  # BENCH_SUBS: which baselines to run (empty = Eternia only, e.g. a rung
  # whose baselines already ran).
  for sub in ${BENCH_SUBS-mpi ccl ishmem}; do
    # NO `|| true` HERE. It makes $? the status of `true`, so every rung
    # summarised as OK however many cells failed -- which is exactly what
    # the 256 rung did with two of its eight cells broken. run_one already
    # never aborts the script (it returns, it does not exit), so the guard
    # was never needed.
    rc=0
    run_one "${wl}_b${B}_${sub}" "${ROOT}/build-spike/clio_${wl}_${sub}_bench${BENCH_BL_SUFFIX:-}" \
            "${base_args}" 0 || rc=$?
    [ "${rc}" -gt "${worst}" ] && worst=${rc}
  done
  # The Eternia arm, on the SAME grid as the baselines above: SLOTS frames per
  # block of 1 MB over B blocks holds the node's whole share, so it is resident.
  # E5 (memory reduction) over the E1 decks: shrink ONLY the Eternia cache.
  #   BENCH_E5_DIV=D  gmx: cap = max(cap_need, GX_CAP / D) with cap_need =
  #                   min(4B, slab + 3) + 2 (so D > 1 needs a smaller grid);
  #                   lbann: cap = max(LB_CAP / D, 2B), sets resized;
  #                   lammps_md: --slots BENCH_E5_MD_SLOTS (x/v frames per block).
  et_tag=""
  if [ -n "${BENCH_E5_DIV:-}" ]; then
    et_tag="_e5d${BENCH_E5_DIV}"
    case "${wl}" in
      gmx)
        local_slab=$(( (GX_K + NRANKS - 1) / NRANKS ))
        need=$(( 4 * B < local_slab + 3 ? 4 * B + 2 : local_slab + 5 ))
        GX_CAP=$(( GX_CAP / BENCH_E5_DIV )); [ "${GX_CAP}" -lt "${need}" ] && GX_CAP=${need} ;;
      lbann)
        LB_CAP=$(( LB_CAP / BENCH_E5_DIV )); [ "${LB_CAP}" -lt $(( 2 * B )) ] && LB_CAP=$(( 2 * B ))
        LB_SET=$(( (LB_CAP + B - 1) / B + 8 )) ;;
    esac
    echo "    E5 div ${BENCH_E5_DIV}: gmx cap ${GX_CAP:-} lbann cap ${LB_CAP:-} lammps slots ${BENCH_E5_MD_SLOTS:-auto}"
  fi
  et_rundir="${BENCH_RUNROOT:-${ROOT}/build-spike}/e1_${JOBTAG}_${wl}_b${B}_eternia${et_tag}"
  mkdir -p "${et_rundir}"
  et_conf "${et_rundir}" $(( PERNODE_MB + 2048 ))
  export CLIO_SERVER_CONF="${et_rundir}/clio_e1.yaml"
  case "${wl}" in
    kmeans)    et_args="--data-mb ${DATA_MB} --hbm-mb ${PERNODE_MB} --iters ${ITERS} --page-kb 1024 --blocks ${B} --threads 256 --slots ${SLOTS}" ;;
    grayscott) et_args="--data-mb ${DATA_MB} --hbm-mb ${PERNODE_MB} --steps ${ITERS} --repeat 1 --page-kb 1024 --blocks ${B} --threads 256 --slots ${SLOTS}" ;;
    # gmx: the slab plus the gather's 3 peer planes, resident; no write-site
    # publish (the host flush before the gather is the exchange).
    gmx)       et_args="--page-kb ${GX_PKB} --atoms ${GX_ATOMS} --repeat ${ITERS} --blocks ${B} --threads 256 --no-dense --cap ${GX_CAP} --no-publish" ;;
    lbann)     et_args="--in ${LB_I} --hidden ${LB_H} --out ${LB_O} --batch 64 --lr ${LB_LR:-1e-7} --steps ${ITERS} --page-kb ${LB_PKB} --blocks ${B} --threads 256 --no-ref --cap ${LB_CAP} --set-size ${LB_SET}" ;;
    # lammps_md: pages of whole rows (8 x-rows, 1 list row) so the force
    # pair loop's one-load fast path engages; MD_LEAN (set below).
    lammps_md) et_args="--md --lattice ${MD_L} --cap 48 --temp 3.0 --steps 20 --rebin 10 --page-kb ${MD_PKB} --nl-page-kb ${MD_NLKB} --blocks 512 --threads 128 --drift-tol 1e-2" ;;
  esac
  rc=0
  # Per-workload binary and extra flags: BENCH_ET_SUFFIX_<wl> / BENCH_ET_ARGS_<wl>
  # override the common BENCH_ET_SUFFIX.
  sfx_var="BENCH_ET_SUFFIX_${wl}"; args_var="BENCH_ET_ARGS_${wl}"
  et_sfx="${!sfx_var:-${BENCH_ET_SUFFIX:-}}"
  et_args="${et_args} ${!args_var:-}"
  # grayscott's two-phase build is subroutine-call (IGC_FunctionControl=2);
  # lammps_md runs resident-lean (no interior publish, no resort clears).
  et_fc=3; [ "${wl}" = grayscott ] && et_fc=${BENCH_GS_FC:-2}
  if [ "${wl}" = lammps_md ]; then export MD_LEAN=1; else unset MD_LEAN; fi
  [ "${wl}" = lammps_md ] && [ -n "${BENCH_E5_MD_SLOTS:-}" ] && et_args="${et_args} --slots ${BENCH_E5_MD_SLOTS}"
  IGC_FunctionControl=${et_fc} \
  run_one "${wl}_b${B}_eternia${et_tag}" "${ROOT}/build-spike/clio_${wl}_paged_newcoro_aot${et_sfx}" \
          "${et_args}" 1 || rc=$?
  unset MD_LEAN
  [ "${rc}" -gt "${worst}" ] && worst=${rc}
  unset CLIO_SERVER_CONF
  ITERS=${ITERS_SAVE}
 done
done

echo "RESULT e1x${NRANKS}: $([ "${worst}" -eq 0 ] && echo OK || echo "FAILED rc=${worst}")"
echo "E1 RUNG DONE ${NRANKS}"
exit 0
