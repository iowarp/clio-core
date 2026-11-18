#!/usr/bin/env bash
#PBS -l select=64
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -l daos=daos_user
#PBS -q debug-scaling
#PBS -A IOWarp
#PBS -j oe
#
# The same bytes moved through DAOS and through clio, on the same nodes, in
# one allocation: BW_MB_PER_NODE MiB per node (default 1600, so 64 nodes move
# 100 GiB each way), written then read back, 1 MiB transfers everywhere.
#
#   DAOS   IOR against the project pool: the DFS API (IOR's native DAOS path,
#          file per process, -C shifted reads, fsync), at BW_IOR_PPN ranks per
#          node and again at BW_IOR_PPN2, in a default container (redundancy
#          factor 3, DFS's default object class) and in an rd_fac:0 / SX
#          container (the peak configuration), then POSIX through dfuse with
#          the libpil4dfs interception library at BW_IOR_PPN in the default
#          container. The containers are created for the job and destroyed
#          after it.
#   clio   clio_cte_vector_stress --bw: one embedded runtime per node, 1 MiB
#          pages hash-distributed over the nodes' DRAM tiers, write / read
#          back / read the next node's pages, each phase barrier-fenced, in
#          the shapes BW_CLIO_SHAPES lists (client threads, records per
#          PodMulti batch, runtime worker threads per node).
#
# Aggregate rates use the slowest node's phase time, as IOR does. The
# SUMMARY block at the end has everything in GB/s (1e9).
#
#   BW_MB_PER_NODE   MiB moved per node per phase (default 1600)
#   BW_IOR_PPN       IOR ranks per node, first DFS run and the POSIX run (8)
#   BW_IOR_PPN2      IOR ranks per node, second DFS run (32)
#   BW_OUT           output root (default flare llogan_e1/bw_<jobid>)
#   DAOS_POOL        (IOWarp)
#   BW_CLIO_SHAPES   clio shapes as <client threads>:<batch>:<runtime workers>
#                    [:<CLIO_PREFAULT size, e.g. 4GB>]  (default
#                    "32:16:8 64:32:8 32:16:8:4GB"). The RAM tier is a sparse
#                    mapping populated inside the write path; CLIO_PREFAULT
#                    commits that prefix at compose instead, and the mode's
#                    warm rewrite/re-read pass shows the rest of the cold cost.
#   BW_SKIP_DAOS / BW_SKIP_CLIO   set to 1 to run only the other half
set -u
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
HERE=${ROOT}/context-transfer-engine/benchmark
JOBTAG=${PBS_JOBID%%.*}
NNODES=$(sort -u "${PBS_NODEFILE}" | wc -l)
OUT=${BW_OUT:-/lus/flare/projects/IOWarp/llogan_e1/bw_${JOBTAG}}
MB=${BW_MB_PER_NODE:-1600}
PPN=${BW_IOR_PPN:-8}
PPN2=${BW_IOR_PPN2:-32}
DAOS_POOL=${DAOS_POOL:-IOWarp}
DAOS_CONT=${DAOS_CONT:-clio_bw_${JOBTAG}}
IOR_INSTALL=/soft/daos/examples/daos_peak_test/ior_mdtest_install_bin
HOSTS=$(sort -u "${PBS_NODEFILE}" | paste -sd,)
mkdir -p "${OUT}"
cd "${OUT}" || exit 1
echo "=== bandwidth: ${NNODES} nodes, ${MB} MiB per node per phase, total $((MB * NNODES / 1024)) GiB, out ${OUT} ==="
date

# ---- DAOS ---------------------------------------------------------------------
if [ "${BW_SKIP_DAOS:-0}" != 1 ]; then
  source /usr/share/lmod/lmod/init/bash
  module use /soft/modulefiles
  module load daos/base
  export LD_LIBRARY_PATH=${IOR_INSTALL}/lib:${LD_LIBRARY_PATH:-}
  export PATH=${IOR_INSTALL}/bin:${PATH}
  echo "--- daos pool ${DAOS_POOL}"
  daos version 2>&1 | head -2
  pidof daos_agent > /dev/null || echo "WARNING: no daos_agent on this node"
  daos pool query "${DAOS_POOL}" 2>&1 | tee "${OUT}/pool_query.txt" | head -30
  daos container destroy "${DAOS_POOL}" "${DAOS_CONT}" > /dev/null 2>&1
  if ! daos container create --type POSIX "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | tail -3; then
    echo "container create failed"
  fi
  daos container query "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | head -12

  # ior_run <tag> <ppn> <api args...>; IOR_PRELOAD names a library to LD_PRELOAD
  ior_run() {
    local tag=$1 ppn=$2; shift 2
    local n=$((NNODES * ppn)) b=$((MB / ppn)) pre=()
    [ -n "${IOR_PRELOAD:-}" ] && pre=(--env LD_PRELOAD="${IOR_PRELOAD}")
    echo "--- ior ${tag}: ${n} ranks (${ppn}/node), -b ${b}m -t 1m"
    local t0=$SECONDS
    timeout --foreground 900 \
      mpiexec -np "${n}" -ppn "${ppn}" --hosts "${HOSTS}" --cpu-bind depth -d $((104 / ppn)) --no-vni -genvall "${pre[@]}" \
        ior -i 1 -b "${b}m" -t 1m -w -r -F -C -e -v "$@" > "${OUT}/ior_${tag}.log" 2>&1
    echo "ior ${tag}: rc=$? ($((SECONDS - t0)) s)"
    grep -a 'Max Write\|Max Read\|^write \|^read \|ERROR\|error' "${OUT}/ior_${tag}.log" | head -8
  }
  # The pool's containers default to redundancy factor 3; an object class
  # the container cannot honour (SX under rd_fac 3) fails every open with
  # DER_INVAL, so the default container takes DFS's own class choice, and
  # the peak case gets its own rd_fac:0 container with SX (the peak test's
  # second configuration).
  daos container get-prop "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | grep -i 'redun\|class\|chunk' | head -6
  ior_run dfs_ppn${PPN} "${PPN}" -a DFS --dfs.pool="${DAOS_POOL}" --dfs.cont="${DAOS_CONT}" -o /bw_dfs1.dat
  ior_run dfs_ppn${PPN2} "${PPN2}" -a DFS --dfs.pool="${DAOS_POOL}" --dfs.cont="${DAOS_CONT}" -o /bw_dfs2.dat
  SXCONT=${DAOS_CONT}_sx
  daos container destroy "${DAOS_POOL}" "${SXCONT}" > /dev/null 2>&1
  if daos container create --type POSIX "${DAOS_POOL}" "${SXCONT}" --properties=rd_fac:0 \
       --file-oclass=SX --dir-oclass=S1 --chunk-size=1048576 2>&1 | tail -2; then
    ior_run dfs_sx_ppn${PPN} "${PPN}" -a DFS --dfs.pool="${DAOS_POOL}" --dfs.cont="${SXCONT}" \
            --dfs.oclass=SX --dfs.dir_oclass=S1 --dfs.chunk_size=1048576 -o /bw_dfs3.dat
    ior_run dfs_sx_ppn${PPN2} "${PPN2}" -a DFS --dfs.pool="${DAOS_POOL}" --dfs.cont="${SXCONT}" \
            --dfs.oclass=SX --dfs.dir_oclass=S1 --dfs.chunk_size=1048576 -o /bw_dfs4.dat
    daos container destroy "${DAOS_POOL}" "${SXCONT}" 2>&1 | tail -1
  fi
  echo "--- dfuse"
  clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1
  launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" 2>&1 | tail -2
  MNT=/tmp/${DAOS_POOL}/${DAOS_CONT}
  if mount | grep -qF "${MNT}"; then
    IOR_PRELOAD=/usr/lib64/libpil4dfs.so ior_run posix_pil4dfs_ppn${PPN} "${PPN}" -a posix -o "${MNT}/bw_posix.dat"
  else
    echo "dfuse NOT mounted at ${MNT}; POSIX run skipped"
  fi
  clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1
  daos container destroy "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | tail -1
fi

# ---- clio ---------------------------------------------------------------------
if [ "${BW_SKIP_CLIO:-0}" != 1 ]; then
  clio_run() {  # clio_run <tag> <threads> <batch> <runtime workers> <prefault|->
    local tag=$1
    if [ "${5}" != "-" ]; then export CLIO_PREFAULT="${5}"; else unset CLIO_PREFAULT; fi
    echo "--- clio ${tag}: ${2} threads, batches of ${3}, ${4} runtime workers per node, CLIO_PREFAULT=${CLIO_PREFAULT:-unset}"
    mpiexec -np "${NNODES}" -ppn 1 --hosts "${HOSTS}" --no-vni \
      bash -c 'pkill -KILL -f "clio_cte_vector_stres[s]" 2>/dev/null; rm -f /dev/shm/clio_* /dev/shm/cte_stress_* 2>/dev/null; true'
    STRESS_OUT="${OUT}/clio_${tag}" STRESS_CAP=900 BENCH_RT_THREADS="${4}" \
    STRESS_ARGS="--bw --pages-per-node ${MB} --page-kb 1024 --threads ${2} --batch ${3}" \
      bash "${HERE}/pbs_stress_aurora.sh" > "${OUT}/clio_${tag}.log" 2>&1
    echo "clio ${tag}: rc=$?"
    grep -a 'RESULT\|STALE\|elapsed' "${OUT}/clio_${tag}.log" | head -3
    grep -ah '^BW ' "${OUT}/clio_${tag}"/rank*.log > "${OUT}/clio_${tag}.bw"
    wc -l < "${OUT}/clio_${tag}.bw" | xargs echo "  BW lines:"
  }
  for shape in ${BW_CLIO_SHAPES:-"32:16:8 64:32:8 32:16:8:4GB"}; do
    IFS=: read -r th ba rt pf <<< "${shape}"
    pf=${pf:--}
    clio_run "t${th}b${ba}rt${rt}$([ "${pf}" != - ] && echo "pf${pf}")" "${th}" "${ba}" "${rt}" "${pf}"
  done
fi

# ---- summary ------------------------------------------------------------------
echo
echo "SUMMARY ${NNODES} nodes, $((MB * NNODES / 1024)) GiB per phase (GB/s = 1e9 bytes/s, slowest node)"
for f in "${OUT}"/ior_*.log; do
  [ -f "${f}" ] || continue
  tag=$(basename "${f}" .log)
  awk -v tag="${tag}" '/^Max Write:/ {w=$3} /^Max Read:/ {r=$3}
       END { if (w != "") printf "  %-28s write %7.2f GB/s   read %7.2f GB/s\n", tag, w*1048576/1e9, r*1048576/1e9;
             else printf "  %-28s (no result)\n", tag }' "${f}"
done
for f in "${OUT}"/clio_*.bw; do
  [ -f "${f}" ] || continue
  tag=$(basename "${f}" .bw)
  awk -v tag="${tag}" -v n="${NNODES}" '
    { for (i = 1; i <= NF; ++i) { split($i, kv, "="); v[kv[1]] = kv[2] }
      bytes = v["bytes"] + 0; lines++
      if (v["write_ms"] + 0 > w) w = v["write_ms"] + 0
      if (v["read_ms"] + 0 > r) r = v["read_ms"] + 0
      if (v["shift_ms"] + 0 > s) s = v["shift_ms"] + 0
      if (v["rewrite_ms"] + 0 > rw) rw = v["rewrite_ms"] + 0
      if (v["reread_ms"] + 0 > rr) rr = v["reread_ms"] + 0
      err += v["get_errors"] + v["put_errors"] + v["mismatches"] }
    END { if (lines == 0) { printf "  %-28s (no result)\n", tag; exit }
          tot = bytes * lines / 1e9
          printf "  %-28s write %7.2f GB/s   read %7.2f GB/s   read-shifted %7.2f GB/s   rewrite %7.2f GB/s   re-read %7.2f GB/s   (%d/%d nodes, errors %d)\n",
                 tag, tot / (w / 1e3), tot / (r / 1e3), tot / (s / 1e3), rw > 0 ? tot / (rw / 1e3) : 0, rr > 0 ? tot / (rr / 1e3) : 0, lines, n, err }' "${f}"
done
date
echo "BW DONE"
