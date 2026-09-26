#!/usr/bin/env bash
# The scale ladder every benchmark configuration climbs before it is allowed
# near a multi-node queue. All rungs run on ONE node with co-located runtimes
# (run_colocated.sh), so the whole ladder costs a few minutes of a debug node:
#
#   rung 1  1 rank,  resident       (the kernel and its checksum)
#   rung 2  2 ranks, resident       (remote pages, generations, tags)
#   rung 3  2 ranks, out of core    (fetch / flush / eviction across nodes)
#   rung 4  4 ranks, out of core    (broadcast fan-out, four-way hashing)
#
# Gates, per rung: every rank exits 0; every rank reports the same checksum;
# the checksum matches rung 1's within BENCH_LADDER_RTOL (default 1e-6:
# kmeans's checksum is an atomically accumulated float whose summation order
# follows the layout, so bit-equality across rank counts is not expected); an
# out-of-core rung must report evictions > 0 (a run that never evicted proves
# nothing). The ladder stops at the first failed rung and leaves its logs in
# <workdir>/rung<k>/ for the diagnosis.
#
#   bench_ladder.sh <workload> <workdir> [exe]
#
# Workloads: kmeans, grayscott. Environment: BENCH_PERNODE_MB per-node deck
# (default 2048), BENCH_CAP per rank (default 300), plus what run_colocated.sh
# and bench_config.sh read. Exit status: 0 when every rung passes.
set -u
wl=${1:?usage: bench_ladder.sh <workload> <workdir> [exe]}
workdir=${2:?workdir}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${here}/../../../.." && pwd)
pernode=${BENCH_PERNODE_MB:-2048}
export BENCH_CAP=${BENCH_CAP:-300}

# Per-workload: binary, argument line for a deck of <total MB> with <slots>
# frames per block, and how to read the result line.
case "${wl}" in
  kmeans)
    exe=${3:-${root}/build-spike/clio_kmeans_paged_newcoro_aot_x_ckpt2}
    args() { echo "--data-mb $1 --iters 3 --page-kb 1024 --blocks 256 --threads 256 --slots $2 --publish-seed --repeat 1"; }
    result_re='^KMEANS mode=paged'; checksum_key='centroid_checksum'
    blocks=256; ooc_slots=2 ;;
  grayscott)
    exe=${3:-${root}/build-spike/clio_grayscott_paged_newcoro_aot_x_fc2_ct8}
    export IGC_FunctionControl=2
    args() { echo "--data-mb $1 --steps 3 --page-kb 1024 --blocks 128 --threads 256 --slots $2 --two-phase --ooc --repeat 1"; }
    result_re='^GRAYSCOTT mode=paged'; checksum_key='v_checksum'
    blocks=128; ooc_slots=8 ;;
  *) echo "bench_ladder: unknown workload ${wl}"; exit 2 ;;
esac
# <blocks> x 1 MB pages: <pernode> MB resident needs pernode/blocks slots.
# The vector keeps at least 8 frames per block whatever --slots says, so an
# out-of-core rung needs a deck above 8 x blocks MB or it is silently
# resident; the out-of-core rungs use twice that floor at least.
resident_slots=$(( pernode / blocks + 8 ))
ooc_pernode=$(( pernode > 16 * blocks ? pernode : 16 * blocks ))

# field <log> <key>: the value of key=... on the result line.
field() { grep -a "${result_re}" "$1" | tail -1 | grep -oE "${2}=[-0-9.]+" | cut -d= -f2; }

rung() {  # rung <k> <ranks> <slots> <label>
  local k=$1 n=$2 slots=$3 label=$4 r rc cs cs0 ev per
  local dir="${workdir}/rung${k}"  # separate: a single local expands before assigning
  per=${pernode}; [ "${label}" = "out of core" ] && per=${ooc_pernode}
  echo "=== rung ${k}: ${n} rank(s), ${label}, ${per} MB/node, ${slots} slots/block"
  if ! bash "${here}/run_colocated.sh" "${n}" "${dir}" "${exe}" $(args $(( per * n )) "${slots}") > "${dir}.out" 2>&1; then
    rc=$?; echo "FAIL rung ${k}: a rank exited non-zero (see ${dir}.out)"; tail -5 "${dir}.out"; return 1
  fi
  cs0=$(field "${dir}/rank0.log" "${checksum_key}")
  [ -n "${cs0}" ] || { echo "FAIL rung ${k}: no result line in rank0.log"; return 1; }
  for (( r = 1; r < n; ++r )); do
    cs=$(field "${dir}/rank${r}.log" "${checksum_key}")
    [ "${cs}" = "${cs0}" ] || { echo "FAIL rung ${k}: rank ${r} checksum ${cs} != rank 0 ${cs0}"; return 1; }
  done
  if [ -n "${LADDER_CS:-}" ] && ! awk -v a="${cs0}" -v b="${LADDER_CS}" -v t="${BENCH_LADDER_RTOL:-1e-6}" \
        'BEGIN { d = a - b; if (d < 0) d = -d; m = (b < 0 ? -b : b); if (m == 0) m = 1; exit !(d / m <= t) }'; then
    echo "FAIL rung ${k}: checksum ${cs0} vs rung 1's ${LADDER_CS} differs by more than ${BENCH_LADDER_RTOL:-1e-6} (paging changed the answer)"; return 1
  fi
  LADDER_CS=${cs0}
  ev=$(field "${dir}/rank0.log" evicts)
  if [ "${label}" = "out of core" ] && [ "${ev:-0}" -eq 0 ]; then
    echo "FAIL rung ${k}: 0 evictions; the cache held the deck, so nothing was tested"; return 1
  fi
  echo "PASS rung ${k}: checksum ${cs0}, faults $(field "${dir}/rank0.log" faults), evicts ${ev}"
}

mkdir -p "${workdir}"
LADDER_CS=""
rung 1 1 "${resident_slots}" resident   || exit 1
rung 2 2 "${resident_slots}" resident   || exit 1
rung 3 2 "${ooc_slots}"      "out of core" || exit 1
rung 4 4 "${ooc_slots}"      "out of core" || exit 1
echo "LADDER PASS: ${wl} is cleared for multi-node runs"
