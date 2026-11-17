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
# an out-of-core rung must report evictions > 0 (a run that never evicted
# proves nothing); and paging must not change the answer. How the last gate
# is checked depends on the workload's checksum:
#   intensive (kmeans's centroids): every rung is compared with rung 1 within
#     BENCH_LADDER_RTOL (1e-6: the atomically accumulated float's summation
#     order follows the layout, so bit-equality across rank counts is not
#     expected);
#   extensive (grayscott's v_checksum, a sum over the whole grid): its value
#     depends on the deck, and not linearly (653.24/MB at 2 GB, 655.16/MB at
#     8 GB), so it is compared only with a RESIDENT run of the same deck and
#     rank count, and then exactly -- measured bit-identical resident vs
#     4,400 evictions at 1 and 2 ranks.
# The ladder stops at the first failed rung and leaves its logs in
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
# frames per block, how to read the result line, and whether the checksum is
# EXTENSIVE (a sum over the whole grid, so it scales with the deck: grayscott's
# v_checksum doubled exactly when rung 2 doubled the deck) or intensive
# (kmeans's centroids). Extensive checksums are compared per MB of deck.
case "${wl}" in
  kmeans)
    exe=${3:-${root}/build-spike/clio_kmeans_paged_newcoro_aot_x_ckpt2}
    args() { echo "--data-mb $1 --iters 3 --page-kb 1024 --blocks 256 --threads 256 --slots $2 --publish-seed --repeat 1"; }
    result_re='^KMEANS mode=paged'; checksum_key='centroid_checksum'
    blocks=256; ooc_slots=2; extensive=0; rtol=1e-6; ooc_mult=16 ;;
  grayscott)
    exe=${3:-${root}/build-spike/clio_grayscott_paged_newcoro_aot_x_fc2_ct8}
    export IGC_FunctionControl=2
    args() { echo "--data-mb $1 --steps 3 --page-kb 1024 --blocks 128 --threads 256 --slots $2 --two-phase --ooc --repeat 1"; }
    result_re='^GRAYSCOTT mode=paged'; checksum_key='v_checksum'
    # Per MB the sum is only APPROXIMATELY deck-invariant: a larger grid has
    # a smaller boundary fraction, measured at 1.07e-6 between 2 and 4 GB.
    # Two arrays (u, v) share the deck, so per block the frames are half the
    # kmeans figure: 2048 MB / (2 x 128) = 8 = exactly the floor, resident.
    blocks=128; ooc_slots=8; extensive=1; rtol=1e-4; ooc_mult=32 ;;
  *) echo "bench_ladder: unknown workload ${wl}"; exit 2 ;;
esac
# <blocks> x 1 MB pages: <pernode> MB resident needs pernode/blocks slots.
# The vector keeps at least 8 frames per block whatever --slots says, so an
# out-of-core rung needs a deck above 8 x blocks MB (x2 for grayscott's two
# arrays) or it is silently resident; the out-of-core rungs use twice that
# floor at least (ooc_mult per workload).
ooc_pernode=$(( pernode > ooc_mult * blocks ? pernode : ooc_mult * blocks ))
# An extensive checksum is compared resident-vs-paged on the SAME deck, so
# every rung uses the out-of-core deck.
[ "${extensive}" -eq 1 ] && pernode=${ooc_pernode}
resident_slots=$(( pernode / blocks + 8 ))

# field <log> <key>: the value of key=... on the result line.
field() { grep -a "${result_re}" "$1" | tail -1 | grep -oE "${2}=[-0-9.]+" | cut -d= -f2; }

rung() {  # rung <k> <ranks> <slots> <label> <ref|check|none>
  local k=$1 n=$2 slots=$3 label=$4 cmp=$5 r rc cs cs0 ev per
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
  if [ "${cmp}" = "check" ]; then
    if [ "${extensive}" -eq 1 ]; then
      [ "${cs0}" = "${LADDER_CS}" ] || { echo "FAIL rung ${k}: checksum ${cs0} != the resident run's ${LADDER_CS} on the same deck (paging changed the answer)"; return 1; }
    elif ! awk -v a="${cs0}" -v b="${LADDER_CS}" -v t="${BENCH_LADDER_RTOL:-${rtol}}" \
          'BEGIN { d = a - b; if (d < 0) d = -d; m = (b < 0 ? -b : b); if (m == 0) m = 1; exit !(d / m <= t) }'; then
      echo "FAIL rung ${k}: checksum ${cs0} vs rung 1's ${LADDER_CS} differs by more than ${BENCH_LADDER_RTOL:-${rtol}} (paging changed the answer)"; return 1
    fi
  fi
  [ "${cmp}" = "ref" ] && LADDER_CS=${cs0}
  ev=$(field "${dir}/rank0.log" evicts)
  if [ "${label}" = "out of core" ] && [ "${ev:-0}" -eq 0 ]; then
    echo "FAIL rung ${k}: 0 evictions; the cache held the deck, so nothing was tested"; return 1
  fi
  echo "PASS rung ${k}: checksum ${cs0}, faults $(field "${dir}/rank0.log" faults), evicts ${ev}"
}

mkdir -p "${workdir}"
LADDER_CS=""
if [ "${extensive}" -eq 0 ]; then
  rung 1  1 "${resident_slots}" resident      ref   || exit 1
  rung 2  2 "${resident_slots}" resident      check || exit 1
  rung 3  2 "${ooc_slots}"      "out of core" check || exit 1
  rung 4  4 "${ooc_slots}"      "out of core" check || exit 1
else
  rung 1  1 "${resident_slots}" resident      none  || exit 1
  rung 2  2 "${resident_slots}" resident      ref   || exit 1
  rung 3  2 "${ooc_slots}"      "out of core" check || exit 1
  rung 4r 4 "${resident_slots}" resident      ref   || exit 1
  rung 4  4 "${ooc_slots}"      "out of core" check || exit 1
fi
echo "LADDER PASS: ${wl} is cleared for multi-node runs"
