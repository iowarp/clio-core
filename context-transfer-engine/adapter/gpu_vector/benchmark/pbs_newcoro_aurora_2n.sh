#!/usr/bin/env bash
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A IOWarp
#PBS -j oe
#
# Run ONE ported paged benchmark across TWO Aurora nodes, one rank per node.
#
# The single-node script's twin (pbs_newcoro_aurora.sh) with the distributed
# plumbing added:
#
#   * a hostfile written from $PBS_NODEFILE, one node per line;
#   * a runtime config: the benchmark's own single-node config (left behind
#     in build-spike/run_<name>/ by a single-node run) with
#     `networking.hostfile` added. CLIO_SERVER_CONF names it, and every
#     benchmark honours an already-set CLIO_SERVER_CONF instead of writing
#     its own -- that hand-off is the documented way to point them at a
#     cluster. Composing with `pool_query: local` on each node is what the
#     distributed integration test does too: the pool id is fixed, node
#     membership comes from the hostfile, and blobs route by hash to the
#     node that owns the container.
#   * `mpiexec -n 2 -ppn 1`, each rank given `--nodes 2 --node <rank>`.
#
# Five-minute walltime, 90 s cap on each rank. Same env as the single-node
# script: one tile, AOT binaries, the shared-FS SYCL cache.
set -u

: "${BENCH_NAME:?set BENCH_NAME}"
: "${BENCH_ARGS:=}"
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
EXE="${ROOT}/build-spike/${BENCH_EXE:-clio_${BENCH_NAME}_paged_newcoro_aot}"
# The config the single-node run left behind. Most benchmarks write
# gv_<name>_bench.yaml; lammps_md writes gpu_vector_md.yaml, hence the glob.
TEMPLATE=$(ls "${ROOT}/build-spike/run_${BENCH_NAME}"/gv_${BENCH_NAME}_bench.yaml \
              "${ROOT}/build-spike/run_${BENCH_NAME}"/*.yaml 2>/dev/null | head -1)
RUNDIR=${RUNDIR:-${ROOT}/build-spike/run2n_${BENCH_NAME}}
NRANKS=2

echo "=== ${BENCH_NAME} x${NRANKS} nodes: $(sort -u "$PBS_NODEFILE" | tr '\n' ' ') ==="
echo "exe:  ${EXE}"
echo "args: ${BENCH_ARGS} --nodes ${NRANKS} --node <rank>"
test -x "${EXE}" || { echo "NO EXECUTABLE -- build it first"; exit 2; }
test -n "${TEMPLATE}" && test -f "${TEMPLATE}" || { echo "NO CONFIG TEMPLATE in build-spike/run_${BENCH_NAME}/ -- run the single-node job first"; exit 2; }

mkdir -p "${RUNDIR}"
cd "${RUNDIR}"
ulimit -c unlimited
rm -f ./rank*.log ./hostfile ./clio_2n.yaml

# One hostname per line. The runtime resolves each entry and picks the one
# bound to a local interface as itself, so PBS's names are enough.
sort -u "${PBS_NODEFILE}" > hostfile
if [ "$(wc -l < hostfile)" -ne "${NRANKS}" ]; then
  echo "expected ${NRANKS} distinct nodes, got: $(cat hostfile | tr '\n' ' ')"
  exit 2
fi
# The template's `networking:` block gains the hostfile; everything else --
# port, workers, tiers sized for the single-node args -- is kept. The
# benchmarks split --data-mb across nodes, so per-node data only shrinks.
# cte_core MUST BE 512.0. A benchmark given an external config assumes the
# cluster layout its own comments describe -- cte_core at 512.0, no
# compressor -- while weights' self-written single-node config puts cte_core
# at 513.0 behind a compressor slot. Templating that unchanged sent every
# weights put to a pool with no container (put_errors=4, "SEED DID NOT
# CONVERGE"). Renumber it; the other four already compose 512.0.
# NEIGHBORHOOD 1: each node's cte_core registers targets only on its own
# node. The default (4) registers every node's bdevs on every node, and
# the placer then writes a GPU page to the OTHER node's HBM -- a bdev write
# whose bulk is the device frame. Every such send stages the frame through
# the host with a synchronous SYCL copy inside the one admin Send task
# (measured 2 ms per 64 KB page on rank 0 of kmeans, 4.1 s of a 6.5 s run),
# while the transfer engine's own remote-owner path bounces in the PutBlob
# coroutine and sends host memory. Blob OWNERSHIP still spreads by hash
# across both containers; only the storage behind each owner stays local.
awk -v hf="${RUNDIR}/hostfile" '
  /pool_id: "513.0"/ { sub(/"513.0"/, "\"512.0\"") }
  { print }
  /pool_id: "512.0"/ { print "    targets:"; print "      neighborhood: 1" }
  /^networking:/ { print "  hostfile: \"" hf "\"" }
' "${TEMPLATE}" > clio_2n.yaml
export CLIO_SERVER_CONF="${RUNDIR}/clio_2n.yaml"
echo "--- config ---"
sed -n '1,4p' clio_2n.yaml

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

echo "--- run (90s cap per rank, ${NRANKS} ranks) ---"
start=$SECONDS
# Each rank writes its own log; the job log gets both, filtered, afterwards.
# --envall carries every export above to the ranks.
mpiexec -n "${NRANKS}" --ppn 1 --envall bash -c '
  r=${PALS_RANKID:-${PMI_RANK:-0}}
  cd "$BENCH_RANK_DIR"
  timeout --signal=TERM --kill-after=10s 90 \
    stdbuf -oL -eL "$BENCH_RANK_EXE" $BENCH_RANK_ARGS --nodes "$BENCH_RANK_N" --node "$r" \
    > "rank$r.log" 2>&1
  rc=$?
  echo "rank $r on $(hostname) exit=$rc" >> "rank$r.log"
  # ALWAYS 0 HERE. PALS tears the whole job down the moment one rank exits
  # non-zero, and a rank that was about to print its gate then never does;
  # the job result is derived from the per-rank exit lines below instead.
  exit 0
'
mrc=$?
echo "--- elapsed $((SECONDS - start))s, mpiexec exit=${mrc} ---"
rc=${mrc}
for r in $(seq 0 $((NRANKS - 1))); do
  echo "----- rank ${r} -----"
  grep --line-buffered -vE "LoadBalance|\[#78[15]|INFO|SUCCESS|WARNING" "rank${r}.log" | tail -40
  rrc=$(grep -oE "^rank ${r} on .* exit=[0-9]+" "rank${r}.log" | tail -1 | grep -oE "[0-9]+$")
  [ -z "${rrc}" ] && rrc=99
  [ "${rrc}" -gt "${rc}" ] && rc=${rrc}
done

case "${rc}" in
  0)   echo "RESULT ${BENCH_NAME}x${NRANKS}: OK" ;;
  124) echo "RESULT ${BENCH_NAME}x${NRANKS}: TIMEOUT (a rank exceeded the 90s cap)" ;;
  *)   echo "RESULT ${BENCH_NAME}x${NRANKS}: FAILED rc=${rc}" ;;
esac
exit "${rc}"
