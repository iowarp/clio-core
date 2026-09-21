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
awk -v hf="${RUNDIR}/hostfile" '
  { print }
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
  exit $rc
'
rc=$?
echo "--- elapsed $((SECONDS - start))s, mpiexec exit=${rc} ---"
for r in $(seq 0 $((NRANKS - 1))); do
  echo "----- rank ${r} -----"
  grep --line-buffered -vE "LoadBalance|\[#78[15]|INFO|SUCCESS|WARNING" "rank${r}.log" | tail -40
done

case "${rc}" in
  0)   echo "RESULT ${BENCH_NAME}x${NRANKS}: OK" ;;
  124) echo "RESULT ${BENCH_NAME}x${NRANKS}: TIMEOUT (a rank exceeded the 90s cap)" ;;
  *)   echo "RESULT ${BENCH_NAME}x${NRANKS}: FAILED rc=${rc}" ;;
esac
exit "${rc}"
