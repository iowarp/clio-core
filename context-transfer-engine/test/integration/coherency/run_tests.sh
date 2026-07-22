#!/bin/bash
# CTE multi-node COHERENCY integration test (Docker, 4 nodes).
#
# Portable counterpart to test/integration/distributed_slurm (which needs Slurm +
# Apptainer and is cluster-specific). Runs the same test_cte_coherency binary --
# write-only / read-only / append-only with every node a client -- on a local
# 4-container cluster, so it can be verified on a laptop.
#
#   ./run_tests.sh                       # run the suite
#   HOST_WORKSPACE=/abs/path ./run_tests.sh
#   KEEP_UP=1 ./run_tests.sh             # leave containers running to inspect
#
# Requires: docker + docker compose, and a build at $WORKSPACE/build/bin
# containing clio_run and test_cte_coherency.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

NODES=4
COMPOSE="docker compose"
docker compose version >/dev/null 2>&1 || COMPOSE="docker-compose"

# Workspace that gets mounted at /workspace inside the containers.
: "${HOST_WORKSPACE:=${IOWARP_CORE_ROOT:-$(cd "$HERE/../../../.." && pwd)}}"
export HOST_WORKSPACE
export HOST_UID="${HOST_UID:-$(id -u)}" HOST_GID="${HOST_GID:-$(id -g)}"

echo "=== CTE coherency (docker, ${NODES} nodes) ==="
echo "workspace : $HOST_WORKSPACE"
echo "image     : ${IOWARP_DOCKER_IMAGE:-iowarp/deps-cpu:latest}"

for b in clio_run test_cte_coherency; do
  if [ ! -x "$HOST_WORKSPACE/build/bin/$b" ]; then
    echo "ERROR: $HOST_WORKSPACE/build/bin/$b not found or not executable."
    echo "       Build it first:  cmake --build build --target $b"
    exit 1
  fi
done

cleanup() {
  if [ "${KEEP_UP:-0}" = "1" ]; then
    echo "KEEP_UP=1 -- leaving containers up ($COMPOSE down -v to clean)"
    return
  fi
  $COMPOSE down -v --remove-orphans >/dev/null 2>&1
}
trap cleanup EXIT

# -v removes the shared barrier volume: stale READY/TESTDONE files from an
# earlier run would otherwise let nodes skip the barriers.
$COMPOSE down -v --remove-orphans >/dev/null 2>&1

echo "=== bringing up cluster ==="
if ! $COMPOSE up --abort-on-container-exit --no-color; then
  echo "--- compose reported a failure; per-node logs follow ---"
fi

# Aggregate: every node runs the test, so the suite fails if ANY node fails.
rc=0
echo
echo "=== per-node result ==="
for i in $(seq 1 $NODES); do
  name="cte-coherency-node$i"
  code=$(docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null || echo "?")
  passed=$($COMPOSE logs --no-color "coh-node$i" 2>/dev/null | grep -c "\[PASS\] COHERENCY")
  echo "node$i (rank $((i-1))): exit=$code  coherency_cases_passed=$passed"
  # 3 cases per node: write-only, read-only, append-only. Anything less is a
  # failure even if the process exited 0 (e.g. a filter matched nothing).
  if [ "$code" != "0" ] || [ "$passed" -lt 3 ]; then rc=1; fi
done

echo
if [ "$rc" = "0" ]; then
  echo "RESULT: PASS -- ${NODES}/${NODES} nodes, 3/3 coherency workloads each"
else
  echo "RESULT: FAIL"
  echo "--- logs (tail) ---"
  $COMPOSE logs --no-color --tail=40 2>/dev/null | tail -60
fi
exit $rc
