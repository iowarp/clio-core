#!/bin/bash
# 4-node distributed gpu_vector stress test (docker).
#
# Requires: nvidia container toolkit; the repo built into build/ (or set
# BUILD_DIR); iowarp/deps-cpu image.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$REPO_ROOT}"
export HOST_UID=$(id -u) HOST_GID=$(id -g)
# cuSZp is a from-source install outside the repo; resolve its host dir off
# the binary so the container can mount it (see /opt/cuszp-lib in compose).
if [ -z "${CUSZP_LIB_DIR:-}" ]; then
  BIN="$HOST_WORKSPACE/${BUILD_DIR:-build}/bin/test_gpu_vector_distributed"
  CUSZP_LIB="$(ldd "$BIN" 2>/dev/null | awk '/libcuSZp/ {print $3}')"
  [ -n "$CUSZP_LIB" ] && export CUSZP_LIB_DIR="$(dirname "$CUSZP_LIB")"
fi
cd "$SCRIPT_DIR"
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
# Wait for EVERY node, not the first exit. --abort-on-container-exit tears
# the cluster down the moment ANY node finishes, which SIGTERMs (143) the
# nodes still printing their PASS summaries -- including the one
# --exit-code-from watched -- so a fully passing run raced into rc=143
# whenever node1 was not the first to finish. The verdict is the four
# nodes' own exit codes, each waited for individually.
docker compose up -d
docker compose logs -f --no-color &
LOGS_PID=$!
rc=0
for n in gvd-node1 gvd-node2 gvd-node3 gvd-node4; do
  code=$(docker wait "$n")
  echo "== $n exited $code"
  [ "$code" -eq 0 ] || rc=1
done
kill "$LOGS_PID" 2>/dev/null || true
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
exit $rc
