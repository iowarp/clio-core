#!/bin/bash
# 2-node distributed generational put-get (docker).
#
# What it gates: that a page written by one node and re-read by another EVERY
# ROUND comes back as that round's bytes -- the coherence case the MD stencil
# needs and the one distributed/ deliberately sidesteps by using a fresh
# region per round. There is no barrier between publish and read; the
# generation is the barrier, so a broken generation shows up as round r
# reading round r-1's pattern.
#
# It also gates that the exchange HAPPENED: each node asserts a nonzero halo
# fault count, so the run cannot pass while serving its own cached copy.
#
# Requires: nvidia container toolkit; the repo built into build/ (or set
# BUILD_DIR); iowarp/deps-cpu image.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$REPO_ROOT}"
export HOST_UID=$(id -u) HOST_GID=$(id -g)
if [ -z "${CUSZP_LIB_DIR:-}" ]; then
  BIN="$HOST_WORKSPACE/${BUILD_DIR:-build}/bin/test_cte_generation_distributed"
  CUSZP_LIB="$(ldd "$BIN" 2>/dev/null | awk '/libcuSZp/ {print $3}')"
  [ -n "$CUSZP_LIB" ] && export CUSZP_LIB_DIR="$(dirname "$CUSZP_LIB")"
fi
cd "$SCRIPT_DIR"
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
# Wait for EVERY node rather than the first exit: --abort-on-container-exit
# tears the cluster down the moment any node finishes and SIGTERMs the others
# mid-summary, which turns a passing run into rc=143. The verdict is each
# node's own exit code. (Same lesson as distributed/run_distributed.sh.)
docker compose up -d
docker compose logs -f --no-color &
LOGS_PID=$!
rc=0
for n in gvgen-node1 gvgen-node2; do
  code=$(docker wait "$n")
  echo "== $n exited $code"
  [ "$code" -eq 0 ] || rc=1
done
kill "$LOGS_PID" 2>/dev/null || true
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
exit $rc
