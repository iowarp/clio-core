#!/bin/bash
# 2-node distributed run of the md benchmark (docker).
#
# Requires: nvidia container toolkit; the repo built into build/ (or set
# BUILD_DIR); iowarp/deps-cpu image.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$REPO_ROOT}"
export HOST_UID=$(id -u) HOST_GID=$(id -g)
if [ -z "${CUSZP_LIB_DIR:-}" ]; then
  BIN="$HOST_WORKSPACE/${BUILD_DIR:-build}/bin/clio_gpu_vector_md_bench"
  CUSZP_LIB="$(ldd "$BIN" 2>/dev/null | awk '/libcuSZp/ {print $3}')"
  [ -n "$CUSZP_LIB" ] && export CUSZP_LIB_DIR="$(dirname "$CUSZP_LIB")"
fi
cd "$SCRIPT_DIR"
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
# Wait for EVERY node rather than the first exit -- see distributed/ for why
# --abort-on-container-exit turns a passing run into rc=143.
docker compose up -d
docker compose logs -f --no-color &
LOGS_PID=$!
rc=0
for n in gvmb-node1 gvmb-node2; do
  code=$(docker wait "$n")
  echo "== $n exited $code"
  [ "$code" -eq 0 ] || rc=1
done
kill "$LOGS_PID" 2>/dev/null || true
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
exit $rc
