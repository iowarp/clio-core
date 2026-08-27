#!/bin/bash
# Distributed run of the md benchmark (docker). GVMB_NODES=2 (default) or 4;
# 4-node runs use the "four" compose profile, hostfile4 and gvmb_conf4.yaml
# (pass GVMB_CONF1..2=gvmb_conf4.yaml so nodes 1-2 read the 4-entry hostfile
# too -- nodes 3-4 default to it already).
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
# The peer barrier is a file each node touches when its bench finishes; a
# leftover from a previous run would let a node skip the wait entirely.
rm -f "$SCRIPT_DIR"/.done_* 2>/dev/null || true
GVMB_NODES="${GVMB_NODES:-2}"
export GVMB_NODES
SVCS="gvmb-node1 gvmb-node2"
if [ "$GVMB_NODES" -ge 3 ]; then
  export COMPOSE_PROFILES=four
  SVCS="$SVCS gvmb-node3"
fi
[ "$GVMB_NODES" -ge 4 ] && SVCS="$SVCS gvmb-node4"
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
# Wait for EVERY node rather than the first exit -- see distributed/ for why
# --abort-on-container-exit turns a passing run into rc=143.
docker compose up -d $SVCS
docker compose logs -f --no-color &
LOGS_PID=$!
rc=0
for n in $SVCS; do
  code=$(docker wait "$n")
  echo "== $n exited $code"
  [ "$code" -eq 0 ] || rc=1
done
kill "$LOGS_PID" 2>/dev/null || true
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
exit $rc
