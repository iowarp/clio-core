#!/bin/bash
# Distributed (2-node docker) run of a paged workload benchmark.
#
#   ./run_workloads_distributed.sh kmeans
#   ./run_workloads_distributed.sh weights
#   ./run_workloads_distributed.sh all
#
# Requires: nvidia container toolkit; the repo built into build/ (or set
# BUILD_DIR); the iowarp/deps-cpu image.
#
# WHAT THIS GATES. Not just "it ran": each workload's distributed result is
# compared against its own SINGLE-NODE reference, because these benches all
# report a checksum whose whole purpose is to be configuration-independent.
# A 2-node run that merely exits 0 proves nothing -- the failure mode these
# workloads actually have is a plausible, wrong number.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$REPO_ROOT}"
export HOST_UID=$(id -u) HOST_GID=$(id -g)
export GVW_NODES="${GVW_NODES:-2}"
BIN_DIR="$HOST_WORKSPACE/${BUILD_DIR:-build}/bin"

# Per-workload: the binary, the deck, and how to pull the gated number out of
# its summary line. The decks are small on purpose -- this harness is testing
# the cross-node path, not throughput.
deck() {
  WITNESS=""
  case "$1" in
    kmeans)
      BENCH=clio_kmeans_paged_bench
      ARGS="--data-mb 64 --blocks 8 --iters 3 --repeat 1"
      # Float sums under atomicAdd are not bit-reproducible; this bench
      # documents ~1e-8 relative spread across page/block layouts, so the
      # comparison below is relative and NOT an equality test.
      KEY='centroid_checksum=([0-9.]+)'; TOL=1e-4 ;;
    weights)
      BENCH=clio_weights_paged_bench
      ARGS="--blocks 8 --pages 4 --repeat 1"
      # Integer accumulation commutes, so this one IS exact at any node count.
      KEY='checksum=([A-Z]+)'; TOL=exact
      # checksum=OK ALONE IS A VACUOUS GATE HERE. Each node compares its own
      # partial `got` against its own partial `want`, so a reduction that
      # silently did nothing still passes. checksum_total is the REDUCED,
      # whole-model number: it must differ from the single-node total, or the
      # nodes are not covering different shards / not combining at all.
      # Measured 2-node/1-node ratio for this deck: 2.00002 -- not 1.0 (no
      # reduction) and not exactly 2.0 (both nodes on the same shard).
      WITNESS='checksum_total=([0-9]+)' ;;
    gmx)
      BENCH=clio_gmx_paged_bench
      ARGS="--page-kb 32 --atoms 20000 --repeat 1"
      # Fixed-point science: CONSERVATION, MESH and GATHER are all bit-exact,
      # so this gate needs no tolerance at any node count. That also makes it
      # the loudest of the three -- a stale mesh page or a double-counted bin
      # fails outright rather than drifting.
      KEY='(ALL GATES PASS)'; TOL=exact ;;
    *) echo "unknown workload: $1" >&2; return 2 ;;
  esac
}

extract() { sed -nE "s/.*${1}.*/\\1/p" "$2" | head -1; }

run_one() {
  local wl="$1"; deck "$wl"
  echo "=== $wl: single-node reference"
  local ref_log="/tmp/gvw_${wl}_ref.log"
  ( cd "$BIN_DIR" && ./"$BENCH" $ARGS ) > "$ref_log" 2>&1 || {
    echo "  reference run FAILED"; tail -5 "$ref_log"; return 1; }
  local ref; ref="$(extract "$KEY" "$ref_log")"
  echo "  reference: $ref"

  echo "=== $wl: 2-node distributed"
  cd "$SCRIPT_DIR"
  # A leftover barrier file would let a node skip the wait entirely.
  rm -f "$SCRIPT_DIR"/.done_* 2>/dev/null || true
  export GVW_BENCH="$BENCH" GVW_ARGS="$ARGS"
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  docker compose up -d gvw-node1 gvw-node2
  docker compose logs -f --no-color > "/tmp/gvw_${wl}_dist.log" 2>&1 &
  local logs_pid=$!
  local rc=0 code
  # Wait for EVERY node rather than the first exit: --abort-on-container-exit
  # turns a passing run into rc=143.
  for n in gvw-node1 gvw-node2; do
    code=$(docker wait "$n"); echo "  == $n exited $code"
    [ "$code" -eq 0 ] || rc=1
  done
  kill "$logs_pid" 2>/dev/null || true
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  [ "$rc" -eq 0 ] || { echo "  DISTRIBUTED RUN FAILED"; tail -25 "/tmp/gvw_${wl}_dist.log"; return 1; }

  local got; got="$(extract "$KEY" "/tmp/gvw_${wl}_dist.log")"
  echo "  distributed: $got"
  if [ -z "$got" ]; then echo "  GATE FAIL: no checksum in the distributed log"; return 1; fi
  if [ "$TOL" = exact ]; then
    [ "$got" = "$ref" ] || { echo "  GATE FAIL: $got != $ref"; return 1; }
  else
    awk -v a="$ref" -v b="$got" -v t="$TOL" 'BEGIN{
      d = (a>b?a-b:b-a); r = (a!=0 ? d/(a<0?-a:a) : d);
      if (r > t) { printf "  GATE FAIL: rel %.3g > %s\n", r, t; exit 1 }
      printf "  GATE PASS: rel %.3g <= %s\n", r, t }' || return 1
  fi
  # The load-bearing check: with a witness declared, the distributed value
  # must differ from the single-node one. Equality means the extra nodes
  # changed nothing, which is the failure this whole harness exists to catch.
  if [ -n "$WITNESS" ]; then
    local wref wgot
    wref="$(extract "$WITNESS" "$ref_log")"
    wgot="$(extract "$WITNESS" "/tmp/gvw_${wl}_dist.log")"
    echo "  witness: 1-node=$wref 2-node=$wgot"
    if [ -z "$wgot" ] || [ -z "$wref" ]; then
      echo "  GATE FAIL: witness missing"; return 1
    fi
    if [ "$wref" = "$wgot" ]; then
      echo "  GATE FAIL: witness identical -- the second node contributed"
      echo "             nothing, so the reduction is not load-bearing"
      return 1
    fi
  fi
  echo "  $wl OK"
}

TARGET="${1:-all}"
if [ "$TARGET" = all ]; then
  rc=0
  for wl in kmeans weights gmx; do run_one "$wl" || rc=1; done
  exit $rc
fi
run_one "$TARGET"
