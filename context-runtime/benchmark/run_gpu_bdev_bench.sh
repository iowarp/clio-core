#!/usr/bin/env bash
# Runs the GPU mem-bdev benchmark twice — async (yield-poll) then sync
# (force-sync, the old worker-blocking path) — over an identical device
# workload, so you can diff the concurrency scaling directly.
#
# Usage:
#   run_gpu_bdev_bench.sh [-- <extra clio_run_gpu_bdev_bench args>]
# Env:
#   BIN_DIR   build bin dir holding clio_run_gpu_bdev_bench (default: ./bin)
#   IO_SIZE, NUM_OPS, DEPTHS, TIER  optional passthrough overrides
set -euo pipefail

BIN_DIR="${BIN_DIR:-./bin}"
BIN="${BIN_DIR}/clio_run_gpu_bdev_bench"
[ -x "$BIN" ] || { echo "not found: $BIN (set BIN_DIR)"; exit 1; }

ARGS=()
[ -n "${IO_SIZE:-}" ] && ARGS+=(--io-size "$IO_SIZE")
[ -n "${NUM_OPS:-}" ] && ARGS+=(--num-ops "$NUM_OPS")
[ -n "${DEPTHS:-}" ]  && ARGS+=(--depths "$DEPTHS")
[ -n "${TIER:-}" ]    && ARGS+=(--tier-bytes "$TIER")
# Anything after `--` is passed straight through.
while [ $# -gt 0 ]; do case "$1" in --) shift; ARGS+=("$@"); break;; *) shift;; esac; done

export CLIO_REPO_PATH="${CLIO_REPO_PATH:-$BIN_DIR}"
export LD_LIBRARY_PATH="${BIN_DIR}:${LD_LIBRARY_PATH:-}"
export CLIO_BIND_ADDR="${CLIO_BIND_ADDR:-127.0.0.1}"

echo "########## ASYNC (yield-poll) ##########"
env -u CLIO_BDEV_FORCE_SYNC "$BIN" "${ARGS[@]}"

echo
echo "########## SYNC (old, worker blocks) ##########"
CLIO_BDEV_FORCE_SYNC=1 "$BIN" "${ARGS[@]}"
