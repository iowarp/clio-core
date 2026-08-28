#!/bin/bash
# Run one gpu_vector pipeline (or every yaml in a directory) with the
# environment the cells need:
#   - <build>/bin on PATH (the benchmark binaries)
#   - CLIO_PREFAULT=0: pre-fault the WHOLE RAM tier at compose, so every
#     measurement runs against warmed memory (the pkg also sets this
#     per-cell; exporting it here covers anything jarvis spawns besides).
#
# Usage:
#   ./run.sh workload_understanding/kmeans_mpi_sweep.yaml
#   ./run.sh workload_understanding            # every yaml in the dir
#   BUILD_DIR=build ./run.sh ...               # non-default build tree
#
# The post: section of each yaml runs automatically after the sweep; to
# re-render figures from stored results without re-running:
#   jarvis ppl post yaml <pipeline>.yaml
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
BUILD_DIR="${BUILD_DIR:-build-gv}"
BIN="$REPO_ROOT/$BUILD_DIR/bin"
[ -d "$BIN" ] || { echo "no such build bin dir: $BIN (set BUILD_DIR)"; exit 1; }
export PATH="$BIN:$PATH"
export CLIO_PREFAULT="${CLIO_PREFAULT:-0}"
JARVIS="$REPO_ROOT/.venv/bin/jarvis"
[ -x "$JARVIS" ] || JARVIS=jarvis

run_one() {
  echo "=== $1"
  "$JARVIS" ppl run yaml "$1"
}

TARGET="${1:?usage: ./run.sh <pipeline.yaml | directory>}"
if [ -d "$TARGET" ]; then
  for y in "$TARGET"/*.yaml; do run_one "$y"; done
else
  run_one "$TARGET"
fi
