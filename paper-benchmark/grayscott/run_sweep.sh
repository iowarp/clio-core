#!/usr/bin/env bash
# The paper's Gray-Scott sweep: one workload, every selection policy.
#
#   ./run_sweep.sh [--L N] [--steps N] [--gap N] [--chunk B] [--regime R]
#                  [--repeats N] [--results DIR] [--configs "a b c"]
#
# Every policy re-runs the simulation, as in the WarpX sweep: the compression
# happens inside the VOL while the application writes, so there is no
# dump-and-replay phase to hold the bytes fixed. Gray-Scott is deterministic
# for a fixed (L, steps, seed, regime), so the policies do see identical data
# -- unlike WarpX, that is a property this workload actually guarantees.
#
# Then: ../collect.py results/  ->  summary.csv + summary.md
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
L=128 STEPS=200 GAP=25 CHUNK=1048576 REGIME=spots REPEATS=1
RESULTS="$HERE/results"
CONFIGS="dynamic dynamic-ratio learn explore best static-zstd static-zstd-s4 static-zstd-s8"
while [ $# -gt 0 ]; do
  case "$1" in
    --L) L=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --regime) REGIME=$2; shift 2;;
    --repeats) REPEATS=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --configs) CONFIGS=$2; shift 2;;
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
mkdir -p "$RESULTS"
echo "configs: $CONFIGS"
FAILED=0
for cfg in $CONFIGS; do
  for ((r=1; r<=REPEATS; r++)); do
    tag=$cfg; [ "$REPEATS" -gt 1 ] && tag="${cfg}_r${r}"
    "$HERE/run_config.sh" "$cfg" --L "$L" --steps "$STEPS" --gap "$GAP" \
        --chunk "$CHUNK" --regime "$REGIME" --results "$RESULTS" --tag "$tag" \
        || { FAILED=$((FAILED+1)); echo "   ^ $tag FAILED"; }
  done
done
echo
"$HERE/../collect.py" "$RESULTS" || true
[ "$FAILED" -eq 0 ] || echo "$FAILED run(s) failed"
exit 0
