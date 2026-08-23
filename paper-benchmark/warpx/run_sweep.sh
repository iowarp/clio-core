#!/usr/bin/env bash
# The paper's WarpX sweep: one in-situ workload, every selection policy.
#
#   ./run_sweep.sh [--ncell "NX NY NZ"] [--steps N] [--interval N] [--chunk B]
#                  [--repeats N] [--results DIR] [--configs "a b c"]
#
# Every policy is a full WarpX run with Clio's VOL in the loop -- there is no
# dump-and-replay phase, because the whole point of this workload is that a
# stock simulation gets compressed with no changes to it at all.
#
# NOTE ON COMPARABILITY: because each policy re-runs the simulation, the
# policies do not see byte-identical inputs the way the Nyx and VPIC sweeps do.
# WarpX on one GPU is deterministic for a fixed input, so in practice they
# match, but that is a property of the run rather than something this harness
# enforces.
#
# Then: ../collect.py results/  ->  summary.csv + summary.md
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NCELL="64 64 512" STEPS=40 INTERVAL=10 CHUNK=1048576 REPEATS=1
RESULTS="$HERE/results"
CONFIGS="dynamic dynamic-ratio learn explore best static-zstd static-zstd-s4 static-zstd-s8"
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --interval) INTERVAL=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --repeats) REPEATS=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --configs) CONFIGS=$2; shift 2;;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$RESULTS"
echo "WarpX in-situ sweep: ${NCELL// /x}, $STEPS steps, diag every $INTERVAL, chunk $CHUNK"
echo "configs: $CONFIGS"
echo

FAILED=0
for rep in $(seq 1 "$REPEATS"); do
  for c in $CONFIGS; do
    tag=$c; [ "$REPEATS" -gt 1 ] && tag="${c}__r${rep}"
    "$HERE/run_config.sh" "$c" --ncell "$NCELL" --steps "$STEPS" \
        --interval "$INTERVAL" --chunk "$CHUNK" \
        --results "$RESULTS" --tag "$tag" || { FAILED=$((FAILED+1)); echo "   ^ $tag FAILED"; }
    # Each run writes a full set of openPMD files; keeping eight of them is
    # gigabytes of duplicate native output nobody reads.
    rm -rf "$RESULTS/$tag/run/diags"
    echo
  done
done

echo "sweep done; $FAILED run(s) failed"
[ -x "$HERE/../collect.py" ] && "$HERE/../collect.py" "$RESULTS"
exit $(( FAILED > 0 ))
