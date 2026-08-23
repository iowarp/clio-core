#!/usr/bin/env bash
# The paper's LAMMPS sweep: one workload, every selection policy.
#
#   ./run_sweep.sh [--box N] [--steps N] [--gap N] [--chunk B] [--repeats N]
#                  [--results DIR] [--configs "a b c"] [--cpu]
#                  [--density X] [--temp X] [--cutoff X] [--skin X]
#                  [--every N] [--seed N] [--dt X] [--var K=V]
#
# The physics options are passed straight through to every run, so a whole
# sweep can be repeated at a different density, temperature or timestep
# without editing the deck. See ./run_config.sh -h for their defaults.
#
# Holds the workload constant (same deck, same box, same steps) and varies
# ONLY how a codec is chosen per chunk, so the comparison isolates the
# selection policy. Each run stores its per-chunk CSV, its NeuroPress
# selection log, and -- for exploration -- every candidate it measured.
#
# Then: ./collect.py results/  ->  summary.csv + summary.md
#
# NOTE ON REPEATS: a GPU (Kokkos) run is not bit-reproducible, so repeats of
# the same configuration differ slightly in the bytes they compress. Use
# --cpu for a bit-reproducible trajectory if you need runs that are
# comparable chunk-for-chunk rather than statistically.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BOX=80 STEPS=300 GAP=50 CHUNK=4194304 REPEATS=1 RESULTS="$HERE/results" DEVICE=""
PHYS=()   # physics + extra -var flags, forwarded verbatim to run_config.sh
CONFIGS="dynamic dynamic-ratio learn explore best static-zstd static-zstd-s4 static-zstd-s8"
while [ $# -gt 0 ]; do
  case "$1" in
    --box) BOX=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --density|--temp|--cutoff|--skin|--every|--seed|--dt|--var)
        PHYS+=("$1" "$2"); shift 2;;
    --repeats) REPEATS=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --configs) CONFIGS=$2; shift 2;;
    --cpu) DEVICE="--cpu"; shift;;
    -h|--help) sed -n '2,28p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$RESULTS"
echo "LAMMPS paper sweep: box=$BOX steps=$STEPS gap=$GAP chunk=$CHUNK repeats=$REPEATS"
[ ${#PHYS[@]} -gt 0 ] && echo "physics overrides: ${PHYS[*]}"
echo "configs: $CONFIGS"
echo "results: $RESULTS"
echo

FAILED=0
for rep in $(seq 1 "$REPEATS"); do
  for c in $CONFIGS; do
    tag=$c; [ "$REPEATS" -gt 1 ] && tag="${c}__r${rep}"
    "$HERE/run_config.sh" "$c" --box "$BOX" --steps "$STEPS" --gap "$GAP" \
        --chunk "$CHUNK" --results "$RESULTS" --tag "$tag" $DEVICE \
        "${PHYS[@]+"${PHYS[@]}"}" || { FAILED=$((FAILED+1)); echo "   ^ $tag FAILED"; }
    echo
  done
done

echo "sweep done; $FAILED run(s) failed"
if command -v python3 >/dev/null && [ -x "$HERE/collect.py" ]; then
  "$HERE/collect.py" "$RESULTS"
fi
exit $(( FAILED > 0 ))
