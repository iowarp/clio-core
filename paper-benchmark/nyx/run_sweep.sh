#!/usr/bin/env bash
# The paper's Nyx sweep: one dataset, every selection policy.
#
#   ./run_sweep.sh [--fields DIR] [--chunk B] [--max-files N] [--repeats N]
#                  [--results DIR] [--configs "a b c"]
#
# Replays the dumps from ./gen_fields.sh. Generate once, sweep many times:
# every policy sees the IDENTICAL bytes, so unlike the LAMMPS sweep (whose GPU
# trajectory is not bit-reproducible) this comparison is exact.
#
# Holds the workload constant (same deck, same box, same steps) and varies
# ONLY how a codec is chosen per chunk, so the comparison isolates the
# selection policy. Each run stores its per-chunk CSV, its NeuroPress
# selection log, and -- for exploration -- every candidate it measured.
#
# Then: ../collect.py results/  ->  summary.csv + summary.md
#
# NOTE ON REPEATS: the replayed bytes are fixed on disk, so repeats measure
# only run-to-run timing variance -- the compression numbers are deterministic
# apart from whatever the selector itself does (learning and exploration carry
# state across chunks).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FIELDS=${FIELDS:-$HERE/fields}
CHUNK=4194304 MAX_FILES=0 REPEATS=1 RESULTS="$HERE/results"
CONFIGS="dynamic dynamic-ratio learn explore best static-zstd static-zstd-s4 static-zstd-s8"
while [ $# -gt 0 ]; do
  case "$1" in
    --fields) FIELDS=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --max-files) MAX_FILES=$2; shift 2;;
    --repeats) REPEATS=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --configs) CONFIGS=$2; shift 2;;
    -h|--help) sed -n '2,28p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$RESULTS"
[ -d "$FIELDS" ] || { echo "no field dumps at $FIELDS -- run ./gen_fields.sh first" >&2; exit 1; }
echo "Nyx paper sweep: fields=$FIELDS chunk=$CHUNK repeats=$REPEATS"
echo "payload: $(du -sh "$FIELDS" | cut -f1) in $(find "$FIELDS" -name '*.f32' | wc -l) file(s)"
echo "configs: $CONFIGS"
echo "results: $RESULTS"
echo

FAILED=0
for rep in $(seq 1 "$REPEATS"); do
  for c in $CONFIGS; do
    tag=$c; [ "$REPEATS" -gt 1 ] && tag="${c}__r${rep}"
    EXTRA=()
    [ "$MAX_FILES" -gt 0 ] && EXTRA=(--max-files "$MAX_FILES")
    "$HERE/run_config.sh" "$c" --fields "$FIELDS" --chunk "$CHUNK" \
        --results "$RESULTS" --tag "$tag" "${EXTRA[@]+"${EXTRA[@]}"}" || { FAILED=$((FAILED+1)); echo "   ^ $tag FAILED"; }
    echo
  done
done

echo "sweep done; $FAILED run(s) failed"
if command -v python3 >/dev/null && [ -x "$HERE/../collect.py" ]; then
  "$HERE/../collect.py" "$RESULTS"
fi
exit $(( FAILED > 0 ))
