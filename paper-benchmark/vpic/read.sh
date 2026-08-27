#!/usr/bin/env bash
# Phase 3: a SEPARATE process, with its own runtime, reads back what a sweep
# stored -- and proves it decompresses from cold storage.
#
#   ./run_config.sh static-zstd-s4
#   ./read.sh --run static-zstd-s4
#
# Why this proves more than the writer's own --verify. That check runs inside
# the process that just handed the compressor those bytes, with its caches
# warm. This one starts cold: CLIO_RESTART=1 replays the metadata log so a
# fresh process can find blobs a previous one wrote, and the only things shared
# with the writer are the store directory and blobs.csv (names, sizes,
# digests). The field files are never opened.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
BIN=${BIN:-$BUILD/bin/neuropress_field_replay}
RESULTS="$HERE/results"
RUN=""
# Write every decompressed blob out as bytes, so something OUTSIDE this
# program can compare them against the simulation's own output. The
# digest above only proves the round trip is self-consistent -- it is
# computed by the same code that computed the original one. This is what
# the original-vs-decompressed figures read.
DUMP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2;;
    --dump-decompressed) DUMP=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$DUMP" ] && mkdir -p "$DUMP"
[ -n "$RUN" ] || { echo "need --run NAME (a directory under $RESULTS)" >&2; exit 2; }
STORE=$RESULTS/$RUN
[ -f "$STORE/blobs.csv" ]   || { echo "no $STORE/blobs.csv" >&2; exit 1; }
[ -f "$STORE/compose.yaml" ] || { echo "no $STORE/compose.yaml" >&2; exit 1; }
[ -x "$BIN" ] || { echo "missing $BIN" >&2; exit 1; }
export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "== cold read from $STORE (writer process is long gone)"
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    "$BIN" --readback "$STORE/blobs.csv" --tag "vpic_$RUN" \
        ${DUMP:+--dump-decompressed "$DUMP"} 2> "$STORE/read.log"
RC=$?
[ $RC -ne 0 ] && { echo "-- last lines of $STORE/read.log:"; grep -vE "DEBUG|INFO" "$STORE/read.log" | tail -12; }
echo "-- exit=$RC (0 = every blob decompressed to its original bytes)"
exit $RC
