#!/usr/bin/env bash
# Phase 2 of 2: a SEPARATE process, with its own runtime, reads back what
# run.sh stored -- and proves it decompresses from cold storage.
#
#   ./run.sh  --box 10 --steps 100 --gap 50 --static nvcomp-zstd
#   ./read.sh
#
# CLIO_RESTART=1 replays the metadata log so a fresh process can find blobs a
# previous one wrote. The reader shares two things with the writer: the store
# directory, and blobs.csv (names, sizes, digests). No LAMMPS runs here, and
# there is no native file to fall through to: the compressed tier is the only
# copy of the data, so bytes that come back correct came from it.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-/home/cc/clio-core}
BUILD=${BUILD:-$REPO/build}
BIN=$BUILD/bin/neuropress_lammps_lib
STORE=${STORE:-$HERE/store}
while [ $# -gt 0 ]; do
  case "$1" in
    --store) STORE=$2; shift 2;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -x "$BIN" ] || { echo "missing $BIN" >&2; exit 1; }
[ -f "$STORE/blobs.csv" ] || { echo "no $STORE/blobs.csv -- run ./run.sh first" >&2; exit 1; }
[ -f "$STORE/compose.yaml" ] || { echo "no $STORE/compose.yaml" >&2; exit 1; }
export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "== cold read from $STORE (writer process is long gone)"
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
    CLIO_LMP_COMPRESSOR_POOL=512.0 \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warn}" \
    "$BIN" --readback "$STORE/blobs.csv" 2> "$STORE/read.log"
RC=$?
grep -c "stored_compressed=1 -> inverting codec" "$STORE/read.log" 2>/dev/null | sed 's/^/-- blobs the compressor inverted a codec for: /'
[ $RC -ne 0 ] && { echo "-- last lines of $STORE/read.log:"; tail -20 "$STORE/read.log"; }
echo "-- exit=$RC (0 = every blob decompressed to its original bytes)"
exit $RC
