#!/usr/bin/env bash
# Phase 2: a SEPARATE process reads back what the in-situ run stored.
#
#   ./run.sh --ncell 30 --steps 50 --int 25 --verify
#   ./read.sh
#
# No VPIC here and no Kokkos: the reader is bin/neuropress_field_replay
# --readback, which knows nothing about the simulation. That works because the
# CSV libclio_vpic_insitu.so writes is the one that program parses --
# blob,bytes,fnv1a64,... -- so the two halves share only a store directory and
# a list of names, sizes and digests.
#
# CLIO_RESTART=1 replays the metadata log so a fresh process finds blobs an
# earlier one wrote. There is no file dump to fall back on: the compressed
# tier is the only copy of the data, so bytes that come back correct came
# from it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
BIN=${BIN:-$BUILD/bin/neuropress_field_replay}
STORE=${STORE:-$HERE/store}
TAG=${TAG:-vpic_insitu}
while [ $# -gt 0 ]; do
  case "$1" in
    --store) STORE=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -x "$BIN" ] || { echo "missing $BIN (cmake --build $BUILD --target neuropress_field_replay)" >&2; exit 1; }
export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

rc=0
for csv in "$STORE"/blobs.csv "$STORE"/rank*/blobs*.csv; do
  [ -f "$csv" ] || continue
  dir=$(dirname "$csv")
  echo "== cold read from $dir (the VPIC process is long gone)"
  set +e
  env CLIO_SERVER_CONF="$dir/compose.yaml" \
      CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
      CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
      CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
      "$BIN" --readback "$csv" --tag "$TAG" 2> "$dir/read.log"
  r=$?
  set -e
  [ $r -ne 0 ] && { rc=$r; echo "-- last lines of $dir/read.log:"; tail -5 "$dir/read.log"; }
done
exit $rc
