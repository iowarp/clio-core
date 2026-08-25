#!/usr/bin/env bash
# Phase 2 of 2: a SEPARATE process reads back what the Nyx run stored.
#
#   ./run.sh --static nvcomp-zstd --ncell 64 --steps 20 --int 10
#   ./read.sh
#
# No Nyx here, and no simulation dependency of any kind: the reader is
# bin/neuropress_field_replay --readback, which knows nothing about AMReX.
# That works because the CSV libclio_nyx_insitu.so writes is exactly the one
# that program writes -- blob,bytes,fnv1a64,... -- so the two halves of the
# check share only the store directory and a list of names, sizes and digests.
#
# CLIO_RESTART=1 replays the metadata log so a fresh process can find blobs a
# previous one wrote. There is no plotfile and no .f32 dump to fall through
# to: the compressed tier is the only copy of the data, so bytes that come
# back correct came from it.
#
# Under MPI the writer was N ranks, each with its own runtime, its own store
# and its own CSV ($STORE/rank0000, rank0001, ...). This reads them ONE AT A
# TIME, in one process per store, and sums the result: N tiers, N metadata
# logs, N cold reads. A rank's store is self-contained, so nothing here needs
# MPI and the reader is still the same Nyx-free binary.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
BIN=$BUILD/bin/neuropress_field_replay
STORE=${STORE:-$HERE/store}
DUMP=
while [ $# -gt 0 ]; do
  case "$1" in
    --store) STORE=$2; shift 2;;
    --dump-decompressed) DUMP=$2; shift 2;;   # for an external byte compare
    -h|--help) sed -n '2,17p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -x "$BIN" ] || { echo "missing $BIN -- cmake --build $BUILD --target neuropress_field_replay" >&2; exit 1; }
export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# One store, or one per rank.
STORES=()
if [ -f "$STORE/compose.yaml" ]; then STORES=("$STORE"); else
  for d in "$STORE"/rank[0-9][0-9][0-9][0-9]; do
    [ -f "$d/compose.yaml" ] && STORES+=("$d")
  done
fi
[ ${#STORES[@]} -gt 0 ] || { echo "no compose.yaml under $STORE -- run ./run.sh first" >&2; exit 1; }

TOT_RC=0 TOT_INV=0
for SD in "${STORES[@]}"; do
  CSV=$(ls "$SD"/blobs*.csv 2>/dev/null | head -1)
  [ -n "$CSV" ] || { echo "no blobs*.csv in $SD" >&2; TOT_RC=1; continue; }
  ARGS=(--readback "$CSV" --tag "${CLIO_NYX_TAG:-nyx_insitu}")
  [ -n "$DUMP" ] && { mkdir -p "$DUMP"; ARGS+=(--dump-decompressed "$DUMP"); }
  echo "== cold read from $SD (the Nyx process is long gone)"
  env CLIO_SERVER_CONF="$SD/compose.yaml" \
      CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
      CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
      "$BIN" "${ARGS[@]}" 2> "$SD/read.log"
  RC=$?
  INV=$(grep -c "stored_compressed=1 -> inverting codec" "$SD/read.log" 2>/dev/null || true)
  TOT_INV=$((TOT_INV + INV))
  echo "-- blobs the compressor inverted a codec for: $INV"
  [ $RC -ne 0 ] && { echo "-- last lines of $SD/read.log:"; tail -20 "$SD/read.log"; TOT_RC=1; }
done
[ ${#STORES[@]} -gt 1 ] && echo "-- $((${#STORES[@]})) store(s) read, $TOT_INV codec inversion(s) in total"
echo "-- exit=$TOT_RC (0 = every blob decompressed to its original bytes)"
exit $TOT_RC
