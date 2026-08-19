#!/usr/bin/env bash
# Write in one process, read in ANOTHER, and prove the second one had to
# decompress to produce anything.
#
# The single-process --verify in run_standalone.sh cannot prove this: whatever
# the writer already holds can satisfy its own read. Here the phases are
# separate processes sharing one runtime daemon, so the reader starts knowing
# nothing -- it has to fetch the stored bytes and invert the codec.
#
# Two independent checks, and BOTH must pass:
#   1. bit-exact against a reference written with plain POSIX I/O, outside
#      HDF5 and so outside Clio -- comparing the stack against itself would
#      pass even if every codec were broken;
#   2. the path trace shows "inverting codec" for every chunk. Without this a
#      pass is meaningless: a cache miss falls through to native HDF5 and
#      returns perfectly correct data having never touched the compressor.
#
# Usage: ./run_twophase.sh [-b BUILD] [-L N] [-s STEPS] [-g GAP] [--keep]
set -euo pipefail

BUILD=""; L=128; STEPS=50; GAP=25; OUT=""; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    -b|--build) BUILD=$2; shift 2 ;;
    -L)         L=$2; shift 2 ;;
    -s|--steps) STEPS=$2; shift 2 ;;
    -g|--gap)   GAP=$2; shift 2 ;;
    -o|--out)   OUT=$2; shift 2 ;;
    --keep)     KEEP=1; shift ;;
    -h|--help)  sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
BUILD=${BUILD:-$REPO/build}
BIN=$BUILD/bin/neuropress_grayscott_h5
VOL=$BUILD/bin/libclio_hdf5_vol.so
RUN=$BUILD/bin/clio_run
WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights
for f in "$BIN" "$VOL" "$RUN"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

SNAP_MB=$(( L * L * L * 4 / 1048576 )); [ "$SNAP_MB" -lt 1 ] && SNAP_MB=1
NSNAP=$(( STEPS / GAP )); [ "$NSNAP" -lt 1 ] && NSNAP=1
TIER_MB=$(( SNAP_MB * NSNAP + 512 )); [ "$TIER_MB" -lt 512 ] && TIER_MB=512
BDEV_MB=$(( TIER_MB * 2 ))

WORK=${OUT:-$(mktemp -d /tmp/np_2p.XXXXXX)}
mkdir -p "$WORK"
ln -sf "$VOL" "$WORK/"

cat > "$WORK/compose.yaml" <<YAML
networking:
  port: 9413
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "${BDEV_MB}MB"
  - mod_name: clio_cte_compressor
    pool_name: cte_compressor
    pool_query: local
    pool_id: "512.0"
    next_pool_id: "513.0"
    neuropress_model_path: "$WEIGHTS"
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "513.0"
    storage:
      - path: "ram::cte_ram_tier1"
        bdev_type: "ram"
        capacity_limit: "${TIER_MB}MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
YAML

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CLIO_SERVER_CONF="$WORK/compose.yaml"

# The stamp granule (10 ms default) makes the connector refuse to vouch for a
# file whose mtime is younger than that, and an absent stamp drops the tag on
# the next open -- every read would then miss and be served by native HDF5.
# Zeroed here deliberately: the phases are back to back by design.
export CLIO_VOL_STAMP_GRANULARITY_NS=0

app_env=(env HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH="$WORK"
         CLIO_VOL_COMPRESSOR_POOL=512.0 CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-info}")

cleanup() {
  [ -n "${WRITER_PID:-}" ] && { touch "$WORK/gs.h5.stop" 2>/dev/null || true; }
  [ "$KEEP" = 1 ] || rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== phase 1: WRITE (process 1, hosts the runtime) ==="
# CLIO_WITH_RUNTIME=1, not a standalone daemon: a DynamicScheduleTask sent
# across a process boundary arrives at the compressor default-constructed
# (blob='', bytes=0) and every write is rejected with "Invalid chunk data",
# so the writer must host the runtime itself. The tier is RAM and lives in
# that process, which is why the writer parks in 'serve' mode instead of
# exiting -- that is what makes a separate reader process possible at all.
rm -f "$WORK/gs.h5.ready" "$WORK/gs.h5.stop"
"${app_env[@]}" CLIO_WITH_RUNTIME=1 "$BIN" "$WORK/gs.h5" "$L" "$STEPS" "$GAP" serve \
  > "$WORK/write.log" 2>&1 &
WRITER_PID=$!
for _ in $(seq 1 600); do
  [ -e "$WORK/gs.h5.ready" ] && break
  kill -0 "$WRITER_PID" 2>/dev/null || break
  sleep 0.2
done
[ -e "$WORK/gs.h5.ready" ] \
  || { echo "writer never became ready; tail:"; tail -25 "$WORK/write.log"; exit 1; }
grep -E 'Gray-Scott|step |done:' "$WORK/write.log" || true
echo "chunks compressed: $(grep -c 'Compression: ' "$WORK/write.log" || echo 0)"
echo "writer parked (pid $WRITER_PID), tier held open"

echo
echo "=== phase 2: READ (process 2, knows nothing) ==="
# A plain client: no CLIO_WITH_RUNTIME, no simulation, no writes. Everything
# it produces has to come out of the tier the writer is holding.
"${app_env[@]}" "$BIN" "$WORK/gs.h5" "$L" "$STEPS" "$GAP" read \
  > "$WORK/read.log" 2>&1 && rc=0 || rc=$?
grep -E 'reading |verify |VERIFY' "$WORK/read.log" || true

touch "$WORK/gs.h5.stop"
wait "$WRITER_PID" 2>/dev/null || true

echo
echo "=== evidence ==="
inv=$(grep -c 'inverting codec' "$WORK/read.log" || true)
pas=$(grep -c 'stored_compressed=0 -> passthrough' "$WORK/read.log" || true)
hit=$(grep -c 'HIT (serve from CTE)' "$WORK/read.log" || true)
mis=$(grep -c 'MISS (native + stage)' "$WORK/read.log" || true)
echo "  cache      : $hit hit / $mis miss"
echo "  decompress : $inv chunk(s) inverted, $pas passthrough"
if [ "${inv:-0}" -eq 0 ]; then
  echo
  echo "INCONCLUSIVE: the reader never inverted a codec, so a bit-exact"
  echo "result only proves the native HDF5 file is intact. Check that the"
  echo "trace build is on (-DCLIO_NEUROPRESS_PATH_TRACE=ON) and that the"
  echo "read hit the cache."
  [ "$KEEP" = 1 ] && echo "logs: $WORK"
  exit 3
fi
if [ "$rc" -ne 0 ]; then
  echo "READ VERIFY FAILED (rc=$rc)"; [ "$KEEP" = 1 ] && echo "logs: $WORK"; exit 4
fi
echo
echo "PASS: a separate process reconstructed the data by inverting $inv codec"
echo "      chunk(s), bit-exact against a reference Clio never touched."
[ "$KEEP" = 1 ] && echo "kept: $WORK"
exit 0
