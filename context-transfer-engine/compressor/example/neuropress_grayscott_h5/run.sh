#!/usr/bin/env bash
# Run Gray-Scott with Clio + NeuroPress INSIDE the application process.
#
# No daemon, no jarvis, no containers. The binary links no Clio library
# (ldd | grep clio == 0); Clio arrives through the environment below.
#
#   CLIO_WITH_RUNTIME=1         host the Clio runtime IN THIS PROCESS.
#                               Load-bearing: against a separate clio_run
#                               daemon the DynamicScheduleTask arrives
#                               default-constructed and every write is
#                               rejected with "Invalid chunk data".
#   HDF5_VOL_CONNECTOR=clio     HDF5 dlopens the Clio VOL connector
#   HDF5_PLUGIN_PATH            where it finds libclio_hdf5_vol.so
#   CLIO_VOL_COMPRESSOR_POOL    the connector builds a compressor client and
#                               calls DynamicSchedule; without it, no
#                               compression happens at all
#
# Usage: ./run.sh [BUILD_DIR] [L] [STEPS] [SNAPSHOT_EVERY]
set -euo pipefail

BUILD=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../build" && pwd)}
L=${2:-128}; STEPS=${3:-100}; GAP=${4:-25}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
BIN=$BUILD/bin/neuropress_grayscott_h5

[ -x "$BIN" ] || { echo "build it first:  make -C $BUILD neuropress_grayscott_h5"; exit 1; }

WORK=$(mktemp -d /tmp/np_gs_h5.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
ln -sf "$BUILD/bin/libclio_hdf5_vol.so" "$WORK/"
sed "s|CHANGE_ME|$REPO|" "$HERE/compose.yaml" > "$WORK/compose.yaml"

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "Gray-Scott ${L}^3, $STEPS steps, snapshot every $GAP -> $WORK/gs.h5"
env CLIO_SERVER_CONF="$WORK/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH="$WORK" \
    CLIO_VOL_COMPRESSOR_POOL=512.0 \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-debug}" \
    "$BIN" "$WORK/gs.h5" "$L" "$STEPS" "$GAP" 2>&1 | tee "$WORK/run.log" \
  | grep -E "Gray-Scott|step |done:" || true

echo
echo "chunks compressed: $(grep -c 'Compression: ' "$WORK/run.log" || echo 0)"
grep -oE 'Compression: [0-9]+ bytes -> [0-9]+ bytes \(ratio: [0-9.]+' "$WORK/run.log" \
  | sed 's/Compression: //; s/ bytes//g; s/(ratio: /ratio /' | sort -u -k4 -g \
  | awk 'NR<=3 || NR>n-3' n=$(grep -c 'Compression: ' "$WORK/run.log") | head -6
