#!/usr/bin/env bash
# Self-contained: Gray-Scott + Clio + NeuroPress, everything in this one file.
#
# Same run as run.sh, but with NO external compose.yaml -- the pool chain is
# generated inline below, so this script plus the built binary is the whole
# deployment. Nothing is read from the example directory at run time.
#
# Usage: ./run_standalone.sh [-b BUILD] [-L N] [-s STEPS] [-g GAP]
#                            [-o OUTDIR] [--learn] [--explore K] [--keep]
set -euo pipefail

# ---- defaults --------------------------------------------------------------
BUILD=""; L=128; STEPS=100; GAP=25; OUT=""; KEEP=0
LEARN=false; EXPLORE=false; EXPLORE_K=3; LR=0.01; MAPE=0.30; SELLOG=""; VERIFY=""

while [ $# -gt 0 ]; do
  case "$1" in
    -b|--build)   BUILD=$2; shift 2 ;;
    -L)           L=$2; shift 2 ;;
    -s|--steps)   STEPS=$2; shift 2 ;;
    -g|--gap)     GAP=$2; shift 2 ;;
    -o|--out)     OUT=$2; shift 2 ;;
    --learn)      LEARN=true; shift ;;
    --explore)    LEARN=true; EXPLORE=true; EXPLORE_K=${2:-31}; shift 2 ;;
    --lr)         LEARN=true; LR=$2; shift 2 ;;
    --mape)       MAPE=$2; shift 2 ;;
    --selection-log) SELLOG=$2; shift 2 ;;
    --verify)     VERIFY=verify; shift ;;
    --keep)       KEEP=1; shift ;;
    -h|--help)    sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---- locate the tree and the binary ---------------------------------------
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
BUILD=${BUILD:-$REPO/build}
BIN=$BUILD/bin/neuropress_grayscott_h5
VOL=$BUILD/bin/libclio_hdf5_vol.so
WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights

for f in "$BIN" "$VOL"; do
  [ -e "$f" ] || { echo "missing: $f
build first:  make -C $BUILD neuropress_grayscott_h5 clio_hdf5_vol" >&2; exit 1; }
done
[ -d "$WEIGHTS" ] || { echo "missing NeuroPress weights: $WEIGHTS" >&2; exit 1; }

# Storage sized from the workload: a snapshot is L^3 * 4 bytes, and the tier
# has to hold every snapshot the run produces (compressed, so this is slack).
SNAP_MB=$(( L * L * L * 4 / 1048576 ))
NSNAP=$(( STEPS / GAP )); [ "$NSNAP" -lt 1 ] && NSNAP=1
TIER_MB=$(( SNAP_MB * NSNAP + 512 )); [ "$TIER_MB" -lt 512 ] && TIER_MB=512
BDEV_MB=$(( TIER_MB * 2 ))

WORK=${OUT:-$(mktemp -d /tmp/np_gs.XXXXXX)}
mkdir -p "$WORK"
[ "$KEEP" = 1 ] || trap 'rm -rf "$WORK"' EXIT

# HDF5 will only dlopen a VOL connector out of HDF5_PLUGIN_PATH, so give it a
# directory containing just this one .so.
ln -sf "$VOL" "$WORK/"

# ---- the pool chain, inline ------------------------------------------------
# clio_cte_compressor sits at 512.0, the CTE entrypoint, so the VOL's client
# lands on it unchanged and forwards to the core at 513.0.
#
# neuropress_model_path is the MASTER SWITCH: the compressor only builds the
# predictor when it is non-empty, and every other neuropress_* key is inert
# without it. It wants the weights DIRECTORY, not the .nnwt file.
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
    neuropress_online_learning_enabled: $LEARN
    neuropress_exploration_enabled: $EXPLORE
    neuropress_exploration_k: $EXPLORE_K
    neuropress_learning_rate: $LR
    neuropress_mape_threshold: $MAPE
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

mode="inference only"
$LEARN   && mode="online learning"
$EXPLORE && mode="online learning + exploration (K=$EXPLORE_K)"
$LEARN && mode="$mode, lr=$LR mape=$MAPE"
echo "Gray-Scott ${L}^3, $STEPS steps, snapshot every $GAP  [$mode]"
echo "data: ${SNAP_MB} MiB/snapshot x $NSNAP = $(( SNAP_MB * NSNAP )) MiB, tier ${TIER_MB} MB"
echo "work dir: $WORK"

# CLIO_WITH_RUNTIME=1 hosts the runtime IN THIS PROCESS. Load-bearing: against
# a separate clio_run daemon the DynamicScheduleTask arrives default-
# constructed and every write is rejected as "Invalid chunk data".
# CLIO_VOL_COMPRESSOR_POOL is what makes the VOL route through the compressor
# at all; without it the model loads and is never consulted, silently.
[ -n "$SELLOG" ] && export CLIO_NEUROPRESS_SELECTION_LOG="$SELLOG"
env CLIO_SERVER_CONF="$WORK/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    HDF5_VOL_CONNECTOR=clio \
    HDF5_PLUGIN_PATH="$WORK" \
    CLIO_VOL_COMPRESSOR_POOL=512.0 \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-debug}" \
    "$BIN" "$WORK/gs.h5" "$L" "$STEPS" "$GAP" $VERIFY > "$WORK/run.log" 2>&1 \
  || { echo "run FAILED -- tail of log:"; tail -20 "$WORK/run.log"; exit 1; }

grep -E "Gray-Scott|step |done:|verify |VERIFY" "$WORK/run.log" || true

n=$(grep -c 'Compression: ' "$WORK/run.log" || true)
echo
echo "chunks compressed: $n"
if [ "${n:-0}" -gt 0 ]; then
  grep -oE 'Compression: [0-9]+ bytes -> [0-9]+ bytes \(ratio: [0-9]+\.[0-9]+' "$WORK/run.log" \
    | sed -E 's/Compression: ([0-9]+) bytes -> ([0-9]+) bytes \(ratio: ([0-9.]+)/\1 -> \2  ratio \3/' \
    | sort -k5 -g | awk 'NR==1{w=$0} END{print "  worst : "w"\n  best  : "$0}'
else
  echo "  NO COMPRESSION. Check that CLIO_VOL_COMPRESSOR_POOL matched the"
  echo "  compressor pool_id, and that the weights directory exists."
fi
[ "$KEEP" = 1 ] && echo "kept: $WORK/gs.h5 and $WORK/run.log"
exit 0
