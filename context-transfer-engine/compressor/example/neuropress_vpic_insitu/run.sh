#!/usr/bin/env bash
# VPIC in situ: the deck hands its field arrays to Clio, no file in between.
#
#   ./run.sh [--ncell N] [--steps N] [--int N] [--nppc N] [--chunk BYTES]
#            [--store DIR] [--port N] [--verify] [--restart]
#            [--learn] [--explore K] [--threshold X] [--best]
#            [--static LIB] [--static-shuffle N] [--bin PATH]
#
# The deck is paper-benchmark/vpic/weibel_clio.cxx built with --insitu, so it
# links libclio_vpic_insitu.so and calls it from begin_diagnostics. Clio's
# runtime is hosted in the VPIC process itself (CLIO_WITH_RUNTIME=1): the
# compressor's tasks have no cross-process wire format, so a runtime elsewhere
# would receive every DynamicSchedule empty. The adapter refuses rather than
# let that look like success.
#
# Same three pools and the same NeuroPress knobs as every sibling example, so
# numbers are comparable with the offline sweep in paper-benchmark/vpic.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
DECK_DIR=${DECK_DIR:-$REPO/paper-benchmark/vpic}
BIN=${BIN:-$DECK_DIR/weibel_clio.Linux}
WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights

NCELL=126 STEPS=200 INT=25 NPPC=8 CHUNK=4194304
PORT=${PORT:-9413}
STORE=${STORE:-$HERE/store}
VERIFY=false RESTART=false
LEARN=false EXPLORE_K=0 THRESH=0.5 BEST=false STATIC_LIB= STATIC_SHUF=0
usage() { sed -n '2,17p' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int) INT=$2; shift 2;;
    --nppc) NPPC=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --store) STORE=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --verify) VERIFY=true; shift;;
    --restart) RESTART=true; shift;;
    --learn) LEARN=true; shift;;
    --explore) EXPLORE_K=$2; shift 2;;
    --threshold) THRESH=$2; shift 2;;
    --best) BEST=true; shift;;
    --static) STATIC_LIB=$2; shift 2;;
    --static-shuffle) STATIC_SHUF=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -x "$BIN" ] || { echo "missing $BIN -- build it with:
  cmake --build $BUILD --target clio_vpic_insitu
  cd $DECK_DIR && ./build_deck.sh --insitu" >&2; exit 1; }
ldd "$BIN" | grep -q libclio_vpic_insitu || {
  echo "$BIN is not linked against libclio_vpic_insitu.so -- rebuild the deck
with ./build_deck.sh --insitu (without it the deck writes files instead)." >&2
  exit 1; }
[ -d "$WEIGHTS" ] || { echo "missing weights: $WEIGHTS" >&2; exit 1; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is busy -- every Clio runtime needs its own (--port)" >&2
  exit 1
fi

if [ "$RESTART" != true ]; then rm -rf "$STORE"; fi
mkdir -p "$STORE"

# Sizing: VPIC's field array carries a ghost layer, so the extent is (N+2)^3
# voxels; 16 float32 variables per frame.
VOX=$(( (NCELL+2) * (NCELL+2) * (NCELL+2) ))
FRAME_MB=$(( VOX * 4 * 16 / 1048576 )); [ "$FRAME_MB" -lt 1 ] && FRAME_MB=1
NFRAMES=$(( STEPS / INT ))
TIER_MB=$(( FRAME_MB * NFRAMES + 512 ))
BDEV_MB=$(( TIER_MB * 2 ))

if [ "$LEARN" = true ] || [ "$EXPLORE_K" -gt 0 ]; then NP_LEARN=true; else NP_LEARN=false; fi
if [ "$EXPLORE_K" -gt 0 ]; then NP_EXPLORE=true; else NP_EXPLORE=false; fi
if [ -n "$STATIC_LIB" ]; then NP_LEARN=false; NP_EXPLORE=false; fi

if [ "$RESTART" != true ]; then
cat > "$STORE/compose.yaml" <<YAML
networking:
  port: $PORT
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "$STORE/chi_bdev.dat"
    pool_query: local
    pool_id: "301.0"
    bdev_type: file
    path: "$STORE/chi_bdev.dat"
    capacity: "${BDEV_MB}MB"
  - mod_name: clio_cte_compressor
    pool_name: cte_compressor
    pool_query: local
    pool_id: "512.0"
    next_pool_id: "513.0"
    neuropress_model_path: "$WEIGHTS"
    neuropress_online_learning_enabled: $NP_LEARN
    neuropress_exploration_enabled: $NP_EXPLORE
    neuropress_exploration_k: $EXPLORE_K
    neuropress_exploration_threshold: $THRESH
    neuropress_best_mode: $BEST
${STATIC_LIB:+    neuropress_static_lib: "$STATIC_LIB"}
${STATIC_LIB:+    neuropress_static_shuffle: $STATIC_SHUF}
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "513.0"
    storage:
      - path: "$STORE/cte_tier.dat"
        bdev_type: "file"
        capacity_limit: "${TIER_MB}MB"
        score: 1.0
        persistence_level: "temporary"
    performance:
      metadata_log_path: "$STORE/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
YAML
fi

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "== VPIC in situ + Clio in one process: ${NCELL}^3 cells ($VOX voxels with ghosts)"
echo "   steps=$STEPS every $INT -> $NFRAMES frame(s) x 16 vars, ~${FRAME_MB} MiB/frame, chunk=$CHUNK"
if [ -n "$STATIC_LIB" ]; then echo "   STATIC codec=$STATIC_LIB shuffle=$STATIC_SHUF (NeuroPress bypassed)"
elif [ "$BEST" = true ]; then echo "   BEST mode (exhaustive, ratio-only)"
else echo "   learn=$NP_LEARN explore=$NP_EXPLORE k=$EXPLORE_K"; fi
echo "   store=$STORE  port=$PORT"

set +e
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_VPIC_TAG=vpic_insitu \
    CLIO_VPIC_POOL=512.0 \
    CLIO_VPIC_CHUNK="$CHUNK" \
    CLIO_VPIC_REPORT="$STORE/blobs.csv" \
    CLIO_VPIC_VERIFY=$([ "$VERIFY" = true ] && echo 1 || echo 0) \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    VPIC_NX="$NCELL" VPIC_NY="$NCELL" VPIC_NZ="$NCELL" VPIC_NPPC="$NPPC" \
    VPIC_STEPS="$STEPS" VPIC_INSITU=1 VPIC_INSITU_INT="$INT" \
    "$BIN" > "$STORE/vpic.log" 2> "$STORE/runtime.log"
RC=$?
set -e

grep -E "^\[clio-(vpic-)?insitu\]|VERIFIED|FAILED|REFUSING" "$STORE/vpic.log" "$STORE/runtime.log" 2>/dev/null | sed 's/^[^:]*://' || true
if [ $RC -ne 0 ]; then
  echo "-- VPIC failed (rc=$RC); last lines of $STORE/runtime.log:" >&2
  grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -8 >&2
fi
echo "-- blobs: $STORE/blobs.csv   vpic: $STORE/vpic.log   runtime: $STORE/runtime.log   exit=$RC"
exit $RC
