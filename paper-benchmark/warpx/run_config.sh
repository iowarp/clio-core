#!/usr/bin/env bash
# Run ONE named configuration of the WarpX paper benchmark -- IN SITU.
#
#   ./run_config.sh <config> [--ncell "NX NY NZ"] [--steps N] [--interval N]
#                            [--chunk BYTES] [--results DIR] [--tag NAME]
#
# Unlike the other three workloads, nothing is dumped and replayed: a STOCK,
# UNPATCHED WarpX runs, writes openPMD-HDF5 as it always does, and HDF5 loads
# Clio's VOL connector because HDF5_VOL_CONNECTOR says so. Clio's runtime is
# hosted in the WarpX process. Each policy is therefore its own WarpX run.
#
# Policies are the same set the other workloads use, so the numbers compare.
#
# RATIO_CAP=<x> overrides NeuroPress's ratio ceiling (default 100, upstream's
# nn_gpu.cu constant). It is worth knowing what it does to SELECTION, not just
# to ranking: the cost the exploration gate compares is
# bytes/(min(ratio,cap)*bw), so when a chunk's predicted AND actual ratio both
# exceed the cap the two costs are numerically identical, the error is exactly
# zero, and exploration cannot fire at any threshold -- including 0.
#
# CLIO_VOL_CHUNK_SIZE: 1 MiB here, NOT the 4 MiB the other workloads use, and
# this is not a tuning preference -- it is a correctness condition for this
# writer. openPMD emits each AMReX box as a separate partial write, so an
# 8 MiB field arrives as 1 MiB pieces at NON-CONTIGUOUS offsets ([0,1M) then
# [4M,5M) ...). The VOL's append path assembles only contiguous runs and drops
# the tail on a discontinuity, so with a 4 MiB chunk NO chunk ever completes and
# ZERO field bytes reach the tier -- while the run succeeds, the native .h5 is
# perfect, and nothing reports a problem. Measured: 0 field blobs staged at
# 4 MiB, 400 at 1 MiB. The chunk must be <= the writer's contiguous granularity.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "${1:-}" in -h|--help) sed -n '2,24p' "$0"; exit 0;; esac
CONFIG=${1:-dynamic}; shift || true

NCELL=${NCELL:-"64 64 512"}
STEPS=${STEPS:-40} INTERVAL=${INTERVAL:-10}
CHUNK=${CHUNK:-1048576}
RESULTS="$HERE/results" TAG=""
WARPX_BIN=${WARPX_BIN:-$(ls "$HOME"/src/warpx/build-clio/bin/warpx.3d.* 2>/dev/null | head -1)}
DECK=${DECK:-$HOME/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --interval) INTERVAL=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --bin) WARPX_BIN=$2; shift 2;;
    --deck) DECK=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
export WARPX_BIN
[ -f "$DECK" ] || { echo "missing WarpX input deck: $DECK" >&2; exit 1; }

NP_LEARN=false NP_EXPLORE=false EXPLORE_K=0 THRESH=0.5 BEST=false
STATIC_LIB="" STATIC_SHUF=0
COST_ENV=()
RATIO_ONLY=(CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1)
case "$CONFIG" in
  dynamic)        ;;
  dynamic-ratio)  COST_ENV=("${RATIO_ONLY[@]}") ;;
  learn)          NP_LEARN=true ;;
  explore)        NP_LEARN=true; NP_EXPLORE=true; EXPLORE_K=31; THRESH=0
                  COST_ENV=("${RATIO_ONLY[@]}") ;;
  best)           BEST=true ;;
  static-zstd)    STATIC_LIB=nvcomp-zstd; STATIC_SHUF=0 ;;
  static-zstd-s4) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=4 ;;
  static-zstd-s8) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=8 ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2;;
esac
export NP_LEARN NP_EXPLORE EXPLORE_K THRESH BEST STATIC_LIB STATIC_SHUF

# shellcheck source=common.sh
. "$HERE/common.sh"
bench_setup || exit 1

NAME=${TAG:-$CONFIG}
STORE=$RESULTS/$NAME
rm -rf "$STORE"; mkdir -p "$STORE"
bench_compose "$STORE"
# HDF5 finds the connector by name under HDF5_PLUGIN_PATH.
ln -sf "$VOL_SO" "$STORE/"

echo "== $NAME: WarpX ${NCELL// /x}, $STEPS steps, diag every $INTERVAL, chunk $CHUNK, port $PORT"

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
RUNDIR="$STORE/run"; mkdir -p "$RUNDIR"
START=$(date +%s.%N)
set +e
( cd "$RUNDIR" && env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH="$STORE" \
    CLIO_VOL_COMPRESSOR_POOL=512.0 \
    CLIO_VOL_CHUNK_SIZE="$CHUNK" \
    CLIO_VOL_STAMP_GRANULARITY_NS=0 \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    $( { [ "$NP_EXPLORE" = true ] || [ "$BEST" = true ]; } && echo CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv" ) \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warn}" \
    ${RATIO_CAP:+CLIO_NEUROPRESS_RATIO_CAP=$RATIO_CAP} \
    "${COST_ENV[@]}" \
    "$WARPX_BIN" "$DECK" \
        max_step="$STEPS" diag1.intervals="$INTERVAL" \
        amr.n_cell="$NCELL" diag1.openpmd_backend=h5 ) \
  > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?
set -e
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')

if [ $RC -ne 0 ]; then
  echo "   FAILED rc=$RC"; grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -8
fi

# Per-chunk record comes from the VOL's path trace: the application is stock
# WarpX, so there is no Clio driver here to write one.
# --explore-log is not optional whenever selection can override the primary:
# without it the stored sizes silently revert to the primary candidate's.
CONV=("$HERE/trace_to_csv.py" "$STORE/runtime.log" "$STORE/blobs.csv" --fields-only)
[ -f "$STORE/explore.csv" ] && CONV+=(--explore-log "$STORE/explore.csv")
"${CONV[@]}" > "$STORE/convert.log" 2>&1 || true
NCHUNKS=$(( $(wc -l < "$STORE/blobs.csv") - 1 ))
PAYLOAD=$(awk -F, 'NR>1{s+=$2} END{printf "%d", s+0}' "$STORE/blobs.csv")
NATIVE=$(du -sb "$RUNDIR/diags" 2>/dev/null | cut -f1 || echo 0)

if [ "$NCHUNKS" -le 0 ]; then
  echo "   WARNING: no field chunks staged. With this writer that usually means"
  echo "   the chunk size exceeds openPMD's contiguous write granularity -- see"
  echo "   the note at the top of this script. Try --chunk 1048576."
fi

cat > "$STORE/meta.json" <<JSON
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"device":"gpu","workload":"warpx-laser",
 "atoms":0,"steps":$STEPS,"gap":$INTERVAL,
 "frames":$(( STEPS / INTERVAL + 1 )),"chunk":$CHUNK,"files":$NCHUNKS,
 "payload_bytes":$PAYLOAD,"native_h5_bytes":$NATIVE,"wall_s":$WALL,
 "verified":0,"port":$PORT,
 "physics":{"problem":"laser_acceleration","precision":"float32","insitu":"yes"}}
JSON
cat "$STORE/convert.log" | sed 's/^/   /'
exit $RC
