#!/usr/bin/env bash
# Run ONE named configuration of the WarpX paper benchmark -- IN SITU.
#
#   ./run_config.sh <config> [--ncell "NX NY NZ"] [--steps N] [--interval N]
#                            [--chunk BYTES] [--results DIR] [--tag NAME]
#                            [--verify]
#
# --verify runs read.sh afterwards: reads field datasets back through the VOL
# from a separate process and requires BOTH that the bytes match a native read
# AND that the trace shows the tier served them. See read.sh for why the second
# half is not optional here.
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
RESULTS="$HERE/results" TAG="" VERIFY=0
WARPX_BIN=${WARPX_BIN:-$(ls "$HOME"/src/warpx/build-clio/bin/warpx.3d.* 2>/dev/null | head -1)}
DECK=${DECK:-$HOME/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}
# --- Cost-model bandwidth, error bound, exploration policy ----------------
# BW is CLIO_NEUROPRESS_COST_BW in BYTES PER MILLISECOND -- the unit of
# RankingWeights::bandwidth_bytes_per_ms (predictor.h), NOT bytes/s or GB/s.
# 1 GB/s = 1e6 B/ms; the shipped default 5e6 is 5 GB/s. It enters the cost as
# bytes/(ratio*bw), so it only changes a DECISION when some other term is
# non-zero -- under the ratio-only weights it is a positive scalar on the sole
# term and cannot reorder candidates at all. See ../BENCHMARK.md.
#
# EB is CLIO_NEUROPRESS_ERROR_BOUND, an ABSOLUTE bound: |orig - decoded| <= EB.
# 0 (empty) is lossless and masks NeuroPress's 16 quantize actions.
#
# THRESH_OPT=0 makes exploration UNCONDITIONAL. That is deliberate for this
# benchmark and not upstream's 0.5: at 0.5 a chunk explores only when the cost
# prediction was already off by >50%, so (a) the number of explored chunks
# varies with the cost model and bandwidth being compared, and (b) it varies
# run to run on a non-bit-reproducible workload -- measured 16/45 then 33/45
# on the same LAMMPS settings. Both make the 2x2 matrix less comparable.
# At 0 every chunk explores, so every compressed chunk yields a measured
# decompression time and the only variable left is the one under test.
# K stays at 3, NeuroPress's own ranked window.
# GPU-ONLY mode, in the only shape this workload can have it.
#
# A stock WarpX hands HDF5 HOST memory (measured -- see clio_stage_append's
# report), and openPMD emits non-contiguous partial writes the VOL assembles
# into a host run buffer, so residency is gone before Clio is called. Unlike
# Nyx/VPIC/LAMMPS there is no device pointer to hand over.
#
# So this takes UPSTREAM's route for a host-resident caller: stage the chunk
# H2D and run the same CUDA kernels on it. That is what
# gpucompress_compress() does -- "Transfers data to GPU, compresses, and
# returns result to host" -- and it keeps the guarantee that every transform
# runs in CUDA, which is the property being enforced. It costs a full H2D per
# chunk, real bandwidth the in-situ workloads do not spend, so WarpX timings
# are NOT comparable with theirs; the ratios are.
#
# REQUIRE_DEVICE stays on beneath it, so if staging ever fails to produce a
# device pointer the run refuses rather than silently reverting.
REQUIRE_DEVICE=0 STAGE_H2D=0
BW="" EB="" EXPLORE_K_OPT=3 THRESH_OPT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --interval) INTERVAL=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --bw) BW=$2; shift 2;;
    --eb) EB=$2; shift 2;;
    --explore-k) EXPLORE_K_OPT=$2; shift 2;;
    --explore-thresh) THRESH_OPT=$2; shift 2;;
    --require-device) REQUIRE_DEVICE=1; shift;;
    --stage-h2d) STAGE_H2D=1; REQUIRE_DEVICE=1; shift;;
    --bin) WARPX_BIN=$2; shift 2;;
    --deck) DECK=$2; shift 2;;
    --verify) VERIFY=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
export WARPX_BIN
[ -f "$DECK" ] || { echo "missing WarpX input deck: $DECK" >&2; exit 1; }

NP_LEARN=false NP_EXPLORE=false EXPLORE_K=0 THRESH=0.5 BEST=false
STATIC_LIB="" STATIC_SHUF=0
COST_ENV=()
# Which of the two cost models this config asks for, recorded in meta.json so
# a run is self-describing. The pre-existing configs keep their historical
# behaviour: `dynamic`/`learn` rank under the default balanced weights, the
# rest under the ratio-only ones.
case "$CONFIG" in
  dynamic|learn|explore-balance) COSTMODEL=balance ;;
  static-*)                      COSTMODEL=none ;;
  *)                             COSTMODEL=ratio ;;
esac
RATIO_ONLY=(CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1)
case "$CONFIG" in
  dynamic)        ;;
  dynamic-ratio)  COST_ENV=("${RATIO_ONLY[@]}") ;;
  learn)          NP_LEARN=true ;;
  explore)        NP_LEARN=true; NP_EXPLORE=true; EXPLORE_K=31; THRESH=0
                  COST_ENV=("${RATIO_ONLY[@]}") ;;
  best)           BEST=true ;;
  # EXPLORATION MODE, the two cost models. NP_LEARN=true is NOT optional and
  # not an extra experimental axis: the per-chunk features that gate the
  # exploration block are computed only when online learning or best mode is
  # on (compressor_runtime.cc:1145-1152), so exploration with it off silently
  # returns a plain inference result. `explore` above has always set it for
  # the same reason.
  #
  # K and the threshold come from --explore-k / --explore-thresh. K defaults
  # to 3, NeuroPress's own ranked window, rather than the exhaustive 31 that
  # `explore` above pins for action-space studies; the threshold defaults to
  # 0 for the comparability reason given where it is declared.
  explore-balance) NP_LEARN=true; NP_EXPLORE=true
                   EXPLORE_K=$EXPLORE_K_OPT; THRESH=$THRESH_OPT ;;
  explore-ratio)   NP_LEARN=true; NP_EXPLORE=true
                   EXPLORE_K=$EXPLORE_K_OPT; THRESH=$THRESH_OPT
                   COST_ENV=("${RATIO_ONLY[@]}") ;;
  static-zstd)    STATIC_LIB=nvcomp-zstd; STATIC_SHUF=0 ;;
  static-zstd-s4) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=4 ;;
  static-zstd-s8) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=8 ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2;;
esac
export NP_LEARN NP_EXPLORE EXPLORE_K THRESH BEST STATIC_LIB STATIC_SHUF

# A positive error bound means the decompressed bytes are NOT the bytes that
# went in, by design. Every verify path here is an FNV-1a digest comparison, so
# under lossy compression it would report FAILED on a run that is behaving
# exactly as asked. Turn it off and say so; the quality number for a lossy run
# is PSNR in selection.csv, not a digest.
MODE=lossless
if [ -n "$EB" ] && awk -v e="$EB" 'BEGIN{exit !(e+0>0)}'; then
  MODE=lossy
  if [ "${VERIFY:-0}" = 1 ]; then
    echo "   (lossy eb=$EB: bit-exact verification disabled -- see run_config.sh)"
    VERIFY=0
  fi
fi

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
    ${EB:+CLIO_NEUROPRESS_ERROR_BOUND=$EB} \
    ${BW:+CLIO_NEUROPRESS_COST_BW=$BW} \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=${MEASURE_DT:-1} \
    ${REQUIRE_DEVICE:+CLIO_NEUROPRESS_REQUIRE_DEVICE=$REQUIRE_DEVICE} \
    ${STAGE_H2D:+CLIO_NEUROPRESS_STAGE_H2D=$STAGE_H2D} \
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

# A refusal means a chunk reached the compressor in host memory. For this
# workload that is expected rather than exceptional -- see the note by
# REQUIRE_DEVICE above -- but it must still fail the run, because the numbers
# would otherwise be CPU-preprocessed numbers published as GPU ones.
HOSTED=0
[ -f "$STORE/runtime.log" ] && HOSTED=$(grep -c "REQUIRE_DEVICE is set" "$STORE/runtime.log" || true)
HOSTED=${HOSTED:-0}
if [ "$HOSTED" -gt 0 ]; then
  echo "   HOST-RESIDENT CHUNKS REFUSED: $HOSTED -- this run did NOT stay on the GPU"
  echo "   (a stock WarpX hands HDF5 host memory; pass --stage-h2d to copy each"
  echo "    chunk up and run the CUDA kernels on it -- see BENCHMARK.md)"
  RC=1
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

VERIFY_RESULT="n/a"
if [ "$VERIFY" = 1 ] && [ $RC -eq 0 ]; then
  if "$HERE/read.sh" --run "$NAME" --results "$RESULTS" > "$STORE/verify.log" 2>&1; then
    VERIFY_RESULT="pass"
  else
    VERIFY_RESULT="FAIL"
  fi
  sed 's/^/   /' "$STORE/verify.log" | tail -6
fi

cat > "$STORE/meta.json" <<JSON
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"mode":"$MODE",
 "cost_model":"$COSTMODEL","bw_bytes_per_ms":${BW:-5e6},
 "error_bound":${EB:-0},"explore_k":$EXPLORE_K,"explore_thresh":$THRESH,
 "residency":"$([ "$STAGE_H2D" = 1 ] && echo device-staged-h2d || { [ "$REQUIRE_DEVICE" = 1 ] && echo device-required || echo host; })",
 "host_refusals":$HOSTED,"device":"gpu","workload":"warpx-laser",
 "atoms":0,"steps":$STEPS,"gap":$INTERVAL,
 "frames":$(( STEPS / INTERVAL + 1 )),"chunk":$CHUNK,"files":$NCHUNKS,
 "payload_bytes":$PAYLOAD,"native_h5_bytes":$NATIVE,"wall_s":$WALL,
 "verified":$VERIFY,"verify_result":"$VERIFY_RESULT","port":$PORT,
 "physics":{"problem":"laser_acceleration","precision":"float32","insitu":"yes"}}
JSON
cat "$STORE/convert.log" | sed 's/^/   /'
exit $RC
