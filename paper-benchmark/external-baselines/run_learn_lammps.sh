#!/usr/bin/env bash
# NeuroPress LEARNING phase on LAMMPS, ratio cost model, 10% MAPE gate.
#
# LAMMPS is IN-SITU: there is no dump to replay. neuropress_lammps_lib runs the
# LJ melt in-process and hands Atom::x, v and f to Clio between GAP-step
# segments, so the simulation and the learning happen in the same job.
#
# Sizing to 2016 chunks, matching Nyx and VPIC at 8 MiB:
#   box 100      -> 4,000,000 atoms
#   3 fields     -> x, v, f, each natoms*3 float64 = 96,000,000 B
#   chunk 8 MiB  -> 12 chunks per field, 36 per frame
#   steps 1120 / gap 20 -> 56 frames -> 2016 chunks, 16.1 GB
#
# --f32 IS NOT OPTIONAL, and the first attempt without it is why.
#
# NeuroPress computes its three statistics as float32 UNCONDITIONALLY -- that
# is upstream-faithful, the model was trained that way, and the runtime says so
# at compressor_runtime.cc:899 ("NeuroPress's unconditional float32"). Handed
# LAMMPS' float64 state it therefore REINTERPRETS the bytes rather than
# converting them. Measured: data_range came back 2.05e38 .. 6.81e38, which is
# the float32 magnitude limit, not anything in an LJ melt.
#
# The quantizer then needs range/(2*eb) indices, ~1e41 of them, refuses with
# "bound needs a finer grid than int32 can index", and stores the chunk
# losslessly. The run completes, reports rc=0, and is silently a LOSSLESS run:
# 1554 of 2052 chunks stored raw, ratio 1.022, quantized=0 and max_error=0 on
# every chunk in the quality sidecar. The refusal is logged at kDebug, so at
# the default log level nothing says this happened.
#
# --f32 makes the driver downcast x/v/f to real float32 before staging and
# declare data_type_=1, so the statistics describe the actual state and the
# quantizer sees a sane range. It halves the payload, hence 2200 steps here
# against the 1120 the float64 attempt used, for the same ~2000 chunks.
#
# The float64 point still stands for the SHUFFLE: NeuroPress encodes it as one
# bit meaning 4 bytes, so the stride matching the original double data is not
# in the action space. --f32 sidesteps that by making 4 the right width.
#
#   run_learn_lammps.sh <box> <steps> <gap> <chunk> <eb> <mape> <store>
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}; PB=$REPO/paper-benchmark
export BUILD=${BUILD:-$REPO/build}
BOX=$1; STEPS=$2; GAP=$3; CHUNK=$4; EB=$5; MAPE=$6; STORE=$7
export BOX STEPS GAP

# NP_LEARN=false from the environment gives the frozen-weights control: same
# inference path, same chunks, no SGD at all. It is what "learning off" means.
# Overridable, previous hardcoded values as defaults, so the published matrix
# runs behave exactly as before.
export NP_LEARN=${NP_LEARN:-true} NP_EXPLORE=${NP_EXPLORE:-false} \
       EXPLORE_K=${EXPLORE_K:-0} THRESH=${THRESH:-0.5} BEST=${BEST:-false}
export STATIC_LIB="${STATIC_LIB:-}" STATIC_SHUF=${STATIC_SHUF:-0}   # env-overridable: Phase 3 pins a fixed codec here

# Cost weights. Default is RATIO-ONLY, which is what makes the MAPE gate a
# ratio gate: under the balanced weights both time terms sit at their 1 ms
# floor and contribute a fixed 2.0 ms while the ratio term contributes ~0.04,
# so a ratio wrong by 5x can pass a 10% gate untouched.
W_CT=${W_CT:-0}; W_DT=${W_DT:-0}; W_IO=${W_IO:-1}
COSTNAME=ratio
[ "$W_CT" = 0 ] && [ "$W_DT" = 0 ] || COSTNAME=balanced

# shellcheck source=/dev/null
. "$PB/lammps/common.sh"
bench_setup || exit 1
rm -rf "$STORE"; mkdir -p "$STORE"
bench_compose "$STORE"

sed -i "/neuropress_best_mode:/a\\    neuropress_mape_threshold: $MAPE" "$STORE/compose.yaml"
grep -q "neuropress_mape_threshold: $MAPE" "$STORE/compose.yaml" || {
  echo "MAPE threshold did not land; refusing to run at the 0.30 default" >&2; exit 3; }
grep -q "neuropress_online_learning_enabled: $NP_LEARN" "$STORE/compose.yaml" || {
  echo "compose does not carry online_learning=$NP_LEARN as asked" >&2; exit 3; }

export LD_LIBRARY_PATH="$BUILD/bin:${LAMMPS_BUILD_DIR:?set LAMMPS_BUILD_DIR to your LAMMPS build}:${LD_LIBRARY_PATH:-}"
NATOMS=$(( 4 * BOX * BOX * BOX ))
echo "== learn/lammps: box $BOX = $NATOMS atoms, $STEPS steps, gap $GAP, chunk $CHUNK"
echo "   eb $EB, MAPE $MAPE, ratio cost, float32 downcast, port $PORT"

START=$(date +%s.%N)
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_LMP_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv" \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    CLIO_NEUROPRESS_ERROR_BOUND="$EB" \
    CLIO_NEUROPRESS_COST_BW=${COST_BW:-5e6} \
    CLIO_NEUROPRESS_COST_W_CT="$W_CT" \
    CLIO_NEUROPRESS_COST_W_DT="$W_DT" \
    CLIO_NEUROPRESS_COST_W_IO="$W_IO" \
    CLIO_NEUROPRESS_MEASURE_QUALITY=1 \
    CLIO_NEUROPRESS_STAGE_H2D=1 \
    "$BIN" --deck "$PB/lammps/in.melt" --box "$BOX" --steps "$STEPS" --gap "$GAP" \
           --chunk "$CHUNK" --order id --kokkos --quiet --expect-lossy --f32 \
           --log "$STORE/log.lammps" --report "$STORE/blobs.csv" \
           --var TEMP=6.0 --var SKIN=0.8 --var EVERY=5 \
    > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
cat > "$STORE/meta.json" <<JSON
{"config":"learn","tag":"lammps_learn2k","rc":$RC,"mode":"lossy",
 "cost_model":"$COSTNAME","cost_weights":{"ct":$W_CT,"dt":$W_DT,"io":$W_IO},"bw_bytes_per_ms":5e6,
 "error_bound":$EB,"mape_threshold":$MAPE,"online_learning":$NP_LEARN,
 "explore":$NP_EXPLORE,"explore_k":$EXPLORE_K,"best":$BEST,
 "device":"gpu","workload":"lammps-ljmelt",
 "atoms":$NATOMS,"steps":$STEPS,"gap":$GAP,"chunk":$CHUNK,"precision":"float32 (downcast from LAMMPS float64)",
 "wall_s":$WALL}
JSON
[ $RC -ne 0 ] && { echo "   FAILED rc=$RC"; grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -12; }
grep -E "^stored|  compressed:|^  codec|  time:|atoms" "$STORE/stdout.log" | sed 's/^/   /'
exit $RC
