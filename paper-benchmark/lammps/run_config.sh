#!/usr/bin/env bash
# Run ONE named configuration of the LAMMPS paper benchmark.
#
#   ./run_config.sh <config> [run options] [physics options]
#
# Run options:
#   --box N        lattice cells/side; atoms = 4*N^3        (default 80)
#   --steps N      total timesteps                          (default 300)
#   --gap N        store a frame every N steps              (default 50)
#   --chunk BYTES  bytes per compressor call                (default 4 MiB)
#   --cpu          run LAMMPS on the CPU (bit-reproducible)
#   --no-verify    skip the round-trip check
#   --results DIR  where run directories go
#   --tag NAME     name this run (defaults to the config name)
#
# Physics options -- passed through to the deck as LAMMPS -var, which takes
# precedence over its `variable ... index` defaults:
#   --density X    fcc lattice reduced density               (default 0.8442)
#   --temp X       initial temperature                       (default 6.0)
#   --cutoff X     lj/cut pair cutoff, sigma                 (default 2.5)
#   --skin X       neighbor-list skin                        (default 0.8)
#   --every N      neigh_modify every                        (default 5)
#   --seed N       velocity RNG seed                         (default 87287)
#   --dt X         timestep                                  (default 0.005)
#   --var K=V      any other deck variable (repeatable)
#
# Configurations differ only in how the codec is chosen for each chunk:
#
#   dynamic        NeuroPress inference, default balanced cost model
#                  (w_ct = w_dt = w_io = 1, bandwidth 5 GB/s) -- THE headline
#                  configuration: one forward pass per chunk, no measurement.
#   dynamic-ratio  same, with the two latency weights zeroed so the cost
#                  collapses to bytes/(ratio*bw) -- a ratio-only objective.
#   learn          dynamic + online SGD from each chunk's measured outcome.
#   explore        dynamic-ratio + exploration: the top-K alternatives are
#                  actually compressed and the measured winner adopted.
#   best           best mode: exhaustive, ratio-only ranking.
#   static-zstd    fixed nvcomp-zstd, no shuffle -- codec control, no model.
#   static-zstd-s4 fixed nvcomp-zstd + 4-byte shuffle (upstream's only width).
#   static-zstd-s8 fixed nvcomp-zstd + 8-byte shuffle (matches float64; NOT
#                  reachable by the model, whose action space encodes shuffle
#                  as one bit meaning 4 bytes).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "${1:-}" in
  -h|--help) sed -n '2,34p' "$0"; exit 0;;
esac
CONFIG=${1:-dynamic}; shift || true
BOX=80 STEPS=300 GAP=50 CHUNK=4194304
DEVICE=gpu VERIFY=1 RESULTS="$HERE/results" TAG=""
# Physics: empty means "let the deck's own default stand", so an unset knob
# never has to be duplicated here and drift from the deck.
# Physics: empty means "let the deck's own default stand". TEMP, SKIN and EVERY
# are NOT empty, and the deck is not changed to match -- in.melt stays exactly
# upstream's melt example so it still runs standalone as upstream wrote it, and
# the benchmark's chosen operating point lives here instead.
#
# TEMP 6.0 rather than upstream's 3.0: the 1,000-step evolution study measured
# 0.4368 mean block evolution with 100% of blocks active against 0.3892 / 83.6%
# at 3.0.
#
# SKIN 0.8 and EVERY 5 rather than 0.3 / 20 are a CORRECTNESS condition for that
# temperature, not tuning. Upstream's skin is sized for T=3.0, where an atom
# moves ~0.22 sigma between rebuilds; at T=6.0 it moves ~0.33 sigma, crosses the
# 0.3 sigma skin, and pairs are missed -- NVE then loses 3.5% of its total
# energy over 1,000 steps against 0.37% here. See "Default Evolving Benchmark
# Configuration" in README.md.
DENSITY="" TEMP=6.0 CUTOFF="" SKIN=0.8 EVERY=5 SEED="" DT=""
VARS=()
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
# At 0 nearly every chunk explores, so nearly every compressed chunk yields a
# measured decompression time and the only variable left is the one under test.
#
# NEARLY, not every: the gate is `error_pct > threshold`, STRICT, and it is
# upstream's own (gpucompress_compress.cpp:733). A chunk whose cost the model
# predicted EXACTLY right has error_pct == 0 and does not explore at any
# threshold, 0 included. That is rare under the balance weights and common
# under the ratio ones, because ratio-only cost is bytes/(min(ratio,100)*bw)
# and nothing else: once the predicted AND actual ratio both clear the 100x
# cap the two costs are the same number, so the model is "exactly right" by
# arithmetic rather than by accuracy. Measured on Nyx 128^3, explore-ratio:
# 56 of 120 chunks lossless and 97 of 120 lossy never explored. Read
# `explored` in summary.csv rather than assuming full coverage.
# K stays at 3, NeuroPress's own ranked window.
# GPU-ONLY mode. Two things at once, because either alone is a half-measure:
#   --order device  makes the driver gather each frame's chunk ON THE GPU into
#                   a CUDA-IPC-registered backend, so the pointer handed to the
#                   compressor is device memory. The default --order id gathers
#                   through lammps_gather_atoms into a HOST array.
#   REQUIRE_DEVICE  refuses a host-resident chunk at the compressor instead of
#                   quietly running quantization, byte shuffle and codec
#                   selection on the CPU -- all three branch on the residency
#                   of that same pointer.
# Without the second, a regression in the first is invisible: the ratios come
# out identical and only the timings move.
REQUIRE_DEVICE=0
BW="" EB="" EXPLORE_K_OPT=3 THRESH_OPT=0 RAW_DIR="" F32=0 DECK=""
DECOMP_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --box) BOX=$2; shift 2;;
    --density) DENSITY=$2; shift 2;;
    --temp) TEMP=$2; shift 2;;
    --cutoff) CUTOFF=$2; shift 2;;
    --skin) SKIN=$2; shift 2;;
    --every) EVERY=$2; shift 2;;
    --seed) SEED=$2; shift 2;;
    --dt) DT=$2; shift 2;;
    --var) VARS+=(--var "$2"); shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --cpu) DEVICE=cpu; shift;;
    --no-verify) VERIFY=0; shift;;
    --results) RESULTS=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --bw) BW=$2; shift 2;;
    --eb) EB=$2; shift 2;;
    --raw) RAW_DIR=$2; shift 2;;
    # The decompressed bytes, for an external comparison against --raw.
    --dump-decompressed) DECOMP_DIR=$2; shift 2;;
    --f32) F32=1; shift;;
    --deck) DECK=$2; shift 2;;
    --explore-k) EXPLORE_K_OPT=$2; shift 2;;
    --explore-thresh) THRESH_OPT=$2; shift 2;;
    --require-device) REQUIRE_DEVICE=1; shift;;
    -h|--help) sed -n '2,34p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
export BOX STEPS GAP

# Physics knobs the caller actually set become -var pairs; the rest stay at
# the deck's defaults.
for kv in DENSITY TEMP CUTOFF SKIN EVERY SEED DT; do
  v=${!kv}
  [ -n "$v" ] && VARS+=(--var "$kv=$v")
done

# Selection policy. COST_ENV carries the cost-model weight overrides, which
# reach BOTH the ranking and the SGD gate (they are one set of globals
# upstream, and Clio follows that).
NP_LEARN=false NP_EXPLORE=false EXPLORE_K=0 THRESH=0.5 BEST=false
STATIC_LIB="" STATIC_SHUF=0
COST_ENV=()
# Which of the two cost models this config asks for, recorded in meta.json so
# a run is self-describing. The pre-existing configs keep their historical
# behaviour: `dynamic`/`learn` rank under the default balanced weights, the
# rest under the ratio-only ones.
case "$CONFIG" in
  dynamic|learn|explore-balance) COSTMODEL=balance ;;
  explore-speed)                 COSTMODEL=speed ;;
  static-*)                      COSTMODEL=none ;;
  *)                             COSTMODEL=ratio ;;
esac
RATIO_ONLY=(CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1)
# SPEED is the third corner of the cost model: latency only, I/O weight zeroed,
# so cost = w_ct*compress_ms + w_dt*decompress_ms and the ratio term drops out
# entirely. It is the complement of `ratio` (which zeroes the two latency
# weights), and together the three span what the model can be asked to optimise.
# Expect it to pick the fastest codec regardless of how little it compresses --
# on data that barely compresses the balanced model already behaves this way,
# because the 1 ms clamps dominate its cost.
SPEED_ONLY=(CLIO_NEUROPRESS_COST_W_CT=1 CLIO_NEUROPRESS_COST_W_DT=1 CLIO_NEUROPRESS_COST_W_IO=0)
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
  explore-speed)   NP_LEARN=true; NP_EXPLORE=true
                   EXPLORE_K=$EXPLORE_K_OPT; THRESH=$THRESH_OPT
                   COST_ENV=("${SPEED_ONLY[@]}") ;;
  static-zstd)    STATIC_LIB=nvcomp-zstd; STATIC_SHUF=0 ;;
  static-zstd-s4) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=4 ;;
  static-zstd-s8) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=8 ;;
  *) echo "unknown config: $CONFIG" >&2; sed -n '2,34p' "$0" >&2; exit 2;;
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

NATOMS=$(( 4 * BOX * BOX * BOX ))
FRAMES=$(( STEPS / GAP + 1 ))
# 4 bytes per element under --f32, not 8: the gather kernel narrows on the
# device (lammps_device_view.cc, GatherIdWindowT<float>), so the payload the
# compressor sees is float32. Sizing it as double over-reported the 30 GiB
# cell as 61,523 MiB and wrote the same 2x error into meta.json.
ELEM_BYTES=8; [ "$F32" = 1 ] && ELEM_BYTES=4
PAYLOAD=$(( NATOMS * 3 * ELEM_BYTES * 3 * FRAMES ))
echo "== $NAME: $NATOMS atoms, $STEPS steps, $FRAMES frames, $(awk -v p=$PAYLOAD "BEGIN{printf \"%.1f\", p/1048576}") MiB payload, ${DEVICE^^}, chunk $CHUNK, port $PORT"

ORDER=id
[ "$REQUIRE_DEVICE" = 1 ] && ORDER=device
ARGS=(--deck "${DECK:-$HERE/in.melt}" --box "$BOX" --steps "$STEPS" --gap "$GAP"
      --chunk "$CHUNK" --order "$ORDER" --log "$STORE/log.lammps"
      --report "$STORE/blobs.csv" --quiet "${VARS[@]+"${VARS[@]}"}")
[ "$DEVICE" = gpu ] && ARGS+=(--kokkos)
[ "$VERIFY" = 1 ]   && ARGS+=(--verify)
# --raw writes the bytes handed to the compressor, which is the only way to see
# this workload at all: it is in-situ, so there is no dump file anywhere.
[ -n "$RAW_DIR" ] && { mkdir -p "$RAW_DIR"; ARGS+=(--raw "$RAW_DIR"); }
[ -n "$DECOMP_DIR" ] && { mkdir -p "$DECOMP_DIR"; ARGS+=(--dump-decompressed "$DECOMP_DIR"); }
# A positive bound means the decompressed bytes are NOT the bytes staged, so
# the digest check would report FAILED on a run doing exactly what was asked.
[ -n "$EB" ] && ARGS+=(--expect-lossy)
# --f32 stages a float32 downcast of LAMMPS' double state. Host gather only, so
# the compressor needs STAGE_H2D to see a device pointer -- which is what makes
# the quantizer reachable at all. See "Can LAMMPS be float32?" in README.md.
if [ "$F32" = 1 ]; then
  ARGS+=(--f32)
  if [ "$REQUIRE_DEVICE" = 1 ]; then
    # GPU-RESIDENT float32. The gather kernel itself writes float
    # (clio_lmp_device_gather_id_window_f32), so the payload is narrowed on
    # the device, in the same store that was already writing the element.
    # Nothing crosses PCIe and no STAGE_H2D is needed -- REQUIRE_DEVICE stays
    # on and still refuses a host-resident chunk at the compressor.
    :
  else
    # HOST float32, the original route: downcast during lammps_gather_atoms,
    # then stage H2D so the compressor still sees a device pointer. Kept
    # because it is the only float32 path for --order id, but it pays a real
    # H2D per chunk that the device gather does not -- so its TIMINGS are not
    # comparable with the device route's. Prefer --require-device.
    export CLIO_NEUROPRESS_STAGE_H2D=1
    export CLIO_NEUROPRESS_REQUIRE_DEVICE=1
  fi
fi

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
START=$(date +%s.%N)
set +e
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_LMP_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    ${NP_EXPLORE:+$([ "$NP_EXPLORE" = true ] && echo CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv")} \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    ${EB:+CLIO_NEUROPRESS_ERROR_BOUND=$EB} \
    ${BW:+CLIO_NEUROPRESS_COST_BW=$BW} \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=${MEASURE_DT:-1} \
    CLIO_NEUROPRESS_MEASURE_QUALITY=${MEASURE_QUALITY:-1} \
    ${REQUIRE_DEVICE:+CLIO_NEUROPRESS_REQUIRE_DEVICE=$REQUIRE_DEVICE} \
    "${COST_ENV[@]}" \
    "$BIN" "${ARGS[@]}" > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?
set -e
# awk, not bc: bc is not installed everywhere and a missing one would leave
# WALL empty, producing invalid JSON that collect.py then skips silently.
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')

if [ $RC -ne 0 ]; then
  echo "   FAILED rc=$RC"
  grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -8
  grep -q "Address already in use" "$STORE/runtime.log" && \
    echo "   (port $PORT was taken; rerun -- a free port is picked per run)"
fi

# A refusal means a chunk reached the compressor in host memory despite
# --order device. The run is worthless for a GPU-only benchmark even if the
# driver exits 0, so fail it loudly rather than publishing CPU numbers.
HOSTED=0
[ -f "$STORE/runtime.log" ] && HOSTED=$(grep -c "REQUIRE_DEVICE is set" "$STORE/runtime.log" || true)
HOSTED=${HOSTED:-0}
if [ "$HOSTED" -gt 0 ]; then
  echo "   HOST-RESIDENT CHUNKS REFUSED: $HOSTED -- this run did NOT stay on the GPU"
  RC=1
fi

# Machine-readable record for collect.py, alongside the per-chunk CSV.
cat > "$STORE/meta.json" <<JSON
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"mode":"$MODE",
 "cost_model":"$COSTMODEL","bw_bytes_per_ms":${BW:-5e6},
 "error_bound":${EB:-0},"explore_k":$EXPLORE_K,"explore_thresh":$THRESH,
 "residency":"$([ "$REQUIRE_DEVICE" = 1 ] && echo device || echo host)",
 "host_refusals":$HOSTED,"device":"$DEVICE","box":$BOX,
 "atoms":$NATOMS,"steps":$STEPS,"gap":$GAP,"frames":$FRAMES,"chunk":$CHUNK,
 "payload_bytes":$PAYLOAD,"wall_s":$WALL,"verified":$VERIFY,"port":$PORT,
 "physics":{"density":"${DENSITY:-deck}","temp":"${TEMP:-deck}",
            "cutoff":"${CUTOFF:-deck}","skin":"${SKIN:-deck}",
            "every":"${EVERY:-deck}","seed":"${SEED:-deck}","dt":"${DT:-deck}"}}
JSON
grep -E "^stored|VERIFIED|FAILED:|  time:" "$STORE/stdout.log" | sed 's/^/   /' || true
exit $RC
