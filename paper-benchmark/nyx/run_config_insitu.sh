#!/usr/bin/env bash
# Run ONE named configuration of the Nyx benchmark IN SITU -- GPU-resident.
#
#   ./run_config_insitu.sh <config> [--ncell N] [--steps N] [--int N]
#                          [--chunk B] [--bw B/ms] [--eb X] [--explore-k K]
#                          [--explore-thresh X] [--verify] [--check-bound]
#                          [--results DIR] [--tag NAME]
#
# The difference from run_config.sh that matters: the simulation hands the
# compressor `fab.dataPtr(comp)` -- AMReX DEVICE memory, uncopied -- from
# inside Nyx::updateInSitu(), so quantization, byte shuffle and codec
# selection all take their GPU paths. The replay route in run_config.sh reads
# .f32 files into host shm, where all three silently take host paths instead.
#
# CLIO_NEUROPRESS_REQUIRE_DEVICE=1 is set unconditionally here: a host-resident
# chunk is REFUSED rather than quietly computed on the CPU. That is the whole
# point of this script, and a silent fallback is indistinguishable from success
# in the results -- the ratios come out identical and only the timings move.
#
# COST OF IN-SITU, stated plainly: every configuration re-runs the simulation,
# so unlike the replay route the cells do NOT see byte-identical input. Nyx on
# one GPU is deterministic for a fixed deck, so in practice they match, but
# that is a property of the run rather than something this enforces.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSITU=$(cd "$HERE/../../context-transfer-engine/compressor/example/neuropress_nyx_insitu" && pwd)

case "${1:-}" in -h|--help) sed -n '2,24p' "$0"; exit 0;; esac
CONFIG=${1:-explore-balance}; shift || true

NCELL=128 STEPS=200 INT=10 CHUNK=4194304
# nyx.cfl. 0.8 is Nyx's own documented default (NyxInputs.rst); the Sedov deck
# ships 0.5. It is the ONLY knob that changes how fast this data evolves at a
# fixed step count -- exp_energy cancels exactly, because the Sedov shock
# advances cfl*dx/c1 cells per timestep whatever the blast energy. See
# "Default Evolving Benchmark Configuration" in README.md.
CFL=0.8
BW="" EB="" EXPLORE_K_OPT=3 THRESH_OPT=0
RESULTS="$HERE/results" TAG="" VERIFY=0 CHECK_BOUND=0
EXP_ENERGY="" STOP_TIME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int|--plot-int) INT=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --bw) BW=$2; shift 2;;
    --eb) EB=$2; shift 2;;
    --explore-k) EXPLORE_K_OPT=$2; shift 2;;
    --explore-thresh) THRESH_OPT=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    # Bit-exact digest check of every blob, through the decompressor. Only
    # meaningful lossless; the adapter downgrades it to a bound check if an
    # error bound is set.
    --verify) VERIFY=1; shift;;
    # LOSSY: |original - decoded| <= eb, elementwise. Unlike the replay route
    # this does NOT re-read a source file -- there is none in situ -- so the
    # adapter holds each frame's submitted bytes for one frame and compares
    # against those.
    --check-bound) CHECK_BOUND=1; shift;;
    # Sedov blast energy and the physical time the run stops at. Both reach
    # the deck verbatim. The defaults leave the data nearly static: the shock
    # radius goes as R = 1.033*(E t^2/rho)^(1/5), so at the deck's E=1 and
    # stop_time=0.01 it reaches only ~0.16 of a unit box and most of the
    # domain never changes. Raising E does two things at once -- it widens the
    # shock AND, because the timestep is CFL-limited by the higher velocities,
    # it takes MORE steps to reach the same stop_time. Measured at 128^3:
    # E=1 -> ~300 steps and a 6.9x ratio range, E=10 -> ~675 and 44x,
    # E=100 -> ~1575 and 114x.
    --exp-energy) EXP_ENERGY=$2; shift 2;;
    --cfl) CFL=$2; shift 2;;
    --stop-time)  STOP_TIME=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Same cost models as the replay route, so the numbers compare.
RATIO_ONLY=(CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1)
# SPEED is the third corner of the cost model: latency only, I/O weight zeroed,
# so cost = w_ct*compress_ms + w_dt*decompress_ms and the ratio term drops out
# entirely. It is the complement of `ratio` (which zeroes the two latency
# weights), and together the three span what the model can be asked to optimise.
# Expect it to pick the fastest codec regardless of how little it compresses --
# on data that barely compresses the balanced model already behaves this way,
# because the 1 ms clamps dominate its cost.
SPEED_ONLY=(CLIO_NEUROPRESS_COST_W_CT=1 CLIO_NEUROPRESS_COST_W_DT=1 CLIO_NEUROPRESS_COST_W_IO=0)
COST_ENV=()
case "$CONFIG" in
  explore-balance) COSTMODEL=balance ;;
  explore-ratio)   COSTMODEL=ratio; COST_ENV=("${RATIO_ONLY[@]}") ;;
  explore-speed)   COSTMODEL=speed; COST_ENV=("${SPEED_ONLY[@]}") ;;
  *) echo "unknown config: $CONFIG (explore-balance|explore-ratio|explore-speed)" >&2; exit 2;;
esac

MODE=lossless
[ -n "$EB" ] && awk -v e="$EB" 'BEGIN{exit !(e+0>0)}' && MODE=lossy

NAME=${TAG:-$CONFIG}
STORE=$RESULTS/$NAME
rm -rf "$STORE"; mkdir -p "$STORE"

echo "== $NAME: Nyx IN SITU ${NCELL}^3, $STEPS steps, dump every $INT, chunk $CHUNK, $MODE/$COSTMODEL"

START=$(date +%s.%N)
set +e
env "${COST_ENV[@]}" \
    ${BW:+CLIO_NEUROPRESS_COST_BW=$BW} \
    ${EB:+CLIO_NEUROPRESS_ERROR_BOUND=$EB} \
    CLIO_NEUROPRESS_REQUIRE_DEVICE=1 \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=${MEASURE_DT:-1} \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv" \
    "$INSITU/run.sh" --ncell "$NCELL" --steps "$STEPS" --int "$INT" \
        --chunk "$CHUNK" --learn --explore "$EXPLORE_K_OPT" \
        --threshold "$THRESH_OPT" --store "$STORE" \
        ${VERIFY:+$([ "$VERIFY" = 1 ] && echo --verify)} \
        ${CHECK_BOUND:+$([ "$CHECK_BOUND" = 1 ] && echo --check-bound)} \
        nyx.cfl=$CFL \
        ${EXP_ENERGY:+prob.exp_energy=$EXP_ENERGY} \
        ${STOP_TIME:+stop_time=$STOP_TIME} \
        > "$RESULTS/.$NAME.stdout" 2>&1
RC=$?
set -e
# run.sh wipes $STORE on entry (rm -rf), so the driver log lands outside and
# is moved in once the store exists again.
mkdir -p "$STORE"; mv -f "$RESULTS/.$NAME.stdout" "$STORE/stdout.log" 2>/dev/null || true
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')

# A refusal means a chunk reached the compressor in host memory. Surface it:
# the run is worthless for this benchmark's purpose even if it exits 0.
HOSTED=0
[ -f "$STORE/runtime.log" ] && HOSTED=$(grep -c "REQUIRE_DEVICE is set" "$STORE/runtime.log" || true)
HOSTED=${HOSTED:-0}
if [ "$HOSTED" -gt 0 ]; then
  echo "   HOST-RESIDENT CHUNKS REFUSED: $HOSTED -- this run did NOT stay on the GPU"
  RC=1
fi
[ $RC -ne 0 ] && { echo "   FAILED rc=$RC"; tail -5 "$STORE/stdout.log"; }

# The adapter prints its verdict on stdout, which run.sh captures into
# nyx.log and echoes back through its [clio-nyx-insitu] grep -- so it lands in
# stdout.log either way. Recorded here so summary.csv does not have to scrape
# it, and so a run that was never asked to verify reads "n/a" rather than a
# silent 0 that looks like a pass.
VERDICT=n/a
if grep -q "BOUND OK:" "$STORE/stdout.log" 2>/dev/null; then VERDICT=bound-ok
elif grep -q "BOUND FAILED:" "$STORE/stdout.log" 2>/dev/null; then VERDICT=BOUND-FAIL
elif grep -q "VERIFIED:" "$STORE/stdout.log" 2>/dev/null; then VERDICT=pass
elif grep -q "^\[clio-nyx-insitu\] FAILED:" "$STORE/stdout.log" 2>/dev/null; then VERDICT=FAIL
fi

NCHUNKS=0
[ -f "$STORE/blobs.csv" ] && NCHUNKS=$(( $(wc -l < "$STORE/blobs.csv") - 1 ))
PAYLOAD=$(awk -F, 'NR>1{s+=$2} END{printf "%d", s+0}' "$STORE/blobs.csv" 2>/dev/null || echo 0)
FRAMES=$(( STEPS / INT + 1 ))

cat > "$STORE/meta.json" <<JSON
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"mode":"$MODE",
 "cost_model":"$COSTMODEL","bw_bytes_per_ms":${BW:-5e6},
 "error_bound":${EB:-0},"explore_k":$EXPLORE_K_OPT,"explore_thresh":$THRESH_OPT,
 "device":"gpu","workload":"nyx-sedov","residency":"device","host_refusals":$HOSTED,
 "exp_energy":${EXP_ENERGY:-1},"stop_time":${STOP_TIME:-0.01},"cfl":$CFL,
 "atoms":0,"steps":$STEPS,"gap":$INT,"frames":$FRAMES,"chunk":$CHUNK,
 "files":$NCHUNKS,"payload_bytes":$PAYLOAD,"wall_s":$WALL,
 "verify_result":"$VERDICT",
 "physics":{"problem":"sedov","precision":"float32","insitu":"yes"}}
JSON
grep -E "stored [0-9]+ blob" "$STORE/stdout.log" | sed 's/^/   /' || true
exit $RC
