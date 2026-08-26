#!/usr/bin/env bash
# Run ONE named configuration of the VPIC benchmark IN SITU -- GPU-resident.
#
#   ./run_config_insitu.sh <config> [--ncell N] [--steps N] [--int N]
#                          [--chunk B] [--bw B/ms] [--eb X] [--explore-k K]
#                          [--explore-thresh X] [--results DIR] [--tag NAME]
#
# The difference from run_config.sh that matters: the deck hands the compressor
# its Kokkos field views from begin_diagnostics -- DEVICE memory, uncopied -- so
# quantization, byte shuffle and codec selection all take their GPU paths. The
# replay route in run_config.sh reads .f32 files into host shm, where all three
# silently take host paths instead.
#
# CLIO_NEUROPRESS_REQUIRE_DEVICE=1 is set unconditionally here: a host-resident
# chunk is REFUSED rather than quietly computed on the CPU. That is the whole
# point of this script, and a silent fallback is indistinguishable from success
# in the results -- the ratios come out identical and only the timings move.
#
# COST OF IN-SITU, stated plainly: every configuration re-runs the simulation,
# so unlike the replay route the cells do NOT see byte-identical input.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INSITU=$(cd "$HERE/../../context-transfer-engine/compressor/example/neuropress_vpic_insitu" && pwd)

case "${1:-}" in -h|--help) sed -n '2,24p' "$0"; exit 0;; esac
CONFIG=${1:-explore-balance}; shift || true

NCELL=126 STEPS=200 INT=25 CHUNK=4194304
BW="" EB="" EXPLORE_K_OPT=3 THRESH_OPT=0
RESULTS="$HERE/results" TAG="" VERIFY=0 CHECK_BOUND=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int|--dump-int) INT=$2; shift 2;;
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
    # LOSSY: |original - decoded| <= eb, elementwise, against the bytes the
    # simulation submitted -- there is no source file in situ.
    --check-bound) CHECK_BOUND=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Same two cost models as the replay route, so the numbers compare.
RATIO_ONLY=(CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1)
COST_ENV=()
case "$CONFIG" in
  explore-balance) COSTMODEL=balance ;;
  explore-ratio)   COSTMODEL=ratio; COST_ENV=("${RATIO_ONLY[@]}") ;;
  *) echo "unknown config: $CONFIG (explore-balance|explore-ratio)" >&2; exit 2;;
esac

MODE=lossless
[ -n "$EB" ] && awk -v e="$EB" 'BEGIN{exit !(e+0>0)}' && MODE=lossy

NAME=${TAG:-$CONFIG}
STORE=$RESULTS/$NAME
rm -rf "$STORE"; mkdir -p "$STORE"

echo "== $NAME: VPIC IN SITU ${NCELL}^3, $STEPS steps, dump every $INT, chunk $CHUNK, $MODE/$COSTMODEL"

# VPIC's run.sh defaults to a FIXED port 9413, which collides with any other
# Clio runtime on the machine -- and the collision kills the client before it
# does anything, so it reads as a flaky run rather than a port clash.
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")

START=$(date +%s.%N)
set +e
env "${COST_ENV[@]}" \
    ${BW:+CLIO_NEUROPRESS_COST_BW=$BW} \
    ${EB:+CLIO_NEUROPRESS_ERROR_BOUND=$EB} \
    CLIO_NEUROPRESS_REQUIRE_DEVICE=1 \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=${MEASURE_DT:-1} \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv" \
    "$INSITU/run.sh" --port "$PORT" --ncell "$NCELL" --steps "$STEPS" --int "$INT" \
        --chunk "$CHUNK" --learn --explore "$EXPLORE_K_OPT" \
        --threshold "$THRESH_OPT" --store "$STORE" \
        ${VERIFY:+$([ "$VERIFY" = 1 ] && echo --verify)} \
        ${CHECK_BOUND:+$([ "$CHECK_BOUND" = 1 ] && echo --check-bound)} \
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

# The adapter prints its verdict on stdout; recorded here so summary.csv does
# not have to scrape it, and so an unverified run reads "n/a" rather than a
# silent 0 that looks like a pass.
VERDICT=n/a
if grep -q "BOUND OK:" "$STORE/stdout.log" 2>/dev/null; then VERDICT=bound-ok
elif grep -q "BOUND FAILED:" "$STORE/stdout.log" 2>/dev/null; then VERDICT=BOUND-FAIL
elif grep -q "VERIFIED:" "$STORE/stdout.log" 2>/dev/null; then VERDICT=pass
elif grep -q "^\[clio-vpic-insitu\] FAILED:" "$STORE/stdout.log" 2>/dev/null; then VERDICT=FAIL
fi

NCHUNKS=0
[ -f "$STORE/blobs.csv" ] && NCHUNKS=$(( $(wc -l < "$STORE/blobs.csv") - 1 ))
PAYLOAD=$(awk -F, 'NR>1{s+=$2} END{printf "%d", s+0}' "$STORE/blobs.csv" 2>/dev/null || echo 0)
FRAMES=$(( STEPS / INT + 1 ))

cat > "$STORE/meta.json" <<JSON
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"mode":"$MODE",
 "cost_model":"$COSTMODEL","bw_bytes_per_ms":${BW:-5e6},
 "error_bound":${EB:-0},"explore_k":$EXPLORE_K_OPT,"explore_thresh":$THRESH_OPT,
 "device":"gpu","workload":"vpic-weibel","residency":"device","host_refusals":$HOSTED,
 "atoms":0,"steps":$STEPS,"gap":$INT,"frames":$FRAMES,"chunk":$CHUNK,
 "files":$NCHUNKS,"payload_bytes":$PAYLOAD,"wall_s":$WALL,
 "verify_result":"$VERDICT",
 "physics":{"problem":"weibel","precision":"float32","insitu":"yes"}}
JSON
grep -E "stored [0-9]+ blob" "$STORE/stdout.log" | sed 's/^/   /' || true
exit $RC
