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
#   --temp X       initial temperature                       (default 3.0)
#   --cutoff X     lj/cut pair cutoff, sigma                 (default 2.5)
#   --skin X       neighbor-list skin                        (default 0.3)
#   --every N      neigh_modify every                        (default 20)
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
DENSITY="" TEMP="" CUTOFF="" SKIN="" EVERY="" SEED="" DT=""
VARS=()
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
  *) echo "unknown config: $CONFIG" >&2; sed -n '2,34p' "$0" >&2; exit 2;;
esac
export NP_LEARN NP_EXPLORE EXPLORE_K THRESH BEST STATIC_LIB STATIC_SHUF

# shellcheck source=common.sh
. "$HERE/common.sh"
bench_setup || exit 1

NAME=${TAG:-$CONFIG}
STORE=$RESULTS/$NAME
rm -rf "$STORE"; mkdir -p "$STORE"
bench_compose "$STORE"

NATOMS=$(( 4 * BOX * BOX * BOX ))
FRAMES=$(( STEPS / GAP + 1 ))
PAYLOAD=$(( NATOMS * 3 * 8 * 3 * FRAMES ))
echo "== $NAME: $NATOMS atoms, $STEPS steps, $FRAMES frames, $(awk -v p=$PAYLOAD "BEGIN{printf \"%.1f\", p/1048576}") MiB payload, ${DEVICE^^}, chunk $CHUNK, port $PORT"

ARGS=(--deck "$HERE/in.melt" --box "$BOX" --steps "$STEPS" --gap "$GAP"
      --chunk "$CHUNK" --order id --log "$STORE/log.lammps"
      --report "$STORE/blobs.csv" --quiet "${VARS[@]+"${VARS[@]}"}")
[ "$DEVICE" = gpu ] && ARGS+=(--kokkos)
[ "$VERIFY" = 1 ]   && ARGS+=(--verify)

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
START=$(date +%s.%N)
set +e
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_LMP_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    ${NP_EXPLORE:+$([ "$NP_EXPLORE" = true ] && echo CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv")} \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warn}" \
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

# Machine-readable record for collect.py, alongside the per-chunk CSV.
cat > "$STORE/meta.json" <<JSON
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"device":"$DEVICE","box":$BOX,
 "atoms":$NATOMS,"steps":$STEPS,"gap":$GAP,"frames":$FRAMES,"chunk":$CHUNK,
 "payload_bytes":$PAYLOAD,"wall_s":$WALL,"verified":$VERIFY,"port":$PORT,
 "physics":{"density":"${DENSITY:-deck}","temp":"${TEMP:-deck}",
            "cutoff":"${CUTOFF:-deck}","skin":"${SKIN:-deck}",
            "every":"${EVERY:-deck}","seed":"${SEED:-deck}","dt":"${DT:-deck}"}}
JSON
grep -E "^stored|VERIFIED|FAILED:|  time:" "$STORE/stdout.log" | sed 's/^/   /' || true
exit $RC
