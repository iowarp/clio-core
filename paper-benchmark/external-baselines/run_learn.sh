#!/usr/bin/env bash
# NeuroPress LEARNING phase, ratio cost model, on a long Nyx stream.
#
# Lives here rather than in the repo: paper-benchmark has no learn+ratio
# config and no way to set the MAPE threshold, and adding either would be a
# benchmark change rather than an experiment. common.sh is sourced so the pool
# layout, weights path and free-port selection stay identical to every other
# arm; only the three things under test are altered afterwards.
#
#   1. neuropress_mape_threshold -> $MAPE. Compose-only key (default 0.30),
#      never emitted by bench_compose. It gates Phase-1 SGD:
#      `error_pct > threshold` (compressor_runtime.cc:1917).
#
#   2. RATIO cost model -- w_ct = w_dt = 0, w_io = 1. This is what makes the
#      gate meaningful. compressor_runtime.cc:1893 says it outright: under
#      BALANCED weights the io term is ~0.4% of the cost, so the ratio head
#      can be wrong by 4-5x while the cost error stays under the gate and is
#      never corrected. With the latency weights zeroed the cost IS the ratio
#      term, so a wrong ratio trips the gate.
#
#   3. Tier sizing from the chunks actually replayed. bench_compose sizes it
#      from `du` over the whole field directory -- 27 GB here -- which would
#      preallocate a 55 GB bdev for a 16 GB run.
#
#   run_learn.sh <fields> <max-files> <chunk> <eb> <mape> <store>
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export BUILD=${BUILD:-$REPO/build}
FIELDS=$1; MAXF=$2; CHUNK=$3; EB=$4; MAPE=$5; STORE=$6
export FIELDS

# NP_LEARN=false from the environment gives the frozen-weights control: same
# inference path, same chunks, no SGD at all. It is what "learning off" means.
# Overridable, with the previous hardcoded values as the defaults, so every
# run in the published matrix behaves exactly as it did.
export NP_LEARN=${NP_LEARN:-true} NP_EXPLORE=${NP_EXPLORE:-false} \
       EXPLORE_K=${EXPLORE_K:-0} THRESH=${THRESH:-0.5} BEST=${BEST:-false}
export STATIC_LIB="" STATIC_SHUF=0

# Cost weights. Default is RATIO-ONLY, which is what makes the MAPE gate a
# ratio gate: under the balanced weights both time terms sit at their 1 ms
# floor and contribute a fixed 2.0 ms while the ratio term contributes ~0.04,
# so a ratio wrong by 5x can pass a 10% gate untouched.
W_CT=${W_CT:-0}; W_DT=${W_DT:-0}; W_IO=${W_IO:-1}
COSTNAME=ratio
[ "$W_CT" = 0 ] && [ "$W_DT" = 0 ] || COSTNAME=balanced

# shellcheck source=/dev/null
. "$REPO/paper-benchmark/nyx/common.sh"
bench_setup || exit 1
rm -rf "$STORE"; mkdir -p "$STORE"
bench_compose "$STORE"

# (1) the MAPE gate
sed -i "/neuropress_best_mode:/a\\    neuropress_mape_threshold: $MAPE" "$STORE/compose.yaml"
grep -q "neuropress_mape_threshold: $MAPE" "$STORE/compose.yaml" || {
  echo "MAPE threshold did not land in the compose; refusing to run at the" >&2
  echo "0.30 default and report it as $MAPE" >&2; exit 3; }
grep -q "neuropress_online_learning_enabled: $NP_LEARN" "$STORE/compose.yaml" || {
  echo "compose does not carry online_learning=$NP_LEARN as asked" >&2; exit 3; }

# (3) tier for the chunks replayed, not the whole dataset
PAY_MB=$(( MAXF * CHUNK / 1048576 ))
TIER_MB=$(( PAY_MB + 512 )); BDEV_MB=$(( TIER_MB * 2 ))
sed -i "s/^    capacity: \"[0-9]*MB\"/    capacity: \"${BDEV_MB}MB\"/" "$STORE/compose.yaml"
sed -i "s/^        capacity_limit: \"[0-9]*MB\"/        capacity_limit: \"${TIER_MB}MB\"/" "$STORE/compose.yaml"

export LD_LIBRARY_PATH="$BUILD/bin:${LD_LIBRARY_PATH:-}"
echo "== learn: $MAXF chunks x $CHUNK B = $((PAY_MB/1024)) GiB, eb $EB, MAPE $MAPE, ratio cost, port $PORT"
echo "   tier ${TIER_MB}MB  bdev ${BDEV_MB}MB"

START=$(date +%s.%N)
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
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
    "$BIN" --dir "$FIELDS" --ext .f32 --chunk "$CHUNK" --max-files "$MAXF" \
           --tag nyx_learn --report "$STORE/blobs.csv" \
    > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
cat > "$STORE/meta.json" <<JSON
{"config":"learn","tag":"learn2k","rc":$RC,"mode":"lossy",
 "cost_model":"$COSTNAME","cost_weights":{"ct":$W_CT,"dt":$W_DT,"io":$W_IO},"bw_bytes_per_ms":5e6,
 "error_bound":$EB,"mape_threshold":$MAPE,"online_learning":$NP_LEARN,
 "explore":$NP_EXPLORE,"explore_k":$EXPLORE_K,"explore_threshold":$THRESH,
 "best":$BEST,"device":"gpu","workload":"nyx-sedov",
 "chunk":$CHUNK,"chunks":$MAXF,"wall_s":$WALL}
JSON
[ $RC -ne 0 ] && { echo "   FAILED rc=$RC"; grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -10; }
grep -E "^stored|  compressed:|^  codec|  time:" "$STORE/stdout.log" | sed 's/^/   /'
exit $RC
