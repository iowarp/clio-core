#!/usr/bin/env bash
#===============================================================================
# make_table.sh -- the whole accuracy table, from the corpus and one campaign.
#
#   ./make_table.sh --corpus CSV --campaign DIR --out DIR
#                   [--feedback-interval N] [--forget L]
#                   [--feedback-scope self|executed|all] [--constant-as NAME]
#                   [--skip-prepare]
#
# Three steps, in order:
#   1. prepare_inputs.py  one evaluation frame per setting + HCompress's seed
#   2. hcompress_ccp_eval the baseline, seeded and then fed back, per setting
#   3. accuracy_table.py  MAPE and R2 per (setting, model, metric) -> CSV + TeX
#
# Step 3 runs twice: once with the deployed policy clamps (the headline) and
# once without, because most measured compression times here are under the 1 ms
# floor those clamps impose and the difference belongs in the record.
#===============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CORPUS="" CAMPAIGN="" OUT="" N=1 FORGET=1.0 SCOPE=self CONSTANT_AS="" PREPARE=1
SEED=seed.csv TAG="" MAPE_CAP=100   # --seed-file picks the profiler sample
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus)   CORPUS=$2; shift 2;;
    --campaign) CAMPAIGN=$2; shift 2;;
    --out)      OUT=$2; shift 2;;
    --feedback-interval) N=$2; shift 2;;
    --forget)   FORGET=$2; shift 2;;
    --feedback-scope) SCOPE=$2; shift 2;;
    --constant-as) CONSTANT_AS=$2; shift 2;;
    --seed-file) SEED=$2; shift 2;;
    --tag)      TAG=$2; shift 2;;
    --mape-cap) MAPE_CAP=$2; shift 2;;
    --skip-prepare) PREPARE=0; shift;;
    -h|--help) sed -n '3,20p' "$0"; exit 0;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done
[ -n "$OUT" ] || { echo "--out DIR is required" >&2; exit 2; }
IN=$OUT/inputs
mkdir -p "$IN"

# Everything this prints is a record of HOW the table was built -- the seed's
# time rescaling factors, each setting's chunk and row counts, how many rows the
# baseline was fed and how often its own pick matched the run's. None of that is
# recoverable from the CSVs afterwards, so the script keeps its own log rather
# than relying on whoever ran it to capture the terminal.
LOG=$OUT/make_table${TAG}.log
exec > >(tee "$LOG") 2>&1
echo "== make_table.sh $(date -Is)"
echo "   corpus=${CORPUS:-<skipped>} campaign=${CAMPAIGN:-<skipped>} out=$OUT"
echo "   interval=$N forget=$FORGET scope=$SCOPE seed=$SEED tag=${TAG:-<none>}"

if [ "$PREPARE" = 1 ]; then
  [ -n "$CORPUS" ] && [ -n "$CAMPAIGN" ] || { echo "--corpus and --campaign are required" >&2; exit 2; }
  python3 "$HERE/prepare_inputs.py" --corpus "$CORPUS" --campaign "$CAMPAIGN" --out "$IN"
fi

[ -x "$HERE/hcompress_ccp_eval" ] || {
  echo "building hcompress_ccp_eval"
  REPO=$(cd "$HERE/../.." && pwd)
  g++ -std=c++17 -Wall -Wextra -O2 -I "$REPO/context-transport-primitives/include" \
    "$HERE/hcompress_ccp_eval.cc" \
    "$REPO/context-transport-primitives/src/compress/model/hcompress_ccp_predictor.cc" \
    -o "$HERE/hcompress_ccp_eval"
}

echo
for d in "$IN"/*/; do
  s=$(basename "$d")
  [ -s "$d/eval.csv" ] || continue
  extra=(); [ -n "$CONSTANT_AS" ] && extra=(--constant-as "$CONSTANT_AS")
  echo "== HCompress CCP: $s"
  "$HERE/hcompress_ccp_eval" --seed "$IN/$SEED" --eval "$d/eval.csv" \
    --out "$d/hcompress$TAG.csv" --feedback-interval "$N" --forget "$FORGET" \
    --feedback-scope "$SCOPE" "${extra[@]}" \
    --seed-json "$IN" 2>&1 | sed 's/^/   /'
done

echo
python3 "$HERE/accuracy_table.py" --inputs "$IN" --out "$OUT" --tag "$TAG" \
  --hc-suffix "$TAG" --mape-cap "$MAPE_CAP"
echo
python3 "$HERE/accuracy_table.py" --inputs "$IN" --out "$OUT" \
  --tag "${TAG}_unclamped" --hc-suffix "$TAG" --mape-cap "$MAPE_CAP" \
  --no-policy-clamp >/dev/null
echo "also wrote the no-clamp variant to $OUT"
echo "log -> $LOG"
