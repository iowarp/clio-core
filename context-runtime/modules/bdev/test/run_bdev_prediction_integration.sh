#!/usr/bin/env bash
# Full-chain bdev -> ML model integration driver (Linux only).
#
# Starts the real Python prediction server with the trained models, then runs
# the bdev test's `bdev_failure_prediction_integration` case, which drives a
# live bdev's Monitor("stats") -> device-health collection -> live inference and
# asserts the prediction round-trips back through the bdev payload.
#
# Usage: run_bdev_prediction_integration.sh <bdev_test_exe> <prediction_server_dir>
# Env overrides: PREDICT_PYTHON, PREDICT_PORT.
#
# Exit codes: 0 pass, 1 fail, 77 SKIP (prerequisites absent — the ctest
# SKIP_RETURN_CODE, so an ordinary build without python/models never fails).

set -u
SKIP=77

BDEV_EXE="${1:?bdev test exe path required}"
SERVER_DIR="${2:?prediction_server dir required}"
PY="${PREDICT_PYTHON:-python3}"
PORT="${PREDICT_PORT:-18082}"

skip() { echo "SKIP: $*"; exit $SKIP; }

# --- Prerequisite gating (skip, don't fail) -------------------------------
[ "$(uname -s)" = "Linux" ] || skip "not Linux (integration is Linux-only)"
[ -x "$BDEV_EXE" ] || skip "bdev test exe not found: $BDEV_EXE"
command -v "$PY" >/dev/null 2>&1 || skip "python interpreter '$PY' not found"

static_model="$SERVER_DIR/models/hdd_static_model.pkl"
sz=$(stat -c%s "$static_model" 2>/dev/null || echo 0)
[ "$sz" -gt 100000 ] || skip "model not materialized (git-lfs?): $static_model ($sz bytes)"

"$PY" -c "import fastapi, uvicorn, sklearn, lightgbm, joblib, pandas, numpy" 2>/dev/null \
  || skip "python deps missing (need fastapi/uvicorn/scikit-learn/lightgbm/joblib)"

# The bdev test links the runtime shared libs; prefer the freshly-built ones.
export LD_LIBRARY_PATH="$(cd "$(dirname "$BDEV_EXE")" && pwd):${LD_LIBRARY_PATH:-}"

# --- Launch server --------------------------------------------------------
workdir="$(mktemp -d)"
srv_log="$workdir/server.log"
( cd "$SERVER_DIR" && exec "$PY" server.py --port "$PORT" --db "$workdir/drive_history.db" ) >"$srv_log" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; rm -rf "$workdir"; }
trap cleanup EXIT

# --- Wait for health ------------------------------------------------------
healthy=0
for _ in $(seq 1 60); do
  if "$PY" - "$PORT" <<'PYEOF' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2).read()
PYEOF
  then healthy=1; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "FAIL: prediction server exited during startup"; echo "--- server log ---"; cat "$srv_log"; exit 1
  fi
  sleep 1
done
[ "$healthy" = 1 ] || { echo "FAIL: server did not become healthy in 60s"; echo "--- server log ---"; cat "$srv_log"; exit 1; }

# --- Drive the bdev integration case against the live server --------------
export CLIO_PREDICT_URL="http://127.0.0.1:$PORT/predict/auto"
"$BDEV_EXE" "bdev_failure_prediction_integration"
rc=$?
if [ "$rc" -ne 0 ]; then echo "--- server log ---"; cat "$srv_log"; fi
exit "$rc"
