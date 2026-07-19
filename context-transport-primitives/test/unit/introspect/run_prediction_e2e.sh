#!/usr/bin/env bash
# End-to-end driver for the drive-failure prediction path (Linux only).
#
# Starts the real Python FastAPI prediction server with the trained models,
# waits for it to become healthy, then runs the C++ checker against it via
# CLIO_PREDICT_URL. Cleans up the server on exit.
#
# Usage: run_prediction_e2e.sh <checker_exe> <prediction_server_dir>
# Env overrides: PREDICT_PYTHON (python interpreter), PREDICT_PORT.
#
# Exit codes: 0 pass, 1 fail, 77 SKIP (prerequisites absent — registered as the
# ctest SKIP_RETURN_CODE so a normal build without python/models doesn't fail).

set -u
SKIP=77

CHECKER="${1:?checker exe path required}"
SERVER_DIR="${2:?prediction_server dir required}"
PY="${PREDICT_PYTHON:-python3}"
PORT="${PREDICT_PORT:-18080}"

skip() { echo "SKIP: $*"; exit $SKIP; }

# --- Prerequisite gating (skip, don't fail) -------------------------------
[ "$(uname -s)" = "Linux" ] || skip "not Linux (E2E is Linux-only)"
[ -x "$CHECKER" ] || skip "checker exe not found: $CHECKER"
command -v "$PY" >/dev/null 2>&1 || skip "python interpreter '$PY' not found"

# Models must be materialized, not Git-LFS pointer stubs (~130 bytes).
static_model="$SERVER_DIR/models/hdd_static_model.pkl"
sz=$(stat -c%s "$static_model" 2>/dev/null || echo 0)
[ "$sz" -gt 100000 ] || skip "model not materialized (is git-lfs installed/pulled?): $static_model ($sz bytes)"

# Server python dependencies must be importable.
"$PY" -c "import fastapi, uvicorn, sklearn, lightgbm, joblib, pandas, numpy" 2>/dev/null \
  || skip "python deps missing (need fastapi/uvicorn/scikit-learn/lightgbm/joblib)"

# The checker links clio_ctp_host; make sure the freshly-built lib is found
# ahead of any stale installed copy.
export LD_LIBRARY_PATH="$(cd "$(dirname "$CHECKER")" && pwd):${LD_LIBRARY_PATH:-}"

# --- Launch server --------------------------------------------------------
workdir="$(mktemp -d)"
srv_log="$workdir/server.log"
cd "$SERVER_DIR" || { echo "FAIL: cannot cd to $SERVER_DIR"; exit 1; }

"$PY" server.py --port "$PORT" --db "$workdir/drive_history.db" >"$srv_log" 2>&1 &
SRV=$!
cleanup() {
  kill "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null
  rm -rf "$workdir"
}
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

# --- Run the checker against the live server ------------------------------
export CLIO_PREDICT_URL="http://127.0.0.1:$PORT/predict/auto"
"$CHECKER"
rc=$?
if [ "$rc" -ne 0 ]; then echo "--- server log ---"; cat "$srv_log"; fi
exit "$rc"
