#!/usr/bin/env bash
# run_e2e_gpu_test.sh
# End-to-end test: llama-server + IOWarp KV cache with GPU inference
#
# Verifies that KV cache save/restore works when model weights are on GPU.
# GPU build required: cmake with -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
#
# Usage (from inside the container):
#   /bin/bash /workspace/run_e2e_gpu_test.sh [BUILD_DIR] [MODEL] [CHI_BUILD_DIR]
#
# Defaults:
#   BUILD_DIR     = /workspace/build_llm_gpu
#   MODEL         = auto-detected from /workspace/models/
#   CHI_BUILD_DIR = /workspace/build

set -e

BUILD_DIR="${1:-/workspace/build_llm_gpu}"
MODEL="${2:-}"
CHI_BUILD_DIR="${3:-/workspace/build}"
COMPOSE_CFG=/workspace/cte_kvcache_compose.yaml
PORT=8089
LOG=/tmp/llama_server_gpu_e2e.log

die() { echo "ERROR: $*" >&2; exit 1; }

export LD_LIBRARY_PATH="$CHI_BUILD_DIR/bin:$BUILD_DIR/bin:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"

[ -f "$BUILD_DIR/bin/llama-server" ]           || die "llama-server not found in $BUILD_DIR/bin"
[ -f "$BUILD_DIR/bin/libwrp_llm_kvcache.so" ]  || die "libwrp_llm_kvcache.so not found in $BUILD_DIR/bin"
[ -f "$CHI_BUILD_DIR/bin/chimaera_start_runtime" ] || die "chimaera_start_runtime not found in $CHI_BUILD_DIR/bin"
[ -f "$COMPOSE_CFG" ]                           || die "CTE compose YAML not found: $COMPOSE_CFG"

if [ -z "$MODEL" ]; then
    MODEL=$(find /workspace/models -name '*.gguf' -not -name '*vocab*' -size +10M 2>/dev/null | head -1)
    [ -n "$MODEL" ] || die "No model found in /workspace/models/"
    echo "Auto-detected model: $MODEL"
fi
[ -f "$MODEL" ] || die "Model file not found: $MODEL"

# ─── CTE runtime ─────────────────────────────────────────────────────────────
echo "=== Starting Chimaera CTE runtime ==="
export CHI_SERVER_CONF=$COMPOSE_CFG
"$CHI_BUILD_DIR/bin/chimaera_start_runtime" > /tmp/chimaera_gpu_e2e.log 2>&1 &
CTE_PID=$!
sleep 4
"$CHI_BUILD_DIR/bin/chimaera_compose" $COMPOSE_CFG >> /tmp/chimaera_gpu_e2e.log 2>&1
echo "CTE runtime PID=$CTE_PID"

cleanup() {
    echo "=== Stopping server and CTE ==="
    kill $SERVER_PID 2>/dev/null || true
    "$CHI_BUILD_DIR/bin/chimaera_stop_runtime" 2>/dev/null || true
    wait $CTE_PID 2>/dev/null || true
}
trap cleanup EXIT

# ─── llama-server with GPU ────────────────────────────────────────────────────
echo "=== Starting GPU llama-server (port $PORT, n-gpu-layers=999) ==="
LD_LIBRARY_PATH="$BUILD_DIR/bin:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH" \
    "$BUILD_DIR/bin/llama-server" \
    --model "$MODEL" \
    --port  $PORT \
    --ctx-size 2048 \
    --n-predict 32 \
    --n-gpu-layers 999 \
    --cache-ram 0 \
    --slot-prompt-similarity 0.0 \
    --log-prefix \
    --verbose \
    > $LOG 2>&1 &
SERVER_PID=$!

echo -n "Waiting for server"
for i in $(seq 1 90); do
    sleep 1
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo " ready."
        break
    fi
    echo -n "."
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo " DIED. Last log:"
        tail -30 $LOG
        exit 1
    fi
done

SHARED_PROMPT="You are a helpful assistant. Answer concisely."
MSG_A="What is 2+2? Answer briefly."
MSG_B="What is the capital of France? Answer briefly."

# ─── Request 1: cold — GPU prefill A, save KV to CTE ─────────────────────────
echo ""
echo "=== Request 1 (cold A — GPU prefill + IOWarp save) ==="
curl -sf http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"system\",\"content\":\"$SHARED_PROMPT\"},{\"role\":\"user\",\"content\":\"$MSG_A\"}],\"max_tokens\":32}" \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1 | sed 's/^/Response: /'

sleep 1

# ─── Request 2: evict — different prompt, evicts A ───────────────────────────
echo ""
echo "=== Request 2 (evict A — different prompt) ==="
curl -sf http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"system\",\"content\":\"$SHARED_PROMPT\"},{\"role\":\"user\",\"content\":\"$MSG_B\"}],\"max_tokens\":32}" \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1 | sed 's/^/Response: /'

sleep 1

# ─── Request 3: restore — same as R1, IOWarp restore from CTE ────────────────
echo ""
echo "=== Request 3 (restore A — IOWarp restore from CTE, skip GPU prefill) ==="
curl -sf http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"system\",\"content\":\"$SHARED_PROMPT\"},{\"role\":\"user\",\"content\":\"$MSG_A\"}],\"max_tokens\":32}" \
    2>/dev/null | grep -o '"content":"[^"]*"' | head -1 | sed 's/^/Response: /'

sleep 1

# ─── GPU memory snapshot ──────────────────────────────────────────────────────
echo ""
echo "=== GPU memory usage ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

# ─── results ─────────────────────────────────────────────────────────────────
echo ""
echo "=== IOWarp KV cache log lines ==="
grep -i "iowarp" $LOG || echo "(none found — check $LOG for details)"

echo ""
echo "=== Summary ==="
SAVED=$(grep -c "IOWarp: saved KV" $LOG 2>/dev/null || true)
RESTORED=$(grep -c "IOWarp: restored KV" $LOG 2>/dev/null || true)
INIT_OK=$(grep -c "IOWarp: KV cache manager initialized" $LOG 2>/dev/null || true)

echo "  IOWarp manager : $([ "$INIT_OK" -gt 0 ] && echo INITIALIZED || echo NOT INITIALIZED)"
echo "  KV saves       : $SAVED"
echo "  KV restores    : $RESTORED"

if [ "$SAVED" -gt 0 ] && [ "$RESTORED" -gt 0 ]; then
    echo ""
    echo "GPU PASS: IOWarp KV cache works with GPU inference (RTX 5060, sm_120)."
elif [ "$SAVED" -gt 0 ]; then
    echo ""
    echo "PARTIAL: saves but no restores — check slot-prompt-similarity or cache-ram flags."
else
    echo ""
    echo "FAIL: no IOWarp KV activity. Check $LOG."
fi
