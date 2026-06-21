#!/bin/bash
# Comparison: IOWarp TCP / IPC / SHM transports vs Redis
# Sweeps thread/client counts: 1, 4, 8, 16
#
# IOWarp transport is selected via CLIO_IPC_MODE={TCP,IPC,SHM} on both
# the runtime and the client; the clio_cte_bench Put/Get/PutGet workload
# is the IOWarp counterpart to redis-benchmark's SET/GET/SETGET.

set -u  # don't `set -e` — bench errors on one config shouldn't kill the rest
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BIN="${BUILD_BIN:-/workspace/build/bin}"
CTE_BENCH="$BUILD_BIN/clio_cte_bench"
CLIO="$BUILD_BIN/clio_run"
CTE_CONFIG="${CTE_CONFIG:-$SCRIPT_DIR/cte_config_ram.yaml}"
REDIS_PORT="${REDIS_PORT:-6390}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/transport_compare}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/run.log"
: > "$LOG"

# Workload
IO_SIZE="${IO_SIZE:-4k}"      # match redis-benchmark default-ish small payload
IO_COUNT="${IO_COUNT:-50000}" # per-thread / per-redis-client op count
DEPTH="${DEPTH:-4}"           # async pipeline depth (IOWarp only)
THREAD_COUNTS=(1 4 8 16)
OPS=(Put Get)                 # Put = redis SET, Get = redis GET

CSV="$RESULTS_DIR/results.csv"
echo "transport,op,threads,total_ops,total_bytes,aggregate_MBps,per_thread_avg_MBps" > "$CSV"

log()  { echo "[compare] $*" | tee -a "$LOG"; }
fail() { echo "[compare][ERROR] $*" | tee -a "$LOG"; }

# ----- IOWarp setup --------------------------------------------------------
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$BUILD_BIN"
export CLIO_REPO_PATH="${CLIO_REPO_PATH:-$BUILD_BIN}"
export CLIO_SERVER_CONF="$CTE_CONFIG"

# Kill any leftover servers
pkill -9 -f "$CLIO runtime start" >/dev/null 2>&1 || true
pkill -9 clio_run >/dev/null 2>&1 || true
sleep 1

start_clio_run() {
  local mode="$1"
  log "Starting Clio runtime (CLIO_IPC_MODE=$mode)"
  CLIO_IPC_MODE="$mode" \
    "$CLIO" runtime start > "$RESULTS_DIR/clio_${mode}.log" 2>&1 &
  CLIO_PID=$!
  # Wait for server to be listening (matches "Successfully started local server"
  # in the clio_run log)
  for i in $(seq 1 30); do
    if grep -q "Successfully started local server" \
         "$RESULTS_DIR/clio_${mode}.log" 2>/dev/null; then
      log "Clio up (pid=$CLIO_PID, mode=$mode)"
      return 0
    fi
    sleep 0.5
  done
  fail "Clio ($mode) did not start within 15s"
  return 1
}

stop_clio_run() {
  if [ -n "${CLIO_PID:-}" ]; then
    kill "$CLIO_PID" 2>/dev/null || true
    wait "$CLIO_PID" 2>/dev/null || true
    CLIO_PID=""
  fi
  pkill -9 -f "$CLIO runtime start" >/dev/null 2>&1 || true
  pkill -9 clio_run >/dev/null 2>&1 || true
  sleep 1
}

run_iowarp() {
  local mode="$1"  # TCP | IPC | SHM
  start_clio_run "$mode" || return
  for op in "${OPS[@]}"; do
    for nt in "${THREAD_COUNTS[@]}"; do
      log "  IOWarp $mode op=$op threads=$nt"
      out=$(CLIO_IPC_MODE="$mode" CLIO_WITH_RUNTIME=0 \
        "$CTE_BENCH" "$op" "$nt" "$DEPTH" "$IO_SIZE" "$IO_COUNT" \
        2>"$RESULTS_DIR/bench_${mode}_${op}_t${nt}.log")
      agg=$(echo "$out" | grep "Aggregate bandwidth" | head -1 \
              | sed -E 's/.*Aggregate bandwidth: ([0-9.]+) MB\/s.*/\1/')
      per=$(echo "$out" | grep "Bandwidth per thread (avg)" | head -1 \
              | sed -E 's/.*\(avg\): ([0-9.]+) MB\/s.*/\1/')
      io_bytes=$(case "$IO_SIZE" in
        *k|*K) echo $(( ${IO_SIZE%[kK]} * 1024 )) ;;
        *m|*M) echo $(( ${IO_SIZE%[mM]} * 1024 * 1024 )) ;;
        *)     echo "$IO_SIZE" ;;
      esac)
      total_ops=$((nt * IO_COUNT))
      total_bytes=$((total_ops * io_bytes))
      echo "iowarp-${mode},${op},${nt},${total_ops},${total_bytes},${agg:-NA},${per:-NA}" >> "$CSV"
    done
  done
  stop_clio_run
}

# ----- Redis setup ---------------------------------------------------------
REDIS_PID_FILE="$RESULTS_DIR/redis.pid"
REDIS_LOG="$RESULTS_DIR/redis.log"

start_redis() {
  log "Starting Redis on port $REDIS_PORT"
  pkill -9 -f "redis-server.*$REDIS_PORT" >/dev/null 2>&1 || true
  sleep 0.5
  redis-server --port "$REDIS_PORT" --daemonize yes \
    --pidfile "$REDIS_PID_FILE" --logfile "$REDIS_LOG" \
    --save "" --appendonly no
  for i in $(seq 1 20); do
    if redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; then
      log "Redis up"
      return 0
    fi
    sleep 0.5
  done
  fail "Redis did not start"
  return 1
}

stop_redis() {
  redis-cli -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true
  pkill -9 -f "redis-server.*$REDIS_PORT" >/dev/null 2>&1 || true
  rm -f "$REDIS_PID_FILE"
}

run_redis() {
  start_redis || return
  local io_bytes
  case "$IO_SIZE" in
    *k|*K) io_bytes=$(( ${IO_SIZE%[kK]} * 1024 )) ;;
    *m|*M) io_bytes=$(( ${IO_SIZE%[mM]} * 1024 * 1024 )) ;;
    *)     io_bytes="$IO_SIZE" ;;
  esac
  for op in set get; do
    local big_op
    [ "$op" = "set" ] && big_op="Put" || big_op="Get"
    if [ "$op" = "get" ]; then
      # Pre-populate so GET has hits
      redis-benchmark -p "$REDIS_PORT" -t set -c 16 \
        -n "$((IO_COUNT * 16))" -d "$io_bytes" -q >/dev/null 2>&1
    fi
    for nt in "${THREAD_COUNTS[@]}"; do
      log "  Redis $op clients=$nt"
      out=$(redis-benchmark -p "$REDIS_PORT" -t "$op" -c "$nt" \
        -n "$IO_COUNT" -d "$io_bytes" --csv 2>/dev/null | tail -1)
      # CSV format: "<test>","<rps>",...
      rps=$(echo "$out" | cut -d',' -f2 | tr -d '"')
      if [ -z "$rps" ] || [ "$rps" = "NA" ]; then
        agg=NA
      else
        agg=$(awk "BEGIN {printf \"%.2f\", ($rps * $io_bytes) / (1024 * 1024)}")
      fi
      total_ops="$IO_COUNT"
      total_bytes=$((total_ops * io_bytes))
      echo "redis,${big_op},${nt},${total_ops},${total_bytes},${agg},${agg}" >> "$CSV"
    done
  done
  stop_redis
}

# ----- Drive ---------------------------------------------------------------
log "=================================================="
log " IO Size:   $IO_SIZE"
log " IO Count:  $IO_COUNT per thread/client"
log " Depth:     $DEPTH"
log " Threads:   ${THREAD_COUNTS[*]}"
log " Ops:       ${OPS[*]} (Redis equivalents: SET, GET)"
log " Results:   $CSV"
log "=================================================="

run_iowarp TCP
run_iowarp IPC
run_iowarp SHM
run_redis

log ""
log "=== Aggregate MB/s by config ==="
column -t -s, "$CSV"
