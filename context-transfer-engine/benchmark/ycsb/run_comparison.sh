#!/bin/bash
# YCSB comparison driver (issue #862): CLIO Core against the best
# configurations of Redis, RocksDB and DynamoDB Local on the core workloads
# that need no scan/delete (A, B, C, F).
#
# Each configuration ("variant") is loaded ONCE, then every workload in
# $WORKLOADS runs at every client thread count in $THREADS, $REPS times,
# each run capped at $RUNTIME_S seconds. A/B/C/F only read and update
# existing keys (same record size), so they can share one load.
#
# Variants:
#   clio              CTE core on a RAM tier + the shm write-through cache
#                     (clio_ycsb_bench.yaml), SHM IPC
#   redis             one redis-server, no persistence, io-threads 1
#   redis-io4         one redis-server, no persistence, io-threads 4
#   redis-cluster4    Redis Cluster, 4 masters on this node, no persistence
#   rocksdb-stock     stock YCSB 0.17 binding (rocksdbjni 5.11.3), on tmpfs
#   rocksdb-tuned     RocksDBTunedClient (rocksdbjni 11.x): 4 GB block cache,
#                     bloom filters, hash index, no compression, WAL on, tmpfs
#   rocksdb-nowal     rocksdb-tuned with the write-ahead log disabled
#   dynamodb          DynamoDB Local, -inMemory -sharedDb, 8 GB heap
#
# Prerequisites (README.md): $DIST with the 0.17.0 redis/rocksdb/dynamodb
# bindings, dynamodb-local/, clio-binding/classes, rocksdb-tuned/{lib,classes};
# a JDK 17; redis-server ($REDIS_SERVER) and redis-cli; a build with clio_run
# and libclio_ycsb_jni.so.
#
# Usage: ./run_comparison.sh [variants...]
set -uo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DIST="${DIST:-$REPO_ROOT/external/ycsb-dist}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results}"
DATA_DIR="${DATA_DIR:-/dev/shm/ycsb-$(whoami)}"   # RocksDB data (tmpfs)
JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
REDIS_SERVER="${REDIS_SERVER:-redis-server}"
REDIS_CLI="${REDIS_CLI:-redis-cli}"
mkdir -p "$OUT_DIR"

RECORDS="${RECORDS:-1000000}"
LOAD_THREADS="${LOAD_THREADS:-16}"
THREADS="${THREADS:-1 4 16}"
WORKLOADS="${WORKLOADS:-c b a f}"
REPS="${REPS:-3}"
RUNTIME_S="${RUNTIME_S:-20}"
REDIS_PORT="${REDIS_PORT:-6399}"
DDB_PORT="${DDB_PORT:-8000}"
CSV="$OUT_DIR/results.csv"

WL_DIR="$DIST/ycsb-redis-binding-0.17.0/workloads"
LIB0="$DIST/ycsb-redis-binding-0.17.0/lib"
BASE_CP="$LIB0/core-0.17.0.jar:$LIB0/HdrHistogram-2.1.4.jar:$LIB0/htrace-core4-4.1.0-incubating.jar"
[ -f "$CSV" ] || echo "variant,workload,threads,rep,ops_per_s,read_avg_us,read_p99_us,update_avg_us,update_p99_us" > "$CSV"

jars() { ls "$1"/*.jar | tr '\n' ':'; }

# $1=variant $2=load|run $3=workload $4=threads $5=rep; uses DB_CLASS CP PROPS
run_phase() {
  local v="$1" phase="$2" wl="$3" th="$4" rep="$5" out mode
  out="$OUT_DIR/${v}_w${wl}_t${th}_r${rep}_${phase}.txt"
  local extra=(-p "maxexecutiontime=$RUNTIME_S" -p "operationcount=2000000000")
  mode="-t"
  if [ "$phase" = load ]; then
    mode="-load"; extra=(-p "operationcount=0")
  fi
  "$JAVA" -cp "$CP" ${JOPTS:-} site.ycsb.Client $mode -db "$DB_CLASS" \
    -P "$WL_DIR/workload$wl" -p "recordcount=$RECORDS" -threads "$th" \
    "${extra[@]}" "${PROPS[@]}" > "$out" 2>&1
  local t ra rp ua up
  t=$(awk -F', ' '/\[OVERALL\], Throughput/{printf "%.0f", $3}' "$out")
  ra=$(awk -F', ' '/\[READ\], AverageLatency/{printf "%.1f", $3}' "$out")
  rp=$(awk -F', ' '/\[READ\], 99thPercentileLatency/{print $3}' "$out")
  ua=$(awk -F', ' '/\[UPDATE\], AverageLatency/{printf "%.1f", $3}' "$out")
  up=$(awk -F', ' '/\[UPDATE\], 99thPercentileLatency/{print $3}' "$out")
  if [ "${t:-0}" = 0 ] || \
     grep -q "\[\(READ\|UPDATE\|INSERT\)-FAILED\]" "$out"; then
    t="FAILED"   # any failed operation invalidates the run
  fi
  if [ "$v" = clio ] && ! grep -aq "over shm cache" "$out"; then
    t="NOCACHE"  # the client fell back to the task path: not the config
  fi
  if [ "$v" = clio ]; then
    # Untimed: wait for the shm cache to write this run's puts through to
    # the core, so the next run does not start behind that backlog.
    local d0=$SECONDS
    timeout 900 "$JAVA" -cp "$CP" ${JOPTS:-} site.ycsb.db.ClioClient drain \
      > "${out%.txt}_drain.txt" 2>&1
    echo "    drain after $phase w$wl t$th r$rep: $((SECONDS - d0)) s" \
      >> "$OUT_DIR/clio_drain.log"
  fi
  echo "  $v $phase w$wl t$th r$rep: ${t:-FAILED} ops/s"
  [ "$phase" = run ] && echo "$v,$wl,$th,$rep,${t:-FAILED},$ra,$rp,$ua,$up" >> "$CSV"
}

start_clio() {
  DB_CLASS=site.ycsb.db.ClioClient
  CP="$BASE_CP:$DIST/clio-binding/classes"
  PROPS=()
  JOPTS="-Djava.library.path=$BUILD_DIR/bin"
  export CLIO_SERVER_CONF="$SCRIPT_DIR/clio_ycsb_bench.yaml" CLIO_IPC_MODE=SHM
  "$BUILD_DIR/bin/clio_run" stop --force >/dev/null 2>&1
  pkill -9 -x clio_run 2>/dev/null; sleep 1
  "$BUILD_DIR/bin/clio_run" start --fresh > "$OUT_DIR/clio_daemon.log" 2>&1 &
  CLIO_PID=$!
  # Wait for the COMPOSE to finish, not just the port: the runtime listens
  # before the shm cache pool exists (prefaulting its segment takes seconds),
  # and a client that connects in that window gets no cache keys and stays
  # on the task path for its whole life.
  local i
  for i in $(seq 120); do
    grep -aq "Compose: All .* pools created successfully" \
      "$OUT_DIR/clio_daemon.log" && break
    sleep 1
  done
  sleep 1
}

start_redis() {  # $1 = io-threads
  DB_CLASS=site.ycsb.db.RedisClient
  CP="$BASE_CP:$(jars "$DIST/ycsb-redis-binding-0.17.0/lib")"
  PROPS=(-p "redis.host=127.0.0.1" -p "redis.port=$REDIS_PORT")
  JOPTS=""
  "$REDIS_SERVER" --port "$REDIS_PORT" --daemonize yes --save '' \
    --appendonly no --io-threads "$1" --protected-mode no \
    --dir "$OUT_DIR" --logfile "$OUT_DIR/redis.log"
  sleep 1
}

start_redis_cluster() {  # $1 = masters
  DB_CLASS=site.ycsb.db.RedisClient
  CP="$BASE_CP:$(jars "$DIST/ycsb-redis-binding-0.17.0/lib")"
  PROPS=(-p "redis.host=127.0.0.1" -p "redis.port=$REDIS_PORT" -p "redis.cluster=true")
  JOPTS=""
  local i p nodes=()
  for ((i = 0; i < $1; i++)); do
    p=$((REDIS_PORT + i))
    mkdir -p "$OUT_DIR/rc$p" && rm -f "$OUT_DIR/rc$p"/nodes.conf
    "$REDIS_SERVER" --port "$p" --daemonize yes --save '' --appendonly no \
      --cluster-enabled yes --cluster-config-file nodes.conf \
      --protected-mode no --dir "$OUT_DIR/rc$p" --logfile "$OUT_DIR/rc$p/redis.log"
    nodes+=("127.0.0.1:$p")
  done
  sleep 1
  "$REDIS_CLI" --cluster create "${nodes[@]}" --cluster-replicas 0 \
    --cluster-yes > "$OUT_DIR/redis_cluster_create.log" 2>&1
  sleep 3
}

start_rocksdb() {  # $1 = stock|tuned|nowal
  rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR"
  JOPTS=""
  if [ "$1" = stock ]; then
    DB_CLASS=site.ycsb.db.rocksdb.RocksDBClient
    CP="$BASE_CP:$(jars "$DIST/ycsb-rocksdb-binding-0.17.0/lib")"
    PROPS=(-p "rocksdb.dir=$DATA_DIR")
  else
    DB_CLASS=site.ycsb.db.rocksdb.RocksDBTunedClient
    CP="$BASE_CP:$DIST/rocksdb-tuned/classes:$(jars "$DIST/rocksdb-tuned/lib")"
    PROPS=(-p "rocksdb.dir=$DATA_DIR")
    [ "$1" = nowal ] && PROPS+=(-p "rocksdb.disableWAL=true")
  fi
}

start_dynamodb() {
  DB_CLASS=site.ycsb.db.DynamoDBClient
  CP="$BASE_CP:$(jars "$DIST/ycsb-dynamodb-binding-0.17.0/lib")"
  printf 'accessKey=fake\nsecretKey=fake\n' > "$OUT_DIR/ddb_creds.properties"
  PROPS=(-p "dynamodb.endpoint=http://127.0.0.1:$DDB_PORT"
         -p "dynamodb.primaryKey=firstname"
         -p "dynamodb.awsCredentialsFile=$OUT_DIR/ddb_creds.properties"
         -p "dynamodb.connectMax=64" -p "table=usertable")
  JOPTS=""
  (cd "$DIST/dynamodb-local" && "$JAVA" -Xmx8g \
     -Djava.library.path=./DynamoDBLocal_lib -jar DynamoDBLocal.jar \
     -inMemory -sharedDb -disableTelemetry -port "$DDB_PORT" \
     > "$OUT_DIR/ddb.log" 2>&1 &)
  local i
  for i in $(seq 60); do
    ss -ltn 2>/dev/null | grep -q ":$DDB_PORT " && break
    sleep 1
  done
  PYTHONPATH="$DIST/pylib" python3 "$SCRIPT_DIR/ddb_create_table.py" "$DDB_PORT"
}

setup() {
  case "$1" in
    clio) start_clio ;;
    redis) start_redis 1 ;;
    redis-io4) start_redis 4 ;;
    redis-cluster4) start_redis_cluster 4 ;;
    rocksdb-stock) start_rocksdb stock ;;
    rocksdb-tuned) start_rocksdb tuned ;;
    rocksdb-nowal) start_rocksdb nowal ;;
    dynamodb) start_dynamodb ;;
    *) echo "unknown variant '$1'"; exit 2 ;;
  esac
}

teardown() {
  case "$1" in
    clio)
      "$BUILD_DIR/bin/clio_run" stop >/dev/null 2>&1
      sleep 2; kill -9 "$CLIO_PID" 2>/dev/null; pkill -9 -x clio_run 2>/dev/null
      sleep 1 ;;
    redis*)
      local i
      for i in 0 1 2 3; do
        "$REDIS_CLI" -p $((REDIS_PORT + i)) shutdown nosave >/dev/null 2>&1
      done
      sleep 1 ;;
    rocksdb*) rm -rf "$DATA_DIR" ;;
    dynamodb) pkill -f DynamoDBLocal.jar 2>/dev/null; sleep 2 ;;
  esac
}

bench_variant() {
  local v="$1" wl th rep
  echo "=== $v ==="
  setup "$v"
  run_phase "$v" load a "$LOAD_THREADS" 0
  # Let post-load background work finish (CLIO's write-through of the load,
  # RocksDB's flushes/compactions) and warm up, then discard the warm-up.
  sleep "${SETTLE_S:-30}"
  run_phase "$v" warmup c "$LOAD_THREADS" 0
  for wl in $WORKLOADS; do
    for th in $THREADS; do
      for rep in $(seq "$REPS"); do
        run_phase "$v" run "$wl" "$th" "$rep"
      done
    done
  done
  teardown "$v"
}

VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(clio redis redis-io4 redis-cluster4 \
  rocksdb-stock rocksdb-tuned rocksdb-nowal dynamodb)
for v in "${VARIANTS[@]}"; do
  bench_variant "$v"
done
echo "results: $CSV"
