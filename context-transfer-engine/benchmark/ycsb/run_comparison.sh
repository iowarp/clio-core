#!/bin/bash
# YCSB comparison driver (issue #862): clio vs redis vs rocksdb vs
# dynamodb-local on the core workloads that need no scan/delete (A, B, C, F).
#
# Prerequisites (the devcontainer provisions all of these):
#   - external/ycsb-dist/ with the 0.17.0 redis/rocksdb/dynamodb binding
#     tarballs extracted and clio-binding/classes compiled (see README.md)
#   - external/ycsb-dist/dynamodb-local/DynamoDBLocal.jar
#   - redis-server on PATH; a JDK (javac/java)
#   - build directory with clio_run + libclio_ycsb_jni.so (BUILD_DIR below)
#
# Usage: ./run_comparison.sh [dbs...]   (default: clio redis rocksdb dynamodb)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DIST="$REPO_ROOT/external/ycsb-dist"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-cpu}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$OUT_DIR"

RECORDS="${RECORDS:-100000}"
OPS="${OPS:-100000}"
THREADS="${THREADS:-4}"
WORKLOADS="${WORKLOADS:-a b c f}"
REDIS_PORT="${REDIS_PORT:-6399}"
DDB_PORT="${DDB_PORT:-8000}"

WL_DIR="$DIST/ycsb-redis-binding-0.17.0/workloads"
CORE_JAR="$DIST/ycsb-redis-binding-0.17.0/lib/core-0.17.0.jar"
HDR_JAR="$DIST/ycsb-redis-binding-0.17.0/lib/HdrHistogram-2.1.4.jar"
HTRACE_JAR="$DIST/ycsb-redis-binding-0.17.0/lib/htrace-core4-4.1.0-incubating.jar"
BASE_CP="$CORE_JAR:$HDR_JAR:$HTRACE_JAR"

COMMON_PROPS=(-p "recordcount=$RECORDS" -p "operationcount=$OPS" -threads "$THREADS")

run_phase() {  # $1=db-label $2=load|run $3=workload-letter $4=classpath $5...=extra props
  local label="$1" phase="$2" wl="$3" cp="$4"; shift 4
  local out="$OUT_DIR/${label}_workload${wl}_${phase}.txt"
  local mode="-t"; [ "$phase" = "load" ] && mode="-load"
  java -cp "$cp" ${CLIO_JAVA_OPTS:-} site.ycsb.Client $mode \
    -db "$DB_CLASS" -P "$WL_DIR/workload$wl" "${COMMON_PROPS[@]}" "$@" \
    > "$out" 2>&1
  local tput
  tput=$(grep -m1 "\[OVERALL\], Throughput" "$out" | awk -F', ' '{printf "%.0f", $3}')
  echo "  $label workload$wl $phase: ${tput:-FAILED} ops/sec  ($out)"
}

bench_db() {  # $1=db-label; per-db classpath/class/props set below
  local label="$1"
  echo "=== $label ==="
  for wl in $WORKLOADS; do
    # workloads share the loaded dataset only within a db+workload pair; each
    # workload gets a fresh load so RMW/update mixes start from clean state.
    setup_db "$label" "$wl"
    run_phase "$label" load "$wl" "$CP" "${PROPS[@]}"
    run_phase "$label" run  "$wl" "$CP" "${PROPS[@]}"
    teardown_db "$label"
  done
}

setup_db() {
  local label="$1" wl="$2"
  case "$label" in
    clio)
      DB_CLASS=site.ycsb.db.ClioClient
      CP="$BASE_CP:$DIST/clio-binding/classes"
      PROPS=()
      export CLIO_JAVA_OPTS="-Djava.library.path=$BUILD_DIR/bin"
      pkill -9 -x clio_run 2>/dev/null; sleep 1
      # The runtime binds 9413 without SO_REUSEADDR, so a fresh daemon can
      # lose to the previous one's TIME_WAIT; retry until it actually listens.
      local attempt ok=0
      for attempt in 1 2 3 4 5; do
        rm -rf /tmp/clio_$(whoami)/* 2>/dev/null
        LD_LIBRARY_PATH="$BUILD_DIR/bin" "$BUILD_DIR/bin/clio_run" runtime start \
          >> "$OUT_DIR/clio_daemon_${wl}.log" 2>&1 &
        CLIO_PID=$!
        sleep 3
        if kill -0 "$CLIO_PID" 2>/dev/null && \
           ss -ltn 2>/dev/null | grep -q ':9413 '; then
          ok=1; break
        fi
        kill -9 "$CLIO_PID" 2>/dev/null; sleep 5
      done
      [ "$ok" = 1 ] || echo "  WARNING: clio daemon failed to start (workload $wl)"
      ;;
    redis)
      DB_CLASS=site.ycsb.db.RedisClient
      CP="$BASE_CP:$(ls "$DIST"/ycsb-redis-binding-0.17.0/lib/*.jar | tr '\n' ':')"
      PROPS=(-p "redis.host=127.0.0.1" -p "redis.port=$REDIS_PORT")
      redis-server --port "$REDIS_PORT" --daemonize yes --save '' --appendonly no
      sleep 1
      ;;
    rocksdb)
      DB_CLASS=site.ycsb.db.rocksdb.RocksDBClient
      CP="$BASE_CP:$(ls "$DIST"/ycsb-rocksdb-binding-0.17.0/lib/*.jar | tr '\n' ':')"
      ROCKS_DIR=$(mktemp -d /tmp/ycsb-rocksdb.XXXXXX)
      PROPS=(-p "rocksdb.dir=$ROCKS_DIR")
      ;;
    dynamodb)
      DB_CLASS=site.ycsb.db.DynamoDBClient
      CP="$BASE_CP:$(ls "$DIST"/ycsb-dynamodb-binding-0.17.0/lib/*.jar | tr '\n' ':')"
      PROPS=(-p "dynamodb.endpoint=http://127.0.0.1:$DDB_PORT"
             -p "dynamodb.primaryKey=firstname"
             -p "dynamodb.awsCredentialsFile=$OUT_DIR/ddb_creds.properties"
             -p "table=usertable")
      printf 'accessKey=fake\nsecretKey=fake\n' > "$OUT_DIR/ddb_creds.properties"
      (cd "$DIST/dynamodb-local" && java -jar DynamoDBLocal.jar -inMemory \
        -port "$DDB_PORT" > "$OUT_DIR/ddb_${wl}.log" 2>&1 &)
      sleep 4
      python3 "$SCRIPT_DIR/ddb_create_table.py" "$DDB_PORT"
      ;;
  esac
}

teardown_db() {
  case "$1" in
    clio)
      kill "$CLIO_PID" 2>/dev/null; pkill -9 -x clio_run 2>/dev/null
      rm -rf /tmp/clio_$(whoami)/* 2>/dev/null; sleep 1 ;;
    redis)
      redis-cli -p "$REDIS_PORT" shutdown nosave 2>/dev/null; sleep 1 ;;
    rocksdb)
      rm -rf "$ROCKS_DIR" ;;
    dynamodb)
      pkill -f DynamoDBLocal.jar 2>/dev/null; sleep 1 ;;
  esac
}

DBS=("$@")
if [ ${#DBS[@]} -eq 0 ]; then
  DBS=(clio redis rocksdb dynamodb)
fi
for db in "${DBS[@]}"; do
  case "$db" in
    clio|redis|rocksdb|dynamodb) bench_db "$db" ;;
    *) echo "unknown db '$db' (expected clio|redis|rocksdb|dynamodb)"; exit 2 ;;
  esac
done

echo
echo "=== Summary (run-phase throughput, ops/sec) ==="
for f in "$OUT_DIR"/*_run.txt; do
  b=$(basename "$f" _run.txt)
  t=$(grep -m1 "\[OVERALL\], Throughput" "$f" | awk -F', ' '{printf "%.0f", $3}')
  echo "$b: ${t:-FAILED}"
done
