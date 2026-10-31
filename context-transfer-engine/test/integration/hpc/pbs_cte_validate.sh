#!/bin/bash
# Validate the CTE on an N-node PBS allocation (no Docker): the repo's two
# multi-node integration suites, launched with mpiexec, one runtime per node.
#
#   1. coherence   -- test_cache_coherence on EVERY node, each against its own
#                     runtime, through the full chain cache(563) ->
#                     replication(561) -> core(512): reader-local caching,
#                     invalidation on write, concurrent / segmented / fragmented
#                     writers.
#   2. distributed -- test_core_functionality on node 0 against an N-node core
#                     pool (put/get/tags/scores/... routed across nodes).
#
#   qsub -l select=16 -v BUILD_DIR=$PWD/build/sycl,OUT=$HOME/cte16 \
#        context-transfer-engine/test/integration/hpc/pbs_cte_validate.sh
#
# Exit 0 only when both suites pass on every node.
#PBS -l place=scatter
#PBS -j oe
set -u
ulimit -c 0
: "${BUILD_DIR:?set BUILD_DIR to a configured build dir}"
OUT=${OUT:-$HOME/cte_validate}/${PBS_JOBID%%.*}
SUITES=${SUITES:-"coherence distributed"}
mkdir -p "$OUT"; cd "$OUT"
SRC=$(sed -n 's/^CMAKE_HOME_DIRECTORY:[A-Z]*=//p' "$BUILD_DIR/CMakeCache.txt")
BIN=$BUILD_DIR/bin
NN=$(sort -u "$PBS_NODEFILE" | wc -l)
sort -u "$PBS_NODEFILE" > hostfile
echo "=== CTE validation x${NN} nodes: $(cut -d. -f1 hostfile | tr '\n' ' ')"

CXX=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$BUILD_DIR/CMakeCache.txt")
case "$CXX" in /*) ;; *) CXX=$(command -v "$CXX") ;; esac
export LD_LIBRARY_PATH="$BIN:$(cd "$(dirname "$CXX")/../lib" && pwd):${LD_LIBRARY_PATH:-}"
export CLIO_BIND_ADDR=0.0.0.0 CLIO_WITH_RUNTIME=0 CTE_NUM_NODES=$NN
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0.0

# Node-local state only: hostfile and ports from this allocation, storage in
# /tmp, a private HOME so no restart log replays between suites.
mkconf() {  # <src yaml> <dst yaml>
  sed -e "s|hostfile: .*|hostfile: $OUT/hostfile|" \
      -e "s|/mnt/hdd|/tmp/cte_val_hdd|" \
      -e "s|file: /tmp/clio.log|file: /tmp/clio_val.log|" "$1" > "$2"
}

# run_suite <name> <conf> <who: all|rank0> <cmd...>
# Every rank starts its runtime, waits for all of them, runs the client
# (everywhere or on rank 0), then lingers until every client is done so no
# node tears down while a peer still reads from it.
run_suite() {
  local name=$1 conf=$2 who=$3; shift 3
  rm -f "$OUT/$name".ready.* "$OUT/$name".done.* "$OUT/$name".peer.*
  CLIO_SERVER_CONF=$conf mpiexec -n "$NN" -ppn 1 --cpu-bind none --envall \
    bash -c '
      r=${PALS_RANKID:-$PMI_RANK}; export NODE_ID=$((r+1))
      export HOME=/tmp/cte_val_home_$$; mkdir -p "$HOME"
      pkill -u "$USER" -x clio_run 2>/dev/null; rm -f /dev/shm/clio_* 2>/dev/null
      "'"$BIN"'"/clio_run runtime start --fresh > "'"$OUT/$name"'.runtime.$r.log" 2>&1 &
      rt=$!
      sleep 10; touch "'"$OUT/$name"'.ready.$r"
      for i in $(seq 1 120); do [ $(ls "'"$OUT/$name"'".ready.* 2>/dev/null | wc -l) -ge '"$NN"' ] && break; sleep 1; done
      rc=0
      if [ "'"$who"'" = all ] || [ "$r" = 0 ]; then
        '"$*"' > "'"$OUT/$name"'.client.$r.log" 2>&1; rc=$?
      fi
      if [ "'"$who"'" = all ] || [ "$r" = 0 ]; then echo $rc > "'"$OUT/$name"'.done.$r"; else touch "'"$OUT/$name"'.peer.$r"; fi
      want='"$NN"'; [ "'"$who"'" = rank0 ] && want=1
      for i in $(seq 1 1800); do [ $(ls "'"$OUT/$name"'".done.* 2>/dev/null | wc -l) -ge $want ] && break; sleep 1; done
      sleep 5; kill $rt 2>/dev/null; wait $rt 2>/dev/null
      exit $rc'
  local fails=0
  for f in "$OUT/$name".done.*; do [ "$(cat "$f")" = 0 ] || fails=$((fails+1)); done
  local ran=$(ls "$OUT/$name".done.* 2>/dev/null | wc -l)
  if [ "$fails" = 0 ] && [ "$ran" -ge 1 ]; then echo "PASS $name ($ran client(s))"
  else echo "FAIL $name ($fails of $ran client(s) failed)"; grep -h -a -E '\[FAIL\]|REQUIRE|FAILED' "$OUT/$name".client.*.log 2>/dev/null | sort | uniq -c | head -10; fi
  return $fails
}

status=0
for s in $SUITES; do
  case $s in
    coherence)
      mkconf "$SRC/context-transfer-engine/test/integration/coherence/clio_config.yaml" coherence.yaml
      export CLIO_CTE_POOL=563.0   # the client binds the chain top
      run_suite coherence "$OUT/coherence.yaml" all \
        "$BIN/test_cache_coherence" || status=1
      unset CLIO_CTE_POOL ;;
    distributed)
      mkconf "$SRC/context-transfer-engine/test/integration/distributed/clio_config.yaml" distributed.yaml
      run_suite distributed "$OUT/distributed.yaml" rank0 \
        "$BIN/test_core_functionality" || status=1 ;;
  esac
done
echo "=== CTE validation x${NN}: $([ $status = 0 ] && echo PASS || echo FAIL)"
exit $status
