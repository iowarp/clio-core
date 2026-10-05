#!/bin/bash
# Issue #624 reproducer: clio_run exit 3 in SWIM dead-node recovery (Windows).
#
# Runs LIVE runtimes on 127.0.0.1 with distinct base ports, sharing a hostfile
# that also lists unreachable peers (DEAD_PEERS, default 10.255.255.1). Node 0 is live,
# so it is the SWIM leader; with LIVE >= 2 the dead peer is a minority, so
# self-fencing (bad*2 > other) does not trigger and node 0 runs
# TriggerRecovery -> RecoverContainers. LIVE=1 is the issue's "node started
# alone, peer never reachable" case (that node self-fences).
# DEAD_FIRST=1 makes the dead peer node 0 (see below).
#
# Usage: CLIO_RUN=<path/to/clio_run[.exe]> [LIVE=3] [SETTLE=45] ./run_repro.sh
# Each node's log is <workdir>/node<i>.log; exit codes are printed at the end.
# Exits 1 if any runtime died before the end of the run (#624), if any node
# logged a dead peer as REJOINED (#1171: the DEAD_PEERS never answer), if a dead
# peer was never declared dead (#1178), or if the survivors were not fenced yet
# recovery created no containers (#1170). Else 0.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIO_RUN=${CLIO_RUN:?set CLIO_RUN to the clio_run binary}
LIVE=${LIVE:-3}
SETTLE=${SETTLE:-45}            # seconds to keep watching after start
DEAD_PEERS=${DEAD_PEERS:-10.255.255.1}  # space-separated; each gets its own port
BASE_PORT=${BASE_PORT:-9500}    # node i uses BASE_PORT + 10*i (+1, +3 too)
WORK=${WORK:-$(mktemp -d)}
mkdir -p "$WORK"

# DEAD_FIRST=1 lists the dead peer as node 0 (as in the issue, where node 0
# died and node 2 recovered it); the lowest live node is then the leader.
DEAD_FIRST=${DEAD_FIRST:-0}
HOSTFILE="$WORK/hostfile"
: > "$HOSTFILE"
add_dead() {
  local j=$LIVE
  for p in $DEAD_PEERS; do
    echo "${p}:$((BASE_PORT + 10 * j))" >> "$HOSTFILE"; j=$((j + 1))
  done
}
if [ "$DEAD_FIRST" = 1 ]; then add_dead; fi
for ((i = 0; i < LIVE; i++)); do
  echo "127.0.0.1:$((BASE_PORT + 10 * i))" >> "$HOSTFILE"
done
if [ "$DEAD_FIRST" != 1 ]; then add_dead; fi

# Native Windows binaries need Windows paths in the config.
to_native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }

pids=()
for ((i = 0; i < LIVE; i++)); do
  port=$((BASE_PORT + 10 * i))
  conf="$WORK/node$i.yaml"
  sed -e "s|@PORT@|$port|" -e "s|@HOSTFILE@|$(to_native "$HOSTFILE")|" \
      "$SCRIPT_DIR/clio_conf.yaml.in" > "$conf"
  mkdir -p "$WORK/storage$i"
  CLIO_SERVER_CONF="$(to_native "$conf")" CLIO_PORT=$port \
  CLIO_STORAGE_ROOT="$(to_native "$WORK/storage$i")" \
  CTP_LOG_LEVEL=${CTP_LOG_LEVEL:-info} \
    "$CLIO_RUN" start --fresh > "$WORK/node$i.log" 2>&1 &
  pids+=($!)
  echo "node $i: port $port pid ${pids[$i]} log $WORK/node$i.log"
done

echo "watching for ${SETTLE}s ..."
deadline=$((SECONDS + SETTLE))
while ((SECONDS < deadline)); do
  sleep 1
done

died=0
for ((i = 0; i < LIVE; i++)); do
  if kill -0 "${pids[$i]}" 2>/dev/null; then
    echo "node $i: ALIVE after ${SETTLE}s"
  else
    wait "${pids[$i]}"; rc=$?
    echo "node $i: EXITED rc=$rc"
    died=1
  fi
  grep -E "confirmed dead|Self-fencing|Recovery:|marked as DEAD" \
    "$WORK/node$i.log" | sed "s/^/  node$i| /" | head -20
done

# Behavioural checks on the logs (see the header).
logs=("$WORK"/node*.log)
if grep -q "REJOINED" "${logs[@]}"; then
  echo "FAIL: an unreachable peer was logged as REJOINED (#1171)"
  died=1
fi
for p in $DEAD_PEERS; do
  if ! grep -q "($p) marked as DEAD" "${logs[@]}"; then
    echo "FAIL: dead peer $p was never declared dead (#1178)"
    died=1
  fi
done
if ! grep -q "Self-fencing" "${logs[@]}" &&
   ! grep -q "Recovery: Creating container" "${logs[@]}"; then
  echo "FAIL: not self-fenced, yet recovery created no containers (#1170)"
  died=1
fi

for ((i = 0; i < LIVE; i++)); do
  kill "${pids[$i]}" 2>/dev/null
done
wait 2>/dev/null
echo "logs: $WORK"
exit $died
