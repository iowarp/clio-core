# shellcheck shell=bash
# Host-side waiting and diagnostics for the docker-compose integration suites.
# SOURCE this file from a run_tests.sh; it defines functions only.
#
# Why this exists. The suites used to `sleep` a guessed number of seconds for
# runtimes to come up and then `docker wait` / `docker exec` with no bound. A
# stuck node therefore turned into a GitHub `timeout-minutes` kill: the step
# dies before the script prints a single container log, and the next run
# passes, so nobody learns what hung (CTE distributed test, 2026-11-18: 25 min
# of silence after "Tests are executing"). The rules enforced here:
#
#   1. ONE deadline per suite (cw_init), sized below the workflow step's
#      timeout-minutes, so the script always outlives its own waits and gets
#      to print diagnostics and tear the cluster down.
#   2. Readiness is a real signal -- `clio_run status` exit 0 (a heartbeat task
#      answered by the runtime) -- either polled with docker exec
#      (cw_wait_runtimes) or published by the node as <sync>/<node>.ready via
#      CI/cluster_node.sh (cw_wait_ready_files).
#   3. Every timeout and every non-zero exit dumps state for EVERY node
#      (cw_dump): container status, the last CW_LOG_TAIL log lines, the live
#      process table and best-effort thread backtraces.
#
# Typical use:
#   source "$REPO_ROOT/CI/compose_wait.sh"
#   cw_init 1200 node1 node2 node3 node4    # budget (s), container names
#   cw_reset_sync "$REPO_ROOT/build/cluster_sync/my_suite"
#   docker compose up -d
#   cw_wait_ready_files 4 || fail
#   cw_wait_containers node1 || fail
#
# Environment overrides:
#   CW_BUDGET_SEC  replaces the budget passed to cw_init
#   CW_READY_CAP   cap on readiness waits (default 180 s)
#   CW_LOG_TAIL    log lines per node in a dump (default 300)
#   CW_NO_BT=1     skip the gdb backtrace attempt in cw_dump
#
# Safe to source under `set -e` / `set -o pipefail`: diagnostics never abort
# the caller, and wait functions report failure only through their status.
#   CW_STATUS_EXEC_OPTS  extra `docker exec` options for cw_wait_runtimes

CW_DEADLINE=0
CW_NODES=()
CW_SYNC_DIR=""
CW_READY_CAP="${CW_READY_CAP:-180}"
CW_LOG_TAIL="${CW_LOG_TAIL:-300}"
declare -gA CW_RC=()

# Print a timestamped harness message.
# @param $* message
cw_log() {
  echo "[compose_wait $(date +%H:%M:%S)] $*"
}

# Start the suite's single deadline and record the containers to diagnose.
# @param $1 budget in seconds; keep it below the workflow step limit by at
#           least the time a dump + teardown takes (~2 min)
# @param $@ remaining args: container names belonging to the suite
cw_init() {
  local budget="${CW_BUDGET_SEC:-$1}"
  shift
  CW_DEADLINE=$(( $(date +%s) + budget ))
  CW_NODES=("$@")
  CW_RC=()
  cw_log "deadline set: ${budget}s from now; nodes: ${CW_NODES[*]}"
}

# Seconds left before the suite deadline (never negative).
cw_remaining() {
  local left=$(( CW_DEADLINE - $(date +%s) ))
  if [ "$left" -lt 0 ]; then left=0; fi
  echo "$left"
}

# Smaller of a cap and the time remaining before the deadline.
# @param $1 cap in seconds
cw_cap() {
  local left
  left=$(cw_remaining)
  if [ "$1" -lt "$left" ]; then echo "$1"; else echo "$left"; fi
}

# Empty and (re)create the shared sync directory nodes write markers to.
# It must be on the bind mount every container sees (e.g. <repo>/build/...).
# @param $1 host-side path of the sync directory
cw_reset_sync() {
  CW_SYNC_DIR="$1"
  rm -rf "$CW_SYNC_DIR"
  mkdir -p "$CW_SYNC_DIR"
  chmod 1777 "$CW_SYNC_DIR" 2>/dev/null || true
}

# Print a one-line state summary for a container.
# @param $1 container name
cw_state() {
  docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}' \
    "$1" 2>/dev/null || echo "missing"
}

# True when the container exists and is running.
# @param $1 container name
cw_is_running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]
}

# Dump processes and best-effort thread backtraces from a running container.
# gdb attach needs CAP_SYS_PTRACE (or a permissive yama scope); when it is not
# allowed, per-thread state + kernel wait channel still shows where each
# thread is parked.
# @param $1 container name
cw_dump_live() {
  local c="$1"
  echo "--- $c: processes ---"
  timeout -k 5 20 docker exec "$c" ps -eo pid,ppid,stat,etime,wchan:24,args 2>&1 | head -60 || true
  if [ "${CW_NO_BT:-0}" = "1" ]; then return 0; fi
  echo "--- $c: thread state / backtraces (best effort) ---"
  timeout -k 5 120 docker exec -i "$c" bash -s <<'CW_BT' 2>&1 || true
skip='^(bash|sh|ps|sleep|gdb|timeout|awk|tini|docker-init|cat|tail|head|grep)$'
for p in $(ps -eo pid=,comm= | awk -v s="$skip" '$2 !~ s {print $1}'); do
  [ "$p" = "1" ] && continue
  [ -d "/proc/$p" ] || continue
  echo "## pid $p: $(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-200)"
  bt=""
  if command -v gdb >/dev/null 2>&1; then
    bt=$(gdb -p "$p" -batch -nx -ex 'set pagination off' \
           -ex 'thread apply all bt 25' 2>/dev/null | grep -E '^(Thread|#)' | head -400)
  fi
  if [ -n "$bt" ]; then
    echo "$bt"
  else
    for t in /proc/"$p"/task/*; do
      echo "  tid ${t##*/} state=$(awk '{print $3}' "$t/stat" 2>/dev/null) wchan=$(cat "$t/wchan" 2>/dev/null) comm=$(cat "$t/comm" 2>/dev/null)"
    done | head -80
  fi
done
CW_BT
  return 0
}

# Dump diagnostics for every suite container (or the ones given): state,
# last CW_LOG_TAIL log lines, and live process / thread state.
# @param $@ optional container names (default: the cw_init list)
cw_dump() {
  local nodes=("$@") c
  if [ ${#nodes[@]} -eq 0 ]; then nodes=("${CW_NODES[@]}"); fi
  echo ""
  echo "==================== cluster diagnostics ===================="
  docker ps -a --format 'table {{.Names}}\t{{.Status}}' 2>&1 | head -40 || true
  if [ -n "$CW_SYNC_DIR" ]; then
    echo "--- sync markers in $CW_SYNC_DIR ---"
    ls -la "$CW_SYNC_DIR" 2>&1 || true
  fi
  for c in "${nodes[@]}"; do
    echo ""
    echo "==================== $c: $(cw_state "$c") ===================="
    docker logs --tail "$CW_LOG_TAIL" "$c" 2>&1 || true
    if cw_is_running "$c"; then
      cw_dump_live "$c"
    fi
  done
  echo "================== end cluster diagnostics =================="
  return 0
}

# Report a harness failure with a dump. Returns 1 so callers can `|| ...`.
# @param $1 message
cw_fail() {
  cw_log "FAILURE: $1"
  cw_dump
  cw_log "FAILURE: $1 (diagnostics above)"
  return 1
}

# Wait until <count> <node>.ready markers exist in the sync directory, written
# by CI/cluster_node.sh once each node's runtime answers `clio_run status`.
# Fails early if a suite container exits before every marker appears.
# @param $1 expected marker count
# @param $2 optional cap in seconds (default CW_READY_CAP)
cw_wait_ready_files() {
  local want="$1" cap start have c
  cap=$(cw_cap "${2:-$CW_READY_CAP}")
  start=$(date +%s)
  cw_log "waiting for $want ready markers in $CW_SYNC_DIR (cap ${cap}s)"
  while :; do
    # Snapshot liveness BEFORE counting markers: a node that exits after the
    # last marker appeared (node1 finishing a fast test) is not a failure.
    local dead=""
    for c in "${CW_NODES[@]}"; do
      if ! cw_is_running "$c"; then dead="$c"; break; fi
    done
    have=0
    for c in "$CW_SYNC_DIR"/*.ready; do
      if [ -e "$c" ]; then have=$((have + 1)); fi
    done
    if [ "$have" -ge "$want" ]; then
      cw_log "all $want nodes ready after $(( $(date +%s) - start ))s"
      return 0
    fi
    if [ -n "$dead" ]; then
      cw_fail "$dead is not running ($(cw_state "$dead")) before readiness ($have/$want ready)"
      return 1
    fi
    if [ $(( $(date +%s) - start )) -ge "$cap" ]; then
      cw_fail "only $have/$want nodes ready after ${cap}s"
      return 1
    fi
    sleep 2
  done
}

# Poll `clio_run status` inside each container until all report RUNNING.
# For suites whose containers just run a runtime and are driven by docker exec.
# CW_STATUS_EXEC_OPTS adds `docker exec` options, e.g. "-e CLIO_SERVER_ADDR=x"
# for a daemon bound to a non-loopback CLIO_BIND_ADDR.
# @param $@ container names (default: the cw_init list)
cw_wait_runtimes() {
  local nodes=("$@") cap start c pending out rc
  if [ ${#nodes[@]} -eq 0 ]; then nodes=("${CW_NODES[@]}"); fi
  cap=$(cw_cap "$CW_READY_CAP")
  start=$(date +%s)
  pending=("${nodes[@]}")
  cw_log "waiting for runtimes RUNNING on: ${pending[*]} (cap ${cap}s)"
  while [ ${#pending[@]} -gt 0 ]; do
    local still=()
    for c in "${pending[@]}"; do
      if ! cw_is_running "$c"; then
        cw_fail "$c is not running ($(cw_state "$c")) while waiting for its runtime"
        return 1
      fi
      rc=0
      # shellcheck disable=SC2086  # CW_STATUS_EXEC_OPTS is word-split on purpose
      out=$(timeout -k 5 30 docker exec -e CLIO_WAIT_SERVER=2 ${CW_STATUS_EXEC_OPTS:-} "$c" clio_run status 2>&1) || rc=$?
      if [ "$rc" -eq 0 ]; then
        cw_log "$c: ${out##*$'\n'}"
      else
        still+=("$c")
      fi
    done
    pending=("${still[@]}")
    if [ ${#pending[@]} -eq 0 ]; then break; fi
    if [ $(( $(date +%s) - start )) -ge "$cap" ]; then
      cw_fail "runtime not RUNNING after ${cap}s on: ${pending[*]} (last status: $out)"
      return 1
    fi
    sleep 2
  done
  cw_log "all runtimes RUNNING after $(( $(date +%s) - start ))s"
  return 0
}

# Wait until a container's log contains a line matching a pattern. For a
# daemon the in-container `clio_run status` probe cannot reach (e.g. one
# bound only to a non-loopback CLIO_BIND_ADDR, where the probe's handshake
# never completes), use the runtime's own "ready" log line instead:
# "Compose: All N pools created successfully" is printed only after the
# compose pools exist, i.e. once the runtime is serving.
# @param $1 container name
# @param $2 extended regex to wait for in `docker logs`
cw_wait_log() {
  local c=$1 pat=$2 cap start
  cap=$(cw_cap "$CW_READY_CAP")
  start=$(date +%s)
  cw_log "waiting for /$pat/ in $c's log (cap ${cap}s)"
  while :; do
    if docker logs "$c" 2>&1 | grep -Eq -- "$pat"; then
      cw_log "$c: ready after $(( $(date +%s) - start ))s"
      return 0
    fi
    if ! cw_is_running "$c"; then
      cw_fail "$c is not running ($(cw_state "$c")) while waiting for /$pat/"
      return 1
    fi
    if [ $(( $(date +%s) - start )) -ge "$cap" ]; then
      cw_fail "/$pat/ not in $c's log after ${cap}s"
      return 1
    fi
    sleep 2
  done
}

# Run `docker exec` bounded by the suite deadline. On a timeout, dumps
# diagnostics and returns 124; any other non-zero exit is returned as-is
# (callers decide whether to dump -- see cw_exec_or_dump).
# @param $1 container name
# @param $@ remaining: arguments passed to `docker exec <container>`
cw_exec() {
  local c="$1" left rc
  shift
  left=$(cw_remaining)
  if [ "$left" -le 0 ]; then
    cw_fail "suite deadline already passed; not running: docker exec $c $*"
    return 124
  fi
  rc=0
  timeout -k 10 "$left" docker exec "$c" "$@" || rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    cw_fail "docker exec on $c exceeded the suite deadline (${left}s left at start)"
    return 124
  fi
  return "$rc"
}

# cw_exec, plus a full dump when the command fails for any reason.
# @param $1 container name
# @param $@ remaining: arguments passed to `docker exec <container>`
cw_exec_or_dump() {
  local rc=0
  cw_exec "$@" || rc=$?
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
    cw_fail "docker exec on $1 exited $rc"
  fi
  return "$rc"
}

# Wait for containers to exit, all against the suite deadline. Records each
# exit code in CW_RC[<name>]. Dumps every node on a timeout or any non-zero
# exit. Returns 0 if all exited 0, 124 on timeout, otherwise the first
# non-zero exit code seen.
# @param $@ container names
cw_wait_containers() {
  local c left out rc first=0
  for c in "$@"; do
    left=$(cw_remaining)
    cw_log "waiting for $c to exit (${left}s left)"
    if [ "$left" -le 0 ]; then
      cw_fail "suite deadline passed before $c exited"
      return 124
    fi
    rc=0
    out=$(timeout -k 10 "$left" docker wait "$c" 2>&1) || rc=$?
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      cw_fail "$c still running at the suite deadline"
      return 124
    fi
    if [ "$rc" -ne 0 ] || ! [[ "$out" =~ ^[0-9]+$ ]]; then
      cw_fail "docker wait $c failed (rc=$rc): $out"
      return 1
    fi
    CW_RC[$c]="$out"
    cw_log "$c exited $out"
    if [ "$out" -ne 0 ] && [ "$first" -eq 0 ]; then first="$out"; fi
  done
  if [ "$first" -ne 0 ]; then
    cw_fail "at least one container exited non-zero"
  fi
  return "$first"
}
