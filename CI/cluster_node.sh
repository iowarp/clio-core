#!/usr/bin/env bash
# Container-side readiness and barrier primitives for the docker-compose
# integration clusters. Every wait here is bounded and fails loudly.
#
# Why this exists. Cluster nodes used to start a runtime in the background,
# `sleep 3`, and assume every peer was up; teardown was a fixed `sleep N`
# "linger" so slow peers would not lose their partner. On a loaded runner both
# guesses are wrong, and the failure was a GitHub step timeout with no output
# (CTE distributed test, 2026-11-18). These replace guesses with signals:
#
#   * ready     -- the LOCAL runtime answered `clio_run status` (a heartbeat
#                  task round-trip, exit 0 = RUNNING), then <node>.ready is
#                  written to a directory every node shares (/workspace bind
#                  mount), so peers and the host runner can see it.
#   * wait-*    -- block until N marker files exist, or a single file exists,
#                  with a hard cap. Used as start ("all runtimes up") and end
#                  ("every node finished its test") barriers.
#
# The host side (CI/compose_wait.sh) resets the sync directory before
# `docker compose up` and watches the same files.
#
# Usage (inside a container; /workspace is the repo):
#   cluster_node.sh ready      <sync_dir|-> <node_name> [runtime_pid] [cap_s]
#   cluster_node.sh wait-ready <sync_dir> <count> [cap_s]
#   cluster_node.sh done       <sync_dir> <node_name> [exit_code]
#   cluster_node.sh wait-done  <sync_dir> <count> [cap_s]
#   cluster_node.sh wait-file  <path> [cap_s]
#   cluster_node.sh mark       <path>
#
# `ready` with sync_dir "-" only waits for RUNNING and writes no marker (for
# single-node scripts).
#
# Exit status: 0 when the condition holds, 1 when the cap expires (after
# printing what was and was not seen), 2 on a usage error.
#
# Environment:
#   CLIO_RUN_BIN   clio_run binary (default: clio_run on PATH)
#   CN_READY_CAP   default cap for `ready` / `wait-ready` (180 s)
#   CN_DONE_CAP    default cap for `wait-done` / `wait-file` (600 s)
#   CN_STATUS_WAIT CLIO_WAIT_SERVER for each status probe (2 s), so a
#                  caller's larger CLIO_WAIT_SERVER cannot stretch one probe

set -uo pipefail

CLIO_RUN_BIN="${CLIO_RUN_BIN:-clio_run}"
CN_READY_CAP="${CN_READY_CAP:-180}"
CN_DONE_CAP="${CN_DONE_CAP:-600}"

# Print a timestamped message tagged with the host name.
# @param $* message text
cn_log() {
  echo "[cluster_node $(hostname 2>/dev/null || echo '?') $(date +%H:%M:%S)] $*"
}

# Run a command with a time limit, portably. GNU `timeout` is not on macOS
# (Homebrew installs it as `gtimeout`, but runners need not have it), and the
# restart integration tests run this script natively on macOS. Without either,
# fall back to a background watchdog: TERM after <secs>, KILL 5 s later.
# Like GNU timeout, the signal goes to the command's whole process group
# (`set -m` gives the background job its own group), so a grandchild cannot
# keep a caller's $( ... ) pipe open. The watchdog's own output goes to
# /dev/null for the same reason.
# @param $1 time limit in seconds
# @param $@ command and arguments
# @return the command's exit status (124/137 from timeout, 143/137 from the
#         watchdog)
cn_bounded() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout -k 5 "$secs" "$@"; return
  fi
  if command -v gtimeout >/dev/null 2>&1; then
    gtimeout -k 5 "$secs" "$@"; return
  fi
  (
    # Job control prints "[1]+ Done ..." notices on the shell's stderr; send
    # those to /dev/null and hand the command the caller's stderr (fd 3).
    exec 3>&2 2>/dev/null
    set -m
    "$@" 2>&3 3>&- &
    pid=$!
    ( sleep "$secs"; kill -TERM -- "-$pid" 2>/dev/null; sleep 5
      kill -KILL -- "-$pid" 2>/dev/null ) >/dev/null 2>&1 &
    wd=$!
    wait "$pid"; rc=$?
    kill -- "-$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    exit "$rc"
  )
}

# Count files matching <dir>/*<suffix>.
# @param $1 directory
# @param $2 file-name suffix (e.g. .ready)
cn_count() {
  local dir="$1" suffix="$2" n=0 f
  for f in "$dir"/*"$suffix"; do
    [ -e "$f" ] && n=$((n + 1))
  done
  echo "$n"
}

# Poll the local runtime until `clio_run status` exits 0 (RUNNING), then
# write <sync_dir>/<node>.ready. Fails fast if the runtime process dies.
# @param $1 sync directory shared by all nodes ("-" = write no marker)
# @param $2 this node's name (marker file stem)
# @param $3 optional runtime pid to watch for an early death
# @param $4 optional cap in seconds (default CN_READY_CAP)
cn_ready() {
  local dir="$1" node="$2" pid="${3:-}" cap="${4:-$CN_READY_CAP}"
  local start out rc
  start=$(date +%s)
  if [ "$dir" != "-" ]; then mkdir -p "$dir"; fi
  while :; do
    # status bounds its own connect (CLIO_WAIT_SERVER=2); cn_bounded guards a
    # wedged probe so one hung client cannot eat the whole cap.
    out=$(CLIO_WAIT_SERVER="${CN_STATUS_WAIT:-2}" \
          cn_bounded 30 "$CLIO_RUN_BIN" status 2>&1)
    rc=$?
    if [ "$rc" -eq 0 ]; then
      cn_log "runtime RUNNING after $(( $(date +%s) - start ))s: ${out##*$'\n'}"
      if [ "$dir" = "-" ]; then return 0; fi
      echo "$(date +%s) $out" > "$dir/$node.ready.tmp" &&
        mv -f "$dir/$node.ready.tmp" "$dir/$node.ready"
      return 0
    fi
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      cn_log "ERROR: runtime pid $pid exited before it became ready"
      echo "$out"
      return 1
    fi
    if [ $(( $(date +%s) - start )) -ge "$cap" ]; then
      cn_log "ERROR: runtime not RUNNING after ${cap}s (last status rc=$rc):"
      echo "$out"
      return 1
    fi
    sleep 1
  done
}

# Wait until <count> files ending in <suffix> exist in <dir>.
# @param $1 directory
# @param $2 suffix (.ready / .done)
# @param $3 expected count
# @param $4 cap in seconds
cn_wait_count() {
  local dir="$1" suffix="$2" want="$3" cap="$4" start have
  start=$(date +%s)
  while :; do
    have=$(cn_count "$dir" "$suffix")
    if [ "$have" -ge "$want" ]; then
      cn_log "barrier '$suffix' reached ($have/$want) after $(( $(date +%s) - start ))s"
      return 0
    fi
    if [ $(( $(date +%s) - start )) -ge "$cap" ]; then
      cn_log "ERROR: barrier '$suffix' NOT reached after ${cap}s ($have/$want); present:"
      ls -la "$dir" 2>&1
      return 1
    fi
    sleep 1
  done
}

# Wait until a single file exists.
# @param $1 path
# @param $2 cap in seconds
cn_wait_file() {
  local path="$1" cap="$2" start
  start=$(date +%s)
  while [ ! -e "$path" ]; do
    if [ $(( $(date +%s) - start )) -ge "$cap" ]; then
      cn_log "ERROR: $path did not appear within ${cap}s"
      return 1
    fi
    sleep 1
  done
  cn_log "$path present after $(( $(date +%s) - start ))s"
  return 0
}

cmd="${1:-}"
[ $# -gt 0 ] && shift
case "$cmd" in
  ready)
    [ $# -ge 2 ] || { echo "usage: $0 ready <dir> <node> [pid] [cap]" >&2; exit 2; }
    cn_ready "$@" ;;
  wait-ready)
    [ $# -ge 2 ] || { echo "usage: $0 wait-ready <dir> <count> [cap]" >&2; exit 2; }
    cn_wait_count "$1" .ready "$2" "${3:-$CN_READY_CAP}" ;;
  done)
    [ $# -ge 2 ] || { echo "usage: $0 done <dir> <node> [rc]" >&2; exit 2; }
    mkdir -p "$1" && echo "${3:-0}" > "$1/$2.done" ;;
  wait-done)
    [ $# -ge 2 ] || { echo "usage: $0 wait-done <dir> <count> [cap]" >&2; exit 2; }
    cn_wait_count "$1" .done "$2" "${3:-$CN_DONE_CAP}" ;;
  wait-file)
    [ $# -ge 1 ] || { echo "usage: $0 wait-file <path> [cap]" >&2; exit 2; }
    cn_wait_file "$1" "${2:-$CN_DONE_CAP}" ;;
  mark)
    [ $# -ge 1 ] || { echo "usage: $0 mark <path>" >&2; exit 2; }
    mkdir -p "$(dirname "$1")" && date +%s > "$1" ;;
  *)
    sed -n '2,36p' "$0" >&2
    exit 2 ;;
esac
