#!/usr/bin/env bash
# Retry a command that talks to the network.
#
# Package mirrors and release CDNs go away for tens of seconds at a time, and
# when they do the job fails with a red X that has nothing to do with the code
# under test -- chocolatey.org being unreachable failed a whole Windows wheels
# build on dev, and apt/brew/pip do the same thing less often. Every install
# step that reaches the network should go through here so a transient outage
# costs a retry instead of a rerun (and a maintainer's attention).
#
# Only for IDEMPOTENT commands: installs, downloads, index refreshes. Never
# wrap a build or a test -- retrying those is how a flaky test gets to hide.
#
# Usage:
#   CI/retry.sh <command> [args...]
#   CI/retry.sh -n 5 -s 10 -- <command> [args...]
#
#   -n  attempts (default 3)
#   -s  seconds to sleep before the 2nd attempt; doubles thereafter (default 5)

set -uo pipefail

attempts=3
sleep_s=5

while [ $# -gt 0 ]; do
  case "$1" in
    -n) attempts="$2"; shift 2 ;;
    -s) sleep_s="$2"; shift 2 ;;
    --) shift; break ;;
    *)  break ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "CI/retry.sh: no command given" >&2
  exit 2
fi

n=1
while true; do
  "$@" && exit 0
  rc=$?
  if [ "$n" -ge "$attempts" ]; then
    echo "CI/retry.sh: '$*' failed $attempts/$attempts attempts (last rc=$rc)" >&2
    exit "$rc"
  fi
  echo "CI/retry.sh: attempt $n/$attempts failed (rc=$rc); retrying in ${sleep_s}s: $*" >&2
  sleep "$sleep_s"
  sleep_s=$((sleep_s * 2))
  n=$((n + 1))
done
