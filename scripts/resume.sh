#!/usr/bin/env bash
#
# resume.sh — resume a previously-stopped stack.
#
# Only calls start.sh, but in PERSISTENT (recover) mode: the runtime starts
# without --ephemeral so it recovers the saved bdev/safe-bdev state (alloc-logs
# + member superblocks) instead of beginning a fresh session. Backing files are
# reused as-is (never re-created).
set -euo pipefail
exec env EPHEMERAL=0 "$(dirname "$(readlink -f "$0")")/start.sh" "$@"
