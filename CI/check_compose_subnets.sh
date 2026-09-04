#!/usr/bin/env bash
# Fail if two integration docker-compose stacks declare overlapping subnets.
#
# Why this exists. Five stacks were all pinned to 172.30.0.0/24 and one to a
# 172.28.0.0/16 that swallowed a sixth. That is invisible until a step is killed
# by `timeout-minutes` before its own teardown -- then the leaked network is
# still attached when the NEXT stack starts, and docker refuses it with
#
#   failed to create network grayscott_congestion_gs-cluster: Error response
#   from daemon: invalid pool request: Pool overlaps with other one on this
#   address space
#
# which reads like a docker problem and is actually two of our compose files
# claiming the same address space. The coherence step timing out at 20 minutes
# took the grayscott step down that way in 16 of 52 Cluster Tests runs.
#
# Unique subnets do not stop a network leaking; they stop a leak in one stack
# from taking an unrelated stack with it, so the failure stays attributable to
# the test that actually broke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

declare -A OWNER
rc=0

# subnet -> first file that claimed it; second claimant is the error.
while IFS= read -r line; do
    file="${line%%:*}"
    subnet="$(printf '%s' "$line" | sed 's/.*subnet: *//' | tr -d '[:space:]')"
    [ -n "$subnet" ] || continue
    if [ -n "${OWNER[$subnet]:-}" ]; then
        echo "ERROR: subnet $subnet claimed twice:" >&2
        echo "         ${OWNER[$subnet]}" >&2
        echo "         $file" >&2
        rc=1
    else
        OWNER[$subnet]="$file"
    fi
done < <(grep -rn "subnet:" \
            context-runtime/test/integration \
            context-transfer-engine/test/integration \
            2>/dev/null | grep -v "/build/")

# A prefix shorter than /24 spans whole /24s, so it can overlap a sibling even
# when the literal strings differ -- that is exactly how the 172.28.0.0/16 hid.
while IFS= read -r line; do
    file="${line%%:*}"
    subnet="$(printf '%s' "$line" | sed 's/.*subnet: *//' | tr -d '[:space:]')"
    prefix="${subnet##*/}"
    case "$subnet" in
        */*) ;;
        *) continue ;;
    esac
    if [ "$prefix" -lt 24 ] 2>/dev/null; then
        echo "ERROR: $file declares $subnet -- a prefix shorter than /24 spans" >&2
        echo "         multiple /24s and can collide with a sibling stack." >&2
        echo "         Give this stack its own /24." >&2
        rc=1
    fi
done < <(grep -rn "subnet:" \
            context-runtime/test/integration \
            context-transfer-engine/test/integration \
            2>/dev/null | grep -v "/build/")

if [ "$rc" -eq 0 ]; then
    echo "compose subnets: ${#OWNER[@]} stacks, all distinct, all /24 or narrower"
fi
exit "$rc"
