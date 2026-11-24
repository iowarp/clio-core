#!/usr/bin/env bash
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A IOWarp
#PBS -j oe
#
# Run the USM atomic probe on one Aurora node. The debug queue will not take a
# walltime under five minutes, so five is requested -- but the run is capped at
# 60s and the job exits when it does, and elapsed time is what is billed. The
# probe itself takes seconds. It answers one question -- which allocation kinds
# take a device atomic on this GPU -- and that answer picks the fix.
set -u
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
EXE="${ROOT}/build-spike/${PROBE_EXE:-usm_atomic_probe}"
echo "=== usm_atomic_probe on $(hostname) ==="
test -x "${EXE}" || { echo "NO EXECUTABLE"; exit 2; }
ONEAPI_DEVICE_SELECTOR=level_zero:gpu timeout --kill-after=10s 60 "${EXE}" 2>&1
echo "--- exit=$? ---"
