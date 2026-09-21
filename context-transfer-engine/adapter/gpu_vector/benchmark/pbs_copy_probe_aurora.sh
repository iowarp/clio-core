#!/usr/bin/env bash
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -l filesystems=home
#PBS -q debug
#PBS -A IOWarp
#PBS -j oe
#
# Host-initiated 64 KB USM copy latency under a persistent kernel, one tile,
# a few seconds of GPU time. See sycl_copy_probe.cc for what is measured and
# why; the numbers decide how the runtime stages device-resident bulk
# buffers for the network.
set -u
ROOT=${ROOT:?ROOT (worktree) must be passed with -v}
EXE="${ROOT}/build-spike/sycl_copy_probe"
echo "=== copy probe on $(hostname) ==="
test -x "${EXE}" || { echo "NO EXECUTABLE ${EXE}"; echo "RESULT copyprobe: FAILED"; exit 2; }
export ZE_AFFINITY_MASK="${BENCH_ZE_MASK:-0.0}"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
mkdir -p "$SYCL_CACHE_DIR"
# Second pass with copy engines forced on, to see whether the environment
# alone moves the busy-tile numbers.
for pass in default copyengine; do
  echo "--- pass ${pass} ---"
  if [ "${pass}" = copyengine ]; then
    export UR_L0_USE_COPY_ENGINE=1 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1
    export UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY=1 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=1
  fi
  timeout --signal=TERM --kill-after=10s 60 "${EXE}"
  echo "pass ${pass} exit=$?"
done
echo "RESULT copyprobe: done"
