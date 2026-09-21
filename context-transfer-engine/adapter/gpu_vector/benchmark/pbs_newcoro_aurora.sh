#!/usr/bin/env bash
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A IOWarp
#PBS -j oe
#
# Run ONE ported paged benchmark on one Aurora node, on the Intel GPU.
#
# Submitted one job per benchmark, so a hang or a crash in one says nothing
# about the others and each gets its own log. Driven by two environment
# variables passed through qsub -v:
#
#   BENCH_NAME  the benchmark (gmx, lbann, weights, ...)
#   BENCH_ARGS  its arguments, sized so the run finishes well inside
#               five minutes
#
# The executables are built on the login node by build_newcoro_aurora.sh;
# this script only runs them.
set -u

: "${BENCH_NAME:?set BENCH_NAME}"
: "${BENCH_ARGS:=}"
ROOT=${ROOT:-/home/llogan/clio-core/.claude/worktrees/gpu-coro}
# BENCH_EXE overrides the executable name, e.g. the _aot variant that carries
# pre-compiled PVC code and pays no JIT at launch.
EXE="${ROOT}/build-spike/${BENCH_EXE:-clio_${BENCH_NAME}_paged_newcoro}"
# On the SHARED filesystem, not /tmp, and with core dumps enabled: grayscott
# SIGSEGV'd one second into runtime init on a code path four other benchmarks
# had passed through, and its core -- written to node-local /tmp -- was gone
# with the node. A core here can be opened with gdb on the login node against
# the same binary and libraries.
RUNDIR=${RUNDIR:-${ROOT}/build-spike/run_${BENCH_NAME}}
ulimit -c unlimited

echo "=== ${BENCH_NAME} on $(hostname) ==="
echo "exe:  ${EXE}"
echo "args: ${BENCH_ARGS}"
test -x "${EXE}" || { echo "NO EXECUTABLE -- build it first"; exit 2; }

mkdir -p "${RUNDIR}"
cd "${RUNDIR}"
rm -f ./*.yaml

echo "--- devices ---"
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls 2>&1 | head -5

# The executables are spir64 JIT, so the first launch pays IGC on a device
# module of the better part of a megabyte. Caching it keeps that cost off every
# subsequent run, including the other jobs on the same filesystem.
# IGC must not inline the coroutine functions into one kernel: on lammps_md
# that yields a 6,475-block function that segfaults IGC's backend. 3 = keep
# them as stack calls. Read by IGC at JIT time; harmless for AOT binaries,
# which had it baked in at their ocloc step. See build_newcoro_aurora.sh.
export IGC_FunctionControl=3
# ONE TILE, BY DEFAULT. An Aurora node is 6 root GPUs x 2 tiles, and under the
# default composite hierarchy a kernel is implicitly scaled across both tiles
# of its root device. lbann passes pinned to a tile and dies without the pin
# on an AtomicAccessViolation (PDE level: an atomic to a page the context has
# not mapped) inside 3s, while kmeans, gmx, grayscott and weights pass either
# way. Every atomic lbann issues targets GpuApi::Malloc device memory, so the
# composite-mode fault is in the driver's view of that memory, not in the
# benchmark; it stays OPEN, and one tile -- ALCF's recommended unit anyway --
# is the configuration these results are reported under. BENCH_ZE_MASK
# overrides (e.g. 0.1); BENCH_ZE_MASK=none runs composite.
case "${BENCH_ZE_MASK:-0.0}" in
  none) ;;
  *)    export ZE_AFFINITY_MASK="${BENCH_ZE_MASK:-0.0}" ;;
esac
# BENCH_TRACE=1 makes the SYCL runtime log every Unified Runtime call, kernel
# launches by name included, so a GPU fault can be attributed to the last
# kernel enqueued before it. Env var only; the binary is untouched.
[ -n "${BENCH_TRACE:-}" ] && export SYCL_UR_TRACE=2
export SYCL_CACHE_PERSISTENT=1
# On a SHARED filesystem, not /tmp: /tmp is node-local, and each job lands on
# whatever node is free, so a /tmp cache re-JITs every single run. The first
# device-ring run spent its whole 90s budget with the banner printed and no
# progress, which is what JIT of a 700 KB spir64 module under
# IGC_FunctionControl=3 looks like.
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-${ROOT}/build-spike/sycl_cache}
mkdir -p "$SYCL_CACHE_DIR"

# 90s inside a 2 minute walltime. Every failure so far landed inside 65s, so
# the extra minutes were buying nothing; and the cap must sit well under the
# walltime or PBS reaps the job first and the log ends mid-sentence.
echo "--- run (90s cap, 2 minute walltime) ---"
start=$SECONDS
# LINE-BUFFERED. stdout to a pipe is fully buffered, and a run that is killed
# by the cap loses everything still in the buffer -- the first ring-fix run
# timed out with an EMPTY log for exactly that reason, which made "hung at
# init" and "hung at the first kernel" indistinguishable. stdbuf makes every
# line reach the log the moment it is printed. `tail` is dropped for the same
# reason: it would hold the last 40 lines until the pipe closed.
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  timeout --signal=TERM --kill-after=10s 90 \
  stdbuf -oL -eL "${EXE}" ${BENCH_ARGS} 2>&1 | grep --line-buffered -vE "LoadBalance|\[#78[15]"
rc=${PIPESTATUS[0]}
echo "--- elapsed $((SECONDS - start))s, exit=${rc} ---"

case "${rc}" in
  0)   echo "RESULT ${BENCH_NAME}: OK" ;;
  124) echo "RESULT ${BENCH_NAME}: TIMEOUT (exceeded the 90s cap)" ;;
  *)   echo "RESULT ${BENCH_NAME}: FAILED rc=${rc}" ;;
esac
exit "${rc}"
