#!/usr/bin/env bash
#
# Submit-path hop breakdown sweep (GSBENCH_SUBMIT_PROBE=1).
#
# Produces the per-submit hop timings behind agents/paper-writing/traces/08-submit-overhead.md.
# Each run drops three CSVs into its own directory under $OUT:
#   probe_dev_<arm>.csv   device stamps (globaltimer ns + clock64 cycles)
#   probe_host_<arm>.csv  host stamps (CLOCK_MONOTONIC ns) + worker backoff state
#   probe_meta_<arm>.csv  clock offset, its uncertainty, globaltimer tick, SM GHz
# Joined on task_ptr by analyze_submit_probe.py.
#
# THIS IS THE MOST TIMING-SENSITIVE MEASUREMENT IN THE PAPER. The CPU worker threads
# busy-spin (Worker::Run only sleeps via SuspendMe() when idle), so any competing CPU load
# perturbs poll latency and worker wake-up directly. Run on a quiescent machine and record
# the load average (this script does).
#
# Usage (inside the CUDA/iowarp dev container):
#   OUT=/tmp/probe_sweep REPS=5 bash run_submit_probe_sweep.sh
#
# Write OUT *outside* the repo bind-mount: `docker exec ... > build/x.csv` truncates the
# host file before the container writes it (this ate a full sweep during the D2H work).

set -u

: "${GSBENCH_BUILD_DIR:=/workspace/build}"
: "${GSBENCH_BIN:=${GSBENCH_BUILD_DIR}/bin/kvhdf5_e2e_tests}"
: "${OUT:=/tmp/probe_sweep}"
: "${REPS:=5}"
: "${TIMEOUT:=900}"

# The paper's reference workload (matches the 09 D2H run): 12 x 156.25 MiB snapshots.
: "${N:=6400}"
: "${SNAPS:=12}"
: "${CHUNKS:=4}"
: "${CAP_MB:=3072}"

bin_dir="$(dirname "${GSBENCH_BIN}")"
export CLIO_BIND_ADDR="${CLIO_BIND_ADDR:-127.0.0.1}"
export CHI_REPO_PATH="${CHI_REPO_PATH:-${bin_dir}}"
export LD_LIBRARY_PATH="${bin_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"

# The CLIO arms require a single server worker. This is also what makes the routing funnel
# (S3) unambiguous: one worker pops the GPU lane AND runs the coroutine, so a hop that
# looks like a queue wait cannot be a cross-core migration in disguise.
nt1_conf="$(mktemp)"
printf 'runtime:\n  num_threads: 1\n' > "${nt1_conf}"
export CLIO_SERVER_CONF="${nt1_conf}"
trap 'rm -f "${nt1_conf}"' EXIT

mkdir -p "${OUT}"
echo "quiescence: load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee "${OUT}/quiescence.txt"
nvidia-smi --query-gpu=name,utilization.gpu,clocks.sm --format=csv,noheader >> "${OUT}/quiescence.txt" 2>/dev/null

reset_state() { pkill -9 -x kvhdf5_e2e_tests 2>/dev/null; rm -f /dev/shm/chi_*; sleep 1; }

# run <label> <catch-tag> <steps_per> <bdev> <probe_on> <rep>
run() {
    local label="$1" tag="$2" steps="$3" bdev="$4" probe="$5" rep="$6"
    local dir="${OUT}/${label}/rep${rep}"
    mkdir -p "${dir}"
    reset_state
    rm -f /tmp/gsbench_probe_bdev.dat
    GSBENCH_SUBMIT_PROBE="${probe}" GSBENCH_PROBE_DIR="${dir}" \
    GSBENCH_N="${N}" GSBENCH_SNAPS="${SNAPS}" GSBENCH_CHUNKS="${CHUNKS}" \
    GSBENCH_STEPS_PER="${steps}" GSBENCH_BDEV="${bdev}" GSBENCH_BDEV_CAP_MB="${CAP_MB}" \
    GSBENCH_BDEV_PATH=/tmp/gsbench_probe_bdev.dat \
    timeout "${TIMEOUT}" "${GSBENCH_BIN}" "[${tag}]" > "${dir}/run.log" 2>&1
    local rc=$?
    # A run with failed puts must never reach the breakdown: it would be timing an I/O that
    # never happened. ThrowIfIoFailed aborts the arm, so a nonzero rc here is fatal, not noise.
    if [[ ${rc} -ne 0 ]]; then
        echo "FAILED ${label} rep${rep} rc=${rc}" | tee -a "${OUT}/failures.txt"
        grep -E "IoFailed|abort|terminate|FAILED" "${dir}/run.log" | head -3 >> "${OUT}/failures.txt"
        return
    fi
    grep -E "GSBENCH_RESULT|GSBENCH_PROBE" "${dir}/run.log" >> "${OUT}/summary.txt"
    echo "ok ${label} rep${rep}: $(grep -oE 'ms=[0-9.]+' "${dir}/run.log" | head -1)"
}

# ---- 1. the headline: sync + async at the reference workload -------------------------
for r in $(seq 1 "${REPS}"); do
    run sync_file_s48   gsbench_sync  48 file 1 "${r}"
    run async_file_s48  gsbench_async 48 file 1 "${r}"
done

# ---- 2. hot-spinning vs idle-wakeup -------------------------------------------------
# steps_per sets the compute gap between snapshots, i.e. how long the worker sits idle
# before the next burst of submits. Short gap -> the worker is still hot-spinning when the
# push lands. Long gap -> it has backed off into SuspendMe() and must be woken. These are
# very different poll latencies and are reported separately (the per-record backoff state
# is the discriminator, so the split is also recoverable within a single run).
for r in $(seq 1 3); do
    run sync_file_s4    gsbench_sync    4 file 1 "${r}"   # hot
    run sync_file_s384  gsbench_sync  384 file 1 "${r}"   # idle
done

# ---- 3. the bdev confound under hop 5 ------------------------------------------------
# kRam's pages are neither pinned nor preallocated, and kPinned carries a lazy
# cudaMallocHost stall INSIDE the I/O path. Both land under execution; if they still hold,
# the breakdown must say so or it misattributes their cost to the transfer engine.
for r in $(seq 1 3); do
    run sync_ram_s48    gsbench_sync 48 ram    1 "${r}"
    run sync_pinned_s48 gsbench_sync 48 pinned 1 "${r}"
done

# ---- 4. probe-off control: is the instrumentation free? ------------------------------
# Interleaved with the probe-on runs above would be better still, but the arms are
# separate processes and the probe's cost is bounded by construction (device stamps go to
# device memory, host stamps are pointer writes). This is the check on that claim.
for r in $(seq 1 "${REPS}"); do
    run sync_file_s48_noprobe  gsbench_sync  48 file 0 "${r}"
    run async_file_s48_noprobe gsbench_async 48 file 0 "${r}"
done

reset_state
rm -f /tmp/gsbench_probe_bdev.dat
echo "done -> ${OUT}"
