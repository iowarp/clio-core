#!/usr/bin/env bash
#
# Three-way Gray-Scott I/O benchmark: no-CLIO (raw) vs CLIO-sync vs CLIO-async.
#
# Drives the hidden Catch2 cases in gray_scott_threeway_bench.cu. Each arm runs in its
# OWN process, so the two CLIO arms don't share iowarp's ~16-large-GPU-backend-per-process
# ceiling. All three run the identical Gray-Scott computation through one shared kernel +
# timed loop; only the snapshot sink differs (raw disk / CLIO sync / CLIO async). A
# cross-arm FNV checksum of the persisted bytes proves the computation was identical.
#
# Run this INSIDE the CUDA/iowarp dev container, after building the target:
#     cmake --build build --target kvhdf5_e2e_tests
#     bash <this-script>
#
# Defaults = the fair, apples-to-apples comparison: ~1.9 GB/arm, both arms durably
# persisted to real disk (raw fdatasyncs each snapshot; CLIO uses a kFile bdev), checksum
# excluded from the timed region on every arm. Override any knob via env, e.g.:
#
#   # RAM tier (software-path only; raw = buffered tmpfs, no durability):
#   GSBENCH_BDEV=ram GSBENCH_RAW_ODIRECT=0 GSBENCH_DISK_DIR=/dev/shm/gsbench_raw \
#       bash run_threeway_bench.sh
#
#   # more repeats for a stable number:
#   GSBENCH_REPEATS=3 bash run_threeway_bench.sh
#
# Benchmark knobs (read by the test binary; see gray_scott_threeway_bench.cu):
#   GSBENCH_N GSBENCH_CHUNKS GSBENCH_SNAPS GSBENCH_STEPS_PER   grid / chunking / schedule
#   GSBENCH_BDEV (file|ram) GSBENCH_BDEV_CAP_MB GSBENCH_BDEV_PATH   CLIO storage
#   GSBENCH_RAW_ODIRECT (0|1) GSBENCH_RAW_FSYNC (0|1) GSBENCH_DISK_DIR   raw storage
#
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- locations (override via env) -------------------------------------------------------
# Default build layout: <iowarp-core>/build/bin. This script now lives at
# context-transfer-engine/adapter/kvhdf5/test/e2e/, so the iowarp-core root is
# five levels up. The CMake `threeway_bench` target overrides GSBENCH_BUILD_DIR
# with the real ${CMAKE_BINARY_DIR}, so this default only matters when running
# the script by hand.
: "${GSBENCH_BUILD_DIR:=${script_dir}/../../../../../build}"
: "${GSBENCH_BIN:=${GSBENCH_BUILD_DIR}/bin/kvhdf5_e2e_tests}"
: "${GSBENCH_SCRATCH:=${GSBENCH_BUILD_DIR}/gsbench_scratch}"
: "${GSBENCH_TIMEOUT:=600}"     # per-arm wall-clock guard (seconds)
: "${GSBENCH_REPEATS:=1}"

if [[ ! -x "${GSBENCH_BIN}" ]]; then
    echo "error: test binary not found/executable: ${GSBENCH_BIN}" >&2
    echo "       build it first: cmake --build ${GSBENCH_BUILD_DIR} --target kvhdf5_e2e_tests" >&2
    exit 1
fi

# --- benchmark config defaults (fair durable-disk comparison) ---------------------------
export GSBENCH_N="${GSBENCH_N:-6400}"
export GSBENCH_CHUNKS="${GSBENCH_CHUNKS:-4}"
export GSBENCH_SNAPS="${GSBENCH_SNAPS:-12}"          # keep <= ~12 (iowarp backend ceiling)
export GSBENCH_STEPS_PER="${GSBENCH_STEPS_PER:-48}"
export GSBENCH_BDEV="${GSBENCH_BDEV:-file}"
export GSBENCH_BDEV_CAP_MB="${GSBENCH_BDEV_CAP_MB:-3072}"
export GSBENCH_BDEV_PATH="${GSBENCH_BDEV_PATH:-${GSBENCH_SCRATCH}/clio_bdev.dat}"
export GSBENCH_RAW_ODIRECT="${GSBENCH_RAW_ODIRECT:-0}"
export GSBENCH_RAW_FSYNC="${GSBENCH_RAW_FSYNC:-1}"
export GSBENCH_DISK_DIR="${GSBENCH_DISK_DIR:-${GSBENCH_SCRATCH}/raw_out}"

mkdir -p "${GSBENCH_SCRATCH}" "${GSBENCH_DISK_DIR}"

# --- iowarp runtime env -----------------------------------------------------------------
bin_dir="$(dirname "${GSBENCH_BIN}")"
export CLIO_BIND_ADDR="${CLIO_BIND_ADDR:-127.0.0.1}"
export CHI_REPO_PATH="${CHI_REPO_PATH:-${bin_dir}}"
export LD_LIBRARY_PATH="${bin_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# The CLIO arms must run at a single server worker (num_threads=1): the concurrent-put-safe
# substrate for multi-chunk async snapshots. Generate the config unless the caller supplied
# one.
nt1_conf=""
if [[ -z "${CLIO_SERVER_CONF:-}" ]]; then
    nt1_conf="$(mktemp)"
    printf 'runtime:\n  num_threads: 1\n' > "${nt1_conf}"
    export CLIO_SERVER_CONF="${nt1_conf}"
fi

logs="$(mktemp -d)"
cleanup() {
    rm -rf "${logs}" "${GSBENCH_DISK_DIR}" 2>/dev/null
    rm -f "${GSBENCH_BDEV_PATH}" 2>/dev/null
    [[ -n "${nt1_conf}" ]] && rm -f "${nt1_conf}" 2>/dev/null
}
trap cleanup EXIT

# A stale/killed run leaves a spinning server + shm segments that make the next run hang.
reset_state() { pkill -9 -x kvhdf5_e2e_test 2>/dev/null; rm -f /dev/shm/chi_*; sleep 1; }

run_arm() {  # $1 = catch tag, $2 = label
    reset_state
    echo ">>> arm: $2"
    timeout "${GSBENCH_TIMEOUT}" "${GSBENCH_BIN}" "$1" 2>&1 \
        | grep -E 'GSBENCH_RESULT|GSBENCH_TRACE|\[bench\]|\[raw\]|FAILED|error' \
        | tee "${logs}/$2.log"
    echo "    (exit ${PIPESTATUS[0]})"
}

for rep in $(seq 1 "${GSBENCH_REPEATS}"); do
    [[ "${GSBENCH_REPEATS}" -gt 1 ]] && echo "================ REPEAT ${rep}/${GSBENCH_REPEATS} ================"
    run_arm "[gsbench_raw]"      "raw"
    run_arm "[gsbench_hostclio]" "hostclio"
    run_arm "[gsbench_sync]"     "sync"
    run_arm "[gsbench_async]"    "async"
done
reset_state

echo
echo "================ BENCHMARK RESULT ================"
cat "${logs}"/raw.log "${logs}"/hostclio.log "${logs}"/sync.log "${logs}"/async.log \
    2>/dev/null | grep GSBENCH_RESULT

python3 - "${logs}" <<'PY' 2>/dev/null || true
import re, sys, glob, os
logs = sys.argv[1]
rows = {}
for f in ("raw", "hostclio", "sync", "async"):
    p = os.path.join(logs, f + ".log")
    if not os.path.exists(p):
        continue
    m = re.search(r'GSBENCH_RESULT .*arm=(\S+).*MB=([\d.]+) ms=([\d.]+) MBps=([\d.]+) checksum=(\d+)',
                  open(p).read())
    if m:
        rows[m.group(1)] = dict(mb=float(m.group(2)), ms=float(m.group(3)),
                                mbps=float(m.group(4)), ck=m.group(5))
if rows:
    base = rows.get("raw", {}).get("ms")
    print(f"\n{'arm':<10}{'MB':>9}{'ms':>11}{'MB/s':>10}{'vs raw':>9}  checksum")
    for a in ("raw", "hostclio", "sync", "async"):
        r = rows.get(a)
        if not r:
            continue
        rel = f"{base / r['ms']:.2f}x" if base else "-"
        print(f"{a:<10}{r['mb']:>9.1f}{r['ms']:>11.2f}{r['mbps']:>10.1f}{rel:>9}  {r['ck']}")
    cks = {r['ck'] for r in rows.values()}
    print("\nchecksums", "MATCH (identical computation)" if len(cks) == 1 else f"DIFFER {cks}")
PY
