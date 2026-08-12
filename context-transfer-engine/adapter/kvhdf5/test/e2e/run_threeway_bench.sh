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
#   GSBENCH_HDF5_DIR GSBENCH_HDF5_RDCC_MB GSBENCH_HDF5_EARLY_ALLOC            hdf5 arm
#   GSBENCH_HDF5_DIRECT_CHUNK (0|1) GSBENCH_HDF5_STOCK (0|1)                  hdf5 tuning
#
# GSBENCH_RAW_INLINE selects the writer STRUCTURE:
#   0 = background writer thread (I/O overlaps compute; pairs with clio-async)
#   1 = inline synchronous       (GPU idle during the write; pairs with clio-sync)
# It applies to the `raw` arm here. The HDF5 arm is run BOTH ways in one sweep (this script
# forces the var per-arm), so it always reports as two rows: hdf5_inline and hdf5_threaded.
#
# GSBENCH_HDF5_ASYNC=1 adds a third HDF5 row via the async I/O VOL connector (see below).
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
export GSBENCH_HDF5_DIR="${GSBENCH_HDF5_DIR:-${GSBENCH_SCRATCH}/hdf5_out}"

mkdir -p "${GSBENCH_SCRATCH}" "${GSBENCH_DISK_DIR}" "${GSBENCH_HDF5_DIR}"

# --- iowarp runtime env -----------------------------------------------------------------
bin_dir="$(dirname "${GSBENCH_BIN}")"
export CLIO_BIND_ADDR="${CLIO_BIND_ADDR:-127.0.0.1}"
export CHI_REPO_PATH="${CHI_REPO_PATH:-${bin_dir}}"
export LD_LIBRARY_PATH="${bin_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# CUDA 12 defaults to CUDA_MODULE_LOADING=LAZY: a kernel's device code is loaded on its FIRST
# launch, and that load is a DEVICE-WIDE SYNCHRONIZING driver op — nothing on any other stream
# dispatches until everything already queued on the GPU drains. The async arm queues ~86 ms of
# compute and then issues its first-ever TwDrainKernel launch, which therefore lands BEHIND that
# whole queue and pins the device for its duration: the bdev worker's D2H copies (and even a 1 KB
# memset) sit unexecuted, so I/O never overlaps compute. Measured: async 165 ms -> 94 ms with
# EAGER, on otherwise unmodified code.
#
# EAGER is set for EVERY arm, not just the CLIO ones, so the treatment is uniform. Verified: no
# other arm's timing moves (raw/hdf5_*/hostclio/sync are flat within run-to-run noise) — they
# never paid this cost, so this is not crediting them for something they don't have. It only
# moves a one-time, per-process module load out of the timed region, which is where a
# process-startup cost belongs.
#
# The benchmark ALSO pre-warms its CLIO kernels in-code (cudaFuncGetAttributes, see the .cu), so a
# direct run of the binary without this script still measures correctly. Under EAGER that pre-warm
# is a no-op.
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"

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
    rm -rf "${logs}" "${GSBENCH_DISK_DIR}" "${GSBENCH_HDF5_DIR}" 2>/dev/null
    rm -f "${GSBENCH_BDEV_PATH}" 2>/dev/null
    [[ -n "${nt1_conf}" ]] && rm -f "${nt1_conf}" 2>/dev/null
}
trap cleanup EXIT

# A stale/killed run leaves a spinning server + shm segments that make the next run hang.
reset_state() { pkill -9 -x kvhdf5_e2e_test 2>/dev/null; rm -f /dev/shm/chi_*; sleep 1; }

run_arm() {  # $1 = catch tag, $2 = label, $3.. = extra VAR=VAL env overrides for THIS arm only
    local tag="$1" label="$2"
    shift 2
    reset_state
    echo ">>> arm: ${label}"
    # HDF5_VOL_CONNECTOR is process-global: if it is exported for the hdf5_async arm it also
    # attaches the async VOL to the PLAIN hdf5 arm's file, quietly costing it ~20% and making
    # the baseline look worse than it is. Strip it for every arm but hdf5_async.
    local -a pre=(env)
    [[ "${label}" == "hdf5_async" ]] || pre+=(-u HDF5_VOL_CONNECTOR)
    # Per-arm overrides. Safe because every arm runs in its OWN process, so forcing e.g.
    # GSBENCH_RAW_INLINE here cannot leak into the `raw` arm's run.
    pre+=("$@")
    timeout "${GSBENCH_TIMEOUT}" "${pre[@]}" "${GSBENCH_BIN}" "${tag}" 2>&1 \
        | grep -E 'GSBENCH_RESULT|GSBENCH_TRACE|\[bench\]|\[raw\]|\[hdf5|FAILED|error' \
        | tee "${logs}/${label}.rep${rep}.log"
    echo "    (exit ${PIPESTATUS[0]})"
}

# The HDF5 baseline is reported as THREE distinct rows, because "HDF5" is not one number —
# the writer structure dominates, and a reviewer will ask for each:
#   hdf5_inline    — synchronous write, GPU idle during it. Matched semantics vs clio-sync
#                    and hostclio (the producer blocks on I/O).
#   hdf5_threaded  — background writer thread, write overlaps the next sim steps. Matched
#                    semantics vs clio-async. THIS IS THE ONE THE PAPER MUST BEAT.
#   hdf5_async     — the HDF5 Asynchronous I/O VOL connector. Measured ~1.9x SLOWER than
#                    hdf5_threaded, so it is due diligence, not a contender.
# The first two are the SAME test case ([gsbench_hdf5]); GSBENCH_RAW_INLINE picks the writer
# structure, and the arm names itself accordingly (see the TEST_CASE in the .cu). Forcing the
# var per-arm is safe: every arm is its own process, so `raw` still honours whatever the
# caller set.
for rep in $(seq 1 "${GSBENCH_REPEATS}"); do
    [[ "${GSBENCH_REPEATS}" -gt 1 ]] && echo "================ REPEAT ${rep}/${GSBENCH_REPEATS} ================"
    run_arm "[gsbench_raw]"      "raw"
    run_arm "[gsbench_hdf5]"     "hdf5_inline"   GSBENCH_RAW_INLINE=1
    run_arm "[gsbench_hdf5]"     "hdf5_threaded" GSBENCH_RAW_INLINE=0
    # The HDF5 async-VOL arm is OPT-IN: it needs a thread-safe libhdf5 + Argobots + the
    # vol-async connector on LD_LIBRARY_PATH/HDF5_PLUGIN_PATH, and it ABORTS rather than
    # silently run synchronously if the connector isn't really loaded. Enable with
    # GSBENCH_HDF5_ASYNC=1 after setting (see the async-VOL build under /opt). Use the
    # nomemcpy build: the stock /opt/vol-async is built ENABLE_WRITE_MEMCPY=ON, whose
    # internal copy path livelocks in Argobots at scale (>=12 in-flight datasets).
    #   export LD_LIBRARY_PATH=/opt/vol-async-nomemcpy/lib:/opt/hdf5ts/lib:/opt/argobots/lib:$LD_LIBRARY_PATH
    #   export HDF5_PLUGIN_PATH=/opt/vol-async-nomemcpy/lib
    #   export HDF5_VOL_CONNECTOR="async under_vol=0;under_info={}"
    if [[ "${GSBENCH_HDF5_ASYNC:-0}" == "1" ]]; then
        run_arm "[gsbench_hdf5_async]" "hdf5_async"
    fi
    run_arm "[gsbench_hostclio]" "hostclio"
    run_arm "[gsbench_gpuh5_sync]"    "gpuh5_sync"
    run_arm "[gsbench_gpuh5_noreuse]" "gpuh5_noreuse"
    run_arm "[gsbench_gpuh5]"         "gpuh5"
    run_arm "[gsbench_persistent]"    "persistent"
done
reset_state

echo
echo "================ BENCHMARK RESULT ================"
cat "${logs}"/*.rep*.log 2>/dev/null | grep GSBENCH_RESULT

# Aggregate ACROSS REPEATS. Each repeat wrote its own <arm>.rep<N>.log, so a run with
# GSBENCH_REPEATS>1 reports the MEDIAN (robust to a single cold-cache outlier) together with the
# spread, instead of silently showing only whichever repeat happened to run last.
python3 - "${logs}" <<'PY' 2>/dev/null || true
import re, sys, glob, os, statistics
logs = sys.argv[1]
ARMS = ("raw", "hdf5_inline", "hdf5_threaded", "hdf5_async", "hostclio", "sync", "async")
PAT = re.compile(r'GSBENCH_RESULT .*arm=(\S+).*MB=([\d.]+) ms=([\d.]+) MBps=([\d.]+) checksum=(\d+)')

rows = {}
for a in ARMS:
    samples, mb, cks = [], None, set()
    for p in sorted(glob.glob(os.path.join(logs, a + ".rep*.log"))):
        m = PAT.search(open(p).read())
        if m:
            samples.append(float(m.group(3)))   # ms
            mb = float(m.group(2))
            cks.add(m.group(5))
    if samples:
        rows[a] = dict(mb=mb, ms=statistics.median(samples), n=len(samples),
                       lo=min(samples), hi=max(samples), cks=cks)

if rows:
    base = rows.get("raw", {}).get("ms")
    print(f"\n{'arm':<15}{'MB':>9}{'ms(med)':>11}{'MB/s':>10}{'vs raw':>9}{'spread':>9}{'n':>4}  checksum")
    for a in ARMS:
        r = rows.get(a)
        if not r:
            continue
        mbps = r['mb'] / (r['ms'] / 1000.0)
        rel = f"{base / r['ms']:.2f}x" if base else "-"
        # spread = (max-min)/median, i.e. how much the repeats disagreed. Large => distrust the row.
        spread = f"{100.0 * (r['hi'] - r['lo']) / r['ms']:.1f}%" if r['n'] > 1 else "-"
        ck = next(iter(r['cks'])) if len(r['cks']) == 1 else "VARIES!"
        print(f"{a:<15}{r['mb']:>9.1f}{r['ms']:>11.2f}{mbps:>10.1f}{rel:>9}{spread:>9}{r['n']:>4}  {ck}")

    # A checksum must be identical across every arm AND every repeat, or the arms did not all
    # compute the same thing and the whole table is meaningless.
    allck = set()
    for r in rows.values():
        allck |= r['cks']
    print("\nchecksums", "MATCH (identical computation)" if len(allck) == 1
          else f"DIFFER {allck}  <-- TABLE IS INVALID")
PY
