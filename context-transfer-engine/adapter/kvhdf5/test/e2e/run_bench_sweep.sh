#!/usr/bin/env bash
#
# Parameter sweeps over the four-way Gray-Scott I/O benchmark, for the advisor's
# scaling study. Three axes (see gray_scott_threeway_bench.cu for the knobs):
#
#   A1  perf vs number of GPU blocks == chunks (one CUDA block per chunk):
#         vary GSBENCH_CHUNKS in {1,2,4,8,16,32}, GSBENCH_SUBMIT_BLOCKS = chunks.
#   A2  perf vs number of GPU (submit) blocks at FIXED chunks (isolates the
#         submission-parallelism effect from I/O-unit count): chunks=32,
#         vary GSBENCH_SUBMIT_BLOCKS in {1,2,4,8,16,32}.
#   B   perf vs per-blob transfer size at fixed parallelism (chunks=16 blocks=16):
#         vary N in {128,512,724,2048} -> per-blob {4KB,64KB,~128KB,1MB}.
#   C   perf vs compute (steps) at fixed 1MB blobs + large parallelism
#         (chunks=16 blocks=16, N=2048): vary GSBENCH_STEPS_PER in {24,48,96,192,384}.
#
# Each point runs up to 5 arms, each in its OWN process (iowarp per-process backend
# ceiling): raw / hostclio / sync / async / async_pinned. async_pinned == async with
# GSBENCH_DATA_PINNED=1 (blob data in pinned host memory). A2 runs only the three
# GPU-producer arms (raw/hostclio ignore GSBENCH_SUBMIT_BLOCKS).
#
# Storage = durable disk (kFile bdev, fsync'd raw), num_threads=1, incompressible.
# Run INSIDE the clio-core devcontainer after building kvhdf5_e2e_tests. Emits a CSV.
#
set -u

BIN="${GSBENCH_BIN:-/workspace/build/bin/kvhdf5_e2e_tests}"
BINDIR="$(dirname "$BIN")"
SCRATCH="${GSBENCH_SWEEP_SCRATCH:-/workspace/build/gsbench_sweep_scratch}"
OUT="${GSBENCH_SWEEP_CSV:-$SCRATCH/results.csv}"
RAWLOG="$SCRATCH/raw_results.log"

if [[ ! -x "$BIN" ]]; then echo "error: no binary at $BIN" >&2; exit 1; fi
mkdir -p "$SCRATCH/raw_out"

# iowarp runtime env
export CLIO_BIND_ADDR=127.0.0.1
export CHI_REPO_PATH="$BINDIR"
export LD_LIBRARY_PATH="$BINDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CONF="$(mktemp)"; printf 'runtime:\n  num_threads: 1\n' > "$CONF"; export CLIO_SERVER_CONF="$CONF"

# fixed benchmark env (durable-disk comparison)
export GSBENCH_BDEV=file
export GSBENCH_BDEV_CAP_MB="${GSBENCH_BDEV_CAP_MB:-3072}"
export GSBENCH_BDEV_PATH="$SCRATCH/clio_bdev.dat"
export GSBENCH_DISK_DIR="$SCRATCH/raw_out"
export GSBENCH_INCOMPRESSIBLE=1
export GSBENCH_RAW_ODIRECT=0
export GSBENCH_RAW_FSYNC=1

: > "$RAWLOG"

reset_state() {
    pkill -9 -x kvhdf5_e2e_tests 2>/dev/null
    rm -f /dev/shm/chi_* "$GSBENCH_BDEV_PATH" 2>/dev/null
    sleep 1
}

# run_point <sweep_label> <arm> : arm in raw|hostclio|sync|async|async_pinned.
# All GSBENCH_* config for the point must already be exported by the caller.
run_point() {
    local sweep="$1" arm="$2" tag pin
    case "$arm" in
        raw)          tag='[gsbench_raw]';      pin=0 ;;
        hostclio)     tag='[gsbench_hostclio]'; pin=0 ;;
        sync)         tag='[gsbench_sync]';     pin=0 ;;
        async)        tag='[gsbench_async]';    pin=0 ;;
        async_pinned) tag='[gsbench_async]';    pin=1 ;;
        *) echo "bad arm $arm" >&2; return 1 ;;
    esac
    reset_state
    local line
    line="$( cd "$BINDIR" && GSBENCH_DATA_PINNED="$pin" timeout 600 "$BIN" "$tag" 2>&1 \
             | grep -m1 'GSBENCH_RESULT' )"
    if [[ -z "$line" ]]; then
        echo "!!! $sweep arm=$arm FAILED/empty (N=${GSBENCH_N:-} chunks=${GSBENCH_CHUNKS:-} blocks=${GSBENCH_SUBMIT_BLOCKS:-} steps_per=${GSBENCH_STEPS_PER:-})"
        echo "SWEEP=$sweep armlabel=$arm FAILED" >> "$RAWLOG"
        return
    fi
    echo "SWEEP=$sweep armlabel=$arm $line" >> "$RAWLOG"
    echo ">>> $sweep [$arm] ${line#*GSBENCH_RESULT }"
}

ALL_ARMS=(raw hostclio sync async async_pinned)
GPU_ARMS=(sync async async_pinned)

echo "==================== SWEEP A1: GPU blocks == chunks (one block/chunk) ===================="
export GSBENCH_N=4096 GSBENCH_SNAPS=8 GSBENCH_STEPS_PER=96
for ch in 1 2 4 8 16 32; do
    export GSBENCH_CHUNKS=$ch GSBENCH_SUBMIT_BLOCKS=$ch
    echo "---- A1 chunks=$ch blocks=$ch ----"
    for a in "${ALL_ARMS[@]}"; do run_point A1 "$a"; done
done

echo "==================== SWEEP A2: submit blocks at fixed chunks=32 ===================="
export GSBENCH_N=4096 GSBENCH_SNAPS=8 GSBENCH_STEPS_PER=96 GSBENCH_CHUNKS=32
for b in 1 2 4 8 16 32; do
    export GSBENCH_SUBMIT_BLOCKS=$b
    echo "---- A2 chunks=32 blocks=$b ----"
    for a in "${GPU_ARMS[@]}"; do run_point A2 "$a"; done
done

echo "==================== SWEEP B: transfer size at chunks=16 blocks=16 ===================="
export GSBENCH_SNAPS=8 GSBENCH_STEPS_PER=96 GSBENCH_CHUNKS=16 GSBENCH_SUBMIT_BLOCKS=16
for n in 128 512 724 2048; do
    export GSBENCH_N=$n
    echo "---- B N=$n (per-blob ~$(( n*n*4/16 )) B) ----"
    for a in "${ALL_ARMS[@]}"; do run_point B "$a"; done
done

echo "==================== SWEEP C: steps at 1MB blobs, chunks=16 blocks=16 ===================="
export GSBENCH_N=2048 GSBENCH_SNAPS=8 GSBENCH_CHUNKS=16 GSBENCH_SUBMIT_BLOCKS=16
for s in 24 48 96 192 384; do
    export GSBENCH_STEPS_PER=$s
    echo "---- C steps_per=$s ----"
    for a in "${ALL_ARMS[@]}"; do run_point C "$a"; done
done

reset_state
rm -f "$CONF"

# ---- build CSV from the raw log ----
python3 - "$RAWLOG" "$OUT" <<'PY'
import re, sys
raw, out = sys.argv[1], sys.argv[2]
cols = ["sweep","armlabel","arm","N","chunks","blocks","snaps","steps","bdev","pinned","MB","ms","MBps","checksum"]
rows = []
for ln in open(raw):
    m = re.search(r'SWEEP=(\S+)\s+armlabel=(\S+)\s', ln)
    if not m: continue
    sweep, armlabel = m.group(1), m.group(2)
    if "FAILED" in ln and "GSBENCH_RESULT" not in ln:
        rows.append({"sweep":sweep,"armlabel":armlabel,"arm":"FAILED"})
        continue
    kv = dict(re.findall(r'(\w+)=([-\d.]+|\w+)', ln.split("GSBENCH_RESULT",1)[1]))
    kv["sweep"]=sweep; kv["armlabel"]=armlabel
    rows.append(kv)
with open(out,"w") as f:
    f.write(",".join(cols)+"\n")
    for r in rows:
        f.write(",".join(str(r.get(c,"")) for c in cols)+"\n")
print(f"\nwrote {len(rows)} rows -> {out}")
PY

echo "==================== DONE. CSV at $OUT ===================="
cat "$OUT"
