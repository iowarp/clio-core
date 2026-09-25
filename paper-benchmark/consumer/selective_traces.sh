#!/bin/bash
# Per-chunk traces for the selective-lossy study (Nyx, VPIC).
#
# Every chunk is compressed with all 32 NeuroPress actions (8 nvCOMP codecs x
# quantize on/off x byte shuffle on/off) under the BALANCED cost model
# (explore-balance: compress + decompress + I/O time, weights 1/1/1). Each
# action is timed in isolation (one chunk's timed GPU work at a time, one
# exploration stream), sized, and -- when quantized -- decoded and compared
# with its source (RMSE, max error, PSNR, SSIM). Same dumps, chunk size, cost
# bandwidth and NUMA pinning as inmem.sh, so a chunk's measured error joins
# onto the in-memory runs of the same data.
#
#   ./selective_traces.sh [-w "nyx vpic"] [--ebs "1e-3 1e-2 1e-4"] [--out DIR]
#
# Environment: REL (eps > 0: value-range-relative bound, traces go to
# <wl>-rel<eps>), THREADS (8 runtime workers), NP_BW (1.2e6 B/ms), NUMA_NODE (the
# GPU's; "none" disables), WORK (/mnt/np-tier2/selective).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$(cd "$HERE/.." && pwd)
DRIVER=$BENCH/nyx/run_config.sh   # replays any field dump; inmem.sh uses it too
BUILD=${BUILD:-$(cd "$BENCH/.." && pwd)/build}
NPE=${NPENV:-$HOME/np-env}
# The binary links the external codecs too; RUNPATH does not reach their deps.
export LD_LIBRARY_PATH="$BUILD/bin:$NPE/cusz/lib:$NPE/cusz/lib64:$NPE/cuszp/lib:$NPE/cuszp/lib64:$NPE/ndzip/lib:/usr/local/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
WLS="nyx vpic" EBS="1e-3 1e-2 1e-4" OUT=$BENCH/results/selective
THREADS=${THREADS:-8} NP_BW=${NP_BW:-1.2e6} CHUNK=8388608
WORK=${WORK:-/mnt/np-tier2/selective}

while [ $# -gt 0 ]; do
  case "$1" in
    -w) WLS=$2; shift 2 ;;
    --ebs) EBS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT" "$WORK"
exec > >(tee -a "$OUT/traces.log") 2>&1

# gpu_numa_node -- the NUMA node GPU 0 is attached to, from its PCI device.
gpu_numa_node() {
  local bus
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 0 2>/dev/null \
        | head -1 | tr 'A-F' 'a-f' | sed 's/^0000//')
  cat "/sys/bus/pci/devices/$bus/numa_node" 2>/dev/null || echo -1
}
NUMA_NODE=${NUMA_NODE:-$(gpu_numa_node)}
PIN=()
if [ "$NUMA_NODE" != none ] && [ "$NUMA_NODE" -ge 0 ] 2>/dev/null; then
  PIN=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
fi

# fields_of <wl> -- the dumps inmem.sh replays for that workload.
fields_of() {
  case "$1" in
    nyx)  echo "${NYX_FIELDS:-/mnt/np-data/nyx-i96/fields}" ;;
    vpic) echo "${VPIC_FIELDS:-/mnt/np-data/vpic-i16/fields}" ;;
    lammps) echo "${LAMMPS_FIELDS:-/mnt/np-data/lammps-b64/fields}" ;;
    warpx) echo "${WARPX_FIELDS:-/mnt/np-tier2/staged/warpx-half/fields}" ;;
  esac
}

# one_trace <wl> <eb> -- one exhaustive, quality-measured trace.
one_trace() {
  local wl=$1 eb=$2 tag dest t0
  tag=$wl-eb$eb
  [ "${REL:-0}" != 0 ] && tag=$wl-rel$REL   # relative mode: eb only enables quantizing
  dest=$OUT/$tag
  rm -rf "${WORK:?}/$tag"
  echo "---- $tag ($(date -u +%H:%M:%S) UTC) ----"
  t0=$(date +%s)
  env CLIO_NEUROPRESS_STAGE_H2D=1 CLIO_NEUROPRESS_ISOLATE_TIMING=1 \
    CLIO_NEUROPRESS_EXPLORE_STREAMS=1 CLIO_REPLAY_FINAL_FLUSH=0 \
    CLIO_NEUROPRESS_COST_W_CT=1 CLIO_NEUROPRESS_COST_W_DT=1 CLIO_NEUROPRESS_COST_W_IO=1 \
    BENCH_NP_LR=0.2 BENCH_NP_MAPE=0.10 MEASURE_DT=1 MEASURE_QUALITY=1 SELECTION_LOG=1 \
    CLIO_NEUROPRESS_REL_BOUND="${REL:-0}" \
    THREADS="$THREADS" "${PIN[@]}" "$DRIVER" explore-balance --fields "$(fields_of "$wl")" \
    --chunk "$CHUNK" --bw "$NP_BW" --eb "$eb" --explore-k 31 --explore-thresh -1 \
    ${MAX_FILES:+--max-files "$MAX_FILES"} --results "$WORK" --tag "$tag" > "$WORK/$tag.log" 2>&1
  echo "   rc=$? after $(( $(date +%s) - t0 )) s"
  mkdir -p "$dest"
  for f in explore.csv selection.csv selection.csv.quality selection.csv.payload \
           blobs.csv stdout.log meta.json compose.yaml; do
    cp "$WORK/$tag/$f" "$dest/" 2>/dev/null
  done
  cp "$WORK/$tag.log" "$dest/driver.log"
  tail -n 300 "$WORK/$tag/runtime.log" > "$dest/runtime_tail.log" 2>/dev/null
  rm -rf "${WORK:?}/$tag"
  python3 - "$dest/explore.csv" <<'PY'
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1])))
per = collections.Counter(r["blob"] for r in rows)
q = [r for r in rows if r["quantize"] == "1"]
print(f"   {len(per)} chunks, {sum(1 for v in per.values() if v == 32)} with all 32 actions; "
      f"{sum(1 for r in q if r['quality_measured'] == '1')} of {len(q)} quantized rows quality-measured")
PY
}

for eb in $EBS; do
  for wl in $WLS; do
    one_trace "$wl" "$eb"
  done
done
