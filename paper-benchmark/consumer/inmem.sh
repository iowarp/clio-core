#!/bin/bash
# In-memory GPU-consumer campaign for the clustering workloads (Nyx, VPIC).
#
# Clio's tier lives in host RAM and the consumer (k-means) runs on the GPU. One
# process per arm and rep -- a RAM tier does not outlive its process:
#   1. the timed write of every field (as cluster.sh writes it),
#   2. READS timed read-backs of the consumer's fields, decoded straight into
#      device memory with WINDOW requests in flight (a prefetching consumer),
#   3. one more read, untimed and sequential, that checks every element against
#      the error bound (lossy arms).
#
#   ./inmem.sh -w nyx|vpic|lammps|warpx [--only L1,L2] [--out DIR] [--work DIR]
#
# Environment: REPS (3), READS (5), WINDOW (8), THREADS (8 runtime workers),
# EB (1e-3), REL (0; eps > 0 quantizes each chunk to eps x its value range --
# EB must still be positive to enable quantization), NP_BW (1.2e6 B/ms: the cost model's bandwidth -- what one read
# request measured from Clio's RAM tier here, ~0.8-1.2 GB/s, the same as
# cluster.sh's NVMe setting), NUMA_NODE (the GPU's; "none" disables), XENV
# (extra KEY=VALUE settings for every run).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$(cd "$HERE/.." && pwd)
BUILD=${BUILD:-$(cd "$BENCH/.." && pwd)/build}
NPE=${NPENV:-$HOME/np-env}
export LD_LIBRARY_PATH="$BUILD/bin:$NPE/cusz/lib:$NPE/cusz/lib64:$NPE/cuszp/lib:$NPE/cuszp/lib64:$NPE/ndzip/lib:/usr/local/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
DRIVER=$BENCH/nyx/run_config.sh
REPS=${REPS:-3} READS=${READS:-5} WINDOW=${WINDOW:-8} THREADS=${THREADS:-8}
EB=${EB:-1e-3} NP_BW=${NP_BW:-1.2e6} CHUNK=8388608 REL=${REL:-0}
WL="" ONLY="" OUT="" WORK=/mnt/np-tier2/inmem

while [ $# -gt 0 ]; do
  case "$1" in
    -w) WL=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
case "$WL" in
  nyx)  SRC=${NYX_FIELDS:-/mnt/np-data/nyx-i96/fields}
        BEST=static-bitcomp-q
        READ_FIELDS="density|xmom|ymom|zmom|rho_e" ;;
  vpic) SRC=${VPIC_FIELDS:-/mnt/np-data/vpic-i16/fields}
        BEST=static-ans-q-s4
        READ_FIELDS="ex|ey|ez|cbx|cby|cbz|jfx|jfy|jfz|rhof" ;;
  # LAMMPS and WarpX (figure 9's best fixed for both): every field is read back.
  lammps) SRC=${LAMMPS_FIELDS:-/mnt/np-data/lammps-b64/fields}
        BEST=static-ans-q-s4
        READ_FIELDS="position|velocity|force" ;;
  warpx) SRC=${WARPX_FIELDS:-/mnt/np-tier2/staged/warpx-half/fields}
        BEST=static-ans-q-s4
        READ_FIELDS="E_x|E_y|E_z|B_x|B_y|B_z|j_x|j_y|j_z|rho" ;;
  *) echo "usage: $0 -w nyx|vpic|lammps|warpx [--only L] [--out DIR]" >&2; exit 2 ;;
esac
OUT=${OUT:-$BENCH/results/inmem/$WL}
mkdir -p "$OUT" "$WORK" && OUT=$(cd "$OUT" && pwd)
exec > >(tee -a "$OUT/run.log") 2>&1

# The GPU's NUMA node, pinned exactly as cluster.sh pins it.
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

ARMS=("Baseline|baseline|0" "Best fixed nvCOMP|$BEST|$EB" "ndzip|static-ndzip|$EB"
      "cuSZp3|static-cuszp|$EB" "cuSZ|static-cusz|$EB" "NeuroPress|learn|$EB"
      ${EXTRA_ARMS:-})

slug() { tr 'A-Z' 'a-z' <<< "$1" | tr -c 'a-z0-9\n' '_' | sed 's/_\+/_/g; s/_$//'; }
wanted() {
  [ -z "$ONLY" ] && return 0
  tr ',' '\n' <<< "$ONLY" | sed 's/^ *//; s/ *$//' | grep -qxF -e "$1" -e "$(slug "$1")"
}

# one_run <label> <cfg> <eb> <rep>: write + READS timed reads + a bound check.
one_run() {
  local label=$1 cfg=$2 eb=$3 rep=$4 tag dest extra
  tag=$(slug "$label")_r$rep dest=$OUT/runs/$(slug "$label")/r$rep
  extra="--verify --read-repeat $READS --read-inflight $WINDOW --read-to-gpu"
  extra+=" --read-match /(fab[0-9]+_comp[0-9]+_)?($READ_FIELDS)/"
  [ "$eb" != 0 ] && extra+=" --check-bound"
  local -a ebarg=(); [ "$eb" != 0 ] && ebarg=(--eb "$eb")
  rm -rf "${WORK:?}/$tag"
  echo "---- $label (rep $rep, $cfg, eb $eb) ----"
  env CLIO_REPLAY_FINAL_FLUSH=0 MEASURE_DT=0 MEASURE_QUALITY=0 SELECTION_LOG=0 \
    CLIO_NEUROPRESS_STAGE_H2D=1 CLIO_NEUROPRESS_PHASE_LOG="$WORK/$tag.phases.csv" \
    CLIO_CUSZ_PHASE_LOG="$WORK/$tag.cusz_setup.csv" BENCH_TIER1_TYPE=ram \
    CLIO_REPLAY_FSYNC= THREADS="$THREADS" \
    CLIO_NEUROPRESS_COST_W_CT=1 CLIO_NEUROPRESS_COST_W_DT=1 CLIO_NEUROPRESS_COST_W_IO=1 \
    BENCH_NP_LR=0.2 BENCH_NP_MAPE=0.10 REPLAY_EXTRA_ARGS="$extra" CLIO_NEUROPRESS_REL_BOUND="$REL" ${XENV:-} \
    "${PIN[@]}" "$DRIVER" "$cfg" --fields "$SRC" --chunk "$CHUNK" --bw "$NP_BW" \
    "${ebarg[@]}" --results "$WORK" --tag "$tag" > "$WORK/$tag.log" 2>&1
  mkdir -p "$dest"
  cp "$WORK/$tag/stdout.log" "$WORK/$tag/blobs.csv" "$WORK/$tag/compose.yaml" "$dest/" 2>/dev/null
  cp "$WORK/$tag.phases.csv" "$dest/phases.csv" 2>/dev/null
  cp "$WORK/$tag.cusz_setup.csv" "$dest/cusz_setup.csv" 2>/dev/null
  tail -n 200 "$WORK/$tag/runtime.log" > "$dest/runtime_tail.log" 2>/dev/null
  rm -rf "${WORK:?}/$tag" "$WORK/$tag".*
  grep -E "^stored |time:|read-back set|get\+decompress|^(VERIFIED|FAILED|BOUND)" "$dest/stdout.log" \
    | sed 's/^ */   /' | cut -c1-160
}

cat > "$OUT/run.json" <<JSON
{"workload":"$WL","source":"$SRC","reps":$REPS,"reads":$READS,"window":$WINDOW,
 "threads":$THREADS,"eb":"$EB","rel":"$REL","np_bw":"$NP_BW","tier":"ram","read_to_gpu":true,
 "read_fields":"$READ_FIELDS","numa_node":"$NUMA_NODE","chunk":$CHUNK}
JSON
for rep in $(seq 1 "$REPS"); do
  for a in "${ARMS[@]}"; do
    IFS='|' read -r label cfg eb <<< "$a"
    wanted "$label" && one_run "$label" "$cfg" "$eb" "$rep"
  done
done
"${PIN[@]}" python3 "$HERE/summarize_inmem.py" "$OUT"
