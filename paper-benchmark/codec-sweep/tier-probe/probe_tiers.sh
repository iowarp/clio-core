#!/usr/bin/env bash
#SBATCH --job-name=tier-probe
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#SBATCH --output=tier-probe-%j.out
#===============================================================================
# probe_tiers.sh -- measure the bandwidth of each storage tier the cost model
# scores against, on a compute node:
#   dram         GPU -> host DRAM, cudaMemcpy in 4 MiB calls (probe_d2h.cu),
#                pinned and pageable, plus host memcpy for reference
#   nvme         node-local NVMe (/tmp, xfs)
#   burst_buffer /work/nvme, Delta's flash Lustre (no dedicated burst buffer)
#   lustre       /work/hdd, the HDD Lustre PFS
# File tiers: SIZE_GB of random bytes written with dd bs=4M conv=fdatasync
# (durable write), then read back with iflag=direct (no page cache). Reps run
# round-robin over the tiers so a slow minute on the shared Lustre does not
# land on one tier only. Probe files are removed after each measurement.
#
#   sbatch probe_tiers.sh
#
# Environment: SIZE_GB (4), REPS (3), OUT (np-codec-sweep/tier-probe/<stamp>-<job>)
#===============================================================================
set -euo pipefail
ulimit -c 0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  SELF=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2; exit}')
else
  SELF=${BASH_SOURCE[0]}
fi
HERE=$(cd "$(dirname "$SELF")" && pwd)
JOB=${SLURM_JOB_ID:-$$}
SIZE_GB=${SIZE_GB:-4}
REPS=${REPS:-3}
OUT=${OUT:-/projects/bekn/$USER/np-codec-sweep/tier-probe/$(date +%m%d%H%M)-$JOB}
BUILD_DIR=${BUILD_DIR:-$HOME/np-build/tier-probe}
BIN=$BUILD_DIR/probe_d2h
CSV=$OUT/tier_bw.csv
SRC=/dev/shm/tier-probe-src-$JOB
declare -A DIRS=([nvme]=/tmp/tier-probe-$JOB
                 [burst_buffer]=/work/nvme/bekn/$USER/tier-probe-$JOB
                 [lustre]=/work/hdd/bekn/$USER/tier-probe-$JOB)
trap 'rm -rf "$SRC" "${DIRS[@]}"' EXIT

# build -- compile the D2H probe when missing or older than its source; link to
# a private name and rename, so a concurrent job never runs a partial binary.
build() {
  mkdir -p "$BUILD_DIR"
  [ -x "$BIN" ] && [ "$BIN" -nt "$HERE/probe_d2h.cu" ] && return 0
  nvcc -O2 -std=c++17 -arch=sm_80 "$HERE/probe_d2h.cu" -o "$BIN.tmp.$$"
  mv -f "$BIN.tmp.$$" "$BIN"
}

# now_ms -- monotonic-enough wall clock in milliseconds
now_ms() { date +%s%N | awk '{printf "%.3f", $1 / 1e6}'; }

# row TIER METHOD REP BYTES MS -- append one CSV row
row() {
  awk -v t="$1" -v m="$2" -v r="$3" -v b="$4" -v ms="$5" \
    'BEGIN{printf "%s,%s,4194304,%s,%s,%.3f,%.3f\n", t, m, r, b, ms, b / (ms * 1e6)}' >> "$CSV"
}

# probe_file TIER REP -- one durable write and one direct read of SRC on TIER
probe_file() {
  local dir=${DIRS[$1]} f t0 t1 bytes
  f=$dir/probe.bin
  bytes=$(stat -c %s "$SRC")
  t0=$(now_ms); dd if="$SRC" of="$f" bs=4M conv=fdatasync status=none; t1=$(now_ms)
  row "$1" write_fdatasync "$2" "$bytes" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b - a}')"
  t0=$(now_ms); dd if="$f" of=/dev/null bs=4M iflag=direct status=none; t1=$(now_ms)
  row "$1" read_direct "$2" "$bytes" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b - a}')"
  rm -f "$f"
}

mkdir -p "$OUT"
build
hostname > "$OUT/node.txt"
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current \
  --format=csv,noheader >> "$OUT/node.txt" || true
for t in nvme burst_buffer lustre; do
  mkdir -p "${DIRS[$t]}"
  echo "$t ${DIRS[$t]} $(df -T "${DIRS[$t]}" | awk 'NR==2{print $2}')" >> "$OUT/node.txt"
done
cat "$OUT/node.txt"

"$BIN" --chunk 4194304 --total $((2 << 30)) --reps 5 > "$CSV"
head -c "${SIZE_GB}G" /dev/urandom > "$SRC"
for r in $(seq 1 "$REPS"); do
  for t in nvme burst_buffer lustre; do probe_file "$t" "$r"; done
done
column -s, -t "$CSV"
echo "csv: $CSV"
