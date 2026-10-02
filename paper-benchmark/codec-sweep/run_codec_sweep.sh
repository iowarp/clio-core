#!/usr/bin/env bash
#SBATCH --job-name=codec-sweep
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#SBATCH --output=codec-sweep-%j.out
#===============================================================================
# run_codec_sweep.sh -- compress a workload's field files in fixed-size chunks
# with every nvcomp codec, ndzip, GPULZ, SPspeed and SPratio (no preprocessing,
# no clio) and write one CSV row per (chunk, codec): compressed bytes, ratio,
# compress/decompress ms, and whether the round trip was bit-exact.
#
#   sbatch run_codec_sweep.sh                         # every Nyx file
#   WL=vpic sbatch run_codec_sweep.sh                 # every VPIC file
#   MAX_GB=1 sbatch --time=00:05:00 run_codec_sweep.sh  # first 1 GiB only
#
# Environment:
#   WL            workload name, names the dump dir and the CSV   (nyx)
#   FIELDS        dump directory          (/work/hdd/bekn/$USER/np-dumps/$WL/fields)
#   OUT           results directory       (np-codec-sweep/{smoke,full}/$WL-<stamp>; smoke when MAX_GB > 0)
#   CHUNK         bytes per chunk         (4194304)
#   MAX_GB        stop after this many GiB of input, 0 = all   (0)
#   CODECS        comma list, default all twelve
#   NPENV         codec prefixes          ($HOME/np-env)
#   BUILD_DIR     where the binary lives  ($HOME/np-build/codec-sweep)
#   STAGE         1 = copy the inputs to node-local /tmp first (1)
#   STAGE_STREAMS parallel copies         (16)
#===============================================================================
set -euo pipefail
# No core dumps: a crashing codec on a 40 GB GPU writes ~37 GB of core into the
# results folder (four did on 2026-10-02).
ulimit -c 0

# Under sbatch, $0 is a spool copy; find the real script through scontrol.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  SELF=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2; exit}')
else
  SELF=${BASH_SOURCE[0]}
fi
HERE=$(cd "$(dirname "$SELF")" && pwd)

WL=${WL:-nyx}
FIELDS=${FIELDS:-/work/hdd/bekn/$USER/np-dumps/$WL/fields}
MAX_GB=${MAX_GB:-0}
# Smoke runs (MAX_GB > 0) and full runs land in separate trees.
KIND=$(awk -v g="$MAX_GB" 'BEGIN{print (g > 0) ? "smoke" : "full"}')
OUT=${OUT:-/projects/bekn/$USER/np-codec-sweep/$KIND/$WL-$(date +%m%d%H%M)-${SLURM_JOB_ID:-local}}
CHUNK=${CHUNK:-4194304}
CODECS=${CODECS:-}
NPENV=${NPENV:-$HOME/np-env}
BUILD_DIR=${BUILD_DIR:-$HOME/np-build/codec-sweep}
STAGE=${STAGE:-1}
STAGE_STREAMS=${STAGE_STREAMS:-16}
BIN=$BUILD_DIR/codec_sweep
SRC=$HERE/codec_sweep.cu
CSV=$OUT/${WL}_chunk$((CHUNK >> 20))m.csv

# build -- compile the benchmark out of tree when the binary is missing or older
# than its source. Libraries are found through RPATH, not LD_LIBRARY_PATH. It
# links to a private name and renames, so concurrent jobs never see a partial
# binary.
build() {
  mkdir -p "$BUILD_DIR"
  [ -x "$BIN" ] && [ "$BIN" -nt "$SRC" ] && [ "$BIN" -nt "$HERE/gpu_codecs.cuh" ] && return 0
  echo "building $BIN"
  local tmp=$BIN.tmp.$$
  nvcc -O2 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
    -I"$NPENV/np/include" -I"$NPENV/ndzip/include" -I"$NPENV/gpulz/include" \
    -I"$NPENV/fpcompress/include" "$SRC" -o "$tmp" \
    -L"$NPENV/np/lib" -L"$NPENV/ndzip/lib" -L"$NPENV/gpulz/lib" \
    -L"$NPENV/fpcompress/lib" -lnvcomp -lndzip-cuda -lndzip -lgpulz -lfpcompress \
    -Xlinker -rpath="$NPENV/np/lib:$NPENV/ndzip/lib:$NPENV/gpulz/lib:$NPENV/fpcompress/lib"
  mv -f "$tmp" "$BIN"
}

# make_list -- the .f32 files in name order, cut at MAX_GB of input.
make_list() {
  (cd "$FIELDS" && find . -name '*.f32' -type f -printf '%P\t%s\n') |
    LC_ALL=C sort |
    awk -F'\t' -v max="$MAX_GB" '
      max > 0 && sum >= max * 1073741824 { next }
      { sum += $2; print $1 }' > "$OUT/files.txt"
}

# stage -- copy the listed files to node-local /tmp with parallel streams and
# point DIR there, so the sweep never waits on the shared filesystem. Refuses a
# network filesystem or one without room for the inputs (exit 4).
stage() {
  STAGED=/tmp/codec-sweep-${SLURM_JOB_ID:-$$}
  trap 'rm -rf "$STAGED"' EXIT
  mkdir -p "$STAGED"
  local t0 t1 want got fs need avail
  fs=$(df -T "$STAGED" | awk 'NR==2{print $2}')
  case "$fs" in lustre|nfs|nfs4|gpfs|beegfs|cifs|ceph|fuse.*)
    echo "stage dir $STAGED is on $fs, not node-local" >&2; exit 4 ;; esac
  need=$(cd "$FIELDS" && xargs -a "$OUT/files.txt" stat -c %s | awk '{s+=$1} END{printf "%.0f", s}')
  avail=$(df -B1 --output=avail "$STAGED" | tail -1)
  if awk -v a="$avail" -v n="$need" 'BEGIN{exit !(a < n * 1.05)}'; then
    echo "stage dir $STAGED: $avail B free, $need B needed" >&2; exit 4
  fi
  # Create every directory first: parallel `cp --parents` race on a shared
  # parent ("cannot make directory ... File exists") and drop files.
  (cd "$STAGED" && awk -F/ 'NF > 1 { NF--; print }' OFS=/ "$OUT/files.txt" | sort -u | xargs -r mkdir -p)
  t0=$(date +%s.%N)
  (cd "$FIELDS" && xargs -a "$OUT/files.txt" -P "$STAGE_STREAMS" -n 16 \
     cp --parents -t "$STAGED")
  t1=$(date +%s.%N)
  want=$(wc -l < "$OUT/files.txt")
  got=$(find "$STAGED" -type f | wc -l)
  [ "$got" -eq "$want" ] || { echo "staged $got of $want files" >&2; exit 5; }
  echo "staged $got files in $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.0f", (b-a)*1000}') ms"
  DIR=$STAGED
}

mkdir -p "$OUT"
build
make_list
echo "$(wc -l < "$OUT/files.txt") files from $FIELDS, chunk $CHUNK B -> $OUT"
DIR=$FIELDS
[ "$STAGE" = 1 ] && stage
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

"$BIN" --dir "$DIR" --list "$OUT/files.txt" --chunk "$CHUNK" \
  --out "$CSV" ${CODECS:+--codecs "$CODECS"} 2>&1 | tee "$OUT/summary.txt"
echo "csv: $CSV"
