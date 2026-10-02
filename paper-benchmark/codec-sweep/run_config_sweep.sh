#!/usr/bin/env bash
#SBATCH --job-name=config-sweep
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#SBATCH --output=config-sweep-%j.out
#===============================================================================
# run_config_sweep.sh -- every lossless codec setting (corpus_grid.py: all
# nvcomp options at 3 internal chunk sizes, ndzip 1-D/2-D/3-D, GPULZ, SPspeed,
# SPratio), each with shuffle none / byte / bit, on K evenly spaced 4 MiB
# chunks of one workload. One corpus_sweep row per (chunk, setting): bytes,
# median compress / decompress ms over REPS, bit-exact round trip.
#
#   WL=nyx sbatch run_config_sweep.sh
#
# Environment:
#   WL        workload: nyx vpic hurricane cesm-atm cesm-atm-3d scale-letkf
#             qmcpack (picks its files and ndzip shapes)                 (nyx)
#   K         chunks to sample, evenly spaced over every (file, chunk)    (32)
#   REPS      timed round trips per (chunk, setting)                      (3)
#   SHUFFLES  comma list for corpus_grid.py --shuffle         (none,byte,bit)
#   GRID      all | main | slow: slow = deflate/gdeflate level 4-5, which
#             take 0.5-3 s per 4 MiB round trip; main = everything else (all)
#   SETTINGS  spec file to use instead of the generated grid
#   LIST      chunk list ("path k" lines) to use instead of sampling
#   DIR       directory LIST is relative to (default: the workload's source)
#   OUT       results dir (np-codec-sweep/config/<wl>-<stamp>-<job>)
#   NPENV, BUILD_DIR  as in run_codec_sweep.sh
#===============================================================================
set -euo pipefail
ulimit -c 0
if [ -n "${SLURM_JOB_ID:-}" ]; then
  SELF=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2; exit}')
else
  SELF=${BASH_SOURCE[0]}
fi
HERE=$(cd "$(dirname "$SELF")" && pwd)
WL=${WL:-nyx}
K=${K:-32}
REPS=${REPS:-3}
SHUFFLES=${SHUFFLES:-none,byte,bit}
GRID=${GRID:-all}
CHUNK=4194304
NPENV=${NPENV:-$HOME/np-env}
BUILD_DIR=${BUILD_DIR:-$HOME/np-build/codec-sweep}
BIN=$BUILD_DIR/corpus_sweep
SRC=$HERE/corpus_sweep.cu
FULL=/projects/bekn/$USER/np-codec-sweep/full
OUT=${OUT:-/projects/bekn/$USER/np-codec-sweep/config/$WL-$(date +%m%d%H%M)-${SLURM_JOB_ID:-local}}

# wl_info WL -- "source_dir ndzip_shapes" (shapes: fastest dims, slowest
# first; 2-D = one row, 3-D = one plane; none where a plane exceeds a chunk)
wl_info() {
  case "$1" in
    nyx)         echo "/work/hdd/bekn/$USER/np-dumps/nyx/fields 128,128x128" ;;
    vpic)        echo "/work/hdd/bekn/$USER/np-dumps/vpic/fields 128,128x128" ;;
    hurricane)   echo "/projects/bekn/$USER/sdrbench/Hurricane-ISABEL-series 500,500x500" ;;
    cesm-atm)    echo "/projects/bekn/$USER/sdrbench/CESM-ATM 3600" ;;
    cesm-atm-3d) echo "/projects/bekn/$USER/sdrbench/CESM-ATM-3D 3600" ;;
    scale-letkf) echo "/projects/bekn/$USER/sdrbench/SCALE-LETKF 1200" ;;
    qmcpack)     echo "/projects/bekn/$USER/sdrbench/QMCPACK 69,69x69" ;;
    *) return 1 ;;
  esac
}

# build -- compile corpus_sweep when missing or older than its sources;
# link to a private name and rename so a concurrent job never sees half.
build() {
  mkdir -p "$BUILD_DIR"
  [ -x "$BIN" ] && [ "$BIN" -nt "$SRC" ] && [ "$BIN" -nt "$HERE/gpu_codecs.cuh" ] && return 0
  local tmp=$BIN.tmp.$$
  nvcc -O2 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
    -I"$NPENV/np/include" -I"$NPENV/ndzip/include" -I"$NPENV/gpulz/include" \
    -I"$NPENV/fpcompress/include" "$SRC" -o "$tmp" \
    -L"$NPENV/np/lib" -L"$NPENV/ndzip/lib" -L"$NPENV/gpulz/lib" \
    -L"$NPENV/fpcompress/lib" -lnvcomp -lndzip-cuda -lndzip -lgpulz -lfpcompress \
    -Xlinker -rpath="$NPENV/np/lib:$NPENV/ndzip/lib:$NPENV/gpulz/lib:$NPENV/fpcompress/lib"
  mv -f "$tmp" "$BIN"
}

# sample_chunks SRC FILES K -- K "path k" lines evenly spaced over every
# 4 MiB chunk of the listed files (in list order)
sample_chunks() {
  python3 - "$1" "$2" "$3" "$CHUNK" <<'EOF'
import os, sys
src, files, k, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
pairs = []
for f in open(files).read().split():
    n = -(-os.path.getsize(os.path.join(src, f)) // chunk)
    pairs += [(f, i) for i in range(n)]
k = min(k, len(pairs))
for j in range(k):
    f, i = pairs[j * (len(pairs) - 1) // max(k - 1, 1)]
    print(f, i)
EOF
}

read -r src shapes < <(wl_info "$WL") || { echo "unknown workload $WL" >&2; exit 2; }
DIR=${DIR:-$src}
mkdir -p "$OUT"
build
if [ -n "${LIST:-}" ]; then
  [ "$LIST" -ef "$OUT/chunks.txt" ] || cp "$LIST" "$OUT/chunks.txt"
else
  full=$(ls -d "$FULL/$WL"-full-* | tail -1)
  sample_chunks "$DIR" "$full/rep1/files.txt" "$K" > "$OUT/chunks.txt"
fi
if [ -n "${SETTINGS:-}" ]; then
  [ "$SETTINGS" -ef "$OUT/settings.txt" ] || cp "$SETTINGS" "$OUT/settings.txt"
else
  args=(--shuffle "$SHUFFLES")
  for s in ${shapes//,/ }; do args+=(--ndzip-shape "$s"); done
  slow='^(gdeflate|deflate) .*level=[45]'
  case "$GRID" in
    all)  python3 "$HERE/corpus_grid.py" settings "${args[@]}" ;;
    main) python3 "$HERE/corpus_grid.py" settings "${args[@]}" | grep -Ev "$slow" ;;
    slow) python3 "$HERE/corpus_grid.py" settings "${args[@]}" | grep -E "$slow" ;;
    *) echo "GRID must be all, main or slow" >&2; exit 2 ;;
  esac > "$OUT/settings.txt"
fi
echo "$WL: $(wc -l < "$OUT/chunks.txt") chunks from $DIR, $(wc -l < "$OUT/settings.txt") settings, reps $REPS -> $OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
t0=$(date +%s)
# corpus_sweep exits 4 when any row fails (e.g. ndzip refusing a size that is
# not whole floats); keep going and report it, the rows say which.
rc=0
"$BIN" --dir "$DIR" --list "$OUT/chunks.txt" --configs "$OUT/settings.txt" \
  --out "$OUT/configs.csv" --reps "$REPS" --chunk "$CHUNK" > "$OUT/progress.log" 2>&1 || rc=$?
tail -n 3 "$OUT/progress.log"
echo "sweep rc=$rc, $(( $(date +%s) - t0 )) s; csv: $OUT/configs.csv"
