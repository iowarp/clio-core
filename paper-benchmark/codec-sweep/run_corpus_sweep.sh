#!/usr/bin/env bash
#SBATCH --job-name=corpus-sweep
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:10:00
#===============================================================================
# run_corpus_sweep.sh -- measure lossless GPU codec settings on the synthetic
# float32 corpus with corpus_sweep (one row per file x setting, median of REPS
# timed round trips, every one checked bit for bit) and write one shard's raw
# CSV. build_corpus_csv.py turns the shards into the 600k-style CSV.
#
#   FILES=f.txt SETTINGS=s.txt OUT=dir sbatch -o dir/%j.log run_corpus_sweep.sh
#
# Environment:
#   FILES      file list, names relative to CORPUS              (required)
#   SETTINGS   codec specs, one per line (corpus_grid.py)       (required)
#   OUT        results directory; gets raw_<SHARD>.csv          (required)
#   SHARD      name of this job's CSV           (${SLURM_JOB_ID:-local})
#   CORPUS     input directory          (/projects/bekn/$USER/synthetic-9800)
#   REPS       timed round trips per row                        (3)
#   NPENV      codec prefixes                                   ($HOME/np-env)
#   BUILD_DIR  where the binary lives          ($HOME/np-build/codec-sweep)
#===============================================================================
set -euo pipefail
ulimit -c 0  # a crashing codec on a 40 GB GPU writes a ~37 GB core

# Under sbatch, $0 is a spool copy; find the real script through scontrol.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  SELF=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2; exit}')
else
  SELF=${BASH_SOURCE[0]}
fi
HERE=$(cd "$(dirname "$SELF")" && pwd)

: "${FILES:?set FILES}" "${SETTINGS:?set SETTINGS}" "${OUT:?set OUT}"
SHARD=${SHARD:-${SLURM_JOB_ID:-local}}
CORPUS=${CORPUS:-/projects/bekn/$USER/synthetic-9800}
REPS=${REPS:-3}
NPENV=${NPENV:-$HOME/np-env}
BUILD_DIR=${BUILD_DIR:-$HOME/np-build/codec-sweep}
BIN=$BUILD_DIR/corpus_sweep
SRC=$HERE/corpus_sweep.cu

# build -- compile out of tree when the binary is missing or older than its
# sources. Libraries are found through RPATH. It links to a private name and
# renames, so concurrent jobs never see a partial binary.
build() {
  mkdir -p "$BUILD_DIR"
  if [ -x "$BIN" ] && [ "$BIN" -nt "$SRC" ] && [ "$BIN" -nt "$HERE/gpu_codecs.cuh" ]; then
    return 0
  fi
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

mkdir -p "$OUT"
build
# Keep what this shard ran next to its rows.
cp "$FILES" "$OUT/files_$SHARD.txt"
cp "$SETTINGS" "$OUT/settings_$SHARD.txt"
echo "$(wc -l < "$FILES") files x $(grep -cv '^\s*\(#\|$\)' "$SETTINGS") settings," \
     "$REPS reps, from $CORPUS -> $OUT/raw_$SHARD.csv"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader || true

"$BIN" --dir "$CORPUS" --list "$FILES" --configs "$SETTINGS" \
  --out "$OUT/raw_$SHARD.csv" --reps "$REPS"
