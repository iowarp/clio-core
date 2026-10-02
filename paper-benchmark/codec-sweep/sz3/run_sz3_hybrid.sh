#!/usr/bin/env bash
#SBATCH --job-name=sz3-hybrid
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=1 --mem=96g
#SBATCH --time=00:10:00
#SBATCH --output=sz3-hybrid-%j.out
#===============================================================================
# run_sz3_hybrid.sh -- sz3_hybrid.py on one workload's dump series: every
# window of T dumps, every field, every 4 MiB chunk, SZ3 in every
# configuration as 3-D per dump and 4-D over the window (CPU, one thread per
# chunk, WORKERS chunks at a time).
#
#   WL=nyx sbatch run_sz3_hybrid.sh
#
# Environment: WL (nyx|vpic), T (4), WINDOWS (0 = all), REL (1e-3),
#              WORKERS (32), OUT (np-codec-sweep/sz3-hybrid/<wl>-<stamp>-<job>)
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
T=${T:-4}
WINDOWS=${WINDOWS:-0}
REL=${REL:-1e-3}
WORKERS=${WORKERS:-32}
ROOT=/work/hdd/bekn/$USER/np-dumps/$WL/fields
FULL=/projects/bekn/$USER/np-codec-sweep/full
OUT=${OUT:-/projects/bekn/$USER/np-codec-sweep/sz3-hybrid/$WL-$(date +%m%d%H%M)-${SLURM_JOB_ID:-local}}
DRIVER=$HOME/np-build/sz3-hybrid/sz3_window

# The driver is header-only SZ3 + system zstd; rebuild when the source is newer.
if [ ! -x "$DRIVER" ] || [ "$HERE/sz3_window.cpp" -nt "$DRIVER" ]; then
  mkdir -p "$(dirname "$DRIVER")"
  g++ -O3 -march=x86-64-v3 -std=c++17 -fopenmp -I"$HOME/np-env/sz3/include" \
    "$HERE/sz3_window.cpp" -o "$DRIVER.tmp.$$" -lzstd
  mv -f "$DRIVER.tmp.$$" "$DRIVER"
fi
mkdir -p "$OUT"
full=$(ls -d "$FULL/$WL"-full-* | tail -1)
cp "$full/rep1/files.txt" "$OUT/files.txt"
echo "$WL: T=$T windows=$WINDOWS rel=$REL workers=$WORKERS -> $OUT"
python3 "$HERE/sz3_hybrid.py" --wl "$WL" --root "$ROOT" --files "$OUT/files.txt" \
  --out "$OUT/sz3.csv" --T "$T" --windows "$WINDOWS" --rel "$REL" \
  --workers "$WORKERS"
echo "csv: $OUT/sz3.csv"
