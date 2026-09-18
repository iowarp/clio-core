#!/usr/bin/env bash
#===============================================================================
# run_campaign.sh -- one workload's prediction-accuracy sweep.
#
#   ./run_campaign.sh <vpic|nyx|warpx|lammps|ai|synth> --out DIR [--local DIR]
#
# Produces, per workload, everything the accuracy table needs on ONE set of
# chunks:
#
#   explore.csv   all 32 candidate configurations MEASURED per chunk (ct, dt,
#                 ratio, and reconstruction quality), beside the NeuroPress
#                 NN's own prediction for each -- the model's inference, not a
#                 replay of it.
#   dist.csv      each chunk's distribution class, HCompress's only
#                 data-dependent input (classify_chunks.py).
#
# Differences from figure_8.sh, and why:
#
#   MEASURE_QUALITY=1  figure_8.sh turns quality measurement OFF, so its runs
#                      have no MEASURED PSNR at all and the table's PSNR column
#                      could only be "n/a". Here it is on, which costs an
#                      inverse chain per candidate per chunk -- hence the
#                      smaller per-workload sizes below.
#   raw bytes          The two in-situ workloads (VPIC, LAMMPS) never write
#                      their buffers anywhere, so their distribution class
#                      cannot be recovered after the fact. Their drivers'
#                      existing dump options (CLIO_VPIC_RAW_DIR, --raw) write
#                      the exact bytes the compressor saw; this script
#                      classifies them IN THE JOB and deletes them, because
#                      they are the run's raw data and are never kept.
#
# Everything else is figure_8.sh's configuration verbatim: explore-balance,
# exploration forced on every chunk (--explore-k 31 --explore-thresh -1) so all
# 32 configurations are measured, eb 0.05, 8 MiB chunks, lr 0.2, SGD above 10%
# cost error, prediction reuse off.
#
# Sizes are cut from figure_8.sh's `full` so each workload fits a 10-minute job
# WITH quality measurement, which roughly doubles the per-chunk cost. The
# exploration log is flushed per row, so a workload that overruns its timeout
# still yields every chunk it finished -- the run reports the count rather than
# failing.
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(cd "$HERE/.." && pwd)

WL=${1:-}; shift || true
case "$WL" in vpic|nyx|warpx|lammps|ai|synth) ;; *)
  sed -n '3,6p' "$0" >&2; exit 2;; esac

OUT="" LOCAL="" 
while [ $# -gt 0 ]; do
  case "$1" in
    --out)   OUT=$2; shift 2;;
    --local) LOCAL=$2; shift 2;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done
[ -n "$OUT" ] || { echo "--out DIR is required" >&2; exit 2; }
mkdir -p "$OUT"

# ---- NeuroPress and measurement configuration (identical for every workload)
CONFIG=explore-balance
NP=(--explore-k 31 --explore-thresh -1 --eb 0.05 --chunk 8388608)
NP_ENV=(BENCH_NP_LR=0.2 BENCH_NP_MAPE=0.10 MEASURE_DT=1 MEASURE_QUALITY=1
        CLIO_NEUROPRESS_EXPLORE_STREAMS=1 CLIO_NEUROPRESS_REUSE_PREDICTIONS=0)

NYX_FIELDS=${NYX_FIELDS:-/projects/bekn/imuradli/np-nyx-256-2000/fields}
WARPX_FIELDS=${WARPX_FIELDS:-/projects/bekn/imuradli/np-warpx-1000-i5/fields}
AI_FIELDS=${AI_FIELDS:-/projects/bekn/imuradli/np-ai-vitb16/fields}
SYNTH_FIELDS=${SYNTH_FIELDS:-/projects/bekn/imuradli/np-hcompress/synth/fields}
WL_TIMEOUT=${WL_TIMEOUT:-540}

run=$OUT; [ -n "$LOCAL" ] && run="$LOCAL/$WL"
mkdir -p "$run"
RAW="$run/raw"

# ---- The workload command, and where its chunk bytes will come from ---------
declare -a CMD; declare -a CLASSIFY
case "$WL" in
  vpic)   # In situ. ~656 chunks (41 dumps x 16 variables at --int 24).
    export CLIO_VPIC_RAW_DIR="$RAW"; mkdir -p "$RAW"
    CMD=( "$BENCH/vpic/run_config_insitu.sh" "$CONFIG" --ncell 126 --steps 1000
          --int 24 "${NP[@]}" --check-bound )
    CLASSIFY=( --raw "$RAW" ) ;;
  lammps) # In situ. ~600 chunks (200 frames x 3 fields at --steps 600).
    CMD=( "$BENCH/lammps/run_config.sh" "$CONFIG" --box 40 --steps 600 --gap 3
          --f32 --require-device --raw "$RAW" "${NP[@]}" )
    CLASSIFY=( --raw "$RAW" ) ;;
  nyx)    # Replay, ~1008 chunks. Bytes stay on disk: no dump needed.
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$NYX_FIELDS" --max-files 0 "${NP[@]}" --check-bound )
    CLASSIFY=( --fields "$NYX_FIELDS" ) ;;
  warpx)  # Replay, one chunk per file: --max-files IS the chunk count.
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$WARPX_FIELDS" --max-files 500 "${NP[@]}" --check-bound )
    CLASSIFY=( --fields "$WARPX_FIELDS" ) ;;
  ai)     # Replay, ~41 chunks per checkpoint file.
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$AI_FIELDS" --max-files 16 "${NP[@]}" --check-bound )
    CLASSIFY=( --fields "$AI_FIELDS" ) ;;
  synth)  # Held-out synthetic data: the class is the generator's, not the
          # classifier's, so the in-distribution block never depends on the two
          # agreeing (gen_synth_fields.py names each frame after its family).
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$SYNTH_FIELDS" --max-files 0 "${NP[@]}" --check-bound )
    CLASSIFY=( --fields "$SYNTH_FIELDS" ) ;;
esac
CMD+=( --results "$run" --tag "$WL" )

echo "== $WL -> $OUT (run dir $run, timeout ${WL_TIMEOUT}s)"
t0=$(date +%s)
timeout -k 5 "$WL_TIMEOUT" env -C "$run" "${NP_ENV[@]}" "${CMD[@]}" \
  > "$run/console.log" 2>&1
rc=$? wall=$(( $(date +%s) - t0 ))
{ [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; } && echo "   TIMEOUT after ${WL_TIMEOUT}s (finished chunks are kept)"

LOG="$run/$WL/explore.csv"
n=0; [ -s "$LOG" ] && n=$(awk -F, 'NR>1 && $4=="primary"' "$LOG" | wc -l)
echo "   rc=$rc  wall ${wall}s  chunks with a pick: $n"
if [ "$n" -eq 0 ]; then echo "   NO CHUNKS -- see $run/console.log"; fi

# ---- The distribution class, while the raw bytes still exist ----------------
if [ "$n" -gt 0 ]; then
  python3 "$HERE/classify_chunks.py" --explore "$LOG" "${CLASSIFY[@]}" \
    --out "$run/$WL/dist.csv" 2>&1 | sed 's/^/   /'
fi

# ---- The run's raw bytes are never kept, in either mode --------------------
rm -rf "$RAW"
find "$run" -maxdepth 3 -type f \( -name chi_bdev.dat -o -name 'cte_tier.dat*' \) -delete 2>/dev/null

if [ "$run" != "$OUT" ]; then
  if tar -C "$run" --exclude='chi_bdev.dat' --exclude='cte_tier.dat*' --exclude='raw' -cf - . \
     | tar -C "$OUT" -xf -; then rm -rf "$run"
  else echo "   COPY FAILED -- the run is still in $run"; fi
fi
exit 0
