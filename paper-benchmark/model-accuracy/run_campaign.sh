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
#   every workload     is a REPLAY. VPIC and LAMMPS used to run in situ, which
#                      made every campaign pay for the simulation and left
#                      their distribution class unrecoverable after the fact.
#                      jobs/fig8_dump.sbatch now runs each simulation ONCE
#                      through the drivers' own dump options
#                      (CLIO_VPIC_RAW_DIR, --raw) and keeps the bytes, so a
#                      campaign replays them like Nyx, WarpX and AI. The dumps
#                      carry the emission index in the file name, so the
#                      replay's chunk order is the simulation's.
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
# EXPLORE_K is the RANKED WINDOW: the runtime measures its top-K predicted
# candidates plus the primary, so K=8 measures 9 of 32 and K=31 measures all 32.
# Two consequences, because both bear on what the figure can claim:
#   - regret is scored against the best MEASURED candidate, so below 31 it is
#     regret against the explored subset, not against the true optimum;
#   - the subset is NeuroPress's own top-K, so every baseline is confined to a
#     shortlist NeuroPress drew up.
# K=31 is the action-space setting; smaller K is closer to a production run.
EXPLORE_K=${EXPLORE_K:-8}
NP=(--explore-k "$EXPLORE_K" --explore-thresh -1 --eb 0.05 --chunk 8388608)
NP_ENV=(BENCH_NP_LR=0.2 BENCH_NP_MAPE=0.10 MEASURE_DT=1 MEASURE_QUALITY=1
        CLIO_NEUROPRESS_EXPLORE_STREAMS=1 CLIO_NEUROPRESS_REUSE_PREDICTIONS=0
        # Every online-learning path on. The ratio head is OFF by default
        # (compressor_runtime.cc:1652 opts IN on "1"), so without this the
        # cost model learns its two time heads and never its ratio head --
        # and ratio is what drives the I/O term of the cost.
        CLIO_NEUROPRESS_SGD_ON_RATIO=1
        # Pinned, not left to the source default: the explore SGD keeps the
        # 7-N cheapest plus the N most mispredicted, and a campaign should not
        # silently change because that default moved.
        CLIO_NEUROPRESS_EXPLORE_SGD_HARD=3)

# Dumped once by jobs/fig8_dump.sbatch. The file names carry the emission
# index, because the replay driver sorts full paths
# (neuropress_field_replay.cc:452) and the drivers flatten '/' to '_' -- without
# the index a dump replays field-major, and LAMMPS's unpadded step_0/step_3/
# step_12 would not even sort numerically. Chunk order IS panel (d)'s x axis.
# Machine-specific paths (gitignored); sourced after the arg loop, so $SIZE is known.
SITE=${SITE:-$HERE/../site.sh}
# shellcheck source=/dev/null
[ -f "$SITE" ] && . "$SITE"

VPIC_FIELDS=${VPIC_FIELDS:-/work/hdd/bekn/imuradli/np-fig8-vpic/fields}
LAMMPS_FIELDS=${LAMMPS_FIELDS:-/work/hdd/bekn/imuradli/np-fig8-lammps/fields}
NYX_FIELDS=${NYX_FIELDS:-/projects/bekn/imuradli/np-nyx-256-2000/fields}
WARPX_FIELDS=${WARPX_FIELDS:-/projects/bekn/imuradli/np-warpx-1000-i5/fields}
AI_FIELDS=${AI_FIELDS:-/projects/bekn/imuradli/np-ai-vitb16/fields}
SYNTH_FIELDS=${SYNTH_FIELDS:-/projects/bekn/imuradli/np-hcompress/synth/fields}
# Chunk counts for the two workloads whose dumps are shared with other figures
# and so cannot be sized by trimming the dump itself. WarpX is one chunk per
# file; AI is ~41 per checkpoint file.
WARPX_FILES=${WARPX_FILES:-500}
AI_FILES=${AI_FILES:-16}
WL_TIMEOUT=${WL_TIMEOUT:-540}

run=$OUT; [ -n "$LOCAL" ] && run="$LOCAL/$WL"
mkdir -p "$run"

# ---- The workload command, and where its chunk bytes will come from ---------
declare -a CMD; declare -a CLASSIFY
case "$WL" in
  vpic)   # Replay, 656 chunks. Was in situ; the simulation is now run ONCE by
          # jobs/fig8_dump.sbatch and every campaign replays its bytes, so no
          # campaign pays for VPIC again. The dump is the driver's own
          # CLIO_VPIC_RAW_DIR output, byte-identical to what the in-situ path
          # handed the compressor (job 22195506: 656 files, 0 missing, 0 wrong
          # size). Each file is exactly one 8 MiB chunk.
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$VPIC_FIELDS" --max-files 0 "${NP[@]}" --check-bound )
    CLASSIFY=( --fields "$VPIC_FIELDS" ) ;;
  lammps) # Replay, 603 chunks, same story (job 22195507). Chunks are 3,072,000 B
          # here, under --chunk, so one chunk per file as in situ.
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$LAMMPS_FIELDS" --max-files 0 "${NP[@]}" --check-bound )
    CLASSIFY=( --fields "$LAMMPS_FIELDS" ) ;;
  nyx)    # Replay, ~1008 chunks. Bytes stay on disk: no dump needed.
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$NYX_FIELDS" --max-files 0 "${NP[@]}" --check-bound )
    CLASSIFY=( --fields "$NYX_FIELDS" ) ;;
  warpx)  # Replay, one chunk per file: --max-files IS the chunk count.
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$WARPX_FIELDS" --max-files "$WARPX_FILES" "${NP[@]}" --check-bound )
    CLASSIFY=( --fields "$WARPX_FIELDS" ) ;;
  ai)     # Replay, ~41 chunks per checkpoint file.
    CMD=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG"
          --fields "$AI_FIELDS" --max-files "$AI_FILES" "${NP[@]}" --check-bound )
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

# ---- The distribution class, read from the replayed field files -------------
if [ "$n" -gt 0 ]; then
  python3 "$HERE/classify_chunks.py" --explore "$LOG" "${CLASSIFY[@]}" \
    --out "$run/$WL/dist.csv" 2>&1 | sed 's/^/   /'
fi

# ---- Storage images are the run's raw bytes and are never kept -------------
find "$run" -maxdepth 3 -type f \( -name chi_bdev.dat -o -name 'cte_tier.dat*' \) -delete 2>/dev/null

if [ "$run" != "$OUT" ]; then
  if tar -C "$run" --exclude='chi_bdev.dat' --exclude='cte_tier.dat*' --exclude='raw' -cf - . \
     | tar -C "$OUT" -xf -; then rm -rf "$run"
  else echo "   COPY FAILED -- the run is still in $run"; fi
fi
exit 0
