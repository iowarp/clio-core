#!/usr/bin/env bash
# The exact runs that produced every file in this directory. ~10 minutes and
# 25 GB of scratch; nothing here is needed to READ the record.
#
#   ./run.sh [--out DIR]      # DIR defaults to /tmp, holds the 25 GB
#
# THE FIELD DATA IS NOT COMMITTED, by the same rule as the rest of
# evolution-study: 20.5 GB of .f32 and a 5.4 GB compressed tier, both
# regenerable from this script. Only the logs, the CSVs and the measurement
# summaries are kept.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NYX=$HERE/../../nyx
OUT=/tmp
[ "${1:-}" = "--out" ] && OUT=$2

# ---------------------------------------------------------------------------
# 1. Dump route -- 51 frames x 6 components x 64 MiB = 20,535,313,913 B.
#
# --ncell 256 --steps 2000 rather than the study default's 128/1000 because a
# CFL timestep halves with the grid: 2000 steps at 256^3 ends at the SAME
# physical state as 1000 at 128^3, so the run stays in the regime the study
# validated instead of moving to a new one. Doubling ncell ALONE would not --
# the Sedov shock advances a roughly fixed number of CELLS per step, so at
# fixed --steps a 2x grid halves the fraction of the domain that ever changes.
#
# gen_fields.sh parks stop_time at 1.0, so --steps is what ends this run.
# ---------------------------------------------------------------------------
"$NYX/gen_fields.sh" --ncell 256 --steps 2000 --plot-int 40 \
    --out "$OUT/nyx-20gb" 2>&1 | tee "$HERE/gen.log"
cp "$OUT/nyx-20gb/gen.json" "$HERE/gen.json"
cp "$OUT/nyx-20gb/nyx.log"  "$HERE/gen.nyx.log"

# ---------------------------------------------------------------------------
# 2. Measure it. --step-scale 40 because the dump directories are numbered by
# dump index, not by timestep.
# ---------------------------------------------------------------------------
"$HERE/../../evolution.py" --source f32 --dir "$OUT/nyx-20gb" --step-scale 40 \
    --label nyx_256_2000_int40 --out "$OUT/ev-nyx-20gb"
cp "$OUT/ev-nyx-20gb/evolution.json" "$HERE/evolution.json"
gzip -9 -c "$OUT/ev-nyx-20gb/blocks.csv" > "$HERE/blocks.csv.gz"

# ---------------------------------------------------------------------------
# 3. In situ, same settings -- the same bytes through the compressor, with no
# plotfile written at all.
#
# --stop-time 1.0 IS NOT OPTIONAL HERE and gen_fields.sh's default does not
# help: run_config_insitu.sh leaves the deck's stop_time = 0.01 in force, which
# at 256^3 and cfl 0.8 ends the run near step 360 -- about 4 GB instead of 20,
# exiting 0.
#
# --steps 2000 SUBMITS 50 FRAMES, NOT 51. The hook fires from
# Nyx::updateInSitu(), which has no step-0 frame, so in-situ frames are
# steps/int where the dump route's are steps/int + 1. That is the whole
# 20,535,313,913 vs 20,132,659,200 B difference between the two routes: exactly
# one frame, 402,653,184 B. Use --steps 2040 to submit 51.
# ---------------------------------------------------------------------------
"$NYX/run_config_insitu.sh" explore-balance --ncell 256 --steps 2000 --int 40 \
    --chunk 4194304 --stop-time 1.0 \
    --results "$OUT/nyx-20gb-insitu" --tag insitu20 2>&1 \
  | tee "$HERE/insitu.stdout.tee.log"

S=$OUT/nyx-20gb-insitu/insitu20
for f in meta.json compose.yaml stdout.log nyx.log runtime.log; do
  cp "$S/$f" "$HERE/insitu.$f"
done
for f in selection explore blobs; do
  gzip -9 -c "$S/$f.csv" > "$HERE/insitu.$f.csv.gz"
done

echo "record updated in $HERE; field data left in $OUT (25 GB, delete it)"
