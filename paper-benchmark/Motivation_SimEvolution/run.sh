#!/usr/bin/env bash
# The motivation figure: a simulation's compressibility ceiling moves, both
# across the volume and across the run, so the choice has to be made per
# chunk and re-made as the data evolves.
#
#   ./run.sh                 # run Nyx density through Clio, draw the figure
#   ./run.sh --redraw        # draw from the CSVs kept here, no GPU needed
#   ./run.sh --fields DIR    # a different dump set
#
# Produces ONE figure, in this directory:
#
#   evolution_begin_middle_end_density.png
#                                 three dumps of the density field on one
#                                 shared colour scale, each ruled into the
#                                 chunks it is stored as and each slab
#                                 labelled with the maximum ratio it reaches
#
# The data and its compressibility are one plate on purpose: shown apart,
# a reader has to carry sixteen numbers from one picture to another. Two
# earlier plates -- the same three dumps with a bar chart beside them, and a
# (slab x dump) map of the whole run -- are gone, and so is the second plotter
# that drew them. What they were read for survives as numbers:
# `./plot_motivation.py --blobs blobs.csv.gz --explore explore.csv.gz --stats`
# prints the spreads over every slab of every dump, not just the three drawn.
#
# ---------------------------------------------------------------------------
# Four things here are not defaults, and each one was learned by getting it
# wrong first.
#
# 1. THE RATIO CAP IS REMOVED. Upstream clamps the ratio at 100x before
#    scoring (predictor.h: `std::min(ratio_cap, compression_ratio)`), and
#    the clamp lands on the I/O term, bytes/(ratio*bw) -- the only term the
#    ratio cost model keeps. On this data most chunks compress far past
#    100x, the ceiling reaching 5,924x, so under the cap every one of them
#    scores IDENTICALLY and the winner among them is a tie-break rather than
#    a choice. CLIO_NEUROPRESS_RATIO_CAP=1e9 lifts it; measured afterwards,
#    the run then adopts the best-measured action on essentially every chunk
#    (median 1.001x of it).
#
#    THE FIGURE does not depend on this: the number on each slab is the
#    maximum measured ratio over all 32 actions, a property of the data and
#    the codecs rather than of the cost model. The cap changes what the run
#    ADOPTS, and so what lands on the tier.
#
# 2. ONE FIELD PER RUN. The replay leaks roughly 1 GB of device memory per
#    dump; over 258 files (six fields x 43 dumps) that reaches 40 GB and the
#    process dies with "CUDA Error 2: out of memory" around dump 30 of 43.
#    Density alone is 43 files, peaks near 7 GB and finishes in 90 s. The
#    symlink farm below is what makes a single-field run possible without
#    copying 2.75 GiB.
#
# 3. QUALITY AND DECOMPRESS-TIME MEASUREMENT ARE OFF. run_config.sh turns
#    both on by default; they are the heaviest per-chunk allocators and the
#    RATIO cost model reads neither -- it is w_io * bytes/(ratio*bw) with the
#    two time weights zeroed, so a measured decompression time cannot change
#    a decision. The error bound is enforced by the quantizer whether or not
#    the PSNR is measured. Under a balanced cost model MEASURE_DT would have
#    to go back on, because a third of that cost is decompression time.
#
# 4. THE FARM IS NAMED BY TIMESTEP, not by dump index. Nyx numbers its dump
#    directories 0,1,2..., so every tool downstream would label the panels
#    "dump 21" or, worse, multiply by the interval and print step 2,016 for a
#    2,000-step run -- AMReX writes its last plotfile at max_step rather than
#    at the next multiple. Naming the links plt<step> makes every label true
#    with no --step-scale anywhere.
# The plate is drawn WITHOUT a figure title: in the paper that text is the
# caption, and a plate that carries its own headline above the caption reads
# as a mistake. Each panel is titled with its timestep, nothing else, for the
# same reason -- which end of the run it is, the caption says.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(cd "$HERE/.." && pwd)

# site.sh carries this machine's paths AND the LD_LIBRARY_PATH the replay
# driver needs: without $NPENV/cusz/lib64 it dies rc=127 on libpsz_cu_core.so
# before compressing anything. SIZE=2k selects the 2,000-step dump set.
SIZE=${SIZE:-2k}
# shellcheck source=/dev/null
[ -f "$BENCH/site.sh" ] && . "$BENCH/site.sh"

FIELDS=${NYX_FIELDS:-$BENCH/nyx/fields}
RESULTS=${RESULTS:-${TIER2_ROOT:-/tmp}/motivation-simevolution}
FARM=${FARM:-$RESULTS/density-dumps}
TAG=nyx_density_k31_ratio
STORE=$RESULTS/$TAG
PY=${PYTHON:-python3}
REDRAW=0
# Where in the run to draw. NOT 0: a Sedov blast at step 0 is a point source
# in an otherwise uniform box, so the first panel would be a blank plate.
AT=()

while [ $# -gt 0 ]; do
  case "$1" in
    --fields)  FIELDS=$2; shift 2;;
    --results) RESULTS=$2; STORE=$RESULTS/$TAG; FARM=$RESULTS/density-dumps; shift 2;;
    --at)      AT+=("$2"); shift 2;;
    --redraw)  REDRAW=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -d "$FIELDS" ] || { echo "no dumps at $FIELDS -- set NYX_FIELDS or --fields" >&2; exit 1; }
[ ${#AT[@]} -gt 0 ] || AT=(0.05 0.5 1.0)
AT_ARGS=(); for f in "${AT[@]}"; do AT_ARGS+=(--at "$f"); done

# --- the farm: density only, one link per dump, named by timestep ----------
PLOT_INT=$($PY -c "
import json,sys
try: print(json.load(open('$FIELDS/gen.json'))['plot_int'])
except Exception: print(1)")
STEPS=$($PY -c "
import json,sys
try: print(json.load(open('$FIELDS/gen.json'))['steps'])
except Exception: print(0)")
mkdir -p "$FARM"
rm -rf "${FARM:?}"/plt*
N=$(find "$FIELDS" -maxdepth 1 -name 'plt*' -type d | wc -l)
i=0
for d in $(find "$FIELDS" -maxdepth 1 -name 'plt*' -type d | sort); do
  if [ "$i" -eq $((N - 1)) ] && [ "$STEPS" -gt 0 ]; then step=$STEPS
  else step=$((i * PLOT_INT)); fi
  sub=$(printf "%s/plt%05d" "$FARM" "$step")
  mkdir -p "$sub"
  ln -sfn "$d"/*_density.f32 "$sub"/
  i=$((i + 1))
done
echo "== $N density dumps, steps 0..$STEPS, linked into $FARM"

# --- the run ---------------------------------------------------------------
if [ "$REDRAW" = 0 ]; then
  echo "== K=31 exploration, ratio cost model, no ratio cap, eb=1e-3"
  # BENCH_TIER1_MB is explicit because bench_compose sizes the tier with
  # `du -sm $FIELDS`, and du does not follow symlinks: on a farm it would
  # compute a few kilobytes of payload and hand the run a 513 MB tier.
  CLIO_NEUROPRESS_STAGE_H2D=1 \
  CLIO_NEUROPRESS_RATIO_CAP=1e9 \
  MEASURE_QUALITY=0 MEASURE_DT=0 BENCH_TIER1_MB=${BENCH_TIER1_MB:-8192} \
      "$BENCH/nyx/run_config.sh" explore-ratio \
      --fields "$FARM" --bw 5e6 --eb 1e-3 \
      --explore-k 31 --explore-thresh 0 \
      --chunk 4194304 --results "$RESULTS" --tag "$TAG"
  # Kept here so the figure redraws without the store, which holds a 35 GB
  # bdev image on a scratch device and does not survive a cleanup. explore.csv
  # is what makes the labels the MAXIMUM achievable ratio rather than the one
  # the run happened to adopt.
  gzip -c "$STORE/blobs.csv"   > "$HERE/blobs.csv.gz"
  gzip -c "$STORE/explore.csv" > "$HERE/explore.csv.gz"
  # THE RUN SAW A FARM OF SYMLINKS, so run_config.sh's own meta.json records
  # the links' size (a couple of kilobytes) as the payload and 0 timesteps,
  # gen.json not being in the farm. Restate both from the dumps the links
  # point at, and record the knobs that are not defaults.
  cp "$STORE/meta.json" "$HERE/meta.json"
  "$PY" - "$HERE/meta.json" "$FIELDS" "$STEPS" <<'FIXUP'
import json, os, sys
path, fields, steps = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(path))
m.update(payload_bytes=sum(os.path.getsize(os.path.join(r, f))
                           for r, _, fs in os.walk(fields) for f in fs
                           if f.endswith("_density.f32")),
         steps=int(steps), field="density",
         source_dumps=os.path.realpath(fields),
         ratio_cap="1e9 (upstream 100, lifted)",
         measure_quality=0, measure_dt=0)
json.dump(m, open(path, "w"), indent=1)
FIXUP
fi
[ -f "$HERE/blobs.csv.gz" ] || { echo "no blobs.csv.gz here -- run without --redraw" >&2; exit 1; }

# --- the figure ------------------------------------------------------------
echo "== drawing"
# plot_motivation.py is this figure's only plotter, beside this script: it
# reads the CSVs itself, reads the frames and draws the panels through
# ../plot/figure_evolution.py, and composes the plate. --axis x
# because a chunk is a z-slab and a z mid-plane slice would lie INSIDE one
# chunk -- there would be no boundary in the picture. --explore is what makes
# each slab's number the MAXIMUM ACHIEVABLE ratio rather than the adopted one.
"$PY" "$HERE/plot_motivation.py" \
    --dir "$FARM" --field density --axis x \
    --blobs "$HERE/blobs.csv.gz" --explore "$HERE/explore.csv.gz" \
    "${AT_ARGS[@]}" \
    --out "$HERE/evolution_begin_middle_end_density.png"

echo
ls -la "$HERE/evolution_begin_middle_end_density.png"
