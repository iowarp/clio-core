#!/usr/bin/env bash
# Every figure here that CAN be rebuilt from this directory, rebuilt from it.
#
#   PYTHON=<interpreter with matplotlib+numpy> ./regenerate.sh [OUTDIR]
#
# Default OUTDIR is this directory, i.e. it overwrites the figures in place.
# Pass a scratch path to render elsewhere and diff first.
#
# NOT regenerable: <workload>/evolution_begin_middle_end.png. That one reads
# the DUMPS (figure_evolution.py --dir), and the dumps were deleted after
# measuring -- about 72 GB across the four sweeps. The five mid-plane slices in
# <workload>.slices.npz cover fig3's top row and nothing else.
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-$HERE}
PF=$HERE/paper_figures.py
PY=${PYTHON:-python3}
"$PY" -c "import matplotlib, numpy" 2>/dev/null || {
  echo "matplotlib/numpy missing from $PY -- set PYTHON=<interpreter that has them>" >&2
  exit 1; }

# <workload>:<evolution run>:<k31 run>:<fig3 field>. LAMMPS has no fig3 --
# atom coordinates have no grid to localize activity on, hence no .slices.npz
# and no field either.
#
# THE FIELD IS NOT RECORDED IN THE .npz. Those files hold five arrays named
# s0..s4 and nothing else, so which variable the top row shows cannot be
# recovered from them -- it is pinned here instead. Only the BOTTOM row reads
# it (from blocks.csv.gz); the top row is the cache verbatim. The values below
# were recovered by matching the committed figures' bottom row against every
# field in blocks.csv.gz: vpic cbx and cby are identical there and the Weibel
# signature is cby; warpx Bx/By/Bz/Ex/Ey are likewise identical, and Ex is the
# laser polarization.
# fig4's --field picks which variable panel (a) opens on; fig3's picks the
# bottom row. One field per workload serves both. LAMMPS has no fig3 but its
# fig4 still needs a field, hence `position` rather than `-`.
ROWS="nyx:nyx_128_1000:nyx_256_2000_k31_4m:density
vpic:vpic_126_2000:vpic_126_2000_k31_4m:cby
lammps:lammps_2000:lammps_2000_k31_4m:position
warpx:warpx_2000:warpx_2000_k31_1m:Ex"

echo "$ROWS" | while IFS=: read -r wl ev k31 field; do
  mkdir -p "$OUT/$wl"
  if [ -f "$HERE/$wl/$ev.slices.npz" ]; then  # no .npz -> no fig3
    "$PY" "$PF" fig3 --slices "$HERE/$wl/$ev.slices.npz" \
                     --blocks "$HERE/$wl/$ev.blocks.csv.gz" \
                     --field "$field" --out "$OUT/$wl/fig3"
  fi
  "$PY" "$PF" fig4   --blobs "$HERE/$wl/$k31.blobs.csv.gz" \
                     --field "$field" --out "$OUT/$wl/fig4"
  "$PY" "$PF" fields --blobs "$HERE/$wl/$k31.blobs.csv.gz" --out "$OUT/$wl/fields_fig"
done

# fig5 spans all four, so it is one file at the top level rather than four
# byte-identical copies in the workload directories.
"$PY" "$PF" fig5 --out "$OUT/fig5" \
  --run nyx=$HERE/nyx/nyx_128_1000.json \
  --run vpic=$HERE/vpic/vpic_126_2000.json \
  --run lammps=$HERE/lammps/lammps_2000.json \
  --run warpx=$HERE/warpx/warpx_2000.json
echo "regenerated into $OUT"
