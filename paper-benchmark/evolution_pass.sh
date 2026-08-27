#!/usr/bin/env bash
# Temporal evolution metric AND the begin/middle/end figure for each workload,
# written into all six of its cells as evolution.csv, evolution.json and
# figures/evolution_begin_middle_end.png.
#
#   ./evolution_pass.sh [--workload W] [--steps N] [--int N]
#                       [--scratch DIR] [--out DIR] [--keep-data]
#
# ONE PASS PER WORKLOAD, NOT PER CELL, and the copy is deliberate. The metric
# describes the SIMULATION's data, not the compressor's: all six configurations
# of a workload run identical simulation parameters, so the evolution is the
# same number six times. Computing it once and writing it into each cell keeps
# every experiment directory self-contained -- section 7 of the spec asks for
# evolution.csv beside results.csv -- without pretending six measurements were
# made. metrics.json records which cell it was computed from.
#
# WHERE THE DATA COMES FROM DIFFERS, and two workloads have none to give:
#
#   warpx   FREE. Writes openPMD natively even in situ -- the VOL compresses
#           on the way past, so the .h5 the metric reads are the same bytes
#           the compressor saw.
#   nyx     FREE. Runs through replay, so its .f32 dumps are already on disk.
#   vpic    COSTS A RUN. In situ writes nothing but the tier, so gen_fields.sh
#           has to produce dumps at the same parameters.
#   lammps  COSTS A RUN. Library in process, nothing on disk; --raw is the only
#           way to see the bytes, and it writes exactly what was staged.
#
# THE FIGURE IS MADE HERE, NOT IN A LATER PASS, because it needs the same data
# the metric does and this script deletes that data when it is done. Splitting
# them would mean producing it twice -- for VPIC at paper scale that is a
# second half-hour run for a picture of bytes we had already read.
#
# The two extra runs use the SAME simulation parameters as the matrix -- the
# evolution study's winners, which are the defaults inside each runner -- so
# what is measured is the data the matrix compressed.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRATCH=${SCRATCH:-/tmp/paper-bench}
OUT_ROOT=$HERE
STEPS=${STEPS:-1000} INT=${INT:-40} ONLY_W="" KEEP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --workload) ONLY_W=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int) INT=$2; shift 2;;
    --scratch) SCRATCH=$2; shift 2;;
    --out) OUT_ROOT=$2; shift 2;;
    --keep-data) KEEP=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

CONFIGS="balance_lossless balance_lossy_0.001 ratio_lossless
         ratio_lossy_0.001 ratio_lossy_0.01 ratio_lossy_0.1"

wait_gpu () {
  for _ in $(seq 120); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    [ "${u:-0}" -lt 500 ] && { sleep 3; return; }
    sleep 3
  done
}

# fan_out <workload> <evdir> <source-note>
fan_out () {
  local w=$1 ev=$2 note=$3 n=0
  [ -f "$ev/blocks.csv" ] || { echo "   no metric produced for $w"; return 1; }
  for c in $CONFIGS; do
    local cell=$OUT_ROOT/$w/$c
    [ -d "$cell" ] || continue
    mkdir -p "$cell/figures"
    cp "$ev/blocks.csv"     "$cell/evolution.csv"
    cp "$ev/evolution.json" "$cell/evolution.json"
    [ -f "$ev/evolution_begin_middle_end.png" ] && \
      cp "$ev/evolution_begin_middle_end.png" "$cell/figures/"
    # Say plainly that this is one measurement shared by six cells, and what
    # it was computed from, so nobody reads six numbers into it later.
    python3 - "$cell/evolution.json" "$note" <<'PY'
import json, sys
p, note = sys.argv[1], sys.argv[2]
d = json.load(open(p))
d["shared_across_configs"] = True
d["measured_from"] = note
d["note"] = ("One measurement per workload, copied into all six cells: the "
             "six configurations run identical simulation parameters, so the "
             "evolution of the data is the same for all of them.")
json.dump(d, open(p, "w"), indent=1)
PY
    n=$((n+1))
  done
  echo "   -> evolution.csv + evolution.json in $n cell(s)"
}

# figure <evdir> <args...> -- the begin/middle/end plate, same data as the
# metric, one shared color scale across the three panels.
figure () {
  local ev=$1; shift
  "$HERE/figure_evolution.py" "$@" \
      --out "$ev/evolution_begin_middle_end.png" 2>&1 | sed 's/^/   /'
}

# ---------------------------------------------------------------------------
run_warpx () {
  # The openPMD any one cell already wrote. Which cell is irrelevant -- the
  # simulation is the same in all six and the native .h5 is written before the
  # compressor ever sees the data.
  local diags=""
  for c in $CONFIGS; do
    local d
    d=$(find "$SCRATCH/warpx/$c" -maxdepth 4 -type d -name diag1 2>/dev/null | head -1)
    [ -n "$d" ] && { diags=$d; break; }
  done
  [ -n "$diags" ] || { echo "   warpx: no openPMD found under $SCRATCH/warpx -- run the matrix first"; return 1; }
  echo "   reading $diags"
  "$HERE/evolution.py" --source openpmd --dir "$diags" \
      --label "warpx_paper" --out "$SCRATCH/ev-warpx" > "$SCRATCH/ev-warpx.log" 2>&1 \
    || { echo "   evolution.py failed, see $SCRATCH/ev-warpx.log"; return 1; }
  tail -4 "$SCRATCH/ev-warpx.log" | sed 's/^/   /'
  # --shape IS NOT OPTIONAL HERE. WarpX's grid is 64x64x512 = 2,097,152 cells,
  # which is exactly 128^3, so a cube-root guess SUCCEEDS and reshapes a slab
  # into a cube -- and the resulting mid-plane lands on zeros, giving a blank
  # panel from a field that reaches 3.4e11. --axis y is the plane that shows
  # the wake; z is antisymmetric for Ez and comes out empty even when correct.
  figure "$SCRATCH/ev-warpx" --source openpmd --dir "$diags" --workload warpx \
      --shape 64,64,512 --axis y
  fan_out warpx "$SCRATCH/ev-warpx" "openPMD written by the WarpX run itself"
}

run_nyx () {
  local f=$SCRATCH/nyx-fields
  [ -d "$f" ] || { echo "   nyx: no fields at $f -- run the matrix first"; return 1; }
  # --step-scale: the dump directories are numbered by dump index, not by
  # timestep, exactly as metrics.py has to reconstruct the step for Nyx blobs.
  "$HERE/evolution.py" --source f32 --dir "$f" --step-scale "$INT" \
      --label "nyx_paper" --out "$SCRATCH/ev-nyx" > "$SCRATCH/ev-nyx.log" 2>&1 \
    || { echo "   evolution.py failed, see $SCRATCH/ev-nyx.log"; return 1; }
  tail -4 "$SCRATCH/ev-nyx.log" | sed 's/^/   /'
  figure "$SCRATCH/ev-nyx" --source f32 --dir "$f" --workload nyx --step-scale "$INT"
  fan_out nyx "$SCRATCH/ev-nyx" "the .f32 dumps the six replay cells read"
}

run_vpic () {
  local f=$SCRATCH/vpic-ev-fields
  if [ ! -d "$f" ]; then
    echo "   generating VPIC dumps (126^3, $STEPS steps, clean_div 10 default)"
    wait_gpu
    "$HERE/vpic/gen_fields.sh" --ncell 126 --nppc 8 --steps "$STEPS" \
        --dump-int "$INT" --out "$f" > "$SCRATCH/ev-vpic-gen.log" 2>&1 \
      || { echo "   gen_fields failed, see $SCRATCH/ev-vpic-gen.log"; return 1; }
  fi
  "$HERE/evolution.py" --source f32 --dir "$f" --step-scale "$INT" \
      --label "vpic_paper" --out "$SCRATCH/ev-vpic" > "$SCRATCH/ev-vpic.log" 2>&1 \
    || { echo "   evolution.py failed, see $SCRATCH/ev-vpic.log"; return 1; }
  tail -4 "$SCRATCH/ev-vpic.log" | sed 's/^/   /'
  figure "$SCRATCH/ev-vpic" --source f32 --dir "$f" --workload vpic --step-scale "$INT"
  fan_out vpic "$SCRATCH/ev-vpic" "a gen_fields.sh run at the matrix's own parameters"
  [ "$KEEP" = 1 ] || rm -rf "$f"
}

run_lammps () {
  local raw=$SCRATCH/lammps-raw
  if [ ! -d "$raw" ]; then
    echo "   staging LAMMPS bytes with --raw ($STEPS steps, gap $INT)"
    wait_gpu
    mkdir -p "$raw"
    local natoms=$(( 4 * 20 * 20 * 20 ))
    # The codec choice cannot change the bytes that were STAGED, so this uses
    # the cheapest configuration rather than repeating one of the six.
    "$HERE/lammps/run_config.sh" static-zstd --box 20 --steps "$STEPS" \
        --gap "$INT" --chunk $(( natoms * 3 * 8 )) --require-device --no-verify \
        --raw "$raw" --results "$SCRATCH/lammps-ev" --tag ev \
        > "$SCRATCH/ev-lammps-run.log" 2>&1 \
      || { echo "   run failed, see $SCRATCH/ev-lammps-run.log"; return 1; }
  fi
  # --f64: LAMMPS stages double-precision atom data, unlike the three field
  # workloads. Getting this wrong halves the element count and silently
  # reinterprets every value.
  "$HERE/evolution.py" --source raw --dir "$raw" --f64 \
      --label "lammps_paper" --out "$SCRATCH/ev-lammps" > "$SCRATCH/ev-lammps.log" 2>&1 \
    || { echo "   evolution.py failed, see $SCRATCH/ev-lammps.log"; return 1; }
  tail -4 "$SCRATCH/ev-lammps.log" | sed 's/^/   /'
  # --atoms: these rows are atom xyz, not a field cube, so they are scattered
  # rather than sliced. --f64 for the same reason the metric needs it.
  figure "$SCRATCH/ev-lammps" --source raw --dir "$raw" --workload lammps --atoms --f64
  fan_out lammps "$SCRATCH/ev-lammps" "bytes staged by a --raw run at the matrix's parameters"
  [ "$KEEP" = 1 ] || rm -rf "$raw"
}

for w in warpx vpic nyx lammps; do
  [ -n "$ONLY_W" ] && [ "$w" != "$ONLY_W" ] && continue
  echo "== $w"
  "run_$w"
done
