#!/usr/bin/env bash
# Temporal evolution metric for each workload, written into all six of its
# cells as evolution.csv + evolution.json.
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
    cp "$ev/blocks.csv"     "$cell/evolution.csv"
    cp "$ev/evolution.json" "$cell/evolution.json"
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
  fan_out lammps "$SCRATCH/ev-lammps" "bytes staged by a --raw run at the matrix's parameters"
  [ "$KEEP" = 1 ] || rm -rf "$raw"
}

for w in warpx vpic nyx lammps; do
  [ -n "$ONLY_W" ] && [ "$w" != "$ONLY_W" ] && continue
  echo "== $w"
  "run_$w"
done
