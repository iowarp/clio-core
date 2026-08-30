#!/usr/bin/env bash
# Phase 1 of the WarpX REPLAY route: run a stock WarpX, dump its openPMD field
# datasets as flat float32, the same shape Nyx and VPIC produce.
#
#   ./warpx_gen_fields.sh [--ncell "64 64 512"] [--steps N] [--interval N]
#                         [--out DIR] [--deck FILE]
#
# WHY THIS EXISTS. Through the HDF5 VOL, WarpX data can never be device-
# resident when Clio sees it, and the path trace proves it rather than assuming
# it: over a 40-step run, 510 of 510 writes reported device_src=0 and whole=0.
# Two independent causes, either sufficient on its own --
#
#   1. openPMD emits one PARTIAL (hyperslab) write per AMReX box, so the VOL
#      takes clio_stage_append, which assembles non-contiguous runs into a HOST
#      buffer. The device-capable branch in clio_vol.cc needs a WHOLE-dataset
#      write and is never reached.
#   2. AMReX/openPMD copy to host before H5Dwrite anyway, so residency is gone
#      before Clio is called at all ("Measured with a stock WarpX: HOST").
#
# Replaying files sidesteps both: there is no hyperslab assembly, and the
# replay driver stages each chunk H2D itself, exactly as the Nyx route does.
#
# WHAT IT COSTS. This is no longer in situ. WarpX's distinctive claim in this
# benchmark -- a stock, unpatched binary compressed while it runs, against
# upstream's 636-line patch -- belongs to the VOL route and is given up here.
# The two answer different questions and both are worth having: the VOL route
# asks whether Clio can compress an unmodified application at all; this one
# asks how WarpX DATA compresses, measured on the same footing as Nyx and VPIC.
#
# Field datasets only. The particle records under /data/<step>/particles are
# skipped: they are 1-D per-particle arrays whose length changes as particles
# enter and leave, so they do not chunk into the fixed-size, fixed-shape blobs
# the other workloads' field dumps produce and would not be comparable.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NCELL="64 64 512" STEPS=40 INTERVAL=10
WARPX_BIN=${WARPX_BIN:-/projects/bekn/imuradli/np-src/warpx/build-clio/bin/warpx.3d}
DECK=${DECK:-/projects/bekn/imuradli/np-src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}
OUT=${OUT:-/projects/bekn/imuradli/np-warpx-replay/fields}
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --interval) INTERVAL=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --deck) DECK=$2; shift 2;;
    --bin) WARPX_BIN=$2; shift 2;;
    -h|--help) sed -n '2,8p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -x "$WARPX_BIN" ] || { echo "missing WarpX: $WARPX_BIN" >&2; exit 1; }
[ -f "$DECK" ]      || { echo "missing deck: $DECK" >&2; exit 1; }
command -v h5dump >/dev/null || { echo "h5dump not on PATH" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

# NO VOL here on purpose: HDF5_VOL_CONNECTOR is left unset so WarpX writes a
# plain native openPMD file. Clio is not in this process at all.
echo "== WarpX ${NCELL// /x}, $STEPS steps, diag every $INTERVAL -> native openPMD"
( cd "$WORK" && "$WARPX_BIN" "$DECK" \
    max_step="$STEPS" diag1.intervals="$INTERVAL" \
    amr.n_cell="$NCELL" diag1.openpmd_backend=h5 ) > "$OUT/warpx.log" 2>&1 \
  || { echo "WarpX failed; tail of $OUT/warpx.log:" >&2; tail -15 "$OUT/warpx.log" >&2; exit 1; }

mapfile -t H5 < <(find "$WORK" -name '*.h5' | sort)
[ "${#H5[@]}" -gt 0 ] || { echo "WarpX wrote no .h5 under $WORK" >&2; exit 1; }
echo "   $(( ${#H5[@]} )) openPMD file(s)"

# /data/<step>/fields/{B,E,j}/{x,y,z} and /data/<step>/fields/rho.
FIELDS=(B/x B/y B/z E/x E/y E/z j/x j/y j/z rho)
n=0
for h in "${H5[@]}"; do
  # The step is a group under /data; read it rather than parsing the filename.
  step=$(h5ls "$h/data" 2>/dev/null | awk '{print $1}' | head -1)
  [ -n "$step" ] || continue
  d=$(printf '%s/step%05d' "$OUT" "$step"); mkdir -p "$d"
  for f in "${FIELDS[@]}"; do
    name=${f//\//_}
    # -b LE writes the raw little-endian payload with no HDF5 framing, which
    # is exactly what --ext .f32 expects. WarpX is built SP, so these are
    # float32 and the replay driver needs no --f64.
    if h5dump -d "/data/$step/fields/$f" -b LE -o "$d/$name.f32" "$h" >/dev/null 2>&1 \
       && [ -s "$d/$name.f32" ]; then
      n=$((n+1))
    else
      rm -f "$d/$name.f32"
    fi
  done
done

[ "$n" -gt 0 ] || { echo "h5dump produced no field files" >&2; exit 1; }
cat > "$OUT/gen.json" <<JSON
{"workload":"warpx-lwfa","ncell":"$NCELL","steps":$STEPS,"interval":$INTERVAL,
 "frames":${#H5[@]},"files":$n,"precision":"float32","route":"replay"}
JSON
echo "   $n field file(s), $(du -sh "$OUT" | cut -f1)"
echo "   fields: $(find "$OUT" -name '*.f32' -printf '%f\n' | sed 's/\.f32//' | sort -u | tr '\n' ' ')"
