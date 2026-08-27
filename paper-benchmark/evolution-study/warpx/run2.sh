#!/usr/bin/env bash
# WarpX evolution study, round 2: isolate a0, and try a longitudinal density
# ramp -- the first round showed a large transient followed by a plateau, which
# is what a quasi-static wake in a co-moving window looks like. A z-dependent
# plasma density means the wake the pulse drives keeps changing as it
# propagates, so the plateau should not be flat.
set -uo pipefail
BIN=$HOME/src/warpx/build-clio/bin/warpx.3d
DECK=$HOME/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d
EV=/home/cc/clio-core/paper-benchmark/evolution.py
OUT=/tmp/wx-study
STEPS=${STEPS:-1000} INT=${INT:-10}

run_one() {
  local name=$1; shift
  local d=$OUT/$name
  echo "== $name: $*"
  rm -rf "$d"; mkdir -p "$d"
  local t0=$(date +%s.%N)
  ( cd "$d" && "$BIN" "$DECK" \
      max_step=$STEPS amr.n_cell="64 64 512" \
      diag1.intervals=$INT diag1.openpmd_backend=h5 diag1.write_species=0 \
      warpx.verbose=0 "$@" > wx.log 2>&1 )
  local rc=$?
  local wall=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')
  if [ $rc -ne 0 ]; then echo "   FAILED rc=$rc"; tail -8 "$d/wx.log"; return 1; fi
  echo "   sim wall ${wall}s, $(ls "$d"/diags/diag1/*.h5 2>/dev/null | wc -l) dumps"
  "$EV" --source openpmd --dir "$d/diags" --out "$OUT/ev/$name" \
        --block 1048576 --label "$name" 2>&1 | sed 's/^/   /'
  python3 - "$OUT/ev/$name/evolution.json" "$wall" "$*" <<'PY'
import json,sys
p,wall,params=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(p)); d["sim_wall_s"]=float(wall); d["params"]=params
json.dump(d,open(p,"w"),indent=1)
PY
  rm -rf "$d/diags"
}

# a0 alone: e_max 32e12 is a0 = 8 at 0.8 um (parameters.rst, <laser_name>.e_max).
run_one a0x2 electrons.density=2.e23 laser1.e_max=32.e12

# Longitudinal density ramp, 2e23 -> 1e24 over the 141 um the window covers in
# 1000 steps. parse_density_function, parameters.rst <species_name>.profile.
run_one zramp laser1.e_max=16.e12 \
  electrons.profile=parse_density_function \
  "electrons.density_function(x,y,z)=2.e23*(1.0+4.0*z/141.e-6)"
echo "== done"
