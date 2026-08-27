#!/usr/bin/env bash
# WarpX round 3: re-run the density ramp with the profile guarded to z > 0.
#
# Round 2's ramp was "2.e23*(1.0+4.0*z/141.e-6)", which is NEGATIVE for
# z < -35.25 um -- a third of the domain, where the deck's own constant profile
# would have injected nothing anyway (electrons.zmin = 0). So that run changed
# the initial plasma extent as well as the ramp, and its result cannot be read
# as "a longitudinal ramp does not help". The (z>0) factor is upstream's own
# idiom for defining a profile by intervals (parameters.rst: "The factor (x>0)
# equals 1 where x>0 and 0 where x<=0").
set -uo pipefail
BIN=$HOME/src/warpx/build-clio/bin/warpx.3d
DECK=$HOME/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d
EV=/home/cc/clio-core/paper-benchmark/evolution.py
OUT=/tmp/wx-study

run_one() {
  local name=$1; shift
  local d=$OUT/$name
  echo "== $name: $*"
  rm -rf "$d"; mkdir -p "$d"
  local t0=$(date +%s.%N)
  ( cd "$d" && "$BIN" "$DECK" \
      max_step=1000 amr.n_cell="64 64 512" \
      diag1.intervals=10 diag1.openpmd_backend=h5 diag1.write_species=0 \
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

# Ramp confined to the region the deck actually fills, and combined with the
# winning a0 so it is tested as an addition to the leader rather than against it.
run_one zramp_z0 laser1.e_max=16.e12 \
  electrons.profile=parse_density_function \
  "electrons.density_function(x,y,z)=2.e23*(1.0+4.0*z/141.e-6)*(z>0)"
run_one a0x2_zramp laser1.e_max=32.e12 \
  electrons.profile=parse_density_function \
  "electrons.density_function(x,y,z)=2.e23*(1.0+4.0*z/141.e-6)*(z>0)"
echo "== done"
