#!/usr/bin/env bash
# Nyx round 2. Round 1 found exp_energy inert at fixed step count and nyx.cfl
# decisive, so: confirm the E-independence still holds at the higher cfl, and
# push cfl to the top of its documented range (NyxInputs.rst: Real > 0, <= 1).
set -uo pipefail
BIN=$HOME/src/Nyx/build-clio/Exec/HydroTests/nyx_HydroTests
DECK=$(dirname "$BIN")/inputs.3d.sph.sedov
EV=/home/cc/clio-core/paper-benchmark/evolution.py
OUT=/tmp/nyx-study
STEPS=1000 INT=10 NCELL=128

run_one() {
  local name=$1 energy=$2 cfl=$3
  local d=$OUT/$name
  echo "== $name: exp_energy=$energy nyx.cfl=$cfl"
  rm -rf "$d"; mkdir -p "$d/fields"
  local w=$(mktemp -d)
  local t0=$(date +%s.%N)
  ( cd "$w" && env NYX_DUMP_FIELDS=1 NYX_DUMP_DIR="$d/fields" \
      "$BIN" "$DECK" \
        amr.n_cell="$NCELL $NCELL $NCELL" amr.max_grid_size=$NCELL \
        max_step=$STEPS amr.plot_int=$INT amr.check_int=0 \
        stop_time=1.0 prob.exp_energy=$energy nyx.cfl=$cfl \
        nyx.v=0 amr.v=0 > "$d/nyx.log" 2>&1 )
  local rc=$?
  local wall=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')
  rm -rf "$w"
  if [ $rc -ne 0 ]; then echo "   FAILED rc=$rc"; tail -6 "$d/nyx.log"; return 1; fi
  echo "   sim wall ${wall}s, $(ls -d "$d"/fields/plt* 2>/dev/null | wc -l) frames"
  "$EV" --source f32 --dir "$d/fields" --out "$OUT/ev/$name" \
        --block 1048576 --step-scale $INT --label "$name" 2>&1 | sed 's/^/   /'
  python3 - "$OUT/ev/$name/evolution.json" "$wall" \
      "exp_energy=$energy nyx.cfl=$cfl ncell=$NCELL" <<'PY'
import json,sys
p,wall,params=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(p)); d["sim_wall_s"]=float(wall); d["params"]=params
json.dump(d,open(p,"w"),indent=1)
PY
  rm -rf "$d/fields"
}

run_one e1_cfl08  1.0  0.8
run_one e10_cfl09 10.0 0.9
echo "== done"
