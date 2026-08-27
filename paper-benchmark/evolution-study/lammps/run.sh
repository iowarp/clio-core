#!/usr/bin/env bash
# LAMMPS evolution study: 1000 steps, a frame every 10.
#
# Runs through run_config.sh with --raw so the measured bytes are exactly the
# bytes the compressor is handed (float64, natoms x 3, atom-ID order). This
# workload is in situ -- there is no dump file to read otherwise.
set -uo pipefail
HERE=/home/cc/clio-core/paper-benchmark/lammps
EV=/home/cc/clio-core/paper-benchmark/evolution.py
OUT=/tmp/lmp-study
STEPS=${STEPS:-1000} GAP=${GAP:-10} BOX=${BOX:-40}

run_one() {
  local name=$1; shift
  local d=$OUT/$name
  echo "== $name: $*"
  rm -rf "$d" "$OUT/raw_$name"; mkdir -p "$d"
  local t0=$(date +%s.%N)
  "$HERE/run_config.sh" static-zstd --box $BOX --steps $STEPS --gap $GAP \
      --no-verify --raw "$OUT/raw_$name" --results "$OUT" --tag "$name" \
      "$@" > "$d.launch.log" 2>&1
  local rc=$?
  local wall=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')
  if [ $rc -ne 0 ]; then echo "   FAILED rc=$rc"; tail -8 "$d.launch.log"; return 1; fi
  echo "   wall ${wall}s, $(ls "$OUT/raw_$name"/position_step_*_chunk_0.bin 2>/dev/null | wc -l) frames"
  "$EV" --source raw --dir "$OUT/raw_$name" --out "$OUT/ev/$name" \
        --block 1048576 --f64 --label "$name" 2>&1 | sed 's/^/   /'
  python3 - "$OUT/ev/$name/evolution.json" "$wall" "$*" <<'PY'
import json,sys
p,wall,params=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(p)); d["sim_wall_s"]=float(wall); d["params"]=params
json.dump(d,open(p,"w"),indent=1)
PY
  rm -rf "$OUT/raw_$name"
}

mkdir -p "$OUT/ev"
run_one melt      --temp 3.0
run_one melt_hot  --temp 6.0
run_one ramp      --deck "$HERE/in.melt_ramp" --var T0=0.05 --var T1=6.0  --var NSTEPS=$STEPS
run_one ramp_hot  --deck "$HERE/in.melt_ramp" --var T0=0.05 --var T1=12.0 --var NSTEPS=$STEPS
echo "== done"
