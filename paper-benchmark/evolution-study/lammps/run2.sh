#!/usr/bin/env bash
# LAMMPS round 2. Round 1's hot configurations gained their evolution partly
# through a neighbour list that is too coarse for them: upstream's
# `neighbor 0.3 bin` + `neigh_modify every 20 delay 0 check no` is sized for
# the example's T=3.0, and at T=6 an atom crosses the 0.3 sigma skin between
# rebuilds, so pairs are missed and NVE loses 3.5% of its total energy.
#
# Round 2 re-runs the hot configurations with a skin and rebuild cadence that
# actually cover their displacement, and re-runs the upstream one the same way
# as a control -- so the comparison is between physics, not between neighbour
# list errors.
set -uo pipefail
HERE=/home/cc/clio-core/paper-benchmark/lammps
EV=/home/cc/clio-core/paper-benchmark/evolution.py
OUT=/tmp/lmp-study
STEPS=1000 GAP=10 BOX=40
NB=(--skin 0.8 --every 5)

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
  echo "   wall ${wall}s"
  awk '$1 ~ /^[0-9]+$/ && NF==6 {t[$1]=$2; e[$1]=$5} END{n=0; for(k in t) if(k+0>n) n=k+0;
       printf "   T %.3f -> %.3f   TotEng %.5f -> %.5f  (drift %+.2f%%)\n",
       t[0],t[n],e[0],e[n],100*(e[n]-e[0])/(e[0]<0?-e[0]:(e[0]==0?1:e[0]))}' "$d/log.lammps"
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

run_one melt_nb      "${NB[@]}" --temp 3.0
run_one melt_hot_nb  "${NB[@]}" --temp 6.0
run_one ramp_hot_nb  "${NB[@]}" --deck "$HERE/in.melt_ramp" \
                     --var T0=0.05 --var T1=12.0 --var NSTEPS=$STEPS
echo "== done"
