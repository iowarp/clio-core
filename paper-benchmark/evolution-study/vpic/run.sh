#!/usr/bin/env bash
# VPIC evolution study: 1000 steps, field dump every 10.
set -uo pipefail
HERE=/home/cc/clio-core/paper-benchmark/vpic
EV=/home/cc/clio-core/paper-benchmark/evolution.py
OUT=/tmp/vpic-study
STEPS=${STEPS:-1000} INT=${INT:-10} NCELL=${NCELL:-126} NPPC=${NPPC:-8}

# Upstream Weibel.cxx: vth_perp = 0.25/sqrt(2), vth_x = 0.05/sqrt(2).
V_BASE=0.1767766953  # 0.25/sqrt(2)
X_BASE=0.0353553391  # 0.05/sqrt(2)
V_HOT=0.3535533906   # 0.50/sqrt(2)
X_COLD=0.0176776695  # 0.025/sqrt(2)

run_one() {
  local name=$1 vthe=$2 vthex=$3 clean=$4
  local d=$OUT/$name
  echo "== $name: vthe=$vthe vthex=$vthex clean_div=$clean"
  rm -rf "$d"; mkdir -p "$d"
  local w=$(mktemp -d)
  local t0=$(date +%s.%N)
  ( cd "$w" && env VPIC_NX=$NCELL VPIC_NY=$NCELL VPIC_NZ=$NCELL VPIC_NPPC=$NPPC \
      VPIC_STEPS=$STEPS VPIC_DUMP_INT=$INT VPIC_CLEAN_DIV_INT=$clean \
      VPIC_VTHE=$vthe VPIC_VTHEX=$vthex \
      VPIC_DUMP_FIELDS=1 VPIC_DUMP_DIR="$d/fields" \
      "$HERE/weibel_clio.Linux" > "$d/vpic.log" 2>&1 )
  local rc=$?
  local wall=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')
  rm -rf "$w"
  if [ $rc -ne 0 ]; then echo "   FAILED rc=$rc"; tail -5 "$d/vpic.log"; return 1; fi
  echo "   sim wall ${wall}s, $(ls -d "$d"/fields/plt* 2>/dev/null | wc -l) frames"
  "$EV" --source f32 --dir "$d/fields" --out "$OUT/ev/$name" \
        --block 1048576 --step-scale $INT --label "$name" 2>&1 | sed 's/^/   /'
  python3 - "$OUT/ev/$name/evolution.json" "$wall" \
      "vthe=$vthe vthex=$vthex clean_div_interval=$clean nppc=$NPPC" <<'PY'
import json,sys
p,wall,params=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(p)); d["sim_wall_s"]=float(wall); d["params"]=params
json.dump(d,open(p,"w"),indent=1)
PY
  grep -E "dx/debye|vthe/c|dt|courant" "$d/vpic.log" | head -4 | sed 's/^/   /'
  rm -rf "$d/fields"   # 12.8 GB per config; the metric has already read it
}

mkdir -p "$OUT/ev"
run_one baseline   $V_BASE $X_BASE  0
run_one cleandiv   $V_BASE $X_BASE  10
run_one hot_a100   $V_HOT  $X_BASE  10
run_one coldx_a100 $V_BASE $X_COLD  10
echo "== done"
