#!/usr/bin/env bash
#===============================================================================
# figure_5.sh -- Figure 5: where time goes inside Clio-NeuroPress, per chunk,
# on the write and the read path.
#
#   ./figure_5.sh -w vpic                  # one workload, paper size
#   ./figure_5.sh -w nyx -w warpx -s smoke
#   ./figure_5.sh --dry-run                # all four, print the commands
#
# Output per workload in <out>/figure_5_<workload>/: phase.csv (the runtime's
# CLIO_NEUROPRESS_PHASE_LOG), benchmark_<workload>_timesteps.csv and
# fig5_<workload>.png, copied here, beside this script.
#
# NeuroPress: explore-balance, K=4, threshold 0.20, learning rate 0.2, SGD
# above 10% error, lossless, 8 MiB, no write-time decompression. VPIC and LAMMPS
# run in situ, Nyx and WarpX dump then replay. Warmup dumps are excluded.
#
# Environment: NYX_BIN, WARPX_BIN, DECK (warpx), WL_TIMEOUT (s, default 540).
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)   # this figure's own directory: the plots land here
BENCH=$(cd "$HERE/../.." && pwd)      # paper-benchmark/: the harnesses and the plotters

WLS=() SIZE=full DRY=0 OUT=""
WL_TIMEOUT=${WL_TIMEOUT:-540}   # one workload per 10-minute job

usage() { sed -n '3,16p' "$0" >&2; cat >&2 <<'U'

  --workload, -w  vpic | nyx | warpx | lammps   (repeatable; default all four)
  --size, -s      full | smoke                  (default full: the paper sizes)
  --out DIR       results directory             (default results/figure5)
  --dry-run       print the commands and stop
U
exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --workload|-w) WLS+=("$2"); shift 2 ;;
    --size|-s)     SIZE=$2; shift 2 ;;
    --out)         OUT=$2; shift 2 ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done
[ ${#WLS[@]} -eq 0 ] && WLS=(vpic nyx warpx lammps)
for w in "${WLS[@]}"; do
  case "$w" in vpic|nyx|warpx|lammps) ;; *) echo "bad --workload $w" >&2; usage ;; esac
done
case "$SIZE" in full|smoke) ;; *) echo "bad --size" >&2; usage ;; esac
OUT=${OUT:-$BENCH/results/figure5}
mkdir -p "$OUT"
FIGDIR=$HERE; [ "$SIZE" = smoke ] && FIGDIR=$HERE/smoke   # the plots land beside this script

# ---- NeuroPress configuration (every workload) -------------------------------
CONFIG=explore-balance
EXPLORE=(--explore-k 4 --explore-thresh 0.20)
CHUNK=${CHUNK:-8388608}
NP_ENV=(BENCH_NP_LR=0.2 BENCH_NP_MAPE=0.10 MEASURE_DT=0 MEASURE_QUALITY=0)

# ---- Workload sizes ------------------------------------------------------------
if [ "$SIZE" = full ]; then
  VPIC_NCELL=150 VPIC_NPPC=2 VPIC_WARMUP=500 VPIC_DUMPS=18 VPIC_INT=190
  NYX_NCELL=64 NYX_STEPS=50 NYX_PLOT_INT=10
  WARPX_NCELL="64 64 128" WARPX_STEPS=50 WARPX_INT=10
  LMP_BOX=40 LMP_WARMUP=100 LMP_DUMPS=5 LMP_GAP=100
else
  VPIC_NCELL=64 VPIC_NPPC=2 VPIC_WARMUP=50 VPIC_DUMPS=3 VPIC_INT=50
  NYX_NCELL=32 NYX_STEPS=20 NYX_PLOT_INT=10
  WARPX_NCELL="32 32 64" WARPX_STEPS=20 WARPX_INT=10
  LMP_BOX=20 LMP_WARMUP=100 LMP_DUMPS=2 LMP_GAP=100
fi
# Dumps land on multiples of the interval. Run just long enough that exactly
# DUMPS of them fall after the warmup.
last_step() {  # last_step <warmup> <dumps> <interval>
  echo $(( ($1 / $3 + 1) * $3 + ($2 - 1) * $3 ))
}
VPIC_STEPS=$(last_step $VPIC_WARMUP $VPIC_DUMPS $VPIC_INT)
LMP_STEPS=$(last_step $LMP_WARMUP $LMP_DUMPS $LMP_GAP)

# Machine-specific paths (gitignored); sourced after the arg loop, so $SIZE is known.
SITE=${SITE:-$BENCH/site.sh}
# shellcheck source=/dev/null
[ -f "$SITE" ] && . "$SITE"

NYX_BIN=${NYX_BIN:-/u/imuradli/np-build/nyx/Exec/HydroTests/nyx_HydroTests}
declare -A NAME=([vpic]=VPIC [nyx]=Nyx [warpx]=WarpX [lammps]=LAMMPS)

# A runtime killed by the timeout leaves its IPC state behind; the next
# workload must not inherit it. This user's files only.
MEMFD_DIR=${CLIO_MEMFD_DIR:-/tmp/clio_${USER:-unknown}}
clean_leftovers() {
  pkill -KILL -u "$USER" -x neuropress_fiel 2>/dev/null
  rm -f /tmp/clio_*.ipc /tmp/clio_server_timing.log 2>/dev/null
  rm -rf /tmp/clio_memfd "$MEMFD_DIR" 2>/dev/null
  find /dev/shm -maxdepth 1 -user "$USER" -name 'sm_segment.*' -delete 2>/dev/null
}

step() {  # step <label> <log> <cmd...>: run one timed-out stage, echo rc
  local label=$1 log=$2; shift 2
  if [ "$DRY" = 1 ]; then printf '   DRY %s:\n        %s\n' "$label" "$*"; return 0; fi
  timeout -k 5 "$WL_TIMEOUT" "$@" > "$log" 2>&1
  local rc=$?
  { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; } && { echo "     $label: TIMEOUT after ${WL_TIMEOUT}s"; clean_leftovers; }
  return $rc
}

run_wl() {
  local wl=$1 dir="$OUT/figure_5_$1"
  local phase="$dir/phase.csv" warmup=-1
  mkdir -p "$dir"; rm -f "$phase"
  local -a env_kv=( "CLIO_NEUROPRESS_PHASE_LOG=$phase" "${NP_ENV[@]}" )
  echo "---- ${NAME[$wl]} ----"

  case "$wl" in
    vpic)
      local fields="$dir/fields"
      step "vpic simulation dumps fields (${VPIC_NCELL}^3, $VPIC_STEPS steps, dump every $VPIC_INT)" "$dir/gen.log" \
        "$BENCH/vpic/gen_fields.sh" --ncell "$VPIC_NCELL" --nppc "$VPIC_NPPC" \
          --steps "$VPIC_STEPS" --dump-int "$VPIC_INT" --out "$fields" \
          || { echo "     field dump failed -- see $dir/gen.log"; return 0; }
      # The warmup frames were trimmed from the plot when this ran in situ.
      if [ "$DRY" != 1 ]; then
        for d in $(ls -d "$fields"/plt* 2>/dev/null | head -n $((VPIC_WARMUP / VPIC_INT))); do rm -rf "$d"; done
      fi
      step "vpic replay through Clio" "$dir/console.log" \
        env "${env_kv[@]}" CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG" \
          --fields "$fields" --chunk "$CHUNK" "${EXPLORE[@]}" --results "$dir" --tag "$wl" ;;
    nyx|warpx)
      local fields="$dir/fields"
      if [ "$wl" = nyx ]; then
        step "nyx simulation dumps fields (${NYX_NCELL}^3, $NYX_STEPS steps)" "$dir/gen.log" \
          "$BENCH/nyx/gen_fields.sh" --ncell "$NYX_NCELL" --steps "$NYX_STEPS" \
            --plot-int "$NYX_PLOT_INT" --out "$fields" --bin "$NYX_BIN" || { echo "     field dump failed -- see $dir/gen.log"; return 0; }
      else
        step "warpx simulation dumps fields ($WARPX_NCELL, $WARPX_STEPS steps)" "$dir/gen.log" \
          "$BENCH/analysis/validate/warpx_gen_fields.sh" --ncell "$WARPX_NCELL" \
            --steps "$WARPX_STEPS" --interval "$WARPX_INT" --out "$fields" || { echo "     field dump failed -- see $dir/gen.log"; return 0; }
      fi
      # Host-memory chunks: staged up to the GPU so every kernel runs there.
      step "$wl replay through Clio" "$dir/console.log" \
        env "${env_kv[@]}" CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG" \
          --fields "$fields" --chunk "$CHUNK" "${EXPLORE[@]}" --results "$dir" --tag "$wl" ;;
    lammps)
      local fields="$dir/fields"
      step "lammps simulation dumps fields (box $LMP_BOX, $LMP_STEPS steps, dump every $LMP_GAP)" "$dir/gen.log" \
        "$BENCH/lammps/run_config.sh" baseline --box "$LMP_BOX" --steps "$LMP_STEPS" \
          --gap "$LMP_GAP" --chunk "$CHUNK" --f32 --require-device \
          --raw "$fields" --results "$dir/gen" --tag "${wl}gen" \
          || { echo "     field dump failed -- see $dir/gen.log"; return 0; }
      # The warmup frames were trimmed from the plot when this ran in situ.
      if [ "$DRY" != 1 ]; then
        for d in "$fields"/step*; do
          [ -d "$d" ] || continue
          st=$((10#${d##*/step}))
          [ "$st" -le "$LMP_WARMUP" ] && rm -rf "$d"
        done
      fi
      step "lammps replay through Clio" "$dir/console.log" \
        env "${env_kv[@]}" CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG" \
          --fields "$fields" --chunk "$CHUNK" "${EXPLORE[@]}" --results "$dir" --tag "$wl" ;;
  esac
  local rc=$?
  [ "$DRY" = 1 ] && return 0

  local verdict
  verdict=$(grep -hE "VERIFIED:|FAILED:|BOUND (OK|FAILED)" "$dir/$wl/stdout.log" 2>/dev/null | tail -2 | tr '\n' ' ')
  echo "     rc=$rc  ${verdict:-(no verification line)}"
  if [ "$rc" -ne 0 ] || [ ! -s "$phase" ] || echo "$verdict" | grep -q "FAILED"; then
    echo "     NOT TABULATED: the run, its phase log, or its round-trip check failed -- see $dir/console.log"
    return 0
  fi
  local ts="$dir/benchmark_${wl}_timesteps.csv"
  python3 "$HERE/plot_fig5.py" table "$phase" --workload "${NAME[$wl]}" \
    --warmup-step "$warmup" --out "$ts" | sed 's/^/     /'
  python3 "$HERE/plot_fig5.py" pies --timesteps "$ts" --no-defaults \
    --out "$dir" > "$dir/plot.log" 2>&1 \
    && grep -E "^wrote .*fig5_${wl}\.png" "$dir/plot.log" | sed 's/^/     /' \
    && mkdir -p "$FIGDIR" && cp "$dir/fig5_${wl}.png" "$FIGDIR/" \
    && echo "     in repo: $FIGDIR/fig5_${wl}.png" \
    || echo "     plot failed -- see $dir/plot.log"
}

echo "== figure 5 ($SIZE): ${WLS[*]} -> $OUT"
[ "$DRY" != 1 ] && clean_leftovers
for wl in "${WLS[@]}"; do run_wl "$wl"; done
[ "$DRY" = 1 ] && exit 0

