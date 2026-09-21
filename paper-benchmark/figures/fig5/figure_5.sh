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

ARGV="$*"                          # kept for run.log; the parse loop shifts $* away
WLS=() SIZE=full DRY=0 OUT="" REPEATS=1 MAXF=0 REUSE=0 COST=balance TARGET=0 MODE=explore
WL_TIMEOUT=${WL_TIMEOUT:-540}   # one workload per 10-minute job

usage() { sed -n '3,16p' "$0" >&2; cat >&2 <<'U'

  --workload, -w  vpic | nyx | warpx | lammps | ai | synth (repeatable; default all six)
  --size, -s      full | smoke                  (default full: the paper sizes)
  --out DIR       results directory             (default ./results)
  --repeats, -r   measured replays per workload (default 1; the dumps are
                  generated once and replayed N times, and every plate is a
                  mean over all N x chunks)
  --chunk BYTES   chunk size                    (default 8 MiB)
  --cost MODEL    balance | ratio               (default balance: the cost
                  model weights compression time, decompression time and I/O
                  equally. ratio zeroes the two time terms, so the selector
                  optimises stored bytes alone)
  --mode MODE     explore | learn                (default explore: the
                  selector also measures K candidate codecs per chunk when it
                  is unsure. learn keeps inference and online SGD but makes
                  no trial compressions, so explore_ms is zero and the bars
                  show what a deployed selector costs rather than what a
                  training one does)
  --max-files N   replay only the first N dump files per rep (0 = all). The
                  stage SHARES are a mean over chunks, so a few files are
                  enough for them; the per-dump table shrinks with it.
  --chunks N      replay about N chunks per rep, whatever the workload (0 =
                  all). A file cap is not comparable across workloads -- an
                  AI checkpoint is 343 MB and a Nyx field 1 MB, so the same
                  --max-files is 86 chunks of one and 1 of the other. This
                  converts N into a per-workload file cap from the measured
                  file size, which is what keeps a re-run cheap without
                  making one workload's sample 80x another's.
  --reuse-fields  keep dumps already under <out>/figure_5_<wl>/fields instead
                  of re-running the simulation
  --dry-run       print the commands and stop
U
exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --workload|-w) WLS+=("$2"); shift 2 ;;
    --size|-s)     SIZE=$2; shift 2 ;;
    --out)         OUT=$2; shift 2 ;;
    --repeats|-r)  REPEATS=$2; shift 2 ;;
    --chunk)       CHUNK=$2; shift 2 ;;
    --max-files)   MAXF=$2; shift 2 ;;
    --chunks)      TARGET=$2; shift 2 ;;
    --cost)        COST=$2; shift 2 ;;
    --mode)        MODE=$2; shift 2 ;;
    --reuse-fields) REUSE=1; shift ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done
[ ${#WLS[@]} -eq 0 ] && WLS=(vpic nyx warpx lammps ai synth)
for w in "${WLS[@]}"; do
  case "$w" in vpic|nyx|warpx|lammps|ai|synth) ;; *) echo "bad --workload $w" >&2; usage ;; esac
done
case "$SIZE" in full|smoke) ;; *) echo "bad --size" >&2; usage ;; esac
case "$COST" in balance|ratio) ;; *) echo "bad --cost $COST" >&2; usage ;; esac
case "$MODE" in explore|learn) ;; *) echo "bad --mode $MODE" >&2; usage ;; esac
# EVERY COST MODEL GETS ITS OWN TREE. The two answer different questions and
# their per-chunk numbers are not comparable in place -- balance drops Nyx
# from 41.8x to 12.2x and its chunk from 20.4 ms to 0.8 ms -- so a second run
# used to overwrite the first and the plates silently mixed. Both the run
# tree and the plots are keyed by the cost model, so the two survive side by
# side and can be diffed.
OUT=${OUT:-$HERE/results}/$COST    # the run tree lands beside this script
mkdir -p "$OUT"
FIGDIR=$HERE/$COST; [ "$SIZE" = smoke ] && FIGDIR=$HERE/smoke/$COST
mkdir -p "$FIGDIR"
FIELDS_ROOT=${OUT%/*}/fields; mkdir -p "$FIELDS_ROOT"

# The whole console, kept with the run it describes. Without it the only
# record of a sweep is this terminal, and the per-rep logs do not carry the
# header saying which mode, cost model and chunk size produced them.
exec > >(tee -a "$OUT/run.log") 2>&1
echo "### $(date -Is)  $0 $ARGV"

# ---- NeuroPress configuration (every workload) -------------------------------
if [ "$MODE" = learn ]; then
  # `learn` is the balanced arm's name in run_config.sh; there is no
  # `learn-balance`. No --explore-* flags at all, so nothing trial-compresses
  # and explore_ms stays 0 for every chunk.
  CONFIG=learn; [ "$COST" = ratio ] && CONFIG=learn-ratio
  EXPLORE=()
else
  CONFIG=explore-$COST
  EXPLORE=(--explore-k 4 --explore-thresh 0.20)
fi
CHUNK=${CHUNK:-8388608}
NP_ENV=(BENCH_NP_LR=0.2 BENCH_NP_MAPE=0.10 MEASURE_DT=0 MEASURE_QUALITY=0)

# ---- Workload sizes ------------------------------------------------------------
if [ "$SIZE" = full ]; then
  # EVERY FIELD IS A WHOLE NUMBER OF 4 MiB CHUNKS. A field smaller than the
  # chunk becomes one short chunk and no --chunk can widen it, so the grid
  # is what has to change:
  #   nyx    128^3 x f32       =  8 MiB = 2 chunks   (64^3 was 1 MiB)
  #   warpx  64x64x256 x f32   =  4 MiB = 1 chunk    (64x64x128 was 2 MiB)
  #   lammps 4*64^3*3 x f32    = 12 MiB = 3 chunks   (box 40 was 2.93 MiB)
  # VPIC (13.396 MiB) and AI (327.3 MiB, an upstream checkpoint) do not
  # divide evenly and cannot be made to; st_full() drops their short tail.
  VPIC_NCELL=150 VPIC_NPPC=2 VPIC_WARMUP=500 VPIC_DUMPS=18 VPIC_INT=190
  NYX_NCELL=128 NYX_STEPS=50 NYX_PLOT_INT=10
  WARPX_NCELL="64 64 256" WARPX_STEPS=50 WARPX_INT=10
  LMP_BOX=64 LMP_WARMUP=100 LMP_DUMPS=5 LMP_GAP=100
  SYNTH_FRAMES=8
else
  VPIC_NCELL=64 VPIC_NPPC=2 VPIC_WARMUP=50 VPIC_DUMPS=3 VPIC_INT=50
  NYX_NCELL=32 NYX_STEPS=20 NYX_PLOT_INT=10
  WARPX_NCELL="32 32 64" WARPX_STEPS=20 WARPX_INT=10
  LMP_BOX=20 LMP_WARMUP=100 LMP_DUMPS=2 LMP_GAP=100
  SYNTH_FRAMES=3
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
declare -A NAME=([vpic]=VPIC [nyx]=Nyx [warpx]=WarpX [lammps]=LAMMPS [ai]=AI [synth]=Synthetic)

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

gen_wl() {  # gen_wl <workload> <dir> <fields>: the dumps, made once
  local wl=$1 dir=$2 fields=$3
  # --reuse-fields: nothing a rep varies changes the dumps, so a
  # re-measurement should not pay for the simulation a second time.
  if [ "$REUSE" = 1 ] && [ -n "$(find "$fields" -name '*.f32' -print -quit 2>/dev/null)" ]; then
    echo "     reusing $(find "$fields" -name '*.f32' | wc -l) dump file(s) in $fields"
    return 0
  fi
  case "$wl" in
    vpic)
      step "vpic simulation dumps fields (${VPIC_NCELL}^3, $VPIC_STEPS steps, dump every $VPIC_INT)" "$dir/gen.log" \
        "$BENCH/vpic/gen_fields.sh" --ncell "$VPIC_NCELL" --nppc "$VPIC_NPPC" \
          --steps "$VPIC_STEPS" --dump-int "$VPIC_INT" --out "$fields" \
          || { echo "     field dump failed -- see $dir/gen.log"; return 1; }
      # The warmup frames were trimmed from the plot when this ran in situ.
      if [ "$DRY" != 1 ]; then
        for d in $(ls -d "$fields"/plt* 2>/dev/null | head -n $((VPIC_WARMUP / VPIC_INT))); do rm -rf "$d"; done
      fi ;;
    ai)
      [ -d "$fields" ] && [ -n "$(find "$fields" -name '*.f32' -print -quit 2>/dev/null)" ] \
        || { echo "     no AI checkpoints at $fields -- set AI_FIELDS, or run ../../ai/gen_fields.sh" >&2; return 1; }
      echo "     replaying $(find "$fields" -name '*.f32' | wc -l) checkpoint file(s) in $fields" ;;
    synth)
      # Not a simulation: a periodic field, a random one and their mixture,
      # each exactly one chunk. They bracket the four workloads -- one end
      # compresses away, the other cannot be compressed at all.
      step "synthetic fields (periodic/random/mixed, $SYNTH_FRAMES frames)" "$dir/gen.log" \
        python3 "$HERE/gen_synth.py" --out "$fields" --frames "$SYNTH_FRAMES" \
          --shape "$((CHUNK / 4 / 128 / 64)),128,64" \
          || { echo "     field dump failed -- see $dir/gen.log"; return 1; } ;;
    nyx)
      step "nyx simulation dumps fields (${NYX_NCELL}^3, $NYX_STEPS steps)" "$dir/gen.log" \
        "$BENCH/nyx/gen_fields.sh" --ncell "$NYX_NCELL" --steps "$NYX_STEPS" \
          --plot-int "$NYX_PLOT_INT" --out "$fields" --bin "$NYX_BIN" \
          || { echo "     field dump failed -- see $dir/gen.log"; return 1; } ;;
    warpx)
      step "warpx simulation dumps fields ($WARPX_NCELL, $WARPX_STEPS steps)" "$dir/gen.log" \
        "$BENCH/analysis/validate/warpx_gen_fields.sh" --ncell "$WARPX_NCELL" \
          --steps "$WARPX_STEPS" --interval "$WARPX_INT" --out "$fields" \
          || { echo "     field dump failed -- see $dir/gen.log"; return 1; } ;;
    lammps)
      step "lammps simulation dumps fields (box $LMP_BOX, $LMP_STEPS steps, dump every $LMP_GAP)" "$dir/gen.log" \
        "$BENCH/lammps/run_config.sh" baseline --box "$LMP_BOX" --steps "$LMP_STEPS" \
          --gap "$LMP_GAP" --chunk "$CHUNK" --f32 --require-device \
          --raw "$fields" --results "$dir/gen" --tag "${wl}gen" \
          || { echo "     field dump failed -- see $dir/gen.log"; return 1; }
      # The warmup frames were trimmed from the plot when this ran in situ.
      if [ "$DRY" != 1 ]; then
        for d in "$fields"/step*; do
          [ -d "$d" ] || continue
          st=$((10#${d##*/step}))
          [ "$st" -le "$LMP_WARMUP" ] && rm -rf "$d"
        done
      fi ;;
  esac
  # The trim loops above end on a false test whenever the last frame is past
  # the warmup, and that status would otherwise become this function's --
  # which the caller reads as "the dumps failed" and skips the workload.
  return 0
}

files_for_chunks() {  # files_for_chunks <fields>: the --max-files that gets ~TARGET chunks
  local fields=$1 first sz per
  [ "$TARGET" -gt 0 ] || { echo "$MAXF"; return; }
  first=$(find "$fields" -name '*.f32' -print -quit 2>/dev/null)
  [ -n "$first" ] || { echo "$MAXF"; return; }
  sz=$(stat -c %s "$first")
  per=$(( (sz + CHUNK - 1) / CHUNK )); [ "$per" -lt 1 ] && per=1
  echo $(( (TARGET + per - 1) / per ))
}

replay_wl() {  # replay_wl <workload> <dir> <fields> <rep>: ONE measured pass
  # Every workload replays .f32 dumps through the same driver, so the four are
  # measured the same way; the chunks are host-resident, hence --stage-h2d.
  local wl=$1 dir=$2 fields=$3 rep=$4
  local maxf; maxf=$(files_for_chunks "$fields")
  local -a env_kv=( "CLIO_NEUROPRESS_PHASE_LOG=$dir/phase_r$rep.csv" "${NP_ENV[@]}" )
  step "$wl replay through Clio (rep $rep/$REPEATS)" "$dir/console_r$rep.log" \
    env "${env_kv[@]}" CLIO_NEUROPRESS_STAGE_H2D=1 "$BENCH/nyx/run_config.sh" "$CONFIG" \
      --fields "$fields" --chunk "$CHUNK" ${EXPLORE[@]+"${EXPLORE[@]}"} \
      ${maxf:+--max-files "$maxf"} \
      --results "$dir" --tag "${wl}_r$rep"
}

run_wl() {
  local wl=$1 dir="$OUT/figure_5_$1" warmup=-1
  # DUMPS ARE SHARED BY EVERY COST MODEL, one level above the per-cost tree.
  # The cost model changes which codec is chosen, never the bytes the
  # simulation wrote, and VPIC's dumps alone are 3.8 GB -- a copy per cost
  # model would cost 5 GB and a re-simulation to say the same thing.
  local fields="$FIELDS_ROOT/$wl"
  # AI replays checkpoints an upstream training run exported; regenerating
  # them means training a ViT, so the dumps are read where they already are.
  [ "$wl" = ai ] && [ -d "${AI_FIELDS:-}" ] && fields=$AI_FIELDS
  mkdir -p "$dir"
  rm -f "$dir"/phase_r*.csv "$dir"/benchmark_${wl}_r*_timesteps.csv "$dir/phase.csv"
  echo "---- ${NAME[$wl]} ----"
  gen_wl "$wl" "$dir" "$fields" || return 0
  if [ "$DRY" = 1 ]; then replay_wl "$wl" "$dir" "$fields" 1; return 0; fi

  local ok=0 k rc tag phase verdict
  for k in $(seq 1 "$REPEATS"); do
    replay_wl "$wl" "$dir" "$fields" "$k"; rc=$?
    tag="${wl}_r$k"; phase="$dir/phase_r$k.csv"
    verdict=$(grep -hE "VERIFIED:|FAILED:|BOUND (OK|FAILED)" "$dir/$tag/stdout.log" 2>/dev/null | tail -1)
    echo "     rep $k/$REPEATS: rc=$rc  ${verdict:-(no verification line)}"
    # The store holds the run's raw bytes -- gigabytes per rep, and nothing
    # downstream reads them. The phase log is what a rep is kept for.
    rm -f "$dir/$tag/chi_bdev.dat" "$dir/$tag"/cte_tier.dat* 2>/dev/null
    if [ "$rc" -ne 0 ] || [ ! -s "$phase" ] || echo "$verdict" | grep -q "FAILED"; then
      echo "     rep $k NOT TABULATED -- see $dir/console_r$k.log"
      continue
    fi
    python3 "$HERE/plot_fig5.py" table "$phase" --workload "${NAME[$wl]}" \
      --warmup-step "$warmup" --out "$dir/benchmark_${wl}_r${k}_timesteps.csv" > /dev/null
    ok=$((ok + 1))
  done
  [ "$ok" -gt 0 ] || { echo "     NOTHING TABULATED for $wl"; return 0; }

  # One phase log for the plate: every rep's chunks end to end, so the bars
  # are a mean over (reps x chunks) rather than over one pass.
  awk 'FNR == 1 && NR != 1 { next } { print }' "$dir"/phase_r*.csv > "$dir/phase.csv"
  # The summary reads the per-rep tables instead, because a dump repeated
  # across reps must be averaged, not summed into one giant dump.
  python3 "$HERE/plot_fig5.py" summary --timesteps "$dir"/benchmark_${wl}_r*_timesteps.csv \
    --no-defaults > "$dir/summary.txt" 2>&1 \
    && grep -E "^${NAME[$wl]} " "$dir/summary.txt" | sed 's/^/     /' \
    || echo "     summary failed -- see $dir/summary.txt"
  echo "     $ok/$REPEATS rep(s) tabulated"
}

echo "== figure 5 ($SIZE): ${WLS[*]} x $REPEATS rep(s), $MODE mode, $COST cost model, chunk $((CHUNK / 1048576)) MiB${TARGET:+, ~$TARGET chunks/rep}${MAXF:+, first $MAXF file(s)} -> $OUT"
[ "$DRY" != 1 ] && clean_leftovers
for wl in "${WLS[@]}"; do run_wl "$wl"; done
[ "$DRY" = 1 ] && exit 0

# The plates read EVERY workload's phase log at once, so they are drawn after
# the loop rather than inside it, and from the run tree rather than from the
# per-dump tables -- a mean over chunks, not over dumps.
#
# WRITE AND READ ARE SEPARATE FIGURES. --split gives the two \columnwidth
# plates, which is what the paper places. The combined \textwidth plate is
# NOT generated: it forced both panels onto one share axis and one legend,
# and the two paths have neither the same stages nor the same magnitudes.
# `plot_fig5.py stacked` without --split still draws it for anyone who wants
# it by hand.
echo
mkdir -p "$FIGDIR"
python3 "$HERE/plot_fig5.py" stacked --results "$OUT" --out "$FIGDIR" --split \
  | sed 's/^/     /'

