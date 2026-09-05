#!/usr/bin/env bash
# Which compression configuration works well for which data.
#
# Builds a PER-CHUNK ORACLE -- replays every chunk through every codec, so the
# best possible choice for each individual chunk is known -- then reports:
#
#   1. Mean per-chunk cost (compression + transfer at storage bandwidth):
#      per-chunk optimum / NeuroPress / best fixed nvCOMP / ndzip / cuSZp3 / cuSZ
#   2. The oracle's per-chunk winner mix, broken down by data content.
#
# Plus the amortization crossover: the storage bandwidth below which a slow
# high-ratio codec (cuSZ) stops losing. The rebuttal asserts ~10-20 MB/s for
# cuSZ; this computes it from the measured kernel times rather than asserting.
#
#   ./compare_perchunk_oracle.sh --workload vpic --size smoke
#   ./compare_perchunk_oracle.sh --workload nyx  --size full --bw 1.2e6
#
# Companion: compare_wallclock.sh measures the same strategies END TO END.
# This one is the offline per-chunk analysis; that one is the stopwatch.
#
# WHY A SWEEP AND NOT FIVE TIMED ARMS. A per-chunk optimum needs every chunk
# compressed by EVERY candidate, not by the five strategies you would deploy.
# The `best` config does that -- best_mode measures all 32 nvcomp actions on
# every chunk -- and "best fixed nvCOMP codec" is then derived from the SAME
# measurements rather than from a separate timed run. That is what makes the
# rows comparable: one set of chunks, one set of timings.
#
# NOTE ON COST MODEL. NeuroPress is RUN under the cost model it is SCORED under
# (W_CT=1, W_IO=1). Scoring a ratio-only run with a time-inclusive cost grades
# it on an objective it was never given, and flatters the fixed baselines.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WL=""; SIZE=smoke; EB=1e-3; CHUNK=""; OUT=""; FIELDS=""; BW=1.2e6; SKIPRUN=0
usage() {
  echo "usage: compare_perchunk_oracle.sh --workload {nyx|vpic|warpx|lammps} --size {smoke|full}" >&2
  echo "       [--eb 1e-3] [--chunk BYTES] [--bw 1.2e6] [--fields DIR] [--out DIR]" >&2
  echo "       [--analyze-only]   reuse existing runs, just rebuild the tables" >&2
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in
    --workload|-w) WL=$2; shift 2 ;;
    --size|-s)     SIZE=$2; shift 2 ;;
    --eb)          EB=$2; shift 2 ;;
    --chunk)       CHUNK=$2; shift 2 ;;
    --bw)          BW=$2; shift 2 ;;
    --fields)      FIELDS=$2; shift 2 ;;
    --out)         OUT=$2; shift 2 ;;
    --analyze-only) SKIPRUN=1; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done
[ -n "$WL" ] || usage
case "$SIZE" in smoke|full) ;; *) echo "--size must be smoke or full" >&2; exit 2 ;; esac

TARGET_CHUNKS=2000
case "$SIZE" in
  smoke) CHUNK=${CHUNK:-2097152} ;;
  full)  CHUNK=${CHUNK:-8388608}; TARGET_CHUNKS=0 ;;
esac
OUT=${OUT:-$PWD/cc2/$WL-$SIZE}; mkdir -p "$OUT"

# ---- data, resolved once and shared by every arm -----------------------
case "$WL" in
  nyx)   FIELDS=${FIELDS:-${NYX_FIELDS:-$HERE/nyx/fields}} ;;
  vpic)  FIELDS=${FIELDS:-${VPIC_FIELDS:-$HERE/vpic/fields}} ;;
  warpx) FIELDS=${FIELDS:-${WARPX_FIELDS:-$HERE/warpx/fields}} ;;
  lammps) FIELDS=insitu ;;
  *) echo "unknown workload: $WL" >&2; exit 2 ;;
esac
LMP_BOX=${LMP_BOX:-80}; LMP_GAP=${LMP_GAP:-50}
case "$SIZE" in smoke) LMP_STEPS=${LMP_STEPS:-1450} ;; full) LMP_STEPS=${LMP_STEPS:-10900} ;; esac
LMP_FIELD=$(( 4*LMP_BOX*LMP_BOX*LMP_BOX*3*8 ))
case "$SIZE" in smoke) LMP_CHUNK=$CHUNK ;; full) LMP_CHUNK=$LMP_FIELD ;; esac
LMP_DECK=${LMP_DECK:-$HERE/lammps/in.melt_ramp}   # in.melt is stationary after step 40

if [ "$WL" != lammps ]; then
  [ -d "$FIELDS" ] || { echo "no dumps at $FIELDS -- generate them first" >&2; exit 1; }
  NFILES=$(find "$FIELDS" -name '*.f32' | wc -l)
  FSIZE=$(stat -c %s "$(find "$FIELDS" -name '*.f32' -print -quit)")
  PERFILE=$(( FSIZE / CHUNK )); [ "$PERFILE" -lt 1 ] && PERFILE=1
  if [ "$TARGET_CHUNKS" -gt 0 ]; then
    MAXF=$(( (TARGET_CHUNKS + PERFILE - 1) / PERFILE ))
    [ "$MAXF" -gt "$NFILES" ] && MAXF=$NFILES
  else MAXF=$NFILES; fi
  echo "== CC2 $WL/$SIZE: $MAXF files x $PERFILE = $((MAXF*PERFILE)) chunks of $((CHUNK/1048576)) MiB, eb $EB"
else
  echo "== CC2 $WL/$SIZE: in situ, $LMP_STEPS steps @ gap $LMP_GAP, chunk $((LMP_CHUNK/1048576)) MiB"
fi

# ---- arms --------------------------------------------------------------
# `best` is the oracle sweep: all 32 nvcomp actions measured on every chunk.
# The external codecs are separate arms because they sit outside the action
# space (kNeuroPressTrainedGpuBaseIds) and the sweep cannot reach them.
# THE REPLAY ROUTE MUST STAGE TO DEVICE. neuropress_field_replay reads .f32
# into HOST shm, and NeuroPress's quantizer and byte shuffle are CUDA-only --
# the host implementations were deliberately removed. Without this every chunk
# whose ranked action wants a shuffle is REFUSED: Compress returns rc=1, the
# blob is never stored, and the read back fails rc=11. Measured on vpic/smoke
# before this line existed: 2000 blobs, 0 B stored, 0 of 2000 verified.
# The runtime names this variable in its own refusal message
# (compressor_runtime.cc:1362). nyx/vpic run_config.sh have no --stage-h2d
# flag, so it is set here in the environment; they use `env VAR=...` rather
# than `env -i`, so an exported value reaches the binary.
# warpx and lammps hand the compressor device pointers already (VOL / in-process
# library), and the variable only acts on HOST-resident chunks, so it is inert
# for them.
case "$WL" in nyx|vpic) export CLIO_NEUROPRESS_STAGE_H2D=1 ;; esac

ARMS=("best" "learn" "static-cusz" "static-cuszp" "static-ndzip")

run_arm() {
  local cfg=$1 store="$OUT/$1"
  local cmd=("$HERE/$WL/run_config.sh" "$cfg" --eb "$EB" --results "$store" --tag "$cfg")
  if [ "$WL" = lammps ]; then
    cmd+=(--deck "$LMP_DECK" --box "$LMP_BOX" --steps "$LMP_STEPS" --gap "$LMP_GAP"
          --chunk "$LMP_CHUNK" --var "NSTEPS=$LMP_STEPS" --require-device)
  else
    cmd+=(--chunk "$CHUNK" --fields "$FIELDS" --max-files "$MAXF")
  fi
  # ndzip is lossless -> bit-exact digest; everything else is checked against eb.
  case "$cfg" in static-ndzip) : ;; *) cmd+=(--check-bound) ;; esac
  echo; echo "---- $cfg ----"
  local s=$(date +%s.%N)
  # NeuroPress must OPTIMISE the cost it is SCORED on, or the comparison is rigged.
  env CLIO_NEUROPRESS_COST_W_CT=1 CLIO_NEUROPRESS_COST_W_DT=0 \
      CLIO_NEUROPRESS_COST_W_IO=1 CLIO_NEUROPRESS_COST_BW="$BW" \
      "${cmd[@]}" > "$store.console" 2>&1
  local rc=$?
  echo "   rc=$rc wall=$(awk -v a=$s -v b=$(date +%s.%N) 'BEGIN{printf "%.1f",b-a}')s"
  grep -E "^stored|BOUND|VERIFIED" "$store.console" 2>/dev/null | head -3 | sed 's/^/   /'
  rm -f "$store/chi_bdev.dat" "$store"/cte_tier.dat*
}

if [ "$SKIPRUN" = 0 ]; then
  for cfg in "${ARMS[@]}"; do run_arm "$cfg"; done
else
  echo "   --analyze-only: reusing runs in $OUT"
fi

echo
python3 "$HERE/perchunk_oracle_tables.py" "$OUT" "$WL" "$BW"
