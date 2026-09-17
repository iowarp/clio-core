#!/usr/bin/env bash
#===============================================================================
# figure_8.sh -- Figure 8: does NeuroPress's online-learning model converge?
# Per chunk: regret of the model's pick against the best of all 32
# configurations, and the cost model's error.
#
#   ./figure_8.sh -s smoke                 # ~100 chunks per workload
#   ./figure_8.sh -s 2k -w vpic            # ~2,000 chunks, one workload
#   ./figure_8.sh --dry-run
#
# Output per workload in <out>/figure_8_<workload>/ (the run and trace.csv);
# plots go to figures/fig8/<size>/.
#
# Exploration is forced on every chunk (explore_k 31, threshold -1), so all 32
# configurations are measured (dt with MEASURE_DT=1). NeuroPress:
# explore-balance, eb 0.05, learning rate 0.2, SGD above 10% cost error, 8 MiB
# chunks. Tier and bdev images are deleted after each workload; --local DIR
# runs on a node-local disk.
#
# Environment: WL_TIMEOUT (s per workload), NYX_FIELDS, WARPX_FIELDS, AI_FIELDS.
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

WLS=() SIZE=smoke DRY=0 OUT="" LOCAL="" PLOT=1
usage() { sed -n '3,12p' "$0" >&2; cat >&2 <<'U'

  --workload, -w  vpic | nyx | warpx | lammps | ai   (repeatable; default the four simulations)
  --size, -s      smoke | full | 2k             (default smoke ~100 chunks; full ~1,000; 2k twice the dumps)
  --out DIR       results directory             (default results/figure8/<size>)
  --local DIR     run in DIR, copy back without the tier/bdev files
  --no-plot       write trace.csv only; plot later with plot_fig8.py
  --dry-run       print the commands and stop
U
exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --workload|-w) WLS+=("$2"); shift 2 ;;
    --size|-s)     SIZE=$2; shift 2 ;;
    --out)         OUT=$2; shift 2 ;;
    --local)       LOCAL=$2; shift 2 ;;
    --no-plot)     PLOT=0; shift ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done
[ ${#WLS[@]} -eq 0 ] && WLS=(vpic nyx warpx lammps)
for w in "${WLS[@]}"; do
  case "$w" in vpic|nyx|warpx|lammps|ai) ;; *) echo "bad --workload $w" >&2; usage ;; esac
done
case "$SIZE" in smoke|full|2k) ;; *) echo "bad --size" >&2; usage ;; esac
OUT=${OUT:-$HERE/results/figure8/$SIZE}
FIGDIR=$HERE/figures/fig8/$SIZE
mkdir -p "$OUT"

# ---- NeuroPress and measurement configuration (every workload) ---------------
CONFIG=explore-balance
NP=(--explore-k 31 --explore-thresh -1 --eb 0.05 --chunk 8388608)
# Prediction reuse off explicitly (an exploring run never uses it).
NP_ENV=(BENCH_NP_LR=0.2 BENCH_NP_MAPE=0.10 MEASURE_DT=1 MEASURE_QUALITY=0
        CLIO_NEUROPRESS_EXPLORE_STREAMS=1 CLIO_NEUROPRESS_REUSE_PREDICTIONS=0)

# ---- Sizes: chunks are 8 MiB on every workload -------------------------------
# Run lengths are each workload's README configuration; only the dump cadence
# changes. Chunks for full / 2k: vpic 992 / 2000, nyx 1008 / ~2064,
# warpx 1000 / 2000, lammps 1002 / 2004, ai 984 / 1968.
if [ "$SIZE" = smoke ]; then
  VPIC_STEPS=160  VPIC_INT=16 NYX_FILES=12   WARPX_FILES=100  LMP_STEPS=90   LMP_GAP=3  AI_FILES=4
  WL_TIMEOUT=${WL_TIMEOUT:-200}                                      # 160 / 96 / 100 / 93 chunks
elif [ "$SIZE" = full ]; then
  VPIC_STEPS=1000 VPIC_INT=16 NYX_FILES=0    WARPX_FILES=1000 LMP_STEPS=1000 LMP_GAP=3  AI_FILES=24
  WL_TIMEOUT=${WL_TIMEOUT:-3000}                                     # 992 / 1008 / 1000 / 1002 chunks
else
  # 2k: the same physics, dumped twice as often.
  VPIC_STEPS=1000 VPIC_INT=8  NYX_FILES=0    WARPX_FILES=2000 LMP_STEPS=1000 LMP_GAP=3  AI_FILES=48
  LMP_BOX_2K=60
  WL_TIMEOUT=${WL_TIMEOUT:-3000}                                     # 2000 / ~2064 / 2000 / 2004 chunks
fi
# 2k LAMMPS uses box 60 (2 chunks per array): no integer gap gives ~2,000 at box 40.
LMP_BOX=${LMP_BOX_2K:-40}
# Nyx 256^3 / 2,000 steps, dumped every 96 (full) or 48 (2k).
NYX_DEFAULT=/projects/bekn/imuradli/np-nyx-256-2000/fields
[ "$SIZE" = 2k ] && NYX_DEFAULT=/projects/bekn/imuradli/np-nyx-256-2000-i48/fields
NYX_FIELDS=${NYX_FIELDS:-$NYX_DEFAULT}
# WarpX 1,000 steps, diagnostics every 5 (2,000 files).
WARPX_FIELDS=${WARPX_FIELDS:-/projects/bekn/imuradli/np-warpx-1000-i5/fields}
# ai/gen_fields.sh: 11 checkpoints of ViT-B/16 on CIFAR-10.
AI_FIELDS=${AI_FIELDS:-/projects/bekn/imuradli/np-ai-vitb16/fields}
declare -A NAME=([vpic]=VPIC [nyx]=Nyx [warpx]=WarpX [lammps]=LAMMPS [ai]=AI)

# In a Slurm job, touch only this job's memfd dir and processes.
[ -n "${SLURM_JOB_ID:-}" ] && export CLIO_MEMFD_DIR=${CLIO_MEMFD_DIR:-/tmp/clio_${USER}_$SLURM_JOB_ID}
MEMFD_DIR=${CLIO_MEMFD_DIR:-/tmp/clio_${USER:-unknown}}
clean_leftovers() {  # a runtime killed by the timeout leaves IPC state behind
  local p
  for p in $(pgrep -u "$USER" -x neuropress_fiel); do
    [ -n "${SLURM_JOB_ID:-}" ] && ! grep -q "job_$SLURM_JOB_ID/" "/proc/$p/cgroup" 2>/dev/null && continue
    kill -KILL "$p" 2>/dev/null
  done
  rm -rf "$MEMFD_DIR" 2>/dev/null
  [ -n "${SLURM_JOB_ID:-}" ] && return 0
  rm -f /tmp/clio_*.ipc /tmp/clio_server_timing.log 2>/dev/null
  rm -rf /tmp/clio_memfd 2>/dev/null
  find /dev/shm -maxdepth 1 -user "$USER" -name 'sm_segment.*' -delete 2>/dev/null
}

run_wl() {
  local wl=$1 dir="$OUT/figure_8_$1"
  local run=$dir
  [ -n "$LOCAL" ] && run="$LOCAL/figure_8_$1"
  local -a cmd
  case "$wl" in
    vpic)   cmd=( "$HERE/vpic/run_config_insitu.sh" "$CONFIG" --ncell 126 --steps "$VPIC_STEPS"
                  --int "$VPIC_INT" "${NP[@]}" --check-bound ) ;;
    nyx)    cmd=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$HERE/nyx/run_config.sh" "$CONFIG"
                  --fields "$NYX_FIELDS" --max-files "$NYX_FILES" "${NP[@]}" --check-bound ) ;;
    warpx)  cmd=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$HERE/nyx/run_config.sh" "$CONFIG"
                  --fields "$WARPX_FIELDS" --max-files "$WARPX_FILES" "${NP[@]}" --check-bound ) ;;
    ai)     cmd=( env CLIO_NEUROPRESS_STAGE_H2D=1 "$HERE/nyx/run_config.sh" "$CONFIG"
                  --fields "$AI_FIELDS" --max-files "$AI_FILES" "${NP[@]}" --check-bound ) ;;
    lammps) cmd=( "$HERE/lammps/run_config.sh" "$CONFIG" --box "$LMP_BOX" --steps "$LMP_STEPS"
                  --gap "$LMP_GAP" --f32 --require-device "${NP[@]}" ) ;;
  esac
  cmd+=( --results "$run" --tag "$wl" )
  if [ "$DRY" = 1 ]; then printf '   DRY %s (in %s):\n        env %s %s\n' "$wl" "$run" "${NP_ENV[*]}" "${cmd[*]}"; return 0; fi

  echo "---- ${NAME[$wl]} ----"
  mkdir -p "$dir" "$run"
  local t0; t0=$(date +%s)
  # From the run directory: in-situ decks write diagnostics to the cwd.
  timeout -k 5 "$WL_TIMEOUT" env -C "$run" "${NP_ENV[@]}" CLIO_NEUROPRESS_PHASE_LOG="$run/phase.csv" \
    "${cmd[@]}" > "$run/console.log" 2>&1
  local rc=$? wall=$(( $(date +%s) - t0 ))
  { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; } && { echo "     TIMEOUT after ${WL_TIMEOUT}s"; clean_leftovers; }
  # The storage images are the run's raw bytes and are never kept, in either mode.
  drop_images() { find "$1" -maxdepth 2 -type f \( -name chi_bdev.dat -o -name 'cte_tier.dat*' \) -delete 2>/dev/null; }
  if [ "$run" != "$dir" ]; then
    # Everything but the storage images; the local copy goes only once that landed.
    if tar -C "$run" --exclude='chi_bdev.dat' --exclude='cte_tier.dat*' -cf - . | tar -C "$dir" -xf -; then
      rm -rf "$run"
    else
      drop_images "$run"
      echo "     COPY FAILED -- the run (without its images) is still in $run"
    fi
  fi
  drop_images "$dir"

  local log="$dir/$wl/explore.csv" verdict nchunks=0
  # An in-situ deck writes its bound check to <wl>.log; the wrapper's stdout.log comes last.
  verdict=$(grep -hE "VERIFIED:|FAILED:|BOUND (OK|FAILED)" "$dir/$wl/stdout.log" "$dir/$wl/$wl.log" 2>/dev/null | tail -2 | tr '\n' ' ')
  [ -s "$log" ] && nchunks=$(awk -F, 'NR>1 && $4=="primary"' "$log" | wc -l)
  echo "     rc=$rc  wall ${wall}s  chunks with a pick: $nchunks  ${verdict:-(no verification line)}"
  [ "$nchunks" -gt 0 ] && echo "     $(awk -v w="$wall" -v n="$nchunks" 'BEGIN{printf "%.2f s per chunk (whole run, incl. simulation and verification)", w/n}')"
  if [ "$nchunks" -eq 0 ] || echo "$verdict" | grep -q FAILED; then
    echo "     NOT PLOTTED -- see $dir/console.log"; return 0
  fi
  python3 "$HERE/plot/fig8_trace.py" "$log" --out "$dir/trace.csv" | sed 's/^/     /'
}

echo "== figure 8 ($SIZE): ${WLS[*]} -> $OUT"
[ "$DRY" != 1 ] && clean_leftovers
for wl in "${WLS[@]}"; do run_wl "$wl"; done
[ "$DRY" = 1 ] && exit 0

TRACES=()
for wl in "${WLS[@]}"; do
  [ -s "$OUT/figure_8_$wl/trace.csv" ] && TRACES+=( --trace "$OUT/figure_8_$wl/trace.csv:$wl" )
done
if [ "$PLOT" = 1 ] && [ ${#TRACES[@]} -gt 0 ]; then
  echo; python3 "$HERE/plot/plot_fig8.py" "${TRACES[@]}" --out "$FIGDIR"
fi
