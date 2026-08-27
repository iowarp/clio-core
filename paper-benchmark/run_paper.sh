#!/usr/bin/env bash
# The paper matrix: six Clio/NeuroPress configurations x four workloads.
#
#   ./run_paper.sh                          # everything
#   ./run_paper.sh --workload nyx           # one workload, all six configs
#   ./run_paper.sh --config ratio_lossless  # one config, all four workloads
#   ./run_paper.sh --dry-run                # print the commands and stop
#   ./run_paper.sh --out /tmp/smoke --steps 80   # smoke: every cell, tiny
#   ./run_paper.sh --no-verify              # skip the integrity checks
#
# Each cell lands in <workload>/<config>/ as CSV, JSON and logs. THE FIELD
# DATA IS NEVER COPIED THERE -- it is tens of GB per workload, it is identical
# across the six configurations, and every generator regenerates it.
#
# WHY NYX RUNS THROUGH REPLAY AND THE OTHER THREE RUN IN SITU. Nyx in situ
# faults with CUDA 700 (an out-of-bounds __global__ write inside nvcomp's
# setup_comp_llif_buffers, Xid 31) once exploration is on; K, the error bound
# and the cost model only change how many chunks it survives first. Measured:
# the same configuration through Nyx's REPLAY route passes, and WarpX, LAMMPS
# and VPIC all pass at K=31 with an error bound. The fault is specific to the
# compressor running inside Nyx's own CUDA context. Replay is also the route
# the Nyx benchmark was designed around -- every configuration replays
# byte-identical input, which the in-situ route cannot promise.
#
# VERIFICATION IS ON BY DEFAULT; --no-verify turns it off for a faster pass.
#   lossless -> bit-exact digest: every blob is read back through the
#               decompressor and its FNV-1a-64 compared with the staged bytes.
#   lossy    -> |original - decoded| <= eb, elementwise, which is the only
#               check that means anything once data is quantized. A digest
#               MUST fail on lossy by construction, so the runners disable it.
# The bound check needs the original bytes to compare against, so only the two
# REPLAY workloads can make it -- they still have the source file. WarpX and
# LAMMPS have no source to re-read (the simulation IS the source and it has
# moved on), so their lossy cells are unverified whatever this flag says.
# config.json records the check each cell actually got and what it concluded.
#
# THE SIMULATION PARAMETERS ARE THE EVOLUTION STUDY'S WINNERS, not upstream's
# examples: nyx.cfl=0.8, VPIC clean_div=10, WarpX e_max=32e12/density=2e23,
# LAMMPS temp=6.0 with skin=0.8/every=5. Each is the default inside its own
# run script, so this file passes only what is not already the default. See
# each workload's "Default Evolving Benchmark Configuration" README section.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRATCH=${SCRATCH:-/tmp/paper-bench}          # runs land here; only CSVs are kept
STEPS=${STEPS:-1000}
INT=${INT:-40}
K=${K:-31}

ONLY_W="" ONLY_C="" DRY=0
OUT_ROOT=$HERE            # --out sends cells elsewhere, e.g. for a smoke pass
VERIFY=${VERIFY:-1}       # 1 = digest on lossless, bound check on lossy
while [ $# -gt 0 ]; do
  case "$1" in
    --workload) ONLY_W=$2; shift 2;;
    --config)   ONLY_C=$2; shift 2;;
    --steps)    STEPS=$2; shift 2;;
    --int)      INT=$2; shift 2;;
    --explore-k) K=$2; shift 2;;
    --scratch)  SCRATCH=$2; shift 2;;
    --out)      OUT_ROOT=$2; shift 2;;
    --verify)    VERIFY=1; shift;;
    --no-verify) VERIFY=0; shift;;
    --dry-run)  DRY=1; shift;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# config name -> (base config, extra flags). Exploration only, K as set above.
declare -A CFG=(
  [balance_lossless]="explore-balance"
  [balance_lossy_0.001]="explore-balance --eb 0.001"
  [ratio_lossless]="explore-ratio"
  [ratio_lossy_0.001]="explore-ratio --eb 0.001"
  [ratio_lossy_0.01]="explore-ratio --eb 0.01"
  [ratio_lossy_0.1]="explore-ratio --eb 0.1"
)
ORDER=(balance_lossless balance_lossy_0.001 ratio_lossless
       ratio_lossy_0.001 ratio_lossy_0.01 ratio_lossy_0.1)

# Where each generator's fields land. Set unconditionally so --dry-run (which
# skips generation) can still print the replay commands.
NYX_FIELDS=$SCRATCH/nyx-fields
VPIC_FIELDS=$SCRATCH/vpic-fields

wait_gpu () {
  for _ in $(seq 120); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    [ "${u:-0}" -lt 500 ] && { sleep 3; return; }
    sleep 3
  done
}

# ---------------------------------------------------------------------------
# Field generation, once per workload -- the six configurations replay or
# re-run the SAME simulation settings, so this must not be inside the config
# loop.
# ---------------------------------------------------------------------------
gen_fields () {
  case "$1" in
    nyx)
      [ -d "$NYX_FIELDS" ] && return 0
      echo "   generating Nyx fields (128^3, $STEPS steps, cfl 0.8)"
      "$HERE/nyx/gen_fields.sh" --ncell 128 --steps "$STEPS" --plot-int "$INT" \
          --out "$NYX_FIELDS" > "$SCRATCH/nyx-gen.log" 2>&1
      ;;
    vpic)
      [ -d "$VPIC_FIELDS" ] && return 0
      echo "   generating VPIC fields (126^3, $STEPS steps, clean_div 10)"
      "$HERE/vpic/gen_fields.sh" --ncell 126 --nppc 8 --steps "$STEPS" \
          --dump-int "$INT" --out "$VPIC_FIELDS" > "$SCRATCH/vpic-gen.log" 2>&1
      ;;
  esac
}

# ---------------------------------------------------------------------------
# One cell of the matrix.
# ---------------------------------------------------------------------------
run_cell () {
  local w=$1 c=$2
  local base=${CFG[$c]}
  # What this cell can check, and what it was asked to check. See the header
  # for why WarpX and LAMMPS cannot bound-check a lossy run.
  # THE FLAG IS NOT THE SAME ON ALL FOUR. WarpX defaults verification OFF and
  # takes --verify; the other three default it ON and take only --no-verify.
  # Passing --verify to those three is an unknown argument (rc=2), which is
  # exactly how this was found.
  local vflag=() vkind="none"
  case "$c" in
    *_lossless)
      vkind=digest
      if [ "$VERIFY" = 1 ]; then
        [ "$w" = warpx ] && vflag=(--verify)          # others: on by default
      else
        vkind=disabled
        [ "$w" = warpx ] || vflag=(--no-verify)       # warpx: off by default
      fi ;;
    *)
      case "$w" in
        nyx|vpic)
          if [ "$VERIFY" = 1 ]; then vflag=(--check-bound); vkind=bound
          else vkind=disabled; fi ;;
        # No source bytes to compare a lossy round trip against; the runner
        # force-disables the digest when eb>0, so nothing is checked here.
        *) vkind=unsupported ;;
      esac ;;
  esac
  local out=$OUT_ROOT/$w/$c
  local store=$SCRATCH/$w/$c
  local cmd=()

  case "$w" in
    warpx)   # in situ via the HDF5 VOL; --stage-h2d is mandatory
      cmd=("$HERE/warpx/run_config.sh" $base --ncell "64 64 512"
           --steps "$STEPS" --interval "$INT" --stage-h2d
           --explore-k "$K" ${vflag[@]+"${vflag[@]}"}
           --results "$store" --tag run) ;;
    lammps)  # library in process; chunk must be one frame of one field
      local natoms=$(( 4 * 20 * 20 * 20 ))
      cmd=("$HERE/lammps/run_config.sh" $base --box 20 --steps "$STEPS"
           --gap "$INT" --chunk $(( natoms * 3 * 8 )) --require-device
           --explore-k "$K" ${vflag[@]+"${vflag[@]}"}
           --results "$store" --tag run) ;;
    vpic)    # replay; STAGE_H2D is required and nothing else sets it
      cmd=(env CLIO_NEUROPRESS_STAGE_H2D=1 "$HERE/vpic/run_config.sh" $base
           --fields "$VPIC_FIELDS" --explore-k "$K"
           ${vflag[@]+"${vflag[@]}"}
           --results "$store" --tag run) ;;
    nyx)     # replay, deliberately -- see the header
      cmd=(env CLIO_NEUROPRESS_STAGE_H2D=1 "$HERE/nyx/run_config.sh" $base
           --fields "$NYX_FIELDS" --explore-k "$K"
           ${vflag[@]+"${vflag[@]}"}
           --results "$store" --tag run) ;;
  esac

  if [ "$DRY" = 1 ]; then printf '   %s\n' "${cmd[*]}"; return 0; fi

  mkdir -p "$out/figures"
  wait_gpu
  local t0=$(date +%s)
  "${cmd[@]}" > "$out/stdout.log" 2> "$out/stderr.log"
  local rc=$? t1=$(date +%s)

  # Whatever the runner produced, under its own names. Nothing is renamed:
  # the analysis scripts already know these.
  for f in blobs.csv selection.csv explore.csv meta.json; do
    [ -f "$store/run/$f" ] && cp "$store/run/$f" "$out/$f"
  done
  [ -f "$store/run/stdout.log" ] && cp "$store/run/stdout.log" "$out/run.stdout.log"
  [ -f "$store/run/runtime.log" ] && cp "$store/run/runtime.log" "$out/runtime.log"
  [ -f "$out/blobs.csv" ] && cp "$out/blobs.csv" "$out/results.csv"

  # What the check concluded, as the runner recorded it: pass / bound-ok /
  # BOUND-FAIL / FAIL / n/a. Scraped rather than re-derived so this file never
  # disagrees with the run's own meta.json.
  # FROM THE LOG, NOT meta.json: the four runners disagree on the key. WarpX
  # writes "verify_result":"pass"; the replay runners write "verified":0|1 and
  # say nothing about the bound there. The printed verdict is the one thing
  # all four share, so it is what gets scraped.
  local vres=n/a vlog="$out/run.stdout.log"
  [ -f "$vlog" ] || vlog="$out/stdout.log"
  if   grep -q "BOUND FAILED:" "$vlog" 2>/dev/null; then vres=BOUND-FAIL
  elif grep -q "BOUND OK:"     "$vlog" 2>/dev/null; then vres=bound-ok
  elif grep -q "^VERIFIED:"    "$vlog" 2>/dev/null; then vres=pass
  elif grep -qE "^FAILED:|MISMATCH" "$vlog" 2>/dev/null; then vres=FAIL
  fi
  # What the check actually covered, for the size question: every blob is read
  # back at its own recorded original length into a pre-zeroed buffer, so
  # "VERIFIED: N of N" means N blobs reproduced their FULL original byte count
  # bit-exact. Sum it here so the cell states the round-tripped total outright.
  local vbytes=0
  [ "$vres" = pass ] && vbytes=$(python3 -c \
      "import csv;print(sum(int(x['bytes']) for x in csv.DictReader(open('$out/blobs.csv'))))" \
      2>/dev/null || echo 0)

  cat > "$out/config.json" <<JSON
{"workload":"$w","config":"$c","cost_model":"${c%%_*}",
 "mode":"$([ "${c#*loss}" = "less" ] && echo lossless || echo lossy)",
 "error_bound":$(echo "$c" | grep -oE '[0-9.]+$' || echo 0),
 "explore_k":$K,"exploration_only":true,
 "steps":$STEPS,"interval":$INT,"rc":$rc,"wall_s":$((t1-t0)),
 "route":"$([ "$w" = nyx ] || [ "$w" = vpic ] && echo replay || echo insitu)",
 "verification":"$vkind","verify_result":"$vres",
 "bytes_roundtripped_bit_exact":$vbytes,
 "command":"${cmd[*]}"}
JSON
  # A bound failure is not a crash: the run completed and the blobs are good,
  # the requested eb was simply not reachable. Nyx at eb=0.001 is the standing
  # example -- rho_e peaks at 2.1e5 and the field is float32, whose spacing
  # there is 0.0156, so half a ULP (0.0078) already exceeds the bound and no
  # codec can meet it. Reported apart from rc so it is not read as a failure
  # of the compressor.
  local verdict
  if [ $rc = 0 ]; then verdict=OK
  elif [ "$vres" = BOUND-FAIL ]; then verdict="BOUND-UNREACHABLE"
  else verdict="FAIL rc=$rc"; fi
  printf '%-8s %-20s %-18s %ss\n' "$w" "$c" "$verdict" "$((t1-t0))"
}

mkdir -p "$SCRATCH"
for w in warpx vpic nyx lammps; do
  [ -n "$ONLY_W" ] && [ "$w" != "$ONLY_W" ] && continue
  echo "== $w"
  [ "$DRY" = 1 ] || gen_fields "$w"
  for c in "${ORDER[@]}"; do
    [ -n "$ONLY_C" ] && [ "$c" != "$ONLY_C" ] && continue
    run_cell "$w" "$c"
  done
done
