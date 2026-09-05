#!/usr/bin/env bash
#===============================================================================
# Compare compression strategies on ONE workload, over EXACTLY the same data.
#
#   NeuroPress (lossy low)    per-chunk selection, online learning on
#   Best fixed nvCOMP codec   one nvcomp action, every chunk
#   ndzip only                lossless float, GPU
#   cuSZp3 only               error-bounded lossy, single-kernel
#   cuSZ only                 error-bounded lossy
#
#   ./compare_wallclock.sh --workload vpic --size smoke
#   ./compare_wallclock.sh -w nyx -s full --eb 1e-3
#
# Companion: compare_perchunk_oracle.sh builds the per-chunk oracle (what the
# best possible choice would have been). This one is the stopwatch.
#
#-------------------------------------------------------------------------------
# THE METRIC
#
#   clio_s = h2d + inference + preprocess + codec
#
# ONLY DIRECTLY MEASURED PER-CHUNK WORK. Four components, each timed at its own
# call site and reported per chunk in the run's ledgers:
#
#   h2d         h2d_ms       host-to-device staging inside Compress().
#                            INCOMPLETE: file replay stages a FIRST copy for
#                            selection (compressor_runtime.cc:1351) that is not
#                            recorded; only the Compress-path copy (:3147) is.
#                            Static arms pay that first copy too.
#   KNOWN LIMITATION -- all three CUDA-event brackets are constructed with
#   stream=nullptr, i.e. the legacy NULL stream, while quantize/shuffle actually
#   run on the per-thread DeviceStatsStream. The NULL stream synchronizes with
#   every blocking stream, so these intervals can EXCLUDE work before the start
#   event and INCLUDE other workers' work, and the instrumentation itself
#   serializes workers. That is the likely cause of preprocessing reading 3.761s
#   under events vs 1.606s under a host clock for identical work. The quantize
#   bracket also spans min/max reduction, a sync, host-side scale computation,
#   async alloc/memset and a validation D2H -- it is NOT kernel-only time.
#   Threading the real stream through is the fix; until then treat preproc as
#   an upper bound.
#   inference   select_us    the model's forward pass (a fixed codec has none,
#                            so omitting it silently subsidised the selector)
#   preprocess  preproc_ms   quantize + byte shuffle kernels
#   codec       compress_ms  CUDA-event bracket around the codec launch
#
# Nothing here is a residual. Anything derived by SUBTRACTING phase totals is
# excluded, because such a figure is a container for everything unmeasured --
# on vpic/smoke `stage+compress` minus the measured work left ~14.5 s, which
# I mislabelled "H2D staging" until the arithmetic said otherwise: the copy
# itself is 0.084 ms/chunk (3.91 GiB at PCIe4), while the residual is
# 7.25 ms/chunk. It is runtime plumbing -- per-chunk cudaMalloc/cudaFree, IPC
# and shm allocation, task creation, queueing -- roughly 6x the compression
# work, identical across arms, and not attributable to any strategy.
#
# The residual is still RECORDED (stage_s, store_s, wall_s, read_s, total_s in
# the CSV, and an `unacctd` column in the table) so it stays visible and can be
# investigated. It is simply not part of the number strategies are ranked on. What
# is EXCLUDED, and why -- each was measured to matter on vpic/smoke:
#
#   process setup  2.7-41.5 s.  Runtime spin-up, CUDA context, per-codec module
#                  load, NeuroPress weight load. One-off, amortizes over a real
#                  run, and varied by codec AND arm position -- enough to invert
#                  the ranking of a 4 GiB run by itself.
#   everything     Reported as `unacctd`. It is NOT a clean "runtime overhead"
#   else           partition and must not be read as one:
#                    * total_s is a MAKESPAN over 4 workers while the measured
#                      components are SUMMED per-task latencies, so subtracting
#                      one from the other is not additive;
#                    * it contains real, sometimes arm-specific work -- the
#                      FIRST H2D (see below), producer SHM alloc + memcpy + FNV
#                      hashing, tier writes, the selection log's full-payload
#                      D2H + hashes, write-time decompression and quality
#                      measurement, and online SGD (NeuroPress only);
#                    * so it is NOT identical across arms, and earlier claims
#                      here that it was are withdrawn.
#   tier write     Not separable: the write completes INSIDE the awaited
#                  stage+compress window, so `total - read - stage+compress`
#                  does not isolate it. Byte counts are the honest I/O proxy --
#                  exact, and stored_bytes/ratio are in the CSV.
#   source read    Pulling dumps off disk is the producing application's cost.
#   verification   Already outside the binary's own `total`.
#
# wall_s, read_s, stage_s and total_s all stay in the CSV so the exclusions can
# be audited or a different window reconstructed.
#
#-------------------------------------------------------------------------------
# FAIRNESS CONDITIONS -- each of these was a bug here before it was a rule:
#
#   * Same data. Dumps are generated (or reused) ONCE, before any arm; every arm
#     gets the same --fields, --chunk and --max-files.
#   * Page cache warmed before timing. Otherwise arm 1 pays ~30 s of cold reads
#     for everyone -- and NeuroPress is arm 1.
#   * The fixed baseline may QUANTIZE (-q). Without it the static path reaches
#     only the 16 lossless actions of the 32-action space and competes lossless
#     against a lossy selector: it read 1.235x instead of 6.811x on VPIC.
#   * Fidelity is reported next to every timing. A lossy codec that missed its
#     bound did less work; its time is not comparable.
#   * ARM_ORDER=reverse re-runs the list backwards. If a result tracks POSITION
#     rather than strategy, it is an artifact.
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#--- configuration -------------------------------------------------------------
WL=""; SIZE=smoke; EB=1e-3; CHUNK=""; OUT=""; FIELDS=""; BESTFIXED=""; DRY=0

usage() {
  sed -n '3,10p' "$0" >&2
  cat >&2 <<'USAGE'

  --workload, -w  nyx | vpic | warpx | lammps        (required)
  --size, -s      smoke (2000 chunks x 2 MiB ~ 4 GiB) | full (~30 GB)
  --eb            error bound, default 1e-3 (0 = lossless)
  --chunk         override the size preset's chunk bytes
  --fields        dump directory (default: per-workload, see env.sh)
  --out           results directory
  --best-fixed    e.g. ans-q-s4; default is the workload's oracle winner
  --dry-run       print the commands without running them

  env: ARM_ORDER=reverse, LMP_BOX, LMP_STEPS, LMP_GAP, LMP_DECK
USAGE
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --workload|-w) WL=$2; shift 2 ;;
    --size|-s)     SIZE=$2; shift 2 ;;
    --eb)          EB=$2; shift 2 ;;
    --chunk)       CHUNK=$2; shift 2 ;;
    --fields)      FIELDS=$2; shift 2 ;;
    --out)         OUT=$2; shift 2 ;;
    --best-fixed)  BESTFIXED=$2; shift 2 ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done
[ -n "$WL" ] || usage
case "$WL"   in nyx|vpic|warpx|lammps) ;; *) echo "bad --workload" >&2; usage ;; esac
case "$SIZE" in smoke|full) ;; *) echo "--size must be smoke or full" >&2; usage ;; esac

#--- size presets --------------------------------------------------------------
# smoke reaches 2000 chunks on ~4 GiB by using a SMALLER chunk, not by sampling
# less of the run, so the simulation still evolves across the sample. A 2 MiB
# chunk gives a codec less context than 8 MiB, so smoke RATIOS are not
# comparable with full ratios; smoke answers "does every arm run and hold its
# bound", not "what is the ratio". Pass --chunk to pin one size for both.
TARGET_CHUNKS=2000
case "$SIZE" in
  smoke) CHUNK=${CHUNK:-2097152} ;;
  full)  CHUNK=${CHUNK:-8388608}; TARGET_CHUNKS=0 ;;
esac

# LAMMPS is in situ: no files to cap, so size drives the solver instead. Sized
# by DATA VOLUME to match the replay workloads (~4 GiB smoke, ~30 GiB full).
# THE DECK MUST BE THE RAMP: in.melt melts in ~40 of its 500 steps and is a
# steady-state liquid after, so entropy, MAD and ratio go flat and every frame
# picks the same action (lammps/README.md). in.melt_ramp drives cold crystal to
# hot fluid across the whole run; NSTEPS must equal --steps because the ramp is
# driven by the global step counter.
LMP_BOX=${LMP_BOX:-80}; LMP_GAP=${LMP_GAP:-50}
case "$SIZE" in
  smoke) LMP_STEPS=${LMP_STEPS:-1450} ;;   #  29 frames ~  4.0 GiB
  full)  LMP_STEPS=${LMP_STEPS:-10900} ;;  # 218 frames ~ 29.9 GiB
esac
LMP_FIELD=$(( 4 * LMP_BOX * LMP_BOX * LMP_BOX * 3 * 8 ))
case "$SIZE" in smoke) LMP_CHUNK=$CHUNK ;; full) LMP_CHUNK=$LMP_FIELD ;; esac
LMP_DECK=${LMP_DECK:-$HERE/lammps/in.melt_ramp}

# The best fixed action is a MEASURED result -- the winner of the 32-action
# oracle sweep under a time-inclusive cost. "-q" is not optional on a lossy run:
# the oracle picks a QUANTIZED action on most chunks of every workload measured.
# Re-derive with compare_perchunk_oracle.sh and override if a fresh sweep
# disagrees; a stale default here understates the baseline, which is exactly the
# error that inflates a per-chunk-selection result.
if [ -z "$BESTFIXED" ]; then
  case "$WL" in
    vpic)  BESTFIXED=ans-q-s4     ;;
    *)     BESTFIXED=bitcomp-q-s4 ;;
  esac
fi
OUT=${OUT:-$PWD/strategies/$WL-$SIZE}; mkdir -p "$OUT"

#--- workload data -------------------------------------------------------------
# Generator settings reproduce the published ~30 GB datasets, NOT each
# gen_fields.sh default (those are small quick-run values: nyx 200 steps -> 21
# frames, vpic 200 -> 8, warpx 40 -> 5 dumps).
case "$WL" in
  nyx)   FIELDS=${FIELDS:-${NYX_FIELDS:-$HERE/nyx/fields}}
         GEN=("$HERE/nyx/gen_fields.sh" --ncell 128 --steps 6400 --plot-int 10) ;;
  vpic)  FIELDS=${FIELDS:-${VPIC_FIELDS:-$HERE/vpic/fields}}
         # clean-div 10 is not cosmetic: at upstream's 0, four of the sixteen
         # variables never recompute and dump bit-identical every frame -- 25%
         # of the payload not evolving, which a compression study must not eat.
         GEN=("$HERE/vpic/gen_fields.sh" --ncell 126 --nppc 8
              --steps 6000 --dump-int 25 --clean-div 10) ;;
  warpx) FIELDS=${FIELDS:-${WARPX_FIELDS:-$HERE/warpx/fields}}; GEN=() ;;
  lammps) FIELDS=insitu; GEN=() ;;
esac

prepare_data() {
  [ "$WL" = lammps ] && { echo "== $WL/$SIZE: in situ, box $LMP_BOX ($((4*LMP_BOX**3)) atoms), $LMP_STEPS steps @ gap $LMP_GAP, deck $(basename "$LMP_DECK")"; return 0; }
  if [ ! -d "$FIELDS" ] || [ -z "$(find "$FIELDS" -name '*.f32' -print -quit 2>/dev/null)" ]; then
    if [ ${#GEN[@]} -eq 0 ]; then
      echo "no dumps at $FIELDS and $WL has no standalone generator." >&2
      echo "WarpX is produced in situ through the HDF5 VOL; the 30 GB set was:" >&2
      echo "  ./warpx/run_config.sh dynamic --ncell \"64 64 512\" \\" >&2
      echo "      --steps 3840 --interval 10 --stage-h2d --results <dir>" >&2
      return 1
    fi
    echo "== generating $WL dumps into $FIELDS (once; every arm replays these)"
    [ "$DRY" = 1 ] || OUT="$FIELDS" "${GEN[@]}" --out "$FIELDS" || return 1
  fi
  NFILES=$(find "$FIELDS" -name '*.f32' | wc -l)
  FSIZE=$(stat -c %s "$(find "$FIELDS" -name '*.f32' -print -quit)")
  PERFILE=$(( FSIZE / CHUNK )); [ "$PERFILE" -lt 1 ] && PERFILE=1
  if [ "$TARGET_CHUNKS" -gt 0 ]; then
    MAXF=$(( (TARGET_CHUNKS + PERFILE - 1) / PERFILE ))
    [ "$MAXF" -gt "$NFILES" ] && MAXF=$NFILES
  else MAXF=$NFILES; fi
  echo "== $WL/$SIZE: $MAXF of $NFILES files x $PERFILE chunks = $((MAXF*PERFILE)) chunks"
  echo "   of $((CHUNK/1048576)) MiB = $((MAXF*FSIZE/1073741824)) GiB, eb $EB"
}

# The replay route reads .f32 into HOST shm, and the quantizer and byte shuffle
# are CUDA-only -- the host implementations were removed. Without staging, every
# chunk whose action wants a shuffle is REFUSED (rc=1, nothing stored, read back
# rc=11). Measured before this line existed: 2000 blobs, 0 B stored. Inert for
# the in-situ workloads, which already hand over device memory.
case "$WL" in nyx|vpic) export CLIO_NEUROPRESS_STAGE_H2D=1 ;; esac

warm_cache() {
  [ "$WL" = lammps ] && return 0
  [ "$DRY" = 1 ] && return 0
  echo "== warming page cache over $MAXF files (untimed)"
  find "$FIELDS" -name '*.f32' | sort | head -n "$MAXF" | xargs -r cat > /dev/null 2>&1
}

#--- arms ----------------------------------------------------------------------
ARMS=(
  "NeuroPress (lossy low)|learn"
  "Best fixed nvCOMP codec|static-$BESTFIXED"
  "ndzip only|static-ndzip"
  "cuSZp3 only|static-cuszp"
  "cuSZ only|static-cusz"
)
if [ "${ARM_ORDER:-forward}" = reverse ]; then
  _rev=(); for ((_i=${#ARMS[@]}-1; _i>=0; _i--)); do _rev+=("${ARMS[$_i]}"); done
  ARMS=("${_rev[@]}"); echo "== ARM_ORDER=reverse"
fi

arm_cmd() {  # arm_cmd <config> <store>  -> echoes the argv, one arg per line
  local cfg=$1 store=$2
  printf '%s\n' "$HERE/$WL/run_config.sh" "$cfg" --eb "$EB" \
                --results "$store" --tag "$cfg"
  if [ "$WL" = lammps ]; then
    # chunk is ONE frame of ONE field at --size full, the unit run_paper.sh
    # pins, so a chunk is never a partial frame.
    printf '%s\n' --deck "$LMP_DECK" --box "$LMP_BOX" --steps "$LMP_STEPS" \
                  --gap "$LMP_GAP" --chunk "$LMP_CHUNK" \
                  --var "NSTEPS=$LMP_STEPS" --require-device
  else
    printf '%s\n' --chunk "$CHUNK" --fields "$FIELDS" --max-files "$MAXF"
  fi
  # ndzip is lossless -> bit-exact digest; the rest are checked against eb.
  [ "$cfg" = static-ndzip ] || printf '%s\n' --check-bound
}

# Phase timings: the binary reports read / stage+compress / total, and the
# per-chunk ledger reports codec and preprocessing separately. preproc_ms is
# kept out of compress_ms upstream of here because compress_ms feeds the
# exploration winner, the SGD target and the error_pct gate and must keep
# matching the kernel time NeuroPress trained on.
measure_arm() {  # measure_arm <store> <cfg> -> sets CLIO_S CODEC_S STAGE_S STORE_S READ_S TOTAL_S BOUND
  local store=$1 cfg=$2 c
  c=$(sed 's/\x1b\[[0-9;]*m//g' "$store.console" 2>/dev/null)
  READ_S=$(echo "$c"  | grep -oE "read [0-9.]+ s"           | head -1 | grep -oE "[0-9.]+")
  local sc; sc=$(echo "$c" | grep -oE "stage\+compress [0-9.]+ s" | head -1 | grep -oE "[0-9.]+")
  TOTAL_S=$(echo "$c" | grep -oE "total [0-9.]+ s"          | head -1 | grep -oE "[0-9.]+")
  read -r CODEC_S INFER_S < <(python3 - "$store" "$cfg" <<'PY'
import csv, os, sys
d, tag = sys.argv[1], sys.argv[2]
def find(name):
    for p in (os.path.join(d, tag, name), os.path.join(d, name)):
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None
bp = find("blobs.csv")
if not bp:
    print("NA NA"); raise SystemExit
r = [x for x in csv.DictReader(open(bp)) if x.get("rc", "0") == "0"]
# preproc_ms absent in ledgers written before it existed -> counted as 0
# A ledger without preproc_ms/h2d_ms predates those columns. Counting them as
# zero silently UNDER-reports, and LAMMPS/WarpX ledgers are in that state, so
# say so rather than quietly producing a smaller number.
missing = [c for c in ("preproc_ms", "h2d_ms") if c not in (r[0] or {})]
if missing:
    print(f"WARN-MISSING:{','.join(missing)}", file=sys.stderr)
codec = sum(float(x["compress_ms"]) + float(x.get("preproc_ms") or 0)
            + float(x.get("h2d_ms") or 0) for x in r) / 1000
# Per-chunk model inference. Only the selector has any; a static arm reports 0
# (or a negative sentinel when the model never ran), clamped to 0 here.
# Sum selection latency over the SAME chunks the codec/preproc sums used --
# blobs rows are filtered to rc==0, so summing every primary selection row
# would mix populations whenever a chunk fails.
ok_blobs = {x["blob"] for x in r}
sp, infer = find("selection.csv"), 0.0
if sp:
    infer = sum(max(0.0, float(x.get("select_us") or 0))
                for x in csv.DictReader(open(sp))
                if x.get("role") == "primary" and x.get("blob") in ok_blobs) / 1e6
print(f"{codec:.3f} {infer:.3f}" if r else "NA NA")
PY
)
  STAGE_S=$(awk -v s="${sc:-0}" -v c="${CODEC_S:-0}" 'BEGIN{printf "%.3f", s-c}')
  STORE_S=$(awk -v t="${TOTAL_S:-0}" -v r="${READ_S:-0}" -v s="${sc:-0}" 'BEGIN{printf "%.3f", t-r-s}')
  # Measured components only. STORE_S and STAGE_S are residuals -- recorded
  # below as diagnostics, never summed into the ranked number.
  CLIO_S=$(awk -v c="${CODEC_S:-NA}" -v i="${INFER_S:-0}" -v t="${TOTAL_S:-0}" \
    'BEGIN{ if (t>0 && c!="NA") printf "%.3f", i+c; else printf "NA" }')
  UNACCT_S=$(awk -v t="${TOTAL_S:-0}" -v r="${READ_S:-0}" -v c="${CLIO_S:-0}" \
    'BEGIN{ if (t>0) printf "%.3f", t-r-c; else printf "NA" }')
  # The binary prints per-chunk BOUND EXCEEDED lines and a BOUND FAILED summary
  # on violation, and stays SILENT when every chunk passes -- verified within a
  # single run, where cuSZp emitted violations and the nvcomp arm, given the
  # same --check-bound, emitted none. So silence on a lossy run is a pass.
  BOUND=$(echo "$c" | grep -oE "BOUND (OK|FAILED)[^,]*" | head -1)
  if [ -z "$BOUND" ]; then
    if echo "$c" | grep -q "lossy eb="; then
      if   echo "$c" | grep -q "BOUND EXCEEDED"; then BOUND="bound VIOLATED"
      elif echo "$c" | grep -q "VERIFIED";       then BOUND="bound OK (0 violations)"
      else BOUND="lossy, not verified"; fi
    elif echo "$c" | grep -q "VERIFIED"; then BOUND="bit-exact"
    else BOUND="unchecked"; fi
  fi
}

run_arm() {  # run_arm <label> <config>
  local label=$1 cfg=$2 store="$OUT/$2" rc w s
  echo; echo "---- $label   [$cfg] ----"
  mapfile -t cmd < <(arm_cmd "$cfg" "$store")
  if [ "$DRY" = 1 ]; then printf '   DRY: %q ' "${cmd[@]}"; echo; return 0; fi
  s=$(date +%s.%N)
  "${cmd[@]}" > "$store.console" 2>&1; rc=$?
  w=$(awk -v a="$s" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
  measure_arm "$store" "$cfg"
  local sb rt
  read -r sb rt < <(python3 - "$store" "$cfg" <<'PY'
import csv, os, sys
d, tag = sys.argv[1], sys.argv[2]
for p in (os.path.join(d, tag, "blobs.csv"), os.path.join(d, "blobs.csv")):
    if os.path.exists(p): break
else:
    print("NA NA"); raise SystemExit
r = [x for x in csv.DictReader(open(p)) if x.get("rc", "0") == "0"]
ti = sum(float(x["bytes"]) for x in r); to = sum(float(x["stored"]) for x in r)
print(f"{to:.0f} {ti/to:.3f}" if to else "NA NA")
PY
)
  printf '   clio %ss = infer %ss + pre+codec %ss   [unaccounted %ss]  ratio %s  [%s]  rc=%s\n' \
         "${CLIO_S:-?}" "${INFER_S:-?}" "${CODEC_S:-?}" "${UNACCT_S:-?}" "$rt" "$BOUND" "$rc"
  echo "\"$label\",$cfg,${CLIO_S:-NA},${INFER_S:-NA},${CODEC_S:-NA},${UNACCT_S:-NA},${STAGE_S:-NA},${STORE_S:-NA},$w,${READ_S:-NA},${TOTAL_S:-NA},$sb,$rt,\"$BOUND\",$rc" >> "$RESULTS"
  rm -f "$store/chi_bdev.dat" "$store"/cte_tier.dat*
}

#--- main ----------------------------------------------------------------------
prepare_data || exit 1
warm_cache
RESULTS="$OUT/wallclock.csv"
echo "strategy,config,clio_s,infer_s,codec_s,unacctd_s,stage_s,store_s,wall_s,read_s,total_s,stored_bytes,ratio,bound,rc" > "$RESULTS"
for entry in "${ARMS[@]}"; do run_arm "${entry%%|*}" "${entry#*|}"; done
[ "$DRY" = 1 ] && exit 0
echo; echo "=== $WL / $SIZE ==="
python3 "$HERE/wallclock_table.py" "$RESULTS"
echo "csv: $RESULTS"
