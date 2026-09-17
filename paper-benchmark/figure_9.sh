#!/usr/bin/env bash
#===============================================================================
# figure_9.sh -- end-to-end wall-clock time per workload, for Figure 9.
#
#   ./figure_9.sh -w nyx -s smoke            # -w nyx|vpic|warpx|lammps|ai
#   ./figure_9.sh -w nyx -s smoke --panel b  # external baselines only
#   ./figure_9.sh -w nyx -s smoke --dry-run
#
# Writes <out>/fig9.csv (panel,workload,strategy,compute_min,io_min,total_min,
# std_min) and plots it; an arm that did not complete is written as TBD.
#
# Arms: Baseline (no compression), a fixed nvCOMP codec, NeuroPress, each with
# a RAM tier over a file tier (+Tier, BENCH_TIER*), a periodic flush (+Async,
# BENCH_FLUSH_MS) and a lossy bound. Every arm ends with a timed FlushData
# (CLIO_REPLAY_FINAL_FLUSH); nothing fsyncs.
#
# compute = simulate time (LAMMPS) + the write loop's time over Baseline's,
# i.e. what compression adds; Baseline's loop (copy, digest, raw put) is I/O.
# io = total - compute. Nyx, VPIC, WarpX and AI replay .f32 dumps (WarpX and
# AI through nyx/run_config.sh); LAMMPS runs in situ. No bound check.
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

WL=nyx SIZE=smoke EB_LOW=1e-3 EB_MED=1e-2 EB_HIGH=1e-1
PANEL=both DRY=0 OUT="" FIELDS=""
BEST_FIXED=${BEST_FIXED:-}     # default per workload, below
ASYNC_MS=${ASYNC_MS:-500}      # periodic flush for the +Async arms
RAM_MB=${RAM_MB:-512}          # RAM tier 1; eagerly committed at runtime start

usage() { sed -n '3,9p' "$0" >&2; cat >&2 <<'U'

  --workload, -w  nyx | vpic | warpx | lammps | ai   (default nyx)
  --size, -s      smoke | full                       (default smoke)
  --panel         a | b | both                       (default both)
  --out DIR       results directory
  --fields DIR    dump directory (default: per workload, as compare_wallclock.sh)
  --dry-run       print the arms and stop
U
exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --workload|-w) WL=$2; shift 2 ;;
    --size|-s)     SIZE=$2; shift 2 ;;
    --panel)       PANEL=$2; shift 2 ;;
    --out)         OUT=$2; shift 2 ;;
    --fields)      FIELDS=$2; shift 2 ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done
case "$WL"   in nyx|vpic|warpx|lammps|ai) ;; *) echo "bad --workload" >&2; usage ;; esac
case "$SIZE" in smoke|full) ;; *) echo "bad --size" >&2; usage ;; esac
case "$WL" in nyx) WLNAME=Nyx ;; vpic) WLNAME=VPIC ;; warpx) WLNAME=WarpX ;; lammps) WLNAME=LAMMPS ;; ai) WLNAME=AI ;; esac
# The oracle sweep's best fixed action, as compare_wallclock.sh.
if [ -z "$BEST_FIXED" ]; then
  case "$WL" in vpic) BEST_FIXED=static-ans-q-s4 ;; *) BEST_FIXED=static-bitcomp-q-s4 ;; esac
fi
OUT=${OUT:-$HERE/results/figure9/$WL-$SIZE}
mkdir -p "$OUT"
CSV="$OUT/fig9.csv"

if [ -z "$FIELDS" ]; then
  case "$WL" in
    nyx)    FIELDS=${NYX_FIELDS:-$HERE/nyx/fields} ;;
    vpic)   FIELDS=${VPIC_FIELDS:-$HERE/vpic/fields} ;;
    warpx)  FIELDS=${WARPX_FIELDS:-$HERE/warpx/fields} ;;
    ai)     FIELDS=${AI_FIELDS:-$HERE/ai/fields} ;;
    lammps) FIELDS=insitu ;;   # LAMMPS runs in situ: no dump directory
  esac
fi
if [ "$FIELDS" != insitu ] && [ -z "$(find "$FIELDS" -name '*.f32' -print -quit 2>/dev/null)" ]; then
  echo "no field dumps at $FIELDS -- pass --fields DIR or run $WL/gen_fields.sh" >&2
  exit 3
fi

# Replay hands the compressor host memory, which it refuses unless staged.
case "$WL" in nyx|vpic|warpx|ai) export CLIO_NEUROPRESS_STAGE_H2D=1 ;; esac

case "$WL" in warpx|ai) DRIVER=$HERE/nyx/run_config.sh ;; *) DRIVER=$HERE/$WL/run_config.sh ;; esac

# ARM_TIMEOUT (s): a smoke arm is ~4 s; a full LAMMPS arm simulates ~8 min.
case "$SIZE" in smoke) CHUNK=2097152; MAXF=${MAXF:-60}; ARM_TIMEOUT=${ARM_TIMEOUT:-40} ;;
                full)  CHUNK=8388608; MAXF=${MAXF:-0};  ARM_TIMEOUT=${ARM_TIMEOUT:-1800} ;; esac
# An AI file is a whole 327 MiB tensor.
[ "$WL" = ai ] && [ "$SIZE" = smoke ] && MAXF=2
# LAMMPS box 80 = 70.3 MiB per frame: smoke 7 frames (~0.5 GiB), full 437 (~30).
LMP_BOX=${LMP_BOX:-80} LMP_GAP=${LMP_GAP:-50}
case "$SIZE" in smoke) LMP_STEPS=${LMP_STEPS:-300} ;; full) LMP_STEPS=${LMP_STEPS:-21800} ;; esac
LMP_DECK=${LMP_DECK:-$HERE/lammps/in.melt_ramp}
[ "$WL" = lammps ] && MAXF=0

# arm  ::  label | panel | config | eb | tier(0/1) | async_ms
#          async_ms 0 = periodic flush disabled entirely
ARMS=(
  "Baseline|a|baseline|0|0|0"
  "nvCOMP|a|$BEST_FIXED|0|0|0"
  "nvCOMP+Tier|a|$BEST_FIXED|0|1|0"
  "NP only|a|dynamic|0|0|0"
  "NP+Tier|a|dynamic|0|1|0"
  "NP+Tier+Async|a|dynamic|0|1|$ASYNC_MS"
  "NP+Tier+Async+Lossy (low)|a|dynamic|$EB_LOW|1|$ASYNC_MS"
  "NP+Tier+Async+Lossy (med)|a|dynamic|$EB_MED|1|$ASYNC_MS"
  "NP+Tier+Async+Lossy (high)|a|dynamic|$EB_HIGH|1|$ASYNC_MS"
  "Best fixed nvCOMP|b|$BEST_FIXED|$EB_LOW|1|$ASYNC_MS"
  "ndzip|b|static-ndzip|$EB_LOW|1|$ASYNC_MS"
  "cuSZp3|b|static-cuszp|$EB_LOW|1|$ASYNC_MS"
  "cuSZ|b|static-cusz|$EB_LOW|1|$ASYNC_MS"
  "NeuroPress|b|dynamic|$EB_LOW|1|$ASYNC_MS"
)

# A runtime killed mid-run leaves IPC state that slows later arms; clear this
# user's leftovers (the ctest cleanup fixture's list). In a Slurm job, only this
# job's: its own memfd dir and processes, so jobs can share a node.
[ -n "${SLURM_JOB_ID:-}" ] && export CLIO_MEMFD_DIR=${CLIO_MEMFD_DIR:-/tmp/clio_${USER}_$SLURM_JOB_ID}
MEMFD_DIR=${CLIO_MEMFD_DIR:-/tmp/clio_${USER:-unknown}}
node_state() {
  printf 'memavail=%sMB clio_tmp=%sK shm_mine=%sK strays=%s gpu_used=%s' \
    "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" \
    "$(du -sk "$MEMFD_DIR" 2>/dev/null | cut -f1)" \
    "$(find /dev/shm -maxdepth 1 -user "$USER" -printf '%k\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')" \
    "$(pgrep -u "$USER" -x 'neuropress_fiel|neuropress_lamm' | wc -l)" \
    "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"
}
clean_leftovers() {
  local p
  for p in $(pgrep -u "$USER" -x 'neuropress_fiel|neuropress_lamm'); do
    [ -n "${SLURM_JOB_ID:-}" ] && ! grep -q "job_$SLURM_JOB_ID/" "/proc/$p/cgroup" 2>/dev/null && continue
    kill -KILL "$p" 2>/dev/null
  done
  rm -rf "$MEMFD_DIR" 2>/dev/null
  [ -n "${SLURM_JOB_ID:-}" ] && return 0
  rm -f /tmp/clio_*.ipc /tmp/clio_server_timing.log 2>/dev/null
  rm -rf /tmp/clio_memfd 2>/dev/null
  find /dev/shm -maxdepth 1 -user "$USER" -name 'sm_segment.*' -delete 2>/dev/null
}

slug() { echo "$1" | tr 'A-Z ()+' 'a-z___' | tr -s '_' | sed 's/_$//'; }

run_arm() {  # run_arm <label> <config> <eb> <tier> <async_ms> -> echoes total minutes, or empty
  local label=$1 cfg=$2 eb=$3 tier=$4 flush=$5
  local tag; tag=$(slug "$label")
  local store="$OUT/$tag"
  mkdir -p "$store"

  # BENCH_FLUSH_MS is ALWAYS set: 0 is the only way to turn the periodic flush off.
  local -a env_kv=( "BENCH_FLUSH_MS=$flush" "CLIO_REPLAY_FINAL_FLUSH=1" )
  [ "$tier" = 1 ] && env_kv+=( "BENCH_TIER1_TYPE=ram" "BENCH_TIER1_MB=$RAM_MB"
                               "BENCH_TIER1_PERSIST=volatile"
                               "BENCH_TIER2_PATH=$store/cte_tier2.dat" )

  local -a cmd=( "$DRIVER" "$cfg" --eb "$eb"
                 --results "$store" --tag "$tag" --chunk "$CHUNK" )
  if [ "$FIELDS" = insitu ]; then
    cmd+=( --deck "$LMP_DECK" --box "$LMP_BOX" --steps "$LMP_STEPS" --gap "$LMP_GAP"
           --var "NSTEPS=$LMP_STEPS" --f32 --require-device )
  else
    cmd+=( --fields "$FIELDS" )
    [ "$MAXF" -gt 0 ] 2>/dev/null && cmd+=( --max-files "$MAXF" )
  fi

  if [ "$DRY" = 1 ]; then
    printf '   DRY %s:\n        env %s \\\n        %s\n' "$label" "${env_kv[*]}" "${cmd[*]}" >&2
    return 0
  fi

  echo "---- $label ----" >&2
  echo "     node before: $(node_state)" >&2
  # Per-arm timeout; `timeout` signals the whole process group.
  timeout -k 5 "$ARM_TIMEOUT" env "${env_kv[@]}" "${cmd[@]}" > "$store/console.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "     TIMEOUT after ${ARM_TIMEOUT}s (rc=$rc) -- arm recorded as TBD" >&2
    echo "     left behind: $(node_state)" >&2
    clean_leftovers
    echo "     after clean: $(node_state)" >&2
    return 0
  fi
  # Replay "time: read A s   stage+compress B s   total C s"
  # LAMMPS "time: simulate A s   stage+compress(wait) B s   total C s"
  local tline sim_s stage_s base_s sc_s="" total_s
  tline=$(grep -hE "time: (read|simulate)" "$store"/*/stdout.log 2>/dev/null | tail -1)
  sim_s=$(echo "$tline" | grep -oE "simulate[[:space:]]+[0-9.]+" | grep -oE "[0-9.]+$")
  stage_s=$(echo "$tline" | grep -oE "stage\+compress(\(wait\))?[[:space:]]+[0-9.]+" | grep -oE "[0-9.]+$")
  [ "$cfg" = baseline ] && [ -n "$stage_s" ] && echo "$stage_s" > "$OUT/baseline_stage_s"
  base_s=$(cat "$OUT/baseline_stage_s" 2>/dev/null)
  [ -n "$stage_s" ] && [ -n "$base_s" ] && sc_s=$(awk -v s="${sim_s:-0}" -v a="$stage_s" \
    -v b="$base_s" 'BEGIN{d = a - b; if (d < 0) d = 0; printf "%.3f", s + d}')
  total_s=$(echo "$tline" | grep -oE "total[[:space:]]+[0-9.]+"            | grep -oE "[0-9.]+$")
  local bound
  bound=$(grep -hoE "BOUND (OK|FAILED)[^,]*" "$store"/*/stdout.log 2>/dev/null | tail -1)
  local fl
  fl=$(grep -hoE "flush: [0-9]+ blob\(s\), [0-9]+ B moved to durable storage in [0-9.]+ s" "$store"/*/stdout.log 2>/dev/null | tail -1)
  # Tier evidence before the images are deleted: tier-2 bytes after a sync
  # (the file is <path>_node<N>) and the pools each blob was placed on.
  local t2 t2rep="" stored_b in_b walrep
  stored_b=$(grep -hoE "^stored [0-9]+ blob\(s\), [0-9]+ B in -> [0-9]+ B" "$store"/*/stdout.log 2>/dev/null \
             | tail -1 | grep -oE "[0-9]+ B$" | grep -oE "[0-9]+")
  in_b=$(grep -hoE "^stored [0-9]+ blob\(s\), [0-9]+ B in" "$store"/*/stdout.log 2>/dev/null \
         | tail -1 | grep -oE "[0-9]+ B in" | grep -oE "[0-9]+")
  t2=$(ls -1 "$store"/cte_tier2.dat* 2>/dev/null | head -1)
  if [ -n "$t2" ]; then
    sync -f "$t2" 2>/dev/null
    t2rep=$(python3 -c "
import numpy as np, os, sys
p = sys.argv[1]; st = os.stat(p)
nz = int(np.count_nonzero(np.memmap(p, dtype=np.uint8, mode='r'))) if st.st_size else 0
print(f'tier2 {os.path.basename(p)}: {st.st_blocks * 512} B allocated, {nz} B non-zero')" "$t2" 2>/dev/null)
  fi
  walrep=$(python3 "$HERE/lib/decode_wal.py" --summary "$store/$tag/cte_metadata_log" 2>/dev/null)
  # The raw stored bytes are never kept: measured above, dropped here.
  find "$store" -maxdepth 2 -type f \( -name chi_bdev.dat -o -name 'cte_tier.dat*' \
       -o -name 'cte_tier2.dat*' \) -delete 2>/dev/null
  local failed
  failed=$(grep -hoE "failed: [0-9]+" "$store"/*/stdout.log 2>/dev/null | tail -1)
  echo "     rc=$rc loop=${stage_s:-?}s compute=${sc_s:-?}s total=${total_s:-?}s ${failed:-} ${bound:-}" >&2
  echo "     tier=$tier async_ms=$flush | stored ${stored_b:-?} B | final ${fl#flush: }" >&2
  [ -n "$t2rep" ] && echo "     $t2rep" >&2
  echo "     WAL: ${walrep:-(none)}" >&2
  # A failed run or a lost chunk is recorded as TBD.
  local verdict
  verdict=$(grep -hE "^(VERIFIED|FAILED):" "$store"/*/stdout.log 2>/dev/null | tail -1)
  if [ "$rc" -ne 0 ] || echo "$verdict" | grep -q FAILED \
     || { [ -n "$failed" ] && [ "${failed#failed: }" != 0 ]; }; then
    echo "     NOT RECORDED: rc=$rc ${failed:-} ${verdict%%(*} -- see $store/console.log" >&2
    return 0
  fi
  # compute_min,io_min,total_min,std_min,ratio -- no split without a Baseline loop time
  [ -n "$total_s" ] && awk -v c="$sc_s" -v t="$total_s" -v i="$in_b" -v b="$stored_b" 'BEGIN{
    if (c == "") printf ",,%.4f", t/60.0
    else printf "%.4f,%.4f,%.4f", c/60.0, (t-c)/60.0, t/60.0
    printf ",,"; if (b > 0) printf "%.3f", i/b}'
}

# Warm the page cache so the first arm does not pay the cold read alone.
warm_cache() {
  [ "$FIELDS" = insitu ] && return 0
  [ "$DRY" = 1 ] && return 0
  local -a files
  mapfile -t files < <(find "$FIELDS" -name '*.f32' | sort)
  [ "$MAXF" -gt 0 ] 2>/dev/null && files=( "${files[@]:0:$MAXF}" )
  echo "== warming page cache over ${#files[@]} file(s) (untimed)"
  printf '%s\0' "${files[@]}" | xargs -0 -r cat > /dev/null 2>&1
  # The driver du's the whole tree per arm: ~11 ms a file cold on Lustre.
  du -sm "$FIELDS" > /dev/null 2>&1
}

echo "== figure 9: $WL / $SIZE -> $OUT"
if [ "$DRY" != 1 ]; then
  echo "== node at start: $(node_state)"
  clean_leftovers
  echo "== after cleaning prior leftovers: $(node_state)"
fi
warm_cache
rm -f "$OUT/baseline_stage_s"
echo "panel,workload,strategy,compute_min,io_min,total_min,std_min,ratio" > "$CSV"

for spec in "${ARMS[@]}"; do
  IFS='|' read -r label panel cfg eb tier flush <<< "$spec"
  [ "$PANEL" = both ] || [ "$PANEL" = "$panel" ] || continue

  split=$(run_arm "$label" "$cfg" "$eb" "$tier" "$flush")
  echo "$panel,$WLNAME,\"$label\",${split:-,,,,}" >> "$CSV"
done

# TBD rows for the other workloads keep the plot layout fixed.
for w in VPIC Nyx LAMMPS WarpX AI; do
  [ "$w" = "$WLNAME" ] && continue
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r label panel _ _ _ _ <<< "$spec"
    [ "$PANEL" = both ] || [ "$PANEL" = "$panel" ] || continue
    echo "$panel,$w,\"$label\",,,,," >> "$CSV"
  done
done

echo; echo "csv: $CSV"
[ "$DRY" = 1 ] && exit 0
python3 "$HERE/plot/plot_fig9.py" --csv "$CSV" --out "$OUT/figures"
