#!/usr/bin/env bash
#===============================================================================
# figure_9.sh -- end-to-end wall-clock time per workload, for Figure 9.
#
#   ./figure_9.sh -w nyx -s smoke            # -w nyx|vpic|warpx|lammps|ai
#   ./figure_9.sh -w nyx -s smoke --panel b  # external baselines only
#   ./figure_9.sh -w nyx -s smoke --dry-run
#
# Writes <out>/fig9.csv (panel,workload,strategy,compute_min,io_min,total_min,
# std_min,ratio,eb) and plots it; an arm that did not complete is written as TBD.
#
# Arms: Baseline (no compression), a fixed nvCOMP codec, NeuroPress, each with
# a RAM tier over a file tier (+Tier, BENCH_TIER*), a periodic flush (+Async,
# BENCH_FLUSH_MS) and a lossy bound. Every arm ends with a timed FlushData
# (CLIO_REPLAY_FINAL_FLUSH); nothing fsyncs. The write-time decompression
# diagnostics (MEASURE_DT, MEASURE_QUALITY) are OFF: see where they are set.
#
# THE SPLIT: solid = the write loop; light = total - that, i.e. the input read
# and the final flush. H2D staging is in NEITHER. EVERY workload is a replay of
# .f32 dumps, LAMMPS included, so all five are measured the same way.
# NOTE the solid segment also contains the awaited tier put, which for an
# UNTIERED arm is a real (buffered) file write -- 435-673 MB/s measured -- so
# "solid" is not compute-only. See README in figures/fig9/full.
# H2D is an artifact of replaying dumps from disk -- upstream's write path never
# stages, its VOL call takes a device pointer -- so it is subtracted from the
# loop and the total before anything else. Simulate evolves the data and is not
# the write path under test. Both halves are measured elapsed time and the split
# is additive and non-negative on every arm.
#
# Two definitions were tried and rejected, because a reader will ask:
#  * loop - Baseline's loop ("what compression adds"), floored at 0. It mixes
#    any I/O difference into the codec term and collapses whenever compression
#    pays for itself: it read 0.00 s for AI's 32x lossy arm and 3.14 s for AI's
#    nvCOMP arm whose codec kernel took 0.74 s.
#  * the per-chunk phase sums (compress + preproc + ...). Chunks PIPELINE, so
#    those are per-chunk latencies, not shares of elapsed time -- on AI they sum
#    to 52x (nvCOMP) and 77x (cuSZ) of the arm's total. They cannot stack.
# The codec's own cost is reported per arm as `codec=` in the log instead: a
# CUDA-event bracket around the compress call, the same quantity NeuroPress
# measures (gpucompress_compress.cpp) and the one its model ranks on. It is a
# per-chunk cost, not a slice of wall-clock, so it does not belong in the bar.
# io = total - compute. Nyx, VPIC, WarpX and AI replay .f32 dumps (WarpX and
# AI through nyx/run_config.sh); LAMMPS runs in situ. No bound check.
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

WL=nyx SIZE=smoke EB_LOW=1e-3
# WHERE THE BYTES LAND. Untiered arms write straight to the PFS (tier 2 is
# unused; the single file target sits under --out, on Lustre). Tiered arms put
# to a RAM tier and spill to TIER2_ROOT, which is NVMe. Point TIER2_ROOT at a
# Lustre path to put both classes back on the same device.
TIER2_ROOT=${TIER2_ROOT:-/work/nvme/bekn/imuradli/fig9-tier2}
PANEL=both DRY=0 OUT="" FIELDS="" ONLY=""
BEST_FIXED=${BEST_FIXED:-}     # default per workload, below
ASYNC_MS=${ASYNC_MS:-500}      # periodic flush for the +Async arms
# RAM tier 1, eagerly committed at runtime start. Sized as RAM_PCT of the payload
# this run will actually replay, so it scales with --size and --max-files instead
# of being a fixed 512 MB that held 2-40% of one full-size arm's output and made
# the +Tier arms spill immediately. Set RAM_MB to override with an absolute value.
RAM_PCT=${RAM_PCT:-10}
# Storage bandwidth in the NeuroPress cost model, BYTES PER MILLISECOND. The
# same 1.2e6 (1.2 GB/s) the per-chunk oracle scores under, so selection
# optimizes the cost it is compared on; the shipped default is 5e6.
COST_BW=${COST_BW:-1.2e6}
# ONLINE-LEARNING ARMS (config `learn`): rate and MAPE gate, matching figure_8.sh
# so a learning arm here is comparable with the learning figure. Without these
# the runtime default rate stands and the arm is not comparable.
NP_LR=${NP_LR:-0.2}
NP_MAPE=${NP_MAPE:-0.10}
# WRITE-TIME DIAGNOSTICS, OFF BY DEFAULT. Both decompress inside the timed loop,
# and this figure measures a write path -- nothing in it decompresses.
#   MEASURE_DT=1      a full extra decompression per chunk. Measured median
#                     1.47 ms/chunk on Nyx, MORE than the compression (0.12 ms)
#                     and the I/O (0.24 ms) together, and it lands on the eb=0
#                     arms only, so panel (a) is not internally comparable.
#   MEASURE_QUALITY=1 a decompression + PSNR, and NOT symmetric across arms:
#                     per 240 Nyx chunks it fired 102x on NeuroPress and 112x on
#                     the best fixed nvCOMP arm, but 0x on cuSZp3.
# Set either to 1 to quantify the tax; arms are comparable only with both equal.
MEASURE_DT=${MEASURE_DT:-0}
MEASURE_QUALITY=${MEASURE_QUALITY:-0}
# WHICH MODE EVERY NEUROPRESS ARM RUNS IN. `learn` = inference + online SGD from
# the measured outcome of the action it executed (one label per chunk); `dynamic`
# = frozen weights, no feedback at all. Two arms below are pinned to `dynamic`
# whatever this is set to, as the no-learning reference.
#   Learning needs chunks: it moved Nyx's achieved ratio 9.78 -> 10.6 over 2000
#   chunks (83% of the way to the best hindsight-fixed action) but is pure noise
#   over 240, so run this with MAXF large enough.
NP_CFG=${NP_CFG:-learn}

usage() { sed -n '3,9p' "$0" >&2; cat >&2 <<'U'

  --workload, -w  nyx | vpic | warpx | lammps | ai   (default nyx)
  --size, -s      smoke | full                       (default smoke)
  --panel         a | b | both                       (default both)
  --out DIR       results directory
  --fields DIR    dump directory (default: per workload, as compare_wallclock.sh)
  --only LABEL    run only the arm with this exact label (rerun one arm; point
                  --out at a FRESH dir, because the CSV is truncated at start)
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
    --only)        ONLY=$2; shift 2 ;;   # rerun ONE arm, by exact label
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
    # LAMMPS is replayed from a dump like the other four, so every workload is
    # measured the same way: a real input read, and no simulate time in the bar.
    # The dump is the driver's own --raw output, byte-identical to what the
    # in-situ path hands the compressor (job 22190101, 3933 blobs verified).
    lammps) FIELDS=${LAMMPS_FIELDS:-/work/hdd/bekn/imuradli/np-lammps-30g/fields} ;;
  esac
fi
if [ -z "$(find "$FIELDS" -name '*.f32' -print -quit 2>/dev/null)" ]; then
  echo "no field dumps at $FIELDS -- pass --fields DIR or run $WL/gen_fields.sh" >&2
  exit 3
fi

# Replay hands the compressor host memory, which it refuses unless staged.
export CLIO_NEUROPRESS_STAGE_H2D=1

case "$WL" in nyx|vpic) DRIVER=$HERE/$WL/run_config.sh ;; *) DRIVER=$HERE/nyx/run_config.sh ;; esac

# ARM_TIMEOUT (s): a smoke arm is ~4 s; a full 30 GB replay arm is ~6 min.
MAXF_ENV=${MAXF:-}   # an explicit MAXF must survive the per-workload overrides
case "$SIZE" in smoke) CHUNK=2097152; MAXF=${MAXF:-60}; ARM_TIMEOUT=${ARM_TIMEOUT:-40} ;;
                full)  CHUNK=8388608; MAXF=${MAXF:-0};  ARM_TIMEOUT=${ARM_TIMEOUT:-1800} ;; esac
# An AI file is a whole 327 MiB tensor.
# An AI file is a whole 327 MiB tensor, so 2 files is the smoke default -- but an
# explicit MAXF wins, because online learning needs far more than 328 chunks.
[ "$WL" = ai ] && [ "$SIZE" = smoke ] && MAXF=${MAXF_ENV:-2}
# ---- payload this run will replay, and the RAM tier as a percentage of it ----
if [ -z "${RAM_MB:-}" ]; then
  PAYLOAD_MB=$(find "$FIELDS" -name '*.f32' -printf '%s\n' 2>/dev/null | sort -rn \
    | awk -v n="$MAXF" 'n == 0 || NR <= n { s += $1 } END { printf "%.0f", s / 1048576 }')
  RAM_MB=$(awk -v p="${PAYLOAD_MB:-0}" -v pc="$RAM_PCT" \
    'BEGIN { v = p * pc / 100.0; if (v < 64) v = 64; printf "%.0f", v }')
  echo "== payload ${PAYLOAD_MB:-?} MiB -> RAM tier ${RAM_MB} MiB (${RAM_PCT}%)"
fi

# arm  ::  label | panel | config | eb | tier(0/1) | async_ms
#          async_ms 0 = periodic flush disabled entirely
ARMS=(
  # PANEL (a): the ablation ladder. Untiered arms write straight to the PFS;
  # every +Tier arm spills to NVMe (TIER2_ROOT). One error bound only, 1e-3.
  "Baseline|a|baseline|0|0|0"
  "nvCOMP|a|$BEST_FIXED|0|0|0"
  "nvCOMP+Tier|a|$BEST_FIXED|0|1|0"
  "NP only|a|$NP_CFG|0|0|0"
  "NP+Tier|a|$NP_CFG|0|1|0"
  "NP+Tier+Async|a|$NP_CFG|0|1|$ASYNC_MS"
  "NP+Tier+Async+Lossy|a|$NP_CFG|$EB_LOW|1|$ASYNC_MS"
  # PANEL (b): every codec twice, at the SAME bound, so the tier is the only
  # variable. The PFS row writes each chunk through to Lustre inside the timed
  # loop; the +Tier row puts to RAM and drains to NVMe behind the 500 ms flush.
  # This is what makes it a controlled comparison -- previously the four
  # external codecs ran untiered while NeuroPress kept the tier, so the panel
  # compared NeuroPress-as-a-system against codecs alone.
  "Best fixed nvCOMP|b|$BEST_FIXED|$EB_LOW|0|0"
  "Best fixed nvCOMP+Tier|b|$BEST_FIXED|$EB_LOW|1|$ASYNC_MS"
  "ndzip|b|static-ndzip|$EB_LOW|0|0"
  "ndzip+Tier|b|static-ndzip|$EB_LOW|1|$ASYNC_MS"
  "cuSZp3|b|static-cuszp|$EB_LOW|0|0"
  "cuSZp3+Tier|b|static-cuszp|$EB_LOW|1|$ASYNC_MS"
  "cuSZ|b|static-cusz|$EB_LOW|0|0"
  "cuSZ+Tier|b|static-cusz|$EB_LOW|1|$ASYNC_MS"
  "NeuroPress|b|$NP_CFG|$EB_LOW|0|0"
  "NeuroPress+Tier|b|$NP_CFG|$EB_LOW|1|$ASYNC_MS"
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

# ELAPSED time covered by a per-chunk phase, in seconds: the UNION of the
# [start, start+dur] intervals, NOT their sum. Chunks pipeline, so a sum
# over-counts -- on AI the per-chunk walls sum to 52-77x the arm's total.
# Falls back to the sum when the log has no start column (an older run).
#   phase_union <log> <dur column> <start column>
phase_union() {
  awk -F, -v dcol="$2" -v scol="$3" '
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; d = col[dcol]; t = col[scol]; next }
    !d { next }
    t && $t != "" && $t + 0 >= 0 && $d + 0 > 0 { n++; st[n] = $t + 0; en[n] = $t + $d * 1e6 }
    $d + 0 > 0 { tot += $d }
    END {
      if (n == 0) { printf "%.3f", tot / 1000.0; exit }
      for (i = 2; i <= n; i++) {
        a = st[i]; b = en[i]; j = i - 1
        while (j >= 1 && st[j] > a) { st[j+1] = st[j]; en[j+1] = en[j]; j-- }
        st[j+1] = a; en[j+1] = b
      }
      cs = st[1]; ce = en[1]; u = 0
      for (i = 2; i <= n; i++) {
        if (st[i] > ce) { u += ce - cs; cs = st[i]; ce = en[i] }
        else if (en[i] > ce) { ce = en[i] }
      }
      u += ce - cs
      printf "%.3f", u / 1e9
    }' "$1"
}

run_arm() {  # run_arm <label> <config> <eb> <tier> <async_ms> -> echoes total minutes, or empty
  local label=$1 cfg=$2 eb=$3 tier=$4 flush=$5
  local tag; tag=$(slug "$label")
  local store="$OUT/$tag"
  mkdir -p "$store"

  # BENCH_FLUSH_MS is ALWAYS set: 0 is the only way to turn the periodic flush off.
  # The phase log is what makes the host-to-device staging visible per chunk, so
  # it can be taken back out below.
  local -a env_kv=( "BENCH_FLUSH_MS=$flush" "CLIO_REPLAY_FINAL_FLUSH=1"
                    "MEASURE_DT=$MEASURE_DT" "MEASURE_QUALITY=$MEASURE_QUALITY"
                    "CLIO_NEUROPRESS_PHASE_LOG=$store/phases.csv" )
  local t2dir="$TIER2_ROOT/$tag"
  if [ "$tier" = 1 ]; then
    mkdir -p "$t2dir" || { echo "cannot create tier-2 dir $t2dir" >&2; return 0; }
    env_kv+=( "BENCH_TIER1_TYPE=ram" "BENCH_TIER1_MB=$RAM_MB"
              "BENCH_TIER1_PERSIST=volatile"
              "BENCH_TIER2_PATH=$t2dir/cte_tier2.dat" )
  fi
  # `learn` turns on neuropress_online_learning_enabled; it needs a rate. It gets
  # ONE label per chunk -- the measured outcome of the action it executed -- so it
  # cannot see an action it never picks. `explore*` is what adds K alternatives.
  case "$cfg" in learn*|explore*) env_kv+=( "BENCH_NP_LR=$NP_LR" "BENCH_NP_MAPE=$NP_MAPE" ) ;; esac

  local -a cmd=( "$DRIVER" "$cfg" --eb "$eb" --bw "$COST_BW"
                 --results "$store" --tag "$tag" --chunk "$CHUNK" )
  cmd+=( --fields "$FIELDS" )
  [ "$MAXF" -gt 0 ] 2>/dev/null && cmd+=( --max-files "$MAXF" )

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
  # "time: read A s   stage+compress B s   total C s"
  local tline stage_s sc_s="" total_s
  tline=$(grep -hE "time: read" "$store"/*/stdout.log 2>/dev/null | tail -1)
  stage_s=$(echo "$tline" | grep -oE "stage\+compress[[:space:]]+[0-9.]+" | grep -oE "[0-9.]+$")
  total_s=$(echo "$tline" | grep -oE "total[[:space:]]+[0-9.]+"            | grep -oE "[0-9.]+$")
  # THE FIGURE ASSUMES THE DATA IS ALREADY ON THE GPU, as upstream's measured
  # path does (its VOL write takes a device pointer). This replay reads dumps
  # from disk and must stage every chunk up, so subtract that staging -- summed
  # per chunk from the phase log -- from the write loop and the total. An
  # in-situ arm stages nothing and loses nothing here; Baseline never stages.
  local h2d_s=0 plog
  plog=$(ls -1 "$store"/phases.csv "$store"/*/phases.csv 2>/dev/null | head -1)
  if [ -n "$plog" ]; then
    # ELAPSED staging, not the sum: chunks run concurrently, so summing the
    # per-chunk h2d over-counts (on AI the per-chunk walls overlap 52-77x, and
    # the sum over-subtracted up to 3.8% of total). h2d_start_ns lets us take
    # the UNION of the [start, start+h2d_ms] intervals, which is what elapsed.
    # A log without that column (a run from before it existed) falls back to
    # the sum, and the harness says so.
    h2d_s=$(phase_union "$plog" h2d_ms h2d_start_ns)
    local sub; sub=$(awk -v a="${stage_s:-0}" -v t="${total_s:-0}" -v h="${h2d_s:-0}" \
      'BEGIN { print (h > 0 && a - h > 0 && t - h > 0) ? 1 : 0 }')
    if [ "$sub" = 1 ]; then
      stage_s=$(awk -v a="$stage_s" -v h="$h2d_s" 'BEGIN{printf "%.3f", a - h}')
      total_s=$(awk -v t="$total_s" -v h="$h2d_s" 'BEGIN{printf "%.3f", t - h}')
    fi
  fi
  # The compression work, summed per chunk from the phase log. An arm with no
  # phase log (Baseline compresses nothing) contributes 0.
  local codec_s=0
  if [ -n "$plog" ]; then
    codec_s=$(awk -F, '
      BEGIN { nw = split("compress_ms preproc_ms stats_ms nn_ms choice_ms factory_ms", w, " ") }
      NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
      { for (n = 1; n <= nw; n++) { i = col[w[n]]; if (i != "") s += $i + 0 } }
      END { printf "%.3f", s / 1000.0 }' "$plog")
  fi
  # io_s is REPORTED but NOT subtracted, and it is NOT what the light bar shows.
  # Audited 2026-09-18: io_ms is a real awaited write (AsyncPutBlob -> PutBlobImpl
  # -> ModifyExistingData -> bdev AsyncWrite, polled to completion in
  # fs_bdev_transport.cc) -- buffered, not fsynced. For an UNTIERED arm that write
  # lands on the file tier inside the loop, so 100% of its durable write is in the
  # solid bar; a +Tier+Async arm puts to RAM at memcpy speed and 0-38% is, with the
  # rest moved later by an untimed periodic FlushData worker. A wall-clock
  # compute/IO split is therefore NOT derivable here: the same field measures a
  # file write on some arms and a memcpy on others, and the per-chunk intervals
  # overlap on AI. `codec=` is a per-chunk cost, not a slice of elapsed time.
  # CAVEAT: phase_union and codec_s do not filter path=="write", so the io= and
  # codec= printed below also include verification-read latency. Log-only.
  # Solid = the write loop alone, capped into [0, total].
  [ -n "$total_s" ] && [ -n "$stage_s" ] && sc_s=$(awk -v l="$stage_s" \
    -v t="$total_s" 'BEGIN{ v = l; if (v > t) v = t; if (v < 0) v = 0; printf "%.3f", v }')
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
  t2=$(ls -1 "$t2dir"/cte_tier2.dat* "$store"/cte_tier2.dat* 2>/dev/null | head -1)
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
  # Tier 2 is on a small NVMe allocation: it MUST go before the next arm starts.
  rm -rf "$t2dir" 2>/dev/null
  local failed
  failed=$(grep -hoE "failed: [0-9]+" "$store"/*/stdout.log 2>/dev/null | tail -1)
  echo "     rc=$rc loop=${stage_s:-?}s codec=${codec_s:-?}s io=${io_s:-?}s compute=${sc_s:-?}s total=${total_s:-?}s (H2D staging ${h2d_s}s removed) ${failed:-} ${bound:-}" >&2
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
echo "== NeuroPress arms run config '$NP_CFG'; the two (no learn) arms are 'dynamic'"
echo "== write-time diagnostics: MEASURE_DT=$MEASURE_DT MEASURE_QUALITY=$MEASURE_QUALITY" \
     "(0 = not charged to the timed loop)"
if [ "$DRY" != 1 ]; then
  echo "== node at start: $(node_state)"
  clean_leftovers
  echo "== after cleaning prior leftovers: $(node_state)"
fi
warm_cache
echo "panel,workload,strategy,compute_min,io_min,total_min,std_min,ratio,eb" > "$CSV"

for spec in "${ARMS[@]}"; do
  IFS='|' read -r label panel cfg eb tier flush <<< "$spec"
  [ "$PANEL" = both ] || [ "$PANEL" = "$panel" ] || continue
  [ -z "$ONLY" ] || [ "$label" = "$ONLY" ] || continue

  split=$(run_arm "$label" "$cfg" "$eb" "$tier" "$flush")
  echo "$panel,$WLNAME,\"$label\",${split:-,,,,},$eb" >> "$CSV"
done

# TBD rows for the other workloads keep the plot layout fixed.
for w in VPIC Nyx LAMMPS WarpX AI; do
  [ "$w" = "$WLNAME" ] && continue
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r label panel _ eb _ _ <<< "$spec"
    [ "$PANEL" = both ] || [ "$PANEL" = "$panel" ] || continue
    echo "$panel,$w,\"$label\",,,,,,$eb" >> "$CSV"
  done
done

echo; echo "csv: $CSV"
[ "$DRY" = 1 ] && exit 0
python3 "$HERE/plot/plot_fig9.py" --csv "$CSV" --out "$OUT/figures"
