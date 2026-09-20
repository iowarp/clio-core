#!/usr/bin/env bash
#===============================================================================
# figure_9.sh -- end-to-end wall-clock time per workload, for Figure 9.
#
#   ./figure_9.sh -w nyx -s smoke            # -w nyx|vpic|warpx|lammps|ai
#   ./figure_9.sh -w nyx -s smoke --panel b  # external baselines only
#   ./figure_9.sh -w nyx -s smoke --dry-run
#
# Writes three files under <out>: fig9.csv (panel,workload,strategy,compute_min,
# io_min,total_min,std_min,ratio,eb), arms.csv (the exact arm set this run used)
# and run.json (the parameters behind it), then plots the CSV. An arm that did
# not complete is left blank and drawn as TBD.
#
# Arms: Baseline (no compression), a fixed nvCOMP codec and NeuroPress, each
# with a RAM tier over a file tier (+Tier, BENCH_TIER*), a periodic flush
# (+Async, BENCH_FLUSH_MS) and a lossy bound. Every arm ends with a timed
# FlushData (CLIO_REPLAY_FINAL_FLUSH); nothing fsyncs. The write-time
# decompression diagnostics (MEASURE_DT, MEASURE_QUALITY) are OFF by default:
# both decompress inside the timed loop, and this figure measures a write path.
#
# THE SPLIT: solid = the write loop; light = total - that, i.e. the input read
# and the final flush. H2D staging is in NEITHER -- the figure assumes the data
# is already on the GPU, as upstream's VOL write does, so the ELAPSED staging
# (a union of the per-chunk intervals, not their sum) comes off the loop and
# the total before anything else. Every workload replays .f32 dumps, LAMMPS
# included, so all five are measured the same way.
#
# `io_min` IS NOT AN I/O MEASUREMENT. It is total - loop, and for an untiered
# arm the durable write happens inside the loop. A true compute/IO split is not
# derivable from this instrumentation; fig9.md beside this script records why, and the two
# alternative splits that were tried and rejected.
#
# Environment: PFS_ROOT, NVME_ROOT, EB_LOW, PANEL_B_TIER, NP_CFG, MAXF, RAM_PCT/RAM_MB,
# COST_BW, NP_LR, NP_MAPE, MEASURE_DT, MEASURE_QUALITY, ARM_TIMEOUT, BEST_FIXED.
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)   # this figure's own directory
BENCH=$(cd "$HERE/../.." && pwd)      # paper-benchmark/: the harnesses and the drivers

WL=nyx SIZE=smoke
# The lossy bound for panel (a)'s lossy arm and for every panel (b) arm. ONE
# bound: an earlier version swept a 1e-3 / 1e-2 / 1e-1 ladder across three
# panel (a) arms, which is three more full-size runs for a comparison the
# figure does not make. Set EB_LOW to move it.
EB_LOW=${EB_LOW:-1e-3}
# PANEL (b) runs every codec TWICE at the same bound -- once writing through to
# the PFS, once over the tier -- so the tier is the only variable. Before that,
# the external codecs ran untiered while NeuroPress kept its tier, which
# compared NeuroPress-as-a-system against codecs alone. PANEL_B_TIER=0 restores
# that older five-arm shape, which is what the published campaign was run in.
PANEL_B_TIER=${PANEL_B_TIER:-1}
# WHERE THE BYTES LAND -- one device class per arm class, set explicitly.
#   PFS_ROOT   an UNTIERED arm (Baseline included) puts its tier-1 file here.
#   NVME_ROOT  a +Tier arm keeps tier 1 in RAM and spills tier 2 here.
# Each arm gets its own subdirectory, removed as soon as it is measured; only
# the CSVs and logs stay under --out.
#
# This used to be one root. The untiered arms took their tier from --out, whose
# default is under the repo on /u -- NFS home, not the PFS. Measured on Delta
# 2026-09-19 (3x512 MiB, conv=fdatasync): /u 587-875 MB/s, /work/nvme 434-502,
# /work/hdd comparable to /work/nvme. So the untiered arms had the FASTEST
# durable device of the three and the +Tier arms the slowest, which understates
# exactly what the tiering ablation is trying to show.
#   /work/hdd  HDD-backed Lustre -- the PFS
#   /work/nvme NVMe-backed Lustre: NVMe media, still over the network
# On a COMPUTE node /tmp is node-local NVMe, so NVME_ROOT=/tmp/fig9-$SLURM_JOB_ID
# makes tier 2 a genuinely local device instead of NVMe over tcp. That is the
# sharper contrast; the default stays on /work/nvme because it survives the job.
CLIO_ACCT=${CLIO_ACCT:-}
if [ -z "$CLIO_ACCT" ]; then
  # The allocation, from group membership rather than hardcoded: delta_<acct>
  # with a matching /work/hdd/<acct>. delta_active_users has no such directory
  # and is skipped.
  for _g in $(id -Gn 2>/dev/null); do
    case $_g in
      delta_*) [ -d "/work/hdd/${_g#delta_}" ] && CLIO_ACCT=${_g#delta_} && break ;;
    esac
  done
  unset _g
fi
_WHO=${USER:-$(id -un)}
PFS_ROOT=${PFS_ROOT:-/work/hdd/$CLIO_ACCT/$_WHO/fig9-pfs}
# TIER2_ROOT is the old spelling, still honoured.
NVME_ROOT=${NVME_ROOT:-${TIER2_ROOT:-/work/nvme/$CLIO_ACCT/$_WHO/fig9-nvme}}
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
  --fields DIR    dump directory (default: per workload, from the environment)
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
# Every workload the figure has a column for. One run fills its own and leaves
# the rest as TBD rows, so a single-workload CSV still plots at the full layout
# and several merge without reordering. Must match the plot's WORKLOAD_ORDER.
WORKLOADS_ALL=(VPIC Nyx LAMMPS WarpX AI)
# The oracle sweep's best fixed action (../../compare_perchunk_oracle.sh).
if [ -z "$BEST_FIXED" ]; then
  case "$WL" in vpic) BEST_FIXED=static-ans-q-s4 ;; *) BEST_FIXED=static-bitcomp-q-s4 ;; esac
fi
OUT=${OUT:-$BENCH/results/figure9/$WL-$SIZE}
mkdir -p "$OUT"
CSV="$OUT/fig9.csv"

if [ -z "$FIELDS" ]; then
  case "$WL" in
    nyx)    FIELDS=${NYX_FIELDS:-$BENCH/nyx/fields} ;;
    vpic)   FIELDS=${VPIC_FIELDS:-$BENCH/vpic/fields} ;;
    warpx)  FIELDS=${WARPX_FIELDS:-$BENCH/warpx/fields} ;;
    ai)     FIELDS=${AI_FIELDS:-$BENCH/ai/fields} ;;
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

case "$WL" in nyx|vpic) DRIVER=$BENCH/$WL/run_config.sh ;; *) DRIVER=$BENCH/nyx/run_config.sh ;; esac

# ARM_TIMEOUT (s): a smoke arm is ~4 s; a full 30 GB replay arm is ~6 min.
MAXF_ENV=${MAXF:-}   # an explicit MAXF must survive the per-workload overrides
case "$SIZE" in smoke) CHUNK=2097152; MAXF=${MAXF:-60}; ARM_TIMEOUT=${ARM_TIMEOUT:-40} ;;
                full)  CHUNK=8388608; MAXF=${MAXF:-0};  ARM_TIMEOUT=${ARM_TIMEOUT:-1800} ;; esac
# An AI file is a whole 327 MiB tensor, so 2 files is the smoke default -- but an
# explicit MAXF wins, because online learning needs far more than 328 chunks.
[ "$WL" = ai ] && [ "$SIZE" = smoke ] && MAXF=${MAXF_ENV:-2}
# ---- payload this run will replay, and the RAM tier as a percentage of it ----
if [ -z "${RAM_MB:-}" ]; then
  # BY PATH, then the first MAXF -- the driver sorts its file list and truncates
  # it (neuropress_field_replay.cc:452-453), so sizing from the MAXF LARGEST
  # files measured a set it never replays whenever the dumps differ in size.
  PAYLOAD_MB=$(find "$FIELDS" -name '*.f32' -printf '%p\t%s\n' 2>/dev/null | sort \
    | awk -F'\t' -v n="$MAXF" 'n == 0 || NR <= n { s += $2 } END { printf "%.0f", s / 1048576 }')
  RAM_MB=$(awk -v p="${PAYLOAD_MB:-0}" -v pc="$RAM_PCT" \
    'BEGIN { v = p * pc / 100.0; if (v < 64) v = 64; printf "%.0f", v }')
  echo "== payload ${PAYLOAD_MB:-?} MiB -> RAM tier ${RAM_MB} MiB (${RAM_PCT}%)"
fi

# arm  ::  label | panel | config | eb | tier(0/1) | async_ms
#          async_ms 0 = periodic flush disabled entirely
# The label is the plot's key, so an arm that is renamed here is a new bar
# there. arms.csv records the set this run built, so a CSV is never ambiguous
# about which shape it came from.
#
# PANEL (a): the ablation ladder. Untiered arms write straight to the PFS;
# every +Tier arm spills to NVMe (TIER2_ROOT).
ARMS=(
  "Baseline|a|baseline|0|0|0"
  "nvCOMP|a|$BEST_FIXED|0|0|0"
  "nvCOMP+Tier|a|$BEST_FIXED|0|1|0"
  "NP only|a|$NP_CFG|0|0|0"
  "NP+Tier|a|$NP_CFG|0|1|0"
  "NP+Tier+Async|a|$NP_CFG|0|1|$ASYNC_MS"
  # ...and the lossy end of that ladder.
  "NP+Tier+Async+Lossy|a|$NP_CFG|$EB_LOW|1|$ASYNC_MS"
)
# PANEL (b): the external codecs and NeuroPress at the SAME bound. With
# PANEL_B_TIER=1 each runs twice so the tier is the only variable: the PFS row
# writes each chunk through to Lustre inside the timed loop, the +Tier row puts
# to RAM and drains to NVMe behind the periodic flush.
for _b in "Best fixed nvCOMP:$BEST_FIXED" "ndzip:static-ndzip" \
          "cuSZp3:static-cuszp" "cuSZ:static-cusz" "NeuroPress:$NP_CFG"; do
  _bl=${_b%%:*}; _bc=${_b#*:}
  ARMS+=( "$_bl|b|$_bc|$EB_LOW|0|0" )
  [ "$PANEL_B_TIER" = 1 ] && ARMS+=( "$_bl+Tier|b|$_bc|$EB_LOW|1|$ASYNC_MS" )
done
unset _b _bl _bc

# arm_selected <label> <panel> -- does --panel / --only let this arm run?
# One definition, used by the preflight below and by the run loop, so the
# preflight cannot demand a resource that no selected arm will touch.
arm_selected() {
  { [ "$PANEL" = both ] || [ "$PANEL" = "$2" ]; } &&
  { [ -z "$ONLY" ] || [ "$1" = "$ONLY" ]; }
}

# Both device roots must be writable before the first arm needs one, not
# discovered by it: a failed mkdir inside run_arm skips that arm silently and
# the CSV just shows a gap. Each root is checked only when a SELECTED arm will
# actually use it, and the check reports the filesystem it resolved to, so a
# root that is not the device class its name claims is visible at the top of
# every job log rather than inferred afterwards.
_want_pfs=0 _want_nvme=0
for _spec in "${ARMS[@]}"; do
  IFS='|' read -r _l _p _c _e _t _f <<< "$_spec"
  if arm_selected "$_l" "$_p"; then
    [ "$_t" = 1 ] && _want_nvme=1 || _want_pfs=1
  fi
done
unset _spec _l _p _c _e _t _f
# require_root <role> <env var> <needed 0/1> <root> <what happens there>
# Prints the filesystem the root resolved to, so a root that is not the device
# class its name claims shows up at the top of the job log instead of being
# inferred from the numbers afterwards.
require_root() {
  [ "$3" = 1 ] || return 0
  if ! mkdir -p "$4" 2>/dev/null || [ ! -w "$4" ]; then
    echo "cannot write the $1 root: $4" >&2
    echo "  every $5 there. Set $2 to a writable path on that device class." >&2
    [ -z "$CLIO_ACCT" ] && echo "  (no delta_<acct> group with a matching /work/hdd/<acct> was" >&2 &&
      echo "   found, so the default path is incomplete -- set CLIO_ACCT or $2.)" >&2
    exit 4
  fi
  echo "== $1 root: $4  [$(df -Th "$4" 2>/dev/null | tail -1 | awk '{print $2}')]"
}
if [ "$DRY" != 1 ]; then
  require_root PFS  PFS_ROOT  "$_want_pfs"  "$PFS_ROOT" \
    "untiered arm (Baseline included) puts its tier-1 file"
  require_root NVMe NVME_ROOT "$_want_nvme" "$NVME_ROOT" \
    "+Tier arm spills tier 2"
fi

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
  # WHERE THIS ARM'S DURABLE BYTES GO. Untiered -> a tier-1 file on the PFS;
  # tiered -> tier 1 in RAM (a `ram` bdev is shared memory, so its path is only
  # a name and no file is created) and tier 2 on NVMe. One device class per arm
  # class, so `+Tier` is not also a change of filesystem.
  # Keyed by WORKLOAD then arm: the campaign runs one job per workload, and
  # every workload builds the same arm tags, so without the $WL level two
  # concurrent jobs would both write .../fig9-pfs/baseline and corrupt each
  # other's tier.
  local t2dir="$NVME_ROOT/$WL/$tag" pfsdir="$PFS_ROOT/$WL/$tag"
  if [ "$tier" = 1 ]; then
    mkdir -p "$t2dir" || { echo "cannot create the NVMe tier-2 dir $t2dir" >&2; return 0; }
    env_kv+=( "BENCH_TIER1_TYPE=ram" "BENCH_TIER1_MB=$RAM_MB"
              "BENCH_TIER1_PERSIST=volatile"
              "BENCH_TIER2_PATH=$t2dir/cte_tier2.dat" )
  else
    mkdir -p "$pfsdir" || { echo "cannot create the PFS tier-1 dir $pfsdir" >&2; return 0; }
    env_kv+=( "BENCH_TIER1_PATH=$pfsdir/cte_tier.dat" )
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
    # THE TIER IMAGE TOO. This path used to return without reaching the cleanup
    # further down, so a killed arm leaked a tier the size of the payload:
    # three timed-out LAMMPS arms left 65 GB behind and exhausted the project
    # quota, which then failed every later arm with "Disk quota exceeded".
    # rc=137 is SIGKILL -- both what `timeout -k` sends and what the OOM killer
    # sends -- so this is the likeliest arm of all to leak.
    rm -rf "$t2dir" "$pfsdir" 2>/dev/null
    find "$store" -maxdepth 2 -type f \( -name chi_bdev.dat \
         -o -name 'cte_tier.dat*' -o -name 'cte_tier2.dat*' \) -delete 2>/dev/null
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
  # The per-chunk cost of DECIDING and compressing -- the codec call plus the
  # preprocessing, the statistics kernel, the forward pass, the ranking and the
  # compressor construction -- summed over chunks. A per-chunk latency, not a
  # slice of elapsed time: chunks pipeline, so this does not stack into the bar.
  # An arm with no phase log (Baseline compresses nothing) contributes 0.
  local codec_s=0
  if [ -n "$plog" ]; then
    codec_s=$(awk -F, '
      BEGIN { nw = split("compress_ms preproc_ms stats_ms nn_ms choice_ms factory_ms", w, " ") }
      NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
      { for (n = 1; n <= nw; n++) { i = col[w[n]]; if (i != "") s += $i + 0 } }
      END { printf "%.3f", s / 1000.0 }' "$plog")
  fi
  # Solid = the write loop alone, capped into [0, total]. There is deliberately
  # no per-arm I/O term: the phase log's io_ms measures a durable file write on
  # an untiered arm and a memcpy on a tiered one, so the two are not the same
  # quantity and cannot share a bar segment. fig9.md, beside this script, has the audit.
  # CAVEAT: codec_s does not filter path=="write", so the codec= printed
  # below also carries verification-read latency. Log-only, never plotted.
  [ -n "$total_s" ] && [ -n "$stage_s" ] && sc_s=$(awk -v l="$stage_s" \
    -v t="$total_s" 'BEGIN{ v = l; if (v > t) v = t; if (v < 0) v = 0; printf "%.3f", v }')
  local bound
  bound=$(grep -hoE "BOUND (OK|FAILED)[^,]*" "$store"/*/stdout.log 2>/dev/null | tail -1)
  local fl
  fl=$(grep -hoE "flush: [0-9]+ blob\(s\), [0-9]+ B moved to durable storage in [0-9.]+ s" "$store"/*/stdout.log 2>/dev/null | tail -1)
  # DEVICE EVIDENCE, taken before the images are deleted: the durable file this
  # arm class is supposed to have written, on the device it is supposed to be
  # on. The file is <path>_node<N>, hence the glob. This is what turns "the
  # untiered arms go to the PFS" from a comment into a per-arm measurement.
  local durable durep="" stored_b in_b walrep
  stored_b=$(grep -hoE "^stored [0-9]+ blob\(s\), [0-9]+ B in -> [0-9]+ B" "$store"/*/stdout.log 2>/dev/null \
             | tail -1 | grep -oE "[0-9]+ B$" | grep -oE "[0-9]+")
  in_b=$(grep -hoE "^stored [0-9]+ blob\(s\), [0-9]+ B in" "$store"/*/stdout.log 2>/dev/null \
         | tail -1 | grep -oE "[0-9]+ B in" | grep -oE "[0-9]+")
  if [ "$tier" = 1 ]; then
    durable=$(ls -1 "$t2dir"/cte_tier2.dat* 2>/dev/null | head -1)
  else
    # $store is the fallback for a run whose compose predates BENCH_TIER1_PATH.
    durable=$(ls -1 "$pfsdir"/cte_tier.dat* "$store"/cte_tier.dat* 2>/dev/null | head -1)
  fi
  if [ -n "$durable" ]; then
    sync -f "$durable" 2>/dev/null
    # The non-zero scan reads the WHOLE file, so it is capped: an untiered
    # Baseline tier holds the entire uncompressed payload (tens of GB) and
    # scanning it would cost more than the arm did. Allocated bytes are a stat.
    #
    # SINGLE-quoted, so Python owns every quote inside. The first version was
    # double-quoted and needed a backslash-escaped quote inside an f-string,
    # which Python 3.9 rejects outright -- and its stderr went to /dev/null, so
    # all nine arms of the first smoke run reported nothing at all. stderr is
    # deliberately left connected now: this line existing is the whole point.
    durep=$(python3 -c '
import os, sys
p = sys.argv[1]
st = os.stat(p)
cap = 256 << 20
if st.st_size and st.st_size <= cap:
    import numpy as np
    nz = "%d B non-zero" % int(
        np.count_nonzero(np.memmap(p, dtype=np.uint8, mode="r")))
elif st.st_size:
    nz = "not scanned (> 256 MiB)"
else:
    nz = "0 B non-zero"
print("%s %s: %d B, %d B allocated, %s"
      % (sys.argv[2], os.path.basename(p), st.st_size,
         st.st_blocks * 512, nz))' \
      "$durable" "$([ "$tier" = 1 ] && echo tier2 || echo tier1)")
  fi
  walrep=$(python3 "$BENCH/lib/decode_wal.py" --summary "$store/$tag/cte_metadata_log" 2>/dev/null)
  # The raw stored bytes are never kept: measured above, dropped here.
  find "$store" -maxdepth 2 -type f \( -name chi_bdev.dat -o -name 'cte_tier.dat*' \
       -o -name 'cte_tier2.dat*' \) -delete 2>/dev/null
  # Both device roots are shared allocations: an arm's images MUST go before
  # the next arm starts, or a full campaign fills the filesystem.
  rm -rf "$t2dir" "$pfsdir" 2>/dev/null
  local failed
  failed=$(grep -hoE "failed: [0-9]+" "$store"/*/stdout.log 2>/dev/null | tail -1)
  echo "     rc=$rc loop=${stage_s:-?}s codec=${codec_s:-?}s compute=${sc_s:-?}s total=${total_s:-?}s (H2D staging ${h2d_s}s removed) ${failed:-} ${bound:-}" >&2
  echo "     tier=$tier async_ms=$flush -> $([ "$tier" = 1 ] \
         && echo "NVMe $t2dir" || echo "PFS $pfsdir") | stored ${stored_b:-?} B | final ${fl#flush: }" >&2
  [ -n "$durep" ] && echo "     $durep" >&2
  echo "     WAL: ${walrep:-(none)}" >&2
  # A failed run or a lost chunk is recorded as TBD.
  local verdict
  verdict=$(grep -hE "^(VERIFIED|FAILED):" "$store"/*/stdout.log 2>/dev/null | tail -1)
  if [ "$rc" -ne 0 ] || echo "$verdict" | grep -q FAILED \
     || { [ -n "$failed" ] && [ "${failed#failed: }" != 0 ]; }; then
    echo "     NOT RECORDED: rc=$rc ${failed:-} ${verdict%%(*} -- see $store/console.log" >&2
    return 0
  fi
  # compute_min,io_min,total_min,std_min,ratio. std_min is left EMPTY: one arm
  # is run once, so there is no spread to report. The column exists because the
  # plot draws an error bar when it is filled -- by a caller that merges repeats
  # -- and dropping it would shift every later column.
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
# THE ARM SET THIS RUN BUILT, beside the results it produced. Panel (a) gained
# a bound and panel (b) gained its +Tier rows between campaigns, and nothing
# recorded which shape a given fig9.csv came from -- so re-plotting an older CSV
# silently dropped every bar whose name the plot did not happen to list.
{
  echo "label,panel,config,eb,tier,async_ms"
  for _spec in "${ARMS[@]}"; do
    IFS='|' read -r _l _p _c _e _t _f <<< "$_spec"
    echo "\"$_l\",$_p,$_c,$_e,$_t,$_f"
  done
} > "$OUT/arms.csv"
unset _spec _l _p _c _e _t _f
cat > "$OUT/run.json" <<JSON
{"workload":"$WLNAME","size":"$SIZE","panel":"$PANEL","arms":${#ARMS[@]},
 "np_config":"$NP_CFG","best_fixed":"$BEST_FIXED","chunk":$CHUNK,"max_files":$MAXF,
 "eb":"$EB_LOW","panel_b_tier":$PANEL_B_TIER,"async_ms":$ASYNC_MS,"ram_mb":${RAM_MB:-0},
 "cost_bw_bytes_per_ms":"$COST_BW","np_lr":$NP_LR,"np_mape":$NP_MAPE,
 "measure_dt":$MEASURE_DT,"measure_quality":$MEASURE_QUALITY,
 "arm_timeout_s":$ARM_TIMEOUT,"fields":"$FIELDS",
 "pfs_root":"$PFS_ROOT","nvme_root":"$NVME_ROOT"}
JSON

echo "panel,workload,strategy,compute_min,io_min,total_min,std_min,ratio,eb" > "$CSV"

for spec in "${ARMS[@]}"; do
  IFS='|' read -r label panel cfg eb tier flush <<< "$spec"
  arm_selected "$label" "$panel" || continue

  split=$(run_arm "$label" "$cfg" "$eb" "$tier" "$flush")
  echo "$panel,$WLNAME,\"$label\",${split:-,,,,},$eb" >> "$CSV"
done

# TBD rows for the other workloads keep the plot layout fixed.
for w in "${WORKLOADS_ALL[@]}"; do
  [ "$w" = "$WLNAME" ] && continue
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r label panel _ eb _ _ <<< "$spec"
    [ "$PANEL" = both ] || [ "$PANEL" = "$panel" ] || continue
    echo "$panel,$w,\"$label\",,,,,,$eb" >> "$CSV"
  done
done

echo; echo "csv: $CSV"
[ "$DRY" = 1 ] && exit 0
python3 "$HERE/plot_fig9.py" --csv "$CSV" --out "$OUT/figures"
