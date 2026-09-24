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
# FlushData (CLIO_REPLAY_FINAL_FLUSH) AND a timed fdatasync of its tier files
# (CLIO_REPLAY_FSYNC), so every bar ends at the same guarantee: bytes on the
# device. Without it a tier arm on the node's xfs was timed against the page
# cache -- its whole output stayed dirty past the timer -- while a PFS arm's
# bytes left inside the loop, which flattered every +Tier bar. The write-time
# decompression diagnostics (MEASURE_DT, MEASURE_QUALITY) are OFF by default:
# both decompress inside the timed loop, and this figure measures a write path.
#
# THE SPLIT: light = ELAPSED device I/O, measured by the bdev transports
# themselves (CLIO_IO_LOG; see modules/bdev/include/clio_runtime/bdev/io_log.h)
# as the union of the per-write intervals -- the same span upstream's VOL calls
# I/O, from the device-to-host copy through the write. Solid = total minus it.
# The INPUT READ and H2D staging are in NEITHER -- the
# figure assumes the data is already in memory and on the GPU, as upstream's
# VOL write does, so the driver's read time and the ELAPSED staging (a union
# of the per-chunk intervals, not their sum) come off the total first. Inputs
# are copied once to node-local storage (stage_inputs) so every arm reads alike. Every workload replays .f32 dumps, LAMMPS
# included, so all five are measured the same way.
#
# `io_min` IS NOT AN I/O MEASUREMENT. It is total - loop, and for an untiered
# arm the durable write happens inside the loop. A true compute/IO split is not
# derivable from this instrumentation; fig9.md beside this script records why, and the two
# alternative splits that were tried and rejected.
#
# Environment: ONLY (same as --only), DUMP_ROOT, PFS_ROOT, NVME_ROOT, ALLOW_NETWORK_TIER2, EB_LOW, PANEL_B_TIER, NP_CFG, SMOKE_GB, MAXF, RAM_PCT/RAM_MB,
# COST_BW, NP_LR, NP_MAPE, MEASURE_DT, MEASURE_QUALITY, ARM_TIMEOUT, BEST_FIXED, WORST_FIXED,
# SELECTION_LOG (default 0), STAGE_INPUT (default 1), STAGE_ROOT, STAGE_STREAMS, EXCLUDE_READ (default 1),
# REPS (runs per arm, default 3), TBD_OTHERS (default 0).
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
# CUSZ_REUSE=1 lets the cuSZ wrapper keep ONE psz_resource per thread instead of
# building one per chunk -- 20.8 ms a chunk at 8 MiB, which was 81% of the cuSZ
# arm's bar. IT IS ONLY CORRECT AGAINST A cuSZ THAT ZEROES ITS OUTLIER COUNTER
# PER COMPRESSION. Upstream e1c0135 resets it only in the Spline branch, so a
# reused manager there emits the previous chunk's outliers and decodes to
# garbage -- SILENTLY, with the codec reporting success (blow-ups REUSE 8/10 vs
# FRESH 0/10, max|err| 199.579 against eb 1e-3, job 22323800). Default 0, and
# the value is recorded in run.json so a bar can be traced to the cuSZ it ran
# against. See compress/cusz.h ReuseManagers.
# UNSET BY DEFAULT, which is not the same as 0: the wrapper decides for itself
# by running a one-shot self-test against whichever cuSZ it linked, and takes
# the fast path only if reuse reproduces a fresh manager's output. Forcing 0
# here used to short-circuit that test before it ran -- every arm built a
# manager per chunk and the guard printed nothing, because =0 returns before
# the test (job 22324612). Set CUSZ_REUSE=0 to pin the per-chunk path for an
# A/B, or 1 to demand reuse.
CUSZ_REUSE=${CUSZ_REUSE:-}
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
#   /work/hdd  HDD-backed Lustre (pool ddn_hdd) -- the PFS
#   /work/nvme SSD-backed Lustre (pool ddn_ssd): NVMe media, but over the
#              network. NOT used for tier 2 any more.
# TIER 2 IS THE NODE'S OWN NVMe. On a Delta compute node /tmp is a local NVMe
# drive, private to the job: nvme0n1 (MZXL51T6HBJR), xfs, 1.5 TB, no quota,
# 1.2 GB/s durable write (3 x 2 GiB, conv=fdatasync, job 22274732) against
# 434-502 MB/s for /work/nvme. It is wiped when the job ends, which costs
# nothing here: every tier image is deleted as soon as its arm is measured, and
# the results live under --out. require_local_tier2 below refuses a network
# filesystem as tier 2, so "+Tier" cannot silently mean NVMe over the network.
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
NVME_ROOT=${NVME_ROOT:-${TIER2_ROOT:-/tmp/fig9-nvme-${SLURM_JOB_ID:-$_WHO}}}
PANEL=both DRY=0 OUT="" FIELDS="" ONLY=${ONLY:-}
BEST_FIXED=${BEST_FIXED:-}     # default per workload, below
WORST_FIXED=${WORST_FIXED:-}   # default per workload, below
ASYNC_MS=${ASYNC_MS:-500}      # periodic flush for the +Async arms
# RAM tier 1, committed at runtime start: run_arm sets CLIO_PREFAULT=0 for a
# tiered arm, which faults the whole mapping in during setup (untimed). Without
# it the tier faulted in 64 MiB at a time INSIDE the timed loop, 40 ms each
# (1.7 s of full Nyx's nvCOMP+Tier loop). Sized as RAM_PCT of the payload
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
# = frozen weights, no feedback at all. EVERY NeuroPress arm runs this config;
# none is pinned to `dynamic` (an older comment here said two were -- arms.csv
# of every run since shows none).
#   Learning needs chunks: it moved Nyx's achieved ratio 9.78 -> 10.6 over 2000
#   chunks (83% of the way to the best hindsight-fixed action) but is pure noise
#   over 240, so run this with MAXF large enough.
NP_CFG=${NP_CFG:-learn}

usage() { sed -n '3,9p' "$0" >&2; cat >&2 <<'U'

  --workload, -w  nyx | vpic | warpx | lammps | ai   (default nyx)
  --size, -s      smoke | full                       (default smoke)
                  smoke replays SMOKE_GB (default 4) GiB per workload;
                  full replays the whole dump. MAXF overrides either.
  --panel         a | b | both                       (default both)
  --out DIR       results directory
  --fields DIR    dump directory (default: per workload, from the environment)
  --only ARMS     run only these arms: a comma-separated list of exact labels
                  or result-dir tags, e.g. --only "Baseline,NP+Tier,best_fixed_nvcomp"
                  (repeatable; env ONLY does the same). --dry-run lists them.
                  An existing fig9.csv in --out is kept as fig9.csv.<time>.bak
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
    --only)        ONLY=${ONLY:+$ONLY,}$2; shift 2 ;;
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
# BEST AND WORST FIXED nvCOMP ACTION, per workload: the fastest and slowest
# median wall clock over all 32 fixed actions (8 nvCOMP libs x -q x -s4) at
# eb=1e-3, 3 interleaved reps each, every one bound-checked.
#   nyx   2026-09-23, nyx-i96 plt00005/13/21 (18 files, 1.13 GiB), local NVMe:
#         best  static-bitcomp-q  1.056 s (8.09x); runner-up static-ans-q-s4
#               1.094 s -- within rep noise of it
#         worst static-deflate    4.454 s (4.40x)
#         At eb=0 (panel a) -q is inert, so best/worst there are static-bitcomp
#         (1.715 s) and static-deflate -- the same libraries, no separate pin.
#   vpic  2026-09-23, vpic-i16, all 16 fields spread over the run (144 files,
#         1.13 GiB), local NVMe; stopped after 53 of 96 runs (every config has
#         rep 1, the top and bottom ones rep 2):
#         best  static-ans-q-s4   1.588 s (6.73x) -- the earlier oracle pick;
#               static-ans-q 1.661 s and static-bitcomp-q-s4 1.665 s close behind
#         worst static-deflate-s4 13.375 s (1.29x); static-zstd-s4 12.406 s next
#         At eb=0 the best NON-quantized action is static-bitcomp (4.77 s), not
#         static-ans-s4 (4.94 s), which panel (a) runs -- within noise, not pinned.
#   lammps 2026-09-23, lammps (site.sh default), position/velocity/force over the
#         run (408 files, 1.14 GiB), local NVMe; stopped after 73 of 96 runs
#         (every config rep 1, top and bottom rep 2):
#         best  static-ans-q-s4   4.328 s (1.99x); static-bitcomp-q-s4 4.468 s
#               (the earlier pick) and static-bitcomp-q 4.655 s close behind
#         worst static-deflate-s4 18.955 s (1.20x); static-deflate 18.163 s next
#         At eb=0 the best non-quantized action is static-ans-s4 -- exactly what
#         panel (a) runs.
#   warpx 2026-09-23, warpx-i5, all 10 fields spread over the run (150 files,
#         1.17 GiB), local NVMe; stopped at 48 of 96 runs (every config rep 1):
#         best  static-ans-q-s4     3.581 s (1.86x); static-bitcomp-q-s4 3.763 s
#               (the earlier pick) next
#         worst static-deflate-q-s4 10.791 s (2.24x) -- a tie with
#               static-deflate-s4 10.771 s on one rep each
#         At eb=0 the best non-quantized action is static-ans-s4, what panel (a)
#         runs.
#   ai    2026-09-23, ai, one file per field (weights/gradients/adam_m/adam_v,
#         1.28 GiB), local NVMe; LOSSLESS ONLY, so the 16 actions without -q at
#         eb=0, each round-tripped bit-exact; stopped at 24 of 48 runs:
#         best  static-bitcomp-s4 3.988 s (1.03x); static-ans-s4 4.055 s and
#               static-cascaded 4.088 s within rep noise of it
#         worst static-deflate-s4 13.653 s (1.17x)
# Every workload is swept; the fallback below only covers a new one.
if [ -z "$BEST_FIXED" ]; then
  case "$WL" in
    nyx)  BEST_FIXED=static-bitcomp-q ;;
    vpic|lammps|warpx) BEST_FIXED=static-ans-q-s4 ;;
    ai)   BEST_FIXED=static-bitcomp-s4 ;;
    *)    BEST_FIXED=static-bitcomp-q-s4 ;;
  esac
fi
if [ -z "$WORST_FIXED" ]; then
  case "$WL" in
    nyx)  WORST_FIXED=static-deflate ;;
    vpic|lammps|ai) WORST_FIXED=static-deflate-s4 ;;
    warpx)       WORST_FIXED=static-deflate-q-s4 ;;
  esac
fi
OUT=${OUT:-$BENCH/results/figure9/$WL-$SIZE}
mkdir -p "$OUT"
CSV="$OUT/fig9.csv"

# ONE PLACE FOR EVERY WORKLOAD'S DUMPS: $DUMP_ROOT/<workload>/fields. The five
# dumps used to live under names that described the figure they were first
# generated for, not the workload they feed -- `np-fig8-nyx` and `np-fig8-vpic`
# WERE figure 9's Nyx and VPIC inputs, and on 2026-09-20 both were deleted as
# "figure 8 data" during a cleanup. A single root named by workload removes
# that trap. Derived from the account, like PFS_ROOT, so nothing is pinned to
# one user's home.
DUMP_ROOT=${DUMP_ROOT:-/work/hdd/$CLIO_ACCT/$_WHO/np-dumps}
if [ -z "$FIELDS" ]; then
  # A per-workload override wins; then the consolidated root; then a dump this
  # checkout generated in-tree with <workload>/gen_fields.sh.
  case "$WL" in
    nyx)    FIELDS=${NYX_FIELDS:-} ;;
    vpic)   FIELDS=${VPIC_FIELDS:-} ;;
    warpx)  FIELDS=${WARPX_FIELDS:-} ;;
    ai)     FIELDS=${AI_FIELDS:-} ;;
    # LAMMPS is replayed from a dump like the other four, so every workload is
    # measured the same way: a real input read, and no simulate time in the bar.
    # The dump is the driver's own --raw output, byte-identical to what the
    # in-situ path hands the compressor (job 22190101, 3933 blobs verified).
    lammps) FIELDS=${LAMMPS_FIELDS:-} ;;
  esac
  [ -n "$FIELDS" ] || FIELDS=$DUMP_ROOT/$WL/fields
  [ -d "$FIELDS" ] || FIELDS=$BENCH/$WL/fields
fi
# ONE metadata scan for the whole run, reused by the size budget, the payload
# and warm_cache. These dumps sit on Lustre, where listing 2,010 files across
# nested step directories takes minutes, and three separate scans paid that
# before the first arm ran. Sorted by path: the order the driver replays in
# (neuropress_field_replay.cc:452-453), so every consumer sees the same set.
FILE_INDEX=$(mktemp "${TMPDIR:-/tmp}/fig9-files-XXXXXX") || exit 3

# THE TIER IMAGE OF THE ARM RUNNING RIGHT NOW. run_arm sets these when it
# composes the arm's device paths and clears them once it has removed the
# images itself; anything still set is an arm that a kill interrupted.
#
# `scancel` sends TERM, the TERM handler exits, and exiting runs the EXIT trap,
# so an interrupted arm's image goes the same way a finished arm's does. Without
# this a cancelled job left behind a tier the size of the payload -- 26 GiB for
# an untiered Nyx arm -- and three such leaks exhausted the project quota once
# already, which then failed every later arm with "Disk quota exceeded".
# Only THIS job's arm dirs are touched: $PFS_ROOT is shared between concurrent
# per-arm jobs, so removing the whole root here would delete a sibling's tier.
ARM_T2DIR=""; ARM_PFSDIR=""; ARM_STORE=""
clean_arm_images() {
  [ -n "$ARM_T2DIR$ARM_PFSDIR$ARM_STORE" ] || return 0
  rm -rf $ARM_T2DIR $ARM_PFSDIR 2>/dev/null
  # The bdev image under the run store goes too: it is raw payload bytes, the
  # same ones the two device roots hold. $store/<tag>/chi_bdev.dat, hence
  # depth 2 -- the same find the finished and timed-out paths run.
  [ -n "$ARM_STORE" ] && find "$ARM_STORE" -maxdepth 2 -type f \
       \( -name chi_bdev.dat -o -name 'cte_tier.dat*' \
          -o -name 'cte_tier2.dat*' \) -delete 2>/dev/null
  ARM_T2DIR=""; ARM_PFSDIR=""; ARM_STORE=""
  return 0
}
trap 'clean_arm_images; rm -f "$FILE_INDEX"' EXIT
trap 'exit 143' TERM INT
find "$FIELDS" -name '*.f32' -printf '%p\t%s\n' 2>/dev/null | sort > "$FILE_INDEX"
if [ ! -s "$FILE_INDEX" ]; then
  echo "no field dumps at $FIELDS -- pass --fields DIR or run $WL/gen_fields.sh" >&2
  exit 3
fi

# Replay hands the compressor host memory, which it refuses unless staged.
export CLIO_NEUROPRESS_STAGE_H2D=1
# NO SELECTION LOG IN A TIMED ARM. It hashes every input chunk and every
# compressed payload byte by byte inside the compressor, ~19 ms per 8 MiB chunk
# that only the compressed arms pay (Baseline never enters the compressor). With
# it on, full Nyx's nvCOMP loop carried 54-65 s of hashing, more than its whole
# gap to Baseline (jobs 22300816, 22300826). Nothing here reads selection.csv.
# SELECTION_LOG=1 turns it back on for a run that wants the per-chunk choices.
export SELECTION_LOG=${SELECTION_LOG:-0}

case "$WL" in nyx|vpic) DRIVER=$BENCH/$WL/run_config.sh ;; *) DRIVER=$BENCH/nyx/run_config.sh ;; esac

# files_for_budget <index> <GiB> -- how many files that budget covers, counted in
# the order the driver replays them. The budget is filled in WHOLE files, so a
# workload whose files are large overshoots it: AI's 327 MiB tensors make 4 GiB
# into 4255 MiB. A dump smaller than the budget replays whole, and the answer is
# never 0.
#   @param 1 file index: "path<TAB>bytes" per line, sorted by path
#   @param 2 payload budget in GiB
#   @return file count, on stdout
files_for_budget() {
  awk -F'\t' -v g="$2" '
      BEGIN { b = g * 1073741824 }
      { s += $2; if (s >= b) { print NR; done = 1; exit } }
      END { if (!done) print (NR ? NR : 1) }' "$1"
}
# SMOKE IS A BYTE BUDGET, NOT A FILE COUNT. The five dumps differ 40x in file
# size -- 8 MiB for VPIC/Nyx/WarpX/LAMMPS against a whole 327 MiB tensor for AI
# -- so one --max-files gave each workload a different payload (480 MiB against
# 655 MiB at 60 files) and AI needed a hardcoded override of its own. Setting
# the payload instead makes the five comparable and needs no per-workload case.
SMOKE_GB=${SMOKE_GB:-4}
# ARM_TIMEOUT (s): at 480 MiB the slowest arm measured was 8.3 s (cuSZ on WarpX,
# 2026-09-20), so a 4 GiB arm is ~70 s and the codec-bound ones rather more;
# 300 leaves room without letting a hung arm hold the allocation. A full 30 GB
# replay arm is ~6 min.
# CHUNK: the unit every arm compresses and puts. Never below MIN_CHUNK (4 MiB):
# at 2 MiB each compressed chunk paid its ~3 ms of fixed runtime cost (task
# hand-off, GPU->host staging, rescheduling) twice as often per byte, and on
# the 8 MiB dumps only a file's own chunks overlap, so small chunks penalised
# every compressed arm against Baseline for reasons unrelated to compression.
MIN_CHUNK=4194304
case "$SIZE" in
  smoke) CHUNK=4194304
         MAXF=${MAXF:-$(files_for_budget "$FILE_INDEX" "$SMOKE_GB")}
         ARM_TIMEOUT=${ARM_TIMEOUT:-300} ;;
  full)  CHUNK=8388608; MAXF=${MAXF:-0}; ARM_TIMEOUT=${ARM_TIMEOUT:-1800} ;;
esac
if [ "$CHUNK" -lt "$MIN_CHUNK" ]; then
  echo "chunk $CHUNK B is below the $MIN_CHUNK B minimum unit" >&2
  exit 2
fi
# ---- payload this run will replay, and the RAM tier as a percentage of it ----
# ALWAYS measured, even when RAM_MB is set by hand: every fig9.csv row carries
# it, so a plot can say how much data its bars moved, and a merged CSV cannot
# silently put a 4 GiB bar beside a 26 GiB one.
# The first MAXF of the index, which is already in the driver's replay order --
# sizing from the MAXF LARGEST files would measure a set it never replays
# whenever the dumps differ in size.
PAYLOAD_MB=$(awk -F'\t' -v n="$MAXF" \
  'n == 0 || NR <= n { s += $2 } END { printf "%.0f", s / 1048576 }' "$FILE_INDEX")
if [ -z "${RAM_MB:-}" ]; then
  RAM_MB=$(awk -v p="${PAYLOAD_MB:-0}" -v pc="$RAM_PCT" \
    'BEGIN { v = p * pc / 100.0; if (v < 64) v = 64; printf "%.0f", v }')
  echo "== payload ${PAYLOAD_MB:-?} MiB -> RAM tier ${RAM_MB} MiB (${RAM_PCT}%)"
else
  echo "== payload ${PAYLOAD_MB:-?} MiB -> RAM tier ${RAM_MB} MiB (RAM_MB set)"
fi

# arm  ::  label | panel | config | eb | tier(0/1) | async_ms
#          async_ms 0 = periodic flush disabled entirely
# The label is the plot's key, so an arm that is renamed here is a new bar
# there. arms.csv records the set this run built, so a CSV is never ambiguous
# about which shape it came from.
#
# LOSSLESS-ONLY WORKLOADS. An AI checkpoint is model weights, not a simulation
# field: there is no error budget to spend, so the figure offers it no lossy
# point at all. Its panel (a) ladder stops before the lossy rung and its panel
# (b) runs at eb=0. The error-bounded codecs are not merely run at eb=0 there,
# they are LEFT OUT: cuSZ and cuSZp3 have no lossless mode, so a lossless arm
# of either is not the codec the name promises.
LOSSLESS_ONLY_WORKLOADS=${LOSSLESS_ONLY_WORKLOADS:-AI}
LOSSY_ONLY_CODECS=${LOSSY_ONLY_CODECS:-"cuSZ cuSZp3"}

# lossless_only <workload display name> -- true when that workload runs no
# lossy arm. Keyed by display name (AI, Nyx, ...), because the TBD rows this
# run writes for the OTHER four workloads have to honour their rules too.
#   @param 1 workload display name
#   @return 0 when lossless-only, 1 otherwise
lossless_only() {
  case " $LOSSLESS_ONLY_WORKLOADS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

# build_arms <workload display name> -- echo that workload's arm set, one spec
# per line. A function rather than one global list because the arm set is now a
# property of the workload: an AI run must emit ITS arms, and the TBD rows it
# writes for Nyx must emit Nyx's, or a merged CSV would show AI a column of
# cuSZ bars marked TBD that were never going to be run.
#   @param 1 workload display name
#   @return arm specs on stdout: label|panel|config|eb|tier|async_ms
build_arms() {
  local w=$1 eb_b=$EB_LOW lossless=0 b bl bc
  lossless_only "$w" && { lossless=1; eb_b=0; }
  # PANEL (a): the ablation ladder. Untiered arms write straight to the PFS;
  # every +Tier arm spills to the node's own NVMe (NVME_ROOT).
  echo "Baseline|a|baseline|0|0|0"
  # NO PANEL (a) nvCOMP RUNGS (2026-09-23). They ran $BEST_FIXED at eb=0 -- the
  # same library panel (b) runs at the bound -- so the figure drew the winning
  # nvCOMP codec twice under two names. Panel (b)'s `Best fixed nvCOMP` and
  # `Worst fixed nvCOMP` are now the only nvCOMP arms, and they are compared
  # against each other at ONE bound, which is the comparison worth making.
  #
  # WHAT THIS COSTS: panel (a)'s ladder loses its lossless external-codec
  # comparator, so plot_fig9.py's `NP only vs nvCOMP` and `NP+Tier vs
  # nvCOMP+Tier` reductions go empty. They are NOT repointed at `Best fixed
  # nvCOMP`: that arm runs at eb=1e-3 while the NP rungs here run lossless, so
  # the comparison would be across different bounds.
  echo "NP only|a|$NP_CFG|0|0|0"
  echo "NP+Tier|a|$NP_CFG|0|1|0"
  echo "NP+Tier+Async|a|$NP_CFG|0|1|$ASYNC_MS"
  # ...and the lossy end of that ladder, which a lossless-only workload skips.
  [ "$lossless" = 1 ] || echo "NP+Tier+Async+Lossy|a|$NP_CFG|$EB_LOW|1|$ASYNC_MS"
  # PANEL (b): the external codecs and NeuroPress at the SAME bound. With
  # PANEL_B_TIER=1 each runs twice so the tier is the only variable: the PFS row
  # writes each chunk through to Lustre inside the timed loop, the +Tier row puts
  # to RAM and drains to NVMe behind the periodic flush.
  # On a lossless-only workload this panel runs at eb=0, so `Best fixed nvCOMP`
  # repeats panel (a)'s `nvCOMP` and `NeuroPress` repeats `NP only`. That is the
  # same measurement drawn in both panels, not a second run of a different arm.
  for b in "Best fixed nvCOMP:$BEST_FIXED" \
           ${WORST_FIXED:+"Worst fixed nvCOMP:$WORST_FIXED"} "ndzip:static-ndzip" \
           "cuSZp3:static-cuszp" "cuSZ:static-cusz" "NeuroPress:$NP_CFG"; do
    bl=${b%%:*}; bc=${b#*:}
    if [ "$lossless" = 1 ]; then
      case " $LOSSY_ONLY_CODECS " in *" $bl "*) continue ;; esac
    fi
    echo "$bl|b|$bc|$eb_b|0|0"
    # NeuroPress+Tier would be BYTE-IDENTICAL to panel (a)'s NP+Tier+Async+Lossy
    # -- same config, same bound, same tier, same flush (arms.csv proves it) --
    # so it drew one measurement twice. It measured 0.311 min against that arm's
    # 0.113 min in the same campaign, each run once, which is how the duplicate
    # was noticed. Panel (a)'s rung is the one kept.
    [ "$bl" = NeuroPress ] && continue
    [ "$PANEL_B_TIER" = 1 ] && echo "$bl+Tier|b|$bc|$eb_b|1|$ASYNC_MS"
  done
}
mapfile -t ARMS < <(build_arms "$WLNAME")

# slug <label> -- the arm's result-dir tag: "NP+Tier+Async" -> np_tier_async.
slug() { echo "$1" | tr 'A-Z ()+' 'a-z___' | tr -s '_' | sed 's/_$//'; }

# only_names -- the --only entries, one per line, surrounding spaces trimmed.
only_names() {
  tr ',' '\n' <<< "$ONLY" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -v '^$'
}

# arm_selected <label> <panel> -- does --panel / --only let this arm run?
# --only matches an arm by its exact label or by its tag (see slug).
# One definition, used by the preflight below and by the run loop, so the
# preflight cannot demand a resource that no selected arm will touch.
arm_selected() {
  { [ "$PANEL" = both ] || [ "$PANEL" = "$2" ]; } || return 1
  [ -z "$ONLY" ] && return 0
  only_names | grep -qxF -e "$1" -e "$(slug "$1")"
}

# A misspelt --only entry would otherwise run nothing and exit clean.
if [ -n "$ONLY" ]; then
  _known=$(for _spec in "${ARMS[@]}"; do _l=${_spec%%|*}; echo "$_l"; slug "$_l"; done)
  while IFS= read -r _n; do
    grep -qxF -e "$_n" <<< "$_known" && continue
    echo "--only: no $WLNAME arm is called '$_n'. Arms (label / tag):" >&2
    for _spec in "${ARMS[@]}"; do _l=${_spec%%|*}; echo "  $_l / $(slug "$_l")" >&2; done
    exit 2
  done < <(only_names)
  unset _known _n _spec _l
fi

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
  echo "== $1 root: $4  [$(df -Th "$4" 2>/dev/null | tail -1 | awk '{print $2" on "$1}')]"
}
# require_local_tier2 <root> -- tier 2 must be a device IN this node. A network
# filesystem there turns "+Tier" into "NVMe over the network", which the figure
# does not want to measure; refuse it unless ALLOW_NETWORK_TIER2=1 says so.
require_local_tier2() {
  [ "$_want_nvme" = 1 ] || return 0
  local fs; fs=$(df -T "$1" 2>/dev/null | awk 'NR==2{print $2}')
  case "$fs" in
    lustre|nfs|nfs4|gpfs|beegfs|cifs|smb3|ceph|panfs|fuse.*)
      [ "${ALLOW_NETWORK_TIER2:-0}" = 1 ] && return 0
      echo "NVME_ROOT $1 is on $fs: a network filesystem, not this node's NVMe." >&2
      echo "  Tier 2 must be node-local (on a Delta compute node: /tmp)." >&2
      echo "  Set ALLOW_NETWORK_TIER2=1 to override on purpose." >&2
      exit 4 ;;
  esac
}
if [ "$DRY" != 1 ]; then
  require_root PFS  PFS_ROOT  "$_want_pfs"  "$PFS_ROOT" \
    "untiered arm (Baseline included) puts its tier-1 file"
  require_root NVMe NVME_ROOT "$_want_nvme" "$NVME_ROOT" \
    "+Tier arm spills tier 2"
  require_local_tier2 "$NVME_ROOT"
fi

# STAGE THE INPUTS ONCE, TO THIS NODE. Replayed from Lustre, the input read was
# 62-73% of every full-size bar and grew with arm order (137 -> 310 s on Nyx):
# Lustre drops cached pages whose locks sit idle 600 s, and warm_cache's single
# stream needs ~11 min for 26 GiB, so each later arm re-read more of it cold. In
# situ the data comes from simulation memory, not a file, so none of that is
# what the figure measures. One parallel copy to the node's own NVMe gives every
# arm the same local bytes. STAGE_INPUT=0 replays from $FIELDS as before.
STAGE_INPUT=${STAGE_INPUT:-1}
STAGE_ROOT=${STAGE_ROOT:-/tmp/fig9-input-${SLURM_JOB_ID:-$_WHO}}
STAGE_STREAMS=${STAGE_STREAMS:-16}
STAGED_DIR=""
# stage_inputs -- copy the files this run replays (the first MAXF of the index,
# or all of it) to $STAGE_ROOT/$WL, then point FIELDS and FILE_INDEX there.
# Directories are created before the parallel copy so no two copies race on
# the same mkdir; the copy is checked by file count and total bytes.
#   @return 0 on success; exits 4 on a network or full target, 5 on a short copy
stage_inputs() {
  local dst=$STAGE_ROOT/$WL src=${FIELDS%/} fs n want avail t0 t1 got_n got_b
  mkdir -p "$dst" || { echo "cannot create input stage $dst" >&2; exit 4; }
  fs=$(df -T "$dst" 2>/dev/null | awk 'NR==2{print $2}')
  case "$fs" in lustre|nfs|nfs4|gpfs|beegfs|cifs|smb3|ceph|panfs|fuse.*)
    echo "input stage $dst is on $fs, not node-local storage" >&2; exit 4 ;; esac
  n=$(awk -v m="$MAXF" 'END{print (m > 0 && m < NR) ? m : NR}' "$FILE_INDEX")
  want=$(awk -F'\t' -v n="$n" 'NR <= n { s += $2 } END { printf "%.0f", s }' "$FILE_INDEX")
  avail=$(df -B1 --output=avail "$dst" | tail -1)
  if awk -v a="$avail" -v w="$want" 'BEGIN{exit !(a < w * 1.05)}'; then
    echo "input stage $dst: $avail B free, $want B needed" >&2; exit 4
  fi
  STAGED_DIR=$dst
  t0=$(date +%s.%N)
  awk -F'\t' -v n="$n" -v p="$src/" 'NR <= n { print substr($1, length(p) + 1) }' "$FILE_INDEX" \
    > "$dst/.files"
  sed -n 's#/[^/]*$##p' "$dst/.files" | sort -u | (cd "$dst" && xargs -r -d '\n' mkdir -p)
  (cd "$src" && xargs -r -d '\n' -P "$STAGE_STREAMS" -I{} cp {} "$dst/{}" < "$dst/.files")
  t1=$(date +%s.%N)
  rm -f "$dst/.files"
  FIELDS=$dst
  find "$FIELDS" -name '*.f32' -printf '%p\t%s\n' 2>/dev/null | sort > "$FILE_INDEX"
  got_n=$(wc -l < "$FILE_INDEX")
  got_b=$(awk -F'\t' '{ s += $2 } END { printf "%.0f", s }' "$FILE_INDEX")
  if [ "$got_n" != "$n" ] || [ "$got_b" != "$want" ]; then
    echo "input stage short: $got_n of $n files, $got_b of $want B" >&2; exit 5
  fi
  echo "== staged $n input file(s), $((want / 1048576)) MiB -> $dst [$fs on" \
       "$(df "$dst" | awk 'NR==2{print $1}')] in" \
       "$(awk -v a="$t0" -v b="$t1" -v w="$want" 'BEGIN{printf "%.1f s (%.0f MiB/s)", b-a, w/1048576/(b-a)}')"
}
if [ "$DRY" != 1 ] && [ "$STAGE_INPUT" = 1 ]; then
  # The staged copy is raw input bytes: never leave it behind, even on a kill.
  trap 'clean_arm_images; rm -f "$FILE_INDEX"; [ -n "$STAGED_DIR" ] && rm -rf "$STAGED_DIR"; rmdir "$STAGE_ROOT" 2>/dev/null' EXIT
  trap 'exit 143' TERM INT
  stage_inputs
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


# ELAPSED time covered by a per-chunk phase, in seconds: the UNION of the
# [start, start+dur] intervals, NOT their sum. Chunks pipeline, so a sum
# over-counts -- on AI the per-chunk walls sum to 52-77x the arm's total.
# Falls back to the sum when the log has no start column (an older run).
#
# CLIPPED to the measured window when one is given. An interval is only part of
# the bar to the extent it lies inside the timed region: a write that started
# before the timer opened, or that the log recorded after it closed, otherwise
# contributes time the bar does not contain, and the pale segment could exceed
# the bar it sits in. The driver prints the window in the same steady-clock
# nanoseconds io_log.h stamps with ("window: start_ns N   end_ns M").
# Without w0/w1 the behaviour is exactly as before.
#   phase_union <log> <dur column> <start column> [window_start_ns] [window_end_ns]
phase_union() {
  awk -F, -v dcol="$2" -v scol="$3" -v w0="${4:-0}" -v w1="${5:-0}" '
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; d = col[dcol]; t = col[scol]; next }
    !d { next }
    t && $t != "" && $t + 0 >= 0 && $d + 0 > 0 {
      a = $t + 0; b = $t + $d * 1e6
      if (w1 > 0) {
        if (a < w0) a = w0
        if (b > w1) b = w1
        if (b <= a) next          # wholly outside the measured window
      }
      n++; st[n] = a; en[n] = b
    }
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

# run_arm <label> <config> <eb> <tier> <async_ms> <rep>
#   One measured run of one arm. Rep 1 verifies every blob (and, on a lossy arm,
#   checks the error bound element-wise); later reps skip that untimed read-back.
#   @return on stdout "compute_min,io_min,total_min,ratio,bound" (bound is ok,
#           exceeded or empty), or nothing when the run is not recordable
run_arm() {
  local label=$1 cfg=$2 eb=$3 tier=$4 flush=$5 rep=${6:-1}
  local tag; tag=$(slug "$label")
  local store="$OUT/$tag/r$rep"
  mkdir -p "$store"

  # BENCH_FLUSH_MS is ALWAYS set: 0 is the only way to turn the periodic flush off.
  # The phase log is what makes the host-to-device staging visible per chunk, so
  # it can be taken back out below.
  local -a env_kv=( "BENCH_FLUSH_MS=$flush" "CLIO_REPLAY_FINAL_FLUSH=1"
                    "MEASURE_DT=$MEASURE_DT" "MEASURE_QUALITY=$MEASURE_QUALITY"
                    "CLIO_NEUROPRESS_PHASE_LOG=$store/phases.csv"
                    "CLIO_IO_LOG=$store/io.csv"
                    "CLIO_CUSZ_PHASE_LOG=$store/cusz_setup.csv"
                    )
  # Only pinned when the operator asked; otherwise the wrapper self-tests.
  [ -n "$CUSZ_REUSE" ] && env_kv+=( "CLIO_CUSZ_REUSE_MANAGER=$CUSZ_REUSE" )
  # WHERE THIS ARM'S DURABLE BYTES GO. Untiered -> a tier-1 file on the PFS;
  # tiered -> tier 1 in RAM (a `ram` bdev is shared memory, so its path is only
  # a name and no file is created) and tier 2 on NVMe. One device class per arm
  # class, so `+Tier` is not also a change of filesystem.
  # Keyed by WORKLOAD then arm: the campaign runs one job per workload, and
  # every workload builds the same arm tags, so without the $WL level two
  # concurrent jobs would both write .../fig9-pfs/baseline and corrupt each
  # other's tier.
  local t2dir="$NVME_ROOT/$WL/$tag" pfsdir="$PFS_ROOT/$WL/$tag"
  ARM_T2DIR="$t2dir"; ARM_PFSDIR="$pfsdir"; ARM_STORE="$store"
  if [ "$tier" = 1 ]; then
    mkdir -p "$t2dir" || { echo "cannot create the NVMe tier-2 dir $t2dir" >&2; return 0; }
    env_kv+=( "BENCH_TIER1_TYPE=ram" "BENCH_TIER1_MB=$RAM_MB"
              "BENCH_TIER1_PERSIST=volatile"
              "BENCH_TIER2_PATH=$t2dir/cte_tier2.dat"
              "CLIO_PREFAULT=0"
              "CLIO_REPLAY_FSYNC=$t2dir/cte_tier2.dat" )
  else
    mkdir -p "$pfsdir" || { echo "cannot create the PFS tier-1 dir $pfsdir" >&2; return 0; }
    env_kv+=( "BENCH_TIER1_PATH=$pfsdir/cte_tier.dat"
              "CLIO_REPLAY_FSYNC=$pfsdir/cte_tier.dat" )
  fi
  # `learn` turns on neuropress_online_learning_enabled; it needs a rate. It gets
  # ONE label per chunk -- the measured outcome of the action it executed -- so it
  # cannot see an action it never picks. `explore*` is what adds K alternatives.
  case "$cfg" in learn*|explore*) env_kv+=( "BENCH_NP_LR=$NP_LR" "BENCH_NP_MAPE=$NP_MAPE" ) ;; esac

  local -a cmd=( "$DRIVER" "$cfg" --eb "$eb" --bw "$COST_BW"
                 --results "$store" --tag "$tag" --chunk "$CHUNK" )
  cmd+=( --fields "$FIELDS" )
  [ "$MAXF" -gt 0 ] 2>/dev/null && cmd+=( --max-files "$MAXF" )
  # Verification is UNTIMED -- the driver prints `time: ... total` from
  # now() - t_work and only then calls verify_records()/report_bound(), and the
  # `window: start_ns/end_ns` that io_s is clipped to closes at the same
  # instant. So no bar moves whether this runs or not; it costs job wall clock
  # (a full read-back) and nothing else. It runs once per arm, in rep 1.
  #
  # NO LOSSY ARM IS BOUND-CHECKED (decision 2026-09-23). Every arm with eb > 0
  # runs --no-verify: the external codecs (cuSZ, cuSZp3, nvCOMP -q, ndzip) AND
  # NeuroPress's own lossy rungs alike. This figure reports END-TO-END WALL
  # CLOCK at a REQUESTED bound. Whether a codec honours that bound is a
  # different question, measured by the accuracy figures against the same
  # dumps; answering it here costs a full read-back per arm and, when it fails,
  # turns a timing campaign into an accuracy triage.
  #
  # DROPPING --check-bound IS NOT ENOUGH ON ITS OWN. With no verification flag
  # the driver falls back to a digest comparison of the decoded bytes, which
  # lossy data fails BY CONSTRUCTION (neuropress_field_replay.cc, "Which check
  # applies is decided by the error bound"). So the read-back is skipped
  # outright and the `bound` column is left empty -- an empty column here means
  # NOT CHECKED, never "checked and fine".
  #
  # A LOSSLESS arm at rep 1 is still verified in full, bit-exact against the
  # input, because that check tests reconstruction rather than a bound.
  if [ "$rep" -gt 1 ] || awk -v e="$eb" 'BEGIN{exit !(e + 0 > 0)}'; then
    cmd+=( --no-verify )
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
    # THE TIER IMAGE TOO. This path used to return without reaching the cleanup
    # further down, so a killed arm leaked a tier the size of the payload:
    # three timed-out LAMMPS arms left 65 GB behind and exhausted the project
    # quota, which then failed every later arm with "Disk quota exceeded".
    # rc=137 is SIGKILL -- both what `timeout -k` sends and what the OOM killer
    # sends -- so this is the likeliest arm of all to leak.
    rm -rf "$t2dir" "$pfsdir" 2>/dev/null
    ARM_T2DIR=""; ARM_PFSDIR=""; ARM_STORE=""
    find "$store" -maxdepth 2 -type f \( -name chi_bdev.dat \
         -o -name 'cte_tier.dat*' -o -name 'cte_tier2.dat*' \) -delete 2>/dev/null
    echo "     after clean: $(node_state)" >&2
    return 0
  fi
  # "time: read A s   stage+compress B s   total C s"
  local tline stage_s sc_s="" total_s
  tline=$(grep -hE "time: read" "$store"/*/stdout.log 2>/dev/null | tail -1)
  # The timed region in steady-clock ns, for clipping the I/O and H2D unions.
  # Empty for a run from before the driver printed it; phase_union then behaves
  # as it always did.
  local wline win0="" win1=""
  wline=$(grep -hE "^  window: start_ns" "$store"/*/stdout.log 2>/dev/null | tail -1)
  if [ -n "$wline" ]; then
    win0=$(echo "$wline" | grep -oE "start_ns[[:space:]]+[0-9]+" | grep -oE "[0-9]+$")
    win1=$(echo "$wline" | grep -oE "end_ns[[:space:]]+[0-9]+"   | grep -oE "[0-9]+$")
  fi
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
    h2d_s=$(phase_union "$plog" h2d_ms h2d_start_ns "$win0" "$win1")
    local sub; sub=$(awk -v a="${stage_s:-0}" -v t="${total_s:-0}" -v h="${h2d_s:-0}" \
      'BEGIN { print (h > 0 && a - h > 0 && t - h > 0) ? 1 : 0 }')
    if [ "$sub" = 1 ]; then
      stage_s=$(awk -v a="$stage_s" -v h="$h2d_s" 'BEGIN{printf "%.3f", a - h}')
      total_s=$(awk -v t="$total_s" -v h="$h2d_s" 'BEGIN{printf "%.3f", t - h}')
    fi
  fi
  # BASELINE'S OWN H2D. Every compressed arm's host->device staging is logged
  # per chunk to phases.csv and removed by the union above. Baseline has no
  # phase log -- it never went through the compressor -- so its staging is
  # reported by the driver on its own line instead, and removed here. Without
  # this the arm that was FIXED to be device-resident would be charged for the
  # upload, which is a replay artifact either way: an in-situ producer already
  # holds the data on the GPU. The D2H on the way OUT stays in the bar, which
  # is the whole point of the fix.
  local bh2d
  bh2d=$(grep -hoE "time: h2d [0-9.]+ s" "$store"/*/stdout.log 2>/dev/null \
         | tail -1 | grep -oE "[0-9.]+" | head -1)
  if [ -n "$bh2d" ] && [ -n "$total_s" ]; then
    local ok; ok=$(awk -v t="$total_s" -v h="$bh2d" 'BEGIN{print (h>0 && t-h>0)?1:0}')
    if [ "$ok" = 1 ]; then
      total_s=$(awk -v t="$total_s" -v h="$bh2d" 'BEGIN{printf "%.3f", t - h}')
      stage_s=$(awk -v a="${stage_s:-0}" -v h="$bh2d" 'BEGIN{printf "%.3f", a - h}')
      echo "     baseline H2D ${bh2d}s removed (replay artifact; the bdev's D2H stays in)" >&2
    fi
  fi
  # THE INPUT READ IS NOT PART OF THE BAR, for the same reason as H2D above: in
  # situ the chunk is already in memory. It is identical work in every arm,
  # runs before the chunk's writes, and on Lustre drifted with arm order by more
  # than any codec difference. The driver times it separately (read_s), so take
  # it out; the pale segment is then the final flush plus per-file slack.
  # EXCLUDE_READ=0 keeps it in, as campaigns before 2026-09-22 did.
  local read_s
  read_s=$(echo "$tline" | grep -oE "read[[:space:]]+[0-9.]+" | head -1 | grep -oE "[0-9.]+$")
  if [ "${EXCLUDE_READ:-1}" = 1 ] && [ -n "$read_s" ] && [ -n "$total_s" ] &&
     awk -v t="$total_s" -v r="$read_s" 'BEGIN{exit !(t - r > 0)}'; then
    total_s=$(awk -v t="$total_s" -v r="$read_s" 'BEGIN{printf "%.3f", t - r}')
  else
    read_s=0
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
  # CUSZ'S PER-CHUNK RESOURCE MANAGER, measured where it is paid (cusz.h's
  # SetupLog, CLIO_CUSZ_PHASE_LOG). cuSZ's rev1 C API has no reset and no
  # documented reuse contract, so the wrapper builds a stream and a
  # psz_resource -- histogram, Huffman book, quantisation buffers -- and tears
  # both down again for EVERY chunk. That is library construction, not
  # compression and not I/O: on full Nyx it was 82% of the arm against a
  # 0.44 ms codec kernel, and caching it corrupted 52 of 1024 chunks
  # (cusz.h:103, REJECTED). Recorded as its own column so the figure can show
  # the codec's cost with or without it, never by subtracting an unattributed
  # residue.
  #
  # stream + mgr + release ONLY. copy_ms is the compressed frame leaving the
  # manager's internal device buffer -- real output work every codec does --
  # so it stays in the bar.
  #
  # THE UNION, CLIPPED TO THE WINDOW -- not a sum. This was a sum on the
  # argument that these are serial host calls, which is true per thread and
  # false across threads: chunks compress on several workers at once, and on a
  # 1024-chunk smoke the sum came to 14.307 s inside a 13.539 s arm, a cost
  # larger than the bar holding it. The codec now stamps each span
  # (cusz.h SetupLog), so this is elapsed time and can never exceed the bar.
  #
  # Every arm sets CLIO_CUSZ_PHASE_LOG; only a run that actually called cuSZ
  # writes the file, which is what makes an arm that merely SELECTS cuSZ for
  # some chunks (any NeuroPress arm) get credited for exactly those chunks.
  local setup_s=0 slog
  slog=$(ls -1 "$store"/cusz_setup.csv "$store"/*/cusz_setup.csv 2>/dev/null | head -1)
  if [ -n "$slog" ]; then
    setup_s=$(phase_union "$slog" ms start_ns "$win0" "$win1")
  fi
  # THE SPLIT, from the bdev's own write timers (CLIO_IO_LOG, io_log.h): pale
  # = ELAPSED device I/O, the union of the per-write intervals, which is what
  # upstream's VOL brackets (d2h copy + queue wait + drain). Solid = total minus
  # that. The union, never the sum: writes overlap, and the phase log's io_ms
  # is the await latency of the put, not a transfer.
  # CAVEAT: codec_s does not filter path=="write", so the codec= printed
  # below also carries verification-read latency. Log-only, never plotted.
  local io_s=0 iolog fsync_s=0
  iolog=$(ls -1 "$store"/io.csv "$store"/*/io.csv 2>/dev/null | head -1)
  [ -n "$iolog" ] && io_s=$(phase_union "$iolog" ms start_ns "$win0" "$win1")
  # The durability barrier is I/O too, and the driver times it outside the bdev
  # (it fdatasyncs by path), so it would otherwise land in the solid segment.
  fsync_s=$(grep -hoE "fsync: [0-9]+ file\(s\) in [0-9.]+ s" "$store"/*/stdout.log 2>/dev/null \
            | tail -1 | grep -oE "[0-9.]+ s$" | grep -oE "[0-9.]+")
  io_s=$(awk -v a="${io_s:-0}" -v b="${fsync_s:-0}" 'BEGIN{printf "%.3f", a + b}')
  [ -n "$total_s" ] && sc_s=$(awk -v i="${io_s:-0}" -v t="$total_s" \
    'BEGIN{ v = t - i; if (v < 0) v = 0; if (v > t) v = t; printf "%.3f", v }')
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
  ARM_T2DIR=""; ARM_PFSDIR=""; ARM_STORE=""
  local failed
  failed=$(grep -hoE "failed: [0-9]+" "$store"/*/stdout.log 2>/dev/null | tail -1)
  echo "     rc=$rc loop=${stage_s:-?}s codec=${codec_s:-?}s cusz_mgr=${setup_s:-0}s io=${io_s:-?}s (fsync ${fsync_s:-0}s) compute=${sc_s:-?}s total=${total_s:-?}s (H2D staging ${h2d_s}s, input read ${read_s}s removed) ${failed:-} ${bound:-}" >&2
  echo "     tier=$tier async_ms=$flush -> $([ "$tier" = 1 ] \
         && echo "NVMe $t2dir" || echo "PFS $pfsdir") | stored ${stored_b:-?} B | final ${fl#flush: }" >&2
  [ -n "$durep" ] && echo "     $durep" >&2
  echo "     WAL: ${walrep:-(none)}" >&2
  # A failed run or a lost chunk is recorded as TBD. A run whose ONLY failure is
  # the error-bound check is still recorded -- its time is real -- but flagged,
  # so a codec that breaks the bound is not silently shown as meeting it.
  local verdict bflag=""
  verdict=$(grep -hE "^(VERIFIED|FAILED):" "$store"/*/stdout.log 2>/dev/null | tail -1)
  case "$bound" in "BOUND OK"*) bflag=ok ;; "BOUND FAILED"*) bflag=exceeded ;; esac
  # THE DURABILITY BARRIER IS NOT WAIVABLE. `bound_only` exists so a codec that
  # misses its error bound is still timed, but a run whose bytes never reached
  # the device has not measured this figure's quantity at all -- so a barrier
  # failure is never "bound only", even when the bound also failed.
  local fsync_bad
  fsync_bad=$(grep -hoE "^FSYNC FAILED:.*" "$store"/*/stdout.log 2>/dev/null | tail -1)
  local bound_only=0
  [ "$rc" -ne 0 ] && [ "$bflag" = exceeded ] && [ -z "$fsync_bad" ] \
    && ! echo "$verdict" | grep -q FAILED && bound_only=1
  if [ -n "$fsync_bad" ]; then
    echo "     NOT RECORDED: $fsync_bad" >&2
    return 0
  fi
  if { [ "$rc" -ne 0 ] && [ "$bound_only" = 0 ]; } || echo "$verdict" | grep -q FAILED \
     || { [ -n "$failed" ] && [ "${failed#failed: }" != 0 ]; }; then
    echo "     NOT RECORDED: rc=$rc ${failed:-} ${verdict%%(*} -- see $store/console.log" >&2
    return 0
  fi
  [ "$bound_only" = 1 ] && echo "     RECORDED WITH BOUND EXCEEDED: $bound" >&2
  [ -n "$total_s" ] && awk -v c="$sc_s" -v t="$total_s" -v i="$in_b" -v b="$stored_b" -v f="$bflag" \
    -v u="${setup_s:-0}" 'BEGIN{
    if (c == "") printf ",,%.4f", t/60.0
    else printf "%.4f,%.4f,%.4f", c/60.0, (t-c)/60.0, t/60.0
    printf ","; if (b > 0) printf "%.3f", i/b
    printf ",%s,%.4f", f, u/60.0}'
}

# Warm the page cache so the first arm does not pay the cold read alone.
warm_cache() {
  [ "$DRY" = 1 ] && return 0
  local -a files
  mapfile -t files < <(cut -f1 "$FILE_INDEX")
  [ "$MAXF" -gt 0 ] 2>/dev/null && files=( "${files[@]:0:$MAXF}" )
  echo "== warming page cache over ${#files[@]} file(s) (untimed)"
  printf '%s\0' "${files[@]}" | xargs -0 -r cat > /dev/null 2>&1
  # The driver du's the whole tree per arm: ~11 ms a file cold on Lustre.
  du -sm "$FIELDS" > /dev/null 2>&1
}

echo "== figure 9: $WL / $SIZE -> $OUT"
echo "== every NeuroPress arm runs config '$NP_CFG'"
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
 "np_config":"$NP_CFG","best_fixed":"$BEST_FIXED","worst_fixed":"$WORST_FIXED","chunk":$CHUNK,"max_files":$MAXF,"smoke_gb":"$SMOKE_GB",
 "eb":"$EB_LOW","panel_b_tier":$PANEL_B_TIER,"async_ms":$ASYNC_MS,"ram_mb":${RAM_MB:-0},
 "cusz_reuse":"${CUSZ_REUSE:-auto}",
 "cost_bw_bytes_per_ms":"$COST_BW","np_lr":$NP_LR,"np_mape":$NP_MAPE,
 "measure_dt":$MEASURE_DT,"measure_quality":$MEASURE_QUALITY,
 "arm_timeout_s":$ARM_TIMEOUT,"fields":"$FIELDS",
 "pfs_root":"$PFS_ROOT","nvme_root":"$NVME_ROOT",
 "selection_log":${SELECTION_LOG:-0},"exclude_read":${EXCLUDE_READ:-1},"staged_input":"$STAGED_DIR",
 "reps":${REPS:-3}}
JSON

# Never truncate earlier results: rerunning a few arms into the same --out
# would otherwise wipe every row the previous run recorded.
RUNS_CSV="$OUT/fig9_runs.csv"
for _f in "$CSV" "$RUNS_CSV"; do
  [ "$DRY" != 1 ] && [ -s "$_f" ] && cp "$_f" "$_f.$(date +%m%d%H%M%S).bak"
done
unset _f
echo "panel,workload,strategy,rep,compute_min,io_min,total_min,ratio,bound,setup_min,eb" > "$RUNS_CSV"

# REPEATED RUNS, ROTATED ORDER. One run per arm in a fixed order was the
# figure's noisiest input: PFS write loops moved 25-30% between identical runs
# (Baseline 59 vs 76 s on full Nyx), single Lustre stalls of 0.1-2 s decided
# which arm "won", and whichever arm ran first read the input warmest. Each
# arm now runs REPS times; rep k starts the arm list k/REPS of the way round,
# so no arm always goes first or last.
SELECTED=()
for spec in "${ARMS[@]}"; do
  IFS='|' read -r label panel _ _ _ _ <<< "$spec"
  arm_selected "$label" "$panel" && SELECTED+=( "$spec" )
done
# The other workloads' TBD rows only keep one CSV's plot at the five-workload
# layout; merging per-workload CSVs does not need them. TBD_OTHERS=1 adds them.
TBD_ROWS="$OUT/.tbd_rows"
: > "$TBD_ROWS"
if [ "${TBD_OTHERS:-0}" = 1 ]; then
  for w in "${WORKLOADS_ALL[@]}"; do
    [ "$w" = "$WLNAME" ] && continue
    while IFS='|' read -r label panel _ eb _ _; do
      [ -n "$label" ] || continue
      [ "$PANEL" = both ] || [ "$PANEL" = "$panel" ] || continue
      echo "$panel,$w,\"$label\",,,,,,$eb,0,,," >> "$TBD_ROWS"
    done < <(build_arms "$w")
  done
fi

# aggregate_csv -- rebuild fig9.csv from every run recorded so far: per arm,
# the MEDIAN run by total (the mean of the middle two for an even count), so
# the plotted segments still add up to the plotted total; std_min is the
# sample standard deviation of the totals. An arm with no recorded run is a
# blank (TBD) row. Rewritten after every run, so a walltime kill keeps it all.
aggregate_csv() {
  printf '%s\n' "${SELECTED[@]}" | python3 -c '
import csv, statistics as st, sys
runs_csv, out_csv, tbd, wl, payload = sys.argv[1:6]
runs = list(csv.DictReader(open(runs_csv)))
f = lambda v: float(v) if v not in ("", None) else None
with open(out_csv, "w") as fh:
    fh.write("panel,workload,strategy,compute_min,io_min,total_min,std_min,ratio,eb,runs,bound,payload_mib,setup_min\n")
    for spec in sys.stdin.read().split("\n"):
        if not spec:
            continue
        label, panel, _, eb = spec.split("|")[:4]
        rs = sorted((r for r in runs if r["strategy"] == label and r["panel"] == panel),
                    key=lambda r: f(r["total_min"]))
        if not rs:
            fh.write(f"{panel},{wl},\"{label}\",,,,,,{eb},0,,{payload},\n")
            continue
        n = len(rs)
        mid = rs[n // 2: n // 2 + 1] if n % 2 else rs[n // 2 - 1: n // 2 + 1]
        def med(col):
            # .get: a runs CSV written before this column existed still
            # aggregates, with the column left blank rather than a KeyError.
            v = [f(r.get(col, "")) for r in mid]
            return "" if None in v else "%.4f" % (sum(v) / len(v))
        std = "%.4f" % st.stdev([f(r["total_min"]) for r in rs]) if n > 1 else ""
        ratio = [f(r["ratio"]) for r in rs if f(r["ratio"]) is not None]
        flags = {r["bound"] for r in rs}
        bound = "exceeded" if "exceeded" in flags else ("ok" if "ok" in flags else "")
        comp, io, tot = med("compute_min"), med("io_min"), med("total_min")
        rat = "%.3f" % st.median(ratio) if ratio else ""
        setup = med("setup_min")
        fh.write(f"{panel},{wl},\"{label}\",{comp},{io},{tot},{std},{rat},{eb},{n},{bound},{payload},{setup}\n")
    fh.write(open(tbd).read())
' "$RUNS_CSV" "$CSV" "$TBD_ROWS" "$WLNAME" "${PAYLOAD_MB:-}"
}

REPS=${REPS:-3}
aggregate_csv
NSEL=${#SELECTED[@]}
for ((rep = 1; rep <= REPS; rep++)); do
  echo "== rep $rep/$REPS"
  off=$(( NSEL > 0 ? (rep - 1) * NSEL / REPS : 0 ))
  for ((i = 0; i < NSEL; i++)); do
    IFS='|' read -r label panel cfg eb tier flush <<< "${SELECTED[$(( (i + off) % NSEL ))]}"
    split=$(run_arm "$label" "$cfg" "$eb" "$tier" "$flush" "$rep")
    [ -n "$split" ] && echo "$panel,$WLNAME,\"$label\",$rep,$split,$eb" >> "$RUNS_CSV"
    [ "$DRY" != 1 ] && aggregate_csv
  done
  [ "$DRY" = 1 ] && break
done

echo; echo "csv: $CSV"
[ "$DRY" = 1 ] && exit 0
python3 "$HERE/plot_fig9.py" --csv "$CSV" --out "$OUT/figures"
