#!/usr/bin/env bash
#PBS -l select=16
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user
#PBS -l daos=daos_user
#PBS -q debug-scaling
#PBS -A IOWarp
#PBS -j oe
#
# IO500 on Aurora: clio-fs (tiered DRAM / DAOS / Flare) vs plain Flare.
#
#   qsub -l select=16 -v MODE=cliofs benchmarks/io500/pbs_io500_aurora.sh
#   qsub -l select=16 -v MODE=flare  benchmarks/io500/pbs_io500_aurora.sh
#   qsub -l select=16 -v MODE=daos   benchmarks/io500/pbs_io500_aurora.sh
#
# MODE=daos  : IO500 datadir on a dfuse mount of DAOS pool $DAOS_POOL.
# MODE=flare : IO500 datadir on Lustre (Flare) directly.
# MODE=cliofs: one clio_run + clio_cte_fuse per node; the CTE has three tiers
#              per node, sized from the node's share of the dataset D:
#                ram::            RAM_PCT% of D  (score 1.0)
#                file on dfuse    DAOS_PCT% of D (DAOS pool $DAOS_POOL, score 0.6)
#                                 (DAOS_NATIVE=1: the same file through libdfs)
#                file on Flare    FLARE_X x D    (score 0.2, overflow tier)
#              With the defaults (25 + 50) 75% of the data fits in DRAM+DAOS
#              and the remaining 25% has to spill to Flare.
#              IO500 datadir is the clio-fs mount (same path on every node).
#
# Knobs (qsub -v): MODE, PPN (IO500 ranks per node, 16), STONEWALL (300 s,
# the IO500 rule), RAM_PCT (25), DAOS_PCT (50), FLARE_X (4), BUILD_DIR
# (build/fs), IO500 (binary), DAOS_POOL (IOWarp), DAOS_CONT (io500_<nodes>n).
# CLIO_FUSE_DIR_ATTR_CACHE_S (unset = off on multi-node): seconds the adapter
# caches directory attributes; passed through to clio_cte_fuse unchanged.
# DAOS_CONT_OPTS (unset): extra comma-separated `daos cont create` options,
# e.g. --dir-oclass=SX to stripe directories across all targets.
# DAOS_API (MODE=daos; dfuse): dfs runs IO500 on libdfs directly, no FUSE.
# NO_FLARE (0): 1 drops the Flare tier; tiers are then DRAM + DAOS only.
# DAOS_NATIVE (unset/0 = off; MODE=cliofs): 1 makes the DAOS tier a native
# DAOS bdev (bdev_type "daos") instead of a file on the dfuse mount: the
# runtime reads and writes daos://$DAOS_POOL/$DAOS_CONT/tiers/<job>/<host>.dat
# through libdfs, with no FUSE layer in the data path. Size, score and
# persistence_level of the tier are unchanged. BUILD_DIR must be a build
# configured with -DCLIO_ENABLE_DAOS=ON (checked up front). The dfuse mount is
# still brought up: the tier file is an ordinary file of the container, so
# the per-node du sampling and the teardown rm keep working through it.
# DAOS_NATIVE_OPTS (unset): options appended to the daos:// path, e.g.
# "chunk_size=4m&oclass=SX&prezero=0" (see
# the native DAOS bdev in context-runtime/config/clio_default.yaml). prezero=0 drops the transport's
# zeroing of every read buffer; safe only once the DAOS client is known to
# return zeros for holes.
# META_RAM_MB (unset = off): MB of an extra DRAM target per node,
# ram::clio_fs_meta (score 1.0), reserved for filesystem metadata and the
# first MiB of each file. The bulk DRAM target (still RAM_PCT% of D) drops to
# score 0.9 and the filesystem module gets data_score: 0.9, so file data
# beyond a file's first MiB no longer competes with directory blocks and
# small files for the fastest target. Works with every other knob.
# HANG_DUMP_S (unset = off): seconds after IO500 starts at which a still-running
# job dumps every node's runtime and adapter stacks (hang diagnosis).
# COMPOSE (split): how the CTE is composed under clio-fs.
#   split : two cores and the dedicated caches (clio_default.yaml's layout):
#           persistent core (DAOS / Flare file tiers) + volatile core (DRAM)
#           under CACHE_POLICY (write_back | write_thru), and the
#           shared-memory cache on top (SHM_CACHE=0 leaves it out;
#           SHM_MD_ONLY, SHM_CAP size it). WB_FILL_PCT (unset = the module's
#           80) is the write-back cache's flush_fill_pct. WB_FLUSH_MS is the write-back
#           flush period.
#   legacy: one core holding every tier under the node-local clio_cte_cache
#           (CACHE, CACHE_MD_ONLY), as before the cache split.
# PFIND_HASH (unset/0 = off): 1 adds
# pfind-parallelize-single-dir-access-using-hashing = TRUE to io500.ini's
# [find] section, so pfind splits one large directory across ranks by
# telldir/seekdir offset. Needs directory offsets that are stable across
# processes (clio-fs's are; see DirEntryOff in
# context-transfer-engine/filesystem/include/clio_cte/filesystem/filesystem_tasks.h).
set -u
MODE=${MODE:-cliofs}
PPN=${PPN:-16}
STONEWALL=${STONEWALL:-300}
# Weak scaling: a fixed amount of work PER RANK (the IOWarp Flare project
# quota is 1 TiB, so it is small: 128 nodes x 16 ranks writes ~230 GB).
# Scores are therefore not valid IO500 submissions.
EASY_MB=${EASY_MB:-96}          # ior-easy bytes per rank (multiple of 2m)
HARD_SEG=${HARD_SEG:-400}       # ior-hard 47008-byte segments per rank (~18 MB)
MD_N=${MD_N:-2000}              # mdtest-easy/hard files per rank
BUILD_DIR=${BUILD_DIR:-/home/llogan/clio-core/build/fs}
IO500=${IO500:-/lus/flare/projects/IOWarp/llogan/io500/src/io500}
DAOS_POOL=${DAOS_POOL:-IOWarp}
FLARE=${FLARE:-/lus/flare/projects/IOWarp/llogan/io500}
HERE=${HERE:-/home/llogan/clio-core/benchmarks/io500}

# Ask the per-node scripts to tear down and wait at most DOWN_WAIT seconds:
# at 256 nodes a teardown that never finished held the whole allocation
# for ten minutes after the benchmark had already failed.
stop_nodes() {
  touch stop
  for _ in $(seq 1 "${DOWN_WAIT:-180}"); do
    kill -0 "$NODES_PID" 2>/dev/null || return 0
    sleep 1
  done
  echo "=== teardown still running after ${DOWN_WAIT:-180} s; ending the job"
  kill "$NODES_PID" 2>/dev/null
}
[ "${CORE_DUMPS:-0}" = 1 ] || ulimit -c 0  # see cliofs_node.sh
NN=$(sort -u "$PBS_NODEFILE" | wc -l)
# MD_MAX_FILES: cap the files alive at any moment in BENCH=mdtest by
# deriving MD_N (files per rank); each layout is deleted before the next is
# created, so at most NN x PPN x MD_N files exist at once.
if [ -n "${MD_MAX_FILES:-}" ]; then
  MD_N=$(( MD_MAX_FILES / (NN * ${PPN:-16}) ))
  [ "$MD_N" -lt 1 ] && MD_N=1
  echo "=== mdtest cap ${MD_MAX_FILES} files alive: MD_N=$MD_N per rank ($(( MD_N * NN * ${PPN:-16} )) files)"
fi
# IOR_FOOTPRINT_GB: cap the IOR data footprint (all ranks together) at this
# many GB by deriving EASY_MB, rounded down to a multiple of 16 MiB (the
# shared16m segment). Rewrites (IOR_MINTIME) stay inside the footprint.
if [ -n "${IOR_FOOTPRINT_GB:-}" ]; then
  EASY_MB=$(( IOR_FOOTPRINT_GB * 1024 / (NN * ${PPN:-16}) / 16 * 16 ))
  [ "$EASY_MB" -lt 16 ] && EASY_MB=16
  echo "=== IOR footprint cap ${IOR_FOOTPRINT_GB} GB: EASY_MB=$EASY_MB per rank ($(( EASY_MB * NN * ${PPN:-16} / 1024 )) GB total)"
fi
# Log the DAOS pool's space at the start and end of every job, so a run's
# footprint on a storage system we do not own is visible in its output.
pool_space() {
  echo "=== DAOS pool $DAOS_POOL space ($1): $(daos pool query "${DAOS_POOL:-IOWarp}" 2>/dev/null | grep -iE 'Total size|Free' | tr -s ' ' | paste -sd ';')"
}
NP=$(( NN * PPN ))
# Per-node share of the dataset in MB: ior-easy + ior-hard + mdtest-hard
# (3901-byte files). Blobs land on the node that wrote them (pool_query
# local, neighborhood 1), so this is what each node's tiers must hold.
DATA_MB=$(( PPN * (EASY_MB + HARD_SEG * 47008 / 1048576) + PPN * MD_N * 3901 / 1048576 + 1 ))
# BENCH=ior writes only the EASY_MB footprint (one pattern at a time).
[ "${BENCH:-io500}" = ior ] && DATA_MB=$(( PPN * EASY_MB + 1 ))
# BENCH=wfcommons writes one WfBench workflow (WF_JSON, made by
# wfcommons/wfc_gen.py): its footprint (every file, staged inputs and task
# outputs, each written once) spread over the nodes. Tasks run wherever
# Parsl places them, so a node may hold more than its share: RAM_PCT /
# DAOS_X give clio-fs the slack.
if [ "${BENCH:-io500}" = wfcommons ]; then
  WF_JSON=${WF_JSON:?BENCH=wfcommons needs WF_JSON}
  WF_FOOT_MB=$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["workflow"]["specification"]; print(sum(f["sizeInBytes"] for f in s["files"]) // 2**20 + 1)' "$WF_JSON") \
    || { echo "cannot read $WF_JSON"; exit 2; }
  DATA_MB=$(( WF_FOOT_MB / NN + 1 ))
  echo "=== wfcommons $(basename "$WF_JSON"): footprint ${WF_FOOT_MB} MB, ${DATA_MB} MB per node"
fi
RAM_PCT=${RAM_PCT:-25}
DAOS_PCT=${DAOS_PCT:-50}
# The Flare tier is the overflow: it only has to take the remaining
# (100 - RAM_PCT - DAOS_PCT)% but is sized FLARE_X x the dataset so the
# filesystem never runs out of space. It grows lazily, so the Flare quota
# only pays for what actually spills.
FLARE_X=${FLARE_X:-4}
# RUN_TAG distinguishes several runs inside one job (pbs_ior_sweep.sh).
JOB=${PBS_JOBID%%.*}${RUN_TAG:+_$RUN_TAG}
OUT=$FLARE/runs/${MODE}_${NN}n_${JOB}
mkdir -p "$OUT"; cd "$OUT"
sort -u "$PBS_NODEFILE" > hostfile
echo "=== IO500 mode=$MODE nodes=$NN ppn=$PPN stonewall=$STONEWALL out=$OUT"
echo "=== commit $(git -C "$HERE" rev-parse --short HEAD) $(git -C "$HERE" status --porcelain -uno | wc -l) dirty files; build $BUILD_DIR"

# The launcher pings every node and aborts the job when one does not answer.
# Under the dfuse mdtest load at 64 and 256 nodes a node stayed silent for
# ~100 s and both DAOS runs were killed in mdtest-easy-write.
export PALS_PING_PERIOD=${PALS_PING_PERIOD:-120} PALS_RPC_TIMEOUT=${PALS_RPC_TIMEOUT:-900}
export PATH=/soft/daos/bin:$PATH DAOS_AGENT_DRPC_DIR=/run/daos_agent_oneScratch
# Same fabric settings as `module load daos` (without them dfuse failed DER_HG).
export FI_CXI_RX_MATCH_MODE=hybrid FI_MR_CACHE_MONITOR=kdreg2 \
  FI_MR_CACHE_MAX_COUNT=2000000 FI_CXI_DEFAULT_CQ_SIZE=131072 NA_OFI_CXI_PROTO_RNR=0
pool_space start

# One DAOS container per node count, so jobs at different scales (different
# queues, possibly concurrent) never share one. DAOS_RESET=1 destroys it at
# the start: failed runs leave millions of mdtest files behind, deleting
# them through dfuse takes hours, and they filled the pool's metadata tier
# (mdtest-hard-write hit ENOSPC at 64 nodes). DAOS_DESTROY lists other
# containers to drop first (the old shared io500_cliofs).
DAOS_CONT=${DAOS_CONT:-io500_${NN}n}
_cont_opts=${DAOS_CONT_OPTS:-}  # set -u safe; see DAOS_CONT_OPTS
# node_snapshot <label>: min/median/max of MemAvailable, /tmp use and the
# 1-minute load over every node, to see what a system's bring-up leaves.
node_snapshot() {
  mpiexec -n "$NN" -ppn 1 --cpu-bind none bash -c \
    'echo "$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) $(df -BG --output=used /tmp | tail -1 | tr -dc 0-9) $(cut -d" " -f1 /proc/loadavg)"' 2>/dev/null \
    | sort -n | awk -v l="$1" '{m[NR]=$1; t[NR]=$2; d[NR]=$3} END{n=NR; if(n==0){print "=== node snapshot " l ": no data"; exit}
      asort(t); asort(d); printf "=== node snapshot %s (%d nodes): MemAvailable GB min %d med %d max %d; /tmp used GB med %d max %d; load1 med %.1f max %.1f\n", l, n, m[1], m[int((n+1)/2)], m[n], t[int((n+1)/2)], t[n], d[int((n+1)/2)], d[n]}'
}
# BENCH=wfcommons: unpack the workflow venv into node-local /tmp BEFORE the
# filesystem under test starts, so its runtime cannot slow or break the unpack
# (a 256-node Context FS run took 810 s and failed when this ran after it).
if [ "${BENCH:-io500}" = wfcommons ] && [ "${WF_LOCAL_VENV:-1}" = 1 ]; then
  node_snapshot "before bring-up"
  t0=$(date +%s.%N)
  mpiexec -n "$NN" -ppn 1 --cpu-bind none bash -c \
    "rm -rf /tmp/$USER/wfc-venv; mkdir -p /tmp/$USER && tar -C /tmp/$USER -xf ${WF_VENV:-/lus/flare/projects/IOWarp/llogan/wfc-venv}.tar" \
    && WF_UNPACKED=1 || echo "=== venv unpack FAILED (before bring-up)"
  echo "=== venv unpacked before bring-up on $NN nodes in $(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b - a}') s"
fi
if [ "$MODE" != flare ]; then
  _destroy=${DAOS_DESTROY:-}
  for c in ${_destroy//,/ }; do  # comma- or space-separated
    daos cont destroy "$DAOS_POOL" "$c" --force 2>&1 | grep -v ' ERR ' | tail -1
  done
  # DAOS_LIST_CONTS=1: print every container in the pool with its size, so a
  # job shows who holds the pool's space (the login node cannot query DAOS).
  if [ "${DAOS_LIST_CONTS:-1}" = 1 ]; then
    echo "=== DAOS containers in $DAOS_POOL:"
    daos cont list "$DAOS_POOL" 2>/dev/null | awk 'NR>2{print $NF}' | while read -r c; do
      [ -z "$c" ] && continue
      echo "===   $c $(daos cont query "$DAOS_POOL" "$c" 2>/dev/null | grep -iE 'Owner|Total|Size|Snapshot' | tr -s ' ' | paste -sd ';' | cut -c1-160)"
    done
  fi
  if [ "${DAOS_RESET:-1}" = 1 ]; then
    daos cont destroy "$DAOS_POOL" "$DAOS_CONT" --force 2>&1 | grep -v ' ERR ' | tail -1
  fi
  daos pool query "$DAOS_POOL" 2>&1 | grep -i -E 'total size|free' | head -4
  # DAOS PREFLIGHT: the pool is shared and small, and a run that fills a
  # target dies with ENOSPC partway through. Skip the run up front (exit 4,
  # the sweep moves on to its next configuration) unless the pool has
  # DAOS_HEADROOM (3) x this run's data free on NVMe, every NVMe target keeps
  # its share of the run (x headroom) plus DAOS_MIN_TGT_MB (64), and every SCM target keeps
  # DAOS_MIN_SCM_MB (64) free. DAOS_PREFLIGHT=0 skips the check.
  # A container destroyed just above frees its space in the background, so
  # poll for up to DAOS_PREFLIGHT_WAIT_S (300) before deciding to skip.
  if [ "${DAOS_PREFLIGHT:-1}" = 1 ]; then
   pf_deadline=$(( SECONDS + ${DAOS_PREFLIGHT_WAIT_S:-300} ))
   while :; do
    daos pool query "$DAOS_POOL" 2>/dev/null | python3 -c '
import re, sys
need_mb, head, min_tgt, min_scm = (float(a) for a in sys.argv[1:5])
u = {"B": 1/2**20, "KB": 1/2**10, "MB": 1, "GB": 2**10, "TB": 2**20, "PB": 2**30}
mb = lambda v, x: float(v) * u[x.upper()]
tiers = re.findall(r"Free:\s*([\d.]+)\s*([KMGTP]?B),\s*min:\s*([\d.]+)\s*([KMGTP]?B).*?mean:\s*([\d.]+)\s*([KMGTP]?B)", sys.stdin.read(), re.I)
if len(tiers) < 2:
    print("=== DAOS PREFLIGHT: pool query unreadable; not running"); sys.exit(1)
scm_min = mb(*tiers[0][2:4]); nvme_free = mb(*tiers[1][:2]); nvme_min = mb(*tiers[1][2:4])
nvme_mean = mb(*tiers[1][4:6])
# Each NVMe target holds about need/targets of this run; require head x that
# plus min_tgt MB on the fullest one (min_tgt default 64), not a fixed 1 GB.
targets = max(1.0, nvme_free / max(nvme_mean, 1e-9))
min_tgt = min_tgt + head * need_mb / targets
ok = nvme_free >= head * need_mb and nvme_min >= min_tgt and scm_min >= min_scm
print("=== DAOS PREFLIGHT %s: run needs %.0f GB (x%.0f headroom); NVMe free %.0f GB, min/target %.0f MB; SCM min/target %.0f MB"
      % ("ok" if ok else "SKIP", need_mb / 1024, head, nvme_free / 1024, nvme_min, scm_min))
sys.exit(0 if ok else 1)
' "$(( DATA_MB * NN ))" "${DAOS_HEADROOM:-3}" "${DAOS_MIN_TGT_MB:-64}" "${DAOS_MIN_SCM_MB:-64}" \
      && break
    if [ "$SECONDS" -ge "$pf_deadline" ]; then
      echo "=== DAOS PREFLIGHT: skipping $MODE run rc=4"; exit 4
    fi
    sleep 20
   done
  fi
fi

if [ "$MODE" = cliofs ]; then
  test -x "$BUILD_DIR/bin/clio_cte_fuse" || { echo "NO $BUILD_DIR/bin/clio_cte_fuse"; exit 2; }
  RAM_GB=$(( DATA_MB * RAM_PCT / 100 )); DAOS_GB=$(( DATA_MB * DAOS_PCT / 100 )); FL_GB=$(( DATA_MB * FLARE_X ))
  # NO_FLARE=1: DRAM + DAOS only (the configuration to compare against DAOS).
  # The DAOS tier becomes the overflow tier, sized DAOS_X (2) x the dataset.
  if [ "${NO_FLARE:-0}" = 1 ]; then DAOS_GB=$(( DATA_MB * ${DAOS_X:-2} )); FL_GB=0; fi
  echo "=== cache module ${CACHE:-1}; data per node ${DATA_MB}MB; tiers per node: DRAM ${RAM_GB}MB (${RAM_PCT}%)  DAOS ${DAOS_GB}MB (${DAOS_PCT}%)  Flare ${FL_GB}MB (overflow)"
  daos pool query "$DAOS_POOL" 2>&1 | grep -i -E 'total size|free' | head -4
  daos cont query "$DAOS_POOL" "$DAOS_CONT" > /dev/null 2>&1 \
    || { cc=$(daos cont create "$DAOS_POOL" "$DAOS_CONT" --type POSIX ${_cont_opts//,/ } 2>&1); \
         echo "=== DAOS cont create $DAOS_CONT ${_cont_opts//,/ }: $(echo "$cc" | grep -v ' ERR ' | tail -2 | tr '\n' ' ')"; }
  mkdir -p "$FLARE/tiers/$JOB"
  find "$FLARE/tiers" -mindepth 1 -maxdepth 1 -mmin +120 ! -name "$JOB" -exec rm -rf {} + 2>/dev/null
  LR=/tmp/$USER/cliofs
  COMPOSE=${COMPOSE:-split}
  # A RAM-only run has no persistent tier for the split layout to flush to.
  [ "${RAM_ONLY:-0}" = 1 ] && COMPOSE=legacy
  CACHE_POLICY=${CACHE_POLICY:-write_back}
  cat > clio_server.yaml <<EOF
# Generated by pbs_io500_aurora.sh (job $JOB, compose $COMPOSE)
memory:
  main_segment_size: auto
  client_data_segment_size: 4GB
networking:
  port: 9519
  hostfile: $OUT/hostfile
  neighborhood_size: 1024
runtime:
  num_threads: ${NUM_THREADS:-16}
  queue_depth: 4096
  conf_dir: $LR/conf
compose:
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
$( [ "$COMPOSE" = split ] && echo "    durability: persistent" )
    storage:
EOF
  # META_RAM_MB: a small DRAM target above the bulk one, for metadata and
  # file heads (see the header). Its name differs from the bulk target's, so
  # the two are separate bdev pools.
  RAM_SCORE=1.0
  RAM_TIERS=""
  if [ -n "${META_RAM_MB:-}" ]; then
    RAM_SCORE=0.9
    echo "=== metadata DRAM target ${META_RAM_MB}MB per node (score 1.0); bulk DRAM score $RAM_SCORE; data_score 0.9"
    RAM_TIERS="      - path: \"ram::clio_fs_meta\"
        bdev_type: \"ram\"
        capacity_limit: \"${META_RAM_MB}MB\"
        score: 1.0
"
  fi
  RAM_TIERS="$RAM_TIERS      - path: \"ram::clio_fs_ram\"
        bdev_type: \"ram\"
        capacity_limit: \"${RAM_GB}MB\"
        score: $RAM_SCORE"
  # Legacy: the DRAM targets are devices of the single core. Split: they are
  # the volatile core's devices (written further down).
  [ "$COMPOSE" = split ] || echo "$RAM_TIERS" >> clio_server.yaml
  # PERSIST=volatile turns off the file tiers' allocator write-ahead log
  # (a synchronous append to the network filesystem per allocate/free).
  # RAM_ONLY=1 leaves out the DAOS and Flare file tiers: a software-stack
  # ceiling with no storage I/O at all (diagnostic, not a real deployment).
  if [ "${RAM_ONLY:-0}" != 1 ]; then
    # DAOS_NATIVE=1: the same file of the container (<host>.dat under
    # tiers/$JOB), reached through libdfs by the bdev's DAOS transport
    # instead of through the dfuse mount.
    if [ "${DAOS_NATIVE:-0}" = 1 ]; then
      ldd "$BUILD_DIR/bin/libclio_bdev_runtime.so" 2>/dev/null | grep -q 'libdfs' \
        || { echo "DAOS_NATIVE=1 needs a build with -DCLIO_ENABLE_DAOS=ON; $BUILD_DIR has no DAOS transport"; exit 2; }
      DAOS_TIER_PATH="daos://$DAOS_POOL/$DAOS_CONT/tiers/$JOB/__HOST__.dat${DAOS_NATIVE_OPTS:+?$DAOS_NATIVE_OPTS}"
      DAOS_TIER_TYPE=daos
      echo "=== DAOS tier: native (bdev_type daos) $DAOS_TIER_PATH"
    else
      DAOS_TIER_PATH="/tmp/$USER/daos_io500/tiers/$JOB/__HOST__.dat"
      DAOS_TIER_TYPE=file
    fi
    cat >> clio_server.yaml <<EOF
      - path: "$DAOS_TIER_PATH"
        bdev_type: "$DAOS_TIER_TYPE"
        capacity_limit: "${DAOS_GB}MB"
        score: 0.6
        persistence_level: "${PERSIST:-temporary}"
EOF
    [ "${NO_FLARE:-0}" = 1 ] || cat >> clio_server.yaml <<EOF
      - path: "$FLARE/tiers/$JOB/__HOST__.dat"
        bdev_type: "file"
        capacity_limit: "${FL_GB}MB"
        score: 0.2
        persistence_level: "${PERSIST:-temporary}"
EOF
  fi
  cat >> clio_server.yaml <<EOF
    dpe:
      dpe_type: "max_bw"
    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000
EOF
  # DIR_LOG_COMPACT sets dir_log_compact_bytes: how much a directory block's
  # blob may grow by appended deltas before it is rewritten as one image
  # (unset = the module default; 0 = rewrite the whole block every commit).
  # META_LOG=1 gives the filesystem module a node-local directory journal
  # (metadata_log_path). Without it every create and delete rewrites its
  # whole directory block as a CTE blob before it is acknowledged.
  # CACHE=1 (default): filesystem -> stream -> cache -> core. The cache
  # module keeps a node-local raw copy (tiers scoring >= min_score) next to
  # the authoritative copy on the blob's hash-owner node, so every byte is
  # stored twice. CACHE_MD_ONLY=true keeps the module for its metadata role
  # (tag write-through) but stores no node-local data copies. CACHE=0 drops it: stream and filesystem chain straight to
  # the core and each byte is stored once, on its owner node.
  if [ "$COMPOSE" = split ]; then
    # Volatile DRAM core (540.0) beside the persistent core (512.0), under
    # the policy cache (567.0), which owns flushing DRAM copies down; the
    # shared-memory cache (568.0) on top serves tag and blob metadata to
    # clients without a task (and blob bytes too unless SHM_MD_ONLY).
    cat >> clio_server.yaml <<EOF
  - mod_name: clio_cte_core
    pool_name: cte_volatile
    pool_query: local
    pool_id: "540.0"
    durability: volatile
    storage:
$RAM_TIERS
    dpe:
      dpe_type: "max_bw"
    targets:
      neighborhood: 1
  - mod_name: clio_cte_core_persist_$CACHE_POLICY
    pool_name: clio_cte_core_persist_$CACHE_POLICY
    pool_query: local
    pool_id: "567.0"
    next_volatile: "540.0"
    next_persistent: "512.0"
    flush_period_ms: ${WB_FLUSH_MS:-10000}
$( [ -n "${WB_FILL_PCT:-}" ] && echo "    flush_fill_pct: ${WB_FILL_PCT}" )
$( [ -n "${WB_DIRTY_MB:-}" ] && echo "    flush_dirty_mb: ${WB_DIRTY_MB}" )
$( [ "$CACHE_POLICY" = write_back ] && [ "${SHM_CACHE:-1}" = 1 ] && [ "${WB_LAND_SHM:-1}" = 1 ] && echo "    flush_pool: \"568.0\"" )
EOF
    NEXT=567.0
    if [ "${SHM_CACHE:-1}" = 1 ]; then
      cat >> clio_server.yaml <<EOF
  - mod_name: clio_cte_core_shm_cache_writethru
    pool_name: clio_cte_core_shm_cache_writethru
    pool_query: local
    pool_id: "568.0"
    next_pool_id: "567.0"
    md_only: ${SHM_MD_ONLY:-true}
    capacity: "${SHM_CAP:-256M}"
$( [ -n "${SHM_MAX_BLOBS:-}" ] && echo "    max_blobs: ${SHM_MAX_BLOBS}" )
$( [ -n "${SHM_FLUSH_DEPTH:-}" ] && echo "    flush_depth: ${SHM_FLUSH_DEPTH}" )
$( [ -n "${SHM_FLUSH_BATCH:-}" ] && echo "    flush_batch_bytes: \"${SHM_FLUSH_BATCH}\"" )
$( [ "$CACHE_POLICY" = write_back ] && [ "${WB_LAND_SHM:-1}" = 1 ] && echo "    invalidate_on_flush: true" )
EOF
      NEXT=568.0
    fi
    echo "=== compose split: persist_$CACHE_POLICY (flush ${WB_FLUSH_MS:-10000} ms), shm cache ${SHM_CACHE:-1} (md_only ${SHM_MD_ONLY:-true}, ${SHM_CAP:-256M}); chain top $NEXT"
  elif [ "${CACHE:-1}" = 1 ]; then
    cat >> clio_server.yaml <<EOF
  - mod_name: clio_cte_cache
    pool_name: clio_cte_cache
    pool_query: local
    pool_id: "563.0"
    next_pool_id: "512.0"
    min_score: 0.5
    md_only: ${CACHE_MD_ONLY:-false}
EOF
    NEXT=563.0
  else
    NEXT=512.0
  fi
  cat >> clio_server.yaml <<EOF
  - mod_name: clio_cte_stream
    pool_name: clio_cte_stream
    pool_query: local
    pool_id: "565.0"
    next_pool_id: "$NEXT"
    staging_pool_id: "512.0"
  - mod_name: clio_cte_filesystem
    pool_name: clio_cte_filesystem
    pool_query: local
    pool_id: "560.0"
    next_pool_id: "$NEXT"
    stream_pool_id: "565.0"
$( [ "${META_LOG:-0}" = 1 ] && echo "    metadata_log_path: \"$LR/meta/cfs_namespace_log\"" )
$( [ -n "${DIR_LOG_COMPACT:-}" ] && echo "    dir_log_compact_bytes: \"$DIR_LOG_COMPACT\"" )
$( [ -n "${META_RAM_MB:-}" ] && echo "    data_score: 0.9" )
EOF
  # Per-node bring-up runs for the whole job in the background; it writes
  # $OUT/ready/<host> once the mount answers and tears down on $OUT/stop.
  mkdir -p ready logs
  mpiexec -n "$NN" -ppn 1 --no-vni --cpu-bind none --envall \
    env OUT="$OUT" BUILD_DIR="$BUILD_DIR" LR="$LR" JOB="$JOB" FLARE="$FLARE" \
        DAOS_POOL="$DAOS_POOL" DAOS_CONT="$DAOS_CONT" \
    bash "$HERE/cliofs_node.sh" > logs/node_up.log 2>&1 &
  NODES_PID=$!
  for _ in $(seq 1 "${UP_WAIT:-240}"); do
    [ "$(ls ready 2>/dev/null | wc -l)" -ge "$NN" ] && break
    kill -0 $NODES_PID 2>/dev/null || break
    sleep 1
  done
  up=$(ls ready 2>/dev/null | wc -l)
  echo "=== clio-fs up on $up/$NN nodes"
  if [ "$up" -lt "$NN" ]; then
    echo "clio-fs bring-up FAILED"; tail -40 logs/node_up.log
    stop_nodes; exit 3
  fi
  # SETTLE: seconds to wait after every mount answers. At 64 nodes pool
  # creation was still in progress on some nodes when IO500 started.
  sleep "${SETTLE:-0}"
  DATADIR=$LR/mnt/io500_$JOB
elif [ "$MODE" = daos ]; then
  daos cont query "$DAOS_POOL" "$DAOS_CONT" > /dev/null 2>&1 \
    || { cc=$(daos cont create "$DAOS_POOL" "$DAOS_CONT" --type POSIX ${_cont_opts//,/ } 2>&1); \
         echo "=== DAOS cont create $DAOS_CONT ${_cont_opts//,/ }: $(echo "$cc" | grep -v ' ERR ' | tail -2 | tr '\n' ' ')"; }
  mkdir -p ready logs
  mpiexec -n "$NN" -ppn 1 --no-vni --cpu-bind none --envall \
    env OUT="$OUT" DAOS_POOL="$DAOS_POOL" DAOS_CONT="$DAOS_CONT" \
    bash "$HERE/daos_node.sh" > logs/node_up.log 2>&1 &
  NODES_PID=$!
  for _ in $(seq 1 "${UP_WAIT:-240}"); do
    [ "$(ls ready 2>/dev/null | wc -l)" -ge "$NN" ] && break
    kill -0 $NODES_PID 2>/dev/null || break
    sleep 1
  done
  up=$(ls ready 2>/dev/null | wc -l)
  echo "=== dfuse up on $up/$NN nodes"
  if [ "$up" -lt "$NN" ]; then
    echo "dfuse bring-up FAILED"; tail -40 logs/node_up.log
    stop_nodes; exit 3
  fi
  DATADIR=/tmp/$USER/daos_io500/data/daos_${NN}n_$JOB
  # DAOS_API=dfs: IO500 talks to the container through libdfs instead of the
  # dfuse mount (the mount stays up but carries no benchmark I/O).
  if [ "${DAOS_API:-dfuse}" = dfs ]; then
    DATADIR=/io500_${NN}n_$JOB
    DFS_API="DFS --dfs.pool=$DAOS_POOL --dfs.cont=$DAOS_CONT"
  fi
else
  # FLARE PREFLIGHT: the project's Lustre quota is shared. Skip (exit 4)
  # unless FLARE_HEADROOM (1.5) x this run's data fits under the hard limit.
  if [ "${FLARE_PREFLIGHT:-1}" = 1 ]; then
    pid=$(lfs project -d "$FLARE" 2>/dev/null | awk '{print $1}')
    read -r used_kb hard_kb < <(lfs quota -p "$pid" /lus/flare 2>/dev/null \
      | awk '$1=="/lus/flare"{gsub(/\*/,""); print $2, $4}')
    need_kb=$(( DATA_MB * NN * 1024 ))
    avail_kb=$(( ${hard_kb:-0} - ${used_kb:-0} ))
    echo "=== FLARE PREFLIGHT: run needs $(( need_kb / 1048576 )) GB x${FLARE_HEADROOM:-1.5}; project quota free $(( avail_kb / 1048576 )) GB (hard limit)"
    if [ -z "$hard_kb" ] || [ "$hard_kb" = 0 ] \
       || [ "$(( avail_kb * 10 ))" -lt "$(( need_kb * ${FLARE_HEADROOM_X10:-15} ))" ]; then
      echo "=== FLARE PREFLIGHT: skipping flare run rc=4"; exit 4
    fi
  fi
  DATADIR=$FLARE/data/flare_${NN}n_$JOB
  mkdir -p "$DATADIR"
fi

DFS_API=${DFS_API:-}
API_LINE=""
[ -n "$DFS_API" ] && API_LINE="API = $DFS_API"
cat > io500.ini <<EOF
[global]
datadir = $DATADIR
${DFS_API:+api = $DFS_API}
resultdir = $OUT/results
timestamp-resultdir = FALSE
drop-caches = FALSE

[debug]
stonewall-time = $STONEWALL

[ior-easy]
blockSize = ${EASY_MB}m
$API_LINE

[ior-hard]
segmentCount = $HARD_SEG
$API_LINE

[mdtest-easy]
n = $MD_N
$API_LINE

[mdtest-hard]
n = $MD_N
$API_LINE

# Not part of the score; random 4 KiB reads ran for 20 minutes at 64 nodes.
[ior-rnd4K-easy-read]
run = FALSE
EOF
if [ -n "$DFS_API" ]; then
  # IO500's built-in pfind walks a POSIX path; a libdfs datadir has none.
  printf '\n[find]\nrun = FALSE\n' >> io500.ini
  echo "=== find skipped: no POSIX path in DAOS_API=dfs mode"
elif [ "${PFIND_HASH:-0}" = 1 ]; then
  cat >> io500.ini <<EOF

[find]
pfind-parallelize-single-dir-access-using-hashing = TRUE
EOF
  echo "=== pfind: single-directory access parallelized by hashing"
fi
echo "=== caps: ior-easy ${EASY_MB}MB/rank, ior-hard $HARD_SEG seg/rank, mdtest $MD_N files/rank"
mkdir -p results
if [ "${BENCH:-io500}" = ior ]; then
  # BENCH=ior: only the two ior-easy phases, with IO500's own arguments
  # (file per process, 2 MiB transfers, fsync before close). Minutes
  # instead of the half hour the metadata phases take.
  IOR=${IOR:-$(dirname "$IO500")/bin/ior}
  mkdir -p "$DATADIR/ior-easy" 2>/dev/null
  # IOR_VARIANTS: "fuse" (through the mount, default), "hard" and/or "preload"
  # (the POSIX interposition adapter: no kernel FUSE, same filesystem client).
  # STRACE=1: the first rank of each node runs under strace -T (ior_wrap.sh).
  # After every phase each node's runtime and adapter get SIGUSR1, which
  # prints the cumulative CLIO_EVLAT report, so phases can be told apart.
  _variants=${IOR_VARIANTS:-fuse}
  for v in ${_variants//,/ }; do  # comma- or space-separated
   if [ $v = preload ]; then
    mkdir -p "$DATADIR/ior-easy-pl" 2>/dev/null
    tgt="clio::/$(basename "$DATADIR")/ior-easy-pl/ior_file_easy"; sfx="-preload"
    pre="env LD_PRELOAD=$BUILD_DIR/bin/libclio_cte_posix.so CLIO_SERVER_CONF=$LR/clio_server.yaml CLIO_WITH_RUNTIME=0 CLIO_IPC_MODE=SHM CLIO_MEMFD_DIR=$LR/memfd"
   else
    tgt="$DATADIR/ior-easy/ior_file_easy"; sfx=""; pre=""
   fi
   # "hard": IO500's ior-hard (one shared file, 47008-byte transfers).
   shape="-t 2m -b ${EASY_MB}m -F"
   if [ $v = hard ]; then
    mkdir -p "$DATADIR/ior-hard" 2>/dev/null
    tgt="$DATADIR/ior-hard/file"; sfx="-hard"; pre=""
    shape="-t 47008 -b 47008 -s $HARD_SEG"
   fi
   # Named access patterns, EASY_MB per rank (the data footprint):
   #   fpp2m / fpp64k  file per process, 2 MiB / 64 KiB transfers
   #   shared16m / shared1m / shared64k  one shared file, segmented: each
   #     rank owns every Nth block of that size
   case $v in
    fpp2m)     shape="-t 2m -b ${EASY_MB}m -F" ;;
    fpp64k)    shape="-t 64k -b ${EASY_MB}m -F" ;;
    shared16m) shape="-t 16m -b 16m -s $(( EASY_MB / 16 ))" ;;
    shared1m)  shape="-t 1m -b 1m -s $EASY_MB" ;;
    shared64k) shape="-t 64k -b 64k -s $(( EASY_MB * 16 ))" ;;
   esac
   case $v in fpp*|shared*)
    mkdir -p "$DATADIR/ior-$v" 2>/dev/null
    tgt="$DATADIR/ior-$v/file"; sfx="-$v"; pre="" ;;
   esac
   # IOR_FSYNC=0 drops IO500's fsync before close (-e): with a DRAM tier
   # that holds the whole dataset nothing then reaches the persistent tier
   # inside the timed phase, which isolates the path into DRAM.
   fs="-e"; [ "${IOR_FSYNC:-1}" = 0 ] && fs=""
   # IOR_MINTIME=<s>: every phase runs for at least that long, starting over
   # from the beginning of each file when it reaches the end (rewrites or
   # rereads the same EASY_MB per rank, so the data footprint stays fixed),
   # with a stonewall at the same time so a phase stops at the mark.
   mintime=""
   [ -n "${IOR_MINTIME:-}" ] && mintime="-D $IOR_MINTIME -O minTimeDuration=$IOR_MINTIME"
   # IOR_WRITE_MINTIME: the write phase's minimum duration, default
   # IOR_MINTIME. 0 writes the footprint once: for DAOS, which keeps every
   # overwritten version until aggregation, so a timed rewrite fills the pool.
   wmintime=$mintime
   if [ -n "${IOR_WRITE_MINTIME:-}" ]; then
    wmintime=""
    [ "$IOR_WRITE_MINTIME" != 0 ] && wmintime="-D $IOR_WRITE_MINTIME -O minTimeDuration=$IOR_WRITE_MINTIME"
   fi
   for ph in write read; do
    [ $ph = write ] && fl="-w" || fl="-r"
    phmin=$mintime; [ $ph = write ] && phmin=$wmintime
    # PROFILE=1: sample thread CPU and stacks on this node during the phase.
    [ "${PROFILE:-0}" = 1 ] && bash "$HERE/sample_stacks.sh" "$OUT/prof$sfx-$ph" \
      "${PROFILE_DELAY:-1}" "${PROFILE_N:-8}" "${PROFILE_GAP:-0.4}" > /dev/null 2>&1 &
    wrap=""
    [ "${STRACE:-0}" = 1 ] && wrap="$HERE/ior_wrap.sh $OUT/ior$sfx-$ph $PPN"
    mpiexec -n $(( NN * PPN )) -ppn "$PPN" --cpu-bind depth -d 4 --envall \
      $wrap $pre "$IOR" -g -G 271828 -k $fs -o "$tgt" -C -Q 1 \
      $shape $fl -a POSIX $phmin ${IOR_EXTRA:-} > "ior-easy$sfx-$ph.txt" 2>&1
    rc=$?
    echo "IORCELL tag=${RUN_TAG:-none} variant=$v nodes=$NN phase=$ph rc=$rc $(grep -E '^(write|read) ' "ior-easy$sfx-$ph.txt" | awk '{print "MiB/s=" $2, "open_s=" $7, "io_s=" $8, "close_s=" $9}')"
    if [ "$MODE" = cliofs ]; then
      mpiexec -n "$NN" -ppn 1 --cpu-bind none bash -c \
        'pkill -USR1 -x clio_run; pkill -USR1 -x clio_cte_fuse; true' >/dev/null 2>&1
      sleep 3
    fi
   done
   # A named pattern's files go once both phases ran, so patterns never
   # stack up on a shared quota.
   case $v in fpp*|shared*) rm -rf "$DATADIR/ior-$v" ;; esac
  done
elif [ "${BENCH:-io500}" = mdtest ]; then
  # BENCH=mdtest: create then delete, for the per-rank-directory layout
  # (IO500 mdtest-easy) and the single shared directory (mdtest-hard).
  # PROFILE=1 samples thread CPU and stacks on this node during each delete.
  MDT=${MDT:-$(dirname "$IO500")/bin/mdtest}
  run_md() {  # <tag> <phase flag> <layout args...>
    local tag=$1 fl=$2; shift 2
    mpiexec -n $(( NN * PPN )) -ppn "$PPN" --cpu-bind depth -d 4 --envall \
      "$MDT" -n "$MD_N" -F -N 1 -P "$fl" "$@" -a POSIX > "mdtest-$tag.txt" 2>&1
    echo "MDCELL tag=${RUN_TAG:-none} nodes=$NN phase=$tag rc=$? $(grep -E 'File (creation|stat|read|removal)' "mdtest-$tag.txt" | awk '{print $1 "_" $2 "_per_s=" $4}' | tr '\n' ' ')"
  }
  mkdir -p "$DATADIR/md-easy" "$DATADIR/md-hard" 2>/dev/null
  run_md easy-create -C -u -L -d "$DATADIR/md-easy"
  run_md easy-stat -T -u -L -d "$DATADIR/md-easy"
  SP=
  [ "${PROFILE:-0}" = 1 ] && { bash "$HERE/sample_stacks.sh" "$OUT/prof_easy_delete" \
    "${PROFILE_DELAY:-2}" "${PROFILE_N:-12}" > /dev/null 2>&1 & SP=$!; }
  run_md easy-delete -r -u -L -d "$DATADIR/md-easy"
  # Only the sampler: a bare `wait` would also wait for the node scripts,
  # which run until the job tears down (this hung two jobs).
  [ -n "$SP" ] && wait "$SP"
  run_md hard-create -C -t -w 3901 -e 3901 -d "$DATADIR/md-hard"
  run_md hard-stat -T -t -d "$DATADIR/md-hard"
  run_md hard-read -E -t -e 3901 -d "$DATADIR/md-hard"
  SP=
  [ "${PROFILE:-0}" = 1 ] && { bash "$HERE/sample_stacks.sh" "$OUT/prof_hard_delete" \
    "${PROFILE_DELAY:-2}" "${PROFILE_N:-12}" > /dev/null 2>&1 & SP=$!; }
  run_md hard-delete -r -t -w 3901 -e 3901 -d "$DATADIR/md-hard"
  [ -n "$SP" ] && wait "$SP"
  rc=0
elif [ "${BENCH:-io500}" = wfcommons ]; then
  # BENCH=wfcommons: one WfBench workflow (WF_JSON) with its working
  # directory on the system under test ($DATADIR/wf). Three timed steps:
  #   stage   : the workflow's input files, written by WF_STAGE_PPN ranks
  #             per node (wfcommons/wfc_stage.py)
  #   run     : the workflow under Parsl, WF_WPN workers per node on every
  #             node (wfcommons/wfc_run.py); makespan = first submit to last
  #             task done, after K null tasks per node warmed the nodes
  #   cleanup : rm -rf of the working directory
  # WF_VENV holds wfcommons, parsl, wfbench and stress-ng.
  # WF_LOCAL_VENV=1 (default): every node unpacks $WF_VENV.tar (a slim copy
  # of the venv: no scipy/matplotlib/wfcommons) into node-local /tmp
  # (tmpfs; /dev/shm is noexec, so extension modules fail to load there)
  # and runs wfbench from it, so the thousands of Python start-ups per node
  # do not go to Flare, whose metadata load would otherwise show up in
  # every system's task times.
  WF_VENV=${WF_VENV:-/lus/flare/projects/IOWarp/llogan/wfc-venv}
  [ "${WF_UNPACKED:-0}" = 1 ] && node_snapshot "after bring-up"
  if [ "${WF_LOCAL_VENV:-1}" = 1 ] && [ "${WF_UNPACKED:-0}" = 1 ]; then
    WF_VENV=/tmp/$USER/wfc-venv
  elif [ "${WF_LOCAL_VENV:-1}" = 1 ]; then
    t0=$(date +%s.%N)
    mpiexec -n "$NN" -ppn 1 --cpu-bind none bash -c \
      "rm -rf /tmp/$USER/wfc-venv; mkdir -p /tmp/$USER && tar -C /tmp/$USER -xf $WF_VENV.tar" \
      || echo "=== venv unpack FAILED"
    WF_VENV=/tmp/$USER/wfc-venv
    echo "=== venv unpacked to $WF_VENV on $NN nodes in $(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b - a}') s"
  fi
  WF_PY=$WF_VENV/bin/python
  WF_DIR=$DATADIR/wf
  WF_NAME=$(basename "$WF_JSON" .json)
  WF_OUT=$OUT/wf_$WF_NAME
  mkdir -p "$WF_DIR" "$WF_OUT" || echo "=== cannot create $WF_DIR"
  t0=$(date +%s.%N)
  mpiexec -n $(( NN * ${WF_STAGE_PPN:-8} )) -ppn "${WF_STAGE_PPN:-8}" --cpu-bind none --envall \
    "$WF_PY" "$HERE/wfcommons/wfc_stage.py" "$WF_JSON" "$WF_DIR" $(( NN * ${WF_STAGE_PPN:-8} )) > "$WF_OUT/stage.log" 2>&1
  src=$?
  t1=$(date +%s.%N)
  st_b=$(awk '/^STAGE/{for(i=1;i<=NF;i++) if($i~/^bytes=/){split($i,a,"=");s+=a[2]}} END{print s+0}' "$WF_OUT/stage.log")
  st_n=$(grep -c '^STAGE' "$WF_OUT/stage.log")
  st_s=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b - a}')
  echo "WFSTAGE tag=${RUN_TAG:-none} mode=$MODE nodes=$NN wf=$WF_NAME rc=$src ranks=$st_n bytes=$st_b secs=$st_s GBps=$(awk -v b="$st_b" -v s="$st_s" 'BEGIN{printf "%.3f", b / s / 1e9}')"
  rc=$src
  if [ "$src" = 0 ]; then
    PATH=$WF_VENV/bin:$PATH timeout "${WF_TIMEOUT:-2400}" "$WF_PY" "$HERE/wfcommons/wfc_run.py" \
      --json "$WF_JSON" --workdir "$WF_DIR" --out "$WF_OUT" --nodes "$NN" \
      --wpn "${WF_WPN:-16}" --python "$WF_PY" ${WF_ADDRESS:+--address "$WF_ADDRESS"} \
      > "$WF_OUT/run.log" 2>&1
    rc=$?
    grep -E '^=== ' "$WF_OUT/run.log"
    tail -3 "$WF_OUT/run.log" | grep -v '^WFRESULT' | grep -v '^=== '
    res=$(grep '^WFRESULT ' "$WF_OUT/run.log" | cut -d' ' -f2-)
    if [ -n "$res" ]; then
      echo "$res" | python3 -c '
import json, sys
s = json.loads(sys.stdin.read())
print("WFCELL tag=%s mode=%s nodes=%s wf=%s rc=%s makespan_s=%s ok=%s failed=%s dep_failed=%s other_failed=%s read_GB=%.1f written_GB=%.1f agg_GBps=%s inputs_bad=%s outputs_bad=%s null_task_s=%s task_p50_s=%.2f task_p99_s=%.2f" % (
    sys.argv[1], sys.argv[2], s["nodes"], sys.argv[3], sys.argv[4], s["makespan_s"], s["ok"], s["failed"],
    s["dep_failed"], s["other_failed"], s["bytes_read"] / 1e9, s["bytes_written"] / 1e9, s["agg_GBps"],
    s["inputs_bad"], s["outputs_bad"], s["null_task_median_s"], s["task_s_all"]["p50"] or 0, s["task_s_all"]["p99"] or 0))
for k, v in s["task_s"].items():
    print("WFCAT %s n=%d p50=%.2f p90=%.2f p99=%.2f max=%.2f" % (k, v["n"], v["p50"], v["p90"], v["p99"], v["max"]))
' "${RUN_TAG:-none}" "$MODE" "$WF_NAME" "$rc"
    else
      echo "WFCELL tag=${RUN_TAG:-none} mode=$MODE nodes=$NN wf=$WF_NAME rc=$rc NO_RESULT"
      tail -30 "$WF_OUT/run.log"
    fi
  fi
  # Leftover Parsl worker pools (a timed-out driver) would keep running.
  mpiexec -n "$NN" -ppn 1 --cpu-bind none bash -c \
    'pkill -u "$USER" -f process_worker_pool; true' > /dev/null 2>&1
  t0=$(date +%s.%N)
  rm -rf "$WF_DIR"
  rmrc=$?
  [ "${WF_LOCAL_VENV:-1}" = 1 ] && mpiexec -n "$NN" -ppn 1 --cpu-bind none \
    bash -c "rm -rf /tmp/$USER/wfc-venv" > /dev/null 2>&1
  echo "WFCLEAN tag=${RUN_TAG:-none} mode=$MODE wf=$WF_NAME rm_rc=$rmrc secs=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.3f", b - a}')"
else
echo "=== IO500 start $(date)"
# HANG_DUMP_S: if IO500 is still running after this many seconds, ask every
# node for the stacks of its runtime and adapter (logs/<host>.*.stacks).
HD=""
[ -n "${HANG_DUMP_S:-}" ] && { ( sleep "$HANG_DUMP_S"; touch dump; echo "=== hang dump requested $(date)" ) & HD=$!; }
# SYNC_FS_ONLY=1: mdtest ends each phase with system("sync"), a whole-node
# flush. On Aurora that blocked in the kernel on some nodes (16 sync processes
# per node in D state for 10+ minutes with the filesystem under test idle),
# hanging the run. This puts a `sync` first in PATH that flushes only the
# filesystem under test (sync -f). It changes what the benchmark flushes, so
# use the same setting for every system being compared.
if [ "${SYNC_FS_ONLY:-0}" = 1 ]; then
  case "$MODE" in
    cliofs) SYNC_DIR=$LR/mnt ;;
    daos)   SYNC_DIR=/tmp/$USER/daos_io500 ;;
    *)      SYNC_DIR=$DATADIR ;;
  esac
  mkdir -p "$OUT/bin"
  printf '#!/bin/sh\nexec /usr/bin/sync -f %s\n' "$SYNC_DIR" > "$OUT/bin/sync"
  chmod +x "$OUT/bin/sync"
  export PATH=$OUT/bin:$PATH
  echo "=== sync limited to $SYNC_DIR"
fi
# libdfs clients reach the DAOS servers over the fabric themselves, which the
# launcher's per-job VNI blocks (NA_HOSTUNREACH); --no-vni as for dfuse.
NOVNI=""
[ -n "$DFS_API" ] && NOVNI="--no-vni"
mpiexec -n $(( NN * PPN )) -ppn "$PPN" $NOVNI --cpu-bind depth -d 4 --envall \
  "$IO500" io500.ini > io500.log 2>&1 &
MPI_PID=$!
if [ -n "$DFS_API" ]; then
  # With libdfs the ranks never exited after IO500 wrote its results (a
  # 2-node run sat until walltime). Once the summary line is out, give them
  # a minute and end the launcher.
  ( while kill -0 $MPI_PID 2>/dev/null; do
      if grep -q 'result files are stored' io500.log 2>/dev/null; then
        sleep 60; kill $MPI_PID 2>/dev/null; break
      fi
      sleep 5
    done ) &
fi
wait $MPI_PID
rc=$?
[ -n "$HD" ] && kill "$HD" 2>/dev/null
echo "=== IO500 rc=$rc end $(date)"
grep -E '^\[RESULT|^\[SCORE' io500.log results/result_summary.txt 2>/dev/null | sort -u
echo "IO500CELL mode=$MODE nodes=$NN ppn=$PPN rc=$rc $(grep -h '^\[SCORE' results/result_summary.txt io500.log 2>/dev/null | head -1)"
fi

if [ "$MODE" = cliofs ]; then
  stop_nodes
  rm -rf "$FLARE/tiers/$JOB"
elif [ "$MODE" = daos ]; then
  stop_nodes
else
  rm -rf "$DATADIR"
fi
pool_space end
exit $rc
