#!/usr/bin/env bash
#
# Scaling-campaign driver for the Gray-Scott GPU-I/O paper (study 10).
#
# Runs FOUR studies across all arms, both storage tiers, and multiple compute
# points, with hardened stale-state hygiene and full raw-log capture so a mid-run
# crash loses nothing and is resumable.
#
#   C  compute / snapshot-period : STEPS_PER {4..768} N=6400 chunks=4 snaps=12  (10 arms +async_pinned)
#                                   RAM full range; file drops 768. Charts the device/pinned crossover.
#   K  chunk-count (fixed N)      : chunks {1,4,16,64,128} N=6400 snaps=12 steps {48,96} (9 arms)
#   B  I/O size                   : N {1600..9024} chunks=16 snaps=12 steps {48,96}       (9 arms)
#   W  weak scaling (~16.8MB/chunk): chunks {1..32}, N=2048*sqrt(chunks) snaps=8 steps {48,96} (9 arms)
#   P  pooled M-sweep             : POOL {0,64,32,16,8} N=6400 chunks=128 G=8 snaps=12 steps=96 (pooled only)
#
# Arms: 9-arm scaling set = raw_inline raw_threaded hdf5_naive hdf5_inline hdf5_threaded hdf5_async
#       hostclio sync async; Study C adds async_pinned; Study P is pooled-only.
# Tiers: BOTH tiers in every study (RAM is the headline + cheap). Scaling studies (K/B/W) run at
#        TWO compute points (48, 96); Study C covers the full compute axis.
# Reps: 3 across the matrix; 5 on the HEADLINE configs (see HEADLINE[]).
# Budget: ~7h target; resumable + incremental CSV, so the tail can be trimmed live if it overruns.
#
# Run INSIDE the iowarp-kvhdf5-gpu container after building kvhdf5_e2e_tests:
#   nohup bash <this> > <RESULTS>/progress.log 2>&1 &
#
set -u

# ---------------------------------------------------------------- locations ---
BIN="${GSBENCH_BIN:-/workspace/build/bin/kvhdf5_e2e_tests}"
BINDIR="$(dirname "$BIN")"
RESULTS="${CAMPAIGN_RESULTS:-/workspace/bench_results/scaling-campaign}"
SCRATCH="${CAMPAIGN_SCRATCH:-/workspace/build/gsbench_campaign_scratch}"   # bulky data; disk; wiped per-arm
LOGDIR="$RESULTS/logs"
CSVDIR="$RESULTS/csv"
MASTER="$RESULTS/raw/master_raw.log"
TIMEOUT="${CAMPAIGN_TIMEOUT:-1200}"

if [[ ! -x "$BIN" ]]; then echo "FATAL: no binary at $BIN" >&2; exit 1; fi
mkdir -p "$LOGDIR" "$CSVDIR" "$RESULTS/raw" "$SCRATCH"
: >> "$MASTER"

# async-VOL toolchain (only the hdf5_async arm gets these on its LD path).
# Use the ENABLE_WRITE_MEMCPY=OFF build (/opt/vol-async-nomemcpy): the stock
# /opt/vol-async is built with ENABLE_WRITE_MEMCPY=ON, whose internal copy path
# livelocks in Argobots' scheduler at scale (>=12 in-flight snapshot datasets,
# 100% CPU spin in pool_is_empty, never emits GSBENCH_RESULT). The bench already
# does manual per-snapshot buffering, so it never needs that copy path anyway.
VOL_LD="/opt/vol-async-nomemcpy/lib:/opt/hdf5ts/lib:/opt/argobots/lib"
VOL_PLUGIN="/opt/vol-async-nomemcpy/lib"
VOL_CONNECTOR="async under_vol=0;under_info={}"

# ------------------------------------------------------------ runtime env -----
export CLIO_BIND_ADDR="${CLIO_BIND_ADDR:-127.0.0.1}"
export CHI_REPO_PATH="$BINDIR"
export CUDA_MODULE_LOADING=EAGER
CONF="$(mktemp)"; printf 'runtime:\n  num_threads: 1\n' > "$CONF"
export CLIO_SERVER_CONF="$CONF"

# fixed benchmark knobs
BASE_ENV=(GSBENCH_INCOMPRESSIBLE=1 GSBENCH_RAW_ODIRECT=0 GSBENCH_RAW_FSYNC=1 GSBENCH_PREWARM=1)

# ----------------------------------------------------------- hard reset -------
# The binary's comm name truncates to 15 chars ("kvhdf5_e2e_test"), so pkill -x on
# the 16-char name matches NOTHING. Kill by full path instead, then poll until dead.
hard_reset() {
    # Match the truncated 15-char comm name ("kvhdf5_e2e_test"), NOT `-f "$BIN"`:
    # the full path appears in THIS script's own cmdline, so `pkill -f "$BIN"` would
    # SIGKILL the driver itself (observed: exit 137). `-x` on the comm name is exact.
    # Zombie-aware: this container's pid 1 does NOT reap children, so timed-out runs
    # leave Z (defunct) entries. Poll on LIVE (non-Z) procs only, else every reset burns
    # its full timeout waiting on zombies that never disappear.
    pkill -9 -x kvhdf5_e2e_test 2>/dev/null
    for _ in $(seq 1 60); do
        [[ -z "$(ps -o pid,stat -C kvhdf5_e2e_tests 2>/dev/null | awk 'NR>1 && $2!~/Z/{print $1}')" ]] && break
        sleep 0.25
    done
    rm -f /dev/shm/chi_* 2>/dev/null
    rm -f "$SCRATCH"/*bdev*.dat 2>/dev/null
    rm -rf "$SCRATCH/raw_out" "$SCRATCH/hdf5_out" 2>/dev/null
    mkdir -p "$SCRATCH/raw_out" "$SCRATCH/hdf5_out"
}

# total persisted MB and a generous bdev cap (undersized cap => silent PutBlob loss)
total_mb() { echo $(( $1 * $1 * 4 * $2 / 1000000 )); }          # N, snaps
bdev_cap() { local t; t=$(total_mb "$1" "$2"); echo $(( t * 3 / 2 + 1024 )); }

# ----------------------------------------------------------- run one arm ------
# run_arm <study> <cfgkey> <rep> <arm> <tag> :  per-config GSBENCH_* already exported.
run_arm() {
    local study="$1" cfgkey="$2" rep="$3" arm="$4" tag="$5"
    local log="$LOGDIR/${study}__${cfgkey}__${arm}__rep${rep}.log"
    # Only a real result line counts. GSBENCH_POOLED is the pooled arm's descriptor,
    # printed BEFORE the work: matching it made a hung pooled run look cached/ok.
    local resultpat='GSBENCH_RESULT'

    # resumable: skip if a good result already on disk
    if [[ -f "$log" ]] && grep -qE "$resultpat" "$log"; then
        local prev; prev=$(grep -E "$resultpat" "$log" | tail -1)
        echo "SKIP  $study/$cfgkey rep$rep $arm (cached)"
        echo "SWEEP=$study CFG=$cfgkey REP=$rep ARM=$arm $prev" >> "$MASTER"
        return
    fi

    # per-arm env: async-VOL isolation + arm-specific knobs
    local -a pre=(env)
    case "$arm" in
        # GSBENCH_HDF5_ASYNC_FWAIT=0 avoids the async-VOL H5VL_async_file_wait busy-spin
        # livelock (drains via bounded H5ESwait instead). Arms run sequentially, so this
        # arm is never oversubscribed by a sibling here.
        hdf5_async) pre+=(LD_LIBRARY_PATH="$VOL_LD:$BINDIR" HDF5_PLUGIN_PATH="$VOL_PLUGIN" HDF5_VOL_CONNECTOR="$VOL_CONNECTOR" GSBENCH_HDF5_ASYNC_FWAIT=0) ;;
        *)          pre+=(-u HDF5_VOL_CONNECTOR LD_LIBRARY_PATH="$BINDIR") ;;
    esac
    case "$arm" in
        raw_inline)    pre+=(GSBENCH_RAW_INLINE=1) ;;                       # non-async synchronous (GPU idle)
        raw_threaded)  pre+=(GSBENCH_RAW_INLINE=0) ;;                       # background CPU writer
        hdf5_naive)    pre+=(GSBENCH_RAW_INLINE=1 GSBENCH_HDF5_NAIVE=1) ;;   # typical-user untuned HDF5
        hdf5_inline)   pre+=(GSBENCH_RAW_INLINE=1) ;;
        hdf5_threaded) pre+=(GSBENCH_RAW_INLINE=0) ;;
        async)         pre+=(GSBENCH_DATA_PINNED=0) ;;
        async_pinned)  pre+=(GSBENCH_DATA_PINNED=1) ;;
        pooled)        pre+=(GSBENCH_POOL="${GSBENCH_POOL:-0}") ;;
    esac

    local attempt line
    for attempt in 1 2; do
        hard_reset
        ( cd "$BINDIR" && timeout "$TIMEOUT" "${pre[@]}" "$BIN" "$tag" ) > "$log" 2>&1
        line=$(grep -E "$resultpat" "$log" | tail -1)
        [[ -n "$line" ]] && break
        echo "RETRY $study/$cfgkey rep$rep $arm (empty, attempt $attempt)"
    done
    hard_reset

    if [[ -z "$line" ]]; then
        echo "!!!! FAIL  $study/$cfgkey rep$rep $arm  (see $log)"
        echo "SWEEP=$study CFG=$cfgkey REP=$rep ARM=$arm FAILED" >> "$MASTER"
        return
    fi
    echo "ok    $study/$cfgkey rep$rep $arm :: ${line#*GSBENCH_}"
    echo "SWEEP=$study CFG=$cfgkey REP=$rep ARM=$arm $line" >> "$MASTER"
}

# map a tier -> per-config storage env (call before run_arm loop)
set_tier() {   # <tier> <N> <snaps>
    local tier="$1" N="$2" snaps="$3" cap; cap=$(bdev_cap "$N" "$snaps")
    if [[ "$tier" == "file" ]]; then
        export GSBENCH_BDEV=file GSBENCH_BDEV_CAP_MB="$cap" \
               GSBENCH_BDEV_PATH="$SCRATCH/clio_bdev.dat" \
               GSBENCH_DISK_DIR="$SCRATCH/raw_out" GSBENCH_HDF5_DIR="$SCRATCH/hdf5_out"
    else  # ram: storage-free for every arm (/dev/shm)
        export GSBENCH_BDEV=ram GSBENCH_BDEV_CAP_MB="$cap" \
               GSBENCH_BDEV_PATH="$SCRATCH/clio_bdev.dat" \
               GSBENCH_DISK_DIR=/dev/shm/gsbench_raw GSBENCH_HDF5_DIR=/dev/shm/gsbench_hdf5
        mkdir -p /dev/shm/gsbench_raw /dev/shm/gsbench_hdf5
    fi
}

# Study C carries async_pinned (to chart the device/pinned crossover). The scaling studies
# (K/B/W) run at ONE compute point where device placement is optimal, so they use 9 arms.
# pooled appears ONLY in Study P (its own memory-vs-throughput study). CUR_ARMS is set per study.
ARMS_C=(raw_inline raw_threaded hdf5_naive hdf5_inline hdf5_threaded hdf5_async hostclio sync async async_pinned)
ARMS_SCALE=(raw_inline raw_threaded hdf5_naive hdf5_inline hdf5_threaded hdf5_async hostclio sync async)
CUR_ARMS=("${ARMS_SCALE[@]}")
tag_of() { case "$1" in
    raw_inline|raw_threaded) echo '[gsbench_raw]';; hdf5_inline|hdf5_threaded) echo '[gsbench_hdf5]';;
    hdf5_naive) echo '[gsbench_hdf5_naive]';;
    hdf5_async) echo '[gsbench_hdf5_async]';; hostclio) echo '[gsbench_hostclio]';;
    sync) echo '[gsbench_sync]';; async|async_pinned) echo '[gsbench_async]';;
    pooled) echo '[gsbench_pooled]';; esac; }

# run every arm in CUR_ARMS for one fully-exported config, at rep
run_all_arms() {   # <study> <cfgkey> <rep>
    local study="$1" cfgkey="$2" rep="$3" a
    for a in "${CUR_ARMS[@]}"; do
        unset GSBENCH_POOL
        run_arm "$study" "$cfgkey" "$rep" "$a" "$(tag_of "$a")"
    done
}

# HEADLINE configs get 5 reps instead of 3 (the points that anchor the key figures)
HEADLINE=(
    "C__steps8_ram"      # the async I/O-bound headline (your reference regime)
    "C__steps8_file"     # its durable-disk contrast
    "C__steps48_file"
    "B__N6400_steps96_file"
    "W__chunks16_steps96_file"
)
reps_for() { local k="$1" h; for h in "${HEADLINE[@]}"; do [[ "$k" == "$h" ]] && { echo 5; return; }; done; echo 3; }

isqrt2048() { # N = round(2048*sqrt(chunks)) with exact integer table (keeps N%chunks==0)
    case "$1" in 1) echo 2048;; 2) echo 2896;; 4) echo 4096;; 8) echo 5792;;
                16) echo 8192;; 32) echo 11584;; *) echo 0;; esac; }

echo "################ CAMPAIGN START $(cat /proc/uptime | awk '{print $1}')s uptime ################"
echo "results=$RESULTS  scratch=$SCRATCH  timeout=${TIMEOUT}s"

# =============================== STUDY C: compute / snapshot-period ============
# The async-advantage curve + the async device/pinned crossover (10 arms incl async_pinned).
# Both tiers; RAM gets the full steps range (cheap), file drops the priciest point (768).
CUR_ARMS=("${ARMS_C[@]}")
for tier in file ram; do
    for steps in 4 8 12 24 48 96 192 384 768; do
        [[ "$tier" == "file" && "$steps" == "768" ]] && continue
        N=6400; chunks=4; snaps=12
        export GSBENCH_N=$N GSBENCH_CHUNKS=$chunks GSBENCH_SUBMIT_BLOCKS=$chunks \
               GSBENCH_SNAPS=$snaps GSBENCH_STEPS_PER=$steps "${BASE_ENV[@]}"
        set_tier "$tier" "$N" "$snaps"
        cfg="steps${steps}_${tier}"
        R=$(reps_for "C__$cfg")
        echo "==== C  $cfg  (N=$N chunks=$chunks snaps=$snaps reps=$R) ===="
        for r in $(seq 1 "$R"); do run_all_arms C "$cfg" "$r"; done
    done
done

# ===================== STUDY K: chunk-count sweep (fixed N, strong scaling) =====
# Fixed N=6400, vary chunks: where hdf5_naive's penalty appears (small chunks) and async's
# submit-parallelism scales. steps=96 (representative), both tiers. chunks<=128 keeps
# chunks*snaps < 2048 (the verified-safe async in-flight ceiling).
CUR_ARMS=("${ARMS_SCALE[@]}")
for tier in file ram; do
  for steps in 48 96; do
    for chunks in 1 4 16 64 128; do
        N=6400; snaps=12
        export GSBENCH_N=$N GSBENCH_CHUNKS=$chunks GSBENCH_SUBMIT_BLOCKS=$chunks \
               GSBENCH_SNAPS=$snaps GSBENCH_STEPS_PER=$steps "${BASE_ENV[@]}"
        set_tier "$tier" "$N" "$snaps"
        cfg="chunks${chunks}_steps${steps}_${tier}"
        R=$(reps_for "K__$cfg")
        echo "==== K  $cfg  (N=$N chunks=$chunks snaps=$snaps steps=$steps reps=$R) ===="
        for r in $(seq 1 "$R"); do run_all_arms K "$cfg" "$r"; done
    done
  done
done

# =============================== STUDY B: I/O size ============================
# Vary bytes/snapshot at two representative compute points (48, 96); both tiers.
CUR_ARMS=("${ARMS_SCALE[@]}")
for tier in file ram; do
  for steps in 48 96; do
    for N in 1600 3200 4800 6400 9024; do
        chunks=16; snaps=12
        export GSBENCH_N=$N GSBENCH_CHUNKS=$chunks GSBENCH_SUBMIT_BLOCKS=$chunks \
               GSBENCH_SNAPS=$snaps GSBENCH_STEPS_PER=$steps "${BASE_ENV[@]}"
        set_tier "$tier" "$N" "$snaps"
        cfg="N${N}_steps${steps}_${tier}"
        R=$(reps_for "B__$cfg")
        echo "==== B  $cfg  (chunks=$chunks snaps=$snaps steps=$steps reps=$R) ===="
        for r in $(seq 1 "$R"); do run_all_arms B "$cfg" "$r"; done
    done
  done
done

# =============================== STUDY W: weak scaling ========================
# Scale chunks + N together (bytes/chunk ~16.8MB const); two compute points (48, 96), both tiers.
CUR_ARMS=("${ARMS_SCALE[@]}")
for tier in file ram; do
  for steps in 48 96; do
    for chunks in 1 2 4 8 16 32; do
        N=$(isqrt2048 "$chunks"); snaps=8
        export GSBENCH_N=$N GSBENCH_CHUNKS=$chunks GSBENCH_SUBMIT_BLOCKS=$chunks \
               GSBENCH_SNAPS=$snaps GSBENCH_STEPS_PER=$steps "${BASE_ENV[@]}"
        set_tier "$tier" "$N" "$snaps"
        cfg="chunks${chunks}_steps${steps}_${tier}"
        R=$(reps_for "W__$cfg")
        echo "==== W  $cfg  (N=$N bytes/chunk~16.8MB snaps=$snaps steps=$steps reps=$R) ===="
        for r in $(seq 1 "$R"); do run_all_arms W "$cfg" "$r"; done
    done
  done
done

# =============================== STUDY P: pooled M-sweep ======================
# chunks=128 stream through M resident buffers by G=8 blocks at depth D=M/G.
# POOL=0 => M==chunks==128 (fused control). Report GSBENCH_POOLED lines + async ref.
for tier in file ram; do
    for M in 0 64 32 16 8; do
        N=6400; chunks=128; G=8; snaps=12; steps=96
        export GSBENCH_N=$N GSBENCH_CHUNKS=$chunks GSBENCH_SUBMIT_BLOCKS=$G \
               GSBENCH_SNAPS=$snaps GSBENCH_STEPS_PER=$steps "${BASE_ENV[@]}"
        set_tier "$tier" "$N" "$snaps"
        cfg="M${M}_${tier}"
        echo "==== P  $cfg  (chunks=$chunks G=$G snaps=$snaps reps=3) ===="
        for r in 1 2 3; do
            export GSBENCH_POOL=$M
            run_arm P "$cfg" "$r" pooled '[gsbench_pooled]'
        done
        unset GSBENCH_POOL
    done
    # one async fire-all reference per tier
    export GSBENCH_N=6400 GSBENCH_CHUNKS=128 GSBENCH_SUBMIT_BLOCKS=8 GSBENCH_SNAPS=12 GSBENCH_STEPS_PER=96 "${BASE_ENV[@]}"
    set_tier "$tier" 6400 12
    for r in 1 2 3; do run_arm P "async_ref_${tier}" "$r" async '[gsbench_async]'; done
done

hard_reset
rm -f "$CONF"
rm -rf /dev/shm/gsbench_raw /dev/shm/gsbench_hdf5 2>/dev/null
echo "################ CAMPAIGN DONE ################"
echo "raw log: $MASTER   ($(grep -c '^SWEEP=' "$MASTER") result lines)"
