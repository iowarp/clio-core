#!/usr/bin/env bash
# One per node (mpiexec -ppn 1) from pbs_io500_aurora.sh: mount DAOS with
# dfuse, start clio_run with this node's config, mount clio_cte_fuse at
# $LR/mnt, report $OUT/ready/<host>, wait for $OUT/stop, tear down.
# EMBED=1 runs the runtime inside clio_cte_fuse instead of a separate clio_run.
set -u
H=$(hostname)
LOG=$OUT/logs/$H
DM=/tmp/$USER/daos_io500
log() { echo "[$H $(date +%T)] $*"; }

pkill -9 -u "$USER" -f '[c]lio_cte_fuse' ; pkill -9 -u "$USER" -f '[c]lio_run' ; sleep 0.5
fusermount3 -u -z "$LR/mnt" 2>/dev/null
rm -rf "$LR"; mkdir -p "$LR/mnt" "$LR/conf" "$LR/memfd" "$LR/meta" "$DM"

for t in 1 2 3 4 5; do
  mountpoint -q "$DM" && break
  # Caching ON: each node's DAOS tier file is private to that node, so there
  # is no cross-node coherence to protect, and with caching disabled every
  # small bdev write was a synchronous round trip (seconds each at 16 nodes).
  if [ $t -le 3 ]; then DOPTS="--thread-count=32 --eq-count=16"; else DOPTS=""; fi
  dfuse --pool "$DAOS_POOL" --cont "$DAOS_CONT" -m "$DM" $DOPTS \
    2>&1 | grep -v ' ERR ' | tail -3; mountpoint -q "$DM" && break
  log "dfuse attempt $t failed"; sleep 20
done
for _ in $(seq 1 20); do mountpoint -q "$DM" && break; sleep 0.5; done
mountpoint -q "$DM" || { log "dfuse not mounted"; exit 2; }
mkdir -p "$DM/tiers/$JOB"
# Killed jobs never reach their teardown: reclaim tier files of jobs that
# ended more than two hours ago (first node only; the container is shared).
[ "$H" = "$(head -1 "$OUT/hostfile")" ] && \
  find "$DM/tiers" -mindepth 1 -maxdepth 1 -mmin +120 ! -name "$JOB" -exec rm -rf {} + 2>/dev/null

sed "s/__HOST__/$H/g" "$OUT/clio_server.yaml" > "$LR/clio_server.yaml"
export CLIO_SERVER_CONF=$LR/clio_server.yaml
export CLIO_WITH_RUNTIME=0 CLIO_IPC_MODE=SHM CLIO_MEMFD_DIR=$LR/memfd
export CLIO_BIND_ADDR=0.0.0.0 CTP_LOG_LEVEL=${CTP_LOG_LEVEL:-warning}
# bdev perf-stat persistence writes ~/.clio/bdev_perf (shared home) from a
# runtime worker; under load that blocked workers for seconds, peers missed
# liveness probes and declared the node dead.
export CLIO_BDEV_PERSIST_STATS=0
# Tier files take only buffered I/O: concurrent O_DIRECT writes to the Flare
# tier failed with EIO once requests ran in parallel (see posix_aio.h).
export CTP_POSIX_AIO_DIRECT=${CTP_POSIX_AIO_DIRECT:-0}
# Readdir stat prefetch (N getattrs in flight per large listing). Off: readdir
# now reports entry types, so pfind stats only its matches and the prefetch
# was 2.6 s of a 4 s find at 64 nodes. Set to 256 for walkers that stat all.
export CLIO_FUSE_READDIR_PREFETCH=${CLIO_FUSE_READDIR_PREFETCH:-0}
export LD_LIBRARY_PATH=$BUILD_DIR/bin:${LD_LIBRARY_PATH:-}

# EMBED=1: no separate clio_run. The adapter starts the runtime inside its
# own process (CLIO_WITH_RUNTIME=1), so a FUSE callback reaches the runtime
# without crossing a process boundary. Its runtime log is the fuse log.
EMBED=${EMBED:-0}
RT=
if [ "$EMBED" != 1 ]; then
  "$BUILD_DIR/bin/clio_run" start --fresh --no-viz </dev/null > "$LOG.runtime.log" 2>&1 &
  RT=$!
  for _ in $(seq 1 180); do
    ss -ltn | grep -q ':9519 ' && break
    kill -0 $RT 2>/dev/null || { log "clio_run exited"; tail -20 "$LOG.runtime.log"; exit 3; }
    sleep 1
  done
  sleep 5
fi
# Per-callback FUSE timing every FUSE_PROFILE seconds (0 = off). The report
# is logged at info level, so only the adapter's log level is raised.
FUSE_PROFILE=${FUSE_PROFILE:-10}
if [ "$FUSE_PROFILE" != 0 ]; then
  FUSE_ENV="CLIO_FUSE_PROFILE=$FUSE_PROFILE CTP_LOG_LEVEL=info"
else
  FUSE_ENV="CTP_LOG_LEVEL=$CTP_LOG_LEVEL"
fi
[ "$EMBED" = 1 ] && FUSE_ENV="$FUSE_ENV CLIO_WITH_RUNTIME=1"
env $FUSE_ENV "$BUILD_DIR/bin/clio_cte_fuse" "$LR/mnt" -f </dev/null > "$LOG.fuse.log" 2>&1 &
FU=$!
for _ in $(seq 1 ${MOUNT_WAIT:-90}); do
  grep -q " $LR/mnt " /proc/self/mountinfo && timeout 20 stat -f "$LR/mnt" >/dev/null 2>&1 && break
  sleep 1
done
if grep -q " $LR/mnt " /proc/self/mountinfo; then
  touch "$OUT/ready/$H"; log "up"
else
  log "mount failed"; tail -20 "$LOG.fuse.log"
fi

# Where the data actually landed: allocated size of this node's DAOS and
# Flare tier files every 30 s (the DRAM tier is whatever is not here).
( while [ ! -e "$OUT/stop" ]; do
    echo "$(date +%T) daos_kb=$(du -sk "$DM/tiers/$JOB/$H"* 2>/dev/null | awk '{s+=$1} END{print s+0}') flare_kb=$(du -sk "$FLARE/tiers/$JOB/$H"* 2>/dev/null | awk '{s+=$1} END{print s+0}') flare_free_kb=$(df -k --output=avail "$FLARE/tiers/$JOB" 2>/dev/null | tail -1 | tr -d ' ')"
    sleep 30
  done > "$LOG.tiers.log" 2>&1 ) &
while [ ! -e "$OUT/stop" ]; do
  [ -z "$RT" ] || kill -0 $RT 2>/dev/null || { log "clio_run DIED"; tail -20 "$LOG.runtime.log"; break; }
  kill -0 $FU 2>/dev/null || { log "clio_cte_fuse DIED"; tail -20 "$LOG.fuse.log"; break; }
  sleep 2
done
timeout 30 fusermount3 -u "$LR/mnt" 2>/dev/null || fusermount3 -u -z "$LR/mnt"
[ -n "$RT" ] && timeout 60 "$BUILD_DIR/bin/clio_run" stop >/dev/null 2>&1
kill -9 $FU $RT 2>/dev/null
rm -rf "$DM/tiers/$JOB"
fusermount3 -u "$DM" 2>/dev/null
log "down"
