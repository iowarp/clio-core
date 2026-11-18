#!/usr/bin/env bash
# One per node (mpiexec -ppn 1) from pbs_io500_aurora.sh MODE=daos: mount
# DAOS with dfuse at /tmp/$USER/daos_io500, report $OUT/ready/<host>, wait
# for $OUT/stop, remove this job's data (first node only) and unmount.
# Core dumps: a crashed runtime writes several GB per node into the run
# directory on Flare; at hundreds of nodes that exceeds the project quota
# and kills every run after it. CORE_DUMPS=1 keeps them for debugging.
[ "${CORE_DUMPS:-0}" = 1 ] || ulimit -c 0
set -u
H=$(hostname)
DM=/tmp/$USER/daos_io500
JOB=${PBS_JOBID%%.*}
log() { echo "[$H $(date +%T)] $*"; }

# A dfuse that died with an earlier job (or run) leaves a dead mount: stat
# fails with EINVAL / ENOTCONN, mkdir cannot create the path and every new
# dfuse fails "Error resolving mount point". Lazily unmount it first.
stat "$DM" > /dev/null 2>&1 || fusermount3 -u -z "$DM" 2>/dev/null
# dfuse processes of an earlier run whose mount is gone hold the container
# open; end them so the mount below starts clean.
mountpoint -q "$DM" || pkill -u "$USER" -f "dfuse .*-m $DM" 2>/dev/null
mkdir -p "$DM"
for t in 1 2 3 4 5; do
  mountpoint -q "$DM" && break
  if [ $t -le 3 ]; then DOPTS="--thread-count=32 --eq-count=16"; else DOPTS=""; fi
  dfuse --pool "$DAOS_POOL" --cont "$DAOS_CONT" -m "$DM" $DOPTS \
    2>&1 | grep -v ' ERR ' | tail -3; mountpoint -q "$DM" && break
  log "dfuse attempt $t failed"; sleep 20
done
for _ in $(seq 1 20); do mountpoint -q "$DM" && break; sleep 0.5; done
mountpoint -q "$DM" || { log "dfuse not mounted"; exit 2; }
mkdir -p "$DM/data"
touch "$OUT/ready/$H"; log "up"

while [ ! -e "$OUT/stop" ]; do sleep 2; done
[ "$H" = "$(head -1 "$OUT/hostfile")" ] && rm -rf "$DM"/data/daos_*_"$JOB"
fusermount3 -u "$DM" 2>/dev/null || fusermount3 -u -z "$DM" 2>/dev/null
log "down"
