#!/usr/bin/env bash
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -l daos=daos_user
#PBS -q debug
#PBS -A IOWarp
#PBS -j oe
#
# Storage-tier probe, one node, well under five minutes: can this job see the
# project's DAOS pool, mount a POSIX container with dfuse, and write to it and
# to Flare at a rate that makes an 8 GB-per-node tier feasible in a 4-minute
# window? Answers those with a 1 GB dd each way and nothing else.
#
#   DAOS_POOL  (default IOWarp)   DAOS_CONT (default clio_tier)
set -u
DAOS_POOL=${DAOS_POOL:-IOWarp}
DAOS_CONT=${DAOS_CONT:-clio_tier}
FLARE_DIR=${FLARE_DIR:-/lus/flare/projects/IOWarp/clio_tier}

echo "=== tier probe on $(hostname) ==="
# PBS runs this script in a non-login bash where `module` is not defined, so
# a bare `module load` did nothing: daos then looked for the agent socket in
# /var/run/daos_agent instead of the DAOS_AGENT_DRPC_DIR the module sets, and
# launch-dfuse.sh was not on PATH. Initialise Lmod first, and say what took.
source /usr/share/lmod/lmod/init/bash
module use /soft/modulefiles
module load daos/base
echo "daos env: launch-dfuse.sh=$(command -v launch-dfuse.sh || echo MISSING) agent_dir=${DAOS_AGENT_DRPC_DIR:-unset} socket=$(ls "${DAOS_AGENT_DRPC_DIR:-/run/daos_agent_oneScratch}" 2>&1 | tr '\n' ' ')"
echo "--- daos pool ---"
daos pool query "${DAOS_POOL}" 2>&1 | head -12
echo "--- daos container ${DAOS_CONT} ---"
if ! daos cont query "${DAOS_POOL}" "${DAOS_CONT}" > /dev/null 2>&1; then
  daos cont create --type=POSIX "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | tail -3
fi
daos cont query "${DAOS_POOL}" "${DAOS_CONT}" 2>&1 | head -6
echo "--- dfuse ---"
clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1
launch-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" 2>&1 | tail -2
MNT="/tmp/${DAOS_POOL}/${DAOS_CONT}"
mount | grep -F "${MNT}" || echo "NOT MOUNTED: ${MNT}"
df -h "${MNT}" 2>&1 | tail -1

echo "--- 1 GB write/read: DAOS (dfuse) ---"
mkdir -p "${MNT}/probe"
dd if=/dev/zero of="${MNT}/probe/$(hostname).dat" bs=16M count=64 oflag=direct conv=fsync 2>&1 | tail -1
dd if="${MNT}/probe/$(hostname).dat" of=/dev/null bs=16M iflag=direct 2>&1 | tail -1
rm -f "${MNT}/probe/$(hostname).dat"

echo "--- 1 GB write/read: Flare ---"
mkdir -p "${FLARE_DIR}"
dd if=/dev/zero of="${FLARE_DIR}/probe_$(hostname).dat" bs=16M count=64 oflag=direct conv=fsync 2>&1 | tail -1
dd if="${FLARE_DIR}/probe_$(hostname).dat" of=/dev/null bs=16M iflag=direct 2>&1 | tail -1
rm -f "${FLARE_DIR}/probe_$(hostname).dat"
df -h "${FLARE_DIR}" | tail -1

clean-dfuse.sh "${DAOS_POOL}:${DAOS_CONT}" > /dev/null 2>&1
echo "RESULT probe: done"
