#!/usr/bin/env bash
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A IOWarp
#PBS -j oe
#
# A held compute node that executes scripts dropped into a queue directory,
# so a debugging session can run many short experiments in one allocation
# instead of waiting in the queue for each.
#
#   DEVNODE_DIR/queue/<name>.sh   -> run with bash; stdout+stderr in
#   DEVNODE_DIR/out/<name>.out    ; then moved to DEVNODE_DIR/done/
#   DEVNODE_DIR/host              the node's hostname, written at start
#   DEVNODE_DIR/stop              stops the loop early
set -u
D=${DEVNODE_DIR:?set DEVNODE_DIR}
mkdir -p "${D}/queue" "${D}/running" "${D}/out" "${D}/done"
hostname > "${D}/host"
echo "devnode $(hostname) up at $(date -u +%H:%M:%S) UTC, job ${PBS_JOBID}"
budget=${DEVNODE_BUDGET_S:-3480}
while [ "${SECONDS}" -lt "${budget}" ] && [ ! -e "${D}/stop" ]; do
  for q in "${D}"/queue/*.sh; do
    [ -e "${q}" ] || continue
    name=$(basename "${q}" .sh)
    mv "${q}" "${D}/running/${name}.sh"
    echo "=== ${name} start $(date -u +%H:%M:%S)"
    bash "${D}/running/${name}.sh" > "${D}/out/${name}.out" 2>&1
    echo "rc=$?" >> "${D}/out/${name}.out"
    mv "${D}/running/${name}.sh" "${D}/done/${name}.sh"
    echo "=== ${name} done $(date -u +%H:%M:%S)"
  done
  sleep 3
done
echo "devnode down at $(date -u +%H:%M:%S) UTC"
