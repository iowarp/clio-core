#!/bin/bash
# Node 1 entrypoint — collective benchmark launcher.
#
# Node 1 is the primary: it sets up passwordless SSH (OpenMPI's default
# launcher), starts its local clio daemon, waits for nodes 2-4 to serve sshd,
# then runs `mpirun -np 4` across all four containers. Rank 0 runs here; ranks
# 1-3 are launched on nodes 2-4 over SSH, and each rank attaches to its OWN
# local daemon. This script's exit code is the benchmark's result.
set -u

CB_DIR=/workspace/context-runtime/test/integration/collective_bench
SSH_SHARE="${CB_DIR}/.mpi_ssh"
export CLIO_SERVER_CONF="${CB_DIR}/clio_conf.yaml"
export COLL_BENCH_ITERS="${COLL_BENCH_ITERS:-1000}"
export COLL_BENCH_WARMUP="${COLL_BENCH_WARMUP:-100}"
# Barrier-semantics check: rounds of staggered arrivals run before the timed
# arms. Set ROUNDS=0 to skip it.
export COLL_BENCH_VERIFY_ROUNDS="${COLL_BENCH_VERIFY_ROUNDS:-4}"
export COLL_BENCH_STAGGER_MS="${COLL_BENCH_STAGGER_MS:-20}"
# Write the CSV inside the container (/tmp), not into the mounted workspace:
# the mount is owned by the host user and the container user may not be able
# to create files there. run_tests.sh docker-cp's it out afterwards.
export COLL_BENCH_CSV="${COLL_BENCH_CSV:-/tmp/coll_bench_results.csv}"

NUM_NODES="${NUM_NODES:-4}"

echo '=== Node 1: SSH setup ==='
# Start from a clean share so a previous run's "done" flag or stale key cannot
# make the workers exit before this run's mpirun reaches them.
sudo rm -rf "${SSH_SHARE}"
sudo mkdir -p "${SSH_SHARE}" && sudo chmod 777 "${SSH_SHARE}"
mkdir -p /home/iowarp/.ssh && chmod 700 /home/iowarp/.ssh
ssh-keygen -t rsa -b 2048 -N '' -f /home/iowarp/.ssh/id_rsa -q 2>/dev/null || true
cat /home/iowarp/.ssh/id_rsa.pub >> /home/iowarp/.ssh/authorized_keys
chmod 600 /home/iowarp/.ssh/authorized_keys
printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' \
    >> /home/iowarp/.ssh/config
# Publish node1's key so the workers can authorize us (via the shared workspace).
cp /home/iowarp/.ssh/id_rsa.pub "${SSH_SHARE}/node1.pub"
sudo /usr/sbin/sshd

echo '=== Node 1: starting local clio daemon ==='
# Daemon output goes to a file: at info level it emits a periodic scheduler
# report that would otherwise bury this script's progress in `docker logs`.
CLIO_COLL_PROF=${CLIO_COLL_PROF:-} CLIO_NET_QPROF=${CLIO_NET_QPROF:-} CLIO_WORKER_RATE=${CLIO_WORKER_RATE:-} /workspace/build/bin/clio_run runtime start > /tmp/clio_daemon.log 2>&1 &
echo "Node 1: daemon PID $!"

echo "=== Node 1: waiting for worker SSH (nodes 2..${NUM_NODES}) ==="
for n in $(seq 2 "${NUM_NODES}"); do
    READY=0
    for _ in $(seq 1 90); do
        if ssh -o ConnectTimeout=3 -o BatchMode=yes "iowarp-node${n}" 'echo ready' >/dev/null 2>&1; then
            READY=1; break
        fi
        sleep 1
    done
    if [ "${READY}" = "0" ]; then
        echo "ERROR: iowarp-node${n} SSH not ready after 90s"
        touch "${SSH_SHARE}/done"
        exit 1
    fi
    echo "Node 1: iowarp-node${n} reachable"
done

# Give all daemons a moment to form the cluster before any rank routes to it.
sleep 8

MPIRUN_ARGS=(
    -np "${NUM_NODES}" --hostfile "${CB_DIR}/mpi_hostfile"
    -x LD_LIBRARY_PATH -x PATH -x CLIO_SERVER_CONF
    -x COLL_BENCH_ITERS -x COLL_BENCH_WARMUP -x COLL_BENCH_CSV
    -x COLL_BENCH_VERIFY_ROUNDS -x COLL_BENCH_STAGGER_MS
    -x OMPI_ALLOW_RUN_AS_ROOT -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
)

echo "=== Node 1: mpirun clio_collective_bench (iters=${COLL_BENCH_ITERS}) ==="
mpirun "${MPIRUN_ARGS[@]}" /workspace/build/bin/clio_collective_bench
RC=$?
echo "=== Node 1: benchmark exit ${RC} ==="

# Signal the workers to stop serving sshd so their containers exit.
touch "${SSH_SHARE}/done"
exit $RC
