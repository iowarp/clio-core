#!/bin/bash
# Node 1 entrypoint — MPI producer/consumer test launcher (issue #714).
#
# Node 1 is the primary: it sets up passwordless SSH (OpenMPI's default
# launcher), starts its local clio daemon, waits for node 2's daemon+sshd, then
# runs `mpirun -np 2` across both containers. Rank 0 (producer) runs here, rank 1
# (consumer) is launched on node 2 over SSH; each rank attaches to its own local
# daemon. The producer/consumer binaries carry the actual #714 assertions; this
# script's exit code is the combined MPI job result.
set -u

PC_DIR=/workspace/context-transfer-engine/test/integration/mpi_producer_consumer
SSH_SHARE="${PC_DIR}/.mpi_ssh"
export CLIO_SERVER_CONF="${PC_DIR}/clio_config.yaml"
# Keep the producer's post-write keep-alive short: the MPI barrier already
# orders write-before-read, and the blob lives in the (independently running)
# daemon, not the rank process.
export PC_KEEPALIVE_SEC="${PC_KEEPALIVE_SEC:-5}"

echo '=== Node 1: SSH setup ==='
mkdir -p /home/iowarp/.ssh && chmod 700 /home/iowarp/.ssh
ssh-keygen -t rsa -b 2048 -N '' -f /home/iowarp/.ssh/id_rsa -q 2>/dev/null || true
cat /home/iowarp/.ssh/id_rsa.pub >> /home/iowarp/.ssh/authorized_keys
chmod 600 /home/iowarp/.ssh/authorized_keys
printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' \
    >> /home/iowarp/.ssh/config
# Publish node1's key so node2 can authorize us (via the shared workspace).
sudo mkdir -p "${SSH_SHARE}" && sudo chmod 777 "${SSH_SHARE}"
cp /home/iowarp/.ssh/id_rsa.pub "${SSH_SHARE}/node1.pub"
sudo /usr/sbin/sshd

echo '=== Node 1: starting local clio daemon ==='
/workspace/build/bin/clio_run runtime start &
echo "Node 1: daemon PID $!"

echo '=== Node 1: waiting for node2 SSH ==='
NODE2_READY=0
for i in $(seq 1 90); do
    if ssh -o ConnectTimeout=3 -o BatchMode=yes iowarp-node2 'echo ready' 2>/dev/null; then
        NODE2_READY=1; break
    fi
    sleep 1
done
if [ "$NODE2_READY" = "0" ]; then
    echo 'ERROR: node2 SSH not ready after 90s'; exit 1
fi
# Give both daemons a moment to form the 2-node cluster.
sleep 6

# The two producer/consumer binaries share the same cluster; run both and
# combine their results. Rank 0 -> node1, rank 1 -> node2 (mpi_hostfile).
MPIRUN_ARGS=(
    -np 2 --hostfile "${PC_DIR}/mpi_hostfile"
    -x LD_LIBRARY_PATH -x PATH -x CLIO_SERVER_CONF -x PC_KEEPALIVE_SEC
    -x OMPI_ALLOW_RUN_AS_ROOT -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
)

OVERALL=0
for bin in test_cte_producer_consumer test_cfs_producer_consumer; do
    echo "=== Node 1: mpirun ${bin} ==="
    mpirun "${MPIRUN_ARGS[@]}" "/workspace/build/bin/${bin}"
    rc=$?
    echo "=== Node 1: ${bin} exit ${rc} ==="
    [ "$rc" != "0" ] && OVERALL=$rc
done

# Signal node2 to stop serving sshd (so its container exits and compose returns).
touch "${SSH_SHARE}/done"
echo "=== Node 1: overall exit ${OVERALL} ==="
exit $OVERALL
