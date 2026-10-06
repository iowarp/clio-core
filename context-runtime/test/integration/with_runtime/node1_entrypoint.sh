#!/bin/bash
# Node 1 entrypoint — "Clio as a cache" test launcher (issue #1015).
#
# Node 1 is the primary: it sets up passwordless SSH (OpenMPI's default
# launcher), waits for node 2's sshd, then runs `mpirun -np 8` across both
# containers. It deliberately does NOT start a clio daemon — the whole point of
# the test is that the ranks bring the runtime up themselves. This script's exit
# code is the MPI job result.
set -u

WR_DIR=/workspace/context-runtime/test/integration/with_runtime
SSH_SHARE="${WR_DIR}/.mpi_ssh"
export CLIO_SERVER_CONF="${WR_DIR}/clio_config.yaml"
export CLIO_WITH_RUNTIME=1

echo '=== Node 1: SSH setup ==='
mkdir -p /home/iowarp/.ssh && chmod 700 /home/iowarp/.ssh
ssh-keygen -t rsa -b 2048 -N '' -f /home/iowarp/.ssh/id_rsa -q 2>/dev/null || true
cat /home/iowarp/.ssh/id_rsa.pub >> /home/iowarp/.ssh/authorized_keys
chmod 600 /home/iowarp/.ssh/authorized_keys
printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' \
    >> /home/iowarp/.ssh/config
# Publish node1's key so node2 can authorize us (via the shared workspace).
sudo mkdir -p "${SSH_SHARE}" && sudo chmod 777 "${SSH_SHARE}"
rm -f "${SSH_SHARE}/done"
cp /home/iowarp/.ssh/id_rsa.pub "${SSH_SHARE}/node1.pub"
sudo /usr/sbin/sshd

echo '=== Node 1: waiting for node2 SSH ==='
NODE2_READY=0
for i in $(seq 1 90); do
    if ssh -o ConnectTimeout=3 -o BatchMode=yes iowarp-wr-node2 'echo ready' 2>/dev/null; then
        NODE2_READY=1; break
    fi
    sleep 1
done
if [ "$NODE2_READY" = "0" ]; then
    echo 'ERROR: node2 SSH not ready after 90s'
    touch "${SSH_SHARE}/done"
    exit 1
fi

# 4 ranks per node (mpi_hostfile slots=4); ranks 0-3 -> node1, 4-7 -> node2.
echo '=== Node 1: mpirun -np 8 test_with_runtime_cache ==='
mpirun -np 8 --hostfile "${WR_DIR}/mpi_hostfile" \
    -x LD_LIBRARY_PATH -x PATH -x CLIO_SERVER_CONF -x CLIO_WITH_RUNTIME \
    -x OMPI_ALLOW_RUN_AS_ROOT -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM \
    /workspace/build/bin/test_with_runtime_cache
RC=$?
echo "=== Node 1: test_with_runtime_cache exit ${RC} ==="

# Signal node2 to stop serving sshd (so its container exits and compose returns).
touch "${SSH_SHARE}/done"
exit $RC
