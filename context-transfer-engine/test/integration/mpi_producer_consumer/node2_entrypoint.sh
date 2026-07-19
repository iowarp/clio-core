#!/bin/bash
# Node 2 entrypoint — MPI producer/consumer test (issue #714).
#
# Node 2 is secondary: it starts its local clio daemon, authorizes node 1's SSH
# key, and serves sshd so mpirun (launched on node 1) can start rank 1 (the
# consumer) here. It does no launching itself — it stays up until node 1 signals
# completion, then exits so docker-compose returns.
set -u

PC_DIR=/workspace/context-transfer-engine/test/integration/mpi_producer_consumer
SSH_SHARE="${PC_DIR}/.mpi_ssh"
export CLIO_SERVER_CONF="${PC_DIR}/clio_config.yaml"

echo '=== Node 2: SSH setup ==='
mkdir -p /home/iowarp/.ssh && chmod 700 /home/iowarp/.ssh
ssh-keygen -t rsa -b 2048 -N '' -f /home/iowarp/.ssh/id_rsa -q 2>/dev/null || true
printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' \
    >> /home/iowarp/.ssh/config

echo '=== Node 2: starting local clio daemon ==='
/workspace/build/bin/clio_run runtime start &
echo "Node 2: daemon PID $!"

echo '=== Node 2: waiting for node1 SSH public key ==='
for i in $(seq 1 120); do
    [ -f "${SSH_SHARE}/node1.pub" ] && break
    sleep 0.5
done
if [ ! -f "${SSH_SHARE}/node1.pub" ]; then
    echo 'ERROR: node1 public key never appeared'; exit 1
fi
cat "${SSH_SHARE}/node1.pub" >> /home/iowarp/.ssh/authorized_keys
chmod 600 /home/iowarp/.ssh/authorized_keys

echo '=== Node 2: starting sshd; serving until node1 signals done ==='
sudo /usr/sbin/sshd

# Stay alive so mpirun can launch rank 1 here; exit once node1 is finished.
for i in $(seq 1 600); do
    [ -f "${SSH_SHARE}/done" ] && { echo 'Node 2: node1 signalled done'; break; }
    sleep 1
done
echo '=== Node 2: exiting ==='
exit 0
