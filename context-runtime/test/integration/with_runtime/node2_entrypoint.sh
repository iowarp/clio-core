#!/bin/bash
# Node 2 entrypoint — "Clio as a cache" test (issue #1015).
#
# Node 2 is secondary: it authorizes node 1's SSH key and serves sshd so mpirun
# (launched on node 1) can start ranks 4-7 here. Like node 1, it starts no clio
# daemon — the ranks bring their own runtime up. It stays alive until node 1
# signals completion, then exits so docker-compose returns.
set -u

WR_DIR=/workspace/context-runtime/test/integration/with_runtime
SSH_SHARE="${WR_DIR}/.mpi_ssh"
export CLIO_SERVER_CONF="${WR_DIR}/clio_config.yaml"
export CLIO_WITH_RUNTIME=1

echo '=== Node 2: SSH setup ==='
mkdir -p /home/iowarp/.ssh && chmod 700 /home/iowarp/.ssh
ssh-keygen -t rsa -b 2048 -N '' -f /home/iowarp/.ssh/id_rsa -q 2>/dev/null || true
printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' \
    >> /home/iowarp/.ssh/config

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

# Stay alive so mpirun can launch ranks 4-7 here; exit once node1 is finished.
for i in $(seq 1 600); do
    [ -f "${SSH_SHARE}/done" ] && { echo 'Node 2: node1 signalled done'; break; }
    sleep 1
done
echo '=== Node 2: exiting ==='
exit 0
