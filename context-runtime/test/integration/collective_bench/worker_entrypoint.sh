#!/bin/bash
# Worker entrypoint (nodes 2..N) — collective benchmark.
#
# A worker starts its local clio daemon, authorizes node 1's SSH key, and serves
# sshd so mpirun (launched on node 1) can start a rank here. It does no
# launching itself; it stays up until node 1 signals completion, then exits so
# docker-compose returns.
#
# The wait for "done" matters: if a worker's container exited while node 1 was
# still running an arm, its daemon would vanish mid-collective and the AllToOne
# barrier would never reach full membership -- the benchmark would hang rather
# than fail, which is the harder failure to read.
set -u

CB_DIR=/workspace/context-runtime/test/integration/collective_bench
SSH_SHARE="${CB_DIR}/.mpi_ssh"
export CLIO_SERVER_CONF="${CB_DIR}/clio_conf.yaml"
NODE_ID="${NODE_ID:-?}"

echo "=== Node ${NODE_ID}: SSH setup ==="
mkdir -p /home/iowarp/.ssh && chmod 700 /home/iowarp/.ssh
ssh-keygen -t rsa -b 2048 -N '' -f /home/iowarp/.ssh/id_rsa -q 2>/dev/null || true
printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' \
    >> /home/iowarp/.ssh/config

echo "=== Node ${NODE_ID}: starting local clio daemon ==="
# Daemon output goes to a file: at info level it emits a periodic scheduler
# report that would otherwise bury this script's progress in `docker logs`.
CLIO_COLL_PROF=${CLIO_COLL_PROF:-} CLIO_NET_QPROF=${CLIO_NET_QPROF:-} CLIO_WORKER_RATE=${CLIO_WORKER_RATE:-} /workspace/build/bin/clio_run runtime start > /tmp/clio_daemon.log 2>&1 &
echo "Node ${NODE_ID}: daemon PID $!"

echo "=== Node ${NODE_ID}: waiting for node1 SSH public key ==="
for _ in $(seq 1 240); do
    [ -f "${SSH_SHARE}/node1.pub" ] && break
    sleep 0.5
done
if [ ! -f "${SSH_SHARE}/node1.pub" ]; then
    echo "ERROR: node1 public key never appeared"; exit 1
fi
cat "${SSH_SHARE}/node1.pub" >> /home/iowarp/.ssh/authorized_keys
chmod 600 /home/iowarp/.ssh/authorized_keys

echo "=== Node ${NODE_ID}: starting sshd; serving until node1 signals done ==="
sudo /usr/sbin/sshd

for _ in $(seq 1 1800); do
    [ -f "${SSH_SHARE}/done" ] && { echo "Node ${NODE_ID}: node1 signalled done"; break; }
    sleep 1
done
echo "=== Node ${NODE_ID}: exiting ==="
exit 0
