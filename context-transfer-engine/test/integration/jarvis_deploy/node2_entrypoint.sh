#!/bin/bash
# Node 2 entrypoint for CTE Jarvis deployment integration test.
# Runs on the secondary node: waits for node1's SSH key then serves SSH connections.
# The IOWarp runtime is started here remotely by Jarvis (via pssh from node1).
set -e

JARVIS_DEPLOY_DIR=/workspace/context-transfer-engine/test/integration/jarvis_deploy

echo '========================================'
echo 'Node 2: Waiting for node1 SSH public key'
echo '========================================'
mkdir -p /home/iowarp/.ssh
chmod 700 /home/iowarp/.ssh
while [ ! -f "${JARVIS_DEPLOY_DIR}/.jarvis_ssh/node1.pub" ]; do
    sleep 0.5
done
cat "${JARVIS_DEPLOY_DIR}/.jarvis_ssh/node1.pub" >> /home/iowarp/.ssh/authorized_keys
chmod 600 /home/iowarp/.ssh/authorized_keys

echo '========================================'
echo 'Node 2: Starting SSH server (foreground)'
echo 'Node 2 is ready - waiting for Jarvis commands from node1'
echo '========================================'
sudo /usr/sbin/sshd -D
