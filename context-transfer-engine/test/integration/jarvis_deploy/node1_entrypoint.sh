#!/bin/bash
# Node 1 entrypoint for CTE Jarvis deployment integration test.
# Runs on the primary node: sets up SSH, initializes Jarvis, and runs the pipeline.
set -e

# Activate the iowarp virtualenv so jarvis and other tools are in PATH
# shellcheck source=/dev/null
source /home/iowarp/venv/bin/activate

JARVIS_DEPLOY_DIR=/workspace/context-transfer-engine/test/integration/jarvis_deploy

echo '========================================'
echo 'Node 1: Setting up SSH'
echo '========================================'
mkdir -p /home/iowarp/.ssh
chmod 700 /home/iowarp/.ssh
ssh-keygen -t rsa -b 2048 -N '' -f /home/iowarp/.ssh/id_rsa -q 2>/dev/null || true

# Create shared SSH key directory (needs sudo since workspace may be owned by another user)
sudo mkdir -p "${JARVIS_DEPLOY_DIR}/.jarvis_ssh"
sudo chmod 777 "${JARVIS_DEPLOY_DIR}/.jarvis_ssh"

cp /home/iowarp/.ssh/id_rsa.pub "${JARVIS_DEPLOY_DIR}/.jarvis_ssh/node1.pub"
cat /home/iowarp/.ssh/id_rsa.pub >> /home/iowarp/.ssh/authorized_keys
chmod 600 /home/iowarp/.ssh/authorized_keys
echo 'StrictHostKeyChecking no' >> /home/iowarp/.ssh/config
echo 'UserKnownHostsFile /dev/null' >> /home/iowarp/.ssh/config

echo '========================================'
echo 'Node 1: Starting SSH server (for self-SSH via pssh)'
echo '========================================'
sudo /usr/sbin/sshd

echo '========================================'
echo 'Node 1: Waiting for node2 SSH server'
echo '========================================'
NODE2_READY=0
for i in $(seq 1 60); do
    if ssh -o ConnectTimeout=3 -o BatchMode=yes iowarp-jarvis-node2 'echo ready' 2>/dev/null; then
        echo 'Node 2 SSH is ready'
        NODE2_READY=1
        break
    fi
    sleep 1
done
if [ "$NODE2_READY" = "0" ]; then
    echo "ERROR: Node 2 SSH not ready after 60 seconds"
    exit 1
fi

echo '========================================'
echo 'Node 1: Initializing Jarvis'
echo '========================================'
sudo mkdir -p "${JARVIS_DEPLOY_DIR}/.jarvis-shared"
sudo chmod 777 "${JARVIS_DEPLOY_DIR}/.jarvis-shared"
jarvis init \
    /home/iowarp/.ppi-jarvis/config \
    /home/iowarp/.ppi-jarvis/private \
    "${JARVIS_DEPLOY_DIR}/.jarvis-shared" \
    +force

echo '========================================'
echo 'Node 1: Adding jarvis_iowarp repository'
echo '========================================'
jarvis repo add /workspace/jarvis_iowarp +force

echo '========================================'
echo 'Node 1: Setting hostfile'
echo '========================================'
jarvis hostfile set "${JARVIS_DEPLOY_DIR}/hostfile"

echo '========================================'
echo 'Node 1: Running Jarvis pipeline'
echo '========================================'
jarvis ppl run yaml "${JARVIS_DEPLOY_DIR}/cte_integration_test.yaml"
PIPELINE_EXIT=$?
echo '========================================'
echo "Node 1: Pipeline exit code: ${PIPELINE_EXIT}"
echo '========================================'
exit $PIPELINE_EXIT
