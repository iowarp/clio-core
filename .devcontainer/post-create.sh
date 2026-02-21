#!/bin/bash
# Shared devcontainer post-create setup
# Called once when the container is first created.

set -e

DEVUSER="iowarp"
HOME_DIR="/home/${DEVUSER}"

# Mark workspace as safe for git
git config --global --add safe.directory /workspace

# Copy host SSH keys into container
sudo rm -rf "${HOME_DIR}/.ssh"
if [ -d /tmp/host-ssh ]; then
    sudo cp -r /tmp/host-ssh "${HOME_DIR}/.ssh"
    sudo chown -R "${DEVUSER}:${DEVUSER}" "${HOME_DIR}/.ssh"
    chmod 700 "${HOME_DIR}/.ssh"
    chmod 600 "${HOME_DIR}/.ssh"/* 2>/dev/null || true
    chmod 644 "${HOME_DIR}/.ssh"/*.pub 2>/dev/null || true
fi

# Copy host Claude config into container
if [ -d /tmp/host-claude ] && [ "$(ls -A /tmp/host-claude)" ]; then
    sudo cp -r /tmp/host-claude "${HOME_DIR}/.claude"
    sudo chown -R "${DEVUSER}:${DEVUSER}" "${HOME_DIR}/.claude"
fi
if [ -f /tmp/host-claude.json ] && [ -s /tmp/host-claude.json ]; then
    sudo cp /tmp/host-claude.json "${HOME_DIR}/.claude.json"
    sudo chown "${DEVUSER}:${DEVUSER}" "${HOME_DIR}/.claude.json"
fi

# Fix docker socket permissions (dynamic GID match)
"$(dirname "$0")/post-start.sh"

# Start SSH service
sudo service ssh start

# Activate Python virtual environment
echo "Python venv available at ${HOME_DIR}/venv (auto-activated in new shells)"
