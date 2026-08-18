#!/bin/bash
# 4-node distributed gpu_vector stress test (docker).
#
# Requires: nvidia container toolkit; the repo built into build/ (or set
# BUILD_DIR); iowarp/deps-cpu image.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$REPO_ROOT}"
export HOST_UID=$(id -u) HOST_GID=$(id -g)
cd "$SCRIPT_DIR"
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
# --abort-on-container-exit stops the cluster the moment any node exits;
# exit code propagates the first failure.
docker compose up --abort-on-container-exit --exit-code-from gvd-node1
rc=$?
docker compose down -v --remove-orphans >/dev/null 2>&1 || true
exit $rc
