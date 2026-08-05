#!/bin/bash
# Provision YCSB + comparison stores into external/ (issue #862).
# Idempotent: safe to re-run; each artifact is skipped if already present.
# Called from .devcontainer/post-create.sh; can also be run manually:
#   ./docker/provision-ycsb.sh [repo-root]
set -uo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXT="$REPO_ROOT/external"
DIST="$EXT/ycsb-dist"
YCSB_VER=0.17.0
mkdir -p "$DIST"

# Source clone (reference + future maven builds of the full harness)
if [ ! -d "$EXT/YCSB/.git" ]; then
  git clone --depth 1 https://github.com/brianfrankcooper/YCSB.git "$EXT/YCSB" \
    || echo "WARN: YCSB clone failed (offline?); release tarballs may still work"
fi

# Prebuilt binding tarballs: runnable without maven
for b in redis rocksdb dynamodb; do
  if [ ! -d "$DIST/ycsb-$b-binding-$YCSB_VER" ]; then
    curl -sfL -o "$DIST/ycsb-$b-binding-$YCSB_VER.tar.gz" \
      "https://github.com/brianfrankcooper/YCSB/releases/download/$YCSB_VER/ycsb-$b-binding-$YCSB_VER.tar.gz" \
      && tar xzf "$DIST/ycsb-$b-binding-$YCSB_VER.tar.gz" -C "$DIST" \
      || echo "WARN: fetch of ycsb-$b-binding failed"
  fi
done

# DynamoDB Local
if [ ! -f "$DIST/dynamodb-local/DynamoDBLocal.jar" ]; then
  mkdir -p "$DIST/dynamodb-local"
  curl -sfL -o "$DIST/dynamodb_local_latest.tar.gz" \
    "https://d1ni2b6xgvw0s0.cloudfront.net/dynamodb_local_latest.tar.gz" \
    && tar xzf "$DIST/dynamodb_local_latest.tar.gz" -C "$DIST/dynamodb-local" \
    || echo "WARN: DynamoDB Local fetch failed"
fi

# boto3 for the usertable-creation helper (venv-local, no sudo needed)
python3 -c 'import boto3' 2>/dev/null || pip install -q boto3 || true

# Compile the Clio YCSB adapter when a JDK and the core jar are available
CORE_JAR="$DIST/ycsb-redis-binding-$YCSB_VER/lib/core-$YCSB_VER.jar"
SRC="$REPO_ROOT/context-transfer-engine/benchmark/ycsb/ClioClient.java"
if command -v javac >/dev/null && [ -f "$CORE_JAR" ] && [ -f "$SRC" ]; then
  mkdir -p "$DIST/clio-binding/classes"
  javac -cp "$CORE_JAR" -d "$DIST/clio-binding/classes" "$SRC" \
    || echo "WARN: ClioClient.java compile failed"
fi

echo "YCSB provisioning complete under $DIST"
