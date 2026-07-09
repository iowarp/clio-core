#!/bin/bash
# Build the #526 regression APPTAINER SIF — this IS the automated "installation"
# for the containerized single_node/distributed pipelines. Two stages:
#   1. `docker build` pipelines/container/Dockerfile.regression -> a local OCI
#      image (the Dockerfile bakes spack iowarp(+libfuse,+redis), juicefs, redis,
#      fio, jarvis-cd, and this repo; its final RUN fails if any binary is missing
#      — the install-cleanliness gate).
#   2. `apptainer build <sif> docker-daemon://<image>` -> a portable .sif.
#
# Strategy A (default): convert the LOCAL docker image — no Docker Hub round-trip,
# no apptainer --fakeroot needed. Docker is available on the Ares login node
# (`docker ps` works) and apptainer is Spack-installed (see validate_apptainer.sh
# Phase 0). For a Docker-free build see pipelines/container/regression.def
# (Strategy B: `apptainer build --fakeroot <sif> regression.def`).
#
# The SIF is written to the jarvis containers cache so the pipeline YAMLs find it
# by basename (container_image: "iowarp-regression-526"):
#     <jarvis shared_dir>/containers/iowarp-regression-526.sif
# On Ares shared_dir is on /mnt/common (shared FS) -> the SIF is visible on every
# compute node automatically (the distributed run needs no per-node copy).
#
# Run from the REPO ROOT so the Dockerfile's `COPY jarvis_clio_core` bakes in the
# local package edits. Re-run daily so #526 tracks the latest IOWarp.
#
# Usage (from repo root, on a host with docker + apptainer):
#   jarvis_clio_core/scripts/build_regression_image.sh
#
# Common overrides (env):
#   IMAGE=iowarp-regression:526             # local docker image tag to build
#   SIF_BASENAME=iowarp-regression-526      # must match YAML container_image
#   SIF_PATH=/abs/path/iowarp-regression-526.sif   # override the SIF location
#   IOWARP_SPEC='iowarp@dev +fuse +redis'   # upstream dev + FUSE + redis bench
#   BASE_IMAGE=iowarp/iowarp-build:latest
#   JARVIS_REF=2925ee8                      # MUST match the Ares jarvis pin (dev)
#   SKIP_DOCKER_BUILD=1                     # reuse an existing local docker image
set -euo pipefail

IMAGE="${IMAGE:-iowarp-regression:526}"
SIF_BASENAME="${SIF_BASENAME:-iowarp-regression-526}"
IOWARP_SPEC="${IOWARP_SPEC:-iowarp@dev +fuse +redis}"
BASE_IMAGE="${BASE_IMAGE:-iowarp/iowarp-build:latest}"
JARVIS_REF="${JARVIS_REF:-2925ee8}"
SKIP_DOCKER_BUILD="${SKIP_DOCKER_BUILD:-0}"

DOCKERFILE="jarvis_clio_core/pipelines/container/Dockerfile.regression"
if [ ! -f "$DOCKERFILE" ]; then
  echo "ERROR: run from the repo root (cannot find $DOCKERFILE)" >&2
  exit 1
fi

# ---- resolve the SIF destination (jarvis containers cache) ----------------
# Derive shared_dir from the jarvis config so the SIF lands where the YAMLs look
# for it by basename. Allow a full SIF_PATH override for non-standard setups.
if [ -z "${SIF_PATH:-}" ]; then
  SHARED_DIR="$(grep -hE '^[[:space:]]*shared_dir:' "$HOME"/.ppi-jarvis/*.yaml 2>/dev/null \
                | head -1 | sed -E 's/^[[:space:]]*shared_dir:[[:space:]]*//; s/["'"'"']//g')"
  if [ -z "$SHARED_DIR" ]; then
    echo "ERROR: could not read shared_dir from ~/.ppi-jarvis/*.yaml." >&2
    echo "       Run 'jarvis init' first, or pass SIF_PATH=/abs/path.sif." >&2
    exit 1
  fi
  SIF_PATH="$SHARED_DIR/containers/$SIF_BASENAME.sif"
fi
mkdir -p "$(dirname "$SIF_PATH")"

echo "=== #526 regression SIF build ==="
echo "  docker image : $IMAGE"
echo "  IOWARP_SPEC  : $IOWARP_SPEC"
echo "  BASE_IMAGE   : $BASE_IMAGE"
echo "  JARVIS_REF   : $JARVIS_REF"
echo "  SIF_PATH     : $SIF_PATH"

# ---- tool presence --------------------------------------------------------
command -v apptainer >/dev/null || {
  echo "ERROR: apptainer not on PATH. Spack-install it first (see"            >&2
  echo "       scripts/validate_apptainer.sh Phase 0) and 'spack load apptainer'." >&2
  exit 1
}

# ---- stage 1: docker build ------------------------------------------------
if [ "$SKIP_DOCKER_BUILD" = "1" ]; then
  echo "=== SKIP_DOCKER_BUILD=1 — reusing existing local image $IMAGE ==="
else
  command -v docker >/dev/null || {
    echo "ERROR: docker not on PATH (needed for Strategy A). For a Docker-free" >&2
    echo "       build use: apptainer build --fakeroot '$SIF_PATH' \\"          >&2
    echo "       jarvis_clio_core/pipelines/container/regression.def"           >&2
    exit 1
  }
  echo "=== docker build $IMAGE ==="
  docker build -f "$DOCKERFILE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg IOWARP_SPEC="$IOWARP_SPEC" \
    --build-arg JARVIS_REF="$JARVIS_REF" \
    -t "$IMAGE" .
  echo "=== docker build OK: $IMAGE ==="
fi

# ---- stage 2: convert the local docker image to a SIF ---------------------
echo "=== apptainer build $SIF_PATH  (docker-daemon://$IMAGE) ==="
apptainer build --force "$SIF_PATH" "docker-daemon://$IMAGE"

echo "=== SIF ready: $SIF_PATH ==="
ls -lh "$SIF_PATH"
echo "The pipeline YAMLs reference this by basename: container_image: \"$SIF_BASENAME\""
