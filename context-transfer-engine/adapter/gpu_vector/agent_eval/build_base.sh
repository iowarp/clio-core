#!/usr/bin/env bash
# Build the sanitized tree once, inside the trial image, at the SAME paths the
# trials use (/work/clio-core, /work/build), so compile_commands.json and
# RPATHs are valid inside every trial container.
#
#   ./build_base.sh           (run from the host side of docker)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

mkdir -p "${RUNS_HOST_VIEW}/base/build" "${RUNS_HOST_VIEW}/base/build-coroc"
cp "${WS}/build-coroc/clio-coroc" "${RUNS_HOST_VIEW}/base/build-coroc/"

docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "${RUNS_DOCKER_VIEW}/base/clio-core:/work/clio-core" \
  -v "${RUNS_DOCKER_VIEW}/base/build:/work/build" \
  "${IMAGE}" bash -lc '
    set -e
    cmake -S /work/clio-core -B /work/build -DCMAKE_BUILD_TYPE=Release \
      -DCLIO_CORE_ENABLE_CUDA=ON -DCLIO_GPU_CLANG=OFF \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=89-real \
      -DCLIO_CTE_ENABLE_COMPRESS=OFF -DCLIO_CORE_ENABLE_TESTS=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > /work/build/configure.log 2>&1
    cmake --build /work/build -j"$(nproc)" > /work/build/build.log 2>&1
    echo BUILD_OK'
