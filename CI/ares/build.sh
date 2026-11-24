#!/bin/bash
# Build clio-core for the nightly Ares distributed run
# (.github/workflows/nightly-ares-distributed.yml). Runs on ONE compute node
# via `srun -N1`; the build dir is on NFS so every allocated node sees it.
#
# Toolchain: spack-built deps under ARES_SPACK_ROOT. ARES_PREFIX_PATH_FILE holds
# a ':'-separated CMAKE_PREFIX_PATH (one spack prefix per dependency).
set -euo pipefail

ARES_SPACK_ROOT=${ARES_SPACK_ROOT:-/mnt/common/llogan/spack/opt/spack/linux-skylake_avx512}
ARES_PREFIX_PATH_FILE=${ARES_PREFIX_PATH_FILE:-/mnt/common/llogan/builds/bench_prefix_path.txt}
SRC=${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}
BUILD=${BUILD_DIR:-$SRC/build}
JOBS=${BUILD_JOBS:-$(nproc)}

CMAKE=$(ls -d "$ARES_SPACK_ROOT"/cmake-3.31.9-*/bin/cmake | head -1)
export CMAKE_PREFIX_PATH="$(tr -d '\n' < "$ARES_PREFIX_PATH_FILE")"
export PKG_CONFIG_PATH="$(ls -d "$ARES_SPACK_ROOT"/libfuse-3.16.2-*/lib{,64}/pkgconfig \
  "$ARES_SPACK_ROOT"/libzmq-4.3.5-*/lib/pkgconfig 2>/dev/null | tr '\n' ':')${PKG_CONFIG_PATH:-}"

echo "host=$(hostname) commit=$(git -C "$SRC" rev-parse --short HEAD) jobs=$JOBS"
mkdir -p "$BUILD"
"$CMAKE" -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLIO_CORE_ENABLE_RUNTIME=ON -DCLIO_CORE_ENABLE_CTE=ON \
  -DCLIO_CORE_ENABLE_CAE=OFF -DCLIO_CORE_ENABLE_CEE=OFF \
  -DCLIO_CORE_ENABLE_TESTS=ON \
  -DCLIO_CORE_ENABLE_DOCKER_CI=ON \
  -DCLIO_CORE_ENABLE_ZMQ=ON -DCLIO_CORE_ENABLE_CEREAL=ON \
  -DCLIO_CORE_ENABLE_PYTHON=OFF -DCLIO_CORE_ENABLE_CONDA=OFF \
  -DCLIO_CORE_ENABLE_IO_URING=OFF -DCLIO_CORE_ENABLE_VISUALIZER=OFF \
  -DCLIO_CORE_ENABLE_RPATH=ON -DCLIO_CORE_ENABLE_CCACHE=OFF
"$CMAKE" --build "$BUILD" -j"$JOBS"

for b in clio_run clio_run_bdev_chimod_tests test_core_functionality; do
  [ -x "$BUILD/bin/$b" ] || { echo "ERROR: $BUILD/bin/$b was not built" >&2; exit 1; }
done
echo "BUILD OK"
