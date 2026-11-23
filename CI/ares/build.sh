#!/bin/bash
# Build clio-core for the nightly Ares runs
# (.github/workflows/nightly-ares-distributed.yml, nightly-ares-bench.yml). Runs on ONE compute node
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
# JDK for the YCSB JNI binding (libclio_ycsb_jni; nightly benchmarks).
JAVA_HOME=${JAVA_HOME:-$(ls -d "$ARES_SPACK_ROOT"/openjdk-17* 2>/dev/null | head -1)}

CMAKE=$(ls -d "$ARES_SPACK_ROOT"/cmake-3.31.9-*/bin/cmake | head -1)
export CMAKE_PREFIX_PATH="$(tr -d '\n' < "$ARES_PREFIX_PATH_FILE")"
# Catch2 v3 for the tests: a clean checkout has no bundled external/Catch2,
# and the prefix file does not list it (the nightly builds failed at
# configure without it).
CATCH2=$(ls -d "$ARES_SPACK_ROOT"/catch2-3* 2>/dev/null | head -1)
[ -n "$CATCH2" ] && CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CATCH2"
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
  -DCLIO_CORE_ENABLE_RPATH=ON -DCLIO_CORE_ENABLE_CCACHE=OFF \
  -DCLIO_CTE_ENABLE_FUSE_ADAPTER=ON \
  -DCLIO_JNI_INCLUDE_DIR="$JAVA_HOME/include"
"$CMAKE" --build "$BUILD" -j"$JOBS"

for b in clio_run clio_run_bdev_chimod_tests test_core_functionality \
         clio_cte_fuse libclio_ycsb_jni.so; do
  [ -e "$BUILD/bin/$b" ] || { echo "ERROR: $BUILD/bin/$b was not built" >&2; exit 1; }
done
echo "BUILD OK"
