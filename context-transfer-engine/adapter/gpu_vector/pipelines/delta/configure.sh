#!/bin/bash
# Configure + build the CUDA-clang gpu_vector stack on NCSA Delta (A100, sm_80).
#
#   ./configure.sh              # configure and build everything gpu_vector needs
#   ./configure.sh --configure  # configure only
#   BUILD_DIR=/path ./configure.sh
#
# This is the `cuda-clang` preset from CMakePresets.json with the three Delta
# deltas: sm_80 instead of sm_89, the out-of-lmod CUDA 12.6.3 toolkit and the
# llvm/19.1.7 clang wired in explicitly (see env.sh for why each), and
# compression ON -- five of the six paged benches guard on TARGET
# clio_cte_compressor_runtime, so with it off only lammps_md_paged gets built.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
source "$HERE/env.sh"

CONFIGURE_ONLY=0
if [ "${1:-}" = "--configure" ]; then CONFIGURE_ONLY=1; shift; fi

BUILD_DIR="${BUILD_DIR:-$REPO/build-clang}"
JOBS="${JOBS:-16}"

CUDA_FLAGS="--cuda-path=$CUDA_HOME --gcc-install-dir=$CLIO_DELTA_GCC_DIR"
CUDA_FLAGS="$CUDA_FLAGS -Wno-unknown-cuda-version"

cmake -S "$REPO" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER="$CLIO_DELTA_CLANGXX" \
  -DCMAKE_CUDA_ARCHITECTURES=80-real \
  -DCMAKE_CUDA_FLAGS="$CUDA_FLAGS" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCLIO_GPU_CLANG=ON \
  -DCLIO_CORE_ENABLE_CUDA=ON \
  -DCLIO_CORE_ENABLE_RUNTIME=ON \
  -DCLIO_CORE_ENABLE_CTE=ON \
  -DCLIO_CORE_ENABLE_CAE=ON \
  -DCLIO_CORE_ENABLE_CEE=OFF \
  -DCLIO_CORE_ENABLE_TESTS=ON \
  -DCLIO_CORE_ENABLE_BENCHMARKS=ON \
  -DCLIO_CORE_ENABLE_ELF=OFF \
  -DCLIO_CTE_ENABLE_COMPRESS=ON \
  -DCLIO_CTE_ENABLE_ADIOS2_ADAPTER=OFF \
  -DCLIO_CORE_ENABLE_GRAY_SCOTT=OFF \
  -DCLIO_CORE_ENABLE_ASAN=OFF \
  -DCLIO_CORE_ENABLE_IO_URING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  "$@"

if [ "$CONFIGURE_ONLY" = 0 ]; then
  cmake --build "$BUILD_DIR" -j "$JOBS"
fi
