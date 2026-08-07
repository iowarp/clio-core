#!/bin/bash
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Build cuSZp V3.0.0 (native sm_80) + the clio-core CUDA GNN targets.
# Run INSIDE the deps-nvidia.sif container on a Delta A100 node:
#   apptainer exec --nv -B /work,/tmp,/u /u/rpawar/containers/deps-nvidia.sif \
#     bash ${BENCH_ROOT}/build.sh
set -euo pipefail

CUDA=/usr/local/cuda-12.6
export PATH="$CUDA/bin:$PATH"
export CUDACXX="$CUDA/bin/nvcc"
JOBS="${JOBS:-16}"

# Delta's Cray PE exports CC=cc / CXX=CC (Cray wrappers). Those do not exist
# inside the container, so CMake's compiler detection fails. Pin the GNU ones.
unset CC CXX FC F77 F90 CFLAGS CXXFLAGS LDFLAGS
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

CUSZP_SRC=/u/rpawar/cuSZp
CUSZP_BUILD=$CUSZP_SRC/build-a100
# NOTE: cuSZp's cmake/Installing.cmake FORCE-overrides CMAKE_INSTALL_PREFIX to
# <src>/install, so that is the real prefix regardless of what we pass.
CUSZP_PREFIX=$CUSZP_SRC/install
CLIO_SRC=/u/rpawar/clio-core
CLIO_BUILD=$CLIO_SRC/build

echo "=============== toolchain ==============="
nvcc --version | tail -2
g++ --version | head -1
cmake --version | head -1

echo "=============== 1) cuSZp V3.0.0 (native sm_80, shared) ==============="
git -C "$CUSZP_SRC" log --oneline -1
if [ -f "$CUSZP_PREFIX/lib/libcuSZp.so" ] && [ "${FORCE_CUSZP:-0}" != "1" ]; then
  echo "cuSZp already installed at $CUSZP_PREFIX (set FORCE_CUSZP=1 to rebuild)"
else
  rm -rf "$CUSZP_BUILD"
  cmake -S "$CUSZP_SRC" -B "$CUSZP_BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
        -DCMAKE_C_COMPILER=/usr/bin/gcc \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_INSTALL_PREFIX="$CUSZP_PREFIX"
  cmake --build "$CUSZP_BUILD" -j "$JOBS"
  cmake --install "$CUSZP_BUILD"
fi
ls -la "$CUSZP_PREFIX/lib/"

echo "--- cuSZp: confirm NATIVE sm_80 SASS (not PTX-only) ---"
cuobjdump "$CUSZP_PREFIX/lib/libcuSZp.so" 2>/dev/null | grep -oE "arch = sm_[0-9]+" | sort -u || true

echo "=============== 2) clio-core (CUDA sm_80, compress ON) ==============="
cmake -S "$CLIO_SRC" -B "$CLIO_BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCLIO_CORE_ENABLE_CUDA=ON \
      -DCLIO_CTE_ENABLE_COMPRESS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
      -DCMAKE_C_COMPILER=/usr/bin/gcc \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -DCLIO_CUSZP_ROOT="$CUSZP_PREFIX"

cmake --build "$CLIO_BUILD" --target test_gpu_vector_gnn_train -j "$JOBS"
cmake --build "$CLIO_BUILD" --target test_gpu_vector_gnn_capacity -j "$JOBS"

echo "=============== built ==============="
ls -la "$CLIO_BUILD"/bin/test_gpu_vector_gnn_train "$CLIO_BUILD"/bin/test_gpu_vector_gnn_capacity

echo "=============== verify NATIVE sm_80 in the test binaries ==============="
for t in test_gpu_vector_gnn_train test_gpu_vector_gnn_capacity; do
  echo -n "$t: "
  cuobjdump "$CLIO_BUILD/bin/$t" 2>/dev/null | grep -oE "arch = sm_[0-9]+" | sort -u | tr '\n' ' '
  echo
done
echo "BUILD OK"
