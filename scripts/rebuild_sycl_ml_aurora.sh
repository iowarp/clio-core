#!/usr/bin/env bash
# Rebuild build/sycl (core + clio-ml SYCL, clio-coroc kernels) on an Aurora
# login node, for bench/spark_compare/clio_ml_bench. Logs MAKE_EXIT <rc>.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# BUILD_DIR overrides the tree; EXTRA_CXX_FLAGS adds defines (e.g.
# -DGV_COMM_TIMING for the gpu_vector cycle counters) to every compile.
B="${BUILD_DIR:-$SRC/build/sycl}"
J=${J:-32}
SP=$(ls -d /home/llogan/spack/opt/spack/linux-icelake/* | tr '\n' ';')
mkdir -p "$B"
step() { echo "=== $*"; "$@"; local rc=$?; [ $rc -eq 0 ] || { echo "MAKE_EXIT $rc ($1)"; exit $rc; }; }
step bash "$SRC/context-runtime/coroc/build.sh" "$B/coroc"
step cmake -S "$SRC" -B "$B" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS:-}" \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_PREFIX_PATH="${SP}${CMAKE_PREFIX_PATH:-}" \
  -DCLIO_CORE_ENABLE_SYCL=ON -DCLIO_CTP_ENABLE_SYCL=ON -DSYCL_TARGET=spir64 \
  -DCLIO_CORE_ENABLE_RUNTIME=ON -DCLIO_CORE_ENABLE_CTE=ON \
  -DCLIO_CORE_ENABLE_CAE=ON -DCLIO_CORE_ENABLE_CEE=OFF \
  -DCLIO_CORE_ENABLE_TESTS=ON -DCLIO_CORE_ENABLE_BENCHMARKS=OFF \
  -DCLIO_CORE_ENABLE_ZMQ=ON -DCLIO_CORE_ENABLE_CLIO_ML=ON \
  -DCLIO_CORE_ROOT="$SRC" -DCLIO_ML_COROC="$B/coroc/clio-coroc" -DCLIO_ML_SYCL_AOT_DEVICE=pvc
step env IGC_FunctionControl=3 make -C "$B" -j"$J" clio_ml_bench
echo "MAKE_EXIT 0"
