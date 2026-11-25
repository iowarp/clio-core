#!/bin/bash
# CPU build of clio-core (ssc-system1-dev) for the SSC harness, inside deps-nvidia.sif.
set -euo pipefail
unset CC CXX FC F77 F90 CFLAGS CXXFLAGS LDFLAGS
export CC=/usr/bin/gcc CXX=/usr/bin/g++
SRC=/u/rpawar/clio-core-ssc
BLD=$SRC/build
cmake -S $SRC -B $BLD -DCMAKE_BUILD_TYPE=Release \
  -DCLIO_CORE_ENABLE_RUNTIME=ON -DCLIO_CORE_ENABLE_CTE=ON -DCLIO_CORE_ENABLE_CAE=OFF -DCLIO_CORE_ENABLE_CEE=OFF \
  -DCLIO_CORE_ENABLE_TESTS=ON -DCLIO_CORE_ENABLE_BENCHMARKS=OFF -DCLIO_CORE_ENABLE_CUDA=OFF \
  -DCLIO_CTE_ENABLE_COMPRESS=ON -DCLIO_CORE_ENABLE_PYTHON=ON -DCLIO_CORE_ENABLE_RPATH=ON \
  -DCLIO_CORE_ENABLE_IO_URING=ON -DCLIO_CORE_ENABLE_GRAY_SCOTT=OFF -DCLIO_CTE_ENABLE_ADIOS2_ADAPTER=OFF \
  -DCLIO_CORE_ENABLE_ZMQ=ON -DCLIO_CORE_ENABLE_CEREAL=ON -DCLIO_CORE_ENABLE_ELF=OFF
cmake --build $BLD -j ${JOBS:-32} 2>&1 | grep -E "error|Error|warning: unused|\[100%\]|Built target clio_cte_core_ext" | tail -40
ls $BLD/bin | grep -E "clio_run$|clio_cte_core_ext|cache|indexer|compressor" | head -30
echo BUILD_DONE
