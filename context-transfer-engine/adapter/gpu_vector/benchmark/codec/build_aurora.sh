#!/bin/bash
# Build the SYCL codec workload benchmark + the per-codec standalone tests for
# Aurora (PVC, AOT).
#   module load lz4 ; DEPS=<prefix with libsnappy+libzstd> OUT=/path/bin bash build_aurora.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)          # clio-core root
CMP=$ROOT/context-transport-primitives/compress/src
TST=$ROOT/context-transport-primitives/base/test/gpu
OUT=${OUT:-$HERE/bin}
DEVICE=${SYCL_DEVICE:-pvc}
DEPS=${DEPS:?set DEPS to a prefix providing snappy and zstd (include/ + lib/)}
mkdir -p "$OUT"
FLAGS=(-std=c++17 -O3 -fsycl -fsycl-targets=spir64_gen
       -Xsycl-target-backend "-device $DEVICE" -I"$DEPS/include")
LIBS=(-L"$DEPS/lib" -Wl,-rpath,"$DEPS/lib" -llz4 -lsnappy -lzstd)
KERNELS=()
for c in lz4 snappy ans zstd; do KERNELS+=("$CMP/sycl_$c/sycl_${c}_kernels.cc"); done
icpx "${FLAGS[@]}" -fiopenmp "$HERE/bench_sycl_codecs_workloads.cc" \
     "${KERNELS[@]}" "${LIBS[@]}" -o "$OUT/bench_sycl_codecs_workloads"
for c in lz4 snappy ans zstd; do
  icpx "${FLAGS[@]}" "$CMP/sycl_$c/sycl_${c}_kernels.cc" \
       "$TST/test_sycl_${c}_standalone.cc" "${LIBS[@]}" \
       -o "$OUT/test_sycl_${c}_standalone"
done
echo "built -> $OUT"
