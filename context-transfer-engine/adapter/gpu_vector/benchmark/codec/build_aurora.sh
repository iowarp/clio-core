#!/bin/bash
# Build the sycl-lz4 workload codec benchmark for Aurora (PVC, AOT).
#   module load lz4 ; OUT=/path/bin bash build_aurora.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../../.." && pwd)          # clio-core root
KERN=$ROOT/context-transport-primitives/compress/src/sycl_lz4/sycl_lz4_kernels.cc
OUT=${OUT:-$HERE/bin}
DEVICE=${SYCL_DEVICE:-pvc}
mkdir -p "$OUT"
FLAGS=(-std=c++17 -O3 -fsycl -fsycl-targets=spir64_gen
       -Xsycl-target-backend "-device $DEVICE")
icpx "${FLAGS[@]}" -fiopenmp "$HERE/bench_sycl_lz4_workloads.cc" "$KERN" \
     -llz4 -o "$OUT/bench_sycl_lz4_workloads"
icpx "${FLAGS[@]}" "$KERN" \
     "$ROOT/context-transport-primitives/base/test/gpu/test_sycl_lz4_standalone.cc" \
     -llz4 -o "$OUT/test_sycl_lz4_standalone"
echo "built -> $OUT"
