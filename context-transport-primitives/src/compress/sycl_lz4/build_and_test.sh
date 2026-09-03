#!/bin/bash
# Build + validate the SYCL LZ4 codec (the "nvcomp equivalent for SYCL").
# This is the VALIDATED recipe: DPC++ targeting NVIDIA (nvptx64) so the codec
# runs on the same A100 as nvcomp, for a like-for-like comparison. Run on a GPU
# node (the login node has no device runtime). liblz4 is linked only as a
# reference oracle in the test.
#
# Usage (inside the deps-nvidia.sif container, on a GPU node):
#   bash build_and_test.sh            # build + run correctness/throughput checks
#
# For AdaptiveCpp instead of DPC++, swap the compiler line for:
#   acpp --acpp-targets=cuda:sm_80 ...
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CTP=$(cd "$HERE/../../.." && pwd)          # context-transport-primitives root
DPCPP=${DPCPP:-/opt/intel/dpcpp/bin/clang++}
ARCH=${SYCL_CUDA_ARCH:-sm_80}
OUT=${OUT:-/tmp/test_sycl_lz4}

echo "== building SYCL LZ4 (DPC++ -> nvptx64/$ARCH) =="
"$DPCPP" -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
  -Xsycl-target-backend --cuda-gpu-arch="$ARCH" \
  "$CTP/src/compress/sycl_lz4_kernels.cc" \
  "$CTP/test/unit/gpu/test_sycl_lz4_standalone.cc" \
  -llz4 -o "$OUT"

echo "== correctness + throughput (A100) =="
"$OUT" 33554432 0.85   # 32 MB, typical redundancy
"$OUT" 67108864 0.70   # 64 MB, lower redundancy
"$OUT"  8388608 0.95   # 8 MB, high redundancy
