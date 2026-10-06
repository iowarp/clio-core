#!/usr/bin/env bash
#===============================================================================
# build.sh -- build gs_dump (the GPU Gray-Scott field dump) outside the source
# tree, into OUT (default ~/np-build/grayscott/gs_dump).
#
#   [ARCH=sm_80] [OUT=~/np-build/grayscott] build.sh
#
# gs_dump links only the CUDA runtime and the repository's Gray-Scott model
# (context-transfer-engine/compressor/generator/grayscott).
#===============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
G=$HERE/../../context-transfer-engine/compressor/generator/grayscott
OUT=${OUT:-$HOME/np-build/grayscott}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
mkdir -p "$OUT"
"$NVCC" -O2 -std=c++17 -arch="${ARCH:-sm_80}" -I"$G" -o "$OUT/gs_dump" \
  "$HERE/gs_dump.cu" "$G/grayscott_sim.cu"
echo "built $OUT/gs_dump"
