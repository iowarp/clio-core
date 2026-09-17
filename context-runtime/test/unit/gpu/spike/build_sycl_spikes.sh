#!/usr/bin/env bash
# Copyright 2024 IOWarp - BSD 3-Clause License
#
# Build and run the SYCL coroutine spikes. NOT part of the CMake build --
# these need a SYCL compiler that the project does not otherwise require.
# See SYCL_COROUTINES.md for the toolchain install and the findings.
set -uo pipefail

DPCPP="${DPCPP_ROOT:-$HOME/opt/dpcpp}"
CUDA="${CUDA_PATH:-/usr/local/cuda}"
OUT="${OUT_DIR:-/tmp/sycl_coro_spikes}"

if [ ! -x "$DPCPP/bin/clang++" ]; then
  echo "no DPC++ at $DPCPP -- see SYCL_COROUTINES.md (set DPCPP_ROOT)" >&2
  exit 1
fi
export PATH="$DPCPP/bin:$PATH"
export LD_LIBRARY_PATH="$DPCPP/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
cd "$(dirname "$0")"

rc=0
for src in spike_sycl_coroutine spike_sycl_lane_coroutine spike_sycl_implicit_lane; do
  echo "=== $src (nvptx64) ==="
  if ! clang++ -fsycl -std=c++20 -fsycl-targets=nvptx64-nvidia-cuda \
        --cuda-path="$CUDA" "$src.cpp" -o "$OUT/$src"; then
    echo "$src: BUILD FAILED"; rc=1; continue
  fi
  if ! "$OUT/$src"; then echo "$src: RUN FAILED"; rc=1; fi
done

# The SPIR-V target is EXPECTED to crash DPC++ codegen (see SYCL_COROUTINES.md).
# Compile-only, and treat the crash as the documented outcome rather than a
# failure of this script.
echo "=== spike_sycl_coroutine (spir64) -- expected to crash the compiler ==="
if clang++ -fsycl -std=c++20 -fsycl-targets=spir64 \
     -c spike_sycl_coroutine.cpp -o "$OUT/spir.o" 2>"$OUT/spir.log"; then
  echo "SPIR-V now COMPILES -- the upstream bug is fixed; re-read the doc."
else
  grep -q "PHINode::setIncomingValue" "$OUT/spir.log" \
    && echo "crashed as documented (coroutine frame PHI type assertion)" \
    || { echo "failed DIFFERENTLY -- see $OUT/spir.log"; rc=1; }
fi

exit $rc
