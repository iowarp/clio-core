#!/bin/bash
set -ex

PRESET="${IOWARP_PRESET:-release}"

# Clean any stale build directory (preset uses ${sourceDir}/build)
rm -rf build

# Suppress GCC false positive warnings from aggressive inlining
export CXXFLAGS="${CXXFLAGS:-} -Wno-array-bounds -Wno-maybe-uninitialized -Wno-stringop-overflow"

# Detect CUDA architecture: use 'native' if a GPU is present, else default
# to a portable set so the package builds on headless CI runners.
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    CUDA_ARCHS="native"
else
    CUDA_ARCHS="80-real"
fi

# On headless CI runners without an NVIDIA driver, libcuda.so is missing.
# The CUDA toolkit ships stub libraries for compile-only builds — add them
# to LIBRARY_PATH so the linker can resolve -lcuda.
for stubs_candidate in \
    /usr/local/cuda/targets/x86_64-linux/lib/stubs \
    /usr/local/cuda/lib64/stubs \
    /usr/local/cuda/stubs; do
    if [ -f "$stubs_candidate/libcuda.so" ]; then
        export LIBRARY_PATH="${stubs_candidate}${LIBRARY_PATH:+:$LIBRARY_PATH}"
        break
    fi
done

cmake --preset="${PRESET}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_FIND_ROOT_PATH="${PREFIX}" \
    -DCLIO_CORE_ENABLE_CONDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"

cmake --build build --parallel "${CPU_COUNT}"
cmake --install build
