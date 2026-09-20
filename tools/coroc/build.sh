#!/usr/bin/env bash
# Build clio-coroc without cmake.
#
# The tool only READS source, so its clang need not be -- and on Aurora is not
# -- the compiler that builds the project. Point LLVM_ROOT at any stock LLVM
# with development files; the project itself is still built by icpx/nvcc/hipcc
# and the two never read the same headers.
set -euo pipefail

LLVM_ROOT="${LLVM_ROOT:-$(ls -d /opt/aurora/*/spack/unified/*/install/linux-x86_64/llvm-develop* 2>/dev/null | head -1)}"
if [[ -z "${LLVM_ROOT}" || ! -x "${LLVM_ROOT}/bin/llvm-config" ]]; then
  echo "set LLVM_ROOT to an LLVM install with development files" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${HERE}/../../build-coroc}"
mkdir -p "${OUT}"

CFG="${LLVM_ROOT}/bin/llvm-config"
RTTI_FLAG="-fno-rtti"
if [[ "$(${CFG} --has-rtti)" == "YES" ]]; then RTTI_FLAG=""; fi

# Bake in where clang's builtin headers and the host C++ library live, so the
# caller passes only -I and -D. The tool's toolchain and the project's are
# unrelated: on Aurora the project is built by icpx and this by GCC + LLVM.
CLANG_MAJOR="$(${CFG} --version | cut -d. -f1)"
RESOURCE_DIR="$(${CFG} --libdir)/clang/${CLANG_MAJOR}"
GCC_TOOLCHAIN="${GCC_TOOLCHAIN:-$(dirname "$(dirname "$(command -v g++)")")}"

set -x
g++ -std=c++20 -O2 ${RTTI_FLAG} \
    -I"${LLVM_ROOT}/include" \
    -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS \
    -D__STDC_LIMIT_MACROS \
    -DCLIO_COROC_RESOURCE_DIR="\"${RESOURCE_DIR}\"" \
    -DCLIO_COROC_GCC_TOOLCHAIN="\"${GCC_TOOLCHAIN}\"" \
    "${HERE}/main.cc" \
    -L"${LLVM_ROOT}/lib" -Wl,-rpath,"${LLVM_ROOT}/lib" \
    -lclang-cpp \
    -o "${OUT}/clio-coroc"
