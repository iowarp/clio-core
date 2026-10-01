#!/usr/bin/env bash
# Build clio-coroc without cmake.
#
# The tool only READS source, so its clang need not be -- and on Aurora is not
# -- the compiler that builds the project. Point LLVM_ROOT at any stock LLVM
# with development files; the project itself is still built by icpx/nvcc/hipcc
# and the two never read the same headers.
#
# ONE CONSTRAINT TIES THEM: the tool parses the project's sources, so when
# those are CUDA its clang must recognise the toolkit they include. clang
# hard-errors on a CUDA it does not know -- clang 20 cannot parse the CUDA 13
# headers at all -- so building this against the distro's oldest LLVM and then
# pointing it at a current toolkit fails in step one. Prefer the newest LLVM
# available; LLVM_ROOT overrides the choice.
set -euo pipefail

pick_llvm_root() {
  if [[ -n "${LLVM_ROOT:-}" ]]; then echo "${LLVM_ROOT}"; return; fi
  local d
  d=$(ls -d /opt/aurora/*/spack/unified/*/install/linux-x86_64/llvm-develop* 2>/dev/null | head -1)
  if [[ -n "${d}" ]]; then echo "${d}"; return; fi
  # Newest /usr/lib/llvm-N with an llvm-config, by version number.
  d=$(ls -d /usr/lib/llvm-* 2>/dev/null \
        | sed 's|.*/llvm-||' | grep -E '^[0-9]+$' | sort -n | tail -1)
  if [[ -n "${d}" && -x "/usr/lib/llvm-${d}/bin/llvm-config" ]]; then
    echo "/usr/lib/llvm-${d}"; return
  fi
  if command -v llvm-config > /dev/null; then
    "$(command -v llvm-config)" --prefix; return
  fi
}

LLVM_ROOT="$(pick_llvm_root)"
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
LIBDIR="$(${CFG} --libdir)"

# The clang AST headers can be split out of the LLVM prefix (Debian keeps them
# in libclang-N-dev, which llvm-N-dev does not pull in). CLANG_INCLUDE_DIR
# names a second prefix's include/ when they are not beside llvm's.
CLANG_INCLUDE_DIR="${CLANG_INCLUDE_DIR:-}"
if [[ ! -f "$(${CFG} --includedir)/clang/AST/ASTConsumer.h" && -z "${CLANG_INCLUDE_DIR}" ]]; then
  echo "clang's AST headers are not in $(${CFG} --includedir)." >&2
  echo "Install the matching clang dev package (Debian: libclang-${LLVM_VER:-N}-dev)," >&2
  echo "or set CLANG_INCLUDE_DIR to the prefix that has include/clang/AST/." >&2
  exit 1
fi

# Linking: -lclang-cpp needs a bare .so, and distros ship only the versioned
# soname in the runtime package. Fall back to it, through a symlink farm, so a
# stock install works without root. libLLVM likewise: on Debian libclang-cpp
# does not carry LLVM's symbols, so link it explicitly when it is there.
STUB="${OUT}/.linkstub"
rm -rf "${STUB}"; mkdir -p "${STUB}"
link_stub() {  # link_stub <basename>  -> echoes -l<name> if it resolved
  local name="$1" so
  for so in "${LIBDIR}/lib${name}.so" \
            "${LIBDIR}"/lib${name}.so.* \
            /usr/lib/"$(uname -m)"-linux-gnu/lib${name}.so.*; do
    [[ -e "${so}" ]] || continue
    ln -sf "$(readlink -f "${so}")" "${STUB}/lib${name}.so"
    echo "-l${name}"
    return
  done
}
LIBS="$(link_stub clang-cpp)"
[[ -n "${LIBS}" ]] || { echo "no libclang-cpp found under ${LIBDIR}" >&2; exit 1; }
LIBS="${LIBS} $(link_stub LLVM)"

# Bake in where clang's builtin headers and the host C++ library live, so the
# caller passes only -I and -D. The tool's toolchain and the project's are
# unrelated: on Aurora the project is built by icpx and this by GCC + LLVM.
CLANG_MAJOR="$(${CFG} --version | cut -d. -f1)"
RESOURCE_DIR="${LIBDIR}/clang/${CLANG_MAJOR}"
GCC_TOOLCHAIN="${GCC_TOOLCHAIN:-$(dirname "$(dirname "$(command -v g++)")")}"

set -x
g++ -std=c++20 -O2 ${RTTI_FLAG} \
    ${CLANG_INCLUDE_DIR:+-I"${CLANG_INCLUDE_DIR}"} \
    -I"$(${CFG} --includedir)" \
    -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS \
    -D__STDC_LIMIT_MACROS \
    -DCLIO_COROC_RESOURCE_DIR="\"${RESOURCE_DIR}\"" \
    -DCLIO_COROC_GCC_TOOLCHAIN="\"${GCC_TOOLCHAIN}\"" \
    "${HERE}/main.cc" \
    -L"${STUB}" -Wl,-rpath,"${LIBDIR}" \
    -Wl,-rpath,/usr/lib/"$(uname -m)"-linux-gnu \
    ${LIBS} \
    -o "${OUT}/clio-coroc"
