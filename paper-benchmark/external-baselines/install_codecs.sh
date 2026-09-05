#!/usr/bin/env bash
# Build and install the external GPU compressors the baseline campaign needs:
#
#   cuSZp v3.0.0   error-bounded lossy   (szcompressor/cuSZp)
#   ndzip          lossless float, CUDA  (celerity/ndzip)
#
# cuSZ is assumed already installed at $NPENV/cusz; it is not built here.
#
# Each needs a patch to build on a current toolchain. They are applied here
# rather than left as folklore:
#
#  1. cuSZp hardcodes CMAKE_CUDA_ARCHITECTURES "60 61 62 70 75 80 86" with a
#     plain set(), which overrides -D. CUDA 13 dropped Pascal AND Volta, so
#     nvcc dies on compute_60 and then compute_70. Pinned to $CUDA_ARCH.
#
#  2. ndzip's find_package(Boost REQUIRED) fails on a system with no Boost
#     headers, even though only its CLI, benchmark and OpenMP CPU path use
#     Boost -- the CUDA library links none of them. Made optional and the CLI
#     guarded behind Boost_FOUND.
#
#  3. ndzip's src/ndzip/cpu_codec.inl:15 uses `#ifdef NDZIP_OPENMP_SUPPORT`
#     while every other guard in the tree uses `#if`. CMake passes
#     -DNDZIP_OPENMP_SUPPORT=0, which `#ifdef` still counts as DEFINED, so the
#     Boost includes compile even with -DNDZIP_WITH_MT=OFF. Upstream never
#     hits this because it requires Boost unconditionally.
#
# ndzip ships no install() rules, so its artifacts are copied by hand.
#
#   install_codecs.sh [--arch 80] [--jobs 16]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
. "$HERE/env.sh"

CUDA_ARCH=80; JOBS=16
while [ $# -gt 0 ]; do
  case "$1" in
    --arch) CUDA_ARCH=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    *) echo "usage: install_codecs.sh [--arch N] [--jobs N]" >&2; exit 2 ;;
  esac
done
SRC=${SRC:-$WORK/ext-src}; mkdir -p "$SRC"
# Prefer a cmake >= 3.28 if one is on PATH ahead of an older system cmake:
# clio-core's set_tests_properties(DIRECTORY ...) needs it, and a 3.26 will
# fail configure with hundreds of "Can not find test to add properties to".
CMAKE=${CMAKE:-cmake}
echo "== using $($CMAKE --version | head -1); arch sm_$CUDA_ARCH; sources in $SRC"

# ---------------------------------------------------------------- cuSZp v3
if [ ! -e "$NPENV/cuszp/lib64/libcuSZp.so" ]; then
  echo "== cuSZp v3.0.0"
  [ -d "$SRC/cuszp" ] || git clone --depth 1 --branch cuSZp-V3.0.0 \
      https://github.com/szcompressor/cuSZp.git "$SRC/cuszp"
  # Patch 1: the arch list overrides -D, and CUDA 13 rejects sm_60/70.
  sed -i "s/^set(CMAKE_CUDA_ARCHITECTURES .*/set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})/" \
      "$SRC/cuszp/CMakeLists.txt"
  $CMAKE -S "$SRC/cuszp" -B "$SRC/cuszp/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$NPENV/cuszp" -DcuSZp_BUILD_EXAMPLES=OFF
  $CMAKE --build "$SRC/cuszp/build" -j "$JOBS"
  $CMAKE --install "$SRC/cuszp/build"
else
  echo "== cuSZp already at $NPENV/cuszp"
fi

# ----------------------------------------------------------------- ndzip
if [ ! -e "$NPENV/ndzip/lib/libndzip-cuda.so" ]; then
  echo "== ndzip (CUDA backend)"
  [ -d "$SRC/ndzip" ] || git clone --depth 1 https://github.com/celerity/ndzip.git "$SRC/ndzip"
  cd "$SRC/ndzip"
  # Patch 2: Boost is only needed by the CLI / benchmark / OpenMP CPU path.
  sed -i 's/find_package(Boost REQUIRED COMPONENTS thread program_options)/find_package(Boost COMPONENTS thread program_options)/' CMakeLists.txt
  grep -q 'if (Boost_FOUND)' CMakeLists.txt || python3 - <<'PY'
import re
s=open('CMakeLists.txt').read()
old="add_executable(compress\n    src/compress/compress.cc\n)"
if old in s and 'if (Boost_FOUND)' not in s:
    i=s.index(old); j=s.index('endif ()', s.index('NDZIP_USE_CUDA', i))+len('endif ()')
    s=s[:i]+"if (Boost_FOUND)\n"+s[i:j]+"\nendif ()"+s[j:]
    open('CMakeLists.txt','w').write(s)
PY
  # Patch 3: -DNDZIP_OPENMP_SUPPORT=0 is still "defined" to #ifdef.
  sed -i 's/^#ifdef NDZIP_OPENMP_SUPPORT/#if NDZIP_OPENMP_SUPPORT/' src/ndzip/cpu_codec.inl
  $CMAKE -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DNDZIP_WITH_CUDA=ON -DNDZIP_WITH_HIPSYCL=OFF -DNDZIP_WITH_MT=OFF \
      -DNDZIP_BUILD_TEST=OFF -DNDZIP_BUILD_BENCHMARK=OFF \
      -DNDZIP_WITH_3RDPARTY_BENCHMARKS=OFF
  $CMAKE --build build -j "$JOBS"
  # No install() rules upstream.
  mkdir -p "$NPENV/ndzip/include" "$NPENV/ndzip/lib"
  cp -r include/ndzip "$NPENV/ndzip/include/"
  cp build/libndzip.so build/libndzip-cuda.so "$NPENV/ndzip/lib/"
else
  echo "== ndzip already at $NPENV/ndzip"
fi

echo
echo "== installed =="
for f in "$NPENV/cuszp/lib64/libcuSZp.so" "$NPENV/ndzip/lib/libndzip-cuda.so"; do
  [ -e "$f" ] && echo "   $f" || echo "   MISSING: $f"
done
cat <<'MSG'

Next: configure clio-core with these on CMAKE_PREFIX_PATH, e.g.
  cmake -S <repo> -B <build> -G Ninja \
    -DCMAKE_PREFIX_PATH="$NPENV/np;$NPENV/cusz;$NPENV/ndzip;$NPENV/cuszp" \
    -DCLIO_CTP_ENABLE_COMPRESS=ON -DCLIO_CTE_ENABLE_COMPRESS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80
Confirm detection: CLIO_CTP_ENABLE_{CUSZ,CUSZP,NDZIP} must all be ON in
CMakeCache.txt, and ldd on libclio_cte_compressor_runtime.so must list
libcuSZp.so and libndzip-cuda.so. If they are OFF the arms still "run" --
WireIdForName falls back to zstd -- and produce plausible but wrong results.
MSG
