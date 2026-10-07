#!/usr/bin/env bash
# Build environment for the dtschedule worktree on the ares head node.
#
# Host toolchain (gcc 11.4, glibc 2.35) plus spack-installed dependencies.
# `spack load` is broken in this shell (it looks for a python under a
# miniconda that no longer exists), so prefixes are resolved with
# `spack location -i` and passed through CMAKE_PREFIX_PATH instead.
#
# Usage:
#   source scripts/dtschedule_env.sh
#   cmake --preset release-fuse -DCLIO_CTE_ENABLE_COMPRESS=ON \
#         -DCMAKE_PREFIX_PATH="$DT_PREFIX_PATH"
#   cmake --build build -j 32
#
# System cmake is 3.22; the assimilation engine needs >= 3.25, so spack's
# cmake 3.31 is put first on PATH.

_dt_spack_root=/mnt/common/llogan/spack
# shellcheck disable=SC1091
. "${_dt_spack_root}/share/spack/setup-env.sh" 2>/dev/null

_dt_specs=(cereal yaml-cpp libzmq boost hdf5 catch2 msgpack-c nlohmann-json
           libaio curl@8.15.0 openmpi libpressio libstdcompat sz3 zfp fpzip
           zstd libfuse snappy brotli bzip2 c-blosc2)

DT_PREFIX_PATH=""
for _spec in "${_dt_specs[@]}"; do
  _dir=$(spack location -i "${_spec}" 2>/dev/null | head -1)
  if [ -n "${_dir}" ]; then
    DT_PREFIX_PATH="${DT_PREFIX_PATH:+${DT_PREFIX_PATH};}${_dir}"
  else
    echo "dtschedule_env: spack package not installed: ${_spec}" >&2
  fi
done
export DT_PREFIX_PATH

# pkg-config needs every spack .pc dir (libzmq.pc pulls libsodium/libbsd).
_dt_pc=$(find "${_dt_spack_root}/opt/spack/linux-skylake_avx512" -maxdepth 3 \
           -type d -name pkgconfig 2>/dev/null | paste -sd:)
export PKG_CONFIG_PATH="/tmp/llogan_pc:${_dt_pc}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

# Two cmake 3.31.9 installs exist; `location -i` refuses the ambiguity.
_dt_cmake=$(spack find --format "{prefix}" cmake@3.31.9 2>/dev/null | head -1)
[ -n "${_dt_cmake}" ] && export PATH="${_dt_cmake}/bin:${PATH}"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset _dt_spack_root _dt_specs _spec _dir _dt_pc _dt_cmake
