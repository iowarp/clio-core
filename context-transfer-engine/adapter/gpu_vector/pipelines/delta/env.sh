# Build/run environment for the gpu_vector CUDA-clang stack on NCSA Delta.
#
# Source it (do not execute):  source .../pipelines/delta/env.sh
#
# THE THREE THINGS DELTA GETS WRONG BY DEFAULT
#
# 1. CUDA VERSION. Delta's default `cudatoolkit/26.5_13.2` is CUDA 13.2, and
#    clang refuses it (`crt/math_functions.hpp: expected function body after
#    function declarator` -- 13.x uses a __func__ macro form clang's CUDA
#    wrapper cannot parse). The newest CUDA any clang here accepts is 12.6,
#    and Delta ships a FULL 12.6.3 toolkit outside lmod at
#    /sw/external/cuda/cuda-12.6.3. The nvhpc `cuda/12.9` module is not a
#    substitute: its cuda/ tree has no curand headers (they live in
#    math_libs/), and clang's __clang_cuda_runtime_wrapper.h includes
#    curand_mtgp32_kernel.h unconditionally.
#
# 2. CPATH. The cudatoolkit module prepends the 13.2 include dir to CPATH,
#    which OVERRIDES --cuda-path -- so unloading is not enough, CPATH has to
#    be cleared as well or clang picks 13.2 headers under a 12.6 --cuda-path.
#
# 3. NO CONDA. The `iowarp` conda env this tree used to build against is
#    gone; the C++ dependencies now come from the spack instance at
#    ${SPACK_ROOT} (`spack install iowarp` -- the deps landed even though the
#    iowarp package itself is built from this source tree by hand).
#
# clang is the module `llvm/19.1.7`, referenced by ABSOLUTE PATH rather than
# `module load`: loading it swaps out gcc-native (Lmod compiler family) and
# deactivates cray-mpich, and we want gcc as the HOST compiler with clang used
# only as the CUDA compiler.

# `module` is a shell function, and an sbatch script is not a login shell --
# sbatch exports MODULEPATH/LOADEDMODULES but not the function, so a
# pre_cmds: hook that just calls `module` dies with "command not found".
if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
fi

module load cmake/3.31.8 2>/dev/null
module unload cudatoolkit 2>/dev/null

# See (2): the 13.2 include dirs outrank --cuda-path if they stay here.
unset CPATH

export CLIO_DELTA_CUDA_HOME="${CLIO_DELTA_CUDA_HOME:-/sw/external/cuda/cuda-12.6.3}"
export CLIO_DELTA_CLANGXX="${CLIO_DELTA_CLANGXX:-/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/llvm-19.1.7-zbfboml/bin/clang++}"
export CUDA_HOME="$CLIO_DELTA_CUDA_HOME"
export CUDAToolkit_ROOT="$CLIO_DELTA_CUDA_HOME"
export PATH="$CLIO_DELTA_CUDA_HOME/bin:$PATH"

# The spack deps (cereal, catch2, yaml-cpp, msgpack-c, libzmq,
# nlohmann-json, libaio, curl, spdlog). CMAKE_PREFIX_PATH rather than
# `spack load`, which needs an interactive shell hook and drags in its own
# python.
# THE INSTANCE MUST NOT LIVE UNDER THE SOURCE TREE. It used to be
# clio-core/spack, and CMake then refuses to generate: every external
# dependency dir lands in an exported INTERFACE_INCLUDE_DIRECTORIES, and
# CMake rejects an exported path "prefixed in the source directory"
# (lightbeam on zmq/sodium/bsd, aio on libaio). The real instance is now
# /u/llogan/spack with clio-core/spack left as a compat symlink so the
# absolute prefixes baked into the installed RPATHs still resolve.
CLIO_SPACK_ROOT="${CLIO_SPACK_ROOT:-${SPACK_ROOT:-/u/llogan/spack}}"
# CANONICALIZE. clio-core/spack is a compat symlink to the real instance (see
# above); SPACK_ROOT is often set to that symlink, and a search prefix under
# it makes find_path record a source-prefixed path all over again.
CLIO_SPACK_ROOT="$(readlink -f "$CLIO_SPACK_ROOT" 2>/dev/null || echo "$CLIO_SPACK_ROOT")"
CLIO_SPACK_OPT="$CLIO_SPACK_ROOT/opt/spack/linux-zen3"
if [ -d "$CLIO_SPACK_OPT" ]; then
  for _p in "$CLIO_SPACK_OPT"/*; do
    [ -d "$_p" ] && CMAKE_PREFIX_PATH="$_p:${CMAKE_PREFIX_PATH:-}"
  done
  unset _p
  export CMAKE_PREFIX_PATH
fi

# gcc-toolset-14 is the host toolchain (PrgEnv-gnu wraps it). clang must use
# the SAME libstdc++ or the C++20 objects it produces will not link against
# the g++-built ones: left to itself it picks /usr/lib/gcc/.../11.
export CLIO_DELTA_GCC_DIR="${CLIO_DELTA_GCC_DIR:-/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14}"

# The build tree. Benchmark binaries live in $CLIO_GV_BUILD/bin; the sweep
# pkg calls them by bare name, so this has to be on PATH for every cell.
export CLIO_GV_BUILD="${CLIO_GV_BUILD:-/u/llogan/clio-core/build-clang}"
if [ -d "$CLIO_GV_BUILD/bin" ]; then
  export PATH="$CLIO_GV_BUILD/bin:$PATH"
  export LD_LIBRARY_PATH="$CLIO_GV_BUILD/bin:${LD_LIBRARY_PATH:-}"
fi

# Warmed memory: pre-fault the WHOLE RAM tier at compose so timings never
# include first-touch page population. The pkg sets this per cell too;
# exporting here covers anything jarvis spawns besides.
export CLIO_PREFAULT="${CLIO_PREFAULT:-0}"
