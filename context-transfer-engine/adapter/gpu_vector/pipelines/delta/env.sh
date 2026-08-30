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

# nvcc FOR THE MPI/NCCL/NVSHMEM BASELINES. Those benches are NOT clang
# targets -- each workload's CMakeLists builds them with a custom nvcc
# command, because they are CTE-free plain CUDA (a baseline that linked
# clio would not be a baseline). nvcc 12.6 refuses gcc > 13
# ("unsupported GNU version!"), and the host toolchain here is
# gcc-toolset-14, so point nvcc at gcc-toolset-13.
#
# NVCC_PREPEND_FLAGS rather than -ccbin in six CMakeLists: nvcc reads it
# from the environment, so it reaches every nvcc invocation the build
# makes without touching the build files. Safe precisely because these
# binaries link nothing from clio -- no ABI is shared with the gcc-14
# objects the rest of the tree produces.
export CLIO_DELTA_NVCC_CCBIN="${CLIO_DELTA_NVCC_CCBIN:-/opt/rh/gcc-toolset-13/root/usr/bin/g++}"
if [ -x "$CLIO_DELTA_NVCC_CCBIN" ]; then
  export NVCC_PREPEND_FLAGS="-ccbin $CLIO_DELTA_NVCC_CCBIN ${NVCC_PREPEND_FLAGS:-}"
fi

# NCCL and NVSHMEM for the GPU-communication baselines. Both ship inside the
# nvhpc 26.5 tree rather than as standalone modules, and the workload
# CMakeLists look for them through NCCL_HOME / NVSHMEM_HOME, so nothing finds
# them unless these are exported.
#
# The comm_libs are the CUDA 12.9 flavour while we build against 12.6.3.
# That is fine and deliberate: NCCL/NVSHMEM have a stable ABI across CUDA
# 12.x, they carry their own runtime linkage, and 12.9 is the oldest flavour
# nvhpc 26.5 ships (the alternative is 13.2, which the clang side cannot use
# at all). Delta's driver (595.x) is newer than both.
export NCCL_HOME="${NCCL_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/comm_libs/12.9/nccl}"
export NVSHMEM_HOME="${NVSHMEM_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/comm_libs/12.9/nvshmem}"
for _d in "$NCCL_HOME/lib" "$NVSHMEM_HOME/lib"; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"
done
unset _d

# OpenMPI FOR THE BASELINES, and specifically for NVSHMEM.
#
# Delta's system MPI is cray-mpich, which has no mpirun and whose library is
# libmpich.so. NVSHMEM's MPI bootstrap is a dlopen'd PLUGIN
# (nvshmem_bootstrap_mpi.so.3) that NVIDIA built against OpenMPI, so it hard
# requires libmpi.so.40:
#
#   bootstrap_loader.cpp:65: Bootstrap unable to load
#   'nvshmem_bootstrap_mpi.so.3' -- libmpi.so.40: cannot open shared object
#
# and the non-MPI path in the benches is single-PE by construction
# (nvshmemx_set_attr_uniqueid_args(0, 1, ...)), so it cannot answer whether
# NVSHMEM actually communicates. Putting cray-mpich in the process AND
# letting the plugin pull in OpenMPI would load two MPIs at once.
#
# So the CTE-free baselines (mpi / nccl / nvshmem -- none of which link
# anything from clio) build and run against the HPC-X OpenMPI that ships
# inside nvhpc, which is the MPI NVSHMEM and NCCL were built alongside.
# Use the CONCRETE path, not comm_libs/hpcx: that dispatcher resolves by
# matching the CUDA driver version and fails outright on a login node with
# no driver ("MPI matching the current driver version (0) ... was not
# installed").
export CLIO_DELTA_OMPI="${CLIO_DELTA_OMPI:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/comm_libs/12.9/hpcx/hpcx-2.50/ompi4}"
if [ -d "$CLIO_DELTA_OMPI" ]; then
  export PATH="$CLIO_DELTA_OMPI/bin:$PATH"
  export LD_LIBRARY_PATH="$CLIO_DELTA_OMPI/lib:${LD_LIBRARY_PATH:-}"
fi

# llvm-config FOR THE COROUTINE REGISTER CAP.
#
# cmake/ClioCoroRegCap.cmake builds an LLVM pass plugin (NVPTXCoroCap) that
# stamps nvvm.maxnreg on the kernels which transitively execute a device
# coroutine AND measured over budget. Its budget is
# CLIO_CORO_REGS_PER_SM / (CLIO_CORO_REF_THREADS * CLIO_CORO_TARGET_BLOCKS)
# = 65536 / (256 * 4) = 64 registers -- i.e. 4 blocks/SM.
#
# It is a NO-OP unless llvm-config is findable, and clang is referenced here
# by absolute path (so LLVM's bin is deliberately NOT on PATH), which left
# CLIO_LLVM_CONFIG-NOTFOUND and the cap silently inactive: the kmeans
# coroutine kernels came out at REG=192, i.e. ONE block/SM and 12.5%
# occupancy, against REG=32 and 100% for the CTE-free baselines.
#
# Pointed at explicitly rather than added to PATH, so nothing else in the
# build starts resolving through the LLVM prefix.
export CLIO_DELTA_LLVM_CONFIG="${CLIO_DELTA_LLVM_CONFIG:-$(dirname "$CLIO_DELTA_CLANGXX")/llvm-config}"
