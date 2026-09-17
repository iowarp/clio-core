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

# ---------------------------------------------------------------------------
# NVSHMEM MULTI-NODE. Source this and `srun --mpi=pmi2` and it works; every
# line below is load-bearing, and each was a separate failure.
#
# BOOTSTRAP THROUGH THE LAUNCHER, NOT MPI. NVSHMEM's MPI bootstrap is a
# dlopen'd plugin NVIDIA built against OpenMPI, so it needs libmpi.so.40 --
# but Delta's system MPI is cray-mpich, and the HPC-X OpenMPI inside nvhpc
# cannot connect across Slingshot at all ("ucp_ep_create failed: Destination
# is unreachable" from its UCX PML). PMI sidesteps both. NVSHMEM_BOOTSTRAP
# takes the FAMILY and NVSHMEM_BOOTSTRAP_PMI the VARIANT -- putting "PMI2"
# in the first one fails with "bootstrap_preinit failed". PMIX is not usable
# here: nvshmem_bootstrap_pmix.so needs HPC-X's libpmix, which has an
# undefined opal_libevent2022_evthread_use_pthreads.
export NVSHMEM_BOOTSTRAP="${NVSHMEM_BOOTSTRAP:-PMI}"
export NVSHMEM_BOOTSTRAP_PMI="${NVSHMEM_BOOTSTRAP_PMI:-PMI2}"

# THE TRANSPORT MUST BE NAMED. Left alone, NVSHMEM defaults to ibrc,
# enumerates InfiniBand devices, finds none on a Slingshot machine, and dies
# with "building transport map failed". Delta's fabric is CXI through
# libfabric 2.3.1 (fi_info -p cxi resolves on the compute nodes).
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-libfabric}"
export NVSHMEM_LIBFABRIC_PROVIDER="${NVSHMEM_LIBFABRIC_PROVIDER:-cxi}"

# CXI TUNING NVSHMEM ASKS FOR IN ITS OWN STRINGS. The libfabric transport
# carries the warnings "FI_CXI_OPTIMIZED_MRS is set. This may cause a hang at
# runtime if the value is not 0" and "FI_CXI_DISABLE_HMEM_DEV_REGISTER ... may
# cause issues with initialization if the value is not 1". Without them the
# run reaches the data path and then hangs in nvshmemt_libfabric_quiet with
# "Connection timed out".
export FI_CXI_OPTIMIZED_MRS="${FI_CXI_OPTIMIZED_MRS:-0}"
export FI_CXI_DISABLE_HMEM_DEV_REGISTER="${FI_CXI_DISABLE_HMEM_DEV_REGISTER:-1}"

# MEASURED with this recipe: NVIDIA's own shmem_put_bw across 2 nodes reaches
# 11.3 GB/s at 4 MiB; clio_kmeans_nvshmem_bench passes its gates at 2 PEs on
# 2 nodes (455.6 ms / 20 iters) and 4 PEs on 4 nodes (294.7 ms), and
# clio_grayscott_nvshmem_bench's halo exchange passes at 4 PEs on 4 nodes.
#
# The benches must be built with CLIO_GV_NVSHMEM_MPI_BOOTSTRAP=OFF, which
# makes them call nvshmem_init() and honour the variables above instead of
# bootstrapping through MPI.

# ---------------------------------------------------------------------------
# THE MPI AND NCCL BASELINES MULTI-NODE.
#
# Everything above concerns NVSHMEM. The other two CTE-free baselines have
# their own multi-node story, and both were silently broken until they were
# checked against the mechanism rather than the exit code.
#
# THE MPI BASELINES NEED CRAY-MPICH, NOT HPC-X. find_package(MPI) resolves to
# the HPC-X OpenMPI inside nvhpc (it is on PATH for NVSHMEM's sake, see
# above), and that MPI cannot connect across Slingshot at all:
#
#   ucp_ep_create(proc=1) failed: Destination is unreachable
#   Failed to resolve UCX endpoint for rank 1
#
# so an mpirun across two nodes dies in MPI_Init or the first collective. The
# system MPI is cray-mpich and it speaks CXI natively. Baselines built
# against it are produced by build_baselines_cray.sh into build*/bin-cray.
export CLIO_DELTA_CRAY_MPI="${CLIO_DELTA_CRAY_MPI:-/opt/cray/pe/mpich/9.1.0/ofi/GNU/11.2}"
export CLIO_DELTA_CRAY_PMI="${CLIO_DELTA_CRAY_PMI:-/opt/cray/pe/pmi/default}"
if [ -d "$CLIO_DELTA_CRAY_MPI/lib" ]; then
  export LD_LIBRARY_PATH="$CLIO_DELTA_CRAY_MPI/lib:$CLIO_DELTA_CRAY_PMI/lib:${LD_LIBRARY_PATH:-}"
fi

# THE LAUNCHER DIFFERS BY SUBSTRATE, and this is not cosmetic:
#
#   cray-mpich (mpi, nccl)   srun --mpi=cray_shasta
#   NVSHMEM                  srun --mpi=pmi2   (NVSHMEM_BOOTSTRAP_PMI=PMI2)
#
# Launching a cray-mpich binary under --mpi=pmi2 does NOT fail. It gets no
# PMI info, does a SINGLETON MPI_Init, and every task runs as its own 1-rank
# job -- two independent serial runs that print "ranks=1" twice and pass
# every science gate, because a gate checks the answer and nothing checks
# that the run was distributed. Assert the rank count in any harness.
export CLIO_DELTA_SRUN_MPI="${CLIO_DELTA_SRUN_MPI:-cray_shasta}"
export CLIO_DELTA_SRUN_NVSHMEM="${CLIO_DELTA_SRUN_NVSHMEM:-pmi2}"

# NCCL NEEDS THE SLINGSHOT PLUGIN OR IT SILENTLY USES TCP. NCCL's built-in
# transports are NVLink/PCIe and IB verbs; on Slingshot it finds no IB device
# and falls back to sockets with GPU Direct RDMA disabled:
#
#   NET/Plugin: Could not find: libnccl-net.so
#   NET/IB : No device found.
#   Using network Socket
#
# which still runs, still passes gates, and is not a Slingshot number.
#
# NEVER `module load aws-ofi-nccl ... | sed`. A pipe runs module in a
# SUBSHELL, so every setenv and prepend_path in the modulefile is discarded
# when it exits -- the banner prints, NCCL_NET_PLUGIN comes back unset, and
# NCCL quietly takes the socket path. These four are what `module show
# aws-ofi-nccl/1.19.2` sets, applied directly so there is no subshell to lose
# them in.
export CLIO_DELTA_OFI_NCCL="${CLIO_DELTA_OFI_NCCL:-/sw/rh9.4/user/aws-ofi-nccl-1.19.2-lf2.3.1-cu13.2}"
export CLIO_DELTA_LIBFABRIC="${CLIO_DELTA_LIBFABRIC:-/opt/cray/libfabric/2.3.1}"
if [ -d "$CLIO_DELTA_OFI_NCCL/lib" ]; then
  export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-ofi}"
  export FI_PROVIDER="${FI_PROVIDER:-cxi}"
  export LD_LIBRARY_PATH="$CLIO_DELTA_OFI_NCCL/lib:$CLIO_DELTA_LIBFABRIC/lib64:${LD_LIBRARY_PATH:-}"
  # The plugin needs this libfabric ahead of any other in the process.
  export CLIO_DELTA_NCCL_PRELOAD="$CLIO_DELTA_LIBFABRIC/lib64/libfabric.so.1"
fi
# CONFIRMED with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET:
#   NET/Plugin: Loaded net plugin Libfabric (v11)
#   NET/OFI Selected provider is cxi, fabric is cxi (found 1 nics)
#   Using network Libfabric      <-- NOT "Using network Socket"
# and all six NCCL baselines pass their gates at 2 ranks on 2 nodes over it.
# NOTE the confirming string is "Using network Libfabric"; a probe grepping
# for "OFI" finds nothing and looks like a failure.
