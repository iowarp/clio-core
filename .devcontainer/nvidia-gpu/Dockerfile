# IOWarp NVIDIA GPU Development Container
# Inherits from deps-nvidia which has all build dependencies including CUDA
#
# This devcontainer adds development-specific tools like Claude Code.
#
FROM iowarp/deps-nvidia:latest
USER root
RUN userdel -r ubuntu || true

# FUSE 3 runtime (fusermount3) + dev headers, needed to build the libfuse
# adapter (clio_cte_fuse) and run its mount integration tests. Installed here
# so a devcontainer rebuild picks it up immediately; it is also in the base
# image (docker/iowarp-base.Dockerfile) as the canonical default. The
# container runs --privileged, so /dev/fuse + SYS_ADMIN are available.
RUN apt-get update && apt-get install -y --no-install-recommends \
      fuse3 libfuse3-dev \
      libboost-context-dev libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# Boost.Context stackful coroutine backend available + on by default (issue #620).
ENV CLIO_CORE_ENABLE_BOOST_COROUTINES=ON

# The deps-nvidia image already includes (inherited from deps-cpu):
# - All apt + source-built dependencies (boost, hdf5, yaml-cpp, zeromq, cereal, etc.)
# - Compression libraries (zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, c-blosc2)
# - Lossy compression (ZFP, SZ3, FPZIP, LibPressio)
# - ADIOS2 with HDF5 and ZeroMQ support
# - Docker-in-Docker support
# - Jarvis (IOWarp runtime deployment tool)
# - MPI (OpenMPI)
# - libaio (for bdev ChiMod)
#
# Plus NVIDIA-specific:
# - CUDA 12.6 toolkit and runtime libraries
# - NVIDIA Container Toolkit
# - GPU environment variables (CUDA_HOME, NVIDIA_VISIBLE_DEVICES, etc.)

# Install nvcomp (NVIDIA GPU compression library) for the CUDA 12 toolchain.
# The redistributable tarball ships include/, lib/ (libnvcomp.so) and CMake
# package files; installing into /usr/local makes find_package(nvcomp) work.
ARG NVCOMP_VERSION=4.2.0.11
RUN set -eux; \
    cd /tmp; \
    archive="nvcomp-linux-x86_64-${NVCOMP_VERSION}_cuda12-archive"; \
    curl -fsSLO "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/${archive}.tar.xz"; \
    tar -xf "${archive}.tar.xz"; \
    cp -a "${archive}/include/." /usr/local/include/; \
    cp -a "${archive}/lib/." /usr/local/lib/; \
    ldconfig; \
    rm -rf /tmp/nvcomp*

# Scientific-data GPU compressors that back three more entries in the GPU
# compression factory (ctp::Cusz / ctp::Ndzip / ctp::Cuszp), complementing
# nvcomp's general-purpose byte codecs. All are built from source against the
# CUDA 12 toolkit and installed into /usr/local so the root CMake auto-detects
# them (CLIO_CTP_ENABLE_CUSZ / CLIO_CTP_ENABLE_NDZIP / CLIO_CTP_ENABLE_CUSZP):
#   - cuSZ  (szcompressor/cuSZ)  : error-bounded LOSSY float compression
#   - ndzip (celerity/ndzip)     : high-throughput LOSSLESS float compression
#   - cuSZp (szcompressor/cuSZp) : ultra-fast single-kernel LOSSY float compression
# ndzip and cuSZp have no upstream Spack/apt package, so source-building them here
# is the only option; cuSZ is also a Spack package (see installers/spack) but is
# source-built here to match the nvcomp/zfp-sycl pattern. Target SM versions are
# overridable (default spans Volta..Hopper).
ARG CUDA_ARCHS="70;75;80;86;89;90"

# cuSZ - GPU error-bounded lossy compressor. Pinned to master; the ctp::Cusz
# wrapper targets cuSZ's psz_* resource-manager C API.
RUN set -eux; \
    cd /tmp; \
    git clone --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/szcompressor/cuSZ.git cusz_src; \
    cmake -S cusz_src -B cusz_src/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DPSZ_BACKEND=cuda \
      -DPSZ_BUILD_EXAMPLES=off \
      -DBUILD_TESTING=off \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5; \
    cmake --build cusz_src/build -j"$(nproc)"; \
    cmake --install cusz_src/build; \
    ldconfig; \
    rm -rf /tmp/cusz_src

# cuSZp - GPU ultra-fast single-kernel error-bounded lossy compressor. Pinned to
# cuSZp-V3.0.0; the ctp::Cuszp wrapper targets the cuSZp v3 C API
# (cuSZp_compress / cuSZp_decompress, header <cuSZp.h>, library -lcuSZp).
RUN set -eux; \
    cd /tmp; \
    git clone --depth 1 --branch cuSZp-V3.0.0 \
      https://github.com/szcompressor/cuSZp.git cuszp_src; \
    cmake -S cuszp_src -B cuszp_src/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DcuSZp_BUILD_EXAMPLES=off \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5; \
    cmake --build cuszp_src/build -j"$(nproc)"; \
    cmake --install cuszp_src/build; \
    ldconfig; \
    rm -rf /tmp/cuszp_src

# ndzip - GPU high-throughput lossless compressor (CUDA backend).
#
# Built with CLANG as the CUDA compiler (clang++ -x cuda), per ndzip's own build
# docs -- NOT nvcc. nvcc's crt/host_defines.h defines __noinline__ as a macro
# (__attribute__((noinline))) that nests illegally inside libstdc++'s own
# __attribute__((__noinline__)) when ndzip's device TUs pull in <memory>/<string>,
# so the nvcc build fails (ndzip ships a single #undef workaround that is mis-
# ordered for this toolchain). clang-18 (already installed above for CUDA kernels)
# does not have that macro and compiles ndzip cleanly. Notes:
#   - libomp (clang OpenMP) is required by ndzip's CPU codec (cpu_codec.inl).
#   - -march=native is recommended by ndzip's docs (affects the host/CPU codec; the
#     CUDA kernels are arch-targeted via CMAKE_CUDA_ARCHITECTURES). Relax it if this
#     image must run on a different CPU than it was built on.
#   - ndzip ships no install() rules, so the libs + headers are copied manually.
#   - clang-18 targets up to sm_90; PTX from the highest arch JITs forward to newer
#     GPUs (e.g. sm_120 / Blackwell) at runtime.
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends libomp-18-dev; \
    rm -rf /var/lib/apt/lists/*; \
    cd /tmp; \
    git clone --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/celerity/ndzip.git ndzip_src; \
    cmake -S ndzip_src -B ndzip_src/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DCMAKE_CXX_FLAGS="-march=native" \
      -DCMAKE_CUDA_FLAGS="--cuda-path=${CUDA_HOME} -Wno-unknown-cuda-version" \
      -DNDZIP_WITH_CUDA=ON \
      -DNDZIP_WITH_MT=OFF \
      -DNDZIP_BUILD_BENCHMARK=OFF \
      -DNDZIP_BUILD_TEST=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5; \
    cmake --build ndzip_src/build -j"$(nproc)"; \
    cp ndzip_src/build/libndzip.so ndzip_src/build/libndzip-cuda.so /usr/local/lib/; \
    cp -r ndzip_src/include/ndzip /usr/local/include/; \
    ldconfig; \
    rm -rf /tmp/ndzip_src

# Intel oneAPI 2024.2 (SYCL/DPC++) toolchain for the GPU compression factory's
# SYCL backend (zfp's SYCL execution policy, mirroring how nvcomp backs the CUDA
# path). Pinned to 2024.2 on purpose: the zfp SYCL port
# (alpers-git/zfp:sycl/develop) is DPCT-generated and requires the matching
# dpcpp-ct (DPCT) runtime headers — newer oneAPI runtimes reject its nested
# device_memory init pattern with "queue::submit cannot be nested".
#   - compiler-dpcpp-cpp : icx/icpx + the CPU OpenCL runtime, so SYCL code
#                          builds AND runs without an Intel GPU (CPU SYCL device)
#   - dpcpp-ct           : DPCT/SYCLomatic helper headers (dpct/dpct.hpp)
#   - libdpstd-devel     : oneDPL (oneapi/dpl/execution), a port dependency
RUN set -eux; \
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list; \
    apt-get update; \
    dpkg -P --force-all hipcc hip-dev hip-runtime-amd rocm-llvm hsa-rocr-dev rocm-device-libs || true; \
    DEBIAN_FRONTEND=noninteractive apt-get --fix-broken install -y; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-dpcpp-cpp-2024.2 \
      intel-oneapi-dpcpp-ct-2024.2 \
      intel-oneapi-libdpstd-devel-2022.6; \
    rm -rf /var/lib/apt/lists/*

# Source the oneAPI environment in every shell so cmake/icpx "just work" in the
# devcontainer terminal without manual `source setvars.sh`.
RUN printf '. /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1\n' \
      > /etc/profile.d/zz-oneapi.sh; \
    chmod +x /etc/profile.d/zz-oneapi.sh; \
    echo '. /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1' >> /etc/bash.bashrc

# Build the SYCL-enabled libzfp (the codec behind ctp::SyclZfp / the "zfp-sycl"
# factory entry) and install it to a DEDICATED prefix /opt/zfp-sycl, so it does
# not shadow the stock CPU libzfp that ADIOS2/LibPressio link against. The root
# CMake auto-detects it there (CLIO_CTP_ENABLE_ZFP_SYCL). cp -a preserves the
# libzfp.so.1 SONAME symlink chain (a bare copy makes the loader fall back to
# the stock non-SYCL libzfp at runtime).
ARG ZFP_SYCL_REF=sycl/develop
# Use a bash LOGIN shell so /etc/profile.d/zz-oneapi.sh sources setvars during
# init (puts icx/icpx on PATH). Sourcing setvars inline under /bin/sh (dash) or
# non-login bash fails; the login shell is the invocation that works.
SHELL ["/bin/bash", "-lc"]
RUN set -eux; \
    git clone --depth 1 -b "${ZFP_SYCL_REF}" https://github.com/alpers-git/zfp.git /tmp/zfp; \
    dpct_inc="$(dirname "$(dirname "$(find /opt/intel/oneapi/dpcpp-ct -name dpct.hpp | head -1)")")"; \
    dpl_inc="$(dirname "$(dirname "$(dirname "$(find /opt/intel/oneapi/dpl -path '*oneapi/dpl/execution' -not -path '*pstl_offload*' | head -1)")")")"; \
    if [ -d /opt/adaptivecpp ]; then mv /opt/adaptivecpp /opt/adaptivecpp_hidden; fi; \
    cmake -S /tmp/zfp -B /tmp/zfp/build -DCMAKE_BUILD_TYPE=Release -DZFP_WITH_SYCL=1 \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_CXX_FLAGS="-I${dpct_inc} -I${dpl_inc}" \
      -DBUILD_TESTING=OFF -DBUILD_UTILITIES=OFF; \
    cmake --build /tmp/zfp/build --target zfp -j"$(nproc)"; \
    mkdir -p /opt/zfp-sycl/lib /opt/zfp-sycl/include; \
    cp -a /tmp/zfp/build/lib/libzfp.so* /opt/zfp-sycl/lib/; \
    cp -a /tmp/zfp/include/. /opt/zfp-sycl/include/; \
    rm -rf /tmp/zfp; \
    if [ -d /opt/adaptivecpp_hidden ]; then mv /opt/adaptivecpp_hidden /opt/adaptivecpp; fi

# Prefer the SYCL libzfp at runtime (must come before /usr/lib's stock libzfp on
# the loader path; oneAPI libs are already on LD_LIBRARY_PATH via setvars above,
# which resolves libzfp's libsycl dependency).
RUN printf 'export LD_LIBRARY_PATH=/opt/zfp-sycl/lib:${LD_LIBRARY_PATH}\n' \
      >> /etc/profile.d/zz-oneapi.sh; \
    echo 'export LD_LIBRARY_PATH=/opt/zfp-sycl/lib:${LD_LIBRARY_PATH}' >> /etc/bash.bashrc

# Remap iowarp user UID/GID to match the host user.
# This avoids file permission issues with bind-mounted volumes.
# Override at build time: --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g)
ARG HOST_UID=1000
ARG HOST_GID=1000

USER root
# Ubuntu 24.04 ships a default "ubuntu" user at UID 1000 which blocks
# updateRemoteUserUID and manual UID remapping (see
# https://github.com/microsoft/vscode-remote-release/issues/10030).
# Move it out of the way so iowarp can claim the host user's UID.
RUN if id ubuntu >/dev/null 2>&1; then \
      usermod -u 59999 ubuntu && \
      groupmod -g 59999 ubuntu; \
    fi && \
    OLD_UID=$(id -u iowarp) && OLD_GID=$(id -g iowarp) && \
    NEW_UID="${HOST_UID:-1000}" && NEW_GID="${HOST_GID:-1000}" && \
    if [ "${NEW_UID}" != "${OLD_UID}" ] || [ "${NEW_GID}" != "${OLD_GID}" ]; then \
      groupmod -o -g "${NEW_GID}" iowarp && \
      usermod  -o -u "${NEW_UID}" -g "${NEW_GID}" iowarp && \
      chown -R "${NEW_UID}:${NEW_GID}" /home/iowarp; \
    fi

USER iowarp
WORKDIR /workspace

# Install Claude Code for AI-assisted development
RUN curl -fsSL https://claude.ai/install.sh | bash

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
