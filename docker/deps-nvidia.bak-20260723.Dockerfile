# IOWarp NVIDIA GPU Dependencies Container
# Inherits from deps-cpu and adds CUDA/NVIDIA support
#
# Usage:
#   docker build -t iowarp/deps-nvidia:latest -f docker/deps-nvidia.Dockerfile .
#
FROM iowarp/deps-cpu:latest
LABEL maintainer="llogan@hawk.iit.edu"
LABEL version="1.0"
LABEL description="IOWarp NVIDIA GPU dependencies Docker image"

# Disable prompt during packages installation.
ARG DEBIAN_FRONTEND=noninteractive

#------------------------------------------------------------
# NVIDIA Container Toolkit and CUDA Installation
#------------------------------------------------------------

USER root

# Install NVIDIA Container Toolkit repository
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install CUDA from NVIDIA's official repository
# This installs the complete CUDA toolkit + runtime libraries for actual GPU execution
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-6 \
    cuda-cudart-12-6 \
    cuda-libraries-12-6 \
    cuda-nvrtc-12-6 \
    cuda-nvml-dev-12-6 \
    libcublas-12-6 \
    libcufft-12-6 \
    libcurand-12-6 \
    libcusolver-12-6 \
    libcusparse-12-6 \
    libnpp-12-6 \
    libnvidia-container-tools \
    libnvidia-container1 \
    && rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------
# Clang (CUDA-capable) Installation
#------------------------------------------------------------

# Install Clang 18 (Ubuntu 24.04 native) for compiling CUDA kernels via
# clang -x cuda --cuda-gpu-arch=sm_XX.  Clang 18 uses the stable legacy
# CUDA offload driver; clang 20+ has a new offload driver that segfaults
# on some device code in this project.
RUN apt-get update \
    && apt-get install -y --no-install-recommends clang-18 \
    && update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-18   100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 \
    && rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------
# CUDA Environment Configuration
#------------------------------------------------------------

# Set CUDA environment variables for runtime execution
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=${CUDA_HOME}/bin:${PATH}
# Set LD_LIBRARY_PATH to include CUDA, /usr/local, and system paths
ENV LD_LIBRARY_PATH=/usr/local/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:/usr/lib/x86_64-linux-gnu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

#------------------------------------------------------------
# Intel DPC++ SYCL Compiler Installation (production SYCL)
#------------------------------------------------------------

USER root

# Download and install Intel DPC++ nightly with CUDA/NVIDIA support
# The prebuilt release includes the CUDA adapter (libur_adapter_cuda.so)
RUN curl -sL "https://github.com/intel/llvm/releases/download/nightly-2026-04-26/sycl_linux.tar.gz" \
    -o /tmp/sycl_linux.tar.gz \
    && mkdir -p /opt/intel/dpcpp \
    && tar -xzf /tmp/sycl_linux.tar.gz -C /opt/intel/dpcpp --strip-components=0 \
    && rm /tmp/sycl_linux.tar.gz

ENV DPCPP_HOME=/opt/intel/dpcpp
ENV PATH=/opt/intel/dpcpp/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/intel/dpcpp/lib:${LD_LIBRARY_PATH}

#------------------------------------------------------------
# LLVM 18 dev packages (required by AdaptiveCpp, built below)
#------------------------------------------------------------
#
# Installed BEFORE the ROCm section on purpose. The ROCm step below
# force-installs hip-dev with `dpkg -i --force-depends` (and adds a
# broad `Package: * Pin-Priority: 600` preference for repo.radeon.com).
# Either of those leaves apt unable to cleanly resolve the distro
# libclang-18-dev / llvm-18-dev / lld-18 afterward ("E: Unmet
# dependencies"). Resolving them here, against pristine Ubuntu noble
# state, keeps both the LLVM toolchain and ROCm installable.
# libnuma-dev / ninja-build (needed by the NIXL build further below) are
# installed here too, on purpose. They must go in before the ROCm section:
# that step force-installs hip-dev with `dpkg -i --force-depends`, which
# leaves apt with intentionally-unmet dependencies. Any `apt-get install`
# run *after* it aborts with "E: Unmet dependencies" (exit 100) during
# apt's pre-install broken-state check — even for packages already present.
# Resolving everything here, against pristine Ubuntu noble state, avoids that.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libclang-18-dev \
    llvm-18-dev \
    lld-18 \
    libnuma-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------
# ROCm / HIP-NVCC Installation
#------------------------------------------------------------
#
# Installs the AMD ROCm 6.4 toolchain configured for the NVIDIA backend
# (HIP-NVCC). hipcc dispatches to nvcc, the HIP host headers map onto
# cudart, and code compiled with CTP_ENABLE_ROCM=1 runs on the local
# CUDA GPU. The hip-dev metapackage normally pulls in `hipcc` (a thin
# wrapper that defaults to AMD); we install `hipcc-nvidia` instead and
# extract hip-dev with --force-overwrite so the dev headers land
# without the AMD-flavored hipcc dropping a conflicting binary.
#
# Required env at build time: HIP_PLATFORM=nvidia (forwarded into the
# CMake configure step by ClioCoreCommon's wrp_core_enable_rocm
# macro). At runtime nothing extra is needed because the HIP runtime
# is header-only on NVIDIA — calls inline directly into cudart.

USER root

# AMD ROCm apt repository (Ubuntu 24.04 noble, ROCm 6.4)
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4 noble main" > /etc/apt/sources.list.d/rocm.list \
    && printf "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n" > /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update

# hipcc-nvidia + hip-dev (with --force-overwrite to step over the file
# clash with the AMD hipcc package — same binary path, different default
# platform — see file header comment).
RUN apt-get install -y --no-install-recommends hipcc-nvidia hip-runtime-nvidia \
    && apt-get download hip-dev \
    && dpkg -i --force-overwrite --force-depends hip-dev_*.deb \
    && rm -f hip-dev_*.deb \
    && rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm
ENV HIP_PLATFORM=nvidia
ENV PATH=/opt/rocm/bin:${PATH}
# /opt/rocm/lib not added to LD_LIBRARY_PATH on purpose: under HIP-NVCC the
# runtime is header-only against cudart, and adding it can cause amdhip64.so
# to override cudart symbols when both end up resident in the same process.

#------------------------------------------------------------
# AdaptiveCpp (Open SYCL) Installation
#------------------------------------------------------------
# (LLVM 18 dev packages were installed earlier, before the ROCm
# section, so they resolve against pristine Ubuntu apt state.)

# Build and install AdaptiveCpp with CUDA backend
RUN git clone --depth=1 https://github.com/AdaptiveCpp/AdaptiveCpp.git /tmp/adaptivecpp-src \
    && mkdir -p /tmp/adaptivecpp-build \
    && cd /tmp/adaptivecpp-build \
    && cmake /tmp/adaptivecpp-src \
        -DCMAKE_INSTALL_PREFIX=/opt/adaptivecpp \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
        -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-18 \
        -DWITH_CUDA_BACKEND=ON \
        -DWITH_ROCM_BACKEND=OFF \
        -DWITH_LEVEL_ZERO_BACKEND=OFF \
        -DWITH_OPENCL_BACKEND=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/adaptivecpp-src /tmp/adaptivecpp-build

ENV ADAPTIVECPP_HOME=/opt/adaptivecpp
ENV PATH=/opt/adaptivecpp/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/adaptivecpp/lib:${LD_LIBRARY_PATH}

#------------------------------------------------------------
# NIXL (NVIDIA Inference Xfer Library) Installation
#------------------------------------------------------------

USER root

# NIXL build deps (libnuma-dev, ninja-build) were installed earlier, before
# the ROCm section, against pristine apt state — see comment there. apt is
# left in an intentionally-broken-deps state by the hip-dev force install,
# so no apt-get install can run here.

# Install meson and pybind11 in the iowarp venv
RUN /bin/bash -c "source /home/iowarp/venv/bin/activate && \
    pip install meson pybind11"

# Build and install NIXL (POSIX backend only; GDS/UCX optional)
RUN mkdir -p /tmp/nixl_src /tmp/nixl_build && \
    git clone https://github.com/ai-dynamo/nixl.git --depth=1 /tmp/nixl_src && \
    cd /tmp/nixl_build && \
    /bin/bash -c "source /home/iowarp/venv/bin/activate && \
        meson setup /tmp/nixl_src \
            --prefix=/usr/local \
            -Ddisable_gds_backend=true \
            -Dbuild_tests=false \
            -Dbuild_examples=false \
            -Denable_plugins=POSIX \
            -Drust=false && \
        ninja -j$(nproc) && \
        ninja install" && \
    rm -rf /tmp/nixl_src /tmp/nixl_build

#------------------------------------------------------------
# User Configuration
#------------------------------------------------------------

# Switch back to iowarp user
USER iowarp
WORKDIR /home/iowarp

# Add CUDA and SYCL paths to bashrc
RUN echo '' >> /home/iowarp/.bashrc \
    && echo '# CUDA environment variables for GPU execution' >> /home/iowarp/.bashrc \
    && echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> /home/iowarp/.bashrc \
    && echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> /home/iowarp/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/lib64/stubs:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> /home/iowarp/.bashrc \
    && echo 'export NVIDIA_VISIBLE_DEVICES=all' >> /home/iowarp/.bashrc \
    && echo 'export NVIDIA_DRIVER_CAPABILITIES=compute,utility' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# Intel DPC++ SYCL environment (production)' >> /home/iowarp/.bashrc \
    && echo 'export DPCPP_HOME=/opt/intel/dpcpp' >> /home/iowarp/.bashrc \
    && echo 'export PATH=/opt/intel/dpcpp/bin:$PATH' >> /home/iowarp/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/opt/intel/dpcpp/lib:$LD_LIBRARY_PATH' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# AdaptiveCpp SYCL environment' >> /home/iowarp/.bashrc \
    && echo 'export ADAPTIVECPP_HOME=/opt/adaptivecpp' >> /home/iowarp/.bashrc \
    && echo 'export PATH=/opt/adaptivecpp/bin:$PATH' >> /home/iowarp/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/opt/adaptivecpp/lib:$LD_LIBRARY_PATH' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# ROCm/HIP environment (HIP-NVCC backend on NVIDIA hardware)' >> /home/iowarp/.bashrc \
    && echo 'export ROCM_PATH=/opt/rocm' >> /home/iowarp/.bashrc \
    && echo 'export HIP_PLATFORM=nvidia' >> /home/iowarp/.bashrc \
    && echo 'export PATH=/opt/rocm/bin:$PATH' >> /home/iowarp/.bashrc

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
