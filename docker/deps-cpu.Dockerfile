# IOWarp CPU Dependencies Container
# Base container with all CPU-only dependencies for building IOWarp
#
# Usage:
#   docker build -t iowarp/deps-cpu:latest -f docker/deps-cpu.Dockerfile .
#
FROM iowarp/iowarp-base:latest
LABEL maintainer="llogan@hawk.iit.edu"
LABEL version="1.0"
LABEL description="IOWarp CPU dependencies Docker image"

# Disable prompt during packages installation.
ARG DEBIAN_FRONTEND=noninteractive

# Update iowarp-install repo
RUN cd ${HOME}/iowarp-install && \
    git fetch origin && \
    git pull origin main

# Update grc-repo repo
RUN cd ${HOME}/grc-repo && \
    git pull origin main

#------------------------------------------------------------
# System Dependencies (not available via conda)
#------------------------------------------------------------

USER root

# Install system packages not provided by conda
RUN apt-get update && apt-get install -y \
    libelf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MPI (openmpi) - not available via conda in our setup
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    mpi-default-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI and Docker-in-Docker dependencies
# Also install network diagnostic tools (netstat, lsof, ss, etc.)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    iptables \
    supervisor \
    net-tools \
    lsof \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Add Docker's official GPG key and repository
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
RUN apt-get update && apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Add iowarp user to docker group
RUN usermod -aG docker iowarp

# Create docker group if it doesn't exist (it should from docker install)
RUN getent group docker || groupadd docker

# Set up Docker socket permissions script
RUN echo '#!/bin/bash\n\
    if [ -S /var/run/docker.sock ]; then\n\
    sudo chmod 666 /var/run/docker.sock\n\
    fi\n\
    exec "$@"' > /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Allow iowarp user to manage docker socket permissions without password
RUN echo "iowarp ALL=(ALL) NOPASSWD: /bin/chmod 666 /var/run/docker.sock" >> /etc/sudoers.d/docker-socket \
    && chmod 0440 /etc/sudoers.d/docker-socket

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

#------------------------------------------------------------
# Conda Dependencies
#------------------------------------------------------------

# Switch to iowarp user for conda setup
USER iowarp
WORKDIR /home/iowarp

# Install Miniconda (skip if already installed in base image)
# Detect architecture and download appropriate installer (x86_64 or aarch64)
RUN if [ ! -d "/home/iowarp/miniconda3" ]; then \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    fi && \
    wget "$MINICONDA_URL" -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /home/iowarp/miniconda3 \
    && rm /tmp/miniconda.sh; \
    fi

# Initialize conda for bash
RUN /home/iowarp/miniconda3/bin/conda init bash \
    && /home/iowarp/miniconda3/bin/conda config --add channels conda-forge \
    && /home/iowarp/miniconda3/bin/conda config --set channel_priority strict

# Accept Anaconda Terms of Service and install all development dependencies via conda
# This avoids library conflicts between system packages and conda packages
# Dependencies installed:
#   - Build tools: cmake, ninja, conda-build, pkg-config
#   - Core libraries: boost, hdf5, yaml-cpp, zeromq, cppzmq, cereal
#   - Testing: catch2, pytest
#   - Network: libcurl, openssl
#   - Compression libraries: zlib, bzip2, lzo, zstd, lz4, xz (lzma), brotli, snappy, c-blosc2
#   - Lossy compression: zfp (scientific compressor, available via conda)
#   - Optional: poco (for Globus support), nlohmann_json
RUN /home/iowarp/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /home/iowarp/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /home/iowarp/miniconda3/bin/conda install -y \
    conda-build \
    cmake \
    ninja \
    pkg-config \
    boost \
    hdf5 \
    yaml-cpp \
    zeromq \
    cppzmq \
    cereal \
    catch2 \
    libcurl \
    openssl \
    zlib \
    bzip2 \
    lzo \
    zstd \
    lz4-c \
    xz \
    brotli \
    snappy \
    c-blosc2 \
    zfp \
    poco \
    nlohmann_json \
    pytest \
    && /home/iowarp/miniconda3/bin/conda clean -ya

# Set conda environment variables for CMake to find packages
ENV CONDA_PREFIX=/home/iowarp/miniconda3
ENV PKG_CONFIG_PATH=/home/iowarp/miniconda3/lib/pkgconfig
ENV CMAKE_PREFIX_PATH=/home/iowarp/miniconda3

#------------------------------------------------------------
# Build Lossy Compression Libraries from Source
#------------------------------------------------------------

# Switch to root to install lossy compression libraries from source
# Add conda's cmake to PATH for root user
USER root
ENV PATH="/home/iowarp/miniconda3/bin:${PATH}"

# Install FPZIP (fast floating-point compressor)
RUN cd /tmp \
    && git clone https://github.com/LLNL/fpzip.git fpzip_src \
    && cd fpzip_src \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_UTILITIES=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /tmp && rm -rf fpzip_src

# Install SZ3 (fast error-bounded lossy compressor for scientific data)
RUN cd /tmp \
    && git clone https://github.com/szcompressor/SZ3.git sz3_src \
    && cd sz3_src \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /tmp && rm -rf sz3_src

# Install std_compat (required dependency for LibPressio)
RUN cd /tmp \
    && git clone https://github.com/robertu94/std_compat.git std_compat_src \
    && cd std_compat_src \
    && mkdir -p ~/builds/std_compat && cd ~/builds/std_compat \
    && cmake /tmp/std_compat_src \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_TESTING=OFF \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/std_compat_src ~/builds/std_compat

# Install LibPressio (meta-compressor library for lossy compression)
# Provides unified interface to ZFP, SZ3, FPZIP and other lossy compressors
RUN cd /tmp \
    && git clone https://github.com/robertu94/libpressio.git libpressio_src \
    && cd libpressio_src \
    && mkdir -p ~/builds/libpressio && cd ~/builds/libpressio \
    && export CMAKE_PREFIX_PATH="/usr/local:/home/iowarp/miniconda3:${CMAKE_PREFIX_PATH}" \
    && cmake /tmp/libpressio_src \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DLIBPRESSIO_HAS_ZFP=ON \
    -DLIBPRESSIO_HAS_SZ3=ON \
    -DLIBPRESSIO_HAS_FPZIP=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/libpressio_src ~/builds/libpressio

#------------------------------------------------------------
# Build ADIOS2 from Source
#------------------------------------------------------------

# Install ADIOS2 from source with HDF5 and ZeroMQ support
# Uses conda's HDF5 and ZeroMQ libraries (CMAKE_PREFIX_PATH points to conda)
# NOTE: Updated to v2.11.0 for C++20 compatibility and ARM64 support
# NOTE: SST is disabled because the DILL library has ARM64 Linux compatibility issues
#       (sys_icache_invalidate is an Apple-specific function not available on ARM64 Linux)
RUN cd /tmp \
    && git clone --depth 1 --branch v2.11.0 https://github.com/ornladios/ADIOS2.git ADIOS2 \
    && mkdir -p adios2-build && cd adios2-build \
    && cmake ../ADIOS2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DADIOS2_BUILD_EXAMPLES=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_HDF5=ON \
    -DADIOS2_USE_ZeroMQ=ON \
    -DADIOS2_USE_Python=OFF \
    -DADIOS2_USE_SST=OFF \
    -DADIOS2_USE_Fortran=OFF \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_PREFIX_PATH="/home/iowarp/miniconda3" \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /tmp && rm -rf ADIOS2 adios2-build

#------------------------------------------------------------
# Final Setup
#------------------------------------------------------------

# Switch back to iowarp user
USER iowarp

# Install libaio for Linux AIO support (required by bdev ChiMod)
SHELL ["/bin/bash", "-c"]
RUN source /home/iowarp/miniconda3/etc/profile.d/conda.sh \
    && conda activate base \
    && conda install -y libaio -c conda-forge

# Install Jarvis (IOWarp runtime deployment tool)
RUN cd /home/iowarp \
    && git clone https://github.com/iowarp/runtime-deployment.git \
    && cd runtime-deployment \
    && source /home/iowarp/miniconda3/etc/profile.d/conda.sh \
    && conda activate base \
    && pip install -e . -r requirements.txt  \
    && jarvis init \
    && jarvis rg build

# Configure Spack to use conda packages
RUN mkdir -p ~/.spack && \
    echo "packages:" > ~/.spack/packages.yaml && \
    echo "  cmake:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: cmake" >> ~/.spack/packages.yaml && \
    echo "      prefix: /home/iowarp/miniconda3" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml && \
    echo "  boost:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: boost" >> ~/.spack/packages.yaml && \
    echo "      prefix: /home/iowarp/miniconda3" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml && \
    echo "  openmpi:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: openmpi" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml && \
    echo "  hdf5:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: hdf5" >> ~/.spack/packages.yaml && \
    echo "      prefix: /home/iowarp/miniconda3" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml && \
    echo "  python:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: python" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml

# Add conda activation and LD_LIBRARY_PATH to bashrc
# Use architecture-aware library path (x86_64 or aarch64)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        LIB_ARCH="aarch64-linux-gnu"; \
    else \
        LIB_ARCH="x86_64-linux-gnu"; \
    fi && \
    echo '' >> /home/iowarp/.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/lib:/home/iowarp/miniconda3/lib:/usr/lib/${LIB_ARCH}:\$LD_LIBRARY_PATH" >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# >>> conda initialize >>>' >> /home/iowarp/.bashrc \
    && echo '# Conda base environment is auto-activated with all dev dependencies' >> /home/iowarp/.bashrc \
    && echo '# This includes: boost, hdf5, yaml-cpp, zeromq, cereal, catch2, pytest, etc.' >> /home/iowarp/.bashrc \
    && echo '# Compression libraries: zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, c-blosc2 (via conda)' >> /home/iowarp/.bashrc \
    && echo '# Lossy compression: ZFP (via conda), SZ3, FPZIP, LibPressio (built from source)' >> /home/iowarp/.bashrc \
    && echo '# ADIOS2: Built from source (installed in /usr/local)' >> /home/iowarp/.bashrc \
    && echo '# Create custom environments if needed: conda create -n myenv' >> /home/iowarp/.bashrc \
    && echo 'eval "$(/home/iowarp/miniconda3/bin/conda shell.bash hook)"' >> /home/iowarp/.bashrc \
    && echo '# <<< conda initialize <<<' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
