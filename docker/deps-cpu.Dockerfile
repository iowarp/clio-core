# IOWarp CPU Dependencies Container
# Base container with all CPU-only dependencies for building IOWarp
#
# All dependencies are installed via apt or built from source.
# Core IOWarp deps (yaml-cpp, zeromq, libsodium, cereal, libaio) are built
# from source with both shared and static libraries (-fPIC) so that
# WRP_CORE_STATIC_DEPS=ON works for self-contained pip wheels.
#
# Source-built libraries install to /usr/local (shared+static with -fPIC).
# Apt libraries install to /usr (shared+static, static without -fPIC).
#
# Usage:
#   docker build -t iowarp/deps-cpu:latest -f docker/deps-cpu.Dockerfile .
#
FROM iowarp/iowarp-base:latest
LABEL maintainer="llogan@hawk.iit.edu"
LABEL version="2.0"
LABEL description="IOWarp CPU dependencies Docker image (apt + source builds)"

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
# System Dependencies (apt)
#------------------------------------------------------------

USER root

# Build tools
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    pkg-config \
    g++ \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-pytest \
    && rm -rf /var/lib/apt/lists/*

# System libraries and services
RUN apt-get update && apt-get install -y \
    libelf-dev \
    libaio-dev \
    liburing-dev \
    redis-server \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# MPI
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    mpi-default-dev \
    && rm -rf /var/lib/apt/lists/*

# Core libraries (apt provides shared + static, static without -fPIC)
# These are used for normal shared-library builds. For static linking into
# shared objects (pip wheels), the source-built versions in /usr/local take
# precedence via CMAKE_PREFIX_PATH ordering.
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libhdf5-dev \
    catch2 \
    libcurl4-openssl-dev \
    libssl-dev \
    nlohmann-json3-dev \
    libpoco-dev \
    && rm -rf /var/lib/apt/lists/*

# Compression libraries
RUN apt-get update && apt-get install -y \
    zlib1g-dev \
    libbz2-dev \
    liblzo2-dev \
    libzstd-dev \
    liblz4-dev \
    liblzma-dev \
    libbrotli-dev \
    libsnappy-dev \
    libblosc2-dev \
    libzfp-dev \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI and Docker-in-Docker dependencies
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
# Build Core IOWarp Dependencies from Source
#------------------------------------------------------------
# These are built with BOTH shared and static libraries, with -fPIC
# on static archives so they can be linked into IOWarp's shared objects
# when WRP_CORE_STATIC_DEPS=ON (for pip wheels).
#
# Install prefix: /usr/local (takes precedence over /usr in default search)

# yaml-cpp 0.8.0 (shared + static with -fPIC)
RUN cd /tmp \
    && curl -sL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz | tar xz \
    && cmake -S yaml-cpp-0.8.0 -B yaml-cpp-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
       -DBUILD_SHARED_LIBS=ON \
       -DYAML_CPP_BUILD_TESTS=OFF \
       -DYAML_CPP_BUILD_TOOLS=OFF \
    && cmake --build yaml-cpp-build -j$(nproc) \
    && cmake --install yaml-cpp-build \
    && cmake -S yaml-cpp-0.8.0 -B yaml-cpp-build-static \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
       -DBUILD_SHARED_LIBS=OFF \
       -DYAML_CPP_BUILD_TESTS=OFF \
       -DYAML_CPP_BUILD_TOOLS=OFF \
    && cmake --build yaml-cpp-build-static -j$(nproc) \
    && cmake --install yaml-cpp-build-static \
    && ldconfig \
    && rm -rf /tmp/yaml-cpp-*

# cereal 1.3.2 (header-only — just installs headers and cmake config)
RUN cd /tmp \
    && curl -sL https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz | tar xz \
    && cmake -S cereal-1.3.2 -B cereal-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
       -DSKIP_PERFORMANCE_COMPARISON=ON \
       -DBUILD_TESTS=OFF \
       -DBUILD_SANDBOX=OFF \
       -DBUILD_DOC=OFF \
    && cmake --build cereal-build -j$(nproc) \
    && cmake --install cereal-build \
    && rm -rf /tmp/cereal-*

# msgpack-c 6.1.0 (pure C library — no Boost dependency, static with -fPIC)
RUN cd /tmp \
    && git clone --depth 1 --branch c-6.1.0 https://github.com/msgpack/msgpack-c.git \
    && cmake -S msgpack-c -B msgpack-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
       -DMSGPACK_BUILD_TESTS=OFF \
       -DMSGPACK_BUILD_EXAMPLES=OFF \
    && cmake --build msgpack-build -j$(nproc) \
    && cmake --install msgpack-build \
    && rm -rf /tmp/msgpack-c /tmp/msgpack-build

# libsodium 1.0.20 (shared + static with -fPIC, required by zeromq)
RUN cd /tmp \
    && curl -sL https://github.com/jedisct1/libsodium/releases/download/1.0.20-RELEASE/libsodium-1.0.20.tar.gz | tar xz \
    && cd libsodium-1.0.20 \
    && ./configure --prefix=/usr/local --with-pic \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/libsodium-*

# zeromq 4.3.5 (shared + static with -fPIC)
RUN cd /tmp \
    && curl -sL https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz | tar xz \
    && cmake -S zeromq-4.3.5 -B zmq-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
       -DBUILD_SHARED=ON \
       -DBUILD_STATIC=ON \
       -DBUILD_TESTS=OFF \
       -DWITH_LIBSODIUM=ON \
       -DWITH_DOCS=OFF \
       -DCMAKE_PREFIX_PATH=/usr/local \
    && cmake --build zmq-build -j$(nproc) \
    && cmake --install zmq-build \
    && ldconfig \
    && rm -rf /tmp/zeromq-* /tmp/zmq-build

# cppzmq 4.10.0 (header-only — C++ bindings for ZeroMQ, must come after zeromq)
RUN cd /tmp \
    && curl -sL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz | tar xz \
    && cmake -S cppzmq-4.10.0 -B cppzmq-build \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCMAKE_PREFIX_PATH=/usr/local \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
       -DCPPZMQ_BUILD_TESTS=OFF \
    && cmake --install cppzmq-build \
    && rm -rf /tmp/cppzmq-*

# libaio 0.3.113 (shared + static with -fPIC)
# Build twice: first with symver intact for a working shared library,
# then with symver stripped for a static archive that can be linked into
# shared objects without "version node not found" errors.
RUN cd /tmp \
    && curl -sL https://pagure.io/libaio/archive/libaio-0.3.113/libaio-libaio-0.3.113.tar.gz | tar xz \
    && cd libaio-libaio-0.3.113 \
    && make prefix=/usr/local CFLAGS="-fPIC -O2" \
    && make prefix=/usr/local install \
    && ldconfig \
    && make clean \
    && sed -i 's/__asm__(".symver.*;//g' src/syscall.h \
    && make prefix=/usr/local CFLAGS="-fPIC -O2" \
    && cp src/libaio.a /usr/local/lib/libaio.a \
    && rm -rf /tmp/libaio-*

#------------------------------------------------------------
# Build Lossy Compression Libraries from Source
#------------------------------------------------------------

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
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /tmp && rm -rf ADIOS2 adios2-build

#------------------------------------------------------------
# Final Setup
#------------------------------------------------------------

# Switch back to iowarp user
USER iowarp

# Install Rust toolchain (needed for CTE Rust wrapper / cdylib builds)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/iowarp/.cargo/bin:${PATH}"

# Install Bun JavaScript runtime (needed for CTE memorybench TypeScript benchmarks)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/home/iowarp/.bun/bin:${PATH}"

# Install Node.js 22 LTS (required by Docusaurus docs site, needs >= 20)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    node --version && \
    chown -R iowarp:iowarp /home/iowarp/.npm || true
USER iowarp

# Install docs site (Docusaurus) dependencies
# Try SSH clone first, fall back to HTTPS if SSH is unavailable
RUN cd /home/iowarp \
    && (git clone -b iowarp-dev git@github.com:iowarp/docs.git 2>/dev/null || \
        git clone -b iowarp-dev https://github.com/iowarp/docs.git)

# Create Python virtual environment and install build tools
RUN python3 -m venv /home/iowarp/venv && \
    /home/iowarp/venv/bin/pip install --upgrade pip && \
    /home/iowarp/venv/bin/pip install scikit-build-core nanobind
ENV VIRTUAL_ENV="/home/iowarp/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install Jarvis-CD (deployment and pipeline management)
RUN cd /home/iowarp \
    && git clone https://github.com/grc-iit/jarvis-cd.git jarvis-cd \
    && cd jarvis-cd \
    && pip install -r requirements.txt \
    && pip install -e .

# Configure Spack to use system packages
RUN mkdir -p ~/.spack && \
    echo "packages:" > ~/.spack/packages.yaml && \
    echo "  cmake:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: cmake" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml && \
    echo "  boost:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: boost" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml && \
    echo "  openmpi:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: openmpi" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml && \
    echo "  hdf5:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: hdf5" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml && \
    echo "  python:" >> ~/.spack/packages.yaml && \
    echo "    externals:" >> ~/.spack/packages.yaml && \
    echo "    - spec: python" >> ~/.spack/packages.yaml && \
    echo "      prefix: /usr" >> ~/.spack/packages.yaml && \
    echo "    buildable: false" >> ~/.spack/packages.yaml

# Set up environment
# Use architecture-aware library path (x86_64 or aarch64)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
    LIB_ARCH="aarch64-linux-gnu"; \
    else \
    LIB_ARCH="x86_64-linux-gnu"; \
    fi && \
    echo '' >> /home/iowarp/.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/${LIB_ARCH}:\$LD_LIBRARY_PATH" >> /home/iowarp/.bashrc \
    && echo "export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/lib/${LIB_ARCH}/pkgconfig:\$PKG_CONFIG_PATH" >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# Source-built libraries (yaml-cpp, zmq, sodium, cereal, libaio) in /usr/local' >> /home/iowarp/.bashrc \
    && echo '# Apt libraries (boost, hdf5, compression, etc.) in /usr' >> /home/iowarp/.bashrc \
    && echo '# Lossy compression (FPZIP, SZ3, LibPressio) + ADIOS2 in /usr/local' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# Rust toolchain' >> /home/iowarp/.bashrc \
    && echo 'export PATH="/home/iowarp/.cargo/bin:$PATH"' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# Bun JavaScript runtime' >> /home/iowarp/.bashrc \
    && echo 'export PATH="/home/iowarp/.bun/bin:$PATH"' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# Python virtual environment' >> /home/iowarp/.bashrc \
    && echo 'source /home/iowarp/venv/bin/activate' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
