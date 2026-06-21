#!/bin/bash
# IOWarp runtime build script — runs inside the jarvis pipeline build
# container (started from container_base, typically ubuntu:24.04).
#
# HDF5, ADIOS2, and the lossy compression stack (fpzip / SZ3 /
# std_compat / libpressio) are NOT built here anymore. Those live in
# reusable jarvis Library packages — builtin.hdf5, builtin.adios2,
# builtin.compress_libs — that must appear before clio_runtime in the
# pipeline. When jarvis runs Phase 1 it checks for each Library's
# cached deploy image (jarvis-deploy-hdf5-2.1.1 etc.) and, if present,
# docker-cp's its /usr/local and /opt into the shared build container
# (see jarvis_cd/core/pipeline.py _build_pipeline_container, commit
# 285fcb4). By the time this script runs those libs are already in
# /usr/local, so CMake's find_package() just picks them up.
set -e

export DEBIAN_FRONTEND=noninteractive

# --- System build deps ------------------------------------------------------
# Only the deps unique to the IOWarp build: the Library packages cover
# compiler toolchain + their own runtime deps before us.
apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    cmake ninja-build pkg-config g++ make \
    python3-dev python3-pip python3-venv \
    libelf-dev libaio-dev liburing-dev \
    libfuse3-dev fuse3 \
    openmpi-bin libopenmpi-dev mpi-default-dev \
    openssh-server openssh-client \
    libboost-all-dev catch2 libcurl4-openssl-dev libssl-dev \
    nlohmann-json3-dev \
    zlib1g-dev libbz2-dev liblzo2-dev libzstd-dev liblz4-dev liblzma-dev \
    libbrotli-dev libsnappy-dev libblosc2-dev libzfp-dev \
 && rm -rf /var/lib/apt/lists/*

# SSH host keys + passwordless root key, required by jarvis_cd's apptainer
# multi-node start path (mpirun ssh's between instances on container_ssh_port).
mkdir -p /var/run/sshd /root/.ssh \
    && ssh-keygen -A \
    && ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 \
    && cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys \
    && sed -i 's/^#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && printf "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null\n" >> /etc/ssh/ssh_config

# --- yaml-cpp 0.8.0 ---------------------------------------------------------
cd /tmp
rm -rf yaml-cpp-0.8.0 yaml-cpp-build
curl -sL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz | tar xz
cmake -S yaml-cpp-0.8.0 -B yaml-cpp-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON \
   -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF
cmake --build yaml-cpp-build -j"$(nproc)"
cmake --install yaml-cpp-build
ldconfig
rm -rf /tmp/yaml-cpp-*

# --- cereal 1.3.2 (header-only) ---------------------------------------------
cd /tmp
rm -rf cereal-1.3.2 cereal-build
curl -sL https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz | tar xz
cmake -S cereal-1.3.2 -B cereal-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DSKIP_PERFORMANCE_COMPARISON=ON \
   -DBUILD_TESTS=OFF -DBUILD_SANDBOX=OFF -DBUILD_DOC=OFF
cmake --install cereal-build
rm -rf /tmp/cereal-*

# --- msgpack-c 6.1.0 --------------------------------------------------------
cd /tmp
rm -rf msgpack-c msgpack-build
git clone --depth 1 --branch c-6.1.0 https://github.com/msgpack/msgpack-c.git
cmake -S msgpack-c -B msgpack-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
   -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF
cmake --build msgpack-build -j"$(nproc)"
cmake --install msgpack-build
rm -rf /tmp/msgpack-c /tmp/msgpack-build

# --- libsodium 1.0.20 -------------------------------------------------------
cd /tmp
rm -rf libsodium-1.0.20
curl -sL https://github.com/jedisct1/libsodium/releases/download/1.0.20-RELEASE/libsodium-1.0.20.tar.gz | tar xz
cd libsodium-1.0.20
./configure --prefix=/usr/local --with-pic
make -j"$(nproc)"
make install
ldconfig
rm -rf /tmp/libsodium-*

# --- zeromq 4.3.5 -----------------------------------------------------------
cd /tmp
rm -rf zeromq-4.3.5 zmq-build
curl -sL https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz | tar xz
cmake -S zeromq-4.3.5 -B zmq-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED=ON -DBUILD_STATIC=ON \
   -DBUILD_TESTS=OFF -DWITH_LIBSODIUM=ON -DWITH_DOCS=OFF \
   -DCMAKE_PREFIX_PATH=/usr/local
cmake --build zmq-build -j"$(nproc)"
cmake --install zmq-build
ldconfig
rm -rf /tmp/zeromq-* /tmp/zmq-build

# --- cppzmq 4.10.0 (header-only) --------------------------------------------
cd /tmp
rm -rf cppzmq-4.10.0 cppzmq-build
curl -sL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz | tar xz
cmake -S cppzmq-4.10.0 -B cppzmq-build \
   -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=/usr/local \
   -DCPPZMQ_BUILD_TESTS=OFF
cmake --install cppzmq-build
rm -rf /tmp/cppzmq-*

# --- Clone and build IOWarp -------------------------------------------------
# HDF5, ADIOS2, and the compression libs were already injected by the
# pipeline before this script started (see header comment).
# Submodules are NOT recursed at clone time: external/jarvis-cd pulls its
# own `awesome-scienctific-applications` submodule via an SSH URL that
# fails in containers, and the core build does not need it.
rm -rf /opt/iowarp
git clone --depth 1 --branch ##GIT_BRANCH## \
    https://github.com/iowarp/clio-core.git /opt/iowarp

cd /opt/iowarp
# CTP_LOG_LEVEL=0 (kDebug) compiles in all kDebug HLOG markers — needed
# to trace the SWIM probe RPC path. Default release preset uses 1 (kInfo)
# which strips them entirely.
cmake --preset ##CMAKE_PRESET## -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DWRP_CORE_ENABLE_BENCHMARKS=ON \
    -DHSHM_LOG_LEVEL=0
cmake --build build -j"$(nproc)"
cmake --install build
ldconfig

# --- Seed default clio config ----------------------------------------------
# ~/.clio/clio.yaml is the canonical user-config location the runtime
# checks first; Dockerfile.deploy COPYs this dir from the builder.
mkdir -p /root/.clio
cp /opt/iowarp/context-runtime/config/clio_default.yaml \
   /root/.clio/clio.yaml
