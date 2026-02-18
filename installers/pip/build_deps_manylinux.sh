#!/bin/bash
# build_deps_manylinux.sh
# Builds IOWarp's external dependencies from source inside a manylinux_2_28
# container (AlmaLinux 8). All libraries are built with both shared and static
# archives, with -fPIC so the static archives can be linked into shared objects.
#
# Install prefix: /usr/local
#
# This script is called by cibuildwheel's CIBW_BEFORE_ALL.
set -euo pipefail

PREFIX=/usr/local
NPROC=$(nproc)

echo "=== Building IOWarp dependencies (prefix=$PREFIX, nproc=$NPROC) ==="

# yaml-cpp 0.8.0
echo "--- yaml-cpp 0.8.0 ---"
cd /tmp
curl -sL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz | tar xz
cmake -S yaml-cpp-0.8.0 -B yaml-cpp-shared \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF
cmake --build yaml-cpp-shared -j$NPROC
cmake --install yaml-cpp-shared
cmake -S yaml-cpp-0.8.0 -B yaml-cpp-static \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF
cmake --build yaml-cpp-static -j$NPROC
cmake --install yaml-cpp-static
rm -rf /tmp/yaml-cpp-*

# cereal 1.3.2 (header-only)
echo "--- cereal 1.3.2 ---"
cd /tmp
curl -sL https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz | tar xz
cmake -S cereal-1.3.2 -B cereal-build \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DSKIP_PERFORMANCE_COMPARISON=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_SANDBOX=OFF \
    -DBUILD_DOC=OFF
cmake --build cereal-build -j$NPROC
cmake --install cereal-build
rm -rf /tmp/cereal-*

# libsodium 1.0.20
echo "--- libsodium 1.0.20 ---"
cd /tmp
curl -sL https://github.com/jedisct1/libsodium/releases/download/1.0.20-RELEASE/libsodium-1.0.20.tar.gz | tar xz
cd libsodium-1.0.20
./configure --prefix=$PREFIX --with-pic
make -j$NPROC
make install
cd /tmp && rm -rf libsodium-*

# zeromq 4.3.5
echo "--- zeromq 4.3.5 ---"
cd /tmp
curl -sL https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz | tar xz
cmake -S zeromq-4.3.5 -B zmq-build \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED=ON \
    -DBUILD_STATIC=ON \
    -DBUILD_TESTS=OFF \
    -DWITH_LIBSODIUM=ON \
    -DWITH_DOCS=OFF \
    -DCMAKE_PREFIX_PATH=$PREFIX
cmake --build zmq-build -j$NPROC
cmake --install zmq-build
rm -rf /tmp/zeromq-* /tmp/zmq-build

# cppzmq 4.10.0 (header-only)
echo "--- cppzmq 4.10.0 ---"
cd /tmp
curl -sL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz | tar xz
cmake -S cppzmq-4.10.0 -B cppzmq-build \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCPPZMQ_BUILD_TESTS=OFF
cmake --install cppzmq-build
rm -rf /tmp/cppzmq-*

# libaio 0.3.113 (strip symver for clean static linking)
echo "--- libaio 0.3.113 ---"
cd /tmp
curl -sL https://pagure.io/libaio/archive/libaio-0.3.113/libaio-libaio-0.3.113.tar.gz | tar xz
cd libaio-libaio-0.3.113
sed -i 's/__asm__(".symver.*;//g' src/syscall.h
make prefix=$PREFIX CFLAGS="-fPIC -O2"
make prefix=$PREFIX install
cd /tmp && rm -rf libaio-*

ldconfig
echo "=== All dependencies built successfully ==="
