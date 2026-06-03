#!/bin/bash
# build_deps_manylinux.sh
# Builds IOWarp's external dependencies from source inside a manylinux_2_34
# container (AlmaLinux 9). All libraries are built with both shared and static
# archives, with -fPIC so the static archives can be linked into shared objects.
#
# Install prefix: /usr/local
#
# This script is called by cibuildwheel's CIBW_BEFORE_ALL.
set -euo pipefail

PREFIX=/usr/local
NPROC=$(nproc)

# AlmaLinux 9 / manylinux_2_34: the default compiler may generate x86_64_v2
# instructions (SSE4.2, POPCNT) because RHEL 9 requires x86_64_v2 hardware.
# Force baseline x86_64 so the wheel runs on any x86_64 CPU.
# On aarch64 we leave the flags alone — the compiler defaults are fine.
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    export CFLAGS="-march=x86-64"
    export CXXFLAGS="-march=x86-64"
fi

echo "=== Building IOWarp dependencies (prefix=$PREFIX, nproc=$NPROC, arch=$ARCH) ==="
echo "=== CFLAGS=${CFLAGS:-} CXXFLAGS=${CXXFLAGS:-} ==="

# Wrapper: download with retries, fail loudly if the response is not a valid
# gzip archive (GitHub occasionally returns an HTML error page on transient
# rate-limit or CDN hiccups).
download_tar() {
    local url="$1"
    local dest="$2"
    local max_attempts=5
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        echo "  Downloading $url (attempt $attempt/$max_attempts)..."
        if curl -fsSL --retry 3 --retry-delay 5 --retry-all-errors "$url" -o "$dest"; then
            if file "$dest" | grep -qE "gzip|XZ|bzip2|tar archive"; then
                return 0
            fi
            echo "  Warning: downloaded file does not look like a tar archive, retrying..."
            rm -f "$dest"
        fi
        sleep $((attempt * 5))
    done
    echo "ERROR: failed to download $url after $max_attempts attempts" >&2
    return 1
}

# yaml-cpp 0.8.0
echo "--- yaml-cpp 0.8.0 ---"
cd /tmp
download_tar https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz yaml-cpp.tar.gz
tar xzf yaml-cpp.tar.gz && rm yaml-cpp.tar.gz
cmake -S yaml-cpp-0.8.0 -B yaml-cpp-shared \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBUILD_SHARED_LIBS=ON \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF
cmake --build yaml-cpp-shared -j$NPROC
cmake --install yaml-cpp-shared
cmake -S yaml-cpp-0.8.0 -B yaml-cpp-static \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBUILD_SHARED_LIBS=OFF \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF
cmake --build yaml-cpp-static -j$NPROC
cmake --install yaml-cpp-static
rm -rf /tmp/yaml-cpp-*

# cereal 1.3.2 (header-only)
echo "--- cereal 1.3.2 ---"
cd /tmp
download_tar https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz cereal.tar.gz
tar xzf cereal.tar.gz && rm cereal.tar.gz
cmake -S cereal-1.3.2 -B cereal-build \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DSKIP_PERFORMANCE_COMPARISON=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_SANDBOX=OFF \
    -DBUILD_DOC=OFF
cmake --build cereal-build -j$NPROC
cmake --install cereal-build
rm -rf /tmp/cereal-*

# msgpack-c 6.1.0 (pure C library — no Boost dependency)
echo "--- msgpack-c 6.1.0 ---"
cd /tmp
download_tar https://github.com/msgpack/msgpack-c/archive/refs/tags/c-6.1.0.tar.gz msgpack.tar.gz
tar xzf msgpack.tar.gz && rm msgpack.tar.gz
cmake -S msgpack-c-c-6.1.0 -B msgpack-build \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DMSGPACK_BUILD_TESTS=OFF \
    -DMSGPACK_BUILD_EXAMPLES=OFF
cmake --build msgpack-build -j$NPROC
cmake --install msgpack-build
rm -rf /tmp/msgpack-c-* /tmp/msgpack-build

# libsodium 1.0.20
echo "--- libsodium 1.0.20 ---"
cd /tmp
download_tar https://github.com/jedisct1/libsodium/releases/download/1.0.20-RELEASE/libsodium-1.0.20.tar.gz libsodium.tar.gz
tar xzf libsodium.tar.gz && rm libsodium.tar.gz
cd libsodium-1.0.20
./configure --prefix=$PREFIX --with-pic
make -j$NPROC
make install
cd /tmp && rm -rf libsodium-*

# zeromq 4.3.5
echo "--- zeromq 4.3.5 ---"
cd /tmp
download_tar https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz zeromq.tar.gz
tar xzf zeromq.tar.gz && rm zeromq.tar.gz
cmake -S zeromq-4.3.5 -B zmq-build \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
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
download_tar https://github.com/zeromq/cppzmq/archive/refs/tags/v4.10.0.tar.gz cppzmq.tar.gz
tar xzf cppzmq.tar.gz && rm cppzmq.tar.gz
cmake -S cppzmq-4.10.0 -B cppzmq-build \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_PREFIX_PATH=$PREFIX \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCPPZMQ_BUILD_TESTS=OFF
cmake --install cppzmq-build
rm -rf /tmp/cppzmq-*

# liburing 2.5 (static + shared, for io_uring async I/O backend)
echo "--- liburing 2.5 ---"
cd /tmp
download_tar https://github.com/axboe/liburing/archive/refs/tags/liburing-2.5.tar.gz liburing.tar.gz
tar xzf liburing.tar.gz && rm liburing.tar.gz
cd liburing-liburing-2.5
./configure --prefix=$PREFIX
make -j$NPROC CFLAGS="${CFLAGS:-} -fPIC"
make install
cd /tmp && rm -rf liburing-*

ldconfig
echo "=== All dependencies built successfully ==="
