#!/bin/bash

set -e

# Initialize submodules if not already initialized
# This is necessary when building in conda-build environment
if [ ! -f "context-transport-primitives/CMakeLists.txt" ]; then
    echo "Submodules not initialized. Initializing submodules..."
    git submodule update --init --recursive
    if [ $? -ne 0 ]; then
        echo "Error: Failed to initialize submodules."
        exit 1
    fi
    echo "Submodules initialized successfully."
else
    echo "Submodules already initialized."
fi

# Collect environment variables with specific prefixes to forward to cmake
CMAKE_EXTRA_ARGS=()
for var in $(compgen -e); do
    if [[ "$var" =~ ^(WRP_CORE_ENABLE_|WRP_CTE_ENABLE_|WRP_CAE_ENABLE_|WRP_CEE_ENABLE_|HSHM_ENABLE_|WRP_CTP_ENABLE_|WRP_RUNTIME_ENABLE_|CHIMAERA_ENABLE_) ]]; then
        CMAKE_EXTRA_ARGS+=("-D${var}=${!var}")
    fi
done

echo "Forwarding environment variables to cmake:"
for arg in "${CMAKE_EXTRA_ARGS[@]}"; do
    echo "  $arg"
done
echo ""

# Clean and create build directory
# Remove any existing build directory to avoid CMakeCache conflicts
rm -rf build
mkdir -p build
cd build

# Configure with CMake using the conda preset
# The conda preset enables WRP_CORE_ENABLE_CONDA which uses $CONDA_PREFIX for paths
cmake .. \
    --preset conda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DWRP_CORE_ENABLE_CONDA=ON \
    "${CMAKE_EXTRA_ARGS[@]}"

# Build and install
make -j${CPU_COUNT} VERBOSE=1
make install
