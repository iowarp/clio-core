#!/bin/bash
# install.sh - Install IOWarp Core and all dependencies to a single prefix
# This script detects missing dependencies and builds/installs them from downloaded sources or submodules

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Initialize and update git submodules recursively (if in a git repository)
if [ -d ".git" ]; then
    echo ">>> Initializing git submodules..."
    git submodule update --init --recursive
    echo ""
else
    echo ">>> Not a git repository, skipping submodule initialization"
    echo ""
fi

# Default install prefix
: ${INSTALL_PREFIX:=/usr/local}
: ${BUILD_JOBS:=$(nproc)}
: ${DEPS_ONLY:=FALSE}
: ${WRP_CORE_ENABLE_BENCHMARKS:=OFF}
: ${WRP_CORE_ENABLE_TESTS:=OFF}
: ${WRP_CORE_ENABLE_MPI:=OFF}

echo "======================================================================"
echo "IOWarp Core Installer"
echo "======================================================================"
echo "Install prefix: $INSTALL_PREFIX"
echo "Build jobs: $BUILD_JOBS"
echo "Dependencies only: $DEPS_ONLY"
echo "Build benchmarks: $WRP_CORE_ENABLE_BENCHMARKS"
echo "Build tests: $WRP_CORE_ENABLE_TESTS"
echo "With MPI: $WRP_CORE_ENABLE_MPI"
echo ""

#------------------------------------------------------------------------------
# Step 0: Check for MPI if enabled
#------------------------------------------------------------------------------
if [ "$WRP_CORE_ENABLE_MPI" = "ON" ] || [ "$WRP_CORE_ENABLE_MPI" = "on" ] || [ "$WRP_CORE_ENABLE_MPI" = "1" ] || [ "$WRP_CORE_ENABLE_MPI" = "TRUE" ] || [ "$WRP_CORE_ENABLE_MPI" = "true" ]; then
    echo ">>> Checking for MPI installation..."

    # Check for mpicc or mpicxx
    if ! command -v mpicc &> /dev/null && ! command -v mpicxx &> /dev/null && ! command -v mpic++ &> /dev/null; then
        echo "ERROR: MPI is not installed on this system!"
        echo ""
        echo "WRP_CORE_ENABLE_MPI=ON was specified, but no MPI compiler wrappers were found."
        echo "Please install MPI before running this script with MPI enabled."
        echo ""
        echo "Installation instructions:"
        echo "  Ubuntu/Debian: sudo apt-get install libmpich-dev mpich"
        echo "                 or: sudo apt-get install libopenmpi-dev openmpi-bin"
        echo "  macOS:         brew install mpich"
        echo "                 or: brew install open-mpi"
        echo ""
        exit 1
    fi

    # Detect which MPI implementation is installed
    if command -v mpicc &> /dev/null; then
        MPI_VERSION=$(mpicc --version 2>&1 | head -n 1)
        echo "✓ MPI found: $MPI_VERSION"
    elif command -v mpicxx &> /dev/null; then
        MPI_VERSION=$(mpicxx --version 2>&1 | head -n 1)
        echo "✓ MPI found: $MPI_VERSION"
    elif command -v mpic++ &> /dev/null; then
        MPI_VERSION=$(mpic++ --version 2>&1 | head -n 1)
        echo "✓ MPI found: $MPI_VERSION"
    fi
    echo ""
fi

#------------------------------------------------------------------------------
# Step 1: Detect Missing Dependencies
#------------------------------------------------------------------------------
echo ">>> Detecting missing dependencies..."
echo ""

# Create build directory for detection
mkdir -p build/detect

# Run CMake dependency detection
cmake -S cmake/detect -B build/detect \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
    > build/detect/cmake_output.log 2>&1

# Source the detection results
if [ -f build/detect/dependency_status.txt ]; then
    echo "Loading dependency detection results..."
    source build/detect/dependency_status.txt
    
    echo "Dependency status:"
    echo "  NEED_BOOST:    $NEED_BOOST"
    echo "  NEED_ZEROMQ:   $NEED_ZEROMQ"
    echo "  NEED_HDF5:     $NEED_HDF5"
    echo ""
else
    echo "Warning: Could not find dependency detection results"
    echo "Assuming all dependencies need to be built"
    NEED_BOOST=1
    NEED_ZEROMQ=1
    NEED_HDF5=1
fi

#------------------------------------------------------------------------------
# Step 2: Build and Install Missing Dependencies
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Boost - Download and build fiber, context, system libraries
#------------------------------------------------------------------------------
if [ "$NEED_BOOST" = "1" ] || [ "$NEED_BOOST" = "TRUE" ]; then
    echo ">>> Downloading and building Boost..."

    BOOST_VERSION="1.89.0"
    BOOST_ARCHIVE="boost-${BOOST_VERSION}-cmake.tar.gz"
    BOOST_URL="https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/${BOOST_ARCHIVE}"
    BOOST_DIR="boost-${BOOST_VERSION}"

    # Download Boost if not already downloaded
    if [ ! -f "build/external/${BOOST_ARCHIVE}" ]; then
        echo "Downloading Boost ${BOOST_VERSION}..."
        mkdir -p build/external
        curl -L -o "build/external/${BOOST_ARCHIVE}" "${BOOST_URL}"
    fi

    # Extract Boost
    echo "Extracting Boost..."
    cd build/external
    tar -xzf "${BOOST_ARCHIVE}"
    cd "${BOOST_DIR}"

    # Build Boost using CMake
    mkdir -p build
    cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DBOOST_INCLUDE_LIBRARIES="fiber;context;system;filesystem;atomic" \
        -DBOOST_ENABLE_CMAKE=ON

    cmake --build . -j${BUILD_JOBS}
    cmake --install .

    cd "$SCRIPT_DIR"
    echo "✓ Boost installed to $INSTALL_PREFIX"
else
    echo "✓ Boost already available, skipping"
fi
echo ""

#------------------------------------------------------------------------------
# ZeroMQ - Download and build static library
#------------------------------------------------------------------------------
if [ "$NEED_ZEROMQ" = "1" ] || [ "$NEED_ZEROMQ" = "TRUE" ]; then
    echo ">>> Downloading and building ZeroMQ..."

    ZEROMQ_VERSION="4.3.5"
    ZEROMQ_ARCHIVE="libzmq-${ZEROMQ_VERSION}.tar.gz"
    ZEROMQ_URL="https://github.com/zeromq/libzmq/archive/refs/tags/v${ZEROMQ_VERSION}.tar.gz"
    ZEROMQ_DIR="libzmq-${ZEROMQ_VERSION}"

    # Download ZeroMQ if not already downloaded
    if [ ! -f "build/external/${ZEROMQ_ARCHIVE}" ]; then
        echo "Downloading ZeroMQ ${ZEROMQ_VERSION}..."
        mkdir -p build/external
        curl -L -o "build/external/${ZEROMQ_ARCHIVE}" "${ZEROMQ_URL}"
    fi

    # Extract ZeroMQ
    echo "Extracting ZeroMQ..."
    cd build/external
    tar -xzf "${ZEROMQ_ARCHIVE}"
    cd "${ZEROMQ_DIR}"

    # Build ZeroMQ using CMake
    mkdir -p build
    cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED=OFF \
        -DBUILD_STATIC=ON \
        -DBUILD_TESTS=OFF \
        -DWITH_PERF_TOOL=OFF \
        -DENABLE_CPACK=OFF

    cmake --build . -j${BUILD_JOBS}
    cmake --install .

    cd "$SCRIPT_DIR"
    echo "✓ ZeroMQ installed to $INSTALL_PREFIX"
else
    echo "✓ ZeroMQ already available, skipping"
fi
echo ""

#------------------------------------------------------------------------------
# HDF5 - Download and build static library
#------------------------------------------------------------------------------
if [ "$NEED_HDF5" = "1" ] || [ "$NEED_HDF5" = "TRUE" ]; then
    echo ">>> Downloading and building HDF5..."

    HDF5_VERSION="2.0.0"
    HDF5_ARCHIVE="hdf5-${HDF5_VERSION}.tar.gz"
    HDF5_URL="https://github.com/HDFGroup/hdf5/archive/refs/tags/${HDF5_VERSION}.tar.gz"
    HDF5_DIR="hdf5-${HDF5_VERSION}"

    # Download HDF5 if not already downloaded
    if [ ! -f "build/external/${HDF5_ARCHIVE}" ]; then
        echo "Downloading HDF5 ${HDF5_VERSION}..."
        mkdir -p build/external
        curl -L -o "build/external/${HDF5_ARCHIVE}" "${HDF5_URL}"
    fi

    # Extract HDF5
    echo "Extracting HDF5..."
    cd build/external
    tar -xzf "${HDF5_ARCHIVE}"
    cd "${HDF5_DIR}"

    # Build HDF5 using CMake
    mkdir -p build
    cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DHDF5_BUILD_EXAMPLES=OFF \
        -DHDF5_BUILD_TOOLS=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_CPP_LIB=OFF \
        -DHDF5_BUILD_FORTRAN=OFF \
        -DHDF5_BUILD_JAVA=OFF \
        -DHDF5_ENABLE_PARALLEL=OFF \
        -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF \
        -DHDF5_ENABLE_SZIP_SUPPORT=OFF

    cmake --build . -j${BUILD_JOBS}
    cmake --install .

    cd "$SCRIPT_DIR"
    echo "✓ HDF5 installed to $INSTALL_PREFIX"
else
    echo "✓ HDF5 already available, skipping"
fi
echo ""

#------------------------------------------------------------------------------
# Step 3: Build and Install IOWarp Core
#------------------------------------------------------------------------------

# Skip IOWarp Core build if DEPS_ONLY is set
if [ "$DEPS_ONLY" = "TRUE" ] || [ "$DEPS_ONLY" = "true" ] || [ "$DEPS_ONLY" = "1" ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Dependencies installed successfully!"
    echo "======================================================================"
    echo "DEPS_ONLY mode enabled - skipping IOWarp Core build"
    echo "Installation prefix: $INSTALL_PREFIX"
    echo ""
    echo "To build IOWarp Core manually, run:"
    echo "  cmake --preset=minimalist -DCMAKE_INSTALL_PREFIX=\"$INSTALL_PREFIX\" -DCMAKE_PREFIX_PATH=\"$INSTALL_PREFIX/lib/cmake;$INSTALL_PREFIX/cmake;$INSTALL_PREFIX\""
    echo "  cmake --build build -j${BUILD_JOBS}"
    echo "  cmake --install build"
    echo ""
    exit 0
fi

echo "======================================================================"
echo ">>> Building IOWarp Core..."
echo "======================================================================"
echo ""

# Set PKG_CONFIG_PATH for dependency detection (ZeroMQ)
export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# Create build directory for IOWarp Core
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

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

# Configure IOWarp Core with the same prefix
# Note: CMAKE_PREFIX_PATH includes multiple paths for different package locations:
#   - $INSTALL_PREFIX/lib/cmake - Standard location (cereal, boost, ZeroMQ)
#   - $INSTALL_PREFIX/cmake - Non-standard location (HDF5)
#   - $INSTALL_PREFIX - General fallback
cmake -S . -B "$BUILD_DIR" \
    --preset=minimalist \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX/lib/cmake;$INSTALL_PREFIX/cmake;$INSTALL_PREFIX" \
    "${CMAKE_EXTRA_ARGS[@]}"

# Build IOWarp Core
cmake --build "$BUILD_DIR" -j${BUILD_JOBS}

# Install IOWarp Core
cmake --install "$BUILD_DIR"

echo ""
echo "======================================================================"
echo "✓ IOWarp Core and dependencies installed successfully!"
echo "======================================================================"
echo "Installation prefix: $INSTALL_PREFIX"
echo ""
echo "To use IOWarp Core, ensure the following environment variables are set:"
echo "  export CMAKE_PREFIX_PATH=\"$INSTALL_PREFIX:\$CMAKE_PREFIX_PATH\""
echo "  export LD_LIBRARY_PATH=\"$INSTALL_PREFIX/lib:\$LD_LIBRARY_PATH\""
echo "  export PKG_CONFIG_PATH=\"$INSTALL_PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH\""
echo "  export PYTHONPATH=\"$INSTALL_PREFIX/lib/python\$(python3 -c 'import sys; print(\".\".join(map(str, sys.version_info[:2])))')/site-packages:\$PYTHONPATH\""
echo ""
