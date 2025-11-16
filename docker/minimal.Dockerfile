#==============================================================================
# IOWarp Core - Minimal Development Container
#==============================================================================
# This Dockerfile creates a minimal development environment for IOWarp Core
# with Python, venv, and essential build dependencies.
#
# Purpose:
# - Provides a clean environment for pip-based installation
# - Includes Python 3 with venv support
# - Contains essential build tools for compiling C++ extensions
# - Source code should be mounted as a volume (not copied during build)
#
# Build Command:
#   docker build -f docker/minimal.Dockerfile -t iowarp-minimal .
#
# Run Command:
#   docker run --rm -it -v /path/to/iowarp-core:/iowarp-core iowarp-minimal
#
# Or use install_docker.sh which handles building and installation automatically
#==============================================================================

FROM ubuntu:22.04

#------------------------------------------------------------------------------
# Install Essential Build Dependencies and Python
#------------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Build tools (cmake from apt will be replaced with 3.28)
    make \
    g++ \
    gcc \
    # Version control (for git submodules)
    git \
    # Download tool (for install.sh and cmake)
    curl \
    wget \
    # Python and venv
    python3 \
    python3-pip \
    python3-venv \
    # Required for ZeroMQ build
    pkg-config \
    libtool \
    autoconf \
    automake \
    # Required for CMake 3.28 build
    libssl-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------------------------
# Install CMake 3.28
#------------------------------------------------------------------------------
RUN cd /tmp && \
    CMAKE_VERSION=3.28.3 && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz && \
    tar -xzf cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz && \
    rm -rf /usr/local/man && \
    cp -rf cmake-${CMAKE_VERSION}-linux-$(uname -m)/* /usr/local/ && \
    rm -rf cmake-${CMAKE_VERSION}-linux-$(uname -m) cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz && \
    cmake --version

#------------------------------------------------------------------------------
# Create Python Virtual Environment
#------------------------------------------------------------------------------
# Create a virtual environment that will be used for installation
# This ensures a clean, isolated Python environment
RUN python3 -m venv /opt/venv && \
    # Activate and upgrade pip in the venv
    /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Add virtual environment to PATH so it's activated by default
ENV PATH="/opt/venv/bin:$PATH"

#------------------------------------------------------------------------------
# Copy Source Code
#------------------------------------------------------------------------------
# Copy the entire source tree into the image
WORKDIR /iowarp-core
COPY . /iowarp-core/

#------------------------------------------------------------------------------
# Initialize Git Submodules
#------------------------------------------------------------------------------
# Configure git to trust the directory and initialize submodules
# Note: We use --init without --recursive because install.sh will handle
# recursive initialization of nested submodules (like nanobind's robin_map)
RUN rm -rf build && git config --global --add safe.directory /iowarp-core 

#------------------------------------------------------------------------------
# Default Command
#------------------------------------------------------------------------------
# Start an interactive shell for development/testing
CMD ["/bin/bash"]
