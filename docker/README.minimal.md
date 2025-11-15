# IOWarp Core - Minimal Docker Build

This directory contains the minimal Docker configuration for pip-based installation testing of IOWarp Core.

## Overview

The `minimal.Dockerfile` provides a containerized build environment that:

- Uses only essential build dependencies (cmake, gcc, make, git, python3)
- Installs IOWarp Core via **pip install** (using setup.py → install.sh workflow)
- Builds dependencies from submodules (Boost, ZeroMQ, cereal, yaml-cpp)
- Installs directly to the Python virtual environment (`/opt/venv`)
- Validates the complete pip-based installation workflow

## Dependencies

### Build Tools (from Ubuntu 22.04)
- `cmake` - Build system generator
- `make` - Build automation
- `g++` / `gcc` - C/C++ compilers
- `git` - Version control (for submodule initialization)
- `python3` - Required for Boost build
- `pkg-config`, `libtool`, `autoconf`, `automake` - Required for ZeroMQ build

### Libraries (from Git Submodules)
All libraries are built from source using git submodules:
- **Boost** - Required components: fiber, context, system
- **ZeroMQ** (libzmq) - Messaging library
- **HDF5** - Hierarchical data format (minimal build: C library only)
- **cereal** - C++ serialization library
- **nanobind** - Python bindings (not used in minimal build)
- **Catch2** - Testing framework (not used in minimal build)
- **yaml-cpp** - YAML parser

## Build Configuration

The minimal build uses the following CMake options (from `minimalist` preset):

```cmake
CMAKE_BUILD_TYPE=Release
WRP_CORE_ENABLE_TESTS=OFF
WRP_CORE_ENABLE_BENCHMARKS=OFF
WRP_CORE_ENABLE_PYTHON=OFF
WRP_CORE_ENABLE_MPI=OFF
WRP_CORE_ENABLE_ELF=OFF
WRP_CORE_ENABLE_RPATH=OFF
WRP_CORE_ENABLE_ZMQ=ON
WRP_CORE_ENABLE_HDF5=ON
WRP_CORE_ENABLE_CEREAL=ON
```

## Quick Start - Automated Installation

The easiest way to test the pip-based installation is using the provided script:

```bash
cd docker
./install_docker.sh
```

This script will:
1. Build the minimal Docker image with source code and initialized submodules
2. Run `pip install` inside the container
3. Install IOWarp Core to `/opt/venv` (the container's virtual environment)
4. Verify the installation by checking for libraries and CMake configs

Expected output:
```
Successfully installed iowarp-core-0.0.post98

Checking installed libraries in /opt/venv/lib:
-rw-r--r-- 1 root root 2.0M libchimaera_cxx.so.1.0.0
-rw-r--r-- 1 root root 803K libwrp_cte_core_runtime.so
...

Checking CMake config files:
/opt/venv/lib/cmake/chimaera/
/opt/venv/lib/cmake/wrp_cte_core/
/opt/venv/lib/cmake/iowarp-core/
...

✓ Installation test complete!
```

## Manual Build Steps

If you want to build manually without the script:

### 1. Build the Docker Image

From the repository root:

```bash
docker build -f docker/minimal.Dockerfile -t iowarp/minimal:latest .
```

### 2. Run pip install in the Container

```bash
docker run --rm iowarp/minimal:latest /bin/bash -c "pip install -v ."
```

## Size Optimization

The minimal build significantly reduces:
- **Build time**: No tests, benchmarks, or optional features
- **Dependencies**: Only essential submodules are built
- **Binary size**: Release mode with optimization
- **Attack surface**: Minimal feature set reduces potential vulnerabilities

## Installation Approach

IOWarp Core uses **direct installation to sys.prefix** (not wheel bundling):

- **Libraries** → `{sys.prefix}/lib/` (e.g., `/opt/venv/lib/`)
- **Headers** → `{sys.prefix}/include/`
- **Binaries** → `{sys.prefix}/bin/` (automatically in PATH)
- **CMake configs** → `{sys.prefix}/lib/cmake/`

This approach is standard for C++ libraries with Python bindings and ensures:
- ✅ CMake can find packages with `find_package(iowarp-core)`
- ✅ Binaries are automatically in PATH
- ✅ Works like system package managers (apt, yum, conda)

To change this behavior, set `IOWARP_BUNDLE_BINARIES=ON` to bundle into the wheel instead.

## Use Cases

This minimal configuration is ideal for:
- **pip-based installation testing** - Validates the setup.py → install.sh workflow
- **CI/CD validation** - Tests that dependencies build correctly from submodules
- **Virtual environment installation** - Standard Python package installation
- **Dependency auditing** - Clear view of what's actually required

## Differences from Full Build

The minimal build **excludes**:
- Unit tests and test frameworks
- Benchmarks
- Python bindings
- MPI support
- ELF/adapter support
- RPATH embedding
- Development tools (ASAN, coverage, doxygen)
- Accelerator support (CUDA, ROCm)

## Troubleshooting

### Submodule Issues
If you see errors about missing submodules:
```bash
# Ensure submodules are initialized before building
git submodule update --init --recursive
```

### Build Failures
Check the build output in the Docker logs:
```bash
docker build -f docker/minimal.Dockerfile -t iowarp-minimal . 2>&1 | tee build.log
```

### Container Won't Start
Verify the image was built successfully:
```bash
docker images | grep iowarp-minimal
```

## Related Files

- `install_docker.sh` - Automated script for building and installing in Docker
- `minimal.Dockerfile` - Dockerfile for minimal build environment
- `/.dockerignore` - Excludes build artifacts from Docker context
- `/setup.py` - Python packaging that calls install.sh
- `/install.sh` - Unified installer for dependencies and IOWarp Core
- `/cmake/detect/` - CMake dependency detection system
- `/CMakePresets.json` - Contains the `minimalist` preset definition
- `/CMakeLists.txt` - Root build configuration with all options

## Comparison with Other Presets

| Preset | Build Type | Tests | Dependencies | Use Case |
|--------|-----------|-------|--------------|----------|
| **minimalist** | Release | OFF | Submodules only | Production, minimal footprint |
| **debug** | Debug | ON | System + submodules | Development, debugging |
| **release** | Release | ON | System + submodules | Release validation, CI/CD |

## Future Enhancements

Potential improvements for the minimal build:
- Multi-stage build to reduce final image size
- Alpine Linux base for even smaller footprint
- Static linking to eliminate runtime dependencies
- Distroless final image for production security
