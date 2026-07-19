# IOWarp Core Installers

This directory contains various installation methods for IOWarp Core.

## Available Installation Methods

### 1. Conda Package (`conda/`)

Install the prebuilt `iowarp-core` package, or build the recipe from source.

**Quick Start:**
```bash
# Install the prebuilt package from the iowarp channel (recommended)
conda install -c iowarp -c conda-forge iowarp-core

# Or build the recipe from source (IOWARP_PRESET selects a CMake preset)
conda install -n base -y conda-build -c conda-forge
IOWARP_PRESET=release conda build installers/conda/ -c conda-forge --output-folder build/conda-output
conda install -c conda-forge build/conda-output/*/iowarp-core-*.conda
```

**Documentation:** See [`conda/README.md`](conda/README.md)

### 2. vcpkg Port (`vcpkg/`)

Install IOWarp Core using Microsoft's vcpkg package manager.

**Quick Start:**
```bash
# Install from local overlay
vcpkg install iowarp-core --overlay-ports=installers/vcpkg

# Or contribute to vcpkg registry (future)
vcpkg install iowarp-core
```

**Documentation:** See [`vcpkg/README.md`](vcpkg/README.md)

## Choosing an Installation Method

| Method | Best For | Platform Support |
|--------|----------|------------------|
| **Conda** | Python users, scientific computing environments | Linux, macOS |
| **vcpkg** | C++ developers, cross-platform projects | Linux, macOS |
| **Source** | Developers, contributors, custom builds | Linux, macOS, Windows (partial) |

## Source Installation

For building from source without a package manager:

```bash
# Clone the repository
git clone https://github.com/iowarp/clio-core.git
cd clio-core

# Build with CMake
cmake --preset=release
cmake --build build --parallel
cmake --install build --prefix /path/to/install
```

See the main [BUILD.md](../BUILD.md) for detailed build instructions.

## Dependencies

All installation methods install these dependencies automatically:
- Boost (fiber, context, system, filesystem)
- ZeroMQ
- yaml-cpp
- cereal
- HDF5
- Catch2 (for testing)

## Platform Support

| Platform | Conda | vcpkg | Source |
|----------|-------|-------|--------|
| Linux    | ✅     | ✅     | ✅      |
| macOS    | ✅     | ✅     | ✅      |
| Windows  | ❌     | ❌     | ⚠️ Partial |

## Contributing

To contribute improvements to installation methods:

1. **Conda**: Update files in `installers/conda/`
2. **vcpkg**: Update files in `installers/vcpkg/`
3. Test your changes locally before submitting a PR

## License

BSD-3-Clause - See [LICENSE](../LICENSE) for details
