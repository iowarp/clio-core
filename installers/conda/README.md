# IOWarp Core - Conda Package

This directory contains the Conda recipe for building and distributing IOWarp Core.

## Quick Start

### Building from Local Source (Recommended)

The easiest way to build and install IOWarp Core from local source is using the `conda-local.sh` script:

```bash
# From the repository root directory
cd conda

# Run the interactive build and install script
./conda-local.sh
```

The script will:
- Check for conda environment and conda-build installation
- Verify and initialize git submodules if needed
- Build the conda package from your local source
- Offer to install the package interactively
- Provide helpful installation options and troubleshooting steps

### Manual Build

If you prefer to build manually:

```bash
# Make sure submodules are initialized
git submodule update --init --recursive

# Build the package
conda build conda/ -c conda-forge

# Install
conda install --use-local iowarp-core
```

## Build System Configuration

IOWarp Core's CMake system is **optimized for Conda builds** with the following features:

### Automatic Conda Detection

When building in a Conda environment, CMake automatically:
- Detects `$CONDA_PREFIX` environment variable
- Sets `CMAKE_INSTALL_PREFIX` to `$CONDA_PREFIX`
- Prioritizes Conda packages in `CMAKE_PREFIX_PATH`
- Configures `PKG_CONFIG_PATH` to find Conda libraries first

### Conda-Optimized CMake Preset

The build uses the `conda` preset (defined in `CMakePresets.json`):
- Build type: Release
- Python bindings: OFF (Conda manages Python packages separately)
- Tests: OFF (for faster builds)
- RPATH: ON (for relocatable installations)
- All core components enabled (runtime, CTE, CAE, CEE)

## Dependencies

The conda recipe automatically handles all dependencies:

### Build Dependencies
- C/C++ compilers
- CMake >= 3.15
- Make
- Git (for submodules)

### Runtime Dependencies (from Conda)
- HDF5
- ZeroMQ
- Boost (fiber, context, system)
- yaml-cpp
- cereal (header-only)

All dependencies are installed automatically via Conda during the build process.

## Customizing the Build

### Environment Variables

You can customize the build by setting environment variables with specific prefixes:

```bash
# Example: Enable MPI support
export WRP_CORE_ENABLE_MPI=ON
conda build conda/ -c conda-forge

# Example: Enable compression
export WRP_CORE_ENABLE_COMPRESS=ON
export HSHM_ENABLE_COMPRESS=ON
conda build conda/ -c conda-forge
```

Supported prefixes (forwarded to CMake):
- `WRP_CORE_ENABLE_*` - Core component options
- `WRP_CTE_ENABLE_*` - Context Transfer Engine options
- `WRP_CAE_ENABLE_*` - Context Assimilation Engine options
- `WRP_CEE_ENABLE_*` - Context Exploration Engine options
- `HSHM_ENABLE_*` - HSHM/transport primitives options
- `WRP_RUNTIME_ENABLE_*` - Runtime options
- `CHIMAERA_ENABLE_*` - Chimaera runtime options

### Modifying the Recipe

Edit `meta.yaml` to change:
- Version number
- Dependencies
- Build number
- Package metadata

Edit `build.sh` to change:
- CMake preset (default: `conda`)
- Build flags
- Post-install steps

## Installation Layout

After installation, IOWarp Core files are organized as follows:

```
$CONDA_PREFIX/
├── bin/                           # Command-line tools
│   ├── chimaera_start_runtime
│   ├── wrp_cte
│   ├── wrp_cae_omni
│   └── ...
├── lib/                           # Shared libraries
│   ├── libchimaera_cxx.so
│   ├── libhermes_shm_host.so
│   ├── chimaera_admin_runtime.so
│   └── ...
├── include/                       # C++ headers
│   ├── chimaera/
│   ├── hshm/
│   └── ...
└── lib/cmake/                     # CMake package configs
    ├── iowarp-core/
    ├── chimaera/
    ├── HermesShm/
    └── ...
```

## Using IOWarp Core from Conda

After installation, you can use IOWarp Core in several ways:

### 1. Command-Line Tools

```bash
# Start the Chimaera runtime
chimaera_start_runtime

# Use CTE tools
wrp_cte --help
```

### 2. C++ Development

```cmake
# In your CMakeLists.txt
find_package(iowarp-core REQUIRED)

target_link_libraries(your_app
    chimaera::admin_client
    wrp_cte::core_client
)
```

### 3. Environment Variables

The Conda environment automatically sets up paths for:
- Libraries (`LD_LIBRARY_PATH`)
- Headers (`CPATH`)
- CMake configs (`CMAKE_PREFIX_PATH`)

## Troubleshooting

### Submodule Issues

If you get errors about missing submodules:

```bash
# Initialize submodules before building
git submodule update --init --recursive
```

### Dependency Conflicts

If you encounter dependency version conflicts:

```bash
# Create a clean conda environment
conda create -n iowarp-dev
conda activate iowarp-dev

# Build in the clean environment
conda build conda/ -c conda-forge
```

### Build Failures

Enable verbose output to diagnose issues:

```bash
# The build.sh already uses VERBOSE=1 for make
# Check the build output for specific error messages
conda build conda/ -c conda-forge
```

## Publishing to Anaconda.org

To publish the package to Anaconda.org:

```bash
# Build the package
conda build conda/ -c conda-forge

# Upload to your Anaconda.org channel
anaconda upload $CONDA_PREFIX/conda-bld/linux-64/iowarp-core-*.tar.bz2

# Or use conda-build's upload
conda build conda/ --output-folder ./output
anaconda upload ./output/linux-64/iowarp-core-*.tar.bz2
```

## Differences from Pip Install

| Feature | Conda Build | Pip Install |
|---------|-------------|-------------|
| Dependency Management | Conda packages | System/bundled |
| Python Bindings | Separate package | Included in wheel |
| Installation Prefix | `$CONDA_PREFIX` | Virtual env or system |
| RPATH | Enabled | Enabled |
| Build Preset | `conda` | `minimalist` |

## More Information

- Main README: `../README.md`
- Build wheel guide: `../BUILD_WHEEL.md`
- Contributing guide: `../docs/contributing.md`
- CMake presets: `../CMakePresets.json`
