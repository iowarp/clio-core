# IOWarp Core - Conda Package

This directory contains the conda recipe for building IOWarp Core using [rattler-build](https://prefix-dev.github.io/rattler-build/).

## Quick Start (Recommended)

The easiest way to install IOWarp Core is using the main install script at the repository root:

```bash
# Clone the repository
git clone --recursive https://github.com/iowarp/iowarp-core.git
cd iowarp-core

# Install with default (release) variant
./install.sh

# Or specify a variant
./install.sh debug
./install.sh cuda
./install.sh rocm
```

The install script will:
1. Install Miniconda if conda is not detected
2. Create and activate an `iowarp` conda environment (if not already in one)
3. Install rattler-build if not present
4. Build the conda package using the specified variant
5. Install the package directly into the current environment

## Alternative: Using conda-local.sh

If you're already in a conda environment and want more control:

```bash
# Activate your conda environment
conda activate myenv

# Build with default (release) variant
./installers/conda/conda-local.sh

# Or specify a variant
./installers/conda/conda-local.sh cuda
./installers/conda/conda-local.sh mpi
./installers/conda/conda-local.sh custom
```

## Installation Modes

IOWarp Core supports two installation modes:

### 1. Precompiled Binary (Recommended)

Uses CMakePresets.json directly for reproducible builds:

```bash
# Build with a specific preset
./conda-local.sh release    # Standard release build
./conda-local.sh debug      # Debug build with symbols
./conda-local.sh conda      # Conda-optimized release build
./conda-local.sh cuda       # CUDA-enabled build (Linux only)
./conda-local.sh rocm       # ROCm-enabled build (Linux only)
```

### 2. Custom Source Build

Allows fine-grained control over build options:

```bash
# Use the custom variant as a template
./conda-local.sh custom

# Or create your own variant file
cp variants/custom.yaml variants/my-config.yaml
# Edit my-config.yaml to your needs
./conda-local.sh variants/my-config.yaml
```

## Available Variants

| Variant | Description | Preset Used |
|---------|-------------|-------------|
| `release` | Standard release build | `release` |
| `debug` | Debug build with symbols | `debug` |
| `conda` | Conda-optimized build | `conda` |
| `cuda` | CUDA-enabled (Linux) | `cuda-debug` |
| `rocm` | ROCm-enabled (Linux) | `rocm-debug` |
| `mpi` | MPI-enabled build | `custom` |
| `full` | All features enabled | `custom` |
| `custom` | Template for custom builds | `custom` |

## Custom Variant Options

When using `preset: custom`, you can set these options:

```yaml
# Build configuration
preset: custom
python: "3.11"
build_type: Release  # or Debug

# GPU support (choose one or none)
cuda_enabled: "OFF"
rocm_enabled: "OFF"

# Distributed computing
mpi_enabled: "OFF"
mpi_impl: nompi  # openmpi, mpich, msmpi (Windows)

# Core components
runtime_enabled: "ON"
cte_enabled: "ON"
cae_enabled: "ON"
cee_enabled: "ON"

# Optional features
zmq_enabled: "ON"
hdf5_enabled: "ON"
elf_enabled: "OFF"
```

## Directory Structure

```
installers/conda/
├── recipe.yaml              # Main recipe (rattler-build format)
├── conda_build_config.yaml  # Variant definitions
├── conda-forge.yml          # conda-forge CI configuration
├── conda-local.sh           # Local build script
├── variants/                # Predefined variant files
│   ├── release.yaml
│   ├── debug.yaml
│   ├── conda.yaml
│   ├── cuda.yaml
│   ├── rocm.yaml
│   ├── mpi.yaml
│   ├── full.yaml
│   └── custom.yaml
└── README.md
```

## Manual Build Commands

If you prefer to run rattler-build directly:

```bash
# Install rattler-build if not present
conda install -y rattler-build -c conda-forge

# From repository root, build the package
rattler-build build \
    --recipe installers/conda/ \
    --variant-config installers/conda/variants/release.yaml \
    --output-dir build/conda-output \
    -c conda-forge

# Find the built package
PACKAGE=$(find build/conda-output -name "iowarp-core-*.conda" | head -1)

# Install directly from the package file
conda install "$PACKAGE" -y
```

## Dependencies

The conda recipe automatically handles all dependencies:

### Build Dependencies
- C/C++ compilers
- CMake >= 3.20
- Ninja
- pkg-config

### Runtime Dependencies (from Conda)
- Python
- HDF5
- ZeroMQ
- yaml-cpp
- cereal
- nanobind (for Python bindings)
- CUDA toolkit (optional, for GPU builds)
- MPI (optional, openmpi/mpich)

## Installation Layout

After installation, IOWarp Core files are organized as follows:

```
$CONDA_PREFIX/
├── bin/                           # Command-line tools
│   ├── chimaera runtime start
│   ├── wrp_cte
│   ├── wrp_cae_omni
│   └── ...
├── lib/                           # Shared libraries
│   ├── libchimaera_cxx.so
│   ├── libhermes_shm_host.so
│   ├── chimaera_admin_runtime.so
│   └── ...
├── lib/python3.X/site-packages/   # Python modules
│   ├── wrp_cte/
│   └── wrp_cee/
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
chimaera runtime start

# Use CTE tools
wrp_cte --help
```

### 2. Python

```python
import wrp_cte
import wrp_cee

# Use the Python bindings
```

### 3. C++ Development

```cmake
# In your CMakeLists.txt
find_package(iowarp-core REQUIRED)

target_link_libraries(your_app
    chimaera::admin_client
    wrp_cte::core_client
)
```

## conda-forge Submission

To submit to conda-forge:

1. Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
2. Copy `recipe.yaml` and `conda_build_config.yaml` to `recipes/iowarp-core/`
3. Submit a pull request

The CI will automatically build all variant combinations defined in `conda_build_config.yaml`.

## Troubleshooting

### rattler-build not found

```bash
conda install rattler-build -c conda-forge
```

### Build fails with missing dependencies

Ensure conda-forge channel is configured:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### Submodule Issues

If you get errors about missing submodules:

```bash
git submodule update --init --recursive
```

### CUDA/ROCm builds fail

GPU builds are only supported on Linux. Check that you have the appropriate toolkit installed:

```bash
# For CUDA
conda install cuda-toolkit -c nvidia

# For ROCm
# Follow AMD's ROCm installation guide
```

## Requirements

- conda or mamba
- rattler-build (`conda install rattler-build -c conda-forge`)
- Git (for submodule initialization)

## More Information

- Main README: `../../README.md`
- Build wheel guide: `../../BUILD_WHEEL.md`
- Contributing guide: `../../docs/contributing.md`
- CMake presets: `../../CMakePresets.json`
