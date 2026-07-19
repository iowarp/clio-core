# IOWarp Core - Conda Package

This directory contains the conda recipe (`meta.yaml`, built with
[conda-build](https://docs.conda.io/projects/conda-build/)) for `iowarp-core`.

## Install (Recommended)

Prebuilt `iowarp-core` packages are published to the
**[`iowarp` channel on Anaconda.org](https://anaconda.org/iowarp/iowarp-core)**
(CI builds and uploads them on each `v*` release tag).

**1. Install Miniconda** (skip if you already have conda or mamba). Download and
run the binary installer for your platform:

```bash
# Linux x86_64
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

For other platforms, use the matching installer from the
[Miniconda downloads](https://www.anaconda.com/download/success):

| Platform | Installer |
|----------|-----------|
| Linux aarch64 | `Miniconda3-latest-Linux-aarch64.sh` |
| macOS (Apple Silicon) | `Miniconda3-latest-MacOSX-arm64.sh` |
| macOS (Intel) | `Miniconda3-latest-MacOSX-x86_64.sh` |
| Windows | `Miniconda3-latest-Windows-x86_64.exe` (run the installer) |

**2. Install `iowarp-core`** from the `iowarp` channel (with `conda-forge` for
its dependencies):

```bash
# Into a fresh environment (recommended)
conda create -n iowarp -c iowarp -c conda-forge iowarp-core
conda activate iowarp

# ...or into the current environment
conda install -c iowarp -c conda-forge iowarp-core
```

This installs the command-line tools, Python bindings, C++ headers, and CMake
package configs (see [Installation Layout](#installation-layout)).

## Building from source

To build the package yourself (e.g. to enable CUDA/ROCm/MPI or to test local
changes), build the recipe with `conda-build`. `IOWARP_PRESET` selects a
[CMake preset](../../CMakePresets.json); this mirrors what CI runs.

```bash
# conda-build is a plugin registered against the BASE env's conda CLI,
# so it must be installed there.
conda install -n base -y conda-build -c conda-forge

# From the repository root. Common presets: release, debug,
# cuda-release, rocm-release (see CMakePresets.json for the full list).
IOWARP_PRESET=release conda build installers/conda/ \
    -c conda-forge \
    --output-folder build/conda-output

# Install the package you just built (conda-forge supplies runtime deps)
conda install -c conda-forge build/conda-output/*/iowarp-core-*.conda
```

> Git submodules must be initialized first: `git submodule update --init --recursive`.

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
│   ├── clio_run runtime start
│   ├── clio_cte
│   ├── clio_cae
│   └── ...
├── lib/                           # Shared libraries
│   ├── libclio_run_cxx.so
│   ├── libclio_ctp_host.so
│   ├── clio_admin_runtime.so
│   └── ...
├── lib/python3.X/site-packages/   # Python modules
│   ├── clio_cte/
│   └── clio_cee/
├── include/                       # C++ headers
│   ├── clio_run/
│   ├── hshm/
│   └── ...
└── lib/cmake/                     # CMake package configs
    ├── iowarp-core/
    ├── clio_run/
    ├── ClioCtp/
    └── ...
```

## Using IOWarp Core from Conda

After installation, you can use IOWarp Core in several ways:

### 1. Command-Line Tools

```bash
# Start the Clio runtime
clio_run runtime start

# Use CTE tools
clio_cte --help
```

### 2. Python

```python
import clio_cte
import clio_cee

# Use the Python bindings
```

### 3. C++ Development

```cmake
# In your CMakeLists.txt
find_package(clio-core REQUIRED)

target_link_libraries(your_app
    clio::run::admin_client
    clio::cte::core_client
)
```

## Directory Structure

```
installers/conda/
├── meta.yaml                 # Main recipe (conda-build format)
├── conda_build_config.yaml   # Variant / pin definitions (e.g. python)
├── conda-forge.yml           # conda-forge feedstock configuration
├── build.sh                  # Recipe build script (runs cmake via IOWARP_PRESET)
├── variants/                 # Preset variant configs
└── README.md
```

## conda-forge Submission

To submit to conda-forge:

1. Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
2. Copy `meta.yaml` and `conda_build_config.yaml` to `recipes/iowarp-core/`
3. Submit a pull request

The CI will automatically build all variant combinations defined in
`conda_build_config.yaml`.

## Troubleshooting

### Build fails with missing dependencies

Ensure the conda-forge channel is configured:

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

GPU builds are only supported on Linux. Check that you have the appropriate
toolkit installed:

```bash
# For CUDA
conda install cuda-toolkit -c nvidia

# For ROCm — follow AMD's ROCm installation guide
```

## More Information

- Main README: `../../README.md`
- Build wheel guide: `../../BUILD_WHEEL.md`
- Contributing guide: `../../docs/contributing.md`
- CMake presets: `../../CMakePresets.json`
</content>
</invoke>
