# IOWarp Core DevContainer

This directory contains the DevContainer configurations for IOWarp Core development.

## 🚀 Quick Start

IOWarp Core provides **two development container options**:

1. **CPU-only container** (recommended for most development) - `.devcontainer/cpu/`
2. **NVIDIA GPU container** (for GPU development) - `.devcontainer/nvidia-gpu/`

**📖 See [README_DEVCONTAINERS.md](README_DEVCONTAINERS.md) for complete setup guide and configuration details.**

## Choosing Your Container

### Use the CPU-only Container For:
- General development and testing
- When you don't have NVIDIA GPU hardware
- Faster build times and lower resource usage
- All CPU-based features

### Use the NVIDIA GPU Container For:
- CUDA kernel development
- Testing GPU-specific features
- When you have NVIDIA GPU hardware available

## Quick Setup

1. **Open in VS Code/Cursor**: When prompted, select "Reopen in Container"
2. **Choose configuration**:
   - `IOWarp Core (CPU-only)` - Default recommendation
   - `IOWarp Core (NVIDIA GPU)` - For GPU development

## Features (Available in Both Containers)

### Conda Package Manager (Recommended)

The devcontainer includes **Miniconda** with conda-forge channel configured. Conda is the **recommended** package manager for IOWarp Core development.

**Conda setup:**
- **Location:** `/home/iowarp/miniconda3`
- **Channels:** conda-forge (priority: strict)
- **Pre-installed dependencies:** boost, hdf5, yaml-cpp, zeromq, cereal, catch2, cmake, ninja
- **Compression libraries:** zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, c-blosc2
- **Lossy compression:** ZFP (via conda), SZ3, FPZIP (built from source)
- **Auto-initialized:** Ready to use when container starts

**Quick start with Conda:**
```bash
# Conda is already initialized with all dependencies installed!

# Build IOWarp Core
cmake --preset=debug
cmake --build build -j$(nproc)

# Run tests
cd build && ctest

# For compression support
cmake --preset=debug -DCLIO_CTE_ENABLE_COMPRESS=ON
cmake --build build -j$(nproc)
```

**Verify conda installation:**
```bash
# List installed dependencies
conda list | grep -E "boost|hdf5|yaml-cpp|zeromq"

# Check compression libraries
conda list | grep -E "zlib|bzip2|lzo|zstd|lz4|xz|brotli|snappy|blosc"
```

**Common Conda commands:**
```bash
# List environments
conda env list

# Create new environment
conda create -n myenv python=3.11

# Activate environment
conda activate myenv

# Install additional packages
conda install -c conda-forge <package-name>
```

**Why use Conda for IOWarp Core?**
- CMake automatically detects and prioritizes Conda packages
- All dependencies available via conda-forge
- Reproducible environments across systems
- Better dependency management than system packages
- Prevents library version conflicts

### Python Virtual Environment (Alternative)

A Python virtual environment is also available at `/home/iowarp/venv` as an alternative to Conda.

**Pre-installed packages:**
- `pip`, `setuptools`, `wheel` (latest versions)
- `pyyaml` (for configuration file parsing)
- `nanobind` (for Python bindings)

**Note:** By default, Conda base environment is auto-activated. To use venv instead:
```bash
source /home/iowarp/venv/bin/activate
```

**Building Python bindings:**
```bash
# Configure with Python support
cmake --preset=debug -DCLIO_CORE_ENABLE_PYTHON=ON

# Build
cmake --build build -j$(nproc)

# Install to site-packages
cmake --install build --prefix $CONDA_PREFIX
```

### VSCode Extensions

The following extensions are automatically installed:

**C/C++ Development:**
- C/C++ (ms-vscode.cpptools)
- CMake Tools (ms-vscode.cmake-tools)
- CMake (twxs.cmake)
- C/C++ Debug (KylinIdeTeam.cppdebug)
- clangd (llvm-vs-code-extensions.vscode-clangd)

**Python Development:**
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Python Debugger (ms-python.debugpy)

**Container & DevOps:**
- Docker (ms-azuretools.vscode-docker)

**AI Assistant:**
- Claude Code (anthropic.claude-code)

**GPU Development (nvidia-gpu container only):**
- NVIDIA Nsight (nvidia.nsight-vscode-edition)

### Docker-in-Docker

Docker is available inside the container with the host's Docker socket mounted, allowing you to:
- Build and run containers from inside the devcontainer
- Use docker-compose
- Interact with the host's Docker daemon
- Test containerized deployments

### Environment Variables

**CPU Container:**
- `IOWARP_CORE_ROOT`: Set to the workspace folder
- `CONDA_DEFAULT_ENV`: Set to "base"
- `CONDA_AUTO_ACTIVATE_BASE`: Set to "true"
- `CONDA_PREFIX`: `/home/iowarp/miniconda3`
- `LD_LIBRARY_PATH`: Includes conda and system library paths

**GPU Container (additional):**
- `CUDA_HOME`: `/usr/local/cuda-12.6`
- `NVIDIA_VISIBLE_DEVICES`: `all`
- `NVIDIA_DRIVER_CAPABILITIES`: `compute,utility`
- `LD_LIBRARY_PATH`: Includes CUDA library paths

## CMake Build Presets

**CPU-only builds** (work in both containers):
- `debug` - Standard debug build with all features
- `release` - Optimized release build

**GPU builds** (require nvidia-gpu container):
- `cuda-debug` - Debug build with CUDA support
- `rocm-debug` - Debug build with ROCm support (AMD GPUs)

**Example:**
```bash
# CPU build
cmake --preset=debug
cmake --build build -j$(nproc)

# GPU build (nvidia-gpu container only)
cmake --preset=cuda-debug
cmake --build build -j$(nproc)
```

## Python Configuration

The VSCode Python extension is configured with:
- **Default interpreter:** `/home/iowarp/miniconda3/bin/python` (Conda base environment)
- **Conda path:** `/home/iowarp/miniconda3/bin/conda`
- **Auto-activate:** Terminal activation is enabled
- **Linting:** flake8 enabled (pylint disabled)
- **Formatting:** black (if installed)

**Switching Python interpreters in VSCode:**
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Python: Select Interpreter"
3. Choose between Conda environments or venv

## Switching Between Containers

To switch between CPU and GPU containers:

1. Press `F1` or `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Dev Containers: Rebuild and Reopen in Container"
3. Select the other configuration

Your workspace files are preserved when switching containers.

## Rebuilding the Container

If you modify the Dockerfile, rebuild the container:

1. In VSCode: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"
2. Or manually: `docker build .devcontainer/cpu/` or `docker build .devcontainer/nvidia-gpu/`

## GPU Container Setup

### Prerequisites for GPU Container

Before using the NVIDIA GPU container, ensure your host system has:

1. **NVIDIA drivers** (version 525+ for CUDA 12.6)
2. **NVIDIA Container Toolkit**

**Quick installation:**
```bash
# Run the included helper script
.devcontainer/install-nvidia-container-toolkit.sh
```

**Or manually:**
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

**📖 See [CUDA_SETUP.md](CUDA_SETUP.md) for detailed GPU setup instructions.**

## Troubleshooting

### Conda Issues

**Conda not initialized:**
```bash
# Initialize conda in current shell
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Or re-run conda init
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

**Conda packages not found:**
```bash
# Verify conda-forge channel is configured
conda config --show channels

# Add conda-forge if missing
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### Python Issues

**Wrong Python interpreter:**
```bash
# Check current Python
which python
python --version

# Should show /home/iowarp/miniconda3/bin/python

# Activate desired conda environment if needed
conda activate myenv
```

**Python packages not found:**
```bash
# For Conda:
which python  # Should show /home/iowarp/miniconda3/.../python
conda list    # Show installed packages

# For venv (if activated):
which python  # Should show /home/iowarp/venv/bin/python
pip list      # Show installed packages
```

### Docker Issues

**Docker permission issues:**
```bash
# The container should automatically fix this, but if issues persist:
sudo chmod 666 /var/run/docker.sock
```

### GPU Container Issues

**GPU container fails to start:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify container toolkit: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi`
3. Check Docker GPU support: `docker info | grep -i nvidia`
4. Review detailed GPU setup: `cat .devcontainer/CUDA_SETUP.md`

**CUDA not found in container:**
```bash
# Verify CUDA installation
nvcc --version

# Check environment variables
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Test GPU access
nvidia-smi
```

### Build Issues

**CMake can't find conda packages:**
```bash
# Verify Conda is activated
conda info --envs

# Check if conda path is in CMAKE_PREFIX_PATH
echo $CMAKE_PREFIX_PATH

# Configure with conda path explicitly
cmake --preset=debug -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
```

**Compression libraries not found:**
```bash
# Install c-blosc2 (if missing)
conda install c-blosc2

# Verify all compression libraries
conda list | grep -E "zlib|bzip2|lzo|zstd|lz4|xz|brotli|snappy|blosc"

# Rebuild with compression enabled
cmake --preset=debug -DCLIO_CTE_ENABLE_COMPRESS=ON
```

## Additional Resources

- **Complete DevContainer Guide**: [README_DEVCONTAINERS.md](README_DEVCONTAINERS.md)
- **CUDA Setup**: [CUDA_SETUP.md](CUDA_SETUP.md)
- **Project Documentation**: [../CLAUDE.md](../CLAUDE.md)
- **Contributing Guide**: [../docs/contributing.md](../docs/contributing.md)
- **Module Development**: [../context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md](../context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md)

## Quick Reference

| Feature | CPU Container | GPU Container |
|---------|---------------|---------------|
| **Location** | `.devcontainer/cpu/` | `.devcontainer/nvidia-gpu/` |
| **Recommended for** | Default dev | GPU dev only |
| **Build time** | ~5-10 min | ~15-20 min |
| **Container size** | ~3GB | ~8GB |
| **Conda dependencies** | ✅ All | ✅ All |
| **Compression libs** | ✅ All | ✅ All |
| **ADIOS2 + MPI** | ✅ | ✅ |
| **Docker-in-Docker** | ✅ | ✅ |
| **CUDA Toolkit** | ❌ | ✅ |
| **GPU libraries** | ❌ | ✅ |
