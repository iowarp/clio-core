# IOWarp Core Development Containers

IOWarp Core provides two development container configurations to support different development scenarios:

## Available Configurations

### 1. CPU-Only Devcontainer (Recommended for Most Development)

**Location**: `.devcontainer/cpu/`

**Use this for**:
- General development and testing
- When you don't have NVIDIA GPU hardware
- Lighter weight container (faster to build and run)
- All CPU-based features and tests

**Includes**:
- Full Conda environment with all dependencies
- Build tools: CMake, Ninja, pkg-config
- Core libraries: Boost, HDF5, yaml-cpp, ZeroMQ, Cereal
- Compression libraries: zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, c-blosc2
- Lossy compression: ZFP, SZ3, FPZIP
- ADIOS2 with MPI support
- Testing frameworks: Catch2, pytest
- Docker-in-Docker support
- Jarvis runtime deployment tool

**Does NOT include**:
- CUDA toolkit
- NVIDIA Container Toolkit
- GPU-specific libraries

### 2. NVIDIA GPU Devcontainer

**Location**: `.devcontainer/nvidia-gpu/`

**Use this for**:
- GPU-accelerated development and testing
- CUDA kernel development
- Testing GPU-specific features
- When you have NVIDIA GPU hardware available

**Includes**:
- Everything from the CPU-only container
- CUDA Toolkit 12.6
- NVIDIA Container Toolkit
- GPU runtime libraries (cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NPP)
- NVIDIA Nsight VSCode extension

**Requirements**:
- NVIDIA GPU with compute capability 8.6+ (for CUDA 12.6)
- NVIDIA drivers installed on host
- NVIDIA Container Toolkit installed on host

## How to Use

### VS Code / Cursor

1. Open the IOWarp Core repository in VS Code or Cursor
2. When prompted, select **"Reopen in Container"**
3. Choose your desired configuration:
   - **IOWarp Core (CPU-only)** - For general development
   - **IOWarp Core (NVIDIA GPU)** - For GPU development

Alternatively:
1. Press `F1` or `Cmd/Ctrl+Shift+P`
2. Type "Dev Containers: Reopen in Container"
3. Select your preferred configuration

### Switching Containers

To switch between CPU and GPU containers:

1. Press `F1` or `Cmd/Ctrl+Shift+P`
2. Type "Dev Containers: Rebuild and Reopen in Container"
3. Select the other configuration

Your workspace files are preserved when switching containers.

## Building the Project

Once inside the devcontainer, build IOWarp Core:

```bash
# CPU-only build (works in both containers)
cmake --preset=debug
cmake --build build -j$(nproc)

# GPU build (only in nvidia-gpu container)
cmake --preset=cuda-debug
cmake --build build -j$(nproc)
```

## CMake Presets

- **debug**: Standard debug build (CPU-only)
- **release**: Optimized release build (CPU-only)
- **cuda-debug**: Debug build with CUDA support (requires nvidia-gpu container)
- **rocm-debug**: Debug build with ROCm support (for AMD GPUs)

## Conda Environment

Both containers include a fully-configured Conda environment with all dependencies. The base environment is automatically activated on container startup.

**Verify installation**:
```bash
conda list | grep -E "boost|hdf5|yaml-cpp|zeromq"
```

**Compression libraries**:
```bash
conda list | grep -E "zlib|bzip2|lzo|zstd|lz4|xz|brotli|snappy|blosc"
```

## GPU Setup (nvidia-gpu container only)

### Host Requirements

Before using the GPU container, ensure your host system has:

1. **NVIDIA drivers** (version 525+ for CUDA 12.6):
   ```bash
   nvidia-smi  # Should show GPU information
   ```

2. **NVIDIA Container Toolkit**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

   Or use the included helper script:
   ```bash
   .devcontainer/install-nvidia-container-toolkit.sh
   ```

### Verify GPU Access

Inside the GPU container:
```bash
# Check CUDA installation
nvcc --version

# Check GPU visibility
nvidia-smi

# Verify CUDA environment
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

### Troubleshooting GPU Container

If the GPU container fails to start:

1. **Check NVIDIA drivers on host**:
   ```bash
   nvidia-smi
   ```

2. **Verify container toolkit**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
   ```

3. **Check Docker GPU support**:
   ```bash
   docker info | grep -i nvidia
   ```

4. **Review CUDA setup guide**:
   ```bash
   cat .devcontainer/CUDA_SETUP.md
   ```

## Docker-in-Docker

Both containers support Docker-in-Docker, allowing you to build and run Docker containers from within the devcontainer. This is useful for:
- Testing containerized applications
- Building Docker images for deployment
- Running multi-container test environments

## SSH and Claude Configuration

Both containers automatically mount and configure:
- SSH keys from `~/.ssh` (read-only, copied to container)
- Claude configuration from `~/.claude` and `~/.claude.json`

These are set up during container creation via the `postCreateCommand`.

## Performance Considerations

### CPU Container
- Faster build times (~5-10 minutes)
- Lower resource usage
- Suitable for most development tasks
- Recommended as default

### GPU Container
- Longer build times (~15-20 minutes due to CUDA toolkit)
- Higher resource usage (CUDA toolkit is ~5GB)
- Required only for GPU-specific development
- Use only when needed

## Contributing

When contributing code:
1. Use the **CPU container** for general development and testing
2. Use the **GPU container** only when working on GPU-specific features
3. Ensure your changes work in both environments when applicable
4. Run tests in the appropriate container for your changes

## Additional Resources

- **Contributing Guide**: `docs/contributing.md`
- **CUDA Setup**: `.devcontainer/CUDA_SETUP.md`
- **Project Documentation**: `CLAUDE.md`
- **Module Development**: `docs/docs/sdk/context-runtime/2.module_dev_guide.md`

## Quick Reference

| Feature | CPU Container | GPU Container |
|---------|---------------|---------------|
| Conda dependencies | ✅ | ✅ |
| Compression libraries | ✅ (all) | ✅ (all) |
| ADIOS2 + MPI | ✅ | ✅ |
| Docker-in-Docker | ✅ | ✅ |
| CUDA Toolkit | ❌ | ✅ |
| GPU runtime libraries | ❌ | ✅ |
| Build time | ~5-10 min | ~15-20 min |
| Container size | ~3GB | ~8GB |
| Recommended for | Default dev | GPU dev only |
