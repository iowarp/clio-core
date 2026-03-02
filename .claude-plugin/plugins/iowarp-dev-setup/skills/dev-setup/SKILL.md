---
description: "Set up, install, build, and debug the IOWarp Core development environment. Covers Docker devcontainers, native builds, IDE configuration, and troubleshooting."
---

# IOWarp Core — Development Environment Setup

You are helping a developer get IOWarp Core running locally. Follow the instructions below precisely. Adapt to the developer's OS and tooling, but always prefer the Docker devcontainer path unless they explicitly ask otherwise.

## Decision Tree

Ask the developer which setup path they need:
1. **Docker DevContainer** (recommended) — fully self-contained, works on any OS
2. **Native Linux** — for bare-metal or VM development
3. **Troubleshooting** — fix an existing broken setup

---

## Path 1: Docker DevContainer (Recommended)

### Prerequisites

The developer needs:
- **Docker Desktop** (Windows/macOS) or **Docker Engine** (Linux) — version 24+
- **VS Code** or **Cursor** with the "Dev Containers" extension (`ms-vscode-remote.remote-containers`)
- **Git** with submodule support
- At least **8 GB RAM** allocated to Docker (16 GB recommended)
- At least **15 GB disk** for the container image

For GPU development, they additionally need:
- NVIDIA drivers v525+ on the host
- NVIDIA Container Toolkit (`nvidia-ctk`)

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core
```

If already cloned without submodules:
```bash
git submodule update --init --recursive
```

#### 2. Open in IDE

**VS Code / Cursor:**
1. Open the `clio-core` folder
2. When prompted "Reopen in Container", click yes
3. Select **"IOWarp Core (CPU-only)"** unless GPU development is needed
4. Wait for the container build (first time: ~5-10 min CPU, ~15-20 min GPU)

**Manual container start** (headless / CLI-only):
```bash
cd .devcontainer/cpu
docker build --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g) \
  -t iowarp/clio-core-devcontainer:latest -f Dockerfile ../..
docker run -it --privileged \
  -v $(pwd)/../..:/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  iowarp/clio-core-devcontainer:latest
```

#### 3. Build IOWarp Core

Inside the container:
```bash
cmake --preset=debug
cmake --build build -j$(nproc)
```

For GPU builds (nvidia-gpu container only):
```bash
cmake --preset=cuda-debug
cmake --build build -j$(nproc)
```

#### 4. Run Tests

```bash
cd build && ctest -VV
```

Component-specific tests:
```bash
ctest -R context_transport   # Transport primitives
ctest -R chimaera            # Runtime
ctest -R cte                 # Context Transfer Engine
ctest -R omni                # Context Assimilation Engine
```

#### 5. Verify the Runtime

```bash
# Start runtime with default config
export CHI_SERVER_CONF=/workspace/docker/wrp_cte_bench/cte_config.yaml
chimaera runtime start &

# After a moment, run a quick benchmark
wrp_run_thrpt_benchmark --test-case latency --threads 4 --duration 5
```

### Container Architecture

The devcontainer uses a layered Docker image:
```
iowarp/iowarp-base:latest     ← Ubuntu 24.04 + base tools + iowarp user
  └─ iowarp/deps-cpu:latest   ← All build deps (apt + source-built)
       └─ devcontainer         ← Claude Code, UID remapping, SSH/Claude config forwarding
```

**What's pre-installed in the container:**
- Build: cmake, ninja, g++, ccache, patchelf
- Core deps: boost, hdf5, yaml-cpp (0.8.0), zeromq (4.3.5), cereal (1.3.2), catch2
- Compression: zlib, bzip2, lzo, zstd, lz4, xz, brotli, snappy, c-blosc2, zfp, sz3, fpzip, libpressio
- Network: OpenMPI, ADIOS2 (v2.11.0)
- Languages: Python 3 + venv, Rust, Node.js 22, Bun
- Tools: Docker-in-Docker, Jarvis-CD, Claude Code

**Key paths inside the container:**
| Path | Purpose |
|------|---------|
| `/workspace` | Bind-mounted source tree |
| `/workspace/build` | Build output directory (NEVER build elsewhere) |
| `/home/iowarp/venv` | Python virtual environment (auto-activated) |
| `/usr/local` | Source-built deps (yaml-cpp, zmq, cereal, etc.) |
| `/usr` | Apt-installed deps (boost, hdf5, compression libs) |

### Host Config Forwarding

The devcontainer automatically forwards from the host:
- `~/.ssh` → SSH keys for git operations
- `~/.claude` → Claude Code configuration and auth
- `~/.claude.json` → Claude Code credentials
- `/var/run/docker.sock` → Docker-in-Docker access

### Container Variants

| Container | Location | Use Case | Build Time | Size |
|-----------|----------|----------|------------|------|
| CPU-only | `.devcontainer/cpu/` | General development | ~5-10 min | ~3 GB |
| NVIDIA GPU | `.devcontainer/nvidia-gpu/` | CUDA kernel dev | ~15-20 min | ~8 GB |

---

## Path 2: Native Linux Build

### Using install.sh (Recommended)

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core
bash install.sh release
```

This installs Miniconda, rattler-build, and IOWarp in one script. Variants are stored in `installers/conda/variants/` — create a new one for custom machine configs.

### Manual Build

Install dependencies (Ubuntu 24.04):
```bash
sudo apt-get install -y cmake ninja-build g++ pkg-config \
  libboost-all-dev libhdf5-dev libyaml-cpp-dev libzmq3-dev \
  libcereal-dev catch2 libaio-dev liburing-dev \
  zlib1g-dev libbz2-dev liblzo2-dev libzstd-dev liblz4-dev \
  liblzma-dev libbrotli-dev libsnappy-dev libblosc2-dev
```

Build:
```bash
cmake --preset=debug
cmake --build build -j$(nproc)
```

Install (requires sudo):
```bash
sudo cmake --install build
```

---

## Path 3: Troubleshooting

### Build Failures

**"CMake can't find yaml-cpp/zeromq/cereal":**
- In container: deps are pre-installed. Run `ldconfig` and retry.
- Native: ensure `/usr/local/lib` is in `LD_LIBRARY_PATH`.

**"Build artifacts in source tree":**
```bash
# CRITICAL: Never build in the source tree. Clean it:
find . -name "CMakeCache.txt" -delete
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "Makefile" -delete
# Then rebuild properly:
cmake --preset=debug
cmake --build build -j$(nproc)
```

**Compilation errors after code changes:**
```bash
# Always reinstall before testing (RPATHs, not LD_LIBRARY_PATH):
cd build && sudo cmake --install . && ctest -VV
```

### Container Issues

**Container fails to start:**
1. Check Docker is running: `docker info`
2. Check image exists: `docker images | grep iowarp`
3. Rebuild: VS Code → Ctrl+Shift+P → "Dev Containers: Rebuild Container"

**Permission denied on files:**
- The container remaps UID/GID to match the host user
- If mismatched: `docker build --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g) ...`

**Docker-in-Docker not working:**
```bash
sudo chmod 666 /var/run/docker.sock
```

**SSH keys not available:**
- Ensure `~/.ssh` exists on the host before opening the container
- The `initializeCommand` creates it if missing, but keys must exist

### GPU Container Issues

**"nvidia-smi: command not found" inside container:**
1. Install NVIDIA drivers on the host
2. Install NVIDIA Container Toolkit: `.devcontainer/install-nvidia-container-toolkit.sh`
3. Restart Docker: `sudo systemctl restart docker`
4. Verify: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi`

**CUDA not detected by CMake:**
```bash
echo $CUDA_HOME   # Should be /usr/local/cuda-12.6
nvcc --version     # Should show CUDA 12.6
cmake --preset=cuda-debug
```

### Runtime Issues

**Shared memory errors / stale IPC segments:**
```bash
# Chimaera auto-cleans on init, but for manual cleanup:
rm -rf /tmp/chimaera_$(whoami)/*
```

**Runtime won't start:**
- Check config path: `echo $CHI_SERVER_CONF`
- Validate YAML syntax of the config file
- Check port availability: `lsof -i :9413`

### CMake Preset Reference

| Preset | Use |
|--------|-----|
| `debug` | Standard CPU debug (all features) |
| `cuda-debug` | CUDA support (arch 86) |
| `rocm-debug` | AMD ROCm GPUs |
| `debug-adios` | ADIOS2 integration |

Key CMake flags:
```
-DWRP_CORE_ENABLE_RUNTIME=ON    # Chimaera runtime
-DWRP_CORE_ENABLE_CTE=ON        # Context Transfer Engine
-DWRP_CORE_ENABLE_CAE=ON        # Context Assimilation Engine
-DWRP_CORE_ENABLE_CEE=ON        # Context Exploration Engine
-DWRP_CORE_ENABLE_TESTS=ON      # Enable test targets
-DWRP_CORE_ENABLE_COMPRESS=ON   # Compression support
-DWRP_CORE_ENABLE_PYTHON=ON     # Python bindings
-DWRP_CORE_ENABLE_ASAN=ON       # AddressSanitizer
```

---

## Quick Reference Card

```
# Clone
git clone --recurse-submodules https://github.com/iowarp/clio-core.git

# Build (inside container)
cmake --preset=debug && cmake --build build -j$(nproc)

# Test
cd build && ctest -VV

# Start runtime
export CHI_SERVER_CONF=/workspace/docker/wrp_cte_bench/cte_config.yaml
chimaera runtime start

# IPC transport modes
export CHI_IPC_MODE=SHM   # Shared memory (lowest latency, same machine)
export CHI_IPC_MODE=TCP   # TCP via ZeroMQ (default, cross-machine)
export CHI_IPC_MODE=IPC   # Unix domain socket (same machine, no TCP overhead)
```
