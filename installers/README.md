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

### 3. Spack Package (`spack/`)

A source build driven by Spack. Every optional feature of the CMake build is
a variant, including the ones no binary installer ships (GPU memory
expansion, the I/O adapters, clio-ml).

**Quick Start:**
```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
spack repo add clio-core/installers/spack
spack install iowarp +fuse                       # CPU build with the FUSE mount
spack install iowarp +cuda cuda_arch=90 +gpu_vector +clio_ml   # NVIDIA
spack install iowarp +sycl +clio_ml %oneapi      # Intel GPUs
```

| Feature | Variant |
|---|---|
| FUSE mount (Context FS) | `+fuse` |
| Dashboard (Poco) | `+viz` (default on) |
| Python bindings | `+python` |
| POSIX / STDIO / MPI-IO interceptors | `+posix +stdio +mpiio` (need `+elf`; default on) |
| HDF5 VOL connector / VFD | `+vol` / `+vfd` (HDF5 1.14+; default on) |
| ADIOS2 engine | `+adios2` |
| NVIDIA GPUDirect Storage adapter | `+gds` (needs `+cuda +elf`) |
| CUDA, with `gpu_vector` | `+cuda cuda_arch=<sm> +gpu_vector` (clang is the CUDA compiler) |
| ROCm | `+rocm amdgpu_target=<gfx>` |
| SYCL, with `gpu_vector` | `+sycl %oneapi` (the compiler last: variants after `%oneapi` apply to it) |
| clio-ml | `+clio_ml` (backend follows `+cuda` / `+sycl`) |
| Compression | `+compress` |
| S3 / GCS / DAOS storage tiers | `+s3_bdev` / `+gcs_bdev` / `+daos` |
| S3 / GCS / Globus import (CAE) | `+s3_cae` / `+gcs` / `+globus` |
| Transports | `+mochi`, `+zmq` (default on) |
| io_uring, OpenMP | `+io_uring` (default on), `+openmp` |

`spack info iowarp` lists every variant. `.github/workflows/spack.yml`
installs each feature group (CPU, cloud and network, CUDA, ROCm, SYCL) from
the checkout on every change to the package, and checks the feature's files
were installed. NIXL and NVSHMEM have no variant yet: they need vendor
installs that Spack does not package.

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
