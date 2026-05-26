# Installing CLIO Core from Source

This guide is for building **CLIO Core from source on a bare-metal Linux
host**. It enumerates every system dependency, when each is required,
and the `apt` / `dnf` package names that satisfy them.

> **If you do not need to extend the C++ side or enable hardware-specific
> features, install a pre-built binary instead — no system packages are
> required.**
>
> | Channel  | Command                                  | Includes                                                                 |
> | -------- | ---------------------------------------- | ------------------------------------------------------------------------ |
> | pip      | `pip install iowarp-core`                | CLIO runtime, CTE, CAE, CEE; `clio_run` / `clio_cte_bench` CLIs; Python `clio_cee` |
> | conda    | `bash install.sh release` (uses Miniconda) | Same as pip; pulls deps from conda-forge instead of statically linking |
> | release  | [GitHub Releases](https://github.com/iowarp/clio-core/releases) | `.deb`, `.rpm`, `.AppImage` per tag (currently v1.9.1+) |
>
> Source builds are necessary only when you want CUDA / ROCm / SYCL,
> MPI, HDF5 / ADIOS2 / FUSE adapters, sanitizer builds, or custom
> ChiMods written against the C++ headers.

---

## 1. Always-required system prerequisites

These are needed for *every* source build, regardless of which optional
features you enable.

| Need              | Why                                                 | apt (Ubuntu 22.04+)                                            | dnf (Fedora 40+ / RHEL 9+)                                   |
| ----------------- | --------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------ |
| C++20 compiler    | GCC ≥ 11 or Clang ≥ 14                              | `build-essential` (GCC 13 on 24.04)                            | `gcc-c++` (GCC 14 on Fedora 40)                              |
| CMake ≥ 3.20      | Build system                                        | `cmake`                                                        | `cmake`                                                      |
| Ninja or Make     | Build driver (Ninja recommended)                    | `ninja-build`                                                  | `ninja-build`                                                |
| pkg-config        | Some deps published via pkg-config                  | `pkg-config`                                                   | `pkgconf-pkg-config`                                         |
| Git + submodules  | Source + vendored deps                              | `git`                                                          | `git`                                                        |
| Python ≥ 3.10     | Build scripts, Python bindings (if enabled)        | `python3 python3-pip python3-venv`                             | `python3 python3-pip`                                        |
| yaml-cpp ≥ 0.7    | Config-file parser                                  | `libyaml-cpp-dev`                                              | `yaml-cpp-devel`                                             |
| Cereal            | C++ serialization (header-only)                     | `libcereal-dev`                                                | `cereal-devel`                                               |
| msgpack-c ≥ 5     | Wire serialization                                  | `libmsgpack-dev`                                               | `msgpack-devel`                                              |
| libsodium         | Crypto primitives used by ZeroMQ                    | `libsodium-dev`                                                | `libsodium-devel`                                            |
| ZeroMQ ≥ 4.3      | RPC transport (default; can be disabled)            | `libzmq3-dev`                                                  | `zeromq-devel`                                               |

### Quick install — apt (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install -y --no-install-recommends \
  build-essential cmake ninja-build pkg-config git \
  python3 python3-pip python3-venv \
  libyaml-cpp-dev libcereal-dev libmsgpack-dev libsodium-dev libzmq3-dev
```

### Quick install — dnf (Fedora / RHEL / Rocky)

```bash
sudo dnf install -y \
  gcc-c++ cmake ninja-build pkgconf-pkg-config git \
  python3 python3-pip \
  yaml-cpp-devel cereal-devel msgpack-devel libsodium-devel zeromq-devel
```

---

## 2. Optional features and their dependencies

Each feature is opt-in via a CMake `-D<FLAG>=ON` switch. Enabling a flag
without its dependency available will fail at configure time with a
`find_package` error pointing at the missing component.

### Networking + RPC

| Feature                       | CMake flag                       | apt                                | dnf                                | Notes                                                                |
| ----------------------------- | -------------------------------- | ---------------------------------- | ---------------------------------- | -------------------------------------------------------------------- |
| MPI (multi-node deployment)   | `CLIO_CORE_ENABLE_MPI=ON`        | `libopenmpi-dev openmpi-bin`       | `openmpi-devel`                    | Or MPICH (`libmpich-dev` / `mpich-devel`); auto-detected.            |
| libfabric (OFI transport)     | `CLIO_CORE_ENABLE_LIBFABRIC=ON`  | `libfabric-dev`                    | `libfabric-devel`                  | Alternative to ZeroMQ on RDMA-capable fabrics.                       |
| Thallium (Mochi RPC)          | `CLIO_CORE_ENABLE_THALLIUM=ON`   | *via Spack / conda-forge*          | *via Spack / conda-forge*          | Not packaged in distro repos; install via Spack or build from source.|
| io\_uring                     | `CLIO_CORE_ENABLE_IO_URING=ON`   | `liburing-dev`                     | `liburing-devel`                   | Linux 5.1+ only. Auto-fallback to libaio / pread+pwrite if disabled. |

### Compression (`CLIO_CTE_ENABLE_COMPRESS=ON`)

Pulls in every codec at once. Trim the list by editing
`context-transfer-engine/CMakeLists.txt` if you only need a subset.

| apt                                                                                                    | dnf                                                                                                |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `libzstd-dev liblz4-dev zlib1g-dev liblzma-dev libbrotli-dev libblosc2-dev`                            | `libzstd-devel lz4-devel zlib-devel xz-devel libbrotli-devel blosc2-devel`                         |

LibPressio is also linked in if found (build from source — not in
distro repos).

### Encryption

| Feature                   | CMake flag                      | apt           | dnf                |
| ------------------------- | ------------------------------- | ------------- | ------------------ |
| OpenSSL (libcrypto)       | `CLIO_CORE_ENABLE_ENCRYPT=ON`   | `libssl-dev`  | `openssl-devel`    |

### CTE Adapters

| Adapter                      | CMake flag                              | apt                                  | dnf                                | Notes                                                  |
| ---------------------------- | --------------------------------------- | ------------------------------------ | ---------------------------------- | ------------------------------------------------------ |
| POSIX file I/O               | `CLIO_CTE_ENABLE_POSIX_ADAPTER=ON`*     | *(always available)*                 | *(always available)*               | **Default ON.** No extra deps.                         |
| STDIO (fopen / fread / fwrite) | `CLIO_CTE_ENABLE_STDIO_ADAPTER=ON`    | *(always available)*                 | *(always available)*               | No extra deps.                                         |
| MPI-IO                       | `CLIO_CTE_ENABLE_MPIIO_ADAPTER=ON`      | `libopenmpi-dev`                     | `openmpi-devel`                    | Requires `CLIO_CORE_ENABLE_MPI=ON`.                    |
| HDF5 VFD                     | `CLIO_CTE_ENABLE_VFD=ON`                | `libhdf5-dev`                        | `hdf5-devel`                       | Adds CLIO as a custom VFD to HDF5 client apps.         |
| HDF5 VOL (≥ HDF5 2.0)        | `CLIO_CTE_ENABLE_HDF5_VOL=ON`           | *(HDF5 2.0 not in apt yet — source)* | *(HDF5 2.0 not in dnf yet — source)* | Higher-level than VFD; full virtual object layer.    |
| ADIOS2                       | `CLIO_CTE_ENABLE_ADIOS2_ADAPTER=ON`     | *via Spack / source*                 | *via Spack / source*               | Pulls in `ADIOS2::adios2` CMake target.                |
| FUSE3 (filesystem mount)     | `CLIO_CTE_ENABLE_FUSE_ADAPTER=ON`       | `libfuse3-dev fuse3`                 | `fuse3-devel fuse3`                | Mount CTE storage as a POSIX FS via FUSE.              |
| NVIDIA GDS (GPU Direct Storage) | `CLIO_CTE_ENABLE_NVIDIA_GDS_ADAPTER=ON` | *NVIDIA CUDA Toolkit*               | *NVIDIA CUDA Toolkit*              | Requires `CLIO_CORE_ENABLE_CUDA=ON` and a CUDA install. |

### GPU acceleration

| Feature                | CMake flag                           | Notes                                                                                                  |
| ---------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| NVIDIA CUDA            | `CLIO_CORE_ENABLE_CUDA=ON`           | NVIDIA HPC SDK or CUDA Toolkit 11.8+ installed (`nvcc` on PATH). Tested with sm_70+.                   |
| GPU runtime (CDP)      | `CLIO_CORE_ENABLE_GPU_RUNTIME=ON`    | Requires CUDA. Enables Dynamic Parallelism for kernel-side task launch.                                |
| AMD ROCm               | `CLIO_CORE_ENABLE_ROCM=ON`           | ROCm 5.5+ (`hipcc` on PATH; `find_package(HIP)` succeeds).                                              |
| Intel SYCL (oneAPI)    | `CLIO_CORE_ENABLE_SYCL=ON`           | oneAPI DPC++ (`icpx -fsycl`) installed. Targets Intel GPUs + SYCL-2020 CPU targets.                    |
| NVSHMEM                | `CLIO_CORE_ENABLE_NVSHMEM=ON`        | NVSHMEM library installed (NVIDIA HPC SDK or standalone). GPU-to-GPU direct messaging.                  |
| NIXL (NVIDIA xfer)     | `CLIO_CORE_ENABLE_NIXL=ON`           | NVIDIA Inference Transfer Library. Source-only.                                                         |

### Python bindings + bench tools

| Feature                | CMake flag                           | Notes                                                              |
| ---------------------- | ------------------------------------ | ------------------------------------------------------------------ |
| Python bindings        | `CLIO_CORE_ENABLE_PYTHON=ON`         | Builds `clio_cee` and `clio_cte_core_ext` Python modules. Requires Python 3.10+ headers (`python3-dev` / `python3-devel`). |
| OpenMP                 | `CLIO_CORE_ENABLE_OPENMP=ON`         | Some benchmark loops use OpenMP. Compiler-provided (no system pkg). |
| Redis bench            | `CLIO_CORE_ENABLE_REDIS=ON`          | Builds the head-to-head Redis comparison benchmark. `libhiredis-dev` / `hiredis-devel`. |
| LLM hooks (llama.cpp)  | `CLIO_CORE_ENABLE_LLAMA=ON`          | Integrates the bundled `external/llama.cpp` submodule.             |
| Globus transfer (CAE)  | `CAE_ENABLE_GLOBUS=ON`               | Globus toolkit installed and on PATH.                              |

### Testing + diagnostics

| Feature                  | CMake flag                       | Notes                                                                  |
| ------------------------ | -------------------------------- | ---------------------------------------------------------------------- |
| Unit + integration tests | `CLIO_CORE_ENABLE_TESTS=ON`      | Pulls in Catch2 via FetchContent (no system dep).                       |
| AddressSanitizer + LSan  | `CLIO_CORE_ENABLE_ASAN=ON`       | Compiler-provided (`-fsanitize=address`).                                |
| UBSan                    | `CLIO_CORE_ENABLE_UBSAN=ON`      | Compiler-provided.                                                       |
| MSan                     | `CLIO_CORE_ENABLE_MSAN=ON`       | Clang only; requires MSan-instrumented libc++.                          |
| Coverage (gcov / llvm-cov)| `CLIO_CORE_ENABLE_COVERAGE=ON`  | `lcov` (apt) / `lcov` (dnf) for HTML reports.                            |
| Doxygen                  | `CLIO_CORE_ENABLE_DOXYGEN=ON`    | `doxygen perl` (apt) / `doxygen perl` (dnf).                             |

---

## 3. Building from source

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core

# Pick a preset (CMakePresets.json lists all of them):
#   release           — default, no GPU
#   cuda-release      — CUDA enabled
#   rocm-debug        — ROCm + debug symbols
#   sycl-debug        — SYCL + debug
#   release-adapter   — release + ADIOS2 + HDF5 VOL + FUSE3
#   release-fuse      — release + just FUSE3
#   asan / ubsan / msan / sanitize — sanitizer builds
cmake --preset release
cmake --build build/release -j$(nproc)
sudo cmake --install build/release
```

The install seeds a default runtime config at `~/.clio/clio.yaml` and
exports CMake config files under `lib/cmake/iowarp-core/` so downstream
projects can `find_package(iowarp-core)`.

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## 4. Feature checklist

Tick what you need; copy the matching `-D` flags into your `cmake`
invocation (or add them to a custom preset under
`installers/conda/variants/` if you use the Conda flow).

```
Core components (all ON by default)
─────────────────────────────────
[x] CLIO_CORE_ENABLE_RUNTIME=ON       Runtime + scheduler
[x] CLIO_CORE_ENABLE_CTE=ON           Context Transfer Engine
[x] CLIO_CORE_ENABLE_CAE=ON           Context Assimilation Engine
[x] CLIO_CORE_ENABLE_CEE=ON           Context Exploration Engine

Transports (ZMQ ON by default)
─────────────────────────────────
[x] CLIO_CORE_ENABLE_ZMQ=ON           ZeroMQ
[ ] CLIO_CORE_ENABLE_MPI=ON           MPI (multi-node)
[ ] CLIO_CORE_ENABLE_LIBFABRIC=ON     libfabric (OFI)
[ ] CLIO_CORE_ENABLE_THALLIUM=ON      Thallium / Mochi
[ ] CLIO_CORE_ENABLE_IO_URING=ON      io_uring async I/O (Linux 5.1+)

GPU / accelerators
─────────────────────────────────
[ ] CLIO_CORE_ENABLE_CUDA=ON          NVIDIA CUDA
[ ] CLIO_CORE_ENABLE_GPU_RUNTIME=ON   CUDA Dynamic Parallelism
[ ] CLIO_CORE_ENABLE_ROCM=ON          AMD ROCm / HIP
[ ] CLIO_CORE_ENABLE_SYCL=ON          Intel oneAPI SYCL
[ ] CLIO_CORE_ENABLE_NVSHMEM=ON       NVSHMEM GPU-to-GPU
[ ] CLIO_CORE_ENABLE_NIXL=ON          NVIDIA NIXL

CTE adapters (POSIX ON by default)
─────────────────────────────────
[x] CLIO_CTE_ENABLE_POSIX_ADAPTER=ON
[ ] CLIO_CTE_ENABLE_STDIO_ADAPTER=ON
[ ] CLIO_CTE_ENABLE_MPIIO_ADAPTER=ON  needs MPI
[ ] CLIO_CTE_ENABLE_VFD=ON            HDF5 VFD
[ ] CLIO_CTE_ENABLE_HDF5_VOL=ON       HDF5 VOL (HDF5 2.0+)
[ ] CLIO_CTE_ENABLE_ADIOS2_ADAPTER=ON
[ ] CLIO_CTE_ENABLE_FUSE_ADAPTER=ON   FUSE3 mount
[ ] CLIO_CTE_ENABLE_NVIDIA_GDS_ADAPTER=ON  needs CUDA

Compression + crypto (all OFF by default)
─────────────────────────────────────────
[ ] CLIO_CTE_ENABLE_COMPRESS=ON       zstd + lz4 + zlib + xz + brotli + blosc2
[ ] CLIO_CORE_ENABLE_ENCRYPT=ON       OpenSSL libcrypto

Misc
─────────────────────────────────
[ ] CLIO_CORE_ENABLE_PYTHON=ON        Python bindings (clio_cee + clio_cte_core_ext)
[ ] CLIO_CORE_ENABLE_OPENMP=ON
[ ] CLIO_CORE_ENABLE_REDIS=ON         Redis comparison benchmark (hiredis)
[ ] CLIO_CORE_ENABLE_LLAMA=ON         llama.cpp LLM hooks
[ ] CAE_ENABLE_GLOBUS=ON              Globus transfers in CAE

Build profiles (mutually exclusive)
─────────────────────────────────
[ ] CMAKE_BUILD_TYPE=Release          default
[ ] CMAKE_BUILD_TYPE=Debug
[ ] CLIO_CORE_ENABLE_ASAN=ON
[ ] CLIO_CORE_ENABLE_UBSAN=ON
[ ] CLIO_CORE_ENABLE_MSAN=ON          Clang only
[ ] CLIO_CORE_ENABLE_COVERAGE=ON
```

---

## 5. Verifying the install

```bash
clio_run --help                          # should print CLIO Runtime usage
clio_run start                           # foreground runtime; Ctrl-C to stop
ctest --test-dir build/release           # if you built with CLIO_CORE_ENABLE_TESTS=ON
```

If `clio_run` is on `PATH`, the install succeeded. If the runtime exits
immediately with a config error, check `~/.clio/clio.yaml` (seeded
automatically at install time).

For container-based deployment, see
[`docker/quickstart/`](docker/quickstart/) for a single-node
docker-compose recipe.
