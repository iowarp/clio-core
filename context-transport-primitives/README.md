# Context Transport Primitives (CTP)

[![IoWarp](https://img.shields.io/badge/IoWarp-GitHub-blue.svg)](https://github.com/iowarp)
[![GRC](https://img.shields.io/badge/GRC-Website-blue.svg)](https://grc.iit.edu/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Compatible-green.svg)](https://developer.nvidia.com/cuda-zone)
[![ROCm](https://img.shields.io/badge/ROCm-Compatible-red.svg)](https://rocmdocs.amd.com/)

The foundational shared-memory layer for CLIO Core: data structures, allocators,
synchronization primitives, and a networking layer (ZeroMQ / libfabric /
Thallium) that are interoperable across host memory, CUDA, and ROCm.

CTP underpins the inter-process communication used by the [CLIO
runtime](../context-runtime), the [Context Transfer
Engine](../context-transfer-engine), the [Context Assimilation
Engine](../context-assimilation-engine), and the [Context Exploration
Engine](../context-exploration-engine).

## What's inside

- **IPC-safe containers** — `vector`, `list`, `unordered_map`, MPSC/SPSC ring
  queues, all usable across process boundaries via shared memory.
- **Allocators** — malloc, stack, heap, and GPU-aware allocators.
- **Synchronization** — locks, atomics, and thread-model abstractions safe to
  share between host and device.
- **Networking** — the `lightbeam` transport stack (ZMQ, libfabric, Thallium).
- **Compression + encryption utilities** — pluggable codecs and crypto
  primitives.
- **Introspection** — system topology / NUMA queries used by the runtime
  scheduler.

## Building

CTP is built as part of CLIO Core; you do not build it standalone:

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core
cmake --preset release
cmake --build build/release -j$(nproc)
```

GPU support is opt-in via the root CMake flags:

| Flag | Effect |
|---|---|
| `CLIO_CORE_ENABLE_CUDA=ON` | Build the CUDA variant of CTP (`ctp::cudacxx`). |
| `CLIO_CORE_ENABLE_ROCM=ON` | Build the ROCm variant of CTP (`ctp::rocmcxx_gpu`). |

See [INSTALL.md](../INSTALL.md) for the full per-feature dependency list.

## Linking against CTP

CTP is exported through the unified `clio-core` CMake package — `find_package`
it once and link the modular CTP targets you need:

```cmake
find_package(clio-core CONFIG REQUIRED)

target_link_libraries(your_target
  ctp::cxx              # Core library (required)
  ctp::configure        # YAML configuration parsing (prefer over yaml-cpp directly)
  ctp::serialize        # Object serialization (prefer over cereal directly)
  ctp::interceptor      # ELF interception for adapters
  ctp::lightbeam        # Network transport (ZMQ, libfabric, Thallium)
  ctp::thread_all       # Threading support (pthread, OpenMP)
  ctp::mpi              # MPI support (only where needed)
  ctp::compress         # Compression utilities
  ctp::encrypt          # Encryption utilities
)
```

**Guidelines:**

- Always link `ctp::cxx` as the base.
- Prefer `ctp::configure` / `ctp::serialize` over linking yaml-cpp / cereal
  directly — each modular target carries the compile definitions CTP needs.
- For GPU code, link `ctp::cudacxx` (CUDA) or `ctp::rocmcxx_gpu` (ROCm)
  instead of `ctp::cxx`.

## Headers

The canonical include root is `<clio_ctp/...>`:

```cpp
#include <clio_ctp/clio_ctp.h>                                // umbrella
#include <clio_ctp/data_structures/ipc/vector.h>
#include <clio_ctp/memory/allocator/stack_allocator.h>
#include <clio_ctp/thread/lock/mutex.h>
```

## Project structure

```
include/clio_ctp/
  data_structures/      IPC-safe containers, serialization
    ipc/                vector, list, unordered_map, ring queues
    internal/           implementation details
    serialization/      serialization utilities
  memory/
    allocator/          malloc, stack, heap, GPU allocators
    backend/            memory backend implementations
  thread/
    lock/               lock implementations
    thread_model/       thread model abstractions
  util/
    compress/           compression utilities
    encrypt/            crypto primitives
  lightbeam/            networking layer (ZMQ, libfabric, Thallium)
  types/                shared type definitions
  introspect/           system topology / NUMA introspection
src/                    core library implementation
```

## Tests

```bash
cd build/release
ctest -R ctp           # core CTP tests
ctest -R mpsc_ring     # MPSC queue tests
```

## License

BSD-3-Clause. See [LICENSE](../LICENSE).
