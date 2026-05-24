# CLIO Core

<p align="center">
  <strong>A Comprehensive Platform for Context Management in Scientific Computing</strong>
  <br />
  <br />
  <a href="#overview">Overview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#components">Components</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#contributing">Contributing</a>
</p>

---

[![Project Site](https://img.shields.io/badge/Project-Site-blue)](https://grc.iit.edu/research/projects/iowarp)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](LICENSE)
[![IoWarp](https://img.shields.io/badge/IoWarp-GitHub-blue.svg)](http://github.com/iowarp)
[![GRC](https://img.shields.io/badge/GRC-Website-blue.svg)](https://grc.iit.edu/)
[![codecov](https://codecov.io/gh/iowarp/clio-core/graph/badge.svg)](https://codecov.io/gh/iowarp/clio-core)

## Overview

**CLIO Core** is a unified framework that integrates multiple high-performance components for context management, data transfer, and scientific computing. Built with a modular architecture, CLIO Core enables developers to create efficient data processing pipelines for HPC, storage systems, and near-data computing applications.

CLIO Core provides:
- **High-Performance Context Management**: Efficient handling of computational contexts and data transformations
- **Heterogeneous-Aware I/O**: Multi-tiered, dynamic buffering for accelerated data access
- **Modular Runtime System**: Extensible architecture with dynamically loadable processing modules
- **Advanced Data Structures**: Shared memory compatible containers with GPU support (CUDA, ROCm)
- **Distributed Computing**: Seamless scaling from single node to cluster deployments

## Architecture

CLIO Core follows a layered architecture integrating five core components:

```
┌──────────────────────────────────────────────────────────────┐
│                      Applications                            │
│          (Scientific Workflows, HPC, Storage Systems)        │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────────────┐   ┌──────────────────┐   ┌────────────────┐
│   Context     │   │    Context       │   │   Context      │
│  Exploration  │   │  Assimilation    │   │   Transfer     │
│    Engine     │   │     Engine       │   │    Engine      │
└───────────────┘   └──────────────────┘   └────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────────────┐
                    │  Chimaera       │
                    │  Runtime        │
                    │  (Module System)│
                    └─────────────────┘
                              │
                ┌─────────────────────────┐
                │  Context Transport      │
                │  Primitives             │
                │  (Shared Memory & IPC)  │
                └─────────────────────────┘
```

## Installation

### Pip (recommended)

The pip wheel is the easiest way to get CLIO Core. It ships a
**portable, self-contained build** with all dependencies statically
linked. No system installs are required beyond glibc and Python 3.10+.

```bash
pip install iowarp-core
```

The wheel includes the Clio runtime, the `clio_run` CLI, the CTE,
CAE, and CEE engines, and the `clio_cee` Python bindings. A default
configuration is seeded at `~/.clio/clio.yaml` (legacy: `~/.chimaera/chimaera.yaml`) on first import.

Newer extensions and advanced/accelerated features are not in the
portable wheel — switch to a source build below if you need any of:

- NVIDIA GPU (CUDA) or AMD GPU (ROCm) acceleration
- MPI for distributed multi-node deployment
- HDF5 / ADIOS2 adapters
- FUSE adapter
- Compression backends (LibPressio, Blosc, etc.)
- Custom ChiMods built against the C++ headers
- Sanitizer or debug builds

### Source build (Conda)

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core
bash install.sh release
```

`release` corresponds to a variant stored under
`installers/conda/variants/`. Other variants (`cuda`, `rocm`, `mpi`,
`full`, `release-fuse`, `debug`, ...) enable the corresponding features.
Feel free to add a new variant for your specific machine there.

If you already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

Other source-install methods (Docker, Spack) are documented in
[docs/getting-started/installation](docs/docs/getting-started/installation.mdx).

## Components

CLIO Core consists of five integrated components, each with its own specialized functionality:

### 1. Context Transport Primitives
**Location:** [`context-transport-primitives/`](context-transport-primitives/)

High-performance shared memory library containing data structures and synchronization primitives compatible with shared memory, CUDA, and ROCm.

**Key Features:**
- Shared memory compatible data structures (vector, list, unordered_map, queues)
- GPU-aware allocators (CUDA, ROCm)
- Thread synchronization primitives
- Networking layer with ZMQ transport
- Compression and encryption utilities

**[Read more →](context-transport-primitives/README.md)**

### 2. Context Runtime
**Location:** [`context-runtime/`](context-runtime/)

High-performance modular runtime for scientific computing and storage systems with coroutine-based task execution.

**Key Features:**
- Ultra-high performance task execution (< 10μs latency)
- Modular Module system for dynamic extensibility
- Coroutine-aware synchronization (CoMutex, CoRwLock)
- Distributed architecture with shared memory IPC
- Built-in storage backends (RAM, file-based, custom block devices)

**[Read more →](context-runtime/README.md)**

### 3. Context Transfer Engine
**Location:** [`context-transfer-engine/`](context-transfer-engine/)

Heterogeneous-aware, multi-tiered, dynamic I/O buffering system designed to accelerate I/O for HPC and data-intensive workloads.

**Key Features:**
- Programmable buffering across memory/storage tiers
- Multiple I/O pathway adapters
- Integration with HPC runtimes and workflows
- Improved throughput, latency, and predictability

**[Read more →](context-transfer-engine/README.md)**

### 4. Context Assimilation Engine
**Location:** [`context-assimilation-engine/`](context-assimilation-engine/)

High-performance data ingestion and processing engine for heterogeneous storage systems and scientific workflows.

**Key Features:**
- OMNI format for YAML-based job orchestration
- MPI-based parallel data processing
- Binary format handlers (Parquet, CSV, custom formats)
- Repository and storage backend abstraction
- Integrity verification with hash validation

**[Read more →](context-assimilation-engine/README.md)**

### 5. Context Exploration Engine
**Location:** [`context-exploration-engine/`](context-exploration-engine/)

Interactive tools and interfaces for exploring scientific data contents and metadata.

**Key Features:**
- Model Context Protocol (MCP) for HDF5 data
- HDF Compass viewer (wxPython-4 based)
- Interactive data exploration interfaces
- Metadata browsing capabilities

**[Read more →](context-exploration-engine/README.md)**

## Quickstart

### Starting the Runtime

Installation seeds a default configuration at `~/.clio/clio.yaml` (legacy: `~/.chimaera/chimaera.yaml`), so
the runtime works out of the box:

```bash
# Foreground
clio_run start

# Background
clio_run start &
```

To override the configuration, point `CLIO_X` at your YAML file:

```bash
export CLIO_X=/path/to/my_config.yaml
clio_run start
```

(The legacy nested form `clio_run runtime start` still works for back-compat.)

### Context Exploration Engine Python Example

Here we show an example of how to use the context exploration engine to
bundle and retrieve data.

```python
import clio_cee as cee

# Create ContextInterface (handles runtime initialization internally)
ctx_interface = cee.ContextInterface()

# Assimilate a file into IOWarp storage
ctx = cee.AssimilationCtx(
    src="file::/path/to/data.bin",      # Source: local file
    dst="iowarp::my_dataset",            # Destination: IOWarp tag
    format="binary"                      # Format: binary, hdf5, etc.
)
result = ctx_interface.context_bundle([ctx])
print(f"Assimilation result: {result}")

# Query for blobs matching a pattern
blobs = ctx_interface.context_query(
    "my_dataset",    # Tag name
    ".*",            # Blob name regex (match all)
    0                # Flags
)
print(f"Found blobs: {blobs}")

# Retrieve blob data
packed_data = ctx_interface.context_retrieve(
    "my_dataset",    # Tag name
    ".*",            # Blob name regex
    0                # Flags
)
print(f"Retrieved {len(packed_data)} bytes")

# Cleanup when done
ctx_interface.context_destroy(["my_dataset"])
```

### Context Transfer Engine C++ Example

Here is an example of the context transfer engine's C++ API.

```cpp
#include <clio_cte/core/core_client.h>
#include <clio_runtime/ipc_manager.h>
#include <cstring>

int main() {
  // 1. Initialize the CTE client.  This auto-connects to the runtime
  //    started by `clio_run start` and creates the CTE pool on the first
  //    call (no separate CLIO_INIT / runtime-mode setup needed in the
  //    consumer process).  Storage targets are configured declaratively
  //    via the runtime's compose YAML — no RegisterTarget call needed
  //    here either.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return 1;
  auto *cte = CLIO_CTE_CLIENT;

  // 2. Get-or-create a named container for blobs.  The async APIs
  //    return a chi::Future immediately; Wait() blocks for completion.
  auto tag_future = cte->AsyncGetOrCreateTag("my_tag");
  tag_future.Wait();
  auto tag_id = tag_future->tag_id_;

  // 3. Stage blob data into a CTE-managed shm buffer and submit the
  //    PutBlob asynchronously.  Submit-then-Wait is the canonical
  //    pattern; multiple AsyncPutBlob calls can be in flight before
  //    the first Wait() to pipeline I/O.  The async signatures take a
  //    type-erased `ShmPtr<>` so we wrap `put_buf.shm_` (a typed
  //    `ShmPtr<char>`) in the void-typed view.
  constexpr size_t kSize = 4096;
  auto put_buf = CLIO_IPC->AllocateBuffer(kSize);
  std::memset(put_buf.ptr_, 'A', kSize);
  ctp::ipc::ShmPtr<> put_data(put_buf.shm_);
  auto put_future = cte->AsyncPutBlob(tag_id, "my_blob",
                                       /*offset=*/0, kSize,
                                       put_data);
  put_future.Wait();
  CLIO_IPC->FreeBuffer(put_buf);

  // 4. Pre-allocate the receive buffer in shm, fire an async GetBlob,
  //    then Wait — the buffer holds the blob data on return.
  auto get_buf = CLIO_IPC->AllocateBuffer(kSize);
  ctp::ipc::ShmPtr<> get_data(get_buf.shm_);
  auto get_future = cte->AsyncGetBlob(tag_id, "my_blob",
                                       /*offset=*/0, kSize,
                                       /*flags=*/0,
                                       get_data);
  get_future.Wait();
  // get_buf.ptr_ now holds the retrieved bytes.
  CLIO_IPC->FreeBuffer(get_buf);

  // 5. Clean up.
  cte->AsyncDelTag(tag_id).Wait();
  return 0;
}
```

**Build and Link:**
```cmake
# Unified package includes everything - ClioCtp, CLIO Runtime, and all ChiMods.
# `clio-core` is the canonical package name; the legacy `iowarp-core` spelling
# still works for backward-compat.
find_package(clio-core REQUIRED)

target_link_libraries(my_app
  clio::cte::core_client    # CTE client (for the example above)
  clio::run::admin_client   # Admin module (always available)
  clio::run::bdev_client    # Block-device module (always available)
)
```

**What `find_package(clio-core)` provides:**

*Core Components:*
- All `ctp::*` modular targets (cxx, configure, serialize, interceptor, lightbeam, thread_all, mpi, compress, encrypt)
- `clio::run::cxx` (core runtime library)
- Module build utilities

*Core ChiMods (Always Available):*
- `clio::run::admin_client`, `clio::run::admin_runtime`
- `clio::run::bdev_client`, `clio::run::bdev_runtime`

*Optional ChiMods (if enabled at build time):*
- `clio::cte::core_client`, `clio::cte::core_runtime` (Context Transfer Engine)
- `clio::cae::core_client`, `clio::cae::core_runtime` (Context Assimilation Engine)

The pre-`::` waypoint spellings (`clio_run::*`, `clio_cte::*`, `clio_cae::*`) and the historical install-time names (`chimaera::*`, `wrp_cte::*`, `wrp_cae::*`) remain available as backward-compat ALIAS targets.

## Testing

CLIO Core includes comprehensive test suites for each component:

```bash
# Run all unit tests
cd build
ctest -VV

# Run specific component tests
ctest -R context_transport  # Transport primitives tests
ctest -R chimaera           # Runtime tests
ctest -R cte                # Context transfer engine tests
ctest -R omni               # Context assimilation engine tests
```

## Benchmarking

CLIO Core includes performance benchmarks for measuring runtime and I/O throughput.

### Runtime Throughput Benchmark (clio_run_thrpt_bench)

Measures task throughput and latency for the Clio runtime.

```bash
clio_run_thrpt_bench [options]
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--test-case <case>` | bdev_io | Test case to run |
| `--threads <N>` | 4 | Number of client worker threads |
| `--duration <seconds>` | 10.0 | Duration to run benchmark |
| `--max-file-size <size>` | 1g | Maximum file size (supports k, m, g suffixes) |
| `--io-size <size>` | 4k | I/O size per operation |
| `--lane-policy <P>` | (from config) | Lane policy: map_by_pid_tid, round_robin, random |
| `--output-dir <dir>` | /tmp/clio_benchmark | Output directory for files |
| `--verbose, -v` | false | Enable verbose output |

**Test Cases:**
- `bdev_io` - Full I/O throughput (Allocate → Write → Free)
- `bdev_allocation` - Allocation-only throughput
- `bdev_task_alloc` - Task allocation/deletion overhead
- `latency` - Round-trip task latency

**Examples:**

```bash
# Full I/O benchmark with 8 threads for 30 seconds
clio_run_thrpt_bench --test-case bdev_io --threads 8 --duration 30

# Latency benchmark with verbose output
clio_run_thrpt_bench --test-case latency --threads 4 --verbose

# Large I/O with 1MB blocks
clio_run_thrpt_bench --test-case bdev_io --io-size 1m --threads 16
```

### CTE Benchmark (clio_cte_bench)

Measures Context Transfer Engine Put/Get performance.

```bash
clio_cte_bench [options]
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--op <Put\|Get\|PutGet>` | Put | Operation to benchmark (alias: `--test-case`) |
| `--threads <N>` | 1 | Number of worker threads |
| `--depth <N>` | 1 | Async requests in flight per thread |
| `--io-size <size>` | 1m | Bytes per op (supports k, m, g suffixes) |
| `--io-count <N>` | 1000 | Ops per thread (ignored when `--time-limit` is set) |
| `--max-total-blobs <N>` | 0 (unbounded) | Total distinct keys across all threads (split evenly) |
| `--time-limit <seconds>` | 0 (off) | Run for N seconds instead of `--io-count` ops |
| `--help, -h` | | Show usage and exit |

A legacy positional form is also accepted:
```bash
clio_cte_bench <test_case> <threads> <depth> <io_size> <io_count>
```

**Examples:**

```bash
# Put benchmark: 4 threads, 8 async depth, 1MB I/O, 200 ops/thread
clio_cte_bench --op Put --threads 4 --depth 8 --io-size 1m --io-count 200

# Get benchmark with a 30-second time limit and 4KB I/O
clio_cte_bench --op Get --threads 2 --depth 4 --io-size 4k --time-limit 30

# Combined Put/Get over a bounded keyspace of 1024 total blobs
clio_cte_bench --op PutGet --threads 8 --depth 16 --io-size 16m \
               --io-count 50 --max-total-blobs 1024
```

**Output Metrics:**
- Total execution time (ms)
- Per-thread bandwidth: min, max, avg (MB/s)
- Aggregate bandwidth across all threads

## Documentation

Comprehensive documentation is available for each component:

- **[AGENTS.md](AGENTS.md)**: Unified development guide and coding standards
- **[Context Transport Primitives](context-transport-primitives/README.md)**: Shared memory data structures
- **[Context Runtime](context-runtime/README.md)**: Modular runtime system and Module development
  - [MODULE_DEVELOPMENT_GUIDE.md](context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md): Complete Module development guide
- **[Context Transfer Engine](context-transfer-engine/README.md)**: I/O buffering and acceleration
  - [CTE API Documentation](context-transfer-engine/docs/cte/cte.md): Complete API reference
- **[Context Assimilation Engine](context-assimilation-engine/README.md)**: Data ingestion and processing
- **[Context Exploration Engine](context-exploration-engine/README.md)**: Interactive data exploration

## Use Cases

**Scientific Computing:**
- High-performance data processing pipelines
- Near-data computing for large datasets
- Custom storage engine development
- Computational workflows with context management

**Storage Systems:**
- Distributed file system backends
- Object storage implementations
- Multi-tiered cache and storage solutions
- High-throughput I/O buffering

**HPC and Data-Intensive Workloads:**
- Accelerated I/O for scientific applications
- Data ingestion and transformation pipelines
- Heterogeneous computing with GPU support
- Real-time streaming analytics

## Performance Characteristics

CLIO Core is designed for high-performance computing scenarios:

- **Task Latency**: < 10 microseconds for local task execution (Context Runtime)
- **Memory Bandwidth**: Up to 50 GB/s with RAM-based storage backends
- **Scalability**: Single node to multi-node cluster deployments
- **Concurrency**: Thousands of concurrent coroutine-based tasks
- **I/O Performance**: Native async I/O with multi-tiered buffering

## Contributing

We welcome contributions to the CLIO Core project!

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Follow** the coding standards in [AGENTS.md](AGENTS.md)
4. **Test** your changes: `ctest --test-dir build`
5. **Submit** a pull request

### Coding Standards

- Follow **Google C++ Style Guide**
- Use semantic naming for IDs and priorities
- Always create docstrings for new functions (Doxygen compatible)
- Add comprehensive unit tests for new functionality
- Never use mock/stub code unless explicitly required - implement real, working code

See [AGENTS.md](AGENTS.md) for complete coding standards and workflow guidelines.

## License

CLIO Core is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE) file for complete license text.

**Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology**

---

## Acknowledgements

CLIO Core is developed at the [GRC lab](https://grc.iit.edu/) at Illinois Institute of Technology as part of the IOWarp project. This work is supported by the National Science Foundation (NSF) and aims to advance next-generation scientific computing infrastructure.

**For more information:**
- IOWarp Project: https://grc.iit.edu/research/projects/iowarp
- IOWarp Organization: https://github.com/iowarp
- Documentation Hub: https://grc.iit.edu/docs/category/iowarp

---

<p align="center">
  Built with ❤️ by the GRC Lab at Illinois Institute of Technology
</p>
