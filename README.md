# IOWarp Core

<p align="center">
  <strong>A Comprehensive Platform for Context Management in Scientific Computing</strong>
  <br />
  <br />
  <a href="#overview">Overview</a> ·
  <a href="#components">Components</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#contributing">Contributing</a>
</p>

---

[![Project Site](https://img.shields.io/badge/Project-Site-blue)](https://grc.iit.edu/research/projects/iowarp)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](LICENSE)
[![IoWarp](https://img.shields.io/badge/IoWarp-GitHub-blue.svg)](http://github.com/iowarp)
[![GRC](https://img.shields.io/badge/GRC-Website-blue.svg)](https://grc.iit.edu/)

## Overview

**IOWarp Core** is a unified framework that integrates multiple high-performance components for context management, data transfer, and scientific computing. Built with a modular architecture, IOWarp Core enables developers to create efficient data processing pipelines for HPC, storage systems, and near-data computing applications.

IOWarp Core provides:
- **High-Performance Context Management**: Efficient handling of computational contexts and data transformations
- **Heterogeneous-Aware I/O**: Multi-tiered, dynamic buffering for accelerated data access
- **Modular Runtime System**: Extensible architecture with dynamically loadable processing modules
- **Advanced Data Structures**: Shared memory compatible containers with GPU support (CUDA, ROCm)
- **Distributed Computing**: Seamless scaling from single node to cluster deployments

## Architecture

IOWarp Core follows a layered architecture integrating five core components:

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
                    │  (ChiMod System)│
                    └─────────────────┘
                              │
                ┌─────────────────────────┐
                │  Context Transport      │
                │  Primitives             │
                │  (Shared Memory & IPC)  │
                └─────────────────────────┘
```

## Components

IOWarp Core consists of five integrated components, each with its own specialized functionality:

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

### 2. Chimaera Runtime
**Location:** [`context-runtime/`](context-runtime/)

High-performance modular runtime for scientific computing and storage systems with coroutine-based task execution.

**Key Features:**
- Ultra-high performance task execution (< 10μs latency)
- Modular ChiMod system for dynamic extensibility
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

## Installation

### Native

The following command will install conda, rattler-build, and iowarp in a single script.
```bash
bash install.sh release
```

Release corresponds to a variant stored in installers/conda/variants.
Feel free to add a new variant for your specific machine there.

## Quickstart

### Starting the Runtime

Before running our code, start the Chimaera runtime:

```bash
# Start with custom configuration
export CHI_SERVER_CONF=/workspace/docker/wrp_cte_bench/cte_config.yaml
chimaera_start_runtime

# Run in background
chimaera_start_runtime &
```

**Environment Variables:**
| Variable | Description |
|----------|-------------|
| `CHI_SERVER_CONF` | Primary path to Chimaera configuration file (checked first) |
| `WRP_RUNTIME_CONF` | Fallback configuration path (used if CHI_SERVER_CONF not set) |

### Chimaera Configuration

Configuration uses YAML format. Example configuration:

```yaml
# Memory segment configuration
memory:
  main_segment_size: 1073741824           # 1GB main segment
  client_data_segment_size: 536870912     # 512MB client data
  runtime_data_segment_size: 536870912    # 512MB runtime data

# Network configuration
networking:
  port: 5555                              # ZeroMQ port
  neighborhood_size: 32                   # Max nodes for range queries

# Runtime configuration
runtime:
  sched_threads: 4                        # Scheduler worker threads
  slow_threads: 0                         # Slow worker threads (long tasks)
  stack_size: 65536                       # 64KB per task
  queue_depth: 10000                      # Maximum queue depth
  local_sched: "default"                  # Local task scheduler (default: "default")

# Compose section for declarative pool creation
compose:
  - mod_name: wrp_cte_core
    pool_name: wrp_cte
    pool_query: local
    pool_id: 512.0

    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000

    storage:
      - path: "ram::cte_storage"          # RAM-based storage
        bdev_type: "ram"
        capacity_limit: "16GB"
        score: 1.0                        # Higher = faster tier (0.0-1.0)

    dpe:
      dpe_type: "max_bw"                  # Options: random, round_robin, max_bw
```

### Context Exploration Engine Python Example

Here we show an example of how to use the context exploration engine to
bundle and retrieve data.

```python
import wrp_cee as cee

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
#include <wrp_cte/core/core_client.h>
#include <chimaera/chimaera.h>

int main() {
  // 1. Initialize Chimaera runtime
  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!success) return 1;

  // 2. Initialize CTE subsystem
  wrp_cte::core::WRP_CTE_CLIENT_INIT();

  // 3. Create CTE client
  wrp_cte::core::Client cte_client;
  wrp_cte::core::CreateParams params;
  cte_client.Create(chi::PoolQuery::Dynamic(),
                    wrp_cte::core::kCtePoolName,
                    wrp_cte::core::kCtePoolId, params);

  // 4. Register a storage target (100MB file-based)
  cte_client.RegisterTarget("/tmp/cte_storage",
                            chimaera::bdev::BdevType::kFile,
                            100 * 1024 * 1024);

  // 5. Create a tag (container for blobs)
  wrp_cte::core::TagId tag_id = cte_client.GetOrCreateTag(
      "my_tag", wrp_cte::core::TagId::GetNull());

  // 6. Store blob data
  std::vector<char> data(4096, 'A');
  hipc::FullPtr<char> shared_data = CHI_IPC->AllocateBuffer(data.size());
  memcpy(shared_data.ptr_, data.data(), data.size());

  cte_client.PutBlob(tag_id, "my_blob",
                     0,                    // offset
                     data.size(),          // size
                     shared_data.shm_,     // shared memory pointer
                     0.8f,                 // importance score
                     0);                   // flags
  CHI_IPC->FreeBuffer(shared_data);

  // 7. Retrieve blob data
  hipc::FullPtr<char> read_buf = CHI_IPC->AllocateBuffer(data.size());
  cte_client.GetBlob(tag_id, "my_blob",
                     0,                    // offset
                     data.size(),          // size
                     0,                    // flags
                     read_buf.shm_);
  // read_buf.ptr_ now contains the retrieved data
  CHI_IPC->FreeBuffer(read_buf);

  // 8. Cleanup
  cte_client.DelTag(tag_id);
  return 0;
}
```

**Build and Link:**
```cmake
# Unified package includes everything - HermesShm, Chimaera, and all ChiMods
find_package(iowarp-core REQUIRED)

target_link_libraries(my_app
  wrp_cte::core_client    # CTE client (for the example above)
  chimaera::admin_client  # Admin ChiMod (always available)
  chimaera::bdev_client   # Block device ChiMod (always available)
)
```

**What `find_package(iowarp-core)` provides:**

*Core Components:*
- All `hshm::*` modular targets (cxx, configure, serialize, interceptor, lightbeam, thread_all, mpi, compress, encrypt)
- `chimaera::cxx` (core runtime library)
- ChiMod build utilities

*Core ChiMods (Always Available):*
- `chimaera::admin_client`, `chimaera::admin_runtime`
- `chimaera::bdev_client`, `chimaera::bdev_runtime`

*Optional ChiMods (if enabled at build time):*
- `wrp_cte::core_client`, `wrp_cte::core_runtime` (Context Transfer Engine)
- `wrp_cae::core_client`, `wrp_cae::core_runtime` (Context Assimilation Engine)

## Testing

IOWarp Core includes comprehensive test suites for each component:

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

IOWarp Core includes performance benchmarks for measuring runtime and I/O throughput.

### Runtime Throughput Benchmark (wrp_run_thrpt_benchmark)

Measures task throughput and latency for the Chimaera runtime.

```bash
wrp_run_thrpt_benchmark [options]
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
| `--output-dir <dir>` | /tmp/wrp_benchmark | Output directory for files |
| `--verbose, -v` | false | Enable verbose output |

**Test Cases:**
- `bdev_io` - Full I/O throughput (Allocate → Write → Free)
- `bdev_allocation` - Allocation-only throughput
- `bdev_task_alloc` - Task allocation/deletion overhead
- `latency` - Round-trip task latency

**Examples:**

```bash
# Full I/O benchmark with 8 threads for 30 seconds
wrp_run_thrpt_benchmark --test-case bdev_io --threads 8 --duration 30

# Latency benchmark with verbose output
wrp_run_thrpt_benchmark --test-case latency --threads 4 --verbose

# Large I/O with 1MB blocks
wrp_run_thrpt_benchmark --test-case bdev_io --io-size 1m --threads 16
```

### CTE Benchmark (wrp_cte_bench)

Measures Context Transfer Engine Put/Get performance.

```bash
wrp_cte_bench <test_case> <num_threads> <depth> <io_size> <io_count>
```

**Parameters:**

| Parameter | Position | Description |
|-----------|----------|-------------|
| `test_case` | 1 | Put, Get, or PutGet |
| `num_threads` | 2 | Number of worker threads |
| `depth` | 3 | Number of async requests per thread |
| `io_size` | 4 | Size per operation (supports k, m, g suffixes) |
| `io_count` | 5 | Number of operations per thread |

**Examples:**

```bash
# Put benchmark: 4 threads, 8 async depth, 1MB I/O, 200 operations each
wrp_cte_bench Put 4 8 1m 200

# Get benchmark: 2 threads, 4 async depth, 4KB I/O, 1000 operations each
wrp_cte_bench Get 2 4 4k 1000

# Combined Put/Get: 8 threads, 16 async depth, 16MB I/O, 50 operations each
wrp_cte_bench PutGet 8 16 16m 50
```

**Output Metrics:**
- Total execution time (ms)
- Per-thread bandwidth: min, max, avg (MB/s)
- Aggregate bandwidth across all threads

## Documentation

Comprehensive documentation is available for each component:

- **[CLAUDE.md](CLAUDE.md)**: Unified development guide and coding standards
- **[Context Transport Primitives](context-transport-primitives/README.md)**: Shared memory data structures
- **[Chimaera Runtime](context-runtime/README.md)**: Modular runtime system and ChiMod development
  - [MODULE_DEVELOPMENT_GUIDE.md](context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md): Complete ChiMod development guide
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

IOWarp Core is designed for high-performance computing scenarios:

- **Task Latency**: < 10 microseconds for local task execution (Chimaera Runtime)
- **Memory Bandwidth**: Up to 50 GB/s with RAM-based storage backends
- **Scalability**: Single node to multi-node cluster deployments
- **Concurrency**: Thousands of concurrent coroutine-based tasks
- **I/O Performance**: Native async I/O with multi-tiered buffering

## Contributing

We welcome contributions to the IOWarp Core project!

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Follow** the coding standards in [CLAUDE.md](CLAUDE.md)
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

IOWarp Core is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE) file for complete license text.

**Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology**

---

## Acknowledgements

IOWarp Core is developed at the [GRC lab](https://grc.iit.edu/) at Illinois Institute of Technology as part of the IOWarp project. This work is supported by the National Science Foundation (NSF) and aims to advance next-generation scientific computing infrastructure.

**For more information:**
- IOWarp Project: https://grc.iit.edu/research/projects/iowarp
- IOWarp Organization: https://github.com/iowarp
- Documentation Hub: https://grc.iit.edu/docs/category/iowarp

---

<p align="center">
  Built with ❤️ by the GRC Lab at Illinois Institute of Technology
</p>
