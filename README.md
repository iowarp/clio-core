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
**Location:** [`runtime/`](runtime/)

High-performance modular runtime for scientific computing and storage systems with coroutine-based task execution.

**Key Features:**
- Ultra-high performance task execution (< 10μs latency)
- Modular ChiMod system for dynamic extensibility
- Coroutine-aware synchronization (CoMutex, CoRwLock)
- Distributed architecture with shared memory IPC
- Built-in storage backends (RAM, file-based, custom block devices)

**[Read more →](runtime/README.md)**

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

## Getting Started

### Prerequisites

IOWarp Core requires the following dependencies:

**System Requirements:**
- C++17 compatible compiler (GCC >= 9, Clang >= 10)
- CMake >= 3.10
- Linux (Ubuntu 20.04+, CentOS 8+, or similar)
- Optional: CUDA toolkit (for GPU support)
- Optional: ROCm (for AMD GPU support)
- Optional: MPI, ZeroMQ, Thallium (for distributed computing)

**Docker Container (Recommended):**
All dependencies are pre-installed in our Docker container:
```bash
docker pull iowarp/iowarp-build:latest
```

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/iowarp/iowarp-core.git
cd iowarp-core

# Configure with CMake preset (debug mode)
cmake --preset=debug

# Build all components
cmake --build build --parallel $(nproc)

# Install to system or custom prefix
cmake --install build --prefix /usr/local
```

### Component Build Options

The unified build system provides options to enable/disable components:

```bash
cmake --preset=debug \
  -DWRP_CORE_ENABLE_RUNTIME=ON \
  -DWRP_CORE_ENABLE_CTE=ON \
  -DWRP_CORE_ENABLE_CAE=ON \
  -DWRP_CORE_ENABLE_CEE=ON
```

**Available Options:**
- `WRP_CORE_ENABLE_RUNTIME`: Enable runtime component (default: ON)
- `WRP_CORE_ENABLE_CTE`: Enable context-transfer-engine (default: ON)
- `WRP_CORE_ENABLE_CAE`: Enable context-assimilation-engine (default: ON)
- `WRP_CORE_ENABLE_CEE`: Enable context-exploration-engine (default: ON)

### Quick Start Example

Here's a simple example using the Chimaera runtime with the bdev ChiMod:

```cpp
#include <chimaera/chimaera.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/admin/admin_client.h>

int main() {
  // Initialize Chimaera client
  chi::CHIMAERA_CLIENT_INIT();

  // Create admin client (always required)
  chimaera::admin::Client admin_client(chi::PoolId(7000, 0));
  admin_client.Create(HSHM_MCTX, chi::PoolQuery::Local());

  // Create bdev client for high-speed RAM storage
  chimaera::bdev::Client bdev_client(chi::PoolId(8000, 0));
  bdev_client.Create(HSHM_MCTX, chi::PoolQuery::Local(),
                    chimaera::bdev::BdevType::kRam, "", 1024*1024*1024); // 1GB RAM

  // Allocate and use a block
  auto block = bdev_client.Allocate(HSHM_MCTX, 4096);  // 4KB block
  std::vector<hshm::u8> data(4096, 0xAB);
  bdev_client.Write(HSHM_MCTX, block, data);
  auto read_data = bdev_client.Read(HSHM_MCTX, block);
  bdev_client.Free(HSHM_MCTX, block);

  return 0;
}
```

**Build and Link:**
```cmake
find_package(chimaera REQUIRED)
find_package(chimaera_admin REQUIRED)
find_package(chimaera_bdev REQUIRED)

target_link_libraries(my_app
  chimaera::cxx
  chimaera::admin_client
  chimaera::bdev_client
)
```

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

## Documentation

Comprehensive documentation is available for each component:

- **[CLAUDE.md](CLAUDE.md)**: Unified development guide and coding standards
- **[Context Transport Primitives](context-transport-primitives/README.md)**: Shared memory data structures
- **[Chimaera Runtime](runtime/README.md)**: Modular runtime system and ChiMod development
  - [MODULE_DEVELOPMENT_GUIDE.md](context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md): Complete ChiMod development guide
- **[Context Transfer Engine](context-transfer-engine/README.md)**: I/O buffering and acceleration
  - [CTE API Documentation](context-transfer-engine/docs/cte/cte.md): Complete API reference
- **[Context Assimilation Engine](context-assimilation-engine/README.md)**: Data ingestion and processing
- **[Context Exploration Engine](context-exploration-engine/README.md)**: Interactive data exploration

## Docker Deployment

IOWarp Core can be deployed using Docker containers for distributed deployments:

```bash
# Build and start 3-node cluster
cd docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f iowarp-node1

# Stop cluster
docker-compose down
```

See [CLAUDE.md](CLAUDE.md) for detailed Docker deployment configuration.

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

See [CLAUDE.md](CLAUDE.md) for complete coding standards and workflow guidelines.

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
