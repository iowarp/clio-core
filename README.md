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

**CLIO Core** is a unified framework that integrates multiple high-performance
components for context management, data transfer, and scientific computing.
CLIO Core enables developers to build efficient data processing pipelines for
HPC, storage systems, and near-data computing applications.

It provides:

- **High-performance context management** for computational contexts and data
  transformations.
- **Heterogeneous-aware I/O** with multi-tiered, dynamic buffering.
- **A modular runtime** with dynamically loadable processing modules.
- **Shared-memory data structures** that work across host, CUDA, and ROCm.
- **Distributed-by-construction** scaling from single node to clusters.

## Architecture

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
                    │  CLIO Runtime   │
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

The pip wheel ships a **portable, self-contained build** with all
dependencies statically linked. No system installs are required beyond
glibc and Python 3.10+.

```bash
pip install iowarp-core
```

The wheel includes the CLIO runtime, the `clio_run` CLI, the CTE, CAE, and
CEE engines, and the `clio_cee` Python bindings. A default configuration is
seeded at `~/.clio/clio.yaml` on first import.

Switch to a source build below if you need any of:

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

`release` corresponds to a variant under `installers/conda/variants/`. Other
variants (`cuda`, `rocm`, `mpi`, `full`, `release-fuse`, `debug`, ...) enable
the corresponding features.

For a full **bare-metal source build** (without Conda) with the per-feature
apt / dnf dependency list and the complete `CLIO_*_ENABLE_*` flag checklist,
see [INSTALL.md](INSTALL.md). Other methods (Docker, Spack) are documented in
[docs/getting-started/installation](docs/docs/getting-started/installation.mdx).

## Components

| Component | Location | Purpose |
|---|---|---|
| **Context Transport Primitives** | [`context-transport-primitives/`](context-transport-primitives/) | Shared-memory containers, allocators, sync primitives, networking (ZMQ / libfabric / Thallium). GPU-aware (CUDA, ROCm). |
| **CLIO Runtime** | [`context-runtime/`](context-runtime/) | Coroutine-based modular runtime (< 10 µs task latency). Hosts ChiMods and provides admin + bdev modules. |
| **Context Transfer Engine** | [`context-transfer-engine/`](context-transfer-engine/) | Multi-tiered, heterogeneous-aware I/O buffering. Tag + blob storage with adapters for POSIX, HDF5 (VFD/VOL), ADIOS2, MPI-IO, FUSE3, GDS. |
| **Context Assimilation Engine** | [`context-assimilation-engine/`](context-assimilation-engine/) | OMNI-YAML-driven data ingestion (binary, HDF5, Globus) into CTE. |
| **Context Exploration Engine** | [`context-exploration-engine/`](context-exploration-engine/) | High-level C++ and Python (`clio_cee`) API for bundling, querying, and retrieving data. Includes an MCP server for AI agents. |

## Quickstart

### Start the runtime

Installation seeds a default configuration at `~/.clio/clio.yaml`, so the
runtime works out of the box:

```bash
clio_run start          # foreground
clio_run start &        # background
```

To override the configuration, point `CLIO_X` at your own YAML file:

```bash
export CLIO_X=/path/to/my_config.yaml
clio_run start
```

### Python: bundle, query, retrieve

```python
import clio_cee as cee

ctx_interface = cee.ContextInterface()

# Assimilate a file into IOWarp storage.
ctx = cee.AssimilationCtx(
    src="file::/path/to/data.bin",
    dst="iowarp::my_dataset",
    format="binary",
)
ctx_interface.context_bundle([ctx])

# Query for blob names matching a regex.
blobs = ctx_interface.context_query("my_dataset", ".*", 0)

# Retrieve blob payloads.
data = ctx_interface.context_retrieve("my_dataset", ".*", 0)

# Clean up.
ctx_interface.context_destroy(["my_dataset"])
```

### C++ (CTE, direct)

For direct CTE put/get from C++, see the canonical example and operation
reference in the [Context Transfer Engine
README](context-transfer-engine/README.md#c-client-api).

## Testing

```bash
cd build/release
ctest -VV                    # all unit tests
ctest -R context_transport   # CTP tests
ctest -R runtime             # runtime tests
ctest -R cte                 # CTE tests
ctest -R omni                # CAE tests
ctest -R context             # CEE tests
```

## Benchmarking

CLIO Core ships two main benchmarks; pass `--help` to either for the full
parameter list.

- **`clio_run_thrpt_bench`** — runtime task throughput and latency
  (`bdev_io`, `bdev_allocation`, `bdev_task_alloc`, `latency` test cases).
- **`clio_cte_bench`** — CTE Put / Get / PutGet throughput across threads,
  async depth, I/O size, and key-space cardinality.

Example:

```bash
clio_run_thrpt_bench --test-case bdev_io --threads 8 --duration 30
clio_cte_bench       --op PutGet --threads 8 --depth 16 --io-size 1m --io-count 200
```

## Documentation

- **[AGENTS.md](AGENTS.md)** — unified development guide and coding standards.
- **[INSTALL.md](INSTALL.md)** — bare-metal source-build instructions.
- **[Context Transport Primitives](context-transport-primitives/README.md)**
- **[CLIO Runtime](context-runtime/README.md)**
- **[Context Transfer Engine](context-transfer-engine/README.md)** — canonical
  C++ CTE API reference.
- **[Context Assimilation Engine](context-assimilation-engine/README.md)**
- **[Context Exploration Engine](context-exploration-engine/README.md)**
- **Full documentation site:** <https://grc.iit.edu/docs/category/iowarp>

## Use Cases

**Scientific computing:** data processing pipelines, near-data computing,
custom storage engines, workflows with context management.

**Storage systems:** distributed file system backends, object storage,
multi-tiered caches, high-throughput I/O buffering.

**HPC and data-intensive workloads:** accelerated I/O, ingestion and
transformation pipelines, heterogeneous computing with GPU support, real-time
streaming analytics.

## Performance Characteristics

- **Task latency:** < 10 µs for local task execution.
- **Memory bandwidth:** up to 50 GB/s with the RAM bdev backend.
- **Scalability:** single node to multi-node clusters.
- **Concurrency:** thousands of concurrent coroutine-based tasks.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-feature`.
3. Follow the standards in [AGENTS.md](AGENTS.md).
4. `ctest --test-dir build/release` before opening a PR.
5. Submit a pull request against `iowarp/clio-core`.

## License

CLIO Core is licensed under the **BSD 3-Clause License**. See
[LICENSE](LICENSE) for the full text.

**Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology**

---

## Acknowledgements

CLIO Core is developed at the [GRC lab](https://grc.iit.edu/) at Illinois
Institute of Technology as part of the IOWarp project, supported by the
National Science Foundation (NSF) to advance next-generation scientific
computing infrastructure.

- IOWarp project: <https://grc.iit.edu/research/projects/iowarp>
- IOWarp organization: <https://github.com/iowarp>
- Documentation hub: <https://grc.iit.edu/docs/category/iowarp>
