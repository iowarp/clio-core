# CLIO Runtime

<p align="center">
  <strong>High-Performance Modular Runtime for Scientific Computing and Storage Systems</strong>
  <br />
  <br />
  <a href="#getting-started">Getting Started</a> ·
  <a href="#external-integration">External Integration</a> ·
  <a href="#module-development">Module Development</a> ·
  <a href="#contributing">Contributing</a>
</p>

---

The **CLIO Runtime** is the distributed, coroutine-based task-execution
runtime that backs CLIO Core. It hosts dynamically-loaded modules (ChiMods),
provides shared-memory IPC via the [Context Transport
Primitives](../context-transport-primitives) (CTP), and powers higher-level
engines like CTE, CAE, and CEE. The on-disk binary is `clio_run`.

## What it provides

- **Microsecond-level task latency** via coroutine scheduling on top of CTP
  shared-memory queues.
- **Pluggable ChiMods** — drop a `.so` and a `clio_mod.yaml` in a repository
  directory and the runtime loads it on startup.
- **Coroutine-aware synchronization** — `CoMutex`, `CoRwLock` for deadlock-free
  coordination across tasks.
- **Distributed by construction** — pools span hosts via the `lightbeam`
  transport (ZMQ, libfabric, or Thallium).
- **Built-in storage modules** — `admin` for pool management, `bdev` for RAM /
  file-based block devices.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Client App 1   │    │  Client App 2   │    │  External App   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ Module Clients  │    │ Module Clients  │    │ Module Clients  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  CLIO Runtime   │
                    │                 │
                    │  Core Services: │
                    │  • IPC Manager  │
                    │  • Work Orch.   │
                    │  • Pool Manager │
                    │  • Module Mgr   │
                    └─────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │ Admin Module  │   │ Bdev Module   │   │ Custom Module │
    │ (Pool Mgmt)   │   │ (Block I/O)   │   │ (User Logic)  │
    └───────────────┘   └───────────────┘   └───────────────┘
```

## Getting Started

### Prerequisites

- C++20 compiler (GCC ≥ 11 or Clang ≥ 14)
- CMake ≥ 3.20
- Linux

See [INSTALL.md](../INSTALL.md) for the full per-feature dependency list.

### Build

The runtime is built as part of CLIO Core:

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core
cmake --preset release
cmake --build build/release -j$(nproc)
sudo cmake --install build/release
```

### Start the runtime

```bash
clio_run start          # foreground
clio_run start &        # background
```

A default configuration is seeded at `~/.clio/clio.yaml` on install. To
override it, point `CLIO_X` at your own YAML file:

```bash
export CLIO_X=/path/to/my_config.yaml
clio_run start
```

### Minimal client example

```cpp
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/admin/admin_client.h>

int main() {
  // Initialize CLIO Runtime in client mode with an embedded runtime.
  clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);

  // Always-required admin client (pool management).
  clio::run::admin::Client admin_client(clio::run::PoolId(7000, 0));
  admin_client.Create(clio::run::PoolQuery::Local());

  // RAM-backed block device.
  clio::run::bdev::Client bdev(clio::run::PoolId(8000, 0));
  bdev.Create(clio::run::PoolQuery::Local(),
              clio::run::bdev::BdevType::kRam, "",
              1024ull * 1024 * 1024);   // 1 GB

  auto block = bdev.Allocate(4096);
  std::vector<ctp::u8> data(4096, 0xAB);
  bdev.Write(block, data);
  auto read_data = bdev.Read(block);
  bdev.Free(block);

  return 0;
}
```

## External Integration

CLIO Core ships a single unified CMake package, `clio-core`, that exposes the
runtime and all enabled ChiMods. Link the modular targets directly:

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyClioApp)

find_package(clio-core CONFIG REQUIRED)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app
  clio::run::cxx              # core runtime library
  clio::run::admin_client     # admin module (always available)
  clio::run::bdev_client      # block-device module (always available)
)
```

If CLIO Core is installed in a non-standard prefix, point CMake at it via
`CMAKE_PREFIX_PATH`:

```bash
export CMAKE_PREFIX_PATH=/path/to/clio-core/install
```

### Available ChiMods

| Module | Purpose | Client target | Runtime target |
|--------|---------|---------------|----------------|
| **admin** | Pool creation and system administration (always available) | `clio::run::admin_client` | `clio::run::admin_runtime` |
| **bdev** | Block-device operations (RAM / file backends, always available) | `clio::run::bdev_client` | `clio::run::bdev_runtime` |
| **CTE core** | Context Transfer Engine (if `CLIO_CORE_ENABLE_CTE=ON`) | `clio::cte::core_client` | `clio::cte::core_runtime` |
| **CAE core** | Context Assimilation Engine (if `CLIO_CORE_ENABLE_CAE=ON`) | `clio::cae::core_client` | `clio::cae::core_runtime` |

### Runtime modes

`clio::run::CLIO_INIT` takes a `RuntimeMode` and an `embedded_runtime` boolean:

```cpp
// Most common: client with an embedded runtime (auto-starts on first init).
clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);

// Client-only — connects to a runtime already started by `clio_run start`.
clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false);

// Server / runtime-only — for embedding the runtime inside another daemon.
clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer, false);
```

## Module Development

Each ChiMod is a self-contained shared library exposing a client interface, a
runtime container, and a set of tasks.

### Layout

```
modules/my_module/
├── clio_mod.yaml                   # module metadata
├── CMakeLists.txt
├── doc/
│   ├── my_module.md                # API reference
│   └── integration.md
├── include/clio_runtime/my_module/ # public headers
│   ├── my_module_client.h
│   ├── my_module_runtime.h
│   └── my_module_tasks.h
└── src/
    ├── my_module_client.cc
    └── my_module_runtime.cc
```

### Sketch

```cpp
// Task definition
struct MyCustomTask : public clio::run::Task {
  uint64_t input_data_;
  uint64_t result_;
  uint32_t result_code_;
};

// Client
class Client : public clio::run::ContainerClient {
 public:
  uint64_t ProcessData(const ctp::ipc::MemContext& mctx, uint64_t data);
  ctp::ipc::FullPtr<MyCustomTask>
      AsyncProcessData(const ctp::ipc::MemContext& mctx, uint64_t data);
};

// Runtime
class Runtime : public clio::run::Container {
 public:
  void ProcessData(ctp::ipc::FullPtr<MyCustomTask> task,
                   clio::run::RunContext& ctx) {
    task->result_ = process_algorithm(task->input_data_);
    task->result_code_ = 0;
  }
};
```

The `MOD_NAME/` directory under [`modules/`](modules/) is a complete starter
template you can copy and rename.

## Testing

```bash
ctest --test-dir build/release            # all runtime tests
ctest --test-dir build/release -R bdev    # block-device module tests
ctest --test-dir build/release -R comutex # synchronization tests
```

## Performance characteristics

- **Task latency:** < 10 µs for local execution.
- **Memory bandwidth:** up to 50 GB/s with the RAM bdev backend.
- **Scalability:** single node to multi-node clusters via `lightbeam`.
- **Concurrency:** thousands of concurrent coroutine-based tasks.

## Contributing

1. Fork the repo and create a feature branch.
2. Follow the standards in [AGENTS.md](../AGENTS.md).
3. `ctest --test-dir build/release` before opening a PR.
4. Submit a PR against `iowarp/clio-core`.

## License

BSD-3-Clause. Developed at the [GRC lab](https://grc.iit.edu/) at Illinois
Institute of Technology as part of the [IOWarp
project](https://grc.iit.edu/research/projects/iowarp).
