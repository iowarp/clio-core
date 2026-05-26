# Context Exploration Engine (CEE)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](../LICENSE)

The high-level C++ and Python API for exploring, querying, and managing
IOWarp scientific data contexts. CEE is the user-facing facade that drives
the [Context Transfer Engine](../context-transfer-engine) (CTE) and [Context
Assimilation Engine](../context-assimilation-engine) (CAE) under the hood.

## What it provides

- **Bundling** — assimilate one or more data files into the IOWarp storage
  system through CAE.
- **Querying** — discover stored data by tag/blob regex.
- **Retrieval** — pull blob payloads back out of CTE.
- **Lifecycle** — create and destroy named contexts.

## Components

- **`ContextInterface`** — the canonical C++ API
  ([`api/include/clio_cee/api/context_interface.h`](api/include/clio_cee/api/context_interface.h)),
  in the `iowarp::` namespace.
- **`clio_cee` Python module** — wraps `ContextInterface` with `snake_case`
  methods and a Pythonic `AssimilationCtx` dataclass.
- **`iowarp-cei-mcp`** — an MCP server exposing CEE operations for AI agents
  (see [`iowarp-cei-mcp/README.md`](iowarp-cei-mcp/README.md)).

## Building

CEE is enabled by default in CLIO Core (`CLIO_CORE_ENABLE_CEE=ON`):

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core
cmake --preset release
cmake --build build/release -j$(nproc)
```

To build a debug variant with CEE explicitly:

```bash
cmake --preset debug -DCLIO_CORE_ENABLE_CEE=ON
cmake --build build/debug -j$(nproc)
```

## C++ Usage

```cpp
#include <clio_cee/api/context_interface.h>
#include <clio_cae/core/factory/assimilation_ctx.h>

int main() {
  // ContextInterface handles runtime initialization internally.
  iowarp::ContextInterface ctx_interface;

  // 1. Bundle a file into IOWarp storage.
  std::vector<clio::cae::core::AssimilationCtx> bundle;
  clio::cae::core::AssimilationCtx ctx;
  ctx.src    = "file::/path/to/data.bin";
  ctx.dst    = "iowarp::my_dataset";
  ctx.format = "binary";
  bundle.push_back(ctx);
  int rc = ctx_interface.ContextBundle(bundle);

  // 2. Query for matching blobs.
  auto blobs = ctx_interface.ContextQuery(
      "my_dataset",   // tag regex
      ".*");          // blob regex

  // 3. Clean up.
  ctx_interface.ContextDestroy({"my_dataset"});
  return 0;
}
```

Link with the unified CMake package:

```cmake
find_package(clio-core CONFIG REQUIRED)
target_link_libraries(my_app
  clio::cee::api
  clio::cae::core_client
  clio::cte::core_client
)
```

## Python Usage

```python
import clio_cee as cee

# Construct (initializes the runtime internally on first use).
ctx_interface = cee.ContextInterface()

# Bundle a file.
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

## Testing

```bash
ctest --test-dir build/release -R context
```

The tests under [`api/test/`](api/test/) demonstrate the full
bundle / query / retrieve / destroy workflow against an embedded runtime.
They use [`api/test/clio_config.yaml`](api/test/clio_config.yaml), which
provisions a 4-worker single-node runtime with a 4 GB RAM-backed CTE target.

Tests can run in two modes:

```bash
# Against an externally-started runtime:
clio_run start &
./build/release/bin/test_context_bundle

# With an embedded runtime (auto-spawned by the test):
INIT_CHIMAERA=1 ./build/release/bin/test_context_bundle
```

## Project structure

```
api/             C++ ContextInterface, Python bindings, tests, demos
iowarp-cei-mcp/  MCP server exposing CEE operations
hdf-compass/     Legacy wxPython HDF Compass viewer (pre-CEE)
mcp-hdf-demo/    Legacy HDF5 MCP demo
```

## License

BSD-3-Clause. See [LICENSE](../LICENSE).
