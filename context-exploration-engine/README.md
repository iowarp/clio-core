# Context Exploration Engine (CEE)

High-level C++ and Python API for exploring, querying, and managing IOWarp scientific data contexts.

[![License](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](LICENSE)

## Overview

The Context Exploration Engine provides a unified interface for:
- **Bundling**: Assimilating data files into the IOWarp storage system
- **Querying**: Discovering and retrieving stored data by tag and blob patterns
- **Managing**: Creating, destroying, and organizing data contexts

CEE integrates with the Context Transfer Engine (CTE) and Context Assimilation Engine (CAE) to provide seamless data management capabilities.

## Components

### ContextInterface API
High-level C++ API for context management operations:
- `ContextBundle()` - Bundle and assimilate files into CTE storage
- `ContextQuery()` - Query for data using regex patterns
- `ContextDestroy()` - Remove contexts by name
- `ContextRetrieve()` - Retrieve data and metadata (planned)
- `ContextSplice()` - Split/splice contexts (planned)

### Python Bindings
Python interface to ContextInterface with snake_case methods:
- `context_bundle()` - Bundle data files
- `context_query()` - Query stored data
- `context_destroy()` - Remove contexts

### Legacy Components
- **mcp-hdf-demo**: HDF5 Model Context Protocol server and client demo
- **hdf-compass**: wxPython-4 based HDF Compass viewer

## Building

CEE is built as part of the IOWarp Core unified build system:

```bash
# Configure with debug preset
cmake --preset=debug -DWRP_CORE_ENABLE_CEE=ON

# Build CEE API and tests
cmake --build build --target wrp_cee_api -j$(nproc)
cmake --build build --target test_context_bundle -j$(nproc)
```

## Testing

### Unit Tests

CEE includes comprehensive unit tests demonstrating the complete workflow:

**Bundle-and-Retrieve Test** ([api/test/test_context_bundle.cc](api/test/test_context_bundle.cc)):
- Generates 1MB test file with patterned data
- Registers RAM storage target with CTE
- Creates CAE pool dynamically
- Bundles file using ContextBundle API
- Queries to verify data storage
- Cleans up test context

**Query Test** ([api/test/test_context_query.cc](api/test/test_context_query.cc)):
- Tests basic query operations
- Validates regex pattern matching

**Destroy Test** ([api/test/test_context_destroy.cc](api/test/test_context_destroy.cc)):
- Tests context deletion
- Validates error handling

### Running Tests

Tests support two modes:

**1. With External Runtime:**
```bash
# Start runtime in separate terminal
cd build && ./bin/chimaera runtime start

# Run tests
./bin/test_context_bundle
./bin/test_context_query
./bin/test_context_destroy
```

**2. With Embedded Runtime:**
```bash
# Tests initialize runtime automatically
INIT_CHIMAERA=1 ./bin/test_context_bundle
INIT_CHIMAERA=1 ./bin/test_context_query
INIT_CHIMAERA=1 ./bin/test_context_destroy
```

### Test Configuration

Tests use [api/test/wrp_config.yaml](api/test/wrp_config.yaml) for runtime configuration:
- 4 worker threads (sched + slow)
- 2GB main segment, 1GB client/runtime segments
- RAM-based storage (4GB capacity)
- Single-node configuration for unit testing

## Usage Example

### C++ API

```cpp
#include <wrp_cee/api/context_interface.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>

// Initialize
iowarp::ContextInterface ctx_interface;

// Bundle a file
std::vector<wrp_cae::core::AssimilationCtx> bundle;
wrp_cae::core::AssimilationCtx ctx;
ctx.src = "file::/path/to/data.bin";
ctx.dst = "iowarp::my_dataset";
ctx.format = "binary";
bundle.push_back(ctx);

int result = ctx_interface.ContextBundle(bundle);

// Query for data
std::vector<std::string> blobs = ctx_interface.ContextQuery(
    "my_dataset",  // Tag pattern
    ".*");         // Blob pattern (all blobs)

// Cleanup
std::vector<std::string> contexts = {"my_dataset"};
ctx_interface.ContextDestroy(contexts);
```

### Python API

```python
from wrp_cee import ContextInterface, AssimilationCtx

# Initialize
ctx = ContextInterface()

# Bundle a file
bundle = [AssimilationCtx(
    src="file::/path/to/data.bin",
    dst="iowarp::my_dataset",
    format="binary"
)]
result = ctx.context_bundle(bundle)

# Query for data
blobs = ctx.context_query("my_dataset", ".*")

# Cleanup
ctx.context_destroy(["my_dataset"])
```

## Quick Start (Legacy)

### MCP HDF5 Demo
```bash
cd mcp-hdf-demo
pip install -r requirements.txt
# Start the MCP server and client
```

### HDF Compass
```bash
cd hdf-compass
# Follow installation instructions in hdf-compass/README.md
```

## License

BSD-3-Clause License - see [LICENSE](LICENSE) file for details.

**Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology**
