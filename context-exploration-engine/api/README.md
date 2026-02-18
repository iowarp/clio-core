# Context Exploration Engine API

This directory contains the ContextInterface C++ wrapper and Python bindings for the IOWarp Context Exploration Engine.

## Overview

The ContextInterface provides a high-level API for context exploration and management, integrating with:
- **Context Assimilation Engine (CAE)**: For bundling and assimilating objects
- **Context Transfer Engine (CTE)**: For querying and managing contexts

## API Reference

### C++ Interface

```cpp
#include <wrp_cee/api/context_interface.h>

namespace iowarp {
class ContextInterface {
public:
  // Bundle a group of related objects together and assimilate them
  int ContextBundle(const std::vector<wrp_cae::core::AssimilationCtx> &bundle);

  // Retrieve the identities of objects matching tag and blob patterns
  std::vector<std::string> ContextQuery(const std::string &tag_re,
                                         const std::string &blob_re);

  // Retrieve the identities and data of objects (NOT YET IMPLEMENTED)
  std::vector<std::string> ContextRetrieve(const std::string &tag_re,
                                            const std::string &blob_re);

  // Split/splice objects into a new context (NOT YET IMPLEMENTED)
  int ContextSplice(const std::string &new_ctx,
                     const std::string &tag_re,
                     const std::string &blob_re);

  // Destroy contexts by name
  int ContextDestroy(const std::vector<std::string> &context_names);
};
}
```

## Implemented Methods

### 1. ContextBundle

**Implementation**: [src/context_interface.cc:32](../src/context_interface.cc#L32)

Wraps the CAE `ParseOmni` functionality to bundle and assimilate a group of related objects.

**Underlying API**: `wrp_cae::core::Client::ParseOmni()`

**Parameters**:
- `bundle`: Vector of `AssimilationCtx` objects defining source, destination, format, and other metadata

**Returns**:
- `0` on success
- Non-zero error code on failure

**Example**:
```cpp
iowarp::ContextInterface ctx;

std::vector<wrp_cae::core::AssimilationCtx> bundle;
wrp_cae::core::AssimilationCtx ctx1;
ctx1.src = "file::/path/to/data.bin";
ctx1.dst = "iowarp::my_tag";
ctx1.format = "binary";
bundle.push_back(ctx1);

int result = ctx.ContextBundle(bundle);
if (result == 0) {
  std::cout << "Bundle assimilated successfully!" << std::endl;
}
```

### 2. ContextQuery

**Implementation**: [src/context_interface.cc:66](../src/context_interface.cc#L66)

Queries the CTE system for blobs matching specified regex patterns.

**Underlying API**: `wrp_cte::core::Client::BlobQuery()`

**Parameters**:
- `tag_re`: Tag regex pattern to match
- `blob_re`: Blob regex pattern to match

**Returns**:
- Vector of matching blob names

**Example**:
```cpp
iowarp::ContextInterface ctx;

// Query all blobs in tags starting with "dataset_"
auto results = ctx.ContextQuery("dataset_.*", ".*");

for (const auto& blob_name : results) {
  std::cout << "Found blob: " << blob_name << std::endl;
}
```

### 3. ContextDestroy

**Implementation**: [src/context_interface.cc:127](../src/context_interface.cc#L127)

Destroys contexts by deleting their corresponding tags from the CTE system.

**Underlying API**: `wrp_cte::core::Client::DelTag()` (called for each context name)

**Parameters**:
- `context_names`: Vector of context names to destroy

**Returns**:
- `0` on success
- `1` if any deletions fail

**Example**:
```cpp
iowarp::ContextInterface ctx;

std::vector<std::string> contexts_to_delete = {"old_context_1", "old_context_2"};
int result = ctx.ContextDestroy(contexts_to_delete);

if (result == 0) {
  std::cout << "All contexts destroyed successfully!" << std::endl;
}
```

### 4. ContextRetrieve (NOT YET IMPLEMENTED)

**Implementation**: [src/context_interface.cc:91](../src/context_interface.cc#L91)

Placeholder for future implementation. Currently returns an empty vector.

**Planned Functionality**: Retrieve both identities and data of objects matching patterns.

### 5. ContextSplice (NOT YET IMPLEMENTED)

**Implementation**: [src/context_interface.cc:103](../src/context_interface.cc#L103)

Placeholder for future implementation. Currently returns error code `1`.

**Planned Functionality**: Split/splice objects into a new context based on patterns.

## AssimilationCtx Structure

```cpp
namespace wrp_cae::core {
struct AssimilationCtx {
  std::string src;         // Source URL (e.g., file::/path/to/file)
  std::string dst;         // Destination URL (e.g., iowarp::tag_name)
  std::string format;      // Data format (e.g., binary, hdf5)
  std::string depends_on;  // Dependency identifier (empty if none)
  size_t range_off;        // Byte offset in source file
  size_t range_size;       // Number of bytes to read
  std::string src_token;   // Authentication token for source
  std::string dst_token;   // Authentication token for destination
};
}
```

## Python Bindings

Python bindings are available when nanobind is installed and enabled during build.

**Module**: `wrp_cee`

**Example**:
```python
import wrp_cee

# Create interface
ctx = wrp_cee.ContextInterface()

# Create an assimilation context
assim_ctx = wrp_cee.AssimilationCtx(
    src="file::/data/input.bin",
    dst="iowarp::my_dataset",
    format="binary"
)

# Bundle and assimilate
result = ctx.context_bundle([assim_ctx])

# Query contexts
blobs = ctx.context_query("my_.*", ".*")
print(f"Found {len(blobs)} blobs")

# Destroy contexts
ctx.context_destroy(["old_context"])
```

## Building

### C++ Library

```bash
# Configure with debug preset
cmake --preset=debug

# Build the library
cmake --build build --target wrp_cee_api

# Library output: build/bin/libwrp_cee_api.so.1.0.0
```

### Python Bindings

Python bindings require nanobind to be installed:

```bash
pip install nanobind
cmake --preset=debug
cmake --build build --target wrp_cee
```

If nanobind is not found, the build will skip Python bindings with a warning.

### Unit Tests

#### Building Tests

```bash
# Build all tests
cmake --build build --target test_context_bundle test_context_query test_context_destroy
```

#### Starting the Chimaera Runtime

**IMPORTANT**: Tests require the Chimaera runtime to be running. Start it before running tests:

```bash
# Terminal 1: Start the Chimaera runtime
cd build
LD_LIBRARY_PATH=/workspace/build/bin:$LD_LIBRARY_PATH \
WRP_CTE_CONF=/workspace/context-assimilation-engine/test/unit/wrp_config.yaml \
./bin/chimaera runtime start

# Wait for message: "Successfully started local server at 127.0.0.1:9129"
```

#### Running Tests

```bash
# Terminal 2: Run tests with CTest
cd build
ctest -L cee

# Or run individual tests directly (must set LD_LIBRARY_PATH)
LD_LIBRARY_PATH=/workspace/build/bin:$LD_LIBRARY_PATH ./bin/test_context_bundle
LD_LIBRARY_PATH=/workspace/build/bin:$LD_LIBRARY_PATH ./bin/test_context_query
LD_LIBRARY_PATH=/workspace/build/bin:$LD_LIBRARY_PATH ./bin/test_context_destroy
```

#### Stopping the Runtime

```bash
# When done testing
cd build
./bin/chimaera runtime stop
```

## Unit Tests

### test_context_bundle
- Tests empty bundle handling
- Tests AssimilationCtx constructor with all parameters
- Verifies bundle processing completes without crashes

### test_context_query
- Tests basic query with wildcard patterns
- Tests specific tag and blob regex patterns
- Verifies query returns valid vector results

### test_context_destroy
- Tests empty context list handling
- Tests deletion of non-existent contexts
- Tests handling of special characters in context names
- Verifies destruction completes without crashes

## Directory Structure

```
api/
├── CMakeLists.txt                          # Build configuration
├── README.md                               # This file
├── cmake/
│   └── wrp_cee_api-config.cmake.in        # Package config template
├── include/wrp_cee/api/
│   └── context_interface.h                # Public API header
├── src/
│   ├── context_interface.cc               # C++ implementation
│   └── python_bindings.cc                 # Python bindings (nanobind)
└── test/
    ├── CMakeLists.txt                     # Test configuration
    ├── test_context_bundle.cc             # Bundle API tests
    ├── test_context_query.cc              # Query API tests
    └── test_context_destroy.cc            # Destroy API tests
```

## Dependencies

**Required**:
- Chimaera runtime (chimaera::cxx)
- Context Transfer Engine (wrp_cte_core_client)
- Context Assimilation Engine (wrp_cae_core_client)

**Optional**:
- nanobind (for Python bindings)

## Integration with External Projects

To use the ContextInterface in external projects:

```cmake
# Find the package
find_package(wrp_cee_api REQUIRED)

# Link against your target
add_executable(my_app main.cc)
target_link_libraries(my_app
  PRIVATE
    wrp_cee::api
)
```

## Status

### Implemented
- ✅ ContextBundle - Fully functional
- ✅ ContextQuery - Fully functional
- ✅ ContextDestroy - Fully functional
- ✅ C++ API and library
- ✅ Python bindings structure (requires nanobind)
- ✅ Unit tests for all implemented methods
- ✅ CMake build system
- ✅ Package configuration for external integration

### Planned
- ⏳ ContextRetrieve - Placeholder implementation
- ⏳ ContextSplice - Placeholder implementation

## See Also

- [CAE Documentation](../../context-assimilation-engine/README.md)
- [CTE Documentation](../../context-transfer-engine/docs/cte/cte.md)
- [IOWarp Core Development Guide](../../CLAUDE.md)
