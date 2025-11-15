# Context Exploration Engine - Python API Documentation

## Overview

The Context Exploration Engine (CEE) provides a high-level Python API for managing and exploring data contexts in IOWarp. The API is accessible through the `wrp_cee` Python module and offers a simple interface for data assimilation, querying, retrieval, and cleanup operations.

**Key Feature:** The CEE API automatically initializes the IOWarp runtime when you create a `ContextInterface` instance. You don't need to manually initialize Chimaera, CTE, or CAE - the `ContextInterface` constructor handles all of this internally.

## Installation

### Prerequisites

1. Build IOWarp with Python bindings enabled:
   ```bash
   cmake --preset=debug -DWRP_CORE_ENABLE_PYTHON=ON
   cmake --build build -j$(nproc)
   sudo cmake --install build
   ```

2. The `wrp_cee` module will be installed to your Python site-packages directory.

### Verification

```python
import wrp_cee
print("CEE API loaded successfully!")
```

## Module: `wrp_cee`

The `wrp_cee` module provides two main classes:

1. **`AssimilationCtx`** - Configuration for data assimilation operations
2. **`ContextInterface`** - Main API for context management

---

## Class: `AssimilationCtx`

Configuration object that defines how data should be assimilated into IOWarp.

### Constructors

#### Default Constructor
```python
ctx = wrp_cee.AssimilationCtx()
```

Creates an empty AssimilationCtx with default values.

#### Full Constructor
```python
ctx = wrp_cee.AssimilationCtx(
    src="file::/path/to/source",
    dst="iowarp::tag_name",
    format="binary",
    depends_on="",           # Optional
    range_off=0,            # Optional
    range_size=0,           # Optional
    src_token="",           # Optional
    dst_token=""            # Optional
)
```

**Parameters:**

- **`src`** (str): Source URL specifying where to read data from
  - Format: `protocol::path`
  - Examples: `file::/tmp/data.bin`, `globus::endpoint_id/path`
  - **Note:** Use `::` separator, not `://`

- **`dst`** (str): Destination URL specifying where to store data in IOWarp
  - Format: `iowarp::tag_name`
  - Example: `iowarp::my_dataset`

- **`format`** (str): Data format specification
  - Supported values: `"binary"`, `"hdf5"`
  - Default: `"binary"`

- **`depends_on`** (str, optional): Dependency identifier for dependent operations
  - Default: `""` (no dependency)

- **`range_off`** (int, optional): Byte offset in source file
  - Useful for partial file assimilation
  - Default: `0` (start from beginning)

- **`range_size`** (int, optional): Number of bytes to read
  - Default: `0` (read entire file)

- **`src_token`** (str, optional): Authentication token for source
  - Used for protected data sources
  - Default: `""`

- **`dst_token`** (str, optional): Authentication token for destination
  - Used for protected destinations
  - Default: `""`

### Attributes

All constructor parameters are accessible as read/write attributes:

```python
ctx = wrp_cee.AssimilationCtx()
ctx.src = "file::/my/data.bin"
ctx.dst = "iowarp::dataset"
ctx.format = "binary"
ctx.range_off = 1024      # Skip first 1KB
ctx.range_size = 10485760  # Read 10MB
```

### String Representation

```python
ctx = wrp_cee.AssimilationCtx(
    src="file::/data.bin",
    dst="iowarp::tag",
    format="binary"
)
print(ctx)
# Output: <AssimilationCtx src='file::/data.bin' dst='iowarp::tag' format='binary'>
```

### Example Usage

```python
import wrp_cee as cee

# Assimilate entire file
ctx1 = cee.AssimilationCtx(
    src="file::/data/experiment1.bin",
    dst="iowarp::experiment_data",
    format="binary"
)

# Assimilate partial file (10MB starting at 1MB offset)
ctx2 = cee.AssimilationCtx(
    src="file::/large_dataset.bin",
    dst="iowarp::dataset_chunk1",
    format="binary",
    range_off=1048576,      # 1MB offset
    range_size=10485760     # 10MB size
)

# Assimilate HDF5 file
ctx3 = cee.AssimilationCtx(
    src="file::/simulation.h5",
    dst="iowarp::simulation_results",
    format="hdf5"
)
```

---

## Class: `ContextInterface`

Main API for context exploration and management operations.

**Important:** The `ContextInterface` constructor automatically initializes the IOWarp runtime (Chimaera + CTE + CAE). This initialization happens once when you create the first `ContextInterface` instance. Subsequent instances will use the already-initialized runtime.

### Constructor

```python
ctx_interface = wrp_cee.ContextInterface()
```

**Parameters:** None

**Notes:**
- Automatically initializes CAE client (which in turn initializes CTE and Chimaera)
- Verifies Chimaera IPC is available
- Sets `is_initialized_` flag on success
- Assumes runtime configuration is already set via environment variables (e.g., `CHI_SERVER_CONF`)

**Typical Environment Setup:**

The runtime needs to know where to find its configuration. This is typically set via environment variable:

```bash
export CHI_SERVER_CONF=/path/to/config.yaml
```

Or in Python:
```python
import os
os.environ['CHI_SERVER_CONF'] = '/path/to/config.yaml'

import wrp_cee
ctx_interface = wrp_cee.ContextInterface()
```

### Methods

---

#### `context_bundle(bundle)`

Assimilate a group of objects into IOWarp.

**Signature:**
```python
result = ctx_interface.context_bundle(bundle)
```

**Parameters:**
- **`bundle`** (List[AssimilationCtx]): List of AssimilationCtx objects to assimilate

**Returns:**
- **int**: `0` on success, non-zero error code on failure

**Description:**

Assimilates one or more data objects into IOWarp. Each `AssimilationCtx` in the bundle describes a source file/dataset to assimilate and where to store it. The method calls the CAE's `ParseOmni` function which schedules assimilation tasks for each context.

**Example:**

```python
import wrp_cee as cee

# Create interface
ctx_interface = cee.ContextInterface()

# Define files to assimilate
contexts = [
    cee.AssimilationCtx(
        src="file::/data/file1.bin",
        dst="iowarp::dataset",
        format="binary"
    ),
    cee.AssimilationCtx(
        src="file::/data/file2.bin",
        dst="iowarp::dataset",
        format="binary"
    ),
]

# Assimilate all files
result = ctx_interface.context_bundle(contexts)
if result == 0:
    print("Success!")
else:
    print(f"Failed with code: {result}")
```

---

#### `context_query(tag_re, blob_re, max_results=0)`

Query for blob names matching tag and blob patterns.

**Signature:**
```python
blob_names = ctx_interface.context_query(tag_re, blob_re, max_results=0)
```

**Parameters:**
- **`tag_re`** (str): Regular expression pattern to match tag names
- **`blob_re`** (str): Regular expression pattern to match blob names
- **`max_results`** (int, optional): Maximum number of results to return
  - `0` = unlimited (default)
  - Positive integer = limit to that many results

**Returns:**
- **List[str]**: List of matching blob names (empty list if none found)

**Description:**

Queries the CTE system for blobs matching the specified regex patterns. Uses `BlobQuery` with `Broadcast` pool query to search across all nodes. Returns only the blob names, not the data.

**Example:**

```python
# Query all blobs in a specific tag
blobs = ctx_interface.context_query("experiment_data", ".*", 0)
print(f"Found {len(blobs)} blobs: {blobs}")

# Query blobs matching a pattern, limit to 10 results
blobs = ctx_interface.context_query("dataset.*", "chunk_[0-9]+", 10)
print(f"Found {len(blobs)} matching blobs")

# Query specific blob name
blobs = ctx_interface.context_query("my_tag", "exact_blob_name", 0)
if blobs:
    print("Blob exists!")
```

---

#### `context_retrieve(tag_re, blob_re, max_results=1024, max_context_size=268435456, batch_size=32)`

Retrieve blob data matching patterns into packed binary context.

**Signature:**
```python
packed_data = ctx_interface.context_retrieve(
    tag_re,
    blob_re,
    max_results=1024,
    max_context_size=256*1024*1024,
    batch_size=32
)
```

**Parameters:**
- **`tag_re`** (str): Regular expression pattern to match tag names
- **`blob_re`** (str): Regular expression pattern to match blob names
- **`max_results`** (int, optional): Maximum number of blobs to retrieve
  - `0` = unlimited
  - Default: `1024`
- **`max_context_size`** (int, optional): Maximum total context size in bytes
  - Default: `268435456` (256MB)
- **`batch_size`** (int, optional): Number of concurrent `AsyncGetBlob` operations
  - Controls parallelism
  - Default: `32`

**Returns:**
- **List[str]**: List containing one string with packed binary context data
  - Returns empty list if no data found or error occurred
  - The string contains raw binary data (concatenated blob contents)

**Description:**

Retrieves blob data matching the specified patterns and packs it into a single binary buffer. The method:
1. Uses `BlobQuery` to find matching blobs
2. Allocates a buffer of size `max_context_size`
3. Retrieves blobs in batches using `AsyncGetBlob`
4. Packs data sequentially into the buffer
5. Returns the packed data as a string

Blobs are processed in batches for efficiency. The buffer is automatically allocated and freed.

**Example:**

```python
# Retrieve all blobs from a tag (using defaults)
packed_data = ctx_interface.context_retrieve("my_tag", ".*")
if packed_data:
    print(f"Retrieved {len(packed_data[0])} bytes")

# Retrieve with custom limits
packed_data = ctx_interface.context_retrieve(
    tag_re="dataset.*",
    blob_re="chunk_.*",
    max_results=100,           # Limit to 100 blobs
    max_context_size=100*1024*1024,  # 100MB total
    batch_size=64              # 64 concurrent operations
)

# Process retrieved data
if packed_data and len(packed_data) > 0:
    binary_data = packed_data[0]
    # binary_data is raw bytes containing concatenated blob contents
    print(f"Total data: {len(binary_data):,} bytes")
```

**Notes:**
- Data is packed sequentially (blob1 + blob2 + blob3...)
- No delimiters or metadata between blobs
- Retrieval stops when buffer is full or all blobs retrieved

---

#### `context_destroy(context_names)`

Destroy contexts (tags) by name.

**Signature:**
```python
result = ctx_interface.context_destroy(context_names)
```

**Parameters:**
- **`context_names`** (List[str]): List of context (tag) names to destroy

**Returns:**
- **int**: `0` on success, non-zero on failure

**Description:**

Deletes the specified contexts from the CTE system. Each context name is treated as a tag name and deleted using CTE's `DelTag` API. This operation removes the tag and all associated blobs.

**Example:**

```python
# Delete single context
result = ctx_interface.context_destroy(["my_tag"])

# Delete multiple contexts
result = ctx_interface.context_destroy([
    "experiment1_data",
    "experiment2_data",
    "temp_results"
])

if result == 0:
    print("All contexts destroyed successfully")
else:
    print(f"Failed to destroy contexts: {result}")
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Complete CEE API example"""

import wrp_cee as cee
import os
import tempfile

# Create test file
test_file = os.path.join(tempfile.gettempdir(), "test_data.bin")
with open(test_file, 'wb') as f:
    f.write(b"Hello, IOWarp!" * 1000)

try:
    # Initialize interface (handles runtime init)
    ctx_interface = cee.ContextInterface()

    # 1. Assimilate file
    ctx = cee.AssimilationCtx(
        src=f"file::{test_file}",
        dst="iowarp::demo_tag",
        format="binary"
    )
    result = ctx_interface.context_bundle([ctx])
    print(f"Assimilation: {'Success' if result == 0 else 'Failed'}")

    # 2. Query for blobs
    blobs = ctx_interface.context_query("demo_tag", ".*", 0)
    print(f"Found {len(blobs)} blobs: {blobs}")

    # 3. Retrieve blob data
    data = ctx_interface.context_retrieve("demo_tag", ".*")
    if data:
        print(f"Retrieved {len(data[0]):,} bytes")

    # 4. Clean up
    result = ctx_interface.context_destroy(["demo_tag"])
    print(f"Cleanup: {'Success' if result == 0 else 'Failed'}")

finally:
    # Remove test file
    if os.path.exists(test_file):
        os.remove(test_file)
```

---

## URL Format Requirements

**Important:** IOWarp uses `::` as the URL separator, **NOT** `://`.

### Correct Format
```python
src="file::/path/to/file.bin"
dst="iowarp::my_tag"
```

### Incorrect Format
```python
src="file:///path/to/file.bin"    # Wrong! Don't use ://
dst="iowarp://my_tag"              # Wrong! Don't use ://
```

---

## Runtime Assumptions

The CEE Python API assumes:

1. **Runtime is Started:** The IOWarp runtime (Chimaera server) should be running, or will be started by the `ContextInterface` constructor.

2. **Configuration Available:** Runtime configuration is available via environment variable:
   ```bash
   export CHI_SERVER_CONF=/path/to/config.yaml
   ```

3. **Proper Permissions:** Your Python process has permission to access shared memory segments and connect to the runtime.

4. **Dependencies Initialized:** When you create a `ContextInterface`, it will:
   - Initialize CAE client
   - Initialize CTE client (via CAE)
   - Initialize Chimaera client (via CTE)
   - Verify IPC manager is available

---

## Error Handling

Methods return error codes or empty results on failure:

```python
# context_bundle returns int
result = ctx_interface.context_bundle([ctx])
if result != 0:
    print(f"Error code: {result}")

# context_query returns empty list on error
blobs = ctx_interface.context_query("tag", ".*")
if not blobs:
    print("No blobs found or error occurred")

# context_retrieve returns empty list on error
data = ctx_interface.context_retrieve("tag", ".*")
if not data:
    print("No data retrieved or error occurred")

# context_destroy returns int
result = ctx_interface.context_destroy(["tag"])
if result != 0:
    print(f"Error code: {result}")
```

---

## See Also

- **C++ API Documentation:** `context-exploration-engine/api/include/wrp_cee/api/context_interface.h`
- **Unit Tests:** `context-exploration-engine/api/test/test_context_interface.py`
- **Demo Script:** `context-exploration-engine/api/demo/simple_assimilation_demo.py`
- **CTE Documentation:** `context-transfer-engine/docs/cte/cte.md`
