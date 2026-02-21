# Quickstart: PyPI (Experimental)

> **Note:** The pip package is experimental. For production use, the
> [conda-based installation](quickstart-conda.md) is recommended.

This guide walks you through installing IOWarp Core from PyPI, starting the
Chimaera runtime, and running a Context Exploration Engine (CEE) example.

## Prerequisites

- Linux (x86_64 or aarch64)
- Python 3.10 -- 3.13
- pip

## 1. Install

```bash
pip install iowarp-core
```

This installs a self-contained wheel that bundles all C++ libraries,
the `chimaera` CLI, and the Python bindings. No system dependencies
are required.

### Verify the installation

```bash
python -c "import iowarp_core; print(iowarp_core.get_version())"
chimaera --help
```

## 2. Default Configuration

The pip package bundles a default configuration inside the wheel at:

```
<site-packages>/iowarp_core/data/chimaera_default.yaml
```

You can also seed a user-local copy for customisation:

```bash
mkdir -p ~/.chimaera
python -c "
import iowarp_core, shutil, os
src = os.path.join(os.path.dirname(iowarp_core.__file__), 'data', 'chimaera_default.yaml')
dst = os.path.expanduser('~/.chimaera/chimaera.yaml')
if not os.path.exists(dst):
    shutil.copy2(src, dst)
    print(f'Seeded config to {dst}')
else:
    print(f'Config already exists at {dst}')
"
```

The runtime resolves its configuration in this order:

| Priority | Source |
|----------|--------|
| 1 | `CHI_SERVER_CONF` environment variable |
| 2 | `~/.chimaera/chimaera.yaml` |
| 3 | Bundled `iowarp_core/data/chimaera_default.yaml` |

The default configuration starts 4 worker threads on port 9413 and
composes three modules automatically:

- **chimaera_bdev** -- 512 MB RAM block device
- **wrp_cte_core** -- Context Transfer Engine with a 512 MB RAM cache
- **wrp_cae_core** -- Context Assimilation Engine

## 3. Start the Runtime

### Option A: Standalone daemon

```bash
# Start in the background
chimaera runtime start &

# Verify it is running
chimaera monitor --once

# When done
chimaera runtime stop
```

### Option B: Embedded runtime (recommended for scripts)

Set the `CHI_WITH_RUNTIME` environment variable so the runtime starts
inside your application process -- no separate daemon needed:

```bash
export CHI_WITH_RUNTIME=1
```

The CEE example below uses this approach.

### Custom configuration

Point to a different YAML file at any time:

```bash
export CHI_SERVER_CONF=/path/to/my_config.yaml
chimaera runtime start
```

## 4. Context Exploration Engine Example

The Context Exploration Engine (CEE) lets you assimilate data into IOWarp,
query for it by name or regex, retrieve it, and clean up -- all from Python.

Save the script below as `cee_quickstart.py`:

```python
#!/usr/bin/env python3
"""IOWarp CEE Quickstart -- assimilate, query, retrieve, destroy."""

import os
import sys
import tempfile

import wrp_cee as cee

# -- 1. Create a sample file -------------------------------------------
data = b"Hello from IOWarp! " * 50_000          # ~950 KB
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
tmp.write(data)
tmp.close()
print(f"Created test file: {tmp.name} ({len(data):,} bytes)")

# -- 2. Initialise the CEE interface -----------------------------------
#    ContextInterface connects to the running Chimaera runtime (or starts
#    an embedded one when CHI_WITH_RUNTIME=1).
iface = cee.ContextInterface()

# -- 3. Bundle (assimilate) the file -----------------------------------
tag = "quickstart_demo"
ctx = cee.AssimilationCtx(
    src=f"file::{tmp.name}",       # source: local file  (note :: not ://)
    dst=f"iowarp::{tag}",          # destination tag in IOWarp
    format="binary",               # raw binary ingest
)
rc = iface.context_bundle([ctx])
assert rc == 0, f"context_bundle failed (rc={rc})"
print(f"Assimilated file into tag '{tag}'")

# -- 4. Query for blobs in the tag -------------------------------------
blobs = iface.context_query(tag, ".*", 0)   # regex ".*" matches all blobs
print(f"Found {len(blobs)} blob(s): {blobs}")

# -- 5. Retrieve the data back -----------------------------------------
packed = iface.context_retrieve(tag, ".*", 0)
if packed:
    print(f"Retrieved {len(packed[0]):,} bytes")

# -- 6. Destroy the tag ------------------------------------------------
iface.context_destroy([tag])
print(f"Destroyed tag '{tag}'")

# -- Cleanup ------------------------------------------------------------
os.unlink(tmp.name)
print("Done!")
```

Run it with the embedded runtime:

```bash
export CHI_WITH_RUNTIME=1
python3 cee_quickstart.py
```

Expected output (log lines omitted):

```
Created test file: /tmp/tmpXXXXXXXX.bin (950,000 bytes)
Assimilated file into tag 'quickstart_demo'
Found 2 blob(s): ['chunk_0', 'description']
Retrieved 950,029 bytes
Destroyed tag 'quickstart_demo'
Done!
```

## 5. Key Environment Variables

| Variable | Description |
|----------|-------------|
| `CHI_SERVER_CONF` | Path to YAML config (highest priority) |
| `CHI_WITH_RUNTIME` | `1` to start an embedded runtime in-process |
| `CHI_IPC_MODE` | Transport: `SHM` (shared memory), `TCP` (default), `IPC` (Unix socket) |
| `HSHM_LOG_LEVEL` | `debug`, `info`, `warning`, `error`, `fatal` |

## 6. Limitations of the Pip Package

The pip wheel is a portable, self-contained build. Compared to the conda
install it has the following limitations:

- **No CUDA / ROCm** -- GPU acceleration is not included in the wheel.
- **No MPI** -- Distributed multi-node support is not available.
- **No HDF5** -- HDF5-based assimilation is not available.
- **Linux only** -- macOS and Windows are not supported yet.

If you need any of the above, use the
[conda-based installation](quickstart-conda.md) instead.

## Next Steps

- Edit `~/.chimaera/chimaera.yaml` to tune thread counts, storage tiers,
  or add file-backed block devices.
- See `context-runtime/config/chimaera_default.yaml` for a fully commented
  configuration reference.
- Consider switching to the [conda installation](quickstart-conda.md) for
  production workloads.
