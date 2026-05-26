# Context Assimilation Engine (CAE)

A CLIO ChiMod for high-performance data ingestion into the IOWarp ecosystem.
CAE assimilates data from external sources — local binary files, HDF5
datasets, and Globus endpoints — into the [Context Transfer
Engine](../context-transfer-engine) (CTE) for distributed storage and
retrieval.

## Overview

CAE runs as a pool inside the [CLIO runtime](../context-runtime) alongside
CTE. Clients submit OMNI YAML files to describe data transfers; CAE parses
them and dispatches assimilation tasks to the appropriate backend (binary,
HDF5, or Globus).

```
External Source          CAE Module              CTE Module
(file, HDF5, Globus) --> ParseOmni task -------> Tag + Blob storage
```

**Supported formats:** `binary`, `hdf5`, `globus`

## Building

CAE is built as part of CLIO Core; it is enabled by default
(`CLIO_CORE_ENABLE_CAE=ON`):

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core
cmake --preset release
cmake --build build/release -j$(nproc)
```

HDF5 ingestion is opt-in (the CMake flag also turns on CTE's HDF5 adapters):

```bash
cmake --preset release -DCLIO_CORE_ENABLE_HDF5=ON
```

Globus ingestion is opt-in via `-DCAE_ENABLE_GLOBUS=ON` and requires the
Globus toolkit on `PATH`.

## Running

### 1. Start the CLIO runtime with CTE and CAE composed

```bash
export CLIO_X=/path/to/clio_config.yaml
clio_run start
```

The example configuration at
[`config/clio_config_example.yaml`](config/clio_config_example.yaml)
deploys both CTE and CAE via the runtime's declarative `compose` section:

```yaml
compose:
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
  - mod_name: clio_cae_core
    pool_name: cae_main
    pool_query: local
    pool_id: "400.0"
```

### 2. Submit an OMNI file

```bash
clio_cae /path/to/my_transfers.yaml
```

## OMNI File Format

OMNI files describe data transfers in YAML using a `transfers` list:

```yaml
name: my_ingestion_job

transfers:
  - src: file::/path/to/data.bin    # Source URL (required)
    dst: iowarp::my_tag             # CTE destination tag (required)
    format: binary                  # Format: binary, hdf5, globus (required)
    depends_on: ""                  # Dependency identifier (optional)
    range_off: 0                    # Byte offset in source (optional)
    range_size: 0                   # Bytes to read, 0 = full file (optional)
    src_token: $GLOBUS_TOKEN        # Auth token, env vars expanded (optional)
```

**HDF5 with dataset filtering:**

```yaml
transfers:
  - src: file::/data/experiment.h5
    dst: iowarp::experiment
    format: hdf5
    dataset_filter:
      include_patterns:
        - "/sensors/*"
      exclude_patterns:
        - "/sensors/raw/*"
```

**Key fields:**

| Field | Description |
|---|---|
| `src` | Source URL (`file::`, `globus::`) |
| `dst` | CTE destination tag (`iowarp::tag_name`) |
| `format` | `binary`, `hdf5`, or `globus` |
| `range_off` / `range_size` | Byte range within source (0 = full file) |
| `src_token` / `dst_token` | Auth tokens; environment variables are expanded |
| `dataset_filter` | HDF5 dataset include/exclude glob patterns |

## Linking from C++

CAE is exported through the unified `clio-core` CMake package:

```cmake
find_package(clio-core CONFIG REQUIRED)

target_link_libraries(my_app
  clio::cae::core_client
  clio::cte::core_client
)
```

For the higher-level Python/C++ ingestion façade that drives CAE under the
hood, see the [Context Exploration Engine](../context-exploration-engine).

## Project Structure

```
config/      Example runtime configuration files
core/        ChiMod implementation
  include/   Public headers (tasks, assimilation context, factory)
  src/       Runtime and client implementation
  util/      clio_cae command-line tool
data/        Sample datasets for testing (HDF5, CSV, Parquet)
test/
  unit/                       Unit tests (binary, HDF5, range, error handling)
  integration/globus_matsci/  Globus integration test
```

## Pool ID Reference

| Component | Pool ID | mod_name        |
|-----------|---------|-----------------|
| Admin     | 1.0     | (auto-created)  |
| CTE Core  | 512.0   | `clio_cte_core` |
| CAE Core  | 400.0   | `clio_cae_core` |

## License

BSD-3-Clause. See [LICENSE](../LICENSE).

## Links

- **IOWarp Organization:** [https://github.com/iowarp](https://github.com/iowarp)
- **Issues:** [https://github.com/iowarp/clio-core/issues](https://github.com/iowarp/clio-core/issues)
