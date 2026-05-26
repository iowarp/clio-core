# Context Transfer Engine (CTE)

[![Project Site](https://img.shields.io/badge/Project-Site-blue)](https://grc.iit.edu/research/projects/iowarp)
[![Documentation](https://img.shields.io/badge/Docs-Hub-green)](https://grc.iit.edu/docs/category/iowarp)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](../LICENSE)

The Context Transfer Engine is a heterogeneous-aware, multi-tiered, dynamic,
distributed I/O buffering layer designed to accelerate I/O for HPC and
data-intensive workloads. CTE runs as a pool inside the [CLIO
runtime](../context-runtime) and is the primary blob/tag storage substrate for
the higher-level engines ([CAE](../context-assimilation-engine),
[CEE](../context-exploration-engine)).

## Overview

CTE provides:

- **Programmable buffering** across memory and storage tiers.
- **Multiple I/O pathways** via adapters (POSIX, STDIO, MPI-IO, HDF5 VFD/VOL,
  ADIOS2, FUSE3, NVIDIA GDS).
- **Tag + Blob storage model** — `Tag`s name a container; `Blob`s store the
  data, addressable by `(tag_id, blob_name, offset, size)`.
- **Async-first API** returning a `Future`, so multiple operations can be
  pipelined before the first `Wait()`.

## Building

CTE is built as part of CLIO Core; it is enabled by default
(`CLIO_CORE_ENABLE_CTE=ON`):

```bash
git clone --recurse-submodules https://github.com/iowarp/clio-core.git
cd clio-core
cmake --preset release
cmake --build build/release -j$(nproc)
```

See [INSTALL.md](../INSTALL.md) for adapter-specific dependencies (HDF5,
ADIOS2, FUSE3, GDS, etc.) and the matching `CLIO_CTE_ENABLE_*` flags.

## C++ Client API

The CTE client lives under `<clio_cte/core/...>` and is initialized through a
single helper. The runtime must already be running (`clio_run start`) — the
client auto-connects on first use and creates the CTE pool on demand. Storage
targets are configured declaratively in the runtime's `compose` YAML, so no
`RegisterTarget` call is required from the client.

```cpp
#include <clio_cte/core/core_client.h>
#include <clio_runtime/ipc_manager.h>
#include <cstring>

int main() {
  // 1. Initialize the CTE client. Auto-connects to the runtime started by
  //    `clio_run start` and creates the CTE pool on the first call.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) return 1;
  auto *cte = CLIO_CTE_CLIENT;

  // 2. Get-or-create a named container for blobs. The async APIs return a
  //    Future immediately; Wait() blocks for completion.
  auto tag_future = cte->AsyncGetOrCreateTag("my_tag");
  tag_future.Wait();
  auto tag_id = tag_future->tag_id_;

  // 3. Stage blob data into a CTE-managed shm buffer and submit the PutBlob
  //    asynchronously. Submit-then-Wait is the canonical pattern; multiple
  //    AsyncPutBlob calls can be in flight before the first Wait() to
  //    pipeline I/O. The async signatures take a type-erased ShmPtr<>, so
  //    we wrap put_buf.shm_ (typed ShmPtr<char>) in the void-typed view.
  constexpr size_t kSize = 4096;
  auto put_buf = CLIO_IPC->AllocateBuffer(kSize);
  std::memset(put_buf.ptr_, 'A', kSize);
  ctp::ipc::ShmPtr<> put_data(put_buf.shm_);
  auto put_future = cte->AsyncPutBlob(tag_id, "my_blob",
                                       /*offset=*/0, kSize,
                                       put_data);
  put_future.Wait();
  CLIO_IPC->FreeBuffer(put_buf);

  // 4. Pre-allocate the receive buffer in shm, fire an async GetBlob, then
  //    Wait — the buffer holds the blob data on return.
  auto get_buf = CLIO_IPC->AllocateBuffer(kSize);
  ctp::ipc::ShmPtr<> get_data(get_buf.shm_);
  auto get_future = cte->AsyncGetBlob(tag_id, "my_blob",
                                       /*offset=*/0, kSize,
                                       /*flags=*/0,
                                       get_data);
  get_future.Wait();
  // get_buf.ptr_ now holds the retrieved bytes.
  CLIO_IPC->FreeBuffer(get_buf);

  // 5. Clean up.
  cte->AsyncDelTag(tag_id).Wait();
  return 0;
}
```

### Build and link

CTE is exported through the unified `clio-core` CMake package:

```cmake
find_package(clio-core CONFIG REQUIRED)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app
  clio::cte::core_client      # CTE client (for the example above)
  clio::run::admin_client     # admin module (always available)
  clio::run::bdev_client      # block-device module (always available)
)
```

### Operation reference

The client header [`core/include/clio_cte/core/core_client.h`](core/include/clio_cte/core/core_client.h)
is the authoritative API. The main operations:

| Operation | Async API | Notes |
|---|---|---|
| Tag lifecycle | `AsyncGetOrCreateTag`, `AsyncDelTag` | Tags are the container abstraction. |
| Blob I/O | `AsyncPutBlob`, `AsyncGetBlob` | Take a `ctp::ipc::ShmPtr<>` payload allocated via `CLIO_IPC->AllocateBuffer`. |
| Blob metadata | `AsyncGetBlobScore`, `AsyncGetBlobSize`, `AsyncGetBlobInfo` | Score drives tier placement. |
| Tag enumeration | `AsyncGetContainedBlobIds`, `AsyncTagExists` | Enumerate blobs within a tag. |

Each async call returns a `Future<Task>` — call `.Wait()` to block, or keep
the future around to pipeline subsequent calls.

## Python and Higher-Level APIs

- **Python** — the [CEE](../context-exploration-engine) module
  (`import clio_cee`) wraps the CTE put/get/query workflow with a higher-level
  `ContextInterface`.
- **OMNI ingestion** — for declarative ingestion from external sources
  (binary files, HDF5, Globus), use the [Context Assimilation
  Engine](../context-assimilation-engine) which lowers OMNI YAML jobs into
  CTE tag/blob writes.

## Testing

```bash
cd build/release
ctest -R cte           # unit + integration tests for CTE
```

A distributed multi-node smoke test lives in
[`test/integration/distributed/`](test/integration/distributed/).

## Project structure

```
core/        CTE client + runtime container
adapter/     I/O pathway adapters (POSIX, STDIO, MPI-IO, VFD, VOL, ADIOS2, FUSE, GDS)
compressor/  Compression backends (zstd, lz4, zlib, xz, brotli, blosc2)
config/      Example runtime YAML deploying CTE
benchmark/   clio_cte_bench and related throughput/latency benchmarks
llm-hooks/   Hooks for the bundled llama.cpp integration
uvm/         Unified-virtual-memory experiments
wrapper/     Python and language bindings
test/        Unit + integration tests
```

## License

BSD-3-Clause. See [LICENSE](../LICENSE).
