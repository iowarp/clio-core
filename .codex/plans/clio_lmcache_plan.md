# CLIO CTE as an LMCache Backend for vLLM V1

This note captures the working plan, implementation status, benchmark findings,
and session-specific operational caveats for continuing the CLIO CTE LMCache
backend work in a new Codex session.

When starting a new session, ask Codex:

```text
Please read /workspace/.codex/plans/clio_lmcache_plan.md before working.
Follow it in addition to /workspace/AGENTS.md.
```

## Current Goal

Use vLLM's existing `LMCacheConnectorV1` unchanged. Add CLIO CTE as an LMCache
dynamic storage plugin implementing LMCache backend semantics directly.

The CTE-side implementation lives under:

```text
/workspace/context-transfer-engine/llm-hooks/lmcache/
```

The LMCache plugin lives under:

```text
/workspace/lmcache/v1/storage_backend/plugins/clio_cte_backend.py
/tmp/others/lmcache_0_5_1/lmcache/v1/storage_backend/plugins/clio_cte_backend.py
```

The installed real LMCache source tree used for testing is:

```text
/tmp/others/lmcache_0_5_1
```

## Naming

- C++ wrapper: `clio_llm::lmcache::LMCacheStore`
- Python extension: `clio_cte_lmcache_ext`
- LMCache plugin class: `ClioCTEBackend(StoragePluginInterface)`
- `ClioCTEBackend.__str__()` returns `"ClioCTEBackend"`

Important CTE files:

```text
/workspace/context-transfer-engine/llm-hooks/lmcache/include/clio_llm/lmcache/lmcache_store.h
/workspace/context-transfer-engine/llm-hooks/lmcache/src/lmcache_store.cc
/workspace/context-transfer-engine/llm-hooks/lmcache/python/lmcache_bindings.cc
```

## Implemented Status

Stage 0 and the core Stage 1 integration are implemented.

Implemented CTE wrapper APIs include:

- `Init(config_path, tag_name, pool_query_mode)`
- `PutBytes(blob_name, data)`
- `GetBytes(blob_name) -> bytes`
- `GetBytesInto(blob_name, destination, destination_size, known_blob_size=None)`
- `Exists(blob_name) -> bool`
- `Size(blob_name) -> uint64_t`
- `Delete(blob_name) -> bool`
- `Close()`

The `GetBytesInto` path was added to reduce one Python-side read copy. The
plugin uses it in the get path with a preallocated destination buffer and a
known blob size, so the C++ layer does not issue a redundant `Size()` call.

Build and install were previously successful from `/workspace/build`, followed
by passing CTE/LMCache binding tests:

- `test_iowarp_lmcache_store`
- `python_lmcache_store_bytes_test`

Do not rerun tests after source changes without installing first. This repo uses
rpaths, and `LD_LIBRARY_PATH` is not enough.

## LMCache Key and Blob Mapping

Canonical LMCache key:

```text
CacheEngineKey.to_string()
LayerCacheEngineKey.to_string()
```

CTE blob name:

```text
lmcache/v1/<sha256(canonical_key).hex()>
```

The encoded CTE record includes a fixed header plus metadata and payload. The
header stores the canonical key and the plugin validates it on read. A mismatch
is treated as a miss/error, not a false hit.

Default CTE tag:

```text
lmcache_kv
```

Configurable via:

```text
lmcache.extra_config["clio_cte.tag_name"]
```

## Scheduler and Worker Role Rules

The plugin must support scheduler/query-only construction with:

```text
local_cpu_backend=None
```

The following must work in scheduler mode:

- `contains`
- `batched_contains`
- `batched_async_contains`
- `pin`
- `unpin`
- `close`

Methods that allocate or store `MemoryObj` require `local_cpu_backend` and must
fail clearly in scheduler mode.

Pinning is currently plugin-local only. It prevents local
`remove(force=False)`, but it is not a durable CTE lease.

## Benchmark Caveat: Run Outside Codex Sandbox

Run this benchmark outside the Codex execution sandbox:

```text
/tmp/others/lmcache_0_5_1/benchmarks/storage_backend_io/storage_backend_io_benchmark.py
```

Reason: the Codex sandbox blocks Python asyncio cross-thread socketpair wakeups.
This breaks `asyncio.run_coroutine_threadsafe()` and causes LMCache
`LocalDiskBackend` async writes to timeout with `completed=0`.

This applies when benchmarking:

- `local_disk`
- `local_cpu`
- `clio_cte`

Use escalated execution for these benchmark runs. The approved prefix used in
this session was:

```text
/home/iowarp/venv/bin/python /tmp/others/lmcache_0_5_1/benchmarks/storage_backend_io/storage_backend_io_benchmark.py
```

## Benchmark Findings So Far

Same parameters used for the main comparison:

```text
--num-ops 256
--concurrency 4
--chunk-size 256
--write_bench False
```

`--write_bench False` means the benchmark first populates/writes, then measures
read in the same run.

Observed `local_cpu` baseline:

```text
write_elapsed_sec: 0.0045928
write_ops_per_sec: 55739.41
read_elapsed_sec: 0.000715158
read_ops_per_sec: 357962.86
```

Observed `clio_cte` after `GetBytesInto`, with warning-level logs:

```text
write_elapsed_sec: 10.666339198
write_ops_per_sec: 24.0007
read_elapsed_sec: 9.011629676
read_ops_per_sec: 28.4077
```

Suppressing debug logs did not materially improve `clio_cte`. The bottleneck is
per-key CTE task/RPC/scheduling plus copies, not logging.

Do not treat `local_cpu` as a fair storage backend comparison. It is an upper
bound because it mostly does a Python dict lookup and returns an existing
`MemoryObj`. Better baselines are `local_disk`, a local-loopback remote backend,
or `rust_raw_block` if buildable.

For around 1 GiB total data with this benchmark shape:

```text
--num-ops 32
--chunk-size 256
--max-local-disk-gb 1
```

Each op at the default benchmark shape is roughly 28 MiB. A 256-op run is around
7 GiB, so `local_disk` needs a larger `--max-local-disk-gb` if using 256 ops.

## LocalDisk Timeout Diagnosis

Inside the Codex sandbox, `LocalDiskBackend.batched_submit_put_task()` timed out
because async callbacks never fired. A 1-op test showed:

- `exists_in_put_tasks=True`
- no file was created
- callback did not fire
- `backend.close()` hung

Direct synchronous `LocalDiskBackend.async_save_bytes_to_disk(...)` did work:

- file created
- callback fired
- `contains=True`
- `get_blocking()` read the tensor back

Standalone diagnosis showed the sandbox blocks the socketpair write used by
Python's event loop wakeup. Outside the sandbox, the same 1-op
`LocalDiskBackend` put/get worked correctly.

## CTE Configuration Notes

The ADIOS2 test config inspected earlier was:

```text
/workspace/context-transfer-engine/test/unit/adapters/adios2/cte_config.yaml
```

It configures CTE to use a CTE runtime/storage path, not the same as LMCache
`local_cpu`. It should not be treated as equivalent to LMCache's native in-process
RAM cache.

## Dataflow Summary

LMCache `local_cpu`:

```text
put: ref count up -> store MemoryObj in in-process mapping
get: dict lookup -> return same MemoryObj
```

Current `clio_cte`:

```text
put:
  MemoryObj.byte_array
  -> Python record/header bytes
  -> nanobind
  -> C++ LMCacheStore
  -> CLIO shared buffer
  -> CTE PutBlob task/RPC/scheduling
  -> CTE storage tier

get:
  CTE Size/GetBlob path
  -> C++ GetBytesInto destination buffer
  -> Python record decode
  -> allocate LMCache MemoryObj
  -> copy payload into MemoryObj.byte_array
```

The main bottleneck is one CTE task/RPC/scheduling path per key, plus copies.

## Does CTE Support Native Batched Put/Get?

Current CTE core does not expose native batched blob payload APIs.

Existing public CTE client APIs are single-blob operations:

- `AsyncPutBlob`
- `AsyncGetBlob`
- `AsyncGetBlobSize`
- `AsyncGetBlobInfo`
- `AsyncDelBlob`
- query/list APIs such as `AsyncBlobQuery` and `AsyncGetContainedBlobs`

Existing tasks are also single-blob:

- `PutBlobTask`
- `GetBlobTask`
- `GetBlobSizeTask`
- `GetBlobInfoTask`
- `DelBlobTask`

No current `PutMany`, `GetMany`, `PutBlobs`, `GetBlobs`, or equivalent native
batched payload method was found.

## Proposed Next Optimization

Add LMCacheStore-level batch APIs while internally still using per-key CTE
tasks/RPCs. This is a Stage 2 client-side async batching optimization, not a
true CTE-native batch operation.

Potential APIs:

```text
ExistsMany(blob_names) -> list[bool]
PutMany(items)
GetMany(blob_names) -> list[bytes | None]
GetManyInto(blob_names, destinations, sizes) -> per-item status/list
```

Expected benefits:

- fewer Python-to-C++ nanobind calls
- less Python loop overhead
- C++ can submit many CTE async operations before waiting
- better wall-clock overlap for benchmark batches

Costs that remain:

- still one CTE task/RPC per blob
- still one scheduling path per blob
- still one shared-memory operation per blob
- still payload/header encode/decode
- read path still allocates/fills LMCache `MemoryObj`

This may improve `clio_cte` benchmark numbers, especially at batch/concurrency,
but it will not approach `local_cpu`.

## True Native CTE Batch Work

A real CTE-native batch design would require new CTE task/methods, for example:

```text
PutBlobsTask
GetBlobsTask
GetBlobSizesTask
ExistsBlobsTask
```

Hard part: current CTE routes each blob by:

```text
HashBlobToContainer(tag_id, blob_name)
```

A batch may contain blobs owned by different containers. Possible approaches:

- group batch items by destination container on the client
- broadcast and let each container handle owned items
- add runtime fanout task

This is higher impact but more invasive.

## vLLM Config Shape

Example vLLM/LMCache connector config:

```json
{
  "kv_connector": "LMCacheConnectorV1",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.storage_plugins": ["clio_cte"],
    "lmcache.store_location": "ClioCTEBackend",
    "lmcache.retrieve_locations": ["ClioCTEBackend"],
    "lmcache.extra_config": {
      "storage_plugin.clio_cte.module_path": "lmcache.v1.storage_backend.plugins.clio_cte_backend",
      "storage_plugin.clio_cte.class_name": "ClioCTEBackend",
      "clio_cte.tag_name": "lmcache_kv",
      "clio_cte.config_path": "/path/to/cte_config.yaml"
    }
  }
}
```

## Known Runtime Issue

The installed compressor runtime library can break CLIO runtime scans:

```text
/usr/local/lib/libclio_cte_compressor_runtime.so
```

It has previously failed due to `libzfp.so.1`/SYCL-related loading problems.
For runtime tests/benchmarks, the workaround used was to temporarily move it out
of `/usr/local/lib`, start the runtime, then restore it. Always restore it.

## Build/Test Reminders

- Build from `/workspace/build`.
- Install before rerunning tests.
- Do not rely on `LD_LIBRARY_PATH` because this repo uses rpaths.
- Do not build in source directories.
- Do not hardcode absolute paths in CMake.
- Preserve unrelated dirty worktree changes.

