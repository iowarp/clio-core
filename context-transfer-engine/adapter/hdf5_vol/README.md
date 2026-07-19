# clio HDF5 VOL connector

A pass-through HDF5 VOL connector that transparently caches HDF5 data in the
clio-core CTE (Context Transfer Engine) while preserving native HDF5 file
semantics. It wraps a native under-VOL: every operation delegates to native, and
whole-dataset `H5Dwrite`/`H5Dread` transfers of atomic datatypes are additionally
mirrored to CTE via async `PutBlob`/`GetBlob`. Loaded transparently through the
standard HDF5 plugin mechanism — no application source changes.

The native file is always written synchronously and remains authoritative. Only
atomic datatypes (integer, float, enum, bitfield, fixed-length string) are cached —
compound/array (mem/file layout can differ) and vlen/reference (pointers, not bytes)
delegate to native. **Selection-aware reads:** hyperslab/point reads are served from
the cached linear image via HDF5 gather/scatter when it is already populated
(serve-only; a miss falls back to native). A non-whole write invalidates the cache
so reads never see stale data. **Safe mode:** `H5Fflush` and `H5Fclose` drain all
in-flight async CTE puts before returning, so a successful flush/close is a real
durability barrier for the cache path (no async write outlives it).

Files: `clio_vol.cc`, `clio_vol.h`. Connector name: **`clio`**.

## Enabling and building

The adapter is OFF by default (`CLIO_CTE_ENABLE_HDF5_VOL`, see the top-level
`CMakeLists.txt`). Enable and build it:

```bash
cmake -S <repo> -B <build> -DCLIO_CTE_ENABLE_HDF5_VOL=ON
cmake --build <build> --target clio_hdf5_vol -j
# -> <build>/bin/libclio_hdf5_vol.so
```

**HDF5 linkage matters.** A VOL connector is ABI-coupled to the `libhdf5` of the
application that `dlopen`s it. The `.so` must link the *same* HDF5 the loading app
(and `h5dump`/`h5ls`) use, or HDF5 will refuse to load it or mismatch the VOL ABI.
Confirm with `ldd <build>/bin/libclio_hdf5_vol.so | grep hdf5`. The connector
requires HDF5 >= 1.14 and does not require ELF support (unlike the VFD adapter).

## Using it

```bash
export HDF5_PLUGIN_PATH=<build>/bin
export HDF5_VOL_CONNECTOR=clio        # under-VOL defaults to native
# a clio_run runtime must be reachable for the CTE cache path
```
Any HDF5 application (h5py, C, tools) then routes through the connector. Files it
writes are valid native HDF5 files readable by standard tools.

## Access telemetry (observability)

Set `CLIO_VOL_TRACE=<dir>` to record per-access observability (observe-only; it
never changes the data path, and when unset it is a single cached bool check with
zero overhead). Only the VOL sees HDF5 semantics — dataset paths, selection shapes,
datatypes — that the blob layer cannot, so it captures them here. Per file, two
artifacts are written to `<dir>`:

- `<file>.<pid>.access.jsonl` — one JSON record per `H5Dread`/`H5Dwrite`: dataset,
  op, datatype, selection kind + a bounds signature (for repeat detection), element
  count, bytes, how it was served (`cache` / `native` / `uncacheable`), whether the
  dataset is chunked and if the access is chunk-aligned, and duration.
- `<file>.<pid>.access.json` — an aggregated summary written at file close: per
  dataset and overall, the read **cache hit rate**, selection-shape histogram, read
  bytes from tier vs native, write mirroring, storage **layout** (chunked? chunk
  dims, aligned vs misaligned read counts — the rechunk signal), **read latency**
  split cache-vs-native, transfer-size min/max/mean, and the count of distinct vs
  maximally-repeated read selections (the cache/prefetch signal).

Filenames carry the pid so concurrent processes (e.g. MPI ranks) don't clobber
each other's output.

This is the data a CLIO-using agent reads to advise tuning (hot datasets, whether
caching is helping, which selections repeat, small-read/per-object-cost patterns).

## Testing and current status

Compatibility is verified by a differential suite (native VOL as the oracle):
each feature case is written/read through the connector and compared to native,
plus `h5diff`/`h5dump`. It is registered as the CTest test
`cte_hdf5_vol_compat_suite` (gated on `CLIO_CTE_ENABLE_HDF5_VOL`; `RUN_SERIAL`
because it manages its own `clio_run`).

```bash
ctest -R cte_hdf5_vol_compat_suite --output-on-failure
# or directly:
python3 benchmarks/hdf5-ingest/vol_compat_suite.py --bin <build>/bin
```

The test runs in **honest mode** (no `--expect-fail` allowlist): it reports the
connector's real compatibility state rather than masking gaps, so any future
regression fails the test. For the authoritative, always-current status run the
test; do not rely on numbers copied into prose. As of this writing it is **green
(20/20)** — compatible for dataset I/O (signed/unsigned int, float, compound, enum,
array-element, scalar, and variable-length string datatypes; contiguous and
chunked; shuffle/fletcher32 filters; whole-dataset, hyperslab, and point
selections; extendible datasets with reopen+append), attributes, group/link
structure, modern-API iteration (C `H5Ovisit3`/`H5Literate2`), selection-aware read
caching, Safe-mode `H5Fflush` durability, and access telemetry.

Feature areas the h5py cases can't reach (h5py has poor non-native VOL support) run
through the C API / a telemetry check, all automated by the suite: iteration
(`vol_c_iteration_test.c`), Safe-mode flush (`vol_c_safeflush_test.c`), selection
caching + partial-write invalidation (`vol_c_selection_test.c`), and access
telemetry (`telemetry` case — runs a workload under `CLIO_VOL_TRACE` and checks
the summary).
