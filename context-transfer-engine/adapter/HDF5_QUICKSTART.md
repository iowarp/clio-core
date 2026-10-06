# Using CLIO with HDF5

CLIO plugs into HDF5 at two points, the VOL layer and the VFD layer, without
any change to the application's source. This page explains where each sits,
which to use, and how to get a working, verified setup. The per-connector
READMEs ([hdf5_vol](hdf5_vol/README.md), [vfd](vfd/README.md)) are the full
reference.

## What always holds

- **The native file is authoritative.** Both connectors write every byte to an
  ordinary HDF5 file. `h5dump`, `h5py` and every other tool read it with or
  without CLIO running. Nothing is lost if the CLIO tier is lost.
- **CLIO never changes results, and never announces itself.** With no
  runtime, no CTE pool or a disabled cache, both connectors quietly fall back to
  the native file. A setup can look healthy and do nothing; see
  [Confirming it works](#confirming-it-works).
- **Selection is by environment variable.** `HDF5_VOL_CONNECTOR=clio` or
  `HDF5_DRIVER=clio_vfd`. Pick one.
- **The `pip` wheel does not include the connectors.** Build them from source.

## Where the connectors sit

```
 application  (C, h5py, h5dump ...)
      │  H5Dread(dataset, selection, type)
 HDF5 API
      │
 VOL layer ─────── native VOL (default)
      │            clio VOL:  sees files, groups, datasets, datatypes, selections
      │                       caches decoded dataset images in the CTE tier
 HDF5 internals    metadata cache, chunking, filters, compression
      │
 VFD layer ─────── sec2 (default)
      │            clio_vfd:  sees byte ranges at file offsets, nothing more
      │                       optional read tier in the CTE (off by default)
 file on disk
```

The **VOL** intercepts the request as the application made it: this dataset,
this hyperslab, this type. A cache hit skips everything below it, including
HDF5's own chunk lookup and filter pipeline, and the connector can record
per-dataset telemetry. It handles only what it understands and passes the rest
through to native.

The **VFD** sees what HDF5 turned that request into: reads and writes of
metadata and raw bytes at file offsets, with no idea which dataset they belong
to. It is a drop-in replacement for the default `sec2` driver.

## Which to use

| | VOL (`clio`) | VFD (`clio_vfd`) |
|---|---|---|
| Caches | Dataset images of atomic types (int, float, enum, bitfield, fixed string) | Byte ranges, only with `CLIO_VFD_READ_TIER=1` |
| Passes to native | Compound, array, vlen, reference types; everything that is not dataset I/O | Everything, unless the read tier is on |
| Telemetry | Per-dataset access trace (`CLIO_VOL_TRACE`) | `CLIO_VFD_DEBUG` prints each I/O |
| Extra cost | A runtime round trip per cached transfer | `fsync` on every file close (deliberate) |

- **Start with the VOL** when the same datasets are read again in later
  sessions, and the file lives on storage slower than the tier: a parallel
  filesystem, network storage, or cold disk.
- **Use the VFD** when the VOL cannot help: datatypes it passes through
  (compound, vlen), or workloads dominated by HDF5's own metadata I/O. It
  caches only with `CLIO_VFD_READ_TIER=1`, and that tier is experimental.
  Without it the VFD is a write-through `sec2` replacement.
- **Use neither** when the file sits on a local disk and fits in the page
  cache. A tier hit then replaces a page-cache hit with a runtime round trip,
  and is slower. Measure against `CLIO_VOL_CACHE=0`, not against running
  without the connector.

## Setup

### 1. Build the connectors

HDF5 1.14 or newer is required. The connectors must link the **same**
`libhdf5` as the application that loads them.

```bash
cmake -S . -B build -DCLIO_CTE_ENABLE_HDF5_VOL=ON -DCLIO_CTE_ENABLE_VFD=ON \
      -DHDF5_ROOT=<prefix of the HDF5 your application uses>
cmake --build build --target clio_run clio_hdf5_vol clio_vfd -j
# -> build/bin/clio_run, build/bin/libclio_hdf5_vol.so, build/bin/libclio_vfd.so
```

HDF5 loads connectors with `dlopen`, so a mismatch shows up as a failed
`H5Fopen` that never names the library it could not load. For the same reason,
a `.so` copied out of the dev container rarely works on the host.

### 2. Start a runtime with a CTE pool

```bash
export CLIO_SERVER_CONF=$PWD/context-transfer-engine/adapter/config/clio_hdf5_quickstart.yaml
build/bin/clio_run start &      # stop it later with: build/bin/clio_run stop
```

Without a config, `clio_run` composes no CTE pool, so there is nothing to cache
in. `clio_hdf5_quickstart.yaml` is a small DRAM-only config that starts
anywhere. For real workloads start from `context-runtime/config/clio_default.yaml`
and add storage tiers, but **rename its CTE pool to `clio_cte_core`**. Clients
look the pool up by that exact name. The default file's `cte_main` misses and
falls back to broadcast routing, which is slower and has hung on attach.

### 3. Select a connector

```bash
export HDF5_PLUGIN_PATH=$PWD/build/bin
export HDF5_VOL_CONNECTOR=clio     # or: export HDF5_DRIVER=clio_vfd
```

Any HDF5 application now routes through the connector. To select one in code
instead, use `H5Pset_vol` or `H5Pset_fapl_clio`; see the READMEs.

### 4. Check it

```bash
scripts/clio-hdf5-doctor --smoke
```

The doctor checks each silent failure below and prints the fix. `--smoke`
round-trips a file through the VOL and reports how many bytes came from the
cache.

## Confirming it works

Correct data proves nothing, since both connectors return correct data with no
tier at all. Two tools tell them apart:

- **`CLIO_REQUIRE_RUNTIME=1`** makes an open fail if the runtime or CTE pool
  cannot be reached, instead of falling back. Set it in tests, benchmarks and CI.
- **`CLIO_VOL_TRACE=<dir>`** writes a `<file>.<pid>.access.json` summary at
  each file close. `totals.read_bytes_from_cache` against
  `read_bytes_from_native` is the hit rate. The `coherence` block explains
  misses: `mismatched` means the file changed between sessions, `absent` means
  there was no stamp, and `ambiguous` means the stamp was withheld on purpose.

**The first read after a write is always native.** At close, the connector
stamps the file's identity so the next open can trust the cache. It withholds
that stamp while the file's mtime is younger than `CLIO_STAMP_GRANULARITY_NS`
(10 ms; 1 s on Windows), because timestamps that recent cannot rule out a later
write. A writing session always closes inside that window. So the first
read-back is native, its close stores a stamp, and the session after it can hit.
A benchmark that writes once and reads once will correctly report 0%.

## When it does not work

| Symptom | Cause and fix |
|---|---|
| `H5Fopen` or `H5Pset_vol` fails with no explanation | The `.so` cannot be loaded. `ldd` it: usually the wrong `libhdf5`, or glibc from a container. |
| Everything works, nothing is cached | No runtime, or no CTE pool composed. Check `CLIO_SERVER_CONF`; set `CLIO_REQUIRE_RUNTIME=1` to make it an error. |
| First open stalls for 30 s | The runtime attach is waiting for a server. Set `CLIO_WAIT_SERVER=0` when you expect none, or turn the cache off. |
| Process aborts before your code runs | `CLIO_SERVER_CONF` names a missing or unparseable file. A missing runtime is survivable; a broken config is not. |
| Results vary between runs | More than one `clio_run` is up. Stop them all and start one. |

## Environment reference

| Variable | Default | Effect |
|---|---|---|
| `HDF5_PLUGIN_PATH` | — | Directory containing the connector `.so` files |
| `HDF5_VOL_CONNECTOR` | — | `clio` selects the VOL connector |
| `HDF5_DRIVER` | — | `clio_vfd` selects the VFD |
| `CLIO_SERVER_CONF` | `~/.clio/clio.yaml` | Runtime config; must compose a `clio_cte_core` pool |
| `CLIO_REQUIRE_RUNTIME` | off | Fail the open instead of silently falling back |
| `CLIO_WAIT_SERVER` | 30 | Seconds to wait for a runtime at attach; `0` does not wait |
| `CLIO_VOL_CACHE` | on | `0` makes the VOL a pure pass-through; no runtime needed |
| `CLIO_VFD_CACHE` | on | `0` makes the VFD native-only; no runtime needed |
| `CLIO_VFD_READ_TIER` | off | `1` lets the VFD populate and read from the tier |
| `CLIO_VOL_TRACE` | — | Directory for VOL access telemetry |
| `CLIO_STAMP_GRANULARITY_NS` | 10 ms | Raise it on filesystems with coarse timestamps (NFS, FAT) |

Tuning knobs (chunk size, admission policy, sieve window, fsync on flush) are
in the per-connector READMEs.

## Limits

- The cache is not coherent across processes. A file modified by another
  process *while open* is undefined; between sessions, the stamp catches it.
- h5py drives dataset I/O through the VOL fine, but some of its other paths
  assume the native VOL. The compat suite exercises those through the C API.
- The VOL tier is a cache, not storage: when it fills, it evicts its coldest
  images.
