# clio HDF5 VOL connector

A pass-through HDF5 VOL connector that transparently caches HDF5 data in the
clio-core CTE (Context Transfer Engine) while preserving native HDF5 file
semantics. It wraps a native under-VOL: every operation delegates to native, and
whole-dataset `H5Dwrite`/`H5Dread` transfers of atomic datatypes are additionally
mirrored to CTE via async `PutBlob`/`GetBlob`. Loaded transparently through the
standard HDF5 plugin mechanism — no application source changes.

The native file is always written synchronously and remains authoritative, and
its status is this connector's status — a native operation that fails is
reported as a failure, never masked. Only atomic datatypes (integer, float,
enum, bitfield, fixed-length string) are cached — compound/array (mem/file
layout can differ) and vlen/reference (pointers, not bytes) delegate to native.
A transfer is cached only when its **memory datatype matches the size of the
dataset's stored type**, since the cached image is sized by the transfer's type.
**Selection-aware reads:** hyperslab/point reads are served from the cached
linear image via HDF5 gather/scatter when it is already populated (serve-only; a
miss falls back to native). A non-whole write, an `H5Dset_extent`, or a failed
cache operation invalidates the image so reads never see stale data.
**Safe mode:** `H5Fflush`, `H5Dflush` and `H5Fclose` drain all in-flight async
CTE puts before returning, so a successful flush/close is a real durability
barrier for the cache path (no async write outlives it).

**The cache is never an authority.** Any doubt about whether it matches the file
resolves by invalidating it and re-reading native. If the CLIO runtime is
unreachable, or `CLIO_VOL_CACHE=0` is set, the connector degrades to a pure
pass-through and stays correct.

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
export HDF5_VOL_CONNECTOR=clio        # under-VOL is always native (see below)
# a clio_run runtime must be reachable for the CTE cache path;
# without one the connector runs as a pure pass-through
```
Any HDF5 application (h5py, C, tools) then routes through the connector. Files it
writes are valid native HDF5 files readable by standard tools.

### Configuration reference

| Knob | Default | Effect |
|---|---|---|
| `CLIO_VOL_CACHE` | on | `0`/`off`/`false`/`no` disables the CTE tier entirely — pure pass-through, no runtime required |
| `CLIO_VOL_CHUNK_SIZE` | 1 MiB | Blob chunk size for staging dataset images |
| `CLIO_VOL_MAX_SERVE_BYTES` | 1 GiB | Datasets larger than this are not served from cache for partial reads (see below) |
| `CLIO_VOL_TRACE` | unset | Directory for access telemetry (see below) |
| `CLIO_VOL_STAMP_GRANULARITY_NS` | 10 ms | Bound on filesystem timestamp granularity. A coherence stamp is withheld when the file's mtime is younger than this, because mtime cannot then rule out a later same-granule write (see below) |

`chunk_size` can also be set programmatically through `clio_vol_info_t` via
`H5Pset_vol`; it is honoured identically on create and open.

Value-initialize that struct — `clio_vol_info_t info = {};` — before filling in
the fields you care about. It is a plain C struct, so a bare stack instance
holds garbage. Zeroed fields mean "use the default": `chunk_size = 0` resolves
to the default chunk size, and `cache_enabled = CLIO_VOL_CACHE_UNSET` leaves the
tier to the environment. Disabling the cache from code takes an explicit
`CLIO_VOL_CACHE_OFF`, so you can never switch the tier off by forgetting to set
a field.

### Known limitations

- **The under-VOL is always native.** `clio_vol_info_t::under_vol_id` is accepted
  but not used for stacking — the connector cannot currently sit above another
  pass-through connector. Tracked as W12; see `translation/VFD_VOL_PLAN.md` ("Architectural decisions" §3).
- **Partial reads materialise the whole dataset.** Serving a hyperslab or point
  selection reassembles the full cached image and gathers out of it, so cost
  scales with the dataset rather than the selection. `CLIO_VOL_MAX_SERVE_BYTES`
  bounds the damage by falling back to native above the ceiling; narrowing the
  fetch to the chunks a selection touches is the real fix (W10).
- **No capacity limit or eviction.** The tier grows without bound (W11).
- **Cache staleness during a session** (the between-sessions case is closed).
  At close the connector records the file's identity and state --
  `dev:ino:size:mtime` -- as a reserved blob in its own tag, and validates it at
  open. Anything but an exact match drops the tag, so a file replaced or edited
  behind HDF5's back between sessions is detected and the cache is not used for
  it. A missing or unreadable stamp counts as a mismatch: the check is
  fail-closed, so losing the stamp costs a cache miss and never a wrong answer.

  **Timestamp granularity.** For an in-place edit that does not change the
  file's length, `dev`, `ino` and `size` are unchanged by construction, so
  `mtime` is the stamp's only signal — and filesystem timestamps are coarse
  (1 ms on ext4 at HZ=1000; a full second on some NFS mounts and FAT). Two
  writes inside one granule get byte-identical mtimes, so a stamp taken inside
  that window would match a file modified immediately afterwards. The stamp is
  therefore **withheld** when the file's mtime is younger than
  `CLIO_VOL_STAMP_GRANULARITY_NS` (default 10 ms): any stamp already stored is
  dropped, and the next open sees no stamp and fails closed. This is the same
  do-not-trust rule the absent and unreadable cases already follow. The cost is
  a cache miss when a file is reopened within the window of its last write; the
  `coherence.ambiguous` counter in the access telemetry reports how often that
  happens, so a benchmark can attribute the miss rather than guess at it. Raise
  the bound on a coarse-timestamp filesystem — that is also the setting where
  storing a content hash at close would start to earn its cost.

  What this does **not** cover is modification *while the file is open*. That is
  the multi-process problem and is explicitly undefined here — see the §1.6
  concurrency matrix. Do not read the closed between-sessions case as meaning
  external modification is handled in general.

  The finding that drove this: a dataset was written, the file was damaged on
  disk so its stored bytes no longer matched their checksum, and the dataset was
  read back. With caching on, the read *succeeded* and returned the original
  values from the staged copy — the file was corrupt and the application was
  told everything was fine. The cache did not merely go stale, it **masked the
  authoritative file's own error**, inverting the property the design rests on.
  That case is now covered: the stamp mismatches, the tag is dropped, and the
  read fails exactly as native does. It is pinned by
  `corrupt_checksum_read` in the error-parity axis of the compat suite, in both
  cache-on and cache-off modes.

- **No `HDF5_VOL_CONNECTOR` config string** — `info_cls.from_str` is not
  implemented, so options come from the environment variables above (W14).

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
  maximally-repeated read selections (the cache/prefetch signal). Plus a file-level
  `coherence` block — `{matched, mismatched, absent, ambiguous}` — recording what
  the stamp said. Every outcome except `matched` shows up in the read counts as
  plain `native`, so without this a low hit rate cannot be attributed: a cold
  file, an evicted stamp, a file genuinely changed between sessions, and one the
  connector declined to trust on timestamp grounds are four different findings.
  `ambiguous` is the self-inflicted one — see the granularity note above.

Filenames carry the pid so concurrent processes (e.g. MPI ranks) don't clobber
each other's output.

This is the data a CLIO-using agent reads to advise tuning (hot datasets, whether
caching is helping, which selections repeat, small-read/per-object-cost patterns).

## Testing and current status

Compatibility is verified by a differential suite (the native stack as the
oracle): each feature case is written/read through the connector and compared to
native across four arms, plus the §1.1 native-compatibility gates — `h5diff`,
structural validity (`h5clear -s` + a clean reopen), `h5dump -pBH`/`h5ls
-vlr`/`h5stat` **output** parity, and error-code parity against native on
negative paths. It is registered as the CTest test `cte_hdf5_vol_compat_suite`
(gated on `CLIO_CTE_ENABLE_HDF5_VOL`; `RUN_SERIAL` because it manages its own
`clio_run`).

The same harness drives the VFD via `--modes vfd`, registered separately as
`cte_hdf5_vfd_compat_suite`. The corpus is shared; what it *means* is not — see
the module docstring.

```bash
ctest -R cte_hdf5_vol_compat_suite --output-on-failure
# or directly:
python3 benchmarks/hdf5-ingest/hdf5_compat_suite.py --modes vol --bin <build>/bin
```

The test reports the connector's real compatibility state rather than masking
gaps, so any future regression fails it. For the authoritative, always-current
status run the test; do not rely on numbers copied into prose.

It carries exactly **one** `--expect-fail` entry, not a blanket allowlist:
`vol/eparity_cacheon/corrupt_checksum_read`. With the cache on, a file damaged on
disk after being read once is served from the staged copy, so the read succeeds
where native fails its fletcher32 check — the external-modification staleness
limitation described above. The same case passes with the cache off. The suite
reports an expected-gap entry that starts passing, so the allowlist cannot
outlive the gap it excuses. As of this writing it is **40/41 — every entry
passing except that one expected gap** — compatible for dataset I/O
(signed/unsigned int, float, compound, enum, array-element, scalar, and
variable-length string datatypes; contiguous, chunked, **compact, external and
virtual/VDS** layouts; shuffle/fletcher32 filters; whole-dataset, hyperslab, and
point selections; extendible datasets with reopen+append), attributes, group/link
structure, modern-API iteration (C `H5Ovisit3`/`H5Literate2`), selection-aware read
caching, Safe-mode `H5Fflush` durability, access telemetry, and error-code parity
with native on negative paths in both cache-on and cache-off configurations.

Feature areas the h5py cases can't reach (h5py has poor non-native VOL support) run
through the C API / a telemetry check, all automated by the suite: iteration
(`vol_c_iteration_test.c`), Safe-mode flush (`vol_c_safeflush_test.c`), selection
caching + partial-write invalidation (`vol_c_selection_test.c`), cache-identity
regressions (`vol_c_cache_identity_test.c` — mem/file datatype mismatch and a
cache surviving `H5F_ACC_TRUNC`, both of which returned wrong data with a success
status), and access telemetry (`telemetry` case — runs a workload under
`CLIO_VOL_TRACE` and checks the summary).
