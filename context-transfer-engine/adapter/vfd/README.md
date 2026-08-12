# CLIO HDF5 VFD (`H5FDclio`)

A [Virtual File Driver](https://portal.hdfgroup.org/display/HDF5/Virtual+File+Drivers)
that writes every byte through to an **authoritative on-disk native HDF5 file**,
while optionally populating a CLIO CTE cache tier alongside it.

The native file is the source of truth. Standard tools (`h5dump`, `h5ls`,
`h5repack`, `h5diff`) read it live, with or without CLIO running, because the
driver adds no superblock driver-info message — the artifact is a plain native
HDF5 file. Reads are served from it; writes are committed to it synchronously
before the call returns. The CTE tier is **populate-only** today: writes push
into it, reads never come from it. Turning it into a read-serving tier is scoped
in `translation/VFD_2.1_READ_CACHE_SCOPING.md` and is not implemented.

Files: `H5FDclio.cc`, `H5FDclio.h`. Driver name: **`clio_vfd`**. Driver value: `3200`.

## Requirements

HDF5 **>= 1.14** — the driver uses the multi-version `H5FD_class_t` API
(`H5FD_CLASS_VERSION`, `read_vector`/`write_vector`). Point the build at a
suitable install with `-DHDF5_ROOT=<prefix>` or `CMAKE_PREFIX_PATH`.

**HDF5 linkage matters.** A dynamically loaded driver is ABI-coupled to the
`libhdf5` of the application that loads it. Confirm with:

```sh
ldd <build>/bin/libclio_vfd.so | grep -i hdf5
```

## Building

```sh
cmake -S <repo> -B <build> -DCLIO_CTE_ENABLE_VFD=ON
cmake --build <build> --target clio_vfd -j
# -> <build>/bin/libclio_vfd.so   (installs to <prefix>/lib)
```

## Using it

### Method 1 — linked into the application

Include `H5FDclio.h`, link `libclio_vfd.so`, and select the driver on a FAPL:

```c
herr_t H5Pset_fapl_clio(hid_t fapl_id, hbool_t cache_enabled);
herr_t H5Pget_fapl_clio(hid_t fapl_id, hbool_t *cache_enabled /*out*/);
```

`cache_enabled` is the only configuration option:

- `true` — populate the CTE cache tier on write. Requires a reachable CLIO
  runtime; if none is found the file opens native-only and a warning is logged.
- `false` — native-only. No CTE traffic, no write amplification, and **no CLIO
  runtime required**. In this mode the driver is a complete, correct HDF5 driver
  on its own.

```c
hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
H5Pset_fapl_clio(fapl, /*cache_enabled*/ 1);
hid_t f = H5Fcreate("/tmp/data.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
```

### Method 2 — loaded by environment variable (no source changes)

```sh
export HDF5_PLUGIN_PATH=<build>/bin
export HDF5_DRIVER=clio_vfd
./my_hdf5_app
```

**Known gap:** the driver does not yet parse `HDF5_DRIVER_CONFIG`, so
`cache_enabled` cannot be set on this path (it uses the default, cache on).
This path is also not yet covered by the test suite. Both are tracked as Q2.1
work in `translation/VFD_VOL_PLAN.md` ("What Q2.x still has to build").

## File naming

**Pass plain filesystem paths** (`/tmp/data.h5`). A `clio::`-marked name is
**refused** with an error; the marker is CLIO-internal and the driver adds it
itself where the CTE tag namespace wants it.

This is a contract, not a preference. The point of this driver is that an
unmodified application gains CLIO by setting `HDF5_DRIVER=clio_vfd` — and a
filename that has to be rewritten is a source edit, at every call site. CLIO is
selected by the driver setting or `H5Pset_fapl_clio`, never by the filename.

Refusing the marker is also what makes two feature flags honest.
`H5FD_FEAT_POSIX_COMPAT_HANDLE` and `H5FD_FEAT_DEFAULT_VFD_COMPATIBLE` both
promise HDF5 that the name it holds is a real path it can `stat()`. HDF5 keeps
whatever name it was given regardless of what the driver does internally, so
while a marked name could arrive, `H5F__build_actual_name` would `stat()` a
non-path and `H5Fopen` would abort. Both flags are now advertised.

## Configuration reference

| Knob | Where | Default | Effect |
|---|---|---|---|
| `cache_enabled` | `H5Pset_fapl_clio` | on | Populate the CTE tier on write |
| `HDF5_DRIVER=clio_vfd` | env | — | Select the driver without source changes |
| `HDF5_PLUGIN_PATH` | env | — | Where HDF5 looks for the `.so` |
| `CLIO_VFD_DEBUG` | env | off | Print every read/write addr+size to stderr |
| `CLIO_VFD_MAX_IO_BYTES` | env | 1 GiB | Largest single `pread`/`pwrite`; larger transfers are split and resumed. Lower it to exercise the multi-pass path in tests |
| `CLIO_CLIENT_RETRY_TIMEOUT` | env | 60s | How long the runtime attach retries before giving up. `0` fails immediately — worth setting when you expect no runtime |

## Runtime availability

The driver attaches to the CLIO runtime **once per process**, on the first open
that wants the cache, and remembers the result. Three cases:

- **`cache_enabled=false`** — no attach is attempted at all. The driver is a
  complete HDF5 driver on its own.
- **Runtime absent** — the attach fails and the file opens native-only, with a
  warning. Everything still works; nothing is cached. Note the attach costs
  `CLIO_CLIENT_RETRY_TIMEOUT` seconds before it gives up, paid once per process.
- **Runtime *misconfigured*** — e.g. `CLIO_SERVER_CONF` pointing at a file that
  cannot be parsed. This is **fatal inside the runtime client**: the process
  aborts before the driver can decide anything. The driver can survive a
  missing runtime; it cannot survive a broken configuration.

A runtime that starts *after* the first open is not picked up for the lifetime
of the process — files stay native-only, which is correct but unaccelerated.

## Durability

- `H5Fflush` / `H5Dflush` → `fsync` on the authoritative file. Fail-closed: a
  flush that did not reach disk returns an error rather than reporting success.
- `H5Fclose` → `fsync` then `close`, both checked. After a successful close the
  on-disk file is a complete native HDF5 image and CLIO holds nothing needed to
  read it.
- Any driver failure either falls through to the native path or returns a real
  HDF5 error with a populated error stack. It never reports success for data that
  was not committed.

## Testing

```sh
ctest --test-dir <build> -R vfd --output-on-failure
```

`clio_cte_vfd_unit_tests` covers the rich round-trip, reopen+append, a
differential against `sec2` as oracle, partial hyperslab overwrite, two files
open at once, flush/`get_handle`, `flock` exclusion, fail-closed error reporting,
on-disk size parity with `sec2`, SWMR, the driver FAPL, vectored I/O, `H5Fdelete`
of both stores, and the no-pending-dirty-state flush barrier.
CI: `.github/workflows/ci-vfd.yml`.

For the current gap list — including what this suite does **not** cover
(multi-process, crash consistency, >2 GiB transfers, the env-var path) — see
`translation/VFD_VOL_PLAN.md` ("What Q2.x still has to build").
