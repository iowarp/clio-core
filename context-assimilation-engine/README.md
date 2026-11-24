# Context Assimilation Engine (CAE)

A CLIO ChiMod for high-performance data ingestion into the IOWarp ecosystem.
CAE assimilates data from external sources — local binary files, HDF5
datasets, Globus endpoints, and cloud object stores (Amazon S3 and Google
Cloud Storage) — into the [Context Transfer
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

**Supported formats:** `binary`, `hdf5`, `hdf5_vol`, `globus`

**Supported source schemes:** `file::`, `string::`, `hdf5::`, `globus://`,
`s3://` (Amazon S3 / MinIO), `gs://` and `gcs://` (Google Cloud Storage),
`dataverse://` (files and zip members of a Dataverse dataset), `http://` and
`https://` (a web file; to `iowarp::`, also the members of a web zip)

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

S3 ingestion is opt-in via `-DCAE_ENABLE_S3=ON` and requires the AWS SDK for
C++ (the same dependency the bdev S3 transport uses). GCS ingestion is opt-in
via `-DCAE_ENABLE_GCS=ON` and requires `google-cloud-cpp` (storage component).
Both default to OFF and self-disable with a warning if their SDK is not found,
so a default build is unaffected.

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
    depends_on: ""                  # src or dst of an EARLIER transfer (optional)
    range_off: 0                    # Byte offset in source (optional)
    range_size: 0                   # Bytes to read, 0 = full file (optional)
    src_token: $GLOBUS_TOKEN        # Auth token, env vars expanded (optional)
```

Transfers run in the order listed, and each one finishes before the next
starts. `depends_on` documents and checks that order: it must be the exact `src`
or `dst` URL of an earlier transfer, for example a `file::` ingest that depends
on the `globus://` download staging that file. `ParseOmni` rejects the whole OMNI
file, before running any transfer, if a `depends_on` names a later transfer,
the transfer itself, or nothing in the list.

**HDF5 with dataset filtering:**

```yaml
transfers:
  - src: hdf5::/data/experiment.h5
    dst: iowarp::experiment
    format: hdf5
    dataset_filter:
      include_patterns:
        - "/sensors/*"
      exclude_patterns:
        - "/sensors/raw/*"
```

**HDF5 staged for the clio HDF5 VOL connector (`hdf5_vol`):**

```yaml
transfers:
  - src: hdf5::/data/experiment.h5
    dst: "iowarp::"                    # no tag name: the connector derives it
    format: hdf5_vol
```

`hdf5` gives each dataset a tag of its own, with a `description` blob
(`tensor<float64, 100, 3>`; a table of compound records, such as a PyTables
`Table`, is `compound<record bytes, rows>{field:type@offset, ...}` over its
packed records) and, when the dataset has HDF5 attributes, an `attributes`
blob holding them as one JSON object. The root group's attributes go to an
`attributes` blob on the destination tag itself. `hdf5_vol` instead writes the file
in the cache layout of CTE's HDF5 VOL connector
(`context-transfer-engine/adapter/hdf5_vol/clio_vol_layout.h`): the native-type
image of every atomic dataset under the connector's tag for the file, then the
coherence stamp that lets the connector trust it. An application that later
reads the file through the connector is served from the CTE tier from its
first read, including hyperslab and MPI-collective reads, instead of reading
the file. The chunk size is `CLIO_VOL_CHUNK_SIZE` in the runtime's environment
and must match the application's; a mismatch is a cache miss, never a wrong
read. `dataset_filter` applies as for `hdf5`.

**Cloud object stores (S3 / GCS):**

```yaml
transfers:
  - src: s3://my-bucket/data.bin       # Amazon S3 / MinIO (s3:// or s3::)
    dst: iowarp::my_tag
    format: binary
  - src: gs://my-bucket/data.bin       # Google Cloud Storage (gs:// or gcs://)
    dst: iowarp::my_other_tag
    format: binary
    range_off: 0                       # optional ranged read
    range_size: 0
```

Credentials and endpoints are resolved from the standard cloud environment at
assimilation time (no secrets are placed in the OMNI file):

- **S3:** `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`
  (or a `~/.aws/credentials` profile via `AWS_PROFILE`), `AWS_DEFAULT_REGION`,
  and an optional S3-compatible endpoint via `S3_ENDPOINT` or
  `AWS_ENDPOINT_URL` (e.g. MinIO). Public buckets (e.g. the AWS Open Data
  registry) are read unsigned, like `aws s3 --no-sign-request`: set
  `AWS_NO_SIGN_REQUEST=1`, or configure no credentials at all and CAE falls
  back to anonymous. A region is still required. Requests go through
  `https_proxy` / `http_proxy` (minus `no_proxy`; loopback is always direct)
  as with curl, which HPC compute nodes need to reach AWS. These variables are
  read by the runtime process that runs the assimilator, not by `clio_cae`.
- **GCS:** Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS`,
  gcloud ADC, or the GCE/GKE metadata server), and an optional endpoint via
  `GCS_ENDPOINT` (e.g. fake-gcs-server).

**Dataverse datasets (`dataverse://`):**

```yaml
transfers:
  - src: dataverse://dataverse.harvard.edu/doi:10.7910/DVN/LCCLLW
    dst: iowarp::golf                  # or file::/some/dir to only stage
    format: hdf5                       # HDF5 files per dataset; others binary
    dataset_filter:
      include_patterns:
        - "Documentation.zip/*.csv"          # members of one zip file
        - "P*.zip/*_stream_data.hdf5"        # one kind of member of many zips
        - "hdf5:*/data"                      # datasets inside those HDF5 files
      exclude_patterns:
        - "*.mp4"
```

CAE lists the dataset with the Dataverse API and selects objects by path:
`<directoryLabel/>filename` for a file of the dataset and
`<file>.zip/<member>` for a member of a zip file. A pattern containing
`.zip/` selects members, read straight out of the remote zip with HTTP range
requests (ZIP64, stored or deflated, CRC-checked), so a few files can be taken
from a large archive without downloading the rest. Patterns starting with
`hdf5:` pick datasets inside the selected HDF5 files. Without include patterns
every file is taken whole.

- **Requests:** each data file's download link is resolved once and reused for
  all its ranged reads, so the Dataverse server sees one request per file, not
  one per member. Throttled or failed requests are retried with backoff.
- **Staging:** objects are staged in `$CAE_DATAVERSE_CACHE` (default
  `~/.clio/cae_dataverse/<pid>`) and reused by a later run when the size
  matches (`CAE_HTTP_OVERWRITE=1` refetches). `CAE_DATAVERSE_JOBS` (default 8)
  objects are fetched at a time.
- **Tags:** each object becomes `<tag>/<path>` (and `<tag>/<path>/<dataset>`
  for HDF5); the tag `<tag>` gets a `manifest` blob listing every object.
- **Credentials:** `src_token` is sent as `X-Dataverse-key` to the Dataverse
  host only (drafts, restricted files); other headers come from the http config
  file below.
- **Without a runtime:** `cae_dataverse_tool list|stage <src> [dir] -i <glob>
  -x <glob>` lists a selection or stages it.

**Web files and zip archives (`http(s)://` to `iowarp::`):**

```yaml
transfers:
  - src: https://example.org/releases/data_release.zip
    dst: iowarp::release
    format: hdf5
    dataset_filter:
      include_patterns:
        - "*.zip/posterior.h5"       # members of the web zip
        - "*.zip/*.py"
```

To clio-core, a web URL is staged like a Dataverse dataset of one file named
after the URL's last path segment: zip members are selected with the same
patterns and range-read out of the remote zip (the server must support range
requests), HDF5 files go through the `hdf5` assimilator, and the result is
listed in the `manifest` blob of the destination tag. The staging directory is
`$CAE_DATAVERSE_CACHE/web_<host>_<path>`. To `file::<path>` the URL is still
downloaded whole, as before.

**Transforming a source first (`run:`):**

```yaml
transfers:
  - src: file::/data/A46_xx.csv
    dst: iowarp::a46_parquet
    format: binary
    run: to_parquet.sh                  # called as: to_parquet.sh <src> <run_out>
    run_out: ${SCRATCH}/a46.parquet     # the file the script writes; it is ingested
```

`clio_cae` runs each `run:` script before it submits anything. The scripts run
in order, as the user running `clio_cae`, and never inside the runtime.

- **Arguments:** the script gets two. The first is the local path for `file::`
  and `hdf5::` sources, and the URL itself for any other scheme, so a script
  can fetch remote data on its own. The second is `run_out`.
- **Success:** the script must exit 0 and leave a regular file at `run_out`. An
  existing `run_out` is deleted first, so a file from an earlier run can't count
  as this run's output.
- **What gets ingested:** the transfer then ingests `run_out`, as `hdf5::` when
  `format` starts with `hdf5` and as `file::` otherwise. `range_off` and
  `range_size` apply to that output. A later `depends_on` that named the old
  `src` is updated to match.
- **Failure:** if any step fails, no transfer is submitted.
- **Paths:** `run` and `run_out` expand `${VAR}`. Relative paths resolve
  against the OMNI file's directory.
- **Visibility:** `run_out` must be readable by the runtime process, so on
  several nodes put it on a shared file system.
- **Windows:** the script runs through `cmd /C`, so `.bat` and `.cmd` files
  work.

**Key fields:**

| Field | Description |
|---|---|
| `src` | Source URL (`file::`, `string::`, `hdf5::`, `globus://`, `s3://`, `gs://`, `dataverse://`) |
| `dst` | CTE destination tag (`iowarp::tag_name`; `iowarp::` for `hdf5_vol`) |
| `format` | `binary`, `hdf5`, `hdf5_vol`, or `globus` |
| `range_off` / `range_size` | Byte range within source (0 = full file) |
| `src_token` / `dst_token` | Auth tokens; environment variables are expanded |
| `dataset_filter` | HDF5 dataset include/exclude glob patterns; for `dataverse://`, file and zip-member patterns |
| `run` / `run_out` | Script that turns `src` into `run_out` before ingest (see above) |

A per-transfer key not in this table is logged as a warning and ignored.

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
