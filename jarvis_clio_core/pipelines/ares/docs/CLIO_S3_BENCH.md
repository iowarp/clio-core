# CLIO vs Zarr — S3 benchmarks on Ares (Issue #968)

Measures how CLIO performs as an intermediary between real Amazon S3 and a
compute node, in **both directions**, against the incumbent cloud-native array
format (Zarr) and — on the write side — against a raw-PUT wire-speed floor.

Two sweeps, one package set:

| | read | write |
|---|---|---|
| Pipeline | [`clio_s3_read.yaml`](../clio_s3_read.yaml) | [`clio_s3_write.yaml`](../clio_s3_write.yaml) |
| CLIO path | CAE assimilator: `ParseOmni` → fork+exec `cae_s3_tool get` → CTE `PutBlob` | `AsyncPutBlob` → CTE → `kS3` bdev `WriteBlocks` → signed PUT from the runtime daemon |
| Jarvis packages | `clio_s3_bench` / `zarr_s3_bench`, `mode: read` | `clio_s3_bench` / `zarr_s3_bench`, `mode: write`, plus `s3_raw_put_bench` |
| Spack variants | `+cae +cte +s3_cae` | `+cae +cte +s3_cae +s3_bdev` |

Writing was out of scope originally: the `kS3` bdev linked the AWS SDK
in-process, which stack-smashes `clio_run` startup. **That is fixed** — the bdev
was reimplemented on Poco + SigV4, mirroring the working GCS transport, and the
AWS SDK is gone from `libclio_bdev_runtime.so` entirely.

Unlike the Issue #526 pipelines (see [PERF_EVAL_BENCH.md](PERF_EVAL_BENCH.md)),
both of these are **non-containerized**: bare-metal binaries from a spack view,
no SIF, no apptainer.

---

## What gets compared

### Read

| | CLIO | Zarr |
|---|---|---|
| Driver | `clio_s3_read_bench` (C++) | `zarr_s3_read.py` (zarr-python + s3fs) |
| Jarvis package | `clio_s3_bench` (`mode: read`) | `zarr_s3_bench` (`mode: read`) |
| Path | `ParseOmni` → `S3FileAssimilator` → fork+exec `cae_s3_tool get` → CTE `PutBlob` | `zarr.open` over `FsspecStore` → `arr[:]` |
| Reads | N flat objects, whole-object GETs | N chunks of a Zarr v3 store |
| Ends up | bytes in a distributed CTE tag | a NumPy array in process memory |
| Compression | none | none |

Both stacks move **the same 2 GiB of logical data** in every row. Across the
granularity axis only the *request count* changes (4096 / 512 / 64 / 8), because
each raw object set is the same 2 GiB buffer re-split.

### Write

| | CLIO | Zarr | Raw floor |
|---|---|---|---|
| Driver | `clio_s3_write_bench` (C++) | `zarr_s3_write.py` (zarr-python + s3fs) | `s3_raw_put.py` |
| Jarvis package | `clio_s3_bench` (`mode: write`) | `zarr_s3_bench` (`mode: write`) | `s3_raw_put_bench` |
| Path | `AsyncPutBlob` → CTE → `kS3` bdev `WriteBlocks` → signed PUT from the runtime daemon | `zarr.create_array` over `FsspecStore` → `arr[:] = data` | K concurrent `cae_s3_tool put` |
| Writes | N blobs, split into `block_<offset>` objects | N chunks of a Zarr v3 store | N flat objects |
| Starts from | bytes in a CLIO shared-memory buffer | a NumPy array in process memory | pre-staged local files |
| Compression | none | none | none |

All three stacks move the same logical bytes in every row, in the same unit.

### Zarr-zstd is deliberately not a comparator

**Nothing in either grid is compressed.** CLIO gets its own compression
mechanism later this year; until it does, comparing an uncompressed CLIO
transfer against a compressed Zarr one measures zstd rather than either system.

Both pipelines therefore pin `variants: ["none"]`, and the `zarr_s3.readzstd.*`
/ `zarr_s3.writezstd.*` columns are gone from the results and from the
`post_cmds` assertions. **Re-enabling is a one-line change per pipeline** —
restore `["none", "zstd"]` and put the zstd column back in `post_cmds` — which
is the intended move once CLIO can compress on its own side of the comparison.

The 2026-08-26 write sweep, which did run zstd, is the evidence for why this
matters. Two numbers from it, kept here as provenance and **not** as a
comparison:

- zstd compressed 1.93× and was link-bound like everything else. Its
  `agg_bw_mbps` reached **20.6** — above the measured 11.1 MB/s link — purely
  because logical bytes exceeded wire bytes. It was the largest number in the
  file and the one most likely to be misquoted as a throughput win over CLIO.
- On the wire it was **10.34 MB/s at K=32, marginally slower than CLIO's 10.11**
  for the same logical payload, having moved 133 MiB against CLIO's 256 MiB.

Note the confound ran in **opposite directions** on the two sweeps: reading,
compression meant Zarr fetched fewer bytes; writing, it meant Zarr sent fewer.
Either way the comparison was about the codec.

The staged read dataset still contains its four zstd stores. Nothing reads them
now; they cost storage, not egress, and leaving them staged is what keeps
re-enabling cheap.

### Read the raw floor first (write side)

The `rawput` row is not a competitor — it is the **bound**. It does the least
possible work: no CTE, no chunking layer, no metadata, no compression, just
concurrent PUTs of files that were staged before the clock started. Nothing in
the comparison should beat it.

Without it a poor CLIO number is uninterpretable. If CLIO is slow *and* rawput
is slow, the bottleneck is the link or the bucket, and no amount of CLIO work
will move it. Only a gap between them is a CLIO finding.

> **Future work, deliberately not done here:** the *read* sweep has no
> equivalent floor, which is the same interpretability gap in the other
> direction. Back-porting a raw-GET floor to `clio_s3_read.yaml` would make
> those numbers attributable the same way. It is a separate change.

### Read the fairness columns, not just the headline

Each driver emits a `Fairness` block alongside its throughput block, so every
results.csv row carries the caveats:

- `agg_bw_mbps` — throughput over **logical** (uncompressed) bytes on every
  stack. This is the directly comparable number.
- `wire_bw_mbps` / `bytes_moved` — what actually crossed the network. With
  every stack uncompressed this should now equal `agg_bw_mbps` on every row, so
  a gap between the two means something other than compression and is worth
  chasing. It remains the column to compare on: it is the one that stays honest
  if a compressed stack is ever added back.
- `objects_read` / `get_count`, `objects_written` / `put_count` — request-rate
  vs bandwidth regime. Note CLIO's write count is derived from block geometry
  (`object_size / block_size`), not from the blob count.
- `compression`, `decode_step` — should read `none` / `no` on every stack in
  every row. They are kept precisely because that is an assertion, not a
  formality: a row where Zarr reports a codec is a row where `variants` was not
  pinned. On the write side `decode_step` is really the *encode* pass, kept
  under the read-side name so one parser key serves both sweeps.
- `subprocess_spawns`, `temp_file_bytes` — **the sharpest contrast between the
  two directions.** Reading, CLIO forks `cae_s3_tool` once per object and stages
  every object whole through node-local disk before it reaches CTE; both are 0
  for Zarr. Writing, both are 0 for CLIO — it signs and PUTs directly from the
  runtime worker — and it is the raw floor that pays one spawn per object by
  construction. These are structural costs of each implementation, not
  measurement noise.
- `runtime_worker_threads` — see the concurrency caveat below.
- `max_rss_kb` — Zarr materializes the whole array in process; CLIO streams it.
  Absent if `/usr/bin/time` is not installed.

**Things that belong in any writeup:**

1. **The end states differ, in both directions.** Reading, CLIO lands bytes in a
   distributed, tiered CTE tag addressable by other CLIO clients while Zarr lands
   a NumPy array in one process's heap; writing, CLIO starts from a CLIO
   shared-memory buffer and goes through that same tiered CTE while Zarr writes
   from one process's heap. CLIO does strictly more work. Zarr's number is *not*
   "CLIO minus overhead."
2. **CLIO's internal read pipelining is not tunable.** `kMaxChunkSize` (1 MiB)
   and `kMaxParallelTasks` (32) are `static constexpr` inside
   `S3FileAssimilator::Schedule`, so object size is the only granularity control
   on the CLIO read side.
3. **Source entropy is currently inert, and will matter again.** The write
   pipeline's `compressibility` (default 0.5) sets how compressible the Zarr
   source data is; with no codec running it changes nothing on the wire. It is
   kept set so that restoring zstd stays a one-line change, and because the
   value has to be stated alongside any future compressed numbers: at `0.0`
   zstd cannot compress at all and in fact slightly *expands* the data, at
   `1.0` it compresses to almost nothing, and neither resembles real scientific
   arrays. On the read side the equivalent knob is the staging script's
   `--pattern`.

### The concurrency caveat (read before interpreting any result)

**Both directions block a runtime worker thread, for different reasons, with
identical consequences.**

- *Read:* `S3FileAssimilator` downloads via `fork()` + **blocking `waitpid()`**
  on a runtime worker thread — not a `CLIO_CO_AWAIT`. The worker is held for the
  entire S3 GET.
- *Write:* `WriteBlocks` is a coroutine body in the bdev, and the signed PUT runs
  to completion inside it.

Either way the effective concurrency ceiling is `clio_runtime.num_threads`, not
the requested `K`. That is why `runtime.num_threads` is swept in lockstep with
the concurrency axis in both pipelines, and why `cpus_per_task` is large.
**Always compare `requested_concurrency` against `effective_concurrency` and
measured scaling before concluding anything.**

If raising K changes nothing on the read side, the worker pool is the ceiling:
raise `runtime.num_threads`, or switch to the multi-process fallback
(`clio_s3.nprocs > 1`, which partitions the key space via `--object-stride` /
`--object-offset`).

On the write side, **compare against `rawput` first, not against K=1** — see the
measured results below for why that rule matters.

---

## What the write sweep measured (2026-08-26)

36 rows: 2 sizes × 6 concurrencies × 3 repeats, all `success`. Wire MB/s, mean of
the three repeats. **This run predates the removal of zstd from the grid** — its
zarr-zstd column is omitted here and summarized under "Zarr-zstd is deliberately
not a comparator" above; nothing else about the run changes, since every figure
below comes from the uncompressed stacks:

| K | \| | CLIO 1M | rawput 1M | zarr 1M | \| | CLIO 4M | rawput 4M | zarr 4M |
|---|---|---|---|---|---|---|---|---|
| 1  | | 2.51 | 1.72 | 4.14 | | 6.13 | 4.76 | 4.84 |
| 4  | | 2.60 | 4.04 | 9.49 | | 6.33 | 7.38 | 10.73 |
| 8  | | 3.41 | 5.46 | 10.75 | | 7.64 | 9.18 | 11.08 |
| 16 | | 3.92 | 7.75 | 10.87 | | 9.27 | 10.46 | 11.10 |
| 32 | | 4.76 | 9.91 | 10.90 | | 9.91 | 10.83 | 11.06 |
| 64 | | 4.87 | 9.95 | 10.83 | | 10.57 | 11.06 | 10.93 |

**The K=64 rawput point settles the ceiling question.** rawput moves +0.3% (1
MiB) and +2.2% (4 MiB) from K=32 to K=64 — flat. rawput forks K processes and
uses no runtime worker, so a per-connection concurrency limit would still be
climbing there. It is the **link**, ~11.1 MB/s, and nothing in CLIO can beat it.
Report **ratio to floor**, which is a property of CLIO; the absolute MB/s is a
property of the night you ran it.

**At 4 MiB CLIO converges on the floor: 0.96× at K=64**, having climbed 0.86 →
0.83 → 0.89 → 0.92 → 0.96.

**At 1 MiB it does not.** CLIO plateaus at 4.87 MB/s — **0.49× the floor** —
while rawput and zarr both reach ~10.9. *CLIO well below rawput ⇒ CLIO's own
ceiling*, and this is that case. It is a **per-object** ceiling, not a bandwidth
one: CLIO saturates at ~5.5 objects/s, worth 5.5 MB/s at 1 MiB but 22 MB/s at
4 MiB — above the link, which is exactly why the 4 MiB rows look healthy and
hide it. **This defect is open.**

The K=1 latency fit says the fixed cost is not the problem. Fitting
`latency = fixed + size/rate` through the two K=1 points:

| stack | fixed | marginal rate |
|---|---|---|
| CLIO | 313 ms | 11.96 MB/s |
| rawput | 497 ms | 11.64 MB/s |

Both see the same ~12 MB/s link, and CLIO's *fixed* per-object cost is the
**lower** of the two — 313 ms against the floor's 497 ms of fork+exec plus temp
file. That is why CLIO beats the floor at K=1 (1.47× at 1 MiB, 1.29× at 4 MiB).
CLIO's problem is that ~180 ms of that per-object work does not pipeline across
concurrency, where the floor's does. Compare the scaling K=1→64: rawput 5.8×,
CLIO 2.2×.

The oversubscription check comes back **clean**: at K=64
(`runtime.num_threads: 64` on `cpus_per_task: 40`) CLIO does not dip below K=32
at either size — 4.76 → 4.87 and 9.91 → 10.57.

**Client memory is a clear CLIO win, by ~6×.** At 4 MiB / K=64: CLIO 266 MB
against zarr's 1664 MB. CLIO's K-slot SHM window grows as K × object_size and
nothing else; zarr materializes the whole 1 GiB array in-process. rawput is flat
at 21 MB only because its bytes live in a temp file — `temp_file_bytes` reaches
256 MiB at K=64, so it moved the cost to disk rather than avoiding it.

Run-to-run spread over the 3 repeats: zarr is the steadiest (median 0.3% CV),
CLIO and rawput median ~3% with occasional 16% outliers — shared-uplink weather,
which is what `repeat: 3` is for.

---

## Addressing: the key prefix is mandatory (write)

CTE registers each target as `device.path_ + "_node<N>"`. For a cloud device
that suffix lands on the **path string**, so:

```
s3://bucket/clio-s3-write-bench/bdev   ->  s3://bucket/clio-s3-write-bench/bdev_node0
```

which is exactly right — it gives free per-node key isolation. But:

```
s3://bucket                            ->  s3://bucket_node0
```

**corrupts the bucket name.** A bare bucket with no prefix will fail against a
bucket that does not exist, or worse, silently target one that does. Always
configure a prefix.

---

## One-time setup

### 0. Build IOWarp and expose it as a view

The read sweep needs `+cae +cte +s3_cae`; the write sweep additionally needs
**`+s3_bdev`** (Poco + SigV4). These are different features that share the word
"S3": `+s3_cae` gates the CAE assimilator and `cae_s3_tool`, `+s3_bdev` gates the
`kS3` block device. The write sweep needs **both**, because the raw-PUT floor
uses `cae_s3_tool`. Building everything at once covers both sweeps:

```bash
spack install iowarp@dev +cae +cte +s3_cae +s3_bdev
spack view --dependencies no symlink /mnt/common/$USER/iowarp-s3-view iowarp@dev
export IOWARP_VIEW="/mnt/common/$USER/iowarp-s3-view"
```

Alternatively, to build a local checkout without a recipe edit:

```bash
spack develop -p "$HOME/clio-core" iowarp@dev
spack install iowarp@dev +cae +cte +s3_cae +s3_bdev
```

Verify the gates actually took — the root `CMakeLists.txt` silently turns
`CAE_ENABLE_S3` back **off** if `find_package(AWSSDK)` fails, in which case the
build succeeds with no S3 support at all:

```bash
ls "$IOWARP_VIEW/bin/clio_s3_read_bench" \
   "$IOWARP_VIEW/bin/clio_s3_write_bench" \
   "$IOWARP_VIEW/bin/cae_s3_tool"
```

All three must exist. Note `aws-sdk-cpp` is unpinned in the recipe and builds all
components by default — expect 30–60 minutes.

Three traps worth knowing before you spend a build on them:

- **`spack view symlink` never overwrites existing links.** With another
  `iowarp` already in the view it logs conflicts and *skips* them, so the
  refresh looks successful while the view keeps serving the OLD install.
  `spack view rm` first, then symlink, then assert the view's `.so` resolves
  into the prefix you just built.
- **Branch versions do not rebuild on new commits.** Spack's hash for a git
  branch does not change when the branch moves, so `spack install` reports it
  already installed and silently skips the compile. Use
  `spack uninstall -y <spec> && spack clean -s && spack install <spec>` — the
  `spack clean -s` is required, or it re-clones the old commit.
- **The write pipeline asserts the build for you** in `pre_cmds`: it fails fast
  if `libclio_bdev_runtime.so` lacks Poco NetSSL, or if it still links the AWS
  SDK.

### 1. Zarr venv

```bash
python3 -m venv "$HOME/zarr-venv"
"$HOME/zarr-venv/bin/pip" install 'zarr>=3' s3fs numpy
export ZARR_VENV="$HOME/zarr-venv"
```

One venv serves both sweeps. The write pipeline also uses its `s3fs` for the
post-run purge, so it must be importable. Do **not** try to reuse
`~/zarr_benchmarks`'s environment: it pins `requires-python >=3.13` and an
unresolvable local path dependency.

### 2. AWS credentials

Long-lived IAM keys in `~/.aws/credentials`, mode 600, under a named profile:

```ini
[clio-bench]
aws_access_key_id = ...
aws_secret_access_key = ...
```
```bash
chmod 600 ~/.aws/credentials
export S3_BENCH_BUCKET=my-bucket
export S3_BENCH_PROFILE=clio-bench
export S3_BENCH_REGION=us-east-2
```

**No secrets are stored in the YAML** — only profile and region names.
Short-lived STS/SSO tokens are a poor fit: the full grids run longer than a
typical 1-hour token lifetime.

**The two sweeps consume those credentials differently, and this is the single
most common way a run fails.**

- *Read:* `cae_s3_tool` resolves `AWS_PROFILE` through the AWS **C++** SDK's
  credential chain, and `s3fs` resolves it through botocore. A named profile is
  the one mechanism both honor with no code changes.
- *Write:* the process that signs is the `clio_run` **daemon**, and the Poco
  SigV4 signer reads **raw environment variables only** — it has no profile
  support at all. The pipeline's `pre_cmds` therefore resolve the profile to keys
  at job time. Ares has **no AWS CLI**, so `aws configure export-credentials` is
  unavailable; the credentials are parsed out of `~/.aws/credentials` with
  stdlib `configparser`.

Exporting them is **not** enough on its own — the daemon does not inherit the job
script's environment. The `clio_runtime` package's `forward_env` option carries
the names listed in the pipeline into the runtime's environment. See the
troubleshooting entry for the full mechanism; get this wrong and every `PutBlob`
fails with `rc=11`.

`S3_BENCH_REGION` is **mandatory for the write sweep and must be the bucket's
real region** — there is deliberately no `us-east-1` default. **SigV4 is
region-scoped**, and a mismatch is an HTTP **301/400**, not a 403 — an unhelpful
error to debug from the runtime log. `pre_cmds` verifies it against
`GetBucketLocation` rather than trusting it, because botocore silently follows
the redirect and the bdev's signer does not.

### 3. Stage the read dataset (once, ~17 GiB, from a host with egress)

The write sweep needs no staging — it creates the data it writes, and the bucket
only needs to exist and be writable. The read sweep needs a dataset:

```bash
"$ZARR_VENV/bin/python3" \
  "$CLIO_REPO/jarvis_clio_core/scripts/stage_s3_read_bench_data.py" \
  --bucket "$S3_BENCH_BUCKET" --prefix clio-s3-read-bench \
  --region "$S3_BENCH_REGION"
```

Writes a 1024³ uint16 array (2 GiB) as 8 Zarr v3 stores (chunk edges
64/128/256/512 × none/zstd) plus 4 flat-object sets at matching sizes, and a
`manifest.json`. **Only the four uncompressed stores are read** now that zstd
is out of the grid; the zstd stores are staged anyway, so re-enabling the
variant needs no re-staging; pass `--only-granularity` / `--only` to trim the
upload if that storage is unwelcome.

Idempotent — the manifest is written last, so a re-run skips completed work and
redoes only partial uploads. Useful flags: `--dry-run`, `--only zarr|raw`,
`--only-granularity 256`, `--force`.

Sanity-check it end-to-end against a local S3-compatible store first if you
like — both the staging script and the Zarr reader accept `--endpoint-url` (or
`S3_ENDPOINT`), and so does `cae_s3_tool`.

`--pattern` shapes the source entropy. It does not affect the current grid —
nothing compressed is read — but it decides what the staged zstd stores are
worth whenever the variant comes back. The default `smooth` compresses ~20–26×
with zstd, close to the zarr_benchmarks reference dataset's 24×; it is
synthetic, so report the ratio (recorded per store in `manifest.json`) rather
than presenting it as a property of real scientific data. `random` is
incompressible and would reduce a compression axis to a measurement of zstd's
CPU cost.

---

## Running

`pre_cmds` expand when the **job** runs, so export overrides *before* submitting.

```bash
export S3_BENCH_BUCKET=my-bucket S3_BENCH_PROFILE=clio-bench S3_BENCH_REGION=us-east-2
export IOWARP_VIEW=/mnt/common/$USER/iowarp-s3-view ZARR_VENV=$HOME/zarr-venv
export CLIO_REPO=$HOME/clio-core JARVIS_VENV=$HOME/jarvis-venv
```

### Read grid (36 rows, ~1.6 h, ~155 GiB egress)

```bash
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_read.yaml"
```

Grid: bytes-per-request {512 KiB, 4 MiB, 32 MiB, 256 MiB} × concurrency
{1, 8, 32} = 12 combinations × `repeat: 3`. Compression is not a sweep axis and
is no longer a comparison either: one uncompressed Zarr pass per row. Output:
`${HOME}/clio_s3_read_results/results.csv`.

### Write sweep (36 rows, ~3.6 h)

```bash
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_write.yaml"
```

Grid: 2 granularities (1 MiB, 4 MiB) × 6 concurrencies (1, 4, 8, 16, 32, 64) ×
3 repeats. Sized for overnight; `time: "08:00:00"` in the scheduler block.
Output: `${HOME}/clio_s3_write_results/results.csv`.

`verify` is **off** here (byte-fidelity is settled; `objects_measured` is the
per-row guard and costs no egress), and `num_objects` is 256 so that K=64 has
several windows of work behind it instead of one. See the header comment in
`clio_s3_write.yaml` for why there is no 16 MiB axis and what the K=64
`rawput` point is there to settle.

### Cost

**Reads dominate.** The read grid moves ~155 GiB of egress ≈ **$14** at
$0.09/GB; raising `repeat` scales that linearly.

Writes are cheap: ingress to S3 is free and PUTs run ~$0.005/1000, so the 36-row
write sweep's ~37k PUTs come to about **$0.18**. Only leftover object storage
accrues, and `post_cmds` purges the write prefix. The one thing that is *not*
free on the write side is `verify`: it re-reads every blob, which is GET egress
(~19 GiB, ~$1.70) — hence off in the sweep.

Object keys are deterministic (`block_<offset>`, `raw_%06d.bin`, zarr chunk
paths), so re-runs overwrite rather than accumulate. Storage does not grow
without bound even if a purge is skipped.

---

## Verifying a run

### Both sweeps

1. **`successful rows: N / 36`** in the `.out` log. A short count means rows
   failed silently.
2. **Check the numbers, not just the color.** `post_cmds` prints
   `GREEN ROWS WITH BLANK THROUGHPUT (== FAILURES):` — it must say `none`. A
   green row with a blank `agg_bw_mbps` is a failure: `_get_stat` is called
   inside a try/except that logs a warning and continues, so a parse failure
   drops the columns silently rather than failing the row.

### Read

3. Cross-check one row by hand: `logical_bytes` should be 2147483648 on both
   stacks, and `objects_read` should equal `get_count` for CLIO.

### Write

3. Required columns: `clio_s3.write.agg_bw_mbps`, `zarr_s3.write.agg_bw_mbps`,
   `raw_put.rawput.agg_bw_mbps`. `objects_written` and `put_count` must be > 0
   on every stack.
4. **`clio_s3.write.objects_measured` equals `num_objects`.** This one is a
   `list` of the bucket prefix rather than a number the benchmark computed, so it
   is the only column a run that wrote nothing cannot fabricate. Zero means the
   row is fiction regardless of what the throughput columns say.

   **More than `num_objects` does not necessarily mean the allocator
   fragmented.** Check `put_count` first: if `put_count == objects_written ==
   num_objects` and only `objects_measured` is high, the allocator was fine and
   the listing picked up **stale objects from an earlier row**. That is what
   the 2026-08-26 sweep hit: all 36 rows share one key prefix
   (`clio-s3-write-bench/bdev`), and every 4 MiB row reported
   `objects_measured: 448` against `num_objects: 256`. The excess was exactly 192
   objects / 201326592 bytes = 192 × 1 MiB — orphaned blocks from the 1 MiB
   rows that teardown's `FreeBlocks` never deleted, constant across all
   eighteen 4 MiB rows rather than accumulating.

   It contaminated nothing but the two `*_measured` columns — every throughput
   figure comes from `logical_bytes` and `wall_time_us`, which the benchmark
   owns — but it blunted this guard from an equality into a lower bound,
   because a row that wrote nothing would still have listed its predecessors'
   objects.

   **Fixed by `purge_prefix` (default on):** the package empties the bdev
   prefix at the start of every row, so the count is exact again. Only that
   prefix is in range — zarr's store sits one level up and rawput's keys in a
   sibling — and an empty prefix is refused outright rather than widening to
   the whole bucket.

   Two consequences worth knowing:

   * **`objects_purged` is a per-row column.** Nonzero means the *previous*
     row leaked that many objects. It is there because the purge fixes the
     measurement, **not the leak**: something in the bdev teardown path is
     still failing to issue a `DeleteObject` per block, and this column is what
     keeps that visible instead of papering over it. A sweep where
     `objects_purged` is 0 everywhere but the first row means the leak is gone.
   * **If the purge fails, it says so and the row still runs.** The log line
     names it, and `objects_measured` reverts to a lower bound for that row —
     compare it against `put_count` rather than `num_objects`.

   To check the prefix by hand — Ares has **no AWS CLI**, so use the zarr venv's
   botocore, which is what the pipeline itself does:

   ```bash
   "$ZARR_VENV/bin/python3" -c '
   import os, botocore.session
   c = botocore.session.get_session().create_client(
       "s3", region_name=os.environ["S3_BENCH_REGION"])
   p = c.get_paginator("list_objects_v2")
   n = sum(len(pg.get("Contents", []))
           for pg in p.paginate(Bucket=os.environ["S3_BENCH_BUCKET"],
                                Prefix="clio-s3-write-bench/bdev"))
   print(n, "objects")'
   ```
5. **`rawput` is the fastest row *at K ≥ 8*, compared on `wire_bw_mbps`.**
   Two qualifications, both learned the hard way on 2026-08-26:

   * **Not at K=1.** The floor forks one `cae_s3_tool` per object and stages
     each through a temp file, so at concurrency 1 it pays `num_objects`
     serialized `fork+exec` calls in the critical path and came back *slower*
     than CLIO (4.72 vs 6.03 MB/s). Its own fairness columns show why —
     `subprocess_spawns: 64`, `temp_file_bytes: 4194304`. At K ≥ 8 that
     overhead pipelines across the concurrent processes and the floor becomes
     honest. It is a floor for **sustained throughput**, not for single-op
     latency.
   * **Compare `wire_bw_mbps`, not `agg_bw_mbps`.** The two agree now that
     every stack is uncompressed, so this costs nothing to honour — and it is
     the habit that kept the comparison honest when `zarr_s3.writezstd` was in
     the grid, where it beat every other stack on `agg_bw_mbps` purely by
     moving roughly half the bytes for the same logical payload.

   If CLIO beats the floor on **wire** bandwidth at high K, *then* something
   is not reaching S3 — check that the CTE tier really is the S3 device and
   not a local fallback, and check `objects_measured`.
6. **Run once with `clio_s3.verify: true`** to prove bytes round-tripped: it
   re-reads every blob through CTE and compares content byte-for-byte. Leave it
   off for timed rows.

---

## Troubleshooting

### Read

| Symptom | Cause |
|---|---|
| `clio_s3_read_bench not on PATH` | IOWarp built without `+s3_cae`, or `AWSSDK` was not found at configure time and `CAE_ENABLE_S3` silently reverted to OFF |
| `Preflight GET failed` | bad credentials/profile, wrong region, wrong bucket, or the dataset was never staged at that prefix |
| CLIO rows blank, Zarr rows fine | the **runtime** could not find `cae_s3_tool`; it forks the helper, so `CAE_S3_TOOL` must be exported in `pre_cmds` (the package's own env does not reach the daemon) |
| `zarr venv broken` | `$ZARR_VENV` missing `zarr`/`s3fs`/`numpy` |
| Raising concurrency changes nothing | the blocking-`waitpid` worker ceiling — raise `runtime.num_threads` or use `clio_s3.nprocs > 1` |
| No `max_rss_kb` column | `/usr/bin/time` not installed; throughput columns are unaffected |
| Disk full under `/tmp` | `TMPDIR` needs `concurrency × object_size` (32 × 256 MiB = 8 GiB) |

### Write

**"Failed to initialize Clio" from the benchmark.** Ares compute nodes run
`ptrace_scope=1`, which blocks the SHM attach path for a detached client. The
pipeline sets `ipc_mode: "ipc"` (unix socket) for this reason — do not change
it.

**HTTP 301 in the runtime log.** Region mismatch. `S3_BENCH_REGION` must be the
bucket's actual region; SigV4 signatures are scoped to it.

**HTTP 403 in the runtime log.** The credential export did not reach the daemon.
Confirm `pre_cmds` printed `credentials exported from [...]`, and that the
profile exists in `~/.aws/credentials`.

**Throughput identical to a RAM tier / suspiciously fast.** The DPE placed blobs
somewhere other than S3. The CTE package must configure **exactly one** device
and it must be the `s3://` one — any local tier present gives the DPE an
alternative.

**`mkdir: cannot create directory 's3:'`.** An older `clio_cte` package that
does not skip cloud paths in its `Mkdir` loop. Pull the branch.

**The bdev link assert fails in `pre_cmds`.** Either the view is stale (see the
`spack view symlink` trap above) or the build lacked `+s3_bdev`.

**Every `PutBlob` fails with `rc=11`.** CTE has no target to place on. `rc` in the
range 11–19 is `10 + alloc_result` from `PlaceBlobBytes`; 11 means allocation
found no viable device. Scroll **up** in the runtime log — the cause is printed
minutes earlier and looks like this:

```
core_config.cc:534 ERROR ParseStorageConfig Config error: Invalid bdev_type 's3'
                     (must be 'file', 'ram', 'hbm', 'pinned', or 'noop')
core_runtime.cc:743 WARNING Create Warning: No storage devices configured
```

That error message is the *old* one — the current build names `'s3', or 'gcs'` in
the same list. So the runtime library predates the s3/gcs allowlist, the S3 tier
was dropped at config-parse time, and CTE came up with zero devices. Note that
this is only a `WARNING`: the pool is created successfully and the failure does
not surface until the first write.

The `pre_cmds` gate greps the compiled-in literal out of
`libclio_cte_core_runtime.so` to catch this before the allocation is spent.

**"mixed IOWarp installs on PATH".** Two different `iowarp` prefixes were
reachable at once — typically a stale `IOWARP_VIEW` supplying `clio_run` while a
freshly built spack prefix supplies `clio_s3_write_bench`. Because
`spack view symlink` never overwrites existing links, refreshing a view that
already has an `iowarp` in it silently keeps serving the old one. The reliable
fix is to skip the view entirely and point `IOWARP_VIEW` straight at the prefix:

```bash
export IOWARP_VIEW=$(spack find --format '{prefix}' iowarp@dev | tail -1)
```

RPATH makes the symlink farm unnecessary, and a prefix has the `bin/` and `lib/`
layout the pipeline expects.

**`S3 bdev: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set`, even though
the job script exported them.** This is the same `rc=11` cascade as above, one
layer down: the bdev fails to initialize, `core_runtime.cc` logs
`Failed to register target ... (error code: 1)` as a **warning**, CTE comes up
with zero devices, and every `PutBlob` returns 11.

Exporting credentials in `pre_cmds` is not enough. Jarvis launches the daemon as
`PsshExecInfo(env=self.env, ...)`, and `self.env` is a dict it builds itself from
`EnvironmentManager.COMMON_ENV_VARS` — a fixed toolchain list (`PATH`,
`LD_LIBRARY_PATH`, `HOME`, `CC`, …) with **no `AWS_*` entry**. The job script's
exports therefore reach jarvis and every benchmark process but never `clio_run`.
The job log names the mechanism:

```
Auto-built environment with N variables (no 'env' field in pipeline)
```

The `clio_runtime` package's `forward_env` option copies named variables from
the submitting shell into the runtime's environment, and this pipeline lists the
AWS names there. Values are never logged — only names, and only whether each was
set. A top-level `env:` dict in the pipeline would also work, but it would put
the secret in a file on disk; `forward_env` reads it from the live shell.

`forward_env` refuses to forward a value containing `$`, a backtick, or a
backslash. The ssh transport emits each variable as an inline `KEY="value"`
prefix and escapes only the double quote, so those characters would reach the
daemon altered — and a corrupted secret is indistinguishable from a permissions
problem at the far end. AWS keys are base64 (`A–Za–z0–9+/=`), so this should
never trigger; if it does, regenerate the credential.

**`clio_s3_write_bench is STALE -- it predates <marker>`.** `spack develop`
builds compile from the working tree, so pulling the branch does **not** rebuild
them, and nothing about the spec, the hash or the view changes to show it. Run:

```bash
spack install iowarp@dev      # dev spec: rebuilds in place
```

For a non-`develop` spec, branch versions never rehash when the branch moves, so
`spack install` reports "already installed" and skips the compile entirely:

```bash
spack uninstall -y iowarp@dev && spack clean -s \
  && spack install iowarp@dev +cae +cte +s3_cae +s3_bdev
```

The gate greps a build stamp (`kBuildMarker` in `clio_s3_write_bench.cc`) out of
the installed binary rather than trusting the spec.

**`S3_BENCH_REGION=... but bucket ... lives in ...`.** SigV4 is region-scoped and
the bdev's signer does not follow redirects, so a wrong region is an HTTP 400/301
on every PUT, from inside a runtime worker. This is easy to miss because
**botocore hides it**: it transparently retries against the correct region, so an
`aws`-style check or a `HeadBucket` preflight goes green while the daemon fails.
The preflight therefore asks S3 for the authoritative answer with
`GetBucketLocation` and refuses to run on a mismatch, naming the export to fix.

**A row is green but `clio_s3.write.objects_measured` is 0.** Nothing reached the
bucket and the throughput columns are fiction. `objects_measured` is a `list`
of the bdev's key prefix taken right after the timed loop — the one column a run
that wrote nothing cannot fabricate. It runs before teardown on purpose:
`FreeBlocks` issues a `DeleteObject` per block, so a count taken later reads
zero even on a healthy run.

**`objects_measured` disagrees with `put_count`.** Not a failure. One `PutBlob`
normally becomes exactly one S3 object: `AllocateFromTarget` hands the whole
request to the allocator, `WriteBlocks` issues one `PutObject` per returned
block, and an unfragmented request gets a single block. More objects than blobs
means the allocator fragmented and split the request.

**`TaskStatModel: failed to open /tmp/clio/models/...`.** Harmless. The runtime
persists a perf model there and logs an error per attempt if it cannot. `/tmp` is
node-local, so creating the directory on the login node does nothing — `pre_cmds`
creates it on the compute node. It does not gate bdev init or `PutBlob`.
