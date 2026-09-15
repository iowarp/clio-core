# CLIO vs Zarr — S3 benchmarks on Ares (Issue #968)

Measures how CLIO performs as an intermediary between real Amazon S3 and a
compute node, in **both directions**, against the incumbent cloud-native array
format (Zarr) and — on the write side — against a raw-PUT wire-speed floor.

Two sweeps, one package set, both **non-containerized**: bare-metal binaries
from a spack view, no SIF, no apptainer.

| | read | write |
|---|---|---|
| Pipeline | [`clio_s3_read.yaml`](../clio_s3_read.yaml) | [`clio_s3_write.yaml`](../clio_s3_write.yaml) |
| CLIO path | CAE assimilator: `ParseOmni` → `S3FileAssimilator` → in-process persistent Poco+SigV4 GET streamed into CTE `PutBlob` (no fork+exec, no temp file) | `AsyncPutBlob` → CTE → `kS3` bdev `WriteBlocks` → signed PUT from the runtime daemon |
| Jarvis packages | `clio_s3_bench` / `zarr_s3_bench`, `mode: read`, plus `s3_raw_get_bench` (wire-speed floor) | `clio_s3_bench` / `zarr_s3_bench`, `mode: write`, plus `s3_raw_put_bench` |
| Spack variants | `+cae +cte +s3_cae` | `+cae +cte +s3_cae +s3_bdev` |
| Exercises the shared persistent-connection S3 transport? | **Yes** (read side, in-process) — the assimilator reuses `S3RestClient` across objects; the `pre_cmds` CAE link gate refuses a pre-change / AWS-SDK-linked build | **Yes** (bdev) — the `pre_cmds` refuse to run against a pre-keep-alive build |

---

## Quick reference — full sequence

The whole setup, if you accept every default. Each step is explained in detail
below; start there if anything fails or you need to deviate.

```bash
# 0. prerequisites: Ares login node, spack on PATH, a clio-core checkout
export CLIO_REPO="$HOME/clio-core"

# 1. jarvis venv
python3 -m venv "$HOME/jarvis-venv"
"$HOME/jarvis-venv/bin/pip" install -e "$HOME/jarvis-cd"
export PATH="$HOME/jarvis-venv/bin:$PATH"

# 2. build IOWarp with the S3 gates and expose it as a view
spack install iowarp@dev +cae +cte +s3_cae +s3_bdev
spack view --dependencies no symlink "/mnt/common/$USER/iowarp-s3-view" iowarp@dev
export IOWARP_VIEW="/mnt/common/$USER/iowarp-s3-view"

# 3. verify the build actually has what both sweeps need (cheap; do this before
#    burning an allocation on a stale library — see Step 2 below for what each
#    line proves)
ls "$IOWARP_VIEW/bin/clio_s3_read_bench" "$IOWARP_VIEW/bin/clio_s3_write_bench" \
   "$IOWARP_VIEW/bin/cae_s3_tool"
LIB="$IOWARP_VIEW/lib/libclio_bdev_runtime.so"
[ -f "$LIB" ] || LIB="$IOWARP_VIEW/lib64/libclio_bdev_runtime.so"
ldd "$LIB" | grep PocoNetSSL                 # must match
ldd "$LIB" | grep aws-cpp-sdk                # must print NOTHING
grep -acF "S3 keepalive worker=" "$LIB"      # must be >= 1

# 4. zarr venv (both sweeps need it)
python3 -m venv "$HOME/zarr-venv"
"$HOME/zarr-venv/bin/pip" install 'zarr>=3' s3fs numpy
export ZARR_VENV="$HOME/zarr-venv"

# 5. credentials — secrets stay in ~/.aws/credentials (mode 600), only names
#    go in the environment
chmod 600 ~/.aws/credentials
export S3_BENCH_BUCKET=my-bucket S3_BENCH_PROFILE=clio-bench S3_BENCH_REGION=us-east-2

# 6. stage the read dataset (once, ~17 GiB — write needs no staging)
"$ZARR_VENV/bin/python3" \
  "$CLIO_REPO/jarvis_clio_core/scripts/stage_s3_read_bench_data.py" \
  --bucket "$S3_BENCH_BUCKET" --prefix clio-s3-read-bench --region "$S3_BENCH_REGION"

# 7. register the package repo
jarvis repo add "$CLIO_REPO/jarvis_clio_core"

# 8. submit
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_read.yaml"
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_write.yaml"
```

---

## The mental model (read this first)

**Both sweeps now exercise the shared persistent-connection S3 transport
(`S3RestClient`), on different consumers.**

- **Read** goes CAE assimilator → `S3FileAssimilator` → an **in-process**
  persistent Poco+SigV4 GET (the same `S3RestClient` the bdev uses, via
  `BeginGetObject`/`ReadBody`), streamed straight into CTE `PutBlob`, landing in
  a RAM tier. There is no `cae_s3_tool` fork+exec and no temp file on this path
  any more — the AWS SDK is confined to a standalone helper the raw-GET floor
  drives. A leased-per-object `S3Connection` is reused across objects; the proof
  is the daemon's `CAE S3 keepalive TOTAL sockets=.. requests=..` line. The
  read `pre_cmds` CAE link gate refuses a build that still links the AWS SDK or
  predates this change.
- **Write** goes `AsyncPutBlob` → CTE → the `kS3` bdev's `WriteBlocks` →
  a signed PUT issued **from inside the runtime daemon**, over `S3RestClient`.
  This is the transport that used to open a fresh `Poco::Net::HTTPClientSession`
  per call and now holds one persistent, per-worker `S3Connection` instead
  (lock-free: S3 ops run synchronously inside a worker's task body, so a
  worker's connection is never touched concurrently — the same property that
  lets `FsBdevTransport::io_contexts_` skip a lock).

**There used to be two more pipelines — `clio_s3_write_smoke.yaml` and
`clio_s3_read_smoke.yaml` — whose entire job was proving the keep-alive
mechanism worked before spending 3.6 hours on the full write sweep. Both are
gone.** Their smoke came back green on 2026-08-26, every gate and credential
path in it was carried over verbatim into `clio_s3_write.yaml`, and the
mechanism check now lives permanently in that pipeline's own `pre_cmds`: it
`grep`s the installed `libclio_bdev_runtime.so` for the `"S3 keepalive
worker="` log literal and **refuses to run** if the library predates it. A
stale build now fails in seconds at job start instead of producing a 3.6-hour
run that quietly measured the old transport. See
[Step 2](#step-2--verify-the-s3-bdev-build-the-keep-alive-check) for what that
check does and how to run it by hand, and
[What the write sweep measured](#what-the-write-sweep-measured-pre-keep-alive-baseline-2026-08-26)
for why "did keep-alive help" is still an open question the numbers below do
not answer.

---

## Prerequisites

- You are on an **Ares login node**.
- **clio-core is checked out** (this repo). Its path is your `CLIO_REPO`
  (default `$HOME/clio-core`).
- **Spack is installed and on `PATH`.**
- **jarvis-cd is checked out.** Its path is `$JARVIS_CD` below (default
  `$HOME/jarvis-cd`).
- Docker is **not** needed for either sweep — both are bare-metal.

> **Shared-filesystem requirement.** The Spack install tree, the view, and
> both venvs must live somewhere the compute node can see — `$HOME` (NFS) or
> `/mnt/common` on Ares.

---

## Step 1 — Build IOWarp and expose it as a view

The read sweep needs `+cae +cte +s3_cae`; the write sweep additionally needs
**`+s3_bdev`** (Poco + SigV4). These gate different features that share the
word "S3": `+s3_cae` gates the CAE assimilator and `cae_s3_tool`, `+s3_bdev`
gates the `kS3` block device. The write sweep needs **both**, because its
raw-PUT floor uses `cae_s3_tool`. Building everything at once covers both
sweeps:

```bash
spack install iowarp@dev +cae +cte +s3_cae +s3_bdev
spack view --dependencies no symlink "/mnt/common/$USER/iowarp-s3-view" iowarp@dev
export IOWARP_VIEW="/mnt/common/$USER/iowarp-s3-view"
```

Verify the gates actually took — the root `CMakeLists.txt` silently turns
`CAE_ENABLE_S3` back **off** if `find_package(AWSSDK)` fails, in which case the
build succeeds with no S3 support at all:

```bash
ls "$IOWARP_VIEW/bin/clio_s3_read_bench" \
   "$IOWARP_VIEW/bin/clio_s3_write_bench" \
   "$IOWARP_VIEW/bin/cae_s3_tool"
```

All three must exist. `aws-sdk-cpp` is unpinned in the recipe and builds all
components by default — expect 30–60 minutes.

### If you are iterating on the bdev itself, use `spack develop` instead

A plain `spack install` above is right for running the sweeps as a consumer.
If you are changing `S3RestClient` / `S3BdevTransport` and re-running the
write sweep to check the effect, build from your checkout instead — an edit
becomes an incremental recompile of a few objects, not a fresh clone:

```bash
spack env activate clio-s3            # or create one: spack env create clio-s3
spack develop -p "$CLIO_REPO" iowarp@dev    # idempotent
spack remove iowarp || true                 # exactly one iowarp spec in the env
spack add iowarp@dev +cae +cte +s3_cae +s3_bdev
spack concretize -f
spack install
```

Confirm develop actually took — the spec must carry `dev_path` and must
**not** carry a `commit=`:

```bash
spack spec -l iowarp@dev | grep -E 'dev_path|commit='
```

Every time after that:

```bash
cd "$CLIO_REPO" && git pull
spack env activate clio-s3
spack install          # REQUIRED — the pull alone rebuilds nothing
```

If `spack install` reports the spec already installed and skips the build
(develop-spec change detection is not always reliable), force it:
`spack install --overwrite -y iowarp@dev`.

### Traps worth knowing before you spend a build on them

- **`spack view symlink` never overwrites existing links.** With another
  `iowarp` already in the view it logs conflicts and *skips* them, so the
  refresh looks successful while the view keeps serving the OLD install.
  `spack view rm` first, then symlink, then confirm the view's `.so` resolves
  into the prefix you just built (`readlink -f`).
- **Branch versions do not rebuild on new commits.** Spack's hash for a git
  branch does not change when the branch moves, so `spack install` reports it
  already installed and silently skips the compile. Use
  `spack uninstall -y <spec> && spack clean -s && spack install <spec>` — the
  `spack clean -s` is required, or it re-clones the old commit.
- **`spack uninstall` says "matches multiple packages".** Old builds
  accumulate — a pre-`s3_bdev` one, a pinned non-develop one, the develop one.
  Do **not** reach for `spack uninstall --force`: it orphans the environment's
  view. Let garbage collection do it instead: `spack env activate clio-s3 &&
  spack gc -y`. If a spec survives `gc`, another environment still lists it —
  find it with `spack env list` and `spack -e <env> find`, and remove it there.
- **`GLIBCXX_3.4.31` / `CXXABI_1.3.15` undefined references building Poco.**
  This is the recurring `~/.bashrc` contamination — `envs/iowarp/bin` on
  `PATH` from an old install leaking a newer libstdc++ into the build. Remove
  those exports from `~/.bashrc`, then `spack clean -s` and retry. It is not a
  problem with the package.
- The write pipeline's own `pre_cmds` assert most of this build for you
  (install root consistency, the CTE bdev-type allowlist, the bdev's link
  line) — but not the keep-alive tally, which only a build with the
  persistent-connection change carries. Check that yourself first: Step 2.

---

## Step 2 — Verify the S3 bdev build (the keep-alive check)

This is the mechanism the standalone smoke pipelines used to prove before
they were deleted. Run it by hand once after any build that touches the bdev
— it is cheap, and `clio_s3_write.yaml` will refuse to run without it anyway,
so finding out now beats finding out 20 minutes into a job.

```bash
LIB="$IOWARP_VIEW/lib/libclio_bdev_runtime.so"
[ -f "$LIB" ] || LIB="$IOWARP_VIEW/lib64/libclio_bdev_runtime.so"
readlink -f "$LIB"                          # note the spack hash / dev_path

grep -acF "S3 keepalive worker=" "$LIB"     # must be >= 1
ldd "$LIB" | grep PocoNetSSL                # must match
ldd "$LIB" | grep aws-cpp-sdk               # must print NOTHING
```

Use `grep -a`, not `strings`: binutils is not guaranteed on a compute node,
and the view goes on `PATH` ahead of `/usr/bin`, so a missing `strings` binary
can silently read as "no tally" instead of "tool absent." `grep -a` reads the
ELF directly and its exit status distinguishes the two: exit 1 means the
literal is genuinely absent, exit 2 means the read itself failed.

- **`grep` finds nothing (exit 1):** you have the pre-keep-alive transport
  installed. `clio_s3_write.yaml` will refuse to run against it — rebuild
  (Step 1) before submitting.
- **`aws-cpp-sdk` shows up in the `ldd` output:** stop and do not run the
  write sweep. Linking the AWS SDK into `libclio_bdev_runtime.so` stack-smashes
  runtime init. Note `aws-sdk-cpp` legitimately appears in `spack find -d`
  output regardless — `+cae` needs it for the out-of-process `cae_s3_tool`.
  The dependency tree is not the gate; this `ldd` line is.

At teardown, a build with the tally present logs a per-worker reuse summary
to the runtime log:

```
S3 keepalive worker=3 sockets=1 requests=64 reuses=63
S3 keepalive TOTAL sockets=4 requests=256 reuse_ratio=64.00
```

`sockets` is TCP connections opened, `requests` is S3 operations sent over
them. **`sockets == 1` with `requests >> 1` is reuse; `sockets == requests`
means the mechanism is dead** even on a build that has the code — worth
checking in the job log after a run, not just in the binary before one. This
is logged at `kInfo`, not `kDebug`: `HLOG` compiles out anything below
`CTP_LOG_LEVEL` (default `kInfo`), so a `kDebug` line would not exist in a
spack build at all.

---

## Step 3 — Zarr venv

```bash
python3 -m venv "$HOME/zarr-venv"
"$HOME/zarr-venv/bin/pip" install 'zarr>=3' s3fs numpy
export ZARR_VENV="$HOME/zarr-venv"
```

One venv serves both sweeps. The write pipeline also uses its `s3fs` (via
`botocore`, which `s3fs` depends on) for the region preflight and the
post-run purge, and it is the tool both pipelines reach for whenever they need
something an AWS CLI would normally do — **Ares has no AWS CLI.** Do **not**
try to reuse `~/zarr_benchmarks`'s environment: it pins
`requires-python >=3.13` and an unresolvable local path dependency.

---

## Step 4 — AWS credentials

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

**No secrets are stored in any YAML** — only profile and region names.
Short-lived STS/SSO tokens are a poor fit: the full grids run longer than a
typical 1-hour token lifetime.

The key pair needs `s3:ListBucket` **on the bucket** (not just object-level
Get/Put): the write sweep's `EnsureBucket` HEADs the bucket before any block
is written. It also needs `s3:GetBucketLocation` for the write sweep's region
check (below).

**The two sweeps consume those credentials completely differently, and this
is the single most common way a run fails.**

- *Read:* `cae_s3_tool` resolves `AWS_PROFILE` through the AWS **C++** SDK's
  credential chain, and `s3fs` resolves it through botocore. A named profile
  is the one mechanism both honor with no code changes. `AWS_DEFAULT_REGION`
  defaults to `us-east-1` if unset — there is no mandatory check on the read
  side.
- *Write:* the process that signs is the `clio_run` **daemon**, and the Poco
  SigV4 signer reads **raw environment variables only** — it has no profile
  support at all. The pipeline's `pre_cmds` therefore resolve the profile to
  keys at job time with stdlib `configparser` (no AWS CLI available to do it
  for you), and `S3_BENCH_REGION` is **mandatory** — there is deliberately no
  default. **SigV4 is region-scoped**, and a mismatch is an HTTP **301/400**,
  not a 403 — an unhelpful error to debug from the runtime log. `pre_cmds`
  verifies the region against `GetBucketLocation` rather than trusting it,
  because botocore silently follows the redirect and the bdev's hand-rolled
  signer does not — a `HeadBucket`-only preflight can go green while every
  real PUT still gets a 301.

**Exporting the keys in `pre_cmds` is not enough on its own for the write
sweep** — the daemon does not inherit the job script's environment. Jarvis
starts `clio_run` as `PsshExecInfo(env=self.env, ...)`, and `self.env` is a
dict jarvis builds itself from `EnvironmentManager.COMMON_ENV_VARS` — a fixed
toolchain list (`PATH`, `LD_LIBRARY_PATH`, `HOME`, `CC`, …) with **no `AWS_*`
entry**. The job script's exports therefore reach jarvis and every benchmark
process but never `clio_run`. The `clio_runtime` package's `forward_env`
option copies named variables from the submitting shell into the runtime's
own environment instead, and `clio_s3_write.yaml` lists the AWS names there:

```yaml
forward_env:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - AWS_SESSION_TOKEN      # optional; skipped when unset
  - AWS_DEFAULT_REGION
```

Values are never logged — only names, and only whether each was set. A
top-level `env:` dict in the pipeline would also work, but it would put the
secret in a file on disk; `forward_env` reads it from the live shell instead.
It also refuses to forward a value containing `$`, a backtick, a backslash, or
a newline — the ssh transport emits each variable as an inline `KEY="value"`
prefix and escapes only the double quote, so those characters would reach the
daemon altered, and a corrupted secret is indistinguishable from a permissions
problem at the far end. AWS keys are base64 (`A–Za–z0–9+/=`), so this should
never trigger; if it does, regenerate the credential rather than working
around the check.

Get any of this wrong and every `PutBlob` fails with `rc=11` — see
[Troubleshooting](#troubleshooting).

---

## Step 5 — Stage the read dataset (once, ~17 GiB, from a host with egress)

The write sweep needs no staging — it creates the data it writes, and the
bucket only needs to exist and be writable. The read sweep needs a dataset:

```bash
"$ZARR_VENV/bin/python3" \
  "$CLIO_REPO/jarvis_clio_core/scripts/stage_s3_read_bench_data.py" \
  --bucket "$S3_BENCH_BUCKET" --prefix clio-s3-read-bench \
  --region "$S3_BENCH_REGION"
```

Writes a 1024³ uint16 array (2 GiB) as 8 Zarr v3 stores (chunk edges
64/128/256/512 × none/zstd) plus 4 flat-object sets at matching sizes, and a
`manifest.json`. **Only the four uncompressed stores are read** now that zstd
is out of the grid (see
[Zarr-zstd is deliberately not a comparator](#zarr-zstd-is-deliberately-not-a-comparator));
the zstd stores are staged anyway, so re-enabling the variant needs no
re-staging. Idempotent — the manifest is written last, so a re-run skips
completed work and redoes only partial uploads.

Useful flags: `--dry-run`, `--only zarr|raw`, `--only-granularity 256`,
`--force`. Sanity-check it end-to-end against a local S3-compatible store
first if you like — both the staging script and the Zarr reader accept
`--endpoint-url` (or `S3_ENDPOINT`), and so does `cae_s3_tool`.

`--pattern` shapes the source entropy. It does not affect the current grid —
nothing compressed is read — but it decides what the staged zstd stores are
worth whenever the variant comes back. The default `smooth` compresses
~20–26× with zstd, close to the `zarr_benchmarks` reference dataset's 24×; it
is synthetic, so report the ratio (recorded per store in `manifest.json`)
rather than presenting it as a property of real scientific data. `random` is
incompressible and would reduce a compression axis to a measurement of
zstd's CPU cost.

---

## Step 6 — Register the clio-core package repo

```bash
jarvis repo add "$CLIO_REPO/jarvis_clio_core"
jarvis repo list          # jarvis_clio_core should appear
```

Both pipelines' `pre_cmds` also run this idempotently, but doing it once now
confirms the clio packages (`clio_runtime`, `clio_cte`, `clio_s3_bench`,
`zarr_s3_bench`, `s3_raw_put_bench`) import cleanly. The write pipeline's
`pre_cmds` additionally check for **stale shadow copies** —
`~/.ppi-jarvis/builtin/clio_*` overriding the checkout — and refuse to run if
one exists rather than silently loading it; remove any such directory instead
of working around the check.

---

## Running

`pre_cmds` expand when the **job** runs, so export overrides *before*
submitting:

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
{1, 8, 32} = 12 combinations × `repeat: 3`. Compression is not a sweep axis
and is no longer a comparison either: one uncompressed Zarr pass per row.
Output: `${HOME}/clio_s3_read_results/results.csv`.

### Write sweep (36 rows, ~3.6 h)

```bash
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_write.yaml"
```

Grid: 2 granularities (1 MiB, 4 MiB) × 6 concurrencies (1, 4, 8, 16, 32, 64) ×
3 repeats. Sized for overnight; `time: "08:00:00"` in the scheduler block.
Output: `${HOME}/clio_s3_write_results/results.csv`.

`verify` is **off** here (byte-fidelity was settled by the deleted smoke;
`objects_measured` is the per-row guard and costs no egress), and
`num_objects` is 256 so that K=64 has several windows of work behind it
instead of one. See the header comment in `clio_s3_write.yaml` for what the
K=64 `rawput` point is there to settle.

**This pipeline will refuse to start if `libclio_bdev_runtime.so` predates
the keep-alive change** (Step 2) — that check is now built into `pre_cmds`, so
a stale library fails in seconds rather than after 3.6 hours.

### Cost

**Reads dominate.** The read grid moves ~155 GiB of egress ≈ **$14** at
$0.09/GB; raising `repeat` scales that linearly.

Writes are cheap: ingress to S3 is free and PUTs run ~$0.005/1000, so the
36-row write sweep's ~37k PUTs come to about **$0.18**. Only leftover object
storage accrues, and `post_cmds` purges the write prefix. The one thing that
is *not* free on the write side is `verify`: it re-reads every blob, which is
GET egress (~19 GiB, ~$1.70) — hence off in the sweep.

Object keys are deterministic (`block_<offset>`, `raw_%06d.bin`, zarr chunk
paths), so re-runs overwrite rather than accumulate. Storage does not grow
without bound even if a purge is skipped.

---

## What gets compared

### Read

| | CLIO | Zarr |
|---|---|---|
| Driver | `clio_s3_read_bench` (C++) | `zarr_s3_read.py` (zarr-python + s3fs) |
| Jarvis package | `clio_s3_bench` (`mode: read`) | `zarr_s3_bench` (`mode: read`) |
| Path | `ParseOmni` → `S3FileAssimilator` → in-process persistent Poco+SigV4 GET → CTE `PutBlob` | `zarr.open` over `FsspecStore` → `arr[:]` |
| Reads | N flat objects, whole-object GETs | N chunks of a Zarr v3 store |
| Ends up | bytes in a distributed CTE tag | a NumPy array in process memory |
| Compression | none | none |

Both stacks move **the same 2 GiB of logical data** in every row. Across the
granularity axis only the *request count* changes (4096 / 512 / 64 / 8),
because each raw object set is the same 2 GiB buffer re-split.

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
transfer against a compressed Zarr one measures zstd rather than either
system. Both pipelines pin `variants: ["none"]`, and the
`zarr_s3.readzstd.*` / `zarr_s3.writezstd.*` columns are gone from the
results and from the `post_cmds` assertions. **Re-enabling is a one-line
change per pipeline** — restore `["none", "zstd"]` and put the zstd column
back in `post_cmds` — which is the intended move once CLIO can compress on
its own side of the comparison.

The 2026-08-26 write sweep, which did run zstd, is the evidence for why this
matters. Two numbers from it, kept here as provenance and **not** as a
comparison:

- zstd compressed 1.93× and was link-bound like everything else. Its
  `agg_bw_mbps` reached **20.6** — above the measured 11.1 MB/s link — purely
  because logical bytes exceeded wire bytes. It was the largest number in the
  file and the one most likely to be misquoted as a throughput win over CLIO.
- On the wire it was **10.34 MB/s at K=32, marginally slower than CLIO's
  10.11** for the same logical payload, having moved 133 MiB against CLIO's
  256 MiB.

Note the confound ran in **opposite directions** on the two sweeps: reading,
compression meant Zarr fetched fewer bytes; writing, it meant Zarr sent
fewer. Either way the comparison was about the codec.

The staged read dataset still contains its four zstd stores; nothing reads
them now, and leaving them staged is what keeps re-enabling cheap.

### Read the raw floor first (both sides)

The `rawput` (write) and `rawget` (read) rows are not competitors — they are
the **bound**. Each does the least possible work: no CTE, no chunking layer,
no metadata, no compression, just concurrent PUTs/GETs against the bucket.
Nothing in the comparison should beat them on sustained throughput.

Without a floor a poor CLIO number is uninterpretable. If CLIO is slow *and*
the raw floor is slow, the bottleneck is the link or the bucket, and no amount
of CLIO work will move it. Only a gap between them is a CLIO finding. Report
**ratio-to-floor**, not absolute MB/s.

> **The read floor now exists.** `s3_raw_get_bench` drives K concurrent
> `cae_s3_tool get` processes over the same `obj_%06d.bin` keys the CLIO read
> row reads, and is wired into `clio_s3_read.yaml` as the `raw_get` package.
> Like `rawput`, it is **not a floor at K=1** (serialized `fork+exec` + a temp
> file per object only pipeline at K≥8); it bounds sustained throughput.

### Read the fairness columns, not just the headline

Each driver emits a `Fairness` block alongside its throughput block, so every
`results.csv` row carries the caveats:

- `agg_bw_mbps` — throughput over **logical** (uncompressed) bytes on every
  stack. This is the directly comparable number.
- `wire_bw_mbps` / `bytes_moved` — what actually crossed the network. With
  every stack uncompressed this should now equal `agg_bw_mbps` on every row,
  so a gap between the two means something other than compression and is
  worth chasing. It remains the column to compare on: it is the one that
  stays honest if a compressed stack is ever added back.
- `objects_read` / `get_count`, `objects_written` / `put_count` — request-rate
  vs bandwidth regime. CLIO's write count is derived from block geometry
  (`object_size / block_size`), not from the blob count.
- `compression`, `decode_step` — should read `none` / `no` on every stack in
  every row; a row where Zarr reports a codec means `variants` was not
  pinned. On the write side `decode_step` is really the *encode* pass, kept
  under the read-side name so one parser key serves both sweeps.
- `subprocess_spawns`, `temp_file_bytes` — the sharpest contrast between the
  two directions. Reading, CLIO forks `cae_s3_tool` once per object and
  stages every object whole through node-local disk before it reaches CTE;
  both are 0 for Zarr. Writing, both are 0 for CLIO — it signs and PUTs
  directly from the runtime worker — and it is the raw floor that pays one
  spawn per object by construction. Structural costs of each implementation,
  not measurement noise.
- `runtime_worker_threads` — see [the concurrency caveat](#the-concurrency-caveat-read-before-interpreting-any-result).
- `max_rss_kb` — Zarr materializes the whole array in process; CLIO streams
  it. Absent if `/usr/bin/time` is not installed.

**Things that belong in any writeup:**

1. **The end states differ, in both directions.** Reading, CLIO lands bytes
   in a distributed, tiered CTE tag addressable by other CLIO clients while
   Zarr lands a NumPy array in one process's heap; writing, CLIO starts from
   a CLIO shared-memory buffer and goes through that same tiered CTE while
   Zarr writes from one process's heap. CLIO does strictly more work. Zarr's
   number is *not* "CLIO minus overhead."
2. **CLIO's internal read pipelining is not tunable.** `kMaxChunkSize` (1
   MiB) and `kMaxParallelTasks` (32) are `static constexpr` inside
   `S3FileAssimilator::Schedule`, so object size is the only granularity
   control on the CLIO read side.
3. **Source entropy is currently inert, and will matter again.** The write
   pipeline's `compressibility` (default 0.5) sets how compressible the Zarr
   source data is; with no codec running it changes nothing on the wire. It
   is kept set so that restoring zstd stays a one-line change, and because
   the value has to be stated alongside any future compressed numbers: at
   `0.0` zstd cannot compress at all and in fact slightly *expands* the data,
   at `1.0` it compresses to almost nothing, and neither resembles real
   scientific arrays. On the read side the equivalent knob is the staging
   script's `--pattern`.

### The concurrency caveat (read before interpreting any result)

**Both directions block a runtime worker thread, for different reasons, with
identical consequences.**

- *Read:* `S3FileAssimilator` downloads via `fork()` + **blocking
  `waitpid()`** on a runtime worker thread — not a `CLIO_CO_AWAIT`. The
  worker is held for the entire S3 GET.
- *Write:* `WriteBlocks` is a coroutine body in the bdev, and the signed PUT
  runs to completion inside it, over the worker's persistent `S3Connection`.

Either way the effective concurrency ceiling is `clio_runtime.num_threads`,
not the requested `K`. That is why `runtime.num_threads` is swept in
lockstep with the concurrency axis in both pipelines, and why
`cpus_per_task` is large. **Always compare `requested_concurrency` against
`effective_concurrency` and measured scaling before concluding anything.**

If raising K changes nothing on the read side, the worker pool is the
ceiling: raise `runtime.num_threads`, or switch to the multi-process
fallback (`clio_s3.nprocs > 1`, which partitions the key space via
`--object-stride` / `--object-offset`).

On the write side, **compare against `rawput` first, not against K=1** — see
the measured results below for why that rule matters, and note this is
exactly the ceiling persistent connections were meant to move: each S3 PUT
occupying a whole worker thread means the concurrency ceiling was always the
worker pool, not a connection limit — so keep-alive helps only insofar as the
*per-object* cost inside that occupied slot shrinks.

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

**corrupts the bucket name.** A bare bucket with no prefix will fail against
a bucket that does not exist, or worse, silently target one that does.
Always configure a prefix.

---

## What the write sweep measured (pre-keep-alive baseline, 2026-08-26)

**This run predates the persistent-connection change** — it is the "before"
number the smoke and the keep-alive build gate exist to keep from being
silently reproduced under an "after" label. It is also exactly the data
`scripts/plot_s3_write_no_pc.py` (local, untracked) plots. Until a full
post-keep-alive sweep has run, this is the only 36-row reference available;
treat every figure below as **pre-keep-alive** unless a newer sweep has
superseded it.

36 rows: 2 sizes × 6 concurrencies × 3 repeats, all `success`. Wire MB/s,
mean of the three repeats. This run also predates the removal of zstd from
the grid — its zarr-zstd column is omitted here and summarized under
[Zarr-zstd is deliberately not a comparator](#zarr-zstd-is-deliberately-not-a-comparator)
above; nothing else about the run changes, since every figure below comes
from the uncompressed stacks:

| K | \| | CLIO 1M | rawput 1M | zarr 1M | \| | CLIO 4M | rawput 4M | zarr 4M |
|---|---|---|---|---|---|---|---|---|
| 1  | | 2.51 | 1.72 | 4.14 | | 6.13 | 4.76 | 4.84 |
| 4  | | 2.60 | 4.04 | 9.49 | | 6.33 | 7.38 | 10.73 |
| 8  | | 3.41 | 5.46 | 10.75 | | 7.64 | 9.18 | 11.08 |
| 16 | | 3.92 | 7.75 | 10.87 | | 9.27 | 10.46 | 11.10 |
| 32 | | 4.76 | 9.91 | 10.90 | | 9.91 | 10.83 | 11.06 |
| 64 | | 4.87 | 9.95 | 10.83 | | 10.57 | 11.06 | 10.93 |

**The K=64 rawput point settles the ceiling question.** rawput moves +0.3%
(1 MiB) and +2.2% (4 MiB) from K=32 to K=64 — flat. rawput forks K processes
and uses no runtime worker, so a per-connection concurrency limit would
still be climbing there. It is the **link**, ~11.1 MB/s, and nothing on
either side of a connection change can beat it. Report **ratio to floor**,
which is a property of CLIO; the absolute MB/s is a property of the night
you ran it.

**At 4 MiB CLIO converges on the floor: 0.96× at K=64**, having climbed 0.86
→ 0.83 → 0.89 → 0.92 → 0.96.

**At 1 MiB it does not.** CLIO plateaus at 4.87 MB/s — **0.49× the floor** —
while rawput and zarr both reach ~10.9. *CLIO well below rawput ⇒ CLIO's own
ceiling*, and this is that case. It is a **per-object** ceiling, not a
bandwidth one: CLIO saturates at ~5.5 objects/s, worth 5.5 MB/s at 1 MiB but
22 MB/s at 4 MiB — above the link, which is exactly why the 4 MiB rows look
healthy and hide it.

The K=1 latency fit says the fixed cost is not the problem. Fitting
`latency = fixed + size/rate` through the two K=1 points:

| stack | fixed | marginal rate |
|---|---|---|
| CLIO | 313 ms | 11.96 MB/s |
| rawput | 497 ms | 11.64 MB/s |

Both see the same ~12 MB/s link, and CLIO's *fixed* per-object cost is the
**lower** of the two — 313 ms against the floor's 497 ms of fork+exec plus
temp file. That is why CLIO beats the floor at K=1 (1.47× at 1 MiB, 1.29× at
4 MiB). CLIO's problem is that ~180 ms of that per-object work did not
pipeline across concurrency on this build, where the floor's does. Compare
the scaling K=1→64: rawput 5.8×, CLIO 2.2×. **That ~180 ms of
non-pipelining per-object work is precisely what a persistent connection
removes** — roughly two round trips plus a TLS negotiation, on every op
after the first on a given worker.

**This is the number the next sweep needs to move, and both outcomes are
results:**

- *Ratio improves* → the per-object handshake was the ~180 ms that failed to
  pipeline, and keep-alive fixed it.
- *Ratio does not move* → still useful. If the runtime log's keepalive
  tallies prove reuse actually happened (Step 2) while the ratio stayed put,
  handshake latency is **eliminated** as the ceiling, re-pointing at worker
  concurrency and task routing instead — each S3 PUT occupies a whole
  runtime worker thread, so the concurrency ceiling may simply be the worker
  pool regardless of connection reuse. That is a clean elimination of the
  leading suspect, not a failure.

Do not declare victory on a green build-gate check alone — it proves the
mechanism is present and (via the job log) that reuse happened, not that
reuse moved the number. That needs a fresh 36-row sweep compared against the
table above.

The oversubscription check on the baseline run came back **clean**: at K=64
(`runtime.num_threads: 64` on `cpus_per_task: 40`) CLIO did not dip below
K=32 at either size — 4.76 → 4.87 and 9.91 → 10.57.

**Client memory is a clear CLIO win, by ~6×.** At 4 MiB / K=64: CLIO 266 MB
against zarr's 1664 MB. CLIO's K-slot SHM window grows as K × object_size and
nothing else; zarr materializes the whole 1 GiB array in-process. rawput is
flat at 21 MB only because its bytes live in a temp file —
`temp_file_bytes` reaches 256 MiB at K=64, so it moved the cost to disk
rather than avoiding it.

Run-to-run spread over the 3 repeats: zarr is the steadiest (median 0.3% CV),
CLIO and rawput median ~3% with occasional 16% outliers — shared-uplink
weather, which is what `repeat: 3` is for.

---

## Plotting results

Three local, **untracked** plotting scripts live in
[`jarvis_clio_core/scripts/`](../../../scripts/) — none is checked into the
repo, so regenerate them from `results.csv` rather than expecting them to
exist on a fresh checkout:

- `plot_s3_write_bench.py <results.csv> [out_dir]` — the post-keep-alive
  write sweep, once one exists.
- `plot_s3_write_no_pc.py <results.csv> [out_dir]` — the pre-keep-alive
  ("no persistent connection") baseline above; this is the 2026-08-26 data.
- `plot_s3_read.py <results.csv> [out_dir]` — the read sweep. There is no
  ratio-to-floor figure here (no raw-GET floor exists to divide by), and a
  missing bar (rather than a hatched one) means that cell's repeats all
  failed outright — check `results.csv`'s `error` column before assuming a
  zero.

All three plot CLIO, Zarr (uncompressed), and — write only — the raw-PUT
floor. **None plots Zarr-zstd**, matching the grids themselves
([Zarr-zstd is deliberately not a comparator](#zarr-zstd-is-deliberately-not-a-comparator)).
Each emits wire-bandwidth, aggregate-bandwidth, throughput (objects/s), and
peak-RSS figures (plus a ratio-to-floor figure on the write side), and prints
a summary table to stdout so the headline numbers are quotable without
opening a PNG.

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
   `list` of the bucket prefix rather than a number the benchmark computed,
   so it is the only column a run that wrote nothing cannot fabricate. Zero
   means the row is fiction regardless of what the throughput columns say.

   **More than `num_objects` does not necessarily mean the allocator
   fragmented.** Check `put_count` first: if `put_count == objects_written ==
   num_objects` and only `objects_measured` is high, the allocator was fine
   and the listing picked up **stale objects from an earlier row** — fixed by
   `purge_prefix` (default on), which empties the bdev prefix at the start of
   every row. `objects_purged` is a per-row column that stays nonzero if a
   leak in the bdev teardown path persists — the purge fixes the
   *measurement*, not the leak, and this column is what keeps that visible.
   If the purge itself fails it says so and the row still runs;
   `objects_measured` then reverts to a lower bound for that row — compare
   it against `put_count` instead of `num_objects`.

   To check the prefix by hand — Ares has **no AWS CLI**, so use the zarr
   venv's botocore, the same way the pipeline itself does:

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
   Two qualifications:

   * **Not at K=1.** The floor forks one `cae_s3_tool` per object and stages
     each through a temp file, so at concurrency 1 it pays `num_objects`
     serialized `fork+exec` calls in the critical path and can come back
     *slower* than CLIO. It is a floor for **sustained throughput**, not for
     single-op latency.
   * **Compare `wire_bw_mbps`, not `agg_bw_mbps`.** The two agree now that
     every stack is uncompressed, so this costs nothing to honour — and it
     is the habit that kept the comparison honest when `zarr_s3.writezstd`
     was in the grid, where it beat every other stack on `agg_bw_mbps`
     purely by moving roughly half the bytes for the same logical payload.

   If CLIO beats the floor on **wire** bandwidth at high K, *then* something
   is not reaching S3 — check that the CTE tier really is the S3 device and
   not a local fallback, and check `objects_measured`.
6. **Confirm reuse actually happened**, not just that the build has the
   mechanism: `grep "S3 keepalive" ~/clio_s3_write-<jobid>.{out,err}` and
   look for `sockets == 1` with `requests >> 1` per worker, and a `TOTAL`
   line with `sockets` well under `requests`. A worker that served exactly
   **one** request is excluded from the reuse expectation — it cannot reuse
   by arithmetic, and the elastic worker pool routinely hands the
   last-spawned workers a single op.
7. **Run once with `clio_s3.verify: true`** to prove bytes round-tripped: it
   re-reads every blob through CTE and compares content byte-for-byte. Leave
   it off for timed rows.

---

## Troubleshooting

### Build / keep-alive

| Symptom | Cause / fix |
|---|---|
| `grep -acF "S3 keepalive worker=" "$LIB"` returns nothing | The installed library predates the persistent-connection change. `clio_s3_write.yaml` will refuse to run against it. Rebuild — Step 1. |
| `ldd "$LIB"` shows `aws-cpp-sdk` | Do not run the write sweep. The AWS SDK must stay absent **from this library's link line**; linking it stack-smashes runtime init. `aws-sdk-cpp` legitimately appears in `spack find -d` for `+cae`'s sake — the dependency tree is not the gate, the `ldd` line is. |
| Job log: `$LIB PREDATES the persistent-connection change` | Same as above, caught automatically by `clio_s3_write.yaml`'s `pre_cmds` before any of the 36 rows run. |
| `reuse_ratio <= 1.00` in the job log | Every operation opened its own socket — the mechanism is not working even though the binary has it. |
| A worker with ≥ 2 requests reused nothing | One worker's slot is reconnecting even though others are not — suspect slot indexing. Workers that served exactly one request are excluded from this check by design. |
| No `S3 keepalive TOTAL` line at all | The transport never reached `Destroy` — the run did not tear down cleanly. |
| `spack view symlink` did not pick up a rebuild | It never overwrites existing links. `spack view rm` first, then symlink, then verify with `readlink -f`. |
| `spack install` "already installed" after a `git pull` on a develop spec | Change detection missed it — force with `spack install --overwrite -y iowarp@dev`. For a non-develop branch spec: `spack uninstall -y iowarp@dev && spack clean -s && spack install iowarp@dev +cae +cte +s3_cae +s3_bdev`. |
| `spack uninstall` "matches multiple packages" | Old builds accumulated. `spack env activate clio-s3 && spack gc -y` reaps unreferenced ones; a survivor is referenced by another environment — find it with `spack env list`. |
| `GLIBCXX_3.4.31` / `CXXABI_1.3.15` undefined references building Poco | `~/.bashrc` contamination (`envs/iowarp/bin` on `PATH`). Remove it, `spack clean -s`, retry. |

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
| A repeat fails with `run exceeded run_timeout of 1800s` | A genuinely wedged row (e.g. the zarr_s3 package hung starting up) rather than a slow one — the sweep continues past it; check whether it recurs before raising `run_timeout` |

### Write

**"Failed to initialize Clio" from the benchmark.** Ares compute nodes run
`ptrace_scope=1`, which blocks the SHM attach path for a detached client. The
pipeline sets `ipc_mode: "ipc"` (unix socket) for this reason — do not change
it.

**HTTP 301 in the runtime log.** Region mismatch. `S3_BENCH_REGION` must be
the bucket's actual region; SigV4 signatures are scoped to it.

**HTTP 403 in the runtime log.** The credential export did not reach the
daemon. Confirm `pre_cmds` printed `credentials exported from [...]`, and
that the profile exists in `~/.aws/credentials`.

**Throughput identical to a RAM tier / suspiciously fast.** The DPE placed
blobs somewhere other than S3. The CTE package must configure **exactly
one** device and it must be the `s3://` one — any local tier present gives
the DPE an alternative.

**`mkdir: cannot create directory 's3:'`.** An older `clio_cte` package that
does not skip cloud paths in its `Mkdir` loop. Pull the branch.

**The bdev link assert fails in `pre_cmds`.** Either the view is stale (see
the `spack view symlink` trap above) or the build lacked `+s3_bdev`.

**Every `PutBlob` fails with `rc=11`.** CTE has no target to place on. `rc`
in the range 11–19 is `10 + alloc_result` from `PlaceBlobBytes`; 11 means
allocation found no viable device. Scroll **up** in the runtime log — the
cause is printed minutes earlier and looks like this:

```
core_config.cc:534 ERROR ParseStorageConfig Config error: Invalid bdev_type 's3'
                     (must be 'file', 'ram', 'hbm', 'pinned', or 'noop')
core_runtime.cc:743 WARNING Create Warning: No storage devices configured
```

That error message is the *old* one — the current build names `'s3', or
'gcs'` in the same list. So the runtime library predates the s3/gcs
allowlist, the S3 tier was dropped at config-parse time, and CTE came up
with zero devices. Note that this is only a `WARNING`: the pool is created
successfully and the failure does not surface until the first write. The
`pre_cmds` gate greps the compiled-in literal out of
`libclio_cte_core_runtime.so` to catch this before the allocation is spent.

**"mixed IOWarp installs on PATH".** Two different `iowarp` prefixes were
reachable at once — typically a stale `IOWARP_VIEW` supplying `clio_run`
while a freshly built spack prefix supplies `clio_s3_write_bench`. Because
`spack view symlink` never overwrites existing links, refreshing a view that
already has an `iowarp` in it silently keeps serving the old one. The
reliable fix is to skip the view entirely and point `IOWARP_VIEW` straight
at the prefix:

```bash
export IOWARP_VIEW=$(spack find --format '{prefix}' iowarp@dev | tail -1)
```

RPATH makes the symlink farm unnecessary, and a prefix has the `bin/` and
`lib/` layout the pipeline expects.

**`S3 bdev: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set`, even
though the job script exported them.** This is the same `rc=11` cascade as
above, one layer down: the bdev fails to initialize,
`core_runtime.cc` logs `Failed to register target ... (error code: 1)` as a
**warning**, CTE comes up with zero devices, and every `PutBlob` returns 11.
The job log names the mechanism: `Auto-built environment with N variables
(no 'env' field in pipeline)`. Fix is `forward_env` — see
[Step 4](#step-4--aws-credentials).

**`clio_s3_write_bench is STALE -- it predates <marker>`.** `spack develop`
builds compile from the working tree, so pulling the branch does **not**
rebuild them, and nothing about the spec, the hash or the view changes to
show it. Run `spack install iowarp@dev` (dev spec: rebuilds in place). For a
non-`develop` spec, branch versions never rehash when the branch moves, so
`spack install` reports "already installed" and skips the compile entirely —
use the uninstall-then-clean-then-install sequence from the build
troubleshooting table above. The gate greps a build stamp (`kBuildMarker` in
`clio_s3_write_bench.cc`) out of the installed binary rather than trusting
the spec.

**`S3_BENCH_REGION=... but bucket ... lives in ...`.** SigV4 is
region-scoped and the bdev's signer does not follow redirects, so a wrong
region is an HTTP 400/301 on every PUT, from inside a runtime worker. This
is easy to miss because **botocore hides it**: it transparently retries
against the correct region, so an `aws`-style check or a `HeadBucket`
preflight goes green while the daemon fails. The preflight therefore asks S3
for the authoritative answer with `GetBucketLocation` and refuses to run on
a mismatch, naming the export to fix.

**A row is green but `clio_s3.write.objects_measured` is 0.** Nothing
reached the bucket and the throughput columns are fiction. It runs before
teardown on purpose: `FreeBlocks` issues a `DeleteObject` per block, so a
count taken later reads zero even on a healthy run.

**`objects_measured` disagrees with `put_count`.** Not a failure. One
`PutBlob` normally becomes exactly one S3 object: `AllocateFromTarget` hands
the whole request to the allocator, `WriteBlocks` issues one `PutObject` per
returned block, and an unfragmented request gets a single block. More
objects than blobs means the allocator fragmented and split the request.

**`TaskStatModel: failed to open /tmp/clio/models/...`.** Harmless. The
runtime persists a perf model there and logs an error per attempt if it
cannot. `/tmp` is node-local, so creating the directory on the login node
does nothing — `pre_cmds` creates it on the compute node. It does not gate
bdev init or `PutBlob`.

**Stale jarvis package copies.** `~/.ppi-jarvis/builtin/clio_*` shadows the
checkout; `pre_cmds` refuses to run if one exists. Remove it rather than
working around it.
