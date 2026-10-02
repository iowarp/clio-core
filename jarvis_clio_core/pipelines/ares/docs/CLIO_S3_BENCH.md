# CLIO vs Zarr — S3 benchmarks on Ares

Two benchmark sweeps that measure how fast CLIO moves data to and from Amazon
S3, compared with Zarr and with a raw S3 client (the "floor"):

| | Read sweep | Write sweep |
|---|---|---|
| Pipeline | [`clio_s3_read.yaml`](../clio_s3_read.yaml) | [`clio_s3_write.yaml`](../clio_s3_write.yaml) |
| Stacks compared | CLIO, Zarr, raw GET | CLIO, Zarr, raw PUT |
| Grid | object size {512 KiB, 4 MiB, 32 MiB, 256 MiB} × concurrency {1, 8, 32} × 3 repeats | object size {1 MiB, 4 MiB} × concurrency {1, 4, 8, 16, 32, 64} × 3 repeats |
| Rows | 36 | 36 |
| Runtime | ~10 h (18 h allocation) | ~3.6 h (8 h allocation) |
| S3 cost | ~$20 (egress) | ~$0.20 |
| Results | `~/clio_s3_read_results/results.csv` | `~/clio_s3_write_results/results.csv` |

Both run on bare metal (no containers) from a single Slurm allocation.

---

## Quick reference

If you accept every default, this is the whole setup. Each step is explained
in [Step by step](#step-by-step) below.

```bash
# 0. On an Ares login node, with spack on PATH and clio-core + jarvis-cd checked out
export CLIO_REPO="$HOME/clio-core"

# 1. Build IOWarp and expose it as a view
spack install iowarp@dev +cae +cte +s3_cae +s3_bdev
spack view --dependencies no symlink "/mnt/common/$USER/iowarp-s3-view" iowarp@dev
export IOWARP_VIEW="/mnt/common/$USER/iowarp-s3-view"

# 2. Check the build
ls "$IOWARP_VIEW"/bin/{clio_s3_read_bench,clio_s3_write_bench,cae_s3_tool}

# 3. Python environments
python3 -m venv "$HOME/jarvis-venv"
"$HOME/jarvis-venv/bin/pip" install -e "$HOME/jarvis-cd"
export PATH="$HOME/jarvis-venv/bin:$PATH"

python3 -m venv "$HOME/zarr-venv"
"$HOME/zarr-venv/bin/pip" install 'zarr>=3' s3fs numpy
export ZARR_VENV="$HOME/zarr-venv"

# 4. AWS credentials (keys live only in ~/.aws/credentials)
chmod 600 ~/.aws/credentials
export S3_BENCH_BUCKET=my-bucket S3_BENCH_PROFILE=clio-bench S3_BENCH_REGION=us-east-2

# 5. Upload the read dataset (once, ~17 GiB; the write sweep needs nothing)
"$ZARR_VENV/bin/python3" "$CLIO_REPO/jarvis_clio_core/scripts/stage_s3_read_bench_data.py" \
  --bucket "$S3_BENCH_BUCKET" --region "$S3_BENCH_REGION"

# 6. Register the jarvis packages and submit
jarvis repo add "$CLIO_REPO/jarvis_clio_core"
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_read.yaml"
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_write.yaml"

# 7. When a job finishes, plot it
python3 "$CLIO_REPO/jarvis_clio_core/scripts/plot_s3_read.py"  ~/clio_s3_read_results/results.csv
python3 "$CLIO_REPO/jarvis_clio_core/scripts/plot_s3_write.py" ~/clio_s3_write_results/results.csv
```

---

## Step by step

### Before you start

- You are on an **Ares login node**.
- **Spack** is installed and on `PATH`.
- **clio-core** is checked out at `$CLIO_REPO` (default `~/clio-core`) and
  **jarvis-cd** at `~/jarvis-cd`.
- You have an **S3 bucket** and an IAM key pair that can read and write it.
- Everything below (spack install tree, view, venvs) must live on a shared
  filesystem the compute nodes can see: `$HOME` or `/mnt/common`.

### Step 1 — Build IOWarp

```bash
spack install iowarp@dev +cae +cte +s3_cae +s3_bdev
spack view --dependencies no symlink "/mnt/common/$USER/iowarp-s3-view" iowarp@dev
export IOWARP_VIEW="/mnt/common/$USER/iowarp-s3-view"
```

The first build takes 30–60 minutes, mostly for the AWS SDK.

**Rebuilding after a `git pull`:** spack does not notice new commits on a
branch, and `spack view symlink` never replaces existing links. To pick up
new code:

```bash
spack uninstall -y iowarp@dev && spack clean -s
spack install iowarp@dev +cae +cte +s3_cae +s3_bdev
spack view rm "$IOWARP_VIEW" iowarp
spack view --dependencies no symlink "$IOWARP_VIEW" iowarp@dev
```

### Step 2 — Check the build

```bash
ls "$IOWARP_VIEW"/bin/{clio_s3_read_bench,clio_s3_write_bench,cae_s3_tool}
```

All three must exist. If one is missing, the build silently dropped S3
support (usually because the AWS SDK failed to build). Rebuild before you
submit.

### Step 3 — Python environments

Jarvis runs the pipelines, and Zarr is one of the stacks being measured:

```bash
python3 -m venv "$HOME/jarvis-venv"
"$HOME/jarvis-venv/bin/pip" install -e "$HOME/jarvis-cd"
export PATH="$HOME/jarvis-venv/bin:$PATH"

python3 -m venv "$HOME/zarr-venv"
"$HOME/zarr-venv/bin/pip" install 'zarr>=3' s3fs numpy
export ZARR_VENV="$HOME/zarr-venv"
```

Ares has no AWS CLI. The Zarr venv also provides the S3 tooling the
pipelines use for checks and cleanup.

### Step 4 — AWS credentials

Put a long-lived key pair in `~/.aws/credentials` under a named profile:

```ini
[clio-bench]
aws_access_key_id = ...
aws_secret_access_key = ...
```

```bash
chmod 600 ~/.aws/credentials
export S3_BENCH_BUCKET=my-bucket      # your bucket
export S3_BENCH_PROFILE=clio-bench    # the profile name above
export S3_BENCH_REGION=us-east-2      # the bucket's real region
```

- The pipelines only ever see the profile name, never the keys.
- `S3_BENCH_REGION` **must** be the bucket's actual region. The write sweep
  checks this and refuses to start on a mismatch.
- The key needs `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`,
  `s3:ListBucket` and `s3:GetBucketLocation` on the bucket.
- Avoid short-lived SSO/STS tokens. The sweeps outlast them.

### Step 5 — Upload the read dataset (once)

The read sweep reads a pre-uploaded dataset. The write sweep creates its own
data and needs nothing here.

```bash
"$ZARR_VENV/bin/python3" "$CLIO_REPO/jarvis_clio_core/scripts/stage_s3_read_bench_data.py" \
  --bucket "$S3_BENCH_BUCKET" --region "$S3_BENCH_REGION"
```

This uploads ~17 GiB under the `clio-s3-read-bench/` prefix. It is safe to
re-run: completed uploads are skipped. Add `--dry-run` to preview what it
would upload.

### Step 6 — Submit

Export your settings **before** submitting, because the job reads them when
it starts:

```bash
export S3_BENCH_BUCKET=my-bucket S3_BENCH_PROFILE=clio-bench S3_BENCH_REGION=us-east-2
export IOWARP_VIEW=/mnt/common/$USER/iowarp-s3-view ZARR_VENV=$HOME/zarr-venv
export CLIO_REPO=$HOME/clio-core JARVIS_VENV=$HOME/jarvis-venv

jarvis repo add "$CLIO_REPO/jarvis_clio_core"     # once
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_read.yaml"
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/clio_s3_write.yaml"
```

Submit one sweep at a time. Two sweeps share the network link and would skew
each other's numbers.

Logs go to `~/clio_s3_read-<jobid>.out` and `~/clio_s3_write-<jobid>.out`.
Each submit deletes that sweep's previous `results.csv`, so copy it somewhere
first if you want to keep it.

### Step 7 — Check the run

At the end of the `.out` log, look for:

```
successful rows: 36 / 36
GREEN ROWS WITH BLANK THROUGHPUT (== FAILURES): none
GREEN ROWS THAT WROTE NO S3 OBJECTS (== FAILURES): none     # write sweep only
```

A count below 36 means some rows failed. Look at the `status` and `error`
columns in `results.csv` to see which ones. The plots average whatever rows
succeeded.

### Step 8 — Plot

The plots need `pandas` and `matplotlib`:

```bash
python3 "$CLIO_REPO/jarvis_clio_core/scripts/plot_s3_read.py"  ~/clio_s3_read_results/results.csv  [out_dir]
python3 "$CLIO_REPO/jarvis_clio_core/scripts/plot_s3_write.py" ~/clio_s3_write_results/results.csv [out_dir]
```

Each script prints a summary table and writes five PNGs, next to the CSV by
default:

| File | Shows |
|---|---|
| `s3_<read\|write>_ratio_to_floor.png` | Each stack's speed relative to the raw floor. **Start here.** |
| `s3_<read\|write>_wire_bw.png` | MB/s actually sent over the network |
| `s3_<read\|write>_agg_bw.png` | MB/s of data delivered (the same as wire here, since nothing is compressed) |
| `s3_<read\|write>_ops.png` | Objects per second |
| `s3_<read\|write>_max_rss.png` | Peak client memory |

---

## Reading the results

- **Compare ratios, not raw MB/s.** The S3 link from Ares tops out around
  11 MB/s and varies from night to night. The ratio to the raw floor, measured
  in the same row, is the number that carries over between runs.
- **The floor is not a floor at concurrency 1.** The raw client starts one
  process per object, so at K=1 it is slow and other stacks can beat it. The
  plots hatch those bars.
- **Hatched bars at high concurrency** mean concurrency exceeded the number of
  objects, so the cell repeats a lower one.
- **Nothing is compressed.** Zarr runs uncompressed so both sides move the
  same bytes.
- **The stacks end in different places.** CLIO stores data in its tiered
  storage engine, where other CLIO clients can reach it. Zarr only fills one
  process's memory. So CLIO does more work per byte.

---

## Troubleshooting

The job log names most setup problems in its first few lines, with an `ERROR:`
prefix.

| Symptom | Fix |
|---|---|
| `ERROR: clio_s3_read_bench` / `clio_s3_write_bench` / `cae_s3_tool` `not on PATH` | Build is missing S3 support or `IOWARP_VIEW` is wrong. Redo Steps 1–2. |
| `ERROR: jarvis not on PATH` | Set `JARVIS_VENV` or create `~/jarvis-venv` (Step 3). |
| `ERROR: zarr venv broken` | Reinstall `zarr s3fs numpy` into `$ZARR_VENV`. |
| `ERROR: S3_BENCH_BUCKET not set` / `S3_BENCH_REGION not set` | Export them in the shell you submit from (Step 6). |
| `ERROR: S3_BENCH_REGION=... but bucket ... lives in ...` | Export the region the message suggests. |
| `ERROR: ~/.aws/credentials missing/unreadable` or `AWS_ACCESS_KEY_ID empty` | Check the file exists and has a `[$S3_BENCH_PROFILE]` section (Step 4). |
| `Preflight GET failed` (read) | Wrong credentials, region or bucket, or the dataset was never uploaded (Step 5). |
| HTTP 403 in the log | The key lacks one of the permissions in Step 4. |
| A rebuild seems ignored | The view still points at an old build. Recreate it (Step 1), or skip the view: `export IOWARP_VIEW=$(spack find --format '{prefix}' iowarp@dev \| tail -1)`. |
| `GLIBCXX_3.4.31` / `CXXABI_1.3.15` errors while building | Remove old `envs/iowarp/bin` `PATH` exports from `~/.bashrc`, run `spack clean -s`, then rebuild. |
| Jarvis loads old package code | Delete any `~/.ppi-jarvis/builtin/clio_*` copies. |
| `Failed to initialize Clio` | Do not change `ipc_mode: "ipc"` in the pipeline. Ares compute nodes need it. |
| `TaskStatModel: failed to open /tmp/clio/models/...` | Harmless; ignore. |
| A row failed with `run exceeded run_timeout` | Usually a one-off hang. The sweep continues. Re-run only if it repeats. |
| Job ended before 36 rows | The allocation ran out. S3 speed varies; raise `time:` in the pipeline's `scheduler:` block. |
