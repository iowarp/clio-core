# Issue #526 perf-eval pipelines — Ares setup guide

This directory holds the two Jarvis regression pipelines for Issue #526:

- [single_node.yaml](../single_node.yaml) — one node, a 4×6 (I/O size × thread
  count) sweep over four storage stacks = **24 rows**.
- [distributed.yaml](../distributed.yaml) — four nodes, a 1→2→4 client-node scaling
  sweep over three stacks = **3 rows**.

Both run their real work inside a pre-built Apptainer container (the SIF). This
guide takes you from a **fresh Ares account that has only Spack and a clio-core
checkout** to a green `jarvis ppl submit`. No conda, and no assumptions beyond
those two things.

---

## Quick reference — full sequence

The whole setup, if you accept every default. Each step is explained in detail
below; start there if anything fails or you need to deviate.

```bash
# 0. prerequisites: on Ares login node, spack on PATH, $CLIO_REPO and $JARVIS_CD set
export CLIO_REPO="$HOME/clio-core"
export JARVIS_CD="$HOME/jarvis-cd"

# 1. spack recipes up to date
spack repo update builtin

# 2. apptainer (compute node preferred; ~suid mandatory)
spack install apptainer@1.5.0~suid          # or: GOFLAGS=-buildvcs=false spack install --dirty apptainer@1.5.0~suid

# 3. spack view on a shared FS — apptainer only, no deps (see step 3 for why)
spack view -d false symlink -i "/mnt/common/$USER/perf-eval-view" apptainer

# 4. jarvis venv
python3 -m venv "$HOME/jarvis-venv"
"$HOME/jarvis-venv/bin/pip" install -e "$JARVIS_CD"

# 5. put both on PATH for the interactive steps below
export PATH="/mnt/common/$USER/perf-eval-view/bin:$HOME/jarvis-venv/bin:$PATH"

# 6. init jarvis + build the SIF + register the clio repo
jarvis init
bash "$JARVIS_CD/docker/build_perf_eval_image.sh"
jarvis repo add "$CLIO_REPO/jarvis_clio_core"

# 7. submit
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/single_node.yaml"
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/distributed.yaml"
```

---

## The mental model (read this first)

A submit succeeds when the login node can put **two host binaries** on `PATH`
before the job starts, and the SIF is staged where Jarvis looks for it:

1. **`apptainer`** — launches the per-node containers. Provided by a **Spack
   view** on a shared filesystem (so every compute node sees the same binary).
2. **`jarvis`** — the launcher itself; the generated sbatch script calls bare
   `jarvis ppl run`. Provided by a **dedicated Python venv**
   (`pip install -e <jarvis-cd>`).
3. **The SIF** — `iowarp-perf-eval.sif`, staged under
   `<jarvis shared_dir>/containers/` so the YAMLs resolve it by basename.

The pipelines never `spack load` or `conda activate` — those install shell
functions that misbehave under the `set -euo pipefail` job script. Everything is
plain `PATH` prepends. The pre_cmds in each YAML re-apply the same `export PATH`
inside the job, driven by three env vars you can override at submit time:

| Env var | Default | What it points at |
|---|---|---|
| `PERF_EVAL_VIEW` | `/mnt/common/$USER/perf-eval-view` | Spack view dir holding `apptainer` (step 3) |
| `JARVIS_VENV` | `$HOME/jarvis-venv` | The jarvis venv (step 4) |
| `CLIO_REPO` | `$HOME/clio-core` | This clio-core checkout (used for `jarvis repo add`) |

If you accept every default below, you never set any of these. Override them
**before** `jarvis ppl submit` (they are read when the job runs, not when it is
queued).

---

## Prerequisites

- You are on an **Ares login node**.
- **Spack is installed and on `PATH`** (`spack --version` works — i.e. you have
  sourced `share/spack/setup-env.sh`). Spack on Ares is a *per-user* install, so
  its package recipes can be out of date — step 1 fixes that.
- **clio-core is checked out** (this repo). Its path is your `CLIO_REPO`
  (default `$HOME/clio-core`).
- **jarvis-cd is checked out at `dev`** — it carries the container deploy,
  `container_fakeroot`, `run_timeout`, the Slurm `scheduler:` block, and the SIF
  build script these pipelines depend on. Call its path `$JARVIS_CD` below.
- **Docker is available** on the machine where you build the SIF (step 6). The
  SIF build is a `docker build` followed by an `apptainer build
  docker-daemon://…`.

> **Shared-filesystem requirement.** The Spack install tree, the view, the venv,
> and the SIF must all live on a filesystem the compute nodes can see. On Ares
> both `$HOME` (NFS) and `/mnt/common` qualify. The defaults put the view under
> `/mnt/common/$USER` and everything else under `$HOME`. If your Spack install
> root is on node-local disk, move it (or the view symlinks will dangle on the
> compute nodes).

---

## Step 1 — Bring Spack's package recipes up to date

The `--buildvcs` failure that older notes warn about is a **version bug in
apptainer ≤ 1.4.4**, not an Ares bug. It is gone in **apptainer 1.5.0** (added
upstream 2026-06-19). A stale per-user Spack packages checkout is the usual
reason `1.5.0` is missing:

```bash
spack repo update builtin
spack versions --safe apptainer | head    # expect 1.5.0 at the top
```

If `spack repo update` reports *"cannot rebase: you have unstaged changes,"* your
builtin checkout has local edits. Find that checkout's path with `spack repo
list`, `git stash` inside it, and re-run the update. `1.5.0` needs
**`go@1.25.7:`**; if that will not resolve on your Spack, use a fallback version —
see step 2.

---

## Step 2 — Build `apptainer` with Spack

**`~suid` is mandatory.** The builtin recipe defaults to `+suid`, which builds a
setuid `starter-suid` that must be root-owned — unbuildable unprivileged and
fatal to the rootless `--fakeroot` path these pipelines use. Always pass `~suid`.

On a **compute node** (recommended — a clean `/tmp` stage tree never trips the Go
VCS check):

```bash
srun -p debug -N1 -n1 --pty bash        # grab an interactive compute node
spack install apptainer@1.5.0~suid
exit
```

If you must build on the **login node**, force the VCS stamp off (this works at
1.5.0 because the Go build flag is no longer clobbered by apptainer's makefile):

```bash
GOFLAGS=-buildvcs=false spack install --dirty apptainer@1.5.0~suid
```

**Version fallbacks** (if `go@1.25.7` will not resolve): `spack install
apptainer@1.4.4~suid` (needs `go@1.23.6`) or `@1.3.6~suid` (needs `go@1.20`).
Use the same version in the view command in step 3.

---

## Step 3 — Build the Spack view (the `apptainer` on `PATH`)

Materialize `apptainer` into a view under `/mnt/common` so every node sees it.
**Link only apptainer itself — no dependencies (`-d false`):**

```bash
rm -rf "/mnt/common/$USER/perf-eval-view"      # if re-running; symlinks only, uninstalls nothing
spack view -d false symlink -i "/mnt/common/$USER/perf-eval-view" apptainer
```

**Why no dependencies.** The view exists for one purpose: put the `apptainer`
binary on `PATH`. Apptainer does not find its helpers through `PATH` — the spack
recipe symlinks the FUSE helpers (`squashfuse`, `fuse-overlayfs`, `gocryptfs`,
`e2fsprogs`) into its own `libexec/apptainer/bin`, and bakes an absolute
`mksquashfs` path into `apptainer.conf`. So a dependency-free view is fully
functional, and it removes two whole classes of failure:

- **Dependency version conflicts can't happen.** Linking the closure pulls in
  build-time artifacts like `compiler-wrapper`; if any two installed specs
  disagree on its version, the view refuses to build (see
  [Troubleshooting](#troubleshooting)). Nothing to conflict when nothing is
  linked.
- **`shadow` can't shadow the system tools.** `shadow` is a run-dependency (via
  `+libsubid`) shipping its own `newuidmap`/`newgidmap`. Because the view is
  PATH-**prepended**, its user-owned, non-setuid copies would beat the system's
  setuid-root `/usr/bin/newuidmap`, and `--fakeroot` would die with *"newuidmap
  must be owned by the root user."* Spack cannot make a setuid binary
  unprivileged, so the system copies are the ones you want — and with `-d false`
  they are never at risk.

Verify:

```bash
export PATH="/mnt/common/$USER/perf-eval-view/bin:$PATH"
command -v apptainer                     # -> the view path
command -v newuidmap                      # -> /usr/bin/newuidmap (NOT the view)
```

> **Alternative — a full runtime closure in the view.** If you want apptainer's
> dependencies on `PATH` too, link them but exclude the two problem packages:
> `spack view -e shadow -e compiler-wrapper symlink -i "/mnt/common/$USER/perf-eval-view" apptainer`
> (`-e` takes a regex and may be repeated). Both forms are verified working on
> Ares; `-d false` is documented as the default because it is immune to
> dependency drift on any machine.

---

## Step 4 — Create the jarvis venv (the `jarvis` on `PATH`)

A dedicated venv keeps jarvis off conda and makes the launcher environment
reproducible:

```bash
python3 -m venv "$HOME/jarvis-venv"
"$HOME/jarvis-venv/bin/pip" install -e "$JARVIS_CD"
export PATH="$HOME/jarvis-venv/bin:$PATH"
command -v jarvis                         # -> $HOME/jarvis-venv/bin/jarvis
```

`pip install -e` pulls jarvis's Python deps; any gap surfaces here (cheap),
before you ever queue a job.

---

## Step 5 — Initialize Jarvis

This writes `~/.ppi-jarvis/*.yaml`, which defines `shared_dir` — where the SIF
build stages the image and where the pipelines look for it:

```bash
jarvis init                               # accepts defaults; shared_dir = ~/.ppi-jarvis/shared
```

`~/.ppi-jarvis/shared` is on NFS home, so it is node-visible and fine. To put it
on `/mnt/common` instead:
`jarvis init ~/.ppi-jarvis/config ~/.ppi-jarvis/private /mnt/common/$USER/jarvis-shared`.

---

## Step 6 — Build the SIF

With `apptainer` (view) and `jarvis` (venv) both on `PATH` from the steps above,
build the image. This does a `docker build` then converts it to a SIF and stages
it at `<shared_dir>/containers/iowarp-perf-eval.sif`:

```bash
bash "$JARVIS_CD/docker/build_perf_eval_image.sh"
```

- It reads `shared_dir` from `~/.ppi-jarvis/*.yaml`, so step 5 must be done
  first.
- It bakes in clio-core at `CLIO_REF` (default `dev`) — the resolved commit SHA
  is echoed; record it, that is exactly what is inside the SIF.
- Re-run it daily so the sweep tracks the latest IOWarp. Useful overrides:
  `CLIO_REF=<branch|sha>`, `IOWARP_SPEC="iowarp@dev +fuse"`, `SKIP_DOCKER_BUILD=1`
  (reuse an existing local docker image).

---

## Step 7 — Register the clio-core package repo

The pipelines' pre_cmds run `jarvis repo add "$CLIO_REPO/jarvis_clio_core"`
idempotently, but doing it once now confirms the clio packages
(`clio_runtime`, `clio_cte`, `clio_cte_libfuse`) import cleanly:

```bash
jarvis repo add "$CLIO_REPO/jarvis_clio_core"
jarvis repo list                          # jarvis_clio_core should appear
```

---

## Step 8 — Submit

Hostfiles are **generated automatically** by the scheduler from
`$SLURM_JOB_NODELIST` at job start — you do **not** create
`~/.jarvis-perf-eval/*_hostfile.txt` by hand.

```bash
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/single_node.yaml"
# then, when that is green:
jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/distributed.yaml"
```

`single_node.yaml` requests 1 node on `debug`; `distributed.yaml` requests 4. If
you accepted every default in steps 3–5, submit with no extra env. Otherwise
export your overrides first, e.g.:

```bash
PERF_EVAL_VIEW=/some/view JARVIS_VENV=$HOME/jarvis-venv CLIO_REPO=$HOME/src/clio-core \
  jarvis ppl submit "$CLIO_REPO/jarvis_clio_core/pipelines/ares/single_node.yaml"
```

---

## Verifying a run

1. In the `.out` log, the toolchain echo must point at **your view and venv**:
   ```
   toolchain: apptainer=/mnt/common/<you>/perf-eval-view/bin/apptainer jarvis=/home/<you>/jarvis-venv/bin/jarvis
   ```
   A miniconda path here means the migrated toolchain did **not** take effect.
2. The post_cmd prints `ALL 24 GREEN` (single_node) / `ALL 3 GREEN`
   (distributed).
3. **Check the numbers, not just the color.** Open the results CSV
   (`$HOME/single_node_results/results.csv` or `$HOME/distributed_results/results.csv`)
   and confirm the throughput columns are populated — a green row with a blank
   `*_max_mibs` column is a failure. Sane single-node reference: `nfs_ior` 4k read
   scales roughly 380 → 5000+ MiB/s across 1 → 32 ranks.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `apptainer@1.5.0` is an unknown version | Stale Spack packages repo — **step 1** (`spack repo update builtin`). |
| `error obtaining VCS status: exit status 128` during build | Building on the login node. Build on a compute node, or use `GOFLAGS=-buildvcs=false spack install --dirty …` (**step 2**). |
| `go@1.25.7` won't concretize | Use a fallback: `apptainer@1.4.4~suid` or `@1.3.6~suid`, and the same version in the step-3 view command. |
| `spack view`: *Package conflict detected: (Linked) X@1.0 … (Specified) X@1.1*, or *Package merge blocked by file: …/.spack/compiler-wrapper/spec.json* | Two specs in apptainer's dependency closure disagree on a package version (typically `compiler-wrapper`, after some deps were built at different times). `-i`/`--ignore-conflicts` does **not** fix it — that only suppresses file collisions between unrelated packages. Use the step-3 command as written (`-d false`), which links no dependencies and cannot hit this. If you need the closure linked, exclude the offenders: `-e shadow -e compiler-wrapper`. Delete the view first (`rm -rf "/mnt/common/$USER/perf-eval-view"`) — it is only symlinks, so this uninstalls nothing. |
| `--fakeroot`: *newuidmap must be owned by the root user* | `shadow` leaked into the view and its non-setuid `newuidmap` is shadowing `/usr/bin/newuidmap`. Cannot happen with the step-3 command (`-d false` links no deps); if you used the full-closure alternative, rebuild it with `-e shadow`. |
| `--fakeroot`: *Target N is owned by a different user … gid mismatch* | Slurm gave the job the project GID. The pipelines already re-exec under `exec sg "$USER"` in pre_cmd #1; for a **hand-run** smoke, wrap it yourself: `sg "$USER" -c 'apptainer instance start --fakeroot <SIF> t'`. |
| `ERROR: SIF missing: …/iowarp-perf-eval.sif` | The SIF was never staged — run **step 6** (`build_perf_eval_image.sh`). |
| `ERROR: apptainer/jarvis not on PATH` in the job log | Your `PERF_EVAL_VIEW` / `JARVIS_VENV` override does not match where you built them, or the view/venv is on node-local disk. Fix the override or move them onto a shared FS. |
