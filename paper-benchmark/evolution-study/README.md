# The evolution study

A compression selector's whole job is to notice that data changed, so a
simulation that reaches steady state early says nothing about whether it can.
Each workload's default configuration was chosen by measuring how fast its data
actually evolves; this directory is the measured record behind those choices
and behind the paper's Figures 3-5.

**About 72 GB of simulation produced the 1.7 MB kept here.** The dumps are not
in git and never were — each `run*.sh` deletes them after measuring. What
survives is enough to regenerate every figure.

## Files

```
<workload>/<run>.blocks.csv.gz   one row per (step_from, step_to, field, block)
                                 -- the only per-field, per-region breakdown
<workload>/<run>.json            evolution.py's summary: mean/median/p10/
                                 last_quarter, pct_active, pct_cells_same, and
                                 the per-interval series the figures plot
<workload>/<run>.slices.npz      five mid-plane slices, ~100-300 KB, standing
                                 in for 4.8-26 GB of dumps: Fig. 3's top row
                                 needs nothing else, and renders byte-identical
<workload>/<run>_k31_*.blobs.csv.gz  one row per stored chunk of the K=31
                                 exploration run -- codec, ratio, bytes
<workload>/<run>_k31_*.meta.json     that run's parameters, self-describing
logs/<workload>_k31.stdout.log.gz    the run's console output -- chunk count,
                                     bytes in/out, achieved ratio, wall time
logs/<workload>_k31.runtime.log.gz   the runtime's own log for the same run
```

The logs are copied here because for two of the four runs there is nowhere
else durable: Nyx's `results/` store was cleaned, and VPIC's lives only in a
session scratch directory that gets deleted. LAMMPS' and WarpX's stores do
survive on this machine under `../<workload>/results/<tag>/`, but they are
gitignored and hold 3.3 GB and 28 GB of tier images respectively, so nothing
in git depended on them either.

**WarpX has no `runtime.log.gz`.** Its runtime log is 49 MB of NeuroPress path
trace -- 4.3 MB gzipped, 98% of everything here put together. It is a debug
trace rather than provenance, and the thing it was used for, reconstructing the
per-chunk blob record, is already kept as `warpx/warpx_2000_k31_1m.blobs.csv.gz`.

The `_4m` / `_8m` / `_1m` suffix is the chunk size. WarpX is 1 MiB and that is
a correctness condition, not a preference: openPMD emits each AMReX box as a
separate partial write, so at 4 MiB no chunk ever completes and zero field
bytes reach the tier while the run reports success.

LAMMPS has no `.slices.npz`: it has no grid, so it has no Fig. 3 either.

## What produced it

Two different measurements live here, and they were **not** run on the same
dumps. Confusing them is easy and the numbers do not transfer.

**The evolution metric** (`.json`, `.blocks.csv.gz`, `.slices.npz`), from
`../evolution.py`:

| file | what it read | frames |
|---|---|---|
| `nyx/nyx_128_1000.*` | a 128³ sedov dump set, 1,000 steps | 101 |
| `vpic/vpic_126_2000.*` | the 126³ weibel dumps the K=31 run replayed | 200 |
| `lammps/lammps_2000.*` | the box-40 raw dumps the K=31 run staged | 201 |
| `warpx/warpx_2000.*` | the K=31 run's own openPMD output | 201 |

Nyx is the one to watch: its evolution metric is 128³ at 1,000 steps, while its
compression run below is 256³ at 2,000. That is deliberate, not a mismatch — a
CFL timestep halves when the grid doubles and the Sedov shock advances a
roughly fixed number of *cells* per step, so the two end at the same physical
state. Running 2,000 steps at 128³ instead drives the front into the domain
boundary.

**The K=31 compression runs** (`*_k31_*.blobs.csv.gz`, `*.meta.json`), all at
error bound 1e-3 with exhaustive exploration — the primary plus all 31
alternatives — at each workload's "Default Evolving Benchmark Configuration",
2,000 timesteps:

| workload | grid / size | route | chunk | wall |
|---|---|---|---|---|
| Nyx | 256³ sedov | replay | 4 MiB (also 8 MiB) | 1179 s |
| VPIC | 126³ weibel | replay | 4 MiB | 2356 s |
| LAMMPS | box 40, 256k atoms | in situ | 4 MiB | 402 s |
| WarpX | 64×64×512 laser | in situ | 1 MiB | 2285 s |

`../README.md` carries the exact invocations. Each `.meta.json` here repeats
the parameters of the run beside it, so a file is readable without them.

## The figures

Beside the data they were computed from, not in the workload directories:

```
<workload>/fig3.png    activity is spatially localized
<workload>/fig4.png    compression varies within one dump
<workload>/fields_fig.png                one field at begin / middle / end
<workload>/evolution_begin_middle_end.png   the same, shared colour scale
fig5.png               the evolution measurement across all FOUR workloads
```

`fig5.png` is at the top level because it is one figure spanning every
workload, computed from the four `.json` summaries below it. It used to sit in
each workload's directory as four byte-identical copies.

LAMMPS has no `fig3.png`: atom coordinates have no grid to localize activity
on, which is the same reason it has no `.slices.npz`.

`./paper_figures.py {fig3,fig4,fig5,fields}`, beside this file, draws them; pass
`--slices <run>.slices.npz` to use the cache instead of the deleted dumps.

## Regenerating

```bash
PYTHON=<interpreter with matplotlib and numpy> ./regenerate.sh [OUTDIR]
```

With no OUTDIR it overwrites the figures in place; pass a scratch path to
render elsewhere and diff first. Verified: a second run reproduces every one
byte-identically.

**`<workload>/evolution_begin_middle_end.png` is NOT regenerable** and
`regenerate.sh` does not attempt it. That figure reads the dumps
(`../plot/figure_evolution.py --dir`), and the dumps were deleted after
measuring. The five mid-plane slices in `<workload>.slices.npz` cover fig3's
top row and nothing else.

One provenance gap worth knowing: **the `.npz` files do not record which field
they hold.** They store five arrays named `s0`..`s4`, so the variable the fig3
top row shows cannot be recovered from them. `regenerate.sh` pins it per
workload instead, and the values there were recovered by matching each
committed figure's bottom row against every field in `blocks.csv.gz`.

## History

An earlier 26-configuration sweep at 1,000 steps chose the defaults these runs
use. It was cleared in `6ce4226f` when this larger campaign replaced it, and is
still in git:

```bash
git show 256e7c2c --stat                                   # the commit that added it
git checkout 6ce4226f~1 -- paper-benchmark/evolution-study/ # all 78 files back
```

The conclusions did not go with it: each workload's README keeps its
"Default Evolving Benchmark Configuration" section, with the parameters, the
upstream reference for each, the values tested and the outcome numbers.

## Two things to clear before re-running

- **`h5dump` must be installed** (`hdf5-tools`). `../evolution.py --source
  openpmd` and `../analysis/validate/warpx_gen_fields.sh` both shell out to it
  and neither fails gracefully without it.
- **Nyx in situ stores zero blobs and exits 0.** The binary lost its Clio hook
  in a rebuild; `nyx/patches/` carries the raw-field-dump and single-precision
  patches but not that one. The replay route is unaffected, and is what the
  Nyx runs here used.
