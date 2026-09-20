# paper-benchmark

Four production simulation workloads — **Nyx**, **VPIC**, **WarpX**, **LAMMPS** —
run through Clio's compressor with NeuroPress selecting a codec per chunk. This
directory produces every number and figure in the paper's evaluation, plus the
evidence behind the benchmark's own configuration choices.

Start here for orientation. [`BENCHMARK.md`](BENCHMARK.md) is the reference: it
explains the cost models, the error-bound semantics, device residency, and the
findings that change how results should be read. It is organised by topic, not
by script, so use the index below to find the tool and that document to
understand it.

```bash
./run_benchmark.sh --smoke          # 16 cells, exercises every path, minutes
./run_benchmark.sh --dry-run        # print the matrix without running it
./run_benchmark.sh --profile mid    # ~5-10 min per run
./run_paper.sh                      # the paper matrix: 6 configs x 4 workloads
```

Requires an NVIDIA GPU. `numpy pandas scipy scikit-learn matplotlib` for the
analysis and plotting scripts; `h5dump` (poppler's `hdf5-tools`) for anything
reading WarpX's openPMD output.

---

## Which script do I run?

### Run a workload

| script | what it does |
|---|---|
| `run_benchmark.sh` | the whole matrix — four workloads, both cost models, lossless and lossy |
| `run_paper.sh` | the paper's six-cell matrix per workload |
| `<workload>/run_config.sh` | **one** named configuration of one workload |
| `nyx/run_config_insitu.sh`, `vpic/run_config_insitu.sh` | the same, but GPU-resident: the simulation hands the compressor a device pointer instead of replaying dumps |
| `<workload>/gen_fields.sh` | write field dumps to disk, for the replay route and for the evolution metric |
| `<workload>/run_sweep.sh` | one workload across several configurations |

WarpX has no replay route: a stock, unmodified WarpX writes through Clio's HDF5
VOL, so `warpx/run_config.sh` always runs the simulation.

### Compare configurations

| script | what it does |
|---|---|
| `compare_wallclock.sh` | NeuroPress against fixed codecs and cuSZ/cuSZp3/ndzip, end to end |
| `compare_perchunk_oracle.sh` | how close the selector gets to a per-chunk oracle |
| `wallclock_table.py` | the summary table for `compare_wallclock.sh` |
| `perchunk_oracle_tables.py` | the tables for `compare_perchunk_oracle.sh` |

### Paper figures that are re-measured, not regenerated from a file

Named for the NeuroPress submission's figure numbers (not those of
[Where the paper's figures come from](#where-the-papers-figures-come-from)).
Plots go to `figures/<figure>/`; runs go under `results/` unless `--out` is given.

| script | what it does |
|---|---|
| `figure_5.sh` → `plot/fig5_timesteps.py`, `plot/plot_fig5.py` | per-chunk time breakdown of the write and read paths, from `CLIO_NEUROPRESS_PHASE_LOG` |
| `figure_8.sh` → `plot/fig8_trace.py`, `plot/plot_fig8.py` | regret and cost MAPE per chunk with every configuration measured; see `figures/fig8/README.md` |
| `figure_9.sh` → `plot/plot_fig9.py` | end-to-end wall clock: Baseline, the ablation and the external codecs, each ending with a timed flush; what the bars do and do not measure is in [`plot/fig9.md`](plot/fig9.md) |
| `ai/gen_fields.sh` | the AI workload's dumps (ViT-B/16 checkpoints, upstream's exporter) |
| `lib/compose_hooks.sh` | compose knobs every `<workload>/common.sh` sources: `BENCH_TIER*`, `BENCH_FLUSH_MS`, `BENCH_NP_LR`, `BENCH_NP_MAPE` |
| `lib/decode_wal.py` | which storage pools a run placed each blob on |

LAMMPS on the GPU needs `--require-device`; replayed WarpX needs
`CLIO_NEUROPRESS_STAGE_H2D=1`.

### Aggregate and check a run

| script | what it does |
|---|---|
| `collect.py` | a whole sweep → `summary.csv` |
| `metrics.py` | one cell → `metrics.json` |
| `audit_run.py` | cross-check one run: does the tier hold what the log claims? |

### Ask why a chunk compressed the way it did

| script | what it does |
|---|---|
| `analysis/analyze_exploration.py` | **the complete analysis pipeline** over an exploration log — mechanism, modelling, cross-workload comparison, figures. One command; the flags only turn work *off*. See [`analysis/README.md`](analysis/README.md). |
| `analysis/plot_ratio_explained.py` | why one chunk gets the ratio it gets |
| `analysis/plot_three_properties.py` | what each of the three data properties does to compressibility |
| `analysis/redundancy_probe.py` | temporal redundancy the run leaves unexploited |

### Measure whether the data evolves

| script | what it does |
|---|---|
| `evolution.py` | **the metric.** Per-block change `E` between consecutive dumps, and the share of cells bit-identical to the previous dump. Reads all four workloads' dump formats (`--source f32 / openpmd / raw`). |
| `evolution_pass.sh` | runs the metric *and* the begin/middle/end figure for a workload, then deletes the dumps |
| `evolution_rank.py` | ranks a workload's configurations from their `evolution.json` — re-scores without re-running anything |
| `analysis/plot_block_evolution.py` | every block's trajectory, the early-vs-late settle test, and cell-level reusability |

### Look at the data

| script | what it does |
|---|---|
| `plot/figure_evolution.py` | one field at the first, middle and last frame, **one shared colour scale** across the three so a panel that looks empty *is* empty |
| `plot/figure_lossy.py` | the same three frames, original against decompressed, plus the error map |
| `plot/viz_fields.py` | a full montage and GIF of an f32 dump sequence, with blast-wave diagnostics |
| `plot/viz_selection.py {actions,bound,chunks}` | a run chunk by chunk: what the model saw, what it picked, and whether the error bound did anything |
| `plot/viz_learning.py {trend,perchunk}` | whether online SGD moves the model: `trend` smooths the cost-model error and draws a learning-off control against it, `perchunk` marks every chunk that produced a gradient. Reads `selection.csv`, gzipped or not. |
| `plot/paper_figures.py {fig3,fig4,fig5,fields}` | the paper's Figures 3–5 and the field montage beside them |
| `plot/viz_openpmd.py`, `plot/viz_atoms.py` | the same for WarpX's openPMD fields and LAMMPS's atom state |

`plot/figure_evolution.py` refuses to write a blank plate: a slice that is
identically zero while the volume is not means the plane or the shape is wrong,
not that the data is static.

---

## Running the large workloads on one local GPU

These are the invocations that produced what is in `evolution-study/` and
`learning-study/`. They were previously driven by Slurm jobs on Delta; the
parameters below are those jobs' own, with the cluster paths localised.

**The chunk size is per workload, and for WarpX it is a correctness condition,
not a tuning knob.** openPMD emits each AMReX box as a separate partial write,
so at 4 MiB no chunk ever completes and zero field bytes reach the tier — while
the run succeeds and the native `.h5` is perfect. Measured: 0 field blobs at
4 MiB, 400 at 1 MiB.

```bash
# --- Nyx, K=31 exploration ----------------------------------------------
# 256^3 at 2,000 steps, NOT 128^3: see the CFL note below.
./nyx/gen_fields.sh --ncell 256 --steps 2000 --plot-int 40 --out "$FIELDS"
CLIO_NEUROPRESS_STAGE_H2D=1 ./nyx/run_config.sh explore-balance \
    --fields "$FIELDS" --bw 5e6 --eb 1e-3 --explore-k 31 \
    --chunk 4194304 --results "$RESULTS" --tag nyx_k31

# --- VPIC, K=31 exploration ---------------------------------------------
CLIO_NEUROPRESS_STAGE_H2D=1 ./vpic/run_config.sh explore-balance \
    --fields "$VPIC_FIELDS" --bw 5e6 --eb 1e-3 --explore-k 31 \
    --chunk 4194304 --results "$RESULTS" --tag vpic_k31

# --- LAMMPS, K=31 exploration (in situ) ---------------------------------
# NO --f32 here: the campaign is float64, 1206 chunks. --f32 halves both.
./lammps/run_config.sh explore-balance \
    --box 40 --steps 2000 --gap 10 --require-device \
    --chunk 4194304 --bw 5e6 --eb 1e-3 --explore-k 31 \
    --results "$RESULTS" --tag lammps_k31

# --- WarpX, K=31 exploration (in situ, stock unpatched WarpX) -----------
# --bin is NOT optional: the default build links system HDF5 1.10, which has
# no VOL plugin API. WarpX then runs to completion, writes a perfect 25 GB of
# native openPMD, and stages ZERO chunks.
./warpx/run_config.sh explore-balance \
    --bin ~/src/warpx/build-h5114/bin/warpx.3d.NOMPI.CUDA.SP.PSP.OPMD.EB.QED \
    --steps 2000 --interval 10 --chunk 1048576 --stage-h2d \
    --bw 5e6 --eb 1e-3 --explore-k 31 --results "$RESULTS" --tag warpx_k31
```

The learning campaign is the same four lines with `explore-balance` replaced by
`learn` / `learn-ratio` / `dynamic` / `dynamic-ratio` and the `--explore-k`
dropped; `learning-study/drivers/` holds them as runnable scripts.

`--require-device` on LAMMPS is a correctness flag, not a performance one:
without it the driver gathers each chunk into host memory, where NeuroPress's
CUDA-only shuffle and quantizer refuse, and the blobs are silently lost.

Three things that bite on a single local GPU:

- **Nyx must scale its grid with the step count.** A CFL timestep halves when
  the grid doubles and the Sedov shock advances a roughly fixed number of
  *cells* per step, so 2,000 steps at 256³ ends at the same physical state as
  1,000 at 128³. Running 2,000 at 128³ instead drives the front into the domain
  boundary and spends the second half of the run measuring a blast with nowhere
  to go.
- **Nyx replay needs `CLIO_NEUROPRESS_STAGE_H2D=1`.** Replay reads `.f32` into
  host shm, and `CLIO_NEUROPRESS_REQUIRE_DEVICE` refuses it. Staging copies the
  chunk up and runs the CUDA kernels on it; the in-situ route needs no such flag
  because AMReX hands over a device pointer directly.
- **WarpX's metric needs `h5dump`** (`hdf5-tools`). `evolution.py --source
  openpmd` and `analysis/validate/warpx_gen_fields.sh` both shell out to it, and
  neither fails gracefully when it is absent.

---

## Where the paper's figures come from

The evaluation's heterogeneity section rests on three figures, each computed
from one measurement file rather than regenerated at paper time. Those files
are in [`evolution-study/`](evolution-study/README.md); the simulation dumps
behind them are not, and never were — about 72 GB, deleted by each `run*.sh`
after measuring.

| figure | source file | what it gives |
|---|---|---|
| **Fig. 3** — activity is spatially localized | `evolution-study/nyx/nyx_128_1000.blocks.csv.gz` | group `pct_cells_same` by block index, density field, last frame pair → outermost 85.6%, central 26.7% |
| **Fig. 4** — compression varies within one dump | `evolution-study/nyx/nyx_256_2000_k31_4m.blobs.csv.gz` | 1039× spread within one dump (1724.6× down to 1.66×), 244× median over 306 dumps; 303 of 306 assigned more than one codec |
| **Fig. 5** — the same measurement on four workloads | `evolution-study/{nyx/nyx_128_1000, vpic/vpic_126_2000, lammps/lammps_2000, warpx/warpx_2000}.json` | the `interval_means` and `interval_pct_cells_same` series in each |

A "dump" in Fig. 4 is one **field at one timestep**, which is how
`paper_figures.py` keys it -- Nyx's six fields over 51 steps give 306. Grouping
by timestep alone instead lumps all six fields together and inflates both the
maximum and the median, so the two counts are not interchangeable.

Fig. 5's four summaries, as the figure plots them:

| workload | pairs | `pct_active` | cells bit-identical, first → last |
|---|---|---|---|
| Nyx | 100 | 76.5% | 100.0% → 50.1% |
| VPIC | 199 | 93.8% | 8.2% → 8.3% |
| LAMMPS | 200 | 100.0% | 0.0% → 0.0% |
| WarpX | 200 | 93.6% | 94.1% → 10.5% |

The published WarpX figure reported 94.2% → 11.0%, so this campaign reproduces
it. LAMMPS at a flat 0.0% is not a bug: atom coordinates are continuous floats
that move every step, so no cell is ever bit-identical, and LAMMPS has no Fig. 3
for the same reason — there is no grid to localize activity on.

The rendered figures sit beside the data, in `evolution-study/`:

```
evolution-study/<workload>/fig3.png  fig4.png    the first two above
evolution-study/fig5.png                         the third -- one figure over
                                                 all four workloads, so one copy
evolution-study/<workload>/fields_fig.png
evolution-study/<workload>/evolution_begin_middle_end.png
```

Not `<workload>/viz/` -- that is each workload's `visualize.sh` render target,
regenerated on every run.

`evolution-study/regenerate.sh` rebuilds all of them from that directory
alone; `plot/paper_figures.py {fig3,fig4,fig5,fields}` is what it drives. Pass
`--slices <run>.slices.npz` to read the cached mid-planes instead of the
deleted dumps — 100–300 KB standing in for 4.8–26 GB, and byte-identical
output. Pass `--shape NX,NY,NZ` for a non-cubic grid: WarpX's 64×64×512 has
exactly 128³ cells, so the cube-root test *passes* and silently reshapes a slab
into a cube.

---

## Does the *model* adapt? — `learning-study/`

[`learning-study/`](learning-study/README.md) asks the other half of the
question. The evolution study establishes that the data changes; this one asks
whether NeuroPress's online SGD notices — sixteen runs, four workloads ×
{balanced, ratio-only cost} × {learning on, off}, exploration off throughout so
the model trains only on the action it actually picked.

The learning-off arms are the point: a prediction error that falls over a run
proves nothing on its own, because the data gets easier or harder by itself.
In 7 of 8 arms the learning run crosses the SGD gate far less often than its
own control.

---

## Configuration: what makes each workload evolve

A compression selector's job is to notice when data changes, so a workload that
reaches steady state early says nothing about it. Each workload's README carries
a **"Default Evolving Benchmark Configuration"** section giving the parameters,
the upstream reference for each, the values tested, and the outcome:

| workload | section | selected |
|---|---|---|
| Nyx | [`nyx/README.md`](nyx/README.md) | `nyx.cfl = 0.8`, `exp_energy` at the deck's 1.0 |
| VPIC | [`vpic/README.md`](vpic/README.md) | `VPIC_CLEAN_DIV_INT = 10`, upstream thermal velocities |
| WarpX | [`warpx/README.md`](warpx/README.md) | `laser1.e_max = 32e12` (a0 = 8), moving window **on** |
| LAMMPS | [`lammps/README.md`](lammps/README.md) | `--temp 6.0 --skin 0.8 --every 5` |

These are defaults in the runner scripts, not flags you must remember — running
any `run_config.sh` with no physics arguments gives the studied configuration.
`BENCHMARK.md` §6 summarises all four, and each README's "Parameters tested"
table carries its own outcome numbers, so the reasoning is self-contained.

The 26-configuration sweep those choices were made from was cleared in
`6ce4226f` when the 2,000-step campaign now in `evolution-study/` replaced it.
It is still in git — `git show 256e7c2c` — and `git checkout 6ce4226f~1 --
paper-benchmark/evolution-study/` restores it, including the WarpX
configuration disqualified on physics rather than on score.
