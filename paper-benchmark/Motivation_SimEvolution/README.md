# Motivation: the compressibility ceiling moves as the simulation evolves

```bash
PYTHON=<python with numpy+matplotlib> ./run.sh            # run + draw, ~100 s on one A100
PYTHON=<python with numpy+matplotlib> ./run.sh --redraw   # draw from the CSVs here, no GPU
./plot_motivation.py --blobs blobs.csv.gz --explore explore.csv.gz --stats
```

```
run.sh              farm, run, figure
plot_motivation.py  the plotter: csv readers, slabs, the plate, --stats
blobs.csv.gz        one row per stored chunk
explore.csv.gz      one row per measured action per chunk (22,016)
meta.json           the run's parameters
evolution_begin_middle_end_density.png
```

The dumps are not here: the figure reads `$NYX_FIELDS`, `/mnt/np-data/nyx-i48/fields`
on this machine. `../plot/figure_evolution.py` draws the panels; with none of
this figure's flags it still reproduces the evolution-study figure byte for
byte (md5 `3ad09a36edad691223665964e9b766f9`).

## Measured

```
spread within one dump, across slabs    median 223x   max 477x
spread within one slab, through the run median 452x   max 559x
ceiling range                           10.6x .. 5,924x over 688 chunks
best fixed action vs per chunk          1.00x (zstd|q1|s4, per-chunk best on 93%)
```

`--stats` recomputes all of it from the CSVs.

## Not defaults in `run.sh`

1. `CLIO_NEUROPRESS_RATIO_CAP=1e9` — upstream clamps at 100×, on the only term
   the ratio cost model keeps, so every chunk above it would tie.
2. One field per run — the replay leaks ~1 GB of device memory per dump.
3. `MEASURE_QUALITY=0 MEASURE_DT=0` — heaviest allocators, and the ratio model
   reads neither.
4. The dump farm is named by timestep, so every panel label is a true step
   (AMReX writes the last plotfile at `max_step`, not the next interval).

## The paragraphs for the paper

> **Setup.** We ran the Nyx Sedov blast at 256³ for 2,000 timesteps (cfl 0.8,
> the workload's default evolving configuration) and wrote the density field
> every 48 steps, giving 43 snapshots of 64 MiB each, 2,752 MiB in total,
> which the runtime stored as 688 chunks of 4 MiB. A chunk is a contiguous
> byte range of a Fortran-ordered field, so at this resolution each one is
> exactly sixteen z-planes — a horizontal slab of the volume, and sixteen
> slabs per snapshot. We replayed all 688 chunks through Clio with NeuroPress
> in exhaustive-exploration mode (K=31), which measures every one of its 32
> candidate actions — eight GPU codecs × quantizer on/off × no shuffle or a
> 4-byte shuffle — on every chunk under an error bound of 1e-3: 22,016
> compressions, 59 s on a single A100. For each chunk we take the best ratio
> any action achieved as that chunk's *compressibility ceiling*. The ceiling
> is a property of the data and of the codec set, not of a selection policy:
> it is what a perfect selector would get.
>
> **What we measured.** That ceiling is not a property of the run, and not
> even of a snapshot. Across the 688 chunks it spans 10.6× to 5,924×, a
> factor of 559. Within a single snapshot, the sixteen slabs differ by a
> median of 223× between the most and the least compressible (max 477×);
> within a single slab followed through the run, it moves by a median of 452×
> (max 559×). Figure X shows the mechanism directly: slabs the blast front
> has not reached are still ambient and compress by three orders of
> magnitude, while slabs it has crossed carry structure and collapse to tens
> of ×, and as the shell expands, one slab after another crosses that line.
> At step 96, twelve of the sixteen slabs are still at the 5,924× ceiling,
> the two at the blast centre are down to ~235× and their neighbours to
> ~1,410×; at step 1,008 only four slabs are still at the ceiling; by step
> 2,000 none is, no slab exceeds 149×, and the shocked interior is at 10.6×.
> A policy that fixes one compression budget for a run is therefore wrong by
> two to three orders of magnitude on most of its data, and fixing one per
> snapshot is wrong by a comparable factor *within* that snapshot.
>
> **What this does not claim.** At this error bound the best action is the
> same one almost everywhere (quantized zstd with a 4-byte shuffle is the
> per-chunk optimum on 93% of chunks), and the best single fixed action
> stores within 1.00× of the total bytes that choosing per chunk would: on
> this workload the case for adapting is the *ceiling*, not the identity of
> the codec. Knowing a chunk will yield 5,924× and its neighbour 10.6× is
> what lets a system place, schedule and budget for it; assuming one number
> for the run is wrong by 500×.

> **Figure X.** Nyx Sedov density at steps 96, 1,008 and 2,000 of a
> 2,000-step 256³ run, on one shared logarithmic colour scale. Horizontal
> lines are the boundaries of the 4 MiB chunks the field was stored as; each
> chunk is sixteen z-planes of the Fortran-ordered volume, and is labelled
> with its maximum achievable compression ratio, i.e. the best of all 32
> measured actions at an error bound of 1e-3. The ratio varies by up to 477×
> between slabs of one snapshot and by up to 559× within one slab over the
> run.
