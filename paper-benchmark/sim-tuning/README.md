# Simulation tuning: making per-chunk codec choice pay

Goal: simulation workloads where NeuroPress v2 (learning on, exploration off)
beats storing every chunk with the best single codec, which is picked by
exhaustive search under the balanced 4-tier cost model. The model is never
retrained on these data.

## Nyx: Sedov blasts in a layered multiphase medium

Patches (apply on top of `../nyx/patches/nyx-raw-field-dump.patch`; rebuild
`nyx_HydroTests`):

- `../nyx/patches/nyx-sedov-multiphase.patch`: new `prob.*` options for the
  Sedov problem in `Exec/HydroTests/Prob.cpp`, and `max_prob_param` raised
  to 32.
  - `nphase`: number of ambient phases.
  - `phase_rho`, `phase_p`: phase density and pressure contrast. With
    `phase_p=1` the phases are in pressure equilibrium, so they stay exactly
    as set until a blast reaches them.
  - `phase_block`: clump size in cells.
  - `phase_pow2`: phase densities are exact powers of two.
  - `nlayer`: alternate clumpy and smooth layers along z. z is the slowest
    index, so one 4 MiB chunk lies in one layer.
  - `smooth_amp`, `smooth_k`, `smooth_exp`: the smooth layers' density wave,
    `rho0 * exp(amp * wave)` when `smooth_exp=1`.
  - `nblast`: number of explosions.
  - `noise_dens`, `noise_vel`: per-cell noise in the ambient density and
    velocity.
- `../nyx/patches/nyx-dump-select-derived.patch`: two environment variables
  for the raw field dump.
  - `NYX_DUMP_STATE="density rho_E"` writes only those state fields, the way
    `amr.plot_vars` selects plotfile variables.
  - `NYX_DUMP_DERIVED="logden ..."` also writes those derived fields.

Chosen setup (`nyx_probe.sh final25g ...`, see `/mnt/nvme0/v2-work/runs/nyx_tuning.log`):

```
NCELL=256 STEPS=2600 PLOT_INT=20 STATE="density rho_E" DERIVED="logden"
prob.nphase=8 prob.phase_pow2=1 prob.phase_p=1 prob.nlayer=4
prob.smooth_exp=1 prob.smooth_amp=1.0 prob.smooth_k=1
prob.nblast=1 prob.exp_energy=0.05 prob.p_ambient=7.62939453125e-06
```

## Why this works

- **Clumpy layers favour `ans` with byte shuffle.** Each cell holds one of 8
  power-of-two densities, mixed at random, so the data has low byte entropy
  but no smoothness. `ans` with byte shuffle compresses it about 5.7×;
  `spratio` gets about 1.2×.
- **Smooth layers favour `spratio`.** The density wave is gentle, so
  `spratio`'s neighbour prediction works well there.
- **The model reads both correctly.** The trained model, unchanged, ranks
  both kinds well. `pattern_lab.py` finds such pairs from the training sweep
  tool and the model's predictions.
- **The data changes over time.** The blast sweeps through both kinds of
  layer, turning them into shocked flow.

## Tuning history (128³ unless noted; cost vs best single codec)

| Probe | Possible gain | NeuroPress, learning, 1 pass |
|---|---|---|
| Sedov, state fields only | 0.1% | +38% |
| + 12 derived fields | 3.8% | +43% |
| 8 phases with different pressures | 5.4% | +8.8% |
| 8 phases in pressure equilibrium | 5.6% | +11.5% |
| + clumpy / smooth layers, power-of-two levels, `exp` wave | 13.7% | +17.5% |
| same, fields `density`, `logden`, `rho_E` only (offline replay) | 19.3% | −7.0% |
| same at 256³, 4 layers | 26.8% | −10.1% (replay); **−11.4% through Clio** |

What hurt NeuroPress, so these were dropped from the output:

- **Momentum fields.** They are exactly zero outside the blast, and the model
  over-predicts the ratio of bit-shuffle codecs on mostly-zero chunks by up
  to 1,000×.
- **`Temp`, `soundspeed`, `divu`, `magvort`, `MachNumber`.** Their chunks are
  noisy, so no codec gains much, or the model mispredicts them.

## VPIC: Weibel deck with layered slabs

The deck `../vpic/weibel_clio.cxx` (LAYERED SLABS) cuts the box along z into
slabs of 16 cells, one letter per slab in `VPIC_SLABS`:

- `C`: vacuum with a frozen "clumpy" electrostatic field (a few exact values).
- `S`: vacuum with a frozen smooth field (rounded to a 2^-22 grid).
- `P`: plasma. The plasma streams out of the `P` slabs over time.

A cell size of 2^-5 (`VPIC_CELL=0.03125`) keeps the frozen fields bit-exact.
At 254 x 254 cells per plane, one 4 MiB chunk of a field lies in one slab.

Chosen setup (`vpic_probe.sh final25g ...`, the workload `vpic-slabs`, see
`/mnt/nvme0/v2-work/runs/vpic_tuning.log` on Chameleon):

```
NCELL=254 NZ=1022 NPPC=8 STEPS=303 DUMP_INT=3 (clean-div 0)
VPIC_SLABS=CSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCPPSCSCSCSCSCSCSCSCSCSCSCSCSCSCSCC
VPIC_CELL=0.03125 VPIC_DT_DX=0.5 VPIC_VTHE=0.03 VPIC_VTHEX=0.006 VPIC_DUMP_VARS=ez
```

The run log prints "254^3"; the grid is 254 x 254 x 1022 (`gen_fields.sh`
passes `--nz` to the deck as `VPIC_NZ`).

## incflo: Rayleigh-Taylor with frozen layered tracers

incflo (AMReX-Fluids) `6f82b5f`, AMReX `8cb3256`, AMReX-Hydro `a153304`;
built by `../incflo/build_incflo.sh` (CUDA sm_80, no MPI, EB on, double
precision) with `../incflo/patches/incflo-rt-layered-tracer.patch`. The patch
adds `prob.tracer_layers` (z-layers alternating clumpy / smooth, per tracer
component, odd components swapped), `prob.tracer_block` (clump size in cells;
1 = per-cell), `prob.tracer_nphase` (8 power-of-two levels), `prob.smooth_amp`,
`prob.smooth_k`. Plotfiles become `fields/plt<step>/<field>.f32` with
`../incflo/plt_to_raw.py` (float32, z slowest); `incflo_probe.sh` runs it all.

Workload `ref-incflo-tune-final25g` (79 frames x 5 fields, 395 files, 25 GiB,
6320 chunks):

```
INPUTS=../incflo/inputs.rt NCELL="256 256 256" STOP=12 PER=0.15 PLOT="velz density tracer" \
  ./incflo_probe.sh final25g "..." prob.tracer_layers=4 prob.tracer_block=1 incflo.ntrac=3 \
  "incflo.mu_s=0 0 0" incflo.advect_tracer=false "geometry.prob_hi=1. 1. 1."
# sha256 plt00000/tracer0.f32 1537ae0515cdeac16587cd5807bb62993df30292f8d4fa21fae1ce4fc4904b45
#        plt01328/velz.f32    b6550e40611b130c509c19484edf9242aff9b9af40f3467bdac95189ad0f15af
```

256^3 in a 1x1x1 box keeps probe D's cell size (1/256); 256x256x512 needs
about 45 GB of GPU memory (AMReX's arena is 30 GB on a 40 GB A100). The
tracers are frozen (`advect_tracer=false`), like VPIC's `C`/`S` slabs: moving
clumps lose their exact values within a few time units. Benchmark cost model
1/40/2.8 at 0.5 GB/s, 10 reads (chosen with `np_cost_sweep.py --wide`).

| Probe (128x128x256, t <= 10) | Possible gain (1/1/1) | NP learning pass 1 |
|---|---|---|
| A: stock RT, 8 fields | 2.6% | +28.7% |
| B: moving layered tracer, blocks of 16, g = -0.02 | 3.2% | +32.5% |
| C: frozen per-cell tracer, 8 fields | 9.7% | +18.7% |
| D: 3 frozen tracers + velz + density | 28.7% | -8.6% |

## Reproduce the benchmark workloads (another machine, e.g. Delta)

The two workloads of the benchmark, with the exact settings:

| Workload | Script | Dump | Chunks of 4 MiB |
|---|---|---|---|
| `nyx-multiphase-50g` | `gen_nyx_multiphase_50g.sh FIELDS_DIR [STAGE_ROOT]` | 801 files x 67108864 B (267 frames x density, rho_E, logden) | 12816 |
| `vpic-slabs` | `gen_vpic_slabs.sh FIELDS_DIR [STAGE_ROOT]` | 101 files x 268435456 B (101 frames x ez) | 6464 |

1. Build the simulation (both write their dumps with one GPU):
   - Nyx: commit `4ecfea2a` (AMReX `6e875b7cc`) with the four patches of
     `../nyx/patches`, as `../nyx/README.md` "Building Nyx" says. The four
     patches give exactly the source used here.
   - VPIC: VPIC-Kokkos `f01d295`, deck by `../vpic/build_deck.sh`.
2. Move the root paths. The benchmark and tuning scripts (about 40 files under
   `paper-benchmark/`) use `/mnt/nvme0/v2-work` (staged chunks, `runs/`,
   `baselines/`) and `/mnt/nvme0/tune` (tuning dumps). Point both at one
   directory on the other machine, in your copy of the repository:

   ```bash
   NEW=/your/scratch/dir      # e.g. a PFS directory on Delta
   grep -rl '/mnt/nvme0/' paper-benchmark | xargs sed -i "s#/mnt/nvme0/#$NEW/#g"
   mkdir -p $NEW/v2-work/runs $NEW/v2-work/baselines $NEW/tune
   ```
3. Run the script. `FIELDS_DIR` can be on any file system (a PFS too). Give
   `STAGE_ROOT=$NEW/v2-work` to also cut the dump into
   `$NEW/v2-work/<workload>/fields`, the chunk files that
   `../model-accuracy/run_kmeans_parallel.sh` replays.
4. Check the result: the scripts print the number of files and the sha256 of
   the first and last frame. The expected values are in the script headers.
   A different GPU or build can change the bits; then compare the chunk count
   and the exhaustive search (best single codec, possible gain).
5. The first `run_kmeans_parallel.sh` run of a workload makes its exhaustive
   search (all 45 settings on every chunk, 1 process) and stores it under
   `baselines/<workload>/exhaustive`; later runs reuse it.

## Multi-process benchmark (processes x chunks in flight)

`../model-accuracy/run_workload_bench.sh` runs the full benchmark of one staged
workload, as done for Nyx and VPIC: best single codec, NeuroPress learning and
oracle at 1x1, 2x8, 4x8 and 8x8, each timed read followed by one k-means
iteration (`KMEANS=8`), the k-means check of every process log
(`check_kmeans.sh`), then the compare CSVs and one figure.

```bash
cd paper-benchmark/model-accuracy
# W = w_ct,w_dt,w_io; COST_BW in bytes per ms; READS timed reads
GOVERNOR= PYTHON=$(which python3) CONFIGS="1x1 2x8 4x8 8x8" \
  ./run_workload_bench.sh nyx-multiphase-50g 1,10,10 1000000 10 <figure dir>
```

On another machine (e.g. a Delta GPU node):
- Rebuild and install first: the replay tool (`neuropress_field_replay`) needs
  the wait after `cudaMemset` in its last bit-exact check; without it, runs
  with many processes can report raw chunks as "NOT bit-exact" although the
  stored data is correct.
- `GOVERNOR=` (empty) leaves the CPU frequency governor alone; the default sets
  it to `performance` with `sudo`, which a batch job does not have.
- `PYTHON` is the Python with numpy, pandas and matplotlib (default
  `~/np-venv/bin/python`).
- The runs pin each process to the GPU's NUMA node (`numactl`); `PIN=0` turns
  this off, `NUMA_NODE=<n>` selects the node.
- Use one whole GPU node per benchmark (no other GPU work during the timed
  runs). The first configuration also makes the exhaustive search if
  `baselines/<workload>/exhaustive` is missing (all 45 settings, 1 process).
- Choose the cost model offline first: `np_cost_sweep.py <workload>` then
  `pick_cost_model.py <workload>` (NeuroPress runtime and ratio vs the best
  single codec per model); `opp_grid.py <workload>` gives the possible gain.

## Scripts

- `gen_nyx_multiphase_50g.sh`, `gen_vpic_slabs.sh`: make the two benchmark
  workloads with the exact settings (see above).
- `nyx_probe.sh`, `vpic_probe.sh`, `warpx_probe.sh`, `gs_probe.sh`, `incflo_probe.sh`: one probe
  run (generate, stage, exhaustive search, score); `data_probe.sh NAME DIR`:
  the same for data already on disk. Each deletes its stored (compressed)
  data after the search.
- `opp_grid.py`: possible gain over a grid of cost models (and field subsets).
- `subset_search.py`: field subsets with the largest possible gain at a minimum size.
- `plot_probes.py`: possible gain of several probes (per model and per field).
- `learn_variants.py`: offline test of NeuroPress learning variants.
- `pick_cost_model.py`: ranks `np_cost_sweep.py`'s models by NeuroPress's
  runtime and ratio gain together.
- `probe_eval.py`: opportunity per field, plus NeuroPress's result from the
  learning replay.
- `np_mistakes.py`: where NeuroPress's extra cost over the best single codec
  comes from.
- `pattern_lab.py`: candidate chunk patterns, swept and scored offline.
- `tune_sims.sh`, `tune_report.py`: parameter sweeps for LAMMPS and Nyx.
