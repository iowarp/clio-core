# Nyx workload — Clio + NeuroPress dynamic compression

A cosmological-hydrodynamics workload for the paper's compression benchmarks:
a **Sedov blast wave** computed by [Nyx](https://github.com/AMReX-Astro/Nyx)
(AMReX, CUDA, **float32**), dumped to flat field files, then replayed through
Clio with NeuroPress choosing a codec per chunk.

It is the deliberate counterpart to `../lammps`. Same policies, same chunking,
same aggregator — but the data could hardly be more different: LJ melt is
float64 and nearly incompressible (~1.1×), Sedov is float32 and highly
structured (up to 156×). Two of this project's findings only become visible
by holding the method fixed and changing the data that way.

This file is the operational document: how to build and run the workload, how
to get its data out, and the parameters that make that data evolve.
**Measurements — compression ratios, error-bound behaviour, the evolution
study's rankings — are in [`RESULTS.md`](RESULTS.md).**

## Why two phases

Nyx is an AMReX application with no library interface to embed, so the LAMMPS
approach — link it in, read its arrays between steps — is not available.
Upstream NeuroPress's own Nyx benchmark does not embed it either: it patches
Nyx to dump raw fields and sweeps those files offline. This mirrors that.

```
  phase 1: ./gen_fields.sh     Nyx (GPU) ──► plt00000/fab0000_comp00_density.f32 …
  phase 2: ./run_config.sh     files ──► Clio compressor ──► NeuroPress ──► CTE tier
```

Splitting them buys something the LAMMPS benchmark cannot have: **every policy
replays the identical bytes**. A Kokkos LAMMPS trajectory is not
bit-reproducible, so its policies see slightly different data; here the
comparison is exact.

## The patches

Four. The first two are needed for every run — see "Building Nyx" for why the
second one's absence is easy to miss. The last two add the options the
multiphase workload uses (see [Multiphase workload](#multiphase-workload-nyx-multiphase));
with their options left at the defaults Nyx behaves exactly as without them.

### `nyx-raw-field-dump.patch`

Adds one self-contained function to
`Source/IO/Nyx_output.cpp` (+87 lines) that writes each FArrayBox component as
a flat float32 file when `NYX_DUMP_FIELDS=1`.

It is derived from the dump block in upstream's
`benchmarks/nyx/patches/Nyx_output.cpp` but deliberately **not** gated on
`AMREX_USE_GPUCOMPRESS`: nothing in it calls a compressor, so Nyx links
neither NeuroPress nor Clio, and the files it produces are equally usable by
either. It is also placed before the HDF5 branch rather than inside it, so it
fires whether or not the build has AMReX HDF5, and it reads the hydro state
directly instead of `derive()`.

Float32 always — downcasting if `amrex::Real` is double — so a consumer never
has to ask how Nyx was configured.

### `nyx-comoving-a-single-precision-eps.patch`

Fixes `get_comoving_a: invalid time`, which aborts a run *after* every dump for
that step has been written — so the output is complete and only the exit status
is lost, which is why it reads as flaky rather than as a bug.

`get_comoving_a()` decides "is `time` at an end of the current a-interval" with
`eps = 1e-4 * (new_a_time - old_a_time)`. That is `1e-4` of the interval
**width**, added to the interval's absolute **position**. In a
single-precision build both `old_a_time ± eps` round straight back to
`old_a_time`, the window collapses to the empty interval, and a `time` that is
*exactly* `old_a_time` — the commonest query there is — falls through every
branch to `amrex::Error()`. Double-precision builds never see it.

The fix floors `eps` at a few ULP of the interval's own magnitude and makes the
endpoint comparisons inclusive. No physics change: the default run's field
dumps are bit-identical to the unpatched binary.

### `nyx-sedov-multiphase.patch`

New `prob.*` options for the Sedov problem (`Exec/HydroTests/Prob.cpp`) that
change the gas the blast goes off in. Upstream fills the box with one uniform,
motionless gas. With the defaults below the initial state is that same uniform
gas.

| Option | Default | Effect |
|---|---|---|
| `prob.nphase` | 0 | > 1: each clump of cells is one of `nphase` gas phases, drawn at random per clump (deterministic hash of the cell index) |
| `prob.phase_rho` | 100 | density contrast between the densest and the lightest phase (geometric ladder) |
| `prob.phase_pow2` | 0 | 1: phase densities are exact powers of two, `dens_ambient * 2^(q - nphase/2)`, instead of the ladder |
| `prob.phase_p` | 10 | pressure contrast between phases; **1 = pressure equilibrium**, so the gas stays exactly at rest until the blast reaches it |
| `prob.phase_block` | 1 | cells per side of one clump (1 = every cell independently) |
| `prob.nlayer` | 0 | > 0: the box is cut into `nlayer` slabs along z, alternately clumpy (the phases above) and smooth |
| `prob.smooth_amp`, `prob.smooth_k`, `prob.smooth_exp` | 0.5, 2, 0 | smooth slabs' density: `dens_ambient * (1 + amp * w)`, or `dens_ambient * exp(amp * w)` when `smooth_exp=1`, with `w = sin(2πkx) sin(2πky+0.7) sin(2πkz+1.3)` |
| `prob.nblast` | 1 | number of explosions; the first is the centred Sedov blast, the others sit at fixed pseudo-random positions |
| `prob.noise_dens`, `prob.noise_vel` | 0, 0 | per-cell random perturbation of the ambient density (relative) and velocity |

It also raises `max_prob_param` in `Source/Driver/Nyx.H` from 20 to 32:
these options are stored in that fixed-size array, and the original size
overflows (a segfault at initialisation) once the layer options are used.

### `nyx-dump-select-derived.patch`

Two environment variables for the raw field dump (`Source/IO/Nyx_output.cpp`),
both off by default:

- `NYX_DUMP_STATE="density rho_E"` writes only the listed hydro-state
  components, the way `amr.plot_vars` selects a plotfile's. Unset or empty
  writes all six, as before.
- `NYX_DUMP_DERIVED="logden pressure …"` also writes those derived quantities
  (any name Nyx's `derive()` knows, e.g. `pressure`, `soundspeed`,
  `MachNumber`, `divu`, `logden`, `x_velocity`, `magvort`, `Temp`; unknown
  names are skipped), as further components of the same step, so each is one
  more `fab0000_compNN_<name>.f32` file in the same `pltNNNNN/` directory.

## Building Nyx

```bash
git clone https://github.com/AMReX-Astro/Nyx.git ~/src/Nyx
cd ~/src/Nyx
git checkout 4ecfea2a      # the commit the benchmark data were made with
git submodule update --init --recursive   # AMReX 6e875b7cc
git apply <clio>/paper-benchmark/nyx/patches/nyx-raw-field-dump.patch
git apply <clio>/paper-benchmark/nyx/patches/nyx-comoving-a-single-precision-eps.patch
# only for the multiphase workload (harmless otherwise: defaults are upstream's)
git apply <clio>/paper-benchmark/nyx/patches/nyx-sedov-multiphase.patch
git apply <clio>/paper-benchmark/nyx/patches/nyx-dump-select-derived.patch
cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release \
      -DNyx_MPI=NO -DNyx_OMP=NO -DNyx_HYDRO=YES -DNyx_HEATCOOL=NO \
      -DNyx_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=Ampere \
      -DAMReX_PRECISION=SINGLE -DAMReX_PARTICLES_PRECISION=SINGLE
cmake --build build-clio --target nyx_HydroTests -j
```

**Both patches, and REBUILD after applying them.** The `get_comoving_a` fix
(above) is easy to miss: the default Sedov run never reaches its trigger, so a
binary built before the patch looks perfectly healthy until a longer run or
another problem aborts partway — LyA aborts at exactly z=6. Nothing checks that
the binary is newer than the patch, so if in doubt compare the timestamps:

```bash
ls -la ~/src/Nyx/Source/Driver/comoving.cpp ~/src/Nyx/build-clio/Exec/*/nyx_*
```

`AMReX_PRECISION=SINGLE` is upstream's recommendation for compression
benchmarks and matters here for a specific reason — see the shuffle result in
`RESULTS.md`. `Nyx_MPI=NO` keeps a single-rank benchmark from needing a parallel HDF5
build; nothing in this path writes HDF5 anyway.

The Clio side is `bin/neuropress_field_replay`
(`context-transfer-engine/compressor/example/neuropress_field_replay/`), which
has no simulation dependency at all:

```bash
cmake --build <clio>/build --target neuropress_field_replay
```

## Running

There are **two routes**, and they are not interchangeable.

**In situ** — the simulation hands the compressor `fab.dataPtr(comp)`, AMReX
DEVICE memory, uncopied, from inside `Nyx::updateInSitu()`, so quantization,
byte shuffle and codec selection all take their GPU paths. This is the route
`run_benchmark.sh` uses, and the one to prefer:

```bash
./run_config_insitu.sh explore-balance --ncell 128 --steps 1000 --int 10 \
    --chunk 8388608 --verify --results /tmp/nyx --tag base
```

It sets `CLIO_NEUROPRESS_REQUIRE_DEVICE=1` unconditionally, so a host-resident
chunk is REFUSED rather than quietly computed on the CPU — a silent fallback is
indistinguishable from success in the results, since the ratios come out
identical and only the timings move. Configs are `explore-balance`,
`explore-ratio`, `explore-speed`; `--check-bound` replaces `--verify` on a
lossy run.

**Replay** — dump the fields to disk, then sweep the files offline. Every
policy replays byte-identical input, which the in-situ route cannot promise
because each configuration re-runs the simulation:

```bash
./gen_fields.sh                              # Nyx -> ./fields (~1 GB, a few min)
./run_config.sh dynamic                      # one policy over those files
../collect.py results/                       # re-aggregate
```

**`CLIO_NEUROPRESS_STAGE_H2D=1` is required for the replay route and nothing
sets it.** `CLIO_NEUROPRESS_REQUIRE_DEVICE` defaults *on*, NeuroPress
preprocessing is CUDA-only, and a file-replay driver hands the compressor host
memory by construction — so without the stage every chunk is refused and every
blob fails `rc=11` with `0 B in -> 0 B on the tier`.

Every run verifies itself: each blob is read back through the decompressor and
its FNV-1a-64 digest compared with the digest of the bytes staged.

`gen_fields.sh` takes `--ncell --steps --plot-int --out --bin --keep-plt`;
`run_config.sh` takes `<config> --fields --chunk --max-files --f64 --results --tag`. Defaults
are 128³, 200 steps, dumping every 10 → 21 frames × 6 components × 8 MiB ≈
1,008 MiB, chunked at 4 MiB into 252 chunks — deliberately the same chunk
count as the LAMMPS benchmark.

Fields dumped are the hydro state: `density`, `xmom`, `ymom`, `zmom`,
`rho_E`, `rho_e`.

## Looking at the data

Dump a small run and render it:

```bash
./gen_fields.sh --ncell 64 --steps 400 --plot-int 16 --exp-energy 10 \
                --keep-plt --out /tmp/nyx-quick             # ~2 s, 157 MiB
../plot/viz_fields.py --fields /tmp/nyx-quick --plt /tmp/nyx-quick-plotfiles \
                --out /tmp/nyx-viz                          # ~4 s
```

Pass `--axis-for x:zmom`: a vector component is antisymmetric about the
mid-plane normal to its own axis, so z-momentum is ~0 across the whole z
mid-plane and the montage comes out blank while the field is perfectly healthy.

`../plot/viz_fields.py` reads the flat `.f32` dumps with nothing but numpy, and that is
the point: those files, not Nyx's plotfiles, are what the compressor is handed,
so what it draws is what a run compresses. Per field it writes a montage of
mid-plane slices across the run and the same slices as a GIF; once per run it
writes `evolution.png`, which puts shock radius, fraction of the domain off
ambient, and a zlib stand-in for the compression ratio on one time axis.

**Does this run actually evolve?** Point `plot/viz_fields.py` at a dataset and the
answer is visible rather than inferred — the `evolution.png` panel carries shock
radius, the fraction of the domain off ambient, and the share of cells
bit-identical to the previous dump. At the deck's `exp_energy=1` and a short
run, the shock barely leaves the centre and most of the box never changes; the
64³ recipe above reaches roughly half the box. `RESULTS.md` has the measured
version of both.

Two details that are easy to get wrong and quiet when you do:

- **The dumps are Fortran-ordered** — `copyToMem` packs in the FAB's own
  layout, x fastest. Read C-ordered, the axes come back reversed. Density is
  spherically symmetric so it looks fine; `xmom` gets sliced along *x*, where
  it is antisymmetric and the midplane is ~0, and every frame renders blank.
  The check is that `xmom` on the z-midplane satisfies `f(x) = -f(-x)`, which
  it does to 7e-6 in float32.
- **Signed fields need a percentile color scale, not a max.** The initial
  energy deposit puts |xmom| ≈ 100 into two cells while the shell that matters
  carries ≈ 10; scale to the max and 24 of 26 frames are blank.

### The tools Nyx itself documents

Nyx writes AMReX plotfiles, which `PostProcessing.rst` says are read natively
by **VisIt, ParaView, and yt**, and it ships `Util/Diagnostics` (an AMReX
program that opens a plotfile) as a starting point for custom analysis. Those
are the right tools if you want AMR levels, the derived fields (`pressure`,
`Temp`, `Ne`) the raw dumps do not carry, or 3-D rendering.

They are not the tools for this benchmark. The plotfiles are a *different
serialisation* of the state — 350 MiB against 157 MiB of dumps on the quick run
— so measuring them measures AMReX's file format, not the bytes Clio moves.
`gen_fields.sh` therefore discards them by default; `--keep-plt` writes them to
`<out>-plotfiles`, a sibling of the dump directory rather than a directory
inside it, because `run_config.sh` sizes the payload with `du -sb $FIELDS` and
counts frames with `ls $FIELDS | grep -c ^plt`, and "plotfiles" would corrupt
both.

Even with `--keep-plt`, `../plot/viz_fields.py` opens the plotfiles only to read
`Header` for the simulation time, so frames are labelled `t=0.0236` rather than
`dump 25`. Without `--plt` it falls back to dump indices and everything else
still works. (yt is not installed here and there is no `pip`, so the documented
route was not exercised on this machine.)

## Default Evolving Benchmark Configuration

Selected by a 1,000-timestep hyperparameter study (`../evolution.py`, six
configurations, 101 dumps each, 4,800 block samples per configuration).

Parameters:

- `nyx.cfl = 0.8`  (`--cfl`, Nyx's own documented default; the Sedov deck ships 0.5)
- `prob.exp_energy` — left at the deck's 1.0; it does not matter, see below
- `--ncell 128`, `stop_time = 1.0` so `max_step` is what ends the run

Evolution sampling interval: 10 timesteps

### Parameters tested

Two, chosen because Nyx's own documentation says they are what sets the
timestep and what sets the blast. Everything else in the deck was left alone.

| | `nyx.cfl` | `prob.exp_energy` |
|---|---|---|
| **controls** | the timestep as a fraction of the CFL limit: "defines the timestep as dt = cfl \* dx / umax_hydro" | the energy deposited in the initial sphere: `p_exp = (gamma-1) * exp_energy / vctr`, `vctr = 4/3 pi r_init^3` |
| **why it should matter** | a larger timestep moves the shock further per step, so a fixed block sees more change between dumps | Sedov-Taylor gives `R = 1.033 (E t^2 / rho)^(1/5)`, so a stronger blast sweeps more of the domain |
| **official reference** | `NyxInputs.rst` — "CFL number for hydro", `Real > 0 and <= 1`, **default 0.8** | `Exec/HydroTests/Prob.cpp`; deck value in `inputs.3d.sph.sedov` |
| **values tested** | 0.5 (deck), 0.8, 0.9 | 1.0 (deck), 10.0, 100.0 |
| **outcome** | **decisive** — 0.5 -> 0.8 moves last-decile cells-identical from 76.3% to 53.2% | **exactly inert** at a fixed step count; moves the same number by 0.3 points over a 100x range |

Not varied, and why: `--ncell` is payload sizing rather than evolution rate, and
the chunk shape depends on it (`--chunk` must be `ncell^3 x 4`);
`prob.dens_ambient` enters `R` through the same `(E/rho)^(1/5)` as `exp_energy`
and so cancels for the same reason; `prob.r_init` changes the initial condition
rather than the rate.

### Why these parameters

**At a fixed step count `exp_energy` does nothing, and that is exact rather than
approximate.** Sedov is self-similar. With `R(t) = 1.033 (E t^2 / rho)^(1/5)` the
shock speed is `u_s = (2/5) R/t`, and the timestep is CFL-limited to
`dt = cfl * dx / (c1 * u_s)` for a `c1` set by the post-shock sound speed. So

```
dR per step = u_s * dt = cfl * dx / c1
```

— **a fixed number of cells per timestep, independent of E, rho and t.** A
stronger blast moves the shock faster and shrinks the timestep by exactly the
same factor. `--exp-energy` therefore buys reach per unit *time* and nothing per
unit *step*, and the measurements bear that out over a hundredfold range in E
while `--cfl` moves the same quantity substantially.

This does not contradict `RESULTS.md`'s `--exp-energy` table, which is measured
at the deck's fixed `stop_time = 0.01`: there a stronger blast buys more *steps*
and the ratio spread follows. Both are true and they answer different questions.
**If the run is ended by `stop_time`, raise `--exp-energy`; if it is ended by
`--steps`, raise `--cfl`.**

**Why not `cfl = 0.9`,** which scores slightly higher. Nyx's own mass sum over
1,000 steps drifts by roughly a thousand times more at 0.9 than at 0.8, because
the scheme produces negative densities that Nyx's `Enforce minimum density`
floor clamps, and the clamping shows up as mass that is not conserved. The run
still exits 0 and holds no NaN, so the evolution metric cannot see it — this had
to be checked separately. 0.8 is Nyx's own documented default and keeps mass to
float32 roundoff.

Rankings, per-decile curves and the mass-conservation figures are in
`RESULTS.md`.

### References

- [`AMReX-Astro/Nyx`](https://github.com/AMReX-Astro/Nyx) (pinned at `4ecfea2`),
  `Docs/sphinx_documentation/source/NyxInputs.rst`: **`nyx.cfl`** — "CFL number
  for hydro", `Real > 0 and <= 1`, **default 0.8**; "defines the timestep as
  dt = cfl \* dx / umax_hydro".
- `Exec/HydroTests/inputs.3d.sph.sedov` — the deck, which sets `nyx.cfl = 0.5`,
  `prob.exp_energy = 1.0`, `prob.r_init = 0.01`, `prob.dens_ambient = 1.0`,
  `prob.p_ambient = 1.e-5`, `stop_time = 0.01`.
- `Exec/HydroTests/Prob.cpp` — `exp_energy` enters as
  `p_exp = (gamma - 1) * exp_energy / vctr` over `vctr = 4/3 pi r_init^3`, i.e.
  the energy deposited in the initial sphere.

## Choosing the parameters

### Sizing: payload is set by `--ncell` and the frame count

One frame is every component of the hydro state, so

```
frame bytes  = ncell^3 x 6 components x 4 B      # 128^3 -> 48 MiB, 256^3 -> 384 MiB
frames       = steps / int                        # capped by stop_time, see below
payload      = frame bytes x frames
```

| target | ncell | int | frames | payload |
|---|---|---|---|---|
| ~1 GB | 128 | 32 | 21 | 1.01 GB |
| ~30 GB | 256 | 6 | ~80 | ~30 GB |

Set `--chunk` to the per-component size (`ncell^3 x 4`) to get one chunk per
component: 8388608 at 128³, 67108864 at 256³.

### `--steps` is NOT what stops the run — `stop_time` is

The deck ends at `stop_time = 0.01` whatever `max_step` says, and because the
timestep is CFL-limited the steps needed to reach it scale with resolution.
Measured, at the default `exp_energy=1`:

| ncell | steps actually run when asked for 1000 |
|---|---|
| 128 | ~300 |
| 256 | **583** |

So a profile that specifies 1000 steps does not get them. Use `--stop-time` to
extend the run, or `--exp-energy` (below), which raises the step count as a
side effect.

### `--deck`: Sedov is a hydro unit problem, not Nyx's science

`inputs.3d.sph.sedov` comes from `Exec/HydroTests` and runs with
`nyx.do_grav = 0`, no dark matter and `comoving_h = 0`, so it exercises Nyx
purely as a hydro solver. `run.sh --deck PATH` points at another `Exec`'s
inputs — the in-situ hook lives in the shared `Source/IO/Nyx_output.cpp`, so
every Exec's binary already carries it.

**LyA, the Lyman-alpha forest problem Nyx is actually for, is the obvious
candidate and is the wrong benchmark.** It runs end to end (z=159 -> 2, 422
steps) once given absolute paths for `64sssss_20mpc.nyx` and `TREECOOL_middle`
(they resolve against the working directory) and gravity tolerances a
single-precision build can reach — its shipped `1e-10` is unattainable, MLMG
bottoms out at `resid/bnorm = 7.9e-6`, so `gravity.ml_tol=gravity.sl_tol=2e-5`.
But its compression ratio sits at **1.0005 .. 1.0022 for the entire history**:
cosmological float32 fields are incompressible losslessly at every redshift, so
there is no signal to measure. Sedov with a raised blast energy is the
configuration that produces one.

## Multiphase workload (`nyx-multiphase`)

### Purpose

This variant exists to answer one question: does choosing a codec per chunk
pay on a simulation? On the default Sedov run it barely can.
- **Default Sedov:** every chunk is smooth float32, so `spratio` is the
  cheapest setting almost everywhere. Choosing per chunk could save at most
  0.4% (`nyx-full`), and NeuroPress v2 loses to that single codec.
- **This variant:** the same Sedov blast goes off in a deliberately
  heterogeneous gas, so that different chunks need different codecs and the
  data changes as the blast sweeps through it.

The settings were found by searching for exactly that, not chosen for the
physics; see [What is contrived](#what-is-contrived).

### Setup

- **The box:** cut along z into 4 slabs that alternate:
  - **Clumpy slabs** (z in [0, ¼) and [½, ¾)): every cell is independently one
    of 8 gas phases, with densities exactly 1/16, 1/8, …, 8, all at the same
    pressure.
  - **Smooth slabs** (z in [¼, ½) and [¾, 1)): density `exp(sin·sin·sin)`,
    from 0.37 to 2.7, one gentle wave per box, at the same pressure.
- **Why it stays put:** because the pressure is uniform, the gas does not
  move until the blast reaches it.
- **The blast:** a weak centred blast (`exp_energy` 0.05 instead of 1.0)
  slowly sweeps outward through both kinds of slab.
- **Output:** only `density` and `rho_E` from the hydro state, plus the
  derived `logden`.

```bash
cd <work dir>
NYX_DUMP_FIELDS=1 NYX_DUMP_DIR=<out> NYX_DUMP_STATE="density rho_E" NYX_DUMP_DERIVED="logden" \
  ~/src/Nyx/build-clio/Exec/HydroTests/nyx_HydroTests \
  ~/src/Nyx/Exec/HydroTests/inputs.3d.sph.sedov \
  amr.n_cell="256 256 256" amr.max_grid_size=256 \
  max_step=2660 amr.plot_int=10 amr.check_int=0 stop_time=1.0 nyx.cfl=0.8 nyx.v=0 amr.v=0 \
  prob.nphase=8 prob.phase_pow2=1 prob.phase_p=1 prob.nlayer=4 \
  prob.smooth_exp=1 prob.smooth_amp=1.0 prob.smooth_k=1 \
  prob.nblast=1 prob.exp_energy=0.05 prob.p_ambient=7.62939453125e-06
```

`../sim-tuning/nyx_probe.sh` wraps this command, stages the output and runs
the exhaustive search. The two saved runs differ only in the step count and
output interval:

| Workload | Steps, output every | Output steps | Chunks | Size |
|---|---|---|---|---|
| `nyx-multiphase-25g` | 2,600, every 20 | 131 | 6,288 | 24.6 GiB (26.4 GB) |
| `nyx-multiphase-50g` | 2,660, every 10 | 267 | 12,816 | 50.1 GiB (53.8 GB) |

`p_ambient` = 2⁻¹⁷ keeps the pressure, and so `rho_E`, an exact power of two
as well.

### Why it works

- **Each chunk is one kind of data.** The slabs run along z, the slowest
  index of a dumped field. At 256³ a field is 64 MiB, which is 16 chunks of
  16 z-planes each, and every slab is exactly 4 of those chunks.
- **Clumpy chunks favour `ans` with byte shuffle.** Values jump at random
  between 8 levels, so there is little information per byte but no
  smoothness. On early clumpy `density` chunks, `ans` and `zstd` with byte
  shuffle reach 5.7× and 6.9×; `spratio`, which predicts each value from its
  neighbour, gets 1.2×. Power-of-two levels matter: with arbitrary levels
  every byte of the value varies and byte-shuffled `ans` only reaches about
  2.5×.
- **Smooth chunks favour `spratio`,** provided the wave is gentle. Faster or
  steeper waves make `lz4` the cheapest setting, and the model mispredicts
  that.
- **The model reads both kinds correctly.** The candidate patterns were
  checked offline against the trained model's predictions
  (`../sim-tuning/pattern_lab.py`). The model is never retrained on these
  data.
- **The data changes over time.** As the blast sweeps outward, both kinds of
  slab turn into shocked flow, and the share of each kind of chunk shifts
  from step to step.

### Fields left out, and why

The fields written were chosen by measurement:

- **`xmom`, `ymom`, `zmom` hurt NeuroPress.** Outside the blast the gas is at
  rest, so momentum is exactly zero, and on mostly-zero chunks the trained
  model over-predicts the ratio of bit-shuffle settings by up to 1,000×.
- **`rho_e` is nearly constant** outside the blast, so it compresses
  extremely well and adds little.
- **`Temp`, `soundspeed`, `MachNumber`, `divu`, `magvort`, `x_velocity` don't
  help.** Their chunks are noisy, so no setting gains much, or the model
  mispredicts them.

Offline replay of the 128³ probe:

| Fields written | NeuroPress learning vs best single codec, one pass |
|---|---|
| `density`, `logden`, `rho_E` | −7.0% |
| `density`, `logden`, `rho_E`, `rho_e` | −0.8% |
| + `soundspeed` | +6.3% |
| + `soundspeed`, `Temp` | +11.9% |
| all 6 state fields + `logden`, `soundspeed`, `Temp` | +17.5% |

### Results

Measured through Clio, balanced 4-tier cost, every chunk read back
bit-exact. Learning is on and exploration off; the best single codec is the
cheapest of all 45 settings over the whole workload.

| | `nyx-full` (default Sedov) | `nyx-multiphase-25g` | `nyx-multiphase-50g` |
|---|---|---|---|
| Size | 19.1 GiB, 4,896 chunks | 24.6 GiB, 6,288 chunks | 50.1 GiB, 12,816 chunks |
| Best single codec | `spratio` | `ans` with byte shuffle | `ans` with byte shuffle |
| Most per-chunk choice could save | 0.4% | 22.4% | 22.0% |
| NeuroPress learning, cost vs best single | +14.4% | **−15.0%** (67% of the possible saving) | **−16.8%** (76%) |
| Compression ratio, best single → NeuroPress (oracle) | 6.23 → 5.50 | 1.91 → **2.13** (2.29) | 1.89 → **2.15** (2.27) |
| Measured write + read time, best single → NeuroPress (oracle) | 39.2 → 40.5 s | 86.7 → 87.1 s (78.2 s) | 160.0 → **147.6 s, −7.7%** (147.1 s) |

At 50 GiB NeuroPress also wins in measured wall-clock time:
- **Write** 89.9 → 80.5 s, **read** 70.0 → 67.1 s, which is within 0.3% of
  the per-chunk oracle.
- Its own overhead (prediction 3.3 s, the write-path decompress that labels
  learning 2.2 s) is outweighed by faster codecs (compress + decompress
  5.2 → 3.9 s) and 12% fewer bytes written.
- What it stores: `spratio` on 40% of chunks (smooth slabs and `rho_E`),
  `ans` with byte shuffle on 26% (clumpy slabs), and 41 other settings on the
  rest.

The cost model assumes slow storage tiers this machine doesn't have: every
run writes to one NVMe. So the cost gains are larger than the measured time
gains, and at 25 GiB the time gain does not appear at all.

### What is contrived

Say so wherever these numbers are used:

- **Not resolved physics.** A medium whose phase changes from one cell to the
  next has structure at the grid scale.
- **Picked for compression, not for science.** The power-of-two densities,
  the slabs aligned with chunk boundaries and the gentle wave all serve the
  codecs.
- **Output chosen for the model.** The fields written were chosen because the
  trained model handles them; with the momenta and temperature included,
  NeuroPress loses.

The full search, every probe in order with its result, is in
[`../sim-tuning/README.md`](../sim-tuning/README.md), and the probe log is
`/mnt/nvme0/v2-work/runs/nyx_tuning.log`.

## Notes

- Lossless throughout: `error_bound_ = 0`, which masks the 16 quantize actions.
  Verified rather than assumed: 2,016 blobs in the float32 sweep, 2,016 in the
  float64 control, and 756 recovered cold from the tier -- zero
  failures, and `quantize=0` / `psnr=-1` on every candidate the exploration
  runs measured.
- Element type is float32 (`Context::data_type_ = 1`) by default, float64
  (`= 2`) under `--f64`, matching the dumps either way.
- The float64 control needs a second Nyx build (drop the two
  `PRECISION=SINGLE` flags) and `NYX_DUMP_NATIVE=1` at dump time. Without that
  environment variable the patch downcasts to float32 whatever the build, which
  would make the control a silent duplicate of the baseline.
- These ratios are much larger than a cosmology production run would show. The
  Sedov blast is a point explosion into uniform background, so early frames are
  nearly constant; upstream reports 141–369× on the same problem. It is chosen
  for the spread it produces across a run, not as a realistic storage estimate.
- Each run picks a free TCP port, for the reason `../lammps/README.md` gives.
- `fields/` and `results/` are generated output and are not tracked.
