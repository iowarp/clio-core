# LAMMPS workload — Clio + NeuroPress dynamic compression

A molecular-dynamics workload for the paper's compression benchmarks. LAMMPS
runs a Lennard-Jones melt as a **library inside the benchmark process**, Clio's
runtime is hosted in that same process, and **NeuroPress chooses a codec per
chunk** from the simulation's own statistics. No file is written and no data
leaves the process to be compressed.

The workload is held constant across every run; the only variable is **how the
codec is chosen**, which makes the measurements in `RESULTS.md` a comparison of
selection policies rather than of workloads.

This file is the operational document: how to build and run the workload, how
to get its data out, and the parameters that make that data evolve.
**Measurements — compression ratios, error-bound behaviour, the evolution
study's rankings — are in [`RESULTS.md`](RESULTS.md).**

## What runs

```
 ┌───────────────── one process ─────────────────────────────────────────┐
 │  liblammps          driver                    Clio runtime (in-proc)  │
 │  run 0        ◄──── setup, forces at step 0                           │
 │   Atom::x,v,f ─────► gather (atom-ID order) ──► AsyncDynamicSchedule  │
 │  run 50       ◄──── advance 50 steps            └─ NeuroPress ranks   │
 │   Atom::x,v,f ─────► gather ─────────────────►     codec, compresses, │
 │  ...                                               stores to CTE tier │
 └───────────────────────────────────────────────────────────────────────┘
```

Default system: `BOX=80` → 4·80³ = **2,048,000 atoms**, 300 steps, a frame every
50 → 7 frames × 3 fields (position, velocity, force) × 49.15 MB = **984 MiB of
float64**, cut into **252 chunks of 4 MiB**.

LAMMPS stores the state as C `double` (`src/atom.h:75`, `double **x, **v, **f`),
so every element is 8 bytes. The driver declares that to Clio as
`Context::data_type_ = 2` with `error_bound_ = 0` (lossless), and the compressor
converts float64→float32 on the GPU for the model's statistics rather than
reinterpreting the bytes.

## Requirements

The driver is `bin/neuropress_lammps_lib`, built from
`context-transfer-engine/compressor/example/neuropress_lammps_lib/`. It needs a
CMake build of LAMMPS with the library present:

```bash
git clone --depth 1 --branch stable https://github.com/lammps/lammps.git ~/src/lammps
cmake -S ~/src/lammps/cmake -B ~/src/lammps/build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_MPI=OFF -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_STANDARD=17
cmake --build ~/src/lammps/build -j

cmake -S <clio> -B <clio>/build -DLAMMPS_SRC_DIR=$HOME/src/lammps \
      -DLAMMPS_BUILD_DIR=$HOME/src/lammps/build
cmake --build <clio>/build --target neuropress_lammps_lib
```

Set `Kokkos_ARCH_*` for your GPU. Without a LAMMPS tree the target is skipped
and the rest of Clio still configures.

## Running

```bash
./run_sweep.sh                          # every policy, 2M atoms — ~6.5 min
./run_sweep.sh --box 20 --steps 100     # quick shakedown
./run_sweep.sh --repeats 3              # variance
./run_config.sh dynamic                 # one policy
./collect.py results/                   # re-aggregate
./run_config.sh -h                      # all options
```

### Configuring the simulation

Size and sampling are run options; the physics is a set of pass-through
options that reach the deck as LAMMPS `-var`. Both `run_config.sh` and
`run_sweep.sh` accept them, so a whole sweep can be repeated at another
state point without editing anything.

| option | deck variable | default | |
|---|---|---|---|
| `--box N` | `BOX` | 80 | lattice cells per side; atoms = 4·N³ |
| `--steps N` | — | 300 | total timesteps |
| `--gap N` | `GAP` | 50 | store a frame every N steps |
| `--chunk B` | — | 4194304 | bytes per compressor call |
| `--density X` | `DENSITY` | 0.8442 | reduced density of the fcc lattice |
| `--temp X` | `TEMP` | **6.0** | initial temperature (upstream's example uses 3.0) |
| `--cutoff X` | `CUTOFF` | 2.5 | `lj/cut` pair cutoff, σ |
| `--skin X` | `SKIN` | **0.8** | neighbor-list skin (upstream 0.3) |
| `--every N` | `EVERY` | **5** | `neigh_modify every` (upstream 20) |
| `--seed N` | `SEED` | 87287 | velocity RNG seed |
| `--dt X` | `DT` | 0.005 | timestep |
| `--var K=V` | any | — | any other deck variable, repeatable |

The three bold defaults are the operating point chosen by the 1,000-timestep
evolution study, and they are set in `run_config.sh` rather than in the deck —
`in.melt` stays exactly upstream's melt example so it still runs standalone as
upstream wrote it. `--skin 0.8 --every 5` are a correctness condition at
`--temp 6.0`, not tuning: see "Default Evolving Benchmark Configuration".

### `--require-device` is a correctness flag, not a performance one

Without it the driver's default `--order id` gathers each chunk through
`lammps_gather_atoms` into HOST memory. NeuroPress's byte shuffle and quantizer
are CUDA-only -- the CPU implementations were deliberately removed -- so any
chunk whose ranked candidate wants a shuffle is REFUSED, `Compress` returns
rc=1, the blob is never stored, and the read back fails with rc=11. Measured at
box 40 / 50 steps: 7 refusals, 7 lost blobs under `dynamic`; 4 and 4 under
`explore-balance`. The run still exits 0 and prints a ratio; only the verify
line reveals it. `run_benchmark.sh` passes the flag for every LAMMPS cell.

### `in.melt` is stationary after step 40 — use `in.melt_ramp` to exercise the selector

The stock deck melts in about **40 of its 500 steps** and is a steady-state
liquid for the rest, so 24 of 26 frames are the same KIND of data. Entropy,
MAD and the ratio are flat, every frame picks the same action, and the run
says almost nothing about a selector whose whole job is to notice that the
data changed.

`in.melt_ramp` drives the thermostat target from a cold crystal to a hot
disordered fluid across the WHOLE run, so every frame is a different state
point:

```bash
./run_config.sh dynamic --deck $PWD/in.melt_ramp --box 20 --steps 2000 \
    --gap 80 --chunk 768000 --require-device --var NSTEPS=2000 \
    --raw /tmp/lmpr-raw --results /tmp/lmpr --tag ramp
```

| | `in.melt` | `in.melt_ramp` |
|---|---|---|
| `position` entropy, first → last | 7.03 → 7.02 | **5.42 → 7.07** |
| `force` MAD, first → last | 9.9e-14 → 18.5 | **5.51 → 78.3** |
| distinct codecs, `force` | 2 | **3** (bitcomp → zstd, some stored raw) |
| distinct codecs, `velocity` | 2 | **3** (bitcomp → ans, some stored raw) |
| byte-level same-as-prev spread | 1.0 pt | **5.2 pts** (23.8% → 18.6%) |
| MSD over the run | 1.41 σ² | **7.60 σ²** |

**THE RAMP IS DRIVEN BY THE GLOBAL `step`, not by `fix nvt`'s own ramp**, and
that is not a stylistic choice. The driver advances the simulation in GAP-step
segments (`run GAP pre no post no`), and `fix nvt temp T0 T1` interpolates over
the *current run command* — so it would sweep the entire range inside every
80-step segment and reset. An equal-style variable reading `step` is immune to
how the run is chopped up. `fix nve` + `fix langevin v_Tramp` rather than
`fix nvt` because langevin accepts a variable for its target (its `Tstop`
argument must still be numeric — `fix_langevin.cpp:76`).

### Looking at the data

This workload is in situ — LAMMPS runs as a library in the benchmark process
and no file is ever written — so there is nothing on disk to look at unless the
run asks for it. `--raw DIR` makes the driver write each staged blob's bytes,
which are exactly what NeuroPress compressed.

```bash
./run_config.sh dynamic --box 20 --steps 500 --gap 20 --chunk 768000 \
    --require-device --raw /tmp/lmp-raw --results /tmp/lmp-lossless --tag ll
./viz_atoms.py --raw /tmp/lmp-raw --out /tmp/lmp-viz          # ~20 s
```

`viz_atoms.py` draws a 1.2σ slab through the middle of the box, coloured by
speed, as a montage and a GIF — a slab and not the whole box, because 32,000
atoms projected through 33σ of depth is a uniform smear at every timestep and
the melt is invisible. `evolution.png` carries MSD, g(r), temperature, the
per-byte redundancy above, and a zlib stand-in for the ratio.

The physics is unambiguous: g(r) goes from sharp fcc shells out to 16σ at step 0
to a single broad liquid peak by step 260, temperature settles from 3.0 to 1.65,
and MSD grows linearly — textbook diffusion. **The melt is complete by about
step 40**, and the compression ratio has finished moving by then too:
position's zlib stand-in drops 7.58× → 1.06× over the first frame and is flat
for the remaining 25.

MSD uses minimum-image displacement accumulated frame to frame. LAMMPS wraps
coordinates into the box, so a plain `x[t] - x[0]` reads an atom crossing a face
as a box-sized jump — at step 500 that put a real -0.37 displacement in as
+33.2.

## Default Evolving Benchmark Configuration

Selected by a 1,000-timestep hyperparameter study (`../evolution.py`, seven
configurations, 101 frames each, 1,800 block samples per configuration).

Parameters:

- `--temp 6.0`   (`velocity all create`, upstream's melt example uses 3.0)
- `--skin 0.8`   (`neighbor`, upstream 0.3)
- `--every 5`    (`neigh_modify every`, upstream 20)
- `--box 40`, `--gap 10`, `fix nve`, everything else upstream's

Evolution sampling interval: 10 timesteps

### Parameters tested

Two that change the physics, and one pair that has to follow them to keep the
integration honest.

| | `velocity ... create T` | `neighbor` skin / `neigh_modify every` | the thermostat ramp (`in.melt_ramp`) |
|---|---|---|---|
| **controls** | the initial temperature: "generates an ensemble of velocities ... at the specified temperature" | the neighbour cutoff (force cutoff + skin) and how often the list is rebuilt | a target temperature that follows the global timestep, via `fix langevin` with an equal-style variable |
| **why it should matter** | hotter atoms decorrelate faster and move further per frame, so both velocity and position blocks change more | not an evolution knob at all — it bounds how far an atom may move between rebuilds before pairs are silently missed | every frame is a different state point, so the block statistics move monotonically instead of being stationary |
| **official reference** | [`velocity`](https://docs.lammps.org/velocity.html); `examples/melt/in.melt` uses 3.0 | [`neighbor`](https://docs.lammps.org/neighbor.html) — "All atom pairs within a neighbor cutoff distance equal to their force cutoff plus the *skin* distance are stored"; `in.melt` uses `0.3` / `every 20` | [`fix langevin`](https://docs.lammps.org/fix_langevin.html) — "Tstart can be a variable"; it "does NOT perform time integration", hence `fix nve` |
| **values tested** | 3.0 (upstream), 6.0 | 0.3 / 20 (upstream), 0.8 / 5 | none (`in.melt`); `T0=0.05 -> T1=6.0`; `T0=0.05 -> T1=12.0` |
| **outcome** | **the evolution knob** at matched neighbour settings | **required at T=6.0**: cuts NVE energy drift from -3.52% to -0.37% *and* raises the score | higher mean, lower sustained score — it starts cold, so its p10 is set by the first frames |

Not varied, and why: `--density` (the fcc lattice's reduced density) and `--dt`
were left at upstream's 0.8442 and 0.005 — `--dt` in particular trades directly
against the same neighbour-list budget the skin controls, so raising it would
have reintroduced the defect this study just removed. `--box` is payload sizing.

### Why these parameters

**The neighbour-list settings are a correctness condition at this temperature,
not tuning.** Upstream's `neighbor 0.3 bin` with
`neigh_modify every 20 delay 0 check no` is sized for the example's T=3.0, where
an atom covers about 0.22 sigma between rebuilds. At T=6.0 it covers about 0.33
sigma, crosses the 0.3 sigma skin, and pairs are silently missed. The cost shows
up in `fix nve`, where total energy is supposed to be conserved: the drift is
roughly an order of magnitude worse at T=6.0 on upstream's settings than with a
skin that actually covers the displacement.

Widening it *inflated* nothing — the repaired run scores higher, lifts block
coverage to the full array, and raises the minimum block evolution by more than
an order of magnitude. The coarse list was leaving some blocks artificially
static as well as leaking energy.

**Raising the temperature is what buys the evolution**, and the run is flat
across the whole 1,000 steps, because an LJ liquid at fixed temperature is
statistically stationary.

**On the temperature ramp.** `in.melt_ramp` at `T1=12.0` has a marginally higher
*mean* — a tie within run-to-run spread — but a lower sustained score, because
it starts from a cold crystal and its 10th percentile is set by the first
frames. On the criteria this study ranks by, the hot melt wins.

**That is not the whole story, and `in.melt_ramp` is still the right deck for a
different question.** A stationary hot liquid gives every frame the same *kind*
of data: high evolution, unchanging character. The ramp deliberately moves the
state point from crystal to hot liquid so the block statistics change
monotonically, which is what exercises a per-chunk *selector* rather than a
per-chunk *codec*. Use `--deck in.melt_ramp --var T0=0.05 --var T1=12.0
--var NSTEPS=<steps> --skin 0.8 --every 5` when that is what is wanted; the
neighbour settings are needed there too, and more so, since it ends hotter.

Rankings, per-decile curves and the energy-drift figures are in `RESULTS.md`.

### References

- [`lammps/lammps`](https://github.com/lammps/lammps) (pinned at `9e42b6f`),
  `examples/melt/in.melt` — the upstream LJ melt: `lattice fcc 0.8442`,
  `velocity all create 3.0 87287 loop geom`, `pair_style lj/cut 2.5`,
  `neighbor 0.3 bin`, `neigh_modify every 20 delay 0 check no`, `fix nve`.
- [`velocity`](https://docs.lammps.org/velocity.html) — "The *create* style
  generates an ensemble of velocities using a random number generator with the
  specified seed at the specified temperature."
- [`neighbor`](https://docs.lammps.org/neighbor.html) — "All atom pairs within a
  neighbor cutoff distance equal to their force cutoff plus the *skin* distance
  are stored in the list"; "the larger the skin distance, the less often
  neighbor lists need to be rebuilt, but more pairs must be checked".
- [`fix langevin`](https://docs.lammps.org/fix_langevin.html) — used by
  `in.melt_ramp`; "Tstart can be a variable", which is what lets the target
  temperature follow the global timestep, and it "does NOT perform time
  integration", hence `fix nve` alongside it.

## The policies

| config | what chooses the codec |
|---|---|
| `dynamic` | NeuroPress inference, **default balanced cost** — `w_ct·t_c + w_dt·t_d + w_io·bytes/(ratio·bw)`, all weights 1, bw 5 GB/s. One forward pass per chunk, nothing measured back. **The headline configuration.** |
| `dynamic-ratio` | same, with the two latency weights zeroed → a ratio-only objective |
| `learn` | `dynamic` plus online SGD from each chunk's measured outcome |
| `explore` | ratio-only ranking, then the top-31 alternatives are *actually compressed* and the measured winner adopted (threshold 0 = every chunk) |
| `best` | best mode: exhaustive, ratio-only ranking |
| `static-zstd`, `-s4`, `-s8` | fixed nvcomp-zstd with a 0/4/8-byte byte shuffle — **controls with no model at all** |

## Notes

- `CLIO_WITH_RUNTIME=1` is load-bearing: the runtime comes up inside the
  benchmark process before LAMMPS is opened.
- Each run picks a **free TCP port** automatically. Every Clio runtime binds
  one, and a fixed port collides with any other Clio process on the machine —
  a silent failure in which the client dies before doing anything. Pin it with
  `PORT=9413` if you need to.
- The data reaches the compressor as a **host** pointer. LAMMPS runs on the GPU
  under Kokkos, but its C library interface exposes only the host mirror, so
  the atoms cross PCIe on the Kokkos sync and nvcomp copies them back to the
  device to compress. Keeping the raw bytes resident needs device views —
  see `neuropress_lammps_gpu_direct`.
- **GPU runs are not bit-reproducible.** Repeats of a configuration differ
  slightly in the bytes they compress; use `--cpu` when you need runs
  comparable chunk-for-chunk.
- Serial only (`BUILD_MPI=OFF`, as upstream's own LAMMPS integration).
- `results/` is generated output and is not tracked.
