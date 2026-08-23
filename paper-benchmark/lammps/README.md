# LAMMPS workload — Clio + NeuroPress dynamic compression

A molecular-dynamics workload for the paper's compression benchmarks. LAMMPS
runs a Lennard-Jones melt as a **library inside the benchmark process**, Clio's
runtime is hosted in that same process, and **NeuroPress chooses a codec per
chunk** from the simulation's own statistics. No file is written and no data
leaves the process to be compressed.

The workload is held constant across every run; the only variable is **how the
codec is chosen**. That is what makes the numbers below a comparison of
selection policies rather than of workloads.

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
| `--temp X` | `TEMP` | 3.0 | initial temperature |
| `--cutoff X` | `CUTOFF` | 2.5 | `lj/cut` pair cutoff, σ |
| `--skin X` | `SKIN` | 0.3 | neighbor-list skin |
| `--every N` | `EVERY` | 20 | `neigh_modify every` |
| `--seed N` | `SEED` | 87287 | velocity RNG seed |
| `--dt X` | `DT` | 0.005 | timestep |
| `--var K=V` | any | — | any other deck variable, repeatable |

```bash
# a colder, denser state point, whole sweep
./run_sweep.sh --density 1.2 --temp 0.4 --dt 0.002

# one policy, longer trajectory, finer sampling, different seed
./run_config.sh dynamic --steps 1000 --gap 100 --seed 12345
```

Each parameter is a `variable NAME index <default>` in `in.melt`, and LAMMPS
gives a command-line `-var` precedence over that default — so the deck still
runs standalone, and an option you do not pass leaves the deck's value alone
rather than restating it in two places.

The physics of every run is recorded in its `meta.json`, and `collect.py`
treats it as part of the run's identity: two runs of the same policy at
different density or temperature are different experiments and are reported
as separate rows, never averaged together.

`run_sweep.sh` writes one directory per run under `results/`, then
`collect.py` produces `summary.csv` and `summary.md`.

Each run directory holds `blobs.csv` (every chunk: size, codec, ratio, stored
bytes, compress time, digest), `selection.csv` (what NeuroPress chose, its
predicted ratio, and what it actually got), `explore.csv` (exploration only:
every candidate measured), `log.lammps`, and `meta.json`.

Every run self-verifies: each blob is read back through the decompressor and
compared against the digest of the bytes staged. `--no-verify` skips it.

## The policies

| config | what chooses the codec |
|---|---|
| `dynamic` | NeuroPress inference, **default balanced cost** — `w_ct·t_c + w_dt·t_d + w_io·bytes/(ratio·bw)`, all weights 1, bw 5 GB/s. One forward pass per chunk, nothing measured back. **The headline configuration.** |
| `dynamic-ratio` | same, with the two latency weights zeroed → a ratio-only objective |
| `learn` | `dynamic` plus online SGD from each chunk's measured outcome |
| `explore` | ratio-only ranking, then the top-31 alternatives are *actually compressed* and the measured winner adopted (threshold 0 = every chunk) |
| `best` | best mode: exhaustive, ratio-only ranking |
| `static-zstd`, `-s4`, `-s8` | fixed nvcomp-zstd with a 0/4/8-byte byte shuffle — **controls with no model at all** |

## Results

LJ melt, 2,048,000 atoms, 984.4 MiB float64, 252 chunks, A100, CUDA 12,
lossless throughout. All 252 blobs verified bit-exact in every run.

| config | ratio | stored | compressed/raw | position | velocity | force | Σ compress ms | wall |
|---|---|---|---|---|---|---|---|---|
| `dynamic` | 1.074× | 916.7 MiB | 168 / 84 | 1.159× | 1.029× | 1.042× | 600 | 40.2 s |
| `dynamic-ratio` | 1.079× | 912.3 MiB | 108 / 144 | 1.101× | 1.079× | 1.058× | 2775 | 35.2 s |
| `learn` | 1.058× | 930.6 MiB | 207 / 45 | 1.080× | 1.024× | 1.071× | 1989 | 49.2 s |
| `explore` | 1.187× | 829.5 MiB | 252 / 0 | 1.325× | 1.087× | 1.172× | 3451 | 92.3 s |
| `best` | 1.187× | 829.5 MiB | 252 / 0 | 1.325× | 1.087× | 1.172× | 2668 | 89.3 s |
| `static-zstd` | 1.124× | 875.8 MiB | 252 / 0 | 1.222× | 1.038× | 1.127× | 2469 | 22.2 s |
| `static-zstd-s4` | 1.159× | 849.4 MiB | 252 / 0 | 1.264× | 1.087× | 1.139× | 3290 | 31.2 s |
| **`static-zstd-s8`** | **1.198×** | 821.9 MiB | 252 / 0 | 1.345× | 1.114× | 1.157× | 4025 | 33.2 s |

Four things in that table are worth stating plainly.

**Melted LJ float64 is close to incompressible.** Nothing reaches 1.2×. The
workload demonstrates the path and the selection machinery, not a compression
headline.

**Inference leaves most of the ratio on the table.** `dynamic` stores 84 of 252
chunks raw — the codec it picked could not shrink them. Exploration, which
measures instead of predicting, reaches 1.187× with *no* raw chunks. That gap
is prediction error: over the 3,780 candidates exploration measured, the
model's top-ranked candidate was the true winner on only 21% of chunks, and its
predicted ratio saturated at the 100× `RATIO_CAP` on 11% of candidates.

**Online learning made it worse, not better.** `learn` lands at 1.058×, below
plain inference, and scatters across eight codecs. SGD from measured outcomes
degrades selection on this data rather than improving it.

**A fixed codec with the right stride beats the entire search.**
`static-zstd-s8` reaches 1.198× in 33 s, above exhaustive exploration (1.187×
in 92 s). The reason is the data type: LAMMPS is **float64**, but NeuroPress
encodes byte-shuffle as a single bit meaning `GPUCOMPRESS_PREPROC_SHUFFLE_4` —
a 4-byte stride, correct for the float32 the model was trained on, which splits
every double across two groups. An 8-byte stride is expressible through Clio's
static path but **no action in the searched space can request it**. That
constraint is deliberate — it keeps runs comparable with upstream — and this
measures its price: about 6% of ratio, here more than the whole selection
machinery recovers.

That last ranking is **scale-dependent**, so do not read it as a general
result. At the paper's default size (2M atoms, 4 MiB chunks) `static-zstd-s8`
edges out exploration; at `--box 20` (32,000 atoms) the same sweep puts
exploration ahead, 1.274× against 1.214×. Chunk content changes with system
size, and measuring beats predicting by more when the chunks are less uniform.
What holds at both sizes is the gap between inference and measurement.

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
