# VPIC workload — Clio + NeuroPress dynamic compression

A **Weibel instability** computed by [VPIC-Kokkos](https://github.com/lanl/vpic-kokkos)
(CUDA, float32), dumped to flat field files, then replayed through Clio with
NeuroPress choosing a codec per chunk.

This is the third workload, and it exists to break a confound. LAMMPS is
**float64 and high-entropy**; Nyx is **float32 and highly structured**. Every
claim resting on the element width was therefore also a claim about how
compressible the data is. VPIC is **float32 and high-entropy** — particle-in-cell
field arrays are close to noise, with ~913,000 distinct values in 941,000 cells —
which separates the two.

This file is the operational document: how to build and run the workload, how
to get its data out, and the parameters that make that data evolve.
**Measurements — compression ratios, error-bound behaviour, the evolution
study's rankings — are in [`RESULTS.md`](RESULTS.md).**

## Why two phases

VPIC has no library interface to embed, and upstream NeuroPress's own VPIC deck
is tightly coupled to it: `benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx`
includes `gpucompress.h`, `gpucompress_vpic.h` and the VOL headers, and runs the
entire selection benchmark — policies, REINFORCE, PSNR — inside the deck. None
of that is reusable by another storage system.

So, as with Nyx: patch the simulation to dump raw fields, sweep the files
offline. `weibel_clio.cxx` links nothing from Clio or NeuroPress.

```
  phase 1: ./build_deck.sh && ./gen_fields.sh    VPIC ──► plt00000/fab0000_comp04_cbx.f32 …
  phase 2: ./run_sweep.sh                        files ──► Clio ──► NeuroPress ──► CTE tier
  phase 3: ./read.sh --run <name>                cold read from the tier alone
```

Splitting them means every policy replays identical bytes, so the comparison is
exact.

## The deck

`weibel_clio.cxx` is `vpic-kokkos/sample/Weibel/Weibel.cxx` with two changes and
no physics touched:

1. **The grid is 3D and env-configurable.** Upstream's deck is 64 × 1 × 1 — 4 KB
   of field data, fine as a physics regression, useless as a compression
   workload. `VPIC_NX/NY/NZ`, `VPIC_NPPC`, `VPIC_STEPS`, `VPIC_DUMP_INT`.
2. **A raw field dump** in `begin_diagnostics` when `VPIC_DUMP_FIELDS=1`.
3. **Divergence cleaning is env-configurable** via `VPIC_CLEAN_DIV_INT`,
   now defaulting to 10. At upstream's 0, four of the sixteen fields never
   change for the whole run; see "Default Evolving Benchmark Configuration".

Two details in that dump are load-bearing:

**De-interleaving.** VPIC stores fields as `Kokkos::View<float*[FIELD_VAR_COUNT]>`
— device-resident and *interleaved*, so variable `m` of voxel `v` sits at
`v*16 + m`. Writing that straight out would produce one file whose every 16th
float belongs to the same physical quantity, which is not what a compressor sees
from any other writer and would make the numbers incomparable. The dump writes
one contiguous `n_voxels` array per variable instead — the same shape Nyx
produces, so one replay driver reads both.

**Skipping step 0.** At step 0 the field array is still identically zero: the
particles are loaded but no field solve has run. A frame dumped there is 16 files
of pure zeros, which compresses ~infinitely and would dominate any ratio averaged
with it. Nyx has no such problem (its step-0 state is the initial condition), so
only this deck needs the guard.

## Building

VPIC first:

```bash
git clone --depth 1 --recursive https://github.com/lanl/vpic-kokkos.git ~/src/vpic-kokkos
cd ~/src/vpic-kokkos
cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_KOKKOS_CUDA=ON -DENABLE_KOKKOS_OPENMP=OFF \
      -DBUILD_INTERNAL_KOKKOS=ON -DKokkos_ARCH_AMPERE80=ON \
      -DKokkos_ENABLE_CUDA_LAMBDA=ON \
      -DCMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper
cmake --build build-clio -j
```

Then the deck and the Clio driver:

```bash
./build_deck.sh
cmake --build <clio>/build --target neuropress_field_replay
```

`build_deck.sh` uses VPIC's own generated `bin/vpic` deck compiler with exactly
one addition: **`-lcuda`**. Kokkos' CUDA backend calls the CUDA *driver* API
(`cuStreamGetCtx`, `cuCtxPushCurrent_v2`, `cuCtxGetDevice`) from
`Kokkos_Cuda_Instance.cpp`, but the stock link line carries only the runtime, so
any deck fails at link with three undefined references. Upstream's own VPIC build
script passes `-lcuda` for the same reason. VPIC's tree is left untouched.

## Running

There are **two routes**, and they are not interchangeable.

**In situ** — the deck hands the compressor its device-resident field array
directly, so the GPU paths are exercised. This is the route `run_benchmark.sh`
uses:

```bash
./build_deck.sh --insitu           # REQUIRED; see the warning below
./run_config_insitu.sh explore-balance --ncell 126 --steps 1000 --int 10 \
    --chunk 8388608 --verify --results /tmp/vpic --tag base
```

It sets `CLIO_NEUROPRESS_REQUIRE_DEVICE=1` unconditionally, so a host-resident
chunk is refused rather than silently computed on the CPU. Configs are
`explore-balance`, `explore-ratio`, `explore-speed`; `--check-bound` replaces
`--verify` on a lossy run, and `--clean-div N` overrides the divergence-cleaning
interval.

**Replay** — dump the fields, sweep the files offline, so every policy replays
byte-identical input:

```bash
./gen_fields.sh                    # VPIC -> ./fields (~1 GB)
./run_sweep.sh                     # every policy over those files
./read.sh --run static-zstd-s4     # cold read-back, separate process
../collect.py results/             # re-aggregate
```

Defaults: 126³ cells, nppc 8, 200 steps, dumping every 25 → 8 frames × 16 vars ×
8 MiB = 1,024 MiB in 256 chunks of 4 MiB.

`--ncell 126` is deliberately not round. VPIC's field array carries a ghost
layer, so the dumped extent is (N+2)³; 126 gives exactly 128³ = 2,097,152 voxels
= 8 MiB per variable, chunking into exactly two whole 4 MiB chunks — the same
shape the Nyx dumps have, so chunk counts compare directly.

**Rebuilding the deck needs `--insitu`.** `./build_deck.sh` alone produces a
binary that writes *files* instead of handing Clio device pointers;
`run_config_insitu.sh` detects the missing link and refuses rather than
silently running the wrong path.

## Looking at the data

VPIC dumps the same shape Nyx does, so the shared viewers read both:

```bash
./gen_fields.sh --ncell 32 --nppc 8 --steps 1000 --dump-int 40 \
    --clean-div 5 --out /tmp/vpic-quick                    # ~4 s
../viz_fields.py --fields /tmp/vpic-quick --out /tmp/vpic-viz --field cby --field ex
```

`--clean-div` is new and matters for anything you intend to *look* at: at 0 the
four divergence fields hold their initial values, so a quarter of every montage
is a still image. `--steps 1000` is what reaches the instability rather than the
noise phase — measured on `cby`, amplitude grows from ±0.045 to ±0.169 and
0 of 24 consecutive dumps are bit-identical. (`rhob` stays constant at any
setting; this deck accumulates no bound charge.)

`viz_fields.py` turns its two blast-wave panels off automatically here — shock
radius and "% off ambient" are defined against a quiet background, and a Weibel
run has none. It substitutes an amplitude panel.

## Default Evolving Benchmark Configuration

Selected by a 1,000-timestep hyperparameter study (`../evolution.py`, four
configurations, 100 dumps each, 12,672 block samples per configuration).

Parameters:

- `VPIC_CLEAN_DIV_INT = 10`   (`clean_div_e_interval` = `clean_div_b_interval`)
- `VPIC_VTHE  = 0.1767766953` (upstream's `0.25/sqrt(2)`, unchanged)
- `VPIC_VTHEX = 0.0353553391` (upstream's `0.05/sqrt(2)`, unchanged)
- `VPIC_NPPC = 8`, `--ncell 126`, 1,000 steps

Evolution sampling interval: 10 timesteps

### Parameters tested

Two, both taken from upstream's own deck: the one thing that decides whether a
field is *computed*, and the one thing that is the instability's free energy.

| | `clean_div_e_interval` / `clean_div_b_interval` | the anisotropy, `vthe` / `vthex` |
|---|---|---|
| **controls** | how often the divergence-cleaning solve runs, and so whether `div_e_err`, `div_b_err`, `rhob`, `rhof` are ever recomputed | the bi-Maxwellian's thermal velocities across x and along it; `(vthe/vthex)^2` is the anisotropy driving Weibel |
| **why it should matter** | a variable that is never recomputed is dumped unchanged every frame, and these are four of sixteen — 25% of the payload | the linear growth rate rises with both the perpendicular thermal velocity and the anisotropy, so filaments should emerge from the noise sooner |
| **official reference** | `sample/Weibel/Weibel.cxx`, which sets both to 0 ("turn off cleaning (GY)") | `sample/Weibel/Weibel.cxx` — `vthe = 0.25/sqrt(2.0)`, `vthex = 0.05/sqrt(2.0)`, an anisotropy of 25 |
| **values tested** | 0 (upstream), 10 | 25 (upstream); 100 by raising `vthe` to `0.5/sqrt(2)`; 100 by lowering `vthex` to `0.025/sqrt(2)` |
| **outcome** | **decisive** — takes the run from 12 of 16 variables evolving to 15 of 16 | **inert**; raising `vthe` actively hurts, saturating the instability sooner |

Not varied, and why: `VPIC_NPPC` changes the shot-noise floor rather than the
physics, and since that floor is already what the metric is measuring here (see
below), lowering it would raise the score by adding noise — the opposite of what
this study is for. `--ncell 126` is fixed by the chunk shape: the ghost layer
makes the dumped extent `(N+2)^3`, and 126 is what gives exactly 128^3 voxels.

### Why these parameters

**Divergence cleaning is the whole result, and the anisotropy is nearly inert.**
Turning cleaning on takes the run from twelve of its sixteen variables evolving
to fifteen. The four it affects — `div_e_err`, `div_b_err`, `rhob`, `rhof` — are
not merely slow at upstream's `0`: they are never recomputed at all, so they are
dumped bit-identical every frame, and they are a quarter of the payload.

Three of the four come back. **`rhob` cannot be revived by any interval**,
because bound charge accumulates from dielectric materials and this deck is
vacuum, so it is structurally zero for the whole run. The twelve already
evolving are unchanged to four decimals, so cleaning makes the diagnostics real
without perturbing the physics being measured.

**The anisotropy is inert because the measurement is saturated by shot noise.**
PIC field arrays sit near noise throughout — ~913,000 distinct values in 941,000
cells — and that noise decorrelates every step regardless of what the
instability is doing, so the Weibel growth is a small signal on top of a large
floor. Raising `vthe` actively hurt, a hotter plasma saturating the instability
sooner, so upstream's values are kept.

This is consequently the most evenly evolving of the four workloads: sustaining
evolution is not the difficulty here, and never was. The difficulty was that a
quarter of the payload was not being computed.

Per-field and per-decile measurements are in `RESULTS.md`.

### References

- Upstream deck: [`lanl/vpic-kokkos`](https://github.com/lanl/vpic-kokkos)
  `sample/Weibel/Weibel.cxx` (pinned at `f01d295`) — `vthe = 0.25/sqrt(2.0)`,
  `vthex = 0.05/sqrt(2.0)`, `nppc`, `cfl_req = 0.99`, `wpedt_max = 0.36`.
- `clean_div_e_interval` / `clean_div_b_interval`: upstream sets both to 0
  ("turn off cleaning (GY)"); the measured consequence is in `RESULTS.md`.
- `VPIC_VTHE` / `VPIC_VTHEX` are this deck's own additions
  (`weibel_clio.cxx`), defaulting to upstream's values so an unset environment
  reproduces upstream exactly.

## Notes

- Lossless throughout; every blob digest-verified, and `read.sh` re-verifies
  from a separate process where the tier is the only copy.
- The dump includes VPIC's ghost layer, so voxel counts are (N+2)³ rather than
  N³. That is real simulation state, not padding, and it is what VPIC itself
  compresses in the upstream integration.
- Single rank (`mpirun` not used). VPIC supports MPI domain decomposition;
  nothing here exercises it.
- `fields/`, `results/` and the built `*.Linux` deck are generated and untracked.
