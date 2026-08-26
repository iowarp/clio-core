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
   defaulting to upstream's 0 (off). See "Choosing the parameters" — with it
   off, four of the sixteen fields never change for the whole run.

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

## Choosing the parameters

### `VPIC_CLEAN_DIV_INT`: without it, a quarter of the payload never changes

Upstream's deck sets `clean_div_e_interval = clean_div_b_interval = 0`
("turn off cleaning (GY)"). The consequence for a *compression* benchmark is
that `div_e_err`, `div_b_err`, `rhob` and `rhof` are never recomputed — they
hold their initial values and are dumped unchanged every frame.

Measured at 126³ / 200 steps / 8 frames, comparing the `fnv1a64` digest of the
bytes the simulation handed over at consecutive dumps:

| | ratio | frames bit-identical to the previous one |
|---|---|---|
| `div_b_err`, `div_e_err` | 452.75x | **7 of 7** |
| `rhob`, `rhof` | 2.30x | **7 of 7** |
| the other twelve | 1.00–1.21x | 0 of 7 |

`selection.csv` shows why: those fields have **entropy 0.0000 and MAD 0.0** —
they are constant arrays. They are 25% of the payload, and because they
compress ~4.2x against the 1.06x the twelve evolving fields manage, they lift
the run's headline ratio from **1.061 to 1.304** — a 1.23x gain from data the
simulation never touched.

`VPIC_CLEAN_DIV_INT=5` makes them real diagnostics computed from the live
state: `div_b_err` goes to 1.03–1.13x with 0 of 7 frames identical, `div_e_err`
to 3.57x, `rhof` to 1.05–1.14x. (`rhob` stays constant — this deck accumulates
no bound charge.) The default stays 0 so existing numbers are unchanged.

### `--steps`: 200 is the noise phase, not the instability

The deck drives the Weibel instability from a **temperature anisotropy**:
`vthe = 0.25/sqrt(2)` perpendicular against `vthex = 0.05/sqrt(2)` in x, an
anisotropy of 25. With `Lx = 10 de`, `nx = 128` and `cfl_req = 0.99`,
`dt*w_pe ~ 0.044`, so 200 steps is only ~9 `w_pe^-1` — roughly 1.5 e-foldings
out of the noise floor.

At 1000 steps the instability is unmistakably growing, from the deck's own
`energies` diagnostic:

```
          step 123     step 1001    growth
by       2.558e-02     5.793e-01     22.6x
bz       2.556e-02     4.859e-01     19.0x
bx       4.914e-02     2.792e-01      5.7x
```

— and still accelerating at step 1000, so it has not saturated.

### But lossless compressibility does NOT track that, and cannot be made to

Over those same 1000 steps every field's digest changes at every dump, yet
every **lossless** ratio sits flat at ~1.14x. Entropy is **7.31 of 8 bits per
byte**: the physics lives in the high-order bits while the float32 mantissa is
effectively random and dominates the entropy, so a 22x change in field energy
moves the lossless ratio by about 1%. No deck parameter changes this.

**Lossy does track it.** At `--eb 1e-3` over the same run, the ratio falls
monotonically as the filaments fill the domain:

| field | step 125 | step 1000 | range |
|---|---|---|---|
| `cby` | 7.26x | 4.51x | 1.61x |
| `cbz` | 7.27x | 4.59x | 1.58x |
| `ex` | 6.01x | 4.59x | 1.31x |

So read VPIC's **lossy** cells as the ones carrying signal. Its lossless cells
are still worth running — they are the workload that stresses the *latency*
path, where the cost model's 1 ms clamps dominate — but their ratio column is a
flat line by construction, not a result.

### float32 in; int8 on the wire when lossy

Every blob is 1,149,984 B = 287,496 **float32** elements
(`Kokkos::View<float*[FIELD_VAR_COUNT]>`, `elem_bytes = sizeof(float)`).

* **Lossless** stays float32 bit-for-bit — quantize is applied to 0 of 128
  adopted rows, and every blob round-trips bit-exact.
* **Lossy** quantizes all 128. On this data the quantizer picks
  `prec=8`: `1149984 -> 287496 bytes`, a **4x narrowing before the codec
  runs**. Of VPIC's 6.96x lossy ratio, 4x is quantization and only ~1.74x is
  the codec. The tier holds int8 plus a 32-byte `QuantHeaderExtension`
  (`error_bound`, `scale`, `data_min`, `data_max`); the read side rebuilds
  float32 from it.

Verified with `--check-bound`: *"4,599,936 elements checked against
|original - decoded| <= 0.001; max observed error 0.001; 0 violations"*. The
`effective_eb` comes out slightly tighter than requested (0.00095 for 1e-3) so
the widest representable error still fits.

**This is not the same mechanism as Nyx's lossy runs**, where the quantizer
picks `prec=32` and narrows nothing — Nyx's entire lossy gain comes from the
codec. Precision is chosen per chunk from its dynamic range, so lossy ratios
are not directly comparable across workloads.

## Results

Weibel instability, 126³ cells, 8 frames, 1,024 MiB float32, 256 chunks, A100,
lossless. All 256 blobs verified bit-exact in every configuration.

| config | ratio | stored | compressed/raw | Σ compress ms | wall |
|---|---|---|---|---|---|
| **`static-zstd-s4`** | **1.491×** | 686.7 MiB | 256 / 0 | 4062 | 38.9 s |
| `explore` | 1.491× | 686.7 MiB | 256 / 0 | 5158 | 127.0 s |
| `best` | 1.490× | 687.1 MiB | 256 / 0 | 4310 | 127.0 s |
| `static-zstd-s8` | 1.419× | 721.8 MiB | 256 / 0 | 3124 | 37.8 s |
| `dynamic-ratio` | 1.403× | 730.0 MiB | 256 / 0 | 3796 | 57.9 s |
| `static-zstd` | 1.358× | 754.3 MiB | 256 / 0 | 2079 | 26.9 s |
| `learn` | 1.330× | 769.7 MiB | 193 / 63 | 1278 | 80.9 s |
| `dynamic` | 1.278× | 801.2 MiB | 200 / 56 | 1253 | 57.9 s |

### The stride result is about the element width, not compressibility

This is the workload that makes that claim safe:

| fixed nvcomp-zstd | LAMMPS float64 | Nyx float32 | **VPIC float32** |
|---|---|---|---|
| | *high entropy* | *structured* | *high entropy* |
| no shuffle | 1.124× | 79.1× | 1.358× |
| **4-byte** | 1.159× | **156.1×** | **1.491×** |
| **8-byte** | **1.198×** | 135.1× | 1.419× |
| best stride | **8** | **4** | **4** |

VPIC and Nyx are both float32 and both prefer 4 bytes despite differing in
compressibility by two orders of magnitude. LAMMPS is float64 and prefers 8. The
stride tracks the *element width*, and nothing else. NeuroPress's single shuffle
bit — always 4 bytes — is therefore right for both float32 workloads and wrong
for the float64 one.

### Field structure the aggregate hides

Two of the sixteen variables behave completely differently from the rest.
`div_e_err` and `div_b_err` reach **1,581×** under a fixed codec, and
`rhob`/`rhof` ~2.5×, while the remaining twelve — the actual E, B, J and TCA
fields — sit between 1.19× and 1.22×. So the headline 1.49× is two
nearly-constant diagnostic fields carrying twelve fields of noise, which is
worth knowing before quoting it.

Those four are not merely *near*-constant: with the deck's default
`VPIC_CLEAN_DIV_INT=0` they are **exactly** constant — entropy 0.0000, MAD 0.0,
and bit-identical between every consecutive dump, because divergence cleaning
never runs and so never writes them. They are 25% of the payload. See
"Choosing the parameters" for the measurement and for the knob that makes them
evolve.

### Inference is again the worst option

`dynamic` reaches 1.278× and leaves **56 of 256 chunks stored raw** — the codec
it picked expanded them. `learn` is barely better at 1.330× with 63 raw. On this
workload exploration and a fixed correctly-strided codec agree to within 0.001×,
which says the achievable ceiling is not in doubt; only the prediction is.

## Notes

- Lossless throughout; every blob digest-verified, and `read.sh` re-verifies
  from a separate process where the tier is the only copy.
- The dump includes VPIC's ghost layer, so voxel counts are (N+2)³ rather than
  N³. That is real simulation state, not padding, and it is what VPIC itself
  compresses in the upstream integration.
- Single rank (`mpirun` not used). VPIC supports MPI domain decomposition;
  nothing here exercises it.
- `fields/`, `results/` and the built `*.Linux` deck are generated and untracked.
