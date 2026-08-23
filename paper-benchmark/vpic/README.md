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
`div_e_err` and `div_b_err` reach **1,581×** under a fixed codec: they are
divergence-cleaning residuals, near-zero almost everywhere. `rhob`/`rhof` reach
~2.5×. The remaining twelve — the actual E, B, J and TCA fields — sit between
1.19× and 1.22×. So the headline 1.49× is two nearly-constant diagnostic fields
carrying twelve fields of noise, which is worth knowing before quoting it.

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
