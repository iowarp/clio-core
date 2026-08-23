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

## Why two phases

Nyx is an AMReX application with no library interface to embed, so the LAMMPS
approach — link it in, read its arrays between steps — is not available.
Upstream NeuroPress's own Nyx benchmark does not embed it either: it patches
Nyx to dump raw fields and sweeps those files offline. This mirrors that.

```
  phase 1: ./gen_fields.sh     Nyx (GPU) ──► plt00000/fab0000_comp00_density.f32 …
  phase 2: ./run_sweep.sh      files ──► Clio compressor ──► NeuroPress ──► CTE tier
```

Splitting them buys something the LAMMPS benchmark cannot have: **every policy
replays the identical bytes**. A Kokkos LAMMPS trajectory is not
bit-reproducible, so its policies see slightly different data; here the
comparison is exact.

## The patch

`patches/nyx-raw-field-dump.patch` adds one self-contained function to
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

## Building Nyx

```bash
git clone --depth 1 --recursive https://github.com/AMReX-Astro/Nyx.git ~/src/Nyx
cd ~/src/Nyx
git apply <clio>/paper-benchmark/nyx/patches/nyx-raw-field-dump.patch
cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release \
      -DNyx_MPI=NO -DNyx_OMP=NO -DNyx_HYDRO=YES -DNyx_HEATCOOL=NO \
      -DNyx_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=Ampere \
      -DAMReX_PRECISION=SINGLE -DAMReX_PARTICLES_PRECISION=SINGLE
cmake --build build-clio --target nyx_HydroTests -j
```

`AMReX_PRECISION=SINGLE` is upstream's recommendation for compression
benchmarks and matters here for a specific reason — see the shuffle result
below. `Nyx_MPI=NO` keeps a single-rank benchmark from needing a parallel HDF5
build; nothing in this path writes HDF5 anyway.

The Clio side is `bin/neuropress_field_replay`
(`context-transfer-engine/compressor/example/neuropress_field_replay/`), which
has no simulation dependency at all:

```bash
cmake --build <clio>/build --target neuropress_field_replay
```

## Running

```bash
./gen_fields.sh                              # Nyx -> ./fields (~1 GB, a few min)
./run_sweep.sh                               # every policy over those files
./run_config.sh dynamic                      # one policy
./read.sh --run dynamic                      # cold read-back, separate process
../collect.py results/                       # re-aggregate
```

Every run verifies itself: each blob is read back through the decompressor and
its FNV-1a-64 digest compared with the digest of the bytes staged. `read.sh`
repeats that from a *different* process with `CLIO_RESTART=1`, sharing only the
store directory and `blobs.csv` with the writer -- the field files are never
opened, so the compressed tier is the only copy of the data in existence.

`gen_fields.sh` takes `--ncell --steps --plot-int --out --bin`; `run_sweep.sh`
takes `--fields --chunk --max-files --repeats --configs --results`. Defaults
are 128³, 200 steps, dumping every 10 → 21 frames × 6 components × 8 MiB ≈
1,008 MiB, chunked at 4 MiB into 252 chunks — deliberately the same chunk
count as the LAMMPS benchmark.

Fields dumped are the hydro state: `density`, `xmom`, `ymom`, `zmom`,
`rho_E`, `rho_e`.

## Results

Sedov blast, 128³, 21 frames, 1,008 MiB float32, 252 chunks, A100, lossless.
All 252 blobs verified bit-exact in every configuration.

| config | ratio | stored | density | rho_E | rho_e | xmom | ymom | zmom | Σ ms | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| **`static-zstd-s4`** | **156.1×** | 6.5 MiB | 206× | 168× | 170× | 146× | 125× | 143× | 739 | 31.8 s |
| `best` | 135.6× | 7.4 MiB | 175× | 168× | 170× | 120× | 104× | 114× | 854 | 123.9 s |
| `static-zstd-s8` | 135.1× | 7.5 MiB | 178× | 144× | 144× | 129× | 109× | 125× | 818 | 30.8 s |
| `explore` | 131.0× | 7.7 MiB | 173× | 154× | 153× | 117× | 103× | 114× | 738 | 105.8 s |
| `dynamic-ratio` | 79.5× | 12.7 MiB | 79× | 65× | 64× | 95× | 89× | 98× | 726 | 50.8 s |
| `static-zstd` | 79.1× | 12.7 MiB | 67× | 59× | 59× | 109× | 104× | 116× | 766 | 16.8 s |
| `dynamic` | 11.9× | 84.5 MiB | 11× | 6.5× | 7.6× | 20× | 21× | 30× | 287 | 77.2 s |
| `learn` | 8.7× | 115.3 MiB | 6.4× | 5.3× | 4.6× | 25× | 25× | 24× | 190 | 106.8 s |

### The shuffle stride is the headline, and it flips with the element width

Nyx can be built at either precision, so this is a **controlled** result:
identical problem, identical code, identical timesteps — only `AMReX_PRECISION`
changes. (`gen_fields.sh` with a double build and `NYX_DUMP_NATIVE=1` emits
`.f64`; replay with `--f64`.)

| fixed nvcomp-zstd | Nyx float32 (1,008 MiB) | Nyx float64 (2,016 MiB) |
|---|---|---|
| no shuffle | 79.1× | 94.3× |
| **4-byte** shuffle | **156.1×** | 104.4× |
| **8-byte** shuffle | 135.1× | **110.8×** |
| best stride | **4** | **8** |

The preference flips with the width, on the same physics. The cross-workload
comparison points the same way — LAMMPS float64 prefers 8 (1.198× against
1.159×) — but that one confounds data and width; this one does not.

The stride has to match the element width, and getting it wrong costs more
than any codec choice. NeuroPress encodes byte-shuffle as a **single bit**
meaning `GPUCOMPRESS_PREPROC_SHUFFLE_4` — a 4-byte stride. That is exactly
right for float32 Nyx, and it is why upstream's benchmarks never surfaced the
limitation: their recommended Nyx build is single precision. On float64 LAMMPS
the same bit is the wrong width, and no action in the searched space can ask
for 8.

Upstream is aware of the gap without having closed it. Its own Nyx bridge
(`examples/nyx_amrex_bridge.hpp:255`) asks for 8 on double data —
`shuffle_size = (h5type == H5T_NATIVE_DOUBLE) ? 8 : 4` — but the filter accepts
only 0 and 4 (`src/hdf5/H5Zgpucompress.c:258`), prints
`"only 0 and 4 are supported), ignoring shuffle"`, and falls back to **no
shuffle at all** rather than to 4.

### The default cost model is badly wrong on this data

`dynamic` — NeuroPress inference under the default balanced cost — reaches
**11.9× where 156× was available**, a 13× miss, and `learn` is worse still at
8.7×. The balanced objective charges compression time, so it picks the fast
codecs (bitcomp ×126, ans ×66) on data whose whole value is in the slow
entropy coders. Zeroing the two latency weights (`dynamic-ratio`) recovers
79.5× immediately.

This is the same failure mode as in the LAMMPS benchmark but far larger:
there, inference reached 1.074× against 1.198× achievable (an 11% miss); here
it forfeits an order of magnitude. Highly compressible data punishes a
selector that optimises for speed.

### Exploration does not reach the fixed codec

`explore` and `best` measure every alternative and still land at 131–136×,
below a fixed zstd with the right stride (156×) that costs a third of the wall
clock. Both facts have the same cause as in the LAMMPS run: the shuffle the
search can request is chosen per action from a space that pairs it with the
codec, and the search is bounded by what the action space can express.

## Notes

- Lossless throughout: `error_bound_ = 0`, which masks the 16 quantize actions.
  Verified rather than assumed: 2,016 blobs in the float32 sweep, 2,016 in the
  float64 control, and 756 recovered cold from the tier by `read.sh` -- zero
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
