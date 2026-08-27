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

## The patches

Two, and both are needed — see "Building Nyx" for why the second one's absence
is easy to miss.

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

## Building Nyx

```bash
git clone --depth 1 --recursive https://github.com/AMReX-Astro/Nyx.git ~/src/Nyx
cd ~/src/Nyx
git apply <clio>/paper-benchmark/nyx/patches/nyx-raw-field-dump.patch
git apply <clio>/paper-benchmark/nyx/patches/nyx-comoving-a-single-precision-eps.patch
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

`gen_fields.sh` takes `--ncell --steps --plot-int --out --bin --keep-plt`;
`run_sweep.sh` takes `--fields --chunk --max-files --repeats --configs --results`. Defaults
are 128³, 200 steps, dumping every 10 → 21 frames × 6 components × 8 MiB ≈
1,008 MiB, chunked at 4 MiB into 252 chunks — deliberately the same chunk
count as the LAMMPS benchmark.

Fields dumped are the hydro state: `density`, `xmom`, `ymom`, `zmom`,
`rho_E`, `rho_e`.

## Looking at the data

```bash
./gen_fields.sh --ncell 64 --steps 400 --plot-int 16 --exp-energy 10 \
                --keep-plt --out /tmp/nyx-quick             # ~2 s, 157 MiB
../viz_fields.py --fields /tmp/nyx-quick --plt /tmp/nyx-quick-plotfiles \
                --out /tmp/nyx-viz                          # ~4 s
```

`../viz_fields.py` reads the flat `.f32` dumps with nothing but numpy, and that is
the point: those files, not Nyx's plotfiles, are what the compressor is handed,
so what it draws is what the sweep compresses. Per field it writes a montage of
mid-plane slices across the run and the same slices as a GIF; once per run it
writes `evolution.png`, which puts shock radius, fraction of the domain off
ambient, and a zlib stand-in for the compression ratio on one time axis.

**Two things it is worth pointing at a dataset for.**

*Does this run actually evolve?* Point it at the default `./fields` and the
answer is visibly no — shock radius reaches 0.18 of the box and **1.8% of cells
are off ambient at the last dump**. That is the deck default `exp_energy=1`
described under "`--exp-energy` is the knob that makes the data evolve", and it
is much faster to see than to infer from a ratio table. The 64³ recipe above
reaches 0.48 and 40%.

*Is the shuffle result real?* The ratio panel reproduces it in three lines of
python: on the quick run, density goes 1005× → 3.8× unshuffled against
1005× → 5.1× with a 4-byte shuffle, the same ordering the nvcomp-zstd sweep
finds at 132.6× vs 162.9×. zlib is not nvcomp-zstd and the magnitudes do not
transfer; the shape and the ordering do.

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

### Seeing what a lossy bound costs

`../viz_lossy.py` puts the original, the decompressed copy and `|error|` on one
plate. It needs the decompressed bytes, which the replay driver will write on a
cold read:

```bash
# Tag explicitly. Deriving one by stripping '0' and '.' collapses 0.001, 0.01
# and 0.1 to the same string, and the second run silently overwrites the first.
for spec in 0.001:eb001 0.01:eb01 0.1:eb10; do
  EB=${spec%%:*}; T=${spec##*:}
  CLIO_NEUROPRESS_STAGE_H2D=1 ./run_config.sh dynamic --fields /tmp/nyx-quick \
      --chunk 1048576 --eb $EB --check-bound --results /tmp/nyx-lossy --tag $T
  $BUILD/bin/neuropress_field_replay --readback /tmp/nyx-lossy/$T/blobs.csv \
      --dump-decompressed /tmp/nyx-decomp/$T --tag nyx_$T \
      --dir /tmp/nyx-quick --ext .f32 --chunk 1048576 --check-bound
done
../viz_lossy.py --orig /tmp/nyx-quick --out /tmp/nyx-viz/lossy \
    --plt /tmp/nyx-quick-plotfiles --compare 0.001:/tmp/nyx-decomp/eb001 \
    --compare 0.01:/tmp/nyx-decomp/eb01 --compare 0.1:/tmp/nyx-decomp/eb10
../viz_actions.py --out /tmp/nyx-viz/actions --sel 0.001:/tmp/nyx-lossy/eb001 \
    --sel 0.01:/tmp/nyx-lossy/eb01 --sel 0.1:/tmp/nyx-lossy/eb10
```

`--compare EB:DIR` is repeatable. The plate is original on top, one decompressed
row per bound, then one `|error|` row per bound. Every field row shares one
color scale so the rows are comparable; each error row is scaled to ITS OWN
bound, so across bounds compare the decompressed rows, not the error rows.

Three things that will otherwise cost an hour:

- **`CLIO_NEUROPRESS_STAGE_H2D=1` is required for the offline sweep, and
  nothing here sets it.** `CLIO_NEUROPRESS_REQUIRE_DEVICE` defaults *on*
  (`compressor_runtime.cc`), NeuroPress preprocessing is CUDA-only, and a
  file-replay driver hands the compressor host memory by construction. Without
  the stage every chunk is refused and every blob fails `rc=11` with
  `0 B in -> 0 B on the tier`. `run_config_insitu.sh` sets residency
  deliberately; `run_config.sh` sets neither, so the offline path is broken
  out of the box on this build.
- **`--chunk` must equal one component** (`ncell^3 x 4`, so 1048576 at 64³),
  or a dumped `.bin` is a fragment spanning field boundaries and will not
  reshape.
- **The readback `--tag` must match the writer's**, which is `nyx_$NAME`, not
  the config name. A wrong tag finds no blobs and reports `rc=11` — the same
  symptom as the residency failure, from an unrelated cause.

Measured on the 64³ quick run, `dynamic`, over the same 156 MiB. Every chunk
landed inside its bound at both settings:

| | stored ratio | worst `max|err|` |
|---|---|---|
| lossless | 4.114× | 0 (bit-exact) |
| `--eb 0.001` | **6.470×** | 0.000949 |
| `--eb 0.01` | 6.198× | 0.00949 |
| `--eb 0.1` | 10.975× | 0.0950 |

**The ratio is not monotone in the bound.** `0.001` stores BETTER than `0.01`
(6.470× against 6.198×) on identical input. The cause is visible in
`viz_actions.py`: under the balanced cost model the two bounds pick different
codecs for the same chunk — on density, `0.01` sits on `zstd` for the first
four dumps and reaches a mean 471.7×, while `0.001` sits on `cascaded` at
26.2×, and the ordering reverses again once both settle on `ans`. Selection,
not quantization, is what moves the total. Do not assume a looser bound is a
smaller file.

The pictures say something the ratio does not. The bound is **absolute**, so
what it costs depends entirely on local magnitude:

| field | eb 0.001 | eb 0.01 | eb 0.1 | 0.1 as % of peak | outcome |
|---|---|---|---|---|---|
| `density` | 25/26 | 26/26 | 24/26 | 3.8% | flattened at 0.01, obliterated at 0.1 |
| `xmom` `ymom` `zmom` | 17–19/26 | 23–24/26 | 25/26 | 0.10% | shell intact at every bound |
| `rho_E` | **0/26** | **0/26** | **0/26** | 0.0057% | **bit-exact at all three** |
| `rho_e` | **0/26** | **0/26** | 9/26 | 0.0085% | **bit-exact below 0.1** |

(chunks quantized, out of 26)

On density the ambient is 1.0 and the evacuated centre reaches 0.002. At 0.001
the reconstruction is visually indistinguishable from the original. At 0.01 the
bound is a 1% perturbation outside the shock but already larger than the value
inside it, so the front is untouched while the interior gradient collapses to a
slab. At 0.1 the interior becomes a single flat level and the quantizer's
Cartesian grid shows through as a **diamond** — the clearest picture of the
effect in the whole set, and still invisible at the shock front. The three
bounds as a ladder are what makes that legible; any one of them alone is
ambiguous.

And the bound is a **ceiling, not an instruction**. NeuroPress decides per chunk
whether quantizing is worth it: `rho_E` declined at all three bounds, `rho_e`
declined twice and then accepted 0.1 on 9 of 26 chunks, and density actually
quantizes FEWER chunks at 0.1 (24) than at 0.01 (26). A "lossy run" is lossy
only where the selector took the offer.

### The action progression

`../viz_actions.py` draws what was selected per dump, and it draws the whole
action rather than just the codec: lane is the library, colour is the bound, a
filled marker means quantize was taken, a square means the 4-byte shuffle.

It lives one level up because it is workload-agnostic: it reads both blob-name
shapes, so the same script serves the LAMMPS benchmark.

The point is that **the selection moves as the physics does**. On density the
run walks cascaded/zstd/gdeflate → snappy → bitcomp → cascaded → ans, settling
only once the blast has filled enough of the box that the block stops changing
character. Measured switches over 26 dumps: density 7 (eb 0.001), 3 (0.01),
6 (0.1); `xmom` 5 / 10 / 4. `rho_e` never leaves `lz4` — its lane is a flat
line, and at eb=0.1 the marker simply fills in from dump 19 onward.

Preset is not drawn: it is 2 throughout these runs, and the script says so
rather than hiding it if that ever changes.

### Temporal redundancy is the number that explains all of it

`evolution.png` now carries a fourth panel: the share of cells **bit-identical
to the previous dump**.

| dump | 1 | 12 | 25 |
|---|---|---|---|
| `density` | 99.9% | 86.5% | 57.1% |
| `xmom` `ymom` `zmom` | 99.7% | 83.5% | 50.9% |
| `rho_E` `rho_e` | 99.9% | 86.3% | 56.7% |

It falls almost linearly across the run. That single column is the clearest
statement of why a fixed codec is the wrong answer here: the block genuinely
stops being the same kind of data, and every other curve on the page —
compressibility, the chosen action, the cost of a given bound — is downstream
of it. It is also the metric `gen_fields.sh` reasons about when it argues for
raising `--exp-energy`, now measured rather than asserted.

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

Even with `--keep-plt`, `../viz_fields.py` opens the plotfiles only to read
`Header` for the simulation time, so frames are labelled `t=0.0236` rather than
`dump 25`. Without `--plt` it falls back to dump indices and everything else
still works. (yt is not installed here and there is no `pip`, so the documented
route was not exercised on this machine.)

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

### `--exp-energy` is the knob that makes the data evolve

The Sedov shock radius goes as `R = 1.033 (E t^2 / rho)^(1/5)`, so at the deck's
`E=1` and `t=0.01` it reaches only **~0.16 of a unit box** — most of the domain
never changes and consecutive dumps are nearly identical.

Raising `E` fixes two things at once, which is not obvious: a wider shock also
means higher velocities, so the CFL timestep shrinks and the *same* `stop_time`
now takes **more** steps. Measured at 128³, `density` ratio range over the run:

| `--exp-energy` | steps | ratio range | spread |
|---|---|---|---|
| 1 (deck default) | ~300 | 12.7x .. 87.5x | 6.9x |
| **10** | ~675 | 4.6x .. 205.3x | **44.4x** |
| 100 | ~1575 | 1.8x .. 205.8x | 113.9x |

A validated 1 GB configuration:

```bash
./run_config_insitu.sh explore-balance   --ncell 128 --steps 2000 --int 32 --chunk 8388608   --exp-energy 10 --explore-k 3 --explore-thresh -1 --verify   --results /tmp/nyx1g --tag e10
```

21 frames, 671 steps, 126/126 blobs verified bit-exact, and **no frame
bit-identical to its predecessor on any field**. Compressibility falls
monotonically from ~86x to ~4.6x as the blast fills the domain, and the
selector moves cascaded -> bitcomp -> ans along the way. Both knobs are
recorded in `meta.json`, so a run stays self-describing.

`--steps 2000` there is deliberately generous: the run stops on `stop_time`, so
the step count is an outcome, not an input.

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

## Results

Sedov blast, 128³, 21 frames, 1,008 MiB float32, 252 chunks, A100, lossless.
All 252 blobs verified bit-exact in every configuration, in process and on a
cold read from a separate process.

Regenerated after `775dd9b4` fixed the dump: it had been writing the first
`validbox().numPts()` elements of the GROWN FArrayBox — a mixture of valid and
ghost cells — instead of the valid box. Every number in this section is from
the corrected dump (each file is exactly 128³ × 4 B).

| config | ratio | stored | density | rho_E | rho_e | xmom | ymom | zmom | Σ ms | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| **`static-zstd-s4`** | **162.9×** | 6.2 MiB | 222× | 192× | 194× | 145× | 123× | 144× | 720 | 33.8 s |
| `static-zstd-s8` | 141.2× | 7.1 MiB | 192× | 166× | 167× | 127× | 107× | 124× | 813 | 32.0 s |
| `best` | 134.1× | 7.5 MiB | 139× | 192× | 184× | 118× | 103× | 115× | 628 | 111.9 s |
| `static-zstd` | 132.6× | 7.6 MiB | 184× | 159× | 160× | 109× | 104× | 117× | 895 | 13.8 s |
| `explore` | 126.9× | 7.9 MiB | 147× | 139× | 164× | 116× | 102× | 114× | 678 | 94.9 s |
| `dynamic-ratio` | 75.3× | 13.4 MiB | 109× | 52× | 51× | 96× | 90× | 100× | 791 | 45.8 s |
| `dynamic` | 23.7× | 42.6 MiB | 29× | 36× | 18× | 19× | 20× | 31× | 245 | 54.9 s |
| `learn` | 20.8× | 48.4 MiB | 29× | 14× | 14× | 26× | 31× | 25× | 151 | 93.8 s |

The mis-sliced data changed more than the magnitudes: `static-zstd` (no
shuffle) moves from 79.1× to 132.6×, from below `dynamic-ratio` to above
`explore`, and `dynamic`/`learn` roughly double. The ORDER of the top result is
unchanged — a fixed zstd with the 4-byte stride still wins — but any statement
about *how far* the others trail had to be re-derived, which is what the rest
of this section does.

### The shuffle stride is the headline, and it flips with the element width

Nyx can be built at either precision, so this is a **controlled** result:
identical problem, identical code, identical timesteps — only `AMReX_PRECISION`
changes. (`gen_fields.sh` with a double build and `NYX_DUMP_NATIVE=1` emits
`.f64`; replay with `--f64`.)

| fixed nvcomp-zstd | Nyx float32 (1,008 MiB) | Nyx float64 (2,016 MiB) |
|---|---|---|
| no shuffle | 132.6× | 94.3× |
| **4-byte** shuffle | **162.9×** | 104.4× |
| **8-byte** shuffle | 141.2× | **110.8×** |
| best stride | **4** | **8** |

(The float64 column was re-measured from the corrected dump too and came back
unchanged to three digits — 94.321 / 104.406 / 110.804 — so only the float32
column moved.)

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
**23.7× where 162.9× was available**, a 6.9× miss, and `learn` is worse still
at 20.8×. The balanced objective charges compression time, so it picks the fast
codecs (bitcomp ×168, ans ×26, cascaded ×20) on data whose whole value is in
the slow entropy coders; `best` and `explore`, which do not, pick zstd for
158–183 of the 252 chunks. Zeroing the two latency weights (`dynamic-ratio`)
recovers 75.3× immediately — still less than a fixed zstd, because the
ratio-only objective ranks on a predicted ratio that is itself wrong.

This is the same failure mode as in the LAMMPS benchmark but far larger:
there, inference reached 1.074× against 1.198× achievable (an 11% miss); here
it forfeits most of an order of magnitude. Highly compressible data punishes a
selector that optimises for speed.

### Exploration does not reach the fixed codec

`explore` and `best` measure every alternative and still land at 127–134×,
below a fixed zstd with the right stride (162.9×) that costs a third of the
wall clock — and below a fixed zstd with NO shuffle at all (132.6×). Both facts have the same cause as in the LAMMPS run: the shuffle the
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
