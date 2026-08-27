# WarpX workload — Clio + NeuroPress, in situ

A **laser-wakefield acceleration** run in [WarpX](https://github.com/BLAST-WarpX/warpx)
(AMReX PIC, CUDA, float32) compressed **while it runs**, by a **stock,
unpatched WarpX binary**.

This is the only in-situ workload here, and the only one where Clio touches a
simulation that knows nothing about it. LAMMPS is embedded as a library; Nyx and
VPIC are patched to dump fields that are replayed afterwards. WarpX writes
openPMD-HDF5 exactly as it always does, HDF5 dlopens Clio's VOL connector
because `HDF5_VOL_CONNECTOR` says so, and the VOL chunks and compresses on the
way past — with Clio's runtime hosted inside the WarpX process.

```console
$ ldd $(which warpx.3d...) | grep -c clio
0
```

Upstream NeuroPress reaches WarpX through a **636-line patch** adding a whole
`FlushFormatGPUCompress` diagnostic backend. Clio needs **zero lines** in WarpX.

## Looking at the data

Nothing needs dumping: WarpX writes openPMD-HDF5 as it always does and the VOL
compresses on the way past, so the native `.h5` files are in
`<run>/run/diags/diag1/` afterwards — the same bytes the compressor saw.

```bash
./run_config.sh dynamic --ncell "64 64 512" --steps 40 --interval 4 \
    --stage-h2d --results /tmp/wx2 --tag ll                  # ~15 s
./viz_openpmd.py --run /tmp/wx2/ll --out /tmp/wx-viz
```

`viz_openpmd.py` needs **no h5py** — datasets come out through the `h5dump` CLI
(`-b LE -o file`), which ships with HDF5 and is therefore already present
anywhere WarpX built against it. It draws x–z slices at mid-y with z upward:
the laser enters at the bottom and the wake oscillations behind it are resolved
by step 40.

**The grid cannot be shrunk to make the run quick.** At 64×64×128 a field is
512 KB against the 1 MiB chunk, so no chunk ever completes and ZERO field bytes
reach the tier — while the run exits 0, the native `.h5` is perfect and nothing
reports a problem. 64×64×512 is the smallest grid that stages anything. This is
the same failure the `CLIO_VOL_CHUNK_SIZE` note at the top of `run_config.sh`
describes, reached from the other direction.

## The state vector spans fifteen orders of magnitude, and the bound only reaches one field

Measured at 64×64×512, 40 steps, 11 diagnostics, 880 chunks, `dynamic`:

| | stored ratio | quantize chosen | quantize ran |
|---|---|---|---|
| lossless | 11.536× | 0 / 880 | 0 |
| `--eb 0.001` | 14.631× | 39 | 30 |
| `--eb 0.01` | 14.676× | 39 | 26 |
| `--eb 0.1` | 15.004× | 40 | 23 |

Monotone, unlike Nyx and VPIC — but for a reason that makes the number almost
meaningless. WarpX writes SI, and the fields do not share a scale:

| field | range (SI) | `eb=0.01` as a share of it |
|---|---|---|
| `j/z` | 2.7e+12 | 3.7e-13 % |
| `E/z` | 1.5e+11 | 6.9e-12 % |
| `rho` | 6.9e+06 | 1.5e-07 % |
| `B/x` | 1.6e+04 | 6.1e-05 % |
| **`B/y`** | **1.1e-01** | **9.4 %** |

So the bound is far below float32 resolution on almost everything and lands on
`B` alone. Of the 16.9 MB it saves at `--eb 0.01`, **8.6 MB is `By` and 7.2 MB
is `Bz`** — 93% — while the E fields, currents and charge density contribute
0.2 MB between them. A `By` chunk goes from ~730 KB to ~700 B, because the
entire field is smaller than the bound and quantizes to zero.

That is a 27% ratio gain from 3% of the chunks, and it is not an accuracy
statement about anything. **An absolute bound is a statement about units.** It
is the same failure VPIC shows from the other direction: there the diagnostics
are 1e-07 and the bound annihilates them; here the currents are 1e+12 and the
bound cannot reach them. Both runs report every chunk inside its bound, and
both reports are true.

**Each policy is its own WarpX run**, so these are not byte-identical
comparisons the way Nyx and VPIC are. Measured run-to-run spread on this
configuration is 0.6% (11.860× vs 11.792× lossless; 15.167× vs 15.197× at
eb=0.01), well below the 27% the bound moves — but it is why these are quoted
to three digits.

## Running

```bash
./run_sweep.sh                      # every policy, each a full WarpX run
./run_config.sh explore             # one policy
./read.sh --run static-zstd-s4      # read back through the VOL
../collect.py results/              # re-aggregate
```

Requires a stock WarpX with openPMD:

```bash
git clone --depth 1 https://github.com/BLAST-WarpX/warpx.git ~/src/warpx
cd ~/src/warpx
cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release -DWarpX_COMPUTE=CUDA \
      -DWarpX_DIMS=3 -DWarpX_MPI=OFF -DWarpX_OPENPMD=ON \
      -DAMReX_CUDA_ARCH=8.0 -DWarpX_PRECISION=SINGLE
cmake --build build-clio -j
```

Defaults: 64×64×512 cells, 40 steps, diagnostics every 10 → 5 openPMD dumps ×
10 field components × 8 MiB = 400 MiB in 400 chunks of 1 MiB.

## The chunk size is a correctness condition, not a tuning knob

`CLIO_VOL_CHUNK_SIZE` is **1 MiB** here, not the 4 MiB the other workloads use,
and that is not a preference.

openPMD emits each AMReX box as a separate partial write, so an 8 MiB field
arrives as 1 MiB pieces at **non-contiguous** offsets — `[0,1M)`, then
`[4M,5M)`, and so on. The VOL's append path assembles only *contiguous* runs and
drops its tail on a discontinuity, so with a 4 MiB chunk **no chunk ever
completes**:

| chunk size | field blobs staged |
|---|---|
| 4 MiB (the default) | **0** |
| 1 MiB | **400** |

At 4 MiB the run still succeeds, the native `.h5` is perfect, exit status is 0,
and nothing anywhere reports a problem — the tier is simply empty of field data.
The rule is that the chunk must be **≤ the writer's contiguous write
granularity**, and this is the first workload here where that binds.

## Results

Laser acceleration, 64×64×512, 5 dumps, 400 MiB float32, 400 chunks, A100,
lossless.

| config | ratio | stored | wall | verified |
|---|---|---|---|---|
| **`static-zstd-s4`** | **17.434×** | 22.9 MiB | 13.1 s | pass |
| `explore` | 17.044× | 23.5 MiB | 22.1 s | pass |
| `best` | 16.544× | 24.2 MiB | 42.1 s | pass |
| `static-zstd-s8` | 16.368× | 24.4 MiB | 12.1 s | pass |
| `static-zstd` | 13.214× | 30.3 MiB | 9.1 s | pass |
| `dynamic-ratio` | 12.069× | 33.1 MiB | 13.1 s | pass |
| `dynamic` | 11.785× | 33.9 MiB | 12.1 s | pass |
| `learn` | 7.107× | 56.3 MiB | 16.1 s | pass |

Every run is verified: `run_sweep.sh` passes `--verify`, so each configuration
reads three field datasets back through the VOL from a separate process and
requires both byte-equality with a native read and trace evidence that the tier
served them. Across the eight runs: **192 chunk inversions, 0 cache misses.**

Consistent with the other three workloads: **the 4-byte stride wins on float32**
(17.43× against 16.37× for 8-byte), inference under the balanced cost model
leaves most of the ratio on the table (11.79×), and online learning is the worst
option (8.76×).

Unlike the other workloads, exploration here **does** essentially reach the best
fixed codec — 17.04× against 17.43×, and 17.44× once the ratio cap is raised.

### What the ratio cap does to exploration

`RATIO_CAP` (default 100, upstream's `nn_gpu.cu` constant, overridable with
`CLIO_NEUROPRESS_RATIO_CAP` or this harness's `RATIO_CAP=`) turns out to gate
*whether exploration runs at all*, not just how candidates rank.

The exploration gate fires on `error_pct > threshold`, where

```
error_pct = |actual_cost − predicted_cost| / actual_cost
cost      = w_io · bytes / (min(ratio, RATIO_CAP) · bandwidth)      [ratio-only]
```

When a chunk's predicted *and* actual ratio both exceed the cap, both clamp to
100, the two costs are numerically identical, `error_pct` is exactly **0**, and
the gate cannot fire at **any** threshold — including 0. Measured:

| | chunks explored | ratio |
|---|---|---|
| cap 100 (default) | 78 / 400 | 17.04× |
| cap 10⁶ | 400 / 400 | 17.44× |

and every one of the 322 unexplored chunks at the default cap was one whose
ratio exceeded 100, while not a single chunk below the cap was skipped. So the
mechanism is certain; the *cost* of it here is only ~3% of ratio, because the 78
chunks the gate did select were the ones that mattered.

## Verification

`read.sh` reads field datasets back through the VOL from a **separate process**
(`CLIO_RESTART=1`, replaying the metadata log) and checks two things: the bytes
match a native read, and the path trace shows the tier actually answered.

```console
$ ./read.sh --run static-zstd-s4
== cold read of diags/diag1/openpmd_000010.h5 (writer process is long gone)
   /data/10/fields/B/x: 8388608 B, identical to native, 8 chunk(s) inverted by a codec
   /data/10/fields/B/y: 8388608 B, identical to native, 8 chunk(s) inverted by a codec
   /data/10/fields/B/z: 8388608 B, identical to native, 8 chunk(s) inverted by a codec

PASS: every dataset came back byte-identical AND was served from the compressed tier
      (24 chunk inversions, 0 cache miss(es))
```

**Both halves are needed, and this workload is the one that proves it.** The
native `.h5` is authoritative and holds the same data uncompressed, so a read
that misses the tier entirely still returns correct bytes. Measured on a fresh
store, reading the identical file by the wrong key:

| | codec inversions | cache misses | bytes correct |
|---|---|---|---|
| tier served the read | 24 | 0 | yes |
| tier never touched | **0** | 1 | **yes** |

Correct bytes with zero tier use — indistinguishable from success on data alone.
`read.sh` fails the run when the trace shows no inversion, so that case cannot
pass silently.

### The file name is the key

The reason the second row happens at all: **the VOL keys its tag on the file
name as given**. WarpX runs in its output directory and writes
`diags/diag1/openpmd_*.h5` *relative*; a reader that opens the same file by
absolute path keys differently and finds nothing the writer stored —
`populated=0`, `chunk_0_size=0`, `MISS (native + stage)` — then silently
re-stages it, so a *second* absolute read appears to work. `read.sh` therefore
cds into the run directory and opens the same relative path the writer used.

This is the same defect class the HDF5 VOL's own tests record for datasets
opened by absolute path versus relative to a group. It is a property of Clio's
VOL, not of WarpX.

## Notes

- Each policy is its own WarpX run, so unlike the Nyx and VPIC sweeps the
  policies are not guaranteed byte-identical inputs. WarpX on one GPU is
  deterministic for fixed inputs, so in practice they match.
- The per-chunk record comes from the VOL path trace via `trace_to_csv.py`,
  since the application is stock WarpX and there is no Clio driver to record
  one. That converter must read the `neuropress FINAL` line for the decision and
  the exploration log for an overridden chunk's size — reading `codec ran`
  instead reports the *primary* candidate and understated exploration by 5×
  (12.13× against 16.86×) while line counts matched 1:1.
- Particle datasets are intercepted too, but the reported numbers are
  `--fields-only`; particle arrays are small here and would dilute the field
  measurement.
- `results/` is generated output and is not tracked.
