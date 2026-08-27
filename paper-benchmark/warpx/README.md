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

This file is the operational document: how to build and run the workload, how
to get its data out, and the parameters that make that data evolve.
**Measurements — compression ratios, error-bound behaviour, the evolution
study's rankings — are in [`RESULTS.md`](RESULTS.md).**

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

### Reading the fields back: `read_fields.sh`

`read.sh` proves the tier answered and the bytes are right. It does not write
them anywhere, so there is nothing to look at. `read_fields.sh` does the
equivalent of `--dump-decompressed` on the other three workloads — which have a
Clio driver to put the flag on, and this one does not:

```bash
./read_fields.sh --results /tmp/wx2 --runs "eb001 eb01 eb10" --out /tmp/wx-cmp
```

It reads each run's own `.h5` **twice** — once through the native HDF5 VOL for
the originals, once through Clio's for the decompressed copies — and writes both
as raw float32. Within a run, deliberately: comparing across runs would measure
WarpX's own 0.6% run-to-run spread instead of the codec.

**The native file is authoritative and uncompressed, so a VOL read that misses
the tier returns correct bytes and looks exactly like perfect lossless
compression.** Every read therefore records its `inverting codec` count into
`inversions.csv`, and a pair whose read reported zero must be discarded, not
published as bit-exact. That is the one way this comparison can lie; every
read in the measurements recorded in `RESULTS.md` inverted at least one codec.

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

## Default Evolving Benchmark Configuration

Selected by a 1,000-timestep hyperparameter study (`../evolution.py`, eight
configurations, 101 dumps each, 8,000 block samples per configuration).

Parameters:

- `laser1.e_max = 32.e12` V/m — `a0 = 8` at 0.8 um (`--e-max`; the deck ships
  `16.e12`, `a0 = 4`)
- `electrons.density = 2.e23` m^-3 — upstream's value, kept (`--density`)
- `warpx.do_moving_window = 1` — upstream's, and **not** optional here
- `amr.n_cell = 64 64 512`, `max_step = 1000` — upstream's own production
  recommendation, and 64x64x512 is already the smallest grid that stages any
  chunk at all

Evolution sampling interval: 10 timesteps

### Parameters tested

Three, plus one profile shape. `amr.n_cell` and `max_step` were held at
upstream's own production recommendation and are not part of the search.

| | `laser1.e_max` | `electrons.density` | `warpx.do_moving_window` | `electrons.profile` |
|---|---|---|---|---|
| **controls** | "Peak amplitude of the laser field, in the focal plane", V/m | "the plasma density in m^-3" | "Whether to use a moving window for the simulation" | `constant` against `parse_density_function`, i.e. a z-dependent plasma |
| **why it should matter** | sets `a0` and so how nonlinear the wake is; a deeper blowout bubble keeps evolving instead of settling | sets `w_p` and the plasma wavelength, so more structure per box and a faster-oscillating wake | with the window co-moving at c the wake is quasi-static in grid index space; off, the structure sweeps through fixed blocks | if the plasma the pulse meets keeps changing, the wake it drives should keep changing |
| **official reference** | `parameters.rst`, `<laser_name>.e_max`; `E_max = a0 * 4.0e12 V/m` at 0.8 um | `parameters.rst`, `<species_name>.profile` / `density` | `parameters.rst`, `warpx.do_moving_window`, `warpx.moving_window_v` | `parameters.rst`, `parse_density_function` and the `(x>0)` interval idiom |
| **values tested** | 16.e12 (deck, `a0` 4), 32.e12 (`a0` 8) | 2.e23 (deck), 8.e23, 32.e23 | 1 (deck), 0 | constant (deck); ramp 2e23 -> 1e24 over 141 um, raw and `z > 0`-guarded, alone and combined with `a0` 8 |
| **outcome** | **the only knob that worked** | **inert and non-monotonic** — 4x is worse than the deck, 16x marginally better | **disqualified** — a resonant PEC cavity, not wakefield physics | **no help**, from both directions |

`amr.n_cell = 64 64 512` is not free to vary here for two independent reasons:
it is upstream's own production recommendation, and it is already the smallest
grid on which any chunk completes at all (at 64x64x128 a field is 512 KB against
the 1 MiB chunk and **zero** field bytes reach the tier). `warpx.cfl` is already
1.0 in the deck, its stable maximum, so there is no headroom of the kind Nyx
turned out to have.

### Why these parameters

**This is the workload where the search mostly failed, and that is the result.**
Every valid configuration has the same shape: a large transient over the first
third of the run while the laser enters and the wake forms, then a flat plateau.
The plateau is not an artefact — WarpX's own `FieldEnergy` reduced diagnostic
shows total field energy constant to five digits from step 400 on — it is what a
**quasi-static wake in a co-moving window** looks like. The window follows the
pulse at exactly c, so a fixed block sees nearly the same structure every dump,
and the residual evolution is the wake pattern slipping backward relative to the
frame because its phase velocity is below c.

Against that, the knobs available do very little:

- **Plasma density is inert and not even monotonic** — 4x is *worse* than the
  deck, 16x marginally better, all three within a few percent.
- **A longitudinal density ramp does not help**, which was the hypothesis this
  study was extended to test: if the plasma the pulse meets keeps changing, the
  wake should keep changing. It does not. Re-running it guarded to `z > 0` — the
  first attempt's profile went negative below z = -35 um — reproduced the same
  numbers to four decimals, and adding the ramp to the *winning* run lowers it
  too, so the hypothesis is falsified from both directions rather than merely
  unsupported.
- **Laser amplitude is the only knob that moved the plateau.** `a0` 4 -> 8
  drives a deeper blowout, so the bubble evolves rather than settling.

**`warpx.do_moving_window = 0` scores far higher than the winner and is
disqualified on physics the metric cannot see.** With the window off, the pulse —
injected at z = 9 um, 3 um below a `pec` boundary — reflects instead of
propagating, and the box becomes a resonant cavity. Field energy is *exactly*
conserved (PEC walls absorb nothing) while E and B exchange periodically: a
standing wave, not laser-wakefield acceleration. Its evolution *rises* through
the run, which is the tell. The run exits 0 and holds no NaN, so
`evolution_rank.py` cannot reject it automatically; the reason is recorded in
its `evolution.json` under `disqualified` and honoured by the ranking, with the
raw diagnostic in `../evolution-study/warpx/FE_nomovingwindow.txt`.

**About a tenth of `E`/`B` blocks and an eighth of `j`/`rho` blocks never change
in any configuration** — the vacuum region ahead of the pulse, where the fields
are identically zero. That ceiling is why no configuration reaches full
coverage.

Rankings, per-decile curves and the field-energy figures are in `RESULTS.md`.

### References

- [`BLAST-WarpX/warpx`](https://github.com/BLAST-WarpX/warpx) (pinned at
  `bd8c12f`), `Docs/source/usage/parameters.rst`:
  - **`<laser_name>.e_max`** — "Peak amplitude of the laser field, in the focal
    plane", with `E_max = a0 * 2 pi m_e c^2 / (e lambda) = a0 * 4.0e12 V/m` at
    `lambda = 0.8 um`.
  - **`<species_name>.profile`** / `density` — "the plasma density in m^-3";
    `parse_density_function` for the ramp, whose `(z>0)` factor is upstream's
    own idiom ("The factor `(x>0)` equals 1 where x>0 and 0 where x<=0").
  - **`warpx.do_moving_window`** / **`warpx.moving_window_v`** — "Whether to use
    a moving window for the simulation"; "The speed of moving window, in units
    of the speed of light".
- `Examples/Physics_applications/laser_acceleration/inputs_base_3d` — the deck,
  whose own comments give this study's grid and step count: "for production, run
  for longer time, e.g. max_step = 1000" and "for production, run with finer
  mesh, e.g. amr.n_cell = 64 64 512".
- [LWFA example](https://warpx.readthedocs.io/en/latest/usage/examples/lwfa/README.html)
  — the physics being modelled, "laser diffraction, beamloading, and bubble
  dynamics in the blowout regime".

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
