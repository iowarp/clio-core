# Clio-NeuroPress benchmark

`run_benchmark.sh` runs four simulation workloads through Clio's compressor with
NeuroPress in **exploration mode**, across the two cost models, lossless and
lossy, and a ladder of assumed storage bandwidths.

```bash
./run_benchmark.sh --smoke                                 # 16 cells, every path
./run_benchmark.sh --profile quick --workloads lammps      # validate the pipeline
./run_benchmark.sh --profile mid                           # ~5-10 min/run
./run_benchmark.sh --profile full                          # ~30 GB/run
./run_benchmark.sh --check-bound                           # self-verify lossy runs
./run_benchmark.sh --dry-run                               # print the matrix
./collect.py results/benchmark                             # re-aggregate
./audit_run.py results/benchmark/<cell>                    # reconcile one run
```

### Flags

| flag | default | what it does |
|---|---|---|
| `--smoke` | off | `--profile quick --repeats 1 --bw-policy one`. Each workload contributes the four cells that cover both modes and both cost models: 16 runs. Run it before starting a campaign. |
| `--profile` | `full` | `quick` ~1 GB, 1-2 min/run; `mid` ~8-10 GB, 5-10 min/run; `full` ~30 GB, 17-25 min/run. All but `quick` run >=1000 timesteps. See section 2 for why the brief's three targets are not simultaneously reachable. |
| `--workloads` | all four | Space-separated subset, e.g. `"nyx vpic"`. |
| `--bw-policy` | `reduced` | `reduced` runs the full ladder on the balance model and a two-point control set on ratio (bandwidth cannot reorder candidates there -- see section 3). `full` runs the ladder on both; `one` pins 5 GB/s. |
| `--repeats` | `3` | Mandatory for readable results, not caution -- see "Exploration-mode selection is nondeterministic run to run". |
| `--check-bound` | off | On lossy runs, verify `|original - decoded| <= eb` elementwise instead of a digest. Only Nyx and VPIC can: they hold the submitted bytes in situ. |
| `--keep-native` | off | Keep WarpX's native openPMD output. It is deleted per run by default; at the larger profiles it is tens of GB of duplicate data, and it is most of what the disk preflight asks for. |
| `--results` | `results/benchmark` | Where cell directories go. |
| `--dry-run` | off | Print the command each cell would run, and stop. |

A run directory keeps its raw per-chunk measurements and nothing is aggregated
away: `blobs.csv` (one row per chunk), `selection.csv` (NeuroPress's own
per-chunk record), `explore.csv` (one row per measured candidate, with the
adopted flag). `collect.py` turns a set of them into `summary.csv` /
`summary.md`; `audit_run.py` reconciles a single run's CSVs against the bytes
that actually reached the tier.

### How each workload verifies itself

Not a preference -- it follows from how the data reaches Clio.

| workload | policy | lossless | lossy |
|---|---|---|---|
| Nyx, VPIC | in situ | `--verify`: the adapter digests each chunk as it stages it, then reads every blob back through the decompressor | `--check-bound`: compares against the bytes the simulation submitted, since no source file exists in situ |
| WarpX | read-back | `--verify`: a separate process reads the datasets back through the VOL and requires both that the bytes match a native read and that the trace shows the tier served them | not checked |
| LAMMPS | inline | the driver always verifies and takes no flag | still runs its digest check, which lossy data must fail -- the log reads `FAILED: N of M`, and `collect.py` records `n/a` rather than propagating it |

`run_benchmark.sh` never *requests* a bit-exact check on a lossy run: decoded
bytes are not the input bytes by design, so a digest check reports `FAILED` on
correct behaviour. LAMMPS's driver verifies unconditionally and so prints one
anyway -- expect `FAILED: 28 of 30` on a lossy LAMMPS cell and read the
`verified` column in `summary.csv`, which is `n/a` there, not the log line.
Whether it fails at all depends on whether the cost model picked any quantize
action, so the same cell can read `FAILED` under `balance` and `VERIFIED`
under `ratio`; neither is a defect. Quality for a lossy run is the PSNR column in
`selection.csv`, plus `--check-bound` where the workload supports it.

Each cell of

```
workload x {lossless, lossy} x {balance, ratio} x bandwidth x repeats
```

lands in `results/benchmark/<workload>_<mode>_<model>_<bw>__r<n>/`, e.g.
`warpx_lossless_balance_nvme_1GBs__r1`, `nyx_lossy_ratio_dram__r3`.

---

## 1. How each workload's data evolves

The point of the brief was that the compressor should see **naturally evolving
simulation data**, not one snapshot repeated. What actually changes between
timesteps differs a lot across the four, and it is what makes their results
incomparable in the interesting way.

### WarpX — laser wakefield acceleration (IN SITU)

A stock, unpatched WarpX writes openPMD-HDF5; HDF5 loads Clio's VOL connector,
which chunks and compresses on the way past. Clio's runtime is hosted inside
the WarpX process. **Nothing in WarpX is modified.**

| | |
|---|---|
| arrays through the compressor | `E` (Ex,Ey,Ez), `B` (Bx,By,Bz), current density `j` (jx,jy,jz), `rho` — float32 |
| what changes | a laser pulse propagates down the box and drives a plasma wake behind it; each dump catches the pulse and its wake further along, so the *location* of the structure moves while the background stays quiet |
| written | every `diag1.intervals` steps, one openPMD dump per diagnostic |
| size controls | `amr.n_cell` (grid), `max_step` (timesteps), `diag1.intervals` (dump frequency) |

**Chunk size is a correctness condition here, not a tuning knob.** openPMD emits
each AMReX box as a separate partial write, so a field arrives as 1 MiB pieces at
non-contiguous offsets. The VOL assembles only contiguous runs, so at a 4 MiB
chunk *no chunk ever completes and zero field bytes reach the tier* — while the
run succeeds and the native `.h5` is perfect. Measured: 0 field blobs staged at
4 MiB, 400 at 1 MiB. All profiles pin `WARPX_CHUNK=1048576`.

WarpX also writes its **native openPMD output** alongside the compressed tier,
so a 30 GB run writes ~30 GB of duplicate files. `run_benchmark.sh` deletes them
after each run unless `--keep-native`.

### VPIC — Weibel instability

VPIC has no library interface, so the deck (`vpic/weibel_clio.cxx`) dumps raw
fields and the sweep replays them.

| | |
|---|---|
| arrays | 16 field variables per voxel: `ex ey ez cbx cby cbz tcax tcay tcaz jfx jfy jfz rhof rhob div_e_err div_b_err` — float32 |
| what changes | counter-streaming beams are unstable; magnetic filaments **grow out of a smooth equilibrium**, so structure increases over the run. PIC field arrays stay close to noise throughout (~913,000 distinct values in 941,000 cells) |
| written | every `VPIC_DUMP_INT` steps |
| size controls | `--ncell` (dumped extent is (N+2)³ — VPIC's ghost layer is real state), `--nppc`, `--steps`, `--dump-int` |

Step 0 is skipped deliberately: before the first field solve the array is
identically zero, and a frame of pure zeros would dominate any averaged ratio.

Two of the sixteen variables behave nothing like the rest — `div_e_err` and
`div_b_err` are divergence-cleaning residuals, near zero almost everywhere, and
reach ~1,581× under a fixed codec while the twelve real E/B/J/TCA fields sit
between 1.19× and 1.22×. Any aggregate ratio for VPIC is those two carrying
twelve of noise.

### Nyx — Sedov blast wave

Patched to dump raw fields; the sweep replays them.

| | |
|---|---|
| arrays | the hydro state: `density`, `xmom`, `ymom`, `zmom`, `rho_E`, `rho_e` — float32 |
| what changes | a point explosion expands into uniform background. Early frames are nearly constant (hugely compressible); the shock front sweeps outward as R(t) = 1.033·(E·t²/ρ)^(1/5), so more of the domain becomes active with time |
| written | every `amr.plot_int` steps |
| size controls | `--ncell`, `--steps`, `--plot-int`, `prob.exp_energy` |

**This is the workload where "evolving data" has to be checked rather than
assumed.** At the stock 200 steps the shock reaches only ~25% of the box, and
**62–91% of blocks are bit-identical between consecutive dumps** for the whole
run — the median block never changes at all. The `mid` and `full` profiles run
1000 steps for this reason. `nyx/gen_fields.sh` documents the measured
step-count/coverage table.

### LAMMPS — Lennard-Jones melt (linked in as a library)

LAMMPS runs **inside the benchmark process**, Clio's runtime is hosted in the
same process, and the driver reads `Atom::x/v/f` between `run` segments. No file
is written and no data leaves the process to be compressed.

| | |
|---|---|
| arrays | `position`, `velocity`, `force` — 3 components each, **float64** (`double **x, **v, **f`) |
| what changes | an fcc lattice melts into a liquid: positions decorrelate from the initial lattice over the run, while velocities and forces are near-Maxwellian noise from early on |
| written | a frame every `--gap` steps |
| size controls | `--box` (atoms = 4·N³), `--steps`, `--gap` |

LAMMPS is the only float64 workload here, and the only one that is **not
bit-reproducible** — a Kokkos trajectory differs slightly between runs, so its
cells do not see byte-identical input the way the two replay workloads do.

---

## 2. What the knobs map to in the code

Nothing here is invented; every knob is an existing environment variable or
compose field.

| axis | how it is set | where it is read |
|---|---|---|
| exploration mode | `explore-balance` / `explore-ratio` config → `neuropress_exploration_enabled`, `neuropress_exploration_k`, `neuropress_exploration_threshold` in the compose file | `compressor_runtime.cc` |
| balance cost model | default weights `w_ct = w_dt = w_io = 1` | `RankingWeights`, `predictor.h` |
| ratio cost model | `CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1` | `models/neuropress_bridge.cc` |
| bandwidth | `CLIO_NEUROPRESS_COST_BW` (`--bw`) | `models/neuropress_bridge.cc` |
| lossless / lossy | `CLIO_NEUROPRESS_ERROR_BOUND` (`--eb`) → `Context::error_bound_` | the three drivers + the VOL |
| decompression timing | `CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=1` | `neuropress_explore.cc` |

The cost NeuroPress minimises is

```
cost = w_ct·compress_ms + w_dt·decompress_ms + bytes / (ratio · bandwidth)
```

**Bandwidth is in BYTES PER MILLISECOND** — the unit of
`RankingWeights::bandwidth_bytes_per_ms` — not bytes/s and not GB/s. So
1 GB/s = 1e6. NeuroPress's shipped default is 5e6 = 5 GB/s.

| name | value (B/ms) | meaning |
|---|---|---|
| `nvme_1GBs` | 1e6 | 1 GB/s |
| `nvme_2GBs` | 2e6 | 2 GB/s |
| `nvme_5GBs` | 5e6 | 5 GB/s — NeuroPress's default |
| `nvme_10GBs` | 1e7 | 10 GB/s — a current PCIe 4.0 x4 drive |
| `dram` | 2.26e8 | 226 GB/s — **measured on this machine**, STREAM triad, 128 threads, dual EPYC 7763 / DDR4-3200 (16 channels, 409.6 GB/s theoretical). Re-measure on other hardware. |

`--eb 1e-3` is the lossy setting: upstream NeuroPress's own benchmark value
(`VPIC_ERROR_BOUND:-0.001`) and the default Clio's `neuropress_explore_sweep.cc`
already used.

---

## 3. Findings that change how the results should be read

### Exploration-mode selection is nondeterministic run to run

Exploration compresses the top-K alternatives, **measures** each one's real
compress and decompress time on the device, and adopts on the resulting cost.
Those measurements vary, so when two candidates are close the winner flips.

Measured on the Nyx replay — the same files, byte for byte — at a **fixed**
5 GB/s, three runs gave

```
1078.96x    879.38x    1082.71x        (23% spread, nothing changed)
```

so **a single run per cell cannot be read on its own**. `--repeats` defaults to
3 and `collect.py` prints the run-to-run band per cell; a bandwidth effect is
real only where two cells' bands do not overlap.

### Bandwidth moves the balance model, and the noise grows with it

Nyx replay, `explore-balance`, three repeats per bandwidth:

| bandwidth | runs | band |
|---|---|---|
| 1 GB/s | 38.27, 37.89, 38.10 | **±1%** — clearly separated |
| 10 GB/s | 43.95, 40.49, 62.42 | wide |
| 226 GB/s | 52.79, 64.00, 47.13 | wide, overlaps 10 GB/s |

1 GB/s separates cleanly from the rest; 10 GB/s and DRAM do not separate from
each other at n=3. The mechanism is visible in the cost function: at low
bandwidth the `bytes/(ratio·bw)` term dominates and selection is stable, and as
bandwidth rises that term shrinks until the **measured** — and therefore noisy —
time terms decide. The high-bandwidth cells are exactly where a single run is
least trustworthy.

### Bandwidth is close to inert for the ratio model, and that is arithmetic

This one is **algebra first, measurement second.** Under the ratio-only weights
the cost is `bytes/(ratio·bw)` and nothing else. For a given chunk, `bytes` and
`bw` are the same for every candidate, so bandwidth is a positive scalar on the
sole term: it rescales every candidate's cost identically and cannot reorder
them. The argmin is bandwidth-free.

The measurement agrees where the run is stable enough to show it. On one Nyx
replay configuration (24 files, 4 MiB chunks) selections at 1e6, 5e6 and 1e7
came back byte-identical — same md5 over `blob,codec,stored` — while the same
ladder under `balance` moved the run ratio 37× → 80× → 120×. In noisier
configurations the ratio cells differ between bandwidths, but they differ by the
same amount when the bandwidth is held **fixed** and the run simply repeated, so
that is the nondeterminism above and not a bandwidth effect. Trust the algebra;
use the repeats to confirm nothing else is going on.

That is why `--bw-policy` defaults to `reduced`: the full ladder runs on
`balance`, where bandwidth demonstrably matters, and `ratio` keeps a two-point
control (5 GB/s and DRAM) so the invariance stays visible in the output rather
than being asserted. `--bw-policy full` runs the whole ladder on both.

### A single absolute error bound does not mean the same thing to every field

`--eb` is an **absolute** bound: `|original − decoded| ≤ eb`. Nyx's six
components span five orders of magnitude, measured over the dumps:

| field | range | field | range |
|---|---|---|---|
| `density` | 2.375 | `rho_E` | 2.10e5 |
| `xmom`/`ymom`/`zmom` | 143.8 | `rho_e` | 2.10e5 |

At `eb = 1e-3` **all six components quantize** on a full 126-file run, but the
same absolute bound buys wildly different quality, because the analytical PSNR
is `10·log₁₀(range² / (eb²/3))` and the range is per field:

| field | range | PSNR from eb=1e-3 | measured max |
|---|---|---|---|
| `density` | 2.375 | 72.3 dB | 72.7 |
| `xmom`/`ymom`/`zmom` | 143.8 | 107.9 dB | 108.5 |
| `rho_E`/`rho_e` | 2.10e5 | 171.2 dB → **capped at 120** | 120.0 |

So one `--eb` delivers ~72 dB on density and ≥120 dB on `rho_E` — a 48 dB
spread across components of the same run, with density taking essentially all
of the error. If you want comparable quality per field you need a per-field
bound, which this knob cannot express.

The **cost model also decides how often quantize is taken at all**. Measured on
the same dumps at 5 GB/s, chunks quantized per field (of 21):

| | density | xmom | ymom | zmom | rho_E | rho_e |
|---|---|---|---|---|---|---|
| `explore-ratio` | 9 | 11 | 10 | 10 | 15 | 21 |
| `explore-balance` | 21 | 21 | 21 | 21 | 21 | 21 |

Balance quantizes everything — quantization is cheap and it is charging for
time — while ratio takes it only where it actually pays in bytes. Read the
per-field columns, not just the aggregate.

### PSNR in `selection.csv` is analytical, not measured

`actual_psnr` is `10·log₁₀(range² / (eb²/3))`, **capped at 120 dB** — upstream's
formula (`gpucompress_compress.cpp`), derived from the bound and the data range
rather than from comparing decoded values. A column of `120.0` means "the bound
is loose relative to this field's range", not "120 dB was measured". `-1` is
upstream's "undefined" sentinel and is what a lossless chunk correctly reports.

### Verifying a lossy run: `--check-bound`

A lossy run cannot be checked with a digest -- the decoded bytes are not the
input bytes by construction -- so `run_config.sh` turns the bit-exact check off
and, without this flag, a lossy run is simply unverified. `--check-bound`
replaces it with the check that does apply: every chunk is decompressed,
re-read from its source file, and compared **element-wise** against the error
bound.

```
BOUND OK: 126 chunk(s) checked against eb=0.100, 0 exceeded, worst max|err|=0.095
BOUND FAILED: 126 chunk(s) checked against eb=0.001, 8 exceeded, worst max|err|=0.005
```

Each violation is named on stderr (`BOUND EXCEEDED <blob> max|err|=... > eb=...`),
the run exits non-zero, and `collect.py` reports `bound-ok` / `BOUND-FAIL` in
the `verified` column. A chunk whose source file cannot be re-read is counted
and reported, never silently passed.

**It covers `nyx` and `vpic` only.** The check re-reads each chunk from the
file it came from, and those are the two workloads whose payload is on disk.
LAMMPS generates its arrays in memory and WarpX's originals are its native
openPMD output, so neither has a source to compare against; those cells stay
unchecked rather than reporting a pass they did not earn.

**Expect `BOUND FAILED` on Nyx at the default `eb=1e-3`** -- see below. That is
a true result about the data, not a broken harness: the run completes, its
numbers are written, and only the bound verdict fails.

### The error bound is respected, except where float32 cannot represent it

Verified against the ORIGINAL `.f32` bytes rather than assumed: a lossy replay
run with `--dump-decompressed`, every decoded chunk compared element-wise with
the file it came from.

| | |
|---|---|
| chunks quantized | 123 of 126 |
| bound respected (`max\|orig-decoded\| <= eb`) | **115 of 123** |
| bound exceeded | **8 of 123**, all `rho_E`/`rho_e`, worst 4.578e-3 = **4.6x** |

The eight are the documented unachievable-bound regime, not a defect here:
`QuantizationResult::bound_achievable_` is false when the requested bound is
below what float32 can represent over the chunk's range, and the quantizer
falls back to the tightest safe bound. `rho_E`/`rho_e` span 2.1e5, where a
float32 ULP is already ~0.025 -- twenty-five times the 1e-3 asked for. Upstream
NeuroPress does the same thing and prints "Using maximum precision quantization
(error may exceed bound)". **If a run needs a guaranteed bound, check the range
of the field it is applied to first.**

### PSNR is conservative, and cannot see a bound violation

`actual_psnr` never overstated the delivered quality on any chunk measured --
it is 2.8 to 41 dB PESSIMISTIC against PSNR computed from the decoded bytes:

| field | reported | measured from decoded bytes |
|---|---|---|
| `density` | 42.8 - 72.7 | 48.5 - 84.7 |
| `xmom`/`ymom`/`zmom` | 93.4 - 108.5 | 91.7 - 126.2 |
| `rho_E`/`rho_e` | 110.8 - 120.0 | 128.4 - 210.0 |

Erring low is the safe direction, but note WHY: the value is derived from
`(range, eb)`, not from comparing decoded values, so it is blind to whether the
bound was actually met. All eight bound-violating chunks above reported a
perfectly healthy PSNR. **A clean PSNR column is not evidence the bound held.**

### Bit-exact verification is off for lossy runs

Every verify path here is an FNV-1a digest comparison, which lossy compression
must fail by design. `run_config.sh` disables it when `--eb > 0` and records
`verified: n/a` rather than reporting `FAILED` on correct behaviour.

---

## 3b. Device residency: the GPU path is enforced, not assumed

Every workload here is supposed to hand NeuroPress **device memory**, so that
quantization, byte shuffle and codec selection all run on the GPU. Measured
with the compressor's own `device=` trace (`IsDevicePointer` on each chunk at
`DynamicSchedule`), the original harness did **none** of that:

| workload | before | after |
|---|---|---|
| Nyx | 0 / 126 device | **120 / 120** (in situ) |
| VPIC | 0 / 128 device | **128 / 128** (in situ) |
| LAMMPS | 0 / 30 device | **30 / 30** (`--order device`) |
| WarpX | 0 / 400 device | **still 0 -- and now fails** |

Across the three fixed workloads: **1109 chunks, 0 host-resident, 0 host
quantizations, 0 `DEVICE->HOST` staging events.**

### One switch, because all three fallbacks read one condition

`CLIO_NEUROPRESS_REQUIRE_DEVICE=1` refuses a host-resident chunk at the
compressor instead of quietly computing on the CPU. That single check is
sufficient, and that is not a shortcut -- the three places the GPU path can
silently become a host one all read the residency of the same pointer:

1. **quantize** -- `want_quant && IsDevicePointer(input_ptr)` selects
   `QuantizeDevice`, else the host `Quantize<float>()`;
2. **byte shuffle** -- the same predicate selects `ByteShuffleDevice`, else the
   host `ByteShuffle()`;
3. **codec** -- `EstCompressionStats` excludes CPU candidates for a
   device-resident buffer, so a CPU library (and the full payload D2H that
   `StageInputIfNeeded` then performs) is only reachable for a host one.

A silent fallback is invisible in results -- the ratios are identical and only
the timings move -- which is exactly why it has to fail rather than degrade.

### What each workload needed

- **Nyx / VPIC**: switched from file replay to the **in-situ** adapters
  (`run_config_insitu.sh`). Replay cannot be device-resident by construction:
  the simulation copies to host, writes `.f32` files, and a separate process
  reads them back into host shm. In situ, the simulation hands the compressor
  `fab.dataPtr(comp)` / its Kokkos field views directly.
- **LAMMPS**: `--require-device`, which selects the driver's `--order device`
  (chunks gathered ON the GPU into CUDA-IPC backends) instead of the default
  `--order id` (gathered through `lammps_gather_atoms` into a host array).

### WarpX has no device pointer, so it stages H2D -- upstream's own route

A **stock WarpX hands HDF5 HOST memory** -- measured, not inferred: the VOL's
`clio_stage_append` reports the application's first partial write, and it is
`HOST`. openPMD emits one non-contiguous partial write per AMReX box, which the
VOL assembles into a host run buffer, so residency is gone before Clio is
called and no VOL change recovers it. (Upstream reaches the GPU here by
patching WarpX with a 636-line diagnostic backend that borrows
`fab.dataPtr(icomp)`; this workload's entire premise is zero lines in WarpX.)

Upstream has an answer for exactly this, because it has no CPU preprocessing
either. Its two entry points are

```c
/* Transfers data to GPU, compresses, and returns result to host. */
gpucompress_compress(const void* input, ...);
/* Compress data already in GPU memory. Avoids host-GPU transfer. */
gpucompress_compress_gpu(const void* d_input, ...);
```

Residency decides whether a **copy** happens, never where the computation
happens. `--stage-h2d` (`CLIO_NEUROPRESS_STAGE_H2D`) does the same: copy the
chunk up, run the same CUDA kernels. Measured on WarpX at 64x64x512, lossy
`eb=1e-3`:

```
chunks stored 160 | H2D stagings 160 | host quantize 0 | D2H 0 | ratio 36.9x
```

**It is staged in BOTH task bodies, and must be.** `DynamicSchedule` runs
selection (statistics, ranking, the exploration sweep) on one buffer; `Compress`
re-reads `task->blob_data_` to do the actual work. Staging only the first made
selection run on the GPU and then refused the transform it had selected --
silently storing nothing while the run still exited 0.

**The copy is not free, and the results say so.** One H2D per chunk is bandwidth
the in-situ workloads never spend, so WarpX **ratios** compare with the other
three and its **timings do not**. `meta.json` records `residency` as
`device-staged-h2d` rather than `device` to keep that visible, and the flag is
opt-in so the copy is chosen rather than inherited. `--stage-h2d` implies
`--require-device`, so a failed staging refuses instead of reverting.

## 4. Sizing: the three targets do not hold at once

Exploration mode sustains **~30 MiB/s** on this machine — 1008 MiB of Nyx in
32.8 s at a 16 MiB chunk. Neither more runtime threads (4/16/32 all give 30.7
MiB/s) nor a larger chunk moves it, and only ~3 s of that 33 is codec time; the
rest is per-chunk task dispatch. So ~30 GB costs **~17–25 min per run**, not
5–10.

| profile | payload | wall/run | timesteps | note |
|---|---|---|---|---|
| `quick` | ~1 GB | ~1–2 min | 200 | pipeline validation; matches the tracked `nyx/fields` dumps |
| `mid` | ~8–10 GB | ~5–10 min | 1000 | holds the **runtime** target |
| `full` | ~30 GB | ~17–25 min | 1000 | holds the **volume** target |

A full campaign is 4 workloads × 14 cells × 3 repeats = **168 runs**. At `mid`
that is roughly a day; scale `--repeats`, `--workloads` or `--bw-policy` to fit.

**Disk.** Nyx and VPIC replay their payload from disk and WarpX writes native
openPMD per run, so the campaign needs roughly one workload's payload free at
once. `run_benchmark.sh` refuses to start if there is not enough and says so
rather than filling the filesystem halfway through.

---

## 5. What comes out

Per run, kept and never aggregated away:

| file | one row per | carries |
|---|---|---|
| `blobs.csv` | chunk | bytes, codec, ratio, stored, `compress_ms`, `decompress_ms` |
| `selection.csv` | chunk | the three input statistics, predicted vs actual ratio/time, quantize/shuffle, PSNR |
| `explore.csv` | measured candidate | every alternative the sweep tried, its measured ratio/ct/dt, and which one was `adopted` |
| `meta.json` | run | the cell: mode, cost model, bandwidth, error bound, K, threshold, steps, wall time |

`collect.py` writes:

| file | contents |
|---|---|
| `summary.csv` | **every** run, all workloads, one row each, with a `workload` column — the only file that supports comparing one workload against another at the same bandwidth |
| `summary_<workload>.csv` | one per workload, that workload's runs only, and **only its own** `ratio_<field>` columns (written when a directory holds more than one workload) |
| `summary.md` | the same data as tables, one section per workload, plus best/worst and the repeat bands |

The per-workload split exists because the physical quantities are disjoint:
LAMMPS has `position`/`velocity`/`force`, Nyx has `density`/`xmom`/`rho_E`/…,
VPIC sixteen field variables, WarpX ten. A combined header is their **union**,
so in `summary.csv` a LAMMPS row carries a `0.0` for `ratio_density` and a Nyx
row a `0.0` for `ratio_force` — about 35 mostly-empty columns once all four
workloads are present, where `0.0` reads the same as "compressed to nothing"
rather than "no such field here". Use the per-workload file for a per-workload
question. Throughput columns are bytes over **codec** time — a CUDA-event
bracket around the codec call — not wall clock, so they measure the compressor
rather than the harness. `ct_chunks` / `dt_chunks` say how many chunks
are behind each figure. Chunks are excluded rather than counted as
instantaneous when they cannot be timed: a chunk stored raw has nothing to
invert, and a **CPU codec in the action space** (brotli, zlib, ...) reports no
time at all, because the measurement is a CUDA-event bracket around a GPU
codec call.
