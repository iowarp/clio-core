# LAMMPS workload — measurements

Measurements. Moved out of `README.md`, which is the operational document: how
to run the workload, how to get its data out, and the parameters that make that
data evolve.

Every number here was measured on one A100 at the settings its own section
states. Where a section predates the current defaults it says so.

### This is the only float64 workload, and it changes what the numbers mean

`Atom::x/v/f` are `double`: 8 bytes per value, where Nyx, VPIC and WarpX are
all float32. Two consequences.

**The model cannot express the right shuffle stride.** NeuroPress's action space
encodes shuffle as one bit meaning 4 bytes, so an 8-byte stride is unreachable
by inference. Only the static configs can use it. Measured, box 40 / 100 steps:

| config | ratio |
|---|---|
| `static-zstd` (no shuffle) | 1.229 |
| `static-zstd-s4` (4-byte, what the model can pick) | 1.229 |
| `static-zstd-s8` (8-byte, matches float64) | **1.245** |

So the correct stride is worth ~1.3% here -- real, but far from the headline it
is on float32 data. Worth knowing before attributing a LAMMPS result to the
shuffle.

**NeuroPress reads every buffer as float32 regardless.** That is upstream's
behaviour and Clio keeps it, so the quantizer sees float64 bytes reinterpreted
as pairs of float32 words.

### The data evolves, but its compressibility barely does

The lattice melts -- that is the point of the deck -- and the statistics say so
clearly:

| field | step 0 | step 250 | step 500 |
|---|---|---|---|
| `position` entropy | 6.03 | 7.13 | 7.02 |
| `force` MAD | **9.9e-14** | 1.85e+01 | 1.85e+01 |

Force MAD at step 0 is essentially zero: on a perfect fcc lattice the forces
cancel by symmetry. Once melted it jumps by fourteen orders of magnitude.

Yet the compression ratio moves hardly at all -- `position` spans 1.18x across
the whole run, `velocity` 1.03x, `force` 1.38x, and the aggregate sits near
1.05-1.09x. Entropy is 6-7.4 of 8 bits per byte: the structure is in the
high-order bits while the float64 mantissa is effectively random and dominates.
Same conclusion as VPIC -- see its README -- and no deck parameter changes it.

### The error bound is INERT on float64, and the selection log does not say so

Three bounds at box 20 / 500 steps / `--require-device`, each with its own
`--raw` (a Kokkos trajectory is not bit-reproducible, so a lossy run must be
compared against its OWN originals — two runs of this deck diverge to 5e-9 by
step 500):

| | stored ratio | quantize chosen | quantize APPLIED | worst \|err\| |
|---|---|---|---|---|
| lossless | 1.043× | — | — | 0 |
| `--eb 0.001` | 1.048× | 27/78 | **0** | **0** |
| `--eb 0.01` | 1.039× | 26/78 | **0** | **0** |
| `--eb 0.1` | 1.043× | 27/78 | **0** | **0** |

**Every blob came back bit-exact at every bound.** The ratio moves by under a
percent and not monotonically — `--eb 0.01` is *worse* than lossless. Repeated
runs give these figures to three decimals, so that is a real selection effect,
not variance.

`selection.csv`'s `quantize` column records that a quantize action was
**chosen**, not that quantization **ran**. The tell is `actual_psnr` in the same
row: it is seeded to -1 and overwritten only inside upstream's
`if (d_quantized && quant_result.isValid())`, so `actual_psnr > 0` is the only
evidence the quantizer executed. Read it before reporting a lossy LAMMPS
number. `../viz_bound.py` plots exactly this comparison:

| | chunks | quantize chosen | applied |
|---|---|---|---|
| nyx (float32) eb 0.1 | 156 | 108 | **107** |
| lammps (float64) eb 0.1 | 78 | 27 | **0** |

The cause is the float32 reinterpretation this README already documents.
Measured on a real velocity chunk: read as float32, **396 of 192,000 words are
non-finite** and the finite span is ±3.4e38. `compressor_runtime.cc` gates
quantization on a finite range and declines cleanly — which is correct, and
invisible unless you look at `actual_psnr`.

An earlier revision of this section reported 1.174× at `--eb 1e-3` on box 40 /
100 steps with 6 of 18 rows quantized. Re-run on this build with
`--require-device`, that configuration gives **1.049×** with 8 chosen and **0
applied**, against 1.091× lossless — the lossless figure reproduces, the lossy
one does not. Treat the old number as stale.

### Can LAMMPS be float32? Yes, and the bound wakes up

`--f32` downcasts `x/v/f` to float32 before staging and declares
`data_type_ = 1`. Host gather only (`--order id`), so `run_config.sh --f32`
also sets `CLIO_NEUROPRESS_STAGE_H2D=1` — the quantizer needs a device pointer,
and the host gather does not produce one. Chunk size must halve with the
element width (`--chunk 384000` at 32,000 atoms) to keep one chunk per field
per frame.

| | float64 | float32 |
|---|---|---|
| payload staged | 57.1 MiB | 28.6 MiB |
| lossless | 1.043× | 1.065× |
| `--eb 0.001` | 1.048× | **1.335×** |
| `--eb 0.01` | 1.039× | **1.378×** |
| `--eb 0.1` | 1.043× | **1.456×** |
| quantize chosen → applied | 27 → **0** | 27 → **26** |
| bytes on the tier, best case | 57.2 MiB | **19.6 MiB** |

Monotone in the bound, which float64 never was. End to end that is **2.8×**
against float64-lossless, where every float64 configuration sits at 1.04×
whatever you ask for. At `--eb 0.1` the velocity error reaches 0.095 on 100% of
elements, `force` moves by 2.7e-13, and `position` is still returned untouched
because the selector declines to quantize it at any bound.

The narrowing is not free and is not NeuroPress's doing: LAMMPS state is
`double`, so `--f32` discards ~29 bits of mantissa before the compressor is
involved. Whether that is acceptable is a question about the trajectory, not
about compression. What the table shows is only that the bound becomes
reachable once it is done.

### `brotli` in a blobs.csv is NOT brotli — it is "stored raw"

NeuroPress's action space is nvcomp codecs only; brotli is not in it, and a
`brotli` row in `blobs.csv` never meant one ran. Two facts collided:

- `compress_lib_ == 0` is how the compressor marks **stored raw** — the chosen
  codec did not shrink the chunk, so the caller's bytes went to the tier as
  they were (`compressor_runtime.cc`, "Compression not beneficial").
- **brotli occupies wire id 0** in Clio's registry
  (`compress_factory.h:492`), so `NameForWireId(0)` answers `"brotli"`.

A chunk no codec ever ran on therefore read as a codec that never ran. The
`stored` column was always right — the drivers already guard it with
`(lib != 0) ? compressed : bytes` — only the NAME was wrong.
`neuropress_lammps_step_trace` and `neuropress_data_path_trace` already named
the outcome instead; the lammps, field-replay, nyx-insitu, vpic-insitu and
gpu-static drivers did not, and now do:

```
velocity/step_0/chunk_0  lib=0   codec=raw(not-beneficial)  stored=768000
position/step_0/chunk_0  lib=22  codec=nvcomp-bitcomp       stored=644392
```

**`selection.csv` was never affected.** Its `lib_name` records what NeuroPress
*chose*, and that is always a real nvcomp codec — on the ramped run, wire ids
13/16/21/22 only, never 0. So "what was chosen" and "what was stored" are
different questions and live in different files:

| file | column | answers |
|---|---|---|
| `selection.csv` | `lib_name` | which codec the model ranked first |
| `blobs.csv` | `codec` | what actually landed on the tier, `raw(not-beneficial)` if nothing shrank |

A stored `ratio` below 1.0 in `blobs.csv` is the tell: the codec ran, expanded
the chunk, and was discarded.

### `quantize` in the selection log is a REQUEST, and the codec is applied either way

Two separate things live in a NeuroPress action, and conflating them is the
easiest mistake to make here:

- the **library** (zstd, ans, bitcomp, …) is *always* applied. A chunk marked
  `quantize=1` was still compressed with its chosen codec — losslessly.
- the **quantize bit** is a request that the quantizer may decline, and on
  float64 it declines every time.

So **the ratio can move between a lossless run and a lossy one without a single
byte being approximated.** The bound changes which codec the model ranks first;
that is all it changes. On the box-20/500-step run, the entire lossless-1.043×
vs eb-0.01-1.039× gap is **one chunk**:

| blob | lossless | eb 0.01 |
|---|---|---|
| `force/step_0/chunk_0` | `lz4`, 495,181 B | stored raw, 768,000 B |
| `velocity` (26 chunks) | — | **0 bytes different** |
| `position` (26 chunks) | — | 16 bytes different |

That single 272,819-byte swing is the whole thing — the step-0 force field,
where a perfect lattice makes forces cancel to ~1e-13 and compress 1.55×, and
where the bound pushed the ranking onto an action that expanded instead.

### Temporal redundancy: zero per value, 22.6% per byte

`viz_atoms.py` measures how much of a frame survives the next timestep. Per
value the answer is **0.00%, at every frame, for every field** — not one of the
96,000 doubles in a frame is unchanged, where the Nyx blast starts at 99.9% and
ends at 57%. That number is saturated and on its own misleading. Per byte:

| byte position | 7 (sign+exp) | 6 | 5 | 4..0 |
|---|---|---|---|---|
| identical to previous frame | **99.4%** | 78.2% | 1.1% | ~0.4% |

22.6% of bytes overall. **That is the float64 story in one row**: there is real
structure, it lives in one byte out of eight, and an 8-byte shuffle is what
puts those bytes next to each other. The 4-byte stride NeuroPress can encode
splits the double down the middle and puts an exponent byte beside a mantissa
byte — which is why the shuffle result above is worth only ~1.3% here.

### `--eb` is an ABSOLUTE bound, and LJ units make it mean three different things

Even where a quantize action is *chosen*, the same absolute bound is wildly
mis-scaled across the three fields:

| field | typical MAD | `1e-3` relative to it | quantized |
|---|---|---|---|
| `force` | 9.9e-14 | 1.0e+10 | 1 of 6 |
| `position` | 1.55e+01 | 6.4e-05 | 2 of 6 |
| `velocity` | 1.50e+00 | 6.7e-04 | 3 of 6 |

For `position` the bound is ~16 bits of precision on a box ~42 wide, so
quantization buys little; for `force` at step 0 it is ten orders of magnitude
looser than the data. Contrast VPIC, where the same 1e-3 lets the quantizer
pick `prec=8` and delivers a 5.3x gain. **A single `--eb` is not a comparable
setting across workloads**, and a LAMMPS lossy result should not be read beside
a VPIC one as if it were.

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

## Results

> **Measured before the evolution study changed the defaults to
> `--temp 6.0 --skin 0.8 --every 5`.** These numbers are at upstream's
> `--temp 3.0` and neighbour settings. The ordering is unaffected.

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


## The 1,000-timestep evolution study

Ranking behind `README.md`'s "Default Evolving Benchmark Configuration". Seven configurations, 101 frames
each, 1,800 block samples per configuration, 1 MiB blocks, sampled every 10
timesteps. Raw per-block values in `../evolution-study/lammps/`.

Average normalized block evolution: **0.4368**, with **100.0% of blocks active**
and **0.00% of cells bit-identical** to the previous frame.

| config | parameters | mean E | median | active blocks | p10 interval | last quarter | NVE energy drift | sim s |
|---|---|---|---|---|---|---|---|---|
| **`melt_hot_nb`** | `--temp 6.0 --skin 0.8 --every 5` | **0.4368** | 0.5048 | 100.0% | 0.4360 | 0.4364 | **-0.37%** | 38 |
| `melt_hot` | `--temp 6.0` (upstream neighbour settings) | 0.4335 | 0.5045 | 97.3% | 0.4269 | 0.4334 | **-3.52%** | 31 |
| `ramp_hot_nb` | `in.melt_ramp T0=0.05 T1=12.0`, `--skin 0.8 --every 5` | 0.4373 | 0.5572 | 100.0% | 0.3952 | 0.4610 | thermostat | 36 |
| `melt_nb` | `--temp 3.0 --skin 0.8 --every 5` | 0.3917 | 0.4440 | 100.0% | 0.3912 | 0.3916 | -0.23% | 30 |
| `ramp_hot` | `in.melt_ramp T0=0.05 T1=12.0` | 0.4338 | 0.5563 | 94.7% | 0.3954 | 0.4575 | thermostat | 31 |
| `ramp` | `in.melt_ramp T0=0.05 T1=6.0` | 0.4153 | 0.5135 | 90.9% | 0.3818 | 0.4422 | thermostat | 32 |
| `melt` | `--temp 3.0` (upstream exactly) | 0.3892 | 0.4441 | 83.6% | 0.3837 | 0.3894 | -0.51% | 32 |

**No cell is ever bit-identical to the previous frame in any configuration.**
This is the one workload where the temporal-redundancy number is identically
zero: float64 positions, velocities and forces in a live MD trajectory never
repeat a bit pattern. Everything this workload gives the compressor is new
bytes, which is why its ratios sit near 1.1x while Nyx reaches 150x.

### Why the evolving configuration was chosen

**The neighbour-list settings are a correctness condition at this temperature,
not tuning.** Upstream's `neighbor 0.3 bin` with
`neigh_modify every 20 delay 0 check no` is sized for the example's T=3.0, where
an atom covers ~0.22 sigma between rebuilds. At T=6.0 it covers ~0.33 sigma,
crosses the 0.3 sigma skin, and pairs are silently missed. The cost is visible
in `fix nve`, where total energy is supposed to be conserved:

| | upstream neighbour settings | `--skin 0.8 --every 5` |
|---|---|---|
| `--temp 3.0` | -0.51% | -0.23% |
| `--temp 6.0` | **-3.52%** | **-0.37%** |

A 9.5x improvement at T=6.0 against 2x at T=3.0 — the drift was the missed
pairs, and it scaled with temperature exactly as the skin argument predicts.
It also *inflated* nothing: the repaired run scores HIGHER (0.4368 against
0.4335), lifts active blocks from 97.3% to 100.0%, and raises the minimum block
evolution from 0.001 to 0.018. The coarse list was leaving some blocks
artificially static as well as leaking energy.

**Raising the temperature is what buys the evolution.** At the same neighbour
settings, T=6.0 against T=3.0 is 0.4368 against 0.3917 mean, and 100.0% against
100.0% active — the gain is in magnitude, not coverage. The run is flat across
the whole 1,000 steps (per-decile means 0.4468 -> 0.4364), because an LJ liquid
at fixed temperature is statistically stationary.

**On the temperature ramp.** `ramp_hot_nb` has a marginally higher *mean*
(0.4373 against 0.4368, a tie within run-to-run spread) but a lower sustained
score, because it starts from a cold crystal: its per-interval mean climbs 0.373
-> 0.461 across the run, so its 10th percentile is set by the first frames.
On the criteria this study ranks by, `melt_hot_nb` wins.

**That is not the whole story, and `in.melt_ramp` is still the right deck for a
different question.** A stationary hot liquid gives every frame the same *kind*
of data — high evolution, unchanging character. The ramp deliberately moves the
state point from crystal to hot liquid so the block statistics change
monotonically, which is what exercises a per-chunk *selector* rather than a
per-chunk *codec*. Use `--deck in.melt_ramp --var T0=0.05 --var T1=12.0
--var NSTEPS=<steps> --skin 0.8 --every 5` when that is what is wanted; the
neighbour settings are needed there too, and more so, since it ends hotter.
