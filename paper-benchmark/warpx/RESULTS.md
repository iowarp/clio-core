# WarpX workload — measurements

Measurements. Moved out of `README.md`, which is the operational document: how
to run the workload, how to get its data out, and the parameters that make that
data evolve.

Every number here was measured on one A100 at the settings its own section
states. Where a section predates the current defaults it says so.

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

### Read back, one bound annihilates one field and never touches another

`B/y` grows from 0 to 0.11 over the run, so an absolute bound is first wider
than the entire field and later a small fraction of it. Error as a share of
`B/y`'s own range, per diagnostic:

| step | 4 | 12 | 20 | 28 | 36 | 40 |
|---|---|---|---|---|---|---|
| `B/y` range | 3.7e-06 | 5.6e-05 | 4.7e-04 | 3.9e-03 | 4.1e-02 | 1.1e-01 |
| `--eb 0.001` | **100%** | **100%** | **100%** | 24% | 2% | 1% |
| `--eb 0.01` | **100%** | **100%** | **100%** | **100%** | 23% | 9% |
| `--eb 0.1` | **100%** | **100%** | **100%** | **100%** | **100%** | 89% |
| `E/z`, any bound | 0% | 0% | 0% | 0% | 0% | 0% |

`E/z` reaches 1.5e+11, so even `--eb 0.1` — whose largest measured error on it
is 1.0 — rounds to **0.0% of its range at every diagnostic**. One bound, one
file, one run: total loss on one field and invisible on the other. The same
bound also changes behaviour *within* the run as the field amplifies, which no
per-run summary number can show.

**Each policy is its own WarpX run**, so these are not byte-identical
comparisons the way Nyx and VPIC are. Measured run-to-run spread on this
configuration is 0.6% (11.860× vs 11.792× lossless; 15.167× vs 15.197× at
eb=0.01), well below the 27% the bound moves — but it is why these are quoted
to three digits.

## Results

> **Measured before the evolution study changed the default `laser1.e_max` to
> `32.e12`.** These numbers are at the deck's `16.e12` (a0 = 4). The ordering
> is unaffected.

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


## The 1,000-timestep evolution study

Ranking behind `README.md`'s "Default Evolving Benchmark Configuration". Nine configurations, 101 dumps
each, 8,000 block samples per configuration, 1 MiB blocks, sampled every 10
timesteps. Raw per-block values in `../evolution-study/warpx/`.

Average normalized block evolution: **0.1636**, with **88.4% of blocks active**
and **21.99% of cells bit-identical** to the previous dump (94.14% on the first
pair, 7.46% on the last).

| config | parameters | mean E | median | active blocks | cells bit-identical | p10 interval | last quarter | sim s |
|---|---|---|---|---|---|---|---|---|
| **`a0x2`** | `e_max=32e12` (a0 8), `density=2e23` | **0.1636** | 0.0993 | 88.4% | 21.99% | 0.0946 | 0.1151 | 19 |
| `dens16x` | `density=32e23`, deck a0 | 0.1545 | 0.0927 | 87.7% | 28.65% | 0.0877 | 0.1011 | 19 |
| `a0x2_zramp` | `e_max=32e12` + the `z > 0` ramp | 0.1538 | 0.0958 | 89.4% | 22.05% | 0.0824 | 0.0951 | 20 |
| `baseline` | the deck exactly | 0.1498 | 0.0947 | 87.7% | 26.72% | 0.0828 | 0.1026 | 19 |
| `dens4x_a0x2` | `density=8e23`, `e_max=32e12` | 0.1535 | 0.0913 | 88.4% | 22.63% | 0.0801 | 0.0947 | 19 |
| `zramp` | longitudinal density ramp 2e23 -> 1e24 | 0.1425 | 0.0842 | 88.8% | 27.82% | 0.0730 | 0.0866 | 19 |
| `zramp_z0` | the same ramp, guarded to `z > 0` | 0.1425 | 0.0841 | 88.8% | 27.82% | 0.0730 | 0.0866 | 19 |
| `dens4x` | `density=8e23`, deck a0 | 0.1436 | 0.0775 | 87.5% | 28.44% | 0.0731 | 0.0882 | 19 |
| `nomovingwindow` | `do_moving_window=0` | **DISQUALIFIED — resonant cavity, see below** | | | | | | |

### Why the evolving configuration was chosen

**This is the workload where the search mostly failed, and that is the result.**
Every valid configuration has the same shape: a large transient over the first
~300 steps while the laser enters and the wake forms, then a flat plateau.
Per-decile means for the deck baseline run 0.205, 0.277, 0.258, 0.151, 0.111,
0.092, 0.096, 0.100, 0.103, 0.103. The plateau is not an artefact — WarpX's own
`FieldEnergy` reduced diagnostic shows total field energy constant to five
digits from step 400 on (0.10785, 0.10784, 0.10784, 0.10784 J) — it is what a
**quasi-static wake in a co-moving window** looks like. The window follows the
pulse at exactly c, so a fixed block sees nearly the same structure every dump,
and the residual E ~ 0.10 is the wake pattern slipping backward relative to the
frame because its phase velocity is below c.

Against that, the knobs available do very little:

- **Plasma density is inert and not even monotonic** — 4x is *worse* than the
  deck (0.1436 against 0.1498), 16x marginally better (0.1545). All three are
  within a few percent of each other.
- **A longitudinal density ramp does not help** (0.1425), which was the
  hypothesis this study was extended to test: if the plasma the pulse meets
  keeps changing, the wake should keep changing. It does not. Re-running it
  guarded to `z > 0` — the first attempt's profile went negative below
  z = -35 um — reproduced the same numbers to four decimals, so the result is
  the ramp's and not the guard's; and adding the ramp to the *winning* run
  lowers it too (`a0x2_zramp` 0.0737 against `a0x2` 0.0837), so the hypothesis
  is falsified from both directions rather than merely unsupported. The ramp
  does raise coverage very slightly — 89.4% active, the highest in the study —
  but it costs more sustained magnitude than it buys.
- **Laser amplitude is the only knob that moved the plateau.** `a0` 4 -> 8
  raises the mean 9%, the p10 14%, and cuts cells bit-identical from 26.7% to
  22.0%. Physically it drives a deeper blowout, so the bubble evolves rather
  than settling.

**`warpx.do_moving_window = 0` scores nearly 2.2x the winner and is disqualified
on physics the metric cannot see.** With the window off, the pulse — injected at
z = 9 um, 3 um below a `pec` boundary — reflects instead of propagating, and the
box becomes a resonant cavity. `FieldEnergy` over 1000 steps:

| step | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| total (J), `mw=1` | 0.2157 | 0.10785 | 0.10784 | 0.10784 | 0.10784 |
| total (J), `mw=0` | 0.30291 | 0.30291 | 0.28452 | 0.30291 | 0.30291 |

Energy is *exactly* conserved with the window off — PEC walls absorb nothing —
and at step 600 E and B are caught mid-exchange (`E` 0.172 -> 0.096 while `B`
0.131 -> 0.188). That is a standing wave, not laser-wakefield acceleration. Its
evolution rises through the run (first interval 0.117, last quarter 0.453),
which is the tell. The run exits 0 and holds no NaN, so `evolution_rank.py`
cannot reject it automatically; the reason is recorded in its `evolution.json`
under `disqualified` and honoured by the ranking, with the raw diagnostic in
`FE_nomovingwindow.txt`.

**About 9.5% of `E`/`B` blocks and 13.5% of `j`/`rho` blocks never change in any
configuration** — the vacuum region ahead of the pulse, where the fields are
identically zero. That ceiling is why no configuration exceeds ~88.8% active.
