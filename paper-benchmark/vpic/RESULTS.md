# VPIC workload — measurements

Measurements. Moved out of `README.md`, which is the operational
document: how to run the workload, how to get its data out, and the
parameters that make that data evolve.

Every number here was measured on one A100 at the settings its own
section states. Where a section predates the current defaults it says so.

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

### A single absolute bound cannot serve sixteen fields

Measured at 34³ × 25 dumps × 16 fields, `dynamic`, all 400 chunks inside their
bound at every setting:

| | stored ratio | quantize chosen | quantize ran |
|---|---|---|---|
| lossless | 1.121× | — | — |
| `--eb 0.001` | 4.429× | 400/400 | 325 |
| `--eb 0.01` | **5.214×** | 352/400 | 324 |
| `--eb 0.1` | **1.325×** | **71/400** | 21 |

**Sharply non-monotone: the loosest bound is the worst by 4×.** At 0.1 the
selector mostly stops choosing quantize and the run falls back to its lossless
profile. The reason is that the fields span roughly ±0.2, so 0.1 leaves three
or four quantization levels and the ranking rejects it — where on Nyx the same
0.1 is 3.8% of density's range and is worth taking.

And a single bound is not survivable across the state vector. Worst relative
error (`max|err| / range`) reached at any of the three bounds:

| field | range | worst relative error | |
|---|---|---|---|
| `rhof`, `jfz`, `ex` | ~4e-01 | 0.6–5.1% | fine at every bound |
| `ey`, `ez` | ~3e-01 | 45.8%, 49.8% | degraded |
| `cbx` | ~3e-01 | 68.6% | degraded |
| `cby`, `cbz` | 3.2e-01 | **100%, 97.2%** | flattened at some bound |
| `div_e_err`, `div_b_err`, `rhob` | ~2e-07 | **100%** | annihilated |

`div_b_err` has a range near 2e-07, so *every* bound on this page is between
five thousand and five hundred thousand times wider than the entire field: it
collapses to one value and the interface reports success. And because amplitude
grows over the run, the same bound changes behaviour within a single run —
`cby` at `--eb 0.1` is flattened at dump 0 (amplitude ±0.045, below the bound)
and returned **bit-exact** from dump 6 on (amplitude ±0.17, so the selector
declines).

Read the `relerr` column, not the `rc` column. Every chunk was inside its bound.

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

> **Measured before the evolution study changed the default
> `VPIC_CLEAN_DIV_INT` to 10.** These numbers are at upstream's 0, where four
> of the sixteen variables are constant arrays and lift the headline ratio
> from 1.061 to 1.304. With cleaning on, a quick-profile in-situ cell reads
> 1.088. The ordering is unaffected.

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


## The 1,000-timestep evolution study

Ranking behind `README.md`'s "Default Evolving Benchmark Configuration".
Four configurations, 100 dumps each, 12,672 block samples per configuration,
1 MiB blocks, sampled every 10 timesteps. Raw per-block values in
`../evolution-study/vpic/`.

Average normalized block evolution: **0.6355**, with **93.8% of blocks active**
and only **8.33% of cells bit-identical** to the previous dump.

| config | parameters | mean E | median | active blocks | cells bit-identical | p10 interval | last quarter | sim s |
|---|---|---|---|---|---|---|---|---|
| **`cleandiv`** | `clean_div_interval=10`, upstream anisotropy | **0.6355** | 0.6908 | 93.8% | 8.33% | 0.6275 | 0.6388 | 133 |
| `coldx_a100` | `vthex=0.0176776695` (anisotropy 100), `clean_div=10` | 0.6314 | 0.6856 | 93.8% | 8.33% | 0.6237 | 0.6316 | 132 |
| `hot_a100` | `vthe=0.3535533906` (anisotropy 100), `clean_div=10` | 0.6091 | 0.6861 | 93.8% | 8.32% | 0.5457 | 0.5468 | 129 |
| `baseline` | upstream exactly, `clean_div=0` | 0.5087 | 0.6786 | 75.0% | 26.89% | 0.4992 | 0.5129 | 131 |


## Why the evolving configuration was chosen — the measurements

**Divergence cleaning is the whole result, and the anisotropy is nearly inert.**
Turning cleaning on moves active blocks from 75.0% to 93.8% and cuts cells
bit-identical from 26.9% to 8.3%. Changing the temperature anisotropy from
upstream's 25 to 100 — by either route — moves the mean by less than 0.005 and
the active share not at all.

Those two numbers are not close to round by accident. Per field:

| | `baseline` | `cleandiv` |
|---|---|---|
| `div_e_err` | mean E **0.0000**, 100.00% cells identical | mean E **0.7842** — the most active variable in the run |
| `div_b_err` | mean E **0.0000**, 100.00% identical | 0.6120 |
| `rhof` | mean E **0.0000**, 100.00% identical | 0.6330 |
| `rhob` | mean E **0.0000**, 100.00% identical | mean E **0.0000**, still 100.00% identical |
| the other twelve | 0.63–0.70 | 0.63–0.70, unchanged to four decimals |

75.0% is exactly 12 of 16 variables and 93.8% is exactly 15 of 16. Cleaning
revives three of the four frozen variables; **`rhob` cannot be revived by any
interval**, because bound charge is accumulated from dielectric materials and
this deck is vacuum, so it is structurally zero for the whole run. The twelve
already-evolving fields are unchanged to four decimals, so cleaning makes the
diagnostics real without perturbing the physics being measured.

**The anisotropy is inert because the metric is saturated by shot noise.** PIC
field arrays sit near noise throughout (~913,000 distinct values in 941,000
cells, measured below), and that noise decorrelates every step regardless of
what the instability is doing. E ≈ 0.63–0.70 *is* the noise floor, and Weibel
growth is a small signal on top of it. Raising `vthe` actively hurt — its last
quarter falls to 0.5468 against 0.6388 — because a hotter plasma saturates the
instability sooner. Upstream's values are therefore kept.

The winning configuration is also the flattest thing in this study: per-decile
means run 0.622 → 0.639 across the 1,000 steps and cells-identical moves 8.22%
→ 8.33%. Sustained is not the difficulty for this workload; the difficulty is
that a quarter of the payload was not being computed at all.
