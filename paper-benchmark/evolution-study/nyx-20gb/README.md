# Nyx at 20 GB — record of a volume-target run

**Not one of the study's 26 configurations.** This is the answer to a different
question: how to make the Nyx workload produce ~20 GB over its timesteps
*without leaving the evolving regime* the study selected. Measured with the
same `../../evolution.py`, so the numbers compare directly.

```
gen.json  gen.log  gen.nyx.log     the dump run
evolution.json  blocks.csv.gz      its block evolution, 19,200 block samples
insitu.*                           the in-situ run at the same settings
run.sh                             the exact commands, re-runnable
chunk_log.py                       everything recorded about any one chunk
```

## Reading it per chunk

```bash
./chunk_log.py                                   # index: fields, frames, chunks
./chunk_log.py --field xmom --step 799 --chunk 5 # one chunk, all four sources
./chunk_log.py --csv /tmp/chunks.csv             # all 4,800, 33 columns each
```

The four CSVs record the same run from four angles and none is a superset of
the others — model inputs and prediction (`selection`), every candidate
measured (`explore`), what reached the tier (`blobs`), and how far the data
itself moved (`blocks`). Joining the fourth to the other three needs two
conversions, which is the whole reason the script exists rather than an awk
one-liner:

- **step** — the in-situ hook fires after the step completes and labels the
  frame with the step index, so `step_00039` is the frame at simulation step
  40, which `blocks.csv` calls `step_to = 40`. `dump_step = insitu_step + 1`.
- **block size** — a compressor chunk is 4 MiB, an evolution block is 1 MiB, so
  chunk `c` covers blocks `4c…4c+3` and the evolution figures are aggregates
  over four rows.

They are also two separate executions of the same configuration, so the
evolution numbers describe the dump route's bytes, not the in-situ run's.

What the join is for: sorting the 4,800 chunks by `ev_mean` puts the
**quietest quartile at a mean stored ratio of 265× and the busiest at 13.9×**
— the same 19× spread the workload was chosen to produce, now attributable to
individual chunks.

The 20.5 GB of `.f32` and the 5.4 GB compressed tier behind these files are
**not committed** — same rule as the rest of `evolution-study`. `run.sh`
regenerates both in about ten minutes.

## The configuration

```bash
# dump route: 20,535,313,913 B on disk, replayable by every policy
./gen_fields.sh --ncell 256 --steps 2000 --plot-int 40 --out /tmp/nyx-20gb

# in situ: the same bytes through the compressor, no plotfile written
./run_config_insitu.sh explore-balance --ncell 256 --steps 2040 --int 40 \
    --chunk 4194304 --stop-time 1.0 --results /tmp/nyx-20gb-insitu --tag insitu20
```

```
frame bytes = ncell^3 x 6 components x 4 B     128^3 -> 48 MiB   256^3 -> 384 MiB
frames      = steps / int + 1                  (dump route)
            = steps / int                      (in situ -- see below)
payload     = frame bytes x frames
```

### Why 256³ × 2000 and not something cheaper

A CFL timestep halves with the grid, so **2000 steps at 256³ ends at the same
physical state as 1000 at 128³** — the run stays in the regime the study
validated. Three cheaper routes to 20 GB do not:

| ncell | steps | int | frames | total | what it costs |
|---|---|---|---|---|---|
| **256** | **2000** | **40** | **51** | **20.5 GB** | nothing — same end state as the studied default |
| 128 | 1000 | 2 | 501 | 25.2 GB | dumps 2 steps apart barely differ; per-interval E collapses |
| 320 | 1000 | 40 | 26 | 20.4 GB | shock sweeps ~40% as much of the box → mostly frozen blocks |
| 128 | 16000 | 40 | 401 | 20.2 GB | shock leaves the domain long before the end |

Row 3 is the trap worth naming: the Sedov shock advances a roughly **fixed
number of cells per step**, so enlarging the grid at fixed `--steps` shrinks
the fraction of the domain that ever changes. Grid and step count have to move
together.

## Measured

Dump route, 51 frames × 6 components, 306 files, 221 s of simulation.

| | 20 GB run (256³, 2000 steps) | study default (128³, 1000 steps) |
|---|---|---|
| block samples | 19,200 | 4,800 |
| mean E | 0.179 | 0.114 |
| median E | 0.147 | 0.069 |
| p10 interval / last quarter | 0.174 / 0.180 | 0.060 / 0.063 |
| blocks active | 65.0% | 76.5% |
| cells bit-identical, whole run | 81.4% | 80.2% |
| cells bit-identical, last pair | 53.4% | 50.1% |
| non-finite blocks | 0 | 0 |

**Neither difference is a regime change.** Each sampled interval here spans
twice the physical time of the study's (40 steps at 256³ against 10 at 128³),
which is why per-interval E is higher; the end-of-run cells-identical figures,
53.4% against 50.1%, are the ones that show the two runs finish in the same
place. The lower active-block share is a block-size artifact: `evolution.py`
blocks are a fixed 1 MiB, so at 256³ a block covers one eighth the domain
fraction and more blocks sit entirely outside the shock.

Evolution is sustained rather than front-loaded — p10 0.174 ≈ mean 0.179 ≈
last quarter 0.180, against a first interval of 0.065.

### Validity, which the evolution metric cannot check

Computed from the density dumps directly, not from the log:

| | |
|---|---|
| mass drift, frame 0 → 50 | **−2.3e−08** (128³ default: −1.2e−07; rejected cfl 0.9: −1.1e−04) |
| density range at the last frame | 6.2e−04 … 3.49, strictly positive |
| peak compression | 3.49, under the γ=5/3 strong-shock limit of 4 |
| `Enforce minimum density` warnings | 0 |
| NaN/Inf/abort in the log | 0 |

## In situ, same settings

`explore-balance`, lossless, `--learn --explore 3 --threshold 0`, 4 MiB chunks,
`CLIO_NEUROPRESS_REQUIRE_DEVICE=1`. `rc=0`, `host_refusals=0` — it stayed on
the GPU end to end.

| | |
|---|---|
| submitted | 20,132,659,200 B (4,800 blobs = 50 frames × 6 components × 16 chunks) |
| on the tier | 4,600,214,967 B, **stored ratio 4.38×** |
| wall / simulation | 316 s / 313 s |
| GPU arena peak | 21.6 GB used of 30.3 GB reserved |

Per-chunk ratio spread: median 3.55×, mean 96.9×, max 1596×, min 1.00×. Codec
mix over the run: `nvcomp-bitcomp` 3,397, `nvcomp-ans` 1,165,
`nvcomp-cascaded` 147, `nvcomp-zstd` 41, `nvcomp-gdeflate` 13, split between
4-byte shuffle and none. That spread is the reason to run Nyx at all — the
selector sees genuinely different chunks rather than one regime.

### Two traps specific to the in-situ route

**`--stop-time 1.0` is required, and nothing supplies it.** `gen_fields.sh`
parks `stop_time` at 1.0 so `--steps` is the control, but
`run_config_insitu.sh` leaves the deck's `stop_time = 0.01` in force. At 256³
and cfl 0.8 that ends the run near step 360 — about 4 GB instead of 20,
**exiting 0**.

**In situ submits one frame fewer than the dump route writes.** The hook fires
from `Nyx::updateInSitu()`, which has no step-0 frame, so in-situ frames are
`steps/int` where the dump route's are `steps/int + 1`. The run recorded here
used `--steps 2000` and submitted 50 frames; the whole 20,535,313,913 vs
20,132,659,200 B gap between the two routes is exactly that one frame,
402,653,184 B. `--steps 2040` submits 51.

### What exploration did

`--explore 3` measures **three alternatives against the model's own pick**, not
three codecs in total: `insitu.explore.csv.gz` holds exactly 4 rows per blob —
one `primary` at `rank -1` plus alts at ranks 0, 1, 2 — over 4,453 blobs, with
no variance (min 4, max 4). At `--threshold 0` every blob is explored, which is
why the count is flat. The 347 blobs in `selection.csv` that never reach
`explore.csv` came from cache rather than a fresh trial.

**The model's own pick was overridden on 70.6% of explored blobs** — 3,146 of
4,453 adoptions went to an alt, only 1,307 to the primary. By rank: alt 0 won
1,551 times, primary 1,307, alt 1 1,120, alt 2 475.
