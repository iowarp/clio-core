# Prediction accuracy: NeuroPress NN vs XGBoost vs HCompress's cost model

Reviewer 2 asked for a reference point against which to judge our model's
accuracy. This directory answers that with one table: MAPE and R2 per predicted
metric, plus prediction correctness, for our model, the XGBoost baseline, and
the cost model from **HCompress** (Devarajan et al., *HCompress: Hierarchical
Data Compression for Multi-Tiered Storage Environments*, IPDPS 2020, Sec. IV-D),
evaluated on identical chunks and identical candidate configurations.

Nothing here selects a codec or drives a compression call. It is a measurement
harness: the HCompress predictor is scored, never deployed.

## The exact command

```bash
# 1. Measure. One job per workload; all 32 candidate configurations are
#    measured per chunk, with decompression time AND reconstruction quality.
cd /projects/bekn/imuradli/np-hcompress/jobs
for w in nyx vpic lammps warpx ai; do
  ACC_OUT=/projects/bekn/imuradli/np-hcompress/runs/acc-09171447 sbatch acc.sbatch $w
done

# 2. Build the table (prepare -> HCompress -> MAPE/R2/correctness).
cd /u/imuradli/clio-core/paper-benchmark/model-accuracy
./make_table.sh \
  --corpus /u/imuradli/clio-core/paper-benchmark/benchmark_results_600k.csv \
  --campaign /projects/bekn/imuradli/np-hcompress/runs/acc-09171447 \
  --out /projects/bekn/imuradli/np-hcompress/out \
  --feedback-interval 1 --feedback-scope executed

# 3. The two variants reported beside it.
./make_table.sh --out /projects/bekn/imuradli/np-hcompress/out --skip-prepare \
  --seed-file seed_largest.csv --tag _seedlargest \
  --feedback-interval 1 --feedback-scope executed
python3 sweep_feedback.py --inputs /projects/bekn/imuradli/np-hcompress/out/inputs \
  --out /projects/bekn/imuradli/np-hcompress/out/feedback_sweep.csv --jobs 12
```

Outputs land in `/projects/bekn/imuradli/np-hcompress/out/`:
`accuracy_long*.csv` (one row per setting x model x metric), `accuracy_table*.tex`
(paste-ready `tabular`, needs booktabs), `accuracy_table*.md`, `coverage*.csv`
(how many chunks and how many MEASURED values back each column),
`correctness*.csv`, `feedback_sweep.csv`. Data stays out of the repo; only code
lives here.

## Running this on another machine

The scoring step is **pure CPU** — "nothing is re-measured and no model is
retrained" — so a second machine can reproduce the whole model comparison with
no GPU, no nvCOMP/cuSZ/ndzip and no simulation dumps, provided it is given the
campaign CSVs. Only the *measurement* step needs the GPU stack.

**In git, so a clone is enough:** every script here, the HCompress predictor
itself (`context-transport-primitives/.../hcompress_ccp_predictor.{h,cc}`),
`hcompress_ccp_eval.cc` (`make_table.sh` compiles it with g++ on first use, so
the binary is not shipped), and **`model.nnwt`** — the NN's weights, found
relative to this file.

**NOT in git — transfer these out of band:**

| artifact | size | needed by | why it is not in git |
|---|---|---|---|
| `benchmark_results_600k.csv` | 124 MB | **both baselines** — XGBoost's held-out split and HCompress's seed | `.gitignore`; measurement data does not belong in the repo |
| `xgb_model.pkl` | 13 MB | XGBoost | it belongs to the upstream **NeuroPress** checkout, not to this repo |
| a campaign, `<wl>/<wl>/{explore.csv,dist.csv}` | ~10s of MB | everything | produced by `run_campaign.sh`; regenerating it needs the GPU stack and ~93 GB of dumps |

**HCompress needs no weights file.** Unlike the NN (`model.nnwt`) and XGBoost
(`xgb_model.pkl`), its cost model is *fitted at runtime*: `Seed()` does a ridge
/ RLS fit over a profiler sample that `prepare_inputs.py` derives from the
corpus into `seed.csv`, and its only data-dependent input at prediction time is
each chunk's distribution class from `dist.csv`. Given the corpus, there is
nothing pre-trained to ship.

**Paths.** `--nnwt` resolves relative to this file, so it needs no
configuration. `--xgb` defaults to `$NEUROPRESS_DIR/neural_net/weights/
xgb_model.pkl` with `NEUROPRESS_DIR` defaulting to `~/NeuroPress`; set that
variable or pass `--xgb`. The field paths in `run_campaign.sh` are all
`${VAR:-...}` overridable, but their defaults point at this cluster and are
only relevant if you re-measure.

Scoring an already-measured campaign elsewhere:

```bash
export NEUROPRESS_DIR=/path/to/NeuroPress        # for xgb_model.pkl
./make_table.sh --corpus /path/to/benchmark_results_600k.csv \
                --campaign /path/to/campaign --out out
python3 fig8_model_chunks.py --inputs out/inputs --out fig8d_chunks.csv
python3 ../plot/plot_fig8_models.py --chunks fig8d_chunks.csv --out figures
```

## What was measured

One campaign, one build, `explore-balance` with exploration forced on every
chunk (`--explore-k 31 --explore-thresh -1`), error bound 0.05, 8 MiB chunks,
`MEASURE_DT=1`, `MEASURE_QUALITY=1`, prediction reuse off. Identical to
`figure_8.sh`'s configuration except that quality measurement is ON -- without
it there is no measured PSNR and the PSNR column could only be n/a.

| Setting | Chunks | Rows | Where the chunk bytes come from |
|---|---|---|---|
| VPIC | 656 | 20,992 | in situ, `CLIO_VPIC_RAW_DIR` |
| Nyx | 1,056 | 33,792 | replay, `np-nyx-256-2000/fields` |
| LAMMPS | 603 | 19,296 | in situ, `--raw` |
| WarpX | 500 | 16,000 | replay, `np-warpx-1000-i5/fields` |
| AI | 656 | 20,992 | replay, `np-ai-vitb16/fields` |

HCompress needs a per-chunk distribution class, which means it needs the
buffers. The two in-situ workloads never write theirs to disk, so the campaign
turns on the drivers' OWN dump options, classifies inside the job, and deletes
the bytes before exiting -- no code change, and the run's raw data is never
kept. `classify_chunks.py` is a line-by-line port of the repo's
`DistributionClassifier`, sub-sampling 65,536 elements per chunk with a stride
over the whole buffer.

Measured class mix (this is HCompress's only data-dependent input, so how much
it varies is how much information that input carries):

| Setting | Class mix |
|---|---|
| VPIC | normal 533, constant 123 |
| Nyx | uniform 413, constant 284, gamma 220, exponential 92, normal 47 |
| LAMMPS | uniform 400, normal 202, constant 1 |
| WarpX | (see `inputs/warpx/rows.csv`; 14 chunks in a class the seed never profiled) |
| AI | 159 chunks in a class the seed never profiled |

The synthetic corpus is NOT a reported setting. It is what every static model
was fitted on -- our NN's and XGBoost's training set, and HCompress's profiler
seed -- so scoring its held-out split measures how well each model reproduces
its own training distribution, a different question from this table's.
`prepare_inputs.py` still writes `inputs/synthetic/`, so it can be scored by
adding it back to `SETTINGS` in `accuracy_table.py`.

## MAPE ceiling

The reported MAPE caps each row's absolute percentage error at **100% before
averaging**, which is this repo's own convention for the statistic
(`plot_fig8.py`'s per-chunk `MAPE_CLIP`). Capping per row, rather than clipping
the reported mean, is what keeps the comparison alive above the ceiling: a
model whose errors are all 30% stays distinguishable from one whose errors are
half 10% and half 2000%, where clipping the mean would put both at 100.
`accuracy_long*.csv` carries `mape_uncapped` beside it, and `median_ape` beside
that. `--mape-cap 0` turns the ceiling off.

## The five rows, and what each one is

| Row | What it is |
|---|---|
| NeuroPress NN | the shipped `model.nnwt`, no online update. The static counterpart to HCompress seed-only, and the only NN state that can report all four metrics from one model. |
| XGBoost | upstream's trained `xgb_model.pkl`, also static. |
| HCompress CCP (seed only) | fitted on the profiler rows, never updated. |
| HCompress CCP (+ feedback) | the same fit, updated from measured outcomes during the run. |
| NeuroPress NN + online learning | the predictions the RUN ITSELF made, from the exploration log, with online SGD active. The adaptive counterpart to (+ feedback). |

Neither shipped model is retrained. The offline NN reproduces the runtime's own
GPU predictions to **5e-6 relative** on the first (cold) chunk of every
workload, so the "NeuroPress NN" row is the deployed network, not a
reimplementation of it.

**Every model is clamped the same way**: predicted times floored at 1 ms,
predicted ratio capped at 100x, which is the deployed policy and what the
kernel already applies to the NN's own outputs. The `*_unclamped*` outputs
report the same table without them; the floor is not cosmetic, because most
measured compression times here are under 1 ms.

## Seeding data

HCompress's design is an offline profiler that "benchmarks every library over
sample data and writes a JSON seed". Here that seed is **the corpus our own
model was trained on**: `benchmark_results_600k.csv`, filtered to
`success == True`, split by FILE with upstream's own procedure
(`neural_net/core/data.py`: sorted files, `RandomState(42)`, last 20% held
out). HCompress is seeded on the **train** side -- 401,241 rows, 6,272 files --
and every model is evaluated on the **val** side, 1,568 files the NN and
XGBoost never saw. Written to `inputs/seed.csv`, and the fitted model to
`inputs/hcompress_ccp_seed.json`.

16 KiB files are dropped from both sides. The shipped weights' own input box
says they were not trained on them (`x_mins[4] = 65,536` while the corpus goes
down to 16,384), so keeping them would put out-of-distribution rows in the
in-distribution block.

A second seed, `inputs/seed_largest.csv`, restricts the profiler to its largest
buffers (4 MiB, 101,115 rows) and is reported as `*_seedlargest*`. The reason is
in the caveats below: this corpus's speeds are strongly size-dependent, and 4
MiB is the closest available size to the deployment's 8 MiB chunks, so it is the
most favourable honest seeding choice for the baseline.

## Feedback interval

Reported at **n = 1** (every operation), forget factor **1.0** (dlib `rls`'s
default), scope **executed** (HCompress's own information budget: one
configuration per chunk, the one that actually ran).

`feedback_sweep.csv` sweeps n over 1, 8, 32, 128 and the forget factor over
1.0 and 0.98, in both scopes. What it shows:

- **n barely matters.** At forget 1.0, scope executed, going from n=1 to n=128
  changes the reported MAPE by less than 2x on most settings; the largest
  effects are LAMMPS compression time (182% -> 1152%) and VPIC decompression
  time (457% -> 5647%). n=1 is the best setting everywhere, so it is the one
  reported. On the synthetic hold-out n has no effect at all.
- **Scope matters far more than n**, by up to 10x. Giving HCompress the same
  information our NN actually received in this campaign -- all 32 measured
  configurations per chunk, because exploration was forced -- takes VPIC
  compression time from 4,339% to 419%, Nyx decompression time from 38,169% to
  5,748%, and LAMMPS ratio from 1,328% to 148%. Both scopes are in the sweep;
  the table reports the stricter one, which is the baseline's own design.
- **A forget factor below 1 is not usable here**, and this is a property of the
  model's inputs rather than a tuning failure. With an all-categorical design
  the regressor is rank-deficient -- every library key not yet executed and
  every distribution class not yet seen is a direction with no data -- and
  discounting the past inflates the covariance without bound in exactly those
  directions. The sweep shows the result: synthetic compression time goes from
  99% to 739%, AI decompression time from 39,878% to 77,182%. dlib's default of
  1.0 is both the paper's setting and the only stable one.

## Reported as n/a, and why

| Column | Rows | Why |
|---|---|---|
| PSNR MAPE | both HCompress rows | **The model has no quality output.** The paper's outputs are compression speed, decompression speed and compression ratio; PSNR, SSIM and pointwise error have no prediction. The predictor returns `psnr_db < 0` ("not predicted") rather than 0, which in this codebase means "measured and lossless", i.e. maximal quality -- reporting 0 would be a fabricated answer. Never imputed. |
| PSNR MAPE | NeuroPress NN + online learning | The exploration log does not carry the network's PSNR head. Its `psnr_db` column is the ANALYTICAL PSNR derived from (range, error bound) and is a different quantity; the static "NeuroPress NN" row reports the network's real PSNR prediction. |
| data format (input) | all settings | **Not available on either side.** The corpus profiles raw `.bin` buffers and the campaign compresses raw in-situ and replay buffers. There is no container, no HDF5, no format label anywhere, so the input is encoded as empty and contributes no term. The encoded model width printed by the tool shows it: `1 intercept + 1 type + 0 format + 32 library + 7 distribution`. |
| data type (input) | all settings | Present but **constant**: `float32` in every corpus row (its only dtype) and in every workload. It is encoded, but a constant input cannot carry information -- so of HCompress's four inputs, two are degenerate here and only library and distribution actually vary. |

PSNR is scored on the quantized rows only, which is where a finite PSNR exists:
a lossless configuration reconstructs exactly, so there is nothing to predict
or to score against. `coverage*.csv` gives the exact row count behind every
column.

## Caveats that change how the table reads

**1. The corpus's time labels are not codec time, and the campaign's are.**
This is the single most important thing to know before quoting a time MAPE.
In the corpus, 256x more bytes moves `compression_time_ms` by only 1.58x
(11.9 ms at 16 KiB, 18.8 ms at 4 MiB): the label is dominated by a fixed
per-call cost. The campaign's `ct_ms` is CUDA-event kernel time, median 2.34 ms
for an 8 MiB chunk. Per codec the two differ by 9x to 78x.

Consequently the time columns on the five real workloads measure a
label-convention mismatch inherited from the training data, not model quality,
and every statically trained model inherits it. Time accuracy is compared
like-for-like only inside the Synthetic hold-out block, whose labels are the
corpus's own. **Ratio and PSNR are the same quantity on both sides and compare
cleanly everywhere** -- as does prediction correctness, which is scale-free.

**2. HCompress's inputs cannot see most of this corpus's variance, by design.**
Its in-sample fit on its own seed data is R2 = 0.04 for compression speed.
That is not a broken fit: adding `log2(size)`, which is NOT one of its four
inputs, lifts the same fit to 0.54, and restricting the seed to one buffer size
lifts it to 0.31. The remaining variance is within-cell -- fill mode, bin width
and perturbation vary inside every (type, format, library, distribution) cell
and the model has no input for any of them. The paper's adjusted R2 of 94% on
seed data is not reachable on this corpus by any fit restricted to those four
categorical inputs; the number to compare against it is the 0.04/0.31 above.

**3. Exploration is what makes all 32 configurations measurable**, and it also
means our NN's online row saw more feedback than a production run would. That
is why the sweep reports both feedback scopes, and why the static
"NeuroPress NN" row is the one placed against HCompress seed-only.

**4. One configuration is timed once per chunk.** A single slow sample reads as
a large error; `accuracy_long.csv` carries `median_ape` beside `mape` for
exactly this reason, and the medians are the typical chunk.

## Files

| File | What it is |
|---|---|
| `run_campaign.sh` | one workload's measured sweep + in-job distribution classification |
| `classify_chunks.py` | the distribution class, ported from `DistributionClassifier` |
| `prepare_inputs.py` | one evaluation frame per setting + HCompress's seed |
| `hcompress_ccp_eval.cc` | drives the predictor: seed-only and + feedback in one pass |
| `models_offline.py` | the shipped NN and XGBoost, run offline |
| `accuracy_table.py` | MAPE, R2, prediction correctness -> CSV + LaTeX + Markdown |
| `sweep_feedback.py` | the feedback-setting sweep |
| `make_table.sh` | all of the above, in order |

The predictor itself lives with the other models, behind the same interface:
`context-transport-primitives/{include/clio_ctp,src}/compress/model/hcompress_ccp_predictor.{h,cc}`.
It is in no CMake target -- the production build is unchanged -- and
`make_table.sh` compiles it on demand.
