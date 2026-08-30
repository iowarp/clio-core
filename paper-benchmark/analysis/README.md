# Exploration-log compressibility analysis

Turns a Clio-NeuroPress **exploration log** into a reproducible answer to:

> Why does this particular scientific-data chunk compress the way it does — and
> can cheap statistics predict the best compression strategy without sweeping
> every option?

```bash
./analyze_exploration.py explore.csv
./analyze_exploration.py --input explore.csv --output analysis_results/
./analyze_exploration.py warpx.csv vpic.csv nyx.csv lammps.csv
```

The default command runs the complete pipeline; the flags only turn work *off*
(`--no-figures`, `--no-cross-workload`, `--seed N`). With several inputs each is
analysed independently and a `cross_workload/` comparison is written beside them.

Requires `numpy pandas scipy scikit-learn matplotlib`.

> Note on multiple inputs: use `-i` for each, since `--name` cannot be
> interleaved with bare positional arguments.
> `./analyze_exploration.py -i a.csv --name a -i b.csv --name b`

---

## The short version of the answer

The report leads with **mechanism**, because one of the three features has a
closed-form consequence rather than only a correlation, and that turns out to
organise everything else.

`entropy` is the Shannon entropy of the 256-bin **byte histogram**. A coder that
assigns one codeword per byte symbol and reads no order cannot beat
`ratio <= 8 / H` — the source coding theorem applied to the exact quantity the
feature measures. So the ratio factorises:

```
ratio  =  (8 / H)              x   excess
          the byte histogram       everything ORDER contributes
          entropy knows this       entropy CANNOT know it, by
          exactly                  construction
```

On the development log that split is stark and it is the whole story:

| | behaviour on lossless, unshuffled data |
|---|---|
| `nvcomp-ans` | never beats `8/H` (0 of 44 chunks), reaches **0.86** of it |
| `nvcomp-zstd` | beats it on **44 of 44**, by a median **2.59×** and up to 338× |

So entropy is not a mediocre predictor of compressibility. It is an **exact**
predictor of one factor and **blind to the other by construction** — and which
codec you ask decides how much of the answer it gives you. For the codec that
actually wins this workload, most of the achieved ratio lives in the factor the
current features provably cannot see. No amount of model capacity closes that;
only a feature that measures *sequence* does.

Three consequences the pipeline then measures rather than assumes:

- **Byte shuffle is a permutation**, so the histogram — and `entropy`, `mad`,
  `second_deriv` with it — is *mathematically invariant* under it. Every bit of
  shuffle's gain (up to **2.08×** here) is therefore order, by construction.
  Which yields a free probe: a codec that never beats the global bound *and yet
  gains from a permutation* is not coding the global histogram at all, but a
  per-block one. `nvcomp-cascaded` and `nvcomp-ans` both do exactly this.
- **The timings are downstream of the ratio.** Time correlates with the data
  properties, but in **96%** of (codec × feature × metric) cells the compressed
  size `chunk_bytes / ratio` tracks it more strongly, and the feature's partial
  correlation collapses once size is held constant. Predict the ratio; the times
  follow.
- **Quantization's split is not identifiable from this log** — and the pipeline
  says so instead of reporting one. The stats are computed once per chunk on the
  *original* buffer, so the post-quantization entropy is never recorded.

---

## What it reads, and where the definitions come from

Every column is interpreted from the code that **writes** it, not from its name.
Three of those definitions change conclusions, so they are worth stating here:

| Column | What it actually is | Where it is written |
|---|---|---|
| `cost` | `w_ct·max(1,ct) + w_dt·max(1,dt) + w_io·bytes/(min(cap,ratio)·bw)`, **lower is better**. Every weight and the cap are env-overridable and *not recorded in the log*. | `models/neuropress_cost.h` |
| `eb_encoded` | The model's **input 3**, `quant ? error_bound : 1e-7`. The `1e-7` is a training sentinel, **not** an error bound. | `neuropress_telemetry.h` |
| `entropy`, `mad`, `second_deriv` | Byte-histogram Shannon entropy in **bits/byte** ∈ [0,8]; `mean\|x-mean(x)\|` and `mean\|x[i+1]-2x[i]+x[i-1]\|` in **raw data units**, on the flattened buffer. | `data_stats_gpu_kernels.cu` |
| `dt_ms`, `meas_*`, `psnr_db` | `-1` means **not measured** — distinct from a measured `0`. `psnr_db` is *analytical* (derived from the bound); only `meas_psnr_db` / `meas_max_error` can witness a bound violation. | `neuropress_telemetry.cc` |

Because the cost weights are not logged, the pipeline **recovers them by least
squares from the `cost` column itself** and reports the residual as proof. On the
development log this recovered `w_ct = w_dt = 0` — that run scored on capped
ratio alone — which no analysis assuming the documented defaults would have found.

### Older logs

Logs written before the sweep carried the model's own inputs have no
`entropy`/`mad`/`second_deriv`. The loader then looks for a `selection.csv` beside
the input and joins those columns on `blob` (they come from the same locals, so
the join is exact) and says so in the report. With neither, every data-property
analysis is reported **unavailable** rather than as a null result.

---

## The things that make it statistically honest

**1. A row is not a sample.** One chunk appears once per candidate configuration —
32 times in a full sweep — carrying the *same* entropy/MAD/second-derivative every
time. So the pipeline keeps two explicit levels (`rows`, one per *(chunk,
configuration)*; `chunks`, one per blob), every correlation is computed either at
chunk level or inside a fixed configuration, and every result carries `n_chunks`
beside `n`.

**2. MAD and the second derivative are in raw data units.** Pooling a density
field and a momentum field and correlating MAD against ratio sorts the chunks by
*field* before it sorts them by data. Every pooled correlation is therefore paired
with a within-field one, and `simpson_flag` marks the cells where the two disagree
in sign. On the development log **many of them do.**

**3. It checks whether the finding is the data or the clock.** On a single run
of an evolving simulation the data properties are very nearly a *function of the
timestep* — on the development log, `corr(entropy, timestep)` is +0.98…+1.0000
inside every field — and compressibility falls over the same run, so the two
correlate without either causing the other. Section 3 of every report gives a
model nothing but the metadata and asks whether it wins; residualises each
correlation on the confounder and flags sign flips; and regroups the
cross-validation on the confounder instead of the chunk. When it fires, the
affected answers are downgraded and say so inline. *(Note that stratifying by
`field` cannot catch this — within a field, "the entropy series" and "the time
series" are the same thing.)*

**4. It answers point 3 rather than only raising it.** The confound has exactly
one answer available inside a single log: compare chunks written *at the same
timestep*, where the clock is held **exactly** constant rather than
statistically. Section 4 of every report builds those matched groups and asks
whether chunks with matching statistics have matching compressibility. On the
development log they do not, and the failure is structured: the three momentum
components of a spherically symmetric blast agree in MAD to 3e-9 relative and in
the second derivative to 2e-8, yet `zmom` compresses 1.76–1.80× better than
`ymom` at *all six* shared timesteps, and `xmom` — which the features rank as the
**roughest** — compresses best of all, up to 2.45× above `ymom`. Within a field,
a rising second derivative tracks a falling ratio; across components at fixed
time it tracks the opposite.

The section also converts this into a bound. Chunks that are numerically
identical in every feature must receive the same prediction from *any* function
of those features, so the spread inside such a class is irreducible and caps
out-of-fold R² for every possible model — reported as a ladder over tolerances,
with how many chunks actually have a twin, because a bound resting on few twins
is loose and loose in the direction of optimism. The workload-independent form
needs no vector fields: fit out of fold, centre each residual on **its own
moment**, and test whether what is left is still ordered by field.

**5. It checks whether the sample is conditioned.** Exploration is trigger-gated
upstream, so a sweep log holds the chunks where the predictor already missed. If
a `selection.csv` sits beside the input, the loader compares the two cohorts,
infers the trigger, and states which way it biases the prediction and regret
figures. On the development log, 44 of 66 chunks were explored and every omitted
one was *under*-predicted — so the explored subset is enriched in over-prediction
by construction.

**6. No leakage.** Splitting rows at random puts the same chunk on both sides of
the train/test boundary and can reach R² ≈ 1 by memorising chunks. Every split is
group-aware on `chunk_uid`, and three progressively harder tests run on top:
chunk holdout → time holdout (early timesteps → later) → field holdout. The *gap*
between them is the finding, and the report prints all of it: on the development
log the chunk-holdout R² is 0.87 while leave-one-field-out has median 0.33 and
the early→later split is −1.89, i.e. worse than predicting the mean.

Ablation verdicts also carry a **re-folding sweep** — the same data dealt into
folds afresh — because at a few dozen chunks a single out-of-fold delta can
change sign across partitions, and a verdict that does is reported as unstable
rather than as an answer.

An analysis that cannot run — no error bounds swept, decompression never measured,
one codec winning every chunk — is recorded in `unavailable_analyses` **with its
reason**, and the report prints the reason where the result would have gone.

---

## Output

```
analysis_results/<workload>/
├── mechanism/     entropy_bound, codec_order_sensitivity,
│                  shuffle_decomposition, locality_probe, timing_mediation,
│                  codec_throughput
├── confounds/     confounder_power, partial_correlations,
│                  feature_confounder_collinearity
├── anisotropy/    matched_pairs, feature_blind_spread, explainable_ceiling,
│                  component_families, component_ordering, residual_by_field
├── summary/       dataset_summary, codec_summary, feature_correlations{,_within_field,_chunk_level},
│                  feature_ablation{,_marginal}, ablation_stability, feature_importance, winners,
│                  codec_win_rates,
│                  codec_{feature_sensitivity,property_profile}, binned_trends, trend_monotonicity,
│                  joint_regimes, conditional_gains
├── paired/        {shuffle,quantization,error_bound}_{comparison,summary}, *_benefit_regimes
├── cost_model/    selection_results, selection_regret
├── prediction/    {ratio,ct,dt}_prediction_errors, prediction_clamps,
│                  prediction_{summary,by_codec,error_drivers},
│                  ranking_quality
├── models/        model_metrics, codec_classifier_metrics, generalisation, discovered_thresholds.json
├── temporal/      property_series, outcome_series, temporal_{series,trends}, codec_switches,
│                  evolution_{joined,effects}
├── counterexamples/counterexamples.csv
├── processed/     chunk_properties.csv, exploration_results.csv
├── figures/       *.png
├── summary.json   every numeric conclusion, machine-readable
└── REPORT.md      the narrative, answering Q1–Q15
```

With more than one input, `cross_workload/` adds sign-agreement, per-workload
feature importance, and **leave-one-workload-out transfer** — run twice, on raw
and on per-workload-standardised features, because a threshold on MAD from one
workload has no meaning in another.

Optional sidecars picked up automatically from the input's directory:
`selection.csv` (features for older logs), `blocks.csv` from
[`../evolution.py`](../evolution.py) (the temporal block-evolution metric).

---

## Reproducibility

Same CSV in ⇒ byte-identical outputs. All models are seeded (`--seed`, default 0),
all tie-breaks are deterministic orderings, and nothing samples. `summary.json`
and `REPORT.md` record the input path, its SHA-256, row and chunk counts, the
analysis git commit, dependency versions, the CLI arguments and the seed.

---

## Tests

```bash
./selftest.py     # synthetic fixtures; asserts on pipeline OUTPUT
```

The fixtures deliberately drive what a given real log may not: the current
31-column header, **several error bounds**, features absent with no sidecar,
`dt_ms = -1` throughout, and a malformed input alongside a good one. Two of them
plant a known relationship (ratio falls with entropy, rises with the bound) and
assert the pipeline recovers its sign.

One fixture plants a **known failure** rather than a known relationship: two
extra components carrying byte-for-byte the same feature values as a third, and
ratios 1.5× and 2× apart. Nothing computable from those features can separate
them, so the matched-control stage must report the log as anisotropic, detect
the vector family from the field names alone, recover the ordering with the
right winner, and pull the R² ceiling below 1. A negative control runs beside
it: on the ordinary fixture, whose fields are unrelated, no vector family may be
invented.

## Extending it

Add a stage as a module under `analysis/`, call it from `pipeline.run()`, register
its table in `LAYOUT` in `analyze_exploration.py`, and render it in
`reporting.write_report()`. Section numbers and every cross-reference in the
prose are derived from the `SECTIONS` list at the top of `reporting.py`, so
adding or reordering a section is a one-line edit there and no number can fall
out of step with the heading it names. A stage that cannot run should call `a.na(name, why)`
rather than return silently — that is what keeps "no effect" and "not measured"
distinguishable in the report.
