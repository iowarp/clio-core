# Figure 8 — regret and cost MAPE

Does NeuroPress's online model improve over a run on data it never trained on?
Per chunk:

    regret (%)    = (measured cost of the pick / cheapest measured cost - 1) x 100
    cost MAPE (%) = |predicted cost of the pick - measured cost| / measured x 100
    cost          = max(F, ct) + max(F, dt) + bytes / (min(ratio, 100) * 5e6 B/ms)

`plot/fig8_trace.py` scores both at F = 5 ms on predicted and measured alike, as
upstream's trace `mape_cost` does. Upstream's regret uses its unfloored
`real_cost` instead (`real_cost_raw` in the trace), which gives much larger
numbers on these workloads because every codec time is under 5 ms:

| 2k campaign `twok-quantfix-09170206` | chunks | regret, last 20%, F = 5 ms (median) | same, unfloored | cost MAPE, last 20% (median) |
|---|---|---|---|---|
| VPIC   | 2,000 | 6.6% (0.06%) | 580% | 3.6% (0.16%) |
| Nyx    | 2,064 | 9.1% (0.53%) | 494% | 4.7% (0.45%) |
| LAMMPS | 2,004 | 2.3% (0.03%) | 152% | 1.5% (0.33%) |
| WarpX  | 2,000 | 1.6% (0.52%) | 34% | 2.7% (1.48%) |
| AI     | 1,804 | 2.0% (0.00%) | 592% | 1.2% (0.00%) |

Every stored chunk was checked against the bound (LAMMPS: decompression only)
and no quantized configuration was refused. A few chunks with large regret
pull the means up; the medians are the typical chunk.

Exploration is forced on every chunk (`--explore-k 31 --explore-thresh -1`), so
all 32 configurations are measured and the optimum is measured, not modelled.

## Run

    ./figure_8.sh -s 2k -w vpic --out <dir>        # smoke | full | 2k; -w repeatable
    FIG8_OUT=<dir> WL_TIMEOUT=900 sbatch --time=00:16:00 jobs/fig8.sbatch 2k vpic

`figure_8.sh`'s cleanup touches only its own Slurm job, so workloads can run
in parallel. Measured 2k walls on one A100, bound check included: VPIC 733 s,
LAMMPS 587 s, Nyx 362 s, WarpX 1,058 s, AI 501 s.
`--chunks-out` writes every chunk's pick, optimum, costs, regret and MAPE, and
the table above, to `full/logs/`, which git ignores along with the run logs.

| workload | data | 2k chunks |
|---|---|---|
| VPIC   | in situ, 126^3, 1,000 steps, dump every 8          | 2,000 |
| Nyx    | `np-nyx-256-2000-i48` (256^3, 2,000 steps, every 48) | ~2,064 |
| WarpX  | `np-warpx-1000-i5` (1,000 steps, every 5)          | 2,000 |
| LAMMPS | in situ, box 60, 1,000 steps, gap 3                | 2,004 |
| AI     | `np-ai-vitb16` (`ai/gen_fields.sh`, 11 checkpoints) | 1,804 |

## Plot

    python3 plot/fig8_trace.py <run>/<wl>/explore.csv --out <run>/trace.csv
    python3 plot/plot_fig8.py --trace <run>/trace.csv:<wl> ... --out figures/fig8/full \
        --chunks-out figures/fig8/full/logs

Defaults: 25-chunk bins, regret capped at 20%, MAPE at 50%.

## Model settings (each with a switch back to upstream)

| setting | default | upstream |
|---|---|---|
| ranking / cost time floor | 1 ms | same |
| prediction time floor | 1 ms | same (`CLIO_NEUROPRESS_PRED_TIME_FLOOR_MS`) |
| SGD ratio target cap | 100 | 10000 (`CLIO_NEUROPRESS_SGD_RATIO_TARGET_CAP`) |
| lossless training input | 1e-7 sentinel | raw bound (`CLIO_NEUROPRESS_TRAIN_RAW_BOUND=1`) |
| decompress label | codec kernel time | wall (`CLIO_NEUROPRESS_DT_LABEL=wall`) |
| explore SGD batch | 4 cheapest + 3 worst-predicted | cheapest (`CLIO_NEUROPRESS_EXPLORE_SGD_HARD=0`) |
| explore SGD dt labels | measured | 0 |
| trust region | all heads | heads with an error |

## Caveats

- Each configuration is timed once per chunk; one slow sample reads as a large
  regret. Report the median too. In this campaign all five workloads ran at
  once, and Nyx shared a node with LAMMPS.
- WarpX's E and j fields (1e7–1e12) are finer than float32 at eb 0.05, so their
  quantized configurations store most values bit-exact; they are quantized,
  not refused, but gain little over lossless.
- The same configuration has diverged and not diverged across runs; use two
  seeds before drawing stability conclusions.

## Panel (d) — the same two metrics for the baselines

`fig8d_models.png`. Panels (a)–(c) show NeuroPress's own model. This one asks
whether a *different* model would have done better, on the same chunks and the
same 32 candidates: one row per workload, one line per model, regret left and
cost MAPE right.

| line | what it is |
|---|---|
| NeuroPress (online) | the run's own predictions, online SGD active — the deployed system, and what panels (a)–(c) plot |
| NeuroPress (static) | the shipped `model.nnwt`, no online update |
| XGBoost | upstream's trained `xgb_model.pkl`, static |
| HCompress CCP (+fb) | HCompress's cost model (IPDPS 2020, Sec. IV-D), updated from measured outcomes during the run |
| HCompress CCP (seed) | the same fit, never updated |

The pick is the argmin of each model's own predicted cost under the **deployed
policy** (times floored at 1 ms, ratio capped at 100x), so it is the choice that
model would have made as the selector. For NeuroPress (online) the pick is the
configuration the run actually adopted; that agrees with its own argmin on **100%
of chunks in all five workloads**, which is what makes the counterfactual
definition fair to the others.

**This panel reports at the 1 ms floor, not panels (a)–(c)'s 5 ms.** A 5 ms floor
is the wrong instrument for comparing models on 8 MiB chunks: the cost's I/O term
is at most 1.7 ms, while `max(5,ct) + max(5,dt)` puts at least 10 ms of constant
underneath it, so every model's cost is pinned near 10 ms and the differences
between them fall into the third decimal. At 1 ms the same regrets are roughly
10x larger and the models separate. `--floor 5` reproduces the (a)–(c)
convention, and `fig8_models_floor5.csv` in the campaign's output directory is
that version.

    # from make_table.sh's prepared inputs -- no GPU, nothing re-measured
    python3 model-accuracy/fig8_model_chunks.py \
        --inputs /projects/bekn/imuradli/np-hcompress/out/inputs \
        --out    /projects/bekn/imuradli/np-hcompress/out/fig8_models.csv
    python3 plot/plot_fig8_models.py \
        --chunks /projects/bekn/imuradli/np-hcompress/out/fig8_models.csv \
        --out figures/fig8/full

Mean over the last 20% of chunks (`fig8d_models_summary.txt` has the full table,
including the first 200 chunks and the top-1 pick rate). The two top rows are
policies that use **no model at all**, both chosen with hindsight over the whole
run, and they are the thing to read every model against:

| regret | VPIC | Nyx | LAMMPS | WarpX | AI |
|---|---|---|---|---|---|
| *one fixed configuration, whole run* | *72.5%* | *13.5%* | *19.3%* | *0.4%* | *90.4%* |
| *best configuration per distribution class* | *69.5%* | *12.1%* | *19.3%* | *0.4%* | *85.1%* |
| NeuroPress (online) | 45.9% | 29.8% | **11.6%** | 12.5% | 176.8% |
| NeuroPress (static) | 35.8% | 162.0% | 32.3% | 37.4% | 161.3% |
| XGBoost | 96.3% | 59.3% | 22.3% | 20.5% | 247.0% |
| HCompress CCP (+fb) | **32.8%** | **21.3%** | 43.0% | 47.8% | 195.8% |
| HCompress CCP (seed) | 96.3% | 56.6% | 22.3% | 20.2% | 246.6% |

| cost MAPE | VPIC | Nyx | LAMMPS | WarpX | AI |
|---|---|---|---|---|---|
| NeuroPress (online) | 11.8% | 10.3% | **6.0%** | **9.2%** | **29.4%** |
| NeuroPress (static) | 9.0% | 48.8% | 14.0% | 31.1% | 27.5% |
| XGBoost | 1097% | 2324% | 1632% | 2035% | 1018% |
| HCompress CCP (+fb) | **8.1%** | 11.0% | 28.8% | 31.1% | 32.4% |
| HCompress CCP (seed) | 3616% | 4820% | 1935% | 4558% | 2862% |

Read it with these five facts:

- **A hindsight-fixed configuration beats both adaptive models on Nyx, WarpX and
  AI.** On WarpX one configuration is near-optimal on essentially every chunk
  (0.4% regret), so nothing a model does there can be read as evidence for
  modelling. Adding the distribution class on top of the fixed choice changes
  almost nothing (Nyx 13.5% → 12.1%), which is the same statement about
  HCompress's input set. Per-chunk modelling pays for itself on **VPIC** (45.9%
  against 72.5%) and **LAMMPS** (11.6% against 19.3%), and those are the two rows
  to argue from.
- **HCompress (+fb) is not uniformly close to us.** It wins on VPIC and Nyx and
  loses by 3.7x on LAMMPS and 3.8x on WarpX. The earlier impression that the two
  models were interchangeable was an artefact of the 5 ms floor.
- **The two static baselines barely choose at all.** HCompress (seed) picks the
  same configuration on every chunk of every workload — one distinct pick, five
  times out of five. XGBoost picks 1 to 3. Both settle on `bitcomp|q0|s0`
  (lossless, unshuffled) for all of VPIC and all of LAMMPS, which is why their
  regret lines coincide exactly on those two rows and only the upper one shows.
  NeuroPress (online) picks 5 to 17 distinct configurations, HCompress (+fb) 4 to 17.
- **What feedback fixes is a scale bias, not the ranking.** HCompress (seed)
  predicts compression times around 100x too large (144 ms against ~1 ms
  measured), because the profiler corpus's `compression_time_ms` is per-call
  overhead rather than codec time. One feedback round removes that bias, and it
  is the whole of the 3616% → 8.1% MAPE collapse on VPIC. Its *regret* barely
  moves on LAMMPS and WarpX, where the seed was already ranking about as well as
  the fitted model.
- **HCompress's feedback rows are NeuroPress's picks.** `--feedback-scope
  executed` feeds back the configuration the run adopted, which is the one
  NeuroPress chose, not the one HCompress would have chosen, so its updates land
  on good configurations selected by a good policy. That is the paper-faithful
  reading of "the measured cost of the executed choice", but it is an advantage,
  and `--feedback-scope all` changes the result by up to 10x.
- **These are not the same chunks as the table above.** Panel (d) is scored on
  the `acc-09171447` campaign (656/1056/603/500/656 chunks), which ran
  `figure_8.sh`'s exact model configuration plus `MEASURE_QUALITY=1` and the
  per-chunk distribution classifier that HCompress needs. The 2k campaign has
  neither, so its runs cannot score HCompress at all.
