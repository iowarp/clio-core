# Figure 8 — regret and cost MAPE

Does NeuroPress's online model improve over a run on data it never trained on?
Per chunk:

    regret (%)    = (measured cost of the pick / cheapest measured cost - 1) x 100
    cost MAPE (%) = |predicted cost of the pick - measured cost| / measured x 100
    cost          = max(F, ct) + max(F, dt) + bytes / (min(ratio, 100) * 5e6 B/ms)

`plot_fig8.py trace` scores both at F = 5 ms on predicted and measured alike, as
upstream's trace `mape_cost` does. Upstream's regret uses its unfloored
`real_cost` instead (`real_cost_raw` in the trace), which gives much larger
numbers on these workloads because every codec time is under 5 ms; `plot_fig8.py panels`
reports that column too.

Exploration is forced on every chunk (`--explore-k 31 --explore-thresh -1`), so
all 32 configurations are measured and the optimum is measured, not modelled.

**This directory holds no figures, traces or summaries.** Panels, `trace.csv` and
the summary tables are what a run produces, not source, so only the scripts and
this description are tracked. Every command below writes its own output; nothing
here needs to be in git for the figure to be rebuilt. Past artefacts are in the
history (`git log -- paper-benchmark/figures/fig8`).

## Run

    ./figure_8.sh -s 2k -w vpic --out <dir>        # smoke | full | 2k; -w repeatable
    FIG8_OUT=<dir> WL_TIMEOUT=1200 sbatch --time=00:25:00 jobs/fig8.sbatch 2k vpic

`figure_8.sh`'s cleanup touches only its own Slurm job, so workloads can run in
parallel. Plots land in `<size>/`, beside the script; `--chunks-out` additionally writes
every chunk's pick, optimum, costs, regret and MAPE to `<size>/logs/`, which git
ignores. `WL_TIMEOUT` has to clear the slowest workload — on one A100, with the
bound check included, measured 2k walls were VPIC 733 s, LAMMPS 587 s, Nyx 362 s,
WarpX 1,058 s, AI 501 s.

`--size` selects only `--max-files`; it does **not** change a dump's cadence.
Each workload reads its `--fields` directory as generated, so a size is only as
long as the dumps behind it: `-s 2k` against a `full`-cadence dump silently
yields a ~1,000-chunk run. Regenerate the fields at the intended cadence first.

| workload | data | 2k chunks |
|---|---|---|
| VPIC   | 126^3, 1,000 steps, dump every 8                   | 2,000 |
| Nyx    | `np-nyx-256-2000-i48` (256^3, 2,000 steps, every 48) | ~2,064 |
| WarpX  | `np-warpx-1000-i5` (1,000 steps, every 5)          | 2,000 |
| LAMMPS | box 60, 1,000 steps, gap 3                         | 2,004 |
| AI     | `np-ai-vitb16` (`ai/gen_fields.sh`, 11 checkpoints) | 1,804 |

## Plot

    ./plot_fig8.py trace <run>/<wl>/explore.csv --out <run>/trace.csv
    ./plot_fig8.py panels --trace <run>/trace.csv:<wl> ... --out <size> \
        --chunks-out figures/fig8/<size>/logs

Defaults: 25-chunk bins, regret capped at 20%, MAPE at 50%. `plot_fig8.py panels`
prints a per-workload summary — chunks, regret over the first 200, regret and
cost MAPE over the last 20%, and the unfloored regret — which is the table to
quote from a run.

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
  regret. Report the median alongside the mean, and say whether workloads shared
  a node.
- WarpX's E and j fields (1e7–1e12) are finer than float32 at eb 0.05, so their
  quantized configurations store most values bit-exact; they are quantized, not
  refused, but gain little over lossless.
- The same configuration has diverged and not diverged across runs; use two
  seeds before drawing stability conclusions.
- Check the bound verdict per workload before plotting. `figure_8.sh` refuses to
  plot a workload whose run printed `FAILED`, but a `(no verification line)` is
  not a pass.

## Panel (d) — the same two metrics for the baselines

Panels (a)–(c) show NeuroPress's own model. Panel (d) asks whether a *different*
model would have done better, on the same chunks and the same 32 candidates: one
row per workload, one line per model, regret left and cost MAPE right.

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
configuration the run actually adopted; `fig8_model_chunks.py` reports how often
that agrees with its own argmin, and the counterfactual is only fair to the
others while that agreement is near-total.

**Panel (d) reports at the 1 ms floor, not panels (a)–(c)'s 5 ms.** A 5 ms floor
is the wrong instrument for comparing models on 8 MiB chunks: the cost's I/O term
is at most 1.7 ms, while `max(5,ct) + max(5,dt)` puts at least 10 ms of constant
underneath it, so every model's cost is pinned near 10 ms and the differences
between them fall into the third decimal. At 1 ms the same regrets are roughly
10x larger and the models separate. `--floor 5` reproduces the (a)–(c) convention.

Read the result against the policies that use **no model at all** — one fixed
configuration for the whole run, and the best configuration per distribution
class, both chosen with hindsight. Those two rows come from
`model-accuracy/accuracy_table.py` (`make_table.sh --constant-as`), not from
`fig8_model_chunks.py`, so they have to be put beside the summary by hand.

    # from make_table.sh's prepared inputs -- no GPU, nothing re-measured
    python3 model-accuracy/fig8_model_chunks.py \
        --inputs <campaign>/inputs --out <campaign>/fig8_models.csv
    ./plot_fig8.py models \
        --chunks <campaign>/fig8_models.csv --out figures/fig8/<size> \
        --name fig8d_models --bin 25 --cap 100

`fig8_model_chunks.py` breaks argmin ties by a draw from a per-(workload, model)
stream, so swapping one model's file cannot move another's line; it drops chunks
whose adopted configuration was not measured on all 32 settings, and writes its
own log beside the CSV. `plot_fig8.py models` clips per-chunk regret at `--cap`
before binning, and writes `<name>_summary.txt` — means are uncapped there, the
two `<=cap` columns clip each chunk first, as the figure does.

Two things to state whenever panel (d)'s numbers are quoted:

- **HCompress's feedback scope.** `--feedback-scope self` is the default: the
  model observes only the configuration *it* would have chosen. `executed` feeds
  back the configuration NeuroPress adopted instead, which makes HCompress a
  passenger on our selector rather than a system of its own, and changes the
  result by up to 10x. `all` is the third option. Say which one was used.
- **Panel (d) cannot be scored from a `figure_8.sh` run.** It needs
  `MEASURE_QUALITY=1` and the per-chunk distribution classifier that HCompress
  takes as its only data-dependent input; `figure_8.sh` sets `MEASURE_QUALITY=0`
  and runs no classifier. Use `model-accuracy/run_campaign.sh`, which enables
  both, and note that its exploration window (`EXPLORE_K`, default 8) is a ranked
  window rather than (a)–(c)'s forced 31.

`model-accuracy/collect_fig8_full.sh <tag>` gathers the scored CSV, the
per-workload measurements, every log, the raw selector traces and frozen copies
of the whole script chain into one bundle outside the repo, so the figure can be
rebuilt from CSV alone without re-measuring anything.
