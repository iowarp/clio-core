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

| 2k campaign `twok-upstream-clamps` | chunks | regret, last 20%, F = 5 ms | same, unfloored | cost MAPE, last 20% |
|---|---|---|---|---|
| VPIC   | 2,000 | 5.9% | 571% | 3.5% |
| Nyx    | 2,064 | 5.0% | 337% | 2.9% |
| LAMMPS | 2,004 | 1.8% | 105% | 1.2% |
| WarpX  | 1,877 (timed out) | 2.3% | 38% | 3.6% |
| AI     | 1,804 (bound check cut off) | 0.9% | 502% | 0.7% |

Exploration is forced on every chunk (`--explore-k 31 --explore-thresh -1`), so
all 32 configurations are measured and the optimum is measured, not modelled.

## Run

    ./figure_8.sh -s 2k -w vpic --out <dir>        # smoke | full | 2k; -w repeatable
    FIG8_OUT=<dir> WL_TIMEOUT=900 sbatch --time=00:16:00 jobs/fig8.sbatch 2k vpic

`figure_8.sh`'s cleanup touches only its own Slurm job.
Measured walls on one A100: VPIC 2k ~770 s, LAMMPS 2k ~550 s, Nyx 2k ~360 s,
WarpX 2k >900 s, AI 2k ~490 s before its bound check.
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
  regret. Report the median too.
- WarpX's E and j fields (1e7–1e12) exceed the quantizer's int32 grid at eb 0.05,
  so in these runs its "quantized" actions often ran lossless. The quantizer
  now quantizes them, storing the values the grid cannot hold bit-exact.
- The same configuration has diverged and not diverged across runs; use two
  seeds before drawing stability conclusions.
