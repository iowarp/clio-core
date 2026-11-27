# SSC / Accio campaign results (CSV)

Exported from the JSONL results on Delta (`ssc_work/results`) with
`env -u PYTHONPATH python3 ../export_csv.py <results dir> data`. Snapshot of 2026-11-27:
1 GB and 0.5 GB grids complete for the 11 WfCommons phases; 0.5 GB + parallel FS partial;
16 GB large-scale, live WfCommons runs and the remaining 0.5 GB units still pending.

| file | one row per |
|---|---|
| `grid.csv` | measured cell: node setup, phase, placement/cache/indexer/compression, repeat, times |
| `decisions.csv` | offline decision: scenario, controller, schedule (seed), window (step), choice, slowdown vs. best |
| `summary.csv` | scenario x controller: accuracy (% within 25% of best), geomean slowdown, decision time |
| `learning.csv` | learning-curve decision, with the subset of measured phases used as experience |
| `live.csv` | live-run window (earlier campaigns; JSON row per window) |

`app = wf` is the WfCommons campaign used in the paper. Slowdowns use the paper's metric
(`paper_plots.slowdown`); `summary.csv` is a plain geomean over all decisions, so the last digit can
differ from the paper's tables.
