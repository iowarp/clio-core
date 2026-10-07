# Benchmark result archive (CSV only)

Copied from the run directories (`/mnt/nvme0/v2-work`) so the measurements are
not lost. Only CSV tables; files over 512 KiB are xz-compressed (`.csv.xz`;
`xz -d` restores them, pandas reads them directly). No simulation data and no
stored chunks.

- `exhaustive-baselines/<workload>/`: the exhaustive search of the Nyx, VPIC,
  WarpX and incflo workloads and their tuning probes (all 45 settings on every chunk): `v2_measured.csv`
  (compress / decompress time and ratio per chunk and setting), `v2_pred.csv`
  (model inputs and predictions per chunk), `blobs.csv`, `candidates.csv`,
  `phases.csv`. The best single codec, the oracle map and every "possible
  gain" are computed from these.
- `exhaustive-search-2026-10-07/<workload>/v2_measured.csv.xz`: the exhaustive search of
  Nyx, VPIC, WarpX and incflo made again on 2026-10-07 (ground-truth campaign,
  run_ground_truth.sh phase 1: whole dataset, 1 process, 1 chunk in flight, page cache dropped
  before the timed read, each workload's final cost model, every chunk bit-exact). One row per
  chunk and setting: `comp_ms`, `decomp_ms` (GPU kernel times), `ratio`, and `cost` under the
  run's cost model; any other cost model follows from these columns. Its ratios are identical
  to `exhaustive-baselines/` (deterministic workloads), its times agree within noise. Chunk
  sizes and order: `exhaustive-baselines/<workload>/v2_pred.csv.xz`.
- `sim-benchmark/<workload>/summary/`: the parallel k-means benchmark of Nyx
  (`nyx-multiphase-50g`), VPIC (`vpic-slabs`), WarpX
  (`ref-warpx-tune-final25g`), incflo (`ref-incflo-tune-final25g`) and incflo
  probe D: compare / walls CSVs per configuration, the configs CSV (with the
  model cost), the oracle map, model-vs-measured tables.
- `sim-benchmark/<workload>/runs/<run>/`: per run (`<ds>-p<i>_<option>_<tag>`)
  `blobs.csv` (stored size and codec per chunk), `v2_measured.csv` (the
  settings used), `phases.csv` (per-chunk time parts: io, kernel, other).
  `*_run1` is the first 2x8 run of incflo (repeated).
- `tuning/`: `np_cost_sweep_*.csv` (NeuroPress, oracle and best single over
  many cost models, offline).
