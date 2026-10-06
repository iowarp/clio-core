# Benchmark result archive (CSV only)

Copied from the run directories (`/mnt/nvme0/v2-work`) so the measurements are
not lost. Only CSV tables; files over 512 KiB are xz-compressed (`.csv.xz`;
`xz -d` restores them, pandas reads them directly). No simulation data and no
stored chunks.

- `exhaustive-baselines/<workload>/`: the exhaustive search of every workload
  and tuning probe (all 45 settings on every chunk): `v2_measured.csv`
  (compress / decompress time and ratio per chunk and setting), `v2_pred.csv`
  (model inputs and predictions per chunk), `blobs.csv`, `candidates.csv`,
  `phases.csv`. The best single codec, the oracle map and every "possible
  gain" are computed from these.
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
