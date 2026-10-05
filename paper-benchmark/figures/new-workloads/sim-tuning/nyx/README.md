# Nyx (`nyx-multiphase-50g`): producer + k-means consumer

Workload: `nyx-multiphase-50g`, 50.06 GiB, 12,816 chunks of 4 MiB (fields `density`, `rho_E`,
`logden`, float32). The producer writes each chunk 1 time through Clio. A k-means consumer
(8 clusters for each field) reads every chunk back into GPU memory 4 times: 1 time for each
iteration. Storage of the measured runs: local NVMe (not a PFS).

Methods (run tag `km4e`):
- `fixed`: the single codec `ans` byte shuffle. It is the fastest single codec by runtime, but it
  is NOT the best single codec under the 0/1/1 cost (that is `snappy` byte shuffle, which has
  no measured run). The figures label it as "single codec: ans byte" and give the % against it
- `learn`: NeuroPress v2 with online learning; it selects with the cost weights 0/1/1
  (compress 0, decompress 1, transfer 1) and one PFS tier at 0.52 GB/s; no exploration;
  no decompress on the write path
- `oracle`: each chunk with its own best setting by the 0/1/1 cost value (from the stored
  exhaustive search)

## Figures

The what-if figures were removed (user, 2026-10-05). Their values stay in the CSV files
`rank_options_nyx-multiphase-50g_w011.csv` and `_w145.csv` (`sim-tuning/rank_options_cost.py`
draws them again from the stored CSV files, with no run).
The measured-run figures were removed earlier: their single-codec run (`ans` byte) is not the
lowest 0/1/1 cost single codec, and `snappy` byte has no measured run. The measured runs' data
stay below.

## Tables

- `v2_nyx-multiphase-50g_kmeans_4reads_km4e_compare.csv`: one row per method (`compare_kmeans_runs.py`)

## Per-chunk data (`per_chunk/<method>/`)

- `blobs.csv.gz`: 1 row per chunk: selected codec, ratio, stored bytes, compress time
- `phases.csv.gz`: 1 write row and 5 read rows per chunk (the first 4 x 12,816 read rows are
  the timed k-means reads; the last is the untimed bit-exact check): time of each step
- `v2_measured.csv.gz`: the stored setting of each chunk, with its measured compress time,
  ratio and cost
- `learn/v2_pred.csv.gz`: the network's input features and its predicted compress time,
  decompress time and ratio for all 45 settings, for each chunk; `updates` = learning updates so far
- `oracle/oracle_map_w011_pfs.csv`: the oracle's setting for each chunk
- `stdout.log`: the run log (timings, k-means centroids of each field and iteration)
- `compose.yaml`: the Clio runtime configuration of the run

Run settings (`run_v2_workloads.sh`): `TIERS=0.52e6:1 COST_BW=520000 READS=4 KMEANS=8`,
with `COST_W=0,1,1` for the learning run.
