# VPIC (`vpic-slabs`): producer + k-means consumer

Workload: `vpic-slabs`, 25.25 GiB, 6,464 chunks of 4 MiB (the `ez` field, float32).
The producer writes each chunk 1 time through Clio. A k-means consumer (8 clusters)
reads every chunk back into GPU memory 4 times: 1 time for each iteration.

Cost model (scenario A): cost = 1 x compress + 4 x decompress + 5 x stored bytes / 0.52 GB/s
(one PFS tier). With these weights the cost is the modelled time of 1 write + 4 reads.
Storage of the measured runs: local NVMe (not a PFS).

Methods:
- `fixed`: the best single codec = the setting with the lowest total 1/4/5 cost (`ans`, byte
  shuffle). All options select by the same cost; with 1/4/5 the cost is the modelled runtime
- `learn`: NeuroPress v2 with online learning; no exploration; no decompress on the write path
- `oracle`: each chunk with its own best setting by the cost-model value (from the stored
  exhaustive search)

## Figures

- `v2_vpic-slabs_kmeans_4reads.png`: measured application time, ratio, modelled PFS time
- `v2_vpic-slabs_kmeans_4reads_per_iteration.png`: each clustering iteration (read + k-means),
  running total, ratio

## Tables

- `v2_vpic-slabs_kmeans_4reads_compare.csv`: one row per method (`compare_kmeans_runs.py`)
- `whatif_vpic-slabs.csv`: the offline what-if values of the scenarios A to G (`costmodel_whatif.py`)

## Per-chunk data (`per_chunk/<method>/`)

- `blobs.csv.gz`: 1 row per chunk: selected codec, ratio, stored bytes, compress time
- `phases.csv.gz`: 1 write row and 5 read rows per chunk (the first 4 x 6,464 read rows
  are the timed k-means reads; the last is the untimed bit-exact check): time of each step
- `v2_measured.csv.gz`: the stored setting of each chunk, with its measured compress time,
  ratio and cost
- `learn/v2_pred.csv.gz`: the network's input features and its predicted compress time,
  decompress time and ratio for all 45 settings, for each chunk; `updates` = learning
  updates so far
- `oracle/oracle_map_w145_pfs.csv`: the oracle's setting for each chunk
- `stdout.log`: the run log (timings, k-means centroids of each iteration)
- `compose.yaml`: the Clio runtime configuration of the run

Run settings (`run_v2_workloads.sh`): `TIERS=0.52e6:1 COST_W=1,4,5 COST_BW=520000 READS=4 KMEANS=8`.
