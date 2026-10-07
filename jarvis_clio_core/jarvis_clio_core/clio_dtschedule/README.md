# clio_dtschedule

DTSchedule policy layer for the CTE stack
(`jarvis_clio_core.clio_dtschedule`). Adds the `clio_cte_dtschedule`
chimod -- per-PutBlob codec, compression-location and tier decisions plus
a decision trace -- to the compose written by `jarvis_clio_core.clio_cte`.

## Where it sits

```
client / POSIX / FUSE / HDF5 / ADIOS2     CLIO_CTE_POOL=566.0 (exported here)
  -> dtschedule (566.0)                   this package
  -> cache (563.0) | indexer (564.0) | core (512.0)    = next_pool_id
  -> ... -> core (512.0)
```

dtschedule is the TOP of the interposition chain (DESIGN.md section 1): it
runs on the submitter's node before the cache's hash-routed write ships
the raw bytes. The package therefore

1. appends its entry to the clio_cte package's `cte_compose.yaml` (after
   every pool it may forward to), dropping a `clio_cte_compressor` entry
   if `iowarp_compress` was on (dtschedule replaces the compressor);
2. resolves `next_pool_id` as the cache pool if the compose has one, else
   the indexer, else `clio_cte_core`, else 512.0 -- no existing entry is
   re-pointed;
3. exports `CLIO_CTE_POOL=<pool_id>` into the pipeline environment.
   `ClientInit` (`context-transfer-engine/core/src/content_transfer_engine.cc`)
   reads it and binds every adapter's CTE client to that pool, so the
   POSIX interposer, FUSE, HDF5 and ADIOS2 adapters enter the chain at
   dtschedule.

`start`/`stop` are no-ops: the clio_cte package's `clio_run compose start`
creates the whole chain. If no clio_cte package precedes this one, the
entry is written to this package's own `dtschedule_compose.yaml` and
composed from `start` instead.

Put the package AFTER `clio_cte` and BEFORE the application in the
pipeline. `dag_path` is derived from a `clio_wfcommons_dist` package
anywhere in the pipeline (its `dag_out`, default
`<shared_dir>/placement.json`), or from `DTSCHEDULE_DAG_PATH` in the env.

## Knobs (DESIGN.md section 7)

| key | default | compose key |
|---|---|---|
| `pool_name` | `clio_cte_dtschedule` | `pool_name` |
| `pool_id` | 566.0 | `pool_id`; also `CLIO_CTE_POOL` |
| `next_pool_id` | auto (cache > indexer > core > 512.0) | `next_pool_id` |
| `pool_query` | `local` | `pool_query` |
| `qos_objective` | `performance` | `qos.objective` (`performance` / `ratio`) |
| `qos_max_error` | 1e-3 | `qos.max_error` (0 = lossless only) |
| `qos_lossy_allowlist` | `['.*\.(fits\|nc\|h5\|bp\|sac\|dat)$']` | `qos.lossy_allowlist` |
| `qos_compression_preference` | `[]` | `qos.compression_preference` |
| `qos_resample_error` | 0.4 | `qos.resample_error` |
| `qos_resample_chance` | 1.0 | `qos.resample_chance` |
| `qos_stages` | `[]` | `qos.stages` (list of `{match: regex, <qos key>: value}`) |
| `ccm` | `qtable` | `ccm` (`qtable` / `ema` / `oracle` / `fixed:<lib>[:<preset>]`) |
| `qtable_model_path` | `<clio-core>/context-transfer-engine/dtschedule/models/qtable_v1` | `qtable_model_path` (directory) |
| `ratio_noise_sigma` | 0.0 | `ratio_noise_sigma` |
| `load_aware` | true | `load_aware` |
| `load_period_ms` | 1000 | `load_period_ms` |
| `load_cap` | 4.0 | `load_cap` |
| `workflow_aware` | `consumer` | `workflow_aware` (`none` / `consumer` / `dag`) |
| `dag_path` | derived (see above) | `dag_path` |
| `dag_colocate_fanin` | true | `dag.colocate_fanin` |
| `dag_replicate_fanout_min` | 4 | `dag.replicate_fanout_min` |
| `dag_replicate_max` | 8 | `dag.replicate_max` |
| `force_scenario` | `auto` | `force_scenario` (`auto` / `1` / `2` / `3`) |
| `decision_order` | `joint` | `decision_order` (`joint` / `codec_first` / `tier_first`) |
| `tiers` | derived from clio_cte `devices` (`ram::` -> ram, `nvme`, `ssd`, `hdd`, `s3`, else `nfs`) | `tiers` (name -> score) |
| `net_bw_gbps` | 25 | `net_bw_gbps` |
| `trace_path` | `${HOME}/jarvis-runs/<pipeline>/dtschedule_trace` (`off` disables) | `trace_path` (chimod appends `.<container>.csv`) |
| `trace_candidates` | false | `trace_candidates` (`<trace_path>.cand.<node>.csv`) |
| `min_compress_bytes` | 4096 | `min_compress_bytes` |

Every configure deletes the previous trace files of the same
`trace_path`, so a pipeline test that reconfigures the pipeline per
combination gets a fresh trace each row. `clean` removes the trace files
and the package's compose copy.

## Stats (`jarvis ppl stat`, pipeline-test `results.csv`)

Parsed from the decision trace (`<trace_path>.<node>.csv`: DESIGN.md
section 9's 26 columns followed by `obs_dtime_ms,select_ms`; a row with a
non-empty `obs_dtime_ms` is a GetBlob row, every other row a PutBlob
decision) and the candidate trace (`<trace_path>.cand.<node>.csv`:
`ts_ms,node,tag,blob,lib,preset,pred_ctime_ms,pred_dtime_ms,pred_ratio,cost_ms,reason`):

| stat | meaning |
|---|---|
| `<pkg>.puts` | PutBlob decisions (compressed + raw) |
| `<pkg>.compressed` | decisions whose `chosen_lib` is a real codec |
| `<pkg>.raw` | decisions stored uncompressed (`chosen_lib` = `raw`, `obs_ratio` = 1) |
| `<pkg>.bytes_in` | sum of `size` over decisions |
| `<pkg>.gets` | GetBlob (decompression) rows |
| `<pkg>.mean_ratio` | mean `obs_ratio` (original / compressed, >= 1) over decisions |
| `<pkg>.mean_obs_ctime_ms` / `.mean_obs_dtime_ms` | mean compress / decompress time (ms) |
| `<pkg>.mean_select_ms` | mean decision (selection) time (ms) |
| `<pkg>.candidates_per_decision` | mean candidate rows per (node, tag, blob) in the candidate trace; absent without `trace_candidates` |
| `<pkg>.lib.<codec>` | per-library histogram (compressed decisions) |
| `<pkg>.scenario.<1\|2\|3>` | per-scenario histogram |
| `<pkg>.tier.<name>` | per-tier histogram |
| `<pkg>.trace_files` | number of decision trace files (one per container) |
| `<pkg>.ccm`, `.workflow_aware`, `.load_aware`, `.force_scenario`, `.decision_order`, `.qos_objective`, `.qos_max_error`, `.ratio_noise_sigma` | the knobs of the run |

Pipelines and experiment sweeps: `pipelines/ares/dtschedule/README.md`.
