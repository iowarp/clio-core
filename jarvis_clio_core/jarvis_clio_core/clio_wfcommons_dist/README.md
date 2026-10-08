# clio_wfcommons_dist

Distributed WfCommons/WfBench workflow runner for DTSchedule evaluations
(`jarvis_clio_core.clio_wfcommons_dist`).

Compared with `builtin.wfcommons`:

| | `builtin.wfcommons` | `clio_wfcommons_dist` |
|---|---|---|
| task data | `os.urandom` (incompressible) | `dt_datagen`: float fields / FASTQ / VCF-like / mixed, keyed on the recipe's file extensions, seeded per file |
| file names | `<task>_outfile_0001` (no extension) | `<task>_outfile_0001.fits` etc. |
| execution | whole DAG on every node, sequential `BashTranslator` | one assignment of tasks to nodes, level by level over ssh (`run_dist.py`) |
| DAG spec | none | `placement.json` (`dag_path` for the `dtschedule` chimod) |

The tools live in `context-transfer-engine/dtschedule/tools/wfcommons/`
(`dt_datagen.py`, `dt_bench.py`, `dt_translator.py`, `run_dist.py`,
`wfbench_dt`; see `PATCH_NOTES.md` there for the exact diff against
wfcommons 1.2).

## Knobs

| key | default | meaning |
|---|---|---|
| `recipe` | `montage` | WfChef recipe (`montage`, `seismology`, `genome`, `epigenomics`, `cycles`, `srasearch`, `blast`, `bwa`, `soykb`, `rnaseq`) |
| `num_tasks` | 60 | DAG size (WfChef refuses fewer than the recipe's base graph; montage min is 60) |
| `data_footprint` | `0` | total bytes over all files (`30G`); every file gets `footprint / num_files` |
| `cpu_work` | 1 | must be > 0 or wfbench does no I/O |
| `percent_cpu` | 1.0 | cpu/mem thread split; < 1.0 needs `stress-ng` |
| `drop_page_cache` | false | `WFBENCH_DROP_CACHE=1`: fsync + fadvise(DONTNEED) after each I/O |
| `clio_prefix` | false | prefix task paths with `clio::` for the CTE POSIX interposer |
| `placement` | `stage_sets` | `round_robin` / `stage_sets` / `single` |
| `stage_sets` | 2 | number of disjoint node subsets; level *i* runs on subset *i mod k* |
| `data_class` | `auto` | force one `dt_datagen` class for every file |
| `data_noise` | 0.05 | noise amplitude of `float_field` data (`DT_DATA_NOISE`) |
| `wf_python` | `${HOME}/venv-dtschedule/bin/python` | interpreter with wfcommons 1.2 (created by `ensure_venv.sh` if missing) |
| `tools_dir` | derived | override the tools directory |
| `out` | `${HOME}/dtschedule-runs/wfcommons` | shared-FS output dir |
| `dag_out` | `<shared_dir>/placement.json` | where `placement.json` is copied; also exported as `DTSCHEDULE_DAG_PATH` |
| `ssh_cmd` | `ssh` | ssh prefix for `run_dist.py` (use `env -u LD_LIBRARY_PATH ssh` on ares) |

## Lifecycle

* `_configure`: ensures the venv, writes `<shared_dir>/hostfile.txt`
  from the pipeline hostfile, runs `dt_translator.py` on the head node
  (generates the benchmark JSON, input files, `placement.json`,
  `levels/L<i>/<node>.sh`, `run_dist.py`, `bin/`), copies
  `placement.json` to `dag_out`.
* `start`: `wf_python <out>/run_dist.py --out <out> --ssh-cmd ...` with
  the interceptor environment (`interceptors: [cte_posix]` puts
  `libclio_cte_posix.so` into `LD_PRELOAD`; `run_dist.py` forwards it to
  the task scripts through `env.sh` while keeping it out of ssh itself).
* `stop`: `kill_stragglers.sh` on every node (`wfbench_dt`, `cpu-benchmark`).
* `clean`: removes `out` on every node and the copied `placement.json`.
* `_get_stat`: `makespan_ms`, `status`, `level<i>_ms` from `makespan.json`.

## Output layout (`out`)

```
bench/<recipe>-benchmark-<N>.json   WfFormat instance (+ generated inputs)
data/                               workflow input files; task outputs land here
levels/L<i>/<node>.sh               one script per (level, node)
placement.json                      DAG spec: nodes, tasks{node,level,inputs,outputs}, files{producer,consumers,consumer_nodes}
run_dist.py, bin/, env.sh, logs/    driver, wfbench_dt + cpu-benchmark, task logs
makespan.json                       per-level and total wall time (ms)
```

## Example

See `jarvis_clio_core/pipelines/ares/dtschedule/montage_2n.yaml`.

## CTE-API mode (`api: true`)

Runs the same WfChef instance with `dtschedule_wfrun`
(`context-transfer-engine/dtschedule/tools/wfrun`) instead of WfBench: files
are CTE tags `<out basename>__<file>` of 1 MiB blobs put/get through the
dtschedule pool (the prodcons `pc_cte_api.cc` backend), so clio-fs's per-file
home-node routing cannot override dtschedule's placement. `_configure` runs
`dt_translator.py --api`, which writes only `placement.json` (plus per-task
instance `runtime`/`cores`, staged inputs' `producer_node`); `start` launches
one mpirun of `ppn` ranks per host (same launcher as `clio_prodcons`).

| key | default | meaning |
|---|---|---|
| `api` | false | use dtschedule_wfrun over the CTE API |
| `seed` | -1 | seed WfChef (and PYTHONHASHSEED) so every configure builds the same instance |
| `size_mode` | `instance` | file sizes proportional to the instance's (same total footprint) or `uniform` |
| `min_file_kb` / `max_file_mb` | 1024 / 256 | size clamps (max also caps wfrun) |
| `runtime_scale` | 0.01 | CPU-bound compute seconds per instance runtime second |
| `size_scale` | 1.0 | run-time factor on every file size |
| `ppn` | 8 | ranks per node; a node's tasks are dealt round-robin over them |
| `write_pending` | 1 | output files left in flight per rank (0 = wait until stored) |
| `input_mem_mb` | 1024 | inputs held at once; wider fan-ins are read/computed in batches |
| `payload` | `` | directory of real files to cut outputs from (default: generated by data class) |
| `mpirun`, `launch_agent`, `bin_dir`, `net_if`, `timeout_s` | | as in `clio_prodcons` |

Outputs in `out`: `placement.json`, `wfrun.log`, `tasks.csv` (per task:
start, wait/read/compute/gen/write seconds, end, bytes), `makespan.json`
(`total_ms` = workflow wall time after staging, `levels`, `stage_ms`,
read/write GB). Offline check without a runtime:
`build/bin/dtschedule_wfrun --dry-run --placement <out>/placement.json
--runtime-scale X --ppn 8`. Pipelines: `pipelines/ares/dtschedule/wf_*_api.yaml`.
