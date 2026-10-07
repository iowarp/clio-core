# DTSchedule pipelines (ares)

Jarvis pipelines that run WfCommons workflows as DTSchedule evaluation
workloads (EVAL_PLAN.md section 3) through the `clio_dtschedule` policy
layer. Each one brings up the CLIO runtime and a two-tier CTE on every
node, puts dtschedule on top of the CTE chain, then runs a WfBench
workflow whose tasks are spread across nodes so inter-stage files cross
the network.

| pipeline | recipe | shape | data classes |
|---|---|---|---|
| `montage_2n.yaml` | montage, 60 tasks | fan-out -> fan-in -> fan-out -> fan-in (8 levels) | `.fits` float_field, `.txt`/`.tbl` text_table, `.jpg` mixed |
| `seismology_2n.yaml` | seismology, 103 tasks | wide fan-in (2 levels) | `.stf` float_field, `.gz` mixed_binary |

## Stack

```
clio_runtime  (shm IPC, swim off, every node)
clio_cte      devices: ram::cte_ram_tier1 16GB score 1.0
                       /mnt/nvme/${USER}/<name>/cte_target.bin 64GB score 0.7
clio_dtschedule   appended on top of the CTE compose chain:
                  clio_cte_dtschedule 566.0 -> 512.0, CLIO_CTE_POOL=566.0
clio_wfcommons_dist  + interceptors: [cte_posix]  (LD_PRELOAD libclio_cte_posix.so)
```

Generated compose (`<shared_dir>/cte_core/cte_compose.yaml`, from
`montage_2n.yaml`):

```yaml
compose:
- mod_name: clio_cte_core          # pool 512.0, ram 1.0 + nvme 0.7
  ...
- mod_name: clio_cte_dtschedule
  pool_name: clio_cte_dtschedule
  pool_query: local
  pool_id: '566.0'
  next_pool_id: '512.0'            # cache 563.0 / indexer 564.0 when composed
  qos: {objective: performance, max_error: 0.001, lossy_allowlist: [...], ...}
  ccm: qtable
  qtable_model_path: <clio-core>/context-transfer-engine/dtschedule/models/qtable_v1
  workflow_aware: dag
  dag_path: <shared_dir>/wfcommons/placement.json
  dag: {colocate_fanin: true, replicate_fanout_min: 4, replicate_max: 8}
  force_scenario: auto
  decision_order: joint
  tiers: {ram: 1.0, nvme: 0.7}     # derived from the clio_cte devices
  net_bw_gbps: 40.0
  trace_path: ${HOME}/jarvis-runs/<pipeline>/dtschedule_trace
  trace_candidates: true
  min_compress_bytes: 4096
```

`pool_query: local` registers every pool on every node before any task
runs (the same multi-node pattern as `pipelines/ares/distributed.yaml`).
The POSIX interposer binds to the dtschedule pool through `CLIO_CTE_POOL`,
which `clio_dtschedule` exports into the pipeline environment. The nvme
path follows the tiering pipelines' convention
(`/mnt/nvme/<user>/<pipeline>/cte_target.bin`); the slurm `pre_cmds`
create the directory. Nothing is written to `/tmp`.

## Experiments

Test YAMLs (`config:` + `vars:`/`loop:`) are jarvis pipeline tests: one
`jarvis ppl submit <abs path>/<file>` wraps every row in one allocation
and writes `results.csv` under `output:`. `sweep.py` drives the same rows
(or ad-hoc `pkg.key=value` overrides) from inside an existing allocation.

| experiment | pipeline | knobs swept | figure script (`dtschedule/eval/`) |
|---|---|---|---|
| E5 scenario ablation | `e5_scenario_ablation.yaml` (montage) | `dtschedule.force_scenario` in {auto, 1, 2, 3}; load / network grid injected outside jarvis (below), one submit per cell | `e5_scenario_heatmap.py` |
| E6 component ablation | `e6_component_ablation.yaml` (montage) | zipped leave-one-out rows over `ccm` (qtable vs `fixed:zstd:balanced`), `load_aware`, `workflow_aware` (dag / consumer / none), `qos_max_error` (1e-3 / 0) | `e6_stacked.py` |
| E7 QoS sweep | `e7_qos_sweep.yaml` (seismology) | `qos_max_error` in {0, 1e-2, 1e-3, 1e-4, 1e-5} + "no QoS" (`1.0`, allowlist `.*`) + per-stage row (`qos_stages`), x `qos_objective` in {performance, ratio} | `e7_qos.py` |
| E10 load-response timeline | `e10_load_timeline.yaml` (seismology, 60 GiB) | none; `load_period_ms: 500`, load steps from `load_steps.sh` | `e10_timeline.py` |
| E1 joint vs sequential | `sweep.py --pipeline montage_2n.yaml --run joint:dtschedule.decision_order=joint --run codec:dtschedule.decision_order=codec_first --run tier:dtschedule.decision_order=tier_first` | `decision_order`; offline replay from `<trace_path>.cand.csv` | `e1_flips.py` |
| E9 prediction-error sensitivity | `sweep.py --pipeline montage_2n.yaml --run s0:dtschedule.ratio_noise_sigma=0 --run s025:dtschedule.ratio_noise_sigma=0.25 ...` | `ratio_noise_sigma` in {0, 0.1, 0.25, 0.5, 1.0} | `e9_noise.py` |
| E4 DAG-aware placement | `sweep.py --pipeline montage_2n.yaml --run none:dtschedule.workflow_aware=none --run consumer:dtschedule.workflow_aware=consumer --run dag:dtschedule.workflow_aware=dag --run dagrep:dtschedule.workflow_aware=dag,dtschedule.dag_replicate_fanout_min=2` | `workflow_aware`, `dag_replicate_fanout_min`, `dag_colocate_fanin` | `e4_dag.py` |
| baseline workflows | `montage_2n.yaml`, `seismology_2n.yaml` | none (full system) | -- |

Figure scripts read the decision trace
(`${HOME}/jarvis-runs/<pipeline>/dtschedule_trace.<container>.csv`, DESIGN.md
section 9), the sweep JSON / `results.csv`, and the workflow's
`makespan.json`; they are written later.

### sweep.py

```
python3 sweep.py --exp e5 --from-test e5_scenario_ablation.yaml --hostfile ${HOME}/hostfile.txt
python3 sweep.py --exp e9 --pipeline montage_2n.yaml \
    --run s0:dtschedule.ratio_noise_sigma=0 --run s05:dtschedule.ratio_noise_sigma=0.5
python3 sweep.py --exp e7 --from-test e7_qos_sweep.yaml --dry-run
```

Per run it writes `<results>/<exp>/yaml/<run>.yaml` (the pipeline with the
`scheduler:` block stripped, `name: <exp>_<run>`, `hostfile:` from
`--hostfile`, overrides applied), runs `jarvis ppl load yaml` +
`jarvis ppl run`, then stores every package's `_get_stat` output and the
wfcommons `makespan.json` in
`${HOME}/jarvis-runs/dtschedule-results/<exp>/<run>.json` and copies the
run's trace CSVs to `<exp>/traces/<run>/` before the next run's configure
resets them (`--results-root` to move both). Override values are parsed as YAML; quote
values that must stay strings (`dtschedule.force_scenario="'3'"`; the
package also accepts the integer). `--skip-existing` resumes a sweep,
`--destroy` removes each run's pipeline after collecting, `--dry-run`
only writes the YAMLs. Run it from inside your own `salloc` allocation
(the slurm `pre_cmds` do not run, so create `/mnt/nvme/${USER}/<name>`
first).

### Load and network injection (E5 grid, E10 steps)

No builtin jarvis package wraps `stress-ng` or `tc`
(`/mnt/common/llogan/jarvis-cd/builtin/builtin`, checked 2026-10-06;
`builtin.wfcommons` only spawns stress-ng memory threads for
`percent_cpu < 1`). Inject by hand on the allocated nodes:

```
# consumer / producer CPU load (E5 cells, E10 steps): all cores, N seconds
ssh <host> stress-ng --cpu 0 --timeout 120s --quiet
# 50 % of the cores
ssh <host> stress-ng --cpu 20 --timeout 120s --quiet
# release early
ssh <host> pkill stress-ng
# network throttle to 5 GbE on the 40 GbE NIC (needs root; ask for a
# tc-capable reservation): tc qdisc add dev <nic> root tbf rate 5gbit burst 32kbit latency 400ms
```

`load_steps.sh` runs the E10 timeline (consumer at t=120 s for 120 s,
producer at t=300 s for 120 s) and backgrounds itself; start it when the
workflow starts, or hook it in through `builtin.my_shell` as shown in
`e10_load_timeline.yaml`. The trace's `producer_cpu` / `consumer_cpu`
columns record what dtschedule saw.

## Knobs that matter

* `clio_dtschedule` (see `jarvis_clio_core/clio_dtschedule/README.md`):
  `ccm`, `workflow_aware`, `load_aware`, `force_scenario`,
  `decision_order`, `qos_*`, `ratio_noise_sigma`, `trace_path`. Defaults
  that are derived rather than typed: `next_pool_id` (cache > indexer >
  core), `tiers` (from the CTE devices), `dag_path` (from the wfcommons
  package), `qtable_model_path` (in-tree `models/qtable_v1`).
* `placement` / `stage_sets` (on `clio_wfcommons_dist`):
  `stage_sets: 2` puts level *i* on node subset *i mod 2*, so with two
  nodes consecutive levels alternate nodes and every inter-level file is
  produced on one node and consumed on the other. `round_robin` spreads
  each level over both nodes (roughly half the files cross nodes);
  `single` is the one-node baseline.
* `data_footprint`: total bytes over all files. Keep it above the RAM
  tier (16 GB) when the experiment is about tiering, below it when it is
  about codec choice only.
* `data_class` / `data_noise`: `auto` picks the class from the file
  extension (`dt_datagen.EXT_TABLE`); `data_noise` scales the Gaussian
  noise on float fields (0.0 = very compressible, 0.05 default gives
  ~2.6x zstd, 0.3+ is nearly incompressible losslessly).
* `clio_prefix` + `interceptors: [cte_posix]`: both are needed for task
  I/O to reach the CTE. Without `clio_prefix` the workflow runs on the
  shared filesystem (NFS baseline) even with the interposer loaded.
* `drop_page_cache` (and `WFBENCH_DROP_CACHE` in `env:`): fsync +
  fadvise(DONTNEED) after each I/O so the NFS baseline cannot serve reads
  from the page cache.
* `cpu_work`: must stay > 0 (wfbench gates its I/O on cpu-benchmark
  progress). `percent_cpu: 1.0` avoids spawning `stress-ng` memory
  threads, which is not installed everywhere and which the evaluation
  uses separately for load injection.
* `ssh_cmd`: `env -u LD_LIBRARY_PATH ssh` on ares (a managed env's
  LD_LIBRARY_PATH breaks the system ssh).
* `wf_python`: a venv with `wfcommons==1.2` and `numpy`; created on first
  configure if missing (`clio_wfcommons_dist/ensure_venv.sh`).

## Outputs

* `${HOME}/jarvis-runs/<pipeline>/dtschedule_trace.<container>.csv` -- the
  decision trace (plus `.cand.csv` with `trace_candidates`); summarised
  by `jarvis ppl stat` as `dtschedule.puts`, `.compressed`,
  `.mean_ratio`, `.mean_obs_ctime_ms`, `.mean_obs_dtime_ms`,
  `.mean_select_ms`, `.raw`, `.candidates_per_decision`, `.lib.<codec>`,
  `.scenario.<k>`, `.tier.<name>`. Reset at every configure.
* `${HOME}/dtschedule-runs/<name>/makespan.json` -- per-level and total
  wall time in ms (also in `jarvis ppl stat` as `wfcommons.makespan_ms`,
  `wfcommons.level<i>_ms`).
* `${HOME}/dtschedule-runs/<name>/placement.json` -- the DAG spec; the
  `dtschedule` chimod takes it as `dag_path` (DESIGN.md section 6). A copy
  is placed in the package's shared dir and exported as
  `DTSCHEDULE_DAG_PATH`.
* `${HOME}/dtschedule-runs/<name>/logs/` -- one log per task and per
  (level, node) script.
* `${HOME}/jarvis-runs/dtschedule-results/<exp>/` -- sweep.py JSON per
  run plus `traces/<run>/` with that run's trace CSVs; pipeline tests
  write `results.csv` there as well (their traces are NOT copied: each
  row's configure resets the trace in place, so use sweep.py when the
  per-row traces are needed).

## Before running on ares

* `salloc` your own 2-node job, check `/mnt/nvme` free space on both
  nodes (>= 64 GB for the NVMe tier, 128 GB for E10) and `TIME_LEFT`.
* The jarvis repo registration must point at the clio-core checkout that
  has `jarvis_clio_core/clio_dtschedule` (`jarvis repo list`); otherwise
  `pkg_type: jarvis_clio_core.clio_dtschedule` fails to load.
* Do not run unit tests on the allocated nodes while the pipeline runs
  (ctest cleanup wipes `/tmp/clio_$USER`).
