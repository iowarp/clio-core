# DTSchedule evaluation figures

Measurement-driven figure scripts for every experiment in
`../docs/EVAL_PLAN.md` section 2. Every figure is built from the chimod's
decision trace (`../docs/DESIGN.md` section 9), the `sweep.py` result JSONs
and the workflow's `makespan.json`; nothing is hand-entered. The scripts
replace the paper repo's `figures/evals/scripts/*.py`, which generated
their numbers from analytic models.

Style follows the `dataviz` skill: single-column width (3.3 in), 8 pt
fonts, fixed-order categorical palette (a codec / scenario / predictor
keeps its colour in every figure), one-hue sequential and blue-gray-red
diverging ramps, hatching as the second channel on stacked segments.
Every script writes `<name>.svg` and `<name>.pdf`.

## Experiment -> script -> inputs -> output

| exp | script | inputs | output |
|---|---|---|---|
| E0 motivation (B3) | `e0_motivation.py` | `--candidates '<trace>.cand.csv'`, `--traces '<trace>.*.csv'` of the compress bench run (`ccm: oracle` for measured costs) | `e0_motivation` |
| E1 joint vs sequential (A1, B1) | `e1_joint_vs_sequential.py` | `--candidates` of the joint run (offline replay), `--results <exp>` of the `decision_order` sweep (online) | `e1_joint_vs_sequential` |
| E2 tier sensitivity (A2, B4) | `e2_tier_sensitivity.py` | `--results <exp>`; runs carry `tier_set` + `capacity` knobs or are named `<tier_set>[__cap]`; per-run traces for bytes / codec per tier | `e2_tier_sensitivity` |
| E5 scenario ablation (B6, B1) | `e5_scenario_ablation.py` | `--results <parent>` with one `sweep.py --exp` dir per grid cell (`c<cons>_p<prod>_n<net>`), runs over `force_scenario`; auto's traces or `dtschedule.scenario.*` stats | `e5_scenario_ablation` |
| E6 component ablation (C1, B6) | `e6_component_ablation.py` | `--results <exp>` of `e6_component_ablation.yaml` (one sub-dir per workflow); component set derived from `workflow_aware`, `prefetch`, `ccm`, `load_aware`, `qos_max_error` | `e6_component_ablation` |
| E7 QoS sweep (C3, A3, B5) | `e7_qos_sweep.py` | `--results <exp>` of `e7_qos_sweep.yaml`; per-run traces; downstream error from stat `analysis.linf_error` or `--errors run,linf_error.csv` | `e7_qos_sweep` |
| E8 predictor drift (C4, C2) | `e8_predictor_drift.py` | `--results <exp>` of the `ccm` sweep with per-run traces, or `--traces '<dir>/e8_{ccm}/dtschedule_trace.*.csv'`; optional `--candidates` of the oracle run | `e8_predictor_drift` |
| E9 prediction error (C5) | `e9_prediction_error.py` | `--results <exp>` of the `ratio_noise_sigma` sweep (+ a `ccm: oracle_ratio` run); per-run traces for exact flip rates | `e9_prediction_error` |
| E10 load timeline (C6) | `e10_load_timeline.py` | `--traces '<trace>.*.csv'` of the E10 run, `--steps 120:consumer +load,240:consumer release,300:producer +load` | `e10_load_timeline` |
| E11 overheads (C7) | `e11_overheads.py` | `--traces` of a Gray-Scott run with 64 KiB / 1 MiB / 16 MiB blocks (`--sizes` to change) | `e11_overheads` |

Common flags: `--results DIR`, `--out DIR` (default `eval/out/`), `--demo`
(synthetic input only to exercise the script; the title says "DEMO DATA"
and the default output moves to `${HOME}/dtschedule-scratch/eval-demo/`,
never `eval/out/`). Every script prints the figure's companion table.

`dtlib.py` holds the loaders (`load_traces`, `load_candidates`,
`load_results`), the statistics (`bootstrap_ci`, `aggregate`,
`normalize_to_baseline`) and the style (`figure`, `savefig`, palette).

## Input layouts

* **Decision trace** `<trace_path>.<node>.csv`: the 26 columns of
  DESIGN.md section 9 followed by `obs_dtime_ms,select_ms` (28 columns).
  Decision rows fill `select_ms` (SelectCodec wall time, ms) and leave
  `obs_dtime_ms` empty; decompress (GetBlob) rows fill `obs_dtime_ms` and
  leave the decision fields empty. Every put that went through selection
  is written, including ones stored raw (`chosen_lib = raw`,
  `obs_ratio = 1`); ratios are original / compressed (>= 1).
  `load_traces` labels rows `kind = decision | decompress`; as fallbacks
  it still accepts the decompression time as a trailing 27th field or
  (phase-2 runtime) in the `knobs_hash` slot.
* **Candidate file** `<trace_path>.cand.<node>.csv` (`trace_candidates:
  true`), one row per candidate per decision:
  `ts_ms,node,tag,blob,lib,preset,pred_ctime_ms,pred_dtime_ms,pred_ratio,cost_ms,reason`
  with `reason` in {ok, qos_lossy_not_allowed, qos_error_bound,
  qos_preference, unavailable, fixed, skip_ratio}; `cost_ms` is empty
  unless `reason == ok`. Scenario and tier are decision-level (read them
  from the trace row with the same (tag, blob)); `dtlib.ok_candidates`
  selects the ranked rows. Pass `'<trace_path>.cand.*.csv'` as the glob.
* **Sweep result** `${HOME}/jarvis-runs/dtschedule-results/<exp>/<run>.json`
  from `jarvis_clio_core/pipelines/ares/dtschedule/sweep.py`: `overrides`
  (knobs, `pkg.key`), `stats` (every package's `_get_stat`, e.g.
  `dtschedule.lib.zstd`, `dtschedule.scenario.3`, `wfcommons.makespan_ms`)
  and `makespan` (`total_ms`, `levels[]`, `nodes[]`). `load_results`
  flattens knobs to plain columns (`force_scenario`, `ccm`, ...), keeps
  stats under their full key, walks sub-directories (name -> `cell`) and
  looks for per-run traces in `<exp>/[<cell>/]traces/<run>/*.csv`.

Repeats: name repeated runs `<run>_r<k>` (or `.r<k>`); `aggregate` groups
by knobs and `bootstrap_ci` gives percentile intervals over repeats.

## From sweep to figure

```bash
# 0. environment (ares): own allocation, nvme space, TIME_LEFT checked
PY=${HOME}/venv-dtschedule/bin/python
PIPES=jarvis_clio_core/pipelines/ares/dtschedule
RES=${HOME}/jarvis-runs/dtschedule-results
EVAL=context-transfer-engine/dtschedule/eval
mkdir -p /mnt/nvme/${USER}/dtschedule_e5          # per pipeline name

# 1. run a sweep (one JSON per run); E5 is one --exp per load/network cell
python3 $PIPES/sweep.py --exp e5/c0_p0_n25  --from-test $PIPES/e5_scenario_ablation.yaml --hostfile ${HOME}/hostfile.txt
#   ... inject stress-ng / tc for the next cell, then:
python3 $PIPES/sweep.py --exp e5/c50_p0_n25 --from-test $PIPES/e5_scenario_ablation.yaml --hostfile ${HOME}/hostfile.txt
python3 $PIPES/sweep.py --exp e6/montage    --from-test $PIPES/e6_component_ablation.yaml --hostfile ${HOME}/hostfile.txt
python3 $PIPES/sweep.py --exp e7            --from-test $PIPES/e7_qos_sweep.yaml          --hostfile ${HOME}/hostfile.txt
python3 $PIPES/sweep.py --exp e1 --pipeline $PIPES/montage_2n.yaml --hostfile ${HOME}/hostfile.txt \
    --run joint:dtschedule.decision_order=joint --run codec:dtschedule.decision_order=codec_first \
    --run tier:dtschedule.decision_order=tier_first
python3 $PIPES/sweep.py --exp e9 --pipeline $PIPES/montage_2n.yaml --hostfile ${HOME}/hostfile.txt \
    --run s0:dtschedule.ratio_noise_sigma=0 --run s01:dtschedule.ratio_noise_sigma=0.1 \
    --run s025:dtschedule.ratio_noise_sigma=0.25 --run s05:dtschedule.ratio_noise_sigma=0.5 \
    --run s1:dtschedule.ratio_noise_sigma=1.0 --run oracle:dtschedule.ccm=oracle_ratio

# 2. keep the traces per run (the runtime resets them at every configure):
#    after each run, before the next one starts
mkdir -p $RES/e7/traces/run0 && cp ${HOME}/jarvis-runs/dtschedule_e7/dtschedule_trace.* $RES/e7/traces/run0/
#    sweep.py can do this between runs when extended; until then copy by hand
#    or with a one-line post_cmd in the pipeline yaml.

# 3. figures (SVG + PDF in eval/out/)
$PY $EVAL/e5_scenario_ablation.py  --results $RES/e5
$PY $EVAL/e6_component_ablation.py --results $RES/e6
$PY $EVAL/e7_qos_sweep.py          --results $RES/e7 --errors ${HOME}/dtschedule-runs/e7_linf.csv
$PY $EVAL/e1_joint_vs_sequential.py --results $RES/e1 --candidates "$RES/e1/traces/joint/*.cand.csv"
$PY $EVAL/e9_prediction_error.py   --results $RES/e9
$PY $EVAL/e2_tier_sensitivity.py   --results $RES/e2
$PY $EVAL/e8_predictor_drift.py    --results $RES/e8 --candidates "$RES/e8/traces/oracle/*.cand.csv"
$PY $EVAL/e0_motivation.py  --candidates "$RES/e0/traces/run0/*.cand.csv" --traces "$RES/e0/traces/run0/*.csv"
$PY $EVAL/e10_load_timeline.py --traces "${HOME}/jarvis-runs/dtschedule_e10/dtschedule_trace.*.csv"
$PY $EVAL/e11_overheads.py     --traces "${HOME}/jarvis-runs/dtschedule_e11/dtschedule_trace.*.csv"

# 4. smoke test every script without cluster data
for s in $EVAL/e*.py; do $PY $s --demo --out ${HOME}/dtschedule-scratch/eval-demo/; done
```

## Notes on what the measurements must contain

* E6's compute / I/O split uses `dtschedule.puts * mean_obs_ctime_ms +
  dtschedule.gets * mean_obs_dtime_ms` divided by the node count; cache hit
  rate and remote bytes are printed once `_get_stat` emits
  `dtschedule.cache_hit_rate` / `dtschedule.remote_read_bytes`.
* E8 regret without an oracle candidate file compares each predictor's
  observed cost per timestep with the oracle run's, so the four runs must
  write the same blobs.
* E10 needs the `producer_cpu` / `consumer_cpu` trace columns, which the
  phase-2 writer leaves empty; the CPU row says so when they are missing.
* E11 shows the selection phase only when the trace has a `select_ms`
  column; otherwise the title says "selection not in trace".
