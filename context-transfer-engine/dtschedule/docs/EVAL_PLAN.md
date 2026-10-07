# DTSchedule EuroSys revision: evaluation plan

Paper: *DTSchedule: A Load-Balanced, Workflow-Aware, and Tiered Transparent
Compression System* (EuroSys '27 Spring, #1754; A: weak reject, B: weak reject,
C: weak accept). Reviews live in
`external/dtschedule_.../eurosys-dtschedule-reviews.txt`.

This document maps every reviewer concern to a concrete experiment, states
what code has to exist for that experiment, and fixes the workloads and
testbed. The matching code lives in `context-transfer-engine/dtschedule/`
(the `clio::cte::dtschedule` chimod) on the `dtschedule` branch.

## 0. Ground truth about the submitted evaluation

The figure scripts in the paper repo (`figures/evals/scripts/*.py`) show
that `end_to_end.svg`, `end_to_end_emulated.svg`, `cost_mape.svg`,
`regret.svg` and `overhead_breakdown.svg` were **generated from analytic
models, not measurements** (the docstrings say "projected (not measured)"
and "numbers are produced from a per-workflow model"). Every figure in
sections 5.5 to 5.9 therefore has to be re-collected from real runs. The
model-accuracy figures (5.1 to 5.3, R², inference latency, SHAP) came from
the Q-table training pipeline and can be regenerated from
`compressor/generator/`.

The tree also does not contain the system the paper describes. What exists
(`context-transfer-engine/compressor/`, built only with
`-DCLIO_CTE_ENABLE_COMPRESS=ON`):

| Paper component | In tree today |
|---|---|
| Q-table CCM | `models/qtable_predictor.h`, trained-model files are **not** checked in |
| Compression ranking (Alg. 1) | `compressor_runtime.cc:EstWorkflowCompressTime`, fixed candidate set of 5 lossless configs, no error-bound filter for lossy |
| Load multiplier | `CompressionFeatures.target_cpu_util` is never set; `PollConsumers` only logs |
| Tier selection (Alg. 2) | `BestCompress*` always returns tier 0 |
| Three scenarios | not implemented; `ScheduleTask` routes compress tasks to the last reader's node only |
| Consumer routing / prefetch (+w) | `RegisterConsumer` + `PickConsumerForTag` (last reader), prefetch is the core's organizer |
| DAG awareness | nothing |
| QoS YAML | nothing (only `compress_lib`/`preset` per Context) |
| Interposed PutBlob codec choice | **none**: `PutBlob` forwards raw when `compress_lib_<=0`, so FUSE/POSIX/HDF5/ADIOS2 traffic is stored uncompressed |

So the chimod is not optional: the ablations below need switches that do
not exist, and the main path needs the selector wired in at all.

## 1. Reviewer concerns, condensed

| id | reviewer | concern |
|---|---|---|
| A1 | A | Existing selectors (DeepPress, PACM, ZFS-QoS) could be coupled with tier/location as extra inputs; no analysis of why joint selection is needed |
| A2 | A | No analysis of how much each storage tier contributes |
| A3 | A | Users who can state error bounds can compress themselves; defaults/auto are not evaluated |
| B1 | B | Codec choice and placement "do not influence one another"; engineering, not novelty |
| B2 | B | "Workflow-aware" should mean the DAG: replicate fan-out outputs, co-locate fan-in inputs |
| B3 | B | Motivation shows one config always best, so nothing to choose |
| B4 | B | Describe storage tiers and their characteristics |
| B5 | B | Is QoS per workflow or per stage; scatter/reduce stages |
| B6 | B | No ablation; compare against always-scenario-1/2/3 |
| C1 | C | Quantify +w (routing/prefetch) vs CCM vs load-aware |
| C2 | C | Define state/action/reward/update or stop calling it RL |
| C3 | C | QoS sensitivity: vary max_error and performance-vs-ratio |
| C4 | C | Online moving-average baseline to isolate the Q-table's benefit |
| C5 | C | Effect of ratio-prediction error (R²=0.57) on decisions |
| C6 | C | A runtime trace of load changing and decisions following |
| C7 | C | 10 ms vs 800 µs compression-time inconsistency |
| all | | Figures 4 and 9 unreadable; related-work numbering |

## 2. Experiments

Every experiment reads one artifact: the **decision trace** the chimod
writes (one CSV row per PutBlob: timestamp, blob id, size, data stats,
predicted cost per candidate, chosen scenario/codec/preset/tier/node,
producer and consumer CPU, observed compress/decompress/store time, observed
ratio). Figures are generated from traces plus the workflow's own makespan
log, never from hand-entered numbers.

Priority P0 experiments answer the most reviewers per hour of cluster time.

### E5 Scenario ablation (B6, B1) — P0

Force the placement scenario: S1 (compress at producer, store at producer),
S2 (compress at producer, store at consumer), S3 (ship raw, compress and
store at consumer), vs. `auto`. Run each over a grid that makes different
scenarios win: consumer load ∈ {idle, 50 %, 100 %}, producer load ∈ {sim
only, sim+50 %}, network ∈ {25 GbE, throttled via `tc` to 5 GbE}.

Deliverable: a 3×(grid) heatmap of makespan normalised to the best fixed
scenario per cell, plus the auto row. Claim to support: no fixed scenario is
within 10 % of best everywhere; auto is within 5 % of best-fixed in every
cell. Secondary bar: distribution of scenarios chosen by auto per cell.

Knob: `dtschedule.force_scenario: auto|1|2|3`.

### E6 Component ablation (C1, B6) — P0

Leave-one-out and cumulative-add over five switches:
routing (+w), prefetch, CCM selection (vs fixed zstd-balanced), load-aware
offload (+l), lossy family. Stacked makespan (compute / I/O) per workflow,
plus cache hit rate and bytes read remotely, which the paper's Limitations
section promised and never delivered.

Knobs: `workflow_aware: none|consumer|dag`, `prefetch: on|off`,
`ccm: fixed:<lib>|qtable|ema|oracle`, `load_aware: on|off`,
`qos.max_error: 0|<rel>`.

### E10 Load-response timeline (C6) — P0

One Gray-Scott (or WfBench stream) producer-consumer pair, 10 minutes.
Inject load steps with `stress-ng --cpu`: consumer at t=120 s, release at
t=240 s, producer at t=300 s. Plot four aligned rows: producer and consumer
CPU %, chosen compression location, chosen codec, per-write latency. The
decisions should visibly flip at each step.

### E7 QoS sweep (C3, A3, B5) — P0

Gray-Scott and one float-dominated WfCommons recipe (seismology or
montage). Sweep `max_error ∈ {lossless, 1e-2, 1e-3, 1e-4, 1e-5 rel}` ×
`objective ∈ {performance, ratio}`. Report makespan, bytes stored per tier,
codec histogram, and the downstream analysis error (Gray-Scott PDF
consumer: L∞ error of the PDF vs the lossless run). Add the "no QoS at all"
column (A3): defaults only, no allowlist, no bound. Per-stage QoS (B5):
one run where stage 2 is lossless and stage 1 lossy, to show the spec is
per stage via file-regex.

### E1 Joint vs sequential decision (A1, B1) — P1

Three deciders on the same trace: (a) codec-then-tier (a DeepPress/PACM
style selector picks the codec from data only, then the tier is picked by
capacity/bandwidth), (b) tier-then-codec, (c) joint (Alg. 1+2). Replay
offline on logged candidates to count decision flips, then run online.
Deliverable: flip-rate table and makespan delta; the cells where (a)≠(c)
are exactly the loaded-producer / slow-tier cells of E5.

Knob: `decision_order: joint|codec_first|tier_first`.

### E2 Tier sensitivity (A2, B4) — P1

Same workload on tier sets {RAM}, {NVMe}, {SSD}, {NFS}, {RAM+NVMe},
{RAM+NVMe+SSD}, {all+NFS}; capacity-constrained (fast tier holds 30 % of
the data) and unconstrained. Report makespan, bytes landed per tier, codec
chosen per tier. Characterise each ares tier first (fio: seq read/write BW,
4 KiB latency) and put that table in §5 (B4).

### E0 Motivation redo (B3) — P1

Replace Figure 1 with a decision-flip figure from the compress bench: for
chunks at four entropy levels × producer load {0,50,100 %} × target tier
{RAM, NVMe, SSD}, mark the best (codec, location). Show that the argmax
changes across cells and that the best static policy is ≥X % off in ≥Y % of
cells.

### E8 Predictor ablation under drift (C4, C2) — P1

Real Gray-Scott, 40 timesteps, single node. Predictors: Q-table with online
running-mean update; per-(lib,preset) EMA of observed cost (no data
features); XGBoost retrained every 8 steps; oracle (all candidates run).
MAPE and regret per timestep, measured not modelled. The Q-table write-up
becomes: state = feature bins, action = (lib, preset), value = running mean
of observed (time, ratio), update = incremental mean; we drop the word
"reinforcement" unless an exploration policy is actually in play.

Knob: `ccm: qtable|ema|xgboost|oracle`, `ccm.update: online|periodic:<n>`.

### E9 Prediction-error sensitivity (C5) — P1

Inject multiplicative noise into the predicted ratio (σ ∈ {0, 0.1, 0.25,
0.5, 1.0}) and separately substitute oracle ratios. Measure decision-flip
rate and makespan vs σ. Shows how much the R²=0.57 ratio prediction costs.

Knob: `ratio_noise_sigma`, `ccm: oracle_ratio`.

### E4 DAG-aware placement (B2, C-Q2) — P2, new capability

The WfCommons DAG is known before the run. Policies: (i) fan-out
replication: a file consumed by ≥R tasks on distinct nodes is replicated
(compressed) to those nodes at write time; (ii) fan-in co-location:
outputs of siblings consumed by one downstream task are placed on that
task's node; (iii) consumer routing alone (current +w). Compare none /
consumer / dag / dag+replicate on Montage (mProject→mDiffFit fan-out,
mConcatFit fan-in), 1000Genome (individuals→frequency), Seismology
(wide fan-in) and Epigenomics. Metrics: makespan, remote-read bytes, local
hit rate, replica bytes.

Knobs: `workflow_aware: dag`, `dag.replicate_fanout_min: R`,
`dag.colocate_fanin: on|off`, `dag_path: <wfformat.json>`,
`dag_placement: <task→node json>`.

### E11 Overheads at several chunk sizes (C7) — P2

Selection + compression + store latency at 64 KiB, 1 MiB, 16 MiB on real
Gray-Scott blocks; state where 10 ms comes from (16 MiB chunks) or delete
the claim.

### E12 Scalability (keep) — P2

Keep the Aurora weak-scaling figure if the old numbers are real; else rerun
on ares up to 24 nodes and on Aurora only if time permits.

## 3. Workloads

Focus on WfCommons recipes (the reviewer-visible benefit: standard,
reproducible DAGs with real fan-out/fan-in). Two problems with stock
WfBench, both fixed in this branch:

1. **Data is `os.urandom`** (`wfbench` lines 226, 337, 599, 764): every
   file is incompressible, so any compression system looks like a no-op.
   We add a data generator selected by file extension: structured float
   fields (smooth 2-D fields with noise, like FITS/NetCDF/SAC) and text
   records (FASTQ-like / VCF-like) so lossless and lossy both have
   something to do. The generator is seeded per file so runs repeat.
2. **The builtin jarvis `wfcommons` package runs the whole DAG on every
   node** via the `BashTranslator` (sequential levels, one node). Producer
   and consumer therefore always share a node and there is no
   inter-stage transfer to schedule. We add a distributed translator that
   assigns tasks to nodes level by level (round-robin, or stage→node-set
   so that stages run on disjoint nodes like the paper's "stream" mode)
   and emits the task→node map that the chimod reads for E4.

Recipes and why:

| recipe | shape | data class | used in |
|---|---|---|---|
| montage | fan-out then fan-in, float images | float | E4, E6, E7 |
| seismology | wide fan-in, float traces | float | E4, E7 |
| genome (1000Genome) | scatter/gather, text | text, lossless only | E4, E6 |
| epigenomics | pipeline with fan-out | text | E4, E6 |
| cycles | parameter sweep, mixed | float+text | E6 |
| srasearch / blast / bwa | compute-heavy text | text | one of them in E6 |
| Gray-Scott (ADIOS2, real sim) | stream | float | E7, E8, E10, E11 |

Gray-Scott stays as the real-data anchor so synthetic compressibility is not
the only evidence.

## 4. Testbed

ares: RAM (tmpfs), NVMe, SATA SSD, NFS as the PFS stand-in; 2 to 24 nodes
(salloc our own job, check `/mnt/nvme` free space and `TIME_LEFT` first).
Load injection with `stress-ng --cpu N` pinned to the producer or consumer
node; network throttling with `tc qdisc ... tbf` on the consumer's NIC.
Aurora only for E12.

Build: host gcc 11.4 + spack deps, `cmake --preset release-fuse
-DCLIO_CTE_ENABLE_COMPRESS=ON` with cmake 3.31 from spack (the env script
is `scripts/dtschedule_env.sh` once committed). Lossy codecs come from
spack libpressio 0.99.4 (sz3 3.2, zfp 1.0.1, fpzip 1.3). snappy, brotli,
bzip2 and blosc2 are being added via spack.

## 5. Code required (branch `dtschedule`)

1. `clio::cte::dtschedule` chimod (new, `context-transfer-engine/dtschedule/`):
   interposer in front of the compressor pool; implements Alg. 1 (ranking
   with error-bound filter and load multiplier), Alg. 2 (tier by capacity,
   three scenarios), DAG policies, the QoS YAML, every knob above, and the
   decision trace. The compressor pool stays the mechanism (codec execution,
   header, consumer registration).
2. Compressor fixes it depends on: preset numbering mismatch
   (`compressor_runtime.cc:470` vs `:844`), `target_cpu_util` never set,
   `library_config_id` encoding, tier always 0, `PollConsumers` samples
   unused, interposed `PutBlob` never calling the selector.
3. Node-load feed: reuse `PollNodeLoadTask` (cpu % from `/proc/stat`,
   worker load) but keep a per-node ring in dtschedule and expose
   `load_mult` to the ranker.
4. Trained Q-table shipped in-tree (`dtschedule/models/qtable_v1.json`)
   plus the EMA and oracle predictors for E8/E9.
5. WfCommons tooling under `jarvis_clio_core/`: patched `wfbench` data
   generator, distributed translator, `clio_dtschedule` jarvis package,
   pipelines for every experiment under `pipelines/ares/dtschedule/`.
6. Analysis scripts under `dtschedule/eval/`: trace → figures for each Ex,
   so the paper's `figures/evals/scripts` are replaced by scripts that read
   measurements.

## 6. Order of work

1. Chimod skeleton with knobs, trace, and Alg. 1/2 on top of the existing
   compressor (unit test with the compress bench).
2. WfBench data generator + distributed translator + jarvis package; one
   2-node montage run end to end.
3. E5, E6, E10, E7 (P0) on 4 to 8 nodes.
4. E1, E2, E0, E8, E9 (P1).
5. DAG policies and E4; E11; E12.
