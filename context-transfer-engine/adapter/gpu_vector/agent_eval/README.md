# agent_eval -- does the gpu-vector skill help agents write paged GPU code?

Agents reimplement gpu_vector benchmarks **from scratch** in a sanitized copy
of this repository, with and without the `gpu-vector` skill
(`.claude/skills/gpu-vector/`). We measure what it cost to get them working
(tokens, dollars, turns, wall time), whether the result is correct, and how
fast it is against our own paged implementation and an in-core baseline.

## Design

| factor | levels |
|---|---|
| model | `claude-opus-5-5`, `claude-sonnet-5` (Haiku 4.5 dropped: in the pilot it wrote plain CUDA and never used the paged vector, with or without the skill) |
| arm | `none` (no skill); `skill` (the gpu-vector skill: API contract, general design rules, cost model, pitfalls) |
| task | `kmeans` (pilot); grayscott, lbann, md to be added under `tasks/` |
| reps | >= 3 per cell for anything reported; agent runs are high-variance |

**The skill is generic by construction.** It names no workload, describes no
workload, and contains no worked implementation: lessons are phrased by access
pattern (streaming, reuse window, scatter/gather, exchange) with the measured
effect behind each. A skill that described the benchmarks would let the agent
reproduce our code rather than apply knowledge.

**Protocol.** The prompt asks only for a correct, working benchmark; it says
nothing about performance. Without the skill, an agent is expected to stop at
its first working draft, and that draft is what gets benchmarked. The skill's
workflow (validate, then optimize in COSTS.md's order) is what carries the
skill arms further. So the measured effect is the skill's knowledge *and* its
instruction to optimize, together; this design does not separate the two.
(Pilot v1, archived under `archive/pilot_v1_make_it_fast/`, used a prompt that
also said "then make it fast".) Every arm receives the identical prompt (`tasks/<task>/TASK.md` +
`tasks/common.md`); the only difference is the skill directory in the tree.

## Isolation ("from scratch")

`sanitize.sh` builds the tree every trial starts from: no git history, no
workload sources or docs, no workload tests, no jarvis wrappers, no skills,
no harness. `build_base.sh` builds it once (Release, nvcc, sm_89, tests off)
so the prebuilt build contains libraries only. Each trial runs in its own
container with a private copy of tree and build, the GPU, and a fresh Claude
config holding only a dedicated token -- no user settings, memory, plugins or
skills.

**Residual leakage, identical across arms:** comments in the gpu_vector and
runtime headers name the workloads and some of their lessons, and the CTE
core ships data organizers named after two workloads.

## Metrics

- **Effort** (`analyze.py`, from the transcript's final `result` event):
  input / output / cache-write / cache-read tokens across all models
  (subagents included), cost in USD, turns, API time, wall time, timeouts;
  plus tool calls, build and run attempts, and whether the skill was loaded
  and which of its files were read.
- **Correctness** (`grade.py`): the deliverable builds from clean; every deck
  exits 0 with exactly one RESULT line; results match an independent CPU
  reference (`tasks/<task>/ref.cc`); oversubscribed decks actually faulted and
  evicted.
- **Performance** (`grade.py`): median over 3 runs on an idle GPU, reported
  as a ratio to our paged implementation at the same cache budget and host-RAM
  store (`vs_paged_ref`) and to the in-core baseline (`vs_incore`).

k-means references (RTX 4070 Laptop, ms/iter, median of 3):

| deck | data | cache | paged ref | in-core |
|---|---|---|---|---|
| resident_128 | 128 MB | 256 MB | 10.7 | 10.2 |
| ooc8x_256 | 256 MB | 32 MB | 93.8 | 20.3 |
| ooc8x_512 | 512 MB | 64 MB | 102.4 | 40.3 |

Agents are told to keep every test under 500 MB of GPU memory, because all
trials run at once on one 8 GB GPU; grading runs afterwards, alone.

## Running

```bash
# once
./sanitize.sh && ./build_base.sh
./grade.py refs kmeans
claude setup-token           # then: echo 'CLAUDE_CODE_OAUTH_TOKEN=...' > $RUNS/secrets.env; chmod 600
cp -L "$(command -v claude)" $RUNS/tools/claude   # pinned CLI; >= 2.1.280 for Opus 5.5
# a grid
TASKS=kmeans REPS="1" PARALLEL=2 ./run_matrix.sh
./analyze.py --csv results.csv
```

`$RUNS` is `<clio-core>/agent_eval_runs` (git-ignored locally). Budget per
trial: `TRIAL_TIMEOUT` (default 4h). Rerunning `run_matrix.sh` skips finished
trials.
