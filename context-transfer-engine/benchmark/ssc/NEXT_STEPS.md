# Next steps (temporary; delete before merging)

Frozen SSC: **v10**. Fast models: qwen3-4B, llama3.1-8B (thinking off). Slow baseline: qwen3-4B,
no SSC, thinking on.

## Code still needed

- [ ] Replication (0/1 copy) and the core data organizer (prefetching: none / frecency / cyclic /
      hotset) as decisions in the SSC, `compose.py` and `run_grid.py`
- [x] A second application (h5bench- or DLIO-style workload): `dl_samples`, `dl_ckpt`, `h5_particles`
      (`workload.APP2_PHASES`, `--app app2`, `dseed:N`). Never in the SSC's memory (`MEMORY_PHASES` =
      Gray-Scott only), so it tests cross-application transfer with v10 unchanged
- [x] Slurm batch script and a short setup guide for Delta: `delta/` (README, env.sh, campaign1/2,
      offline_lane + lanes/, smoke, build, tests)
- [x] Run CLIO's own cache and replication unit tests: 17/17 pass (cache, replication, indexer, compressor)

## Runs on Delta (one GPU node with local NVMe; about 16-20 node-hours)

| # | Run | Why | Time |
|---|---|---|---|
| 1 | Grid at 1 GB and 0.5 GB DRAM (+ one larger, e.g. 4 GB), 3 repeats | Real hardware; repeat spread should drop below 30% | ~3 h |
| 2 | Offline, frozen v10, test seeds 101-110: fixed, fixed guess, rules, lookup, best possible, each model with/without SSC | Main offline table | 1-2 h |
| 3 | Model-size sweep offline: qwen3 1.7B / 4B / 8B, a ~14B model | Where the SSC starts working | ~1 h |
| 4 | Live, 5 test schedules x 2 node sizes: fixed, rules, qwen3-4B and llama3.1-8B with/without SSC, qwen3-4B no SSC thinking on | End-to-end speed and score with error bars | 8-10 h |
| 5 | Live with `--async-decide` for the SSC runs | Hide decision time (51 s of 260 s locally) | ~2 h |
| 6 | Guard ablation offline: guards only / model only / both | "Are the guards doing the work?" | <1 h |
| 7 | Portability: rules for one node size applied to the others vs a rebuilt SSC | Rules need rewriting; the SSC rebuilds itself | <1 h |

## Paper (Phd_Research/papers/ssc/paper/main.tex)

- [ ] Replace every red TODO with Delta numbers
- [ ] Fill authors/venues in refs.bib
- [ ] Final system name (current macro: Agora)

## WfCommons on Delta (unseen real applications)

- [ ] Converter: WfCommons workflow instances (tasks, files, sizes, dependencies) -> harness phases
      (one window per workflow stage); decide which outputs are "searched later" (e.g. final outputs)
- [ ] Fix the workflow set before any results (e.g. Montage, 1000Genome or Epigenomics, BLAST or Seismology)
- [ ] Runs: workflow grid (3 repeats); SSC with Gray-Scott-only experience on workflows; leave-one-workflow-out;
      live makespan (SSC / no SSC / rules / fixed / thinking) with error bars; mixed workflow runs;
      cold start on a new workflow
- [ ] Optional: one live run of real WfBench through the CLIO filesystem interface

## Paper changes after Delta (papers/ssc/paper/main.tex; figure scripts in papers/ssc/scripts)

- [ ] Abstract/intro: Delta numbers; add WfCommons; frame as "no rewriting when node or app changes"
- [ ] Section 2: heatmap + best-setup table from the Delta grid, with WfCommons stages as rows
- [ ] Sections 3-4: decision and swap times from Delta; async decisions and cold start if built; grid cost
- [ ] Section 5: unit tests done; replication/prefetch as decisions if added
- [ ] Section 6: Delta setup + error bars; workflows; new "unseen applications" subsection; makespan;
      SSC-parts and guard ablations; model-size sweep; cold-start curve; drop or redo the Claude check
- [ ] Related work: WfCommons/workflow I/O; fill refs.bib; Limitations rewritten
- [ ] Regenerate fig0-fig4 + new unseen-application and model-size/cold-start figures
- [ ] Final name (DBLP check), ACM AI-use disclosure, use the remaining pages (6 of 11 now)
## Status on Delta (2026-11-24)

Workspace `ssc_work/` in the checkout (git-excluded): results, figures, logs, models.
Figures + `summary.md`: `python3 paper_plots.py ssc_work/results ssc_work/figures` (host python).

Added beyond the plan (reviewer-driven):
- Guard ablation in `ablate_ssc.py` (`ng_<variant>`, `guards`); a broken declared requirement scores 0
- Second metric: slowdown vs the best setup (geomean of window time / best), since the 0-1 score
  saturates on this node; score clipped at 0; 95% bootstrap CIs over schedules
- Strongest System 2 baseline: qwen3-14B thinking, no SSC
- Noise sensitivity: every key arm rescored with each single grid repeat as ground truth

Done: grids 1 GB / 4 GB (10 phases, 3 repeats), 0.5 GB (one cell being repaired); offline 1 GB and
4 GB for gs and app2, model sweep (1 GB), guard ablation (1 GB).
Running/queued: 0.5 GB + generalization-phase replays, remaining System 2 schedules (offline lanes
a/b/c), live runs (campaign2).

Findings so far (slowdown vs best, lower is better):
- 1 GB gs: S1+SSC llama-8B 1.20x / qwen-4B 1.38x at ~1 s; S2 no SSC (4B think) 1.19x at ~47 s;
  S2 no SSC 14B think 6.6x; S1 no SSC 5.6x; expert rules 1.08x
- 1 GB app2: S1+SSC 1.03x; S2 no SSC ~1.36x; rules 1.62x; S1 no SSC 4.8x
- SSC lifts every model size (1.7B-14B); guards never fire with the SSC (ablation identical)
- Caveats to state in the paper: on gs the rules are as good as any model; on 4 GB app2 guards-only
  (1.36x) beats S1+SSC (1.53-1.71x); differences within ~20% flip across grid repeats (report ties)
