# Next steps (temporary; delete before merging)

Frozen SSC: **v10**. Fast models: qwen3-4B, llama3.1-8B (thinking off). Slow baseline: qwen3-4B,
no SSC, thinking on.

## Code still needed

- [ ] Replication (0/1 copy) and the core data organizer (prefetching: none / frecency / cyclic /
      hotset) as decisions in the SSC, `compose.py` and `run_grid.py`
- [ ] A second application (h5bench- or DLIO-style workload)
- [ ] Slurm batch script and a short setup guide for Delta (build or Apptainer image, Ollama,
      `SSC_ROOT` on local NVMe, `SSC_QUEUE_ROOT` node-local, bridges on the same node)
- [ ] Run CLIO's own cache and replication unit tests (build with tests on)

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
