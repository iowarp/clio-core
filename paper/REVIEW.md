# Self-review as an ICS reviewer (draft of 2026-11-25, before WfCommons results)

**Score now: weak reject → borderline.** The idea is timely (fast vs. slow LLMs for online systems
decisions), the SSC design is well motivated, and the evaluation is unusually careful (frozen
design, unseen schedules, CIs over schedules, noise sensitivity, guard ablation). It is not yet an
accept because the headline claims rest on offline replays and on scenarios where simple
baselines are as good as the model.

## Strengths a reviewer will credit
- Clear problem (who chooses the composition?) and a concrete mechanism on a real runtime (CLIO).
- The SSC's design rules are principled and ablated; leave-workload-out prevents answer leakage.
- Three applications, one from real WfCommons traces; three node setups; five model sizes.
- Honest negative results (rules competitive on the design application; noise flips small gaps).

## Weaknesses a reviewer will raise, and what answers each

| # | Reviewer concern | Status | Action |
|---|---|---|---|
| W1 | "On Gray-Scott the expert rules / guards-only / a model-free lookup are as good. Why an LLM?" | partly answered (DL/HDF5: rules 1.62x; 4 GB: lookup 57%) | WfCommons on 0.5 GB and DRAM+PFS, where the default is not best; report rules, lookup and S1+SSC side by side (running) |
| W2 | "40x faster decisions is not end-to-end speed" | 1 live schedule | live WfCommons, 3 schedules, all arms incl. **async** S1+SSC and S2 blocking/async (running) |
| W3 | "S2 without SSC is a strawman; give S2 the SSC" | GS: S2+SSC 1.09x at 81 s | state it: the SSC helps S2 too, but S2+SSC is 80x slower; nothing more needed |
| W4 | "Is it the node experience or just the window facts?" | old laptop numbers only | cold-start (v10x) on WfCommons 1 GB / 0.5 GB / PFS (queued in the S1 units) |
| W5 | "The grid costs an hour per node" | not addressed | report it honestly; cold start bounds what is lost without it; future work: incremental experience |
| W6 | "Portability: does the SSC rebuild itself where rules fail?" | not shown | DRAM+PFS setup: re-measured experience vs. unchanged rules (queued) |
| W7 | "How are module decisions made? when is the cache/indexer/compressor used?" | table + SSC figure | Table `modules` per phase for WfCommons on all setups (generated when units finish) |
| W8 | "Statistical significance" | CIs over schedules; paired comparisons in summary.md | keep; S2 needs all 10 schedules (running) |
| W9 | "Synthetic data" | sizes/reuse from traces, formats synthesized | state clearly; optional: real 2MASS / 1000 Genomes samples |
| W10 | "One node type, single-node composition" | limitation | state; DRAM+PFS is a second storage configuration |

## Decision: what to add to the runs
1. Keep the WfCommons stream order: S2 (10 schedules, 1 GB) → S1 units incl. cold start → 0.5 GB grid
   → 0.5 GB units → DRAM+PFS grid → DRAM+PFS units; live 1 GB with async arms in parallel.
2. Do **not** add S2+SSC on WfCommons: 41 x 81 s does not fit a 1-hour job and the Gray-Scott result
   already answers W3.
3. After the PFS grid: if S1+SSC beats rules/lookup there, that is the paper's portability result;
   if not, report it and make the cold-start/incremental-experience story the honest limitation.

## Expected verdict once the queued results are in
If WfCommons on 0.5 GB / PFS shows S1+SSC ahead of rules and the live async run is within ~10% of the
best fixed setup while S2 blocking is ~10x slower: **weak accept / accept** (clear systems
contribution, strong methodology). If S1+SSC only ties the simple baselines everywhere: **borderline**,
and the paper should be reframed around decision speed + robustness rather than accuracy gains.

## Update 2026-11-25 03:30 (after WfCommons 1 GB offline + live schedule 201)

| # | Concern | New evidence | Status |
|---|---|---|---|
| W2 | end-to-end speed | live WfCommons s201: S1+SSC async 130 s, S2 blocking 1598 s (12.3x), S2 async 140 s with only 4/41 answers applied (10 windows late) | **answered** (1 schedule; 2 more running) |
| W1 | rules as good? | the rule set used live is 1.61-2.15x everywhere offline; the capacity-codec variant ties at 1 GB, fails on DL/HDF5 (1.62x) | **mostly answered**; PFS setup will test the variant |
| W4 | experience vs. window facts | WfCommons 1 GB cold start: 0% / 2.87x vs full SSC 100% / 1.006x | **answered** |
| W7 | how decisions are made | per-phase module table, SSC figure, agreement table (placement always follows evidence; cache is where the model breaks ties) | **answered** |

Revised verdict: **weak accept** if the 0.5 GB / PFS results and the S2 WfCommons accuracy land as
expected; the remaining risk is that at 1 GB every simple baseline is also perfect, so the paper must
lead with decision speed + robustness across settings, not raw accuracy.
