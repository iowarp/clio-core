# Self-review as an ICS reviewer (draft of 2026-11-26, fixed-stack framing)

The paper is now framed as Luke suggested: CLIO has no automatic composition; the competitor is the
fixed I/O stack (the default, and a stack hand-tuned with hindsight); S2, rules and the model-free
lookup are design-justification baselines. STELLAR (SC'25) is positioned as complementary offline
parameter tuning.

**Score now: borderline → weak accept**, conditional on the live and DRAM+PFS results below.

## Strengths a reviewer will credit
- A real gap: composable stacks (CLIO, Mochi, ADIOS2, HDF5 VOL, Hermes) have no chooser; tuners
  (STELLAR, autotuners) produce one static config between runs, need the target's trace and reruns.
- Honest baselines: default as shipped, default + declared indexer, and a hindsight-tuned fixed
  stack (an upper bound for any static configuration, STELLAR included).
- Leave-workload-out: the SSC never measures the target workload; the hand-tuned stack does.
- Careful protocol: practice/test split, CIs over schedules, noise and threshold sensitivity, v10
  reported next to v12 with a disclosure of how v12 was derived.
- Cost answered: learning curve (3 measured phases, ~20 min, suffice at 1 GB).

## Weaknesses a reviewer will raise

| # | Concern | Status | Action |
|---|---|---|---|
| R1 | "On WfCommons the default + indexer is already optimal; where is the gain?" | stated honestly | lead with Gray-Scott / DL-HDF5 / DRAM+PFS, where fixed stacks fail; WfCommons = "no harm" |
| R2 | "Part of the gain over the default is the declared indexer" | both default columns reported | done |
| R3 | "Loses to a fixed stack on DL/HDF5 4 GB" | cause found (cache evidence from GS neighbours is inverted) | stated as limitation; a more diverse experience set is the fix (not run) |
| R4 | "Expert rules + capacity codec beat you on GS 4 GB" | in the table | stated in the text |
| R5 | "Live evidence is one schedule" | 202 mostly done; v12 + default arms queued for 201–203 | **needed for accept** |
| R6 | "Is the SSC's value only on fast NVMe nodes?" | DRAM+PFS grid 23/24 (exclusive) | kg_vcf is ~5x slower with the default there: **the strongest fixed-stack failure; needed** |
| R7 | "v12 was designed after seeing test results" | disclosed; v10 reported throughout | done |
| R8 | "Why an LLM if the lookup is as good?" | lookup brittle at 4 GB (57% vs 81%); v12d = decisive evidence + model on near-ties | done; learning-curve model arm queued |
| R9 | "Single node type" | limitation | Chameleon second platform if an allocation exists |
| R10 | "Decision latency 1.0 s (v12) vs 0.4 s (v10)" | earlier campaign vs this one | check on the same lane (pending) |
| R11 | "Bibliography: many TODO entries" | TODO-marked | verify every entry before submission |

## Verdict
With R5 and R6 in (live default vs. S1+SSC on three schedules, and DRAM+PFS showing a large
fixed-stack failure that S1+SSC avoids): **weak accept**. Without them, a reviewer can say the
measured gains over a sensible fixed stack are small except on DL/HDF5 1 GB and Gray-Scott 1 GB.
