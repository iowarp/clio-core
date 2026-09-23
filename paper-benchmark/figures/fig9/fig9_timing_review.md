# Overall verdict

The figure does not currently support a literal “compute versus device I/O” interpretation. At best, it shows an adjusted elapsed time divided into:

- time covered by at least one instrumented storage-path interval or an added fsync duration; and
- everything else.

That is an occupancy classification, not a causal cost decomposition. For asynchronous arms, a two-segment compute/I/O stack is structurally incapable of showing the intended overlap honestly.

More seriously, `total` is not actually end-to-end wall-clock time after `read_s` and `h2d_s` are subtracted. It is a synthetic adjusted duration. Calling it end-to-end is incorrect.

## Defects ranked by impact on paper claims

1. **One run per arm:** invalidates rankings, speedups, and comparative claims.
2. **Subtracting read and H2D from elapsed time:** can distort totals and treats overlap incorrectly.
3. **Error-bound violations:** makes cuSZ and cuSZp3 invalid participants in a quality-matched comparison.
4. **Serial stacked representation of overlapped work:** invalidates compute/I/O attribution, especially for `+Async`.
5. **Possible `/tmp`/tier-NVMe contention:** confounds the measured benefit of tiering.
6. **Silent union-to-sum fallback and clamping:** can conceal exactly the accounting failures previously found.
7. **Scalar fsync addition:** may double-count time and can be hidden by clamping.
8. **RAM traffic labelled device I/O:** materially misstates several tiered bars, but does not change their totals.
9. **Asymmetric codec setup instrumentation:** invalidates setup-deducted comparisons, though not the main totals.
10. **Window clipping:** mathematically correct for an elapsed-time intersection, but it can conceal an incorrect completion boundary.

---

## Q1 — Is the stacked decomposition valid?

**Verdict: flawed.**

`non-I/O elapsed = total − io` is a residue. It includes:

- actual compression;
- codec construction and allocation;
- CPU and GPU idle time;
- scheduling and queueing;
- file metadata operations;
- logging and hashing;
- synchronization not included in an I/O interval;
- uninstrumented data movement;
- any malformed or missing instrumentation.

Renaming it from “compute” to “non-I/O elapsed” prevents one false claim, but it does not turn it into a direct measurement. The stacked shape still suggests two serial, mutually exclusive costs.

It is defensible only if described precisely as:

> Time with at least one instrumented I/O interval active, versus time with none active.

Even that requires fixing the fsync, RAM, fallback, clipping, and adjusted-window issues below.

### Smallest honest change

If no additional instrumentation is available:

- Retain the total bar.
- Remove the stacked decomposition.
- Show I/O-interval coverage as a separate marker or adjacent, non-additive bar.
- Rename it “wall time covered by instrumented storage-path calls.”

To measure compute directly, timestamp CPU computation and GPU execution using CPU monotonic timestamps plus GPU events. Then calculate explicit states:

- compute only;
- I/O only;
- compute and I/O overlapping;
- neither or uninstrumented.

Those four states can form an honest stack because they are mutually exclusive.

**Claim impact:** High for any claim about where time is spent; potentially low for total-time rankings if the adjusted totals themselves are later validated.

---

## Q2 — Does subtracting the I/O union from wall time type-check?

**Verdict: flawed as a compute decomposition; sound only as an occupancy complement.**

Seconds can be subtracted from seconds, but the semantics do not follow. If compute runs during an I/O interval, the entire overlapping interval is assigned to the pale segment and none of it appears in the solid segment. Thus:

- the pale segment is not the causal contribution of I/O;
- the solid segment is not total compute;
- neither segment answers how much faster the arm would become if one activity were removed.

Overlap is certain by design for the `+Async` arms. It is also plausible for every `+Tier` arm because tier draining is described as periodic, but the prompt does not establish whether those drains run concurrently with compression. Untiered arms may also overlap work across workers or chunks. The single-threaded cuSZ setup trace establishes only that setup spans do not overlap one another; it does not establish that setup does not overlap writes.

### Required evidence

Produce timestamped unions for compute \(C\) and I/O \(I\), then report:

- `compute_only = |C \ I|`;
- `io_only = |I \ C|`;
- `overlap = |C ∩ I|`;
- `unclassified = total − |C ∪ I|`.

For multiple devices, also report per-device activity separately rather than forcing RAM, NVMe, and Lustre into one class.

**Claim impact:** High for the `+Async` explanation and all compute/I/O balance claims. Total rankings are unaffected by this particular defect if `total` is otherwise correct.

---

## Q3 — Is the pale segment actually device I/O?

**Verdict: flawed. Both the label and the device-level interpretation are wrong.**

RAM-tier writes are memory copies, not device I/O. Their contribution to the current union is material:

| Arm | RAM contribution added beyond file-only union | Fraction of all-tier union |
|---|---:|---:|
| cuSZp3+Tier | 0.387 s | 17.5% |
| NP+Tier+Async+Lossy | 0.451 s | 14.2% |
| nvCOMP+Tier | 0.253 s | 1.7% |

Removing RAM rows would transfer those durations from the pale segment to the residue without changing total time.

The union itself is correct if the intended metric is “time during which any storage-stack transfer was active.” Counting concurrent RAM and NVMe intervals once is necessary for wall-clock coverage. It is wrong for device utilization or work: concurrent activity on two resources must remain visible as two resource timelines or an explicit overlap state.

There is a second naming problem: a timer around a buffered write call measures storage-path call latency, not necessarily physical device activity. Much of the physical write can occur later and be observed only as writeback or fsync waiting.

### Smallest honest change

Either:

- rename the segment to “instrumented storage-path activity” and disclose that it includes RAM copies; or
- exclude `tier=ram` and call the remaining result “file-tier call coverage.”

For analysis, show RAM, NVMe, and Lustre separately, including their intersections.

**Claim impact:** Medium. It changes the reported composition but not the total or lower-is-better ranking.

---

## Q4 — Is fsync double-counted or misplaced?

**Verdict: cannot tell from the supplied evidence. The implementation is unsafe.**

Adding `fsync_s` as a scalar is valid only if all of the following hold:

1. the fsync interval starts after every logged write interval ends;
2. it is not already represented in `io.csv`;
3. no asynchronous flush remains active concurrently;
4. its entire duration lies inside the same timing window.

The prompt says the barrier is enforced, but that does not prove temporal disjointness. For a tiered asynchronous drain, fsync may wait while a flush thread still has a logged write active. In that case, `union(write intervals) + fsync duration` double-counts the intersection.

The clamp can hide this defect: if `io_s > total_s`, setting the solid segment to zero produces a plausible-looking bar rather than exposing failed accounting.

### Smallest honest change

Record `fsync_start_ns` and `fsync_duration` and insert it into the same interval union. Also:

- tag it by destination tier;
- assert that accounting never requires clamping;
- make any interval crossing or duplicate instrumentation a fatal measurement error.

**Claim impact:** Potentially high for segment composition, particularly tiered arms; likely no impact on total unless fsync is also mishandled in the total timer.

---

## Q5 — Are read and H2D removed correctly and only once?

**Verdict: cannot tell whether they are duplicated; the subtraction method is flawed regardless.**

The prompt does not provide row provenance for `io.csv`. It must be established that:

- input reads never emit `file` rows;
- H2D input staging never appears in a bdev write interval;
- output timers that include GPU copies include D2H only, not H2D;
- read and H2D spans do not overlap one another.

If read or H2D activity appears in `io.csv`, the adjusted total excludes it while the pale segment still contains it. The subsequent residue subtraction effectively charges it twice and may trigger the clamp.

There is another union error: `read_s` and `h2d_s` are subtracted independently. If input reading and H2D staging are pipelined, the intersection is subtracted twice. The correct excluded duration would be `|read ∪ H2D|`, not `read_s + h2d_s`.

More fundamentally, subtracting an overlapping H2D interval does not remove H2D’s causal effect. H2D can contend for copy engines, PCIe bandwidth, GPU memory bandwidth, CPU time, and pinned memory while other work proceeds.

The resulting `total_s` is therefore not end-to-end wall-clock time.

### Smallest honest change

The clean solution is to define a real timing boundary after input preparation and start every arm from the same state. Otherwise:

- report inclusive elapsed time as the primary metric;
- report read and H2D coverage separately and non-additively;
- construct a timestamped union of excluded phases if an adjusted diagnostic is retained;
- filter `io.csv` by operation type and request identity.

**Claim impact:** High because this can change totals, not merely their coloring.

---

## Q6 — Is the `/tmp` versus tier-NVMe comparison fair?

**Verdict: cannot tell from this; the design contains a potentially major confound.**

First establish that `/tmp` and the tier file resolve to the same block device using mount and block-device information. “Node-local NVMe” does not prove they are the same device.

If they are the same device and input reads overlap output spilling, tiered arms experience mixed read/write contention while untiered arms read from NVMe and write to Lustre. Expected effects include:

- longer NVMe write intervals;
- longer input reads;
- queueing and latency amplification;
- page-cache and writeback interference.

At the storage-device level, this normally penalizes tiered arms. Therefore, removing the contention would likely strengthen rather than explain large tiering wins. However, subtracting read time can create non-intuitive accounting effects, so the net bias on adjusted `total_s` cannot be determined analytically.

If all input files are fully read before output processing begins and the device is quiescent, simultaneous device contention may be zero. Cache state and memory pressure can still differ.

### Smallest honest change

Run one paired control with input held in tmpfs or RAM, which easily fits within 128 GB, or place input on a separate physical device. Collect per-device utilization and queue-depth traces.

The possible bias ranges from essentially zero to device saturation; it cannot be quantified from the current table.

**Claim impact:** High for claims about the benefit of `+Tier`. It cannot currently explain or validate a specific magnitude.

---

## Q7 — Is setup instrumentation biased?

**Verdict: flawed.**

`setup_min = 0` means two different things:

- for cuSZ, setup was measured and had a value;
- for other codecs, setup was not measured.

Encoding “unmeasured” as zero is false data.

The main total still includes setup for every codec, so this does not by itself bias the ordinary total-time ranking. It does invalidate:

- the `--deduct-setup` comparative chart;
- statements that other codecs have no setup overhead;
- comparisons of setup fractions across codecs.

For cuSZp3, an upper bound of 1.6 ms per chunk corresponds to as much as:

\[
3354 \times 1.6\text{ ms} = 5.37\text{ s}
\]

The elapsed union may be smaller under concurrency, but 5.37 s is material against 9.8–15.7 s bars.

### Smallest honest change

Immediately:

- encode unmeasured setup as `NA`, not zero;
- suppress setup-deducted cross-codec charts.

For a valid comparison, instrument equivalent lifecycle operations for every codec using timestamps and unions. Any setup-deducted total must also account for setup/I/O overlap; simply subtracting the setup union from elapsed time is not valid when they intersect.

**Claim impact:** High for setup-deducted results; low for the unmodified total bars.

---

## Q8 — Are any differences statistically established?

**Verdict: flawed. No comparative difference is statistically established with one run per arm.**

The 22% Baseline swing is direct evidence that system variability is large. It is not a confidence interval for every arm, so it cannot be used mechanically to declare differences greater than 22% significant. Every codec and storage configuration may have a different variance.

Clearly unsupported rankings include:

- 8.8 versus 9.0 s;
- 9.0 versus 9.8 s;
- 15.7 versus 17.3 s;
- 17.3 versus 17.7 versus 17.9 s;
- 20.1 versus 20.9 s;
- 20.9 versus 23.8 s;
- 68.9 versus 75.1 s.

Large observed gaps—such as approximately 42 s for Baseline versus 8.8–23.8 s for several compressed arms—are stronger hypotheses, but they still lack arm-specific and paired uncertainty. A single pathological run can occur for any arm on a shared filesystem.

The prompt does not include the manuscript’s prose or caption, so specific paper claims beyond the table cannot be audited. Any exact speedup, winner, ranking, or statement that one arm outperforms another is unsupported.

### Smallest honest change

Use randomized or rotated paired blocks with at least five repetitions per arm. Report:

- every observation;
- median and dispersion;
- paired differences within blocks;
- confidence intervals;
- seeds and codec-selection traces for nondeterministic NP arms.

**Claim impact:** Critical. This affects virtually every comparative conclusion.

---

## Q9 — May bound-exceeding arms remain in the lossy comparison?

**Verdict: flawed.**

An arm that does not satisfy the stated quality constraint is not feasible under that constraint. Its time and compression ratio cannot compete directly with compliant arms.

The cuSZ overshoot is only 0.7%, but it is systematic rather than an isolated numerical accident. Whether 0.7% is scientifically acceptable is a policy decision; it cannot silently be treated as compliance with a strict `1e-3` bound. The magnitude of cuSZp3’s exceedance is not supplied and must be reported.

### Smallest honest change

Choose one:

- request a slightly tighter codec tolerance, verify every chunk meets `1e-3`, and rerun timing and ratio;
- move noncompliant arms to a separate panel;
- retain them with conspicuous “constraint violated” styling and exclude them from winners and rankings;
- compare codecs at measured distortion rather than requested parameters.

Do not replace `exceeded` with a footnote while continuing to rank those arms as though they were feasible.

**Claim impact:** High for all quality-normalized rankings involving cuSZ or cuSZp3.

---

## Q10 — Is clipping to the timing window correct?

**Verdict: sound geometrically, but it must not conceal an invalid completion boundary.**

For a wall-clock interval `[t0,t1]`, only an operation’s intersection with that interval belongs in the elapsed-time partition. If a write starts inside the window and finishes after `t1`, the tail after `t1` is not part of wall-clock time inside `[t0,t1]`. It does not belong in either stack segment.

Therefore, the premise that the clipped tail “still has cost wall-clock time inside total” is incorrect if `total = t1 − t0`. The portion before `t1` is retained; the portion afterward is outside total.

The real defect would be stopping the clock before required output work completes. If durability is part of the benchmark contract, the final barrier should make post-window output intervals impossible. A crossing interval then indicates:

- the barrier did not cover that operation;
- the asynchronous worker was not joined;
- clocks are inconsistent;
- the interval was attributed to the wrong run.

Clipping such an interval makes the bar look valid while hiding the lifecycle failure.

There is also a separate inconsistency: intervals are clipped against the original physical window, while `total_s` is subsequently shortened by subtracting read and H2D. The interval domain and adjusted total no longer describe the same interval.

### Smallest honest change

Keep clipping for legitimate boundary intersections, but:

- record the unclipped duration;
- fail the run if any output write crosses `t1` after the durability barrier;
- fail if an arm’s interval begins before `t0`;
- use an actual contiguous timing window rather than comparing raw-window intervals with an arithmetically shortened total.

**Claim impact:** Usually diagnostic, but critical if crossing intervals exist.

---

## Additional accounting defects

### Silent fallback from union to sum

**Severity: potentially claim-changing; current activation cannot be determined.**

If the start column is missing or unusable, `phase_union` silently changes from elapsed union to summed work. Under concurrency, those are different physical quantities. This is exactly the previously identified failure mode.

The helper must fail closed. It should never silently produce a bar using a different metric.

Minimum fix:

- require a valid timestamp for every included row;
- report parsed, rejected, and total row counts;
- abort on missing or malformed timestamps;
- store the aggregation mode in the output CSV.

### Clamping hides invalid accounting

**Severity: claim-changing for the decomposition.**

If `io_s > total_s`, that is evidence of double counting, mismatched windows, malformed timestamps, or an invalid scalar addition. Setting the residue to zero converts an accounting failure into a plausible chart.

Minimum fix: abort the arm whenever the unclamped value falls outside a small, declared rounding tolerance. Preserve and report the raw values.

## Recommended replacement figure

Use one primary bar or point per arm for directly measured elapsed time. If read and H2D are deliberately excluded, enforce that exclusion with real timing boundaries and common starting state rather than subtraction.

Next to the elapsed result, show either:

1. a four-state overlap stack—compute-only, I/O-only, overlap, and unclassified; or
2. separate non-additive activity bars for GPU execution, RAM copy, NVMe/Lustre calls, and fsync.

Do not use a two-piece “compute plus I/O” stack for asynchronous arms. The system’s advantage is overlap; collapsing overlap into one side of a serial stack misrepresents the mechanism the figure is supposed to demonstrate.
