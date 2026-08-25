# Debate round 2 — adversary's settlement of D1–D8

Against `../investigator/DEBATE_R1.md` and the revised `FINDINGS.md`. They conceded almost
every attack I made, several of them (F7's saturated-regime examples, F8's confidence claim,
F5's in-distribution definition, F9's missing knob) more completely than I would have. Round 2
runs the other way: of the eight open points, **I concede four outright (D4, D5, D6, D7-in-part)
and one in part (D3)**, and two of those concessions are bugs in my own round-1 evidence that I
found by publishing the code they asked for.

**Scoreboard of this round, up front, so the concessions are not buried:**

| point | verdict | who moved |
|---|---|---|
| D1 | ratio-MAPE OR gate: **AGREED — not a net win** (regret 0.3043 vs 0.1367 idle) | **me** — my own idle run refutes my round-1 claim |
| D2 | replaced gate + `n_x_oob==0`: **AGREED — both design properties hold; still not a net win** (0.1741 vs 0.1367) | both; their prediction half right |
| D3 | F7 phase 2: **AGREED — PARTIAL, not REFUTED**; my 0.910 was a sign flip | **me** |
| D4 | random-32 null: **CONCEDED — my null was stratified by algorithm** | **me** |
| D5 | `f3` reachability: **CONCEDED — reachable; my inference was wrong** | **me** |
| D6 | non-float32 scope: **CONCEDED — both paths, not device-only** | **me** |
| D7 | exploration helps adaptation: **AGREED — I withdraw "does not rescue it"** | **me** |
| D8 | residual table magnitude: **AGREED — tail-carried; their summary is the better one** | **me** |

I concede six of the eight outright and the other two are joint adjustments. Two of the six are
bugs in my own round-1 evidence that surfaced only because they asked me to publish the code.

**One loose end from round 1 also closes here.** My round-1 ctest run was 16/17 with a
`cudaMemGetInfo` failure I attributed to the shared GPU. Re-run on an idle device:
**100 % tests passed, 0 failed out of 18**. Their 17/17 was right and the joint statement is
"17/17 on an idle device, with a known shared-GPU false failure in
`ctp_neuropress_sgd_choreography_parity`".

**GPU discipline this round.** All timing runs are on my 28-regime × 20-chunk schedule
(560 chunks, seed 1234, fresh weights per run) through `scripts/runsR.sh`, which (a) runs
**one** `np_eval_exec` at a time and (b) blocks on `nvidia-smi --query-compute-apps` until the
device is empty before every arm — my own L4 finding is why. Every arm is scored against **one
common measured table** (`R0`'s) with `scripts/r2.py`; **no number below is self-scored.**

Their `h0`/`h0b` result (identical picks, self-scored top-1 0.122 vs 0.000) is correct and I
adopt the method. One qualification in my own favour, measured: on my 28-regime schedule the
self-scoring error is small — `R0` and `R0b` are byte-identical replicates and agree exactly
(common-GT top-1 0.361, regret 0.1367, bytes 0.67837 both), and `R0b` scored against its *own*
table gives regret 0.1379 vs 0.1367, a 0.0012 difference. Their 3-block stream is far more
sensitive to it than my 28-regime one, because with 3 regimes the true-best action flips much
more readily under timing noise. Common ground truth is still the right method; the round-1
top-1 column of my L battery is less damaged than their `h0`/`h0b` pair implies.


## D1 — does the ratio-MAPE gate make a fresh process *better* than not learning? **VERDICT: AGREED — no. It is not a net win, and on my schedule it is clearly worse.**

Their `h10` vs `h0` said worse (regret 1.2525 vs 1.0918). I ran the same comparison on my
28-regime schedule with an **idle** GPU on every arm and a **common ground truth** (`R0`'s
measured table). All eight arms ran alone on the device — median `act_ct` over all rows is
2.20-2.27 ms for every one of them, against 8.25 ms for a contended round-1 run, so the load
confound that ruined my round-1 busy group is gone.

| arm | gate | SGD calls | % of firings on an OOB chunk | top-1 | **mean regret** | median regret | bytes | median `act_ct` |
|---|---|---|---|---|---|---|---|---|
| **R0** | none (no learning) | 0 | — | **0.361** | **0.1367** | 0.0138 | **0.67837** | 2.265 |
| R1 | cost (shipped) | 41 | **98 %** | 0.268 | 0.1373 | 0.0138 | 0.68301 | 2.234 |
| **R2** | **ratio OR cost** (shipped `CLIO_NEUROPRESS_SGD_ON_RATIO`) | 177 | 64 % | 0.329 | **0.3043** | **0.0048** | 0.67980 | 2.250 |
| R3 | ratio only (replaced) | 205 | 67 % | 0.177 | 0.4692 | 0.0347 | 0.69680 | 2.227 |
| R4 | ratio only **+ `n_x_oob==0`** | 52 | **0 %** | 0.323 | 0.1741 | 0.0289 | 0.69733 | 2.239 |
| R6 | cost **+ `n_x_oob==0`** | **1** | 0 % | 0.289 | 0.1442 | 0.0236 | 0.68556 | 2.204 |
| R5 | cost + explore K=7 (see D7) | 50 | 92 % | 0.168 | **0.1277** | 0.0329 | 0.69979 | 2.216 |

**My round-1 claim is refuted by my own run.** I presented the ratio gate as "the only learning
configuration that does not cost top-1". On an idle GPU against an idle baseline it costs mean
regret badly: **0.3043 vs 0.1367, 2.2x worse**, and never crosses over (its cumulative regret is
above the baseline's for the whole run; Q3 alone is +0.7232). Their `h10` result was right and
mine on the busy group was reading a contention artefact. **I concede D1.**

One nuance that is mine and that their stream did not surface, and it cuts *for* the ratio gate
on the typical chunk: R2 has by far the **best median regret of any arm, 0.0048 against the
baseline's 0.0138** — it improves ~2/3 of chunks and destroys a handful. That is the F1/F14 tail
signature again, and it means "worse" here is a mean-regret statement, not a
most-chunks statement. It does not change the verdict: a policy that triples mean regret is not
deployable, and bytes (0.67980) are no better than not learning.

**AGREED SENTENCE.** *"On an idle GPU, scored against a common measured table, no trunk-SGD
configuration makes a fresh process better than not learning on mean regret and bytes together.
The shipped ratio-MAPE OR gate (`CLIO_NEUROPRESS_SGD_ON_RATIO`) is worse than not learning on
both corpora tested — regret 1.2525 vs 1.0918 on the 3-block stream and 0.3043 vs 0.1367 on the
28-regime schedule — although it has the best median regret of any arm on the latter (0.0048 vs
0.0138). Its value is that its trigger is load-invariant, not that it improves selection."*


## D2 — a *replaced* (not ORed) ratio gate plus an `n_x_oob == 0` precondition. **VERDICT: AGREED — feasible (built, 10 lines), both design properties proved, and still not a net win.**

**Feasibility: two knobs, ~10 lines, in the gitignored harness.** I added them to
`parity/np_eval.cu` (`git status --porcelain` on the parity dir is empty, so nothing tracked
moved):

* `NPE_GATE=ratio_only` — one `else if` branch: `gate_open = (rmape > sgd_thresh);`, i.e. the
  ratio MAPE **replaces** the cost MAPE instead of being ORed onto it.
* `NPE_GATE_OOB_PRECOND=1` — one line after the gate block:
  `if (gate_oob_precond && raws[prim].n_oob > 0) gate_open = false;`

Both default to the shipped behaviour, so every earlier run is unaffected. In the runtime the
same change would be `neuropress_bridge`'s OOB count plus replacing the `||` at
`compressor_runtime.cc:1570-1577` with the ratio term — comparable size, but it needs the
predictor to expose an out-of-bounds count that Clio does not currently surface outside the
harness, which is the only real implementation cost.

**Deterministic proof of both claimed properties** (my own sweep over f0a's 590 primary rows,
each gate computed exactly as the runtime does, measured ct scaled by a device-load factor `k`;
OOB base rate 0.397):

| k | cost gate | ratio gate | cost **OR** ratio | **ratio_only ∧ `n_x_oob==0`** | cost: %OOB | ratio: %OOB |
|---|---|---|---|---|---|---|
| 0.5x | 0.234 | 0.475 | 0.476 | **0.163** | 0.99 | 0.66 |
| **1.0x (idle)** | 0.239 | 0.475 | 0.478 | **0.163** | 0.98 | 0.66 |
| 2x | 0.317 | 0.475 | 0.607 | **0.163** | 0.58 | 0.66 |
| 4x | 0.408 | 0.475 | 0.678 | **0.163** | 0.54 | 0.66 |
| 8x | 0.683 | 0.475 | 0.803 | **0.163** | 0.50 | 0.66 |
| **14x** | **0.997** | 0.475 | **0.997** | **0.163** | 0.40 | 0.66 |
| 20x | 1.000 | 0.475 | 1.000 | **0.163** | 0.40 | 0.66 |

This reproduces the investigator's F9 table exactly (0.239 / 0.475 / 0.478 idle; 0.997 / 0.475 /
0.997 at 14x) and adds their proposed row. Both properties they predicted hold **by
construction and are now measured**: the combination is **load-invariant** (0.163 at every `k`,
where the shipped OR knob runs 0.478 → 0.997) and **never trains on an extrapolation** (0 % OOB,
against the cost gate's 98 % and the ratio gate's 66 %). It also trains on a useful amount of
data — 0.163 × 560 ≈ 91 chunks, roughly twice the shipped cost gate's idle firing count.

**Live arms, idle GPU, common ground truth (the D1 table above).** Two arms test the proposal:
`R3` is the replaced gate alone, `R4` is the replaced gate **plus** the `n_x_oob == 0`
precondition.

| arm | SGD calls | % on OOB chunks | top-1 | mean regret | median | bytes |
|---|---|---|---|---|---|---|
| R0 no learning | 0 | — | **0.361** | **0.1367** | 0.0138 | **0.67837** |
| R1 shipped cost gate | 41 | 98 % | 0.268 | 0.1373 | 0.0138 | 0.68301 |
| R2 shipped OR knob | 177 | 64 % | 0.329 | 0.3043 | **0.0048** | 0.67980 |
| **R3 ratio only (replaced)** | 205 | 67 % | 0.177 | **0.4692** | 0.0347 | 0.69680 |
| **R4 ratio only + `n_x_oob==0`** | 52 | **0 %** | 0.323 | **0.1741** | 0.0289 | 0.69733 |
| **R6 cost + `n_x_oob==0`** | **1** | **0 %** | 0.289 | **0.1442** | 0.0236 | 0.68556 |

**VERDICT: AGREED on the two design properties, AGREED that it is still not a net win — so their
prediction is half right and I say so in both directions.**

* Replacing the cost gate **without** the precondition is the **worst** arm I measured
  (0.4692, 3.4x the baseline): it fires 205 times, 67 % of them on extrapolations. Replacement
  alone is not the fix.
* Adding the precondition does exactly what they predicted: firings **205 → 52**, OOB share
  **67 % → 0 %**, and regret **0.4692 → 0.1741** — a 2.7x repair. Both claimed properties hold,
  live and deterministically.
* But 0.1741 is still worse than not learning (0.1367) and worse than the shipped cost gate
  (0.1373), and it costs 2.8 % bytes. So the combination is **the best *replaced*-gate design and
  still not deployable as a win**.
* The single most informative arm is one neither of us proposed: **`R6`, the shipped cost gate
  with the same `n_x_oob == 0` precondition, fires once in 560 chunks** (41 → 1). That is the
  cost gate's 98 %-OOB property stated as a policy: forbid training on extrapolations and the
  shipped trigger has essentially nothing left to fire on. It is the second-best learning arm
  (0.1442) purely because it barely learns.

**AGREED SENTENCE.** *"An `n_x_oob == 0` precondition delivers both properties predicted for it —
it makes the trigger fire on zero extrapolation chunks by construction and, combined with the
load-invariant ratio MAPE, gives a gate whose rate is 0.163 at every device-load factor from 0.5x
to 20x. It repairs the replaced-ratio gate from 0.4692 to 0.1741 mean regret. It does not make
trunk SGD a net win: 0.1741 against 0.1367 for not learning. Applied to the shipped cost gate it
reduces firings from 41 to 1 in 560 chunks, which is the OOD-detector finding restated as a
policy."*


## D3 — is F7 refuted for the batched phase-2 path? **VERDICT: AGREED — PARTIAL, not REFUTED. I withdraw "REFUTED".**

Their objection is correct on method. My round-1 A1 varied sample 0's ct label over
2.70 / 2.40 / 2.00 / 1.60 / 0.80 against a prediction of **1.984**, so the last two labels sit on
the *other* side of zero error. The `cos = 0.910` I reported is therefore the **sign-flipped**
case, which is exactly what F7 predicts, not a counterexample to it. Re-run with every label on
the same side of the prediction (`np_adv.cu` **A1c**, batched; **A1d**, single-sample control;
**A1e**, the harder test where sample 0 is made the *most*- then the *least*-wrong member of an
otherwise identical same-sign batch):

| probe | ct labels (prediction = 1.984) | `cos(ΔW₀, ΔWᵢ)` | ‖ΔW‖ ratio |
|---|---|---|---|
| **A1d** single sample, same sign | 1.60 / 1.20 / 0.80 / 0.40 / 0.20 | **1.000000000** ×4 | 1.00000009 |
| **A1c** 7-sample batch, same sign | same | 0.983220 / 0.972210 / 0.972210 / 0.972210 | 0.9999976-0.9999981 |
| **A1e** 7-sample batch, sample 0 made most-wrong (0.20) vs least-wrong (1.90), others fixed at 1.90 | — | **0.989431** | 1.0000023 |
| (round 1) A1, batch, **sign flip included** | 2.70 … 0.80 | 0.999482 … **0.909978** | — |

So:

* **Single sample: exact.** `cos = 1.000000000` to nine decimals across an 8x range of label
  magnitude. F7's core claim is confirmed by an independent probe at an independent operating
  point, and this is the regime phase 1 runs in.
* **Batch: a real but small leak.** With every label on the same side, a 8x change in one
  sample's label still moves the direction by **1.7-2.8 %** (`cos` 0.972-0.989) and moves the
  step size by **nothing** (‖ΔW‖ equal to 6 s.f., because the trust step is saturated). My leak
  is larger than their 0.2 % — different operating point and a wider label spread — but the
  order of magnitude is the same and neither supports "REFUTED".
* My round-1 `cos = 0.910` was the sign-flipped case and should never have been the headline.

**AGREED SENTENCE.** *"For a single sample — the phase-1 path, and the shipped path since
exploration is default-off — the SGD update direction depends only on the sign pattern of the
per-output errors and is invariant to their magnitudes exactly (cos = 1.000000000). For a batch
(phase 2, up to 7 samples), `out_grad` accumulates across samples before the per-output clip, so
relative sample magnitudes do leak into the direction; measured with all labels on the same side
the leak is 0.2-2.8 % of direction and nil in the step size. F7 is therefore PARTIAL, not
REFUTED: the mechanism is exact where it matters and approximate where it does not."*


## D4 — the "random 32-way grouping removes 21 %" null. **VERDICT: CONCEDED. My null was broken, and I can name the bug.**

They asked me to publish the code. Here it is, and it convicts me. My round-1 null was:

```python
rk = {}                                   # random 8-way label per row, seed 7
for r in rows: rk.setdefault(key(r), random.randrange(8))
...
('RANDOM-32 (null)', lambda r: (rk[key(r)], int(r['action']) % 4))   # <-- the bug
```

I built the 32-way null as *random-8 crossed with `action % 4`*. At eb=0,
`action = algo + 16·shuffle`, so **`action % 4 == algo % 4` on 9440/9440 rows** (I checked).
My "random" null was therefore **stratified by algorithm** — it leaked exactly the signal it was
supposed to exclude. Re-run with a genuine `random.randrange(32)` per row, everything else
identical (median offsets, seed 7, same 9440 rows):

| grouping | groups | ct median \|log err\| | change |
|---|---|---|---|
| per-algorithm | 8 | 1.210 → 0.499 | **−59 %** |
| per-action | 16 | 1.210 → 0.442 | **−63 %** |
| RANDOM-8 (proper) | 8 | 1.210 → 1.242 | +3 % |
| **RANDOM-32 (proper)** | 32 | 1.210 → **1.225** | **+1 %** |
| my round-1 "RANDOM-32" (= random-8 × `action%4`) | 32 | 1.210 → 0.960 | −21 % |

Their null is right (+1 % here, +2 % on theirs); mine was not. It is **not** a mean-vs-median
artefact — I used medians, as they did; for the record, mean offsets make every column worse
(per-algo −2 %, random-8 +37 %), which is what a heavy tail does and why medians are correct.

**AGREED SENTENCE.** *"A random grouping of the same cardinality removes no ct log error at 8,
16 or 32 keys (+1 % to +3 %), so the per-algorithm reduction (−59 %) and the per-action
reduction (−63 %) are real signal, not degrees-of-freedom inflation. The reason to prefer a
per-algorithm table is not DOF but out-of-sample generalisation: fit on `f0a` and scored on
`f0b`, per-algorithm gives balanced regret 0.0330 and per-action 0.0530."*

---


## D5 — is the `f3` configuration (ratio-only cost weights + learning) reachable? **VERDICT: CONCEDED. It is reachable; my inference was wrong.**

Source check, all three links:

1. `neuropress_bridge.cc:66-67` — `o.ct = read("CLIO_NEUROPRESS_COST_W_CT", 1.0, &seen);`
   `o.dt = read(... "_W_DT" ...)`. `read()` accepts any value `strtod` parses, and there is
   **no positivity guard** on `ct`/`dt` — the only guard in the function is
   `if (!(o.cap > 0.0)) o.cap = 100.0;` for the cap. So `0` is accepted.
2. `compressor_runtime.cc:1447-1449` — the gate's cost is
   `{best_mode ? 0.0 : kCw.ct, best_mode ? 0.0 : kCw.dt, kCw.io, ...}`. With best mode **off**
   and the env weights zeroed, the gate's cost is `io` alone — a ratio-only objective.
3. `compressor_runtime.cc:1575-1577` — `np_will_train = (np_cost_gate || np_ratio_gate) &&
   !config_.neuropress_best_mode_`. Best mode is a **separate** flag; zeroed weights do not
   touch it, so training stays enabled.

So a deployment can run a ratio-only objective *with* learning, and `NPE_COST=ratio
NPE_MODE=learn` models it. My round-1 claim that `--best` disables both SGD phases is correct
and they verified it; the **inference** I drew from it — that `f3` is therefore unreachable —
does not follow, because `f3` is not `--best`. I withdraw the "row invalid" strike.

**AGREED SENTENCE.** *"`--best` disables both SGD phases (`compressor_runtime.cc:1577`, `:2290`;
upstream `gpucompress_compress.cpp:705`). A ratio-only cost objective with learning enabled is a
different and reachable configuration, via `CLIO_NEUROPRESS_COST_W_CT=0` /
`CLIO_NEUROPRESS_COST_W_DT=0` with best mode off, and that is what the `f3` row measures."*

---


## D6 — is the non-float32 defect device-path-only? **VERDICT: CONCEDED. It is on both paths.**

`ComputeNeuroPressFeatures` (`data_stats_gpu.h:196-222`) special-cases `data_type == 2` twice —
device-resident float64 and host-resident float64 — and then falls through to:

```cpp
const size_t n = chunk_bytes / sizeof(float);
return ComputeCompressionFeatures(chunk, n, DataType::FLOAT32,
                                  out_entropy, out_mad, out_second_derivative);
```

`data_type` is not consulted again. Every non-float64 type — uint8, int32, int64, float16 — is
read as float32 on the **host** path exactly as on the device path, and their E10 numbers came
through this entry point. My round-1 narrowing ("the host path already honours `data_type_`")
was wrong: it honours it only for float64. F13 stands unnarrowed.

**AGREED SENTENCE.** *"Both the device path (`neuropress_selection.cc:47-49`) and the host path
(`data_stats_gpu.h:219-222`) read every non-float64 payload as float32; only float64 is
converted. A uint8 count field yields mad = 3.02e36 and a NaN or extreme feature collapses the
network to its bias vector, making selection the action-index tie-break (lz4) with no log line.
The missing guard — a finiteness/bounds check before ranking — is missing on both paths."*

---


## D7 — does phase-2 exploration help in-run adaptation? **VERDICT: AGREED in direction. It is the only trunk-SGD arm that beats not-learning on mean regret — and it is the worst arm on top-1, bytes and the median chunk.**

They were right that my round-1 L5 was contended and compared against an idle baseline. Re-run
idle-vs-idle on the same schedule (`R5` = `NPE_MODE=learn NPE_SGD_THRESH=0.30
NPE_EXPLORE_K=7`, 47 chunks explored, 50 SGD calls):

| arm | SGD | top-1 | **mean regret** | median | bytes | 1st half | 2nd half | crossover |
|---|---|---|---|---|---|---|---|---|
| R0 no learning | 0 | **0.361** | 0.1367 | **0.0138** | **0.67837** | 0.1620 | 0.1114 | — |
| R1 phase 1 only | 41 | 0.268 | 0.1373 | 0.0138 | 0.68301 | 0.1535 | 0.1210 | never |
| **R5 phase 1 + explore K=7** | 50 | 0.168 | **0.1277** | 0.0329 | 0.69979 | **0.1250** | 0.1305 | chunk 560 |

**I withdraw "phase-2 exploration does not rescue it either".** That sentence was based on a
contended run. Idle, `R5` is the **only** trunk-SGD arm of the seven whose mean regret is below
the no-learning baseline (0.1277 vs 0.1367, −6.6 %), and its first-half regret is the best of any
arm (0.1250 vs 0.1620) — the off-policy labels do help early, which is their B1 mechanism and
their A4 "on-policy trap" being relieved.

**Two qualifications that keep this from being an endorsement**, both from the same table:
its top-1 is the **worst** measured (0.168 vs 0.361), its bytes are the **worst** (0.69979,
+3.2 %), and its **median** chunk gets worse (0.0329 vs 0.0138). And the crossover on my
schedule is chunk **560** — the very last chunk — against their 5-17 on the 3-block stream, so
the "converges in tens of chunks" claim is strongly schedule-dependent.

**AGREED SENTENCE.** *"Feeding the learner phase-2 exploration samples is the only trunk-SGD
configuration measured that improves mean regret over not learning (0.1277 vs 0.1367 idle,
common ground truth; first-half regret 0.1250 vs 0.1620), which is the on-policy trap being
relieved by off-policy labels. It simultaneously produces the worst top-1 (0.168 vs 0.361), the
worst bytes (+3.2 %) and a worse median chunk (0.0329 vs 0.0138) of any arm, and its crossover
point is schedule-dependent — chunk 5-17 on a 3-block stream, chunk 560 on a 28-regime one. The
value of exploration for **adoption** (F6) is separate and much larger than its value as a
training signal."*


## D8 — the online residual table: size of the adaptation curve. **VERDICT: AGREED in sign and in mean/median regret; I concede the magnitude, and I add a caveat against myself.**

They accepted the mechanism and asked only that the magnitude be stated with both numbers. I did
more than that: I re-measured it **paired per chunk**, which removes the schedule from the
comparison entirely (same chunks, same ground truth, only the pick differs).

Whole-run, my 28-regime schedule, common ground truth (`R0`-method):

| | mean regret | median regret | bytes | half-ratio (2nd/1st) |
|---|---|---|---|---|
| L0 no learning | 0.1380 | 0.0138 | 0.67837 | 0.723 |
| M1 residual table (per-algo, ct+dt, α=0.3, **0 SGD calls**) | **0.0908** | **0.0082** | 0.68282 | **0.134** |
| their `h0` no learning | 1.0918 | 0.8070 | 0.90259 | 0.691 |
| their `h9` residual table (same config) | **1.0146** | 0.8068 | 0.91821 | **0.625** |
| their `h3b` residual table (per-action, ct) | **0.9570** | 0.8024 | 0.93897 | 0.646 |

Their objection was that my 5.4x came from a schedule whose halves differ in difficulty. That
turns out **not** to be the explanation — the two baselines have almost the same half-ratio
(mine 0.723, theirs 0.691), so the schedules are equally "easing". The real explanation is worse
for me, and it is my own round-1 tail caveat coming back:

| quartile | picks changed vs L0 | better | worse | net measured cost |
|---|---|---|---|---|
| Q1 (0-139) | 14/140 | 4 | 10 | −0.1 ms |
| Q2 (140-279) | 0/140 | 0 | 0 | 0.0 ms |
| Q3 (280-419) | 120/140 | 80 | 40 | **+4.3 ms (worse)** |
| Q4 (420-559) | 92/140 | 12 | **80** | **−118.1 ms** |

Across the whole run the table changes 226 picks and is **worse on 130 of them**. Its entire
mean-regret advantage is Q4, and inside Q4 it comes from a handful of chunks with very large cost
deltas (`const`, `drift`, `sizes` — where the ct head is most wrong). So my "5.4x adaptation
curve" is a **tail** result, exactly as my own round-1 caveat about mean regret warned; their
0.625/0.646 is the better description of the typical chunk. The median still improves on both
schedules, which is the part that is not tail-driven.

**AGREED SENTENCE.** *"An online per-algorithm log-residual table on the time heads only,
updated on-policy from measured outcomes with no trunk SGD at all, improves a fresh process
within its own lifetime on both schedules and with zero SGD calls: mean regret 0.1380 → 0.0908
and median 0.0138 → 0.0082 on the 28-regime schedule, 1.0918 → 1.0146 (per-algo) and → 0.9570
(per-action, ct only) on the 3-block stream. It costs bytes (+0.7 % / +1.7 % / +4.0 %), and its
mean-regret gain is tail-carried — it changes 226 picks on the 28-regime schedule and is worse
on 130 of them, with the whole net gain in the last quarter. Adding the ratio head to the table
makes it worse than not adapting at all (0.0576 → 0.0905 simulated; live `h6` 1.2348 vs 1.0918),
independently reproducing F3/M6."*


---

## Agreed gaps and bottlenecks

Both sides now sign these. Each line carries the single number that decides it.

1. **Absolute-magnitude `mad`/`d2` put ordinary scientific data hundreds of σ out of
   distribution, and one or both heads saturate above mad ≈ 40.** Predicted ratio spread over
   the 8 algorithms collapses **46.4x → 1.00x** between mad 0.0057 and mad 57.4.
2. **The ct head's dynamic range is ~21x too narrow, and its per-algorithm error is a constant.**
   Within-chunk measured max/min **96.8x** against a predicted **4.52x**; removing one
   per-algorithm log offset cuts median |log ct error| **1.210 → 0.499 (−59 %)** against a
   **+1 %** random-grouping null.
3. **The ratio head's error is per-chunk, not per-algorithm.** Per-chunk offset **−79 %**,
   per-algorithm **+3 %**. This is why supplying the true ratio alone makes the balanced pick
   *worse*.
4. **Which head is worth fixing is set by storage bandwidth, not by the model.** A perfect ct
   head is worth regret **0.0169 at 5 GB/s** and **0.2633 at 50 MB/s** — worse than the
   uncorrected model's 0.2297.
5. **The 100x ratio cap decides 43 % of lossy chunks by the action-index tie-break.** Raising it
   takes eb=1e-3 ratio-only regret **35.99 → 0.435**; ~25 % of chunks still tie with the cap
   gone, so the cap owns about 60 % of the problem.
6. **The model is close to blind to the byte-shuffle half of its own action space.** Shuffle
   improves the measured ratio on **89.3 %** of (chunk, algo) pairs; the model predicts an
   improvement on **43.4 %** (recall 0.458, and 0.333 on a held-out corpus).
7. **Chunk sizes outside [64 KiB, 4 MiB] are out of distribution in both directions.** Median
   ratio APE **1.63 at 1 MiB, 4.36 at 0.25 MiB, 46.2 at 8 MiB**; at 64 MiB the model predicts
   34.5 dB PSNR for a *lossless* action.
8. **The PSNR head cannot rank, and cannot represent its own label.** Exactly **1** distinct
   actual PSNR per chunk on **440/440** chunks, while the head is clamped to [0, 120] and the
   analytic label reaches **−283.6 dB** on 544/7040 rows.
9. **Non-float64, non-float32 payloads are misread as float32 on BOTH the device and host
   paths, silently.** A uint8 count field yields **mad = 3.02e36**; a NaN feature yields 1/8
   distinct predictions and selection degenerates to the lz4 tie-break with no log line.
10. **The phase-1 gate is not an error detector.** It is an OOD detector on an idle GPU
    (**93 %** of firings land on chunks with an out-of-bounds input, base rate 40 %) and a
    device-load detector on a busy one (gate rate **0.239 → 0.997** as measured ct scales 1x → 14x).
11. **Phase-1 learning is on-policy with a batch of one, so it cannot correct a wrong pick.**
    60 steps converge the *picked* action's ratio to the measured 1.251 and the pick never moves
    off a5 when the true best is a7.
12. **The trunk cannot travel far in one process.** 590 SGD steps move ‖W‖ by **0.90 %** — less
    than 226 steps do (0.1817 vs 0.2050) — against 1.77 for a coherent walk of the same length.
13. **One SGD step is sign-directed.** `cos = 1.000000000` within a sign class; magnitude enters
    only below trust-step saturation (‖ΔW‖ exactly linear in |d5[0]| between the 0.10 noise gate
    and ≈0.87) and, in a batch, only through relative sample weights.
14. **The 1 ms cost floor is load-bearing on this hardware.** It truncates **29 %** of measured
    compress times idle (7 % loaded); lowering it to 0.1 ms costs 0.061 top-1 and **+22 %** mean
    regret.
15. **Exploration is default-OFF, so what ships is the network's own top-1.**
    `neuropress_exploration_enabled_ = false`; offline, regret is **0.1572 at K=0**, 0.0708 at
    K=3, 0.0000 at K=31.
16. **On both corpora tested the model does not beat a fixed action out of sample.** Ratio-only
    bytes **0.6721 vs 0.5883** for `zstd`+4-byte-shuffle, which equals the per-chunk oracle to
    4 dp — though in-distribution under the balanced objective the gap is only **1.23x**.
17. **Measurement caveat both sides accept.** Every balanced-cost number is conditioned on an
    idle GPU (codec times inflate 2-30x under contention), and mean regret is a tail statistic —
    the top **10 of 590** chunks carry **52.6 %** of it, so medians are quoted beside means.


---

## Agreed ranked suggestions

Two ladders, because they improve different things and are not interchangeable: **(a)** makes
the weights every fresh process starts from better, and helps every chunk immediately;
**(b)** makes the in-process loop better, and has to converge in tens of chunks.
Two conditioning facts govern both: **which head is worth fixing depends on storage bandwidth**
(at ≤500 MB/s the ct fixes become harmful and the ratio fixes become the valuable ones), and
`--best` is ratio-only **and** disables both SGD phases, so nothing in (b) applies there.

### (a) Better starting weights / inference-time inputs

| # | change | measured gain | cost (bytes / top-1) | faithfulness to upstream |
|---|---|---|---|---|
| **a1** | **Retrain `model.nnwt` on this deployment's labels**: codec-kernel compress time instead of upstream's whole-API wall-clock bracket; chunk sizes 0.25-64 MiB; amplitudes past mad = 0.5 | the whole headroom: regret **0.1367 → 0.0019** and bytes **0.678 → 0.650** (the "true ct + ratio" and oracle rows) | unknown until retrained; no code change, file swap only | **none** for the label and coverage changes — it is upstream's own training recipe; range-normalising mad/d2 diverges from upstream's feature definition |
| **a2** | **Raise the ratio cap, scoped to the ratio-only objective** | eb=1e-3 ratio-only regret **35.99 → 0.435 (83x)**, bytes **0.4314 → 0.3824 (−11.4 %)**; lossless 0.5545 → 0.2331 | **zero** effect on the balanced objective; no effect at all on a corpus whose predictions collapse low (my held-out set); fixes ~60 % of the tie problem | `CLIO_NEUROPRESS_RATIO_CAP` exists; the standing rule keeps the default at 100, so scope it to `--best` |
| **a3** | **Clamp the four continuous inputs to the `.nnwt` `x_mins`/`x_maxs`** before standardization (5 sites) | ratio-only bytes **0.6710 → 0.6033**, eb=1e-3 **−34.7 %**, held-out corpus **−11.5 %**, all deterministic; balanced regret 0.1572 → 0.0395 | balanced **bytes get worse**, and by a corpus-dependent amount: **+1.9 %** (f0a), **+1.5 %** (28-regime), **−1.7 %** (3-block); **no in-distribution effect at all** (predictions bit-identical on 5696/5696 in-bounds rows) | near zero — the bounds ship inside `model.nnwt` and upstream computes them (`neural_net/core/data.py:163-164`) but never consults them at inference |
| **a4** | **Per-*algorithm* ct log-bias table** (static, shipped beside the weights) | out-of-sample balanced regret **0.1629 → 0.0330 (−80 %)** | bytes **+4.6 %**, top-1 0.375 → 0.317, **median chunk worse** (0.0031 → 0.0073) — it is a tail fix; **harmful below ~500 MB/s** | new mechanism upstream does not have; env-gate it **and** bandwidth-gate it. Do **not** extend it to the ratio head (that error is per-chunk, −79 %, not per-algorithm, +3 %) |
| **a5** | **Guard the feature path**: reject/warn on non-float payloads and on non-finite or wildly out-of-range statistics, on **both** the device and host paths | converts a silent "always lz4" into the legacy-heuristic fallback that already exists at `neuropress_selection.cc:177-182` | none | none — upstream never sees non-float data, and has no `data_type_` field to contradict |
| **a6** | **Keep the 1 ms cost floor** | lowering it to 0.1 ms costs **0.061 top-1** and **+22 %** mean regret, and the direction survives a paired timing replicate | — | faithful (`fmaxf(1.0f, ·)`, `nn_gpu.cu:229-231`). Scope: an idle-device result — the floor truncates 29 % of measurements idle, ~7 % loaded |

### (b) Better in-run adaptation

Ranked by measured value. The headline finding of round 2 is that **the harm from trunk SGD
scales with how often it fires, whatever the trigger**: across seven idle arms on one schedule,
0 calls → regret 0.1367, 1 → 0.1442, 41 → 0.1373, 50 (with exploration) → 0.1277, 52 → 0.1741,
177 → 0.3043, 205 → 0.4692.

| # | change | measured gain | cost (bytes / top-1) | faithfulness to upstream |
|---|---|---|---|---|
| **b1** | **Clamp the inputs (= a3). It makes the loop quiescent rather than harmful.** The gate fires because the model is extrapolating, so removing the extrapolation removes the trigger | phase-1 firings **51 → 3** (my 560-chunk schedule) and **74 → 0** (their 450-chunk stream, where `learn+clamp` is then bit-identical to `infer+clamp`); regret on their stream **1.0918 → 0.8088**, median **0.8070 → 0.3822** | the a3 byte trade, corpus-dependent (+1.9 % / +1.5 % / −1.7 %) | as a3 |
| **b2** | **Learn the magnitude outside the network: an online per-algorithm log-residual table on the *time* heads, trunk untouched** | with **zero** SGD calls: regret **0.1380 → 0.0908**, median **0.0138 → 0.0082** (28-regime); **1.0918 → 1.0146** per-algo and **→ 0.9570** per-action-ct (3-block). Only mechanism measured that beats not-learning on both | bytes **+0.7 % / +1.7 % / +4.0 %**, top-1 down; **tail-carried** — it changes 226 picks and is worse on 130 of them, all the net gain in the last quarter | new mechanism; `np_eval` implements it (`NPE_BIAS`), the runtime does not. Do **not** add the ratio head (worse than not adapting: 1.2348 vs 1.0918, three independent reproductions) |
| **b3** | **If the trunk learner stays on, feed it phase-2 exploration samples** (`NPE_EXPLORE_K=7`) | the only trunk-SGD arm beating no-learning on mean regret: **0.1277 vs 0.1367**, best first-half regret of any arm (0.1250 vs 0.1620) | **worst** top-1 (0.168 vs 0.361) and **worst** bytes (+3.2 %) of any arm; median chunk worse; K extra codec runs per gated chunk; crossover is schedule-dependent (chunk 5-17 vs chunk 560) | configuration change only — exploration is default-**off** (`compressor_tasks.h:103`) |
| **b4** | **Trigger hygiene, if you must trigger at all**: prefer the ratio MAPE (bit-deterministic, gate rate **0.475 at every device load** vs the cost gate's 0.239 → 0.997) and add an `n_x_oob == 0` precondition (gate rate 0.163 at every load, 0 % OOD by construction) | precondition repairs the replaced-ratio gate **0.4692 → 0.1741**; applied to the cost gate it cuts firings **41 → 1** | none of these beats not learning: OR knob **0.3043**, replaced+precondition **0.1741**, against **0.1367**. Best median regret of any arm is the OR knob's 0.0048 | `CLIO_NEUROPRESS_SGD_ON_RATIO` ships the OR form; replacing the gate and exposing an OOB count are new |
| **b5** | **Do not simply widen the cost gate** | training every chunk is the worst configuration measured: regret 0.1572 → 0.2624, top-1 0.407 → 0.207, bytes +3.7 % | — | — |
| **b6** | **Honest fallback: a static codec** | ratio-only bytes **0.5883** (`zstd` + 4-byte shuffle) vs the model's 0.6721 — equal to the per-chunk oracle to 4 dp | decisive for the ratio objective (which is degenerate on both corpora); much weaker for the balanced one **in distribution**, where the gap is 1.23x and the model wins on bytes by 8.8 % | `neuropress_static_lib_` + `neuropress_static_shuffle_ = 4` already exist (`compressor_runtime.cc:434-448`) |

**Joint bottom line.** For the starting point, a1 is the only change that removes the causes
rather than the symptoms; a3 is the best deterministic inference-time win and is a bytes-for-time
trade under the balanced objective. For the in-run loop, nothing that updates the **trunk** is a
net win on an idle GPU with a common ground truth; what works is to stop the trigger firing on
extrapolations (b1) and to learn the magnitude in a table outside the network (b2).


