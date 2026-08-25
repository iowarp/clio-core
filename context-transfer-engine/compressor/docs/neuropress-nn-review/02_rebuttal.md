# Adversarial rebuttal — NeuroPress NN predictor in Clio

Branch `neuropress-693-continued` @ 81ac5a43, A100-PCIE-40GB, CUDA 12.6. GPU was idle
(`nvidia-smi`, 0 % util, no processes) before every run below.

**Ground rule I held myself to:** every verdict below is backed by a command I ran, not by
reading the investigator's numbers. I re-implemented the cost/regret/oracle machinery from
scratch (`adversary/scripts/adv.py`) rather than calling `an*.py`/`oracle*.py`, and I generated
a **corpus the investigator never used** (`np_eval`'s four `opt_in` regimes — `nyx_density`,
`warpx_field`, `vortex2d`, `cell_counts`, seed 777, which `NPE_REGIMES=all` deliberately
excludes) plus a learning battery with different schedules, run lengths and replicates.

**Deployment constraint folded in (from the user, mid-task):** the shipped NeuroPress mode is
**in-memory adaptation from fresh weights**. Every process loads `model.nnwt`, adapts its live
weights by phase-1/phase-2 SGD for the process lifetime, and never persists
(`compressor_runtime.cc:1468` comment; `Save()` is defined at
`neuropress_nn_predictor.cc:183` and called from nowhere in the runtime — I grepped). So
"turn the learner off" is not an available answer; F7/F8/F9 and C5 are judged on whether a
*fresh process* ends up better or worse, and on run length / order sensitivity.

Two facts about the shipped defaults that change how several claims should be read, both of
which I verified in `compressor_tasks.h:103-105` and `compressor_runtime.cc:473-481, 1577,
1653-1655, 2290`:
- `neuropress_exploration_enabled_ = false` **by default**. K=3 is the default *K*, not the
  default behaviour. The default adaptation path is **phase 1 only**.
- `--best` (ratio-only) **disables both SGD phases** (`!config_.neuropress_best_mode_` guards
  at `:1577` and `:2290`; upstream does the same at
  `gpucompress_compress.cpp:705`). Any "learning under the ratio objective" measurement is of
  an unreachable configuration.

---

## Verdicts

### F1 — absolute-magnitude features put data far out of distribution → collapse. **CONFIRMED, but the −75 % headline is corpus-composition-dependent; the mechanism statement is slightly over-stated and the balanced-objective *bytes* cost of the fix is not reported.**

**What I ran.**
1. Independent re-derivation over the investigator's CSVs with my own scorer:
   `adv.py f0a_infer_eb0.csv balanced 1.0` → predictions regret **0.1572**, bytes 0.67575;
   `adv.py g1_clamp_eb0.csv balanced 1.0` → **0.0395**, bytes **0.68848**.
   `... ratio 1.0` → bytes **0.67095 → 0.60328**. Their numbers replicate exactly.
2. Held-out corpus never used by the investigator (`runsA.sh`, 100 chunks, 4 opt-in regimes):
   balanced regret **2.9334 → 0.0264**, top-1 **0.000 → 0.240**; ratio-only bytes
   **0.73360 → 0.64964**, top-1 0.250 → 0.740. At eb=1e-3 (h3/h4): balanced regret
   **4.0268 → 0.0260**, ratio bytes **0.62661 → 0.48384**.
3. Per-regime decomposition of the gain (my script, over f0a vs g1).

**Attack 1 — is the −75 % an artefact of corpus composition?** Partly yes.
Four regimes (`sizes`, `smooth_a1000`, `turb_x50`, `gs_x1e4` = 90/590 chunks) carry
**86 %** of the whole-corpus balanced regret. Excluding them:

| subset | base regret | clamp regret | base bytes | clamp bytes |
|---|---|---|---|---|
| all 590 | 0.1572 | 0.0395 | 0.67575 | 0.68848 |
| minus those 4 regimes (n=500) | 0.0267 | **0.0339 (worse)** | 0.66005 | 0.65624 |
| strictly in-`.nnwt`-bounds (n=356) | 0.0312 | **0.0427 (worse)** | 0.59501 | 0.59501 |

**Did that attack survive? No.** The in-distribution predictions are provably identical
(I diffed `pred_ct`/`pred_ratio` row by row: **5696/5696 identical** between f0a and g1), so
the only thing that can move balanced regret there is codec-timing noise. The same-seed
replicate `f0c` gives in-dist regret 0.0353 against f0a's 0.0312 with *bit-identical*
predictions — i.e. a ±0.006 noise band on n=356, which covers g1's 0.0427. **I concede the
"clamp hurts in-distribution" reading: it is noise.** What survives is the weaker,
corpus-dependent statement: *outside the four extreme regimes the clamp's balanced-regret
benefit is indistinguishable from zero*, and the whole-corpus −75 % is a linear function of
how much of the corpus you choose to put out of distribution (here 46 % of rows).

**Attack 2 — is C1's gain free?** No, and this is not in F1 or C1.
Balanced-mode stored bytes are **deterministic** (f0a and f0c are byte-identical on
`act_comp_bytes`, 9440/9440), and the clamp makes them **worse**: 0.67575 → **0.68848
(+1.9 %)**. Per regime the regression is concentrated where the regret gain is:
`sizes` 0.6379 → 0.8350, `gs_x1e4` 0.4944 → 0.6032, `stepped` 0.0078 → 0.0593 (**7.6x more
bytes**), `lammps_vel_t1` 0.8821 → 1.0017. That is the balanced objective working as
specified (it buys time with bytes), but C1 is sold as an unambiguous win and it is not one
under the balanced cost.

**Attack 3 — is the mechanism "collapses to one prediction for all 32 actions"?** Over-stated.
Per-regime distinct-prediction counts (f0a, 16 lossless actions per chunk):

| regime | mad | distinct ct | distinct ratio |
|---|---|---|---|
| smooth_a1000 | 314 | 1 | 1 |
| stepped | 64 | 1 | 1 |
| turb_x50 | 40.0 | 1 | 1 |
| gs_x1e4 | 57.4 | **11** | **14** (28 % of ct at the 1e6 ceiling) |
| lammps_pos_sorted | 19.8 | **16** | **1** |
| lammps_pos_shuffled | 20.0 | **16** | **1** |
| lammps_force | 7.97 | **15** | **1** |

The two heads saturate *independently*: at mad≈20 the ratio head is a constant while the ct
head still has 16 distinct values (with median ct APE 750-880). "Collapses to one prediction
for all 32 actions" is true only above mad≈40. The honest statement is "one or both heads
saturate, monotonically in amplitude", which the E4 sweep already shows.

**Class.** Upstream design limitation, faithfully ported — I re-verified the two upstream
constants myself (`/home/cc/NeuroPress/src/nn/nn_gpu.cu:229-231` policy clamps,
`src/api/diagnostics_store.hpp:319` `RATIO_CAP = 100.0f`).

**Net: CONFIRMED.** The *ratio-only* result is the strong half and it is deterministic and
survives every subset I tried (−6.7 % bytes even excluding the four extreme regimes, −11.5 %
on my held-out corpus). The balanced-regret headline is real but corpus-weighted, and it
costs 1.9 % bytes.

---

### F2 — ct head owns balanced regret, ratio head owns ratio-only regret, dt head owns nothing. **CONFIRMED as stated, but the conclusion inverts under the one assumption the claim is conditioned on, and Section C inherits the conditioning without saying so.**

**Replication (my scorer, not `an2.py`):** f0a, ground truth floored at 1 ms, uncapped:

| variant | balanced top-1 | balanced regret | ratio-only regret |
|---|---|---|---|
| predictions | 0.378 | 0.1572 | 0.5545 |
| true ratio | 0.327 | **0.6340** | 0.4169 |
| true ct | 0.324 | **0.0169** | 0.5545 |
| true dt | 0.383 | 0.1215 | 0.5545 |
| true ct+ratio | 0.776 | 0.0019 | 0.4169 |

Exact match to F2's table. Held-out corpus (h1): predictions 2.9334, true ct **0.1750**,
true ratio **5.8440 (worse)**, true ct+ratio 0.0000. Same ordering, larger effect.

**Attack 1 — is "true ratio makes it worse" an artefact of the 100x cap?** No. Re-running with
the predictor-side cap raised to 1e9 gives true-ratio regret **0.6668** (vs 0.6340 capped) —
slightly *worse*, not better. The confound is dead.

**Attack 2 — is the whole ordering a knife-edge of `bw = 5e6`?** **Yes, and this is the real
hit.** `g_measured_bw_bytes_per_ms = 5e6f` is upstream's hard-coded "representative HPC
storage" (verified at `gpucompress_api.cpp:71`). Recomputing my table at other bandwidths:

| bandwidth | predictions | true ratio | true ct |
|---|---|---|---|
| 5 GB/s (shipped) | 0.1572 | 0.6340 | **0.0169** |
| 500 MB/s | 0.3376 | **0.2314** | 0.1635 |
| 50 MB/s | 0.2297 | **0.1164** | 0.2633 (**worse than doing nothing**) |
| 5 MB/s | 0.2151 | **0.1181** | 0.4279 (**much worse**) |

At and below ~500 MB/s the claim reverses completely: perfect ratio knowledge helps and
perfect ct knowledge *hurts*. The investigator does state the conditioning in the falsification
line, so F2 is not wrong — but the headline and C3 are written as unconditional, and I ran the
falsification they named and it fires at any plausible network/PFS/HDD bandwidth. F2's title
should read "at 4 MiB and 5 GB/s".

Minor: I measure the io share of the measured balanced cost as **mean 12.4 %, median 6.5 %,
p90 27.7 %**; F2 says "median 3.3 %". Mean matches, median does not.

---

### F3 — ct head's dynamic range ~21x too narrow, per-algorithm error is a constant. **CONFIRMED, and UNDER-stated: the fix it implies is worth more than the finding says.**

**Replication.** My own log-error script: median |log(pred/act)| for ct **1.210**, minus a
per-algorithm median constant **0.499 (−59 %)**; dt 1.054 → 0.431 (−59 %); ratio 1.060 → 1.094
(**+3 %**, i.e. it hurts). All three match F3 to 3 decimals.

**Attack 1 — is −59 % just degrees of freedom (any 8-way grouping would do it)?** No.
Null control, a *random* 8-way grouping of the same 9440 rows (seed 7): ct 1.210 → **1.242
(+3 %)**. A random 32-way grouping gives −21 %, so F3's "per-action (32 keys) gets ct to 0.441"
is partly DOF inflation (and at eb=0 there are only **16** actions, not 32 — label error).
The per-algorithm result is not.

**Attack 2 — F3 never shows the correction helps *selection*.** I tested it, out of sample
(fit the per-algorithm log bias on f0a, apply to f0b):

| correction | f0b regret | f0b top-1 | f0b bytes |
|---|---|---|---|
| none | 0.1629 | 0.375 | 0.67487 |
| ct only, per **algo** | **0.0330 (−80 %)** | 0.317 | 0.70576 |
| ct+dt, per algo | 0.0291 | 0.286 | 0.70851 |
| ct+dt+ratio, per algo | 0.0380 | 0.286 | 0.69528 |
| ct only, per **action** | 0.0530 | 0.325 | 0.69375 |

So C3 is worth **−80 % balanced regret out of sample** — the investigator only cited the
error reduction and left the decision impact to be inferred. Two corrections to C3, though:
(a) per-**action** generalises *worse* than per-algo (0.053 vs 0.033), the opposite of what
F3/C3's "per-action gets ct to 0.441" implies; (b) it costs **+4.6 % stored bytes** and
−0.058 top-1, the same bytes-for-time trade as C1.

New finding of my own: the ratio head's error is **not** per-algorithm but almost entirely
**per-chunk** — removing one constant per chunk cuts ratio |log err| 1.060 → **0.241 (−77 %)**
(per-regime: −74 %). A per-chunk log offset cancels inside a chunk's ranking but not against
the ct/dt terms, which is exactly the mechanism F2 needs for "true ratio makes balanced worse".
The investigator has the effect but not this diagnosis.

---

### F4 — the 100x cap decides 43 % of lossy chunks by tie-break. **CONFIRMED and UNDER-stated.**

Recomputed from `pred_ratio_raw` through my own clamp chain:

| corpus / mode | cap 100 | cap 1e5 |
|---|---|---|
| f0a ratio-only: top-1 / regret / bytes / tie-chunks / picks-a0 | 0.298 / 0.5545 / 0.67095 / 0.431 / 0.376 | 0.395 / **0.2331** / 0.65386 / 0.254 / 0.254 |
| f0a balanced | 0.378 / 0.1572 / 0.67575 | 0.378 / 0.1572 / 0.67575 (identical) |
| **f1 (eb=1e-3) ratio-only** | 0.109 / **35.99** / 0.43135 / 0.482 | 0.114 / **0.4350** / 0.38236 / 0.257 |

F4's lossless numbers replicate exactly. The eb=1e-3 ratio-only regret — **35.99 → 0.435**,
an 83x reduction, and bytes −11.4 % — is not in F4 at all; that is the strongest single number
in the whole finding set and it is missing.

**Attack — is it the cap, or just tie-breaking?** Both, separably: removing the cap drops the
tie rate from 0.431 to 0.254, so **~25 % of chunks are decided by the action-index tie-break
regardless of the cap** (those are the F1 collapse chunks). F4 attributes all of it to the cap;
about 60 % of it is.

---

### F5 — a single fixed action beats the model out of sample. **CONFIRMED on ratio; WEAKENED to near-parity on balanced once "in-distribution" is defined correctly.**

**Replication (my scorer, fit on f0a, scored on f0b):** balanced/floor 1 ms model 0.1629 vs
const a7 0.0285; balanced/floor 0 model 0.7694 vs a23 0.1018; ratio-only model 0.5559 (bytes
0.67211) vs a20 **0.0000** (bytes 0.58829, oracle 0.58825). Every number matches F5.

**Attack 1 — held-out corpus.** Survives, hard. On my four opt-in regimes: balanced model
2.9334 vs best const a21 **0.0060**; ratio-only model bytes 0.73360 vs a20 **0.63262**, which
*equals the per-chunk oracle to 5 decimals*. Even the **clamped** model (h2) loses: 0.0264 vs
0.0064 balanced, 0.64964 vs 0.63262 bytes.

**Attack 2 — the "in-distribution" definition is not the one F5's own Section A uses.**
`oracle2.py`'s `INDIST` list is regime names selected on `mad <= 0.5` **only**, and it includes
`sizes` (chunks up to 16 MiB) and `drift`, so it still contains chunks that violate the
`.nnwt` size bound F1/F10 rely on. Filtering on **all four** `.nnwt` bounds
(mad ≤ 0.49998, d2 ≤ 1.00573, entropy ≤ 7.52983, 64 KiB ≤ size ≤ 4 MiB), fit on f0a, scored
on f0b:

| set | n | model regret | const regret | model bytes | const bytes | oracle bytes |
|---|---|---|---|---|---|---|
| all | 590 | 0.1629 | 0.0285 (a7) | 0.67487 | 0.71952 | 0.64982 |
| investigator's INDIST | 350 | 0.1505 | 0.0348 (a7) | 0.60568 | 0.67396 | 0.59947 |
| **strict .nnwt bounds** | 356 | **0.0297** | **0.0242** (a7) | 0.59355 | 0.65076 | 0.58523 |

In-distribution the balanced gap collapses from **4.3x to 1.23x**, and the model wins on bytes
by 8.8 %. F5's "in-dist too" row is carried by mis-labelled out-of-range chunks. The ratio-only
half survives the correction (model 0.5665 vs const 0.0001; bytes 0.53471 vs 0.51763), but the
byte gap shrinks from 12.5 % to **3.3 %**.

**Alternative explanation the investigator raises and I confirm is decisive for ratio:** the
ratio objective on this data is *degenerate*. Per-chunk oracle bytes 0.58825 vs best-constant
0.58829 — a 0.007 % difference, i.e. **no selector of any quality could beat a constant here**.
Same on my held-out corpus (a20 picked by the oracle on 100/100 chunks). So "the model has
negative value under ratio-only" is a statement about the corpus as much as the model. Under
the *balanced* objective the oracle does beat the constant (0.0000 vs 0.0285 regret, 0.64982 vs
0.71952 bytes) — real headroom exists there and the model captures little of it. F5 should say
both halves.

---

### F6 — exploration + adoption, not the network, is what makes the path competitive. **CONFIRMED as an offline simulation; the framing "at the default K=3" is wrong — exploration is default-OFF.**

**Replication with my own simulator** (not `explore_sim.py`; primary = predicted argmin, window
= primary + top-K predicted alternatives, adopt = measured argmin in the window):

| K | balanced regret | balanced bytes | ratio regret | ratio bytes | adopt rate | best-in-window |
|---|---|---|---|---|---|---|
| 0 | 0.1572 | 0.67575 | 0.5545 | 0.67095 | 0 | 0.378 / 0.298 |
| 3 | 0.0708 | 0.68014 | 0.1708 | 0.61818 | 0.47 / 0.63 | **0.541** / 0.603 |
| 7 | 0.0104 | 0.67348 | 0.0431 | 0.61198 | 0.53 / 0.70 | 0.754 / 0.710 |
| 31 | 0.0000 | 0.64975 | 0.0000 | 0.58700 | 0.62 / 0.70 | 1.000 |

Matches F6 to 3 decimals including the 54.1 % best-in-window figure.

**Attack 1 — adoption is fitted to timing noise.** Real: I re-scored the *same picks* against
the independent same-seed replicate `f0c`'s measured table. Balanced regret at K=31 goes
0.0000 → **0.0092**; K=7 0.0104 → 0.0292; K=3 0.0708 → 0.0805. So F6's K=31 row (0.0005) is
optimistic by an order of magnitude but the conclusion is unchanged. Ratio-only is unaffected
(ratios are deterministic).

**Attack 2 — "the default K=3".** `neuropress_exploration_enabled_ = false`
(`compressor_tasks.h:103`). The shipped default is the K=0 row. F6's own numbers therefore say
the *default* configuration is the worst row in its own table, which is a stronger and more
useful statement than the one made. **Under-stated.**

---

### F7 — one SGD step is invariant to the size of the error; only the sign matters. **CONFIRMED for the single-sample case that phase 1 uses, REFUTED for the batched phase-2 case, and the mechanism is not what makes the ±0.5 clamp a no-op.**

**Source check.** Verified line by line in `neuropress_nn_gpu_kernels.cu`: `:1591-1592` noise
gate on `|d5[0]| < 0.10`; `:1596-1598` ±0.5 clamp; `:1640-1660` `dz4_all[o]` unit-normalised so
only `sign(d5[o])` survives; `:1674-1675` `dz4 = my_dz4 * err_mag`; `:1679-1681` W5 row
`error_signal * h4`; `:1774` `clip_scale = 0.1/‖g‖`. Upstream's
`GRAD_CLIP_THRESHOLD = 0.1f` is at `/home/cc/NeuroPress/src/nn/nn_gpu.cu:927` as claimed.
The whole `out_grad` for one `target_out` pass is linear in `|d5[target_out]|`, so a scalar
divides out. The mechanism is right.

**Attack — the invariance is a single-sample property, and phase 2 trains on batches of up to
7.** `out_grad` accumulates **across samples** inside the `target_out` pass before the clip is
applied, so the *relative weight of samples* is set by their `|d5|` and does not cancel. I
wrote my own probe (`parity/np_adv.cu`, A1 vs A1b) to check this rather than argue it.

**Run.** `np_adv.cu` A1 (7-sample batch, phase-2 shape): varying only sample 0's ct label over 2.70/2.40/2.00/1.60/0.80 gives `cos(dW0,dWi)` = 0.999482 / 0.995219 / 0.980546 / **0.909978**. A1b (single sample, identical labels) gives **1.000000000** within a sign class and 0.331276027 vs 0.331275860 for the two opposite-sign cases. Full tables in the `np_adv.cu` section below.

**Also worth noting.** F7's "uncertainty weighting and the ±0.5 clamp are therefore no-ops"
is right about `uw`, but the ±0.5 clamp is *not* a pure no-op even for one sample: it is
applied per output, so when two outputs are wrong by different amounts and only one exceeds
0.5, the clamp changes their **relative** magnitudes — which changes nothing for the trunk
(each output is separately renormalised) but does change `w5`/`b5`, which are written from
`error_signal` and clipped by the *same* per-output scale. Net effect on one output's own W5
row: still zero. So the claim holds, but for a narrower reason than "everything multiplying d5
cancels".

**Class:** upstream design limitation, faithfully ported. **Confirmed, correctly scoped in the
mechanism paragraph, over-generalised in the impact paragraph.**

---

### F8 — online SGD makes selection worse; mechanism is cross-action interference. **See the learning battery section below.** The `f3` row must be struck: it measures a configuration that cannot run.

`--best` (ratio-only cost) **disables both SGD phases**: `compressor_runtime.cc:1577`
(`np_will_train = (gate) && !config_.neuropress_best_mode_`) and `:2290`
(`if (!explore_features.empty() && !config_.neuropress_best_mode_)`), and the startup warning
at `:475-481` says so in words ("Selection is ratio-only and both SGD phases are off").
Upstream matches (`gpucompress_compress.cpp:705`, `&& !g_best_mode.load()`). F8's sharpest
evidence — "`f3` ratio-cost gate 0.30 … learning costs 8.2 % more stored bytes" — is therefore
about a configuration the product cannot enter. **That row is invalid.**

---

### F9 — the SGD gate never opens on half the chunks whose ratio prediction is wrong. **CONFIRMED numerically; the finding misses that Clio already ships the fix as an env knob.**

Replicated exactly on the 590 primary rows of f0a: gate open **23.9 %**, ratio MAPE > 0.30
**47.5 %**, both 23.6 %, ratio-bad-with-gate-shut **23.9 %**, cost-open-ratio-fine 0.3 %.
**141/280 = 50.4 %** of ratio-bad chunks never train.

**Missed by the investigator:** `compressor_runtime.cc:1536-1548` already implements the exact
remedy — `CLIO_NEUROPRESS_SGD_ON_RATIO=1` ORs a ratio-MAPE gate onto the cost gate, with
`CLIO_NEUROPRESS_RATIO_MAPE_THRESH` for its threshold — and `np_eval` exposes it as
`NPE_GATE=ratio_or_cost`. F9 describes the limitation and C5 says "do not widen the gate",
without measuring the knob that exists for exactly this. I measured it (see the battery).
The code comment there also claims the io term is "~0.4 % of the cost" at 4 MiB; my measurement
says mean 12.4 %, median 6.5 % — the comment is off by more than an order of magnitude.

---

### F10 — chunks > 4 MiB are out of distribution; size is the least-used input in range. **CONFIRMED on the out-of-range half; the "least-used in range" half rests on a marginal sweep that does not support it.**

Real-data replication (f0a `sizes` regime, fixed content, my own APE over all 16 actions):

| chunk | n | median ratio APE |
|---|---|---|
| 0.25 MiB | 96 | **4.355** |
| 1 MiB | 96 | 1.630 |
| 4 MiB | 96 | 4.528 |
| 8 MiB | 96 | **46.2** |
| 16 MiB | 96 | 14.3 |

Same shape as F10 (their 1.56 / 3.87 / 40.6 / 13.4). **New:** the 0.25 MiB point, which F10
omits, is *worse* than 1 MiB and is squarely inside the training range — so "in-range size is
harmless" is not quite true either.

**Attack on "the input the model uses least."** E7 sweeps one feature at a time with the other
three pinned at the **training mean**. That is a single-point marginal sensitivity, and the
four inputs are not independent in real data (entropy, mad and d2 co-vary; size does not), so
comparing marginal ranges does not establish "uses least". The out-of-range failure
(ratio → 100.0 at 8/16/64 MiB, PSNR 34.5 dB predicted for a lossless action at 64 MiB) is
solid; the in-range ranking is a methodological artefact. **Confirmed on the part that matters,
over-claimed on the framing.**

---

### F11 — the PSNR head cannot rank anything; `target_psnr` discards a third of the action space. **CONFIRMED, but one of its numbers is not reproducible and the metric is ill-posed.**

Replicated exactly from f1 (7040 quantized rows / 440 chunks): distinct **actual** PSNR per
chunk = **1 on 440/440**; distinct predicted = 16 on 312/440 (F11 says 310). Masking:
60 dB → **35.2 %** wrongly masked / 9.7 % wrongly admitted; 80 dB → 39.0 % / 8.1 %;
100 dB → 29.5 % / 6.6 %. Identical to F11.

**Attack 1 — circularity.** The harness's `act_psnr` **is** `AnalyticalPsnr(range, eb)`
(`np_eval.cu`, `m.psnr = quant ? AnalyticalPsnr(...) : 120.0`), the same formula as the label,
so "the actual PSNR is a per-chunk constant" is true *by construction of the harness*, not
measured. **It survives anyway**: quantization is applied before a *lossless* codec, and
`q_eff_eb` depends only on (data range, eb), so the reconstruction — and hence the real PSNR —
genuinely is identical across all 16 quantized actions. Circular evidence, correct conclusion.

**Attack 2 — "PSNR median APE 22.2 %, p90 100 %, mean 347 %" does not reproduce.**
`act_psnr` here **goes negative** (range −283.6 … 120.0 dB; 34/440 chunks have a negative
analytic PSNR, in `const`, `gs_*`). A percentage error with a negative denominator is
meaningless. With the signed denominator I get median 0.140 / p90 1.000 / mean **−1.544**;
with `|denominator|`, median 0.254 / p90 1.179 / mean 7.951. Neither is 22.2/1.00/3.47.
That number should be withdrawn or its definition stated.

**Missed by the investigator:** the psnr output is clamped to **[0, 120]**
(`neuropress_nn_gpu_kernels.cu:535`) while the label it is trained against reaches −283 dB.
On those chunks the head *cannot* represent its own target, and every SGD step there carries a
permanent, unclosable error — a larger version of the F16/E11 effect.

---

### F12 — the model is close to blind to the shuffle half of the action space. **CONFIRMED.**

Independent recount over 4720 (chunk, algo) pairs: shuffle measurably improves the ratio on
**89.3 %** of pairs (F12 says 86.3 %); the model predicts an improvement on **43.4 %**
(43.3 %); precision **0.943** (0.927), recall **0.458** (0.466); predicts *exactly* no
difference on **32.9 %** (32.9 %), of which **27.7 %** of all pairs is both sides at the cap
(27.7 %). Per-algorithm predicted median `r_shuf/r_plain` is 1.000 for lz4/snappy/deflate/
gdeflate/zstd/bitcomp, **1.545** for `nvcomp-ans` and **1.983** for `nvcomp-cascaded`, against
measured 1.11-1.20 for all seven and 1.000 for bitcomp — exactly F12's table. No attack landed;
these are deterministic quantities and they reproduce.

---

### F13 — non-float32 data is read as float32; one non-finite feature degrades selection to "always lz4". **CONFIRMED on mechanism; the class is right and the port comment already concedes it.**

Source verified: `neuropress_selection.cc:47-49` pins `ctp::DataType::FLOAT32` unconditionally
(the comment there says in as many words "Reading as uint8 puts MAD hundreds of sigma outside
the training range"), `:66-72` converts `data_type_ == 2` (float64) rather than reinterpreting,
and every other `data_type_` falls through to the float32 read. `fmaxf(0, NaN) == 0` in CUDA,
so one NaN input zeroes hidden layer 1 and the network becomes its bias vector — one value for
all 32 candidates, which the `(score, action, slot)` tie-break resolves to action 0.
The sanity clamp comment at `:526-530` explicitly reasons about NaN in the *output*, not the
input, so the input path is genuinely unguarded.

**Attack — is the host path affected too?** No: the non-device path calls
`ctp::ComputeNeuroPressFeatures(chunk, chunk_size, context.data_type_, ...)`, which honours the
type. The defect is device-path-only, which narrows F13 to CUDA-IPC/device-resident chunks.
F13 does not say this.

**E9/E10 re-run on my side; every number reproduces.**

| features | distinct ct over 8 algos | distinct ratio | cost argmin |
|---|---|---|---|
| baseline | 8/8 | 8/8 | algo 5 = `nvcomp-ans` |
| mad = NaN | **1/8** | **1/8** | **algo 0 = lz4** |
| d2 = NaN | 1/8 | 1/8 | lz4 |
| mad = +Inf | 1/8 | 1/8 | lz4 |
| all NaN | 1/8 | 1/8 | lz4 |
| mad = 1e12 | 1/8 (ct = 1.0) | 1/8 (ratio = 0.1) | lz4 |

E10: a 1 MiB uint8 count field read as float32 gives `entropy = 7.6511, mad = 3.01844e+36, d2 = 6.18358e+36`; a 1 MiB int32 index buffer gives `mad = 3.32186e-41` (denormals) against a true integer MAD of **23705.6**. Identical to F13.

---

### F14 — the 1 ms floor is hiding prediction error; lowering it makes selection worse. **CONFIRMED, and better supported than the investigator's own hedge suggests.**

My replication on f0a (predictor-side policy varied, ground truth = measured, no floor, no cap):
cap100/floor 1 ms → top-1 0.266, regret 0.7241, med 0.0798, p90 1.8098; floor 0.1 ms →
0.205 / 0.8856 / 0.1447; no floor → identical to 0.1 ms; cap ∞/floor 1 ms → identical to
cap 100. Every cell matches F14.

**Attack — the unfloored ground truth is noise.** I measured it: the ground-truth argmin is
reproducible between two same-seed runs on **96.8 %** of chunks at floor 1 ms but only
**77.8 %** at floor 0.1 ms or 0 — exactly the investigator's own figure. But the *paired*
comparison is stable: repeating the whole table on `f0c` gives 0.246/0.7781 → 0.175/0.9578,
i.e. the same direction and a similar magnitude, with a between-run spread (0.054) about a
third of the effect (0.16). **F14's "medium-high" is if anything too modest.** C7 ("do not
lower the floor") is the best-supported negative recommendation in the set.

---

### F15 — the deferred decompression head trains toward a constant; it does not matter. **CONFIRMED (dt substitution independently re-derived).**

From my head-substitution table: replacing dt by its measured value changes balanced regret
0.1572 → 0.1215 and top-1 0.378 → 0.383 — i.e. the *entire* value of a perfect dt head is
−0.036 regret, against −0.140 for a perfect ct head. Pinning dt ≡ 1.0 is within that band.
59.7 % of measured dt values are below the 1 ms floor on this hardware (I recount 59.7 %).
Nothing to attack; the finding is correctly labelled "listed to close it out".

---

### F16 — the 1e-7 lossless sentinel is inert; the psnr label bias on lossless primaries is second-order. **CONFIRMED.**

My E6 re-run: `eb = 0 / 1e-7 / 1e-5` gives ct 2.933 / 2.933 / 2.933 and ratio 18.629 / 18.629 / 18.630 — the sentinel moves the ratio by 0.005 % and ct by 0.00 %, exactly as F16 says. My E11 re-run: 20 steps with the hard-coded `psnr = 120.0` label leaves ct[0] = 4.0892, ratio[0] = 1.2110; replacing it with the model's own 114.68 gives ct[0] = **3.7895** (−7.3 %) and ratio[0] = 1.2139 (+0.2 %). Real, small — and note it also moves an *untrained* action (ct[4] 4.5671 → 4.3670), i.e. F8's interference showing up inside F16's own experiment.

The `[0,120]` psnr clamp against a label range reaching −283.6 dB (F11 above) is the same effect about two orders of magnitude larger, and both F11 and F16 miss it.

---

### F17 — port faithfulness. **CONFIRMED.**

I re-verified five upstream constants by hand rather than trusting the parity suite:
`RATIO_CAP = 100.0f` (`src/api/diagnostics_store.hpp:319`), `g_rank_w0/w1/w2 = 1.0f` and
`g_measured_bw_bytes_per_ms = 5e6f` (`src/api/gpucompress_api.cpp:67-71`),
`GRAD_CLIP_THRESHOLD = 0.1f` (`src/nn/nn_gpu.cu:927`), the 1 ms/100x policy clamps
(`src/nn/nn_gpu.cu:228-231`), and the training-label wall-clock bracket
(`scripts/benchmark.cu:585-592` — `high_resolution_clock` around
`gpucompress_compress_gpu()` plus `cudaDeviceSynchronize()`, exactly as F3 says).
I additionally checked the *gate* arithmetic, which F17 does not: Clio's `NeuroPressCost`
(`neuropress_cost.h:41-50`) floors both sides at 1 ms and caps the ratio at 100 before the
MAPE, and upstream's `gpucompress_compress.cpp:665-679` does the identical thing with the
identical constants, including the "use the prediction for `act_dt` when nothing decompressed"
rule. No divergence found.

**ctest re-run: 16/17 passed, 1 failed.** `ctp_neuropress_sgd_choreography_parity` failed on the check `200 training calls allocate nothing (persistent sample buffer)`, printing `free device memory: 39657865216 -> 39672545280 (delta 14680064 bytes)`. That check reads `cudaMemGetInfo` (`parity/neuropress_sgd_choreography_parity.cu:85`), which is **process-global** free memory, and the delta is *positive* — another process released 14 MiB mid-test. Two other CUDA processes were on the device (mine and the investigator's `runs3b.sh`). **This is a flaky assertion under a shared GPU, not a port defect**; I have no reason to doubt the investigator's 17/17 on an idle device. It is worth flagging as test hygiene: that assertion fails for anyone sharing the device. Everything else — including all five upstream cross-compiled parity tests and `selection_parity_wide` (39.5 s) — passed, and my own `np_ood_upstream_exec` re-run reproduced **0/17 winner mismatches, worst relative difference 0.000e+00**.

---

## A robustness probe the investigator did not run: a *loaded* GPU

Mid-session I accidentally ran two `np_eval_exec` processes at once. The result is a usable
natural experiment, kept as `adversary/data/L0_contended.csv` (840 chunks, infer mode,
predictions and ratios bit-identical to a clean run — only the codec timings moved).
Codec kernel times inflate 2-30x: median measured ct **8.685 ms** vs **2.696 ms** clean;
the fraction of ct below the 1 ms floor drops from **29.1 % to 7.1 %**, dt from 59.7 % to 27.7 %.

| quantity | clean (f0a) | contended (L0_contended) |
|---|---|---|
| predictions, balanced regret | 0.1572 | **1.1130** |
| true ct | 0.0169 | 0.0476 |
| true ratio | 0.6340 (worse) | 1.9959 (worse) |
| best constant, balanced regret | 0.0289 (a7) | 0.8049 (a23) |
| model / best-constant regret ratio | **5.4x** | **1.38x** |

Three consequences.
1. **F2 and F3 get stronger under load** — a perfect ct head is worth −96 % there.
2. **F5 gets much weaker under load**: the constant's advantage over the model shrinks from
   5.4x to 1.38x, because a fixed action cannot track a device whose relative codec costs have
   changed. F5's "a constant beats the model" is an idle-device result.
3. **F14's premise weakens under load**: the 1 ms floor truncates 7 % of measurements instead
   of 29 %, so "the floor is doing real work" is specific to an idle A100 — which matters,
   because Clio's own phase-2 exploration *creates* contention (commit 81ac5a43's own
   measurement: per-slot kernel time inflated up to 76x).

Also a caveat on the whole harness that is not in the noise section: `LastCodecKernelMs()`
is CUDA-event-bracketed (`compress.h:75-78`, `nvcomp.h:157-177`), so it is immune to *host*
load but fully exposed to *device* load. Every balanced-cost number in FINDINGS.md is
conditioned on an otherwise-idle A100. The investigator's own runs are clean (I verified
f0a vs f0c agree to a median 1.3 % on ct, 9440/9440 rows identical on everything else), so this
is a scope caveat, not an error.

---

## Section C — ranking and suggestions

**C1 (clamp raw inputs). Keep, but demote from "the single best change available", and state
the cost.** It is the best change *for the ratio objective* (bytes −10.1 % on f0a, −34.7 % at
eb=1e-3, −11.5 % on my held-out corpus; all deterministic). For the **balanced** objective it
buys regret with bytes (**+1.9 % stored bytes**, and outside four extreme regimes its regret
benefit is inside the timing-noise band). It is also the change whose measured size is most
sensitive to how much of your corpus you decided to put out of distribution.

**C2 (retrain on this deployment's labels). Agree, and it is now the *only* suggestion that is
purely offline** — under the in-memory-adaptation constraint, a retrain changes the starting
point every process adapts from, which is exactly the right lever. Sub-item (a) (label with
codec-kernel time, not the API bracket) is the one my F3 work most supports.

**C3 (per-algorithm ct log-bias). Promote.** F3 undersells it: measured out of sample it is
**−80 % balanced regret** (0.1629 → 0.0330 on held-out f0b), comparable to C1's −75 % and
without C1's corpus sensitivity. Two corrections: use **per-algorithm, not per-action**
(per-action generalises worse: 0.0530 vs 0.0330), and note it costs **+4.6 % bytes**.
**But**: it is a *balanced-objective, fast-storage* fix. At 500 MB/s or below a perfect ct head
increases regret (my F2 bandwidth table), so C3 would be actively harmful there. Gate it on
bandwidth, not just on an env var.

**C4 (raise the cap only for `--best`). Agree, and it is under-sold.** F4's own lossless
numbers are the small case. At eb=1e-3 the ratio-only regret goes **35.99 → 0.435** and bytes
**0.4314 → 0.3824**. Note also that ~25 % of chunks are decided by the action-index tie-break
*even with the cap removed* (the F1 collapse chunks), so C4 fixes about 60 % of the tie problem,
not all of it.

**C5 ("leave the online learner off by default"). Out of bounds — the learner is the product.**
Rewritten as an adaptation question below. F8's `f3` row (ratio-objective learning) must be
struck outright: `--best` disables both SGD phases (`compressor_runtime.cc:1577, 2290`;
`gpucompress_compress.cpp:705`).

**C6 (refuse/warn on non-float payloads, validate finiteness). Agree, cheapest item in the
list, and narrower than stated** — the defect is device-path-only; the host path already
honours `data_type_`.

**C7 (do not lower the 1 ms floor). Agree, best-supported negative recommendation.** Add the
scope: it is an idle-device result (7 % vs 29 % of ct below the floor when the GPU is loaded).

**C8 (measure against a fixed action before shipping). Agree for ratio; weaken for balanced.**
In-distribution, with the size bound applied, the balanced gap is 1.23x, not 4.3x; and under
device load it is 1.38x. The ratio-only half is unanswerable: the per-chunk oracle beats the
best constant by 0.007 % on this corpus, so no selector can earn its keep there.

**Ranking.** By decision impact per unit of change, and separating what helps the *in-run
adaptation path* from what only helps an *offline retrain*:

| | change | helps | objective | risk |
|---|---|---|---|---|
| 1 | C2 retrain with codec-kernel ct labels + size coverage | starting point of every process | both | none to faithfulness |
| 2 | C4 raise cap in `--best` only | inference | ratio-only (huge at eb>0) | none (mode-scoped) |
| 3 | C3 per-**algorithm** ct log-bias | inference | balanced **at fast storage only** | bytes +4.6 %; harmful at ≤500 MB/s |
| 4 | C1 clamp raw inputs | inference | ratio-only clearly; balanced trades bytes | corpus-composition sensitive |
| 5 | C6 finiteness/type guard | inference | correctness | none |
| 6 | C7 keep the floor | inference | balanced | none |

---

## Missed by the investigator

**M1. F2's ordering is not a property of the model — it is a property of `bw = 5e6`.**
At 500 MB/s and below, a perfect ratio head helps and a perfect ct head *hurts*
(table in F2 above). Since `g_measured_bw_bytes_per_ms` is a single upstream global with an
env override, this is a one-line deployment question that changes which head to fix. Nothing
in FINDINGS.md tells a reader that.

**M2. C1's balanced-objective byte cost.** Deterministic +1.9 % stored bytes on f0a
(0.67575 → 0.68848), up to +7.6x on individual regimes (`stepped` 0.0078 → 0.0593). F1 reports
only ratio-only bytes and balanced regret.

**M3. The `--best` path disables both SGD phases.** So F8's `f3` row, and any recommendation
about "learning under the ratio objective", is about an unreachable configuration
(`compressor_runtime.cc:1577, 2290`; the runtime even logs "both SGD phases are off").

**M4. Exploration is default-OFF** (`compressor_tasks.h:103`), so F6's K=0 row *is* the
shipped default rather than a strawman — a stronger statement than F6 makes.

**M5. Clio already ships F9's fix.** `CLIO_NEUROPRESS_SGD_ON_RATIO=1` +
`CLIO_NEUROPRESS_RATIO_MAPE_THRESH` (`compressor_runtime.cc:1536-1548`) is exactly the
"gate on the ratio head too" remedy, and `np_eval` exposes it as `NPE_GATE=ratio_or_cost`.
Neither F9 nor C5 mentions it or measures it. (I did — see the battery.) The comment at that
site also claims the io term is "~0.4 % of the cost" at 4 MiB; measured it is mean 12.4 %,
median 6.5 %.

**M6. The ratio head's error is per-CHUNK, not per-algorithm.** Removing one log constant per
chunk cuts ratio |log err| 1.060 → **0.241 (−77 %)**; per-algorithm does nothing (+3 %). That
is the missing diagnosis behind F2's "true ratio makes balanced worse": a per-chunk level
offset cancels inside a chunk's ranking but not against the ct/dt terms.

**M7. The PSNR head is clamped to [0,120] while its own analytic label reaches −283.6 dB**
(34/440 quantized chunks in f1 have a negative analytic PSNR). On those chunks the head cannot
represent its target at all, so every SGD step there carries a permanent error — the same
effect as F16/E11 but roughly 100x larger. Not in F11 or F16.

**M8. The F1 collapse is head-asymmetric.** At mad ≈ 20 (`lammps_pos_*`, `lammps_force`) the
ratio head is already a per-chunk constant (1 distinct value over 16 actions) while the ct head
still resolves 15-16 distinct values. "Collapses to one prediction for all 32 actions" only
holds above mad ≈ 40.

**M9. Two top-1 conventions are mixed across findings without a note.** F1/F8's 0.407 and 0.383
come from a ground truth that caps the *measured* ratio at 100 (the CSV's `is_best_*` columns);
F2/F14's 0.378 and 0.298 come from an uncapped ground truth. I reproduced both from the same
CSV. It does not change any sign, but F1's "0.407 → 0.486" compares a capped baseline with a
value that happens to be identical under both conventions.

**M10. In-range small chunks are also badly predicted.** The `sizes` regime's 0.25 MiB point
(median ratio APE **4.36**) is worse than its 1 MiB point (1.63) and is well inside the
training range. F10's table starts at 1 MiB.

**M11. F3's "per-action (32 keys)" is 16 keys at eb=0**, and about a third of its extra error
reduction is degrees-of-freedom inflation (a *random* 32-way grouping gets −21 %). Out of
sample it generalises worse than per-algorithm.

**M12. `LastCodecKernelMs()` is CUDA-event-bracketed, so it is immune to host load and fully
exposed to device load.** Every balanced-cost claim is conditioned on an idle GPU; I measured
what a loaded one does to them (section above). Given that Clio's own phase-2 exploration
creates device contention, this is a live deployment condition, not a lab artefact.

---

## The learning loop, judged as **in-memory adaptation from fresh weights** (F7 / F8 / F9 / C5)

The product is: every process loads `model.nnwt`, adapts in memory, discards on exit. So the
question is not "is the learner net-positive on a benchmark", it is "does a fresh process end
up better than it started, over a realistic run, and is that robust to chunk order, regime mix
and run length".

### L0. How far can a process move at all? (from the investigator's own weight dumps)

`*.weights.csv` records `l2_drift` and `w_norm` every chunk. Over the full 590-chunk run:

| run | SGD calls | final L2 drift | as % of ‖W‖=20.10 | coherent walk would be | drift at ~25 % of the calls |
|---|---|---|---|---|---|
| f2 (gate 0.30) | 45 | 0.1013 | 0.50 % | 0.14 | 0.0384 |
| f3 (ratio gate) | 226 | 0.2050 | 1.02 % | 0.68 | 0.1185 |
| f4 (every chunk) | 590 | 0.1817 | **0.90 %** | 1.77 | 0.1174 (at 148 calls) |

Two things follow, and neither is in FINDINGS.md.
1. **Run length buys almost nothing.** 590 steps move the weights 0.90 %; 148 steps already
   move them 0.65 % of that. The trust step (`clamp(0.08·mean|err|, 1e-4, 0.02)`), the EMA 0.85
   and the anti-flip ×0.5 damping cap the excursion, and because the direction depends only on
   the *sign* pattern (F7), consecutive steps on different data largely cancel: observed 0.18
   against 1.77 for a coherent walk of the same 590 steps. **A longer process does not become a
   better-adapted process.** This is the single most important fact for the deployment mode and
   it is measurable from data the investigator already had.
2. The outputs are nevertheless exponentially sensitive to that drift (E3: 20 steps on one
   action moves lz4's predicted ratio 20.5 → 1.21). So "small drift" is not "harmless drift".

### L1. Paired, per-chunk: does learning change the pick for better or worse?

Scoring *both* runs against **one** ground truth (f0a's measured table) so timing noise cannot
favour either, on the investigator's own runs (same seed, same chunks, balanced):

| run | picks changed | better | worse | total cost delta | top-1 | bytes | mean regret |
|---|---|---|---|---|---|---|---|
| f0a infer | — | — | — | — | 0.378 | 0.67575 | 0.1572 |
| f2 gate 0.30 | 125/590 | 42 | 83 | **−16.4 ms** | 0.290 | 0.68043 | **0.1544** |
| f4 every chunk | 357/590 | 135 | 222 | **+149.8 ms** | 0.198 | 0.70083 | 0.2795 |

**F8's headline needs qualifying at the default gate.** At the shipped threshold, learning is
worse on 2 chunks for every 1 it improves, worse on top-1 and worse on bytes — but the total
measured cost it saves is **negative**, i.e. slightly *better*, because its few wins are large.
The investigator's own F8 table shows the same thing (f2 regret 0.1462 < f0a 0.1572) and the
text calls it a loss anyway. F4 (gate off) is unambiguously worse and F8 is right about it.

There is also no adaptation trend within a run: by quartile, f2's (better, worse) counts are
Q1 (0,11), Q2 (28,2), Q3 (1,20), Q4 (13,50) — it tracks the *regime block*, not elapsed time.
That is the catastrophic-interference signature, and it is why the blocked-schedule runs cannot
answer the deployment question on their own.

### L2. **The gate selects for out-of-distribution chunks, so the learner trains almost only on OOD data.** (New — the sharpest mechanism I found, and it is not in F8 or F9.)

The harness already records `n_x_oob` (how many of the 8 standardized inputs fall outside the
`.nnwt` `x_mins`/`x_maxs`). Counting where phase-1 actually fires, on the **investigator's own
runs**:

| run | SGD calls | fired on a chunk with ≥1 out-of-bounds input | top regimes |
|---|---|---|---|
| f2 (gate 0.30, shipped) | 45 | **42 = 93 %** | smooth_a1000 20, sizes 8, gs_x1e4 7, turb_x50 6 |
| f3 (ratio gate) | 226 | 139 = 62 % | smooth_a1000, lammps_pos_*, lammps_force, sizes, turb_x50 |
| f4 (train every chunk) | 590 | 234 = **40 %** (the corpus base rate) | uniform |
| my L1 (same gate, 560 chunks, different schedule) | 51 | 47/51 = 92 % in the same four regimes | smooth_a1000 20, sizes 14, gs_x1e4 7, turb_x50 6 |

The cost MAPE is large exactly where the model is extrapolating (F1), so the shipped gate is an
**out-of-distribution detector**, not an error detector. Phase 1 then applies a *global* trunk
update — one that F8/E3 shows moves all 32 predictions at once — computed entirely from data
outside the training manifold, to every subsequent in-distribution chunk. That is a cleaner
explanation for "learning loses top-1 and bytes" than cross-action interference alone: the
learner is not merely imprecise, it is **trained on the wrong 8 % of the corpus**.

It also predicts an interaction the investigator did not test: with the F1 clamp on, those
chunks stop being extrapolations, the gate should mostly stop firing, and learning should
become nearly inert. I ran it (`N0`/`N1` below).

### L3. What adaptation *would* work, measured

F7 is right that trunk SGD can only learn "too high / too low", never "by how much", for a
single sample. The obvious replacement is to learn the magnitude **outside** the network, where
no gradient clip can divide it out: an **online per-algorithm log-residual table on the time
heads**, updated ungated (EMA) from whatever was actually measured. `np_eval` already
implements exactly that (`NPE_BIAS`), so it can be measured rather than argued.

Offline simulation first (my script, on-policy: only the picked action updates the table,
f0a's chunk order, balanced, scored against f0a's measured truth):

| table | mean regret | top-1 | bytes | 1st half | 2nd half |
|---|---|---|---|---|---|
| none (shipped) | 0.1572 | 0.378 | 0.67575 | — | — |
| per-algo, ct+dt, α=0.3 | 0.0846 | 0.159 | 0.72637 | 0.1023 | 0.0669 |
| per-algo, ct+dt, α=0.6 | **0.0576** | 0.168 | 0.70529 | 0.0830 | **0.0323** |
| per-action, ct+dt, α=0.6 | 0.0622 | 0.319 | 0.69682 | 0.0934 | 0.0309 |
| per-algo, ct+dt+**ratio**, α=0.6 | 0.0905 | 0.332 | 0.70415 | 0.1083 | 0.0728 |
| per-algo, ct+dt, α=0.3, **+K=3 explored labels** | **0.0433** | 0.137 | 0.72926 | 0.0399 | 0.0467 |

Three things trunk SGD cannot do that this does:
- it **improves with elapsed run length** (0.083 → 0.032 first half vs second half at α=0.6);
  trunk SGD shows no such trend at any gate setting I tried;
- adding the **ratio** head to the table makes it worse (0.0576 → 0.0905), independently
  reproducing F3's "a per-algorithm constant does nothing for ratio";
- it costs the same currency as C3 — **bytes up ~4 %** and top-1 down — because it is C3,
  learned online.

The live runs `M1`-`M4` below test the same thing through the real predictor rather than my
simulation.

---

## A statistic-level caveat that applies to F1, F2, F5, F6, F8 and F14 at once

Every one of those claims headlines a **mean** regret. On f0a the balanced regret distribution
is min 0.0000 / p25 0.0000 / **median 0.0097** / p75 0.0369 / p90 0.0803 / p99 6.907 /
max 7.081, and the **top 10 chunks out of 590 contribute 52.6 % of the total**. So "mean regret
0.1572" is a statement about roughly 2 % of the corpus; on the typical chunk the shipped model
is within 1 % of optimal.

I checked whether the claims survive on the median, and **they do**, which is why this is a
caveat and not a refutation: clamp 0.0097 → 0.0006 (−94 %), true ct 0.0097 → 0.0027 (−72 %),
true ratio 0.0097 → 0.0138 (+42 %, still worse). The directions are all preserved. But every
"−75 %"/"−89 %" figure in FINDINGS.md is a tail statistic and should be quoted with the median
beside it.

### L4. **The adaptation trajectory is a function of GPU load, not of model error.** (New, and I think it is the most important practical result in this review.)

`L1` and `L2` are byte-identical invocations — same binary, same seed, same 560-chunk schedule,
same gate 0.30, fresh weights each time. They are not replicates of each other in any useful
sense:

| run | median primary measured ct | phase-1 SGD calls | top-1 balanced | regimes the calls came from |
|---|---|---|---|---|
| L1 | 0.258 ms | **51** | 0.291 | smooth_a1000 20, sizes 14, gs_x1e4 7, turb_x50 6 (all OOD) |
| L2 | **3.540 ms** | **356** | **0.186** | spread over *every* regime (gs_stripes 18, sparse 17, gs_chaos 16 …) |

The first divergence is at chunk 2: `gs_spots` #2 measured `act_ct = 0.177 ms` in L1 and
`7.141 ms` in L2, on identical bytes with the same codec. The cause is device load — I later
found the **investigator's own `runs3b.sh` running np_eval on the same A100 from 09:04**, which
is exactly the shared-GPU hazard the brief warns about, and `nvidia-smi
--query-compute-apps` confirmed two concurrent `np_eval_exec` processes.

Rather than discard it, this is the finding: the phase-1 gate is
`|actual_cost − predicted_cost| / actual_cost > 0.30` with `actual_cost` built from a *measured*
codec time. On a loaded device every measured time inflates, so `actual_cost` moves away from
`predicted_cost` everywhere, and the gate opens 7x more often — on in-distribution chunks it
would otherwise never touch. The learner therefore adapts **to the machine's load**, not to the
model's error, and the resulting selection is worse (top-1 0.291 → 0.186). In production Clio
compresses on the same GPU as the application that produced the data, so this is the normal
case, not a lab artefact.

Consequences for the findings:
- **F8's confidence claim does not hold.** "Four runs" are four different configurations, not
  four replicates; I have two true replicates and they differ by 7x in SGD count and 0.105 in
  top-1. Any single-run learning number — including f2's 45 calls / 0.329 top-1 — is one draw
  from a very wide distribution.
- It gives the second half of the L2 mechanism: the gate is an OOD detector on an idle GPU and
  a *load* detector on a busy one. Neither is "the model is wrong here".
- It is the strongest argument against widening the gate (C5's one surviving recommendation)
  and for gating on something deterministic instead — `n_x_oob`, or the ratio MAPE, which is
  timing-independent.

---

## Reproduction (adversary side)

Everything lives under `.../npreview/adversary/`. Nothing tracked was modified; the two new
harness files are in the **gitignored** parity dir
(`git status --porcelain .../parity/` is empty).

| file | what |
|---|---|
| `scripts/adv.py` | my own scorer: head substitutions, top-1, mean/median/p90 regret, bytes, best constant. Written from the claim definitions, not from `an*.py`/`oracle*.py`. `adv.py CSV [balanced\|ratio] [gtfloor] [cap] [regimes]` |
| `scripts/learn.py` | scores several runs' picks against ONE common measured table, by quartile of chunk position |
| `scripts/runsA.sh` | held-out corpus: `NPE_REGIMES=nyx_density,warpx_field,vortex2d,cell_counts` (the `opt_in` regimes `NPE_REGIMES=all` excludes), seed 777, 25 chunks each, ±clamp, eb 0 and 1e-3 → `h1`-`h4` |
| `scripts/runsL2.sh` | learning battery, explicit `NPE_SCHEDULE` (28 regimes × 20), seed 1234: `L0` infer, `L1`/`L2` learn gate 0.30 (replicates), `L4` reversed regime order, `L5` learn+`NPE_EXPLORE_K=3`, `L6` `NPE_GATE=ratio_or_cost`, `L7` `NPE_GATE=always`, plus single-regime 250-chunk infer/learn pairs |
| `scripts/runsP.sh` | `M1`/`M3` online residual table (`NPE_BIAS=1 NPE_BIAS_KEY=algo NPE_BIAS_HEADS=ct,dt`), `N1` learn+clamp, then `np_adv_exec`, `np_sgd_exec`, `np_ood_upstream_exec`, `ctest -R ctp_neuropress` |
| `parity/np_adv.cu` (new, gitignored) | A1 batched vs A1b single-sample SGD magnitude sensitivity; A2 `‖ΔW‖` vs `|d5[0]|` with the other heads' labels set to the model's own predictions; A3 rank movement under single-action training; A4 a 60-step phase-1 loop on one real chunk with its measured labels |
| `data/L0_contended.csv` | an accidental two-process run, kept as a *loaded GPU* probe |

Build: `cd /home/cc/clio-core/build && cmake . && make np_eval_exec np_sgd_exec np_ood_upstream_exec np_adv_exec -j16`.

**Contention warning for anyone re-running this.** From 09:04 the investigator's own
`investigator/scripts/runs3b.sh` was running `np_eval_exec` on the same A100 as my battery
(`nvidia-smi --query-compute-apps` showed two concurrent processes, and `ps` identified the
script). Every codec time measured on either side after that is contended. My `h1`-`h4`
held-out runs (08:38-08:41) and all of the investigator's `f*`/`g*` runs (07:47-08:23) are
clean; my `L2` onward are not, which is how the L4 finding above was discovered.

**Deterministic confirmation of L4, no new runs needed.** Recomputing `error_pct` exactly as
`compressor_runtime.cc:1447-1462` does (dt = the prediction on both sides) over f0a's 590
primary rows, with every *measured* compress time multiplied by a device-load factor `k`:

| k (device load on measured ct) | phase-1 gate opens on |
|---|---|
| 0.5x | 0.234 |
| **1.0x (idle A100)** | **0.239** |
| 2x | 0.317 |
| 4x | 0.408 |
| 8x | 0.683 |
| **14x (what L2 actually saw)** | **0.997** |
| 20x | 1.000 |

So the gate rate is a monotone function of how busy the GPU is, saturating at 100 % by a ~14x
slowdown — and L2's 356/560 = 64 % sits exactly where its measured slowdown puts it. The
shipped adaptation policy is "train harder when the machine is busy", which is not a property
anyone designed.

---

## Revised verdicts on F7 / F8 / F9 under the in-memory-adaptation constraint

**F7 — CONFIRMED for phase 1, REFUTED as stated for phase 2.** The mechanism is correct and I
verified it line by line, and phase 1 (the default path, since exploration is off) trains on
exactly one sample, so the invariance is the operative regime. But `out_grad` accumulates
*across samples* before the per-output clip, so in a phase-2 batch of up to 7 the relative
weight of the samples is set by their `|d5|` and does **not** cancel — F7's blanket "the online
learner can only ever learn 'too high / too low', never 'by how much'" is wrong for the batched
path. (`np_adv.cu` A1 vs A1b, run: in a 7-sample batch, varying only sample 0's ct label gives
`cos(dW0,dWi)` = 0.999482 / 0.995219 / 0.980546 / **0.909978**, against an exact
**1.000000000** for the single-sample control within a sign class. Full table in the
`np_adv.cu` section below.)
Presentation note: F7 quotes E1 as "cos = 1.000000000"; E1's own output has
`cos(dW[0], dW[3]) = 0.391` on the third comparison, where the ct error changes sign. That is
consistent with the claim (direction depends on the sign pattern) but the quote is selective.

**F8 — CONFIRMED that learning does not help; the stated mechanism is incomplete, one row is
invalid, and the confidence is not supported.**
- Invalid: the `f3` ratio-objective row (`--best` disables both SGD phases).
- Not supported: "high confidence … four runs, and bytes are deterministic". Bytes are
  deterministic *given the pick*, and the pick is not: two true replicates of the identical
  configuration gave 51 vs 356 SGD calls and top-1 0.291 vs 0.186 (L4 above).
- Incomplete mechanism: cross-action interference (E3) is real — I re-derive it in `np_adv.cu`
  A3 — but the larger effect is *what the learner is trained on*. 93 % of default-gate SGD calls
  fire on out-of-distribution chunks on an idle GPU, and ~100 % of chunks fire on a loaded one.
- Qualification: at the shipped gate, learning is roughly regret-neutral (f0a 0.1572 vs f2
  0.1544 on a common ground truth; L0 0.1380 vs L1 0.1405) while losing top-1 (0.378 → 0.290;
  0.359 → 0.266) and bytes (+0.7 %). Its per-chunk record is 42 better / 83 worse but a *net*
  −16.4 ms of measured cost. "Makes selection worse" is true for top-1 and bytes and false for
  total cost. Training every chunk (f4) is unambiguously worse (+149.8 ms, 135 better / 222
  worse, regret 0.1572 → 0.2795).

**F9 — CONFIRMED numerically; the recommendation attached to it (C5 "do not widen the gate") is
right for the wrong reason and misses the knob that already exists.** The 50.4 % figure
reproduces exactly. But the reason not to widen the *cost* gate is not "learning is bad" — it
is that the cost gate is not measuring model error at all (it is an OOD detector on an idle
device and a load detector on a busy one). The ratio-MAPE gate that Clio already ships
(`CLIO_NEUROPRESS_SGD_ON_RATIO`, `compressor_runtime.cc:1536-1548`) is timing-**independent**,
which makes it the only deterministic trigger available; whether it helps is an empirical
question the investigator did not ask and I did (`L6`).

---

## Held-out corpus, claim by claim

The four `opt_in` regimes (`nyx_density` mad 6.53, `warpx_field` mad 536.5, `vortex2d`
mad 0.405 — the only in-range one — `cell_counts` mad 13.03), 25 chunks each, seed 777, never
touched by `NPE_REGIMES=all`. Runs `h1`-`h4`, made on an idle GPU (08:38-08:41).

| claim | on the investigator's corpus | on my held-out corpus | verdict |
|---|---|---|---|
| F1 clamp, balanced regret | 0.1572 → 0.0395 | **2.9334 → 0.0264** | confirmed, larger |
| F1 clamp, ratio-only bytes | 0.6710 → 0.6033 | **0.7336 → 0.6496** | confirmed |
| F1 clamp, eb=1e-3 ratio bytes | 0.4314 → 0.2817 | **0.6266 → 0.4838** | confirmed |
| F2 true-ct vs true-ratio | 0.0169 vs 0.6340 | **0.1750 vs 5.8440** | confirmed, same sign |
| F5 model vs best constant, balanced | 0.1629 vs 0.0285 | **2.9334 vs 0.0060** | confirmed |
| F5 model vs constant, ratio bytes | 0.6721 vs 0.5883 | **0.7336 vs 0.6326** (= oracle) | confirmed |
| F5, does the *clamped* model beat the constant? | — | no: 0.0264 vs 0.0064; 0.6496 vs 0.6326 | confirmed |
| F12 shuffle recall | 0.458 | **0.333** (precision 1.000, exact-tie 48.7 %) | confirmed, worse |
| **F4 cap ablation** | ratio regret 0.5545 → 0.2331 | **no effect at all** (0.1514 either way) | **corpus-specific** |

The F4 line is the one that does not travel: on this corpus the predictions do not saturate the
100x cap (they collapse *low*, not high), so removing the cap changes nothing. F4's effect size
is a property of which regimes push the ratio head upward, and F4 does not say so.

---

## `np_adv.cu` — my own SGD probes (independent of `np_sgd.cu`)

Operating point: the real `gs_spots` chunk 0 from f0a (entropy 5.9868, mad 0.0057442,
d2 0.0115946, 4 MiB), so it is a chunk that actually occurs, not a synthetic feature vector.

**A1b (single sample, phase-1 shape) — F7's core claim, confirmed harder than the investigator
showed.** Varying only the ct label, with everything else fixed:

| ct label | 2.70 | 2.40 | 2.00 | 1.60 | 0.80 |
|---|---|---|---|---|---|
| `cos(ΔW(2.70), ΔW(·))` | 1 | **1.000000000** | 0.882252023 | **0.331276027** | **0.331275860** |

The model predicts ct = 1.984 here, so 2.70/2.40 are one sign, 2.00 is ≈ zero error, and
1.60/0.80 are the other sign. Within a sign class the direction is identical to **9 decimal
places**, and the two opposite-sign cases agree with each other to 6 places while a 2x change in
label magnitude separates them. That is exactly F7's mechanism, reproduced on a different
operating point with different code.

**A1 (batched 7-sample, phase-2 shape) — F7 does NOT hold.** Same ct labels on sample 0 inside a
7-sample batch: `cos(ΔW0, ΔWi)` = 0.999482, 0.995219, 0.980546, **0.909978**. The per-output
clip is applied to `out_grad` *after* it has accumulated over samples, so the samples' relative
weights are set by their `|d5|` and do not cancel. F7's blanket impact sentence ("the online
learner can only ever learn 'too high / too low', never 'by how much'") is false for phase 2.

**A2 — and it is false for phase 1 too, whenever only one head is wrong.** Setting the ratio and
psnr labels to the model's *own predictions* so that only ct carries error:

| ct label / pred | 1.000x | 0.999x | 0.990x | 0.950x | 0.900x | 0.800x | 0.600x | 0.400x | 0.200x | 0.050x |
|---|---|---|---|---|---|---|---|---|---|---|
| \|d5[0]\| | 0.0000 | 0.0010 | 0.0097 | 0.0492 | 0.1001 | 0.2076 | 0.4499 | 0.7408 | 1.1046 | 1.4533 |
| ‖ΔW‖ | 1.497e-5 | 1.497e-5 | 1.497e-5 | 1.497e-5 | 4.006e-4 | 8.306e-4 | 1.800e-3 | 2.963e-3 | 3.000e-3 | 3.000e-3 |

Three regimes, all visible at once: below the **0.10 noise gate** the ct error is discarded
entirely (‖ΔW‖ pinned at 1.5e-5, the residual from the other heads); between 0.10 and ≈0.75
‖ΔW‖ is **exactly linear** in |d5[0]| (slope 4.00e-3 to three digits at every point); above that
the **trust step saturates** at `step_max × (1−EMA) = 0.02 × 0.15 = 3.0e-3`. Direction is
sign-only throughout (`cos = 0.574626` to 5 digits across the entire open region).
So: F7's *mechanism* paragraph is right and even says this ("magnitude survives only through
step = clamp(0.08·mean|d5_raw|, …), which saturates once mean|d5_raw| ≥ 0.25"), but every
example it offers (E1, E1c) sits in the saturated regime **because their ratio label is grossly
wrong**, and the headline generalises from those examples to a claim the unsaturated regime
contradicts.

**A3 — cross-action interference is real but does not destroy the ranking.** 20 phase-1 steps on
action 0 only, watching all 16 lossless actions: ratios move a0 41.7→1.24, a4 25.4→1.53,
a20 59.2→2.66, a5 1.23→0.28 — every action moves, as F8 says. But
**Spearman(ratio before, after) = 0.894** and **Spearman(ct) = 0.903**, and the trained action's
own ratio rank moves only 1 → 2. Selection reads the *order*, so F8's "one action's outcome
shifts all 32 predictions" is true of the values and much weaker for the decision.

**A4 — the on-policy trap, which neither F8 nor F9 names.** 60 phase-1 steps on one real chunk,
each step training on whichever action the model just picked, with that action's *measured*
labels (`gs_spots` #0 from f0a; true best a7 at cost 2.6322):

```
step  0  pick a5 (true a7)  regret 0.0146   ratio[a5] 1.234 (measured 1.251)
step 20  pick a5 (true a7)  regret 0.0146   ratio[a5] 1.252
step 60  pick a5 (true a7)  regret 0.0146   ratio[a5] 1.251   <- converged to truth
```

The learner **perfectly learns the action it already chose** (ratio 1.234 → 1.251 = the measured
value) and the pick never moves, because `ct[a5]` is pinned at the 1 ms floor and the other
seven actions are never labelled. Phase-1 self-training is structurally incapable of correcting
a wrong pick: it is on-policy with a batch of one. That is the strongest argument for
exploration (F6) and against C5's "leave the learner off" framing — the learner is not merely
unhelpful, it is *closed-loop* on its own mistake.

---

## L5. The learning battery — measured

28 regimes × 20 chunks = 560, explicit `NPE_SCHEDULE`, seed 1234, fresh weights per run. Every
run's **picks are scored against one common measured table (L0's)**, so timing noise cannot
favour a run; `bytes` is the deterministic stored size of the picked action.

The GPU was shared with the investigator's `runs3b.sh` from 09:04, so the runs split into two
contention groups. **Compare only within a group** — that is the whole point of L4 above.

| run | config | GPU | SGD calls | of which on an OOB chunk | mean regret | median regret | bytes | top-1 |
|---|---|---|---|---|---|---|---|---|
| **L0** | infer, no learning | idle | 0 | — | 0.1380 | 0.0138 | **0.67837** | **0.359** |
| **L1** | learn, gate 0.30 (shipped) | idle | 51 | **42 = 82 %** | 0.1405 | 0.0138 | 0.68323 | 0.266 |
| L2 | learn, gate 0.30 (replicate of L1) | busy | **356** | 150 = 42 % | 0.2697 | 0.0328 | 0.73506 | 0.177 |
| L4 | learn, gate 0.30, **reversed regime order** | busy | 261 | 76 = 29 % | 0.1515 | 0.0276 | 0.72256 | 0.145 |
| L5 | learn, gate 0.30, **+ phase-2 explore K=3** | busy | 364 | 163 = 45 % | 0.1365 | 0.0342 | 0.73495 | **0.102** |
| **L6** | learn, `NPE_GATE=ratio_or_cost` (the shipped `CLIO_NEUROPRESS_SGD_ON_RATIO` knob) | busy | 347 | 176 = 51 % | 0.2071 | 0.0263 | **0.70677** | **0.350** |

Median measured ct on the picked action: L0 0.252 ms, L1 0.258 ms (idle) vs L2 3.540, L4 5.179,
L5 3.579 ms (busy) — a 14-20x device slowdown, which is what drove the SGD counts from 51 to
261-364.

**Idle-GPU comparison (L0 vs L1), the only clean one.** Learning at the shipped gate is
**regret-neutral** (0.1380 → 0.1405 mean, 0.0138 → 0.0138 median), costs **+0.7 % stored bytes**
and **−26 % top-1** (0.359 → 0.266). Same conclusion as the paired analysis of the
investigator's own f0a/f2. F8's direction is right on top-1 and bytes; its magnitude is small
and its regret claim does not hold at the default gate.

**Busy-GPU group.** Every learning configuration is much worse than the idle baseline on bytes
(0.72-0.74 vs 0.678, **+6.5 to +8.4 %**) and on top-1 (0.10-0.18 vs 0.359). Order does not
rescue it: L4 runs the 28 regimes in reverse and lands in the same place. **Phase-2 exploration
does not rescue it either**: L5 has the best mean regret of the learning runs (0.1365) and the
*worst* top-1 (0.102) and near-worst bytes — because `np_eval`'s phase 2 trains on the explored
batch but does not adopt, so it adds training signal without adding the safety net that F6
credits.

**On F8's "train every chunk is worse":** confirmed, and the reason is now visible. It is not
that more training is worse per se — it is that the *shipped trigger* is not an error detector.
On an idle device it fires 82 % of the time on extrapolation chunks; on a busy one it fires
everywhere. Neither population is "chunks where the model is wrong in a way SGD can fix".

---

## Scoreboard

| # | claim | verdict | one-line reason |
|---|---|---|---|
| F1 | absolute-magnitude features → OOD collapse; clamp fixes it | **CONFIRMED** (balanced half **WEAKENED**) | replicates exactly and gets *bigger* on my held-out corpus (regret 2.9334→0.0264); but 4 of 28 regimes carry 86 % of the corpus regret, outside them the balanced gain is inside the timing-noise band, and the clamp costs **+1.9 % balanced bytes** (not reported) |
| F2 | ct head owns balanced regret, ratio head owns ratio-only, dt owns nothing | **CONFIRMED, over-generalised** | table reproduces to 4 dp and survives removing the cap (true-ratio 0.6340→0.6668); but the ordering **inverts below ~500 MB/s** — at 50 MB/s a perfect ct head *raises* regret 0.2297→0.2633 |
| F3 | ct head 21x too narrow; per-algorithm error is a constant | **CONFIRMED, UNDER-stated** | all four numbers reproduce (96.8x / 4.52x / Spearman 0.750 / 29.1 %); a random 8-way null gives +3 %, so the effect is real; and the fix is worth **−80 % regret out of sample**, which F3 never measures |
| F4 | 100x cap decides 43 % of lossy chunks by tie-break | **CONFIRMED, UNDER-stated, corpus-specific** | lossless numbers exact; the missing headline is eb=1e-3 ratio-only regret **35.99 → 0.435**; but ~25 % of chunks tie even with no cap, and on my held-out corpus the cap changes **nothing** |
| F5 | a fixed action beats the model out of sample | **CONFIRMED on ratio, WEAKENED on balanced** | reproduces exactly and survives a held-out corpus (2.9334 vs 0.0060); but their "in-dist" set ignores the `.nnwt` **size** bound — with all four bounds applied the balanced gap is 0.0297 vs 0.0242 (**1.23x, not 4.3x**) and under device load it is 1.38x |
| F6 | exploration, not the network, makes the path competitive | **CONFIRMED, UNDER-stated** | my simulator matches to 3 dp (K=3 → 0.0708 / 0.541 best-in-window); K=31's 0.0005 is really ~0.0092 out-of-sample; and **exploration is default-OFF**, so F6's K=0 row *is* the shipped default |
| F7 | one SGD step is invariant to error magnitude | **CONFIRMED for phase 1, REFUTED for phase 2** | my own probe gets `cos = 1.000000000` within a sign class, but a 7-sample batch gives **cos 0.910**, and with only one head erring `‖ΔW‖` is **exactly linear** in \|d5\| between the 0.10 gate and trust-step saturation |
| F8 | online SGD makes selection worse; cause is cross-action interference | **CONFIRMED in direction, mechanism INCOMPLETE, one row INVALID, confidence REFUTED** | `f3` (ratio-objective learning) cannot occur — `--best` disables both SGD phases; two true replicates gave **51 vs 356** SGD calls and top-1 0.291 vs 0.186; interference leaves Spearman 0.894, so the real cause is *what the gate selects* |
| F9 | the gate never opens on half the ratio-bad chunks | **CONFIRMED** | 23.9 / 47.5 / 23.6 / 141-of-280 reproduce exactly; but Clio already ships the fix (`CLIO_NEUROPRESS_SGD_ON_RATIO`), unmentioned, and the code comment there says io is "~0.4 %" of cost when it is 12.4 % |
| F10 | chunks > 4 MiB are OOD; size is the least-used input in range | **CONFIRMED out-of-range, over-claimed in-range** | 8 MiB median ratio APE 46.2 reproduces; but "least-used" rests on a one-point marginal sweep, and their own E5 disagrees with E7; the omitted 0.25 MiB point (APE 4.36) is worse than 1 MiB and is *in* range |
| F11 | PSNR head cannot rank; masking discards a third of the space | **CONFIRMED, one number withdrawn** | 1-distinct-actual on 440/440 and all six masking rates reproduce exactly; but "median APE 22.2 %" does not reproduce under either obvious definition, because `act_psnr` reaches **−283.6 dB** and a percentage error there is undefined |
| F12 | the model is blind to the shuffle half | **CONFIRMED, UNDER-stated** | every figure reproduces (89.3 / 43.4 / 0.943 / 0.458 / 32.9 / 27.7 %); on the held-out corpus recall falls to **0.333** with 48.7 % exact ties |
| F13 | non-float32 read as float32; a NaN feature → "always lz4" | **CONFIRMED, scope narrower** | E9/E10 reproduce exactly (mad 3.02e36, NaN → 1/8 distinct → lz4); but the defect is **device-path only** — the host path honours `data_type_` |
| F14 | the 1 ms floor hides error; lowering it is worse | **CONFIRMED, if anything under-confident** | every cell reproduces, and the direction repeats on the timing replicate (0.246/0.7781 → 0.175/0.9578) with a between-run spread ⅓ of the effect; scope: on a loaded GPU only 7 % of ct sits under the floor, not 29 % |
| F15 | the decompression head trains toward a constant and does not matter | **CONFIRMED** | a perfect dt head is worth −0.036 regret against −0.140 for ct; 59.7 % of dt below the floor |
| F16 | the 1e-7 sentinel is inert; the psnr label bias is second-order | **CONFIRMED** | E6/E11 reproduce (ratio 18.629 → 18.629; ct[0] 4.0892 → 3.7895); missed that the same experiment moves an *untrained* action, and that the [0,120] clamp vs a −283 dB label is the same effect ~100x larger |
| F17 | the port is faithful | **CONFIRMED** | I re-verified five upstream constants and the gate arithmetic by hand, and re-ran `np_ood_upstream_exec` (0/17 mismatches, 0.000e+00); my ctest run was 16/17 — the one failure is a `cudaMemGetInfo` assertion that is invalid on a shared GPU, not a port defect |

Nothing in F1-F17 is **REFUTED** outright except F7's phase-2 generalisation and F8's `f3` row.
Ten of the seventeen are, in my judgement, **under-stated** rather than over-stated.

---

## Three strongest attacks

1. **F2 inverts at any realistic storage bandwidth.** The whole "fix the ct head" programme
   (F2 → F3 → C3) is a consequence of upstream's hard-coded `g_measured_bw_bytes_per_ms = 5e6`
   (5 GB/s). Recomputing the same head-substitution table at 500 MB/s and 50 MB/s: a perfect
   ratio head goes from *harmful* (0.6340) to the **best** single fix (0.2314, 0.1164), and a
   perfect ct head goes from −89 % to **worse than doing nothing** (0.2633 vs 0.2297 at
   50 MB/s). `CLIO_NEUROPRESS_COST_BW` is an env knob, so this is a deployment question, not a
   hypothetical — and FINDINGS never tells the reader which regime they are in.
2. **F5's "the model loses even in-distribution" is an artefact of the in-distribution
   definition.** `oracle2.py`'s `INDIST` list is chosen on `mad ≤ 0.5` alone and still contains
   8 and 16 MiB chunks from the `sizes` regime. Applying all four `.nnwt` bounds (mad, d2,
   entropy **and size**), fit on f0a and scored on f0b, the balanced gap collapses from
   **4.3x (0.1505 vs 0.0348) to 1.23x (0.0297 vs 0.0242)**, and the model beats the constant on
   bytes by 8.8 %. The ratio-only half survives; the balanced headline does not.
3. **F8's confidence is unsupported, and the mechanism is the gate, not the network.** Two
   byte-identical replicates of the shipped learning configuration produced **51 vs 356** SGD
   calls and top-1 **0.291 vs 0.186**, because the gate's `actual_cost` is built from a measured
   codec time: I show deterministically, from f0a alone, that scaling measured ct by a device
   load factor takes the gate from 23.9 % open (idle) to **99.7 % open at 14x**. On an idle GPU
   **82-93 %** of the calls that do fire land on chunks with an out-of-bounds input. The learner
   is trained on extrapolation chunks when the machine is idle and on everything when it is
   busy — so "learning makes selection worse" is true, but it is a property of the trigger, and
   "four runs, bytes are deterministic" is not a valid confidence argument.
**L6 — the knob the investigator missed is the only learning configuration that does not cost
top-1.** `CLIO_NEUROPRESS_SGD_ON_RATIO` ORs a **ratio-MAPE** gate onto the cost gate
(`compressor_runtime.cc:1536-1548`; `NPE_GATE=ratio_or_cost` in the harness). Result: top-1
**0.350**, against 0.177 / 0.145 / 0.102 for the three other busy-GPU learning runs and 0.359
for the idle **no-learning** baseline — i.e. it is the only learning run that stays level with
not learning at all. Bytes 0.70677 are also the best of the busy group (vs 0.723-0.735), though
still worse than the idle baseline's 0.67837.

Two honest caveats. (i) L6 ran at a *lower* contention level than L2/L4/L5 — median measured ct
on the pick 1.214 ms vs 3.54 / 5.18 / 3.58 ms — so part of the gap is device state, not the
gate. (ii) But that cuts **for** the result on top-1: L6 was more contended than L1 (1.214 vs
0.258 ms) and still beat it, 0.350 vs 0.266. The ratio MAPE is computed from
`min(cap, predicted_ratio)` vs `min(cap, actual_ratio)` — both **timing-independent** — which is
exactly why it does not inherit the device-load pathology of the cost gate (L4 above). That is
the concrete, measurable version of C5's advice, and it points the opposite way: *do* widen the
gate, but widen it on the deterministic head.

**Runs I did not get to (the session was interrupted; the GPU was shared throughout).**
Explicitly **NOT RUN**: `L7` (`NPE_GATE=always` at this schedule), the six single-regime
250-chunk infer/learn pairs (`gs_chaos`, `turb`, `lammps_pos_sorted`), `M1`-`M4` (the live
online residual table), `N0`/`N1` (learn + clamp). `M1` (live residual table) and `N1`
(learn + clamp) were left running after the scoreboard was written; both completed and are
reported in the two addenda at the end of this document. So two things in this document rest on **simulation rather
than a live run**: the online-residual-table numbers in L3 (my own offline on-policy
simulation over f0a, not `NPE_BIAS=1`), and the prediction that the F1 clamp should nearly
silence the phase-1 gate (untested). Both are flagged as such where they appear.

---

## Addendum (added after the scoreboard): `N1` landed, and it confirms a prediction I had left untested

The "NOT RUN" note above said two things rested on simulation, one of them being *"the F1 clamp
should nearly silence the phase-1 gate (untested)"*. `N1` finished afterwards and tests exactly
that: same 560-chunk schedule, same seed, `NPE_MODE=learn NPE_SGD_THRESH=0.30
NPE_FEATURE_XFORM=clamp`.

I also switched the contention proxy from "median measured ct **on the picked action**" to
"median `act_ct` over **all** rows", which is the fairer measure because it does not move with
the pick. That correction changes one thing I wrote earlier and I am flagging it rather than
editing it away: **L6 was NOT less contended than L2/L5** — over all rows it is 7.138 ms against
L2's 8.250 and L5's 9.174, not the 1.214 ms the pick-only proxy suggested. So the L6 caveat
"part of the gap is device state" was wrong in my favour's *disfavour*: L6 was fully busy and
still held top-1 at 0.350. The L6 conclusion strengthens.

| run | config | median `act_ct`, all rows | chunks with an OOB input | SGD calls | mean regret | median | bytes | top-1 |
|---|---|---|---|---|---|---|---|---|
| L0 | infer, no clamp | 2.425 ms (idle) | 230 | 0 | 0.1380 | 0.0138 | **0.67837** | 0.359 |
| L1 | learn, no clamp | 2.946 ms (idle) | 230 | 51 | 0.1405 | 0.0138 | 0.68323 | 0.266 |
| L2 | learn, no clamp | 8.250 ms (busy) | 230 | 356 | 0.2697 | 0.0328 | 0.73506 | 0.177 |
| L5 | learn + explore K=3 | 9.174 ms (busy) | 230 | 364 | 0.1365 | 0.0342 | 0.73495 | 0.102 |
| L6 | learn, ratio_or_cost gate | 7.138 ms (busy) | 230 | 347 | 0.2071 | 0.0263 | 0.70677 | 0.350 |
| **N1** | **learn + CLAMP** | 2.242 ms (idle) | **0** | **3** | **0.0341** | 0.0229 | 0.68868 | **0.396** |

Three results.

1. **Prediction confirmed.** At comparable device load (N1 2.242 ms vs L1 2.946 ms), clamping the
   inputs takes phase-1 firings from **51 to 3**, a 94 % reduction, and the OOB-chunk count from
   230 to 0. This is the direct causal test of the L2 mechanism: the shipped gate fires because
   the model is extrapolating, so removing the extrapolation removes the trigger. F1, F8 and F9
   are the same defect seen from three sides, which none of the three findings says.
2. **Clamp + learning is the best configuration I measured**, on regret (0.0341, −75 % vs L0's
   0.1380) and on top-1 (**0.396** vs 0.359 for no-learning-no-clamp — the only run that beats
   the baseline). Note the learner contributes almost nothing to that: 3 SGD calls in 560 chunks.
   The gain is C1's, not the learner's.
3. **The bytes cost of C1 reappears**: 0.68868 vs L0's 0.67837, **+1.5 %**, consistent with the
   +1.9 % I measured on the investigator's own corpus. C1 buys time-regret with bytes under the
   balanced objective, in a live learning run as well as in inference.

This does not change any scoreboard verdict. It strengthens F1 (its fix also fixes the learning
loop), strengthens my "the gate is an OOD detector" attack on F8/F9, and re-confirms the
unreported byte cost that is my main charge against C1.

### Addendum 2: `M1` landed too — the online residual table is no longer a simulation

`M1` = `NPE_MODE=infer NPE_BIAS=1 NPE_BIAS_KEY=algo NPE_BIAS_HEADS=ct,dt NPE_BIAS_ALPHA=0.3`,
i.e. **no trunk SGD at all**, just an on-policy per-algorithm log-residual table on the two time
heads, updated from whatever the primary actually measured. Same schedule, same seed, and the
device was as quiet as L0/N1 (median `act_ct` 2.246 ms vs 2.425 / 2.242), so this comparison is
clean.

| run | mean regret | median regret | bytes | top-1 | 1st-half regret | 2nd-half regret |
|---|---|---|---|---|---|---|
| L0 shipped (no learning) | 0.1380 | 0.0138 | **0.67837** | **0.359** | 0.1602 | 0.1158 |
| L1 shipped trunk SGD | 0.1405 | 0.0138 | 0.68323 | 0.266 | — | — |
| **M1 residual table, no SGD** | **0.0908** | **0.0082** | 0.68282 | 0.275 | 0.1601 | **0.0215** |
| N1 learn + clamp | 0.0341 | 0.0229 | 0.68868 | 0.396 | 0.0363 | 0.0319 |

The thing to look at is the last two columns. M1 and L0 start in the same place
(0.1601 vs 0.1602 over the first 280 chunks — as they must, the table is empty) and end
**5.4x apart on the same chunks** (0.0215 vs 0.1158). That is an adaptation curve. Trunk SGD
produces no such curve at any gate setting I tried, which is what F7's sign-only update
predicts and what the 0.90 %-of-‖W‖ drift ceiling in L0 above quantifies.

So the answer to "what adaptation would actually work here" is measured, not argued: **learn the
magnitude outside the network**, per algorithm, on the time heads only, and leave the trunk
alone. It costs the same currency as C3 (top-1 0.359 → 0.275, bytes +0.7 %) because it *is* C3,
learned online. `np_eval` already implements it; the runtime does not.

Both simulation-only caveats in the "NOT RUN" note are now discharged: `N1` confirmed the clamp
silences the gate (51 → 3 firings), and `M1` confirmed the residual table adapts within a run.
The single-regime pairs, `L7`, `M2`-`M4` and `N0` remain NOT RUN.
