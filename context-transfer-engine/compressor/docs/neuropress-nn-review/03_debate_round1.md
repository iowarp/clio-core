# Debate round 1 — investigator's response

Against `../adversary/REBUTTAL.md` (1100-line final). Every attack, every new adversary claim
(L0-L6, M1-M12, the held-out table, `np_adv.cu` A1-A4, the scoreboard) gets one of
**CONCEDE** / **CONTEST** / **PARTIAL**. Conceded items are already folded into `FINDINGS.md`,
which remains the single source of truth; this file records only the adjudication and the fresh
numbers I ran to reach it.

**Standing constraint both sides now work under:** the deployment mode is *in-memory adaptation
from fresh weights* (`compressor_runtime.cc:376`, `:1465-1471`, `Save()` never called). "Turn
the learner off" is out of bounds for both of us.

**Two things I could not run.** The GPU has been shared with the adversary's battery since
09:04 and was still at 96 % utilisation with their `runsQ.sh` active when I finished. So (a) a
clean idle re-run of the ratio-MAPE gate on my own 3-block stream and (b) a clean idle re-run of
the live residual table (`h3`) are **not done**; both appear in the dispute table with the exact
command that settles them. Everything below is either CPU-only analysis of already-collected
CSVs, source verification, or a GPU probe that does not time codecs (`np_sgd_exec`).

---

## Part 1 — the seventeen findings

### F1 — CONCEDE ×3, and the corpus caveat is now in the finding

| adversary attack | verdict | what I did |
|---|---|---|
| "collapses to one prediction for all 32 actions" is over-stated; the collapse is head-asymmetric | **CONCEDE** | My own per-regime table already showed it (`lammps_pos_*`: 16 distinct ct, 1 distinct ratio) and my headline contradicted my own data. Heading rewritten to "one or both heads saturate, monotonically in amplitude"; the mad≈40 threshold is stated. |
| the clamp costs **+1.9 % balanced bytes**, unreported | **CONCEDE** | Deterministic and in my own CSVs: 0.67575 → 0.68848. Added to F1 and to C-A/A3, with the per-regime regressions (`stepped` 0.0078 → 0.0593). |
| the −75 % is corpus-composition-dependent (4 regimes carry 86 % of the regret; in-bounds it is inside the noise band) | **CONCEDE** | Their own diff (5696/5696 in-bounds predictions bit-identical) is the right argument, and they concede the "clamp hurts in-dist" reading is noise. F1 now says the balanced benefit *is* the out-of-bounds chunks and scales with how much of the corpus is out of bounds. |
| held-out corpus makes F1 **larger** (regret 2.9334 → 0.0264, ratio bytes 0.7336 → 0.6496) | **ACCEPT** (their measurement, I did not run it) | Cited as theirs in F1. |

### F2 — CONCEDE the conditioning; it is now in the title

The bandwidth inversion is the strongest attack in the rebuttal and it reproduces exactly on my
own scorer (`scripts/debate1.py`): 5 GB/s → true-ct 0.0169 / true-ratio 0.6340; 500 MB/s →
0.1635 / 0.2314; 50 MB/s → **0.2633 (worse than doing nothing)** / 0.1164. F2's title now reads
"**At 4 MiB and 5 GB/s** … **the ordering inverts below ~500 MB/s**", and C-A/A4 carries a
"gate it on bandwidth" instruction.
**CONCEDE** the io-share discrepancy too: my "mean 12.4 %, median 3.3 %" mixed two
denominators — 3.3 % is the median share of the *predicted* cost, 6.5 % of the *measured* cost.
Both are printed in my own earlier output; the write-up picked one from each row. Fixed.
**CONTEST nothing.** Their attack-1 (is it the cap?) independently confirms my result
(true-ratio 0.6340 capped → 0.6668 uncapped).

### F3 — CONCEDE the label error and the per-action recommendation; CONTEST the DOF null

- **CONCEDE**: "per-action (32 keys)" is **16** keys at eb=0.
- **CONCEDE**: F3 never measured the decision impact. Re-derived out of sample myself
  (`scripts/debate2.py`, fit on `f0a`, score on `f0b`): ct-only per-**algorithm** takes balanced
  regret **0.1629 → 0.0330 (−80 %)**; per-**action** only to 0.0530. Their number, my
  reproduction. C-A/A4 now says per-algorithm.
- **CONCEDE, extending them**: the correction is a **tail** fix. Median regret gets *worse*
  (0.0031 → 0.0073), top-1 falls 0.375 → 0.317, bytes +4.6 %. Neither of us said this.
- **CONTEST**: "a *random* 32-way grouping gives −21 %, so a third of F3's per-action reduction
  is degrees-of-freedom inflation." I cannot reproduce that. My null (median log offsets, seed 7,
  same 9440 rows): random 8-way 1.209 → **1.242**, random 16-way → **1.246**, random 32-way →
  **1.230** — i.e. no reduction at any width, against real per-algo 0.499 and real per-action
  0.441. A random grouping with ~300 rows per group cannot have a group median far from the
  global median, so −21 % is implausible; I suspect a *mean* offset over a heavy-tailed
  distribution. Their **out-of-sample** result (per-action generalises worse) is the substantive
  point and I accept it; the DOF explanation for it is not established.
- **ACCEPT and adopt (M6)**: the ratio head's error is per-**chunk**, not per-algorithm. My own
  reproduction: median |log| 1.060 → per-algo 1.094 (+3 %), per-regime **0.279 (−74 %)**,
  per-chunk **0.218 (−79 %)**. This is the missing diagnosis behind F2's "true ratio makes
  balanced worse" and is now in F3.

### F4 — CONCEDE ×3 (all under-statements)

**CONCEDE** the missing eb=1e-3 headline; re-derived through my own clamp chain: ratio-only
regret **35.99 → 0.435 (83x)**, bytes **0.43135 → 0.38236 (−11.4 %)**, ties 0.482 → 0.257.
**CONCEDE** that ~25 % of chunks tie on the action index with the cap already removed (0.431 →
0.254 lossless, 0.482 → 0.257 lossy), so the cap owns ~60 % of the tie problem, not all of it.
**CONCEDE** that the effect is corpus-specific — on their held-out regimes the predictions
collapse *low* and removing the cap changes nothing. All three are in F4 and C-A/A2.

### F5 — CONCEDE the in-distribution definition

Their strict-`.nnwt`-bounds filter (mad **and** d2 **and** entropy **and** size) is the right
one and my `INDIST` regime-name list was not: it admits `sizes` chunks of 8 and 16 MiB, exactly
the chunks F1/F10 call out of distribution. Reproduced on my own scorer (`debate1.py`, fit f0a /
score f0b): strict-bounds balanced **model 0.0297 vs const 0.0242 (1.23x, not 4.3x)** and the
model **wins on bytes 0.59355 vs 0.65076**. The ratio-only half survives (0.5665 vs 0.0001) with
the byte gap shrinking 12.5 % → 3.3 %. F5 rewritten; C-B/B5 scoped accordingly.
**ACCEPT** their held-out confirmation and their point that the ratio objective is *degenerate*
on both corpora (oracle beats best-constant by 0.007 %), which I had raised as a caveat and they
promoted to decisive — correctly.

### F6 — CONCEDE both

**CONCEDE**: `neuropress_exploration_enabled_ = false` (`compressor_tasks.h:103`) — verified
myself; also `neuropress_online_learning_enabled_ = false` (`:73`). "At the default K=3" was
wrong; the shipped path is phase 1 only, i.e. the K=0 row. Their framing is stronger and F6 now
uses it.
**CONCEDE**: adoption at K=31 is partly fitted to timing noise (0.0000 → 0.0092 when re-scored
against `f0c`). Ratio-only unaffected (bit-deterministic). Both in F6.

### F7 — PARTIAL: concede the over-generalisation, contest "REFUTED for phase 2"

**CONCEDE, and it is the fairest hit in the rebuttal.** My E1/E1c examples all sat in the
*saturated* regime of the trust step because their ratio label was grossly wrong, and I
generalised from them to "the learner can only ever learn too-high/too-low, never by how much".
I reproduced their A2 with my own probe (`np_sgd_exec` **E1e**, ratio/psnr labels set to the
model's own predictions so only ct errs) and the three regimes are unambiguous:

| \|d5[0]\| | 0.0000 | 0.0110 | 0.0558 | 0.1139 | 0.2374 | 0.5213 | 0.8743 | 1.3413 |
|---|---|---|---|---|---|---|---|---|
| ‖ΔW‖ | 1.4976e-5 | 1.4976e-5 | 1.4976e-5 | 4.556e-4 | 9.498e-4 | 2.085e-3 | 3.000e-3 | 3.000e-3 |

below the 0.10 noise gate the error is discarded; between 0.10 and ≈0.87 ‖ΔW‖ is exactly linear
(slope 4.00e-3 at every point); above that it saturates at `0.02 × 0.15`. Direction constant to
6 digits throughout. F7's title and impact paragraph are rewritten.

**CONTEST the "REFUTED for the batched phase-2 case" verdict, on method.** Their A1 varies the
ct label 2.70 / 2.40 / 2.00 / 1.60 / 0.80 against a prediction of 1.984 — so 1.60 and 0.80 are
on the *other side* of zero error. Their large departure (cos 0.910) is the sign-flipped case,
which F7 predicts; their same-sign pair gives cos 0.999482. I built my own batched probe
(`np_sgd_exec` **E1d**: 7 samples, only sample 0's ct label varied, same sign throughout) and
get ‖ΔW‖ = 2.999981e-3 / 2.999977e-3 / 2.999973e-3 / 2.999972e-3 and cos = **0.999735 /
0.998661 / 0.998154**. So their *mechanism* is right — `out_grad` accumulates across samples
before the clip, so the relative sample weights do not cancel — and I concede the strict
"cancels exactly" property is single-sample only. But the measured leak is **≤0.2 % of the
update direction and nil in the step size**, which does not support "REFUTED". Verdict:
**PARTIAL — mechanism conceded, magnitude disputed.** Settling experiment in the table below.

**CONCEDE the presentation point**: quoting E1's `cos = 1.000000000` while E1's own output has
`cos(dW[0], dW[3]) = 0.391` for the sign-flipped case was selective; F7 now shows both.

### F8 — CONCEDE most of it; CONTEST the `f3` strike

**CONCEDE — confidence.** "High confidence, four runs, bytes are deterministic" is not a valid
argument: bytes are deterministic *given the pick*, and the pick is not. Withdrawn in F8.4.
**CONCEDE — mechanism.** Their L2 is the sharpest thing in the review and it reproduces on **my
own runs** (`scripts/debate1.py`, using the harness's `n_x_oob` column): the shipped gate fires
on a chunk with ≥1 out-of-bounds input **42/45 = 93 %** of the time in `f2`, against a corpus
base rate of 40 %; `f4` (gate off) sits exactly at the base rate. The gate is an OOD detector,
not an error detector. Now F8.3, credited to them.
**CONCEDE — L4.** Their two "replicates" were contended by *my* `runs3b.sh`, so the 51-vs-356
pair is not a controlled experiment — but their deterministic k-sweep is, and it reproduces on
my `f0a` exactly: gate open 0.234 / **0.239** / 0.317 / 0.408 / 0.683 / **0.997** / 1.000 at
k = 0.5 / 1 / 2 / 4 / 8 / 14 / 20. The shipped adaptation policy is "train harder when the
machine is busy". Now F8.4.
**CONCEDE — L0.** Verified on my own `*.weights.csv`: final L2 drift 0.1013 (45 calls) / 0.2050
(226) / **0.1817 (590)** against ‖W‖ = 20.10, i.e. 590 steps move the weights *less* than 226 do.
"A longer process does not become a better-adapted process" is now F8.2.
**CONCEDE — L1's qualification.** At the shipped gate the picture is mixed rather than uniformly
bad: 42 better / 83 worse but **−16.4 ms** total cost. F8 now says "true for top-1, bytes and the
median chunk, false for total cost".
**CONCEDE — A3.** Interference leaves Spearman(before, after) = 0.894/0.903, so it moves values
far more than order; demoted to a second-order cause (F8.5).
**ACCEPT — A4** (the on-policy trap). I have no independent run of it; it is credited to them and
is now F8.6, and it is the cleanest statement of why phase 1 alone cannot work.

**CONTEST — the `f3` row is not an unreachable configuration.** `--best` does disable both SGD
phases (`compressor_runtime.cc:1577`, `:2290`) — I verified it. But `f3` is not `--best`. The
ratio-only *cost weights* are reachable through `CLIO_NEUROPRESS_COST_W_CT=0` /
`CLIO_NEUROPRESS_COST_W_DT=0`, and `:1443-1449` then builds the gate's cost from those env
weights (`kCw`) while `:1577` leaves training enabled — best mode is a *separate* flag in the
same expression. `NPE_COST=ratio` in the harness models precisely that. The row is **relabelled**
(it is about the cost-weight override, not `--best`) and **kept**. What I do concede is that my
write-up let it read as "learning under the ratio objective", which invites the `--best` reading.

### F9 — CONCEDE the omission; and a new result that refines their remedy

**CONCEDE**: `CLIO_NEUROPRESS_SGD_ON_RATIO=1` + `CLIO_NEUROPRESS_RATIO_MAPE_THRESH`
(`compressor_runtime.cc:1536-1548`, `NPE_GATE=ratio_or_cost`) is exactly the remedy F9 describes
and I neither mentioned nor measured it. Now in F9 and C-B/B2.
**CONCEDE**: the code comment there claiming the io term is "~0.4 % of the cost" is off by
1.5 orders of magnitude (measured 12.4 % mean).

**New, deterministic, mine** (`scripts/debate3.py`, f0a primary rows, both gates computed
exactly as the runtime does, with measured ct scaled by a device-load factor k):

| k | cost gate | ratio gate | cost **OR** ratio | cost: % on OOB chunks | ratio: % on OOB | OR: % on OOB |
|---|---|---|---|---|---|---|
| 0.5x | 0.234 | **0.475** | 0.476 | 0.99 | 0.66 | 0.66 |
| **1.0x (idle)** | 0.239 | **0.475** | 0.478 | 0.98 | 0.66 | 0.66 |
| 2x | 0.317 | **0.475** | 0.607 | 0.58 | 0.66 | 0.52 |
| 8x | 0.683 | **0.475** | 0.803 | 0.50 | 0.66 | 0.45 |
| 14x | 0.997 | **0.475** | **0.997** | 0.40 | 0.66 | 0.40 |

Corpus base rate of ≥1 out-of-bounds input: 0.40.

Two refinements to their L6 recommendation, both in my favour and both supporting the general
direction they argue for:
1. The ratio gate is **exactly load-invariant** (0.475 at every k). That is the property they
   claim and it is now proved deterministically rather than inferred from one run.
2. But the knob **ORs** the ratio gate onto the cost gate, so the combination still saturates to
   **0.997 at 14x load** — it does *not* remove the L4 pathology, it only masks it while the
   device is quiet. And the ratio gate still selects OOD chunks at **66 %** against a 40 % base
   rate, so it only *reduces* the L2 pathology (98 % → 66 %), it does not remove that either.
   A load-independent trigger requires **replacing** the cost gate, not ORing onto it — plus an
   `n_x_oob == 0` precondition to stop training on extrapolations. Neither of us has measured
   that combination.

### F10 — CONCEDE both

"The input the model uses least in range" is withdrawn: E7 is a single-point marginal sweep with
the other three inputs pinned at the training mean, and those inputs co-vary in real data. And
their M10 is right from my own table — the 0.25 MiB point (median ratio APE **4.12**) is worse
than 1 MiB (1.56) and is inside the training range; I quoted my table from 1 MiB. Both in F10.
The out-of-range half (8 MiB APE 46.2 → their 46.2, my 40.6 on a different action subset;
regret 0 → 6.997 at 16 MiB; ratio → 100.0 and PSNR 34.5 dB for a *lossless* action at 64 MiB)
is unattacked and stands.

### F11 — PARTIAL on the number, CONCEDE M7

**PARTIAL / CONTEST the "does not reproduce" wording.** The figure is reproducible; my write-up
omitted its filter. My script computed the APE over the **6496 of 7040 rows whose analytic PSNR
is positive**, which gives exactly median 22.2 % / p90 100 % / mean 347 %. Over all 7040 rows it
is median 14.0 % / p90 100 % / mean −154 % (signed denominator) or median 25.4 % / p90 118 % /
mean 795 % (|denominator|) — all three now printed in F11. **CONCEDE** that the metric is
ill-posed without the filter and that not stating it was a defect.
**CONCEDE M7 in full** — and it is a better finding than the one it corrects: the psnr head is
clamped to [0, 120] (`neuropress_nn_gpu_kernels.cu:534`) while its analytic label reaches
**−283.6 dB** on 544/7040 rows (34 chunks). On those chunks the head cannot represent its target
at all and every SGD step carries a permanent error. Added to F11.
Their "circular evidence, correct conclusion" reading of the 1-distinct-actual result is fair and
I have said so in F11's own text (the label *is* the formula) — the physical argument
(quantization output feeds a lossless codec, so the reconstruction is identical across all 16
quantized actions) is what carries it.

### F12 — no attack landed; ACCEPT their held-out extension

All eight figures reproduce on their independent recount. Their held-out corpus makes it worse
(recall 0.333, 48.7 % exact ties). Nothing to adjudicate.

### F13 — CONTEST the narrowing

**CONTEST.** Their scope reduction — "the defect is device-path-only, the host path already
honours `data_type_`" — is wrong. `ComputeNeuroPressFeatures` special-cases **only**
`data_type_ == 2` (float64, `data_stats_gpu.h:196-217`) and falls through to a **float32 read for
every other type** (`:219-222`). My E10 numbers (uint8 → mad 3.02e36; int32 → mad 3.32e-41)
were produced through that *host* entry point, not the device one. So a uint8 or int32 payload is
misread on **both** paths. What is genuinely device-path-specific is nothing in F13; what is
missing on both paths is a finiteness guard. F13 stands unnarrowed and C-A/A5 now says "both
paths".
Everything else in their F13 verdict (mechanism, `fmaxf(0,NaN) == 0`, the sanity-clamp comment
reasoning about outputs rather than inputs) is agreement.

### F14 — ACCEPT (they strengthen it) and CONCEDE the scope

They ran the paired replicate I flagged as missing (`f0c`: 0.246/0.7781 → 0.175/0.9578, spread
⅓ of the effect) and conclude my "medium-high" was too modest. Accepted.
**CONCEDE the scope**: on a loaded GPU only ~7 % of measured ct sits below the 1 ms floor instead
of 29.1 %, so "the floor is doing real work" is an idle-device statement. Added to F14 and A6.

### F15, F16, F17 — agreement, with one correction to their F17

F15 and F16 are confirmed with independently re-derived numbers; nothing to adjudicate.
F17: they re-verified five upstream constants by hand and re-ran `np_ood_upstream_exec`
(0/17 mismatches, 0.000e+00) — agreement. **Their ctest was 16/17**; mine was **17/17**, run at
07:5x on an idle GPU. Their single failure is a `cudaMemGetInfo` assertion, which they themselves
identify as invalid on a shared GPU — and the GPU was shared because of my `runs3b.sh`. So the
correct joint statement is **17/17 on an idle device**, with a known shared-GPU false failure.
Their addition (the gate arithmetic matches `gpucompress_compress.cpp:665-679` including the
"use the prediction for act_dt" rule) is a real extension of F17 and I accept it.

---

## Part 2 — the adversary's new material

| item | verdict | response |
|---|---|---|
| **L0** — run length buys almost nothing (drift 0.90 % at 590 steps, less than at 226) | **CONCEDE / ACCEPT** | Reproduced on my own `*.weights.csv`. This is the single most important fact for the deployment mode and I missed it despite having the data. Now F8.2. |
| **L1** — paired per-chunk: 42 better / 83 worse but **−16.4 ms** net | **CONCEDE** | Correct qualification of my F8 headline. In F8.1. |
| **L2** — the gate is an OOD detector (93 % of firings on OOB chunks) | **CONCEDE / ACCEPT** | Reproduced exactly on my `f2` (42/45 = 93 %, base rate 40 %). Best new mechanism in the review. Now F8.3. |
| **L3** — offline simulation: an online per-algo residual table improves *with run length* (0.083 → 0.032) | **ACCEPT as simulation; PARTIAL as evidence** | It is their offline on-policy simulation, not `NPE_BIAS=1`. My *live* `h3` (ct-only, key=action, α=0.3, no SGD) gives whole-run top-1 0.207 vs 0.122 for no learning and crosses over at chunk 91 — same direction — but ran under contention, so I mark it provisional. Their independent finding that adding the **ratio** head makes it worse (0.0576 → 0.0905) is reproduced live by my `h6` (top-1 0.120 ≈ no-learning 0.122). |
| **L4** — the adaptation trajectory is a function of GPU load | **CONCEDE the mechanism, CONTEST the L1/L2 pair as evidence** | Their k-sweep is deterministic and reproduces on my `f0a` (0.239 idle → 0.997 at 14x); that alone establishes it, and it is now F8.4. The 51-vs-356 replicate pair is *not* a controlled experiment — the contention came from my `runs3b.sh`, so it is a confound they discovered, not a variable they varied. The conclusion survives on the k-sweep. |
| **L5 battery, idle group (L0 vs L1)** | **ACCEPT** | Learning at the shipped gate: regret-neutral, +0.7 % bytes, −26 % top-1. Same sign and magnitude as my `f0a` vs `f2`. |
| **L5 battery, busy group (L2/L4/L5/L6)** | **CONTEST as a basis for comparison** | All four ran contended and are compared against an **idle** no-learning baseline (L0). Their own L4 finding says that is exactly the comparison that cannot be made. The within-group ordering is usable; the "every learning configuration is much worse than the idle baseline" conclusion is confounded with device state. |
| **L5's "phase-2 exploration does not rescue it either" (L5 run: top-1 0.102)** | **CONTEST** | L5 is contended and L0 is not. My idle-vs-idle pair says the opposite: on a 3-block 450-chunk stream with fresh weights, `h2` (gate 0.30 + `NPE_EXPLORE_K=7`) crosses over the no-learning baseline at **chunk 5** and **chunk 17** in two of three blocks, with second-half regret 2.07 → 0.81 and 1.22 → 0.42, against `h1` (phase 1 only) which needs **112 chunks** and only in one block. Both runs idle, same stream, same seed. I agree with their caveat that `np_eval`'s phase 2 trains without adopting, so this measures the *training* value of exploration only — the adoption value is F6's table. |
| **L6** — the ratio-MAPE gate is the only learning config that holds top-1 (0.350 vs 0.102-0.177) | **PARTIAL — accept the direction, contest that it is demonstrated net-positive** | See the dedicated section below. |
| **M1** (bandwidth) | **CONCEDE** | Now F2's title. |
| **M2** (clamp's balanced byte cost) | **CONCEDE** | Now in F1 and A3. |
| **M3** (`--best` disables both SGD phases) | **PARTIAL** | The source claim is right and verified; the inference that `f3` is unreachable is wrong (see F8 above). |
| **M4** (exploration default-OFF) | **CONCEDE** | Now F6's title. |
| **M5** (Clio ships F9's fix) | **CONCEDE** | Now F9 and B2, with my own load-invariance measurement added. |
| **M6** (ratio error is per-chunk) | **CONCEDE / ACCEPT** | Reproduced: −79 % per-chunk, −74 % per-regime, +3 % per-algo. Now in F3. |
| **M7** (psnr clamped to [0,120], label reaches −283 dB) | **CONCEDE** | Now in F11. |
| **M8** (collapse is head-asymmetric) | **CONCEDE** | Now F1's heading. |
| **M9** (two top-1 conventions mixed) | **CONCEDE** | True: `np_eval`'s `is_best_*` columns cap the *measured* ratio at 100 (0.407), my `an4.py`/`oracle.py` ground truth does not (0.378). Both reproduce from the same CSV. FINDINGS now states which convention each table uses; no sign changes. |
| **M10** (0.25 MiB is worse than 1 MiB and is in range) | **CONCEDE** | Now in F10. |
| **M11** (per-action is 16 keys at eb=0; DOF inflation) | **PARTIAL** | Key-count label error conceded. The DOF explanation contested — my random-grouping null shows no reduction at 8/16/32 keys (see F3 above). Their out-of-sample result is accepted regardless. |
| **M12** (`LastCodecKernelMs` is device-load exposed; every balanced number is idle-conditioned) | **CONCEDE** | Now a standing caveat in the noise section. |
| **Held-out corpus table** | **ACCEPT** | I did not run those regimes. Nine of ten rows confirm or strengthen my findings; the tenth (F4's cap effect absent) is a genuine corpus-specificity result and is now in F4. |
| **`np_adv.cu` A1b** (single-sample, cos = 1.000000000 within a sign class) | **ACCEPT** | Independent confirmation of E1c at a different operating point. |
| **A1** (batched, cos 0.910) | **PARTIAL** | Mechanism accepted, magnitude contested — their comparison crosses a sign flip; my same-sign batched probe gives cos ≥ 0.9982. |
| **A2** (three regimes: gate / linear / saturated) | **CONCEDE** | Reproduced with my own E1e. F7 rewritten around it. |
| **A3** (interference leaves Spearman 0.894) | **CONCEDE** | Now F8.5. |
| **A4** (phase 1 is closed-loop on its own mistake) | **ACCEPT** | Now F8.6; I have no independent run. |
| **Tail-statistic caveat** (top 10 of 590 chunks carry 52.6 % of regret) | **CONCEDE** | Reproduced exactly (52.6 %). Medians now quoted beside means throughout; every direction survives on the median. |
| **Loaded-GPU probe** (`L0_contended`) | **ACCEPT** | A genuine natural experiment; its three consequences (F2/F3 stronger, F5 weaker, F14's premise weaker) are all now scoped in FINDINGS. |
| **N1** — learn + clamp: firings 51 → 3, OOB chunks 230 → 0, regret 0.0341, top-1 0.396 | **CONCEDE, confirmed harder** | Re-run idle on my stream: firings **74 → 0**, so learn+clamp is *bit-identical* to infer+clamp. Detail in Part 3. |
| **M1** — online per-algo residual table adapts within a run (0.1601 → 0.0215 vs baseline 0.1602 → 0.1158) | **CONCEDE mechanism, PARTIAL on magnitude** | Reproduced live and idle (`h9`, `h3b`): both beat the baseline with 0 SGD calls. My stream's own halves are not equally hard, so my curve is smaller; their schedule is the better evidence. Detail in Part 3. |
| **L6 self-correction** — L6 was fully contended (7.138 ms all-rows) and still held top-1 0.350 | **CONCEDE the correction, CONTEST the conclusion** | I withdraw my "uncontrolled contention" objection. But an idle-GPU run against an idle baseline on a common table (`h10`) puts the ratio gate at regret 1.2525 vs 1.0918 for not learning: best trunk-SGD configuration, still not a net win. Detail in Part 3. |
| **F1/F8/F9 are one defect seen from three sides** | **CONCEDE and adopt** | Their formulation, confirmed by the 74 → 0 firing count. It is now the framing of C-B/B0 in FINDINGS.md. |
| **Their Section C re-ranking** | **MOSTLY ACCEPT** | Adopted with two changes: A3 (clamp) is ranked below the retrain and the cap for the ratio objective but above the ct bias table, because it is the only item that is a pure win on the deterministic metric; and C-B is now a separate ladder rather than a single line, because the constraint makes the in-run loop a first-class deliverable. |

---

## L6 in detail — do I accept the ratio-MAPE gate as *the* in-run adaptation recommendation?

**Yes as the trigger to use; no as a demonstrated net-positive.** Concretely:

**What I accept.** The ratio MAPE is computed from `min(cap, pred_ratio)` vs
`min(cap, act_ratio)`, both bit-deterministic across runs (9440/9440 rows identical between my
`f0a` and `f0c`). I proved deterministically what they inferred from one run: the ratio gate
fires on **0.475 of chunks at every device-load factor from 0.5x to 14x**, while the cost gate
goes 0.239 → 0.997 over the same range. It is the only trigger in the runtime that cannot be
saturated by a busy GPU, and that alone makes it the right thing to gate on. It is in
`FINDINGS.md` as **C-B/B2**.

**What I contest.**
1. *L6 is not shown to beat no-learning.* Its top-1 is **0.350** against the **no-learning**
   baseline L0's **0.359**. The comparison the rebuttal makes ("0.350 vs 0.102-0.177") is against
   three *other learning runs*, all of which are more contended. The honest reading of their own
   table is: the ratio gate is the only learning configuration that does not **cost** top-1 — it
   does not **gain** anything. Its bytes (0.70677) are still 4.2 % worse than not learning
   (0.67837).
2. *L6's contention level is uncontrolled.* They report it themselves: median measured ct on the
   pick 1.214 ms for L6 against 3.54 / 5.18 / 3.58 ms for L2/L4/L5 and 0.252-0.258 ms for
   L0/L1. Their counter-argument (L6 was more contended than L1 and still beat it) compares two
   *learning* runs, which does not establish the effect against not learning.
3. *ORing does not remove the load pathology.* The knob keeps the cost gate and adds the ratio
   gate. My table above: `cost OR ratio` is 0.478 at idle but **0.997 at 14x** — identical to the
   cost gate alone. To get a load-independent trigger you must **replace** the cost gate.
4. *The ratio gate still trains mostly on extrapolations.* 66 % of its firings land on chunks
   with ≥1 out-of-bounds input, against a 40 % base rate. Better than the cost gate's 98 %, but
   it does not solve L2 — it halves it.

**So my recommendation, and it is now C-B/B2 in FINDINGS.md**: replace (not OR) the cost gate
with the ratio-MAPE gate, and add an `n_x_oob == 0` precondition. Neither of us has measured
that combination; it is the top item in the dispute table.

---

## Part 3 — the two addenda (`N1`, `M1`) and the `L6` correction, re-run on an idle GPU

The GPU went idle after the rebuttal was finalised, so rather than argue I re-ran all three
configurations on **my own** 3-block, 450-chunk stream
(`turb_x50 → smooth_a1_n1e-2 → lammps_pos_sorted`, seed 1234, fresh weights per run) and scored
**every run's picks against one common measured table** (`h0`'s), so codec-timing noise cannot
favour any run. Bytes are deterministic given the pick. Script: `scripts/common_gt.py`.

| run | config | SGD calls | top-1 | mean regret | median | bytes | 1st half | 2nd half |
|---|---|---|---|---|---|---|---|---|
| `h0` | infer, no learning | 0 | 0.122 | 1.0918 | 0.8070 | 0.90259 | 1.2912 | 0.8924 |
| `h0b` | **replicate of `h0`** | 0 | 0.122 | 1.0918 | 0.8070 | 0.90259 | 1.2912 | 0.8924 |
| `h1` | learn, cost gate 0.30 (shipped) | 74 | 0.042 | 1.5859 | 1.0476 | 0.96209 | 1.2313 | 1.9406 |
| `h10` | learn, **`NPE_GATE=ratio_or_cost`** | **195** | 0.087 | **1.2525** | 0.8135 | 0.90398 | 1.3062 | 1.1988 |
| `h2` | learn + explore K=7 | 120 | 0.102 | 1.1181 | **0.5663** | 0.98782 | 0.4047 | 1.8315 |
| `h9` | **residual table, per-algo ct+dt** (= their `M1`) | 0 | 0.142 | 1.0146 | 0.8068 | 0.91821 | 1.2491 | 0.7801 |
| `h3b` | residual table, per-action ct | 0 | **0.173** | **0.9570** | 0.8024 | 0.93897 | 1.1629 | 0.7512 |
| `h6` | residual table, ct **+ ratio** | 0 | 0.131 | 1.2348 | 0.8121 | 0.91382 | 1.4568 | 1.0128 |
| `h5b` | **infer + clamp** (no learning) | 0 | 0.102 | **0.8088** | **0.3822** | **0.88697** | 0.5748 | 1.0429 |
| `h8` | **learn + clamp** (= their `N1`) | **0** | 0.102 | **0.8088** | **0.3822** | **0.88697** | 0.5748 | 1.0429 |

### A methodological result that lands first, and it affects both of us

`h0` and `h0b` are the same invocation run twice. Their **picks are identical** — every
prediction is bit-deterministic — yet `np_eval`'s own summary line reported
`top1_bal = 0.122` for one and **`0.000`** for the other, because each run scores itself against
*its own* measured table and the true-best action flips under timing noise. **Any top-1 number
that compares two runs each using its own ground truth is unreliable at the ±0.12 level on this
stream.** That includes the top-1 column of the adversary's entire L/M/N battery (0.359 / 0.266 /
0.350 / 0.396 …) and my own earlier `h0` vs `h3` comparison. Everything in the table above is
scored against one common table, which is the adversary's own L1 method and the correct one.

### `N1` (learn + clamp) — **CONCEDE, and confirmed in the strongest form**

Their prediction was that clamping the inputs would nearly silence the phase-1 gate; they
measured 51 → 3 firings. On my stream it silences it **completely: 74 → 0 SGD calls**, and
`h8` (learn + clamp) is therefore **bit-identical to `h5b`** (infer + clamp) in every column.
That is the cleanest possible confirmation of their L2 mechanism: the shipped gate fires because
the model is extrapolating, so removing the extrapolation removes the trigger.
I also **agree with their own caveat, and would put it more strongly**: `N1` is not a learning
result. Its gain (regret 1.0918 → 0.8088, −26 %; median 0.8070 → 0.3822, −53 %) is entirely the
clamp's — the learner contributed 3 calls in their run and 0 in mine. F1, F8 and F9 being "one
defect seen from three sides" is their formulation and I adopt it; it is now the framing of
C-B in `FINDINGS.md`.
**One difference to record:** the clamp's byte cost is corpus-dependent, not uniform. On `f0a`
it is +1.9 % (their M2, which I conceded); on their 28-regime schedule +1.5 %; on this 3-block
stream it is **−1.7 %** (0.90259 → 0.88697, better). So "C1 buys time-regret with bytes" is true
where the balanced objective is time-dominated and not a general property.

### `M1` (online per-algorithm residual table, no trunk SGD) — **CONCEDE the mechanism, PARTIAL on the size of the adaptation curve**

Reproduced live on my stream as `h9` (their exact configuration: `NPE_BIAS=1 NPE_BIAS_KEY=algo
NPE_BIAS_HEADS=ct,dt NPE_BIAS_ALPHA=0.3`, no SGD) and `h3b` (per-action, ct only). Both beat the
no-learning baseline on mean regret with **zero** SGD calls — 1.0146 and **0.9570** against
1.0918 — and both improve in the second half. So the substantive claim, *learn the magnitude
outside the network and leave the trunk alone*, is confirmed by an independent live run.

**PARTIAL on the curve's size.** Their evidence is 1st half 0.1601 → 2nd half 0.0215 against a
baseline that goes 0.1602 → 0.1158, i.e. a 5.4x gap. On my stream the baseline itself falls
1.2912 → 0.8924 (the third block is simply easier), so the honest measure is the half-ratio
relative to the baseline: `h0` 0.691, `h9` **0.625**, `h3b` **0.646** — an improvement, but a
modest one, not 5.4x. Their 28-regime schedule is better balanced across halves than my
3-block one, so **their number is the better evidence for this specific point** and I accept it;
mine confirms the sign on an independent stream and schedule.
Two things my runs add: (a) the table's byte cost is real and shows up live (0.90259 → 0.91821
per-algo, 0.93897 per-action), consistent with C3's +4.6 %; (b) adding the **ratio** head makes
it worse than not adapting at all (`h6`, 1.2348 vs 1.0918) — a third independent reproduction of
F3/M6.

### `L6` correction (the ratio-MAPE gate was fully contended and still held top-1) — **CONCEDE the correction, CONTEST the conclusion, with a clean run**

I accept their self-correction: the pick-only contention proxy was misleading, and on the
all-rows measure `L6` (7.138 ms) was as busy as `L2` (8.250) and `L5` (9.174). That removes my
objection (ii) in the L6 section above, and I withdraw it.

But the conclusion still does not follow, and I now have the run that tests it directly.
`h10` is `NPE_GATE=ratio_or_cost` on an **idle** GPU against an idle no-learning baseline, scored
on a common table:

* it fires **195 times in 450 chunks** — 2.6x more often than the cost gate's 74 — and
* it is **worse than not learning**: regret **1.2525 vs 1.0918**, top-1 0.087 vs 0.122, bytes
  0.90398 vs 0.90259.

It is, however, clearly the **best of the trunk-SGD configurations** — better than the cost gate
on every column (1.2525 vs 1.5859 regret, 0.90398 vs 0.96209 bytes) — which is exactly what their
L6 shows, and my F9 k-sweep explains why (the ratio trigger is load-invariant at 0.475 for every
device-load factor from 0.5x to 14x, while the cost gate runs 0.239 → 0.997).
So the joint statement I propose: **the ratio-MAPE gate is the right *trigger* and the wrong
*conclusion*.** It is the only load-invariant trigger the runtime has and it is strictly better
than the shipped cost gate, but on an idle GPU with a common ground truth it still leaves a
fresh process worse off than not adapting at all. What beats the baseline in my table is not any
trunk-SGD configuration: it is the clamp (`h5b`/`h8`) and the residual table (`h9`/`h3b`).

**Effect on the C-B ranking** (updated in `FINDINGS.md`): B2 (ratio gate) drops below B3
(residual table), and the clamp is promoted into C-B as well as C-A, because it is the only
change measured to make the in-run loop *quiescent* rather than merely less harmful.

---

## Points still in dispute

| # | point | my position | adversary's position | evidence that settles it |
|---|---|---|---|---|
| D1 | Does the ratio-MAPE gate make a fresh process **better** than not learning, or merely not worse? | **Now measured, idle GPU, common ground truth (`h10` vs `h0`): worse than not learning** — regret 1.2525 vs 1.0918, bytes 0.90398 vs 0.90259 — though clearly the best trunk-SGD configuration (cost gate 1.5859) | "the only learning configuration that does not cost top-1", presented as the in-run recommendation | **Settled on my stream.** Remaining question is whether it holds on their 28-regime schedule; the run is `NPE_SCHEDULE=<theirs> NPE_MODE=learn NPE_GATE=ratio_or_cost` against an **idle** `NPE_MODE=infer` baseline on the same schedule, scored on a common table. |
| D2 | Is a **replaced** (not ORed) ratio gate, plus an `n_x_oob == 0` precondition, better than either shipped gate? | Predicted yes — it is the only combination that is both load-invariant (my k-sweep) and not OOD-selecting (66 % → 0 by construction) | not addressed | Needs a fourth arm in D1's run; `NPE_GATE` has no "ratio only" mode, so it needs a two-line harness knob (gitignored). |
| D3 | Is F7 **refuted** for the batched phase-2 path? | PARTIAL: mechanism conceded, magnitude disputed — my same-sign 7-sample probe gives cos ≥ 0.9982 and identical ‖ΔW‖ to 5 s.f.; their 0.910 crosses a sign flip | REFUTED | Re-run their A1 with all label variants on the **same side** of the model's prediction, and report cos and ‖ΔW‖ separately. `np_sgd_exec` E1d already does this; their `np_adv.cu` A1 needs the label set changed. |
| D4 | Does a random 32-way grouping really remove 21 % of the ct log error? | No: my null gives +2 % at 8, 16 and 32 keys (median offsets, seed 7) | yes, so a third of F3's per-action gain is DOF inflation | Publish the null's code. If it uses *mean* offsets on a heavy-tailed log-error distribution that would explain the difference; the out-of-sample conclusion (per-algo > per-action) is agreed either way. |
| D5 | Is the `f3` (ratio-only cost weights + learning) configuration reachable? | Yes, via `CLIO_NEUROPRESS_COST_W_CT=0` / `_W_DT=0` with best mode off — `:1443-1449` uses the env weights, `:1577` gates only on `neuropress_best_mode_` | No — "`--best` disables both SGD phases", row invalid | Source is decisive and already quoted on both sides; the disagreement is over which configuration `f3` models, not over what `--best` does. Relabelled in FINDINGS. |
| D6 | Is the non-float32 defect device-path-only? | No — `data_stats_gpu.h:219-222` reads every non-float64 type as float32 on the host path too, and my E10 numbers came through the host entry point | device-path-only; the host path honours `data_type_` | Source; run `np_sgd_exec` E10 (host path) and confirm mad = 3.02e36 for uint8. Already done on my side. |
| D7 | Does phase-2 exploration help in-run adaptation? | Yes on the training signal: idle-vs-idle, crossover falls from 112 chunks to 5-17 (`h1` vs `h2`) | "does not rescue it either" (their L5, top-1 0.102) | Their L5 is contended and compared to an idle baseline. Needs L5 re-run idle against an idle no-learning baseline on the same schedule. |
| D8 | Is the live per-algorithm/per-action residual table a real in-run fix? | **Now measured on an idle GPU** (`h9`, `h3b` vs `h0`, common ground truth): yes — regret 1.0146 / 0.9570 vs 1.0918 with **zero** SGD calls, and both improve in the second half. Size of the adaptation curve is smaller than their 5.4x once the stream's own easing is removed (half-ratio 0.625 / 0.646 vs the baseline's 0.691) | their live `M1` shows 0.1601 → 0.0215 against a baseline 0.1602 → 0.1158 | **Agreed in sign; only the magnitude is open.** Settled by running both on a schedule whose halves are equally hard (theirs is; mine is not). |

Everything not in this table is agreed. Of the seventeen findings, **none is withdrawn**;
F7's impact sentence, F10's "least-used input" half and F5's in-distribution row are corrected,
F1/F2/F4/F6/F11/F13/F14 are rescoped, and F8/F9 are rewritten around the adversary's L0/L2/L4
mechanisms, which are better than the one I had.
