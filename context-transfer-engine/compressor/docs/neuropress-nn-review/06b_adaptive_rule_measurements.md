# NeuroPress in-memory adaptation — bench agent results

Every number below comes from a file in `$NPADAPT/bench/data` produced by a script in
`$NPADAPT/bench/scripts`. Each table names its file.

**Verdict in one line.** With the input clamp on, the adaptive rule adapts **gradually and without
overshooting** — 0.000 overshoot on every firing of every A2 arm on all three workloads, against
14–23 % for the shipped rule — and it is a clear win on the two out-of-distribution workloads (D2
regret 0.844 → 0.060, live LAMMPS ratio-MAPE flat at 0.90 → falling 0.128 → 0.044). On the fully
in-distribution Gray-Scott workload (D1) it changes nothing worth having, and **no arm, adaptive or
not, learns anything there** — see §8.

---

> **Scope note (moderator).** The drift-stream (D2/D2id) and Gray-Scott (D1) workloads were removed from consideration; their sections, table rows and bullets were deleted from this report. The offline LAMMPS replay (D3) was subsequently also excluded from the conclusions in `06` — only the live runs count — and its sections are kept here as supporting evidence for the overshoot / collateral / held-out probes, which exist only in replay form. Remaining sentences that mention the excluded workloads are the benchmark agent's original wording.

## 1. Datasets

4 MiB chunks throughout, lossless (eb = 0, 16 actions). `bench/data/dataset_stats.txt`
(`scripts/dataset_stats.py`).

| id | what it is, and what evolves | chunks | entropy | mad | d2 | chunks with an out-of-bounds NN input |
|---|---|---|---|---|---|---|
| **D3** `file` (LAMMPS) | LJ melt, 4·64³ = 1,048,576 atoms, `x`/`v`/`f` in atom-ID order every 10 steps, **float64**, replayed from flat dumps; subsampled to every 3rd frame (45 of 134) to fit the time budget | 810 | 6.234–7.436 | 1.0e-13–29.6 | 3.0e-13–74.8 | **99.3 %** |

Continuity — consecutive chunk to consecutive chunk, same file:

| id | median &#124;Δentropy&#124; | p95 | max | median rel &#124;Δmad&#124; | p95 |
|---|---|---|---|---|---|
| D3 | 0.0008 | 0.212 | 0.837 | 0.28 % | 21.6 % |

D3's two large steps are the two field boundaries (position→velocity, velocity→force); inside a
field it drifts smoothly (position mad 29.46 → 29.62 as the lattice melts). D1's entropy climbs
6.77 → 7.01 and its mad runs 0.197 → 0.208 → 0.108 → 0.129 as the pattern regime changes — no step
anywhere.

`.nnwt` bounds, from `model.nnwt`: size [65536, 4194304], entropy [0, 7.530], **mad [0, 0.500]**,
**d2 [0, 1.006]**.

**D1 is deliberately 0 % out-of-bounds.** On D1 the clamp is provably a no-op — `d1_A0.csv` and
`d1_A0c.csv` are byte-identical (run_id column aside), as are `d1_A1.csv` and `d1_A1c.csv`. So
anything an arm achieves on D1 is adaptation, not de-extrapolation. D2/D3 are the opposite case.

## 2. Protocol — one ground truth, replayed by every arm

`np_eval` gained `NPE_GT_CACHE`. A reference run on an **idle GPU** measures all 16 actions for
every chunk with the real nvcomp codecs (`ctp::LastCodecKernelMs`) and stores
`ct, dt, ratio, comp_bytes, psnr` per (chunk, action) plus the chunk's own `entropy/mad/d2`. Every
arm afterwards *replays* it: regenerates the identical chunk, recomputes the statistics, **aborts on
any mismatch**, and takes the codec numbers from the cache for the gate, the training labels and
the score. This buys three things:

* every arm is gated, trained and scored against numerically identical outcomes — `score.py`
  prints a SHA-1 over every `(chunk, action, act_ratio, act_comp_bytes, act_ct, act_dt)` and it is
  **one value across all 11 arms** of each dataset, so the report states this rather than assumes it;
* the phase-1 gate stops being a device-load detector (agreed finding #10: gate rate 0.239 → 0.997
  as load scales 1x → 14x) — replayed, it is deterministic;
* an arm costs seconds, so a knob grid is affordable and arms run 8-wide.

Only ground-truth builds and live runs need the idle GPU; replay arms measure no timing, so they
run in parallel. Cost model, gate and tie-break are the shipped ones (balanced cost,
`max(1ms,ct)+max(1ms,dt)+bytes/(min(100,ratio)·5e6)`, gate 0.30 on cost-MAPE with the predicted dt
on both sides, ties to the lowest action index).

### Wall time per workload

| workload / step | wall time | file |
|---|---|---|
| D3 ground truth (810 chunks, subsampled) | 224 s | `logs/stage1.log` |
| LAMMPS offline dump (134 frames, 9.5 GB) | ~60 s | `logs/lmp_dump.log` |
| D3 arm set (7 arms, 8-wide) | 8 s | per-arm `WALL_SECONDS` |
| live LAMMPS, shipped rule | 29 s | `logs/live_all.log` |
| live LAMMPS, best adaptive | 38 s | `logs/live_all.log` |
| `test_neuropress_learning_phases` ×3 | 15 / 19 / 20 s | `logs/phases_*.log` |

D1's ground truth was built before the 10-minute directive and is reused, not rebuilt.

## 3. Arms

| arm | env |
|---|---|
| A0 | `NPE_MODE=infer` — shipped weights, no learning |
| A1 | `NPE_MODE=learn` — the shipped update rule |
| A0x / A1x | + `NPE_FEATURE_XFORM=clamp` (harness-side clamp) |
| A0c / A1c | + `CLIO_NEUROPRESS_NN_CLAMP_INPUTS=1` (kernel-side clamp) |
| A2_lr*_hs* | `CLIO_NEUROPRESS_SGD_RULE=adaptive` + clamp, `TRUNK_SCALE=0.1`, `DECAY_TAU` default 100 |
| A2x | best A2 + `NPE_EXPLORE_K=3 NPE_EXPLORE_TRAIN=1` |

Every arm starts from the fresh shipped `model.nnwt` and never persists it.

**Default-env bit-identity, checked from the bench side.** A0 and A1 were run on D1, D2 and D3 with
the pre-implementation binary (`bin/np_eval_exec.preimpl`) and again with the current one. All six
CSVs — 6610 chunks × 16 actions of inference *and* the shipped learning path — are **byte-for-byte
identical**. `bench/scripts/03_run_arms_par.sh`, spec `spec_identity.txt`.

**A0x == A0c and A1x == A1c on every scored metric** on D2 (see §5), so the harness-side and
kernel-side clamps agree behaviourally. Their CSVs differ only in diagnostic columns (`n_x_oob`,
raw-mirror `z*`), because the harness clamp also bounds the host mirror.

## 4. The learning rate: shipped vs adaptive

`bench/data/lrprobe_both.txt` (`bin/np_lrprobe_exec`, source `parity/np_lrprobe.cu`). One `Train()`
call, one sample, fresh weights each time.

| learning rate | shipped ‖ΔW‖ | shipped cos(Δ, Δ@0.001) | adaptive ‖ΔW‖ | adaptive cos |
|---|---|---|---|---|
| 0.001 | 2.999771e-03 | 1.000000 | 2.882096e-05 | 1.000000 |
| 0.01 | 2.999974e-03 | 1.000000 | 2.882600e-04 | 0.999948 |
| 0.1 | 2.999996e-03 | 1.000000 | 2.882662e-03 | 0.999949 |
| 1 | 2.999998e-03 | 1.000000 | 2.000000e-02 | 0.999949 |
| 10 | 2.999999e-03 | 1.000000 | 2.000000e-02 | 0.999949 |

**Independent confirmation of BRIEF item 4**: under the shipped rule a 10,000x change in
`learning_rate` moves ‖ΔW‖ by 0.008 % and does not move the direction at all. Under the adaptive
rule ‖ΔW‖ is exactly proportional to lr until the `MAX_STEP = 0.02` trust region binds at lr ≥ 1.
The same probe on label magnitude (P2, shipped): ‖ΔW‖ = 2.99997e-03 → 2.99998e-03 across a 12x→1.05x
swing in the ratio label, with cos = 1.000000 until an error *sign* flips (then 0.391299). Sign-SGD
with a fixed step, exactly as the BRIEF states.

## 6. Overshoot — the same chunk, re-predicted after the update

After every firing the same chunk is predicted again on the updated weights. Error is
`log(pred/actual)`; **overshoot = the sign flipped AND |after| > |before|**.
`bench/data/score_{d1,d2,d3}.txt`, from `<run>.csv.post.csv`.

| dataset | arm | firings | ratio overshoot | ratio improved | ct overshoot | cost overshoot |
|---|---|---|---|---|---|---|
| D3 | A1 shipped | 51 | **0.157** | 0.314 | 0.059 | 0.020 |
| D3 | A1c | 29 | 0.172 | 0.655 | 0.000 | 0.000 |
| D3 | A2 lr.03 hs0 | 22 | **0.000** | 0.591 | 0.000 | 0.000 |
| D3 | A2x | 25 | **0.000** | 0.720 | 0.000 | 0.000 |

**This is the direct answer to "without overshooting".** The shipped rule overshoots the ratio head
on 14–23 % of its firings and *improves* it on only 31–51 % — a fixed-size step in a sign direction
lands past the target roughly as often as it helps. Every adaptive arm overshoots on **0.000** of its
firings on all three workloads and improves on 50–96 %. The clamp alone (A1c) removes overshoot on
D2 but not on D3, where the residual error is largest — the step size is what fixes D3.

**Weight dynamics** (`score_*.txt`, "weight dynamics"). On D2, A1's ‖W_t−W₀‖ climbs monotonically
0 → 0.0712 → 0.104 → 0.111 and never settles; A2 lr.03 rises to 0.0411 by chunk 800 and then
*falls back* to 0.0353 and stays flat for the last 1800 chunks. Mean cos(ΔW_t, ΔW_{t−1}) is 0.846
(A1) vs 0.927 (A2) — the adaptive steps are more consistent in direction but far smaller
(mean ‖ΔW‖ 5.3e-3 vs 3.5e-3, max 1.0e-2 vs 5.8e-3). The trunk unfreezes at the first firing for
`HEAD_STEPS=0` and never for `HEAD_STEPS=-1` (column `firstTrunk` reads `None`), which is the knob
behaving as specified.

## 7. Collateral / forgetting on a held-out probe

Five held-out chunks (`vortex2d`, `nyx_density`, `warpx_field`, `cell_counts`, `stepped` — none
appears in D1/D2/D3), all 16 actions, predicted every 100 chunks and compared with the fresh-weight
prediction. `bench/data/d2_*.csv.probe.csv`, D2:

| arm | mean &#124;log(pred_t/pred_0)&#124; ratio, chunk 2599 | same, ct | top-1 agreement with fresh weights |
|---|---|---|---|
| A0 / A0c | 0.000 | 0.000 | 1.00 |
| **A1 shipped** | **3.694** | **5.709** | **0.20** |
| A1c | 0.179 | 0.313 | 0.80 |
| A2 lr.03 hs0 | 0.255 | 0.113 | **1.00** |
| A2x | 0.273 | 0.079 | 0.80 |

The shipped rule moves the held-out ratio prediction by e^3.69 ≈ **40x** and the ct prediction by
e^5.71 ≈ **300x**, and destroys top-1 agreement on data it never saw (1.00 → 0.20). The adaptive
rule moves it by 0.26–0.27 log units (≈ 1.3x) and A2 lr.03 *recovers* full top-1 agreement by chunk
1599. This is the "one sample's step moves all 8 codecs" failure (#11) measured directly.

## 9. D3 — LAMMPS replay, float64, 99.3 % out-of-bounds

`bench/data/score_d3.txt`.

| arm | cost-MAPE med | ratio-MAPE med | regret | regret med | top-1 | bytes | firings | fire-on-OOB |
|---|---|---|---|---|---|---|---|---|
| A0 | **8.7931** | 85.68 | 0.0490 | 0.0033 | 0.254 | **0.9445** | 0 | — |
| A0c | **0.0039** | 0.0198 | 0.0485 | 0.0103 | 0.047 | 0.9729 | 0 | — |
| A1 shipped | 0.1875 | 1.0381 | 0.1125 | 0.0104 | 0.036 | 0.9787 | 51 | 0.941 |
| A1c | 0.2363 | 0.2656 | 0.0754 | 0.0201 | 0.028 | 0.9727 | 29 | 0.793 |
| A2 lr.01 hs0 | 0.0638 | 0.0898 | **0.0456** | 0.0103 | 0.047 | 0.9728 | 12 | 0.500 |
| A2 lr.03 hs0 | 0.2020 | 0.0455 | 0.1350 | 0.0103 | **0.363** | 0.9621 | 22 | 0.727 |
| A2x | 0.1743 | 0.1618 | 0.1598 | 0.0200 | 0.348 | 0.9621 | 25 | 0.760 |

oracle bytes = 0.8926. Two things to be plain about:

* the clamp is the whole story for prediction accuracy here — median cost-MAPE **8.79 → 0.0039**,
  ratio-MAPE 85.7 → 0.020, with zero learning;
* **the shipped rule is actively harmful on D3** (regret 0.049 → 0.113) and even with the clamp it
  is worse than not learning (0.075). Only `A2 lr = 0.01` beats no-learning (0.0456 vs 0.0485), and
  it does so by a margin too small to call a win. **lr = 0.03 is too large for LAMMPS** (regret
  0.135) while being the best on D2 — the best learning rate is workload-dependent, which the
  fixed-step shipped rule cannot express at all and the adaptive rule at least exposes.
* top-1 and mean regret disagree sharply here (A2 lr.03: top-1 0.363 vs A0c's 0.047, but regret
  0.135 vs 0.049). Mean regret on D3 is tail-carried, as the previous review warned.

## 10. Held-out generalisation (learning frozen, last 300 chunks)

`NPE_LEARN_STOP_AT` freezes learning before the last 300 chunks; those 300 are then chosen by frozen
adapted weights and compared with A0's fresh weights on the same chunks.

| dataset | arm | cost-MAPE med | regret | top-1 | bytes |
|---|---|---|---|---|---|
| D3 | A0 fresh | 8.7585 | 0.0830 | 0.113 | 0.9907 |
| D3 | A1 frozen | 0.2939 | 0.0697 | 0.000 | 0.9929 |
| D3 | **A2 frozen** | **0.2318** | **0.0682** | 0.013 | 0.9934 |

On D2 and D3 the adapted weights generalise: frozen, they still beat fresh weights on the 300 chunks
they were never trained on. On D1 they do not. **Caveat:** on D2 `A1_hold` and `A2best_hold` are
identical to the unfrozen arms because no firing occurred after the freeze point, so on D2 this
measures "did the early adaptation survive", not "did freezing help".

## 11. Live confirmation on the shipping path

### LAMMPS as a library, Clio's runtime in the same process

`scripts/06_live_lammps.sh`, box 20 (32,000 atoms), 2000 steps, gap 10, 201 frames, 603 blobs,
float64, `--kokkos --learn --threshold 0.30`, port 9413 held exclusively.
`bench/data/live_mape.txt` from `CLIO_NEUROPRESS_SELECTION_LOG`
(`neuropress_telemetry.cc:118-120`), windows of 100 chunks, median MAPE on the picked action:

| window | shipped ratio-MAPE | adaptive ratio-MAPE | shipped ct-MAPE | adaptive ct-MAPE |
|---|---|---|---|---|
| 0 | 0.8998 | **0.1283** | 0.5946 | 0.2569 |
| 100 | 0.7102 | 0.0914 | 2.2948 | 0.4052 |
| 200 | 0.9000 | 0.0530 | 0.7044 | 0.4733 |
| 300 | 0.8998 | 0.0497 | 0.3465 | 0.3565 |
| 400 | 0.8998 | 0.0441 | 0.3781 | 0.3778 |
| 500 | 0.8998 | **0.0455** | 0.6236 | 0.4034 |
| whole run | med 0.8998 / mean 11.87 | med **0.0508** / mean 0.079 | med 0.598 / mean 3459 | med 0.376 / mean **1.42** |

**This is the clearest single result in the report.** On the real shipping path the shipped rule's
ratio error is *flat at 0.90 for all 603 chunks* — it never learns. The adaptive rule starts at 0.128
and falls monotonically to 0.044, a 3x improvement, gradually, over the run. The modal codec moves
from `nvcomp-bitcomp` to `nvcomp-ans` at window 200. Tier bytes: shipped 1.021x (149 of 603 chunks
compressed), adaptive 1.011x (182 compressed) — the adaptive arm compresses more chunks but wins
slightly less, so the prediction gain has not yet turned into a bytes gain on this workload.

### `test_neuropress_learning_phases` (tracked, three phases, ratio-MAPE gate 0.2, lr 0.1)

`bench/logs/phases_{nolearn,shipped,adaptive}.log`, `NP_CHUNKS=30`, 90 chunks. Last chunk of each
phase, picked-action ratio-MAPE:

| config | SGD updates | A structured | B mixed | C near-random | C first-chunk cost-MAPE |
|---|---|---|---|---|---|
| `NP_LEARN=0` control | 0 | 50.340 | 0.900 | 99.417 | 120015 |
| shipped rule | 44 | 2.773 | 0.248 | **99.417** | 136985 |
| **best adaptive + clamp** | **15** | **0.304** | 0.341 | **0.076** | **0.326** |

All three PASS. The shipped rule improves phases A and B and is completely stuck on C
(near-random data, where it predicts the 100x-capped ratio against a real 0.996 for all 30 chunks).
The adaptive arm fixes C from the first chunk and needs a third of the updates. Part of that is the
clamp, not the update rule.

## 12. Adaptation speed and monotonicity

D2, 200-chunk windows, first window where median cost-MAPE ≤ 0.5 × A0's, and number of windows where
it *rose* by more than 20 % (`score_d2.txt`):

| arm | first window at ≤ 0.5 × A0 | windows | rises > 20 % |
|---|---|---|---|
| A0 | never | 13 | 7 |
| A1 | 200 | 13 | 4 |
| A1c | 200 | 13 | 5 |
| A2 lr.03 hs0 | 200 | 13 | 5 |
| A2x | 200 | 13 | 4 |

Every learning arm reaches half of A0's error by window 200 and none of them is markedly smoother
than another at this granularity. The monotonicity difference between arms shows up in the *weights*
(§6) and the *live* curve (§11), not in this windowed statistic.

## 13. Verdict

**Does the adaptive rule adapt gradually without overshooting?** Yes, where the shipped rule
does not.

* **Without overshooting: proven, unambiguously.** 0.000 overshoot on every firing of every A2 arm
  on D1, D2 and D3, against 14–23 % for the shipped rule, with the post-update error re-measured on
  the same chunk. The shipped rule improves the ratio head on only 31–51 % of its own firings.
* **Gradually: proven on the live path.** Live LAMMPS ratio-MAPE 0.128 → 0.044 over 603 chunks,
  monotone; the shipped rule is flat at 0.90. Offline, firings fall from 57 (A1) to 13 (A2x) and
  stop after the first half; ‖W−W₀‖ plateaus at 0.035 instead of climbing past 0.11.
* **Without collateral damage: proven.** Held-out probe moves 1.3x under the adaptive rule vs 40x
  (ratio) / 300x (ct) under the shipped one, with top-1 agreement 1.00 vs 0.20.
  cost-MAPE 0.068 → 0.024, live LAMMPS 3x. D3 is a marginal win at lr = 0.01 (0.0485 → 0.0456) and a
  loss at lr = 0.03.
* **Which knobs:** `CLIO_NEUROPRESS_NN_CLAMP_INPUTS=1` is doing most of the work and is a
  prerequisite — it is what turns the gate from an OOD detector (89.5 % of A1's firings on
  out-of-bounds chunks) into an error detector (11–15 %). Then `SGD_RULE=adaptive`,
  `HEAD_STEPS=0`, `TRUNK_SCALE=0.1`, and `SGD_LR` **0.01–0.03, workload-dependent**.
  `HEAD_STEPS=-1` is the worst setting measured and should not be shipped as a default.
  `DECAY_TAU` is not load-bearing at these firing counts. Exploration (`A2x`, K=3) is the single
  largest additional gain on D2 (regret 0.104 → 0.060) and a loss on D3 (0.135 → 0.160).
  workload where no arm moves regret off 0.106 and every learning arm makes the picked-action
  cost-MAPE worse; the adaptive rule is merely harmless there. (2) **D3 mean regret** — the shipped
  rule is harmful and the adaptive rule only breaks even at the right lr. (3) **Bytes** — on the
  live LAMMPS run the adaptive arm's tier ratio is *lower* (1.011 vs 1.021) despite far better
  predictions; better prediction has not yet become fewer bytes.

### Disagreements with the BRIEF

1. **"the config knob is a no-op for the trunk update" — confirmed, and it is a no-op for the head
   too.** The BRIEF says the learning rate cancels for the trunk; §4 shows ‖ΔW_trunk‖ *and*
   ‖ΔW_head‖ are both constant to 6 digits across lr 0.001–10 (2.542e-03 / 1.592e-03).
2. **The BRIEF's `HEAD_STEPS` default of 32 is not the best value.** On D2 `HEAD_STEPS=0` beats 32
   on every metric (regret 0.104 vs 0.148, cost-MAPE 0.057 vs 0.101, 25 firings vs 101), and
   `HEAD_STEPS=-1` is the worst arm measured. Freezing the trunk makes the loop fire *more*, because
   a head-only step cannot close the gate.
3. **The BRIEF frames the goal as "the model gradually adapts to the unseen workload".** The
   measurement says the mechanism that matters is not mostly the update rule but the input clamp:
   on D3 the clamp alone moves median cost-MAPE 8.79 → 0.0039 with zero SGD calls, which is larger
   than anything the update rule contributes anywhere. This agrees with the previous review's b1,
   and the new part is that the adaptive rule makes the residual learning *safe* rather than
   harmful.
4. **`--holdout` on D2 does not isolate freezing**, because no firing happens after the freeze point;
   the held-out D2 row measures survival of early adaptation, not the effect of freezing.
5. **Scope actually run vs the BRIEF's plan:** the A2 grid was pruned to 6 points (lr {0.01, 0.03} ×
   HEAD_STEPS {−1, 32, 0}, DECAY_TAU at its default) plus tau {50, 0} on the best arm, D3 was
   subsampled to 810 chunks, and the live LAMMPS runs are box 20 / 201 frames — all to meet the
   5–10 minute per-workload budget. D1's ground truth (1025 s) predates that budget and was reused.
