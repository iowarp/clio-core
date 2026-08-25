# NeuroPress NN predictor in Clio — investigator findings

Branch `neuropress-693-continued` @ 81ac5a43. GPU A100-PCIE-40GB, CUDA 12.6, nvcomp.
All numbers below are from runs made in **this** session unless a line says "reused".
Nothing tracked was modified or committed; every new file lives in the gitignored
`context-transport-primitives/test/unit/compress/model/parity/` dir or under this scratchpad.

**Status: revised after debate round 1** against
`../adversary/REBUTTAL.md`. Conceded corrections are folded in below and marked
"round 1"; what I still contest is listed in `DEBATE_R1.md`. Sections B and C are the single
source of truth.

**Noise handling.** Two runs with the *same* data seed (`f0a`, `f0c`) give bit-identical
features, ratios, compressed bytes and predictions (9440/9440 rows, max rel diff 0.0e+00);
only the codec kernel times move — median 1.3 %, p90 14 %, p99 81 %, max 97x. That is enough
to flip which action is genuinely cheapest on 3.2 % of chunks under the balanced cost
(22.2 % if the 1 ms floor is removed) and 0.0 % under the ratio-only cost. So: every
*ratio/bytes* number here is deterministic; every *balanced-cost* number carries ~±0.003 mean
regret (measured spread over three runs: 0.1572 / 0.1609 / 0.1629). I therefore lead with
bytes and ratio-only regret wherever a claim has to survive re-measurement, and I never call a
single balanced-cost run decisive.

---

## Section A — How it works

> **Deployment mode: in-memory adaptation from fresh model weights.** Every process loads
> the shipped `model.nnwt` (`compressor_runtime.cc:376`), adapts its *live* weights through
> phase-1/phase-2 SGD for the lifetime of that process, and never persists them
> (`compressor_runtime.cc:1465-1471`; `Save()` is never called). Nothing carries over between
> runs, so the shipped weights are the starting point of every run and the adaptation has
> only the chunks of that one run to work with. Every claim below is written against that
> mode: "turn learning off" is not an available answer.

**Features (3 data + 5 configuration).** `Runtime::NeuroPressRankChunk` reads the *whole*
chunk as **float32 unconditionally** (`neuropress_selection.cc:48,53`); a `data_type_==2`
chunk is *converted* (not reinterpreted) to float32 first (`:68-74`,
`data_stats_gpu_kernels.cu:241`). Three statistics are computed on-device in two passes:
a 256-bin **byte** histogram → Shannon entropy in bits (`data_stats_gpu_kernels.cu:32,66-89`
→ `:198-218`), the mean |second derivative| `|x[i+1]-2x[i]+x[i-1]|` over i∈[1,n-2]
(`:98-106`, normalised at `:258`), and MAD = mean |x−mean| in a second pass (`:135`/`:163`,
normalised at `:255`). All three accumulate in `double`, land in a 24-byte
`DeviceFeatureStats`, and are read by the inference kernel **on the device** — the host never
sees them before the ranking (`neuropress_nn_predictor.cc:468-471`).

**The 8 inputs** are `[algo_id, quant, shuffle, error_bound, data_size, entropy, mad, d2]`
(`neuropress_nn_predictor.cc:283-313`). On the device path the first three are *decoded from
the action index* in-kernel (`neuropress_nn_gpu_kernels.cu:457-462, 677-714`), `data_size` and
`error_bound` ride in as scalars, and lossless configs get `error_bound = 1e-7` (upstream's
training sentinel, `:690-705`); both SGD paths feed the raw bound instead
(`neuropress_nn_predictor.cc:641, 728`). `algo_id` is a **scalar** 0-7 mapped from the nvcomp
base id by an explicit table (`:263-281`).

**Standardization** is `(x − x_means[i]) / max(x_stds[i], 1e-8)` per input, from the `.nnwt`
header (`neuropress_nn_gpu_kernels.cu:715-717`). Shipped values (measured, `model.nnwt`
= `/home/cc/NeuroPress/neural_net/weights/model.nnwt`, byte-identical):
means `[3.5, 0, 0, 0.02775, 1.4226e6, 4.5139, 0.18917, 0.23892]`, stds
`[2.2913, 1, 1, 0.041895, 1.6949e6, 2.1876, 0.106154, 0.211254]`. `quant`/`shuffle` are
deliberately left unstandardized (mean 0, std 1) — upstream standardizes only
`CONTINUOUS_FEATURES` (`neural_net/core/data.py:41,151-159`), and the port matches.
Training bounds also in the file: size ∈ [64 KiB, 4 MiB], entropy ≤ 7.53, **mad ≤ 0.49998**,
d2 ≤ 1.0057, eb ≤ 0.1. **Nothing consults those bounds at inference** on either side.

**Network:** 8 → 64 → 64 → 64 → 64 → 8, ReLU, one CUDA block per candidate, one thread per
hidden unit (`neuropress_nn_gpu_kernels.cu:476-518`).

**Heads and inverse transforms** (`:520-539`): outputs 0-2 are `expm1(y·y_std + y_mean)`,
output 3 is affine. Sanity clamps first — ct/dt to [1e-6, 1e6], ratio to [0.1, 1e5], psnr to
[0,120] — then the *policy* clamps: `ct = max(1, ·)`, `dt = max(1, ·)`, `ratio = min(cap, ·)`
with `cap` = upstream's 100. Outputs 4-7 (rmse/max_err/mae/ssim) are computed only when asked
(`:552-568`); selection never reads them. Post-clamps repeat on the host at
`neuropress_nn_predictor.cc:590-602`. Shipped y-normalization implies a training mean of
2.85 ms compress, 1.39 ms decompress, 7.82x ratio, 65.6 dB.

**32 candidates.** The bridge enumerates `algo + 8·quant + 16·shuffle` in exactly that loop
order, after filtering to the 8 trained nvcomp base ids and to what the build can construct
(`neuropress_bridge.cc:117-195`); slot order therefore equals action order. The quantize half
is enumerated even on a lossless write because `RankKernel` masks it
(`:182-194`, `neuropress_nn_gpu_kernels.cu:799`).

**Scoring** happens entirely in `RankKernel` (`:754-845`): one warp, lane = candidate,
`score = −(w_ct·max(1,ct) + w_dt·max(1,dt) + w_io·bytes/(min(cap,ratio)·bw))`, defaults
`w = (1,1,1)`, `bw = 5e6 B/ms`, `cap = 100` (`neuropress_bridge.cc:66-78`; identical to
upstream's `g_rank_w0/w1/w2`, `g_measured_bw_bytes_per_ms`, `gpucompress_api.cpp:67-71`).
Quantize candidates and sub-`min_psnr` candidates are set to −∞. A 32-lane bitonic sort with
composite key `(score desc, action asc, slot asc)` returns the order.

**Runtime loop** (`compressor_runtime.cc`). After the primary is compressed:
`error_pct = |actual_cost − predicted_cost| / actual_cost`, where **the decompress term uses
the *prediction* on both sides** (`:1447-1462`) so it contributes exactly zero. Phase 1 fires
`TrainDeviceStats` on the single primary sample when `error_pct > 0.30`
(`:1557-1579, 1614-1626`), with `decompress_time = 0.0f` (`:1619`) and `psnr = 120.0` for a
lossless primary (`:1611-1613`). Phase 2 fires when `error_pct > 0.50`, walks the top-K
*predicted* window (default K=3; `--best` pins K=31), measures each alternative on its own
stream, **adopts** any whose measured cost beats the primary's (`:1989-2028, 2158-2284`), and
trains on the 7 cheapest explored samples (`:2290-2324`). The decompression head is trained
only on reads, batched, with the measured time **floored at 1 ms** first (`:935`), through a
head-only kernel that writes just `w5[1][*]` and `b5[1]` (`neuropress_nn_gpu_kernels.cu:860-992`).
Phase-1/2 SGD deliberately skips those two slots (`:1876-1889`).

**What SGD updates:** all 13576 parameters except `w5` row 1 / `b5[1]`. Per-output errors in
normalized log space, ct noise-gated at 0.10 (`:1591-1592`), clamped to ±0.5 (`:1596-1598`),
Kendall uncertainty weighting on `log_var` (`:1604-1623`), PCGrad-lite on unit-normalised L4
deltas (`:1626-1798`), a per-output gradient clip at norm 0.1 (`:1774`), global unit
normalisation (`:1818-1820`), trust step `clamp(0.08·mean|err|, 1e-4, 0.02)` (`:1836`),
anti-flip damping ×0.5 (`:1860`), EMA 0.85 (`:1867-1869`), weight clamp ±10 (`:1888`).

---

## Section B — Claims

Two conventions are in play and round 1 asked for them to be stated. `np_eval`'s own
`top1_bal`/`top1_ratio` summary lines (0.407 / 0.383 on `f0a`) come from the harness's
`is_best_*` columns, whose ground truth caps the **measured** ratio at 100; my `an4.py`,
`oracle.py` and `debate*.py` tables (0.378 / 0.298) use an uncapped measured ratio. Both
reproduce from the same CSV and no sign changes between them; each table below says which it is
by which script produced it.

Impact column convention: "balanced" = `w=(1,1,1)`, ground-truth cost from *measured*
ct/dt/ratio; "ratio-only" = `w_ct=w_dt=0` (Clio's `--best`). "bytes" = stored compressed
bytes / raw bytes over the whole corpus. Corpus = 28 synthetic regimes × 20 chunks
(590 chunks × 16 lossless actions) and 28 × 15 at eb=1e-3 (440 chunks × 32 actions), 4 MiB
float32, every action measured with the real nvcomp codec.

---

### F1 — Absolute-magnitude features put ordinary scientific data hundreds of σ out of distribution; one or both heads then saturate, monotonically in amplitude. **Largest single effect on this corpus.**

**Mechanism.** `mad` and `second_derivative` are absolute magnitudes
(`data_stats_gpu_kernels.cu:255-260`), standardized against a training set whose values lived
in [0,1] (`mad` max 0.49998 in `model.nnwt`). Nothing clamps them
(`neuropress_nn_gpu_kernels.cu:715-717` divides and feeds the result straight in).

**Evidence (fresh).**
`build/bin/np_sgd_exec` **E4** — a pure feature-space sweep, no data, shape fixed, only
`mad`/`d2` scaled:

| mad | d2 | predicted ratio spread over 8 algos | predicted ct spread |
|---|---|---|---|
| 0.00574 | 0.0116 | 46.4x | 3.5x |
| 0.0574 | 0.116 | 10.6x | 10.3x |
| 0.574 | 1.16 | 5.7x | 13.8x |
| 5.74 | 11.6 | 6.3x | 35.9x |
| 57.4 | 116 | **1.00x** (all 8 = 100.0) | 4287x (max = the 1e6 sanity ceiling) |

On real data (`f0a_infer_eb0.csv`), the same collapse: `smooth_a1000` (mad 314), `turb_x50`
(mad 40.0) and `stepped` (mad 64) each produce **1 distinct ct and 1 distinct ratio across all
16 actions**; their per-chunk mean regret is 0.826 / 0.825 / 0.004. `gs_x1e4` (mad 57.4) has
28 % of its ct predictions pinned at the 1e6 ceiling. At fixed *shape*, gs_spots → gs_x100 →
gs_x1e4 moves median ct APE 0.66 → 0.54 → **516.6** and layer-1 activation L2 4.86 → 6.45 → 366.

**Impact on selection.** Clamping the four raw inputs to the `x_mins/x_maxs` already in
`model.nnwt` (harness knob `NPE_FEATURE_XFORM=clamp`, run `g1`) gives:
balanced mean regret **0.1572 → 0.0395 (−75 %)**, top-1 0.407 → 0.486;
ratio-only bytes **0.6710 → 0.6033 (−10.1 %)**;
ratio-only at eb=1e-3 (run `g3`) bytes **0.4314 → 0.2817 (−34.7 %)**, median regret 0.198 → 0.084.
Predictions are **bit-identical on all 5120 in-distribution rows** and change 87 % of OOD rows.

**Revisions from debate round 1 (conceded).**
1. *The collapse is head-asymmetric.* "One prediction for all 32 actions" holds only above
   mad ≈ 40. At mad ≈ 20 (`lammps_pos_*`, `lammps_force`) the **ratio** head is already a single
   value across all 16 actions while the **ct** head still resolves 15-16 distinct values (median
   ct APE 750-880). Corrected in the heading above.
2. *The clamp is not free under the balanced objective.* Stored bytes are deterministic and get
   **worse**: 0.67575 → **0.68848 (+1.9 %)** on `f0a`, concentrated where the regret gain is
   (`stepped` 0.0078 → 0.0593, `sizes` 0.6379 → 0.8350). That is the balanced cost buying time
   with bytes, but C1 was written as an unambiguous win and is not one there.
3. *The −75 % is corpus-weighted.* Four regimes (`sizes`, `smooth_a1000`, `turb_x50`, `gs_x1e4`
   = 90/590 chunks) carry 86 % of the whole-corpus balanced regret. Restricted to chunks inside
   **all four** `.nnwt` bounds (n=356) the predictions are provably identical (5696/5696 rows
   bit-equal), so any difference there is timing noise (±0.006 band). Honest statement:
   *the clamp's balanced-regret benefit is entirely the out-of-bounds chunks, and its size scales
   with how much of the corpus is out of bounds — 46 % of rows here.*
4. *Median beside the mean:* balanced median regret 0.0097 → 0.0006 (−94 %).
   The **ratio-only** half is deterministic and survives every subset: bytes −10.1 % on `f0a`,
   −34.7 % at eb=1e-3, and −11.5 % on the adversary's held-out corpus.

**Confidence: high.** Falsified by showing a corpus where the clamp *loses* ratio-only
bytes, or by a feature-space sweep where the prediction spread does not saturate above mad≈5.

**Class: upstream design limitation, faithfully ported.** `build/bin/np_ood_upstream_exec`
runs upstream's own `runNNInference` (compiled from `/home/cc/NeuroPress/src/nn/nn_gpu.cu`)
side by side with Clio over 17 sweep points: **0/17 winner mismatches, worst relative
difference on the winner's ratio/ct = 0.000e+00**, live-order agreement 16/16 (lossless) and
32/32 (lossy). Upstream itself predicts ratio = 100.0 and ct = 214 ms at mad = 57.4.

---

### F2 — **At 4 MiB and 5 GB/s**, the compress-time head owns the balanced regret; the ratio head owns the ratio-only regret; the decompress head owns nothing. **The ordering inverts below ~500 MB/s.**

**Mechanism.** All three enter the same linear cost (`neuropress_cost.h:41-50`,
`neuropress_nn_gpu_kernels.cu:785-791`), but at 4 MiB and 5 GB/s the io term is small, so `ct`
dominates the balanced ranking. Corrected io-share figures (the first version mixed two
denominators): of the **predicted** balanced cost, mean 12.4 % / median 3.3 % / p90 38.3 %;
of the **measured** balanced cost, mean 12.4 % / median 6.5 % / p90 27.7 %.

**Evidence (fresh, `f0a`, 590 chunks, one head at a time replaced by its measured value,
ground-truth cost floored at 1 ms so the comparison is self-consistent):**

| variant | balanced top-1 | balanced mean regret | ratio-only top-1 | ratio-only mean regret |
|---|---|---|---|---|
| predictions | 0.378 | 0.1572 | 0.298 | 0.5545 |
| true ratio | 0.327 | **0.6340 (worse)** | 0.915 | 0.4169 |
| true ct | 0.324 | **0.0169 (−89 %)** | 0.298 | 0.5545 |
| true dt | 0.383 | 0.1215 (−23 %) | 0.298 | 0.5545 |
| true ct + ratio | 0.776 | 0.0019 | 0.915 | 0.4169 |

The "true ratio alone makes it worse" result reproduces the prior session's and is *not* an
artefact: the ratio head over-predicts by a factor e^1.238 = 3.45 in the median, which shrinks
the io term toward zero and lets the (wrong but rank-correlated) time heads drive the choice;
restoring the true ratio re-inflates io for incompressible chunks and lets the ct error
express itself. Worst regimes for the `true ratio` variant: `turb_x50` 23.6, `smooth_a1000` 15.3.

**Impact.** Any effort spent on the ratio head alone *increases* balanced regret. Effort on
the ct head is worth −89 %.

**Replicated on all three runs** (`f0a` / `f0b`, independent data / `f0c`, timing replicate):
predictions 0.1572 / 0.1628 / 0.1608; true ct 0.0169 / 0.0159 / 0.0169; true ratio
0.6340 / 0.6421 / 0.6494; true ct+ratio 0.0019 / 0.0004 / 0.0006.

**The falsification fires — conceded, and re-derived here.**
`g_measured_bw_bytes_per_ms = 5e6` is a single upstream global
(`/home/cc/NeuroPress/src/api/gpucompress_api.cpp:71`) with an env override
(`CLIO_NEUROPRESS_COST_BW`). Re-scoring `f0a` at other bandwidths (`scripts/debate1.py`):

| bandwidth | predictions | true ratio | true ct |
|---|---|---|---|
| 5 GB/s (shipped) | 0.1572 | 0.6340 | **0.0169** |
| 500 MB/s | 0.3376 | **0.2314** | 0.1635 |
| 50 MB/s | 0.2297 | **0.1164** | 0.2633 (*worse than doing nothing*) |
| 5 MB/s | 0.2151 | **0.1181** | 0.4279 (*much worse*) |

At and below ~500 MB/s a perfect ratio head helps and a perfect ct head **hurts**. So F2 —
and everything downstream of it, C3 especially — is a statement about fast local storage.
Which head is worth fixing is a one-line deployment question, and it is now in the title.

**Confidence: high** (three replicate runs, ±0.003 on the mean) **at the stated operating
point**; the operating point is load-bearing.

**Class: design limitation** (label mismatch, see F3), not a port bug.

---

### F3 — The ct head's dynamic range is ~21x too narrow and its per-algorithm error is a constant; the cause is upstream's training label.

**Mechanism.** Upstream's training label `compression_time_ms` is a **host wall-clock bracket
around the whole `gpucompress_compress_gpu()` API call plus `cudaDeviceSynchronize()`**
(`/home/cc/NeuroPress/scripts/benchmark.cu:585-592`) on upstream's hardware. Clio's
`actual_compress_time_ms_` is the codec kernel time.

**Evidence (fresh, `f0a`).** Within one chunk, across the 16 actions:
measured ct max/min **median 96.8x** (p90 135x); predicted ct max/min **median 4.5x**.
Within-chunk Spearman(pred, measured) = **+0.746 median** — the *order* is largely right, the
*spread* is not. Removing one constant per algorithm in log space cuts the median |log error|:
ct 1.209 → 0.499 (**−59 %**), dt 1.054 → 0.431 (−59 %); per-*action* (32 keys) gets ct to 0.441
(−63 %). The same treatment does **nothing** for ratio (1.060 → 1.094, −3 %).
Per-algorithm median log(pred/actual): bitcomp **+2.10** (measured 0.18 ms), ans **+1.50**
(0.24 ms), zstd −0.67 (7.4 ms), deflate −0.62 (8.4 ms) — i.e. the model believes the two
fastest codecs on an A100 are the slowest.
29.1 % of measured compress times are below the model's 1 ms output floor.

**Impact, measured on the decision and out of sample** (added in round 1; the first version
only reported the error reduction). Fit the per-key log bias on `f0a`, apply to `f0b`, balanced:

| correction | regret | median regret | top-1 | bytes |
|---|---|---|---|---|
| none | 0.1629 | 0.0031 | 0.375 | 0.67487 |
| ct only, per **algorithm** | **0.0330 (−80 %)** | 0.0073 (*worse*) | 0.317 | 0.70576 (+4.6 %) |
| ct+dt, per algorithm | 0.0291 | 0.0099 | 0.286 | 0.70851 |
| ct+dt+**ratio**, per algorithm | 0.0380 | 0.0138 | 0.286 | 0.69528 |
| ct only, per **action** | 0.0530 | 0.0047 | 0.325 | 0.69375 |

Three corrections: (a) the decision impact is **−80 % mean regret out of sample**, larger than
the error reduction implied; (b) it is a **tail** fix — median regret and top-1 both get worse
and it costs +4.6 % bytes; (c) **per-algorithm generalises better than per-action**
(0.0330 vs 0.0530), the opposite of what the in-sample error reduction suggests. Label fix:
at eb=0 there are **16** actions, not 32.

*Degrees-of-freedom null (`scripts/debate2.py`):* a **random** 8/16/32-way grouping of the same
9440 rows moves median |log| ct error 1.209 → 1.242 / 1.246 / 1.230 — no reduction at all. The
−59 %/−63 % are not DOF artefacts.

**New (adversary's M6), verified here:** the **ratio** head's error is not per-algorithm but
almost entirely **per-chunk**. Median log offset removed per key: per-algo 1.060 → 1.094 (+3 %),
per-action → 1.081, **per-regime → 0.279 (−74 %)**, **per-chunk → 0.218 (−79 %)**. A per-chunk
level offset cancels inside a chunk's ranking but not against the ct/dt terms — the missing
mechanism behind F2's "true ratio makes balanced worse".

**Confidence: high** for the numbers; **medium** for "the API-call bracket is the whole cause"
— a single additive overhead does not fit (ans implies +0.84 ms, deflate implies −3.9 ms), so
hardware/nvcomp-version differences contribute too.

**Class: deployment mismatch** (Clio measures a different quantity than the model was trained
on) on top of an upstream data-collection choice.

---

### F4 — The 100x ratio cap decides 43 % of lossy chunks by tie-break, and the tie-break hands them to action 0 (lz4, lossless).

**Mechanism.** `ratio = min(cap, ·)` (`neuropress_nn_gpu_kernels.cu:538`), then the ranking
tie-break is "lowest action index" (`:780-781, 811-813`) — which is upstream's behaviour, its
lanes *are* actions.

**Evidence (fresh, `f1_infer_eb1e-3.csv`, 440 chunks × 32 actions).**
Distribution of how many candidates saturate the cap: 0 on 182 chunks, **32 (all of them) on
107 chunks**; ≥2 on **43.4 %**. On **191/191** of those the ratio-only pick is exactly the
lowest-indexed saturating action. The ratio-only pick is action 0 — plain lz4, *lossless*,
on a run with eb=1e-3 — on **38.2 %** of chunks. True-best-ratio / picked-ratio: median 1.205,
p90 13.2, max 3591.
Removing the cap (recomputed from the harness's pre-clamp `pred_ratio_raw` column through the
kernel's own clamp chain), both corpora:

| corpus / mode | cap 100 | cap 1e5 |
|---|---|---|
| `f0a` ratio-only: top-1 / regret / med / bytes / ties / picks-a0 | 0.298 / 0.5545 / 0.0903 / 0.67095 / 0.431 / 0.376 | 0.395 / **0.2331** / 0.0112 / 0.65386 / 0.254 / 0.254 |
| **`f1` (eb=1e-3) ratio-only** | 0.109 / **35.99** / 0.1984 / 0.43135 / 0.482 / 0.382 | 0.114 / **0.4350** / 0.0315 / **0.38236** / 0.257 / 0.257 |
| `f0a` balanced | 0.378 / 0.1572 / 0.0097 / 0.67575 | identical to 4 dp |

The eb=1e-3 line — regret **35.99 → 0.435 (83x)** and bytes **−11.4 %** — was missing from the
first version and is the largest single number in the finding set (adversary's catch,
re-derived here). Under the *balanced* cost the cap changes nothing: at ratio 100 the io term
is 0.0084 ms.

**Two corrections from round 1.** (a) *Not all of the tie problem is the cap*: the tie rate
falls only 0.431 → 0.254 (0.482 → 0.257 at eb=1e-3), so **~25 % of chunks are decided by the
action-index tie-break with the cap already removed** — those are the F1 saturation chunks.
The cap accounts for roughly 60 % of the tie problem. (b) *The effect does not travel to every
corpus*: on the adversary's held-out `opt_in` regimes the predictions collapse **low**, never
reach the cap, and removing it changes nothing (0.1514 either way).

**Impact.** Ratio-only / `--best` only. Also the mechanism behind half of F12.

**Confidence: high** (ratio and bytes are bit-deterministic across runs).

**Class: upstream design limitation, faithfully ported** (`RATIO_CAP = 100.0f`,
`/home/cc/NeuroPress/src/api/diagnostics_store.hpp:319`). Per the standing rule this is
quantified, not "fixed".

---

### F5 — On this corpus a single fixed action beats the model out-of-sample, on regret and (in ratio mode) on bytes — in-distribution too.

**Evidence (fresh).** Best constant action fitted on `f0a` (seed 1234), scored on `f0b`
(seed 4321, different data):

| objective | model regret | model bytes | best constant | const regret | const bytes | oracle bytes |
|---|---|---|---|---|---|---|
| balanced, floor 1 ms | 0.1629 | 0.6749 | a7 = bitcomp | **0.0285** | 0.7195 | 0.6498 |
| balanced, floor 0 | 0.7694 | 0.6749 | a23 = bitcomp+shuffle | **0.1018** | 0.7311 | 0.6848 |
| ratio-only | 0.5559 | 0.6721 | a20 = zstd+shuffle | **0.0000** | **0.5883** | 0.5883 |
| balanced, in-dist only (n=350) | 0.1505 | 0.6057 | a7 | **0.0348** | 0.6740 | 0.5995 |
| ratio-only, in-dist only | 0.5804 | 0.5779 | a20 | **0.0001** | **0.5400** | 0.5399 |

At eb=1e-3 (`f1`, in-sample constant): ratio-only model bytes **0.4314** vs constant
`a28 = zstd+quant+shuffle` **0.2500** and oracle 0.2484 — the model stores **72.5 % more
bytes** than one fixed action.

**Correction from round 1 (conceded).** The `INDIST` row above filters on regime names with
`mad ≤ 0.5` only, so it still admits chunks that violate the `.nnwt` **size** bound (`sizes`
runs to 16 MiB) — exactly the chunks F1/F10 call out of distribution. Filtering on **all four**
`.nnwt` bounds (mad ≤ 0.49998, d2 ≤ 1.00573, entropy ≤ 7.52983, 64 KiB ≤ size ≤ 4 MiB), fit on
`f0a`, scored on `f0b` (`scripts/debate1.py`):

| set | n | model regret | const regret | model bytes | const bytes |
|---|---|---|---|---|---|
| all | 590 | 0.1629 | 0.0285 (a7) | 0.67487 | 0.71952 |
| my earlier INDIST | 350 | 0.1505 | 0.0348 (a7) | 0.60568 | 0.67396 |
| **strict `.nnwt` bounds, balanced** | 356 | **0.0297** | **0.0242** (a7) | **0.59355** | 0.65076 |
| **strict `.nnwt` bounds, ratio-only** | 356 | 0.5665 | 0.0001 (a20) | 0.53471 | 0.51763 |

In distribution the **balanced** gap collapses from 4.3x to **1.23x** and the model wins on
bytes by 8.8 %. The ratio-only half survives but its byte gap shrinks from 12.5 % to **3.3 %**.

**Impact, restated.** Under the *ratio-only* objective the model has negative value on this
corpus — but so would any selector: the per-chunk oracle beats the best constant by 0.007 %
(0.58825 vs 0.58829 bytes), so the objective is degenerate here, and that is a statement about
the corpus as much as about the model. Under the *balanced* objective real headroom exists
(oracle 0.0000 vs constant 0.0285 regret; 0.64982 vs 0.71952 bytes); the model captures little
of it out of distribution and roughly as much as a constant in distribution.

**Confidence: high** for the numbers, **medium** for the generalisation — the corpus is
float32 scientific-simulation-like data where `zstd+shuffle` is the true ratio winner on
550/590 lossless chunks, so the action space is close to degenerate for that objective. On a
mixed-modality corpus a selector could earn its keep. This is the honest headline: *the model
must be shown to beat a constant, and here it does not.*

**Class:** consequence of F1-F4.

---

### F6 — Exploration + adoption, not the network, is what makes Clio's NeuroPress path competitive — and exploration is **default-OFF**, so the shipped default is the K=0 row of the table below.

**Mechanism.** Phase 2 measures the top-K *predicted* alternatives and adopts the cheapest
measured (`compressor_runtime.cc:1652-1692, 1989-2028, 2158-2284`).

**Evidence (fresh, simulated offline from the full 32-action measurements in `f0a`/`f1`, so
every number uses the same measured codec outcomes — script `scripts/explore_sim.py`):**

| K | balanced regret | balanced bytes | ratio-only regret | ratio-only bytes | adopt rate | codec runs/chunk |
|---|---|---|---|---|---|---|
| 0 | 0.1572 | 0.6758 | 0.5545 | 0.6710 | 0 | 1 |
| 3 (Clio default) | 0.0711 | 0.6798 | 0.1708 | 0.6182 | 0.47 / 0.63 | 4 |
| 7 | 0.0108 | 0.6728 | 0.0431 | 0.6120 | 0.53 / 0.70 | 8 |
| 31 (`--best`) | 0.0005 | 0.6492 | 0.0000 | 0.5870 | 0.62 / 0.70 | 16 |

At eb=1e-3, ratio-only: bytes 0.4314 (K=0) → 0.3507 (K=3) → 0.2484 (K=31).
The primary is overridden on **47-83 %** of chunks at K=3 — a direct measure of how bad the
primary is. And the window is often blind: the true best action lies inside the explored
top-K on **54.1 %** of chunks at K=3 (balanced, lossless) and **39.8 %** at K=3 (ratio-only,
eb=1e-3); even K=15 only reaches 55.5 % there.

**Impact.** Exploration is the safety net; it is also on-policy, so an action the model
systematically ranks low is never measured and never corrected.

**Corrections from round 1 (both conceded).**
1. *"Default K=3" is wrong.* `neuropress_exploration_enabled_ = false`
   (`compressor_tasks.h:103`), and `neuropress_online_learning_enabled_ = false` (`:73`); K=3 is
   the default *K*, not the default behaviour. With learning enabled but exploration left alone,
   the adaptation path is **phase 1 only** — the K=0 row, the worst row in the table. That is a
   stronger statement than the one first made.
2. *Adoption is partly fitted to timing noise.* Re-scoring the same picks against the
   independent same-seed replicate `f0c`'s measured table moves K=31 from 0.0000 to 0.0092 and
   K=7 from 0.0104 to 0.0292. Ordering unchanged, ratio-only column unaffected (ratios are
   bit-deterministic), but the balanced K=31 row is optimistic by an order of magnitude.

**Confidence: high** for the offline simulation; **medium** for the absolute values in the
live runtime, because real sweep slots contend for the GPU (the reason for commit 81ac5a43;
its own measurement: per-slot kernel time inflated up to 76x, median 1.40x, before the
4-stream cap). Simulated adoption uses uncontended timings and is therefore an upper bound.

**Class: not a bug** — it is the design working. The finding is that the *network's* share of
the credit is small.

---

### F7 — Within one phase-1 sample the SGD update **direction** is sign-only; its **size** is linear in the error between the 0.10 noise gate and saturation, then constant. Uncertainty weighting is a no-op on the direction.

**Mechanism** (all line refs `neuropress_nn_gpu_kernels.cu`). For a single sample the whole
per-output gradient vector is linear in `|d5[target]|` (W5 row: `error·h4`, `:1679-1681`;
trunk: `dz4 = unit(dz4_all)·err_mag`, `:1660-1675`). The per-output clip renormalises it to a
fixed norm: `clip_scale = 0.1/‖g‖` whenever `‖g‖ > 0.1` (`:1774`). Every constant multiplying
`d5` — the Kendall weight `uw = exp(−lv/2)` (`:1621-1622`) and the ±0.5 clamp (`:1596-1598`) —
therefore drops out of the **direction**. Magnitude survives only through
`step = clamp(0.08·mean|d5_raw|, 1e-4, 0.02)` (`:1836`), which saturates at 0.02 once
`mean|d5_raw| ≥ 0.25`, and through the noise gate that zeroes `d5[0]` below 0.10 (`:1591-1592`).

**Evidence, corrected in round 1.** My original E1/E1c examples all sat in the *saturated*
regime, because their ratio label was grossly wrong; generalising from them to "the learner can
only ever learn too-high/too-low, never by how much" was an over-reach. The adversary's A2
probe caught it and I reproduced the full picture with my own probe (`np_sgd_exec` **E1e**:
one sample, ratio and psnr labels set to the model's *own* predictions so only ct carries error):

| ct label / pred | 1.000x | 0.990x | 0.950x | 0.900x | 0.800x | 0.600x | 0.400x | 0.200x |
|---|---|---|---|---|---|---|---|---|
| \|d5[0]\| | 0.0000 | 0.0110 | 0.0558 | 0.1139 | 0.2374 | 0.5213 | 0.8743 | 1.3413 |
| ‖ΔW‖ | 1.4976e-5 | 1.4976e-5 | 1.4976e-5 | 4.556e-4 | 9.498e-4 | 2.085e-3 | 3.000e-3 | 3.000e-3 |
| cos(ΔW(0.99x), ΔW(·)) | 1.000000 | 1.000000 | 1.000000 | 0.621288 | 0.621299 | 0.621301 | 0.621301 | 0.621301 |

Three regimes, all visible at once: **below the 0.10 noise gate** the ct error is discarded
entirely (‖ΔW‖ pinned at the other heads' residual); **between 0.10 and ≈0.87** ‖ΔW‖ is
*exactly linear* in |d5[0]| (slope 4.00e-3 = `0.08·(1−EMA)/3 heads` to three digits at every
point); **above that** the trust step saturates at `0.02 × 0.15 = 3.0e-3`. The **direction is
constant to 6 digits across the whole open region** — that half of the original claim stands.
E1c's exact invariance (cos = 1.000000000, ‖ΔW‖ equal to 7 s.f. across a 5x range of |d5[0]|)
is the saturated corner of the same picture.

**Batched (phase-2) case, my own probe `np_sgd_exec` E1d.** `out_grad` accumulates across
samples *before* the clip, so the samples' relative weights are set by their `|d5|` and do not
cancel exactly. Varying only sample 0's ct label inside a 7-sample batch (same sign throughout):
‖ΔW‖ = 2.999981e-3 / 2.999977e-3 / 2.999973e-3 / 2.999972e-3 and
cos = 0.999735 / 0.998661 / 0.998154. So the leak is real but **≤0.2 % of the direction and
nil in the step size** for same-sign perturbations. (The adversary reports cos 0.910; that
comparison crosses a sign flip, which F7 predicts.) See DEBATE_R1.md — partially disputed.

**Impact.** The uncertainty weighting cannot change what the learner does. The magnitude the
learner *can* use is a single scalar step shared by all 13576 parameters, clipped at 0.02, and
below the 0.10 noise gate the ct error is simply thrown away — which is why the ct head settles
7 % from its target and stays (E2) and why every SGD-dynamics knob the prior session tried
landed inside noise.

**Confidence: high** (analytic mechanism + numerical confirmation in three regimes).
Falsified by an SGD call where two same-sign label magnitudes in the linear regime give
`cos < 0.99` for a single sample.

**Class: upstream design limitation, faithfully ported** (`GRAD_CLIP_THRESHOLD = 0.1f`,
`/home/cc/NeuroPress/src/nn/nn_gpu.cu:927`).

---

### F8 — Judged as in-memory adaptation from fresh weights: a fresh process does **not** end up better adapted. The mechanism is not mainly cross-action interference — it is *what the gate selects to train on*, and how little the weights can move.

Reframed in round 1 under the deployment constraint. "Turn the learner off" is not an available
answer, so the question is: does a fresh process, starting from `model.nnwt` and adapting over
the chunks it happens to see, end up better than it started, and how many chunks does that take?

**F8.1 — Whole-run outcome (fresh, 590 chunks each, same seed, same data as `f0a`, idle GPU).**

| run | SGD calls | balanced top-1 | balanced regret | median regret | bytes |
|---|---|---|---|---|---|
| `f0a` no learning | 0 | 0.407 | 0.1572 | 0.0097 | 0.6758 |
| `f2` gate 0.30 (shipped) | 45 | 0.329 | 0.1462 | 0.0138 | 0.6804 |
| `f4` train every chunk | 590 | **0.207** | **0.2624** | 0.0138 | 0.7008 |
| `f3` ratio-only **cost weights**, gate 0.30 | 226 | — | — | — | ratio-only bytes 0.7260 vs 0.6710 |

**Qualification conceded in round 1:** at the *shipped* gate the picture is mixed, not uniformly
bad. Scoring `f0a` and `f2` picks against **one** common measured table (the adversary's L1),
`f2` is better on 42 chunks and worse on 83, loses top-1 (0.378 → 0.290) and bytes (+0.7 %), but
its *total* measured cost is **−16.4 ms**, i.e. very slightly better, because its few wins are
large. So "learning makes selection worse" is true for top-1, bytes and the median chunk and
**false for total cost** at the default gate. Training every chunk (`f4`) is unambiguously worse
on every metric (+149.8 ms total cost, 135 better / 222 worse).

**Disputed:** the adversary strikes the `f3` row as "a configuration that cannot run". `--best`
does disable both SGD phases (`compressor_runtime.cc:1577, 2290`) — but `f3` is not `--best`.
The ratio-only *cost weights* are reachable through `CLIO_NEUROPRESS_COST_W_CT=0
CLIO_NEUROPRESS_COST_W_DT=0` with best mode off, and `:1447-1449` then builds the gate cost from
those env weights while `:1577` leaves training enabled. `NPE_COST=ratio` models exactly that.
The row is relabelled (it is *not* about `--best`) and kept. See DEBATE_R1.md.

**F8.2 — How far can a fresh process move at all? (from `*.weights.csv`, verified in round 1.)**

| run | SGD calls | final L2 drift | ‖W‖ | drift as % of ‖W‖ |
|---|---|---|---|---|
| `f2` (gate 0.30) | 45 | 0.1013 | 20.103 | 0.50 % |
| `f3` (ratio-weight gate) | 226 | 0.2050 | 20.100 | 1.02 % |
| `f4` (every chunk) | 590 | 0.1817 | 20.098 | **0.90 %** |

590 steps move the weights **less** than 226 do. The trust step, EMA 0.85 and the anti-flip ×0.5
cap the excursion, and because the direction is sign-only (F7) consecutive steps on different
data largely cancel. **A longer process does not become a better-adapted process.** (Credit:
the adversary's L0; I verified it on my own dumps.) The outputs are nevertheless very sensitive
to that small drift — 20 steps on one action moves lz4's predicted ratio 20.5 → 1.21 (E3).

**F8.3 — What the gate selects to train on (the adversary's L2, verified on my runs).** Using
the harness's `n_x_oob` column (how many of the 8 standardized inputs fall outside the `.nnwt`
bounds), counting the chunks where phase 1 actually fired:

| run | SGD calls | fired on a chunk with ≥1 out-of-bounds input | corpus base rate | top regimes |
|---|---|---|---|---|
| `f2` gate 0.30 (shipped) | 45 | **42 = 93 %** | 40 % | smooth_a1000 20, sizes 8, gs_x1e4 7, turb_x50 6 |
| `f3` ratio-weight gate | 226 | 139 = 62 % | 40 % | smooth_a1000, lammps_pos_*, lammps_force |
| `f4` every chunk | 590 | 234 = 40 % | 40 % | uniform (as expected) |

The cost MAPE is large exactly where the model is extrapolating (F1), so **the shipped gate is
an out-of-distribution detector, not an error detector**. Phase 1 then applies a *global* trunk
update computed entirely from off-manifold data to every subsequent in-distribution chunk. This
is a cleaner explanation for "learning loses top-1 and bytes" than cross-action interference
alone, and it is the adversary's finding, not mine.

**F8.4 — And the gate is a device-load detector too (the adversary's L4, verified here).**
Recomputing `error_pct` exactly as `compressor_runtime.cc:1447-1462` does over `f0a`'s 590
primary rows, with every measured compress time multiplied by a device-load factor `k`:

| k | 0.5x | **1.0x (idle)** | 2x | 4x | 8x | 14x | 20x |
|---|---|---|---|---|---|---|---|
| phase-1 gate opens on | 0.234 | **0.239** | 0.317 | 0.408 | 0.683 | 0.997 | 1.000 |

`actual_cost` is built from a *measured* codec time, so on a busy GPU it moves away from
`predicted_cost` everywhere and the gate saturates. The shipped adaptation policy is therefore
"train harder when the machine is busy", which is not a property anyone designed, and it means
any single learning run is one draw from a wide distribution. **I withdraw the "high confidence,
four runs" wording of the first version** — those were four configurations, not four replicates,
and the adversary has two nominally identical runs that differ 7x in SGD count.

**F8.5 — The interference mechanism, rescoped.** Training 20 steps on action 0 alone moves every
action's predicted ratio (lz4 20.51→1.21, snappy 8.07→1.53, zstd 25.46→3.17, bitcomp 1.01→0.29;
`np_sgd_exec` E3). The adversary measured what that does to the *order* — Spearman(before,after)
= 0.894 for ratio and 0.903 for ct — so the values move far more than the ranking does.
Interference is real but it is a second-order cause of the selection loss, behind F8.3/F8.4.

**F8.6 — Phase 1 is closed-loop on its own mistake (the adversary's A4; I have no independent
run of it).** Training the action the model just picked, with that action's measured labels,
converges the prediction for *that* action to the measured value in ~20 steps and never changes
the pick, because the other actions are never labelled. Phase-1 self-training is on-policy with
a batch of one and is structurally incapable of correcting a wrong pick. This is the strongest
argument for exploration and against any "phase 1 alone will adapt" expectation.

**F8.7 — How many chunks does adaptation need to beat no-learning? (fresh runs `h0`-`h3`,
450 chunks = three 150-chunk regime blocks, `turb_x50 → smooth_a1_n1e-2 → lammps_pos_sorted`,
fresh weights, balanced. `h0`/`h1`/`h2` were measured on an idle GPU; `h3` was not — see below.)**

Crossover = the first chunk within a block from which the run's running-mean regret stays at or
below the no-learning baseline's for the rest of the block.

| block | `h1` phase-1 only (gate 0.30) | `h2` phase-1 fed by exploration K=7 |
|---|---|---|
| `turb_x50` | sgd 10, never crosses, 2nd-half regret 2.98 vs base 2.07 | sgd 5, **crosses at chunk 5**, 0.81 vs 2.07 |
| `smooth_a1_n1e-2` | sgd 62, **crosses at chunk 112**, 0.875 vs 1.223 | sgd 109, **crosses at chunk 17**, 0.420 vs 1.223 |
| `lammps_pos_sorted` (after two other blocks) | sgd 2, never crosses, 1.35 vs 0.854 | sgd 6, never crosses, 2.76 vs 0.854 |

Read: **phase-1-only adaptation needs ~112 chunks / 62 SGD calls to beat fresh weights, and only
in a block where the gate fires often; feeding it exploration samples cuts that to 5-17 chunks.
Neither survives a regime change** — the block that follows two others is worse than fresh
weights in both cases, which is the interference/OOD-training signature above.
(`h3`, a per-action ct residual table instead of SGD, gave whole-run top-1 0.207 vs `h0` 0.122
and crossed at chunk 91 on `smooth`, but it ran while another GPU job was active, so its
*balanced* numbers are not comparable to `h0`'s and are marked provisional; its deterministic
ratio-only column is unchanged from `h0` by construction, since the table only corrects ct.)

**Class:** the loss is caused by (a) what the gate selects — upstream design, `:1447-1462`;
(b) how little the weights can move — upstream constants; (c) scalar `algo_id` through a shared
trunk — upstream representation. None of it is a port bug.

---

### F9 — The SGD gate never opens on half the chunks whose ratio prediction is wrong — and Clio already ships the remedy as an env knob, which the first version did not mention.

**Mechanism.** The gate is on the *cost* MAPE only (`compressor_runtime.cc:1557-1559`), and its
decompress term is identical on both sides by construction (`:1457-1459`), so a ratio miss must
move the io term — 12.4 % of the cost on average — by >30 % of the total to be seen.

**Evidence (fresh, `f0a` primary rows, n=590, threshold 0.30).**
Gate open 23.9 %; ratio MAPE > 0.30 on 47.5 %. Cross-tab: both 23.6 %, **ratio-bad with the
gate shut 23.9 %**, cost-open-ratio-fine 0.3 %, neither 52.2 %. So **141/280 (50 %)** of
ratio-bad chunks never train; the misses they hide have median ratio MAPE 0.82 and reach 99.2.

**Correction from round 1 (conceded).** `compressor_runtime.cc:1536-1548` already implements
exactly this remedy — `CLIO_NEUROPRESS_SGD_ON_RATIO=1` ORs a ratio-MAPE gate onto the cost gate,
with `CLIO_NEUROPRESS_RATIO_MAPE_THRESH` for its threshold — and `np_eval` exposes it as
`NPE_GATE=ratio_or_cost`. I described the limitation and recommended against widening the gate
without measuring the knob that exists for it. The adversary measured it; I did not.
Also: the code comment at that site claims the io term is "~0.4 % of the cost" at 4 MiB; measured
it is mean 12.4 %, median 3.3 % (predicted) / 6.5 % (measured) — the comment is off by more than
an order of magnitude.

**Why the ratio gate is the interesting one under the deployment constraint.** The cost gate is
built from a *measured* codec time, so it is an OOD detector on an idle device (F8.3) and a load
detector on a busy one (F8.4) — neither is "the model is wrong here". The ratio MAPE is
**timing-independent** (ratios are bit-deterministic across runs, 9440/9440 rows), so it is the
only deterministic trigger the runtime already has. Whether it *helps* is the open question in
DEBATE_R1.md.

**Confidence: high** for the numbers.  **Class: upstream design (gate on cost alone), faithfully
ported; the ratio-gate knob is a Clio addition.**

---

### F10 — Chunks larger than 4 MiB are out of distribution and the ratio head saturates. (The claim that chunk size is the input the model uses *least* in range is withdrawn — see below.)

**Evidence (fresh).** `np_sgd_exec` **E7**, one feature swept over its full training range with
the others at the training mean: predicted ratio moves 26.1x for `d2`, 6.5x for `entropy`,
2.4x for `mad`, and **1.20x for `size`**.
`np_sgd_exec` **E5** / `np_ood_upstream_exec`, identical content, size swept:
predicted-ratio range over the 8 lossless algorithms
[0.77, 18.1] @64 KiB, [0.55, 25.5] @4 MiB, [1.76, **100.0**] @8 MiB, [4.99, **100.0**] @16 MiB,
[41.2, **100.0**] @64 MiB. Upstream reproduces this exactly (same winner, 0.0e+00 difference).
On real measured data (`sizes` regime, fixed content): median ratio APE
1.56 (1 MiB) → 3.87 (4 MiB) → **40.6** (8 MiB) → 13.4 (16 MiB); top-1 1.00 → 1.00 → 0.00 → 0.00;
balanced mean regret 0.000 → 0.000 → 0.058 → **6.997**.
Also at 64 MiB the model predicts **34.5 dB PSNR for a lossless action** — enough to be masked
out by any `target_psnr` setting.

**Two corrections from round 1 (conceded).**
1. *"The input the model uses least in range" is withdrawn.* E7 pins the other three inputs at
   the training **mean** and sweeps one — a single-point marginal sensitivity. Entropy, mad and
   d2 co-vary in real data and size does not, so comparing marginal ranges does not establish
   relative importance. The E7 numbers stand as what they are; the ranking conclusion does not.
2. *In-range small chunks are also badly predicted.* My own `sizes` table starts at 0.25 MiB and
   I quoted it from 1 MiB: median ratio APE is **4.12 at 0.25 MiB** vs 1.56 at 1 MiB and 3.87 at
   4 MiB — all inside the training range. "In-range size is harmless" is not true either.

**Impact.** Any deployment with chunks >4 MiB. The `clamp` transform of F1 covers this too
(it clamps `chunk_size_bytes` into [64 KiB, 4 MiB], `np_eval.cu:1424-1426`).

**Confidence: high.**  **Class: upstream design limitation** (training set caps at 4 MiB).

---

### F11 — The PSNR head cannot rank anything, because its own ground truth is a per-chunk constant; when `target_psnr` is set it discards a third of the legal action space.

**Mechanism.** The label is `AnalyticalPsnr(data_range, error_bound)`
(`compressor_runtime.cc:131-136`, verbatim upstream) — a function of two per-chunk constants,
so it is identical for all 16 quantized candidates. The mask compares the *prediction* to
`min_psnr` (`neuropress_nn_gpu_kernels.cu:800-802`).

**Evidence (fresh, `f1`, 7040 quantized rows / 440 chunks).**
Distinct **actual** PSNR values per chunk across the 16 quantized actions: **1, on 440/440
chunks**. Distinct **predicted** values: 16 on 310/440 chunks — the head manufactures spread
that the label says does not exist. PSNR APE **over the 6496 of 7040 rows whose analytic PSNR is positive** (that filter was in my
script but not in the first write-up): median 22.2 %, p90 100 %, mean 347 %. The filter matters
because 544 rows (7.7 %) have a **negative** analytic PSNR, where a percentage error is
meaningless; over all 7040 rows the same quantity is median 14.0 % / p90 100 % / mean −154 %
with the signed denominator and median 25.4 % / p90 118 % / mean 795 % with |denominator|.
All three reproduce; none changes the conclusion.
With `min_psnr = 60 dB`: **35.2 % of quantized candidates are wrongly masked out** and 9.7 %
wrongly admitted (80 dB: 39.0 % / 8.1 %; 100 dB: 29.5 % / 6.6 %).

**New (adversary's M7), verified here:** the psnr output is clamped to **[0, 120]**
(`neuropress_nn_gpu_kernels.cu:534`) while the analytic label it is trained against reaches
**−283.6 dB** (544/7040 quantized rows, 34 chunks, in `const` and `gs_*`). On those chunks the
head cannot represent its own target at all, so every SGD step there carries a permanent,
unclosable error — the F16/E11 effect two orders of magnitude larger.

**Impact.** Zero when `target_psnr_ = 0` (the default). Large when it is set.

**Confidence: high.**  **Class: upstream design limitation, faithfully ported.**

---

### F12 — The model is close to blind to the byte-shuffle half of its own action space, and the ratio cap causes a third of that blindness.

**Evidence (fresh, `f0a`, 4720 (chunk, algo) pairs comparing action `a` with action `a+16`).**
Byte shuffle actually improves the ratio on **86.3 %** of pairs (median +12.7 %).
The model predicts an improvement on **43.3 %** — precision 0.927, **recall 0.466**. It
predicts *exactly no difference* on **32.9 %** of pairs, and on **27.7 %** of all pairs that is
because both the plain and the shuffled prediction sit at the 100x cap.
Per algorithm, predicted median r_shuf/r_plain: 1.000 for lz4/snappy/deflate/gdeflate/zstd/
bitcomp, 1.543 for ans, 1.983 for cascaded — against a measured 1.12-1.20 for all but bitcomp.
The quantize dimension is better but still weak: direction agreement **71.4 %** at eb=1e-3.

**Impact.** Half the 32-action space is chosen at chance. Explains why the true best is so
often outside the explored window (F6).

**Confidence: high** (deterministic quantities).

**Class: upstream design limitation** — `shuffle` is a single 0/1 input
(`neuropress_nn_gpu_kernels.cu:453-455`) and the entropy/mad/d2 the network is given describe
the *un*shuffled bytes, while the codec sees the shuffled ones.

---

### F13 — Non-float32 data is read as float32 with no error, producing features 10³⁶ σ out; a single non-finite feature silently degrades selection to "always lz4".

**Mechanism.** `neuropress_selection.cc:48` pins `DataType::FLOAT32` for every
`context.data_type_` except 2 (float64, which is *converted*, `:68-74`);
`ComputeNeuroPressFeatures` does the same (`data_stats_gpu.h:219-222`). In the network,
`fmaxf(0, NaN)` returns 0, so a NaN input zeroes the entire first hidden layer and the output
becomes the bias vector — one constant for all 32 actions.

**Evidence (fresh, `np_sgd_exec` E10 / E9).**
A 1 MiB uint8 count field read as float32 → **mad = 3.02e36, d2 = 6.18e36** (training max 0.5).
A 1 MiB int32 index buffer read as float32 → mad = 3.32e-41 (denormals); its true integer MAD
is 2.37e4.
Feeding a non-finite feature to the predictor:

| features | distinct ct over 8 algos | distinct ratio | cost argmin |
|---|---|---|---|
| baseline | 8/8 | 8/8 | ans |
| mad = NaN | **1/8** | **1/8** | **lz4 (action 0)** |
| d2 = NaN | 1/8 | 1/8 | lz4 |
| mad = +Inf | 1/8 | 1/8 | lz4 |
| mad = 1e12 | 1/8 (ct = 1.0) | 1/8 (ratio = 0.1) | lz4 |

**Impact.** Silent. No log line, no failure path — `features_ok` is true and the ranking looks
well-formed. Selection becomes the action-index tie-break.

**Round-1 note — one adversary correction rejected.** The adversary narrows F13 to
"device-path-only, because the host path already honours `data_type_`". That is not what the
host path does: `ComputeNeuroPressFeatures` special-cases **only** `data_type_ == 2` (float64)
and falls through to a float32 read for every other type
(`data_stats_gpu.h:196-217` then `:219-222`). So a uint8 or int32 payload is read as float32 on
**both** paths; the E10 measurements above were made through the host entry point. What *is*
device-path-only is the missing *finiteness* guard on the way into the kernel — the host path is
equally unguarded but at least fails loudly if the buffer is unreadable. F13 stands unnarrowed;
see DEBATE_R1.md.

**Confidence: high** for the mechanism and the numbers; **medium** for the deployment
frequency, which depends on whether non-float payloads reach a NeuroPress-enabled pool.

**Class: deployment/data mismatch.** Upstream also assumes float32, but upstream has no
`data_type_` field to contradict; Clio does, and honours it only for float64.

---

### F14 — The cost model's 1 ms time floor is currently *hiding* prediction error, so lowering it would make selection worse — but it also costs mean regret 0.42 against the unfloored optimum even with perfect predictions.

**Evidence (fresh, `f0a`, ground truth = measured ct+dt+io with **no** floor and no cap).**

| predictor-side policy | top-1 | mean regret | median | p90 |
|---|---|---|---|---|
| shipped: cap 100, floor 1 ms | 0.266 | 0.7241 | 0.0790 | 1.8098 |
| cap 100, floor 0.1 ms | 0.205 | 0.8856 | 0.1446 | 1.8743 |
| cap 100, no floor | 0.205 | 0.8856 | 0.1446 | 1.8743 |
| cap ∞, floor 1 ms | 0.266 | 0.7241 | 0.0790 | 1.8098 |

And with **all three heads replaced by measured truth** but the 1 ms floor still applied to the
predictor side: top-1 0.485, mean regret **0.4203** against the unfloored optimum (0.0000
against the floored one). 29.1 % of measured compress times and 59.7 % of measured decompress
times are below 1 ms on this hardware.

**Impact.** The prior session's improvement idea "I7: drop the cost floor to 0.1 ms" is
**refuted** — it costs 0.061 top-1 and +22 % mean regret. The floor is doing real work by
truncating a head whose predictions are 21x too narrow (F3).

**Confidence: medium-high.** This is the most timing-noise-sensitive claim in the set; the
unfloored ground-truth best is only 77.8 % reproducible between two identical runs, so treat
the *unfloored* column as directional. The floored comparison (96.8 % reproducible) is solid.

**Scope added in round 1:** this is an **idle-device** result. Under GPU contention the
fraction of measured compress times below 1 ms falls from 29.1 % to ~7 %, so the floor truncates
far less and "the floor is doing real work" weakens correspondingly. Since Clio's own phase-2
exploration creates contention, both regimes occur in production.

**Class: upstream design (`fmaxf(1.0f, ·)`, `/home/cc/NeuroPress/src/nn/nn_gpu.cu:229-231`),
faithfully ported, but interacting badly with A100-class hardware.**

---

### F15 — The deferred decompression head trains toward a constant on this hardware; it does not matter.

**Mechanism.** `LearnDecompTime` floors the measured time at 1 ms before it becomes a target
(`compressor_runtime.cc:935`), and the dt output is floored at 1 ms anyway
(`neuropress_nn_gpu_kernels.cu:537`).

**Evidence (fresh, `np_sgd_exec` E8).** With the runtime's floored target of 1.0 ms the head
converges in 5 updates to dt = 1.012 / 1.000 / 1.078 / 1.000 for algos 0/1/4/7 — effectively a
constant; the trunk (ct, ratio) is untouched, which is correct head-only behaviour. With an
informative target of 2.5 ms it converges to 2.481 and keeps per-algorithm spread. 59.7 % of
measured decompress times in `f0a` are below 1 ms and therefore floored to the same value.

**Impact: small.** Pinning dt to 1.0 for every candidate changes the balanced pick on **2.0 %**
of chunks; top-1 0.378 → 0.378, mean regret 0.1572 → 0.1569, bytes 0.6758 → 0.6752.
At write time the dt term contributes exactly zero to `error_pct` (`:1457-1459`) and the
phase-1 label is hard-coded 0.0 (`:1619`), so the head is untrained by both write-side phases.

**Confidence: high.** Listed to close it out, not because it is worth fixing.

**Class: faithful port of an upstream design that assumes ms-scale decompression.**

---

### F16 — The lossless `1e-7` error-bound sentinel is numerically inert, and the psnr label bias on lossless primaries is second-order.

Two things the port worries about that measurement says do not matter:
`np_sgd_exec` **E6** — sweeping input 3 with `quant=1` so the substitution does not fire:
eb 0 → 1e-7 → 1e-5 changes the predicted ratio by 0.005 % and ct by 0.00 % (18.629 → 18.629 →
18.630). The sentinel is 2.4e-6 σ of input 3; matching upstream here is correct but changes
nothing.
`np_sgd_exec` **E11** — a lossless primary is trained with `psnr = 120.0`
(`compressor_runtime.cc:1611-1613`) while the head predicts 114.68, so every lossless SGD step
carries a guaranteed |d5[3]| = 0.147. Replacing that label with the model's own prediction
(zero psnr error) after 20 steps changes ct[0] 4.089 → 3.790 (7.9 %) and ratio[0]
1.2110 → 1.2139 (0.2 %). Real, small.

**Confidence: high.**  **Class:** both faithful to upstream.

---

### F17 — Port faithfulness (context for everything above).

17/17 `ctp_neuropress_*` parity ctests pass (16.97 s total, on an **idle** GPU; the adversary's
16/17 was run on a shared device and its single failure is a `cudaMemGetInfo` assertion that is
invalid there, not a port defect); they compile
`/home/cc/NeuroPress/src/nn/nn_gpu.cu`, `src/stats/stats_kernel.cu` and
`src/stats/entropy_kernel.cu` and diff Clio against them
(`parity/CMakeLists.txt:22-26, 82-85`). My own cross-check
(`build/bin/np_ood_upstream_exec`) adds 17 sweep points across amplitude 1x-10⁴x, size
64 KiB-64 MiB and eb 1e-6-1e-1: **0 winner mismatches, 0.000e+00 worst relative difference**
on the winner's ratio/ct, live-half ordering identical (16/16 lossless, 32/32 lossy).
One caveat found: at mad = 57.4 the live-half order agrees 14/16 — several candidates have
*exactly* equal cost there (times clamped, ratio capped) and Clio's explicit
`(score, action, slot)` comparator breaks the tie deterministically where upstream's bitonic
network does not. The winner still matched. Every accuracy claim above is therefore about the
model and its deployment, not about the port.

---

## Section C — Ranked improvements

Restructured in round 1 under the deployment constraint. Two things are being improved and they
are not interchangeable:

* **C-A — a better starting point.** Every process begins from `model.nnwt`. Anything that makes
  that file, or the inputs fed to it, better helps *every* chunk of *every* run immediately and
  needs no chunks to converge. Offline or inference-time.
* **C-B — a better in-run adaptation loop.** The learner has only the chunks of one process, it
  starts fresh each time, and F8.2 shows it barely moves. Anything here has to converge in tens
  of chunks, not hundreds, and must not be poisoned by what the gate selects (F8.3/F8.4).

Two conditioning facts govern the whole list: **which head is worth fixing depends on storage
bandwidth** (F2 — at ≤500 MB/s the ct fixes become harmful and the ratio fixes become the
valuable ones), and **which objective you are running** (`--best` is ratio-only *and* disables
both SGD phases, so nothing in C-B applies there).

### C-A. Better starting point (ranked)

**A1. Retrain the `.nnwt` on this deployment's own labels (file swap, no code change).**
Addresses F1, F3, F10, F12 at the root, and is the only change that improves the *starting*
weights every fresh process adapts from — which under this deployment mode is the highest-value
lever there is. Three changes to the data recipe, each justified by a measurement here:
(a) label compress time with the **codec kernel** time, not upstream's whole-API-call wall-clock
bracket (F3, verified in `/home/cc/NeuroPress/scripts/benchmark.cu:585-592`);
(b) cover chunk sizes past 4 MiB and below 1 MiB (F10, both ends are bad);
(c) cover amplitudes past mad = 0.5, or range-normalise mad/d2 as upstream's own docstring
already claims (F1; the `range` transform alone gets balanced regret 0.157 → 0.097 and
ratio-only bytes 0.671 → 0.630 on the *shipped* weights).
Headroom from my own data: regret 0.157 → 0.002 and bytes 0.676 → 0.650 (F2's "true ct+ratio"
row and F5's oracle). Faithfulness cost: none for (a)/(b); (c) diverges from upstream's code.

**A2. Raise the ratio cap, mode-scoped to the ratio-only objective.** Addresses F4, F12.
Measured: at eb=1e-3 ratio-only regret **35.99 → 0.435 (83x)** and bytes **0.4314 → 0.3824
(−11.4 %)**; lossless ratio-only regret 0.5545 → 0.2331 and bytes 0.6710 → 0.6539.
**No effect at all** on the balanced objective. `CLIO_NEUROPRESS_RATIO_CAP` already exists
(`neuropress_bridge.cc:77`); the standing rule keeps the default at 100, so scope it to `--best`.
Caveats from round 1: it fixes ~60 % of the tie problem (25 % of chunks still tie on the
action index with the cap gone), and it does nothing on a corpus whose predictions collapse
*low* rather than saturating high.

**A3. Clamp the four continuous raw inputs to `x_mins`/`x_maxs` before standardization.**
Addresses F1, F10 and part of F13. Five standardization sites, all in
`neuropress_nn_gpu_kernels.cu`: `:588-591`, `:626-629`, `:715-717`, `:890-893`, `:1520-1522`.
Measured: **ratio-only** bytes 0.6710 → 0.6033 on `f0a`, 0.4314 → 0.2817 (−34.7 %) at eb=1e-3,
−11.5 % on the adversary's held-out corpus — all deterministic, all wins.
**Balanced**: mean regret 0.1572 → 0.0395 and median 0.0097 → 0.0006, but stored bytes get
**worse** (0.67575 → 0.68848, +1.9 %), and inside the strict `.nnwt` bounds the predictions are
bit-identical so there is no in-distribution effect at all. Demoted from "the single best change
available" to "the best inference-time change for the ratio objective, a bytes-for-time trade
for the balanced one". Faithfulness cost near zero — the bounds ship in the file and upstream
computes them (`neural_net/core/data.py:163-164`) but never consults them at inference.

**A4. Per-**algorithm** ct log-bias correction (static table shipped with the weights).**
Addresses F3. Out of sample: balanced regret **0.1629 → 0.0330 (−80 %)**. Corrections from
round 1: use **per-algorithm, not per-action** (per-action generalises worse, 0.0530); it is a
**tail** fix that makes the median chunk and top-1 slightly worse and costs **+4.6 % bytes**;
and per F2 it is **harmful below ~500 MB/s storage**, so gate it on bandwidth, not only on an
env var. Do not extend it to the ratio head — that head's error is per-*chunk*, not per-algorithm
(F3/M6), and adding it makes things worse (0.0330 → 0.0380).

**A5. Guard the feature path: reject/warn on non-float payloads and on non-finite statistics.**
Addresses F13. A NaN or a 1e36 statistic currently produces a well-formed ranking that is always
action 0, with no log line. A finiteness and bounds check at `neuropress_selection.cc:131-137`
costs nothing and routes to the "falling back to Clio's legacy heuristics" path already spelled
out at `:177-182`. Note this applies to both the device and host paths (the host path also reads
non-float64 as float32, `data_stats_gpu.h:219-222`).

**A6. Keep the 1 ms cost floor.** F14: lowering it to 0.1 ms costs 0.061 top-1 and +22 % mean
regret today, and the finding survives the noise check on a paired replicate. Scope: this is an
idle-device result (the floor truncates 29 % of measurements idle, ~7 % loaded). Revisit only
after A1, when the ct head's dynamic range matches the hardware's.

### C-B. Better in-run adaptation (ranked) — what actually blocks it, and what to change

The three things that stop adaptation from helping, in order of measured weight:

1. **The trigger is not model error.** The cost gate is an OOD detector on an idle GPU (93 % of
   its firings, F8.3) and a device-load detector on a busy one (F8.4). It trains a *global* trunk
   update from off-manifold or load-inflated data and applies it to every later chunk.
2. **The learner is on-policy with a batch of one.** Phase 1 only ever labels the action it just
   picked, so it converges that action's prediction to the truth and never moves the pick
   (F8.6). Nothing in phase 1 can discover that a different action was better.
3. **The update is a single sign-directed step shared by 13576 parameters** (F7), the trust
   region caps the excursion, and consecutive steps cancel — 590 steps move ‖W‖ by 0.90 %, fewer
   than 226 steps do (F8.2). Longer runs do not become better-adapted runs.

**B1. Feed the learner exploration samples, not the primary alone.** This is the single measured
fix. Fresh runs `h0`/`h1`/`h2`: phase-1-only adaptation needs **~112 chunks / 62 SGD calls** to
beat fresh weights and only where the gate fires often; with `NPE_EXPLORE_K=7` at the same gate
it crosses over at **chunk 5** and **chunk 17** in the two blocks where it crosses at all, and
second-half regret falls 2.07 → 0.81 and 1.22 → 0.42. It also directly attacks blockers 1 and 2:
off-policy labels for actions the model did not pick. Cost: K extra codec runs per gated chunk
(F6's table quantifies the trade; K=3 already halves regret in simulation). Exploration is
**default-off** (`compressor_tasks.h:103`), so this is a configuration change, not new code.

**B2. Trigger on the ratio MAPE, and *replace* the cost gate rather than ORing onto it.**
Ratios are bit-deterministic across runs (9440/9440 rows); measured times are not (median 1.3 %,
p99 81 %, max 97x). Clio already ships `CLIO_NEUROPRESS_SGD_ON_RATIO=1` +
`CLIO_NEUROPRESS_RATIO_MAPE_THRESH` (`compressor_runtime.cc:1536-1548`,
`NPE_GATE=ratio_or_cost`), and my F9 table proves the property it is wanted for: the ratio gate
fires on **0.475 of chunks at every device-load factor from 0.5x to 14x**, while the cost gate
runs 0.239 → 0.997 over the same range. Two refinements the shipped knob does not give you:
(a) because it **ORs** the two gates, the combination still saturates to 0.997 under load, so the
cost gate has to be *replaced*, not supplemented; (b) the ratio gate still fires on out-of-bounds
chunks 66 % of the time (base rate 40 %), so pair it with an `n_x_oob == 0` precondition to stop
the learner training on extrapolations at all (F8.3). The adversary measured the shipped OR knob
live (their L6: top-1 0.350, the only learning configuration that does not *cost* top-1 —
though also not better than the 0.359 of not learning, and at an uncontrolled contention level).
**The replaced-gate + OOB-precondition combination is unmeasured by either of us** —
DEBATE_R1.md points D1/D2.

**B3. Learn the magnitude outside the network.** F7 shows the trunk can only use a clipped scalar
step; a per-algorithm log-residual table on the time heads is not subject to the clip, is
per-action so it cannot smear one action's outcome across the other 31 (F8.5), and converges in
a handful of updates. `np_eval` implements it (`NPE_BIAS`). My live run `h3` (ct-only, key=action,
α=0.3, no SGD) gave whole-run top-1 0.207 vs 0.122 for no learning and crossed over at chunk 91
in the `smooth` block — but it shared the GPU with another job, so I mark those balanced numbers
**provisional pending a clean re-run**. What is not provisional: adding the **ratio** head to the
table destroys the gain (`h6`, top-1 0.120 ≈ no-learning 0.122), independently reproducing
F3/M6. The adversary's offline simulation of the same mechanism shows the property that matters
most here — it *improves with run length* (first-half 0.083 → second-half 0.032), which trunk SGD
never does.

**B4. Do not simply widen the cost gate.** `f4` (train every chunk) is the worst configuration
measured: regret 0.1572 → 0.2624, top-1 0.407 → 0.207, bytes +3.7 %. The reason is B-blocker 1,
not "learning is bad": widening a trigger that selects OOD and load-inflated chunks trains on
more of exactly the wrong data.

**B5. If none of B1-B3 is acceptable, the honest fallback is a static codec.** Clio already has
the escape hatch (`neuropress_static_lib_` + `neuropress_static_shuffle_ = 4`,
`compressor_runtime.cc:434-448`). Round-1 scoping: this is decisive for the **ratio-only**
objective (`zstd`+shuffle equals the per-chunk oracle to 4 dp on this corpus — though so would
any decent selector, since the objective is degenerate here) and much weaker for the balanced
one **in distribution**, where the model's gap to a constant is 1.23x, not 4.3x, and the model
wins on bytes by 8.8 % (F5). Under GPU load the gap narrows further.

## Section D — Reproduction appendix

All harness sources live in the **gitignored** parity dir
`/home/cc/clio-core/context-transport-primitives/test/unit/compress/model/parity/`; all
analysis scripts and data live under
`/tmp/claude-1000/-home-cc-clio-core-context-transfer-engine-compressor-results-explore-lossless-3chunk/d11d119b-5ef8-47f7-8c95-20ae441a8787/scratchpad/npreview/investigator/`
(referred to below as `$INV`).

### Harnesses added (all untracked / gitignored)

| path | what it does |
|---|---|
| `parity/np_eval.cu` | restored from the prior session's `repo_backup/np_eval.cu`, plus two 2-line fixes so it builds against the current tree (`DebugLogVar()` and the `bias_tbl`/`bias_upd` I6 leftovers were removed/stubbed). One CSV row per (chunk, action) with predictions **and** real codec measurements for every action. |
| `parity/np_sgd.cu` | **new.** Mechanism probes E0-E11: baseline predictions, SGD update-direction invariance (E1/E1b/E1c), fixed-sample convergence (E2), cross-action interference (E3), amplitude sweep (E4), chunk-size sweep (E5), error-bound sweep (E6), single-feature sensitivity (E7), decomp-head convergence (E8), non-finite features (E9), integer-data feature path (E10), lossless psnr label (E11). |
| `parity/np_ood_upstream.cu` | **new.** Runs **upstream** `gpucompress::runNNInference` (compiled from `/home/cc/NeuroPress/src/nn/nn_gpu.cu`) beside Clio's predictor over the amplitude / size / eb sweeps and diffs winner, per-action values and the 32-order. Registered as a DISABLED ctest so it never runs in the suite. |
| `parity/CMakeLists.txt` | targets `np_eval_exec` (from the prior session's `parity_cmake_np_eval_target.patch`), `np_sgd_exec`, `np_ood_upstream_exec` appended. |

### Build

```
cd /home/cc/clio-core/build && cmake . && make np_eval_exec np_sgd_exec np_ood_upstream_exec -j16
```

### Runs (all data under `$INV/data/`)

```
# batch 1 -- baselines, replicate, lossy, three learning configs   ($INV/scripts/runs1.sh)
NPE_MODE=infer NPE_COST=balanced NPE_EB=0    NPE_REGIMES=all NPE_CHUNKS_PER_REGIME=20 \
  NPE_DRIFT_CHUNKS=40 NPE_SIZE_CHUNKS=30 NPE_SEED=1234 NPE_RUN_ID=F0a \
  NPE_OUT=$INV/data/f0a_infer_eb0.csv       ./bin/np_eval_exec     # 2m46s, 590 chunks
#   f0b: NPE_SEED=4321  (independent data)   f0c: NPE_SEED=1234 (timing-noise replicate)
#   f1 : NPE_EB=1e-3 NPE_CHUNKS_PER_REGIME=15 NPE_DRIFT_CHUNKS=30 NPE_SIZE_CHUNKS=20
#   f2 : NPE_MODE=learn NPE_SGD_THRESH=0.30
#   f3 : NPE_MODE=learn NPE_COST=ratio NPE_SGD_THRESH=0.30
#   f4 : NPE_MODE=learn NPE_SGD_THRESH=-1        (train every chunk)
# batch 2 -- feature transforms                                    ($INV/scripts/runs2.sh)
#   g1 : NPE_FEATURE_XFORM=clamp  eb=0      g2 : NPE_FEATURE_XFORM=range  eb=0
#   g3 : NPE_FEATURE_XFORM=clamp  eb=1e-3
./bin/np_sgd_exec          > $INV/data/np_sgd.out            # E0-E11, ~20 s
./bin/np_ood_upstream_exec > $INV/data/np_ood_upstream.out   # upstream cross-check, ~3 s
ctest -R ctp_neuropress --output-on-failure                  # 17/17 pass, 16.97 s
```

### Analysis scripts (`$INV/scripts/`)

| script | claim it backs | invocation |
|---|---|---|
| `an.py` | per-head MAPE/bias, first-cut counterfactuals | `an.py CSV` |
| `an2.py` | head substitutions + per-regime regret | `an2.py CSV balanced "" perreg` |
| `an3.py` | early cap/floor ablation (superseded by `an4.py`) | `an3.py CSV [balanced\|ratio]` |
| `an4.py` | **F4, F14** — cap/floor ablation using the kernel's real clamp chain on `pred_*_raw` | `an4.py CSV balanced 0.0` |
| `oracle.py` | **F5, F8** — model vs oracle vs best constant, regret and bytes | `oracle.py CSV [balanced\|ratio] [gtfloor]` |
| `oracle2.py` | **F5** — best constant fitted on one CSV, scored on another; `indist` restricts to the 15 regimes with mad ≤ 0.5 | `oracle2.py TRAIN TEST balanced 1.0 [indist]` |
| `spread.py` | **F3** — within-chunk Spearman and max/min spread | `spread.py CSV` |
| `noise.py` | noise section — paired diff between two same-seed runs | `noise.py CSV_A CSV_B` |
| `explore_sim.py` | **F6** — offline exploration+adoption over K | `explore_sim.py CSV [balanced\|ratio] [gtfloor]` |

The per-regime OOD table (F1), the gate cross-tab (F9), the per-algorithm bias table (F3), the
cap-tie analysis (F4), the PSNR analysis (F11), the shuffle-direction analysis (F12), the
chunk-size table (F10) and the dt-pinning table (F15) were produced by short inline
`python3 - <<'PY'` snippets over the same CSVs; each reads only the columns named in the claim
and is reproducible in a few lines from the CSV header
(`run_id,regime,chunk_idx,chunk_bytes,eb,entropy,mad,d2,data_min,data_max,action,algo,quant,
shuffle,pred_ct,pred_dt,pred_ratio,pred_psnr,pred_ct_raw,...,act_ct,act_dt,act_ratio,act_psnr,
act_comp_bytes,...,cost_mape,ratio_mape,z_*,h1_*,ok,...`).

### Added in debate round 1

Runs (`$INV/scripts/runs3.sh`, `runs3b.sh`) — a 3-block stream
`turb_x50 → smooth_a1_n1e-2 → lammps_pos_sorted`, 150 chunks each, seed 1234, eb=0, balanced,
each starting from fresh `model.nnwt`:

```
h0_infer           NPE_MODE=infer                                          # baseline      [idle GPU]
h1_learn           NPE_MODE=learn NPE_SGD_THRESH=0.30                      # shipped phase 1 [idle]
h2_learn_explore7  NPE_MODE=learn NPE_SGD_THRESH=0.30 NPE_EXPLORE_K=7 \
                   NPE_EXPLORE_THRESH=0.30 NPE_EXPLORE_TRAIN=1             # phase 1+2      [idle]
h3_bias_ct         NPE_MODE=infer NPE_BIAS=1 NPE_BIAS_HEADS=ct \
                   NPE_BIAS_KEY=action NPE_BIAS_ALPHA=0.3                  # residual table [CONTENDED]
h5_infer_clamp     NPE_MODE=infer NPE_FEATURE_XFORM=clamp                  #                [CONTENDED]
h6_bias_ct_ratio   NPE_MODE=infer NPE_BIAS=1 NPE_BIAS_HEADS=ct,ratio ...   #                [CONTENDED]
```

`h3`/`h5`/`h6` ran from 09:04 while another GPU job was active, so their **balanced** numbers
are not comparable with `h0`'s and are marked provisional wherever quoted; their ratio-only and
byte columns are deterministic and unaffected. (`h4`, `h7` were killed mid-run and are not used.)

New probes in `parity/np_sgd.cu`: **E1d** (batched 7-sample update, magnitude sensitivity) and
**E1e** (single sample with ratio/psnr labels set to the model's own predictions, so only the ct
head carries error — the noise-gate / linear / saturated regimes of F7).

New analysis scripts in `$INV/scripts/`:

| script | claim it backs |
|---|---|
| `adapt.py` | F8.7 — per-chunk regret over time, learn vs no-learn, whole run and by quartile |
| `adapt_block.py` | F8.7 — per-regime-block crossover chunk |
| `debate1.py` | F8.3 (gate vs `n_x_oob`), F8.4 (gate rate vs device-load factor `k`), F8.2 (weight drift), F3/M6 (per-chunk ratio offset), F5 (strict `.nnwt` bounds), F2 (bandwidth sweep), F11 (psnr denominators) |
| `debate2.py` | F3 (out-of-sample per-key bias correction, DOF null), noise section (regret tail statistics) |

### Reused (not re-measured) material

Only two things: the prior session's `np_eval.cu` **source** (rebuilt and re-run here; no prior
CSV is used for any number above) and the prior session's retrain headline in C2, explicitly
labelled as reused. Every number in Sections A, B and D is from this session.
