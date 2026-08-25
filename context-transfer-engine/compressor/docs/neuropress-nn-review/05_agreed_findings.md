# NeuroPress NN predictor in Clio — agreed findings and suggestions

**Scope.** The shipped `model.nnwt` as used by Clio's NeuroPress path, judged as **in-memory
adaptation from fresh weights** (every process loads the file, adapts via SGD, never persists —
`compressor_runtime.cc:376`, `:1465-1471`). Two independent Opus agents (investigator, adversary)
measured every claim on an A100 with real nvcomp codecs, 4 MiB float32 chunks, every candidate
action compressed for real (~300k measurements), then debated to agreement in two rounds
(`01`–`04`). Every number below was reproduced by both sides; balanced-cost numbers are idle-GPU,
common-ground-truth. Nothing here is a port bug: 17/17 parity ctests pass and Clio matches
upstream's own inference to 0.000e+00 on 17 sweep points (F17).

## How the prediction works

Eight inputs `[algo_id, quant, shuffle, error_bound, size, byte_entropy, mad, mean|d2|]`; the
last three are **absolute-magnitude** statistics over float32 elements, standardized with
training constants from data in [0,1] (`mad` mean 0.189, std 0.106). MLP 8→64×4→8, ReLU; heads
`expm1`-inverted then clamped: `ct,dt ≥ 1 ms`, `ratio ≤ 100`, `psnr ∈ [0,120]`. 32 actions =
8 codecs × quant × 4-byte shuffle, scored `ct + dt + bytes/(ratio·5 GB/s)`, ties → lowest action
index. Phase-1 SGD fires on the picked action when cost-MAPE > 0.30; phase-2 exploration is
**default-off**; `--best` is ratio-only and disables SGD.

## Agreed gaps and bottlenecks (ranked by decision impact)

| # | gap | deciding number | class |
|---|---|---|---|
| 1 | **Amplitude OOD.** `mad`/`d2` are absolute magnitudes; ordinary data at ±10..±1000 is hundreds of σ out and one or both heads saturate above mad≈40 | ratio spread over 8 codecs 46.4x → **1.00x** (mad 0.006 → 57); 3/28 regimes give one prediction for all actions; clamping inputs to the `.nnwt` bounds: balanced regret **0.157 → 0.040**, ratio-only bytes **−10.1 %** (held-out −11.5 %), bit-identical in-distribution | upstream design, faithfully ported |
| 2 | **ct head range 21x too narrow; per-codec error is a constant.** Upstream labelled the whole API call wall-clock; Clio measures kernel time | within-chunk spread measured **96.8x** vs predicted **4.5x**; model thinks bitcomp/ans (0.2 ms) are slowest; one per-codec log offset removes **59 %** of error (random null +1 %); true ct alone: regret **0.157 → 0.017** at 5 GB/s | deployment label mismatch |
| 3 | **Which head matters is set by storage bandwidth, not the model** | perfect ct: 0.017 @5 GB/s, **0.263 @50 MB/s (worse than nothing, 0.230)**; perfect ratio: 0.634 @5 GB/s, 0.116 @50 MB/s. `bw` is upstream's hard-coded 5e6 B/ms (`CLIO_NEUROPRESS_COST_BW`) | deployment constant |
| 4 | **Ratio head error is per-chunk (a ×3.45 level offset), not per-codec** — so a better ratio alone makes the balanced pick *worse* | per-chunk offset **−79 %**, per-codec **+3 %**; true ratio alone: regret 0.157 → **0.634** | upstream training data |
| 5 | **100x cap → tie-break by action index** | **43 %** of eb=1e-3 chunks tie; 38 % pick plain lossless lz4; uncapped ratio-only regret **35.99 → 0.435**, bytes −11.4 %; zero balanced effect; ~25 % of ties remain without the cap | upstream design (standing rule: keep 100) |
| 6 | **Shuffle half of the action space is chosen near chance** | shuffle helps ratio on 89 % of pairs; model predicts it on 43 % (recall **0.458**, held-out 0.333); exact-tie prediction on 33 % | upstream feature design |
| 7 | **Chunk size OOD at both ends** | median ratio APE 1.6 @1 MiB, **4.4 @0.25 MiB**, **46 @8 MiB**; regret 0 → 7.0 @16 MiB; 64 MiB predicts 34.5 dB PSNR for a lossless action | upstream training range |
| 8 | **PSNR head cannot rank and cannot represent its label** | actual PSNR is 1 distinct value per chunk on 440/440; head clamped [0,120] vs labels to **−283.6 dB** (544/7040 rows); `target_psnr` wrongly masks **35 %** of legal candidates | upstream design |
| 9 | **Non-float payloads read as float32 on both host and device paths, silently** | uint8 → mad **3e36**; NaN → 1/8 distinct predictions → always lz4, no log line | Clio deployment gap |
| 10 | **The phase-1 gate is an OOD/load detector, not an error detector** | **93 %** of firings on out-of-bounds chunks idle (base 40 %); gate rate **0.239 → 0.997** as device load scales 1x → 14x | upstream design + shared GPU |
| 11 | **Trunk SGD cannot adapt a fresh process usefully** — sign-directed, on-policy batch of one, one codec's outcome moves all 8 | cos(ΔW)=1.000000000 across error magnitudes; 590 steps move ‖W‖ **0.90 %** (less than 226 do); 10 steps on lz4 move every codec's ratio 68–94 %; harm scales with firing count: 0 → **0.137**, 41 → 0.137, 52 → 0.174, 177 → 0.304, 205 → 0.469 regret | upstream design |
| 12 | **1 ms cost floor is load-bearing** | truncates 29 % of ct / 60 % of dt idle; lowering to 0.1 ms → **+22 %** regret, −0.061 top-1 | upstream design |
| 13 | **Exploration is default-off, so what ships is the network's raw top-1** | offline regret K=0 **0.157**, K=3 0.071, K=31 ≈0.009; true best inside the K=3 window only 54 % | config |
| 14 | **A fixed action beats the model out of sample on the ratio objective** | `zstd+shuffle` bytes **0.588 = per-chunk oracle** vs model 0.672; in-distribution balanced gap only 1.23x and the model wins bytes by 8.8 % | consequence of 1–6 |

**Caveats both sides sign.** Balanced-cost numbers are idle-GPU (codec times inflate 2–30x under
contention); mean regret is a tail statistic (top 10 of 590 chunks carry 52.6 %), every direction
above survives on the median.

## Agreed suggestions

Two ladders: (a) better starting weights / inference inputs helps every chunk immediately;
(b) in-run adaptation must converge within a process. Fixes to the **ct** head (a4, b2) help only
at ≥ ~500 MB/s storage; below that the ratio-side fixes (a2, a3) are the valuable ones.

### (a) Starting point

| # | change | measured gain | cost / faithfulness |
|---|---|---|---|
| **a1** | **Retrain `model.nnwt`** with codec-kernel ct labels, sizes 0.25–64 MiB, amplitudes past mad 0.5 (file swap, no code) | whole headroom: regret 0.137 → 0.002, bytes 0.678 → 0.650 (true ct+ratio / oracle rows); prior session's retrain: top-1 0.46 → 0.73 | none — upstream's own recipe. Range-normalising `mad`/`d2` would diverge from upstream's feature code |
| **a2** | **Raise the ratio cap for `--best` only** (`CLIO_NEUROPRESS_RATIO_CAP`) | eb=1e-3 ratio-only regret **35.99 → 0.435**, bytes −11.4 %; lossless 0.55 → 0.23 | zero balanced effect; default stays 100 per standing rule |
| **a3** | **Clamp the 4 continuous inputs to the `.nnwt` `x_mins/x_maxs`** (5 standardization sites in `neuropress_nn_gpu_kernels.cu`) | ratio-only bytes −10.1 % / −34.7 % at eb=1e-3 / −11.5 % held-out (deterministic); balanced regret −75 % | balanced bytes **+1.9 % / +1.5 % / −1.7 %** (corpus-dependent); near-zero faithfulness cost — bounds already ship in the file |
| **a4** | **Static per-codec ct log-bias table** (never the ratio head) | out-of-sample balanced regret **−80 %** (0.163 → 0.033) | bytes +4.6 %, top-1 0.375 → 0.317, median chunk worse; harmful ≤ 500 MB/s → env- **and** bandwidth-gate it |
| **a5** | **Guard the feature path on both paths**: reject non-float payloads, non-finite/out-of-range stats → existing legacy fallback (`neuropress_selection.cc:177-182`) | silent always-lz4 → logged fallback | none |
| **a6** | **Keep the 1 ms floor** | lowering costs +22 % regret | faithful |

### (b) In-run adaptation from fresh weights

| # | change | measured gain | cost / faithfulness |
|---|---|---|---|
| **b1** | **Clamp the inputs (= a3): it makes the loop quiescent.** The gate fires on extrapolations; remove them and it stops | firings **51 → 3** and **74 → 0** (learn+clamp bit-identical to infer+clamp); regret 1.092 → 0.809 | as a3 |
| **b2** | **Learn magnitude outside the network: online per-codec log-residual table on the time heads, trunk untouched** (harness `NPE_BIAS`; runtime lacks it) | zero SGD calls; regret **0.138 → 0.091**, median 0.0138 → 0.0082; 1.092 → 1.015 on a second schedule — the only mechanism that beats no-learning on both | bytes +0.7–4 %, tail-carried (worse on 130/226 changed picks, gain in last quarter); **never add the ratio head** (worse than not adapting) |
| **b3** | **If trunk SGD stays on, feed it phase-2 exploration samples** (K=7) | only trunk-SGD arm below no-learning: **0.128 vs 0.137**; best first-half regret | worst top-1 (0.168 vs 0.361) and bytes (+3.2 %); crossover schedule-dependent (chunk 5–17 vs 560) |
| **b4** | **Trigger hygiene**: ratio-MAPE gate is load-invariant (0.475 at every load vs 0.239→0.997); add `n_x_oob == 0` precondition (0.163 at every load, 0 % OOD) | repairs replaced-ratio gate 0.469 → 0.174; on the cost gate cuts firings 41 → 1 | still not a net win (0.174 vs 0.137); shipped OR knob `CLIO_NEUROPRESS_SGD_ON_RATIO` = 0.304 (best median, worst mean) |
| **b5** | **Do not widen the cost gate** | train-every-chunk: regret +67 %, top-1 0.41 → 0.21 | — |
| **b6** | **Static-codec fallback for ratio-driven deployments** (`neuropress_static_lib_` + `neuropress_static_shuffle_ = 4`) | bytes 0.588 = oracle | exists today |

**Bottom line.** The starting weights are wrong for this hardware and data (amplitude, size, ct
label) and a1 is the only change that removes causes; a3/a2 are the best deterministic
inference-time wins. Nothing that updates the **trunk** in-process is a net win — what works
in-run is stopping the trigger from firing on extrapolations (b1) and learning the time-head
magnitudes in a table outside the network (b2).
