# NeuroPress in-memory adaptation — implementation

Branch `neuropress-693-continued`, working tree only (nothing committed, nothing staged).
Every number below comes from a file in `$NPADAPT/impl/data`, produced by `$NPADAPT/impl/probe.sh`.

## Tracked files changed

| file | what |
|---|---|
| `context-transport-primitives/src/compress/model/neuropress_nn_gpu_kernels.cu` | `AdaptOptions` :73, `Opts()` :156, device `x_mins/x_maxs` :95 + upload :373-430, `ClampInput` :538, 5 clamp sites :673 :714 :804 :981 :1618, `SgdForwardAndErrors` :1595, `AdaptiveSGDKernel` :2024, dispatch :2217, `NeuroPressGpuSgdCallCount` :2238 |
| `context-transport-primitives/include/clio_ctp/compress/model/neuropress_nn_gpu_kernels.h` | `NeuroPressGpuLoad` gains defaulted `x_mins/x_maxs`; declares `NeuroPressGpuSgdCallCount` |
| `context-transport-primitives/src/compress/model/neuropress_nn_predictor.cc` | passes `x_mins_/x_maxs_` to Load :150; `DebugSgdCallCount()` :666; corrected the "nothing consults these" comment :120 |
| `context-transport-primitives/include/clio_ctp/compress/model/neuropress_nn_predictor.h` | `int DebugSgdCallCount();` :342 |
| `context-transport-primitives/test/unit/compress/model/test_models.cc` | two new cases :515 and :546 |
| `context-transport-primitives/test/unit/compress/model/CMakeLists.txt` | registers `ctp_neuropress_adaptive_sgd` :37 |

Untracked/gitignored: `test/unit/compress/model/parity/np_adapt.cu` + its target in `parity/CMakeLists.txt`.
`compressor/example/CMakeLists.txt` shows in `git diff --stat` — that is **not mine**, it was already modified.

## Knob semantics (all parsed once per process, `Opts()`, :156)

| env | default | exact meaning |
|---|---|---|
| `CLIO_NEUROPRESS_NN_CLAMP_INPUTS` | `0` | any non-empty value other than `0` enables. Clamps **inputs 4..7 only** (size, entropy, mad, d2) to the `.nnwt`'s `x_mins/x_maxs` at all five standardization sites (both inference kernels, the device-stats kernel, the deferred decomp-head kernel, and both SGD kernels). Inputs 0-3 are categorical and never clamped. A v1 `.nnwt` (bounds ±1e30) makes it a no-op. |
| `CLIO_NEUROPRESS_SGD_RULE` | `upstream` | exactly the string `adaptive` selects `AdaptiveSGDKernel`; anything else runs the ported kernel unchanged. |
| `CLIO_NEUROPRESS_SGD_LR` | unset | overrides the predictor's `learning_rate_` when `> 0`. Applied to both rules (it is provably a no-op for `upstream`, see P1). |
| `CLIO_NEUROPRESS_SGD_HEAD_STEPS` | `32` | trunk (W1..b4) is updated only once `sgd_call_count >= HEAD_STEPS`. `-1` = never (head-only forever), `0` = from the first firing. Read **before** the count is incremented, so `HEAD_STEPS=32` means firings 0..31 are head-only. |
| `CLIO_NEUROPRESS_SGD_TRUNK_SCALE` | `0.1` | trunk step = `lr_t * TRUNK_SCALE * ema`; the head uses `lr_t * ema`. |
| `CLIO_NEUROPRESS_SGD_DECAY_TAU` | `100` | `lr_t = lr / sqrt(1 + sgd_call_count / tau)`; `<= 0` disables decay. |
| `CLIO_NEUROPRESS_SGD_MAX_STEP` | `0.02` | trust region on the **final** update vector: if `‖Δ‖ > MAX_STEP`, scale by `MAX_STEP/‖Δ‖`. Scale-down only; `<= 0` disables. |
| `CLIO_NEUROPRESS_SGD_GRAD_CLIP` | `1.0` | global gradient-norm clip, scale-down only, applied before the momentum EMA; `<= 0` disables. |
| `CLIO_NEUROPRESS_SGD_MOMENTUM` | `0.85` | `ema = m*ema + (1-m)*clipped_grad`. `0` = plain SGD. Uses the same per-thread EMA buffer as the ported rule. |

`adaptive` keeps, unchanged: the ±0.5 error clamp, the 0.10 ct noise gate, the Kendall `uw` weighting
(all of it lives in `SgdForwardAndErrors`, shared by both kernels), the ±10 weight clamp, the
withheld W5 row 1 / b5[1], `sgd_call_count` semantics, and `DecompHeadSGDKernel`. It drops the
per-output unit-normalisation, the per-output 0.1 clip-renormalise, PCGrad, the global unit
normalisation, the `0.08*mean|d5_raw|` trust step and the anti-flip damping.

## Default bit-identity

`cd build && ctest -R neuropress --output-on-failure` on an **idle GPU** → **exit 0, 100 % passed,
0 failed out of 30** (1 disabled), recorded in `data/ctest_default.txt`. That is the 17 ctp parity harnesses (which diff
against upstream NeuroPress at `/home/cc/NeuroPress`), 8 compressor NeuroPress tests, and the new
`ctp_neuropress_adaptive_sgd`. P4 below adds an in-process check: with `CLAMP_INPUTS=1`, all 8
codecs' predictions on an in-bounds chunk are **bit-identical** to `CLAMP_INPUTS=0`.

### A flaky parity test, and a wrong conclusion I had to retract

`ctp_neuropress_train_device_stats_parity` fails **nondeterministically under GPU contention** —
~70 % of runs when 7 other processes share the A100, 0 % when the GPU is idle. When it fails, ~7 000
of 13 576 weights differ and the *control* comparison shifts too, so it is the reference training
run, not the device-stats path, that comes out wrong.

I first attributed this to extracting the SGD forward/error phase into a plain `__device__` helper
and "fixed" it with `__forceinline__` (8/8 clean afterwards). **That was a confound** — the clean run
happened during a quiet period. A controlled interleaved A/B/C of three binaries built from the same
tree (HEAD, mine force-inlined, mine non-inlined) gives, per 12 rounds each:

| GPU load | HEAD | mine (forceinline) | mine (no inline) |
|---|---|---|---|
| 7 competing processes | 7/10 fail | 8/10 fail | — |
| idle | 12/12 pass | 12/12 pass | 12/12 pass |

So: **pre-existing contention flake, not caused by this change**, and `__forceinline__` fixes nothing.
It is kept only because it matches `NeuroPressForwardShared` above it and avoids an ABI call in a hot
single-block kernel; the comment at :1595 now says exactly that. Anyone re-running this suite should
do it on an idle GPU. The same applies to `ctp_neuropress_sgd_choreography_parity`, whose "200
training calls allocate nothing" check reads *global* free device memory and therefore attributes a
neighbouring process's allocations to Train(); it too fails at HEAD under load and passes idle.

## P1 — does the learning rate reach the weights? (`data/p1_lr.csv`)

One sample, one `Train()` from fresh weights, ratio label 3x low and ct label 2x high.

| rule | lr | ‖ΔW‖ | cos vs lr=0.01 | ‖ΔW‖ ratio vs lr=0.01 |
|---|---|---|---|---|
| upstream | 0.01 | 2.999988e-03 | 1.000000000 | 1.000000 |
| upstream | 0.03 | 2.999996e-03 | 1.000000000 | 1.000003 |
| upstream | 10 | 3.000001e-03 | 1.000000000 | 1.000004 |
| adaptive | 0.01 | 1.377572e-04 | 1.000000000 | 1.000000 |
| adaptive | 0.03 | 4.132813e-04 | 0.999999885 | **3.000069** |
| adaptive | 10 | 1.999999e-02 | 0.999999872 | 145.18 (= MAX_STEP 0.02, trust region binding) |

The BRIEF's diagnosis holds: a **1000x** change in `learning_rate` moves the shipped rule's step by
**4 parts per million** — not literally bit-identical (the rate scales the accumulator *before* a
float normalisation, so it perturbs rounding), but a no-op in every sense that matters. Under
`adaptive`, 0.01 → 0.03 scales ‖ΔW‖ by exactly 3.000, and lr=10 saturates the trust region at 0.02.

## P2 — 200 firings on one sample (`data/p2_converge.csv`)

Same sample as P1; `predict → train → predict` each step. `cross` = times the ratio prediction
crossed the label; `over` = worst overshoot as a fraction of the label; `flip` = steps with
cos(ΔW_t, ΔW_{t-1}) < 0; `dW_q1/q4` = mean ‖ΔW‖ over steps 0-49 / 150-199.

| rule | HEAD_STEPS | lr | ratio err @0 / @50 / @200 | ct err @0 / @200 | cross | over | flip | dW_q1 → dW_q4 |
|---|---|---|---|---|---|---|---|---|
| adaptive | -1 | 0.003 | 2.000 / 1.759 / 1.210 | 0.500 / 0.379 | **0** | **0.000** | **0** | 2.18e-04 → 1.67e-04 |
| adaptive | -1 | 0.01 | 2.000 / 1.265 / **0.0072** | 0.500 / 0.062 | **0** | **0.000** | **0** | 7.26e-04 → 5.44e-04 |
| adaptive | -1 | 0.03 | 2.000 / 0.261 / 0.0000 | 0.500 / 0.015 | 16 | 0.042 | 22 | 2.17e-03 → 2.23e-07 |
| adaptive | 32 | 0.003 | 2.000 / 0.690 / 0.0000 | 0.500 / 0.049 | 16 | 0.102 | 15 | 2.58e-04 → 2.87e-07 |
| adaptive | 32 | 0.01 | 2.000 / 0.172 / 0.0000 | 0.500 / 0.013 | 32 | 0.201 | 34 | 8.22e-04 → 6.02e-08 |
| adaptive | 32 | 0.03 | 2.000 / 0.206 / 0.0000 | 0.500 / 0.036 | 60 | 0.448 | 77 | 1.96e-03 → 1.92e-07 |
| adaptive | 0 | 0.003 | 2.000 / 0.082 / 0.0000 | 0.500 / 0.047 | 16 | 0.109 | 16 | 3.43e-04 → 3.27e-08 |
| adaptive | 0 | 0.01 | 2.000 / 0.006 / 0.0000 | 0.500 / 0.029 | 32 | 0.206 | 39 | 6.37e-04 → 4.76e-09 |
| adaptive | 0 | 0.03 | 2.000 / 0.024 / 0.0130 | 0.500 / 0.036 | 89 | 0.339 | 85 | 1.04e-03 → 2.47e-04 |
| upstream | (n/a) | 0.003 | 2.000 / 0.0018 / 0.0017 | 0.500 / 0.044 | **65** | **0.517** | **62** | 8.02e-04 → 1.03e-05 |
| upstream | (n/a) | 0.01 | identical to lr 0.003 to 4 d.p. | | 65 | 0.517 | 62 | |
| upstream | (n/a) | 0.03 | identical to lr 0.003 to 4 d.p. | | 65 | 0.517 | 62 | |

Findings:
- **Monotone, no-overshoot convergence exists and is reachable**: `adaptive`, HEAD_STEPS=-1, lr ≤ 0.01
  gives 0 crossings, 0 overshoot, 0 direction flips over 200 steps. lr=0.01 lands the ratio at 0.7 %
  error; lr=0.003 is too slow (still 1.21 after 200).
- **The upstream control oscillates as the BRIEF says**: 65 crossings, 52 % worst overshoot, 62
  direction flips, and its three lr rows are identical to 4 decimals.
- **Contradiction with the BRIEF's expectation #1**: `‖ΔW‖ decaying with DECAY_TAU` is *not* what the
  head-only runs do — theirs grows or holds flat over the first ~100 steps. Cause: the ±0.5 error
  clamp is a Huber region, so while `|d5| > 0.5` the gradient magnitude is constant and the momentum
  EMA is still ramping (`1 - 0.85^t`), which outruns the `1/sqrt(1+t/100)` decay. ‖ΔW‖ collapses by
  3-4 orders of magnitude only once the error enters the linear region. Every config that converged
  shows the q1→q4 collapse; the two that had not converged by step 200 do not.
- **Contradiction with the BRIEF's expectation #2**: unfreezing the trunk makes convergence *worse*,
  not better. HEAD_STEPS=0 or 32 reaches zero ratio error faster but overshoots 10-45 % and flips
  direction 15-85 times; HEAD_STEPS=-1 is the only setting with no overshoot at all. On this evidence
  the useful default for the trunk is `-1`, not `32`. The 32 default is kept as the BRIEF specifies.

## P3 — one codec's outcome moves all eight (`data/p3_collateral.csv`)

200 firings on **lz4 only** (algo 0), then all 8 codecs re-predicted on the same chunk, against fresh
weights. `lr=0.01`. Range over the seven *untrained* codecs:

| rule | HEAD_STEPS | ratio move, algos 1-7 | ct move, algos 1-7 | trained algo 0 ratio move |
|---|---|---|---|---|
| adaptive | -1 | **-65 % … -93 %** (all down) | +79 % … +112 % | -66.6 % |
| adaptive | 32 | -24 % … -56 % (all down) | +27 % … +87 % | -66.7 % |
| upstream | 32 | -27 % … -55 %, **and +10 % … +18 % on algos 5-7** | +4 % … +85 % | -66.7 % |

**This is the clearest place the BRIEF's diagnosis is wrong.** Freezing the trunk does **not**
localise the update — it makes the collateral *worse* (all seven codecs move 65-93 % instead of
24-56 %). The reason is structural: the ratio head is a single row of W5 shared by every candidate,
and the eight candidates differ in exactly one of eight inputs, so their `h4` activations are nearly
parallel. Moving W5 row 2 shifts all eight predictions together by construction; the trunk is the
*only* capacity that could tell them apart. Finding #11's "one codec moves all 8" is a property of
on-policy SGD with a batch of one on a shared network, and no setting of `HEAD_STEPS` removes it.
(Upstream is not better here, just incoherent: it moves algos 5-7 the *opposite* way.)

## P4 — the input clamp (`data/p4_clamp.csv`)

Chunk: 4 MiB, entropy 4.964, d2 0.01159, mad swept. Shipped `.nnwt` bounds for the four continuous
inputs: size [65536, 4194304], entropy [0, 7.5298], mad [0, 0.4999780058860779], d2 [0, 1.00573].

- **In-bounds is bit-identical.** For every mad ≤ x_max and all 8 codecs, `CLAMP_INPUTS=1` reproduces
  `CLAMP_INPUTS=0` exactly (0 of 24 rows differ, compared as printed decimals at 9 d.p.).
- **Out-of-bounds predicts as if at the bound.** For mad ∈ {0.6, 5.0, 57.0}, `CLAMP_INPUTS=1` gives
  results **exactly equal**, for all 8 codecs, to `CLAMP_INPUTS=0` at mad = 0.4999780058860779.
- The size bound is 4 MiB exactly, so the 4 MiB probe chunk sits on it and the clamp is a no-op there.
- What it prevents, on this chunk (algo 0, clamp=0): ratio 27.8 → 100.0 (the cap) and ct 1.69 → 1.00
  (the floor) as mad goes 0.5 → 5, i.e. the saturation the review measured as gap #1.

## Unit tests added (`test_models.cc`)

- `:515` "The shipped SGD rule's step size does not depend on the learning rate" — runs in the default
  `ctp_compress_model`. Trains once at lr 0.01 and once at lr 10 from fresh weights and requires
  `cos > 1 - 1e-6` and `|‖ΔW₁₀‖/‖ΔW₀.₀₁‖ - 1| < 1e-4`. Deliberately **not** a bit-identity assertion:
  P1 shows the two differ at the 7th significant digit, and claiming bitwise equality would be false.
- `:546` "CLIO_NEUROPRESS_SGD_RULE=adaptive converges one sample without overshooting" — tagged
  `[.np_adaptive]` (hidden), registered as its own ctest `ctp_neuropress_adaptive_sgd` with
  `ENVIRONMENT "CLIO_NEUROPRESS_SGD_RULE=adaptive"`, because the knobs are parsed once per process
  and cannot be flipped inside a running binary. 50 firings at the default knobs: ratio error monotone
  (± 1e-6) and no crossing of the label for the 30 steps while the trunk is frozen, final error below
  half the initial, `DebugSgdCallCount() == 50`. Both are GPU-gated with the neighbours' `SKIP` idiom.

## New API

One accessor, `NeuroPressNNPredictor::DebugSgdCallCount()` (`gpu::NeuroPressGpuSgdCallCount`), read-only
and host-synchronizing. `DebugWeights()/DebugBiases()` cannot distinguish "the rule fired and moved
nothing" from "the gate never fired", and the unit test needs that distinction.

## Addendum — soft clamp

`CLIO_NEUROPRESS_NN_CLAMP_INPUTS=soft` selects `NeuroPressClamp::mode = 2`; `CLIO_NEUROPRESS_NN_CLAMP_SOFT_K`
(default 0.25) sets the scale as a fraction of the `.nnwt` range. `ClampInput()` now takes a
`NeuroPressClamp` by value (mode 0 = off, 1 = hard, 2 = soft) at the same five sites; past the
bound the input is `bound ± s·log1p(|x − bound|/s)` with `s = k·(max − min)`. Default env still
mode 0, and the ctp parity/compress-model ctests pass 20/20 after the change.

## Addendum — clamp removed

The input clamp (hard and soft, `CLIO_NEUROPRESS_NN_CLAMP_INPUTS`, `_SOFT_K`, the `NeuroPressClamp`
struct, `ClampInput()`, the device `x_mins/x_maxs` fields and their upload) was removed from the
tree after measurement, at the user's direction: inputs reach standardization raw, exactly as
upstream. `x_mins_/x_maxs_` are still parsed on the host so a v2 `.nnwt` round-trips through
`Save()`, but nothing reads them. `NeuroPressGpuLoad()` is back to its original signature. The
adaptive SGD rule and its knobs are unchanged. Parity/compress-model ctests pass after the removal.
