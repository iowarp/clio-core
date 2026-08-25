# In-memory adaptation from fresh weights — first step, measured on LAMMPS

**Question.** Without retraining `model.nnwt`, make the in-process learner do what it is meant to:
after a chunk is predicted and measured, if its cost-MAPE exceeds the gate, adjust the weights in
memory so the model adapts *gradually* to an unseen workload as the simulation evolves, *without
overshooting*. Follow-up to `05_agreed_findings.md`; implementation in `06a`, measurements in `06b`.
Scope: **live LAMMPS simulations through the shipping runtime** (`neuropress_lammps_lib`, runtime in
the same process, `run.sh --learn`), SGD firings recovered from the runtime's debug log. An offline
LAMMPS replay, a synthetic drift stream and a Gray-Scott stream were also run and were removed from
consideration at the user's request; `06b` retains the replay-based overshoot / collateral /
held-out measurements as supporting evidence only.

## Why the shipped loop cannot do it

`SGDKernel` (`neuropress_nn_gpu_kernels.cu`) turns every update into a **fixed-size step in a sign
direction**: per-output deltas are unit-normalised, the per-output clip *re*normalises to 0.1, the
whole gradient is normalised to unit norm, and the trust step saturates at 0.02 for any chunk that
passed the gate. Measured consequences:

* `neuropress_learning_rate` is a no-op — for the trunk **and** the head: ‖ΔW‖ = 2.9998e-3 at lr 0.001
  and 3.0000e-3 at lr 10, cos = 1.000000 (`06a` P1, `06b` §4, two independent probes).
* It cannot settle: 16 % of its firings on LAMMPS **overshoot** the ratio head (sign flips and the
  error grows on the very chunk it just trained on); only 31 % of firings improve it (`06b` §6).
* 94 % of its firings land on out-of-bounds inputs: it fits extrapolations, not the workload.
* One firing moves every codec: held-out predictions move **40x** (ratio) / **300x** (ct), top-1
  agreement with fresh weights 1.00 → 0.20 (`06b` §7).

## What was changed (env-gated; default bit-identical, `ctest -R neuropress` green, upstream parity 0.000e+00)

| knob | default | change |
|---|---|---|
| *(input clamp — evaluated, then removed)* | — | a hard clamp of inputs 4..7 to the `.nnwt` `x_mins/x_maxs` and a soft (log-compressed) variant were implemented and measured (tables below) and then **removed from the tree** at the user's direction: inputs reach the network raw, as upstream. The "hard clamp" / "soft clamp" arms below document what bounding the inputs did; they are not reproducible from the current source |
| `CLIO_NEUROPRESS_SGD_RULE=adaptive` | `upstream` | true backprop, global norm clip (scale-down only), momentum EMA, `lr_t = lr/√(1+t/τ)`, per-firing ‖ΔW‖ cap 0.02, trunk step × `TRUNK_SCALE` (0.1) after `HEAD_STEPS` firings. Keeps the ±0.5 Huber clamp, the 0.10 ct dead-band, Kendall weighting, the withheld decomp-head row, the ±10 weight clamp |
| `CLIO_NEUROPRESS_SGD_LR`, `_HEAD_STEPS`, `_TRUNK_SCALE`, `_DECAY_TAU`, `_MAX_STEP`, `_GRAD_CLIP`, `_MOMENTUM` | see `06a` | the rate is now effective: 0.01 → 0.03 scales ‖ΔW‖ by exactly 3.000 |

Files: `neuropress_nn_gpu_kernels.{cu,h}`, `neuropress_nn_predictor.{cc,h}`, tracked test
`ctp_neuropress_adaptive_sgd` (`test_models.cc`). Parity ctests green after the clamp removal.

## Workloads

| | blobs | what |
|---|---|---|
| live melt | 603 | box 20 (32 000 atoms), T = 3.0, 2000 steps, a frame every 10, x/v/f per frame; ≈ 30 s wall per run |
| live heating ramp | 603 | same box from a cold fcc lattice (T 0.02) heated to T 4.0 with a Nosé–Hoover thermostat during the run |

Note the data: velocity and force byte-entropy stays at 7.34–7.43 bits for every frame, position
goes 5.4 → 7.07 in the first frames and stays; the ramp only moves position entropy 5.42 → 6.87 →
7.01. An equilibrated LJ melt is stationary and float64 mantissas are noise to a byte codec, so this
workload tests "does the learner stay safe and converge on hard, stationary data", not "does it
track an evolving distribution".

## Results

**Live melt**, ratio-MAPE of the picked codec per 100-blob window (debug re-runs, firings recovered):

| arm | blobs 0–9 | windows | median | firings |
|---|---|---|---|---|
| fresh, no learning | 85x | 88x … 90x | 89x | 0 |
| hard clamp only, no learning | 0.196 | 0.150 flat | 0.150 | 0 |
| shipped rule, gate 30 % | 12.5x | 0.90 · 0.73 · 0.90 · 0.90 · 0.31 · 0.33 · 0.90 | 0.90 | 48 |
| adaptive, clamp OFF, lr 0.01 | 85x | 41x · 0.62 · 0.57 · 0.53 · 0.59 · 0.59 · 0.67 | — | 404 |
| adaptive, clamp OFF, lr 0.1, gate 20 % | 64x | 0.83 · 0.71 · 0.52 · 0.52 · 0.49 · 0.47 · 0.32 | — | 98 |
| hard clamp + adaptive lr 0.01 | 0.194 | 0.146 · 0.120 · 0.107 · 0.093 · 0.093 · 0.093 · **0.084** | 0.113 | 14 |
| hard clamp + adaptive lr 0.1, gate 20 % | 0.179 | 0.106 · 0.106 · 0.080 · 0.038 · 0.041 · 0.045 · **0.048** | 0.080 | 104 |
| hard clamp + adaptive lr 0.2, gate 20 % | 0.072 | 0.045 · 0.142 · 0.109 · 0.459 · 0.460 · 0.429 · 0.105 | 0.168 | 75 |
| adaptive, clamp OFF, lr 0.2, gate 20 % | 7.8x | 0.90 · 0.90 · 0.86 · 0.90 · 0.90 · 0.51 · 6.3x | 0.90 | 323 |
| soft clamp k 0.1, no learning | 0.090 | 0.092 · 0.095 · 0.097 · 0.099 · 0.101 · 0.104 · 0.105 | **0.098** | 0 |
| soft clamp k 0.1 + adaptive lr 0.01 | 0.080 | 0.082 · 0.093 · 0.097 · 0.112 · 0.119 · 0.121 · 0.121 | 0.102 | 19 |
| soft clamp k 0.1 + adaptive lr 0.1, gate 20 % | 0.091 | 0.135 · 0.086 · 0.122 · 0.122 · 0.190 · 0.190 · 0.190 | 0.134 | 30 |
| soft clamp k 0.1 + adaptive lr 0.2, gate 20 % | 0.164 | 0.074 · 0.061 · 0.300 · 0.057 · 0.072 · 0.165 · 0.177 | 0.089 | 102 |
| soft clamp k 0.25, no learning | 0.870 | 0.864 → 0.867 | 0.87 | 0 |
| soft clamp k 0.25 + adaptive lr 0.1 | — | 0.54 · 0.20 · 0.45 · 0.25 · 0.17 · 0.15 · 0.21 | 0.22 | 29 |
| soft clamp k 0.25 + adaptive lr 0.2 | 0.283 | 0.163 · 0.163 · 0.163 · 0.163 · 0.163 · 0.121 · 0.064 | 0.163 | 8 |

Heating ramp: shipped 0.20 → … → 5.1x at the end (177 firings); hard clamp + adaptive lr 0.1
0.21 → 0.20 → 0.08 → 0.02 → 0.03 → 0.12 → 0.11 (20 firings); soft k 0.1 + adaptive lr 0.1 0.14 → 0.11 →
0.12 → 0.17 → 0.17 → 0.16 → 0.14 (27 firings); soft k 0.25 + adaptive lr 0.1 0.24 → 0.14 → 0.17 → 0.04 →
0.08 → 0.46 → 0.53 (40 firings); at lr 0.2: hard clamp 0.06 → 0.10 → 0.19 → 0.08 → 0.07 → 0.34 → 0.34 (45),
soft k 0.1 0.28 → 0.09 → 0.17 → 0.16 → 0.13 → 0.11 → 0.25 (53).

Supporting evidence from the (excluded) offline replay, `06b` §6–§10: overshoot per firing shipped
0.157 vs adaptive **0.000**; held-out collateral shipped 40x / 300x (top-1 agreement 0.20) vs
adaptive ≈ 1.3x (1.00); learning frozen on the last 300 chunks, cost-MAPE 8.76 (fresh) → 0.23 (adapted).

## Verdict

* **The weights do self-update from measured errors, gradually and without runaway.** With the
  clamp off, the adaptive rule alone drives the predicted ratio from 100x down to 47 → 5.5 → 1.3
  over 300 blobs (lr 0.01) or to 1.27 within 50 blobs (lr 0.1); the shipped rule on the same inputs
  never leaves 90 % error. But learning on unbounded inputs is coarse: 98–404 updates, oscillation
  (1.27 → 4.2 → 0.42 → 0.84), plateau at 30–60 % error.
* **On LAMMPS the input clamp is the fix, not the learning.** Hard clamp with zero SGD: 8.79 → 0.0039
  offline, 85x → 0.15 live from blob 0. That is prior knowledge (high entropy → ratio ≈ 1)
  becoming usable once the inputs are in range, not adaptation. Learning on top refines 0.15 → 0.05–0.08
  live (14–104 firings, 0 overshoots).
* **The shipped rule is harmful here** (live: flat at 90 % error for all 603 blobs, 48 firings that
  never settle; 5.1x at the end of the ramp); the adaptive rule is safe (0 overshoot, ≈ 1.3x
  collateral in the replay probes) and converges (firings stop after first contact with each field;
  ‖W−W₀‖ plateaus).
* **Soft clamp: k decides everything.** k 0.25 leaves the inputs 8–13 σ out, the prior extrapolates
  again (0.87 with no learning) and the learner only gets part of it back (0.22). k 0.1 lands closer
  (4–8 σ) where the prior happens to be *better* than the hard clamp's boundary value — 0.098 with no
  learning vs 0.150 — but learning on top of it makes it slightly worse (0.10–0.13), and on the ramp
  it is flat at 0.13 while the hard clamp reaches 0.02–0.03. Neither k makes learning more useful
  on this data; the best live result remains hard clamp + adaptive lr 0.1 (0.048).
* **lr 0.1 / gate 20 % is the best live setting** (0.106 → 0.048); lr 0.01 / gate 30 % converges
  slower (0.146 → 0.084) with far fewer firings (14 vs 104). **lr 0.2 is past the stable range**:
  with the hard clamp it swings 0.045 → 0.46 → 0.105 (median 0.168, twice lr 0.1's), the ramp ends
  at 0.34, and with the clamp off it diverges (0.90 flat, 6.3x at the end, 323 firings). At lr 0.2
  the 0.02 per-firing cap binds on most updates, which turns the rule back into a fixed-size step —
  the shipped rule's failure mode.
* **Not shown by this workload**: adaptation to an evolving distribution. LAMMPS raw float64 arrays are
  stationary after melting; a field-type output (density/velocity grids) or single-precision,
  precision-limited dumps would be needed to test that.

## Final arm set, both cost models (live, ratio-MAPE of the picked codec: first → last 100-blob window, median, firings, tier bytes ratio)

Arms as requested: fresh weights; adaptive lr 0.2 / gate 20 % with hard clamp, soft k 0.1, soft k 0.25,
clamp off; adaptive lr 0.3 / gate 30 % with clamp on and off; the original rule at lr 0.2 / 20 % and
lr 0.3 / 30 % (its lr is a no-op, so those two differ only by gate). Balanced cost =
ct + dt + bytes/(ratio·5 GB/s); ratio cost = bytes/(ratio·5 GB/s) only (`CLIO_NEUROPRESS_COST_W_CT=W_DT=0`).

| arm | melt · balanced | ramp · balanced | melt · ratio cost | ramp · ratio cost |
|---|---|---|---|---|
| fresh, no learning | 88x flat · 1.035 | 0.31 → **86x** (4.4) · 1.043 | 99x flat · 1.017 | 0.50 → **99x** (8.8) · 1.037 |
| adaptive lr 0.2 g 20 %, hard clamp | 0.045 → 0.46 → 0.105 (0.168, 75) · 1.033 | 0.06 → 0.34 (0.125, 45) · 1.034 | **0.025 flat** (0.025, 5) · **1.041** | 0.11 → 0.10 (0.093, 34) · **1.054** |
| adaptive lr 0.2, soft k 0.1 | 0.074 → 0.177 (0.089, 102) · 1.028 | 0.28 → 0.25 (0.158, 53) · 1.034 | 0.11 → 0.056 (0.056, 30) · 1.040 | 0.14 → 0.066 (0.106, 31) · 1.034 |
| adaptive lr 0.2, soft k 0.25 | 0.163 → 0.064 (0.163, 8) · 1.039 | 0.40 → 0.055 (0.192, 96) · 1.023 | 0.22 → 0.142 (0.142, 32) · 1.043 | 0.13 → 0.087 (0.115, 24) · 1.040 |
| adaptive lr 0.2, clamp off | 0.90 → **6.3x** (0.90, 323) · 1.023 | 0.24 → 0.23, spike 0.90 (0.385, 123) · 1.038 | 0.90 → 0.22 (0.90, 509) · 1.038 | 0.40 → **3.5x** (0.205, 311) · 1.014 |
| adaptive lr 0.3 g 30 %, hard clamp | 0.61 flat (0.61, 6) · 1.040 | 0.17 → 0.115 (0.194, 44) · 1.037 | 0.098 flat (0.098, 5) · 1.020 | 0.08 → 0.025 (**0.070**, 8) · **1.053** |
| adaptive lr 0.3, clamp off | 0.90 flat (0.90, 94) · 1.016 | 0.35 → 0.62 (0.614, 96) · 1.036 | 0.90 → 0.071 (0.151, 167) · 1.036 | 0.45 → 0.24 (0.248, 200) · 1.009 |
| original rule, gate 20 % | 0.87 → 0.75 (0.853, 366) · 1.018 | 0.20 → 0.34 (0.328, 214) · 1.034 | 0.90 flat (0.90, 488) · 1.007 | 0.17 → 0.25 (0.168, 264) · 1.030 |
| original rule, gate 30 % | 0.90 flat (0.90, 269) · 1.017 | 0.20 → **1.8x** (0.559, 149) · 1.030 | 0.90 → 0.116 (0.90, 397) · 1.011 | 0.22 → 0.90 (0.217, 115) · 1.036 |

* **Ratio cost model is where in-memory learning pays.** The gate then sees the ratio alone, the
  hard-clamped adaptive rule settles at 2.5 % (lr 0.2) / 9.8 % (lr 0.3) error within the first
  window with 5 firings, and — for the first time — better predictions become fewer bytes: tier
  ratio 1.041 vs 1.017 fresh on the melt, 1.053–1.054 vs 1.037 on the ramp. The original rule fires
  400–500 times on the melt and stays at 90 %.
* **Under the balanced cost, lr 0.2–0.3 is too hot for the adaptive rule**: the 0.02 per-firing
  cap binds on most updates and the rule degrades toward a fixed-size step (0.045 → 0.46 swings;
  lr 0.3 stuck at 0.61 after 6 firings). The stable balanced setting remains lr 0.1 / gate 20 %.
* **Clamp off never converges** at these rates — 94–509 firings, 0.90 medians, 3.5–6.3x blow-ups.
* **The heating ramp is the one genuinely evolving case** (fresh weights go from 0.3 to 86–99x as
  the crystal melts): every learning arm holds the error at 0.06–0.6 through the transition, the
  hard-clamped adaptive rule best (0.07–0.19 median).

## A visible learning curve

A steady descent needs a real initial error (the hard clamp starts at 15–20 %, so there is little
to descend), small decaying steps, and a gate that keeps firing until the error is under it. Live
melt, balanced cost, soft clamp k 0.25, per-50-blob medians:

| arm | curve | firings |
|---|---|---|
| adaptive lr 0.01, gate 30 % | 0.85 · 0.78 · 0.79 · 0.64 · 0.64 · 0.58 · 0.58 · 0.50 · 0.50 · 0.46 · 0.36 · 0.36 · 0.36 — near-linear, not yet at the floor by blob 600 | 18 |
| adaptive lr 0.03, gate 20 %, τ 100 | 0.79 · 0.19 · 0.12 · 0.16 · 0.12 · 0.09 · 0.11 · 0.07 · 0.08 · 0.05 · 0.05 · 0.06 · 0.05 — fast first drop, then a smooth descent to the 5 % floor | 97 |
| adaptive lr 0.03, gate 20 %, no decay | 0.66 · 0.32 · 0.26 · 0.26 · 0.13 · 0.12 · 0.10 · 0.05 · 0.29 · 0.29 · 0.16 · 0.22 · 0.27 — decays then re-oscillates without the 1/√t schedule | 134 |

Note this is a demonstration of the weights learning from errors, not the most accurate arm: the
hard-clamped rule is lower at every point in time (0.15 → 0.05) because it starts from the prior
instead of learning it.

## Recommended settings and next step

With the clamp removed, the shipping options are the "clamp off" arms above:
`CLIO_NEUROPRESS_SGD_RULE=adaptive CLIO_NEUROPRESS_SGD_HEAD_STEPS=0` with `SGD_LR` 0.1–0.3 and the gate
at 0.20–0.30 — under the ratio cost model (`CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0`)
the learner brings a 99x starting error down to 7–22 % within the run (lr 0.3 / gate 30 %: 0.90 →
0.071 on the melt, 167 firings; tier bytes 1.036 vs 1.017 fresh). Under the balanced cost it does not
converge at these rates (0.90 medians, 3.5–6.3x blow-ups) — the balanced gate keys on compress-time
error that the 1 ms floor and kernel-time noise dominate. Next: the adaptive input normaliser
(running per-input mean/std initialised from the `.nnwt`, EMA-tracked from the stream) as the
non-clamping way to keep inputs well-conditioned; decoupled, schedule-driven exploration feeding
the learner; compose keys for the knobs. Next: promote the knobs to
compose keys; decouple exploration from the error gate (schedule/novelty-triggered K-way sweeps
feeding the learner) so ranking can be learned off-policy; a per-codec capacity the shared trunk
lacks (finding #11) if accuracy beyond the clamp is wanted on this class of data.
