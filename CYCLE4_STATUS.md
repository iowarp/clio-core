# Cycle 4: NeuroPress Online Learning Loop (issue #693)

Cycle 4 is the effort to port NeuroPress's **online learning loop** — the
mechanism where the compression-selection model doesn't just use static
pretrained weights, but continuously improves its predictions from real,
observed compression outcomes via SGD (stochastic gradient descent).

## Done (4a–4e, 4h) — the learning machinery

The math and compute side of online learning is built, GPU-accelerated,
and tested:

- **4a** — Extended `TrainingLabels` to include `decompression_time_ms`.
- **4b** — Core SGD: forward pass, backward pass, weight updates.
- **4c** — Uncertainty weighting (Kendall's learned log-variance), so the
  model weighs noisy signals less.
- **4d** — Gradient clipping + noise gating, stability safeguards against
  bad updates.
- **4e** — PCGrad-lite: resolves conflicting gradients when multiple
  objectives (ratio, speed, quality) pull in different directions.
- **4h** — Ported all of the above to real CUDA kernels, device-resident.

In short: given a training sample (features + actual outcome), the model
can correctly update its own weights.

## Not done (4f, 4g) — actually closing the loop

Nothing currently *uses* the learning machinery above. The loop is open,
not closed:

- **4f — Wire `Train()` into `compressor_runtime.cc` from real outcomes.**
  `Runtime::Compress` already records the *actual* observed outcome after
  a real compression runs (`context.actual_compression_ratio_`,
  `actual_compress_time_ms_`, etc. — these fields exist today), but
  nothing feeds them back into the model's `Train()`. The model predicts
  once per candidate and never learns whether it was right.

- **4g — K-way exploration policy + regret tracking.** Without this, the
  system always exploits the model's top-ranked candidate and never tries
  alternatives to gather fresh training signal, and there's no
  measurement of *regret* (how much worse the chosen candidate performed
  versus the best possible choice). Standard explore/exploit machinery
  for an online learner.

## Net effect

The model *can* learn — it just isn't being taught. Wiring 4f (and later
4g) is a separate feature track from the CUDA IPC / GPU-aware storage work
done earlier in this branch — no overlap, purely about closing the
predict → observe → train feedback loop.
