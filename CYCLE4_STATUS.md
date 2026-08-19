# Cycle 4: NeuroPress Online Learning Loop (issue #693)

Cycle 4 is the effort to port NeuroPress's **online learning loop** — the
mechanism where the compression-selection model doesn't just use static
pretrained weights, but continuously improves its predictions from real,
observed compression outcomes via SGD (stochastic gradient descent).

**Status: complete (4a–4h).** The loop is closed: the model predicts,
observes what actually happened, and trains on the difference.

## 4a–4e, 4h — the learning machinery

The math and compute side of online learning, GPU-accelerated and tested:

- **4a** — Extended `TrainingLabels` to include `decompression_time_ms`.
- **4b** — Core SGD: forward pass, backward pass, weight updates.
- **4c** — Uncertainty weighting (Kendall's learned log-variance), so the
  model weighs noisy signals less.
- **4d** — Gradient clipping + noise gating, stability safeguards against
  bad updates.
- **4e** — PCGrad-lite: resolves conflicting gradients when multiple
  objectives (ratio, speed, quality) pull in different directions.
- **4h** — Ported all of the above to real CUDA kernels, device-resident.

## 4f — SGD Phase 1: learn from the primary result

`DynamicSchedule` now trains on the real, just-measured outcome for the
algorithm that was actually used. Gated by a weighted-cost MAPE — the same
`cost = w0*ct + w1*dt + w2*size/(ratio*bw)` formula and default weights and
bandwidth as NeuroPress's `g_rank_w0/w1/w2` and
`g_measured_bw_bytes_per_ms` — against `neuropress_mape_threshold_`
(default `0.30`, matching `g_reinforce_mape_threshold`). Training fires only
when the prediction was wrong by more than that fraction, not on every
chunk.

## 4g — SGD Phase 2: K-way exploration + regret

When error crosses the higher `neuropress_exploration_threshold_` (default
`0.50`, mirrors `g_exploration_threshold`), up to K=3 alternative candidates
are compressed purely to generate more real-outcome training samples, and a
**regret** metric is logged (how much worse the chosen candidate was than
the best alternative found). The explored outputs are never stored — the
primary's already-persisted result stays authoritative.

Off by default, matching `g_exploration_enabled`. Runs the K candidates
sequentially through the existing `Compressor::Compress()` rather than
reimplementing the original's parallel CUDA streams — same data flow and
training outcome, and exploration is opt-in and off the storage critical
path either way.

## The `.nnwt` file is never written

Both phases are **in-memory only**. `Train()` adjusts the live weights for
the process's lifetime and never calls `Save()`, so `model.nnwt` on disk is
never modified — verified byte-identical across runs.

This matches NeuroPress itself: no runtime API there persists a trained
model either. Its only "save" functions
(`gpucompress_nn_save_snapshot`/`_restore_snapshot`) copy weights to a
caller-supplied buffer as a rollback mechanism, not to a file; only
`gpucompress_load_nn`/`_reload_nn` touch the `.nnwt` path, and both are
read-only.

Consequence: learning persists for the life of the `clio_run` daemon.
Restart it (or recreate the compressor pool) and it reloads the same cold
weights it started with. Cross-restart persistence would be new work beyond
what either project does today.

## Action space is restricted to NeuroPress's trained algorithms

Dynamic selection ranks **only** the 8 GPU-lossless nvcomp algorithms that
NeuroPress's `decodeAction` scheme hard-codes as "algorithm index 0-7":
LZ4, Snappy, Deflate, GDeflate, Zstd, ANS, Cascaded, Bitcomp.

No CPU library, and none of zfp-sycl/cuSZ/nDzip/cuSZp. None of those were in
that action space upstream either — their algorithm ids fall outside 0-7 and
are reachable only via explicit/static selection. Ranking them meant
`FeaturesTo8Input`'s `base_id % 8` fallback aliased each onto whichever
trained algorithm shared that remainder and returned *its* prediction as
though it were a learned opinion about the untrained one.

The candidate set is 24 (8 algorithms × 3 presets). Those four codecs remain
available through explicit/static selection — just not through the ranked,
dynamic path.
