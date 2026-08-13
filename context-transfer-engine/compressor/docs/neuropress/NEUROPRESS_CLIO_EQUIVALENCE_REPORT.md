# NeuroPress ↔ CLIO equivalence report

**Final deliverable of the 28-phase differential investigation (issue #693).**

| | |
|---|---|
| Native baseline | `/home/cc/NeuroPress` @ `b23b8f6` |
| Port baseline | `/home/cc/clio-core` @ `neuropress-693-continued`, HEAD `9bcba093` |
| Full evidence | [`NEUROPRESS_CLIO_INVESTIGATION_STATE.md`](NEUROPRESS_CLIO_INVESTIGATION_STATE.md) (28 phase checkpoints) |
| Requirement matrix | [`NEUROPRESS_CLIO_TRACEABILITY.md`](NEUROPRESS_CLIO_TRACEABILITY.md) |
| Date | 2026-08-13 |

---

## Verdict

**EQUIVALENT IN DECISIONS AND IN RECONSTRUCTED DATA. NOT BYTE-IDENTICAL IN COMPRESSED
PAYLOAD. DIVERGENT — BY DESIGN — IN INTEGRATION ARCHITECTURE.**

Everything that determines *what a chunk becomes* — statistics, neural-network input,
prediction, ranking, model selection, shuffle decision, shuffled bytes, quantization
decision, quantization parameters, quantized bytes, SGD, weights — is equivalent, and
tighter than "equivalent" usually means. Inference is **bit-identical** (max relative
error exactly 0). Shuffle, quantize and dequantize are **byte-identical**. The `.nnwt`
file is the same file by md5. A 1 GiB round-trip has **zero** differing bytes.

Those results were not read out of the port's comments. They come from test binaries
that link upstream's own `.cu` sources and diff against them, re-run in this session.

The divergences are in the plumbing, and one of them is intrinsic:

- **Compressed payload bytes differ on 209 of 256 chunks** while every selection is
  identical (D18-1). Root cause found: **nvcomp's ANS encoder is history-dependent**
  (D18-2), so byte-identical payloads were never an achievable requirement.
- **CLIO deliberately does not link native NeuroPress.** NeuroPress's logic is
  integrated *into* CLIO, so container format, chunk unit, and the HDF5 write path are
  CLIO's, not upstream's. Four such divergences were reviewed and accepted as WONTFIX
  with CLIO's architecture explicitly preserved.

---

## The 31 questions

Legend — **PASS-E** exactly equivalent · **PASS-A** expected CLIO architectural
difference · **PASS-N** numerically equivalent · **DIFF** difference found ·
**PARTIAL** measured but not fully attributed.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Does CLIO pass the same GPU-resident chunk to NeuroPress? | **PASS-E** | Phase 19; `neuropress_gpu_direct` launches all 8 CLIO device kernels on an IPC-opened device pointer |
| 2 | Are the chunk bytes identical? | **PASS-E** | Phase 19/23; CLIO metadata never enters the buffer handed across the boundary |
| 3 | Are callback sequences identical? | **PASS-A** | Phase 6/7; CLIO has no callback registry — the same stages run in the same order, dispatched statically |
| 4 | Are statistics identical? | **PASS-N** | Phase 5; ≤ **5.4e-15** rel over 256 × 4 MiB chunks. Bit-exactness is impossible in principle — native uses `atomicAdd(double)` and is not bit-reproducible against itself |
| 5 | Are per-chunk diagnostics equivalent? | **DIFF (D6-1, WONTFIX)** | Phase 6; not ported. Native's only consumers are its own tests |
| 6 | Is NN input byte-identical? | **PASS-E** | Phase 11; same 8 features, same order, same normalization |
| 7 | Is NN prediction identical? | **PASS-E** | Phase 11; **max relative error = 0**, bit-identical |
| 8 | Is model selection identical? | **PASS-E** | Phase 18; **256/256** selections identical vs native on the 1 GiB e2e |
| 9 | Is ranking identical? | **PASS-E** | Phase 9; 1530 + 251 checks incl. tie-break order. D9-1 float↔double bounded by measurement |
| 10 | Is static selection identical? | **PASS-E** | Phase 7; 0/256 divergent. D7-1's host-path claim was **corrected — it was overstated** |
| 11 | Is inference identical? | **PASS-E** | Phase 11 |
| 12 | Is inference + exploration identical? | **PASS-E** | Phases 8/9/11; D8-1 (clock) **fixed** in `66c86af5` |
| 13 | Is inference learning identical? | **PASS-N** | Phase 13; weights diffed every step, worst **2.98e-08** (~1 ULP) |
| 14 | Is runtime SGD behavior equivalent? | **PASS-E** | Phase 13; multi-flow EMA scoping reproduces native's own divergence exactly (0.00198627 both sides) |
| 15 | Is shuffle selection identical? | **PASS-E** | Phase 3 |
| 16 | If selected, are shuffled bytes identical? | **PASS-E** | Phase 3; 3 element sizes × 30 buffer sizes, incl. CLIO inverting native's own blob |
| 17 | Is quantization selection identical? | **PASS-E** | Phase 4 |
| 18 | Are quantization parameters identical? | **PASS-E** | Phase 4; `scale`/`min`/`max` exact across 5 regimes × 5 bounds |
| 19 | Are quantized bytes identical? | **PASS-E** | Phase 4; packed bytes byte-equal |
| 20 | Are error bounds respected? | **PASS-E** | Phase 4/16; linear quantizer within bound. cuSZ mode (D16-1) **fixed** — default was `Rel`, native uses `Abs` |
| 21 | Is lossless compression equivalent? | **DIFF (D18-1)** | Phase 18; identical selections and identical payload *size*, differing payload *bytes* — see Q31 |
| 22 | Is lossless decompression equivalent? | **PASS-E** | Phase 15/23; **256/256 exact, 0 of 1,073,741,824 bytes differ** |
| 23 | Is static quantized decompression equivalent? | **PASS-E** | Phase 16; after the D16-1 fix |
| 24 | Does the read path reconstruct identical data? | **PASS-E** | Phase 17; same inversion order, same metadata-from-header rule, loud failure on mismatch |
| 25 | Extra D→H copies in CLIO? | **DIFF (VOL-1, accepted)** | Phase 21; native's RELEASE mode copies only compressed bytes, CLIO stages the full uncompressed dataset. Consequence of the accepted D21-1 decision |
| 26 | Extra H→D copies in CLIO? | **PARTIAL** | Phase 21; counts measured, **per-callsite attribution not done** |
| 27 | Any CPU fallback? | **DIFF (D20-1)** | Phase 20/22; host-resident chunks take host stats/shuffle/rank. Proven **not to change selections** (`compressor_residency_invariance`) |
| 28 | Does CUDA IPC preserve GPU residency? | **PASS-E** | Phase 19; D19-1 leak **fixed** in `66c86af5` via a new `DeregisterMemory` admin method |
| 29 | Does runtime learning modify `.nnwt`? | **PASS-E** | Phase 14; **no** — byte-identical file, md5 `df52a926af026fc617e172dcc990a395`, same as native |
| 30 | Which differences are legitimate architecture? | *see below* | D21-1, D21-2, D15-1, D6-1, D22-1, D23-1, VOL-1 |
| 31 | Which are actual functional divergence? | *see below* | D18-1 (intrinsic), D20-1, D22-2 |

---

## Q30 — Legitimate CLIO architectural differences

Each was reviewed against the constraint the user set — *CLIO's core architecture is
preserved* — and accepted.

| ID | Difference | Why accepted |
|---|---|---|
| **D21-1** | CLIO uses an out-of-band stamp/tag where native uses an H5Z filter (id 305) | Preserving CLIO's core architecture. Drives VOL-1 |
| **D21-2** | CLIO's chunk unit is a flat byte range; native's is an N-D HDF5 chunk | Preserving CLIO's chunking architecture |
| **D15-1** | Blob containers are disjoint; neither side parses the other's | CLIO does not link native NeuroPress — by design |
| **D6-1** | Per-chunk diagnostics struct not ported | Native's only consumers are its own tests |
| **D22-1** | No `best_mode` equivalent | `exploration_threshold_ = 0` achieves the main effect. Severity **downgraded MEDIUM→LOW** on inspection |
| **D23-1** | Concurrency knob differs | `runtime.num_threads` is the bound. The original *"no knob exists"* claim was **wrong** |
| **VOL-1** | CLIO stages the full uncompressed dataset H→D→H | Direct consequence of D21-1, which was accepted |

Confirmed as **not** gaps, because the upstream feature is dead code: `vmin`/`vmax`,
`gpucompress_enable/disable/active_learning_enabled` (no consumer in native `src/`),
`ct_mape_threshold` (native ignores it too), and native's BYPASS/TRACE VOL modes
(benchmark harnesses, set only by upstream's own scripts).

---

## Q31 — Actual functional divergence

**D18-1 — payload bytes differ on 209/256 chunks. Intrinsic; not fixable.**

Every selection is identical and every payload is the same *size*, but the bytes differ.
Phase 26 found the cause (**D18-2**): nvcomp's ANS encoder carries state across calls.

| Experiment | ANS hashes changed |
|---|---|
| `--flush-cache` | **189 / 191** |
| `--zero-output` | **0** |
| native vs native, same conditions | **0 / 256** (256/256 identical) |

ANS diverges unconditionally; cascaded diverges only with shuffle; **LZ4 + shuffle is
identical**. Byte-identical payloads are therefore not achievable against a
history-dependent third-party encoder — and it does not matter, because decompression is
exact on both sides (Q22).

**D20-1 — CPU fallback on host-resident chunks.** `nsys` over the shipped functional
test shows exactly **one** CLIO-owned kernel launched (`InferKernel`, the host-stats
variant); `StatsPass1/2`, `Shuffle`, `Quantize`, `RankKernel`, `SGDKernel` and
`DecompHeadSGDKernel` never launch. The device path is complete and correct — 8 kernels
on `neuropress_gpu_direct` — so this is a property of *which memory callers hand in*.

*Severity re-graded here.* The state doc rates D20-1 **HIGH**, but that grade was
justified solely by *"this is what makes D7-1 reachable"*, and **D7-1 was subsequently
corrected as overstated**: host-resident chunks *are* ranked by NeuroPress, via the same
`RankIntoStats` → `predictor.Rank()` path as device chunks. The new test
`compressor_residency_invariance` then proved directly that host- and device-resident
copies of identical bytes produce the **same selection**. What survives is a performance
and coverage concern, not a correctness one: **MEDIUM**.

**D22-2 (LOW)** — native's cost weights and bandwidth are runtime-settable; CLIO
hardcodes `kCostW0/W1/W2 = 1.0` and `kCostBandwidthBytesPerMs = 5e6`. Values agree at
the defaults, so nothing diverges today, but a caller that retunes the cost model is
silently ignored.

---

## What was fixed

Commit `66c86af5` — five divergences, each with a test:

| ID | Fix |
|---|---|
| **D19-1** | Device scratch never freed (379 `cudaMalloc` vs 73 `cudaFree`). Added `kDeregisterMemory = 36` admin method; cross-process backends now close the IPC handle before free, guarded by an `in_process` flag |
| **VOL-5** | SIGSEGV on the VOL cache-hit read path — now uses `DeviceAwareMemcpy`, scatter staged through host memory |
| **VOL-2** | VOL read into a device pointer — allocates a `kManagedUvm` backend when the destination is device memory |
| **D16-1** | `Cusz` defaulted to `Rel`; native uses `Abs`. Test was 1000× too loose to catch it (G5) |
| **D8-1** | Exploration cost used wall-clock where native uses kernel time — now CUDA-event kernel time on both sides |
| **D23-2** | Exploration SGD sampling unbounded — now cost-sorted and capped at 7 |

Commit `9bcba093` — dispositions, the Phase 25 measurement, and three corrections to my
own earlier findings.

---

## Performance (Phase 25)

Measured against a from-source NeuroPress build, pure codec kernel time via CUDA events
on both sides:

```
clio    n=512   median 0.2732 ms   p10 0.2279   p90 3.5022
native  n=256   median 0.3553 ms   p10 0.2202   p90 1.3578
median clio/native = 0.769
```

Two caveats belong with that number: CLIO's **p90 is 2.6× native's**, and CLIO
compresses every chunk twice, so *total* codec time is **976.3 ms vs 237.4 ms**. The
median is favourable; the aggregate is not.

---

## Corrections issued during the investigation

Four of my own findings were wrong or overstated and are corrected in the record. Each
came from reading code without driving it, and each was caught by running something.

| Finding | Correction |
|---|---|
| **D7-1** | Claimed CLIO silently substitutes its own model for host chunks. **Wrong as a general claim** — NeuroPress ranks host-resident chunks too. Narrowed to the `!features_ok` hand-off, which is silent on the host path only. Severity HIGH → LOW |
| **D22-1** | Severity MEDIUM → LOW; `exploration_threshold_ = 0` achieves the main effect |
| **D23-1** | The *"no knob exists"* claim was simply wrong — `runtime.num_threads` is the bound |
| **VOL-3** | Mechanism wrong. The stamp **is** persisted and compared; `clio_stamp_ambiguous` withholds it inside one mtime granule as a deliberate fail-closed guard |

Two measurement figures were also discarded as unsound: an 89/235 selection comparison
(two uncontrolled e2e runs, native had online learning on) and a 352/512 "regression"
(my own error — joined on async completion order instead of blob; rejoined correctly it
is 256/256).

---

## What remains open

| Item | Status |
|---|---|
| **Phase 21** — per-callsite H↔D transfer attribution | Counts measured; attribution not done. Mainly sharpens VOL-1, already accepted |
| **Task 27** — prediction MAPE on a 2-D/3-D dataset | The one follow-up D21-2 generated: the shipped `.nnwt` appears trained on N-D chunk features while CLIO feeds flat byte ranges. The 1 GiB e2e is 1-D and cannot show it |
| **G4** — phases 10-14 share three test binaries | Accepted risk; they link upstream's real sources |

Neither open item can change the verdict above. Task 27 could change how well the
predictor *performs* on N-D data — not whether CLIO and native agree.

---

## Note on test state

Three cross-process tests fail on this branch — `cr_gpu_ipc_devmem_register_cuda`,
`cte_devmem_putget_cross_process_cuda`, `cte_devmem_getblob_cross_process_cuda`.
Verified twice (once by stashing all changes) as **pre-existing**: they fail at
`server.WaitForReady()` before any code from this branch runs. Cause is four stale
`iowarp-distributed-node*` Docker containers holding runtimes from a timed-out test.
