# NeuroPress integration — GPU/host execution-locality audit

Cross-examination of upstream NeuroPress (`/home/cc/NeuroPress`, git `b23b8f6`) against
Clio's integration of it, function by function, looking for one specific class of
divergence:

> Where does native NeuroPress do work **on the GPU** — entropy, MAD, second
> derivative, inference, preprocessing, codec — while the Clio integration does that
> same work **on the host**, or falls back to the CPU silently?

Deliberately out of scope: naming, style, refactoring, and format differences with no
host-vs-device execution consequence.

## Contents

**Start with `SUMMARY.md`** — the consolidated post-verification verdict.

| File | Scope | Findings |
|---|---|---|
| `SUMMARY.md` | **consolidated verdict after verification — read this first** | — |
| `00-kernel-inventory.md` | `__global__` inventory of both trees + `cuobjdump` of the shipped binary; structural leads | — |
| `01-findings-stats.md` | entropy, MAD, second derivative, min/max/mean | STATS-1 … STATS-8 |
| `02-findings-nn.md` | NN inference, SGD/REINFORCE, weights, decomp head | NN-1 … NN-10 |
| `03-findings-preprocess.md` | quantize / dequantize / byte-shuffle / unshuffle | PRE-1 … PRE-14 |
| `04-findings-orchestration.md` | top-level compress/decompress flow, public API, exploration, diagnostics | ORCH-1 … ORCH-16 |
| `05-findings-codec.md` | nvcomp + cuSZ/cuSZp/nDzip, codec factory, blob header | CODEC-1 … CODEC-10 |
| `06-findings-hdf5-vol.md` | HDF5 VOL connector — how data reaches the compressor | VOL-1 … VOL-10 |
| `07-runtime-profile-evidence.md` | nsys profile of the shipped tests: which kernels actually launch | — |
| `08-review-stats-nn.md` | adversarial verification of STATS + NN | verdicts |
| `09-review-preprocess-codec.md` | adversarial verification of PRE + CODEC | verdicts |
| `10-review-orchestration-vol.md` | adversarial verification of ORCH + VOL | verdicts |

**68 findings** total, produced by six scope-partitioned agents reading both trees
function by function, then put through an adversarial verification pass whose default
posture was that every finding is wrong.

**Outcome: 3 struck, 29 confirmed, 36 confirmed-with-correction; 12 high-severity after
review, down from 25 as filed.** The `0X-findings-*.md` documents are preserved **as
originally filed** — the reviews correct them rather than rewriting them, so the
corrections stay auditable. Where a finding document and its review disagree, **the review
wins.**

> **Line numbers drift.** Another session edited this working tree throughout the audit;
> port-side citations run 27-100 lines stale and two NN files gained ~30 lines mid-audit.
> Locate findings by content. Native citations (`b23b8f6`) were checked and are accurate.

## Method

Each finding carries: the claim, `file:line` evidence with quoted snippets from **both**
sides, a category, a severity, an explicit reachability statement (the call path and
gating condition), and a **"How to falsify"** field naming the experiment that would
prove it wrong. The falsification field is not decoration — it is the first thing the
verification pass attempts.

Categories used:
- `host-compute` — native runs it in a kernel, the port runs it in host C++
- `silent-cpu-fallback` — the port quietly substitutes a CPU path without telling the caller
- `added-host-roundtrip` — a D2H/H2D copy or stream sync the port introduces and native lacks
- `device-struct-became-host` — device-resident state upstream became host-resident here
- `missing-gpu-work` — native GPU work with no port counterpart
- `unreachable-device-path` — the port has a faithful device implementation that nothing calls
- `loud-error-became-silent-degrade` — native returns an error code; the port degrades and reports success

## Ground rules this audit was held to

1. **Comments are claims, not evidence.** The port is heavily annotated with parity
   assertions citing upstream line numbers. Every such citation was checked against the
   named upstream line. Most proved accurate; the exceptions are recorded as findings or
   as explicit non-findings.
2. **Running beats reading.** Source review alone has historically produced both false
   alarms and misses on this code. `07-runtime-profile-evidence.md` records what actually
   executes; the codec scope additionally compiled probes against the real installed
   nvcomp/cuSZ/cuSZp.
3. **Non-findings are recorded too.** Each file ends with the subcategories that were
   checked and found to genuinely match. A cleared item is a result.

## The single most important empirical result

On the shipped `compressor_dynamic_neuropress` test, the complete list of Clio-owned
CUDA kernels that launch is **one**: `InferKernel`. Not launched: `StatsPass1Kernel`,
`StatsPass2Kernel`, `EntropyFromHistKernel`, `FinalizeFeatureStatsKernel`,
`MinMaxKernel`, `ShuffleKernel`, `QuantizeKernel`, `InferKernelDeviceStats`, `SGDKernel`.

Entropy, MAD and the second derivative were therefore computed **on the CPU** on that
path, and inference took the host-stats variant rather than the device-stats one. See
`07-runtime-profile-evidence.md` for method and caveats.

Counterweight, so the picture stays honest: the codecs themselves genuinely run on the
GPU. nvcomp LZ4, Zstd and Bitcomp all executed as device kernels in that same run, and a
standalone probe confirmed all eight of NeuroPress's trained algorithms resolve to GPU
wire ids. The hypothesis that a ranked "LZ4" silently becomes CPU LZ4 was tested and is
**false**.

## Relationship to the existing parity suite

`ctest -R ctp_neuropress` (6 tests) compiles upstream's own `nn_gpu.cu`, `stats_kernel.cu`
and `entropy_kernel.cu` and diffs numbers against the port. It passes 6/6 and did so
before this audit.

Those tests assert **numerical** agreement, not **execution locality**. A port that
computes upstream's exact entropy value in a host loop passes them. So the parity suite
neither confirms nor refutes anything here — and a finding claiming *different numbers*
on a path those tests cover would contradict a passing test and should be treated with
suspicion.
