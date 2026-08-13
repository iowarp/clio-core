# NeuroPress integration — verification record

Cross-examination of upstream NeuroPress (`/home/cc/NeuroPress`, git `b23b8f6`) against
Clio's integration of it. This directory holds **two efforts with different scopes**, and
knowing which one you are reading matters:

1. **The GPU/host execution-locality audit** (`00-` … `10-`, `SUMMARY.md`) — 68 findings
   answering one question:

   > Where does native NeuroPress do work **on the GPU** — entropy, MAD, second
   > derivative, inference, preprocessing, codec — while the Clio integration does that
   > same work **on the host**, or falls back to the CPU silently?

   Deliberately out of scope: naming, style, refactoring, and format differences with no
   host-vs-device execution consequence.

2. **The equivalence investigation** (`NEUROPRESS_CLIO_*.md`) — 29 phases asking the
   broader question the audit did not: *is the port byte-for-byte and
   decision-for-decision the same as upstream?* It re-derives rather than imports the
   audit's conclusions, because the port changed after the audit was written and the
   audit's port-side line citations are stale.

## Contents

**Start with `SUMMARY.md`** for the locality audit, or
`NEUROPRESS_CLIO_EQUIVALENCE_REPORT.md` for the equivalence verdict.

| File | Scope | Findings |
|---|---|---|
| `NEUROPRESS_CLIO_EQUIVALENCE_REPORT.md` | **equivalence verdict — the 31 protocol questions answered** | — |
| `NEUROPRESS_CLIO_INVESTIGATION_STATE.md` | source of truth for the investigation: phase status, known differences, per-phase checkpoints | D7-1 … D29-1 |
| `NEUROPRESS_CLIO_TRACEABILITY.md` | functionality → file map across both trees | — |
| `SUMMARY.md` | **consolidated locality verdict after verification — read this first** | — |
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

Executable checks, as opposed to documents, live outside this directory:
`context-transfer-engine/compressor/example/neuropress_gpu_chunk_equivalence/` (the
callback-trace harness — see "Running the checks" below) and
`context-transport-primitives/test/unit/compress/model/parity/` (the differential parity
suite that compiles upstream's kernels alongside the port).

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

**Second counterweight, added later and important:** that one-kernel result is a property
of *how the shipped test supplies memory*, not of a missing implementation. When the chunk
is genuinely device-resident, the device path is complete and it is the one taken. The
callback-trace harness (below) measures this directly — across 8 GPU-resident chunks it
records **zero CPU fallback on either side**, every data stage launching kernels, and no
production D→H of the payload at all: the largest production transfer anywhere is 1152 B
against a 4 MiB chunk. So the correct reading of the audit's headline is the one recorded
as **D20-1** — the default caller hands the compressor host memory, and that is what makes
the host paths reachable. Both statements are true and they are about different things.

## Running the checks

### The callback-trace equivalence harness

`examples/neuropress_gpu_chunk_equivalence/` runs **native NeuroPress and Clio over the
same GPU-resident chunk** and compares them **callback by callback**, reporting the
**first** point of divergence rather than only whether the final bytes match. It is the
one check here that asserts execution locality as well as numbers: CUPTI records every
`cudaMemcpy` and kernel launch both implementations make, so "did this stage run on the
GPU" and "did anything sneak a D→H copy in" are observed, not inferred.

```bash
cd build

# via ctest
ctest -R cte_neuropress_callback_trace_equivalence --output-on-failure

# or directly, with control over the run
./bin/neuropress_gpu_chunk_equivalence --out ./npeq                        # default mixed matrix
./bin/neuropress_gpu_chunk_equivalence --out ./npeq --error-bound 0.01     # one absolute bound everywhere
./bin/neuropress_gpu_chunk_equivalence --out ./npeq --error-bound-rel 0.005 # bound sized per chunk; drives quantization
```

| flag | env | effect |
|---|---|---|
| `--out DIR` | `NPEQ_OUT_DIR` | where traces and the report are written (default `.`) |
| `--error-bound EB` | `NPEQ_ERROR_BOUND` | one absolute bound on every chunk |
| `--error-bound-rel R` | `NPEQ_ERROR_BOUND_REL` | absolute bound sized as `R × that chunk's range` (`0.005` → 8-bit, `0.001` → 16-bit) |
| `--no-e2e` | `NPEQ_E2E=0` | skip the `AsyncDynamicSchedule` cross-check; much faster, does not start the runtime |
| `--no-self-check` | `NPEQ_SELF_CHECK=0` | skip running each side twice for reproducibility |

Exit `0` all chunks passed, `2` at least one failed, `77` no GPU (ctest reports SKIP).
Outputs: `callback_trace_report.txt` (human-readable walk, per-callback PASS/FAIL, first
divergence, final matrix), `native_trace.json`, `clio_trace.json`,
`callback_comparison.json`, `e2e_selection.csv`.

```bash
grep -E "^OVERALL" build/npeq/callback_trace_report.txt
sed -n '/FINAL CALLBACK TRACE MATRIX/,$p' build/npeq/callback_trace_report.txt
grep -A 3 "FIRST DIVERGENCE" build/npeq/callback_trace_report.txt
```

NeuroPress is **not** a Clio build dependency, so the target skips itself when the native
library or CUPTI is absent. To enable it, build upstream first:

```bash
cmake -S ~/NeuroPress -B ~/np-build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=80 -DNVCOMP_PREFIX=<prefix with include/nvcomp.hpp and lib/libnvcomp.so>
cmake --build ~/np-build --target gpucompress -j"$(nproc)"

cmake -S . -B build -DCLIO_NEUROPRESS_SRC_DIR=$HOME/NeuroPress \
                    -DCLIO_NEUROPRESS_BUILD_DIR=$HOME/np-build
cmake --build build --target neuropress_gpu_chunk_equivalence -j"$(nproc)"
```

Configure prints `NeuroPress callback-trace equivalence: enabled (…)` when the
dependencies are found, and a `skipped (…)` line naming what is missing when they are not.
See that directory's `README.md` for why the harness is shaped the way it is — in
particular, that native has **no callback API**, so its own stage entry points and its own
per-chunk diagnostics record are used as the callback boundaries rather than a parallel
mechanism being invented.

### The differential parity suite

```bash
ctest -R "neuropress|compressor" --output-on-failure
```

`ctest -R ctp_neuropress` (6 tests) compiles upstream's own `nn_gpu.cu`, `stats_kernel.cu`
and `entropy_kernel.cu` and diffs numbers against the port. It passes 6/6 and did so
before this audit.

Those tests assert **numerical** agreement, not **execution locality**. A port that
computes upstream's exact entropy value in a host loop passes them. So the parity suite
neither confirms nor refutes anything in the locality audit — and a finding claiming
*different numbers* on a path those tests cover would contradict a passing test and should
be treated with suspicion.

The callback-trace harness above is the one that closes that gap, which is why it is worth
running alongside rather than instead of the parity suite.
