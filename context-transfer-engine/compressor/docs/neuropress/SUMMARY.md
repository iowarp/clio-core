# NeuroPress GPU/host locality audit — consolidated verdict

68 findings filed by six scope agents, then each one individually attacked by three
adversarial verifiers whose default posture was that the finding is wrong.

| | count |
|---|---|
| Filed | 68 |
| **REFUTED / struck** | **3** |
| CONFIRMED | 29 |
| CONFIRMED-WITH-CORRECTION | 36 |
| High severity after review | 12 (down from 25 as filed) |

Plus **two defects discovered during verification that fall outside the audit's frame**
and are not in any findings document — see "Found while verifying" below. One of them is
arguably more serious than most of what the audit set out to find.

---

## The answer to the question asked

> Where does native NeuroPress compute on the GPU while Clio does it on the host or
> falls back to CPU silently?

**Confirmed and reachable on shipped paths:**

| What | Native | Clio | Evidence |
|---|---|---|---|
| Entropy / MAD / 2nd derivative | GPU kernels always | **host loops** on the ordinary path | `compressor_dynamic_neuropress` launches *zero* stats kernels; 18-row nsys list |
| Byte shuffle | `byte_shuffle_kernel_specialized` | **host scalar loop**, 40/64 chunks | 40 gdb hits on `ByteShuffle`, 0 on `ByteShuffleDevice`; 1.204 ms vs 0.070 ms |
| Byte unshuffle | `byte_unshuffle_kernel_specialized` | **host scalar loop**, 40/64 chunks | 40 gdb hits, backtrace to `Runtime::Decompress:2929` |
| Decomp-head SGD | `nnBatchedDecompSGDKernel` on device | **no kernel exists**; scalar host C++ | 217 live `TrainDecompHead` calls; batch grows every read (O(n²)) |
| Decomp-head weight update | in-place, 65 floats, zero transfer | **full model round trip** | 9× 54,304 B H2D + 156× 54,304 B D2H to edit 65 floats |
| HDF5 `H5Dread` decompression | GPU, into the user's device buffer | **does not happen at all** | 0 decompress calls at default stamp granularity |
| Whole NN (CPU-only build) | no CPU path exists | full host network, `IsReady()` still true, **silent** | compiled and ran the real TU |

**Tested and found FALSE** — worth stating, because it was the most likely-sounding
hypothesis: the ranked NeuroPress selection does **not** silently resolve to CPU codecs.
All 8 trained algorithms map to GPU wire ids (`is_gpu=1`, verified by compiled probe),
nvcomp LZ4/Zstd/Bitcomp all run as real device kernels, and 0 of 72 real chunk selections
took the CPU fallback list.

---

## The 12 high-severity findings after review

| ID | Claim | Why it survived |
|---|---|---|
| **VOL-3** | After write→close→reopen, `H5Dread` decompresses nothing and serves the uncompressed HDF5 copy | Controlled single-variable experiment; default also **compresses the dataset twice** |
| **VOL-5** | Cache-served hyperslab read `memcpy`s into a possibly-device buffer | **Reproduced a SIGSEGV** at the exact cited line from a legal HDF5 sequence |
| **VOL-1** | Every device-buffer `H5Dwrite` pays a full-dataset D2H native never pays | Measured: a 64 MiB D2H attributable to `native_write_staging` |
| **VOL-2** | Decompressor never gets a device destination; device unshuffle unreachable from the VOL | Forced the cache-hit path: 32 decompressions, **0 `UnshuffleKernel`** |
| **ORCH-1** | Whole compress pipeline runs on host for a host-resident chunk | nsys: zero Clio stats/preprocess kernels in the shipped dynamic test |
| **STATS-5** | Host-resident chunk gets all three statistics from CPU loops, silently | Same profile; native has no host path at all (it errors) |
| **PRE-2** | Byte shuffle on host | 40 gdb hits; adds ~73% on top of the codec it feeds |
| **PRE-5** | Byte unshuffle on host | Read-side twin of PRE-2, same fraction, same cost |
| **NN-1** | Decomp-head SGD is a kernel natively, host C++ here | 217 live calls; native kernel confirmed non-dead |
| **NN-2** | Full model host↔device round trip per decomp-head update | Byte-exact CUPTI histogram |
| **NN-5** | CPU-only build silently runs the whole network on host | Compiled and ran it; **this is the default cmake configuration** |
| **PRE-14** | `DequantizeDevice` has no pointer-kind guard | Probe: sticky `cudaErrorIllegalAddress`, **process-wide GPU outage** (latent) |

---

## Struck outright

- **STATS-7** (native's `vmin`/`vmax` have no port counterpart) — `vmin`/`vmax` are **dead
  upstream too**; no production consumer anywhere in NeuroPress. Missing a port of dead
  code is not a gap. The finding's own concession refuted it.
- **PRE-10** (port adds a D2D copy after device unshuffle) — **backwards.** Native performs
  the same full-size D2D into the caller's buffer 36 lines below the snippet the finding
  quoted (`gpucompress_compress.cpp:1298-1302`). The port uses one *fewer* device buffer.
- **ORCH-2** (NN inference silently falls back to a CPU reimplementation) — the auditor read
  across an `#endif`. On a CUDA build `Load()` hard-fails and the host network **is not
  compiled into the binary**. This was the second-highest-severity claim in that set.

---

## Found while verifying — not in any findings document

These fell out of the verification and are outside the GPU/host locality frame the audit
used, which is exactly why nothing else in this directory mentions them.

1. **Device scratch is leaked.** `IpcManager::FreeGpuBackend` (`ipc_manager.cc:3947-3956`)
   never calls `cudaFree` — it only erases a registration map entry. Measured: 346
   `AllocateAndRegisterGpuBackend` calls in a 64-chunk run, **475 `cudaMalloc` against 119
   `cudaFree`**, and GPU memory climbing 1653 → 2635 MiB during the run. ~4-6 MiB per chunk
   on the device path. Both PRE-9 and CODEC-6 assumed this function frees and costed it as a
   synchronizing free; the truth is worse than what they filed.

2. **cuSZ runs in the wrong error-bound mode.** The port's `Cusz` defaults to **relative**
   (`cusz.h:87`), native uses **absolute** (`external_compressors.cu:118 rc.mode = Abs;`).
   Measured round-trip max error at `eb=1e-3` on ±100 data: **0.1–0.2, i.e. 100–200× the
   nominal bound.** Independently reproduced by two agents. Affects fidelity and ratio, not
   locality — but it matters to whoever owns lossy parity.

---

## Where the audit was weakest

- **Severity inflation on unmeasured perf claims.** STATS-2, STATS-3, STATS-8, PRE-6, PRE-7
  and NN-10 all assert serialization harm from null-stream launches or device-wide syncs.
  Measurement found **12 cross-stream overlapping kernel pairs out of 1279**, 13% GPU
  occupancy, and 155 `cudaDeviceSynchronize` calls totalling 2.96 ms against `cudaMalloc`'s
  106 ms. The divergences are real; the costs are unproven and possibly nil at current
  concurrency.
- **Two findings written against a tree that had already changed.** NN-3 and NN-4 claim the
  cost model, argmax and PSNR mask are host-side. A `RankKernel` doing all three **on the
  GPU**, with upstream's exact bitonic network, is in the working tree and the shipped `.so`
  — and was missing from `00-kernel-inventory.md`, which advertises itself as the
  cross-check for precisely this. (Cause: another session was editing the tree during the
  audit; `RankKernel` is not in git HEAD.)
- **Reachability graded from code shape rather than configuration.** ORCH-3/9/10/11 and
  CODEC-1/3 describe real code on paths that the shipped config never enters — the
  interposer is commented out of `clio_default.yaml`, `target_psnr_` is set nowhere in
  production, and a configured-but-unloadable NeuroPress aborts pool creation rather than
  degrading.

## Cross-cutting caveat: line numbers

Another session edited this working tree throughout the audit. Port-side citations drift
27–100 lines; `neuropress_nn_gpu_kernels.cu` and `neuropress_nn_predictor.cc` each gained
~30 lines mid-audit. **Locate findings by content, not line number.** All native
(`/home/cc/NeuroPress` @ b23b8f6) citations were checked and are accurate.

## Relationship to the parity suite

`ctest -R ctp_neuropress` passes 6/6, before and after. Those tests compile upstream's own
`.cu` files and assert **numerical** agreement — not execution locality. A port that
computes upstream's exact entropy in a host loop passes them. They neither confirm nor
refute anything here, and every verifier was instructed not to misuse them in either
direction.
