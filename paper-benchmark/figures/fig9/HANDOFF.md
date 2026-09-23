# Figure 9 — state of play, 2026-09-23

Where the figure-9 work stands, what is measured, what is not, and what has to
happen before the campaign is re-run across all five workloads. Written as a
handoff: the next stretch of fixes happens on Chameleon, and the final
benchmark comes back to Delta.

---

## 1. What is finished and trustworthy

**Figure 9, all 16 Nyx arms.** `live/fig9.png`, seconds on the y-axis, 26.2 GiB
payload, ε = 1e-3, one run per arm.

| arm | non-I/O (s) | device I/O (s) | total (s) | ratio |
|---|---|---|---|---|
| Best fixed nvCOMP+Tier | 4.45 | 4.37 | **8.8** | 6.812 |
| NP+Tier+Async+Lossy | 4.12 | 4.88 | **9.0** | 6.413 |
| cuSZp3+Tier | 6.26 | 3.53 | **9.8** | 8.529 |
| cuSZp3 | 8.69 | 6.97 | **15.7** | 8.529 |
| NeuroPress | 8.41 | 8.92 | **17.3** | 6.435 |
| ndzip+Tier | 2.07 | 15.59 | **17.7** | 1.579 |
| NP+Tier+Async | 4.28 | 13.66 | **17.9** | 1.416 |
| Best fixed nvCOMP | 12.92 | 7.19 | **20.1** | 6.812 |
| nvCOMP+Tier | 5.21 | 15.71 | **20.9** | 1.327 |
| NP+Tier | 6.43 | 17.39 | **23.8** | 1.406 |
| ndzip | 7.34 | 29.98 | **37.3** | 1.579 |
| nvCOMP | 5.58 | 31.93 | **37.5** | 1.327 |
| NP only | 11.13 | 27.51 | **38.6** | 1.406 |
| Baseline | 4.32 | 37.63 | **42.0** | 1.000 |
| cuSZ+Tier | 66.25 | 2.63 | **68.9** | 9.938 |
| cuSZ | 66.44 | 8.65 | **75.1** | 9.938 |

Headline: NP+Tier+Async is 57.2% faster than Baseline and 14.2% faster than
nvCOMP+Tier; adding the bound takes it a further 49.8% (17.9 -> 9.0 s).

**The cuSZ manager-reuse fix.** cuSZ's own batch example (`batch_run.cc`)
builds ONE `psz_resource` and runs a whole file list through it. Against
upstream `e1c0135` that corrupts output, because `compressor.inl` zeroes the
outlier counter only in the `Spline` branch while every Lorenzo kernel claims
outlier slots with `atomicAdd` on that same counter. A fresh manager is correct
only because `malloc_device` memsets at allocation; nothing zeroes it again.

Fix: hoist that one memset above the predictor branch.
`paper-benchmark/patches/cusz-e1c0135-reset-outlier-counter.patch`, 2 lines of
code. Verified standalone on real Nyx (job 22323800): REUSE blow-ups 8/10 ->
0/10, 0/64 at scale, `max|err|` identical to fresh.

Measured end to end, same binary, same library, only the flag differing:

```
cuSZ+Tier  per-chunk manager   74.1 s   setup 67.1 s
cuSZ+Tier  reused manager       8.8 s   setup  0.1 s     8.4x, ratio identical
```

**The reuse guard.** `compress/cusz.h` reuses a manager per thread by default,
but proves it first: `ReuseSelfTest` compresses two different buffers through
one manager and compares `psz_header::splen`, the very device counter the
defect fails to reset. Verified three ways in one job (22324723):

```
A  patched cuSZ, auto    ENABLED   outliers 861737 == 861737   setup  7.01 ms
B  patched cuSZ, REUSE=0 pinned off                            setup 70.49 ms
C  STOCK cuSZ,   auto    DISABLED  outliers 1772864 vs 861737  setup 70.49 ms
```

`1772864 - 861737 = 911127` is exactly the priming buffer's outlier count, so C
caught the real mechanism. `CLIO_CUSZ_REUSE_MANAGER=0` forces the per-chunk
path; the harness knob is `CUSZ_REUSE`, recorded in `run.json`.

---

## 2. What is NOT settled — fix these first

### 2.1 cuSZ and cuSZp3 miss the error bound, badly (NO LONGER CHECKED)

**Decision (2026-09-23):** `figure_9.sh` no longer runs `--check-bound` on the
cuSZ and cuSZp3 arms (`--no-verify` instead, `bound` column empty). Both codecs
quantize internally, and `compressor_runtime.cc` disables our quantizer for
them. cuSZ's Lorenzo kernel prequantizes with `round(x * 1/(2eb))` in fp32
(`lrz_c.cu.inl`). cuSZp does `cvt.s32(x * 0.5f/eb)` (`cuSZp_kernels_1D_f32.cu`).
The bound is their contract, so the figure times them and does not certify
them. Arms that use our quantizer are still checked.

Root cause, checked locally 2026-09-23 on nyx-i96 (12 files, 192 chunks),
with the verifier compared against each codec run standalone (no Clio code):

- **cuSZ, max|err| 1.048576e+03: our wrapper's config, not the verifier.**
  `cusz.h` hardcodes `codec1 = HF`. cuSZ's own CLI with `--pred lrz --hist
  generic --codec1 hf` decodes an all-zero chunk as a ramp of -512 quant steps
  per element (1024 x 1.024 = 1048.576), which is exactly our result. With
  `hfr-v3` (cuSZ's DEFAULT_CODEC, and what upstream NeuroPress uses) the same
  chunk decodes exactly. Every 1048.576 chunk is flat.
- **cuSZ and cuSZp, overshoots of 1.0376e-3 to 1.5625e-2: the codecs'.**
  Standalone cuSZ and a standalone cuSZp probe reproduce the same worst errors
  on the same chunks. They come from fp32 reconstruction; at |x| ~ 2.4e5 the
  fp32 spacing is 1.5625e-2.
- The two codecs did NOT fail identically. cuSZp never shows 1048.576.



```
BOUND FAILED: 3354 chunk(s) against eb=1e-3, 2931 exceeded, worst max|err|=1.048576e+03
```

**87% of chunks**, worst error the magnitude of the field itself. Identical
across stock cuSZ, patched cuSZ, per-chunk and reused — so it PREDATES all of
this work and the reuse fix neither caused nor cured it. It matches the known
flat-chunk zero-bit-Huffman failure.

Until this is understood, **cuSZ and cuSZp3 are not valid entries in a
quality-matched comparison at any speed**, and the 8.4x speedup does not rescue
them. This is the single most important thing to fix on Chameleon.

Note: an earlier reading of this as "a 0.7% overshoot" was wrong. That number
(`1.007080e-03`) came from a synthetic probe, not from the campaign.

### 2.2 ndzip runs LOSSLESS inside the lossy panel

Panel (b) claims every codec runs at the same bound, but the arm specs are

```
Best fixed nvCOMP: static-bitcomp-q-s4   -q, quantized at eb
ndzip:             static-ndzip          NO -q
cuSZp3 / cuSZ:     static-cuszp/-cusz    quantize internally
```

`ndzip` has no `-q`, so it is lossless (ratio 1.579, `bound: ok` trivially).
`run_config.sh` warns about exactly this: "without -q a fixed-codec arm ...
competes lossless against a lossy selector -- which is a broken control, not a
baseline". DELIBERATE FOR NOW — ndzip is to stay lossless — but every
NeuroPress-vs-ndzip number is lossy-vs-lossless and must be labelled as such.

### 2.3 Who quantizes, for the record

| codec | quantization | how ε arrives |
|---|---|---|
| cuSZ, cuSZp3 | **theirs, internal** | `SetErrorBound` -> `psz_rc2` / `cuSZp_compress` |
| nvCOMP, ndzip | **ours**, a preprocessing kernel | only with `-q` |

`compressor_runtime.cc:556` force-disables our quantizer for cuSZ/cuSZp so the
bound is never applied twice.

### 2.4 Measurement methodology — an external audit found real defects

Full report: `fig9_timing_review.md` beside this file. Verified against the
data, highest impact first:

1. **`runs=1` everywhere.** `std_min` is empty. An identical Baseline config
   measured 0.9002 and 0.6992 min on different days (22%), and a per-chunk cuSZ
   arm measured 61.4 s and 74.9 s of setup (22%). **No comparison in the table
   above is statistically established.** 8.8 vs 9.0 vs 9.8 is noise.
2. **The solid segment is a residue, not a measurement.** `compute = total - io`
   absorbs GPU idle, allocator time, WAL writes, scheduler latency. Worse, the
   I/O union may overlap compute, so for the `+Async` arms — whose whole design
   is overlap — overlapped time is charged to I/O and removed from compute. A
   two-segment stack asserts a serial decomposition the system does not have.
3. **The pale segment is not all device I/O.** `io.csv` logs `tier` in
   {`file`,`ram`} and the union takes every row. RAM-tier writes are memcpy:
   0.387 s / 17.5% for cuSZp3+Tier, 0.451 s / 14.2% for NP+Tier+Async+Lossy,
   0.253 s / 1.7% for nvCOMP+Tier. Only tiered arms are affected — which is the
   comparison panel (a) is built on.
4. **`setup_min = 0` means "not measured" for every codec except cuSZ.**
   Encoding unmeasured as zero is false data; it invalidates `--deduct-setup`
   cross-codec charts. cuSZp3's per-chunk setup is bounded at ~1.6 ms but has
   never been measured (3354 x 1.6 ms = 5.37 s against 9.8-15.7 s bars).
5. **Input and tier-2 share a device.** `staged_input=/tmp/fig9-input-<job>`
   and `nvme_root=/tmp/fig9-nvme-<job>` — same filesystem, same NVMe. Penalises
   tiered arms, so removing it should *strengthen* the tiering result. Needs one
   paired tmpfs control to bound.

Verified as NOT active (latent, worth hardening anyway): the clamp never fired
(`compute+io-total` is +/-0.0001 across all 16 arms); the silent union->sum
fallback never fired (all logs carry `start_ns`); reads do not leak into
`io.csv` (Baseline logs exactly 26.20 GiB = the payload written, cuSZ 2.64 GiB
= its compressed size), so `read_s`/`h2d_s` are not double-subtracted.

One real inconsistency: intervals are clipped to the raw window while `total_s`
is then shortened by `read_s` and `h2d_s`. `driver_total == window` exactly
(52.980 == 52.980), so the two stop describing the same interval. Benign today
only because the read is a separate up-front phase that emits no I/O rows.

### 2.5 Unfinished runs

- **cuSZ untiered, reuse ON** — never ran. Control is measured (85.2 s, setup
  74.9 s); predicted ~10 s.
- **nvCOMP best/worst sweep** — never completed. `Best fixed nvCOMP` is still
  pinned to `static-bitcomp-q-s4` from an offline oracle sweep whose provenance
  is a comment (`figure_9.sh:190`). A "worst fixed nvCOMP" arm was requested and
  does not exist yet. Sweep script: `np-fix/jobs/fig9_nvcomp_sweep.sbatch`,
  8 backends over 1 GiB. Two cautions: `figure_9.sh` defaults `REPS=3`, and
  rep-to-rep spread reached 4.2x (snappy 2.372 s vs 0.564 s), so ranking eight
  backends by arm total on 1 GiB may not be reliable — rank on codec-kernel time
  instead.

---

## 3. Changes in this commit

**Codec wrapper** (`context-transport-primitives/include/clio_ctp/compress/`)
- `cusz.h` — thread-local manager+stream reuse; `ReuseSelfTest` guard; the
  stale "REJECTED: reuse corrupts" note rewritten (it also claimed `compress()`
  zeroes its outlier state every call, which is the opposite of the truth).
- `cuszp.h`, `compress.h` — `SetErrorBound` made a real virtual override so the
  requested ε reaches the codec.

**Harness** (`paper-benchmark/figures/fig9/`)
- `figure_9.sh` — `CUSZ_REUSE` knob (unset = the wrapper self-tests); tier
  images and the bdev image now removed on `scancel` (the TERM handler used to
  skip both cleanups and leak a tier the size of the payload); `setup_min` is a
  clipped interval union, not a sum.
- `plot_fig9.py` — seconds instead of minutes, converted once at the CSV
  boundary so the axis, labels, `--ylim` and sanity checks all agree.

**Third-party patch** (`paper-benchmark/patches/`)
- the cuSZ fix, because `np-src/cuSZ` is an external checkout that is not ours
  to push. Apply with `git apply` on `e1c0135`, build to a separate prefix.

---

## 4. Rebuilding the environment

```
np-env/cusz          stock e1c0135            reuse corrupts, guard refuses
np-env/cusz-fixed    + the patch              reuse correct, guard allows
np-build/clio-test       -> links stock
np-build/clio-cuszfix    -> links cusz-fixed
```

`fig9_arm.sbatch` takes `BUILD` and `CUSZ_REUSE` from the environment and logs
the resolved `libcusz.so` path, so a bar can be traced to the cuSZ it ran
against rather than to an assumption about it.

---

## 5. Before the all-workload run

1. Bound failures (2.1): no longer checked for cuSZ/cuSZp3. Before trusting
   their bars, rule out the shared-cause caveat there.
2. Decide reps. At `runs=1` nothing is established; five paired reps is what the
   audit asks for. **Budget: 141 GPU-hours left of 4947** — this is the binding
   constraint, and reps are the expensive axis.
3. Relabel or split the pale segment (2.3), and encode unmeasured `setup_min` as
   `NA` rather than 0 (2.4).
4. Run the nvCOMP sweep once and hardcode best + worst per workload (2.5).
5. Remaining workloads are VPIC, LAMMPS, WarpX, AI; dumps live under
   `/work/hdd/$ACCT/$USER/np-dumps/<workload>/fields`. All five are fig9 inputs
   — `figure_9.sh:189` records that two of them were once deleted as "figure 8
   data" and had to be regenerated.

### Operational notes learned the hard way

- **Never edit a script while a job is executing it.** Bash reads scripts
  incrementally by byte offset; editing `figure_9.sh` mid-run made a job execute
  a fragment of an embedded heredoc (`line 1036: syntax error near 'open'`).
- `/work/hdd` is a shared PROJECT quota. imuradli holds ~103 GB of it; the bulk
  is other members. Untiered arms write the payload there, so serialise them
  with `--dependency=singleton`.
- Raw bytes (`chi_bdev.dat`, `cte_tier*.dat*`) are deleted after every arm and
  must never be committed. Plots belong under `paper-benchmark/figures/<fig>/`;
  CSVs stay out of the repo.
