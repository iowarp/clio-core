# NeuroPress kernels — where the time actually is

Context for accelerating the NeuroPress write path, written from the figure-9
campaign of 2026-09-23. Everything here is **measured on Delta A100s**, not
estimated. Read `fig9.md` for what the figure measures and `HANDOFF.md` for the
campaign's open problems; this file is only about kernel cost.

**The headline: the codec kernel is 5-37% of the bar, and on the two workloads
where NeuroPress looks best it is 5%.** Do not start by optimising a codec.
Sections 2 and 5 say where the rest of the time is.

---

## 1. The workloads and what they cost

Five workloads, 8 MiB chunks, one run per arm, ε = 1e-3 where lossy applies.
`NeuroPress` is the panel (b) arm (untiered, writes each chunk through to
Lustre); `NP+Tier+Async+Lossy` is the panel (a) rung (RAM tier 1 -> node NVMe,
500 ms periodic flush). The gap between those two is I/O placement, not codec.

| workload | payload | NeuroPress (s) | NP+Tier+Async+Lossy (s) | Baseline (s) | NP ratio |
|---|---|---|---|---|---|
| Nyx    | 26.8 GiB | 17.3 |  9.3 | 44.0 | 6.435 |
| VPIC   | 40.0 GiB | 29.2 | 14.0 | 64.8 | 5.275 |
| WarpX  | 40.1 GiB | 82.5 | 24.3 | 64.7 | 1.592 |
| LAMMPS | 30.0 GiB | 60.9 | 20.9 | 55.1 | 1.623 |
| AI     | 14.1 GiB | 27.1 |  — (lossless-only) | 23.1 | 1.006 |

Baseline is an uncompressed GPU -> host -> PFS write. Its bar starts at the D2H
and ends with the bytes on Lustre; the host->device staging that only exists
because this is a replay is subtracted. Every Baseline lands at 0.61-0.65 GiB/s,
which is Lustre's single-stream ceiling here — confirmed independently by a
no-Clio `cudaMemcpy`+`write()`+`fdatasync` probe.

**WarpX and AI are the cases where NeuroPress loses to Baseline.** That is the
problem to solve, and section 5 says why it loses.

---

## 2. Kernel time as a share of the bar

Summed over every chunk of the `NeuroPress` arm, from `blobs.csv`:

| workload | chunks | codec (s) | preproc (s) | sum (s) | bar (s) | kernel share |
|---|---|---|---|---|---|---|
| Nyx    | 3354 |  0.52 | 0.49 |  1.01 | 17.3 |  **5.9%** |
| VPIC   | 5120 |  0.71 | 0.73 |  1.44 | 29.2 |  **4.9%** |
| WarpX  | 5130 | 28.48 | 1.78 | 30.26 | 82.5 | **36.7%** |
| LAMMPS | 3933 |  9.07 | 0.42 |  9.49 | 60.9 | **15.6%** |
| AI     | 1804 |  8.68 | 0.07 |  8.75 | 27.1 | **32.3%** |

`compress_ms` is a CUDA-event bracket around the codec launch alone — it is
byte-identical to upstream's `actual_comp_time_ms`. `preproc_ms` is the quantize
and byte-shuffle kernels, summed. Neither includes allocation, manager setup or
staging; those are in section 5.

On Nyx and VPIC a codec that ran in **zero time** would move the bar by 6% and
5%. There is no kernel win available on those two.

---

## 3. Which kernels are worth touching

Per-chunk codec cost, and how often NeuroPress picks each codec:

| workload | 1st | 2nd | 3rd |
|---|---|---|---|
| Nyx    | bitcomp 54.5% @ 0.12 ms | ans 44.7% @ 0.15 ms | gdeflate 0.4% @ 2.01 ms |
| VPIC   | bitcomp 98.0% @ 0.12 ms | ans 1.9% @ 0.18 ms | lz4 0.1% @ 10.54 ms |
| WarpX  | **zstd 47.2% @ 10.40 ms** | ans 42.7% @ 0.17 ms | lz4 4.2% @ 11.95 ms |
| LAMMPS | ans 66.3% @ 0.17 ms | **zstd 33.2% @ 6.59 ms** | bitcomp 0.4% @ 0.15 ms |
| AI     | **raw 79.4% @ 3.37 ms** | lz4 13.9% @ 9.05 ms | snappy 6.1% @ 13.68 ms |

**`nvcomp-zstd` is 40-90x more expensive per chunk than `ans` or `bitcomp`**, and
it is the most-chosen codec on WarpX. 2421 chunks × 10.40 ms = 25.2 s of WarpX's
28.5 s of codec time. LAMMPS: 1307 × 6.59 ms = 8.6 s of 9.1 s. **Essentially all
codec time in this campaign is zstd.** If one kernel is worth accelerating, that
is the one — but note it is nvCOMP's kernel, not ours.

**`raw(not-beneficial)` on AI is pure waste: 1432 chunks × 3.37 ms = 4.8 s spent
compressing data that was then stored uncompressed.** The codec ran, did not
shrink the chunk (mean ratio 0.994 — it grew), and the original bytes went to
the tier. That is 18% of AI's 27.1 s bar, and it is avoidable in the *selector*
rather than in a kernel: predicting "not beneficial" before paying for the
attempt. This is probably the single best return in the table.

---

## 4. What NeuroPress chooses, and why the choice matters

Codec choice, from `blobs.csv` (`codec` column) of each `NeuroPress` arm:

- The winning codec differs on four of the five workloads: bitcomp (Nyx, VPIC),
  zstd (WarpX), ans (LAMMPS), and "do not compress" (AI).
- VPIC is 98% one codec; Nyx, WarpX and LAMMPS need 2-4 to cover 95% of chunks.
- Byte-shuffle is chosen on **99.9%** (LAMMPS), **96.8%** (Nyx), **21.4%**
  (WarpX), **20.6%** (VPIC), **17.0%** (AI) of chunks on the lossless arms. It is
  not a rare path; on two workloads it runs on essentially every chunk.

Caveat: the chooser learns online, so these distributions **drift run to run**.
Two runs of the same Nyx arm gave 44.1% vs 75.6% for the same codec. Ratios
reproduce to 1-3%; choice shares do not. Pin frozen weights before treating any
share as a constant.

`blobs.csv` records the library but **not** the quantize/shuffle flags —
`core_tasks.h` carries `compress_lib_` and a summed `actual_preproc_time_ms_`,
and the chosen action's booleans never leave the compressor. The shuffle numbers
above are inferred from `preproc_ms > 0` on `eb=0` arms, where quantize cannot
run. `selection.csv` has the exact flags but is off in timed arms because it
FNV-hashes every input and output byte.

---

## 5. Where the non-kernel time goes

From an Nsight audit of WarpX, 8 MiB chunks, one chunk in flight
(`/projects/bekn/imuradli/np-temporal/nsys/{off,on}.sqlite`). **These are
measured; do not re-derive them.**

- **GPU is 89.1% idle.** Per chunk: 8.31 ms busy inside a 42.6 ms
  `stage+compress`, 61.0 ms wall. The 30 largest idle gaps hold 2002.6 ms of
  wall and only 23.0 ms of CUDA API — the GPU is waiting on the host, not on
  itself.
- **FNV-1a-64 digests cost 10.57 ms per 8 MiB pass**, and there are up to three
  per chunk: runtime (`compressor_runtime.cc:1707`, now gated), driver
  (`neuropress_field_replay.cc:132`, still unconditional even with verify off),
  and `LogCompressedPayload` (`neuropress_telemetry.cc:263-285`, ~1.6 ms). This
  was ~50% of the "40 ms/chunk write cost" that framed earlier analysis.
  Measure host costs on the compute node — a login-node microbenchmark
  overstates this by 1.7x.
- **Exactly 2.00 full-chunk H2D per chunk.** `DynamicSchedule` stages the chunk,
  then hands `AsyncCompress` the original *host* ShmPtr
  (`compressor_runtime.cc:1632`). One of these two copies is redundant.
- **`cudaMalloc`+`cudaFree` ≈ 2.2 ms/chunk on medians**, 99.8% of it with the
  GPU idle. Use medians — the 4.07 ms mean is five first-touch outliers.
- NeuroPress's own NN/selection kernels are **2.4%** of GPU kernel time; the
  whole selection chain is 77.8 µs. Prediction reuse moves GPU busy by
  0.03 ms/chunk. **The model is not the bottleneck.**
- **34% of GPU kernel time in a write-only run is decompression**, from
  `MEASURE_DT` / `MEASURE_QUALITY`. Both are off by default in figure 9; if you
  see decompress kernels in a write profile, a diagnostic is on.

### Do not "fix" these

Three findings that look like overhead and are not:

- `cudaEventSynchronize` at 4.17 ms/chunk and `cudaStreamSynchronize` at 19.7
  calls/chunk are **97% / 86% time spent waiting for the GPU**. Absorbed wait,
  not overhead.
- A predicted per-read full-chunk D2D copy-back **does not happen** — 2 D2D
  copies / 4 MiB across a whole read run against 28 `UnshuffleKernel` launches
  (`/projects/bekn/imuradli/np-audit/Q_read.sqlite`).
- The method that produced all of this: intersect every
  `CUPTI_ACTIVITY_KIND_RUNTIME` interval with the merged KERNEL∪MEMCPY∪MEMSET
  busy set. GPU-idle API time is real overhead; GPU-busy API time is not.

---

## 6. Measuring a change

**Noise floor of the replay harness**, from a near-null control arm over 50
chunks: 0.05 s on `stage+compress`, 0.215 s on `total`, 0.735 s on job wall.
A fix must beat **5%** on non-noisy data to count.

**Do not A/B on `stage+compress` alone.** It does not contain the
MEASURE_DT/MEASURE_QUALITY decompression: toggling the diagnostics moves `total`
by +0.599 s / 50 chunks and `stage+compress` by −0.048 s — *opposite signs*.

**Figure 9's own bars are a bad regression test for a kernel change.** Same arm,
same config, run twice: `NP+Tier+Async+Lossy` on WarpX gave 24.3 s and 49.2 s
(102% spread); `NP+Tier` gave 35.3 / 43.5 / 57.6 s. Untiered arms are tighter
(6-10%). Compression *ratio* reproduces to under 3% and is the reliable signal.
Use `blobs.csv` per-chunk `compress_ms` medians, not the bar.

**The compute/I/O split of a bar is not derivable** from current
instrumentation — three methods were tried and all fail, because compute and I/O
genuinely overlap in wall clock. Do not re-attempt without new timers.

---

## 7. Where the data is

Campaign roots on Delta (`/work/hdd/bekn/imuradli/`):

```
fig9-full-nyx-09222322/       fig9-full-vpic-40g-09231439/
fig9-full-warpx-40g-09231547/ fig9-full-lammps-09230423/
fig9-full-ai-09230423/
```

Per arm, under `<campaign>/<workload>/<arm>/`:

| file | what |
|---|---|
| `fig9.csv` | the bar: compute_min, io_min, total_min, ratio, eb, payload_mib |
| `fig9_runs.csv` | one row per rep, before aggregation |
| `arms.csv` | the arm set that was run |
| `.../r<N>/<arm>/blobs.csv` | **per chunk**: codec, ratio, stored, compress_ms, preproc_ms, h2d_ms |
| `.../r<N>/<arm>/phases.csv` | per-chunk phase log (appends across reps — dedupe before summing) |
| `harness.log` | the harness's own trace, including `NOT RECORDED` |

`blobs.csv` is the kernel-level source and is written by every arm at no cost.
`selection.csv` (exact quantize/shuffle flags, predicted vs actual) needs
`SELECTION_LOG=1` and is off by default — it hashes every byte.

Regenerate the tables in sections 2-4 with `kernel_costs.py`, beside this file:

```bash
./kernel_costs.py \
  --camp Nyx=/work/hdd/bekn/imuradli/fig9-full-nyx-09222322/nyx \
  --camp VPIC=/work/hdd/bekn/imuradli/fig9-full-vpic-40g-09231439/vpic \
  --camp WarpX=/work/hdd/bekn/imuradli/fig9-full-warpx-40g-09231547/warpx \
  --camp LAMMPS=/work/hdd/bekn/imuradli/fig9-full-lammps-09230423/lammps \
  --camp AI=/work/hdd/bekn/imuradli/fig9-full-ai-09230423/ai
```

It reads only `blobs.csv` and each arm's own `fig9.csv`, so it needs no rerun.

---

## 8. Constraint: figure 9 does not run on Chameleon

The arms are pinned to Delta's two storage classes — untiered writes tier 1 to
`/work/hdd/<acct>`, `+Tier` spills tier 2 to the node's own NVMe. On a node with
one storage class both roots land on the same device, `+Tier` stops being a
change of tier, and the ablation compares an arm against itself. `require_root`
exits 4 before the first arm rather than let that happen.

What *is* portable: kernel-level work measured with `blobs.csv` per-chunk times
on any A100, and the offline model comparison in `../../model-accuracy/`, which
scores a recorded trace and needs no codecs, no tiers and no GPU. Bring a kernel
change back to Delta for the end-to-end number.
