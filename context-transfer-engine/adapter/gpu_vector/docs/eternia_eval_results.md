# Eternia — MegaMmap-style Evaluation Results

All experiments on a **single RTX 5060 Laptop GPU (8 GiB HBM, sm_120)**, real **SIFT1M**
(1M × 128-d float descriptors, tiled to size), cuSZp v3 (abs error 1e-3, preset BALANCED),
k = 256 clusters. Eternia = tier-aware **compressed** GPU vector (HBM→DRAM→NVMe), streamed via
`SequentialTransaction`. "Traditional" = in-core GPU k-means (whole dataset resident in HBM).

Harnesses: `test_gpu_vector_kmeans_capacity_gpu.cc` (A1–A3), `clio_gs_checkpoint_bench.cu` (A4),
correctness in `test_gpu_vector_kmeans_real_gpu.cc`. Plan for the Delta experiments:
`eternia_megammap_eval_plan.md`.

### Files (`eternia_results/`)
- `a1..a4_*.csv`, `correctness_inertia_parity.csv` — the numbers below, machine-readable.
- `figures/fig_a1..a4_*.{png,pdf}` — plots (regenerate: `python3 eternia_results/plot_results.py`).
- The capacity harness now emits CSV natively: set `CLIO_KM_CSV=<file>` to append a row per run
  (unified A1/A2/A3 schema), and it also echoes `[CSV],<header>` / `[CSV],<row>` to stdout. The CSVs
  here were captured from the sweep logs; future runs can write them directly.

---

## Correctness (prerequisite) — compressed clustering == uncompressed

`test_gpu_vector_kmeans_real_gpu.cc`, inertia parity vs an in-memory uncompressed baseline.

| points | compression | inertia gap (compressed vs baseline) | centroids matching baseline |
|-------:|:-----------:|:------------------------------------:|:---------------------------:|
| 131,072   | 3.36× | 0.09%  | 237/256 (92.6%) within 0.05 |
| 999,424   | 3.36× | 0.093% | **256/256 (100%)** within 0.05 |

Verdict: clustering the compressed data is statistically identical to clustering the raw data
(k-means is multi-modal, so a few centroids land in different-but-equal optima → inertia parity is
the correct check, not per-centroid position).

---

## A1 — Capacity crossover  (MegaMmap Fig 6 analog)

Traditional needs the whole dataset resident in HBM; Eternia streams a fixed 4 MiB window.
1 MiB pages, 3 iterations.

| dataset | Traditional | Eternia (compressed+tiered) | Eternia peak GPU |
|--------:|:-----------:|:---------------------------:|:----------------:|
| 3 GiB  | OK, 9.5 s (pins 3072 MiB) | OK — 900 MiB stored (3.41×), km 33 s | 4 MiB |
| 6 GiB  | OK, 18 s (pins 6144 MiB)  | OK — 1800 MiB (3.41×), km 66 s | 4 MiB |
| **9 GiB**  | **OOM** (needs 9216 > 6944 MiB free) | **OK** — 2701 MiB (3.41×), km 100 s | 4 MiB |
| **12 GiB** | **OOM** | **OK** — 3601 MiB (3.41×), km 131 s | 4 MiB |

- Traditional peak GPU memory = the dataset → hard wall at ~7 GiB (physical HBM).
- Eternia peak GPU memory = constant **4 MiB** streaming window (768×–3072× smaller than the dataset).
- **12 GiB workload ran on an 8 GiB GPU.**
- Tradeoff: Eternia is slower (no free lunch) — it runs workloads that traditional simply cannot.
- Impl note: a doomed multi-GB `cudaMalloc` HANGS on the 12.x driver, so the traditional path is
  gated on a `cudaMemGetInfo` free-memory check and reports OOM cleanly.

---

## A2 — Compression ablation  (Eternia's novel axis vs MegaMmap)

Same k-means; "Eternia" routes storage through the compressor pool (600), "Tier-only" routes raw to
cte_core (512) — MegaMmap-equivalent (tiering, no compression). 3 iterations.

| dataset | mode | stored in tiers | ratio | k-means time |
|--------:|:-----|:---------------:|:-----:|:------------:|
| 4 GiB | Eternia   | 1200 MiB | **3.41×** | 47.3 s |
| 4 GiB | Tier-only | 4096 MiB | 1.00×     | 41.3 s |
| 9 GiB | Eternia   | 2701 MiB | **3.41×** | 105.8 s |
| 9 GiB | Tier-only | 9216 MiB | 1.00×     | 92.0 s |

- Compression = **3.41× more effective capacity** than tiering alone (a fixed tier holds 3.4× more
  data), at a **~13–15% runtime cost** (decompression). This is what Eternia adds over MegaMmap.
- Gotcha: `ctx.dynamic_compress_ = 0` does NOT disable compression — the `CLIO_CTE_COMPRESS_LIB=cuszp`
  pin on pool 600 always compresses. Ablating requires routing to pool 512 directly.

---

## A3 — HBM working-window scaling  (MegaMmap Fig 8 analog)

Fixed **6 GiB** dataset; vary the GPU working-window size (`CLIO_KM_WINDOW` pages, 1 MiB each,
double-buffered → peak = 2×window). 3 iterations.

| window | peak GPU | k-means time |
|-------:|:--------:|:------------:|
| 32 pages | 64 MiB | 122.9 s |
| 16 pages | 32 MiB | 119.6 s |
| 8 pages  | 16 MiB | 70.8 s |
| 4 pages  | 8 MiB  | 70.2 s |
| 2 pages  | 4 MiB  | **67.6 s** |
| 1 page   | 2 MiB  | 85.6 s |

- Shrinking the HBM window **32× (64→2 MiB) does not hurt** — runtime is flat-to-*better*, fastest at
  a 4 MiB window. (MegaMmap Fig 8: ≤10% penalty at 2.6× less DRAM; ours is stronger because compute
  overlaps streaming.)
- Not monotone: large windows are slower (the single-block gather kernel serializes a bigger window);
  window=1 ticks up from per-transaction overhead. Sweet spot = 2–8 pages.

---

## A4 — Gray-Scott checkpoint  (2nd workload, write-intensive; MegaMmap Gray-Scott analog)

`clio_gs_checkpoint_bench`, 8192×8192 field (256 MiB), 100 steps, checkpoint every 25 (4 checkpoints).

| method | per-checkpoint | eff. bandwidth | footprint | location |
|:-------|:--------------:|:--------------:|:---------:|:--------:|
| Traditional (D2H + write full field to disk) | 2198 ms | 0.11 GiB/s | 1024 MiB (uncompressed) | disk |
| **Eternia** (compress in HBM) | **448 ms** | 0.56 GiB/s | **140 MiB (7.3×)** | HBM |

- **Footprint reduction 7.3×** (smooth Gray-Scott field compresses far better than noisy SIFT's 3.41×),
  data stays on-GPU. **4.9× faster** checkpoint.
- Caveat: the 4.9× *speed* is partly inflated by a slow Windows Docker bind-mount disk (0.11 GiB/s).
  On a real PFS the speedup shrinks; the **7.3× footprint reduction and on-GPU locality are robust**.

---

## Global caveats

- The container's **"HBM" bdev tier is emulated** (host-memory-backed), and the `max_bw` DPE routed
  all compressed data to the DRAM tier — so the HBM→DRAM placement *split* shows 0 on HBM. The
  **capacity results are unaffected**; a real multi-level HBM+DRAM+NVMe split needs Delta hardware.
- cuSZp worked cleanly on sm_120 here (no PTX-JIT clobbering observed) despite the documented caution.
- Needs Delta (see `eternia_megammap_eval_plan.md`, Part B): weak scaling vs Spark/NCCL (Fig 5),
  real tier perf/cost (Fig 7), DBSCAN + Random Forest workloads, and Gadget-4 cosmological data for
  exact MegaMmap parity.
