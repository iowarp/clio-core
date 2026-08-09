# Eternia Evaluation Plan — MegaMmap-style

Eternia (the tier-aware, **compressed** GPU vector: HBM→DRAM→NVMe→PFS, streamed via
`SequentialTransaction`) is the GPU sibling of MegaMmap (Logan 2024), whose flagship example is
literally k-means over `mm::Vector` with `SeqTxBegin`. This plan mirrors MegaMmap's evaluation,
with the capacity wall being **GPU HBM** instead of node DRAM, and adds the axis MegaMmap lacks:
**compression**.

MegaMmap testbed for reference: 32-node cluster, dual Xeon Silver 4114 (48 threads), 48 GB DRAM +
128 GB NVMe + 256 GB SSD + 1 TB HDD per node, 40/10 GbE RoCE; up to 768 procs. Workloads: KMeans‖,
DBSCAN, Random Forest, Gray-Scott on Gadget-4 cosmological data. Baselines: Spark 3.4.1 MLlib, MPICH.

---

## Part A — DONE on a single RTX 5060 (8 GB HBM), real SIFT1M (tiled)

| # | MegaMmap analog | Eternia experiment | Result |
|---|---|---|---|
| A1 | **Fig 6** dataset resolution (MPI OOMs at L=2688; Mega runs to 3456) | Traditional in-core GPU k-means vs Eternia, increasing dataset | Traditional **OOMs at ~7 GiB**; Eternia runs **12 GiB on an 8 GiB GPU**, peak GPU = 4 MiB window |
| A2 | *(none — Eternia only)* | Compression ablation: compress vs tier-only | Eternia stores **3.41× smaller** than tiering-only → 3.41× more effective capacity, ~14% slower |
| A3 | **Fig 8** lower DRAM (KMeans 2.6× less DRAM, ≤10% slower) | Shrink the GPU HBM working window on a fixed 6 GiB dataset | Sustained across **2–64 MiB windows**, fastest at ~4 MiB (flat-to-better; not monotone — see notes) |
| A4 | **Gray-Scott** (write-intensive checkpoint) | `clio_gs_checkpoint_bench`: traditional D2H+PFS write vs compressed in-HBM checkpoint | (see results doc) |

Harnesses: `test_gpu_vector_kmeans_capacity_gpu.cc` (A1–A3, env knobs `CLIO_KM_PAGES`,
`CLIO_KM_WINDOW`, `CLIO_KM_COMPRESS`, `CLIO_KM_HBM_MIB`/`CLIO_KM_DRAM_MIB`), `clio_gs_checkpoint_bench.cu` (A4);
correctness in `test_gpu_vector_kmeans_real_gpu.cc` (inertia parity vs uncompressed baseline).

Single-GPU caveat: the container's "hbm" bdev is **emulated** (host-backed), and `max_bw` DPE routed
all compressed data to the DRAM tier (HBM split = 0). The *capacity* results hold; a genuine
HBM→DRAM→NVMe placement split needs real hardware (Part B).

---

## Part B — NEEDS Delta (A100 multi-GPU + real HBM/NVMe tiers) — TO RUN

### B1. Weak scaling vs baselines  *(MegaMmap Fig 5)*
- **Goal:** show Eternia's coherence/streaming isn't a scaling bottleneck, and it uses far less HBM.
- **Setup:** weak scaling 1→N A100 GPUs (e.g. 1,2,4,8), k-means, dataset that FITS in aggregate HBM
  so baselines can run (e.g. 2 GB/GPU, k=8, ≤4 iters — MegaMmap's exact config).
- **Baselines:** (a) traditional in-core multi-GPU k-means (NCCL all-reduce of centroids);
  (b) RAPIDS cuML `KMeans` (dask-cuGraph multi-GPU) if available; (c) optionally Spark-RAPIDS.
- **Metrics:** runtime + **peak HBM %** vs #GPUs. Expected: Eternia competitive runtime at a fraction
  of the HBM (streams a window), analogous to MegaMmap's "3–4× less DRAM than Spark."
- **Also:** NVSHMEM comparison — NVSHMEM pools capacity by *adding* GPUs (N×HBM, bytes unchanged);
  Eternia adds capacity per-GPU via compression+tiering. Show Eternia-on-each-PE holds
  `N × tierstack / ratio` vs NVSHMEM's `N × HBM`. Complementary, not competing.
  **PoC status (`nvshmem_poc/`):** NVSHMEM 3.7.2 builds + runs on our sm_120 GPU (1-PE PoC passes),
  but a multi-PE cluster is empirically impossible on 1 GPU (same-GPU IPC rejected; no MPS/RDMA) —
  confirming this experiment needs ≥2 real GPUs. `poc_nvshmem.cu` + `build_and_run.sh N` are
  Delta-ready; extend to a partitioned k-means (nvshmem reduce of centroid sums) for the head-to-head.

### B2. Increasing dataset resolution to real out-of-core  *(MegaMmap Fig 6, full)*
- **Goal:** the A1 crossover, but pushed to real multi-tier (HBM→DRAM→NVMe) at 100s of GB.
- **Setup:** k-means / Gray-Scott, dataset grown until it exceeds A100 HBM (80 GB) AND DRAM, so it
  spills to node NVMe. Traditional and NCCL-multi-GPU OOM; Eternia keeps running.
- **Metrics:** max feasible dataset size per method; runtime vs size; the real HBM+DRAM+NVMe split
  (which was emulated-to-DRAM-only on the laptop).

### B3. Tiered perf/cost  *(MegaMmap Fig 7)*
- **Goal:** effect of tier composition on out-of-core runtime + $ cost.
- **Setup:** fixed large out-of-core workload; vary DMSH composition across HBM/DRAM/NVMe/SSD; compute
  $/GB (NVMe ≈ .08, SSD ≈ .04, HDD ≈ .02 per MegaMmap). Add the compression axis: compressed vs raw
  at each composition (compression shifts the cost curve — fewer bytes on every tier).
- **Metrics:** runtime + cost per tiering strategy, with/without compression.

### B4. Lowering HBM at scale  *(MegaMmap Fig 8, real)*
- The A3 window sweep, on A100 with a real (non-emulated) HBM tier and a fixed large dataset, across
  the DBSCAN / RF / Gray-Scott workloads too (not just k-means), to match MegaMmap's 4-workload panel.

### Additional workloads to reach MegaMmap parity
- **DBSCAN** and **Random Forest** GPU implementations over the compressed vector (MegaMmap ran both).
  Currently we have k-means (read-streaming) and Gray-Scott (write-checkpoint); adding DBSCAN
  (k-d tree, irregular) and RF (bagging) would complete the 4-workload story.
- **Dataset:** switch SIFT → **Gadget-4** cosmological particle data (3D positions/velocities, HDF5)
  to match MegaMmap exactly, if reviewers want identical inputs.

### Delta logistics
- Build: same recipe as the laptop but native `-DCMAKE_CUDA_ARCHITECTURES=80` (A100 native SASS,
  cuSZp reliable on sm_80). Deploy multi-node via Jarvis / the existing `distributed_slurm` harness.
- cuSZp: install once per node (or into the shared image) — see build recipe.
