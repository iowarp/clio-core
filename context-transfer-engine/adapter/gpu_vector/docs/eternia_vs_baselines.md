# Eternia vs GPU-memory-expansion baselines — real-SIFT k-means

Comparison of Eternia against the technologies that either **expand single-GPU memory**
(UVM, BaM) or **pool memory across GPUs** (NVSHMEM, CUDA-aware MPI, NCCL), on the same
workload used throughout: **k-means over real SIFT** (tiled), single RTX 5060 (8 GiB, WSL2/
Docker-on-Windows). Data: `eternia_results/comparison_baselines.csv`; figure:
`eternia_results/figures/fig_comparison_baselines.png`.

## Taxonomy (what each technology actually does)
| Class | Tools | How it "expands" memory | On 1 GPU |
|---|---|---|---|
| Single-GPU oversubscription | **CUDA UVM**, **BaM** | page/stream data beyond HBM (host / NVMe) | genuinely runs > HBM |
| Multi-GPU comms | **NVSHMEM**, **CUDA-aware MPI**, **NCCL** | add GPUs → N×HBM (bytes unchanged) | in-core, caps at HBM |
| **Eternia** | — | compress + tier + stream a fixed window | runs > HBM, less data moved |

Only UVM/BaM/Eternia are single-GPU rivals; NVSHMEM/MPI/NCCL need ≥2 GPUs to add capacity.

## Local results (real SIFT k-means, per iteration)
| technology | 3 GiB | 6 GiB | 9 GiB | 12 GiB | runs > HBM? |
|---|---:|---:|---:|---:|:--:|
| **Eternia** (compress+stream) | 11.1 s | 22.2 s | 33.4 s | 43.8 s | **yes** |
| Traditional in-core (cudaMalloc) | 3.0 s | 6.0 s | **OOM** | **OOM** | no |
| CUDA UVM (managed) | **649 s** | — | — | — | yes* |
| NVSHMEM (1 GPU, symmetric heap) | capped | — | — | — | no |

Also measured (WSL2-clean, from the compression ablation A2): **tier-only** = uncompressed
explicit-chunk streaming — runs > HBM like Eternia, ~14% *faster* per-iter but stores 3.4× more
(no compression). This isolates two Eternia advantages: **compression** (3.4× less data) and
**explicit streaming** (vs UVM's implicit page-faulting).

## The environment blocker (critical — determines what's trustworthy)
**Every multi-GPU/oversubscription baseline is distorted or blocked by WSL2 (Docker-on-Windows),
not by having one GPU:**
- **CUDA IPC/P2P is broken on WSL2** → NVSHMEM symmetric heap caps at ~2–3 GiB (`cudaIpcGetMemHandle`
  OOM); 2-PE NVSHMEM and 2-rank CUDA-aware-MPI GPU-pointer transfer **segfault**.
- **UVM managed memory is host-resident on WSL2** (no real migration) → 3 GiB k-means took **649 s/iter**
  vs traditional 3 s/iter (~200×), even though it fits in HBM. It *does* oversubscribe (10 GiB alloc +
  touch OK), but the timing is **not representative** of UVM on real hardware.
- **BaM** needs a raw NVMe block device + custom driver → no `/dev/nvme` in the container: **infeasible**.
- **NCCL is the exception — it DOES run as "artificial nodes" on one GPU on WSL2** (proven:
  `data/nvshmem/nccl_test.cu`, 2 ranks/GPU0, `NCCL_MULTI_RANK_GPU_ENABLE=1` + socket transport → correct
  allreduce). NCCL has a TCP socket data-plane (bypasses the broken CUDA IPC) and an explicit
  multi-rank-per-GPU flag. So a Docker cluster CAN do NCCL collectives here. Caveat: the artificial
  nodes share the one 8 GB GPU and one set of SMs, so it validates the distributed *mechanism* and is
  Delta-portable, but it does NOT expand memory past 8 GB and its numbers (socket overhead + on-GPU
  contention) don't represent real multi-GPU. NVSHMEM has no socket data-plane → still WSL2-blocked.

→ **Trustworthy baseline numbers require a native-Linux GPU (Delta).** Presenting the WSL2 UVM/NVSHMEM
timings as representative would be false, so we don't. Harnesses are built and Delta-ready
(`nvshmem_poc/`, `km_nvshmem.cu`, `km_baselines.cu`, `poc_nvshmem.cu`).

## Are the results correct in terms of Eternia's objective?
- **Capacity objective — YES, and the one clean local comparison supports it.** Eternia and UVM are the
  only single-GPU methods that run past HBM; traditional/NVSHMEM/MPI/NCCL OOM or cap. Eternia is **~60×
  faster than UVM** here (even with WSL2 inflating UVM), because it moves 3.4× less data (compression)
  and streams explicitly instead of page-faulting. That gap will shrink on native Linux (where UVM
  migrates properly) but the structural advantages — less data moved, controlled streaming — remain.
- **Comparative-performance claims vs UVM/NVSHMEM/NCCL — NOT yet earned with trustworthy numbers.** The
  WSL2 distortion means the *magnitude* (60×, the NVSHMEM cap) is environment-specific. These must be
  re-run on Delta before any number goes in a paper.

## Next: Delta (native Linux, ≥2 A100)
Run on Delta (see `eternia_megammap_eval_plan.md` Part B, `nvshmem_poc/`):
1. **UVM vs Eternia** at 3–64 GiB — the clean single-GPU oversubscription head-to-head (real page migration).
2. **NVSHMEM / NCCL / CUDA-aware MPI** weak-scaling across N GPUs — the multi-GPU capacity story.
3. **BaM** on a node with a raw NVMe — GPU-initiated storage vs Eternia's tiered spill.
4. 2–3 more recent (≤5 yr) systems: e.g. **DRAGON** (NVMe-backed UM), **GMT/Gimbal**, **HALO** — most
   need bare-metal, hence Delta.
