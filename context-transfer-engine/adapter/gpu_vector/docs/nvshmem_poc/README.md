# NVSHMEM PoC — attempt on a single-GPU Docker box

**Goal.** Try to stand up an NVSHMEM "cluster" (Docker + GPUs) as a same-hardware baseline for
Eternia, without waiting for Delta.

**Outcome (empirical, 2026-08).** The NVSHMEM *software stack works on our arch*, but a real
**multi-PE cluster is impossible on one physical GPU** — confirmed by direct test, not assumption.

| Step | Result |
|---|---|
| `pip install nvidia-nvshmem-cu12` (3.7.2) | ✅ |
| Compile `poc_nvshmem.cu` for sm_120 (sm_90 bitcode → PTX → JIT) | ✅ |
| **1 PE** on GPU0: `nvshmem_malloc`, GPU-initiated `nvshmem_int_p`, `barrier_all` | ✅ `... got 0 from peer (expected 0) OK` |
| **2 PEs** on GPU0 (default VMM symmetric heap) | ❌ `cudaIpcOpenMemHandle failed (400)` |
| 2 PEs, `NVSHMEM_DISABLE_CUDA_VMM=1` (legacy IPC) + `NVSHMEM_REMOTE_TRANSPORT=none` | ❌ same IPC failure |
| MPS (the only multi-process-per-GPU path with working IPC) | ❌ `nvidia-cuda-mps-control` absent (Docker-on-Windows) |
| **2-container Docker cluster** (both `--gpus all --ipc=host`, shared volume), cross-container CUDA IPC | ❌ exporter `cudaIpcGetMemHandle` OK, importer `cudaIpcOpenMemHandle` → **"invalid device context"** |

**Docker-cluster attempt (`ipc_test.cu`).** Directly tested the capability NVSHMEM needs — one
container exports a GPU IPC handle, a second container imports it. Export succeeds; **import fails with
`invalid device context`**. This is the textbook symptom of **CUDA IPC being unsupported on WSL2**
(Docker Desktop on Windows = WSL2 GPU paravirtualization; `cudaIpc*` is not implemented). So a Docker
"cluster" cannot run NVSHMEM here regardless of how the containers are wired — the block is the WSL2
GPU driver, not the container topology. On a **native-Linux** host (Delta), cross-process/-container
CUDA IPC works and the cluster runs.

**Why it can't work on 1 GPU.** NVSHMEM maps **one PE per GPU**. Two PEs on the *same* device try to
`cudaIpcOpenMemHandle` each other's symmetric heap; the driver rejects same-GPU IPC peer mapping.
MPS would give each process an isolated context that can IPC, but it isn't available here. True
*multi-node* NVSHMEM needs an RDMA transport (IBGDA/IBRC/UCX over InfiniBand) — a Docker bridge
network doesn't provide it. So neither "many PEs on one GPU" nor "many containers over TCP" forms a
real NVSHMEM cluster.

**What this establishes for the paper.** The earlier claim — *NVSHMEM's capacity comparison
fundamentally requires ≥2 physical GPUs* — is now empirically verified, not asserted. NVSHMEM buys
capacity by adding GPUs (N × HBM); it cannot pool beyond one GPU's memory. Eternia, on the same
single 8 GiB GPU, ran a 12 GiB workload via compression + streaming. They are complementary: Eternia
could run on each PE of an NVSHMEM job to hold `N × tierstack / ratio` instead of `N × HBM`.

**Delta-ready.** `poc_nvshmem.cu` + `build_and_run.sh` run unchanged on a real N-GPU node:
`./build_and_run.sh N` (use `-arch=sm_80` native for A100). The next step there is a distributed
k-means (partition SIFT across PEs, `nvshmem` reduce of centroid sums) for the head-to-head vs
Eternia — see `../eternia_megammap_eval_plan.md` Part B / B1.

## Files
- `poc_nvshmem.cu` — N-PE symmetric-heap + GPU-initiated one-sided put, MPI-bootstrapped.
- `build_and_run.sh` — compile + run recipe (validated); `./build_and_run.sh <num_pes>`.
