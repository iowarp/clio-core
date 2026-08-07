# Tier-Aware Compressed GPU Vector — Project Concept

> A `vector`-like abstraction whose pages live across **HBM → DRAM → NVMe → PFS**, kept resident
> as long as possible by **automatically-selected, tier-aware compression**, with a CPU-side
> predictor steering placement and an **asynchronous pipeline** that hides data movement behind
> GPU compute — so memory-bound GPU workloads run as if the GPU had many times its real memory,
> with **no explicit I/O**.

Target venues: USENIX FAST / ASPLOS. Intended as a CLIO feature.

---

## 1. The core problem: GPU memory is tiny, fast, and expensive

A GPU has a small pool of extremely fast memory (**HBM**, High-Bandwidth Memory) — 8 GB on an
RTX 5060, 80 GB on an H100. Below it sits a hierarchy that trades speed for capacity:

| Tier | Typical size | Bandwidth (rough) | Latency |
|------|------|------|------|
| **HBM** (on-GPU) | 8–80 GB | 1,000–3,000 GB/s | ~100s of ns |
| **DRAM** (CPU RAM) | 128 GB–2 TB | 100–400 GB/s | ~100 ns |
| **NVMe** (local SSD) | 1–30 TB | 3–14 GB/s | ~10–100 µs |
| **PFS** (parallel FS, networked) | petabytes | varies | ms |

As you descend, capacity explodes but speed collapses. The central tension of GPU computing is
that the working set you want (a large model, a big scientific dataset, a giant graph) is far
bigger than fast HBM, so data constantly shuffles up and down — and that shuffling, not the math,
is the bottleneck.

**Thesis:** use compression to make the fast tiers behave as if they were bigger, and use
asynchronous scheduling to hide the cost of moving data.

## 2. What compression buys: capacity expansion

Compression trades **compute for capacity**. At 3× ratio, 8 GB of HBM holds 24 GB of working set.
But to *use* compressed data the GPU must **decompress** it, costing cycles. Compression wins only
when the data is compressible **and** decompression is cheaper than fetching the raw bytes from a
slower tier. That second condition is the whole game: decompressing in HBM at ~2,000 GB/s can be
far cheaper than reading raw bytes from NVMe at ~5 GB/s. So compression is most valuable at the
**boundary between a fast tier and a slow tier** — it lets you avoid a trip to the slow tier.

## 3. Why "tiered" compression is the key insight

The right compression decision depends on **where the data is going**:

- Staying in **HBM** under pressure → a **fast, GPU-native, lightweight** compressor (low ratio,
  near-zero latency); you just need breathing room.
- Evicting to **DRAM** → a **medium** compressor; more time, want a better ratio.
- Heading to **NVMe/PFS** → a **heavy, high-ratio** compressor (possibly on the CPU); the slow
  tier's bandwidth is so precious that spending many cycles to shrink 5× pays off.

The *same* page deserves *different* compression depending on its destination tier. Most systems
pick one compressor and apply it everywhere. **Tier-aware compression selection** is a core
novelty of this project.

## 4. The vector abstraction — a three-layer cache

The user sees a simple array: `x = vec[25]` (read), `vec[25] = x` (write). Underneath, the vector
is paged, and each page lives in one of three places:

```
   ┌─────────────────────────────────────────────┐
   │  Binary cache (HBM, UNCOMPRESSED, fast)      │  ← GPU computes directly here
   └─────────────────────────────────────────────┘
             │ evict ▼            ▲ decompress
   ┌─────────────────────────────────────────────┐
   │  Compressed cache (HBM/DRAM, COMPRESSED)     │  ← holds 3-5× more data
   └─────────────────────────────────────────────┘
             │ evict ▼            ▲ fetch
   ┌─────────────────────────────────────────────┐
   │  Storage hierarchy (NVMe → PFS via Clio)     │  ← cold, vast backing store
   └─────────────────────────────────────────────┘
```

**Write path** `vec[25] = x`:
1. The page must be in the **binary cache** (uncompressed) — the GPU computes only on raw bytes.
2. If absent and the cache is full → **evict** a victim *down* into the **compressed cache**
   (compressed on the way).
3. The compressed cache is **drained aggressively in the background** toward storage, so there is
   always room.
4. If the compressed cache is also full → evict its coldest compressed pages to **Clio**
   (storage → NVMe → PFS).

**Read path** `x = vec[25]`:
1. In the **binary cache**? → return (fast path).
2. Else, in the **compressed cache**? → **decompress** into the binary cache, return.
3. Else → **fetch** compressed bytes from storage → compressed cache → decompress → binary cache.

This is a multi-level cache hierarchy with a twist: one level stores data **compressed**, and the
moves between levels are compress/decompress operations rather than plain copies.

## 5. The prediction algorithm — which pages deserve the compressed cache

A CPU-side predictor (the system's brain) decides, per page, three coupled questions:
1. **Will this page be touched soon?** (locality/access prediction, like a prefetcher)
2. **How well does it compress, and how costly is that?** (data characterization)
3. **Given current per-tier pressure, where should it live and in what form?** (placement +
   compressor selection)

Running it on the CPU is deliberate: the CPU has spare cycles while the GPU computes, and the
logic is branchy/irregular (bad for GPUs). This is the lineage of prior work **DTSchedule** and
**NeuroPress**. We will generally choose CPU compression *or* GPU compression per situation, not
both at once.

## 6. Asynchronous I/O — overlapping compute and data movement

The naïve loop stalls: the GPU asks for a page, waits for NVMe, then computes — wasting the
expensive resource. **Asynchronous I/O** pipelines via double-buffering: while the GPU computes on
data it has, the system **simultaneously** prefetches the next data and evicts the data just
finished. I/O cost hides behind compute. (This generalizes the per-layer prefetch/compute/evict
pipeline with deferred release already used in our FlexGen-style weight offloading.) Technologies
like **GPUDirect Storage** (NVMe→GPU without a CPU bounce) make this faster still.

## 7. Persistent memory — eliminating the copy

Normally "data in memory" and "data in a file" are two copies moved with `read()`/`write()` — pure
overhead. A **single-level / persistent store** makes the `vector` *be* the file: the same logical
bytes, addressed the same way, whether in HBM or out on PFS. No explicit save/load — writes persist
as a natural consequence of eviction, reads pull transparently. Our **Clio / IOWarp CTE** blob
storage is the substrate.

## 8. The three contributions

1. **A dynamically compressed GPU buffer cache** that accelerates memory-intensive workloads using
   **automatic, tier-aware compression-algorithm selection** across both CPU and GPU compressors.
2. **An aggressive asynchronous data-placement algorithm** that overlaps compute and I/O to stage
   data onto the GPU as fast as possible, with a GPU-resident cache for highly-compressed data.
3. **A persistent-memory abstraction** that eliminates the copy between memory and storage.

## 9. Where this sits in the literature (one line)

> G10 showed a unified GPU-memory/storage abstraction wins by smart *migration*; FlexGen showed
> offloading with a *single fixed* compressor works for LLMs. Neither compresses adaptively. We
> unify a **persistent** single-copy abstraction with **automatic, tier-aware, CPU+GPU compression
> selection** across the full **HBM→DRAM→NVMe→PFS** hierarchy, for general workloads.

See `../../../../Literature_Survey_GPU_Compression.docx` for the full survey and novelty table.

## 10. Working name

**ZipTier** (primary) — *"Automatic Tier-Aware Compression for GPU Memory Capacity Expansion across
the Storage Hierarchy."* Alternates: **Cascade**, **Helix**.

---

## Appendix A — Proof-of-concept plan (this branch)

Validate the mechanics with a **Gray-Scott reaction-diffusion** microbenchmark that writes I/O per
block per step, comparing four data-transfer strategies. Parameters: `blocks`, `threads/block`,
`bytes written per block per step`, `number of steps`.

| Case | Description |
|------|-------------|
| **raw transfer** | GS step → wait for kernel → D2H copy → relaunch. Serial baseline. |
| **async transfer** | Double-buffered: 2 GPU buffers + 1 CPU buffer; GS writes buf[0], async D2H while GS computes buf[1]. Overlaps compute + copy. |
| **compressed transfer** | GS writes device buf[0]; on step completion **cuSZp compresses via CDP** (child kernel launched from the parent) into buf[1]; async D2H buf[1]→host. |
| **async + compressed** | Three buffers; combines double-buffering with CDP compression. |

**Key PoC question:** can we invoke a compressor *from within* a CUDA kernel via **CUDA Dynamic
Parallelism (CDP)**? Answer: **cuSZp can** (it is a `__global__` kernel, CDP-launchable). nvcomp's
public batched API is host-only and cannot be CDP-invoked. Metrics: end-to-end time and scaling
across the four cases, run locally (RTX 5060, sm_120) then on **Delta**.
