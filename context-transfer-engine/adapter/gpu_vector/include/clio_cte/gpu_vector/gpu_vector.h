/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_GPU_VECTOR_H_
#define CLIO_CTE_GPU_VECTOR_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector_kernels.h>
#include <clio_cte/gpu_vector/gpu_vector_page.h>
#include <clio_cte/gpu_vector/gpu_vector_view.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <deque>
#include <functional>
#include <unordered_map>
#include <thread>

#if CTP_IS_GPU_COMPILER

namespace clio::cte::gpu_vector {

/**
 * Cache management strategy. kLegacy reproduces the original
 * pre-async-thread behavior (single HBM tier, periodic flush kernel,
 * no rescore queue). kAsync wires in the new four-phase
 * CacheManagerKernel, two-tier cache, and per-block rescore queues.
 *
 * `host_pages_per_block > 0` auto-promotes the mode to kAsync. Existing
 * callers that pass only the legacy parameters get kLegacy by default.
 */
enum class CacheMode {
  kLegacy = 0,
  kAsync = 1,
};

/**
 * Producer-only GPU-resident vector backed by CTE blob storage. See
 * the file-level comment in gpu_vector_page.h for the data layout and
 * AGENTS.md "GPU Producer-Only Model" for the CLIO_IPC->Send model.
 *
 * Lifecycle:
 *   - Caller must have already initialized the CLIO Runtime runtime and
 *     the CTE core pool (see clio::cte::core::CLIO_CTE_CLIENT_INIT plus
 *     AsyncCreate(...)).
 *   - ctor: allocates the backends (HBM pages [kDeviceMem], optional
 *           DRAM pages [kPinnedHost], swap scratch [kDeviceMem], meta
 *           [kDeviceMem], task pools [kPinnedHost]); registers them
 *           with the runtime; resolves/creates the tag; placement-news
 *           every PutBlob/GetBlob slot; runs an InitMeta kernel; spawns
 *           the cache-management thread.
 *   - dtor: stops the cache thread, FlushAllSync, unregisters + frees
 *           backends.
 */
template <typename T>
class Vector {
 public:
  /**
   * @param tag_name                  CTE tag name (per-page blobs land here).
   * @param nblocks                   Number of independent block streams.
   * @param gpu_id                    GPU device id.
   * @param gpu_pages_per_block       HBM cache slots per block.
   * @param host_pages_per_block      DRAM cache slots per block.
   *                                  0 ⇒ legacy single-tier mode.
   * @param page_size_bytes           Per-page byte size.
   * @param cache_period_us           Cache-thread tick in MICROSECONDS.
   *                                  0 disables the thread; FlushAllSync
   *                                  still works. Default 50000 us = 50 ms.
   * @param mode                      Force CacheMode. Default: kLegacy
   *                                  when host_pages_per_block == 0,
   *                                  else kAsync (auto-promoted in ctor).
   * @param manager_threads_per_block Threads per CacheManagerKernel
   *                                  block. Must be a multiple of 32
   *                                  (default 32, i.e. one warp).
   */
  Vector(const std::string &tag_name, clio::run::u32 nblocks,
         clio::run::u32 gpu_id = 0,
         clio::run::u32 gpu_pages_per_block = 4,
         clio::run::u32 host_pages_per_block = 0,
         clio::run::u64 page_size_bytes = 1ULL << 20,
         clio::run::u32 cache_period_us = 50000,
         CacheMode mode = CacheMode::kLegacy,
         clio::run::u32 manager_threads_per_block = 32,
         bool allow_cold_miss_fault = false,
         clio::run::PoolId storage_pool_id = clio::run::PoolId(0, 0),
         clio::run::u64 family_pages = 0,
         std::function<std::string(clio::run::u64)> host_namer = nullptr,
         bool flat_layout = false);
  ~Vector();

  /**
   * Blob name for global page `gp` under the vector's family policy —
   * the ONE host-side place page names are composed. Default:
   * "<tag>_b<gp / family_pages>_pi<gp>" (family 0 when family_pages == 0),
   * which matches both the CAE ModelWeightsAssimilator's sharding and the
   * runtime-side composition driven by gpu_family_idx_. `host_namer`
   * overrides the whole string for exotic layouts.
   */
  std::string PageBlobName(clio::run::u64 gp) const;
  /** 0 for the per-page-blob layout; gp * page_size for the flat layout. */
  clio::run::u64 PageBlobOffset(clio::run::u64 gp) const;
  bool FlatLayout() const;

  Vector(const Vector &) = delete;
  Vector &operator=(const Vector &) = delete;

  /** Synchronously drain every dirty page. Launches the cache-mgmt
   *  kernel once and a drain kernel that calls Wait() on all in-flight
   *  put / get futures. */
  void FlushAllSync();

  /**
   * Host-driven materialization of every stored page into its HBM slot.
   *
   * Read-side counterpart to eviction, needed for the COMPRESSED vector: an
   * on-device page fault spin-waits on the GPU, but a compressor GetBlob is
   * serviced by launching a GPU decompression kernel (cuSZp, which internally
   * cudaMalloc/cudaMemcpy-synchronizes the device) -- so the fault kernel and
   * the decompressor deadlock on a single GPU. FaultAllSync avoids this by
   * decompressing from the HOST while the GPU is idle: for every HBM slot it
   * issues a GetBlob (routed through the storage pool; if that pool is a
   * compressor, it decompresses) straight INTO the slot's HBM address, then
   * marks the slot resident. A subsequent device read kernel finds every page
   * present -- no on-device fault, no spin, no deadlock.
   *
   * Legacy/single-tier only (host_pages_per_block == 0): the HBM tier holds the
   * whole logical extent, so materializing every slot materializes the vector.
   *
   * @param num_model_pages  Number of pages the stored object actually occupies.
   *   Only these pages have a backing blob; the remaining HBM slots are unused.
   *   Pass 0 (default) to fault every slot -- correct only when the object fills
   *   the cache exactly. When the object is SMALLER than the cache (gpu_ppb),
   *   passing 0 makes GetBlob fail on the first page past the object (e.g.
   *   "<tag>_b0_pi<num_model_pages>" does not exist), which aborts the whole
   *   pre-fault and forces the slow on-device-fault path. A model-weights tag is
   *   almost always smaller than the 4096-slot cache, so callers should pass the
   *   real page count.
   */
  void FaultAllSync(clio::run::u64 num_model_pages = 0);

  /**
   * Windowed host-driven prefetch (the #700 SequentialTransaction primitive).
   *
   * Decompress the contiguous run of `gpu_pages_per_block` global pages starting
   * at `first_page` into this (single-block) vector's HBM slots and mark them
   * resident, binding slot s -> global page (first_page + s). The GPU is idle
   * during the decompress (no on-device fault). Unlike FaultAllSync (which loads
   * the whole vector), this loads only ONE window, so a dataset far larger than
   * the HBM cache can be swept window-by-window: prefetch a window -> read it
   * on-device (pages resident) -> advance. Single-block, single-tier only
   * (nblocks == 1, host_pages_per_block == 0); the window size is the HBM cache
   * (gpu_pages_per_block). Pages are named "<tag>_b0_pi<global_page>" to match
   * the store.
   */
  void PrefetchWindowSync(clio::run::u64 first_page);

  /**
   * Prefetch `count` global pages [first_page, first_page+count) into HBM slots
   * [slot_base, slot_base+count) and bind them resident. The building block for
   * PIPELINED (double-buffered) prefetch: keep gpu_pages_per_block == 2*window
   * and alternate slot_base between 0 and window, so window W+1 can be prefetched
   * into one buffer while a device kernel reads window W from the other. Unlike
   * PrefetchWindowSync this does NOT device-synchronize (it syncs only its own
   * mark stream), so it will not stall a concurrently-running read kernel.
   * Single-block, single-tier; slot_base + count <= gpu_pages_per_block.
   *
   * `batched`: issue all `count` GetBlobs asynchronously then wait for them
   * together (vs one serial round-trip per page). Intended to let the compressor
   * decompress a window's pages concurrently -- but MEASURED SLOWER on a single
   * GPU (the cuSZp decompress kernels serialize on the device anyway, so
   * concurrent issue only adds scheduling / pinned-pool contention). Default is
   * therefore FALSE (serial); the flag is kept for A/B measurement.
   */
  void PrefetchPagesSync(clio::run::u64 first_page, clio::run::u32 count,
                         clio::run::u32 slot_base, bool batched = false);

  /**
   * Re-score `count` page blobs [first_page, first_page+count) so the CTE data
   * organizer moves them between storage tiers: score 1.0 stages a page into
   * the fastest tier (HBM in a tiered compose) BEFORE the window loader needs
   * it, score 0.9 drains a finished page back to the DRAM tier. This is the
   * backing-store half of deterministic schedule prefetch — the schedule knows
   * exactly which pages are needed soon and which are done until next token.
   *
   * Reorganize is a metadata/placement op owned by the CTE CORE, and the
   * compressor entrypoint does not forward it, so this always targets
   * kCtePoolId regardless of the page-traffic storage pool. Blocking (waits
   * every future); call it from a background thread, never the compute path.
   */
  void ReorganizePagesSync(clio::run::u64 first_page, clio::run::u32 count,
                           float score);

  /** Cache-thread-only: drains the per-block host_prefetch_q, issues
   *  AsyncGetBlob via the CPU-side CTE client for each directive,
   *  waits for completion, then launches a tiny kernel to clear
   *  kPageBusy + kPageGetInFlight on the prefetched slot. The
   *  stream parameter is the non-blocking cache-thread stream so
   *  the clear-flags kernel doesn't block behind any default-stream
   *  user kernel (which would deadlock against the slot lock the
   *  user kernel is spin-waiting on). */
  void DrainHostPrefetchQueue(void *cuda_stream);

  /** POD captured by user kernels — pass by value into a __global__. */
  DeviceView<T> Device() const { return view_; }

  const clio::cte::core::TagId &TagId() const { return view_.base.tag_id; }

  // ---- Vector-owned out-of-core tiers ------------------------------------
  // The DRAM mirror, the pinned-span HBM cache, and the transient stage
  // ring are tiers OF THE VECTOR: it allocates them, fills them through the
  // CTE (which stays the source of truth for bytes), owns the admission
  // policy and the copy stream, and publishes the mirror in its device
  // view. Callers only orchestrate WHICH spans a launch needs.

  /** Build the pinned-host DRAM tier under a byte budget. Full coverage
   *  (budget >= total): the arena is the flat decompressed image, filled
   *  through the CTE up front and published in DeviceViewBase::host_mirror
   *  for direct kernel reads. Partial coverage (ANY model size): the arena
   *  is a page-granular cache with clock eviction, filled on demand through
   *  the CTE, served ONLY through CopySpan/PinnedSpan -- kernels never see
   *  a raw pointer they could dereference out of coverage. Returns false
   *  (fault path remains fully functional) on any failure. */
  bool BuildHostTier(clio::run::u64 total_bytes, clio::run::u64 budget_bytes);
  /** Full-coverage convenience wrapper. */
  bool BuildHostMirror(clio::run::u64 total_bytes) {
    return BuildHostTier(total_bytes, total_bytes);
  }
  /** Stop and join the cache-manager thread. Idempotent. A process whose
   *  Vector is a leaked global MUST call this before exit (atexit works):
   *  otherwise the thread races CUDA teardown and its next
   *  CacheManagerKernel launch segfaults inside cudart. */
  void StopManager() {
#if !CTP_IS_DEVICE_PASS
    if (!impl_) return;
    impl_->cache_thread_run.store(false, std::memory_order_release);
    if (impl_->cache_thread.joinable()) impl_->cache_thread.join();
#endif
  }

  /** In-flight device fetches: issue many spans with FetchSpanDeviceAsync /
   *  PinnedSpan(..., &list), then WaitFetches ONCE for the whole batch --
   *  the server decompresses the pages CONCURRENTLY across its workers.
   *  Per-span blocking waits serialized the pipeline (measured
   *  dispatch-bound at ~1.3-3 ms/page). */
  using FetchList =
      std::vector<clio::run::Future<clio::cte::core::GetBlobTask>>;

  /** Issue the page-gets for [off, off+len) -> dst_dev WITHOUT waiting;
   *  futures are appended to fl. */
  bool FetchSpanDeviceAsync(unsigned char *dst_dev, clio::run::u64 off,
                            clio::run::u64 len, FetchList *fl);

  /** Wait every fetch in fl; true iff all succeeded. Clears fl. */
  bool WaitFetches(FetchList &fl) {
#if !CTP_IS_DEVICE_PASS
    bool ok = true;
    for (auto &f : fl) {
      f.Wait();
      if (f->GetReturnCode() != 0) ok = false;
    }
    fl.clear();
    return ok;
#else
    (void) fl;
    return false;
#endif
  }

  /** Fetch [off, off+len) of the model image STRAIGHT INTO DEVICE MEMORY
   *  through the CTE: per-page GetBlobs (16 in flight) routed via the
   *  storage pool, so a COMPRESSED tag's pages decompress on the GPU --
   *  kHBM-resident pages never cross PCIe at all (D2D + nvcomp), ram-tier
   *  pages cross compressed. Blocking: data is resident on return. */
  bool FetchSpanDevice(unsigned char *dst_dev, clio::run::u64 off,
                       clio::run::u64 len);

  /** Logical model bytes for span clamping when no DRAM tier is built
   *  (compressed mode fetches device-direct instead). */
  void SetLogicalBytes(clio::run::u64 n) {
#if !CTP_IS_DEVICE_PASS
    if (impl_ && impl_->mirror_bytes == 0) impl_->mirror_bytes = n;
#endif
  }

  /** MEASURED stored/original ratio, sampled from the first `n_sample`
   *  page blobs after assimilation. 0 when unknown (no pages, or the
   *  sizes are unavailable). Callers use this to size tiers by what the
   *  data ACTUALLY compresses to instead of a per-model guess. */
  double MeasuredRatio(clio::run::u64 n_sample = 24);

  /** Raise the scores of the pages spanning [off, off+len) so the CTE's
   *  organizer migrates them INTO the top tier (kHbm) before use. This is
   *  the prefetch primitive: a metadata op, never a data copy. Promotions
   *  are fire-and-forget with a bounded in-flight window; score 1.0 is
   *  required (a lower score puts the 1.0 tier in the DPE's fallback
   *  group). Demote with score < 0.9 when the span leaves the window. */
  void PromoteSpan(clio::run::u64 off, clio::run::u64 len, float score);
  /** Drain outstanding promotions (bounded; call before teardown). */
  void DrainPromotions();
  /** True when this page's cached record already shows a top-tier score,
   *  i.e. a promotion has landed and the zero-task read path can serve it. */
  bool PageIsPromoted(clio::run::u64 gp);

  /** Partial-mode page residency (see BuildHostTier). */
  unsigned char *EnsureHostPage(clio::run::u64 gp);

  /** True iff the host tier exists at all (full OR partial coverage). */
  bool HostTier() const {
#if !CTP_IS_DEVICE_PASS
    return impl_ && impl_->host_arena != nullptr;
#else
    return false;
#endif
  }
  const unsigned char *HostMirror() const {
#if !CTP_IS_DEVICE_PASS
    return impl_ ? impl_->host_mirror : nullptr;
#else
    return nullptr;
#endif
  }
  clio::run::u64 MirrorBytes() const {
#if !CTP_IS_DEVICE_PASS
    return impl_ ? impl_->mirror_bytes : 0;
#else
    return 0;
#endif
  }

  /** Size the VRAM span tiers: a transient ring of ring_bytes and a
   *  pinned-span slab of min(slab_cap_max, free - vram_reserve). Creates
   *  the copy stream + event pool. Halves each request until it fits. */
  bool ReserveSpanCaches(clio::run::u64 ring_bytes,
                         clio::run::u64 vram_reserve,
                         clio::run::u64 slab_cap_max);

  /** Stream every span copy runs on; wait launches on RecordCopyEvent(). */
  void *CopyStream() const {
#if !CTP_IS_DEVICE_PASS
    return impl_ ? (void *) impl_->copy_stream : nullptr;
#else
    return nullptr;
#endif
  }
  /** Record a pooled event on the copy stream and return it. */
  void *RecordCopyEvent();

  /** Look up / admit the span [off, off+len) of the mirror in the pinned
   *  cache. permanent=true admits on first touch with an exact allocation
   *  (static tensors); otherwise two-touch admission into the slab. On a
   *  miss that admits, the H2D copy is issued on CopyStream() and *copied
   *  is set. Returns nullptr when not admitted (caller stages transient
   *  or reads the mirror). */
  const unsigned char *PinnedSpan(clio::run::u64 off, clio::run::u64 len,
                                  bool permanent, bool *copied,
                                  FetchList *fl = nullptr);

  /** Bump-allocate len bytes of the transient ring. On wrap, fences the
   *  copy stream on main_stream progress so no in-flight consumer still
   *  reads the region being reused. */
  unsigned char *StageTransient(clio::run::u64 len, void *main_stream);

  /** Issue the H2D copy of mirror span [off,off+len) into dst on the copy
   *  stream. dst is normally a StageTransient() slice. */
  bool CopySpan(unsigned char *dst, clio::run::u64 off, clio::run::u64 len,
                FetchList *fl = nullptr);

  clio::run::u64 RingCap() const {
#if !CTP_IS_DEVICE_PASS
    return impl_ ? impl_->ring_cap : 0;
#else
    return 0;
#endif
  }

  /** Span-tier counters (pinned hits/misses, slab/static bytes). */
  void SpanStats(clio::run::u64 *hits, clio::run::u64 *misses,
                 clio::run::u64 *slab_used,
                 clio::run::u64 *static_bytes) const {
#if !CTP_IS_DEVICE_PASS
    if (hits) *hits = impl_ ? impl_->span_hits : 0;
    if (misses) *misses = impl_ ? impl_->span_misses : 0;
    if (slab_used) *slab_used = impl_ ? impl_->slab_used : 0;
    if (static_bytes) *static_bytes = impl_ ? impl_->static_bytes : 0;
#endif
  }

  /** Read a coherent snapshot of the pinned-host counters. Caller
   *  should cudaDeviceSync first so the device's atomicAdd writes
   *  are visible to the host through PCIe cache snooping. */
  ::clio::cte::gpu_vector::VectorStats StatsSnapshot() const {
    ::clio::cte::gpu_vector::VectorStats out{};
#if !CTP_IS_DEVICE_PASS
    if (impl_ && impl_->stats) out = *impl_->stats;
#endif
    return out;
  }
  /** Zero out all counters so the caller can measure a single phase.
   *  Caller should cudaDeviceSync afterward so any in-flight manager
   *  writes don't race with the host memset. */
  void StatsReset() {
#if !CTP_IS_DEVICE_PASS
    if (impl_ && impl_->stats) {
      std::memset(impl_->stats, 0,
                  sizeof(::clio::cte::gpu_vector::VectorStats));
    }
#endif
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  DeviceView<T> view_{};
};

namespace detail {

/** One-shot init kernel — non-template so nvcc reliably registers it
 *  through the standard <<<>>> launch glue. Operates on DeviceViewBase
 *  so it doesn't need T. Initializes BOTH tiers' Page slots, plus the
 *  per-block rescore/prefetch queue headers and swap_scratch slot.
 *
 *  Grid contract: <<<nblocks, threads_per_block>>>. Each block owns its
 *  Block struct and walks all its slots in stride. */
__global__ void InitMetaKernel(DeviceViewBase v, char *pages_base,
                                char *host_pages_base, char *swap_base) {
  if (blockIdx.x >= v.nblocks) return;
  Block *b = GetBlock(v, blockIdx.x);
  clio::run::u32 total = TotalPagesPerBlock(v);
  if (threadIdx.x == 0) {
    b->block_idx = blockIdx.x;
    b->num_modified = 0;
    b->gpu_pages_per_block = v.gpu_pages_per_block;
    b->host_pages_per_block = v.host_pages_per_block;
    b->swap_scratch = swap_base
        ? swap_base + static_cast<clio::run::u64>(blockIdx.x) * v.page_size_bytes
        : nullptr;
    b->flush_cursor = 0;
    b->rescore_q.head = 0;
    b->rescore_q.tail = 0;
    b->prefetch_q.head = 0;
    b->prefetch_q.tail = 0;
  }
  __syncthreads();
  for (clio::run::u32 s = threadIdx.x; s < total; s += blockDim.x) {
    Page *p = &b->pages[s];
    if (s < v.gpu_pages_per_block) {
      p->tier = 0;
      p->device_ptr = pages_base +
          (static_cast<clio::run::u64>(blockIdx.x) * v.gpu_pages_per_block + s) *
           v.page_size_bytes;
    } else {
      p->tier = 1;
      clio::run::u32 host_slot = s - v.gpu_pages_per_block;
      p->device_ptr = host_pages_base +
          (static_cast<clio::run::u64>(blockIdx.x) * v.host_pages_per_block +
           host_slot) * v.page_size_bytes;
    }
    p->page_idx = -1;
    p->modify_min = -1;
    p->modify_max = -1;
    p->flags = 0;
    p->lru_clock = 0;
    p->score = 0.0f;
    new (&p->active_put) clio::run::gpu::Future<clio::cte::core::PodPutBlobTask>();
    new (&p->active_get) clio::run::gpu::Future<clio::cte::core::PodGetBlobTask>();
  }
}

/** Legacy cache-management kernel — preserved verbatim from the
 *  pre-async-thread design for callers that opt into kLegacy mode.
 *  Walks every (block, slot) pair and submits PutBlob for any with a
 *  non-empty modify range. */
__global__ void LegacyFlushKernel(::clio::run::IpcManagerGpuInfo info,
                                   DeviceViewBase v) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  clio::run::u32 total_per_block = TotalPagesPerBlock(v);
  clio::run::u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  clio::run::u32 total = v.nblocks * total_per_block;
  if (idx >= total) return;
  clio::run::u32 b_idx = idx / total_per_block;
  clio::run::u32 slot = idx - b_idx * total_per_block;
  Block *b = GetBlock(v, b_idx);
  Page *p = &b->pages[slot];
  if (p->modify_min < 0) return;
  if (detail::AtomicOrU32(&p->flags, kPagePutInFlight) & kPagePutInFlight) {
    return;
  }
  FlushPageBase(g_ipc_manager_ptr, v, b_idx, p, slot);
  (void)g_ipc_manager;
}

/** Mark every HBM slot resident and clean, binding slot (block, s) to the
 *  global logical page (block * gpu_pages_per_block + s). Used by
 *  Vector::FaultAllSync after the host has decompressed each page's bytes
 *  into the slot's device_ptr; the read kernel's Resolve then finds every
 *  page present (page_idx set) instead of cold-faulting. Legacy/single-tier
 *  only. */
__global__ void MarkAllResidentKernel(DeviceViewBase v) {
  if (blockIdx.x >= v.nblocks) return;
  Block *b = GetBlock(v, blockIdx.x);
  for (clio::run::u32 s = threadIdx.x; s < v.gpu_pages_per_block;
       s += blockDim.x) {
    Page *p = &b->pages[s];
    p->page_idx = static_cast<int32_t>(
        static_cast<clio::run::u64>(blockIdx.x) * v.gpu_pages_per_block + s);
    p->modify_min = -1;
    p->modify_max = -1;
    p->flags = 0;
    p->tier = 0;
  }
}

/** Windowed variant for Vector::PrefetchWindowSync: single-block, bind slot s to
 *  global page (first_page + s) so the read kernel's Resolve finds the just-
 *  prefetched window resident. */
__global__ void MarkWindowResidentKernel(DeviceViewBase v,
                                         clio::run::u64 first_page) {
  Block *b = GetBlock(v, 0);
  for (clio::run::u32 s = threadIdx.x; s < v.gpu_pages_per_block;
       s += blockDim.x) {
    Page *p = &b->pages[s];
    p->page_idx = static_cast<int32_t>(first_page + s);
    p->modify_min = -1;
    p->modify_max = -1;
    p->flags = 0;
    p->tier = 0;
  }
}

/** Bind slots [slot_base, slot_base+count) to global pages
 *  [first_page, first_page+count). Used by Vector::PrefetchPagesSync for
 *  double-buffered (pipelined) prefetch: two windows live in disjoint slot
 *  ranges, so the next window's prefetch never touches the window being read. */
__global__ void MarkPagesResidentKernel(DeviceViewBase v,
                                        clio::run::u64 first_page,
                                        clio::run::u32 count,
                                        clio::run::u32 slot_base) {
  Block *b = GetBlock(v, 0);
  for (clio::run::u32 i = threadIdx.x; i < count; i += blockDim.x) {
    Page *p = &b->pages[slot_base + i];
    p->page_idx = static_cast<int32_t>(first_page + i);
    p->modify_min = -1;
    p->modify_max = -1;
    p->flags = 0;
    p->tier = 0;
  }
}

/** Atomic-clear kPageBusy + kPageGetInFlight on a single (block, slot).
 *  Used by the cache thread after a host-driven prefetch's AsyncGetBlob
 *  completes (the data has landed in the page; we just need to release
 *  the slot lock so the user kernel can use it). */
__global__ void ClearHostPrefetchFlagsKernel(DeviceViewBase v,
                                              clio::run::u32 block_idx,
                                              clio::run::u32 slot_idx) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  Block *b = GetBlock(v, block_idx);
  Page *p = &b->pages[slot_idx];
  AtomicClearBitsU32(&p->flags, kPageBusy | kPageGetInFlight);
}

/** Batched clear: takes a fixed-size list of (block, slot) pairs and
 *  clears kPageBusy+kPageGetInFlight on each. Launched as a single
 *  kernel per drain pass so we don't hit the multi-launch hang
 *  observed when many ClearHostPrefetchFlagsKernel calls are queued
 *  in sequence on the cache-thread stream while the user kernel is
 *  spin-waiting on FaultPage futures. */
inline constexpr clio::run::u32 kClearBatchCap = 256;
__global__ void ClearHostPrefetchFlagsBatchKernel(DeviceViewBase v,
                                                   clio::run::u32 *block_arr,
                                                   clio::run::u32 *slot_arr,
                                                   clio::run::u32 n) {
  clio::run::u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  Block *b = GetBlock(v, block_arr[idx]);
  Page *p = &b->pages[slot_arr[idx]];
  AtomicClearBitsU32(&p->flags, kPageBusy | kPageGetInFlight);
}

/** Drain kernel — non-template. Waits on every page's active_put /
 *  active_get and clears the slots. */
__global__ void DrainKernel(::clio::run::IpcManagerGpuInfo info, DeviceViewBase v) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  clio::run::u32 total_per_block = TotalPagesPerBlock(v);
  clio::run::u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  clio::run::u32 total = v.nblocks * total_per_block;
  if (idx >= total) return;
  clio::run::u32 b_idx = idx / total_per_block;
  clio::run::u32 slot = idx - b_idx * total_per_block;
  Page *p = &GetBlock(v, b_idx)->pages[slot];
  DrainPut(p);
  DrainGet(p);
  (void)g_ipc_manager;
}

/** Warp-parallel reduce: each thread feeds a (value, index) pair.
 *  Returns the lane with the minimum value via shuffles. Result is
 *  broadcast to all lanes. */
CTP_GPU_FUN void WarpReduceMinScore(float &val, int &idx) {
  for (int off = 16; off > 0; off >>= 1) {
    float other_val = __shfl_xor_sync(0xffffffff, val, off);
    int other_idx = __shfl_xor_sync(0xffffffff, idx, off);
    if (other_val < val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

CTP_GPU_FUN void WarpReduceMaxScore(float &val, int &idx) {
  for (int off = 16; off > 0; off >>= 1) {
    float other_val = __shfl_xor_sync(0xffffffff, val, off);
    int other_idx = __shfl_xor_sync(0xffffffff, idx, off);
    if (other_val > val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

CTP_GPU_FUN clio::run::u64 WarpReduceMinU64(clio::run::u64 val) {
  for (int off = 16; off > 0; off >>= 1) {
    clio::run::u64 hi = __shfl_xor_sync(0xffffffff, (unsigned int)(val >> 32), off);
    clio::run::u64 lo = __shfl_xor_sync(0xffffffff, (unsigned int)(val & 0xffffffffu), off);
    clio::run::u64 other = (hi << 32) | lo;
    if (other < val) val = other;
  }
  return val;
}

CTP_GPU_FUN clio::run::u64 WarpReduceMaxU64(clio::run::u64 val) {
  for (int off = 16; off > 0; off >>= 1) {
    clio::run::u64 hi = __shfl_xor_sync(0xffffffff, (unsigned int)(val >> 32), off);
    clio::run::u64 lo = __shfl_xor_sync(0xffffffff, (unsigned int)(val & 0xffffffffu), off);
    clio::run::u64 other = (hi << 32) | lo;
    if (other > val) val = other;
  }
  return val;
}

/** Scan one tier of a Block's pages[] and report the slot with the
 *  smallest score (skipping kPageBusy, empty slots, and in-flight). */
CTP_GPU_FUN void TierMinScore(Block *b, clio::run::u32 lo, clio::run::u32 hi,
                                clio::run::u32 lane,
                                float &out_min, int &out_slot) {
  float best_v = INFINITY;
  int best_s = -1;
  for (clio::run::u32 s = lo + lane; s < hi; s += 32) {
    Page *p = &b->pages[s];
    if (p->page_idx < 0) continue;
    if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
    if (p->score < best_v) { best_v = p->score; best_s = (int)s; }
  }
  WarpReduceMinScore(best_v, best_s);
  out_min = best_v;
  out_slot = best_s;
}

CTP_GPU_FUN void TierMaxScore(Block *b, clio::run::u32 lo, clio::run::u32 hi,
                                clio::run::u32 lane,
                                float &out_max, int &out_slot) {
  float best_v = -INFINITY;
  int best_s = -1;
  for (clio::run::u32 s = lo + lane; s < hi; s += 32) {
    Page *p = &b->pages[s];
    if (p->page_idx < 0) continue;
    if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
    if (p->score > best_v) { best_v = p->score; best_s = (int)s; }
  }
  WarpReduceMaxScore(best_v, best_s);
  out_max = best_v;
  out_slot = best_s;
}

/** Warp-cooperative byte copy: copy `bytes` from src to dst via 4-byte
 *  loads/stores spread across 32 lanes. Both pointers must be 4-byte
 *  aligned and `bytes` must be a multiple of 4. */
CTP_GPU_FUN void WarpCopy4(void *dst, const void *src, clio::run::u64 bytes,
                             clio::run::u32 lane) {
  clio::run::u64 n = bytes >> 2;
  unsigned int *d = static_cast<unsigned int *>(dst);
  const unsigned int *s = static_cast<const unsigned int *>(src);
  for (clio::run::u64 i = lane; i < n; i += 32) d[i] = s[i];
}

/**
 * Four-phase async cache manager. Launched as <<<nblocks, warpsize>>>
 * where `nblocks` matches the user kernel grid. Each manager block
 * services one user Block.
 *
 * Phase 1 — Rescore:
 *   1a. warp-reduce min/max(lru_clock) across non-empty slots.
 *   1b. p->score = (lru_clock - min_c) / max(1, max_c - min_c);
 *       empty slots -> score = -INF (won't beat anything in min/max).
 *   1c. lane 0 drains rescore_q; matching cached pages get their score
 *       overwritten; non-matching entries forward to prefetch_q (drop if
 *       full).
 *   1d. (computed lazily in Phase 4 via warp-parallel rescan.)
 * Phase 2 — Reorganize:
 *   If min(DRAM) > min(HBM) AND there exists an HBM slot below min(DRAM)
 *   AND a DRAM slot above min(HBM), CAS-acquire kPageBusy on both, drain
 *   their puts, swap their contents via swap_scratch in a 3-step
 *   warp-cooperative copy, swap metadata (page_idx, modify_min/max,
 *   score, lru_clock, active_put, active_get) — tier stays slot-bound.
 *   Release kPageBusy.
 * Phase 3 — Flush:
 *   For every page in both tiers, if dirty and not busy/in-flight,
 *   CAS-acquire kPagePutInFlight and call FlushPageBase. No Wait.
 * Phase 4 — Prefetch:
 *   Pop prefetch_q. For each entry: warp-parallel rescan to find the
 *   current min in each tier. If e.score >= min_hbm evict that HBM
 *   slot and FaultPage. Else if e.score >= min_dram evict that DRAM
 *   slot and FaultPage. Else drop. Bounded to a few pops per tick.
 */
/**
 * Run a single Phase 1→2→3→4 pass on one block. Returns the number of
 * "units of work" performed (rescore drained + flushes kicked + prefetches
 * claimed + swaps). 0 means the block was idle this tick. Used by the
 * persistent kernel to decide whether to back off / self-terminate.
 */
CTP_GPU_FUN clio::run::u32 RunCachePass(::clio::run::gpu::IpcManager *ipc,
                                    DeviceViewBase v, Block *b,
                                    clio::run::u32 lane, clio::run::u32 total) {
  clio::run::u32 work = 0;
  (void)ipc;
  if (v.stats && lane == 0) atomicAdd_system(&v.stats->manager_iters, 1ULL);

  // ── Phase 1a: warp-reduce min/max clock ─────────────────────────
  clio::run::u64 my_min = ~0ULL;
  clio::run::u64 my_max = 0ULL;
  for (clio::run::u32 s = lane; s < total; s += 32) {
    Page *p = &b->pages[s];
    if (p->page_idx < 0) continue;
    if (p->lru_clock < my_min) my_min = p->lru_clock;
    if (p->lru_clock > my_max) my_max = p->lru_clock;
  }
  clio::run::u64 min_c = WarpReduceMinU64(my_min);
  clio::run::u64 max_c = WarpReduceMaxU64(my_max);
  float range = (max_c > min_c) ? (float)(max_c - min_c) : 1.0f;

  // ── Phase 1b: normalize ─────────────────────────────────────────
  for (clio::run::u32 s = lane; s < total; s += 32) {
    Page *p = &b->pages[s];
    if (p->page_idx < 0) { p->score = -INFINITY; continue; }
    p->score = (float)(p->lru_clock - min_c) / range;
  }
  __syncwarp();

  // ── Phase 1c: drain rescore_q ───────────────────────────────────
  // CRITICAL: snapshot tail at start. The user kernel produces into
  // tail concurrently; without snapshotting, a fast producer can keep
  // tail ahead of head forever and this loop never terminates.
  clio::run::u32 phase1c_work = 0;
  if (lane == 0) {
    clio::run::u32 snap_tail = b->rescore_q.tail;
    while (b->rescore_q.head != snap_tail) {
      RescoreEntry e =
          b->rescore_q.slots[b->rescore_q.head & (kRescoreQueueCap - 1)];
      ++b->rescore_q.head;
      ++phase1c_work;
      bool matched = false;
      for (clio::run::u32 s = 0; s < total; ++s) {
        Page *p = &b->pages[s];
        if (p->page_idx == (int32_t)e.page_idx) {
          // Score sign is only used to signal alloc-vs-prefetch on
          // unmatched entries (Phase 4). For already-cached pages we
          // just update the cache score using the magnitude.
          if (!(p->flags & kPageBusy)) {
            p->score = (e.score < 0.0f) ? -e.score : e.score;
          }
          matched = true;
          break;
        }
      }
      if (!matched) {
        clio::run::u32 used = b->prefetch_q.tail - b->prefetch_q.head;
        if (used < kPrefetchQueueCap) {
          b->prefetch_q.slots[b->prefetch_q.tail & (kPrefetchQueueCap - 1)] = e;
          ++b->prefetch_q.tail;
        }
      }
    }
  }
  phase1c_work = __shfl_sync(0xffffffff, phase1c_work, 0);
  work += phase1c_work;
  if (v.stats && lane == 0 && phase1c_work > 0)
    atomicAdd_system(&v.stats->phase1c_drained, (unsigned long long)phase1c_work);
  __syncwarp();

  // ── Phase 2: Reorganize — promote hot DRAM pages to HBM ────────
  // Disabled by default (CLIO_CTE_PHASE2=1 to re-enable). Phase 2's
  // 3-step swap holds kPageBusy on both an HBM and a DRAM slot for
  // ~240 µs; the user kernel's read_range spin-waits on those flags
  // and falls into the cold-miss path, dwarfing whatever benefit
  // promotion provides. For streaming workloads, leave pages where
  // Phase 4 placed them.
#if defined(CLIO_CTE_PHASE2)
  if (v.host_pages_per_block > 0) {
    clio::run::u32 max_swaps_this_tick = v.gpu_pages_per_block;
    for (clio::run::u32 si = 0; si < max_swaps_this_tick; ++si) {
      float min_hbm; int min_hbm_slot;
      TierMinScore(b, 0, v.gpu_pages_per_block, lane, min_hbm, min_hbm_slot);
      float max_dram; int max_dram_slot;
      TierMaxScore(b, v.gpu_pages_per_block, total, lane,
                   max_dram, max_dram_slot);
      int do_swap = 0;
      if (lane == 0 && min_hbm_slot >= 0 && max_dram_slot >= 0 &&
          max_dram > min_hbm) {
        Page *ph = &b->pages[min_hbm_slot];
        Page *pd = &b->pages[max_dram_slot];
        if (TryAcquireBusy(ph)) {
          if (TryAcquireBusy(pd)) {
            DrainPut(ph); DrainPut(pd);
            DrainGet(ph); DrainGet(pd);
            do_swap = 1;
          } else {
            ReleaseBusy(ph);
          }
        }
      }
      do_swap = __shfl_sync(0xffffffff, do_swap, 0);
      if (!do_swap) break;
      Page *ph = &b->pages[min_hbm_slot];
      Page *pd = &b->pages[max_dram_slot];
      WarpCopy4(b->swap_scratch, ph->device_ptr, v.page_size_bytes, lane);
      __syncwarp();
      WarpCopy4(ph->device_ptr, pd->device_ptr, v.page_size_bytes, lane);
      __syncwarp();
      WarpCopy4(pd->device_ptr, b->swap_scratch, v.page_size_bytes, lane);
      __syncwarp();
      if (lane == 0) {
        int32_t pi = ph->page_idx;   ph->page_idx   = pd->page_idx;   pd->page_idx   = pi;
        int32_t mn = ph->modify_min; ph->modify_min = pd->modify_min; pd->modify_min = mn;
        int32_t mx = ph->modify_max; ph->modify_max = pd->modify_max; pd->modify_max = mx;
        clio::run::u64 lc = ph->lru_clock; ph->lru_clock  = pd->lru_clock;  pd->lru_clock  = lc;
        float sc = ph->score;        ph->score      = pd->score;      pd->score      = sc;
        __threadfence();
        ReleaseBusy(ph);
        ReleaseBusy(pd);
      }
      ++work;
      if (v.stats && lane == 0) atomicAdd_system(&v.stats->phase2_swaps, 1ULL);
      __syncwarp();
    }
  }
#endif  // CLIO_CTE_PHASE2

  // ── Phase 3: Flush dirty unlocked pages (bounded) ──────────────
  // Bound work per tick so we don't saturate the gpu2cpu queue
  // while the user kernel is also pushing PutBlob/GetBlob entries
  // (the queue is system-scoped 64-bit-atomic so producer Sends
  // contend on the tail). vec.flush() handles the rest synchronously.
  // Phase 3: flush dirty unlocked pages, capped to keep the gpu2cpu
  // queue from saturating (the manager's Sends contend with the user
  // kernel's). Cap = host_pages_per_block so we can flush the full
  // DRAM tier in one tick if needed.
  clio::run::u32 phase3_work = 0;
  if (lane == 0) {
    const clio::run::u32 max_flush_per_tick = v.host_pages_per_block > 0
                                            ? v.host_pages_per_block
                                            : v.gpu_pages_per_block;
    clio::run::u32 s = b->flush_cursor;
    for (clio::run::u32 i = 0; i < total && phase3_work < max_flush_per_tick; ++i) {
      Page *p = &b->pages[s];
      if (p->page_idx >= 0 && p->modify_min >= 0 &&
          !(p->flags & kPageBusy) &&
          !(AtomicOrU32(&p->flags, kPagePutInFlight) & kPagePutInFlight)) {
        FlushPageBase(ipc, v, blockIdx.x, p, s);
        ++phase3_work;
      }
      s = (s + 1u) % total;
    }
    b->flush_cursor = s;
  }
  phase3_work = __shfl_sync(0xffffffff, phase3_work, 0);
  work += phase3_work;
  if (v.stats && lane == 0 && phase3_work > 0)
    atomicAdd_system(&v.stats->phase3_flushes, (unsigned long long)phase3_work);
  __syncwarp();

  // ── Phase 4: Prefetch into DRAM tier ONLY ──────────────────────
  // GPU-side Send from the manager kernel races with user kernel's
  // Sends on the gpu2cpu queue tail (system-scoped 64-bit atomic
  // contention crashes the GPU). So Phase 4 is host-driven:
  //   1. Kernel picks a free/lowest-score DRAM slot (NOT HBM —
  //      prefetching into HBM would steal a slot the user kernel
  //      may be actively using). User kernel can read DRAM pages
  //      directly via Resolve's two-tier scan; they're slower but
  //      free up HBM for hot pages.
  //   2. Kernel CAS-acquires kPageBusy on the DRAM victim, sets
  //      page_idx + kPageGetInFlight, pushes a directive into the
  //      pinned-host host_prefetch_q. Does NOT release kPageBusy.
  //   3. Cache thread (post-kernel) issues AsyncGetBlob via the
  //      CPU client (separate cpu2cpu path, no system-atomic
  //      contention), waits, then launches a tiny clear-flags
  //      kernel on the cache-thread's non-blocking stream to
  //      release kPageBusy + kPageGetInFlight.
  // Phase 4: device-side prefetch claim. Kernel acquires kPageBusy on a
  // free/low-score DRAM slot, sets page_idx + kPageGetInFlight on the
  // device. Does NOT write to pinned host memory (that empirically
  // hangs the user kernel — see git log for the iter-9 debug trace).
  // The cache thread will later cudaMemcpyAsync the meta region to a
  // host scratch, scan for kPageGetInFlight slots, issue AsyncGetBlob
  // per slot, and launch a clear-flags kernel to release the slot
  // when each fault completes.
  // Phase 4: DRAM-tier prefetch. The kernel acquires kPageBusy on a
  // DRAM slot and sets kPageGetInFlight; the cache thread (CPU)
  // releases both flags after the AsyncGetBlob completes via
  // DrainHostPrefetchQueue. Bounded per tick to avoid claiming all
  // DRAM slots and starving the user kernel.
  if (v.host_pages_per_block > 0 && lane == 0) {
    // Prefetch depth scales with DRAM tier capacity so the manager can
    // populate the entire DRAM tier in a single tick when there's a
    // backlog. Bounded by kPrefetchQueueCap (32) naturally.
    const int kMaxPrefetchPerTick =
        static_cast<int>(v.host_pages_per_block);
    for (int i = 0; i < kMaxPrefetchPerTick; ++i) {
      clio::run::u32 pq_used = b->prefetch_q.tail - b->prefetch_q.head;
      if (pq_used == 0) break;
      RescoreEntry e =
          b->prefetch_q.slots[b->prefetch_q.head & (kPrefetchQueueCap - 1)];
      ++b->prefetch_q.head;
      if (v.stats) atomicAdd_system(&v.stats->phase4_pops, 1ULL);
      // Already cached? Skip.
      bool already_cached = false;
      for (clio::run::u32 s = 0; s < total; ++s) {
        if (b->pages[s].page_idx == (int32_t)e.page_idx) {
          already_cached = true; break;
        }
      }
      if (already_cached) {
        if (v.stats) atomicAdd_system(&v.stats->phase4_skip_cached, 1ULL);
        continue;
      }
      // Target HBM first when EvictSlot is disabled (no race against
      // Resolve's `p->flags = 0` after EvictSlot). Falls back to DRAM
      // if no HBM slot is acquirable. When EvictSlot is on, we have
      // to limit to DRAM to avoid the slot-clobber race.
      clio::run::u32 tier_lo = v.allow_cold_miss_fault
                            ? v.gpu_pages_per_block
                            : 0;
      int target = -1;
      float lo_score = INFINITY;
      for (clio::run::u32 s = tier_lo; s < total; ++s) {
        Page *p = &b->pages[s];
        if (p->flags & kPageBusy) continue;
        if (p->page_idx < 0) {
          target = (int)s;
          break;
        }
        if (p->score < lo_score) {
          lo_score = p->score;
          target = (int)s;
        }
      }
      if (target < 0) {
        if (v.stats) atomicAdd_system(&v.stats->phase4_skip_nofree, 1ULL);
        continue;
      }
      Page *p = &b->pages[target];
      if (!TryAcquireBusy(p)) {
        if (v.stats) atomicAdd_system(&v.stats->phase4_skip_nofree, 1ULL);
        continue;
      }
      DrainPut(p);
      DrainGet(p);
      p->page_idx = (int32_t)e.page_idx;
      p->modify_min = -1;
      p->modify_max = -1;
      p->lru_clock = clock64();
      // Negative score = "alloc-only" (write hint): set up the slot
      // but skip the AsyncGetBlob (no data exists yet). Release
      // kPageBusy so the user kernel can claim the slot immediately.
      // Positive score = read prefetch: leave kPageBusy + set
      // kPageGetInFlight; cache thread issues AsyncGetBlob and clears.
      if (e.score < 0.0f) {
        p->score = -e.score;            // store positive score in slot
        // Fence so the user kernel (concurrent on a different SM)
        // sees the page_idx / modify_min/max stores before the
        // ReleaseBusy clears kPageBusy.
        __threadfence();
        ReleaseBusy(p);
      } else {
        p->score = e.score;
        __threadfence();
        AtomicOrU32(&p->flags, kPageGetInFlight);
      }
      if (v.stats) atomicAdd_system(&v.stats->phase4_prefetches, 1ULL);
      ++work;
    }
  }
  if (v.stats && lane == 0 && work > 0)
    atomicAdd_system(&v.stats->manager_work_iters, 1ULL);
  (void)g_ipc_manager;
  return work;
}

/**
 * Persistent cache-manager kernel. Launched ONCE per Vector lifetime on
 * a non-blocking stream. Each block loops, running RunCachePass; when a
 * pass finds no work the block backs off exponentially (base→max
 * nanoseconds via `__nanosleep`); after a contiguous idle period of
 * `idle_exit_ns`, the block self-terminates. The kernel as a whole
 * exits when all blocks have terminated OR when the host sets the
 * `stop_flag` (pinned host u32, polled via volatile read each iter).
 *
 * The host does NOT relaunch this kernel each tick — a single launch
 * powers an entire Vector's lifetime, plus optional resurrection on
 * Vector dtor or if the host detects new rescore_q activity post-exit.
 */
__global__ void PersistentCacheManagerKernel(
    ::clio::run::IpcManagerGpuInfo info,
    DeviceViewBase v,
    clio::run::u32 *stop_flag_pinned,
    clio::run::u32 base_sleep_ns,
    clio::run::u32 max_sleep_ns,
    clio::run::u64 idle_exit_ns) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (blockIdx.x >= v.nblocks) return;
  Block *b = GetBlock(v, blockIdx.x);
  clio::run::u32 total = TotalPagesPerBlock(v);
  clio::run::u32 lane = threadIdx.x & 31;
  clio::run::u32 cur_sleep_ns = base_sleep_ns;
  clio::run::u64 idle_ns_total = 0;
  volatile clio::run::u32 *stop_flag = stop_flag_pinned;

  while (true) {
    clio::run::u32 stop = 0;
    if (lane == 0) stop = *stop_flag;
    stop = __shfl_sync(0xffffffff, stop, 0);
    if (stop) break;

    clio::run::u32 work = RunCachePass(g_ipc_manager_ptr, v, b, lane, total);

    if (work > 0) {
      cur_sleep_ns = base_sleep_ns;
      idle_ns_total = 0;
    } else {
      idle_ns_total += cur_sleep_ns;
      if (idle_ns_total >= idle_exit_ns) break;
      clio::run::u32 doubled = cur_sleep_ns * 2u;
      cur_sleep_ns = (doubled > max_sleep_ns) ? max_sleep_ns : doubled;
    }
    __nanosleep(cur_sleep_ns);
    __syncwarp();
  }
  (void)g_ipc_manager;
}

/**
 * Original one-shot cache-manager kernel, kept for FlushAllSync's drain
 * path (LegacyFlushKernel covers that already) and as a fallback if the
 * persistent kernel can't run (compute_cap < sm_70). New code should
 * launch PersistentCacheManagerKernel.
 */
__global__ void CacheManagerKernel(::clio::run::IpcManagerGpuInfo info,
                                    DeviceViewBase v) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (blockIdx.x >= v.nblocks) return;
  Block *b = GetBlock(v, blockIdx.x);
  clio::run::u32 total = TotalPagesPerBlock(v);
  clio::run::u32 lane = threadIdx.x & 31;
  (void)RunCachePass(g_ipc_manager_ptr, v, b, lane, total);
  (void)g_ipc_manager;
}

/** Compute the byte stride between adjacent Block structs in the meta
 *  backend, accounting for the flexible Page array and total tier
 *  count. Aligned to 16. */
inline clio::run::u32 BlockStrideBytes(clio::run::u32 total_pages_per_block) {
  size_t raw = sizeof(Block) + sizeof(Page) * total_pages_per_block;
  return static_cast<clio::run::u32>((raw + 15) & ~static_cast<size_t>(15));
}

/** sizeof(TaskT), 16-byte aligned. The Task is self-contained (its fut_ holds
 *  the completion record), so there is no co-located gpu::FutureShm. */
template <typename TaskT>
inline clio::run::u32 TaskSlotStride() {
  size_t raw = sizeof(TaskT);
  return static_cast<clio::run::u32>((raw + 15) & ~static_cast<size_t>(15));
}

}  // namespace detail

template <typename T>
struct Vector<T>::Impl {
  clio::run::u32 gpu_id = 0;
  clio::run::u32 cache_period_us = 50;
  clio::run::u32 manager_threads_per_block = 32;
  CacheMode mode = CacheMode::kLegacy;
  ctp::ipc::AllocatorId pages_alloc_id;
  ctp::ipc::AllocatorId host_pages_alloc_id;
  ctp::ipc::AllocatorId meta_alloc_id;
  ctp::ipc::AllocatorId swap_alloc_id;
  ctp::ipc::AllocatorId put_alloc_id;
  ctp::ipc::AllocatorId get_alloc_id;
  char *pages_base = nullptr;
  char *host_pages_base = nullptr;
  char *meta_base = nullptr;
  char *swap_base = nullptr;
  char *put_base = nullptr;
  char *get_base = nullptr;
  /** Pinned host scratch (block_stride * nblocks bytes). The cache
   *  thread cudaMemcpyAsync's the device meta_base here every tick so
   *  it can scan for kPageGetInFlight slots and issue AsyncGetBlobs
   *  on the host side (the manager kernel only marks slots and
   *  doesn't issue Sends or pinned-host writes). */
  char *host_meta_scratch = nullptr;
  clio::run::u32 block_stride_cached = 0;
  /** Pinned-host index arrays sized kClearBatchCap; cache thread fills
   *  them with (block, slot) pairs of slots whose prefetch completed,
   *  then launches ClearHostPrefetchFlagsBatchKernel once per tick to
   *  release the locks. Launching one batched kernel avoids the
   *  multi-launch hang observed with per-slot launches. */
  clio::run::u32 *clear_block_arr = nullptr;
  clio::run::u32 *clear_slot_arr = nullptr;
  /** Pinned-host instrumentation counters. The user kernel and the
   *  manager kernel atomicAdd_system into these so the host can read
   *  a coherent snapshot via Vector::GetStatsSnapshot(). */
  ::clio::cte::gpu_vector::VectorStats *stats = nullptr;
  /** Number of host-driven AsyncGetBlobs currently in flight. Cache
   *  thread bumps before dispatching, decrements after the .Wait()
   *  returns. Vector dtor waits for this to reach 0 before freeing
   *  backends — without this gate, the runtime's cudaMemcpyAsync may
   *  reference a freed device_ptr and trip CUDA error 700. */
  std::atomic<clio::run::u32> async_inflight{0};
  /** Pinned-host u32 read by the persistent CacheManagerKernel each
   *  iteration. Host sets to 1 to signal the kernel to exit. The
   *  kernel can also self-terminate after the idle window elapses
   *  (in which case the host detects via cudaStreamQuery). */
  clio::run::u32 *kernel_stop_flag = nullptr;
  /** CUDA stream the persistent kernel runs on. Non-blocking so it
   *  doesn't depend on the main thread's default-stream work. */
  cudaStream_t persistent_stream = nullptr;
  /** Separate non-blocking stream for DrainHostPrefetchQueue's
   *  cudaMemcpyAsync + clear-flags kernel. Distinct from
   *  persistent_stream so meta snapshots don't queue behind the
   *  persistent kernel (which never returns until idle-exit). */
  cudaStream_t drain_stream = nullptr;
  /** True iff PersistentCacheManagerKernel is currently in flight on
   *  persistent_stream. Cache thread (re)launches when this is false
   *  and there is work that needs handling. */
  bool persistent_kernel_running = false;
  std::string tag_name;
  // ---- Vector-owned out-of-core tiers (see BuildHostMirror et al.) ----
  /** Pinned-host DRAM tier. FULL coverage (arena holds every model page in
   *  order): host_mirror == host_arena and spans are raw pointer math.
   *  PARTIAL coverage (model larger than the host budget): host_mirror
   *  stays null, the arena is a page-granular cache with clock eviction,
   *  and spans are served through EnsureHostPage/CopySpan only -- which is
   *  what makes the tier correct for a model of ANY size. */
  unsigned char *host_mirror = nullptr;
  clio::run::u64 mirror_bytes = 0;   /**< logical model bytes (both modes). */
  unsigned char *host_arena = nullptr;
  clio::run::u64 host_slots = 0;     /**< pinned pages in the arena. */
  std::vector<long long> host_page_slot;       /**< model page -> slot|-1. */
  std::vector<long long> host_slot_page;       /**< slot -> model page|-1. */
  std::vector<uint8_t> host_slot_ref;          /**< clock reference bits. */
  clio::run::u64 host_clock = 0;
  clio::run::u64 host_tier_hits = 0, host_tier_fills = 0;
  /** Pinned-span HBM cache slab (bump-allocated, two-touch admission). */
  unsigned char *span_slab = nullptr;
  clio::run::u64 slab_cap = 0, slab_used = 0;
  /** Transient stage ring for spans read exactly once. */
  unsigned char *stage_ring = nullptr;
  clio::run::u64 ring_cap = 0, ring_head = 0;
  /** Copy stream all span DMA runs on, + reusable event pool. */
  cudaStream_t copy_stream = nullptr;
  cudaEvent_t copy_events[256] = {};
  clio::run::u32 copy_ev_idx = 0;
  /** Span cache: flat offset -> (device pointer, span length). Length is
   *  part of the contract: page-aligned spans of ADJACENT tensors can
   *  share a start offset, and returning a shorter cached buffer for a
   *  longer request let the kernel read past the allocation (illegal
   *  memory access on the compressed llama path). */
  std::unordered_map<clio::run::u64,
                     std::pair<unsigned char *, clio::run::u64>> span_cache;
  std::unordered_map<clio::run::u64, clio::run::u32> span_seen;
  clio::run::u64 span_hits = 0, span_misses = 0;
  /** In-flight score promotions (see PromoteSpan). */
  std::deque<clio::run::Future<clio::cte::core::ReorganizeBlobTask>> promos;
  /** Permanent (first-touch) admissions get exact cudaMallocs under
   *  their own budget -- static model tensors, reused every token. */
  clio::run::u64 static_bytes = 0, static_cap = 3ull << 30;
  /** Page->family policy (see Vector ctor). 0 => single family b0. */
  clio::run::u64 fam_ppb = 0;
  /** Optional full-name override for host-side page name composition. */
  std::function<std::string(clio::run::u64)> host_namer;
  bool flat_layout = false;   /**< model stored as one flat blob "w". */
  clio::run::PoolId cte_pool_id = clio::run::PoolId(0, 0);
  std::thread cache_thread;
  std::atomic<bool> cache_thread_run{false};
  clio::run::u32 nblocks_cached = 0;
  clio::run::u64 page_size_cached = 0;
  clio::run::u32 gpu_ppb_cached = 0;
  clio::run::u32 host_ppb_cached = 0;
};

template <typename T>
inline Vector<T>::Vector(const std::string &tag_name, clio::run::u32 nblocks,
                          clio::run::u32 gpu_id, clio::run::u32 gpu_pages_per_block,
                          clio::run::u32 host_pages_per_block,
                          clio::run::u64 page_size_bytes,
                          clio::run::u32 cache_period_us,
                          CacheMode mode,
                          clio::run::u32 manager_threads_per_block,
                          bool allow_cold_miss_fault,
                          clio::run::PoolId storage_pool_id,
                          clio::run::u64 family_pages,
                          std::function<std::string(clio::run::u64)> host_namer,
                          bool flat_layout) {
#if !CTP_IS_DEVICE_PASS
  // Body gated for the host pass only.
  if (nblocks == 0 || gpu_pages_per_block == 0 || page_size_bytes == 0) {
    throw std::invalid_argument(
        "clio::cte::gpu_vector::Vector: nblocks/gpu_pages_per_block/page_size "
        "must all be > 0");
  }
  if (page_size_bytes % sizeof(T) != 0) {
    throw std::invalid_argument(
        "clio::cte::gpu_vector::Vector: page_size_bytes must be a multiple "
        "of sizeof(T)");
  }
  if (manager_threads_per_block == 0 ||
      (manager_threads_per_block % 32) != 0) {
    throw std::invalid_argument(
        "clio::cte::gpu_vector::Vector: manager_threads_per_block must be a "
        "positive multiple of 32");
  }
  // Auto-promote mode when DRAM tier is requested.
  if (host_pages_per_block > 0 && mode == CacheMode::kLegacy) {
    mode = CacheMode::kAsync;
  }
  impl_ = std::make_unique<Impl>();
  impl_->cache_period_us = cache_period_us;
  impl_->manager_threads_per_block = manager_threads_per_block;
  impl_->mode = mode;
  impl_->gpu_id = gpu_id;

  auto *cpu_ipc = CLIO_CPU_IPC;

  clio::run::u32 total_ppb = gpu_pages_per_block + host_pages_per_block;

  // 1. Allocate the HBM page backend.
  clio::run::u64 hbm_bytes = static_cast<clio::run::u64>(nblocks) * gpu_pages_per_block *
                       page_size_bytes;
  impl_->pages_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem, hbm_bytes,
      &impl_->pages_base);
  if (impl_->pages_alloc_id.IsNull()) {
    throw std::runtime_error("gpu_vector: HBM pages backend allocation failed");
  }

  // 1b. Allocate the DRAM (pinned host) page backend if requested.
  if (host_pages_per_block > 0) {
    clio::run::u64 dram_bytes = static_cast<clio::run::u64>(nblocks) *
                          host_pages_per_block * page_size_bytes;
    impl_->host_pages_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost, dram_bytes,
        &impl_->host_pages_base);
    if (impl_->host_pages_alloc_id.IsNull()) {
      throw std::runtime_error(
          "gpu_vector: DRAM (pinned-host) pages backend allocation failed");
    }
  } else {
    impl_->host_pages_alloc_id = ctp::ipc::AllocatorId::GetNull();
    impl_->host_pages_base = nullptr;
  }

  // 1c. Allocate the swap scratch (one HBM page per block) for in-kernel
  //     reorganize. Only needed when DRAM tier is active.
  if (host_pages_per_block > 0) {
    clio::run::u64 swap_bytes =
        static_cast<clio::run::u64>(nblocks) * page_size_bytes;
    impl_->swap_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem, swap_bytes,
        &impl_->swap_base);
    if (impl_->swap_alloc_id.IsNull()) {
      throw std::runtime_error(
          "gpu_vector: swap scratch backend allocation failed");
    }
  } else {
    impl_->swap_alloc_id = ctp::ipc::AllocatorId::GetNull();
    impl_->swap_base = nullptr;
  }

  // 2. Meta backend (per-block Block struct with full Page array).
  clio::run::u32 block_stride = detail::BlockStrideBytes(total_ppb);
  clio::run::u64 meta_bytes = static_cast<clio::run::u64>(block_stride) * nblocks;
  impl_->meta_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem, meta_bytes,
      &impl_->meta_base);
  if (impl_->meta_alloc_id.IsNull()) {
    throw std::runtime_error("gpu_vector: meta_backend allocation failed");
  }

  // 3. Task pools cover the FULL tier slot count.
  clio::run::u32 put_stride =
      detail::TaskSlotStride<clio::cte::core::PodPutBlobTask>();
  clio::run::u32 get_stride =
      detail::TaskSlotStride<clio::cte::core::PodGetBlobTask>();
  clio::run::u64 task_count = static_cast<clio::run::u64>(nblocks) * total_ppb;
  impl_->put_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost,
      task_count * put_stride, &impl_->put_base);
  if (impl_->put_alloc_id.IsNull()) {
    throw std::runtime_error("gpu_vector: put_pool allocation failed");
  }
  impl_->get_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost,
      task_count * get_stride, &impl_->get_base);
  if (impl_->get_alloc_id.IsNull()) {
    throw std::runtime_error("gpu_vector: get_pool allocation failed");
  }
  std::memset(impl_->put_base, 0, task_count * put_stride);
  std::memset(impl_->get_base, 0, task_count * get_stride);

  // 3b. Allocate the host meta scratch. Pinned host memory so
  //     cudaMemcpyAsync can use DMA. The cache thread copies the
  //     device meta_base here every tick. Reads of pinned host from
  //     device (the DMA copy direction) are safe — the hang only
  //     happens when GPU kernels WRITE to pinned host concurrently
  //     with spin-wait kernels.
  if (host_pages_per_block > 0) {
    clio::run::u64 meta_scratch_bytes = static_cast<clio::run::u64>(block_stride) * nblocks;
    impl_->host_meta_scratch = ctp::GpuApi::MallocHost<char>(meta_scratch_bytes);
    if (!impl_->host_meta_scratch) {
      throw std::runtime_error(
          "gpu_vector: host_meta_scratch allocation failed");
    }
    std::memset(impl_->host_meta_scratch, 0, meta_scratch_bytes);
    impl_->block_stride_cached = block_stride;
    impl_->clear_block_arr =
        ctp::GpuApi::MallocHost<clio::run::u32>(
            sizeof(clio::run::u32) * detail::kClearBatchCap);
    impl_->clear_slot_arr =
        ctp::GpuApi::MallocHost<clio::run::u32>(
            sizeof(clio::run::u32) * detail::kClearBatchCap);
    if (!impl_->clear_block_arr || !impl_->clear_slot_arr) {
      throw std::runtime_error(
          "gpu_vector: clear-batch index array allocation failed");
    }
    // Pinned-host stop flag the persistent kernel polls each iteration.
    impl_->kernel_stop_flag = ctp::GpuApi::MallocHost<clio::run::u32>(1);
    if (!impl_->kernel_stop_flag) {
      throw std::runtime_error(
          "gpu_vector: kernel_stop_flag allocation failed");
    }
    *impl_->kernel_stop_flag = 0;
  }

  // 4. Create the CTE tag.
  clio::cte::core::Client cte_client(clio::cte::core::kCtePoolId);
  auto tag_fut = cte_client.AsyncGetOrCreateTag(tag_name);
  tag_fut.Wait();
  if (tag_fut->GetReturnCode() != 0) {
    throw std::runtime_error(
        "gpu_vector: GetOrCreateTag failed for tag '" + tag_name + "'");
  }
  view_.base.tag_id = tag_fut->tag_id_;
  impl_->tag_name = tag_name;
  // Per-page PutBlob/GetBlob routing. Default (null storage_pool_id) sends
  // page traffic straight to the CTE core (kCtePoolId). Passing a compressor
  // pool here makes this a *compressed* vector: page evictions are routed to
  // the compressor, which compresses in HBM and forwards the compressed blob
  // to its downstream core (next_pool_id_ == kCtePoolId), and page faults are
  // routed to the compressor, which fetches + decompresses. The CTE tag is
  // always created on kCtePoolId above, so the tag_id is valid at the core the
  // compressor forwards to regardless of where page traffic is routed.
  impl_->fam_ppb = family_pages;
  impl_->host_namer = std::move(host_namer);
  impl_->cte_pool_id = storage_pool_id.IsNull()
                           ? clio::cte::core::kCtePoolId
                           : storage_pool_id;

  // 5. Populate DeviceView.
  view_.base.blocks = reinterpret_cast<Block *>(impl_->meta_base);
  view_.base.block_stride_bytes = block_stride;
  view_.base.pages_base = impl_->pages_base;
  view_.base.host_pages_base = impl_->host_pages_base;
  view_.base.put_pool_base = impl_->put_base;
  view_.base.get_pool_base = impl_->get_base;
  view_.base.put_slot_stride = put_stride;
  view_.base.get_slot_stride = get_stride;
  view_.base.pages_alloc_id = impl_->pages_alloc_id;
  view_.base.host_pages_alloc_id = impl_->host_pages_alloc_id;
  view_.base.put_pool_alloc_id = impl_->put_alloc_id;
  view_.base.get_pool_alloc_id = impl_->get_alloc_id;
  view_.base.nblocks = nblocks;
  view_.base.gpu_pages_per_block = gpu_pages_per_block;
  view_.base.host_pages_per_block = host_pages_per_block;
  view_.base.page_size_bytes = page_size_bytes;
  view_.base.allow_cold_miss_fault = allow_cold_miss_fault;
  view_.base.fam_ppb = family_pages;
  impl_->flat_layout = flat_layout;
  view_.base.flat_layout = flat_layout ? 1u : 0u;
  // Pinned-host instrumentation counters. Mapped so the device can
  // atomicAdd_system into them and the host can read them directly
  // (no cudaMemcpy round-trip). cudaHostAllocMapped + UVA means the
  // same pointer works on both sides.
  {
    void *p = nullptr;
    cudaHostAlloc(&p, sizeof(::clio::cte::gpu_vector::VectorStats),
                  cudaHostAllocMapped | cudaHostAllocPortable);
    impl_->stats = static_cast<::clio::cte::gpu_vector::VectorStats *>(p);
    if (impl_->stats) {
      std::memset(impl_->stats, 0,
                  sizeof(::clio::cte::gpu_vector::VectorStats));
    }
  }
  view_.base.stats = impl_->stats;
  view_.page_capacity_t = page_size_bytes / sizeof(T);
  impl_->nblocks_cached = nblocks;
  impl_->page_size_cached = page_size_bytes;
  impl_->gpu_ppb_cached = gpu_pages_per_block;
  impl_->host_ppb_cached = host_pages_per_block;

  // 6. Placement-new every PutBlob/GetBlob slot.
  for (clio::run::u32 b = 0; b < nblocks; ++b) {
    for (clio::run::u32 s = 0; s < total_ppb; ++s) {
      clio::run::u64 slot_idx =
          static_cast<clio::run::u64>(b) * total_ppb + s;
      char *put_addr = impl_->put_base + slot_idx * put_stride;
      char *get_addr = impl_->get_base + slot_idx * get_stride;
      // BARE tag: the family is not baked into the slot name any more. The
      // device fault/flush paths stamp gpu_family_idx_ from the vector's
      // family policy (or the user's FamFn lambda) and the runtime composes
      // "<tag>_b<family>_pi<page>". A per-block family here once made a
      // multi-block vector fault blobs that were never written — the GetBlob
      // parked forever and the faulting warp deadlocked spinning on it.
      // Empty stem: the handler composes "b<fam>_pi<page>" from the task's
      // gpu_family_idx_/gpu_page_idx_. Sending the tag here is redundant with
      // tag_id_ and overflowed the small-string buffer for long model names.
      std::string blob_name = flat_layout ? "w" : "";
      auto put_task = new (put_addr) clio::cte::core::PodPutBlobTask(
          clio::run::CreateTaskId(), impl_->cte_pool_id,
          clio::run::PoolQuery::ToLocalCpu(), view_.base.tag_id,
          blob_name.c_str(), /*offset=*/0, /*size=*/0,
          ctp::ipc::ShmPtr<>::GetNull(), /*score=*/-1.0f,
          clio::cte::core::Context(), /*flags=*/0);
      put_task->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(*put_task));

      auto get_task = new (get_addr) clio::cte::core::PodGetBlobTask(
          clio::run::CreateTaskId(), impl_->cte_pool_id,
          clio::run::PoolQuery::ToLocalCpu(), view_.base.tag_id,
          blob_name.c_str(), /*offset=*/0, /*size=*/0,
          /*flags=*/0, ctp::ipc::ShmPtr<>::GetNull());
      get_task->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(*get_task));
    }
  }

  // 7. Initialize the meta backend on-device. One block per Block; threads
  //    walk slots in stride.
  clio::run::u32 init_threads = (total_ppb < 32u) ? total_ppb : 32u;
  if (init_threads == 0) init_threads = 1;
  detail::InitMetaKernel<<<nblocks, init_threads>>>(
      view_.base, static_cast<char *>(impl_->pages_base),
      static_cast<char *>(impl_->host_pages_base),
      static_cast<char *>(impl_->swap_base));
  ctp::GpuApi::Synchronize();

  // 7b. Pre-launch the manager kernel ONCE from the main thread so the
  //     CUDA module is fully resident on the device before the cache
  //     thread (a separate host thread) tries to launch it concurrently
  //     with active user kernels. Without this warmup, the first
  //     cross-thread launch can deadlock against a long-running user
  //     kernel that uses spin-wait primitives (gpu::Future::Wait).
  //     Also pre-launches the host-prefetch flag-clear batch kernel so
  //     IT doesn't hit the same first-launch deadlock from the cache
  //     thread.
  if (mode == CacheMode::kAsync) {
    auto *gpu_ipc_mgr0 = CLIO_CPU_IPC->GetGpuIpcManager();
    clio::run::IpcManagerGpuInfo info0 = gpu_ipc_mgr0->GetGpuInfo(gpu_id);
    detail::CacheManagerKernel<<<nblocks, manager_threads_per_block>>>(
        info0, view_.base);
    detail::ClearHostPrefetchFlagsBatchKernel<<<1, 1>>>(
        view_.base, impl_->clear_block_arr, impl_->clear_slot_arr, 0);
    ctp::GpuApi::Synchronize();
  }

  // 8. Spawn the cache-management thread. It periodically launches a
  // short-lived CacheManagerKernel that does rescore / reorganize /
  // flush / prefetch in one pass, then exits. The thread relaunches
  // every cache_period_us. This is the only pattern that actually
  // schedules concurrently with user kernels on consumer GPUs —
  // a true "persistent" __nanosleep-loop kernel blocks the scheduler.
  //
  // Cache management is the Vector's responsibility, NOT the user's:
  // the user never calls flush, the user kernel's cudaStreamSync only
  // waits for its own stream, and the cache thread runs on its own
  // non-blocking stream.
  if (mode != CacheMode::kAsync || cache_period_us == 0) {
    return;
  }
  impl_->cache_thread_run.store(true);
  Vector<T> *self = this;
  impl_->cache_thread = std::thread([self]() {
    auto *gpu_ipc_mgr = CLIO_CPU_IPC->GetGpuIpcManager();
    clio::run::IpcManagerGpuInfo info =
        gpu_ipc_mgr->GetGpuInfo(self->impl_->gpu_id);
    cudaFree(0);
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    self->impl_->persistent_stream = stream;
    cudaStream_t drain_stream = nullptr;
    cudaStreamCreateWithFlags(&drain_stream, cudaStreamNonBlocking);
    self->impl_->drain_stream = drain_stream;
    while (self->impl_->cache_thread_run.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(self->impl_->cache_period_us));
      if (!self->impl_->cache_thread_run.load()) break;
      const auto &v = self->view_.base;
      // One-shot pass: launches, runs all four phases, exits. The
      // kernel returns the SM scheduler queue slot so user kernels
      // on other streams can run between ticks.
      detail::CacheManagerKernel<<<v.nblocks,
                                    self->impl_->manager_threads_per_block,
                                    0, stream>>>(info, v);
      cudaStreamSynchronize(stream);
      if (self->impl_->host_meta_scratch != nullptr) {
        self->DrainHostPrefetchQueue(drain_stream);
      }
    }
    if (drain_stream) cudaStreamSynchronize(drain_stream);
    if (stream) cudaStreamDestroy(stream);
    if (drain_stream) cudaStreamDestroy(drain_stream);
    self->impl_->persistent_stream = nullptr;
    self->impl_->drain_stream = nullptr;
  });
#else
  (void)tag_name; (void)nblocks; (void)gpu_id;
  (void)gpu_pages_per_block; (void)host_pages_per_block;
  (void)page_size_bytes; (void)cache_period_us; (void)mode;
  (void)manager_threads_per_block;
#endif  // !CTP_IS_DEVICE_PASS
}

template <typename T>
inline bool Vector<T>::BuildHostTier(clio::run::u64 total_bytes,
                                     clio::run::u64 budget_bytes) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_ || total_bytes == 0) return false;
  if (impl_->host_arena != nullptr) return true;
  const clio::run::u64 page = view_.base.page_size_bytes;
  const clio::run::u64 n_pages = (total_bytes + page - 1) / page;
  clio::run::u64 slots = budget_bytes / page;
  if (slots > n_pages) slots = n_pages;
  if (slots == 0) return false;
  unsigned char *arena = nullptr;
  while (cudaMallocHost(&arena, slots * page) != cudaSuccess) {
    cudaGetLastError();
    arena = nullptr;
    slots >>= 1;                      // degrade gracefully under RAM pressure
    if (slots < 64) return false;
  }
  impl_->host_arena = arena;
  impl_->host_slots = slots;
  impl_->mirror_bytes = total_bytes;
  if (slots == n_pages) {
    // FULL coverage: prefill through the CTE and publish the flat image.
    bool ok = true;
    if (view_.base.flat_layout) {
      clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
      const clio::run::u64 kChunk = 64ull << 20;
      auto buf = CLIO_IPC->AllocateBuffer(kChunk);
      ok = buf.ptr_ != nullptr;
      for (clio::run::u64 off = 0; ok && off < total_bytes; off += kChunk) {
        const clio::run::u64 n =
            std::min<clio::run::u64>(total_bytes - off, kChunk);
        auto gt = cte.AsyncGetBlob(view_.base.tag_id, "w", off, n, 0,
                                   buf.shm_.template Cast<void>(),
                                   clio::run::PoolQuery::Local());
        gt.Wait();
        if (gt->GetReturnCode() != 0) { ok = false; break; }
        std::memcpy(arena + off, buf.ptr_, n);
      }
      if (buf.ptr_ != nullptr) CLIO_IPC->FreeBuffer(buf);
    } else {
      // Paged (e.g. COMPRESSED) fill: K gets in flight so dispatch and
      // decompression pipeline instead of serializing per page.
      clio::cte::core::Client cte(impl_->cte_pool_id);
      constexpr clio::run::u64 kInflight = 16;
      struct Slot {
        decltype(CLIO_IPC->AllocateBuffer(0)) buf;
        decltype(cte.AsyncGetBlob(view_.base.tag_id, "", 0, 0, 0,
                                  buf.shm_.template Cast<void>(),
                                  clio::run::PoolQuery::Local())) fut;
        clio::run::u64 gp = 0;
        bool busy = false;
      };
      std::vector<Slot> ring(kInflight);
      for (auto &sl2 : ring) {
        sl2.buf = CLIO_IPC->AllocateBuffer(page);
        if (sl2.buf.ptr_ == nullptr) ok = false;
      }
      clio::run::u64 next = 0, done = 0;
      while (ok && done < n_pages) {
        for (auto &sl2 : ring) {
          if (!ok) break;
          if (sl2.busy) {
            sl2.fut.Wait();
            if (sl2.fut->GetReturnCode() != 0) { ok = false; break; }
            std::memcpy(arena + sl2.gp * page, sl2.buf.ptr_,
                        std::min<clio::run::u64>(
                            total_bytes - sl2.gp * page, page));
            sl2.busy = false;
            ++done;
          }
          if (next < n_pages) {
            const std::string name = PageBlobName(next);
            sl2.fut = cte.AsyncGetBlob(view_.base.tag_id, name.c_str(), 0,
                                       page, 0,
                                       sl2.buf.shm_.template Cast<void>(),
                                       clio::run::PoolQuery::Local());
            sl2.gp = next++;
            sl2.busy = true;
          }
        }
      }
      for (auto &sl2 : ring) {
        if (sl2.busy) {
          sl2.fut.Wait();
          if (ok && sl2.fut->GetReturnCode() == 0) {
            std::memcpy(arena + sl2.gp * page, sl2.buf.ptr_,
                        std::min<clio::run::u64>(
                            total_bytes - sl2.gp * page, page));
            ++done;
          }
          sl2.busy = false;
        }
        if (sl2.buf.ptr_ != nullptr) CLIO_IPC->FreeBuffer(sl2.buf);
      }
    }
    if (!ok) {
      cudaFreeHost(arena);
      impl_->host_arena = nullptr;
      impl_->host_slots = 0;
      return false;
    }
    impl_->host_mirror = arena;
    view_.base.host_mirror = arena;
    return true;
  }
  // PARTIAL coverage: page-granular cache, filled on demand. host_mirror
  // stays null so no kernel ever sees a raw arena pointer.
  impl_->host_page_slot.assign(n_pages, -1);
  impl_->host_slot_page.assign(slots, -1);
  impl_->host_slot_ref.assign(slots, 0);
  return true;
#else
  (void) total_bytes; (void) budget_bytes;
  return false;
#endif
}

// Make model page gp resident in the host tier and return its pinned bytes.
// Partial mode only. Clock eviction; the copy stream is synced before a
// recycled slot is overwritten so no queued H2D still reads it.
template <typename T>
inline unsigned char *Vector<T>::EnsureHostPage(clio::run::u64 gp) {
#if !CTP_IS_DEVICE_PASS
  const clio::run::u64 page = view_.base.page_size_bytes;
  long long sl = impl_->host_page_slot[gp];
  if (sl >= 0) {
    impl_->host_slot_ref[sl] = 1;
    impl_->host_tier_hits++;
    return impl_->host_arena + (clio::run::u64) sl * page;
  }
  // Clock: find a victim slot.
  for (;;) {
    sl = (long long) (impl_->host_clock++ % impl_->host_slots);
    if (impl_->host_slot_ref[sl] == 0) break;
    impl_->host_slot_ref[sl] = 0;
  }
  if (impl_->copy_stream != nullptr) {
    cudaStreamSynchronize(impl_->copy_stream);
  }
  const long long old_gp = impl_->host_slot_page[sl];
  if (old_gp >= 0) impl_->host_page_slot[old_gp] = -1;
  unsigned char *dst = impl_->host_arena + (clio::run::u64) sl * page;
  clio::cte::core::Client cte(view_.base.flat_layout
                                  ? clio::cte::core::kCtePoolId
                                  : impl_->cte_pool_id);
  auto buf = CLIO_IPC->AllocateBuffer(page);
  if (buf.ptr_ == nullptr) return nullptr;
  bool ok = false;
  if (view_.base.flat_layout) {
    auto gt = cte.AsyncGetBlob(view_.base.tag_id, "w", gp * page, page, 0,
                               buf.shm_.template Cast<void>(),
                               clio::run::PoolQuery::Local());
    gt.Wait();
    ok = gt->GetReturnCode() == 0;
  } else {
    const std::string name = PageBlobName(gp);
    auto gt = cte.AsyncGetBlob(view_.base.tag_id, name.c_str(), 0, page, 0,
                               buf.shm_.template Cast<void>(),
                               clio::run::PoolQuery::Local());
    gt.Wait();
    ok = gt->GetReturnCode() == 0;
  }
  if (ok) {
    std::memcpy(dst, buf.ptr_,
                std::min<clio::run::u64>(impl_->mirror_bytes - gp * page,
                                         page));
  }
  CLIO_IPC->FreeBuffer(buf);
  if (!ok) return nullptr;
  impl_->host_page_slot[gp] = sl;
  impl_->host_slot_page[sl] = (long long) gp;
  impl_->host_slot_ref[sl] = 1;
  impl_->host_tier_fills++;
  return dst;
#else
  (void) gp;
  return nullptr;
#endif
}

template <typename T>
inline bool Vector<T>::ReserveSpanCaches(clio::run::u64 ring_bytes,
                                         clio::run::u64 vram_reserve,
                                         clio::run::u64 slab_cap_max) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_) return false;
  for (clio::run::u64 want = ring_bytes; want >= (64ull << 20);
       want >>= 1) {
    if (cudaMalloc(&impl_->stage_ring, want) == cudaSuccess) {
      impl_->ring_cap = want;
      break;
    }
    impl_->stage_ring = nullptr;
    cudaGetLastError();
  }
  size_t mfree = 0, mtot = 0;
  cudaMemGetInfo(&mfree, &mtot);
  clio::run::u64 slab_max =
      (mfree > vram_reserve) ? (clio::run::u64) mfree - vram_reserve : 0;
  if (slab_max > slab_cap_max) slab_max = slab_cap_max;
  for (clio::run::u64 want = slab_max; want >= (256ull << 20);
       want >>= 1) {
    if (cudaMalloc(&impl_->span_slab, want) == cudaSuccess) {
      impl_->slab_cap = want;
      break;
    }
    impl_->span_slab = nullptr;
    cudaGetLastError();
  }
  if (impl_->copy_stream == nullptr) {
    cudaStreamCreateWithFlags(&impl_->copy_stream, cudaStreamNonBlocking);
    for (auto &ev : impl_->copy_events) {
      cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    }
  }
  return impl_->stage_ring != nullptr;
#else
  (void) ring_bytes; (void) vram_reserve; (void) slab_cap_max;
  return false;
#endif
}

template <typename T>
inline void *Vector<T>::RecordCopyEvent() {
#if !CTP_IS_DEVICE_PASS
  if (!impl_ || impl_->copy_stream == nullptr) return nullptr;
  cudaEvent_t ev = impl_->copy_events[impl_->copy_ev_idx++ % 256];
  cudaEventRecord(ev, impl_->copy_stream);
  return (void *) ev;
#else
  return nullptr;
#endif
}

template <typename T>
inline const unsigned char *Vector<T>::PinnedSpan(clio::run::u64 off,
                                                  clio::run::u64 len,
                                                  bool permanent,
                                                  bool *copied,
                                                  FetchList *fl) {
#if !CTP_IS_DEVICE_PASS
  if (copied) *copied = false;
  if (!impl_) return nullptr;
  if (impl_->mirror_bytes != 0 && off >= impl_->mirror_bytes) return nullptr;
  if (len > impl_->mirror_bytes - off) len = impl_->mirror_bytes - off;
  auto it = impl_->span_cache.find(off);
  if (it != impl_->span_cache.end() && it->second.second >= len) {
    impl_->span_hits++;
    return it->second.first;
  }
  impl_->span_misses++;
  unsigned char *dst = nullptr;
  if (permanent) {
    if (impl_->static_bytes + len > impl_->static_cap) return nullptr;
    if (cudaMalloc(&dst, len) != cudaSuccess) {
      cudaGetLastError();
      return nullptr;
    }
    impl_->static_bytes += len;
  } else {
    if (impl_->span_slab == nullptr ||
        impl_->slab_used + len > impl_->slab_cap ||
        ++impl_->span_seen[off] < 2) {
      return nullptr;
    }
    dst = impl_->span_slab + impl_->slab_used;
    impl_->slab_used += (len + 255ull) & ~255ull;
  }
  if (!CopySpan(dst, off, len, fl)) {
    // dst stays allocated but unmapped; do not cache a bad span.
    return nullptr;
  }
  impl_->span_cache[off] = {dst, len};   // longer span replaces a shorter
  if (copied) *copied = true;
  return dst;
#else
  (void) off; (void) len; (void) permanent; (void) copied;
  return nullptr;
#endif
}

template <typename T>
inline bool Vector<T>::CopySpan(unsigned char *dst, clio::run::u64 off,
                                clio::run::u64 len, FetchList *fl) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_ || dst == nullptr) return false;
  if (impl_->host_arena == nullptr) {
    // No DRAM tier (compressed mode): fetch device-direct through the CTE.
    // kHBM-resident pages decompress VRAM->VRAM; nothing uncompressed
    // crosses PCIe. With fl, the caller batches waits across MANY spans.
    if (fl != nullptr) return FetchSpanDeviceAsync(dst, off, len, fl);
    return FetchSpanDevice(dst, off, len);
  }
  if (off >= impl_->mirror_bytes) return false;
  if (len > impl_->mirror_bytes - off) len = impl_->mirror_bytes - off;
  if (impl_->host_mirror != nullptr) {   // full coverage: one flat copy
    return cudaMemcpyAsync(dst, impl_->host_mirror + off, len,
                           cudaMemcpyHostToDevice,
                           impl_->copy_stream) == cudaSuccess;
  }
  // Partial coverage: gather per page through the host cache. Misses fill
  // through the CTE (decompressing exactly once for compressed tags).
  const clio::run::u64 page = view_.base.page_size_bytes;
  clio::run::u64 cur = off;
  while (cur < off + len) {
    const clio::run::u64 gp   = cur / page;
    const clio::run::u64 poff = cur - gp * page;
    const clio::run::u64 seg  =
        std::min<clio::run::u64>(page - poff, off + len - cur);
    unsigned char *hp = EnsureHostPage(gp);
    if (hp == nullptr) return false;
    if (cudaMemcpyAsync(dst + (cur - off), hp + poff, seg,
                        cudaMemcpyHostToDevice,
                        impl_->copy_stream) != cudaSuccess) {
      return false;
    }
    cur += seg;
  }
  return true;
#else
  (void) dst; (void) off; (void) len;
  return false;
#endif
}

template <typename T>
inline bool Vector<T>::FetchSpanDeviceAsync(unsigned char *dst_dev,
                                            clio::run::u64 off,
                                            clio::run::u64 len,
                                            FetchList *fl) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_ || dst_dev == nullptr || fl == nullptr) return false;
  if (impl_->mirror_bytes != 0) {
    if (off >= impl_->mirror_bytes) return false;
    if (len > impl_->mirror_bytes - off) len = impl_->mirror_bytes - off;
  }
  clio::cte::core::Client cte(impl_->cte_pool_id);
  const clio::run::u64 page = view_.base.page_size_bytes;
  // Route every fetch of this TAG to ONE lane: per-blob hashing scattered
  // same-tag gets across workers, so the compressor's SmashBatch only ever
  // saw ~1/n of an issue burst (avg batch 3.3 of 16). Converging them is
  // what lets one batched nvcomp decompress span the whole burst.
  const clio::run::PoolQuery pq = clio::run::PoolQuery::DirectHash(
      (clio::run::u32) (((clio::run::u64) view_.base.tag_id.major_ << 8) ^
                        view_.base.tag_id.minor_));
  clio::run::u64 cur = off;
  while (cur < off + len) {
    const clio::run::u64 gp   = cur / page;
    const clio::run::u64 poff = cur - gp * page;
    const clio::run::u64 seg  =
        std::min<clio::run::u64>(page - poff, off + len - cur);
    const std::string name =
        view_.base.flat_layout ? std::string("w") : PageBlobName(gp);
    const clio::run::u64 boff = view_.base.flat_layout ? cur : poff;
    fl->push_back(cte.AsyncGetBlob(
        view_.base.tag_id, name, boff, seg, 0,
        (char *) dst_dev + (cur - off), pq));
    cur += seg;
  }
  return true;
#else
  (void) dst_dev; (void) off; (void) len; (void) fl;
  return false;
#endif
}

template <typename T>
inline bool Vector<T>::FetchSpanDevice(unsigned char *dst_dev,
                                       clio::run::u64 off,
                                       clio::run::u64 len) {
#if !CTP_IS_DEVICE_PASS
  FetchList fl;
  if (!FetchSpanDeviceAsync(dst_dev, off, len, &fl)) return false;
  return WaitFetches(fl);
#else
  (void) dst_dev; (void) off; (void) len;
  return false;
#endif
}

template <typename T>
inline double Vector<T>::MeasuredRatio(clio::run::u64 n_sample) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_ || view_.base.flat_layout) return 0.0;
  const clio::run::u64 page = view_.base.page_size_bytes;
  const clio::run::u64 pages = impl_->mirror_bytes / page;
  if (pages == 0) return 0.0;
  clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
  cte.AttachShmCache();
  // Sample evenly across the model: a prefix would over-weight whatever
  // the first tensors happen to compress to.
  const clio::run::u64 n = std::min<clio::run::u64>(n_sample, pages);
  const clio::run::u64 stride = pages / n;
  clio::run::u64 stored = 0, seen = 0;
  for (clio::run::u64 k = 0; k < n; ++k) {
    // The SHM record's total_size_ is the STORED byte count (for a
    // transformed blob, post-transform); GetBlobSize reports the LOGICAL
    // size, which for a compressed blob is the original and would measure
    // every model as incompressible (observed: ratio 0.96).
    clio::cte::core::ShmBlobRecord rec;
    if (cte.TryGetBlobRecordShm(view_.base.tag_id, PageBlobName(k * stride),
                                &rec) &&
        rec.total_size_ > 0) {
      stored += rec.total_size_;
      ++seen;
    }
  }
  if (seen == 0) return 0.0;
  return (double) stored / (double) (seen * page);
#else
  (void) n_sample;
  return 0.0;
#endif
}

template <typename T>
inline void Vector<T>::PromoteSpan(clio::run::u64 off, clio::run::u64 len,
                                   float score) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_) return;
  const clio::run::u64 page = view_.base.page_size_bytes;
  clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
  for (clio::run::u64 gp = off / page; gp <= (off + len - 1) / page; ++gp) {
    if (view_.base.flat_layout) break;   // flat blob: one blob, no per-page
    impl_->promos.push_back(cte.AsyncReorganizeBlob(
        view_.base.tag_id, PageBlobName(gp), score,
        clio::run::PoolQuery::Local()));
    while (impl_->promos.size() > 192) {
      impl_->promos.front().Wait();
      impl_->promos.pop_front();
    }
  }
#else
  (void) off; (void) len; (void) score;
#endif
}

template <typename T>
inline void Vector<T>::DrainPromotions() {
#if !CTP_IS_DEVICE_PASS
  if (!impl_) return;
  while (!impl_->promos.empty()) {
    impl_->promos.front().Wait();
    impl_->promos.pop_front();
  }
#endif
}

template <typename T>
inline bool Vector<T>::PageIsPromoted(clio::run::u64 gp) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_) return false;
  clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
  clio::cte::core::ShmBlobRecord rec;
  return cte.TryGetBlobRecordShm(view_.base.tag_id, PageBlobName(gp), &rec) &&
         rec.score_ >= 0.9f;
#else
  (void) gp;
  return false;
#endif
}

template <typename T>
inline unsigned char *Vector<T>::StageTransient(clio::run::u64 len,
                                                void *main_stream) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_ || impl_->stage_ring == nullptr || len > impl_->ring_cap) {
    return nullptr;
  }
  if (impl_->ring_head + len > impl_->ring_cap) {
    impl_->ring_head = 0;
    // Ring wrap with copies on the copy stream: fence it on the consumer
    // stream's progress so no in-flight reader still uses the region.
    cudaEvent_t wev = impl_->copy_events[impl_->copy_ev_idx++ % 256];
    cudaEventRecord(wev, (cudaStream_t) main_stream);
    cudaStreamWaitEvent(impl_->copy_stream, wev, 0);
  }
  unsigned char *out = impl_->stage_ring + impl_->ring_head;
  impl_->ring_head += (len + 255ull) & ~255ull;
  return out;
#else
  (void) len; (void) main_stream;
  return nullptr;
#endif
}

template <typename T>
inline Vector<T>::~Vector() {
#if !CTP_IS_DEVICE_PASS
  if (!impl_) return;
  // Stop the cache thread. It owns persistent_stream / drain_stream,
  // syncs them, and destroys them as it exits.
  impl_->cache_thread_run.store(false, std::memory_order_release);
  if (impl_->cache_thread.joinable()) impl_->cache_thread.join();
  // Wait for any host-driven AsyncGetBlobs to settle.
  {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(5);
    while (impl_->async_inflight.load(std::memory_order_acquire) > 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  auto *cpu_ipc = CLIO_CPU_IPC;
  if (impl_->pages_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->pages_alloc_id);
  if (impl_->host_pages_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->host_pages_alloc_id);
  if (impl_->meta_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->meta_alloc_id);
  if (impl_->swap_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->swap_alloc_id);
  if (impl_->put_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->put_alloc_id);
  if (impl_->get_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->get_alloc_id);
  if (impl_->host_meta_scratch)
    ctp::GpuApi::FreeHost(impl_->host_meta_scratch);
  if (impl_->clear_block_arr)
    ctp::GpuApi::FreeHost(impl_->clear_block_arr);
  if (impl_->clear_slot_arr)
    ctp::GpuApi::FreeHost(impl_->clear_slot_arr);
  if (impl_->kernel_stop_flag)
    ctp::GpuApi::FreeHost(impl_->kernel_stop_flag);
  if (impl_->stats)
    cudaFreeHost(impl_->stats);
  if (impl_->copy_stream) {
    cudaStreamSynchronize(impl_->copy_stream);
    for (auto &ev : impl_->copy_events) {
      if (ev) cudaEventDestroy(ev);
    }
    cudaStreamDestroy(impl_->copy_stream);
  }
  for (auto &kv : impl_->span_cache) {
    // Slab entries are freed with the slab; permanent spans are exact
    // allocations outside it.
    unsigned char *ptr = kv.second.first;
    if (impl_->span_slab == nullptr || ptr < impl_->span_slab ||
        ptr >= impl_->span_slab + impl_->slab_cap) {
      cudaFree(ptr);
    }
  }
  if (impl_->span_slab) cudaFree(impl_->span_slab);
  if (impl_->stage_ring) cudaFree(impl_->stage_ring);
  if (impl_->host_arena) cudaFreeHost(impl_->host_arena);
#endif  // !CTP_IS_DEVICE_PASS
}

template <typename T>
inline void Vector<T>::DrainHostPrefetchQueue(void *cuda_stream) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_ || !impl_->host_meta_scratch) return;
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  clio::cte::core::Client cte_client(impl_->cte_pool_id);
  const char *dbg = std::getenv("GPU_VECTOR_DEBUG_CACHE");
  clio::run::u32 nb = impl_->nblocks_cached;
  clio::run::u32 gpu_ppb = impl_->gpu_ppb_cached;
  clio::run::u32 host_ppb = impl_->host_ppb_cached;
  clio::run::u64 psz = impl_->page_size_cached;
  clio::run::u32 block_stride = impl_->block_stride_cached;
  clio::run::u32 total_ppb = gpu_ppb + host_ppb;
  // Snapshot the device meta region into the host scratch via the
  // dedicated cache-thread stream.
  clio::run::u64 meta_bytes = static_cast<clio::run::u64>(block_stride) * nb;
  cudaMemcpyAsync(impl_->host_meta_scratch, view_.base.blocks,
                  meta_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  clio::run::u32 n_to_clear = 0;
  // Two-pass pipeline so prefetches actually overlap:
  //   Pass 1: scan meta, dispatch AsyncGetBlob for every slot with
  //           kPageGetInFlight set. Collect futures + clear info.
  //   Pass 2: Wait on each future (they run in parallel through the
  //           runtime). Bump/decrement async_inflight bookkeeping.
  //   Pass 3 (after the loop): one batched clear-flags kernel.
  std::vector<clio::run::Future<clio::cte::core::PodGetBlobTask>> futs;
  futs.reserve(static_cast<size_t>(detail::kClearBatchCap));
  for (clio::run::u32 b = 0; b < nb; ++b) {
    Block *bp = reinterpret_cast<Block *>(impl_->host_meta_scratch +
                  static_cast<clio::run::u64>(b) * block_stride);
    for (clio::run::u32 s = 0; s < total_ppb; ++s) {
      Page *p = &bp->pages[s];
      if (!(p->flags & kPageGetInFlight)) continue;
      if (p->page_idx < 0) continue;
      if (n_to_clear >= detail::kClearBatchCap) break;
      char *device_ptr = nullptr;
      if (s < gpu_ppb) {
        device_ptr = impl_->pages_base +
            (static_cast<clio::run::u64>(b) * gpu_ppb + s) * psz;
      } else {
        clio::run::u32 host_slot = s - gpu_ppb;
        device_ptr = impl_->host_pages_base +
            (static_cast<clio::run::u64>(b) * host_ppb + host_slot) * psz;
      }
      std::string blob_name = PageBlobName(p->page_idx);
      ctp::ipc::ShmPtr<> blob_data;
      blob_data.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
      blob_data.off_ = reinterpret_cast<clio::run::u64>(device_ptr);
      if (dbg && dbg[0] == '1') {
        std::fprintf(stderr, "[DRAIN] block %u slot %u page %d dispatch\n",
                     b, s, p->page_idx);
      }
      impl_->async_inflight.fetch_add(1, std::memory_order_acq_rel);
      futs.push_back(cte_client.AsyncPodGetBlob(view_.base.tag_id, blob_name,
                                                 PageBlobOffset(p->page_idx), psz,
                                                 /*flags=*/0, blob_data,
                                                 clio::run::PoolQuery::ToLocalCpu()));
      impl_->clear_block_arr[n_to_clear] = b;
      impl_->clear_slot_arr[n_to_clear] = s;
      ++n_to_clear;
    }
  }
  // Now wait — futures complete in parallel through the runtime.
  for (auto &fut : futs) {
    fut.Wait();
    impl_->async_inflight.fetch_sub(1, std::memory_order_acq_rel);
  }
  if (n_to_clear > 0) {
    clio::run::u32 threads = (n_to_clear < 32) ? n_to_clear : 32;
    clio::run::u32 blocks = (n_to_clear + threads - 1) / threads;
    detail::ClearHostPrefetchFlagsBatchKernel<<<blocks, threads, 0, stream>>>(
        view_.base, impl_->clear_block_arr, impl_->clear_slot_arr,
        n_to_clear);
  }
  cudaStreamSynchronize(stream);
#endif
}

template <typename T>
inline void Vector<T>::FlushAllSync() {
#if !CTP_IS_DEVICE_PASS
  // No-op. The persistent manager kernel flushes dirty pages
  // continuously. The Vector is automatically coherent — the user
  // never needs to flush. This method is retained as a no-op so
  // legacy callers compile, but it does NOT do anything.
#endif
}

template <typename T>
inline void Vector<T>::FaultAllSync(clio::run::u64 num_model_pages) {
#if !CTP_IS_DEVICE_PASS
  if (impl_->host_ppb_cached != 0) {
    throw std::runtime_error(
        "gpu_vector: FaultAllSync is single-tier (legacy) only");
  }
  const clio::run::u32 nblocks = impl_->nblocks_cached;
  const clio::run::u32 gpu_ppb = impl_->gpu_ppb_cached;
  const clio::run::u64 page_size = impl_->page_size_cached;

  // Decompress every page from the store straight into its HBM slot. The
  // GetBlob is routed to impl_->cte_pool_id (the storage pool); when that is a
  // compressor it decompresses. blob_data is a zero-copy ShmPtr whose off_ is
  // the slot's HBM device address (MakeBlobShmPtr), so the (de)compressor
  // writes the page bytes directly into HBM. The GPU is idle here (no user
  // kernel), so the on-GPU decompressor runs without contending with a
  // spin-waiting fault kernel.
  clio::cte::core::Client client(impl_->cte_pool_id);
  for (clio::run::u32 b = 0; b < nblocks; ++b) {
    for (clio::run::u32 s = 0; s < gpu_ppb; ++s) {
      clio::run::u64 gp = static_cast<clio::run::u64>(b) * gpu_ppb + s;
      // Only the first num_model_pages global pages have a backing blob; the
      // rest of the cache is unused. Faulting past the object would GetBlob a
      // page that was never stored (rc != 0) and abort the whole pre-fault.
      // num_model_pages == 0 means "the object fills the cache" -> fault all.
      if (num_model_pages != 0 && gp >= num_model_pages) {
        continue;
      }
      char *hbm = static_cast<char *>(impl_->pages_base) + gp * page_size;
      // Matches the eviction name: writer stores "<tag>_b<block>" and the
      // (de)compressor appends "_pi<gpu_page_idx>" where gpu_page_idx is the
      // global page index. Legacy slot (b,s) holds global page b*gpu_ppb+s.
      std::string name = PageBlobName(gp);
      // Zero-copy blob_data: null alloc id + off_ = the slot's HBM device
      // address, so the (de)compressor writes decompressed bytes straight into
      // HBM. (Host-side equivalent of detail::MakeBlobShmPtr, which is
      // device-only.)
      ctp::ipc::ShmPtr<> blob_data;
      blob_data.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
      blob_data.off_ = reinterpret_cast<clio::run::u64>(hbm);
      auto gf = client.AsyncGetBlob(view_.base.tag_id, name, PageBlobOffset(gp),
                                    page_size, /*flags=*/0, blob_data,
                                    clio::run::PoolQuery::Local());
      gf.Wait();
      if (gf->GetReturnCode() != 0) {
        throw std::runtime_error(
            "gpu_vector: FaultAllSync GetBlob failed for '" + name +
            "' rc=" + std::to_string(gf->GetReturnCode()));
      }
    }
  }

  // Bind every slot to its global page and mark resident/clean so the read
  // kernel's Resolve finds the page present instead of cold-faulting.
  clio::run::u32 threads = (gpu_ppb < 256u) ? gpu_ppb : 256u;
  if (threads == 0) threads = 1;
  detail::MarkAllResidentKernel<<<nblocks, threads>>>(view_.base);
  ctp::GpuApi::Synchronize();
#endif
}

template <typename T>
inline void Vector<T>::PrefetchWindowSync(clio::run::u64 first_page) {
#if !CTP_IS_DEVICE_PASS
  // Load the whole cache (gpu_pages_per_block) starting at first_page into
  // slots [0, gpu_pages_per_block).
  //
  // SERIAL, deliberately. Batching this call is NOT safe: it is the whole-cache
  // rebind, so `count` is the entire slot count (hundreds to thousands), and
  // batched mode issues every GetBlob before waiting on any. Measured 2026-07-24:
  // flipping this to batched=true HANGS a 2x-oversubscribed 0.5B run that works
  // serially -- that many simultaneously outstanding tasks wedges the runtime.
  // Any fix for the demand-path cost has to bound concurrency (chunked batches),
  // not simply flip this flag.
  PrefetchPagesSync(first_page, impl_->gpu_ppb_cached, /*slot_base=*/0,
                    /*batched=*/false);
#endif
}

template <typename T>
inline std::string Vector<T>::PageBlobName(clio::run::u64 gp) const {
#if !CTP_IS_DEVICE_PASS
  if (impl_->host_namer) {
    return impl_->host_namer(gp);
  }
  // No tag prefix: tag_id already scopes the lookup, and keeping the name
  // short is what keeps PodGetBlobTask::blob_name_ inside its SSO buffer.
  if (impl_->flat_layout) return "w";   // flat: one blob, offset-addressed
  const clio::run::u64 fam = impl_->fam_ppb ? gp / impl_->fam_ppb : 0;
  return "b" + std::to_string(fam) + "_pi" + std::to_string(gp);
#else
  (void)gp; return {};
#endif
}

template <typename T>
inline clio::run::u64 Vector<T>::PageBlobOffset(clio::run::u64 gp) const {
#if !CTP_IS_DEVICE_PASS
  return impl_->flat_layout ? gp * view_.base.page_size_bytes : 0;
#else
  (void)gp; return 0;
#endif
}

template <typename T>
inline bool Vector<T>::FlatLayout() const {
#if !CTP_IS_DEVICE_PASS
  return impl_->flat_layout;
#else
  return false;
#endif
}

template <typename T>
inline void Vector<T>::PrefetchPagesSync(clio::run::u64 first_page,
                                         clio::run::u32 count,
                                         clio::run::u32 slot_base,
                                         bool batched) {
#if !CTP_IS_DEVICE_PASS
  if (impl_->nblocks_cached != 1 || impl_->host_ppb_cached != 0) {
    throw std::runtime_error(
        "gpu_vector: PrefetchPagesSync is single-block single-tier only");
  }
  const clio::run::u64 page_size = impl_->page_size_cached;
  clio::cte::core::Client client(impl_->cte_pool_id);
  auto issue = [&](clio::run::u32 i) {
    clio::run::u64 gp = first_page + i;
    char *hbm =
        static_cast<char *>(impl_->pages_base) + (slot_base + i) * page_size;
    std::string name = PageBlobName(gp);
    ctp::ipc::ShmPtr<> blob_data;
    blob_data.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    blob_data.off_ = reinterpret_cast<clio::run::u64>(hbm);
    return client.AsyncGetBlob(view_.base.tag_id, name, PageBlobOffset(gp), page_size,
                               /*flags=*/0, blob_data,
                               clio::run::PoolQuery::Local());
  };
  auto check = [&](clio::run::Future<clio::cte::core::GetBlobTask> &gf,
                   clio::run::u32 i) {
    if (gf->GetReturnCode() != 0) {
      throw std::runtime_error(
          "gpu_vector: PrefetchPagesSync GetBlob failed for '" +
          PageBlobName(first_page + i) +
          "' rc=" + std::to_string(gf->GetReturnCode()));
    }
  };
  if (batched) {
    // Issue every page's GetBlob, THEN wait -- the compressor decompresses them
    // concurrently across its worker pool instead of one serial round-trip each.
    std::vector<clio::run::Future<clio::cte::core::GetBlobTask>> futs;
    futs.reserve(count);
    for (clio::run::u32 i = 0; i < count; ++i) futs.push_back(issue(i));
    for (clio::run::u32 i = 0; i < count; ++i) {
      futs[i].Wait();
      check(futs[i], i);
    }
  } else {
    for (clio::run::u32 i = 0; i < count; ++i) {
      auto gf = issue(i);
      gf.Wait();
      check(gf, i);
    }
  }
  // Optional MODELED slow-tier read latency. On this cluster, GPU compression
  // shrinks each page ~16x, so the real Lustre read is sub-ms and the reader is
  // compute-bound, not IO-bound -- prefetch then has almost nothing to hide.
  // IOWarp tiers, however, span HBM (~ns) -> NVMe (~us) -> PFS (~ms) -> object
  // store / tape (~10-100ms). Setting CLIO_CTE_SLOW_TIER_US models such a slow
  // tier's per-page read latency so the IO-bound regime -- and the value of
  // prefetch-ahead -- can be demonstrated. It is a latency MODEL, not a measured
  // read; it is off (0) unless the benchmark sets it.
  static const clio::run::u64 slow_us = [] {
    const char *e = std::getenv("CLIO_CTE_SLOW_TIER_US");
    return e ? std::strtoull(e, nullptr, 10) : 0ULL;
  }();
  if (slow_us) {
    std::this_thread::sleep_for(std::chrono::microseconds(slow_us * count));
  }
  // Mark resident on a dedicated stream and sync only that stream, so this
  // prefetch does not device-synchronize (and thus won't stall a concurrently
  // running read kernel on another stream) -- the key to overlap.
  static thread_local cudaStream_t mark_stream = nullptr;
  if (!mark_stream) cudaStreamCreateWithFlags(&mark_stream, cudaStreamNonBlocking);
  clio::run::u32 threads = (count < 256u) ? count : 256u;
  if (threads == 0) threads = 1;
  detail::MarkPagesResidentKernel<<<1, threads, 0, mark_stream>>>(
      view_.base, first_page, count, slot_base);
  cudaStreamSynchronize(mark_stream);
#endif
}

template <typename T>
inline void Vector<T>::ReorganizePagesSync(clio::run::u64 first_page,
                                           clio::run::u32 count, float score) {
#if !CTP_IS_DEVICE_PASS
  if (impl_->nblocks_cached != 1 || count == 0) {
    return;  // single-block only, same constraint as the prefetch paths
  }
  // Reorganize is handled by the CTE core (the tag owner); the compressor
  // entrypoint has no handler for it, so target kCtePoolId directly even when
  // page traffic routes through a storage pool.
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  std::vector<clio::run::Future<clio::cte::core::ReorganizeBlobTask>> futs;
  futs.reserve(count);
  for (clio::run::u32 i = 0; i < count; ++i) {
    std::string name =
        PageBlobName(first_page + i);
    futs.push_back(core.AsyncReorganizeBlob(view_.base.tag_id, name, score,
                                            clio::run::PoolQuery::Local()));
  }
  for (auto &f : futs) {
    f.Wait();  // rc!=0 (e.g. blob still mid-write) is fine — placement is a
               // hint; the next token's pass simply retries the same score.
  }
#endif
}

}  // namespace clio::cte::gpu_vector

#endif  // CTP_IS_GPU_COMPILER

#endif  // CLIO_CTE_GPU_VECTOR_H_
