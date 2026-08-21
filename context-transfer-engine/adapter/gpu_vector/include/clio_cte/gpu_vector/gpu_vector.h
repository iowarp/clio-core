/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * gpu_vector: a vector whose backing store is the CTE and whose working set
 * lives in GPU memory, paged in on demand from inside a kernel.
 *
 * Host side (this class) owns the allocations and hands out a device view;
 * device side (DeviceVector) does the paging. The split is deliberate: a
 * kernel can allocate nothing and block on nothing, so every buffer, task
 * slot and page table is allocated and registered here, before launch.
 *
 * Construction takes only what the spec calls for -- the CTE tag name, the
 * GPUs to use, and the page granularity -- and resolves the tag to its
 * TagId itself.
 */
#ifndef CLIO_CTE_GPU_VECTOR_GPU_VECTOR_H_
#define CLIO_CTE_GPU_VECTOR_GPU_VECTOR_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_cte/gpu_vector/page.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int clio_cte_locate(const void *tag_id, const char *name,
                               unsigned long long *pool_u64,
                               unsigned long long *target_off,
                               unsigned long long *stored_size);
extern "C" char *clio_direct_dev_base(unsigned long long pool_id);

namespace clio::cte::gpu_vector {

#if !CTP_IS_DEVICE_PASS

/**
 * Host-side vector.
 *
 * @tparam T element type. The page granularity is given in BYTES, so T only
 *           affects how offsets are interpreted, never the page layout.
 */
template <typename T>
class Vector {
 public:
  /**
   * @param tag_name   CTE tag representing this vector
   * @param gpu_ids    GPUs to build device views for
   * @param page_bytes page granularity
   * @param nblocks    CUDA blocks that will run against this vector
   * @param pages_per_block resident pages each block may hold
   * @param num_elems  logical length of the vector
   */
  Vector(const std::string &tag_name, const std::vector<int> &gpu_ids,
         clio::run::u64 page_bytes, clio::run::u32 nblocks,
         clio::run::u32 pages_per_block, clio::run::u64 num_elems,
         clio::run::PoolId storage_pool_id = clio::run::PoolId::GetNull(),
         int compress_lib = 0, int compress_preset = 1,
         // The CTE CORE pool, BEHIND the compressor. A spilled encoded page
         // is fetched from here so the raw get returns the STORED bytes and
         // the interposer never decompresses it host-side. Defaults to the
         // storage pool, which is correct when no compressor is composed.
         clio::run::PoolId core_pool_id = clio::run::PoolId::GetNull())
      : core_pool_id_(core_pool_id),
        storage_pool_id_(storage_pool_id.IsNull()
                             ? clio::cte::core::kCtePoolId
                             : storage_pool_id),
        compress_lib_(compress_lib),
        compress_preset_(compress_preset),
        page_bytes_(page_bytes),
        nblocks_(nblocks),
        pages_per_block_(pages_per_block),
        num_elems_(num_elems) {
    if (page_bytes_ == 0 || nblocks_ == 0 || pages_per_block_ == 0) {
      throw std::runtime_error("gpu_vector: zero page size / blocks / pages");
    }
    // Resolve the tag ourselves: the caller names the vector, not an id.
    clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
    auto tag = cte.AsyncGetOrCreateTag(tag_name);
    tag.Wait();
    if (tag->GetReturnCode() != 0) {
      throw std::runtime_error("gpu_vector: could not resolve tag " + tag_name);
    }
    tag_id_ = tag->tag_id_;
    // If one GPU fails to build, release the ones already built: a throwing
    // constructor means no destructor runs, so this is the only chance.
    try {
      for (int gpu : gpu_ids) {
        BuildDevice(gpu);
      }
    } catch (...) {
      for (auto &kv : devs_) {
        Free(kv.second);
      }
      devs_.clear();
      throw;
    }
  }

  /** Releases every device allocation this vector made. Callers free nothing. */
  ~Vector() {
    for (auto &kv : devs_) {
      Free(kv.second);
    }
  }

  /**
   * Non-copyable BY DESIGN, not as an oversight: a vector owns registered GPU
   * backends and raw device buffers, and a second owner of those would free
   * them twice. Share one with a std::shared_ptr<Vector<T>> rather than
   * copying it; the device VIEW (GetDevice) is the thing meant to be copied,
   * and it is a non-owning value type precisely so kernels can take it.
   */
  Vector(const Vector &) = delete;
  Vector &operator=(const Vector &) = delete;

  /**
   * @return true if `wire_id` names a codec that decompresses ON the GPU.
   *
   * Only these need the page cache to be reachable from another CUDA context.
   * The ids are the compressor registry's frozen wire values for the nvcomp
   * family (see CompressionFactory's registry); they are matched by range
   * rather than by name so this header need not pull in the codec factory.
   */
  static bool IsGpuCodec(int wire_id) {
    return wire_id >= 11 && wire_id <= 16;  // nvcomp-lz4 .. nvcomp-ans
  }

  /**
   * Pages spanned by the whole vector.
   *
   * ONE definition, because getting it wrong is silent: `size_` counts
   * ELEMENTS while `page_bytes_` counts BYTES, and the three call sites that
   * open-coded `(size_ + page_bytes_ - 1) / page_bytes_` all undercounted by
   * sizeof(T). For a u32 vector that sized the tier-offset table at a quarter
   * of the page count, so every page past the quarter mark read tier_off_ out
   * of bounds and served another page's bytes -- while `mapped == npages`
   * still reported the map fully built.
   */
  static clio::run::u64 NumPagesOf(const VecHeader &h) {
    if (h.elems_per_page_ == 0) return 0;
    return (h.size_ + h.elems_per_page_ - 1) / h.elems_per_page_;
  }

  /** Device view for `dev_id`, to be passed to a kernel by value. */
  DeviceVector<T> GetDevice(int dev_id) const {
    auto it = devs_.find(dev_id);
    if (it == devs_.end()) {
      throw std::runtime_error("gpu_vector: no device view for this gpu");
    }
    return it->second.view;
  }

  /** Paging activity since the last ResetStats(). */
  struct Stats {
    clio::run::u64 faults = 0;   // pages read in from the CTE
    clio::run::u64 puts = 0;     // pages written back
    clio::run::u64 evicts = 0;   // slots reclaimed
    clio::run::u64 prefetches = 0;      // gets issued by BeginFetch
    clio::run::u64 prefetch_hits = 0;   // arrivals found already in flight
    clio::run::u64 rescores = 0;        // placement hints sent
    clio::run::u64 prefetch_late = 0;   // prefetched, but not landed in time
    /** Gets whose POD already read COMPLETE the instant it was sent -- a
     *  stale completion flag from the slot's previous request. */
    clio::run::u64 early_complete = 0;
    /** Resolves that returned a slot outside the block's own page table. */
    clio::run::u64 bad_slot = 0;
    clio::run::u64 bad_slot_site = 0;
    /** Claims that found no usable slot, and the peak number of slots
     *  pinned at such a moment -- the floor a cache must exceed. */
    clio::run::u64 claim_fails = 0;
    clio::run::u64 peak_pinned_on_fail = 0;
    clio::run::u64 get_errors = 0;      // gets that returned non-zero
    clio::run::u64 pf_dropped = 0;      // async batch hints dropped (slots busy)
    clio::run::u64 verify_ok = 0;       // re-probe after compute: page intact
    clio::run::u64 verify_lost = 0;     // re-probe: page evicted mid-read
    clio::run::u64 put_errors = 0;      // writebacks that returned non-zero
  };
  static constexpr int kNumStats = 17;

  /**
   * Turn on device-side paging counters for every device view.
   *
   * Off by default -- the counters cost an atomic per page event, which is
   * nothing against a fault's round trip but is not free. Tests use them to
   * assert the paging POLICY (hit/miss, writeback, victim choice), which the
   * returned data alone cannot distinguish.
   */
  struct DevState;   // defined below; only the name is needed here

  /** Copy the host image of the header to the device and point the view at it. */
  void PublishHeader(DevState &st) {
#if CTP_ENABLE_CUDA
    if (st.d_hdr == nullptr &&
        (st.d_hdr = ctp::GpuApi::Malloc<VecHeader>(sizeof(VecHeader))) ==
            nullptr) {
      throw std::runtime_error("gpu_vector: header allocation failed");
    }
    ctp::GpuApi::Memcpy(st.d_hdr, &st.hdr, sizeof(VecHeader));
    st.view.h_ = st.d_hdr;
#endif
  }

  void EnableStats() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      if (kv.second.stats != nullptr) continue;
      void *mem = nullptr;
      const size_t bytes = kNumStats * sizeof(unsigned long long);
      if ((mem = ctp::GpuApi::Malloc<char>(bytes)) == nullptr) {
        throw std::runtime_error("gpu_vector: stats allocation failed");
      }
      ctp::GpuApi::Memset(mem, 0, bytes);
      auto *c = static_cast<unsigned long long *>(mem);
      kv.second.stats = c;
      kv.second.hdr.stat_faults_ = c;
      kv.second.hdr.stat_puts_ = c + 1;
      kv.second.hdr.stat_evicts_ = c + 2;
      kv.second.hdr.stat_prefetches_ = c + 3;
      kv.second.hdr.stat_prefetch_hits_ = c + 4;
      kv.second.hdr.stat_rescores_ = c + 5;
      kv.second.hdr.stat_prefetch_late_ = c + 6;
      kv.second.hdr.stat_get_errors_ = c + 7;
      kv.second.hdr.stat_pf_dropped_ = c + 8;
      kv.second.hdr.stat_verify_ok_ = c + 9;
      kv.second.hdr.stat_verify_lost_ = c + 10;
      kv.second.hdr.stat_put_errors_ = c + 11;
      kv.second.hdr.stat_early_complete_ = c + 12;
      kv.second.hdr.stat_bad_slot_ = c + 13;   // c + 14 is its site id
      // OFF unless asked for: the claim-failure sampler walks the whole
      // table under the block lock, and that cost is enough to move the
      // livelock boundary it exists to measure (40 slots ran at 58.8
      // ms/step without it and wedged with it). Opt in with
      // CLIO_GV_CLAIM_PROF=1 when profiling, never in a timed run.
      kv.second.hdr.stat_claim_fail_ =
          (std::getenv("CLIO_GV_CLAIM_PROF") != nullptr) ? c + 15 : nullptr;
      kv.second.hdr.trace_put_errors_ =
          (std::getenv("CLIO_GV_TRACE_PUT_ERRORS") != nullptr) ? 1u : 0u;
      if (getenv("CLIO_FAULT_HIST") != nullptr) {
        const clio::run::u64 npg = NumPagesOf(kv.second.hdr);
        void *hm = nullptr;
        hm = ctp::GpuApi::Malloc<unsigned int>(npg * sizeof(unsigned int));
        if (hm != nullptr) {
          ctp::GpuApi::Memset(hm, 0, npg * sizeof(unsigned int));
          kv.second.hdr.fault_hist_ = static_cast<unsigned int *>(hm);
        }
      }
      PublishHeader(kv.second);   // the device reads the header, not the view
    }
#endif
  }

  /**
   * ZERO-COPY DEVICE-TIER MAP. If this tag's page blobs live on a kHbm bdev
   * (device memory), build a per-page offset table and hand the device view
   * a direct pointer: faults become pointer arithmetic — no slot, no fetch,
   * no DMA (a D2H read under a resident kernel queues behind its channel
   * and can wedge the fault service; mapping removes the transfer class).
   * Call AFTER ingest so the blobs exist. No-op for host tiers.
   */
  /**
   * Publish the per-page STORED-size table for an ENCODED tag.
   *
   * This is metadata, not aliasing: the table is the run-fetch's existence
   * proof (a page with a known stored size has been written and may join a
   * multi-get run; during SEEDING the table is absent, so runs are
   * structurally off and first-touch write-allocate keeps its slots — the
   * gate that fixed the 459600-round seed livelock).
   *
   * The zero-copy device-tier map that used to be built here is GONE, by
   * decision: it handed GPU kernels pointers into SHARED CTE tier memory,
   * racing every tier mover (organizer, target evacuation, eviction). All
   * data now crosses the vector/CTE boundary through orchestrated
   * transfers only, and the fits-in-VRAM fast path is a PRIVATE cache
   * sized to the data (slots >= pages), seeded once.
   *
   * @return pages whose stored size was found (0 for raw tags, where the
   *         table is not needed: run-fetch only serves encoded tags).
   *         Compressor puts land asynchronously, so a caller that wants
   *         full coverage retries while the count is short of NumPages().
   */
  clio::run::u64 PublishStoredSizes() {
#if CTP_ENABLE_CUDA
    if (devs_.empty() || compress_lib_ == 0) return 0;
    const auto &eh = devs_.begin()->second.hdr;
    const clio::run::u64 enp = NumPagesOf(eh);
    if (enp == 0) return 0;
    std::vector<unsigned long long> esz(enp, 0);
    clio::run::u64 found = 0;
    for (clio::run::u64 pg = 0; pg < enp; ++pg) {
      char name[32];
      PageBlobName(pg, name);
      unsigned long long p2 = 0, o2 = 0, s2 = 0;
      if (clio_cte_locate(&tag_id_, name, &p2, &o2, &s2) == 0) {
        esz[pg] = s2;
        ++found;
      }
    }
    unsigned long long *dev_esz = ctp::GpuApi::Malloc<unsigned long long>(
        enp * sizeof(unsigned long long));
    if (dev_esz == nullptr) {
      return 0;
    }
    ctp::GpuApi::Memcpy<unsigned long long>(dev_esz, esz.data(),
                                            enp * sizeof(unsigned long long));
    for (auto &kv : devs_) {
      // Successive retries leak the previous table (enp * 8 bytes); accepted
      // until the retry loop moves inside (issue #966 cleanup).
      kv.second.hdr.stored_size_ =
          static_cast<const unsigned long long *>(dev_esz);
      PublishHeader(kv.second);
    }
    return found;
#else
    return 0;
#endif
  }

  void ResetStats() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      if (kv.second.stats != nullptr) {
        ctp::GpuApi::Memset(kv.second.stats, 0,
                            kNumStats * sizeof(unsigned long long));
      }
    }
#endif
  }

  /** Copy back the per-page fault histogram (empty if not enabled). */
  std::vector<unsigned int> ReadFaultHist(int dev_id) const {
    std::vector<unsigned int> h;
#if CTP_ENABLE_CUDA
    auto it = devs_.find(dev_id);
    if (it == devs_.end() || it->second.hdr.fault_hist_ == nullptr) return h;
    const clio::run::u64 npg = NumPagesOf(it->second.hdr);
    h.resize(npg);
    ctp::GpuApi::Memcpy(h.data(), it->second.hdr.fault_hist_,
                        npg * sizeof(unsigned int));
#endif
    return h;
  }

  Stats ReadStats(int dev_id) const {
    Stats s;
#if CTP_ENABLE_CUDA
    auto it = devs_.find(dev_id);
    if (it == devs_.end() || it->second.stats == nullptr) return s;
    unsigned long long h[kNumStats] = {0};
    ctp::GpuApi::Memcpy(h, it->second.stats, sizeof(h));
    s.faults = h[0];
    s.puts = h[1];
    s.evicts = h[2];
    s.prefetches = h[3];
    s.prefetch_hits = h[4];
    s.rescores = h[5];
    s.put_errors = h[11];
    s.early_complete = h[12];
    s.bad_slot = h[13];
    s.bad_slot_site = h[14];
    s.claim_fails = h[15];
    s.peak_pinned_on_fail = h[16];
    s.prefetch_late = h[6];
    s.get_errors = h[7];
    s.pf_dropped = h[8];
    s.verify_ok = h[9];
    s.verify_lost = h[10];
#else
    (void) dev_id;
#endif
    return s;
  }

  const clio::cte::core::TagId &TagId() const { return tag_id_; }
  clio::run::u64 PageBytes() const { return page_bytes_; }
  clio::run::u64 NumPages() const {
    const clio::run::u64 per = page_bytes_ / sizeof(T);
    return (num_elems_ + per - 1) / per;
  }

  // ---- Host data transfer -------------------------------------------------
  // A vector page IS the blob named "p<page>" under the vector's tag, so the
  // host fills or reads the backing store page by page and never holds the
  // whole array -- which is the point of an out-of-core vector. Transfers
  // are pipelined: a window of async round trips in flight, then drained
  // (a scalar put-then-Wait costs a full CTE round trip per page and
  // overlaps nothing). Each in-flight page owns its buffer for the life of
  // its future; reusing one scratch buffer across an un-drained window is
  // how a pipelined writer corrupts its own data.
  static constexpr clio::run::u64 kHostPipelineDepth = 32;

  /**
   * Drop every cached page on every device view: the next kernel launch
   * starts from an empty cache and faults fresh bytes from the backing
   * store. THE host-side invalidation verb -- when the host rewrites a
   * vector's data (Preload, a new upload), it calls this, and kernels
   * never carry per-vector "drop my cache" flags of their own.
   *
   * Host-only and QUIESCENT-only: call it between kernel launches, never
   * while one is resident. It rewrites the device page and batch tables to
   * their pristine construction state, which is only sound when nothing is
   * pinned, fetching, or flushing -- exactly the state a returned kernel
   * leaves behind (guards died in scope, AwaitFlush settled the puts).
   */
  void ClearCache() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      InitPageTable(kv.second);
      InitBatchTable(kv.second);
      // The cache is empty again, so the resident claim is void until a
      // Prefetch re-establishes it.
      kv.second.hdr.no_evict_ = 0;
      PublishHeader(kv.second);
    }
#endif
  }

  /**
   * Prefault pages [pg_lo, pg_hi) into the caches of blocks
   * [blk_lo, blk_hi) FROM THE HOST -- outside any kernel, and therefore
   * outside its timing. The kernel then starts with the pages resident and
   * its holds all take the fast path; nothing faults, nothing parks.
   *
   * Each page's bytes are fetched from the backing store ONCE and copied
   * into every target block's claimed frame. Placement uses the SAME
   * DeviceVector::WaySlot function the device looks up with. A page whose
   * candidate ways are all occupied by other pages is skipped -- it simply
   * demand-faults later -- so this is always safe, merely best-effort.
   *
   * QUIESCENT-ONLY, like ClearCache: the host rewrites live table entries.
   * The intended sequence is Preload/upload, ClearCache, Prefetch, launch.
   *
   * @return pages fetched from the backing store (once each).
   */
  clio::run::u64 Prefetch(clio::run::u64 pg_lo, clio::run::u64 pg_hi,
                          clio::run::u32 blk_lo = 0,
                          clio::run::u32 blk_hi = ~0u) {
    clio::run::u64 fetched = 0;
#if CTP_ENABLE_CUDA
    if (pg_hi <= pg_lo) return 0;
    const clio::run::u32 ppb = pages_per_block_;

    for (auto &kv : devs_) {
      DevState &st = kv.second;
      // Whether this call leaves the vector FULLY RESIDENT -- every page
      // placed in every target block, with a dedicated slot each. Only
      // then may the device skip the eviction machinery
      // (VecHeader::no_evict_); anything less and a demand fault is still
      // reachable, so the handshake must stay armed.
      const clio::run::u64 total_pages = NumPagesOf(st.hdr);
      bool all_placed = (pg_lo == 0) && (pg_hi >= total_pages) &&
                        (total_pages <= ppb);
      const clio::run::u32 b_lo = (blk_lo < nblocks_) ? blk_lo : nblocks_;
      const clio::run::u32 b_hi = (blk_hi < nblocks_) ? blk_hi : nblocks_;
      if (b_lo >= b_hi) continue;
      // Pull the affected block tables down once, place, push back once.
      const size_t ntb = static_cast<size_t>(b_hi - b_lo) * ppb;
      std::vector<Page> tbl(ntb);
      Page *dev_tbl =
          reinterpret_cast<Page *>(st.table_base) +
          static_cast<size_t>(b_lo) * ppb;
      ctp::GpuApi::Memcpy(tbl.data(), dev_tbl, ntb * sizeof(Page));
      fetched = DownloadPages(
          pg_lo, pg_hi, [&](clio::run::u64 pg, const char *bytes) {
            for (clio::run::u32 b = b_lo; b < b_hi; ++b) {
              Page *bt = tbl.data() + static_cast<size_t>(b - b_lo) * ppb;
              clio::run::u32 slot = ~0u;
              const clio::run::u32 d = DeviceVector<T>::Ways(ppb);
              for (clio::run::u32 w = 0; w < d; ++w) {
                const clio::run::u32 i = DeviceVector<T>::WaySlot(pg, w, ppb);
                if (bt[i].page_num == pg) {
                  slot = i;   // already mapped: refresh its bytes
                  break;
                }
                if (slot == ~0u && bt[i].page_num == kNoPage &&
                    !bt[i].fetching && !bt[i].flushing && bt[i].pins == 0) {
                  slot = i;   // first free way
                }
              }
              if (slot == ~0u) {
                all_placed = false;
                continue;   // every way taken: demand-fault
              }
              Page &p = bt[slot];
              p.page_num = pg;
              p.score = 1.0f;
              p.user_score = 0.0f;
              p.has_user = 0;
              p.last_access = 0;
              p.pins = 0;
              p.dirty = 0;
              p.flushing = 0;
              p.fetching = 0;
              p.evicting = 0;
              p.rescoring = 0;
              // p.data is the slot's fixed device frame; land the bytes.
              ctp::GpuApi::Memcpy(static_cast<char *>(p.data), bytes,
                                  page_bytes_);
            }
          });
      ctp::GpuApi::Memcpy(dev_tbl, tbl.data(), ntb * sizeof(Page));
      if (fetched < (pg_hi - pg_lo)) all_placed = false;
      st.hdr.no_evict_ = all_placed ? 1u : 0u;
      PublishHeader(st);
    }
#else
    (void) pg_lo; (void) pg_hi; (void) blk_lo; (void) blk_hi;
#endif
    return fetched;
  }

  /**
   * DIAGNOSTIC: every slot must own the frame its index names --
   * InitPageTable sets slot i -> pages_base + i * page_bytes and nothing may
   * ever rewrite it. A violation means a reader can be handed a pointer into
   * a frame no transfer for its page ever targeted.
   * @return number of slots whose data pointer disagrees with their index.
   */
  clio::run::u64 AuditFrames(const char *when) {
    clio::run::u64 bad = 0;
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      DevState &st = kv.second;
      const clio::run::u64 nslots =
          static_cast<clio::run::u64>(nblocks_) * pages_per_block_;
      std::vector<Page> tbl(static_cast<size_t>(nslots));
      ctp::GpuApi::Memcpy(tbl.data(), reinterpret_cast<Page *>(st.table_base),
                          nslots * sizeof(Page));
      for (clio::run::u64 i = 0; i < nslots; ++i) {
        const char *want = st.pages_base + i * page_bytes_;
        if (static_cast<const char *>(tbl[i].data) != want) {
          if (bad < 8) {
            std::fprintf(stderr,
                         "[frame-audit %s] slot %llu: data=%p want=%p "
                         "(off by %lld frames), page_num=%lld\n",
                         when, (unsigned long long)i, tbl[i].data,
                         (const void *)want,
                         (long long)((static_cast<const char *>(tbl[i].data) -
                                      want) / (long long)page_bytes_),
                         (long long)tbl[i].page_num);
          }
          ++bad;
        }
      }
    }
#else
    (void)when;
#endif
    return bad;
  }

  /**
   * DIAGNOSTIC (chasing intermittent post-Prefetch corruption): compare every
   * resident cache frame in blocks [blk_lo, blk_hi) against the caller's
   * expected bytes, and independently re-get each page from the backing
   * store and compare that too. `expect(pg)` returns the PageBytes() the
   * page was uploaded with. Logs each mismatch to stderr.
   * @return mismatch count (frames + gets).
   */
  template <typename ExpectFn>
  clio::run::u64 VerifyCache(clio::run::u64 pg_lo, clio::run::u64 pg_hi,
                             clio::run::u32 blk_lo, clio::run::u32 blk_hi,
                             ExpectFn expect) {
    clio::run::u64 bad = 0;
#if CTP_ENABLE_CUDA
    const clio::run::u32 ppb = pages_per_block_;
    std::vector<char> frame(page_bytes_);
    // (1) the frames the kernel will actually read
    for (auto &kv : devs_) {
      DevState &st = kv.second;
      // Whether this call leaves the vector FULLY RESIDENT -- every page
      // placed in every target block, with a dedicated slot each. Only
      // then may the device skip the eviction machinery
      // (VecHeader::no_evict_); anything less and a demand fault is still
      // reachable, so the handshake must stay armed.
      const clio::run::u64 total_pages = NumPagesOf(st.hdr);
      bool all_placed = (pg_lo == 0) && (pg_hi >= total_pages) &&
                        (total_pages <= ppb);
      const clio::run::u32 b_lo = (blk_lo < nblocks_) ? blk_lo : nblocks_;
      const clio::run::u32 b_hi = (blk_hi < nblocks_) ? blk_hi : nblocks_;
      if (b_lo >= b_hi) continue;
      const size_t ntb = static_cast<size_t>(b_hi - b_lo) * ppb;
      std::vector<Page> tbl(ntb);
      Page *dev_tbl = reinterpret_cast<Page *>(st.table_base) +
                      static_cast<size_t>(b_lo) * ppb;
      ctp::GpuApi::Memcpy(tbl.data(), dev_tbl, ntb * sizeof(Page));
      for (clio::run::u64 pg = pg_lo; pg < pg_hi; ++pg) {
        const char *want = expect(pg);
        for (clio::run::u32 b = b_lo; b < b_hi; ++b) {
          Page *bt = tbl.data() + static_cast<size_t>(b - b_lo) * ppb;
          const clio::run::u32 d = DeviceVector<T>::Ways(ppb);
          for (clio::run::u32 w = 0; w < d; ++w) {
            const clio::run::u32 i = DeviceVector<T>::WaySlot(pg, w, ppb);
            if (bt[i].page_num != pg) continue;
            if (bt[i].data == nullptr) {
              std::fprintf(stderr,
                           "[vfy] blk=%u pg=%llu slot=%u NULL frame ptr\n",
                           b, (unsigned long long)pg, i);
              ++bad;
              break;
            }
            ctp::GpuApi::Memcpy(frame.data(),
                                static_cast<const char *>(bt[i].data),
                                page_bytes_);
            if (std::memcmp(frame.data(), want, page_bytes_) != 0) {
              size_t o = 0;
              while (o < page_bytes_ && frame[o] == want[o]) ++o;
              std::fprintf(stderr,
                           "[vfy] blk=%u pg=%llu slot=%u FRAME mismatch at "
                           "byte %zu: got %08x want %08x\n",
                           b, (unsigned long long)pg, i, o,
                           *reinterpret_cast<const unsigned *>(
                               frame.data() + (o & ~3ull)),
                           *reinterpret_cast<const unsigned *>(
                               want + (o & ~3ull)));
              ++bad;
            }
            break;
          }
        }
      }
    }
    // (2) the backing store itself, re-got fresh
    std::vector<char> gotbuf(page_bytes_);
    for (clio::run::u64 pg = pg_lo; pg < pg_hi; ++pg) {
      const char *want = expect(pg);
      clio::run::u64 n = DownloadPages(pg, pg + 1,
                                       [&](clio::run::u64, const char *bb) {
                                         std::memcpy(gotbuf.data(), bb,
                                                     page_bytes_);
                                       });
      if (n == 0) {
        std::fprintf(stderr, "[vfy] pg=%llu GET returned nothing\n",
                     (unsigned long long)pg);
        ++bad;
        continue;
      }
      if (std::memcmp(gotbuf.data(), want, page_bytes_) != 0) {
        size_t o = 0;
        while (o < page_bytes_ && gotbuf[o] == want[o]) ++o;
        std::fprintf(stderr,
                     "[vfy] pg=%llu GET mismatch at byte %zu: got %08x "
                     "want %08x\n",
                     (unsigned long long)pg, o,
                     *reinterpret_cast<const unsigned *>(gotbuf.data() +
                                                         (o & ~3ull)),
                     *reinterpret_cast<const unsigned *>(want + (o & ~3ull)));
        ++bad;
      }
    }
#else
    (void)pg_lo; (void)pg_hi; (void)blk_lo; (void)blk_hi; (void)expect;
#endif
    return bad;
  }

  /**
   * DIAGNOSTIC: how many pages are held by MORE THAN ONE slot of the same
   * block table, and how many of those duplicates are dirty.
   *
   * A page in two slots is the hazard WaySlot's comment warns about: a
   * racy "not resident" probe re-fetches a page that is already present,
   * and afterwards a lookup can pick either copy. If one of them carries
   * writes the other does not, reads go back in time and updates vanish --
   * which is what an out-of-core run losing exactly one half-kick per page
   * looks like. Cheap, host-side, quiescent-only.
   *
   * @return duplicated page count; *dirty_dups gets those with >1 dirty.
   */
  clio::run::u64 CountDuplicateSlots(int dev_id, clio::run::u64 *dirty_dups) {
    clio::run::u64 dups = 0;
    if (dirty_dups != nullptr) *dirty_dups = 0;
#if CTP_ENABLE_CUDA
    auto it = devs_.find(dev_id);
    if (it == devs_.end()) return 0;
    DevState &st = it->second;
    const clio::run::u32 ppb = pages_per_block_;
    std::vector<Page> tbl(static_cast<size_t>(nblocks_) * ppb);
    ctp::GpuApi::Memcpy(tbl.data(), reinterpret_cast<Page *>(st.table_base),
                        tbl.size() * sizeof(Page));
    for (clio::run::u32 b = 0; b < nblocks_; ++b) {
      const Page *t = tbl.data() + static_cast<size_t>(b) * ppb;
      for (clio::run::u32 i = 0; i < ppb; ++i) {
        if (t[i].page_num == kNoPage) continue;
        for (clio::run::u32 j = i + 1; j < ppb; ++j) {
          if (t[j].page_num != t[i].page_num) continue;
          ++dups;
          if (dirty_dups != nullptr && (t[i].dirty || t[j].dirty)) {
            ++(*dirty_dups);
          }
        }
      }
    }
#else
    (void)dev_id;
#endif
    return dups;
  }

  /** Put pages [pg_lo, pg_hi): `fill(pg, buf)` writes page `pg`'s bytes into
   *  `buf` (PageBytes() long). */
  template <typename FillFn>
  bool PreloadPages(clio::run::u64 pg_lo, clio::run::u64 pg_hi, FillFn fill) {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    if (pg_hi <= pg_lo) return true;
    const clio::run::u64 depth =
        std::min(pg_hi - pg_lo, kHostPipelineDepth);
    std::vector<std::vector<char>> bufs(
        static_cast<size_t>(depth), std::vector<char>(page_bytes_));
    bool ok = true;
    for (clio::run::u64 pg = pg_lo; pg < pg_hi;) {
      const clio::run::u64 lo = pg;
      const clio::run::u64 hi = std::min(lo + depth, pg_hi);
      std::vector<decltype(core.AsyncPutBlob(tag_id_, std::string(), 0, 0,
                                             nullptr, 1.0f))> futs;
      futs.reserve(hi - lo);
      for (clio::run::u64 q = lo; q < hi; ++q) {
        std::vector<char> &buf = bufs[static_cast<size_t>(q - lo)];
        fill(q, buf.data());
        char name[32];
        PageBlobName(q, name);
        // Constant score, matching the device flush: a reput whose score
        // differs from the previous put relocates the blob, and gets can
        // then return the stale replica.
        futs.push_back(core.AsyncPutBlob(tag_id_, std::string(name), 0,
                                         page_bytes_, buf.data(), 1.0f));
      }
      for (auto &fut : futs) {
        fut.Wait();
        if (fut.get() == nullptr || fut->GetReturnCode() != 0) ok = false;
      }
      pg = hi;
    }
    return ok;
  }

  /** Preload the backing store with `count` elements from `src`, starting at
   *  page-aligned `elem_off`. The tail page is zero-padded. */
  bool Preload(const T *src, clio::run::u64 count,
               clio::run::u64 elem_off = 0) {
    const clio::run::u64 per = page_bytes_ / sizeof(T);
    if (per == 0 || (elem_off % per) != 0) return false;
    const clio::run::u64 pg0 = elem_off / per;
    const clio::run::u64 npg = (count + per - 1) / per;
    return PreloadPages(pg0, pg0 + npg, [&](clio::run::u64 pg, char *buf) {
      const clio::run::u64 e0 = (pg - pg0) * per;
      const clio::run::u64 nn = std::min(per, count - e0);
      std::memcpy(buf, src + e0, nn * sizeof(T));
      if (nn < per) {
        std::memset(buf + nn * sizeof(T), 0, (per - nn) * sizeof(T));
      }
    });
  }

  /** Get pages [pg_lo, pg_hi): `sink(pg, buf)` consumes each page that
   *  exists. A page the kernel never wrote is skipped -- that is the
   *  write-allocate case, not an error.
   *  @return pages delivered to `sink`. */
  template <typename SinkFn>
  clio::run::u64 DownloadPages(clio::run::u64 pg_lo, clio::run::u64 pg_hi,
                               SinkFn sink) {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    if (pg_hi <= pg_lo) return 0;
    const clio::run::u64 depth =
        std::min(pg_hi - pg_lo, kHostPipelineDepth);
    std::vector<std::vector<char>> bufs(
        static_cast<size_t>(depth), std::vector<char>(page_bytes_));
    clio::run::u64 got = 0;
    for (clio::run::u64 pg = pg_lo; pg < pg_hi;) {
      const clio::run::u64 lo = pg;
      const clio::run::u64 hi = std::min(lo + depth, pg_hi);
      std::vector<decltype(core.AsyncGetBlob(tag_id_, std::string(), 0, 0, 0u,
                                             nullptr))> futs;
      futs.reserve(hi - lo);
      for (clio::run::u64 q = lo; q < hi; ++q) {
        char name[32];
        PageBlobName(q, name);
        futs.push_back(core.AsyncGetBlob(
            tag_id_, std::string(name), 0, page_bytes_, 0u,
            bufs[static_cast<size_t>(q - lo)].data()));
      }
      for (clio::run::u64 q = lo; q < hi; ++q) {
        auto &fut = futs[static_cast<size_t>(q - lo)];
        fut.Wait();
        if (fut.get() == nullptr || fut->GetReturnCode() != 0) continue;
        sink(q, static_cast<const char *>(
                    bufs[static_cast<size_t>(q - lo)].data()));
        ++got;
      }
      pg = hi;
    }
    return got;
  }

  /** Read `count` elements into `dst` from page-aligned `elem_off`. Pages
   *  never written leave their span of `dst` untouched.
   *  @return pages copied. */
  clio::run::u64 Download(T *dst, clio::run::u64 count,
                          clio::run::u64 elem_off = 0) {
    const clio::run::u64 per = page_bytes_ / sizeof(T);
    if (per == 0 || (elem_off % per) != 0) return 0;
    const clio::run::u64 pg0 = elem_off / per;
    const clio::run::u64 npg = (count + per - 1) / per;
    return DownloadPages(
        pg0, pg0 + npg, [&](clio::run::u64 pg, const char *buf) {
          const clio::run::u64 e0 = (pg - pg0) * per;
          const clio::run::u64 nn = std::min(per, count - e0);
          std::memcpy(dst + e0, buf, nn * sizeof(T));
        });
  }

 private:
  // NOTE: forward-declared in the public section above; the definition must
  // carry the same access, or clang rejects the redeclaration ([class.mem]).
 public:
  struct DevState {
    int gpu_id = 0;
    DeviceVector<T> view;
    /** Host-side image of the shared state, and its device copy. */
    VecHeader hdr;
    VecHeader *d_hdr = nullptr;
    char *pages_base = nullptr;     // page bytes
    char *table_base = nullptr;     // Page[] table
    char *tasks_base = nullptr;     // task slots
    char *multi_task_base = nullptr;  // batched-put task slots
    char *multi_tbl_base = nullptr;   // MultiBatch[] table
    char *cscratch_base = nullptr;    // per-slot compressed-image scratch
    ctp::ipc::AllocatorId cscratch_alloc;
    ctp::ipc::AllocatorId multi_tbl_alloc;
    ctp::ipc::AllocatorId pages_alloc;
    ctp::ipc::AllocatorId table_alloc;
    ctp::ipc::AllocatorId tasks_alloc;
    unsigned long long *stats = nullptr;   // [faults, puts, evicts], or null
  };

 private:
  /**
   * Allocate and register everything one GPU needs, then publish a view.
   *
   * Three device backends: the page bytes, the page table, and the task
   * slots. They are separate because only the task slots must be reachable
   * by the runtime through a registered allocator id; keeping them apart
   * makes that requirement explicit rather than incidental.
   */
  void BuildDevice(int gpu_id) {
    auto *ipc = CLIO_CPU_IPC;
    DevState st;
    st.gpu_id = gpu_id;
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nblocks_) * pages_per_block_;

    // Page bytes are MANAGED when a GPU codec is in play, plain device memory
    // otherwise.
    //
    // A GPU codec cannot run in the context whose kernel is spinning on the
    // fault, so it runs in its own context -- and a plain cudaMalloc pointer is
    // only valid in the context that made it. Managed memory is addressable
    // from every context on the device, which is what lets the codec
    // decompress STRAIGHT INTO the page the fault is waiting on: no host
    // staging, no peer copy, no second buffer. That is the whole point of
    // compressing on the GPU and it is not expressible with kDeviceMem.
    //
    // It is not free: managed pages measured ~12% slower on the raw read path
    // (1.95 vs 2.22 GB/s), which is migration bookkeeping nobody who is not
    // compressing should pay. So the choice follows the codec.
    //
    // Only the page bytes are affected either way. The page table and task
    // slots stay device-resident: the kernel alone touches them, never the
    // codec, so managing them would buy nothing and cost migration.
    const auto page_kind =
        IsGpuCodec(compress_lib_)
            ? clio::run::gpu::IpcManager::MemKind::kManagedUvm
            : clio::run::gpu::IpcManager::MemKind::kDeviceMem;
    st.pages_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, page_kind, nslots * page_bytes_, &st.pages_base);
    st.table_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nslots * sizeof(Page), &st.table_base);
    // Per-slot scratch for a SPILLED encoded page's stored image, so the
    // faulting warp can decode it without the compressor. Same memory kind as
    // the pages: the bdev writes into it from the runtime side, so it has to
    // be addressable there too. One page's worth per slot -- a stored image
    // is never larger than the page, or it would have been stored raw.
    // Only allocated for a GPU codec; nothing else can decode in-kernel.
    if (IsGpuCodec(compress_lib_)) {
      st.cscratch_alloc = ipc->AllocateAndRegisterGpuBackend(
          gpu_id, page_kind, nslots * page_bytes_, &st.cscratch_base);
    }
    // Batched writeback slots: enough per block that the block's WHOLE page
    // cache flushes in that many submissions (256 pages -> 4 batches). The
    // submission is the cost on this path, so the count is derived from the
    // cache size rather than being a tunable.
    const clio::run::u64 multi_per_block =
        (pages_per_block_ + clio::cte::core::kPodMultiMax - 1) /
        clio::cte::core::kPodMultiMax;
    const clio::run::u64 nbatch =
        static_cast<clio::run::u64>(nblocks_) * multi_per_block;
    const clio::run::u64 scalar_task_bytes =
        nslots * (sizeof(PutSlot) + sizeof(GetSlot) + sizeof(RescoreSlot));
    const clio::run::u64 multi_task_bytes =
        nbatch *
        (sizeof(MultiPutSlot) + sizeof(MultiGetSlot) + sizeof(MultiScoreSlot));
    // The batch tasks share the per-page slots' backend ON PURPOSE. SlotPtr
    // stamps `task_alloc_id_` -- ONE allocator id -- on every task it sends,
    // so a task living in a different registered backend reaches the runtime
    // labeled with the wrong allocator and is never resolved: the kernel then
    // waits forever on a completion that cannot come.
    const clio::run::u64 task_bytes = scalar_task_bytes + multi_task_bytes;
    st.tasks_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem, task_bytes,
        &st.tasks_base);
    st.multi_task_base = st.tasks_base + scalar_task_bytes;

    st.multi_tbl_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nbatch * sizeof(MultiBatch), &st.multi_tbl_base);

    if (st.pages_alloc.IsNull() || st.table_alloc.IsNull() ||
        st.tasks_alloc.IsNull() || st.multi_tbl_alloc.IsNull()) {
      throw std::runtime_error("gpu_vector: device backend allocation failed");
    }

    // Build the page table on the host, then upload it once. Doing this
    // device-side would mean a kernel writing pointers it cannot compute.
    InitPageTable(st);

    // The tasks must be CONSTRUCTED, not zeroed: each one now embeds its own
    // RunContext (the separate FutureShm is gone), and the runtime derefs it
    // on arrival -- a zeroed slot reaches the worker as "null RunContext --
    // task not executing" and aborts. So build them here with their real
    // constructors and upload; the kernel then only mutates POD input fields,
    // which is exactly the contract IpcGpu2Cpu documents.
    std::vector<char> tasks(static_cast<size_t>(task_bytes), 0);
    // ToLocalCpu, not Local: this is how a GPU-submitted task is routed to
    // a CPU worker for execution.
    const clio::run::PoolQuery local = clio::run::PoolQuery::ToLocalCpu();
    for (clio::run::u64 i = 0; i < nslots; ++i) {
      char *slot = tasks.data() +
                   i * (sizeof(PutSlot) + sizeof(GetSlot) + sizeof(RescoreSlot));
      new (slot) PutSlot(clio::run::CreateTaskId(), clio::cte::core::kCtePoolId,
                         local, tag_id_, "p0", 0, page_bytes_,
                         ctp::ipc::ShmPtr<>::GetNull(), 0.0f,
                         clio::cte::core::Context(), 0);
      new (slot + sizeof(PutSlot))
          GetSlot(clio::run::CreateTaskId(), clio::cte::core::kCtePoolId, local,
                  tag_id_, "p0", 0, page_bytes_, 0,
                  ctp::ipc::ShmPtr<>::GetNull(), clio::cte::core::Context());
      new (slot + sizeof(PutSlot) + sizeof(GetSlot))
          RescoreSlot(clio::run::CreateTaskId(), clio::cte::core::kCtePoolId,
                      local, tag_id_, "p0", 0.0f);
      // Give each task its RunContext NOW. The constructors leave it null,
      // and it rides along in the memcpy'd task on the GPU path -- a task
      // that arrives without one aborts the worker ("null RunContext -- task
      // not executing"). EnsureRunCtx is exactly this: allocate the storage,
      // no container resolution, which is what the client send path does
      // before handing a task over.
      // Stamp the POD size into the task's own future. RecvIn reads it from
      // the queue entry to know how many bytes to copy back off the device
      // WITHOUT dereferencing the task first; a zero here is reported as
      // "kernel did not stamp fut_.task_size_ before Send".
      reinterpret_cast<PutSlot *>(slot)->fut_.task_size_ =
          static_cast<clio::run::u32>(sizeof(PutSlot));
      reinterpret_cast<GetSlot *>(slot + sizeof(PutSlot))->fut_.task_size_ =
          static_cast<clio::run::u32>(sizeof(GetSlot));
      reinterpret_cast<RescoreSlot *>(slot + sizeof(PutSlot) + sizeof(GetSlot))
          ->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(RescoreSlot));
    }
    UploadBytes(tasks.data(), st.tasks_base, task_bytes);

    // Batch tasks: same rules as the per-page slots -- CONSTRUCTED (a zeroed
    // task arrives at the worker with a null RunContext and aborts it) and
    // stamped with their POD size so RecvIn knows how many bytes to copy back
    // without dereferencing the task.
    {
      const size_t pair =
          sizeof(MultiPutSlot) + sizeof(MultiGetSlot) + sizeof(MultiScoreSlot);
      std::vector<char> mtasks(static_cast<size_t>(nbatch) * pair, 0);
      std::vector<MultiBatch> mtbl(static_cast<size_t>(nbatch));
      for (clio::run::u64 i = 0; i < nbatch; ++i) {
        char *slot = mtasks.data() + static_cast<size_t>(i) * pair;
        new (slot) MultiPutSlot(clio::run::CreateTaskId(),
                                clio::cte::core::kCtePoolId, local, tag_id_);
        new (slot + sizeof(MultiPutSlot))
            MultiGetSlot(clio::run::CreateTaskId(), clio::cte::core::kCtePoolId,
                         local, tag_id_);
        reinterpret_cast<MultiPutSlot *>(slot)->fut_.task_size_ =
            static_cast<clio::run::u32>(sizeof(MultiPutSlot));
        reinterpret_cast<MultiGetSlot *>(slot + sizeof(MultiPutSlot))
            ->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(MultiGetSlot));
        new (slot + sizeof(MultiPutSlot) + sizeof(MultiGetSlot))
            MultiScoreSlot(clio::run::CreateTaskId(),
                           clio::cte::core::kCtePoolId, local, tag_id_);
        reinterpret_cast<MultiScoreSlot *>(slot + sizeof(MultiPutSlot) +
                                           sizeof(MultiGetSlot))
            ->fut_.task_size_ =
            static_cast<clio::run::u32>(sizeof(MultiScoreSlot));
      }
      UploadBytes(mtasks.data(), st.multi_task_base,
                  static_cast<clio::run::u64>(nbatch) * pair);
    }
    InitBatchTable(st);

    // One device-global counter per vector, feeding unique task ids.
    void *seq = nullptr;
#if CTP_ENABLE_CUDA
    seq = ctp::GpuApi::Malloc<void>(sizeof(unsigned long long));
    if (seq != nullptr) {
      ctp::GpuApi::Memset(seq, 0, sizeof(unsigned long long));
    } else {
      seq = nullptr;
    }
#endif

    // One page-table lock per block, zeroed (0 == free).
    void *locks = nullptr;
#if CTP_ENABLE_CUDA
    locks = ctp::GpuApi::Malloc<void>(nblocks_ * sizeof(int));
    if (locks != nullptr) {
      ctp::GpuApi::Memset(locks, 0, nblocks_ * sizeof(int));
    } else {
      throw std::runtime_error("gpu_vector: page-table lock allocation failed");
    }
#endif

    // Per-block outstanding-transfer counters, zeroed. Advisory fast-out for
    // the reap/vote scans (see VecHeader::xfer_cnt_).
    void *xfers = nullptr;
#if CTP_ENABLE_CUDA
    if (std::getenv("CLIO_GV_NO_XFERCNT") != nullptr) {
      // Bisect switch: null counter -> XferIdle() always false -> the scans
      // run unconditionally, exactly the pre-counter behavior.
    } else if ((xfers = ctp::GpuApi::Malloc<void>(
                    nblocks_ * sizeof(unsigned int))) != nullptr) {
      ctp::GpuApi::Memset(xfers, 0, nblocks_ * sizeof(unsigned int));
    } else {
      throw std::runtime_error("gpu_vector: xfer counter allocation failed");
    }
#endif

    // Admission counter for hold sets (see VecHeader::hold_admits_). One
    // u32 for the whole vector; the cap is the table's slot count, so the
    // slots blocks have collectively reserved can never exceed the slots
    // that exist.
    // OPT-IN (CLIO_GV_ADMIT=1). Admission provably converts the small-cache
    // livelock into a slow-but-correct run -- a 32-slot/64-block config that
    // wedged indefinitely completes with it -- but as currently sized it
    // reserves a chunk's WORST-CASE guard count, which admits only a few of
    // 64 blocks and costs 2.5-3x (48 slots: 43 ms/step without, 112-124
    // with). Until the reservation is sized from the true concurrent hold
    // set rather than a bound, paying that on every run is the wrong
    // default. A null counter makes Enter/ExitHoldSet no-ops.
    void *admits = nullptr;
    if (std::getenv("CLIO_GV_ADMIT") != nullptr) {
      admits = ctp::GpuApi::Malloc<void>(sizeof(unsigned int));
      if (admits == nullptr) {
        throw std::runtime_error("gpu_vector: admission counter alloc failed");
      }
      ctp::GpuApi::Memset(admits, 0, sizeof(unsigned int));
    }

    VecHeader v;   // filled here, then uploaded; the view only points at it
    v.hold_admits_ = static_cast<unsigned int *>(admits);
    v.hold_admit_cap_ = pages_per_block_;
    v.xfer_cnt_ = static_cast<unsigned int *>(xfers);
    v.block_locks_ = static_cast<int *>(locks);
    v.task_seq_ = static_cast<unsigned long long *>(seq);
    v.tag_id_ = tag_id_;
    v.pages_ = reinterpret_cast<Page *>(st.table_base);
    v.nblocks_ = nblocks_;
    v.pages_per_block_ = pages_per_block_;
    v.page_bytes_ = page_bytes_;
    v.elems_per_page_ = page_bytes_ / sizeof(T);
    // Precompute the shift so device-side element access is a shift and a mask
    // instead of a 64-bit divide, which a GPU has to emulate in software.
    v.page_shift_ = 0;
    v.page_mask_ = 0;
    const clio::run::u64 epp = v.elems_per_page_;
    if (epp != 0 && (epp & (epp - 1)) == 0) {
      clio::run::u32 sh = 0;
      while ((static_cast<clio::run::u64>(1) << sh) < epp) ++sh;
      v.page_shift_ = sh;
      v.page_mask_ = epp - 1;
    }
    v.size_ = num_elems_;
    v.pool_id_ = storage_pool_id_;
    v.core_pool_id_ =
        core_pool_id_.IsNull() ? storage_pool_id_ : core_pool_id_;
    v.cscratch_ = st.cscratch_base;
    v.compress_lib_ = compress_lib_;
    v.compress_preset_ = compress_preset_;
    v.task_alloc_id_ = st.tasks_alloc;
    v.multi_ = reinterpret_cast<MultiBatch *>(st.multi_tbl_base);
    v.multi_per_block_ = static_cast<clio::run::u32>(multi_per_block);
    st.hdr = v;
    PublishHeader(st);
    devs_[gpu_id] = st;
  }

  /** Build one GPU's page table in its pristine state -- every slot empty,
   *  every flag clear, the slot/task pointers wired -- and upload it. Used
   *  at construction AND by ClearCache(); resetting is re-initializing. */
  void InitPageTable(DevState &st) {
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nblocks_) * pages_per_block_;
    std::vector<Page> table(static_cast<size_t>(nslots));
    for (clio::run::u64 i = 0; i < nslots; ++i) {
      Page &p = table[static_cast<size_t>(i)];
      p.page_num = kNoPage;
      p.dbg_get_page = kNoPage;
      p.data = st.pages_base + i * page_bytes_;
      p.score = 0.0f;
      p.user_score = 0.0f;
      p.has_user = 0;      // no rescore hint until the kernel sets one
      p.last_access = 0;
      p.pins = 0;
      p.dirty = 0;
      p.flushing = 0;
      p.fetching = 0;
      p.evicting = 0;
      p.rescoring = 0;
      p.seq = 0;
      char *slot = st.tasks_base +
                   i * (sizeof(PutSlot) + sizeof(GetSlot) + sizeof(RescoreSlot));
      p.put = reinterpret_cast<PutSlot *>(slot);
      p.get = reinterpret_cast<GetSlot *>(slot + sizeof(PutSlot));
      p.rescore = reinterpret_cast<RescoreSlot *>(
          slot + sizeof(PutSlot) + sizeof(GetSlot));
      p.put_fut = clio::run::gpu::Future<clio::cte::core::PodPutBlobTask>();
      p.get_fut = clio::run::gpu::Future<clio::cte::core::PodGetBlobTask>();
      p.rescore_fut =
          clio::run::gpu::Future<clio::cte::core::PodReorganizeBlobTask>();
    }
    UploadTable(table, st.table_base);
  }

  /** Same for the batched-op table: pointers wired, futures null, nothing
   *  pending. The batch TASK slots are untouched -- they are constructed
   *  once (they embed live RunContexts) and only their POD inputs mutate. */
  void InitBatchTable(DevState &st) {
    const clio::run::u64 multi_per_block =
        (pages_per_block_ + clio::cte::core::kPodMultiMax - 1) /
        clio::cte::core::kPodMultiMax;
    const clio::run::u64 nbatch =
        static_cast<clio::run::u64>(nblocks_) * multi_per_block;
    const size_t pair =
        sizeof(MultiPutSlot) + sizeof(MultiGetSlot) + sizeof(MultiScoreSlot);
    std::vector<MultiBatch> mtbl(static_cast<size_t>(nbatch));
    for (clio::run::u64 i = 0; i < nbatch; ++i) {
      MultiBatch &b = mtbl[static_cast<size_t>(i)];
      char *dev = st.multi_task_base + static_cast<size_t>(i) * pair;
      b.put = reinterpret_cast<MultiPutSlot *>(dev);
      b.get = reinterpret_cast<MultiGetSlot *>(dev + sizeof(MultiPutSlot));
      b.score = reinterpret_cast<MultiScoreSlot *>(
          dev + sizeof(MultiPutSlot) + sizeof(MultiGetSlot));
      b.put_fut = clio::run::gpu::Future<clio::cte::core::PodMultiPutBlobTask>();
      b.get_fut = clio::run::gpu::Future<clio::cte::core::PodMultiGetBlobTask>();
      b.score_fut = clio::run::gpu::Future<clio::cte::core::PodMultiScoreTask>();
      b.async_pending = 0;
      b.async_n = 0;
      b.score_pending = 0;
      b.put_pending = 0;
      b.put_n = 0;
    }
    UploadBytes(mtbl.data(), st.multi_tbl_base, nbatch * sizeof(MultiBatch));
  }

  static void UploadTable(const std::vector<Page> &table, void *dst) {
#if CTP_ENABLE_CUDA
    ctp::GpuApi::Memcpy<char>(static_cast<char *>(dst),
                              reinterpret_cast<const char *>(table.data()),
                              table.size() * sizeof(Page));
#else
    (void) table;
    (void) dst;
#endif
  }

  static void UploadBytes(const void *src, void *dst, clio::run::u64 bytes) {
#if CTP_ENABLE_CUDA
    ctp::GpuApi::Memcpy(dst, src, static_cast<size_t>(bytes));
#else
    (void) src;
    (void) dst;
    (void) bytes;
#endif
  }

  /**
   * Release everything one device view owns.
   *
   * A vector allocates three registered GPU backends (page bytes, page table,
   * task slots) plus two raw device buffers (the task-id counter and, if
   * enabled, the stats block). All five are owned by THIS object and are
   * released here, so a Vector that goes out of scope leaves nothing behind --
   * the caller never frees anything by hand. Vectors that must outlive a scope
   * or be shared belong in a shared_ptr; copying is deleted precisely because
   * two owners of these allocations would double-free them.
   */
  static void Free(DevState &st) {
    auto *ipc = CLIO_CPU_IPC;
    const auto gpu = static_cast<clio::run::u32>(st.gpu_id);
    if (ipc != nullptr) {
      if (!st.pages_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.pages_alloc);
      if (!st.table_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.table_alloc);
      if (!st.tasks_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.tasks_alloc);
      if (!st.multi_tbl_alloc.IsNull()) {
        ipc->FreeGpuBackend(gpu, st.multi_tbl_alloc);
      }
    }
    st.pages_alloc = ctp::ipc::AllocatorId::GetNull();
    st.table_alloc = ctp::ipc::AllocatorId::GetNull();
    st.tasks_alloc = ctp::ipc::AllocatorId::GetNull();
    st.multi_tbl_alloc = ctp::ipc::AllocatorId::GetNull();
#if CTP_ENABLE_CUDA
    if (st.stats != nullptr) {
      ctp::GpuApi::Free(st.stats);
      st.stats = nullptr;
    }
    if (st.hdr.task_seq_ != nullptr) {
      ctp::GpuApi::Free(st.hdr.task_seq_);
    }
    if (st.hdr.hold_admits_ != nullptr) {
      ctp::GpuApi::Free(st.hdr.hold_admits_);
      st.hdr.hold_admits_ = nullptr;
    }
    if (st.hdr.block_locks_ != nullptr) {
      ctp::GpuApi::Free(st.hdr.block_locks_);
    }
    if (st.hdr.xfer_cnt_ != nullptr) {
      ctp::GpuApi::Free(st.hdr.xfer_cnt_);
    }
#endif
    st.hdr.block_locks_ = nullptr;
    st.hdr.xfer_cnt_ = nullptr;
    st.hdr.task_seq_ = nullptr;
    st.hdr.stat_faults_ = nullptr;
    st.hdr.stat_puts_ = nullptr;
    st.hdr.stat_evicts_ = nullptr;
    st.hdr.pages_ = nullptr;
    st.pages_base = nullptr;
    st.table_base = nullptr;
    st.tasks_base = nullptr;
  }

  /** Where page tasks are addressed. Point this at a COMPRESSOR pool to have
   *  pages stored compressed; it interposes and forwards to the core. */
  clio::run::PoolId core_pool_id_;
  clio::run::PoolId storage_pool_id_;
  /** Codec wire id stamped on page puts; 0 stores raw. */
  int compress_lib_ = 0;
  /** Compressor preset: 1 FAST, 2 BALANCED, 3 BEST. Defaults to FAST, and
   *  deliberately so. A page store sits inside a device fault, and the
   *  compressor maps BALANCED (the module-wide default) to LZ4_compress_HC
   *  level 6 -- tens of MB/s on data that does not compress easily, against
   *  hundreds for LZ4_compress_default. Paying HC prices to shrink a page
   *  cache entry is the wrong trade: the page is about to be read back over
   *  the bus, and the ratio difference between FAST and HC is small next to
   *  the order of magnitude in throughput. */
  int compress_preset_ = 1;
  clio::cte::core::TagId tag_id_;
  clio::run::u64 page_bytes_;
  clio::run::u32 nblocks_;
  clio::run::u32 pages_per_block_;
  clio::run::u64 num_elems_;
  std::map<int, DevState> devs_;
};

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_GPU_VECTOR_H_
