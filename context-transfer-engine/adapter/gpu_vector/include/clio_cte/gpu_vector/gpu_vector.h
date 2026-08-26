/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * gpu_vector: a vector whose backing store is the CTE and whose working set
 * lives in GPU memory, paged in on demand from inside a kernel.
 *
 * Host side (this class) owns the allocations and hands out a device view;
 * device side (DeviceVector) does the paging. A kernel can allocate nothing,
 * so every buffer, task and page table is allocated and registered here.
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

#include <cstring>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace clio::cte::gpu_vector {

/**
 * Resume a parked block ONLY when the completion word its wait tag names has
 * flipped.
 *
 * Pass this as RunToCompletion's resume_when. Without it the driver relaunches
 * every parked block every round, which polls a fault by kernel launch: the
 * block wakes, sees its transfer has not landed, and parks again. With it the
 * host reads the completion word and holds the block back until the task is
 * actually done -- one relaunch per fault.
 *
 * A wait tag of 0 means "no condition", so those blocks always resume.
 */
inline bool ResumeWhenComplete(clio::run::u32 /*block*/, clio::run::u64 tag) {
#if CTP_ENABLE_CUDA
  if (tag == 0) return true;
  unsigned int done = 0;
  ctp::GpuApi::Memcpy(reinterpret_cast<char *>(&done),
                      reinterpret_cast<const char *>(tag), sizeof(done));
  return (done & 1u) != 0u;
#else
  (void) tag;
  return true;
#endif
}

// The host class is HOST-ONLY. Its body calls host-only client methods
// (AsyncGetOrCreateTag), and those are non-dependent names, so the device
// pass would fail to parse them even though it never instantiates the class.
#if !CTP_IS_DEVICE_PASS

/**
 * The process-wide fatal channel: eight slots of PINNED HOST memory that the
 * device writes just before it traps.
 *
 * Host memory because nothing else survives a trap -- device printf is
 * buffered and dies with the context, and so does device memory, which is why
 * a trapping kernel reports only "CUDA Error 715: an illegal instruction" and
 * nothing about the cause. Under unified addressing the device can store
 * straight into a cudaMallocHost allocation, so this costs nothing until it
 * is used.
 *
 * One buffer for the process, not one per vector: a trap is fatal, so only
 * the first one matters, and every vector can point at the same slots.
 */
inline unsigned long long *FatalSlots() {
#if CTP_ENABLE_CUDA
  static unsigned long long *slots = [] {
    auto *p = ctp::GpuApi::MallocHost<unsigned long long>(
        8 * sizeof(unsigned long long));
    if (p != nullptr) std::memset(p, 0, 8 * sizeof(unsigned long long));
    return p;
  }();
  return slots;
#else
  return nullptr;
#endif
}

/** Human-readable form of whatever the device latched, or "" if nothing. */
inline std::string FatalReport() {
  unsigned long long *f = FatalSlots();
  if (f == nullptr || f[0] == kFatalNone) return std::string();
  const char *what = "unknown";
  switch (f[0]) {
    case kFatalInitBlock:   what = "Init(block) beyond the task table"; break;
    case kFatalNotResident: what = "HoldPage: page not resident"; break;
    case kFatalNotCovered:  what = "HoldPage: range never fetched"; break;
    case kFatalUnbound:     what = "vector used before Init()"; break;
    case kFatalSetFull:     what = "AllocatePage: set full"; break;
    case kFatalFlushSplit:  what = "flush range needs more records"; break;
    default: break;
  }
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "[gpu_vector] DEVICE FATAL %llu (%s): a1=%llu a2=%llu a3=%llu "
                "block=%llu", f[0], what, f[1], f[2], f[3], f[4]);
  return std::string(buf);
}

template <typename T>
class Vector {
 public:
  /** Counters, populated only when EnableStats() was called. */
  struct Stats {
    clio::run::u64 faults = 0;   // pages read in
    clio::run::u64 puts = 0;     // pages written back
    clio::run::u64 evicts = 0;   // frames reclaimed
    clio::run::u64 get_errors = 0;   // faults that returned non-zero
    clio::run::u64 put_errors = 0;   // writebacks that returned non-zero
  };

  /**
   * @param tag_name   CTE tag representing this vector
   * @param gpu_ids    GPUs to build device views for
   * @param page_bytes page granularity
   * @param nblocks    CUDA BLOCKS the kernel will launch. This dimensions the
   *        per-block task sets -- each block owns its MultiGet and MultiPut,
   *        because two blocks staging into one slot would overwrite each
   *        other's records mid-transfer -- and, by default, the set count.
   * @param set_size   frames in one associative set. Enough of them to cover
   *        every block that can hold a page of one set at once, plus room for
   *        hash collisions.
   * @param num_elems  logical length of the vector
   * @param nsets      associative sets; 0 means "as many as there are blocks".
   *        Separate from nblocks because the cache's geometry and the grid's
   *        width are unrelated numbers -- they were one field, and reading
   *        `nblocks` as either "tables" or "sets" depending on context is
   *        exactly what made this class hard to follow.
   *
   * THERE IS ONE CACHE. `nblocks` sets, `pages_per_block` frames each, and a
   * page's home set is Hash(page_num) % nblocks -- a function of the PAGE, so
   * every CUDA block looks in the same place and a page touched by all of
   * them is stored ONCE. The per-block page table this class used to build is
   * gone: it stored a hot page once per block, and every cross-block
   * correctness rule (invalidate after a resort, borrow a peer's frame,
   * publish before another block can read) existed to paper over that.
   */
  Vector(const std::string &tag_name, const std::vector<int> &gpu_ids,
         clio::run::u64 page_bytes, clio::run::u32 nblocks,
         clio::run::u32 set_size, clio::run::u64 num_elems,
         clio::run::PoolId storage_pool_id = clio::run::PoolId::GetNull(),
         int compress_lib = 0, int compress_preset = 1,
         clio::run::u32 nsets = 0, clio::run::u32 capacity_pages = 0)
      : storage_pool_id_(storage_pool_id.IsNull() ? clio::cte::core::kCtePoolId
                                                  : storage_pool_id),
        compress_lib_(compress_lib),
        compress_preset_(compress_preset),
        page_bytes_(page_bytes),
        nblocks_(nblocks),
        nsets_(nsets != 0 ? nsets : nblocks),
        set_size_(set_size),
        // CAPACITY IS PAGES, NOT SLOTS. Slots are 64-byte tags; capacity is
        // the page-sized storage behind them, and it is what the cache
        // actually costs. 0 keeps the old meaning (one region per slot) so a
        // caller that has not thought about it is no worse off.
        capacity_pages_(capacity_pages != 0
                            ? capacity_pages
                            : (nsets != 0 ? nsets : nblocks) * set_size),
        num_elems_(num_elems) {
    if (page_bytes_ == 0 || nblocks_ == 0 || set_size_ == 0 || nsets_ == 0) {
      throw std::runtime_error("gpu_vector: zero page size / blocks / sets");
    }
    clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
    auto tag = cte.AsyncGetOrCreateTag(tag_name);
    tag.Wait();
    if (tag->GetReturnCode() != 0) {
      throw std::runtime_error("gpu_vector: could not resolve tag " + tag_name);
    }
    tag_id_ = tag->tag_id_;
    try {
      for (int gpu : gpu_ids) BuildDevice(gpu);
    } catch (...) {
      for (auto &kv : devs_) Free(kv.second);
      devs_.clear();
      throw;
    }
  }

  ~Vector() {
    for (auto &kv : devs_) Free(kv.second);
  }

  Vector(const Vector &) = delete;
  Vector &operator=(const Vector &) = delete;

  /** The value a kernel takes by value. */
  DeviceVector<T> GetDevice(int gpu_id) const {
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) {
      throw std::runtime_error("gpu_vector: no device view for this GPU");
    }
    return it->second.view;
  }

  clio::cte::core::TagId TagId() const { return tag_id_; }
  clio::run::u64 PageBytes() const { return page_bytes_; }
  clio::run::u64 ElemsPerPage() const { return page_bytes_ / sizeof(T); }
  clio::run::u64 NumPages() const {
    const clio::run::u64 epp = ElemsPerPage();
    return (num_elems_ + epp - 1) / epp;
  }

  void EnableStats() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      if (kv.second.stats != nullptr) continue;
      const size_t bytes = 5 * sizeof(unsigned long long);
      auto *c = reinterpret_cast<unsigned long long *>(
          ctp::GpuApi::Malloc<char>(bytes));
      if (c == nullptr) throw std::runtime_error("gpu_vector: stats alloc failed");
      ctp::GpuApi::Memset(c, 0, bytes);
      kv.second.stats = c;
      kv.second.hdr.stat_faults_ = c;
      kv.second.hdr.stat_puts_ = c + 1;
      kv.second.hdr.stat_evicts_ = c + 2;
      kv.second.hdr.stat_get_errors_ = c + 3;
      kv.second.hdr.stat_put_errors_ = c + 4;
      PublishHeader(kv.second);
    }
#endif
  }

  /**
   * HOST-SIDE data movement. Not part of the device API: a kernel never calls
   * these. They exist because host data has to reach the vector somehow, and
   * because benches reset residency between measured rounds.
   */

  /**
   * A page's home set, host side. MUST mirror DeviceVector::SetOf exactly --
   * the host and the device disagreeing about where a page lives is a page
   * the device can never find and a slot nothing can reclaim.
   */
  clio::run::u32 HomeSet(clio::run::u64 pn) const {
    const clio::run::u64 mixed = (pn ^ (pn >> 29)) * 0x9E3779B97F4A7C15ull;
    return static_cast<clio::run::u32>((mixed >> 32) % nsets_);
  }

  /**
   * Make pages [pg_lo, pg_hi) resident, best effort: ONCE each, in the page's
   * home set, because the whole grid shares one cache.
   *
   * `tables` named how many per-block tables to fill and is ignored -- there
   * is one cache. Filling a table at a time put each page in as many wrong
   * sets as there were blocks: invisible to a device lookup, which searches
   * only the home set, while occupying every slot the real fetch then needed.
   */
  void Prefetch(clio::run::u64 pg_lo, clio::run::u64 pg_hi, int gpu_id = 0,
                clio::run::u32 tables = 0) {
#if CTP_ENABLE_CUDA
    (void) tables;
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) return;
    clio::cte::core::Client core(storage_pool_id_);
    PrefetchShared(it->second, pg_lo, pg_hi, core);
#else
    (void) pg_lo; (void) pg_hi; (void) gpu_id; (void) tables;
#endif
  }

  /** Drop every resident page. Dirty pages are DISCARDED -- flush first. */
  void ClearCache(int gpu_id = 0) {
#if CTP_ENABLE_CUDA
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) return;
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nsets_) * set_size_;
    std::vector<Page> tbl(static_cast<size_t>(nslots));
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(tbl.data()),
                        it->second.table_base,
                        static_cast<size_t>(nslots * sizeof(Page)));
    for (auto &p : tbl) {
      p.page_num = kNoPage;
      p.data = nullptr;      // the region goes back to its owner's free list
      p.pins = 0;
      p.flushing = 0;
      p.fetching = 0;
      p.score = kDefaultScore;
      p.valid_lo = 0;
      p.valid_hi = 0;
      p.last_access = 0;
    }
    ctp::GpuApi::Memcpy(it->second.table_base,
                        reinterpret_cast<const char *>(tbl.data()),
                        static_cast<size_t>(nslots * sizeof(Page)));
    // AND EVERY REGION GOES BACK. Emptying the slots while leaving the free
    // lists as they were would strand the storage those slots held: the tags
    // say the cache is empty and the allocator says it has nothing to give.
    RebuildFreeLists(it->second, 0);
#else
    (void) gpu_id;
#endif
  }

  /** Write pages [pg_lo, pg_hi); `fill(pg, buf)` supplies each page's bytes. */
  template <typename FillFn>
  void PreloadPages(clio::run::u64 pg_lo, clio::run::u64 pg_hi, FillFn fill) {
    clio::cte::core::Client core(storage_pool_id_);
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    for (clio::run::u64 pg = pg_lo; pg < pg_hi; ++pg) {
      std::memset(buf.data(), 0, buf.size());
      fill(pg, buf.data());
      auto f = core.AsyncPutBlob(tag_id_, PageName(pg), 0, page_bytes_,
                                 buf.data(), 0.5f, PageContext(), 0u,
                                 clio::run::PoolQuery::Dynamic());
      f.Wait();
    }
  }

  /** Write `n` elements from host memory into the vector. */
  void Preload(const T *src, clio::run::u64 n) {
    const clio::run::u64 epp = ElemsPerPage();
    PreloadPages(0, (n + epp - 1) / epp, [&](clio::run::u64 pg, char *buf) {
      const clio::run::u64 off = pg * epp;
      const clio::run::u64 cnt = (epp < n - off) ? epp : (n - off);
      std::memcpy(buf, src + off, static_cast<size_t>(cnt * sizeof(T)));
    });
  }

  /** Read `n` elements out of the vector into host memory.
   *  @return elements actually read. */
  clio::run::u64 Download(T *dst, clio::run::u64 n) {
    const clio::run::u64 epp = ElemsPerPage();
    clio::run::u64 got = 0;
    DownloadPages(0, (n + epp - 1) / epp,
                  [&](clio::run::u64 pg, const char *bytes) {
                    const clio::run::u64 off = pg * epp;
                    if (off >= n) return;
                    const clio::run::u64 cnt = (epp < n - off) ? epp : (n - off);
                    std::memcpy(dst + off, bytes,
                                static_cast<size_t>(cnt * sizeof(T)));
                    got += cnt;
                  });
    return got;
  }

  /** Read pages [pg_lo, pg_hi); `sink(pg, bytes)` receives each page. */
  template <typename SinkFn>
  clio::run::u64 DownloadPages(clio::run::u64 pg_lo, clio::run::u64 pg_hi,
                               SinkFn sink) {
    clio::cte::core::Client core(storage_pool_id_);
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    clio::run::u64 got = 0;
    for (clio::run::u64 pg = pg_lo; pg < pg_hi; ++pg) {
      if (!ReadPage(core, pg, buf.data())) continue;
      sink(pg, static_cast<const char *>(buf.data()));
      ++got;
    }
    return got;
  }

  /** Zero the counters, for benches that measure per-round. */
  void ResetStats() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      if (kv.second.stats == nullptr) continue;
      ctp::GpuApi::Memset(kv.second.stats, 0, 5 * sizeof(unsigned long long));
    }
#endif
  }

  /**
   * Copy every frame's metadata to the host.
   *
   * THE DUPLICATION CHECK NEEDS TO SEE THE WHOLE ARRAY AT ONCE. "Is this
   * cache actually shared" is answered by counting how many frames hold the
   * same page_num: one in shared mode, up to one-per-CUDA-block in private
   * mode. That is not visible from any single set.
   */
  std::vector<Page> ReadTable(int gpu_id) const {
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) return {};
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nsets_) * set_size_;
    std::vector<Page> out(static_cast<size_t>(nslots));
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(out.data()),
                        reinterpret_cast<const char *>(it->second.table_base),
                        static_cast<size_t>(nslots * sizeof(Page)));
    return out;
  }

  /** How many frames hold the most-duplicated resident page. 1 == shared. */
  clio::run::u32 MaxPageCopies(int gpu_id) const {
    const std::vector<Page> tbl = ReadTable(gpu_id);
    std::map<clio::run::u64, clio::run::u32> seen;
    clio::run::u32 worst = 0;
    for (const Page &p : tbl) {
      if (p.page_num == kNoPage) continue;
      const clio::run::u32 n = ++seen[p.page_num];
      if (n > worst) worst = n;
    }
    return worst;
  }

  Stats ReadStats(int gpu_id) const {
    Stats s;
#if CTP_ENABLE_CUDA
    auto it = devs_.find(gpu_id);
    if (it == devs_.end() || it->second.stats == nullptr) return s;
    unsigned long long h[5] = {0, 0, 0, 0, 0};
    ctp::GpuApi::Memcpy(h, it->second.stats, sizeof(h));
    s.faults = h[0];
    s.puts = h[1];
    s.evicts = h[2];
    s.get_errors = h[3];
    s.put_errors = h[4];
#else
    (void) gpu_id;
#endif
    return s;
  }

 private:
  struct DevState {
    int gpu_id = 0;
    DeviceVector<T> view;
    VecHeader hdr;
    VecHeader *d_hdr = nullptr;
    char *pages_base = nullptr;    // page bytes
    char *table_base = nullptr;    // Page[]
    char *tasks_base = nullptr;    // the four task objects per block
    char *btbl_base = nullptr;     // BlockTasks[]
    ctp::ipc::AllocatorId pages_alloc;
    ctp::ipc::AllocatorId table_alloc;
    ctp::ipc::AllocatorId tasks_alloc;
    ctp::ipc::AllocatorId btbl_alloc;
    ctp::ipc::AllocatorId locks_alloc;
    ctp::ipc::AllocatorId count_alloc;
    char *locks_base = nullptr;
    char *count_base = nullptr;
    clio::run::u32 nregions = 0;
    clio::run::u32 per_block = 0;
    unsigned long long *stats = nullptr;
  };

  /** Bytes of one block's four tasks, laid out in this order. */
  static constexpr size_t kTaskSetBytes =
      sizeof(MultiPutSlot) + sizeof(MultiGetSlot);

  void BuildDevice(int gpu_id) {
    auto *ipc = CLIO_CPU_IPC;
    DevState st;
    st.gpu_id = gpu_id;
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nsets_) * set_size_;
    // REGIONS ARE THE COST; SLOTS ARE TAGS. Round the capacity up so every
    // block owns the same number, which is what makes ownership a divide
    // rather than a table.
    const clio::run::u32 per_block =
        (capacity_pages_ + nblocks_ - 1) / nblocks_;
    const clio::run::u32 nregions = per_block * nblocks_;

    st.pages_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        static_cast<size_t>(nregions) * page_bytes_, &st.pages_base);
    st.table_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nslots * sizeof(Page), &st.table_base);
    st.tasks_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nblocks_ * kTaskSetBytes, &st.tasks_base);
    st.btbl_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nblocks_ * sizeof(BlockTasks), &st.btbl_base);
    if (st.pages_alloc.IsNull() || st.table_alloc.IsNull() ||
        st.tasks_alloc.IsNull() || st.btbl_alloc.IsNull()) {
      throw std::runtime_error("gpu_vector: device backend allocation failed");
    }
    {
      // One lock per set and one per block free list, strided so no two share
      // a cache line, plus the free lists themselves: nregions region indices
      // and a head/tail per block. All of it is small -- four bytes per
      // region against the page it stands for.
      const clio::run::u32 kLockStride = 32;
      const size_t lock_ints =
          static_cast<size_t>(nsets_ + nblocks_) * kLockStride;
      st.locks_alloc = ipc->AllocateAndRegisterGpuBackend(
          gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          lock_ints * sizeof(int), &st.locks_base);
      st.count_alloc = ipc->AllocateAndRegisterGpuBackend(
          gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          (static_cast<size_t>(nregions) + 2 * nblocks_) *
              sizeof(clio::run::u32),
          &st.count_base);
      if (st.locks_alloc.IsNull() || st.count_alloc.IsNull()) {
        throw std::runtime_error("gpu_vector: shared-cache allocation failed");
      }
      ctp::GpuApi::Memset(st.locks_base, 0, lock_ints * sizeof(int));
    }

    InitPageTable(st, nslots);
    InitFreeLists(st, nregions, per_block);
    InitTasks(st);

    st.hdr.pages_ = reinterpret_cast<Page *>(st.table_base);
    st.hdr.tasks_ = reinterpret_cast<BlockTasks *>(st.btbl_base);
    st.hdr.page_bytes_ = page_bytes_;
    st.hdr.elems_per_page_ = page_bytes_ / sizeof(T);
    st.hdr.num_elems_ = num_elems_;
    st.hdr.nblocks_ = nblocks_;
    st.hdr.nsets_ = nsets_;
    st.hdr.set_size_ = set_size_;
    st.hdr.set_locks_ = reinterpret_cast<int *>(st.locks_base);
    // The block free-list locks live after the set locks in the same block.
    st.hdr.free_lock_ =
        reinterpret_cast<int *>(st.locks_base) +
        static_cast<size_t>(nsets_) * 32;
    st.nregions = nregions;
    st.per_block = per_block;
    st.hdr.regions_ = st.pages_base;
    st.hdr.nregions_ = nregions;
    st.hdr.regions_per_block_ = per_block;
    st.hdr.free_q_ = reinterpret_cast<clio::run::u32 *>(st.count_base);
    st.hdr.free_head_ = st.hdr.free_q_ + nregions;
    st.hdr.free_tail_ = st.hdr.free_head_ + nblocks_;
    st.hdr.tag_id_ = tag_id_;
    st.hdr.pool_id_ = storage_pool_id_;
    st.hdr.task_alloc_id_ = st.tasks_alloc;
    st.hdr.compress_lib_ = compress_lib_;
    st.hdr.compress_preset_ = compress_preset_;
    st.hdr.stat_faults_ = nullptr;
    st.hdr.stat_puts_ = nullptr;
    st.hdr.stat_evicts_ = nullptr;
    st.hdr.stat_get_errors_ = nullptr;
    st.hdr.stat_put_errors_ = nullptr;
    st.hdr.fatal_ = FatalSlots();
    PublishHeader(st);
    devs_[gpu_id] = st;
  }

  /** Build the Page[] on the host -- it holds pointers a kernel cannot
   *  compute -- then upload it once. */
  /**
   * Prefetch into a SHARED cache: one copy of each page, in its home set.
   *
   * Stops at the byte budget rather than at the end of the frame array. The
   * array is 2x over-provisioned for collisions, so filling it would hand the
   * device a cache twice the size it was asked for and leave cache_frames_
   * describing a budget already blown. A page whose home set is full is
   * simply left out -- residency here is best effort, and the device fault
   * path will bring it in.
   */
  void PrefetchShared(DevState &st, clio::run::u64 pg_lo, clio::run::u64 pg_hi,
                      clio::cte::core::Client &core) {
#if CTP_ENABLE_CUDA
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nsets_) * set_size_;
    std::vector<Page> tbl(static_cast<size_t>(nslots));
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(tbl.data()), st.table_base,
                        static_cast<size_t>(nslots * sizeof(Page)));
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    unsigned long long placed = 0;
    for (const auto &p : tbl) {
      if (p.page_num != kNoPage) ++placed;
    }
    const unsigned long long budget = nslots;
    for (clio::run::u64 pg = pg_lo; pg < pg_hi && placed < budget; ++pg) {
      const size_t base =
          static_cast<size_t>(HomeSet(pg)) * set_size_;
      size_t slot = base;
      const size_t end = base + set_size_;
      while (slot < end && tbl[slot].page_num != kNoPage) ++slot;
      if (slot >= end) continue;                 // home set full
      if (!ReadPage(core, pg, buf.data())) continue;
      // A SLOT NEEDS A REGION NOW. Hand them out in order and rebuild each
      // block's free list from what is left, which keeps the host and the
      // device agreeing about who owns what.
      if (placed >= st.nregions) break;
      const clio::run::u32 ridx = static_cast<clio::run::u32>(placed);
      char *region = st.pages_base + static_cast<size_t>(ridx) * page_bytes_;
      ctp::GpuApi::Memcpy(region, buf.data(),
                          static_cast<size_t>(page_bytes_));
      tbl[slot].data = region;
      tbl[slot].page_num = pg;
      tbl[slot].pins = 0;
      tbl[slot].flushing = 0;
      tbl[slot].fetching = 0;
      tbl[slot].score = kDefaultScore;
      tbl[slot].valid_lo = 0;
      tbl[slot].valid_hi = static_cast<clio::run::u32>(page_bytes_ / sizeof(T));
      ++placed;
    }
    ctp::GpuApi::Memcpy(st.table_base,
                        reinterpret_cast<const char *>(tbl.data()),
                        static_cast<size_t>(nslots * sizeof(Page)));
    RebuildFreeLists(st, static_cast<clio::run::u32>(placed));
#else
    (void) st; (void) pg_lo; (void) pg_hi; (void) core;
#endif
  }

  /**
   * Fill every block's free list with the regions it owns.
   *
   * Block b owns [b*per_block, (b+1)*per_block), which is what makes
   * RegionOwner a divide on the address instead of a lookup.
   */
  void InitFreeLists(DevState &st, clio::run::u32 nregions,
                     clio::run::u32 per_block) {
#if CTP_ENABLE_CUDA
    std::vector<clio::run::u32> q(nregions);
    for (clio::run::u32 i = 0; i < nregions; ++i) q[i] = i;
    std::vector<clio::run::u32> head(nblocks_, 0), tail(nblocks_, per_block);
    auto *base = reinterpret_cast<clio::run::u32 *>(st.count_base);
    UploadBytes(q.data(), reinterpret_cast<char *>(base),
                q.size() * sizeof(clio::run::u32));
    UploadBytes(head.data(), reinterpret_cast<char *>(base + nregions),
                head.size() * sizeof(clio::run::u32));
    UploadBytes(tail.data(),
                reinterpret_cast<char *>(base + nregions + nblocks_),
                tail.size() * sizeof(clio::run::u32));
#else
    (void) st; (void) nregions; (void) per_block;
#endif
  }

  /**
   * Rebuild every block's free list, given that regions [0, used) are now
   * held by slots. A block keeps the ones it owns that nobody took.
   */
  void RebuildFreeLists(DevState &st, clio::run::u32 used) {
#if CTP_ENABLE_CUDA
    if (st.count_base == nullptr || st.per_block == 0) return;
    std::vector<clio::run::u32> q(st.nregions, 0);
    std::vector<clio::run::u32> head(nblocks_, 0), tail(nblocks_, 0);
    for (clio::run::u32 b = 0; b < nblocks_; ++b) {
      const clio::run::u32 lo = b * st.per_block;
      clio::run::u32 n = 0;
      for (clio::run::u32 i = lo; i < lo + st.per_block; ++i) {
        if (i >= used) q[lo + n++] = i;
      }
      head[b] = 0;
      tail[b] = n;
    }
    auto *base = reinterpret_cast<clio::run::u32 *>(st.count_base);
    UploadBytes(q.data(), reinterpret_cast<char *>(base),
                q.size() * sizeof(clio::run::u32));
    UploadBytes(head.data(), reinterpret_cast<char *>(base + st.nregions),
                head.size() * sizeof(clio::run::u32));
    UploadBytes(tail.data(),
                reinterpret_cast<char *>(base + st.nregions + nblocks_),
                tail.size() * sizeof(clio::run::u32));
#else
    (void) st; (void) used;
#endif
  }

  void InitPageTable(DevState &st, clio::run::u64 nslots) {
    std::vector<Page> tbl(static_cast<size_t>(nslots));
    for (clio::run::u64 i = 0; i < nslots; ++i) {
      Page &p = tbl[static_cast<size_t>(i)];
      p.page_num = kNoPage;
      // A SLOT STARTS WITH NO STORAGE. It gets a region when it claims a
      // page and gives it back when it is evicted.
      p.data = nullptr;
      p.score = kDefaultScore;
      p.last_access = 0;
      p.pins = 0;
      p.valid_lo = 0;
      p.valid_hi = 0;
      p.flushing = 0;
      p.fetching = 0;
    }
    UploadBytes(tbl.data(), st.table_base, nslots * sizeof(Page));
  }

  /**
   * Construct each block's four tasks and its BlockTasks record.
   *
   * The tasks must be CONSTRUCTED, not zeroed: each embeds its own
   * RunContext and the runtime derefs it on arrival, so a zeroed slot
   * reaches the worker as "null RunContext" and aborts it. Each is also
   * stamped with its POD size, which RecvIn reads to know how many bytes to
   * copy back off the device without dereferencing the task.
   */
  void InitTasks(DevState &st) {
    const clio::run::PoolQuery local = clio::run::PoolQuery::ToLocalCpu();
    std::vector<char> raw(nblocks_ * kTaskSetBytes, 0);
    std::vector<BlockTasks> btbl(nblocks_);
    for (clio::run::u32 b = 0; b < nblocks_; ++b) {
      char *slot = raw.data() + static_cast<size_t>(b) * kTaskSetBytes;
      char *dev = st.tasks_base + static_cast<size_t>(b) * kTaskSetBytes;

      auto *fl = new (slot) MultiPutSlot(clio::run::CreateTaskId(),
                                         clio::cte::core::kCtePoolId, local,
                                         tag_id_);
      fl->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(MultiPutSlot));

      char *p2 = slot + sizeof(MultiPutSlot);
      auto *ft = new (p2) MultiGetSlot(clio::run::CreateTaskId(),
                                       clio::cte::core::kCtePoolId, local,
                                       tag_id_);
      ft->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(MultiGetSlot));

      BlockTasks &bt = btbl[b];
      std::memset(&bt, 0, sizeof(bt));
      bt.flush = reinterpret_cast<MultiPutSlot *>(dev);
      bt.fetch = reinterpret_cast<MultiGetSlot *>(dev + sizeof(MultiPutSlot));
    }
    UploadBytes(raw.data(), st.tasks_base, nblocks_ * kTaskSetBytes);
    UploadBytes(btbl.data(), st.btbl_base, nblocks_ * sizeof(BlockTasks));
  }

  void PublishHeader(DevState &st) {
#if CTP_ENABLE_CUDA
    if (st.d_hdr == nullptr) {
      st.d_hdr = ctp::GpuApi::Malloc<VecHeader>(sizeof(VecHeader));
      if (st.d_hdr == nullptr) {
        throw std::runtime_error("gpu_vector: header allocation failed");
      }
    }
    ctp::GpuApi::Memcpy(st.d_hdr, &st.hdr, sizeof(VecHeader));
    st.view = DeviceVector<T>(st.d_hdr);
#endif
  }

  /** A page's blob name: the page number in decimal. Matches what the
   *  runtime renders for a device request (Context::kBlobNameRawInt32). */
  std::string PageName(clio::run::u64 pg) const { return std::to_string(pg); }

  clio::cte::core::Context PageContext() const {
    clio::cte::core::Context c;
    c.compress_lib_ = compress_lib_;
    c.compress_preset_ = compress_preset_;
    return c;
  }

  /** One page's bytes into `out`; false when the page has no blob yet. */
  bool ReadPage(clio::cte::core::Client &core, clio::run::u64 pg, char *out) {
    auto ctx = PageContext();
    auto f = core.AsyncGetBlob(tag_id_, PageName(pg), 0, page_bytes_, 0u, out,
                               clio::run::PoolQuery::Dynamic(), ctx);
    f.Wait();
    return f.get() != nullptr && f->GetReturnCode() == 0;
  }

  static void UploadBytes(const void *src, void *dst, clio::run::u64 bytes) {
#if CTP_ENABLE_CUDA
    ctp::GpuApi::Memcpy(dst, src, static_cast<size_t>(bytes));
#else
    (void) src; (void) dst; (void) bytes;
#endif
  }

  static void Free(DevState &st) {
    auto *ipc = CLIO_CPU_IPC;
    const auto gpu = static_cast<clio::run::u32>(st.gpu_id);
    if (ipc != nullptr) {
      if (!st.pages_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.pages_alloc);
      if (!st.table_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.table_alloc);
      if (!st.tasks_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.tasks_alloc);
      if (!st.btbl_alloc.IsNull()) ipc->FreeGpuBackend(gpu, st.btbl_alloc);
    }
    st.pages_alloc = ctp::ipc::AllocatorId::GetNull();
    st.table_alloc = ctp::ipc::AllocatorId::GetNull();
    st.tasks_alloc = ctp::ipc::AllocatorId::GetNull();
    st.btbl_alloc = ctp::ipc::AllocatorId::GetNull();
#if CTP_ENABLE_CUDA
    if (st.stats != nullptr) {
      ctp::GpuApi::Free(st.stats);
      st.stats = nullptr;
    }
    if (st.d_hdr != nullptr) {
      ctp::GpuApi::Free(st.d_hdr);
      st.d_hdr = nullptr;
    }
#endif
  }

  clio::run::PoolId storage_pool_id_;
  int compress_lib_ = 0;
  int compress_preset_ = 1;
  clio::run::u64 page_bytes_ = 0;
  clio::run::u32 nblocks_ = 0;   // CUDA blocks == task sets
  clio::run::u32 nsets_ = 0;     // associative sets in the cache
  clio::run::u32 set_size_ = 0;  // TAGS per set
  clio::run::u32 capacity_pages_ = 0;  // page-sized regions in the allocator
  clio::run::u64 num_elems_ = 0;
  clio::cte::core::TagId tag_id_;
  std::map<int, DevState> devs_;
};

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_GPU_VECTOR_H_
