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

// The host class is HOST-ONLY. Its body calls host-only client methods
// (AsyncGetOrCreateTag), and those are non-dependent names, so the device
// pass would fail to parse them even though it never instantiates the class.
#if !CTP_IS_DEVICE_PASS

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
   * @param nblocks    page tables to create
   * @param pages_per_block resident pages each table may hold
   * @param num_elems  logical length of the vector
   */
  Vector(const std::string &tag_name, const std::vector<int> &gpu_ids,
         clio::run::u64 page_bytes, clio::run::u32 nblocks,
         clio::run::u32 pages_per_block, clio::run::u64 num_elems,
         clio::run::PoolId storage_pool_id = clio::run::PoolId::GetNull(),
         int compress_lib = 0, int compress_preset = 1)
      : storage_pool_id_(storage_pool_id.IsNull() ? clio::cte::core::kCtePoolId
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

  /** Make pages [pg_lo, pg_hi) resident in every table, best effort. */
  void Prefetch(clio::run::u64 pg_lo, clio::run::u64 pg_hi, int gpu_id = 0,
                clio::run::u32 tables = 0) {
#if CTP_ENABLE_CUDA
    auto it = devs_.find(gpu_id);
    if (it == devs_.end()) return;
    DevState &st = it->second;
    const clio::run::u32 ntab = (tables == 0) ? nblocks_ : tables;
    clio::cte::core::Client core(storage_pool_id_);
    std::vector<char> buf(static_cast<size_t>(page_bytes_));
    std::vector<Page> tbl(pages_per_block_);
    for (clio::run::u32 b = 0; b < ntab && b < nblocks_; ++b) {
      Page *dev = reinterpret_cast<Page *>(st.table_base) +
                  static_cast<size_t>(b) * pages_per_block_;
      ctp::GpuApi::Memcpy(reinterpret_cast<char *>(tbl.data()),
                          reinterpret_cast<const char *>(dev),
                          static_cast<size_t>(pages_per_block_ * sizeof(Page)));
      clio::run::u32 slot = 0;
      for (clio::run::u64 pg = pg_lo; pg < pg_hi; ++pg) {
        while (slot < pages_per_block_ && tbl[slot].page_num != kNoPage) ++slot;
        if (slot >= pages_per_block_) break;
        if (!ReadPage(core, pg, buf.data())) continue;
        ctp::GpuApi::Memcpy(static_cast<char *>(tbl[slot].data), buf.data(),
                            static_cast<size_t>(page_bytes_));
        tbl[slot].page_num = pg;
        tbl[slot].dirty = 0;
        tbl[slot].pins = 0;
        tbl[slot].flushing = 0;
        tbl[slot].fetching = 0;
        tbl[slot].score = kDefaultScore;
        ++slot;
      }
      ctp::GpuApi::Memcpy(reinterpret_cast<char *>(dev),
                          reinterpret_cast<const char *>(tbl.data()),
                          static_cast<size_t>(pages_per_block_ * sizeof(Page)));
    }
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
        static_cast<clio::run::u64>(nblocks_) * pages_per_block_;
    std::vector<Page> tbl(static_cast<size_t>(nslots));
    ctp::GpuApi::Memcpy(reinterpret_cast<char *>(tbl.data()),
                        it->second.table_base,
                        static_cast<size_t>(nslots * sizeof(Page)));
    for (auto &p : tbl) {
      p.page_num = kNoPage;
      p.dirty = 0;
      p.pins = 0;
      p.flushing = 0;
      p.fetching = 0;
      p.score = kDefaultScore;
      p.last_access = 0;
    }
    ctp::GpuApi::Memcpy(it->second.table_base,
                        reinterpret_cast<const char *>(tbl.data()),
                        static_cast<size_t>(nslots * sizeof(Page)));
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
    unsigned long long *stats = nullptr;
  };

  /** Bytes of one block's four tasks, laid out in this order. */
  static constexpr size_t kTaskSetBytes =
      sizeof(MultiPutSlot) + sizeof(GetSlot);

  void BuildDevice(int gpu_id) {
    auto *ipc = CLIO_CPU_IPC;
    DevState st;
    st.gpu_id = gpu_id;
    const clio::run::u64 nslots =
        static_cast<clio::run::u64>(nblocks_) * pages_per_block_;

    st.pages_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nslots * page_bytes_, &st.pages_base);
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

    InitPageTable(st, nslots);
    InitTasks(st);

    st.hdr.pages_ = reinterpret_cast<Page *>(st.table_base);
    st.hdr.tasks_ = reinterpret_cast<BlockTasks *>(st.btbl_base);
    st.hdr.page_bytes_ = page_bytes_;
    st.hdr.elems_per_page_ = page_bytes_ / sizeof(T);
    st.hdr.num_elems_ = num_elems_;
    st.hdr.nblocks_ = nblocks_;
    st.hdr.pages_per_block_ = pages_per_block_;
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
    PublishHeader(st);
    devs_[gpu_id] = st;
  }

  /** Build the Page[] on the host -- it holds pointers a kernel cannot
   *  compute -- then upload it once. */
  void InitPageTable(DevState &st, clio::run::u64 nslots) {
    std::vector<Page> tbl(static_cast<size_t>(nslots));
    for (clio::run::u64 i = 0; i < nslots; ++i) {
      Page &p = tbl[static_cast<size_t>(i)];
      p.page_num = kNoPage;
      p.data = st.pages_base + i * page_bytes_;
      p.score = kDefaultScore;
      p.last_access = 0;
      p.pins = 0;
      p.dirty = 0;
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
      auto *ft = new (p2)
          GetSlot(clio::run::CreateTaskId(), clio::cte::core::kCtePoolId, local,
                  tag_id_, "p0", 0, page_bytes_, 0,
                  ctp::ipc::ShmPtr<>::GetNull(), clio::cte::core::Context());
      ft->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(GetSlot));

      BlockTasks &bt = btbl[b];
      std::memset(&bt, 0, sizeof(bt));
      bt.flush = reinterpret_cast<MultiPutSlot *>(dev);
      bt.fault = reinterpret_cast<GetSlot *>(dev + sizeof(MultiPutSlot));
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
  clio::run::u32 nblocks_ = 0;
  clio::run::u32 pages_per_block_ = 0;
  clio::run::u64 num_elems_ = 0;
  clio::cte::core::TagId tag_id_;
  std::map<int, DevState> devs_;
};

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_GPU_VECTOR_H_
