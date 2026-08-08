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
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/device_vector.h>
#include <clio_cte/gpu_vector/page.h>

#include <cstring>
#include <new>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

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
         int compress_lib = 0)
      : storage_pool_id_(storage_pool_id.IsNull()
                             ? clio::cte::core::kCtePoolId
                             : storage_pool_id),
        compress_lib_(compress_lib),
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
  };

  /**
   * Turn on device-side paging counters for every device view.
   *
   * Off by default -- the counters cost an atomic per page event, which is
   * nothing against a fault's round trip but is not free. Tests use them to
   * assert the paging POLICY (hit/miss, writeback, victim choice), which the
   * returned data alone cannot distinguish.
   */
  void EnableStats() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      if (kv.second.stats != nullptr) continue;
      void *mem = nullptr;
      if (cudaMalloc(&mem, 3 * sizeof(unsigned long long)) != cudaSuccess) {
        throw std::runtime_error("gpu_vector: stats allocation failed");
      }
      cudaMemset(mem, 0, 3 * sizeof(unsigned long long));
      auto *c = static_cast<unsigned long long *>(mem);
      kv.second.stats = c;
      kv.second.view.stat_faults_ = c;
      kv.second.view.stat_puts_ = c + 1;
      kv.second.view.stat_evicts_ = c + 2;
    }
#endif
  }

  void ResetStats() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      if (kv.second.stats != nullptr) {
        cudaMemset(kv.second.stats, 0, 3 * sizeof(unsigned long long));
      }
    }
#endif
  }

  Stats ReadStats(int dev_id) const {
    Stats s;
#if CTP_ENABLE_CUDA
    auto it = devs_.find(dev_id);
    if (it == devs_.end() || it->second.stats == nullptr) return s;
    unsigned long long h[3] = {0, 0, 0};
    cudaMemcpy(h, it->second.stats, sizeof(h), cudaMemcpyDeviceToHost);
    s.faults = h[0];
    s.puts = h[1];
    s.evicts = h[2];
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

 private:
  struct DevState {
    int gpu_id = 0;
    DeviceVector<T> view;
    char *pages_base = nullptr;     // page bytes
    char *table_base = nullptr;     // Page[] table
    char *tasks_base = nullptr;     // task slots
    ctp::ipc::AllocatorId pages_alloc;
    ctp::ipc::AllocatorId table_alloc;
    ctp::ipc::AllocatorId tasks_alloc;
    unsigned long long *stats = nullptr;   // [faults, puts, evicts], or null
  };

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

    st.pages_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nslots * page_bytes_, &st.pages_base);
    st.table_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
        nslots * sizeof(Page), &st.table_base);
    const clio::run::u64 task_bytes =
        nslots * (sizeof(PutSlot) + sizeof(GetSlot) + sizeof(RescoreSlot));
    st.tasks_alloc = ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem, task_bytes,
        &st.tasks_base);
    if (st.pages_alloc.IsNull() || st.table_alloc.IsNull() ||
        st.tasks_alloc.IsNull()) {
      throw std::runtime_error("gpu_vector: device backend allocation failed");
    }

    // Build the page table on the host, then upload it once. Doing this
    // device-side would mean a kernel writing pointers it cannot compute.
    std::vector<Page> table(static_cast<size_t>(nslots));
    char *pages_bytes = st.pages_base;
    char *tasks_bytes = st.tasks_base;
    for (clio::run::u64 i = 0; i < nslots; ++i) {
      Page &p = table[static_cast<size_t>(i)];
      p.page_num = kNoPage;
      p.data = pages_bytes + i * page_bytes_;
      p.score = 0.0f;
      p.last_access = 0;
      p.dirty = 0;
      p.flushing = 0;
      p.seq = 0;
      char *slot = tasks_bytes +
                   i * (sizeof(PutSlot) + sizeof(GetSlot) + sizeof(RescoreSlot));
      p.put = reinterpret_cast<PutSlot *>(slot);
      p.get = reinterpret_cast<GetSlot *>(slot + sizeof(PutSlot));
      p.rescore = reinterpret_cast<RescoreSlot *>(
          slot + sizeof(PutSlot) + sizeof(GetSlot));
      p.put_fut = clio::run::gpu::Future<clio::cte::core::PodPutBlobTask>();
    }
    UploadTable(table, st.table_base);

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

    // One device-global counter per vector, feeding unique task ids.
    void *seq = nullptr;
#if CTP_ENABLE_CUDA
    if (cudaMalloc(&seq, sizeof(unsigned long long)) == cudaSuccess) {
      cudaMemset(seq, 0, sizeof(unsigned long long));
    } else {
      seq = nullptr;
    }
#endif

    // One page-table lock per block, zeroed (0 == free).
    void *locks = nullptr;
#if CTP_ENABLE_CUDA
    if (cudaMalloc(&locks, nblocks_ * sizeof(int)) == cudaSuccess) {
      cudaMemset(locks, 0, nblocks_ * sizeof(int));
    } else {
      throw std::runtime_error("gpu_vector: page-table lock allocation failed");
    }
#endif

    DeviceVector<T> v;
    v.block_locks_ = static_cast<int *>(locks);
    v.task_seq_ = static_cast<unsigned long long *>(seq);
    v.tag_id_ = tag_id_;
    v.pages_ = reinterpret_cast<Page *>(st.table_base);
    v.nblocks_ = nblocks_;
    v.pages_per_block_ = pages_per_block_;
    v.page_bytes_ = page_bytes_;
    v.elems_per_page_ = page_bytes_ / sizeof(T);
    v.size_ = num_elems_;
    v.pool_id_ = storage_pool_id_;
    v.compress_lib_ = compress_lib_;
    v.task_alloc_id_ = st.tasks_alloc;
    st.view = v;
    devs_[gpu_id] = st;
  }

  static void UploadTable(const std::vector<Page> &table, void *dst) {
#if CTP_ENABLE_CUDA
    cudaMemcpy(dst, table.data(), table.size() * sizeof(Page),
               cudaMemcpyHostToDevice);
#else
    (void) table;
    (void) dst;
#endif
  }

  static void UploadBytes(const void *src, void *dst, clio::run::u64 bytes) {
#if CTP_ENABLE_CUDA
    cudaMemcpy(dst, src, static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
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
    }
    st.pages_alloc = ctp::ipc::AllocatorId::GetNull();
    st.table_alloc = ctp::ipc::AllocatorId::GetNull();
    st.tasks_alloc = ctp::ipc::AllocatorId::GetNull();
#if CTP_ENABLE_CUDA
    if (st.stats != nullptr) {
      cudaFree(st.stats);
      st.stats = nullptr;
    }
    if (st.view.task_seq_ != nullptr) {
      cudaFree(st.view.task_seq_);
    }
    if (st.view.block_locks_ != nullptr) {
      cudaFree(st.view.block_locks_);
    }
#endif
    st.view.block_locks_ = nullptr;
    st.view.task_seq_ = nullptr;
    st.view.stat_faults_ = nullptr;
    st.view.stat_puts_ = nullptr;
    st.view.stat_evicts_ = nullptr;
    st.view.pages_ = nullptr;
    st.pages_base = nullptr;
    st.table_base = nullptr;
    st.tasks_base = nullptr;
  }

  /** Where page tasks are addressed. Point this at a COMPRESSOR pool to have
   *  pages stored compressed; it interposes and forwards to the core. */
  clio::run::PoolId storage_pool_id_;
  /** Codec wire id stamped on page puts; 0 stores raw. */
  int compress_lib_ = 0;
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
