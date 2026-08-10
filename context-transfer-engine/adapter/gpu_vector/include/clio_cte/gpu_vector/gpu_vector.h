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
         int compress_lib = 0, int compress_preset = 1)
      : storage_pool_id_(storage_pool_id.IsNull()
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
    clio::run::u64 get_errors = 0;      // gets that returned non-zero
    clio::run::u64 pf_dropped = 0;      // async batch hints dropped (slots busy)
    clio::run::u64 verify_ok = 0;       // re-probe after compute: page intact
    clio::run::u64 verify_lost = 0;     // re-probe: page evicted mid-read
  };
  static constexpr int kNumStats = 11;
  bool fully_device_mapped_ = false;

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
        cudaMalloc(&st.d_hdr, sizeof(VecHeader)) != cudaSuccess) {
      throw std::runtime_error("gpu_vector: header allocation failed");
    }
    cudaMemcpy(st.d_hdr, &st.hdr, sizeof(VecHeader), cudaMemcpyHostToDevice);
    st.view.h_ = st.d_hdr;
#endif
  }

  void EnableStats() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      if (kv.second.stats != nullptr) continue;
      void *mem = nullptr;
      const size_t bytes = kNumStats * sizeof(unsigned long long);
      if (cudaMalloc(&mem, bytes) != cudaSuccess) {
        throw std::runtime_error("gpu_vector: stats allocation failed");
      }
      cudaMemset(mem, 0, bytes);
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
      if (getenv("CLIO_FAULT_HIST") != nullptr) {
        const clio::run::u64 npg =
            (kv.second.hdr.size_ + kv.second.hdr.page_bytes_ - 1) /
            kv.second.hdr.page_bytes_;
        void *hm = nullptr;
        if (cudaMalloc(&hm, npg * sizeof(unsigned int)) == cudaSuccess) {
          cudaMemset(hm, 0, npg * sizeof(unsigned int));
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
  /** @return true when BuildDeviceTierMap mapped every page (reads never
   *  fault; warm-up and slot prefetching are pointless). */
  bool FullyDeviceMapped() const { return fully_device_mapped_; }

  void BuildDeviceTierMap() {
#if CTP_ENABLE_CUDA
    if (devs_.empty()) return;
    // GENERALITY GATE: a direct pointer is only valid when the STORED bytes
    // are the representation the kernels read. Identity-stored tags (raw
    // F16/Q4_K, and FP8 whose kernels decode e4m3 in-register) qualify; a
    // tag read through a codec (compress_lib_ != 0) stores an nvcomp stream
    // — those pages MUST materialize through the fetch+decompress path, so
    // they never map. The page cache's contract: the tier holds the encoded
    // representation, the cache holds the decoded one; mapping is the
    // identity-codec specialization of that contract, not a bypass.
    if (compress_lib_ != 0) return;
    const auto &h0 = devs_.begin()->second.hdr;
    const clio::run::u64 npages =
        (h0.size_ + h0.page_bytes_ - 1) / h0.page_bytes_;
    if (npages == 0) return;
    unsigned long long pool = 0, toff = 0, ssz = 0;
    if (clio_cte_locate(&tag_id_, "p0", &pool, &toff, &ssz) != 0) return;
    char *base = clio_direct_dev_base(pool);
    if (base == nullptr) return;   // not a device tier
    std::vector<unsigned long long> offs(npages, ~0ull);
    std::vector<unsigned long long> csz(npages, 0);
    clio::run::u64 mapped = 0;
    for (clio::run::u64 pg = 0; pg < npages; ++pg) {
      char name[32];
      PageBlobName(pg, name);
      unsigned long long p2 = 0, o2 = 0, s2 = 0;
      if (clio_cte_locate(&tag_id_, name, &p2, &o2, &s2) == 0 && p2 == pool) {
        offs[pg] = o2;
        csz[pg] = s2;
        ++mapped;
      }
    }
    void *dev_offs = nullptr;
    void *dev_csz = nullptr;
    if (cudaMalloc(&dev_offs, npages * sizeof(unsigned long long)) !=
            cudaSuccess ||
        cudaMalloc(&dev_csz, npages * sizeof(unsigned long long)) !=
            cudaSuccess) {
      return;
    }
    cudaMemcpy(dev_offs, offs.data(), npages * sizeof(unsigned long long),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csz, csz.data(), npages * sizeof(unsigned long long),
               cudaMemcpyHostToDevice);
    for (auto &kv : devs_) {
      kv.second.hdr.tier_base_ = base;
      kv.second.hdr.tier_off_ =
          static_cast<const unsigned long long *>(dev_offs);
      kv.second.hdr.tier_csize_ =
          static_cast<const unsigned long long *>(dev_csz);
      PublishHeader(kv.second);
    }
    fully_device_mapped_ = (mapped == npages);
    std::fprintf(stderr,
                 "gpu_vector: DEVICE-TIER MAP active — %llu/%llu pages "
                 "zero-copy\n",
                 (unsigned long long) mapped, (unsigned long long) npages);
#endif
  }

  void ResetStats() {
#if CTP_ENABLE_CUDA
    for (auto &kv : devs_) {
      if (kv.second.stats != nullptr) {
        cudaMemset(kv.second.stats, 0, kNumStats * sizeof(unsigned long long));
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
    const clio::run::u64 npg =
        (it->second.hdr.size_ + it->second.hdr.page_bytes_ - 1) /
        it->second.hdr.page_bytes_;
    h.resize(npg);
    cudaMemcpy(h.data(), it->second.hdr.fault_hist_,
               npg * sizeof(unsigned int), cudaMemcpyDeviceToHost);
#endif
    return h;
  }

  Stats ReadStats(int dev_id) const {
    Stats s;
#if CTP_ENABLE_CUDA
    auto it = devs_.find(dev_id);
    if (it == devs_.end() || it->second.stats == nullptr) return s;
    unsigned long long h[kNumStats] = {0};
    cudaMemcpy(h, it->second.stats, sizeof(h), cudaMemcpyDeviceToHost);
    s.faults = h[0];
    s.puts = h[1];
    s.evicts = h[2];
    s.prefetches = h[3];
    s.prefetch_hits = h[4];
    s.rescores = h[5];
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

 private:
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
    ctp::ipc::AllocatorId multi_tbl_alloc;
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
        nbatch * (sizeof(MultiPutSlot) + sizeof(MultiGetSlot));
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
      p.fetching = 0;
      p.rescoring = 0;
      p.seq = 0;
      char *slot = tasks_bytes +
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
      const size_t pair = sizeof(MultiPutSlot) + sizeof(MultiGetSlot);
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
        char *dev = st.multi_task_base + static_cast<size_t>(i) * pair;
        mtbl[static_cast<size_t>(i)].put = reinterpret_cast<MultiPutSlot *>(dev);
        mtbl[static_cast<size_t>(i)].get =
            reinterpret_cast<MultiGetSlot *>(dev + sizeof(MultiPutSlot));
        mtbl[static_cast<size_t>(i)].put_fut =
            clio::run::gpu::Future<clio::cte::core::PodMultiPutBlobTask>();
        mtbl[static_cast<size_t>(i)].get_fut =
            clio::run::gpu::Future<clio::cte::core::PodMultiGetBlobTask>();
        mtbl[static_cast<size_t>(i)].async_pending = 0;
        mtbl[static_cast<size_t>(i)].async_n = 0;
      }
      UploadBytes(mtasks.data(), st.multi_task_base,
                  static_cast<clio::run::u64>(nbatch) * pair);
      UploadBytes(mtbl.data(), st.multi_tbl_base,
                  nbatch * sizeof(MultiBatch));
    }

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

    VecHeader v;   // filled here, then uploaded; the view only points at it
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
    v.compress_lib_ = compress_lib_;
    v.compress_preset_ = compress_preset_;
    v.task_alloc_id_ = st.tasks_alloc;
    v.multi_ = reinterpret_cast<MultiBatch *>(st.multi_tbl_base);
    v.multi_per_block_ = static_cast<clio::run::u32>(multi_per_block);
    st.hdr = v;
    PublishHeader(st);
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
      cudaFree(st.stats);
      st.stats = nullptr;
    }
    if (st.hdr.task_seq_ != nullptr) {
      cudaFree(st.hdr.task_seq_);
    }
    if (st.hdr.block_locks_ != nullptr) {
      cudaFree(st.hdr.block_locks_);
    }
#endif
    st.hdr.block_locks_ = nullptr;
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
