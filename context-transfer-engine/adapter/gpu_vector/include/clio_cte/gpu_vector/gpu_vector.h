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
         clio::run::u32 pages_per_block, clio::run::u64 num_elems)
      : page_bytes_(page_bytes),
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
    for (int gpu : gpu_ids) {
      BuildDevice(gpu);
    }
  }

  ~Vector() {
    for (auto &kv : devs_) {
      Free(kv.second);
    }
  }

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

  const clio::cte::core::TagId &TagId() const { return tag_id_; }
  clio::run::u64 PageBytes() const { return page_bytes_; }
  clio::run::u64 NumPages() const {
    const clio::run::u64 per = page_bytes_ / sizeof(T);
    return (num_elems_ + per - 1) / per;
  }

 private:
  struct DevState {
    DeviceVector<T> view;
    char *pages_base = nullptr;     // page bytes
    char *table_base = nullptr;     // Page[] table
    char *tasks_base = nullptr;     // task slots
    ctp::ipc::AllocatorId pages_alloc;
    ctp::ipc::AllocatorId table_alloc;
    ctp::ipc::AllocatorId tasks_alloc;
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
    const clio::run::PoolQuery local = clio::run::PoolQuery::Local();
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
    }
    UploadBytes(tasks.data(), st.tasks_base, task_bytes);

    DeviceVector<T> v;
    v.tag_id_ = tag_id_;
    v.pages_ = reinterpret_cast<Page *>(st.table_base);
    v.nblocks_ = nblocks_;
    v.pages_per_block_ = pages_per_block_;
    v.page_bytes_ = page_bytes_;
    v.elems_per_page_ = page_bytes_ / sizeof(T);
    v.size_ = num_elems_;
    v.pool_id_ = clio::cte::core::kCtePoolId;
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

  static void Free(DevState &st) {
    // The backends are owned by the IpcManager registration; nothing to
    // release here beyond dropping our pointers.
    st.pages_base = nullptr;
    st.table_base = nullptr;
    st.tasks_base = nullptr;
  }

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
