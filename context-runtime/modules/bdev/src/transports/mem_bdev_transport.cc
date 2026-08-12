/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/mem_bdev_transport.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>
#include <clio_runtime/bdev/bdev_runtime.h>
#include <clio_ctp/util/config_parse.h>
#include <clio_ctp/util/gpu_api.h>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace clio::run::bdev {

bool MemBdevTransport::Init(const CreateParams& params,
                            const std::string& /*pool_name*/, Runtime* runtime) {
  ram_capacity_ = (params.total_size_ == 0) ? DefaultRamCapacityBytes() : params.total_size_;
  bdev_type_ = params.bdev_type_;
  populate_unit_ = params.populate_unit_;
  force_sync_gpu_ = (std::getenv("CLIO_BDEV_FORCE_SYNC") != nullptr);

  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() :
                                           16;
  // RAM is byte-addressable — the alignment quantum here is only allocation
  // granularity, not a device requirement. At the 4096 default a ~1KB blob
  // pins a whole 4KB block (measured 3.6x capacity amplification, issue #862:
  // a 256MB tier "filled" at 65,536 small blobs). Drop the DEFAULT to 512 so
  // the new sub-4KB buckets (block_allocator.h) are actually reachable; an
  // explicit non-default config value is honored unchanged.
  clio::run::u32 align =
      (params.alignment_ == 0 || params.alignment_ == 4096) ? 512
                                                            : params.alignment_;
  allocator_.Init(num_workers, ram_capacity_, align);

  // issue #783: back a plain RAM device with shared memory so client processes
  // can read blob payloads directly.
  //
  // kPinned is deliberately excluded: its pages come from cudaMallocHost and
  // cannot be SHM-backed, so it keeps the private-heap path.
  //
  // BEST-EFFORT. If the segment cannot be created the device still works
  // exactly as before on the private heap; only the client fast path is lost.
  if (bdev_type_ == BdevType::kRam && runtime != nullptr) {
    InitShmBacking(runtime->pool_id_);
  }

  // CLIO_PREFAULT: pre-fault the RAM segment at init instead of (or ahead of)
  // the incremental allocation-time population. "0" means the WHOLE mapping;
  // any other value is a size string (ConfigParse::ParseSize, e.g. "512MB",
  // "4GB") pre-faulting that prefix, clamped to the mapping. Unset keeps the
  // default: pages are populated one populate_unit at a time as allocations
  // cross the watermark.
  //
  // This trades memory for cold-write latency EXPLICITLY: whatever is
  // pre-faulted is committed RAM immediately, so "0" on a 16GB tier commits
  // 16GB at compose. The populated watermark starts past the pre-faulted
  // prefix, so incremental population resumes seamlessly beyond it.
  if (shm_backed_ && shm_base_ != nullptr) {
    const char *prefault = std::getenv("CLIO_PREFAULT");
    if (prefault != nullptr && *prefault != '\0') {
      clio::run::u64 want;
      if (std::string(prefault) == "0") {
        want = shm_usable_;
      } else {
        want = std::min<clio::run::u64>(
            ctp::ConfigParse::ParseSize(prefault), shm_usable_);
      }
      if (want > 0) {
        auto begin = std::chrono::steady_clock::now();
        ctp::SystemInfo::BulkFault(shm_base_, static_cast<size_t>(want));
        populated_bytes_.store(want, std::memory_order_release);
        auto ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
        HLOG(kInfo,
             "RAM bdev '{}': CLIO_PREFAULT={} pre-faulted {} of {} bytes in "
             "{} ms",
             shm_name_, prefault, want, shm_usable_,
             static_cast<clio::run::u64>(ms));
      }
    }
  }

  // Resolve the page allocation policy (see AllocPolicy in bdev_tasks.h).
  //
  // A pool with total_size_ == 0 has no meaningful size to preallocate: the
  // capacity above fell back to DefaultRamCapacityBytes() (80% of system DRAM),
  // and committing that eagerly would OOM the node. So kEager degrades to lazy
  // there, with a warning — degrading is strictly safer than throwing, which
  // would break existing unsized configs for no gain.
  bool preallocate = false;
  switch (params.alloc_policy_) {
    case AllocPolicy::kEager:
      preallocate = (params.total_size_ != 0);
      if (!preallocate) {
        HLOG(kWarning,
             "bdev: alloc policy 'eager' requires an explicit capacity; this "
             "pool has none (capacity would default to {} bytes), so pages "
             "will be allocated lazily instead",
             ram_capacity_);
      }
      break;
    case AllocPolicy::kLazy:
      preallocate = false;
      break;
    case AllocPolicy::kAuto:
    default:
      preallocate = (params.total_size_ != 0);
      break;
  }

  if (preallocate) {
    PreallocateRamPages();
  }

  return true;
}

void MemBdevTransport::InitShmBacking(const clio::run::PoolId &pool_id) {
  try {
    auto pid = static_cast<clio::run::u32>(ctp::SystemInfo::GetPid());
    shm_name_ = ShmSegmentName(pid, pool_id);
    // Reserve header + full capacity in one sparse mapping. memfd reservations
    // are lazily faulted, so this costs only the bytes actually written.
    // Ask for the device capacity PLUS slack for the backend's own header.
    // The backend carves that out of the region, so data_capacity_ comes back
    // smaller than requested (measured: exactly 65536 less). Without the slack
    // the last 64 KB of the device maps to nothing, and the original code
    // indexed into it -- an out-of-bounds write that Linux tolerated and
    // Windows turned into a SIGSEGV in cte_tiered_storage_all.
    static constexpr size_t kBackendHeaderSlack = 1024 * 1024;
    size_t total = sizeof(ShmRamHeader) + static_cast<size_t>(ram_capacity_) +
                   kBackendHeaderSlack;
    ctp::ipc::MemoryBackendId bid(pid, pool_id.major_);
    if (!shm_backend_.shm_init(bid, ctp::Unit<size_t>::Bytes(total),
                               shm_name_)) {
      HLOG(kWarning,
           "[#783] RAM bdev '{}': shm_init failed ({} bytes) -- falling back "
           "to private heap, client direct reads disabled",
           shm_name_, total);
      return;
    }
    char *base = reinterpret_cast<char *>(shm_backend_.data_);
    if (base == nullptr) {
      return;
    }
    // Trust the backend's reported capacity, not the size we asked for. The
    // backend carves out its own header (kBackendHeaderSize) and a platform
    // may hand back a smaller view than requested -- on Windows the metadata
    // segment failed outright with "MapViewOfFile: Access is denied" at a
    // similar size. Indexing to the REQUESTED capacity in that case walks off
    // the end of the mapping, which is what segfaulted cte_tiered_storage_all
    // on Windows while going unnoticed on Linux.
    // Use what the backend ACTUALLY mapped, which is less than requested: it
    // carves out its own header, and a platform may return a shorter view.
    // Do NOT reject a short mapping -- an all-or-nothing check disables SHM
    // backing entirely and sends every page to the 1 GiB private-heap path,
    // which OOM-killed the bdev suite. Pages past the end fall back
    // individually via ShmPageInBounds().
    size_t usable = shm_backend_.data_capacity_;
    if (usable <= sizeof(ShmRamHeader)) {
      HLOG(kWarning,
           "[#783] RAM bdev '{}': mapping unusably small ({} bytes) -- "
           "private heap",
           shm_name_, usable);
      shm_backend_.shm_destroy();
      return;
    }
    shm_header_ = reinterpret_cast<ShmRamHeader *>(base);
    std::memset(shm_header_, 0, sizeof(ShmRamHeader));
    shm_header_->version_ = ShmRamHeader::kVersion;
    shm_header_->capacity_ = ram_capacity_;
    shm_header_->data_off_ = sizeof(ShmRamHeader);
    shm_base_ = base + sizeof(ShmRamHeader);
    shm_usable_ = usable - sizeof(ShmRamHeader);
    // ready_ LAST, so a client that sees ready_ == 1 knows the mapping behind
    // it is fully described.
    shm_header_->ready_ = 1;
    shm_backed_ = true;
    HLOG(kInfo,
         "[#783] RAM bdev '{}' is shared-memory backed ({} of {} bytes mapped)",
         shm_name_, shm_usable_, ram_capacity_);
  } catch (const std::exception &e) {
    HLOG(kWarning, "[#783] RAM bdev shm backing failed: {}", e.what());
    shm_backed_ = false;
    shm_base_ = nullptr;
    shm_header_ = nullptr;
  }
}

void MemBdevTransport::PreallocateRamPages() {
  // SHM-backed devices need no per-page private-heap allocation at all:
  // EnsureRamPage resolves every offset directly into shm_base_ and never
  // touches ram_pages_ for them (see its shm_backed_ branch). Committing and
  // zeroing private-heap pages here would just be memory and time nobody
  // reads back through -- and at kRamPageSize == 1 GiB, doing that per pool
  // for every small kRam device a test suite creates is enough to exhaust a
  // CI runner outright.
  if (shm_backed_) return;

  size_t num_pages =
      static_cast<size_t>((ram_capacity_ + kRamPageSize - 1) / kRamPageSize);
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  ram_pages_.resize(num_pages);
  for (RamPage &page : ram_pages_) {
    AllocRamPage(page);
    if (page.data == nullptr) continue;
    // Pre-fault (pageable) / zero (both). Pageable pages are otherwise faulted
    // in 4 KiB at a time by the first write, inside the I/O path; a GPU D2H into
    // never-touched pageable memory is the slowest copy the driver offers. The
    // zeroing also preserves the read semantics below: a region that was never
    // written must read back as zeros, and a preallocated page can no longer be
    // detected by a null pointer.
    memset(page.data, 0, kRamPageSize);
  }
}

void MemBdevTransport::Destroy() {
  if (shm_backed_) {
    // Mark unusable before tearing the mapping down so a client that is
    // mid-attach refuses rather than reading a dying segment.
    if (shm_header_ != nullptr) {
      shm_header_->ready_ = 0;
    }
    shm_header_ = nullptr;
    shm_base_ = nullptr;
    shm_backed_ = false;
    shm_backend_.shm_destroy();
  }
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  for (RamPage &page : ram_pages_) {
    FreeRamPage(page);
  }
  ram_pages_.clear();
}

void MemBdevTransport::FreeRamPage(RamPage &page) {
  if (page.data == nullptr) return;
#if CTP_ENABLE_GPU
  if (page.pinned) {
    ctp::GpuApi::FreeHost(page.data);
    page.data = nullptr;
    page.pinned = false;
    return;
  }
#endif
  delete[] page.data;
  page.data = nullptr;
  page.pinned = false;
}

bool MemBdevTransport::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  if (!allocator_.AllocateBlocks(size, worker_id, blocks)) {
    return false;
  }
  // Bulk-fault any not-yet-populated pages these blocks cover, so the
  // upcoming data memcpy does not take one demand fault per 4KB. Recycled and
  // below-watermark blocks skip this with a single atomic load.
  clio::run::u64 end = 0;
  for (const Block &b : blocks) {
    clio::run::u64 e = b.offset_ + b.size_;
    if (e > end) end = e;
  }
  EnsurePopulated(end);
  return true;
}

void MemBdevTransport::EnsurePopulated(clio::run::u64 end) {
  if (!shm_backed_ || shm_base_ == nullptr || populate_unit_ == 0) {
    return;
  }
  clio::run::u64 cur = populated_bytes_.load(std::memory_order_acquire);
  while (end > cur) {
    clio::run::u64 target =
        ((end + populate_unit_ - 1) / populate_unit_) * populate_unit_;
    if (target > shm_usable_) {
      target = shm_usable_;
    }
    if (target <= cur) {
      return;  // range already covered (or mapping exhausted)
    }
    // Claim [cur, target) by CAS, then populate it. Claiming FIRST is what
    // makes this lock-free-safe: population is advisory, so a thread that
    // sees the advanced watermark before our BulkFault completes simply
    // demand-faults those pages as it always did. Losers re-read `cur` and
    // either find their range covered or claim the next disjoint span, so
    // concurrent claims never overlap and never double-populate.
    if (populated_bytes_.compare_exchange_weak(cur, target,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
      ctp::SystemInfo::BulkFault(shm_base_ + cur, target - cur);
      return;
    }
  }
}

void MemBdevTransport::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
  allocator_.FreeBlocks(worker_id, blocks);
}

char* MemBdevTransport::AllocRamPage(RamPage &page) {
  if (page.data != nullptr) return page.data;
#if CTP_ENABLE_GPU
  if (bdev_type_ == BdevType::kPinned) {
    // Page-locked host memory keeps cudaMemcpyAsync/hipMemcpyAsync truly
    // asynchronous, so concurrent GPU transfers overlap instead of the
    // driver silently serializing them through a pageable staging copy.
    page.data = ctp::GpuApi::MallocHost<char>(kRamPageSize);
    if (page.data != nullptr) {
      page.pinned = true;
    }
    // MallocHost returns nullptr on a host with no usable GPU backend; fall
    // through to the pageable path below so kPinned still functions there.
  }
#endif
  if (page.data == nullptr) {
    page.data = new char[kRamPageSize];
    page.pinned = false;
  }
  return page.data;
}

char* MemBdevTransport::EnsureRamPage(size_t page_idx) {
  // SHM-backed devices need no per-page allocation at all: the whole capacity
  // is one sparse mapping, so a page is just an offset into it.
  //
  // The bounds check is NOT redundant: without it an offset past the mapping
  // returns a pointer the caller happily memcpy's into. That is a silent
  // out-of-bounds write, and it is what crashed Windows CI.
  if (shm_backed_ && shm_base_ != nullptr) {
    if (ShmPageInBounds(page_idx)) {
      return shm_base_ + page_idx * kRamPageSize;
    }
    HLOG(kWarning,
         "[#783] RAM bdev '{}': page {} is outside the {}-byte mapping -- "
         "using private heap for it",
         shm_name_, page_idx, shm_usable_);
    // Fall through to the private-heap path rather than hand back a bad
    // pointer; correctness first, the fast path is only an optimization.
  }
  // Fast path for a preallocated pool (total_size_ != 0): the page already
  // exists, so this is a lookup, not an allocation. Only an unsized pool still
  // allocates here, on the I/O path.
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  if (page_idx >= ram_pages_.size()) {
    ram_pages_.resize(page_idx + 1);
  }
  return AllocRamPage(ram_pages_[page_idx]);
}

char* MemBdevTransport::GetRamPage(size_t page_idx) const {
  if (shm_backed_ && shm_base_ != nullptr && ShmPageInBounds(page_idx)) {
    return shm_base_ + page_idx * kRamPageSize;
  }
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  if (page_idx >= ram_pages_.size()) return nullptr;
  return ram_pages_[page_idx].data;
}

bool MemBdevTransport::IsRamPageCommitted(size_t page_idx) const {
  // GetRamPage takes ram_pages_mu_ and bounds-checks page_idx, so an
  // out-of-range index reports "not committed" rather than reading past the end.
  return GetRamPage(page_idx) != nullptr;
}

clio::run::u64 MemBdevTransport::CommittedRamBytes() const {
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  clio::run::u64 committed = 0;
  for (const RamPage &page : ram_pages_) {
    // Every committed page costs exactly kRamPageSize, whatever fraction of it
    // the pool is actually using (see AllocRamPage).
    if (page.data != nullptr) {
      committed += kRamPageSize;
    }
  }
  return committed;
}

void MemBdevTransport::WriteBlocksCpu(const ctp::ipc::FullPtr<WriteTask>& task,
                                      char* data) {
  clio::run::u64 total_bytes_written = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    clio::run::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    clio::run::u64 block_write_size = std::min(remaining, block.size_);

    if (ram_capacity_ != std::numeric_limits<clio::run::u64>::max() &&
        block.offset_ + block_write_size > ram_capacity_) {
      task->return_code_ = 1;
      task->bytes_written_ = total_bytes_written;
      return;
    }

    clio::run::u64 cur_off = block.offset_;
    clio::run::u64 left = block_write_size;
    while (left > 0) {
      size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
      clio::run::u64 intra = cur_off % kRamPageSize;
      clio::run::u64 chunk = std::min<clio::run::u64>(left, kRamPageSize - intra);
      char* page = EnsureRamPage(page_idx);
      memcpy(page + intra, data + data_offset, chunk);
      cur_off += chunk;
      data_offset += chunk;
      left -= chunk;
    }

    total_bytes_written += block_write_size;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
}

int MemBdevTransport::LaunchWriteBlocksGpu(const ctp::ipc::FullPtr<WriteTask>& task,
                                           char* data, void* stream,
                                           clio::run::u64& bytes_written) {
  clio::run::u64 total_bytes_written = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    clio::run::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    clio::run::u64 block_write_size = std::min(remaining, block.size_);

    if (ram_capacity_ != std::numeric_limits<clio::run::u64>::max() &&
        block.offset_ + block_write_size > ram_capacity_) {
      bytes_written = total_bytes_written;
      return 1;
    }

    clio::run::u64 cur_off = block.offset_;
    clio::run::u64 left = block_write_size;
    while (left > 0) {
      size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
      clio::run::u64 intra = cur_off % kRamPageSize;
      clio::run::u64 chunk = std::min<clio::run::u64>(left, kRamPageSize - intra);
      char* page = EnsureRamPage(page_idx);
      // Enqueue only; the caller yields and waits on the stream afterward.
      ctp::GpuApi::MemcpyAsync(page + intra, data + data_offset, chunk, stream);
      cur_off += chunk;
      data_offset += chunk;
      left -= chunk;
    }

    total_bytes_written += block_write_size;
  }

  bytes_written = total_bytes_written;
  return 0;
}

clio::run::TaskResume MemBdevTransport::WriteBlocks(ctp::ipc::FullPtr<WriteTask> task) {
  CLIO_TASK_BODY_BEGIN

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  // A task's data pointer comes from another process; if it does not resolve
  // (unregistered/reaped segment), fail the write instead of memcpy'ing from
  // NULL — this segfaulted the whole runtime under the #807 stress test.
  if (data_ptr.ptr_ == nullptr && task->length_ > 0) {
    HLOG(kError,
         "MemBdevTransport::WriteBlocks: unresolvable data ShmPtr "
         "alloc_id=({}.{}) off={} length={} — failing with EIO",
         task->data_.alloc_id_.major_, task->data_.alloc_id_.minor_,
         task->data_.off_.load(), task->length_);
    task->return_code_ = EIO;
    task->bytes_written_ = 0;
    CLIO_CO_RETURN;
  }

  // Host source: a synchronous host->host memcpy is fastest and gains nothing
  // from a GPU stream.
  if (!ctp::IsDevicePointer(data_ptr.ptr_)) {
    WriteBlocksCpu(task, data_ptr.ptr_);
    CLIO_CO_RETURN;
  }

  // Device source: enqueue every chunk copy asynchronously on a per-task stream
  // and yield the worker while the transfers are in flight, so concurrent write
  // tasks overlap on the copy engines instead of each blocking a worker.
  void *stream = ctp::GpuApi::CreateStream();
  clio::run::u64 bytes_written = 0;
  int rc = LaunchWriteBlocksGpu(task, data_ptr.ptr_, stream, bytes_written);
  if (force_sync_gpu_) {
    // Benchmark A/B: block the worker like the old synchronous path.
    ctp::GpuApi::Synchronize(stream);
  } else {
    while (!ctp::GpuApi::StreamQuery(stream)) {
      CLIO_CO_AWAIT(clio::run::yield(10.0));
    }
  }
  ctp::GpuApi::DestroyStream(stream);

  task->return_code_ = rc;
  task->bytes_written_ = bytes_written;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void MemBdevTransport::ReadBlocksCpu(const ctp::ipc::FullPtr<ReadTask>& task,
                                     char* data) {
  clio::run::u64 total_bytes_read = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    clio::run::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    clio::run::u64 block_read_size = std::min(remaining, block.size_);

    if (ram_capacity_ != std::numeric_limits<clio::run::u64>::max() &&
        block.offset_ + block_read_size > ram_capacity_) {
      task->return_code_ = 1;
      task->bytes_read_ = total_bytes_read;
      return;
    }

    clio::run::u64 cur_off = block.offset_;
    clio::run::u64 left = block_read_size;
    while (left > 0) {
      size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
      clio::run::u64 intra = cur_off % kRamPageSize;
      clio::run::u64 chunk = std::min<clio::run::u64>(left, kRamPageSize - intra);
      char* page = GetRamPage(page_idx);
      char *dst = data + data_offset;
      if (page) {
        memcpy(dst, page + intra, chunk);
      } else {
        // Never-written region reads back as zeros.
        memset(dst, 0, chunk);
      }
      cur_off += chunk;
      data_offset += chunk;
      left -= chunk;
    }

    total_bytes_read += block_read_size;
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;
}

int MemBdevTransport::LaunchReadBlocksGpu(const ctp::ipc::FullPtr<ReadTask>& task,
                                          char* data, void* stream,
                                          clio::run::u64& bytes_read) {
  clio::run::u64 total_bytes_read = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    clio::run::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    clio::run::u64 block_read_size = std::min(remaining, block.size_);

    if (ram_capacity_ != std::numeric_limits<clio::run::u64>::max() &&
        block.offset_ + block_read_size > ram_capacity_) {
      bytes_read = total_bytes_read;
      return 1;
    }

    clio::run::u64 cur_off = block.offset_;
    clio::run::u64 left = block_read_size;
    while (left > 0) {
      size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
      clio::run::u64 intra = cur_off % kRamPageSize;
      clio::run::u64 chunk = std::min<clio::run::u64>(left, kRamPageSize - intra);
      char* page = GetRamPage(page_idx);
      char *dst = data + data_offset;
      // dst is device memory here; enqueue the copy (or a zero-fill for a
      // never-written region) on the stream without waiting.
      if (page) {
        ctp::GpuApi::MemcpyAsync(dst, page + intra, chunk, stream);
      } else {
        ctp::GpuApi::MemsetAsync(dst, 0, chunk, stream);
      }
      cur_off += chunk;
      data_offset += chunk;
      left -= chunk;
    }

    total_bytes_read += block_read_size;
  }

  bytes_read = total_bytes_read;
  return 0;
}

clio::run::TaskResume MemBdevTransport::ReadBlocks(ctp::ipc::FullPtr<ReadTask> task) {
  CLIO_TASK_BODY_BEGIN

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  // Mirror of the WriteBlocks guard: never memset/memcpy into an
  // unresolvable destination pointer.
  if (data_ptr.ptr_ == nullptr && task->length_ > 0) {
    HLOG(kError,
         "MemBdevTransport::ReadBlocks: unresolvable data ShmPtr "
         "alloc_id=({}.{}) off={} length={} — failing with EIO",
         task->data_.alloc_id_.major_, task->data_.alloc_id_.minor_,
         task->data_.off_.load(), task->length_);
    task->return_code_ = EIO;
    task->bytes_read_ = 0;
    CLIO_CO_RETURN;
  }

  // Host destination: synchronous host<->host copy.
  if (!ctp::IsDevicePointer(data_ptr.ptr_)) {
    ReadBlocksCpu(task, data_ptr.ptr_);
    CLIO_CO_RETURN;
  }

  // Device destination: enqueue async copies on a per-task stream and yield
  // while they run.
  void *stream = ctp::GpuApi::CreateStream();
  clio::run::u64 bytes_read = 0;
  int rc = LaunchReadBlocksGpu(task, data_ptr.ptr_, stream, bytes_read);
  if (force_sync_gpu_) {
    // Benchmark A/B: block the worker like the old synchronous path.
    ctp::GpuApi::Synchronize(stream);
  } else {
    while (!ctp::GpuApi::StreamQuery(stream)) {
      CLIO_CO_AWAIT(clio::run::yield(10.0));
    }
  }
  ctp::GpuApi::DestroyStream(stream);

  task->return_code_ = rc;
  task->bytes_read_ = bytes_read;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

} // namespace clio::run::bdev
