/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <x86intrin.h>
extern "C" void clio_evlat_add(int which, unsigned long long cycles);
#include <clio_runtime/bdev/transports/mem_bdev_transport.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>
#include <clio_runtime/bdev/bdev_runtime.h>
#include <clio_ctp/util/config_parse.h>
#include <clio_ctp/util/gpu_api.h>
#include <chrono>
#include <cstdlib>

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

  // kHbm is DEVICE memory. Until this existed it was silently the same host
  // heap as kRam, so a "GPU tier" cost exactly what the tier below it cost.
  if (bdev_type_ == BdevType::kHbm) {
    InitDeviceBacking();
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

  return true;
}

void MemBdevTransport::InitDeviceBacking() {
#if CTP_ENABLE_GPU
  if (ram_capacity_ == 0) {
    return;
  }
  device_base_ = ctp::GpuApi::Malloc<char>(static_cast<size_t>(ram_capacity_));
  if (device_base_ == nullptr) {
    HLOG(kWarning,
         "HBM bdev: cudaMalloc of {} bytes failed -- falling back to host "
         "memory. This tier is NOT device-resident; treat any tiering "
         "measurement taken against it as host-to-host.",
         ram_capacity_);
    return;
  }
  device_usable_ = static_cast<size_t>(ram_capacity_);
  device_backed_ = true;
  // PROVEN deadlock (see gpu-vector kHbm hang, 2026-08-10): intra-device
  // cudaMemcpyAsync needs SM execution, and a resident faulting kernel
  // occupying every SM starves it forever while spinning for that very
  // copy. Bounce device-tier reads through this pinned slab instead:
  // D2H + H2D are copy-engine DMA, immune to SM occupancy. Allocated HERE
  // (init, no kernel resident) because cudaMallocHost on the fault path is
  // its own deadlock.
  bounce_ = ctp::GpuApi::MallocHost<char>(kBounceBytes * kBounceSlabs);
  HLOG(kInfo, "HBM bdev: {} bytes of DEVICE memory allocated", ram_capacity_);
#else
  HLOG(kWarning,
       "HBM bdev requested in a build with no GPU support -- this tier is "
       "host memory, not device memory");
#endif
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
#if CTP_ENABLE_GPU
  if (device_backed_ && device_base_ != nullptr) {
    ctp::GpuApi::Free(device_base_);
  }
#endif
  device_base_ = nullptr;
  device_usable_ = 0;
  device_backed_ = false;

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

char* MemBdevTransport::EnsureRamPage(size_t page_idx) {
  // Device-backed (kHbm): one cudaMalloc covers the capacity, so a page is an
  // offset into it -- nothing to allocate, and nothing on the host heap.
  if (device_backed_ && device_base_ != nullptr &&
      DevicePageInBounds(page_idx)) {
    return device_base_ + page_idx * kRamPageSize;
  }
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
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  if (page_idx >= ram_pages_.size()) {
    ram_pages_.resize(page_idx + 1);
  }
  RamPage &page = ram_pages_[page_idx];
  if (page.data == nullptr) {
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
      if (bdev_type_ == BdevType::kPinned) {
        // NOT silent: a pageable page on a kPinned tier turns every async
        // DMA out of it into a driver-staged synchronous copy (~2x per-read
        // cost). One run with a few of these looks like an unexplained
        // 50->75 ms/tok mode flip — log it so the flip is attributable.
        HLOG(kError,
             "kPinned bdev '{}': page-locked alloc of page {} ({} MB) FAILED"
             " — falling back to pageable memory; reads of this page will be"
             " staged-synchronous",
             shm_name_, page_idx, kRamPageSize >> 20);
      }
    }
  }
  return page.data;
}

char* MemBdevTransport::GetRamPage(size_t page_idx) const {
  if (device_backed_ && device_base_ != nullptr &&
      DevicePageInBounds(page_idx)) {
    return device_base_ + page_idx * kRamPageSize;
  }
  if (shm_backed_ && shm_base_ != nullptr && ShmPageInBounds(page_idx)) {
    return shm_base_ + page_idx * kRamPageSize;
  }
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  if (page_idx >= ram_pages_.size()) return nullptr;
  return ram_pages_[page_idx].data;
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

  // Host source AND host pages: a synchronous host->host memcpy is fastest and
  // gains nothing from a GPU stream.
  //
  // device_backed_ has to be part of this test, not just the source pointer.
  // On a kHbm pool the PAGE is device memory, so plain memcpy into it is
  // invalid no matter where the source lives -- the host->host path would
  // fault or silently corrupt. Routing on the source alone was safe only
  // while kHbm quietly used host memory.
  if (!device_backed_ && !ctp::IsDeviceAccessible(data_ptr.ptr_)) {
    WriteBlocksCpu(task, data_ptr.ptr_);
    CLIO_CO_RETURN;
  }

  // Device source: enqueue every chunk copy asynchronously on a per-task stream
  // and yield the worker while the transfers are in flight, so concurrent write
  // tasks overlap on the copy engines instead of each blocking a worker.
  // Borrowed from the pre-created pool -- see GpuApi::BorrowStream and the
  // matching comment in ReadBlocks below.
  void *stream = ctp::GpuApi::BorrowStream();
  while (stream == nullptr) {
    CLIO_CO_AWAIT(clio::run::yield(10.0));
    stream = ctp::GpuApi::BorrowStream();
  }
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
  ctp::GpuApi::ReturnStream(stream);

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
        if (device_backed_ && bounce_ != nullptr) {
          // Device-tier source: NEVER D2D (SM-starved deadlock, see Init).
          // Serialize through the pinned bounce slab under a plain mutex —
          // synchronous, no coroutine yield inside the hold, so fibers on
          // this worker cannot interleave a second claim of the slab.
          const unsigned si =
              bounce_rr_.fetch_add(1, std::memory_order_relaxed) %
              kBounceSlabs;
          char *slab = bounce_ + (size_t) si * kBounceBytes;
          std::lock_guard<std::mutex> bl(bounce_mu_[si]);
          clio::run::u64 done = 0;
          while (done < chunk) {
            const clio::run::u64 c2 =
                std::min<clio::run::u64>(chunk - done, kBounceBytes);
            ctp::GpuApi::MemcpyAsync(slab, page + intra + done, c2, stream);
            ctp::GpuApi::MemcpyAsync(dst + done, slab, c2, stream);
            ctp::GpuApi::Synchronize(stream);
            done += c2;
          }
        } else {
          ctp::GpuApi::MemcpyAsync(dst, page + intra, chunk, stream);
        }
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

  // Host destination AND host pages: synchronous host<->host copy.
  //
  // As on the write side, device_backed_ must be part of the test: on a kHbm
  // pool the SOURCE page is device memory, so memcpy out of it is invalid
  // however the destination is allocated.
  if (!device_backed_ && !ctp::IsDeviceAccessible(data_ptr.ptr_)) {
    ReadBlocksCpu(task, data_ptr.ptr_);
    CLIO_CO_RETURN;
  }

  // Device destination: enqueue async copies on a per-task stream and yield
  // while they run.
  // Borrowed from a pre-created pool, never created here: creating a stream
  // while a kernel is resident blocks until that kernel finishes, and the
  // kernels this serves spin until this very read completes. Yield until one
  // frees up rather than making a new one. See GpuApi::BorrowStream.
  void *stream = ctp::GpuApi::BorrowStream();
  while (stream == nullptr) {
    CLIO_CO_AWAIT(clio::run::yield(10.0));
    stream = ctp::GpuApi::BorrowStream();
  }
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
  ctp::GpuApi::ReturnStream(stream);

  task->return_code_ = rc;
  task->bytes_read_ = bytes_read;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

int MemBdevTransport::DirectRead(clio::run::u64 off, clio::run::u64 size,
                                 char* dst) {
  if (size == 0) return 0;
  if (ram_capacity_ != std::numeric_limits<clio::run::u64>::max() &&
      off + size > ram_capacity_) {
    return -1;
  }
  // A device-backed tier's pages are device memory; that path keeps the task
  // route (its D2D copies have their own scheduling constraints).
  if (device_backed_) return -1;

  const bool dev_dst = ctp::IsDeviceAccessible(dst);
  void* stream = nullptr;
  if (dev_dst) {
    // Borrowed, never created (creating a stream needs the context write lock
    // a resident faulting kernel holds — see ReadBlocks). If none is free
    // RIGHT NOW, fall back to the task path rather than spin on the caller's
    // fiber.
    stream = ctp::GpuApi::BorrowStream();
    if (stream == nullptr) return -1;
  }

  clio::run::u64 cur_off = off;
  clio::run::u64 left = size;
  clio::run::u64 data_offset = 0;
  while (left > 0) {
    size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
    clio::run::u64 intra = cur_off % kRamPageSize;
    clio::run::u64 chunk = std::min<clio::run::u64>(left, kRamPageSize - intra);
    char* page = GetRamPage(page_idx);
    char* d = dst + data_offset;
    if (dev_dst) {
      if (page != nullptr) {
        ctp::GpuApi::MemcpyAsync(d, page + intra, chunk, stream);
      } else {
        ctp::GpuApi::MemsetAsync(d, 0, chunk, stream);
      }
    } else {
      if (page != nullptr) {
        std::memcpy(d, page + intra, chunk);
      } else {
        std::memset(d, 0, chunk);
      }
    }
    cur_off += chunk;
    data_offset += chunk;
    left -= chunk;
  }

  if (dev_dst) {
    // Copy-engine work only — safe to block on even with a faulting kernel
    // resident (kernels block later LAUNCHES, not DMA).
    const unsigned long long ev_s0 = __rdtsc();
    ctp::GpuApi::Synchronize(stream);
    ctp::GpuApi::ReturnStream(stream);
    clio_evlat_add(3, __rdtsc() - ev_s0);  // ch3: DirectRead sync portion
  }
  return 0;
}

} // namespace clio::run::bdev
