/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/mem_bdev_transport.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>
#include <clio_ctp/util/gpu_api.h>
#include <cstdlib>

namespace clio::run::bdev {

bool MemBdevTransport::Init(const CreateParams& params,
                            const std::string& /*pool_name*/, Runtime* runtime) {
  ram_capacity_ = (params.total_size_ == 0) ? DefaultRamCapacityBytes() : params.total_size_;
  bdev_type_ = params.bdev_type_;
  force_sync_gpu_ = (std::getenv("CLIO_BDEV_FORCE_SYNC") != nullptr);

  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  allocator_.Init(num_workers, ram_capacity_, params.alignment_);

  return true;
}

void MemBdevTransport::Destroy() {
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
  return allocator_.AllocateBlocks(size, worker_id, blocks);
}

void MemBdevTransport::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
  allocator_.FreeBlocks(worker_id, blocks);
}

char* MemBdevTransport::EnsureRamPage(size_t page_idx) {
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
    }
  }
  return page.data;
}

char* MemBdevTransport::GetRamPage(size_t page_idx) const {
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
