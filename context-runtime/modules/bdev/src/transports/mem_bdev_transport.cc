/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/mem_bdev_transport.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>

namespace clio::run::bdev {

bool MemBdevTransport::Init(const CreateParams& params,
                            const std::string& /*pool_name*/, Runtime* runtime) {
  ram_capacity_ = (params.total_size_ == 0) ? DefaultRamCapacityBytes() : params.total_size_;

  chi::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  allocator_.Init(num_workers, ram_capacity_, params.alignment_);

  return true;
}

void MemBdevTransport::Destroy() {
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  ram_pages_.clear();
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
  if (!ram_pages_[page_idx]) {
    ram_pages_[page_idx].reset(new char[kRamPageSize]);
  }
  return ram_pages_[page_idx].get();
}

char* MemBdevTransport::GetRamPage(size_t page_idx) const {
  std::lock_guard<std::mutex> lock(ram_pages_mu_);
  if (page_idx >= ram_pages_.size()) return nullptr;
  return ram_pages_[page_idx].get();
}

chi::TaskResume MemBdevTransport::WriteBlocks(ctp::ipc::FullPtr<WriteTask> task, chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  chi::u64 total_bytes_written = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    chi::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    chi::u64 block_write_size = std::min(remaining, block.size_);

    if (ram_capacity_ != std::numeric_limits<chi::u64>::max() &&
        block.offset_ + block_write_size > ram_capacity_) {
      task->return_code_ = 1;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    chi::u64 cur_off = block.offset_;
    chi::u64 left = block_write_size;
    while (left > 0) {
      size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
      chi::u64 intra = cur_off % kRamPageSize;
      chi::u64 chunk = std::min<chi::u64>(left, kRamPageSize - intra);
      char* page = EnsureRamPage(page_idx);
      ctp::DeviceAwareMemcpy(page + intra, data_ptr.ptr_ + data_offset, chunk);
      cur_off += chunk;
      data_offset += chunk;
      left -= chunk;
    }

    total_bytes_written += block_write_size;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume MemBdevTransport::ReadBlocks(ctp::ipc::FullPtr<ReadTask> task, chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  chi::u64 total_bytes_read = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    chi::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    chi::u64 block_read_size = std::min(remaining, block.size_);

    if (ram_capacity_ != std::numeric_limits<chi::u64>::max() &&
        block.offset_ + block_read_size > ram_capacity_) {
      task->return_code_ = 1;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    chi::u64 cur_off = block.offset_;
    chi::u64 left = block_read_size;
    while (left > 0) {
      size_t page_idx = static_cast<size_t>(cur_off / kRamPageSize);
      chi::u64 intra = cur_off % kRamPageSize;
      chi::u64 chunk = std::min<chi::u64>(left, kRamPageSize - intra);
      char* page = GetRamPage(page_idx);
      char *dst = data_ptr.ptr_ + data_offset;
      if (page) {
        ctp::DeviceAwareMemcpy(dst, page + intra, chunk);
      } else if (ctp::IsDevicePointer(dst)) {
        static const char kZeroScratch[4096] = {};
        chi::u64 z_left = chunk;
        chi::u64 z_off = 0;
        while (z_left > 0) {
          chi::u64 z_chunk = std::min<chi::u64>(z_left, sizeof(kZeroScratch));
          ctp::DeviceAwareMemcpy(dst + z_off, kZeroScratch, z_chunk);
          z_off += z_chunk;
          z_left -= z_chunk;
        }
      } else {
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

} // namespace clio::run::bdev
