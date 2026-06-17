/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/fs_bdev_transport.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>

namespace clio::run::bdev {

bool WorkerIOContext::Init(const std::string &file_path, chi::u32 io_depth,
                           chi::u32 worker_id) {
  if (is_initialized_) return true;

#if defined(__linux__)
  async_io_ = ctp::AsyncIoFactory::Get(io_depth, ctp::AsyncIoBackend::kNixl);
#else
  async_io_ = ctp::AsyncIoFactory::Get(io_depth);
#endif

  if (!async_io_ || !async_io_->Init()) {
    HLOG(kError, "Failed to initialize generic AsyncIo for worker {}",
         worker_id);
    return false;
  }

  if (!async_io_->Open(file_path)) {
    HLOG(kError, "Worker {} failed to open file {}", worker_id, file_path);
    async_io_->Cleanup();
    return false;
  }

  is_initialized_ = true;
  return true;
}

void WorkerIOContext::Cleanup() {
  if (is_initialized_) {
    if (async_io_) {
      async_io_->Close();
      async_io_->Cleanup();
    }
    is_initialized_ = false;
  }
}

bool FsBdevTransport::Init(const CreateParams& params, Runtime* runtime) {
  file_path_ = params.file_path_;
  io_depth_ = params.io_depth_;

  auto setup_io = ctp::AsyncIoFactory::Get(io_depth_);
  if (!setup_io->Init()) {
    HLOG(kError, "Failed to initialize setup AsyncIo for filesystem tier");
    return false;
  }

  if (!setup_io->Open(file_path_)) {
    HLOG(kError, "Failed to open bdev file: {}", file_path_);
    return false;
  }

  chi::u64 file_size = 0;
  off_t current_size = setup_io->GetSize();
  if (current_size < 0) {
    HLOG(kError, "Failed to get file size for: {}", file_path_);
    setup_io->Close();
    return false;
  }
  file_size = static_cast<chi::u64>(current_size);

  if (params.total_size_ > 0 && params.total_size_ < file_size) {
    file_size = params.total_size_;
  }

  if (file_size == 0) {
    file_size = (params.total_size_ > 0) ? params.total_size_ : (1ULL << 30);
    if (!setup_io->Truncate(static_cast<size_t>(file_size))) {
      HLOG(kError, "Failed to truncate file");
      setup_io->Close();
      return false;
    }
  }

  setup_io->Close();

  if (!InitializeWorkerIOContexts()) {
    HLOG(kWarning, "Failed to initialize per-worker I/O contexts");
  }

  chi::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  allocator_.Init(num_workers, file_size, params.alignment_);

  return true;
}

void FsBdevTransport::Destroy() {
  CleanupWorkerIOContexts();
}

bool FsBdevTransport::AllocateBlocks(size_t size, int worker_id, std::vector<Block>& blocks) {
  return allocator_.AllocateBlocks(size, worker_id, blocks);
}

void FsBdevTransport::FreeBlocks(int worker_id, const std::vector<Block>& blocks) {
  allocator_.FreeBlocks(worker_id, blocks);
}

bool FsBdevTransport::InitializeWorkerIOContexts() {
  chi::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;

  io_contexts_.resize(num_workers);
  bool success = true;
  for (size_t i = 0; i < num_workers; ++i) {
    if (!io_contexts_[i].Init(file_path_, io_depth_, static_cast<chi::u32>(i))) {
      success = false;
    }
  }
  return success;
}

void FsBdevTransport::CleanupWorkerIOContexts() {
  for (auto &ctx : io_contexts_) {
    ctx.Cleanup();
  }
}

WorkerIOContext *FsBdevTransport::GetWorkerIOContext(size_t worker_id) {
  if (worker_id >= io_contexts_.size()) {
    return nullptr;
  }
  WorkerIOContext *ctx = &io_contexts_[worker_id];
  if (!ctx->is_initialized_) {
    if (!ctx->Init(file_path_, io_depth_, static_cast<chi::u32>(worker_id))) {
      return nullptr;
    }
  }
  return ctx;
}

chi::TaskResume FsBdevTransport::WriteBlocks(ctp::ipc::FullPtr<WriteTask> task, chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN
  chi::Worker *worker = CLIO_CUR_WORKER;
  size_t worker_id = worker ? worker->GetId() : 0;
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
    ctp::DeviceAwareMemcpy(staging.data(), data_ptr.ptr_, task->length_);
  }

  chi::u64 total_bytes_written = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    chi::u64 block_write_size = std::min(remaining, block.size_);

    void *block_data = data_on_device
                           ? static_cast<void *>(staging.data() + data_offset)
                           : static_cast<void *>(data_ptr.ptr_ + data_offset);

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || !io_ctx->async_io_) {
      HLOG(kError, "WriteToFile called with invalid I/O context");
      task->return_code_ = 1;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    ctp::IoToken token = io_ctx->async_io_->Write(
        block_data, static_cast<size_t>(block_write_size),
        static_cast<off_t>(block.offset_));
    if (token == ctp::kInvalidIoToken) {
      task->return_code_ = 2;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    ctp::IoResult result;
    while (!io_ctx->async_io_->IsComplete(token, result)) {
      CLIO_CO_AWAIT(chi::yield(10.0));
    }

    if (result.error_code != 0) {
      task->return_code_ = 4;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    chi::u64 actual_bytes = std::min(
        static_cast<chi::u64>(result.bytes_transferred), block_write_size);
    total_bytes_written += actual_bytes;
    data_offset += actual_bytes;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume FsBdevTransport::ReadBlocks(ctp::ipc::FullPtr<ReadTask> task, chi::RunContext &ctx) {
  chi::RunContext& rctx = ctx;
  CLIO_TASK_BODY_BEGIN
  chi::Worker *worker = CLIO_CUR_WORKER;
  size_t worker_id = worker ? worker->GetId() : 0;
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
  }

  chi::u64 total_bytes_read = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    chi::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    chi::u64 block_read_size = std::min(remaining, block.size_);

    void *block_data = data_on_device
                           ? static_cast<void *>(staging.data() + data_offset)
                           : static_cast<void *>(data_ptr.ptr_ + data_offset);

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || !io_ctx->async_io_) {
      task->return_code_ = 1;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    ctp::IoToken token = io_ctx->async_io_->Read(
        block_data, static_cast<size_t>(block_read_size),
        static_cast<off_t>(block.offset_));
    if (token == ctp::kInvalidIoToken) {
      task->return_code_ = 2;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    ctp::IoResult result;
    while (!io_ctx->async_io_->IsComplete(token, result)) {
      CLIO_CO_AWAIT(chi::yield(10.0));
    }

    if (result.error_code != 0) {
      task->return_code_ = 4;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    chi::u64 actual_bytes = std::min(
        static_cast<chi::u64>(result.bytes_transferred), block_read_size);
    total_bytes_read += actual_bytes;
    data_offset += actual_bytes;
  }

  if (data_on_device && total_bytes_read > 0) {
    ctp::DeviceAwareMemcpy(data_ptr.ptr_, staging.data(), total_bytes_read);
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

} // namespace clio::run::bdev
