/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/fs_bdev_transport.h>
#include <clio_ctp/introspect/system_info.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/work_orchestrator.h>
#include <fcntl.h>

namespace clio::run::bdev {

namespace {

// Open `file_path` through the best AsyncIO backend that actually works at
// runtime. AsyncIoFactory's kDefault selects the preferred backend at *compile*
// time (NIXL > io_uring > libaio > POSIX), but a backend can be compiled in yet
// be unavailable at runtime: under a container's default seccomp profile
// io_uring_queue_init() fails, returning a negative errno as its return value
// while leaving the global errno untouched, so IoUringAsyncIO::Open() reports
// failure (errno=0) even though the file itself opened fine. When the preferred
// backend cannot open the file, transparently fall back to POSIX AIO, which
// works in restricted environments (containers/CI). Returns an opened AsyncIO,
// or nullptr only if even POSIX AIO cannot open the file (a genuine error).
std::unique_ptr<ctp::AsyncIO> OpenBackingFile(clio::run::u32 io_depth,
                                              const std::string &file_path) {
  auto io = ctp::AsyncIoFactory::Get(io_depth);
  if (io && io->Open(file_path, O_RDWR | O_CREAT, 0644)) {
    return io;
  }
  // Preferred backend is unusable in this environment. IoUringAsyncIO::Open()
  // closes any fds it opened before returning false, so it is safe to discard
  // it and retry with POSIX AIO.
  io = ctp::AsyncIoFactory::Get(io_depth, ctp::AsyncIoBackend::kPosixAio);
  if (io && io->Open(file_path, O_RDWR | O_CREAT, 0644)) {
    return io;
  }
  return nullptr;
}

}  // namespace

bool WorkerIOContext::Init(const std::string &file_path, clio::run::u32 io_depth,
                           clio::run::u32 worker_id) {
  if (is_initialized_) return true;

  async_io_ = OpenBackingFile(io_depth, file_path);
  if (!async_io_) {
    HLOG(kError, "Worker {} failed to open file {}", worker_id, file_path);
    return false;
  }

  is_initialized_ = true;
  return true;
}

void WorkerIOContext::Cleanup() {
  if (is_initialized_) {
    if (async_io_) {
      async_io_->Close();
      async_io_.reset();
    }
    is_initialized_ = false;
  }
}

bool FsBdevTransport::Init(const CreateParams& params,
                           const std::string& pool_name, Runtime* runtime) {
  // The pool name doubles as the backing file path.
  file_path_ = pool_name;
  io_depth_ = params.io_depth_;

  auto setup_io = OpenBackingFile(io_depth_, file_path_);
  if (!setup_io) {
    HLOG(kError, "Failed to open bdev file: {}", file_path_);
    return false;
  }

  clio::run::u64 file_size = 0;
  off_t current_size = setup_io->GetFileSize();
  if (current_size < 0) {
    HLOG(kError, "Failed to get file size for: {}", file_path_);
    setup_io->Close();
    return false;
  }
  file_size = static_cast<clio::run::u64>(current_size);

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

  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
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
  clio::run::WorkOrchestrator *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  size_t num_workers = work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;

  io_contexts_.resize(num_workers);
  bool success = true;
  for (size_t i = 0; i < num_workers; ++i) {
    if (!io_contexts_[i].Init(file_path_, io_depth_, static_cast<clio::run::u32>(i))) {
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
    if (!ctx->Init(file_path_, io_depth_, static_cast<clio::run::u32>(worker_id))) {
      return nullptr;
    }
  }
  return ctx;
}

clio::run::TaskResume FsBdevTransport::WriteBlocks(ctp::ipc::FullPtr<WriteTask> task, clio::run::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  clio::run::Worker *worker = CLIO_CUR_WORKER;
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

  clio::run::u64 total_bytes_written = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    clio::run::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    clio::run::u64 block_write_size = std::min(remaining, block.size_);

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
      CLIO_CO_AWAIT(clio::run::yield(10.0));
    }

    if (result.error_code != 0) {
      task->return_code_ = 4;
      task->bytes_written_ = total_bytes_written;
      CLIO_CO_RETURN;
    }

    clio::run::u64 actual_bytes = std::min(
        static_cast<clio::run::u64>(result.bytes_transferred), block_write_size);
    total_bytes_written += actual_bytes;
    data_offset += actual_bytes;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume FsBdevTransport::ReadBlocks(ctp::ipc::FullPtr<ReadTask> task, clio::run::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  clio::run::Worker *worker = CLIO_CUR_WORKER;
  size_t worker_id = worker ? worker->GetId() : 0;
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CLIO_IPC;
  ctp::ipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  bool data_on_device = ctp::IsDevicePointer(data_ptr.ptr_);
  std::vector<char> staging;
  if (data_on_device) {
    staging.resize(task->length_);
  }

  clio::run::u64 total_bytes_read = 0;
  clio::run::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];
    clio::run::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    clio::run::u64 block_read_size = std::min(remaining, block.size_);

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
      CLIO_CO_AWAIT(clio::run::yield(10.0));
    }

    if (result.error_code != 0) {
      task->return_code_ = 4;
      task->bytes_read_ = total_bytes_read;
      CLIO_CO_RETURN;
    }

    clio::run::u64 actual_bytes = std::min(
        static_cast<clio::run::u64>(result.bytes_transferred), block_read_size);
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
