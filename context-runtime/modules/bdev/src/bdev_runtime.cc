/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <chimaera/bdev/bdev_runtime.h>
#include <chimaera/comutex.h>
#include <chimaera/work_orchestrator.h>
#include <chimaera/worker.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

#include "hermes_shm/util/timer.h"

namespace chimaera::bdev {

//===========================================================================
// WorkerIOContext Implementation
//===========================================================================

bool WorkerIOContext::Init(const std::string &file_path, chi::u32 io_depth,
                           chi::u32 worker_id) {
  if (is_initialized_) {
    return true;  // Already initialized
  }

  // Create async I/O backend via factory (io_depth passed at construction)
  async_io_ = hshm::AsyncIoFactory::Get(io_depth);
  if (!async_io_) {
    HLOG(kError, "Worker {} failed to create async I/O backend", worker_id);
    return false;
  }

  // AsyncIO owns the file descriptors internally
  if (!async_io_->Open(file_path, O_RDWR | O_CREAT, 0644)) {
    HLOG(kError, "Worker {} failed to open file: {}", worker_id, file_path);
    async_io_.reset();
    return false;
  }

  is_initialized_ = true;
  HLOG(kDebug,
       "Worker {} I/O context initialized: event_fd={}",
       worker_id, async_io_->GetEventFd());
  return true;
}

void WorkerIOContext::Cleanup() {
  if (!is_initialized_) {
    return;
  }

  if (async_io_) {
    async_io_->Close();
    async_io_.reset();
  }

  is_initialized_ = false;
}

// Block size constants (in bytes) - 4KB, 16KB, 32KB, 64KB, 128KB, 1MB
static const size_t kBlockSizes[] = {
    4096,     // 4KB
    16384,    // 16KB
    32768,    // 32KB
    65536,    // 64KB
    131072,   // 128KB
    1048576   // 1MB
};

//===========================================================================
// Helper Functions
//===========================================================================

/**
 * Find the block type for a given I/O size (rounds to next largest)
 * @param io_size Requested I/O size
 * @param out_block_size Output parameter for the actual block size
 * @return Block type index, or -1 if larger than all cached sizes
 */
static int FindBlockTypeForSize(size_t io_size, size_t &out_block_size) {
  // Find the next block size that is larger than or equal to io_size
  for (int i = 0; i < static_cast<int>(BlockSizeCategory::kMaxCategories);
       ++i) {
    if (kBlockSizes[i] >= io_size) {
      out_block_size = kBlockSizes[i];
      return i;
    }
  }
  // If io_size is larger than all cached sizes, return -1
  out_block_size = io_size;  // Use exact size
  return -1;
}

//===========================================================================
// WorkerBlockMap Implementation
//===========================================================================

WorkerBlockMap::WorkerBlockMap() {
  // Initialize vector with 5 empty lists (one for each block size category)
  blocks_.resize(static_cast<size_t>(BlockSizeCategory::kMaxCategories));
}

bool WorkerBlockMap::AllocateBlock(int block_type, Block &block) {
  if (block_type < 0 ||
      block_type >= static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    return false;
  }

  // Pop from the head of the list for this block type
  if (blocks_[block_type].empty()) {
    return false;
  }

  block = blocks_[block_type].front();
  blocks_[block_type].pop_front();
  return true;
}

void WorkerBlockMap::FreeBlock(Block block) {
  int block_type = static_cast<int>(block.block_type_);
  if (block_type >= 0 &&
      block_type < static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    // Append to the block list
    blocks_[block_type].push_back(block);
  }
}

//===========================================================================
// GlobalBlockMap Implementation
//===========================================================================

GlobalBlockMap::GlobalBlockMap() {}

void GlobalBlockMap::Init(size_t num_workers) {
  // Pre-allocate vectors for specified number of workers
  worker_maps_.resize(num_workers);
  worker_locks_.resize(num_workers);
}

int GlobalBlockMap::FindBlockType(size_t io_size) {
  // Use the shared helper function to find block type
  size_t block_size;  // Not needed here, but required by the function signature
  return FindBlockTypeForSize(io_size, block_size);
}

bool GlobalBlockMap::AllocateBlock(int worker, size_t io_size, Block &block) {
  if (worker < 0 || static_cast<size_t>(worker) >= worker_maps_.size()) {
    return false;
  }

  size_t worker_idx = static_cast<size_t>(worker);

  // Find the next block size that is larger than this
  int block_type = FindBlockType(io_size);
  if (block_type == -1) {
    return false;  // No suitable cached size
  }

  // Acquire this worker's mutex using ScopedCoMutex
  {
    chi::ScopedCoMutex lock(worker_locks_[worker_idx]);
    // First attempt to allocate from this worker's map
    if (worker_maps_[worker_idx].AllocateBlock(block_type, block)) {
      return true;
    }
  }

  // If we fail, try up to 4 other workers (iterate linearly)
  size_t num_workers = worker_maps_.size();
  for (size_t i = 1; i <= 4 && i < num_workers; ++i) {
    size_t other_worker = (worker_idx + i) % num_workers;
    chi::ScopedCoMutex lock(worker_locks_[other_worker]);
    if (worker_maps_[other_worker].AllocateBlock(block_type, block)) {
      return true;
    }
  }

  return false;
}

bool GlobalBlockMap::FreeBlock(int worker, Block &block) {
  if (worker < 0 || static_cast<size_t>(worker) >= worker_maps_.size()) {
    return false;
  }

  size_t worker_idx = static_cast<size_t>(worker);

  // Free on this worker's map (with lock for thread safety)
  chi::ScopedCoMutex lock(worker_locks_[worker_idx]);
  worker_maps_[worker_idx].FreeBlock(block);
  return true;
}

//===========================================================================
// Heap Implementation
//===========================================================================

Heap::Heap() : heap_(0), total_size_(0), alignment_(4096) {}

void Heap::Init(chi::u64 total_size, chi::u32 alignment) {
  total_size_ = total_size;
  alignment_ = (alignment == 0) ? 4096 : alignment;
  heap_.store(0);
}

bool Heap::Allocate(size_t block_size, int block_type, Block &block) {
  // Align the requested block size to alignment boundary for O_DIRECT I/O
  // Formula: aligned_size = ((block_size + alignment_ - 1) / alignment_) *
  // alignment_
  chi::u32 alignment = (alignment_ == 0) ? 4096 : alignment_;

  // Align the requested size
  chi::u64 aligned_size =
      ((block_size + alignment - 1) / alignment) * alignment;
  HLOG(kDebug,
       "Allocating block: block_size = {}, alignment = {}, aligned_size = {}",
       block_size, alignment, aligned_size);

  // Atomic fetch-and-add to allocate from heap using aligned size
  chi::u64 old_heap = heap_.fetch_add(aligned_size);

  if (old_heap + aligned_size > total_size_) {
    // Out of space - rollback
    return false;
  }

  // Allocation successful - both offset and size are aligned
  block.offset_ = old_heap;
  block.size_ = aligned_size;
  block.block_type_ = static_cast<chi::u32>(block_type);
  return true;
}

chi::u64 Heap::GetRemainingSize() const {
  chi::u64 current_heap = heap_.load();
  if (current_heap >= total_size_) {
    return 0;
  }
  return total_size_ - current_heap;
}

Runtime::~Runtime() {
  // Clean up libaio (only for file-based storage)
  if (bdev_type_ == BdevType::kFile) {
    CleanupAsyncIO();
    CleanupWorkerIOContexts();
  }

  // Clean up RAM backend
  if (bdev_type_ == BdevType::kRam && ram_buffer_ != nullptr) {
    delete[] ram_buffer_;
    ram_buffer_ = nullptr;
  }

  // Note: GlobalBlockMap and Heap destructors will clean up automatically
}

bool Runtime::InitializeWorkerIOContexts() {
  // Pre-allocate vector based on actual number of workers
  chi::WorkOrchestrator *work_orchestrator = CHI_WORK_ORCHESTRATOR;
  size_t num_workers =
      work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  worker_io_contexts_.resize(num_workers);
  // Contexts are lazily initialized when first accessed
  return true;
}

void Runtime::CleanupWorkerIOContexts() {
  for (auto &ctx : worker_io_contexts_) {
    ctx.Cleanup();
  }
  worker_io_contexts_.clear();
}

WorkerIOContext *Runtime::GetWorkerIOContext(size_t worker_id) {
  // Check bounds - vector is pre-allocated in InitializeWorkerIOContexts
  if (worker_id >= worker_io_contexts_.size()) {
    HLOG(kWarning, "Worker ID {} exceeds pre-allocated size {}", worker_id,
         worker_io_contexts_.size());
    return nullptr;
  }

  WorkerIOContext *ctx = &worker_io_contexts_[worker_id];

  // Lazy initialization: initialize on first access
  if (!ctx->is_initialized_) {
    if (!ctx->Init(file_path_, io_depth_, static_cast<chi::u32>(worker_id))) {
      HLOG(kError, "Failed to initialize I/O context for worker {}", worker_id);
      return nullptr;
    }

    // Register the eventfd with the worker's EventManager for completion notification
    int event_fd = ctx->async_io_ ? ctx->async_io_->GetEventFd() : -1;
    chi::Worker *worker = CHI_CUR_WORKER;
    if (worker != nullptr && event_fd >= 0) {
      auto &em = worker->GetEventManager();
      if (em.AddEvent(event_fd) < 0) {
        HLOG(kWarning, "Failed to register eventfd with worker {} EventManager",
             worker_id);
      } else {
        HLOG(kDebug, "Registered eventfd {} with worker {} EventManager",
             event_fd, worker_id);
      }
    }
  }

  return ctx;
}

chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task,
                                chi::RunContext &ctx) {
  // Get the creation parameters
  CreateParams params = task->GetParams();

  // Get the pool name which serves as the file path for file-based operations
  std::string pool_name = task->pool_name_.str();

  HLOG(kDebug,
       "Bdev runtime received params: bdev_type={}, pool_name='{}', "
       "total_size={}, io_depth={}, alignment={}",
       static_cast<chi::u32>(params.bdev_type_), pool_name, params.total_size_,
       params.io_depth_, params.alignment_);

  // Store backend type
  bdev_type_ = params.bdev_type_;

  // Initialize storage backend based on type
  if (bdev_type_ == BdevType::kFile) {
    // Store file path for per-worker FD creation
    file_path_ = pool_name;

    // Use a temporary AsyncIO to set up the file (create/truncate)
    auto setup_io = hshm::AsyncIoFactory::Get(io_depth_);
    if (!setup_io) {
      HLOG(kError, "Failed to create setup async I/O backend");
      task->return_code_ = 1;
      co_return;
    }

    if (!setup_io->Open(pool_name, O_RDWR | O_CREAT, 0644)) {
      HLOG(kError, "Failed to open file: {}", pool_name);
      task->return_code_ = 1;
      co_return;
    }

    // Get file size
    ssize_t current_size = setup_io->GetFileSize();
    if (current_size < 0) {
      task->return_code_ = 2;
      setup_io->Close();
      co_return;
    }

    file_size_ = static_cast<chi::u64>(current_size);
    HLOG(kDebug, "File stat: file_size={}, params.total_size={}", file_size_,
         params.total_size_);

    if (params.total_size_ > 0 && params.total_size_ < file_size_) {
      file_size_ = params.total_size_;
    }

    // If file is empty, create it with default size (1GB)
    if (file_size_ == 0) {
      file_size_ = (params.total_size_ > 0) ? params.total_size_
                                            : (1ULL << 30);  // 1GB default
      HLOG(kDebug,
           "File is empty, setting file_size_ to {} and calling Truncate",
           file_size_);
      if (!setup_io->Truncate(static_cast<size_t>(file_size_))) {
        task->return_code_ = 3;
        HLOG(kError, "Failed to truncate file: {}", pool_name);
        setup_io->Close();
        co_return;
      }
      HLOG(kDebug, "Truncate succeeded, file_size_={}", file_size_);
    }
    HLOG(kDebug, "Create: Final file_size_={}, initializing allocator",
         file_size_);

    // Close setup I/O — per-worker contexts will open their own
    setup_io->Close();

    // Initialize async I/O for file backend (legacy POSIX AIO)
    InitializeAsyncIO();

    // Initialize per-worker I/O contexts for parallel file access
    if (!InitializeWorkerIOContexts()) {
      HLOG(kWarning,
           "Failed to initialize per-worker I/O contexts, "
           "falling back to single FD");
    }

  } else if (bdev_type_ == BdevType::kRam) {
    // RAM-based storage initialization
    if (params.total_size_ == 0) {
      // RAM backend requires explicit size
      task->return_code_ = 4;
      co_return;
    }

    ram_size_ = params.total_size_;
    ram_buffer_ = new (std::nothrow) char[ram_size_];
    if (ram_buffer_ == nullptr) {
      task->return_code_ = 5;
      co_return;
    }
    memset(ram_buffer_, 0, ram_size_);
    file_size_ = ram_size_;  // Use file_size_ for common allocation logic
  }

  // Initialize common parameters
  alignment_ = params.alignment_;
  io_depth_ = params.io_depth_;

  // Initialize the data allocator
  InitializeAllocator();

  // Initialize performance tracking
  start_time_ = std::chrono::high_resolution_clock::now();
  total_reads_ = 0;
  total_writes_ = 0;
  total_bytes_read_ = 0;
  total_bytes_written_ = 0;

  // Store user-provided performance characteristics
  perf_metrics_ = params.perf_metrics_;

  // Note: max_blocks_per_operation_ is already initialized in Runtime
  // constructor to 64

  // Set success result
  task->return_code_ = 0;
  (void)ctx;
  co_return;
}

chi::TaskResume Runtime::AllocateBlocks(hipc::FullPtr<AllocateBlocksTask> task,
                                        chi::RunContext &ctx) {
  HLOG(kDebug,
       "bdev::AllocateBlocks: ENTER - pool_id_=({},{}), size={}, "
       "container_id={}",
       task->pool_id_.major_, task->pool_id_.minor_, task->size_,
       container_id_);

  // Get worker ID for allocation
  int worker_id = static_cast<int>(GetWorkerID(ctx));

  chi::u64 total_size = task->size_;
  if (total_size == 0) {
    HLOG(kDebug, "bdev::AllocateBlocks: size is 0, returning empty blocks");
    task->blocks_.clear();
    task->return_code_ = 0;  // Nothing to allocate
    co_return;
  }

  // Create local vector in private memory to build up the block list
  std::vector<Block> local_blocks;

  // Divide the I/O request into blocks
  // If I/O size >= largest cached block, divide into units of that size
  // Else, just use this I/O size
  std::vector<size_t> io_divisions;

  const size_t kMaxBlock =
      kBlockSizes[static_cast<int>(BlockSizeCategory::kMaxCategories) - 1];
  if (total_size >= kMaxBlock) {
    // Divide into max-block-sized chunks
    chi::u64 remaining = total_size;
    while (remaining >= kMaxBlock) {
      io_divisions.push_back(kMaxBlock);
      remaining -= kMaxBlock;
    }
    // Add remaining bytes if any
    if (remaining > 0) {
      io_divisions.push_back(static_cast<size_t>(remaining));
    }
  } else {
    // Use the entire I/O size as a single division
    io_divisions.push_back(static_cast<size_t>(total_size));
  }

  // For each expected I/O size division, allocate a block
  for (size_t io_size : io_divisions) {
    Block block;
    bool allocated = false;

    // First attempt to allocate from the GlobalBlockMap
    if (global_block_map_.AllocateBlock(worker_id, io_size, block)) {
      allocated = true;
    } else {
      // If that fails, allocate from heap
      // Find the appropriate block type and size for this I/O size
      size_t alloc_size;
      int block_type = FindBlockTypeForSize(io_size, alloc_size);

      // If no cached size fits, use largest category
      if (block_type == -1) {
        block_type = static_cast<int>(BlockSizeCategory::kMaxCategories) - 1;
      }

      if (heap_.Allocate(alloc_size, block_type, block)) {
        allocated = true;
      }
    }

    // If allocation failed, clean up and return error
    if (!allocated) {
      // Return all allocated blocks to the GlobalBlockMap
      for (Block &allocated_block : local_blocks) {
        global_block_map_.FreeBlock(worker_id, allocated_block);
      }
      task->blocks_.clear();
      // HLOG(kError, "Out of space: {} bytes requested", total_size);
      task->return_code_ = 1;  // Out of space
      co_return;
    }

    // Check if we would exceed max_blocks limit
    if (local_blocks.size() >= max_blocks_per_operation_) {
      // Return all allocated blocks to the GlobalBlockMap
      for (Block &allocated_block : local_blocks) {
        global_block_map_.FreeBlock(worker_id, allocated_block);
      }
      task->blocks_.clear();
      HLOG(kError,
           "Operation requires {} blocks but max_blocks_per_operation is {}",
           io_divisions.size(), max_blocks_per_operation_);
      task->return_code_ = 2;  // Too many blocks required
      co_return;
    }

    // Add the allocated block to the local vector
    local_blocks.push_back(block);
  }

  // Copy the local vector to the task's shared memory vector using assignment
  // operator
  // task->blocks_ = local_blocks;
  for (size_t i = 0; i < local_blocks.size(); i++) {
    task->blocks_.push_back(local_blocks[i]);
  }

  HLOG(kDebug,
       "bdev::AllocateBlocks: SUCCESS - allocated {} blocks, "
       "task->blocks_.size()={}",
       local_blocks.size(), task->blocks_.size());

  task->return_code_ = 0;
  (void)ctx;
  co_return;
}

chi::TaskResume Runtime::FreeBlocks(hipc::FullPtr<FreeBlocksTask> task,
                                    chi::RunContext &ctx) {
  // Get worker ID for free operation
  int worker_id = static_cast<int>(GetWorkerID(ctx));

  // Free all blocks in the vector using GlobalBlockMap
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    Block block_copy = task->blocks_[i];  // Make a copy since FreeBlock takes
                                          // non-const reference
    global_block_map_.FreeBlock(worker_id, block_copy);
  }

  task->return_code_ = 0;
  (void)ctx;
  co_return;
}

chi::TaskResume Runtime::Write(hipc::FullPtr<WriteTask> task,
                               chi::RunContext &ctx) {
  switch (bdev_type_) {
    case BdevType::kFile:
      co_await WriteToFile(task, ctx);
      break;
    case BdevType::kRam:
      WriteToRam(task);
      break;
    default:
      task->return_code_ = 1;
      task->bytes_written_ = 0;
      break;
  }
  co_return;
}

chi::TaskResume Runtime::Read(hipc::FullPtr<ReadTask> task,
                              chi::RunContext &ctx) {
  switch (bdev_type_) {
    case BdevType::kFile:
      co_await ReadFromFile(task, ctx);
      break;
    case BdevType::kRam:
      ReadFromRam(task);
      break;
    default:
      task->return_code_ = 1;
      task->bytes_read_ = 0;
      break;
  }
  co_return;
}

chi::TaskResume Runtime::WriteToFile(hipc::FullPtr<WriteTask> task,
                                     chi::RunContext &ctx) {
  size_t worker_id = GetWorkerID(ctx);
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  chi::u64 total_bytes_written = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    chi::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) break;
    chi::u64 block_write_size = std::min(remaining, block.size_);

    void *block_data = data_ptr.ptr_ + data_offset;

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || !io_ctx->async_io_) {
      HLOG(kError, "WriteToFile called with invalid I/O context");
      task->return_code_ = 1;
      task->bytes_written_ = total_bytes_written;
      co_return;
    }

    hshm::IoToken token = io_ctx->async_io_->Write(
        block_data, static_cast<size_t>(block_write_size),
        static_cast<off_t>(block.offset_));
    if (token == hshm::kInvalidIoToken) {
      HLOG(kError, "Failed to submit async write: offset={}, size={}",
           block.offset_, block_write_size);
      task->return_code_ = 2;
      task->bytes_written_ = total_bytes_written;
      co_return;
    }

    hshm::IoResult result;
    while (!io_ctx->async_io_->IsComplete(token, result)) {
      co_await chi::yield();
    }

    if (result.error_code != 0) {
      HLOG(kError, "Async write failed: error_code={}", result.error_code);
      task->return_code_ = 4;
      task->bytes_written_ = total_bytes_written;
      co_return;
    }

    chi::u64 actual_bytes = std::min(
        static_cast<chi::u64>(result.bytes_transferred), block_write_size);
    total_bytes_written += actual_bytes;
    data_offset += actual_bytes;
  }

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;
  total_writes_.fetch_add(1);
  total_bytes_written_.fetch_add(task->bytes_written_);
  co_return;
}

chi::TaskResume Runtime::ReadFromFile(hipc::FullPtr<ReadTask> task,
                                      chi::RunContext &ctx) {
  size_t worker_id = GetWorkerID(ctx);
  WorkerIOContext *io_ctx = GetWorkerIOContext(worker_id);

  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  chi::u64 total_bytes_read = 0;
  chi::u64 data_offset = 0;

  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    chi::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) break;
    chi::u64 block_read_size = std::min(remaining, block.size_);

    void *block_data = data_ptr.ptr_ + data_offset;

    if (io_ctx == nullptr || !io_ctx->is_initialized_ || !io_ctx->async_io_) {
      HLOG(kError, "ReadFromFile called with invalid I/O context");
      task->return_code_ = 1;
      task->bytes_read_ = total_bytes_read;
      co_return;
    }

    hshm::IoToken token = io_ctx->async_io_->Read(
        block_data, static_cast<size_t>(block_read_size),
        static_cast<off_t>(block.offset_));
    if (token == hshm::kInvalidIoToken) {
      HLOG(kError, "Failed to submit async read: offset={}, size={}",
           block.offset_, block_read_size);
      task->return_code_ = 2;
      task->bytes_read_ = total_bytes_read;
      co_return;
    }

    hshm::IoResult result;
    while (!io_ctx->async_io_->IsComplete(token, result)) {
      co_await chi::yield();
    }

    if (result.error_code != 0) {
      HLOG(kError, "Async read failed: error_code={}", result.error_code);
      task->return_code_ = 4;
      task->bytes_read_ = total_bytes_read;
      co_return;
    }

    chi::u64 actual_bytes = std::min(
        static_cast<chi::u64>(result.bytes_transferred), block_read_size);
    total_bytes_read += actual_bytes;
    data_offset += actual_bytes;
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;
  total_reads_.fetch_add(1);
  total_bytes_read_.fetch_add(total_bytes_read);
  co_return;
}

chi::TaskResume Runtime::GetStats(hipc::FullPtr<GetStatsTask> task,
                                  chi::RunContext &ctx) {
  // Return the user-provided performance characteristics
  task->metrics_ = perf_metrics_;
  // Get remaining size from heap allocator
  chi::u64 remaining = heap_.GetRemainingSize();
  task->remaining_size_ = remaining;
  HLOG(kDebug, "GetStats: file_size_={}, remaining={}", file_size_, remaining);
  task->return_code_ = 0;
  (void)ctx;
  co_return;
}

chi::TaskResume Runtime::Destroy(hipc::FullPtr<DestroyTask> task,
                                 chi::RunContext &ctx) {
  // Worker I/O contexts (and their AsyncIO instances) are cleaned up by destructor
  // Note: GlobalBlockMap and Heap cleanup is handled by their destructors

  task->return_code_ = 0;
  (void)ctx;
  co_return;
}

void Runtime::InitializeAllocator() {
  // Initialize global block map with actual number of workers
  chi::WorkOrchestrator *work_orchestrator = CHI_WORK_ORCHESTRATOR;
  size_t num_workers =
      work_orchestrator ? work_orchestrator->GetWorkerCount() : 16;
  global_block_map_.Init(num_workers);

  // Initialize heap with total file size and alignment requirement
  heap_.Init(file_size_, alignment_);
}

size_t Runtime::GetBlockSize(int block_type) {
  if (block_type >= 0 &&
      block_type < static_cast<int>(BlockSizeCategory::kMaxCategories)) {
    return kBlockSizes[block_type];
  }
  return 0;
}

size_t Runtime::GetWorkerID(chi::RunContext &ctx) {
  // Get current worker from thread-local storage using CHI_CUR_WORKER macro
  chi::Worker *worker = CHI_CUR_WORKER;
  if (worker == nullptr) {
    return 0;  // Fallback to worker 0 if not in worker context
  }
  return worker->GetId();
}

chi::u64 Runtime::AlignSize(chi::u64 size) {
  if (alignment_ == 0) {
    alignment_ = 4096;  // Set to default if somehow it's 0
  }
  return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void Runtime::UpdatePerformanceMetrics(bool is_write, chi::u64 bytes,
                                       double duration_us) {
  // This is a simplified implementation
  // In a real implementation, you'd maintain running averages or histograms
}

void Runtime::InitializeAsyncIO() {
  // No initialization needed for POSIX AIO fallback
}

void Runtime::CleanupAsyncIO() {
  // No cleanup needed for POSIX AIO fallback
}

void Runtime::WriteToRam(hipc::FullPtr<WriteTask> task) {
  static thread_local size_t ram_write_count = 0;
  static thread_local double t_resolve_ms = 0, t_memcpy_ms = 0;
  hshm::Timer timer;

  // Convert hipc::ShmPtr<> to hipc::FullPtr<char> for data access
  timer.Resume();
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();
  timer.Pause();
  t_resolve_ms += timer.GetMsec();
  timer.Reset();

  chi::u64 total_bytes_written = 0;
  chi::u64 data_offset = 0;

  // Iterate over all blocks
  timer.Resume();
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    // Calculate how much data to write to this block
    chi::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) {
      break;  // All data has been written
    }
    chi::u64 block_write_size = std::min(remaining, block.size_);

    // Check bounds
    if (block.offset_ + block_write_size > ram_size_) {
      task->return_code_ = 1;  // Write beyond buffer bounds
      task->bytes_written_ = total_bytes_written;
      HLOG(kError,
           "Write to RAM beyond buffer bounds offset: {}, length: {}, "
           "ram_size: {}",
           block.offset_, block_write_size, ram_size_);
      return;
    }

    // Simple memory copy
    memcpy(ram_buffer_ + block.offset_, data_ptr.ptr_ + data_offset,
           block_write_size);

    // Update counters
    total_bytes_written += block_write_size;
    data_offset += block_write_size;
  }
  timer.Pause();
  t_memcpy_ms += timer.GetMsec();
  timer.Reset();

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;

  // Update performance metrics
  total_writes_.fetch_add(1);
  total_bytes_written_.fetch_add(task->bytes_written_);

  ++ram_write_count;
  if (ram_write_count % 100 == 0) {
    HLOG(kDebug, "[WriteToRam] ops={} resolve={} ms memcpy={} ms",
         ram_write_count, t_resolve_ms, t_memcpy_ms);
    t_resolve_ms = t_memcpy_ms = 0;
  }
}

void Runtime::ReadFromRam(hipc::FullPtr<ReadTask> task) {
  // Convert hipc::ShmPtr<> to hipc::FullPtr<char> for data access
  auto *ipc_mgr = CHI_IPC;
  hipc::FullPtr<char> data_ptr = ipc_mgr->ToFullPtr(task->data_).Cast<char>();

  chi::u64 total_bytes_read = 0;
  chi::u64 data_offset = 0;

  // Iterate over all blocks
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block &block = task->blocks_[i];

    // Calculate how much data to read from this block
    chi::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) {
      break;  // All data has been read
    }
    chi::u64 block_read_size = std::min(remaining, block.size_);

    // Check bounds
    if (block.offset_ + block_read_size > ram_size_) {
      task->return_code_ = 1;  // Read beyond buffer bounds
      task->bytes_read_ = total_bytes_read;
      HLOG(kError,
           "Read from RAM beyond buffer bounds offset: {}, length: {}, "
           "ram_size: {}",
           block.offset_, block_read_size, ram_size_);
      return;
    }

    // Copy data from RAM buffer to task output
    memcpy(data_ptr.ptr_ + data_offset, ram_buffer_ + block.offset_,
           block_read_size);

    // Update counters
    total_bytes_read += block_read_size;
    data_offset += block_read_size;
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;

  // Update performance metrics
  total_reads_.fetch_add(1);
  total_bytes_read_.fetch_add(total_bytes_read);
}

// VIRTUAL METHOD IMPLEMENTATIONS (now in autogen/bdev_lib_exec.cc)

chi::u64 Runtime::GetWorkRemaining() const { return 0; }

}  // namespace chimaera::bdev

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(chimaera::bdev::Runtime)