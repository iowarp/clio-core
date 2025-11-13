#include <chimaera/bdev/bdev_runtime.h>
#include <chimaera/comutex.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cmath>
#include <cstring>
#include <thread>

namespace chimaera::bdev {

// Block size constants (in bytes) - 4KB, 16KB, 32KB, 64KB, 128KB
static const size_t kBlockSizes[] = {
    4096,   // 4KB
    16384,  // 16KB
    32768,  // 32KB
    65536,  // 64KB
    131072  // 128KB
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
  out_block_size = io_size; // Use exact size
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
  worker_maps_.resize(num_workers);
  worker_locks_.resize(num_workers);
}

int GlobalBlockMap::FindBlockType(size_t io_size) {
  // Use the shared helper function to find block type
  size_t block_size; // Not needed here, but required by the function signature
  return FindBlockTypeForSize(io_size, block_size);
}

bool GlobalBlockMap::AllocateBlock(int worker, size_t io_size, Block &block) {
  if (worker < 0 || worker >= static_cast<int>(worker_maps_.size())) {
    return false;
  }

  // Find the next block size that is larger than this
  int block_type = FindBlockType(io_size);
  if (block_type == -1) {
    return false; // No suitable cached size
  }

  // Acquire this worker's mutex using ScopedCoMutex
  {
    chi::ScopedCoMutex lock(worker_locks_[worker]);
    // First attempt to allocate from this worker's map
    if (worker_maps_[worker].AllocateBlock(block_type, block)) {
      return true;
    }
  }

  // If we fail, try up to 4 other workers (iterate linearly)
  size_t num_workers = worker_maps_.size();
  for (int i = 1; i <= 4 && i < static_cast<int>(num_workers); ++i) {
    int other_worker = (worker + i) % num_workers;
    chi::ScopedCoMutex lock(worker_locks_[other_worker]);
    if (worker_maps_[other_worker].AllocateBlock(block_type, block)) {
      return true;
    }
  }

  return false;
}

bool GlobalBlockMap::FreeBlock(int worker, Block &block) {
  if (worker < 0 || worker >= static_cast<int>(worker_maps_.size())) {
    return false;
  }

  // Free on this worker's map (with lock for thread safety)
  chi::ScopedCoMutex lock(worker_locks_[worker]);
  worker_maps_[worker].FreeBlock(block);
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
  // Formula: aligned_size = ((block_size + alignment_ - 1) / alignment_) * alignment_
  chi::u32 alignment = (alignment_ == 0) ? 4096 : alignment_;

  // Align the requested size
  chi::u64 aligned_size =
      ((block_size + alignment - 1) / alignment) * alignment;
  HILOG(kInfo,
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
  }

  // Clean up storage backend
  if (bdev_type_ == BdevType::kFile && file_fd_ >= 0) {
    close(file_fd_);
  } else if (bdev_type_ == BdevType::kRam && ram_buffer_ != nullptr) {
    free(ram_buffer_);
    ram_buffer_ = nullptr;
  }

  // Note: GlobalBlockMap and Heap destructors will clean up automatically
}

void Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext &ctx) {
  // Get the creation parameters using task's allocator
  auto alloc = task->GetCtxAllocator();
  CreateParams params = task->GetParams(alloc);

  // Get the pool name which serves as the file path for file-based operations
  std::string pool_name = task->pool_name_.str();

  HILOG(kInfo,
        "DEBUG: Bdev runtime received params: bdev_type={}, pool_name='{}', "
        "total_size={}, io_depth={}, alignment={}",
        static_cast<chi::u32>(params.bdev_type_), pool_name, params.total_size_,
        params.io_depth_, params.alignment_);

  // Store backend type
  bdev_type_ = params.bdev_type_;

  // Initialize storage backend based on type
  if (bdev_type_ == BdevType::kFile) {
    // File-based storage initialization - use pool_name as file path
    file_fd_ = open(pool_name.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
    if (file_fd_ < 0) {
      HELOG(kError, "Failed to open file: {}, fd: {}, errno: {}, strerror: {}",
            pool_name, file_fd_, errno, strerror(errno));
      task->return_code_ = 1;
      return;
    }

    // Get file size
    struct stat st;
    if (fstat(file_fd_, &st) != 0) {
      task->return_code_ = 2;
      close(file_fd_);
      file_fd_ = -1;
      return;
    }

    file_size_ = st.st_size;
    HILOG(kInfo, "File stat: st.st_size={}, params.total_size={}", file_size_, params.total_size_);

    if (params.total_size_ > 0 && params.total_size_ < file_size_) {
      file_size_ = params.total_size_;
    }

    // If file is empty, create it with default size (1GB)
    if (file_size_ == 0) {
      file_size_ = (params.total_size_ > 0) ? params.total_size_
                                            : (1ULL << 30); // 1GB default
      HILOG(kInfo, "File is empty, setting file_size_ to {} and calling ftruncate", file_size_);
      if (ftruncate(file_fd_, file_size_) != 0) {
        task->return_code_ = 3;
        HELOG(kError, "Failed to truncate file: {}, errno: {}, strerror: {}", pool_name, errno, strerror(errno));
        close(file_fd_);
        file_fd_ = -1;
        return;
      }
      HILOG(kInfo, "ftruncate succeeded, file_size_={}", file_size_);
    }
    HILOG(kInfo, "Create: Final file_size_={}, initializing allocator", file_size_);

    // Initialize async I/O for file backend
    InitializeAsyncIO();

  } else if (bdev_type_ == BdevType::kRam) {
    // RAM-based storage initialization
    if (params.total_size_ == 0) {
      // RAM backend requires explicit size
      task->return_code_ = 4;
      return;
    }

    ram_size_ = params.total_size_;
    ram_buffer_ = static_cast<char *>(malloc(ram_size_));
    if (ram_buffer_ == nullptr) {
      task->return_code_ = 5;
      return;
    }

    // Initialize RAM buffer to zero
    file_size_ = ram_size_; // Use file_size_ for common allocation logic
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

  // Set success result
  task->return_code_ = 0;
}

void Runtime::AllocateBlocks(hipc::FullPtr<AllocateBlocksTask> task,
                             chi::RunContext &ctx) {
  // Get worker ID for allocation
  int worker_id = static_cast<int>(GetWorkerID(ctx));

  chi::u64 total_size = task->size_;
  if (total_size == 0) {
    task->blocks_.clear();
    task->return_code_ = 0; // Nothing to allocate
    return;
  }

  // Create local vector in private memory to build up the block list
  std::vector<Block> local_blocks;

  // Divide the I/O request into blocks
  // If I/O size >= 128KB, then divide into units of 128KB
  // Else, just use this I/O size
  std::vector<size_t> io_divisions;

  const size_t k128KB =
      kBlockSizes[static_cast<int>(BlockSizeCategory::k128KB)];
  if (total_size >= k128KB) {
    // Divide into 128KB chunks
    chi::u64 remaining = total_size;
    while (remaining >= k128KB) {
      io_divisions.push_back(k128KB);
      remaining -= k128KB;
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
        block_type = static_cast<int>(BlockSizeCategory::k128KB);
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
      HELOG(kError, "Out of space: {} bytes requested", total_size);
      task->return_code_ = 1; // Out of space
      return;
    }

    // Add the allocated block to the local vector
    local_blocks.push_back(block);
  }

  // Copy the local vector to the task's shared memory vector using assignment
  // operator
  // task->blocks_ = local_blocks;
  for (size_t i = 0; i < local_blocks.size(); i++) {
    task->blocks_.emplace_back(local_blocks[i]);
  }

  task->return_code_ = 0;
}

void Runtime::FreeBlocks(hipc::FullPtr<FreeBlocksTask> task,
                         chi::RunContext &ctx) {
  // Get worker ID for free operation
  int worker_id = static_cast<int>(GetWorkerID(ctx));

  // Free all blocks in the vector using GlobalBlockMap
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    Block block_copy = task->blocks_[i]; // Make a copy since FreeBlock takes
                                         // non-const reference
    global_block_map_.FreeBlock(worker_id, block_copy);
  }

  task->return_code_ = 0;
}

void Runtime::Write(hipc::FullPtr<WriteTask> task, chi::RunContext &ctx) {
  // Set I/O size in task stat for routing decisions
  task->stat_.io_size_ = task->length_;

  switch (bdev_type_) {
  case BdevType::kFile:
    WriteToFile(task);
    break;
  case BdevType::kRam:
    WriteToRam(task);
    break;
  default:
    task->return_code_ = 1; // Unknown backend type
    task->bytes_written_ = 0;
    break;
  }
}

void Runtime::Read(hipc::FullPtr<ReadTask> task, chi::RunContext &ctx) {
  // Set I/O size in task stat for routing decisions
  task->stat_.io_size_ = task->length_;

  switch (bdev_type_) {
  case BdevType::kFile:
    ReadFromFile(task);
    break;
  case BdevType::kRam:
    ReadFromRam(task);
    break;
  default:
    task->return_code_ = 1; // Unknown backend type
    task->bytes_read_ = 0;
    break;
  }
}

void Runtime::GetStats(hipc::FullPtr<GetStatsTask> task, chi::RunContext &ctx) {
  // Return the user-provided performance characteristics
  task->metrics_ = perf_metrics_;
  // Get remaining size from heap allocator
  chi::u64 remaining = heap_.GetRemainingSize();
  task->remaining_size_ = remaining;
  HILOG(kInfo, "GetStats: file_size_={}, remaining={}", file_size_, remaining);
  task->return_code_ = 0;
}

void Runtime::Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &ctx) {
  // Close file descriptor if open
  if (file_fd_ >= 0) {
    close(file_fd_);
    file_fd_ = -1;
  }

  // Note: GlobalBlockMap and Heap cleanup is handled by their destructors

  task->return_code_ = 0;
}

void Runtime::InitializeAllocator() {
  // Initialize global block map with number of workers
  global_block_map_.Init(kMaxWorkers);

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
  // Get current worker from thread-local storage
  auto *worker =
      HSHM_THREAD_MODEL->GetTls<chi::Worker>(chi::chi_cur_worker_key_);
  if (worker == nullptr) {
    return 0; // Fallback to worker 0 if not in worker context
  }
  chi::u32 worker_id = worker->GetId();
  return worker_id % kMaxWorkers;
}

chi::u64 Runtime::AlignSize(chi::u64 size) {
  if (alignment_ == 0) {
    alignment_ = 4096; // Set to default if somehow it's 0
  }
  return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void Runtime::UpdatePerformanceMetrics(bool is_write, chi::u64 bytes,
                                       double duration_us) {
  // This is a simplified implementation
  // In a real implementation, you'd maintain running averages or histograms
}

void Runtime::InitializeAsyncIO() {
  // No initialization needed - will create aiocb on-demand
}

void Runtime::CleanupAsyncIO() {
  // No cleanup needed - aiocb created on stack
}

chi::u32 Runtime::PerformAsyncIO(bool is_write, chi::u64 offset, void *buffer,
                                 chi::u64 size, chi::u64 &bytes_transferred,
                                 hipc::FullPtr<chi::Task> task) {
  // Create aiocb on-demand
  struct aiocb aiocb_storage;
  struct aiocb *aiocb = &aiocb_storage;

  // Initialize the AIO control block
  memset(aiocb, 0, sizeof(struct aiocb));
  aiocb->aio_fildes = file_fd_;
  aiocb->aio_buf = buffer;
  aiocb->aio_nbytes = size;
  aiocb->aio_offset = offset;
  aiocb->aio_lio_opcode = is_write ? LIO_WRITE : LIO_READ;

  // Submit the I/O operation
  int result;
  if (is_write) {
    result = aio_write(aiocb);
  } else {
    result = aio_read(aiocb);
  }

  if (result != 0) {
    return 2; // Failed to submit I/O
  }

  // Poll for completion
  while (true) {
    int error_code = aio_error(aiocb);
    if (error_code == 0) {
      // Operation completed successfully
      break;
    } else if (error_code != EINPROGRESS) {
      // Operation failed
      HELOG(kError, "Failed to perform async I/O: {}, errno: {}, strerror: {}", error_code, errno, strerror(errno));
      return 3;
    }
    // Operation still in progress, yield the current task
    if (!task.IsNull()) {
      task->Yield();
    } else {
      std::this_thread::yield();
    }
  }

  // Get the result
  ssize_t bytes_result = aio_return(aiocb);
  if (bytes_result < 0) {
    return 4; // I/O operation failed
  }

  bytes_transferred = bytes_result;
  return 0; // Success
}

// Backend-specific write operations
void Runtime::WriteToFile(hipc::FullPtr<WriteTask> task) {
  // Convert hipc::Pointer to hipc::FullPtr<char> for data access
  hipc::FullPtr<char> data_ptr(task->data_);

  chi::u64 total_bytes_written = 0;
  chi::u64 data_offset = 0;

  // Iterate over all blocks
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block& block = task->blocks_[i];

    // Calculate how much data to write to this block
    chi::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) {
      break; // All data has been written
    }
    chi::u64 block_write_size = std::min(remaining, block.size_);

    // Get data pointer offset for this block
    void* block_data = data_ptr.ptr_ + data_offset;

    // Align buffer for direct I/O
    chi::u64 aligned_size = AlignSize(block_write_size);

    // Check if the buffer is already aligned
    bool is_aligned =
        (reinterpret_cast<uintptr_t>(block_data) % alignment_ == 0) &&
        (block_write_size == aligned_size);

    void* buffer_to_use;
    void* aligned_buffer = nullptr;
    bool needs_free = false;

    if (is_aligned) {
      // Buffer is already aligned, use it directly
      buffer_to_use = block_data;
    } else {
      // Allocate aligned buffer
      if (posix_memalign(&aligned_buffer, alignment_, aligned_size) != 0) {
        task->return_code_ = 1;
        task->bytes_written_ = total_bytes_written;
        return;
      }
      needs_free = true;
      buffer_to_use = aligned_buffer;

      // Copy data to aligned buffer
      memcpy(aligned_buffer, block_data, block_write_size);
      if (aligned_size > block_write_size) {
        memset(static_cast<char*>(aligned_buffer) + block_write_size, 0,
               aligned_size - block_write_size);
      }
    }

    // Perform async write using POSIX AIO
    chi::u64 bytes_written;
    chi::u32 result = PerformAsyncIO(true, block.offset_, buffer_to_use,
                                      aligned_size, bytes_written,
                                      task.Cast<chi::Task>());

    if (needs_free) {
      free(aligned_buffer);
    }

    if (result != 0) {
      task->return_code_ = result;
      task->bytes_written_ = total_bytes_written;
      return;
    }

    // Update counters
    chi::u64 actual_bytes = std::min(bytes_written, block_write_size);
    total_bytes_written += actual_bytes;
    data_offset += actual_bytes;
  }

  // Update task results
  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;

  // Update performance metrics
  total_writes_.fetch_add(1);
  total_bytes_written_.fetch_add(task->bytes_written_);
}

void Runtime::WriteToRam(hipc::FullPtr<WriteTask> task) {
  // Convert hipc::Pointer to hipc::FullPtr<char> for data access
  hipc::FullPtr<char> data_ptr(task->data_);

  chi::u64 total_bytes_written = 0;
  chi::u64 data_offset = 0;

  // Iterate over all blocks
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block& block = task->blocks_[i];

    // Calculate how much data to write to this block
    chi::u64 remaining = task->length_ - total_bytes_written;
    if (remaining == 0) {
      break; // All data has been written
    }
    chi::u64 block_write_size = std::min(remaining, block.size_);

    // Check bounds
    if (block.offset_ + block_write_size > ram_size_) {
      task->return_code_ = 1; // Write beyond buffer bounds
      task->bytes_written_ = total_bytes_written;
      HELOG(kError,
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

  task->return_code_ = 0;
  task->bytes_written_ = total_bytes_written;

  // Update performance metrics
  total_writes_.fetch_add(1);
  total_bytes_written_.fetch_add(task->bytes_written_);
}

// Backend-specific read operations
void Runtime::ReadFromFile(hipc::FullPtr<ReadTask> task) {
  // Convert hipc::Pointer to hipc::FullPtr<char> for data access
  hipc::FullPtr<char> data_ptr(task->data_);

  chi::u64 total_bytes_read = 0;
  chi::u64 data_offset = 0;

  // Iterate over all blocks
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block& block = task->blocks_[i];

    // Calculate how much data to read from this block
    chi::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) {
      break; // All data has been read
    }
    chi::u64 block_read_size = std::min(remaining, block.size_);

    // Get data pointer offset for this block
    void* block_data = data_ptr.ptr_ + data_offset;

    // Align buffer for direct I/O
    chi::u64 aligned_size = AlignSize(block_read_size);

    // Check if the buffer is already aligned
    bool is_aligned =
        (reinterpret_cast<uintptr_t>(block_data) % alignment_ == 0) &&
        (block_read_size >= aligned_size);

    void* buffer_to_use;
    void* aligned_buffer = nullptr;
    bool needs_free = false;

    if (is_aligned) {
      // Buffer is already aligned, use it directly
      buffer_to_use = block_data;
    } else {
      // Allocate aligned buffer
      if (posix_memalign(&aligned_buffer, alignment_, aligned_size) != 0) {
        task->return_code_ = 1;
        task->bytes_read_ = total_bytes_read;
        return;
      }
      needs_free = true;
      buffer_to_use = aligned_buffer;
    }

    // Perform async read using POSIX AIO
    chi::u64 bytes_read;
    chi::u32 result = PerformAsyncIO(false, block.offset_, buffer_to_use,
                                      aligned_size, bytes_read,
                                      task.Cast<chi::Task>());

    if (result != 0) {
      task->return_code_ = result;
      task->bytes_read_ = total_bytes_read;
      if (needs_free) {
        free(aligned_buffer);
      }
      return;
    }

    // Copy data to task output if we used an aligned buffer
    chi::u64 actual_bytes = std::min(bytes_read, block_read_size);

    if (needs_free) {
      memcpy(block_data, aligned_buffer, actual_bytes);
      free(aligned_buffer);
    }

    // Update counters
    total_bytes_read += actual_bytes;
    data_offset += actual_bytes;
  }

  task->return_code_ = 0;
  task->bytes_read_ = total_bytes_read;

  // Update performance metrics
  total_reads_.fetch_add(1);
  total_bytes_read_.fetch_add(total_bytes_read);
}

void Runtime::ReadFromRam(hipc::FullPtr<ReadTask> task) {
  // Convert hipc::Pointer to hipc::FullPtr<char> for data access
  hipc::FullPtr<char> data_ptr(task->data_);

  chi::u64 total_bytes_read = 0;
  chi::u64 data_offset = 0;

  // Iterate over all blocks
  for (size_t i = 0; i < task->blocks_.size(); ++i) {
    const Block& block = task->blocks_[i];

    // Calculate how much data to read from this block
    chi::u64 remaining = task->length_ - total_bytes_read;
    if (remaining == 0) {
      break; // All data has been read
    }
    chi::u64 block_read_size = std::min(remaining, block.size_);

    // Check bounds
    if (block.offset_ + block_read_size > ram_size_) {
      task->return_code_ = 1; // Read beyond buffer bounds
      task->bytes_read_ = total_bytes_read;
      HELOG(kError,
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

} // namespace chimaera::bdev

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(chimaera::bdev::Runtime)