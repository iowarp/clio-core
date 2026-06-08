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

#ifndef BDEV_RUNTIME_H_
#define BDEV_RUNTIME_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/comutex.h>
#include "bdev_client.h"
#include "bdev_tasks.h"
#include <clio_ctp/io/async_io_factory.h>
#include <vector>
#include <list>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

/**
 * Runtime container for bdev ChiMod
 *
 * Provides block device operations with async I/O and data allocation management
 */

namespace clio::run::bdev {

/**
 * Per-worker I/O context for parallel file access
 * Each worker has its own file descriptor and Linux AIO context
 * for efficient parallel I/O without contention
 */
struct WorkerIOContext {
  std::unique_ptr<ctp::AsyncIO> async_io_;  /**< Async I/O backend for this worker */
  bool is_initialized_ = false;              /**< Whether this context is initialized */

  WorkerIOContext() = default;

  WorkerIOContext(WorkerIOContext &&other) noexcept
      : async_io_(std::move(other.async_io_)),
        is_initialized_(other.is_initialized_) {
    other.is_initialized_ = false;
  }

  WorkerIOContext &operator=(WorkerIOContext &&other) noexcept {
    if (this != &other) {
      Cleanup();
      async_io_ = std::move(other.async_io_);
      is_initialized_ = other.is_initialized_;
      other.is_initialized_ = false;
    }
    return *this;
  }

  WorkerIOContext(const WorkerIOContext &) = delete;
  WorkerIOContext &operator=(const WorkerIOContext &) = delete;

  /**
   * Initialize the worker I/O context
   * @param file_path Path to the file to open
   * @param io_depth Maximum number of concurrent I/O operations
   * @param worker_id Worker ID for logging
   * @return true if initialization successful, false otherwise
   */
  bool Init(const std::string &file_path, chi::u32 io_depth, chi::u32 worker_id);

  /**
   * Cleanup and close all resources
   */
  void Cleanup();

  ~WorkerIOContext() {
    Cleanup();
  }
};


/**
 * Block size categories for data allocator
 * We cache the following block sizes: 256B, 1KB, 4KB, 64KB, 128KB, 1MB
 */
enum class BlockSizeCategory : chi::u32 {
  k256B = 0,
  k1KB = 1,
  k4KB = 2,
  k64KB = 3,
  k128KB = 4,
  k1MB = 5,
  kMaxCategories = 6
};

/**
 * Per-worker block cache
 * Maintains free lists for different block sizes without locking
 */
class WorkerBlockMap {
 public:
  WorkerBlockMap();

  /**
   * Allocate a block from the cache.
   *
   * For buckets where every freed block is exactly the nominal class size
   * the head element is always a fit, so the search is O(1). The largest
   * bucket doubles as the fallthrough sink for over-cap blocks (size >
   * kBlockSizes[max]) and therefore holds heterogeneous sizes; the
   * `min_size` filter is what keeps a caller from being handed a
   * spuriously undersized block out of that bucket.
   *
   * @param block_type Block size category index
   * @param block Output block to populate
   * @param min_size Reject any block whose size_ is smaller than this
   * @return true if allocation succeeded, false otherwise
   */
  bool AllocateBlock(int block_type, Block& block, size_t min_size = 0);

  /**
   * Free a block back to the cache
   * @param block Block to free
   */
  void FreeBlock(Block block);

 private:
  std::vector<std::list<Block>> blocks_;
};

/**
 * Global block map with per-worker caching and locking
 */
class GlobalBlockMap {
 public:
  GlobalBlockMap();

  /**
   * Initialize with number of workers
   * @param num_workers Number of worker threads
   */
  void Init(size_t num_workers);

  /**
   * Allocate a block for a given worker
   * @param worker Worker ID
   * @param io_size Requested I/O size
   * @param block Output block to populate
   * @return true if allocation succeeded, false otherwise
   */
  bool AllocateBlock(int worker, size_t io_size, Block& block);

  /**
   * Free a block for a given worker
   * @param worker Worker ID
   * @param block Block to free
   * @return true if free succeeded
   */
  bool FreeBlock(int worker, Block& block);

 private:
  std::vector<WorkerBlockMap> worker_maps_;
  std::vector<chi::CoMutex> worker_locks_;

  /**
   * Find the next block size category larger than the requested size
   * @param io_size Requested I/O size
   * @return Block type index, or -1 if no suitable size
   */
  int FindBlockType(size_t io_size);
};

/**
 * Heap allocator for new blocks
 */
class Heap {
 public:
  Heap();

  /**
   * Initialize heap with total size and alignment
   * @param total_size Total size available for allocation
   * @param alignment Alignment requirement for offsets and sizes (default 4096)
   */
  void Init(chi::u64 total_size, chi::u32 alignment = 4096);

  /**
   * Allocate a block from the heap
   * @param block_size Size of block to allocate
   * @param block_type Block type category
   * @param block Output block to populate
   * @return true if allocation succeeded, false if out of space
   */
  bool Allocate(size_t block_size, int block_type, Block& block);

  /**
   * Get remaining allocatable space
   * @return Number of bytes remaining for allocation
   */
  chi::u64 GetRemainingSize() const;

 private:
  std::atomic<chi::u64> heap_;
  chi::u64 total_size_;
  chi::u32 alignment_;
};

/**
 * Runtime container for bdev operations
 */
class Runtime : public chi::Container {
 public:
  // Required typedef for CLIO_TASK_CC macro
  using CreateParams = clio::run::bdev::CreateParams;
  
  Runtime() : bdev_type_(BdevType::kFile), file_size_(0), alignment_(4096),
              io_depth_(32),
              ram_capacity_(0),
              total_reads_(0), total_writes_(0),
              total_bytes_read_(0), total_bytes_written_(0) {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  ~Runtime() override;

  /**
   * Get live task statistics for this task instance.
   * For Read/Write, reads the task's `length_` so the scheduler routes
   * actual large I/O to I/O workers and small ops to the scheduler
   * worker, instead of bucketing every read/write at a placeholder size.
   */
  chi::TaskStat GetTaskStats(const chi::Task *task) const override;

  /**
   * Create the container (Method::kCreate)
   * This method both creates and initializes the container
   */
  chi::TaskResume Create(ctp::ipc::FullPtr<CreateTask> task, chi::RunContext& ctx);

  /**
   * Allocate multiple blocks (Method::kAllocateBlocks)
   */
  chi::TaskResume AllocateBlocks(ctp::ipc::FullPtr<AllocateBlocksTask> task, chi::RunContext& ctx);

  /**
   * Free data blocks (Method::kFreeBlocks)
   */
  chi::TaskResume FreeBlocks(ctp::ipc::FullPtr<FreeBlocksTask> task, chi::RunContext& ctx);

  /**
   * Write data to a block (Method::kWrite)
   */
  chi::TaskResume Write(ctp::ipc::FullPtr<WriteTask> task, chi::RunContext& ctx);

  /**
   * Read data from a block (Method::kRead)
   */
  chi::TaskResume Read(ctp::ipc::FullPtr<ReadTask> task, chi::RunContext& ctx);

  /**
   * Get performance statistics (Method::kGetStats)
   */
  chi::TaskResume GetStats(ctp::ipc::FullPtr<GetStatsTask> task, chi::RunContext& ctx);

  /**
   * Update GPU container with device/pinned memory pointers (Method::kUpdate)
   * No-op on the CPU side; UpdateTask is primarily handled by GpuRuntime.
   */
  chi::TaskResume Update(ctp::ipc::FullPtr<UpdateTask> task, chi::RunContext& ctx);

  /**
   * Monitor container state (Method::kMonitor)
   */
  chi::TaskResume Monitor(ctp::ipc::FullPtr<MonitorTask> task, chi::RunContext &rctx);

  /**
   * Destroy the container (Method::kDestroy)
   */
  chi::TaskResume Destroy(ctp::ipc::FullPtr<DestroyTask> task, chi::RunContext& ctx);

  /**
   * REQUIRED VIRTUAL METHODS FROM chi::Container
   */

  /**
   * Initialize container with pool information
   */
  void Init(const chi::PoolId &pool_id, const std::string &pool_name,
            chi::u32 container_id = 0) override;

  /**
   * Send UpdateTask to GPU container after GPU container is registered.
   * Called from pool_manager after RegisterGpuOrchestratorContainer().
   */
  void PostGpuContainerCreate() override;

  /**
   * Execute a method on a task - using autogen dispatcher
   */
  chi::TaskResume Run(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr,
                      chi::RunContext& rctx) override;

  /**
   * Get remaining work count for this container
   */
  chi::u64 GetWorkRemaining() const override;

  /**
   * Serialize task parameters for network transfer (unified method)
   */
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Deserialize task parameters into an existing task from network transfer
   */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive,
                ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task parameters from network transfer
   */
  ctp::ipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive& archive) override;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   */
  void LocalLoadTask(chi::u32 method, chi::DefaultLoadArchive& archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   */
  ctp::ipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::DefaultLoadArchive& archive) override;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   */
  void LocalSaveTask(chi::u32 method, chi::DefaultSaveArchive& archive,
                     ctp::ipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  ctp::ipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task_ptr,
                                        bool deep) override;

  /**
   * Create a new task of the specified method type
   */
  ctp::ipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;
  void Aggregate(chi::u32 method, ctp::ipc::FullPtr<chi::Task> orig_task,
                 const ctp::ipc::FullPtr<chi::Task>& replica_task) override;
  void DelTask(chi::u32 method, ctp::ipc::FullPtr<chi::Task> task_ptr) override;

 private:
  // Client for making calls to this ChiMod
  Client client_;

  // Storage backend configuration
  BdevType bdev_type_;                            // Backend type (file or RAM)

  // File-based storage (kFile)
  std::string file_path_;                         // Path to the file (for per-worker FD creation)
  std::vector<WorkerIOContext> worker_io_contexts_;  // Per-worker I/O contexts
  chi::u64 file_size_;                            // Total file size
  chi::u32 alignment_;                            // I/O alignment requirement
  chi::u32 io_depth_;                             // Max concurrent I/O operations

  // RAM-based storage (kRam) — vector of fixed-size pages, lazily allocated.
  // Pages are not pre-faulted: each page is allocated only when first written
  // to, and the OS faults the underlying physical pages on first touch. This
  // keeps memory usage proportional to data actually stored, not capacity.
  static constexpr chi::u64 kRamPageSize = 1ULL << 30;  // 1 GiB
  std::vector<std::unique_ptr<char[]>> ram_pages_;
  chi::u64 ram_capacity_;  // Soft cap; UINT64_MAX = unbounded
  mutable std::mutex ram_pages_mu_;

  // kHbm / kPinned removed — see WriteToRam comment above.

  // New allocator components
  GlobalBlockMap global_block_map_;              // Global block cache with per-worker locking
  Heap heap_;                                     // Heap allocator for new blocks

  // Performance tracking
  std::atomic<chi::u64> total_reads_;
  std::atomic<chi::u64> total_writes_;
  std::atomic<chi::u64> total_bytes_read_;
  std::atomic<chi::u64> total_bytes_written_;
  // True LIVE allocated bytes (incremented in AllocateBlocks regardless of
  // whether the block came from the GlobalBlockMap free list or the heap;
  // decremented in FreeBlocks). GetStats reports remaining = capacity -
  // allocated_bytes_ from this, NOT heap_'s monotonic bump high-water —
  // the bump pointer is never rolled back on free, so under concurrent
  // free-list misses it raced past the true live set and made CTE's
  // StatTargets see ~0 remaining (the 16-thread rc=12 placement bug).
  std::atomic<chi::u64> allocated_bytes_{0};
  std::chrono::high_resolution_clock::time_point start_time_;
  
  // User-provided performance characteristics
  PerfMetrics perf_metrics_;
  
  /**
   * Initialize the data allocator
   */
  void InitializeAllocator();

  /**
   * Initialize POSIX AIO control blocks
   */
  void InitializeAsyncIO();

  /**
   * Cleanup POSIX AIO control blocks
   */
  void CleanupAsyncIO();

  /**
   * Get worker ID from runtime context
   * @param ctx Runtime context containing worker information
   * @return Worker ID
   */
  size_t GetWorkerID(chi::RunContext& ctx);

  /**
   * Get block size for a given block type
   * @param block_type Block type category index
   * @return Size in bytes
   */
  static size_t GetBlockSize(int block_type);

  /**
   * Get or create the worker I/O context for the given worker
   * Lazily initializes per-worker file descriptors and AIO contexts
   * @param worker_id Worker ID
   * @return Pointer to the worker's I/O context, or nullptr if initialization fails
   */
  WorkerIOContext *GetWorkerIOContext(size_t worker_id);

  /**
   * Initialize per-worker I/O contexts
   * Called during Create to set up worker-specific file descriptors
   * @return true if initialization successful, false otherwise
   */
  bool InitializeWorkerIOContexts();

  /**
   * Cleanup all per-worker I/O contexts
   */
  void CleanupWorkerIOContexts();

  /**
   * Align size to required boundary
   */
  chi::u64 AlignSize(chi::u64 size);

  /**
   * Backend-specific file operations (coroutines that yield on I/O)
   */
  chi::TaskResume WriteToFile(ctp::ipc::FullPtr<WriteTask> task, chi::RunContext &ctx);
  chi::TaskResume ReadFromFile(ctp::ipc::FullPtr<ReadTask> task, chi::RunContext &ctx);

  /**
   * Backend-specific RAM operations (synchronous, no coroutine needed).
   * Uses ctp::DeviceAwareMemcpy so the data ShmPtr may resolve to either
   * a host or a device-USM pointer.
   */
  void WriteToRam(ctp::ipc::FullPtr<WriteTask> task);
  void ReadFromRam(ctp::ipc::FullPtr<ReadTask> task);

  // BdevType::kHbm / BdevType::kPinned tiers were removed. PutBlob /
  // GetBlob with HBM-resident ShmPtr data buffers route through kRam
  // (or kFile, with host-buffer staging) and the bdev uses the
  // device-aware memcpy + IsDevicePointer hooks installed at server
  // init by gpu/gpu2cpu_init_sycl.cc.

  // Get (allocating if needed) the page at page_idx. Allocation uses
  // default-initialized new char[] so the OS reserves virtual address space
  // without pre-faulting physical pages.
  char* EnsureRamPage(size_t page_idx);
  // Lookup without allocating; returns nullptr for never-written pages.
  char* GetRamPage(size_t page_idx) const;

  /**
   * Update performance metrics
   */
  void UpdatePerformanceMetrics(bool is_write, chi::u64 bytes,
                                double duration_us);
};

} // namespace clio::run::bdev

#endif // BDEV_RUNTIME_H_