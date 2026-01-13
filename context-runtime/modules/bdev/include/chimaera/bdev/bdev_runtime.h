#ifndef BDEV_RUNTIME_H_
#define BDEV_RUNTIME_H_

#include <chimaera/chimaera.h>
#include <chimaera/comutex.h>
#include "bdev_client.h"
#include "bdev_tasks.h"
#include <sys/types.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <fcntl.h>
#include <aio.h>
#include <libaio.h>
#include <vector>
#include <list>
#include <atomic>
#include <chrono>

/**
 * Runtime container for bdev ChiMod
 *
 * Provides block device operations with async I/O and data allocation management
 */

namespace chimaera::bdev {

/**
 * Per-worker I/O context for parallel file access
 * Each worker has its own file descriptor and Linux AIO context
 * for efficient parallel I/O without contention
 */
struct WorkerIOContext {
  int file_fd_;          /**< File descriptor for this worker */
  io_context_t aio_ctx_; /**< Linux AIO context for this worker */
  int event_fd_;         /**< eventfd for I/O completion notification */
  bool is_initialized_;  /**< Whether this context is initialized */

  WorkerIOContext()
      : file_fd_(-1), aio_ctx_(0), event_fd_(-1), is_initialized_(false) {}

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
 * We cache the following block sizes: 256B, 1KB, 4KB, 64KB, 128KB
 */
enum class BlockSizeCategory : chi::u32 {
  k256B = 0,
  k1KB = 1,
  k4KB = 2,
  k64KB = 3,
  k128KB = 4,
  kMaxCategories = 5
};

/**
 * Per-worker block cache
 * Maintains free lists for different block sizes without locking
 */
class WorkerBlockMap {
 public:
  WorkerBlockMap();

  /**
   * Allocate a block from the cache
   * @param block_type Block size category index
   * @param block Output block to populate
   * @return true if allocation succeeded, false if cache is empty
   */
  bool AllocateBlock(int block_type, Block& block);

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
  // Required typedef for CHI_TASK_CC macro
  using CreateParams = chimaera::bdev::CreateParams;
  
  Runtime() : bdev_type_(BdevType::kFile), file_fd_(-1), file_size_(0), alignment_(4096),
              io_depth_(32), max_blocks_per_operation_(64), ram_buffer_(nullptr), ram_size_(0),
              total_reads_(0), total_writes_(0),
              total_bytes_read_(0), total_bytes_written_(0) {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  ~Runtime() override;

  /**
   * Create the container (Method::kCreate)
   * This method both creates and initializes the container
   */
  chi::TaskResume Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx);

  /**
   * Allocate multiple blocks (Method::kAllocateBlocks)
   */
  chi::TaskResume AllocateBlocks(hipc::FullPtr<AllocateBlocksTask> task, chi::RunContext& ctx);

  /**
   * Free data blocks (Method::kFreeBlocks)
   */
  chi::TaskResume FreeBlocks(hipc::FullPtr<FreeBlocksTask> task, chi::RunContext& ctx);

  /**
   * Write data to a block (Method::kWrite)
   */
  chi::TaskResume Write(hipc::FullPtr<WriteTask> task, chi::RunContext& ctx);

  /**
   * Read data from a block (Method::kRead)
   */
  chi::TaskResume Read(hipc::FullPtr<ReadTask> task, chi::RunContext& ctx);

  /**
   * Get performance statistics (Method::kGetStats)
   */
  chi::TaskResume GetStats(hipc::FullPtr<GetStatsTask> task, chi::RunContext& ctx);

  /**
   * Destroy the container (Method::kDestroy)
   */
  chi::TaskResume Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext& ctx);

  /**
   * REQUIRED VIRTUAL METHODS FROM chi::Container
   */

  /**
   * Initialize container with pool information
   */
  void Init(const chi::PoolId &pool_id, const std::string &pool_name,
            chi::u32 container_id = 0) override;

  /**
   * Execute a method on a task - using autogen dispatcher
   */
  chi::TaskResume Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
                      chi::RunContext& rctx) override;

  /**
   * Delete/cleanup a task - using autogen dispatcher
   */
  void DelTask(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Get remaining work count for this container
   */
  chi::u64 GetWorkRemaining() const override;

  /**
   * Serialize task parameters for network transfer (unified method)
   */
  void SaveTask(chi::u32 method, chi::SaveTaskArchive& archive,
                hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Deserialize task parameters into an existing task from network transfer
   */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive,
                hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task parameters from network transfer
   */
  hipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive& archive) override;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   */
  void LocalLoadTask(chi::u32 method, chi::LocalLoadTaskArchive& archive,
                     hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   */
  hipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::LocalLoadTaskArchive& archive) override;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   */
  void LocalSaveTask(chi::u32 method, chi::LocalSaveTaskArchive& archive,
                     hipc::FullPtr<chi::Task> task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  hipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method, hipc::FullPtr<chi::Task> orig_task_ptr,
                                        bool deep) override;

  /**
   * Create a new task of the specified method type
   */
  hipc::FullPtr<chi::Task> NewTask(chi::u32 method) override;

  /**
   * Aggregate a replica task into the origin task (for merging replica results)
   */
  void Aggregate(chi::u32 method,
                 hipc::FullPtr<chi::Task> origin_task_ptr,
                 hipc::FullPtr<chi::Task> replica_task_ptr) override;

 private:
  // Client for making calls to this ChiMod
  Client client_;

  // Storage backend configuration
  BdevType bdev_type_;                            // Backend type (file or RAM)

  // File-based storage (kFile)
  std::string file_path_;                         // Path to the file (for per-worker FD creation)
  int file_fd_;                                   // Legacy single file descriptor (for fallback)
  std::vector<WorkerIOContext> worker_io_contexts_;  // Per-worker I/O contexts
  chi::u64 file_size_;                            // Total file size
  chi::u32 alignment_;                            // I/O alignment requirement
  chi::u32 io_depth_;                             // Max concurrent I/O operations
  chi::u32 max_blocks_per_operation_;             // Maximum blocks per I/O operation

  // RAM-based storage (kRam)
  char* ram_buffer_;                              // RAM storage buffer
  chi::u64 ram_size_;                            // Total RAM buffer size

  // New allocator components
  GlobalBlockMap global_block_map_;              // Global block cache with per-worker locking
  Heap heap_;                                     // Heap allocator for new blocks

  // Performance tracking
  std::atomic<chi::u64> total_reads_;
  std::atomic<chi::u64> total_writes_;
  std::atomic<chi::u64> total_bytes_read_;
  std::atomic<chi::u64> total_bytes_written_;
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
   * Perform async I/O operation using per-worker context
   * @param io_ctx Worker's I/O context
   * @param is_write true for write, false for read
   * @param offset File offset
   * @param buffer Data buffer
   * @param size Size of I/O operation
   * @param bytes_transferred Output: bytes actually transferred
   * @param task Task pointer for context
   * @return 0 on success, non-zero error code on failure
   */
  chi::u32 PerformAsyncIO(WorkerIOContext *io_ctx, bool is_write,
                          chi::u64 offset, void *buffer, chi::u64 size,
                          chi::u64 &bytes_transferred,
                          hipc::FullPtr<chi::Task> task);

  /**
   * Align size to required boundary
   */
  chi::u64 AlignSize(chi::u64 size);
  
  /**
   * Backend-specific write operations
   */
  void WriteToFile(hipc::FullPtr<WriteTask> task, chi::RunContext &ctx);
  void WriteToRam(hipc::FullPtr<WriteTask> task);

  /**
   * Backend-specific read operations
   */
  void ReadFromFile(hipc::FullPtr<ReadTask> task, chi::RunContext &ctx);
  void ReadFromRam(hipc::FullPtr<ReadTask> task);
  
  /**
   * Update performance metrics
   */
  void UpdatePerformanceMetrics(bool is_write, chi::u64 bytes, 
                                double duration_us);
};

} // namespace chimaera::bdev

#endif // BDEV_RUNTIME_H_