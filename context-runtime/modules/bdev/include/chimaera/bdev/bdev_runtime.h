#ifndef BDEV_RUNTIME_H_
#define BDEV_RUNTIME_H_

#include <chimaera/chimaera.h>
#include <chimaera/comutex.h>
#include "bdev_client.h"
#include "bdev_tasks.h"
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <aio.h>
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
              io_depth_(32), ram_buffer_(nullptr), ram_size_(0),
              total_reads_(0), total_writes_(0),
              total_bytes_read_(0), total_bytes_written_(0) {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  ~Runtime() override;

  /**
   * Create the container (Method::kCreate)
   * This method both creates and initializes the container
   */
  void Create(hipc::FullPtr<CreateTask> task, chi::RunContext& ctx);

  /**
   * Allocate multiple blocks (Method::kAllocateBlocks)
   */
  void AllocateBlocks(hipc::FullPtr<AllocateBlocksTask> task, chi::RunContext& ctx);

  /**
   * Free data blocks (Method::kFreeBlocks)
   */
  void FreeBlocks(hipc::FullPtr<FreeBlocksTask> task, chi::RunContext& ctx);

  /**
   * Write data to a block (Method::kWrite)
   */
  void Write(hipc::FullPtr<WriteTask> task, chi::RunContext& ctx);

  /**
   * Read data from a block (Method::kRead)
   */
  void Read(hipc::FullPtr<ReadTask> task, chi::RunContext& ctx);

  /**
   * Get performance statistics (Method::kGetStats)
   */
  void GetStats(hipc::FullPtr<GetStatsTask> task, chi::RunContext& ctx);

  /**
   * Destroy the container (Method::kDestroy)
   */
  void Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext& ctx);

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
  void Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
           chi::RunContext& rctx) override;

  /**
   * Delete/cleanup a task - using autogen dispatcher
   */
  void Del(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override;

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
   * Deserialize task parameters from network transfer (unified method)
   */
  void LoadTask(chi::u32 method, chi::LoadTaskArchive& archive,
                hipc::FullPtr<chi::Task>& task_ptr) override;

  /**
   * Create a new copy of a task (deep copy for distributed execution)
   */
  void NewCopy(chi::u32 method, const hipc::FullPtr<chi::Task>& orig_task,
               hipc::FullPtr<chi::Task>& dup_task, bool deep) override;

  /**
   * Aggregate a replica task into the origin task (for merging replica results)
   */
  void Aggregate(chi::u32 method,
                 hipc::FullPtr<chi::Task> origin_task,
                 hipc::FullPtr<chi::Task> replica_task) override;

 private:
  // Client for making calls to this ChiMod
  Client client_;

  // Storage backend configuration
  BdevType bdev_type_;                            // Backend type (file or RAM)
  
  // File-based storage (kFile)
  int file_fd_;                                    // File descriptor
  chi::u64 file_size_;                            // Total file size
  chi::u32 alignment_;                            // I/O alignment requirement
  chi::u32 io_depth_;                             // Max concurrent I/O operations
  
  // RAM-based storage (kRam)
  char* ram_buffer_;                              // RAM storage buffer
  chi::u64 ram_size_;                            // Total RAM buffer size

  // New allocator components
  GlobalBlockMap global_block_map_;              // Global block cache with per-worker locking
  Heap heap_;                                     // Heap allocator for new blocks
  static constexpr size_t kMaxWorkers = 8;       // Maximum number of workers
  
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
   * @return Worker ID (0 to kMaxWorkers-1)
   */
  size_t GetWorkerID(chi::RunContext& ctx);

  /**
   * Get block size for a given block type
   * @param block_type Block type category index
   * @return Size in bytes
   */
  static size_t GetBlockSize(int block_type);
  
  /**
   * Perform async I/O operation
   */
  chi::u32 PerformAsyncIO(bool is_write, chi::u64 offset, void* buffer, 
                          chi::u64 size, chi::u64& bytes_transferred,
                          hipc::FullPtr<chi::Task> task);
  
  /**
   * Align size to required boundary
   */
  chi::u64 AlignSize(chi::u64 size);
  
  /**
   * Backend-specific write operations
   */
  void WriteToFile(hipc::FullPtr<WriteTask> task);
  void WriteToRam(hipc::FullPtr<WriteTask> task);
  
  /**
   * Backend-specific read operations
   */
  void ReadFromFile(hipc::FullPtr<ReadTask> task);
  void ReadFromRam(hipc::FullPtr<ReadTask> task);
  
  /**
   * Update performance metrics
   */
  void UpdatePerformanceMetrics(bool is_write, chi::u64 bytes, 
                                double duration_us);
};

} // namespace chimaera::bdev

#endif // BDEV_RUNTIME_H_