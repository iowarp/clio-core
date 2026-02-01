#ifndef BDEV_TASKS_H_
#define BDEV_TASKS_H_

#include <chimaera/chimaera.h>
#include <chimaera/config_manager.h>
#include <yaml-cpp/yaml.h>

#include "autogen/bdev_methods.h"
// Include admin tasks for BaseCreateTask
#include <chimaera/admin/admin_tasks.h>

/**
 * Task struct definitions for bdev
 *
 * Defines tasks for block device operations with libaio and data allocation
 */

namespace chimaera::bdev {

/**
 * Block device type enumeration
 */
enum class BdevType : chi::u32 {
  kFile = 0,  // File-based block device (default)
  kRam = 1    // RAM-based block device
};

/**
 * Block structure for data allocation
 */
struct Block {
  chi::u64 offset_;      // Offset within file
  chi::u64 size_;        // Size of block
  chi::u32 block_type_;  // Block size category (0=4KB, 1=64KB, 2=256KB, 3=1MB)

  Block() : offset_(0), size_(0), block_type_(0) {}
  Block(chi::u64 offset, chi::u64 size, chi::u32 block_type)
      : offset_(offset), size_(size), block_type_(block_type) {}

  // Cereal serialization
  template <class Archive>
  void serialize(Archive &ar) {
    ar(offset_, size_, block_type_);
  }
};


/**
 * Performance metrics structure
 */
struct PerfMetrics {
  double read_bandwidth_mbps_;   // Read bandwidth in MB/s
  double write_bandwidth_mbps_;  // Write bandwidth in MB/s
  double read_latency_us_;       // Average read latency in microseconds
  double write_latency_us_;      // Average write latency in microseconds
  double iops_;                  // I/O operations per second

  PerfMetrics()
      : read_bandwidth_mbps_(0.0),
        write_bandwidth_mbps_(0.0),
        read_latency_us_(0.0),
        write_latency_us_(0.0),
        iops_(0.0) {}

  // Cereal serialization
  template <class Archive>
  void serialize(Archive &ar) {
    ar(read_bandwidth_mbps_, write_bandwidth_mbps_, read_latency_us_,
       write_latency_us_, iops_);
  }
};

/**
 * CreateParams for bdev chimod
 * Contains configuration parameters for bdev container creation
 */
struct CreateParams {
  // bdev-specific parameters
  BdevType bdev_type_;   // Block device type (file or RAM)
  chi::u64 total_size_;  // Total size for allocation (0 = file size for kFile,
                         // required for kRam)
  chi::u32 io_depth_;    // libaio queue depth (ignored for kRam)
  chi::u32 alignment_;   // I/O alignment (default 4096)

  // Performance characteristics (user-defined instead of benchmarked)
  PerfMetrics perf_metrics_;  // User-provided performance characteristics

  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "chimaera_bdev";

  // Default constructor (defaults to file-based with conservative performance
  // estimates)
  CreateParams()
      : bdev_type_(BdevType::kFile),
        total_size_(0),
        io_depth_(32),
        alignment_(4096) {
    // Set conservative default performance characteristics
    perf_metrics_.read_bandwidth_mbps_ = 100.0;  // 100 MB/s
    perf_metrics_.write_bandwidth_mbps_ = 80.0;  // 80 MB/s
    perf_metrics_.read_latency_us_ = 1000.0;     // 1ms
    perf_metrics_.write_latency_us_ = 1200.0;    // 1.2ms
    perf_metrics_.iops_ = 1000.0;                // 1000 IOPS
  }

  // Constructor with basic parameters (uses default performance)
  CreateParams(BdevType bdev_type, chi::u64 total_size = 0,
               chi::u32 io_depth = 32, chi::u32 alignment = 4096)
      : bdev_type_(bdev_type),
        total_size_(total_size),
        io_depth_(io_depth),
        alignment_(alignment) {
    // Set conservative default performance characteristics
    perf_metrics_.read_bandwidth_mbps_ = 100.0;
    perf_metrics_.write_bandwidth_mbps_ = 80.0;
    perf_metrics_.read_latency_us_ = 1000.0;
    perf_metrics_.write_latency_us_ = 1200.0;
    perf_metrics_.iops_ = 1000.0;

    // Debug: Log what parameters were received
    HLOG(kDebug,
         "DEBUG: CreateParams constructor called with: bdev_type={}, "
         "total_size={}, io_depth={}, alignment={}",
         static_cast<chi::u32>(bdev_type_), total_size_, io_depth_, alignment_);
  }

  // Constructor with optional performance metrics (as last parameter)
  CreateParams(BdevType bdev_type, chi::u64 total_size, chi::u32 io_depth,
               chi::u32 alignment, const PerfMetrics *perf_metrics = nullptr)
      : bdev_type_(bdev_type),
        total_size_(total_size),
        io_depth_(io_depth),
        alignment_(alignment) {
    // Set performance metrics (use provided metrics or defaults)
    if (perf_metrics != nullptr) {
      perf_metrics_ = *perf_metrics;
      HLOG(kDebug,
           "DEBUG: CreateParams constructor called with custom performance: "
           "bdev_type={}, total_size={}, io_depth={}, alignment={}, "
           "read_bw={}, write_bw={}",
           static_cast<chi::u32>(bdev_type_), total_size_, io_depth_,
           alignment_, perf_metrics_.read_bandwidth_mbps_,
           perf_metrics_.write_bandwidth_mbps_);
    } else {
      // Use default performance characteristics
      perf_metrics_.read_bandwidth_mbps_ = 100.0;
      perf_metrics_.write_bandwidth_mbps_ = 80.0;
      perf_metrics_.read_latency_us_ = 1000.0;
      perf_metrics_.write_latency_us_ = 1200.0;
      perf_metrics_.iops_ = 1000.0;
      HLOG(kDebug,
           "DEBUG: CreateParams constructor called with default performance: "
           "bdev_type={}, total_size={}, io_depth={}, alignment={}",
           static_cast<chi::u32>(bdev_type_), total_size_, io_depth_,
           alignment_);
    }
  }

  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive &ar) {
    ar(bdev_type_, total_size_, io_depth_, alignment_, perf_metrics_);
  }

  /**
   * Load configuration from PoolConfig (for compose mode)
   * @param pool_config Pool configuration from compose section
   */
  void LoadConfig(const chi::PoolConfig &pool_config) {
    // Parse YAML config string
    YAML::Node config = YAML::Load(pool_config.config_);

    // Load bdev type (optional, defaults to kFile)
    if (config["bdev_type"]) {
      std::string type_str = config["bdev_type"].as<std::string>();
      if (type_str == "file") {
        bdev_type_ = BdevType::kFile;
      } else if (type_str == "ram") {
        bdev_type_ = BdevType::kRam;
      }
    }

    // Load capacity/total_size (parse size strings like "2GB", "512MB")
    if (config["capacity"]) {
      std::string capacity_str = config["capacity"].as<std::string>();
      total_size_ = hshm::ConfigParse::ParseSize(capacity_str);
    }

    // Load io_depth (optional)
    if (config["io_depth"]) {
      io_depth_ = config["io_depth"].as<chi::u32>();
    }

    // Load alignment (optional)
    if (config["alignment"]) {
      alignment_ = config["alignment"].as<chi::u32>();
    }

    // Load performance metrics (optional)
    if (config["perf_metrics"]) {
      auto perf = config["perf_metrics"];
      if (perf["read_bandwidth_mbps"]) {
        perf_metrics_.read_bandwidth_mbps_ =
            perf["read_bandwidth_mbps"].as<double>();
      }
      if (perf["write_bandwidth_mbps"]) {
        perf_metrics_.write_bandwidth_mbps_ =
            perf["write_bandwidth_mbps"].as<double>();
      }
      if (perf["read_latency_us"]) {
        perf_metrics_.read_latency_us_ = perf["read_latency_us"].as<double>();
      }
      if (perf["write_latency_us"]) {
        perf_metrics_.write_latency_us_ = perf["write_latency_us"].as<double>();
      }
      if (perf["iops"]) {
        perf_metrics_.iops_ = perf["iops"].as<double>();
      }
    }
  }
};

/**
 * CreateTask - Initialize the bdev container
 * Type alias for GetOrCreatePoolTask with CreateParams (uses kGetOrCreatePool
 * method) Non-admin modules should use GetOrCreatePoolTask instead of
 * BaseCreateTask
 */
using CreateTask = chimaera::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * AllocateBlocksTask - Allocate multiple blocks with specified total size
 */
struct AllocateBlocksTask : public chi::Task {
  // Task-specific data
  IN chi::u64 size_;  // Requested total size
  OUT chi::priv::vector<Block> blocks_;  // Allocated blocks information

  /** SHM default constructor */
  AllocateBlocksTask() : chi::Task(), size_(0), blocks_(HSHM_MALLOC) {}

  /** Emplace constructor */
  explicit AllocateBlocksTask(const chi::TaskId &task_node,
                              const chi::PoolId &pool_id,
                              const chi::PoolQuery &pool_query, chi::u64 size)
      : chi::Task(task_node, pool_id, pool_query, 10), size_(size), blocks_(HSHM_MALLOC) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kAllocateBlocks;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(size_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(blocks_);
  }

  /**
   * Copy from another AllocateBlocksTask (assumes this task is already
   * constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<AllocateBlocksTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy AllocateBlocksTask-specific fields
    size_ = other->size_;
    blocks_ = other->blocks_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<AllocateBlocksTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * FreeBlocksTask - Free allocated blocks
 */
struct FreeBlocksTask : public chi::Task {
  // Task-specific data
  IN chi::priv::vector<Block> blocks_;  // Blocks to free

  /** SHM default constructor */
  FreeBlocksTask() : chi::Task(), blocks_(HSHM_MALLOC) {}

  /** Emplace constructor for multiple blocks */
  explicit FreeBlocksTask(const chi::TaskId &task_node,
                          const chi::PoolId &pool_id,
                          const chi::PoolQuery &pool_query,
                          const std::vector<Block> &blocks)
      : chi::Task(task_node, pool_id, pool_query, 10), blocks_(HSHM_MALLOC) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kFreeBlocks;
    task_flags_.Clear();
    pool_query_ = pool_query;

    // Copy blocks from std::vector to chi::priv::vector
    for (const auto &block : blocks) {
      blocks_.push_back(block);
    }
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(blocks_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // No additional output parameters
  }

  /**
   * Copy from another FreeBlocksTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<FreeBlocksTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy FreeBlocksTask-specific fields
    blocks_ = other->blocks_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<FreeBlocksTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * WriteTask - Write data to a block using libaio
 */
struct WriteTask : public chi::Task {
  // Task-specific data
  IN chi::priv::vector<Block> blocks_;  // Blocks to write to
  IN hipc::ShmPtr<> data_;              // Data to write (pointer-based)
  IN size_t length_;                    // Size of data to write
  OUT chi::u64 bytes_written_;          // Number of bytes actually written

  /** SHM default constructor */
  WriteTask() : chi::Task(), blocks_(HSHM_MALLOC), length_(0), bytes_written_(0) {}

  /** Emplace constructor */
  explicit WriteTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                     const chi::PoolQuery &pool_query,
                     const chi::priv::vector<Block> &blocks, hipc::ShmPtr<> data,
                     size_t length)
      : chi::Task(task_node, pool_id, pool_query, 10),
        blocks_(blocks),
        data_(data),
        length_(length),
        bytes_written_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kWrite;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Destructor - free buffer if TASK_DATA_OWNER is set */
  ~WriteTask() {
    if (task_flags_.Any(TASK_DATA_OWNER) && !data_.IsNull()) {
      auto *ipc_manager = CHI_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(data_.Cast<char>());
      }
    }
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(blocks_, length_);
    // Use bulk transfer for data pointer - BULK_XFER for actual data
    // transmission
    ar.bulk(data_, length_, BULK_XFER);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(bytes_written_);
  }

  /** Aggregate */
  void Aggregate(const hipc::FullPtr<WriteTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }

  /**
   * Copy from another WriteTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<WriteTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy WriteTask-specific fields
    blocks_ = other->blocks_;
    data_ = other->data_;
    length_ = other->length_;
    bytes_written_ = other->bytes_written_;
  }
};

/**
 * ReadTask - Read data from a block using libaio
 */
struct ReadTask : public chi::Task {
  // Task-specific data
  IN chi::priv::vector<Block> blocks_;  // Blocks to read from
  OUT hipc::ShmPtr<> data_;             // Read data (pointer-based)
  INOUT size_t
      length_;  // Size of data buffer (IN: buffer size, OUT: actual size)
  OUT chi::u64 bytes_read_;  // Number of bytes actually read

  /** SHM default constructor */
  ReadTask() : chi::Task(), blocks_(HSHM_MALLOC), length_(0), bytes_read_(0) {}

  /** Emplace constructor */
  explicit ReadTask(const chi::TaskId &task_node, const chi::PoolId &pool_id,
                    const chi::PoolQuery &pool_query,
                    const chi::priv::vector<Block> &blocks, hipc::ShmPtr<> data,
                    size_t length)
      : chi::Task(task_node, pool_id, pool_query, 10),
        blocks_(blocks),
        data_(data),
        length_(length),
        bytes_read_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kRead;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Destructor - free buffer if TASK_DATA_OWNER is set */
  ~ReadTask() {
    if (task_flags_.Any(TASK_DATA_OWNER) && !data_.IsNull()) {
      auto *ipc_manager = CHI_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(data_.Cast<char>());
      }
    }
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(blocks_, length_);
    // Use BULK_EXPOSE to indicate metadata only - receiver will allocate buffer
    ar.bulk(data_, length_, BULK_EXPOSE);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(length_, bytes_read_);
    // Use BULK_XFER to actually transfer the read data back
    ar.bulk(data_, length_, BULK_XFER);
  }

  /** Aggregate */
  void Aggregate(const hipc::FullPtr<ReadTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }

  /**
   * Copy from another ReadTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<ReadTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy ReadTask-specific fields
    blocks_ = other->blocks_;
    data_ = other->data_;
    length_ = other->length_;
    bytes_read_ = other->bytes_read_;
  }
};

/**
 * GetStatsTask - Get performance statistics and remaining size
 */
struct GetStatsTask : public chi::Task {
  // Task-specific data (no inputs)
  OUT PerfMetrics metrics_;      // Performance metrics
  OUT chi::u64 remaining_size_;  // Remaining allocatable space

  /** SHM default constructor */
  GetStatsTask() : chi::Task(), remaining_size_(0) {}

  /** Emplace constructor */
  explicit GetStatsTask(const chi::TaskId &task_node,
                        const chi::PoolId &pool_id,
                        const chi::PoolQuery &pool_query)
      : chi::Task(task_node, pool_id, pool_query, 10), remaining_size_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kGetStats;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No additional input parameters
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(metrics_, remaining_size_);
  }

  /**
   * Copy from another GetStatsTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const hipc::FullPtr<GetStatsTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy GetStatsTask-specific fields
    metrics_ = other->metrics_;
    remaining_size_ = other->remaining_size_;
  }

  /** Aggregate replica results into this task */
  void Aggregate(const hipc::FullPtr<GetStatsTask> &other) {
    Task::Aggregate(other.template Cast<Task>());
    Copy(other);
  }
};

/**
 * Standard DestroyTask for bdev
 * All ChiMods should use the same DestroyTask structure from admin
 */
using DestroyTask = chimaera::admin::DestroyTask;

}  // namespace chimaera::bdev

#endif  // BDEV_TASKS_H_