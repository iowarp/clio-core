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

#ifndef BDEV_TASKS_H_
#define BDEV_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/config_manager.h>
#include <clio_ctp/introspect/system_info.h>
#include <yaml-cpp/yaml.h>

#include "autogen/bdev_methods.h"
// Include admin tasks for BaseCreateTask
#include <clio_runtime/admin/admin_tasks.h>

/**
 * Task struct definitions for bdev
 *
 * Defines tasks for block device operations with libaio and data allocation
 */

namespace clio::run::bdev {

using MonitorTask = clio::run::admin::MonitorTask;

/**
 * Default RAM-tier sizing policy.
 *
 * When a RAM bdev is configured with a capacity of 0 (e.g. "0g"), it is
 * NOT treated as unbounded. An unbounded RAM tier lets the allocator
 * hand out more than physical memory and OOM-kills the daemon on a
 * shared compute node. Instead it defaults to a fixed fraction of the
 * machine's total physical DRAM. This is the single source of truth for
 * that policy — CTE (context-transfer-engine) uses the same helper so a
 * "0g" RAM device means the same thing whether the bdev is created
 * directly or through CTE's storage config.
 */
constexpr double kDefaultRamCapacityFraction = 0.80;

/** Bytes to use for a RAM bdev configured with capacity 0/"0g": 80% of
 *  total system DRAM. */
inline clio::run::u64 DefaultRamCapacityBytes() {
  return static_cast<clio::run::u64>(
      static_cast<double>(ctp::SystemInfo::GetRamCapacity()) *
      kDefaultRamCapacityFraction);
}

/**
 * Block device type enumeration
 */
enum class BdevType : clio::run::u32 {
  kFile = 0,    // File-based block device (default)
  kRam = 1,     // RAM-based block device
  kHbm = 2,     // GPU High-Bandwidth Memory via cudaMalloc (device memory)
  kPinned = 3,  // Pinned host memory via cudaMallocHost
  kNoop = 4,    // No-op backend for latency testing (no actual I/O)
  kS3 = 5       // Amazon S3 object store backend
};

/**
 * Block structure for data allocation
 */
struct Block {
  clio::run::u64 offset_;      // Offset within file
  clio::run::u64 size_;        // Size of block
  clio::run::u32 block_type_;  // Block size category (0=4KB, 1=64KB, 2=256KB, 3=1MB)

  CTP_GPU_FUN Block() : offset_(0), size_(0), block_type_(0) {}
  CTP_GPU_FUN Block(clio::run::u64 offset, clio::run::u64 size, clio::run::u32 block_type)
      : offset_(offset), size_(size), block_type_(block_type) {}

  // Cereal serialization
  template <class Archive>
  CTP_CROSS_FUN void serialize(Archive &ar) {
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

  CTP_CROSS_FUN PerfMetrics()
      : read_bandwidth_mbps_(0.0),
        write_bandwidth_mbps_(0.0),
        read_latency_us_(0.0),
        write_latency_us_(0.0),
        iops_(0.0) {}

  // Cereal serialization
  template <class Archive>
  CTP_CROSS_FUN void serialize(Archive &ar) {
    ar(read_bandwidth_mbps_, write_bandwidth_mbps_, read_latency_us_,
       write_latency_us_, iops_);
  }
};

/**
 * Persistence level for block devices
 */
enum class PersistenceLevel : clio::run::u32 {
  kVolatile = 0,            // RAM-backed, lost on crash (e.g., RAM bdev)
  kTemporaryNonVolatile = 1, // File-backed but not long-term (e.g., local SSD scratch)
  kLongTerm = 2             // Durable persistent storage (e.g., PFS, NVMe)
};

/**
 * CreateParams for bdev chimod
 * Contains configuration parameters for bdev container creation
 */
struct CreateParams {
  // bdev-specific parameters
  BdevType bdev_type_;   // Block device type (file or RAM)
  clio::run::u64 total_size_;  // Total size for allocation (0 = file size for kFile,
                         // required for kRam)
  clio::run::u32 io_depth_;    // libaio queue depth (ignored for kRam)
  clio::run::u32 alignment_;   // I/O alignment (default 4096)

  // Performance characteristics (user-defined instead of benchmarked)
  PerfMetrics perf_metrics_;  // User-provided performance characteristics

  // Persistence level for this block device
  PersistenceLevel persistence_level_ = PersistenceLevel::kVolatile;

  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "clio_bdev";

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
  CreateParams(BdevType bdev_type, clio::run::u64 total_size = 0,
               clio::run::u32 io_depth = 32, clio::run::u32 alignment = 4096)
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
         static_cast<clio::run::u32>(bdev_type_), total_size_, io_depth_, alignment_);
  }

  // Constructor with optional performance metrics (as last parameter)
  CreateParams(BdevType bdev_type, clio::run::u64 total_size, clio::run::u32 io_depth,
               clio::run::u32 alignment, const PerfMetrics *perf_metrics = nullptr)
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
           static_cast<clio::run::u32>(bdev_type_), total_size_, io_depth_,
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
           static_cast<clio::run::u32>(bdev_type_), total_size_, io_depth_,
           alignment_);
    }
  }

  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive &ar) {
    ar(bdev_type_, total_size_, io_depth_, alignment_, perf_metrics_, persistence_level_);
  }

  /**
   * Load configuration from PoolConfig (for compose mode)
   * @param pool_config Pool configuration from compose section
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    // Parse YAML config string
    YAML::Node config = YAML::Load(pool_config.config_);

    // Load bdev type (optional, defaults to kFile)
    if (config["bdev_type"]) {
      std::string type_str = config["bdev_type"].as<std::string>();
      if (type_str == "file") {
        bdev_type_ = BdevType::kFile;
      } else if (type_str == "ram") {
        bdev_type_ = BdevType::kRam;
      } else if (type_str == "hbm") {
        bdev_type_ = BdevType::kHbm;
      } else if (type_str == "pinned") {
        bdev_type_ = BdevType::kPinned;
      } else if (type_str == "noop") {
        bdev_type_ = BdevType::kNoop;
      }
    }

    // Load capacity/total_size (parse size strings like "2GB", "512MB")
    if (config["capacity"]) {
      std::string capacity_str = config["capacity"].as<std::string>();
      total_size_ = ctp::ConfigParse::ParseSize(capacity_str);
    }

    // Load io_depth (optional)
    if (config["io_depth"]) {
      io_depth_ = config["io_depth"].as<clio::run::u32>();
    }

    // Load alignment (optional)
    if (config["alignment"]) {
      alignment_ = config["alignment"].as<clio::run::u32>();
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

    if (config["persistence_level"]) {
      std::string pl_str = config["persistence_level"].as<std::string>();
      if (pl_str == "volatile") {
        persistence_level_ = PersistenceLevel::kVolatile;
      } else if (pl_str == "temporary") {
        persistence_level_ = PersistenceLevel::kTemporaryNonVolatile;
      } else if (pl_str == "long_term") {
        persistence_level_ = PersistenceLevel::kLongTerm;
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
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * AllocateBlocksTask - Allocate multiple blocks with specified total size
 */
struct AllocateBlocksTask : public clio::run::Task {
  // Task-specific data
  IN clio::run::u64 size_;  // Requested total size
  OUT clio::run::priv::vector<Block> blocks_;  // Allocated blocks information

  /** SHM default constructor */
  CTP_CROSS_FUN AllocateBlocksTask() : clio::run::Task(), size_(0), blocks_(CLIO_PRIV_ALLOC) {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit AllocateBlocksTask(const clio::run::TaskId &task_node,
                              const clio::run::PoolId &pool_id,
                              const clio::run::PoolQuery &pool_query, clio::run::u64 size)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kAllocateBlocks), size_(size), blocks_(CLIO_PRIV_ALLOC) {
  }

  /** Fix up priv::vector SVO pointer after cudaMemcpy D→H */
  CTP_CROSS_FUN void FixupAfterCopy() {
    blocks_.FixupSvoPtr();
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(size_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(blocks_);
  }

  /**
   * Copy from another AllocateBlocksTask (assumes this task is already
   * constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<AllocateBlocksTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy AllocateBlocksTask-specific fields
    size_ = other->size_;
    blocks_ = other->blocks_;
  }

  /** AggregateOut replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<AllocateBlocksTask>());
  }
};

/**
 * FreeBlocksTask - Free allocated blocks
 */
struct FreeBlocksTask : public clio::run::Task {
  // Task-specific data
  IN clio::run::priv::vector<Block> blocks_;  // Blocks to free

  /** SHM default constructor */
  CTP_CROSS_FUN FreeBlocksTask() : clio::run::Task(), blocks_(CLIO_PRIV_ALLOC) {}

  /** Emplace constructor for multiple blocks */
  explicit FreeBlocksTask(const clio::run::TaskId &task_node,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const std::vector<Block> &blocks)
      : clio::run::Task(task_node, pool_id, pool_query, 10), blocks_(CLIO_PRIV_ALLOC) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kFreeBlocks;
    task_flags_.Clear();
    pool_query_ = pool_query;

    // Copy blocks from std::vector to clio::run::priv::vector
    for (const auto &block : blocks) {
      blocks_.push_back(block);
    }
  }

  /** Emplace constructor for GPU (priv::vector) */
  CTP_CROSS_FUN explicit FreeBlocksTask(const clio::run::TaskId &task_node,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const clio::run::priv::vector<Block> &blocks)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kFreeBlocks),
        blocks_(blocks) {
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(blocks_);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // No additional output parameters
  }

  /** Fix up SVO pointer after POD copy (e.g., CPU→GPU memcpy) */
  CTP_CROSS_FUN void FixupAfterCopy() {
    blocks_.FixupSvoPtr();
  }

  /**
   * Copy from another FreeBlocksTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<FreeBlocksTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy FreeBlocksTask-specific fields
    blocks_ = other->blocks_;
  }

  /** AggregateOut replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<FreeBlocksTask>());
  }
};

/**
 * WriteTask - Write data to a block using libaio
 */
struct WriteTask : public clio::run::Task {
  // Task-specific data
  IN clio::run::priv::vector<Block> blocks_;  // Blocks to write to
  IN ctp::ipc::ShmPtr<> data_;              // Data to write (pointer-based)
  IN size_t length_;                    // Size of data to write
  OUT clio::run::u64 bytes_written_;          // Number of bytes actually written

  /** SHM default constructor */
  CTP_CROSS_FUN WriteTask() : clio::run::Task(), blocks_(CLIO_PRIV_ALLOC), length_(0), bytes_written_(0) {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit WriteTask(const clio::run::TaskId &task_node, const clio::run::PoolId &pool_id,
                     const clio::run::PoolQuery &pool_query,
                     const clio::run::priv::vector<Block> &blocks, ctp::ipc::ShmPtr<> data,
                     size_t length)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kWrite),
        blocks_(blocks),
        data_(data),
        length_(length),
        bytes_written_(0) {
  }

  /** Destructor - free buffer if TASK_DATA_OWNER is set */
  CTP_CROSS_FUN ~WriteTask() {
#if !CTP_IS_DEVICE_PASS
    if (task_flags_.Any(TASK_DATA_OWNER) && !data_.IsNull()) {
      auto *ipc_manager = CLIO_CPU_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(data_.Cast<char>());
      }
    }
#endif
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(blocks_, length_);
    // Use bulk transfer for data pointer - BULK_XFER for actual data
    // transmission
    ar.bulk(data_, length_, BULK_XFER);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(bytes_written_);
  }

  /** Fix up priv::vector SVO pointer after cudaMemcpy D→H */
  CTP_CROSS_FUN void FixupAfterCopy() {
    blocks_.FixupSvoPtr();
  }

  /** AggregateOut */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<WriteTask>());
  }

  /**
   * Copy from another WriteTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<WriteTask> &other) {
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
struct ReadTask : public clio::run::Task {
  // Task-specific data
  IN clio::run::priv::vector<Block> blocks_;  // Blocks to read from
  OUT ctp::ipc::ShmPtr<> data_;             // Read data (pointer-based)
  INOUT size_t
      length_;  // Size of data buffer (IN: buffer size, OUT: actual size)
  OUT clio::run::u64 bytes_read_;  // Number of bytes actually read

  /** SHM default constructor */
  CTP_CROSS_FUN ReadTask() : clio::run::Task(), blocks_(CLIO_PRIV_ALLOC), length_(0), bytes_read_(0) {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit ReadTask(const clio::run::TaskId &task_node, const clio::run::PoolId &pool_id,
                    const clio::run::PoolQuery &pool_query,
                    const clio::run::priv::vector<Block> &blocks, ctp::ipc::ShmPtr<> data,
                    size_t length)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kRead),
        blocks_(blocks),
        data_(data),
        length_(length),
        bytes_read_(0) {
  }

  /** Destructor - free buffer if TASK_DATA_OWNER is set */
  CTP_CROSS_FUN ~ReadTask() {
#if !CTP_IS_DEVICE_PASS
    if (task_flags_.Any(TASK_DATA_OWNER) && !data_.IsNull()) {
      auto *ipc_manager = CLIO_CPU_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(data_.Cast<char>());
      }
    }
#endif
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(blocks_, length_);
    // Use BULK_EXPOSE to indicate metadata only - receiver will allocate buffer
    ar.bulk(data_, length_, BULK_EXPOSE);
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(length_, bytes_read_);
    // Use BULK_XFER to actually transfer the read data back
    ar.bulk(data_, length_, BULK_XFER);
  }

  /** Fix up priv::vector SVO pointer after cudaMemcpy D→H */
  CTP_CROSS_FUN void FixupAfterCopy() {
    blocks_.FixupSvoPtr();
  }

  /** AggregateOut */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<ReadTask>());
  }

  /**
   * Copy from another ReadTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<ReadTask> &other) {
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
struct GetStatsTask : public clio::run::Task {
  // Task-specific data (no inputs)
  OUT PerfMetrics metrics_;      // Performance metrics
  OUT clio::run::u64 remaining_size_;  // Remaining allocatable space

  /** SHM default constructor */
  GetStatsTask() : clio::run::Task(), remaining_size_(0) {}

  /** Emplace constructor */
  explicit GetStatsTask(const clio::run::TaskId &task_node,
                        const clio::run::PoolId &pool_id,
                        const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_node, pool_id, pool_query, 10), remaining_size_(0) {
    // Initialize task
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kGetStats;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Serialize IN and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No additional input parameters
  }

  /** Serialize OUT and INOUT parameters */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(metrics_, remaining_size_);
  }

  /**
   * Copy from another GetStatsTask (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  void Copy(const ctp::ipc::FullPtr<GetStatsTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy GetStatsTask-specific fields
    metrics_ = other->metrics_;
    remaining_size_ = other->remaining_size_;
  }

  /** AggregateOut replica results into this task */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<GetStatsTask>());
  }
};

/**
 * UpdateTask - Send device/pinned memory pointers to the GPU container.
 * Called by the CPU bdev runtime after allocating kHbm or kPinned memory,
 * so the GPU-side GpuRuntime container can perform direct device memcpy.
 */
struct UpdateTask : public clio::run::Task {
  IN clio::run::u64 hbm_ptr_;     ///< Device pointer to HBM buffer (0 if none)
  IN clio::run::u64 pinned_ptr_;  ///< Pinned host pointer (0 if none)
  IN clio::run::u64 hbm_size_;    ///< Byte size of the HBM buffer
  IN clio::run::u64 pinned_size_; ///< Byte size of the pinned buffer
  IN clio::run::u64 total_size_;  ///< Allocatable size (max of hbm/pinned)
  IN clio::run::u32 bdev_type_;   ///< BdevType enum value
  IN clio::run::u32 alignment_;   ///< Allocation alignment in bytes

  CTP_CROSS_FUN UpdateTask()
      : clio::run::Task(), hbm_ptr_(0), pinned_ptr_(0), hbm_size_(0),
        pinned_size_(0), total_size_(0), bdev_type_(0), alignment_(4096) {}

  explicit UpdateTask(const clio::run::TaskId &task_node,
                      const clio::run::PoolId &pool_id,
                      const clio::run::PoolQuery &pool_query,
                      clio::run::u64 hbm_ptr, clio::run::u64 pinned_ptr,
                      clio::run::u64 hbm_size, clio::run::u64 pinned_size,
                      clio::run::u64 total_size, clio::run::u32 bdev_type,
                      clio::run::u32 alignment)
      : clio::run::Task(task_node, pool_id, pool_query, 10),
        hbm_ptr_(hbm_ptr), pinned_ptr_(pinned_ptr),
        hbm_size_(hbm_size), pinned_size_(pinned_size),
        total_size_(total_size), bdev_type_(bdev_type), alignment_(alignment) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kUpdate;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(hbm_ptr_, pinned_ptr_, hbm_size_, pinned_size_,
       total_size_, bdev_type_, alignment_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  void Copy(const ctp::ipc::FullPtr<UpdateTask> &other) {
    Task::Copy(other.template Cast<Task>());
    hbm_ptr_     = other->hbm_ptr_;
    pinned_ptr_  = other->pinned_ptr_;
    hbm_size_    = other->hbm_size_;
    pinned_size_ = other->pinned_size_;
    total_size_  = other->total_size_;
    bdev_type_   = other->bdev_type_;
    alignment_   = other->alignment_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<UpdateTask>());
  }
};

/**
 * Standard DestroyTask for bdev
 * All ChiMods should use the same DestroyTask structure from admin
 */
using DestroyTask = clio::run::admin::DestroyTask;

}  // namespace clio::run::bdev

#endif  // BDEV_TASKS_H_