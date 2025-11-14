#include "chimaera/worker.h"
#include "hermes_shm/util/logging.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <wrp_cte/core/core_config.h>
#include <wrp_cte/core/core_dpe.h>
#include <wrp_cte/core/core_runtime.h>

namespace wrp_cte::core {

// Bring chi namespace items into scope for CHI_CUR_WORKER macro
using chi::chi_cur_worker_key_;
using chi::Worker;

// No more static member definitions - using instance-based locking

chi::u64 Runtime::ParseCapacityToBytes(const std::string &capacity_str) {
  if (capacity_str.empty()) {
    return 0;
  }

  // Parse numeric part
  double value = 0.0;
  size_t pos = 0;
  try {
    value = std::stod(capacity_str, &pos);
  } catch (const std::exception &) {
    HILOG(kWarning, "Invalid capacity format: {}", capacity_str);
    return 0;
  }

  // Parse suffix (case-insensitive)
  std::string suffix = capacity_str.substr(pos);
  // Remove whitespace
  suffix.erase(std::remove_if(suffix.begin(), suffix.end(), ::isspace),
               suffix.end());

  // Convert to uppercase for case-insensitive comparison
  std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::toupper);

  chi::u64 multiplier = 1;
  if (suffix.empty() || suffix == "B" || suffix == "BYTES") {
    multiplier = 1;
  } else if (suffix == "KB" || suffix == "K") {
    multiplier = 1024ULL;
  } else if (suffix == "MB" || suffix == "M") {
    multiplier = 1024ULL * 1024ULL;
  } else if (suffix == "GB" || suffix == "G") {
    multiplier = 1024ULL * 1024ULL * 1024ULL;
  } else if (suffix == "TB" || suffix == "T") {
    multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
  } else {
    HILOG(kWarning, "Unknown capacity suffix: {}", suffix);
    return static_cast<chi::u64>(value);
  }

  return static_cast<chi::u64>(value * multiplier);
}

void Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext &ctx) {
  // Initialize unordered_map_ll instances with 64 buckets to match lock count
  // This ensures each bucket can have its own lock for maximum concurrency
  registered_targets_ =
      chi::unordered_map_ll<chi::PoolId, TargetInfo>(kMaxLocks);
  target_name_to_id_ =
      chi::unordered_map_ll<std::string, chi::PoolId>(kMaxLocks);
  tag_name_to_id_ = chi::unordered_map_ll<std::string, TagId>(kMaxLocks);
  tag_id_to_info_ = chi::unordered_map_ll<TagId, TagInfo>(kMaxLocks);
  tag_blob_name_to_info_ =
      chi::unordered_map_ll<std::string, BlobInfo>(kMaxLocks);

  // Initialize lock vectors for concurrent access
  target_locks_.reserve(kMaxLocks);
  tag_locks_.reserve(kMaxLocks);
  for (size_t i = 0; i < kMaxLocks; ++i) {
    target_locks_.emplace_back(std::make_unique<chi::CoRwLock>());
    tag_locks_.emplace_back(std::make_unique<chi::CoRwLock>());
  }

  // Get main allocator from IPC manager
  auto *ipc_manager = CHI_IPC;
  auto *main_allocator = ipc_manager->GetMainAllocator();

  // Initialize telemetry ring buffer
  telemetry_log_ = hipc::circular_mpsc_queue<CteTelemetry>(main_allocator,
                                                           kTelemetryRingSize);

  // Initialize atomic counters
  next_tag_id_minor_ = 1;
  telemetry_counter_ = 0;

  // Get configuration from params (loaded from pool_config.config_ via
  // LoadConfig)
  auto params = task->GetParams(main_allocator);
  config_ = params.config_;

  // Configuration is now loaded from compose pool_config via
  // CreateParams::LoadConfig()

  // Store storage configuration in runtime
  storage_devices_ = config_.storage_.devices_;

  // Initialize the client with the pool ID
  client_.Init(task->new_pool_id_);

  // Register targets for each configured storage device and neighborhood node
  if (!storage_devices_.empty()) {
    // Get neighborhood size from configuration
    chi::u32 neighborhood_size = config_.targets_.neighborhood_;

    // Get number of nodes from IPC manager
    chi::u32 num_nodes = ipc_manager->GetNumHosts();

    // Set actual neighborhood size to minimum of configured size and available
    // nodes
    chi::u32 actual_neighborhood = std::min(neighborhood_size, num_nodes);

    HILOG(kDebug,
          "Registering targets for storage devices across neighborhood (size: "
          "{} nodes):",
          actual_neighborhood);

    // Iterate over storage devices
    for (size_t device_idx = 0; device_idx < storage_devices_.size();
         ++device_idx) {
      const auto &device = storage_devices_[device_idx];

      // Capacity is already in bytes
      chi::u64 capacity_bytes = device.capacity_limit_;

      // Determine bdev type enum
      chimaera::bdev::BdevType bdev_type = chimaera::bdev::BdevType::kFile;
      if (device.bdev_type_ == "ram") {
        bdev_type = chimaera::bdev::BdevType::kRam;
      }

      // Iterate over neighborhood nodes (container hashes from 0 to
      // actual_neighborhood-1)
      for (chi::u32 container_hash = 0; container_hash < actual_neighborhood;
           ++container_hash) {
        // Generate unique target path for this device-node combination
        std::string target_path =
            device.path_ + "_node" + std::to_string(container_hash);

        // Create target query using DirectHash for this specific container
        chi::PoolQuery target_query =
            chi::PoolQuery::DirectHash(container_hash);

        // Generate unique bdev_id: base major (513) + device index, minor is
        // container hash
        chi::PoolId bdev_id(513 + static_cast<chi::u32>(device_idx), 0);

        // Call RegisterTarget using client member variable with target_query
        // and bdev_id
        HILOG(kDebug,
              "Registering target ({}): {} ({}, {} bytes) on node {} with "
              "bdev_id=({},{})",
              client_.pool_id_, target_path, device.bdev_type_, capacity_bytes,
              container_hash, bdev_id.major_, bdev_id.minor_);
        chi::u32 result =
            client_.RegisterTarget(hipc::MemContext(), target_path, bdev_type,
                                   capacity_bytes, target_query, bdev_id);

        if (result == 0) {
          HILOG(kDebug, "  - Registered target: {} ({}, {} bytes) on node {}",
                target_path, device.bdev_type_, capacity_bytes, container_hash);
        } else {
          HILOG(kWarning,
                "  - Failed to register target {} on node {} (error code: {})",
                target_path, container_hash, result);
        }
      }
    }
  } else {
    HELOG(kWarning, "Warning: No storage devices configured");
  }

  // Queue management has been removed - queues are now managed by Chimaera
  // runtime Local queues (kTargetManagementQueue, kTagManagementQueue,
  // kBlobOperationsQueue, kStatsQueue) are no longer created explicitly

  HILOG(kInfo,
        "CTE Core container created and initialized for pool: {} (ID: {})",
        pool_name_, task->new_pool_id_);

  HILOG(kInfo, "Configuration: neighborhood={}, poll_period_ms={}",
        config_.targets_.neighborhood_, config_.targets_.poll_period_ms_);
}

void Runtime::Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &ctx) {
  try {
    // Clear all registered targets and their associated data
    registered_targets_.clear();
    target_name_to_id_.clear();

    // Clear tag and blob management structures
    tag_name_to_id_.clear();
    tag_id_to_info_.clear();
    tag_blob_name_to_info_.clear();

    // Reset atomic counters
    next_tag_id_minor_.store(1);

    // Clear storage device configuration
    storage_devices_.clear();

    // Clear lock vectors
    target_locks_.clear();
    tag_locks_.clear();

    // Set success status
    task->return_code_ = 0;

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
}

void Runtime::RegisterTarget(hipc::FullPtr<RegisterTargetTask> task,
                             chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    return;
  }

  try {
    std::string target_name = task->target_name_.str();
    chimaera::bdev::BdevType bdev_type = task->bdev_type_;
    chi::u64 total_size = task->total_size_;
    chi::PoolId bdev_pool_id = task->bdev_id_;
    HILOG(kDebug, "Registering target ({}): {} ({} bytes) with bdev_id=({},{})",
          client_.pool_id_, target_name, total_size, bdev_pool_id.major_,
          bdev_pool_id.minor_);

    // Create bdev client and container first to get the TargetId (pool_id)
    chimaera::bdev::Client bdev_client;
    std::string bdev_pool_name =
        target_name; // Use target_name as the bdev pool name

    HILOG(kDebug, "Creating bdev with pool ID: major={}, minor={}",
          bdev_pool_id.major_, bdev_pool_id.minor_);

    // Create the bdev container using the client
    chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
    bdev_client.Create(hipc::MemContext(), pool_query, target_name,
                       bdev_pool_id, bdev_type, total_size);

    // Check if creation was successful
    if (bdev_client.return_code_ != 0) {
      HELOG(kError, "Failed to create bdev container {} : {}", target_name,
            bdev_client.return_code_);
      task->return_code_.store(1);
      return;
    }

    // Get the TargetId (bdev_client's pool_id) for indexing
    chi::PoolId target_id = bdev_client.pool_id_;

    // Check if target is already registered using TargetId
    size_t lock_index = GetTargetLockIndex(target_id);
    {
      chi::ScopedCoRwReadLock read_lock(*target_locks_[lock_index]);
      TargetInfo *existing_target = registered_targets_.find(target_id);
      if (existing_target != nullptr) {
        return;
      }
    }

    // Get actual statistics from bdev using GetStats method
    chi::u64 remaining_size;
    chimaera::bdev::PerfMetrics perf_metrics =
        bdev_client.GetStats(hipc::MemContext(), remaining_size);

    // Create target info with bdev client and performance stats
    auto *ipc_manager = CHI_IPC;
    auto *main_allocator = ipc_manager->GetMainAllocator();
    TargetInfo target_info(main_allocator);
    target_info.target_name_ = target_name;
    target_info.bdev_pool_name_ = bdev_pool_name;
    target_info.bdev_client_ = std::move(bdev_client);
    target_info.target_query_ =
        task->target_query_; // Store target query for bdev API calls
    target_info.bytes_read_ = 0;
    target_info.bytes_written_ = 0;
    target_info.ops_read_ = 0;
    target_info.ops_written_ = 0;
    // Check if this target has a manually configured score from storage device
    // config
    float manual_score = GetManualScoreForTarget(target_name);
    if (manual_score >= 0.0f) {
      target_info.target_score_ = manual_score; // Use configured manual score
      HILOG(kDebug, "Target '{}' using manual score: {:.2f}", target_name,
            manual_score);
    } else {
      target_info.target_score_ =
          0.0f; // Will be calculated based on performance metrics
    }
    target_info.remaining_space_ =
        total_size; // Use actual remaining space from bdev
    target_info.perf_metrics_ =
        perf_metrics; // Store the entire PerfMetrics structure

    // Register the target using TargetId as key
    {
      chi::ScopedCoRwWriteLock write_lock(*target_locks_[lock_index]);
      registered_targets_.insert_or_assign(target_id, target_info);
      target_name_to_id_.insert_or_assign(target_name,
                                          target_id); // Maintain reverse lookup
    }

    task->return_code_.store(0); // Success
    HILOG(kDebug,
          "Target '{}' registered with ID (major={}, minor={}) - bdev pool: {} "
          "(type={}, path={}, "
          "size={}, remaining={})",
          target_name, target_id.major_, target_id.minor_, bdev_pool_name,
          static_cast<int>(bdev_type), target_name, total_size, remaining_size);
    HILOG(kDebug,
          "  Initial statistics: read_bw={} MB/s, write_bw={} MB/s, "
          "avg_latency={} μs, iops={}",
          perf_metrics.read_bandwidth_mbps_, perf_metrics.write_bandwidth_mbps_,
          (target_info.perf_metrics_.read_latency_us_ +
           target_info.perf_metrics_.write_latency_us_) /
              2.0,
          perf_metrics.iops_);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::UnregisterTarget(hipc::FullPtr<UnregisterTargetTask> task,
                               chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    return;
  }

  try {
    std::string target_name = task->target_name_.str();

    // Look up TargetId from target_name
    chi::PoolId *target_id_ptr = target_name_to_id_.find(target_name);
    if (target_id_ptr == nullptr) {
      task->return_code_.store(1);
      return;
    }

    const chi::PoolId &target_id = *target_id_ptr;

    // Check if target exists and remove it (don't destroy bdev container)
    size_t lock_index = GetTargetLockIndex(target_id);
    {
      chi::ScopedCoRwWriteLock write_lock(*target_locks_[lock_index]);
      if (!registered_targets_.contains(target_id)) {
        task->return_code_.store(1);
        return;
      }

      registered_targets_.erase(target_id);
      target_name_to_id_.erase(target_name); // Remove reverse lookup
    }

    task->return_code_.store(0); // Success
    HILOG(kDebug, "Target '{}' unregistered", target_name);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::ListTargets(hipc::FullPtr<ListTargetsTask> task,
                          chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    return;
  }

  try {
    // Clear the output vector and populate with current target names
    task->target_names_.clear();

    // Use a single lock based on hash of operation type for listing
    size_t lock_index =
        std::hash<std::string>{}("list_targets") % target_locks_.size();
    chi::ScopedCoRwReadLock read_lock(*target_locks_[lock_index]);

    // Populate target name list while lock is held
    task->target_names_.reserve(registered_targets_.size());
    registered_targets_.for_each(
        [&task](const chi::PoolId &target_id, const TargetInfo &target_info) {
          task->target_names_.emplace_back(target_info.target_name_.c_str());
        });

    task->return_code_.store(0); // Success

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::StatTargets(hipc::FullPtr<StatTargetsTask> task,
                          chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    return;
  }

  try {
    // Update performance stats for all registered targets
    // Use a single lock based on hash of operation type for stats
    size_t lock_index =
        std::hash<std::string>{}("stat_targets") % target_locks_.size();
    chi::ScopedCoRwReadLock read_lock(*target_locks_[lock_index]);

    // Update stats for all targets - read lock is sufficient since we're only
    // updating values, not modifying map structure
    registered_targets_.for_each(
        [this](const chi::PoolId &target_id, TargetInfo &target_info) {
          UpdateTargetStats(target_id, target_info);
        });

    task->return_code_.store(0); // Success

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

template <typename CreateParamsT>
void Runtime::GetOrCreateTag(
    hipc::FullPtr<GetOrCreateTagTask<CreateParamsT>> task,
    chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    std::string tag_name = task->tag_name_.str();
    // Check if tag exists locally
    TagId *existing_tag_id = tag_name_to_id_.find(tag_name);
    if (existing_tag_id != nullptr) {
      // Tag exists locally, resolve locally
      task->pool_query_ = chi::PoolQuery::Local();
    } else {
      // Tag doesn't exist locally, route to canonical node using DirectHash
      std::hash<std::string> string_hasher;
      chi::u32 hash_value = static_cast<chi::u32>(string_hasher(tag_name));
      task->pool_query_ = chi::PoolQuery::DirectHash(hash_value);
    }
    return;
  }

  try {
    std::string tag_name = task->tag_name_.str();
    TagId preferred_id = task->tag_id_;
    auto *ipc_manager = CHI_IPC;
    chi::u32 local_node_id = ipc_manager->GetNodeId();

    // Check if this is a returning task from a remote canonical node
    // If preferred_id is already set and not local, we're receiving a remote
    // tag
    bool is_remote_tag =
        (preferred_id.major_ != 0 && preferred_id.major_ != local_node_id);

    if (is_remote_tag) {
      // Non-canonical node: Only cache the name→TagId mapping
      size_t tag_lock_index = GetTagLockIndex(tag_name);
      chi::ScopedCoRwWriteLock write_lock(*tag_locks_[tag_lock_index]);

      // Check if already cached
      TagId *existing_tag_id_ptr = tag_name_to_id_.find(tag_name);
      if (existing_tag_id_ptr == nullptr) {
        // Cache the mapping without creating TagInfo
        tag_name_to_id_.insert_or_assign(tag_name, preferred_id);
      }

      task->tag_id_ = preferred_id;
      task->return_code_.store(0);
      return;
    }

    // Canonical node: Create full TagInfo structure
    TagId tag_id = GetOrAssignTagId(tag_name, preferred_id);
    task->tag_id_ = tag_id;

    // Update timestamp and log telemetry
    size_t tag_lock_index = GetTagLockIndex(tag_name);
    auto now = std::chrono::steady_clock::now();
    {
      chi::ScopedCoRwReadLock read_lock(*tag_locks_[tag_lock_index]);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr != nullptr) {
        // Update read timestamp
        tag_info_ptr->last_read_ = now;

        // Log telemetry for GetOrCreateTag operation
        LogTelemetry(CteOp::kGetOrCreateTag, 0, 0, tag_id,
                     tag_info_ptr->last_modified_, now);
      }
    }

    task->return_code_.store(0); // Success

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::PutBlob(hipc::FullPtr<PutBlobTask> task, chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    chi::u64 offset = task->offset_;
    chi::u64 size = task->size_;
    hipc::Pointer blob_data = task->blob_data_;
    float blob_score = task->score_;
    chi::u32 flags = task->flags_;

    // Suppress unused variable warning for flags - may be used in future
    (void)flags;

    // Validate input parameters
    if (size == 0) {
      task->return_code_.store(2); // Error: Invalid size (zero)
      return;
    }

    if (blob_data.IsNull()) {
      task->return_code_.store(3); // Error: Null data pointer
      return;
    }

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_.store(4); // Error: No blob name provided
      return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    bool blob_found = (blob_info_ptr != nullptr);

    // Step 2: Create blob if it doesn't exist
    if (!blob_found) {
      blob_info_ptr = CreateNewBlob(blob_name, tag_id, blob_score);
      if (blob_info_ptr == nullptr) {
        task->return_code_.store(5); // Error: Failed to create blob
        return;
      }
    }

    // Step 2.5: Track blob size before modification for tag total_size_
    // accounting (no lock needed - blob_info_ptr is already obtained)
    chi::u64 old_blob_size = blob_info_ptr->GetTotalSize();

    // Step 3: Allocate additional space if needed for blob extension
    // (no lock held during expensive bdev allocation)
    chi::u32 allocation_result =
        AllocateNewData(*blob_info_ptr, offset, size, blob_score);

    if (allocation_result != 0) {
      HELOG(kError, "Allocation failure: {}", allocation_result);
      task->return_code_.store(
          10 + allocation_result); // Error: Allocation failure (10-19 range)
      return;
    }

    // Step 4: Write data to blob blocks
    // (no lock held during expensive I/O operations)
    chi::u32 write_result =
        ModifyExistingData(blob_info_ptr->blocks_, blob_data, size, offset);

    if (write_result != 0) {
      task->return_code_.store(
          20 + write_result); // Error: Write failure (20-29 range)
      return;
    }

    // Step 5: Calculate size change after I/O completes
    chi::u64 new_blob_size = blob_info_ptr->GetTotalSize();
    chi::i64 size_change = static_cast<chi::i64>(new_blob_size) -
                           static_cast<chi::i64>(old_blob_size);

    // Step 6: Update metadata (read lock only for map access - not modifying
    // map structure)
    auto now = std::chrono::steady_clock::now();
    size_t tag_lock_index = GetTagLockIndex(tag_id);
    size_t tag_total_size = 0;

    // Update blob timestamp and score (blob_info_ptr already obtained, no
    // additional lock needed)
    blob_info_ptr->last_modified_ = now;
    blob_info_ptr->score_ = blob_score;

    // Acquire read lock for tag map access and value updates
    {
      chi::ScopedCoRwReadLock tag_lock(*tag_locks_[tag_lock_index]);

      // Update tag's total_size_ and timestamps
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr != nullptr) {
        tag_info_ptr->last_modified_ = now;

        // Use signed arithmetic to handle size decreases
        if (size_change >= 0) {
          tag_info_ptr->total_size_.fetch_add(static_cast<size_t>(size_change));
        } else {
          HELOG(kError, "Size should not decrese");
          task->return_code_.store(1);
          return;
        }
      }
    } // Release read lock

    // Log telemetry and success messages
    LogTelemetry(CteOp::kPutBlob, offset, size, tag_id, now,
                 blob_info_ptr->last_read_);

    task->return_code_.store(0);

  } catch (const std::exception &e) {
    HILOG(kError, "PutBlob failed with exception: {}", e.what());
    task->return_code_.store(1); // Error: General exception
  }
}

void Runtime::GetBlob(hipc::FullPtr<GetBlobTask> task, chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    chi::u64 offset = task->offset_;
    chi::u64 size = task->size_;
    chi::u32 flags = task->flags_;

    // Suppress unused variable warning for flags - may be used in future
    (void)flags;

    // Validate input parameters
    if (size == 0) {
      task->return_code_.store(1);
      return;
    }

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_.store(1);
      return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    // If blob doesn't exist, error
    if (blob_info_ptr == nullptr) {
      task->return_code_.store(1);
      return;
    }

    // Use the pre-provided data pointer from the task
    hipc::Pointer blob_data_ptr = task->blob_data_;

    // Step 2: Read data from blob blocks (no lock held during I/O)
    chi::u32 read_result =
        ReadData(blob_info_ptr->blocks_, blob_data_ptr, size, offset);
    if (read_result != 0) {
      task->return_code_.store(read_result);
      return;
    }

    // Step 3: Update timestamp (no lock needed - just updating values, not
    // modifying map structure)
    auto now = std::chrono::steady_clock::now();
    size_t tag_lock_index = GetTagLockIndex(tag_id);
    (void)tag_lock_index; // Suppress unused variable warning
    size_t num_blocks = 0;
    blob_info_ptr->last_read_ = now;
    num_blocks = blob_info_ptr->blocks_.size();

    // Log telemetry and success messages after releasing lock
    LogTelemetry(CteOp::kGetBlob, offset, size, tag_id,
                 blob_info_ptr->last_modified_, now);

    task->return_code_.store(0);
    HILOG(kDebug, "GetBlob successful: name={}, offset={}, size={}, blocks={}",
          blob_name, offset, size, num_blocks);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::ReorganizeBlob(hipc::FullPtr<ReorganizeBlobTask> task,
                             chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    float new_score = task->new_score_;

    // Validate inputs
    if (blob_name.empty()) {
      task->return_code_.store(1); // Invalid input - empty blob name
      return;
    }

    if (new_score < 0.0f || new_score > 1.0f) {
      task->return_code_.store(1); // Invalid score range
      return;
    }

    // Get configuration for score difference threshold
    const Config &config = GetConfig();
    float score_difference_threshold =
        config.performance_.score_difference_threshold_;

    // Step 1: Get blob info directly from table
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      task->return_code_.store(3); // Blob not found
      return;
    }

    // Step 2: Check if score needs updating
    float current_score = blob_info_ptr->score_;
    float score_diff = std::abs(new_score - current_score);
    HILOG(kDebug,
          "SCORE CHECK: blob={}, current={}, new={}, diff={}, threshold={}",
          blob_name, current_score, new_score, score_diff,
          score_difference_threshold);

    if (score_diff < score_difference_threshold) {
      // Score difference too small, no reorganization needed
      task->return_code_.store(0);
      HILOG(kDebug,
            "ReorganizeBlob: score difference below threshold, skipping");
      return;
    }

    // Step 3: Update blob score
    BlobInfo &blob_info = *blob_info_ptr;

    HILOG(kDebug, "UPDATING SCORE: blob={}, old_score={}, new_score={}",
          blob_name, blob_info.score_, new_score);
    blob_info.score_ = new_score;

    // Step 4: Get blob size from blob_info
    chi::u64 blob_size = blob_info.GetTotalSize();

    if (blob_size == 0) {
      // Empty blob, no data to reorganize
      task->return_code_.store(0);
      return;
    }

    // Step 5: Allocate buffer for blob data
    auto *ipc_manager = CHI_IPC;
    hipc::FullPtr<char> blob_data_buffer =
        ipc_manager->AllocateBuffer(blob_size);
    if (blob_data_buffer.IsNull()) {
      HILOG(kError, "Failed to allocate buffer for blob during reorganization");
      task->return_code_.store(5); // Buffer allocation failed
      return;
    }

    // Step 6: Get blob data
    auto get_task =
        client_.AsyncGetBlob(hipc::MemContext(), tag_id, blob_name, 0,
                             blob_size, 0, blob_data_buffer.shm_);
    get_task->Wait();

    if (get_task->return_code_.load() != 0) {
      HILOG(kWarning, "Failed to get blob data during reorganization");
      CHI_IPC->DelTask(get_task);
      task->return_code_.store(6); // Get blob failed
      return;
    }
    CHI_IPC->DelTask(get_task);

    // Step 7: Put blob with new score (data reorganization)
    HILOG(kDebug,
          "ReorganizeBlob calling AsyncPutBlob for blob={}, new_score={}",
          blob_name, new_score);
    auto put_task =
        client_.AsyncPutBlob(hipc::MemContext(), tag_id, blob_name, 0,
                             blob_size, blob_data_buffer.shm_, new_score, 0);
    put_task->Wait();

    if (put_task->return_code_.load() != 0) {
      HILOG(kWarning, "Failed to put blob during reorganization");
      CHI_IPC->DelTask(put_task);
      task->return_code_.store(7); // Put blob failed
      return;
    }
    CHI_IPC->DelTask(put_task);

    // Success
    task->return_code_.store(0);

    HILOG(kDebug,
          "ReorganizeBlob completed: tag_id={},{}, blob={}, new_score={}",
          tag_id.major_, tag_id.minor_, blob_name, new_score);

  } catch (const std::exception &e) {
    HILOG(kError, "ReorganizeBlob failed: {}", e.what());
    task->return_code_.store(1); // Error during reorganization
  }
}

void Runtime::DelBlob(hipc::FullPtr<DelBlobTask> task, chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_.store(1);
      return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    if (blob_info_ptr == nullptr) {
      task->return_code_.store(1); // Blob not found
      return;
    }

    // Step 2: Get blob size before deletion for tag size accounting
    chi::u64 blob_size = blob_info_ptr->GetTotalSize();

    // Step 2.5: Free all blocks back to their targets before removing blob
    chi::u32 free_result = FreeAllBlobBlocks(*blob_info_ptr);
    if (free_result != 0) {
      HILOG(kWarning,
            "Failed to free some blocks for blob={}, continuing with deletion",
            blob_name);
      // Continue with deletion even if freeing fails to avoid orphaned blob
      // entries
    }

    // Step 3: Update tag's total_size_
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr != nullptr) {
      // Step 4: Decrement tag's total_size_
      if (blob_size <= tag_info_ptr->total_size_) {
        tag_info_ptr->total_size_ -= blob_size;
      } else {
        tag_info_ptr->total_size_ = 0; // Clamp to 0 if we would underflow
      }
    }

    // Step 5: Remove blob from tag_blob_name_to_info_ map
    std::string compound_key = std::to_string(tag_id.major_) + "." +
                               std::to_string(tag_id.minor_) + "." + blob_name;
    tag_blob_name_to_info_.erase(compound_key);

    // Step 6: Log telemetry for DelBlob operation
    auto now = std::chrono::steady_clock::now();
    LogTelemetry(CteOp::kDelBlob, 0, blob_size, tag_id, now, now);

    // Success
    task->return_code_.store(0);
    HILOG(kDebug, "DelBlob successful: name={}, blob_size={}", blob_name,
          blob_size);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::DelTag(hipc::FullPtr<DelTagTask> task, chi::RunContext &ctx) {
  try {
    TagId tag_id = task->tag_id_;
    std::string tag_name = task->tag_name_.str();

    // Step 1: Resolve tag ID if tag name was provided instead
    if (tag_id.IsNull() && !tag_name.empty()) {
      // Look up tag ID by name
      TagId *found_tag_id_ptr = tag_name_to_id_.find(tag_name);
      if (found_tag_id_ptr == nullptr) {
        task->return_code_.store(1); // Tag not found by name
        return;
      }
      tag_id = *found_tag_id_ptr;
      task->tag_id_ = tag_id; // Update task with resolved tag ID
    } else if (tag_id.IsNull() && tag_name.empty()) {
      task->return_code_.store(1); // Neither tag ID nor tag name provided
      return;
    }

    // Step 2: Find the tag by ID
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr == nullptr) {
      task->return_code_.store(1); // Tag not found by ID
      return;
    }

    // Step 3: Delete all blobs in this tag using client AsyncDelBlob to
    // properly clean up blocks
    // Collect all blob names first by scanning tag_blob_name_to_info_
    std::string tag_prefix = std::to_string(tag_id.major_) + "." +
                             std::to_string(tag_id.minor_) + ".";
    std::vector<std::string> blob_names_to_delete;
    tag_blob_name_to_info_.for_each(
        [&tag_prefix, &blob_names_to_delete](const std::string &compound_key,
                                             const BlobInfo &blob_info) {
          if (compound_key.compare(0, tag_prefix.length(), tag_prefix) == 0) {
            blob_names_to_delete.push_back(blob_info.blob_name_);
          }
        });

    // Process blobs in batches to limit concurrent async tasks
    constexpr size_t kMaxConcurrentDelBlobTasks = 32;
    std::vector<hipc::FullPtr<DelBlobTask>> async_tasks;
    size_t processed_blobs = 0;

    for (size_t i = 0; i < blob_names_to_delete.size();
         i += kMaxConcurrentDelBlobTasks) {
      // Create a batch of async tasks (up to kMaxConcurrentDelBlobTasks)
      async_tasks.clear();
      size_t batch_end =
          std::min(i + kMaxConcurrentDelBlobTasks, blob_names_to_delete.size());

      for (size_t j = i; j < batch_end; ++j) {
        const std::string &blob_name = blob_names_to_delete[j];

        // Call AsyncDelBlob from client
        auto async_task =
            client_.AsyncDelBlob(hipc::MemContext(), tag_id, blob_name);
        async_tasks.push_back(async_task);
      }

      // Wait for all async DelBlob operations in this batch to complete
      for (auto task : async_tasks) {
        task->Wait();

        // Check if DelBlob succeeded
        if (task->return_code_.load() != 0) {
          HILOG(kWarning,
                "DelBlob failed for blob during tag deletion, continuing");
          // Continue with other blobs even if one fails
        }

        // Clean up the task
        CHI_IPC->DelTask(task);
        ++processed_blobs;
      }
    }

    // Step 4: Remove all blob name mappings for this tag (DelBlob should have
    // removed them, but ensure cleanup)
    std::vector<std::string> keys_to_erase;
    tag_blob_name_to_info_.for_each(
        [&tag_prefix, &keys_to_erase](const std::string &compound_key,
                                      const BlobInfo &blob_info) {
          if (compound_key.compare(0, tag_prefix.length(), tag_prefix) == 0) {
            keys_to_erase.push_back(compound_key);
          }
        });
    for (const auto &key : keys_to_erase) {
      tag_blob_name_to_info_.erase(key);
    }

    // Step 5: Remove tag name mapping if it exists
    if (!tag_info_ptr->tag_name_.empty()) {
      tag_name_to_id_.erase(tag_info_ptr->tag_name_);
    }

    // Step 6: Log telemetry and remove tag from tag_id_to_info_ map
    size_t blob_count = processed_blobs;
    size_t total_size = tag_info_ptr->total_size_;

    // Log telemetry for DelTag operation
    auto now = std::chrono::steady_clock::now();
    LogTelemetry(CteOp::kDelTag, 0, total_size, tag_id, now, now);

    tag_id_to_info_.erase(tag_id);

    // Success
    task->return_code_.store(0);
    HILOG(kDebug,
          "DelTag successful: tag_id={},{}, removed {} blobs, total_size={}",
          tag_id.major_, tag_id.minor_, blob_count, total_size);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::GetTagSize(hipc::FullPtr<GetTagSizeTask> task,
                         chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
    return;
  }

  try {
    TagId tag_id = task->tag_id_;

    // Find the tag
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr == nullptr) {
      task->return_code_.store(1); // Tag not found
      task->tag_size_ = 0;
      return;
    }

    // Update timestamp and return the total size
    auto now = std::chrono::steady_clock::now();
    tag_info_ptr->last_read_ = now;

    task->tag_size_ = tag_info_ptr->total_size_;
    task->return_code_.store(0);

    // Log telemetry for GetTagSize operation
    LogTelemetry(CteOp::kGetTagSize, 0, tag_info_ptr->total_size_, tag_id,
                 tag_info_ptr->last_modified_, now);

    HILOG(kDebug, "GetTagSize successful: tag_id={},{}, total_size={}",
          tag_id.major_, tag_id.minor_, task->tag_size_);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
    task->tag_size_ = 0;
  }
}

// Private helper methods
const Config &Runtime::GetConfig() const { return config_; }

void Runtime::UpdateTargetStats(const chi::PoolId &target_id,
                                TargetInfo &target_info) {
  // Get actual statistics from bdev using the GetStats method
  chi::u64 remaining_size;
  chimaera::bdev::PerfMetrics perf_metrics =
      target_info.bdev_client_.GetStats(hipc::MemContext(), remaining_size);

  // Update target info with real performance metrics from bdev
  target_info.perf_metrics_ = perf_metrics;
  target_info.remaining_space_ = remaining_size;

  // Check if this target has a manually configured score - if so, don't
  // overwrite it
  float manual_score = GetManualScoreForTarget(target_info.target_name_);
  if (manual_score >= 0.0f) {
    // Keep the manually configured score, don't auto-calculate
    target_info.target_score_ = manual_score;
  } else {
    // Auto-calculate target score using normalized log bandwidth
    double max_bandwidth =
        std::max(target_info.perf_metrics_.read_bandwidth_mbps_,
                 target_info.perf_metrics_.write_bandwidth_mbps_);
    if (max_bandwidth > 0.0) {
      // Find the maximum bandwidth across all targets for normalization
      double global_max_bandwidth =
          1000.0; // TODO: Calculate actual max from all targets

      // Use logarithmic scaling for target score: log(bandwidth_i) /
      // log(bandwidth_MAX)
      target_info.target_score_ = static_cast<float>(
          std::log(max_bandwidth + 1.0) / std::log(global_max_bandwidth + 1.0));

      // Clamp to [0, 1] range
      target_info.target_score_ =
          std::max(0.0f, std::min(1.0f, target_info.target_score_));
    } else {
      target_info.target_score_ = 0.0f; // No bandwidth, lowest score
    }
  }
}

float Runtime::GetManualScoreForTarget(const std::string &target_name) {
  // Check if the target name matches a configured storage device with manual
  // score
  for (size_t i = 0; i < storage_devices_.size(); ++i) {
    const auto &device = storage_devices_[i];

    // Create the expected target name based on how targets are registered
    std::string expected_target_name = "storage_device_" + std::to_string(i);

    // Also check if target name matches the device path directly
    if (target_name == expected_target_name || target_name == device.path_) {
      return device.score_; // Return configured score (-1.0f if not set)
    }
  }

  return -1.0f; // No manual score configured for this target
}

TagId Runtime::GetOrAssignTagId(const std::string &tag_name,
                                const TagId &preferred_id) {
  size_t tag_lock_index = GetTagLockIndex(tag_name);
  chi::ScopedCoRwWriteLock write_lock(*tag_locks_[tag_lock_index]);

  // Check if tag already exists
  TagId *existing_tag_id_ptr = tag_name_to_id_.find(tag_name);
  if (existing_tag_id_ptr != nullptr) {
    return *existing_tag_id_ptr;
  }

  // Assign new tag ID
  TagId tag_id;
  if ((preferred_id.major_ != 0 || preferred_id.minor_ != 0) &&
      !tag_id_to_info_.contains(preferred_id)) {
    tag_id = preferred_id;
  } else {
    tag_id = GenerateNewTagId();
  }

  // Create tag info
  auto *ipc_manager = CHI_IPC;
  auto *main_allocator = ipc_manager->GetMainAllocator();
  TagInfo tag_info(main_allocator);
  tag_info.tag_name_ = tag_name;
  tag_info.tag_id_ = tag_id;

  // Store mappings
  tag_name_to_id_.insert_or_assign(tag_name, tag_id);
  tag_id_to_info_.insert_or_assign(tag_id, tag_info);

  return tag_id;
}

// GetWorkRemaining implementation (required pure virtual method)
chi::u64 Runtime::GetWorkRemaining() const {
  // Return approximate work remaining (simple implementation)
  // In a real implementation, this would sum tasks across all queues
  return 0; // For now, always return 0 work remaining
}

// Helper methods for lock index calculation
size_t Runtime::GetTargetLockIndex(const chi::PoolId &target_id) const {
  // Use hash of target_id to distribute locks evenly
  std::hash<chi::PoolId> hasher;
  return hasher(target_id) % target_locks_.size();
}

size_t Runtime::GetTagLockIndex(const std::string &tag_name) const {
  // Use same hash function as chi::unordered_map_ll to ensure lock maps to same
  // bucket
  std::hash<std::string> hasher;
  return hasher(tag_name) % tag_locks_.size();
}

size_t Runtime::GetTagLockIndex(const TagId &tag_id) const {
  // Use same hash function as chi::unordered_map_ll for TagId keys
  // std::hash<chi::UniqueId> is defined in types.h
  std::hash<TagId> hasher;
  return hasher(tag_id) % tag_locks_.size();
}

TagId Runtime::GenerateNewTagId() {
  // Get node_id from IPC manager as the major component
  auto *ipc_manager = CHI_IPC;
  chi::u32 node_id = ipc_manager->GetNodeId();

  // Get next minor component from atomic counter
  chi::u32 minor_id = next_tag_id_minor_.fetch_add(1);

  return TagId{node_id, minor_id};
}

// Explicit template instantiations for required template methods
template void Runtime::GetOrCreateTag<CreateParams>(
    hipc::FullPtr<GetOrCreateTagTask<CreateParams>> task, chi::RunContext &ctx);

// Blob management helper functions
BlobInfo *Runtime::CheckBlobExists(const std::string &blob_name,
                                   const TagId &tag_id) {
  // Validate that blob name is provided
  if (blob_name.empty()) {
    return nullptr;
  }

  // Construct composite key for lookup
  std::string composite_key = std::to_string(tag_id.major_) + "." +
                              std::to_string(tag_id.minor_) + "." + blob_name;

  // Acquire read lock ONLY for map lookup
  size_t tag_lock_index = GetTagLockIndex(tag_id);
  chi::ScopedCoRwReadLock tag_lock(*tag_locks_[tag_lock_index]);

  // Search by composite key in tag_blob_name_to_info_
  BlobInfo *blob_info_ptr = tag_blob_name_to_info_.find(composite_key);

  // Return result (lock released automatically at scope exit)
  return blob_info_ptr;
}

BlobInfo *Runtime::CreateNewBlob(const std::string &blob_name,
                                 const TagId &tag_id, float blob_score) {
  // Validate that blob name is provided
  if (blob_name.empty()) {
    return nullptr;
  }

  // Prepare blob info structure BEFORE acquiring lock
  auto *ipc_manager = CHI_IPC;
  auto *main_allocator = ipc_manager->GetMainAllocator();
  BlobInfo new_blob_info(main_allocator);
  new_blob_info.blob_name_ = blob_name;
  new_blob_info.score_ = blob_score;

  // Construct composite key for blob storage
  std::string composite_key = std::to_string(tag_id.major_) + "." +
                              std::to_string(tag_id.minor_) + "." + blob_name;

  // Acquire write lock ONLY for map insertion
  size_t tag_lock_index = GetTagLockIndex(tag_id);
  BlobInfo *blob_info_ptr = nullptr;
  {
    chi::ScopedCoRwWriteLock tag_lock(*tag_locks_[tag_lock_index]);

    // Store blob info directly in tag_blob_name_to_info_
    auto insert_result =
        tag_blob_name_to_info_.insert_or_assign(composite_key, new_blob_info);
    blob_info_ptr = insert_result.second;
  } // Release lock immediately after insertion

  return blob_info_ptr;
}

chi::u32 Runtime::AllocateNewData(BlobInfo &blob_info, chi::u64 offset,
                                  chi::u64 size, float blob_score) {
  HILOG(kDebug, "AllocateNewData");
  // Calculate required additional space
  chi::u64 current_blob_size = blob_info.GetTotalSize();
  chi::u64 required_size = offset + size;

  if (required_size <= current_blob_size) {
    // No additional allocation needed
    return 0;
  }

  chi::u64 additional_size = required_size - current_blob_size;

  // Get all available targets for data placement
  std::vector<TargetInfo> available_targets;
  available_targets.reserve(registered_targets_.size());
  registered_targets_.for_each(
      [&available_targets](const chi::PoolId &target_id,
                           const TargetInfo &target_info) {
        available_targets.push_back(target_info);
      });
  HILOG(kDebug, "AllocateNewData: Ordered targets: {}",
        available_targets.size());
  if (available_targets.empty()) {
    return 1;
  }

  // Create Data Placement Engine based on configuration
  const Config &config = GetConfig();
  std::unique_ptr<DataPlacementEngine> dpe =
      DpeFactory::CreateDpe(config.dpe_.dpe_type_);

  // Select targets using DPE algorithm before allocation loop
  std::vector<TargetInfo> ordered_targets =
      dpe->SelectTargets(available_targets, blob_score, additional_size);

  if (ordered_targets.empty()) {
    return 2;
  }

  // Use for loop to iterate over pre-selected targets in order
  chi::u64 remaining_to_allocate = additional_size;
  for (const auto &selected_target_info : ordered_targets) {
    // Termination condition: exit when no more space to allocate
    if (remaining_to_allocate == 0) {
      break;
    }

    chi::PoolId selected_target_id = selected_target_info.bdev_client_.pool_id_;

    // Find the selected target info for allocation using TargetId
    TargetInfo *target_info = registered_targets_.find(selected_target_id);
    if (target_info == nullptr) {
      continue; // Try next target
    }

    // Calculate how much we can allocate from this target
    chi::u64 allocate_size =
        std::min(remaining_to_allocate, target_info->remaining_space_);

    HILOG(kDebug,
          "Target [{}]: remaining_space={} bytes, allocate_size={} bytes, "
          "remaining_to_allocate={} bytes",
          selected_target_id.ToU64(), target_info->remaining_space_,
          allocate_size, remaining_to_allocate);

    if (allocate_size == 0) {
      // No space available, try next target
      HILOG(kDebug, "No space available, trying next target?");
      continue;
    }

    // Allocate space using bdev client
    chi::u64 allocated_offset;
    if (!AllocateFromTarget(*target_info, allocate_size, allocated_offset)) {
      // Allocation failed, try next target
      continue;
    }

    // Create new block for the allocated space
    BlobBlock new_block(target_info->bdev_client_, target_info->target_query_,
                        allocated_offset, allocate_size);
    blob_info.blocks_.emplace_back(new_block);

    remaining_to_allocate -= allocate_size;
  }

  // Error condition: if we've exhausted all targets but still have remaining
  // space
  if (remaining_to_allocate > 0) {
    return 3;
  }

  return 0; // Success
}

chi::u32 Runtime::ModifyExistingData(const std::vector<BlobBlock> &blocks,
                                     hipc::Pointer data, size_t data_size,
                                     size_t data_offset_in_blob) {
  HILOG(kDebug,
        "ModifyExistingData: blocks={}, data_size={}, data_offset_in_blob={}",
        blocks.size(), data_size, data_offset_in_blob);

  // Step 1: Initially store the remaining_size equal to data_size
  size_t remaining_size = data_size;

  // Vector to store async write tasks for later waiting
  std::vector<hipc::FullPtr<chimaera::bdev::WriteTask>> write_tasks;
  std::vector<size_t> expected_write_sizes;

  // Step 2: Store the offset of the block in the blob. The first block is
  // offset 0
  size_t block_offset_in_blob = 0;

  // Iterate over every block in the blob
  for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
    const BlobBlock &block = blocks[block_idx];
    HILOG(
        kDebug,
        "ModifyExistingData: block[{}] - target_offset={}, size={}, pool_id={}",
        block_idx, block.target_offset_, block.size_,
        block.bdev_client_.pool_id_.ToU64());

    // Step 7: If remaining size is 0, quit the for loop
    if (remaining_size == 0) {
      break;
    }

    // Step 3: Check if the data we are writing is within the range
    // [block_offset_in_blob, block_offset_in_blob + block.size)
    size_t block_end_in_blob = block_offset_in_blob + block.size_;
    size_t data_end_in_blob = data_offset_in_blob + data_size;

    if (data_offset_in_blob < block_end_in_blob &&
        data_end_in_blob > block_offset_in_blob) {
      // Step 4: Clamp the range [data_offset_in_blob, data_offset_in_blob +
      // data_size) to the range [block_offset_in_blob, block_offset_in_blob +
      // block.size)
      size_t write_start_in_blob =
          std::max(data_offset_in_blob, block_offset_in_blob);
      size_t write_end_in_blob = std::min(data_end_in_blob, block_end_in_blob);
      size_t write_size = write_end_in_blob - write_start_in_blob;

      // Calculate offset within the block
      size_t write_start_in_block = write_start_in_blob - block_offset_in_blob;

      // Calculate offset into the data buffer
      size_t data_buffer_offset = write_start_in_blob - data_offset_in_blob;

      HILOG(kDebug,
            "ModifyExistingData: block[{}] - writing write_size={}, "
            "write_start_in_block={}, data_buffer_offset={}",
            block_idx, write_size, write_start_in_block, data_buffer_offset);

      // Step 5: Perform async write on the updated range
      chimaera::bdev::Block bdev_block(
          block.target_offset_ + write_start_in_block, write_size, 0);
      hipc::Pointer data_ptr = data + data_buffer_offset;

      // Wrap single block in ArrayVector for AsyncWrite
      chimaera::bdev::ArrayVector<chimaera::bdev::Block, 16> blocks;
      blocks.push_back(bdev_block);

      chimaera::bdev::Client cte_clientcopy = block.bdev_client_;
      auto write_task =
          cte_clientcopy.AsyncWrite(hipc::MemContext(), block.target_query_,
                                    blocks, data_ptr, write_size);

      write_tasks.push_back(write_task);
      expected_write_sizes.push_back(write_size);

      // Step 6: Subtract the amount of data we have written from the
      // remaining_size
      remaining_size -= write_size;
    }

    // Update block offset for next iteration
    block_offset_in_blob += block.size_;
  }

  // Step 7: Wait for all Async write operations to complete
  HILOG(kDebug,
        "ModifyExistingData: Waiting for {} async write tasks to complete",
        write_tasks.size());
  for (size_t task_idx = 0; task_idx < write_tasks.size(); ++task_idx) {
    auto task = write_tasks[task_idx];
    size_t expected_size = expected_write_sizes[task_idx];

    task->Wait();

    HILOG(kDebug,
          "ModifyExistingData: task[{}] completed - bytes_written={}, "
          "expected={}, status={}",
          task_idx, task->bytes_written_, expected_size,
          (task->bytes_written_ == expected_size ? "SUCCESS" : "FAILED"));

    if (task->bytes_written_ != expected_size) {
      CHI_IPC->DelTask(task);
      HILOG(kError,
            "ModifyExistingData: WRITE FAILED - task[{}] wrote {} bytes, "
            "expected {}",
            task_idx, task->bytes_written_, expected_size);
      return 1;
    }

    CHI_IPC->DelTask(task);
  }

  HILOG(kDebug, "ModifyExistingData: All write tasks completed successfully");
  return 0; // Success
}

chi::u32 Runtime::ReadData(const std::vector<BlobBlock> &blocks,
                           hipc::Pointer data, size_t data_size,
                           size_t data_offset_in_blob) {
  HILOG(kDebug, "ReadData: blocks={}, data_size={}, data_offset_in_blob={}",
        blocks.size(), data_size, data_offset_in_blob);

  // Step 1: Initially store the remaining_size equal to data_size
  size_t remaining_size = data_size;

  // Vector to store async read tasks for later waiting
  std::vector<hipc::FullPtr<chimaera::bdev::ReadTask>> read_tasks;
  std::vector<size_t> expected_read_sizes;

  // Step 2: Store the offset of the block in the blob. The first block is
  // offset 0
  size_t block_offset_in_blob = 0;

  // Iterate over every block in the blob
  for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
    const BlobBlock &block = blocks[block_idx];
    HILOG(kDebug, "ReadData: block[{}] - target_offset={}, size={}, pool_id={}",
          block_idx, block.target_offset_, block.size_,
          block.bdev_client_.pool_id_.ToU64());

    // Step 7: If remaining size is 0, quit the for loop
    if (remaining_size == 0) {
      break;
    }

    // Step 3: Check if the data we are reading is within the range
    // [block_offset_in_blob, block_offset_in_blob + block.size)
    size_t block_end_in_blob = block_offset_in_blob + block.size_;
    size_t data_end_in_blob = data_offset_in_blob + data_size;

    if (data_offset_in_blob < block_end_in_blob &&
        data_end_in_blob > block_offset_in_blob) {
      // Step 4: Clamp the range [data_offset_in_blob, data_offset_in_blob +
      // data_size) to the range [block_offset_in_blob, block_offset_in_blob +
      // block.size)
      size_t read_start_in_blob =
          std::max(data_offset_in_blob, block_offset_in_blob);
      size_t read_end_in_blob = std::min(data_end_in_blob, block_end_in_blob);
      size_t read_size = read_end_in_blob - read_start_in_blob;

      // Calculate offset within the block
      size_t read_start_in_block = read_start_in_blob - block_offset_in_blob;

      // Calculate offset into the data buffer
      size_t data_buffer_offset = read_start_in_blob - data_offset_in_blob;

      HILOG(kDebug,
            "ReadData: block[{}] - reading read_size={}, "
            "read_start_in_block={}, data_buffer_offset={}",
            block_idx, read_size, read_start_in_block, data_buffer_offset);

      // Step 5: Perform async read on the range
      chimaera::bdev::Block bdev_block(
          block.target_offset_ + read_start_in_block, read_size, 0);
      hipc::Pointer data_ptr = data + data_buffer_offset;

      // Wrap single block in ArrayVector for AsyncRead
      chimaera::bdev::ArrayVector<chimaera::bdev::Block, 16> blocks;
      blocks.push_back(bdev_block);

      chimaera::bdev::Client cte_clientcopy = block.bdev_client_;
      auto read_task =
          cte_clientcopy.AsyncRead(hipc::MemContext(), block.target_query_,
                                   blocks, data_ptr, read_size);

      read_tasks.push_back(read_task);
      expected_read_sizes.push_back(read_size);

      // Step 6: Subtract the amount of data we have read from the
      // remaining_size
      remaining_size -= read_size;
    }

    // Update block offset for next iteration
    block_offset_in_blob += block.size_;
  }

  // Step 7: Wait for all Async read operations to complete
  HILOG(kDebug, "ReadData: Waiting for {} async read tasks to complete",
        read_tasks.size());
  for (size_t task_idx = 0; task_idx < read_tasks.size(); ++task_idx) {
    auto task = read_tasks[task_idx];
    size_t expected_size = expected_read_sizes[task_idx];

    task->Wait();

    HILOG(
        kDebug,
        "ReadData: task[{}] completed - bytes_read={}, expected={}, status={}",
        task_idx, task->bytes_read_, expected_size,
        (task->bytes_read_ == expected_size ? "SUCCESS" : "FAILED"));

    // Log first few bytes of data that was read for debugging
    if (task->bytes_read_ > 0) {
      hipc::Pointer read_data_ptr =
          data + (task_idx > 0 ? expected_read_sizes[task_idx - 1] : 0);
      char *read_data =
          CHI_IPC->GetMainAllocator()->Convert<char>(read_data_ptr);
      std::string data_preview = "data=[";
      for (size_t i = 0; i < std::min(static_cast<size_t>(task->bytes_read_),
                                      static_cast<size_t>(16));
           ++i) {
        if (i > 0)
          data_preview += ",";
        data_preview +=
            std::to_string(static_cast<unsigned char>(read_data[i]));
      }
      if (task->bytes_read_ > 16)
        data_preview += ",...";
      data_preview += "]";
      HILOG(kDebug, "ReadData: task[{}] - {}", task_idx, data_preview);
    }

    if (task->bytes_read_ != expected_size) {
      CHI_IPC->DelTask(task);
      HILOG(kError,
            "ReadData: READ FAILED - task[{}] read {} bytes, expected {}",
            task_idx, task->bytes_read_, expected_size);
      return 1;
    }

    CHI_IPC->DelTask(task);
  }

  HILOG(kDebug, "ReadData: All read tasks completed successfully");
  return 0; // Success
}

// Block management helper functions

bool Runtime::AllocateFromTarget(TargetInfo &target_info, chi::u64 size,
                                 chi::u64 &allocated_offset) {
  // Check if target has sufficient space
  if (target_info.remaining_space_ < size) {
    return false;
  }

  try {
    // Use bdev client AllocateBlocks method to get actual offset
    std::vector<chimaera::bdev::Block> allocated_blocks =
        target_info.bdev_client_.AllocateBlocks(
            hipc::MemContext(), target_info.target_query_, size);

    // Check if we got any blocks
    if (allocated_blocks.empty()) {
      return false;
    }

    // Use the first block (for single allocation case)
    chimaera::bdev::Block allocated_block = allocated_blocks[0];
    allocated_offset = allocated_block.offset_;

    // Update remaining space
    target_info.remaining_space_ -= size;
    // HILOG(kInfo,
    //       "Allocated from target {}: offset={}, size={} remaining_space={}",
    //       target_info.target_name_, allocated_offset, size,
    //       target_info.remaining_space_);

    return true;
  } catch (const std::exception &e) {
    // Allocation failed
    return false;
  }
}

chi::u32 Runtime::FreeAllBlobBlocks(BlobInfo &blob_info) {
  // Map: PoolId -> (target_query, vector<Block>)
  std::unordered_map<chi::PoolId, std::pair<chi::PoolQuery,
                                            std::vector<chimaera::bdev::Block>>>
      blocks_by_pool;

  // Group blocks by PoolId
  for (const auto &blob_block : blob_info.blocks_) {
    chi::PoolId pool_id = blob_block.bdev_client_.pool_id_;
    chimaera::bdev::Block block;
    block.offset_ = blob_block.target_offset_;
    block.size_ = blob_block.size_;
    block.block_type_ = 0; // Default block type

    // Store target_query with blocks for this pool
    if (blocks_by_pool.find(pool_id) == blocks_by_pool.end()) {
      blocks_by_pool[pool_id] = std::make_pair(
          blob_block.target_query_, std::vector<chimaera::bdev::Block>());
    }
    blocks_by_pool[pool_id].second.push_back(block);
  }

  // Call FreeBlocks once per PoolId
  for (const auto &pool_entry : blocks_by_pool) {
    const chi::PoolId &pool_id = pool_entry.first;
    const chi::PoolQuery &target_query = pool_entry.second.first;
    const std::vector<chimaera::bdev::Block> &blocks = pool_entry.second.second;
    // Get bdev client for this pool from first blob block
    chimaera::bdev::Client bdev_client(pool_id);
    chi::u32 free_result =
        bdev_client.FreeBlocks(hipc::MemContext(), target_query, blocks);
    if (free_result != 0) {
      HILOG(kWarning, "Failed to free blocks from pool {}", pool_id.major_);
    }
  }

  // Clear all blocks
  blob_info.blocks_.clear();
  return 0;
}

void Runtime::LogTelemetry(CteOp op, size_t off, size_t size,
                           const TagId &tag_id, const Timestamp &mod_time,
                           const Timestamp &read_time) {
  // Increment atomic counter and get current logical time
  std::uint64_t logical_time = telemetry_counter_.fetch_add(1) + 1;

  // Create telemetry entry with logical time and enqueue it
  CteTelemetry telemetry_entry(op, off, size, tag_id, mod_time, read_time,
                               logical_time);

  // Circular queue automatically overwrites oldest entries when full
  telemetry_log_.push(telemetry_entry);
}

size_t Runtime::GetTelemetryQueueSize() { return telemetry_log_.GetSize(); }

size_t Runtime::GetTelemetryEntries(std::vector<CteTelemetry> &entries,
                                    size_t max_entries) {
  entries.clear();
  size_t queue_size = telemetry_log_.GetSize();
  size_t entries_to_read = std::min(max_entries, queue_size);

  entries.reserve(entries_to_read);

  // Read entries by popping and re-pushing them (since peek may not be
  // available)
  std::vector<CteTelemetry> temp_entries;
  temp_entries.reserve(entries_to_read);

  // Pop entries temporarily
  for (size_t i = 0; i < entries_to_read; ++i) {
    CteTelemetry entry;
    auto token = telemetry_log_.pop(entry);
    if (!token.IsNull()) {
      temp_entries.push_back(entry);
    } else {
      break; // Queue is empty
    }
  }

  // Re-push entries back to queue (in reverse order to maintain order)
  for (auto it = temp_entries.rbegin(); it != temp_entries.rend(); ++it) {
    telemetry_log_.push(*it);
  }

  // Copy to output vector
  entries = temp_entries;
  return entries.size();
}

void Runtime::PollTelemetryLog(hipc::FullPtr<PollTelemetryLogTask> task,
                               chi::RunContext &ctx) {
  try {
    std::uint64_t minimum_logical_time = task->minimum_logical_time_;

    // Get telemetry entries with logical time filtering
    std::vector<CteTelemetry> all_entries;
    size_t retrieved_count = GetTelemetryEntries(all_entries, 1000);

    // Filter entries by minimum logical time
    task->entries_.clear();
    std::uint64_t max_logical_time = minimum_logical_time;

    for (const auto &entry : all_entries) {
      if (entry.logical_time_ >= minimum_logical_time) {
        task->entries_.emplace_back(entry);
        max_logical_time = std::max(max_logical_time, entry.logical_time_);
      }
    }

    task->last_logical_time_ = max_logical_time;
    task->return_code_.store(0);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
    task->last_logical_time_ = 0;
  }
  (void)ctx;
}

void Runtime::GetBlobScore(hipc::FullPtr<GetBlobScoreTask> task,
                           chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_.store(1);
      return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    if (blob_info_ptr == nullptr) {
      task->return_code_.store(1); // Blob not found
      return;
    }

    // Step 2: Return the blob score
    task->score_ = blob_info_ptr->score_;

    // Step 3: Update timestamps and log telemetry
    auto now = std::chrono::steady_clock::now();
    blob_info_ptr->last_read_ = now;

    // No specific telemetry enum for GetBlobScore, using GetBlob as closest
    // match
    LogTelemetry(CteOp::kGetBlob, 0, 0, tag_id, blob_info_ptr->last_modified_,
                 now);

    // Success
    task->return_code_.store(0);
    HILOG(kDebug, "GetBlobScore successful: name={}, score={}", blob_name,
          blob_info_ptr->score_);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::GetBlobSize(hipc::FullPtr<GetBlobSizeTask> task,
                          chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_.store(1);
      return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      task->return_code_.store(1); // Blob not found
      return;
    }

    // Step 2: Calculate and return the blob size
    task->size_ = blob_info_ptr->GetTotalSize();

    // Step 3: Update timestamps and log telemetry
    auto now = std::chrono::steady_clock::now();
    blob_info_ptr->last_read_ = now;

    // No specific telemetry enum for GetBlobSize, using GetBlob as closest
    // match
    LogTelemetry(CteOp::kGetBlob, 0, 0, tag_id, blob_info_ptr->last_modified_,
                 now);

    // Success
    task->return_code_.store(0);
    HILOG(kDebug, "GetBlobSize successful: name={}, size={}", blob_name,
          task->size_);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
  }
}

void Runtime::GetContainedBlobs(hipc::FullPtr<GetContainedBlobsTask> task,
                                chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
    return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;

    // Validate tag exists
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr == nullptr) {
      task->return_code_.store(1); // Tag not found
      return;
    }

    // Clear output vector
    task->blob_names_.clear();

    // Construct prefix for this tag's blobs
    std::string prefix = std::to_string(tag_id.major_) + "." +
                         std::to_string(tag_id.minor_) + ".";

    // Iterate through tag_blob_name_to_info_ and filter by prefix
    tag_blob_name_to_info_.for_each(
        [&prefix, &task](const std::string &composite_key,
                         const BlobInfo &blob_info) {
          // Check if composite key starts with the tag prefix
          if (composite_key.rfind(prefix, 0) == 0) {
            // Extract blob name (everything after the prefix)
            std::string blob_name = composite_key.substr(prefix.length());
            task->blob_names_.emplace_back(blob_name.c_str());
          }
        });

    // Success
    task->return_code_.store(0);

    // Log telemetry for this operation
    LogTelemetry(CteOp::kGetOrCreateTag, task->blob_names_.size(), 0, tag_id,
                 std::chrono::steady_clock::now(),
                 std::chrono::steady_clock::now());

    HILOG(kDebug, "GetContainedBlobs successful: tag_id={},{}, found {} blobs",
          tag_id.major_, tag_id.minor_, task->blob_names_.size());

  } catch (const std::exception &e) {
    task->return_code_.store(1); // Error during operation
    HILOG(kError, "GetContainedBlobs failed: {}", e.what());
  }
}

void Runtime::TagQuery(hipc::FullPtr<TagQueryTask> task, chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
    return;
  }

  try {
    std::string tag_regex = task->tag_regex_.str();

    // Create regex pattern
    std::regex pattern(tag_regex);

    // Collect matching tags (name + id)
    std::vector<std::pair<std::string, TagId>> matching_tags;
    tag_name_to_id_.for_each(
        [&pattern, &matching_tags](const std::string &tag_name,
                                   const TagId &tag_id) {
          if (std::regex_match(tag_name, pattern)) {
            matching_tags.emplace_back(tag_name, tag_id);
          }
        });

    // Total matched tags (summed across replicas during Aggregate)
    task->total_tags_matched_ = matching_tags.size();

    // Build results: just tag names matching the query. Respect max_tags_ if non-zero.
    task->results_.clear();
    for (const auto &tn : matching_tags) {
      if (task->max_tags_ != 0 && task->results_.size() >= task->max_tags_) {
        break;
      }
      const std::string &tag_name = tn.first;
      task->results_.emplace_back(tag_name.c_str());
    }

    // Success
    task->return_code_.store(0);
    HILOG(kDebug, "TagQuery successful: pattern={}, found {} tags",
          tag_regex, matching_tags.size());

  } catch (const std::exception &e) {
    task->return_code_.store(1);
    HILOG(kError, "TagQuery failed: {}", e.what());
  }
}

void Runtime::BlobQuery(hipc::FullPtr<BlobQueryTask> task,
                        chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
    return;
  }

  try {
    std::string tag_regex = task->tag_regex_.str();
    std::string blob_regex = task->blob_regex_.str();

    // Create regex patterns
    std::regex tag_pattern(tag_regex);
    std::regex blob_pattern(blob_regex);

    // Find matching tag IDs and names
    std::vector<std::pair<std::string, TagId>> matching_tags;
    tag_name_to_id_.for_each(
        [&tag_pattern, &matching_tags](const std::string &tag_name,
                                      const TagId &tag_id) {
          if (std::regex_match(tag_name, tag_pattern)) {
            matching_tags.emplace_back(tag_name, tag_id);
          }
        });

    // Build results: pairs of (tag_name, blob_name) for matching blobs.
    // Also compute total_blobs_matched_.
    task->results_.clear();
    task->total_blobs_matched_ = 0;

    for (const auto &tn : matching_tags) {
      const std::string &tag_name = tn.first;
      const TagId &tag_id = tn.second;

      // Construct prefix for this tag's blobs
      std::string prefix = std::to_string(tag_id.major_) + "." +
                           std::to_string(tag_id.minor_) + ".";

      // Iterate and collect matching blobs for this tag
      tag_blob_name_to_info_.for_each(
          [&prefix, &blob_pattern, &tag_name, &task](
              const std::string &composite_key, const BlobInfo &blob_info) {
            (void)blob_info;
            if (composite_key.rfind(prefix, 0) == 0) {
              std::string blob_name = composite_key.substr(prefix.length());
              if (std::regex_match(blob_name, blob_pattern)) {
                // Increase total matched counter (counts all matches)
                task->total_blobs_matched_++;
                // Respect max_blobs_ if set
                if (task->max_blobs_ == 0 || 
                    task->results_.size() < static_cast<size_t>(task->max_blobs_)) {
                  hipc::pair<hipc::string, hipc::string> pair(
                      task->results_.GetAllocator(), tag_name.c_str(), blob_name.c_str());
                  task->results_.emplace_back(std::move(pair));
                }
              }
            }
          });
    }

    // Success
    task->return_code_.store(0);
    HILOG(
        kDebug,
        "BlobQuery successful: tag_pattern={}, blob_pattern={}, found {} blobs total",
        tag_regex, blob_regex, task->total_blobs_matched_);

  } catch (const std::exception &e) {
    task->return_code_.store(1);
    HILOG(kError, "BlobQuery failed: {}", e.what());
  }
}

// ==============================================================================
// Helper Functions for Dynamic Scheduling
// ==============================================================================

chi::PoolQuery Runtime::HashBlobToContainer(const TagId &tag_id,
                                            const std::string &blob_name) {
  // Compute hash from tag_id and blob_name
  std::hash<std::string> string_hasher;
  std::hash<chi::u32> u32_hasher;

  // Combine tag_id major, minor, and blob_name into a single hash
  chi::u32 hash_value = u32_hasher(tag_id.major_);
  hash_value ^= u32_hasher(tag_id.minor_) + 0x9e3779b9 + (hash_value << 6) +
                (hash_value >> 2);
  hash_value ^= static_cast<chi::u32>(string_hasher(blob_name)) + 0x9e3779b9 +
                (hash_value << 6) + (hash_value >> 2);

  return chi::PoolQuery::DirectHash(hash_value);
}

} // namespace wrp_cte::core

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(wrp_cte::core::Runtime)