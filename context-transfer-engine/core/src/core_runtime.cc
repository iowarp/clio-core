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

#include <chimaera/admin/admin_client.h>
#include <wrp_cte/core/core_config.h>
#include <wrp_cte/core/core_dpe.h>
#include <wrp_cte/core/core_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "chimaera/worker.h"
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/timer.h"

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
    HLOG(kWarning, "Invalid capacity format: {}", capacity_str);
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
    HLOG(kWarning, "Unknown capacity suffix: {}", suffix);
    return static_cast<chi::u64>(value);
  }

  return static_cast<chi::u64>(value * multiplier);
}

chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task,
                                chi::RunContext &ctx) {
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

  // Get IPC manager for later use
  auto *ipc_manager = CHI_IPC;

  // Initialize telemetry ring buffer using unique_ptr with HSHM_MALLOC
  telemetry_log_ = std::make_unique<
      hipc::circular_mpsc_ring_buffer<CteTelemetry, hipc::MallocAllocator>>(
      HSHM_MALLOC, kTelemetryRingSize);

  // Initialize atomic counters
  next_tag_id_minor_ = 1;
  telemetry_counter_ = 0;

  // Initialize WAL vectors (will be opened later if metadata_log_path is set)
  blob_txn_logs_.clear();
  tag_txn_logs_.clear();

  // Get configuration from params (loaded from pool_config.config_ via
  // LoadConfig)
  HLOG(kDebug, "CTE Create: About to call GetParams(), do_compose_={}",
       task->do_compose_);
  auto params = task->GetParams();
  config_ = params.config_;
  HLOG(kDebug,
       "CTE Create: GetParams() returned, storage devices in config: {}",
       config_.storage_.devices_.size());

  // Configuration is now loaded from compose pool_config via
  // CreateParams::LoadConfig()

  // Store storage configuration in runtime
  storage_devices_ = config_.storage_.devices_;
  HLOG(kDebug, "CTE Create: Copied storage devices to runtime, count: {}",
       storage_devices_.size());

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

    HLOG(kDebug,
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

        // Generate unique bdev_id: base major 512 and (1 + device index), minor
        // is container hash
        chi::PoolId bdev_id(512, 1 + static_cast<chi::u32>(device_idx));

        // Call RegisterTarget using client member variable with target_query
        // and bdev_id
        HLOG(kDebug,
             "Registering target ({}): {} ({}, {} bytes) on node {} with "
             "bdev_id=({},{})",
             client_.pool_id_, target_path, device.bdev_type_, capacity_bytes,
             container_hash, bdev_id.major_, bdev_id.minor_);
        auto reg_task = client_.AsyncRegisterTarget(
            target_path, bdev_type, capacity_bytes, target_query, bdev_id);
        co_await reg_task;
        chi::u32 result = reg_task->GetReturnCode();

        if (result == 0) {
          HLOG(kDebug, "  - Registered target: {} ({}, {} bytes) on node {}",
               target_path, device.bdev_type_, capacity_bytes, container_hash);
        } else {
          HLOG(kWarning,
               "  - Failed to register target {} on node {} (error code: {})",
               target_path, container_hash, result);
        }
      }
    }
  } else {
    HLOG(kWarning, "Warning: No storage devices configured");
  }

  // Queue management has been removed - queues are now managed by Chimaera
  // runtime Local queues (kTargetManagementQueue, kTagManagementQueue,
  // kBlobOperationsQueue, kStatsQueue) are no longer created explicitly

  HLOG(kInfo,
       "CTE Core container created and initialized for pool: {} (ID: {})",
       pool_name_, task->new_pool_id_);

  HLOG(kInfo,
       "Configuration: neighborhood={}, poll_period_ms={}, "
       "stat_targets_period_ms={}",
       config_.targets_.neighborhood_, config_.targets_.poll_period_ms_,
       config_.performance_.stat_targets_period_ms_);

  // If this is a restart, restore metadata from the persistent log
  if (is_restart_) {
    RestoreMetadataFromLog();
    ReplayTransactionLogs();
  }

  // Open WAL files if metadata_log_path is configured
  if (!config_.performance_.metadata_log_path_.empty()) {
    chi::u32 num_workers = std::max(
        CHI_WORK_ORCHESTRATOR->GetTotalWorkerCount(), (chi::u32)1);
    chi::u64 per_worker_capacity = std::max(
        config_.performance_.transaction_log_capacity_bytes_ / num_workers,
        (chi::u64)4096);
    blob_txn_logs_.resize(num_workers);
    tag_txn_logs_.resize(num_workers);
    for (chi::u32 i = 0; i < num_workers; ++i) {
      blob_txn_logs_[i] = std::make_unique<TransactionLog>();
      blob_txn_logs_[i]->Open(
          config_.performance_.metadata_log_path_ + ".blob." +
              std::to_string(i),
          per_worker_capacity);
      tag_txn_logs_[i] = std::make_unique<TransactionLog>();
      tag_txn_logs_[i]->Open(
          config_.performance_.metadata_log_path_ + ".tag." +
              std::to_string(i),
          per_worker_capacity);
    }
    HLOG(kInfo, "WAL: Opened {} blob and {} tag transaction logs",
         num_workers, num_workers);
  }

  // Start periodic StatTargets task to keep target stats updated
  chi::u32 stat_period_ms = config_.performance_.stat_targets_period_ms_;
  if (stat_period_ms > 0) {
    HLOG(kInfo, "Starting periodic StatTargets task with period {} ms",
         stat_period_ms);
    client_.AsyncStatTargets(chi::PoolQuery::Local(), stat_period_ms);
  }

  // Spawn periodic FlushMetadata if metadata_log_path is configured and period
  // > 0
  if (!config_.performance_.metadata_log_path_.empty() &&
      config_.performance_.flush_metadata_period_ms_ > 0) {
    client_.AsyncFlushMetadata(
        chi::PoolQuery::Local(),
        config_.performance_.flush_metadata_period_ms_ * 1000.0);
  }

  // Spawn periodic FlushData if configured
  if (config_.performance_.flush_data_period_ms_ > 0) {
    client_.AsyncFlushData(chi::PoolQuery::Local(),
                           config_.performance_.flush_data_min_persistence_,
                           config_.performance_.flush_data_period_ms_ * 1000.0);
  }
  co_return;
}

chi::TaskResume Runtime::Destroy(hipc::FullPtr<DestroyTask> task,
                                 chi::RunContext &ctx) {
  try {
    // Close WAL files before clearing data structures
    for (auto &log : blob_txn_logs_) { if (log) log->Close(); }
    blob_txn_logs_.clear();
    for (auto &log : tag_txn_logs_) { if (log) log->Close(); }
    tag_txn_logs_.clear();

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
  co_return;
}

chi::TaskResume Runtime::RegisterTarget(hipc::FullPtr<RegisterTargetTask> task,
                                        chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
  }

  try {
    std::string target_name = task->target_name_.str();
    chimaera::bdev::BdevType bdev_type = task->bdev_type_;
    chi::u64 total_size = task->total_size_;
    chi::PoolId bdev_pool_id = task->bdev_id_;
    HLOG(kDebug, "Registering target ({}): {} ({} bytes) with bdev_id=({},{})",
         client_.pool_id_, target_name, total_size, bdev_pool_id.major_,
         bdev_pool_id.minor_);

    // Create bdev client and container first to get the TargetId (pool_id)
    chimaera::bdev::Client bdev_client;
    std::string bdev_pool_name =
        target_name;  // Use target_name as the bdev pool name

    HLOG(kDebug, "Creating bdev with pool ID: major={}, minor={}",
         bdev_pool_id.major_, bdev_pool_id.minor_);

    // Create the bdev container using the client
    chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
    HLOG(kDebug,
         "RegisterTarget: Creating bdev with custom_pool_id=({},{}), "
         "target_name={}",
         bdev_pool_id.major_, bdev_pool_id.minor_, target_name);
    auto create_task = bdev_client.AsyncCreate(
        pool_query, target_name, bdev_pool_id, bdev_type, total_size);
    co_await create_task;
    HLOG(kDebug,
         "RegisterTarget: After create, create_task->new_pool_id_=({},{}), "
         "create_task->return_code_={}",
         create_task->new_pool_id_.major_, create_task->new_pool_id_.minor_,
         create_task->return_code_.load());
    bdev_client.pool_id_ = create_task->new_pool_id_;
    bdev_client.return_code_ = create_task->return_code_;
    HLOG(kDebug,
         "RegisterTarget: After assignment, bdev_client.pool_id_=({},{})",
         bdev_client.pool_id_.major_, bdev_client.pool_id_.minor_);

    // Check if creation was successful
    if (bdev_client.return_code_ != 0) {
      HLOG(kError, "Failed to create bdev container {} : {}", target_name,
           bdev_client.return_code_);
      task->return_code_ = 1;
      co_return;
    }

    // Get the TargetId (bdev_client's pool_id) for indexing
    chi::PoolId target_id = bdev_client.pool_id_;

    // Check if target is already registered using TargetId
    size_t lock_index = GetTargetLockIndex(target_id);
    {
      chi::ScopedCoRwReadLock read_lock(*target_locks_[lock_index]);
      TargetInfo *existing_target = registered_targets_.find(target_id);
      if (existing_target != nullptr) {
        co_return;
      }
    }

    // Get actual statistics from bdev using AsyncGetStats method
    chi::u64 remaining_size;
    auto stats_task = bdev_client.AsyncGetStats();
    co_await stats_task;
    chimaera::bdev::PerfMetrics perf_metrics = stats_task->metrics_;
    remaining_size = stats_task->remaining_size_;

    // Create target info with bdev client and performance stats
    // Use default constructor (allocator not used in struct)
    TargetInfo target_info;
    HLOG(kDebug, "RegisterTarget: Before move, bdev_client.pool_id_=({},{})",
         bdev_client.pool_id_.major_, bdev_client.pool_id_.minor_);
    target_info.target_name_ = target_name;
    target_info.bdev_pool_name_ = bdev_pool_name;
    target_info.bdev_client_ = std::move(bdev_client);
    HLOG(
        kDebug,
        "RegisterTarget: After move, target_info.bdev_client_.pool_id_=({},{})",
        target_info.bdev_client_.pool_id_.major_,
        target_info.bdev_client_.pool_id_.minor_);
    target_info.target_query_ =
        task->target_query_;  // Store target query for bdev API calls
    target_info.bytes_read_ = 0;
    target_info.bytes_written_ = 0;
    target_info.ops_read_ = 0;
    target_info.ops_written_ = 0;
    // Check if this target has a manually configured score from storage device
    // config
    float manual_score = GetManualScoreForTarget(target_name);
    if (manual_score >= 0.0f) {
      target_info.target_score_ = manual_score;  // Use configured manual score
      HLOG(kDebug, "Target '{}' using manual score: {:.2f}", target_name,
           manual_score);
    } else {
      target_info.target_score_ =
          0.0f;  // Will be calculated based on performance metrics
    }
    target_info.remaining_space_ =
        total_size;  // Use actual remaining space from bdev
    target_info.perf_metrics_ =
        perf_metrics;  // Store the entire PerfMetrics structure
    target_info.persistence_level_ = GetPersistenceLevelForTarget(target_name);

    // Register the target using TargetId as key
    {
      chi::ScopedCoRwWriteLock write_lock(*target_locks_[lock_index]);
      registered_targets_.insert_or_assign(target_id, target_info);
      target_name_to_id_.insert_or_assign(
          target_name,
          target_id);  // Maintain reverse lookup
    }

    task->return_code_ = 0;  // Success
    HLOG(kDebug,
         "Target '{}' registered with ID (major={}, minor={}) - bdev pool: {} "
         "(type={}, path={}, "
         "size={}, remaining={})",
         target_name, target_id.major_, target_id.minor_, bdev_pool_name,
         static_cast<int>(bdev_type), target_name, total_size, remaining_size);
    HLOG(kDebug,
         "  Initial statistics: read_bw={} MB/s, write_bw={} MB/s, "
         "avg_latency={} μs, iops={}",
         perf_metrics.read_bandwidth_mbps_, perf_metrics.write_bandwidth_mbps_,
         (target_info.perf_metrics_.read_latency_us_ +
          target_info.perf_metrics_.write_latency_us_) /
             2.0,
         perf_metrics.iops_);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::UnregisterTarget(
    hipc::FullPtr<UnregisterTargetTask> task, chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
  }

  try {
    std::string target_name = task->target_name_.str();

    // Look up TargetId from target_name
    chi::PoolId *target_id_ptr = target_name_to_id_.find(target_name);
    if (target_id_ptr == nullptr) {
      task->return_code_ = 1;
      co_return;
    }

    const chi::PoolId &target_id = *target_id_ptr;

    // Check if target exists and remove it (don't destroy bdev container)
    size_t lock_index = GetTargetLockIndex(target_id);
    {
      chi::ScopedCoRwWriteLock write_lock(*target_locks_[lock_index]);
      if (!registered_targets_.contains(target_id)) {
        task->return_code_ = 1;
        co_return;
      }

      registered_targets_.erase(target_id);
      target_name_to_id_.erase(target_name);  // Remove reverse lookup
    }

    task->return_code_ = 0;  // Success
    HLOG(kDebug, "Target '{}' unregistered", target_name);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::ListTargets(hipc::FullPtr<ListTargetsTask> task,
                                     chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
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
          task->target_names_.push_back(target_info.target_name_);
        });

    task->return_code_ = 0;  // Success

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::StatTargets(hipc::FullPtr<StatTargetsTask> task,
                                     chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
  }

  try {
    // Update performance stats for all registered targets
    // Use a single lock based on hash of operation type for stats
    size_t lock_index =
        std::hash<std::string>{}("stat_targets") % target_locks_.size();
    chi::ScopedCoRwReadLock read_lock(*target_locks_[lock_index]);

    // Collect all target IDs first (can't co_await inside for_each lambda)
    std::vector<chi::PoolId> target_ids;
    registered_targets_.for_each(
        [&target_ids](const chi::PoolId &target_id, TargetInfo &target_info) {
          (void)target_info;
          target_ids.push_back(target_id);
        });

    // Now iterate and co_await each UpdateTargetStats call
    for (const auto &target_id : target_ids) {
      TargetInfo *target_info = registered_targets_.find(target_id);
      if (target_info != nullptr) {
        co_await UpdateTargetStats(target_id, *target_info);
      }
    }

    task->return_code_ = 0;  // Success

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

template <typename CreateParamsT>
chi::TaskResume Runtime::GetOrCreateTag(
    hipc::FullPtr<GetOrCreateTagTask<CreateParamsT>> task,
    chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
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
    co_return;
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
      task->return_code_ = 0;
      co_return;
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

    task->return_code_ = 0;  // Success

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::GetTargetInfo(hipc::FullPtr<GetTargetInfoTask> task,
                                       chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Local();
    co_return;
  }

  try {
    std::string target_name = task->target_name_.str();

    // Look up target by name
    chi::PoolId *target_id_ptr = target_name_to_id_.find(target_name);
    if (target_id_ptr == nullptr) {
      task->return_code_ = 1;  // Target not found
      co_return;
    }

    chi::PoolId target_id = *target_id_ptr;
    size_t lock_index = GetTargetLockIndex(target_id);
    chi::ScopedCoRwReadLock read_lock(*target_locks_[lock_index]);

    // Find target in registered_targets_
    auto target_ptr = registered_targets_.find(target_id);
    if (!target_ptr) {
      task->return_code_ = 2;  // Target not in registered list
      co_return;
    }

    // Copy target information to task output
    task->target_score_ = target_ptr->target_score_;
    task->remaining_space_ = target_ptr->remaining_space_;
    task->bytes_read_ = target_ptr->bytes_read_;
    task->bytes_written_ = target_ptr->bytes_written_;
    task->ops_read_ = target_ptr->ops_read_;
    task->ops_written_ = target_ptr->ops_written_;

    task->return_code_ = 0;  // Success

  } catch (const std::exception &e) {
    task->return_code_ = 3;
  }
  co_return;
}

chi::TaskResume Runtime::PutBlob(hipc::FullPtr<PutBlobTask> task,
                                 chi::RunContext &ctx) {
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    co_return;
  }

  try {
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    chi::u64 offset = task->offset_;
    chi::u64 size = task->size_;
    hipc::ShmPtr<> blob_data = task->blob_data_;
    float blob_score = task->score_;

    // Validate inputs
    if (size == 0) {
      task->return_code_ = 2;
      co_return;
    }
    if (blob_data.IsNull()) {
      task->return_code_ = 3;
      co_return;
    }
    if (blob_name.empty()) {
      task->return_code_ = 4;
      co_return;
    }

    // Check if blob exists and resolve score
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    bool blob_found = (blob_info_ptr != nullptr);
    if (blob_score < 0.0f) {
      blob_score = blob_found ? blob_info_ptr->score_ : 1.0f;
    }

    // Step 1: ClearBlob — free blocks if full replacement
    chi::u64 old_blob_size = 0;
    if (blob_found) {
      bool cleared = false;
      co_await ClearBlob(*blob_info_ptr, blob_score, offset, size, cleared);
      if (cleared) {
        // WAL: log blob clear
        if (!blob_txn_logs_.empty()) {
          chi::u32 wid = CHI_CUR_WORKER->GetWorkerStats().worker_id_;
          TxnClearBlob txn;
          txn.tag_major_ = tag_id.major_;
          txn.tag_minor_ = tag_id.minor_;
          txn.blob_name_ = blob_name;
          blob_txn_logs_[wid % blob_txn_logs_.size()]->Log(
              TxnType::kClearBlob, txn);
        }
      } else {
        old_blob_size = blob_info_ptr->GetTotalSize();
      }
    }

    // Create blob metadata if new
    if (!blob_found) {
      blob_info_ptr = CreateNewBlob(blob_name, tag_id, blob_score);
      if (!blob_info_ptr) {
        task->return_code_ = 5;
        co_return;
      }
    }

    // Step 2: ExtendBlob — allocate new blocks if needed
    chi::u32 alloc_result = 0;
    co_await ExtendBlob(*blob_info_ptr, offset, size, blob_score, alloc_result,
                        task->context_.min_persistence_level_);
    if (alloc_result != 0) {
      task->return_code_ = 10 + alloc_result;
      co_return;
    }

    // WAL: log all current blocks (full replacement semantics)
    if (!blob_txn_logs_.empty() && !blob_info_ptr->blocks_.empty()) {
      chi::u32 wid = CHI_CUR_WORKER->GetWorkerStats().worker_id_;
      TxnExtendBlob txn;
      txn.tag_major_ = tag_id.major_;
      txn.tag_minor_ = tag_id.minor_;
      txn.blob_name_ = blob_name;
      for (const auto &blk : blob_info_ptr->blocks_) {
        TxnExtendBlobBlock tb;
        tb.bdev_major_ = blk.bdev_client_.pool_id_.major_;
        tb.bdev_minor_ = blk.bdev_client_.pool_id_.minor_;
        tb.target_query_ = blk.target_query_;
        tb.target_offset_ = blk.target_offset_;
        tb.size_ = blk.size_;
        txn.new_blocks_.push_back(tb);
      }
      blob_txn_logs_[wid % blob_txn_logs_.size()]->Log(
          TxnType::kExtendBlob, txn);
    }

    // Step 3: ModifyExistingData — write data to blocks
    chi::u32 write_result = 0;
    co_await ModifyExistingData(blob_info_ptr->blocks_, blob_data, size, offset,
                                write_result);
    if (write_result != 0) {
      task->return_code_ = 20 + write_result;
      co_return;
    }

    // Update compression metadata
    Context &context = task->context_;
    blob_info_ptr->compress_lib_ = context.compress_lib_;
    blob_info_ptr->compress_preset_ = context.compress_preset_;
    blob_info_ptr->trace_key_ = context.trace_key_;

    // Update tag size
    chi::u64 new_blob_size = blob_info_ptr->GetTotalSize();
    chi::i64 size_change = static_cast<chi::i64>(new_blob_size) -
                           static_cast<chi::i64>(old_blob_size);
    auto now = std::chrono::steady_clock::now();
    blob_info_ptr->last_modified_ = now;
    blob_info_ptr->score_ = blob_score;
    {
      size_t tag_lock_index = GetTagLockIndex(tag_id);
      chi::ScopedCoRwReadLock tag_lock(*tag_locks_[tag_lock_index]);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr) {
        tag_info_ptr->last_modified_ = now;
        if (size_change >= 0) {
          tag_info_ptr->total_size_.fetch_add(static_cast<size_t>(size_change));
        } else {
          HLOG(kError, "Size should not decrease");
          task->return_code_ = 1;
          co_return;
        }
      }
    }

    LogTelemetry(CteOp::kPutBlob, offset, size, tag_id, now,
                 blob_info_ptr->last_read_);
    task->return_code_ = 0;
  } catch (const std::exception &e) {
    HLOG(kError, "PutBlob failed with exception: {}", e.what());
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::GetBlob(hipc::FullPtr<GetBlobTask> task,
                                 chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    co_return;
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
      task->return_code_ = 1;
      co_return;
    }

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;
      co_return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    // If blob doesn't exist, error
    if (blob_info_ptr == nullptr) {
      task->return_code_ = 1;
      co_return;
    }

    // Use the pre-provided data pointer from the task
    hipc::ShmPtr<> blob_data_ptr = task->blob_data_;

    // Step 2: Read data from blob blocks (no lock held during I/O)
    chi::u32 read_result = 0;
    co_await ReadData(blob_info_ptr->blocks_, blob_data_ptr, size, offset,
                      read_result);
    if (read_result != 0) {
      task->return_code_ = read_result;
      co_return;
    }

    // Step 3: Update timestamp (no lock needed - just updating values, not
    // modifying map structure)
    auto now = std::chrono::steady_clock::now();
    size_t num_blocks = 0;
    blob_info_ptr->last_read_ = now;
    num_blocks = blob_info_ptr->blocks_.size();

    // Log telemetry and success messages after releasing lock
    LogTelemetry(CteOp::kGetBlob, offset, size, tag_id,
                 blob_info_ptr->last_modified_, now);

    task->return_code_ = 0;

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::ReorganizeBlob(hipc::FullPtr<ReorganizeBlobTask> task,
                                        chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    co_return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    float new_score = task->new_score_;

    // Validate inputs
    if (blob_name.empty()) {
      task->return_code_ = 1;  // Invalid input - empty blob name
      co_return;
    }

    if (new_score < 0.0f || new_score > 1.0f) {
      task->return_code_ = 1;  // Invalid score range
      co_return;
    }

    // Get configuration for score difference threshold
    const Config &config = GetConfig();
    float score_difference_threshold =
        config.performance_.score_difference_threshold_;

    // Step 1: Get blob info directly from table
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      task->return_code_ = 3;  // Blob not found
      co_return;
    }

    // Step 2: Check if score needs updating
    float current_score = blob_info_ptr->score_;
    float score_diff = std::abs(new_score - current_score);
    HLOG(kDebug,
         "SCORE CHECK: blob={}, current={}, new={}, diff={}, threshold={}",
         blob_name, current_score, new_score, score_diff,
         score_difference_threshold);

    if (score_diff < score_difference_threshold) {
      // Score difference too small, no reorganization needed
      task->return_code_ = 0;
      HLOG(kDebug,
           "ReorganizeBlob: score difference below threshold, skipping");
      co_return;
    }

    // Step 3: Get blob info (don't update score yet - PutBlob will handle it)
    BlobInfo &blob_info = *blob_info_ptr;

    HLOG(kDebug, "ReorganizeBlob: blob={}, current_score={}, target_score={}",
         blob_name, blob_info.score_, new_score);

    // Step 4: Get blob size from blob_info
    chi::u64 blob_size = blob_info.GetTotalSize();

    if (blob_size == 0) {
      // Empty blob, no data to reorganize
      task->return_code_ = 0;
      co_return;
    }

    // Step 5: Allocate buffer for blob data
    auto *ipc_manager = CHI_IPC;
    hipc::FullPtr<char> blob_data_buffer =
        ipc_manager->AllocateBuffer(blob_size);
    if (blob_data_buffer.IsNull()) {
      HLOG(kError, "Failed to allocate buffer for blob during reorganization");
      task->return_code_ = 5;  // Buffer allocation failed
      co_return;
    }

    // Step 6: Get blob data
    auto get_task =
        client_.AsyncGetBlob(tag_id, blob_name, 0, blob_size, 0,
                             blob_data_buffer.shm_.template Cast<void>());
    co_await get_task;

    if (get_task->return_code_ != 0u) {
      HLOG(kWarning, "Failed to get blob data during reorganization");
      task->return_code_ = 6;  // Get blob failed
      co_return;
    }

    // Step 7: Put blob with new score (data reorganization)
    HLOG(kDebug,
         "ReorganizeBlob calling AsyncPutBlob for blob={}, new_score={}",
         blob_name, new_score);
    auto put_task = client_.AsyncPutBlob(
        tag_id, blob_name, 0, blob_size,
        blob_data_buffer.shm_.template Cast<void>(), new_score, Context(), 0);
    co_await put_task;

    if (put_task->return_code_ != 0) {
      HLOG(kWarning, "Failed to put blob during reorganization");
      task->return_code_ = 7;  // Put blob failed
      co_return;
    }

    // Success
    task->return_code_ = 0;

    HLOG(kDebug,
         "ReorganizeBlob completed: tag_id={},{}, blob={}, new_score={}",
         tag_id.major_, tag_id.minor_, blob_name, new_score);

  } catch (const std::exception &e) {
    HLOG(kError, "ReorganizeBlob failed: {}", e.what());
    task->return_code_ = 1;  // Error during reorganization
  }
  co_return;
}

chi::TaskResume Runtime::DelBlob(hipc::FullPtr<DelBlobTask> task,
                                 chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    co_return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;
      co_return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    if (blob_info_ptr == nullptr) {
      task->return_code_ = 1;  // Blob not found
      co_return;
    }

    // Step 2: Get blob size before deletion for tag size accounting
    chi::u64 blob_size = blob_info_ptr->GetTotalSize();

    // Step 2.5: Free all blocks back to their targets before removing blob
    chi::u32 free_result = 0;
    co_await FreeAllBlobBlocks(*blob_info_ptr, free_result);
    if (free_result != 0) {
      HLOG(kWarning,
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
        tag_info_ptr->total_size_ = 0;  // Clamp to 0 if we would underflow
      }
    }

    // Step 5: Remove blob from tag_blob_name_to_info_ map
    std::string compound_key = std::to_string(tag_id.major_) + "." +
                               std::to_string(tag_id.minor_) + "." + blob_name;
    tag_blob_name_to_info_.erase(compound_key);

    // Step 6: Log telemetry for DelBlob operation
    auto now = std::chrono::steady_clock::now();
    LogTelemetry(CteOp::kDelBlob, 0, blob_size, tag_id, now, now);

    // WAL: log blob deletion
    if (!blob_txn_logs_.empty()) {
      chi::u32 wid = CHI_CUR_WORKER->GetWorkerStats().worker_id_;
      TxnDelBlob txn;
      txn.tag_major_ = tag_id.major_;
      txn.tag_minor_ = tag_id.minor_;
      txn.blob_name_ = blob_name;
      blob_txn_logs_[wid % blob_txn_logs_.size()]->Log(
          TxnType::kDelBlob, txn);
    }

    // Success
    task->return_code_ = 0;
    HLOG(kDebug, "DelBlob successful: name={}, blob_size={}", blob_name,
         blob_size);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::DelTag(hipc::FullPtr<DelTagTask> task,
                                chi::RunContext &ctx) {
  try {
    TagId tag_id = task->tag_id_;
    std::string tag_name = task->tag_name_.str();

    // Step 1: Resolve tag ID if tag name was provided instead
    if (tag_id.IsNull() && !tag_name.empty()) {
      // Look up tag ID by name
      TagId *found_tag_id_ptr = tag_name_to_id_.find(tag_name);
      if (found_tag_id_ptr == nullptr) {
        task->return_code_ = 1;  // Tag not found by name
        co_return;
      }
      tag_id = *found_tag_id_ptr;
      task->tag_id_ = tag_id;  // Update task with resolved tag ID
    } else if (tag_id.IsNull() && tag_name.empty()) {
      task->return_code_ = 1;  // Neither tag ID nor tag name provided
      co_return;
    }

    // Step 2: Find the tag by ID
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr == nullptr) {
      task->return_code_ = 1;  // Tag not found by ID
      co_return;
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
    std::vector<chi::Future<DelBlobTask>> async_tasks;
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
        auto async_task = client_.AsyncDelBlob(tag_id, blob_name);
        async_tasks.push_back(async_task);
      }

      // Wait for all async DelBlob operations in this batch to complete
      for (auto task : async_tasks) {
        co_await task;

        // Check if DelBlob succeeded
        if (task->return_code_ != 0) {
          HLOG(kWarning,
               "DelBlob failed for blob during tag deletion, continuing");
          // Continue with other blobs even if one fails
        }

        // Clean up the task
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

    // WAL: log tag deletion
    if (!tag_txn_logs_.empty()) {
      chi::u32 wid = CHI_CUR_WORKER->GetWorkerStats().worker_id_;
      TxnDelTag txn;
      txn.tag_name_ = tag_info_ptr->tag_name_;
      txn.tag_major_ = tag_id.major_;
      txn.tag_minor_ = tag_id.minor_;
      tag_txn_logs_[wid % tag_txn_logs_.size()]->Log(TxnType::kDelTag, txn);
    }

    tag_id_to_info_.erase(tag_id);

    // Success
    task->return_code_ = 0;
    HLOG(kDebug,
         "DelTag successful: tag_id={},{}, removed {} blobs, total_size={}",
         tag_id.major_, tag_id.minor_, blob_count, total_size);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::GetTagSize(hipc::FullPtr<GetTagSizeTask> task,
                                    chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
    co_return;
  }

  try {
    TagId tag_id = task->tag_id_;

    // Find the tag
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr == nullptr) {
      task->return_code_ = 1;  // Tag not found
      task->tag_size_ = 0;
      co_return;
    }

    // Update timestamp and return the total size
    auto now = std::chrono::steady_clock::now();
    tag_info_ptr->last_read_ = now;

    task->tag_size_ = tag_info_ptr->total_size_;
    task->return_code_ = 0;

    // Log telemetry for GetTagSize operation
    LogTelemetry(CteOp::kGetTagSize, 0, tag_info_ptr->total_size_, tag_id,
                 tag_info_ptr->last_modified_, now);

    HLOG(kDebug, "GetTagSize successful: tag_id={},{}, total_size={}",
         tag_id.major_, tag_id.minor_, task->tag_size_);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
    task->tag_size_ = 0;
  }
  co_return;
}

// Private helper methods
const Config &Runtime::GetConfig() const { return config_; }

chi::TaskResume Runtime::UpdateTargetStats(const chi::PoolId &target_id,
                                           TargetInfo &target_info) {
  // Get actual statistics from bdev using the AsyncGetStats method
  chi::u64 remaining_size;
  auto stats_task = target_info.bdev_client_.AsyncGetStats();
  co_await stats_task;
  chimaera::bdev::PerfMetrics perf_metrics = stats_task->metrics_;
  remaining_size = stats_task->remaining_size_;

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
          1000.0;  // TODO: Calculate actual max from all targets

      // Use logarithmic scaling for target score: log(bandwidth_i) /
      // log(bandwidth_MAX)
      target_info.target_score_ = static_cast<float>(
          std::log(max_bandwidth + 1.0) / std::log(global_max_bandwidth + 1.0));

      // Clamp to [0, 1] range
      target_info.target_score_ =
          std::max(0.0f, std::min(1.0f, target_info.target_score_));
    } else {
      target_info.target_score_ = 0.0f;  // No bandwidth, lowest score
    }
  }
  co_return;
}

float Runtime::GetManualScoreForTarget(const std::string &target_name) {
  // Check if the target name matches a configured storage device with manual
  // score
  for (size_t i = 0; i < storage_devices_.size(); ++i) {
    const auto &device = storage_devices_[i];

    // Create the expected target name based on how targets are registered
    std::string expected_target_name = "storage_device_" + std::to_string(i);

    // Check if target name matches:
    // 1. Exact match with "storage_device_N"
    // 2. Exact match with device path
    // 3. Starts with device path (to handle "_nodeX" suffix added during
    // registration)
    if (target_name == expected_target_name || target_name == device.path_ ||
        (target_name.rfind(device.path_, 0) == 0 &&
         (target_name.size() == device.path_.size() ||
          target_name[device.path_.size()] == '_'))) {
      HLOG(kDebug,
           "GetManualScoreForTarget: target '{}' matched device path '{}', "
           "score={}",
           target_name, device.path_, device.score_);
      return device.score_;  // Return configured score (-1.0f if not set)
    }
  }

  HLOG(kDebug,
       "GetManualScoreForTarget: target '{}' has no manual score configured",
       target_name);
  return -1.0f;  // No manual score configured for this target
}

chimaera::bdev::PersistenceLevel Runtime::GetPersistenceLevelForTarget(
    const std::string &target_name) {
  for (size_t i = 0; i < storage_devices_.size(); ++i) {
    const auto &device = storage_devices_[i];
    std::string expected_target_name = "storage_device_" + std::to_string(i);
    if (target_name == expected_target_name || target_name == device.path_ ||
        (target_name.rfind(device.path_, 0) == 0 &&
         (target_name.size() == device.path_.size() ||
          target_name[device.path_.size()] == '_'))) {
      // Convert string persistence level to enum
      if (device.persistence_level_ == "temporary") {
        return chimaera::bdev::PersistenceLevel::kTemporaryNonVolatile;
      } else if (device.persistence_level_ == "long_term") {
        return chimaera::bdev::PersistenceLevel::kLongTerm;
      }
      return chimaera::bdev::PersistenceLevel::kVolatile;
    }
  }
  return chimaera::bdev::PersistenceLevel::kVolatile;
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

  // Create tag info (use default constructor, allocator not used in struct)
  TagInfo tag_info;
  tag_info.tag_name_ = tag_name;
  tag_info.tag_id_ = tag_id;

  // Store mappings
  tag_name_to_id_.insert_or_assign(tag_name, tag_id);
  tag_id_to_info_.insert_or_assign(tag_id, tag_info);

  // WAL: log tag creation
  if (!tag_txn_logs_.empty()) {
    chi::u32 wid = CHI_CUR_WORKER->GetWorkerStats().worker_id_;
    TxnCreateTag txn;
    txn.tag_name_ = tag_name;
    txn.tag_major_ = tag_id.major_;
    txn.tag_minor_ = tag_id.minor_;
    tag_txn_logs_[wid % tag_txn_logs_.size()]->Log(TxnType::kCreateTag, txn);
  }

  return tag_id;
}

chi::TaskResume Runtime::FlushMetadata(hipc::FullPtr<FlushMetadataTask> task,
                                       chi::RunContext &ctx) {
  task->entries_flushed_ = 0;

  const std::string &log_path = config_.performance_.metadata_log_path_;
  if (log_path.empty()) {
    task->return_code_ = 0;
    (void)ctx;
    co_return;
  }

  try {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(log_path).parent_path());

    std::ofstream ofs(log_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      HLOG(kError, "FlushMetadata: Failed to open log file: {}", log_path);
      task->return_code_ = 1;
      co_return;
    }

    // Write TagInfo entries (entry_type 0)
    tag_id_to_info_.for_each([&](const TagId &id, const TagInfo &info) {
      uint8_t entry_type = 0;
      uint32_t name_len = static_cast<uint32_t>(info.tag_name_.size());
      chi::u64 total_size = info.total_size_.load();
      ofs.write(reinterpret_cast<const char *>(&entry_type),
                sizeof(entry_type));
      ofs.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
      ofs.write(info.tag_name_.data(), name_len);
      ofs.write(reinterpret_cast<const char *>(&id), sizeof(id));
      ofs.write(reinterpret_cast<const char *>(&total_size),
                sizeof(total_size));
      task->entries_flushed_++;
    });

    // Write BlobInfo entries (entry_type 1)
    tag_blob_name_to_info_.for_each([&](const std::string &key,
                                        const BlobInfo &blob_info) {
      uint8_t entry_type = 1;
      uint32_t key_len = static_cast<uint32_t>(key.size());
      uint32_t blob_name_len =
          static_cast<uint32_t>(blob_info.blob_name_.size());
      float score = blob_info.score_;
      int32_t compress_lib = blob_info.compress_lib_;
      int32_t compress_preset = blob_info.compress_preset_;
      chi::u64 trace_key = blob_info.trace_key_;
      uint32_t num_blocks = static_cast<uint32_t>(blob_info.blocks_.size());

      ofs.write(reinterpret_cast<const char *>(&entry_type),
                sizeof(entry_type));
      ofs.write(reinterpret_cast<const char *>(&key_len), sizeof(key_len));
      ofs.write(key.data(), key_len);
      ofs.write(reinterpret_cast<const char *>(&blob_name_len),
                sizeof(blob_name_len));
      ofs.write(blob_info.blob_name_.data(), blob_name_len);
      ofs.write(reinterpret_cast<const char *>(&score), sizeof(score));
      ofs.write(reinterpret_cast<const char *>(&compress_lib),
                sizeof(compress_lib));
      ofs.write(reinterpret_cast<const char *>(&compress_preset),
                sizeof(compress_preset));
      ofs.write(reinterpret_cast<const char *>(&trace_key), sizeof(trace_key));
      ofs.write(reinterpret_cast<const char *>(&num_blocks),
                sizeof(num_blocks));

      // Write per-block data
      for (const auto &block : blob_info.blocks_) {
        chi::u32 bdev_major = block.bdev_client_.pool_id_.major_;
        chi::u32 bdev_minor = block.bdev_client_.pool_id_.minor_;
        ofs.write(reinterpret_cast<const char *>(&bdev_major),
                  sizeof(bdev_major));
        ofs.write(reinterpret_cast<const char *>(&bdev_minor),
                  sizeof(bdev_minor));

        // Write target_query as raw bytes (POD-like struct)
        ofs.write(reinterpret_cast<const char *>(&block.target_query_),
                  sizeof(chi::PoolQuery));

        chi::u64 offset = block.target_offset_;
        chi::u64 size = block.size_;
        ofs.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
        ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
      }
      task->entries_flushed_++;
    });

    ofs.close();

    // WAL: sync and compact transaction logs after snapshot
    if (!blob_txn_logs_.empty()) {
      chi::u64 total_wal_size = 0;
      for (auto &log : blob_txn_logs_) {
        if (log) { log->Sync(); total_wal_size += log->Size(); }
      }
      for (auto &log : tag_txn_logs_) {
        if (log) { log->Sync(); total_wal_size += log->Size(); }
      }
      if (total_wal_size >
          config_.performance_.transaction_log_capacity_bytes_) {
        for (auto &log : blob_txn_logs_) { if (log) log->Truncate(); }
        for (auto &log : tag_txn_logs_) { if (log) log->Truncate(); }
        HLOG(kDebug, "FlushMetadata: Truncated WAL files (was {} bytes)",
             total_wal_size);
      }
    }

    task->return_code_ = 0;
    HLOG(kDebug, "FlushMetadata: Flushed {} entries to {}",
         task->entries_flushed_, log_path);
  } catch (const std::exception &e) {
    HLOG(kError, "FlushMetadata: Exception: {}", e.what());
    task->return_code_ = 99;
  }
  (void)ctx;
  co_return;
}

chi::TaskResume Runtime::FlushData(hipc::FullPtr<FlushDataTask> task,
                                   chi::RunContext &ctx) {
  task->bytes_flushed_ = 0;
  task->blobs_flushed_ = 0;

  int target_level = task->target_persistence_level_;

  // Find non-volatile targets that meet the persistence level requirement
  std::vector<chi::PoolId> nonvolatile_targets;
  registered_targets_.for_each(
      [&](const chi::PoolId &id, const TargetInfo &info) {
        if (static_cast<int>(info.persistence_level_) >= target_level) {
          nonvolatile_targets.push_back(id);
        }
      });

  if (nonvolatile_targets.empty()) {
    HLOG(kDebug, "FlushData: No non-volatile targets available at level >= {}",
         target_level);
    task->return_code_ = 0;
    (void)ctx;
    co_return;
  }

  // Collect blobs that have volatile blocks
  struct FlushEntry {
    std::string composite_key;
    std::string blob_name;
    TagId tag_id;
    chi::u64 total_size;
    float score;
  };
  std::vector<FlushEntry> blobs_to_flush;

  tag_blob_name_to_info_.for_each([&](const std::string &key,
                                      const BlobInfo &blob_info) {
    if (blob_info.blocks_.empty()) return;

    bool has_volatile_blocks = false;
    for (const auto &block : blob_info.blocks_) {
      chi::PoolId pool_id = block.bdev_client_.pool_id_;
      TargetInfo *tinfo = registered_targets_.find(pool_id);
      if (tinfo && static_cast<int>(tinfo->persistence_level_) < target_level) {
        has_volatile_blocks = true;
        break;
      }
    }

    if (has_volatile_blocks) {
      FlushEntry entry;
      entry.composite_key = key;
      entry.blob_name = blob_info.blob_name_;
      entry.total_size = blob_info.GetTotalSize();
      entry.score = blob_info.score_;

      // Parse tag_id from composite key: "major.minor.blob_name"
      size_t first_dot = key.find('.');
      size_t second_dot = key.find('.', first_dot + 1);
      if (first_dot != std::string::npos && second_dot != std::string::npos) {
        entry.tag_id.major_ =
            static_cast<chi::u32>(std::stoul(key.substr(0, first_dot)));
        entry.tag_id.minor_ = static_cast<chi::u32>(
            std::stoul(key.substr(first_dot + 1, second_dot - first_dot - 1)));
      }

      blobs_to_flush.push_back(std::move(entry));
    }
  });

  HLOG(kDebug, "FlushData: Found {} blobs with volatile blocks to flush",
       blobs_to_flush.size());

  // Flush each blob: read data, free volatile blocks, re-put with persistence
  for (const auto &entry : blobs_to_flush) {
    BlobInfo *blob_info_ptr = tag_blob_name_to_info_.find(entry.composite_key);
    if (!blob_info_ptr || blob_info_ptr->blocks_.empty()) continue;

    chi::u64 total_size = entry.total_size;
    if (total_size == 0) continue;

    // Step 1: Allocate buffer and read data from current blocks
    auto *ipc_manager = CHI_IPC;
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(total_size);
    if (buffer.IsNull()) {
      HLOG(kError,
           "FlushData: Failed to allocate buffer of size {} for blob {}",
           total_size, entry.blob_name);
      continue;
    }

    hipc::ShmPtr<> shm_ptr(buffer.shm_);
    chi::u32 read_error = 0;
    co_await ReadData(blob_info_ptr->blocks_, shm_ptr, total_size, 0,
                      read_error);
    if (read_error != 0) {
      HLOG(kError, "FlushData: Failed to read blob data for {}",
           entry.blob_name);
      ipc_manager->FreeBuffer(buffer);
      continue;
    }

    // Step 2: Free only volatile blocks
    std::vector<BlobBlock> nonvolatile_blocks;
    std::unordered_map<
        chi::PoolId,
        std::pair<chi::PoolQuery, std::vector<chimaera::bdev::Block>>>
        volatile_blocks_by_pool;

    for (const auto &block : blob_info_ptr->blocks_) {
      chi::PoolId pool_id = block.bdev_client_.pool_id_;
      TargetInfo *tinfo = registered_targets_.find(pool_id);
      if (tinfo && static_cast<int>(tinfo->persistence_level_) < target_level) {
        // Volatile block - collect for freeing
        chimaera::bdev::Block bdev_block;
        bdev_block.offset_ = block.target_offset_;
        bdev_block.size_ = block.size_;
        bdev_block.block_type_ = 0;
        if (volatile_blocks_by_pool.find(pool_id) ==
            volatile_blocks_by_pool.end()) {
          volatile_blocks_by_pool[pool_id] = std::make_pair(
              block.target_query_, std::vector<chimaera::bdev::Block>());
        }
        volatile_blocks_by_pool[pool_id].second.push_back(bdev_block);
      } else {
        // Nonvolatile block - keep
        nonvolatile_blocks.push_back(block);
      }
    }

    // Free volatile blocks from bdevs
    for (const auto &pool_entry : volatile_blocks_by_pool) {
      const chi::PoolId &pool_id = pool_entry.first;
      const chi::PoolQuery &target_query = pool_entry.second.first;
      const std::vector<chimaera::bdev::Block> &blocks =
          pool_entry.second.second;

      chi::u64 bytes_freed = 0;
      for (const auto &block : blocks) {
        bytes_freed += block.size_;
      }

      chimaera::bdev::Client bdev_client(pool_id);
      auto free_task = bdev_client.AsyncFreeBlocks(target_query, blocks);
      co_await free_task;
      if (free_task->GetReturnCode() == 0) {
        TargetInfo *target_info = registered_targets_.find(pool_id);
        if (target_info) {
          target_info->remaining_space_ += bytes_freed;
        }
      }
    }

    // Update blob blocks to only keep nonvolatile blocks
    blob_info_ptr->blocks_ = nonvolatile_blocks;

    // Step 3: Re-put data using AsyncPutBlob with persistence context
    Context flush_ctx;
    flush_ctx.min_persistence_level_ = target_level;
    auto put_task =
        client_.AsyncPutBlob(entry.tag_id, entry.blob_name, 0, total_size,
                             shm_ptr, entry.score, flush_ctx);
    co_await put_task;

    if (put_task->GetReturnCode() != 0) {
      HLOG(kError, "FlushData: PutBlob failed for blob {} (error {})",
           entry.blob_name, put_task->GetReturnCode());
    } else {
      task->blobs_flushed_++;
      task->bytes_flushed_ += total_size;
    }

    ipc_manager->FreeBuffer(buffer);
  }

  task->return_code_ = 0;
  HLOG(kDebug, "FlushData: Flushed {} blobs ({} bytes)", task->blobs_flushed_,
       task->bytes_flushed_);
  (void)ctx;
  co_return;
}

void Runtime::RestoreMetadataFromLog() {
  const std::string &log_path = config_.performance_.metadata_log_path_;
  if (log_path.empty()) {
    HLOG(kInfo, "RestoreMetadataFromLog: No metadata log path configured");
    return;
  }

  namespace fs = std::filesystem;
  if (!fs::exists(log_path)) {
    HLOG(kInfo, "RestoreMetadataFromLog: No log file found at {}", log_path);
    return;
  }

  std::ifstream ifs(log_path, std::ios::binary);
  if (!ifs.is_open()) {
    HLOG(kError, "RestoreMetadataFromLog: Failed to open log file: {}",
         log_path);
    return;
  }

  chi::u32 max_minor = 0;
  chi::u32 tags_restored = 0;
  chi::u32 blobs_restored = 0;

  while (ifs.peek() != EOF) {
    uint8_t entry_type;
    ifs.read(reinterpret_cast<char *>(&entry_type), sizeof(entry_type));
    if (!ifs.good()) break;

    if (entry_type == 0) {
      // TagInfo entry
      uint32_t name_len;
      ifs.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
      std::string tag_name(name_len, '\0');
      ifs.read(tag_name.data(), name_len);
      TagId tag_id;
      ifs.read(reinterpret_cast<char *>(&tag_id), sizeof(tag_id));
      chi::u64 total_size;
      ifs.read(reinterpret_cast<char *>(&total_size), sizeof(total_size));

      if (!ifs.good()) break;

      // Populate maps
      tag_name_to_id_.insert_or_assign(tag_name, tag_id);
      TagInfo tag_info(tag_name, tag_id);
      tag_info.total_size_.store(total_size);
      tag_id_to_info_.insert_or_assign(tag_id, tag_info);

      if (tag_id.minor_ >= max_minor) {
        max_minor = tag_id.minor_ + 1;
      }
      tags_restored++;

    } else if (entry_type == 1) {
      // BlobInfo entry
      uint32_t key_len;
      ifs.read(reinterpret_cast<char *>(&key_len), sizeof(key_len));
      std::string composite_key(key_len, '\0');
      ifs.read(composite_key.data(), key_len);

      uint32_t blob_name_len;
      ifs.read(reinterpret_cast<char *>(&blob_name_len), sizeof(blob_name_len));
      std::string blob_name(blob_name_len, '\0');
      ifs.read(blob_name.data(), blob_name_len);

      float score;
      ifs.read(reinterpret_cast<char *>(&score), sizeof(score));
      int32_t compress_lib;
      ifs.read(reinterpret_cast<char *>(&compress_lib), sizeof(compress_lib));
      int32_t compress_preset;
      ifs.read(reinterpret_cast<char *>(&compress_preset),
               sizeof(compress_preset));
      chi::u64 trace_key;
      ifs.read(reinterpret_cast<char *>(&trace_key), sizeof(trace_key));
      uint32_t num_blocks;
      ifs.read(reinterpret_cast<char *>(&num_blocks), sizeof(num_blocks));

      if (!ifs.good()) break;

      BlobInfo blob_info;
      blob_info.blob_name_ = blob_name;
      blob_info.score_ = score;
      blob_info.compress_lib_ = compress_lib;
      blob_info.compress_preset_ = compress_preset;
      blob_info.trace_key_ = trace_key;

      // Read per-block data
      for (uint32_t i = 0; i < num_blocks; i++) {
        chi::u32 bdev_major, bdev_minor;
        ifs.read(reinterpret_cast<char *>(&bdev_major), sizeof(bdev_major));
        ifs.read(reinterpret_cast<char *>(&bdev_minor), sizeof(bdev_minor));

        // Read target_query as raw bytes (POD-like struct)
        chi::PoolQuery target_query;
        ifs.read(reinterpret_cast<char *>(&target_query),
                 sizeof(chi::PoolQuery));

        chi::u64 offset, size;
        ifs.read(reinterpret_cast<char *>(&offset), sizeof(offset));
        ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

        if (!ifs.good()) break;

        // Filter by persistence level: skip volatile blocks
        chi::PoolId bdev_pool_id(bdev_major, bdev_minor);
        TargetInfo *tinfo = registered_targets_.find(bdev_pool_id);
        if (tinfo && tinfo->persistence_level_ ==
                         chimaera::bdev::PersistenceLevel::kVolatile) {
          continue;  // Volatile data is lost on restart
        }

        // Reconstruct block
        chimaera::bdev::Client bdev_client(bdev_pool_id);
        BlobBlock block(bdev_client, target_query, offset, size);
        blob_info.blocks_.push_back(block);
      }

      tag_blob_name_to_info_.insert_or_assign(composite_key, blob_info);
      blobs_restored++;

    } else {
      HLOG(kWarning, "RestoreMetadataFromLog: Unknown entry type {}",
           entry_type);
      break;
    }
  }

  ifs.close();

  // Update next_tag_id_minor_ to be past any restored tag IDs
  chi::u32 current_minor = next_tag_id_minor_.load();
  if (max_minor > current_minor) {
    next_tag_id_minor_.store(max_minor);
  }

  HLOG(kInfo, "RestoreMetadataFromLog: Restored {} tags and {} blobs from {}",
       tags_restored, blobs_restored, log_path);
}

void Runtime::ReplayTransactionLogs() {
  const std::string &log_path = config_.performance_.metadata_log_path_;
  if (log_path.empty()) return;

  chi::u32 tags_replayed = 0;
  chi::u32 blobs_replayed = 0;
  chi::u32 max_minor = next_tag_id_minor_.load();

  // Phase 1: Replay all tag logs first (tags must exist before blob ops)
  for (size_t i = 0; ; ++i) {
    std::string tag_log_path = log_path + ".tag." + std::to_string(i);
    if (!std::filesystem::exists(tag_log_path)) break;

    TransactionLog loader;
    loader.Open(tag_log_path, 0);
    auto entries = loader.Load();
    loader.Close();

    for (const auto &[type, payload] : entries) {
      if (type == TxnType::kCreateTag) {
        auto txn = TransactionLog::DeserializeCreateTag(payload);
        TagId tag_id{txn.tag_major_, txn.tag_minor_};
        tag_name_to_id_.insert_or_assign(txn.tag_name_, tag_id);
        TagInfo tag_info(txn.tag_name_, tag_id);
        tag_id_to_info_.insert_or_assign(tag_id, tag_info);
        if (tag_id.minor_ >= max_minor) max_minor = tag_id.minor_ + 1;
        tags_replayed++;
      } else if (type == TxnType::kDelTag) {
        auto txn = TransactionLog::DeserializeDelTag(payload);
        TagId tag_id{txn.tag_major_, txn.tag_minor_};
        // Erase tag name mapping
        tag_name_to_id_.erase(txn.tag_name_);
        // Erase all blobs belonging to this tag
        std::string tag_prefix = std::to_string(tag_id.major_) + "." +
                                 std::to_string(tag_id.minor_) + ".";
        std::vector<std::string> keys_to_erase;
        tag_blob_name_to_info_.for_each(
            [&tag_prefix, &keys_to_erase](const std::string &key,
                                          const BlobInfo &) {
              if (key.compare(0, tag_prefix.length(), tag_prefix) == 0) {
                keys_to_erase.push_back(key);
              }
            });
        for (const auto &key : keys_to_erase) {
          tag_blob_name_to_info_.erase(key);
        }
        tag_id_to_info_.erase(tag_id);
        tags_replayed++;
      }
    }
  }

  // Phase 2: Replay all blob logs
  for (size_t i = 0; ; ++i) {
    std::string blob_log_path = log_path + ".blob." + std::to_string(i);
    if (!std::filesystem::exists(blob_log_path)) break;

    TransactionLog loader;
    loader.Open(blob_log_path, 0);
    auto entries = loader.Load();
    loader.Close();

    for (const auto &[type, payload] : entries) {
      if (type == TxnType::kCreateNewBlob) {
        auto txn = TransactionLog::DeserializeCreateNewBlob(payload);
        TagId tag_id{txn.tag_major_, txn.tag_minor_};
        std::string composite_key = std::to_string(tag_id.major_) + "." +
                                    std::to_string(tag_id.minor_) + "." +
                                    txn.blob_name_;
        BlobInfo blob_info;
        blob_info.blob_name_ = txn.blob_name_;
        blob_info.score_ = txn.score_;
        tag_blob_name_to_info_.insert_or_assign(composite_key, blob_info);
        blobs_replayed++;

      } else if (type == TxnType::kExtendBlob) {
        auto txn = TransactionLog::DeserializeExtendBlob(payload);
        TagId tag_id{txn.tag_major_, txn.tag_minor_};
        std::string composite_key = std::to_string(tag_id.major_) + "." +
                                    std::to_string(tag_id.minor_) + "." +
                                    txn.blob_name_;
        BlobInfo *blob_info_ptr =
            tag_blob_name_to_info_.find(composite_key);
        if (blob_info_ptr) {
          // Replace blocks with replayed blocks (full replacement semantics)
          blob_info_ptr->blocks_.clear();
          for (const auto &tb : txn.new_blocks_) {
            chi::PoolId bdev_pool_id(tb.bdev_major_, tb.bdev_minor_);
            // Filter volatile targets (matching RestoreMetadataFromLog)
            TargetInfo *tinfo = registered_targets_.find(bdev_pool_id);
            if (tinfo &&
                tinfo->persistence_level_ ==
                    chimaera::bdev::PersistenceLevel::kVolatile) {
              continue;
            }
            chimaera::bdev::Client bdev_client(bdev_pool_id);
            BlobBlock block(bdev_client, tb.target_query_,
                            tb.target_offset_, tb.size_);
            blob_info_ptr->blocks_.push_back(block);
          }
        }
        blobs_replayed++;

      } else if (type == TxnType::kClearBlob) {
        auto txn = TransactionLog::DeserializeClearBlob(payload);
        TagId tag_id{txn.tag_major_, txn.tag_minor_};
        std::string composite_key = std::to_string(tag_id.major_) + "." +
                                    std::to_string(tag_id.minor_) + "." +
                                    txn.blob_name_;
        BlobInfo *blob_info_ptr =
            tag_blob_name_to_info_.find(composite_key);
        if (blob_info_ptr) {
          blob_info_ptr->blocks_.clear();
        }
        blobs_replayed++;

      } else if (type == TxnType::kDelBlob) {
        auto txn = TransactionLog::DeserializeDelBlob(payload);
        TagId tag_id{txn.tag_major_, txn.tag_minor_};
        std::string composite_key = std::to_string(tag_id.major_) + "." +
                                    std::to_string(tag_id.minor_) + "." +
                                    txn.blob_name_;
        tag_blob_name_to_info_.erase(composite_key);
        blobs_replayed++;
      }
    }
  }

  // Phase 3: Recompute tag total_size_ from blob blocks
  tag_id_to_info_.for_each([&](const TagId &tag_id, TagInfo &tag_info) {
    chi::u64 total = 0;
    std::string tag_prefix = std::to_string(tag_id.major_) + "." +
                             std::to_string(tag_id.minor_) + ".";
    tag_blob_name_to_info_.for_each(
        [&tag_prefix, &total](const std::string &key,
                              const BlobInfo &blob_info) {
          if (key.compare(0, tag_prefix.length(), tag_prefix) == 0) {
            total += blob_info.GetTotalSize();
          }
        });
    tag_info.total_size_.store(total);
  });

  // Phase 4: Update next_tag_id_minor_
  chi::u32 current_minor = next_tag_id_minor_.load();
  if (max_minor > current_minor) {
    next_tag_id_minor_.store(max_minor);
  }

  HLOG(kInfo,
       "ReplayTransactionLogs: Replayed {} tag ops and {} blob ops",
       tags_replayed, blobs_replayed);
}

// GetWorkRemaining implementation (required pure virtual method)
chi::u64 Runtime::GetWorkRemaining() const {
  // Return approximate work remaining (simple implementation)
  // In a real implementation, this would sum tasks across all queues
  return 0;  // For now, always return 0 work remaining
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
template chi::TaskResume Runtime::GetOrCreateTag<CreateParams>(
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
  // Use default constructor (allocator not used in struct)
  BlobInfo new_blob_info;
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
  }  // Release lock immediately after insertion

  // WAL: log blob creation
  if (!blob_txn_logs_.empty()) {
    chi::u32 wid = CHI_CUR_WORKER->GetWorkerStats().worker_id_;
    TxnCreateNewBlob txn;
    txn.tag_major_ = tag_id.major_;
    txn.tag_minor_ = tag_id.minor_;
    txn.blob_name_ = blob_name;
    txn.score_ = blob_score;
    blob_txn_logs_[wid % blob_txn_logs_.size()]->Log(
        TxnType::kCreateNewBlob, txn);
  }

  return blob_info_ptr;
}

chi::TaskResume Runtime::ExtendBlob(BlobInfo &blob_info, chi::u64 offset,
                                    chi::u64 size, float blob_score,
                                    chi::u32 &error_code,
                                    int min_persistence_level) {
  // Calculate required additional space
  chi::u64 current_blob_size = blob_info.GetTotalSize();
  chi::u64 required_size = offset + size;

  if (required_size <= current_blob_size) {
    // No additional allocation needed
    error_code = 0;
    co_return;
  }

  chi::u64 additional_size = required_size - current_blob_size;

  // Get ALL available targets for data placement (no pre-filtering)
  std::vector<TargetInfo> available_targets;
  available_targets.reserve(registered_targets_.size());
  registered_targets_.for_each(
      [&available_targets](const chi::PoolId &target_id,
                           const TargetInfo &target_info) {
        (void)target_id;
        available_targets.push_back(target_info);
      });
  if (available_targets.empty()) {
    error_code = 1;
    co_return;
  }

  // Create Data Placement Engine based on configuration
  const Config &config = GetConfig();
  std::unique_ptr<DataPlacementEngine> dpe =
      DpeFactory::CreateDpe(config.dpe_.dpe_type_);

  // DPE selects targets from ALL available targets
  std::vector<TargetInfo> ordered_targets =
      dpe->SelectTargets(available_targets, blob_score, additional_size);

  // Filter AFTER DPE by persistence level
  if (min_persistence_level > 0) {
    ordered_targets.erase(
        std::remove_if(ordered_targets.begin(), ordered_targets.end(),
                       [min_persistence_level](const TargetInfo &t) {
                         return static_cast<int>(t.persistence_level_) <
                                min_persistence_level;
                       }),
        ordered_targets.end());
  }

  if (ordered_targets.empty()) {
    error_code = 2;
    co_return;
  }

  // Allocate from pre-selected targets in order
  chi::u64 remaining_to_allocate = additional_size;
  for (const auto &selected_target_info : ordered_targets) {
    if (remaining_to_allocate == 0) {
      break;
    }

    chi::PoolId selected_target_id = selected_target_info.bdev_client_.pool_id_;

    // Find the selected target info for allocation using TargetId
    TargetInfo *target_info = registered_targets_.find(selected_target_id);
    if (target_info == nullptr) {
      continue;
    }

    // Calculate how much we can allocate from this target
    chi::u64 allocate_size =
        std::min(remaining_to_allocate, target_info->remaining_space_);

    if (allocate_size == 0) {
      continue;
    }

    // Allocate space using bdev client
    chi::u64 allocated_offset;
    bool alloc_success = false;
    co_await AllocateFromTarget(*target_info, allocate_size, allocated_offset,
                                alloc_success);
    if (!alloc_success) {
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
    error_code = 3;
    co_return;
  }

  error_code = 0;  // Success
  co_return;
}

chi::TaskResume Runtime::ModifyExistingData(
    const std::vector<BlobBlock> &blocks, hipc::ShmPtr<> data, size_t data_size,
    size_t data_offset_in_blob, chi::u32 &error_code) {
  HLOG(kDebug,
       "ModifyExistingData: blocks={}, data_size={}, data_offset_in_blob={}",
       blocks.size(), data_size, data_offset_in_blob);

  static thread_local size_t mod_count = 0;
  static thread_local double t_setup_ms = 0, t_vec_alloc_ms = 0;
  static thread_local double t_async_send_ms = 0, t_co_await_ms = 0;
  hshm::Timer timer;

  // Step 1: Initially store the remaining_size equal to data_size
  size_t remaining_size = data_size;

  // Vector to store async write tasks for later waiting
  std::vector<chi::Future<chimaera::bdev::WriteTask>> write_tasks;
  std::vector<size_t> expected_write_sizes;

  // Step 2: Store the offset of the block in the blob. The first block is
  // offset 0
  size_t block_offset_in_blob = 0;

  // Iterate over every block in the blob
  for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
    const BlobBlock &block = blocks[block_idx];
    HLOG(
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
      // Step 4: Clamp the range
      timer.Resume();
      size_t write_start_in_blob =
          std::max(data_offset_in_blob, block_offset_in_blob);
      size_t write_end_in_blob = std::min(data_end_in_blob, block_end_in_blob);
      size_t write_size = write_end_in_blob - write_start_in_blob;
      size_t write_start_in_block = write_start_in_blob - block_offset_in_blob;
      size_t data_buffer_offset = write_start_in_blob - data_offset_in_blob;

      chimaera::bdev::Block bdev_block(
          block.target_offset_ + write_start_in_block, write_size, 0);
      hipc::ShmPtr<> data_ptr = data + data_buffer_offset;
      timer.Pause();
      t_setup_ms += timer.GetMsec();
      timer.Reset();

      // Wrap single block in chi::priv::vector for AsyncWrite
      timer.Resume();
      chi::priv::vector<chimaera::bdev::Block> blocks(HSHM_MALLOC);
      blocks.push_back(bdev_block);
      timer.Pause();
      t_vec_alloc_ms += timer.GetMsec();
      timer.Reset();

      // Create and send the async write task
      timer.Resume();
      chimaera::bdev::Client cte_clientcopy = block.bdev_client_;
      auto write_task = cte_clientcopy.AsyncWrite(block.target_query_, blocks,
                                                  data_ptr, write_size);
      write_tasks.push_back(std::move(write_task));
      expected_write_sizes.push_back(write_size);
      timer.Pause();
      t_async_send_ms += timer.GetMsec();
      timer.Reset();

      remaining_size -= write_size;
    }

    // Update block offset for next iteration
    block_offset_in_blob += block.size_;
  }

  // Step 7: Wait for all Async write operations to complete
  timer.Resume();
  for (size_t task_idx = 0; task_idx < write_tasks.size(); ++task_idx) {
    auto &task = write_tasks[task_idx];
    size_t expected_size = expected_write_sizes[task_idx];
    co_await task;
    if (task->bytes_written_ != expected_size) {
      error_code = 1;
      co_return;
    }
  }
  timer.Pause();
  t_co_await_ms += timer.GetMsec();
  timer.Reset();

  ++mod_count;
  if (mod_count % 100 == 0) {
    HLOG(kInfo, "[ModifyExistingData] ops={} setup={:.3f} ms vec_alloc={:.3f} ms "
                "async_send={:.3f} ms co_await={:.3f} ms",
         mod_count, t_setup_ms, t_vec_alloc_ms, t_async_send_ms,
         t_co_await_ms);
    t_setup_ms = t_vec_alloc_ms = t_async_send_ms = t_co_await_ms = 0;
  }

  error_code = 0;  // Success
  co_return;
}

chi::TaskResume Runtime::ReadData(const std::vector<BlobBlock> &blocks,
                                  hipc::ShmPtr<> data, size_t data_size,
                                  size_t data_offset_in_blob,
                                  chi::u32 &error_code) {
  HLOG(kDebug, "ReadData: blocks={}, data_size={}, data_offset_in_blob={}",
       blocks.size(), data_size, data_offset_in_blob);

  // Step 1: Initially store the remaining_size equal to data_size
  size_t remaining_size = data_size;

  // Vector to store async read tasks for later waiting
  std::vector<chi::Future<chimaera::bdev::ReadTask>> read_tasks;
  std::vector<size_t> expected_read_sizes;

  // Step 2: Store the offset of the block in the blob. The first block is
  // offset 0
  size_t block_offset_in_blob = 0;

  // Iterate over every block in the blob
  for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
    const BlobBlock &block = blocks[block_idx];
    HLOG(kDebug, "ReadData: block[{}] - target_offset={}, size={}, pool_id={}",
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

      HLOG(kDebug,
           "ReadData: block[{}] - reading read_size={}, "
           "read_start_in_block={}, data_buffer_offset={}",
           block_idx, read_size, read_start_in_block, data_buffer_offset);

      // Step 5: Perform async read on the range
      chimaera::bdev::Block bdev_block(
          block.target_offset_ + read_start_in_block, read_size, 0);
      hipc::ShmPtr<> data_ptr = data + data_buffer_offset;

      // Wrap single block in chi::priv::vector for AsyncRead
      chi::priv::vector<chimaera::bdev::Block> blocks(HSHM_MALLOC);
      blocks.push_back(bdev_block);

      chimaera::bdev::Client cte_clientcopy = block.bdev_client_;
      auto read_task = cte_clientcopy.AsyncRead(block.target_query_, blocks,
                                                data_ptr, read_size);

      read_tasks.push_back(std::move(read_task));
      expected_read_sizes.push_back(read_size);

      // Step 6: Subtract the amount of data we have read from the
      // remaining_size
      remaining_size -= read_size;
    }

    // Update block offset for next iteration
    block_offset_in_blob += block.size_;
  }

  // Step 7: Wait for all Async read operations to complete
  HLOG(kDebug, "ReadData: Waiting for {} async read tasks to complete",
       read_tasks.size());
  for (size_t task_idx = 0; task_idx < read_tasks.size(); ++task_idx) {
    auto &task = read_tasks[task_idx];
    size_t expected_size = expected_read_sizes[task_idx];

    co_await task;

    HLOG(kDebug,
         "ReadData: task[{}] completed - bytes_read={}, expected={}, status={}",
         task_idx, task->bytes_read_, expected_size,
         (task->bytes_read_ == expected_size ? "SUCCESS" : "FAILED"));

    if (task->bytes_read_ != expected_size) {
      HLOG(kError,
           "ReadData: READ FAILED - task[{}] read {} bytes, expected {}",
           task_idx, task->bytes_read_, expected_size);
      error_code = 1;
      co_return;
    }
  }

  HLOG(kDebug, "ReadData: All read tasks completed successfully");
  error_code = 0;  // Success
  co_return;
}

// Block management helper functions

chi::TaskResume Runtime::AllocateFromTarget(TargetInfo &target_info,
                                            chi::u64 size,
                                            chi::u64 &allocated_offset,
                                            bool &success) {
  HLOG(kDebug,
       "AllocateFromTarget: ENTER - target_name={}, "
       "bdev_client_.pool_id_=({},{}), size={}, remaining_space={}",
       target_info.target_name_, target_info.bdev_client_.pool_id_.major_,
       target_info.bdev_client_.pool_id_.minor_, size,
       target_info.remaining_space_);

  // Check if target has sufficient space
  if (target_info.remaining_space_ < size) {
    HLOG(kDebug,
         "AllocateFromTarget: Insufficient space - remaining={} < size={}",
         target_info.remaining_space_, size);
    success = false;
    co_return;
  }

  try {
    HLOG(
        kDebug,
        "AllocateFromTarget: Calling AsyncAllocateBlocks with pool_id_=({},{})",
        target_info.bdev_client_.pool_id_.major_,
        target_info.bdev_client_.pool_id_.minor_);

    // Use bdev client AsyncAllocateBlocks method to get actual offset
    auto alloc_task = target_info.bdev_client_.AsyncAllocateBlocks(
        target_info.target_query_, size);

    HLOG(kDebug,
         "AllocateFromTarget: AsyncAllocateBlocks returned, IsComplete()={}, "
         "co_awaiting...",
         alloc_task.IsComplete() ? "true" : "false");

    co_await alloc_task;

    HLOG(kDebug,
         "AllocateFromTarget: co_await complete, "
         "alloc_task->blocks_.size()={}, return_code={}",
         alloc_task->blocks_.size(), alloc_task->return_code_.load());

    std::vector<chimaera::bdev::Block> allocated_blocks;
    for (size_t i = 0; i < alloc_task->blocks_.size(); ++i) {
      allocated_blocks.push_back(alloc_task->blocks_[i]);
    }

    // Check if we got any blocks
    if (allocated_blocks.empty()) {
      HLOG(kDebug, "AllocateFromTarget: FAILED - allocated_blocks is empty");
      success = false;
      co_return;
    }

    // Use the first block (for single allocation case)
    chimaera::bdev::Block allocated_block = allocated_blocks[0];
    allocated_offset = allocated_block.offset_;

    // Update remaining space
    target_info.remaining_space_ -= size;
    // HLOG(kInfo,
    //       "Allocated from target {}: offset={}, size={} remaining_space={}",
    //       target_info.target_name_, allocated_offset, size,
    //       target_info.remaining_space_);

    success = true;
    co_return;
  } catch (const std::exception &e) {
    // Allocation failed
    success = false;
    co_return;
  }
}

chi::TaskResume Runtime::ClearBlob(BlobInfo &blob_info, float blob_score,
                                   chi::u64 offset, chi::u64 size,
                                   bool &cleared) {
  cleared = false;
  // Score must be in [0, 1]
  if (blob_score < 0.0f || blob_score > 1.0f) {
    co_return;
  }
  // Must be full-blob replacement
  chi::u64 current_size = blob_info.GetTotalSize();
  if (offset != 0 || size < current_size || current_size == 0) {
    co_return;
  }
  // Free all existing blocks
  chi::u32 free_result = 0;
  co_await FreeAllBlobBlocks(blob_info, free_result);
  if (free_result == 0) {
    cleared = true;
  }
  co_return;
}

chi::TaskResume Runtime::FreeAllBlobBlocks(BlobInfo &blob_info,
                                           chi::u32 &error_code) {
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
    block.block_type_ = 0;  // Default block type

    // Store target_query with blocks for this pool
    if (blocks_by_pool.find(pool_id) == blocks_by_pool.end()) {
      blocks_by_pool[pool_id] = std::make_pair(
          blob_block.target_query_, std::vector<chimaera::bdev::Block>());
    }
    blocks_by_pool[pool_id].second.push_back(block);
  }

  // Call FreeBlocks once per PoolId and update target capacities
  for (const auto &pool_entry : blocks_by_pool) {
    const chi::PoolId &pool_id = pool_entry.first;
    const chi::PoolQuery &target_query = pool_entry.second.first;
    const std::vector<chimaera::bdev::Block> &blocks = pool_entry.second.second;

    // Calculate total bytes to be freed for this pool
    chi::u64 bytes_freed = 0;
    for (const auto &block : blocks) {
      bytes_freed += block.size_;
    }

    // Get bdev client for this pool from first blob block
    chimaera::bdev::Client bdev_client(pool_id);
    auto free_task = bdev_client.AsyncFreeBlocks(target_query, blocks);
    co_await free_task;
    chi::u32 free_result = free_task->GetReturnCode();
    if (free_result != 0) {
      HLOG(kWarning, "Failed to free blocks from pool {}", pool_id.major_);
    } else {
      // Successfully freed blocks - update target's remaining_space_
      size_t lock_index =
          std::hash<std::string>{}("free_blocks") % target_locks_.size();
      chi::ScopedCoRwWriteLock write_lock(*target_locks_[lock_index]);

      TargetInfo *target_info = registered_targets_.find(pool_id);
      if (target_info != nullptr) {
        target_info->remaining_space_ += bytes_freed;
        HLOG(kDebug, "Updated target {} remaining_space_ by +{} bytes (now {})",
             pool_id.major_, bytes_freed, target_info->remaining_space_);
      }
    }
  }

  // Clear all blocks
  blob_info.blocks_.clear();
  error_code = 0;
  co_return;
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
  telemetry_log_->Push(telemetry_entry);
}

size_t Runtime::GetTelemetryQueueSize() { return telemetry_log_->Size(); }

size_t Runtime::GetTelemetryEntries(std::vector<CteTelemetry> &entries,
                                    size_t max_entries) {
  entries.clear();
  size_t queue_size = telemetry_log_->Size();
  size_t entries_to_read = std::min(max_entries, queue_size);

  entries.reserve(entries_to_read);

  // Read entries by popping and re-pushing them (since peek may not be
  // available)
  std::vector<CteTelemetry> temp_entries;
  temp_entries.reserve(entries_to_read);

  // Pop entries temporarily
  for (size_t i = 0; i < entries_to_read; ++i) {
    CteTelemetry entry;
    bool success = telemetry_log_->Pop(entry);
    if (success) {
      temp_entries.push_back(entry);
    } else {
      break;  // Queue is empty
    }
  }

  // Re-push entries back to queue (in reverse order to maintain order)
  for (auto it = temp_entries.rbegin(); it != temp_entries.rend(); ++it) {
    telemetry_log_->Push(*it);
  }

  // Copy to output vector
  entries = temp_entries;
  return entries.size();
}

chi::TaskResume Runtime::PollTelemetryLog(
    hipc::FullPtr<PollTelemetryLogTask> task, chi::RunContext &ctx) {
  try {
    std::uint64_t minimum_logical_time = task->minimum_logical_time_;

    // Get telemetry entries with logical time filtering
    std::vector<CteTelemetry> all_entries;
    size_t retrieved_count = GetTelemetryEntries(all_entries, 1000);
    (void)retrieved_count;

    // Filter entries by minimum logical time
    task->entries_.clear();
    std::uint64_t max_logical_time = minimum_logical_time;

    for (const auto &entry : all_entries) {
      if (entry.logical_time_ >= minimum_logical_time) {
        task->entries_.push_back(entry);
        max_logical_time = std::max(max_logical_time, entry.logical_time_);
      }
    }

    task->last_logical_time_ = max_logical_time;
    task->return_code_ = 0;

  } catch (const std::exception &e) {
    task->return_code_ = 1;
    task->last_logical_time_ = 0;
  }
  (void)ctx;
  co_return;
}

chi::TaskResume Runtime::GetBlobScore(hipc::FullPtr<GetBlobScoreTask> task,
                                      chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    co_return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;
      co_return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    if (blob_info_ptr == nullptr) {
      task->return_code_ = 1;  // Blob not found
      co_return;
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
    task->return_code_ = 0;
    HLOG(kDebug, "GetBlobScore successful: name={}, score={}", blob_name,
         blob_info_ptr->score_);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::GetBlobSize(hipc::FullPtr<GetBlobSizeTask> task,
                                     chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    co_return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;
      co_return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      task->return_code_ = 1;  // Blob not found
      co_return;
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
    task->return_code_ = 0;
    HLOG(kDebug, "GetBlobSize successful: name={}, size={}", blob_name,
         task->size_);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::GetBlobInfo(hipc::FullPtr<GetBlobInfoTask> task,
                                     chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ =
        HashBlobToContainer(task->tag_id_, task->blob_name_.str());
    co_return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;  // Error: empty blob name
      co_return;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      task->return_code_ = 2;  // Blob not found
      co_return;
    }

    // Step 2: Populate output fields
    task->score_ = blob_info_ptr->score_;
    task->total_size_ = blob_info_ptr->GetTotalSize();

    // Step 3: Populate block information
    // NOTE: Temporarily disabled to debug serialization issue
    task->blocks_.clear();
    // task->blocks_.reserve(blob_info_ptr->blocks_.size());
    // for (const auto &block : blob_info_ptr->blocks_) {
    //   task->blocks_.emplace_back(
    //       block.bdev_client_.pool_id_,
    //       block.size_,
    //       block.target_offset_);
    // }

    // Step 4: Update timestamps
    auto now = std::chrono::steady_clock::now();
    blob_info_ptr->last_read_ = now;

    // Success
    task->return_code_ = 0;
    HLOG(kDebug,
         "GetBlobInfo successful: name={}, score={}, size={}, blocks={}",
         blob_name, task->score_, task->total_size_, task->blocks_.size());

  } catch (const std::exception &e) {
    HLOG(kError, "GetBlobInfo failed: {}", e.what());
    task->return_code_ = 1;
  }
  co_return;
}

chi::TaskResume Runtime::GetContainedBlobs(
    hipc::FullPtr<GetContainedBlobsTask> task, chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
    co_return;
  }

  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;

    // Validate tag exists
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr == nullptr) {
      task->return_code_ = 1;  // Tag not found
      co_return;
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
            task->blob_names_.push_back(blob_name);
          }
        });

    // Success
    task->return_code_ = 0;

    // Log telemetry for this operation
    LogTelemetry(CteOp::kGetOrCreateTag, task->blob_names_.size(), 0, tag_id,
                 std::chrono::steady_clock::now(),
                 std::chrono::steady_clock::now());

    HLOG(kDebug, "GetContainedBlobs successful: tag_id={},{}, found {} blobs",
         tag_id.major_, tag_id.minor_, task->blob_names_.size());

  } catch (const std::exception &e) {
    task->return_code_ = 1;  // Error during operation
    HLOG(kError, "GetContainedBlobs failed: {}", e.what());
  }
  co_return;
}

chi::TaskResume Runtime::TagQuery(hipc::FullPtr<TagQueryTask> task,
                                  chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
    co_return;
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

    // Build results: just tag names matching the query. Respect max_tags_ if
    // non-zero.
    task->results_.clear();
    for (const auto &tn : matching_tags) {
      if (task->max_tags_ != 0 && task->results_.size() >= task->max_tags_) {
        break;
      }
      const std::string &tag_name = tn.first;
      task->results_.push_back(tag_name);
    }

    // Success
    task->return_code_ = 0;
    HLOG(kDebug, "TagQuery successful: pattern={}, found {} tags", tag_regex,
         matching_tags.size());

  } catch (const std::exception &e) {
    task->return_code_ = 1;
    HLOG(kError, "TagQuery failed: {}", e.what());
  }
  co_return;
}

chi::TaskResume Runtime::BlobQuery(hipc::FullPtr<BlobQueryTask> task,
                                   chi::RunContext &ctx) {
  // Dynamic scheduling phase - determine routing
  if (ctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    task->pool_query_ = chi::PoolQuery::Broadcast();
    co_return;
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
    task->tag_names_.clear();
    task->blob_names_.clear();
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
                    task->tag_names_.size() <
                        static_cast<size_t>(task->max_blobs_)) {
                  task->tag_names_.push_back(tag_name);
                  task->blob_names_.push_back(blob_name);
                }
              }
            }
          });
    }

    // Success
    task->return_code_ = 0;
    HLOG(kDebug,
         "BlobQuery successful: tag_pattern={}, blob_pattern={}, found {} "
         "blobs total",
         tag_regex, blob_regex, task->total_blobs_matched_);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
    HLOG(kError, "BlobQuery failed: {}", e.what());
  }
  co_return;
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

}  // namespace wrp_cte::core

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(wrp_cte::core::Runtime)
