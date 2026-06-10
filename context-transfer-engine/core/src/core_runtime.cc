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

#include <clio_runtime/admin/admin_client.h>
#include <clio_cte/core/core_config.h>
#include <clio_cte/core/core_dpe.h>
#include <clio_cte/core/core_runtime.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "clio_ctp/types/atomic.h"  // ctp::ipc::atomic_ref
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
#include <unordered_set>
#include <vector>

#include "clio_runtime/worker.h"
#include "clio_ctp/util/gpu_api.h"
#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/timer.h"

namespace clio::cte::core {

// Bring chi namespace items into scope for CLIO_CUR_WORKER macro
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

void Runtime::FixupAfterCopy(chi::u32 method,
                              ctp::ipc::FullPtr<chi::Task> task_ptr) {
  switch (method) {
    case Method::kRegisterTarget:
      task_ptr.template Cast<RegisterTargetTask>().ptr_->FixupAfterCopy();
      break;
    case Method::kGetOrCreateTag:
      task_ptr.template Cast<GetOrCreateTagTask<CreateParams>>()
          .ptr_->FixupAfterCopy();
      break;
    case Method::kPutBlob:
      task_ptr.template Cast<PutBlobTask>().ptr_->FixupAfterCopy();
      break;
    case Method::kGetBlob:
      task_ptr.template Cast<GetBlobTask>().ptr_->FixupAfterCopy();
      break;
    default:
      break;
  }
}

chi::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task,
                                chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  // Initialize unordered_map_ll instances with appropriately sized bucket
  // counts. Tag/blob maps are large to avoid excessive collisions at scale.
  // Target maps stay small — target counts are O(1–10), not O(100K) — so
  // for_each over registered_targets_ does not have to scan 100K empty slots
  // on every PutBlob.
  static const size_t kTargetMapSize = 64;
  registered_targets_ =
      ctp::priv::unordered_map_ll<chi::PoolId, TargetInfo>(kTargetMapSize);
  target_name_to_id_ =
      ctp::priv::unordered_map_ll<std::string, chi::PoolId>(kTargetMapSize);
  tag_name_to_id_ =
      ctp::priv::unordered_map_ll<std::string, TagId>(kTagMapSize);
  tag_id_to_info_ = ctp::priv::unordered_map_ll<TagId, TagInfo>(kTagMapSize);
  tag_blob_name_to_info_ =
      ctp::priv::unordered_map_ll<std::string, BlobInfo>(kBlobMapSize);

  // Initialize lock vectors for concurrent access
  target_locks_.reserve(kMaxLocks);
  tag_locks_.reserve(kMaxLocks);
  for (size_t i = 0; i < kMaxLocks; ++i) {
    target_locks_.emplace_back(std::make_unique<chi::CoRwLock>());
    tag_locks_.emplace_back(std::make_unique<chi::CoRwLock>());
  }

  // Get IPC manager for later use
  auto *ipc_manager = CLIO_IPC;

  // Initialize telemetry ring buffer using unique_ptr with CTP_MALLOC
  telemetry_log_ = std::make_unique<
      ctp::ipc::circular_mpsc_ring_buffer<CteTelemetry, ctp::ipc::MallocAllocator>>(
      CTP_MALLOC, kTelemetryRingSize);

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
       "CTE Create: GetParams() returned, storage devices in config: {}, "
       "gpu_metadata_cache.enabled={}",
       config_.storage_.devices_.size(),
       config_.gpu_metadata_cache_.enabled_);

  // Configuration is now loaded from compose pool_config via
  // CreateParams::LoadConfig()

  // Build the DPE once from config so ExtendBlob does not pay a heap alloc
  // (and the per-call vtable construction) on every PutBlob.
  dpe_ = DpeFactory::CreateDpe(config_.dpe_.dpe_type_);

  // Store storage configuration in runtime
  storage_devices_ = config_.storage_.devices_;
  HLOG(kDebug, "CTE Create: Copied storage devices to runtime, count: {}",
       storage_devices_.size());

  // Initialize the client with the pool ID
  client_.Init(task->new_pool_id_);

  // Register targets across this container's `neighborhood` window.
  //
  // The prior buggy loop used `target_query = DirectHash(i)` where
  // `i` was the loop iterator alone — every clio_cte_core container
  // on every node registered targets to the same bdev containers
  // (0, 1, ..., neighborhood-1) — and HashBlobToContainer-distributed
  // PutBlobs all funnelled into bdev container 0 (since neighborhood=1
  // made every target route via DirectHash(0)). At 4n × 48 PPN this
  // 144→1 cross-node fan-in saturated libzmq's DEALER send path and
  // the receiving daemon aborted with "Bad address" in tcp.cpp.
  //
  // The intended pattern is a sliding window around the current
  // container: target_query = DirectHash((container_id_ + i) %
  // num_nodes). So clio_cte_core container 0 registers neighborhood
  // targets at bdev containers 0..neighborhood-1, container 1 at
  // 1..neighborhood, etc. With neighborhood=1 each clio_cte_core
  // container registers exactly one target — its own local bdev —
  // and HashBlobToContainer spreads blobs across the N clio_cte_core
  // containers, keeping the write path node-local. Higher neighborhood
  // values give replication breadth without collapsing onto one node.
  if (!storage_devices_.empty()) {
    chi::u32 neighborhood_size = config_.targets_.neighborhood_;
    chi::u32 num_nodes = ipc_manager->GetNumHosts();
    chi::u32 actual_neighborhood = std::min(neighborhood_size, num_nodes);
    HLOG(kDebug,
         "Registering targets across neighborhood window (size: {} nodes, "
         "this container_id_={})",
         actual_neighborhood, container_id_);

    for (size_t device_idx = 0; device_idx < storage_devices_.size();
         ++device_idx) {
      const auto &device = storage_devices_[device_idx];
      chi::u64 capacity_bytes = device.capacity_limit_;
      clio::run::bdev::BdevType bdev_type = clio::run::bdev::BdevType::kFile;
      if (device.bdev_type_ == "ram") {
        bdev_type = clio::run::bdev::BdevType::kRam;
      } else if (device.bdev_type_ == "hbm") {
        bdev_type = clio::run::bdev::BdevType::kHbm;
      } else if (device.bdev_type_ == "pinned") {
        bdev_type = clio::run::bdev::BdevType::kPinned;
      } else if (device.bdev_type_ == "noop") {
        bdev_type = clio::run::bdev::BdevType::kNoop;
      }

      for (chi::u32 i = 0; i < actual_neighborhood; ++i) {
        // Sliding-window neighbor index. Modulo num_nodes wraps the
        // window for containers near the end of the cluster.
        chi::u32 target_node =
            (container_id_ + i) % std::max<chi::u32>(num_nodes, 1u);

        std::string target_path =
            device.path_ + "_node" + std::to_string(target_node);
        chi::PoolQuery target_query =
            chi::PoolQuery::DirectHash(target_node);
        chi::PoolId bdev_id(512 + static_cast<chi::u32>(device_idx),
                            1 + target_node);

        HLOG(kDebug,
             "Registering target ({}): {} ({}, {} bytes) on node {} (i={}) "
             "with bdev_id=({},{})",
             client_.pool_id_, target_path, device.bdev_type_, capacity_bytes,
             target_node, i, bdev_id.major_, bdev_id.minor_);
        auto reg_task = client_.AsyncRegisterTarget(
            target_path, bdev_type, capacity_bytes, target_query, bdev_id);
        CLIO_CO_AWAIT(reg_task);
        chi::u32 result = reg_task->GetReturnCode();
        if (result == 0) {
          HLOG(kDebug, "  - Registered target: {} on node {}", target_path,
               target_node);
        } else {
          HLOG(kWarning,
               "  - Failed to register target {} on node {} (error code: {})",
               target_path, target_node, result);
        }
      }
    }
  } else {
    HLOG(kWarning, "Warning: No storage devices configured");
  }

  // Queue management has been removed - queues are now managed by CLIO Runtime
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
    chi::u32 num_workers =
        std::max(CLIO_WORK_ORCHESTRATOR->GetTotalWorkerCount(), (chi::u32)1);
    chi::u64 per_worker_capacity = std::max(
        config_.performance_.transaction_log_capacity_bytes_ / num_workers,
        (chi::u64)4096);
    blob_txn_logs_.resize(num_workers);
    tag_txn_logs_.resize(num_workers);
    for (chi::u32 i = 0; i < num_workers; ++i) {
      blob_txn_logs_[i] = std::make_unique<TransactionLog>();
      blob_txn_logs_[i]->Open(config_.performance_.metadata_log_path_ +
                                  ".blob." + std::to_string(i),
                              per_worker_capacity);
      tag_txn_logs_[i] = std::make_unique<TransactionLog>();
      tag_txn_logs_[i]->Open(
          config_.performance_.metadata_log_path_ + ".tag." + std::to_string(i),
          per_worker_capacity);
    }
    HLOG(kInfo, "WAL: Opened {} blob and {} tag transaction logs", num_workers,
         num_workers);
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

  // Allocate the optional GPU metadata cache. The OUT pointer is
  // re-serialized back into chimod_params_ (a chi::priv::string) so
  // the client's GetParams() sees the populated gpu_cache_ptr_ after
  // Wait().
  CreateParams out_params;
  out_params.config_ = config_;
  out_params.gpu_cache_ptr_ =
      GpuCacheCreate() ? reinterpret_cast<chi::u64>(gpu_cache_)
                       : static_cast<chi::u64>(0);
  chi::Task::Serialize(CLIO_PRIV_ALLOC, task->chimod_params_, out_params);

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Destroy(ctp::ipc::FullPtr<DestroyTask> task,
                                 chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Close WAL files before clearing data structures
    for (auto &log : blob_txn_logs_) {
      if (log) log->Close();
    }
    blob_txn_logs_.clear();
    for (auto &log : tag_txn_logs_) {
      if (log) log->Close();
    }
    tag_txn_logs_.clear();

    // Clear all registered targets and their associated data
    registered_targets_.clear();
    target_list_.clear();
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::PoolQuery Runtime::ScheduleTask(const ctp::ipc::FullPtr<chi::Task> &task) {
  using namespace clio::cte::core;
  switch (task->method_) {
    // Methods that route locally
    case Method::kRegisterTarget:
    case Method::kUnregisterTarget:
    case Method::kListTargets:
    case Method::kStatTargets:
    case Method::kGetTargetInfo:
      return chi::PoolQuery::Local();

    // GetOrCreateTag: check local tag cache, hash to container if not found
    case Method::kGetOrCreateTag: {
      auto typed = task.template Cast<GetOrCreateTagTask<CreateParams>>();
      std::string tag_name = typed->tag_name_.str();
      bool tag_exists = false;
      {
        chi::ScopedCoRwReadLock lock(tag_map_lock_);
        tag_exists = (tag_name_to_id_.find(tag_name) != nullptr);
      }
      if (tag_exists) {
        return chi::PoolQuery::Local();
      }
      std::hash<std::string> string_hasher;
      chi::u32 hash_value = static_cast<chi::u32>(string_hasher(tag_name));
      return chi::PoolQuery::DirectHash(hash_value);
    }

    // Blob operations: hash blob name to container
    case Method::kPutBlob: {
      auto typed = task.template Cast<PutBlobTask>();
      return HashBlobToContainer(typed->tag_id_, typed->blob_name_.str());
    }
    case Method::kGetBlob: {
      auto typed = task.template Cast<GetBlobTask>();
      return HashBlobToContainer(typed->tag_id_, typed->blob_name_.str());
    }
    case Method::kReorganizeBlob: {
      auto typed = task.template Cast<ReorganizeBlobTask>();
      return HashBlobToContainer(typed->tag_id_, typed->blob_name_.str());
    }
    case Method::kDelBlob: {
      auto typed = task.template Cast<DelBlobTask>();
      return HashBlobToContainer(typed->tag_id_, typed->blob_name_.str());
    }
    case Method::kGetBlobScore: {
      auto typed = task.template Cast<GetBlobScoreTask>();
      return HashBlobToContainer(typed->tag_id_, typed->blob_name_.str());
    }
    case Method::kGetBlobSize: {
      auto typed = task.template Cast<GetBlobSizeTask>();
      return HashBlobToContainer(typed->tag_id_, typed->blob_name_.str());
    }
    case Method::kGetBlobInfo: {
      auto typed = task.template Cast<GetBlobInfoTask>();
      return HashBlobToContainer(typed->tag_id_, typed->blob_name_.str());
    }

    // Broadcast operations
    case Method::kGetTagSize:
    case Method::kGetContainedBlobs:
    case Method::kTagQuery:
    case Method::kBlobQuery:
      return chi::PoolQuery::Broadcast();

    default:
      return task->pool_query_;
  }
}

chi::TaskResume Runtime::RegisterTarget(ctp::ipc::FullPtr<RegisterTargetTask> task,
                                        chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string target_name = task->target_name_.str();
    clio::run::bdev::BdevType bdev_type = task->bdev_type_;
    chi::u64 total_size = task->total_size_;
    chi::PoolId bdev_pool_id = task->bdev_id_;
    HLOG(kDebug, "Registering target ({}): {} ({} bytes) with bdev_id=({},{})",
         client_.pool_id_, target_name, total_size, bdev_pool_id.major_,
         bdev_pool_id.minor_);

    // Create bdev client and container first to get the TargetId (pool_id)
    clio::run::bdev::Client bdev_client;
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
    CLIO_CO_AWAIT(create_task);
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
      CLIO_CO_RETURN;
    }

    // Get the TargetId (bdev_client's pool_id) for indexing
    chi::PoolId target_id = bdev_client.pool_id_;

    // Check if target is already registered using TargetId
    {
      chi::ScopedCoRwReadLock read_lock(target_lock_);
      TargetInfo *existing_target = registered_targets_.find(target_id);
      if (existing_target != nullptr) {
        CLIO_CO_RETURN;
      }
    }

    // Get actual statistics from bdev using AsyncGetStats method
    chi::u64 remaining_size;
    auto stats_task = bdev_client.AsyncGetStats();
    CLIO_CO_AWAIT(stats_task);
    clio::run::bdev::PerfMetrics perf_metrics = stats_task->metrics_;
    remaining_size = stats_task->remaining_size_;

    // Create target info with bdev client and performance stats
    TargetInfo target_info(target_name, bdev_pool_name);
    HLOG(kDebug, "RegisterTarget: Before move, bdev_client.pool_id_=({},{})",
         bdev_client.pool_id_.major_, bdev_client.pool_id_.minor_);
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
    target_info.bdev_type_ = task->bdev_type_;

    // Register the target using TargetId as key. Mirror into target_list_ so
    // iteration sites (ExtendBlob, ListTargets, StatTargets, FlushData) can
    // walk live entries directly without scanning empty map slots.
    {
      chi::ScopedCoRwWriteLock write_lock(target_lock_);
      registered_targets_.insert_or_assign(target_id, target_info);
      target_name_to_id_.insert_or_assign(
          target_name,
          target_id);  // Maintain reverse lookup
      // Replace existing entry if present, else append.
      bool found_in_list = false;
      for (auto &t : target_list_) {
        if (t.bdev_client_.pool_id_ == target_id) {
          t = target_info;
          found_in_list = true;
          break;
        }
      }
      if (!found_in_list) {
        target_list_.push_back(target_info);
      }
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::UnregisterTarget(
    ctp::ipc::FullPtr<UnregisterTargetTask> task, chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string target_name = task->target_name_.str();

    // Check if target exists and remove it (don't destroy bdev container)
    {
      chi::ScopedCoRwWriteLock write_lock(target_lock_);

      // Look up TargetId from target_name (under lock)
      chi::PoolId *target_id_ptr = target_name_to_id_.find(target_name);
      if (target_id_ptr == nullptr) {
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }

      // Copy by value: target_id_ptr points into the map node that
      // erase(target_name) below frees, and target_id is still read in the
      // target_list_ loop after that (ASan heap-use-after-free, issue #520).
      const chi::PoolId target_id = *target_id_ptr;
      if (!registered_targets_.contains(target_id)) {
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }

      registered_targets_.erase(target_id);
      target_name_to_id_.erase(target_name);  // Remove reverse lookup
      // Remove from target_list_ via swap-and-pop (order doesn't matter)
      for (size_t i = 0; i < target_list_.size(); ++i) {
        if (target_list_[i].bdev_client_.pool_id_ == target_id) {
          if (i + 1 != target_list_.size()) {
            target_list_[i] = target_list_.back();
          }
          target_list_.pop_back();
          break;
        }
      }
    }

    task->return_code_ = 0;  // Success
    HLOG(kDebug, "Target '{}' unregistered", target_name);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::ListTargets(ctp::ipc::FullPtr<ListTargetsTask> task,
                                     chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Clear the output vector and populate with current target names
    task->target_names_.clear();

    chi::ScopedCoRwReadLock read_lock(target_lock_);

    // Populate target name list from the contiguous mirror (live entries only)
    task->target_names_.reserve(target_list_.size());
    for (const auto &t : target_list_) {
      task->target_names_.push_back(t.target_name_.str());
    }

    task->return_code_ = 0;  // Success

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::StatTargets(ctp::ipc::FullPtr<StatTargetsTask> task,
                                     chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Collect all target IDs under read lock (can't co_await inside lambda)
    std::vector<chi::PoolId> target_ids;
    {
      chi::ScopedCoRwReadLock read_lock(target_lock_);
      target_ids.reserve(target_list_.size());
      for (const auto &t : target_list_) {
        target_ids.push_back(t.bdev_client_.pool_id_);
      }
    }

    // Now iterate and co_await each UpdateTargetStats call
    // Cannot hold lock across co_await, so acquire/release per-target
    for (const auto &target_id : target_ids) {
      // Copy bdev_client under read lock for the async call
      clio::run::bdev::Client bdev_client_copy;
      bool found = false;
      {
        chi::ScopedCoRwReadLock read_lock(target_lock_);
        TargetInfo *target_info = registered_targets_.find(target_id);
        if (target_info != nullptr) {
          bdev_client_copy = target_info->bdev_client_;
          found = true;
        }
      }
      if (!found) continue;

      // Perform async stats query WITHOUT holding lock
      chi::u64 remaining_size;
      auto stats_task = bdev_client_copy.AsyncGetStats();
      CLIO_CO_AWAIT(stats_task);
      clio::run::bdev::PerfMetrics perf_metrics = stats_task->metrics_;
      remaining_size = stats_task->remaining_size_;

      // Re-acquire write lock to update target info. Mutate the map and the
      // mirror in target_list_ in lockstep so DPE selection sees fresh stats.
      {
        chi::ScopedCoRwWriteLock write_lock(target_lock_);
        TargetInfo *target_info = registered_targets_.find(target_id);
        if (target_info != nullptr) {
          target_info->perf_metrics_ = perf_metrics;
          target_info->remaining_space_ = remaining_size;

          float manual_score =
              GetManualScoreForTarget(target_info->target_name_.str());
          if (manual_score >= 0.0f) {
            target_info->target_score_ = manual_score;
          } else {
            double max_bandwidth =
                std::max(target_info->perf_metrics_.read_bandwidth_mbps_,
                         target_info->perf_metrics_.write_bandwidth_mbps_);
            if (max_bandwidth > 0.0) {
              double global_max_bandwidth = 1000.0;
              target_info->target_score_ =
                  static_cast<float>(std::log(max_bandwidth + 1.0) /
                                     std::log(global_max_bandwidth + 1.0));
              target_info->target_score_ =
                  std::max(0.0f, std::min(1.0f, target_info->target_score_));
            }
          }
          // Mirror into target_list_
          for (auto &t : target_list_) {
            if (t.bdev_client_.pool_id_ == target_id) {
              t.perf_metrics_ = target_info->perf_metrics_;
              t.remaining_space_ = target_info->remaining_space_;
              t.target_score_ = target_info->target_score_;
              break;
            }
          }
        }
      }
    }

    task->return_code_ = 0;  // Success

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

template <typename CreateParamsT>
chi::TaskResume Runtime::GetOrCreateTag(
    ctp::ipc::FullPtr<GetOrCreateTagTask<CreateParamsT>> task,
    chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string tag_name = task->tag_name_.str();
    TagId preferred_id = task->tag_id_;
    auto *ipc_manager = CLIO_IPC;
    chi::u32 local_node_id = ipc_manager->GetNodeId();

    // Check if this is a returning task from a remote canonical node
    bool is_remote_tag =
        (preferred_id.major_ != 0 && preferred_id.major_ != local_node_id);

    if (is_remote_tag) {
      chi::ScopedCoRwWriteLock write_lock(tag_map_lock_);
      TagId *existing_tag_id_ptr = tag_name_to_id_.find(tag_name);
      if (existing_tag_id_ptr == nullptr) {
        tag_name_to_id_.insert_or_assign(tag_name, preferred_id);
      }
      task->tag_id_ = preferred_id;
      GpuCacheOnGetOrCreateTag(preferred_id, tag_name);
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    TagId tag_id = GetOrAssignTagId(tag_name, preferred_id);
    task->tag_id_ = tag_id;

    auto now = GetCurrentTimeNs();
    {
      chi::ScopedCoRwWriteLock write_lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr != nullptr) {
        tag_info_ptr->last_read_ = now;
        LogTelemetry(CteOp::kGetOrCreateTag, 0, 0, tag_id,
                     tag_info_ptr->last_modified_, now);
      }
    }
    GpuCacheOnGetOrCreateTag(tag_id, tag_name);
    task->return_code_ = 0;

  } catch (const std::exception &e) {
    HLOG(kError, "GetOrCreateTag: Exception: {}", e.what());
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::GetTargetInfo(ctp::ipc::FullPtr<GetTargetInfoTask> task,
                                       chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string target_name = task->target_name_.str();

    // Look up target by name (under lock for concurrent safety)
    chi::ScopedCoRwReadLock read_lock(target_lock_);
    chi::PoolId *target_id_ptr = target_name_to_id_.find(target_name);
    if (target_id_ptr == nullptr) {
      task->return_code_ = 1;  // Target not found
      CLIO_CO_RETURN;
    }

    chi::PoolId target_id = *target_id_ptr;

    // Find target in registered_targets_
    auto target_ptr = registered_targets_.find(target_id);
    if (!target_ptr) {
      task->return_code_ = 2;  // Target not in registered list
      CLIO_CO_RETURN;
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::PutBlob(ctp::ipc::FullPtr<PutBlobTask> task,
                                 chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  // Per-PutBlob diagnostic logging — disabled in perf builds. Was burning
  // an atomic fetch_add + clock_gettime + branch on every 64 KB chunk plus
  // an HLOG every 8th chunk (300+/s at FUSE saturation), measurably slowing
  // the FUSE→CTE write path. Re-enable by flipping `#if 0` → `#if 1`.
#if 0
  // DEBUG: unconditional log to verify the handler is hit and to
  // print whether submit_ts_ns_ survived the client→daemon hop.
  {
    static std::atomic<uint64_t> s_seen{0};
    uint64_t n = s_seen.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & 7) == 0 || n <= 4) {
      HLOG(kInfo,
           "[PutLat-DBG] handler entered #{} submit_ts_ns={} size={}",
           n, task->submit_ts_ns_, task->size_);
    }
  }
  // Submit→handler-entry latency. submit_ts_ns_ is stamped at
  // AsyncPutBlob time on the client; reading it here measures
  // (client newtask) + (ipc send) + (cross-node serialize+wire+
  // deserialize, if remote) + (lane queue wait) + (worker dispatch)
  // — i.e. everything between when the rank issued the put and when
  // chimaera actually started executing it. Dumped sparsely to keep
  // log volume sane; the per-task ns are kInfo at 1 in 64 and the
  // running aggregate is kInfo every kPutLatDumpPeriod tasks.
  {
    static std::atomic<uint64_t> s_n{0};
    static std::atomic<uint64_t> s_sum_ns{0};
    static std::atomic<uint64_t> s_max_ns{0};
    static constexpr uint64_t kPutLatDumpPeriod = 16;
    if (task->submit_ts_ns_ != 0) {
      uint64_t now_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      // Guard against clock skew on cross-node tasks: drop negatives.
      if (now_ns > task->submit_ts_ns_) {
        uint64_t dt = now_ns - task->submit_ts_ns_;
        uint64_t n = s_n.fetch_add(1, std::memory_order_relaxed) + 1;
        s_sum_ns.fetch_add(dt, std::memory_order_relaxed);
        // Lock-free max
        uint64_t cur_max = s_max_ns.load(std::memory_order_relaxed);
        while (dt > cur_max &&
               !s_max_ns.compare_exchange_weak(cur_max, dt,
                                               std::memory_order_relaxed)) {
        }
        if ((n & 7) == 0) {
          HLOG(kInfo,
               "[PutLat] sample dt_us={} task={} size={}",
               dt / 1000, task->task_id_, task->size_);
        }
        if ((n % kPutLatDumpPeriod) == 0) {
          uint64_t avg_us =
              (s_sum_ns.load(std::memory_order_relaxed) / n) / 1000;
          uint64_t max_us =
              s_max_ns.load(std::memory_order_relaxed) / 1000;
          HLOG(kInfo,
               "[PutLat] n={} avg_us={} max_us={}", n, avg_us, max_us);
        }
      }
    }
  }
#endif

  try {
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    // Append the per-page suffix when a GPU client (gpu_vector::Vector)
    // routed this put through a per-(block, page) sub-blob — keeps cache
    // pages from colliding on a shared blob name. Sentinel kNoPageIdx
    // means "no suffix", which is the path non-GPU clients take.
    if (task->gpu_page_idx_ != PutBlobTask::kNoPageIdx) {
      blob_name += "_pi" + std::to_string(task->gpu_page_idx_);
    }
    chi::u64 offset = task->offset_;
    chi::u64 size = task->size_;
    ctp::ipc::ShmPtr<> blob_data = task->blob_data_;
    float blob_score = task->score_;

    // Validate inputs
    if (size == 0) {
      task->return_code_ = 2;
      CLIO_CO_RETURN;
    }
    if (blob_data.IsNull()) {
      task->return_code_ = 3;
      CLIO_CO_RETURN;
    }
    if (blob_name.empty()) {
      task->return_code_ = 4;
      CLIO_CO_RETURN;
    }

    // Check if blob exists and resolve score
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    bool blob_found = (blob_info_ptr != nullptr);
    if (blob_score < 0.0f) {
      blob_score = blob_found ? blob_info_ptr->score_ : 1.0f;
    }
    if (blob_score > 1.0f) {
      task->return_code_ = 5;
      CLIO_CO_RETURN;
    }

    // Step 1: ClearBlob — free blocks if full replacement
    chi::u64 old_blob_size = 0;
    if (blob_found) {
      bool cleared = false;
      CLIO_CO_AWAIT(ClearBlob(*blob_info_ptr, blob_score, offset, size, cleared));
      if (cleared) {
        // WAL: log blob clear
        if (!blob_txn_logs_.empty()) {
          chi::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
          TxnClearBlob txn;
          txn.tag_major_ = tag_id.major_;
          txn.tag_minor_ = tag_id.minor_;
          txn.blob_name_ = blob_name;
          blob_txn_logs_[wid % blob_txn_logs_.size()]->Log(TxnType::kClearBlob,
                                                           txn);
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
        CLIO_CO_RETURN;
      }
    }

    // Step 2: ExtendBlob — allocate new blocks if needed
    chi::u32 alloc_result = 0;
    CLIO_CO_AWAIT(ExtendBlob(*blob_info_ptr, offset, size, blob_score, alloc_result,
                        task->context_.min_persistence_level_));
    if (alloc_result != 0) {
      task->return_code_ = 10 + alloc_result;
      CLIO_CO_RETURN;
    }

    // WAL: log all current blocks (full replacement semantics)
    if (!blob_txn_logs_.empty() && !blob_info_ptr->blocks_.empty()) {
      chi::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
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
      blob_txn_logs_[wid % blob_txn_logs_.size()]->Log(TxnType::kExtendBlob,
                                                       txn);
    }

    // Step 3: ModifyExistingData — write data to blocks
    chi::u32 write_result = 0;
    CLIO_CO_AWAIT(ModifyExistingData(blob_info_ptr->blocks_, blob_data, size, offset,
                                write_result));
    if (write_result != 0) {
      task->return_code_ = 20 + write_result;
      CLIO_CO_RETURN;
    }

#if CTP_ENABLE_COMPRESS
    // Update compression metadata
    Context &context = task->context_;
    blob_info_ptr->compress_lib_ = context.compress_lib_;
    blob_info_ptr->compress_preset_ = context.compress_preset_;
    blob_info_ptr->trace_key_ = context.trace_key_;
#endif

    // Update tag size
    chi::u64 new_blob_size = blob_info_ptr->GetTotalSize();
    chi::i64 size_change = static_cast<chi::i64>(new_blob_size) -
                           static_cast<chi::i64>(old_blob_size);
    auto now = GetCurrentTimeNs();
    blob_info_ptr->last_modified_ = now;
    blob_info_ptr->score_ = blob_score;
    {
      // Write lock: we may need to insert a fresh tag_info entry. The
      // tag's name lives on whichever container `GetOrCreateTag`'s
      // `DirectHash(tag_name)` selected; this container only owns the
      // blobs that `HashBlobToContainer(tag_id, blob_name)` routed
      // here. To keep the `GetTagSize` broadcast-and-Aggregate sum
      // correct, every container that holds any of the tag's bytes
      // must carry a `TagInfo` whose `total_size_` reflects its share.
      // The silent-skip variant of this block dropped the accounting
      // when the tag wasn't locally registered, so on 2n the
      // tag-owning container saw total_size_ = 0 (no PutBlobs hashed
      // to it) and the blob-owning container had no TagInfo at all
      // (rc=1, tag_size_=0). stat() then returned 0 after writes.
      chi::ScopedCoRwWriteLock lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr == nullptr) {
        // No prior local accounting; seed an entry. tag_name_ stays
        // empty -- the canonical name<->id binding lives on the
        // tag-owning container and isn't this container's concern.
        TagInfo seed;
        seed.tag_id_ = tag_id;
        seed.last_modified_ = now;
        seed.last_read_ = now;
        seed.total_size_ = 0;
        auto ins = tag_id_to_info_.insert_or_assign(tag_id, seed);
        tag_info_ptr = ins.value;
      }
      if (tag_info_ptr) {
        tag_info_ptr->last_modified_ = now;
        if (size_change >= 0) {
          tag_info_ptr->total_size_ += static_cast<chi::u64>(size_change);
        } else {
          chi::u64 abs_change = static_cast<chi::u64>(-size_change);
          tag_info_ptr->total_size_ -= abs_change;
        }
      }
    }

    LogTelemetry(CteOp::kPutBlob, offset, size, tag_id, now,
                 blob_info_ptr->last_read_);
    GpuCacheOnPutBlob(tag_id, blob_name, *blob_info_ptr);
    task->return_code_ = 0;
  } catch (const std::exception &e) {
    HLOG(kError, "PutBlob failed with exception: {}", e.what());
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::GetBlob(ctp::ipc::FullPtr<GetBlobTask> task,
                                 chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    if (task->gpu_page_idx_ != GetBlobTask::kNoPageIdx) {
      blob_name += "_pi" + std::to_string(task->gpu_page_idx_);
    }
    chi::u64 offset = task->offset_;
    chi::u64 size = task->size_;
    chi::u32 flags = task->flags_;

    // Suppress unused variable warning for flags - may be used in future
    (void)flags;

    // Validate input parameters
    if (size == 0) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    // If blob doesn't exist, error
    if (blob_info_ptr == nullptr) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Use the pre-provided data pointer from the task
    ctp::ipc::ShmPtr<> blob_data_ptr = task->blob_data_;

    // Step 2: Read data from blob blocks (no lock held during I/O)
    chi::u32 read_result = 0;
    CLIO_CO_AWAIT(ReadData(blob_info_ptr->blocks_, blob_data_ptr, size, offset,
                      read_result));
    if (read_result != 0) {
      task->return_code_ = read_result;
      CLIO_CO_RETURN;
    }

    // Step 3: Update timestamp (no lock needed - just updating values, not
    // modifying map structure)
    auto now = GetCurrentTimeNs();
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::ReorganizeBlob(ctp::ipc::FullPtr<ReorganizeBlobTask> task,
                                        chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    float new_score = task->new_score_;

    // Validate inputs
    if (blob_name.empty()) {
      task->return_code_ = 1;  // Invalid input - empty blob name
      CLIO_CO_RETURN;
    }

    if (new_score < 0.0f || new_score > 1.0f) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Get configuration for score difference threshold
    const Config &config = GetConfig();
    float score_difference_threshold =
        config.performance_.score_difference_threshold_;

    // Step 1: Get blob info directly from table
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      task->return_code_ = 3;  // Blob not found
      CLIO_CO_RETURN;
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
      CLIO_CO_RETURN;
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
      CLIO_CO_RETURN;
    }

    // Step 5: Allocate buffer for blob data
    auto *ipc_manager = CLIO_IPC;
    ctp::ipc::FullPtr<char> blob_data_buffer =
        ipc_manager->AllocateBuffer(blob_size);
    if (blob_data_buffer.IsNull()) {
      HLOG(kError, "Failed to allocate buffer for blob during reorganization");
      task->return_code_ = 5;  // Buffer allocation failed
      CLIO_CO_RETURN;
    }

    // Step 6: Get blob data
    auto get_task =
        client_.AsyncGetBlob(tag_id, blob_name, 0, blob_size, 0,
                             blob_data_buffer.shm_.template Cast<void>());
    CLIO_CO_AWAIT(get_task);

    if (get_task->return_code_ != 0u) {
      HLOG(kWarning, "Failed to get blob data during reorganization");
      task->return_code_ = 6;  // Get blob failed
      CLIO_CO_RETURN;
    }

    // Step 7: Put blob with new score (data reorganization)
    HLOG(kDebug,
         "ReorganizeBlob calling AsyncPutBlob for blob={}, new_score={}",
         blob_name, new_score);
    auto put_task = client_.AsyncPutBlob(
        tag_id, blob_name, 0, blob_size,
        blob_data_buffer.shm_.template Cast<void>(), new_score, Context(), 0);
    CLIO_CO_AWAIT(put_task);

    if (put_task->return_code_ != 0) {
      HLOG(kWarning, "Failed to put blob during reorganization");
      task->return_code_ = 7;  // Put blob failed
      CLIO_CO_RETURN;
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::DelBlob(ctp::ipc::FullPtr<DelBlobTask> task,
                                 chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    if (blob_info_ptr == nullptr) {
      task->return_code_ = 1;  // Blob not found
      CLIO_CO_RETURN;
    }

    // Step 2: Get blob size before deletion for tag size accounting
    chi::u64 blob_size = blob_info_ptr->GetTotalSize();

    // Step 2.5: Free all blocks back to their targets before removing blob
    chi::u32 free_result = 0;
    CLIO_CO_AWAIT(FreeAllBlobBlocks(*blob_info_ptr, free_result));
    if (free_result != 0) {
      HLOG(kWarning,
           "Failed to free some blocks for blob={}, continuing with deletion",
           blob_name);
      // Continue with deletion even if freeing fails to avoid orphaned blob
      // entries
    }

    // Step 3: Update tag's total_size_
    {
      chi::ScopedCoRwWriteLock lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr != nullptr) {
        if (blob_size <= tag_info_ptr->total_size_) {
          tag_info_ptr->total_size_ -= blob_size;
        } else {
          tag_info_ptr->total_size_ = 0;
        }
      }
    }

    // Step 5: Remove blob from tag_blob_name_to_info_ map
    {
      chi::ScopedCoRwWriteLock lock(blob_map_lock_);
      std::string compound_key = std::to_string(tag_id.major_) + "." +
                                 std::to_string(tag_id.minor_) + "." +
                                 blob_name;
      tag_blob_name_to_info_.erase(compound_key);
    }

    // Step 6: Log telemetry for DelBlob operation
    auto now = GetCurrentTimeNs();
    LogTelemetry(CteOp::kDelBlob, 0, blob_size, tag_id, now, now);

    // WAL: log blob deletion
    if (!blob_txn_logs_.empty()) {
      chi::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
      TxnDelBlob txn;
      txn.tag_major_ = tag_id.major_;
      txn.tag_minor_ = tag_id.minor_;
      txn.blob_name_ = blob_name;
      blob_txn_logs_[wid % blob_txn_logs_.size()]->Log(TxnType::kDelBlob, txn);
    }

    // Success
    GpuCacheOnDelBlob(tag_id, blob_name);
    task->return_code_ = 0;
    HLOG(kDebug, "DelBlob successful: name={}, blob_size={}", blob_name,
         blob_size);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::DelTag(ctp::ipc::FullPtr<DelTagTask> task,
                                chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    TagId tag_id = task->tag_id_;
    std::string tag_name = task->tag_name_.str();

    // Step 1: Resolve tag ID if tag name was provided instead
    if (tag_id.IsNull() && !tag_name.empty()) {
      chi::ScopedCoRwReadLock lock(tag_map_lock_);
      TagId *found_tag_id_ptr = tag_name_to_id_.find(tag_name);
      if (found_tag_id_ptr == nullptr) {
        task->return_code_ = 1;  // Tag not found by name
        CLIO_CO_RETURN;
      }
      tag_id = *found_tag_id_ptr;
      task->tag_id_ = tag_id;
    } else if (tag_id.IsNull() && tag_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Step 2: Find the tag by ID
    std::string cached_tag_name;
    {
      chi::ScopedCoRwReadLock lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr == nullptr) {
        task->return_code_ = 1;  // Tag not found by ID
        CLIO_CO_RETURN;
      }
      cached_tag_name = tag_info_ptr->tag_name_.str();
    }

    // Step 3: Collect blob names under read lock, then delete
    std::string tag_prefix = std::to_string(tag_id.major_) + "." +
                             std::to_string(tag_id.minor_) + ".";
    std::vector<std::string> blob_names_to_delete;
    {
      chi::ScopedCoRwReadLock lock(blob_map_lock_);
      tag_blob_name_to_info_.for_each(
          [&tag_prefix, &blob_names_to_delete](const std::string &compound_key,
                                               const BlobInfo &blob_info) {
            if (compound_key.compare(0, tag_prefix.length(), tag_prefix) == 0) {
              blob_names_to_delete.push_back(blob_info.blob_name_.str());
            }
          });
    }

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
        CLIO_CO_AWAIT(task);

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

    // Step 4: Remove all blob name mappings for this tag
    {
      chi::ScopedCoRwWriteLock lock(blob_map_lock_);
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
    }

    // Step 5: Remove tag name and tag info mappings
    size_t blob_count = processed_blobs;
    size_t total_size = 0;
    {
      chi::ScopedCoRwWriteLock lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr != nullptr) {
        total_size = tag_info_ptr->total_size_;
        if (!tag_info_ptr->tag_name_.empty()) {
          tag_name_to_id_.erase(tag_info_ptr->tag_name_.str());
        }
      }
    }

    // Log telemetry for DelTag operation
    auto now = GetCurrentTimeNs();
    LogTelemetry(CteOp::kDelTag, 0, total_size, tag_id, now, now);

    // WAL: log tag deletion
    if (!tag_txn_logs_.empty()) {
      chi::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
      TxnDelTag txn;
      txn.tag_name_ = cached_tag_name;
      txn.tag_major_ = tag_id.major_;
      txn.tag_minor_ = tag_id.minor_;
      tag_txn_logs_[wid % tag_txn_logs_.size()]->Log(TxnType::kDelTag, txn);
    }

    {
      chi::ScopedCoRwWriteLock lock(tag_map_lock_);
      tag_id_to_info_.erase(tag_id);
    }

    // Success
    GpuCacheOnDelTag(tag_id);
    task->return_code_ = 0;
    HLOG(kDebug,
         "DelTag successful: tag_id={},{}, removed {} blobs, total_size={}",
         tag_id.major_, tag_id.minor_, blob_count, total_size);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::GetTagSize(ctp::ipc::FullPtr<GetTagSizeTask> task,
                                    chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    TagId tag_id = task->tag_id_;

    // Find the tag
    chi::ScopedCoRwWriteLock lock(tag_map_lock_);
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr == nullptr) {
      task->return_code_ = 1;  // Tag not found
      task->tag_size_ = 0;
      CLIO_CO_RETURN;
    }

    // Update timestamp and return the total size
    auto now = GetCurrentTimeNs();
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// Private helper methods
const Config &Runtime::GetConfig() const { return config_; }

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
      return device.score_;  // Return configured score (-1.0f if not set)
    }
  }

  return -1.0f;  // No manual score configured for this target
}

clio::run::bdev::PersistenceLevel Runtime::GetPersistenceLevelForTarget(
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
        return clio::run::bdev::PersistenceLevel::kTemporaryNonVolatile;
      } else if (device.persistence_level_ == "long_term") {
        return clio::run::bdev::PersistenceLevel::kLongTerm;
      }
      return clio::run::bdev::PersistenceLevel::kVolatile;
    }
  }
  return clio::run::bdev::PersistenceLevel::kVolatile;
}

TagId Runtime::GetOrAssignTagId(const std::string &tag_name,
                                const TagId &preferred_id) {
  chi::ScopedCoRwWriteLock write_lock(tag_map_lock_);

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
    chi::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
    TxnCreateTag txn;
    txn.tag_name_ = tag_name;
    txn.tag_major_ = tag_id.major_;
    txn.tag_minor_ = tag_id.minor_;
    tag_txn_logs_[wid % tag_txn_logs_.size()]->Log(TxnType::kCreateTag, txn);
  }

  return tag_id;
}

chi::TaskResume Runtime::FlushMetadata(ctp::ipc::FullPtr<FlushMetadataTask> task,
                                       chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  task->entries_flushed_ = 0;

  const std::string &log_path = config_.performance_.metadata_log_path_;
  if (log_path.empty()) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }

  try {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(log_path).parent_path());

    std::ofstream ofs(log_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      HLOG(kError, "FlushMetadata: Failed to open log file: {}", log_path);
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Write TagInfo entries (entry_type 0)
    tag_id_to_info_.for_each([&](const TagId &id, const TagInfo &info) {
      uint8_t entry_type = 0;
      uint32_t name_len = static_cast<uint32_t>(info.tag_name_.size());
      chi::u64 total_size = info.total_size_;
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
        if (log) {
          log->Sync();
          total_wal_size += log->Size();
        }
      }
      for (auto &log : tag_txn_logs_) {
        if (log) {
          log->Sync();
          total_wal_size += log->Size();
        }
      }
      if (total_wal_size >
          config_.performance_.transaction_log_capacity_bytes_) {
        for (auto &log : blob_txn_logs_) {
          if (log) log->Truncate();
        }
        for (auto &log : tag_txn_logs_) {
          if (log) log->Truncate();
        }
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::FlushData(ctp::ipc::FullPtr<FlushDataTask> task,
                                   chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  task->bytes_flushed_ = 0;
  task->blobs_flushed_ = 0;

  int target_level = task->target_persistence_level_;

  // Find non-volatile targets that meet the persistence level requirement
  std::vector<chi::PoolId> nonvolatile_targets;
  {
    chi::ScopedCoRwReadLock read_lock(target_lock_);
    for (const auto &t : target_list_) {
      if (static_cast<int>(t.persistence_level_) >= target_level) {
        nonvolatile_targets.push_back(t.bdev_client_.pool_id_);
      }
    }
  }

  if (nonvolatile_targets.empty()) {
    HLOG(kDebug, "FlushData: No non-volatile targets available at level >= {}",
         target_level);
    task->return_code_ = 0;
    CLIO_CO_RETURN;
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

  {
    chi::ScopedCoRwReadLock read_lock(target_lock_);
    tag_blob_name_to_info_.for_each([&](const std::string &key,
                                        const BlobInfo &blob_info) {
      if (blob_info.blocks_.empty()) return;

      bool has_volatile_blocks = false;
      for (const auto &block : blob_info.blocks_) {
        chi::PoolId pool_id = block.bdev_client_.pool_id_;
        TargetInfo *tinfo = registered_targets_.find(pool_id);
        if (tinfo &&
            static_cast<int>(tinfo->persistence_level_) < target_level) {
          has_volatile_blocks = true;
          break;
        }
      }

      if (has_volatile_blocks) {
        FlushEntry entry;
        entry.composite_key = key;
        entry.blob_name = blob_info.blob_name_.str();
        entry.total_size = blob_info.GetTotalSize();
        entry.score = blob_info.score_;

        // Parse tag_id from composite key: "major.minor.blob_name"
        size_t first_dot = key.find('.');
        size_t second_dot = key.find('.', first_dot + 1);
        if (first_dot != std::string::npos && second_dot != std::string::npos) {
          entry.tag_id.major_ =
              static_cast<chi::u32>(std::stoul(key.substr(0, first_dot)));
          entry.tag_id.minor_ = static_cast<chi::u32>(std::stoul(
              key.substr(first_dot + 1, second_dot - first_dot - 1)));
        }

        blobs_to_flush.push_back(std::move(entry));
      }
    });
  }

  HLOG(kDebug, "FlushData: Found {} blobs with volatile blocks to flush",
       blobs_to_flush.size());

  // Flush each blob: read data, free volatile blocks, re-put with persistence
  for (const auto &entry : blobs_to_flush) {
    BlobInfo *blob_info_ptr = tag_blob_name_to_info_.find(entry.composite_key);
    if (!blob_info_ptr || blob_info_ptr->blocks_.empty()) continue;

    chi::u64 total_size = entry.total_size;
    if (total_size == 0) continue;

    // Step 1: Allocate buffer and read data from current blocks
    auto *ipc_manager = CLIO_IPC;
    ctp::ipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(total_size);
    if (buffer.IsNull()) {
      HLOG(kError,
           "FlushData: Failed to allocate buffer of size {} for blob {}",
           total_size, entry.blob_name);
      continue;
    }

    ctp::ipc::ShmPtr<> shm_ptr(buffer.shm_);
    chi::u32 read_error = 0;
    CLIO_CO_AWAIT(ReadData(blob_info_ptr->blocks_, shm_ptr, total_size, 0,
                      read_error));
    if (read_error != 0) {
      HLOG(kError, "FlushData: Failed to read blob data for {}",
           entry.blob_name);
      ipc_manager->FreeBuffer(buffer);
      continue;
    }

    // Step 2: Free only volatile blocks
    chi::priv::vector<BlobBlock> nonvolatile_blocks(CLIO_PRIV_ALLOC);
    std::unordered_map<
        chi::PoolId,
        std::pair<chi::PoolQuery, std::vector<clio::run::bdev::Block>>>
        volatile_blocks_by_pool;

    {
      chi::ScopedCoRwReadLock read_lock(target_lock_);
      for (const auto &block : blob_info_ptr->blocks_) {
        chi::PoolId pool_id = block.bdev_client_.pool_id_;
        TargetInfo *tinfo = registered_targets_.find(pool_id);
        if (tinfo &&
            static_cast<int>(tinfo->persistence_level_) < target_level) {
          // Volatile block - collect for freeing
          clio::run::bdev::Block bdev_block;
          bdev_block.offset_ = block.target_offset_;
          bdev_block.size_ = block.size_;
          bdev_block.block_type_ = 0;
          if (volatile_blocks_by_pool.find(pool_id) ==
              volatile_blocks_by_pool.end()) {
            volatile_blocks_by_pool[pool_id] = std::make_pair(
                block.target_query_, std::vector<clio::run::bdev::Block>());
          }
          volatile_blocks_by_pool[pool_id].second.push_back(bdev_block);
        } else {
          // Nonvolatile block - keep
          nonvolatile_blocks.push_back(block);
        }
      }
    }

    // Free volatile blocks from bdevs
    for (const auto &pool_entry : volatile_blocks_by_pool) {
      const chi::PoolId &pool_id = pool_entry.first;
      const chi::PoolQuery &target_query = pool_entry.second.first;
      const std::vector<clio::run::bdev::Block> &blocks =
          pool_entry.second.second;

      chi::u64 bytes_freed = 0;
      for (const auto &block : blocks) {
        bytes_freed += block.size_;
      }

      clio::run::bdev::Client bdev_client(pool_id);
      auto free_task = bdev_client.AsyncFreeBlocks(target_query, blocks);
      CLIO_CO_AWAIT(free_task);
      if (free_task->GetReturnCode() == 0) {
        chi::ScopedCoRwWriteLock write_lock(target_lock_);
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
    CLIO_CO_AWAIT(put_task);

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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
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
      tag_info.total_size_ = total_size;
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
        bool is_volatile = false;
        {
          chi::ScopedCoRwReadLock read_lock(target_lock_);
          TargetInfo *tinfo = registered_targets_.find(bdev_pool_id);
          if (tinfo && tinfo->persistence_level_ ==
                           clio::run::bdev::PersistenceLevel::kVolatile) {
            is_volatile = true;
          }
        }
        if (is_volatile) {
          continue;  // Volatile data is lost on restart
        }

        // Reconstruct block
        clio::run::bdev::Client bdev_client(bdev_pool_id);
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
  for (size_t i = 0;; ++i) {
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
  for (size_t i = 0;; ++i) {
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
        BlobInfo *blob_info_ptr = tag_blob_name_to_info_.find(composite_key);
        if (blob_info_ptr) {
          // Replace blocks with replayed blocks (full replacement semantics)
          blob_info_ptr->blocks_.clear();
          for (const auto &tb : txn.new_blocks_) {
            chi::PoolId bdev_pool_id(tb.bdev_major_, tb.bdev_minor_);
            // Filter volatile targets (matching RestoreMetadataFromLog)
            bool is_volatile = false;
            {
              chi::ScopedCoRwReadLock read_lock(target_lock_);
              TargetInfo *tinfo = registered_targets_.find(bdev_pool_id);
              if (tinfo && tinfo->persistence_level_ ==
                               clio::run::bdev::PersistenceLevel::kVolatile) {
                is_volatile = true;
              }
            }
            if (is_volatile) {
              continue;
            }
            clio::run::bdev::Client bdev_client(bdev_pool_id);
            BlobBlock block(bdev_client, tb.target_query_, tb.target_offset_,
                            tb.size_);
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
        BlobInfo *blob_info_ptr = tag_blob_name_to_info_.find(composite_key);
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
    tag_info.total_size_ = total;
  });

  // Phase 4: Update next_tag_id_minor_
  chi::u32 current_minor = next_tag_id_minor_.load();
  if (max_minor > current_minor) {
    next_tag_id_minor_.store(max_minor);
  }

  HLOG(kInfo, "ReplayTransactionLogs: Replayed {} tag ops and {} blob ops",
       tags_replayed, blobs_replayed);
}

// GetWorkRemaining implementation (required pure virtual method)
chi::u64 Runtime::GetWorkRemaining() const {
  // Return approximate work remaining (simple implementation)
  // In a real implementation, this would sum tasks across all queues
  return 0;  // For now, always return 0 work remaining
}

chi::TaskStat Runtime::GetTaskStats(const chi::Task *task) const {
  if (!task) return chi::TaskStat();
  switch (task->method_) {
    case Method::kPutBlob: {
      auto *t = static_cast<const PutBlobTask *>(task);
      chi::TaskStat stat;
      stat.io_size_ = t->size_;
      // Rough wall-time estimate at ~500 MB/s for routing decisions only.
      // The learned model in InferWallClockTime adjusts the coefficient
      // over time; this is just the initial seed.
      stat.wall_time_ =
          static_cast<float>(t->size_) / 500.0f;
      return stat;
    }
    case Method::kGetBlob: {
      auto *t = static_cast<const GetBlobTask *>(task);
      chi::TaskStat stat;
      stat.io_size_ = t->size_;
      stat.wall_time_ =
          static_cast<float>(t->size_) / 500.0f;
      return stat;
    }
    default:
      return chi::TaskStat();
  }
}

// Helper methods for lock index calculation
size_t Runtime::GetTargetLockIndex(const chi::PoolId &target_id) const {
  // Use hash of target_id to distribute locks evenly
  std::hash<chi::PoolId> hasher;
  return hasher(target_id) % target_locks_.size();
}

size_t Runtime::GetTagLockIndex(const std::string &tag_name) const {
  // Use same hash function as ctp::priv::unordered_map_ll to ensure lock maps
  // to same bucket
  std::hash<std::string> hasher;
  return hasher(tag_name) % tag_locks_.size();
}

size_t Runtime::GetTagLockIndex(const TagId &tag_id) const {
  // Use same hash function as ctp::priv::unordered_map_ll for TagId keys
  // std::hash<chi::UniqueId> is defined in types.h
  std::hash<TagId> hasher;
  return hasher(tag_id) % tag_locks_.size();
}

TagId Runtime::GenerateNewTagId() {
  // Get node_id from IPC manager as the major component
  auto *ipc_manager = CLIO_IPC;
  chi::u32 node_id = ipc_manager->GetNodeId();

  // Get next minor component from atomic counter
  chi::u32 minor_id = next_tag_id_minor_.fetch_add(1);

  return TagId{node_id, minor_id};
}

// Explicit template instantiations for required template methods
template chi::TaskResume Runtime::GetOrCreateTag<CreateParams>(
    ctp::ipc::FullPtr<GetOrCreateTagTask<CreateParams>> task, chi::RunContext &ctx);

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

  // Acquire read lock for map lookup (single lock for map-wide safety)
  chi::ScopedCoRwReadLock lock(blob_map_lock_);

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

  // Acquire write lock for map insertion (single lock for map-wide safety)
  BlobInfo *blob_info_ptr = nullptr;
  {
    chi::ScopedCoRwWriteLock lock(blob_map_lock_);

    // Store blob info directly in tag_blob_name_to_info_
    auto insert_result =
        tag_blob_name_to_info_.insert_or_assign(composite_key, new_blob_info);
    blob_info_ptr = insert_result.value;
  }  // Release lock immediately after insertion

  // WAL: log blob creation
  if (!blob_txn_logs_.empty()) {
    chi::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
    TxnCreateNewBlob txn;
    txn.tag_major_ = tag_id.major_;
    txn.tag_minor_ = tag_id.minor_;
    txn.blob_name_ = blob_name;
    txn.score_ = blob_score;
    blob_txn_logs_[wid % blob_txn_logs_.size()]->Log(TxnType::kCreateNewBlob,
                                                     txn);
  }

  return blob_info_ptr;
}

chi::TaskResume Runtime::ExtendBlob(BlobInfo &blob_info, chi::u64 offset,
                                    chi::u64 size, float blob_score,
                                    chi::u32 &error_code,
                                    int min_persistence_level) {
#ifdef __NVCOMPILER
  thread_local chi::RunContext _fb_rctx;
  chi::RunContext* _fp = chi::GetCurrentRunContextFromWorker();
  chi::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  // Calculate required additional space
  chi::u64 current_blob_size = blob_info.GetTotalSize();
  chi::u64 required_size = offset + size;

  if (required_size <= current_blob_size) {
    // No additional allocation needed
    error_code = 0;
    CLIO_CO_RETURN;
  }

  chi::u64 additional_size = required_size - current_blob_size;

  // Snapshot available targets for the DPE. target_list_ is the contiguous
  // mirror of registered_targets_ — copying it under the read lock is O(N_live)
  // with no map iteration over empty slots.
  std::vector<TargetInfo> available_targets;
  {
    chi::ScopedCoRwReadLock read_lock(target_lock_);
    available_targets = target_list_;
  }
  if (available_targets.empty()) {
    error_code = 1;
    CLIO_CO_RETURN;
  }

  // Use cached Data Placement Engine (built once in Create() from config)
  std::vector<TargetInfo> ordered_targets =
      dpe_->SelectTargets(available_targets, blob_score, additional_size);

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
    CLIO_CO_RETURN;
  }

  // Allocate from pre-selected targets in order
  chi::u64 remaining_to_allocate = additional_size;
  for (const auto &selected_target_info : ordered_targets) {
    if (remaining_to_allocate == 0) {
      break;
    }

    chi::PoolId selected_target_id = selected_target_info.bdev_client_.pool_id_;

    // Copy target info under lock (can't hold lock across co_await)
    TargetInfo target_info_copy;
    bool found = false;
    {
      chi::ScopedCoRwReadLock read_lock(target_lock_);
      TargetInfo *target_info = registered_targets_.find(selected_target_id);
      if (target_info != nullptr) {
        target_info_copy = *target_info;
        found = true;
      }
    }
    if (!found) {
      continue;
    }

    // Calculate how much we can allocate from this target
    chi::u64 allocate_size =
        std::min(remaining_to_allocate, target_info_copy.remaining_space_);

    if (allocate_size == 0) {
      continue;
    }

    // Allocate space using bdev client
    chi::u64 allocated_offset;
    bool alloc_success = false;
    CLIO_CO_AWAIT(AllocateFromTarget(target_info_copy, allocate_size,
                                allocated_offset, alloc_success));
    if (!alloc_success) {
      // Allocation failed, try next target
      continue;
    }

    // Create new block for the allocated space
    BlobBlock new_block(target_info_copy.bdev_client_,
                        target_info_copy.target_query_, allocated_offset,
                        allocate_size);
    blob_info.blocks_.push_back(new_block);

    // Debit the CANONICAL target's remaining_space_ (mirror of
    // FreeAllBlobBlocks' credit). AllocateFromTarget only decremented the
    // throwaway target_info_copy, so without this allocs never reduced
    // the real counter and accounting drifted between StatTargets polls.
    //
    // registered_targets_ is structurally STATIONARY on the data path
    // (only RegisterTarget inserts, at setup), so a shared READ lock is
    // sufficient to traverse/find it — no exclusive write lock for a
    // plain integer update. The counter is mutated lock-free via
    // ctp::ipc::atomic_ref with a CAS loop that saturates at 0 instead
    // of underflowing the unsigned value.
    {
      chi::ScopedCoRwReadLock read_lock(target_lock_);
      TargetInfo *ti = registered_targets_.find(selected_target_id);
      if (ti != nullptr) {
        ctp::ipc::atomic_ref<chi::u64> rs(ti->remaining_space_);
        chi::u64 cur = rs.load(std::memory_order_relaxed);
        while (!rs.compare_exchange_weak(
            cur, (cur > allocate_size) ? cur - allocate_size : 0,
            std::memory_order_relaxed)) {
        }
      }
    }

    remaining_to_allocate -= allocate_size;
  }

  // Error condition: if we've exhausted all targets but still have remaining
  // space
  if (remaining_to_allocate > 0) {
    error_code = 3;
    CLIO_CO_RETURN;
  }

  error_code = 0;  // Success
  CLIO_CO_RETURN;
}

chi::TaskResume Runtime::ModifyExistingData(
    const chi::priv::vector<BlobBlock> &blocks, ctp::ipc::ShmPtr<> data, size_t data_size,
    size_t data_offset_in_blob, chi::u32 &error_code) {
#ifdef __NVCOMPILER
  thread_local chi::RunContext _fb_rctx;
  chi::RunContext* _fp = chi::GetCurrentRunContextFromWorker();
  chi::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  HLOG(kDebug,
       "ModifyExistingData: blocks={}, data_size={}, data_offset_in_blob={}",
       blocks.size(), data_size, data_offset_in_blob);

  static thread_local size_t mod_count = 0;
  static thread_local double t_setup_ms = 0, t_vec_alloc_ms = 0;
  static thread_local double t_async_send_ms = 0, t_co_await_ms = 0;
  ctp::Timer timer;

  // Step 1: Initially store the remaining_size equal to data_size
  size_t remaining_size = data_size;

  // Vector to store async write tasks for later waiting
  std::vector<chi::Future<clio::run::bdev::WriteTask>> write_tasks;
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

      clio::run::bdev::Block bdev_block(
          block.target_offset_ + write_start_in_block, write_size, 0);
      ctp::ipc::ShmPtr<> data_ptr = data + data_buffer_offset;
      timer.Pause();
      t_setup_ms += timer.GetMsec();
      timer.Reset();

      // Wrap single block in chi::priv::vector for AsyncWrite
      timer.Resume();
      chi::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
      blocks.push_back(bdev_block);
      timer.Pause();
      t_vec_alloc_ms += timer.GetMsec();
      timer.Reset();

      // Create and send the async write task
      timer.Resume();
      clio::run::bdev::Client cte_clientcopy = block.bdev_client_;
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
    CLIO_CO_AWAIT(task);
    if (task->bytes_written_ != expected_size) {
      error_code = 1;
      CLIO_CO_RETURN;
    }
  }
  timer.Pause();
  t_co_await_ms += timer.GetMsec();
  timer.Reset();

  ++mod_count;
  if (mod_count % 100 == 0) {
    HLOG(kDebug,
         "[ModifyExistingData] ops={} setup={:.3f} ms vec_alloc={:.3f} ms "
         "async_send={:.3f} ms co_await={:.3f} ms",
         mod_count, t_setup_ms, t_vec_alloc_ms, t_async_send_ms, t_co_await_ms);
    t_setup_ms = t_vec_alloc_ms = t_async_send_ms = t_co_await_ms = 0;
  }

  error_code = 0;  // Success
  CLIO_CO_RETURN;
}

chi::TaskResume Runtime::ReadData(const chi::priv::vector<BlobBlock> &blocks,
                                  ctp::ipc::ShmPtr<> data, size_t data_size,
                                  size_t data_offset_in_blob,
                                  chi::u32 &error_code) {
#ifdef __NVCOMPILER
  thread_local chi::RunContext _fb_rctx;
  chi::RunContext* _fp = chi::GetCurrentRunContextFromWorker();
  chi::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  HLOG(kDebug, "ReadData: blocks={}, data_size={}, data_offset_in_blob={}",
       blocks.size(), data_size, data_offset_in_blob);

  // Step 1: Initially store the remaining_size equal to data_size
  size_t remaining_size = data_size;

  // Vector to store async read tasks for later waiting
  std::vector<chi::Future<clio::run::bdev::ReadTask>> read_tasks;
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
      clio::run::bdev::Block bdev_block(
          block.target_offset_ + read_start_in_block, read_size, 0);
      ctp::ipc::ShmPtr<> data_ptr = data + data_buffer_offset;

      // Wrap single block in chi::priv::vector for AsyncRead
      chi::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
      blocks.push_back(bdev_block);

      clio::run::bdev::Client cte_clientcopy = block.bdev_client_;
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

    CLIO_CO_AWAIT(task);

    HLOG(kDebug,
         "ReadData: task[{}] completed - bytes_read={}, expected={}, status={}",
         task_idx, task->bytes_read_, expected_size,
         (task->bytes_read_ == expected_size ? "SUCCESS" : "FAILED"));

    if (task->bytes_read_ != expected_size) {
      HLOG(kError,
           "ReadData: READ FAILED - task[{}] read {} bytes, expected {}",
           task_idx, task->bytes_read_, expected_size);
      // Wait for all remaining in-flight tasks before returning to avoid
      // use-after-free when buffers are freed by the caller.
      for (size_t j = task_idx + 1; j < read_tasks.size(); ++j) {
        co_await read_tasks[j];
      }
      error_code = 1;
      CLIO_CO_RETURN;
    }
  }

  HLOG(kDebug, "ReadData: All read tasks completed successfully");
  error_code = 0;  // Success
  CLIO_CO_RETURN;
}

// Block management helper functions

chi::TaskResume Runtime::AllocateFromTarget(TargetInfo &target_info,
                                            chi::u64 size,
                                            chi::u64 &allocated_offset,
                                            bool &success) {
#ifdef __NVCOMPILER
  thread_local chi::RunContext _fb_rctx;
  chi::RunContext* _fp = chi::GetCurrentRunContextFromWorker();
  chi::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  HLOG(kDebug,
       "AllocateFromTarget: ENTER - target_name={}, "
       "bdev_client_.pool_id_=({},{}), size={}, remaining_space={}",
       target_info.target_name_.data(), target_info.bdev_client_.pool_id_.major_,
       target_info.bdev_client_.pool_id_.minor_, size,
       target_info.remaining_space_);

  // Check if target has sufficient space
  if (target_info.remaining_space_ < size) {
    HLOG(kDebug,
         "AllocateFromTarget: Insufficient space - remaining={} < size={}",
         target_info.remaining_space_, size);
    success = false;
    CLIO_CO_RETURN;
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

    CLIO_CO_AWAIT(alloc_task);

    HLOG(kDebug,
         "AllocateFromTarget: co_await complete, "
         "alloc_task->blocks_.size()={}, return_code={}",
         alloc_task->blocks_.size(), alloc_task->return_code_.load());

    std::vector<clio::run::bdev::Block> allocated_blocks;
    for (size_t i = 0; i < alloc_task->blocks_.size(); ++i) {
      allocated_blocks.push_back(alloc_task->blocks_[i]);
    }

    // Check if we got any blocks
    if (allocated_blocks.empty()) {
      HLOG(kDebug, "AllocateFromTarget: FAILED - allocated_blocks is empty");
      success = false;
      CLIO_CO_RETURN;
    }

    // Use the first block (for single allocation case)
    clio::run::bdev::Block allocated_block = allocated_blocks[0];
    allocated_offset = allocated_block.offset_;

    // Update remaining space
    target_info.remaining_space_ -= size;
    // HLOG(kInfo,
    //       "Allocated from target {}: offset={}, size={} remaining_space={}",
    //       target_info.target_name_, allocated_offset, size,
    //       target_info.remaining_space_);

    success = true;
    CLIO_CO_RETURN;
  } catch (const std::exception &e) {
    // Allocation failed
    success = false;
    CLIO_CO_RETURN;
  }
}

chi::TaskResume Runtime::ClearBlob(BlobInfo &blob_info, float blob_score,
                                   chi::u64 offset, chi::u64 size,
                                   bool &cleared) {
#ifdef __NVCOMPILER
  thread_local chi::RunContext _fb_rctx;
  chi::RunContext* _fp = chi::GetCurrentRunContextFromWorker();
  chi::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  cleared = false;
  // Score must be in [0, 1]
  if (blob_score < 0.0f || blob_score > 1.0f) {
    CLIO_CO_RETURN;
  }
  // Must be full-blob replacement (offset == 0 with non-empty blob)
  chi::u64 current_size = blob_info.GetTotalSize();
  if (offset != 0 || current_size == 0) {
    CLIO_CO_RETURN;
  }
  // Free all existing blocks
  chi::u32 free_result = 0;
  CLIO_CO_AWAIT(FreeAllBlobBlocks(blob_info, free_result));
  if (free_result == 0) {
    cleared = true;
  }
  CLIO_CO_RETURN;
}

chi::TaskResume Runtime::FreeAllBlobBlocks(BlobInfo &blob_info,
                                           chi::u32 &error_code) {
#ifdef __NVCOMPILER
  thread_local chi::RunContext _fb_rctx;
  chi::RunContext* _fp = chi::GetCurrentRunContextFromWorker();
  chi::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  // Map: PoolId -> (target_query, vector<Block>)
  std::unordered_map<chi::PoolId, std::pair<chi::PoolQuery,
                                            std::vector<clio::run::bdev::Block>>>
      blocks_by_pool;

  // Group blocks by PoolId
  for (const auto &blob_block : blob_info.blocks_) {
    chi::PoolId pool_id = blob_block.bdev_client_.pool_id_;
    clio::run::bdev::Block block;
    block.offset_ = blob_block.target_offset_;
    block.size_ = blob_block.size_;
    // BlobBlock does not track the allocator's size class; bdev
    // Runtime::FreeBlocks re-derives block_type_ from size_ so the block
    // returns to the same partition AllocateBlock draws from. Leave 0.
    block.block_type_ = 0;

    // Store target_query with blocks for this pool
    if (blocks_by_pool.find(pool_id) == blocks_by_pool.end()) {
      blocks_by_pool[pool_id] = std::make_pair(
          blob_block.target_query_, std::vector<clio::run::bdev::Block>());
    }
    blocks_by_pool[pool_id].second.push_back(block);
  }

  // Call FreeBlocks once per PoolId and update target capacities
  for (const auto &pool_entry : blocks_by_pool) {
    const chi::PoolId &pool_id = pool_entry.first;
    const chi::PoolQuery &target_query = pool_entry.second.first;
    const std::vector<clio::run::bdev::Block> &blocks = pool_entry.second.second;

    // Calculate total bytes to be freed for this pool
    chi::u64 bytes_freed = 0;
    for (const auto &block : blocks) {
      bytes_freed += block.size_;
    }

    // Get bdev client for this pool from first blob block
    clio::run::bdev::Client bdev_client(pool_id);
    auto free_task = bdev_client.AsyncFreeBlocks(target_query, blocks);
    CLIO_CO_AWAIT(free_task);
    chi::u32 free_result = free_task->GetReturnCode();
    if (free_result != 0) {
      HLOG(kWarning, "Failed to free blocks from pool {}", pool_id.major_);
    } else {
      // Successfully freed blocks - credit target's remaining_space_.
      // Shared READ lock only: registered_targets_ is structurally
      // stationary on the data path; the counter is bumped lock-free
      // via ctp::ipc::atomic_ref (no exclusive lock for an integer add).
      chi::ScopedCoRwReadLock read_lock(target_lock_);
      TargetInfo *target_info = registered_targets_.find(pool_id);
      if (target_info != nullptr) {
        chi::u64 now =
            ctp::ipc::atomic_ref<chi::u64>(target_info->remaining_space_)
                .fetch_add(bytes_freed, std::memory_order_relaxed) +
            bytes_freed;
        HLOG(kDebug, "Updated target {} remaining_space_ by +{} bytes (now {})",
             pool_id.major_, bytes_freed, now);
      }
    }
  }

  // Clear all blocks
  blob_info.blocks_.clear();
  error_code = 0;
  CLIO_CO_RETURN;
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
    ctp::ipc::FullPtr<PollTelemetryLogTask> task, chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::GetBlobScore(ctp::ipc::FullPtr<GetBlobScoreTask> task,
                                      chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);

    if (blob_info_ptr == nullptr) {
      task->return_code_ = 1;  // Blob not found
      CLIO_CO_RETURN;
    }

    // Step 2: Return the blob score
    task->score_ = blob_info_ptr->score_;

    // Step 3: Update timestamps and log telemetry
    auto now = GetCurrentTimeNs();
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::GetBlobSize(ctp::ipc::FullPtr<GetBlobSizeTask> task,
                                     chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      task->return_code_ = 1;  // Blob not found
      CLIO_CO_RETURN;
    }

    // Step 2: Calculate and return the blob size
    task->size_ = blob_info_ptr->GetTotalSize();

    // Step 3: Update timestamps and log telemetry
    auto now = GetCurrentTimeNs();
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::GetBlobInfo(ctp::ipc::FullPtr<GetBlobInfoTask> task,
                                     chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();

    // Validate that blob_name is provided
    if (blob_name.empty()) {
      task->return_code_ = 1;  // Error: empty blob name
      CLIO_CO_RETURN;
    }

    // Step 1: Check if blob exists
    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      task->return_code_ = 2;  // Blob not found
      CLIO_CO_RETURN;
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
    auto now = GetCurrentTimeNs();
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::GetContainedBlobs(
    ctp::ipc::FullPtr<GetContainedBlobsTask> task, chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Extract input parameters
    TagId tag_id = task->tag_id_;

    // Validate tag exists
    TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
    if (tag_info_ptr == nullptr) {
      task->return_code_ = 1;  // Tag not found
      CLIO_CO_RETURN;
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
                 GetCurrentTimeNs(),
                 GetCurrentTimeNs());

    HLOG(kDebug, "GetContainedBlobs successful: tag_id={},{}, found {} blobs",
         tag_id.major_, tag_id.minor_, task->blob_names_.size());

  } catch (const std::exception &e) {
    task->return_code_ = 1;  // Error during operation
    HLOG(kError, "GetContainedBlobs failed: {}", e.what());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::TagQuery(ctp::ipc::FullPtr<TagQueryTask> task,
                                  chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::BlobQuery(ctp::ipc::FullPtr<BlobQueryTask> task,
                                   chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ==============================================================================
// SemanticSearch — BM25 over blob contents
// ==============================================================================

namespace {

// Tokenize raw bytes as lowercase alphanumeric runs of length >= 2.
// Anything else (whitespace, punctuation, non-ASCII) splits a token.
// Deliberately simple; good enough for English-ish text from labels.
std::vector<std::string> SemSearchTokenize(const char *data, size_t size) {
  std::vector<std::string> tokens;
  std::string cur;
  cur.reserve(16);
  for (size_t i = 0; i < size; ++i) {
    unsigned char c = static_cast<unsigned char>(data[i]);
    if (std::isalnum(c)) {
      cur.push_back(static_cast<char>(std::tolower(c)));
    } else {
      if (cur.size() >= 2) tokens.push_back(std::move(cur));
      cur.clear();
    }
  }
  if (cur.size() >= 2) tokens.push_back(std::move(cur));
  return tokens;
}

struct SemSearchDoc {
  TagId tag_id;
  std::string tag_name;
  std::string blob_name;
  std::unordered_map<std::string, int> tf;
  size_t length;
};

}  // namespace

chi::TaskResume Runtime::SemanticSearch(
    ctp::ipc::FullPtr<SemanticSearchTask> task, chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext &rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  task->results_.clear();
  task->return_code_ = 0;

  std::string tag_regex_str = task->tag_regex_.str();
  std::string blob_regex_str = task->blob_regex_.str();
  std::string query_text = task->query_text_.str();
  chi::u32 k = task->k_;

  std::regex tag_pattern;
  std::regex blob_pattern;
  try {
    tag_pattern = std::regex(tag_regex_str);
    blob_pattern = std::regex(blob_regex_str);
  } catch (const std::regex_error &e) {
    HLOG(kError, "SemanticSearch: bad regex (tag='{}' blob='{}'): {}",
         tag_regex_str, blob_regex_str, e.what());
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  // Step 1: pick matching (tag_name, tag_id) pairs from tag metadata
  // (same iteration as BlobQuery).
  std::vector<std::pair<std::string, TagId>> matching_tags;
  tag_name_to_id_.for_each(
      [&tag_pattern, &matching_tags](const std::string &tag_name,
                                     const TagId &tag_id) {
        if (std::regex_match(tag_name, tag_pattern)) {
          matching_tags.emplace_back(tag_name, tag_id);
        }
      });

  // Step 2: walk the tag-blob metadata and collect (tag_id, tag_name,
  // blob_name) triples whose blob_name matches blob_regex_. We collect names
  // first and read bytes afterward — keeping the metadata iteration short
  // means the for_each lambda doesn't block on bdev I/O.
  struct Candidate { TagId tag_id; std::string tag_name; std::string blob_name; };
  std::vector<Candidate> candidates;
  for (const auto &tn : matching_tags) {
    const std::string &tag_name = tn.first;
    const TagId &tag_id = tn.second;
    std::string prefix = std::to_string(tag_id.major_) + "." +
                         std::to_string(tag_id.minor_) + ".";
    tag_blob_name_to_info_.for_each(
        [&prefix, &blob_pattern, &tag_id, &tag_name, &candidates](
            const std::string &composite_key, const BlobInfo &blob_info) {
          (void)blob_info;
          if (composite_key.rfind(prefix, 0) != 0) return;
          std::string blob_name = composite_key.substr(prefix.length());
          if (std::regex_match(blob_name, blob_pattern)) {
            candidates.push_back({tag_id, tag_name, blob_name});
          }
        });
  }
  if (candidates.empty()) {
    HLOG(kDebug,
         "SemanticSearch: no candidates for tag='{}' blob='{}'",
         tag_regex_str, blob_regex_str);
    CLIO_CO_RETURN;
  }

  // Step 3: read each candidate's bytes, tokenize, and accumulate
  // BM25 corpus statistics (tf per doc, doc length, df, avgdl).
  auto *ipc_manager = CLIO_IPC;
  std::vector<SemSearchDoc> docs;
  docs.reserve(candidates.size());
  for (auto &cand : candidates) {
    BlobInfo *info = CheckBlobExists(cand.blob_name, cand.tag_id);
    if (info == nullptr) continue;
    chi::u64 total = info->GetTotalSize();
    if (total == 0) continue;

    ctp::ipc::FullPtr<char> buf = ipc_manager->AllocateBuffer(total);
    if (buf.IsNull()) {
      HLOG(kWarning,
           "SemanticSearch: AllocateBuffer({}) failed for blob '{}'; "
           "skipping",
           total, cand.blob_name);
      continue;
    }
    ctp::ipc::ShmPtr<> shm(buf.shm_);
    chi::u32 read_rc = 0;
    CLIO_CO_AWAIT(ReadData(info->blocks_, shm, total, 0, read_rc));
    if (read_rc != 0) {
      HLOG(kWarning,
           "SemanticSearch: ReadData failed for blob '{}' (rc={}); "
           "skipping",
           cand.blob_name, read_rc);
      ipc_manager->FreeBuffer(buf);
      continue;
    }

    auto tokens = SemSearchTokenize(buf.ptr_, total);
    ipc_manager->FreeBuffer(buf);

    SemSearchDoc doc;
    doc.tag_id = cand.tag_id;
    doc.tag_name = std::move(cand.tag_name);
    doc.blob_name = std::move(cand.blob_name);
    doc.length = tokens.size();
    for (auto &t : tokens) ++doc.tf[t];
    docs.push_back(std::move(doc));
  }
  if (docs.empty()) {
    CLIO_CO_RETURN;
  }

  // Step 4: BM25. Corpus stats are computed over the working set
  // (the matched slice), not over all CTE blobs — this is "rank
  // within this regex" semantics. k1=1.5 / b=0.75 are the standard
  // Okapi defaults.
  constexpr double kK1 = 1.5;
  constexpr double kB = 0.75;
  std::unordered_map<std::string, int> df;
  double total_len = 0.0;
  for (auto &d : docs) {
    total_len += static_cast<double>(d.length);
    for (auto &kv : d.tf) df[kv.first]++;
  }
  double avgdl = (docs.empty() ? 1.0 : total_len / docs.size());
  if (avgdl <= 0.0) avgdl = 1.0;
  const size_t N = docs.size();

  auto qtokens = SemSearchTokenize(query_text.data(), query_text.size());
  std::unordered_set<std::string> uniq_q(qtokens.begin(), qtokens.end());

  std::vector<SemanticSearchResult> scored;
  scored.reserve(docs.size());
  for (auto &d : docs) {
    double score = 0.0;
    for (auto &q : uniq_q) {
      auto df_it = df.find(q);
      if (df_it == df.end()) continue;
      auto tf_it = d.tf.find(q);
      if (tf_it == d.tf.end()) continue;
      double df_q = static_cast<double>(df_it->second);
      double idf = std::log((static_cast<double>(N) - df_q + 0.5) /
                                (df_q + 0.5) +
                            1.0);
      double tf_q = static_cast<double>(tf_it->second);
      double norm =
          1.0 - kB + kB * (static_cast<double>(d.length) / avgdl);
      score += idf * (tf_q * (kK1 + 1.0)) / (tf_q + kK1 * norm);
    }
    scored.emplace_back(d.tag_id, d.tag_name, d.blob_name, score);
  }

  std::sort(scored.begin(), scored.end(),
            [](const SemanticSearchResult &a,
               const SemanticSearchResult &b) { return a.score_ > b.score_; });
  if (k > 0 && scored.size() > k) scored.resize(k);
  task->results_ = std::move(scored);
  HLOG(kDebug,
       "SemanticSearch: tag='{}' blob='{}' query='{}' -> {} results "
       "(from {} candidates)",
       tag_regex_str, blob_regex_str, query_text, task->results_.size(),
       candidates.size());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ==============================================================================
// TemporalSearch — timestamp-window scan over blob metadata
// ==============================================================================

chi::TaskResume Runtime::TemporalSearch(
    ctp::ipc::FullPtr<TemporalSearchTask> task, chi::RunContext &ctx) {
#ifdef __NVCOMPILER
  chi::RunContext &rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  task->results_.clear();
  task->return_code_ = 0;

  std::string tag_regex_str = task->tag_regex_.str();
  std::string blob_regex_str = task->blob_regex_.str();
  Timestamp time_begin = task->time_begin_;
  Timestamp time_end = task->time_end_;
  chi::u32 max_entries = task->max_entries_;

  std::regex tag_pattern;
  std::regex blob_pattern;
  try {
    tag_pattern = std::regex(tag_regex_str);
    blob_pattern = std::regex(blob_regex_str);
  } catch (const std::regex_error &e) {
    HLOG(kError, "TemporalSearch: bad regex (tag='{}' blob='{}'): {}",
         tag_regex_str, blob_regex_str, e.what());
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }

  // Step 1: collect matching tags (same as BlobQuery / SemanticSearch).
  std::vector<std::pair<std::string, TagId>> matching_tags;
  tag_name_to_id_.for_each(
      [&tag_pattern, &matching_tags](const std::string &tag_name,
                                     const TagId &tag_id) {
        if (std::regex_match(tag_name, tag_pattern)) {
          matching_tags.emplace_back(tag_name, tag_id);
        }
      });

  // Step 2: scan blob metadata; filter by blob regex and time window.
  // last_modified_ == 0 means the blob has never been written and is
  // excluded from all time-range queries.
  std::vector<TemporalSearchResult> hits;
  for (const auto &tn : matching_tags) {
    const std::string &tag_name = tn.first;
    const TagId &tag_id = tn.second;
    std::string prefix = std::to_string(tag_id.major_) + "." +
                         std::to_string(tag_id.minor_) + ".";
    tag_blob_name_to_info_.for_each(
        [&](const std::string &composite_key, const BlobInfo &blob_info) {
          if (composite_key.rfind(prefix, 0) != 0) return;
          std::string blob_name = composite_key.substr(prefix.length());
          if (!std::regex_match(blob_name, blob_pattern)) return;
          Timestamp ts = blob_info.last_modified_;
          if (ts == 0) return;
          if (time_begin != 0 && ts < time_begin) return;
          if (time_end != 0 && ts > time_end) return;
          hits.emplace_back(tag_id, tag_name, blob_name, ts);
        });
  }

  std::sort(hits.begin(), hits.end(),
            [](const TemporalSearchResult &a, const TemporalSearchResult &b) {
              return a.last_modified_ < b.last_modified_;
            });
  if (max_entries > 0 && hits.size() > max_entries)
    hits.resize(max_entries);
  task->results_ = std::move(hits);
  HLOG(kDebug,
       "TemporalSearch: tag='{}' blob='{}' [{}, {}] max={} -> {} results",
       tag_regex_str, blob_regex_str, time_begin, time_end, max_entries,
       task->results_.size());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
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

chi::TaskResume Runtime::Monitor(ctp::ipc::FullPtr<MonitorTask> task,
                                 chi::RunContext &rctx) {
  task->SetReturnCode(0);
  (void)rctx;
  CLIO_CO_RETURN;
}

// =====================================================================
// GPU metadata cache helpers
// ---------------------------------------------------------------------
// These helpers are the ONLY places that mutate the GPU cache. Methods
// like PutBlob / DelBlob / GetOrCreateTag / DelTag stay free of cache-
// management noise — they invoke the matching GpuCacheOn* helper and
// move on. The cache lives in managed/shared USM, so calls to the
// inline GpuCacheUpsert* / GpuCacheRemove* primitives in
// gpu_metadata_cache.h work directly from the host. A pure-device-
// memory variant (one-WI kernel per mutation) is a future extension.
// =====================================================================

bool Runtime::GpuCacheCreate() {
  if (!config_.gpu_metadata_cache_.enabled_) {
    gpu_cache_ = nullptr;
    gpu_cache_bytes_ = 0;
    return true;
  }

#if !(CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL)
  HLOG(kWarning,
       "GpuMetadataCache: enabled in config, but no GPU backend was built "
       "in. Cache will not be allocated.");
  gpu_cache_ = nullptr;
  gpu_cache_bytes_ = 0;
  return false;
#else
  // Cap the slot counts at what the requested capacity can fit.
  chi::u32 max_tags = config_.gpu_metadata_cache_.max_tags_;
  chi::u32 max_blobs = config_.gpu_metadata_cache_.max_blobs_;
  size_t needed = GpuMetadataCacheHeader::Layout(max_tags, max_blobs);
  size_t cap = static_cast<size_t>(config_.gpu_metadata_cache_.capacity_bytes_);
  if (needed > cap) {
    // Shrink slot counts proportionally so we stay within budget.
    double scale =
        static_cast<double>(cap - sizeof(GpuMetadataCacheHeader)) /
        static_cast<double>(needed - sizeof(GpuMetadataCacheHeader));
    if (scale < 0.0) scale = 0.0;
    if (scale > 1.0) scale = 1.0;
    max_tags = std::max<chi::u32>(
        1u, static_cast<chi::u32>(static_cast<double>(max_tags) * scale));
    max_blobs = std::max<chi::u32>(
        1u, static_cast<chi::u32>(static_cast<double>(max_blobs) * scale));
    needed = GpuMetadataCacheHeader::Layout(max_tags, max_blobs);
    HLOG(kWarning,
         "GpuMetadataCache: requested capacity {} bytes too small for the "
         "configured slot counts; rescaled to max_tags={} max_blobs={} "
         "({} bytes).",
         cap, max_tags, max_blobs, needed);
  }

  // Managed/shared USM is host- and device-readable through the same
  // virtual address. CUDA -> cudaMallocManaged, ROCm -> hipMallocManaged,
  // SYCL -> sycl::malloc_shared. All three give us a pointer the CPU can
  // call GpuCacheUpsert*/Remove* through directly.
  void *region = ctp::GpuApi::MallocManaged<char>(needed);
  if (!region) {
    HLOG(kError,
         "GpuMetadataCache: MallocManaged({} bytes) failed", needed);
    gpu_cache_ = nullptr;
    gpu_cache_bytes_ = 0;
    return false;
  }
  std::memset(region, 0, needed);
  gpu_cache_ = reinterpret_cast<GpuMetadataCacheHeader *>(region);
  gpu_cache_bytes_ = needed;
  gpu_cache_->Init(max_tags, max_blobs, needed);
  HLOG(kInfo,
       "GpuMetadataCache: allocated {} bytes (max_tags={}, max_blobs={}) "
       "at {}",
       needed, max_tags, max_blobs, static_cast<void *>(gpu_cache_));
  return true;
#endif
}

void Runtime::GpuCacheDestroy() {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
  if (gpu_cache_ != nullptr) {
    ctp::GpuApi::Free(reinterpret_cast<char *>(gpu_cache_));
    gpu_cache_ = nullptr;
    gpu_cache_bytes_ = 0;
  }
#else
  gpu_cache_ = nullptr;
  gpu_cache_bytes_ = 0;
#endif
}

void Runtime::GpuCacheOnPutBlob(const TagId &tag_id,
                                const std::string &blob_name,
                                const BlobInfo &blob_info) {
  if (gpu_cache_ == nullptr) return;
  std::string bdev_type = GetBdevTypeForBlob(blob_info);
  chi::u32 sc = gpu_cache::BdevTypeToStorageClass(bdev_type.c_str());
  chi::u64 size = blob_info.GetTotalSize();
  float score = blob_info.score_;
  if (gpu_cache::IsGpuVisible(sc)) {
    GpuCacheUpsertBlob(gpu_cache_, tag_id.major_, tag_id.minor_,
                       blob_name.c_str(), size, score, sc);
  } else {
    GpuCacheRemoveBlob(gpu_cache_, tag_id.major_, tag_id.minor_,
                       blob_name.c_str());
  }
}

std::string Runtime::GetBdevTypeForBlob(const BlobInfo &blob_info) {
  // Empty-blob (no blocks placed yet) -> nothing the GPU can reach.
  if (blob_info.blocks_.empty()) return std::string();

  // Resolve the bdev_type from the TargetInfo recorded at RegisterTarget
  // time. This is the source of truth for any target — both YAML-composed
  // ones AND those registered programmatically by tests / external code.
  const auto &first_block = blob_info.blocks_[0];
  chi::ScopedCoRwReadLock lock(target_lock_);
  TargetInfo *target_info =
      registered_targets_.find(first_block.bdev_client_.pool_id_);
  if (!target_info) return std::string();
  switch (target_info->bdev_type_) {
    case clio::run::bdev::BdevType::kRam:    return std::string("ram");
    case clio::run::bdev::BdevType::kHbm:    return std::string("hbm");
    case clio::run::bdev::BdevType::kPinned: return std::string("pinned");
    case clio::run::bdev::BdevType::kFile:   return std::string("file");
    case clio::run::bdev::BdevType::kNoop:   return std::string("noop");
    default:                                return std::string();
  }
}

void Runtime::GpuCacheOnDelBlob(const TagId &tag_id,
                                const std::string &blob_name) {
  if (gpu_cache_ == nullptr) return;
  GpuCacheRemoveBlob(gpu_cache_, tag_id.major_, tag_id.minor_,
                     blob_name.c_str());
}

void Runtime::GpuCacheOnGetOrCreateTag(const TagId &tag_id,
                                       const std::string &tag_name) {
  if (gpu_cache_ == nullptr) return;
  GpuCacheUpsertTag(gpu_cache_, tag_id.major_, tag_id.minor_,
                    tag_name.c_str());
}

void Runtime::GpuCacheOnDelTag(const TagId &tag_id) {
  if (gpu_cache_ == nullptr) return;
  GpuCacheRemoveTag(gpu_cache_, tag_id.major_, tag_id.minor_);
}

}  // namespace clio::cte::core

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::cte::core::Runtime)
