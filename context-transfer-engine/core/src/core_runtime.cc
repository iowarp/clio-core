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
using clio::run::chi_cur_worker_key_;
using clio::run::Worker;

// ===========================================================================
// Hierarchical tag-name encoding helpers.
//
// Absolute-path tags (names beginning with '/') are stored RELATIVELY so that
// renaming/moving a directory tag is O(1): a child holds the canonical string
//     "$tagid{<major>.<minor>}/<leaf>"
// where <major>.<minor> is its PARENT tag's id and <leaf> is the single path
// component. The root "/" is stored literally as "/". Flat (non-path) names
// are stored verbatim. Resolution (ResolveTagName) walks the parent ids to
// rebuild the full path. Non-path names are unaffected by any of this.
// ===========================================================================
namespace {

constexpr const char *kTagRefPrefix = "$tagid{";

// Escape regex metacharacters so `s` matches itself literally. Used to build
// exact/prefix patterns for the tag search index (#598).
std::string EscapeRegexLiteral(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '.' || c == '[' || c == ']' || c == '(' || c == ')' ||
        c == '{' || c == '}' || c == '+' || c == '*' || c == '?' ||
        c == '\\' || c == '^' || c == '$' || c == '|' || c == '/') {
      out += '\\';
    }
    out += c;
  }
  return out;
}

// True for absolute-path names that participate in the hierarchy.
bool IsHierPath(const std::string &name) {
  return !name.empty() && name[0] == '/';
}

// Build the relative stored form for a child of `parent` with leaf `leaf`.
std::string MakeRelativeName(const TagId &parent, const std::string &leaf) {
  return std::string(kTagRefPrefix) + std::to_string(parent.major_) + "." +
         std::to_string(parent.minor_) + "}/" + leaf;
}

// Split "/a/b/c" -> ["a","b","c"]; "/" or "" -> []. Repeated and trailing
// slashes are collapsed (so "/a/b/" == "/a/b").
std::vector<std::string> SplitPathComponents(const std::string &path) {
  std::vector<std::string> out;
  size_t i = 0;
  const size_t n = path.size();
  while (i < n) {
    while (i < n && path[i] == '/') ++i;
    size_t j = i;
    while (j < n && path[j] != '/') ++j;
    if (j > i) out.push_back(path.substr(i, j - i));
    i = j;
  }
  return out;
}

// If `stored` is a "$tagid{M.m}/leaf" reference, parse out the parent id and
// the leaf and return true; otherwise return false (flat name or root).
bool ParseTagRef(const std::string &stored, TagId &parent_out,
                 std::string &leaf_out) {
  const size_t plen = std::char_traits<char>::length(kTagRefPrefix);
  if (stored.compare(0, plen, kTagRefPrefix) != 0) return false;
  size_t close = stored.find('}', plen);
  if (close == std::string::npos) return false;
  const std::string id_str = stored.substr(plen, close - plen);
  size_t dot = id_str.find('.');
  if (dot == std::string::npos) return false;
  try {
    parent_out.major_ =
        static_cast<clio::run::u32>(std::stoul(id_str.substr(0, dot)));
    parent_out.minor_ =
        static_cast<clio::run::u32>(std::stoul(id_str.substr(dot + 1)));
  } catch (const std::exception &) {
    return false;
  }
  std::string suffix = stored.substr(close + 1);  // e.g. "/leaf"
  if (!suffix.empty() && suffix[0] == '/') suffix.erase(0, 1);
  leaf_out = std::move(suffix);
  return true;
}

// Join an already-resolved parent path with a leaf, avoiding "//".
std::string JoinPath(const std::string &parent_full, const std::string &leaf) {
  if (parent_full == "/") return "/" + leaf;
  return parent_full + "/" + leaf;
}

// Split an absolute path into (parent_path, leaf): "/a/b/c" -> ("/a/b","c"),
// "/c" -> ("/","c"). Returns false for "/" or names with no component.
bool SplitParentLeaf(const std::string &path, std::string &parent_out,
                     std::string &leaf_out) {
  std::vector<std::string> comps = SplitPathComponents(path);
  if (comps.empty()) return false;
  leaf_out = comps.back();
  parent_out = "/";
  for (size_t i = 0; i + 1 < comps.size(); ++i) {
    parent_out += (i == 0 ? "" : "/") + comps[i];
  }
  return true;
}

}  // namespace

// No more static member definitions - using instance-based locking

clio::run::u64 Runtime::ParseCapacityToBytes(const std::string &capacity_str) {
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

  clio::run::u64 multiplier = 1;
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
    return static_cast<clio::run::u64>(value);
  }

  return static_cast<clio::run::u64>(value * multiplier);
}

void Runtime::FixupAfterCopy(clio::run::u32 method,
                              ctp::ipc::FullPtr<clio::run::Task> task_ptr) {
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

clio::run::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task,
                                clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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
      ctp::priv::unordered_map_ll<clio::run::PoolId, TargetInfo>(kTargetMapSize);
  target_name_to_id_ =
      ctp::priv::unordered_map_ll<std::string, clio::run::PoolId>(kTargetMapSize);
  tag_name_to_id_ =
      ctp::priv::unordered_map_ll<std::string, TagId>(kTagMapSize);
  tag_id_to_info_ = ctp::priv::unordered_map_ll<TagId, TagInfo>(kTagMapSize);
  tag_blob_name_to_info_ =
      ctp::priv::unordered_map_ll<std::string, BlobInfo>(kBlobMapSize);

  // Initialize lock vectors for concurrent access
  target_locks_.reserve(kMaxLocks);
  tag_locks_.reserve(kMaxLocks);
  for (size_t i = 0; i < kMaxLocks; ++i) {
    target_locks_.emplace_back(std::make_unique<clio::run::CoRwLock>());
    tag_locks_.emplace_back(std::make_unique<clio::run::CoRwLock>());
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
    clio::run::u32 neighborhood_size = config_.targets_.neighborhood_;
    clio::run::u32 num_nodes = ipc_manager->GetNumHosts();
    clio::run::u32 actual_neighborhood = std::min(neighborhood_size, num_nodes);
    HLOG(kDebug,
         "Registering targets across neighborhood window (size: {} nodes, "
         "this container_id_={})",
         actual_neighborhood, container_id_);

    for (size_t device_idx = 0; device_idx < storage_devices_.size();
         ++device_idx) {
      const auto &device = storage_devices_[device_idx];
      clio::run::u64 capacity_bytes = device.capacity_limit_;
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

      for (clio::run::u32 i = 0; i < actual_neighborhood; ++i) {
        // Sliding-window neighbor index. Modulo num_nodes wraps the
        // window for containers near the end of the cluster.
        clio::run::u32 target_node =
            (container_id_ + i) % std::max<clio::run::u32>(num_nodes, 1u);

        std::string target_path =
            device.path_ + "_node" + std::to_string(target_node);
        clio::run::PoolQuery target_query =
            clio::run::PoolQuery::DirectHash(target_node);
        clio::run::PoolId bdev_id(512 + static_cast<clio::run::u32>(device_idx),
                            1 + target_node);

        HLOG(kDebug,
             "Registering target ({}): {} ({}, {} bytes) on node {} (i={}) "
             "with bdev_id=({},{})",
             client_.pool_id_, target_path, device.bdev_type_, capacity_bytes,
             target_node, i, bdev_id.major_, bdev_id.minor_);
        auto reg_task = client_.AsyncRegisterTarget(
            target_path, bdev_type, capacity_bytes, target_query, bdev_id);
        CLIO_CO_AWAIT(reg_task);
        clio::run::u32 result = reg_task->GetReturnCode();
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
    // Both paths populate the tag table directly (bypassing GetOrAssignTagId's
    // per-insert indexing), so rebuild the regex search index once from the
    // final tag set (#598).
    clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
    RebuildTagSearchIndexLocked();
  }

  // Open WAL files if metadata_log_path is configured
  if (!config_.performance_.metadata_log_path_.empty()) {
    clio::run::u32 num_workers =
        std::max(CLIO_WORK_ORCHESTRATOR->GetTotalWorkerCount(), (clio::run::u32)1);
    clio::run::u64 per_worker_capacity = std::max(
        config_.performance_.transaction_log_capacity_bytes_ / num_workers,
        (clio::run::u64)4096);
    blob_txn_logs_.resize(num_workers);
    tag_txn_logs_.resize(num_workers);
    for (clio::run::u32 i = 0; i < num_workers; ++i) {
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
  clio::run::u32 stat_period_ms = config_.performance_.stat_targets_period_ms_;
  if (stat_period_ms > 0) {
    HLOG(kInfo, "Starting periodic StatTargets task with period {} ms",
         stat_period_ms);
    client_.AsyncStatTargets(clio::run::PoolQuery::Local(), stat_period_ms);
  }

  // Spawn periodic FlushMetadata if metadata_log_path is configured and period
  // > 0
  if (!config_.performance_.metadata_log_path_.empty() &&
      config_.performance_.flush_metadata_period_ms_ > 0) {
    client_.AsyncFlushMetadata(
        clio::run::PoolQuery::Local(),
        config_.performance_.flush_metadata_period_ms_ * 1000.0);
  }

  // Spawn periodic FlushData if configured
  if (config_.performance_.flush_data_period_ms_ > 0) {
    client_.AsyncFlushData(clio::run::PoolQuery::Local(),
                           config_.performance_.flush_data_min_persistence_,
                           config_.performance_.flush_data_period_ms_ * 1000.0);
  }

  // Allocate the optional GPU metadata cache. The OUT pointer is
  // re-serialized back into chimod_params_ (a clio::run::priv::string) so
  // the client's GetParams() sees the populated gpu_cache_ptr_ after
  // Wait().
  CreateParams out_params;
  out_params.config_ = config_;
  out_params.gpu_cache_ptr_ =
      GpuCacheCreate() ? reinterpret_cast<clio::run::u64>(gpu_cache_)
                       : static_cast<clio::run::u64>(0);
  clio::run::Task::Serialize(CLIO_PRIV_ALLOC, task->chimod_params_, out_params);

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

Runtime::~Runtime() {
  // Mirror Destroy()'s cleanup so deleting the container on graceful shutdown
  // frees this module's runtime-heap state (the member containers below are
  // also RAII, but closing the WAL files explicitly here is the safe, intended
  // teardown). Destructor-safe: no task/RunContext, no coroutine, no worker
  // state access.
  for (auto &log : blob_txn_logs_) {
    if (log) log->Close();
  }
  blob_txn_logs_.clear();
  for (auto &log : tag_txn_logs_) {
    if (log) log->Close();
  }
  tag_txn_logs_.clear();

  registered_targets_.clear();
  target_list_.clear();
  target_name_to_id_.clear();
  tag_name_to_id_.clear();
  tag_id_to_info_.clear();
  tag_blob_name_to_info_.clear();
  storage_devices_.clear();
  target_locks_.clear();
  tag_locks_.clear();
}

clio::run::TaskResume Runtime::Destroy(ctp::ipc::FullPtr<DestroyTask> task,
                                 clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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

clio::run::PoolQuery Runtime::ScheduleTask(const ctp::ipc::FullPtr<clio::run::Task> &task) {
  using namespace clio::cte::core;
  switch (task->method_) {
    // Methods that route locally
    case Method::kRegisterTarget:
    case Method::kUnregisterTarget:
    case Method::kListTargets:
    case Method::kStatTargets:
    case Method::kGetTargetInfo:
      return clio::run::PoolQuery::Local();

    // GetOrCreateTag: check local tag cache, hash to container if not found
    case Method::kGetOrCreateTag: {
      auto typed = task.template Cast<GetOrCreateTagTask<CreateParams>>();
      std::string tag_name = typed->tag_name_.str();
      bool tag_exists = false;
      {
        clio::run::ScopedCoRwReadLock lock(tag_map_lock_);
        tag_exists = (tag_name_to_id_.find(tag_name) != nullptr);
      }
      if (tag_exists) {
        return clio::run::PoolQuery::Local();
      }
      std::hash<std::string> string_hasher;
      clio::run::u32 hash_value = static_cast<clio::run::u32>(string_hasher(tag_name));
      return clio::run::PoolQuery::DirectHash(hash_value);
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
      return clio::run::PoolQuery::Broadcast();

    default:
      return task->pool_query_;
  }
}

clio::run::TaskResume Runtime::RegisterTarget(ctp::ipc::FullPtr<RegisterTargetTask> task,
                                        clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string target_name = task->target_name_.str();
    clio::run::bdev::BdevType bdev_type = task->bdev_type_;
    clio::run::u64 total_size = task->total_size_;
    clio::run::PoolId bdev_pool_id = task->bdev_id_;
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
    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
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
    clio::run::PoolId target_id = bdev_client.pool_id_;

    // Check if target is already registered using TargetId
    {
      clio::run::ScopedCoRwReadLock read_lock(target_lock_);
      TargetInfo *existing_target = registered_targets_.find(target_id);
      if (existing_target != nullptr) {
        CLIO_CO_RETURN;
      }
    }

    // Get actual statistics from bdev using AsyncGetStats method
    clio::run::u64 remaining_size;
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
    target_info.max_capacity_ =
        total_size;  // Total (max) capacity, fixed for the life of the target
    target_info.perf_metrics_ =
        perf_metrics;  // Store the entire PerfMetrics structure
    target_info.persistence_level_ = GetPersistenceLevelForTarget(target_name);
    target_info.bdev_type_ = task->bdev_type_;

    // Register the target using TargetId as key. Mirror into target_list_ so
    // iteration sites (ExtendBlob, ListTargets, StatTargets, FlushData) can
    // walk live entries directly without scanning empty map slots.
    {
      clio::run::ScopedCoRwWriteLock write_lock(target_lock_);
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

clio::run::TaskResume Runtime::UnregisterTarget(
    ctp::ipc::FullPtr<UnregisterTargetTask> task, clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string target_name = task->target_name_.str();

    // Check if target exists and remove it (don't destroy bdev container)
    {
      clio::run::ScopedCoRwWriteLock write_lock(target_lock_);

      // Look up TargetId from target_name (under lock)
      clio::run::PoolId *target_id_ptr = target_name_to_id_.find(target_name);
      if (target_id_ptr == nullptr) {
        task->return_code_ = 1;
        CLIO_CO_RETURN;
      }

      // Copy by value: target_id_ptr points into the map node that
      // erase(target_name) below frees, and target_id is still read in the
      // target_list_ loop after that (ASan heap-use-after-free, issue #520).
      const clio::run::PoolId target_id = *target_id_ptr;
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

clio::run::TaskResume Runtime::ListTargets(ctp::ipc::FullPtr<ListTargetsTask> task,
                                     clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Clear the output vector and populate with current target names
    task->target_names_.clear();

    clio::run::ScopedCoRwReadLock read_lock(target_lock_);

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

clio::run::TaskResume Runtime::StatTargets(ctp::ipc::FullPtr<StatTargetsTask> task,
                                     clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    // Collect all target IDs under read lock (can't co_await inside lambda)
    std::vector<clio::run::PoolId> target_ids;
    {
      clio::run::ScopedCoRwReadLock read_lock(target_lock_);
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
        clio::run::ScopedCoRwReadLock read_lock(target_lock_);
        TargetInfo *target_info = registered_targets_.find(target_id);
        if (target_info != nullptr) {
          bdev_client_copy = target_info->bdev_client_;
          found = true;
        }
      }
      if (!found) continue;

      // Perform async stats query WITHOUT holding lock
      clio::run::u64 remaining_size;
      auto stats_task = bdev_client_copy.AsyncGetStats();
      CLIO_CO_AWAIT(stats_task);
      clio::run::bdev::PerfMetrics perf_metrics = stats_task->metrics_;
      remaining_size = stats_task->remaining_size_;

      // Re-acquire write lock to update target info. Mutate the map and the
      // mirror in target_list_ in lockstep so DPE selection sees fresh stats.
      {
        clio::run::ScopedCoRwWriteLock write_lock(target_lock_);
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
clio::run::TaskResume Runtime::GetOrCreateTag(
    ctp::ipc::FullPtr<GetOrCreateTagTask<CreateParamsT>> task,
    clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string tag_name = task->tag_name_.str();
    TagId preferred_id = task->tag_id_;
    auto *ipc_manager = CLIO_IPC;
    clio::run::u32 local_node_id = ipc_manager->GetNodeId();

    // Check if this is a returning task from a remote canonical node
    bool is_remote_tag =
        (preferred_id.major_ != 0 && preferred_id.major_ != local_node_id);

    if (is_remote_tag) {
      clio::run::ScopedCoRwWriteLock write_lock(tag_map_lock_);
      TagId *existing_tag_id_ptr = tag_name_to_id_.find(tag_name);
      if (existing_tag_id_ptr == nullptr) {
        tag_name_to_id_.insert_or_assign(tag_name, preferred_id);
        // Mirror the binding into the search index so a Local TagQuery still
        // sees remote-canonical tag names (parity with the old full scan). (#598)
        tag_search_.Insert(ResolveTagName(tag_name), preferred_id);
      }
      task->tag_id_ = preferred_id;
      GpuCacheOnGetOrCreateTag(preferred_id, tag_name);
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    // Absolute paths are created as a hierarchy ("/a/b/c" -> "/", "/a", "/a/b",
    // "/a/b/c") with each child stored relative to its parent; the returned id
    // is the deepest tag. Flat names create a single tag (legacy behavior).
    TagId tag_id = GetOrCreateTagChain(tag_name, preferred_id);
    task->tag_id_ = tag_id;

    auto now = GetCurrentTimeNs();
    {
      clio::run::ScopedCoRwWriteLock write_lock(tag_map_lock_);
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

clio::run::TaskResume Runtime::GetTargetInfo(ctp::ipc::FullPtr<GetTargetInfoTask> task,
                                       clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string target_name = task->target_name_.str();

    // Look up target by name (under lock for concurrent safety)
    clio::run::ScopedCoRwReadLock read_lock(target_lock_);
    clio::run::PoolId *target_id_ptr = target_name_to_id_.find(target_name);
    if (target_id_ptr == nullptr) {
      task->return_code_ = 1;  // Target not found
      CLIO_CO_RETURN;
    }

    clio::run::PoolId target_id = *target_id_ptr;

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

clio::run::TaskResume Runtime::PutBlob(ctp::ipc::FullPtr<PutBlobTask> task,
                                 clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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
  // clio actually started executing it. Dumped sparsely to keep
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
    clio::run::u64 offset = task->offset_;
    clio::run::u64 size = task->size_;
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

    // Create blob metadata if new; otherwise remember its current size.
    clio::run::u64 old_blob_size = 0;
    if (!blob_found) {
      blob_info_ptr = CreateNewBlob(blob_name, tag_id, blob_score);
      if (!blob_info_ptr) {
        task->return_code_ = 5;
        CLIO_CO_RETURN;
      }
    } else {
      old_blob_size = blob_info_ptr->GetTotalSize();
    }

    // Step 1+2: size the blob to fit the write.
    //  - default (partial modify): grow to cover [offset, offset+size) but
    //    NEVER shrink — writing must not truncate the tail (POSIX write
    //    semantics). This is what makes out-of-order / descending partial
    //    writes (e.g. HDF5 writing data high, then the superblock at offset 0)
    //    safe; the old offset==0 "clear the whole blob" heuristic corrupted
    //    them.
    //  - kCtePutReplace (wholesale replace): resize to exactly offset+size,
    //    shrinking if needed, via the shared ResizeBlob helper.
    clio::run::u32 alloc_result = 0;
    if (task->flags_ & kCtePutReplace) {
      CLIO_CO_AWAIT(ResizeBlob(*blob_info_ptr, offset + size, blob_score,
                          alloc_result,
                          task->context_.min_persistence_level_));
    } else {
      CLIO_CO_AWAIT(ExtendBlob(*blob_info_ptr, offset, size, blob_score,
                          alloc_result,
                          task->context_.min_persistence_level_));
    }
    if (alloc_result != 0) {
      task->return_code_ = 10 + alloc_result;
      CLIO_CO_RETURN;
    }

    // WAL: log all current blocks (full replacement semantics)
    if (!blob_txn_logs_.empty() && !blob_info_ptr->blocks_.empty()) {
      clio::run::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
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

    // Step 2.5: Zero any hole created by writing past the old end-of-blob
    // (sparse write). Newly allocated space is NOT guaranteed to be zero — the
    // bdev recycles freed blocks, so a fresh block can hold stale data — and
    // POSIX requires allocated-but-unwritten bytes to read as zeros. Only the
    // gap [old_blob_size, offset) needs it; sequential appends (offset ==
    // old_blob_size) and overwrites (offset < old_blob_size) create no hole.
    if (offset > old_blob_size) {
      clio::run::u64 hole = offset - old_blob_size;
      auto *ipc_mgr = CLIO_IPC;
      ctp::ipc::FullPtr<char> zbuf = ipc_mgr->AllocateBuffer(hole);
      if (zbuf.IsNull()) {
        task->return_code_ = 6;
        CLIO_CO_RETURN;
      }
      std::memset(zbuf.ptr_, 0, hole);
      clio::run::u32 zero_result = 0;
      CLIO_CO_AWAIT(ModifyExistingData(blob_info_ptr->blocks_,
                                  zbuf.shm_.template Cast<void>(), hole,
                                  old_blob_size, zero_result));
      ipc_mgr->FreeBuffer(zbuf);
      if (zero_result != 0) {
        task->return_code_ = 20 + zero_result;
        CLIO_CO_RETURN;
      }
    }

    // Step 3: ModifyExistingData — write data to blocks
    clio::run::u32 write_result = 0;
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
    clio::run::u64 new_blob_size = blob_info_ptr->GetTotalSize();
    clio::run::i64 size_change = static_cast<clio::run::i64>(new_blob_size) -
                           static_cast<clio::run::i64>(old_blob_size);
    auto now = GetCurrentTimeNs();
    blob_info_ptr->last_modified_ = now;
    blob_info_ptr->score_ = blob_score;
    {
      // Write lock: we may need to insert a fresh tag_info entry. The
      // tag's name lives on whichever container `GetOrCreateTag`'s
      // `DirectHash(tag_name)` selected; this container only owns the
      // blobs that `HashBlobToContainer(tag_id, blob_name)` routed
      // here. To keep the `GetTagSize` broadcast-and-AggregateOut sum
      // correct, every container that holds any of the tag's bytes
      // must carry a `TagInfo` whose `total_size_` reflects its share.
      // The silent-skip variant of this block dropped the accounting
      // when the tag wasn't locally registered, so on 2n the
      // tag-owning container saw total_size_ = 0 (no PutBlobs hashed
      // to it) and the blob-owning container had no TagInfo at all
      // (rc=1, tag_size_=0). stat() then returned 0 after writes.
      clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr == nullptr) {
        // No prior local accounting; seed an entry. tag_name_ stays
        // empty -- the canonical name<->id binding lives on the
        // tag-owning container and isn't this container's concern.
        TagInfo seed;
        seed.tag_id_ = tag_id;
        seed.last_modified_ = now;
        seed.last_read_ = now;
        seed.last_changed_ = now;
        seed.total_size_ = 0;
        auto ins = tag_id_to_info_.insert_or_assign(tag_id, seed);
        tag_info_ptr = ins.value;
      }
      if (tag_info_ptr) {
        tag_info_ptr->last_modified_ = now;
        tag_info_ptr->last_changed_ = now;  // size change => ctime bump
        if (size_change >= 0) {
          tag_info_ptr->total_size_ += static_cast<clio::run::u64>(size_change);
        } else {
          clio::run::u64 abs_change = static_cast<clio::run::u64>(-size_change);
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

clio::run::TaskResume Runtime::GetBlob(ctp::ipc::FullPtr<GetBlobTask> task,
                                 clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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
    clio::run::u64 offset = task->offset_;
    clio::run::u64 size = task->size_;
    clio::run::u32 flags = task->flags_;

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
    clio::run::u32 read_result = 0;
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

clio::run::TaskResume Runtime::ReorganizeBlob(ctp::ipc::FullPtr<ReorganizeBlobTask> task,
                                        clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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
    clio::run::u64 blob_size = blob_info.GetTotalSize();

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

clio::run::TaskResume Runtime::DelBlob(ctp::ipc::FullPtr<DelBlobTask> task,
                                 clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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
    clio::run::u64 blob_size = blob_info_ptr->GetTotalSize();

    // Step 2.5: Free all blocks back to their targets before removing blob
    clio::run::u32 free_result = 0;
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
      clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr != nullptr) {
        if (blob_size <= tag_info_ptr->total_size_) {
          tag_info_ptr->total_size_ -= blob_size;
        } else {
          tag_info_ptr->total_size_ = 0;
        }
        tag_info_ptr->last_changed_ = GetCurrentTimeNs();  // size change => ctime
      }
    }

    // Step 5: Remove blob from tag_blob_name_to_info_ map
    {
      clio::run::ScopedCoRwWriteLock lock(blob_map_lock_);
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
      clio::run::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
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

clio::run::TaskResume Runtime::TruncateBlob(ctp::ipc::FullPtr<TruncateBlobTask> task,
                                      clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    TagId tag_id = task->tag_id_;
    std::string blob_name = task->blob_name_.str();
    clio::run::u64 new_size = task->new_size_;
    if (blob_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    BlobInfo *blob_info_ptr = CheckBlobExists(blob_name, tag_id);
    if (blob_info_ptr == nullptr) {
      // A missing blob is already "empty" — nothing to truncate.
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }
    clio::run::u64 old_size = blob_info_ptr->GetTotalSize();
    float blob_score = blob_info_ptr->score_;

    // Shared resize helper (also used by PutBlob's replace path).
    clio::run::u32 resize_result = 0;
    CLIO_CO_AWAIT(ResizeBlob(*blob_info_ptr, new_size, blob_score,
                        resize_result, 0));
    if (resize_result != 0) {
      task->return_code_ = 10 + resize_result;
      CLIO_CO_RETURN;
    }
    clio::run::u64 final_size = blob_info_ptr->GetTotalSize();

    // Update the tag's total_size_ by the delta.
    {
      clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr != nullptr) {
        if (final_size >= old_size) {
          tag_info_ptr->total_size_ += (final_size - old_size);
        } else {
          clio::run::u64 d = old_size - final_size;
          tag_info_ptr->total_size_ =
              (d <= tag_info_ptr->total_size_) ? tag_info_ptr->total_size_ - d
                                               : 0;
        }
        tag_info_ptr->last_changed_ = GetCurrentTimeNs();  // truncate => ctime
      }
    }

    // WAL: record the resized block list (full-replacement semantics), so a
    // restart replays the truncated blob.
    if (!blob_txn_logs_.empty() && !blob_info_ptr->blocks_.empty()) {
      clio::run::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
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

    blob_info_ptr->last_modified_ = GetCurrentTimeNs();
    task->return_code_ = 0;
  } catch (const std::exception &e) {
    HLOG(kError, "TruncateBlob failed: {}", e.what());
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RenameTag(ctp::ipc::FullPtr<RenameTagTask> task,
                                   clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    TagId tag_id = task->tag_id_;
    std::string old_name = task->old_name_.str();
    std::string new_name = task->new_name_.str();
    if (old_name.empty() || new_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }
    if (old_name == new_name) {
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    // The tag keeps its TagId — only the name changes, so its blobs (keyed by
    // TagId) are untouched, and so are any CHILDREN (they reference this tag's
    // id, which does not change). That is what makes moving a directory tag
    // O(1) regardless of how many descendants it has.
    if (IsHierPath(old_name) && IsHierPath(new_name)) {
      // ---- Hierarchical move: rebind only the leaf under its new parent. ----
      // Split destination into parent path + leaf.
      std::vector<std::string> comps = SplitPathComponents(new_name);
      if (comps.empty()) {
        task->return_code_ = 1;  // cannot rename onto the root
        CLIO_CO_RETURN;
      }
      std::string new_leaf = comps.back();
      std::string new_parent_path = "/";
      for (size_t i = 0; i + 1 < comps.size(); ++i) {
        new_parent_path += (i == 0 ? "" : "/") + comps[i];
      }

      // Get-or-create the destination parent chain FIRST, before taking the tag
      // map lock — GetOrCreateTagChain acquires tag_map_lock_ itself, so it
      // cannot run while we hold it. Auto-creates missing parents (mkdir -p).
      TagId new_parent_id = GetOrCreateTagChain(new_parent_path);
      std::string new_rel = MakeRelativeName(new_parent_id, new_leaf);

      // Resolve the source, move the name binding, and refresh the stored name
      // as a SINGLE atomic read-modify-write under the write lock. The previous
      // design read the tag's current name under a read lock, released it, then
      // erased that value under a separate write lock — so a name read before
      // the lock could go stale and erase a binding a racing rename/create had
      // since reassigned to a *different* tag. That left a tag still resolvable
      // upward (readdir/stat list it) but not forward (unlink/rmdir cannot find
      // it). Reading the canonical name under the SAME lock that erases it, with
      // no gap, serializes overlapping renames and keeps tag_name_to_id_ and the
      // per-tag canonical name in agreement. See issue #596.
      {
        clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
        if (tag_id.IsNull()) {
          tag_id = ResolvePathToIdLocked(old_name);
        }
        if (tag_id.IsNull()) {
          task->return_code_ = 1;  // source path not found
          CLIO_CO_RETURN;
        }
        TagInfo *info = tag_id_to_info_.find(tag_id);
        if (info == nullptr) {
          task->return_code_ = 1;  // source tag has no metadata
          CLIO_CO_RETURN;
        }
        // Fresh read of the canonical name, under the lock that erases it.
        std::string cur_rel = info->tag_name_.str();
        // Absolute paths BEFORE/AFTER the move. old_abs == ResolveTagName(
        // cur_rel) is exactly the search-index key stored at insert time.
        std::string old_abs = ResolveTagName(cur_rel);
        std::string new_abs = ResolveTagName(new_rel);
        if (cur_rel != new_rel) {
          tag_name_to_id_.erase(cur_rel);
        }
        tag_name_to_id_.insert_or_assign(new_rel, tag_id);
        info->tag_name_ = clio::run::priv::string(CLIO_PRIV_ALLOC, new_rel);
        info->last_modified_ = GetCurrentTimeNs();
        info->last_changed_ = info->last_modified_;  // rename => ctime bump

        // Re-key this tag and its descendants in the search index from old_abs
        // to new_abs (#598). A directory move changes the absolute names of the
        // whole subtree, so move every entry under old_abs; a file moves just
        // one. Enumerated via the index's own trigram-prefiltered prefix search
        // => O(subtree), not O(N).
        std::string esc = EscapeRegexLiteral(old_abs);
        auto self_hit = tag_search_.Search("^" + esc + "$");
        auto descendants = tag_search_.Search("^" + esc + "/.*");
        std::vector<std::string> movers = self_hit.keys();
        for (const auto &k : descendants.keys()) {
          movers.push_back(k);
        }
        for (const auto &k : movers) {
          tag_search_.Rename(k, new_abs + k.substr(old_abs.size()));
        }
      }
      task->tag_id_ = tag_id;
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    // ---- Flat rename (non-path tags): move the verbatim name binding. ----
    // Broadcast: each container moves the name->id binding it happens to hold.
    clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
    TagId *idp = tag_name_to_id_.find(old_name);
    if (idp != nullptr) {
      TagId bound = *idp;
      if (tag_id.IsNull()) {
        tag_id = bound;
      }
      tag_name_to_id_.erase(old_name);
      tag_name_to_id_.insert_or_assign(new_name, bound);
    }
    if (!tag_id.IsNull()) {
      TagInfo *info = tag_id_to_info_.find(tag_id);
      if (info != nullptr) {
        info->tag_name_ = clio::run::priv::string(CLIO_PRIV_ALLOC, new_name);
        info->last_modified_ = GetCurrentTimeNs();
        info->last_changed_ = info->last_modified_;  // rename => ctime bump
      }
    }
    // Flat tags have no hierarchy, so the index key is the verbatim name; move
    // the single entry (no-op if it was never indexed). (#598)
    tag_search_.Rename(old_name, new_name);
    task->tag_id_ = tag_id;
    task->return_code_ = 0;
  } catch (const std::exception &e) {
    HLOG(kError, "RenameTag failed: {}", e.what());
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetOrCreateTagAlias(
    ctp::ipc::FullPtr<GetOrCreateTagAliasTask> task, clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    TagId tag_id = task->tag_id_;
    std::string existing_name = task->existing_name_.str();
    std::string alias_name = task->alias_name_.str();
    if (alias_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Phase A: resolve + verify the target tag exists. The target may be given
    // by id or by name (absolute paths are walked through the hierarchy).
    {
      clio::run::ScopedCoRwReadLock lock(tag_map_lock_);
      if (tag_id.IsNull() && !existing_name.empty()) {
        if (IsHierPath(existing_name)) {
          tag_id = ResolvePathToIdLocked(existing_name);
        }
        if (tag_id.IsNull()) {
          TagId *p = tag_name_to_id_.find(existing_name);
          if (p != nullptr) tag_id = *p;
        }
      }
      if (tag_id.IsNull() || tag_id_to_info_.find(tag_id) == nullptr) {
        task->found_ = 0;
        task->tag_id_ = TagId::GetNull();
        task->return_code_ = 0;  // found_ conveys "target missing"; not an error
        CLIO_CO_RETURN;
      }
    }
    task->found_ = 1;

    // Compute the binding KEY for the alias. An absolute-path alias becomes a
    // first-class hierarchy entry: create its parent chain (outside the lock —
    // GetOrCreateTagChain takes tag_map_lock_) and bind the relative key
    // "$tagid{parent}/leaf" so the link resolves, lists, and opens like any
    // other path. A flat alias binds verbatim (legacy behavior).
    std::string alias_key = alias_name;
    if (IsHierPath(alias_name)) {
      std::string parent_path, leaf;
      if (!SplitParentLeaf(alias_name, parent_path, leaf)) {
        task->return_code_ = 1;  // cannot alias onto the root
        CLIO_CO_RETURN;
      }
      TagId parent_id = GetOrCreateTagChain(parent_path);
      alias_key = MakeRelativeName(parent_id, leaf);
    }

    // Phase C: bind the alias key to the target id (a tag-level hard link — the
    // alias shares the target's id and therefore all of its blobs). GetOrCreate:
    // if the key is already bound, return whatever it points at unchanged.
    {
      clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
      TagId *existing_alias = tag_name_to_id_.find(alias_key);
      if (existing_alias != nullptr) {
        tag_id = *existing_alias;
      } else {
        tag_name_to_id_.insert_or_assign(alias_key, tag_id);
        // Index the alias's absolute name so it is findable via TagQuery
        // (getattr/readdir) just like a canonical name (#598). Aliases bypass
        // GetOrAssignTagId, so this is the only place they enter the index.
        tag_search_.Insert(ResolveTagName(alias_key), tag_id);
        // Record the alias key on the target so DelTag cascades to it when the
        // canonical tag is deleted. The canonical name is never added here.
        TagInfo *info = tag_id_to_info_.find(tag_id);
        if (info != nullptr) {
          bool present = (alias_key == info->tag_name_.str());
          for (size_t i = 0; !present && i < info->aliases_.size(); ++i) {
            if (info->aliases_[i].str() == alias_key) present = true;
          }
          if (!present) {
            info->aliases_.push_back(
                clio::run::priv::string(CLIO_PRIV_ALLOC, alias_key));
          }
          info->last_modified_ = GetCurrentTimeNs();
          info->last_changed_ = info->last_modified_;  // link added => ctime
        }
      }
    }
    task->tag_id_ = tag_id;
    task->return_code_ = 0;
  } catch (const std::exception &e) {
    HLOG(kError, "GetOrCreateTagAlias failed: {}", e.what());
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DelTag(ctp::ipc::FullPtr<DelTagTask> task,
                                clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    TagId tag_id = task->tag_id_;
    std::string tag_name = task->tag_name_.str();

    // Step 1: Resolve the tag id AND the tag_name_to_id_ key the request
    // resolves *through* (resolved_key). For an absolute path that key is the
    // relative "$tagid{parent}/leaf"; for a flat name it is the name itself;
    // for a by-id delete it stays empty. resolved_key is what distinguishes
    // deleting an alias (a non-canonical name) from deleting the tag itself.
    std::string resolved_key;
    if (tag_id.IsNull() && !tag_name.empty()) {
      clio::run::ScopedCoRwReadLock lock(tag_map_lock_);
      if (IsHierPath(tag_name)) {
        std::string parent_path, leaf;
        if (tag_name == "/") {
          TagId *r = tag_name_to_id_.find(std::string("/"));
          if (r != nullptr) { tag_id = *r; resolved_key = "/"; }
        } else if (SplitParentLeaf(tag_name, parent_path, leaf)) {
          TagId parent_id = ResolvePathToIdLocked(parent_path);
          if (!parent_id.IsNull()) {
            std::string key = MakeRelativeName(parent_id, leaf);
            TagId *p = tag_name_to_id_.find(key);
            if (p != nullptr) { tag_id = *p; resolved_key = key; }
          }
        }
      }
      // Fall back to a verbatim lookup (flat tags and flat aliases).
      if (tag_id.IsNull()) {
        TagId *p = tag_name_to_id_.find(tag_name);
        if (p != nullptr) { tag_id = *p; resolved_key = tag_name; }
      }
      if (tag_id.IsNull()) {
        task->return_code_ = 1;  // Tag not found by name
        CLIO_CO_RETURN;
      }
      task->tag_id_ = tag_id;
    } else if (tag_id.IsNull() && tag_name.empty()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Step 2: Find the tag by ID and capture its canonical (own) stored name
    // and its absolute path (the search-index key) while the subtree is intact.
    std::string canonical;
    std::string del_abs;
    {
      clio::run::ScopedCoRwReadLock lock(tag_map_lock_);
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr == nullptr) {
        task->return_code_ = 1;  // Tag not found by ID
        CLIO_CO_RETURN;
      }
      canonical = tag_info_ptr->tag_name_.str();
      del_abs = ResolveTagName(canonical);
    }

    // If the request resolved through a key that is NOT the tag's own canonical
    // name, it is an alias/hard-link "unlink": drop only that one name binding
    // and leave the tag, its blobs, and other names intact. Deleting by id, or
    // through the canonical name, falls through to a full recursive delete that
    // cascades to every alias below.
    if (!resolved_key.empty() && resolved_key != canonical) {
      clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
      tag_name_to_id_.erase(resolved_key);
      // Drop just this alias name from the search index; the tag and its other
      // names (canonical + remaining aliases) stay. (#598)
      tag_search_.Delete(ResolveTagName(resolved_key));
      TagInfo *tag_info_ptr = tag_id_to_info_.find(tag_id);
      if (tag_info_ptr != nullptr) {
        for (size_t i = 0; i < tag_info_ptr->aliases_.size(); ++i) {
          if (tag_info_ptr->aliases_[i].str() == resolved_key) {
            tag_info_ptr->aliases_.erase(tag_info_ptr->aliases_.begin() + i);
            break;
          }
        }
        tag_info_ptr->last_changed_ = GetCurrentTimeNs();  // unlink => ctime
        tag_info_ptr->last_modified_ = GetCurrentTimeNs();
      }
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    // Full recursive delete: remove this tag and its whole subtree from the
    // search index (#598), using the absolute path captured before any deletion.
    // O(subtree) via the index's trigram-prefiltered prefix search.
    {
      clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
      std::string esc = EscapeRegexLiteral(del_abs);
      auto self_hit = tag_search_.Search("^" + esc + "$");
      auto descendants = tag_search_.Search("^" + esc + "/.*");
      // keys() is a snapshot independent of the engine's maps, so deleting from
      // the engine while iterating it is safe.
      for (const auto &k : self_hit.keys()) {
        tag_search_.Delete(k);
      }
      for (const auto &k : descendants.keys()) {
        tag_search_.Delete(k);
      }
    }

    // Step 3: Determine the full set of tags to delete — the target plus every
    // transitive hierarchical descendant. A child stores its parent's id in
    // its name ("$tagid{parent}/leaf"), so build parent->children once and BFS
    // from the target. A flat or leaf tag has no descendants and deletes only
    // itself; a directory tag deletes its entire subtree (rm -r semantics).
    std::vector<TagId> to_delete;
    std::unordered_map<std::string, TagId> prefix_to_id;  // "M.m." -> id
    // Alias absolute names of the deleted tags. Canonical names live under
    // del_abs and were removed by the prefix cleanup above, but an alias can
    // resolve OUTSIDE that subtree (e.g. a hard link elsewhere), so collect and
    // remove those from the search index too. (#598)
    std::vector<std::string> dead_alias_abs;
    {
      clio::run::ScopedCoRwReadLock lock(tag_map_lock_);
      std::unordered_map<TagId, std::vector<TagId>> children;
      tag_id_to_info_.for_each([&](const TagId &id, const TagInfo &info) {
        TagId parent;
        std::string leaf;
        if (ParseTagRef(info.tag_name_.str(), parent, leaf)) {
          children[parent].push_back(id);
        }
      });
      std::vector<TagId> frontier{tag_id};
      while (!frontier.empty()) {
        TagId cur = frontier.back();
        frontier.pop_back();
        to_delete.push_back(cur);
        prefix_to_id[std::to_string(cur.major_) + "." +
                     std::to_string(cur.minor_) + "."] = cur;
        TagInfo *cinfo = tag_id_to_info_.find(cur);
        if (cinfo != nullptr) {
          for (size_t i = 0; i < cinfo->aliases_.size(); ++i) {
            dead_alias_abs.push_back(ResolveTagName(cinfo->aliases_[i].str()));
          }
        }
        auto it = children.find(cur);
        if (it != children.end()) {
          for (const TagId &c : it->second) frontier.push_back(c);
        }
      }
    }
    if (!dead_alias_abs.empty()) {
      clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
      for (const auto &a : dead_alias_abs) {
        tag_search_.Delete(a);
      }
    }

    // Step 4: collect every blob across all those tags in a single metadata
    // scan. Keys are "major.minor.blobname"; match by the "major.minor."
    // prefix against the deletion set.
    auto compound_prefix = [](const std::string &key) -> std::string {
      size_t d1 = key.find('.');
      if (d1 == std::string::npos) return std::string();
      size_t d2 = key.find('.', d1 + 1);
      if (d2 == std::string::npos) return std::string();
      return key.substr(0, d2 + 1);
    };
    std::vector<std::pair<TagId, std::string>> blobs_to_delete;
    {
      clio::run::ScopedCoRwReadLock lock(blob_map_lock_);
      tag_blob_name_to_info_.for_each(
          [&](const std::string &compound_key, const BlobInfo &blob_info) {
            (void)blob_info;
            auto it = prefix_to_id.find(compound_prefix(compound_key));
            if (it != prefix_to_id.end()) {
              blobs_to_delete.emplace_back(
                  it->second, compound_key.substr(it->first.size()));
            }
          });
    }

    // Step 5: delete blobs in bounded-concurrency batches.
    constexpr size_t kMaxConcurrentDelBlobTasks = 32;
    std::vector<clio::run::Future<DelBlobTask>> async_tasks;
    size_t processed_blobs = 0;
    for (size_t i = 0; i < blobs_to_delete.size();
         i += kMaxConcurrentDelBlobTasks) {
      async_tasks.clear();
      size_t batch_end =
          std::min(i + kMaxConcurrentDelBlobTasks, blobs_to_delete.size());
      for (size_t j = i; j < batch_end; ++j) {
        async_tasks.push_back(client_.AsyncDelBlob(blobs_to_delete[j].first,
                                                   blobs_to_delete[j].second));
      }
      for (auto t : async_tasks) {
        CLIO_CO_AWAIT(t);
        if (t->return_code_ != 0) {
          HLOG(kWarning, "DelBlob failed during tag deletion, continuing");
        }
        ++processed_blobs;
      }
    }

    // Step 6: erase blob-name mappings for all deleted tags.
    {
      clio::run::ScopedCoRwWriteLock lock(blob_map_lock_);
      std::vector<std::string> keys_to_erase;
      tag_blob_name_to_info_.for_each(
          [&](const std::string &compound_key, const BlobInfo &blob_info) {
            (void)blob_info;
            if (prefix_to_id.count(compound_prefix(compound_key)) != 0) {
              keys_to_erase.push_back(compound_key);
            }
          });
      for (const auto &key : keys_to_erase) {
        tag_blob_name_to_info_.erase(key);
      }
    }

    // Step 7: erase each tag's name binding(s) + aliases, WAL-log the delete,
    // and drop the TagInfo.
    size_t total_size = 0;
    const clio::run::u32 wid = tag_txn_logs_.empty()
                             ? 0
                             : CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
    for (const TagId &del_id : to_delete) {
      std::string del_name;
      {
        clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
        TagInfo *info = tag_id_to_info_.find(del_id);
        if (info != nullptr) {
          total_size += info->total_size_;
          del_name = info->tag_name_.str();
          if (!info->tag_name_.empty()) {
            tag_name_to_id_.erase(info->tag_name_.str());
          }
          // Cascade: remove every alias name bound to this tag.
          for (size_t i = 0; i < info->aliases_.size(); ++i) {
            tag_name_to_id_.erase(info->aliases_[i].str());
          }
        }
      }
      if (!tag_txn_logs_.empty()) {
        TxnDelTag txn;
        txn.tag_name_ = del_name;
        txn.tag_major_ = del_id.major_;
        txn.tag_minor_ = del_id.minor_;
        tag_txn_logs_[wid % tag_txn_logs_.size()]->Log(TxnType::kDelTag, txn);
      }
      {
        clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
        tag_id_to_info_.erase(del_id);
      }
      GpuCacheOnDelTag(del_id);
    }

    // Log telemetry for the DelTag operation (attributed to the target tag).
    auto now = GetCurrentTimeNs();
    LogTelemetry(CteOp::kDelTag, 0, total_size, tag_id, now, now);

    task->return_code_ = 0;
    HLOG(kDebug,
         "DelTag successful: tag_id={},{}, removed {} tags, {} blobs, "
         "total_size={}",
         tag_id.major_, tag_id.minor_, to_delete.size(), processed_blobs,
         total_size);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetTagName(ctp::ipc::FullPtr<GetTagNameTask> task,
                                    clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    TagId tag_id = task->tag_id_;
    std::string stored;
    bool found = false;
    {
      clio::run::ScopedCoRwReadLock lock(tag_map_lock_);
      TagInfo *info = tag_id_to_info_.find(tag_id);
      if (info != nullptr) {
        stored = info->tag_name_.str();
        found = true;
      }
      // ResolveTagName walks parent ids in tag_id_to_info_; keep it under the
      // same read lock so the hierarchy can't shift mid-resolution.
      if (found) {
        std::string full = ResolveTagName(stored);
        task->tag_name_ = clio::run::priv::string(CLIO_PRIV_ALLOC, full);
      }
    }
    task->found_ = found ? 1 : 0;
    task->return_code_ = 0;  // found_ conveys existence; the op itself is fine
  } catch (const std::exception &e) {
    HLOG(kError, "GetTagName failed: {}", e.what());
    task->return_code_ = 1;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetTagSize(ctp::ipc::FullPtr<GetTagSizeTask> task,
                                    clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    TagId tag_id = task->tag_id_;

    // Find the tag
    clio::run::ScopedCoRwWriteLock lock(tag_map_lock_);
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
    task->ctime_ = tag_info_ptr->last_changed_;  // surface ctime to getattr
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

clio::run::TaskResume Runtime::GetCapacity(
    ctp::ipc::FullPtr<GetCapacityTask> task, clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext &rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  // Sum the total and remaining capacity of every target registered on this
  // node. A Local query returns this node's capacity; the task's AggregateOut
  // sums replicas, so a Broadcast returns the whole cluster's capacity.
  //
  // Iterate registered_targets_ (the canonical map) rather than the target_list_
  // mirror: the PutBlob/Free data path debits/credits remaining_space_ on the
  // canonical entry only (the mirror is refreshed lazily by StatTargets), so the
  // mirror lags real usage. for_each takes the map's internal lock for a
  // consistent snapshot; target_lock_ (read) guards structural stability.
  clio::run::u64 total = 0;
  clio::run::u64 remaining = 0;
  {
    clio::run::ScopedCoRwReadLock read_lock(target_lock_);
    registered_targets_.for_each(
        [&total, &remaining](const clio::run::PoolId & /*key*/, const TargetInfo &t) {
          total += t.max_capacity_;
          remaining += t.remaining_space_;
        });
  }
  task->total_capacity_ = total;
  task->remaining_capacity_ = remaining;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetNumAliases(
    ctp::ipc::FullPtr<GetNumAliasesTask> task, clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext &rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    task->num_aliases_ = 0;
    task->found_ = 0;

    clio::run::ScopedCoRwReadLock lock(tag_map_lock_);

    // Resolve the tag id: prefer an explicit id, else resolve the name the same
    // way DelTag does — hierarchical "$tagid{parent}/leaf" key first, then a
    // verbatim lookup for flat names and flat aliases.
    TagId tag_id = task->tag_id_;
    if (tag_id.IsNull()) {
      std::string tag_name = task->tag_name_.str();
      if (!tag_name.empty()) {
        if (IsHierPath(tag_name)) {
          if (tag_name == "/") {
            TagId *r = tag_name_to_id_.find(std::string("/"));
            if (r != nullptr) tag_id = *r;
          } else {
            std::string parent_path, leaf;
            if (SplitParentLeaf(tag_name, parent_path, leaf)) {
              TagId parent_id = ResolvePathToIdLocked(parent_path);
              if (!parent_id.IsNull()) {
                TagId *p = tag_name_to_id_.find(MakeRelativeName(parent_id, leaf));
                if (p != nullptr) tag_id = *p;
              }
            }
          }
        }
        if (tag_id.IsNull()) {
          TagId *p = tag_name_to_id_.find(tag_name);
          if (p != nullptr) tag_id = *p;
        }
      }
    }

    if (!tag_id.IsNull()) {
      TagInfo *info = tag_id_to_info_.find(tag_id);
      if (info != nullptr) {
        task->num_aliases_ = static_cast<clio::run::u32>(info->aliases_.size());
        task->found_ = 1;
      }
    }
    task->return_code_ = 0;  // found_ conveys existence; the op itself is fine
  } catch (const std::exception &e) {
    HLOG(kError, "GetNumAliases failed: {}", e.what());
    task->return_code_ = 1;
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
  clio::run::ScopedCoRwWriteLock write_lock(tag_map_lock_);

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
  tag_info.last_changed_ = GetCurrentTimeNs();  // ctime at creation

  // Store mappings
  tag_name_to_id_.insert_or_assign(tag_name, tag_id);
  tag_id_to_info_.insert_or_assign(tag_id, tag_info);

  // Index the new tag's ABSOLUTE name for regex TagQuery (#598). The parent
  // chain is already created (GetOrCreateTagChain builds parents first), so
  // ResolveTagName yields the full path. We hold the write lock here.
  tag_search_.Insert(ResolveTagName(tag_name), tag_id);

  // WAL: log tag creation
  if (!tag_txn_logs_.empty()) {
    clio::run::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
    TxnCreateTag txn;
    txn.tag_name_ = tag_name;
    txn.tag_major_ = tag_id.major_;
    txn.tag_minor_ = tag_id.minor_;
    tag_txn_logs_[wid % tag_txn_logs_.size()]->Log(TxnType::kCreateTag, txn);
  }

  return tag_id;
}

std::string Runtime::ResolveTagName(const std::string &stored_name,
                                    int depth) {
  // Guard against pathological / cyclic parent references.
  if (depth > 256) {
    return stored_name;
  }
  TagId parent;
  std::string leaf;
  if (!ParseTagRef(stored_name, parent, leaf)) {
    // Flat name, root "/", or a legacy absolute name stored verbatim.
    return stored_name;
  }
  TagInfo *pinfo = tag_id_to_info_.find(parent);
  if (pinfo == nullptr) {
    // Dangling parent (e.g. partially-replayed metadata). Best effort: present
    // the leaf as a top-level name so the result is still a usable path.
    return "/" + leaf;
  }
  std::string parent_full = ResolveTagName(pinfo->tag_name_.str(), depth + 1);
  return JoinPath(parent_full, leaf);
}

TagId Runtime::ResolvePathToIdLocked(const std::string &path) {
  // Walk from the root, looking up each component's relative key. Returns null
  // if the root or any intermediate component does not exist.
  TagId *root = tag_name_to_id_.find(std::string("/"));
  if (root == nullptr) {
    return TagId::GetNull();
  }
  TagId cur = *root;
  for (const std::string &comp : SplitPathComponents(path)) {
    TagId *child = tag_name_to_id_.find(MakeRelativeName(cur, comp));
    if (child == nullptr) {
      return TagId::GetNull();
    }
    cur = *child;
  }
  return cur;
}

void Runtime::RebuildTagSearchIndexLocked() {
  // Repopulate the regex search index from the authoritative tag table (#598).
  // Order-independent: every tag's parents are present, so ResolveTagName always
  // yields the full path. Used after WAL/metadata restore, where tags are
  // inserted directly (bypassing GetOrAssignTagId's per-insert indexing).
  tag_search_.Clear();
  tag_id_to_info_.for_each([&](const TagId & /*id*/, const TagInfo &info) {
    std::string abs = ResolveTagName(info.tag_name_.str());
    if (!abs.empty()) {
      tag_search_.Insert(abs, info.tag_id_);
    }
  });
}

TagId Runtime::GetOrCreateTagChain(const std::string &name,
                                   const TagId &preferred_id) {
  if (!IsHierPath(name)) {
    // Non-path tag: a single flat tag stored verbatim (legacy behavior).
    return GetOrAssignTagId(name, preferred_id);
  }
  // Every absolute path is rooted at the "/" tag (stored literally).
  TagId parent = GetOrAssignTagId(std::string("/"));
  std::vector<std::string> comps = SplitPathComponents(name);
  if (comps.empty()) {
    // name was "/" (or all slashes): the root tag itself.
    return parent;
  }
  for (size_t i = 0; i < comps.size(); ++i) {
    const bool is_leaf = (i + 1 == comps.size());
    // preferred_id (a cross-node hint) only applies to the deepest tag.
    TagId id = GetOrAssignTagId(MakeRelativeName(parent, comps[i]),
                                is_leaf ? preferred_id : TagId::GetNull());
    parent = id;
  }
  return parent;
}

clio::run::TaskResume Runtime::FlushMetadata(ctp::ipc::FullPtr<FlushMetadataTask> task,
                                       clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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
      clio::run::u64 total_size = info.total_size_;
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
      clio::run::u64 trace_key = blob_info.trace_key_;
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
        clio::run::u32 bdev_major = block.bdev_client_.pool_id_.major_;
        clio::run::u32 bdev_minor = block.bdev_client_.pool_id_.minor_;
        ofs.write(reinterpret_cast<const char *>(&bdev_major),
                  sizeof(bdev_major));
        ofs.write(reinterpret_cast<const char *>(&bdev_minor),
                  sizeof(bdev_minor));

        // Write target_query as raw bytes (POD-like struct)
        ofs.write(reinterpret_cast<const char *>(&block.target_query_),
                  sizeof(clio::run::PoolQuery));

        clio::run::u64 offset = block.target_offset_;
        clio::run::u64 size = block.size_;
        ofs.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
        ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
      }
      task->entries_flushed_++;
    });

    ofs.close();

    // WAL: sync and compact transaction logs after snapshot
    if (!blob_txn_logs_.empty()) {
      clio::run::u64 total_wal_size = 0;
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

clio::run::TaskResume Runtime::FlushData(ctp::ipc::FullPtr<FlushDataTask> task,
                                   clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  task->bytes_flushed_ = 0;
  task->blobs_flushed_ = 0;

  int target_level = task->target_persistence_level_;

  // Find non-volatile targets that meet the persistence level requirement
  std::vector<clio::run::PoolId> nonvolatile_targets;
  {
    clio::run::ScopedCoRwReadLock read_lock(target_lock_);
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
    clio::run::u64 total_size;
    float score;
  };
  std::vector<FlushEntry> blobs_to_flush;

  {
    clio::run::ScopedCoRwReadLock read_lock(target_lock_);
    tag_blob_name_to_info_.for_each([&](const std::string &key,
                                        const BlobInfo &blob_info) {
      if (blob_info.blocks_.empty()) return;

      bool has_volatile_blocks = false;
      for (const auto &block : blob_info.blocks_) {
        clio::run::PoolId pool_id = block.bdev_client_.pool_id_;
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
              static_cast<clio::run::u32>(std::stoul(key.substr(0, first_dot)));
          entry.tag_id.minor_ = static_cast<clio::run::u32>(std::stoul(
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

    clio::run::u64 total_size = entry.total_size;
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
    clio::run::u32 read_error = 0;
    CLIO_CO_AWAIT(ReadData(blob_info_ptr->blocks_, shm_ptr, total_size, 0,
                      read_error));
    if (read_error != 0) {
      HLOG(kError, "FlushData: Failed to read blob data for {}",
           entry.blob_name);
      ipc_manager->FreeBuffer(buffer);
      continue;
    }

    // Step 2: Free only volatile blocks
    clio::run::priv::vector<BlobBlock> nonvolatile_blocks(CLIO_PRIV_ALLOC);
    std::unordered_map<
        clio::run::PoolId,
        std::pair<clio::run::PoolQuery, std::vector<clio::run::bdev::Block>>>
        volatile_blocks_by_pool;

    {
      clio::run::ScopedCoRwReadLock read_lock(target_lock_);
      for (const auto &block : blob_info_ptr->blocks_) {
        clio::run::PoolId pool_id = block.bdev_client_.pool_id_;
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
      const clio::run::PoolId &pool_id = pool_entry.first;
      const clio::run::PoolQuery &target_query = pool_entry.second.first;
      const std::vector<clio::run::bdev::Block> &blocks =
          pool_entry.second.second;

      clio::run::u64 bytes_freed = 0;
      for (const auto &block : blocks) {
        bytes_freed += block.size_;
      }

      clio::run::bdev::Client bdev_client(pool_id);
      auto free_task = bdev_client.AsyncFreeBlocks(target_query, blocks);
      CLIO_CO_AWAIT(free_task);
      if (free_task->GetReturnCode() == 0) {
        clio::run::ScopedCoRwWriteLock write_lock(target_lock_);
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

  clio::run::u32 max_minor = 0;
  clio::run::u32 tags_restored = 0;
  clio::run::u32 blobs_restored = 0;

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
      clio::run::u64 total_size;
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
      clio::run::u64 trace_key;
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
        clio::run::u32 bdev_major, bdev_minor;
        ifs.read(reinterpret_cast<char *>(&bdev_major), sizeof(bdev_major));
        ifs.read(reinterpret_cast<char *>(&bdev_minor), sizeof(bdev_minor));

        // Read target_query as raw bytes (POD-like struct)
        clio::run::PoolQuery target_query;
        ifs.read(reinterpret_cast<char *>(&target_query),
                 sizeof(clio::run::PoolQuery));

        clio::run::u64 offset, size;
        ifs.read(reinterpret_cast<char *>(&offset), sizeof(offset));
        ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

        if (!ifs.good()) break;

        // Filter by persistence level: skip volatile blocks
        clio::run::PoolId bdev_pool_id(bdev_major, bdev_minor);
        bool is_volatile = false;
        {
          clio::run::ScopedCoRwReadLock read_lock(target_lock_);
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
  clio::run::u32 current_minor = next_tag_id_minor_.load();
  if (max_minor > current_minor) {
    next_tag_id_minor_.store(max_minor);
  }

  HLOG(kInfo, "RestoreMetadataFromLog: Restored {} tags and {} blobs from {}",
       tags_restored, blobs_restored, log_path);
}

void Runtime::ReplayTransactionLogs() {
  const std::string &log_path = config_.performance_.metadata_log_path_;
  if (log_path.empty()) return;

  clio::run::u32 tags_replayed = 0;
  clio::run::u32 blobs_replayed = 0;
  clio::run::u32 max_minor = next_tag_id_minor_.load();

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
            clio::run::PoolId bdev_pool_id(tb.bdev_major_, tb.bdev_minor_);
            // Filter volatile targets (matching RestoreMetadataFromLog)
            bool is_volatile = false;
            {
              clio::run::ScopedCoRwReadLock read_lock(target_lock_);
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
    clio::run::u64 total = 0;
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
  clio::run::u32 current_minor = next_tag_id_minor_.load();
  if (max_minor > current_minor) {
    next_tag_id_minor_.store(max_minor);
  }

  HLOG(kInfo, "ReplayTransactionLogs: Replayed {} tag ops and {} blob ops",
       tags_replayed, blobs_replayed);
}

// GetWorkRemaining implementation (required pure virtual method)
clio::run::u64 Runtime::GetWorkRemaining() const {
  // Return approximate work remaining (simple implementation)
  // In a real implementation, this would sum tasks across all queues
  return 0;  // For now, always return 0 work remaining
}

clio::run::TaskStat Runtime::GetTaskStats(const clio::run::Task *task) const {
  if (!task) return clio::run::TaskStat();
  switch (task->method_) {
    case Method::kPutBlob: {
      auto *t = static_cast<const PutBlobTask *>(task);
      clio::run::TaskStat stat;
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
      clio::run::TaskStat stat;
      stat.io_size_ = t->size_;
      stat.wall_time_ =
          static_cast<float>(t->size_) / 500.0f;
      return stat;
    }
    default:
      return clio::run::TaskStat();
  }
}

// Helper methods for lock index calculation
size_t Runtime::GetTargetLockIndex(const clio::run::PoolId &target_id) const {
  // Use hash of target_id to distribute locks evenly
  std::hash<clio::run::PoolId> hasher;
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
  // std::hash<clio::run::UniqueId> is defined in types.h
  std::hash<TagId> hasher;
  return hasher(tag_id) % tag_locks_.size();
}

TagId Runtime::GenerateNewTagId() {
  // Get node_id from IPC manager as the major component
  auto *ipc_manager = CLIO_IPC;
  clio::run::u32 node_id = ipc_manager->GetNodeId();

  // Get next minor component from atomic counter
  clio::run::u32 minor_id = next_tag_id_minor_.fetch_add(1);

  return TagId{node_id, minor_id};
}

// Explicit template instantiations for required template methods
template clio::run::TaskResume Runtime::GetOrCreateTag<CreateParams>(
    ctp::ipc::FullPtr<GetOrCreateTagTask<CreateParams>> task, clio::run::RunContext &ctx);

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
  clio::run::ScopedCoRwReadLock lock(blob_map_lock_);

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
    clio::run::ScopedCoRwWriteLock lock(blob_map_lock_);

    // Store blob info directly in tag_blob_name_to_info_
    auto insert_result =
        tag_blob_name_to_info_.insert_or_assign(composite_key, new_blob_info);
    blob_info_ptr = insert_result.value;
  }  // Release lock immediately after insertion

  // WAL: log blob creation
  if (!blob_txn_logs_.empty()) {
    clio::run::u32 wid = CLIO_CUR_WORKER->GetWorkerStats().worker_id_;
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

clio::run::TaskResume Runtime::ExtendBlob(BlobInfo &blob_info, clio::run::u64 offset,
                                    clio::run::u64 size, float blob_score,
                                    clio::run::u32 &error_code,
                                    int min_persistence_level) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  // Calculate required additional space
  clio::run::u64 current_blob_size = blob_info.GetTotalSize();
  clio::run::u64 required_size = offset + size;

  if (required_size <= current_blob_size) {
    // No additional allocation needed
    error_code = 0;
    CLIO_CO_RETURN;
  }

  clio::run::u64 additional_size = required_size - current_blob_size;

  // Snapshot available targets for the DPE. target_list_ is the contiguous
  // mirror of registered_targets_ — copying it under the read lock is O(N_live)
  // with no map iteration over empty slots.
  std::vector<TargetInfo> available_targets;
  {
    clio::run::ScopedCoRwReadLock read_lock(target_lock_);
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
  clio::run::u64 remaining_to_allocate = additional_size;
  for (const auto &selected_target_info : ordered_targets) {
    if (remaining_to_allocate == 0) {
      break;
    }

    clio::run::PoolId selected_target_id = selected_target_info.bdev_client_.pool_id_;

    // Copy target info under lock (can't hold lock across co_await)
    TargetInfo target_info_copy;
    bool found = false;
    {
      clio::run::ScopedCoRwReadLock read_lock(target_lock_);
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
    clio::run::u64 allocate_size =
        std::min(remaining_to_allocate, target_info_copy.remaining_space_);

    if (allocate_size == 0) {
      continue;
    }

    // Allocate space using bdev client
    clio::run::u64 allocated_offset;
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
      clio::run::ScopedCoRwReadLock read_lock(target_lock_);
      TargetInfo *ti = registered_targets_.find(selected_target_id);
      if (ti != nullptr) {
        ctp::ipc::atomic_ref<clio::run::u64> rs(ti->remaining_space_);
        clio::run::u64 cur = rs.load(std::memory_order_relaxed);
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

clio::run::TaskResume Runtime::ResizeBlob(BlobInfo &blob_info, clio::run::u64 new_size,
                                    float blob_score, clio::run::u32 &error_code,
                                    int min_persistence_level) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  error_code = 0;
  clio::run::u64 current_size = blob_info.GetTotalSize();
  if (new_size == current_size) {
    CLIO_CO_RETURN;
  }
  if (new_size > current_size) {
    // Grow: allocate appended blocks up to new_size (shared with ExtendBlob)...
    CLIO_CO_AWAIT(ExtendBlob(blob_info, 0, new_size, blob_score, error_code,
                             min_persistence_level));
    if (error_code != 0) {
      CLIO_CO_RETURN;
    }
    // ...then zero the grown region. Newly allocated blocks are not zeroed
    // (the bdev recycles freed blocks), and a resize-grow (e.g. ftruncate up,
    // or a truncate-down whose target exceeds the blob's physical extent after
    // sparse writes) must read back as zeros. Unlike PutBlob, no caller data
    // overwrites this region, so zero all of [current_size, new_size).
    clio::run::u64 grow = new_size - current_size;
    auto *ipc_mgr = CLIO_IPC;
    ctp::ipc::FullPtr<char> zbuf = ipc_mgr->AllocateBuffer(grow);
    if (zbuf.IsNull()) {
      error_code = 4;
      CLIO_CO_RETURN;
    }
    std::memset(zbuf.ptr_, 0, grow);
    clio::run::u32 zero_result = 0;
    CLIO_CO_AWAIT(ModifyExistingData(blob_info.blocks_,
                                zbuf.shm_.template Cast<void>(), grow,
                                current_size, zero_result));
    ipc_mgr->FreeBuffer(zbuf);
    error_code = zero_result;
    CLIO_CO_RETURN;
  }

  // Shrink: keep the blocks covering [0, new_size); free the rest. The block
  // straddling new_size is trimmed logically (its physical tail stays
  // allocated within that block and is reclaimed when the block is freed).
  std::vector<BlobBlock> keep;
  std::vector<BlobBlock> drop;
  clio::run::u64 block_start = 0;
  for (size_t i = 0; i < blob_info.blocks_.size(); ++i) {
    BlobBlock blk = blob_info.blocks_[i];
    clio::run::u64 block_end = block_start + blk.size_;
    if (block_start >= new_size) {
      drop.push_back(blk);  // entirely beyond the new end
    } else if (block_end > new_size) {
      blk.size_ = new_size - block_start;  // boundary block: trim
      keep.push_back(blk);
    } else {
      keep.push_back(blk);
    }
    block_start = block_end;
  }

  // Rebuild blocks_ with only the kept blocks.
  blob_info.blocks_.clear();
  for (auto &b : keep) {
    blob_info.blocks_.push_back(b);
  }

  // Free the dropped blocks, grouped by pool, and credit remaining_space_
  // (mirrors FreeAllBlobBlocks).
  std::unordered_map<clio::run::PoolId, std::pair<clio::run::PoolQuery,
                                            std::vector<clio::run::bdev::Block>>>
      blocks_by_pool;
  for (const auto &blob_block : drop) {
    clio::run::PoolId pool_id = blob_block.bdev_client_.pool_id_;
    clio::run::bdev::Block block;
    block.offset_ = blob_block.target_offset_;
    block.size_ = blob_block.size_;
    block.block_type_ = 0;
    if (blocks_by_pool.find(pool_id) == blocks_by_pool.end()) {
      blocks_by_pool[pool_id] = std::make_pair(
          blob_block.target_query_, std::vector<clio::run::bdev::Block>());
    }
    blocks_by_pool[pool_id].second.push_back(block);
  }
  for (const auto &pool_entry : blocks_by_pool) {
    const clio::run::PoolId &pool_id = pool_entry.first;
    const clio::run::PoolQuery &target_query = pool_entry.second.first;
    const std::vector<clio::run::bdev::Block> &blocks = pool_entry.second.second;
    clio::run::u64 bytes_freed = 0;
    for (const auto &block : blocks) {
      bytes_freed += block.size_;
    }
    clio::run::bdev::Client bdev_client(pool_id);
    auto free_task = bdev_client.AsyncFreeBlocks(target_query, blocks);
    CLIO_CO_AWAIT(free_task);
    if (free_task->GetReturnCode() == 0) {
      clio::run::ScopedCoRwReadLock read_lock(target_lock_);
      TargetInfo *target_info = registered_targets_.find(pool_id);
      if (target_info != nullptr) {
        ctp::ipc::atomic_ref<clio::run::u64>(target_info->remaining_space_)
            .fetch_add(bytes_freed, std::memory_order_relaxed);
      }
    }
  }
  error_code = 0;
  CLIO_CO_RETURN;
}

clio::run::TaskResume Runtime::ModifyExistingData(
    const clio::run::priv::vector<BlobBlock> &blocks, ctp::ipc::ShmPtr<> data, size_t data_size,
    size_t data_offset_in_blob, clio::run::u32 &error_code) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
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
  std::vector<clio::run::Future<clio::run::bdev::WriteTask>> write_tasks;
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

      // Wrap single block in clio::run::priv::vector for AsyncWrite
      timer.Resume();
      clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
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

clio::run::TaskResume Runtime::ReadData(const clio::run::priv::vector<BlobBlock> &blocks,
                                  ctp::ipc::ShmPtr<> data, size_t data_size,
                                  size_t data_offset_in_blob,
                                  clio::run::u32 &error_code) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  HLOG(kDebug, "ReadData: blocks={}, data_size={}, data_offset_in_blob={}",
       blocks.size(), data_size, data_offset_in_blob);

  // Step 1: Initially store the remaining_size equal to data_size
  size_t remaining_size = data_size;

  // Vector to store async read tasks for later waiting
  std::vector<clio::run::Future<clio::run::bdev::ReadTask>> read_tasks;
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

      // Wrap single block in clio::run::priv::vector for AsyncRead
      clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
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

clio::run::TaskResume Runtime::AllocateFromTarget(TargetInfo &target_info,
                                            clio::run::u64 size,
                                            clio::run::u64 &allocated_offset,
                                            bool &success) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
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

clio::run::TaskResume Runtime::ClearBlob(BlobInfo &blob_info, float blob_score,
                                   clio::run::u64 offset, clio::run::u64 size,
                                   bool &cleared) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  cleared = false;
  // Score must be in [0, 1]
  if (blob_score < 0.0f || blob_score > 1.0f) {
    CLIO_CO_RETURN;
  }
  // Must be full-blob replacement (offset == 0 with non-empty blob)
  clio::run::u64 current_size = blob_info.GetTotalSize();
  if (offset != 0 || current_size == 0) {
    CLIO_CO_RETURN;
  }
  // Free all existing blocks
  clio::run::u32 free_result = 0;
  CLIO_CO_AWAIT(FreeAllBlobBlocks(blob_info, free_result));
  if (free_result == 0) {
    cleared = true;
  }
  CLIO_CO_RETURN;
}

clio::run::TaskResume Runtime::FreeAllBlobBlocks(BlobInfo &blob_info,
                                           clio::run::u32 &error_code) {
#ifdef __NVCOMPILER
  thread_local clio::run::RunContext _fb_rctx;
  clio::run::RunContext* _fp = clio::run::GetCurrentRunContextFromWorker();
  clio::run::RunContext& rctx = _fp ? *_fp : _fb_rctx;
#endif
  // Map: PoolId -> (target_query, vector<Block>)
  std::unordered_map<clio::run::PoolId, std::pair<clio::run::PoolQuery,
                                            std::vector<clio::run::bdev::Block>>>
      blocks_by_pool;

  // Group blocks by PoolId
  for (const auto &blob_block : blob_info.blocks_) {
    clio::run::PoolId pool_id = blob_block.bdev_client_.pool_id_;
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
    const clio::run::PoolId &pool_id = pool_entry.first;
    const clio::run::PoolQuery &target_query = pool_entry.second.first;
    const std::vector<clio::run::bdev::Block> &blocks = pool_entry.second.second;

    // Calculate total bytes to be freed for this pool
    clio::run::u64 bytes_freed = 0;
    for (const auto &block : blocks) {
      bytes_freed += block.size_;
    }

    // Get bdev client for this pool from first blob block
    clio::run::bdev::Client bdev_client(pool_id);
    auto free_task = bdev_client.AsyncFreeBlocks(target_query, blocks);
    CLIO_CO_AWAIT(free_task);
    clio::run::u32 free_result = free_task->GetReturnCode();
    if (free_result != 0) {
      HLOG(kWarning, "Failed to free blocks from pool {}", pool_id.major_);
    } else {
      // Successfully freed blocks - credit target's remaining_space_.
      // Shared READ lock only: registered_targets_ is structurally
      // stationary on the data path; the counter is bumped lock-free
      // via ctp::ipc::atomic_ref (no exclusive lock for an integer add).
      clio::run::ScopedCoRwReadLock read_lock(target_lock_);
      TargetInfo *target_info = registered_targets_.find(pool_id);
      if (target_info != nullptr) {
        clio::run::u64 now =
            ctp::ipc::atomic_ref<clio::run::u64>(target_info->remaining_space_)
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

clio::run::TaskResume Runtime::PollTelemetryLog(
    ctp::ipc::FullPtr<PollTelemetryLogTask> task, clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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

clio::run::TaskResume Runtime::GetBlobScore(ctp::ipc::FullPtr<GetBlobScoreTask> task,
                                      clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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

clio::run::TaskResume Runtime::GetBlobSize(ctp::ipc::FullPtr<GetBlobSizeTask> task,
                                     clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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

clio::run::TaskResume Runtime::GetBlobInfo(ctp::ipc::FullPtr<GetBlobInfoTask> task,
                                     clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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

clio::run::TaskResume Runtime::GetContainedBlobs(
    ctp::ipc::FullPtr<GetContainedBlobsTask> task, clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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

clio::run::TaskResume Runtime::TagQuery(ctp::ipc::FullPtr<TagQueryTask> task,
                                  clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  try {
    std::string tag_regex = task->tag_regex_.str();

    // Query the secondary search index (#598): its keys are the absolute tag
    // names, so a regex match over them needs no per-tag ResolveTagName scan.
    // The index derives the trigrams every match must contain and only verifies
    // a small candidate set against the full regex (falling back to a scan of
    // its own entries when the pattern has no usable trigram). Held under the
    // tag map read lock since the index is mutated under the write lock.
    task->results_.clear();
    task->result_ids_.clear();
    size_t total = 0;
    {
      clio::run::ScopedCoRwReadLock lock(tag_map_lock_);
      auto result = tag_search_.Search(tag_regex);
      total = result.size();
      // Iterate (name, TagId) pairs so each result carries a packed id, used by
      // the filesystem readdir to assign a stable inode without a second lookup.
      for (const auto &kv : result) {
        if (task->max_tags_ != 0 && task->results_.size() >= task->max_tags_) {
          break;
        }
        task->results_.push_back(kv.first);
        const TagId &id = kv.second;
        task->result_ids_.push_back(
            (static_cast<clio::run::u64>(id.major_) << 32) |
            static_cast<clio::run::u64>(id.minor_));
      }
    }

    // Total matched tags (summed across replicas during AggregateOut)
    task->total_tags_matched_ = total;

    // Success
    task->return_code_ = 0;
    HLOG(kDebug, "TagQuery successful: pattern={}, found {} tags", tag_regex,
         total);

  } catch (const std::exception &e) {
    task->return_code_ = 1;
    HLOG(kError, "TagQuery failed: {}", e.what());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::BlobQuery(ctp::ipc::FullPtr<BlobQueryTask> task,
                                   clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext& rctx = ctx;
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

    // Find matching tag IDs and resolved names.
    std::vector<std::pair<std::string, TagId>> matching_tags;
    tag_name_to_id_.for_each(
        [&](const std::string &stored, const TagId &tag_id) {
          std::string full = ResolveTagName(stored);
          if (std::regex_match(full, tag_pattern)) {
            matching_tags.emplace_back(full, tag_id);
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

clio::run::TaskResume Runtime::SemanticSearch(
    ctp::ipc::FullPtr<SemanticSearchTask> task, clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext &rctx = ctx;
#else
  (void)ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  task->results_.clear();
  task->return_code_ = 0;

  std::string tag_regex_str = task->tag_regex_.str();
  std::string blob_regex_str = task->blob_regex_.str();
  std::string query_text = task->query_text_.str();
  clio::run::u32 k = task->k_;

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
      [&](const std::string &stored, const TagId &tag_id) {
        std::string full = ResolveTagName(stored);
        if (std::regex_match(full, tag_pattern)) {
          matching_tags.emplace_back(full, tag_id);
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
    clio::run::u64 total = info->GetTotalSize();
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
    clio::run::u32 read_rc = 0;
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

clio::run::TaskResume Runtime::TemporalSearch(
    ctp::ipc::FullPtr<TemporalSearchTask> task, clio::run::RunContext &ctx) {
#ifdef __NVCOMPILER
  clio::run::RunContext &rctx = ctx;
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
  clio::run::u32 max_entries = task->max_entries_;

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
      [&](const std::string &stored, const TagId &tag_id) {
        std::string full = ResolveTagName(stored);
        if (std::regex_match(full, tag_pattern)) {
          matching_tags.emplace_back(full, tag_id);
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

clio::run::PoolQuery Runtime::HashBlobToContainer(const TagId &tag_id,
                                            const std::string &blob_name) {
  // Compute hash from tag_id and blob_name
  std::hash<std::string> string_hasher;
  std::hash<clio::run::u32> u32_hasher;

  // Combine tag_id major, minor, and blob_name into a single hash
  clio::run::u32 hash_value = u32_hasher(tag_id.major_);
  hash_value ^= u32_hasher(tag_id.minor_) + 0x9e3779b9 + (hash_value << 6) +
                (hash_value >> 2);
  hash_value ^= static_cast<clio::run::u32>(string_hasher(blob_name)) + 0x9e3779b9 +
                (hash_value << 6) + (hash_value >> 2);

  return clio::run::PoolQuery::DirectHash(hash_value);
}

clio::run::TaskResume Runtime::Monitor(ctp::ipc::FullPtr<MonitorTask> task,
                                 clio::run::RunContext &rctx) {
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
  clio::run::u32 max_tags = config_.gpu_metadata_cache_.max_tags_;
  clio::run::u32 max_blobs = config_.gpu_metadata_cache_.max_blobs_;
  size_t needed = GpuMetadataCacheHeader::Layout(max_tags, max_blobs);
  size_t cap = static_cast<size_t>(config_.gpu_metadata_cache_.capacity_bytes_);
  if (needed > cap) {
    // Shrink slot counts proportionally so we stay within budget.
    double scale =
        static_cast<double>(cap - sizeof(GpuMetadataCacheHeader)) /
        static_cast<double>(needed - sizeof(GpuMetadataCacheHeader));
    if (scale < 0.0) scale = 0.0;
    if (scale > 1.0) scale = 1.0;
    max_tags = std::max<clio::run::u32>(
        1u, static_cast<clio::run::u32>(static_cast<double>(max_tags) * scale));
    max_blobs = std::max<clio::run::u32>(
        1u, static_cast<clio::run::u32>(static_cast<double>(max_blobs) * scale));
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
  clio::run::u32 sc = gpu_cache::BdevTypeToStorageClass(bdev_type.c_str());
  clio::run::u64 size = blob_info.GetTotalSize();
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
  clio::run::ScopedCoRwReadLock lock(target_lock_);
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
