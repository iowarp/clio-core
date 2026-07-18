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

#ifndef WRPCTE_CORE_TASKS_H_
#define WRPCTE_CORE_TASKS_H_

#include <algorithm>
#include <cassert>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>
#include <clio_cte/core/core_config.h>
// Include admin tasks for GetOrCreatePoolTask
#include <clio_runtime/admin/admin_tasks.h>
// Include bdev tasks for BdevType enum
#include <clio_runtime/bdev/bdev_tasks.h>
// Include bdev client for TargetInfo
#include <clio_runtime/bdev/bdev_client.h>
// Include mutex for BlobInfo prealloc_lock_
#include <clio_ctp/thread/lock/mutex.h>
// Include atomic_ref for BlobInfo write_owner_ (per-blob async write lock)
#include <clio_ctp/types/atomic.h>
#if CTP_IS_HOST
#include <yaml-cpp/yaml.h>

#include <clio_ctp/data_structures/serialization/global_serialize.h>
#include <chrono>
#endif

namespace clio::cte::core {

using MonitorTask = clio::run::admin::MonitorTask;

// CTE Core Pool ID constant (major: 512, minor: 0)
static constexpr clio::run::PoolId kCtePoolId(512, 0);

// CTE Core Pool Name constant
static constexpr const char *kCtePoolName = "clio_cte_core";

// Timestamp type definition - nanoseconds since epoch (0 on GPU)
using Timestamp = clio::run::u64;

#if CTP_IS_GPU_COMPILER
// CUDA/ROCm GPU compiler: CROSS_FUN so __device__ bodies can call it in
// both passes. Use __CUDA_ARCH__ (defined in device pass only) to select
// the right impl.
CTP_CROSS_FUN inline Timestamp GetCurrentTimeNs() {
#ifdef __CUDA_ARCH__
  return 0;  // GPU device code: return 0 (no clock available)
#else
  return static_cast<clio::run::u64>(
      std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}
#elif CTP_IS_SYCL_COMPILER
// SYCL: device pass has no portable clock and the SYCL frontend traces
// through inline functions during kernel parsing, so std::chrono is
// unreachable. Pick by CTP_IS_DEVICE_PASS — host pass uses chrono,
// device pass returns 0. Both branches must compile under the SYCL
// driver since it processes the file twice.
inline Timestamp GetCurrentTimeNs() {
#if CTP_IS_DEVICE_PASS
  return 0;
#else
  return static_cast<clio::run::u64>(
      std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}
#else
inline Timestamp GetCurrentTimeNs() {
  return static_cast<clio::run::u64>(
      std::chrono::steady_clock::now().time_since_epoch().count());
}
#endif

/**
 * CreateParams for CTE Core chimod
 * Contains configuration parameters for CTE container creation
 */
struct CreateParams {
  // CTE configuration object (not serialized, loaded from pool_config)
  Config config_;

  // OUT: Raw pointer (uintptr_t cast) to GpuMetadataCacheHeader allocated by
  // the server during Create. Zero when the GPU metadata cache is disabled
  // or no GPU backend is built in. The pointer is a managed/shared USM
  // address valid for both host and device access in the SAME process; it
  // is NOT a cross-process IPC handle (cross-process sharing requires
  // Level-Zero IPC on SYCL or cudaIpc on CUDA — see
  // gpu_metadata_cache.h header for the eventual extension).
  clio::run::u64 gpu_cache_ptr_ = 0;

  // Required: chimod library name for module manager
  static constexpr const char *chimod_lib_name = "clio_cte_core";

  // Default constructor
  CreateParams() {}

  // Copy constructor (required for task creation)
  CreateParams(const CreateParams &other)
      : config_(other.config_),
        gpu_cache_ptr_(other.gpu_cache_ptr_) {}

  // Constructor with pool_id and CreateParams (required for admin
  // task creation)
  CreateParams(const clio::run::PoolId &pool_id, const CreateParams &other)
      : config_(other.config_),
        gpu_cache_ptr_(other.gpu_cache_ptr_) {
    // pool_id is used by the admin task framework, but we don't need to store
    // it
    (void)pool_id;  // Suppress unused parameter warning
  }

#if CTP_IS_HOST
  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive &ar) {
    // Most of Config is loaded server-side from pool_config.config_ via
    // LoadConfig (compose mode). The GPU metadata cache settings are
    // serialized explicitly so callers can opt in directly via
    // CreateParams (no compose YAML required) — the server reads them
    // from CreateParams::config_.gpu_metadata_cache_.
    //
    // gpu_cache_ptr_ flows back on the OUT path: the server writes the
    // managed-USM cache pointer into chimod_params_ at the end of
    // Create, and the client's GetParams() returns it.
    ar(config_.gpu_metadata_cache_.enabled_,
       config_.gpu_metadata_cache_.capacity_bytes_,
       config_.gpu_metadata_cache_.max_blobs_,
       config_.gpu_metadata_cache_.max_tags_,
       config_.performance_.stat_targets_period_ms_,
       config_.organizer_.name_,
       config_.organizer_.organizer_tasks_,
       config_.organizer_.period_ms_,
       gpu_cache_ptr_);
  }

  /**
   * Load configuration from PoolConfig (for compose mode)
   * Required for compose feature support
   * @param pool_config Pool configuration from compose section
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    // The pool_config.config_ contains the full CTE configuration YAML
    // in the format of config/cte_config.yaml (targets, storage, dpe sections).
    // Parse it directly into the Config object
    HLOG(kDebug, "CTE CreateParams::LoadConfig() - config string length: {}",
         pool_config.config_.length());
    if (!pool_config.config_.empty()) {
      bool success = config_.LoadFromString(pool_config.config_);
      if (!success) {
        HLOG(kError,
             "CTE CreateParams::LoadConfig() - Failed to load config from "
             "string");
      } else {
        HLOG(kDebug,
             "CTE CreateParams::LoadConfig() - Successfully loaded config with "
             "{} storage devices",
             config_.storage_.devices_.size());
      }
    } else {
      HLOG(kWarning,
           "CTE CreateParams::LoadConfig() - Empty config string provided");
    }
  }
#endif
};

/**
 * CreateTask - Initialize the CTE Core container
 * Type alias for GetOrCreatePoolTask with CreateParams (uses kGetOrCreatePool
 * method) Non-admin modules should use GetOrCreatePoolTask instead of
 * BaseCreateTask
 */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;

/**
 * DestroyTask - Destroy the CTE Core container
 * Type alias for DestroyPoolTask (uses kDestroy method)
 */
using DestroyTask = clio::run::admin::DestroyTask;

/**
 * Target information structure
 */
struct TargetInfo {
  clio::run::priv::string target_name_;
  clio::run::priv::string bdev_pool_name_;
  clio::run::bdev::Client bdev_client_;  // Bdev client for this target
  clio::run::PoolQuery target_query_;         // Target pool query for bdev API calls
  clio::run::u64 bytes_read_;
  clio::run::u64 bytes_written_;
  clio::run::u64 ops_read_;
  clio::run::u64 ops_written_;
  float target_score_;        // Target score (0-1, normalized log bandwidth)
  clio::run::u64 remaining_space_;  // Remaining allocatable space in bytes
  clio::run::u64 max_capacity_;     // Total (max) capacity in bytes, fixed at register
  clio::run::bdev::PerfMetrics perf_metrics_;  // Performance metrics from bdev
  clio::run::u32 expected_ttl_days_; // Predictive device TTL (defaults to 999999)
  clio::run::bdev::PersistenceLevel persistence_level_;
  // Underlying bdev type, captured at RegisterTarget time. Used by the
  // GPU metadata cache projection to decide whether a blob landed in a
  // GPU-reachable tier (kRam / kHbm / kPinned).
  clio::run::bdev::BdevType bdev_type_;

  CTP_CROSS_FUN TargetInfo()
      : target_name_(CLIO_PRIV_ALLOC),
        bdev_pool_name_(CLIO_PRIV_ALLOC),
        bytes_read_(0),
        bytes_written_(0),
        ops_read_(0),
        ops_written_(0),
        target_score_(0.0f),
        remaining_space_(0),
        max_capacity_(0),
        expected_ttl_days_(999999),
        persistence_level_(clio::run::bdev::PersistenceLevel::kVolatile),
        bdev_type_(clio::run::bdev::BdevType::kFile) {}

#if CTP_IS_HOST
  TargetInfo(const std::string &name, const std::string &bdev_name)
      : target_name_(CLIO_PRIV_ALLOC, name),
        bdev_pool_name_(CLIO_PRIV_ALLOC, bdev_name),
        bytes_read_(0),
        bytes_written_(0),
        ops_read_(0),
        ops_written_(0),
        target_score_(0.0f),
        remaining_space_(0),
        max_capacity_(0),
        expected_ttl_days_(999999),
        persistence_level_(clio::run::bdev::PersistenceLevel::kVolatile) {}
#endif

  CTP_CROSS_FUN TargetInfo(const TargetInfo &other)
      : target_name_(other.target_name_),
        bdev_pool_name_(other.bdev_pool_name_),
        bdev_client_(other.bdev_client_),
        target_query_(other.target_query_),
        bytes_read_(other.bytes_read_),
        bytes_written_(other.bytes_written_),
        ops_read_(other.ops_read_),
        ops_written_(other.ops_written_),
        target_score_(other.target_score_),
        remaining_space_(other.remaining_space_),
        max_capacity_(other.max_capacity_),
        perf_metrics_(other.perf_metrics_),
        expected_ttl_days_(other.expected_ttl_days_),
        persistence_level_(other.persistence_level_),
        bdev_type_(other.bdev_type_) {}

  CTP_CROSS_FUN TargetInfo &operator=(const TargetInfo &other) {
    if (this != &other) {
      target_name_ = other.target_name_;
      bdev_pool_name_ = other.bdev_pool_name_;
      bdev_client_ = other.bdev_client_;
      target_query_ = other.target_query_;
      bytes_read_ = other.bytes_read_;
      bytes_written_ = other.bytes_written_;
      ops_read_ = other.ops_read_;
      ops_written_ = other.ops_written_;
      target_score_ = other.target_score_;
      remaining_space_ = other.remaining_space_;
      max_capacity_ = other.max_capacity_;
      perf_metrics_ = other.perf_metrics_;
      expected_ttl_days_ = other.expected_ttl_days_;
      persistence_level_ = other.persistence_level_;
      bdev_type_ = other.bdev_type_;
    }
    return *this;
  }
};

/**
 * RegisterTarget task - Get/create bdev locally, create Target struct
 */
struct RegisterTargetTask : public clio::run::Task {
  // Task-specific data using CTP macros
  IN clio::run::priv::string
      target_name_;  // Name and file path of the target to register
  IN clio::run::bdev::BdevType bdev_type_;  // Block device type enum
  IN clio::run::u64 total_size_;                 // Total size for allocation
  IN clio::run::PoolQuery target_query_;  // Target pool query for bdev API calls
  IN clio::run::PoolId bdev_id_;          // PoolId to create for the underlying bdev
  // 0 = create a new bdev at bdev_id_ (default, as before); 1 = bind to the
  // ALREADY-EXISTING pool at bdev_id_ without creating it (e.g. a safe-bdev
  // pool). When attaching, the handler validates the pool via GetStats and
  // skips AsyncCreate.
  IN clio::run::u32 attach_existing_;

  // SHM constructor
  CTP_CROSS_FUN RegisterTargetTask()
      : clio::run::Task(),
        target_name_(CLIO_PRIV_ALLOC),
        bdev_type_(clio::run::bdev::BdevType::kFile),
        total_size_(0),
        bdev_id_(clio::run::PoolId::GetNull()),
        attach_existing_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit RegisterTargetTask(
      const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, const std::string &target_name,
      clio::run::bdev::BdevType bdev_type, clio::run::u64 total_size,
      const clio::run::PoolQuery &target_query, const clio::run::PoolId &bdev_id,
      clio::run::u32 attach_existing = 0)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kRegisterTarget),
        target_name_(CLIO_PRIV_ALLOC, target_name),
        bdev_type_(bdev_type),
        total_size_(total_size),
        target_query_(target_query),
        bdev_id_(bdev_id),
        attach_existing_(attach_existing) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kRegisterTarget;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_name_, bdev_type_, total_size_, target_query_, bdev_id_,
       attach_existing_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  /** Fix up priv::string SSO pointer after cudaMemcpy D→H */
  CTP_CROSS_FUN void FixupAfterCopy() {
    target_name_.FixupSsoPointer();
  }

  /**
   * Copy from another RegisterTargetTask
   * Used when creating replicas for remote execution
   */
  void Copy(const ctp::ipc::FullPtr<RegisterTargetTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // Copy RegisterTargetTask-specific fields
    target_name_ = other->target_name_;
    bdev_type_ = other->bdev_type_;
    total_size_ = other->total_size_;
    target_query_ = other->target_query_;
    bdev_id_ = other->bdev_id_;
    attach_existing_ = other->attach_existing_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<RegisterTargetTask>());
  }
};

/**
 * UnregisterTarget task - Unlink bdev from container (don't destroy bdev
 * container)
 */
struct UnregisterTargetTask : public clio::run::Task {
  IN clio::run::priv::string target_name_;  // Name of the target to unregister

  // SHM constructor
  UnregisterTargetTask() : clio::run::Task(), target_name_(CLIO_PRIV_ALLOC) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit UnregisterTargetTask(const clio::run::TaskId &task_id,
                                               const clio::run::PoolId &pool_id,
                                               const clio::run::PoolQuery &pool_query,
                                               const std::string &target_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kUnregisterTarget),
        target_name_(CLIO_PRIV_ALLOC, target_name) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kUnregisterTarget;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // No output parameters (return_code_ handled by base class)
  }

  /**
   * Copy from another UnregisterTargetTask
   */
  void Copy(const ctp::ipc::FullPtr<UnregisterTargetTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    target_name_ = other->target_name_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<UnregisterTargetTask>());
  }
};

/**
 * ListTargets task - Return set of registered target names on this node
 */
struct ListTargetsTask : public clio::run::Task {
  OUT std::vector<std::string>
      target_names_;  // List of registered target names

  // SHM constructor
  ListTargetsTask() : clio::run::Task() {}

  // Emplace constructor
  CTP_CROSS_FUN explicit ListTargetsTask(const clio::run::TaskId &task_id,
                                          const clio::run::PoolId &pool_id,
                                          const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kListTargets) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kListTargets;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No input parameters
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(target_names_);
  }

  /**
   * Copy from another ListTargetsTask
   */
  void Copy(const ctp::ipc::FullPtr<ListTargetsTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    target_names_ = other->target_names_;
  }

  /**
   * AggregateOut entries from another ListTargetsTask
   * Appends all target names from the other task to this one
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<ListTargetsTask>();
    for (const auto &target_name : other->target_names_) {
      target_names_.push_back(target_name);
    }
  }
};

/**
 * StatTargets task - Poll each target in vector, update performance stats
 */
struct StatTargetsTask : public clio::run::Task {
  // SHM constructor
  StatTargetsTask() : clio::run::Task() {}

  // Emplace constructor
  CTP_CROSS_FUN explicit StatTargetsTask(const clio::run::TaskId &task_id,
                                          const clio::run::PoolId &pool_id,
                                          const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kStatTargets) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kStatTargets;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    // No input parameters
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // No output parameters (return_code_ handled by base class)
  }

  /**
   * Copy from another StatTargetsTask
   */
  void Copy(const ctp::ipc::FullPtr<StatTargetsTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    // No task-specific fields to copy
    (void)other;  // Suppress unused parameter warning
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<StatTargetsTask>());
  }
};

/**
 * GetTargetInfo task - Get information about a specific target
 * Returns target score, remaining space, and performance metrics
 */
struct GetTargetInfoTask : public clio::run::Task {
  IN clio::run::priv::string target_name_;  // Name of target to query
  OUT float target_score_;  // Target score (0-1, normalized log bandwidth)
  OUT clio::run::u64 remaining_space_;  // Remaining allocatable space in bytes
  OUT clio::run::u64 bytes_read_;       // Bytes read from target
  OUT clio::run::u64 bytes_written_;    // Bytes written to target
  OUT clio::run::u64 ops_read_;         // Read operations
  OUT clio::run::u64 ops_written_;      // Write operations

  // SHM constructor
  GetTargetInfoTask()
      : clio::run::Task(),
        target_name_(CLIO_PRIV_ALLOC),
        target_score_(0.0f),
        remaining_space_(0),
        bytes_read_(0),
        bytes_written_(0),
        ops_read_(0),
        ops_written_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetTargetInfoTask(const clio::run::TaskId &task_id,
                                            const clio::run::PoolId &pool_id,
                                            const clio::run::PoolQuery &pool_query,
                                            const std::string &target_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetTargetInfo),
        target_name_(CLIO_PRIV_ALLOC, target_name),
        target_score_(0.0f),
        remaining_space_(0),
        bytes_read_(0),
        bytes_written_(0),
        ops_read_(0),
        ops_written_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetTargetInfo;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(target_score_, remaining_space_, bytes_read_, bytes_written_, ops_read_,
       ops_written_);
  }

  /**
   * Copy from another GetTargetInfoTask
   */
  void Copy(const ctp::ipc::FullPtr<GetTargetInfoTask> &other) {
    Task::Copy(other.template Cast<Task>());
    target_name_ = other->target_name_;
    target_score_ = other->target_score_;
    remaining_space_ = other->remaining_space_;
    bytes_read_ = other->bytes_read_;
    bytes_written_ = other->bytes_written_;
    ops_read_ = other->ops_read_;
    ops_written_ = other->ops_written_;
  }

  /**
   * AggregateOut replica results
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    // For target info, just copy (should be same across replicas)
    Copy(other_base.template Cast<GetTargetInfoTask>());
  }
};

/**
 * TagId type definition
 * Uses clio::run::UniqueId with node_id as major and atomic counter as minor
 */
using TagId = clio::run::UniqueId;

}  // namespace clio::cte::core

// Hash specialization for TagId (TagId uses same hash as clio::run::UniqueId)
namespace ctp {
template <>
struct hash<clio::cte::core::TagId> {
  std::size_t operator()(const clio::cte::core::TagId &id) const {
    std::hash<clio::run::u32> hasher;
    return hasher(id.major_) ^ (hasher(id.minor_) << 1);
  }
};
}  // namespace ctp

namespace clio::cte::core {

/**
 * Tag information structure for blob grouping
 */
struct TagInfo {
  clio::run::priv::string tag_name_;  // Canonical (non-alias) name of this tag
  TagId tag_id_;
  clio::run::u64
      total_size_;  // Total size of all blobs in this tag (non-atomic for GPU)
  Timestamp last_modified_;
  Timestamp last_read_;
  // Change time (ctime): bumped whenever this tag's METADATA changes — create,
  // rename, hard-link add/remove, or size change. Distinct from last_modified_
  // (content/mtime) and last_read_ (atime). Tags only; blobs are not tracked.
  Timestamp last_changed_;
  // Additional names bound to this tag's id (tag-level hard links). The
  // canonical name lives in tag_name_; aliases_ holds every extra name. When
  // the canonical tag is deleted, all of these bindings are removed too.
  clio::run::priv::vector<clio::run::priv::string> aliases_;

  CTP_CROSS_FUN TagInfo()
      : tag_name_(CLIO_PRIV_ALLOC),
        tag_id_(TagId::GetNull()),
        total_size_(0),
        last_modified_(0),
        last_read_(0),
        last_changed_(0),
        aliases_(CLIO_PRIV_ALLOC) {}

  CTP_CROSS_FUN TagInfo(const clio::run::priv::string &tag_name, const TagId &tag_id)
      : tag_name_(tag_name),
        tag_id_(tag_id),
        total_size_(0),
        last_modified_(0),
        last_read_(0),
        last_changed_(0),
        aliases_(CLIO_PRIV_ALLOC) {}

#if CTP_IS_HOST
  TagInfo(const std::string &tag_name, const TagId &tag_id)
      : tag_name_(CLIO_PRIV_ALLOC, tag_name),
        tag_id_(tag_id),
        total_size_(0),
        last_modified_(GetCurrentTimeNs()),
        last_read_(GetCurrentTimeNs()),
        last_changed_(GetCurrentTimeNs()),
        aliases_(CLIO_PRIV_ALLOC) {}
#endif

  CTP_CROSS_FUN TagInfo(const TagInfo &other)
      : tag_name_(other.tag_name_),
        tag_id_(other.tag_id_),
        total_size_(other.total_size_),
        last_modified_(other.last_modified_),
        last_read_(other.last_read_),
        last_changed_(other.last_changed_),
        aliases_(other.aliases_) {}

  CTP_CROSS_FUN TagInfo &operator=(const TagInfo &other) {
    if (this != &other) {
      tag_name_ = other.tag_name_;
      tag_id_ = other.tag_id_;
      total_size_ = other.total_size_;
      last_modified_ = other.last_modified_;
      last_read_ = other.last_read_;
      last_changed_ = other.last_changed_;
      aliases_ = other.aliases_;
    }
    return *this;
  }
};

/**
 * Block structure for blob management
 * Each block represents a portion of a blob stored in a target
 */
struct BlobBlock {
  clio::run::bdev::Client bdev_client_;  // Bdev client for this block's target
  clio::run::PoolQuery target_query_;         // Target pool query for bdev API calls
  clio::run::u64 target_offset_;  // Offset within target where this block is stored
  clio::run::u64 size_;           // Logical bytes used in this block
  // Physical bytes allocated on the bdev (>= size_). The bdev rounds every
  // allocation up to a 4 KB slab, so a tiny append otherwise burns a whole slab
  // for a few bytes AND pushes a new block. Tracking the slab capacity lets
  // ExtendBlob grow size_ into a block's spare capacity on the next append
  // instead of allocating a fresh block -- collapsing the O(writes) block list
  // and physical waste that made generic/069 O(N^2)+ENOSPC. Frees and
  // remaining_space_ accounting use capacity_ (the real footprint). The 4-arg
  // ctor defaults capacity_ = size_ (no spare) so legacy call sites are safe.
  clio::run::u64 capacity_;

  CTP_CROSS_FUN BlobBlock() : target_offset_(0), size_(0), capacity_(0) {}

  CTP_CROSS_FUN BlobBlock(const clio::run::bdev::Client &client,
                           const clio::run::PoolQuery &target_query, clio::run::u64 offset,
                           clio::run::u64 size)
      : bdev_client_(client),
        target_query_(target_query),
        target_offset_(offset),
        size_(size),
        capacity_(size) {}

  CTP_CROSS_FUN BlobBlock(const clio::run::bdev::Client &client,
                           const clio::run::PoolQuery &target_query, clio::run::u64 offset,
                           clio::run::u64 size, clio::run::u64 capacity)
      : bdev_client_(client),
        target_query_(target_query),
        target_offset_(offset),
        size_(size),
        capacity_(capacity) {}
};

/**
 * Blob information structure with block-based management
 */
struct BlobInfo {
  clio::run::priv::string blob_name_;
  clio::run::priv::vector<BlobBlock> blocks_;
  float score_;  // 0-1 score for reorganization
  Timestamp last_modified_;
  Timestamp last_read_;
  // Number of data ops (PutBlob/GetBlob) served by this blob since creation.
  // Consumed by the frecency data organizer (issue #738) as the "frequency"
  // half of the frecency score. Transient runtime stat — not persisted to
  // the WAL/metadata log, so it resets on restart (organizers must treat a
  // zero count as "cold or freshly restored", which decays gracefully).
  clio::run::u64 access_count_;
  int compress_lib_;     // Compression library ID used for this blob (0 = no
                         // compression)
  int compress_preset_;  // Compression preset used (1=FAST, 2=BALANCED, 3=BEST)
  clio::run::u64
      trace_key_;  // Unique trace ID for linking to trace logs (0 = not traced)
  clio::run::u64 preallocated_size_;  // Total preallocated capacity in bytes
  ctp::Mutex prealloc_lock_;   // Mutex for preallocation
  // Per-blob async write-serialization token (issue #680 / generic/074).
  // Concurrent PutBlob/Resize/Truncate tasks for the SAME blob hash to the same
  // container and interleave on ONE worker across co_await points, racing on
  // blocks_ (the block layout) and the size read-modify-write — corrupting
  // data (fsx O_DIRECT mismatches). This token serializes them WITHOUT blocking
  // the worker thread: a contender busy-polls via `co_await yield()` (see
  // PutBlobImpl), which is lost-wakeup-proof because it re-checks every worker
  // iteration. A thread-blocking lock (ctp::Mutex / CoRwLock) CANNOT be used
  // here — it would deadlock the single worker the instant the holder suspends
  // at a co_await. 0 == unlocked; otherwise a non-zero per-task owner token.
  clio::run::u64 write_owner_;
  // Maintained mirror of sum(blocks_[i].size_). GetTotalSize() returns this in
  // O(1) instead of an O(blocks) sum; every blocks_ mutation MUST keep it in
  // sync (ExtendBlob updates it incrementally; cold paths call
  // RecomputeTotalSize()). Without it, a file built by millions of tiny
  // O_APPEND writes pays an O(blocks) sum on every put -> O(N^2) (generic/069).
  clio::run::u64 total_size_cache_;

  CTP_CROSS_FUN BlobInfo()
      : blob_name_(CLIO_PRIV_ALLOC),
        blocks_(CLIO_PRIV_ALLOC),
        score_(0.0f),
        last_modified_(0),
        last_read_(0),
        access_count_(0),
        compress_lib_(0),
        compress_preset_(2),
        trace_key_(0),
        preallocated_size_(0),
        write_owner_(0),
        total_size_cache_(0) {
    prealloc_lock_.Init();
  }

  CTP_CROSS_FUN BlobInfo(const clio::run::priv::string &blob_name, float score)
      : blob_name_(blob_name),
        blocks_(CLIO_PRIV_ALLOC),
        score_(score),
        last_modified_(0),
        last_read_(0),
        access_count_(0),
        compress_lib_(0),
        compress_preset_(2),
        trace_key_(0),
        preallocated_size_(0),
        write_owner_(0),
        total_size_cache_(0) {
    prealloc_lock_.Init();
  }

#if CTP_IS_HOST
  BlobInfo(const std::string &blob_name, float score)
      : blob_name_(CLIO_PRIV_ALLOC, blob_name),
        blocks_(CLIO_PRIV_ALLOC),
        score_(score),
        last_modified_(GetCurrentTimeNs()),
        last_read_(GetCurrentTimeNs()),
        access_count_(0),
        compress_lib_(0),
        compress_preset_(2),
        trace_key_(0),
        preallocated_size_(0),
        write_owner_(0),
        total_size_cache_(0) {
    prealloc_lock_.Init();
  }
#endif

  CTP_CROSS_FUN BlobInfo(const BlobInfo &other)
      : blob_name_(other.blob_name_),
        blocks_(other.blocks_),
        score_(other.score_),
        last_modified_(other.last_modified_),
        last_read_(other.last_read_),
        access_count_(other.access_count_),
        compress_lib_(other.compress_lib_),
        compress_preset_(other.compress_preset_),
        trace_key_(other.trace_key_),
        preallocated_size_(other.preallocated_size_),
        write_owner_(0),  // a fresh copy is unlocked; never inherit lock state
        total_size_cache_(other.total_size_cache_) {
    prealloc_lock_.Init();
  }

  CTP_CROSS_FUN BlobInfo &operator=(const BlobInfo &other) {
    if (this != &other) {
      blob_name_ = other.blob_name_;
      blocks_ = other.blocks_;
      score_ = other.score_;
      last_modified_ = other.last_modified_;
      last_read_ = other.last_read_;
      access_count_ = other.access_count_;
      compress_lib_ = other.compress_lib_;
      compress_preset_ = other.compress_preset_;
      trace_key_ = other.trace_key_;
      preallocated_size_ = other.preallocated_size_;
      total_size_cache_ = other.total_size_cache_;
    }
    return *this;
  }

  // Authoritative O(blocks) sum; the source of truth for total_size_cache_.
  CTP_CROSS_FUN clio::run::u64 ComputeTotalSizeSlow() const {
    clio::run::u64 total = 0;
    for (size_t i = 0; i < blocks_.size(); ++i) {
      total += blocks_[i].size_;
    }
    return total;
  }

  // Reset the cache to the authoritative sum. Call after any blocks_ mutation
  // that does not update the cache incrementally (Resize/Truncate/WAL replay/
  // restore) -- these are cold paths where the O(blocks) recompute is fine.
  CTP_CROSS_FUN void RecomputeTotalSize() {
    total_size_cache_ = ComputeTotalSizeSlow();
  }

  // O(1) total size. Returns the maintained cache; a debug/sanitizer build
  // asserts it still equals the authoritative sum so any missed mutation site
  // is caught during testing before it can corrupt a size in release.
  CTP_CROSS_FUN clio::run::u64 GetTotalSize() const {
#if !defined(NDEBUG) && CTP_IS_HOST
    assert(ComputeTotalSizeSlow() == total_size_cache_ &&
           "BlobInfo::total_size_cache_ drifted from sum(blocks_)");
#endif
    return total_size_cache_;
  }

#if CTP_IS_HOST
  /**
   * Try to acquire the per-blob write lock for owner token `tok` (must be
   * non-zero). Returns true if it was free (now held by `tok`) or already held
   * by `tok` (reentrant, same task). There is NO co_await inside, so this whole
   * method is atomic with respect to other coroutines on the same worker
   * (cooperative scheduling only switches at co_await). The atomic_ref makes it
   * additionally safe if the container is ever serviced from another thread.
   * @param tok Non-zero per-task owner token (e.g. the Task pointer).
   * @return true if the caller now holds the lock.
   */
  bool TryLockWrite(clio::run::u64 tok) {
    ctp::ipc::atomic_ref<clio::run::u64> ref(write_owner_);
    clio::run::u64 expected = 0;
    if (ref.compare_exchange_strong(expected, tok)) return true;
    return ref.load() == tok;  // reentrant: already ours
  }

  /**
   * Release the per-blob write lock if (and only if) it is held by `tok`.
   * Synchronous (no co_await), so it is safe to call from a RAII destructor on
   * any coroutine exit path.
   * @param tok The owner token used to acquire.
   */
  void UnlockWrite(clio::run::u64 tok) {
    ctp::ipc::atomic_ref<clio::run::u64> ref(write_owner_);
    clio::run::u64 expected = tok;
    ref.compare_exchange_strong(expected, 0);
  }
#endif  // CTP_IS_HOST
};

#if CTP_IS_HOST
/**
 * RAII release guard for BlobInfo's per-blob write lock. Acquisition is done by
 * the caller (it needs `co_await yield()` to busy-poll, which cannot live in a
 * constructor); this guard only guarantees the lock is released on EVERY scope
 * exit — normal return, early CLIO_CO_RETURN, or a thrown exception — so no
 * error path can leak the lock and wedge the blob. UnlockWrite is a plain CAS,
 * legal in a destructor (no co_await).
 */
struct BlobWriteLockGuard {
  BlobInfo *blob_;
  clio::run::u64 tok_;
  BlobWriteLockGuard(BlobInfo *blob, clio::run::u64 tok)
      : blob_(blob), tok_(tok) {}
  ~BlobWriteLockGuard() {
    if (blob_ != nullptr) blob_->UnlockWrite(tok_);
  }
  BlobWriteLockGuard(const BlobWriteLockGuard &) = delete;
  BlobWriteLockGuard &operator=(const BlobWriteLockGuard &) = delete;
};
#endif  // CTP_IS_HOST

/**
 * Context structure for workflow-aware compression
 * Provides metadata for compression decision-making
 */
struct Context {
  int persistence_target_;     // Specific persistence level to target (-1 = use
                               // min_persistence_level_)
  int min_persistence_level_;  // 0=volatile,
                               //   1=temp-nonvolatile, 2=long-term

  clio::run::u64 preallocate_;  // Preallocate this many bytes for GPU block storage
                          // (0 = disabled)

  // I/O emulation (issue #747). When emulate_ is set, PutBlob/GetBlob skip
  // the data-transfer phase (and all WAL logging) and instead estimate how
  // long the I/O WOULD have taken from the targets' latency-bandwidth model,
  // returned in emulated_time_ns_. Metadata effects (blob creation, block
  // allocation, tag sizes, access stats) still happen, so tier capacities
  // and placement decisions stay realistic — but emulated blob bytes are
  // never written (reads of them return whatever the recycled blocks hold).
  // Built for training RL data-organization policies without paying for the
  // actual I/O.
  bool emulate_;                    // IN: skip real I/O, model the time
  clio::run::u64 emulated_time_ns_;  // OUT: modeled duration of the skipped I/O

#if CTP_ENABLE_COMPRESS
  int dynamic_compress_;  // 0 - skip, 1 - static, 2 - dynamic
  int compress_lib_;      // The compression library to apply (0-10)
  int compress_preset_;   // Compression preset: 1=FAST, 2=BALANCED, 3=BEST
                          // (default=2)
  clio::run::u32 target_psnr_;  // The acceptable PSNR for lossy compression (0 means
                          // infinity)
  int psnr_chance_;       // The chance PSNR will be validated (default 100%)
  bool max_performance_;  // Compression objective (performance vs ratio)

  int consumer_node_;   // The node where consumer will access data (-1 for
                        // unknown)
  int data_type_;       // The type of data (e.g., float, char, int, double)
  bool trace_;          // Enable tracing for this operation
  clio::run::u64 trace_key_;  // Unique trace ID for this Put operation
  int trace_node_;      // Node ID where trace was initiated
                        //
  // Dynamic statistics (populated after compression)
  clio::run::u64 actual_original_size_;    // Original data size in bytes
  clio::run::u64 actual_compressed_size_;  // Actual size after compression in bytes
  double actual_compression_ratio_;  // Actual compression ratio
                                     // (original/compressed)
  double actual_compress_time_ms_;   // Actual compression time in milliseconds
  double actual_psnr_db_;  // Actual PSNR for lossy compression (0 if lossless)
#endif

  CTP_CROSS_FUN Context()
      : persistence_target_(-1),
        min_persistence_level_(0),
        preallocate_(0),
        emulate_(false),
        emulated_time_ns_(0)
#if CTP_ENABLE_COMPRESS
        ,
        dynamic_compress_(0),
        compress_lib_(0),
        compress_preset_(2),
        target_psnr_(0),
        psnr_chance_(100),
        max_performance_(false),
        consumer_node_(-1),
        data_type_(0),
        trace_(false),
        trace_key_(0),
        trace_node_(-1),
        actual_original_size_(0),
        actual_compressed_size_(0),
        actual_compression_ratio_(1.0),
        actual_compress_time_ms_(0.0),
        actual_psnr_db_(0.0)
#endif
  {
  }

  template <class Archive>
  CTP_CROSS_FUN void serialize(Archive &ar) {
    ar.range(persistence_target_, min_persistence_level_, preallocate_,
             emulate_, emulated_time_ns_);
#if CTP_ENABLE_COMPRESS
    ar.range(dynamic_compress_, compress_lib_, compress_preset_, target_psnr_,
             psnr_chance_, max_performance_, consumer_node_, data_type_,
             trace_, trace_key_, trace_node_, actual_original_size_,
             actual_compressed_size_, actual_compression_ratio_,
             actual_compress_time_ms_, actual_psnr_db_);
#endif
  }

  CTP_CROSS_FUN static Context Preallocate(clio::run::u64 size) {
    Context ctx;
    ctx.preallocate_ = size;
    return ctx;
  }

  /** Convenience: a context with I/O emulation enabled (issue #747). */
  CTP_CROSS_FUN static Context Emulate() {
    Context ctx;
    ctx.emulate_ = true;
    return ctx;
  }
};

/**
 * CTE Operation types for telemetry
 */
enum class CteOp : clio::run::u32 {
  kPutBlob = 0,
  kGetBlob = 1,
  kDelBlob = 2,
  kGetOrCreateTag = 3,
  kDelTag = 4,
  kGetTagSize = 5
};

/**
 * PutBlobTask::flags_ bits.
 *
 * By default PutBlob is a partial modify: it writes [offset, offset+size),
 * growing the blob if needed, and NEVER shrinks/clears it (POSIX write
 * semantics). kCtePutReplace opts into wholesale replacement: the blob is
 * resized to exactly offset+size (shrinking if needed) before the write, so a
 * smaller put truly replaces a larger blob. Both the replace path and the
 * explicit TruncateBlob op share Runtime::ResizeBlob.
 */
GLOBAL_CROSS_CONST clio::run::u32 kCtePutReplace = 0x1u;

/**
 * CTE Telemetry data structure for performance monitoring
 */
struct CteTelemetry {
  CteOp op_;                    // Operation type
  size_t off_;                  // Offset within blob (for Put/Get operations)
  size_t size_;                 // Size of operation (for Put/Get operations)
  TagId tag_id_;                // Tag ID involved
  Timestamp mod_time_;          // Last modification time
  Timestamp read_time_;         // Last read time
  std::uint64_t logical_time_;  // Logical time for ordering telemetry entries

  CteTelemetry()
      : op_(CteOp::kPutBlob),
        off_(0),
        size_(0),
        tag_id_(TagId::GetNull()),
        mod_time_(0),
        read_time_(0),
        logical_time_(0) {}

#if CTP_IS_HOST
  CteTelemetry(CteOp op, size_t off, size_t size, const TagId &tag_id,
               const Timestamp &mod_time, const Timestamp &read_time,
               std::uint64_t logical_time = 0)
      : op_(op),
        off_(off),
        size_(size),
        tag_id_(tag_id),
        mod_time_(mod_time),
        read_time_(read_time),
        logical_time_(logical_time) {}

  // Serialization support for cereal
  template <class Archive>
  void serialize(Archive &ar) {
    ar(op_, off_, size_, tag_id_, mod_time_, read_time_, logical_time_);
  }
#endif
};

/**
 * GetOrCreateTag task - Get or create a tag for blob grouping
 * Template parameter allows different CreateParams types
 */
template <typename CreateParamsT = CreateParams>
struct GetOrCreateTagTask : public clio::run::Task {
  IN clio::run::priv::string tag_name_;  // Tag name (required)
  INOUT TagId tag_id_;  // Tag unique ID (default null, output on creation)

  // SHM constructor
  CTP_CROSS_FUN GetOrCreateTagTask()
      : clio::run::Task(), tag_name_(CLIO_PRIV_ALLOC), tag_id_(TagId::GetNull()) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetOrCreateTagTask(
      const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, const std::string &tag_name,
      const TagId &tag_id = TagId::GetNull())
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetOrCreateTag),
        tag_name_(CLIO_PRIV_ALLOC, tag_name),
        tag_id_(tag_id) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetOrCreateTag;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_name_, tag_id_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tag_id_);
  }

  /** Fix up priv::string SSO pointer after cudaMemcpy for CPU→GPU */
  CTP_CROSS_FUN void FixupAfterCopy() {
    tag_name_.FixupSsoPointer();
  }

  /**
   * Copy from another GetOrCreateTagTask
   */
  void Copy(const ctp::ipc::FullPtr<GetOrCreateTagTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_name_ = other->tag_name_;
    tag_id_ = other->tag_id_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<GetOrCreateTagTask>());
  }
};

/**
 * PutBlob task - Store a blob with optional compression context
 */
struct PutBlobTask : public clio::run::Task {
  IN TagId tag_id_;                    // Tag ID for blob grouping
  INOUT clio::run::priv::string blob_name_;  // Blob name (required)
  IN clio::run::u64 offset_;                 // Offset within blob
  IN clio::run::u64 size_;                   // Size of blob data
  IN ctp::ipc::ShmPtr<> blob_data_;        // Blob data (shared memory pointer)
  IN float score_;         // Score for placement: -1.0=unknown (use defaults),
                           // 0.0-1.0=explicit
  INOUT Context context_;  // Context for compression control and statistics
  IN clio::run::u32 flags_;      // Operation flags
  /**
   * Optional page index appended to blob_name_ at handler time as
   * "_pi<gpu_page_idx_>" so each cache page in a gpu_vector::Vector
   * resolves to its own blob. Sentinel kNoPageIdx (~0u) disables the
   * suffix (default for non-GPU clients). Mutated from device kernels
   * (FlushPage / FaultPage) which can't safely rebuild a clio::run::priv::
   * string at flush time, so the suffix is composed runtime-side.
   */
  static constexpr clio::run::u32 kNoPageIdx = ~static_cast<clio::run::u32>(0);
  IN clio::run::u32 gpu_page_idx_;

  // Submit-side timestamp (steady_clock nanoseconds), stamped by the
  // CTE client just before Send so the receiving daemon can compute
  // end-to-end submit→recv latency. 0 means "not stamped" and the
  // receiver-side accumulator ignores those tasks. Cross-node clocks
  // need ntp synchronization; on the same cluster that's usually
  // within a few hundred μs which is fine for ms-resolution diagnostics.
  IN clio::run::u64 submit_ts_ns_;

  // SHM constructor
  // Default score -1.0f means "unknown" - runtime will use 1.0 for new blobs
  // or preserve existing score for modifications
  CTP_CROSS_FUN PutBlobTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(CLIO_PRIV_ALLOC),
        offset_(0),
        size_(0),
        blob_data_(ctp::ipc::ShmPtr<>::GetNull()),
        score_(-1.0f),
        context_(),
        flags_(0),
        gpu_page_idx_(kNoPageIdx),
        submit_ts_ns_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit PutBlobTask(const clio::run::TaskId &task_id,
                                      const clio::run::PoolId &pool_id,
                                      const clio::run::PoolQuery &pool_query,
                                      const TagId &tag_id,
                                      const std::string &blob_name,
                                      clio::run::u64 offset, clio::run::u64 size,
                                      ctp::ipc::ShmPtr<> blob_data, float score,
                                      const Context &context, clio::run::u32 flags)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPutBlob),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        offset_(offset),
        size_(size),
        blob_data_(blob_data),
        score_(score),
        context_(context),
        flags_(flags),
        gpu_page_idx_(kNoPageIdx),
        submit_ts_ns_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kPutBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  // GPU-compatible emplace constructor (const char* instead of std::string)
  CTP_CROSS_FUN explicit PutBlobTask(const clio::run::TaskId &task_id,
                                      const clio::run::PoolId &pool_id,
                                      const clio::run::PoolQuery &pool_query,
                                      const TagId &tag_id,
                                      const char *blob_name, clio::run::u64 offset,
                                      clio::run::u64 size, ctp::ipc::ShmPtr<> blob_data,
                                      float score, const Context &context,
                                      clio::run::u32 flags)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPutBlob),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        offset_(offset),
        size_(size),
        blob_data_(blob_data),
        score_(score),
        context_(context),
        flags_(flags),
        gpu_page_idx_(kNoPageIdx),
        submit_ts_ns_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kPutBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Destructor — frees blob_data_ when this task owns the buffer.
   *
   * The destructor runs for every PutBlobTask, but TASK_DATA_OWNER is
   * only set on the receiver side when LoadTaskArchive::bulk had to copy
   * a ZMQ-owned BULK_XFER input into a fresh CHI buffer (zmq zero-copy
   * recv on the client TCP/IPC path — IpcCpu2CpuZmq::RuntimeRecv, or the
   * cross-node admin RecvIn path). The original libzmq buffer is freed
   * separately via ClearRecvHandles right after AllocLoadTask; this frees
   * the owned copy when the task is destroyed.
   *
   * Client-side PutBlobTask (created by an emplace constructor) calls
   * task_flags_.Clear() so the flag is off and the destructor leaves the
   * client's blob_data_ alone (the client allocated it and frees it
   * itself after task.Wait()). The SHM zero-copy recv path also leaves
   * the flag clear (no copy was made). Without this destructor every
   * inbound ZMQ BULK_XFER PutBlob leaked one io_size allocation.
   */
  CTP_CROSS_FUN ~PutBlobTask() {
#if !CTP_IS_DEVICE_PASS
    if (task_flags_.Any(TASK_DATA_OWNER) && !blob_data_.IsNull()) {
      auto *ipc_manager = CLIO_CPU_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(blob_data_.template Cast<char>());
      }
    }
#endif
  }

  /**
   * Serialize IN and INOUT parameters.
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    ar.PushPod(blob_name_.UsingSso());
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, offset_, size_, blob_data_,
       score_, context_, flags_, gpu_page_idx_, submit_ts_ns_);
    ar.PopPod();
    // Emulated puts (issue #747) never write the payload, so don't ship it
    // over the wire either — the whole point is to not pay for the I/O.
    // context_ is (de)serialized above, so both the save and load sides
    // agree on whether the bulk follows.
    if (!context_.emulate_) {
      ar.bulk(blob_data_, size_, BULK_XFER);
    }
  }

  /**
   * Serialize OUT and INOUT parameters.
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    ar.PushPod(blob_name_.UsingSso());
    Task::SerializeOut(ar);
    // Only INOUT fields travel back on the OUT path. The IN-only fields —
    // crucially blob_data_ — must NOT be serialized here: blob_data_ is the
    // CLIENT's SHM buffer pointer, but the receiver runs with its own
    // (daemon-allocated) buffer. Echoing the receiver's blob_data_ back would
    // clobber the origin task's blob_data_ during AggregateOut, so a later
    // FreeBuffer(task->blob_data_) frees a foreign/already-freed buffer and
    // corrupts the allocator — observed as a hang in CAE assimilators under
    // CLIO_FORCE_NET (issue #500).
    ar(blob_name_, context_);
    ar.PopPod();
  }

  /** Fix up priv::string SSO pointer after cudaMemcpy D→H */
  CTP_CROSS_FUN void FixupAfterCopy() {
    blob_name_.FixupSsoPointer();
  }

  /**
   * Copy from another PutBlobTask
   */
  void Copy(const ctp::ipc::FullPtr<PutBlobTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    blob_data_ = other->blob_data_;
    score_ = other->score_;
    context_ = other->context_;
    flags_ = other->flags_;
    gpu_page_idx_ = other->gpu_page_idx_;
    submit_ts_ns_ = other->submit_ts_ns_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<PutBlobTask>());
  }
};

/**
 * GetBlob task - Retrieve a blob (unimplemented for now)
 */
struct GetBlobTask : public clio::run::Task {
  IN TagId tag_id_;                 // Tag ID for blob lookup
  IN clio::run::priv::string blob_name_;  // Blob name (required)
  IN clio::run::u64 offset_;              // Offset within blob
  IN clio::run::u64 size_;                // Size of data to retrieve
  IN clio::run::u32 flags_;               // Operation flags
  IN ctp::ipc::ShmPtr<>
      blob_data_;  // Input buffer for blob data (shared memory pointer)
  /**
   * Optional page index appended to blob_name_ at handler time as
   * "_pi<gpu_page_idx_>". See PutBlobTask::gpu_page_idx_ for rationale.
   */
  static constexpr clio::run::u32 kNoPageIdx = ~static_cast<clio::run::u32>(0);
  IN clio::run::u32 gpu_page_idx_;
  INOUT Context context_;  // Emulation control + modeled time (issue #747)

  // SHM constructor
  CTP_CROSS_FUN GetBlobTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(CLIO_PRIV_ALLOC),
        offset_(0),
        size_(0),
        flags_(0),
        blob_data_(ctp::ipc::ShmPtr<>::GetNull()),
        gpu_page_idx_(kNoPageIdx),
        context_() {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetBlobTask(const clio::run::TaskId &task_id,
                                      const clio::run::PoolId &pool_id,
                                      const clio::run::PoolQuery &pool_query,
                                      const TagId &tag_id,
                                      const std::string &blob_name,
                                      clio::run::u64 offset, clio::run::u64 size,
                                      clio::run::u32 flags, ctp::ipc::ShmPtr<> blob_data,
                                      const Context &context = Context())
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetBlob),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        offset_(offset),
        size_(size),
        flags_(flags),
        blob_data_(blob_data),
        gpu_page_idx_(kNoPageIdx),
        context_(context) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /** Destructor — frees blob_data_ when this task owns the buffer.
   *
   * The destructor runs for every GetBlobTask, but TASK_DATA_OWNER is only
   * set on cross-node-receiver instances (admin RecvIn sets it when
   * LoadTaskArchive::bulk had to AllocateBuffer for the BULK_EXPOSE
   * input — i.e. on the remote daemon that allocated a fresh buffer to
   * receive the read response from its bdev).
   *
   * Client-side GetBlobTask (created by emplace constructor) has
   * task_flags_.Clear() so the flag is off and the destructor leaves the
   * client's blob_data_ alone (the client allocated it and will free it
   * itself after task.Wait() returns).
   *
   * Without this destructor the BULK_EXPOSE buffer on the responder side
   * leaked one 1 MiB allocation per cross-node read; at 24 PPN × 256 MiB
   * blocks the 1 GiB main shared-memory segment ran out, AllocateBuffer
   * started failing, RecvIn's deserialization silently dropped the
   * response, and the client's task.Wait() spun forever.
   */
  CTP_CROSS_FUN ~GetBlobTask() {
#if !CTP_IS_DEVICE_PASS
    if (task_flags_.Any(TASK_DATA_OWNER) && !blob_data_.IsNull()) {
      auto *ipc_manager = CLIO_CPU_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(blob_data_.template Cast<char>());
      }
    }
#endif
  }

  // GPU-compatible emplace constructor (const char* instead of std::string)
  CTP_CROSS_FUN explicit GetBlobTask(const clio::run::TaskId &task_id,
                                      const clio::run::PoolId &pool_id,
                                      const clio::run::PoolQuery &pool_query,
                                      const TagId &tag_id,
                                      const char *blob_name, clio::run::u64 offset,
                                      clio::run::u64 size, clio::run::u32 flags,
                                      ctp::ipc::ShmPtr<> blob_data,
                                      const Context &context = Context())
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetBlob),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        offset_(offset),
        size_(size),
        flags_(flags),
        blob_data_(blob_data),
        gpu_page_idx_(kNoPageIdx),
        context_(context) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters.
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    ar.PushPod(blob_name_.UsingSso());
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, offset_, size_, flags_, blob_data_,
       gpu_page_idx_, context_);
    ar.PopPod();
    // Emulated gets (issue #747) return no data, so the caller's buffer
    // does not need to be exposed for the response.
    if (!context_.emulate_) {
      ar.bulk(blob_data_, size_, BULK_EXPOSE);
    }
  }

  /**
   * Serialize OUT and INOUT parameters.
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // context_ is INOUT: carries emulated_time_ns_ back (issue #747). An
    // emulated get read nothing, so no data bulk follows — the caller's
    // buffer is left untouched and the wire cost is metadata-only.
    ar(context_);
    if (!context_.emulate_) {
      ar.bulk(blob_data_, size_, BULK_XFER);
    }
  }

  /** Fix up priv::string SSO pointer after cudaMemcpy D→H */
  CTP_CROSS_FUN void FixupAfterCopy() {
    blob_name_.FixupSsoPointer();
  }

  /**
   * Copy from another GetBlobTask
   */
  void Copy(const ctp::ipc::FullPtr<GetBlobTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    flags_ = other->flags_;
    blob_data_ = other->blob_data_;
    gpu_page_idx_ = other->gpu_page_idx_;
    context_ = other->context_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<GetBlobTask>());
  }
};

/**
 * ReorganizeBlob task - Change score for a single blob
 */
struct ReorganizeBlobTask : public clio::run::Task {
  IN TagId tag_id_;                 // Tag ID containing blob
  IN clio::run::priv::string blob_name_;  // Blob name to reorganize
  IN float new_score_;              // New score for the blob (0-1)

  // SHM constructor
  ReorganizeBlobTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(CLIO_PRIV_ALLOC),
        new_score_(0.0f) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit ReorganizeBlobTask(const clio::run::TaskId &task_id,
                                             const clio::run::PoolId &pool_id,
                                             const clio::run::PoolQuery &pool_query,
                                             const TagId &tag_id,
                                             const std::string &blob_name,
                                             float new_score)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kReorganizeBlob),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        new_score_(new_score) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kReorganizeBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, new_score_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // No output parameters (return_code_ handled by base class)
  }

  /**
   * Copy from another ReorganizeBlobTask
   */
  void Copy(const ctp::ipc::FullPtr<ReorganizeBlobTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    new_score_ = other->new_score_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<ReorganizeBlobTask>());
  }
};

// ===========================================================================
// Fully-POD, GPU-compatible blob tasks (issue #556).
//
// These mirror PutBlob/GetBlob/ReorganizeBlob but carry the blob name in an
// inline clio::run::priv::fixed_string<32> (no allocator, no SSO) instead of a
// priv::string. Every field is therefore POD and the whole task is
// bitwise-relocatable: a cudaMemcpy D<->H of the task is correct with ZERO
// FixupAfterCopy (none is defined). Blob names are capped at 31 chars + NUL.
// They share the runtime handler logic with the non-POD tasks via the
// Runtime::*Impl<TaskT> member templates (same field names; both name types
// expose .str()).
// ===========================================================================

/** Inline POD blob name used by all Pod*Blob tasks. */
using PodBlobName = clio::run::priv::fixed_string<32>;

/**
 * PodPutBlob - fully-POD, GPU-compatible variant of PutBlobTask.
 */
struct PodPutBlobTask : public clio::run::Task {
  IN TagId tag_id_;
  INOUT PodBlobName blob_name_;
  IN clio::run::u64 offset_;
  IN clio::run::u64 size_;
  IN ctp::ipc::ShmPtr<> blob_data_;
  IN float score_;
  INOUT Context context_;
  IN clio::run::u32 flags_;
  static constexpr clio::run::u32 kNoPageIdx = ~static_cast<clio::run::u32>(0);
  IN clio::run::u32 gpu_page_idx_;
  IN clio::run::u64 submit_ts_ns_;

  // SHM constructor
  CTP_CROSS_FUN PodPutBlobTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(),
        offset_(0),
        size_(0),
        blob_data_(ctp::ipc::ShmPtr<>::GetNull()),
        score_(-1.0f),
        context_(),
        flags_(0),
        gpu_page_idx_(kNoPageIdx),
        submit_ts_ns_(0) {}

  // GPU-compatible emplace constructor (const char* blob name, no allocator)
  CTP_CROSS_FUN explicit PodPutBlobTask(
      const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, const TagId &tag_id,
      const char *blob_name, clio::run::u64 offset, clio::run::u64 size,
      ctp::ipc::ShmPtr<> blob_data, float score, const Context &context,
      clio::run::u32 flags)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPodPutBlob),
        tag_id_(tag_id),
        blob_name_(blob_name),
        offset_(offset),
        size_(size),
        blob_data_(blob_data),
        score_(score),
        context_(context),
        flags_(flags),
        gpu_page_idx_(kNoPageIdx),
        submit_ts_ns_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kPodPutBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

#if CTP_IS_HOST
  // std::string convenience overload (host clients).
  explicit PodPutBlobTask(const clio::run::TaskId &task_id,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const TagId &tag_id, const std::string &blob_name,
                          clio::run::u64 offset, clio::run::u64 size,
                          ctp::ipc::ShmPtr<> blob_data, float score,
                          const Context &context, clio::run::u32 flags)
      : PodPutBlobTask(task_id, pool_id, pool_query, tag_id, blob_name.c_str(),
                       offset, size, blob_data, score, context, flags) {}
#endif

  /** Destructor — frees blob_data_ when this task owns the buffer (see
   *  PutBlobTask::~PutBlobTask for the TASK_DATA_OWNER rationale). */
  CTP_CROSS_FUN ~PodPutBlobTask() {
#if !CTP_IS_DEVICE_PASS
    if (task_flags_.Any(TASK_DATA_OWNER) && !blob_data_.IsNull()) {
      auto *ipc_manager = CLIO_CPU_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(blob_data_.template Cast<char>());
      }
    }
#endif
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, offset_, size_, blob_data_, score_, context_,
       flags_, gpu_page_idx_, submit_ts_ns_);
    // Emulated puts never write the payload — skip the wire transfer too
    // (issue #747; see PutBlobTask::SerializeIn).
    if (!context_.emulate_) {
      ar.bulk(blob_data_, size_, BULK_XFER);
    }
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // Only INOUT fields travel back (see PutBlobTask::SerializeOut for why
    // blob_data_ must NOT be echoed).
    ar(blob_name_, context_);
  }

  // No FixupAfterCopy — fixed_string is position-independent.

  void Copy(const ctp::ipc::FullPtr<PodPutBlobTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    blob_data_ = other->blob_data_;
    score_ = other->score_;
    context_ = other->context_;
    flags_ = other->flags_;
    gpu_page_idx_ = other->gpu_page_idx_;
    submit_ts_ns_ = other->submit_ts_ns_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<PodPutBlobTask>());
  }
};

/**
 * PodGetBlob - fully-POD, GPU-compatible variant of GetBlobTask.
 */
struct PodGetBlobTask : public clio::run::Task {
  IN TagId tag_id_;
  IN PodBlobName blob_name_;
  IN clio::run::u64 offset_;
  IN clio::run::u64 size_;
  IN clio::run::u32 flags_;
  IN ctp::ipc::ShmPtr<> blob_data_;
  static constexpr clio::run::u32 kNoPageIdx = ~static_cast<clio::run::u32>(0);
  IN clio::run::u32 gpu_page_idx_;
  INOUT Context context_;  // Emulation control + modeled time (issue #747)

  // SHM constructor
  CTP_CROSS_FUN PodGetBlobTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(),
        offset_(0),
        size_(0),
        flags_(0),
        blob_data_(ctp::ipc::ShmPtr<>::GetNull()),
        gpu_page_idx_(kNoPageIdx),
        context_() {}

  // GPU-compatible emplace constructor (const char* blob name, no allocator)
  CTP_CROSS_FUN explicit PodGetBlobTask(
      const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, const TagId &tag_id,
      const char *blob_name, clio::run::u64 offset, clio::run::u64 size,
      clio::run::u32 flags, ctp::ipc::ShmPtr<> blob_data,
      const Context &context = Context())
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPodGetBlob),
        tag_id_(tag_id),
        blob_name_(blob_name),
        offset_(offset),
        size_(size),
        flags_(flags),
        blob_data_(blob_data),
        gpu_page_idx_(kNoPageIdx),
        context_(context) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kPodGetBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

#if CTP_IS_HOST
  // std::string convenience overload (host clients).
  explicit PodGetBlobTask(const clio::run::TaskId &task_id,
                          const clio::run::PoolId &pool_id,
                          const clio::run::PoolQuery &pool_query,
                          const TagId &tag_id, const std::string &blob_name,
                          clio::run::u64 offset, clio::run::u64 size,
                          clio::run::u32 flags, ctp::ipc::ShmPtr<> blob_data,
                          const Context &context = Context())
      : PodGetBlobTask(task_id, pool_id, pool_query, tag_id, blob_name.c_str(),
                       offset, size, flags, blob_data, context) {}
#endif

  /** Destructor — frees blob_data_ when this task owns the buffer (see
   *  GetBlobTask::~GetBlobTask for the TASK_DATA_OWNER rationale). */
  CTP_CROSS_FUN ~PodGetBlobTask() {
#if !CTP_IS_DEVICE_PASS
    if (task_flags_.Any(TASK_DATA_OWNER) && !blob_data_.IsNull()) {
      auto *ipc_manager = CLIO_CPU_IPC;
      if (ipc_manager) {
        ipc_manager->FreeBuffer(blob_data_.template Cast<char>());
      }
    }
#endif
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, offset_, size_, flags_, blob_data_, gpu_page_idx_,
       context_);
    // Emulated gets return no data — no buffer expose needed (issue #747).
    if (!context_.emulate_) {
      ar.bulk(blob_data_, size_, BULK_EXPOSE);
    }
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // context_ is INOUT: carries emulated_time_ns_ back (issue #747). No
    // data bulk on the emulated path — the caller's buffer stays untouched.
    ar(context_);
    if (!context_.emulate_) {
      ar.bulk(blob_data_, size_, BULK_XFER);
    }
  }

  // No FixupAfterCopy — fixed_string is position-independent.

  void Copy(const ctp::ipc::FullPtr<PodGetBlobTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    offset_ = other->offset_;
    size_ = other->size_;
    flags_ = other->flags_;
    blob_data_ = other->blob_data_;
    gpu_page_idx_ = other->gpu_page_idx_;
    context_ = other->context_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<PodGetBlobTask>());
  }
};

/**
 * PodReorganizeBlob - fully-POD, GPU-compatible variant of ReorganizeBlobTask.
 */
struct PodReorganizeBlobTask : public clio::run::Task {
  IN TagId tag_id_;
  IN PodBlobName blob_name_;
  IN float new_score_;

  // SHM constructor
  CTP_CROSS_FUN PodReorganizeBlobTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(),
        new_score_(0.0f) {}

  // GPU-compatible emplace constructor (const char* blob name, no allocator)
  CTP_CROSS_FUN explicit PodReorganizeBlobTask(
      const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, const TagId &tag_id,
      const char *blob_name, float new_score)
      : clio::run::Task(task_id, pool_id, pool_query,
                        Method::kPodReorganizeBlob),
        tag_id_(tag_id),
        blob_name_(blob_name),
        new_score_(new_score) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kPodReorganizeBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

#if CTP_IS_HOST
  // std::string convenience overload (host clients).
  explicit PodReorganizeBlobTask(const clio::run::TaskId &task_id,
                                 const clio::run::PoolId &pool_id,
                                 const clio::run::PoolQuery &pool_query,
                                 const TagId &tag_id,
                                 const std::string &blob_name, float new_score)
      : PodReorganizeBlobTask(task_id, pool_id, pool_query, tag_id,
                              blob_name.c_str(), new_score) {}
#endif

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, new_score_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  void Copy(const ctp::ipc::FullPtr<PodReorganizeBlobTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    new_score_ = other->new_score_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<PodReorganizeBlobTask>());
  }
};

/**
 * DelBlob task - Remove blob and decrement tag size
 */
struct DelBlobTask : public clio::run::Task {
  IN TagId tag_id_;                 // Tag ID for blob lookup
  IN clio::run::priv::string blob_name_;  // Blob name (required)

  // SHM constructor
  DelBlobTask()
      : clio::run::Task(), tag_id_(TagId::GetNull()), blob_name_(CLIO_PRIV_ALLOC) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit DelBlobTask(const clio::run::TaskId &task_id,
                                      const clio::run::PoolId &pool_id,
                                      const clio::run::PoolQuery &pool_query,
                                      const TagId &tag_id,
                                      const std::string &blob_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDelBlob),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kDelBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // No output parameters (return_code_ handled by base class)
  }

  /**
   * Copy from another DelBlobTask
   */
  void Copy(const ctp::ipc::FullPtr<DelBlobTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<DelBlobTask>());
  }
};

/**
 * TruncateBlob task - resize a blob to an exact logical size.
 * Grows (zero-extends) or shrinks (frees trailing blocks). Shares
 * Runtime::ResizeBlob with PutBlob's replace path.
 */
struct TruncateBlobTask : public clio::run::Task {
  IN TagId tag_id_;                 // Tag ID for blob lookup
  IN clio::run::priv::string blob_name_;  // Blob name (required)
  IN clio::run::u64 new_size_;            // Target logical size in bytes

  // SHM constructor
  TruncateBlobTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(CLIO_PRIV_ALLOC),
        new_size_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit TruncateBlobTask(const clio::run::TaskId &task_id,
                                          const clio::run::PoolId &pool_id,
                                          const clio::run::PoolQuery &pool_query,
                                          const TagId &tag_id,
                                          const std::string &blob_name,
                                          clio::run::u64 new_size)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kTruncateBlob),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        new_size_(new_size) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kTruncateBlob;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_, new_size_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
  }

  void Copy(const ctp::ipc::FullPtr<TruncateBlobTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    new_size_ = other->new_size_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<TruncateBlobTask>());
  }
};

/**
 * DelTag task - Remove all blobs from tag and remove tag
 * Supports lookup by either tag ID or tag name
 */
struct DelTagTask : public clio::run::Task {
  INOUT TagId tag_id_;             // Tag ID to delete (input or lookup result)
  IN clio::run::priv::string tag_name_;  // Tag name for lookup (optional)
  // POSIX-unlink semantics (#680): when 1, deleting a tag's canonical NAME while
  // hard-link aliases remain promotes a surviving alias and keeps the tag/blobs
  // alive (the file survives until its last link is removed). When 0 (default),
  // deleting the canonical name/id cascade-deletes the tag and all its aliases.
  // The FUSE/filesystem unlink+rename set this; direct core "delete tag" does not.
  IN clio::run::u32 posix_unlink_ = 0;

  // SHM constructor
  DelTagTask()
      : clio::run::Task(), tag_id_(TagId::GetNull()), tag_name_(CLIO_PRIV_ALLOC) {}

  // Emplace constructor with tag ID
  CTP_CROSS_FUN explicit DelTagTask(const clio::run::TaskId &task_id,
                                     const clio::run::PoolId &pool_id,
                                     const clio::run::PoolQuery &pool_query,
                                     const TagId &tag_id)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDelTag),
        tag_id_(tag_id),
        tag_name_(CLIO_PRIV_ALLOC) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kDelTag;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  // Emplace constructor with tag name
  CTP_CROSS_FUN explicit DelTagTask(const clio::run::TaskId &task_id,
                                     const clio::run::PoolId &pool_id,
                                     const clio::run::PoolQuery &pool_query,
                                     const std::string &tag_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kDelTag),
        tag_id_(TagId::GetNull()),
        tag_name_(CLIO_PRIV_ALLOC, tag_name) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kDelTag;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, tag_name_, posix_unlink_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tag_id_);
  }

  /**
   * Copy from another DelTagTask
   */
  void Copy(const ctp::ipc::FullPtr<DelTagTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    tag_name_ = other->tag_name_;
    posix_unlink_ = other->posix_unlink_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<DelTagTask>());
  }
};

/**
 * RenameTag task - change a tag's name, keeping its TagId (and therefore all
 * of its blobs) intact. Broadcast: each container moves the name->id binding
 * it holds (old_name_ -> new_name_) and updates the tag's stored name.
 */
struct RenameTagTask : public clio::run::Task {
  INOUT TagId tag_id_;                 // Tag ID (input, or resolved from name)
  IN clio::run::priv::string old_name_;      // Current tag name
  IN clio::run::priv::string new_name_;      // Desired tag name

  RenameTagTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        old_name_(CLIO_PRIV_ALLOC),
        new_name_(CLIO_PRIV_ALLOC) {}

  CTP_CROSS_FUN explicit RenameTagTask(const clio::run::TaskId &task_id,
                                       const clio::run::PoolId &pool_id,
                                       const clio::run::PoolQuery &pool_query,
                                       const TagId &tag_id,
                                       const std::string &old_name,
                                       const std::string &new_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kRenameTag),
        tag_id_(tag_id),
        old_name_(CLIO_PRIV_ALLOC, old_name),
        new_name_(CLIO_PRIV_ALLOC, new_name) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kRenameTag;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, old_name_, new_name_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tag_id_);
  }

  void Copy(const ctp::ipc::FullPtr<RenameTagTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    old_name_ = other->old_name_;
    new_name_ = other->new_name_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<RenameTagTask>());
  }
};

/**
 * GetOrCreateTagAlias task - bind an additional name (alias_name_) to an
 * EXISTING tag's TagId, so the alias shares the target's id and therefore all
 * of its blobs (a hard link at the tag level). The target may be given by id
 * (tag_id_) or by name (existing_name_); if it does not exist, found_ stays 0
 * and the caller treats it as an error. Idempotent: if alias_name_ is already
 * bound, its existing id is returned. Broadcast.
 */
struct GetOrCreateTagAliasTask : public clio::run::Task {
  INOUT TagId tag_id_;              // Target tag id (input, or resolved by name)
  IN clio::run::priv::string existing_name_;  // Target tag name (used if tag_id_ null)
  IN clio::run::priv::string alias_name_;     // New name to bind to the target id
  OUT clio::run::u32 found_;              // 1 if the target tag exists (success)

  GetOrCreateTagAliasTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        existing_name_(CLIO_PRIV_ALLOC),
        alias_name_(CLIO_PRIV_ALLOC),
        found_(0) {}

  CTP_CROSS_FUN explicit GetOrCreateTagAliasTask(
      const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, const TagId &tag_id,
      const std::string &existing_name, const std::string &alias_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetOrCreateTagAlias),
        tag_id_(tag_id),
        existing_name_(CLIO_PRIV_ALLOC, existing_name),
        alias_name_(CLIO_PRIV_ALLOC, alias_name),
        found_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetOrCreateTagAlias;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, existing_name_, alias_name_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tag_id_, found_);
  }

  void Copy(const ctp::ipc::FullPtr<GetOrCreateTagAliasTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    existing_name_ = other->existing_name_;
    alias_name_ = other->alias_name_;
    found_ = other->found_;
  }

  // Broadcast aggregation: the alias exists if ANY container confirmed the
  // target, and the resolved id is whichever non-null id a replica reported.
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<GetOrCreateTagAliasTask>();
    if (other->found_) {
      found_ = 1;
    }
    if (tag_id_.IsNull() && !other->tag_id_.IsNull()) {
      tag_id_ = other->tag_id_;
    }
  }
};

/**
 * GetTagName task - resolve a TagId to its full (absolute) tag name.
 *
 * Tag names are stored RELATIVELY: a hierarchical child holds
 * "$tagid{major.minor}/leaf" referencing its parent's id, so moving a
 * directory tag is O(1). This task walks those parent references on the
 * owning container and returns the fully-resolved name (e.g. "/a/b/c").
 * Flat (non-path) tags resolve to themselves. Broadcast; the container that
 * owns the tag's metadata produces the answer.
 */
struct GetTagNameTask : public clio::run::Task {
  IN TagId tag_id_;                 // Tag ID to resolve
  OUT clio::run::priv::string tag_name_;  // Resolved full name (empty if not found)
  OUT clio::run::u32 found_;              // 1 if the tag's metadata was located

  GetTagNameTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        tag_name_(CLIO_PRIV_ALLOC),
        found_(0) {}

  CTP_CROSS_FUN explicit GetTagNameTask(const clio::run::TaskId &task_id,
                                        const clio::run::PoolId &pool_id,
                                        const clio::run::PoolQuery &pool_query,
                                        const TagId &tag_id)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetTagName),
        tag_id_(tag_id),
        tag_name_(CLIO_PRIV_ALLOC),
        found_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetTagName;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tag_name_, found_);
  }

  void Copy(const ctp::ipc::FullPtr<GetTagNameTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    tag_name_ = other->tag_name_;
    found_ = other->found_;
  }

  // Broadcast aggregation: keep the answer from whichever container owns the
  // tag (the one that set found_ and a non-empty resolved name).
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<GetTagNameTask>();
    if (other->found_ && !found_) {
      found_ = 1;
      tag_name_ = other->tag_name_;
    }
  }
};

/**
 * GetTagSize task - Get the total size of a tag
 */
struct GetTagSizeTask : public clio::run::Task {
  IN TagId tag_id_;      // Tag ID to query
  OUT size_t tag_size_;  // Total size of all blobs in tag
  OUT clio::run::u64 ctime_;   // Tag change-time (last_changed_), ns; 0 if unknown
  OUT clio::run::u64 mtime_;   // Tag modify-time (last_modified_), ns; 0 if unknown
  OUT clio::run::u64 atime_;   // Tag access-time (last_read_), ns; 0 if unknown

  // SHM constructor
  GetTagSizeTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        tag_size_(0),
        ctime_(0),
        mtime_(0),
        atime_(0) {
    // Seed the aggregate accumulator as "not found" so a Broadcast GetTagSize
    // reports rc=1 only when EVERY container missed the tag; AggregateOut flips
    // it to 0 as soon as any container reports a share (see AggregateOut). A
    // locally-executed task overwrites this in Runtime::GetTagSize.
    return_code_.store(1);
  }

  // Emplace constructor
  CTP_CROSS_FUN explicit GetTagSizeTask(const clio::run::TaskId &task_id,
                                         const clio::run::PoolId &pool_id,
                                         const clio::run::PoolQuery &pool_query,
                                         const TagId &tag_id)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetTagSize),
        tag_id_(tag_id),
        tag_size_(0),
        ctime_(0),
        mtime_(0),
        atime_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetTagSize;
    task_flags_.Clear();
    pool_query_ = pool_query;
    // See the SHM constructor: seed "not found" for the Broadcast aggregate.
    return_code_.store(1);
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(tag_size_, ctime_, mtime_, atime_);
  }

  /**
   * Copy from another GetTagSizeTask
   */
  void Copy(const ctp::ipc::FullPtr<GetTagSizeTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    tag_size_ = other->tag_size_;
    ctime_ = other->ctime_;
    mtime_ = other->mtime_;
    atime_ = other->atime_;
  }

  /**
   * AggregateOut results from a replica task.
   *
   * GetTagSize is a Broadcast SUM: the tag's bytes are spread across every
   * container that `HashBlobToContainer` routed one of its blobs to, and each
   * such container carries a TagInfo holding its share. A container that holds
   * NONE of the tag's blobs has no local TagInfo and legitimately returns
   * return_code_=1 ("not found here") with tag_size_=0 — that is a zero
   * contribution to the sum, NOT an error for the whole query.
   *
   * The base Task::AggregateOut propagates any non-zero replica return code onto
   * the aggregate, which poisoned the result: on a multi-node cluster the
   * reader's own container often holds none of a just-written file's blobs, so
   * its rc=1 replica made the aggregate rc=1 even though another container
   * returned the real size. CfsIo::Open only trusts the size when rc==0, so it
   * discarded the correct size and every cross-node read of that file returned
   * 0 bytes (issue #714, Bug 2).
   *
   * Correct semantics — "found if ANY": the aggregate succeeds (rc=0) if at
   * least one replica found the tag, summing the found replicas' shares; it
   * stays "not found" (rc=1) only if EVERY replica missed (e.g. the tag was
   * deleted cluster-wide — readdir's liveness check relies on this). The origin
   * accumulator is seeded rc=1 in the constructors so that this holds
   * regardless of the order replicas are aggregated in.
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    if (other_base.IsNull()) return;
    auto replica = other_base.template Cast<GetTagSizeTask>();
    // Keep the base completer bookkeeping, but NOT its return-code poisoning.
    SetCompleter(other_base->GetCompleter());
    // Sum each container's share. A container that holds none of the tag's
    // blobs reports tag_size_=0, so summing it is a no-op — the total is the
    // sum of the found containers' shares.
    tag_size_ += replica->tag_size_;
    // "found if ANY": the tag exists cluster-wide if at least one container has
    // a local TagInfo (rc==0). Clear the "not found" seed as soon as a replica
    // reports the tag; a not-found replica (rc=1) must NEVER poison the
    // aggregate (that discarded correctly-summed sizes on cross-node reads --
    // #714). rc stays 1 only if every container missed (tag deleted
    // cluster-wide), which readdir's liveness check relies on.
    if (replica->GetReturnCode() == 0) {
      SetReturnCode(0);
    }
    if (replica->ctime_ > ctime_) ctime_ = replica->ctime_;
    if (replica->mtime_ > mtime_) mtime_ = replica->mtime_;
    if (replica->atime_ > atime_) atime_ = replica->atime_;
  }
};

/**
 * GetCapacity task - total and remaining storage capacity.
 *
 * The handler sums max_capacity_ (total) and remaining_space_ (free) over the
 * targets registered on this node, so a Local query returns this node's
 * capacity. Because AggregateOut sums both fields across replicas, a Broadcast
 * query returns the whole cluster's total and remaining capacity.
 */
struct GetCapacityTask : public clio::run::Task {
  OUT clio::run::u64 total_capacity_;      // Summed total capacity in bytes
  OUT clio::run::u64 remaining_capacity_;  // Summed remaining (free) capacity in bytes

  // SHM constructor
  GetCapacityTask()
      : clio::run::Task(), total_capacity_(0), remaining_capacity_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetCapacityTask(const clio::run::TaskId &task_id,
                                            const clio::run::PoolId &pool_id,
                                            const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetCapacity),
        total_capacity_(0),
        remaining_capacity_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetCapacity;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(total_capacity_, remaining_capacity_);
  }

  void Copy(const ctp::ipc::FullPtr<GetCapacityTask> &other) {
    Task::Copy(other.template Cast<Task>());
    total_capacity_ = other->total_capacity_;
    remaining_capacity_ = other->remaining_capacity_;
  }

  /**
   * AggregateOut: sum capacity from each node so a Broadcast yields the total
   * and remaining cluster capacity.
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<GetCapacityTask>();
    total_capacity_ += other->total_capacity_;
    remaining_capacity_ += other->remaining_capacity_;
  }
};

/**
 * GetNumAliases task - number of extra names (tag-level hard links) bound to a
 * tag, identified by name or id. The canonical name is NOT counted, so a file
 * with no hard links returns 0; the POSIX link count is num_aliases_ + 1.
 *
 * The tag's metadata (and its aliases_ list) lives on a single container, so
 * this is a Broadcast-safe op: AggregateOut keeps the answer from whichever
 * replica located the tag (mirrors GetTagName).
 */
struct GetNumAliasesTask : public clio::run::Task {
  IN clio::run::priv::string tag_name_;  // Tag name/path (empty => use tag_id_)
  IN TagId tag_id_;                // Tag id (used when tag_name_ is empty)
  OUT clio::run::u32 num_aliases_;       // # of extra names bound to the tag
  OUT clio::run::u32 found_;             // 1 if the tag's metadata was located

  // SHM constructor
  GetNumAliasesTask()
      : clio::run::Task(),
        tag_name_(CLIO_PRIV_ALLOC),
        tag_id_(TagId::GetNull()),
        num_aliases_(0),
        found_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetNumAliasesTask(const clio::run::TaskId &task_id,
                                           const clio::run::PoolId &pool_id,
                                           const clio::run::PoolQuery &pool_query,
                                           const std::string &tag_name,
                                           const TagId &tag_id)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetNumAliases),
        tag_name_(CLIO_PRIV_ALLOC, tag_name),
        tag_id_(tag_id),
        num_aliases_(0),
        found_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetNumAliases;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_name_, tag_id_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(num_aliases_, found_);
  }

  void Copy(const ctp::ipc::FullPtr<GetNumAliasesTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_name_ = other->tag_name_;
    tag_id_ = other->tag_id_;
    num_aliases_ = other->num_aliases_;
    found_ = other->found_;
  }

  // Broadcast aggregation: keep the answer from whichever container owns the
  // tag (the one that set found_).
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<GetNumAliasesTask>();
    if (other->found_ && !found_) {
      found_ = 1;
      num_aliases_ = other->num_aliases_;
    }
  }
};

/**
 * PollTelemetryLog task - Poll telemetry log with minimum logical time filter
 */
struct PollTelemetryLogTask : public clio::run::Task {
  IN std::uint64_t minimum_logical_time_;        // Minimum logical time filter
  OUT std::uint64_t last_logical_time_;          // Last logical time scanned
  OUT clio::run::priv::vector<CteTelemetry> entries_;  // Retrieved telemetry entries

  // SHM constructor
  PollTelemetryLogTask()
      : clio::run::Task(),
        minimum_logical_time_(0),
        last_logical_time_(0),
        entries_(CLIO_PRIV_ALLOC) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit PollTelemetryLogTask(
      const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, std::uint64_t minimum_logical_time)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kPollTelemetryLog),
        minimum_logical_time_(minimum_logical_time),
        last_logical_time_(0),
        entries_(CLIO_PRIV_ALLOC) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kPollTelemetryLog;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(minimum_logical_time_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(last_logical_time_, entries_);
  }

  /**
   * Copy from another PollTelemetryLogTask
   */
  void Copy(const ctp::ipc::FullPtr<PollTelemetryLogTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    minimum_logical_time_ = other->minimum_logical_time_;
    last_logical_time_ = other->last_logical_time_;
    entries_ = other->entries_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<PollTelemetryLogTask>());
  }
};

/**
 * GetBlobScore task - Get the score of a blob
 */
struct GetBlobScoreTask : public clio::run::Task {
  IN TagId tag_id_;                 // Tag ID for blob lookup
  IN clio::run::priv::string blob_name_;  // Blob name (required)
  OUT float score_;                 // Blob score (0-1)

  // SHM constructor
  GetBlobScoreTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(CLIO_PRIV_ALLOC),
        score_(0.0f) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetBlobScoreTask(const clio::run::TaskId &task_id,
                                           const clio::run::PoolId &pool_id,
                                           const clio::run::PoolQuery &pool_query,
                                           const TagId &tag_id,
                                           const std::string &blob_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetBlobScore),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        score_(0.0f) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetBlobScore;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(score_);
  }

  /**
   * Copy from another GetBlobScoreTask
   */
  void Copy(const ctp::ipc::FullPtr<GetBlobScoreTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    score_ = other->score_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<GetBlobScoreTask>());
  }
};

/**
 * GetBlobSize task - Get the size of a blob
 */
struct GetBlobSizeTask : public clio::run::Task {
  IN TagId tag_id_;                 // Tag ID for blob lookup
  IN clio::run::priv::string blob_name_;  // Blob name (required)
  OUT clio::run::u64 size_;               // Blob size in bytes

  // SHM constructor
  GetBlobSizeTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(CLIO_PRIV_ALLOC),
        size_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetBlobSizeTask(const clio::run::TaskId &task_id,
                                          const clio::run::PoolId &pool_id,
                                          const clio::run::PoolQuery &pool_query,
                                          const TagId &tag_id,
                                          const std::string &blob_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetBlobSize),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        size_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetBlobSize;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(size_);
  }

  /**
   * Copy from another GetBlobSizeTask
   */
  void Copy(const ctp::ipc::FullPtr<GetBlobSizeTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    size_ = other->size_;
  }

  /**
   * AggregateOut replica results into this task
   * @param other Pointer to the replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<GetBlobSizeTask>());
  }
};

/**
 * Block information for GetBlobInfo response
 * Contains the target pool ID and size for each block
 */
struct BlobBlockInfo {
  clio::run::PoolId
      target_pool_id_;     // Pool ID of the target (bdev) storing this block
  clio::run::u64 block_size_;    // Size of this block in bytes
  clio::run::u64 block_offset_;  // Offset within target where block is stored

  BlobBlockInfo() : target_pool_id_(), block_size_(0), block_offset_(0) {}
  BlobBlockInfo(const clio::run::PoolId &pool_id, clio::run::u64 size, clio::run::u64 offset)
      : target_pool_id_(pool_id), block_size_(size), block_offset_(offset) {}

  template <typename Archive>
  void serialize(Archive &ar) {
    clio::run::u64 pool_id_u64 =
        target_pool_id_.IsNull() ? 0 : target_pool_id_.ToU64();
    ar(pool_id_u64, block_size_, block_offset_);
    // Restore PoolId from u64 when deserializing
    target_pool_id_ = clio::run::PoolId::FromU64(pool_id_u64);
  }
};

/**
 * GetBlobInfo task - Get comprehensive blob metadata
 * Returns score, size, and block placement information
 */
struct GetBlobInfoTask : public clio::run::Task {
  IN TagId tag_id_;                        // Tag ID for blob lookup
  IN clio::run::priv::string blob_name_;         // Blob name (required)
  OUT float score_;                        // Blob score (0.0-1.0)
  OUT clio::run::u64 total_size_;                // Total blob size in bytes
  OUT std::vector<BlobBlockInfo> blocks_;  // Block placement info

  // SHM constructor
  GetBlobInfoTask()
      : clio::run::Task(),
        tag_id_(TagId::GetNull()),
        blob_name_(CLIO_PRIV_ALLOC),
        score_(0.0f),
        total_size_(0),
        blocks_() {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetBlobInfoTask(const clio::run::TaskId &task_id,
                                          const clio::run::PoolId &pool_id,
                                          const clio::run::PoolQuery &pool_query,
                                          const TagId &tag_id,
                                          const std::string &blob_name)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetBlobInfo),
        tag_id_(tag_id),
        blob_name_(CLIO_PRIV_ALLOC, blob_name),
        score_(0.0f),
        total_size_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetBlobInfo;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_, blob_name_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(score_, total_size_);
    // NOTE: blocks_ temporarily removed from serialization for debugging
  }

  /**
   * Copy from another GetBlobInfoTask
   */
  void Copy(const ctp::ipc::FullPtr<GetBlobInfoTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_name_ = other->blob_name_;
    score_ = other->score_;
    total_size_ = other->total_size_;
    blocks_ = other->blocks_;
  }

  /**
   * AggregateOut replica results into this task
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<GetBlobInfoTask>());
  }
};

/**
 * GetContainedBlobs task - Get all blob names contained in a tag
 */
struct GetContainedBlobsTask : public clio::run::Task {
  IN TagId tag_id_;                          // Tag ID to query
  OUT std::vector<std::string> blob_names_;  // Vector of blob names in the tag

  // SHM constructor
  GetContainedBlobsTask() : clio::run::Task(), tag_id_(TagId::GetNull()) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit GetContainedBlobsTask(
      const clio::run::TaskId &task_id, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, const TagId &tag_id)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetContainedBlobs),
        tag_id_(tag_id) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kGetContainedBlobs;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_id_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(blob_names_);
  }

  /**
   * Copy from another GetContainedBlobsTask
   */
  void Copy(const ctp::ipc::FullPtr<GetContainedBlobsTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_id_ = other->tag_id_;
    blob_names_ = other->blob_names_;
  }

  /**
   * AggregateOut results from a replica task
   * Merges the blob_names_ vectors from multiple nodes
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto replica = other_base.template Cast<GetContainedBlobsTask>();
    // Merge blob names from replica into this task's blob_names_
    for (size_t i = 0; i < replica->blob_names_.size(); ++i) {
      blob_names_.push_back(replica->blob_names_[i]);
    }
  }
};

/**
 * TagQuery task - Query tags by regex pattern
 * New behavior:
 * - Accepts an input maximum number of tags to store (max_tags_). 0 means no
 *   limit.
 * - Returns a vector of tag names matching the query.
 * - total_tags_matched_ sums the total number of tags that matched the
 *   pattern across replicas during AggregateOut.
 */
struct TagQueryTask : public clio::run::Task {
  IN clio::run::priv::string tag_regex_;
  IN clio::run::u32 max_tags_;
  OUT clio::run::u64 total_tags_matched_;
  OUT std::vector<std::string> results_;
  // Packed TagId per matched tag, index-aligned with results_:
  // (major_ << 32) | minor_. Lets callers (e.g. the filesystem readdir) assign
  // a stable per-tag inode without a second round trip per entry.
  OUT std::vector<clio::run::u64> result_ids_;

  // SHM constructor
  TagQueryTask()
      : clio::run::Task(),
        tag_regex_(CLIO_PRIV_ALLOC),
        max_tags_(0),
        total_tags_matched_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit TagQueryTask(const clio::run::TaskId &task_id,
                                       const clio::run::PoolId &pool_id,
                                       const clio::run::PoolQuery &pool_query,
                                       const std::string &tag_regex,
                                       clio::run::u32 max_tags = 0)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kTagQuery),
        tag_regex_(CLIO_PRIV_ALLOC, tag_regex),
        max_tags_(max_tags),
        total_tags_matched_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kTagQuery;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_regex_, max_tags_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(total_tags_matched_, results_, result_ids_);
  }

  /**
   * Copy from another TagQueryTask
   */
  void Copy(const ctp::ipc::FullPtr<TagQueryTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_regex_ = other->tag_regex_;
    max_tags_ = other->max_tags_;
    total_tags_matched_ = other->total_tags_matched_;
    results_ = other->results_;
    result_ids_ = other->result_ids_;
  }

  /**
   * AggregateOut results from multiple nodes
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<TagQueryTask>();
    // Sum total matched tags across replicas
    total_tags_matched_ += other->total_tags_matched_;

    // Append results (and their packed ids, in lockstep) up to max_tags_.
    for (size_t i = 0; i < other->results_.size(); ++i) {
      if (max_tags_ != 0 && results_.size() >= static_cast<size_t>(max_tags_))
        break;
      results_.push_back(other->results_[i]);
      if (i < other->result_ids_.size())
        result_ids_.push_back(other->result_ids_[i]);
    }
  }
};

/**
 * BlobQuery task - Query blobs by tag and blob regex patterns
 * New behavior:
 * - Accepts an input maximum number of blobs to store (max_blobs_). 0 means no
 *   limit.
 * - Returns a vector of pairs where each pair contains (tag_name, blob_name)
 *   for blobs matching the query.
 * - total_blobs_matched_ sums the total number of blobs that matched across
 *   replicas during AggregateOut.
 */
struct BlobQueryTask : public clio::run::Task {
  IN clio::run::priv::string tag_regex_;
  IN clio::run::priv::string blob_regex_;
  IN clio::run::u32 max_blobs_;
  OUT clio::run::u64 total_blobs_matched_;
  OUT std::vector<std::string> tag_names_;
  OUT std::vector<std::string> blob_names_;

  // SHM constructor
  BlobQueryTask()
      : clio::run::Task(),
        tag_regex_(CLIO_PRIV_ALLOC),
        blob_regex_(CLIO_PRIV_ALLOC),
        max_blobs_(0),
        total_blobs_matched_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit BlobQueryTask(const clio::run::TaskId &task_id,
                                        const clio::run::PoolId &pool_id,
                                        const clio::run::PoolQuery &pool_query,
                                        const std::string &tag_regex,
                                        const std::string &blob_regex,
                                        clio::run::u32 max_blobs = 0)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kBlobQuery),
        tag_regex_(CLIO_PRIV_ALLOC, tag_regex),
        blob_regex_(CLIO_PRIV_ALLOC, blob_regex),
        max_blobs_(max_blobs),
        total_blobs_matched_(0) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kBlobQuery;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  /**
   * Serialize IN and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_regex_, blob_regex_, max_blobs_);
  }

  /**
   * Serialize OUT and INOUT parameters
   */
  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(total_blobs_matched_, tag_names_, blob_names_);
  }

  /**
   * Copy from another BlobQueryTask
   */
  void Copy(const ctp::ipc::FullPtr<BlobQueryTask> &other) {
    // Copy base Task fields
    Task::Copy(other.template Cast<Task>());
    tag_regex_ = other->tag_regex_;
    blob_regex_ = other->blob_regex_;
    max_blobs_ = other->max_blobs_;
    total_blobs_matched_ = other->total_blobs_matched_;
    tag_names_ = other->tag_names_;
    blob_names_ = other->blob_names_;
  }

  /**
   * AggregateOut results from multiple nodes
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<BlobQueryTask>();
    // Sum total matched blobs across replicas
    total_blobs_matched_ += other->total_blobs_matched_;

    // Append results up to max_blobs_ (if non-zero)
    for (size_t i = 0; i < other->tag_names_.size(); ++i) {
      if (max_blobs_ != 0 &&
          tag_names_.size() >= static_cast<size_t>(max_blobs_))
        break;
      tag_names_.push_back(other->tag_names_[i]);
      blob_names_.push_back(other->blob_names_[i]);
    }
  }
};

/**
 * FlushMetadataTask - Periodic task to flush tag/blob metadata to durable
 * storage
 */
struct FlushMetadataTask : public clio::run::Task {
  OUT clio::run::u64 entries_flushed_;

  /** SHM default constructor */
  FlushMetadataTask() : clio::run::Task(), entries_flushed_(0) {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit FlushMetadataTask(const clio::run::TaskId &task_node,
                                            const clio::run::PoolId &pool_id,
                                            const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kFlushMetadata),
        entries_flushed_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kFlushMetadata;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(entries_flushed_);
  }

  void Copy(const ctp::ipc::FullPtr<FlushMetadataTask> &other) {
    Task::Copy(other.template Cast<Task>());
    entries_flushed_ = other->entries_flushed_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<FlushMetadataTask>());
  }
};

/**
 * FlushDataTask - Periodic task to flush data from volatile to non-volatile
 * targets
 */
struct FlushDataTask : public clio::run::Task {
  IN int target_persistence_level_;
  OUT clio::run::u64 bytes_flushed_;
  OUT clio::run::u64 blobs_flushed_;

  /** SHM default constructor */
  FlushDataTask()
      : clio::run::Task(),
        target_persistence_level_(1),
        bytes_flushed_(0),
        blobs_flushed_(0) {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit FlushDataTask(const clio::run::TaskId &task_node,
                                        const clio::run::PoolId &pool_id,
                                        const clio::run::PoolQuery &pool_query,
                                        int target_persistence_level = 1)
      : clio::run::Task(task_node, pool_id, pool_query, Method::kFlushData),
        target_persistence_level_(target_persistence_level),
        bytes_flushed_(0),
        blobs_flushed_(0) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kFlushData;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(target_persistence_level_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(bytes_flushed_, blobs_flushed_);
  }

  void Copy(const ctp::ipc::FullPtr<FlushDataTask> &other) {
    Task::Copy(other.template Cast<Task>());
    target_persistence_level_ = other->target_persistence_level_;
    bytes_flushed_ = other->bytes_flushed_;
    blobs_flushed_ = other->blobs_flushed_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<FlushDataTask>());
  }
};

/**
 * DynamicReorganizeTask - Periodic driver for the internal data organizer
 * (issue #738).
 *
 * Spawned `organizer_tasks` times from Create() when an organizer is
 * configured; each firing delegates to the configured DataOrganizer:
 * organizer->Reorganize(this, task->replica_id_). replica_id_ identifies
 * which of the parallel organizer replicas this task is (0-based) so the
 * organizer can partition the blob space across replicas.
 */
struct DynamicReorganizeTask : public clio::run::Task {
  IN clio::run::u32 replica_id_;  // 0-based organizer replica index

  /** SHM default constructor */
  DynamicReorganizeTask() : clio::run::Task(), replica_id_(0) {}

  /** Emplace constructor */
  CTP_CROSS_FUN explicit DynamicReorganizeTask(
      const clio::run::TaskId &task_node, const clio::run::PoolId &pool_id,
      const clio::run::PoolQuery &pool_query, clio::run::u32 replica_id = 0)
      : clio::run::Task(task_node, pool_id, pool_query,
                        Method::kDynamicReorganize),
        replica_id_(replica_id) {
    task_id_ = task_node;
    pool_id_ = pool_id;
    method_ = Method::kDynamicReorganize;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(replica_id_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    // No output parameters (return_code_ handled by base class)
  }

  void Copy(const ctp::ipc::FullPtr<DynamicReorganizeTask> &other) {
    Task::Copy(other.template Cast<Task>());
    replica_id_ = other->replica_id_;
  }

  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    Copy(other_base.template Cast<DynamicReorganizeTask>());
  }
};

/**
 * One row in a SemanticSearchTask result vector. blob_name_ acts as
 * the "BlobId" within tag_id_ — CTE doesn't have a separate BlobId
 * type, blobs are addressed by (TagId, name).
 */
struct SemanticSearchResult {
  TagId tag_id_;
  std::string tag_name_;
  std::string blob_name_;
  double score_;  // BM25; higher = better

  SemanticSearchResult() : tag_id_(TagId::GetNull()), score_(0.0) {}
  SemanticSearchResult(const TagId &t, std::string tn, std::string n, double s)
      : tag_id_(t), tag_name_(std::move(tn)), blob_name_(std::move(n)), score_(s) {}

  template <typename Archive>
  void serialize(Archive &ar) {
    ar(tag_id_, tag_name_, blob_name_, score_);
  }
};

/**
 * SemanticSearchTask — keyword search over blob contents with BM25.
 *
 * Step 1: tag_regex_ AND blob_regex_ filter the candidate (tag, blob)
 *         pairs, exactly like BlobQuery.
 * Step 2: every candidate blob's bytes are tokenized into lowercase
 *         alphanumeric terms and BM25-scored against the same
 *         tokenization of query_text_.
 * Step 3: results are sorted descending by score and trimmed to k_.
 *
 * BM25 corpus statistics (avgdl, df) are computed over the matched
 * working set rather than the whole CTE — the query is "find the
 * best matches *within this regex slice*", not "rank against
 * everything CTE has ever seen".
 *
 * Regexes use std::regex_match (full-string) for parity with
 * BlobQueryTask. Use ".*pattern.*" for substring matching.
 */
struct SemanticSearchTask : public clio::run::Task {
  IN clio::run::priv::string tag_regex_;
  IN clio::run::priv::string blob_regex_;
  IN clio::run::priv::string query_text_;
  IN clio::run::u32 k_;
  OUT std::vector<SemanticSearchResult> results_;

  // SHM constructor
  SemanticSearchTask()
      : clio::run::Task(),
        tag_regex_(CLIO_PRIV_ALLOC),
        blob_regex_(CLIO_PRIV_ALLOC),
        query_text_(CLIO_PRIV_ALLOC),
        k_(10) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit SemanticSearchTask(const clio::run::TaskId &task_id,
                                             const clio::run::PoolId &pool_id,
                                             const clio::run::PoolQuery &pool_query,
                                             const std::string &tag_regex,
                                             const std::string &blob_regex,
                                             const std::string &query_text,
                                             clio::run::u32 k)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kSemanticSearch),
        tag_regex_(CLIO_PRIV_ALLOC, tag_regex),
        blob_regex_(CLIO_PRIV_ALLOC, blob_regex),
        query_text_(CLIO_PRIV_ALLOC, query_text),
        k_(k) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kSemanticSearch;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_regex_, blob_regex_, query_text_, k_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(results_);
  }

  void Copy(const ctp::ipc::FullPtr<SemanticSearchTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_regex_ = other->tag_regex_;
    blob_regex_ = other->blob_regex_;
    query_text_ = other->query_text_;
    k_ = other->k_;
    results_ = other->results_;
  }

  /**
   * AggregateOut one replica's results into this (origin) task.
   *
   * SemanticSearch defaults to a Broadcast query, so every tag-owning
   * container runs BM25 over its own slice and returns its local top-k
   * (already sorted descending by score). AggregateOut is called once per
   * replica; it must MERGE those partial result sets and keep the global
   * top-k by score — not overwrite (the previous Copy() dropped every
   * replica but the last). Merge → sort descending by BM25 score → trim
   * to k_ (k_ == 0 means "no cap", keep all).
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<SemanticSearchTask>();
    if (other->results_.empty()) {
      return;
    }
    results_.insert(results_.end(), other->results_.begin(),
                    other->results_.end());
    std::sort(results_.begin(), results_.end(),
              [](const SemanticSearchResult &a, const SemanticSearchResult &b) {
                return a.score_ > b.score_;
              });
    if (k_ > 0 && results_.size() > k_) {
      results_.resize(k_);
    }
  }
};

/**
 * One row in a TemporalSearchTask result vector.
 */
struct TemporalSearchResult {
  TagId tag_id_;
  std::string tag_name_;
  std::string blob_name_;
  Timestamp last_modified_;  // epoch nanoseconds from blob metadata

  TemporalSearchResult() : tag_id_(TagId::GetNull()), last_modified_(0) {}
  TemporalSearchResult(const TagId &t, std::string tn, std::string n, Timestamp ts)
      : tag_id_(t), tag_name_(std::move(tn)), blob_name_(std::move(n)), last_modified_(ts) {}

  template <typename Archive>
  void serialize(Archive &ar) {
    ar(tag_id_, tag_name_, blob_name_, last_modified_);
  }
};

/**
 * TemporalSearchTask — find blobs by last-modified timestamp.
 *
 * Filters tags by tag_regex_ and blobs by blob_regex_, then returns
 * every matching blob whose last_modified_ falls within
 * [time_begin_, time_end_] (inclusive; 0 on either bound means
 * "no constraint on that side").  Results are sorted by ascending
 * last_modified_.  max_entries_ caps the output (0 = unlimited).
 *
 * This is a pure metadata scan — no blob bytes are read.
 */
struct TemporalSearchTask : public clio::run::Task {
  IN clio::run::priv::string tag_regex_;
  IN clio::run::priv::string blob_regex_;
  IN Timestamp time_begin_;
  IN Timestamp time_end_;
  IN clio::run::u32 max_entries_;
  OUT std::vector<TemporalSearchResult> results_;

  // SHM constructor
  TemporalSearchTask()
      : clio::run::Task(),
        tag_regex_(CLIO_PRIV_ALLOC),
        blob_regex_(CLIO_PRIV_ALLOC),
        time_begin_(0),
        time_end_(0),
        max_entries_(0) {}

  // Emplace constructor
  CTP_CROSS_FUN explicit TemporalSearchTask(const clio::run::TaskId &task_id,
                                             const clio::run::PoolId &pool_id,
                                             const clio::run::PoolQuery &pool_query,
                                             const std::string &tag_regex,
                                             const std::string &blob_regex,
                                             Timestamp time_begin,
                                             Timestamp time_end,
                                             clio::run::u32 max_entries)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kTemporalSearch),
        tag_regex_(CLIO_PRIV_ALLOC, tag_regex),
        blob_regex_(CLIO_PRIV_ALLOC, blob_regex),
        time_begin_(time_begin),
        time_end_(time_end),
        max_entries_(max_entries) {
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = Method::kTemporalSearch;
    task_flags_.Clear();
    pool_query_ = pool_query;
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(tag_regex_, blob_regex_, time_begin_, time_end_, max_entries_);
  }

  template <typename Archive>
  CTP_CROSS_FUN void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(results_);
  }

  void Copy(const ctp::ipc::FullPtr<TemporalSearchTask> &other) {
    Task::Copy(other.template Cast<Task>());
    tag_regex_ = other->tag_regex_;
    blob_regex_ = other->blob_regex_;
    time_begin_ = other->time_begin_;
    time_end_ = other->time_end_;
    max_entries_ = other->max_entries_;
    results_ = other->results_;
  }

  /**
   * AggregateOut one replica's results into this (origin) task.
   *
   * Like SemanticSearch, TemporalSearch defaults to a Broadcast query: every
   * tag-owning container returns its own oldest `max_entries_` blobs (sorted
   * ascending by last-modified timestamp). AggregateOut is called once per
   * replica; it must MERGE those partial sets and keep the global oldest
   * `max_entries_` — not overwrite (the previous Copy() dropped every replica
   * but the last). Merge → sort ascending by timestamp → trim to max_entries_
   * (max_entries_ == 0 means "no cap", keep all).
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    auto other = other_base.template Cast<TemporalSearchTask>();
    if (other->results_.empty()) {
      return;
    }
    results_.insert(results_.end(), other->results_.begin(),
                    other->results_.end());
    std::sort(results_.begin(), results_.end(),
              [](const TemporalSearchResult &a, const TemporalSearchResult &b) {
                return a.last_modified_ < b.last_modified_;
              });
    if (max_entries_ > 0 && results_.size() > max_entries_) {
      results_.resize(max_entries_);
    }
  }
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_TASKS_H_
