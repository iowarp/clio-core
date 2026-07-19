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

#ifndef CLIO_RUNTIME_INCLUDE_CONTAINER_H_
#define CLIO_RUNTIME_INCLUDE_CONTAINER_H_

#include <cmath>
#include <clio_ctp/data_structures/serialization/global_serialize.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "clio_runtime/config_manager.h"
#include "clio_runtime/corwlock.h"
#include "clio_runtime/pool_query.h"
#include "clio_runtime/task.h"
#include "clio_runtime/task_archives.h"
#include "clio_runtime/local_task_archives.h"
#include "clio_runtime/types.h"

// Forward declarations to avoid circular dependencies
namespace clio::run {
class WorkOrchestrator;
}

/**
 * Container Base Class with Default Implementations
 *
 * Provides default implementations of ChiContainer methods for simpler modules.
 * Modules can inherit from this class instead of ChiContainer to get basic
 * queue and lane management functionality out of the box.
 */

namespace clio::run {

/**
 * Monitor mode identifiers for task scheduling
 */
enum class MonitorModeId : u32 {
  kLocalSchedule = 0,   ///< Route task to local container queue lane
  kGlobalSchedule = 1,  ///< Coordinate global task distribution
  kEstLoad = 2,         ///< Estimate task execution time for waiting
};

/**
 * Queue identifier
 */
using QueueId = u32;

/**
 * Container - Base class for all containers
 *
 * Unified container class that provides all functionality for task processing,
 * monitoring, and scheduling. Replaces the previous ChiContainer/Container
 * split.
 */
class Container {
 public:
  static constexpr u32 CONTAINER_PLUG = BIT_OPT(u32, 0);

  /** RPC visibility, as a small bitfield. A "private" RPC rejects calls from
   *  external user clients; runtime-internal callers are always allowed. */
  struct MethodProperty {
    static constexpr u32 kPrivate = BIT_OPT(u32, 0);
    u32 bits_ = 0;
    bool IsPrivate() const { return (bits_ & kPrivate) != 0; }
  };

  PoolId pool_id_;         ///< The unique ID of this pool
  std::string pool_name_;  ///< The semantic name of this pool
  u32 container_id_;       ///< The logical ID of this container instance
  ctp::abitfield32_t flags_;  ///< Atomic bitfield for container state

  /** Group affinity map: TaskGroup id -> pinned Worker* */
  std::unordered_map<int64_t, Worker*> task_group_map_;
  /** Lock protecting task_group_map_ */
  CoRwLock task_group_lock_;

 protected:
  PoolQuery pool_query_;
  std::vector<float> method_model_;  ///< Per-method CPU coefficient: predicted_us = a * compute
  std::vector<float> method_mape_;   ///< Per-method CPU MAPE (exponential moving average)
  std::vector<float> method_model_wall_;  ///< Per-method wall clock coefficient "b"
  std::vector<float> method_mape_wall_;   ///< Per-method wall clock MAPE
  std::vector<std::string> method_names_;  ///< Per-method human-readable names
  float learning_rate_ = 0.2f;      ///< SGD learning rate for model updates
  /** Default RPC visibility for this container (from compose
   *  container_visibility). */
  MethodProperty container_visibility_;
  /** Per-method visibility overrides keyed by method id (from compose
   *  container_rpc_acl). Built once in ConfigureAcl, then read-only — no
   *  locking needed on the enforcement hot path. */
  std::unordered_map<u32, MethodProperty> method_acl_;

 public:
  Container() = default;
  virtual ~Container() {
    // Note: Lane mappings are managed by WorkOrchestrator lifecycle
    // No explicit cleanup needed since lanes are mapped, not registered
  }

  /**
   * Initialize container with pool information
   * @param pool_id The unique ID of this pool
   * @param pool_name The semantic name of this pool (user-provided)
   * @param container_id The container ID (typically the node ID where this container exists)
   *
   * ChiMod runtime classes should override this method to initialize their client member.
   */
  virtual void Init(const PoolId& pool_id, const std::string& pool_name,
                    u32 container_id = 0) {
    pool_id_ = pool_id;
    pool_name_ = pool_name;
    container_id_ = container_id;
    flags_.Clear();
    pool_query_ = PoolQuery();  // Default pool query
    task_group_map_.clear();
  }

  /**
   * Initialize the per-method linear model table.
   * Called by autogenerated Init() after container setup.
   * Override to set custom initial coefficients per method.
   * @param max_method_id Size of model table (from Method::kMaxMethodId)
   */
  virtual void DefineModel(u32 max_method_id) {
    method_model_.resize(max_method_id, 1.0f);
    method_mape_.resize(max_method_id, 0.0f);
    method_model_wall_.resize(max_method_id, 1.0f);
    method_mape_wall_.resize(max_method_id, 0.0f);
    auto *config = CLIO_CONFIG_MANAGER;
    if (config) {
      learning_rate_ = config->GetLearningRate();
    }
  }

  /**
   * Set method names for model introspection.
   * Called by autogenerated Init() after DefineModel().
   * @param names Vector of method names indexed by method ID
   */
  void SetMethodNames(const std::vector<std::string>& names) {
    method_names_ = names;
  }

  /**
   * Configure per-RPC access control from a compose PoolConfig. Must be called
   * once after Init()/SetMethodNames() (so method_names_ is populated), before
   * the container serves tasks. Translates the name-keyed ACL into a
   * method_id-keyed map; afterwards the ACL state is read-only.
   * @param container_visibility default visibility (0 public, 1 private)
   * @param rpc_acl per-RPC overrides: method NAME -> 0 public / 1 private
   */
  void ConfigureAcl(u32 container_visibility,
                    const std::unordered_map<std::string, u32>& rpc_acl) {
    container_visibility_.bits_ =
        container_visibility ? MethodProperty::kPrivate : 0u;
    method_acl_.clear();
    for (const auto& kv : rpc_acl) {
      for (u32 id = 0; id < method_names_.size(); ++id) {
        if (method_names_[id] == kv.first) {
          method_acl_[id].bits_ = kv.second ? MethodProperty::kPrivate : 0u;
          break;
        }
      }
    }
  }

  /**
   * @return true if a caller may invoke `method_id`. Runtime-internal callers
   * (is_external == false) are always allowed. External user clients are
   * rejected when the method is private (per-method override if present, else
   * the container default visibility).
   */
  bool IsRpcAllowed(u32 method_id, bool is_external) const {
    if (!is_external) {
      return true;
    }
    auto it = method_acl_.find(method_id);
    const MethodProperty& mp =
        (it != method_acl_.end()) ? it->second : container_visibility_;
    return !mp.IsPrivate();
  }

  /**
   * Get live task statistics for THIS specific task instance.
   *
   * Receives a pointer to the actual task so the override can read its
   * payload-bearing fields (e.g. WriteTask::length_, PutBlobTask::size_)
   * and report the real I/O size for routing/scheduling. Returning a
   * static per-method estimate is almost never what callers want — the
   * scheduler routes "large I/O" tasks to I/O workers and "small/
   * metadata" tasks to the scheduler worker, and that decision needs the
   * actual byte count, not a placeholder constant.
   *
   * Default returns zeros; override in modules whose tasks carry
   * variable-size payloads.
   *
   * @param task Pointer to the task instance (must not be null; cast to
   *             the module's concrete task type to read payload fields).
   * @return TaskStat with compute/io_size/wall_time populated for this task.
   */
  virtual TaskStat GetTaskStats(const Task *task) const {
    (void)task;
    return TaskStat();
  }

  /**
   * Predict CPU time: a * (compute + 1).
   * @param method_id Method being executed
   * @param stat Task statistics from GetTaskStats()
   * @return Predicted CPU time in microseconds
   */
  float InferCpuTime(u32 method_id, const TaskStat &stat) {
    float x = static_cast<float>(stat.compute_) + 1.0f;
    if (method_id < method_model_.size()) {
      return method_model_[method_id] * x;
    }
    return x;
  }

  /**
   * Predict wall clock time: b * (wall_time + 1).
   * @param method_id Method being executed
   * @param stat Task statistics from GetTaskStats()
   * @return Predicted wall clock time in microseconds
   */
  float InferWallClockTime(u32 method_id, const TaskStat &stat) {
    float x = stat.wall_time_ + 1.0f;
    if (method_id < method_model_wall_.size()) {
      return method_model_wall_[method_id] * x;
    }
    return x;
  }

  /**
   * Update CPU model coefficient after task completion.
   * SGD: a' <- a - LR * (e / x), where e = predicted - real.
   */
  void ReinforceCpuModel(u32 method_id, float pred_cpu, float real_cpu,
                         const TaskStat &stat) {
    if (method_id >= method_model_.size()) return;
    float x = static_cast<float>(stat.compute_) + 1.0f;
    float e = pred_cpu - real_cpu;
    method_model_[method_id] -= learning_rate_ * (e / x);
    if (real_cpu > 0) {
      float ape = std::abs(e) / real_cpu;
      method_mape_[method_id] = (1.0f - learning_rate_) * method_mape_[method_id]
                               + learning_rate_ * ape;
    }
  }

  /**
   * Update wall clock model coefficient after task completion.
   * SGD: b' <- b - LR * (e / x), where e = predicted - real.
   */
  void ReinforceWallModel(u32 method_id, float pred_wall, float real_wall,
                          const TaskStat &stat) {
    if (method_id >= method_model_wall_.size()) return;
    float x = stat.wall_time_ + 1.0f;
    float e = pred_wall - real_wall;
    method_model_wall_[method_id] -= learning_rate_ * (e / x);
    if (real_wall > 0) {
      float ape = std::abs(e) / real_wall;
      method_mape_wall_[method_id] = (1.0f - learning_rate_) * method_mape_wall_[method_id]
                                    + learning_rate_ * ape;
    }
  }

  /**
   * Get the MAPE for a given method.
   * @param method_id Method to query
   * @return MAPE as a fraction (0.0 = perfect, 1.0 = 100% error)
   */
  float GetMethodMape(u32 method_id) const {
    if (method_id < method_mape_.size()) {
      return method_mape_[method_id];
    }
    return 0.0f;
  }

  const std::vector<float>& GetMethodModel() const { return method_model_; }
  const std::vector<float>& GetMethodMapeVec() const { return method_mape_; }
  const std::vector<float>& GetMethodModelWall() const { return method_model_wall_; }
  const std::vector<float>& GetMethodMapeWallVec() const { return method_mape_wall_; }
  const std::vector<std::string>& GetMethodNames() const { return method_names_; }
  float GetLearningRate() const { return learning_rate_; }

  /** Mark container as plugged (no new tasks accepted) */
  void SetPlugged() { flags_.SetBits(CONTAINER_PLUG); }

  /** Check if container is plugged */
  bool IsPlugged() const { return flags_.Any(CONTAINER_PLUG) != 0; }

  /**
   * Schedule a task by resolving its PoolQuery before routing.
   * Called from RouteTask on the static container (no container state).
   * Override in chimods to implement dynamic scheduling logic (e.g.,
   * checking local caches, hashing blob names to containers).
   * Default: returns the task's existing pool_query_ unchanged.
   *
   * @param task Full pointer to the task being scheduled
   * @return The PoolQuery to use for routing this task
   */
  virtual PoolQuery ScheduleTask(
      const clio::run::shared_ptr<Task> &task) {
    return task->pool_query_;
  }

  /**
   * Execute a method on a task - must be implemented by derived classes
   *
   * This method returns TaskResume to support C++20 coroutine-based execution.
   * The returned TaskResume holds the coroutine handle that the worker uses
   * to suspend and resume task execution. The RunContext is obtained inside
   * handlers via clio::run::GetCurrentRunContext() (the worker sets it before
   * every Run/resume), so it is no longer a parameter.
   */
  virtual TaskResume Run(u32 method,
                         clio::run::shared_ptr<Task> task_ptr) = 0;

  /**
   * Fix up POD bytewise-copied tasks (clio::run::priv::string SSO data_
   * pointers, etc.) before dispatching Run. Called by the GPU2CPU pop
   * path when the kernel's task was in kDeviceMem and the worker had
   * to D2H-copy the POD bytes into a host scratch buffer — at that
   * point any embedded `data_` pointers still reference the original
   * device buffer offsets and must be re-pointed at the scratch copy.
   *
   * Default no-op for chimods whose tasks are already trivially
   * relocatable (no SSO strings, no SVO vectors). Each chimod that
   * has SSO/SVO members should override and dispatch per method to
   * the per-task `FixupAfterCopy()`.
   */
  virtual void FixupAfterCopy(u32 method,
                              clio::run::shared_ptr<Task> &task_ptr) {
    (void)method;
    (void)task_ptr;
  }


  /**
   * Get remaining work count for this container - PURE VIRTUAL
   * Must be implemented by all derived container classes
   * @return Number of work units remaining in this container
   */
  virtual u64 GetWorkRemaining() const = 0;

  /**
   * Update work count for a task - should be overridden by derived classes
   * @param task_ptr Task being executed
   * @param rctx Current run context
   * @param increment Work count change (positive or negative)
   */
  virtual void UpdateWork(clio::run::shared_ptr<Task> &task_ptr,
                          i64 increment) {
    // Default: no work tracking
    (void)task_ptr;
    (void)increment;  // Suppress unused warnings
  }

  /**
   * Restart container on the SAME node after a brief shutdown.
   * Called during warm-start via RestartContainers / Compose pathway.
   * Aims to rebuild metadata only (data assumed intact on local storage).
   * Default: calls Init. Override to reload metadata from WAL/config.
   */
  virtual void Restart(const PoolId& pool_id, const std::string& pool_name,
                       u32 container_id = 0) {
    Init(pool_id, pool_name, container_id);
  }

  /**
   * Schedule which node should host recovery of this container.
   * Called on the LEADER node's local_container_ during ComputeRecoveryPlan.
   * Default: return static_cast<u32>(-1) to let the admin choose at random.
   * Override to direct recovery to a specific node (e.g., nearest replica).
   * @return Destination node ID, or static_cast<u32>(-1) for random placement
   */
  virtual u32 ScheduleRecover() {
    return static_cast<u32>(-1);
  }

  /**
   * Recover container after node failure onto a DIFFERENT node.
   * Called on the DESTINATION node during RecoverContainers.
   * Aims to reconstruct both data and metadata from replicas/checkpoints.
   * Default: calls Init (clean slate). Override for state restoration.
   * @param pool_id The pool this container belongs to
   * @param pool_name The pool name
   * @param container_id The container ID being recovered
   */
  virtual void Recover(const PoolId& pool_id, const std::string& pool_name,
                       u32 container_id = 0) {
    Init(pool_id, pool_name, container_id);
  }

  /**
   * Expand container to accommodate a new node in the cluster
   * Called when a new node is registered via Admin::AddNode.
   * Default implementation is a no-op.
   * Override to re-partition data or update routing when nodes join.
   * @param new_host The newly registered host
   */
  virtual void Expand(const Host& new_host) {
    (void)new_host;
  }

  /**
   * Migrate this container's data to a destination node
   * Called during container migration. Override to serialize and transfer state.
   * Default implementation is a no-op.
   * @param dest_node_id The node ID to migrate to
   */
  virtual void Migrate(u32 dest_node_id) {
    (void)dest_node_id;
  }

  /**
   * Called after the GPU container for this pool has been allocated and
   * registered with the GPU work orchestrator (and the orchestrator has been
   * resumed). Override to send GPU-init tasks that must arrive after the
   * GPU container is registered (e.g., bdev's UpdateTask).
   * Default implementation is a no-op.
   */
  virtual void PostGpuContainerCreate() {}

  /**
   * Serialize task parameters for network transfer (unified method)
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate serialization
   * @param method The method ID to serialize
   * @param archive SaveTaskArchive configured with srl_mode (true=In, false=Out)
   * @param task_ptr Full pointer to the task to serialize
   */
  virtual void SaveTask(u32 method, SaveTaskArchive& archive,
                        clio::run::shared_ptr<Task> &task_ptr) = 0;

  /**
   * Deserialize task parameters into an existing task from network transfer
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate deserialization
   * Does not allocate - assumes task_ptr is already allocated
   * @param method The method ID to deserialize
   * @param archive LoadTaskArchive configured with srl_mode (true=In, false=Out)
   * @param task_ptr Full pointer to the pre-allocated task to load into
   */
  virtual void LoadTask(u32 method, LoadTaskArchive& archive,
                        clio::run::shared_ptr<Task> &task_ptr) = 0;

  /**
   * Allocate and deserialize task parameters from network transfer
   * Wrapper that calls NewTask followed by LoadTask
   * @param method The method ID to deserialize
   * @param archive LoadTaskArchive configured with srl_mode (true=In, false=Out)
   * @return Full pointer to the newly allocated and deserialized task
   */
  virtual clio::run::shared_ptr<Task> AllocLoadTask(u32 method, LoadTaskArchive& archive) = 0;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate deserialization
   * Does not allocate - assumes task_ptr is already allocated
   * @param method The method ID to deserialize
   * @param archive DefaultLoadArchive for deserializing inputs
   * @param task_ptr Full pointer to the pre-allocated task to load into
   */
  virtual void LocalLoadTask(u32 method, DefaultLoadArchive& archive,
                             clio::run::shared_ptr<Task> &task_ptr) = 0;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   * Wrapper that calls NewTask followed by LocalLoadTask
   * @param method The method ID to deserialize
   * @param archive DefaultLoadArchive for deserializing inputs
   * @return Full pointer to the newly allocated and loaded task
   */
  virtual clio::run::shared_ptr<Task> LocalAllocLoadTask(u32 method, DefaultLoadArchive& archive) = 0;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate serialization
   * @param method The method ID to serialize
   * @param archive DefaultSaveArchive for serializing outputs
   * @param task_ptr Full pointer to the task to save outputs from
   */
  virtual void LocalSaveTask(u32 method, DefaultSaveArchive& archive,
                              clio::run::shared_ptr<Task> &task_ptr) = 0;

  /**
   * Create a new copy of a task (deep copy for distributed execution) - must be
   * implemented by derived classes Uses switch-case structure based on method
   * ID to dispatch to appropriate task type copying
   * @param method The method ID for the task type
   * @param orig_task_ptr Full pointer to the original task
   * @param deep Whether to perform a deep copy
   * @return Full pointer to the newly created copy
   */
  CTP_DLL virtual clio::run::shared_ptr<Task> NewCopyTask(u32 method,
                                                    clio::run::shared_ptr<Task> &orig_task_ptr,
                                                    bool deep) = 0;

  /**
   * Create a new task of the specified method type
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate task type allocation
   * @param method The method ID for the task type to create
   * @return Full pointer to the newly allocated task (cast to base Task type)
   */
  CTP_DLL virtual clio::run::shared_ptr<Task> NewTask(u32 method) = 0;

  /**
   * AggregateOut replica OUTPUTS into the origin task via Container dispatch
   * (a.k.a. AggregateOut). This is the existing output-merge semantics: every
   * chimod's per-task AggregateOut() merges OUT fields of a replica into the
   * origin (used by the replica/gather path in RecvOutAggregate and by the
   * ManyToOne result broadcast).
   * Replaces virtual Task::AggregateOut to avoid vtable on Task
   * @param method The method ID for proper task type casting
   * @param orig_task The origin task to aggregate into
   * @param replica_task The replica task to aggregate from
   */
  virtual void AggregateOut(u32 method, clio::run::shared_ptr<Task> &orig_task,
                          const clio::run::shared_ptr<Task>& replica_task) = 0;

  /**
   * AggregateOut member INPUTS into a collective aggregate task (ManyToOne).
   * Combines the IN fields of a batched member into the synthetic aggregate
   * task that will run once for the whole batch. Distinct from AggregateOut
   * (AggregateOut), which merges OUT fields. Default is a no-op: with no
   * override the aggregate runs as a copy of the first member (e.g. a barrier
   * / dedup collective). Chimods whose collective combines inputs (sum, max,
   * concat, ...) override this. Dispatched per method by the chimod.
   * @param method The method ID for proper task type casting
   * @param agg_task The synthetic aggregate task to combine into
   * @param member_task A batched member whose inputs are folded in
   */
  virtual void AggregateIn(u32 method, clio::run::shared_ptr<Task> &agg_task,
                           const clio::run::shared_ptr<Task>& member_task) {
    (void)method;
    (void)agg_task;
    (void)member_task;
  }

  // NOTE: There is no DelTask virtual. Tasks are clio::run::shared_ptr handles
  // freed automatically (RAII) when their last owner drops. Type-correct
  // destruction is guaranteed by the type-erased deleter in the control header.
};

/**
 * Container Client Interface (Client-Side)
 *
 * Minimal client interface for task submission.
 * Executes in user processes, performs only task allocation and queueing.
 */
class ContainerClient {
 public:
  PoolId pool_id_;  ///< The unique ID of the pool this client connects to
  u32 return_code_; ///< Return code from the last Create operation (0=success, non-zero=error)

  /**
   * Default constructor
   */
  CTP_CROSS_FUN ContainerClient() : pool_id_(), return_code_(0) {}

#if CTP_IS_HOST
  /**
   * Initialize client with pool ID
   * @param pool_id Pool identifier to connect to
   */
  virtual void Init(const PoolId& pool_id) {
    pool_id_ = pool_id;
    return_code_ = 0;
  }

  /**
   * Virtual destructor
   */
  virtual ~ContainerClient() = default;
#else
  /**
   * Initialize client with pool ID (GPU version, non-virtual)
   * @param pool_id Pool identifier to connect to
   */
  CTP_GPU_FUN void Init(const PoolId& pool_id) {
    pool_id_ = pool_id;
    return_code_ = 0;
  }
#endif

  /**
   * Serialization support
   */
  template <typename Ar>
  void serialize(Ar& ar) {
    ar(pool_id_, return_code_);
  }

  /**
   * Check if the client's pool ID is null/invalid
   * @return true if pool_id_ is null, false otherwise
   */
  bool IsNull() const {
    return pool_id_.IsNull();
  }

  /**
   * Get the return code from the last Create operation
   * @return Return code (0=success, non-zero=error)
   */
  u32 GetReturnCode() const {
    return return_code_;
  }

  /**
   * Set the return code for the client
   * @param return_code Return code to set (0=success, non-zero=error)
   */
  void SetReturnCode(u32 return_code) {
    return_code_ = return_code;
  }

 protected:
  /**
   * Helper method to allocate a new task
   * @param args Arguments for task construction
   * @return Full pointer to allocated task
   */
  template <typename TaskT, typename... Args>
  clio::run::shared_ptr<TaskT> AllocateTask(MemorySegment segment, Args&&... args);
};

}  // namespace clio::run

/**
 * ChiMod Entry Point Macros
 *
 * These macros must be used in the runtime implementation file to
 * export the required C symbols for dynamic loading.
 */

extern "C" {
// Required ChiMod entry points
typedef clio::run::Container* (*alloc_chimod_t)();
typedef clio::run::Container* (*new_chimod_t)(const clio::run::PoolId* pool_id,
                                        const char* pool_name);
typedef const char* (*get_chimod_name_t)(void);
typedef void (*destroy_chimod_t)(clio::run::Container* container);
}

/**
 * Macro to define ChiMod entry points in runtime source file (deprecated)
 *
 * Usage: CLIO_CHIMOD_CC(MyContainerClass, "my_chimod_name")
 * Note: Use CLIO_TASK_CC instead for new modules
 */
#define CLIO_CHIMOD_CC(CONTAINER_CLASS, MOD_NAME)                    \
  extern "C" {                                                       \
  clio::run::Container* alloc_chimod() {                                   \
    return reinterpret_cast<clio::run::Container*>(new CONTAINER_CLASS()); \
  }                                                                  \
                                                                     \
  clio::run::Container* new_chimod(const clio::run::PoolId* pool_id,             \
                             const char* pool_name) {                \
    clio::run::Container* container =                                      \
        reinterpret_cast<clio::run::Container*>(new CONTAINER_CLASS());    \
    /* Initialization is handled by the container's Create method */ \
    return container;                                                \
  }                                                                  \
                                                                     \
  const char* get_chimod_name() { return MOD_NAME; }                 \
                                                                     \
  void destroy_chimod(clio::run::Container* container) {                   \
    delete reinterpret_cast<CONTAINER_CLASS*>(container);            \
  }                                                                  \
                                                                     \
  static bool is_clio_chimod_ = true;                            \
  }
// Backward-compat alias (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.

/**
 * Macro to define ChiMod entry points for task-based modules
 *
 * Usage: CLIO_TASK_CC(MyContainerClass)
 * This macro provides a cleaner interface for modules that use the Container
 * base class. The ChiMod name is automatically retrieved from
 * CONTAINER_CLASS::CreateParams::chimod_lib_name.
 */
#define CLIO_TASK_CC(CONTAINER_CLASS)                                \
  extern "C" {                                                       \
  clio::run::Container* alloc_chimod() {                                   \
    return reinterpret_cast<clio::run::Container*>(new CONTAINER_CLASS()); \
  }                                                                  \
                                                                     \
  clio::run::Container* new_chimod(const clio::run::PoolId* pool_id,             \
                             const char* pool_name) {                \
    auto* container = new CONTAINER_CLASS();                         \
    /* Initialization is handled by the container's Create method */ \
    return reinterpret_cast<clio::run::Container*>(container);             \
  }                                                                  \
                                                                     \
  const char* get_chimod_name() {                                    \
    return CONTAINER_CLASS::CreateParams::chimod_lib_name;           \
  }                                                                  \
                                                                     \
  void destroy_chimod(clio::run::Container* container) {                   \
    delete reinterpret_cast<CONTAINER_CLASS*>(container);            \
  }                                                                  \
                                                                     \
  static bool is_clio_chimod_ = true;                            \
  }
// Backward-compat alias (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.

#endif  // CLIO_RUNTIME_INCLUDE_CONTAINER_H_