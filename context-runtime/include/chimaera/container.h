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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_CONTAINER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_CONTAINER_H_

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "chimaera/pool_query.h"
#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "chimaera/local_task_archives.h"
#include "chimaera/types.h"

// Forward declarations to avoid circular dependencies
namespace chi {
class WorkOrchestrator;
}

/**
 * Container Base Class with Default Implementations
 *
 * Provides default implementations of ChiContainer methods for simpler modules.
 * Modules can inherit from this class instead of ChiContainer to get basic
 * queue and lane management functionality out of the box.
 */

namespace chi {

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

  PoolId pool_id_;         ///< The unique ID of this pool
  std::string pool_name_;  ///< The semantic name of this pool
  u32 container_id_;       ///< The logical ID of this container instance
  hshm::abitfield32_t flags_;  ///< Atomic bitfield for container state

 protected:
  PoolQuery pool_query_;

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
  }

  /** Mark container as plugged (no new tasks accepted) */
  void SetPlugged() { flags_.SetBits(CONTAINER_PLUG); }

  /** Check if container is plugged */
  bool IsPlugged() const { return flags_.Any(CONTAINER_PLUG) != 0; }

  /**
   * Execute a method on a task - must be implemented by derived classes
   *
   * This method returns TaskResume to support C++20 coroutine-based execution.
   * The returned TaskResume holds the coroutine handle that the worker uses
   * to suspend and resume task execution.
   */
  virtual TaskResume Run(u32 method, hipc::FullPtr<Task> task_ptr,
                         RunContext& rctx) = 0;

  /**
   * Delete/cleanup a task - must be implemented by derived classes
   */
  virtual void DelTask(u32 method, hipc::FullPtr<Task> task_ptr) = 0;

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
  virtual void UpdateWork(hipc::FullPtr<Task> task_ptr, RunContext& rctx,
                          i64 increment) {
    // Default: no work tracking
    (void)task_ptr;
    (void)rctx;
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
   * Serialize task parameters for network transfer (unified method)
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate serialization
   * @param method The method ID to serialize
   * @param archive SaveTaskArchive configured with srl_mode (true=In, false=Out)
   * @param task_ptr Full pointer to the task to serialize
   */
  virtual void SaveTask(u32 method, SaveTaskArchive& archive,
                        hipc::FullPtr<Task> task_ptr) = 0;

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
                        hipc::FullPtr<Task> task_ptr) = 0;

  /**
   * Allocate and deserialize task parameters from network transfer
   * Wrapper that calls NewTask followed by LoadTask
   * @param method The method ID to deserialize
   * @param archive LoadTaskArchive configured with srl_mode (true=In, false=Out)
   * @return Full pointer to the newly allocated and deserialized task
   */
  virtual hipc::FullPtr<Task> AllocLoadTask(u32 method, LoadTaskArchive& archive) = 0;

  /**
   * Deserialize task input parameters into an existing task using LocalSerialize
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate deserialization
   * Does not allocate - assumes task_ptr is already allocated
   * @param method The method ID to deserialize
   * @param archive LocalLoadTaskArchive for deserializing inputs
   * @param task_ptr Full pointer to the pre-allocated task to load into
   */
  virtual void LocalLoadTask(u32 method, LocalLoadTaskArchive& archive,
                             hipc::FullPtr<Task> task_ptr) = 0;

  /**
   * Allocate and deserialize task input parameters using LocalSerialize
   * Wrapper that calls NewTask followed by LocalLoadTask
   * @param method The method ID to deserialize
   * @param archive LocalLoadTaskArchive for deserializing inputs
   * @return Full pointer to the newly allocated and loaded task
   */
  virtual hipc::FullPtr<Task> LocalAllocLoadTask(u32 method, LocalLoadTaskArchive& archive) = 0;

  /**
   * Serialize task output parameters using LocalSerialize (for local transfers)
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate serialization
   * @param method The method ID to serialize
   * @param archive LocalSaveTaskArchive for serializing outputs
   * @param task_ptr Full pointer to the task to save outputs from
   */
  virtual void LocalSaveTask(u32 method, LocalSaveTaskArchive& archive,
                              hipc::FullPtr<Task> task_ptr) = 0;

  /**
   * Create a new copy of a task (deep copy for distributed execution) - must be
   * implemented by derived classes Uses switch-case structure based on method
   * ID to dispatch to appropriate task type copying
   * @param method The method ID for the task type
   * @param orig_task_ptr Full pointer to the original task
   * @param deep Whether to perform a deep copy
   * @return Full pointer to the newly created copy
   */
  HSHM_DLL virtual hipc::FullPtr<Task> NewCopyTask(u32 method,
                                                    hipc::FullPtr<Task> orig_task_ptr,
                                                    bool deep) = 0;

  /**
   * Create a new task of the specified method type
   * Must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate task type allocation
   * @param method The method ID for the task type to create
   * @return Full pointer to the newly allocated task (cast to base Task type)
   */
  HSHM_DLL virtual hipc::FullPtr<Task> NewTask(u32 method) = 0;

  /**
   * Aggregate a replica task into the origin task - must be implemented by derived classes
   * Uses switch-case structure based on method ID to dispatch to appropriate task type aggregation
   * This is used for merging replica results back into the origin task after distributed execution
   * @param method The method ID for the task type
   * @param origin_task_ptr Full pointer to the origin task to aggregate into
   * @param replica_task_ptr Full pointer to the replica task to aggregate from
   */
  HSHM_DLL virtual void Aggregate(u32 method,
                                   hipc::FullPtr<Task> origin_task_ptr,
                                   hipc::FullPtr<Task> replica_task_ptr) = 0; 
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
  ContainerClient() : pool_id_(), return_code_(0) {}

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
  hipc::FullPtr<TaskT> AllocateTask(MemorySegment segment, Args&&... args);
};

}  // namespace chi

/**
 * ChiMod Entry Point Macros
 *
 * These macros must be used in the runtime implementation file to
 * export the required C symbols for dynamic loading.
 */

extern "C" {
// Required ChiMod entry points
typedef chi::Container* (*alloc_chimod_t)();
typedef chi::Container* (*new_chimod_t)(const chi::PoolId* pool_id,
                                        const char* pool_name);
typedef const char* (*get_chimod_name_t)(void);
typedef void (*destroy_chimod_t)(chi::Container* container);
}

/**
 * Macro to define ChiMod entry points in runtime source file (deprecated)
 *
 * Usage: CHI_CHIMOD_CC(MyContainerClass, "my_chimod_name")
 * Note: Use CHI_TASK_CC instead for new modules
 */
#define CHI_CHIMOD_CC(CONTAINER_CLASS, MOD_NAME)                     \
  extern "C" {                                                       \
  chi::Container* alloc_chimod() {                                   \
    return reinterpret_cast<chi::Container*>(new CONTAINER_CLASS()); \
  }                                                                  \
                                                                     \
  chi::Container* new_chimod(const chi::PoolId* pool_id,             \
                             const char* pool_name) {                \
    chi::Container* container =                                      \
        reinterpret_cast<chi::Container*>(new CONTAINER_CLASS());    \
    /* Initialization is handled by the container's Create method */ \
    return container;                                                \
  }                                                                  \
                                                                     \
  const char* get_chimod_name() { return MOD_NAME; }                 \
                                                                     \
  void destroy_chimod(chi::Container* container) {                   \
    delete reinterpret_cast<CONTAINER_CLASS*>(container);            \
  }                                                                  \
                                                                     \
  static bool is_chimaera_chimod_ = true;                            \
  }

/**
 * Macro to define ChiMod entry points for task-based modules
 *
 * Usage: CHI_TASK_CC(MyContainerClass)
 * This macro provides a cleaner interface for modules that use the Container
 * base class. The ChiMod name is automatically retrieved from
 * CONTAINER_CLASS::CreateParams::chimod_lib_name.
 */
#define CHI_TASK_CC(CONTAINER_CLASS)                                 \
  extern "C" {                                                       \
  chi::Container* alloc_chimod() {                                   \
    return reinterpret_cast<chi::Container*>(new CONTAINER_CLASS()); \
  }                                                                  \
                                                                     \
  chi::Container* new_chimod(const chi::PoolId* pool_id,             \
                             const char* pool_name) {                \
    auto* container = new CONTAINER_CLASS();                         \
    /* Initialization is handled by the container's Create method */ \
    return reinterpret_cast<chi::Container*>(container);             \
  }                                                                  \
                                                                     \
  const char* get_chimod_name() {                                    \
    return CONTAINER_CLASS::CreateParams::chimod_lib_name;           \
  }                                                                  \
                                                                     \
  void destroy_chimod(chi::Container* container) {                   \
    delete reinterpret_cast<CONTAINER_CLASS*>(container);            \
  }                                                                  \
                                                                     \
  static bool is_chimaera_chimod_ = true;                            \
  }

#endif  // CHIMAERA_INCLUDE_CHIMAERA_CONTAINER_H_