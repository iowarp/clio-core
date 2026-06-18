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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_POOL_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_POOL_MANAGER_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <atomic>
#include <shared_mutex>
#include "clio_runtime/types.h"

namespace clio::run {

// Forward declarations for ChiMod system
// Container is always a class forward declaration (defined in container.h)
class Container;
class Task;
struct RunContext;
class TaskResume;

/**
 * Pool metadata containing container references and address mappings
 */
struct PoolInfo {
  PoolId pool_id_;
  std::string pool_name_;
  std::string chimod_name_;
  std::string chimod_params_;
  u32 num_containers_;
  bool is_active_;

  /** Containers on THIS node (ContainerId -> Container*) */
  std::unordered_map<ContainerId, Container*> containers_;
  /** ALL container address mappings across cluster (ContainerId -> NodeId) */
  std::unordered_map<ContainerId, u32> address_map_;
  /** Static container for stateless APIs (alloc, serialize, deserialize tasks) */
  Container* static_container_ = nullptr;
  /** Local (default) container for this node. Initially static_container_.
      When migrated away, another from containers_ is chosen. If none, falls back to static. */
  Container* local_container_ = nullptr;
  /** GPU device pointer to gpu::Container (nullptr if no GPU companion) */
  void* gpu_container_ptr_ = nullptr;

  PoolInfo() : pool_id_(), num_containers_(0), is_active_(false) {}

  PoolInfo(PoolId pool_id, const std::string& pool_name,
           const std::string& chimod_name, const std::string& chimod_params,
           u32 num_containers)
      : pool_id_(pool_id), pool_name_(pool_name), chimod_name_(chimod_name),
        chimod_params_(chimod_params), num_containers_(num_containers), is_active_(true) {}

  /** Select a new local_container_ after migration/removal */
  void RecalculateLocalContainer() {
    if (!containers_.empty()) {
      local_container_ = containers_.begin()->second;
    } else {
      local_container_ = static_container_;
    }
  }
};

/**
 * Pool Manager singleton for managing ChiPools and Containers
 * 
 * Maps PoolId to Containers on this node and manages the lifecycle
 * of pools in the distributed system.
 * Uses CTP global cross pointer variable singleton pattern.
 */
class PoolManager {
 public:
  /**
   * Initialize pool manager (server/runtime mode)  
   * Full initialization for pool management and creates admin chimod pool
   * @return true if initialization successful, false otherwise
   */
  bool ServerInit();

  /**
   * Finalize and cleanup pool resources
   */
  void Finalize();

  /**
   * Destroy (delete) every registered container via its ChiMod's destroy_func,
   * running each container's C++ destructor so its runtime-heap data is freed
   * (CTE's metadata maps, bdev's RAM pages + file descriptors, the container
   * object itself). Used on graceful shutdown to release this memory instead of
   * leaking it until process exit. Must be called while the ChiMod shared
   * libraries are still loaded (before ModuleManager::Finalize()) and after the
   * workers are stopped (no concurrent container access).
   */
  void DestroyAllContainers();

  /**
   * Register a Container with a specific PoolId and ContainerId
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @param container Pointer to Container
   * @param is_static Whether this is the static container for the pool
   * @return true if registration successful, false otherwise
   */
  bool RegisterContainer(PoolId pool_id, ContainerId container_id,
                          Container* container, bool is_static = false);

  /**
   * Unregister a specific Container
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @return true if unregistration successful, false otherwise
   */
  bool UnregisterContainer(PoolId pool_id, ContainerId container_id);

  /**
   * Unregister all containers for a pool
   * @param pool_id Pool identifier
   */
  void UnregisterAllContainers(PoolId pool_id);

  /**
   * Plug a container: mark CONTAINER_PLUG and wait for all work to complete
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   */
  void PlugContainer(PoolId pool_id, ContainerId container_id);

  /**
   * Get Container by PoolId and ContainerId, with plug state
   * If container_id is kInvalidContainerId, falls back to local container.
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @param is_plugged Output: true if the container is plugged
   * @return Pointer to Container or nullptr if not found
   */
  Container* GetContainer(PoolId pool_id, ContainerId container_id,
                           bool &is_plugged) const;

  /**
   * Get the static container for a pool (for stateless ops: alloc, serialize, etc.)
   * @param pool_id Pool identifier
   * @return Pointer to static Container or nullptr if not found
   */
  Container* GetStaticContainer(PoolId pool_id) const;

  /**
   * Check if pool exists on this node
   * @param pool_id Pool identifier
   * @return true if pool exists locally, false otherwise
   */
  bool HasPool(PoolId pool_id) const;

  /**
   * Check if a specific container exists on this node for a given pool
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @return true if the container exists locally, false otherwise
   */
  bool HasContainer(PoolId pool_id, ContainerId container_id) const;

  /**
   * Find pool by name (globally unique)
   * @param pool_name Pool name
   * @return PoolId if found, PoolId::GetNull() if not found
   */
  PoolId FindPoolByName(const std::string& pool_name) const;

  /**
   * Get number of registered pools
   * @return Count of registered pools on this node
   */
  size_t GetPoolCount() const;

  /**
   * Get all registered pool IDs
   * @return Vector of PoolId values for all registered pools
   */
  std::vector<PoolId> GetAllPoolIds() const;

  /**
   * Generate a new unique pool ID
   * @return New pool ID
   */
  PoolId GeneratePoolId();

  /**
   * Validate pool creation parameters
   * @param chimod_name ChiMod name
   * @param pool_name Pool name  
   * @return true if parameters are valid, false otherwise
   */
  bool ValidatePoolParams(const std::string& chimod_name, const std::string& pool_name);

  /**
   * Initialize address map for a pool (ContainerId -> NodeId)
   * @param pool_id Pool identifier
   * @param num_containers Number of containers in the pool
   */
  void InitAddressMap(PoolId pool_id, u32 num_containers);

  /**
   * Create or get a complete pool with get-or-create semantics
   * Extracts all parameters from the task (chimod_name, pool_name, chimod_params)
   * This is a coroutine that can co_await nested Create methods
   * @param task Task containing pool creation parameters (updated with final pool ID)
   * @param run_ctx RunContext for container initialization
   * @return TaskResume coroutine handle
   */
  TaskResume CreatePool(FullPtr<Task> task, RunContext* run_ctx);


  /**
   * Destroy a complete pool including metadata and local containers
   * This is a coroutine for consistency with CreatePool
   * @param pool_id Pool identifier
   * @return TaskResume coroutine handle
   */
  TaskResume DestroyPool(PoolId pool_id);

  /**
   * Destroy a local pool and its containers on this node (simple version)
   * @param pool_id Pool identifier
   * @return true if pool destruction successful, false otherwise
   */
  bool DestroyLocalPool(PoolId pool_id);

  /**
   * Get pool information
   * @param pool_id Pool identifier
   * @return Pointer to PoolInfo or nullptr if not found
   */
  const PoolInfo* GetPoolInfo(PoolId pool_id) const;

  /**
   * Update pool metadata
   * @param pool_id Pool identifier
   * @param info Pool information to store
   */
  void UpdatePoolMetadata(PoolId pool_id, const PoolInfo& info);

  /**
   * Check if pool manager is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const;

  /**
   * Get physical node ID for a container in a pool
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @return Physical node ID, or 0 if not found or local node
   */
  u32 GetContainerNodeId(PoolId pool_id, ContainerId container_id) const;

  /**
   * Update the global->physical mapping for a container in a pool's address table
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @param new_node_id New physical node ID for the container
   * @return true if mapping was updated, false if pool not found
   */
  bool UpdateContainerNodeMapping(PoolId pool_id, ContainerId container_id, u32 new_node_id);

  /**
   * Write a WAL entry for an address table change
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @param old_node Previous node ID
   * @param new_node New node ID
   */
  void WriteAddressTableWAL(PoolId pool_id, ContainerId container_id, u32 old_node, u32 new_node);

  /**
   * Replay WAL entries to recover address table state after restart
   * Reads all .bin files from conf_dir/wal/ and applies each entry's
   * container-to-node mapping using UpdateContainerNodeMapping.
   */
  void ReplayAddressTableWAL();

 private:
  /**
   * Internal: Get Container by PoolId and ContainerId (no plug check)
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @return Pointer to Container or nullptr if not found
   */
  Container* GetContainerRaw(PoolId pool_id, ContainerId container_id) const;

  /**
   * Internal: erase a pool's metadata entry under the write lock. Used by the
   * CreatePool error paths and DestroyPool so they don't touch pool_metadata_
   * without holding pool_metadata_mutex_.
   * @param pool_id Pool identifier
   */
  void ErasePoolMetadata(PoolId pool_id);

  bool is_initialized_ = false;

  // Map PoolId to pool metadata (contains containers, address map, etc.)
  std::unordered_map<PoolId, PoolInfo> pool_metadata_;

  // Guards pool_metadata_ (and the PoolInfo contents reachable through it)
  // against concurrent access by worker threads. Lookups on the hot task-
  // routing path (GetStaticContainer/GetContainer via IpcManager::RouteTask)
  // run on workers while CreatePool/UpdatePoolMetadata insert (and rehash) the
  // map from another worker; without this lock a concurrent rehash tears the
  // bucket array and find() dereferences null (issue #572). Readers take a
  // shared lock, structural/PoolInfo mutators take an exclusive lock. Locks are
  // always scoped to a single map operation so the lock is never held across
  // CreatePool's co_await.
  mutable std::shared_mutex pool_metadata_mutex_;

  // Pool ID counter for generating unique IDs (used as minor number)
  std::atomic<u32> next_pool_minor_{5}; // Start at 5 for safety, 1 reserved for admin

};

}  // namespace clio::run

// Global pointer variable declaration for Pool manager singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_H(chi::PoolManager, g_pool_manager);

// Macro for accessing the Pool manager singleton using global pointer variable
#define CLIO_POOL_MANAGER CTP_GET_GLOBAL_PTR_VAR(::chi::PoolManager, g_pool_manager)
// Backward-compat alias (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.
#define CHI_POOL_MANAGER  CLIO_POOL_MANAGER

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_POOL_MANAGER_H_