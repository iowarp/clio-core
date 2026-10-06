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

#ifndef CLIO_RUNTIME_INCLUDE_MANAGERS_POOL_MANAGER_H_
#define CLIO_RUNTIME_INCLUDE_MANAGERS_POOL_MANAGER_H_

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include "clio_runtime/types.h"
#include "clio_runtime/dynamic_container.h"

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

  /** Containers on THIS node (ContainerId -> DynamicContainer). Each entry is
      a by-value handle sharing one ContainerPtr; copies handed out by
      GetContainer stay valid even if the entry is later erased. */
  std::unordered_map<ContainerId, DynamicContainer> containers_;
  /** ALL container address mappings across cluster (ContainerId -> NodeId) */
  std::unordered_map<ContainerId, u32> address_map_;
  /** Static container for stateless APIs (alloc, serialize, deserialize,
      ScheduleTask). Created once per pool per node at pool-creation time
      (issue #956); it is deliberately NOT a member of containers_: the
      module's Create method never runs on it, no task is ever routed to it,
      and it therefore holds no module state. It also owns no learned state:
      the task-stat model is per real container (issue #994), so the static
      container's own model table stays at its seed and nothing reads it. */
  DynamicContainer static_container_;
  /** Local (default) container for this node. Initially static_container_.
      When migrated away, another from containers_ is chosen. If none, falls back to static. */
  DynamicContainer local_container_;
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
   * Register a Container with a specific PoolId and ContainerId.
   *
   * The container must already be Init'd (its method-name table populated):
   * before publishing it, this restores the task-stat model that THIS
   * container id previously learned on this node (issue #994), so a restarted
   * runtime schedules with what the last one learned. Nothing is shared with
   * the pool's other containers.
   *
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @param container Pointer to Container
   * @return true if registration successful, false otherwise
   */
  bool RegisterContainer(PoolId pool_id, ContainerId container_id,
                          DynamicContainer container);

  /**
   * Get (creating it if absent) the pool's static container — the stateless
   * per-pool container used for alloc/serialize/deserialize/ScheduleTask.
   * Called by CreatePool and by RegisterContainer, so a container created by a
   * later path (recovery, migration) still finds one.
   *
   * @param pool_id Pool identifier
   * @return handle to the static container (invalid if the pool is unknown or
   *         the ChiMod could not be instantiated)
   */
  DynamicContainer EnsureStaticContainer(PoolId pool_id);

  /**
   * Persist the task-stat model of every registered container whose weights
   * changed since the last save. Called periodically (admin SystemMonitor) and
   * unconditionally on shutdown, so a `kill -9` costs at most one flush
   * interval of learning.
   * @param force write even if not dirty and regardless of the flush interval
   */
  void FlushModels(bool force = false);

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
   * Get a DynamicContainer handle by PoolId and ContainerId.
   * If container_id is kInvalidContainerId, falls back to local container.
   * Callers check the result with DynamicContainer::IsValid()/IsPlugged().
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @return DynamicContainer handle (invalid if not found)
   */
  DynamicContainer GetContainer(PoolId pool_id, ContainerId container_id) const;

  /**
   * Get the static container for a pool (for stateless ops: alloc, serialize,
   * deserialize). Returns the by-value DynamicContainer handle; stateless
   * callers deref it via operator-> / current().
   * @param pool_id Pool identifier
   * @return DynamicContainer handle to the static container (invalid if not found)
   */
  DynamicContainer GetStaticContainer(PoolId pool_id) const;

  /**
   * Get the real (local) container for a pool if this node hosts one, falling
   * back to the static container otherwise.
   * @param pool_id Pool identifier
   * @return DynamicContainer handle (invalid if the pool is unknown)
   */
  DynamicContainer GetRealOrStaticContainer(PoolId pool_id) const;

  /**
   * Check if pool exists on this node
   * @param pool_id Pool identifier
   * @return true if pool exists locally, false otherwise
   */
  bool HasPool(PoolId pool_id) const;

  /**
   * Whether this node destroyed `pool_id` and has not re-created it since.
   * Routing retires a periodic task of such a pool instead of retrying it
   * forever.
   * @param pool_id Pool identifier
   * @return true if the pool was destroyed here and not re-created
   */
  bool WasDestroyed(PoolId pool_id) const;

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
   * Get every REAL container of a pool hosted on this node (the static
   * container is not included). Handles are copied out under the read lock,
   * so the caller may use them without holding pool_metadata_mutex_.
   * @param pool_id Pool identifier
   * @return by-value container handles, sorted by container id (empty if the
   *         pool is unknown or hosts no container here)
   */
  std::vector<DynamicContainer> GetLocalContainers(PoolId pool_id) const;

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
   * Make a pool that lives on OTHER nodes routable from this one, without
   * creating a container here.
   *
   * A pool composed on a single node (e.g. a per-node bdev created with
   * PoolQuery::Physical) has metadata only on that node. Any other node that
   * submits a task to it needs the pool's static container (to serialize the
   * task) and its address map (to resolve DirectHash/Physical routing); without
   * them SendIn drops the task and its waiter hangs. This installs exactly
   * those two things: metadata, a ContainerId == NodeId address map, and the
   * static container. No module Create runs and no task executes here.
   *
   * Does nothing if this node already knows the pool (a real local pool is
   * never overwritten).
   *
   * @param pool_id Pool identifier of the remote pool
   * @param pool_name Pool name (as created on the owning node)
   * @param chimod_name ChiMod library name the pool was created with
   * @param chimod_params Serialized ChiMod create parameters
   * @param num_containers Number of containers in the pool's address map
   * @return true if the pool is routable from this node afterwards
   */
  bool RegisterRemotePool(PoolId pool_id, const std::string& pool_name,
                          const std::string& chimod_name,
                          const std::string& chimod_params,
                          u32 num_containers);

  /**
   * Create or get a complete pool with get-or-create semantics
   * Extracts all parameters from the task (chimod_name, pool_name, chimod_params)
   * This is a coroutine that can co_await nested Create methods
   * @param task Task containing pool creation parameters (updated with final pool ID)
   * @return TaskResume coroutine handle
   */
  TaskResume CreatePool(clio::run::shared_ptr<Task> &task);

  /**
   * Record that a pool create has started (issue #1180). Paired with
   * EndCreate; CreatePool does both through a scope guard.
   * @param pool_name the pool being created
   * @param chimod_name its module
   * @return a handle for EndCreate
   */
  u64 BeginCreate(const std::string &pool_name, const std::string &chimod_name);

  /**
   * Record that a pool create has finished, however it ended.
   * @param handle the value BeginCreate returned
   */
  void EndCreate(u64 handle);

  /**
   * Describe every pool create still in progress, oldest first, for a stall
   * report: a stalled GetOrCreatePool otherwise names only its method, not
   * which pool it is creating (issue #1180). Safe from any thread.
   * @return e.g. "creating 'cte_main' (clio_cte_core) for 297321 ms", or ""
   */
  std::string DescribeCreatesInProgress() const;


  /**
   * Destroy a complete pool including metadata and local containers
   * This is a coroutine for consistency with CreatePool
   * @param pool_id Pool identifier
   * @param keep_in_pool_log true to leave the pool's pool-log entry, so the
   *        next start re-creates it (`compose stop`)
   * @return TaskResume coroutine handle
   */
  TaskResume DestroyPool(PoolId pool_id, bool keep_in_pool_log = false);

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

  /** One durable pool as this node's pool log keeps it: what re-creating
   *  its container here after a restart takes. */
  struct PoolLogEntry {
    PoolId pool_id;
    std::string pool_name;
    std::string chimod_name;
    /** Compose: the serialized PoolConfig. API: the ChiMod's serialized
     *  CreateParams. */
    std::string chimod_params;
    bool compose = false;
  };

  /**
   * Record (add) or forget (!add) a durable pool in this node's pool log,
   * <conf_dir>/wal/pools.<node>.bin -- the ONE restart registry for every
   * pool, however it was created: compose pools with `restart: true` and
   * API pools created by a client with SetPersistent(true).
   * @param add true on create, false on destroy
   * @param e the pool
   */
  void LogPool(bool add, const PoolLogEntry &e);

  /**
   * The live entries of this node's pool log, in creation order (a pool may
   * need one created before it, e.g. an array's member disks), compacting
   * the log to that set.
   * @return live entries
   */
  std::vector<PoolLogEntry> LoadPoolLog();

  /** Read the live entries of a pool log file without compacting it.
   *  @param path the log file  @return live entries */
  static std::vector<PoolLogEntry> ReadPoolLogFile(const std::string &path);

  /** Forget every durable pool of this node and its address-table WAL: a
   *  fresh (`start --fresh`) start begins a new cluster lifetime. */
  void ClearPoolLog();

  /** @return path of this node's pool log. */
  std::string PoolLogPath() const;

  /** While true, pools being created are re-creations from the pool log:
   *  their containers take the Restart() path and are not logged again. */
  void SetReplayingPools(bool v) { replaying_pools_ = v; }

 private:
  /** One pool create in progress (issue #1180). */
  struct CreateInProgress {
    std::string pool_name_;  /**< pool being created */
    std::string chimod_;     /**< its module */
    std::chrono::steady_clock::time_point start_;  /**< when it began */
  };
  mutable std::mutex creates_mu_;  /**< guards creates_ and next_create_ */
  std::unordered_map<u64, CreateInProgress> creates_;  /**< by handle */
  u64 next_create_ = 1;  /**< next BeginCreate handle */

  bool replaying_pools_ = false;
  /**
   * Internal: Get a DynamicContainer by PoolId and ContainerId (no fallback to
   * local container; no plug check)
   * @param pool_id Pool identifier
   * @param container_id Container identifier
   * @return DynamicContainer handle (invalid if not found)
   */
  DynamicContainer GetContainerRaw(PoolId pool_id, ContainerId container_id) const;

  /**
   * Internal: erase a pool's metadata entry under the write lock. Used by the
   * CreatePool error paths and DestroyPool so they don't touch pool_metadata_
   * without holding pool_metadata_mutex_.
   * @param pool_id Pool identifier
   */
  void ErasePoolMetadata(PoolId pool_id);

  /**
   * Internal: write one container's task-stat model to disk. The caller must
   * NOT hold pool_metadata_mutex_ — this does filesystem I/O.
   * @param chimod_name ChiMod owning the pool (part of the file name)
   * @param pool_name Pool name (part of the file name)
   * @param container The container whose model is saved (its container_id_
   *        is part of the file name)
   * @param force save even if no weight changed since the last save
   */
  void SaveModel(const std::string &chimod_name, const std::string &pool_name,
                 const DynamicContainer &container, bool force);

  /**
   * Internal: load the model file previously saved for `container` (matched
   * by container_id_ on this node) into it. A missing file is the normal
   * first-run case and leaves the seeded table alone. The caller must NOT hold
   * pool_metadata_mutex_ — this reads a file.
   * @param chimod_name ChiMod owning the pool (part of the file name)
   * @param pool_name Pool name (part of the file name)
   * @param container The (already Init'd) container to restore into
   */
  void RestoreModel(const std::string &chimod_name,
                    const std::string &pool_name,
                    const DynamicContainer &container);

  bool is_initialized_ = false;

  // Wall time of the last FlushModels() write, so the periodic flush driven by
  // the admin SystemMonitor (1 Hz) does not rewrite every pool's model file
  // once a second. Only touched from FlushModels under model_flush_mutex_.
  std::chrono::steady_clock::time_point last_model_flush_{};
  // Serializes model-file writers. Guards last_model_flush_ above AND the whole
  // export-emit-rename in SaveModel, which is the part that actually needs
  // exclusion: every writer renames its own "<path>.tmp" over the shared model
  // file. Never taken while pool_metadata_mutex_ is held.
  std::mutex model_flush_mutex_;

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
  /** Pools destroyed on this node and not re-created (see WasDestroyed). */
  std::unordered_set<PoolId> destroyed_pools_;
  mutable std::mutex destroyed_pools_mu_;

  // Pool ID counter for generating unique IDs (used as minor number)
  std::atomic<u32> next_pool_minor_{5}; // Start at 5 for safety, 1 reserved for admin

};

}  // namespace clio::run

// Global pointer variable declaration for Pool manager singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_H(clio::run::PoolManager, g_pool_manager);

// Macro for accessing the Pool manager singleton using global pointer variable
#define CLIO_POOL_MANAGER CTP_GET_GLOBAL_PTR_VAR(::clio::run::PoolManager, g_pool_manager)
// Backward-compat alias (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.

#endif  // CLIO_RUNTIME_INCLUDE_MANAGERS_POOL_MANAGER_H_