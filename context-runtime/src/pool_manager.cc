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

/**
 * Pool manager implementation
 */

#include "clio_runtime/pool_manager.h"

#include "clio_runtime/admin/admin_tasks.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/container.h"
#include "clio_runtime/module_manager.h"
#include "clio_runtime/task.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <shared_mutex>

// Global pointer variable definition for Pool manager singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_CC(clio::run::PoolManager, g_pool_manager);

namespace clio::run {

// Convenience aliases for the pool_metadata_ reader/writer lock (issue #572).
// Readers (lookups) take the shared lock; structural and PoolInfo mutators take
// the exclusive lock.
using PoolMetaReadLock = std::shared_lock<std::shared_mutex>;
using PoolMetaWriteLock = std::unique_lock<std::shared_mutex>;

// Constructor and destructor removed - handled by CTP singleton pattern

bool PoolManager::ServerInit() {
  if (is_initialized_) {
    return true;
  }

  // Initialize pool metadata
  {
    PoolMetaWriteLock lock(pool_metadata_mutex_);
    pool_metadata_.clear();
  }

  is_initialized_ = true;

  // Create the admin chimod pool (kAdminPoolId = 1)
  // This is required for flush operations and other admin tasks
  PoolId admin_pool_id;

  // Create proper admin task and RunContext for pool creation
  auto* ipc_manager = CLIO_IPC;
  if (!ipc_manager) {
    HLOG(kError, "PoolManager: IPC manager not available during ServerInit");
    return false;
  }

  auto admin_task = ipc_manager->NewTask<clio::run::admin::CreateTask>(
      CreateTaskId(),
      kAdminPoolId,  // Use admin pool for admin container creation
      PoolQuery::Local(), "clio_admin", "admin", kAdminPoolId,
      nullptr);  // No client for internal admin pool creation

  RunContext run_ctx;

  // CreatePool is now a coroutine - we need to run it to completion
  // For admin pool creation during ServerInit, the coroutine won't yield
  // (admin Create doesn't co_await anything), so we can run it synchronously
  TaskResume task_resume = CreatePool(admin_task.Cast<Task>(), &run_ctx);
  auto handle = task_resume.release();
  if (handle) {
    // Run the coroutine to completion
    handle.resume();
    // For admin Create, it should complete immediately (no yields)
    if (!handle.done()) {
      HLOG(kError, "PoolManager: Admin pool creation coroutine didn't complete");
      handle.destroy();
      ipc_manager->DelTask(admin_task);
      return false;
    }
    handle.destroy();
  }

  // Check if pool creation succeeded by examining the task return code
  if (admin_task->GetReturnCode() != 0) {
    // Cleanup the task we created
    ipc_manager->DelTask(admin_task);
    HLOG(kError,
         "PoolManager: Failed to create admin chimod pool during ServerInit");
    return false;
  }

  // Get the pool ID from the updated task
  admin_pool_id = admin_task->new_pool_id_;

  // Cleanup the task after successful pool creation
  ipc_manager->DelTask(admin_task);

  HLOG(kInfo,
       "PoolManager: Admin chimod pool created successfully with PoolId {}",
       admin_pool_id);
  return true;
}

void PoolManager::DestroyAllContainers() {
  if (!is_initialized_) {
    return;
  }
  auto *module_manager = CLIO_MODULE_MANAGER;
  if (!module_manager) {
    return;
  }
  // Delete every container via its ChiMod's destroy_func, running ~Runtime() so
  // the module frees its own runtime-heap data: CTE's metadata maps (members),
  // bdev's RAM pages + file descriptors (~Runtime / CleanupWorkerIOContexts),
  // and the container object itself. Without this the container state leaks
  // until process exit — the leaks papered over by CI/lsan_suppressions.txt
  // (new_chimod, Container::Init, the CTE/bdev module allocations, etc.).
  //
  // This runs the C++ destructor only. It does NOT invoke the ChiMod Destroy
  // *task method* (e.g. CTE's explicit WAL flush + map clear): that is a
  // coroutine and must be executed by a worker with a fully-initialized
  // RunContext — driving it inline during finalize jumps through uninitialized
  // coroutine continuation state and crashes. Running the Destroy method on
  // shutdown (route it through the workers before StopWorkers) is tracked as a
  // follow-up in #563.
  PoolMetaWriteLock lock(pool_metadata_mutex_);  // #572 locking discipline
  size_t destroyed = 0;
  for (auto &pair : pool_metadata_) {
    PoolInfo &info = pair.second;
    for (auto &cpair : info.containers_) {
      if (cpair.second) {
        module_manager->DestroyContainer(info.chimod_name_, cpair.second);
        ++destroyed;
      }
    }
    info.containers_.clear();
    info.static_container_ = nullptr;
    info.local_container_ = nullptr;
  }
  if (destroyed > 0) {
    HLOG(kInfo, "PoolManager: Destroyed {} container(s) on shutdown", destroyed);
  }
}

void PoolManager::Finalize() {
  if (!is_initialized_) {
    return;
  }

  // Clear all containers in each PoolInfo, then clear metadata. Container
  // objects are deleted earlier by DestroyAllContainers() (while ChiMod
  // libraries are still loaded), so by the time Finalize() runs the maps may
  // already be empty; this is a metadata-only clear under the #572 write lock.
  {
    PoolMetaWriteLock lock(pool_metadata_mutex_);
    for (auto &pair : pool_metadata_) {
      pair.second.containers_.clear();
      pair.second.static_container_ = nullptr;
      pair.second.local_container_ = nullptr;
    }
    pool_metadata_.clear();
  }

  is_initialized_ = false;
}

bool PoolManager::RegisterContainer(PoolId pool_id, ContainerId container_id,
                                     Container* container, bool is_static) {
  if (!is_initialized_ || container == nullptr) {
    return false;
  }

  PoolMetaWriteLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    return false;
  }

  PoolInfo &info = it->second;
  info.containers_[container_id] = container;

  if (is_static || info.static_container_ == nullptr) {
    info.static_container_ = container;
  }
  if (info.local_container_ == nullptr) {
    info.local_container_ = container;
  }

  return true;
}

bool PoolManager::UnregisterContainer(PoolId pool_id, ContainerId container_id) {
  if (!is_initialized_) {
    return false;
  }

  PoolMetaWriteLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    return false;
  }

  PoolInfo &info = it->second;
  auto cit = info.containers_.find(container_id);
  if (cit == info.containers_.end()) {
    return false;
  }

  Container *removed = cit->second;
  info.containers_.erase(cit);

  // static_container_ is never modified after pool creation — it is a
  // persistent reference used for stateless operations (task deserialization).
  if (info.local_container_ == removed) {
    info.RecalculateLocalContainer();
  }

  return true;
}

void PoolManager::UnregisterAllContainers(PoolId pool_id) {
  if (!is_initialized_) {
    return;
  }

  PoolMetaWriteLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    return;
  }

  PoolInfo &info = it->second;
  info.containers_.clear();
  info.static_container_ = nullptr;
  info.local_container_ = nullptr;
}

Container* PoolManager::GetContainer(PoolId pool_id, ContainerId container_id,
                                      bool &is_plugged) const {
  is_plugged = false;
  if (!is_initialized_) {
    return nullptr;
  }

  PoolMetaReadLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    return nullptr;
  }

  const PoolInfo &info = it->second;
  Container *container = nullptr;
  if (container_id != kInvalidContainerId) {
    auto cit = info.containers_.find(container_id);
    if (cit != info.containers_.end()) {
      container = cit->second;
    }
  }
  if (!container) {
    container = info.local_container_;
  }
  if (container) {
    is_plugged = container->IsPlugged();
  }
  return container;
}

Container* PoolManager::GetContainerRaw(PoolId pool_id, ContainerId container_id) const {
  if (!is_initialized_) {
    return nullptr;
  }

  PoolMetaReadLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    return nullptr;
  }

  const PoolInfo &info = it->second;
  auto cit = info.containers_.find(container_id);
  return (cit != info.containers_.end()) ? cit->second : nullptr;
}

Container* PoolManager::GetStaticContainer(PoolId pool_id) const {
  if (!is_initialized_) {
    return nullptr;
  }

  PoolMetaReadLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    return nullptr;
  }

  return it->second.static_container_;
}

void PoolManager::PlugContainer(PoolId pool_id, ContainerId container_id) {
  Container *container = GetContainerRaw(pool_id, container_id);
  if (!container) {
    return;
  }
  container->SetPlugged();
  while (container->GetWorkRemaining() > 0) {
    CTP_THREAD_MODEL->Yield();
  }
}

bool PoolManager::HasPool(PoolId pool_id) const {
  if (!is_initialized_) {
    return false;
  }

  PoolMetaReadLock lock(pool_metadata_mutex_);
  return pool_metadata_.find(pool_id) != pool_metadata_.end();
}

bool PoolManager::HasContainer(PoolId pool_id, ContainerId container_id) const {
  if (!is_initialized_) {
    return false;
  }

  PoolMetaReadLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    return false;
  }

  return it->second.containers_.find(container_id) != it->second.containers_.end();
}

PoolId PoolManager::FindPoolByName(const std::string& pool_name) const {
  if (!is_initialized_) {
    return PoolId::GetNull();
  }

  PoolMetaReadLock lock(pool_metadata_mutex_);
  // Iterate through pool metadata to find matching pool_name (globally unique)
  for (const auto& pair : pool_metadata_) {
    const PoolInfo& pool_info = pair.second;
    if (pool_info.pool_name_ == pool_name) {
      return pair.first;  // Return the PoolId
    }
  }

  return PoolId::GetNull();  // Not found
}

size_t PoolManager::GetPoolCount() const {
  if (!is_initialized_) {
    return 0;
  }
  PoolMetaReadLock lock(pool_metadata_mutex_);
  return pool_metadata_.size();
}

std::vector<PoolId> PoolManager::GetAllPoolIds() const {
  std::vector<PoolId> pool_ids;
  if (!is_initialized_) {
    return pool_ids;
  }

  PoolMetaReadLock lock(pool_metadata_mutex_);
  pool_ids.reserve(pool_metadata_.size());
  for (const auto& pair : pool_metadata_) {
    pool_ids.push_back(pair.first);
  }
  return pool_ids;
}

bool PoolManager::IsInitialized() const { return is_initialized_; }

bool PoolManager::DestroyLocalPool(PoolId pool_id) {
  if (!is_initialized_) {
    HLOG(kError, "PoolManager: Not initialized for pool destruction");
    return false;
  }

  // Check if pool exists
  if (!HasPool(pool_id)) {
    HLOG(kError, "PoolManager: Pool {} not found on this node", pool_id);
    return false;
  }

  try {
    // Unregister all containers for this pool
    UnregisterAllContainers(pool_id);

    HLOG(kInfo, "PoolManager: Destroyed local pool {}", pool_id);
    return true;

  } catch (const std::exception& e) {
    HLOG(kError, "PoolManager: Exception during local pool destruction: {}",
         e.what());
    return false;
  }
}

PoolId PoolManager::GeneratePoolId() {
  if (!is_initialized_) {
    return PoolId::GetNull();
  }

  // Use atomic fetch_add to get unique minor number, then construct PoolId
  u32 minor = next_pool_minor_.fetch_add(1);
  auto* ipc_manager = CLIO_IPC;
  u32 major = ipc_manager->GetNodeId();  // Use this node's ID as major number
  return PoolId(major, minor);
}

bool PoolManager::ValidatePoolParams(const std::string& chimod_name,
                                     const std::string& pool_name) {
  if (!is_initialized_) {
    return false;
  }

  // Check for empty or invalid names
  if (chimod_name.empty() || pool_name.empty()) {
    HLOG(kError, "PoolManager: ChiMod name and pool name cannot be empty");
    return false;
  }

  // Check if the ChiMod exists
  auto* module_manager = CLIO_MODULE_MANAGER;
  if (!module_manager) {
    HLOG(kError, "PoolManager: Module manager not available for validation");
    return false;
  }

  auto* chimod = module_manager->GetChiMod(chimod_name);
  if (!chimod) {
    HLOG(kError, "PoolManager: ChiMod '{}' not found", chimod_name);
    return false;
  }

  return true;
}

void PoolManager::InitAddressMap(PoolId pool_id, u32 num_containers) {
  if (!is_initialized_) {
    return;
  }

  PoolMetaWriteLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    return;
  }

  PoolInfo &info = it->second;
  info.address_map_.clear();

  HLOG(kDebug, "=== Address Map for Pool {} ===", pool_id);
  HLOG(kDebug, "Creating address map with {} containers", num_containers);

  // Initially ContainerId == NodeId (one container per node)
  for (u32 container_idx = 0; container_idx < num_containers; ++container_idx) {
    info.address_map_[container_idx] = container_idx;
    HLOG(kDebug, "  Container[{}] -> Node[{}] (pool: {})", container_idx,
         container_idx, pool_id);
  }

  HLOG(kDebug, "=== Address Map Complete ===");
}

TaskResume PoolManager::CreatePool(FullPtr<Task> task, RunContext* run_ctx) {
  // For NVHPC: create a RunContext reference alias so CLIO_TASK_BODY_BEGIN
  // lambda can capture it by reference (the macro uses [=, &rctx])
#ifdef __NVCOMPILER
  RunContext& rctx = *run_ctx;
#endif
  CLIO_TASK_BODY_BEGIN
  if (!is_initialized_) {
    HLOG(kError, "PoolManager: Not initialized for pool creation");
    CLIO_CO_RETURN;
  }

  // Cast generic Task to BaseCreateTask to access pool operation parameters
  auto* create_task = reinterpret_cast<
      clio::run::admin::BaseCreateTask<clio::run::admin::CreateParams>*>(
      task.ptr_);

  // Debug: Log do_compose_ value after cast
  HLOG(kDebug, "PoolManager::CreatePool: After cast, do_compose_={}, is_admin_={}",
       create_task->do_compose_, create_task->is_admin_);

  // Extract parameters from the task
  const std::string chimod_name = create_task->chimod_name_.str();
  const std::string pool_name = create_task->pool_name_.str();
  const std::string chimod_params = create_task->chimod_params_.str();

  // Set num_containers equal to number of nodes in the cluster
  auto* ipc_manager = CLIO_IPC;
  std::vector<Host> all_hosts = ipc_manager->GetAllHosts();
  const u32 num_containers = static_cast<u32>(all_hosts.size());

  HLOG(kInfo,
       "PoolManager: Creating pool '{}' with {} containers (one per node)",
       pool_name, num_containers);

  // Make was_created a local variable
  bool was_created;

  // Validate pool parameters
  if (!ValidatePoolParams(chimod_name, pool_name)) {
    CLIO_CO_RETURN;
  }

  // Check if pool already exists by name (get-or-create semantics)
  PoolId existing_pool_id = FindPoolByName(pool_name);
  if (!existing_pool_id.IsNull()) {
    // Pool with this name already exists, update task with existing pool ID
    create_task->new_pool_id_ = existing_pool_id;
    was_created = false;
    HLOG(kInfo,
         "PoolManager: Pool with name '{}' for ChiMod '{}' already exists "
         "with PoolId {}, returning existing pool",
         pool_name, chimod_name, existing_pool_id);
    CLIO_CO_RETURN;
  }

  // Get the target pool ID from the task
  PoolId target_pool_id = create_task->new_pool_id_;

  // CRITICAL: Reject null pool IDs - users must provide explicit pool IDs
  if (target_pool_id.IsNull()) {
    HLOG(kError,
         "PoolManager: Cannot create pool with null PoolId. Users must provide "
         "explicit pool ID.");
    CLIO_CO_RETURN;
  }

  // Check if pool already exists by ID (should not happen with proper
  // generation, but safety check)
  if (HasPool(target_pool_id)) {
    // Pool already exists by ID, task already has correct new_pool_id_
    was_created = false;
    HLOG(kInfo,
         "PoolManager: Pool {} already exists by ID, returning existing pool",
         target_pool_id);
    CLIO_CO_RETURN;
  }

  // Create pool metadata
  PoolInfo pool_info(target_pool_id, pool_name, chimod_name, chimod_params,
                     num_containers);

  // Store pool metadata first so InitAddressMap can find it
  UpdatePoolMetadata(target_pool_id, pool_info);

  // Initialize address map for the pool (ContainerId -> NodeId)
  InitAddressMap(target_pool_id, num_containers);

  // Create local pool with containers (merged from CreateLocalPool)
  // Get module manager to create containers
  auto* module_manager = CLIO_MODULE_MANAGER;
  if (!module_manager) {
    HLOG(kError, "PoolManager: Module manager not available");
    ErasePoolMetadata(target_pool_id);
    CLIO_CO_RETURN;
  }

  Container* container = nullptr;
  auto* ipc_manager2 = CLIO_IPC;
  u32 node_id = ipc_manager2->GetNodeId();
  try {
    // Create container
    container =
        module_manager->CreateContainer(chimod_name, target_pool_id, pool_name);
    if (!container) {
      HLOG(kError, "PoolManager: Failed to create container for ChiMod: {}",
           chimod_name);
      ErasePoolMetadata(target_pool_id);
      CLIO_CO_RETURN;
    }

    // node_id already obtained above try block
    HLOG(kInfo,
         "Creating container for pool {} on node {} with container_id={}",
         target_pool_id, node_id, node_id);

    // Check if this is a restart scenario (compose mode with restart flag).
    // The compose PoolConfig also carries optional per-RPC access control
    // (container_visibility / container_rpc_acl), applied after Init below.
    bool is_restart = false;
    clio::run::PoolConfig pool_config;
    if (create_task->do_compose_) {
      pool_config =
          clio::run::Task::Deserialize<clio::run::PoolConfig>(create_task->chimod_params_);
      is_restart = pool_config.restart_;
    }

    // Initialize container with pool ID, name, and container ID
    if (is_restart) {
      HLOG(kInfo, "PoolManager: Restart detected for pool {}, calling Restart()", pool_name);
      container->Restart(target_pool_id, pool_name, node_id);
    } else {
      container->Init(target_pool_id, pool_name, node_id);
    }

    // Apply per-RPC access control now that the module's Init has populated the
    // method-name table (via SetMethodNames). Defaults (public, no overrides)
    // make this a no-op for pools that don't opt in.
    container->ConfigureAcl(pool_config.container_visibility_,
                            pool_config.rpc_acl_);

    // pool_external: this is a stub for a pool hosted on the fallback runtime.
    // Mark it external (tasks targeting it are punted to the fallback) and skip
    // running its Create method below — the real resources live on the fallback.
    container->SetExternal(pool_config.external_);

    HLOG(kInfo,
         "Container initialized with pool ID {}, name {}, and container ID {} "
         "(external={})",
         target_pool_id, pool_name, container->container_id_,
         pool_config.external_);

    // Register the container BEFORE running Create method
    // This allows Create to spawn tasks that can find this container in the map
    if (!RegisterContainer(target_pool_id, node_id, container, /*is_static=*/true)) {
      HLOG(kError, "PoolManager: Failed to register container");
      module_manager->DestroyContainer(chimod_name, container);
      ErasePoolMetadata(target_pool_id);
      CLIO_CO_RETURN;
    }

    // Run create method on container as a coroutine
    // The Create method returns a TaskResume that may yield (co_await) for
    // nested pool creation (e.g., CTE Create calling bdev Create).
    // By using co_await, we properly suspend and resume, allowing the worker
    // to process nested tasks while we wait.
    //
    // External stubs skip Create entirely: the real container (and its
    // resources) live on the fallback runtime, so there is nothing to allocate
    // here — running Create would build local resources we never use.
    if (!pool_config.external_) {
      HLOG(kInfo, "CreatePool: Running Create method for pool {}",
           target_pool_id);
      CLIO_CO_AWAIT(container->Run(0, task, *run_ctx));  // Method::kCreate = 0
      HLOG(kInfo, "CreatePool: Create method completed for pool {}",
           target_pool_id);

      if (task->GetReturnCode() != 0) {
        HLOG(kError, "PoolManager: Failed to create container for ChiMod: {}",
             chimod_name);
        // Unregister the container since Create failed
        UnregisterContainer(target_pool_id, node_id);
        module_manager->DestroyContainer(chimod_name, container);
        ErasePoolMetadata(target_pool_id);
        CLIO_CO_RETURN;
      }
    } else {
      HLOG(kInfo,
           "CreatePool: pool {} is external (hosted on fallback runtime); "
           "skipping local Create",
           target_pool_id);
    }

    // GPU container allocation removed along with the GPU runtime.
    // ChiMods now have CPU-only handlers; kernels submit tasks via
    // gpu2cpu_queue and the CPU dispatches into the standard
    // clio::run::Container path.

  } catch (const std::exception& e) {
    HLOG(kError, "PoolManager: Exception during pool creation: {}", e.what());
    if (container) {
      // Unregister if it was registered before the exception
      UnregisterContainer(target_pool_id, node_id);
      module_manager->DestroyContainer(chimod_name, container);
    }
    ErasePoolMetadata(target_pool_id);
    CLIO_CO_RETURN;
  }

  // Set success results
  was_created = true;
  (void)was_created;  // Suppress unused variable warning
  // Note: create_task->new_pool_id_ already contains target_pool_id

  HLOG(kInfo,
       "PoolManager: Created complete pool {} with ChiMod {} ({} containers)",
       target_pool_id, chimod_name, num_containers);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

TaskResume PoolManager::DestroyPool(PoolId pool_id) {
  // For NVHPC: provide a dummy RunContext reference so CLIO_TASK_BODY_BEGIN
  // lambda can compile (the macro captures rctx by ref, but we never use it).
#ifdef __NVCOMPILER
  clio::run::RunContext _dummy_rctx;
  clio::run::RunContext& rctx = _dummy_rctx;
#endif
  CLIO_TASK_BODY_BEGIN
  if (!is_initialized_) {
    HLOG(kError, "PoolManager: Not initialized for pool destruction");
    CLIO_CO_RETURN;
  }

  // Check if pool exists in metadata
  if (!HasPool(pool_id)) {
    HLOG(kError, "PoolManager: Pool {} metadata not found", pool_id);
    CLIO_CO_RETURN;
  }

  // Destroy local pool components
  if (!DestroyLocalPool(pool_id)) {
    HLOG(kError,
         "PoolManager: Failed to destroy local pool components for pool {}",
         pool_id);
    CLIO_CO_RETURN;
  }

  // Remove pool metadata
  ErasePoolMetadata(pool_id);

  HLOG(kInfo, "PoolManager: Destroyed complete pool {}", pool_id);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

const PoolInfo* PoolManager::GetPoolInfo(PoolId pool_id) const {
  if (!is_initialized_) {
    return nullptr;
  }

  PoolMetaReadLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  return (it != pool_metadata_.end()) ? &it->second : nullptr;
}

void PoolManager::UpdatePoolMetadata(PoolId pool_id, const PoolInfo& info) {
  if (!is_initialized_) {
    return;
  }

  PoolMetaWriteLock lock(pool_metadata_mutex_);
  pool_metadata_[pool_id] = info;
}

void PoolManager::ErasePoolMetadata(PoolId pool_id) {
  PoolMetaWriteLock lock(pool_metadata_mutex_);
  pool_metadata_.erase(pool_id);
}

u32 PoolManager::GetContainerNodeId(PoolId pool_id,
                                    ContainerId container_id) const {
  HLOG(kDebug, "GetContainerNodeId - pool_id={}, container_id={}", pool_id,
       container_id);

  if (!is_initialized_) {
    HLOG(kDebug, "GetContainerNodeId - not initialized, returning 0");
    return 0;  // Default to local node
  }

  // Hold the read lock across the address_map_ lookup. Resolving via a
  // PoolInfo* and reading address_map_ after the lock dropped would race a
  // concurrent UpdateContainerNodeMapping / pool insert (issue #572).
  PoolMetaReadLock lock(pool_metadata_mutex_);
  auto pit = pool_metadata_.find(pool_id);
  if (pit == pool_metadata_.end()) {
    HLOG(kDebug, "GetContainerNodeId - pool not found, returning 0");
    return 0;  // Pool not found, assume local
  }
  const PoolInfo &pool_info = pit->second;

  HLOG(kDebug, "GetContainerNodeId - pool has {} containers",
       pool_info.num_containers_);

  // Look up node ID from the address map
  auto it = pool_info.address_map_.find(container_id);
  if (it != pool_info.address_map_.end()) {
    HLOG(kDebug,
         "GetContainerNodeId - found mapping: container_id={} -> node_id={}",
         container_id, it->second);
    return it->second;
  }

  HLOG(kDebug, "GetContainerNodeId - mapping not found, returning 0");
  // Default to local node if mapping not found
  return 0;
}

bool PoolManager::UpdateContainerNodeMapping(PoolId pool_id,
                                              ContainerId container_id,
                                              u32 new_node_id) {
  if (!is_initialized_) {
    HLOG(kError, "PoolManager: Not initialized for mapping update");
    return false;
  }

  PoolMetaWriteLock lock(pool_metadata_mutex_);
  auto it = pool_metadata_.find(pool_id);
  if (it == pool_metadata_.end()) {
    HLOG(kError, "PoolManager: Pool {} not found for mapping update", pool_id);
    return false;
  }

  PoolInfo &pool_info = it->second;
  pool_info.address_map_[container_id] = new_node_id;

  HLOG(kInfo,
       "PoolManager: Updated mapping for pool {} container {} -> node {}",
       pool_id, container_id, new_node_id);
  return true;
}

void PoolManager::WriteAddressTableWAL(PoolId pool_id,
                                        ContainerId container_id,
                                        u32 old_node, u32 new_node) {
  auto *config_manager = CLIO_CONFIG_MANAGER;
  if (!config_manager) {
    HLOG(kError, "PoolManager: ConfigManager not available for WAL write");
    return;
  }

  // Create WAL directory
  std::string wal_dir = config_manager->GetConfDir() + "/wal";
  std::filesystem::create_directories(wal_dir);

  // Determine this node's ID for the WAL filename
  auto *ipc_manager = CLIO_IPC;
  u32 node_id = ipc_manager->GetNodeId();

  std::string wal_path = wal_dir + "/domain_table." +
                          std::to_string(pool_id.major_) + "." +
                          std::to_string(pool_id.minor_) + "." +
                          std::to_string(node_id) + ".bin";

  // Append WAL entry: [timestamp:u64][pool_id:PoolId][container_id:u32][old_node:u32][new_node:u32]
  std::ofstream ofs(wal_path, std::ios::binary | std::ios::app);
  if (!ofs.is_open()) {
    HLOG(kError, "PoolManager: Failed to open WAL file: {}", wal_path);
    return;
  }

  auto now = std::chrono::system_clock::now();
  u64 timestamp = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch())
          .count());

  ofs.write(reinterpret_cast<const char *>(&timestamp), sizeof(timestamp));
  ofs.write(reinterpret_cast<const char *>(&pool_id), sizeof(pool_id));
  ofs.write(reinterpret_cast<const char *>(&container_id),
            sizeof(container_id));
  ofs.write(reinterpret_cast<const char *>(&old_node), sizeof(old_node));
  ofs.write(reinterpret_cast<const char *>(&new_node), sizeof(new_node));

  HLOG(kDebug, "PoolManager: WAL entry written to {}", wal_path);
}

void PoolManager::ReplayAddressTableWAL() {
  auto *config_manager = CLIO_CONFIG_MANAGER;
  if (!config_manager) {
    HLOG(kError, "ReplayAddressTableWAL: ConfigManager not available");
    return;
  }

  std::string wal_dir = config_manager->GetConfDir() + "/wal";

  namespace fs = std::filesystem;
  if (!fs::exists(wal_dir) || !fs::is_directory(wal_dir)) {
    HLOG(kInfo, "ReplayAddressTableWAL: No WAL directory at {}", wal_dir);
    return;
  }

  size_t entries_replayed = 0;
  for (const auto &dir_entry : fs::directory_iterator(wal_dir)) {
    if (dir_entry.path().extension() != ".bin") continue;

    std::ifstream ifs(dir_entry.path(), std::ios::binary);
    if (!ifs.is_open()) continue;

    while (ifs.good()) {
      u64 timestamp;
      PoolId pool_id;
      u32 container_id, old_node, new_node;

      ifs.read(reinterpret_cast<char*>(&timestamp), sizeof(timestamp));
      ifs.read(reinterpret_cast<char*>(&pool_id), sizeof(pool_id));
      ifs.read(reinterpret_cast<char*>(&container_id), sizeof(container_id));
      ifs.read(reinterpret_cast<char*>(&old_node), sizeof(old_node));
      ifs.read(reinterpret_cast<char*>(&new_node), sizeof(new_node));
      if (ifs.fail()) break;

      // Apply the last-writer-wins mapping
      UpdateContainerNodeMapping(pool_id, container_id, new_node);
      entries_replayed++;
    }
  }

  HLOG(kInfo, "ReplayAddressTableWAL: Replayed {} entries", entries_replayed);
}

}  // namespace clio::run