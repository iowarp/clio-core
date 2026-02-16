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

#include "chimaera/pool_manager.h"

#include "chimaera/admin/admin_tasks.h"
#include "chimaera/config_manager.h"
#include "chimaera/container.h"
#include "chimaera/task.h"

#include <chrono>
#include <filesystem>
#include <fstream>

// Global pointer variable definition for Pool manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::PoolManager, g_pool_manager);

namespace chi {

// Constructor and destructor removed - handled by HSHM singleton pattern

bool PoolManager::ServerInit() {
  if (is_initialized_) {
    return true;
  }

  // Initialize pool metadata
  pool_metadata_.clear();

  is_initialized_ = true;

  // Create the admin chimod pool (kAdminPoolId = 1)
  // This is required for flush operations and other admin tasks
  PoolId admin_pool_id;

  // Create proper admin task and RunContext for pool creation
  auto* ipc_manager = CHI_IPC;
  if (!ipc_manager) {
    HLOG(kError, "PoolManager: IPC manager not available during ServerInit");
    return false;
  }

  auto admin_task = ipc_manager->NewTask<chimaera::admin::CreateTask>(
      CreateTaskId(),
      kAdminPoolId,  // Use admin pool for admin container creation
      PoolQuery::Local(), "chimaera_admin", "admin", kAdminPoolId,
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

void PoolManager::Finalize() {
  if (!is_initialized_) {
    return;
  }

  // Clear all containers in each PoolInfo, then clear metadata
  for (auto &pair : pool_metadata_) {
    pair.second.containers_.clear();
    pair.second.static_container_ = nullptr;
    pair.second.local_container_ = nullptr;
  }
  pool_metadata_.clear();

  is_initialized_ = false;
}

bool PoolManager::RegisterContainer(PoolId pool_id, ContainerId container_id,
                                     Container* container, bool is_static) {
  if (!is_initialized_ || container == nullptr) {
    return false;
  }

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

  // Never null out static_container_ — it is needed for task deserialization
  // (e.g., ClientRecv) even after migration removes the execution container.
  // Only reassign if another container is available.
  if (info.static_container_ == removed && !info.containers_.empty()) {
    info.static_container_ = info.containers_.begin()->second;
  }
  if (info.local_container_ == removed) {
    info.RecalculateLocalContainer();
  }

  return true;
}

void PoolManager::UnregisterAllContainers(PoolId pool_id) {
  if (!is_initialized_) {
    return;
  }

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
    HSHM_THREAD_MODEL->Yield();
  }
}

bool PoolManager::HasPool(PoolId pool_id) const {
  if (!is_initialized_) {
    return false;
  }

  return pool_metadata_.find(pool_id) != pool_metadata_.end();
}

bool PoolManager::HasContainer(PoolId pool_id, ContainerId container_id) const {
  if (!is_initialized_) {
    return false;
  }

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
  return is_initialized_ ? pool_metadata_.size() : 0;
}

std::vector<PoolId> PoolManager::GetAllPoolIds() const {
  std::vector<PoolId> pool_ids;
  if (!is_initialized_) {
    return pool_ids;
  }

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
  auto* ipc_manager = CHI_IPC;
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
  auto* module_manager = CHI_MODULE_MANAGER;
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
  if (!is_initialized_) {
    HLOG(kError, "PoolManager: Not initialized for pool creation");
    co_return;
  }

  // Cast generic Task to BaseCreateTask to access pool operation parameters
  auto* create_task = reinterpret_cast<
      chimaera::admin::BaseCreateTask<chimaera::admin::CreateParams>*>(
      task.ptr_);

  // Debug: Log do_compose_ value after cast
  HLOG(kDebug, "PoolManager::CreatePool: After cast, do_compose_={}, is_admin_={}",
       create_task->do_compose_, create_task->is_admin_);

  // Extract parameters from the task
  const std::string chimod_name = create_task->chimod_name_.str();
  const std::string pool_name = create_task->pool_name_.str();
  const std::string chimod_params = create_task->chimod_params_.str();

  // Set num_containers equal to number of nodes in the cluster
  auto* ipc_manager = CHI_IPC;
  std::vector<Host> all_hosts = ipc_manager->GetAllHosts();
  const u32 num_containers = static_cast<u32>(all_hosts.size());

  HLOG(kInfo,
       "PoolManager: Creating pool '{}' with {} containers (one per node)",
       pool_name, num_containers);

  // Make was_created a local variable
  bool was_created;

  // Validate pool parameters
  if (!ValidatePoolParams(chimod_name, pool_name)) {
    co_return;
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
    co_return;
  }

  // Get the target pool ID from the task
  PoolId target_pool_id = create_task->new_pool_id_;

  // CRITICAL: Reject null pool IDs - users must provide explicit pool IDs
  if (target_pool_id.IsNull()) {
    HLOG(kError,
         "PoolManager: Cannot create pool with null PoolId. Users must provide "
         "explicit pool ID.");
    co_return;
  }

  // Check if pool already exists by ID (should not happen with proper
  // generation, but safety check)
  if (HasPool(target_pool_id)) {
    // Pool already exists by ID, task already has correct new_pool_id_
    was_created = false;
    HLOG(kInfo,
         "PoolManager: Pool {} already exists by ID, returning existing pool",
         target_pool_id);
    co_return;
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
  auto* module_manager = CHI_MODULE_MANAGER;
  if (!module_manager) {
    HLOG(kError, "PoolManager: Module manager not available");
    pool_metadata_.erase(target_pool_id);
    co_return;
  }

  Container* container = nullptr;
  auto* ipc_manager2 = CHI_IPC;
  u32 node_id = ipc_manager2->GetNodeId();
  try {
    // Create container
    container =
        module_manager->CreateContainer(chimod_name, target_pool_id, pool_name);
    if (!container) {
      HLOG(kError, "PoolManager: Failed to create container for ChiMod: {}",
           chimod_name);
      pool_metadata_.erase(target_pool_id);
      co_return;
    }

    // node_id already obtained above try block
    HLOG(kInfo,
         "Creating container for pool {} on node {} with container_id={}",
         target_pool_id, node_id, node_id);

    // Check if this is a restart scenario (compose mode with restart flag)
    bool is_restart = false;
    if (create_task->do_compose_) {
      chi::PoolConfig pool_config =
          chi::Task::Deserialize<chi::PoolConfig>(create_task->chimod_params_);
      is_restart = pool_config.restart_;
    }

    // Initialize container with pool ID, name, and container ID
    if (is_restart) {
      HLOG(kInfo, "PoolManager: Restart detected for pool {}, calling Restart()", pool_name);
      container->Restart(target_pool_id, pool_name, node_id);
    } else {
      container->Init(target_pool_id, pool_name, node_id);
    }

    HLOG(kInfo,
         "Container initialized with pool ID {}, name {}, and container ID {}",
         target_pool_id, pool_name, container->container_id_);

    // Register the container BEFORE running Create method
    // This allows Create to spawn tasks that can find this container in the map
    if (!RegisterContainer(target_pool_id, node_id, container, /*is_static=*/true)) {
      HLOG(kError, "PoolManager: Failed to register container");
      module_manager->DestroyContainer(chimod_name, container);
      pool_metadata_.erase(target_pool_id);
      co_return;
    }

    // Run create method on container as a coroutine
    // The Create method returns a TaskResume that may yield (co_await) for
    // nested pool creation (e.g., CTE Create calling bdev Create).
    // By using co_await, we properly suspend and resume, allowing the worker
    // to process nested tasks while we wait.
    co_await container->Run(0, task, *run_ctx);  // Method::kCreate = 0

    if (task->GetReturnCode() != 0) {
      HLOG(kError, "PoolManager: Failed to create container for ChiMod: {}",
           chimod_name);
      // Unregister the container since Create failed
      UnregisterContainer(target_pool_id, node_id);
      module_manager->DestroyContainer(chimod_name, container);
      pool_metadata_.erase(target_pool_id);
      co_return;
    }

  } catch (const std::exception& e) {
    HLOG(kError, "PoolManager: Exception during pool creation: {}", e.what());
    if (container) {
      // Unregister if it was registered before the exception
      UnregisterContainer(target_pool_id, node_id);
      module_manager->DestroyContainer(chimod_name, container);
    }
    pool_metadata_.erase(target_pool_id);
    co_return;
  }

  // Set success results
  was_created = true;
  (void)was_created;  // Suppress unused variable warning
  // Note: create_task->new_pool_id_ already contains target_pool_id

  HLOG(kInfo,
       "PoolManager: Created complete pool {} with ChiMod {} ({} containers)",
       target_pool_id, chimod_name, num_containers);
  co_return;
}

TaskResume PoolManager::DestroyPool(PoolId pool_id) {
  if (!is_initialized_) {
    HLOG(kError, "PoolManager: Not initialized for pool destruction");
    co_return;
  }

  // Check if pool exists in metadata
  auto metadata_it = pool_metadata_.find(pool_id);
  if (metadata_it == pool_metadata_.end()) {
    HLOG(kError, "PoolManager: Pool {} metadata not found", pool_id);
    co_return;
  }

  // Destroy local pool components
  if (!DestroyLocalPool(pool_id)) {
    HLOG(kError,
         "PoolManager: Failed to destroy local pool components for pool {}",
         pool_id);
    co_return;
  }

  // Remove pool metadata
  pool_metadata_.erase(metadata_it);

  HLOG(kInfo, "PoolManager: Destroyed complete pool {}", pool_id);
  co_return;
}

const PoolInfo* PoolManager::GetPoolInfo(PoolId pool_id) const {
  if (!is_initialized_) {
    return nullptr;
  }

  auto it = pool_metadata_.find(pool_id);
  return (it != pool_metadata_.end()) ? &it->second : nullptr;
}

void PoolManager::UpdatePoolMetadata(PoolId pool_id, const PoolInfo& info) {
  if (!is_initialized_) {
    return;
  }

  pool_metadata_[pool_id] = info;
}

u32 PoolManager::GetContainerNodeId(PoolId pool_id,
                                    ContainerId container_id) const {
  HLOG(kDebug, "GetContainerNodeId - pool_id={}, container_id={}", pool_id,
       container_id);

  if (!is_initialized_) {
    HLOG(kDebug, "GetContainerNodeId - not initialized, returning 0");
    return 0;  // Default to local node
  }

  // Get pool metadata
  const PoolInfo* pool_info = GetPoolInfo(pool_id);
  if (!pool_info) {
    HLOG(kDebug, "GetContainerNodeId - pool not found, returning 0");
    return 0;  // Pool not found, assume local
  }

  HLOG(kDebug, "GetContainerNodeId - pool has {} containers",
       pool_info->num_containers_);

  // Look up node ID from the address map
  auto it = pool_info->address_map_.find(container_id);
  if (it != pool_info->address_map_.end()) {
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
  auto *config_manager = CHI_CONFIG_MANAGER;
  if (!config_manager) {
    HLOG(kError, "PoolManager: ConfigManager not available for WAL write");
    return;
  }

  // Create WAL directory
  std::string wal_dir = config_manager->GetConfDir() + "/wal";
  std::filesystem::create_directories(wal_dir);

  // Determine this node's ID for the WAL filename
  auto *ipc_manager = CHI_IPC;
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

}  // namespace chi