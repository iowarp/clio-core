/**
 * Pool manager implementation
 */

#include "chimaera/pool_manager.h"

#include "chimaera/admin/admin_tasks.h"
#include "chimaera/container.h"
#include "chimaera/task.h"

// Global pointer variable definition for Pool manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::PoolManager, g_pool_manager);

namespace chi {

// Constructor and destructor removed - handled by HSHM singleton pattern

bool PoolManager::ServerInit() {
  if (is_initialized_) {
    return true;
  }

  // Initialize pool container map and metadata
  pool_container_map_.clear();
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

  // Clear pool container mappings
  pool_container_map_.clear();

  is_initialized_ = false;
}

bool PoolManager::RegisterContainer(PoolId pool_id, Container* container) {
  if (!is_initialized_ || container == nullptr) {
    return false;
  }

  pool_container_map_[pool_id] = container;
  return true;
}

bool PoolManager::UnregisterContainer(PoolId pool_id) {
  if (!is_initialized_) {
    return false;
  }

  auto it = pool_container_map_.find(pool_id);
  if (it != pool_container_map_.end()) {
    pool_container_map_.erase(it);
    return true;
  }
  return false;
}

Container* PoolManager::GetContainer(PoolId pool_id) const {
  if (!is_initialized_) {
    return nullptr;
  }

  auto it = pool_container_map_.find(pool_id);
  return (it != pool_container_map_.end()) ? it->second : nullptr;
}

bool PoolManager::HasPool(PoolId pool_id) const {
  if (!is_initialized_) {
    return false;
  }

  return pool_container_map_.find(pool_id) != pool_container_map_.end();
}

bool PoolManager::HasContainer(PoolId pool_id, ContainerId container_id) const {
  if (!is_initialized_) {
    return false;
  }

  // Check if pool exists on this node
  if (!HasPool(pool_id)) {
    return false;
  }

  // Get this node's ID
  auto* ipc_manager = CHI_IPC;
  u32 node_id = ipc_manager->GetNodeId();

  // Container exists locally if container_id matches this node's ID
  // This follows the pattern where container_id == node_id for locally owned
  // containers
  return container_id == node_id;
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
  return is_initialized_ ? pool_container_map_.size() : 0;
}

std::vector<PoolId> PoolManager::GetAllPoolIds() const {
  std::vector<PoolId> pool_ids;
  if (!is_initialized_) {
    return pool_ids;
  }

  pool_ids.reserve(pool_container_map_.size());
  for (const auto& pair : pool_container_map_) {
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

  // Get the container before unregistering
  auto* container = GetContainer(pool_id);
  if (!container) {
    HLOG(kError, "PoolManager: Container for pool {} is null", pool_id);
    return false;
  }

  // Get module manager to destroy the container
  auto* module_manager = CHI_MODULE_MANAGER;
  if (!module_manager) {
    HLOG(kError, "PoolManager: Module manager not available");
    return false;
  }

  try {
    // Unregister first
    if (!UnregisterContainer(pool_id)) {
      HLOG(kError, "PoolManager: Failed to unregister container for pool {}",
           pool_id);
      return false;
    }

    // TODO: Determine ChiMod name for destruction - for now assume it's stored
    // in container This would require extending ChiContainer interface to store
    // chimod_name For now, we'll skip the destruction call and rely on
    // container cleanup

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

AddressTable PoolManager::CreateAddressTable(PoolId pool_id,
                                             u32 num_containers) {
  AddressTable address_table;

  if (!is_initialized_) {
    return address_table;
  }

  HLOG(kDebug, "=== Address Table Mapping for Pool {} ===", pool_id);
  HLOG(kDebug, "Creating address table with {} containers", num_containers);

  // Create one address per container in the global table
  for (u32 container_idx = 0; container_idx < num_containers; ++container_idx) {
    Address global_address(pool_id, Group::kGlobal, container_idx);
    Address physical_address(pool_id, Group::kPhysical, container_idx);

    // Map each global address to its corresponding physical address
    address_table.AddGlobalToPhysicalMapping(global_address, physical_address);

    HLOG(kDebug, "  Global[{}] -> Physical[{}] (pool: {})", container_idx,
         container_idx, pool_id);
  }

  // Create exactly one local address that maps to the global address of the
  // container on this node. Use this node's ID to determine which global
  // container this node owns.
  auto* ipc_manager = CHI_IPC;
  u32 node_id = ipc_manager->GetNodeId();

  Address local_address(pool_id, Group::kLocal,
                        0);  // One local address for this node
  Address global_address(pool_id, Group::kGlobal,
                         node_id);  // Maps to this node's global container

  // Map the single local address to its global counterpart
  address_table.AddLocalToGlobalMapping(local_address, global_address);

  HLOG(kDebug, "  Local[0] -> Global[{}] (pool: {})", node_id, pool_id);
  HLOG(kDebug, "=== Address Table Complete ===");

  return address_table;
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

  // Create address table for the pool
  AddressTable address_table =
      CreateAddressTable(target_pool_id, num_containers);

  // Create pool metadata
  PoolInfo pool_info(target_pool_id, pool_name, chimod_name, chimod_params,
                     num_containers);
  pool_info.address_table_ = address_table;

  // Store pool metadata
  UpdatePoolMetadata(target_pool_id, pool_info);

  // Create local pool with containers (merged from CreateLocalPool)
  // Get module manager to create containers
  auto* module_manager = CHI_MODULE_MANAGER;
  if (!module_manager) {
    HLOG(kError, "PoolManager: Module manager not available");
    pool_metadata_.erase(target_pool_id);
    co_return;
  }

  Container* container = nullptr;
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

    // Get this node's ID to use as the container ID
    auto* ipc_manager = CHI_IPC;
    u32 node_id = ipc_manager->GetNodeId();
    HLOG(kInfo,
         "Creating container for pool {} on node {} with container_id={}",
         target_pool_id, node_id, node_id);

    // Initialize container with pool ID, name, and container ID (this will call
    // InitClient internally)
    container->Init(target_pool_id, pool_name, node_id);

    HLOG(kInfo,
         "Container initialized with pool ID {}, name {}, and container ID {}",
         target_pool_id, pool_name, container->container_id_);

    // Register the container BEFORE running Create method
    // This allows Create to spawn tasks that can find this container in the map
    if (!RegisterContainer(target_pool_id, container)) {
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
      UnregisterContainer(target_pool_id);
      module_manager->DestroyContainer(chimod_name, container);
      pool_metadata_.erase(target_pool_id);
      co_return;
    }

  } catch (const std::exception& e) {
    HLOG(kError, "PoolManager: Exception during pool creation: {}", e.what());
    if (container) {
      // Unregister if it was registered before the exception
      UnregisterContainer(target_pool_id);
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

  // Create global address for the container
  Address global_address(pool_id, Group::kGlobal, container_id);

  // Look up physical address from the address table
  Address physical_address;
  if (pool_info->address_table_.GlobalToPhysical(global_address,
                                                 physical_address)) {
    // Return the minor_id which represents the node ID
    HLOG(kDebug,
         "GetContainerNodeId - found mapping: container_id={} -> node_id={}",
         container_id, physical_address.minor_id_);
    return physical_address.minor_id_;
  }

  HLOG(kDebug, "GetContainerNodeId - mapping not found, returning 0");
  // Default to local node if mapping not found
  return 0;
}

}  // namespace chi