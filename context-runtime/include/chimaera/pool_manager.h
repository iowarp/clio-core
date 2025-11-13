#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_POOL_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_POOL_MANAGER_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <atomic>
#include "chimaera/types.h"

namespace chi {

// Forward declarations for ChiMod system
// Container is always a class forward declaration (defined in container.h)
class Container;
class Task;
struct RunContext;

/**
 * Address mapping table for pool management
 * 
 * Contains two unordered_maps for address translation:
 * - Local to Global address mapping
 * - Global to Physical address mapping
 */
struct AddressTable {
  // Local to global: Maps local addresses to global addresses
  std::unordered_map<Address, Address, AddressHash> local_to_global_map_;
  
  // Global to physical: Maps global addresses to physical addresses
  std::unordered_map<Address, Address, AddressHash> global_to_physical_map_;
  
  /**
   * Add local to global mapping
   */
  void AddLocalToGlobalMapping(const Address& local_addr, const Address& global_addr) {
    local_to_global_map_[local_addr] = global_addr;
  }
  
  /**
   * Add global to physical mapping
   */
  void AddGlobalToPhysicalMapping(const Address& global_addr, const Address& physical_addr) {
    global_to_physical_map_[global_addr] = physical_addr;
  }
  
  /**
   * Convert local address to global address
   */
  bool LocalToGlobal(const Address& local_addr, Address& global_addr) const {
    auto it = local_to_global_map_.find(local_addr);
    if (it != local_to_global_map_.end()) {
      global_addr = it->second;
      return true;
    }
    return false;
  }
  
  /**
   * Convert global address to physical address
   */
  bool GlobalToPhysical(const Address& global_addr, Address& physical_addr) const {
    auto it = global_to_physical_map_.find(global_addr);
    if (it != global_to_physical_map_.end()) {
      physical_addr = it->second;
      return true;
    }
    return false;
  }
  
  /**
   * Remove local to global mapping
   */
  void RemoveLocalToGlobalMapping(const Address& local_addr) {
    local_to_global_map_.erase(local_addr);
  }
  
  /**
   * Remove global to physical mapping
   */
  void RemoveGlobalToPhysicalMapping(const Address& global_addr) {
    global_to_physical_map_.erase(global_addr);
  }
  
  /**
   * Clear all mappings
   */
  void Clear() {
    local_to_global_map_.clear();
    global_to_physical_map_.clear();
  }

  /**
   * Get global address for a container ID
   * @param container_id Container identifier
   * @return Global address for the container
   */
  Address GetGlobalAddress(u32 container_id) const {
    // For now, assume container_id maps to global address with same minor_id
    // This could be made more sophisticated based on addressing scheme
    if (!global_to_physical_map_.empty()) {
      auto it = global_to_physical_map_.begin();
      PoolId pool_id = it->first.pool_id_;
      return Address(pool_id, Group::kGlobal, container_id);
    }
    return Address();
  }

  /**
   * Get physical nodes for a global address
   * @param global_address Global address to look up
   * @return Vector of physical node IDs
   */
  std::vector<u32> GetPhysicalNodes(const Address& global_address) const {
    std::vector<u32> nodes;
    Address physical_address;
    if (GlobalToPhysical(global_address, physical_address)) {
      nodes.push_back(physical_address.minor_id_);
    }
    return nodes;
  }
};

/**
 * Pool metadata containing domain tables and configuration
 */
struct PoolInfo {
  PoolId pool_id_;
  std::string pool_name_;
  std::string chimod_name_;
  std::string chimod_params_;
  u32 num_containers_;
  AddressTable address_table_;
  bool is_active_;
  
  PoolInfo() : pool_id_(), num_containers_(0), is_active_(false) {}
  
  PoolInfo(PoolId pool_id, const std::string& pool_name, 
           const std::string& chimod_name, const std::string& chimod_params,
           u32 num_containers)
      : pool_id_(pool_id), pool_name_(pool_name), chimod_name_(chimod_name),
        chimod_params_(chimod_params), num_containers_(num_containers), is_active_(true) {}
};

/**
 * Pool Manager singleton for managing ChiPools and Containers
 * 
 * Maps PoolId to Containers on this node and manages the lifecycle
 * of pools in the distributed system.
 * Uses HSHM global cross pointer variable singleton pattern.
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
   * Register a Container with a specific PoolId
   * @param pool_id Pool identifier
   * @param container Pointer to Container
   * @return true if registration successful, false otherwise
   */
  bool RegisterContainer(PoolId pool_id, Container* container);

  /**
   * Unregister a Container
   * @param pool_id Pool identifier
   * @return true if unregistration successful, false otherwise
   */
  bool UnregisterContainer(PoolId pool_id);

  /**
   * Get Container by PoolId
   * @param pool_id Pool identifier
   * @return Pointer to Container or nullptr if not found
   */
  Container* GetContainer(PoolId pool_id) const;

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
   * Create address table for a pool
   * @param pool_id Pool identifier
   * @param num_containers Number of containers in the pool
   * @return Address table for the pool
   */
  AddressTable CreateAddressTable(PoolId pool_id, u32 num_containers);

  /**
   * Create or get a complete pool with get-or-create semantics
   * Extracts all parameters from the task (chimod_name, pool_name, chimod_params)
   * @param task Task containing pool creation parameters (updated with final pool ID)
   * @param run_ctx RunContext for container initialization
   * @return true if operation successful, false otherwise
   */
  bool CreatePool(FullPtr<Task> task, RunContext* run_ctx);


  /**
   * Destroy a complete pool including metadata and local containers
   * @param pool_id Pool identifier
   * @return true if pool destruction successful, false otherwise
   */
  bool DestroyPool(PoolId pool_id);

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

 private:


  bool is_initialized_ = false;
  
  // Map PoolId to Containers on this node
  std::unordered_map<PoolId, Container*> pool_container_map_;
  
  // Map PoolId to pool metadata
  std::unordered_map<PoolId, PoolInfo> pool_metadata_;
  
  // Pool ID counter for generating unique IDs (used as minor number)
  std::atomic<u32> next_pool_minor_{5}; // Start at 5 for safety, 1 reserved for admin

};

}  // namespace chi

// Global pointer variable declaration for Pool manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::PoolManager, g_pool_manager);

// Macro for accessing the Pool manager singleton using global pointer variable
#define CHI_POOL_MANAGER HSHM_GET_GLOBAL_PTR_VAR(::chi::PoolManager, g_pool_manager)

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_POOL_MANAGER_H_