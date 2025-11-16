/**
 * Chimaera manager implementation
 */

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "chimaera/admin/admin_client.h"
#include "chimaera/singletons.h"

// Global pointer variable definition for Chimaera manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::Chimaera, g_chimaera_manager);

namespace chi {

// HSHM Thread-local storage key definitions
hshm::ThreadLocalKey chi_cur_worker_key_;
hshm::ThreadLocalKey chi_task_counter_key_;

/**
 * Create a new TaskId with current process/thread info and next major counter
 */
TaskId CreateTaskId() {
  // Get thread-local task counter at the beginning
  TaskCounter *counter =
      HSHM_THREAD_MODEL->GetTls<TaskCounter>(chi_task_counter_key_);
  if (!counter) {
    // Initialize counter if not present
    counter = new TaskCounter();
    HSHM_THREAD_MODEL->SetTls(chi_task_counter_key_, counter);
  }

  // Get node_id from IpcManager
  auto *ipc_manager = CHI_IPC;
  u64 node_id = ipc_manager ? ipc_manager->GetNodeId() : 0;

  // In runtime mode, check if we have a current worker
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager && chimaera_manager->IsRuntime()) {
    Worker *current_worker = CHI_CUR_WORKER;
    if (current_worker) {
      // Get current task from worker
      FullPtr<Task> current_task = current_worker->GetCurrentTask();
      if (!current_task.IsNull()) {
        // Copy TaskId from current task, keep replica_id_ same, and allocate
        // new unique from counter
        TaskId new_id = current_task->task_id_;
        new_id.unique_ = counter->GetNext();
        new_id.node_id_ = node_id;
        return new_id;
      }
    }
  }

  // Fallback: Create new TaskId using counter (client mode or no current task)
  // Get system information singleton (avoid direct dereferencing)
  auto *system_info = HSHM_SYSTEM_INFO;
  u32 pid = system_info ? system_info->pid_ : 0;

  // Get thread ID
  u32 tid = static_cast<u32>(HSHM_THREAD_MODEL->GetTid().tid_);

  // Get next counter value for both major and unique
  u32 major = counter->GetNext();

  return TaskId(
      pid, tid, major, 0, major,
      node_id); // replica_id_ starts at 0, unique = major for root tasks
}

Chimaera::~Chimaera() {
  if (is_initialized_) {
    // Always finalize client components if client mode was initialized
    if (is_client_mode_) {
      ClientFinalize();
    }

    // Only finalize server components if runtime mode was initialized
    if (is_runtime_mode_) {
      ServerFinalize();
    }
  }
}

bool Chimaera::ClientInit() {
  HILOG(kInfo, "Chimaera::ClientInit");
  if (is_client_mode_ || client_is_initializing_ || runtime_is_initializing_) {
    return true;
  }

  client_is_initializing_ = true;

  HILOG(kDebug, "IpcManager::ClientInit");
  // Initialize configuration manager
  auto *config_manager = CHI_CONFIG_MANAGER;
  if (!config_manager->Init()) {
    client_is_initializing_ = false;
    return false;
  }

  HILOG(kDebug, "IpcManager::ClientInit");
  // Initialize IPC manager for client
  auto *ipc_manager = CHI_IPC;
  if (!ipc_manager->ClientInit()) {
    client_is_initializing_ = false;
    return false;
  }

  // Pool manager is not initialized in client mode
  // It's only needed for server/runtime mode

  // Initialize CHI_ADMIN singleton
  // The admin container is already created by the runtime, so we just
  // construct the admin client directly with the admin pool ID
  HILOG(kDebug, "Initializing CHI_ADMIN singleton");
  if (CHI_ADMIN == nullptr) {
    g_admin = new chimaera::admin::Client(chi::kAdminPoolId);
  }

  is_client_mode_ = true;
  is_initialized_ = true;
  client_is_initializing_ = false;

  return true;
}

bool Chimaera::ServerInit() {
  if (is_runtime_mode_ || runtime_is_initializing_ || client_is_initializing_) {
    return true;
  }

  runtime_is_initializing_ = true;

  // Initialize configuration manager first
  auto *config_manager = CHI_CONFIG_MANAGER;
  if (!config_manager->Init()) {
    runtime_is_initializing_ = false;
    return false;
  }

  // Initialize IPC manager for server
  auto *ipc_manager = CHI_IPC;
  if (!ipc_manager->ServerInit()) {
    runtime_is_initializing_ = false;
    return false;
  }

  HILOG(kDebug, "Host identification successful: {}",
        ipc_manager->GetCurrentHostname());

  // Initialize module manager first (needed for admin chimod)
  auto *module_manager = CHI_MODULE_MANAGER;
  if (!module_manager->Init()) {
    runtime_is_initializing_ = false;
    return false;
  }

  // Initialize work orchestrator before pool manager
  auto *work_orchestrator = CHI_WORK_ORCHESTRATOR;
  if (!work_orchestrator->Init()) {
    runtime_is_initializing_ = false;
    return false;
  }

  // Start worker threads
  if (!work_orchestrator->StartWorkers()) {
    runtime_is_initializing_ = false;
    return false;
  }

  // Initialize pool manager (server mode only) after work orchestrator
  auto *pool_manager = CHI_POOL_MANAGER;
  if (!pool_manager->ServerInit()) {
    runtime_is_initializing_ = false;
    return false;
  }

  // Process compose section if present
  const auto &compose_config = config_manager->GetComposeConfig();
  if (!compose_config.pools_.empty()) {
    HILOG(kInfo, "Processing compose configuration with {} pools",
          compose_config.pools_.size());

    // Get admin client to process compose
    auto *admin_client = CHI_ADMIN;
    if (!admin_client) {
      HELOG(kError, "Failed to get admin client for compose processing");
      return false;
    }

    // Call compose to create all configured pools
    if (!admin_client->Compose(compose_config)) {
      HELOG(kError, "Compose processing failed");
      return false;
    }

    HILOG(kInfo, "Compose processing completed successfully");
  }

  // Start local server last - after all other initialization is complete
  // This ensures clients can connect only when runtime is fully ready
  if (!ipc_manager->StartLocalServer()) {
    HELOG(kError, "Failed to start local server - runtime initialization failed");
    runtime_is_initializing_ = false;
    return false;
  }



  is_runtime_mode_ = true;
  is_initialized_ = true;
  runtime_is_initializing_ = false;

  return true;
}

void Chimaera::ClientFinalize() {
  if (!is_initialized_ || !is_client_mode_) {
    return;
  }

  // Finalize client components
  auto *pool_manager = CHI_POOL_MANAGER;
  pool_manager->Finalize();
  auto *ipc_manager = CHI_IPC;
  ipc_manager->ClientFinalize();

  is_client_mode_ = false;
  // Only set is_initialized_ = false if both modes are inactive
  if (!is_runtime_mode_) {
    is_initialized_ = false;
  }
}

void Chimaera::ServerFinalize() {
  if (!is_initialized_ || !is_runtime_mode_) {
    return;
  }

  // Stop workers and finalize server components
  auto *work_orchestrator = CHI_WORK_ORCHESTRATOR;
  work_orchestrator->StopWorkers();
  work_orchestrator->Finalize();
  auto *module_manager = CHI_MODULE_MANAGER;
  module_manager->Finalize();

  // Finalize shared components
  auto *pool_manager = CHI_POOL_MANAGER;
  pool_manager->Finalize();
  auto *ipc_manager = CHI_IPC;
  ipc_manager->ServerFinalize();

  is_runtime_mode_ = false;
  // Only set is_initialized_ = false if both modes are inactive
  if (!is_client_mode_) {
    is_initialized_ = false;
  }
}

bool Chimaera::IsInitialized() const { return is_initialized_; }

bool Chimaera::IsClient() const { return is_client_mode_; }

bool Chimaera::IsRuntime() const { return is_runtime_mode_; }

const std::string &Chimaera::GetCurrentHostname() const {
  auto *ipc_manager = CHI_IPC;
  return ipc_manager->GetCurrentHostname();
}

u64 Chimaera::GetNodeId() const {
  auto *ipc_manager = CHI_IPC;
  return ipc_manager->GetNodeId();
}

bool Chimaera::IsInitializing() const {
  return client_is_initializing_ || runtime_is_initializing_;
}

} // namespace chi