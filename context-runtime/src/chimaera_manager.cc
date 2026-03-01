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
 * Chimaera manager implementation
 */

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "chimaera/admin/admin_client.h"
#include "chimaera/singletons.h"

// Global pointer variable definition for Chimaera manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::Chimaera, g_chimaera_manager);

static void ChimaeraCleanupAtExit() {
  if (g_chimaera_manager) {
    delete g_chimaera_manager;
    g_chimaera_manager = nullptr;
  }
}

namespace chi {

// HSHM Thread-local storage key definitions
hshm::ThreadLocalKey chi_cur_worker_key_;
bool chi_cur_worker_key_created_ = false;
hshm::ThreadLocalKey chi_task_counter_key_;
hshm::ThreadLocalKey chi_is_client_thread_key_;

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
      node_id);  // replica_id_ starts at 0, unique = major for root tasks
}

Chimaera::~Chimaera() {
  if (is_initialized_) {
    // Finalize server first (stops worker threads that may be processing tasks)
    if (is_runtime_mode_) {
      ServerFinalize();
    }

    // Then finalize client (closes DEALER socket on the shared ZMQ context)
    if (is_client_mode_) {
      ClientFinalize();
    }
  }
}

bool Chimaera::ClientInit() {
  HLOG(kInfo, "Chimaera::ClientInit");
  if (is_client_initialized_ || client_is_initializing_ ||
      runtime_is_initializing_) {
    return true;
  }

  // Set mode flags at the start
  is_client_mode_ = true;
  client_is_initializing_ = true;

  HLOG(kDebug, "IpcManager::ClientInit");
  // Initialize configuration manager
  auto *config_manager = CHI_CONFIG_MANAGER;
  if (!config_manager->Init()) {
    is_client_mode_ = false;
    client_is_initializing_ = false;
    return false;
  }

  HLOG(kDebug, "IpcManager::ClientInit");
  // Initialize IPC manager for client
  auto *ipc_manager = CHI_IPC;
  if (!ipc_manager->ClientInit()) {
    is_client_mode_ = false;
    client_is_initializing_ = false;
    return false;
  }

  // Pool manager is not initialized in client mode
  // It's only needed for server/runtime mode

  // Initialize CHI_ADMIN singleton
  // The admin container is already created by the runtime, so we just
  // construct the admin client directly with the admin pool ID
  HLOG(kDebug, "Initializing CHI_ADMIN singleton");
  // IMPORTANT: Check g_admin directly, NOT CHI_ADMIN macro
  // CHI_ADMIN uses GetGlobalPtrVar which auto-creates with default constructor!
  if (g_admin == nullptr) {
    HLOG(kInfo, "ClientInit: Creating admin client with kAdminPoolId={}",
         chi::kAdminPoolId);
    g_admin = new chimaera::admin::Client(chi::kAdminPoolId);
    HLOG(kInfo, "ClientInit: Admin client created, pool_id_={}",
         g_admin->pool_id_);
  } else {
    HLOG(kInfo, "ClientInit: g_admin already exists, pool_id_={}",
         g_admin->pool_id_);
  }

  is_client_initialized_ = true;
  is_initialized_ = true;
  client_is_initializing_ = false;
  std::atexit(ChimaeraCleanupAtExit);

  return true;
}

bool Chimaera::ServerInit() {
  if (is_runtime_initialized_ || runtime_is_initializing_ ||
      client_is_initializing_) {
    return true;
  }

  // Set mode flags at the start
  is_runtime_mode_ = true;
  runtime_is_initializing_ = true;

  // Initialize configuration manager first
  auto *config_manager = CHI_CONFIG_MANAGER;
  if (!config_manager->Init()) {
    is_runtime_mode_ = false;
    runtime_is_initializing_ = false;
    return false;
  }

  // Initialize IPC manager for server
  auto *ipc_manager = CHI_IPC;
  if (!ipc_manager->ServerInit()) {
    is_runtime_mode_ = false;
    runtime_is_initializing_ = false;
    return false;
  }

  HLOG(kDebug, "Host identification successful: {}",
       ipc_manager->GetCurrentHostname());

  // Initialize module manager first (needed for admin chimod)
  auto *module_manager = CHI_MODULE_MANAGER;
  if (!module_manager->Init()) {
    is_runtime_mode_ = false;
    runtime_is_initializing_ = false;
    return false;
  }

  // Initialize work orchestrator before pool manager
  auto *work_orchestrator = CHI_WORK_ORCHESTRATOR;
  if (!work_orchestrator->Init()) {
    is_runtime_mode_ = false;
    runtime_is_initializing_ = false;
    return false;
  }

  // Start worker threads
  if (!work_orchestrator->StartWorkers()) {
    is_runtime_mode_ = false;
    runtime_is_initializing_ = false;
    return false;
  }

  // Initialize pool manager (server mode only) after work orchestrator
  auto *pool_manager = CHI_POOL_MANAGER;
  if (!pool_manager->ServerInit()) {
    is_runtime_mode_ = false;
    runtime_is_initializing_ = false;
    return false;
  }

  // Process compose section if present
  const auto &compose_config = config_manager->GetComposeConfig();
  if (!compose_config.pools_.empty()) {
    HLOG(kInfo, "Processing compose configuration with {} pools",
         compose_config.pools_.size());

    // Get admin client to process compose
    auto *admin_client = CHI_ADMIN;
    if (!admin_client) {
      HLOG(kError, "Failed to get admin client for compose processing");
      return false;
    }

    // Iterate over each pool configuration and create asynchronously
    for (auto pool_config : compose_config.pools_) {
      // On restart, force restart_=true so containers call Restart() instead of
      // Init()
      if (is_restart_) {
        pool_config.restart_ = true;
      }

      HLOG(kInfo, "Compose: Creating pool {} (module: {}, restart: {})",
           pool_config.pool_name_, pool_config.mod_name_, pool_config.restart_);

      // Create pool asynchronously and wait
      auto task = admin_client->AsyncCompose(pool_config);
      task.Wait();

      // Check return code
      u32 return_code = task->GetReturnCode();
      if (return_code != 0) {
        HLOG(kError,
             "Compose: Failed to create pool {} (module: {}), return code: {}",
             pool_config.pool_name_, pool_config.mod_name_, return_code);
        return false;
      }

      HLOG(kInfo, "Compose: Successfully created pool {} (module: {})",
           pool_config.pool_name_, pool_config.mod_name_);

      // Cleanup task
    }

    HLOG(kInfo, "Compose: All {} pools created successfully",
         compose_config.pools_.size());

    // After compose, replay WAL to recover address table state from before
    // crash
    if (is_restart_) {
      HLOG(kInfo, "Replaying address table WAL for restart recovery...");
      pool_manager->ReplayAddressTableWAL();
    }
  }

  // Start local server last - after all other initialization is complete
  // This ensures clients can connect only when runtime is fully ready
  if (!ipc_manager->StartLocalServer()) {
    HLOG(kError,
         "Failed to start local server - runtime initialization failed");
    is_runtime_mode_ = false;
    runtime_is_initializing_ = false;
    return false;
  }

  is_runtime_initialized_ = true;
  is_initialized_ = true;
  runtime_is_initializing_ = false;
  std::atexit(ChimaeraCleanupAtExit);

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
  is_client_initialized_ = false;
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

  // Reap all shared memory segments before finalizing IPC
  size_t reaped = ipc_manager->WreapAllIpcs();
  if (reaped > 0) {
    HLOG(kInfo, "ServerFinalize: Reaped {} shared memory segments", reaped);
  }

  ipc_manager->ServerFinalize();

  is_runtime_mode_ = false;
  is_runtime_initialized_ = false;
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

}  // namespace chi