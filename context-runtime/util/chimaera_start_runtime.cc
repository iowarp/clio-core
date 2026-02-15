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
 * Chimaera runtime startup utility
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/config_manager.h"
#include "chimaera/admin/admin_client.h"
#include "chimaera/singletons.h"
#include "chimaera/types.h"

namespace {
volatile bool g_keep_running = true;

/**
 * Find and initialize the admin ChiMod
 * Creates a ChiPool for the admin module using PoolManager
 * @return true if successful, false on failure
 */
bool InitializeAdminChiMod() {
  HLOG(kDebug, "Initializing admin ChiMod...");

  // Get the module manager to find the admin chimod
  auto* module_manager = CHI_MODULE_MANAGER;
  if (!module_manager) {
    HLOG(kError, "Module manager not available");
    return false;
  }

  // Check if admin chimod is available
  auto* admin_chimod = module_manager->GetChiMod("chimaera_admin");
  if (!admin_chimod) {
    HLOG(kError, "CRITICAL: Admin ChiMod not found! This is a required system component.");
    return false;
  }

  // Get the pool manager to register the admin pool
  auto* pool_manager = CHI_POOL_MANAGER;
  if (!pool_manager) {
    HLOG(kError, "Pool manager not available");
    return false;
  }

  try {
    // Use PoolManager to create the admin pool
    // This functionality is now handled by PoolManager::ServerInit()
    // which calls CreatePool internally with proper task and RunContext
    // No need to manually create admin pool here anymore
    HLOG(kDebug, "Admin pool creation handled by PoolManager::ServerInit()");

    // Verify the pool was created successfully
    if (!pool_manager->HasPool(chi::kAdminPoolId)) {
      HLOG(kError, "Admin pool creation reported success but pool is not found");
      return false;
    }

    HLOG(kDebug, "Admin ChiPool created successfully (ID: {})", chi::kAdminPoolId);
    return true;

  } catch (const std::exception& e) {
    HLOG(kError, "Exception during admin ChiMod initialization: {}", e.what());
    return false;
  }
}

/**
 * Shutdown the admin ChiMod properly
 */
void ShutdownAdminChiMod() {
  HLOG(kDebug, "Shutting down admin ChiMod...");

  try {
    // Get the pool manager to destroy the admin pool
    auto* pool_manager = CHI_POOL_MANAGER;
    if (pool_manager && pool_manager->HasPool(chi::kAdminPoolId)) {
      // Use PoolManager to destroy the admin pool locally
      if (pool_manager->DestroyLocalPool(chi::kAdminPoolId)) {
        HLOG(kDebug, "Admin pool destroyed successfully");
      } else {
        HLOG(kError, "Failed to destroy admin pool");
      }
    }

  } catch (const std::exception& e) {
    HLOG(kError, "Exception during admin ChiMod shutdown: {}", e.what());
  }

  HLOG(kDebug, "Admin ChiMod shutdown complete");
}

/**
 * Induct this node into an existing cluster by broadcasting AsyncAddNode.
 * The new node's own IP and port are used so all existing nodes register it.
 * @return true on success, false on failure
 */
bool InductNode() {
  auto* ipc_manager = CHI_IPC;
  auto* config = CHI_CONFIG_MANAGER;
  auto* admin_client = CHI_ADMIN;

  std::string my_ip = ipc_manager->GetCurrentHostname();
  chi::u32 my_port = config->GetPort();

  HLOG(kInfo, "Inducting this node ({}:{}) into the cluster...", my_ip, my_port);

  auto task = admin_client->AsyncAddNode(
      chi::PoolQuery::Broadcast(), my_ip, my_port);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    HLOG(kError, "Failed to induct node: {}", task->error_message_.str());
    return false;
  }

  HLOG(kInfo, "Node inducted successfully as node_id={}", task->new_node_id_);
  return true;
}

void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [--induct]\n";
  std::cout << "  Starts the Chimaera runtime server\n";
  std::cout << "  --induct: Register this node with all existing cluster nodes\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  // Parse arguments
  bool induct = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--induct") == 0) {
      induct = true;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      PrintUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  HLOG(kDebug, "Starting Chimaera runtime...");

  // Initialize Chimaera runtime
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kRuntime, true)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }

  HLOG(kDebug, "Chimaera runtime started successfully");

  // Find and initialize admin ChiMod
  if (!InitializeAdminChiMod()) {
    HLOG(kError, "FATAL ERROR: Failed to find or initialize admin ChiMod");
    return 1;
  }

  HLOG(kDebug, "Admin ChiMod initialized successfully with pool ID {}", chi::kAdminPoolId);

  // Induct this node into the cluster if requested
  if (induct) {
    if (!InductNode()) {
      HLOG(kError, "FATAL ERROR: Failed to induct node into cluster");
      return 1;
    }
  }

  // Main runtime loop
  while (g_keep_running) {
    // Sleep for a short period
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  HLOG(kDebug, "Shutting down Chimaera runtime...");

  // Shutdown admin pool first
  ShutdownAdminChiMod();

  HLOG(kDebug, "Chimaera runtime stopped (finalization will happen automatically)");
  return 0;
}