/**
 * Chimaera runtime shutdown utility
 *
 * Connects to the running Chimaera runtime and sends a StopRuntimeTask
 * via the admin ChiMod client to initiate graceful shutdown.
 */

#include <chrono>
#include <iostream>
#include <thread>

#include "chimaera/admin/admin_client.h"
#include "chimaera/chimaera.h"
#include "chimaera/pool_query.h"
#include "chimaera/types.h"

int main(int argc, char* argv[]) {
  HILOG(kDebug, "Stopping Chimaera runtime...");

  try {
    // Initialize Chimaera client components
    HILOG(kDebug, "Initializing Chimaera client...");
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
      HELOG(kError, "Failed to initialize Chimaera client components");
      return 1;
    }

    HILOG(kDebug, "Creating admin client connection...");
    // Create admin client connected to admin pool
    chimaera::admin::Client admin_client(chi::kAdminPoolId);

    // Check if IPC manager is available
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager || !ipc_manager->IsInitialized()) {
      HELOG(kError, "IPC manager not available - is Chimaera runtime running?");
      return 1;
    }

    // Additional validation: check if TaskQueue is accessible
    auto* task_queue = ipc_manager->GetTaskQueue();
    if (!task_queue || task_queue->IsNull()) {
      HELOG(kError, "TaskQueue not available - runtime may not be properly initialized");
      return 1;
    }

    // Validate that task queue has valid lane configuration
    try {
      chi::u32 num_lanes = task_queue->GetNumLanes();
      if (num_lanes == 0) {
        HELOG(kError, "TaskQueue has no lanes configured - runtime initialization incomplete");
        return 1;
      }
      HILOG(kDebug, "TaskQueue validated with {} lanes", num_lanes);
    } catch (const std::exception& e) {
      HELOG(kError, "TaskQueue validation failed: {}", e.what());
      return 1;
    }

    // Create domain query for local execution
    chi::PoolQuery pool_query;

    // Parse command line arguments for shutdown parameters
    chi::u32 shutdown_flags = 0;
    chi::u32 grace_period_ms = 5000;  // Default 5 seconds

    if (argc >= 2) {
      grace_period_ms = static_cast<chi::u32>(std::atoi(argv[1]));
      if (grace_period_ms == 0) {
        grace_period_ms = 5000;  // Fallback to default
      }
    }

    HILOG(kDebug, "Sending stop runtime task to admin pool (grace period: {}ms)...", grace_period_ms);

    // Send StopRuntimeTask via admin client
    HILOG(kDebug, "Calling admin client AsyncStopRuntime...");
    auto start_time = std::chrono::steady_clock::now();

    // Use the admin client's AsyncStopRuntime method - fire and forget
    hipc::FullPtr<chimaera::admin::StopRuntimeTask> stop_task;
    try {
      stop_task = admin_client.AsyncStopRuntime(
          HSHM_MCTX, pool_query, shutdown_flags, grace_period_ms);
      if (stop_task.IsNull()) {
        HELOG(kError, "Failed to create stop runtime task - runtime may not be running");
        return 1;
      }
    } catch (const std::exception& e) {
      HELOG(kError, "Error creating stop runtime task: {}", e.what());
      return 1;
    }

    HILOG(kDebug, "Stop runtime task submitted successfully (fire-and-forget)");

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    HILOG(kDebug, "Runtime stop task submitted in {}ms", duration);

    return 0;

  } catch (const std::exception& e) {
    HELOG(kError, "Error stopping runtime: {}", e.what());
    return 1;
  }
}