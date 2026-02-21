#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include "chimaera/admin/admin_client.h"
#include "chimaera/chimaera.h"
#include "chimaera/pool_query.h"
#include "chimaera/types.h"
#include "chimaera_commands.h"

int RuntimeStop(int argc, char* argv[]) {
  HLOG(kDebug, "Stopping Chimaera runtime...");

  try {
    HLOG(kDebug, "Initializing Chimaera client...");
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
      HLOG(kError, "Failed to initialize Chimaera client components");
      return 1;
    }

    // RAII guard: call ClientFinalize() on every return path so the background
    // ZMQ receive thread is joined and the DEALER socket is closed before the
    // ZMQ shared-context static destructor runs.
    struct ClientFinalizeGuard {
      ~ClientFinalizeGuard() {
        auto* mgr = CHI_CHIMAERA_MANAGER;
        if (mgr) {
          mgr->ClientFinalize();
        }
      }
    } finalize_guard;

    HLOG(kDebug, "Creating admin client connection...");
    chimaera::admin::Client admin_client(chi::kAdminPoolId);

    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager || !ipc_manager->IsInitialized()) {
      HLOG(kError, "IPC manager not available - is Chimaera runtime running?");
      return 1;
    }

    // Note: TaskQueue is only accessible in SHM mode. In TCP mode (used for
    // distributed/Docker deployments), the task queue is not mapped into the
    // client process. Skip the task queue validation in that case and rely on
    // the ZMQ transport to deliver the stop task to the runtime.
    auto* task_queue = ipc_manager->GetTaskQueue();
    if (task_queue) {
      try {
        chi::u32 num_lanes = task_queue->GetNumLanes();
        if (num_lanes == 0) {
          HLOG(kError, "TaskQueue has no lanes configured - runtime initialization incomplete");
          return 1;
        }
        HLOG(kDebug, "TaskQueue validated with {} lanes", num_lanes);
      } catch (const std::exception& e) {
        HLOG(kError, "TaskQueue validation failed: {}", e.what());
        return 1;
      }
    } else {
      HLOG(kDebug, "TaskQueue not in shared memory (TCP mode) - sending stop via ZMQ transport");
    }

    chi::PoolQuery pool_query;
    chi::u32 shutdown_flags = 0;
    chi::u32 grace_period_ms = 5000;

    // Parse --grace-period flag
    for (int i = 0; i < argc; ++i) {
      if (std::strcmp(argv[i], "--grace-period") == 0 && i + 1 < argc) {
        grace_period_ms = static_cast<chi::u32>(std::atoi(argv[++i]));
        if (grace_period_ms == 0) grace_period_ms = 5000;
      }
    }

    HLOG(kDebug, "Sending stop runtime task to admin pool (grace period: {}ms)...", grace_period_ms);

    auto start_time = std::chrono::steady_clock::now();

    chi::Future<chimaera::admin::StopRuntimeTask> stop_task;
    try {
      stop_task = admin_client.AsyncStopRuntime(pool_query, shutdown_flags, grace_period_ms);
      if (stop_task.IsNull()) {
        HLOG(kError, "Failed to create stop runtime task - runtime may not be running");
        return 1;
      }
    } catch (const std::exception& e) {
      HLOG(kError, "Error creating stop runtime task: {}", e.what());
      return 1;
    }

    HLOG(kDebug, "Stop runtime task submitted successfully (fire-and-forget)");

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time).count();

    HLOG(kDebug, "Runtime stop task submitted in {}ms", duration);
    return 0;

  } catch (const std::exception& e) {
    HLOG(kError, "Error stopping runtime: {}", e.what());
    return 1;
  }
}
