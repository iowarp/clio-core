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

#ifndef CLIO_RUN_MANAGER_H_
#define CLIO_RUN_MANAGER_H_

#include "clio_runtime/api.h"
#include "clio_runtime/types.h"
#include <atomic>
#include <memory>
#include <string>

namespace clio::run {

/**
 * Main CLIO Runtime manager singleton class
 *
 * Central coordinator for the distributed task execution framework.
 * Manages initialization and coordination between client and runtime modes.
 * Uses CTP global pointer variable singleton pattern.
 */
class RuntimeManager {
 public:
  /**
   * Destructor - handles automatic finalization
   */
  ~RuntimeManager();

  /**
   * Initialize client components
   * @return true if initialization successful, false otherwise
   */
  bool ClientInit();

  /**
   * Initialize server/runtime components
   * @return true if initialization successful, false otherwise
   */
  bool ServerInit();


  /**
   * Finalize client components only
   */
  void ClientFinalize();

  /**
   * Finalize server/runtime components only
   */
  void ServerFinalize();

  /**
   * Graceful-shutdown drain: wait (bounded) for all in-flight non-periodic
   * tasks to complete and the cross-node/client net queues to empty before the
   * workers are stopped and containers torn down. Periodic tasks are excluded
   * from the work count (see Worker::StartCoroutine), so this converges instead
   * of spinning on monitoring tasks. Called at the start of ServerFinalize.
   * @param timeout_ms hard cap; logs and returns if work hasn't drained by then
   */
  void DrainPendingTasks(u64 timeout_ms = 5000);

  /**
   * Check if CLIO Runtime is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const;

  /**
   * Check if running in client mode
   * @return true if client mode, false otherwise
   */
  bool IsClient() const;

  /**
   * Check if running in runtime/server mode
   * @return true if runtime mode, false otherwise
   */
  bool IsRuntime() const;

  /**
   * Get the current hostname identified during initialization
   * @return hostname string
   */
  const std::string& GetCurrentHostname() const;

  /**
   * Get the 64-bit node ID stored in shared memory
   * @return 64-bit node ID, 0 if not available
   */
  u64 GetNodeId() const;

  /**
   * Check if CLIO Runtime is currently in the process of initializing
   * @return true if either client or runtime initialization is in progress, false otherwise
   */
  bool IsInitializing() const;

  /** How a requested runtime stop should be carried out */
  enum class StopMode {
    kGraceful, /**< full teardown (ServerFinalize) guarded by a watchdog */
    kForce     /**< immediate death: unlink artifacts, then _exit(0) */
  };

  /**
   * Request an asynchronous shutdown of this runtime process.
   *
   * Safe to call from any thread, including a worker executing a task
   * (the admin StopRuntime handler): it never tears down state inline.
   * kGraceful sets a flag the runtime main loop polls (see IsStopRequested)
   * so the proven normal-exit path (atexit -> ServerFinalize) runs on the
   * main thread, and spawns a detached watchdog that force-exits the process
   * (exit code 2, artifacts unlinked) if teardown wedges past
   * grace_period_ms plus a fixed margin. kForce spawns a detached thread
   * that briefly waits for the task ack to flush, unlinks this runtime's
   * filesystem artifacts, and _exit(0)s without running atexit handlers.
   * Re-entrant calls are ignored; no-op unless in runtime mode.
   *
   * @param mode graceful (default stop) or force (stop --force)
   * @param grace_period_ms budget for draining in-flight tasks
   */
  void RequestStop(StopMode mode, u32 grace_period_ms);

  /**
   * Check whether a graceful stop has been requested via RequestStop.
   * Polled by the runtime main loop (clio_run start/restart).
   * @return true if a stop was requested
   */
  bool IsStopRequested() const;

 public:
  bool is_restart_ = false;  /**< If true, force restart on compose pools and replay WAL */

 private:
  bool is_initialized_ = false;
  bool is_client_mode_ = false;
  bool is_runtime_mode_ = false;
  bool is_client_initialized_ = false;
  bool is_runtime_initialized_ = false;
  bool client_is_initializing_ = false;
  bool runtime_is_initializing_ = false;

  std::atomic<bool> stop_requested_{false};    /**< set once by RequestStop */
  std::atomic<bool> finalize_complete_{false}; /**< ServerFinalize finished */
  std::atomic<u32> stop_grace_period_ms_{5000}; /**< drain budget for stop */
};

}  // namespace clio::run

// Global pointer variable declaration for CLIO Runtime manager singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_H(clio::run::RuntimeManager, g_runtime_manager);

// Macro for accessing the CLIO Runtime manager singleton using global pointer variable
#define CLIO_RUNTIME_MANAGER \
  CTP_GET_GLOBAL_PTR_VAR(::clio::run::RuntimeManager, g_runtime_manager)

#endif  // CLIO_RUN_MANAGER_H_
