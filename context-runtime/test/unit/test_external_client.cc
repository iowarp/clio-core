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
 * External Client Connection Tests
 *
 * Tests client-only mode connecting to an existing server process.
 * This exercises the ClientInit() code paths that are skipped when using
 * integrated server+client mode.
 */

#include "../simple_test.h"
#include "../runtime_server.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"

using namespace clio::run;

// The runtime server is launched out-of-process via clio::run::test::RuntimeServer
// (clio_run start) instead of fork()+CLIO_INIT(kServer): fork-without-exec
// deadlocks on macOS when the child dlopen()s ChiMods (and nondeterministically
// leaks a port-holding process). The client children below are real fork()s --
// client mode does not dlopen, so they remain macOS-safe.

/**
 * Helper to cleanup leftover shared-memory segment from a prior run.
 */
void CleanupSharedMemory() {
  const char *user = std::getenv("USER");
  std::string memfd_path = std::string("/tmp/clio_") +
                           (user ? user : "unknown") +
                           "/chi_main_segment_" + (user ? user : "");
  unlink(memfd_path.c_str());
}

// ============================================================================
// External Client Connection Tests
// ============================================================================

TEST_CASE("ExternalClient - Basic Connection", "[external_client][ipc]") {
  // Start the runtime daemon out-of-process
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  // Now connect as EXTERNAL CLIENT (not integrated server+client)
  setenv("CLIO_WITH_RUNTIME", "0", 1);  // Force client-only mode
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(success);

  // Verify client initialized successfully
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());

  // Test basic operations from external client
  // Note: node_id 0 is valid in single-node setup (localhost)
  u64 node_id = ipc->GetNodeId();
  (void)node_id;

  // In TCP mode (default), the client does not attach to shared memory
  // so GetTaskQueue() returns nullptr and that is correct behavior
  auto *queue = ipc->GetTaskQueue();
  if (ipc->GetIpcMode() == IpcMode::kShm) {
    REQUIRE(queue != nullptr);
  } else {
    REQUIRE(queue == nullptr);
  }
  // server stopped by RuntimeServer destructor (RAII)
}

TEST_CASE("ExternalClient - Multiple Clients", "[external_client][ipc]") {
  // Start the runtime daemon out-of-process
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  // Start multiple client processes
  const int num_clients = 3;
  pid_t client_pids[num_clients];

  for (int i = 0; i < num_clients; ++i) {
    client_pids[i] = fork();
    if (client_pids[i] == 0) {
      // Suppress child output to prevent log flood
      (void)freopen("/dev/null", "w", stdout);
      (void)freopen("/dev/null", "w", stderr);

      // Child process: Connect as client
      setenv("CLIO_WITH_RUNTIME", "0", 1);
      bool success = CLIO_INIT(RuntimeMode::kClient, false);
      if (!success) {
        _exit(1);
      }

      auto *ipc = CLIO_IPC;
      if (!ipc || !ipc->IsInitialized()) {
        _exit(1);
      }

      // Verify we can get node ID (0 is valid for localhost)
      u64 node_id = ipc->GetNodeId();
      (void)node_id;

      // Client test passed
      _exit(0);
    }
  }

  // Wait for all clients to complete
  bool all_success = true;
  for (int i = 0; i < num_clients; ++i) {
    int status;
    waitpid(client_pids[i], &status, 0);
    if (WEXITSTATUS(status) != 0) {
      all_success = false;
    }
  }

  REQUIRE(all_success);
  // server stopped by RuntimeServer destructor (RAII)
}

TEST_CASE("ExternalClient - Connection Without Server",
          "[external_client][ipc][errors]") {
  // Clean up any leftover shared memory from previous tests
  CleanupSharedMemory();
  // Wait briefly for ports to be freed
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Try to connect as client when NO server exists
  setenv("CLIO_WITH_RUNTIME", "0", 1);

  // This should fail gracefully (not crash)
  // Note: May succeed if a stale server from another test is still running
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  (void)success;  // Just verify it doesn't crash
}

TEST_CASE("ExternalClient - Client Operations", "[external_client][ipc]") {
  // Start the runtime daemon out-of-process
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  // Connect as client
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // In TCP mode (default), num_sched_queues_ is not set so
  // GetNumSchedQueues returns 0. In SHM mode it would be > 0.
  u32 num_queues = ipc->GetNumSchedQueues();
  if (ipc->GetIpcMode() == IpcMode::kShm) {
    REQUIRE(num_queues > 0);
  }

  // Note: GetNumHosts, GetHost, and GetAllHosts are server-only operations.
  // The hostfile_map_ is populated during ServerInit and is NOT shared via
  // shared memory, so external clients cannot access host information.

  // server stopped by RuntimeServer destructor (RAII)
}

SIMPLE_TEST_MAIN()
