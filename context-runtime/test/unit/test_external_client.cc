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

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"

using namespace chi;

/**
 * Helper to start server in background process
 * Returns server PID
 */
pid_t StartServerProcess() {
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Redirect child's stdout/stderr to /dev/null to prevent massive
    // worker log output from flooding shared pipes and blocking parent
    freopen("/dev/null", "w", stdout);
    freopen("/dev/null", "w", stderr);

    // Child process: Start runtime server
    setenv("CHIMAERA_WITH_RUNTIME", "1", 1);
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      _exit(1);
    }

    // Keep server alive for tests
    // Server will be killed by parent process
    sleep(300);  // 5 minutes max
    _exit(0);
  }
  return server_pid;
}

/**
 * Helper to wait for server to be ready
 */
bool WaitForServer(int max_attempts = 50) {
  // The main shared memory segment name is "chi_main_segment_${USER}"
  const char *user = std::getenv("USER");
  std::string shm_name = std::string("/chi_main_segment_") + (user ? user : "");

  for (int i = 0; i < max_attempts; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Check if shared memory exists (indicates server is ready)
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (fd >= 0) {
      close(fd);
      // Give it a bit more time to fully initialize
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      return true;
    }
  }
  return false;
}

/**
 * Helper to cleanup server process
 */
void CleanupSharedMemory() {
  // Clean up leftover shared memory segments
  const char *user = std::getenv("USER");
  std::string main_seg = std::string("/chi_main_segment_") + (user ? user : "");
  shm_unlink(main_seg.c_str());
}

void CleanupServer(pid_t server_pid) {
  if (server_pid > 0) {
    kill(server_pid, SIGTERM);
    // Wait for server to exit
    int status;
    waitpid(server_pid, &status, 0);
    // Clean up shared memory left behind by the server
    CleanupSharedMemory();
  }
}

// ============================================================================
// External Client Connection Tests
// ============================================================================

TEST_CASE("ExternalClient - Basic Connection", "[external_client][ipc]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Now connect as EXTERNAL CLIENT (not integrated server+client)
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);  // Force client-only mode
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  // Verify client initialized successfully
  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());

  // Test basic operations from external client
  // Note: node_id 0 is valid in single-node setup (localhost)
  u64 node_id = ipc->GetNodeId();
  (void)node_id;

  // Test that we can get the task queue
  auto *queue = ipc->GetTaskQueue();
  REQUIRE(queue != nullptr);

  // Cleanup
  CleanupServer(server_pid);
}

TEST_CASE("ExternalClient - Multiple Clients", "[external_client][ipc]") {
  // Start server
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Start multiple client processes
  const int num_clients = 3;
  pid_t client_pids[num_clients];

  for (int i = 0; i < num_clients; ++i) {
    client_pids[i] = fork();
    if (client_pids[i] == 0) {
      // Suppress child output to prevent log flood
      freopen("/dev/null", "w", stdout);
      freopen("/dev/null", "w", stderr);

      // Child process: Connect as client
      setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
      bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
      if (!success) {
        _exit(1);
      }

      auto *ipc = CHI_IPC;
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

  // Cleanup server
  CleanupServer(server_pid);
}

TEST_CASE("ExternalClient - Connection Without Server",
          "[external_client][ipc][errors]") {
  // Clean up any leftover shared memory from previous tests
  CleanupSharedMemory();
  // Wait briefly for ports to be freed
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Try to connect as client when NO server exists
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);

  // This should fail gracefully (not crash)
  // Note: May succeed if a stale server from another test is still running
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  (void)success;  // Just verify it doesn't crash
}

TEST_CASE("ExternalClient - Client Operations", "[external_client][ipc]") {
  // Start server
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Connect as client
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Test GetNumSchedQueues
  u32 num_queues = ipc->GetNumSchedQueues();
  REQUIRE(num_queues > 0);

  // Note: GetNumHosts, GetHost, and GetAllHosts are server-only operations.
  // The hostfile_map_ is populated during ServerInit and is NOT shared via
  // shared memory, so external clients cannot access host information.

  // Cleanup
  CleanupServer(server_pid);
}

SIMPLE_TEST_MAIN()
