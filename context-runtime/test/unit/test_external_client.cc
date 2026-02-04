/**
 * External Client Connection Tests
 *
 * Tests client-only mode connecting to an existing server process.
 * This exercises the ClientInit() code paths that are skipped when using
 * integrated server+client mode.
 */

#include "../simple_test.h"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
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
    // Child process: Start runtime server
    setenv("CHIMAERA_WITH_RUNTIME", "1", 1);
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      exit(1);
    }

    // Keep server alive for tests
    // Server will be killed by parent process
    sleep(300);  // 5 minutes max
    exit(0);
  }
  return server_pid;
}

/**
 * Helper to wait for server to be ready
 */
bool WaitForServer(int max_attempts = 20) {
  for (int i = 0; i < max_attempts; ++i) {
    // Try to connect (this will create a test connection)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check if shared memory exists (indicates server is ready)
    std::string shm_name = "/chimaera_main_segment";
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (fd >= 0) {
      close(fd);
      // Give it a bit more time to fully initialize
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      return true;
    }
  }
  return false;
}

/**
 * Helper to cleanup server process
 */
void CleanupServer(pid_t server_pid) {
  if (server_pid > 0) {
    kill(server_pid, SIGTERM);
    // Wait for server to exit
    int status;
    waitpid(server_pid, &status, 0);
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
  u64 node_id = ipc->GetNodeId();
  REQUIRE(node_id != 0);

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
      // Child process: Connect as client
      setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
      bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
      if (!success) {
        exit(1);
      }

      auto *ipc = CHI_IPC;
      if (!ipc || !ipc->IsInitialized()) {
        exit(1);
      }

      // Verify we can get node ID
      u64 node_id = ipc->GetNodeId();
      if (node_id == 0) {
        exit(1);
      }

      // Client test passed
      exit(0);
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
  // Try to connect as client when NO server exists
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);

  // This should fail gracefully (not crash)
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(!success);
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

  // Test GetNumHosts
  size_t num_hosts = ipc->GetNumHosts();
  REQUIRE(num_hosts >= 1);  // At least localhost

  // Test GetHost
  u64 node_id = ipc->GetNodeId();
  const Host *host = ipc->GetHost(node_id);
  REQUIRE(host != nullptr);
  REQUIRE(host->node_id == node_id);

  // Test GetAllHosts
  const std::vector<Host> &hosts = ipc->GetAllHosts();
  REQUIRE(hosts.size() >= 1);

  // Cleanup
  CleanupServer(server_pid);
}

SIMPLE_TEST_MAIN()
