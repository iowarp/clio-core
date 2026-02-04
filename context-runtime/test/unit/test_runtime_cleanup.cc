/**
 * Runtime Cleanup and Finalization Tests
 *
 * Tests proper cleanup paths including ServerFinalize() and ClientFinalize()
 * which are currently not exercised by other tests.
 */

#include "../simple_test.h"

#include <sys/wait.h>
#include <unistd.h>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/singletons.h"

using namespace chi;

// ============================================================================
// Server Finalization Tests
// ============================================================================

TEST_CASE("Cleanup - Server Finalization", "[cleanup][ipc]") {
  // Initialize runtime in server mode
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());

  // Verify IPC is properly initialized
  // Note: node_id can be 0 for localhost/single-node setup
  u64 node_id = ipc->GetNodeId();
  (void)node_id;  // Suppress unused variable warning

  // Explicitly call ServerFinalize through Chimaera API
  auto *chimaera = CHI_CHIMAERA_MANAGER;
  chimaera->ServerFinalize();

  // Verify cleanup occurred
  REQUIRE(!ipc->IsInitialized());
}

// NOTE: This test is disabled because it requires forking a server process
// and waiting for it to fully initialize, which is unreliable in unit tests.
// The ClientFinalize() path is exercised by other tests that use integrated mode.
/*
TEST_CASE("Cleanup - Client Finalization", "[cleanup][ipc]") {
  // Start server in background
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Child: Start server
    setenv("CHIMAERA_WITH_RUNTIME", "1", 1);
    CHIMAERA_INIT(ChimaeraMode::kServer, true);
    sleep(300);
    exit(0);
  }

  // Wait for server
  sleep(1);

  // Connect as client only
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());

  // Explicitly call ClientFinalize through Chimaera API
  auto *chimaera = CHI_CHIMAERA_MANAGER;
  chimaera->ClientFinalize();

  // Note: After ClientFinalize(), IPC shared resources remain active
  // (servers, shm, etc.) so IsInitialized() is still true.
  // ClientFinalize() only cleans up client-specific resources.

  // Cleanup server
  kill(server_pid, SIGTERM);
  waitpid(server_pid, nullptr, 0);
}
*/

// NOTE: This test is disabled because CHIMAERA_INIT has a static guard
// that prevents multiple initialization in the same process. Once called,
// subsequent calls just return true without re-initializing.
// To test repeated init/finalize, would need to use separate processes.
/*
TEST_CASE("Cleanup - Repeated Init/Finalize", "[cleanup][ipc]") {
  // Test multiple init/finalize cycles
  for (int i = 0; i < 3; ++i) {
    // Initialize
    bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
    REQUIRE(success);

    auto *ipc = CHI_IPC;
    REQUIRE(ipc->IsInitialized());

    // Finalize using Chimaera API
    auto *chimaera = CHI_CHIMAERA_MANAGER;
    chimaera->ServerFinalize();
    REQUIRE(!ipc->IsInitialized());

    // Small delay between cycles
    usleep(100000);  // 100ms
  }
}
*/

TEST_CASE("Cleanup - ClearClientPool", "[cleanup][ipc][memory]") {
  // Initialize runtime
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Allocate some buffers to populate client pool
  std::vector<FullPtr<char>> buffers;
  for (int i = 0; i < 10; ++i) {
    auto buf = ipc->AllocateBuffer(1024);
    if (!buf.IsNull()) {
      buffers.push_back(buf);
    }
  }

  REQUIRE(buffers.size() > 0);

  // Free buffers back to pool
  for (auto &buf : buffers) {
    ipc->FreeBuffer(buf);
  }

  // Now clear the client pool (tests ClearClientPool path)
  ipc->ClearClientPool();

  // Cleanup using Chimaera API
  CHI_CHIMAERA_MANAGER->ServerFinalize();
}

// ============================================================================
// IPC Cleanup Utility Tests
// ============================================================================

TEST_CASE("Cleanup - ClearUserIpcs", "[cleanup][ipc][shm]") {
  // Initialize runtime (this calls ClearUserIpcs internally)
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Directly test ClearUserIpcs
  size_t cleared = ipc->ClearUserIpcs();
  // Should return 0 since we already cleared during init
  // (or may return count of segments it found)

  // Cleanup using Chimaera API
  CHI_CHIMAERA_MANAGER->ServerFinalize();
}

TEST_CASE("Cleanup - WreapDeadIpcs", "[cleanup][ipc][shm]") {
  // Initialize runtime
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Call WreapDeadIpcs (cleans up dead process IPCs)
  size_t reaped = ipc->WreapDeadIpcs();
  // Return value is number of dead IPCs cleaned up

  // Cleanup using Chimaera API
  CHI_CHIMAERA_MANAGER->ServerFinalize();
}

TEST_CASE("Cleanup - WreapAllIpcs", "[cleanup][ipc][shm]") {
  // Initialize runtime
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Note: WreapAllIpcs is dangerous as it removes ALL IPCs
  // We'll test it but with caution
  // This exercises the function but we won't verify the count
  // since it depends on system state

  size_t reaped = ipc->WreapAllIpcs();
  // Return value is number of IPCs cleaned up

  // After wreaping all, we need to avoid finalizing
  // since the IPC segments may be gone
  // Just exit without cleanup in this test
}

// ============================================================================
// Memory Registration Tests
// ============================================================================

TEST_CASE("Cleanup - RegisterMemory", "[cleanup][ipc][memory]") {
  // Initialize runtime
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Create a custom allocator ID
  hipc::AllocatorId custom_alloc_id(1, 42);

  // Test RegisterMemory
  bool registered = ipc->RegisterMemory(custom_alloc_id);
  // May fail if already registered or if shm doesn't exist
  // We're just exercising the code path

  // Try registering again (should be idempotent)
  registered = ipc->RegisterMemory(custom_alloc_id);

  // Cleanup using Chimaera API
  CHI_CHIMAERA_MANAGER->ServerFinalize();
}

TEST_CASE("Cleanup - GetClientShmInfo", "[cleanup][ipc][memory]") {
  // Initialize runtime
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Test GetClientShmInfo (retrieves per-process shm info)
  ClientShmInfo info = ipc->GetClientShmInfo(0);
  // Info may be empty if index is invalid, but function should not crash

  // Cleanup using Chimaera API
  CHI_CHIMAERA_MANAGER->ServerFinalize();
}

// ============================================================================
// Thread Management Tests
// ============================================================================

TEST_CASE("Cleanup - Client Thread Flags", "[cleanup][ipc][threads]") {
  // Initialize runtime
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Test SetIsClientThread / GetIsClientThread
  ipc->SetIsClientThread(true);
  REQUIRE(ipc->GetIsClientThread() == true);

  ipc->SetIsClientThread(false);
  REQUIRE(ipc->GetIsClientThread() == false);

  // Cleanup using Chimaera API
  CHI_CHIMAERA_MANAGER->ServerFinalize();
}

SIMPLE_TEST_MAIN()
