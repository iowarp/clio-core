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
 * IPC Error Handling Tests
 *
 * Tests error conditions and failure paths in IpcManager that are not
 * exercised by happy-path tests.
 */

#include "../simple_test.h"

#include <sys/wait.h>
#include <unistd.h>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"

using namespace chi;

// ============================================================================
// Global Setup - Initialize once for all tests
// ============================================================================
static bool InitializeRuntime() {
  static bool initialized = false;
  if (!initialized) {
    bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
    initialized = success;
    return success;
  }
  return true;
}

// ============================================================================
// Connection Error Tests
// ============================================================================

// NOTE: This test is disabled because CHIMAERA_INIT has a static guard
// that prevents multiple initializations in the same process. Once it
// succeeds in any test, it will return true in all subsequent tests.
// This test would need to run in a separate process to work correctly.
/*
TEST_CASE("IpcErrors - Client Connect Without Server", "[ipc][errors]") {
  // Ensure no server is running by clearing any existing IPCs
  // (This may fail if no IPCs exist, which is fine)

  // Try to connect as client when NO server exists
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);

  // This should timeout and fail gracefully (not crash)
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(!success);

  // Verify IPC manager is not initialized
  // Note: CHI_IPC may be null or in uninitialized state
  auto *ipc = CHI_IPC;
  if (ipc != nullptr) {
    REQUIRE(!ipc->IsInitialized());
  }
}
*/

// NOTE: This test is disabled because it tries to call CHIMAERA_INIT
// which can only be called once per process. It would need to run in
// a separate process to work correctly.
/*
TEST_CASE("IpcErrors - Connection Timeout", "[ipc][errors]") {
  // Start a server that will immediately exit (simulating crash)
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Child: Start server then immediately exit
    setenv("CHIMAERA_WITH_RUNTIME", "1", 1);
    CHIMAERA_INIT(ChimaeraMode::kServer, true);
    exit(0);  // Exit immediately
  }

  // Wait for server to exit
  int status;
  waitpid(server_pid, &status, 0);

  // Small delay
  usleep(100000);

  // Now try to connect - server is gone
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);

  // May succeed or fail depending on timing and leftover shm
  // The important thing is it doesn't crash
}
*/

// ============================================================================
// Memory Allocation Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Huge Buffer Allocation", "[ipc][errors][memory]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to allocate impossibly large buffer
  size_t huge_size = hshm::Unit<size_t>::Terabytes(100);
  auto buf = ipc->AllocateBuffer(huge_size);

  // Should return null pointer, not crash
  REQUIRE(buf.IsNull());

  // Note: Cleanup happens once at end of all tests
}

TEST_CASE("IpcErrors - Zero Size Allocation", "[ipc][errors][memory]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to allocate zero-size buffer
  auto buf = ipc->AllocateBuffer(0);

  // Should handle gracefully (either null or valid pointer)
  // The important thing is it doesn't crash

  // If we got a buffer, free it
  if (!buf.IsNull()) {
    ipc->FreeBuffer(buf);
  }

  // Note: Cleanup happens once at end of all tests
}

TEST_CASE("IpcErrors - Invalid Buffer Free", "[ipc][errors][memory]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to free null buffer - this should be handled gracefully
  FullPtr<char> null_buf;
  ipc->FreeBuffer(null_buf);

  // Note: We don't test freeing invalid pointers (like 0xDEADBEEF) because
  // that's undefined behavior and will cause segfaults. The allocator can't
  // validate if a pointer is valid before dereferencing it.

  // Note: Cleanup happens once at end of all tests
}

TEST_CASE("IpcErrors - Memory Increase Invalid Size", "[ipc][errors][memory]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to increase memory by 0
  // Note: IncreaseMemory(0) actually succeeds because 32MB metadata overhead
  // is always added, creating a valid 32MB shared memory segment.
  bool result = ipc->IncreaseMemory(0);
  // Just verify it doesn't crash; it may succeed due to overhead allocation

  // Try to increase by huge amount (should fail)
  result = ipc->IncreaseMemory(hshm::Unit<size_t>::Terabytes(100));
  REQUIRE(!result);

  // Note: Cleanup happens once at end of all tests
}

// ============================================================================
// Host/Network Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Invalid Node ID", "[ipc][errors][network]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to get host with invalid node ID
  auto *host = ipc->GetHost(0xDEADBEEF);
  REQUIRE(host == nullptr);

  // Note: GetHost(0) returns a valid host in single-node setup (node_id 0 =
  // localhost), so we use a different large invalid ID instead.
  host = ipc->GetHost(0xFFFFFFFF);
  REQUIRE(host == nullptr);

  // Note: Cleanup happens once at end of all tests
}

TEST_CASE("IpcErrors - Invalid IP Address", "[ipc][errors][network]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to get host by invalid IP
  auto *host = ipc->GetHostByIp("999.999.999.999");
  REQUIRE(host == nullptr);

  // Try empty IP
  host = ipc->GetHostByIp("");
  REQUIRE(host == nullptr);

  // Try malformed IP
  host = ipc->GetHostByIp("not.an.ip.address");
  REQUIRE(host == nullptr);

  // Note: Cleanup happens once at end of all tests
}

TEST_CASE("IpcErrors - Network Client Creation Failure",
          "[ipc][errors][network]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to create client with invalid address
  // TransportFactory::GetClient may throw for invalid protocols
  try {
    auto *client = ipc->GetOrCreateClient("invalid://address", 0);
    // May return nullptr or valid client depending on implementation
    (void)client;
  } catch (const std::exception &e) {
    // Exception is acceptable for invalid address
  }

  // Note: Cleanup happens once at end of all tests
}

// ============================================================================
// Queue Operation Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Network Queue Operations", "[ipc][errors][queue]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to pop from empty network queue
  Future<Task> future;
  bool result = ipc->TryPopNetTask(NetQueuePriority::kSendIn, future);
  REQUIRE(!result);  // Should return false for empty queue

  // Try with different priority
  result = ipc->TryPopNetTask(NetQueuePriority::kSendOut, future);
  REQUIRE(!result);

  // Note: Cleanup happens once at end of all tests
}

// ============================================================================
// Shared Memory Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Invalid Allocator Registration", "[ipc][errors][shm]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to register with invalid allocator ID
  hipc::AllocatorId invalid_id(0xFFFF, 0xFFFF);
  bool registered = ipc->RegisterMemory(invalid_id);
  // May succeed or fail, but shouldn't crash

  // Note: Cleanup happens once at end of all tests
}

TEST_CASE("IpcErrors - GetClientShmInfo Invalid Index",
          "[ipc][errors][shm]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Try to get info with invalid index
  ClientShmInfo info = ipc->GetClientShmInfo(9999);
  // Should return empty/default info, not crash

  // Note: Cleanup happens once at end of all tests
}

// ============================================================================
// Scheduler Error Tests
// ============================================================================

TEST_CASE("IpcErrors - SetNumSchedQueues Edge Cases", "[ipc][errors][sched]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);

  // Get current value
  u32 original = ipc->GetNumSchedQueues();
  REQUIRE(original > 0);

  // Try to set to 0 (should be rejected or handled gracefully)
  ipc->SetNumSchedQueues(0);

  // Try to set to huge value
  ipc->SetNumSchedQueues(1000000);

  // Restore original (if possible)
  ipc->SetNumSchedQueues(original);

  // Note: Cleanup happens once at end of all tests
}

// ============================================================================
// Multi-Process Error Tests
// ============================================================================

// NOTE: This test is disabled because it forks multiple processes that each
// try to start a full runtime (workers, modules, servers). This is very heavy
// and causes timeouts. It's better suited for integration tests.
/*
TEST_CASE("IpcErrors - Concurrent Init/Finalize", "[ipc][errors][multiproc]") {
  // Test concurrent initialization attempts
  const int num_procs = 3;
  pid_t pids[num_procs];

  for (int i = 0; i < num_procs; ++i) {
    pids[i] = fork();
    if (pids[i] == 0) {
      // Child: Try to initialize
      bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);

      if (success) {
        auto *ipc = CHI_IPC;
        if (ipc && ipc->IsInitialized()) {
          // Do some operations
          ipc->GetNodeId();
          ipc->GetNumSchedQueues();

          // Finalize using Chimaera API
          CHI_CHIMAERA_MANAGER->ServerFinalize();
        }
        exit(0);
      } else {
        exit(1);
      }
    }
  }

  // Wait for all children
  int success_count = 0;
  for (int i = 0; i < num_procs; ++i) {
    int status;
    waitpid(pids[i], &status, 0);
    if (WEXITSTATUS(status) == 0) {
      success_count++;
    }
  }

  // At least one should succeed, others may fail due to race
  REQUIRE(success_count >= 1);
}
*/

// ============================================================================
// Global Cleanup - Finalize once at the end
// ============================================================================

TEST_CASE("IpcErrors - ZZZ Final Cleanup", "[ipc][errors][cleanup]") {
  // This test runs last (ZZZ prefix ensures it's last alphabetically)
  // and cleans up the shared runtime
  CHI_CHIMAERA_MANAGER->ServerFinalize();
}

SIMPLE_TEST_MAIN()
