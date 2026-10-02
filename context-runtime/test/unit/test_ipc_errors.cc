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

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "clio_runtime/admin/admin_tasks.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc/ipc_cpu2cpu.h"
#include "clio_runtime/ipc/ipc_cpu2cpu_impl.h"
#include "clio_runtime/ipc/ipc_cpu2cpu_zmq.h"
#include "clio_runtime/ipc/ipc_cpu2cpu_zmq_impl.h"
#include "clio_runtime/ipc_manager.h"

using namespace clio::run;

// ============================================================================
// Global Setup - Initialize once for all tests
// ============================================================================
static bool InitializeRuntime() {
  static bool initialized = false;
  if (!initialized) {
    bool success = CLIO_INIT(RuntimeMode::kClient, true);
    initialized = success;
    if (success) SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
    return success;
  }
  return true;
}

// ============================================================================
// Connection Error Tests
// ============================================================================

// NOTE: This test is disabled because CLIO_INIT has a static guard
// that prevents multiple initializations in the same process. Once it
// succeeds in any test, it will return true in all subsequent tests.
// This test would need to run in a separate process to work correctly.
/*
TEST_CASE("IpcErrors - Client Connect Without Server", "[ipc][errors]") {
  // Ensure no server is running by clearing any existing IPCs
  // (This may fail if no IPCs exist, which is fine)

  // Try to connect as client when NO server exists
  setenv("CLIO_WITH_RUNTIME", "0", 1);

  // This should timeout and fail gracefully (not crash)
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(!success);

  // Verify IPC manager is not initialized
  // Note: CLIO_IPC may be null or in uninitialized state
  auto *ipc = CLIO_IPC;
  if (ipc != nullptr) {
    REQUIRE(!ipc->IsInitialized());
  }
}
*/

// NOTE: This test is disabled because it tries to call CLIO_INIT
// which can only be called once per process. It would need to run in
// a separate process to work correctly.
/*
TEST_CASE("IpcErrors - Connection Timeout", "[ipc][errors]") {
  // Start a server that will immediately exit (simulating crash)
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Child: Start server then immediately exit
    setenv("CLIO_WITH_RUNTIME", "1", 1);
    CLIO_INIT(RuntimeMode::kServer, true);
    exit(0);  // Exit immediately
  }

  // Wait for server to exit
  int status;
  waitpid(server_pid, &status, 0);

  // Small delay
  usleep(100000);

  // Now try to connect - server is gone
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  bool success = CLIO_INIT(RuntimeMode::kClient, false);

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

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Try to allocate impossibly large buffer
  size_t huge_size = ctp::Unit<size_t>::Terabytes(100);
  auto buf = ipc->AllocateBuffer(huge_size);

  // Should return null pointer, not crash
  REQUIRE(buf.IsNull());

  // Note: Cleanup happens once at end of all tests
}

TEST_CASE("IpcErrors - Zero Size Allocation", "[ipc][errors][memory]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
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

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Try to free null buffer - this should be handled gracefully
  FullPtr<char> null_buf;
  ipc->FreeBuffer(null_buf);

  // Note: We don't test freeing invalid pointers (like 0xDEADBEEF) because
  // that's undefined behavior and will cause segfaults. The allocator can't
  // validate if a pointer is valid before dereferencing it.

  // Note: Cleanup happens once at end of all tests
}

// ============================================================================
// Host/Network Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Invalid Node ID", "[ipc][errors][network]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
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

  auto *ipc = CLIO_IPC;
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

  auto *ipc = CLIO_IPC;
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

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Try to pop from empty network queue across the latency/IO lanes
  Future<Task> future;
  REQUIRE(!ipc->TryPopNetTask(NetQueuePriority::kSendInLatency, future));
  REQUIRE(!ipc->TryPopNetTask(NetQueuePriority::kSendInIO, future));
  REQUIRE(!ipc->TryPopNetTask(NetQueuePriority::kSendOutLatency, future));
  REQUIRE(!ipc->TryPopNetTask(NetQueuePriority::kSendOutIO, future));

  // Note: Cleanup happens once at end of all tests
}

// ============================================================================
// Shared Memory Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Invalid Allocator Registration", "[ipc][errors][shm]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Try to register with invalid allocator ID
  ctp::ipc::AllocatorId invalid_id(0xFFFF, 0xFFFF);
  bool registered = ipc->RegisterMemory(invalid_id);
  // May succeed or fail, but shouldn't crash

  // Note: Cleanup happens once at end of all tests
}

TEST_CASE("IpcErrors - GetClientShmInfo Invalid Index",
          "[ipc][errors][shm]") {
  // Use shared runtime initialization
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
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

  auto *ipc = CLIO_IPC;
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
      bool success = CLIO_INIT(RuntimeMode::kClient, true);

      if (success) {
        auto *ipc = CLIO_IPC;
        if (ipc && ipc->IsInitialized()) {
          // Do some operations
          ipc->GetNodeId();
          ipc->GetNumSchedQueues();

          // Finalize using CLIO Runtime API
          CLIO_RUNTIME_MANAGER->ServerFinalize();
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

// ============================================================================
// Never-Sent Future Tests
// ============================================================================

/**
 * Build the shape the client-side read fast paths hand back: a REAL task,
 * filled in locally, SetComplete()'d, and NEVER Sent. That is the documented
 * "synthesized-task contract" of CoreClient::TryShmGet and its deferred-put /
 * vectored twins, which serve a read straight out of the shared metadata cache
 * and hand the caller an already-complete future so every call site gets the
 * optimization with no special-casing.
 *
 * The defining property is that task_id_.net_key_ is still the TaskId
 * constructor's zero: both SendIn paths stamp it with the task's own heap
 * address, so only a task that never went over IPC can carry a zero.
 */
static clio::run::shared_ptr<clio::run::admin::FlushTask> MakeNeverSentTask(
    IpcManager *ipc, Future<clio::run::admin::FlushTask> *fut,
    ClientOrigin origin) {
  auto task = ipc->NewTask<clio::run::admin::FlushTask>(
      CreateTaskId(), kAdminPoolId, PoolQuery::Local());
  if (task.IsNull()) return task;
  *fut = Future<clio::run::admin::FlushTask>(task->pool_id_, task->method_,
                                             task);
  fut->GetFutureShm()->origin_ = origin;
  task->SetReturnCode(0);
  task->SetComplete();
  return task;
}

/**
 * Regression for the #968 guard vs. the client-side read fast paths.
 *
 * Both RecvOut twins fail a complete task that has no parked response archive,
 * on the reasoning that the only client-path writer of IsComplete() is the
 * recv demux, which always parks the archive first. A synthesized never-Sent
 * future breaks that assumption: it is complete and has no archive, which is
 * exactly the aliasing signature the guard looks for. Before the fix every
 * cache-hit read came back rc = -1 with the bytes already correctly copied.
 *
 * Driving RecvOut directly (rather than through a real cache hit) is
 * deliberate: whether TryShmGet actually hits depends on the machine -- it
 * needs the SHM main segment to map and the metadata cache to attach, which a
 * CI runner or a login node with a small ulimit -v will not do, and there the
 * fast path silently never runs. This exercises the contract everywhere.
 */
TEST_CASE("IpcErrors - never-Sent future needs no response archive",
          "[ipc][errors][968]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  SECTION("SHM RecvOut returns the locally-produced result") {
    Future<clio::run::admin::FlushTask> fut;
    auto task = MakeNeverSentTask(ipc, &fut, ClientOrigin::kClientShm);
    REQUIRE(!task.IsNull());
    // The contract the RecvOut check keys off.
    REQUIRE(task->task_id_.net_key_ == 0);

    // max_sec is a deadline, not a delay: a correct RecvOut returns at once.
    // Before the fix this returned false after logging the #968 error.
    REQUIRE(IpcCpu2Cpu::RecvOut(ipc, fut, 5.0f));
    REQUIRE(task->GetReturnCode() == 0);
  }

  SECTION("ZMQ RecvOut returns the locally-produced result") {
    Future<clio::run::admin::FlushTask> fut;
    auto task = MakeNeverSentTask(ipc, &fut, ClientOrigin::kClientTcp);
    REQUIRE(!task.IsNull());
    REQUIRE(task->task_id_.net_key_ == 0);

    REQUIRE(IpcCpu2CpuZmq::RecvOut(ipc, fut, 5.0f));
    REQUIRE(task->GetReturnCode() == 0);
  }

  SECTION("ZMQ RecvOut does not resend a kClientShm future when the server is "
          "gone") {
    // The synthesized futures carry origin_ == kClientShm, so IpcManager::Recv
    // routes them to the ZMQ twin the moment server_alive_ goes false. That
    // twin's kClientShm head would otherwise push one through
    // WaitForServerAndReconnect + ResendTask -- resending a request the client
    // already satisfied out of its own cache -- so the never-Sent check has to
    // sit AHEAD of that head, not merely ahead of the archive claim.
    Future<clio::run::admin::FlushTask> fut;
    auto task = MakeNeverSentTask(ipc, &fut, ClientOrigin::kClientShm);
    REQUIRE(!task.IsNull());
    REQUIRE(task->task_id_.net_key_ == 0);

    REQUIRE(IpcCpu2CpuZmq::RecvOut(ipc, fut, 5.0f));
    REQUIRE(task->GetReturnCode() == 0);
  }
}

TEST_CASE("IpcErrors - ZZZ Final Cleanup", "[ipc][errors][cleanup]") {
  // This test runs last (ZZZ prefix ensures it's last alphabetically).
  // Force exit to avoid hanging on worker thread joins during finalization.
  // SIMPLE_TEST_PROCESS_EXIT is TerminateProcess on Windows (bypasses the
  // libzmq teardown abort) and ::_exit elsewhere.
  SIMPLE_TEST_PROCESS_EXIT(0);
}

SIMPLE_TEST_MAIN()
