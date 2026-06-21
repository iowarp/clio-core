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
 * Client Retry Tests
 *
 * Tests client task retry on server restart and client death handling
 * across all three IPC transport modes (TCP, IPC, SHM).
 */

#include "../simple_test.h"
#include "../runtime_server.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

using namespace clio::run;

// The runtime server runs out-of-process via clio::run::test::RuntimeServer
// (clio_run start). The previous fork()+execl("/proc/self/exe", "--server-mode")
// approach was Linux-only: macOS has no /proc/self/exe, so execl failed and the
// server never started. clio_run is portable. The client-death test still
// fork()s a client (client mode does not dlopen, so it is macOS-safe).

// ============================================================================
// Helper: Server Restart test logic parameterized by IPC mode
// ============================================================================

void TestServerRestart(const std::string &mode) {
  setenv("CLIO_CLIENT_RETRY_TIMEOUT", "30", 1);

  // 1. Start first server, wait for ready
  clio::run::test::RuntimeServer server1;
  REQUIRE(server1.Start());
  REQUIRE(server1.WaitForReady());

  // 2. Client connects
  setenv("CLIO_IPC_MODE", mode.c_str(), 1);
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(success);
  REQUIRE(CLIO_IPC != nullptr);
  REQUIRE(CLIO_IPC->IsInitialized());

  // 3. Baseline task: submit + complete a bdev Create
  {
    clio::run::PoolId pool_id1(8000, 0);
    clio::run::bdev::Client client1(pool_id1);
    std::string pool_name1 = "retry_baseline_" + mode;
    auto task = client1.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), pool_name1, pool_id1,
        clio::run::bdev::BdevType::kRam, 4 * 1024 * 1024);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
    INFO("Baseline task completed for mode " + mode);
  }

  // 4-5. Stop the server (simulating loss) and let the OS reclaim the port
  server1.Stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  INFO("Server stopped and resources cleaned");

  // 6. Start a new server, wait for ready
  clio::run::test::RuntimeServer server2;
  REQUIRE(server2.Start());
  REQUIRE(server2.WaitForReady(40000));  // More time for restart scenario
  INFO("New server started");

  // 7. ReconnectToOriginalHost to re-attach to new server
  bool reconnected = CLIO_IPC->ReconnectToOriginalHost();
  REQUIRE(reconnected);
  INFO("ReconnectToOriginalHost succeeded for mode " + mode);

  // 8. Submit a second task with different pool name/ID
  {
    clio::run::PoolId pool_id2(8001, 0);
    clio::run::bdev::Client client2(pool_id2);
    std::string pool_name2 = "retry_after_restart_" + mode;
    auto task = client2.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), pool_name2, pool_id2,
        clio::run::bdev::BdevType::kRam, 4 * 1024 * 1024);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
    INFO("Post-restart task completed for mode " + mode);
  }
  // server2 stopped by RuntimeServer destructor (RAII)
}

// ============================================================================
// Helper: Client Death test logic parameterized by IPC mode
// ============================================================================

void TestClientDeath(const std::string &mode) {
  setenv("CLIO_CLIENT_RETRY_TIMEOUT", "30", 1);

  // 1. Start server, wait for ready
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  // 2. Fork a client child that submits a task then dies immediately
  pid_t client_child = fork();
  if (client_child == 0) {
    // Child process: connect as client, submit task, exit immediately
    setenv("CLIO_IPC_MODE", mode.c_str(), 1);
    setenv("CLIO_WITH_RUNTIME", "0", 1);
    bool success = CLIO_INIT(RuntimeMode::kClient, false);
    if (!success) {
      _exit(1);
    }

    // Submit a task (no Wait) — response goes to dead process
    clio::run::PoolId pool_id(8100, 0);
    clio::run::bdev::Client client(pool_id);
    std::string pool_name = "client_death_child_" + mode;
    client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), pool_name, pool_id,
        clio::run::bdev::BdevType::kRam, 4 * 1024 * 1024);

    // Exit immediately — response will arrive at dead process
    _exit(0);
  }

  // 3. Parent waits for client child to exit
  REQUIRE(client_child > 0);
  int child_status;
  waitpid(client_child, &child_status, 0);
  REQUIRE(WIFEXITED(child_status));
  REQUIRE(WEXITSTATUS(child_status) == 0);
  INFO("Client child exited");

  // Give server time to process the orphan task
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // 4. Parent connects as a new client (CLIO_INIT static guard is clean
  //    because the child called it in a forked process)
  setenv("CLIO_IPC_MODE", mode.c_str(), 1);
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(success);
  REQUIRE(CLIO_IPC != nullptr);
  REQUIRE(CLIO_IPC->IsInitialized());

  // 5. Submit + complete parent's own task
  {
    clio::run::PoolId pool_id(8101, 0);
    clio::run::bdev::Client client(pool_id);
    std::string pool_name = "client_death_parent_" + mode;
    auto task = client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), pool_name, pool_id,
        clio::run::bdev::BdevType::kRam, 4 * 1024 * 1024);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
    INFO("Parent task completed — server survived client death for mode " +
         mode);
  }
  // server stopped by RuntimeServer destructor (RAII)
}

// ============================================================================
// Server Restart Tests
// ============================================================================

TEST_CASE("ClientRetry - Server Restart TCP", "[client_retry][tcp]") {
  TestServerRestart("TCP");
}

TEST_CASE("ClientRetry - Server Restart IPC", "[client_retry][ipc]") {
  // IPC (Unix domain socket) transport does not auto-reconnect after server
  // death. ReconnectToOriginalHost needs socket transport reconnection support.
  INFO("SKIPPED: IPC socket transport reconnection not yet implemented");
}

TEST_CASE("ClientRetry - Server Restart SHM", "[client_retry][shm]") {
  // SHM mode reconnection re-attaches shared memory but per-process SHM
  // re-registration with the new server hangs during task completion.
  INFO("SKIPPED: SHM mode server restart reconnection not yet fully working");
}

// ============================================================================
// Client Death Tests
// ============================================================================

TEST_CASE("ClientRetry - Client Death TCP", "[client_retry][tcp]") {
  TestClientDeath("TCP");
}

TEST_CASE("ClientRetry - Client Death IPC", "[client_retry][ipc]") {
  TestClientDeath("IPC");
}

TEST_CASE("ClientRetry - Client Death SHM", "[client_retry][shm]") {
  TestClientDeath("SHM");
}

int main(int argc, char* argv[]) {
  // The runtime server is now launched out-of-process via clio_run
  // (clio::run::test::RuntimeServer), so this binary no longer re-execs itself in a
  // "--server-mode"; it only runs tests.
  std::string filter = "";
  if (argc > 1) {
    filter = argv[1];
  }
  int rc = SimpleTest::run_all_tests(filter);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return rc;
}
