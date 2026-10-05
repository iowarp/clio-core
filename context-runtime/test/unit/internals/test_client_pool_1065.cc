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
 * Regression tests for issue #1065: the runtime never released a client's
 * dial-back connection.
 *
 * Every TCP client left a DEALER in IpcManager::client_pool_ for the life of
 * the daemon, and EvictClientByIdentity only dropped a raw-pointer lookaside,
 * so the 1024th distinct client hit ZeroMQ's default ZMQ_MAX_SOCKETS (1023):
 * zmq_socket() returned NULL, dial-back failed with "not a socket", and that
 * client's response was dropped. Dial-backs now live in an LRU-bounded table
 * whose entries are shared with the in-flight responses that use them.
 *
 * These run with ZeroMQ's default socket limit, so they fail if the table is
 * unbounded again, and they check that eviction never frees a connection an
 * in-flight response still holds.
 */

#include "simple_test.h"

#include <iostream>
#include <memory>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/singletons.h>

namespace {

/** Distinct fake clients to dial back; above libzmq's 1023-socket default. */
constexpr int kAttempts = 1100;
/** First fake client port; client i uses kBasePort + i. */
constexpr int kBasePort = 24000;

bool g_initialized = false;

/**
 * Start an in-process runtime once for this executable and register its
 * finalizer with the test framework.
 */
void EnsureInitialized() {
  if (!g_initialized) {
    clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    g_initialized = true;
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  }
}

/**
 * Build a fake dial-back identity unique to this test.
 * @param tag distinguishes the test case
 * @param i client index
 * @return identity string
 */
std::string FakeIdentity(const char *tag, int i) {
  return std::string("repro1065-") + tag + ":" + std::to_string(i);
}

}  // namespace

TEST_CASE("ClientPool - more distinct clients than ZMQ_MAX_SOCKETS (#1065)",
          "[ipc][clientpool][1065]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Each client is served and its response sent, so nothing outside the
  // table holds the connection -- the pattern of a long-lived daemon serving
  // many short-lived clients. No listener is needed: a TCP DEALER connects
  // asynchronously, so failure here can only be socket exhaustion.
  int failures = 0;
  int first_failure = -1;
  for (int i = 0; i < kAttempts; ++i) {
    auto t = ipc->GetOrCreateClientByIdentity(FakeIdentity("many", i),
                                              "127.0.0.1", kBasePort + i);
    if (!t) {
      ++failures;
      if (first_failure < 0) first_failure = i;
    }
  }
  std::cout << "[#1065] " << kAttempts << " distinct clients: " << failures
            << " failed (first at " << first_failure << "), "
            << ipc->GetClientDialBackCount() << " dial-backs cached (max "
            << ipc->GetMaxClientDialBacks() << ")" << std::endl;
  REQUIRE(failures == 0);
  REQUIRE(ipc->GetClientDialBackCount() <= ipc->GetMaxClientDialBacks());
}

TEST_CASE("ClientPool - eviction keeps an in-flight response's transport",
          "[ipc][clientpool][1065]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  const std::string held_id = FakeIdentity("held", 0);
  const int held_port = kBasePort + kAttempts + 10;
  std::shared_ptr<ctp::lbm::Transport> held =
      ipc->GetOrCreateClientByIdentity(held_id, "127.0.0.1", held_port);
  REQUIRE(held != nullptr);
  REQUIRE(held.use_count() >= 2);  // the table and this in-flight holder

  // Push it out of the table with enough newer clients to fill the cap.
  const size_t cap = ipc->GetMaxClientDialBacks();
  for (size_t i = 0; i <= cap; ++i) {
    int port = kBasePort + 2 * kAttempts + static_cast<int>(i);
    REQUIRE(ipc->GetOrCreateClientByIdentity(
                FakeIdentity("churn", static_cast<int>(i)), "127.0.0.1",
                port) != nullptr);
  }

  // Evicted from the table, but alive: only the in-flight holder owns it.
  REQUIRE(held.use_count() == 1);
  // The same client asking again gets a freshly dialed connection.
  auto again = ipc->GetOrCreateClientByIdentity(held_id, "127.0.0.1",
                                                held_port);
  REQUIRE(again != nullptr);
  REQUIRE(again.get() != held.get());
  held.reset();  // the response is sent; the old socket closes now
}

TEST_CASE("ClientPool - EvictClientByIdentity releases the connection",
          "[ipc][clientpool][1065]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  const std::string id = FakeIdentity("evict", 0);
  const int port = kBasePort + 4 * kAttempts;
  std::weak_ptr<ctp::lbm::Transport> watch =
      ipc->GetOrCreateClientByIdentity(id, "127.0.0.1", port);
  REQUIRE(!watch.expired());  // the table still holds it
  size_t before = ipc->GetClientDialBackCount();

  ipc->EvictClientByIdentity(id, port);
  REQUIRE(ipc->GetClientDialBackCount() == before - 1);
  REQUIRE(watch.expired());  // nothing else held it: the socket is closed

  // Evicting an unknown client is a no-op.
  ipc->EvictClientByIdentity(id, port);
  REQUIRE(ipc->GetClientDialBackCount() == before - 1);
}

TEST_CASE("ClientPool - lowering the cap trims the table",
          "[ipc][clientpool][1065]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  const size_t original = ipc->GetMaxClientDialBacks();
  for (int i = 0; i < 8; ++i) {
    REQUIRE(ipc->GetOrCreateClientByIdentity(FakeIdentity("trim", i),
                                             "127.0.0.1",
                                             kBasePort + 5 * kAttempts + i) !=
            nullptr);
  }
  ipc->SetMaxClientDialBacks(4);
  REQUIRE(ipc->GetClientDialBackCount() <= 4);
  ipc->SetMaxClientDialBacks(original);
  REQUIRE(ipc->GetMaxClientDialBacks() == original);
}

SIMPLE_TEST_MAIN()
