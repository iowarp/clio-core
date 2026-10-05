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
 * Reproducer for issue #1065: the runtime's client pool never releases a
 * dial-back connection.
 *
 * IpcManager::GetOrCreateClient() inserts one owning TransportPtr per
 * "addr:port" key into client_pool_ and nothing on the dial-back path ever
 * erases it: SetDead() only erases hostfile nodes, and EvictClientByIdentity()
 * only drops the identity -> Transport* lookaside (client_conn_cache_), not the
 * owning client_pool_ entry, so the DEALER socket lives on. The shared ZeroMQ
 * context never sets ZMQ_MAX_SOCKETS, so libzmq's default ceiling of 1023
 * applies: once the pool reaches it, zmq_socket() returns NULL, the dial-back
 * fails with "not a socket", and that client's response is dropped.
 *
 * Kept in its own executable so its expected failure does not mask the other
 * IpcManager internals tests; ctest marks it WILL_FAIL until #1065 is fixed.
 */

#include "simple_test.h"

#include <iostream>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/singletons.h>

namespace {

/** Number of distinct fake clients to dial back; above libzmq's 1023. */
constexpr int kAttempts = 1100;
/** First fake client port; client i uses kBasePort + i. */
constexpr int kBasePort = 24000;
/** How many early identities to evict before retrying a fresh dial-back. */
constexpr int kEvictCount = 50;

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
 * Build the fake dial-back identity for client index i.
 * @param i client index
 * @return identity string unique to this test
 */
std::string FakeIdentity(int i) { return "repro1065:" + std::to_string(i); }

}  // namespace

TEST_CASE("ClientPool - dial-back sockets are released (issue #1065)",
          "[ipc][clientpool][1065]") {
  EnsureInitialized();
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // No listener is needed: zmq_connect() on a TCP DEALER is asynchronous, so
  // a failure here comes from exhausting the context's socket slots, not from
  // a refused connection.
  int first_failure = -1;
  int failure_count = 0;
  for (int i = 0; i < kAttempts; ++i) {
    ctp::lbm::Transport *t = ipc->GetOrCreateClientByIdentity(
        FakeIdentity(i), "127.0.0.1", kBasePort + i);
    if (t == nullptr) {
      ++failure_count;
      if (first_failure == -1) first_failure = i;
    }
  }
  std::cout << "[#1065 repro] first dial-back failure at attempt "
            << first_failure << " (" << failure_count << "/" << kAttempts
            << " failed)" << std::endl;

  // If eviction released the pooled socket, evicting earlier identities would
  // make room for a new one.
  bool recovered_after_evict = (first_failure == -1);
  if (first_failure != -1) {
    for (int i = 0; i < kEvictCount && i < first_failure; ++i) {
      ipc->EvictClientByIdentity(FakeIdentity(i), kBasePort + i);
    }
    ctp::lbm::Transport *retry = ipc->GetOrCreateClientByIdentity(
        "repro1065:retry", "127.0.0.1", kBasePort + kAttempts + 1);
    recovered_after_evict = (retry != nullptr);
    std::cout << "[#1065 repro] after evicting " << kEvictCount
              << " earlier identities, a fresh dial-back "
              << (recovered_after_evict ? "SUCCEEDED" : "STILL FAILED")
              << std::endl;
  }

  // Both encode the fixed behavior and fail on a tree that still has #1065.
  REQUIRE(first_failure == -1);
  REQUIRE(recovered_after_evict);
}

SIMPLE_TEST_MAIN()
