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

// Peer dial robustness tests: the per-peer backoff / single-flight dial table
// (used by IpcManager::GetOrCreateClient), the connect path's host-lookup
// cache and failure reasons, and (with Thallium) the address exchange under
// a burst of concurrent dialers and next to an unresponsive peer. Every
// timing assertion is a generous upper bound or is driven by a fake clock.

#include "../../../context-runtime/test/simple_test.h"

#include <clio_ctp/lightbeam/peer_dial_table.h>
#include <clio_ctp/lightbeam/posix_socket.h>
#if CTP_ENABLE_THALLIUM
#include <clio_ctp/lightbeam/thallium_engine.h>
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace ctp::lbm;
using Ms = std::chrono::milliseconds;
using SteadyClock = std::chrono::steady_clock;

namespace {

/** A dialed connection in the tests: owns a connected socket (or nothing). */
struct FakeConn {
  sock::socket_t fd_ = sock::kInvalidSocket; /**< connected socket */
  /** Close the socket. */
  ~FakeConn() { sock::Close(fd_); }
};
using Table = PeerDialTable<FakeConn>;

/**
 * Open a TCP listener on an OS-chosen loopback port.
 * @param port Receives the bound port.
 * @return The listening socket.
 */
sock::socket_t ListenEphemeral(int *port) {
  sock::socket_t fd = sock::Listen("127.0.0.1", 0, "tcp");
  REQUIRE(fd != sock::kInvalidSocket);
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);
  REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr *>(&sin), &len) == 0);
  *port = ntohs(sin.sin_port);
  return fd;
}

/**
 * Find a loopback port nothing listens on (bind port 0, then close it).
 * @return The port.
 */
int ClosedPort() {
  int port = 0;
  sock::Close(ListenEphemeral(&port));
  return port;
}

/**
 * Dialer that makes a real TCP connection to 127.0.0.1:port.
 * @param port Target port.
 * @return A Table::Dialer.
 */
Table::Dialer TcpDialer(int port) {
  return [port](std::string *error) -> Table::Ptr {
    sock::ConnectError err;
    sock::socket_t fd = sock::ConnectTimed("127.0.0.1", port, 5000, &err);
    if (fd == sock::kInvalidSocket) {
      *error = err.ToString();
      return nullptr;
    }
    auto conn = std::make_unique<FakeConn>();
    conn->fd_ = fd;
    return conn;
  };
}

/**
 * Upper bound on real dials to a peer that keeps failing for @p elapsed:
 * one per backoff window (50 ms doubling to a 2000 ms cap), plus slack.
 * @param elapsed Time the loop ran.
 * @return The bound.
 */
uint64_t MaxDials(Ms elapsed) {
  double ms = static_cast<double>(elapsed.count());
  double doubling = std::log2(std::max(1.0, ms / 50.0) + 1.0) + 1.0;
  return static_cast<uint64_t>(doubling + ms / 2000.0) + 3;
}

}  // namespace

TEST_CASE("PeerDial: backoff bounds dials (fake clock)", "[peer_dial]") {
  Table table;
  auto now = SteadyClock::now();
  table.SetClock([&now]() { return now; });
  std::atomic<int> dials{0};
  bool healthy = false;
  Table::Dialer dial = [&](std::string *error) -> Table::Ptr {
    dials++;
    if (!healthy) {
      *error = "refused";
      return nullptr;
    }
    return std::make_unique<FakeConn>();
  };
  // 10000 tight retries spread over 10 s of (fake) time.
  for (int i = 0; i < 10000; ++i) {
    REQUIRE(table.GetOrDial("p:1", "p:1", dial) == nullptr);
    now += Ms(1);
  }
  // Windows 50,100,...,1600 then 2000 ms: ~7 + (10000-3150)/2000 = ~11.
  INFO("dials=" << dials.load());
  REQUIRE(dials.load() >= 8);
  REQUIRE(dials.load() <= 14);
  REQUIRE(table.DialAttempts() == static_cast<uint64_t>(dials.load()));

  // A second key of the same peer is backed off too (no extra dial).
  int before = dials.load();
  REQUIRE(table.GetOrDial("p:1#resp", "p:1", dial) == nullptr);
  REQUIRE(dials.load() == before);

  // Recovery: once the window passes, the next call dials and succeeds.
  healthy = true;
  now += Ms(2001);
  PeerDialReport rep;
  REQUIRE(table.GetOrDial("p:1", "p:1", dial, &rep) != nullptr);
  REQUIRE(rep.status_ == PeerDialStatus::kDialed);
  REQUIRE(rep.log_);  // ends a failure streak
  REQUIRE(table.GetOrDial("p:1", "p:1", dial, &rep) != nullptr);
  REQUIRE(rep.status_ == PeerDialStatus::kCached);

  // Backoff was reset: a new failure starts again at the 50 ms window.
  healthy = false;
  REQUIRE(table.GetOrDial("p:1#x", "p:1", dial, &rep) == nullptr);
  REQUIRE(rep.status_ == PeerDialStatus::kFailed);
  REQUIRE(rep.streak_ == 1u);
  REQUIRE(rep.backoff_ == Ms(50));
}

TEST_CASE("PeerDial: failure log is rate limited", "[peer_dial]") {
  Table table;
  auto now = SteadyClock::now();
  table.SetClock([&now]() { return now; });
  Table::Dialer dial = [](std::string *error) -> Table::Ptr {
    *error = "refused";
    return nullptr;
  };
  int logged = 0;
  int failed = 0;
  // 60 s of fake time, a retry every 10 ms.
  for (int i = 0; i < 6000; ++i) {
    PeerDialReport rep;
    table.GetOrDial("p:2", "p:2", dial, &rep);
    if (rep.status_ == PeerDialStatus::kFailed) failed++;
    if (rep.log_) logged++;
    now += Ms(10);
  }
  INFO("failed=" << failed << " logged=" << logged);
  // First failure, then at most one per 5 s.
  REQUIRE(logged >= 2);
  REQUIRE(logged <= 1 + 60 / 5 + 1);
  REQUIRE(failed > logged);
}

TEST_CASE("PeerDial: closed port backs off, then recovers", "[peer_dial]") {
  Table table;
  int port = ClosedPort();
  Table::Dialer dial = TcpDialer(port);
  PeerDialReport rep;
  REQUIRE(table.GetOrDial("k", "peer", dial, &rep) == nullptr);
  REQUIRE(rep.status_ == PeerDialStatus::kFailed);
  INFO("error=" << rep.error_);
  REQUIRE(rep.error_.find("refused") != std::string::npos);

  // Tight retries for 1500 ms of wall time: ~5 windows (50..800 ms), so a
  // handful of real connects instead of one per call.
  auto t0 = SteadyClock::now();
  uint64_t calls = 0;
  Ms elapsed(0);
  while (elapsed < Ms(1500)) {
    REQUIRE(table.GetOrDial("k", "peer", dial) == nullptr);
    calls++;
    elapsed = std::chrono::duration_cast<Ms>(SteadyClock::now() - t0);
  }
  INFO("calls=" << calls << " attempts=" << table.DialAttempts()
                  << " elapsed_ms=" << elapsed.count());
  REQUIRE(calls > 1000);
  REQUIRE(table.DialAttempts() >= 2);
  REQUIRE(table.DialAttempts() <= MaxDials(elapsed));

  // The port comes up: the table reconnects within one capped window.
  sock::socket_t lfd = sock::Listen("127.0.0.1", port, "tcp");
  REQUIRE(lfd != sock::kInvalidSocket);
  FakeConn *conn = nullptr;
  auto deadline = SteadyClock::now() + Ms(30000);
  while (conn == nullptr && SteadyClock::now() < deadline) {
    conn = table.GetOrDial("k", "peer", dial);
    if (conn == nullptr) std::this_thread::sleep_for(Ms(5));
  }
  REQUIRE(conn != nullptr);
  REQUIRE(table.GetOrDial("k", "peer", dial) == conn);
  REQUIRE(table.Size() == 1u);
  sock::Close(lfd);
}

TEST_CASE("PeerDial: one dial in flight per key", "[peer_dial]") {
  Table table;
  std::atomic<bool> release{false};
  std::atomic<int> dials{0};
  Table::Dialer slow = [&](std::string *) -> Table::Ptr {
    dials++;
    while (!release.load()) std::this_thread::sleep_for(Ms(1));
    return std::make_unique<FakeConn>();
  };
  std::atomic<bool> first_ok{false};
  std::thread t([&]() { first_ok = table.GetOrDial("k", "p", slow) != nullptr; });
  while (dials.load() == 0) std::this_thread::sleep_for(Ms(1));
  // A second caller for the same key does not dial; it gives up after the
  // in-flight wait (100 ms) with nullptr.
  PeerDialReport rep;
  auto t0 = SteadyClock::now();
  REQUIRE(table.GetOrDial("k", "p", slow, &rep) == nullptr);
  REQUIRE(rep.status_ == PeerDialStatus::kInFlight);
  REQUIRE(SteadyClock::now() - t0 < Ms(20000));
  REQUIRE(dials.load() == 1);
  release = true;
  t.join();
  REQUIRE(first_ok.load());
  REQUIRE(table.GetOrDial("k", "p", slow, &rep) != nullptr);
  REQUIRE(rep.status_ == PeerDialStatus::kCached);
  REQUIRE(dials.load() == 1);
}

TEST_CASE("PeerDial: host names are looked up once", "[peer_dial]") {
  int port = 0;
  sock::socket_t lfd = ListenEphemeral(&port);
  uint64_t before = sock::GetHostLookupCount();
  for (int i = 0; i < 200; ++i) {
    sock::ConnectError err;
    sock::socket_t fd = sock::ConnectTimed("localhost", port, 5000, &err);
    if (fd == sock::kInvalidSocket) INFO("i=" << i << " err=" << err.ToString());
    REQUIRE(fd != sock::kInvalidSocket);
    sock::Close(fd);
    sock::socket_t afd = sock::Accept(lfd);
    sock::Close(afd);
  }
  REQUIRE(sock::GetHostLookupCount() - before == 1u);
  // A failed connect drops the entry; the next dial looks the name up again.
  sock::Close(lfd);
  sock::ConnectError err;
  REQUIRE(sock::ConnectTimed("localhost", port, 5000, &err) ==
          sock::kInvalidSocket);
  REQUIRE(err.stage_ == sock::ConnectError::kConnect);
  uint64_t after_fail = sock::GetHostLookupCount();
  sock::ConnectTimed("localhost", port, 5000, &err);
  REQUIRE(sock::GetHostLookupCount() == after_fail + 1);
  // An unknown name reports a lookup failure, not a connect failure.
  REQUIRE(sock::ConnectTimed("no-such-host.invalid", port, 5000, &err) ==
          sock::kInvalidSocket);
  REQUIRE(err.stage_ == sock::ConnectError::kResolve);
}

#if CTP_ENABLE_THALLIUM

namespace {

/** Bring up the Thallium engine once and publish its address on a port.
 *  @return The published port. */
int PublishedPort() {
  static int port = [] {
    ThalliumEngine::Get("127.0.0.1");
    int p = ClosedPort();
    ThalliumEngine::Publish(p);
    return p;
  }();
  return port;
}

}  // namespace

TEST_CASE("Thallium: 1024 concurrent address exchanges", "[peer_dial]") {
  int port = PublishedPort();
  std::string self = ThalliumEngine::SelfAddress();
  REQUIRE(!self.empty());
  constexpr int kThreads = 1024;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::atomic<int> ok{0};
  std::vector<std::string> errors(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]() {
      ready++;
      while (!go.load()) std::this_thread::yield();
      std::string why;
      std::string addr = ThalliumEngine::Resolve("127.0.0.1", port, true, &why);
      if (addr == self) {
        ok++;
      } else {
        errors[i] = why;
      }
    });
  }
  while (ready.load() < kThreads) std::this_thread::sleep_for(Ms(1));
  auto t0 = SteadyClock::now();
  go = true;
  for (auto &t : threads) t.join();
  auto ms = std::chrono::duration_cast<Ms>(SteadyClock::now() - t0).count();
  std::string first_error;
  for (auto &e : errors) {
    if (!e.empty()) {
      first_error = e;
      break;
    }
  }
  INFO("ok=" << ok.load() << " ms=" << ms << " first_error=" << first_error);
  std::cout << "[peer_dial] " << ok.load() << "/" << kThreads
            << " concurrent resolves in " << ms << " ms" << std::endl;
  REQUIRE(ok.load() == kThreads);
  REQUIRE(ms < 120000);
}

TEST_CASE("Thallium: unresponsive peer does not block others", "[peer_dial]") {
  int good_port = PublishedPort();
  // A listener that accepts the handshake (kernel backlog) but never answers.
  int bad_port = 0;
  sock::socket_t bad_lfd = ListenEphemeral(&bad_port);

  using StrTable = PeerDialTable<std::string>;
  StrTable table;
  /** Dialer resolving a port's Mercury address through the exchange. */
  auto resolver = [](int port, std::atomic<int> *started) {
    return [port, started](std::string *error) -> StrTable::Ptr {
      if (started != nullptr) (*started)++;
      std::string why;
      std::string addr = ThalliumEngine::Resolve("127.0.0.1", port, true, &why);
      if (addr.empty()) {
        *error = why;
        return nullptr;
      }
      return std::make_unique<std::string>(addr);
    };
  };
  std::atomic<int> bad_started{0};
  std::atomic<bool> bad_done{false};
  PeerDialReport bad_rep;
  std::thread bad([&]() {
    table.GetOrDial("bad", "bad", resolver(bad_port, &bad_started), &bad_rep);
    bad_done = true;
  });
  while (bad_started.load() == 0) std::this_thread::sleep_for(Ms(1));
  std::this_thread::sleep_for(Ms(50));  // let it reach the reply wait

  auto t0 = SteadyClock::now();
  std::string *good = table.GetOrDial("good", "good", resolver(good_port, nullptr));
  auto good_ms =
      std::chrono::duration_cast<Ms>(SteadyClock::now() - t0).count();
  // The bad dial is still waiting for a reply (timeout is 60 s here).
  bool bad_still_waiting = !bad_done.load();
  std::string good_addr = good != nullptr ? *good : std::string();

  // Closing the silent listener resets its queued connection: the bad dial
  // fails promptly with a reason.
  sock::Close(bad_lfd);
  bad.join();
  INFO("good_ms=" << good_ms << " bad error=" << bad_rep.error_);
  REQUIRE(good != nullptr);
  REQUIRE(good_addr == ThalliumEngine::SelfAddress());
  REQUIRE(good_ms < 3000);
  REQUIRE(bad_still_waiting);
  REQUIRE(bad_rep.status_ == PeerDialStatus::kFailed);
  REQUIRE(!bad_rep.error_.empty());
}

#endif  // CTP_ENABLE_THALLIUM

/** Program entry: make the reply wait long so the unresponsive-peer test is
 *  decided by the test, not by the timeout; then run the suite. */
static const bool kEnvSet = [] {
  setenv("CLIO_LBM_THALLIUM_RESOLVE_TIMEOUT_MS", "60000", 0);
  return true;
}();

SIMPLE_TEST_MAIN()
