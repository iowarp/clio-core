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

// =============================================================================
// Isolated reproduction of the CLIO runtime ZeroMQ threading topology.
//
//   Client side                         Server side ("runtime")
//   -----------                         -----------------------
//   4 sender threads  --- requests -->  1 receiver thread  (ROUTER, port S)
//   1 receiver thread <-- responses --  1 sender thread    (DEALER -> port C)
//      (ROUTER, port C)
//
// The two ROUTERs (client-receiver on port C, server-receiver on port S) are
// DISTINCT sockets on DISTINCT ports — they are not the same thing.
//
// Two toggles model the two suspected rule-violations of "one socket, one
// thread":
//   per_thread_client : false = 4 client senders share ONE DEALER (+ mutex),
//                       like the runtime's zmq_transport_ / zmq_client_send_mutex_.
//                       true  = each sender thread owns its own DEALER.
//   dialback_on_recv  : true  = the server creates its response DEALER on the
//                       RECV thread (like RecvIn's GetOrCreateClientByIdentity)
//                       but sends on it from the SEND thread (cross-thread).
//                       false = the DEALER is created on the SEND thread.
//
// Run one scenario per process (selected by argv[1]); a wedge shows up as the
// process failing to exit, so drive it under an external timeout.
// =============================================================================

#include <clio_ctp/lightbeam/zmq_transport.h>
#include <clio_ctp/lightbeam/transport_factory_impl.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace ctp::lbm;

class MsgMeta : public LbmMeta<> {
 public:
  uint64_t req_id = 0;
  int kind = 0;  // 0 = request, 1 = response

  template <typename Ar>
  void serialize(Ar& ar) {
    LbmMeta<>::serialize(ar);
    ar(req_id, kind);
  }
};

struct ScenarioCfg {
  std::string name;
  int port_s = 0;          // server receiver (ROUTER) port
  int port_c = 0;          // client receiver (ROUTER) port — DIFFERENT from S
  bool per_thread_client = false;
  bool dialback_on_recv = true;
  int n_senders = 4;
  int msgs_per = 100;
  int timeout_sec = 20;
  int bulk_size = 0;  // >0 attaches a BULK_XFER payload to each message
};

static void sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
static void sleep_us(int us) {
  std::this_thread::sleep_for(std::chrono::microseconds(us));
}

static bool RunScenario(const ScenarioCfg& cfg) {
  const int total = cfg.n_senders * cfg.msgs_per;
  std::cout << "\n==== Scenario '" << cfg.name << "' S=" << cfg.port_s
            << " C=" << cfg.port_c
            << " per_thread_client=" << cfg.per_thread_client
            << " dialback_on_recv=" << cfg.dialback_on_recv
            << " total=" << total << " ====\n"
            << std::flush;

  const std::string addr = "127.0.0.1";

  // Two receivers (ROUTERs) on separate ports.
  auto server_recv = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", cfg.port_s);
  auto client_recv = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", cfg.port_c);
  sleep_ms(150);

  std::atomic<bool> shutdown{false};
  std::atomic<int> responses_received{0};
  std::atomic<int> requests_sent{0};
  std::atomic<int> requests_recvd{0};

  // Server-side response DEALER -> client receiver (port C). Lazily created;
  // a shared slot so it can be born on the recv thread (clio pattern) and used
  // on the send thread.
  std::shared_ptr<ZeroMqTransport> server_send;
  std::mutex server_send_mtx;
  auto get_server_send = [&]() -> ZeroMqTransport* {
    std::lock_guard<std::mutex> lk(server_send_mtx);
    if (!server_send) {
      server_send = std::make_shared<ZeroMqTransport>(
          TransportMode::kClient, addr, "tcp", cfg.port_c);
      sleep_ms(50);
    }
    return server_send.get();
  };

  // recv thread -> send thread handoff queue.
  std::mutex sq_mtx;
  std::vector<uint64_t> send_queue;

  // ---- Server receiver thread (ROUTER on port S) -------------------------
  std::thread server_recv_thread([&]() {
    while (!shutdown.load(std::memory_order_acquire)) {
      MsgMeta meta;
      auto info = server_recv->Recv(meta);
      if (info.rc != 0) {
        sleep_us(150);
        continue;
      }
      uint64_t id = meta.req_id;
      // Read (and free) any bulk frames so the multipart stream stays aligned —
      // a leftover bulk frame desyncs the next Recv onto the wrong frame.
      server_recv->ClearRecvHandles(meta);
      requests_recvd.fetch_add(1, std::memory_order_relaxed);
      if (cfg.dialback_on_recv) {
        get_server_send();  // born on the RECV thread (clio RecvIn pattern)
      }
      std::lock_guard<std::mutex> lk(sq_mtx);
      send_queue.push_back(id);
    }
  });

  // ---- Server sender thread (DEALER -> port C) ---------------------------
  std::thread server_send_thread([&]() {
    while (!shutdown.load(std::memory_order_acquire)) {
      uint64_t id = 0;
      bool have = false;
      {
        std::lock_guard<std::mutex> lk(sq_mtx);
        if (!send_queue.empty()) {
          id = send_queue.back();
          send_queue.pop_back();
          have = true;
        }
      }
      if (!have) {
        sleep_us(150);
        continue;
      }
      ZeroMqTransport* snd = get_server_send();  // born here if !dialback_on_recv
      MsgMeta resp;
      resp.req_id = id;
      resp.kind = 1;
      std::vector<char> rbuf;
      if (cfg.bulk_size > 0) {
        rbuf.assign(cfg.bulk_size, 'R');
        resp.send.push_back(snd->Expose(
            ctp::ipc::FullPtr<char>(rbuf.data()), rbuf.size(), BULK_XFER));
      } else {
        resp.send_bulks = 0;
      }
      snd->Send(resp);
    }
  });

  // ---- Client receiver thread (ROUTER on port C) -------------------------
  std::thread client_recv_thread([&]() {
    while (!shutdown.load(std::memory_order_acquire)) {
      MsgMeta meta;
      auto info = client_recv->Recv(meta);
      if (info.rc != 0) {
        sleep_us(150);
        continue;
      }
      client_recv->ClearRecvHandles(meta);
      responses_received.fetch_add(1, std::memory_order_relaxed);
    }
  });

  // ---- Client sender threads (many-to-one -> port S) ---------------------
  std::shared_ptr<ZeroMqTransport> shared_client_send;
  std::mutex client_send_mtx;
  if (!cfg.per_thread_client) {
    shared_client_send = std::make_shared<ZeroMqTransport>(
        TransportMode::kClient, addr, "tcp", cfg.port_s);
    sleep_ms(50);
  }
  std::vector<std::thread> senders;
  for (int t = 0; t < cfg.n_senders; ++t) {
    senders.emplace_back([&, t]() {
      std::unique_ptr<ZeroMqTransport> own;
      ZeroMqTransport* snd;
      if (cfg.per_thread_client) {
        own = std::make_unique<ZeroMqTransport>(
            TransportMode::kClient, addr, "tcp", cfg.port_s);
        sleep_ms(50);
        snd = own.get();
      } else {
        snd = shared_client_send.get();
      }
      for (int m = 0; m < cfg.msgs_per; ++m) {
        uint64_t id = (static_cast<uint64_t>(t) << 32) | static_cast<uint64_t>(m);
        MsgMeta req;
        req.req_id = id;
        req.kind = 0;
        std::vector<char> buf;
        if (cfg.bulk_size > 0) {
          buf.assign(cfg.bulk_size, 'Q');
          req.send.push_back(snd->Expose(
              ctp::ipc::FullPtr<char>(buf.data()), buf.size(), BULK_XFER));
        } else {
          req.send_bulks = 0;
        }
        if (cfg.per_thread_client) {
          snd->Send(req);
        } else {
          std::lock_guard<std::mutex> lk(client_send_mtx);
          snd->Send(req);
        }
        requests_sent.fetch_add(1, std::memory_order_relaxed);
      }
      // Keep the per-thread DEALER alive until shutdown so in-flight responses
      // have somewhere to route back to at the protocol level.
      while (cfg.per_thread_client && !shutdown.load(std::memory_order_acquire)) {
        sleep_us(500);
      }
    });
  }

  // ---- Wait for completion or timeout ------------------------------------
  auto start = std::chrono::steady_clock::now();
  bool ok = false;
  int last = -1;
  while (true) {
    int got = responses_received.load(std::memory_order_relaxed);
    if (got >= total) {
      ok = true;
      break;
    }
    if (got != last) {
      last = got;
    }
    if (std::chrono::steady_clock::now() - start >
        std::chrono::seconds(cfg.timeout_sec)) {
      break;
    }
    sleep_ms(50);
  }

  std::cout << "[" << cfg.name << "] sent=" << requests_sent.load()
            << " server_recvd=" << requests_recvd.load()
            << " responses=" << responses_received.load() << "/" << total
            << " -> " << (ok ? "PASS" : "FAIL/HANG") << "\n"
            << std::flush;

  if (!ok) {
    // A recv thread is likely wedged inside a blocking frame read, so joining
    // would deadlock. Report and hard-exit; the OS reclaims sockets.
    std::cerr << "SCENARIO HUNG — hard exit\n" << std::flush;
    std::_Exit(1);
  }

  shutdown.store(true, std::memory_order_release);
  for (auto& s : senders) s.join();
  server_recv_thread.join();
  server_send_thread.join();
  client_recv_thread.join();
  return ok;
}

int main(int argc, char** argv) {
  std::string which = (argc > 1) ? argv[1] : "clio";

  ScenarioCfg sc;
  if (which == "clio") {
    // Faithful to the current runtime: shared client DEALER + mutex, response
    // DEALER born on the recv thread, used on the send thread.
    sc = {"clio-faithful", 8400, 8401, /*per_thread*/ false,
          /*dialback_on_recv*/ true};
  } else if (which == "dialback-send") {
    // Fix #1: response DEALER created on the send thread (no cross-thread sock).
    sc = {"dialback-on-send", 8410, 8411, false, false};
  } else if (which == "perthread") {
    // Fix #2: each client sender owns its DEALER (no shared send socket).
    sc = {"per-thread-client", 8420, 8421, true, true};
  } else if (which == "both") {
    // Both fixes: strictly one socket per thread everywhere.
    sc = {"both-fixes", 8430, 8431, true, false};
  } else if (which == "clio-bulk") {
    // Faithful clio + bulk payloads (models real task traffic).
    sc = {"clio-bulk", 8440, 8441, false, true};
    sc.bulk_size = 4096;
  } else if (which == "both-bulk") {
    sc = {"both-bulk", 8450, 8451, true, false};
    sc.bulk_size = 4096;
  } else if (which == "clio-bulk-heavy") {
    // Shared client DEALER + 1 MiB bulks + heavier message count, to stress
    // the multipart framing under back-pressure (mirrors 1 MiB GetBlob traffic).
    sc = {"clio-bulk-heavy", 8460, 8461, false, true};
    sc.bulk_size = 1024 * 1024;
    sc.msgs_per = 250;
    sc.timeout_sec = 40;
  } else {
    std::cerr << "unknown scenario '" << which
              << "' (clio|dialback-send|perthread|both|clio-bulk|both-bulk)\n";
    return 2;
  }

  bool ok = RunScenario(sc);
  std::cout << "\n" << (ok ? "RESULT: PASS" : "RESULT: FAIL") << std::endl;
  return ok ? 0 : 1;
}
