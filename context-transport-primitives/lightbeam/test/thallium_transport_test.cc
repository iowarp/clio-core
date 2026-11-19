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


// ThalliumTransport tests. No arguments: the in-process suite. Two-process
// use (run the listener first):
//   thallium_transport_test server <port> <messages>
//   thallium_transport_test client <host> <port> <threads> <msgs_per_thread>
//   thallium_transport_test backpressure      (needs RECV_QUEUE_MSGS=4)
// The Mercury protocol comes from CLIO_LBM_THALLIUM_PROTOCOL.

#include <clio_ctp/lightbeam/thallium_transport.h>
#include <clio_ctp/lightbeam/transport_factory_impl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace ctp::lbm;

/** Abort with a message if a condition is false (active in release builds). */
#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::cerr << "CHECK FAILED: " #cond " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                               \
      std::abort();                                                     \
    }                                                                   \
  } while (0)

/** Test metadata: sender id and sequence number on top of the bulk lists. */
class TestMeta : public LbmMeta<> {
 public:
  int request_id = 0;
  int sender = 0;
  std::string operation;

  /** @param ar Archive to (de)serialize through. */
  template <typename Ar>
  void serialize(Ar &ar) {
    LbmMeta<>::serialize(ar);
    ar(request_id, sender, operation);
  }
};

/**
 * Deterministic payload byte.
 * @param seed Per-segment seed.
 * @param i Byte offset.
 * @return The byte value expected at offset i.
 */
static char PatternByte(size_t seed, size_t i) {
  return static_cast<char>((seed * 131 + i * 7 + (i >> 8)) & 0xff);
}

/**
 * Build a payload buffer filled with the deterministic pattern.
 * @param size Bytes.
 * @param seed Pattern seed.
 * @return The buffer.
 */
static std::vector<char> MakePayload(size_t size, size_t seed) {
  std::vector<char> buf(size);
  for (size_t i = 0; i < size; ++i) buf[i] = PatternByte(seed, i);
  return buf;
}

/**
 * Check a received buffer against the deterministic pattern.
 * @param p Received bytes.
 * @param size Bytes.
 * @param seed Pattern seed.
 * @return true if every byte matches.
 */
static bool CheckPayload(const char *p, size_t size, size_t seed) {
  for (size_t i = 0; i < size; ++i) {
    if (p[i] != PatternByte(seed, i)) return false;
  }
  return true;
}

/**
 * Append a buffer to a message as a BULK_XFER segment.
 * @param t Transport the message will be sent on.
 * @param meta Message.
 * @param buf Payload.
 */
static void AddXfer(Transport *t, TestMeta &meta, std::vector<char> &buf) {
  ctp::ipc::FullPtr<char> ptr;
  ptr.ptr_ = buf.data();
  ptr.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  ptr.shm_.off_ = reinterpret_cast<size_t>(buf.data());
  meta.send.push_back(t->Expose(ptr, buf.size(), BULK_XFER));
  meta.send_bulks++;
}

/**
 * Receive one message, blocking in PollRecv between attempts.
 * @param server Listener.
 * @param meta Receives the message.
 * @param max_ms Give up after this long.
 * @return The ClientInfo of the last Recv.
 */
template <typename MetaT>
ClientInfo RecvWait(Transport *server, MetaT &meta, int max_ms = 10000) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(max_ms);
  while (true) {
    auto info = server->Recv(meta);
    if (info.rc != EAGAIN) return info;
    if (std::chrono::steady_clock::now() >= deadline) return info;
    server->PollRecv(50);
  }
}

/** @return A TCP port unlikely to collide with another test run. */
static int TestPort(int offset) {
  return 20000 + static_cast<int>(::getpid() % 20000) + offset;
}

/** Metadata-only message over a direct Mercury address. */
static void TestRoundTrip() {
  std::cout << "==== round trip, no bulk, direct address ====" << std::endl;
  TransportPtr server = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kServer, "tcp", 0);
  std::string server_addr = server->GetAddress();
  std::cout << "  server addr = " << server_addr << std::endl;
  CHECK(!server_addr.empty());
  TransportPtr client = TransportFactory::Get(
      server_addr, TransportType::kThallium, TransportMode::kClient, "tcp", 0);
  TestMeta send_meta;
  send_meta.request_id = 42;
  send_meta.operation = "hello-thallium";
  CHECK(client->Send(send_meta) == 0);
  TestMeta recv_meta;
  CHECK(server->PollRecv(5000) == 1);
  CHECK(server->Recv(recv_meta).rc == 0);
  CHECK(recv_meta.request_id == 42);
  CHECK(recv_meta.operation == "hello-thallium");
  TestMeta none;
  CHECK(server->Recv(none).rc == EAGAIN);
  CHECK(server->PollRecv(20) == 0);
  std::cout << "  ok" << std::endl;
}

/**
 * One message mixing an exposed-only entry with segments below, at and above
 * the bulk threshold, resolved through the address exchange.
 * @param threshold Bulk threshold to send with.
 */
static void TestBulkMix(size_t threshold) {
  std::cout << "==== mixed segments, threshold " << threshold
            << ", address exchange ====" << std::endl;
  int port = TestPort(1);
  TransportPtr server = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kServer, "tcp",
      port);
  TransportPtr client = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kClient, "tcp",
      port);
  CHECK(client->IsServerAlive());
  static_cast<ThalliumTransport *>(client.get())->SetBulkThreshold(threshold);
  const size_t sizes[] = {0,     1,         100,        4096,
                          65535, 65536,     1 << 20,    (3 << 20) + 17};
  std::vector<std::vector<char>> bufs;
  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
    bufs.push_back(MakePayload(sizes[i], i + 1));
  }
  TestMeta send_meta;
  send_meta.request_id = 7;
  send_meta.operation = "bulk";
  ctp::ipc::FullPtr<char> none = ctp::ipc::FullPtr<char>::GetNull();
  send_meta.send.push_back(client->Expose(none, 12345, BULK_EXPOSE));
  for (auto &b : bufs) AddXfer(client.get(), send_meta, b);
  CHECK(client->Send(send_meta) == 0);
  TestMeta recv_meta;
  CHECK(RecvWait(server.get(), recv_meta).rc == 0);
  CHECK(recv_meta.request_id == 7);
  CHECK(recv_meta.recv.size() == bufs.size() + 1);
  CHECK(recv_meta.recv[0].data.ptr_ == nullptr);
  CHECK(recv_meta.recv[0].size == 12345);
  for (size_t i = 0; i < bufs.size(); ++i) {
    const Bulk &b = recv_meta.recv[i + 1];
    CHECK(b.size == sizes[i]);
    CHECK(b.data.ptr_ != nullptr);
    CHECK(CheckPayload(b.data.ptr_, b.size, i + 1));
  }
  server->ClearRecvHandles(recv_meta);
  std::cout << "  ok (" << bufs.size() << " segments)" << std::endl;
}

/** Two listeners in one process each receive only their own messages. */
static void TestTwoListeners() {
  std::cout << "==== two listeners on one engine ====" << std::endl;
  int pa = TestPort(2), pb = TestPort(3);
  TransportPtr sa = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kServer, "tcp", pa);
  TransportPtr sb = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kServer, "tcp", pb);
  TransportPtr ca = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kClient, "tcp", pa);
  TransportPtr cb = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kClient, "tcp", pb);
  TestMeta ma, mb;
  ma.request_id = 1;
  mb.request_id = 2;
  CHECK(ca->Send(ma) == 0);
  CHECK(cb->Send(mb) == 0);
  TestMeta ra, rb, extra;
  CHECK(RecvWait(sa.get(), ra).rc == 0);
  CHECK(RecvWait(sb.get(), rb).rc == 0);
  CHECK(ra.request_id == 1);
  CHECK(rb.request_id == 2);
  CHECK(sa->Recv(extra).rc == EAGAIN);
  CHECK(sb->Recv(extra).rc == EAGAIN);
  std::cout << "  ok" << std::endl;
}

/**
 * Sender thread body: sends `count` messages with a size that cycles from
 * inline to pulled, on its own connection or a shared one.
 * @param t Connection to send on.
 * @param sender Sender id stamped on each message.
 * @param count Messages to send.
 * @param failures Incremented for every failed Send.
 */
static void SenderLoop(Transport *t, int sender, int count,
                       std::atomic<int> *failures) {
  const size_t sizes[] = {0, 64, 3000, 70000, 300000};
  for (int i = 0; i < count; ++i) {
    size_t size = sizes[(i + sender) % 5];
    std::vector<char> buf =
        MakePayload(size, static_cast<size_t>(sender) * 1000 + i);
    TestMeta meta;
    meta.sender = sender;
    meta.request_id = i;
    AddXfer(t, meta, buf);
    int rc = t->Send(meta);
    while (rc == EAGAIN) rc = t->Send(meta);
    if (rc != 0) failures->fetch_add(1);
  }
}

/**
 * Receive `total` SenderLoop messages, checking payloads and that each
 * sender's messages arrive in the order it sent them.
 * @param server Listener.
 * @param total Messages expected.
 */
static void ReceiveAll(Transport *server, int total) {
  const size_t sizes[] = {0, 64, 3000, 70000, 300000};
  std::map<int, int> next;
  for (int n = 0; n < total; ++n) {
    TestMeta meta;
    CHECK(RecvWait(server, meta, 60000).rc == 0);
    CHECK(meta.request_id == next[meta.sender]);
    next[meta.sender]++;
    CHECK(meta.recv.size() == 1);
    size_t want = sizes[(meta.request_id + meta.sender) % 5];
    CHECK(meta.recv[0].size == want);
    CHECK(CheckPayload(meta.recv[0].data.ptr_, want,
                       static_cast<size_t>(meta.sender) * 1000 +
                           meta.request_id));
    server->ClearRecvHandles(meta);
  }
}

/**
 * Plain std::threads (not Argobots execution streams) send concurrently:
 * half on their own connections, half sharing one connection.
 * @param threads Sender threads.
 * @param per_thread Messages per sender.
 */
static void TestConcurrentSenders(int threads, int per_thread) {
  std::cout << "==== " << threads << " std::thread senders x " << per_thread
            << " messages ====" << std::endl;
  int port = TestPort(4);
  TransportPtr server = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kServer, "tcp",
      port);
  TransportPtr shared = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kClient, "tcp",
      port);
  std::vector<TransportPtr> own;
  std::vector<std::thread> senders;
  std::atomic<int> failures{0};
  auto t0 = std::chrono::steady_clock::now();
  std::thread receiver(
      [&]() { ReceiveAll(server.get(), threads * per_thread); });
  for (int s = 0; s < threads; ++s) {
    Transport *t = shared.get();
    if (s % 2 == 0) {
      own.push_back(TransportFactory::Get(
          "127.0.0.1", TransportType::kThallium, TransportMode::kClient, "tcp",
          port));
      t = own.back().get();
    }
    senders.emplace_back(SenderLoop, t, s, per_thread, &failures);
  }
  for (auto &t : senders) t.join();
  receiver.join();
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
  CHECK(failures.load() == 0);
  std::cout << "  ok (" << threads * per_thread << " messages in " << ms
            << " ms)" << std::endl;
}

/** Unreachable peer, peer that goes away, peer that comes back. */
static void TestPeerDown() {
  std::cout << "==== peer down / restart ====" << std::endl;
  int port = TestPort(5);
  bool threw = false;
  try {
    TransportPtr c = TransportFactory::Get(
        "127.0.0.1", TransportType::kThallium, TransportMode::kClient, "tcp",
        port);
  } catch (const std::exception &e) {
    threw = true;
    std::cout << "  no listener: constructor threw: " << e.what() << std::endl;
  }
  CHECK(threw);
  TransportPtr server = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kServer, "tcp",
      port);
  TransportPtr client = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kClient, "tcp",
      port);
  TestMeta m;
  m.request_id = 1;
  CHECK(client->Send(m) == 0);
  server.reset();
  auto t0 = std::chrono::steady_clock::now();
  int rc = client->Send(m);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
  std::cout << "  listener gone: Send rc=" << rc << " after " << ms << " ms"
            << std::endl;
  CHECK(rc != 0 && rc != EAGAIN);
  CHECK(!client->IsServerAlive());
  rc = client->Send(m);
  std::cout << "  still gone: Send rc=" << rc << std::endl;
  CHECK(rc != 0 && rc != EAGAIN);
  server = TransportFactory::Get("127.0.0.1", TransportType::kThallium,
                                 TransportMode::kServer, "tcp", port);
  m.request_id = 2;
  CHECK(client->Send(m) == 0);
  TestMeta r;
  CHECK(RecvWait(server.get(), r).rc == 0);
  CHECK(r.request_id == 2);
  std::cout << "  listener back: Send ok" << std::endl;
}

/** Finalize the engine and bring a new one up in the same process. */
static void TestFinalizeRestart() {
  std::cout << "==== finalize and restart engine ====" << std::endl;
  CHECK(ThalliumEngine::IsUp());
  ThalliumEngine::Finalize();
  CHECK(!ThalliumEngine::IsUp());
  ThalliumEngine::Finalize();
  TestBulkMix(64 * 1024);
  ThalliumEngine::Finalize();
  CHECK(!ThalliumEngine::IsUp());
  std::cout << "  ok" << std::endl;
}

/**
 * Full queue: with CLIO_LBM_THALLIUM_RECV_QUEUE_MSGS=4 the fifth unreceived
 * message is refused with EAGAIN after RECV_QUEUE_WAIT_MS and is accepted
 * once the receiver takes one.
 */
static int RunBackpressure() {
  const ThalliumConfig &cfg = ThalliumConfig::Get();
  std::cout << "==== back-pressure, queue of " << cfg.queue_msgs_
            << " messages, wait " << cfg.queue_wait_ms_ << " ms ===="
            << std::endl;
  int port = TestPort(6);
  TransportPtr server = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kServer, "tcp",
      port);
  TransportPtr client = TransportFactory::Get(
      "127.0.0.1", TransportType::kThallium, TransportMode::kClient, "tcp",
      port);
  std::vector<char> buf = MakePayload(200000, 9);
  for (size_t i = 0; i < cfg.queue_msgs_; ++i) {
    TestMeta m;
    m.request_id = static_cast<int>(i);
    AddXfer(client.get(), m, buf);
    CHECK(client->Send(m) == 0);
  }
  TestMeta over;
  over.request_id = 99;
  AddXfer(client.get(), over, buf);
  auto t0 = std::chrono::steady_clock::now();
  int rc = client->Send(over);
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
  std::cout << "  queue full: Send rc=" << rc << " (EAGAIN=" << EAGAIN
            << ") after " << ms << " ms" << std::endl;
  CHECK(rc == EAGAIN);
  // A sender blocked on the full queue is released when room appears.
  std::thread late([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    TestMeta r;
    CHECK(server->Recv(r).rc == 0);
    CHECK(r.request_id == 0);
    server->ClearRecvHandles(r);
  });
  CHECK(client->Send(over) == 0);
  late.join();
  for (size_t i = 1; i <= cfg.queue_msgs_; ++i) {
    TestMeta r;
    CHECK(RecvWait(server.get(), r).rc == 0);
    int want = i == cfg.queue_msgs_ ? 99 : static_cast<int>(i);
    CHECK(r.request_id == want);
    server->ClearRecvHandles(r);
  }
  std::cout << "  ok" << std::endl;
  return 0;
}

/**
 * Listener process of the two-process test.
 * @param port Listener port.
 * @param total Messages to receive before exiting.
 * @return Process exit code.
 */
static int RunServer(int port, int total) {
  TransportPtr server = TransportFactory::Get(
      "0.0.0.0", TransportType::kThallium, TransportMode::kServer, "tcp", port);
  std::cout << "server: listening on port " << port << " at "
            << server->GetAddress() << std::endl;
  ReceiveAll(server.get(), total);
  std::cout << "server: received and verified " << total << " messages"
            << std::endl;
  server.reset();
  ThalliumEngine::Finalize();
  return 0;
}

/**
 * Sender process of the two-process test.
 * @param host Listener host.
 * @param port Listener port.
 * @param threads Sender threads (plain std::threads, one connection each).
 * @param per_thread Messages per thread.
 * @return Process exit code.
 */
static int RunClient(const std::string &host, int port, int threads,
                     int per_thread) {
  std::vector<TransportPtr> conns;
  for (int s = 0; s < threads; ++s) {
    conns.push_back(TransportFactory::Get(
        host, TransportType::kThallium, TransportMode::kClient, "tcp", port));
  }
  std::cout << "client: peer address " << conns[0]->GetAddress() << std::endl;
  std::atomic<int> failures{0};
  std::vector<std::thread> senders;
  auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < threads; ++s) {
    senders.emplace_back(SenderLoop, conns[s].get(), s, per_thread, &failures);
  }
  for (auto &t : senders) t.join();
  double ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
  std::cout << "client: sent " << threads * per_thread << " messages in "
            << ms << " ms, failures=" << failures.load() << std::endl;
  conns.clear();
  ThalliumEngine::Finalize();
  return failures.load() == 0 ? 0 : 1;
}

/**
 * Entry point; see the file comment for the modes.
 * @param argc Argument count.
 * @param argv Arguments.
 * @return Process exit code.
 */
int main(int argc, char **argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "server" && argc == 4) {
    return RunServer(std::atoi(argv[2]), std::atoi(argv[3]));
  }
  if (mode == "client" && argc == 6) {
    return RunClient(argv[2], std::atoi(argv[3]), std::atoi(argv[4]),
                     std::atoi(argv[5]));
  }
  if (mode == "backpressure") return RunBackpressure();
  TestRoundTrip();
  TestBulkMix(64 * 1024);
  TestBulkMix(0);
  TestBulkMix(size_t{1} << 40);
  TestTwoListeners();
  TestConcurrentSenders(8, 200);
  TestPeerDown();
  TestFinalizeRestart();
  std::cout << "All thallium transport tests passed." << std::endl;
  return 0;
}
