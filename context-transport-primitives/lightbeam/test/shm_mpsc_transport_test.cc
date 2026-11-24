/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

// Unit tests for the named MPSC SHM transport (issue #642).
//  - single-thread 16KB and 1MB transfers
//  - multiple producer threads each transferring 1MB into one Recv consumer

#include <catch2/catch_all.hpp>

#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "clio_ctp/lightbeam/shm_mpsc_transport.h"

using ctp::lbm::ShmMpscTransport;

namespace {

// Build a self-describing buffer: the first 4 bytes encode `idx`, the rest is a
// deterministic pattern derived from the byte index and idx. This lets the
// consumer verify a received message without knowing which producer sent it.
std::vector<char> MakePattern(size_t n, uint32_t idx) {
  std::vector<char> v(n);
  if (n >= 4) std::memcpy(v.data(), &idx, 4);
  for (size_t i = 4; i < n; ++i) {
    v[i] = static_cast<char>((i + idx) & 0xFF);
  }
  return v;
}

bool CheckPattern(const std::vector<char>& v, size_t n) {
  if (v.size() != n) return false;
  uint32_t idx = 0;
  if (n >= 4) std::memcpy(&idx, v.data(), 4);
  for (size_t i = 4; i < n; ++i) {
    if (static_cast<uint8_t>(v[i]) != static_cast<uint8_t>((i + idx) & 0xFF)) {
      return false;
    }
  }
  return true;
}

// Run one producer (its own connection) sending `size` bytes with pattern idx.
void RunProducer(const std::string& name, size_t size, uint32_t idx,
                 std::atomic<int>* ok) {
  std::vector<char> data = MakePattern(size, idx);
  ShmMpscTransport cli;
  if (!cli.ClientInit(name)) return;
  if (cli.SendBytes(data.data(), data.size()) == 0) ok->fetch_add(1);
  cli.Shutdown();
}

// Run one producer over a connection SHARED with other threads. SendBytes must
// serialize per connection so the messages don't interleave under one conn_id.
void RunProducerShared(ShmMpscTransport* cli, size_t size, uint32_t idx,
                       std::atomic<int>* ok) {
  std::vector<char> data = MakePattern(size, idx);
  if (cli->SendBytes(data.data(), data.size()) == 0) ok->fetch_add(1);
}

// Drain `count` complete messages from `srv`, asserting each decodes a distinct
// producer index in [0,count) and matches that producer's pattern.
void DrainAndVerify(ShmMpscTransport* srv, int count, size_t size) {
  std::vector<bool> seen(count, false);
  for (int i = 0; i < count; ++i) {
    std::vector<char> out;
    ctp::u64 conn = 0;
    REQUIRE(srv->RecvBytes(out, &conn, 0) == 0);
    REQUIRE(out.size() == size);
    uint32_t idx = 0;
    std::memcpy(&idx, out.data(), 4);
    REQUIRE(idx < static_cast<uint32_t>(count));
    REQUIRE(!seen[idx]);
    seen[idx] = true;
    REQUIRE(CheckPattern(out, size));
  }
}

// Like DrainAndVerify but each producer index carries its OWN message size
// (sizes[idx]). This is the shape the runtime response path actually produces:
// SerializeOut of a ReaddirTask yields a variable-length archive whose size
// depends on how many directory entries it holds, and many such responses race
// through one client's MPSC server. The consumer must size each message from
// its own header, not from the first message it happened to reassemble.
void DrainAndVerifyVariable(ShmMpscTransport* srv, int count,
                            const std::vector<size_t>& sizes) {
  std::vector<bool> seen(count, false);
  for (int i = 0; i < count; ++i) {
    std::vector<char> out;
    ctp::u64 conn = 0;
    REQUIRE(srv->RecvBytes(out, &conn, 0) == 0);
    uint32_t idx = 0;
    REQUIRE(out.size() >= 4);
    std::memcpy(&idx, out.data(), 4);
    REQUIRE(idx < static_cast<uint32_t>(count));
    REQUIRE(!seen[idx]);
    seen[idx] = true;
    // The crux: the delivered length must equal THIS producer's size, and the
    // bytes must be its pattern -- not truncated to some other message's size.
    REQUIRE(out.size() == sizes[idx]);
    REQUIRE(CheckPattern(out, sizes[idx]));
  }
}

// One producer over a SHARED connection, sending its own variable-size message.
void RunProducerSharedVariable(ShmMpscTransport* cli, size_t size, uint32_t idx,
                               std::atomic<int>* ok) {
  std::vector<char> data = MakePattern(size, idx);
  if (cli->SendBytes(data.data(), data.size()) == 0) ok->fetch_add(1);
}

}  // namespace

TEST_CASE("ShmMpsc - single thread 16KB", "[shm_mpsc][single]") {
  const std::string name = "clio-mpsc-utest-16k";
  const size_t kSize = 16 * 1024;
  ShmMpscTransport srv;
  REQUIRE(srv.ServerInit(name));

  std::atomic<int> ok{0};
  std::thread prod(RunProducer, name, kSize, 7u, &ok);

  std::vector<char> out;
  ctp::u64 conn = 0;
  REQUIRE(srv.RecvBytes(out, &conn, 0) == 0);
  REQUIRE(CheckPattern(out, kSize));

  prod.join();
  REQUIRE(ok.load() == 1);
  srv.Shutdown();
}

TEST_CASE("ShmMpsc - single thread 1MB", "[shm_mpsc][single][large]") {
  const std::string name = "clio-mpsc-utest-1m";
  const size_t kSize = 1024 * 1024;
  ShmMpscTransport srv;
  REQUIRE(srv.ServerInit(name));  // 1MB streamed through the default 128KB ring

  std::atomic<int> ok{0};
  std::thread prod(RunProducer, name, kSize, 42u, &ok);

  std::vector<char> out;
  ctp::u64 conn = 0;
  REQUIRE(srv.RecvBytes(out, &conn, 0) == 0);
  REQUIRE(CheckPattern(out, kSize));

  prod.join();
  REQUIRE(ok.load() == 1);
  srv.Shutdown();
}

// Concurrency stress, scenario A: many threads, each with its OWN SHM client,
// all sending into one server. Small ring (default 128KB) vs 512KB messages
// forces heavy wraparound and slot contention across independent connections.
TEST_CASE("ShmMpsc - concurrent per-thread clients", "[shm_mpsc][concurrent]") {
  const std::string name = "clio-mpsc-utest-conc-own";
  const int kProducers = 8;
  const size_t kSize = 512 * 1024;
  ShmMpscTransport srv;
  REQUIRE(srv.ServerInit(name));  // default 128KB ring

  std::atomic<int> ok{0};
  std::vector<std::thread> prods;
  prods.reserve(kProducers);
  for (int t = 0; t < kProducers; ++t) {
    prods.emplace_back(RunProducer, name, kSize, static_cast<uint32_t>(t), &ok);
  }

  DrainAndVerify(&srv, kProducers, kSize);

  for (auto& p : prods) p.join();
  REQUIRE(ok.load() == kProducers);
  srv.Shutdown();
}

// Concurrency stress, scenario B: many threads SHARING a single SHM client
// (mirrors IpcManager::GetOrCreateShmConn caching one conn per destination and
// many workers sending through it). All share one conn_id, so SendBytes must
// serialize each message; otherwise interleaved chunks corrupt the consumer's
// per-conn reassembly.
TEST_CASE("ShmMpsc - concurrent shared client", "[shm_mpsc][concurrent]") {
  const std::string name = "clio-mpsc-utest-conc-shared";
  const int kProducers = 8;
  const size_t kSize = 512 * 1024;
  ShmMpscTransport srv;
  REQUIRE(srv.ServerInit(name));  // default 128KB ring

  ShmMpscTransport cli;
  REQUIRE(cli.ClientInit(name));  // ONE client, shared by every producer thread

  std::atomic<int> ok{0};
  std::vector<std::thread> prods;
  prods.reserve(kProducers);
  for (int t = 0; t < kProducers; ++t) {
    prods.emplace_back(RunProducerShared, &cli, kSize, static_cast<uint32_t>(t),
                       &ok);
  }

  DrainAndVerify(&srv, kProducers, kSize);

  for (auto& p : prods) p.join();
  REQUIRE(ok.load() == kProducers);
  cli.Shutdown();
  srv.Shutdown();
}

TEST_CASE("ShmMpsc - multi producer 1MB", "[shm_mpsc][multi]") {
  const std::string name = "clio-mpsc-utest-multi";
  const int kProducers = 4;
  const size_t kSize = 1024 * 1024;
  ShmMpscTransport srv;
  REQUIRE(srv.ServerInit(name, 256 * 1024));

  std::atomic<int> ok{0};
  std::vector<std::thread> prods;
  prods.reserve(kProducers);
  for (int t = 0; t < kProducers; ++t) {
    prods.emplace_back(RunProducer, name, kSize, static_cast<uint32_t>(t), &ok);
  }

  // The single consumer drains kProducers complete messages; each must decode a
  // distinct producer index and match that producer's pattern.
  std::vector<bool> seen(kProducers, false);
  for (int i = 0; i < kProducers; ++i) {
    std::vector<char> out;
    ctp::u64 conn = 0;
    REQUIRE(srv.RecvBytes(out, &conn, 0) == 0);
    REQUIRE(out.size() == kSize);
    uint32_t idx = 0;
    std::memcpy(&idx, out.data(), 4);
    REQUIRE(idx < static_cast<uint32_t>(kProducers));
    REQUIRE(!seen[idx]);
    seen[idx] = true;
    REQUIRE(CheckPattern(out, kSize));
  }

  for (auto& p : prods) p.join();
  REQUIRE(ok.load() == kProducers);
  srv.Shutdown();
}

// Regression for the CFS SHM Readdir corruption (issue #768 / PR #769):
// concurrent, VARIABLE-SIZE messages racing through one client's MPSC server.
// This mirrors many AsyncReaddir responses returning to a single waiting
// client thread, each SerializeOut'd to a different length. The consumer
// de-muxes by conn id and sizes the reassembly buffer from the first chunk it
// sees for that conn; if a second message for the same/short buffer interleaves
// it truncates -> the client's GlobalDeserialize reads past end of data.
//
// Distinct message SIZES per producer are what the earlier uniform-size
// "concurrent shared client" test could not exercise: with every buffer the
// same length, a mis-sized reassembly still happens to be the right length.
TEST_CASE("ShmMpsc - concurrent shared client, variable sizes",
          "[shm_mpsc][concurrent][regression]") {
  const std::string name = "clio-mpsc-utest-conc-var";
  const int kProducers = 16;
  ShmMpscTransport srv;
  REQUIRE(srv.ServerInit(name));

  ShmMpscTransport cli;
  REQUIRE(cli.ClientInit(name));  // ONE shared client, as the runtime caches

  // A spread of sizes straddling the 32KB chunk boundary: single-chunk small,
  // multi-chunk large, and non-multiples so the tail chunk is partial.
  std::vector<size_t> sizes(kProducers);
  for (int t = 0; t < kProducers; ++t) {
    sizes[t] = 1024 + static_cast<size_t>(t) * 37 * 1024 + (t * 111);
  }

  std::atomic<int> ok{0};
  std::vector<std::thread> prods;
  prods.reserve(kProducers);
  for (int t = 0; t < kProducers; ++t) {
    prods.emplace_back(RunProducerSharedVariable, &cli, sizes[t],
                       static_cast<uint32_t>(t), &ok);
  }

  DrainAndVerifyVariable(&srv, kProducers, sizes);

  for (auto& p : prods) p.join();
  REQUIRE(ok.load() == kProducers);
  cli.Shutdown();
  srv.Shutdown();
}

// Tighter variant: the SAME connection sends many back-to-back messages of
// alternating small/large size from one thread (serial, not raced), while a
// second thread on the same conn also sends. Exercises conn-id reuse across
// messages -- st.total must reset per message, never carry over.
TEST_CASE("ShmMpsc - repeated messages reuse conn id",
          "[shm_mpsc][regression]") {
  const std::string name = "clio-mpsc-utest-reuse";
  const int kMsgs = 32;
  ShmMpscTransport srv;
  REQUIRE(srv.ServerInit(name));

  ShmMpscTransport cli;
  REQUIRE(cli.ClientInit(name));

  std::atomic<int> ok{0};
  // Alternating tiny (single-chunk) and large (multi-chunk) messages from one
  // thread, so consecutive messages under the same conn id differ in size.
  std::thread prod([&] {
    for (int i = 0; i < kMsgs; ++i) {
      size_t sz = (i % 2 == 0) ? 200 : (200u * 1024u + i);
      std::vector<char> data = MakePattern(sz, static_cast<uint32_t>(i));
      if (cli.SendBytes(data.data(), data.size()) == 0) ok.fetch_add(1);
    }
  });

  std::vector<bool> seen(kMsgs, false);
  for (int i = 0; i < kMsgs; ++i) {
    std::vector<char> out;
    ctp::u64 conn = 0;
    REQUIRE(srv.RecvBytes(out, &conn, 0) == 0);
    REQUIRE(out.size() >= 4);
    uint32_t idx = 0;
    std::memcpy(&idx, out.data(), 4);
    REQUIRE(idx < static_cast<uint32_t>(kMsgs));
    REQUIRE(!seen[idx]);
    seen[idx] = true;
    size_t expect = (idx % 2 == 0) ? 200 : (200u * 1024u + idx);
    REQUIRE(out.size() == expect);
    REQUIRE(CheckPattern(out, expect));
  }

  prod.join();
  REQUIRE(ok.load() == kMsgs);
  cli.Shutdown();
  srv.Shutdown();
}

TEST_CASE("ShmMpsc - high-level Send/Recv (metadata + bulk)", "[shm_mpsc][meta]") {
  const std::string name = "clio-mpsc-utest-meta";
  const std::string magic = "hello-mpsc-high-level-Send-Recv-payload-0123456789";
  ShmMpscTransport srv;
  REQUIRE(srv.ServerInit(name));

  std::atomic<int> ok{0};
  std::thread prod([&] {
    ShmMpscTransport cli;
    if (!cli.ClientInit(name)) return;
    // Expose `magic` as a private-memory BULK_XFER (alloc_id null => bytes are
    // copied into the message).
    ctp::ipc::FullPtr<char> ptr;
    ptr.ptr_ = const_cast<char*>(magic.data());
    ptr.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    ptr.shm_.off_ = reinterpret_cast<size_t>(magic.data());
    ctp::lbm::LbmMeta<> send_meta;
    send_meta.send.push_back(cli.Expose(ptr, magic.size(), BULK_XFER));
    if (cli.Send(send_meta) == 0) ok.fetch_add(1);
    cli.Shutdown();
  });

  ctp::lbm::LbmMeta<> recv_meta;
  ctp::lbm::ClientInfo info = srv.Recv(recv_meta);
  REQUIRE(info.rc == 0);
  REQUIRE(recv_meta.recv.size() == 1);
  REQUIRE(recv_meta.recv[0].size == magic.size());
  std::string got(recv_meta.recv[0].data.ptr_,
                  recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  REQUIRE(got == magic);

  prod.join();
  REQUIRE(ok.load() == 1);
  srv.Shutdown();
}
