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
