/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#include <clio_ctp/lightbeam/thallium_transport.h>
#include <clio_ctp/lightbeam/transport_factory_impl.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace ctp::lbm;

class TestMeta : public LbmMeta<> {
 public:
  int request_id = 0;
  std::string operation;

  template <typename Ar>
  void serialize(Ar& ar) {
    LbmMeta<>::serialize(ar);
    ar(request_id, operation);
  }
};

template <typename MetaT>
ClientInfo RecvWithRetry(Transport* server, MetaT& meta, int max_ms = 5000) {
  int attempts = 0;
  while (true) {
    auto info = server->Recv(meta);
    if (info.rc == 0) return info;
    if (info.rc != EAGAIN) return info;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (++attempts > max_ms) {
      std::cerr << "RecvWithRetry timed out\n";
      return info;
    }
  }
}

void TestRoundTrip() {
  std::cout << "\n==== Thallium round-trip (no bulk) ====\n";
  // Server addr arg is unused for thallium server-mode; the engine
  // listens on whatever the protocol implies. We use a placeholder.
  auto server = std::make_unique<ThalliumTransport>(
      TransportMode::kServer, "127.0.0.1", "tcp", 0);
  std::string server_addr = server->GetAddress();
  std::cout << "  server addr = " << server_addr << "\n";
  assert(!server_addr.empty());

  auto client = std::make_unique<ThalliumTransport>(
      TransportMode::kClient, server_addr, "tcp", 0);

  TestMeta send_meta;
  send_meta.request_id = 42;
  send_meta.operation = "hello-thallium";

  int rc = client->Send(send_meta);
  assert(rc == 0);

  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 42);
  assert(recv_meta.operation == "hello-thallium");
  std::cout << "  ok (id=" << recv_meta.request_id << ", op='"
            << recv_meta.operation << "')\n";
}

void TestBulkTransfer() {
  std::cout << "\n==== Thallium round-trip (with BULK_XFER) ====\n";
  auto server = std::make_unique<ThalliumTransport>(
      TransportMode::kServer, "127.0.0.1", "tcp", 0);
  std::string server_addr = server->GetAddress();
  auto client = std::make_unique<ThalliumTransport>(
      TransportMode::kClient, server_addr, "tcp", 0);

  std::vector<char> payload(4096);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>(i & 0xff);
  }
  ctp::ipc::FullPtr<char> ptr;
  ptr.ptr_ = payload.data();
  ptr.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  ptr.shm_.off_ = reinterpret_cast<size_t>(payload.data());

  TestMeta send_meta;
  send_meta.request_id = 7;
  send_meta.operation = "bulk";
  Bulk bulk = client->Expose(ptr, payload.size(), BULK_XFER);
  send_meta.send.push_back(bulk);
  send_meta.send_bulks = 1;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 7);
  assert(recv_meta.recv.size() == 1);
  assert(recv_meta.recv[0].size == payload.size());
  assert(recv_meta.recv[0].data.ptr_ != nullptr);
  bool ok = std::memcmp(recv_meta.recv[0].data.ptr_, payload.data(),
                        payload.size()) == 0;
  assert(ok);
  server->ClearRecvHandles(recv_meta);
  std::cout << "  ok (" << payload.size() << " bytes)\n";
}

int main() {
  TestRoundTrip();
  TestBulkTransfer();
  std::cout << "\nAll thallium transport tests passed.\n";
  return 0;
}
