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

#include <hermes_shm/lightbeam/socket_transport.h>
#include <hermes_shm/lightbeam/transport_factory_impl.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace hshm::lbm;

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

// Helper: recv with retry loop
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

void TestTcpBasic() {
  std::cout << "\n==== Testing Socket TCP Basic ====\n";

  std::string addr = "127.0.0.1";
  int port = 9100;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  const char* data1 = "Hello, Socket World!";
  const char* data2 = "Testing Socket Transport";
  size_t size1 = strlen(data1);
  size_t size2 = strlen(data2);

  TestMeta send_meta;
  send_meta.request_id = 42;
  send_meta.operation = "tcp_test";
  send_meta.send.push_back(client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data1)), size1, BULK_XFER));
  send_meta.send.push_back(client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data2)), size2, BULK_XFER));
  send_meta.send_bulks = 2;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 42);
  assert(recv_meta.operation == "tcp_test");

  std::string received1(recv_meta.recv[0].data.ptr_,
                         recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  std::string received2(recv_meta.recv[1].data.ptr_,
                         recv_meta.recv[1].data.ptr_ + recv_meta.recv[1].size);
  assert(received1 == data1);
  assert(received2 == data2);

  server->ClearRecvHandles(recv_meta);
  std::cout << "[Socket TCP Basic] Test passed!\n";
}

void TestMultipleBulks() {
  std::cout << "\n==== Testing Socket Multiple Bulks ====\n";

  std::string addr = "127.0.0.1";
  int port = 9101;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  std::vector<std::string> data_chunks = {"Chunk 1", "Chunk 2 is longer",
                                          "Chunk 3", "Final chunk 4"};

  LbmMeta<> send_meta;
  for (const auto& chunk : data_chunks) {
    send_meta.send.push_back(client->Expose(
        hipc::FullPtr<char>(const_cast<char*>(chunk.data())),
        chunk.size(), BULK_XFER));
    send_meta.send_bulks++;
  }

  int rc = client->Send(send_meta);
  assert(rc == 0);

  LbmMeta<> recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.send.size() == data_chunks.size());

  for (size_t i = 0; i < data_chunks.size(); ++i) {
    std::string received(recv_meta.recv[i].data.ptr_,
                         recv_meta.recv[i].data.ptr_ + recv_meta.recv[i].size);
    assert(received == data_chunks[i]);
  }

  server->ClearRecvHandles(recv_meta);
  std::cout << "[Socket Multiple Bulks] Test passed!\n";
}

void TestUnixDomain() {
  std::cout << "\n==== Testing Socket Unix Domain ====\n";

  std::string sock_path = "/tmp/lightbeam_full_test.sock";

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, sock_path, "ipc", 0);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, sock_path, "ipc", 0);

  const char* data = "IPC test data over Unix socket";
  size_t size = strlen(data);

  TestMeta send_meta;
  send_meta.request_id = 99;
  send_meta.operation = "ipc_test";
  send_meta.send.push_back(client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data)), size, BULK_XFER));
  send_meta.send_bulks = 1;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 99);
  assert(recv_meta.operation == "ipc_test");

  std::string received(recv_meta.recv[0].data.ptr_,
                       recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  assert(received == data);

  server->ClearRecvHandles(recv_meta);
  std::cout << "[Socket Unix Domain] Test passed!\n";
}

void TestMetadataOnly() {
  std::cout << "\n==== Testing Socket Metadata Only ====\n";

  std::string addr = "127.0.0.1";
  int port = 9102;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  TestMeta send_meta;
  send_meta.request_id = 7;
  send_meta.operation = "ping";
  send_meta.send_bulks = 0;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 7);
  assert(recv_meta.operation == "ping");
  assert(recv_meta.send.empty());

  std::cout << "[Socket Metadata Only] Test passed!\n";
}

void TestBulkExpose() {
  std::cout << "\n==== Testing Socket BULK_EXPOSE ====\n";

  std::string addr = "127.0.0.1";
  int port = 9103;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  const char* data = "expose_only_data";
  size_t size = strlen(data);

  LbmMeta<> send_meta;
  send_meta.send.push_back(client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data)), size, BULK_EXPOSE));
  send_meta.send_bulks = 0;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  LbmMeta<> recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.recv.size() == 1);
  // BULK_EXPOSE: no data transferred, pointer should be null
  assert(recv_meta.recv[0].data.IsNull());
  assert(recv_meta.recv[0].size == size);

  std::cout << "[Socket BULK_EXPOSE] Test passed!\n";
}

void TestLargeTransfer() {
  std::cout << "\n==== Testing Socket Large Transfer (1MB) ====\n";

  std::string addr = "127.0.0.1";
  int port = 9104;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  size_t large_size = 1024 * 1024;
  std::vector<char> large_data(large_size);
  for (size_t i = 0; i < large_size; ++i) {
    large_data[i] = static_cast<char>('A' + (i % 26));
  }

  LbmMeta<> send_meta;
  send_meta.send.push_back(client->Expose(
      hipc::FullPtr<char>(large_data.data()), large_size, BULK_XFER));
  send_meta.send_bulks = 1;

  // Send in thread to avoid TCP buffer deadlock with large data
  int send_rc = -1;
  std::thread sender([&]() {
    send_rc = client->Send(send_meta);
  });

  LbmMeta<> recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.recv.size() == 1);
  assert(recv_meta.recv[0].size == large_size);

  sender.join();
  assert(send_rc == 0);

  int mismatches = 0;
  for (size_t i = 0; i < large_size; ++i) {
    if (recv_meta.recv[0].data.ptr_[i] != large_data[i]) {
      mismatches++;
    }
  }
  assert(mismatches == 0);

  server->ClearRecvHandles(recv_meta);
  std::cout << "[Socket Large Transfer] Test passed! (" << large_size << " bytes)\n";
}

void TestGetAddress() {
  std::cout << "\n==== Testing Socket GetAddress ====\n";

  std::string addr = "127.0.0.1";
  int port = 9105;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);

  assert(server->GetAddress() == "127.0.0.1");

  std::cout << "[Socket GetAddress] Test passed!\n";
}

void TestClearRecvHandles() {
  std::cout << "\n==== Testing Socket ClearRecvHandles ====\n";

  std::string addr = "127.0.0.1";
  int port = 9107;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  const char* data = "clear_handles_test";
  size_t size = strlen(data);

  LbmMeta<> send_meta;
  send_meta.send.push_back(client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data)), size, BULK_XFER));
  send_meta.send_bulks = 1;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  LbmMeta<> recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);

  // After recv, data pointer should be non-null (malloc'd buffer)
  assert(recv_meta.recv[0].data.ptr_ != nullptr);

  // Clear handles should free the buffers
  server->ClearRecvHandles(recv_meta);
  assert(recv_meta.recv[0].data.ptr_ == nullptr);

  std::cout << "[Socket ClearRecvHandles] Test passed!\n";
}

void TestBidirectional() {
  std::cout << "\n==== Testing Socket Bidirectional ====\n";

  std::string addr = "127.0.0.1";
  int port = 9108;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  // Client -> Server
  const char* request_data = "client_request";
  TestMeta send_meta;
  send_meta.request_id = 1;
  send_meta.operation = "request";
  send_meta.send.push_back(client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(request_data)),
      strlen(request_data), BULK_XFER));
  send_meta.send_bulks = 1;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 1);
  assert(info.fd_ >= 0);

  std::string received(recv_meta.recv[0].data.ptr_,
                       recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  assert(received == request_data);
  server->ClearRecvHandles(recv_meta);

  // Server -> Client (using client_info_.fd_)
  const char* response_data = "server_response";
  TestMeta resp_meta;
  resp_meta.request_id = 2;
  resp_meta.operation = "response";
  resp_meta.client_info_.fd_ = info.fd_;
  resp_meta.send.push_back(server->Expose(
      hipc::FullPtr<char>(const_cast<char*>(response_data)),
      strlen(response_data), BULK_XFER));
  resp_meta.send_bulks = 1;

  rc = server->Send(resp_meta);
  assert(rc == 0);

  // Client receives
  TestMeta client_recv;
  auto client_info = RecvWithRetry(client.get(), client_recv);
  assert(client_info.rc == 0);
  assert(client_recv.request_id == 2);
  assert(client_recv.operation == "response");

  std::string resp_received(client_recv.recv[0].data.ptr_,
                            client_recv.recv[0].data.ptr_ + client_recv.recv[0].size);
  assert(resp_received == response_data);
  client->ClearRecvHandles(client_recv);

  std::cout << "[Socket Bidirectional] Test passed!\n";
}

void TestMultiClient() {
  std::cout << "\n==== Testing Socket Multi-Client ====\n";

  std::string addr = "127.0.0.1";
  int port = 9109;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);

  // Create 3 clients
  auto client1 = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);
  auto client2 = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);
  auto client3 = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  // Each client sends with different request_id
  for (int i = 1; i <= 3; ++i) {
    SocketTransport* client = nullptr;
    if (i == 1) client = client1.get();
    else if (i == 2) client = client2.get();
    else client = client3.get();

    std::string payload = "client_" + std::to_string(i);
    TestMeta send_meta;
    send_meta.request_id = i;
    send_meta.operation = "multi";
    send_meta.send.push_back(client->Expose(
        hipc::FullPtr<char>(const_cast<char*>(payload.data())),
        payload.size(), BULK_XFER));
    send_meta.send_bulks = 1;
    int rc = client->Send(send_meta);
    assert(rc == 0);
  }

  // Server receives all 3
  std::vector<int> received_fds;
  for (int i = 0; i < 3; ++i) {
    TestMeta recv_meta;
    auto info = RecvWithRetry(server.get(), recv_meta);
    assert(info.rc == 0);
    assert(info.fd_ >= 0);
    received_fds.push_back(info.fd_);
    server->ClearRecvHandles(recv_meta);
  }

  // All 3 should have distinct fds
  assert(received_fds.size() == 3);
  for (size_t i = 0; i < received_fds.size(); ++i) {
    for (size_t j = i + 1; j < received_fds.size(); ++j) {
      assert(received_fds[i] != received_fds[j]);
    }
  }

  std::cout << "[Socket Multi-Client] Test passed!\n";
}

void TestMultiClientWithEM() {
  std::cout << "\n==== Testing Socket Multi-Client With EventManager ====\n";

  std::string addr = "127.0.0.1";
  int port = 9110;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);

  EventManager em;
  server->RegisterEventManager(em);

  auto client1 = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);
  auto client2 = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  // Give server time to accept
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Send from both clients
  for (int i = 1; i <= 2; ++i) {
    SocketTransport* client = (i == 1) ? client1.get() : client2.get();
    std::string payload = "em_client_" + std::to_string(i);
    TestMeta send_meta;
    send_meta.request_id = i;
    send_meta.operation = "em_multi";
    send_meta.send.push_back(client->Expose(
        hipc::FullPtr<char>(const_cast<char*>(payload.data())),
        payload.size(), BULK_XFER));
    send_meta.send_bulks = 1;
    int rc = client->Send(send_meta);
    assert(rc == 0);
  }

  // Server receives using EventManager-driven loop
  int received_count = 0;
  int max_attempts = 100;
  while (received_count < 2 && max_attempts-- > 0) {
    int nfds = em.Wait(100000);  // 100ms
    if (nfds <= 0) continue;

    TestMeta recv_meta;
    auto info = server->Recv(recv_meta);
    if (info.rc == 0) {
      received_count++;
      server->ClearRecvHandles(recv_meta);
    }
  }
  assert(received_count == 2);

  std::cout << "[Socket Multi-Client With EventManager] Test passed!\n";
}

void TestFactoryTcp() {
  std::cout << "\n==== Testing Socket TransportFactory TCP ====\n";

  auto server = TransportFactory::Get(
      "127.0.0.1", TransportType::kSocket, TransportMode::kServer, "tcp", 9111);
  auto client = TransportFactory::Get(
      "127.0.0.1", TransportType::kSocket, TransportMode::kClient, "tcp", 9111);
  assert(server != nullptr);
  assert(client != nullptr);
  assert(server->IsServer());
  assert(client->IsClient());

  std::cout << "[Socket TransportFactory TCP] Test passed!\n";
}

void TestFactoryIpc() {
  std::cout << "\n==== Testing Socket TransportFactory IPC ====\n";

  std::string sock_path = "/tmp/lightbeam_factory_ipc.sock";
  auto server = TransportFactory::Get(
      sock_path, TransportType::kSocket, TransportMode::kServer, "ipc", 0);
  auto client = TransportFactory::Get(
      sock_path, TransportType::kSocket, TransportMode::kClient, "ipc", 0);
  assert(server != nullptr);
  assert(client != nullptr);

  std::cout << "[Socket TransportFactory IPC] Test passed!\n";
}

void TestFactoryDefault() {
  std::cout << "\n==== Testing Socket TransportFactory Default ====\n";

  // Empty protocol and port=0 should use defaults: "tcp", 8193
  auto server = TransportFactory::Get(
      "127.0.0.1", TransportType::kSocket, TransportMode::kServer);
  assert(server != nullptr);
  assert(server->GetAddress() == "127.0.0.1");

  std::cout << "[Socket TransportFactory Default] Test passed!\n";
}

void TestFactoryWithDomain() {
  std::cout << "\n==== Testing Socket TransportFactory With Domain ====\n";

  auto server = TransportFactory::Get(
      "127.0.0.1", TransportType::kSocket, TransportMode::kServer,
      "tcp", 9113, "test_domain");
  assert(server != nullptr);
  assert(server->GetAddress() == "127.0.0.1");

  std::cout << "[Socket TransportFactory With Domain] Test passed!\n";
}

void TestIsServerIsClient() {
  std::cout << "\n==== Testing Socket IsServer/IsClient ====\n";

  std::string addr = "127.0.0.1";

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", 9114);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", 9114);

  assert(server->IsServer());
  assert(!server->IsClient());
  assert(client->IsClient());
  assert(!client->IsServer());

  std::cout << "[Socket IsServer/IsClient] Test passed!\n";
}

void TestAcceptNewClient() {
  std::cout << "\n==== Testing Socket AcceptNewClient ====\n";

  std::string addr = "127.0.0.1";
  int port = 9200;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);

  // Connect a client
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  // Client sends data
  const char* data = "accept_test";
  TestMeta send_meta;
  send_meta.request_id = 77;
  send_meta.operation = "accept";
  send_meta.send.push_back(client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data)), strlen(data), BULK_XFER));
  send_meta.send_bulks = 1;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  // First Recv triggers accept internally
  TestMeta recv_meta;
  int attempts = 0;
  ClientInfo info;
  while (true) {
    info = server->Recv(recv_meta);
    if (info.rc == 0) break;
    if (info.rc != EAGAIN) {
      std::cerr << "Recv failed: " << info.rc << "\n";
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (++attempts > 5000) {
      std::cerr << "Recv timed out after 5000 attempts\n";
      break;
    }
  }
  assert(info.rc == 0);
  assert(recv_meta.request_id == 77);
  // The fd should be a valid accepted client fd
  assert(info.fd_ >= 0);

  server->ClearRecvHandles(recv_meta);
  std::cout << "[Socket AcceptNewClient] Test passed!\n";
}

int main() {
  TestTcpBasic();
  TestMultipleBulks();
#ifndef _WIN32
  TestUnixDomain();
#else
  std::cout << "\n[Skipped] Unix Domain (not supported on Windows)\n";
#endif
  TestMetadataOnly();
  TestBulkExpose();
  TestLargeTransfer();
  TestGetAddress();
  TestClearRecvHandles();
  TestBidirectional();
  TestMultiClient();
  TestMultiClientWithEM();
  TestFactoryTcp();
#ifndef _WIN32
  TestFactoryIpc();
#else
  std::cout << "\n[Skipped] Factory IPC (not supported on Windows)\n";
#endif
  TestFactoryDefault();
  TestFactoryWithDomain();
  TestIsServerIsClient();
  TestAcceptNewClient();

  std::cout << "\nAll socket transport full tests passed!" << std::endl;
  return 0;
}
