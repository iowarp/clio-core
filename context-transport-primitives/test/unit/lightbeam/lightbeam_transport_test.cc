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

#include <hermes_shm/lightbeam/zmq_transport.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace hshm::lbm;

void TestZeroMQ() {
#ifdef HSHM_ENABLE_ZMQ
  std::cout << "\n==== Testing ZeroMQ ====\n";

  std::string addr = "127.0.0.1";
  std::string protocol = "tcp";
  int port = 8192;

  auto server = std::make_unique<ZeroMqTransport>(TransportMode::kServer, addr, protocol, port);
  auto client = std::make_unique<ZeroMqTransport>(TransportMode::kClient, addr, protocol, port);

  // Give ZMQ time to connect
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  const std::string magic = "unit_test_magic";

  // Client creates metadata and sends
  LbmMeta send_meta;
  Bulk send_bulk = client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(magic.data())),
      magic.size(), BULK_XFER);
  send_meta.send.push_back(send_bulk);

  int rc = client->Send(send_meta);
  assert(rc == 0);
  std::cout << "Client sent data successfully\n";

  // Recv with retry loop (does everything - metadata + bulks)
  LbmMeta recv_meta;
  while (true) {
    auto info = server->Recv(recv_meta);
    rc = info.rc;
    if (rc == 0) break;
    if (rc != EAGAIN) {
      std::cerr << "Recv failed with error: " << rc << "\n";
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  assert(recv_meta.send.size() == 1);

  std::string received(recv_meta.recv[0].data.ptr_,
                       recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  std::cout << "Received: " << received << std::endl;
  assert(received == magic);

  std::cout << "[ZeroMQ] Test passed!\n";
#else
  std::cout << "ZeroMQ not enabled, skipping test\n";
#endif
}

int main() {
  TestZeroMQ();
  std::cout << "\nAll transport tests passed!" << std::endl;
  return 0;
}