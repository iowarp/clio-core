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

#include <hermes_shm/lightbeam/event_manager.h>
#include <hermes_shm/lightbeam/socket_transport.h>
#include <hermes_shm/lightbeam/transport_factory_impl.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <atomic>
#ifndef _WIN32
#include <signal.h>
#include <unistd.h>
#endif

using namespace hshm::lbm;

// Custom metadata class for transport tests
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

// Custom EventAction that records invocations
class RecordAction : public EventAction {
 public:
  std::atomic<int> call_count{0};
  EventInfo last_event{};

  void Run(const EventInfo& event) override {
    last_event = event;
    call_count.fetch_add(1);
  }
};

#ifndef _WIN32
void TestConstruction() {
  std::cout << "\n==== Testing EventManager Construction ====\n";

  EventManager em;
  assert(em.GetEpollFd() >= 0);
  assert(em.GetSignalFd() == -1);

  std::cout << "[EventManager Construction] Test passed!\n";
}

void TestAddEventBasic() {
  std::cout << "\n==== Testing EventManager AddEvent Basic ====\n";

  EventManager em;

  // Create a pipe
  int pipefd[2];
  int rc = pipe(pipefd);
  assert(rc == 0);

  int event_id = em.AddEvent(pipefd[0], kDefaultReadEvent);
  assert(event_id >= 0);

  // Write to pipe to trigger event
  char buf = 'x';
  ssize_t written = write(pipefd[1], &buf, 1);
  assert(written == 1);

  // Wait should return 1 (one event fired)
  int nfds = em.Wait(100000);  // 100ms timeout
  assert(nfds == 1);

  close(pipefd[0]);
  close(pipefd[1]);

  std::cout << "[EventManager AddEvent Basic] Test passed!\n";
}

void TestAddEventWithAction() {
  std::cout << "\n==== Testing EventManager AddEvent With Action ====\n";

  EventManager em;
  RecordAction action;

  int pipefd[2];
  int rc = pipe(pipefd);
  assert(rc == 0);

  int event_id = em.AddEvent(pipefd[0], kDefaultReadEvent, &action);
  assert(event_id >= 0);

  // Write to trigger
  char buf = 'y';
  ssize_t written = write(pipefd[1], &buf, 1);
  assert(written == 1);

  int nfds = em.Wait(100000);
  assert(nfds == 1);
  assert(action.call_count.load() == 1);
  assert(action.last_event.trigger_.fd_ == pipefd[0]);
  assert(action.last_event.trigger_.event_id_ == event_id);
  assert((action.last_event.events_ & kDefaultReadEvent) != 0);

  close(pipefd[0]);
  close(pipefd[1]);

  std::cout << "[EventManager AddEvent With Action] Test passed!\n";
}

void TestAddSignalEvent() {
  std::cout << "\n==== Testing EventManager AddSignalEvent ====\n";

  EventManager em;

  int event_id = em.AddSignalEvent();
  assert(event_id >= 0);
  assert(em.GetSignalFd() >= 0);

  // Signal the current thread
  pid_t pid = getpid();
  pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  int rc = EventManager::Signal(pid, tid);
  assert(rc == 0);

  int nfds = em.Wait(100000);
  assert(nfds == 1);

  std::cout << "[EventManager AddSignalEvent] Test passed!\n";
}

void TestSignalEventWithAction() {
  std::cout << "\n==== Testing EventManager Signal Event With Action ====\n";

  EventManager em;
  RecordAction action;

  int event_id = em.AddSignalEvent(&action);
  assert(event_id >= 0);

  pid_t pid = getpid();
  pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  int rc = EventManager::Signal(pid, tid);
  assert(rc == 0);

  int nfds = em.Wait(100000);
  assert(nfds == 1);
  assert(action.call_count.load() == 1);
  assert(action.last_event.trigger_.fd_ == em.GetSignalFd());

  std::cout << "[EventManager Signal Event With Action] Test passed!\n";
}

void TestWaitTimeout() {
  std::cout << "\n==== Testing EventManager Wait Timeout ====\n";

  EventManager em;

  int pipefd[2];
  int rc = pipe(pipefd);
  assert(rc == 0);

  em.AddEvent(pipefd[0], kDefaultReadEvent);

  // Don't write anything - should timeout
  auto start = std::chrono::steady_clock::now();
  int nfds = em.Wait(100000);  // 100ms
  auto end = std::chrono::steady_clock::now();

  assert(nfds == 0);

  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  // Should have waited at least ~50ms (allowing some slack)
  assert(elapsed_ms >= 50);

  close(pipefd[0]);
  close(pipefd[1]);

  std::cout << "[EventManager Wait Timeout] Test passed!\n";
}

void TestMultipleEvents() {
  std::cout << "\n==== Testing EventManager Multiple Events ====\n";

  EventManager em;

  int pipe1[2] = {}, pipe2[2] = {}, pipe3[2] = {};
  assert(pipe(pipe1) == 0);
  assert(pipe(pipe2) == 0);
  assert(pipe(pipe3) == 0);

  em.AddEvent(pipe1[0], kDefaultReadEvent);
  em.AddEvent(pipe2[0], kDefaultReadEvent);
  em.AddEvent(pipe3[0], kDefaultReadEvent);

  // Trigger 2 out of 3 pipes
  char buf = 'a';
  (void)write(pipe1[1], &buf, 1);
  (void)write(pipe3[1], &buf, 1);

  // Give a moment for epoll to see events
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  int nfds = em.Wait(100000);
  assert(nfds == 2);

  close(pipe1[0]); close(pipe1[1]);
  close(pipe2[0]); close(pipe2[1]);
  close(pipe3[0]); close(pipe3[1]);

  std::cout << "[EventManager Multiple Events] Test passed!\n";
}
#endif  // !_WIN32

void TestSocketTransportWithEM() {
  std::cout << "\n==== Testing SocketTransport With EventManager ====\n";

  std::string addr = "127.0.0.1";
  int port = 8400;

  auto server = std::make_unique<SocketTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<SocketTransport>(
      TransportMode::kClient, addr, "tcp", port);

  // Register EventManager on server
  EventManager em;
  server->RegisterEventManager(em);

  // Client sends data
  const char* data = "event_manager_test_data";
  size_t size = strlen(data);

  TestMeta send_meta;
  send_meta.request_id = 55;
  send_meta.operation = "em_test";
  Bulk bulk = client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data)), size, BULK_XFER);
  send_meta.send.push_back(bulk);
  send_meta.send_bulks = 1;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  // Wait for event on server
  int nfds = em.Wait(500000);  // 500ms
  assert(nfds >= 1);

  // Server should now be able to recv
  TestMeta recv_meta;
  int attempts = 0;
  while (true) {
    auto info = server->Recv(recv_meta);
    rc = info.rc;
    if (rc == 0) break;
    if (rc != EAGAIN) {
      std::cerr << "Recv failed with error: " << rc << "\n";
      assert(false);
    }
    // Re-wait on the event manager
    nfds = em.Wait(100000);
    if (++attempts > 50) {
      std::cerr << "Recv timed out\n";
      assert(false);
    }
  }

  assert(recv_meta.request_id == 55);
  assert(recv_meta.operation == "em_test");
  std::string received(recv_meta.recv[0].data.ptr_,
                       recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  assert(received == data);

  server->ClearRecvHandles(recv_meta);

  std::cout << "[SocketTransport With EventManager] Test passed!\n";
}

#ifdef HSHM_ENABLE_ZMQ
#include <hermes_shm/lightbeam/zmq_transport.h>

void TestZmqTransportWithEM() {
  std::cout << "\n==== Testing ZmqTransport With EventManager ====\n";

  std::string addr = "127.0.0.1";
  int port = 8401;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", port);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  EventManager em;
  server->RegisterEventManager(em);

  const char* data = "zmq_em_test_data";
  size_t size = strlen(data);

  TestMeta send_meta;
  send_meta.request_id = 66;
  send_meta.operation = "zmq_em";
  Bulk bulk = client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data)), size, BULK_XFER);
  send_meta.send.push_back(bulk);
  send_meta.send_bulks = 1;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  // Wait for event and recv (ZMQ FD is edge-triggered, so data may not be
  // immediately available after epoll fires — retry with short waits)
  TestMeta recv_meta;
  ClientInfo info;
  for (int attempt = 0; attempt < 20; ++attempt) {
    int nfds = em.Wait(500000);
    (void)nfds;
    info = server->Recv(recv_meta);
    if (info.rc == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  assert(info.rc == 0);
  assert(recv_meta.request_id == 66);
  assert(recv_meta.operation == "zmq_em");

  std::string received(recv_meta.recv[0].data.ptr_,
                       recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  assert(received == data);

  server->ClearRecvHandles(recv_meta);

  std::cout << "[ZmqTransport With EventManager] Test passed!\n";
}
#endif

int main() {
#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);  // Prevent SIGPIPE from terminating process
  TestConstruction();
  TestAddEventBasic();
  TestAddEventWithAction();
  TestAddSignalEvent();
  TestSignalEventWithAction();
  TestWaitTimeout();
  TestMultipleEvents();
#endif
  TestSocketTransportWithEM();

#ifdef HSHM_ENABLE_ZMQ
  TestZmqTransportWithEM();
#else
  std::cout << "\n[Skipped] ZmqTransport With EventManager (ZMQ not enabled)\n";
#endif

  std::cout << "\nAll EventManager tests passed!" << std::endl;
  return 0;
}
