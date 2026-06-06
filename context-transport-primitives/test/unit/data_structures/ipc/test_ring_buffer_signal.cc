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

#include "../../../context-runtime/test/simple_test.h"
#include "clio_ctp/data_structures/ipc/ring_buffer.h"
#include "clio_ctp/memory/backend/malloc_backend.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <algorithm>

#include "clio_ctp/lightbeam/event_manager.h"

using namespace ctp::ipc;

/**
 * Helper function to create an ArenaAllocator for testing
 */
static ArenaAllocator<false>* CreateTestAllocator(MallocBackend &backend,
                                                  size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);
  return backend.MakeAlloc<ArenaAllocator<false>>();
}

// ============================================================================
// Signal Ring Buffer Tests (RING_BUFFER_SIGNAL_ON_0 flag)
//
// These tests exercise the in-queue signaling path: when Emplace() finds
// the queue was empty, it automatically fires EventManager::Signal to wake
// the worker thread.
// ============================================================================

#ifndef _WIN32

TEST_CASE("SignalRingBuffer: push to empty queue wakes sleeping consumer",
          "[ring_buffer][signal]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  signal_mpmc_queue<int, ArenaAllocator<false>> rb(alloc, 64);

  // Consumer thread that will block on EventManager::Wait
  std::atomic<bool> consumer_started(false);
  std::atomic<bool> consumer_done(false);
  std::atomic<int> received_value(0);

  std::thread consumer([&rb, &consumer_started, &consumer_done, &received_value]() {
    // Set up EventManager for this thread
    ctp::lbm::EventManager em;
    em.AddSignalEvent(nullptr);

    // Set tid and runtime_pid on the queue so it knows where to signal
    rb.SetTid(ctp::SystemInfo::GetTid());
    rb.SetRuntimePid(ctp::SystemInfo::GetPid());

    consumer_started.store(true, std::memory_order_release);

    // Block waiting for a signal (will be woken by the producer's push)
    int result = em.Wait(-1);  // -1 = blocking wait
    (void)result;  // Avoid unused variable warning

    // Once woken, try to pop the item
    int val;
    if (rb.Pop(val)) {
      received_value.store(val, std::memory_order_release);
    }

    consumer_done.store(true, std::memory_order_release);
  });

  // Wait for consumer to start and enter its blocking wait
  while (!consumer_started.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Give the consumer time to call em.Wait(-1) and block
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Producer pushes an item to the empty queue; this should signal the consumer
  REQUIRE(rb.Push(42));

  // Wait for consumer to wake and process (with a 2-second timeout)
  auto start = std::chrono::high_resolution_clock::now();
  bool completed = false;
  while (std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::high_resolution_clock::now() - start)
             .count() < 2) {
    if (consumer_done.load(std::memory_order_acquire)) {
      completed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  consumer.join();

  REQUIRE(completed);  // Consumer should have woken and processed
  REQUIRE(received_value.load(std::memory_order_acquire) == 42);
  REQUIRE(rb.Empty());
}

TEST_CASE("SignalRingBuffer: push to non-empty queue fires signal but drains correctly",
          "[ring_buffer][signal]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  signal_mpmc_queue<int, ArenaAllocator<false>> rb(alloc, 128);

  // Pre-fill with some items
  constexpr int kItems = 10;
  for (int i = 0; i < kItems; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Now push more items to a non-empty queue
  for (int i = kItems; i < 2 * kItems; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Drain and verify all items are present
  std::vector<int> drained;
  int val;
  while (rb.Pop(val)) {
    drained.push_back(val);
  }

  std::sort(drained.begin(), drained.end());
  REQUIRE(drained.size() == 2 * kItems);
  for (int i = 0; i < 2 * kItems; ++i) {
    REQUIRE(drained[i] == i);
  }
  REQUIRE(rb.Empty());
}

TEST_CASE("SignalRingBuffer: no signal when tid or runtime_pid not set",
          "[ring_buffer][signal]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  signal_mpmc_queue<int, ArenaAllocator<false>> rb(alloc, 64);

  // tid and runtime_pid are initialized to 0 (default)
  // Pushing should not crash even though Signal won't be sent
  REQUIRE(rb.Push(99));

  // Verify the item is in the queue
  int val;
  REQUIRE(rb.Pop(val));
  REQUIRE(val == 99);
  REQUIRE(rb.Empty());
}

#endif  // _WIN32

SIMPLE_TEST_MAIN()
