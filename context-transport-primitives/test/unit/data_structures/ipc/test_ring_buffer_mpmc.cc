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
// MPMC Ring Buffer Tests (Multiple Producer Multiple Consumer)
//
// These tests exercise the RING_BUFFER_LOCK_POP path: Pop() is serialized
// with ctp::Mutex so multiple consumer threads can drain the buffer safely.
// The stress tests verify (a) no duplicate deliveries and (b) no lost items.
// ============================================================================

TEST_CASE("MPMC RingBuffer: single consumer sanity", "[ring_buffer][mpmc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  mpmc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 64);

  for (int i = 0; i < 32; ++i) {
    REQUIRE(rb.Push(i));
  }
  for (int i = 0; i < 32; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == i);
  }
  REQUIRE(rb.Empty());
}

TEST_CASE("MPMC RingBuffer: concurrent consumers drain a pre-filled buffer",
          "[ring_buffer][mpmc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  constexpr int kItems = 1000;
  mpmc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, kItems + 16);

  // Pre-fill the buffer with a known set of values.
  for (int i = 0; i < kItems; ++i) {
    REQUIRE(rb.Push(i));
  }

  // 8 consumer threads race to pop. Each thread records its claimed values
  // in a private vector so we can verify exactly the produced set was
  // consumed exactly once across all consumers.
  constexpr int kConsumers = 8;
  std::vector<std::thread> consumers;
  std::vector<std::vector<int>> per_consumer(kConsumers);

  for (int c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&rb, &per_consumer, c]() {
      int val;
      while (rb.Pop(val)) {
        per_consumer[c].push_back(val);
      }
    });
  }
  for (auto &t : consumers) t.join();

  std::vector<int> all;
  for (auto &v : per_consumer) {
    all.insert(all.end(), v.begin(), v.end());
  }
  std::sort(all.begin(), all.end());

  REQUIRE(all.size() == static_cast<size_t>(kItems));
  for (int i = 0; i < kItems; ++i) {
    REQUIRE(all[i] == i);  // exactly the produced set, no dups, no losses
  }
  REQUIRE(rb.Empty());
}

TEST_CASE("MPMC RingBuffer: producers and consumers running in parallel",
          "[ring_buffer][mpmc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kPerProducer = 500;
  constexpr int kTotal = kProducers * kPerProducer;

  mpmc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 128);

  std::atomic<int> produced(0);
  std::atomic<int> consumed(0);
  std::atomic<bool> producers_done(false);

  // Encode (producer_id, seq) into a single int so we can detect duplicates.
  // Producer i emits values [i*kPerProducer .. (i+1)*kPerProducer).
  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&rb, &produced, p]() {
      int base = p * kPerProducer;
      for (int i = 0; i < kPerProducer; ++i) {
        // Push retries on transient full conditions (WAIT_FOR_SPACE mode
        // blocks internally, so Push really only fails on capacity exhaustion
        // — but be defensive).
        while (!rb.Push(base + i)) {
          std::this_thread::yield();
        }
        produced.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  std::vector<std::thread> consumer_threads;
  std::vector<std::vector<int>> per_consumer(kConsumers);
  for (int c = 0; c < kConsumers; ++c) {
    consumer_threads.emplace_back([&rb, &consumed, &producers_done,
                                   &per_consumer, c]() {
      int val;
      while (true) {
        if (rb.Pop(val)) {
          per_consumer[c].push_back(val);
          consumed.fetch_add(1, std::memory_order_relaxed);
        } else if (producers_done.load(std::memory_order_acquire)) {
          // Drain remainder.
          while (rb.Pop(val)) {
            per_consumer[c].push_back(val);
            consumed.fetch_add(1, std::memory_order_relaxed);
          }
          return;
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
      }
    });
  }

  for (auto &t : producers) t.join();
  producers_done.store(true, std::memory_order_release);
  for (auto &t : consumer_threads) t.join();

  REQUIRE(produced.load() == kTotal);
  REQUIRE(consumed.load() == kTotal);
  REQUIRE(rb.Empty());

  // Verify that the union of consumer outputs is exactly the produced set,
  // with no duplicates. This is the load-bearing MPMC correctness check.
  std::vector<int> all;
  for (auto &v : per_consumer) {
    all.insert(all.end(), v.begin(), v.end());
  }
  std::sort(all.begin(), all.end());
  REQUIRE(all.size() == static_cast<size_t>(kTotal));
  for (int i = 0; i < kTotal; ++i) {
    REQUIRE(all[i] == i);
  }
}

TEST_CASE("MPMC RingBuffer: heavy pop contention with small buffer",
          "[ring_buffer][mpmc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  // Small buffer + many consumers maximizes contention on pop_lock_.
  constexpr int kItems = 5000;
  constexpr int kConsumers = 16;
  mpmc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 32);

  std::atomic<int> consumed(0);
  std::atomic<bool> producer_done(false);

  std::thread producer([&rb, &producer_done]() {
    for (int i = 0; i < kItems; ++i) {
      while (!rb.Push(i)) {
        std::this_thread::yield();
      }
    }
    producer_done.store(true, std::memory_order_release);
  });

  std::vector<std::thread> consumers;
  std::vector<std::vector<int>> per_consumer(kConsumers);
  for (int c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&rb, &consumed, &producer_done, &per_consumer, c]() {
      int val;
      while (true) {
        if (rb.Pop(val)) {
          per_consumer[c].push_back(val);
          consumed.fetch_add(1, std::memory_order_relaxed);
        } else if (producer_done.load(std::memory_order_acquire)) {
          while (rb.Pop(val)) {
            per_consumer[c].push_back(val);
            consumed.fetch_add(1, std::memory_order_relaxed);
          }
          return;
        }
      }
    });
  }

  producer.join();
  for (auto &t : consumers) t.join();

  REQUIRE(consumed.load() == kItems);
  REQUIRE(rb.Empty());

  std::vector<int> all;
  for (auto &v : per_consumer) {
    all.insert(all.end(), v.begin(), v.end());
  }
  std::sort(all.begin(), all.end());
  REQUIRE(all.size() == static_cast<size_t>(kItems));
  for (int i = 0; i < kItems; ++i) {
    REQUIRE(all[i] == i);
  }
}

SIMPLE_TEST_MAIN()
