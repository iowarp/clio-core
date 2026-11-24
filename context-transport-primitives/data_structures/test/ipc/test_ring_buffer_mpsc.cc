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

#include "../../../../context-runtime/test/simple_test.h"
#include "clio_ctp/data_structures/ipc/ring_buffer.h"
#include "clio_ctp/memory/backend/malloc_backend.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

using namespace ctp::ipc;

/**
 * Helper function to create an ArenaAllocator for testing
 */
ArenaAllocator<false>* CreateTestAllocator(MallocBackend &backend,
                                           size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);
  return backend.MakeAlloc<ArenaAllocator<false>>();
}

// ============================================================================
// MPSC Ring Buffer Tests (Multiple Producer Single Consumer)
// ============================================================================

TEST_CASE("MPSC RingBuffer: single producer baseline", "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 64);

  // Baseline: single "producer" pushes 32 items
  for (int i = 0; i < 32; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Consumer pops all items
  for (int i = 0; i < 32; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == i);
  }

  REQUIRE(rb.Empty());

}

TEST_CASE("MPSC RingBuffer: concurrent producers", "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 128);

  // 4 producer threads, each pushing 25 items
  std::vector<std::thread> producers;
  std::atomic<int> push_count(0);

  for (int producer_id = 0; producer_id < 4; ++producer_id) {
    producers.emplace_back([&rb, &push_count, producer_id]() {
      for (int i = 0; i < 25; ++i) {
        int value = producer_id * 1000 + i;
        if (rb.Push(value)) {
          push_count.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // Wait for all producers to complete
  for (auto &t : producers) {
    t.join();
  }

  // Verify that all 100 items were pushed
  REQUIRE(push_count.load() == 100);

  // Consumer pops all items (order may vary due to concurrent access)
  std::vector<int> popped_values;
  int val;
  while (rb.Pop(val)) {
    popped_values.push_back(val);
  }

  REQUIRE(popped_values.size() == 100);
  REQUIRE(rb.Empty());

}

TEST_CASE("MPSC RingBuffer: producer/consumer coordination",
          "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 32);

  std::atomic<bool> producer_done(false);
  std::atomic<int> consumed_count(0);

  // Producer thread: continuously push items until stopped
  std::thread producer([&rb, &producer_done]() {
    for (int i = 0; i < 100; ++i) {
      rb.Push(i);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    producer_done.store(true, std::memory_order_release);
  });

  // Consumer thread: continuously pop items until producer is done
  std::thread consumer([&rb, &producer_done, &consumed_count]() {
    while (true) {
      int val;
      if (rb.Pop(val)) {
        consumed_count.fetch_add(1, std::memory_order_relaxed);
      } else if (producer_done.load(std::memory_order_acquire)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  producer.join();
  consumer.join();

  // Verify all items were consumed
  REQUIRE(consumed_count.load() == 100);
  REQUIRE(rb.Empty());

}

TEST_CASE("MPSC RingBuffer: contention under capacity limit",
          "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  // Small buffer to induce contention (use non-waiting variant so Push returns false when full)
  using test_mpsc_no_wait = ring_buffer<int, ArenaAllocator<false>,
      (RING_BUFFER_MPSC_FLAGS | RING_BUFFER_FIXED_SIZE | RING_BUFFER_ERROR_ON_NO_SPACE)>;
  test_mpsc_no_wait rb(alloc, 16);

  std::atomic<int> total_pushed(0);
  std::atomic<int> total_popped(0);
  std::atomic<bool> producers_done(false);

  // 3 producer threads push items concurrently
  std::vector<std::thread> producers;
  for (int producer_id = 0; producer_id < 3; ++producer_id) {
    producers.emplace_back([&rb, &total_pushed, producer_id]() {
      for (int i = 0; i < 50; ++i) {
        int value = producer_id * 1000 + i;
        if (rb.Push(value)) {
          total_pushed.fetch_add(1, std::memory_order_relaxed);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    });
  }

  // Consumer thread drains the buffer concurrently
  std::thread consumer([&rb, &total_popped, &producers_done]() {
    while (true) {
      int val;
      if (rb.Pop(val)) {
        total_popped.fetch_add(1, std::memory_order_relaxed);
      } else if (producers_done.load(std::memory_order_acquire)) {
        // Drain any remaining items
        while (rb.Pop(val)) {
          total_popped.fetch_add(1, std::memory_order_relaxed);
        }
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  // Wait for all producers to complete
  for (auto &t : producers) {
    t.join();
  }
  producers_done.store(true, std::memory_order_release);

  // Wait for consumer to finish
  consumer.join();

  // Verify all pushed items were consumed
  REQUIRE(total_popped.load() == total_pushed.load());
  REQUIRE(rb.Empty());

}

TEST_CASE("MPSC RingBuffer: stress test with varying producer count",
          "[ring_buffer][mpsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  // Larger buffer for stress test to accommodate all producers
  mpsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 512);

  std::atomic<int> total_pushed(0);
  std::vector<std::thread> producers;

  // Launch 8 producer threads pushing in parallel
  for (int producer_id = 0; producer_id < 8; ++producer_id) {
    producers.emplace_back([&rb, &total_pushed, producer_id]() {
      for (int i = 0; i < 50; ++i) {
        int value = producer_id * 1000 + i;
        if (rb.Push(value)) {
          total_pushed.fetch_add(1, std::memory_order_relaxed);
        } else {
          // Retry on failure
          std::this_thread::sleep_for(std::chrono::microseconds(10));
          if (rb.Push(value)) {
            total_pushed.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    });
  }

  // Wait for all producers
  for (auto &t : producers) {
    t.join();
  }

  // Consumer drains the entire buffer
  int consumed = 0;
  int val;
  while (rb.Pop(val)) {
    consumed++;
  }

  // Verify all pushes eventually succeeded
  REQUIRE(total_pushed.load() == 400);
  REQUIRE(consumed == 400);
  REQUIRE(rb.Empty());

}

SIMPLE_TEST_MAIN()
