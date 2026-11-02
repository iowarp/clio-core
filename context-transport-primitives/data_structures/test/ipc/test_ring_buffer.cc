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
#include <string>

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
// Constructor and Destructor Tests
// ============================================================================

TEST_CASE("RingBuffer: constructor with capacity", "[ring_buffer]") {
  printf("[TEST] Creating backend...\n"); fflush(stdout);
  MallocBackend backend;
  printf("[TEST] Creating allocator...\n"); fflush(stdout);
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);
  printf("[TEST] Allocator created at %p\n", (void*)alloc); fflush(stdout);

  {
    printf("[TEST] Creating ring_buffer...\n"); fflush(stdout);
    ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);
    printf("[TEST] ring_buffer created\n"); fflush(stdout);

    REQUIRE(rb.Capacity() == 16);
    REQUIRE(rb.Empty());
    REQUIRE_FALSE(rb.Full());
    REQUIRE(rb.Size() == 0);
  }  // rb destructor runs here (backend auto-destroyed)

}

TEST_CASE("RingBuffer: constructor with small capacity", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 4);

  REQUIRE(rb.Capacity() == 4);
  REQUIRE(rb.Empty());
  REQUIRE(rb.Size() == 0);

}

// ============================================================================
// Push and TryPush Tests
// ============================================================================

TEST_CASE("RingBuffer: single push and pop", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  bool result = rb.Push(42);
  REQUIRE(result);
  REQUIRE(rb.Size() == 1);
  REQUIRE_FALSE(rb.Empty());

  int value;
  result = rb.Pop(value);
  REQUIRE(result);
  REQUIRE(value == 42);
  REQUIRE(rb.Empty());

}

TEST_CASE("RingBuffer: multiple pushes and pops", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  // Push 5 elements
  for (int i = 1; i <= 5; ++i) {
    REQUIRE(rb.Push(i * 10));
  }

  REQUIRE(rb.Size() == 5);
  REQUIRE_FALSE(rb.Empty());

  // Pop all elements and verify order
  for (int i = 1; i <= 5; ++i) {
    int value;
    REQUIRE(rb.Pop(value));
    REQUIRE(value == i * 10);
  }

  REQUIRE(rb.Empty());
}

TEST_CASE("RingBuffer: push to capacity", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 4);

  // Push 4 elements (capacity)
  REQUIRE(rb.Push(1));
  REQUIRE(rb.Push(2));
  REQUIRE(rb.Push(3));
  REQUIRE(rb.Push(4));

  REQUIRE(rb.Size() == 4);
  REQUIRE(rb.Full());

  // Next push should fail
  REQUIRE_FALSE(rb.Push(5));

}

TEST_CASE("RingBuffer: wrap-around", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 4);

  // Fill the buffer
  for (int i = 1; i <= 3; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Pop one element
  int value;
  REQUIRE(rb.Pop(value));
  REQUIRE(value == 1);

  // Push another element (should wrap around)
  REQUIRE(rb.Push(4));
  REQUIRE(rb.Size() == 3);

  // Pop remaining elements
  for (int expected : {2, 3, 4}) {
    REQUIRE(rb.Pop(value));
    REQUIRE(value == expected);
  }

  REQUIRE(rb.Empty());
}

TEST_CASE("RingBuffer: multiple wrap-arounds", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 8);

  // Do multiple cycles of push and pop
  for (int cycle = 0; cycle < 3; ++cycle) {
    for (int i = 1; i <= 6; ++i) {
      REQUIRE(rb.Push(cycle * 100 + i));
    }

    for (int i = 1; i <= 6; ++i) {
      int value;
      REQUIRE(rb.Pop(value));
      REQUIRE(value == cycle * 100 + i);
    }
  }

  REQUIRE(rb.Empty());
}

// ============================================================================
// TryPush and TryPop Tests
// ============================================================================

TEST_CASE("RingBuffer: try_push and try_pop", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  REQUIRE(rb.TryPush(99));
  REQUIRE(rb.Size() == 1);

  int value;
  REQUIRE(rb.TryPop(value));
  REQUIRE(value == 99);
  REQUIRE(rb.Empty());

}

TEST_CASE("RingBuffer: try_pop on empty buffer", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  int value;
  REQUIRE_FALSE(rb.TryPop(value));
  REQUIRE(rb.Empty());

}

TEST_CASE("RingBuffer: try_push on full buffer", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 4);

  // Fill to capacity
  REQUIRE(rb.TryPush(1));
  REQUIRE(rb.TryPush(2));
  REQUIRE(rb.TryPush(3));
  REQUIRE(rb.TryPush(4));

  // Buffer is now full
  REQUIRE(rb.Full());

  // Next try_push should fail
  REQUIRE_FALSE(rb.TryPush(5));

}

// ============================================================================
// Clear and Reset Tests
// ============================================================================

TEST_CASE("RingBuffer: clear", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  // Push some elements
  for (int i = 1; i <= 5; ++i) {
    REQUIRE(rb.Push(i));
  }

  REQUIRE(rb.Size() == 5);

  // Clear the buffer
  rb.Clear();

  REQUIRE(rb.Empty());
  REQUIRE(rb.Size() == 0);

  // Should be able to push again
  REQUIRE(rb.Push(100));
  REQUIRE(rb.Size() == 1);

}

TEST_CASE("RingBuffer: reset", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  // Push, pop, push again (to trigger wrap-around)
  for (int i = 1; i <= 10; ++i) {
    REQUIRE(rb.Push(i));
  }

  int value;
  for (int i = 0; i < 5; ++i) {
    REQUIRE(rb.Pop(value));
  }

  REQUIRE(rb.Size() == 5);

  // Reset the buffer
  rb.Reset();

  REQUIRE(rb.Empty());
  REQUIRE(rb.Size() == 0);

  // Should work normally after reset
  REQUIRE(rb.Push(999));
  REQUIRE(rb.Pop(value));
  REQUIRE(value == 999);

}

// ============================================================================
// Capacity and Size Tests
// ============================================================================

TEST_CASE("RingBuffer: capacity consistency", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 32);

  REQUIRE(rb.Capacity() == 32);

  // Push some elements
  for (int i = 0; i < 10; ++i) {
    rb.Push(i);
  }

  // Capacity should not change
  REQUIRE(rb.Capacity() == 32);

}

TEST_CASE("RingBuffer: size after operations", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  REQUIRE(rb.Size() == 0);

  rb.Push(1);
  REQUIRE(rb.Size() == 1);

  rb.Push(2);
  REQUIRE(rb.Size() == 2);

  int value;
  rb.Pop(value);
  REQUIRE(rb.Size() == 1);

  rb.Pop(value);
  REQUIRE(rb.Size() == 0);

}

// ============================================================================
// String and Complex Type Tests
// ============================================================================

TEST_CASE("RingBuffer: string elements", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<std::string, ArenaAllocator<false>> rb(alloc, 8);

  // Push strings
  REQUIRE(rb.Push(std::string("hello")));
  REQUIRE(rb.Push(std::string("world")));

  // Pop and verify
  std::string value;
  REQUIRE(rb.Pop(value));
  REQUIRE(value == "hello");

  REQUIRE(rb.Pop(value));
  REQUIRE(value == "world");

  REQUIRE(rb.Empty());

}

// ============================================================================
// Complex Data Type Tests
// ============================================================================

// Custom structure for testing
struct KeyValue {
  int key;
  int value;

  KeyValue() : key(0), value(0) {}
  KeyValue(int k, int v) : key(k), value(v) {}
};

TEST_CASE("RingBuffer: Complex data type", "[ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  ring_buffer<KeyValue, ArenaAllocator<false>> rb(alloc, 8);

  // Push KeyValue pairs
  REQUIRE(rb.Push(KeyValue(10, 20)));
  REQUIRE(rb.Push(KeyValue(30, 40)));

  // Pop and verify
  KeyValue entry;
  REQUIRE(rb.Pop(entry));
  REQUIRE(entry.key == 10);
  REQUIRE(entry.value == 20);

  REQUIRE(rb.Pop(entry));
  REQUIRE(entry.key == 30);
  REQUIRE(entry.value == 40);

}

SIMPLE_TEST_MAIN()
