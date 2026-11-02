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
// SPSC Ring Buffer Tests (Single Producer Single Consumer)
// ============================================================================

TEST_CASE("SPSC RingBuffer: sequential workload", "[ring_buffer][spsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  spsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 64);

  // Sequential: push all, then pop all
  for (size_t i = 0; i < 32; ++i) {
    REQUIRE(rb.Push(static_cast<int>(i)));
  }

  for (size_t i = 0; i < 32; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == static_cast<int>(i));
  }

}

TEST_CASE("SPSC RingBuffer: interleaved push/pop", "[ring_buffer][spsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  spsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 64);

  // Interleaved: alternate push and pop operations
  for (size_t i = 0; i < 32; ++i) {
    REQUIRE(rb.Push(static_cast<int>(i)));
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == static_cast<int>(i));
  }

}

TEST_CASE("SPSC RingBuffer: wrap-around operations", "[ring_buffer][spsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  spsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 32);

  // Fill to capacity
  for (size_t i = 0; i < 32; ++i) {
    REQUIRE(rb.Push(static_cast<int>(i)));
  }

  // Pop half to create wrap-around opportunity
  for (size_t i = 0; i < 16; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == static_cast<int>(i));
  }

  // Refill the popped portion
  for (size_t i = 32; i < 48; ++i) {
    REQUIRE(rb.Push(static_cast<int>(i)));
  }

  // Pop remaining items (tests wrap-around)
  for (size_t i = 16; i < 48; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == static_cast<int>(i));
  }

}

TEST_CASE("SPSC RingBuffer: capacity limit enforcement", "[ring_buffer][spsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  spsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 8);

  // Fill to capacity
  for (int i = 0; i < 8; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Verify buffer is full
  REQUIRE(rb.Full());
  REQUIRE(rb.Size() == 8);

  // Next push should fail
  REQUIRE_FALSE(rb.Push(99));

  // Pop one item
  int val;
  REQUIRE(rb.Pop(val));
  REQUIRE(val == 0);
  REQUIRE_FALSE(rb.Full());

  // Now push should succeed
  REQUIRE(rb.Push(99));
  REQUIRE(rb.Full());

}

TEST_CASE("SPSC RingBuffer: mixed operations", "[ring_buffer][spsc]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  spsc_ring_buffer<int, ArenaAllocator<false>> rb(alloc, 16);

  // Scenario: push several, pop some, push more, pop all
  for (int i = 0; i < 10; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Pop 5
  for (int i = 0; i < 5; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == i);
  }

  // Push 5 more
  for (int i = 10; i < 15; ++i) {
    REQUIRE(rb.Push(i));
  }

  // Pop all remaining 10 items
  for (int i = 5; i < 15; ++i) {
    int val;
    REQUIRE(rb.Pop(val));
    REQUIRE(val == i);
  }

  REQUIRE(rb.Empty());

}

SIMPLE_TEST_MAIN()
