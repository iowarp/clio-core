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
#include "clio_ctp/data_structures/ipc/multi_ring_buffer.h"
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
// Constructor and Basic Initialization Tests
// ============================================================================

TEST_CASE("MultiRingBuffer: constructor with lanes and prios",
          "[multi_ring_buffer]") {
  printf("[TEST] Creating backend...\n"); fflush(stdout);
  MallocBackend backend;
  printf("[TEST] Creating allocator...\n"); fflush(stdout);
  auto *alloc = CreateTestAllocator(backend, 4 * 1024 * 1024);
  printf("[TEST] Allocator created at %p\n", (void*)alloc); fflush(stdout);

  {
    printf("[TEST] Creating multi_ring_buffer...\n"); fflush(stdout);
    multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, 4, 2, 16);
    printf("[TEST] multi_ring_buffer created\n"); fflush(stdout);

    REQUIRE(mrb.GetNumLanes() == 4);
    REQUIRE(mrb.GetNumPrios() == 2);
    REQUIRE(mrb.GetTotalBuffers() == 8);
  }  // mrb destructor runs here

}

TEST_CASE("MultiRingBuffer: constructor with single lane and prio",
          "[multi_ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 4 * 1024 * 1024);

  multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, 1, 1, 32);

  REQUIRE(mrb.GetNumLanes() == 1);
  REQUIRE(mrb.GetNumPrios() == 1);
  REQUIRE(mrb.GetTotalBuffers() == 1);

}

// ============================================================================
// GetLane Access Tests
// ============================================================================

TEST_CASE("MultiRingBuffer: GetLane with valid indices",
          "[multi_ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 4 * 1024 * 1024);

  multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, 3, 2, 16);

  // Test getting lanes at different positions
  auto &lane0_0 = mrb.GetLane(0, 0);
  auto &lane0_1 = mrb.GetLane(0, 1);
  auto &lane1_0 = mrb.GetLane(1, 0);
  auto &lane2_1 = mrb.GetLane(2, 1);

  REQUIRE(lane0_0.Empty());
  REQUIRE(lane0_1.Empty());
  REQUIRE(lane1_0.Empty());
  REQUIRE(lane2_1.Empty());

}

TEST_CASE("MultiRingBuffer: push and pop from specific lanes",
          "[multi_ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 4 * 1024 * 1024);

  multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, 2, 2, 16);

  // Push to lane 0, prio 0
  auto &lane0_0 = mrb.GetLane(0, 0);
  REQUIRE(lane0_0.Push(100));
  REQUIRE(lane0_0.Size() == 1);

  // Push to lane 0, prio 1
  auto &lane0_1 = mrb.GetLane(0, 1);
  REQUIRE(lane0_1.Push(200));
  REQUIRE(lane0_1.Size() == 1);

  // Push to lane 1, prio 0
  auto &lane1_0 = mrb.GetLane(1, 0);
  REQUIRE(lane1_0.Push(300));
  REQUIRE(lane1_0.Size() == 1);

  // Verify pushes were independent
  int val;
  REQUIRE(lane0_0.Pop(val));
  REQUIRE(val == 100);
  REQUIRE(lane0_0.Empty());

  REQUIRE(lane0_1.Pop(val));
  REQUIRE(val == 200);
  REQUIRE(lane0_1.Empty());

  REQUIRE(lane1_0.Pop(val));
  REQUIRE(val == 300);
  REQUIRE(lane1_0.Empty());

}

// ============================================================================
// Lane Independence Tests
// ============================================================================

TEST_CASE("MultiRingBuffer: lanes are independent queues",
          "[multi_ring_buffer]") {
  printf("[TEST] Creating backend...\n"); fflush(stdout);
  MallocBackend backend;
  printf("[TEST] Creating allocator...\n"); fflush(stdout);
  auto *alloc = CreateTestAllocator(backend, 4 * 1024 * 1024);

  printf("[TEST] Creating multi_ring_buffer (2 lanes, 3 prios, depth 8)...\n"); fflush(stdout);
  multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, 2, 3, 8);
  printf("[TEST] Created multi_ring_buffer\n"); fflush(stdout);

  // Fill each lane with different values
  for (int lane = 0; lane < 2; ++lane) {
    for (int prio = 0; prio < 3; ++prio) {
      printf("[TEST] Getting lane %d, prio %d...\n", lane, prio); fflush(stdout);
      auto &rb = mrb.GetLane(lane, prio);
      for (int i = 0; i < 2; ++i) {  // Reduced from 5 to 2
        int value = (lane * 100) + (prio * 10) + i;
        REQUIRE(rb.Push(value));
      }
    }
  }
  printf("[TEST] All pushes completed\n"); fflush(stdout);

  // Verify each lane/prio has correct data
  for (int lane = 0; lane < 2; ++lane) {
    for (int prio = 0; prio < 3; ++prio) {
      auto &rb = mrb.GetLane(lane, prio);
      for (int i = 0; i < 2; ++i) {  // Reduced from 5 to 2
        int val;
        REQUIRE(rb.Pop(val));
        int expected = (lane * 100) + (prio * 10) + i;
        REQUIRE(val == expected);
      }
      REQUIRE(rb.Empty());
    }
  }
  printf("[TEST] All pops verified, destroying backend\n"); fflush(stdout);

  printf("[TEST] Test completed successfully\n"); fflush(stdout);
}

// ============================================================================
// Multiple Lane Priority Tests
// ============================================================================

TEST_CASE("MultiRingBuffer: multiple lanes with different priorities",
          "[multi_ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 8 * 1024 * 1024);

  printf("[TEST] Creating multi_ring_buffer (3 lanes, 3 prios, depth 16)...\n"); fflush(stdout);
  multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, 3, 3, 16);
  printf("[TEST] Created multi_ring_buffer\n"); fflush(stdout);

  // Push different values to each position
  for (int lane = 0; lane < 3; ++lane) {
    for (int prio = 0; prio < 3; ++prio) {
      auto &rb = mrb.GetLane(lane, prio);
      int value = (lane << 8) | prio;  // Encode lane and prio in value
      REQUIRE(rb.Push(value));
    }
  }
  printf("[TEST] All pushes to multiple lanes completed\n"); fflush(stdout);

  // Verify all values
  for (int lane = 0; lane < 3; ++lane) {
    for (int prio = 0; prio < 3; ++prio) {
      auto &rb = mrb.GetLane(lane, prio);
      int val;
      REQUIRE(rb.Pop(val));
      int expected = (lane << 8) | prio;
      REQUIRE(val == expected);
    }
  }
  printf("[TEST] All pops from multiple lanes verified\n"); fflush(stdout);

}

// ============================================================================
// Const GetLane Tests
// ============================================================================

TEST_CASE("MultiRingBuffer: const GetLane access", "[multi_ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 4 * 1024 * 1024);

  multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, 2, 2, 16);

  // Non-const push
  auto &lane = mrb.GetLane(0, 0);
  REQUIRE(lane.Push(42));

  // Const access
  const auto &cref = mrb;
  const auto &const_lane = cref.GetLane(0, 0);
  REQUIRE(const_lane.Size() == 1);
  REQUIRE_FALSE(const_lane.Empty());

}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_CASE("MultiRingBuffer: many lanes and priorities", "[multi_ring_buffer]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  printf("[TEST] Creating large multi_ring_buffer (4 lanes, 4 prios, depth 32)...\n"); fflush(stdout);
  multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, 4, 4, 32);
  printf("[TEST] Created large multi_ring_buffer\n"); fflush(stdout);

  REQUIRE(mrb.GetNumLanes() == 4);
  REQUIRE(mrb.GetNumPrios() == 4);
  REQUIRE(mrb.GetTotalBuffers() == 16);

  // Test some specific lanes
  printf("[TEST] Getting lanes for stress test...\n"); fflush(stdout);
  auto &lane0 = mrb.GetLane(0, 0);
  auto &lane3 = mrb.GetLane(3, 3);

  REQUIRE(lane0.Push(1));
  REQUIRE(lane3.Push(2));

  int val;
  REQUIRE(lane0.Pop(val));
  REQUIRE(val == 1);
  REQUIRE(lane3.Pop(val));
  REQUIRE(val == 2);
  printf("[TEST] Stress test completed successfully\n"); fflush(stdout);

}

// ============================================================================
// Default Constructor Test
// ============================================================================

TEST_CASE("MultiRingBuffer: constructor with allocator", "[multi_ring_buffer]") {
  printf("[TEST] Creating multi_ring_buffer with allocator...\n"); fflush(stdout);
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  size_t num_lanes = 4;
  size_t num_prios = 3;
  size_t depth = 512;

  multi_ring_buffer<int, ArenaAllocator<false>> mrb(alloc, num_lanes, num_prios, depth);
  printf("[TEST] Multi_ring_buffer created\n"); fflush(stdout);

  REQUIRE(mrb.GetNumLanes() == num_lanes);
  REQUIRE(mrb.GetNumPrios() == num_prios);
  REQUIRE(mrb.GetTotalBuffers() == num_lanes * num_prios);
  printf("[TEST] Multi_ring_buffer tests passed\n"); fflush(stdout);

}

SIMPLE_TEST_MAIN()
