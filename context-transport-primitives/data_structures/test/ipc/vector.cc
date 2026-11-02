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
#include "clio_ctp/data_structures/ipc/vector.h"
#include "clio_ctp/memory/backend/malloc_backend.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"
#include <string>
#include <vector>

using namespace ctp::ipc;

/**
 * Helper function to create an ArenaAllocator for testing
 */
ArenaAllocator<false>* CreateTestAllocator(MallocBackend &backend, size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);
  return backend.MakeAlloc<ArenaAllocator<false>>();
}

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_CASE("Vector: constructor default", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  REQUIRE(vec.size() == 0);
  REQUIRE(vec.capacity() == 0);
  REQUIRE(vec.empty());
  REQUIRE(vec.data() == nullptr);

}

TEST_CASE("Vector: constructor with size", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, 5);

  REQUIRE(vec.size() == 5);
  REQUIRE(vec.capacity() >= 5);
  REQUIRE_FALSE(vec.empty());
  REQUIRE(vec.data() != nullptr);

  // Check default initialization
  for (size_t i = 0; i < vec.size(); ++i) {
    REQUIRE(vec[i] == 0);
  }

}

TEST_CASE("Vector: constructor with fill value", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, 5, 42);

  REQUIRE(vec.size() == 5);
  REQUIRE(vec.capacity() >= 5);
  REQUIRE_FALSE(vec.empty());

  // Check fill value
  for (size_t i = 0; i < vec.size(); ++i) {
    REQUIRE(vec[i] == 42);
  }

}

// Copy constructor test removed - copy constructor is intentionally deleted
// for IPC data structures since they must be allocated via allocator,
// not copied on the stack

// Move constructor test removed - move constructor is intentionally deleted
// for IPC data structures since they must be allocated via allocator,
// not moved on the stack

TEST_CASE("Vector: range constructor", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  std::vector<int> std_vec = {10, 20, 30, 40, 50};

  vector<int, ArenaAllocator<false>> vec(alloc, std_vec.begin(), std_vec.end());

  REQUIRE(vec.size() == 5);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);
  REQUIRE(vec[3] == 40);
  REQUIRE(vec[4] == 50);

}

TEST_CASE("Vector: initializer list constructor", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {1, 2, 3, 4, 5});

  REQUIRE(vec.size() == 5);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 2);
  REQUIRE(vec[2] == 3);
  REQUIRE(vec[3] == 4);
  REQUIRE(vec[4] == 5);

}

// ============================================================================
// Element Access Tests
// ============================================================================

TEST_CASE("Vector: at access", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  REQUIRE(vec.at(0) == 10);
  REQUIRE(vec.at(1) == 20);
  REQUIRE(vec.at(2) == 30);

  // Modify through at
  vec.at(1) = 99;
  REQUIRE(vec.at(1) == 99);

}

TEST_CASE("Vector: operator bracket", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);

  // Modify through operator[]
  vec[1] = 99;
  REQUIRE(vec[1] == 99);

}

TEST_CASE("Vector: front element", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  REQUIRE(vec.front() == 10);

  vec.front() = 99;
  REQUIRE(vec.front() == 99);
  REQUIRE(vec[0] == 99);

}

TEST_CASE("Vector: back element", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  REQUIRE(vec.back() == 30);

  vec.back() = 99;
  REQUIRE(vec.back() == 99);
  REQUIRE(vec[2] == 99);

}

TEST_CASE("Vector: data pointer", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  int* ptr = vec.data();
  REQUIRE(ptr != nullptr);
  REQUIRE(ptr[0] == 10);
  REQUIRE(ptr[1] == 20);
  REQUIRE(ptr[2] == 30);

  // Modify through pointer
  ptr[1] = 99;
  REQUIRE(vec[1] == 99);

}

TEST_CASE("Vector: const access methods", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});
  const vector<int, ArenaAllocator<false>>& const_vec = vec;

  REQUIRE(const_vec.at(0) == 10);
  REQUIRE(const_vec[1] == 20);
  REQUIRE(const_vec.front() == 10);
  REQUIRE(const_vec.back() == 30);
  REQUIRE(const_vec.data() != nullptr);

}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST_CASE("Vector: begin end iterators", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  auto it = vec.begin();
  REQUIRE(*it == 10);
  ++it;
  REQUIRE(*it == 20);
  ++it;
  REQUIRE(*it == 30);
  ++it;
  REQUIRE(it == vec.end());

}

TEST_CASE("Vector: const iterators", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});
  const vector<int, ArenaAllocator<false>>& const_vec = vec;

  auto it = const_vec.begin();
  REQUIRE(*it == 10);
  ++it;
  REQUIRE(*it == 20);
  ++it;
  REQUIRE(*it == 30);
  ++it;
  REQUIRE(it == const_vec.end());

}

TEST_CASE("Vector: cbegin cend", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  auto it = vec.cbegin();
  REQUIRE(*it == 10);
  ++it;
  REQUIRE(*it == 20);
  ++it;
  REQUIRE(*it == 30);
  ++it;
  REQUIRE(it == vec.cend());

}

TEST_CASE("Vector: iterator arithmetic", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30, 40, 50});

  auto it = vec.begin();

  // Test increment
  auto it2 = it;
  ++it2;
  REQUIRE(*it2 == 20);

  // Test decrement
  --it2;
  REQUIRE(*it2 == 10);

  // Test addition
  auto it3 = it + 2;
  REQUIRE(*it3 == 30);

  // Test subtraction
  auto it4 = it3 - 1;
  REQUIRE(*it4 == 20);

  // Test difference
  REQUIRE((it3 - it) == 2);

  // Test subscript
  REQUIRE(it[2] == 30);

  // Test compound assignment
  it += 2;
  REQUIRE(*it == 30);
  it -= 1;
  REQUIRE(*it == 20);

}

TEST_CASE("Vector: iterator comparison", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  auto it1 = vec.begin();
  auto it2 = vec.begin();
  auto it3 = vec.begin() + 1;

  REQUIRE(it1 == it2);
  REQUIRE(it1 != it3);
  REQUIRE(it1 < it3);
  REQUIRE(it1 <= it3);
  REQUIRE(it3 > it1);
  REQUIRE(it3 >= it1);

}

TEST_CASE("Vector: range loop", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30, 40, 50});

  int sum = 0;
  for (int val : vec) {
    sum += val;
  }
  REQUIRE(sum == 150);

  // Modify through range loop
  for (int& val : vec) {
    val *= 2;
  }
  REQUIRE(vec[0] == 20);
  REQUIRE(vec[1] == 40);
  REQUIRE(vec[2] == 60);

}

TEST_CASE("Vector: reverse iteration", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  auto it = vec.end();
  --it;
  REQUIRE(*it == 30);
  --it;
  REQUIRE(*it == 20);
  --it;
  REQUIRE(*it == 10);
  REQUIRE(it == vec.begin());

}

// ============================================================================
// Modification Tests
// ============================================================================

TEST_CASE("Vector: push_back copy", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  int val1 = 10;
  int val2 = 20;
  vec.push_back(val1);
  vec.push_back(val2);

  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);

}

TEST_CASE("Vector: push_back move", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  vec.push_back(10);
  vec.push_back(20);

  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);

}

TEST_CASE("Vector: emplace_back", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  vec.emplace_back(10);
  vec.emplace_back(20);
  vec.emplace_back(30);

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);

}

TEST_CASE("Vector: push_back growth", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  // Add elements beyond initial capacity to trigger growth
  for (int i = 0; i < 100; ++i) {
    vec.push_back(i);
  }

  REQUIRE(vec.size() == 100);
  for (int i = 0; i < 100; ++i) {
    REQUIRE(vec[i] == i);
  }

}

TEST_CASE("Vector: insert copy", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 30, 40});

  int val = 20;
  vec.insert(vec.begin() + 1, val);

  REQUIRE(vec.size() == 4);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);
  REQUIRE(vec[3] == 40);

}

TEST_CASE("Vector: insert move", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 30, 40});

  vec.insert(vec.begin() + 1, 20);

  REQUIRE(vec.size() == 4);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);
  REQUIRE(vec[3] == 40);

}

TEST_CASE("Vector: emplace", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 30, 40});

  vec.emplace(vec.begin() + 1, 20);

  REQUIRE(vec.size() == 4);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);
  REQUIRE(vec[3] == 40);

}

TEST_CASE("Vector: erase single", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30, 40});

  vec.erase(vec.begin() + 1);

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 30);
  REQUIRE(vec[2] == 40);

}

TEST_CASE("Vector: erase range", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30, 40, 50});

  vec.erase(vec.begin() + 1, vec.begin() + 4);

  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 50);

}

TEST_CASE("Vector: clear", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30, 40, 50});

  size_t old_capacity = vec.capacity();
  vec.clear();

  REQUIRE(vec.size() == 0);
  REQUIRE(vec.empty());
  REQUIRE(vec.capacity() == old_capacity); // Capacity unchanged

}

TEST_CASE("Vector: multiple operations", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  // Build up
  vec.push_back(10);
  vec.push_back(20);
  vec.push_back(30);
  REQUIRE(vec.size() == 3);

  // Insert
  vec.insert(vec.begin() + 1, 15);
  REQUIRE(vec.size() == 4);
  REQUIRE(vec[1] == 15);

  // Erase
  vec.erase(vec.begin() + 2);
  REQUIRE(vec.size() == 3);

  // Verify final state
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 15);
  REQUIRE(vec[2] == 30);

}

// ============================================================================
// Capacity Tests
// ============================================================================

TEST_CASE("Vector: size method", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);
  REQUIRE(vec.size() == 0);

  vec.push_back(10);
  REQUIRE(vec.size() == 1);

  vec.push_back(20);
  REQUIRE(vec.size() == 2);

  vec.clear();
  REQUIRE(vec.size() == 0);

}

TEST_CASE("Vector: capacity method", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);
  REQUIRE(vec.capacity() == 0);

  vec.reserve(10);
  REQUIRE(vec.capacity() >= 10);

  size_t cap = vec.capacity();
  vec.push_back(1);
  REQUIRE(vec.capacity() == cap); // No reallocation

}

TEST_CASE("Vector: empty check", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);
  REQUIRE(vec.empty());

  vec.push_back(10);
  REQUIRE_FALSE(vec.empty());

  vec.clear();
  REQUIRE(vec.empty());

}

TEST_CASE("Vector: reserve growth", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  vec.reserve(100);
  REQUIRE(vec.capacity() >= 100);
  REQUIRE(vec.size() == 0);

  // Add elements without reallocation
  for (int i = 0; i < 100; ++i) {
    vec.push_back(i);
  }
  REQUIRE(vec.size() == 100);

}

TEST_CASE("Vector: reserve no shrink", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  vec.reserve(100);
  size_t cap = vec.capacity();

  vec.reserve(50);
  REQUIRE(vec.capacity() == cap); // No shrinkage

}

TEST_CASE("Vector: shrink_to_fit", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  vec.reserve(100);
  vec.push_back(10);
  vec.push_back(20);
  vec.push_back(30);

  REQUIRE(vec.capacity() >= 100);
  REQUIRE(vec.size() == 3);

  vec.shrink_to_fit();
  REQUIRE(vec.capacity() == 3);
  REQUIRE(vec.size() == 3);

  // Data still intact
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);

}

TEST_CASE("Vector: resize grow", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  vec.resize(6);

  REQUIRE(vec.size() == 6);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);
  REQUIRE(vec[3] == 0); // Default initialized
  REQUIRE(vec[4] == 0);
  REQUIRE(vec[5] == 0);

}

TEST_CASE("Vector: resize with value", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  vec.resize(6, 99);

  REQUIRE(vec.size() == 6);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);
  REQUIRE(vec[3] == 99);
  REQUIRE(vec[4] == 99);
  REQUIRE(vec[5] == 99);

}

TEST_CASE("Vector: resize shrink", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30, 40, 50});

  vec.resize(3);

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);

}

// ============================================================================
// Assignment Operator Tests
// ============================================================================

TEST_CASE("Vector: copy assignment", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec1(alloc, {10, 20, 30});
  vector<int, ArenaAllocator<false>> vec2(alloc);

  vec2 = vec1;

  REQUIRE(vec2.size() == 3);
  REQUIRE(vec2[0] == 10);
  REQUIRE(vec2[1] == 20);
  REQUIRE(vec2[2] == 30);

  // Verify deep copy
  vec2[0] = 99;
  REQUIRE(vec1[0] == 10);

}

TEST_CASE("Vector: copy assignment self", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  auto &self = vec;
  vec = self; // Self-assignment

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);

}

TEST_CASE("Vector: move assignment", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec1(alloc, {10, 20, 30});
  vector<int, ArenaAllocator<false>> vec2(alloc);

  size_t original_capacity = vec1.capacity();

  vec2 = std::move(vec1);

  REQUIRE(vec2.size() == 3);
  REQUIRE(vec2.capacity() == original_capacity);
  REQUIRE(vec2[0] == 10);
  REQUIRE(vec2[1] == 20);
  REQUIRE(vec2[2] == 30);

  // Source is cleared
  REQUIRE(vec1.size() == 0);
  REQUIRE(vec1.capacity() == 0);

}

TEST_CASE("Vector: move assignment self", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30});

  vec = std::move(vec); // Self-move

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 20);
  REQUIRE(vec[2] == 30);

}

// ============================================================================
// Comparison Operator Tests
// ============================================================================

TEST_CASE("Vector: equality same content", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec1(alloc, {10, 20, 30});
  vector<int, ArenaAllocator<false>> vec2(alloc, {10, 20, 30});

  REQUIRE(vec1 == vec2);

}

TEST_CASE("Vector: equality different size", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec1(alloc, {10, 20, 30});
  vector<int, ArenaAllocator<false>> vec2(alloc, {10, 20});

  REQUIRE_FALSE(vec1 == vec2);

}

TEST_CASE("Vector: equality different content", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec1(alloc, {10, 20, 30});
  vector<int, ArenaAllocator<false>> vec2(alloc, {10, 99, 30});

  REQUIRE_FALSE(vec1 == vec2);

}

TEST_CASE("Vector: inequality", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec1(alloc, {10, 20, 30});
  vector<int, ArenaAllocator<false>> vec2(alloc, {10, 99, 30});

  REQUIRE(vec1 != vec2);

}

TEST_CASE("Vector: empty vector equality", "[vector]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec1(alloc);
  vector<int, ArenaAllocator<false>> vec2(alloc);

  REQUIRE(vec1 == vec2);
  REQUIRE_FALSE(vec1 != vec2);

}

// ============================================================================
// POD Type Tests
// ============================================================================

TEST_CASE("Vector: pod push_back", "[vector][pod]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<double, ArenaAllocator<false>> vec(alloc);

  vec.push_back(1.5);
  vec.push_back(2.5);
  vec.push_back(3.5);

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 1.5);
  REQUIRE(vec[1] == 2.5);
  REQUIRE(vec[2] == 3.5);

}

// Pod copy constructor test removed - copy constructor is intentionally deleted
// for IPC data structures since they must be allocated via allocator,
// not copied on the stack

TEST_CASE("Vector: pod insert", "[vector][pod]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<float, ArenaAllocator<false>> vec(alloc, {1.0f, 3.0f});
  vec.insert(vec.begin() + 1, 2.0f);

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 1.0f);
  REQUIRE(vec[1] == 2.0f);
  REQUIRE(vec[2] == 3.0f);

}

TEST_CASE("Vector: pod erase", "[vector][pod]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc, {10, 20, 30, 40});
  vec.erase(vec.begin() + 1);

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 10);
  REQUIRE(vec[1] == 30);
  REQUIRE(vec[2] == 40);

}

TEST_CASE("Vector: pod large vector", "[vector][pod]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 10 * 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  const int NUM_ELEMENTS = 10000;
  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    vec.push_back(i);
  }

  REQUIRE(vec.size() == NUM_ELEMENTS);
  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    REQUIRE(vec[i] == i);
  }

}

// ============================================================================
// Non-POD Type Tests
// ============================================================================

struct NonPodType {
  int value;
  std::string str;

  NonPodType() : value(0), str("") {}
  NonPodType(int v, const std::string& s) : value(v), str(s) {}

  bool operator==(const NonPodType& other) const {
    return value == other.value && str == other.str;
  }
};

TEST_CASE("Vector: non-pod construction", "[vector][non-pod]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<NonPodType, ArenaAllocator<false>> vec(alloc);

  vec.emplace_back(1, "first");
  vec.emplace_back(2, "second");

  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0].value == 1);
  REQUIRE(vec[0].str == "first");
  REQUIRE(vec[1].value == 2);
  REQUIRE(vec[1].str == "second");

}

// Non-pod copy and move tests removed - copy/move constructors are
// intentionally deleted for IPC data structures since they must be
// allocated via allocator, not copied/moved on the stack

// ============================================================================
// Edge Cases & Stress Tests
// ============================================================================

TEST_CASE("Vector: empty vector operations", "[vector][edge]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  REQUIRE(vec.size() == 0);
  REQUIRE(vec.empty());
  REQUIRE(vec.data() == nullptr);
  REQUIRE(vec.begin() == vec.end());

  // Clear on empty vector
  vec.clear();
  REQUIRE(vec.size() == 0);

}

TEST_CASE("Vector: single element", "[vector][edge]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);
  vec.push_back(42);

  REQUIRE(vec.size() == 1);
  REQUIRE(vec.front() == 42);
  REQUIRE(vec.back() == 42);
  REQUIRE(vec[0] == 42);

  // Iterate single element
  int count = 0;
  for (int val : vec) {
    REQUIRE(val == 42);
    count++;
  }
  REQUIRE(count == 1);

}

TEST_CASE("Vector: capacity doubling", "[vector][edge]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  size_t prev_capacity = 0;
  for (int i = 0; i < 100; ++i) {
    vec.push_back(i);
    if (vec.capacity() > prev_capacity) {
      // Verify at least 2x growth (or initial allocation)
      if (prev_capacity > 0) {
        REQUIRE(vec.capacity() >= prev_capacity * 2);
      }
      prev_capacity = vec.capacity();
    }
  }

}

TEST_CASE("Vector: repeated reserve", "[vector][edge]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  vec.reserve(10);
  size_t cap1 = vec.capacity();

  vec.reserve(20);
  size_t cap2 = vec.capacity();
  REQUIRE(cap2 >= 20);

  vec.reserve(15); // Should not shrink
  REQUIRE(vec.capacity() == cap2);

}

TEST_CASE("Vector: all operations sequence", "[vector][stress]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  vector<int, ArenaAllocator<false>> vec(alloc);

  // Build up
  for (int i = 0; i < 10; ++i) {
    vec.push_back(i);
  }
  REQUIRE(vec.size() == 10);

  // Insert in middle
  vec.insert(vec.begin() + 5, 99);
  REQUIRE(vec.size() == 11);
  REQUIRE(vec[5] == 99);

  // Erase from middle
  vec.erase(vec.begin() + 3);
  REQUIRE(vec.size() == 10);

  // Resize
  vec.resize(15, 42);
  REQUIRE(vec.size() == 15);
  REQUIRE(vec[14] == 42);

  // Shrink
  vec.resize(8);
  REQUIRE(vec.size() == 8);

  // Reserve
  vec.reserve(100);
  REQUIRE(vec.capacity() >= 100);

  // Shrink to fit
  vec.shrink_to_fit();
  REQUIRE(vec.capacity() == 8);

  // Clear
  vec.clear();
  REQUIRE(vec.size() == 0);
  REQUIRE(vec.empty());

}

SIMPLE_TEST_MAIN()
