/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "../../../context-runtime/test/simple_test.h"
#include "hermes_shm/data_structures/priv/vector.h"
#include <string>
#include <vector>
#include <memory>

using namespace hshm::priv;

// ============================================================================
// Helper: Simple wrapper allocator for private-memory vectors
// Implements the allocator API required by the refactored vector
// ============================================================================

/**
 * Simple heap-based allocator wrapper for testing private memory vectors.
 * Wraps malloc/free with the library's FullPtr interface.
 */
class SimpleHeapAllocator {
 public:
  /**
   * Allocate memory for count objects of type T.
   * Returns a FullPtr with only the private pointer set (no shared memory).
   *
   * @tparam T The object type
   * @param count Number of objects to allocate space for
   * @return FullPtr to allocated memory
   */
  template <typename T>
  hipc::FullPtr<T> AllocateObjs(size_t count) {
    size_t size = count * sizeof(T);
    T* ptr = static_cast<T*>(malloc(size));
    // Create a FullPtr with only the private pointer (no shared backing)
    hipc::FullPtr<T> result;
    result.ptr_ = ptr;
    result.shm_.off_ = 0;
    result.shm_.alloc_id_ = hipc::AllocatorId::GetNull();
    return result;
  }

  /**
   * Allocate memory of specified byte size.
   * Returns a FullPtr with only the private pointer set (no shared memory).
   *
   * @param size Number of bytes to allocate
   * @return FullPtr to allocated memory
   */
  template <typename T = char>
  hipc::FullPtr<T> Allocate(size_t size) {
    T* ptr = static_cast<T*>(malloc(size));
    hipc::FullPtr<T> result;
    result.ptr_ = ptr;
    result.shm_.off_ = 0;
    result.shm_.alloc_id_ = hipc::AllocatorId::GetNull();
    return result;
  }

  /**
   * Free memory pointed to by a FullPtr.
   * Only frees the private pointer (no shared memory backend).
   *
   * @tparam T The object type
   * @param ptr FullPtr to memory to free
   */
  template <typename T, bool ATOMIC = false>
  void Free(const hipc::FullPtr<T, ATOMIC>& ptr) {
    if (ptr.ptr_ != nullptr) {
      free(ptr.ptr_);
    }
  }
};

// ============================================================================
// Global allocator instances for all tests
// ============================================================================
static SimpleHeapAllocator g_allocator;

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_CASE("Vector: constructor default", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  REQUIRE(vec.size() == 0);
  REQUIRE(vec.capacity() == 0);
  REQUIRE(vec.empty());
  REQUIRE(vec.data() == nullptr);
}

TEST_CASE("Vector: constructor with count and value", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec(5, 42, &g_allocator);

  REQUIRE(vec.size() == 5);
  REQUIRE(vec.capacity() >= 5);
  REQUIRE(!vec.empty());
  REQUIRE(vec[0] == 42);
  REQUIRE(vec[4] == 42);
}

TEST_CASE("Vector: constructor with count", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec(5, &g_allocator);

  REQUIRE(vec.size() == 5);
  REQUIRE(vec.capacity() >= 5);
  REQUIRE(!vec.empty());
  for (size_t i = 0; i < 5; ++i) {
    REQUIRE(vec[i] == 0);
  }
}

TEST_CASE("Vector: constructor with initializer list", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  REQUIRE(vec.size() == 5);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[4] == 5);
}

TEST_CASE("Vector: copy constructor", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec1({1, 2, 3, 4, 5}, &g_allocator);
  vector<int, SimpleHeapAllocator> vec2(vec1);

  REQUIRE(vec2.size() == 5);
  REQUIRE(vec2[0] == 1);
  REQUIRE(vec2[4] == 5);

  vec2[0] = 99;
  REQUIRE(vec1[0] == 1);
}

TEST_CASE("Vector: move constructor", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec1({1, 2, 3, 4, 5}, &g_allocator);
  vector<int, SimpleHeapAllocator> vec2(std::move(vec1));

  REQUIRE(vec2.size() == 5);
  REQUIRE(vec2[0] == 1);
  REQUIRE(vec2[4] == 5);
  REQUIRE(vec1.size() == 0);
  REQUIRE(vec1.empty());
}

// ============================================================================
// Assignment Operator Tests
// ============================================================================

TEST_CASE("Vector: copy assignment", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec1({1, 2, 3}, &g_allocator);
  vector<int, SimpleHeapAllocator> vec2({4, 5, 6, 7}, &g_allocator);

  vec2 = vec1;

  REQUIRE(vec2.size() == 3);
  REQUIRE(vec2[0] == 1);
  REQUIRE(vec2[2] == 3);

  vec2[0] = 99;
  REQUIRE(vec1[0] == 1);
}

TEST_CASE("Vector: move assignment", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec1({1, 2, 3}, &g_allocator);
  vector<int, SimpleHeapAllocator> vec2({4, 5, 6, 7}, &g_allocator);

  vec2 = std::move(vec1);

  REQUIRE(vec2.size() == 3);
  REQUIRE(vec2[0] == 1);
  REQUIRE(vec1.size() == 0);
}

TEST_CASE("Vector: initializer list assignment", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3}, &g_allocator);

  vec = {4, 5, 6, 7, 8};

  REQUIRE(vec.size() == 5);
  REQUIRE(vec[0] == 4);
  REQUIRE(vec[4] == 8);
}

// ============================================================================
// Element Access Tests
// ============================================================================

TEST_CASE("Vector: element access via operator[]", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  REQUIRE(vec[0] == 1);
  REQUIRE(vec[2] == 3);
  REQUIRE(vec[4] == 5);

  vec[2] = 99;
  REQUIRE(vec[2] == 99);
}

TEST_CASE("Vector: element access via at()", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  REQUIRE(vec.at(0) == 1);
  REQUIRE(vec.at(2) == 3);
  REQUIRE(vec.at(4) == 5);

  vec.at(2) = 99;
  REQUIRE(vec.at(2) == 99);
}

TEST_CASE("Vector: at() out of bounds", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3}, &g_allocator);

  bool caught = false;
  try {
    vec.at(5);
  } catch (const std::out_of_range&) {
    caught = true;
  }
  REQUIRE(caught);
}

TEST_CASE("Vector: front() and back()", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  REQUIRE(vec.front() == 1);
  REQUIRE(vec.back() == 5);

  vec.front() = 99;
  vec.back() = 88;

  REQUIRE(vec[0] == 99);
  REQUIRE(vec[4] == 88);
}

TEST_CASE("Vector: data() pointer", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  int *ptr = vec.data();

  REQUIRE(ptr != nullptr);
  REQUIRE(ptr[0] == 1);
  REQUIRE(ptr[4] == 5);
}

// ============================================================================
// Push/Pop Tests
// ============================================================================

TEST_CASE("Vector: push_back single element", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  vec.push_back(42);

  REQUIRE(vec.size() == 1);
  REQUIRE(vec[0] == 42);
}

TEST_CASE("Vector: push_back multiple elements", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  for (int i = 0; i < 10; ++i) {
    vec.push_back(i * 10);
  }

  REQUIRE(vec.size() == 10);
  for (int i = 0; i < 10; ++i) {
    REQUIRE(vec[i] == i * 10);
  }
}

TEST_CASE("Vector: push_back with capacity growth", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  REQUIRE(vec.capacity() == 0);

  vec.push_back(1);
  REQUIRE(vec.capacity() >= 1);

  vec.push_back(2);
  size_t cap_after_2 = vec.capacity();
  REQUIRE(cap_after_2 >= 2);

  vec.push_back(3);
  REQUIRE(vec.capacity() >= cap_after_2);

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 2);
  REQUIRE(vec[2] == 3);
}

TEST_CASE("Vector: pop_back", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  vec.pop_back();
  REQUIRE(vec.size() == 4);
  REQUIRE(vec.back() == 4);

  vec.pop_back();
  REQUIRE(vec.size() == 3);
  REQUIRE(vec.back() == 3);
}

TEST_CASE("Vector: push_back with move semantics", "[priv_vector]") {
  vector<std::string, SimpleHeapAllocator> vec(&g_allocator);

  std::string str = "hello";
  vec.push_back(std::move(str));

  REQUIRE(vec.size() == 1);
  REQUIRE(vec[0] == "hello");
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST_CASE("Vector: iterator traversal", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  int sum = 0;
  for (auto it = vec.begin(); it != vec.end(); ++it) {
    sum += *it;
  }

  REQUIRE(sum == 15);
}

TEST_CASE("Vector: const iterator", "[priv_vector]") {
  const vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  int sum = 0;
  for (auto it = vec.begin(); it != vec.end(); ++it) {
    sum += *it;
  }

  REQUIRE(sum == 15);
}

TEST_CASE("Vector: reverse iterator", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  std::vector<int> result;
  for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
    result.push_back(*it);
  }

  REQUIRE(result.size() == 5);
  REQUIRE(result[0] == 5);
  REQUIRE(result[4] == 1);
}

TEST_CASE("Vector: iterator arithmetic", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  auto it = vec.begin();
  it += 2;

  REQUIRE(*it == 3);

  auto it2 = it + 1;
  REQUIRE(*it2 == 4);

  auto diff = it2 - it;
  REQUIRE(diff == 1);
}

TEST_CASE("Vector: iterator subscript", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  auto it = vec.begin();
  REQUIRE(it[0] == 1);
  REQUIRE(it[2] == 3);
  REQUIRE(it[4] == 5);
}

// ============================================================================
// Insert/Erase Tests
// ============================================================================

TEST_CASE("Vector: insert single element", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 4, 5}, &g_allocator);

  vec.insert(vec.cbegin() + 2, 3);

  REQUIRE(vec.size() == 5);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[2] == 3);
  REQUIRE(vec[3] == 4);
  REQUIRE(vec[4] == 5);
}

TEST_CASE("Vector: insert at beginning", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({2, 3, 4}, &g_allocator);

  vec.insert(vec.cbegin(), 1);

  REQUIRE(vec.size() == 4);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 2);
}

TEST_CASE("Vector: insert at end", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3}, &g_allocator);

  vec.insert(vec.cend(), 4);

  REQUIRE(vec.size() == 4);
  REQUIRE(vec[3] == 4);
}

TEST_CASE("Vector: insert range", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 5}, &g_allocator);
  vector<int, SimpleHeapAllocator> vals({2, 3, 4}, &g_allocator);

  vec.insert(vec.cbegin() + 1, vals.cbegin(), vals.cend());

  REQUIRE(vec.size() == 5);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 2);
  REQUIRE(vec[2] == 3);
  REQUIRE(vec[3] == 4);
  REQUIRE(vec[4] == 5);
}

TEST_CASE("Vector: erase single element", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  vec.erase(vec.cbegin() + 2);

  REQUIRE(vec.size() == 4);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 2);
  REQUIRE(vec[2] == 4);
  REQUIRE(vec[3] == 5);
}

TEST_CASE("Vector: erase first element", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3}, &g_allocator);

  vec.erase(vec.cbegin());

  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == 2);
  REQUIRE(vec[1] == 3);
}

TEST_CASE("Vector: erase last element", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3}, &g_allocator);

  vec.erase(vec.cbegin() + 2);

  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 2);
}

TEST_CASE("Vector: erase range", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  vec.erase(vec.cbegin() + 1, vec.cbegin() + 4);

  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 5);
}

// ============================================================================
// Clear/Resize Tests
// ============================================================================

TEST_CASE("Vector: clear", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  REQUIRE(vec.size() == 5);

  vec.clear();

  REQUIRE(vec.size() == 0);
  REQUIRE(vec.empty());
  REQUIRE(vec.capacity() > 0);
}

TEST_CASE("Vector: resize grow", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3}, &g_allocator);

  vec.resize(6);

  REQUIRE(vec.size() == 6);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 2);
  REQUIRE(vec[2] == 3);
  REQUIRE(vec[3] == 0);
  REQUIRE(vec[4] == 0);
  REQUIRE(vec[5] == 0);
}

TEST_CASE("Vector: resize grow with value", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3}, &g_allocator);

  vec.resize(6, 99);

  REQUIRE(vec.size() == 6);
  REQUIRE(vec[3] == 99);
  REQUIRE(vec[4] == 99);
  REQUIRE(vec[5] == 99);
}

TEST_CASE("Vector: resize shrink", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);

  vec.resize(2);

  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[1] == 2);
}

TEST_CASE("Vector: reserve", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  vec.reserve(100);

  REQUIRE(vec.capacity() >= 100);
  REQUIRE(vec.size() == 0);
}

TEST_CASE("Vector: shrink_to_fit", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);
  vec.reserve(100);

  REQUIRE(vec.capacity() >= 100);

  vec.shrink_to_fit();

  REQUIRE(vec.capacity() == 0);
  REQUIRE(vec.empty());
}

TEST_CASE("Vector: shrink_to_fit with elements", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3, 4, 5}, &g_allocator);
  vec.reserve(100);

  REQUIRE(vec.capacity() >= 100);

  vec.shrink_to_fit();

  REQUIRE(vec.capacity() >= 5);
  REQUIRE(vec.size() == 5);
  REQUIRE(vec[0] == 1);
  REQUIRE(vec[4] == 5);
}

// ============================================================================
// POD Type Optimization Tests
// ============================================================================

TEST_CASE("Vector: POD type handling", "[priv_vector][pod]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  for (int i = 0; i < 1000; ++i) {
    vec.push_back(i);
  }

  REQUIRE(vec.size() == 1000);
  REQUIRE(vec[0] == 0);
  REQUIRE(vec[999] == 999);
}

TEST_CASE("Vector: POD type resize", "[priv_vector][pod]") {
  vector<double, SimpleHeapAllocator> vec(&g_allocator);
  vec.reserve(100);

  for (int i = 0; i < 100; ++i) {
    vec.push_back(i * 1.5);
  }

  vec.shrink_to_fit();

  REQUIRE(vec.size() == 100);
  REQUIRE(vec[50] == 50 * 1.5);
}

// ============================================================================
// Complex Type Tests
// ============================================================================

TEST_CASE("Vector: string elements", "[priv_vector][complex]") {
  vector<std::string, SimpleHeapAllocator> vec(&g_allocator);

  vec.push_back("hello");
  vec.push_back("world");
  vec.push_back("test");

  REQUIRE(vec.size() == 3);
  REQUIRE(vec[0] == "hello");
  REQUIRE(vec[1] == "world");
  REQUIRE(vec[2] == "test");
}

TEST_CASE("Vector: string move", "[priv_vector][complex]") {
  vector<std::string, SimpleHeapAllocator> vec(&g_allocator);

  std::string s1 = "hello";
  vec.push_back(std::move(s1));

  REQUIRE(vec.size() == 1);
  REQUIRE(vec[0] == "hello");
}

TEST_CASE("Vector: complex type copy", "[priv_vector][complex]") {
  vector<std::string, SimpleHeapAllocator> vec1(&g_allocator);
  vec1.push_back("test");

  vector<std::string, SimpleHeapAllocator> vec2 = vec1;

  REQUIRE(vec2.size() == 1);
  REQUIRE(vec2[0] == "test");

  vec2[0] = "modified";
  REQUIRE(vec1[0] == "test");
}

// ============================================================================
// Swap Tests
// ============================================================================

TEST_CASE("Vector: swap", "[priv_vector]") {
  vector<int, SimpleHeapAllocator> vec1({1, 2, 3}, &g_allocator);
  vector<int, SimpleHeapAllocator> vec2({4, 5}, &g_allocator);

  vec1.swap(vec2);

  REQUIRE(vec1.size() == 2);
  REQUIRE(vec1[0] == 4);
  REQUIRE(vec1[1] == 5);

  REQUIRE(vec2.size() == 3);
  REQUIRE(vec2[0] == 1);
  REQUIRE(vec2[2] == 3);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("Vector: empty vector pop_back does nothing", "[priv_vector][edge]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  vec.pop_back();
  REQUIRE(vec.empty());
}

TEST_CASE("Vector: clear empty vector", "[priv_vector][edge]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  vec.clear();
  REQUIRE(vec.empty());
}

TEST_CASE("Vector: reserve same capacity", "[priv_vector][edge]") {
  vector<int, SimpleHeapAllocator> vec({1, 2, 3}, &g_allocator);
  size_t original_capacity = vec.capacity();

  vec.reserve(original_capacity);

  REQUIRE(vec.capacity() == original_capacity);
  REQUIRE(vec.size() == 3);
}

TEST_CASE("Vector: large capacity growth", "[priv_vector][stress]") {
  vector<int, SimpleHeapAllocator> vec(&g_allocator);

  for (int i = 0; i < 10000; ++i) {
    vec.push_back(i);
  }

  REQUIRE(vec.size() == 10000);
  REQUIRE(vec[0] == 0);
  REQUIRE(vec[9999] == 9999);
}

// ============================================================================
// Serialization Tests (Cereal)
// ============================================================================

#ifdef HSHM_ENABLE_CEREAL
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <sstream>

// Define a simple struct with cereal serialization for testing
struct Point {
  int x;
  int y;

  template<class Archive>
  void serialize(Archive& ar) {
    ar(x, y);
  }

  bool operator==(const Point& other) const {
    return x == other.x && y == other.y;
  }
};

TEST_CASE("Vector: serialize POD type (int)", "[priv_vector][serialization]") {
  vector<int, SimpleHeapAllocator> original({1, 2, 3, 4, 5}, &g_allocator);

  // Serialize
  std::ostringstream os(std::ios::binary);
  {
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(original);
  }

  // Deserialize
  vector<int, SimpleHeapAllocator> restored(&g_allocator);
  std::istringstream is(os.str(), std::ios::binary);
  {
    cereal::BinaryInputArchive iarchive(is);
    iarchive(restored);
  }

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(restored.size() == 5);
  for (size_t i = 0; i < 5; ++i) {
    REQUIRE(restored[i] == original[i]);
  }
}

TEST_CASE("Vector: serialize POD type (double)", "[priv_vector][serialization]") {
  vector<double, SimpleHeapAllocator> original(&g_allocator);
  for (int i = 0; i < 10; ++i) {
    original.push_back(i * 1.5);
  }

  // Serialize
  std::ostringstream os(std::ios::binary);
  {
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(original);
  }

  // Deserialize
  vector<double, SimpleHeapAllocator> restored(&g_allocator);
  std::istringstream is(os.str(), std::ios::binary);
  {
    cereal::BinaryInputArchive iarchive(is);
    iarchive(restored);
  }

  // Verify
  REQUIRE(restored.size() == original.size());
  for (size_t i = 0; i < restored.size(); ++i) {
    REQUIRE(restored[i] == original[i]);
  }
}

TEST_CASE("Vector: serialize empty vector", "[priv_vector][serialization]") {
  vector<int, SimpleHeapAllocator> original(&g_allocator);

  // Serialize
  std::ostringstream os(std::ios::binary);
  {
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(original);
  }

  // Deserialize
  vector<int, SimpleHeapAllocator> restored(&g_allocator);
  std::istringstream is(os.str(), std::ios::binary);
  {
    cereal::BinaryInputArchive iarchive(is);
    iarchive(restored);
  }

  // Verify
  REQUIRE(restored.size() == 0);
  REQUIRE(restored.empty());
}

TEST_CASE("Vector: serialize complex type (std::string)", "[priv_vector][serialization]") {
  vector<std::string, SimpleHeapAllocator> original(&g_allocator);
  original.push_back("hello");
  original.push_back("world");
  original.push_back("test");

  // Serialize
  std::ostringstream os(std::ios::binary);
  {
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(original);
  }

  // Deserialize
  vector<std::string, SimpleHeapAllocator> restored(&g_allocator);
  std::istringstream is(os.str(), std::ios::binary);
  {
    cereal::BinaryInputArchive iarchive(is);
    iarchive(restored);
  }

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(restored.size() == 3);
  REQUIRE(restored[0] == "hello");
  REQUIRE(restored[1] == "world");
  REQUIRE(restored[2] == "test");
}

TEST_CASE("Vector: serialize large vector", "[priv_vector][serialization]") {
  vector<int, SimpleHeapAllocator> original(&g_allocator);
  for (int i = 0; i < 1000; ++i) {
    original.push_back(i);
  }

  // Serialize
  std::ostringstream os(std::ios::binary);
  {
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(original);
  }

  // Deserialize
  vector<int, SimpleHeapAllocator> restored(&g_allocator);
  std::istringstream is(os.str(), std::ios::binary);
  {
    cereal::BinaryInputArchive iarchive(is);
    iarchive(restored);
  }

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(restored.size() == 1000);
  for (size_t i = 0; i < 1000; ++i) {
    REQUIRE(restored[i] == static_cast<int>(i));
  }
}

TEST_CASE("Vector: serialize single element", "[priv_vector][serialization]") {
  vector<int, SimpleHeapAllocator> original(&g_allocator);
  original.push_back(42);

  // Serialize
  std::ostringstream os(std::ios::binary);
  {
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(original);
  }

  // Deserialize
  vector<int, SimpleHeapAllocator> restored(&g_allocator);
  std::istringstream is(os.str(), std::ios::binary);
  {
    cereal::BinaryInputArchive iarchive(is);
    iarchive(restored);
  }

  // Verify
  REQUIRE(restored.size() == 1);
  REQUIRE(restored[0] == 42);
}

TEST_CASE("Vector: serialize struct type", "[priv_vector][serialization]") {
  vector<Point, SimpleHeapAllocator> original(&g_allocator);
  original.push_back({1, 2});
  original.push_back({3, 4});
  original.push_back({5, 6});

  // Serialize
  std::ostringstream os(std::ios::binary);
  {
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(original);
  }

  // Deserialize
  vector<Point, SimpleHeapAllocator> restored(&g_allocator);
  std::istringstream is(os.str(), std::ios::binary);
  {
    cereal::BinaryInputArchive iarchive(is);
    iarchive(restored);
  }

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(restored.size() == 3);
  for (size_t i = 0; i < 3; ++i) {
    REQUIRE(restored[i] == original[i]);
  }
}

#endif  // HSHM_ENABLE_CEREAL

SIMPLE_TEST_MAIN()
