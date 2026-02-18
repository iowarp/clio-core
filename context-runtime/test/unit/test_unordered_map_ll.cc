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

/**
 * Unit tests for hshm::priv::unordered_map_ll
 *
 * Tests the unordered map implementation without requiring
 * the Chimaera runtime to be started.
 * NOTE: External locking is required for thread safety.
 */

#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <cassert>
#include <hermes_shm/util/logging.h>

#include <hermes_shm/data_structures/priv/unordered_map_ll.h>

// Simple test helper macros
#define EXPECT_EQ(a, b) do { \
  if ((a) != (b)) { \
    HLOG(kError, "FAIL: Expected {} == {} at line {}", (a), (b), __LINE__); \
    return 1; \
  } \
} while(0)

#define EXPECT_NE(a, b) do { \
  if ((a) == (b)) { \
    HLOG(kError, "FAIL: Expected {} != {} at line {}", (a), (b), __LINE__); \
    return 1; \
  } \
} while(0)

#define EXPECT_TRUE(a) do { \
  if (!(a)) { \
    HLOG(kError, "FAIL: Expected true at line {}", __LINE__); \
    return 1; \
  } \
} while(0)

#define EXPECT_FALSE(a) do { \
  if ((a)) { \
    HLOG(kError, "FAIL: Expected false at line {}", __LINE__); \
    return 1; \
  } \
} while(0)

#define EXPECT_GT(a, b) do { \
  if ((a) <= (b)) { \
    HLOG(kError, "FAIL: Expected {} > {} at line {}", (a), (b), __LINE__); \
    return 1; \
  } \
} while(0)

#define EXPECT_THROW(expr, exc_type) do { \
  try { \
    expr; \
    HLOG(kError, "FAIL: Expected exception at line {}", __LINE__); \
    return 1; \
  } catch (const exc_type&) { \
  } \
} while(0)

#define TEST(suite, name) int test_##suite##_##name()
#define TEST_F(fixture, name) int test_##fixture##_##name()

/**
 * Test basic insertion and retrieval
 */
TEST_F(UnorderedMapLLTest, BasicInsertAndFind) {
  hshm::priv::unordered_map_ll<int, std::string> map(16);

  // Insert some elements
  auto [inserted1, val1] = map.insert(1, "one");
  EXPECT_TRUE(inserted1);
  EXPECT_NE(val1, nullptr);
  EXPECT_EQ(*val1, "one");

  auto [inserted2, val2] = map.insert(2, "two");
  EXPECT_TRUE(inserted2);
  EXPECT_NE(val2, nullptr);
  EXPECT_EQ(*val2, "two");

  auto [inserted3, val3] = map.insert(3, "three");
  EXPECT_TRUE(inserted3);
  EXPECT_NE(val3, nullptr);
  EXPECT_EQ(*val3, "three");

  // Check size
  EXPECT_EQ(map.size(), 3);
  EXPECT_FALSE(map.empty());

  // Find elements
  std::string* found1 = map.find(1);
  EXPECT_NE(found1, nullptr);
  EXPECT_EQ(*found1, "one");

  std::string* found2 = map.find(2);
  EXPECT_NE(found2, nullptr);
  EXPECT_EQ(*found2, "two");

  std::string* found3 = map.find(3);
  EXPECT_NE(found3, nullptr);
  EXPECT_EQ(*found3, "three");

  // Find non-existent element
  std::string* not_found = map.find(999);
  EXPECT_EQ(not_found, nullptr);
  return 0;
}

/**
 * Test duplicate insertion
 */
TEST_F(UnorderedMapLLTest, DuplicateInsertion) {
  hshm::priv::unordered_map_ll<int, std::string> map(8);

  // First insertion should succeed
  auto [inserted1, val1] = map.insert(42, "first");
  EXPECT_TRUE(inserted1);
  EXPECT_EQ(*val1, "first");

  // Second insertion with same key should fail
  auto [inserted2, val2] = map.insert(42, "second");
  EXPECT_FALSE(inserted2);
  EXPECT_NE(val2, nullptr);
  EXPECT_EQ(*val2, "first");  // Should still have original value

  // Check size
  EXPECT_EQ(map.size(), 1);
  return 0;
}

/**
 * Test insert_or_assign
 */
TEST_F(UnorderedMapLLTest, InsertOrAssign) {
  hshm::priv::unordered_map_ll<int, std::string> map(8);

  // First insertion
  auto [inserted1, val1] = map.insert_or_assign(10, "original");
  EXPECT_TRUE(inserted1);
  EXPECT_EQ(*val1, "original");

  // Second insertion with same key should update
  auto [inserted2, val2] = map.insert_or_assign(10, "updated");
  EXPECT_FALSE(inserted2);  // Not inserted, updated
  EXPECT_EQ(*val2, "updated");

  // Verify the update
  std::string* found = map.find(10);
  EXPECT_NE(found, nullptr);
  EXPECT_EQ(*found, "updated");

  EXPECT_EQ(map.size(), 1);
  return 0;
}

/**
 * Test operator[]
 */
TEST_F(UnorderedMapLLTest, OperatorBracket) {
  hshm::priv::unordered_map_ll<std::string, int> map(16);

  // Access non-existent key creates default value
  int& val1 = map["key1"];
  EXPECT_EQ(val1, 0);  // Default int is 0

  // Modify value
  val1 = 100;
  EXPECT_EQ(map["key1"], 100);

  // Access again returns modified value
  int& val2 = map["key1"];
  EXPECT_EQ(val2, 100);

  EXPECT_EQ(map.size(), 1);
  return 0;
}

/**
 * Test at() method
 */
TEST_F(UnorderedMapLLTest, AtMethod) {
  hshm::priv::unordered_map_ll<int, std::string> map(8);

  map.insert(5, "five");
  map.insert(10, "ten");

  // Access existing keys
  EXPECT_EQ(map.at(5), "five");
  EXPECT_EQ(map.at(10), "ten");

  // Modify via at()
  map.at(5) = "FIVE";
  EXPECT_EQ(map.at(5), "FIVE");

  // Access non-existent key should throw
  EXPECT_THROW(map.at(999), std::out_of_range);
  return 0;
}

/**
 * Test erase operation
 */
TEST_F(UnorderedMapLLTest, Erase) {
  hshm::priv::unordered_map_ll<int, std::string> map(8);

  map.insert(1, "one");
  map.insert(2, "two");
  map.insert(3, "three");
  EXPECT_EQ(map.size(), 3);

  // Erase existing key
  size_t erased1 = map.erase(2);
  EXPECT_EQ(erased1, 1);
  EXPECT_EQ(map.size(), 2);
  EXPECT_EQ(map.find(2), nullptr);

  // Erase non-existent key
  size_t erased2 = map.erase(999);
  EXPECT_EQ(erased2, 0);
  EXPECT_EQ(map.size(), 2);

  // Verify remaining elements
  EXPECT_NE(map.find(1), nullptr);
  EXPECT_NE(map.find(3), nullptr);
  return 0;
}

/**
 * Test clear operation
 */
TEST_F(UnorderedMapLLTest, Clear) {
  hshm::priv::unordered_map_ll<int, std::string> map(8);

  map.insert(1, "one");
  map.insert(2, "two");
  map.insert(3, "three");
  EXPECT_EQ(map.size(), 3);
  EXPECT_FALSE(map.empty());

  map.clear();
  EXPECT_EQ(map.size(), 0);
  EXPECT_TRUE(map.empty());

  // Verify all elements are gone
  EXPECT_EQ(map.find(1), nullptr);
  EXPECT_EQ(map.find(2), nullptr);
  EXPECT_EQ(map.find(3), nullptr);
  return 0;
}

/**
 * Test contains and count methods
 */
TEST_F(UnorderedMapLLTest, ContainsAndCount) {
  hshm::priv::unordered_map_ll<int, std::string> map(8);

  map.insert(10, "ten");
  map.insert(20, "twenty");

  // Test contains
  EXPECT_TRUE(map.contains(10));
  EXPECT_TRUE(map.contains(20));
  EXPECT_FALSE(map.contains(30));

  // Test count
  EXPECT_EQ(map.count(10), 1);
  EXPECT_EQ(map.count(20), 1);
  EXPECT_EQ(map.count(30), 0);
  return 0;
}

/**
 * Test for_each iteration
 */
TEST_F(UnorderedMapLLTest, ForEach) {
  hshm::priv::unordered_map_ll<int, int> map(8);

  map.insert(1, 10);
  map.insert(2, 20);
  map.insert(3, 30);

  // Sum all values using for_each
  int sum = 0;
  map.for_each([&sum](const int& key, const int& value) {
    sum += value;
  });
  EXPECT_EQ(sum, 60);

  // Modify values using for_each
  map.for_each([](const int& key, int& value) {
    value *= 2;
  });

  // Verify modifications
  EXPECT_EQ(*map.find(1), 20);
  EXPECT_EQ(*map.find(2), 40);
  EXPECT_EQ(*map.find(3), 60);
  return 0;
}

/**
 * Test concurrent insertions from multiple threads
 * NOTE: Uses external mutex for thread safety
 */
TEST_F(UnorderedMapLLTest, ConcurrentInsertions) {
  hshm::priv::unordered_map_ll<int, std::string> map(32);  // More buckets for better concurrency
  const int num_threads = 8;
  const int insertions_per_thread = 100;

  std::vector<std::thread> threads;
  std::atomic<int> successful_insertions{0};
  std::mutex map_mutex;  // External mutex for thread safety

  // Each thread inserts unique keys
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&map, &map_mutex, &successful_insertions, t, insertions_per_thread]() {
      for (int i = 0; i < insertions_per_thread; ++i) {
        int key = t * insertions_per_thread + i;
        std::string value = "thread_" + std::to_string(t) + "_value_" + std::to_string(i);

        std::lock_guard<std::mutex> lock(map_mutex);
        auto [inserted, val] = map.insert(key, value);
        if (inserted) {
          successful_insertions.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify all insertions succeeded
  int expected_total = num_threads * insertions_per_thread;
  EXPECT_EQ(successful_insertions.load(), expected_total);
  EXPECT_EQ(map.size(), static_cast<size_t>(expected_total));
  return 0;
}

/**
 * Test concurrent insertions with collisions (same keys from different threads)
 * NOTE: Uses external mutex for thread safety
 */
TEST_F(UnorderedMapLLTest, ConcurrentInsertionsWithCollisions) {
  hshm::priv::unordered_map_ll<int, int> map(16);
  const int num_threads = 10;
  const int num_keys = 50;

  std::vector<std::thread> threads;
  std::mutex map_mutex;  // External mutex for thread safety

  // All threads try to insert the same set of keys
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&map, &map_mutex, num_keys, t]() {
      for (int key = 0; key < num_keys; ++key) {
        std::lock_guard<std::mutex> lock(map_mutex);
        map.insert(key, t);  // Value is thread ID
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify exactly num_keys elements exist (no duplicates)
  EXPECT_EQ(map.size(), static_cast<size_t>(num_keys));

  // Verify all keys are present
  for (int key = 0; key < num_keys; ++key) {
    EXPECT_TRUE(map.contains(key));
  }
  return 0;
}

/**
 * Test concurrent mixed operations (insert, find, erase)
 * NOTE: Uses external mutex for thread safety
 */
TEST_F(UnorderedMapLLTest, ConcurrentMixedOperations) {
  hshm::priv::unordered_map_ll<int, int> map(32);
  const int num_threads = 6;
  const int operations_per_thread = 100;

  // Pre-populate with some data
  for (int i = 0; i < 200; ++i) {
    map.insert(i, i * 10);
  }

  std::vector<std::thread> threads;
  std::mutex map_mutex;  // External mutex for thread safety

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&map, &map_mutex, t, operations_per_thread]() {
      for (int i = 0; i < operations_per_thread; ++i) {
        int key = (t * operations_per_thread + i) % 200;

        std::lock_guard<std::mutex> lock(map_mutex);

        // Mix of operations
        if (i % 3 == 0) {
          // Insert
          map.insert_or_assign(key + 1000, i);
        } else if (i % 3 == 1) {
          // Find
          int* val = map.find(key);
          (void)val;  // Suppress unused warning
        } else {
          // Erase
          map.erase(key);
        }
      }
    });
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Just verify the map is still in a valid state (no crashes)
  (void)map.size();  // Map should still be in valid state
  return 0;
}

/**
 * Test bucket distribution
 */
TEST_F(UnorderedMapLLTest, BucketDistribution) {
  const size_t num_buckets = 16;
  hshm::priv::unordered_map_ll<int, int> map(num_buckets);

  EXPECT_EQ(map.bucket_count(), num_buckets);

  // Insert many elements
  for (int i = 0; i < 1000; ++i) {
    map.insert(i, i * 2);
  }

  EXPECT_EQ(map.size(), 1000);
  return 0;
}

// Main function to run all tests
int main() {
  int failed = 0;
  int total = 0;

  #define RUN_TEST(suite, name) do { \
    total++; \
    HIPRINT("Running " #suite "." #name "..."); \
    if (test_##suite##_##name() != 0) { \
      HLOG(kError, "FAILED: " #suite "." #name); \
      failed++; \
    } else { \
      HLOG(kInfo, "PASSED: " #suite "." #name); \
    } \
  } while(0)

  RUN_TEST(UnorderedMapLLTest, BasicInsertAndFind);
  RUN_TEST(UnorderedMapLLTest, DuplicateInsertion);
  RUN_TEST(UnorderedMapLLTest, InsertOrAssign);
  RUN_TEST(UnorderedMapLLTest, OperatorBracket);
  RUN_TEST(UnorderedMapLLTest, AtMethod);
  RUN_TEST(UnorderedMapLLTest, Erase);
  RUN_TEST(UnorderedMapLLTest, Clear);
  RUN_TEST(UnorderedMapLLTest, ContainsAndCount);
  RUN_TEST(UnorderedMapLLTest, ForEach);
  RUN_TEST(UnorderedMapLLTest, ConcurrentInsertions);
  RUN_TEST(UnorderedMapLLTest, ConcurrentInsertionsWithCollisions);
  RUN_TEST(UnorderedMapLLTest, ConcurrentMixedOperations);
  RUN_TEST(UnorderedMapLLTest, BucketDistribution);

  HIPRINT("{}/{} tests passed", (total - failed), total);
  return failed > 0 ? 1 : 0;
}
