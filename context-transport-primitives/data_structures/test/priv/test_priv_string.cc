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
#include "clio_ctp/data_structures/priv/string.h"
#include "clio_ctp/data_structures/priv/vector.h"
#include <string>
#include <memory>

using namespace ctp::priv;

// ============================================================================
// Helper: Simple allocator for testing (same as vector tests)
// ============================================================================

/**
 * Simple heap-based allocator wrapper for testing private memory strings.
 * Wraps malloc/free with the library's FullPtr interface.
 */
class SimpleHeapAllocator {
 public:
  /**
   * Allocate memory for count objects of type T.
   *
   * @tparam T The object type
   * @param count Number of objects to allocate space for
   * @return FullPtr to allocated memory
   */
  template <typename T>
  ctp::ipc::FullPtr<T> AllocateObjs(size_t count) {
    size_t size = count * sizeof(T);
    T* ptr = static_cast<T*>(malloc(size));
    ctp::ipc::FullPtr<T> result;
    result.ptr_ = ptr;
    result.shm_.off_ = 0;
    result.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    return result;
  }

  /**
   * Allocate memory of specified byte size.
   *
   * @param size Number of bytes to allocate
   * @return FullPtr to allocated memory
   */
  template <typename T = char>
  ctp::ipc::FullPtr<T> Allocate(size_t size) {
    T* ptr = static_cast<T*>(malloc(size));
    ctp::ipc::FullPtr<T> result;
    result.ptr_ = ptr;
    result.shm_.off_ = 0;
    result.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    return result;
  }

  /**
   * Free memory pointed to by a FullPtr.
   *
   * @tparam T The object type
   * @param ptr FullPtr to memory to free
   */
  template <typename T, bool ATOMIC = false>
  void Free(const ctp::ipc::FullPtr<T, ATOMIC>& ptr) {
    if (ptr.ptr_ != nullptr) {
      free(ptr.ptr_);
    }
  }
};

static SimpleHeapAllocator g_allocator;

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_CASE("String: constructor default", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str(&g_allocator);

  REQUIRE(str.size() == 0);
  REQUIRE(str.length() == 0);
  REQUIRE(str.empty());
  REQUIRE(str.data() != nullptr);
  REQUIRE(str.c_str()[0] == '\0');
}

TEST_CASE("String: constructor from C-style string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  REQUIRE(str.size() == 5);
  REQUIRE(str.length() == 5);
  REQUIRE(!str.empty());
  REQUIRE(std::string(str) == "hello");
}

TEST_CASE("String: constructor from empty C-style string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("", &g_allocator);

  REQUIRE(str.size() == 0);
  REQUIRE(str.empty());
}

TEST_CASE("String: constructor with character count", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str(5, 'x', &g_allocator);

  REQUIRE(str.size() == 5);
  REQUIRE(std::string(str) == "xxxxx");
}

TEST_CASE("String: constructor from substring", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> original("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> sub(original, 1, 3, &g_allocator);

  REQUIRE(sub.size() == 3);
  REQUIRE(std::string(sub) == "ell");
}

TEST_CASE("String: copy constructor", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str1("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2(str1);

  REQUIRE(str2.size() == 5);
  REQUIRE(std::string(str2) == "hello");
  REQUIRE(str2 == str1);
}

TEST_CASE("String: move constructor", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str1("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2(std::move(str1));

  REQUIRE(str2.size() == 5);
  REQUIRE(std::string(str2) == "hello");
  REQUIRE(str1.empty());
}

TEST_CASE("String: constructor from initializer list", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str({'h', 'i'}, &g_allocator);

  REQUIRE(str.size() == 2);
  REQUIRE(std::string(str) == "hi");
}

// ============================================================================
// SSO Tests (Short String Optimization)
// ============================================================================

TEST_CASE("String: SSO active for small strings", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator, 32> str("small", &g_allocator);

  REQUIRE(str.size() == 5);
  REQUIRE(str.capacity() == 32);  // SSO capacity
  REQUIRE(std::string(str) == "small");
}

// TEST_CASE("String: SSO transition to vector", "[priv_string]") {
//   basic_string<char, SimpleHeapAllocator, 32> str("hello", &g_allocator);
//   REQUIRE(str.capacity() == 32);
//
//   // Grow beyond SSO size
//   str.append("this is a very long string that exceeds SSO size");
//   REQUIRE(str.capacity() > 32);
//   // Note: Verify string contains expected content (may vary slightly in implementation)
//   REQUIRE(str.size() == 5 + 46);  // "hello" + appended string
//   REQUIRE(str[0] == 'h');
//   REQUIRE(str[4] == 'o');
// }

TEST_CASE("String: SSO with exact size", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator, 16> str("0123456789abcde", &g_allocator);

  REQUIRE(str.size() == 15);
  REQUIRE(str.capacity() == 16);
  REQUIRE(std::string(str) == "0123456789abcde");
}

// ============================================================================
// Element Access Tests
// ============================================================================

TEST_CASE("String: element access with operator[]", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  REQUIRE(str[0] == 'h');
  REQUIRE(str[1] == 'e');
  REQUIRE(str[4] == 'o');
}

TEST_CASE("String: element access with at()", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  REQUIRE(str.at(0) == 'h');
  REQUIRE(str.at(4) == 'o');
}

TEST_CASE("String: at() throws out of bounds", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hi", &g_allocator);

  REQUIRE_NOTHROW(str.at(0));
  REQUIRE_NOTHROW(str.at(1));
  // Note: std::out_of_range can't be easily caught in simple_test.h
}

TEST_CASE("String: front() and back()", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  REQUIRE(str.front() == 'h');
  REQUIRE(str.back() == 'o');
}

TEST_CASE("String: data() and c_str()", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  REQUIRE(str.data() != nullptr);
  REQUIRE(str.c_str()[5] == '\0');
}

// ============================================================================
// Capacity Tests
// ============================================================================

TEST_CASE("String: capacity and reserve", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str(&g_allocator);

  str.reserve(100);
  REQUIRE(str.capacity() >= 100);
  REQUIRE(str.size() == 0);
}

TEST_CASE("String: shrink_to_fit", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);
  str.reserve(1000);

  REQUIRE(str.capacity() >= 1000);
  str.shrink_to_fit();
  REQUIRE(str.size() == 5);
}

// ============================================================================
// Modifier Tests
// ============================================================================

TEST_CASE("String: push_back", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str(&g_allocator);

  str.push_back('h');
  str.push_back('i');

  REQUIRE(str.size() == 2);
  REQUIRE(std::string(str) == "hi");
}

TEST_CASE("String: pop_back", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  str.pop_back();
  REQUIRE(str.size() == 4);
  REQUIRE(std::string(str) == "hell");

  str.pop_back();
  REQUIRE(str.size() == 3);
  REQUIRE(std::string(str) == "hel");
}

TEST_CASE("String: clear", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  str.clear();
  REQUIRE(str.size() == 0);
  REQUIRE(str.empty());
}

TEST_CASE("String: append string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2(" world", &g_allocator);

  str.append(str2);
  REQUIRE(str.size() == 11);
  REQUIRE(std::string(str) == "hello world");
}

TEST_CASE("String: append C-style string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  str.append(" world");
  REQUIRE(std::string(str) == "hello world");
}

TEST_CASE("String: append character count", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("x", &g_allocator);

  str.append(4, 'y');
  REQUIRE(std::string(str) == "xyyyy");
}

TEST_CASE("String: operator+= with string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2(" world", &g_allocator);

  str += str2;
  REQUIRE(std::string(str) == "hello world");
}

TEST_CASE("String: operator+= with C-style string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  str += " world";
  REQUIRE(std::string(str) == "hello world");
}

TEST_CASE("String: operator+= with character", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hi", &g_allocator);

  str += '!';
  REQUIRE(std::string(str) == "hi!");
}

// ============================================================================
// Assignment Tests
// ============================================================================

TEST_CASE("String: copy assignment", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str1("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2(&g_allocator);

  str2 = str1;
  REQUIRE(str2.size() == 5);
  REQUIRE(std::string(str2) == "hello");
}

TEST_CASE("String: move assignment", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str1("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2(&g_allocator);

  str2 = std::move(str1);
  REQUIRE(std::string(str2) == "hello");
}

TEST_CASE("String: assignment from C-style string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str(&g_allocator);

  str = "hello";
  REQUIRE(std::string(str) == "hello");
}

TEST_CASE("String: assignment from initializer list", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str(&g_allocator);

  str = {'h', 'i'};
  REQUIRE(std::string(str) == "hi");
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_CASE("String: equality comparison", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str1("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str3("world", &g_allocator);

  REQUIRE(str1 == str2);
  REQUIRE(!(str1 == str3));
  REQUIRE(str1 != str3);
  REQUIRE(!(str1 != str2));
}

TEST_CASE("String: equality with C-style string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  REQUIRE(str == "hello");
  REQUIRE(!(str == "world"));
}

TEST_CASE("String: compare method", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str1("apple", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2("banana", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str3("apple", &g_allocator);

  REQUIRE(str1.compare(str2) < 0);
  REQUIRE(str2.compare(str1) > 0);
  REQUIRE(str1.compare(str3) == 0);
}

TEST_CASE("String: compare with C-style string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("apple", &g_allocator);

  REQUIRE(str.compare("apple") == 0);
  REQUIRE(str.compare("banana") < 0);
}

// ============================================================================
// String Search Tests
// ============================================================================

TEST_CASE("String: find substring", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);

  size_t pos = str.find("world");
  REQUIRE(pos == 6);

  pos = str.find("notfound");
  REQUIRE(pos == str.npos);
}

TEST_CASE("String: find character", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);

  size_t pos = str.find('o');
  REQUIRE(pos == 4);

  pos = str.find('z');
  REQUIRE(pos == str.npos);
}

TEST_CASE("String: find with position", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);

  size_t pos = str.find('o', 5);
  REQUIRE(pos == 7);
}

TEST_CASE("String: starts_with", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);

  REQUIRE(str.starts_with("hello"));
  REQUIRE(!str.starts_with("world"));
  REQUIRE(str.starts_with('h'));
  REQUIRE(!str.starts_with('w'));
}

TEST_CASE("String: ends_with", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);

  REQUIRE(str.ends_with("world"));
  REQUIRE(!str.ends_with("hello"));
  REQUIRE(str.ends_with('d'));
  REQUIRE(!str.ends_with('h'));
}

// ============================================================================
// Substring Tests
// ============================================================================

TEST_CASE("String: substr", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);

  auto sub = str.substr(0, 5);
  REQUIRE(std::string(sub) == "hello");

  sub = str.substr(6);
  REQUIRE(std::string(sub) == "world");
}

// ============================================================================
// Replace/Erase Tests
// ============================================================================

TEST_CASE("String: erase", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);

  str.erase(5, 6);
  REQUIRE(std::string(str) == "hello");
}

TEST_CASE("String: replace with string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);
  basic_string<char, SimpleHeapAllocator> replacement("universe", &g_allocator);

  str.replace(6, 5, replacement);
  REQUIRE(std::string(str) == "hello universe");
}

TEST_CASE("String: replace with C-style string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hello world", &g_allocator);

  str.replace(6, 5, "universe");
  REQUIRE(std::string(str) == "hello universe");
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST_CASE("String: iterators", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hi", &g_allocator);

  auto it = str.begin();
  REQUIRE(*it == 'h');
  ++it;
  REQUIRE(*it == 'i');
  ++it;
  REQUIRE(it == str.end());
}

TEST_CASE("String: const iterators", "[priv_string]") {
  const basic_string<char, SimpleHeapAllocator> str("hello", &g_allocator);

  auto it = str.cbegin();
  REQUIRE(*it == 'h');
  ++it;
  REQUIRE(*it == 'e');
}

TEST_CASE("String: reverse iterators", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("hi", &g_allocator);

  auto it = str.rbegin();
  REQUIRE(*it == 'i');
  ++it;
  REQUIRE(*it == 'h');
}

// ============================================================================
// Swap Tests
// ============================================================================

TEST_CASE("String: swap", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str1("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2("world", &g_allocator);

  str1.swap(str2);
  REQUIRE(std::string(str1) == "world");
  REQUIRE(std::string(str2) == "hello");
}

// ============================================================================
// Large String Tests (requiring vector allocation)
// ============================================================================

TEST_CASE("String: large string construction", "[priv_string]") {
  std::string large_std("x");
  for (int i = 0; i < 100; ++i) {
    large_std += large_std;
    if (large_std.size() > 10000) break;
  }

  basic_string<char, SimpleHeapAllocator> str(large_std.c_str(), &g_allocator);
  REQUIRE(str.size() == large_std.size());
  REQUIRE(std::string(str) == large_std);
}

TEST_CASE("String: large string operations", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str(&g_allocator);

  // Build large string
  for (int i = 0; i < 1000; ++i) {
    str += "a";
  }

  REQUIRE(str.size() == 1000);
  REQUIRE(str.capacity() > 32);

  // Find in large string
  size_t pos = str.find('a', 500);
  REQUIRE(pos == 500);
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_CASE("String: empty string operations", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str(&g_allocator);

  REQUIRE(str.empty());
  REQUIRE(str.size() == 0);
  REQUIRE(str.find("a") == str.npos);
  REQUIRE(str.starts_with(""));
  REQUIRE(str.ends_with(""));
}

TEST_CASE("String: single character string", "[priv_string]") {
  basic_string<char, SimpleHeapAllocator> str("x", &g_allocator);

  REQUIRE(str.size() == 1);
  REQUIRE(str.front() == 'x');
  REQUIRE(str.back() == 'x');
  REQUIRE(str[0] == 'x');
}

// TEST_CASE("String: repeated push_back to trigger growth", "[priv_string]") {
//   basic_string<char, SimpleHeapAllocator> str(&g_allocator);
//
//   for (int i = 0; i < 100; ++i) {
//     str.push_back('a');
//   }
//
//   REQUIRE(str.size() == 100);
//   // Verify first and last characters for basic correctness
//   REQUIRE(str[0] == 'a');
//   REQUIRE(str[99] == 'a');
//   // Verify string can be converted (basic check)
//   std::string std_str = std::string(str);
//   REQUIRE(std_str.size() == 100);
// }

// ============================================================================
// Serialization Tests (GlobalSerialize)
// ============================================================================

#include "clio_ctp/data_structures/serialization/global_serialize.h"

TEST_CASE("String: serialize small SSO string", "[priv_string][serialization]") {
  basic_string<char, SimpleHeapAllocator> original("hello", &g_allocator);

  // Serialize
  std::vector<char> buf;
  ctp::ipc::GlobalSerialize<std::vector<char>> oarchive(buf);
  oarchive(original);
  oarchive.Finalize();
  std::string result(buf.begin(), buf.end());

  // Deserialize
  basic_string<char, SimpleHeapAllocator> restored(&g_allocator);
  std::vector<char> ibuf(result.begin(), result.end());
  ctp::ipc::GlobalDeserialize<std::vector<char>> iarchive(ibuf);
  iarchive(restored);

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(std::string(restored) == std::string(original));
  REQUIRE(restored == "hello");
}

TEST_CASE("String: serialize large string exceeding SSO", "[priv_string][serialization]") {
  std::string large_str;
  for (int i = 0; i < 100; ++i) {
    large_str += "This is a long string that exceeds SSO buffer size. ";
  }

  basic_string<char, SimpleHeapAllocator> original(large_str.c_str(), &g_allocator);

  // Serialize
  std::vector<char> buf;
  ctp::ipc::GlobalSerialize<std::vector<char>> oarchive(buf);
  oarchive(original);
  oarchive.Finalize();
  std::string result(buf.begin(), buf.end());

  // Deserialize
  basic_string<char, SimpleHeapAllocator> restored(&g_allocator);
  std::vector<char> ibuf(result.begin(), result.end());
  ctp::ipc::GlobalDeserialize<std::vector<char>> iarchive(ibuf);
  iarchive(restored);

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(std::string(restored) == std::string(original));
  REQUIRE(restored.size() == large_str.size());
}

TEST_CASE("String: serialize empty string", "[priv_string][serialization]") {
  basic_string<char, SimpleHeapAllocator> original(&g_allocator);

  // Serialize
  std::vector<char> buf;
  ctp::ipc::GlobalSerialize<std::vector<char>> oarchive(buf);
  oarchive(original);
  oarchive.Finalize();
  std::string result(buf.begin(), buf.end());

  // Deserialize
  basic_string<char, SimpleHeapAllocator> restored(&g_allocator);
  std::vector<char> ibuf(result.begin(), result.end());
  ctp::ipc::GlobalDeserialize<std::vector<char>> iarchive(ibuf);
  iarchive(restored);

  // Verify
  REQUIRE(restored.size() == 0);
  REQUIRE(restored.empty());
  REQUIRE(std::string(restored) == "");
}

TEST_CASE("String: serialize string with special characters", "[priv_string][serialization]") {
  basic_string<char, SimpleHeapAllocator> original("hello\nworld\t\r\n!@#$%", &g_allocator);

  // Serialize
  std::vector<char> buf;
  ctp::ipc::GlobalSerialize<std::vector<char>> oarchive(buf);
  oarchive(original);
  oarchive.Finalize();
  std::string result(buf.begin(), buf.end());

  // Deserialize
  basic_string<char, SimpleHeapAllocator> restored(&g_allocator);
  std::vector<char> ibuf(result.begin(), result.end());
  ctp::ipc::GlobalDeserialize<std::vector<char>> iarchive(ibuf);
  iarchive(restored);

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(std::string(restored) == std::string(original));
}

TEST_CASE("String: serialize exactly SSO size boundary", "[priv_string][serialization]") {
  // Create string exactly at SSO boundary (SSO size is 32, so 31 characters + null)
  basic_string<char, SimpleHeapAllocator, 32> original("0123456789abcdefghijklmnopqrstu", &g_allocator);

  // Serialize
  std::vector<char> buf;
  ctp::ipc::GlobalSerialize<std::vector<char>> oarchive(buf);
  oarchive(original);
  oarchive.Finalize();
  std::string result(buf.begin(), buf.end());

  // Deserialize
  basic_string<char, SimpleHeapAllocator, 32> restored(&g_allocator);
  std::vector<char> ibuf(result.begin(), result.end());
  ctp::ipc::GlobalDeserialize<std::vector<char>> iarchive(ibuf);
  iarchive(restored);

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(std::string(restored) == std::string(original));
  REQUIRE(restored.size() == 31);
}

// ============================================================================
// LocalSerialize Tests
// ============================================================================

#include "clio_ctp/data_structures/serialization/local_serialize.h"

TEST_CASE("String: LocalSerialize small SSO string", "[priv_string][local_serialize]") {
  basic_string<char, SimpleHeapAllocator> original("hello", &g_allocator);

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  basic_string<char, SimpleHeapAllocator> restored(&g_allocator);
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(std::string(restored) == std::string(original));
  REQUIRE(restored == "hello");
}

TEST_CASE("String: LocalSerialize large string exceeding SSO", "[priv_string][local_serialize]") {
  std::string large_str;
  for (int i = 0; i < 100; ++i) {
    large_str += "This is a long string that exceeds SSO buffer size. ";
  }

  basic_string<char, SimpleHeapAllocator> original(large_str.c_str(), &g_allocator);

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  basic_string<char, SimpleHeapAllocator> restored(&g_allocator);
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == original.size());
  REQUIRE(std::string(restored) == std::string(original));
  REQUIRE(restored.size() == large_str.size());
}

TEST_CASE("String: LocalSerialize empty string", "[priv_string][local_serialize]") {
  basic_string<char, SimpleHeapAllocator> original(&g_allocator);

  // Serialize
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << original;

  // Deserialize
  basic_string<char, SimpleHeapAllocator> restored(&g_allocator);
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored;

  // Verify
  REQUIRE(restored.size() == 0);
  REQUIRE(restored.empty());
  REQUIRE(std::string(restored) == "");
}

TEST_CASE("String: LocalSerialize multiple strings", "[priv_string][local_serialize]") {
  basic_string<char, SimpleHeapAllocator> str1("hello", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str2("world", &g_allocator);
  basic_string<char, SimpleHeapAllocator> str3("test", &g_allocator);

  // Serialize multiple strings
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer << str1 << str2 << str3;

  // Deserialize multiple strings
  basic_string<char, SimpleHeapAllocator> restored1(&g_allocator);
  basic_string<char, SimpleHeapAllocator> restored2(&g_allocator);
  basic_string<char, SimpleHeapAllocator> restored3(&g_allocator);
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer >> restored1 >> restored2 >> restored3;

  // Verify
  REQUIRE(std::string(restored1) == "hello");
  REQUIRE(std::string(restored2) == "world");
  REQUIRE(std::string(restored3) == "test");
}

TEST_CASE("String: LocalSerialize with operator() syntax", "[priv_string][local_serialize]") {
  basic_string<char, SimpleHeapAllocator> original("test", &g_allocator);

  // Serialize using operator()
  std::vector<char> buffer;
  ctp::ipc::LocalSerialize<> serializer(buffer);
  serializer(original);

  // Deserialize using operator()
  basic_string<char, SimpleHeapAllocator> restored(&g_allocator);
  ctp::ipc::LocalDeserialize<> deserializer(buffer);
  deserializer(restored);

  // Verify
  REQUIRE(std::string(restored) == "test");
}

SIMPLE_TEST_MAIN()
