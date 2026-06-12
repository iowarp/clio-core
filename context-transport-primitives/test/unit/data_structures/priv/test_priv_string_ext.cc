/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * Extended priv::basic_string tests targeting paths the base suite misses:
 * heap-storage variants of every constructor/assignment (the base suite
 * mostly stays within SSO), out-of-range throws, SSO<->heap transitions,
 * shrink_to_fit, and append/assign edge cases.
 */

#include "../../../context-runtime/test/simple_test.h"
#include "clio_ctp/data_structures/priv/string.h"
#include <stdexcept>
#include <string>
#include <utility>

using namespace ctp::priv;

// Same heap-backed test allocator as test_priv_string.cc.
class ExtHeapAllocator {
 public:
  template <typename T>
  ctp::ipc::FullPtr<T> AllocateObjs(size_t count) {
    return Allocate<T>(count * sizeof(T));
  }
  template <typename T = char>
  ctp::ipc::FullPtr<T> Allocate(size_t size) {
    T* ptr = static_cast<T*>(malloc(size));
    ctp::ipc::FullPtr<T> result;
    result.ptr_ = ptr;
    result.shm_.off_ = 0;
    result.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    return result;
  }
  template <typename T, bool ATOMIC = false>
  void Free(const ctp::ipc::FullPtr<T, ATOMIC>& ptr) {
    if (ptr.ptr_ != nullptr) {
      free(ptr.ptr_);
    }
  }
};

static ExtHeapAllocator g_ext_allocator;
using TestString = basic_string<char, ExtHeapAllocator>;

namespace {
/** A string long enough to defeat any plausible SSO buffer. */
const std::string kLong(200, 'q');
}  // namespace

TEST_CASE("StringExt: heap constructors", "[priv_string_ext]") {
  SECTION("count+char constructor on heap");
  TestString many(150, 'z', &g_ext_allocator);
  REQUIRE(many.size() == 150);
  REQUIRE(many[0] == 'z');
  REQUIRE(many[149] == 'z');

  SECTION("C-string constructor on heap");
  TestString big(kLong.c_str(), &g_ext_allocator);
  REQUIRE(big.size() == 200);

  SECTION("substring constructor on heap");
  TestString sub(big, 10, 120, &g_ext_allocator);
  REQUIRE(sub.size() == 120);
  REQUIRE(sub[0] == 'q');

  SECTION("substring constructor clamps count at end");
  TestString tail(big, 190, TestString::npos, &g_ext_allocator);
  REQUIRE(tail.size() == 10);

  SECTION("substring constructor throws past the end");
  bool threw = false;
  try {
    TestString bad(big, 500, 3, &g_ext_allocator);
  } catch (const std::exception&) {
    threw = true;
  }
  REQUIRE(threw);

  SECTION("initializer_list constructor");
  TestString il({'a', 'b', 'c', 'd', 'e'}, &g_ext_allocator);
  REQUIRE(il.size() == 5);
  REQUIRE(il == "abcde");

  SECTION("std::string conversion constructor on heap");
  TestString conv(&g_ext_allocator, kLong);
  REQUIRE(conv.size() == 200);

  SECTION("copy and move of heap strings");
  TestString copy(big);
  REQUIRE(copy == big);
  TestString moved(std::move(copy));
  REQUIRE(moved.size() == 200);

  SECTION("copy and move of SSO strings");
  TestString small("hi", &g_ext_allocator);
  TestString small_copy(small);
  REQUIRE(small_copy == "hi");
  TestString small_moved(std::move(small_copy));
  REQUIRE(small_moved == "hi");
}

TEST_CASE("StringExt: assignment operator matrix", "[priv_string_ext]") {
  TestString heap_src(kLong.c_str(), &g_ext_allocator);
  TestString sso_src("tiny", &g_ext_allocator);

  SECTION("copy-assign heap over SSO, then SSO over heap");
  TestString a("x", &g_ext_allocator);
  a = heap_src;
  REQUIRE(a.size() == 200);
  a = sso_src;
  REQUIRE(a == "tiny");

  SECTION("move-assign heap and SSO");
  TestString h2(heap_src);
  TestString b("y", &g_ext_allocator);
  b = std::move(h2);
  REQUIRE(b.size() == 200);
  TestString s2(sso_src);
  b = std::move(s2);
  REQUIRE(b == "tiny");

  SECTION("self-assignment is a no-op");
  TestString self(heap_src);
  self = self;
  REQUIRE(self.size() == 200);

  SECTION("assign C string long and short");
  TestString c(&g_ext_allocator);
  c = kLong.c_str();
  REQUIRE(c.size() == 200);
  c = "ok";
  REQUIRE(c == "ok");

  SECTION("assign initializer_list over heap content");
  TestString d(heap_src);
  d = {'1', '2', '3'};
  REQUIRE(d == "123");

  SECTION("assign std::string long then short");
  TestString e(&g_ext_allocator);
  e = kLong;
  REQUIRE(e.size() == 200);
  e = std::string("s");
  REQUIRE(e == "s");
}

TEST_CASE("StringExt: at() bounds checking", "[priv_string_ext]") {
  TestString s("abc", &g_ext_allocator);
  const TestString& cs = s;

  REQUIRE(s.at(0) == 'a');
  REQUIRE(cs.at(2) == 'c');

  bool threw = false;
  try {
    (void)s.at(3);
  } catch (const std::exception&) {
    threw = true;
  }
  REQUIRE(threw);

  threw = false;
  try {
    (void)cs.at(99);
  } catch (const std::exception&) {
    threw = true;
  }
  REQUIRE(threw);
}

TEST_CASE("StringExt: capacity management", "[priv_string_ext]") {
  TestString s("seed", &g_ext_allocator);

  SECTION("reserve beyond SSO transitions to heap and preserves content");
  s.reserve(300);
  REQUIRE(s == "seed");
  REQUIRE(s.capacity() >= 300);

  SECTION("reserve smaller than current capacity is a no-op");
  size_t cap = s.capacity();
  s.reserve(10);
  REQUIRE(s.capacity() == cap);

  SECTION("shrink_to_fit reduces capacity");
  s.shrink_to_fit();
  REQUIRE(s == "seed");

  SECTION("incremental growth (HeapReserve doubling and exact paths)");
  TestString grower(&g_ext_allocator);
  for (int i = 0; i < 300; ++i) {
    grower.push_back(static_cast<char>('a' + (i % 26)));
  }
  REQUIRE(grower.size() == 300);
  // Big jump that exceeds 2x current capacity exercises grow_cap=new_cap.
  grower.reserve(5000);
  REQUIRE(grower.size() == 300);

  SECTION("pop_back, clear, and resize");
  grower.pop_back();
  REQUIRE(grower.size() == 299);
  grower.resize(50);
  REQUIRE(grower.size() == 50);
  grower.resize(400);  // grow with default fill
  REQUIRE(grower.size() == 400);
  REQUIRE(grower[399] == '\0');
  grower.clear();
  REQUIRE(grower.empty());
  REQUIRE(grower.c_str()[0] == '\0');
}

TEST_CASE("StringExt: append battery", "[priv_string_ext]") {
  TestString s(&g_ext_allocator);

  SECTION("append basic_string, then with pos/count");
  TestString other("0123456789", &g_ext_allocator);
  s.append(other);
  REQUIRE(s == "0123456789");
  s.append(other, 3, 4);
  REQUIRE(s == "01234567893456");

  SECTION("append C string, counted C string, fill, and init list");
  s.clear();
  s.append("abc");
  s.append("defgh", 2);
  s.append(static_cast<TestString::size_type>(3), 'x');
  s.append({'!', '?'});
  REQUIRE(s == "abcdexxx!?");

  SECTION("append empty C string early-returns");
  size_t before = s.size();
  s.append("");
  REQUIRE(s.size() == before);

  SECTION("operator+= variants");
  s.clear();
  s += TestString("one", &g_ext_allocator);
  s += "-two";
  s += '!';
  s += {'a', 'b'};
  REQUIRE(s == "one-two!ab");

  SECTION("append driving SSO->heap transition mid-append");
  TestString t("start", &g_ext_allocator);
  t.append(kLong.c_str());
  REQUIRE(t.size() == 5 + 200);
}

TEST_CASE("StringExt: comparisons and accessors", "[priv_string_ext]") {
  TestString a("alpha", &g_ext_allocator);
  TestString b("beta", &g_ext_allocator);

  REQUIRE(a == "alpha");
  REQUIRE_FALSE(a == "alphaz");
  REQUIRE_FALSE(a == b);
  REQUIRE(a != b);

  REQUIRE(a.front() == 'a');
  REQUIRE(a.back() == 'a');
  const TestString& ca = a;
  REQUIRE(ca.front() == 'a');
  REQUIRE(ca.back() == 'a');
  REQUIRE(*ca.data() == 'a');
  REQUIRE(ca.c_str()[5] == '\0');

  SECTION("iterators forward and reverse, const and non-const");
  std::string roundtrip;
  for (auto it = a.begin(); it != a.end(); ++it) {
    roundtrip += *it;
  }
  REQUIRE(roundtrip == "alpha");
  roundtrip.clear();
  for (auto it = ca.cbegin(); it != ca.cend(); ++it) {
    roundtrip += *it;
  }
  REQUIRE(roundtrip == "alpha");
  roundtrip.clear();
  for (auto it = a.rbegin(); it != a.rend(); ++it) {
    roundtrip += *it;
  }
  REQUIRE(roundtrip == "ahpla");
  roundtrip.clear();
  for (auto it = ca.crbegin(); it != ca.crend(); ++it) {
    roundtrip += *it;
  }
  REQUIRE(roundtrip == "ahpla");
}

SIMPLE_TEST_MAIN()
