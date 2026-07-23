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

#include <catch2/catch_test_macros.hpp>

#include "clio_ctp/data_structures/ipc/string.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"
#include "clio_ctp/memory/backend/malloc_backend.h"

#include <string>

using namespace ctp::ipc;

using Alloc = ArenaAllocator<false>;
using shm_string = string<Alloc>;

static Alloc *CreateTestAllocator(MallocBackend &backend, size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);
  return backend.MakeAlloc<Alloc>();
}

TEST_CASE("IpcString: default and empty", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  shm_string s(alloc);
  REQUIRE(s.size() == 0);
  REQUIRE(s.empty());
  REQUIRE(s.is_inline());
  REQUIRE(std::string(s.c_str()) == "");
}

TEST_CASE("IpcString: short string stays inline", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  shm_string s(alloc, "blob_name");
  REQUIRE(s.size() == 9);
  REQUIRE(s.is_inline());  // the whole point of SSO for short keys
  REQUIRE(std::string(s.c_str()) == "blob_name");
  REQUIRE(s == "blob_name");
  REQUIRE(s != "blob_nam");
}

TEST_CASE("IpcString: exactly at the SSO boundary", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  // kSsoCapacity bytes must still be inline; one more must spill.
  std::string at(shm_string::kSsoCapacity, 'x');
  std::string over(shm_string::kSsoCapacity + 1, 'y');

  shm_string s_at(alloc, at);
  REQUIRE(s_at.size() == shm_string::kSsoCapacity);
  REQUIRE(s_at.is_inline());
  REQUIRE(s_at.str() == at);

  shm_string s_over(alloc, over);
  REQUIRE(s_over.size() == shm_string::kSsoCapacity + 1);
  REQUIRE_FALSE(s_over.is_inline());
  REQUIRE(s_over.str() == over);
}

TEST_CASE("IpcString: long string round-trips", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  std::string big(5000, 'z');
  big[0] = 'a';
  big[4999] = 'b';

  shm_string s(alloc, big);
  REQUIRE(s.size() == 5000);
  REQUIRE_FALSE(s.is_inline());
  REQUIRE(s.str() == big);
  REQUIRE(s[0] == 'a');
  REQUIRE(s[4999] == 'b');
}

TEST_CASE("IpcString: assign replaces contents and can shrink", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  shm_string s(alloc, std::string(100, 'q'));
  REQUIRE_FALSE(s.is_inline());

  REQUIRE(s.assign("short"));
  REQUIRE(s.size() == 5);
  REQUIRE(s == "short");
  // Growth is one-way: the buffer stays on the heap so a reader that already
  // resolved the offset is never left pointing at freed memory.
  REQUIRE_FALSE(s.is_inline());
}

TEST_CASE("IpcString: embedded NUL is preserved", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  const char raw[] = {'a', '\0', 'b'};
  shm_string s(alloc, raw, 3);
  REQUIRE(s.size() == 3);
  REQUIRE(s[0] == 'a');
  REQUIRE(s[1] == '\0');
  REQUIRE(s[2] == 'b');
}

TEST_CASE("IpcString: copy into allocator", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  shm_string a(alloc, "hello world, this is a longer string");
  shm_string b(alloc, a);
  REQUIRE(b.str() == a.str());
  REQUIRE(a == b);

  // Mutating the copy must not disturb the original (no shared buffer).
  REQUIRE(b.assign("changed"));
  REQUIRE(b == "changed");
  REQUIRE(a == "hello world, this is a longer string");
}

TEST_CASE("IpcString: clear", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  shm_string s(alloc, "something");
  s.clear();
  REQUIRE(s.empty());
  REQUIRE(s.size() == 0);
  REQUIRE(s == "");
}

TEST_CASE("IpcString: hash is content-based and stable", "[ipc_string]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 1024 * 1024);

  shm_string a(alloc, "same_key");
  shm_string b(alloc, "same_key");
  shm_string c(alloc, "other_key");

  REQUIRE(a.hash() == b.hash());
  REQUIRE(a.hash() != c.hash());

  // Equal for inline and spilled representations of the same content, so a
  // hash table cannot lose an entry just because it crossed the SSO boundary.
  std::string longkey(shm_string::kSsoCapacity + 10, 'k');
  shm_string d(alloc, longkey);
  REQUIRE(d.hash() == shm_string::HashBytes(longkey.data(), longkey.size()));

  // Hashing a raw key must match hashing the string -- the map probes with
  // const char* to avoid constructing a string per lookup.
  REQUIRE(a.hash() == shm_string::HashBytes("same_key", 8));
}

TEST_CASE("IpcString: allocation failure degrades to empty", "[ipc_string]") {
  MallocBackend backend;
  // Tiny arena: a large assign cannot be satisfied.
  auto *alloc = CreateTestAllocator(backend, 4096);

  shm_string s(alloc);
  std::string huge(1024 * 1024, 'x');
  bool ok = s.assign(huge.data(), huge.size());
  // Either it fit or it failed -- but it must never be left half-written,
  // because the cache falls back to RPC on failure and a torn string would
  // be read by clients before that happens.
  if (!ok) {
    REQUIRE(s.empty());
    REQUIRE(s.size() == 0);
  }
}
