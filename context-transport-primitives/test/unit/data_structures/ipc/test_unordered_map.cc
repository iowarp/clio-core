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

#include "clio_ctp/data_structures/ipc/unordered_map.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"
#include "clio_ctp/memory/backend/malloc_backend.h"

#include <string>
#include <vector>

using namespace ctp::ipc;

using Alloc = ArenaAllocator<false>;
using shm_string = string<Alloc>;

/** Stand-in for a cached BlobInfo record: trivially copyable POD. */
struct Rec {
  ctp::u64 a;
  ctp::u64 b;
  Rec() : a(0), b(0) {}
  Rec(ctp::u64 x, ctp::u64 y) : a(x), b(y) {}
  bool operator==(const Rec &o) const { return a == o.a && b == o.b; }
};

using StrMap = unordered_map<shm_string, Rec, Alloc>;

static Alloc *CreateTestAllocator(MallocBackend &backend, size_t arena_size) {
  backend.shm_init(MemoryBackendId(0, 0), arena_size);
  return backend.MakeAlloc<Alloc>();
}

static bool Put(StrMap &m, const std::string &k, const Rec &v) {
  StrMap::BytesProbe p{k.data(), k.size()};
  return m.InsertOrAssign(p, shm_string::HashBytes(k.data(), k.size()), v);
}

TEST_CASE("IpcMap: capacity rounds up to a power of two", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  StrMap m(alloc, 100);
  REQUIRE(m.valid());
  REQUIRE(m.capacity() == 128);
  REQUIRE(m.size() == 0);
}

TEST_CASE("IpcMap: insert and read back", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  StrMap m(alloc, 64);
  REQUIRE(Put(m, "tag/blob_a", Rec(1, 2)));
  REQUIRE(Put(m, "tag/blob_b", Rec(3, 4)));
  REQUIRE(m.size() == 2);

  Rec out;
  REQUIRE(m.TryGetBytes("tag/blob_a", 10, &out));
  REQUIRE(out == Rec(1, 2));
  REQUIRE(m.TryGetBytes("tag/blob_b", 10, &out));
  REQUIRE(out == Rec(3, 4));
}

TEST_CASE("IpcMap: missing key returns false", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  StrMap m(alloc, 64);
  REQUIRE(Put(m, "present", Rec(7, 7)));

  Rec out;
  REQUIRE_FALSE(m.TryGetBytes("absent", 6, &out));
}

TEST_CASE("IpcMap: overwrite replaces value, not size", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  StrMap m(alloc, 64);
  REQUIRE(Put(m, "k", Rec(1, 1)));
  REQUIRE(Put(m, "k", Rec(9, 9)));
  REQUIRE(m.size() == 1);

  Rec out;
  REQUIRE(m.TryGetBytes("k", 1, &out));
  REQUIRE(out == Rec(9, 9));
}

TEST_CASE("IpcMap: erase leaves probe sequence intact", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  StrMap m(alloc, 64);
  // Insert enough keys that some collide and form probe chains.
  std::vector<std::string> keys;
  for (int i = 0; i < 20; ++i) {
    keys.push_back("key_" + std::to_string(i));
    REQUIRE(Put(m, keys.back(), Rec(i, i * 2)));
  }

  // Erase every third key, then confirm the survivors are all still findable.
  // A tombstone that terminated a probe would lose them.
  for (size_t i = 0; i < keys.size(); i += 3) {
    StrMap::BytesProbe p{keys[i].data(), keys[i].size()};
    REQUIRE(m.Erase(p, shm_string::HashBytes(keys[i].data(), keys[i].size())));
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    Rec out;
    bool found = m.TryGetBytes(keys[i].data(), keys[i].size(), &out);
    if (i % 3 == 0) {
      REQUIRE_FALSE(found);
    } else {
      REQUIRE(found);
      REQUIRE(out == Rec(i, i * 2));
    }
  }
}

TEST_CASE("IpcMap: erased slot is reusable", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  StrMap m(alloc, 64);
  REQUIRE(Put(m, "recycle", Rec(1, 1)));
  StrMap::BytesProbe p{"recycle", 7};
  REQUIRE(m.Erase(p, shm_string::HashBytes("recycle", 7)));
  REQUIRE(m.size() == 0);

  REQUIRE(Put(m, "recycle", Rec(5, 5)));
  Rec out;
  REQUIRE(m.TryGetBytes("recycle", 7, &out));
  REQUIRE(out == Rec(5, 5));
}

TEST_CASE("IpcMap: full table reports failure and never grows", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  StrMap m(alloc, 8);  // rounds to 8
  REQUIRE(m.capacity() == 8);

  int inserted = 0;
  for (int i = 0; i < 64; ++i) {
    if (Put(m, "k" + std::to_string(i), Rec(i, i))) {
      ++inserted;
    }
  }
  // Must have refused the overflow rather than rehashing -- a rehash would
  // free the table out from under cross-process readers.
  //
  // Note it stops at the LOAD CAP (7/8), not at 100%. Filling every slot
  // leaves no empty slot to terminate a probe run, which makes every MISS scan
  // the entire table: measured at 108.9 us on a 65536-slot table, i.e. worse
  // than the RPC the cache exists to avoid. Reserving 1/8 keeps misses at
  // ~0.1 us.
  REQUIRE(inserted == 7);
  REQUIRE(m.capacity() == 8);
  REQUIRE(m.size() == 7);

  // The entries that WERE accepted must still be readable, and a miss on the
  // saturated table must still return quickly rather than scanning it.
  Rec out;
  REQUIRE(m.TryGetBytes("k0", 2, &out));
  REQUIRE_FALSE(m.TryGetBytes("k63", 3, &out));
}

TEST_CASE("IpcMap: load cap leaves headroom at scale", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 256 * 1024 * 1024);

  StrMap m(alloc, 4096);
  REQUIRE(m.capacity() == 4096);
  int inserted = 0;
  for (int i = 0; i < 8192; ++i) {
    if (Put(m, "load_key_" + std::to_string(i), Rec(i, i))) {
      ++inserted;
    }
  }
  // 7/8 of 4096. The load factor must be the binding constraint, NOT the
  // probe-length cap: an over-tight probe cap silently rejects inserts far
  // below the load limit (measured: a 64-probe cap stopped a 65536-slot table
  // at ~51% occupancy).
  REQUIRE(inserted == 3584);
  REQUIRE(m.size() == 3584);

  // Overwrites of existing keys must still succeed at the cap, so a saturated
  // cache keeps its entries fresh instead of going stale.
  REQUIRE(Put(m, "load_key_0", Rec(999, 999)));
  Rec out;
  REQUIRE(m.TryGetBytes("load_key_0", 10, &out));
  REQUIRE(out == Rec(999, 999));
  REQUIRE(m.size() == 3584);
}

TEST_CASE("IpcMap: many keys round-trip", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 64 * 1024 * 1024);

  StrMap m(alloc, 4096);
  const int kN = 2000;
  for (int i = 0; i < kN; ++i) {
    REQUIRE(Put(m, "blob_name_number_" + std::to_string(i), Rec(i, i + 1)));
  }
  REQUIRE(m.size() == kN);
  for (int i = 0; i < kN; ++i) {
    std::string k = "blob_name_number_" + std::to_string(i);
    Rec out;
    REQUIRE(m.TryGetBytes(k.data(), k.size(), &out));
    REQUIRE(out == Rec(i, i + 1));
  }
}

TEST_CASE("IpcMap: long keys crossing the SSO boundary", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 64 * 1024 * 1024);

  StrMap m(alloc, 64);
  std::string shortk(10, 's');
  std::string longk(200, 'l');
  REQUIRE(Put(m, shortk, Rec(1, 1)));
  REQUIRE(Put(m, longk, Rec(2, 2)));

  Rec out;
  REQUIRE(m.TryGetBytes(shortk.data(), shortk.size(), &out));
  REQUIRE(out == Rec(1, 1));
  REQUIRE(m.TryGetBytes(longk.data(), longk.size(), &out));
  REQUIRE(out == Rec(2, 2));
}

TEST_CASE("IpcMap: seqlock rejects a torn read", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  StrMap m(alloc, 64);
  REQUIRE(Put(m, "torn", Rec(1, 1)));

  // Simulate a writer that died mid-update: leave the slot generation odd.
  // A reader must refuse to return a value rather than read half of one.
  auto *slot = m.FindSlot(StrMap::BytesProbe{"torn", 4},
                          shm_string::HashBytes("torn", 4));
  REQUIRE(slot != nullptr);
  slot->gen_.fetch_add(1);  // now odd == "write in progress"

  Rec out;
  REQUIRE_FALSE(m.TryGetBytes("torn", 4, &out, 4));

  // Completing the write makes it readable again.
  slot->gen_.fetch_add(1);
  REQUIRE(m.TryGetBytes("torn", 4, &out, 4));
  REQUIRE(out == Rec(1, 1));
}

TEST_CASE("IpcMap: POD keys work without an allocator-aware key", "[ipc_map]") {
  MallocBackend backend;
  auto *alloc = CreateTestAllocator(backend, 16 * 1024 * 1024);

  using PodMap = unordered_map<ctp::u64, Rec, Alloc>;
  PodMap m(alloc, 64);

  for (ctp::u64 i = 0; i < 30; ++i) {
    REQUIRE(m.InsertOrAssign(i, PodMap::Hash(i), Rec(i, i * 3)));
  }
  REQUIRE(m.size() == 30);
  for (ctp::u64 i = 0; i < 30; ++i) {
    Rec out;
    REQUIRE(m.TryGet(i, PodMap::Hash(i), &out));
    REQUIRE(out == Rec(i, i * 3));
  }
  Rec out;
  REQUIRE_FALSE(m.TryGet(ctp::u64(999), PodMap::Hash(ctp::u64(999)), &out));
}
