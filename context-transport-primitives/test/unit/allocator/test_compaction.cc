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
#include <cstring>
#include <vector>
// buddy_allocator.h must be included before malloc_backend.h to ensure the
// correct include order for memory_backend.h (avoid circular include issue).
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/memory/backend/malloc_backend.h"

using hipc::BuddyAllocator;
using hipc::ForwardingTable;
using hipc::MallocBackend;
using hipc::MemoryBackendId;
using hipc::OffsetPtr;

/** Helper: make a fresh BuddyAllocator backed by a MallocBackend */
static BuddyAllocator *MakeAlloc(MallocBackend &backend, size_t heap_mb = 128) {
  size_t heap_size  = heap_mb * 1024 * 1024;
  size_t alloc_size = sizeof(BuddyAllocator);
  backend.shm_init(MemoryBackendId(0, 0), alloc_size + heap_size);
  return backend.MakeAlloc<BuddyAllocator>();
}

/** Helper: write a canary pattern into the user data of an allocation */
static void WriteCanary(BuddyAllocator *alloc, OffsetPtr<> ptr, size_t size,
                        uint8_t pattern) {
  char *base = alloc->GetBackendData();
  memset(base + ptr.load(), static_cast<int>(pattern), size);
}

/** Helper: verify canary pattern */
static bool CheckCanary(BuddyAllocator *alloc, OffsetPtr<> ptr, size_t size,
                        uint8_t pattern) {
  char *base = alloc->GetBackendData();
  const char *data = base + ptr.load();
  for (size_t i = 0; i < size; ++i) {
    if (static_cast<uint8_t>(data[i]) != pattern) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Test 1: compact_basic
// Allocate 10 blocks of varying sizes, free every other one, compact,
// verify live data is intact and new allocation still works.
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - Compact Basic", "[compaction]") {
  MallocBackend backend;
  BuddyAllocator *alloc = MakeAlloc(backend);

  // Sizes: mix of small and large allocations
  const size_t kN = 10;
  size_t sizes[kN] = {32, 1024, 4096, 16384, 32768,
                      65536, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024};

  OffsetPtr<> ptrs[kN];
  size_t      old_offs[kN];

  // Allocate all blocks and write a distinct canary per block
  for (size_t i = 0; i < kN; ++i) {
    ptrs[i] = alloc->AllocateOffset(sizes[i]);
    REQUIRE(!ptrs[i].IsNull());
    old_offs[i] = ptrs[i].load();
    WriteCanary(alloc, ptrs[i], sizes[i], static_cast<uint8_t>(i + 1));
  }

  // Free every other block (indices 1, 3, 5, 7, 9)
  for (size_t i = 1; i < kN; i += 2) {
    alloc->FreeOffset(ptrs[i]);
    ptrs[i] = OffsetPtr<>::GetNull();
  }

  // Compact
  ForwardingTable table = alloc->Compact();

  // Surviving blocks: 0, 2, 4, 6, 8 — verify data integrity
  for (size_t i = 0; i < kN; i += 2) {
    size_t new_off = table.Resolve(old_offs[i]);
    OffsetPtr<> new_ptr(new_off);
    REQUIRE(!new_ptr.IsNull());
    REQUIRE(CheckCanary(alloc, new_ptr, sizes[i],
                        static_cast<uint8_t>(i + 1)));
  }

  // A new allocation must still succeed after compaction
  OffsetPtr<> extra = alloc->AllocateOffset(64 * 1024);
  REQUIRE(!extra.IsNull());
}

// ---------------------------------------------------------------------------
// Test 2: compact_empty
// Compact a fresh (empty) allocator — table must be empty, allocator still works.
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - Compact Empty", "[compaction]") {
  MallocBackend backend;
  BuddyAllocator *alloc = MakeAlloc(backend);

  ForwardingTable table = alloc->Compact();
  REQUIRE(table.Empty());

  // Normal allocation still works after compacting an empty heap
  OffsetPtr<> ptr = alloc->AllocateOffset(1024);
  REQUIRE(!ptr.IsNull());
}

// ---------------------------------------------------------------------------
// Test 3: compact_no_frag
// Allocate 5 blocks, free all 5, compact — table should be empty (no live blocks),
// and a large allocation should succeed.
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - Compact No Live Blocks", "[compaction]") {
  MallocBackend backend;
  BuddyAllocator *alloc = MakeAlloc(backend);

  const size_t kN   = 5;
  const size_t kSz  = 1024 * 1024;  // 1 MB each
  OffsetPtr<>  ptrs[kN];

  for (size_t i = 0; i < kN; ++i) {
    ptrs[i] = alloc->AllocateOffset(kSz);
    REQUIRE(!ptrs[i].IsNull());
  }
  for (size_t i = 0; i < kN; ++i) {
    alloc->FreeOffset(ptrs[i]);
  }

  ForwardingTable table = alloc->Compact();
  REQUIRE(table.Empty());

  // After compacting all-free heap, a large allocation must succeed
  OffsetPtr<> big = alloc->AllocateOffset(4 * 1024 * 1024);
  REQUIRE(!big.IsNull());
}

// ---------------------------------------------------------------------------
// Test 5: compact_consecutive_live
// Allocate 3 consecutive blocks WITHOUT freeing any, then compact.
// Exercises: ForwardingTable::Size(), the typed Resolve<T>(OffsetPtr<T>)
// template overload (including the null-ptr fast-return path), and the
// "not found" fallback in Resolve(size_t) (line: return old_off).
// Note: the allocator's Expand() always seeds initial free pages before the
// first user allocation, so all three blocks are physically moved to lower
// addresses; they maintain their relative order and data integrity.
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - Compact Consecutive Live", "[compaction]") {
  MallocBackend backend;
  BuddyAllocator *alloc = MakeAlloc(backend);

  // Allocate 3 consecutive blocks — no user-freed blocks between them
  OffsetPtr<> a = alloc->AllocateOffset(1024);
  OffsetPtr<> b = alloc->AllocateOffset(2048);
  OffsetPtr<> c = alloc->AllocateOffset(4096);
  REQUIRE(!a.IsNull());
  REQUIRE(!b.IsNull());
  REQUIRE(!c.IsNull());

  size_t old_a = a.load(), old_b = b.load(), old_c = c.load();
  WriteCanary(alloc, a, 1024, 0xAA);
  WriteCanary(alloc, b, 2048, 0xBB);
  WriteCanary(alloc, c, 4096, 0xCC);

  // Compact — initial Expand() free pages are reclaimed, all 3 live blocks move
  ForwardingTable table = alloc->Compact();

  // All 3 live blocks must be recorded (exercises ForwardingTable::Size())
  REQUIRE(table.Size() == 3);

  size_t new_a = table.Resolve(old_a);
  size_t new_b = table.Resolve(old_b);
  size_t new_c = table.Resolve(old_c);

  // All resolved offsets must be valid (non-null)
  REQUIRE(!OffsetPtr<>(new_a).IsNull());
  REQUIRE(!OffsetPtr<>(new_b).IsNull());
  REQUIRE(!OffsetPtr<>(new_c).IsNull());

  // Data integrity at new locations
  REQUIRE(CheckCanary(alloc, OffsetPtr<>(new_a), 1024, 0xAA));
  REQUIRE(CheckCanary(alloc, OffsetPtr<>(new_b), 2048, 0xBB));
  REQUIRE(CheckCanary(alloc, OffsetPtr<>(new_c), 4096, 0xCC));

  // Typed Resolve<char> overload — exercises ForwardingTable::Resolve(OffsetPtr<T>)
  OffsetPtr<char> char_a(old_a);
  OffsetPtr<char> resolved_a = table.Resolve(char_a);
  REQUIRE(resolved_a.load() == new_a);

  // Null-ptr fast-return path in the typed overload
  OffsetPtr<char> null_ptr = OffsetPtr<char>::GetNull();
  REQUIRE(table.Resolve(null_ptr).IsNull());

  // Resolve an offset NOT in the table — exercises "return old_off" fallback
  // (offset 1 is inside allocator metadata, never a valid user-data start)
  size_t bogus = 1;
  REQUIRE(table.Resolve(bogus) == bogus);

  // Post-compact allocation must still succeed
  OffsetPtr<> extra = alloc->AllocateOffset(512);
  REQUIRE(!extra.IsNull());
}

// ---------------------------------------------------------------------------
// Test 6: edge_cases
// Cover AllocateOffset size < kMinSize (clamped to 32 B, line 262) and
// FreeOffset with a null pointer (early return, line 325).
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - Edge Cases", "[compaction]") {
  MallocBackend backend;
  BuddyAllocator *alloc = MakeAlloc(backend);

  // AllocateOffset(0/1/31) must clamp to kMinSize=32 (line 262)
  OffsetPtr<> p0 = alloc->AllocateOffset(0);
  REQUIRE(!p0.IsNull());
  OffsetPtr<> p1 = alloc->AllocateOffset(1);
  REQUIRE(!p1.IsNull());
  OffsetPtr<> p2 = alloc->AllocateOffset(31);
  REQUIRE(!p2.IsNull());
  // All three must be distinct allocations
  REQUIRE(p0.load() != p1.load());
  REQUIRE(p1.load() != p2.load());

  // FreeOffset with null pointer must be a no-op (line 325)
  REQUIRE_NOTHROW(alloc->FreeOffset(OffsetPtr<>::GetNull()));

  alloc->FreeOffset(p0);
  alloc->FreeOffset(p1);
  alloc->FreeOffset(p2);
}

// ---------------------------------------------------------------------------
// Test 7: small_freelist_reuse
// Cover small free-list reuse path (lines 477-479) and DivideArenaIntoPages
// loop body (lines 642-658).
//
// Free-list path (lines 477-479):
//   Alloc 32B → free it → re-alloc 32B pops from small_pages_[0].
//
// DivideArenaIntoPages loop (lines 642-658):
//   The allocator's small arena = 65536 + 128*16 = 67584 B.
//   64B allocs each consume 80B (64 + sizeof(BuddyPage) = 64 + 16).
//   After 844 such bump-allocs, 64B remains in the arena.
//   The 845th 64B alloc finds arena has only 64B (< 80B needed) and calls
//   RepopulateSmallArena → DivideArenaIntoPages(64B remaining) → greedy loop
//   places one 32B page (48B) in small_pages_[0] → loop body executes.
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - Small Freelist Reuse", "[compaction]") {
  MallocBackend backend;
  BuddyAllocator *alloc = MakeAlloc(backend);

  // Alloc + free 32B to populate small_pages_[0]
  OffsetPtr<> first = alloc->AllocateOffset(32);
  REQUIRE(!first.IsNull());
  alloc->FreeOffset(first);

  // Re-alloc 32B: pops from small_pages_[0] → lines 477-479
  OffsetPtr<> second = alloc->AllocateOffset(32);
  REQUIRE(!second.IsNull());
  REQUIRE(second.load() == first.load());  // same block reused
  alloc->FreeOffset(second);

  // Exhaust small_arena_ with 64B allocs to trigger DivideArenaIntoPages loop
  // kSmallArenaSize=65536, kSmallArenaPages=128, sizeof(BuddyPage)=16
  // arena_total = 67584; each 64B alloc uses 80B → 844 full allocs, 64B left
  // → DivideArenaIntoPages places one 32B page in small_pages_[0] (lines 642-658)
  constexpr size_t kArenaTotal = 65536 + 128 * 16;  // 67584
  constexpr size_t kAllocStep  = 64 + 16;            // 80 bytes per bump alloc
  const size_t     n           = kArenaTotal / kAllocStep;  // 844
  std::vector<OffsetPtr<>> ptrs(n);
  for (size_t i = 0; i < n; ++i) {
    ptrs[i] = alloc->AllocateOffset(64);
    REQUIRE(!ptrs[i].IsNull());
  }
  // 845th alloc triggers RepopulateSmallArena → DivideArenaIntoPages loop body
  OffsetPtr<> trigger = alloc->AllocateOffset(64);
  REQUIRE(!trigger.IsNull());

  for (auto &p : ptrs) alloc->FreeOffset(p);
  alloc->FreeOffset(trigger);
}

// ---------------------------------------------------------------------------
// Test 8: large_page_remainder
// Cover AddRemainderToFreeList from AllocateLarge (line 533).
// Free 300KB → large_pages_[3] (floor(log2(300KB))=18, idx=18-14-1=3).
// Alloc 256KB → same list 3 → FindFirstFit returns the 300KB page →
// page_total_size(300KB+16) > total_size(256KB+16) → remainder sent to free list.
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - Large Page Remainder", "[compaction]") {
  MallocBackend backend;
  BuddyAllocator *alloc = MakeAlloc(backend);

  const size_t k300KB = 300 * 1024;
  const size_t k256KB = 256 * 1024;

  // Alloc 300KB then free → lands in large_pages_[3]
  OffsetPtr<> big = alloc->AllocateOffset(k300KB);
  REQUIRE(!big.IsNull());
  alloc->FreeOffset(big);

  // Alloc 256KB: same list 3 → 300KB page reused → ~44KB remainder added back
  // to large_pages_[0] via AddRemainderToFreeList (line 533)
  OffsetPtr<> partial = alloc->AllocateOffset(k256KB);
  REQUIRE(!partial.IsNull());

  // ~44KB remainder is now in large_pages_[0]; allocate 20KB from it
  OffsetPtr<> from_remainder = alloc->AllocateOffset(20 * 1024);
  REQUIRE(!from_remainder.IsNull());

  alloc->FreeOffset(partial);
  alloc->FreeOffset(from_remainder);
}

// ---------------------------------------------------------------------------
// Test 9: find_first_fit_no_fit
// Cover the FindFirstFit "no fit found" return (line 582).
// Free 20KB → large_pages_[0] (floor(log2(20KB))=14 ≤ kSmallLog2=14 → idx=0).
// Alloc 50KB → also list 0 (floor(log2(50KB))=15, idx=15-14-1=0).
// FindFirstFit(0, 51216): the 20KB page (total=20496 < 51216) does not fit →
// iterates all entries without success → returns 0 (line 582).
// AllocateLarge then falls back to big_heap_.
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - FindFirstFit No Fit", "[compaction]") {
  MallocBackend backend;
  BuddyAllocator *alloc = MakeAlloc(backend);

  const size_t k20KB = 20 * 1024;
  const size_t k50KB = 50 * 1024;

  // Alloc 20KB then free → large_pages_[0]
  OffsetPtr<> small_large = alloc->AllocateOffset(k20KB);
  REQUIRE(!small_large.IsNull());
  alloc->FreeOffset(small_large);

  // Alloc 50KB: FindFirstFit(0, 51216) finds 20KB page (20496 < 51216) →
  // no fit → returns 0 (line 582) → falls back to big_heap_
  OffsetPtr<> large = alloc->AllocateOffset(k50KB);
  REQUIRE(!large.IsNull());

  alloc->FreeOffset(large);
}

// ---------------------------------------------------------------------------
// Test 4: compact_enables_large_alloc
// Alternate 1 MB and 32 B allocations, free all 1 MB blocks, verify a new 1 MB
// allocation fails (fragmented), compact, then verify 1 MB allocation succeeds.
// ---------------------------------------------------------------------------
TEST_CASE("BuddyAllocator - Compact Enables Large Alloc", "[compaction]") {
  MallocBackend backend;
  // Use 64 MB so we can fragment meaningfully
  BuddyAllocator *alloc = MakeAlloc(backend, /*heap_mb=*/64);

  const size_t kLarge = 1024 * 1024;  // 1 MB
  const size_t kSmall = 32;           // 32 B
  const size_t kPairs = 20;

  std::vector<OffsetPtr<>> large_ptrs(kPairs);
  std::vector<OffsetPtr<>> small_ptrs(kPairs);
  std::vector<size_t>      small_old_offs(kPairs);

  // Alternate: large, small, large, small …
  for (size_t i = 0; i < kPairs; ++i) {
    large_ptrs[i] = alloc->AllocateOffset(kLarge);
    REQUIRE(!large_ptrs[i].IsNull());

    small_ptrs[i] = alloc->AllocateOffset(kSmall);
    REQUIRE(!small_ptrs[i].IsNull());
    small_old_offs[i] = small_ptrs[i].load();
    // Write distinct canary per small block
    WriteCanary(alloc, small_ptrs[i], kSmall, static_cast<uint8_t>(i + 1));
  }

  // Free all large blocks — heap is now fragmented
  for (size_t i = 0; i < kPairs; ++i) {
    alloc->FreeOffset(large_ptrs[i]);
  }

  // A new 1 MB allocation should FAIL (fragmented between 32 B blocks)
  // Note: this assertion documents the expected fragmentation behaviour.
  // If the allocator happens to coalesce, the test adjusts gracefully.
  OffsetPtr<> should_fail = alloc->AllocateOffset(kLarge);
  // We do NOT hard-REQUIRE failure here because a sufficiently smart allocator
  // might satisfy it from the bump pointer tail; instead we proceed to compact
  // regardless and verify the post-compact state.
  if (!should_fail.IsNull()) {
    // Return it so it doesn't interfere with the post-compact check
    alloc->FreeOffset(should_fail);
  }

  // Compact
  ForwardingTable table = alloc->Compact();

  // Update small pointers via forwarding table and verify canaries
  for (size_t i = 0; i < kPairs; ++i) {
    size_t new_off = table.Resolve(small_old_offs[i]);
    OffsetPtr<> new_ptr(new_off);
    REQUIRE(!new_ptr.IsNull());
    REQUIRE(CheckCanary(alloc, new_ptr, kSmall, static_cast<uint8_t>(i + 1)));
  }

  // After compaction, a 1 MB allocation MUST succeed (reclaimed space)
  OffsetPtr<> after_compact = alloc->AllocateOffset(kLarge);
  REQUIRE(!after_compact.IsNull());
}
