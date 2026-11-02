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
#include "allocator_test.h"
#include "clio_ctp/memory/backend/malloc_backend.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"

using ctp::testing::AllocatorTest;

/**
 * Helper function to create a MallocBackend and ArenaAllocator<false>
 * Returns the allocator pointer (caller must manage backend lifetime)
 */
ctp::ipc::ArenaAllocator<false>* CreateArenaAllocator(ctp::ipc::MallocBackend &backend) {
  // Initialize backend with space for allocator + heap
  size_t heap_size = 256 * 1024 * 1024;  // 256 MB heap
  size_t alloc_size = sizeof(ctp::ipc::ArenaAllocator<false>);
  backend.shm_init(ctp::ipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  // Construct and initialize allocator using MakeAlloc
  auto *alloc = backend.MakeAlloc<ctp::ipc::ArenaAllocator<false>>(heap_size);

  return alloc;
}

TEST_CASE("SubAllocator - Basic Creation and Destruction", "[SubAllocator]") {
  ctp::ipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  SECTION("Create and destroy a single sub-allocator") {
    // Create a sub-allocator with 64 MB
    size_t sub_alloc_size = 64 * 1024 * 1024;
    auto sub_alloc = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
        sub_alloc_size);

    REQUIRE(!sub_alloc.IsNull());
    REQUIRE(sub_alloc.ptr_->GetId() == parent_alloc->GetId());

    // Free the sub-allocator
    parent_alloc->FreeSubAllocator(sub_alloc);
  }

  SECTION("Create multiple sub-allocators with different IDs") {
    size_t sub_alloc_size = 32 * 1024 * 1024;

    auto sub_alloc1 = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
        sub_alloc_size);
    auto sub_alloc2 = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
        sub_alloc_size);
    auto sub_alloc3 = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
        sub_alloc_size);

    REQUIRE(!sub_alloc1.IsNull());
    REQUIRE(!sub_alloc2.IsNull());
    REQUIRE(!sub_alloc3.IsNull());


    // Free all sub-allocators
    parent_alloc->FreeSubAllocator(sub_alloc1);
    parent_alloc->FreeSubAllocator(sub_alloc2);
    parent_alloc->FreeSubAllocator(sub_alloc3);
  }

}

TEST_CASE("SubAllocator - Allocations within SubAllocator", "[SubAllocator]") {
  ctp::ipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  // Create a sub-allocator with 64 MB
  size_t sub_alloc_size = 64 * 1024 * 1024;
  auto sub_alloc = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
      sub_alloc_size);

  REQUIRE(!sub_alloc.IsNull());

  SECTION("Allocate and free immediately") {


    for (size_t i = 0; i < 1000; ++i) {
      auto ptr = sub_alloc.ptr_->template Allocate<void>(1024);
      REQUIRE_FALSE(ptr.IsNull());
      sub_alloc.ptr_->Free(ptr);
    }
  }

  SECTION("Batch allocations") {

    std::vector<ctp::ipc::FullPtr<void>> ptrs;

    // Allocate batch
    for (size_t i = 0; i < 100; ++i) {
      auto ptr = sub_alloc.ptr_->template Allocate<void>(4096);
      REQUIRE_FALSE(ptr.IsNull());
      ptrs.push_back(ptr);
    }

    // Free batch
    for (auto &ptr : ptrs) {
      sub_alloc.ptr_->Free(ptr);
    }
  }

  // Free the sub-allocator
  parent_alloc->FreeSubAllocator(sub_alloc);

}

TEST_CASE("SubAllocator - Random Allocation Test", "[SubAllocator]") {
  ctp::ipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  // Create a sub-allocator with 64 MB
  size_t sub_alloc_size = 64 * 1024 * 1024;
  auto sub_alloc = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
      sub_alloc_size);

  REQUIRE(!sub_alloc.IsNull());

  // Use the AllocatorTest framework to run random tests
  AllocatorTest<ctp::ipc::ArenaAllocator<false>> tester(sub_alloc.ptr_);

  SECTION("1 iteration of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(1));
  }

  // Free the sub-allocator
  parent_alloc->FreeSubAllocator(sub_alloc);

}

TEST_CASE("SubAllocator - Multiple SubAllocators with Random Tests", "[SubAllocator]") {
  ctp::ipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  // Create 3 sub-allocators, each with 32 MB
  size_t sub_alloc_size = 32 * 1024 * 1024;
  auto sub_alloc1 = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
      sub_alloc_size);
  auto sub_alloc2 = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
      sub_alloc_size);
  auto sub_alloc3 = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
      sub_alloc_size);

  REQUIRE(!sub_alloc1.IsNull());
  REQUIRE(!sub_alloc2.IsNull());
  REQUIRE(!sub_alloc3.IsNull());

  SECTION("Run random tests on all three sub-allocators") {
    AllocatorTest<ctp::ipc::ArenaAllocator<false>> tester1(sub_alloc1.ptr_);
    AllocatorTest<ctp::ipc::ArenaAllocator<false>> tester2(sub_alloc2.ptr_);
    AllocatorTest<ctp::ipc::ArenaAllocator<false>> tester3(sub_alloc3.ptr_);

    REQUIRE_NOTHROW(tester1.TestRandomAllocation(1));
    REQUIRE_NOTHROW(tester2.TestRandomAllocation(1));
    REQUIRE_NOTHROW(tester3.TestRandomAllocation(1));
  }

  // Free all sub-allocators
  parent_alloc->FreeSubAllocator(sub_alloc1);
  parent_alloc->FreeSubAllocator(sub_alloc2);
  parent_alloc->FreeSubAllocator(sub_alloc3);

}

TEST_CASE("SubAllocator - Nested SubAllocators", "[SubAllocator][nested]") {
  ctp::ipc::MallocBackend backend;
  auto *parent_alloc = CreateArenaAllocator(backend);

  // Create a sub-allocator from parent (128 MB)
  size_t sub_alloc1_size = 128 * 1024 * 1024;
  auto sub_alloc1 = parent_alloc->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
      sub_alloc1_size);

  REQUIRE(!sub_alloc1.IsNull());

  // Create a nested sub-allocator from the first sub-allocator (32 MB)
  size_t sub_alloc2_size = 32 * 1024 * 1024;
  auto sub_alloc2 = sub_alloc1.ptr_->CreateSubAllocator<ctp::ipc::ArenaAllocator<false>>(
      sub_alloc2_size);

  REQUIRE(!sub_alloc2.IsNull());

  // Test allocations in the nested sub-allocator
  AllocatorTest<ctp::ipc::ArenaAllocator<false>> tester(sub_alloc2.ptr_);
  REQUIRE_NOTHROW(tester.TestRandomAllocation(1));

  // Free nested sub-allocator first, then parent sub-allocator
  sub_alloc1.ptr_->FreeSubAllocator(sub_alloc2);
  parent_alloc->FreeSubAllocator(sub_alloc1);

}
