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

TEST_CASE("ArenaAllocator<false> - Allocate and Free Immediate", "[ArenaAllocator<false>]") {
  ctp::ipc::MallocBackend backend;
  size_t heap_size = 128 * 1024 * 1024;  // 128 MB heap
  size_t alloc_size = sizeof(ctp::ipc::ArenaAllocator<false>);
  backend.shm_init(ctp::ipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<ctp::ipc::ArenaAllocator<false>>();

  AllocatorTest<ctp::ipc::ArenaAllocator<false>> tester(alloc);

  SECTION("Small allocations (1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(100, 1024));
  }
  
}

