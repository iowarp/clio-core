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
#include <thread>
#include <vector>
#include <cstring>
#include <atomic>
#include "clio_ctp/memory/allocator/thread_allocator.h"
#include "clio_ctp/memory/backend/posix_mmap.h"

using ctp::ipc::PartitionedAllocator;

static PartitionedAllocator* MakeThreadAllocator(ctp::ipc::PosixMmap &backend,
                                                  size_t heap_size,
                                                  int max_threads = 32) {
  size_t total_size = sizeof(PartitionedAllocator) + heap_size;
  backend.shm_init(ctp::ipc::MemoryBackendId(0, 0), total_size);
  auto *alloc = backend.Cast<PartitionedAllocator>();
  new (alloc) PartitionedAllocator();
  size_t thread_unit = (total_size - sizeof(PartitionedAllocator)) / max_threads;
  alloc->shm_init(backend, 0, max_threads, thread_unit);
  return alloc;
}

TEST_CASE("ThreadAllocator - AllocFreeImmediate", "[ThreadAllocator]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 64 * 1024 * 1024;  // 64 MB
  auto *alloc = MakeThreadAllocator(backend, heap_size);

  SECTION("Single thread alloc/free cycles (1KB)") {
    for (int i = 0; i < 1000; ++i) {
      auto ptr = alloc->Allocate<char>(1024);
      REQUIRE(!ptr.IsNull());
      std::memset(ptr.ptr_, static_cast<unsigned char>(i & 0xFF), 1024);
      alloc->Free(ptr);
    }
  }

  SECTION("Single thread alloc/free cycles (64KB)") {
    for (int i = 0; i < 100; ++i) {
      auto ptr = alloc->Allocate<char>(64 * 1024);
      REQUIRE(!ptr.IsNull());
      std::memset(ptr.ptr_, static_cast<unsigned char>(i & 0xFF), 64 * 1024);
      alloc->Free(ptr);
    }
  }
}

TEST_CASE("ThreadAllocator - AllocFreeBatch", "[ThreadAllocator]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 128 * 1024 * 1024;  // 128 MB
  auto *alloc = MakeThreadAllocator(backend, heap_size);

  SECTION("Batch of 100 allocations of 4KB") {
    for (int iter = 0; iter < 10; ++iter) {
      std::vector<ctp::ipc::FullPtr<char>> ptrs;
      ptrs.reserve(100);

      for (int i = 0; i < 100; ++i) {
        auto ptr = alloc->Allocate<char>(4096);
        REQUIRE(!ptr.IsNull());
        std::memset(ptr.ptr_, static_cast<unsigned char>(i & 0xFF), 4096);
        ptrs.push_back(ptr);
      }

      for (auto &ptr : ptrs) {
        alloc->Free(ptr);
      }
    }
  }
}

TEST_CASE("ThreadAllocator - MultiThreadedAlloc", "[ThreadAllocator][multithread]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 256 * 1024 * 1024;  // 256 MB
  int num_threads = 8;
  auto *alloc = MakeThreadAllocator(backend, heap_size, num_threads);
  // Access the core allocator for the 2-arg AllocateOffset
  auto *core = static_cast<ctp::ipc::_PartitionedAllocator*>(alloc);

  std::atomic<int> tid_counter{0};
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([core, &tid_counter]() {
      int tid = tid_counter.fetch_add(1);

      for (int iter = 0; iter < 50; ++iter) {
        std::vector<ctp::ipc::OffsetPtr<>> ptrs;
        ptrs.reserve(20);

        for (int i = 0; i < 20; ++i) {
          auto ptr = core->AllocateOffset(1024, tid);
          if (ptr.IsNull()) break;

          // Write pattern to verify no corruption
          char *data = core->GetBackendData() + ptr.load();
          std::memset(data, static_cast<unsigned char>((tid * 100 + i) & 0xFF), 1024);
          ptrs.push_back(ptr);
        }

        for (auto &ptr : ptrs) {
          core->FreeOffset(ptr);
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

TEST_CASE("ThreadAllocator - LazyInit", "[ThreadAllocator]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 64 * 1024 * 1024;  // 64 MB
  int max_threads = 8;
  auto *alloc = MakeThreadAllocator(backend, heap_size, max_threads);

  auto *core = static_cast<ctp::ipc::_PartitionedAllocator*>(alloc);

  // Initially no thread blocks should be initialized
  for (int i = 0; i < max_threads; ++i) {
    REQUIRE(core->GetThreadBlock(i)->initialized_.load() == 0);
  }

  // Allocate using tid=0 — only thread 0 should be initialized
  auto ptr = core->AllocateOffset(256, 0);
  REQUIRE(!ptr.IsNull());
  REQUIRE(core->GetThreadBlock(0)->initialized_.load() == 1);

  // Other threads still uninitialized
  for (int i = 1; i < max_threads; ++i) {
    REQUIRE(core->GetThreadBlock(i)->initialized_.load() == 0);
  }

  // Allocate using tid=3
  auto ptr2 = core->AllocateOffset(256, 3);
  REQUIRE(!ptr2.IsNull());
  REQUIRE(core->GetThreadBlock(3)->initialized_.load() == 1);

  // tid=1, 2, 4..7 still uninitialized
  REQUIRE(core->GetThreadBlock(1)->initialized_.load() == 0);
  REQUIRE(core->GetThreadBlock(2)->initialized_.load() == 0);

  core->FreeOffset(ptr);
  core->FreeOffset(ptr2);
}
