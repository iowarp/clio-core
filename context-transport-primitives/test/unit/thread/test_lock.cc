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

#include "basic_test.h"
#include "clio_ctp/thread/lock.h"
#include "omp.h"

using ctp::CvRwLock;
using ctp::Mutex;
using ctp::RwLock;

void MutexTest(int nthreads) {
  size_t loop_count = 10000;
  size_t count = 0;
  Mutex lock;

  omp_set_dynamic(0);
#pragma omp parallel shared(lock) num_threads(nthreads)
  {
    // Support parallel write
#pragma omp barrier
    for (size_t i = 0; i < loop_count; ++i) {
      lock.Lock(i);
      count += 1;
      lock.Unlock();
    }
#pragma omp barrier
    REQUIRE(count == loop_count * nthreads);
#pragma omp barrier
  }
}

void RwLockTest(int producers, int consumers, size_t loop_count) {
  size_t nthreads = producers + consumers;
  size_t count = 0;
  RwLock lock;

  omp_set_dynamic(0);
#pragma omp parallel shared(lock, nthreads, producers, consumers, loop_count, \
                                count) num_threads(nthreads)
  {  // NOLINT
    int tid = omp_get_thread_num();

#pragma omp barrier
    size_t total_size = producers * loop_count;
    if (tid < consumers) {
      // The left 2 threads will be readers
      lock.ReadLock(tid);
      for (size_t i = 0; i < loop_count; ++i) {
        REQUIRE(count <= total_size);
      }
      lock.ReadUnlock();
    } else {
      // The right 4 threads will be writers
      lock.WriteLock(tid);
      for (size_t i = 0; i < loop_count; ++i) {
        count += 1;
      }
      lock.WriteUnlock();
    }

#pragma omp barrier
    REQUIRE(count == total_size);
  }
}

TEST_CASE("Mutex") { MutexTest(8); }

TEST_CASE("RwLock") {
  RwLockTest(8, 0, 1000000);
  RwLockTest(7, 1, 1000000);
  RwLockTest(4, 4, 1000000);
}

void CvRwLockTest(int producers, int consumers, size_t loop_count) {
  size_t nthreads = producers + consumers;
  size_t count = 0;
  CvRwLock lock;

  omp_set_dynamic(0);
#pragma omp parallel shared(lock, nthreads, producers, consumers, loop_count, \
                                count) num_threads(nthreads)
  {  // NOLINT
    int tid = omp_get_thread_num();

#pragma omp barrier
    size_t total_size = producers * loop_count;
    if (tid < consumers) {
      // Readers observe the shared counter under the (parallel) read lock; it
      // must never exceed the final total while writers are still running.
      lock.ReadLock();
      for (size_t i = 0; i < loop_count; ++i) {
        REQUIRE(count <= total_size);
      }
      lock.ReadUnlock();
    } else {
      // Writers increment under the exclusive lock; mutual exclusion means no
      // increment is lost.
      lock.WriteLock();
      for (size_t i = 0; i < loop_count; ++i) {
        count += 1;
      }
      lock.WriteUnlock();
    }

#pragma omp barrier
    REQUIRE(count == total_size);
  }
}

TEST_CASE("CvRwLock") {
  CvRwLockTest(8, 0, 1000000);
  CvRwLockTest(7, 1, 1000000);
  CvRwLockTest(4, 4, 1000000);
}

#if CTP_ENABLE_THALLIUM
TEST_CASE("AbtThread") {
  ctp::thread::Argobots argobots;
  ctp::thread::ThreadGroup group = argobots.CreateThreadGroup({});
  ctp::thread::Thread thread = argobots.Spawn(
      group,
      [](int tid) { std::cout << "Hello, world! (abt) " << tid << std::endl; },
      1);
  argobots.Join(thread);
}
#endif

#ifdef CTP_ENABLE_PTHREADS
TEST_CASE("Pthread") {
  ctp::thread::Pthread pthread;
  ctp::thread::ThreadGroup group = pthread.CreateThreadGroup({});
  ctp::thread::Thread thread = pthread.Spawn(
      group,
      [](int tid) {
        std::cout << "Hello, world! (pthread) " << tid << std::endl;
      },
      1);
  pthread.Join(thread);
}
#endif

TEST_CASE("StdThread") {
  ctp::thread::StdThread std_thread;
  ctp::thread::ThreadGroup group = std_thread.CreateThreadGroup({});
  ctp::thread::Thread thread = std_thread.Spawn(
      group,
      [](int tid) { std::cout << "Hello, world! (std) " << tid << std::endl; },
      1);
  std_thread.Join(thread);
}
