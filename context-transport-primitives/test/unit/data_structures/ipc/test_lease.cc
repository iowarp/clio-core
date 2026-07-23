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

#include "clio_ctp/memory/smart_ptr/lease.h"

// The cross-process half of this suite needs fork()/mmap()/waitpid(), which
// are POSIX-only. On Windows the single-process Lease/SeqRead cases still run;
// the kill-readers torture test is compiled out rather than faked, because a
// weaker stand-in would give false confidence in the one property that matters.
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#endif
#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace ctp::ipc;

namespace {

constexpr size_t kFields = 16;

/**
 * A record with a checkable invariant: every field must equal `value`. A torn
 * read shows up as fields from two different generations.
 */
struct Record : public Leaseable {
  ctp::min_u64 value;
  ctp::min_u64 fields[kFields];
};

struct Shared {
  Record rec;
  ctp::ipc::atomic<ctp::min_u64> torn_count;   /**< MUST stay 0 */
  ctp::ipc::atomic<ctp::min_u64> read_ok;
  ctp::ipc::atomic<ctp::min_u64> read_retry;
  ctp::ipc::atomic<ctp::min_u64> writes;
  ctp::ipc::atomic<ctp::min_u32> stop;
};

Shared *MapShared() {
#ifndef _WIN32
  void *m = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) {
    return nullptr;
  }
#else
  // Single-process cases only need shared-nothing memory; the cross-process
  // cases are compiled out on Windows.
  void *m = std::malloc(sizeof(Shared));
  if (m == nullptr) {
    return nullptr;
  }
#endif
  auto *s = new (m) Shared();
  s->rec.InitLeaseable();
  s->rec.value = 0;
  for (size_t i = 0; i < kFields; ++i) {
    s->rec.fields[i] = 0;
  }
  s->torn_count = 0;
  s->read_ok = 0;
  s->read_retry = 0;
  s->writes = 0;
  s->stop = 0;
  return s;
}

#ifndef _WIN32
/** Reader loop: seqlock-read the record and check the invariant. */
void ReaderLoop(Shared *s) {
  ctp::min_u64 snap[kFields];
  while (s->stop.load() == 0) {
    bool ok = SeqRead(s->rec, [&]() {
      for (size_t i = 0; i < kFields; ++i) {
        snap[i] = s->rec.fields[i];
      }
    });
    if (!ok) {
      s->read_retry.fetch_add(1);
      continue;
    }
    for (size_t i = 1; i < kFields; ++i) {
      if (snap[i] != snap[0]) {
        s->torn_count.fetch_add(1);
        break;
      }
    }
    s->read_ok.fetch_add(1);
  }
}
#endif  // !_WIN32

}  // namespace

#ifdef _WIN32
static void UnmapShared(Shared *s) { std::free(s); }
#else
static void UnmapShared(Shared *s) { munmap(s, sizeof(Shared)); }
#endif

TEST_CASE("Lease: basic acquire and release", "[lease]") {
  auto *s = MapShared();
  REQUIRE(s != nullptr);

  {
    Lease<Record> l(&s->rec);
    REQUIRE(l.Owns());
    REQUIRE(l.get() == &s->rec);
    REQUIRE(static_cast<bool>(l));
  }
  // Released on scope exit -- a fresh lease must succeed.
  {
    Lease<Record> l2 = Lease<Record>::TryAcquire(&s->rec);
    REQUIRE(l2.Owns());
  }
  // NOTE: every lease must be destroyed BEFORE the mapping goes away. A lease
  // outliving its segment unlocks through a dangling pointer -- which is
  // exactly what a client must avoid when the runtime tears the metadata
  // segment down underneath it.
  UnmapShared(s);
}

TEST_CASE("Lease: TryAcquire fails while held", "[lease]") {
  auto *s = MapShared();
  REQUIRE(s != nullptr);

  Lease<Record> a(&s->rec);
  REQUIRE(a.Owns());

  Lease<Record> b = Lease<Record>::TryAcquire(&s->rec);
  REQUIRE_FALSE(b.Owns());
  // A non-owning lease must not hand out the object -- a missed Owns() check
  // should fail loudly, not silently read unprotected memory.
  REQUIRE(b.get() == nullptr);

  a.Release();
  {
    Lease<Record> c = Lease<Record>::TryAcquire(&s->rec);
    REQUIRE(c.Owns());
  }

  UnmapShared(s);
}

TEST_CASE("Lease: move transfers ownership exactly once", "[lease]") {
  auto *s = MapShared();
  REQUIRE(s != nullptr);

  {
    Lease<Record> a(&s->rec);
    REQUIRE(a.Owns());
    Lease<Record> b(std::move(a));
    REQUIRE(b.Owns());
    REQUIRE_FALSE(a.Owns());  // NOLINT(bugprone-use-after-move)
  }
  // If the moved-from lease had also released, this would still pass; the real
  // check is that the lock is free exactly once and is acquirable now.
  {
    Lease<Record> c = Lease<Record>::TryAcquire(&s->rec);
    REQUIRE(c.Owns());
  }

  UnmapShared(s);
}

TEST_CASE("SeqRead: rejects a write in progress", "[lease]") {
  auto *s = MapShared();
  REQUIRE(s != nullptr);

  s->rec.BeginUpdate();  // leave it odd
  REQUIRE(s->rec.IsUpdating());
  bool ran = false;
  bool ok = SeqRead(s->rec, [&]() { ran = true; }, 4);
  REQUIRE_FALSE(ok);
  REQUIRE_FALSE(ran);  // never even attempted while odd

  s->rec.EndUpdate();
  ok = SeqRead(s->rec, [&]() { ran = true; }, 4);
  REQUIRE(ok);
  REQUIRE(ran);

  UnmapShared(s);
}

// ===========================================================================
// The Phase 2 exit criterion (design §6): concurrent cross-process readers
// that are SIGKILLed at random mid-read, against a continuously mutating
// writer. Asserts the two properties the whole design rests on:
//   1. No reader EVER observes a torn record.
//   2. Killing readers -- including ones holding leases -- never wedges the
//      writer, and abandoned leases are reclaimed.
// ===========================================================================
#ifndef _WIN32
TEST_CASE("Lease torture: readers killed at random never tear or wedge",
          "[lease][torture]") {
  auto *s = MapShared();
  REQUIRE(s != nullptr);

  constexpr int kReaders = 6;
  std::vector<pid_t> kids;

  // Plain seqlock readers.
  for (int i = 0; i < kReaders; ++i) {
    pid_t p = fork();
    if (p == 0) {
      ReaderLoop(s);
      _exit(0);
    }
    kids.push_back(p);
  }

  // Two lease-holding readers that will be killed WHILE HOLDING the lease.
  // These are the ones that would deadlock a non-reclaimable lock.
  std::vector<pid_t> lease_kids;
  for (int i = 0; i < 2; ++i) {
    pid_t p = fork();
    if (p == 0) {
      Lease<Record> l(&s->rec);
      while (true) {
        pause();  // hold forever; parent kills us
      }
      _exit(0);
    }
    lease_kids.push_back(p);
  }

  // Writer: mutate all fields inside an update window, with a deliberate gap
  // between field writes so an unsynchronized reader WOULD see a tear.
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  unsigned rng = 12345;
  size_t kill_idx = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    s->rec.BeginUpdate();
    ctp::min_u64 v = s->writes.load() + 1;
    s->rec.value = v;
    for (size_t i = 0; i < kFields; ++i) {
      s->rec.fields[i] = v;
      if ((i & 3) == 0) {
        std::this_thread::yield();  // widen the tear window
      }
    }
    s->rec.EndUpdate();
    s->writes.fetch_add(1);

    // Periodically kill a reader and replace it.
    rng = rng * 1103515245u + 12345u;
    if ((rng >> 16) % 64 == 0 && kill_idx < kids.size()) {
      kill(kids[kill_idx], SIGKILL);
      int st = 0;
      waitpid(kids[kill_idx], &st, 0);
      pid_t p = fork();
      if (p == 0) {
        ReaderLoop(s);
        _exit(0);
      }
      kids[kill_idx] = p;
      ++kill_idx;
    }
  }

  // Kill the lease holders while they still hold the lease.
  for (pid_t p : lease_kids) {
    kill(p, SIGKILL);
    int st = 0;
    waitpid(p, &st, 0);
  }

  // The writer must still be able to take the lease: the abandoned one has to
  // be reclaimed rather than blocking forever. Short threshold to keep the
  // test quick.
  auto t0 = std::chrono::steady_clock::now();
  Lease<Record> recovered(&s->rec, 200);
  auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
  REQUIRE(recovered.Owns());
  REQUIRE(wait_ms < 3000);  // reclaimed, not wedged
  recovered.Release();

  s->stop.store(1);
  for (pid_t p : kids) {
    kill(p, SIGKILL);
    int st = 0;
    waitpid(p, &st, 0);
  }

  // The invariant that matters.
  INFO("writes=" << s->writes.load() << " read_ok=" << s->read_ok.load()
                 << " retries=" << s->read_retry.load()
                 << " steals=" << s->rec.lease_mutex_.GetStealCount());
  REQUIRE(s->torn_count.load() == 0);
  // Sanity bounds only -- "did this actually exercise anything", NOT
  // correctness. Keep them loose: the writer is heavily starved by the reader
  // fleet, so on a slow or contended runner it lands far lower than on a quiet
  // one. A macos-26-intel run recorded writes=10 against 8.2M reads and failed
  // a "> 10" bound by exactly one, which says nothing about the code.
  //
  // The starvation itself is worth noting: 57M retries for 8.2M successful
  // reads, with the WRITER getting only a handful of update windows. That is
  // the contention question tracked in #787, surfacing here as a side effect.
  REQUIRE(s->writes.load() >= 1);
  REQUIRE(s->read_ok.load() > 1000);
  REQUIRE(s->rec.lease_mutex_.GetStealCount() >= 1);

  UnmapShared(s);
}
#endif  // !_WIN32
