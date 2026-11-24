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

/**
 * Multi-node stress for the shm write-through cache. Every node runs this
 * binary bound to the cache pool (CLIO_CTE_POOL=568.0). The scenarios are
 * collective and self-verifying: every byte read is checked against the
 * pattern its writer must have produced.
 *
 * Knobs (env): STRESS_ROUNDS (rotating rounds, default 16), STRESS_BLOBS
 * (blobs per rotating round, default 48), STRESS_THREADS (reader threads,
 * default 4), STRESS_TORN_MS (torn-read run time in ms, default 5000),
 * STRESS_EVICT_BLOBS (1 MiB blobs written per node, default 48).
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

using clio::cte::core::TagId;
using clio::run::u64;

/** Wall-clock ms, for ordering events across nodes in a failure report. */
double WallMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

/** Integer env knob with a default. */
int Knob(const char *name, int def) {
  const char *e = std::getenv(name);
  const int v = (e != nullptr && *e) ? std::atoi(e) : def;
  return v > 0 ? v : def;
}

/** Client processes per node (STRESS_PPN, default 1). */
int Ppn() { return Knob("STRESS_PPN", 1); }

/** This process's index on its node (LOCAL_ID, 0-based). */
int LocalId() {
  const char *env = std::getenv("LOCAL_ID");
  return env != nullptr ? std::atoi(env) : 0;
}

/** This node's index (NODE_ID is 1-based). */
int NodeIdx() {
  const char *env = std::getenv("NODE_ID");
  return env != nullptr ? std::atoi(env) - 1 : 0;
}

/** Total client ranks: CTE_NUM_NODES (default 4) x processes per node. */
int NumNodes() { return Knob("CTE_NUM_NODES", 4) * Ppn(); }

/** This process's global rank. */
int Rank() { return NodeIdx() * Ppn() + LocalId(); }

/** The process client, bound to the cache pool. */
clio::cte::core::Client *Io() { return CLIO_CTE_CLIENT; }

/** Deterministic bytes for (key, version). */
std::string Pattern(u64 size, u64 key, u64 version) {
  std::string v(size, '\0');
  u64 x = key * 0x9E3779B97F4A7C15ull + version * 0xBF58476D1CE4E5B9ull + 1;
  for (u64 i = 0; i < size; ++i) {
    x ^= x >> 31;
    x *= 0x94D049BB133111EBull;
    v[i] = static_cast<char>(x >> 56);
  }
  return v;
}

/** Make this node's buffered writes for `tag` durable and visible. */
bool Sync(const TagId &tag) {
  auto s = Io()->AsyncSyncTag(tag);
  s.Wait();
  if (s->GetReturnCode() != 0) {
    fprintf(stderr, "[SYNC-ERR] rank=%d rc=%u (an earlier write failed)\n",
            Rank(), s->GetReturnCode());
  }
  return s->GetReturnCode() == 0;
}

/** Put bytes; score >= 0 forces the task path through the cache pool. */
bool Put(const TagId &tag, const std::string &name, u64 off, u64 size,
         const char *src, float score = -1.0f) {
  return Io()->PutBlob(tag, name, off, size, src, score) == 0;
}

/** Return code of the last failed Read on this thread (0 = mismatch). */
thread_local unsigned g_last_read_rc = 0;

/** Read [off, off+out->size()) of a blob; a failure's rc is kept in
 *  g_last_read_rc so a report can tell an error from wrong bytes. */
bool Read(const TagId &tag, const std::string &name, std::string *out,
          u64 off = 0) {
  g_last_read_rc = Io()->GetBlob(tag, name, off, out->size(), out->data());
  return g_last_read_rc == 0;
}

/** Size of a blob, or -1 when it does not exist. */
long long Size(const TagId &tag, const std::string &name) {
  u64 sz = 0;
  return Io()->GetBlobSize(tag, name, &sz) == 0 ? static_cast<long long>(sz)
                                                 : -1;
}

int g_epoch = 0;
std::string g_phase = "init";

/**
 * Start a scenario's barrier namespace. Every scenario calls this first, so
 * a scenario that failed part-way (skipping its later barriers) cannot
 * shift the barrier epochs of the scenarios after it.
 */
void BeginPhase(const char *name) {
  g_phase = name;
  g_epoch = 0;
}
/**
 * Cluster barrier over blobs: each rank publishes a marker, syncs it, then
 * waits until every rank's marker is visible.
 * @param tag tag holding the markers (one per epoch per rank)
 * @return false on a 300 s timeout
 */
bool Barrier(const TagId &tag) {
  const int epoch = g_epoch++;
  char one = 1;
  const std::string pre =
      "barrier_" + g_phase + "_" + std::to_string(epoch) + "_";
  if (!Put(tag, pre + std::to_string(Rank()), 0, 1, &one) || !Sync(tag)) {
    return false;
  }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
  for (int r = 0; r < NumNodes(); ++r) {
    while (Size(tag, pre + std::to_string(r)) != 1) {
      if (std::chrono::steady_clock::now() > deadline) {
        fprintf(stderr, "[BARRIER-TIMEOUT] rank=%d epoch=%d waiting_on=%d\n",
                Rank(), epoch, r);
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  return true;
}

/** Shared tag (every rank resolves the same id). */
TagId SharedTag(const std::string &name) {
  auto t = Io()->AsyncGetOrCreateTag(name);
  t.Wait();
  REQUIRE(t->GetReturnCode() == 0);
  fprintf(stderr, "[TAG] rank=%d name=%s id=%u.%u created=%u\n", Rank(),
          name.c_str(), t->tag_id_.major_, t->tag_id_.minor_,
          static_cast<unsigned>(t->created_));
  return t->tag_id_;
}

/** Barrier tag shared by all scenarios. */
TagId BarrierTag() {
  static TagId t = SharedTag("wtc_stress_barrier");
  return t;
}

/** Size of rotating blob `b` (4 KiB .. 260 KiB, varies per blob). */
u64 RotSize(int b) { return 4096ull * (1 + (b * 7) % 64); }

/**
 * Write blob `b` for `round` as shuffled pieces of random size. On odd
 * rounds every third piece takes the task path, so cache-path and task-path
 * writes of the same blob interleave.
 */
bool WriteRotating(const TagId &tag, int b, int round, std::mt19937_64 *rng) {
  const u64 size = RotSize(b);
  const std::string v = Pattern(size, b, round);
  std::vector<std::pair<u64, u64>> pieces;
  for (u64 off = 0; off < size;) {
    u64 len = std::min<u64>(size - off, 512 + (*rng)() % 65536);
    pieces.emplace_back(off, len);
    off += len;
  }
  std::shuffle(pieces.begin(), pieces.end(), *rng);
  for (size_t i = 0; i < pieces.size(); ++i) {
    const float score = (round % 2 == 1 && i % 3 == 0) ? 0.5f : -1.0f;
    if (!Put(tag, "rot_" + std::to_string(b), pieces[i].first,
             pieces[i].second, v.data() + pieces[i].first, score)) {
      fprintf(stderr, "[PUT-FAIL] rank=%d blob=%d round=%d\n", Rank(), b,
              round);
      return false;
    }
  }
  return true;
}

/**
 * Readers verify every rotating blob holds round `round`.
 * @return number of mismatching reads (0 = pass)
 */
int VerifyRotating(const TagId &tag, int blobs, int round, int threads) {
  std::atomic<int> bad{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; ++t) {
    ts.emplace_back([&, t] {
      for (int i = 0; i < blobs; ++i) {
        const int b = (i + t * 7) % blobs;
        const u64 size = RotSize(b);
        std::string got(size, '\0');
        // Twice: the second read is normally a cache hit.
        for (int rep = 0; rep < 2; ++rep) {
          if (!Read(tag, "rot_" + std::to_string(b), &got) ||
              got != Pattern(size, b, round)) {
            fprintf(stderr,
                    "[STALE] rank=%d thr=%d blob=%d round=%d rep=%d rc=%u\n",
                    Rank(), t, b, round, rep, g_last_read_rc);
            bad.fetch_add(1);
          }
        }
      }
    });
  }
  for (auto &th : ts) th.join();
  return bad.load();
}

/** Seconds since `t0`, in ms. */
double Ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

/** Print this node's cache counters. */
void PrintStats(const char *when) {
  const auto *s = Io()->CacheStats();
  if (s == nullptr) return;
  fprintf(stderr,
          "[STATS %s] rank=%d puts=%llu hits=%llu misses=%llu fills=%llu "
          "inval=%llu evict=%llu flush_batches=%llu flush_ranges=%llu "
          "waits=%llu\n",
          when, Rank(), (unsigned long long)s->puts_.load(),
          (unsigned long long)s->get_hits_.load(),
          (unsigned long long)s->get_misses_.load(),
          (unsigned long long)s->fills_.load(),
          (unsigned long long)s->invalidations_.load(),
          (unsigned long long)s->evictions_.load(),
          (unsigned long long)s->flush_batches_.load(),
          (unsigned long long)s->flush_ranges_.load(),
          (unsigned long long)s->waits_.load());
}

}  // namespace

TEST_CASE("ShmWtc stress - rotating writers, all nodes verify every round",
          "[shm_wtc][stress][rotate]") {
  BeginPhase("rotate");
  // STRESS_EXPECT_WTC=0 runs the suite against a pool below the cache (A/B).
  const char *expect_wtc = std::getenv("STRESS_EXPECT_WTC");
  if (expect_wtc == nullptr || std::string(expect_wtc) != "0") {
    REQUIRE(Io()->HasCache());
  }
  const TagId tag = SharedTag("wtc_stress_rot");
  const int n = NumNodes();
  const int blobs = Knob("STRESS_BLOBS", 48);
  const int rounds = Knob("STRESS_ROUNDS", 16);
  const int threads = Knob("STRESS_THREADS", 4);
  std::mt19937_64 rng(1234 + Rank());
  auto t0 = std::chrono::steady_clock::now();
  for (int round = 0; round < rounds; ++round) {
    for (int b = 0; b < blobs; ++b) {
      if ((b + round) % n == Rank()) {
        REQUIRE(WriteRotating(tag, b, round, &rng));
      }
    }
    REQUIRE(Sync(tag));
    REQUIRE(Barrier(BarrierTag()));
    const int stale = VerifyRotating(tag, blobs, round, threads);
    REQUIRE(Barrier(BarrierTag()));
    REQUIRE(stale == 0);
  }
  fprintf(stderr, "[ROTATE] rank=%d rounds=%d blobs=%d %.1f ms\n", Rank(),
          rounds, blobs, Ms(t0));
  PrintStats("rotate");
}

namespace {

/** One torn-read record: every 8-byte word holds the version. */
std::string Record(u64 size, u64 version) {
  std::string v(size, '\0');
  for (u64 i = 0; i + 8 <= size; i += 8) std::memcpy(&v[i], &version, 8);
  return v;
}

/**
 * Check a record is untorn; return its version (or ~0 when torn/empty).
 */
u64 RecordVersion(const std::string &v) {
  u64 first = 0;
  std::memcpy(&first, v.data(), 8);
  for (u64 i = 8; i + 8 <= v.size(); i += 8) {
    u64 w = 0;
    std::memcpy(&w, v.data() + i, 8);
    if (w != first) return ~0ull;
  }
  return first;
}

}  // namespace

TEST_CASE("ShmWtc stress - same-node readers never see a torn write",
          "[shm_wtc][stress][torn]") {
  BeginPhase("torn");
  // One blob per NODE: process 0 writes, every process on the node reads,
  // so readers in other processes copy out of the shared segment.
  const TagId tag = SharedTag("wtc_stress_torn_" + std::to_string(NodeIdx()));
  const bool writes_here = LocalId() == 0;
  const u64 size = 64 * 1024;
  const int ms = Knob("STRESS_TORN_MS", 5000);
  const int threads = Knob("STRESS_THREADS", 4);
  // Every Nth version takes the task path (0 = never).
  const char *te = std::getenv("STRESS_TORN_TASK_EVERY");
  const u64 task_every = te != nullptr ? std::strtoull(te, nullptr, 10) : 16;
  std::string v1 = Record(size, 1);
  if (writes_here) {
    REQUIRE(Put(tag, "rec", 0, size, v1.data()));
  }
  REQUIRE(Barrier(BarrierTag()));
  std::atomic<bool> stop{false};
  std::atomic<u64> torn{0}, backwards{0}, reads{0}, writes{0};
  std::thread writer([&] {
    for (u64 ver = 2; writes_here && !stop.load(); ++ver) {
      std::string v = Record(size, ver);
      const bool task = task_every != 0 && ver % task_every == 0;
      if (!Put(tag, "rec", 0, size, v.data(), task ? 0.5f : -1.0f)) {
        torn.fetch_add(1);
      }
      writes.fetch_add(1);
    }
  });
  std::vector<std::thread> rs;
  for (int t = 0; t < threads; ++t) {
    rs.emplace_back([&] {
      u64 last = 0;
      std::string got(size, '\0');
      while (!stop.load()) {
        if (!Read(tag, "rec", &got)) continue;
        const u64 ver = RecordVersion(got);
        reads.fetch_add(1);
        if (ver == ~0ull || ver == 0) {
          if (torn.fetch_add(1) < 3) {
            u64 a = 0, b = 0;
            std::memcpy(&a, got.data(), 8);
            std::memcpy(&b, got.data() + size - 8, 8);
            fprintf(stderr, "[TORN-SAMPLE] rank=%d first=%llu last=%llu\n",
                    Rank(), (unsigned long long)a, (unsigned long long)b);
          }
        } else if (ver < last) {
          backwards.fetch_add(1);
        } else {
          last = ver;
        }
      }
    });
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  stop.store(true);
  writer.join();
  for (auto &th : rs) th.join();
  fprintf(stderr, "[TORN] rank=%d writes=%llu reads=%llu torn=%llu back=%llu\n",
          Rank(), (unsigned long long)writes.load(),
          (unsigned long long)reads.load(), (unsigned long long)torn.load(),
          (unsigned long long)backwards.load());
  // Reach the scenario's last barrier BEFORE asserting, so a failing node
  // does not strand every other rank in it.
  const bool synced = Sync(tag);
  REQUIRE(Barrier(BarrierTag()));
  REQUIRE(synced);
  REQUIRE(reads.load() > 0);
  REQUIRE(torn.load() == 0);
  REQUIRE(backwards.load() == 0);
}

TEST_CASE("ShmWtc stress - working set larger than the cache",
          "[shm_wtc][stress][evict]") {
  BeginPhase("evict");
  const TagId tag = SharedTag("wtc_stress_evict");
  const int n = NumNodes();
  const int per = Knob("STRESS_EVICT_BLOBS", 48);
  const u64 size = 1024 * 1024;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < per; ++i) {
    const u64 key = Rank() * 100000ull + i;
    const std::string v = Pattern(size, key, 7);
    REQUIRE(Put(tag, "ev_" + std::to_string(key), 0, size, v.data()));
  }
  REQUIRE(Sync(tag));
  fprintf(stderr, "[T] rank=%d evict-synced t=%.1f\n", Rank(), WallMs());
  REQUIRE(Barrier(BarrierTag()));
  fprintf(stderr, "[T] rank=%d evict-barrier-out t=%.1f\n", Rank(), WallMs());
  std::mt19937_64 rng(99 + Rank());
  std::vector<u64> keys;
  for (int r = 0; r < n; ++r) {
    for (int i = 0; i < per; ++i) keys.push_back(r * 100000ull + i);
  }
  int bad = 0;
  for (int pass = 0; pass < 2; ++pass) {
    std::shuffle(keys.begin(), keys.end(), rng);
    for (u64 key : keys) {
      std::string got(size, '\0');
      if (!Read(tag, "ev_" + std::to_string(key), &got) ||
          got != Pattern(size, key, 7)) {
        const unsigned rc0 = g_last_read_rc;
        const double t_bad = WallMs();
        // Late or lost? Re-read once the cluster has had time to settle.
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::string again(size, '\0');
        const bool late = Read(tag, "ev_" + std::to_string(key), &again) &&
                          again == Pattern(size, key, 7);
        fprintf(stderr, "[EVICT-BAD] rank=%d key=%llu pass=%d rc=%u t=%.1f %s\n",
                Rank(), (unsigned long long)key, pass, rc0, t_bad,
                late ? "LATE(visible after 2 s)" : "LOST(still bad)");
        ++bad;
      }
    }
  }
  fprintf(stderr, "[EVICT] rank=%d blobs=%zu %.1f ms\n", Rank(), keys.size(),
          Ms(t0));
  PrintStats("evict");
  REQUIRE(Barrier(BarrierTag()));
  REQUIRE(bad == 0);
}

TEST_CASE("ShmWtc stress - delete and recreate a tag every node cached",
          "[shm_wtc][stress][deltag]") {
  BeginPhase("deltag");
  const int n = NumNodes();
  const u64 size = 16 * 1024;
  for (int iter = 0; iter < 4; ++iter) {
    const std::string tname = "wtc_stress_deltag";
    const TagId tag = SharedTag(tname);
    const int writer = iter % n;
    if (Rank() == writer) {
      for (int b = 0; b < 16; ++b) {
        const std::string v = Pattern(size, b, 100 + iter);
        REQUIRE(Put(tag, "d_" + std::to_string(b), 0, size, v.data()));
      }
      REQUIRE(Sync(tag));
    }
    REQUIRE(Barrier(BarrierTag()));
    for (int b = 0; b < 16; ++b) {
      std::string got(size, '\0');
      if (!Read(tag, "d_" + std::to_string(b), &got)) {
        fprintf(stderr, "[DELTAG-READ] rank=%d iter=%d blob=%d rc=%u\n",
                Rank(), iter, b, g_last_read_rc);
        FAIL("read of a synced blob failed");
      }
      REQUIRE(got == Pattern(size, b, 100 + iter));
    }
    REQUIRE(Barrier(BarrierTag()));
    if (Rank() == writer) {
      // By NAME: the cluster-wide delete (by id removes only the local
      // container's share of the tag, by contract).
      auto d = Io()->AsyncDelTag(tname);
      d.Wait();
      REQUIRE(d->GetReturnCode() == 0);
    }
    REQUIRE(Barrier(BarrierTag()));
    // The old id must be gone everywhere, cached copies included.
    for (int b = 0; b < 16; ++b) {
      if (Size(tag, "d_" + std::to_string(b)) >= 0) {
        fprintf(stderr, "[DELTAG-STALE] rank=%d iter=%d blob=%d\n", Rank(),
                iter, b);
        FAIL("blob survived DelTag");
      }
    }
    REQUIRE(Barrier(BarrierTag()));
  }
}

/** Cluster-wide tag size (Broadcast; cached by the shm cache). */
u64 TagSize(const TagId &tag) {
  auto f = Io()->AsyncGetTagSize(tag, clio::run::PoolQuery::Broadcast(0.0f));
  f.Wait();
  return f->GetReturnCode() == 0 ? static_cast<u64>(f->tag_size_) : ~0ull;
}

TEST_CASE("ShmWtc stress - tag size and opens stay exact across nodes",
          "[shm_wtc][stress][tagmeta]") {
  BeginPhase("tagmeta");
  const int n = NumNodes();
  // Without the cache (STRESS_EXPECT_WTC=0) the core reports per-container
  // shares, not the tag's total: check ids and the delete only.
  const char *ew = std::getenv("STRESS_EXPECT_WTC");
  const bool check_sizes = ew == nullptr || std::string(ew) != "0";
  const std::string tname = "wtc_stress_tagmeta";
  TagId tag = SharedTag(tname);
  std::vector<u64> sizes;  // every rank tracks each blob's size
  int last = -1;           // newest blob not yet truncated
  for (int iter = 0; iter < 8; ++iter) {
    const int writer = iter % n;
    // Grow by a new blob, or (every third round) truncate the newest one.
    const u64 add = 4096 * static_cast<u64>(iter + 1);
    const bool trunc = iter % 3 == 2 && last >= 0;
    if (Rank() == writer) {
      if (trunc) {
        auto t = Io()->AsyncTruncateBlob(tag, "m" + std::to_string(last), 100);
        t.Wait();
        REQUIRE(t->GetReturnCode() == 0);
      } else {
        const std::string v = Pattern(add, iter, 7);
        REQUIRE(Put(tag, "m" + std::to_string(iter), 0, add, v.data()));
      }
      REQUIRE(Sync(tag));
    }
    if (trunc) {
      sizes[last] = 100;
      last = -1;
    } else {
      sizes.resize(iter + 1, 0);
      sizes[iter] = add;
      last = iter;
    }
    u64 expect = 0;
    for (u64 v : sizes) expect += v;
    REQUIRE(Barrier(BarrierTag()));
    // Twice: the second answer comes from the segment when cached.
    for (int k = 0; k < 2; ++k) {
      const u64 got = check_sizes ? TagSize(tag) : expect;
      if (got != expect) {
        fprintf(stderr, "[TAGMETA-SIZE] rank=%d iter=%d k=%d got=%llu "
                "want=%llu\n", Rank(), iter, k, (unsigned long long)got,
                (unsigned long long)expect);
        FAIL("stale tag size");
      }
      TagId opened;
      u64 opened_size = 0;
      REQUIRE(Io()->GetOrCreateTag(tname, &opened, &opened_size) == 0);
      if (!(opened == tag) || (check_sizes && opened_size != expect)) {
        fprintf(stderr, "[TAGMETA-OPEN] rank=%d iter=%d id=%u.%u size=%llu "
                "want=%u.%u/%llu\n", Rank(), iter, opened.major_,
                opened.minor_, (unsigned long long)opened_size,
                tag.major_, tag.minor_, (unsigned long long)expect);
        FAIL("stale open");
      }
    }
    REQUIRE(Barrier(BarrierTag()));
  }
  // Delete and recreate: every node must see the new id and an empty tag.
  if (Rank() == 0) {
    auto d = Io()->AsyncDelTag(tname);
    d.Wait();
    REQUIRE(d->GetReturnCode() == 0);
  }
  REQUIRE(Barrier(BarrierTag()));
  TagId reopened;
  REQUIRE(Io()->GetOrCreateTag(tname, &reopened) == 0);
  if (reopened == tag) {
    fprintf(stderr, "[TAGMETA-DELETED] rank=%d still opens the old id\n",
            Rank());
    FAIL("deleted tag still opened by name");
  }
  REQUIRE(Barrier(BarrierTag()));
}

int main(int argc, char *argv[]) {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    fprintf(stderr, "CLIO_INIT failed\n");
    return 2;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "CTE client init failed\n");
    return 2;
  }
  fprintf(stderr, "shm_wtc stress: rank %d of %d ready (cache %s)\n", Rank(),
          NumNodes(), Io()->HasCache() ? "attached" : "NOT attached");
  std::string filter = argc > 1 ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);
  if (SimpleTest::g_test_finalize) SimpleTest::g_test_finalize();
  return result;
}
